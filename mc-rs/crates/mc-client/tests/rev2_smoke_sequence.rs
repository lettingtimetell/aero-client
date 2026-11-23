//! A smoke run of the real client, headless, through the sequences where the
//! memory-saving code releases state and makes it again, with every switch at its
//! default: the mod menu's glass chain (dropped on close), the scene snapshot (dropped after 240
//! frames that do not read it), the spectator outline targets (made on first use), the terrain
//! arena's first page (grows, and is dropped with the world), the replay viewer's records kept
//! in the file, and the memory collect when a world is left.
//!
//! wgpu's default error handler panics on a validation error, so a GPU resource released while
//! something still uses it fails this test, as it would crash the game.
//!
//! Ignored (GPU, vanilla jar, two recordings):
//! `REV2_REPLAY_A=<file> REV2_REPLAY_B=<file> MC_JARS_DIR=<jars> cargo test -p mc-client --test
//! rev2_smoke_sequence -- --ignored --nocapture`. `MC_GPU_ADAPTER=radeon` runs it on another GPU;
//! `REV2_GAME_DIR` puts the game folder somewhere chosen (a temporary folder otherwise).

mod common;

use std::path::PathBuf;

use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_render::input::{keys, Input, InputState, KeyBindings};
use mc_render::{Camera, Gpu, PhaseList};
use winit::keyboard::KeyCode;

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;

struct Rig {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    _dir: Option<tempfile::TempDir>,
}

impl Rig {
    fn new() -> Option<Rig> {
        let gpu = common::gpu(WIDTH, HEIGHT)?;
        let (dir, game_dir) = match std::env::var_os("REV2_GAME_DIR") {
            Some(path) => {
                let path = PathBuf::from(path);
                std::fs::create_dir_all(&path).expect("create the game folder");
                (None, path)
            }
            None => {
                let dir = tempfile::tempdir().expect("tempdir");
                let path = dir.path().to_path_buf();
                (Some(dir), path)
            }
        };
        let Some(config) = common::config(&game_dir, "Dev") else {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        };
        let mut client = Client::new(config, &mut |_, _| {}).expect("start");
        // Never reach the network: recordings carry signed skin profiles.
        client.set_skin_downloads(false);
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, WIDTH, HEIGHT);
        println!("adapter: {}", gpu.adapter_info().name);
        Some(Rig {
            client,
            gpu,
            phases,
            camera: Camera::default(),
            _dir: dir,
        })
    }

    fn pump(&mut self, frames: usize) {
        for _ in 0..frames {
            if self.client.replay_is_active() {
                common::pump_no_tick(
                    &mut self.client,
                    &mut self.gpu,
                    &mut self.phases,
                    &mut self.camera,
                );
            } else {
                common::pump(
                    &mut self.client,
                    &mut self.gpu,
                    &mut self.phases,
                    &mut self.camera,
                    &MoveInput::default(),
                );
            }
        }
    }

    /// A window resize as the shell applies it: the GPU's targets, the camera's viewport, every
    /// phase's screen-sized targets, then the app.
    fn resize_to(&mut self, width: u32, height: u32) {
        let rebuilt = self.gpu.resize(width, height);
        let (width, height) = (self.gpu.config.width, self.gpu.config.height);
        self.camera.set_viewport(width, height);
        self.camera.update();
        if rebuilt {
            self.phases.resize(&self.gpu, width, height);
            let Rig { client, gpu, .. } = self;
            mc_render::ShellApp::resized(client, gpu, width, height);
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.resize_to(width, height);
        self.pump(4);
    }

    /// Open the Right Shift menu (its glass reads the scene), then close it.
    fn menu_cycle(&mut self) {
        self.client.leak_harness_open_mod_menu();
        self.pump(12);
        self.client.leak_harness_close_mod_menu();
        // The closing animation, then the phase switches itself off.
        self.pump(20);
    }

    fn tap(&mut self, code: KeyCode) {
        self.client.handle_raw_key(code, true, false, None);
        self.pump(2);
        self.client.handle_raw_key(code, false, false, None);
        self.pump(2);
    }

    /// Spectator game mode on the replay's session and the outline binding held (G), as the
    /// game's input handling sees them; the next frame draws the outline.
    fn outline_held(&mut self, held: bool) {
        if let Some(session) = self.client.session_mut() {
            session.state_mut().local.game_mode = mc_protocol::play::s2c::GameMode::Spectator;
        }
        let mut bindings = KeyBindings::vanilla();
        let key = Input::Key(KeyCode::KeyG);
        assert!(bindings.set_binding(keys::SPECTATOR_OUTLINES, key));
        bindings.set_state(key, held);
        self.client
            .handle_game_input(&InputState::new(), &mut bindings);
    }

    fn report(&mut self, stage: &str) {
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
            "{stage:44} gpu {:7.1} MiB in {count:5} allocations; private {private:7.1} MB; \
             working set {working_set:7.1} MB; snapshot {:6.1} MiB; outline mask {:6.2} MiB; \
             glass {:5.1} MiB",
            bytes as f64 / 1_048_576.0,
            labelled_mib(&self.gpu, "mc-render scene snapshot"),
            labelled_mib(&self.gpu, "mc-render spectator outline mask"),
            labelled_mib(&self.gpu, "modmenu glass mips"),
        );
        if std::env::var_os("REV2_NAMES").is_some() {
            print_largest(&self.gpu);
        }
    }
}

/// `REV2_NAMES=1`: the largest GPU allocations by debug name (count, MiB).
fn print_largest(gpu: &Gpu) {
    let Some(report) = gpu.device.generate_allocator_report() else {
        return;
    };
    let mut names: std::collections::BTreeMap<&str, (usize, u64)> =
        std::collections::BTreeMap::new();
    for allocation in &report.allocations {
        let entry = names.entry(allocation.name.as_str()).or_default();
        entry.0 += 1;
        entry.1 += allocation.size;
    }
    let mut rows: Vec<_> = names.into_iter().collect();
    rows.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
    for (name, (count, bytes)) in rows.into_iter().take(10) {
        println!(
            "        {:7.2} MiB x{count:<3} {name}",
            bytes as f64 / 1_048_576.0
        );
    }
}

fn labelled_mib(gpu: &Gpu, label: &str) -> f64 {
    gpu.device
        .generate_allocator_report()
        .map_or(0.0, |report| {
            let bytes: u64 = report
                .allocations
                .iter()
                .filter(|allocation| allocation.name.contains(label))
                .map(|allocation| allocation.size)
                .sum();
            bytes as f64 / 1_048_576.0
        })
}

/// Open a recording and drive it the way a player does: jumps forward and back (a backward
/// jump rebuilds the world; later ones restore checkpoints), the menu over it, frames nobody
/// reads the scene in, the spectator outline, resizes, F1, chat, a shader pack, anti-aliasing,
/// then close.
fn watch(rig: &mut Rig, path: &std::path::Path, label: &str) {
    rig.client.replay_open(path).expect("open the replay");
    assert!(rig.client.replay_is_active());
    rig.pump(8);
    rig.report(&format!("{label}: opened"));
    let (start, end) = rig.client.replay_view_range();
    let span = end - start;
    for target in [
        start + span / 2,
        start + span / 4,
        end,
        start,
        start + span * 3 / 4,
        start + span / 3,
    ] {
        rig.client.replay_render_at(target);
        rig.pump(8);
    }
    rig.report(&format!("{label}: after six jumps"));
    for _ in 0..3 {
        rig.menu_cycle();
    }
    rig.report(&format!("{label}: menu x3 over the replay"));
    // Resizes while the outline is still dormant.
    rig.resize(640, 360);
    rig.resize(WIDTH, HEIGHT);
    // 260 frames that read no scene: the snapshot goes; then the menu makes it again.
    rig.pump(260);
    rig.report(&format!("{label}: 260 idle frames"));
    rig.menu_cycle();
    rig.report(&format!("{label}: menu after the idle frames"));
    // The spectator outline: made at the latest size on its first frame.
    rig.outline_held(true);
    rig.pump(1);
    rig.outline_held(true);
    rig.pump(3);
    rig.report(&format!("{label}: outline held"));
    let woke = labelled_mib(&rig.gpu, "mc-render spectator outline mask");
    println!("outline mask after the first outline frames: {woke:.2} MiB (woken if > 0.5)");
    for (width, height) in [(640, 360), (320, 240), (WIDTH, HEIGHT)] {
        rig.resize_to(width, height);
        rig.outline_held(true);
        rig.pump(3);
    }
    rig.client.leak_harness_open_mod_menu();
    for _ in 0..12 {
        rig.outline_held(true);
        rig.pump(1);
    }
    rig.client.leak_harness_close_mod_menu();
    rig.pump(20);
    rig.outline_held(false);
    rig.pump(4);
    rig.report(&format!("{label}: outline, resizes, menu"));
    // F1 twice, chat over the replay, a jump while the GUI is hidden.
    rig.tap(KeyCode::F1);
    rig.client.replay_render_at(start + span / 5);
    rig.pump(8);
    rig.tap(KeyCode::F1);
    rig.client.open_chat("");
    rig.pump(6);
    rig.tap(KeyCode::Escape);
    rig.pump(4);
    // A shader pack on and off (its passes read the scene), with the menu over it.
    rig.client
        .mod_settings_mut()
        .set_enabled("shaders_classic", true);
    rig.pump(8);
    rig.menu_cycle();
    rig.client
        .mod_settings_mut()
        .set_enabled("shaders_classic", false);
    rig.pump(8);
    rig.report(&format!("{label}: F1, chat, shader pack"));
    // Anti-aliasing (its passes read the scene and keep history over it): TAA, FXAA, off, frames
    // that read no scene (the snapshot goes), then TAA again (it is made again), the menu over it.
    rig.client
        .mod_settings_mut()
        .set_enabled("anti_aliasing", true);
    rig.pump(8);
    rig.client
        .mod_settings_mut()
        .set_choice("anti_aliasing", "mode", 1);
    rig.pump(8);
    rig.client
        .mod_settings_mut()
        .set_enabled("anti_aliasing", false);
    rig.pump(260);
    rig.client
        .mod_settings_mut()
        .set_choice("anti_aliasing", "mode", 0);
    rig.client
        .mod_settings_mut()
        .set_enabled("anti_aliasing", true);
    rig.pump(8);
    rig.menu_cycle();
    rig.client
        .mod_settings_mut()
        .set_enabled("anti_aliasing", false);
    rig.pump(8);
    rig.report(&format!("{label}: anti-aliasing TAA, FXAA, off, TAA"));
    rig.client.replay_render_at(end);
    rig.pump(8);
    rig.client.replay_close();
    assert!(!rig.client.replay_is_active());
    rig.pump(30);
    rig.report(&format!("{label}: closed, back on the title"));
}

/// The spectator outline drawn over a recording after a resize made while it was still dormant:
/// writes `<REV2_PNG_DIR>/outline-<REV2_PNG_TAG>.png` (and the frame without the outline). Run
/// once as is and once with `MC_W21_MR_LAZY_OUTLINE=0` (targets made at install, as before):
/// the two pictures must be byte-identical.
#[test]
#[ignore = "GPU + vanilla jar + REV2_REPLAY_A + REV2_PNG_DIR; writes pictures"]
fn outline_after_a_dormant_resize_png() {
    let (Some(path), Some(out)) = (
        std::env::var_os("REV2_REPLAY_A").map(PathBuf::from),
        std::env::var_os("REV2_PNG_DIR").map(PathBuf::from),
    ) else {
        eprintln!("skip: set REV2_REPLAY_A and REV2_PNG_DIR");
        return;
    };
    let tag = std::env::var("REV2_PNG_TAG").unwrap_or_else(|_| "run".to_owned());
    let Some(mut rig) = Rig::new() else {
        return;
    };
    rig.pump(10);
    rig.client.replay_open(&path).expect("open the replay");
    let (start, end) = rig.client.replay_view_range();
    // `REV2_AT`: where in the recording, 0..1 (the middle by default).
    let at: f64 = std::env::var("REV2_AT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let offset = ((end - start) as f64 * at.clamp(0.0, 1.0)) as mc_replay::ReplayTime;
    rig.client.replay_render_at(start + offset);
    for _ in 0..12 {
        rig.pump(10);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    // A resize while the outline has never been drawn: its targets must wake at this size.
    let (width, height) = (640, 360);
    rig.resize(width, height);
    rig.pump(10);
    let save = |rig: &Rig, name: &str| {
        let pixels = rig.gpu.read_pixels().expect("read back");
        std::fs::create_dir_all(&out).expect("png folder");
        std::fs::write(out.join(format!("{name}-{tag}.rgba")), &pixels).expect("write rgba");
        let image = image::RgbaImage::from_raw(width, height, pixels).expect("frame size");
        image
            .save(out.join(format!("{name}-{tag}.png")))
            .expect("write png");
    };
    // Spectator without the key first, so the two pictures differ only by the outline.
    for _ in 0..6 {
        rig.outline_held(false);
        rig.pump(1);
    }
    save(&rig, "no-outline");
    for _ in 0..6 {
        rig.outline_held(true);
        rig.pump(1);
    }
    save(&rig, "outline");
    let woke = labelled_mib(&rig.gpu, "mc-render spectator outline mask");
    println!("outline mask: {woke:.2} MiB (640x360 = 0.88 MiB)");
    rig.outline_held(false);
    rig.client.replay_close();
    rig.pump(4);
}

#[test]
#[ignore = "GPU + vanilla jar + REV2_REPLAY_A/B; a smoke run"]
fn title_menu_replays_and_back() {
    let (Some(first), Some(second)) = (
        std::env::var_os("REV2_REPLAY_A").map(PathBuf::from),
        std::env::var_os("REV2_REPLAY_B").map(PathBuf::from),
    ) else {
        eprintln!("skip: set REV2_REPLAY_A and REV2_REPLAY_B to .aeroreplay files");
        return;
    };
    let Some(mut rig) = Rig::new() else {
        return;
    };
    rig.pump(30);
    rig.report("title");
    for _ in 0..6 {
        rig.menu_cycle();
    }
    rig.report("title: menu x6");
    rig.resize(1280, 720);
    rig.menu_cycle();
    rig.resize(WIDTH, HEIGHT);
    rig.pump(260);
    rig.report("title: resizes, 260 frames");
    watch(&mut rig, &first, "replay A");
    for _ in 0..2 {
        rig.menu_cycle();
    }
    rig.report("title: menu x2 between replays");
    watch(&mut rig, &second, "replay B");
    watch(&mut rig, &first, "replay A again");
    rig.menu_cycle();
    rig.pump(30);
    rig.report("title: end");
}
