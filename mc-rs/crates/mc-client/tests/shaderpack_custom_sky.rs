//! A resource pack's custom sky under the shader packs, on the real client over the SkyWars
//! benchmark capture (islands over the void, so the sky shows below the horizon).
//!
//! What a custom sky draws at and below the horizon must stay visible with a shader pack on, the
//! way it shows with the pack off (the pack's own colour work on top). The Dreamy pack's ground
//! fog used to march every downward sky ray into its endless layer, so the fog sea covered the
//! whole lower half of a custom sky, cut off exactly at the horizon.
//!
//! * `the_custom_sky_shows_below_the_horizon_with_each_pack` (runs when the vanilla jar and the
//!   capture are present): a generated sky whose lower half is a colour nothing else in the scene
//!   has, the vanilla sky parts off, looking down over the void. Every pack must keep it there.
//! * `a_faded_out_custom_sky_gives_dreamys_fog_sea_back` (same conditions): a layer drawn only
//!   at night. At night Dreamy keeps it in view; at noon nothing of it is drawn, and Dreamy must
//!   look exactly as with Custom Sky off (its fog sea back below the horizon).
//! * `custom_sky_pictures` (ignored; by hand): PNGs of each pack beside the pack off, with the
//!   vanilla sky parts on and off, at noon, sunset and night. `MC_SKY_PACK` names a folder pack to
//!   use instead of the generated sky.
//!
//! ```text
//! MC_NO_SKIN_DOWNLOADS=1 MC_JARS_DIR=<jars> cargo test -p mc-client --test shaderpack_custom_sky
//! MC_NO_SKIN_DOWNLOADS=1 MC_SKY_PNG_DIR=<dir> [MC_SKY_PACK=<folder pack>] MC_JARS_DIR=<jars> \
//!     cargo test -p mc-client --test shaderpack_custom_sky -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use mc_client::app::campath::CameraScript;
use mc_client::app::config::LoginMode;
use mc_client::{Client, ClientConfig};
use mc_math::{BlockPos, DVec3};
use mc_mods::mods::shaders_dreamy::{self as dreamy, ids};
use mc_physics::state::MoveInput;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList};

/// Ticks to run before posing: the join, the chunks and the players have settled by 280.
const SETTLE_TICKS: u32 = 285;
/// The SkyWars benchmark capture, relative to this crate.
const CAPTURE: &str =
    "../../optimization/scratch/skywars-scene-2026-11-05/captures/sw-mega-doubles-mid.mccap";
/// The generated pack's folder name.
const SKY_PACK: &str = "generated-sky";
/// The generated sky's colour above the horizon (display RGB).
const UPPER: [u8; 3] = [70, 110, 210];
/// The generated sky's colour below the horizon: an orange nothing else in the scene has.
const LOWER: [u8; 3] = [235, 120, 25];

const DREAMY: &str = dreamy::ID;
const CLASSIC: &str = "shaders_classic";
const UNDERWATER: &str = "shaders_underwater";
/// Every shader pack, by its mod id.
const PACKS: [&str; 3] = [DREAMY, CLASSIC, UNDERWATER];

/// Where the camera stands and looks.
#[derive(Debug, Clone, Copy)]
struct Pose {
    feet: DVec3,
    yaw: f32,
    pitch: f32,
}

struct Rig {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    width: u32,
    height: u32,
    _dir: tempfile::TempDir,
    _packs: tempfile::TempDir,
}

/// A 3x2 grid of `cell`-pixel faces (bottom, top, south / west, north, east): the walls'
/// upper halves and the top face [`UPPER`], the walls' lower halves and the bottom face
/// [`LOWER`].
fn sky_texture(cell: u32) -> Vec<u8> {
    let (w, h) = (cell * 3, cell * 2);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (col, row) = (x / cell, y / cell);
            let upper_half = y % cell < cell / 2;
            let colour = match (row, col) {
                (0, 0) => LOWER,
                (0, 1) => UPPER,
                _ if upper_half => UPPER,
                _ => LOWER,
            };
            rgba.extend_from_slice(&colour);
            rgba.push(255);
        }
    }
    rgba
}

/// The generated layer's properties: always drawn.
const ALWAYS: &str = "blend=replace\nrotate=false\n";
/// The generated layer's properties: drawn only at night (fully from 20:00 to 4:00, nothing
/// between 5:00 and 19:00).
const NIGHT_ONLY: &str =
    "blend=replace\nrotate=false\nstartFadeIn=19:00\nendFadeIn=20:00\nendFadeOut=5:00\n";

/// A folder pack holding one overworld layer drawn from [`sky_texture`], with `properties`.
fn write_sky_pack(dir: &Path, properties: &str) {
    let sky = dir.join("assets/minecraft/mcpatcher/sky/world0");
    std::fs::create_dir_all(&sky).expect("pack folders");
    std::fs::write(
        dir.join("pack.mcmeta"),
        r#"{"pack":{"pack_format":1,"description":"generated sky"}}"#,
    )
    .expect("pack.mcmeta");
    std::fs::write(sky.join("sky1.properties"), properties).expect("layer properties");
    let cell = 64;
    image::save_buffer(
        sky.join("sky1.png"),
        &sky_texture(cell),
        cell * 3,
        cell * 2,
        image::ColorType::Rgba8,
    )
    .expect("layer texture");
}

impl Rig {
    /// The client over the capture with a custom sky pack selected: `pack` is a folder pack, or
    /// `None` for the generated one with an always-drawn layer.
    fn new(width: u32, height: u32, pack: Option<&Path>) -> Option<Rig> {
        Rig::build(width, height, pack, ALWAYS)
    }

    /// The same with the generated pack, its layer's properties `layer`.
    fn generated(width: u32, height: u32, layer: &str) -> Option<Rig> {
        Rig::build(width, height, None, layer)
    }

    fn build(width: u32, height: u32, pack: Option<&Path>, layer: &str) -> Option<Rig> {
        let capture = Path::new(env!("CARGO_MANIFEST_DIR")).join(CAPTURE);
        if !capture.is_file() {
            eprintln!("SKIP: no SkyWars capture at {}", capture.display());
            return None;
        }
        let dir = tempfile::tempdir().ok()?;
        let packs = tempfile::tempdir().ok()?;
        let (pack_dir, pack_name) = match pack {
            Some(path) => (
                path.parent()?.to_path_buf(),
                path.file_name()?.to_string_lossy().into_owned(),
            ),
            None => {
                write_sky_pack(&packs.path().join(SKY_PACK), layer);
                (packs.path().to_path_buf(), SKY_PACK.to_owned())
            }
        };
        let mut config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("BenchBot".to_owned()),
            replay: Some(capture),
            replay_step_micros: 50_000,
            replay_camera: CameraScript::Fixed,
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        config.resource_pack_dir = pack_dir;
        config.resource_pack_fallback_dir = None;
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar (set MC_JARS_DIR)");
            return None;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        client.set_skin_downloads(false);
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
        client.settings.resource_packs.push(pack_name);
        client
            .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut |_, _| {})
            .expect("the sky pack loads");
        client.connect("replay.invalid", 25565);
        let mut camera = Camera::default();
        camera.set_viewport(width, height);
        Some(Rig {
            client,
            gpu,
            phases,
            camera,
            width,
            height,
            _dir: dir,
            _packs: packs,
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

    fn draw(&mut self) {
        self.client.frame(&self.gpu, &mut self.camera, 1.0);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, 1.0);
        frame.present(&self.gpu.queue);
    }

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

    fn settings(&mut self, edit: impl FnOnce(&mut mc_mods::ModSettings)) {
        edit(self.client.mod_settings_mut());
    }

    /// A new time of day takes effect in a tick.
    fn retime(&mut self, hours: f32, pose: Pose) {
        self.settings(|s| {
            s.set_enabled("time_changer", true);
            s.set_f32("time_changer", "time", hours);
        });
        for _ in 0..3 {
            self.pose(pose);
            self.tick();
        }
    }

    /// Settle after a settings change with frames only, then keep the last frame.
    fn shot(&mut self, pose: Pose) -> Vec<u8> {
        for _ in 0..60 {
            self.pose(pose);
            self.draw();
        }
        self.pose(pose);
        self.render()
    }

    fn viewer(&self) -> DVec3 {
        self.client
            .session()
            .and_then(|s| s.predictor())
            .map(|p| p.state.pos)
            .expect("the viewer exists")
    }

    fn solid(&self, pos: DVec3) -> bool {
        let session = self.client.session().expect("a session");
        let state = session.state().world.block_at(BlockPos::from_vec3(pos));
        state.props().blocks_movement()
    }

    /// Standing `height` blocks above the viewer, looking `pitch` degrees down the way with the
    /// most open void: of 32 directions, the one whose centre ray and the rays 15 degrees to
    /// either side run longest (up to 160 blocks) before meeting a block.
    fn void_pose(&self, me: DVec3, height: f64, pitch: f32) -> Pose {
        let feet = me + DVec3::new(0.0, height, 0.0);
        let eye = feet + DVec3::new(0.0, 1.62, 0.0);
        let open = |yaw: f64| {
            let pitch = f64::from(pitch).to_radians();
            let dir = DVec3::new(
                -yaw.sin() * pitch.cos(),
                -pitch.sin(),
                yaw.cos() * pitch.cos(),
            );
            (1..=320)
                .find(|&i| self.solid(eye + dir * (f64::from(i) * 0.5)))
                .unwrap_or(321)
        };
        let mut best = (0, 0.0_f64);
        for k in 0..32 {
            let yaw = f64::from(k) / 32.0 * std::f64::consts::TAU;
            let spread = 15.0_f64.to_radians();
            let reach = open(yaw).min(open(yaw - spread)).min(open(yaw + spread));
            if reach > best.0 {
                best = (reach, yaw);
            }
        }
        Pose {
            feet,
            yaw: best.1.to_degrees() as f32,
            pitch,
        }
    }
}

/// Exactly one shader pack (or none) on, each at its defaults. Dreamy's fog top is pinned with
/// World height mode at the default distance below the viewer's real feet: what Below you mode
/// settles to standing on the island, without its easing following the posed camera.
fn only_pack(s: &mut mc_mods::ModSettings, pack: Option<&str>, feet_y: f64) {
    for id in PACKS {
        s.reset_options(id);
        s.set_enabled(id, false);
    }
    s.set_enabled("custom_sky", true);
    if let Some(id) = pack {
        s.set_enabled(id, true);
    }
    if pack == Some(DREAMY) {
        let depth = f64::from(s.f32(DREAMY, ids::FOG_DEPTH));
        s.set_choice(DREAMY, ids::FOG_HEIGHT_MODE, dreamy::FOG_WORLD_HEIGHT);
        s.set_f32(DREAMY, ids::FOG_LEVEL, (feet_y - depth).round() as f32);
    }
}

/// The vanilla sky parts (the Sky mod: on is vanilla).
fn vanilla_sky(s: &mut mc_mods::ModSettings, on: bool) {
    s.set_enabled("sky", on);
}

fn rgb(pixels: &[u8], i: usize) -> [i32; 3] {
    [
        i32::from(pixels[i * 4]),
        i32::from(pixels[i * 4 + 1]),
        i32::from(pixels[i * 4 + 2]),
    ]
}

/// The pixels that show the generated sky's lower half in a frame with no pack: exactly
/// [`LOWER`] give or take a level.
fn lower_sky_pixels(off: &[u8]) -> Vec<usize> {
    (0..off.len() / 4)
        .filter(|&i| {
            let c = rgb(off, i);
            (0..3).all(|k| (c[k] - i32::from(LOWER[k])).abs() <= 2)
        })
        .collect()
}

/// Of `pixels`, the share that still reads as the lower sky's orange in `frame`: red well above
/// blue and green between them. Fog, haze and the vanilla sky are all bluish or grey there.
fn orange_share(frame: &[u8], pixels: &[usize]) -> f64 {
    let orange = pixels
        .iter()
        .filter(|&&i| {
            let [r, g, b] = rgb(frame, i);
            r > b + 80 && g > b && r > g
        })
        .count();
    orange as f64 / pixels.len().max(1) as f64
}

#[test]
fn the_custom_sky_shows_below_the_horizon_with_each_pack() {
    let Some(mut rig) = Rig::new(480, 270, None) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    let pose = rig.void_pose(me, 12.0, 30.0);
    rig.settings(|s| vanilla_sky(s, false));
    rig.settings(|s| only_pack(s, None, me.y));
    rig.retime(12.0, pose);
    let off = rig.shot(pose);
    let out = std::env::var_os("MC_SKY_PNG_DIR").map(PathBuf::from);
    if let Some(out) = &out {
        std::fs::create_dir_all(out).expect("output folder");
        save(out, "check_off", &[&off], rig.width, rig.height);
    }
    let lower = lower_sky_pixels(&off);
    // Looking 30 degrees down over the void, the sky's lower half is a good part of the frame.
    assert!(
        lower.len() > (rig.width * rig.height / 10) as usize,
        "the generated sky's lower half covers only {} pixels with no pack",
        lower.len()
    );
    for id in PACKS {
        rig.settings(|s| only_pack(s, Some(id), me.y));
        let on = rig.shot(pose);
        if let Some(out) = &out {
            save(out, &format!("check_{id}"), &[&on], rig.width, rig.height);
        }
        let share = orange_share(&on, &lower);
        eprintln!(
            "{id}: {:.1}% of the lower sky keeps its colour",
            share * 100.0
        );
        if id == UNDERWATER {
            // Everywhere mode puts the whole world under the sea: below the horizon is open
            // water by design, and the sky shows only through the surface overhead.
            continue;
        }
        assert!(
            share > 0.95,
            "{id}: only {:.1}% of the custom sky below the horizon still shows",
            share * 100.0
        );
    }
}

/// Of `pixels`, the share whose colour is the same (within 2 levels) in frames `a` and `b`.
fn same_share(a: &[u8], b: &[u8], pixels: &[usize]) -> f64 {
    let same = pixels
        .iter()
        .filter(|&&i| {
            let (p, q) = (rgb(a, i), rgb(b, i));
            (0..3).all(|k| (p[k] - q[k]).abs() <= 2)
        })
        .count();
    same as f64 / pixels.len().max(1) as f64
}

/// A sky pack whose only layer shows at night. At night Dreamy keeps its lower half in view; at
/// noon the layer is faded out and nothing of it is drawn, so the vanilla sky shows and Dreamy
/// must look exactly as with Custom Sky off: its fog sea back over the sky below the horizon
/// (the switch follows what the custom sky draws, not merely that the pack has layers).
#[test]
fn a_faded_out_custom_sky_gives_dreamys_fog_sea_back() {
    let Some(mut rig) = Rig::generated(480, 270, NIGHT_ONLY) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    let pose = rig.void_pose(me, 12.0, 30.0);
    rig.settings(|s| vanilla_sky(s, false));
    rig.settings(|s| only_pack(s, None, me.y));
    rig.retime(22.0, pose);
    let night_off = rig.shot(pose);
    let lower = lower_sky_pixels(&night_off);
    assert!(
        lower.len() > (rig.width * rig.height / 10) as usize,
        "the night layer's lower half covers only {} pixels with no pack",
        lower.len()
    );
    rig.settings(|s| only_pack(s, Some(DREAMY), me.y));
    let night = rig.shot(pose);
    let kept = orange_share(&night, &lower);
    eprintln!(
        "night: {:.1}% of the lower sky keeps its colour",
        kept * 100.0
    );
    assert!(
        kept > 0.95,
        "at night the layer stays in view: {:.1}%",
        kept * 100.0
    );

    rig.retime(12.0, pose);
    let noon_with = rig.shot(pose);
    rig.settings(|s| {
        s.set_enabled("custom_sky", false);
    });
    let noon_without = rig.shot(pose);
    rig.settings(|s| only_pack(s, None, me.y));
    let noon_plain = rig.shot(pose);
    let out = std::env::var_os("MC_SKY_PNG_DIR").map(PathBuf::from);
    if let Some(out) = &out {
        std::fs::create_dir_all(out).expect("output folder");
        let frames = [
            &night_off[..],
            &night[..],
            &noon_plain[..],
            &noon_with[..],
            &noon_without[..],
        ];
        save(
            out,
            "faded_night_off_dreamy_noon_off_with_without",
            &frames,
            rig.width,
            rig.height,
        );
    }
    // Dreamy changes the sky below the horizon at noon (its fog sea and its colours) ...
    let changed = 1.0 - same_share(&noon_plain, &noon_without, &lower);
    eprintln!(
        "noon: Dreamy changes {:.1}% of the lower sky",
        changed * 100.0
    );
    assert!(
        changed > 0.5,
        "Dreamy changes the lower sky at noon: {changed}"
    );
    // ... and the faded-out custom sky leaves it exactly as without the mod.
    let same = same_share(&noon_with, &noon_without, &lower);
    eprintln!(
        "noon: {:.1}% of the lower sky as with Custom Sky off",
        same * 100.0
    );
    assert!(
        same > 0.98,
        "with every layer faded out, Dreamy looks as with Custom Sky off: {:.1}%",
        same * 100.0
    );
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

fn save(out: &Path, name: &str, frames: &[&[u8]], width: u32, height: u32) {
    let path = out.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &side_by_side(frames, width, height),
        width * frames.len() as u32,
        height,
        image::ColorType::Rgba8,
    )
    .expect("write png");
    eprintln!("wrote {}", path.display());
}

#[test]
#[ignore = "writes PNGs of the real SkyWars scene; run by hand (see the module docs)"]
fn custom_sky_pictures() {
    let Some(out) = std::env::var_os("MC_SKY_PNG_DIR").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_SKY_PNG_DIR");
        return;
    };
    std::fs::create_dir_all(&out).expect("output folder");
    let pack = std::env::var_os("MC_SKY_PACK").map(PathBuf::from);
    let (width, height) = (960, 540);
    let Some(mut rig) = Rig::new(width, height, pack.as_deref()) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    eprintln!("viewer at {me:?}");
    let down = rig.void_pose(me, 12.0, 30.0);
    let horizon = Pose { pitch: 6.0, ..down };
    let west = Pose {
        feet: me + DVec3::new(0.0, 2.0, 0.0),
        yaw: 90.0,
        pitch: 2.0,
    };
    for (sky_label, sky_on) in [("sky_on", true), ("sky_off", false)] {
        for (time_label, hours) in [("noon", 12.0), ("sunset", 18.5), ("night", 22.0)] {
            for (view_label, pose) in [("horizon", horizon), ("down", down), ("west", west)] {
                rig.settings(|s| {
                    vanilla_sky(s, sky_on);
                    only_pack(s, None, me.y);
                });
                rig.retime(hours, pose);
                let off = rig.shot(pose);
                let mut frames = vec![off];
                for id in PACKS {
                    rig.settings(|s| only_pack(s, Some(id), me.y));
                    frames.push(rig.shot(pose));
                }
                let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
                save(
                    &out,
                    &format!("{sky_label}_{time_label}_{view_label}_off_dreamy_classic_underwater"),
                    &refs,
                    width,
                    height,
                );
            }
        }
    }
}
