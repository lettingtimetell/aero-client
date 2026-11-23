//! M9 Waypoints: the client half.
//!
//! What the player gets, all of it off until the mod is switched on in the Social window:
//!
//! * **Create.** The create key (B) marks the block at the player's feet, with an automatic
//!   name and colour, and says so in a client-local chat line. In game only (`Ui::InGame`), and
//!   never while F3 is held, so F3+B still toggles hitboxes.
//! * **Death waypoint.** When the player dies (`player_is_dead()` turning true, the 20 Hz test
//!   vanilla's `GuiGameOver` hangs off), the spot is marked: one per server, replaced by the
//!   next death.
//! * **Labels.** Name and distance, through walls, at a constant size on screen
//!   (`mc_ui::waypoint_label`), in their own `NameplatePhase` at order 645: after the
//!   screen-space fog (640), so a far label is not fogged away, and before the hand (650).
//!   Hidden with F1, like every label.
//! * **Beams.** Vanilla's beacon beam in the waypoint's colour, from the block up to y 256, in
//!   their own `BeaconPhase` beside the vanilla beams (`PhaseOrder::ENTITIES`), depth-tested
//!   like world geometry. Hidden within 4 blocks horizontally and past the far plane.
//! * **Menu.** Dynamic rows under the expanded mod: the server, then one row per waypoint with
//!   its distance, an eye toggle and a delete button. Works with the mod off too.
//! * **Storage.** `<gameDir>/mc-rs-waypoints.json` (`mc_mods::mods::waypoints::store`), keyed by
//!   server. Loaded once, on first use, and written after every change by a background thread:
//!   on a slow disk one write measured 38 ms to 1.2 s, and B is pressed in the
//!   middle of fights. `--replay` and unknown servers are kept in memory only.
//!
//! Nothing here sends a packet.
//!
//! # Cost
//!
//! Off: every hook is one `u64` compare and a `bool` test (the frame hook also checks that the
//! menu event list is empty). The two phases are registered at install as empty wrappers that
//! report `enabled() == false`; their pipelines and textures are built the first time the mod
//! draws, so a session that never turns the mod on builds nothing.
//!
//! On: per visible waypoint, a distance, a matrix, 6 vertices per glyph plus 12, and 48 beam
//! vertices. A label's text is laid out again only when its rounded distance changes.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use mc_math::DVec3;
use mc_mods::mods::waypoints::store::{self, Waypoint, WaypointStore};
use mc_mods::{ModEvent, RowButton, RowIcon};
use mc_render::entity::tile::{self, BeaconPhase, BeaconVertex};
use mc_render::entity::NameplatePhase;
use mc_render::{Camera, Gpu, Input, InputState, PhaseList, PhaseOrder, RenderPhase};
use mc_ui::font::{FontRenderer, FontTexture};
use mc_ui::waypoint_label::{self, LabelScratch, WaypointLabel};

use super::super::{Client, Shared, Ui};

const MOD_ID: &str = "waypoints";

/// Where the labels draw: after the screen-space fog (`FogPhase::ORDER` = 640), which would
/// otherwise fade a far label into the sky, and before the hand (`PhaseOrder::FIRST_PERSON`).
pub(crate) const LABEL_ORDER: PhaseOrder = PhaseOrder(645);

/// Beams closer than this, horizontally (blocks), are hidden: standing in one fills the screen.
const BEAM_MIN_DISTANCE: f64 = 4.0;

/// The top of a beam: vanilla's beacon beam reaches the build limit.
const BEAM_TOP: i32 = 256;

/// The server key in `--replay` (kept in memory only).
const REPLAY_KEY: &str = "~replay";

/// The server key when the address is unknown (kept in memory only).
const LOCAL_KEY: &str = "~local";

/// Dynamic-row keys that are not waypoint ids (ids count up from 1).
const ROW_HEADER: u64 = u64::MAX;
const ROW_EMPTY: u64 = u64::MAX - 1;

/// The settings the hooks read, refreshed only when the settings generation moves.
#[derive(Debug, Clone, Copy)]
struct Snapshot {
    seen: u64,
    enabled: bool,
    key: Input,
    death: bool,
    beam: bool,
    distance: bool,
    /// `label_scale` as a factor (1.0 = 100%).
    scale: f32,
}

impl Default for Snapshot {
    fn default() -> Self {
        Snapshot {
            seen: 0,
            enabled: false,
            key: Input::Unbound,
            death: true,
            beam: true,
            distance: true,
            scale: 1.0,
        }
    }
}

/// Everything M9 keeps on the client (`Client::m9`).
#[derive(Default)]
pub(crate) struct WaypointsState {
    snap: Snapshot,
    /// Every server's waypoints. Loaded on first use.
    store: Option<WaypointStore>,
    /// `<gameDir>/mc-rs-waypoints.json`, set when the store is loaded.
    path: PathBuf,
    /// The file could not be read, or not moved aside: never write it this session.
    read_only: bool,
    /// The current server's key, and the `joins()` count it was derived at.
    server_key: String,
    key_joins: Option<u64>,
    /// The create key was down last frame.
    key_down: bool,
    /// The player was dead last tick; `armed` once that has been sampled since enabling.
    was_dead: bool,
    armed: bool,
    /// One layout cache per waypoint of the current server, in list order, and the
    /// `(store revision, joins)` they were sized for.
    labels: Vec<WaypointLabel>,
    labels_for: Option<(u64, u64)>,
    scratch: LabelScratch,
    /// This frame's visible waypoints as `(distance, index)`, far to near.
    order: Vec<(f64, usize)>,
    beam_inner: Vec<BeaconVertex>,
    beam_outer: Vec<BeaconVertex>,
    label_phase: Option<Shared<WaypointLabelPhase>>,
    beam_phase: Option<Shared<WaypointBeamPhase>>,
    /// Font sheets uploaded to (or failed for) the label phase, for `font_generation`.
    sheets: Vec<FontTexture>,
    font_generation: u64,
    /// The phases were left enabled by the last frame.
    active: bool,
    writer: WaypointWriter,
}

impl WaypointsState {
    /// Refresh the snapshot. One `u64` compare when nothing changed.
    fn sync(&mut self, settings: &mc_mods::ModSettings) {
        if self.snap.seen == settings.generation() {
            return;
        }
        let was_enabled = self.snap.enabled;
        self.snap = Snapshot {
            seen: settings.generation(),
            enabled: settings.enabled(MOD_ID),
            key: Input::from_lwjgl(settings.key(MOD_ID, "key_create")),
            death: settings.bool(MOD_ID, "death_waypoint"),
            beam: settings.bool(MOD_ID, "show_beam"),
            distance: settings.bool(MOD_ID, "show_distance"),
            scale: settings.f32(MOD_ID, "label_scale") / 100.0,
        };
        if self.snap.enabled && !was_enabled {
            // A key already held when the mod turns on is not a press, and a player who is
            // already dead has not just died.
            self.key_down = true;
            self.armed = false;
        }
    }
}

/// The label `NameplatePhase`, built the first time the mod draws.
#[derive(Default)]
struct WaypointLabelPhase {
    inner: Option<NameplatePhase>,
    active: bool,
}

impl RenderPhase for WaypointLabelPhase {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        if let Some(inner) = self.inner.as_mut() {
            inner.record(gpu, enc, view, cam);
        }
    }

    fn name(&self) -> &'static str {
        "waypoints/labels"
    }

    fn enabled(&self) -> bool {
        self.active && self.inner.is_some()
    }
}

/// The beam `BeaconPhase`, built (and its texture loaded) the first time the mod draws.
#[derive(Default)]
struct WaypointBeamPhase {
    inner: Option<BeaconPhase>,
    texture_tried: bool,
    active: bool,
}

impl RenderPhase for WaypointBeamPhase {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        if let Some(inner) = self.inner.as_mut() {
            inner.record(gpu, enc, view, cam);
        }
    }

    fn name(&self) -> &'static str {
        "waypoints/beams"
    }

    fn enabled(&self) -> bool {
        self.active && self.inner.is_some()
    }
}

/// Writes the waypoint file off the main thread. Jobs run in order and a newer snapshot of the
/// same file replaces one still waiting. Dropping the writer waits for the queue (at most 10 s,
/// so a wedged disk cannot hang exit).
#[derive(Default)]
struct WaypointWriter {
    tx: Option<mpsc::Sender<WriterJob>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

enum WriterJob {
    Write(PathBuf, String),
    Flush(mpsc::Sender<()>),
}

impl WaypointWriter {
    /// Queue `text` for `path`. The thread starts on the first save.
    fn submit(&mut self, path: PathBuf, text: String) {
        if self.tx.is_none() {
            let (tx, rx) = mpsc::channel();
            match std::thread::Builder::new()
                .name("mc-rs waypoint writer".into())
                .spawn(move || writer_loop(&rx))
            {
                Ok(handle) => {
                    self.tx = Some(tx);
                    self.thread = Some(handle);
                }
                Err(error) => {
                    tracing::warn!(%error, "waypoint writer thread unavailable; writing inline");
                    write_file(&path, &text);
                    return;
                }
            }
        }
        if let Some(tx) = &self.tx {
            if let Err(mpsc::SendError(job)) = tx.send(WriterJob::Write(path, text)) {
                // The thread is gone: write inline rather than lose the change.
                if let WriterJob::Write(path, text) = job {
                    write_file(&path, &text);
                }
                self.tx = None;
            }
        }
    }

    /// Wait until everything queued so far is on disk. `false` if that took over 10 s.
    fn flush(&self) -> bool {
        let Some(tx) = &self.tx else {
            return true;
        };
        let (ack, done) = mpsc::channel();
        if tx.send(WriterJob::Flush(ack)).is_err() {
            return false;
        }
        done.recv_timeout(Duration::from_secs(10)).is_ok()
    }
}

impl Drop for WaypointWriter {
    fn drop(&mut self) {
        let flushed = self.flush();
        // Closing the channel ends the thread once its queue is empty.
        self.tx = None;
        if let Some(thread) = self.thread.take() {
            if flushed {
                let _ = thread.join();
            } else {
                tracing::warn!("the waypoint writer did not finish within 10 s");
            }
        }
    }
}

fn writer_loop(rx: &mpsc::Receiver<WriterJob>) {
    while let Ok(job) = rx.recv() {
        match job {
            WriterJob::Write(mut path, mut text) => {
                // Only the newest snapshot of the file needs to reach the disk.
                let mut acks = Vec::new();
                while let Ok(next) = rx.try_recv() {
                    match next {
                        WriterJob::Write(next_path, next_text) => {
                            if next_path != path {
                                write_file(&path, &text);
                            }
                            path = next_path;
                            text = next_text;
                        }
                        WriterJob::Flush(ack) => acks.push(ack),
                    }
                }
                write_file(&path, &text);
                for ack in acks {
                    let _ = ack.send(());
                }
            }
            WriterJob::Flush(ack) => {
                let _ = ack.send(());
            }
        }
    }
}

fn write_file(path: &Path, text: &str) {
    match store::write_atomic(path, text) {
        Ok(()) => tracing::debug!(path = %path.display(), "waypoints saved"),
        Err(error) => tracing::warn!(%error, path = %path.display(), "waypoints were not saved"),
    }
}

/// The centre of the marked block, in world coordinates.
fn center(waypoint: &Waypoint) -> DVec3 {
    let [x, y, z] = waypoint.center();
    DVec3::new(x, y, z)
}

/// `0xRRGGBB` as linear `0..1` floats, for the beam tint.
fn rgb_f32(rgb: u32) -> [f32; 3] {
    [
        ((rgb >> 16) & 0xFF) as f32 / 255.0,
        ((rgb >> 8) & 0xFF) as f32 / 255.0,
        (rgb & 0xFF) as f32 / 255.0,
    ]
}

/// `text` without its `§x` formatting codes.
fn push_plain(out: &mut String, text: &str) {
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\u{a7}' {
            chars.next();
        } else {
            out.push(c);
        }
    }
}

/// One waypoint's beam, mirroring `tiles.rs::emit_beacon` for a single full-height segment.
fn push_beam(
    waypoint: &Waypoint,
    camera: &Camera,
    world_time: i64,
    partial_ticks: f32,
    inner: &mut Vec<BeaconVertex>,
    outer: &mut Vec<BeaconVertex>,
) {
    let dx = f64::from(waypoint.x) + 0.5 - camera.pos.x;
    let dz = f64::from(waypoint.z) + 0.5 - camera.pos.z;
    let horizontal = dx.hypot(dz);
    if !(BEAM_MIN_DISTANCE..=f64::from(camera.far)).contains(&horizontal) {
        return;
    }
    let base = waypoint.y.clamp(0, BEAM_TOP);
    let height = BEAM_TOP - base;
    if height <= 0 {
        return;
    }
    let beam = tile::beacon_beam(world_time, partial_ticks, height, 1.0);
    let xz = [
        (f64::from(waypoint.x) - camera.pos.x) as f32,
        (f64::from(waypoint.z) - camera.pos.z) as f32,
    ];
    let bottom = (f64::from(base) - camera.pos.y) as f32;
    let top = (f64::from(BEAM_TOP) - camera.pos.y) as f32;
    let rgb = rgb_f32(waypoint.color);
    tile::emit_beacon_inner_shell(xz, bottom, top, &beam, rgb, inner);
    tile::emit_beacon_outer_shell(xz, bottom, top, &beam, rgb, outer);
}

/// Decode `textures/entity/beacon_beam.png`, as `tiles.rs::ensure_beacon_texture` does.
fn beacon_texture(resources: &mc_assets::ResourceManager) -> Option<(Vec<u8>, u32, u32)> {
    let location = mc_assets::ResourceLocation::parse("textures/entity/beacon_beam.png").ok()?;
    let bytes = resources.load(&location).ok()?;
    // Refused (missing) when the device cannot hold it.
    let image = mc_assets::decode_texture_png(&bytes).ok()?.into_rgba8();
    let (width, height) = image.dimensions();
    Some((image.into_raw(), width, height))
}

/// Make one font sheet resident in the label phase. A sheet that will not load keeps the
/// phase's white fallback (boxes), as entity nameplates do. Shared with the markers (X13).
pub(super) fn upload_sheet(
    phase: &mut NameplatePhase,
    gpu: &Gpu,
    font: &FontRenderer,
    resources: &mc_assets::ResourceManager,
    sheet: FontTexture,
) {
    let result = match sheet {
        FontTexture::Ascii => {
            mc_ui::nameplate::upload_ascii_font(phase, gpu, font.data());
            Ok(())
        }
        FontTexture::StandardGalactic => {
            mc_ui::nameplate::upload_standard_galactic(phase, gpu, resources)
        }
        FontTexture::UnicodePage(page) => {
            mc_ui::nameplate::upload_unicode_page(phase, gpu, resources, page)
        }
    };
    if let Err(error) = result {
        tracing::debug!(%error, ?sheet, "a waypoint label font sheet is unavailable");
    }
}

impl Client {
    /// Register the two phases as empty wrappers. No GPU object is built
    /// here; that waits for the first frame the mod draws. Runs again after a resource reload,
    /// which rebuilt the phase list, so everything derived from the old phases is dropped.
    pub(crate) fn install_waypoint_phases(&mut self, phases: &mut PhaseList) {
        let labels = Shared::new(WaypointLabelPhase::default());
        let beams = Shared::new(WaypointBeamPhase::default());
        phases.insert(LABEL_ORDER, Box::new(labels.clone()));
        // `insert` is stable: after the vanilla beacon beams registered earlier in `install`.
        phases.insert(PhaseOrder::ENTITIES, Box::new(beams.clone()));
        self.m9.label_phase = Some(labels);
        self.m9.beam_phase = Some(beams);
        self.m9.active = false;
        self.m9.sheets.clear();
        self.m9.font_generation = 0;
    }

    /// The death edge.
    pub(crate) fn waypoints_tick(&mut self) {
        self.m9.sync(&self.mods.settings);
        if !self.m9.snap.enabled {
            return;
        }
        let dead = self.player_is_dead();
        if !self.m9.armed {
            self.m9.armed = true;
            self.m9.was_dead = dead;
            return;
        }
        let died = dead && !self.m9.was_dead;
        self.m9.was_dead = dead;
        if died && self.m9.snap.death {
            self.mark_death();
        }
    }

    /// The create key, as a press edge of the live input. Runs after
    /// `handle_screen_input`, so `self.ui` is current.
    pub(crate) fn waypoints_shell_frame(&mut self, input: &InputState) {
        self.m9.sync(&self.mods.settings);
        if !self.m9.snap.enabled {
            return;
        }
        let down = input.is_down(self.m9.snap.key);
        let pressed = down && !self.m9.key_down;
        self.m9.key_down = down;
        // B creates a waypoint: in game only. B with F3 held is vanilla's F3+B.
        if pressed && matches!(self.ui, Ui::InGame) && !self.debug_f3_held {
            self.create_waypoint_here();
        }
    }

    /// Menu clicks, then this frame's labels and beams.
    pub(crate) fn waypoints_frame(&mut self, gpu: &Gpu, camera: &Camera, partial_ticks: f32) {
        // The list can be managed from the menu with the mod off, so clicks come first.
        if !self.mods.events.is_empty() {
            self.handle_waypoint_events();
        }
        self.m9.sync(&self.mods.settings);
        if !self.m9.snap.enabled {
            if self.m9.active {
                self.deactivate_waypoint_phases();
            }
            return;
        }
        self.draw_waypoints(gpu, camera, partial_ticks);
    }

    /// The server, then one row per waypoint on it (all dimensions):
    /// `name: x y z`, `[N]`/`[E]` outside the overworld, the distance while in its dimension,
    /// an eye toggle (button 0) and delete (button 1). Greyed out while hidden.
    pub(crate) fn fill_waypoint_rows(&mut self, index: usize) {
        self.ensure_waypoint_context();
        let here = self.session.as_ref().and_then(|session| {
            let dimension = session.state().local.dimension;
            session
                .predictor()
                .map(|predictor| (predictor.state.eye_pos(), dimension))
        });
        let Client { m9, mods, .. } = self;
        let Some(rows) = mods.rows.get_mut(index) else {
            return;
        };
        rows.begin();
        let header = rows.push(ROW_HEADER);
        header.label.push_str("This server");
        match m9.server_key.as_str() {
            REPLAY_KEY => header.badge.push_str("replay, not saved"),
            LOCAL_KEY => header.badge.push_str("unknown, not saved"),
            key => header.badge.push_str(key),
        }
        if m9.read_only {
            header.badge.push_str(" (read-only)");
        }
        header.muted = true;
        let list = m9
            .store
            .as_ref()
            .map_or(&[][..], |store| store.list(&m9.server_key));
        if list.is_empty() {
            let row = rows.push(ROW_EMPTY);
            row.label.push_str("No waypoints yet");
            row.muted = true;
        }
        for waypoint in list {
            let row = rows.push(waypoint.id);
            push_plain(&mut row.label, &waypoint.name);
            let _ = write!(row.label, ": {} {} {}", waypoint.x, waypoint.y, waypoint.z);
            match waypoint.dim {
                -1 => row.label.push_str(" [N]"),
                1 => row.label.push_str(" [E]"),
                _ => {}
            }
            if let Some((eye, dimension)) = here {
                if dimension == waypoint.dim {
                    let distance = (center(waypoint) - eye).length();
                    waypoint_label::push_distance(
                        &mut row.badge,
                        waypoint_label::distance_key(distance),
                    );
                }
            }
            let eye_icon = if waypoint.visible {
                RowIcon::Eye
            } else {
                RowIcon::EyeOff
            };
            row.buttons = [
                Some(RowButton::Icon(eye_icon)),
                Some(RowButton::Icon(RowIcon::Delete)),
            ];
            row.muted = !waypoint.visible;
        }
    }

    /// Load the store on first use, and re-derive the server key after every join.
    fn ensure_waypoint_context(&mut self) {
        if self.m9.store.is_none() {
            let path = self.config.game_dir.join(store::FILE_NAME);
            let (loaded, report) = WaypointStore::load(&path);
            if let Some(error) = &report.error {
                tracing::warn!(path = %path.display(), %error, "waypoint file unusable");
            }
            if let Some(bad) = &report.moved_aside {
                tracing::warn!(kept = %bad.display(), "unreadable waypoint file kept aside");
            }
            if report.read_only {
                tracing::warn!("waypoints will not be saved this session");
            }
            if report.skipped > 0 {
                tracing::warn!(
                    skipped = report.skipped,
                    "unusable waypoint entries skipped"
                );
            }
            self.m9.read_only = report.read_only;
            self.m9.path = path;
            self.m9.store = Some(loaded);
        }
        let joins = self.machine.joins();
        if self.m9.key_joins != Some(joins) {
            self.m9.key_joins = Some(joins);
            self.m9.server_key = self.waypoint_server_key();
            self.m9.labels_for = None;
        }
    }

    /// The key for the current server: the address the player picked (saved server or direct
    /// connect), else `--server`/`--port`, else memory-only.
    fn waypoint_server_key(&self) -> String {
        if self.config.replay.is_some() {
            return REPLAY_KEY.to_owned();
        }
        if let Some(address) = self.current_server_address.as_deref() {
            if let Ok(parsed) = crate::net::ping::parse_server_address(address) {
                if let Ok(port) = parsed.socket_port() {
                    return store::server_key(&parsed.host, port);
                }
            }
        }
        if let Some((host, port)) = &self.config.server {
            return store::server_key(host, *port);
        }
        LOCAL_KEY.to_owned()
    }

    /// The block the player's feet are in, and the dimension.
    fn waypoint_feet(&self) -> Option<([i32; 3], i32)> {
        let session = self.session.as_ref()?;
        let pos = session.predictor()?.state.pos;
        if !pos.is_finite() {
            return None;
        }
        let block = [
            mc_math::floor_f64(pos.x),
            mc_math::floor_f64(pos.y),
            mc_math::floor_f64(pos.z),
        ];
        Some((block, session.state().local.dimension))
    }

    fn create_waypoint_here(&mut self) {
        let Some((pos, dim)) = self.waypoint_feet() else {
            return;
        };
        self.ensure_waypoint_context();
        let session_only = store::is_session_only(&self.m9.server_key);
        let Some(waypoints) = self.m9.store.as_mut() else {
            return;
        };
        let mut line = String::from("\u{a7}7[Waypoints] ");
        let created = match waypoints.create(&self.m9.server_key, pos, dim) {
            Some(waypoint) => {
                let _ = write!(
                    line,
                    "Created \u{a7}{}{}\u{a7}7 at {} {} {}",
                    store::chat_color_code(waypoint.color),
                    waypoint.name,
                    pos[0],
                    pos[1],
                    pos[2]
                );
                if session_only {
                    line.push_str(" \u{a7}8(this session only)");
                }
                true
            }
            None => {
                let _ = write!(
                    line,
                    "\u{a7}cThis server already has {} waypoints.",
                    store::MAX_PER_SERVER
                );
                false
            }
        };
        if created {
            self.save_waypoints();
        }
        self.waypoint_chat(line);
    }

    fn mark_death(&mut self) {
        let Some((pos, dim)) = self.waypoint_feet() else {
            return;
        };
        self.ensure_waypoint_context();
        if let Some(waypoints) = self.m9.store.as_mut() {
            waypoints.set_death(&self.m9.server_key, pos, dim);
            self.save_waypoints();
        }
    }

    /// A client-local chat line. Never sent to the server.
    fn waypoint_chat(&mut self, line: String) {
        let layout = mc_ui::screens::chat::ChatLayout::from_settings(&self.settings);
        let chat_open = matches!(self.ui, Ui::Chat(_));
        self.chat.add_message(
            mc_ui::font::component::ChatComponent::text(line),
            0,
            self.tick as i32,
            &layout,
            &self.font,
            &self.assets.language,
            chat_open,
        );
    }

    /// Queue the file for the writer thread. Memory-only servers and a read-only file are
    /// never written.
    fn save_waypoints(&mut self) {
        if self.m9.read_only || store::is_session_only(&self.m9.server_key) {
            return;
        }
        let Some(waypoints) = self.m9.store.as_ref() else {
            return;
        };
        let text = waypoints.to_json_string();
        self.m9.writer.submit(self.m9.path.clone(), text);
    }

    /// This frame's row clicks: button 0 toggles the eye, button 1 deletes.
    fn handle_waypoint_events(&mut self) {
        let mut changed = false;
        for event in &self.mods.events {
            let ModEvent::RowAction {
                mod_id,
                row,
                action,
            } = *event
            else {
                continue;
            };
            if mod_id != MOD_ID {
                continue;
            }
            let Some(waypoints) = self.m9.store.as_mut() else {
                continue;
            };
            changed |= match action {
                0 => waypoints.toggle_visible(row).is_some(),
                1 => waypoints.remove(row),
                _ => false,
            };
        }
        if changed {
            self.save_waypoints();
        }
    }

    fn deactivate_waypoint_phases(&mut self) {
        self.m9.active = false;
        if let Some(phase) = &self.m9.label_phase {
            phase.with(|phase| phase.active = false);
        }
        if let Some(phase) = &self.m9.beam_phase {
            phase.with(|phase| phase.active = false);
        }
    }

    fn draw_waypoints(&mut self, gpu: &Gpu, camera: &Camera, partial_ticks: f32) {
        let (Some(label_phase), Some(beam_phase)) =
            (self.m9.label_phase.clone(), self.m9.beam_phase.clone())
        else {
            return;
        };
        // The beam animates on the same resync-smoothed world clock as the
        // vanilla beacon beam (`Environment::render_world_time`), so it never turns backwards.
        let world = self.session.as_mut().map(|session| {
            let state = session.state_mut();
            let (ticks, partial) = state.env.render_world_time(partial_ticks);
            (state.local.dimension, ticks, partial)
        });
        let Some((dim, world_time, beam_partial)) = world else {
            if self.m9.active {
                self.deactivate_waypoint_phases();
            }
            return;
        };
        self.ensure_waypoint_context();
        let anaglyph = self.settings.anaglyph;
        let hide_gui = self.hide_gui;

        // The GPU halves, on first use.
        let built = label_phase.with(|phase| {
            phase.active = true;
            match phase.inner.as_mut() {
                Some(inner) => {
                    if inner.anaglyph_enabled() != anaglyph {
                        inner.set_anaglyph_with_gpu(gpu, anaglyph);
                    }
                    false
                }
                None => {
                    phase.inner = Some(NameplatePhase::new_with_anaglyph(gpu, anaglyph));
                    true
                }
            }
        });
        let resources = &self.assets.resources;
        beam_phase.with(|phase| {
            phase.active = true;
            let inner = phase.inner.get_or_insert_with(|| BeaconPhase::new(gpu));
            if !phase.texture_tried {
                phase.texture_tried = true;
                match beacon_texture(resources) {
                    Some((rgba, width, height)) => inner.upload_texture(gpu, &rgba, width, height),
                    None => tracing::warn!("waypoint beams: beacon_beam.png is unavailable"),
                }
            }
        });
        self.m9.active = true;

        let font_generation = self.font.layout_generation();
        let Client {
            m9, font, assets, ..
        } = self;
        if built || m9.font_generation != font_generation {
            m9.font_generation = font_generation;
            m9.sheets.clear();
        }
        let Some(waypoints) = m9.store.as_ref() else {
            return;
        };
        let list = waypoints.list(&m9.server_key);
        let sized_for = (waypoints.revision(), m9.key_joins.unwrap_or(0));
        if m9.labels_for != Some(sized_for) || m9.labels.len() != list.len() {
            m9.labels.clear();
            m9.labels.resize_with(list.len(), WaypointLabel::default);
            m9.labels_for = Some(sized_for);
        }

        // Visible waypoints in this dimension, far to near, so nearer labels draw on top.
        m9.order.clear();
        for (index, waypoint) in list.iter().enumerate() {
            if waypoint.visible && waypoint.dim == dim {
                m9.order
                    .push(((center(waypoint) - camera.pos).length(), index));
            }
        }
        m9.order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        m9.beam_inner.clear();
        m9.beam_outer.clear();
        if m9.snap.beam {
            for &(_, index) in &m9.order {
                push_beam(
                    &list[index],
                    camera,
                    world_time,
                    beam_partial,
                    &mut m9.beam_inner,
                    &mut m9.beam_outer,
                );
            }
        }
        beam_phase.with(|phase| {
            if let Some(inner) = phase.inner.as_mut() {
                inner.begin();
                inner.push_inner(&m9.beam_inner);
                inner.push_outer(&m9.beam_outer);
                inner.finish(gpu);
            }
        });

        label_phase.with(|phase| {
            let Some(inner) = phase.inner.as_mut() else {
                return;
            };
            inner.begin();
            // F1 hides labels; the beams are world geometry and stay.
            if !hide_gui {
                for &(distance, index) in &m9.order {
                    let waypoint = &list[index];
                    let rel = center(waypoint) - camera.pos;
                    let Some(matrix) =
                        waypoint_label::label_matrix(rel, camera.yaw, camera.pitch, m9.snap.scale)
                    else {
                        continue;
                    };
                    let label = &mut m9.labels[index];
                    label.update(
                        font,
                        &waypoint.name,
                        waypoint.color,
                        m9.snap.distance.then_some(distance),
                    );
                    for &sheet in label.sheets() {
                        if !m9.sheets.contains(&sheet) {
                            m9.sheets.push(sheet);
                            upload_sheet(inner, gpu, font, &assets.resources, sheet);
                        }
                    }
                    label.emit(matrix, &mut m9.scratch, inner);
                }
            }
            inner.finish(gpu);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::session::{JoinSettings, Session};
    use mc_physics::state::MoveInput;
    use winit::keyboard::KeyCode;

    type Channels = (
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    );

    /// A real client over the vanilla jar, with a temp game dir (never the user's).
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// Join a world with no socket behind it, standing at `feet`, in game.
    fn join(client: &mut Client, feet: DVec3) -> Channels {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(mc_protocol::play::s2c::JoinGame {
                    entity_id: 1,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                }),
            ))
            .expect("join channel");
        session.poll(16);
        let predictor = session
            .predictor_mut()
            .expect("JoinGame builds the predictor");
        // These tests were written for the old origin-facing-+Z JoinGame pose. JoinGame now
        // builds vanilla's first player (spawn + (0.5, 1, 0.5), yaw -180), so pin
        // the pose they assume before placing the feet.
        predictor.state.set_position(DVec3::ZERO);
        predictor.state.prev_pos = DVec3::ZERO;
        predictor.state.yaw = 0.0;
        predictor.state.pitch = 0.0;
        predictor.state.pos = feet;
        client.session = Some(session);
        client.ui = Ui::InGame;
        (to_game, to_net)
    }

    fn stand_at(client: &mut Client, feet: DVec3) {
        client
            .session
            .as_mut()
            .and_then(Session::predictor_mut)
            .expect("joined")
            .state
            .pos = feet;
    }

    fn set_health(client: &mut Client, health: f32) {
        client
            .session
            .as_mut()
            .expect("joined")
            .state_mut()
            .local
            .health = health;
    }

    fn list(client: &Client) -> Vec<Waypoint> {
        client
            .m9
            .store
            .as_ref()
            .map(|store| store.list(&client.m9.server_key).to_vec())
            .unwrap_or_default()
    }

    fn press(input: &mut InputState, code: KeyCode, down: bool) {
        let _ = input.on_key(code, down, false);
    }

    #[test]
    fn with_the_mod_off_nothing_is_loaded_created_or_written() {
        let Some((mut client, dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5));
        assert!(!client.mods.settings.enabled(MOD_ID), "off by default");
        let mut input = InputState::new();
        press(&mut input, KeyCode::KeyB, false);
        client.waypoints_shell_frame(&input);
        press(&mut input, KeyCode::KeyB, true);
        client.waypoints_shell_frame(&input);
        set_health(&mut client, 0.0);
        client.waypoints_tick();
        client.waypoints_tick();
        assert!(client.m9.store.is_none(), "the file is not even read");
        assert!(!dir.path().join(store::FILE_NAME).exists());
    }

    #[test]
    fn the_create_key_marks_the_feet_block_once_per_press_and_saves_per_server() {
        let Some((mut client, dir)) = offline_client() else {
            return;
        };
        client.current_server_address = Some("Play.Example.NET".to_owned());
        let _net = join(&mut client, DVec3::new(10.7, 64.0, -3.2));
        client.mods.settings.set_enabled(MOD_ID, true);
        let chat_before = client.chat.received_count();
        let mut input = InputState::new();

        // Held when the mod turns on: not a press.
        press(&mut input, KeyCode::KeyB, true);
        client.waypoints_shell_frame(&input);
        assert!(list(&client).is_empty());
        press(&mut input, KeyCode::KeyB, false);
        client.waypoints_shell_frame(&input);

        press(&mut input, KeyCode::KeyB, true);
        client.waypoints_shell_frame(&input);
        client.waypoints_shell_frame(&input); // still held: no repeat
        let waypoints = list(&client);
        assert_eq!(waypoints.len(), 1);
        assert_eq!(client.m9.server_key, "play.example.net:25565");
        assert_eq!(waypoints[0].name, "Waypoint 1");
        assert_eq!(
            (
                waypoints[0].x,
                waypoints[0].y,
                waypoints[0].z,
                waypoints[0].dim
            ),
            (10, 64, -4, 0),
            "the block at the feet, floored"
        );
        assert_eq!(
            client.chat.received_count(),
            chat_before + 1,
            "a local chat line"
        );

        // F3 held: F3+B is the hitbox toggle, not a waypoint.
        press(&mut input, KeyCode::KeyB, false);
        client.waypoints_shell_frame(&input);
        client.debug_f3_held = true;
        press(&mut input, KeyCode::KeyB, true);
        client.waypoints_shell_frame(&input);
        assert_eq!(list(&client).len(), 1);
        client.debug_f3_held = false;

        // A screen is open: no waypoint either.
        press(&mut input, KeyCode::KeyB, false);
        client.waypoints_shell_frame(&input);
        client.ui = Ui::Loading;
        press(&mut input, KeyCode::KeyB, true);
        client.waypoints_shell_frame(&input);
        assert_eq!(list(&client).len(), 1);
        client.ui = Ui::InGame;

        // A second press, somewhere else.
        press(&mut input, KeyCode::KeyB, false);
        client.waypoints_shell_frame(&input);
        stand_at(&mut client, DVec3::new(-0.5, 70.0, 100.0));
        press(&mut input, KeyCode::KeyB, true);
        client.waypoints_shell_frame(&input);
        let waypoints = list(&client);
        assert_eq!(waypoints.len(), 2);
        assert_eq!(waypoints[1].name, "Waypoint 2");
        assert_eq!(
            (waypoints[1].x, waypoints[1].y, waypoints[1].z),
            (-1, 70, 100)
        );

        // Written off the main thread, to the temp game dir only.
        assert!(client.m9.writer.flush());
        let (saved, report) = WaypointStore::load(&dir.path().join(store::FILE_NAME));
        assert_eq!(report, store::LoadReport::default());
        assert_eq!(saved.list("play.example.net:25565").len(), 2);
        assert_eq!(saved.servers().count(), 1);
    }

    #[test]
    fn a_death_marks_the_spot_and_replaces_the_previous_death() {
        let Some((mut client, dir)) = offline_client() else {
            return;
        };
        client.current_server_address = Some("pvp.example:25570".to_owned());
        let _net = join(&mut client, DVec3::new(1.5, 80.0, 2.5));
        // Already dead when the mod is turned on: not a new death.
        set_health(&mut client, 0.0);
        client.mods.settings.set_enabled(MOD_ID, true);
        client.tick(&MoveInput::default());
        client.tick(&MoveInput::default());
        assert!(list(&client).iter().all(|w| !w.death));

        // Respawn, die: marked once, however long the player stays dead.
        set_health(&mut client, 20.0);
        client.tick(&MoveInput::default());
        stand_at(&mut client, DVec3::new(5.2, 63.0, -7.9));
        set_health(&mut client, 0.0);
        client.tick(&MoveInput::default());
        client.tick(&MoveInput::default());
        let deaths: Vec<_> = list(&client).into_iter().filter(|w| w.death).collect();
        assert_eq!(deaths.len(), 1);
        assert_eq!((deaths[0].x, deaths[0].y, deaths[0].z), (5, 63, -8));
        assert_eq!(deaths[0].name, store::DEATH_NAME);
        let first = deaths[0].id;

        // Die again elsewhere: the old death waypoint is replaced, not kept.
        set_health(&mut client, 20.0);
        client.tick(&MoveInput::default());
        stand_at(&mut client, DVec3::new(-30.5, 12.0, 4.5));
        set_health(&mut client, 0.0);
        client.tick(&MoveInput::default());
        let deaths: Vec<_> = list(&client).into_iter().filter(|w| w.death).collect();
        assert_eq!(deaths.len(), 1, "one death waypoint per server");
        assert_ne!(deaths[0].id, first);
        assert_eq!((deaths[0].x, deaths[0].y, deaths[0].z), (-31, 12, 4));

        // The option off: deaths are not marked.
        client
            .mods
            .settings
            .set_bool(MOD_ID, "death_waypoint", false);
        set_health(&mut client, 20.0);
        client.tick(&MoveInput::default());
        stand_at(&mut client, DVec3::new(100.5, 64.0, 100.5));
        set_health(&mut client, 0.0);
        client.tick(&MoveInput::default());
        let deaths: Vec<_> = list(&client).into_iter().filter(|w| w.death).collect();
        assert_eq!((deaths[0].x, deaths[0].z), (-31, 4), "unchanged");

        assert!(client.m9.writer.flush());
        let (saved, _) = WaypointStore::load(&dir.path().join(store::FILE_NAME));
        assert_eq!(saved.list("pvp.example:25570").len(), 1);
    }

    #[test]
    fn replay_and_unknown_servers_are_never_written() {
        let Some((mut client, dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5));
        client.mods.settings.set_enabled(MOD_ID, true);
        let mut input = InputState::new();
        client.waypoints_shell_frame(&input);
        press(&mut input, KeyCode::KeyB, true);
        client.waypoints_shell_frame(&input);
        assert_eq!(client.m9.server_key, LOCAL_KEY);
        assert_eq!(list(&client).len(), 1, "kept for the session");
        assert!(client.m9.writer.flush());
        assert!(!dir.path().join(store::FILE_NAME).exists());
    }

    #[test]
    fn labels_beams_and_menu_rows_follow_the_mod_and_the_list() {
        let Some((mut client, dir)) = offline_client() else {
            return;
        };
        let Ok(gpu) = Gpu::headless(64, 64, mc_render::GpuOptions::default()) else {
            eprintln!("SKIP: no wgpu adapter");
            return;
        };
        client.current_server_address = Some("labels.example".to_owned());
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5));
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, 64, 64);
        let mut camera = super::super::super::menu_camera();
        let counts = |client: &Client| {
            let labels = client
                .m9
                .label_phase
                .as_ref()
                .expect("installed")
                .with(|p| {
                    p.inner
                        .as_ref()
                        .map(|inner| (inner.vertex_counts().0, inner.glyph_vertex_counts().0))
                });
            let beams = client
                .m9
                .beam_phase
                .as_ref()
                .expect("installed")
                .with(|p| p.inner.as_ref().map(BeaconPhase::vertex_counts));
            (labels, beams)
        };
        let enabled = |client: &Client| {
            (
                client
                    .m9
                    .label_phase
                    .as_ref()
                    .unwrap()
                    .with(|p| p.enabled()),
                client.m9.beam_phase.as_ref().unwrap().with(|p| p.enabled()),
            )
        };

        // Off: registered, but nothing built and nothing recorded.
        client.frame(&gpu, &mut camera, 0.0);
        assert_eq!(
            counts(&client),
            (None, None),
            "no pipelines until the mod draws"
        );
        assert_eq!(enabled(&client), (false, false));

        // On, with one waypoint 20 blocks north and one in the Nether.
        client.mods.settings.set_enabled(MOD_ID, true);
        client.ensure_waypoint_context();
        let key = client.m9.server_key.clone();
        assert_eq!(key, "labels.example:25565");
        let waypoints = client.m9.store.as_mut().unwrap();
        let north = waypoints.create(&key, [0, 64, -20], 0).unwrap().id;
        waypoints.create(&key, [8, 40, 8], -1);
        client.frame(&gpu, &mut camera, 0.0);
        assert_eq!(enabled(&client), (true, true));
        let (labels, beams) = counts(&client);
        let (solid, glyphs) = labels.expect("built on first use");
        assert_eq!(
            solid, 12,
            "one label: backdrop and marker; the Nether one is not drawn"
        );
        assert!(
            glyphs >= 6 * "Waypoint1(20m)".len(),
            "{glyphs} glyph vertices"
        );
        assert_eq!(beams, Some((24, 24)), "one beam: both shells");

        // F1 hides the label, not the beam.
        client.hide_gui = true;
        client.frame(&gpu, &mut camera, 0.0);
        let (labels, beams) = counts(&client);
        assert_eq!(labels, Some((0, 0)));
        assert_eq!(beams, Some((24, 24)));
        client.hide_gui = false;

        // The menu lists this server's waypoints (both dimensions) with their buttons.
        let index = client.mods.settings.index_of(MOD_ID).unwrap();
        client.mods.menu_expanded = Some(index);
        client.fill_menu_rows();
        let rows = client.mods.rows[index].rows().to_vec();
        assert_eq!(rows.len(), 3, "header and two waypoints: {rows:?}");
        assert_eq!(rows[0].badge, "labels.example:25565");
        assert_eq!(rows[1].key, north);
        assert_eq!(rows[1].label, "Waypoint 1: 0 64 -20");
        assert!(
            rows[1].badge.ends_with('m'),
            "distance badge: {:?}",
            rows[1].badge
        );
        assert_eq!(
            rows[1].buttons,
            [
                Some(RowButton::Icon(RowIcon::Eye)),
                Some(RowButton::Icon(RowIcon::Delete))
            ]
        );
        assert!(rows[2].label.ends_with("[N]") && rows[2].badge.is_empty());

        // The eye button hides it: nothing drawn, row greyed, file written.
        client.mods.events.push(ModEvent::RowAction {
            mod_id: MOD_ID,
            row: north,
            action: 0,
        });
        client.frame(&gpu, &mut camera, 0.0);
        assert!(
            client.mods.events.is_empty(),
            "the frame consumed this frame's events"
        );
        assert_eq!(counts(&client), (Some((0, 0)), Some((0, 0))));
        client.fill_menu_rows();
        let rows = client.mods.rows[index].rows();
        assert!(rows[1].muted);
        assert_eq!(rows[1].buttons[0], Some(RowButton::Icon(RowIcon::EyeOff)));
        assert!(client.m9.writer.flush());
        let (saved, _) = WaypointStore::load(&dir.path().join(store::FILE_NAME));
        assert!(!saved.list("labels.example:25565")[0].visible);

        // Delete works with the mod off, and turning it off disables both phases.
        client.mods.settings.set_enabled(MOD_ID, false);
        client.mods.events.push(ModEvent::RowAction {
            mod_id: MOD_ID,
            row: north,
            action: 1,
        });
        client.frame(&gpu, &mut camera, 0.0);
        assert_eq!(enabled(&client), (false, false));
        assert_eq!(list(&client).len(), 1);
        assert!(client.m9.writer.flush());
        let (saved, _) = WaypointStore::load(&dir.path().join(store::FILE_NAME));
        assert_eq!(saved.list("labels.example:25565").len(), 1);

        assert_eq!(phases.remove_named("waypoints/labels"), 1);
        assert_eq!(phases.remove_named("waypoints/beams"), 1);
    }

    /// End to end through the installed stack (sky, fog, HUD and every other phase): a real
    /// `Client` frame, rendered by its own `PhaseList` and read back. The waypoint's exact
    /// colour reaches the screen only with the mod on.
    #[test]
    fn a_waypoint_label_reaches_the_screen_through_the_whole_client_frame() {
        const SIZE: u32 = 256;
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let Ok(mut gpu) = Gpu::headless(
            SIZE,
            SIZE,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) else {
            eprintln!("SKIP: no wgpu adapter");
            return;
        };
        client.current_server_address = Some("render.example".to_owned());
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5));
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE, SIZE);
        // 200% labels and no beam, so the label alone is a few hundred pixels at 256x256.
        client.mods.settings.set_bool(MOD_ID, "show_beam", false);
        client.mods.settings.set_f32(MOD_ID, "label_scale", 200.0);
        client.ensure_waypoint_context();
        let key = client.m9.server_key.clone();
        // 30 blocks straight ahead (the player faces +Z), in the first automatic colour.
        let waypoint = client
            .m9
            .store
            .as_mut()
            .unwrap()
            .create(&key, [0, 64, 30], 0);
        assert_eq!(waypoint.map(|w| w.color), Some(0xFF_AA00));

        let mut render = |client: &mut Client| {
            let mut camera = super::super::super::menu_camera();
            camera.set_viewport(SIZE, SIZE);
            client.frame(&gpu, &mut camera, 0.0);
            camera.update();
            let frame = gpu.acquire().expect("acquire an offscreen frame");
            phases.render(&gpu, frame.view(), &camera, 0.0);
            frame.present(&gpu.queue);
            let pixels = gpu.read_pixels().expect("read back");
            pixels
                .chunks(4)
                .filter(|p| p[..3] == [0xFF, 0xAA, 0x00])
                .count()
        };

        let off = render(&mut client);
        assert_eq!(
            off, 0,
            "nothing in a vanilla frame is exactly the waypoint gold"
        );
        client.mods.settings.set_enabled(MOD_ID, true);
        let on = render(&mut client);
        assert!(
            on > 100,
            "the label and marker must reach the screen: {on} gold pixels"
        );
        client.mods.settings.set_enabled(MOD_ID, false);
        let off_again = render(&mut client);
        assert_eq!(off_again, 0, "switching the mod off removes it again");
    }
}
