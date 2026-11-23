//! The replay viewer's client half: the title screen's Replays
//! button, the browser, the controls over a replay and the export window.
//!
//! The screens themselves are [`mc_ui::screens::replay`] (plain state, headless-tested); this file
//! feeds them and acts on what they ask:
//!
//! * **The browser** is `Ui::ReplayBrowser`. It lists the library ([`mc_replay::Library`])
//!   on a background thread (the project lives on a USB disk: a hundred headers can take a
//!   moment), and renames, deletes and opens Explorer for the player. From the title screen it
//!   draws over the greenhouse (the title screen's scene keeps running behind it); from the
//!   `replay` mod's "Replay viewer: Open" button in game, over the dimmed game, and Watch then
//!   asks for a second click because it leaves the server.
//! * **Watching** goes through the engine (`replay_open`, `replay_play`, `replay_seek`...,
//!   in `tw_replay_play.rs`). As RB asked, the UI stays `Ui::InGame` during playback, so the HUD
//!   and every mod draw exactly as recorded; the controls ([`ReplayViewerOverlay`]) are an
//!   overlay drawn by this file's own glass phase, and its input hooks take the keys and the mouse
//!   first. The cursor is free while the controls are up, and grabbed only while the free camera
//!   flies (the right mouse button held).
//! * **Render settings** opens the mod menu over the replay; closing it returns to `Ui::InGame`,
//!   where the controls come back.
//! * **Export** hands the export job the window's request (`export_start`). While it runs,
//!   RD's preview and panel own the window and nothing here draws or takes input; its outcome
//!   (`export_take_finished`) comes back to the window. A clip's one-click Export in the browser is
//!   RD's `export_clip`; when it ends the browser returns on the clip with "Saved ... [Show]".
//!
//! # Hooks in `driver.rs` (one line each)
//! * `fields` / `init`: [`ReplayViewState`] in `Client::tw_view`.
//! * `install`: [`Client::replay_view_install_phases`] registers the phase slot at [`UI_ORDER`].
//! * `ui`: the `Ui::ReplayBrowser` variant and its arms.
//! * `cursor`: [`Client::replay_view_wants_cursor`] in `reconcile_cursor_grab`.
//! * `pointer`, `key`, `click`, `release`, `wheel`: the input hooks in `handle_screen_input`,
//!   `key_pressed`, `mouse_clicked`, `mouse_released` and `mouse_wheel`.
//! * `frame`: [`Client::replay_view_frame`] in `Client::frame`, before `drive_mods_frame` drops the
//!   menu events (the `replay` mod's "Replay viewer: Open" button).
//! * `draw`: [`Client::build_replay_ui`] at the end of `build_ui`.
//! * `focus`: no vanilla pause menu over a replay when the window loses focus.
//! * The title screen's Replays click and its greenhouse behind the browser:
//!   `mods/aero_title.rs`.
//!
//! # Cost
//! No replay screen up: every hook is an enum or `Option` test; the slot is empty, so the phase
//! list skips it and asks for no scene copy. The glass phase (pipelines, the scene's mip chain)
//! exists only while a replay screen is up and is dropped when the last one goes.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Instant;

use mc_mods::mods::replay as options;
use mc_mods::ModEvent;
use mc_render::{Camera, Gpu, PhaseList, PhaseOrder, RenderPhase, SceneTextures};
use mc_replay::{Library, LibraryEntry, ReplayKind};
use mc_ui::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use mc_ui::screens::modmenu::text::{self as menu_text, MenuText};
use mc_ui::screens::replay::export::{ExportProgress, ExportRequest, ExportSettings, ExportState};
use mc_ui::screens::replay::{
    key, BrowserEvent, CameraMode, LibraryView, PlaybackStatus, ReplayBrowserScreen, ReplayEntry,
    ReplayEnv, ReplayInfo, ReplayViewerOverlay, ViewerEvent,
};
// The browser's folders and orders.
use mc_ui::screens::replay::arrange::Arrangement;
use mc_ui::screens::replay::browser::Tab;
use mc_ui::widget::DrawList;

use super::super::{Client, Shared, Ui};
use super::tw_replay_export::{default_export_folder, safe_file_name, ExportPhase, ExportTarget};
use super::tw_replay_play::ReplayCameraMode;

/// Where the replay screens draw: after the UI phase (the HUD), the mod menu (`UI + 50`) and the
/// title screen greenhouse's VHS finish (`UI + 80`, so the browser's small text stays crisp), and
/// before the title screen's crisp marks (`UI + 90`).
pub(crate) const UI_ORDER: PhaseOrder = PhaseOrder(PhaseOrder::UI.0 + 85);

/// The `replay` mod's id.
const MOD_ID: &str = "replay";

/// A fixed place in the phase list holding the replay UI's glass phase only while a replay
/// screen is up. Empty, it is disabled and the phase list skips it.
pub(crate) struct ReplaySlot {
    phase: Option<ModMenuPhase>,
}

impl std::fmt::Debug for ReplaySlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplaySlot")
            .field("filled", &self.phase.is_some())
            .finish()
    }
}

impl RenderPhase for ReplaySlot {
    fn name(&self) -> &'static str {
        "replay_ui"
    }

    fn enabled(&self) -> bool {
        self.phase.as_ref().is_some_and(RenderPhase::enabled)
    }

    fn needs_scene_color(&self) -> bool {
        self.phase
            .as_ref()
            .is_some_and(RenderPhase::needs_scene_color)
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        if let Some(phase) = self.phase.as_mut() {
            phase.record(gpu, enc, view, cam);
        }
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        if let Some(phase) = self.phase.as_mut() {
            phase.record_scene(gpu, enc, view, cam, scene);
        }
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        if let Some(phase) = self.phase.as_mut() {
            phase.resize(gpu, width, height);
        }
    }
}

/// The library as listed by the background thread.
struct Listed {
    full: Vec<LibraryEntry>,
    clips: Vec<LibraryEntry>,
    /// Every folder under each tab's root, as the walk found them (real
    /// paths; links and junctions are never followed).
    full_folders: Vec<PathBuf>,
    clip_folders: Vec<PathBuf>,
}

impl Listed {
    fn empty() -> Listed {
        Listed {
            full: Vec::new(),
            clips: Vec::new(),
            full_folders: Vec::new(),
            clip_folders: Vec::new(),
        }
    }
}

/// The field on [`Client`] (`tw_view`).
pub(crate) struct ReplayViewState {
    slot: Option<Shared<ReplaySlot>>,
    /// The slot holds a phase (so the idle frame path is this `bool`, not a `RefCell` borrow).
    phase_live: bool,
    /// Segoe UI for the replay screens (CPU only; kept across screens).
    text: Option<MenuText>,
    /// The glyph atlas must go up again (a new phase).
    atlas_stale: bool,
    /// The glass shader's `uTime` origin.
    epoch: Instant,
    /// The cursor in physical pixels.
    cursor: (f32, f32),
    /// The controls while a replay is open.
    overlay: Option<ReplayViewerOverlay>,
    /// The file being watched and its browser id (the browser selects it again on exit).
    watching: Option<(PathBuf, u64)>,
    /// A library listing on its way from the background thread.
    listing: Option<Receiver<Listed>>,
    /// The browser's ids and their files.
    paths: Vec<(u64, PathBuf)>,
    /// The free camera is flying: the right mouse button is held, the cursor grabbed.
    flying: bool,
    /// When the current seek started (the "Seeking" chip shows only for slow seeks).
    seek_started: Option<Instant>,
    /// A one-click clip export the browser started (the `export_clip`): the browser comes
    /// back on this clip with the outcome when it ends.
    clip_export: Option<(PathBuf, u64)>,
    /// The listing's folders of each tab (recordings, clips) by key: the
    /// only folders the browser can ask to open in Explorer.
    folders: [Vec<(String, PathBuf)>; 2],
    /// The folder each tab of the browser showed when it last closed (this session only: never
    /// written to a file, folder names can be personal).
    browser_folders: [Option<String>; 2],
}

impl Default for ReplayViewState {
    fn default() -> Self {
        ReplayViewState {
            slot: None,
            phase_live: false,
            text: None,
            atlas_stale: true,
            epoch: Instant::now(),
            cursor: (-1.0, -1.0),
            overlay: None,
            watching: None,
            listing: None,
            paths: Vec::new(),
            flying: false,
            seek_started: None,
            clip_export: None,
            folders: [Vec::new(), Vec::new()],
            browser_folders: [None, None],
        }
    }
}

impl std::fmt::Debug for ReplayViewState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplayViewState")
            .field("overlay", &self.overlay.is_some())
            .field("watching", &self.watching)
            .field("flying", &self.flying)
            .finish_non_exhaustive()
    }
}

/// A stable id for a replay file (the same across library refreshes).
fn id_of(path: &Path) -> u64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    hasher.finish()
}

/// Milliseconds since the Unix epoch now.
fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// A library entry under the tab root `root` as the browser shows it.
fn view_entry(entry: &LibraryEntry, root: &Path) -> ReplayEntry {
    let (start, end) = entry.meta.window();
    let title = if entry.meta.title.trim().is_empty() {
        entry
            .path
            .file_stem()
            .map_or_else(|| "Replay".to_owned(), |s| s.to_string_lossy().into_owned())
    } else {
        // A default title written in another time zone (a friend's recording)
        // shows this computer's time; the file keeps its title.
        mc_replay::localize_title(&entry.meta.title, entry.meta.title_unix_ms())
            .unwrap_or_else(|| entry.meta.title.clone())
    };
    ReplayEntry {
        bookmarks: u32::try_from(entry.meta.bookmark_count()).unwrap_or(u32::MAX),
        id: id_of(&entry.path),
        title,
        server: entry.meta.server.clone(),
        started_unix_ms: entry.meta.started_unix_ms,
        duration_us: end.saturating_sub(start),
        size_bytes: entry.size_bytes,
        clip: entry.is_clip || matches!(entry.meta.kind, ReplayKind::Clip { .. }),
        has_mic: entry.meta.has_mic,
        problem: None,
        // The folder it is in (the Folders view) and who recorded it (the
        // Account order).
        folder: entry
            .path
            .parent()
            .and_then(|dir| mc_replay::folder_key(root, dir))
            .unwrap_or_default(),
        player: entry.meta.player_name.clone(),
    }
}

/// List the library of `game_dir` on a background thread, newest first.
fn spawn_listing(game_dir: PathBuf) -> Receiver<Listed> {
    let (tx, rx) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("mc-rs replay library".into())
        .spawn(move || {
            // A panic while listing (a damaged or shared file) ends the
            // listing (its sender is dropped), not the game, so it is not the game's crash.
            let _ = mc_core::panic_scope::contain(|| {
                let library = Library::new(&game_dir);
                // Old flat files into their date folders first (safe moves:
                // never a copy, a delete or an overwrite), so the list shows where they are now.
                library.organize();
                let newest_first = |mut entries: Vec<LibraryEntry>| {
                    entries.sort_by(|a, b| {
                        b.meta
                            .started_unix_ms
                            .cmp(&a.meta.started_unix_ms)
                            .then_with(|| b.modified.cmp(&a.modified))
                    });
                    entries
                };
                // With every folder under each root (the Folders view).
                let full = library.list_tree(false);
                let clips = library.list_tree(true);
                let _ = tx.send(Listed {
                    full: newest_first(full.entries),
                    clips: newest_first(clips.entries),
                    full_folders: full.folders,
                    clip_folders: clips.folders,
                });
            });
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "the replay library could not be listed");
    }
    rx
}

/// Show `path` selected in Explorer (or open its folder elsewhere).
fn show_in_folder(path: &Path) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt as _;
        // `/select,"<path>"` must reach Explorer unquoted as a whole: `raw_arg`.
        let spawned = std::process::Command::new("explorer.exe")
            .raw_arg(format!("/select,\"{}\"", path.display()))
            .spawn();
        if let Err(error) = spawned {
            tracing::warn!(%error, "Explorer could not be opened");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Some(parent) = path.parent() {
            let _ = super::super::open_external_target(parent);
        }
    }
}

/// Open a folder (created first, so a new library opens too).
fn open_folder(path: &Path) {
    let _ = std::fs::create_dir_all(path);
    if let Err(error) = super::super::open_external_target(path) {
        tracing::warn!(%error, "the folder could not be opened");
    }
}

fn camera_from(mode: ReplayCameraMode) -> CameraMode {
    match mode {
        ReplayCameraMode::Pov => CameraMode::Pov,
        ReplayCameraMode::ThirdPerson => CameraMode::Follow,
        ReplayCameraMode::Free => CameraMode::Free,
    }
}

fn camera_to(mode: CameraMode) -> ReplayCameraMode {
    match mode {
        CameraMode::Pov => ReplayCameraMode::Pov,
        CameraMode::Follow => ReplayCameraMode::ThirdPerson,
        CameraMode::Free => ReplayCameraMode::Free,
    }
}

/// A frame rate from one of the `replay` mod's choice lists ("60" → 60).
fn fps_of(choices: &[&str], index: usize, fallback: u32) -> u32 {
    choices
        .get(index)
        .and_then(|label| label.parse::<u32>().ok())
        .unwrap_or(fallback)
}

impl Client {
    /// The replay UI's phase slot (from `Client::install`, also after a
    /// resource reload rebuilt every phase; a replay screen that is up refills it next frame).
    pub(crate) fn replay_view_install_phases(&mut self, phases: &mut PhaseList) {
        let slot = Shared::new(ReplaySlot { phase: None });
        phases.insert(UI_ORDER, Box::new(slot.clone()));
        self.tw_view.slot = Some(slot);
        self.tw_view.phase_live = false;
        self.tw_view.atlas_stale = true;
    }

    /// Open the replay browser: from the title screen's Replays button, or the `replay` mod's
    /// "Replay viewer: Open" button (in game too: watching then leaves the server, after a
    /// second click).
    pub(crate) fn open_replay_browser(&mut self) {
        let now = Instant::now();
        let mut screen = ReplayBrowserScreen::new(now);
        if self.session.is_some() && !self.replay_is_active() {
            // The browser's "This leaves mc.example.net".
            let server = self.current_server_address.clone().unwrap_or_default();
            screen.set_in_game(server);
        }
        // The order and view the player left it in, and (this session) the
        // folders it showed.
        let settings = &self.mods.settings;
        screen.set_arrangement(Arrangement::from_tokens(
            settings.text(MOD_ID, options::ids::BROWSER_SORT),
            settings.text(MOD_ID, options::ids::BROWSER_VIEW),
        ));
        let [recordings, clips] = self.tw_view.browser_folders.clone();
        screen.set_start_folders(recordings, clips);
        self.tw_view.listing = Some(spawn_listing(self.config.game_dir.clone()));
        self.show_screen(Some(Ui::ReplayBrowser(Box::new(screen))));
    }

    /// Whether the browser is up over the title screen (no world): the title screen's
    /// greenhouse keeps drawing behind it (`mods/aero_title.rs`).
    pub(crate) fn replay_browser_over_title(&self) -> bool {
        matches!(self.ui, Ui::ReplayBrowser(_)) && self.session.is_none()
    }

    /// The cursor is free while the replay controls are up (`Ui::InGame`
    /// during playback), except while the free camera flies. An `Option` test otherwise.
    pub(crate) fn replay_view_wants_cursor(&self) -> bool {
        self.tw_view.overlay.is_some()
            && !self.tw_view.flying
            && matches!(self.ui, Ui::InGame)
            && self.replay_is_active()
    }

    /// Whether the replay controls take input this frame.
    fn replay_controls_up(&self) -> bool {
        self.tw_view.overlay.is_some() && matches!(self.ui, Ui::InGame) && self.replay_is_active()
    }

    /// This frame's cursor in physical pixels (an enum and an `Option`
    /// test unless a replay screen is up).
    pub(crate) fn replay_view_pointer(&mut self, cursor: (f64, f64)) {
        let browsing = matches!(self.ui, Ui::ReplayBrowser(_));
        if !browsing && self.tw_view.overlay.is_none() {
            return;
        }
        let (x, y) = (cursor.0 as f32, cursor.1 as f32);
        self.tw_view.cursor = (x, y);
        let now = Instant::now();
        if let Ui::ReplayBrowser(screen) = &mut self.ui {
            screen.pointer_moved(x, y);
        } else if self.replay_controls_up() && !self.tw_view.flying {
            if let Some(overlay) = self.tw_view.overlay.as_mut() {
                overlay.pointer_moved(x, y, now);
            }
        }
    }

    /// A key press. The browser takes every key; during playback the
    /// controls take every key except F1 and F3 (vanilla's hide-HUD and debug screen), so no
    /// in-game key (Esc's pause menu, E, T, F5) reaches the game. Returns whether it was used.
    pub(crate) fn replay_view_key(&mut self, press: &crate::app::input::KeyPress) -> bool {
        // An export takes every key itself (Esc cancels it).
        if self.export_running() {
            return false;
        }
        let now = Instant::now();
        if let Ui::ReplayBrowser(screen) = &mut self.ui {
            // Held keys repeat only to move through the list or to type a name: a held Delete
            // must never arm and confirm a delete by itself, nor a held Enter open twice.
            let repeatable = screen.renaming()
                || matches!(
                    press.lwjgl,
                    key::UP | key::DOWN | key::PAGE_UP | key::PAGE_DOWN
                );
            if press.repeat && !repeatable {
                return true;
            }
            if press.mods.ctrl && press.lwjgl == key::V && screen.renaming() {
                use mc_ui::widget::keys::Clipboard as _;
                let text = self.clipboard.get();
                screen.paste(&text);
                return true;
            }
            screen.key(press.lwjgl, press.typed, press.mods, now);
            return true;
        }
        if !self.replay_controls_up() {
            return false;
        }
        // F1 (hide the HUD) and F3 (debug) stay vanilla's.
        if matches!(press.lwjgl, 59 | 61) {
            return false;
        }
        let Some(overlay) = self.tw_view.overlay.as_mut() else {
            return false;
        };
        if press.mods.ctrl && press.lwjgl == key::V && overlay.typing() {
            use mc_ui::widget::keys::Clipboard as _;
            let text = self.clipboard.get();
            overlay.paste(&text);
            return true;
        }
        // Held keys repeat only where repeating makes sense (typing, jumping, stepping).
        let repeatable = overlay.typing()
            || matches!(
                press.lwjgl,
                key::LEFT | key::RIGHT | key::COMMA | key::PERIOD | key::BACKSPACE
            );
        if press.repeat && !repeatable {
            return true;
        }
        overlay.key(press.lwjgl, press.typed, press.mods, now);
        true
    }

    /// A mouse button edge. The browser takes every
    /// edge; during playback the controls do, and the right button flies the free camera.
    pub(crate) fn replay_view_mouse(&mut self, button: i32, pressed: bool) -> bool {
        if self.export_running() {
            self.tw_view.flying = false;
            return false;
        }
        let (x, y) = self.tw_view.cursor;
        let now = Instant::now();
        if let Ui::ReplayBrowser(screen) = &mut self.ui {
            let Ok(button) = u8::try_from(button) else {
                return true;
            };
            if pressed {
                screen.mouse_pressed(x, y, button);
            } else {
                screen.mouse_released(x, y, button, now);
            }
            return true;
        }
        if !self.replay_controls_up() && !self.tw_view.flying {
            return false;
        }
        // The free camera: hold the right button to look around and fly.
        if button == 1 {
            let free = self.replay_camera_mode() == ReplayCameraMode::Free;
            let exporting = self
                .tw_view
                .overlay
                .as_ref()
                .is_some_and(ReplayViewerOverlay::exporting);
            self.tw_view.flying = pressed && free && !exporting;
            return true;
        }
        let Ok(button) = u8::try_from(button) else {
            return true;
        };
        if let Some(overlay) = self.tw_view.overlay.as_mut() {
            if pressed {
                overlay.mouse_pressed(x, y, button, now);
            } else {
                overlay.mouse_released(x, y, button, now);
            }
        }
        true
    }

    /// The wheel over the browser's list, or the controls (the timeline jumps,
    /// the speed button changes the speed). During playback it never reaches the hotbar.
    pub(crate) fn replay_view_wheel(&mut self, delta: f32) -> bool {
        if self.export_running() {
            return false;
        }
        let (x, y) = self.tw_view.cursor;
        if let Ui::ReplayBrowser(screen) = &mut self.ui {
            screen.scrolled(x, y, delta);
            return true;
        }
        if !self.replay_controls_up() {
            return false;
        }
        if self.tw_view.flying {
            // The free camera's speed: the camera reads the wheel itself while it flies.
            return false;
        }
        if let Some(overlay) = self.tw_view.overlay.as_mut() {
            overlay.scrolled(x, y, delta, Instant::now());
        }
        true
    }

    /// From `Client::frame`, before `drive_mods_frame` drops this frame's menu
    /// events. A length test while none is queued.
    pub(crate) fn replay_view_frame(&mut self) {
        if !self.mods.events.is_empty() {
            self.replay_view_take_open_request();
        }
    }

    /// The `replay` mod's "Replay viewer: Open" button: open the browser (the mod menu closes
    /// first). The event is taken out of the queue, so it opens the browser once.
    fn replay_view_take_open_request(&mut self) {
        let before = self.mods.events.len();
        self.mods.events.retain(|event| {
            !matches!(
                event,
                ModEvent::Button {
                    mod_id: MOD_ID,
                    option_id: options::ids::OPEN_VIEWER,
                }
            )
        });
        if self.mods.events.len() == before {
            return;
        }
        if matches!(self.ui, Ui::ModMenu(_)) {
            self.close_mod_menu(Instant::now());
        }
        if !matches!(self.ui, Ui::ReplayBrowser(_)) {
            self.open_replay_browser();
        }
    }

    /// The export settings the export window starts with, from the `replay` mod.
    fn replay_export_defaults(&self) -> ExportSettings {
        let settings = &self.mods.settings;
        ExportSettings {
            selection: false,
            resolution: settings.choice(MOD_ID, options::ids::EXPORT_RESOLUTION),
            fps: fps_of(
                options::EXPORT_FPS_CHOICES,
                settings.choice(MOD_ID, options::ids::EXPORT_FPS),
                60,
            ),
            quality: settings.choice(MOD_ID, options::ids::EXPORT_QUALITY),
            max_size: 0,
            h265: settings.choice(MOD_ID, options::ids::EXPORT_CODEC) == 1,
            motion_blur: settings.choice(MOD_ID, options::ids::EXPORT_MOTION_BLUR),
            game_audio: settings.bool(MOD_ID, options::ids::EXPORT_AUDIO),
            mic: settings.bool(MOD_ID, options::ids::EXPORT_MIC),
            desktop: settings.bool(MOD_ID, options::ids::EXPORT_DESKTOP),
            separate_tracks: settings.bool(MOD_ID, options::ids::EXPORT_TRACKS),
            file_name: String::new(),
        }
    }

    /// The library listing, when it arrives: into the browser.
    fn replay_view_poll_listing(&mut self) {
        let Some(rx) = self.tw_view.listing.as_ref() else {
            return;
        };
        let listed = match rx.try_recv() {
            Ok(listed) => listed,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Listed::empty(),
        };
        self.tw_view.listing = None;
        self.tw_view.paths = listed
            .full
            .iter()
            .chain(listed.clips.iter())
            .map(|entry| (id_of(&entry.path), entry.path.clone()))
            .collect();
        // Each folder's key under its tab's root; the browser gets the keys,
        // this side keeps the paths the walk found (the only ones it will open).
        let library = Library::new(&self.config.game_dir);
        let roots = [
            library.tab_root(false).to_path_buf(),
            library.tab_root(true).to_path_buf(),
        ];
        let keyed = |root: &Path, dirs: &[PathBuf]| -> Vec<(String, PathBuf)> {
            dirs.iter()
                .filter_map(|dir| mc_replay::folder_key(root, dir).map(|key| (key, dir.clone())))
                .filter(|(key, _)| !key.is_empty())
                .collect()
        };
        self.tw_view.folders = [
            keyed(&roots[0], &listed.full_folders),
            keyed(&roots[1], &listed.clip_folders),
        ];
        let keys = |tab: usize| -> Vec<String> {
            self.tw_view.folders[tab]
                .iter()
                .map(|(key, _)| key.clone())
                .collect()
        };
        let (recording_folders, clip_folders) = (keys(0), keys(1));
        let settings = &self.mods.settings;
        let view = LibraryView {
            recordings: listed
                .full
                .iter()
                .map(|e| view_entry(e, &roots[0]))
                .collect(),
            clips: listed
                .clips
                .iter()
                .map(|e| view_entry(e, &roots[1]))
                .collect(),
            cap_bytes: (f64::from(settings.f32(MOD_ID, options::ids::MAX_STORAGE_GB))
                * 1_000_000_000.0) as u64,
            recording_on: settings.enabled(MOD_ID)
                && settings.bool(MOD_ID, options::ids::AUTO_START),
            clip_key_set: settings.key(MOD_ID, options::ids::KEY_CLIP) != 0,
            recording_folders,
            clip_folders,
        };
        let select = self.tw_view.watching.take().map(|(_, id)| id);
        if let Ui::ReplayBrowser(screen) = &mut self.ui {
            screen.set_library(view, now_unix_ms());
            if let Some(id) = select {
                screen.select(id);
            }
        }
    }

    fn replay_path_of(&self, id: u64) -> Option<PathBuf> {
        self.tw_view
            .paths
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, path)| path.clone())
    }

    /// Watch `path` with the viewer's controls, as the browser's Watch does
    /// (the `MC_RD_EXPORT` harness uses it for pictures of the viewer and its Options).
    pub(crate) fn replay_view_watch_file(&mut self, path: &Path) {
        let id = id_of(path);
        if self.replay_path_of(id).is_none() {
            self.tw_view.paths.push((id, path.to_path_buf()));
        }
        self.replay_watch(id);
    }

    /// Show a message in the open browser.
    fn replay_browser_message(&mut self, text: String, error: bool) {
        if let Ui::ReplayBrowser(screen) = &mut self.ui {
            screen.show_message(text, error, Instant::now());
        }
    }

    /// Open one of the library's folders in Explorer (the browser's Open in
    /// Explorer, and Open folder in the Folders view): a tab's root, or only a folder the
    /// listing found under that key (never a path made from the key), checked again just before
    /// (still a real folder inside the tab's root, no link or junction below the root).
    fn replay_open_library_folder(&mut self, clips: bool, key: &str) {
        let library = Library::new(&self.config.game_dir);
        let root = library.tab_root(clips).to_path_buf();
        let found = if key.is_empty() {
            // A new library has no folder yet: made, as the Open folder button always did.
            let _ = std::fs::create_dir_all(&root);
            Some(root.clone())
        } else {
            self.tw_view.folders[usize::from(clips)]
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, path)| path.clone())
        };
        let Some(path) = found else {
            self.replay_browser_message("That folder is not there any more".to_owned(), true);
            return;
        };
        if !mc_replay::is_safe_folder(&root, &path) {
            tracing::warn!("a replay folder was not opened: not a plain folder inside the library");
            self.replay_browser_message("That folder can't be opened from here".to_owned(), true);
            return;
        }
        if let Err(error) = super::super::open_external_target(&path) {
            tracing::warn!(%error, "the folder could not be opened");
        }
    }

    /// The browser's requests.
    fn replay_browser_events(&mut self) {
        let events: Vec<BrowserEvent> = match &mut self.ui {
            Ui::ReplayBrowser(screen) => {
                let events: Vec<BrowserEvent> = screen.drain_events().collect();
                // Where each tab is, for the next time the browser opens
                // this session (any event may close or replace it).
                if !events.is_empty() {
                    self.tw_view.browser_folders = [
                        Some(screen.folder_key(Tab::Recordings).to_owned()),
                        Some(screen.folder_key(Tab::Clips).to_owned()),
                    ];
                }
                events
            }
            _ => return,
        };
        for event in events {
            match event {
                BrowserEvent::Watch(id) => self.replay_watch(id),
                BrowserEvent::ExportClip(id) => self.replay_export_clip(id),
                BrowserEvent::ShowFile(path) => show_in_folder(Path::new(&path)),
                BrowserEvent::Rename { id, title } => {
                    let Some(path) = self.replay_path_of(id) else {
                        continue;
                    };
                    let library = Library::new(&self.config.game_dir);
                    if let Err(error) = library.rename(&path, &title) {
                        self.replay_browser_message(format!("Couldn't rename it: {error}"), true);
                        self.tw_view.listing = Some(spawn_listing(self.config.game_dir.clone()));
                    } else {
                        self.replay_browser_message("Renamed".to_owned(), false);
                    }
                }
                BrowserEvent::Delete(id) => {
                    let Some(path) = self.replay_path_of(id) else {
                        continue;
                    };
                    let library = Library::new(&self.config.game_dir);
                    if let Err(error) = library.delete(&path) {
                        self.replay_browser_message(format!("Couldn't delete it: {error}"), true);
                        self.tw_view.listing = Some(spawn_listing(self.config.game_dir.clone()));
                    }
                }
                BrowserEvent::ShowInFolder(id) => {
                    if let Some(path) = self.replay_path_of(id) {
                        show_in_folder(&path);
                    }
                }
                BrowserEvent::OpenFolder => {
                    let library = Library::new(&self.config.game_dir);
                    open_folder(library.replays_dir());
                }
                BrowserEvent::Close => self.show_screen(None),
                // The order and view are remembered in the `replay` mod's
                // settings (saved with the other mod settings).
                BrowserEvent::Arrange(arrangement) => {
                    let settings = &mut self.mods.settings;
                    settings.set_text(MOD_ID, options::ids::BROWSER_SORT, arrangement.sort.token());
                    settings.set_text(MOD_ID, options::ids::BROWSER_VIEW, arrangement.view.token());
                }
                BrowserEvent::OpenFolderAt { clips, folder } => {
                    self.replay_open_library_folder(clips, &folder);
                }
            }
        }
    }

    /// Leave whatever is running before a replay opens: another replay (the browser was opened
    /// over it from the mod menu), or a server (in game the browser asked for a second click).
    fn replay_leave_for_replay(&mut self) {
        if self.replay_is_active() {
            self.replay_export_cancel();
            self.tw_view.overlay = None;
            self.replay_close();
        }
        if self.session.is_some() && !self.replay_is_active() {
            self.leave_server(false);
            if self.session.is_some() {
                self.unload_world();
            }
        }
    }

    /// The browser's one-click Export of a clip: The clip export (the clip settings, H.264,
    /// into `Videos\Aero Client\Clips`). It opens the clip, renders it while its own preview and
    /// panel show, and closes it again; then the browser comes back on the clip with the outcome
    /// ([`Client::replay_clip_export_poll`]).
    fn replay_export_clip(&mut self, id: u64) {
        let Some(path) = self.replay_path_of(id) else {
            return;
        };
        self.replay_leave_for_replay();
        match self.export_clip(&path) {
            Ok(()) => self.tw_view.clip_export = Some((path, id)),
            Err(error) => {
                if !matches!(self.ui, Ui::ReplayBrowser(_)) {
                    self.open_replay_browser();
                }
                self.tw_view.watching = Some((path, id));
                self.replay_browser_message(format!("Couldn't export this clip: {error}"), true);
            }
        }
    }

    /// The browser's clip export ended (the export closed the clip): back to the browser on the
    /// clip, saying how it went, with a Show button for the video.
    fn replay_clip_export_poll(&mut self) {
        if self.export_running() || self.replay_is_active() {
            return;
        }
        let Some((path, id)) = self.tw_view.clip_export.take() else {
            return;
        };
        let Some(outcome) = self.export_take_finished() else {
            return;
        };
        if !matches!(self.ui, Ui::ReplayBrowser(_)) {
            self.open_replay_browser();
        }
        self.tw_view.watching = Some((path, id));
        let now = Instant::now();
        let Ui::ReplayBrowser(screen) = &mut self.ui else {
            return;
        };
        match outcome.phase {
            ExportPhase::Done { path, bytes } => {
                let name = path
                    .file_name()
                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                screen.show_file_message(
                    format!(
                        "Saved \u{201c}{name}\u{201d} ({})",
                        mc_ui::screens::replay::format::size(bytes)
                    ),
                    path.display().to_string(),
                    now,
                );
            }
            ExportPhase::Failed(reason) => {
                screen.show_message(format!("The export failed: {reason}"), true, now);
            }
            ExportPhase::Cancelled => screen.show_message("Export cancelled", false, now),
            ExportPhase::Starting
            | ExportPhase::Settling
            | ExportPhase::Rendering
            | ExportPhase::Finishing => {}
        }
    }

    /// Open replay `id` in the viewer.
    fn replay_watch(&mut self, id: u64) {
        let Some(path) = self.replay_path_of(id) else {
            return;
        };
        self.replay_leave_for_replay();
        if let Err(error) = self.replay_open(&path) {
            // Back to a browser that says why.
            if !matches!(self.ui, Ui::ReplayBrowser(_)) {
                self.open_replay_browser();
            }
            self.tw_view.watching = Some((path, id));
            self.replay_browser_message(format!("Couldn't open this replay: {error}"), true);
            return;
        }
        let now = Instant::now();
        let meta = self.replay_meta().cloned().unwrap_or_default();
        let (start, _) = self.replay_view_range();
        let title = if meta.title.trim().is_empty() {
            path.file_stem()
                .map_or_else(|| "Replay".to_owned(), |s| s.to_string_lossy().into_owned())
        } else {
            // This computer's time in a default title from another zone.
            mc_replay::localize_title(&meta.title, meta.title_unix_ms())
                .unwrap_or_else(|| meta.title.clone())
        };
        let info = ReplayInfo {
            title,
            server: meta.server.clone(),
            started_unix_ms: meta.started_unix_ms,
            clip: matches!(meta.kind, ReplayKind::Clip { .. }),
            has_mic: meta.has_mic,
            markers: meta
                .markers
                .iter()
                .filter(|marker| marker.time >= start)
                .map(|marker| (marker.time - start, marker.label.clone()))
                .collect(),
            window_size: (0, 0),
        };
        let mut overlay = ReplayViewerOverlay::new(info, now);
        overlay.set_export_defaults(self.replay_export_defaults());
        // H.265 needs a GPU encoder or the HEVC Video Extensions (the probe, cached).
        overlay.set_h265_available(Client::export_h265_available());
        // The export window offers desktop audio when the replay has it; the
        // Sound menu starts from the channels' volumes (unmuted).
        overlay.set_has_desktop(meta.has_desktop);
        overlay.set_sound_channels(self.nw_re_channels());
        self.tw_view.overlay = Some(overlay);
        self.tw_view.watching = Some((path, id));
        self.tw_view.flying = false;
        self.tw_view.seek_started = None;
    }

    /// The playback state for the controls, on the timeline's clock (0 = the start of the
    /// visible window).
    fn replay_status(&mut self) -> PlaybackStatus {
        let (start, end) = self.replay_view_range();
        let seeking = self.replay_seek_progress();
        let now = Instant::now();
        // The "Seeking" chip shows only for a seek that takes a while (a rebuild), never for
        // the quick ones that finish in a frame or two.
        let seeking = match seeking {
            Some(progress) => {
                let since = *self.tw_view.seek_started.get_or_insert(now);
                (now.saturating_duration_since(since).as_millis() > 150).then_some(progress)
            }
            None => {
                self.tw_view.seek_started = None;
                None
            }
        };
        PlaybackStatus {
            duration_us: end.saturating_sub(start),
            time_us: self.replay_time().saturating_sub(start),
            playing: self.replay_is_playing(),
            speed: self.replay_speed(),
            camera: camera_from(self.replay_camera_mode()),
            seeking,
            flying: self.tw_view.flying,
        }
    }

    /// The controls' requests, into the engine (and the export job).
    fn replay_viewer_events(&mut self) {
        let events: Vec<ViewerEvent> = match self.tw_view.overlay.as_mut() {
            Some(overlay) => overlay.drain_events().collect(),
            None => return,
        };
        if events.is_empty() {
            return;
        }
        self.replay_viewer_apply(events);
        // The controls see the new state at once (a second key press before the next frame's
        // refresh must not act on the old one: Space twice is play, then pause).
        if self.replay_controls_up() {
            let status = self.replay_status();
            if let Some(overlay) = self.tw_view.overlay.as_mut() {
                overlay.set_status(status);
            }
        }
    }

    fn replay_viewer_apply(&mut self, events: Vec<ViewerEvent>) {
        let (start, _) = self.replay_view_range();
        for event in events {
            match event {
                ViewerEvent::Play => self.replay_play(),
                ViewerEvent::Pause => self.replay_pause(),
                ViewerEvent::Seek(t) => self.replay_seek(start + t),
                ViewerEvent::Step(ticks) => self.replay_step_tick(ticks > 0),
                ViewerEvent::Speed(speed) => self.replay_set_speed(speed),
                ViewerEvent::Camera(mode) => {
                    self.replay_set_camera_mode(camera_to(mode));
                    self.tw_view.flying = false;
                }
                ViewerEvent::RenderSettings => {
                    self.tw_view.flying = false;
                    self.open_mod_menu(Instant::now());
                }
                // The game's options over the replay; Done comes back.
                ViewerEvent::Options => {
                    self.tw_view.flying = false;
                    self.nw_rd_open_replay_options();
                }
                ViewerEvent::Export(request) => self.replay_export_start(request),
                ViewerEvent::CancelExport => self.replay_export_cancel(),
                ViewerEvent::ShowFile(path) => show_in_folder(Path::new(&path)),
                // The Sound menu: a channel's mute and volume.
                ViewerEvent::Sound {
                    channel,
                    muted,
                    volume,
                } => self.nw_re_set_channel(channel, muted, volume),
                ViewerEvent::Exit => {
                    self.replay_export_cancel();
                    let watching = self.tw_view.watching.clone();
                    self.tw_view.overlay = None;
                    self.tw_view.flying = false;
                    self.replay_close();
                    // Back to the browser, on the replay just watched.
                    self.open_replay_browser();
                    self.tw_view.watching = watching;
                }
            }
        }
    }

    /// Start the export the window asked for, as the export job: the form's range (on the
    /// timeline's clock) into the file's clock, its size, rate, codec, quality or size limit,
    /// motion blur and sound, into `Videos\Aero Client\<name>.mp4`. While it runs, RD's preview
    /// and panel own the window (this file draws and takes nothing); the outcome comes back to
    /// the window when it ends ([`Client::replay_export_poll`]).
    fn replay_export_start(&mut self, request: ExportRequest) {
        let (start, _) = self.replay_view_range();
        let mut job = self.export_defaults();
        job.start = start + request.start_us;
        job.end = start + request.end_us;
        job.width = request.width & !1;
        job.height = request.height & !1;
        job.fps = mc_video::FrameRate::whole(request.fps.max(1));
        job.codec = if request.h265 {
            mc_video::VideoCodec::H265
        } else {
            mc_video::VideoCodec::H264
        };
        job.target = match request.max_bytes {
            Some(bytes) => ExportTarget::MaxBytes(bytes),
            None => ExportTarget::Quality(request.quality),
        };
        job.motion_blur = request.blend_frames.max(1);
        job.game_audio = request.game_audio;
        job.mic = request.mic;
        job.desktop = request.desktop;
        job.separate_tracks = request.separate_tracks;
        let stem = request.file_name.trim_end_matches(".mp4");
        job.path = default_export_folder(&self.config.game_dir)
            .join(format!("{}.mp4", safe_file_name(stem)));
        if let Err(error) = self.export_start(job) {
            self.replay_export_show(ExportProgress {
                frames_done: 0,
                frames_total: 1,
                elapsed_s: 0.0,
                state: ExportState::Failed(error),
            });
        }
    }

    /// Stop the running export, if any (the export deletes the partial file).
    fn replay_export_cancel(&mut self) {
        if self.export_running() {
            self.export_cancel();
        }
    }

    /// Show an export's state in the open export window.
    fn replay_export_show(&mut self, progress: ExportProgress) {
        if let Some(dialog) = self
            .tw_view
            .overlay
            .as_mut()
            .and_then(ReplayViewerOverlay::export_dialog)
        {
            dialog.set_progress(progress);
        }
    }

    /// The export the window started has ended: its outcome (saved, failed, cancelled) into the
    /// window. Only taken while this window waits for one, so the other exports (the clip
    /// auto-export) keep theirs.
    fn replay_export_poll(&mut self) {
        let waiting = self
            .tw_view
            .overlay
            .as_ref()
            .is_some_and(ReplayViewerOverlay::export_running);
        if !waiting || self.export_running() {
            return;
        }
        let Some(outcome) = self.export_take_finished() else {
            return;
        };
        let state = match outcome.phase {
            ExportPhase::Done { path, bytes } => ExportState::Done {
                path: path.display().to_string(),
                bytes,
            },
            ExportPhase::Failed(reason) => ExportState::Failed(reason),
            ExportPhase::Cancelled => ExportState::Cancelled,
            ExportPhase::Starting
            | ExportPhase::Settling
            | ExportPhase::Rendering
            | ExportPhase::Finishing => ExportState::Failed("The export stopped.".to_owned()),
        };
        self.replay_export_show(ExportProgress {
            frames_done: outcome.frames_done,
            frames_total: outcome.frames_total,
            elapsed_s: outcome.elapsed.as_secs_f32(),
            state,
        });
    }

    /// Draw the browser or the replay controls into the replay UI's phase
    /// (end of `build_ui`), and act on what they asked. With no replay screen up: an enum test,
    /// an `Option` test and a scan of this frame's (usually empty) menu events.
    pub(crate) fn build_replay_ui(&mut self, gpu: &Gpu) {
        // An export owns the window (the preview and panel; the frames are the video):
        // nothing of this file draws until it ends.
        if self.export_running() {
            self.replay_view_release_phase();
            return;
        }
        if self.tw_view.overlay.is_some() {
            self.replay_export_poll();
        }
        if self.tw_view.clip_export.is_some() {
            self.replay_clip_export_poll();
        }
        // A replay that closed by itself (or was never opened) takes its controls with it.
        if self.tw_view.overlay.is_some() && !self.replay_is_active() {
            self.tw_view.overlay = None;
            self.tw_view.flying = false;
        }
        let browsing = matches!(self.ui, Ui::ReplayBrowser(_));
        let viewing = self.replay_controls_up();
        if !browsing && !viewing {
            if self.tw_view.flying {
                self.tw_view.flying = false;
            }
            self.replay_view_release_phase();
            return;
        }
        if browsing {
            self.replay_view_poll_listing();
        }
        if viewing {
            let status = self.replay_status();
            if let Some(overlay) = self.tw_view.overlay.as_mut() {
                overlay.set_window_size((gpu.config.width, gpu.config.height));
                overlay.set_status(status);
            }
        }
        let Some(slot) = self.tw_view.slot.clone() else {
            return;
        };
        if !self.tw_view.phase_live {
            slot.with(|slot| slot.phase = Some(ModMenuPhase::new(gpu)));
            self.tw_view.phase_live = true;
            self.tw_view.atlas_stale = true;
            menu_text::prewarm();
        }
        let mut list = slot.with(|slot| {
            slot.phase
                .as_mut()
                .map_or_else(DrawList::new, ModMenuPhase::take_spare_list)
        });
        let now = Instant::now();
        // Wrapped before the f32 conversion: no stutter after a long uptime.
        let time_s = crate::app::stability::cycle_seconds(
            now.saturating_duration_since(self.tw_view.epoch),
            crate::app::stability::UI_ANIMATION_CYCLE,
        );
        let text = self.tw_view.text.get_or_insert_with(MenuText::load);
        let mut env = ReplayEnv {
            text,
            mc_font: &mut self.font,
            screen: (gpu.config.width as f32, gpu.config.height as f32),
            now,
            time_s,
        };
        let glass = match &mut self.ui {
            Ui::ReplayBrowser(screen) => screen.build(&mut env, &mut list),
            _ => match self.tw_view.overlay.as_mut() {
                Some(overlay) => overlay.build(&mut env, &mut list),
                None => GlassFrame::default(),
            },
        };
        let stale = std::mem::take(&mut self.tw_view.atlas_stale);
        slot.with(|slot| {
            let Some(phase) = slot.phase.as_mut() else {
                return;
            };
            if let Some(text) = self.tw_view.text.as_mut() {
                if stale {
                    text.mark_dirty();
                }
                if let Some(pixels) = text.take_upload() {
                    phase.upload_atlas(gpu, pixels, menu_text::ATLAS_SIZE);
                }
                if !text.is_segoe() {
                    let _ = phase
                        .renderer
                        .ensure_list_textures(gpu, &self.assets.resources, &list);
                }
            }
            if list.is_empty() && !glass.is_visible() {
                // The controls have faded out while the replay plays: no pass at all.
                phase.deactivate();
            } else {
                phase.submit(list, glass);
            }
        });
        self.replay_browser_events();
        self.replay_viewer_events();
    }

    /// No replay screen is up: empty the slot (dropping the glass phase's pipelines and its
    /// scene copy). One `bool` test when it is already empty.
    fn replay_view_release_phase(&mut self) {
        if !self.tw_view.phase_live {
            return;
        }
        self.tw_view.phase_live = false;
        if let Some(slot) = &self.tw_view.slot {
            slot.with(|slot| slot.phase = None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_slot_sits_above_the_greenhouse_finish_and_under_the_title_marks() {
        assert!(UI_ORDER > mc_render::menu_scene::FINISH_ORDER);
        assert!(UI_ORDER < super::super::aero_title::CRISP_ORDER);
        assert!(
            UI_ORDER > PhaseOrder(PhaseOrder::UI.0 + 50),
            "over the mod menu's glass"
        );
        let slot = ReplaySlot { phase: None };
        assert!(!slot.enabled() && !slot.needs_scene_color());
    }

    #[test]
    fn ids_are_stable_per_path_and_differ_between_files() {
        let a = Path::new("C:/game/replays/a.aeroreplay");
        let b = Path::new("C:/game/replays/b.aeroreplay");
        assert_eq!(id_of(a), id_of(a));
        assert_ne!(id_of(a), id_of(b));
    }

    #[test]
    fn camera_modes_map_both_ways() {
        for mode in CameraMode::ALL {
            assert_eq!(camera_from(camera_to(mode)), mode);
        }
        assert_eq!(fps_of(options::CLIP_FPS_CHOICES, 1, 0), 60);
        assert_eq!(fps_of(options::EXPORT_FPS_CHOICES, 9, 0), 2000);
        assert_eq!(fps_of(options::EXPORT_FPS_CHOICES, 99, 30), 30);
    }
}

/// The replay screens through the real client: `Client::new` on the real assets, `install`, the
/// real `build_ui` and input paths, a headless GPU, real `.aeroreplay` files written with the writer, and the real playback engine. Needs the vanilla jar (skips without it; set
/// `MC_JARS_DIR`).
#[cfg(test)]
mod client_tests {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use mc_render::{Gpu, GpuOptions, InputState, PhaseList};
    use mc_replay::{ReplayKind, ReplayMeta, StreamId, Writer, WriterOptions};
    use winit::event::MouseButton;
    use winit::keyboard::KeyCode;

    use super::super::super::{Client, Ui};
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};

    const SIZE: (u32, u32) = (1280, 720);

    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    fn gpu(size: (u32, u32)) -> Option<Gpu> {
        match Gpu::headless(
            size.0,
            size.1,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => Some(gpu),
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("skip: no wgpu adapter available on this machine");
                None
            }
            Err(error) => panic!("headless GPU: {error}"),
        }
    }

    /// A small real replay file (a few packet records) in the client's library.
    fn write_replay(game_dir: &Path, name: &str, title: &str, clip: bool) -> PathBuf {
        let mut dir = game_dir.join(mc_replay::REPLAYS_DIR);
        if clip {
            dir = dir.join(mc_replay::CLIPS_DIR);
        }
        std::fs::create_dir_all(&dir).expect("replays dir");
        let path = dir.join(format!("{name}.{}", mc_replay::EXTENSION));
        let mut meta = ReplayMeta {
            title: title.to_owned(),
            server: "mc.example.net".to_owned(),
            started_unix_ms: 1_794_537_660_000,
            kind: if clip {
                ReplayKind::Clip {
                    start: 0,
                    end: 900_000,
                }
            } else {
                ReplayKind::Full
            },
            ..ReplayMeta::default()
        };
        let mut writer =
            Writer::create(&path, &meta, WriterOptions::default()).expect("the writer");
        for i in 0..10u64 {
            writer.push(i * 100_000, StreamId::Packet, &[0x02, 0, 0]);
        }
        // 30 s: the browser folds recordings under 20 s away.
        meta.duration = 30_000_000;
        writer
            .finish(meta)
            .wait(Duration::from_secs(20))
            .expect("the file is written");
        path
    }

    /// Frames until the browser's background listing has arrived.
    fn wait_for_listing(client: &mut Client, gpu: &Gpu) {
        for _ in 0..500 {
            client.build_ui(gpu, 0.0);
            if client.tw_view.listing.is_none() {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("the replay library listing never arrived");
    }

    fn slot_filled(client: &Client) -> bool {
        client
            .tw_view
            .slot
            .as_ref()
            .expect("installed")
            .with(|slot| slot.phase.is_some())
    }

    fn key(client: &mut Client, code: KeyCode, typed: Option<char>) {
        client.handle_raw_key(code, true, false, typed);
        let mut input = InputState::new();
        client.handle_screen_input(&mut input, SIZE.0, SIZE.1);
        client.handle_raw_key(code, false, false, None);
        client.handle_screen_input(&mut input, SIZE.0, SIZE.1);
    }

    #[test]
    fn the_browser_lists_renames_and_deletes_real_replay_files() {
        let Some(gpu) = gpu(SIZE) else { return };
        let Some((mut client, dir)) = offline_client() else {
            return;
        };
        let first = write_replay(dir.path(), "a", "mc.example.net", false);
        let second = write_replay(dir.path(), "b", "SkyWars", false);
        let clip = write_replay(dir.path(), "c", "Clip, mc.example.net", true);
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);
        assert!(
            phases.names().any(|n| n == "replay_ui"),
            "the slot is registered"
        );
        client.build_ui(&gpu, 0.0);
        assert!(!slot_filled(&client), "no replay screen: an empty slot");

        client.open_replay_browser();
        assert!(matches!(client.ui, Ui::ReplayBrowser(_)));
        wait_for_listing(&mut client, &gpu);
        assert!(
            slot_filled(&client),
            "the browser draws through its own phase"
        );
        assert_eq!(client.tw_view.paths.len(), 3, "two recordings and a clip");
        assert!(client.tw_view.paths.iter().any(|(_, p)| *p == clip));

        // Rename the selected recording from the keyboard: F2, type, Enter.
        let selected = match &client.ui {
            Ui::ReplayBrowser(screen) => screen.selected().expect("a selection"),
            _ => unreachable!(),
        };
        let path = client.replay_path_of(selected).expect("its file");
        // R renames (F2 is the game's screenshot key: it never reaches a screen).
        key(&mut client, KeyCode::KeyR, Some('r'));
        for c in "Best game".chars() {
            client.handle_raw_key(KeyCode::KeyA, true, false, Some(c));
        }
        let mut input = InputState::new();
        client.handle_screen_input(&mut input, SIZE.0, SIZE.1);
        key(&mut client, KeyCode::Enter, None);
        client.build_ui(&gpu, 0.0);
        let info = mc_replay::read_info(&path).expect("still a replay");
        assert_eq!(info.meta.title, "Best game", "renamed in the file");

        // Delete it: Delete twice.
        key(&mut client, KeyCode::Delete, None);
        client.build_ui(&gpu, 0.0);
        key(&mut client, KeyCode::Delete, None);
        client.build_ui(&gpu, 0.0);
        assert!(!path.exists(), "deleted");
        assert!(first.exists() || second.exists(), "only the one");

        // Esc goes back to the title screen (acted on at the end of that frame); the next frame
        // drops the phase.
        key(&mut client, KeyCode::Escape, None);
        client.build_ui(&gpu, 0.0);
        assert!(matches!(client.ui, Ui::MainMenu(_)));
        client.build_ui(&gpu, 0.0);
        assert!(
            !slot_filled(&client),
            "nothing kept once no replay screen is up"
        );
    }

    #[test]
    fn the_replay_mods_open_button_opens_the_browser_once() {
        let Some(gpu) = gpu(SIZE) else { return };
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);
        let mut camera = crate::app::driver::menu_camera();
        camera.set_viewport(SIZE.0, SIZE.1);
        client.mods.events.push(ModEvent::Button {
            mod_id: MOD_ID,
            option_id: options::ids::OPEN_VIEWER,
        });
        // The whole frame: the request must be taken before `drive_mods_frame` drops the menu
        // events (an earlier version looked for it in `build_ui`, after they were gone).
        client.frame(&gpu, &mut camera, 0.0);
        assert!(matches!(client.ui, Ui::ReplayBrowser(_)));
        assert!(client.mods.events.is_empty(), "the request is taken out");
        client.show_screen(None);
        client.frame(&gpu, &mut camera, 0.0);
        assert!(
            matches!(client.ui, Ui::MainMenu(_)),
            "not opened a second time"
        );
    }

    #[test]
    fn watching_frees_the_cursor_and_the_controls_take_the_keys() {
        let Some(gpu) = gpu(SIZE) else { return };
        let Some((mut client, dir)) = offline_client() else {
            return;
        };
        let path = write_replay(dir.path(), "game", "SkyWars", false);
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);
        client.open_replay_browser();
        wait_for_listing(&mut client, &gpu);
        // Enter watches the selected (only) replay through the engine.
        key(&mut client, KeyCode::Enter, None);
        client.build_ui(&gpu, 0.0);
        assert!(client.replay_is_active(), "the replay opened");
        assert!(client.tw_view.overlay.is_some(), "with its controls");
        assert_eq!(
            client.tw_view.watching.as_ref().map(|(p, _)| p),
            Some(&path)
        );
        // The world is up (RB puts the UI back in game when the replay's world is ready).
        client.ui = Ui::InGame;
        let mut input = InputState::new();
        assert!(
            client.reconcile_cursor_grab(&mut input, SIZE.0, SIZE.1),
            "the cursor is free over the replay"
        );
        client.build_ui(&gpu, 0.0);
        assert!(slot_filled(&client));
        // Space plays (the controls take it; the game never sees it).
        assert!(!client.replay_is_playing());
        key(&mut client, KeyCode::Space, Some(' '));
        client.build_ui(&gpu, 0.0);
        assert!(client.replay_is_playing(), "Space reached the replay");
        key(&mut client, KeyCode::Space, Some(' '));
        client.build_ui(&gpu, 0.0);
        assert!(!client.replay_is_playing());
        // The free camera flies only while the right button is held: then the cursor is
        // grabbed.
        client.replay_set_camera_mode(ReplayCameraMode::Free);
        input.on_mouse_button(MouseButton::Right, true);
        client.handle_screen_input(&mut input, SIZE.0, SIZE.1);
        assert!(client.tw_view.flying);
        assert!(!client.reconcile_cursor_grab(&mut input, SIZE.0, SIZE.1));
        input.on_mouse_button(MouseButton::Right, false);
        client.handle_screen_input(&mut input, SIZE.0, SIZE.1);
        assert!(client.reconcile_cursor_grab(&mut input, SIZE.0, SIZE.1));
        // Esc leaves the replay, back to the browser on it.
        key(&mut client, KeyCode::Escape, None);
        client.build_ui(&gpu, 0.0);
        assert!(!client.replay_is_active(), "the replay closed");
        assert!(
            matches!(client.ui, Ui::ReplayBrowser(_)),
            "back to the browser"
        );
        wait_for_listing(&mut client, &gpu);
        let selected = match &client.ui {
            Ui::ReplayBrowser(screen) => screen.selected(),
            _ => None,
        };
        assert_eq!(selected, Some(id_of(&path)), "on the replay just watched");
    }

    /// The Folders view through the real client: real files in a folder
    /// of the player's own, an empty folder and a junction out of the library. The listing's keys
    /// reach the browser; Explorer opens only a folder the listing found (never one made from a
    /// key, nor one swapped for a junction since); the view is remembered in the `replay` mod's
    /// settings and the open folder for the session.
    #[test]
    fn the_browser_shows_folders_and_opens_only_folders_it_listed() {
        use crate::app::driver::take_opened_external_targets;
        use mc_ui::screens::replay::arrange::ListView;

        let Some(gpu) = gpu(SIZE) else { return };
        let Some((mut client, dir)) = offline_client() else {
            return;
        };
        let replays = dir.path().join(mc_replay::REPLAYS_DIR);
        let flat = write_replay(dir.path(), "a", "mc.example.net", false);
        let best = replays.join("Best games");
        std::fs::create_dir_all(&best).unwrap();
        std::fs::rename(&flat, best.join("a.aeroreplay")).unwrap();
        write_replay(dir.path(), "b", "SkyWars", false);
        std::fs::create_dir_all(replays.join("Empty one")).unwrap();
        // A junction out of the library, to a folder holding a replay (made with `mklink /J`,
        // which needs no special rights).
        let outside = tempfile::tempdir().unwrap();
        write_replay(outside.path(), "secret", "outside", false);
        let link = replays.join("link out");
        let junction = |link: &Path, target: &Path| {
            cfg!(windows)
                && std::process::Command::new("cmd")
                    .arg("/C")
                    .arg("mklink")
                    .arg("/J")
                    .arg(link)
                    .arg(target)
                    .output()
                    .is_ok_and(|out| out.status.success())
        };
        let linked = junction(&link, &outside.path().join(mc_replay::REPLAYS_DIR));
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);
        client.open_replay_browser();
        wait_for_listing(&mut client, &gpu);

        // The listing: the two folders (the empty one too), never the junction.
        let mut keys: Vec<&str> = client.tw_view.folders[0]
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, ["Best games", "Empty one"]);
        assert_eq!(
            client.tw_view.paths.len(),
            2,
            "the outside replay is not listed"
        );

        // F: the Folders view, remembered in the mod's settings, on the selected replay's folder.
        key(&mut client, KeyCode::KeyF, Some('f'));
        client.build_ui(&gpu, 0.0);
        assert_eq!(
            client
                .mods
                .settings
                .text(MOD_ID, options::ids::BROWSER_VIEW),
            "folders"
        );
        let Ui::ReplayBrowser(screen) = &mut client.ui else {
            unreachable!()
        };
        assert_eq!(screen.arrangement().view, ListView::Folders);
        assert_eq!(screen.folder_key(Tab::Recordings), "Best games");
        assert_eq!(screen.rows(), [(false, "mc.example.net".to_owned())]);
        // Backspace: the root, folders first.
        key(&mut client, KeyCode::Backspace, None);
        client.build_ui(&gpu, 0.0);
        let Ui::ReplayBrowser(screen) = &mut client.ui else {
            unreachable!()
        };
        assert_eq!(
            screen.rows(),
            [
                (true, "Best games".to_owned()),
                (true, "Empty one".to_owned()),
                (false, "SkyWars".to_owned())
            ]
        );
        assert_eq!(screen.selected_folder(), Some("Best games"));

        // Explorer opens a listed folder (its real path), and the tab's root...
        let _ = take_opened_external_targets();
        client.replay_open_library_folder(false, "Best games");
        assert_eq!(take_opened_external_targets(), std::slice::from_ref(&best));
        client.replay_open_library_folder(false, "");
        assert_eq!(
            take_opened_external_targets(),
            std::slice::from_ref(&replays)
        );
        // ...never a key the listing did not find, whatever it says.
        for key in [
            "Missing",
            "..",
            "../..",
            "Best games/..",
            "link out",
            "C:",
            "/",
        ] {
            client.replay_open_library_folder(false, key);
        }
        assert!(take_opened_external_targets().is_empty());
        // A listed folder swapped for a junction since the listing: refused.
        if linked {
            let moved = replays.join("moved away");
            std::fs::rename(&best, &moved).unwrap();
            assert!(junction(&best, outside.path()), "the swap");
            client.replay_open_library_folder(false, "Best games");
            assert!(take_opened_external_targets().is_empty());
            std::fs::remove_dir(&best).unwrap();
            std::fs::rename(&moved, &best).unwrap();
        } else {
            eprintln!("SKIP: no junction could be made here (mklink /J)");
        }

        // Open "Best games", leave, come back: the view and the folder are where they were.
        key(&mut client, KeyCode::Enter, None);
        client.build_ui(&gpu, 0.0);
        key(&mut client, KeyCode::Escape, None);
        client.build_ui(&gpu, 0.0);
        assert!(matches!(client.ui, Ui::MainMenu(_)));
        client.open_replay_browser();
        wait_for_listing(&mut client, &gpu);
        let Ui::ReplayBrowser(screen) = &mut client.ui else {
            unreachable!()
        };
        assert_eq!(screen.arrangement().view, ListView::Folders);
        assert_eq!(screen.folder_key(Tab::Recordings), "Best games");
        assert_eq!(screen.rows(), [(false, "mc.example.net".to_owned())]);
        if linked {
            let _ = std::fs::remove_dir(&link);
        }
    }

    /// The Folders view and the Server order over a real library: a game
    /// folder of the player's own replays (copies), listed, organized and drawn by the real
    /// client, read back as PNGs. Only by hand, with both variables set:
    ///
    /// `MC_BROWSER_REVIEW_GAMEDIR=<a copy of a game folder> MC_BROWSER_REVIEW_DIR=<png dir>
    /// cargo test -p mc-client --lib browser_folders_review -- --ignored`
    #[test]
    #[ignore = "pictures of a real library; run by hand with MC_BROWSER_REVIEW_GAMEDIR"]
    fn browser_folders_review_renders() {
        use mc_ui::screens::replay::arrange::{Arrangement, ListView, SortOrder};

        let (Some(game_dir), Some(out)) = (
            std::env::var_os("MC_BROWSER_REVIEW_GAMEDIR").map(PathBuf::from),
            std::env::var_os("MC_BROWSER_REVIEW_DIR").map(PathBuf::from),
        ) else {
            eprintln!("skip: set MC_BROWSER_REVIEW_GAMEDIR and MC_BROWSER_REVIEW_DIR");
            return;
        };
        std::fs::create_dir_all(&out).expect("review dir");
        let size = (1920, 1080);
        let Some(mut gpu) = gpu(size) else { return };
        let config = ClientConfig {
            game_dir,
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, size.0, size.1);
        let mut camera = crate::app::driver::menu_camera();
        camera.set_viewport(size.0, size.1);
        client.open_replay_browser();
        for _ in 0..400 {
            client.frame(&gpu, &mut camera, 0.0);
            if client.tw_view.listing.is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        for (arrangement, name) in [
            (
                Arrangement {
                    sort: SortOrder::Newest,
                    view: ListView::Folders,
                },
                "real-folders-root",
            ),
            (
                Arrangement {
                    sort: SortOrder::Server,
                    view: ListView::List,
                },
                "real-by-server",
            ),
        ] {
            if let Ui::ReplayBrowser(screen) = &mut client.ui {
                screen.set_arrangement(arrangement);
            }
            for _ in 0..12 {
                client.frame(&gpu, &mut camera, 0.0);
                std::thread::sleep(Duration::from_millis(25));
            }
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &camera, 0.0);
            let pixels = gpu.read_pixels().expect("readback");
            let path = out.join(format!("{name}-{}x{}.png", size.0, size.1));
            image::save_buffer(
                &path,
                &pixels,
                size.0,
                size.1,
                image::ExtendedColorType::Rgba8,
            )
            .expect("write the PNG");
            eprintln!("wrote {}", path.display());
        }
    }

    /// The browser through the real client: three real replay files, the title screen's
    /// greenhouse behind (a later visit, past its intro), `Client::frame` and the installed phase
    /// list, read back at 1920x1080 and 1280x720. Writes PNGs only when `MC_RC_REVIEW_DIR` is set:
    ///
    /// `MC_RC_REVIEW_DIR=<dir> cargo test -p mc-client --lib replay_browser_review -- --ignored`
    #[test]
    #[ignore = "writes the review PNGs; run explicitly with MC_RC_REVIEW_DIR"]
    fn replay_browser_review_renders() {
        let Some(out) = std::env::var_os("MC_RC_REVIEW_DIR").map(PathBuf::from) else {
            eprintln!("skip: set MC_RC_REVIEW_DIR to write the review images");
            return;
        };
        std::fs::create_dir_all(&out).expect("review dir");
        for size in [(1920, 1080), (1280, 720)] {
            let Some(mut gpu) = gpu(size) else { return };
            let Some((mut client, dir)) = offline_client() else {
                return;
            };
            write_replay(dir.path(), "a", "mc.example.net", false);
            write_replay(dir.path(), "b", "SkyWars \u{b7} Dwarf Fortress", false);
            write_replay(dir.path(), "c", "Clip, mc.example.net", true);
            let mut phases = PhaseList::new();
            client.install(&gpu, &mut phases, size.0, size.1);
            let mut camera = crate::app::driver::menu_camera();
            camera.set_viewport(size.0, size.1);
            // The session's first title screen plays the greenhouse's intro; leave and come back
            // so the browser opens over the settled scene, as it does for a player.
            client.frame(&gpu, &mut camera, 0.0);
            client.show_screen(Some(Ui::Options(Box::new(
                mc_ui::screens::options::OptionsScreen::new(false),
            ))));
            client.frame(&gpu, &mut camera, 0.0);
            client.show_screen(None);
            client.frame(&gpu, &mut camera, 0.0);
            client.open_replay_browser();
            for _ in 0..40 {
                client.frame(&gpu, &mut camera, 0.0);
                if client.tw_view.listing.is_none() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            // The window's pop-in (200 ms of real time) and the uploads.
            for _ in 0..12 {
                client.frame(&gpu, &mut camera, 0.0);
                std::thread::sleep(Duration::from_millis(25));
            }
            assert!(client.replay_browser_over_title());
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &camera, 0.0);
            let pixels = gpu.read_pixels().expect("readback");
            let path = out.join(format!("client-browser-{}x{}.png", size.0, size.1));
            image::save_buffer(
                &path,
                &pixels,
                size.0,
                size.1,
                image::ExtendedColorType::Rgba8,
            )
            .expect("write the PNG");
            eprintln!("wrote {}", path.display());
        }
    }
}
