//! The PvP mod framework's client half.
//!
//! This module is declared *inside* `app::driver` (`#[path = "mods/mod.rs"] mod mods;`), so the
//! code here and in every lane file under `app/mods/` can read `Client`'s private fields and
//! call its private methods. Lanes keep their state and logic in their own file here, as an
//! `impl Client` block plus a state struct, and call one hook from `driver.rs` at their
//! `<<MOD-Mx:...>>` marker.
//!
//! What lives here:
//! * [`ModRuntime`]: the settings store, its debounced persistence to `<gameDir>/mc-rs-mods.json`,
//!   the HUD pipeline and the per-frame [`HudData`];
//! * the `impl Client` hooks the driver calls: [`Client::drive_mods_frame`],
//!   [`Client::draw_mod_hud`], [`Client::sample_mod_keys`], [`Client::flush_mod_settings`].

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_mods::{DynamicRows, ModEvent, ModSettings};
use mc_ui::modhud::{HudCtx, HudData, HudEffect, HudKeys, ModHud};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use mc_ui::screens::modmenu::text::{self as menu_text, MenuText};
use mc_ui::screens::modmenu::{
    KeyOutcome, MenuEnv, MenuGame, MenuInput, MenuPlace, ModMenuScreen, EDITOR_DIM_ARGB,
};
use mc_ui::widget::{DrawList, ScaledResolution};
use winit::keyboard::KeyCode;

use super::{Client, Shared, Ui};

// ---- lane modules: declare yours under your marker ----------------------------------------
// The MC|Brand value servers are told (always `Aero`).
pub(crate) mod client_brand;
pub(crate) mod independent_gui;
pub(crate) mod m1_hud;
pub(crate) mod m2_interface;
pub(crate) mod m3_controls;
pub(crate) mod m4_visual;
pub(crate) mod m5_animations;
pub(crate) mod m6_particles;
pub(crate) mod m7_postfx;
pub(crate) mod m8_ctm;
pub(crate) mod m9_waypoints;
// ---- One marker per lane ----------
pub(crate) mod w3_sky;
pub(crate) mod w4_fog_vignette;
pub(crate) mod w5_grass;
pub(crate) mod w9_spotify;
// ---- the Aero title screen -----------------------
pub(crate) mod aero_title;
// ---- One marker per lane ---------
pub(crate) mod auto_gg;
pub(crate) mod x2_chat;
pub(crate) mod x2_screenshot;
pub(crate) mod x3_screens;
pub(crate) mod x4_input;
pub(crate) mod x5_cosmetics;
pub(crate) mod x6_entities;
pub(crate) mod x7_world;
pub(crate) mod x8_ao;
pub(crate) mod x9_aa;
pub(crate) mod x10_blur;
pub(crate) mod x11_clouds;
pub(crate) mod x13_markers;
// ---- One file per lane --------------
pub(crate) mod tw_shaders;
pub(crate) mod tw_dreamy;
pub(crate) mod tw_underwater;
pub(crate) mod tw_replay_rec;
pub(crate) mod tw_replay_play;
pub(crate) mod tw_replay_view;
pub(crate) mod tw_replay_export;
pub(crate) mod tw_loading;
// The title screen's three scenes.
pub(crate) mod nw_menu;
pub(crate) mod nw_text;
// The game's options over a replay (the viewer's Options button).
pub(crate) mod nw_replay_options;
// The ASCII Super Secret Setting.
pub(crate) mod nw_rfx;
pub(crate) mod nw_dynamic_fov;
// The replay viewer's sound channels (recorded voices, volumes, mutes).
pub(crate) mod nw_replay_voice;
// HUD/text FPS candidates.
pub(crate) mod nw_f3;
// Entity submission CPU candidates.
pub(crate) mod nw_f1_entities;
// Busy-lobby and tick-frame CPU candidates.
pub(crate) mod w18_oa;
// Always show mob names; Confirm Disconnect.
pub(crate) mod w18_fb;
// Client-lifecycle parity hooks.
pub(crate) mod w18_pb;
// Symbol shadows (the Chat mod's option).
pub(crate) mod w18_fa;
// GUI container sheets prewarmed at install.
pub(crate) mod w19_o2;
// Tests of the F3 memory figures (OS counters).
mod w21_mb;
// The leak harness's mod-menu hooks, and bounds on
// state that outlived what it belonged to.
pub(crate) mod w21_ml_bounds;
pub(crate) mod w21_ml_hooks;
// Accounts (title corner, Accounts screen).
pub(crate) mod w21_accounts;
// The updater's title-screen card and hooks.
pub(crate) mod w21_updater;
// Background FPS: the frame cap while the window is in the background.
pub(crate) mod background_fps;
// The version the client plays (1.8.9 or 1.7.10): the title screen's slider, the mod menu's
// Game version row, `--gameVersion`, and what a connection takes from them.
pub(crate) mod game_version;

/// A change is written once settings have been still for this long...
const SAVE_DEBOUNCE: Duration = Duration::from_millis(750);
/// ...or this long after the first unsaved change, whichever comes first (a long slider drag
/// still reaches the disk).
const SAVE_MAX_DELAY: Duration = Duration::from_millis(3000);

#[derive(Debug, Clone, Copy)]
struct PendingSave {
    generation: u64,
    first_change: Instant,
    last_change: Instant,
}

/// Writes settings snapshots off the main thread (on a slow disk one write + rename measured
/// 38 ms to 1.2 s). The main thread only serialises (~9 KB) and queues;
/// a newer snapshot replaces an older one that has not been written yet; shutdown waits.
#[derive(Debug, Default)]
struct SettingsWriter {
    shared: std::sync::Arc<(std::sync::Mutex<WriterState>, std::sync::Condvar)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[derive(Debug, Default)]
struct WriterState {
    queued: Option<(PathBuf, String)>,
    writing: bool,
    shutdown: bool,
}

impl SettingsWriter {
    /// Queue `text` for `path`, replacing a snapshot still waiting. The thread starts on the
    /// first save, so a session that never changes a setting never spawns it.
    fn submit(&mut self, path: PathBuf, text: String) {
        {
            let (lock, wake) = &*self.shared;
            let mut state = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.queued = Some((path, text));
            wake.notify_all();
        }
        if self.thread.is_none() {
            let shared = std::sync::Arc::clone(&self.shared);
            match std::thread::Builder::new()
                .name("mc-rs mod settings writer".into())
                .spawn(move || writer_loop(&shared))
            {
                Ok(handle) => self.thread = Some(handle),
                Err(error) => {
                    // No thread: write inline rather than lose the change.
                    tracing::warn!(%error, "settings writer thread unavailable; writing inline");
                    let job = {
                        let (lock, _) = &*self.shared;
                        lock.lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .queued
                            .take()
                    };
                    if let Some((path, text)) = job {
                        write_settings_file(&path, &text);
                    }
                }
            }
        }
    }

    /// Block until nothing is queued or being written (shutdown, tests). Bounded, so a wedged
    /// disk cannot hang exit forever. `false` when the bound ran out.
    fn wait_idle(&self) -> bool {
        if self.thread.is_none() {
            return true;
        }
        let (lock, wake) = &*self.shared;
        let deadline = Instant::now() + WRITER_EXIT_WAIT;
        let mut state = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.queued.is_some() || state.writing {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                tracing::warn!("mod settings writer did not finish within 10 s");
                return false;
            }
            state = wake
                .wait_timeout(state, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        true
    }
}

/// How long shutdown waits for the mod settings writer.
const WRITER_EXIT_WAIT: Duration = Duration::from_secs(10);

impl Drop for SettingsWriter {
    fn drop(&mut self) {
        let idle = self.wait_idle();
        {
            let (lock, wake) = &*self.shared;
            lock.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .shutdown = true;
            wake.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            // A write still stuck after the bounded wait (a disk that
            // stopped answering: a pulled USB drive, a sleeping network share) is left to finish
            // on its own, as the waypoint writer does. Joining it hung the exit for as long as
            // the write did.
            if idle {
                let _ = thread.join();
            }
        }
    }
}

fn writer_loop(shared: &(std::sync::Mutex<WriterState>, std::sync::Condvar)) {
    let (lock, wake) = shared;
    let mut state = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if let Some((path, text)) = state.queued.take() {
            state.writing = true;
            drop(state);
            write_settings_file(&path, &text);
            state = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.writing = false;
            wake.notify_all();
            continue;
        }
        if state.shutdown {
            return;
        }
        state = wake
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

/// Write through a temporary file, flushed to the disk, and a rename, so neither a crash nor a
/// power cut mid-write truncates the file (`mc_core::durable::write_atomic`;
/// this runs on the writer thread, so the flush never costs a frame).
fn write_settings_file(path: &Path, text: &str) {
    let result = mc_core::durable::write_atomic(path, text.as_bytes());
    match result {
        Ok(()) => tracing::debug!(path = %path.display(), "mod settings saved"),
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "mod settings were not saved");
        }
    }
}

/// Cached facts the frame path checks, refreshed when the settings generation moves.
#[derive(Debug, Default, Clone, Copy)]
struct FrameworkSnapshot {
    seen: u64,
    keystrokes: bool,
}

/// Everything the framework keeps on the client (`Client::mods`).
#[derive(Debug)]
pub struct ModRuntime {
    /// The settings store. Mutating it bumps its generation; the frame hook persists it.
    pub settings: ModSettings,
    /// `<gameDir>/mc-rs-mods.json`.
    path: PathBuf,
    /// The generation that is on disk (or was loaded from it, or is queued for the writer).
    saved_generation: u64,
    pending: Option<PendingSave>,
    writer: SettingsWriter,
    /// The HUD element pipeline.
    pub hud: ModHud,
    /// This frame's HUD snapshot; filled only while a HUD element is enabled.
    pub hud_data: HudData,
    snapshot: FrameworkSnapshot,
    /// This frame's menu events (button clicks, row actions, cycle clicks). Pushed while the
    /// menu handles input, read by the lanes' `<<MOD-Mx:frame>>` hooks, cleared at the end of
    /// `Client::frame`. Use [`ModRuntime::events_for`].
    pub events: Vec<ModEvent>,
    /// Dynamic menu rows, one list per catalog index. Filled by the lanes in
    /// [`Client::fill_menu_rows`] while the menu shows that mod's section.
    pub rows: Vec<DynamicRows>,
    /// Catalog index of the mod whose section is expanded in the open menu.
    pub(crate) menu_expanded: Option<usize>,
    /// The menu's render phase (`install_mod_phases`); `None` until the world renderer exists.
    menu_phase: Option<Shared<ModMenuPhase>>,
    /// The menu's Segoe UI text and glyph atlas, created on the first open.
    menu_text: Option<MenuText>,
    /// The phase's renderer has not seen the current glyph atlas (a phase rebuilt by a
    /// resource reload starts empty).
    menu_atlas_stale: bool,
    /// A closed menu still playing its closing animation (the game already has input back).
    closing_menu: Option<Box<ModMenuScreen>>,
    /// The menu is open or animating out. While `false` every menu hook is one `bool` test.
    menu_live: bool,
    /// The cursor in physical pixels (the menu works in them), refreshed every frame while
    /// `menu_live`.
    menu_cursor: (f32, f32),
    /// The scaled GUI size and the physical pixels per GUI pixel, for the HUD editor.
    menu_gui: ((f32, f32), f32),
    /// Origin of the glass shader's `uTime`.
    menu_epoch: Instant,
    // The open menu's search, as typed (empty
    // when not searching). While it is non-empty every mod with a dynamic-row hook fills its
    // rows each frame (`Client::fill_menu_rows`), so the menu can match their labels.
    menu_search: String,
    // Where the last menu was (its list and mod), so
    // the next open of this client run starts there (memory only, never saved).
    menu_place: Option<MenuPlace>,
    /// The screen the menu was opened over (`Client::open_mod_menu_over`): closing the menu
    /// returns to it instead of the game.
    menu_return: Option<Box<Ui>>,
}

impl ModRuntime {
    /// Load `<game_dir>/mc-rs-mods.json` (a missing file is a first run). A corrupt file is
    /// copied to `mc-rs-mods.json.bad` before anything can overwrite it.
    #[must_use]
    pub fn load(game_dir: &Path) -> ModRuntime {
        let path = game_dir.join(mc_mods::SETTINGS_FILE_NAME);
        let (settings, report) = ModSettings::load(&path);
        if report.file_missing {
            tracing::info!(
                path = %path.display(),
                "no mod settings yet; every mod is at its default"
            );
        } else {
            tracing::info!(path = %path.display(), "mod settings loaded");
        }
        for warning in &report.warnings {
            tracing::warn!("mod settings: {warning}");
        }
        // Entries from an older build, read in today's meaning (the file
        // itself is upgraded by the next save).
        for note in &report.migrated {
            tracing::info!("mod settings: {note}");
        }
        // A file read with replacement characters is kept too.
        if (report.corrupt || report.reencoded) && path.exists() {
            let mut bad = path.as_os_str().to_owned();
            bad.push(".bad");
            if let Err(error) = std::fs::copy(&path, PathBuf::from(&bad)) {
                tracing::warn!(%error, "could not keep a copy of the unreadable mod settings");
            }
        }
        ModRuntime::with_settings(settings, path)
    }

    /// A runtime over explicit settings (tests).
    #[must_use]
    pub fn with_settings(settings: ModSettings, path: PathBuf) -> ModRuntime {
        let saved_generation = settings.generation();
        let rows = vec![DynamicRows::default(); settings.catalog().len()];
        ModRuntime {
            settings,
            path,
            saved_generation,
            pending: None,
            writer: SettingsWriter::default(),
            hud: ModHud::new(),
            hud_data: HudData::default(),
            snapshot: FrameworkSnapshot::default(),
            events: Vec::new(),
            rows,
            menu_expanded: None,
            menu_phase: None,
            menu_text: None,
            menu_atlas_stale: false,
            closing_menu: None,
            menu_live: false,
            menu_cursor: (-1.0, -1.0),
            menu_gui: ((0.0, 0.0), 1.0),
            menu_epoch: Instant::now(),
            menu_search: String::new(),
            menu_place: None,
            menu_return: None,
        }
    }

    /// The open menu's search as typed, empty when not searching. A lane whose dynamic rows
    /// are a long list (a per-sound list) may filter them by it in its `fill_menu_rows` hook;
    /// the menu matches row labels against it either way (all space-separated terms, ASCII
    /// case-insensitive, see `mc_ui::screens::modmenu`).
    #[must_use]
    #[allow(dead_code)]
    pub fn menu_search(&self) -> &str {
        &self.menu_search
    }

    /// This frame's menu events for one mod.
    // The lane-facing API is used by the lanes' files as they land.
    #[allow(dead_code)]
    pub fn events_for<'a>(&'a self, mod_id: &'a str) -> impl Iterator<Item = &'a ModEvent> + 'a {
        self.events
            .iter()
            .filter(move |event| event.mod_id() == mod_id)
    }

    /// One mod's dynamic rows (see [`Client::fill_menu_rows`]).
    #[allow(dead_code)]
    pub fn rows_mut(&mut self, mod_id: &str) -> Option<&mut DynamicRows> {
        let index = self.settings.index_of(mod_id)?;
        self.rows.get_mut(index)
    }

    /// Whether the open menu shows this mod's expanded section right now (so its dynamic rows
    /// are worth building).
    #[must_use]
    #[allow(dead_code)]
    pub fn menu_shows(&self, mod_id: &str) -> bool {
        self.menu_expanded
            .is_some_and(|index| self.settings.catalog()[index].id == mod_id)
    }

    /// Whether the store has changes that are not on disk yet.
    #[must_use]
    pub fn has_unsaved_changes(&self) -> bool {
        self.settings.generation() != self.saved_generation
    }

    fn refresh_snapshot(&mut self) {
        let generation = self.settings.generation();
        if self.snapshot.seen == generation {
            return;
        }
        self.snapshot = FrameworkSnapshot {
            seen: generation,
            keystrokes: self.settings.enabled("keystrokes"),
        };
    }

    /// The debounced save. One `u64` compare on a frame with nothing pending.
    pub fn drive_persistence(&mut self, now: Instant) {
        let generation = self.settings.generation();
        if generation == self.saved_generation {
            self.pending = None;
            return;
        }
        let pending = self.pending.get_or_insert(PendingSave {
            generation,
            first_change: now,
            last_change: now,
        });
        if pending.generation != generation {
            pending.generation = generation;
            pending.last_change = now;
        }
        let quiet = now.saturating_duration_since(pending.last_change) >= SAVE_DEBOUNCE;
        let overdue = now.saturating_duration_since(pending.first_change) >= SAVE_MAX_DELAY;
        if quiet || overdue {
            self.save_now();
        }
    }

    /// Queue anything unsaved and **wait** until it is on disk (shutdown).
    pub fn flush(&mut self) {
        if self.has_unsaved_changes() {
            self.save_now();
        }
        self.writer.wait_idle();
    }

    /// Wait for queued writes without queueing anything (tests).
    #[cfg(test)]
    fn wait_for_writes(&self) {
        self.writer.wait_idle();
    }

    /// Serialise on this thread (cheap) and hand the bytes to the writer thread; no file I/O
    /// here. A failed write is logged by the writer and retried by the next change.
    fn save_now(&mut self) {
        let generation = self.settings.generation();
        let mut text = self.settings.to_json_string();
        text.push('\n');
        self.writer.submit(self.path.clone(), text);
        self.saved_generation = generation;
        self.pending = None;
    }
}

impl Client {
    /// Raw keyboard edges at event time, before the shell applies them to the key bindings and
    /// before any screen sees them. Return `true` to swallow the event (it then reaches neither
    /// the screen queue nor, if a screen is open by then, the bindings).
    ///
    /// Repeats arrive here too (`repeat == true`); most hooks want to ignore them.
    pub(crate) fn mod_raw_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        pressed: bool,
        repeat: bool,
    ) -> bool {
        // The menu key (Right Shift by default) in game opens the mod menu (see `mod_menu_raw_key`).
        if self.mod_menu_raw_key(code, pressed, repeat) {
            return true;
        }
        false
    }

    /// Register the framework's render phases (the menu's glass pass). Called from
    /// `Client::install`, so it also runs after a resource-pack reload rebuilt every phase.
    ///
    /// The menu phase records after the UI (`PhaseOrder::UI + 50`): its glass refracts the
    /// finished frame, HUD included, the way glass.js refracts the desktop. While the menu is
    /// closed the phase is disabled, so the frame path is exactly today's.
    pub(crate) fn install_mod_phases(
        &mut self,
        gpu: &mc_render::Gpu,
        phases: &mut mc_render::PhaseList,
    ) {
        let phase = Shared::new(ModMenuPhase::new(gpu));
        phases.insert(
            mc_render::PhaseOrder(mc_render::PhaseOrder::UI.0 + 50),
            Box::new(phase.clone()),
        );
        self.mods.menu_phase = Some(phase);
        // A new phase has a new renderer: the glyph atlas goes up again on the next build.
        self.mods.menu_atlas_stale = true;
        // Read the Segoe UI files off the main thread now, so the first open is instant.
        menu_text::prewarm();
    }

    /// The framework's per-frame work at the end of `Client::frame`: debounced persistence,
    /// then this frame's menu events are dropped (the lanes' frame hooks have run).
    pub(crate) fn drive_mods_frame(&mut self) {
        self.mods.drive_persistence(Instant::now());
        self.mods.events.clear();
    }

    /// Let the expanded mod's lane rebuild its dynamic menu rows. The menu calls this once per
    /// frame, **only while it is open and a mod with a lane hook is expanded**, so it costs
    /// nothing in game. Each lane: `rows.begin()`, then `rows.push(key)` per row, filling the
    /// returned row in place (no allocation once warm).
    ///
    /// While the menu's search is non-empty, every mod with a hook fills
    /// its rows (the menu shows the rows whose label matches, under their mod, without it being
    /// expanded). `ModRuntime::menu_search` holds the search for hooks that filter long lists.
    pub(crate) fn fill_menu_rows(&mut self) {
        if !self.mods.menu_search.is_empty() {
            for index in 0..self.mods.rows.len() {
                self.fill_menu_rows_at(index);
            }
            return;
        }
        if let Some(index) = self.mods.menu_expanded {
            self.fill_menu_rows_at(index);
        }
    }

    /// One mod's dynamic rows: the lanes' hooks, by mod id (a no-op for mods without rows).
    pub(crate) fn fill_menu_rows_at(&mut self, index: usize) {
        if self.mods.settings.catalog()[index].id == "waypoints" {
            self.fill_waypoint_rows(index);
        }
        if self.mods.settings.catalog()[index].id == game_version::MENU_MOD_ID {
            self.fill_game_version_row(index);
        }
    }

    /// Write unsaved mod settings now (shutdown).
    pub(crate) fn flush_mod_settings(&mut self) {
        self.mods.flush();
    }

    /// Sample the binding states keystrokes draws. Called once per frame from
    /// `ShellApp::frame`, the only place the live (shell-owned) key bindings are in scope; a
    /// `bool` test while keystrokes is off.
    pub(crate) fn sample_mod_keys(&mut self, keys: &mc_render::input::KeyBindings) {
        self.mods.refresh_snapshot();
        if !self.mods.snapshot.keystrokes {
            return;
        }
        use mc_render::input::keys as bind;
        self.mods.hud_data.keys = HudKeys {
            forward: keys.is_down(bind::FORWARD),
            left: keys.is_down(bind::LEFT),
            back: keys.is_down(bind::BACK),
            right: keys.is_down(bind::RIGHT),
            jump: keys.is_down(bind::JUMP),
            sneak: keys.is_down(bind::SNEAK),
            attack: keys.is_down(bind::ATTACK),
            use_item: keys.is_down(bind::USE_ITEM),
        };
    }

    /// Draw every enabled mod HUD element. Called from `draw_ingame_overlay` after the vanilla
    /// HUD and scoreboard, before chat, the tab list and F3. With every HUD mod off and the
    /// menu closed this is a generation compare and two `bool` tests.
    ///
    /// * **HUD editor open:** a light 25 % dim first (over the world and the vanilla HUD, which
    ///   stay readable), then every enabled element with sample content where it has no data.
    /// * **F3 open:** mod elements are hidden; the debug columns cover the corners they live in
    ///   (research/HUD-EDITOR.md). The exception (M2) is the elements that
    ///   stand in for vanilla pieces vanilla draws under F3, the boss bar and the sidebar
    ///   (`HudElement::draws_under_debug`).
    pub(crate) fn draw_mod_hud(&mut self, list: &mut DrawList, ctx: &ScreenCtx) {
        self.mods.hud.sync(&self.mods.settings);
        let editing = self.mods.menu_live && self.mod_hud_editing();
        if editing {
            list.rect(0, 0, ctx.width, ctx.height, EDITOR_DIM_ARGB);
        }
        if !self.mods.hud.any_enabled() {
            return;
        }
        // Attribution only (a no-op unless `MC_PROFILE` is set).
        mc_core::profile_scope!("hud.mods");
        let under_debug = self.debug.visible && !editing;
        self.fill_hud_data();
        let hud_ctx = HudCtx {
            data: &self.mods.hud_data,
            settings: &self.mods.settings,
            screen: (ctx.width as f32, ctx.height as f32),
            pixel_scale: ctx.resolution.scale_factor() as f32,
            editing,
        };
        if under_debug {
            self.mods
                .hud
                .draw_under_debug(&hud_ctx, &mut self.font, list);
        } else {
            self.mods.hud.draw(&hud_ctx, &mut self.font, list);
        }
    }

    /// Whether the HUD editor is open (elements then draw samples and placeholders).
    pub(crate) fn mod_hud_editing(&self) -> bool {
        matches!(&self.ui, Ui::ModMenu(menu) if menu.hud_editing())
    }

    /// Refresh [`HudData`] in place. Only called while a HUD element is enabled.
    fn fill_hud_data(&mut self) {
        mc_core::profile_scope!("hud.mods.fill");
        // A benchmark's pinned clock (one relaxed load otherwise).
        let now_ms =
            mc_render::entity::item::animation_clock_ms(self.started.elapsed().as_millis() as u64);
        // Stacks are cloned only when they changed (NBT clones allocate).
        if self.held_item() != self.mods.hud_data.held.as_ref() {
            self.mods.hud_data.held = self.held_item().cloned();
        }
        let armor: [Option<&mc_protocol::ItemStack>; 4] =
            std::array::from_fn(|i| self.player_container.slot_stack(5 + i));
        let data = &mut self.mods.hud_data;
        data.now_ms = now_ms;
        data.tick = self.tick;
        data.fps = self.debug_fps.fps;
        match self.session.as_ref() {
            Some(session) => {
                // 1.7.10's player list is keyed by name (`app/ui_v5.rs`).
                data.ping_ms = super::ui_v5::session_latency_ms(session);
                if let Some(predictor) = session.predictor() {
                    let pos = predictor.state.pos;
                    data.position = [pos.x, pos.y, pos.z];
                    data.yaw = predictor.state.yaw;
                    data.pitch = predictor.state.pitch;
                }
            }
            None => data.ping_ms = None,
        }
        for (slot, stack) in data.armor.iter_mut().zip(armor) {
            if slot.as_ref() != stack {
                *slot = stack.cloned();
            }
        }
        data.effects.clear();
        data.effects
            .extend(self.active_effects.iter().map(|(&id, effect)| HudEffect {
                id: id as u8,
                amplifier: effect.amplifier as u8,
                duration_ticks: effect.duration,
                duration_max: effect.duration_max,
            }));
        data.effects.sort_unstable_by_key(|effect| effect.id);
        self.m1.publish(data);
        self.m2.fill(data, &self.hud, self.sidebar_cache.as_ref());
        // The Spotify widget (a `bool` while the mod is off)
        self.w9.fill_hud(&mut data.spotify);
    }
}

// ---- the mod menu (the menu key) --------------------------------------------------------------
//
// `Ui::ModMenu` holds the open screen. Everything below is called from one line each in
// `driver.rs`: the raw-key hook (open), `key_pressed` / `mouse_clicked` / `mouse_released` /
// `mouse_wheel` (input while open), `handle_screen_input` (the cursor) and `build_ui` (draw).
// While the menu is closed and not animating, every one of them is a `bool` test.
impl Client {
    /// The menu key (`ModSettings::menu_key`, Right Shift unless rebound in the menu's
    /// taskbar), in game: open the menu **at event time** and swallow the press.
    ///
    /// It used to be Shift+Tab, but Tab is the player
    /// list and Shift is sneak; now both reach the game untouched (no modifier is looked at).
    ///
    /// The shell folds each key edge into the key bindings right after this event, gated on
    /// `game_input_allowed()`. Opening here makes that gate false before the press can reach any
    /// binding on the same key (someone who binds the menu to their sneak key does not sneak);
    /// swallowing the press keeps the new menu from receiving it as its own close key. Closing is
    /// left to the frame key path ([`Client::mod_menu_key`]): that press was already gated by
    /// the open menu. Only in game: in chat or any other screen the key types or does nothing.
    /// Cost per raw key edge: a press/repeat test, then one scan of the LWJGL key table.
    fn mod_menu_raw_key(&mut self, code: KeyCode, pressed: bool, repeat: bool) -> bool {
        if !pressed || repeat {
            return false;
        }
        if mc_render::input::lwjgl_of_keycode(code) != Some(self.mods.settings.menu_key()) {
            return false;
        }
        if !self.mod_menu_can_open() {
            return false;
        }
        self.open_mod_menu(Instant::now());
        true
    }

    /// Where the menu key opens the menu: in game (no other screen), in a world.
    fn mod_menu_can_open(&self) -> bool {
        matches!(self.ui, Ui::InGame)
            && self.session.is_some()
            && !self.is_loading()
            && self.diagnostic_progress.is_none()
    }

    /// Open the menu; its opening animation starts at `now`. It goes through `show_screen`
    /// like every screen: bindings unpressed, world input blocked, the cursor released (and
    /// centred by the shared screen-open path).
    ///
    /// It opens where the last menu of this client run was (the same list
    /// and mod, the search cleared); the first open opens on My setup, or HUD when nothing is
    /// changed or pinned.
    pub(crate) fn open_mod_menu(&mut self, now: Instant) {
        let catalog = self.mods.settings.catalog();
        let screen = match self.mods.menu_place {
            Some(place) => ModMenuScreen::reopen(catalog, now, place),
            None => ModMenuScreen::new(catalog, now),
        };
        self.mods.closing_menu = None;
        self.mods.menu_live = true;
        self.show_screen(Some(Ui::ModMenu(Box::new(screen))));
    }

    /// Open the menu over the screen that is up now (a "Mods" button on the title screen or
    /// the pause menu): closing the menu returns to that screen, not to the game. Without a
    /// world the menu greys out "Edit HUD" and "Move on screen" (see `MenuGame::no_world`).
    /// The screen behind is not drawn while the menu is up.
    #[allow(dead_code)] // the "Mods" buttons call it
    pub(crate) fn open_mod_menu_over(&mut self, now: Instant) {
        let back = std::mem::replace(&mut self.ui, Ui::InGame);
        if !matches!(back, Ui::InGame | Ui::ModMenu(_)) {
            self.mods.menu_return = Some(Box::new(back));
        }
        self.open_mod_menu(now);
    }

    /// Close the menu: the game (or the screen it was opened over) gets input and the cursor
    /// back at once, while the windows play their (faster) closing animation from
    /// `closing_menu`.
    pub(crate) fn close_mod_menu(&mut self, now: Instant) {
        if !matches!(self.ui, Ui::ModMenu(_)) {
            return;
        }
        if let Ui::ModMenu(mut screen) = std::mem::replace(&mut self.ui, Ui::InGame) {
            screen.begin_close(now);
            self.mods.menu_place = Some(screen.place());
            self.mods.closing_menu = Some(screen);
        }
        // `show_screen(None)` still picks the death screen or the main menu when those apply.
        let back = self.mods.menu_return.take().map(|back| *back);
        self.show_screen(back);
    }

    /// What the menu's input methods may change, for an input event at `now`.
    fn menu_input(
        mods: &mut ModRuntime,
        now: Instant,
        keys: mc_ui::widget::keys::Modifiers,
    ) -> MenuInput<'_> {
        let (gui_screen, gui_scale) = mods.menu_gui;
        MenuInput {
            settings: &mut mods.settings,
            hud: &mut mods.hud,
            events: &mut mods.events,
            gui_screen,
            gui_scale,
            now,
            alt: keys.alt,
            shift: keys.shift,
        }
    }

    /// A key press while the menu is open (from `key_pressed`, after the fullscreen and
    /// screenshot bindings). The open menu takes every key; returns `false` when it is not open.
    pub(crate) fn mod_menu_key(&mut self, press: &crate::app::input::KeyPress) -> bool {
        let Ui::ModMenu(screen) = &mut self.ui else {
            return false;
        };
        // Holding the menu key (or Esc) auto-repeats it; the menu (it accepts repeats while a
        // search is typed, for Backspace and the arrows) must not take a repeat as a close key:
        // a long Esc clears the search and stops there.
        let closing_key = press.lwjgl == self.mods.settings.menu_key()
            || press.lwjgl == mc_ui::widget::keys::ESCAPE;
        if press.repeat && closing_key && !screen.capturing_menu_key() {
            return true;
        }
        let now = Instant::now();
        let mut input = Self::menu_input(&mut self.mods, now, press.mods);
        if screen.clipboard_shortcut(press.lwjgl, press.mods, &mut self.clipboard, &mut input) {
            return true;
        }
        if screen.key(press.lwjgl, press.typed, &mut input) == KeyOutcome::Close {
            self.close_mod_menu(now);
        }
        true
    }

    /// A mouse button edge while the menu is open (from `mouse_clicked` / `mouse_released`).
    /// Returns `false` when the menu is not open.
    pub(crate) fn mod_menu_mouse(&mut self, button: i32, pressed: bool) -> bool {
        let Ui::ModMenu(screen) = &mut self.ui else {
            return false;
        };
        let Ok(button) = u8::try_from(button) else {
            return true;
        };
        let (x, y) = self.mods.menu_cursor;
        let keys = self.keyboard.modifiers();
        let mut input = Self::menu_input(&mut self.mods, Instant::now(), keys);
        if pressed {
            screen.mouse_pressed(x, y, button, &mut input);
        } else {
            screen.mouse_released(x, y, button, &mut input);
        }
        true
    }

    /// The wheel while the menu is open (from `mouse_wheel`). Returns `false` when it is not.
    pub(crate) fn mod_menu_wheel(&mut self, delta: f32) -> bool {
        let Ui::ModMenu(screen) = &mut self.ui else {
            return false;
        };
        let (x, y) = self.mods.menu_cursor;
        let keys = self.keyboard.modifiers();
        let mut input = Self::menu_input(&mut self.mods, Instant::now(), keys);
        screen.scrolled(x, y, delta, &mut input);
        true
    }

    /// This frame's cursor (physical pixels) and GUI geometry, from `handle_screen_input`
    /// before any of the frame's clicks. While the menu is open this also drives its drags
    /// (sliders, colour sliders, the HUD editor), so a drag preview is never a frame late.
    pub(crate) fn mod_menu_pointer(&mut self, cursor: (f64, f64), resolution: ScaledResolution) {
        if !self.mods.menu_live {
            return;
        }
        self.mods.menu_cursor = (cursor.0 as f32, cursor.1 as f32);
        self.mods.menu_gui = (
            (
                resolution.scaled_width() as f32,
                resolution.scaled_height() as f32,
            ),
            resolution.scale_factor().max(1) as f32,
        );
        let Ui::ModMenu(screen) = &mut self.ui else {
            return;
        };
        let (x, y) = self.mods.menu_cursor;
        let keys = self.keyboard.modifiers();
        let mut input = Self::menu_input(&mut self.mods, Instant::now(), keys);
        screen.pointer_moved(x, y, &mut input);
    }

    /// Lay the menu out for this frame (the open one, or the one animating out) into `list`,
    /// in physical pixels. `None` once nothing is left to draw. Also refreshes the expanded
    /// mod's dynamic rows. Separate from [`Client::build_mod_menu`] so tests can lay the menu
    /// out (and record its hit regions) without a GPU.
    pub(super) fn layout_mod_menu(
        &mut self,
        size: (u32, u32),
        list: &mut DrawList,
        now: Instant,
    ) -> Option<GlassFrame> {
        if self
            .mods
            .closing_menu
            .as_ref()
            .is_some_and(|menu| menu.finished(now))
        {
            self.mods.closing_menu = None;
        }
        let (expanded, search) = match &self.ui {
            Ui::ModMenu(screen) if screen.searching() => (screen.expanded(), screen.search_query()),
            Ui::ModMenu(screen) => (screen.expanded(), ""),
            _ if self.mods.closing_menu.is_some() => (None, ""),
            _ => return None,
        };
        // The search, for the row hooks (copied only when it changes).
        // When a search ends, rows filled for it are dropped (the expanded mod refills below).
        if self.mods.menu_search != search {
            self.mods.menu_search.clear();
            self.mods.menu_search.push_str(search);
            if search.is_empty() {
                for rows in &mut self.mods.rows {
                    rows.begin();
                }
            }
        }
        // The lanes' dynamic rows for the expanded mod: built only while it is on screen.
        self.mods.menu_expanded = expanded;
        self.fill_menu_rows();
        let gui_scale = self.mods.menu_gui.1;
        // Wrapped before the f32 conversion, like the title screen's clock: it
        // used to lose precision with uptime (8 ms steps after a day, whole seconds after 194).
        let time_s = crate::app::stability::cycle_seconds(
            now.saturating_duration_since(self.mods.menu_epoch),
            crate::app::stability::UI_ANIMATION_CYCLE,
        );
        let text = self.mods.menu_text.get_or_insert_with(MenuText::load);
        let screen: &mut ModMenuScreen = match &mut self.ui {
            Ui::ModMenu(screen) => screen,
            _ => self.mods.closing_menu.as_mut()?,
        };
        let mut env = MenuEnv {
            text,
            mc_font: &mut self.font,
            settings: &self.mods.settings,
            rows: &self.mods.rows,
            hud: &self.mods.hud,
            screen: (size.0 as f32, size.1 as f32),
            gui_scale,
            now,
            time_s: time_s % 3600.0,
            // No world: Edit HUD and Move on screen grey out.
            game: MenuGame {
                no_world: self.session.is_none(),
                vanilla_keys: Some(&self.settings.keys),
                // Window 3's live warnings read options.txt; profiles live
                // next to mc-rs-mods.json.
                options: Some(&self.settings),
                game_dir: self.mods.path.parent(),
            },
        };
        Some(screen.build(&mut env, list))
    }

    /// Draw the menu into its render phase (end of `build_ui`). One `bool` test while the menu
    /// is closed and done animating; the frame it finishes, the phase is switched off, which
    /// returns the frame path to exactly what it is without the menu.
    pub(crate) fn build_mod_menu(&mut self, gpu: &mc_render::Gpu) {
        if !self.mods.menu_live {
            return;
        }
        let Some(phase) = self.mods.menu_phase.clone() else {
            // No world renderer (yet): nothing to draw, and nothing left to animate once closed.
            if !matches!(self.ui, Ui::ModMenu(_)) {
                self.mods.closing_menu = None;
                self.mods.menu_live = false;
            }
            return;
        };
        let mut list = phase.with(ModMenuPhase::take_spare_list);
        let size = (gpu.config.width, gpu.config.height);
        let Some(glass) = self.layout_mod_menu(size, &mut list, Instant::now()) else {
            phase.with(ModMenuPhase::deactivate);
            self.mods.menu_live = false;
            self.mods.menu_expanded = None;
            return;
        };
        let stale = std::mem::take(&mut self.mods.menu_atlas_stale);
        phase.with(|phase| {
            if let Some(text) = self.mods.menu_text.as_mut() {
                if stale {
                    text.mark_dirty();
                }
                if let Some(pixels) = text.take_upload() {
                    phase.upload_atlas(gpu, pixels, menu_text::ATLAS_SIZE);
                }
                if !text.is_segoe() {
                    // The Minecraft-font fallback draws with the vanilla font pages.
                    let _ = phase
                        .renderer
                        .ensure_list_textures(gpu, &self.assets.resources, &list);
                }
            }
            phase.submit(list, glass);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mc-rs-modrt-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn nothing_is_written_until_something_changes() {
        let dir = temp_dir("quiet");
        let mut runtime = ModRuntime::load(&dir);
        let start = Instant::now();
        runtime.drive_persistence(start + Duration::from_secs(10));
        runtime.flush();
        assert!(
            !dir.join(mc_mods::SETTINGS_FILE_NAME).exists(),
            "an untouched store must not create the settings file"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn saves_are_debounced_then_written() {
        let dir = temp_dir("debounce");
        let path = dir.join(mc_mods::SETTINGS_FILE_NAME);
        let mut runtime = ModRuntime::load(&dir);
        let t0 = Instant::now();
        runtime.settings.set_enabled("fps", true);
        runtime.drive_persistence(t0);
        assert!(!path.exists(), "not written on the frame of the change");
        runtime.drive_persistence(t0 + Duration::from_millis(100));
        runtime.wait_for_writes();
        assert!(!path.exists());
        runtime.drive_persistence(t0 + SAVE_DEBOUNCE + Duration::from_millis(1));
        // Queued for the writer thread, not written inline on this (the main) thread.
        runtime.wait_for_writes();
        assert!(path.exists(), "written once the store has been quiet");
        assert!(!runtime.has_unsaved_changes());
        let reloaded = ModRuntime::load(&dir);
        assert!(reloaded.settings.enabled("fps"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_long_drag_is_still_saved_by_the_max_delay() {
        let dir = temp_dir("drag");
        let path = dir.join(mc_mods::SETTINGS_FILE_NAME);
        let mut runtime = ModRuntime::load(&dir);
        let t0 = Instant::now();
        let mut t = t0;
        let mut on = false;
        while t < t0 + SAVE_MAX_DELAY + Duration::from_millis(200) {
            on = !on;
            runtime.settings.set_enabled("fps", on);
            runtime.drive_persistence(t);
            t += Duration::from_millis(100);
        }
        runtime.wait_for_writes();
        assert!(
            path.exists(),
            "continuous changes must not postpone the save forever"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_newest_snapshot_wins_and_flush_waits_for_it() {
        let dir = temp_dir("coalesce");
        let mut runtime = ModRuntime::load(&dir);
        runtime.settings.set_enabled("fps", true);
        runtime.save_now();
        runtime.settings.set_enabled("fps", false);
        runtime.settings.set_enabled("ping", true);
        // Shutdown path: queue the latest and block until it is on disk.
        runtime.flush();
        let reloaded = ModRuntime::load(&dir);
        assert!(!reloaded.settings.enabled("fps"));
        assert!(reloaded.settings.enabled("ping"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A write that never finishes (a disk that stopped answering) no
    /// longer hangs the exit: dropping the writer returns after the bounded wait while the stuck
    /// write is still going. The old `Drop` joined that thread and never returned.
    #[test]
    fn a_stuck_settings_write_does_not_hang_the_exit() {
        let shared: std::sync::Arc<(std::sync::Mutex<WriterState>, std::sync::Condvar)> =
            std::sync::Arc::default();
        shared.0.lock().expect("state").writing = true;
        let still_writing = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let (release, released) = std::sync::mpsc::channel::<()>();
        let stuck = {
            let still_writing = std::sync::Arc::clone(&still_writing);
            std::thread::spawn(move || {
                let _ = released.recv();
                still_writing.store(false, std::sync::atomic::Ordering::SeqCst);
            })
        };
        let writer = SettingsWriter {
            shared,
            thread: Some(stuck),
        };
        let (done, dropped) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            drop(writer);
            let _ = done.send(());
        });
        let returned = dropped.recv_timeout(WRITER_EXIT_WAIT + Duration::from_secs(20));
        let write_was_still_going = still_writing.load(std::sync::atomic::Ordering::SeqCst);
        let _ = release.send(());
        assert!(
            returned.is_ok(),
            "dropping the writer waited for the stuck write"
        );
        assert!(
            write_was_still_going,
            "the drop returned without joining the stuck write"
        );
    }

    /// A file with a byte that is not UTF-8 keeps its settings, and is
    /// still copied aside before the next save writes the replacement character.
    #[test]
    fn a_file_with_a_foreign_byte_keeps_its_settings_and_a_copy() {
        let dir = temp_dir("foreign-byte");
        let path = dir.join(mc_mods::SETTINGS_FILE_NAME);
        let mut settings = ModSettings::new();
        settings.set_enabled("fps", true);
        let text = settings.to_json_string();
        let open = text.find('{').expect("a JSON object");
        let mut bytes = text.as_bytes()[..=open].to_vec();
        bytes.extend_from_slice(b"\"note\":\"\xFF\",");
        bytes.extend_from_slice(&text.as_bytes()[open + 1..]);
        std::fs::write(&path, &bytes).unwrap();
        let runtime = ModRuntime::load(&dir);
        assert!(runtime.settings.enabled("fps"), "the settings survived");
        let mut bad = path.as_os_str().to_owned();
        bad.push(".bad");
        assert_eq!(std::fs::read(PathBuf::from(bad)).unwrap(), bytes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_file_is_kept_aside() {
        let dir = temp_dir("corrupt");
        let path = dir.join(mc_mods::SETTINGS_FILE_NAME);
        std::fs::write(&path, "{ this is not json").unwrap();
        let runtime = ModRuntime::load(&dir);
        assert!(!runtime.settings.enabled("fps"));
        let mut bad = path.as_os_str().to_owned();
        bad.push(".bad");
        assert!(PathBuf::from(bad).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The Shift+Tab menu through the real client: key edges exactly as the shell delivers them,
/// clicks through `handle_screen_input`, the animation clock, and the settings file. Needs the
/// vanilla jar (skips loudly without it; set `MC_JARS_DIR`), no GPU.
#[cfg(test)]
mod menu_tests {
    use std::time::{Duration, Instant};

    use mc_render::input::keys::{FORWARD, PLAYER_LIST};
    use mc_render::{InputState, KeyBindings, ShellApp as _};
    use mc_ui::widget::DrawList;
    use winit::event::MouseButton;
    use winit::keyboard::KeyCode;

    use super::super::{Client, Ui};
    use super::ModRuntime;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::session::{JoinSettings, Session};

    const SIZE: (u32, u32) = (1280, 720);

    struct Harness {
        client: Client,
        dir: tempfile::TempDir,
        // Keep the fake connection's halves alive for the session.
        _net: (
            tokio::sync::mpsc::Sender<crate::net::ToGame>,
            tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
        ),
    }

    /// An offline client in game, in an (empty) world, with its game dir in a temp folder.
    fn harness() -> Option<Harness> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        client.session = Some(Session::new(handle, JoinSettings::default()));
        client.ui = Ui::InGame;
        Some(Harness {
            client,
            dir,
            _net: (to_game, to_net),
        })
    }

    /// One keyboard edge as the shell delivers it: the client's raw-key hook first, then the
    /// edge is folded into the key bindings only if `game_input_allowed()` allows it right then.
    fn shell_key(client: &mut Client, keys: &mut KeyBindings, code: KeyCode, pressed: bool) {
        client.handle_raw_key(code, pressed, false, None);
        let allowed = client.game_input_allowed();
        keys.set_state_when_input_allowed(mc_render::Input::Key(code), pressed, allowed);
    }

    /// The screen half of a frame: queued keys and mouse events reach the UI.
    fn frame(client: &mut Client, input: &mut InputState) {
        client.handle_screen_input(input, SIZE.0, SIZE.1);
    }

    fn open_with_menu_key(client: &mut Client, keys: &mut KeyBindings) {
        shell_key(client, keys, KeyCode::ShiftRight, true);
        shell_key(client, keys, KeyCode::ShiftRight, false);
    }

    /// Exercise the real routing without reading or changing the user's OS clipboard. The
    /// complete copy/paste sequence is covered with MemoryClipboard by mc-ui's menu tests.
    #[test]
    fn search_clipboard_shortcuts_reach_the_menu_before_typed_characters() {
        use crate::app::input::KeyPress;
        use mc_ui::widget::keys::{self, Modifiers};

        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        client.open_mod_menu(Instant::now());
        let press = |client: &mut Client, code, lwjgl, typed, mods| {
            assert!(client.mod_menu_key(&KeyPress {
                sequence: 0,
                code,
                lwjgl,
                typed,
                mods,
                repeat: false,
            }));
        };
        let query = |client: &Client| match &client.ui {
            Ui::ModMenu(screen) => screen.search_query().to_owned(),
            _ => panic!("the menu remains open"),
        };
        press(client, KeyCode::KeyZ, 44, Some('z'), Modifiers::NONE);
        press(client, KeyCode::KeyA, keys::A, Some('a'), Modifiers::CTRL);
        assert_eq!(query(client), "z", "Ctrl+A must not append A");
        press(client, KeyCode::KeyF, 33, Some('f'), Modifiers::NONE);
        assert_eq!(query(client), "f", "typing replaces the selected query");
        press(client, KeyCode::KeyC, keys::C, Some('c'), Modifiers::CTRL);
        assert_eq!(
            query(client),
            "f",
            "Ctrl+C without a selection does nothing"
        );
        press(client, KeyCode::KeyA, keys::A, Some('a'), Modifiers::CTRL);
        press(
            client,
            KeyCode::Backspace,
            keys::BACKSPACE,
            None,
            Modifiers::NONE,
        );
        assert_eq!(query(client), "", "Backspace removes the selected query");
    }

    /// Shift+Tab reaches the game (Shift sneaks, Tab shows
    /// the player list) and opens nothing; Right Shift opens the menu at event time, its OS
    /// repeats never close it, Tab does nothing in it, and Right Shift closes it again.
    #[test]
    fn right_shift_opens_and_closes_the_menu_and_shift_tab_reaches_the_game() {
        use mc_render::input::keys::SNEAK;
        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        let mut keys = client.settings.keys.clone();
        assert_eq!(client.mods.settings.menu_key(), 54, "Right Shift");
        // Shift+Tab: sneak and the player list, as in vanilla; no menu.
        shell_key(client, &mut keys, KeyCode::ShiftLeft, true);
        shell_key(client, &mut keys, KeyCode::Tab, true);
        assert!(
            matches!(client.ui, Ui::InGame),
            "Shift+Tab no longer opens the menu: {}",
            client.ui.name()
        );
        assert!(keys.is_down(SNEAK), "Shift reached sneak");
        assert!(keys.is_down(PLAYER_LIST), "Tab reached key.playerlist");
        client.handle_game_input(&InputState::new(), &mut keys);
        assert!(client.tab_list_held, "the player list shows");
        shell_key(client, &mut keys, KeyCode::Tab, false);
        shell_key(client, &mut keys, KeyCode::ShiftLeft, false);
        client.handle_game_input(&InputState::new(), &mut keys);

        // W held in game reaches the bindings.
        shell_key(client, &mut keys, KeyCode::KeyW, true);
        assert!(keys.is_down(FORWARD));
        shell_key(client, &mut keys, KeyCode::ShiftRight, true);
        assert!(
            matches!(client.ui, Ui::ModMenu(_)),
            "Right Shift opens at event time"
        );
        assert!(client.ui.wants_cursor(), "the cursor is released");
        assert!(!client.game_input_allowed(), "world input is blocked");
        // Opening unpresses every binding, as any screen does (drained by the next tick).
        client.handle_game_input(&InputState::new(), &mut keys);
        assert!(!keys.is_down(FORWARD), "W is released behind the menu");
        // Still held: the OS repeats the key; the menu must not take that as its close key.
        client.handle_raw_key(KeyCode::ShiftRight, true, true, None);
        client.handle_raw_key(KeyCode::ShiftRight, true, true, None);
        frame(client, &mut InputState::new());
        assert!(matches!(client.ui, Ui::ModMenu(_)), "repeats are ignored");
        shell_key(client, &mut keys, KeyCode::ShiftRight, false);
        shell_key(client, &mut keys, KeyCode::KeyW, false);
        // The swallowed opening press is not delivered to the menu as a close.
        frame(client, &mut InputState::new());
        assert!(matches!(client.ui, Ui::ModMenu(_)));
        // Tab no longer closes the menu.
        shell_key(client, &mut keys, KeyCode::Tab, true);
        shell_key(client, &mut keys, KeyCode::Tab, false);
        frame(client, &mut InputState::new());
        assert!(matches!(client.ui, Ui::ModMenu(_)), "Tab is not a menu key");

        // Right Shift again closes; that press is gated by the still-open menu.
        shell_key(client, &mut keys, KeyCode::ShiftRight, true);
        frame(client, &mut InputState::new());
        assert!(matches!(client.ui, Ui::InGame), "got {}", client.ui.name());
        assert!(!client.ui.wants_cursor(), "the cursor is recaptured");
        shell_key(client, &mut keys, KeyCode::ShiftRight, false);
    }

    /// The menu key is the user's: rebound to M it opens and closes the menu and Right Shift is
    /// free again; over any other screen (pause here; chat the same) it does nothing to the
    /// menu, so it can be typed.
    #[test]
    fn a_rebound_menu_key_opens_and_closes_the_menu_only_in_game() {
        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        let mut keys = client.settings.keys.clone();
        client.mods.settings.set_menu_key(50); // LWJGL 50 = M
        open_with_menu_key(client, &mut keys);
        assert!(matches!(client.ui, Ui::InGame), "Right Shift is free again");
        shell_key(client, &mut keys, KeyCode::KeyM, true);
        assert!(matches!(client.ui, Ui::ModMenu(_)), "M opens");
        shell_key(client, &mut keys, KeyCode::KeyM, false);
        frame(client, &mut InputState::new());
        assert!(matches!(client.ui, Ui::ModMenu(_)));
        shell_key(client, &mut keys, KeyCode::KeyM, true);
        frame(client, &mut InputState::new());
        assert!(
            matches!(client.ui, Ui::InGame),
            "M closes: {}",
            client.ui.name()
        );
        shell_key(client, &mut keys, KeyCode::KeyM, false);
        shell_key(client, &mut keys, KeyCode::Escape, true);
        frame(client, &mut InputState::new());
        assert!(matches!(client.ui, Ui::Pause(_)));
        shell_key(client, &mut keys, KeyCode::Escape, false);
        shell_key(client, &mut keys, KeyCode::KeyM, true);
        assert!(matches!(client.ui, Ui::Pause(_)), "{}", client.ui.name());
        shell_key(client, &mut keys, KeyCode::KeyM, false);
    }

    #[test]
    fn esc_closes_the_menu_to_the_game_and_the_menu_key_needs_the_game() {
        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        let mut keys = client.settings.keys.clone();
        open_with_menu_key(client, &mut keys);
        assert!(matches!(client.ui, Ui::ModMenu(_)));
        shell_key(client, &mut keys, KeyCode::Escape, true);
        frame(client, &mut InputState::new());
        assert!(
            matches!(client.ui, Ui::InGame),
            "Esc returns to the game, not to the pause screen: {}",
            client.ui.name()
        );
        shell_key(client, &mut keys, KeyCode::Escape, false);
        // Esc in game is vanilla again, and the menu key over another screen does nothing.
        shell_key(client, &mut keys, KeyCode::Escape, true);
        frame(client, &mut InputState::new());
        assert!(matches!(client.ui, Ui::Pause(_)));
        shell_key(client, &mut keys, KeyCode::Escape, false);
        open_with_menu_key(client, &mut keys);
        assert!(matches!(client.ui, Ui::Pause(_)), "{}", client.ui.name());
    }

    #[test]
    fn a_click_on_a_switch_toggles_the_mod_and_the_writer_saves_it() {
        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        let t0 = Instant::now();
        client.open_mod_menu(t0);
        // The first frame samples the cursor and the GUI geometry.
        frame(client, &mut InputState::new());
        // Lay out (records hit regions) once the opening animation is over.
        client
            .layout_mod_menu(SIZE, &mut DrawList::new(), t0 + Duration::from_secs(1))
            .expect("the menu is on screen");
        let fps = client.mods.settings.index_of("fps").unwrap();
        let Ui::ModMenu(menu) = &client.ui else {
            panic!("menu");
        };
        let (x, y) = menu
            .switch_rect(fps)
            .expect("the FPS switch is drawn")
            .center();
        assert!(!client.mods.settings.enabled("fps"));

        let mut input = InputState::new();
        input.on_cursor_moved(f64::from(x), f64::from(y));
        input.on_mouse_button_at_ms_with_sequence(MouseButton::Left, true, 0, 0);
        input.on_mouse_button_at_ms_with_sequence(MouseButton::Left, false, 1, 1);
        frame(client, &mut input);
        assert!(
            client.mods.settings.enabled("fps"),
            "the click turned FPS on"
        );
        assert!(
            client.mods.has_unsaved_changes(),
            "saved later, off the main thread"
        );

        // Shutdown: the writer thread is flushed and waited for.
        client.flush_mod_settings();
        assert!(!client.mods.has_unsaved_changes());
        let back = ModRuntime::load(h.dir.path());
        assert!(back.settings.enabled("fps"), "the file holds the click");
    }

    #[test]
    fn the_windows_animate_in_and_out_on_the_real_clock() {
        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        let t0 = Instant::now();
        client.open_mod_menu(t0);
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let start = client
            .layout_mod_menu(SIZE, &mut DrawList::new(), t0)
            .expect("opening");
        assert!(start.dim < 0.01, "the dim starts at zero: {}", start.dim);
        let open = client
            .layout_mod_menu(SIZE, &mut DrawList::new(), at(500))
            .expect("open");
        assert_eq!(
            open.rects.len(),
            4,
            "three windows (Where, What, How) and the taskbar's Edit HUD pane"
        );
        assert!((open.dim - 0.35).abs() < 1e-4);

        client.close_mod_menu(at(600));
        assert!(matches!(client.ui, Ui::InGame), "input returns at once");
        let closing = client
            .layout_mod_menu(SIZE, &mut DrawList::new(), at(640))
            .expect("still animating out");
        assert!(closing.dim > 0.0 && closing.dim < 0.35, "{}", closing.dim);
        assert!(
            client
                .layout_mod_menu(SIZE, &mut DrawList::new(), at(1000))
                .is_none(),
            "gone once the close animation ends"
        );
        assert!(client.mods.closing_menu.is_none());
    }

    /// The real render stack: `Client::install` registers the menu phase; it stays off (the
    /// frame on its direct path) while the menu is closed, and draws glass and text over the
    /// client's own frame while it is open. `MC_MODMENU_CLIENT_PNG=<path>` saves that frame.
    #[test]
    fn the_menu_draws_through_the_client_render_stack() {
        let Some(mut h) = harness() else { return };
        let mut gpu = match mc_render::Gpu::headless(
            SIZE.0,
            SIZE.1,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                return;
            }
            Err(error) => panic!("headless GPU: {error}"),
        };
        let client = &mut h.client;
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);
        assert!(
            phases.names().any(|name| name == "modmenu"),
            "the menu phase is registered"
        );
        client.build_ui(&gpu, 0.0);
        assert!(
            !phases.scene_mode(),
            "closed: the frame stays on its direct path"
        );

        // Opened a second ago, so the opening animation is over.
        client.open_mod_menu(Instant::now() - Duration::from_secs(1));
        frame(client, &mut InputState::new());
        client.build_ui(&gpu, 0.0);
        assert!(
            phases.scene_mode(),
            "open: the glass reads the frame behind it"
        );
        let glass = client
            .mods
            .menu_phase
            .as_ref()
            .expect("installed")
            .with(|phase| phase.glass().clone());
        assert_eq!(
            glass.rects.len(),
            4,
            "three windows (Where, What, How) and the taskbar's Edit HUD pane"
        );
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &mc_render::Camera::default(), 0.0);
        let pixels = gpu.read_pixels().expect("readback");
        if let Ok(path) = std::env::var("MC_MODMENU_CLIENT_PNG") {
            image::save_buffer(
                &path,
                &pixels,
                SIZE.0,
                SIZE.1,
                image::ExtendedColorType::Rgba8,
            )
            .expect("write the PNG");
        }
        // The first window's title, near-white text, is in its top-left corner.
        let [x, y, _, _] = glass.rects[0].rect;
        let mut brightest = 0u8;
        for py in (y as u32 + 8)..(y as u32 + 48) {
            for px in (x as u32 + 8)..(x as u32 + 128) {
                let i = ((py * SIZE.0 + px) * 4) as usize;
                brightest = brightest.max(pixels[i].min(pixels[i + 1]).min(pixels[i + 2]));
            }
        }
        assert!(
            brightest > 200,
            "the title text reached the frame: {brightest}"
        );

        // Closed long enough ago that the closing animation is over: the phase switches off.
        client.close_mod_menu(Instant::now() - Duration::from_secs(1));
        client.build_ui(&gpu, 0.0);
        assert!(
            !phases.scene_mode(),
            "closed again: back on the direct path"
        );
        assert!(!client.mods.menu_live);
    }

    /// Closing the menu keeps its place (list and mod) for this
    /// run, and the next open starts there.
    #[test]
    fn the_menu_reopens_where_the_last_one_was() {
        use mc_ui::screens::modmenu::{MenuPlace, View};
        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        let t0 = Instant::now();
        client.open_mod_menu(t0);
        client
            .layout_mod_menu(SIZE, &mut DrawList::new(), t0 + Duration::from_millis(500))
            .expect("open");
        let Ui::ModMenu(menu) = &client.ui else {
            panic!("menu");
        };
        let first = menu.place();
        assert_eq!(
            first.view,
            View::Category(0),
            "a fresh setup opens on HUD (My setup is empty)"
        );
        client.close_mod_menu(t0 + Duration::from_millis(600));
        assert_eq!(client.mods.menu_place, Some(first), "kept in memory");
        // Somewhere else (as clicks in the last menu would leave it): the next open starts there.
        let zoom = client.mods.settings.index_of("zoom").unwrap();
        let place = MenuPlace {
            view: View::AllMods,
            selected: Some(zoom),
        };
        client.mods.menu_place = Some(place);
        client.open_mod_menu(t0 + Duration::from_secs(2));
        let Ui::ModMenu(menu) = &client.ui else {
            panic!("menu");
        };
        assert_eq!(menu.place(), place);
    }

    /// A "Mods" button on another screen (the title screen, the
    /// pause menu) opens the menu over it; closing returns to that screen. Without a world,
    /// "Edit HUD" is greyed out (no click target).
    #[test]
    fn the_menu_opens_over_a_screen_without_a_world_and_closing_returns_to_it() {
        let Some(mut h) = harness() else { return };
        let client = &mut h.client;
        client.session = None;
        client.ui = Ui::Options(Box::new(mc_ui::screens::options::OptionsScreen::new(false)));
        let t0 = Instant::now();
        client.open_mod_menu_over(t0);
        assert!(matches!(client.ui, Ui::ModMenu(_)), "{}", client.ui.name());
        client
            .layout_mod_menu(SIZE, &mut DrawList::new(), t0 + Duration::from_secs(1))
            .expect("on screen");
        let Ui::ModMenu(menu) = &client.ui else {
            panic!("menu");
        };
        assert!(
            menu.edit_hud_rect().is_none(),
            "no world: Edit HUD is greyed out"
        );
        client.close_mod_menu(t0 + Duration::from_secs(2));
        assert!(
            matches!(client.ui, Ui::Options(_)),
            "back to the screen it opened over: {}",
            client.ui.name()
        );
        assert!(client.mods.menu_return.is_none());
    }
}
