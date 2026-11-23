//! The Aero title screen's client half (`aero-title-2026-11-07`): the `aero_title` mod,
//! on by default.
//!
//! While the mod is on and the title screen (`Ui::MainMenu`) is up, the client draws
//! [`mc_ui::screens::aero_title`] instead of vanilla's `MainMenuScreen`, over the block
//! greenhouse (`mc_render::menu_scene`). Vanilla's screen object stays in `Ui::MainMenu`
//! untouched, so every path that opens, leaves or returns to the title screen is vanilla's, and
//! turning the mod off brings vanilla's title screen back exactly.
//!
//! # Hooks in `driver.rs` (one line each)
//! * `fields` / `init`: [`AeroTitleState`] in `Client::aero`.
//! * `install`: [`Client::aero_title_install_phases`] registers four phase slots.
//! * `draw`: [`Client::build_aero_title`] at the top of `build_ui`'s screen `match`; when it
//!   draws, the `Ui::MainMenu` arm draws nothing.
//! * `pointer`, `click`, `release`: [`Client::aero_title_pointer`] and
//!   [`Client::aero_title_mouse`] in `handle_screen_input`, `mouse_clicked` and `mouse_released`.
//! * `frame-cap`: [`Client::aero_title_frame_cap`] in `ShellApp::effective_frame_cap`.
//! * `monitor`: `ShellApp::monitor_refresh_changed` stores the shell's monitor refresh rate.
//!
//! # Phases
//! `install` registers four [`PhaseSlot`]s: the greenhouse behind everything that draws on the
//! title screen ([`BACKGROUND_ORDER`]), the title UI's own glass phase ([`UI_ORDER`], after the UI
//! phase and before the Shift+Tab menu's), the greenhouse's VHS and glow finish over the whole
//! frame, UI included ([`FINISH_ORDER`]), and a crisp copy of the buttons' marks over the finish
//! ([`CRISP_ORDER`]): measured, the full finish halved the labels' contrast, and this gives the
//! small text the lighter version of it that the plan allows. A slot is empty, and disabled,
//! unless the title screen is up; it is filled when the title screen appears and emptied the frame
//! it is left. The greenhouse itself is built once per session and parked while the title screen
//! is away (rebuilding it per visit grew the GPU driver's memory, the leak follow-up); it
//! gives its big GPU memory back (about 200 MB at 1080p) once the title screen has been gone for
//! [`PARKED_RELEASE_AFTER`].
//!
//! While the title screen is up the world passes the client records anyway (the sky dome, the
//! clouds, the screen-space fog) are switched off: the greenhouse covers every pixel.
//!
//! # Three scenes
//! The background is one of three scenes ([`Backdrop`]): the greenhouse as described here, or the earlier Glass or Shore (`mc_render::loading_scene`), picked by the mod's "Menu scene"
//! option (Random by default: decided once per launch). Glass and Shore fill the background slot
//! only: they have no VHS finish, so the crisp layer is not used with them. Everything else above
//! (the slots, parking, the release after a minute away, the frame cap) is the same for all three.
//! The choice, the launch's pick and the Glass/Shore wrapper are in `nw_menu.rs`.
//!
//! # Cost
//! Mod off, or any screen but the title screen: `build_aero_title` is a `bool`, an enum and an
//! `Option` test (plus, while a world is loaded and the greenhouse was never prewarmed, two more
//! field tests), the input hooks an `Option` test, the frame-cap hook an enum test, and the three
//! slots are three disabled phases (one `RefCell` borrow and an `Option` test each). Nothing
//! allocates, locks, hashes or formats. The vanilla title screen (mod off) is drawn by the same
//! code as before, at vanilla's 30 FPS.
//!
//! On the title screen: the greenhouse's own cost (the report), the title UI's glass pass
//! (the Shift+Tab menu's shader over the whole frame plus a six-level mip chain), a few hundred
//! UI triangles, and nothing per frame on the CPU beyond building them. The title bitmap is
//! rasterised once per window size; the greenhouse is baked once per session (on a background
//! thread from the first frame), its GPU side is built when the title screen first appears and
//! reused on every later visit. While it is parked: one `Option` test per frame, plus a clock
//! read until its memory is released.

use std::time::{Duration, Instant};

use mc_render::menu_scene::{self, MenuScene};
use mc_render::{Camera, Gpu, PhaseList, PhaseOrder, RenderPhase, SceneTextures};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::aero_title::font::{LogoStyle, TitleFont};
use mc_ui::screens::aero_title::phase::AeroTitlePhase;
use mc_ui::screens::aero_title::{self, AeroTitleEnv, AeroTitleScreen};
use mc_ui::screens::modmenu::glass::GlassFrame;
use mc_ui::screens::modmenu::text::MenuText;
use mc_ui::widget::DrawList;

use super::super::{Client, Shared, Ui};
use super::nw_menu::{self, MenuKind, SceneBackdrop};

/// Where the greenhouse goes (the own constant): after every world phase the client still
/// records on the title screen (the sky dome and clouds draw even without a world), so nothing
/// paints over it, and before the UI.
pub(crate) const BACKGROUND_ORDER: PhaseOrder = menu_scene::BACKGROUND_ORDER;
/// The title UI: after the UI phase, before the Shift+Tab menu's glass (`UI + 50`).
pub(crate) const UI_ORDER: PhaseOrder = PhaseOrder(PhaseOrder::UI.0 + 40);
/// The greenhouse's VHS and glow finish (the own constant): over the whole frame, the title
/// UI and the Shift+Tab menu's glass included.
pub(crate) const FINISH_ORDER: PhaseOrder = menu_scene::FINISH_ORDER;
/// The buttons' crisp marks, just after the finish: a lighter version of the VHS for the small
/// text (`AeroTitleScreen::build_layers`).
pub(crate) const CRISP_ORDER: PhaseOrder = PhaseOrder(FINISH_ORDER.0 + 10);
/// The title screen's frame cap when the monitor's refresh rate is unknown.
pub(crate) const FALLBACK_REFRESH_HZ: u32 = 60;
/// The first title screen of a session opens with the greenhouse's intro, a fade in from a bright
/// glow ([`menu_scene::INTRO_SECONDS`]). The white title and glass would pop in unseen over the
/// glow, so they start this long into it, as the greenhouse shows through.
const INTRO_UI_DELAY: Duration = Duration::from_millis(1000);
/// Later visits (back from Options, the server list, a server) skip the intro: the scene starts
/// this far into its clock, past the glow and the camera's glide.
const INTRO_SKIP_S: f64 = menu_scene::INTRO_SECONDS * 1.25;
/// How long the title screen must have been gone before the parked greenhouse gives its big GPU
/// memory back (about 200 MB at 1080p; rebuilt in a few milliseconds when the title screen
/// returns). Quick trips (Options, the server list, a short game) keep it, so they rebuild
/// nothing.
const PARKED_RELEASE_AFTER: Duration = Duration::from_secs(60);
/// Vanilla's copyright line (`GuiMainMenu.drawScreen`).
const COPYRIGHT: &str = "Copyright Mojang AB. Do not distribute!";

/// A phase-list slot: a fixed place in the [`PhaseList`] that holds a phase only while the title
/// screen is up. Empty, it is disabled, so the list skips it.
pub(crate) struct PhaseSlot {
    name: &'static str,
    phase: Option<Box<dyn RenderPhase>>,
}

impl PhaseSlot {
    fn new(name: &'static str) -> PhaseSlot {
        PhaseSlot { name, phase: None }
    }

    fn fill(&mut self, phase: Box<dyn RenderPhase>) {
        self.phase = Some(phase);
    }

    fn empty(&mut self) {
        self.phase = None;
    }

    /// Whether a phase is in the slot (tests).
    #[cfg(test)]
    pub(crate) fn is_filled(&self) -> bool {
        self.phase.is_some()
    }
}

impl std::fmt::Debug for PhaseSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhaseSlot")
            .field("name", &self.name)
            .field("filled", &self.phase.is_some())
            .finish()
    }
}

impl RenderPhase for PhaseSlot {
    fn name(&self) -> &'static str {
        self.name
    }

    fn enabled(&self) -> bool {
        self.phase.as_ref().is_some_and(|phase| phase.enabled())
    }

    fn needs_scene_color(&self) -> bool {
        self.phase
            .as_ref()
            .is_some_and(|phase| phase.needs_scene_color())
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

/// What draws behind the title screen (the "Menu scene"
/// option's three scenes).
pub(crate) enum Backdrop {
    /// The greenhouse: its background, its VHS finish, and the crisp layer over the finish.
    Greenhouse(MenuScene),
    /// Glass or Shore: one background phase (the scene and its composite), no finish.
    Scene(SceneBackdrop),
}

impl Backdrop {
    /// Which of the three it is.
    pub(crate) fn kind(&self) -> MenuKind {
        match self {
            Backdrop::Greenhouse(_) => MenuKind::Greenhouse,
            Backdrop::Scene(scene) => scene.kind(),
        }
    }

    /// A phase handle for the background slot.
    fn background_phase(&self) -> Box<dyn RenderPhase> {
        match self {
            Backdrop::Greenhouse(scene) => scene.background_phase(),
            Backdrop::Scene(scene) => scene.phase(),
        }
    }

    /// A phase handle for the finish slot: the greenhouse's VHS finish; the others have none.
    fn finish_phase(&self) -> Option<Box<dyn RenderPhase>> {
        match self {
            Backdrop::Greenhouse(scene) => Some(scene.finish_phase()),
            Backdrop::Scene(_) => None,
        }
    }

    /// Whether it lays a finish over the frame (and the crisp layer is needed over it).
    fn has_finish(&self) -> bool {
        matches!(self, Backdrop::Greenhouse(_))
    }

    /// Give the big GPU memory back, keeping the scene (rebuilt the next time it draws).
    fn release_gpu_memory(&mut self) {
        match self {
            Backdrop::Greenhouse(scene) => scene.release_gpu_memory(),
            Backdrop::Scene(scene) => scene.release_gpu_memory(),
        }
    }

    /// Whether the big GPU memory is currently freed (tests).
    #[cfg(test)]
    pub(crate) fn gpu_memory_released(&self) -> bool {
        match self {
            Backdrop::Greenhouse(scene) => scene.gpu_memory_released(),
            Backdrop::Scene(scene) => scene.gpu_memory_released(),
        }
    }

    /// The title screen went away: a scene's clock stops (the greenhouse's restarts per visit).
    fn pause(&mut self) {
        if let Backdrop::Scene(scene) = self {
            scene.pause();
        }
    }

    /// The Glass or Shore wrapper (tests and the review renders).
    #[cfg(test)]
    pub(crate) fn as_scene_mut(&mut self) -> Option<&mut SceneBackdrop> {
        match self {
            Backdrop::Greenhouse(_) => None,
            Backdrop::Scene(scene) => Some(scene),
        }
    }
}

/// The four slots `install` registered.
#[derive(Debug)]
struct Slots {
    background: Shared<PhaseSlot>,
    ui: Shared<PhaseSlot>,
    finish: Shared<PhaseSlot>,
    crisp: Shared<PhaseSlot>,
}

/// The title screen while it is up. Dropping it frees the UI phases and (with the slots emptied)
/// the scene.
struct Live {
    /// Created on the first frame drawn, so its pop-in starts on a frame the user sees.
    screen: Option<AeroTitleScreen>,
    /// The title, the glass and the buttons, under the finish.
    phase: Shared<AeroTitlePhase>,
    /// The buttons' crisp marks, over the finish (no glass).
    crisp: Shared<AeroTitlePhase>,
    /// The scene this title screen was set up for (the "Menu scene" option, resolved).
    kind: MenuKind,
    /// The scene behind it; `None` when it could not be built (the stand-in shows instead).
    scene: Option<Backdrop>,
    /// When the title screen first drew (the scene's clock).
    shown_at: Option<Instant>,
    /// Added to the scene's clock: 0 plays the intro, [`INTRO_SKIP_S`] skips it.
    scene_offset_s: f64,
    /// How long after the first frame the pop-in starts.
    ui_delay: Duration,
}

impl std::fmt::Debug for Live {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Live")
            .field("screen", &self.screen)
            .field("kind", &self.kind)
            .field("scene", &self.scene.is_some())
            .field("shown_at", &self.shown_at)
            .field("scene_offset_s", &self.scene_offset_s)
            .finish_non_exhaustive()
    }
}

/// The `aero_title` mod's client state (`Client::aero`).
pub(crate) struct AeroTitleState {
    /// The settings generation `enabled` was read at; `0` = never (the store starts at 1).
    seen: std::cell::Cell<u64>,
    enabled: std::cell::Cell<bool>,
    /// The "Logo style" option, re-read with `enabled`.
    logo: std::cell::Cell<LogoStyle>,
    /// The window's monitor refresh rate in millihertz, from the shell.
    monitor_millihertz: Option<u32>,
    /// The cursor in physical pixels.
    cursor: (f32, f32),
    slots: Option<Slots>,
    live: Option<Live>,
    /// The labels' and footer's Segoe UI (kept across visits: CPU only).
    text: Option<MenuText>,
    /// The title's font (kept across visits: CPU only).
    title_font: Option<TitleFont>,
    /// The button labels and the language code they were read in.
    labels: Option<(String, [String; aero_title::AeroButton::COUNT])>,
    /// The origin of the glass shader's `uTime`.
    epoch: Instant,
    /// The title screen's scene started preparing (or was ruled out) for this session: at boot
    /// for the game binary, otherwise on the first frame.
    pub(crate) prewarmed: bool,
    /// The greenhouse's intro has played once this session; later visits skip it.
    intro_played: bool,
    /// The greenhouse while the title screen is away, kept so the next visit reuses it: building
    /// and dropping it on every visit grew the GPU driver's own memory by megabytes over the
    /// first visits (the leak follow-up; see `mc_render::menu_scene::MenuScene`). Any of the three scenes, the same way.
    parked: Option<ParkedScene>,
    /// How many times a scene was built this session: once, then reused (a change of the "Menu
    /// scene" option builds the new one).
    scene_builds: u64,
    /// The "Menu scene" option, this launch's Random pick and the work started
    /// for it (`nw_menu.rs`).
    pub(crate) menu: nw_menu::MenuState,
    /// The greenhouse's VHS finish is out of its slot while the mod menu
    /// is up over the title screen (the finish would blur the menu's small text); it goes back
    /// when the title screen shows again.
    finish_off: bool,
    /// The updater's side of the title screen: its thread while "Check for
    /// updates" is on in a portable install, and the card's view (`w21_updater.rs`).
    pub(crate) update: super::w21_updater::UpdateUi,
}

/// The scene, out of the phase list while the title screen is away.
struct ParkedScene {
    scene: Backdrop,
    since: Instant,
    released: bool,
}

impl ParkedScene {
    fn new(mut scene: Backdrop) -> ParkedScene {
        scene.pause();
        ParkedScene {
            scene,
            since: Instant::now(),
            released: false,
        }
    }

    /// Give the big GPU memory back once the title screen has been gone a while. Afterwards a
    /// `bool` test per frame.
    fn release_if_due(&mut self) {
        if !self.released && self.since.elapsed() >= PARKED_RELEASE_AFTER {
            self.scene.release_gpu_memory();
            self.released = true;
        }
    }
}

impl Default for AeroTitleState {
    fn default() -> Self {
        AeroTitleState {
            seen: std::cell::Cell::new(0),
            enabled: std::cell::Cell::new(false),
            logo: std::cell::Cell::new(LogoStyle::default()),
            monitor_millihertz: None,
            cursor: (-1.0, -1.0),
            slots: None,
            live: None,
            text: None,
            title_font: None,
            labels: None,
            epoch: Instant::now(),
            prewarmed: false,
            intro_played: false,
            parked: None,
            scene_builds: 0,
            menu: nw_menu::MenuState::default(),
            finish_off: false,
            update: super::w21_updater::UpdateUi::default(),
        }
    }
}

impl std::fmt::Debug for AeroTitleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AeroTitleState")
            .field("enabled", &self.enabled.get())
            .field("monitor_millihertz", &self.monitor_millihertz)
            .field("live", &self.live.is_some())
            .finish_non_exhaustive()
    }
}

impl AeroTitleState {
    /// Whether the mod is on, re-read only when the settings changed.
    fn enabled(&self, settings: &mc_mods::ModSettings) -> bool {
        let generation = settings.generation();
        if self.seen.get() != generation {
            self.seen.set(generation);
            self.enabled.set(settings.enabled("aero_title"));
            self.logo.set(LogoStyle::from_settings(settings));
        }
        self.enabled.get()
    }

    /// The "Logo style" option (cached like `enabled`): the loading screen draws the logo in it
    /// too.
    pub(crate) fn logo_style(&self, settings: &mc_mods::ModSettings) -> LogoStyle {
        let _ = self.enabled(settings);
        self.logo.get()
    }

    /// The live title screen (if one is up) and the cursor in physical
    /// pixels, for the update card's clicks (`w21_updater.rs`).
    pub(crate) fn live_screen(&mut self) -> Option<(&mut AeroTitleScreen, (f32, f32))> {
        let cursor = self.cursor;
        let screen = self.live.as_mut()?.screen.as_mut()?;
        Some((screen, cursor))
    }

    /// Tests: put the live title screen's clocks at `shown` since its
    /// scene appeared and `popin` since its pop-in started (pictures of a settled screen).
    #[cfg(test)]
    pub(crate) fn set_live_clocks(&mut self, shown: Duration, popin: Duration) {
        let now = Instant::now();
        if let Some(live) = self.live.as_mut() {
            live.shown_at = Some(now - shown);
            if let Some(screen) = live.screen.as_mut() {
                screen.restart(now - popin);
            }
        }
    }

    /// Store the shell's report of the window's monitor refresh rate.
    pub(crate) fn set_monitor_refresh(&mut self, millihertz: Option<u32>) {
        self.monitor_millihertz = millihertz;
    }

    /// Take the live title screen down, parking its greenhouse for the next visit.
    fn park_live(&mut self) {
        if let Some(scene) = self.live.take().and_then(|live| live.scene) {
            self.parked = Some(ParkedScene::new(scene));
        }
    }
}

/// The title screen's frame cap for a monitor refreshing at `millihertz`: its rate in whole
/// hertz (59.94 Hz is 60), [`FALLBACK_REFRESH_HZ`] when unknown, and never below vanilla's 30.
#[must_use]
pub(crate) fn refresh_cap(millihertz: Option<u32>) -> u32 {
    match millihertz {
        Some(mhz) if mhz >= 1000 => mhz.saturating_add(500) / 1000,
        _ => FALLBACK_REFRESH_HZ,
    }
    .max(30)
}

impl Client {
    /// Whether the title screen on show is the Aero one: the mod is on and the title screen is
    /// up (not the loading screen). `pub(crate)` for the account corner's clicks.
    pub(crate) fn aero_title_showing(&self) -> bool {
        // The replay browser opened from the title screen keeps the
        // greenhouse behind it.
        (matches!(self.ui, Ui::MainMenu(_))
            || self.replay_browser_over_title()
            || self.mod_menu_over_title())
            && self.aero.enabled(&self.mods.settings)
            && !self.is_loading()
    }

    /// The mod menu is up with no world (the title screen's Mods
    /// button): the greenhouse keeps running behind its glass while the title and the buttons
    /// step aside, as for the replay browser.
    pub(crate) fn mod_menu_over_title(&self) -> bool {
        matches!(self.ui, Ui::ModMenu(_)) && self.session.is_none()
    }

    /// Register the three slots (from `Client::install`, so also after a
    /// resource reload rebuilt every phase; a live title screen is rebuilt on its next frame,
    /// into the new slots).
    pub(crate) fn aero_title_install_phases(&mut self, phases: &mut PhaseList) {
        let slots = Slots {
            background: Shared::new(PhaseSlot::new("aero_title/background")),
            ui: Shared::new(PhaseSlot::new("aero_title")),
            finish: Shared::new(PhaseSlot::new("aero_title/finish")),
            crisp: Shared::new(PhaseSlot::new("aero_title/crisp")),
        };
        phases.insert(BACKGROUND_ORDER, Box::new(slots.background.clone()));
        phases.insert(UI_ORDER, Box::new(slots.ui.clone()));
        phases.insert(FINISH_ORDER, Box::new(slots.finish.clone()));
        phases.insert(CRISP_ORDER, Box::new(slots.crisp.clone()));
        self.aero.slots = Some(slots);
        // The greenhouse uses the vanilla jar only, so a resource reload leaves it valid: park it
        // for the rebuilt title screen rather than building it again.
        self.aero.park_live();
    }

    /// How many times this session built a title-screen scene (the greenhouse, Glass or Shore):
    /// once, on the first title screen, and then never
    /// again however often the title screen comes and goes (the leak test checks this), unless
    /// the "Menu scene" option changes. `0` until a title screen with a scene has drawn.
    #[must_use]
    pub fn aero_title_scene_builds(&self) -> u64 {
        self.aero.scene_builds
    }

    /// On the Aero title screen, the monitor's refresh rate (the user's
    /// "link menu fps cap to monitor cap", 2026-11-07); `None` everywhere else, where vanilla's
    /// rule stands (30 FPS on every other worldless screen).
    pub(crate) fn aero_title_frame_cap(&self) -> Option<u32> {
        self.aero_title_showing()
            .then(|| refresh_cap(self.aero.monitor_millihertz))
    }

    /// The monitor's refresh rate as a frame cap ([`refresh_cap`]), for the animated loading
    /// screen (`tw_loading.rs`).
    pub(crate) fn aero_title_refresh_cap(&self) -> u32 {
        refresh_cap(self.aero.monitor_millihertz)
    }

    /// Whether the live title screen's clock (its intro and pop-in) has started. Tests: the
    /// loading screen holds it until its fade.
    #[cfg(test)]
    pub(crate) fn aero_title_clock_started(&self) -> bool {
        self.aero
            .live
            .as_ref()
            .is_some_and(|live| live.shown_at.is_some())
    }

    /// This frame's cursor, in physical pixels (an `Option` test unless the
    /// title screen is up).
    pub(crate) fn aero_title_pointer(&mut self, cursor: (f64, f64)) {
        let Some(screen) = self
            .aero
            .live
            .as_mut()
            .and_then(|live| live.screen.as_mut())
        else {
            return;
        };
        let (x, y) = (cursor.0 as f32, cursor.1 as f32);
        self.aero.cursor = (x, y);
        screen.pointer_moved(x, y);
    }

    /// A mouse button edge on the Aero title screen.
    /// Returns `true` when the title screen took it (it takes every edge while it is up; the
    /// hidden vanilla buttons get none). Releasing over the button the press landed on plays
    /// vanilla's button sound and does what vanilla's button does.
    pub(crate) fn aero_title_mouse(&mut self, button: i32, pressed: bool) -> bool {
        if !matches!(self.ui, Ui::MainMenu(_)) {
            return false;
        }
        let (x, y) = self.aero.cursor;
        // The update card (bottom left) takes clicks on itself first.
        if let Some(taken) = self.upd_title_mouse(button, pressed) {
            return taken;
        }
        // The account corner (top right) takes its clicks first.
        if self.acc_title_mouse(x, y, button, pressed) {
            return true;
        }
        let Some(screen) = self
            .aero
            .live
            .as_mut()
            .and_then(|live| live.screen.as_mut())
        else {
            return false;
        };
        let Ok(button) = u8::try_from(button) else {
            return true;
        };
        if pressed {
            screen.mouse_pressed(x, y, button);
            return true;
        }
        if let Some(clicked) = screen.mouse_released(x, y, button) {
            self.play_button_press();
            // Replays has no vanilla action: it opens the replay browser.
            if clicked == aero_title::AeroButton::Replays {
                self.open_replay_browser();
            } else {
                self.handle_action(clicked.action());
            }
        }
        true
    }

    /// The title screen was left: empty the slots, drop the UI phases, park the scene for the next
    /// visit (it gives its big GPU memory back after [`PARKED_RELEASE_AFTER`]), and give the
    /// world's sky and fog passes back.
    fn aero_title_leave(&mut self) {
        if let Some(slots) = &self.aero.slots {
            slots.background.with(PhaseSlot::empty);
            slots.ui.with(PhaseSlot::empty);
            slots.finish.with(PhaseSlot::empty);
            slots.crisp.with(PhaseSlot::empty);
        }
        self.aero.park_live();
        if let Some(render) = &self.render {
            render.sky.with(|phase| phase.hidden = false);
            render.fog.with(|phase| phase.visible = true);
        }
    }

    /// While the title screen is up, the world passes the client still records on it (the sky
    /// dome, the clouds, the screen-space fog) draw nothing anyone sees: the greenhouse covers
    /// every pixel. Switch them off for this frame. The clouds' mode is re-read from the settings
    /// every frame and the sky and fog are given back by [`Client::aero_title_leave`].
    fn aero_title_quiet_world(&self) {
        if let Some(render) = &self.render {
            render.sky.with(|phase| phase.hidden = true);
            render
                .clouds
                .with(|phase| phase.mode = mc_render::sky::CloudMode::Off);
            render.fog.with(|phase| phase.visible = false);
        }
    }

    /// Draw this frame of the Aero title screen, if it is up. Returns `true`
    /// when it drew, and the vanilla menu then draws nothing. `list` is the UI phase's list (a
    /// plain backdrop goes there if the greenhouse could not be built). The frame that the title
    /// screen is left, the scene and the phase are dropped. Also starts the greenhouse's
    /// background bake on the session's first frame (one `bool` test afterwards).
    pub(crate) fn build_aero_title(
        &mut self,
        gpu: &Gpu,
        list: &mut DrawList,
        ctx: &ScreenCtx,
    ) -> bool {
        if !self.aero.prewarmed {
            self.aero_title_prewarm();
        }
        if let Some(parked) = self.aero.parked.as_mut() {
            parked.release_if_due();
        }
        // (the replay browser over the title screen keeps the scene)
        // (so does the mod menu)
        if !matches!(self.ui, Ui::MainMenu(_))
            && !self.replay_browser_over_title()
            && !self.mod_menu_over_title()
        {
            if self.aero.live.is_some() {
                self.aero_title_leave();
            }
            return false;
        }
        // A title screen (Aero's or vanilla's) is up and loading is over: a
        // just-installed update works. One relaxed atomic load per menu frame after the first.
        if !self.is_loading() {
            mc_update::helper::title_screen_reached();
        }
        if !self.aero_title_showing() || self.aero.slots.is_none() {
            if self.aero.live.is_some() {
                self.aero_title_leave();
            }
            return false;
        }
        self.aero_title_ensure_live(gpu);
        // The "Menu scene" option changed while the title screen is up.
        self.aero_title_follow_choice(gpu);
        // While the animated loading screen still covers everything, the title
        // screen is built (its GPU side, out of sight) but its clocks, the greenhouse's intro
        // glow and the buttons' pop-in, wait for the fade (`tw_loading.rs`). An `Option` test.
        if self.tw_loading_holds_title() {
            return true;
        }
        // After building whatever the title screen needed: its pop-in starts on this frame.
        self.aero_title_frame(gpu, list, ctx, Instant::now());
        true
    }

    /// Start preparing the title screen's scene on a background thread, once, as early as
    /// possible, so it is ready when the title screen appears: at boot for the game binary
    /// (`nw_menu_boot`, before the window exists), otherwise on the first frame. Only the scene
    /// this launch will show: the greenhouse's bake, or Glass's or
    /// Shore's few vanilla textures. Only for a session headed for the title screen with the mod
    /// on.
    pub(crate) fn aero_title_prewarm(&mut self) {
        if self.aero.prewarmed
            || self.session.is_some()
            || self.config.replay.is_some()
            || self.diagnostic_progress.is_some()
        {
            return;
        }
        self.aero.prewarmed = true;
        if !self.aero.enabled(&self.mods.settings) {
            return;
        }
        let kind = self.aero.menu.kind(&self.mods.settings);
        if let Some(jar) = self.config.version_jar.clone() {
            match kind.scene() {
                None => menu_scene::prewarm(&jar),
                Some(scene) => self.aero.menu.prefetch(scene, jar),
            }
        }
    }

    /// Build `kind`'s scene for the title screen: the greenhouse (from the vanilla jar,
    /// waiting for its bake if it is not done), or Glass or Shore from their vanilla textures (the
    /// ones read at boot, or read now). Never the user's resource packs. `None` when it cannot be
    /// built (the plain stand-in backdrop shows instead).
    fn aero_title_build_backdrop(&mut self, gpu: &Gpu, kind: MenuKind) -> Option<Backdrop> {
        let jar = self.config.version_jar.clone();
        let backdrop = match kind.scene() {
            None => match jar.as_deref().map(|jar| MenuScene::new(gpu, jar)) {
                Some(Ok(scene)) => Some(Backdrop::Greenhouse(scene)),
                Some(Err(error)) => {
                    tracing::warn!(%error, "the Aero title screen keeps its plain backdrop");
                    None
                }
                None => None,
            },
            Some(scene) => {
                let textures = self.aero.menu.textures(scene, jar.as_deref());
                let seed = self.aero.menu.seed();
                SceneBackdrop::new(gpu, kind, &textures, seed).map(Backdrop::Scene)
            }
        };
        if backdrop.is_some() {
            self.aero.scene_builds += 1;
        }
        backdrop
    }

    /// `kind`'s scene for a title screen: the parked one when it is that scene (an earlier
    /// visit's), otherwise a new one. A parked scene of another kind (the option changed) is
    /// dropped, freeing its GPU memory.
    fn aero_title_take_backdrop(&mut self, gpu: &Gpu, kind: MenuKind) -> Option<Backdrop> {
        match self.aero.parked.take() {
            Some(parked) if parked.scene.kind() == kind => Some(parked.scene),
            stale => {
                drop(stale);
                self.aero_title_build_backdrop(gpu, kind)
            }
        }
    }

    /// The "Menu scene" option changed while the title screen is up: put the
    /// new scene in the slots at once. The old one is parked (a switch back reuses it). Switching
    /// to the greenhouse before its bake is done starts the bake on a background thread and keeps
    /// the old scene up until it is (rather than freezing the menu on it). A `bool` and an enum
    /// test per frame otherwise.
    fn aero_title_follow_choice(&mut self, gpu: &Gpu) {
        let kind = self.aero.menu.kind(&self.mods.settings);
        let Some(live) = self.aero.live.as_ref() else {
            return;
        };
        if live.kind == kind {
            return;
        }
        if kind == MenuKind::Greenhouse && !menu_scene::is_prepared() {
            if let Some(jar) = self.config.version_jar.as_deref() {
                menu_scene::prewarm(jar);
                return;
            }
        }
        let old = self.aero.live.as_mut().and_then(|live| live.scene.take());
        if let Some(slots) = &self.aero.slots {
            slots.background.with(PhaseSlot::empty);
            slots.finish.with(PhaseSlot::empty);
        }
        let new = self.aero_title_take_backdrop(gpu, kind);
        if let Some(old) = old {
            self.aero.parked = Some(ParkedScene::new(old));
        }
        if let (Some(slots), Some(scene)) = (&self.aero.slots, &new) {
            slots
                .background
                .with(|slot| slot.fill(scene.background_phase()));
            if !self.aero.finish_off {
                if let Some(finish) = scene.finish_phase() {
                    slots.finish.with(|slot| slot.fill(finish));
                }
            }
        }
        tracing::info!(scene = kind.name(), "title screen scene switched");
        if let Some(live) = self.aero.live.as_mut() {
            live.kind = kind;
            live.scene = new;
            // A greenhouse switched to mid-visit skips its intro glow.
            live.scene_offset_s = INTRO_SKIP_S;
        }
        // A new phase has a new renderer: both glyph textures go up again.
        if let Some(text) = self.aero.text.as_mut() {
            text.mark_dirty();
        }
        if let Some(title) = self.aero.title_font.as_mut() {
            title.mark_dirty();
        }
    }

    /// Fill the slots for a title screen that just appeared: its glass phase, and the scene the
    /// "Menu scene" option asks for (the vanilla jar's own textures, never the user's packs).
    fn aero_title_ensure_live(&mut self, gpu: &Gpu) {
        if self.aero.live.is_some() || self.aero.slots.is_none() {
            return;
        }
        let kind = self.aero.menu.kind(&self.mods.settings);
        // The parked scene from an earlier visit, or (the first time) a new one.
        let scene = self.aero_title_take_backdrop(gpu, kind);
        let Some(slots) = &self.aero.slots else {
            return;
        };
        let phase = Shared::new(AeroTitlePhase::new(gpu));
        slots.ui.with(|slot| slot.fill(Box::new(phase.clone())));
        let crisp = Shared::new(AeroTitlePhase::new(gpu));
        slots.crisp.with(|slot| slot.fill(Box::new(crisp.clone())));
        if let Some(scene) = &scene {
            slots
                .background
                .with(|slot| slot.fill(scene.background_phase()));
            if let Some(finish) = scene.finish_phase() {
                slots.finish.with(|slot| slot.fill(finish));
            }
        }
        // A new phase has a new renderer: both glyph textures go up again.
        if let Some(text) = self.aero.text.as_mut() {
            text.mark_dirty();
        }
        if let Some(title) = self.aero.title_font.as_mut() {
            title.mark_dirty();
        }
        // The session's first title screen opens with an intro: the greenhouse's glow (the UI
        // waits for it), or for Glass and Shore a short delay while the loading screen's fade
        // uncovers the scene.
        let first = scene.is_some() && !self.aero.intro_played;
        self.aero.intro_played |= scene.is_some();
        let glow = first && kind == MenuKind::Greenhouse;
        let ui_delay = match (first, kind) {
            (false, _) => Duration::ZERO,
            (true, MenuKind::Greenhouse) => INTRO_UI_DELAY,
            (true, _) => Duration::from_millis(nw_menu::SCENE_UI_DELAY_MS),
        };
        self.aero.live = Some(Live {
            screen: None,
            phase,
            crisp,
            kind,
            scene,
            shown_at: None,
            scene_offset_s: if glow { 0.0 } else { INTRO_SKIP_S },
            ui_delay,
        });
    }

    /// One frame of the live title screen at `now`.
    fn aero_title_frame(&mut self, gpu: &Gpu, list: &mut DrawList, ctx: &ScreenCtx, now: Instant) {
        let (width, height) = (gpu.config.width, gpu.config.height);
        // The labels, in the game's language (re-read when it changes).
        let code = self.assets.language.code();
        if self
            .aero
            .labels
            .as_ref()
            .is_none_or(|(seen, _)| seen != code)
        {
            self.aero.labels = Some((code.to_owned(), aero_title::labels(&self.assets.language)));
        }
        // Wrapped before the f32 conversion: no stutter after a long uptime.
        let time_s = crate::app::stability::cycle_seconds(
            now.saturating_duration_since(self.aero.epoch),
            crate::app::stability::UI_ANIMATION_CYCLE,
        );
        let logo = self.aero.logo_style(&self.mods.settings);
        // The mod menu over the title screen steps the UI aside too.
        let menu_over = self.mod_menu_over_title();
        let browsing = self.replay_browser_over_title() || menu_over;
        let Some(live) = self.aero.live.as_mut() else {
            return;
        };
        let shown_at = *live.shown_at.get_or_insert(now);
        match live.scene.as_mut() {
            Some(Backdrop::Greenhouse(scene)) => {
                let since = now.saturating_duration_since(shown_at).as_secs_f64();
                scene.update(live.scene_offset_s + since, width, height);
            }
            // Glass and Shore: one clock step and the frame's size.
            Some(Backdrop::Scene(scene)) => scene.advance(now, width, height),
            None => {}
        }
        // The replay browser is up over the title screen: the
        // greenhouse keeps running behind it; the title and the buttons step aside (and pop in
        // again when the browser closes).
        if browsing {
            let (phase, crisp) = (live.phase.clone(), live.crisp.clone());
            if let Some(screen) = live.screen.as_mut() {
                screen.restart(now);
            }
            let has_scene = live.scene.is_some();
            phase.with(AeroTitlePhase::deactivate);
            crisp.with(AeroTitlePhase::deactivate);
            if !has_scene {
                aero_title::draw_stand_in_backdrop(list, ctx.width, ctx.height);
            }
            // The VHS finish goes over everything under it, the mod
            // menu's glass included (`FINISH_ORDER` is above the menu's phase): take it out while
            // the menu is up, so the menu's small text stays crisp.
            if menu_over && !self.aero.finish_off {
                if let Some(slots) = &self.aero.slots {
                    slots.finish.with(PhaseSlot::empty);
                }
                self.aero.finish_off = true;
            }
            self.aero_title_quiet_world();
            return;
        }
        // Back on the title screen: the finish returns (the greenhouse's;
        // Glass and Shore have none).
        if self.aero.finish_off {
            if let (Some(slots), Some(finish)) = (
                &self.aero.slots,
                live.scene.as_ref().and_then(Backdrop::finish_phase),
            ) {
                slots.finish.with(|slot| slot.fill(finish));
            }
            self.aero.finish_off = false;
        }
        let has_scene = live.scene.is_some();
        // Only the greenhouse lays a finish over the UI.
        let has_finish = live.scene.as_ref().is_some_and(Backdrop::has_finish);
        let ui_delay = live.ui_delay;
        let screen = live
            .screen
            .get_or_insert_with(|| AeroTitleScreen::new(now + ui_delay));
        // The update card follows the updater (`w21_updater.rs`).
        super::w21_updater::sync_notice(
            &mut self.aero.update,
            &mut screen.notice,
            &self.mods.settings,
        );
        let text = self.aero.text.get_or_insert_with(MenuText::load);
        let title_font = self.aero.title_font.get_or_insert_with(TitleFont::embedded);
        title_font.set_style(logo);
        let Some((_, labels)) = self.aero.labels.as_ref() else {
            return;
        };
        let version = match &self.ui {
            Ui::MainMenu(menu) => menu.version_line.as_str(),
            _ => "",
        };
        let (phase, crisp) = (live.phase.clone(), live.crisp.clone());
        let mut content = phase.with(AeroTitlePhase::take_spare_list);
        let mut crisp_content = crisp.with(AeroTitlePhase::take_spare_list);
        let mut env = AeroTitleEnv {
            text,
            title_font,
            mc_font: &mut self.font,
            labels,
            version,
            copyright: COPYRIGHT,
            screen: (width as f32, height as f32),
            now,
            time_s,
        };
        // The greenhouse's finish blurs everything under it; the crisp copy of the buttons' marks
        // goes over it (only with the greenhouse: without it there is no finish).
        let mut glass = if has_finish {
            screen.build_layers(&mut env, &mut content, Some(&mut crisp_content))
        } else {
            screen.build(&mut env, &mut content)
        };
        // The account corner, top right (`w21_accounts.rs`), and its
        // head's texture in both title phases.
        let opened_at = screen.opened_at();
        super::w21_accounts::title_corner(
            &mut self.acc,
            &mut env,
            opened_at,
            self.aero.cursor,
            &mut content,
            has_finish.then_some(&mut crisp_content),
            &mut glass,
        );
        self.acc_title_head(gpu, [&phase, &crisp]);
        let text_upload = self
            .aero
            .text
            .as_mut()
            .and_then(|text| text.take_upload().map(<[u8]>::to_vec));
        let minecraft_font = self.aero.text.as_ref().is_some_and(|text| !text.is_segoe());
        phase.with(|phase| {
            if let Some(pixels) = &text_upload {
                phase.upload_text_atlas(gpu, pixels);
            }
            if minecraft_font {
                // The Minecraft-font fallback draws with the vanilla font pages.
                let _ =
                    phase
                        .renderer()
                        .ensure_list_textures(gpu, &self.assets.resources, &content);
            }
            if let Some((pixels, w, h)) = self
                .aero
                .title_font
                .as_mut()
                .and_then(TitleFont::take_upload)
            {
                phase.upload_title(gpu, pixels, w, h);
            }
            phase.submit(content, glass);
        });
        crisp.with(|phase| {
            if let Some(pixels) = &text_upload {
                phase.upload_text_atlas(gpu, pixels);
            }
            if minecraft_font {
                let _ = phase.renderer().ensure_list_textures(
                    gpu,
                    &self.assets.resources,
                    &crisp_content,
                );
            }
            if crisp_content.is_empty() {
                phase.deactivate();
            } else {
                // No glass: content only, so no scene copy and no full-screen pass.
                phase.submit(crisp_content, GlassFrame::default());
            }
        });
        // Without the greenhouse (no jar, a failed bake): a bright plain backdrop under the glass.
        if !has_scene {
            aero_title::draw_stand_in_backdrop(list, ctx.width, ctx.height);
        }
        self.aero_title_quiet_world();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_is_the_monitors_rate_in_whole_hertz_with_a_sixty_fallback() {
        assert_eq!(refresh_cap(Some(360_000)), 360);
        assert_eq!(refresh_cap(Some(143_856)), 144);
        assert_eq!(refresh_cap(Some(59_940)), 60);
        assert_eq!(refresh_cap(None), FALLBACK_REFRESH_HZ);
        assert_eq!(refresh_cap(Some(0)), FALLBACK_REFRESH_HZ);
        assert_eq!(
            refresh_cap(Some(24_000)),
            30,
            "never stricter than vanilla's menu cap"
        );
    }

    #[test]
    fn the_slots_sit_around_the_ui_and_the_shift_tab_menu() {
        assert!(BACKGROUND_ORDER < PhaseOrder::UI && BACKGROUND_ORDER > PhaseOrder::POST);
        assert!(UI_ORDER > PhaseOrder::UI);
        assert!(
            UI_ORDER < PhaseOrder(PhaseOrder::UI.0 + 50),
            "under the menu's glass"
        );
        assert!(
            FINISH_ORDER > PhaseOrder(PhaseOrder::UI.0 + 50),
            "over the menu's glass"
        );
        assert!(
            CRISP_ORDER > FINISH_ORDER,
            "the crisp marks go over the finish"
        );
        let mut slot = PhaseSlot::new("x");
        assert!(!slot.enabled() && !slot.needs_scene_color());
        slot.empty();
        assert!(!slot.is_filled());
    }
}

/// The title screen through the real client: `Client::new` on the real assets, `install`, the
/// real `build_ui` and input paths, a headless GPU. Needs the vanilla jar (skips without it; set
/// `MC_JARS_DIR`).
#[cfg(test)]
mod client_tests {
    use std::time::{Duration, Instant};

    use mc_render::{Gpu, GpuOptions, InputState, PhaseList, ShellApp as _};
    use mc_ui::screens::aero_title::AeroButton;
    use mc_ui::screens::menu::MainMenuScreen;
    use winit::event::MouseButton;

    use super::super::super::{Client, Ui};
    use crate::app::config::{ClientConfig, LoginMode};

    const SIZE: (u32, u32) = (854, 480);

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

    /// Put the live title screen's clocks at: the greenhouse shown `shown` ago, the pop-in started
    /// `popin` ago.
    fn set_clock(client: &mut Client, shown: Duration, popin: Duration) {
        let now = Instant::now();
        let live = client.aero.live.as_mut().expect("the title screen is live");
        live.shown_at = Some(now - shown);
        live.screen
            .as_mut()
            .expect("the title screen has drawn")
            .restart(now - popin);
    }

    fn ui_slot_filled(client: &Client) -> bool {
        client
            .aero
            .slots
            .as_ref()
            .expect("installed")
            .ui
            .with(|slot| slot.is_filled())
    }

    fn button_centre(client: &Client, button: AeroButton) -> (f32, f32) {
        client
            .aero
            .live
            .as_ref()
            .and_then(|live| live.screen.as_ref())
            .expect("the title screen is live")
            .button_rect(button)
            .center()
    }

    /// A left click at `(x, y)` (physical pixels), pressed and released in two input passes.
    fn click(client: &mut Client, (x, y): (f32, f32)) {
        let mut input = InputState::new();
        input.on_cursor_moved(f64::from(x), f64::from(y));
        input.on_mouse_button(MouseButton::Left, true);
        client.handle_screen_input(&mut input, SIZE.0, SIZE.1);
        input.on_mouse_button(MouseButton::Left, false);
        client.handle_screen_input(&mut input, SIZE.0, SIZE.1);
    }

    #[test]
    fn the_title_screen_is_aero_by_default_and_vanillas_when_the_mod_is_off() {
        let Some(gpu) = gpu(SIZE) else { return };
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);
        for name in [
            "aero_title/background",
            "aero_title",
            "aero_title/finish",
            "aero_title/crisp",
        ] {
            assert!(phases.names().any(|n| n == name), "{name} slot registered");
        }
        assert!(
            !ui_slot_filled(&client),
            "empty until the title screen draws"
        );

        client.build_ui(&gpu, 0.0);
        assert!(
            ui_slot_filled(&client),
            "the title screen's glass phase is in its slot"
        );
        let render = client.render.as_ref().expect("installed");
        assert!(
            !render.ui.with(|phase| phase.has_panorama_submission()),
            "the vanilla panorama is not drawn under the Aero title screen"
        );
        // The world passes the greenhouse covers are off while it shows.
        assert!(render.sky.with(|phase| phase.hidden));
        assert!(!render.fog.with(|phase| phase.visible));
        assert_eq!(
            render.clouds.with(|phase| phase.mode),
            mc_render::sky::CloudMode::Off
        );
        // The frame cap: the monitor's refresh rate, 60 until the shell has reported one.
        let settings = client.settings.clone();
        assert_eq!(client.effective_frame_cap(&settings, true), Some(60));
        client.monitor_refresh_changed(Some(359_966));
        assert_eq!(client.effective_frame_cap(&settings, true), Some(360));
        client.monitor_refresh_changed(Some(143_856));
        assert_eq!(client.effective_frame_cap(&settings, false), Some(144));

        // Off: vanilla's title screen, its panorama and its 30 FPS, and the slots are empty.
        client.mods.settings.set_enabled("aero_title", false);
        client.build_ui(&gpu, 0.0);
        assert!(!ui_slot_filled(&client));
        assert!(
            client.aero.live.is_none(),
            "the phase is dropped and the scene parked"
        );
        assert!(client.aero.parked.is_some(), "the greenhouse is parked");
        let render = client.render.as_ref().expect("installed");
        assert!(render.ui.with(|phase| phase.has_panorama_submission()));
        assert!(!render.sky.with(|phase| phase.hidden), "the sky is back");
        assert!(render.fog.with(|phase| phase.visible), "and the fog");
        assert_eq!(client.effective_frame_cap(&settings, true), Some(30));
        // And back on: the parked greenhouse comes back, it is not built again.
        client.mods.settings.set_enabled("aero_title", true);
        client.build_ui(&gpu, 0.0);
        assert!(ui_slot_filled(&client));
        assert!(client.aero.parked.is_none());
        assert_eq!(
            client.aero_title_scene_builds(),
            1,
            "built once, then reused"
        );
    }

    /// The leak follow-up: the greenhouse is built once, parked (not dropped) while the
    /// title screen is away, gives its big GPU memory back after [`super::PARKED_RELEASE_AFTER`],
    /// and the same scene comes back, rebuilt on its first frame.
    #[test]
    fn the_greenhouse_is_built_once_then_parked_and_released_after_a_while_away() {
        let Some(mut gpu) = gpu(SIZE) else { return };
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);
        let camera = mc_render::Camera::default();
        let frame = |client: &mut Client, gpu: &mut Gpu, phases: &mut PhaseList| {
            client.build_ui(gpu, 0.0);
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(gpu, target.view(), &camera, 0.0);
        };
        frame(&mut client, &mut gpu, &mut phases);
        assert_eq!(client.aero_title_scene_builds(), 1);
        // Away: parked with its memory.
        client.mods.settings.set_enabled("aero_title", false);
        frame(&mut client, &mut gpu, &mut phases);
        let parked = client.aero.parked.as_ref().expect("parked");
        assert!(!parked.released && !parked.scene.gpu_memory_released());
        // A while later the big GPU memory goes back; the scene stays.
        client.aero.parked.as_mut().expect("parked").since -= super::PARKED_RELEASE_AFTER;
        frame(&mut client, &mut gpu, &mut phases);
        let parked = client.aero.parked.as_ref().expect("still parked");
        assert!(parked.released && parked.scene.gpu_memory_released());
        // Back: the same scene, rebuilt on its first frame, not built again.
        client.mods.settings.set_enabled("aero_title", true);
        frame(&mut client, &mut gpu, &mut phases);
        assert_eq!(client.aero_title_scene_builds(), 1);
        let live = client.aero.live.as_ref().expect("the title screen is live");
        assert!(!live
            .scene
            .as_ref()
            .expect("the greenhouse")
            .gpu_memory_released());
    }

    #[test]
    fn each_glass_button_does_what_vanillas_button_does_and_leaving_frees_the_screen() {
        let Some(gpu) = gpu(SIZE) else { return };
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);
        type Check = fn(&Client) -> bool;
        let cases: [(AeroButton, Check); 6] = [
            (AeroButton::Multiplayer, |c| {
                matches!(c.ui, Ui::Multiplayer(_))
            }),
            // Replays opens the replay browser.
            (AeroButton::Replays, |c| {
                matches!(c.ui, Ui::ReplayBrowser(_))
            }),
            // Mods opens the mod menu over the title screen.
            (AeroButton::Mods, |c| matches!(c.ui, Ui::ModMenu(_))),
            (AeroButton::Options, |c| matches!(c.ui, Ui::Options(_))),
            (AeroButton::Language, |c| matches!(c.ui, Ui::Language(_))),
            (AeroButton::Quit, |c| !c.keep_running()),
        ];
        for (index, (button, opened)) in cases.into_iter().enumerate() {
            client.ui = Ui::MainMenu(Box::new(MainMenuScreen::new("splash")));
            client.build_ui(&gpu, 0.0);
            set_clock(&mut client, Duration::from_secs(5), Duration::from_secs(1));
            client.build_ui(&gpu, 0.0);
            // Pressing and releasing between the buttons does nothing, silently.
            click(&mut client, (5.0, 5.0));
            assert!(matches!(client.ui, Ui::MainMenu(_)), "{button:?}: a miss");
            if index == 0 {
                assert!(client.audio.last_play().is_none(), "a miss plays no sound");
            }
            let centre = button_centre(&client, button);
            click(&mut client, centre);
            assert!(opened(&client), "{button:?} did not do vanilla's action");
            let sound = client
                .audio
                .last_play()
                .expect("a click plays vanilla's button sound");
            assert_eq!(sound.request.event.path, crate::app::audio::BUTTON_PRESS);
            if button == AeroButton::Replays {
                // The browser opens over the title screen: the greenhouse keeps running behind
                // it (the title screen stays live), its buttons step aside, and Esc returns.
                client.build_ui(&gpu, 0.0);
                assert!(
                    client.aero.live.is_some(),
                    "the greenhouse stays behind the browser"
                );
                assert!(client.replay_browser_over_title());
                client.show_screen(None);
                assert!(matches!(client.ui, Ui::MainMenu(_)));
                continue;
            }
            if button == AeroButton::Mods {
                // The menu opens over the title screen: the greenhouse
                // keeps running behind its glass without the VHS finish (crisp menu text), and
                // closing the menu returns to the title screen with the finish back.
                client.build_ui(&gpu, 0.0);
                assert!(
                    client.aero.live.is_some(),
                    "the greenhouse stays behind the menu"
                );
                assert!(client.mod_menu_over_title());
                if client
                    .aero
                    .live
                    .as_ref()
                    .is_some_and(|live| live.scene.is_some())
                {
                    assert!(client.aero.finish_off, "no VHS finish over the menu");
                }
                client.close_mod_menu(Instant::now());
                assert!(
                    matches!(client.ui, Ui::MainMenu(_)),
                    "back to the title screen"
                );
                client.build_ui(&gpu, 0.0);
                assert!(!client.aero.finish_off, "the finish is back");
                continue;
            }
            if button != AeroButton::Quit {
                // The next frame is on another screen: the title screen is gone, and so are its
                // phase and scene, with vanilla's 30 FPS rule back.
                client.build_ui(&gpu, 0.0);
                assert!(client.aero.live.is_none(), "{button:?}: left");
                assert!(!ui_slot_filled(&client));
                let settings = client.settings.clone();
                assert_eq!(client.effective_frame_cap(&settings, true), Some(30));
            }
        }
    }

    /// The review images for the user, rendered through the real
    /// client: `Client::frame` (the whole per-frame path, `build_ui` included) and the installed
    /// phase list, at 1920x1080, read back from a headless GPU. Writes PNGs only when
    /// `MC_AERO_REVIEW_DIR` is set:
    ///
    /// `MC_AERO_REVIEW_DIR=<dir> cargo test -p mc-client --lib aero_title_review -- --ignored`
    #[test]
    #[ignore = "writes the review PNGs; run explicitly with MC_AERO_REVIEW_DIR"]
    fn aero_title_review_renders() {
        const REVIEW: (u32, u32) = (1920, 1080);
        let Some(out) = std::env::var_os("MC_AERO_REVIEW_DIR").map(std::path::PathBuf::from) else {
            eprintln!("skip: set MC_AERO_REVIEW_DIR to write the review images");
            return;
        };
        std::fs::create_dir_all(&out).expect("review dir");
        let Some(mut gpu) = gpu(REVIEW) else { return };
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, REVIEW.0, REVIEW.1);
        let mut camera = crate::app::driver::menu_camera();
        camera.set_viewport(REVIEW.0, REVIEW.1);
        // One frame creates the title screen (the session's first, so with the intro) and its
        // scene, and uploads their textures.
        let started = Instant::now();
        client.frame(&gpu, &mut camera, 0.0);
        eprintln!(
            "first title frame (bake + scene build): {:?}",
            started.elapsed()
        );
        // `MC_AERO_VHS=<0..1>`: render with another strength of the greenhouse's VHS finish (to
        // judge what it does to the UI).
        let vhs: Option<f32> = std::env::var("MC_AERO_VHS")
            .ok()
            .and_then(|v| v.parse().ok());
        let set_vhs = |client: &mut Client| {
            if let (Some(strength), Some(super::Backdrop::Greenhouse(scene))) = (
                vhs,
                client
                    .aero
                    .live
                    .as_mut()
                    .and_then(|live| live.scene.as_mut()),
            ) {
                scene.set_vhs_strength(strength);
            }
        };
        set_vhs(&mut client);
        assert!(
            client
                .aero
                .live
                .as_ref()
                .is_some_and(|live| live.scene.is_some()),
            "the greenhouse was built from the vanilla jar"
        );

        // `shown`: since the greenhouse appeared; `popin`: since the pop-in started. `revisit`:
        // leave and come back first (a later visit, without the intro).
        let mut shoot = |client: &mut Client,
                         (shown, popin): (Duration, Duration),
                         cursor: Option<AeroButton>,
                         revisit: bool,
                         name: &str| {
            if revisit {
                client.aero_title_leave();
                let started = Instant::now();
                client.frame(&gpu, &mut camera, 0.0);
                eprintln!(
                    "revisit first frame (scene rebuilt): {:?}",
                    started.elapsed()
                );
                set_vhs(client);
            }
            set_clock(client, shown, popin);
            let mut input = InputState::new();
            if let Some(button) = cursor {
                let (x, y) = button_centre(client, button);
                input.on_cursor_moved(f64::from(x), f64::from(y));
            }
            client.handle_screen_input(&mut input, REVIEW.0, REVIEW.1);
            // A few frames: everything uploaded, and the hover eased in (it eases on real time).
            for _ in 0..2 {
                client.frame(&gpu, &mut camera, 0.0);
                if cursor.is_some() {
                    std::thread::sleep(Duration::from_millis(250));
                }
                set_clock(client, shown, popin);
            }
            let started = Instant::now();
            client.frame(&gpu, &mut camera, 0.0);
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &camera, 0.0);
            let pixels = gpu.read_pixels().expect("readback");
            eprintln!("{name}: frame + render + readback {:?}", started.elapsed());
            let path = out.join(name);
            image::save_buffer(
                &path,
                &pixels,
                REVIEW.0,
                REVIEW.1,
                image::ExtendedColorType::Rgba8,
            )
            .expect("write the PNG");
            eprintln!("wrote {}", path.display());
        };
        let ms = Duration::from_millis;
        // The session's first title screen: the greenhouse fading in from its glow, the title and
        // buttons popping in 140 ms after their delayed start.
        shoot(
            &mut client,
            (super::INTRO_UI_DELAY + ms(140), ms(140)),
            None,
            false,
            "aero-title-1080p-intro.png",
        );
        // 12 s in: The recommended moment of the greenhouse's camera drift.
        let settled = (Duration::from_secs(12), Duration::from_secs(11));
        shoot(&mut client, settled, None, false, "aero-title-1080p.png");
        shoot(
            &mut client,
            settled,
            Some(AeroButton::Options),
            false,
            "aero-title-1080p-hover.png",
        );
        // The alternative title weight, for the user to choose (the first entry), or every
        // bundled weight with `MC_AERO_WEIGHT_STUDY` set.
        let weights: [(&str, &'static [u8]); 4] = [
            (
                "semibold",
                include_bytes!("../../../../mc-ui/assets/fonts/inter/Inter_28pt-SemiBold.ttf"),
            ),
            (
                "light",
                include_bytes!("../../../../mc-ui/assets/fonts/inter/Inter_28pt-Light.ttf"),
            ),
            (
                "regular",
                include_bytes!("../../../../mc-ui/assets/fonts/inter/Inter_28pt-Regular.ttf"),
            ),
            (
                "medium",
                include_bytes!("../../../../mc-ui/assets/fonts/inter/Inter_28pt-Medium.ttf"),
            ),
        ];
        let study = std::env::var_os("MC_AERO_WEIGHT_STUDY").is_some();
        let inter = client.aero.title_font.take();
        for (weight, bytes) in weights.into_iter().take(if study { 4 } else { 1 }) {
            client.aero.title_font = Some(mc_ui::screens::aero_title::font::TitleFont::from_bytes(
                bytes,
            ));
            let name = if study {
                format!("weight-{weight}.png")
            } else {
                format!("aero-title-1080p-alt-{weight}.png")
            };
            shoot(&mut client, settled, None, false, &name);
        }
        client.aero.title_font = inter;
        if let Some(font) = client.aero.title_font.as_mut() {
            font.mark_dirty();
        }
        // A later visit (back from Options): no intro, the pop-in 140 ms in.
        shoot(
            &mut client,
            (ms(140), ms(140)),
            None,
            true,
            "aero-title-1080p-popin.png",
        );
        // The cost at 1080p, with `MC_GPU_TIMING=1`: GPU time per phase over 120 settled frames,
        // and the CPU time of `build_ui` on the Aero title screen against vanilla's.
        if std::env::var_os("MC_GPU_TIMING").is_some() {
            for _ in 0..120 {
                client.frame(&gpu, &mut camera, 0.0);
                let target = gpu.acquire().expect("offscreen frame");
                phases.render(&gpu, target.view(), &camera, 0.0);
            }
            let _ = gpu.read_pixels();
            eprintln!("{}", phases.gpu_timings().format_table());
            let build_ui = |client: &mut Client| {
                for _ in 0..20 {
                    client.build_ui(&gpu, 0.0);
                }
                let started = Instant::now();
                for _ in 0..200 {
                    client.build_ui(&gpu, 0.0);
                }
                started.elapsed() / 200
            };
            let aero = build_ui(&mut client);
            client.mods.settings.set_enabled("aero_title", false);
            let vanilla = build_ui(&mut client);
            eprintln!("build_ui CPU per frame: Aero title {aero:?}, vanilla title {vanilla:?}");
        }
    }

    /// The logo in every "Logo style", over the real title screen
    /// (greenhouse, glass buttons, finish) at 1920x1080, through `Client::frame` and the installed
    /// phases. Writes one PNG per style and `logo-gallery.png` (the top-left of each, labelled) to
    /// `MC_LOGO_GALLERY_DIR`:
    ///
    /// `MC_LOGO_GALLERY_DIR=<dir> cargo test -p mc-client --lib logo_gallery -- --ignored`
    #[test]
    #[ignore = "writes the logo gallery PNGs; run explicitly with MC_LOGO_GALLERY_DIR"]
    fn logo_gallery_renders() {
        use mc_ui::screens::aero_title::font::{LogoStyle, TitleFont};
        const REVIEW: (u32, u32) = (1920, 1080);
        let Some(out) = std::env::var_os("MC_LOGO_GALLERY_DIR").map(std::path::PathBuf::from)
        else {
            eprintln!("skip: set MC_LOGO_GALLERY_DIR to write the gallery");
            return;
        };
        std::fs::create_dir_all(&out).expect("gallery dir");
        let Some(mut gpu) = gpu(REVIEW) else { return };
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, REVIEW.0, REVIEW.1);
        let mut camera = crate::app::driver::menu_camera();
        camera.set_viewport(REVIEW.0, REVIEW.1);
        client.frame(&gpu, &mut camera, 0.0);
        // The top-left of the frame: the logo and the first buttons.
        let crop = (0_u32, 40_u32, 960_u32, 500_u32);
        let mut gallery = image::RgbaImage::new(crop.2 * 2, crop.3 * 3);
        let mut label_font = TitleFont::for_style(LogoStyle::Classic);
        for (index, style) in LogoStyle::ALL.into_iter().enumerate() {
            let _ = client.mods.settings.set_value(
                "aero_title",
                "logo_style",
                mc_mods::OptionValue::Choice(index),
            );
            assert_eq!(
                client.mods.settings.choice("aero_title", "logo_style"),
                index
            );
            for _ in 0..2 {
                client.frame(&gpu, &mut camera, 0.0);
                set_clock(
                    &mut client,
                    Duration::from_secs(12),
                    Duration::from_secs(11),
                );
            }
            assert_eq!(
                client.aero.title_font.as_ref().map(TitleFont::style),
                Some(style),
                "the title screen follows the option"
            );
            client.frame(&gpu, &mut camera, 0.0);
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &camera, 0.0);
            let pixels = gpu.read_pixels().expect("readback");
            drop(target);
            let name = format!(
                "logo-{}.png",
                style.label().to_lowercase().replace(' ', "-")
            );
            let frame = image::RgbaImage::from_raw(REVIEW.0, REVIEW.1, pixels).expect("frame");
            frame.save(out.join(&name)).expect("write the PNG");
            eprintln!("wrote {}", out.join(&name).display());
            // Into the gallery, labelled with the option's name in its bottom-right corner.
            let mut cell =
                image::imageops::crop_imm(&frame, crop.0, crop.1, crop.2, crop.3).to_image();
            if label_font.prepare(style.label(), 34.0) {
                if let Some((rgba, w, h)) = label_font.take_upload() {
                    // The sharp image is the top third of the texture (see `TitleFont`).
                    let rows = (h - 4) / 3;
                    let (x0, y0) = (
                        crop.2.saturating_sub(w + 16),
                        crop.3.saturating_sub(rows + 6),
                    );
                    for y in 0..rows.min(crop.3) {
                        for x in 0..w.min(crop.2) {
                            let a = f32::from(rgba[((y * w + x) * 4 + 3) as usize]) / 255.0;
                            if a > 0.0 && x0 + x < crop.2 && y0 + y < crop.3 {
                                let p = cell.get_pixel_mut(x0 + x, y0 + y);
                                for k in 0..3 {
                                    p.0[k] = (f32::from(p.0[k]) * (1.0 - a) + 255.0 * a) as u8;
                                }
                            }
                        }
                    }
                }
            }
            let (gx, gy) = ((index as u32 % 2) * crop.2, (index as u32 / 2) * crop.3);
            image::imageops::replace(&mut gallery, &cell, i64::from(gx), i64::from(gy));
        }
        gallery
            .save(out.join("logo-gallery.png"))
            .expect("write the gallery");
        eprintln!("wrote {}", out.join("logo-gallery.png").display());
    }

    // ---- The three title-screen scenes ---------------------

    use super::super::nw_menu::{MenuKind, OPT_MENU_SCENE};
    use super::Backdrop;

    fn background_filled(client: &Client) -> bool {
        let slots = client.aero.slots.as_ref().expect("installed");
        slots.background.with(|slot| slot.is_filled())
    }

    fn finish_filled(client: &Client) -> bool {
        let slots = client.aero.slots.as_ref().expect("installed");
        slots.finish.with(|slot| slot.is_filled())
    }

    fn live_kind(client: &Client) -> Option<MenuKind> {
        client.aero.live.as_ref().map(|live| live.kind)
    }

    fn backdrop_kind(client: &Client) -> Option<MenuKind> {
        client
            .aero
            .live
            .as_ref()
            .and_then(|live| live.scene.as_ref())
            .map(Backdrop::kind)
    }

    fn render(client: &mut Client, gpu: &mut Gpu, phases: &mut PhaseList) -> Vec<u8> {
        client.build_ui(gpu, 0.0);
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(gpu, target.view(), &mc_render::Camera::default(), 0.0);
        drop(target);
        gpu.read_pixels().expect("readback")
    }

    /// Pixels whose colour differs between two frames.
    fn differing(a: &[u8], b: &[u8]) -> usize {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(p, q)| p[..3] != q[..3])
            .count()
    }

    /// The "Menu scene" option through the real client: a client without a launch shows the
    /// greenhouse; Glass and Shore draw in the background slot with no finish and no crisp
    /// layer; a change switches the scene on the next frame and parks the old one (a switch back
    /// reuses it, built once); leaving the title screen parks the scene and stops its clock;
    /// a minute away gives its screen-sized memory back, and it is rebuilt on return; the frame
    /// cap stays the monitor's rate; the mod off is vanilla's title screen.
    #[test]
    fn each_menu_scene_draws_behind_the_title_and_the_option_switches_it_live() {
        let Some(mut gpu) = gpu(SIZE) else { return };
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, SIZE.0, SIZE.1);

        // No launch (`Client::new`): Random is the greenhouse, exactly as before this change.
        let greenhouse = render(&mut client, &mut gpu, &mut phases);
        assert_eq!(live_kind(&client), Some(MenuKind::Greenhouse));
        assert_eq!(backdrop_kind(&client), Some(MenuKind::Greenhouse));
        assert!(background_filled(&client) && finish_filled(&client));
        assert_eq!(client.aero_title_scene_builds(), 1);

        // Glass: switched on the next frame; the greenhouse parked; no finish.
        client
            .mods
            .settings
            .set_choice("aero_title", OPT_MENU_SCENE, 2);
        let glass = render(&mut client, &mut gpu, &mut phases);
        assert_eq!(backdrop_kind(&client), Some(MenuKind::Glass));
        assert!(background_filled(&client), "the scene draws");
        assert!(!finish_filled(&client), "Glass has no VHS finish");
        assert!(
            !client
                .aero
                .live
                .as_ref()
                .expect("live")
                .crisp
                .with(|phase| phase.is_active()),
            "no crisp layer without a finish"
        );
        assert_eq!(
            client.aero.parked.as_ref().map(|p| p.scene.kind()),
            Some(MenuKind::Greenhouse)
        );
        assert_eq!(client.aero_title_scene_builds(), 2);
        let settings = client.settings.clone();
        client.monitor_refresh_changed(Some(143_856));
        assert_eq!(client.effective_frame_cap(&settings, true), Some(144));

        // Shore: the glass parked (the greenhouse dropped), a third picture.
        client
            .mods
            .settings
            .set_choice("aero_title", OPT_MENU_SCENE, 3);
        let shore = render(&mut client, &mut gpu, &mut phases);
        assert_eq!(backdrop_kind(&client), Some(MenuKind::Shore));
        assert_eq!(client.aero_title_scene_builds(), 3);
        // Three different pictures, each over most of the frame.
        let total = (SIZE.0 * SIZE.1) as usize;
        for (a, b, what) in [
            (&greenhouse, &glass, "greenhouse vs glass"),
            (&glass, &shore, "glass vs shore"),
            (&greenhouse, &shore, "greenhouse vs shore"),
        ] {
            let n = differing(a, b);
            assert!(n > total / 2, "{what}: only {n} of {total} pixels differ");
        }

        // Back to Glass: the parked one, not built again.
        client
            .mods
            .settings
            .set_choice("aero_title", OPT_MENU_SCENE, 2);
        let _ = render(&mut client, &mut gpu, &mut phases);
        assert_eq!(backdrop_kind(&client), Some(MenuKind::Glass));
        assert_eq!(client.aero_title_scene_builds(), 3, "reused");

        // Away (the mod off: vanilla's title screen): parked with its clock stopped.
        for _ in 0..3 {
            let _ = render(&mut client, &mut gpu, &mut phases);
        }
        let clock = |client: &mut Client| {
            client
                .aero
                .live
                .as_mut()
                .and_then(|live| live.scene.as_mut())
                .and_then(Backdrop::as_scene_mut)
                .map(|scene| scene.clock())
        };
        let before = clock(&mut client).expect("the glass is up");
        client.mods.settings.set_enabled("aero_title", false);
        let _ = render(&mut client, &mut gpu, &mut phases);
        assert!(client.aero.live.is_none());
        assert_eq!(
            client.aero.parked.as_ref().map(|p| p.scene.kind()),
            Some(MenuKind::Glass)
        );
        assert!(!background_filled(&client));
        std::thread::sleep(Duration::from_millis(120));
        client.mods.settings.set_enabled("aero_title", true);
        let _ = render(&mut client, &mut gpu, &mut phases);
        assert_eq!(backdrop_kind(&client), Some(MenuKind::Glass));
        assert_eq!(
            client.aero_title_scene_builds(),
            3,
            "the parked glass came back"
        );
        let after = clock(&mut client).expect("the glass is up");
        assert!(
            (after - before).abs() < 1e-6,
            "the clock stood still while away: {before} -> {after}"
        );

        // A minute away: the screen-sized memory goes back; the scene is rebuilt on return.
        client.mods.settings.set_enabled("aero_title", false);
        let _ = render(&mut client, &mut gpu, &mut phases);
        client.aero.parked.as_mut().expect("parked").since -= super::PARKED_RELEASE_AFTER;
        let _ = render(&mut client, &mut gpu, &mut phases);
        let parked = client.aero.parked.as_ref().expect("still parked");
        assert!(parked.released && parked.scene.gpu_memory_released());
        client.mods.settings.set_enabled("aero_title", true);
        let back = render(&mut client, &mut gpu, &mut phases);
        assert!(!client
            .aero
            .live
            .as_ref()
            .and_then(|live| live.scene.as_ref())
            .expect("the glass")
            .gpu_memory_released());
        assert!(
            differing(&back, &shore) > total / 2,
            "the glass draws again"
        );
        assert_eq!(client.aero_title_scene_builds(), 3);
    }

    /// The pictures for the user: the three title-screen scenes with the logo and the buttons, at
    /// 1920x1080 and 2560x1440, through `Client::frame` and the installed phases. Writes
    /// `menu-<scene>-<w>x<h>.png` (and `-backdrop` without the title UI) to `MC_MENU_PNG_DIR`
    /// and prints readability figures: the backdrop's luminance behind the logo, and the
    /// contrast of the white labels on each glass button. With `MC_GPU_TIMING=1` it also prints
    /// the GPU time per phase over 240 frames of each scene. `MC_MENU_SIZES`, `MC_AERO_VHS`,
    /// `MC_AERO_GLOW` and `MC_AERO_BLOOM` change the sizes and the greenhouse's finish (below).
    ///
    /// `MC_MENU_PNG_DIR=<dir> cargo test -p mc-client --lib menu_scene_pictures -- --ignored`
    #[test]
    #[ignore = "writes the review PNGs; run by hand with MC_MENU_PNG_DIR"]
    fn menu_scene_pictures() {
        let Some(out) = std::env::var_os("MC_MENU_PNG_DIR").map(std::path::PathBuf::from) else {
            eprintln!("skip: set MC_MENU_PNG_DIR");
            return;
        };
        std::fs::create_dir_all(&out).expect("output dir");
        let timing = std::env::var_os("MC_GPU_TIMING").is_some();
        // Relative luminance of an sRGB-encoded pixel.
        let luma = |p: &[u8]| {
            let lin = |c: u8| {
                let c = f32::from(c) / 255.0;
                if c <= 0.04045 {
                    c / 12.92
                } else {
                    ((c + 0.055) / 1.055).powf(2.4)
                }
            };
            0.2126 * lin(p[0]) + 0.7152 * lin(p[1]) + 0.0722 * lin(p[2])
        };
        // `MC_MENU_ONLY=<scene>` and `MC_MENU_ONLY_1080` narrow a tuning run.
        let only = std::env::var("MC_MENU_ONLY").ok();
        let only_1080 = std::env::var_os("MC_MENU_ONLY_1080").is_some();
        // `MC_MENU_SIZES=<w>x<h>,...`: other frame sizes (a taller frame for a web page, say).
        let sizes: Vec<(u32, u32)> = std::env::var("MC_MENU_SIZES")
            .ok()
            .map(|list| {
                list.split(',')
                    .filter_map(|size| {
                        let (w, h) = size.trim().split_once('x')?;
                        Some((w.parse().ok()?, h.parse().ok()?))
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![(1920, 1080), (2560, 1440)]);
        for size in sizes {
            if only_1080 && size.0 != 1920 {
                continue;
            }
            let Some(mut gpu) = gpu(size) else { return };
            let Some((mut client, _dir)) = offline_client() else {
                return;
            };
            let mut phases = PhaseList::new();
            client.install(&gpu, &mut phases, size.0, size.1);
            let mut camera = crate::app::driver::menu_camera();
            camera.set_viewport(size.0, size.1);
            for (index, kind, moment) in [
                (1, MenuKind::Greenhouse, 12.0_f32),
                (2, MenuKind::Glass, 3.0),
                (3, MenuKind::Shore, 1.6),
            ] {
                if only.as_deref().is_some_and(|only| only != kind.name()) {
                    continue;
                }
                client
                    .mods
                    .settings
                    .set_choice("aero_title", OPT_MENU_SCENE, index);
                for _ in 0..200 {
                    client.frame(&gpu, &mut camera, 0.0);
                    if backdrop_kind(&client) == Some(kind) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                assert_eq!(backdrop_kind(&client), Some(kind), "switched");
                // Tuning without a rebuild: `MC_MENU_SCRIM_GLASS`, `MC_MENU_SCRIM_SHORE`,
                // `MC_MENU_GLASS_SPOT=x,y` (the scene's own coordinates) and
                // `MC_MENU_GLASS_MIRROR=0` rebuild the scene with another look or style.
                if let Some(scene_kind) = kind.scene() {
                    let mut style = kind.style();
                    let mut look = super::super::nw_menu::menu_glass_look();
                    let mut changed = false;
                    let scrim_var = if kind == MenuKind::Glass {
                        "MC_MENU_SCRIM_GLASS"
                    } else {
                        "MC_MENU_SCRIM_SHORE"
                    };
                    if let Some(scrim) = std::env::var(scrim_var).ok().and_then(|v| v.parse().ok())
                    {
                        style.scrim = scrim;
                        changed = true;
                    }
                    if kind == MenuKind::Glass {
                        if let Ok(v) = std::env::var("MC_MENU_GLASS_SPOT") {
                            let p: Vec<f32> =
                                v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                            if let [x, y] = p[..] {
                                look.spot = [x, y];
                                changed = true;
                            }
                        }
                        if std::env::var("MC_MENU_GLASS_MIRROR").as_deref() == Ok("0") {
                            style.mirror = false;
                            changed = true;
                        }
                    }
                    if changed {
                        let jar = client.config.version_jar.clone();
                        let textures = jar.as_deref().map_or_else(
                            mc_render::loading_scene::JarTextures::empty,
                            |jar| {
                                mc_render::loading_scene::JarTextures::load(
                                    jar,
                                    scene_kind.jar_textures(),
                                )
                            },
                        );
                        let seed = client.aero.menu.seed();
                        let backdrop = super::super::nw_menu::SceneBackdrop::with_look(
                            &gpu, kind, &textures, seed, look, style,
                        )
                        .expect("a scene");
                        if let Some(slots) = &client.aero.slots {
                            slots.background.with(|slot| slot.fill(backdrop.phase()));
                        }
                        if let Some(live) = client.aero.live.as_mut() {
                            live.scene = Some(Backdrop::Scene(backdrop));
                        }
                        eprintln!("override: {kind:?} spot {:?} {style:?}", look.spot);
                    }
                }
                // `MC_AERO_VHS=<0..1>`, `MC_AERO_GLOW=<0..1>` and `MC_AERO_BLOOM=<f>`: the
                // greenhouse with another strength of its VHS finish, Orton glow or bloom.
                if let Some(Backdrop::Greenhouse(scene)) =
                    client.aero.live.as_mut().and_then(|l| l.scene.as_mut())
                {
                    let var = |name: &str| std::env::var(name).ok().and_then(|v| v.parse().ok());
                    if let Some(strength) = var("MC_AERO_VHS") {
                        scene.set_vhs_strength(strength);
                    }
                    let mut look = scene.look();
                    if let Some(glow) = var("MC_AERO_GLOW") {
                        look.glow = glow;
                    }
                    if let Some(bloom) = var("MC_AERO_BLOOM") {
                        look.bloom = bloom;
                    }
                    scene.set_look(look);
                }
                let shoot = |client: &mut Client,
                             gpu: &mut Gpu,
                             phases: &mut PhaseList,
                             camera: &mut mc_render::Camera| {
                    if let Some(scene) = client.aero.live.as_mut().and_then(|l| l.scene.as_mut()) {
                        match scene {
                            Backdrop::Greenhouse(scene) => {
                                scene.wait_for_clouds(gpu, Duration::from_secs(20));
                            }
                            Backdrop::Scene(scene) => scene.set_clock(moment),
                        }
                    }
                    set_clock(
                        client,
                        Duration::from_secs_f32(moment),
                        Duration::from_secs(11),
                    );
                    client.frame(gpu, camera, 0.0);
                    let target = gpu.acquire().expect("offscreen frame");
                    phases.render(gpu, target.view(), camera, 0.0);
                    let pixels = gpu.read_pixels().expect("readback");
                    drop(target);
                    pixels
                };
                // A few frames: everything uploaded and the pop-in done.
                for _ in 0..3 {
                    let _ = shoot(&mut client, &mut gpu, &mut phases, &mut camera);
                }
                let pixels = shoot(&mut client, &mut gpu, &mut phases, &mut camera);
                let tag = std::env::var("MC_MENU_TAG").unwrap_or_default();
                let name = format!("menu-{}-{}x{}{tag}.png", kind.name(), size.0, size.1);
                image::save_buffer(
                    out.join(&name),
                    &pixels,
                    size.0,
                    size.1,
                    image::ExtendedColorType::Rgba8,
                )
                .expect("write the PNG");
                eprintln!("wrote {}", out.join(&name).display());

                // The labels' contrast: white on each button's glass (its right part, where
                // there is no icon, label or chevron).
                let rects: Vec<_> = AeroButton::ALL
                    .iter()
                    .map(|&button| {
                        client
                            .aero
                            .live
                            .as_ref()
                            .and_then(|live| live.screen.as_ref())
                            .expect("the title screen")
                            .button_rect(button)
                    })
                    .collect();
                let mut worst = f32::INFINITY;
                for rect in &rects {
                    let (mut sum, mut n) = (0.0_f32, 0_u32);
                    let x0 = (rect.x + rect.w * 0.62) as u32;
                    let x1 = (rect.x + rect.w * 0.85) as u32;
                    let y0 = (rect.y + rect.h * 0.25) as u32;
                    let y1 = (rect.y + rect.h * 0.75) as u32;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let i = ((y * size.0 + x) * 4) as usize;
                            sum += luma(&pixels[i..i + 4]);
                            n += 1;
                        }
                    }
                    let mean = sum / n.max(1) as f32;
                    worst = worst.min((1.0 + 0.05) / (mean + 0.05));
                }

                // The logo's backdrop: the frame without the title UI, in the logo's box.
                for layer in ["aero_title", "aero_title/crisp"] {
                    phases.remove_named(layer);
                }
                client.frame(&gpu, &mut camera, 0.0);
                let target = gpu.acquire().expect("offscreen frame");
                phases.render(&gpu, target.view(), &camera, 0.0);
                let bare = gpu.read_pixels().expect("readback");
                drop(target);
                let name = format!(
                    "menu-{}-{}x{}{tag}-backdrop.png",
                    kind.name(),
                    size.0,
                    size.1
                );
                image::save_buffer(
                    out.join(&name),
                    &bare,
                    size.0,
                    size.1,
                    image::ExtendedColorType::Rgba8,
                )
                .expect("write the PNG");
                let s = mc_ui::screens::aero_title::layout::scale((size.0 as f32, size.1 as f32));
                let (lx0, lx1) = ((128.0 * s) as u32, (700.0 * s) as u32);
                let mid = size.1 as f32 / 8.0;
                let (ly0, ly1) = ((mid - 45.0 * s) as u32, (mid + 45.0 * s) as u32);
                let mut lumas = Vec::new();
                for y in ly0..ly1 {
                    for x in lx0..lx1 {
                        let i = ((y * size.0 + x) * 4) as usize;
                        lumas.push(luma(&bare[i..i + 4]));
                    }
                }
                lumas.sort_by(f32::total_cmp);
                let p50 = lumas[lumas.len() / 2];
                let p90 = lumas[lumas.len() * 9 / 10];
                eprintln!(
                    "readability {} {}x{}: logo backdrop luminance p50 {p50:.3} p90 {p90:.3} \
                     (white on p90: {:.1}:1); worst button label contrast {worst:.1}:1",
                    kind.name(),
                    size.0,
                    size.1,
                    1.05 / (p90 + 0.05),
                );
                // Back: new slots (the old two out first), the scene parked and taken again on
                // the next frame.
                for layer in ["aero_title/background", "aero_title/finish"] {
                    phases.remove_named(layer);
                }
                client.aero_title_install_phases(&mut phases);
                client.frame(&gpu, &mut camera, 0.0);

                if timing {
                    // This scene's frames only: the difference of the running totals.
                    let before = phases.gpu_timings();
                    for _ in 0..240 {
                        let _ = shoot(&mut client, &mut gpu, &mut phases, &mut camera);
                    }
                    let after = phases.gpu_timings();
                    let mean = |name: &str, a: &mc_render::pipeline::GpuTimings| {
                        a.scopes
                            .iter()
                            .find(|(n, _)| *n == name)
                            .map_or((0, 0), |(_, s)| (s.total_ns, s.calls))
                    };
                    let frame = (after.frame_ns.total_ns - before.frame_ns.total_ns) as f64
                        / (after.frame_ns.calls - before.frame_ns.calls).max(1) as f64;
                    let mut line = format!(
                        "GPU {} {}x{}: frame {:.3} ms;",
                        kind.name(),
                        size.0,
                        size.1,
                        frame / 1e6
                    );
                    for (name, _) in &after.scopes {
                        let (t1, c1) = mean(name, &after);
                        let (t0, c0) = mean(name, &before);
                        if c1 > c0 {
                            let ms = (t1 - t0) as f64 / (c1 - c0) as f64 / 1e6;
                            if ms >= 0.005 {
                                line.push_str(&format!(" {name} {ms:.3}"));
                            }
                        }
                    }
                    eprintln!("{line}");
                }
            }
        }
    }
}
