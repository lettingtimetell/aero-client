//! The loading screen's client half: the Aero logo and a slim progress bar over a calm dark
//! backdrop while the client loads, then a short fade into the title screen.
//!
//! **History.** An earlier version showed an animated scene here (Glass or
//! Shore) for at least two seconds. Requested: "loading screens were
//! supposed to be main menu screens, it just increases loading time like this. please make it
//! into 3 randomly selected main menus". The scenes moved to the title screen (`nw_menu.rs`,
//! `aero_title.rs`) and this screen is fast again: no scene, no jar read, no minimum
//! time. It is gone the moment the game has loaded, fading out over the live title screen.
//!
//! Like every file under `app/mods/`, this module is declared inside `app::driver`, so it can
//! read `Client`'s private fields.
//!
//! # What happens
//!
//! With the Aero Title Screen mod on (its default), a deferred start (the game binary):
//!
//! 1. **Boot** ([`Client::tw_loading_boot`], from `Client::new_deferred`): plan the screen.
//! 2. **First frame** ([`Client::tw_loading_install`], from `install_loading_ui`): build the
//!    overlay phase (the title screen's own content phase: its logo code and painter) above every
//!    other phase. Nothing else: one small UI pass per frame.
//! 3. **While loading**: one asset stage per frame runs exactly as before; the overlay covers the
//!    frame with a dark backdrop, the logo (in the user's "Logo style") and the progress bar, and
//!    the dirt screen under it is not drawn. The frame cap is the monitor's refresh rate, like the
//!    title screen's (vanilla's 30 FPS menu cap would idle up to 33 ms between stages; vanilla's
//!    own `startGame` loads in one go, with no limiter at all).
//! 4. **Loaded**: the game switches to the title screen exactly as before, underneath, and the
//!    overlay starts its fade ([`FADE_OUT`]) on that same frame; the title screen takes the
//!    input from then on. The fade's clock steps by at most `MAX_STEP_S` a frame, so the title
//!    screen's first frame (which builds its scene) cannot make it jump.
//! 5. **Gone**: the phase is removed from the phase list and dropped. From then on every hook is
//!    one `Option` test.
//!
//! The moment a world is on screen (a `--server` join straight from the load, a replay opened,
//! an export rendering one), it is cut short: off on the world's first frame, out of the phase
//! list on the next (the ghost-logo fix). The mod off, `--replay`
//! and diagnostic runs never build anything: vanilla's loading screen is drawn exactly as before.
//!
//! # Hooks in `driver.rs` (one line each)
//! * `fields` / `init`: [`TwLoadingState`] in `Client::tw_loading`.
//! * `boot`: [`Client::tw_loading_boot`] at the end of `Client::new_deferred`.
//! * `install`: [`Client::tw_loading_install`] in `install_loading_ui` and in `install` (the
//!   phase list is rebuilt when loading finishes; the overlay goes back in while it fades).
//! * `shell-frame`: [`Client::tw_loading_shell_frame`] at the top of `ShellApp::frame`: once the
//!   fade is over, removing the phase.
//! * `draw`: [`Client::tw_loading_frame`] in `build_ui`, before the title screen; while it covers
//!   the screen the dirt screen is not drawn under it.
//! * `frame-cap`: [`Client::tw_loading_frame_cap`] in `ShellApp::effective_frame_cap`.
//! * in `app/mods/aero_title.rs`: [`Client::tw_loading_holds_title`] in `build_aero_title`.

use std::time::{Duration, Instant};

use mc_render::loading_scene::{self, MAX_STEP_S};
use mc_render::{Camera, Gpu, PhaseList, RenderPhase, SceneTextures};
use mc_ui::screens::aero_title::font::{LogoStyle, TitleFont};
use mc_ui::screens::aero_title::phase::AeroTitlePhase;
use mc_ui::screens::aero_title::TITLE;
use mc_ui::screens::modmenu::glass::GlassFrame;
use mc_ui::screens::modmenu::paint::{Fill, Painter, Rect};

use super::super::{Client, Shared};

/// The Aero Title Screen mod (on: this loading screen; off: vanilla's).
pub(crate) const MOD_ID: &str = "aero_title";

/// The fade into the title screen (the earlier was 0.7 s; quicker now that nothing holds
/// the screen up).
pub(crate) const FADE_OUT: Duration = Duration::from_millis(500);
/// [`RenderPhase::name`] of the overlay (the backdrop, the logo and the bar).
pub(crate) const OVERLAY_PHASE_NAME: &str = "loading/overlay";
/// The backdrop's top and bottom colours: a calm, cool near-black, a little lighter at the top,
/// that none of the three title-screen scenes clashes with.
const BACKDROP_TOP: [f32; 4] = [0.071, 0.078, 0.106, 1.0];
/// See [`BACKDROP_TOP`].
const BACKDROP_BOTTOM: [f32; 4] = [0.035, 0.039, 0.055, 1.0];

/// The overlay's phase: the title screen's content phase (the logo bitmap and the painter's
/// shapes in physical pixels, no glass) under its own name.
#[derive(Debug)]
pub(crate) struct OverlayPhase(AeroTitlePhase);

impl RenderPhase for OverlayPhase {
    fn name(&self) -> &'static str {
        OVERLAY_PHASE_NAME
    }

    fn enabled(&self) -> bool {
        self.0.enabled()
    }

    fn needs_scene_color(&self) -> bool {
        self.0.needs_scene_color()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.0.record(gpu, enc, view, cam);
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        self.0.record_scene(gpu, enc, view, cam, scene);
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        self.0.resize(gpu, width, height);
    }
}

/// The overlay while it is up.
struct Live {
    overlay: Shared<OverlayPhase>,
    font: TitleFont,
    /// The settings generation the logo style was read at.
    style_seen: u64,
    /// The first frame drawn.
    started: Option<Instant>,
    last: Option<Instant>,
    /// Loading progress as shown, `0..=1`: eased, never going back.
    progress: f32,
    loaded_at: Option<Instant>,
    /// Seconds into the fade into the title screen (steps of at most [`MAX_STEP_S`]); `None`
    /// until loading has finished.
    fade: Option<f32>,
    /// Whether a world on screen cut it short.
    cut_short: bool,
    /// The fade is over: remove the phase on the next shell frame.
    finished: bool,
    /// Frames drawn.
    frames: u64,
}

impl std::fmt::Debug for Live {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Live")
            .field("progress", &self.progress)
            .field("fade", &self.fade)
            .field("finished", &self.finished)
            .field("frames", &self.frames)
            .finish_non_exhaustive()
    }
}

/// What the loading screen did, for the log and the tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LoadingReport {
    /// Frames it drew.
    pub(crate) frames: u64,
    /// From its first frame to loading done.
    pub(crate) loading: Duration,
    /// Whether a world on screen cut it short.
    pub(crate) cut_short: bool,
}

/// The loading screen's client state (`Client::tw_loading`). Empty unless the screen was planned
/// for this start.
#[derive(Debug, Default)]
pub(crate) struct TwLoadingState {
    planned: bool,
    live: Option<Live>,
    report: Option<LoadingReport>,
}

/// Smootherstep over `0..=1`.
fn ease(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

/// The overlay's layout at `(w, h)` physical pixels: the logo's em size, its caps' centre line,
/// and the progress bar's rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OverlayLayout {
    pub(crate) scale: f32,
    pub(crate) logo_size: f32,
    pub(crate) caps_center_y: f32,
    pub(crate) bar: Rect,
}

impl OverlayLayout {
    pub(crate) fn at(w: f32, h: f32) -> OverlayLayout {
        // Physical pixels per design pixel: 1080p is 1 (like the title screen's layout).
        let scale = (h / 1080.0).min(w / 1536.0).clamp(0.4, 4.0);
        let logo_size = 112.0 * scale;
        let caps_center_y = (h * 0.43).round();
        let bar_w = (260.0 * scale).round();
        let bar_h = (3.0 * scale).max(2.0).round();
        let bar = Rect::new(
            ((w - bar_w) * 0.5).round(),
            (caps_center_y + 92.0 * scale).round(),
            bar_w,
            bar_h,
        );
        OverlayLayout {
            scale,
            logo_size,
            caps_center_y,
            bar,
        }
    }
}

impl Client {
    /// Plan the loading screen for this start: a deferred start (the game
    /// binary) headed for the title screen with the Aero Title Screen mod on. Nothing is read or
    /// started here (the title screen's scene is prepared by `nw_menu_boot`).
    pub(crate) fn tw_loading_boot(&mut self) {
        if self.loader.is_none()
            || self.config.replay.is_some()
            || self.diagnostic_progress.is_some()
            || !self.mods.settings.enabled(MOD_ID)
        {
            return;
        }
        tracing::info!("loading screen planned");
        self.tw_loading.planned = true;
    }

    /// Put the overlay into `phases`, building it the first time (the first
    /// loading frame). Also called from `install`, which rebuilds the phase list when loading
    /// finishes: the overlay goes back in on top while it fades. Nothing without a plan.
    pub(crate) fn tw_loading_install(&mut self, gpu: &Gpu, phases: &mut PhaseList) {
        if self.tw_loading.live.is_none() {
            if !std::mem::take(&mut self.tw_loading.planned) {
                return;
            }
            let started = Instant::now();
            let overlay = Shared::new(OverlayPhase(AeroTitlePhase::new(gpu)));
            tracing::info!(
                ms = started.elapsed().as_secs_f64() * 1000.0,
                "loading screen built"
            );
            self.tw_loading.live = Some(Live {
                overlay,
                font: TitleFont::embedded(),
                style_seen: 0,
                started: None,
                last: None,
                progress: 0.0,
                loaded_at: None,
                fade: None,
                cut_short: false,
                finished: false,
                frames: 0,
            });
        }
        let Some(live) = self.tw_loading.live.as_mut() else {
            return;
        };
        if live.finished {
            return;
        }
        // Idempotent: never two copies in one list.
        phases.remove_named(OVERLAY_PHASE_NAME);
        phases.insert(loading_scene::OVERLAY_ORDER, Box::new(live.overlay.clone()));
        // A new renderer: the logo bitmap goes up again.
        live.font.mark_dirty();
    }

    /// At the top of `ShellApp::frame`. Once the fade is over it removes
    /// the phase and drops the overlay. One `Option` test when there is no loading screen.
    /// (The earlier screen also took clicks and keys here while it held the title screen
    /// hidden; nothing is hidden behind this one, so input is left alone.)
    pub(crate) fn tw_loading_shell_frame(
        &mut self,
        phases: &mut PhaseList,
        _input: &mut mc_render::InputState,
    ) {
        let Some(live) = self.tw_loading.live.as_ref() else {
            return;
        };
        if !live.finished {
            return;
        }
        phases.remove_named(OVERLAY_PHASE_NAME);
        let report = LoadingReport {
            frames: live.frames,
            loading: match (live.started, live.loaded_at) {
                (Some(start), Some(done)) => done.saturating_duration_since(start),
                _ => Duration::ZERO,
            },
            cut_short: live.cut_short,
        };
        tracing::info!(
            frames = report.frames,
            loading_ms = report.loading.as_millis() as u64,
            cut_short = report.cut_short,
            "loading screen done"
        );
        self.tw_loading.report = Some(report);
        self.tw_loading.live = None;
    }

    /// The monitor's refresh rate while the loading screen is up (like the
    /// Aero title screen); `None` otherwise.
    pub(crate) fn tw_loading_frame_cap(&self) -> Option<u32> {
        self.tw_loading
            .live
            .as_ref()
            .map(|_| self.aero_title_refresh_cap())
    }

    /// Whether the loading screen still covers everything (it has not started fading). Only
    /// true while the game loads: the fade starts on the frame loading finishes, so the title
    /// screen's clock (the greenhouse's intro, the buttons' pop-in) starts with it. One `Option`
    /// test.
    pub(crate) fn tw_loading_holds_title(&self) -> bool {
        self.tw_loading
            .live
            .as_ref()
            .is_some_and(|live| live.fade.is_none() && !live.finished)
    }

    /// The report of the loading screen this run showed, once it is gone.
    #[cfg(test)]
    pub(crate) fn tw_loading_report(&self) -> Option<LoadingReport> {
        self.tw_loading.report
    }

    /// Advance the loading screen by one frame (from `build_ui`, before the title
    /// screen draws). Returns `true` while it covers the whole screen, so the dirt screen under it
    /// is not drawn. One `Option` test when there is no loading screen.
    pub(crate) fn tw_loading_frame(&mut self, gpu: &Gpu) -> bool {
        if self.tw_loading.live.is_none() {
            return false;
        }
        let now = Instant::now();
        let loaded = !self.is_loading();
        // A world on screen (a server joined, a replay opened, an export rendering one) ends the
        // loading screen at once. It hands over to the title screen and the menus around it,
        // never to a world: a replay opened during the earlier fade once played under the
        // fading loading screen and its logo, a "ghost logo" in its frames.
        let world_on_screen = self.machine.state().is_in_world() || self.replay_is_active();
        let target = match &self.loader {
            Some(loader) => loader.percent().clamp(0, 100) as f32 / 100.0,
            None => 1.0,
        };
        // The logo style: read again only when the settings changed.
        let generation = self.mods.settings.generation();
        let style = self
            .tw_loading
            .live
            .as_ref()
            .is_some_and(|live| live.style_seen != generation)
            .then(|| LogoStyle::from_settings(&self.mods.settings));
        let (w, h) = (gpu.config.width as f32, gpu.config.height as f32);
        let Some(live) = self.tw_loading.live.as_mut() else {
            return false;
        };
        if live.finished {
            return false;
        }

        // The clock: real time in steps of at most `MAX_STEP_S`, so a slow frame (the title
        // screen's first, which builds its scene) cannot make the fade jump.
        live.started.get_or_insert(now);
        let step = live
            .last
            .map_or(0.0, |last| {
                now.saturating_duration_since(last).as_secs_f32()
            })
            .min(MAX_STEP_S);
        live.last = Some(now);
        live.frames += 1;

        // Loading progress, eased toward the loader's figure and never going back; full once
        // loaded, and the fade starts on that frame (still fully covered).
        let gap = (target - live.progress).max(0.0);
        live.progress += gap * (1.0 - (-step * 9.0).exp());
        if loaded {
            live.loaded_at.get_or_insert(now);
            live.progress = 1.0;
            live.fade = Some(live.fade.map_or(0.0, |fade| fade + step));
        }
        let mut opacity = live
            .fade
            .map_or(1.0, |fade| 1.0 - ease(fade / FADE_OUT.as_secs_f32()));
        // Cut short by a world on screen (see `world_on_screen`): off for this frame, the
        // world's first, and out of the phase list on the next.
        if world_on_screen {
            live.fade.get_or_insert(0.0);
            live.cut_short = true;
            tracing::info!("loading screen cut short: a world is on screen");
            opacity = 0.0;
        }
        if opacity <= 0.0 {
            live.finished = true;
            live.overlay.with(|overlay| overlay.0.deactivate());
            return false;
        }

        // The overlay: the backdrop, the logo in the user's style, and the progress bar.
        if let Some(style) = style {
            live.style_seen = generation;
            live.font.set_style(style);
        }
        let layout = OverlayLayout::at(w, h);
        let logo = if live.font.prepare(TITLE, layout.logo_size) {
            let cap = live.font.cap_height(layout.logo_size);
            let ink_mid = (live.font.ink_left() + live.font.ink_right()) * 0.5;
            let x = (w * 0.5 - ink_mid).round();
            let baseline = (layout.caps_center_y + cap * 0.5).round();
            Some((x, baseline))
        } else {
            None
        };
        let font = &mut live.font;
        let progress = live.progress;
        live.overlay.with(|overlay| {
            if let Some((pixels, tw, th)) = font.take_upload() {
                overlay.0.upload_title(gpu, pixels, tw, th);
            }
            let mut list = overlay.0.take_spare_list();
            {
                let mut painter = Painter::new(&mut list);
                painter.alpha = opacity;
                // Past the window's edges, so the shape's anti-aliased rim is off screen.
                let screen = Rect::new(-2.0, -2.0, w + 4.0, h + 4.0);
                let backdrop = Fill::css_angle(screen, 180.0, BACKDROP_TOP, BACKDROP_BOTTOM);
                painter.round_rect(screen, 0.0, backdrop);
                if let Some((x, baseline)) = logo {
                    font.draw_logo(&mut painter, x, baseline);
                }
                let bar = layout.bar;
                let radius = bar.h * 0.5;
                painter.round_rect(bar, radius, Fill::Solid([1.0, 1.0, 1.0, 0.16]));
                let filled = Rect::new(bar.x, bar.y, (bar.w * progress).max(bar.h), bar.h);
                painter.round_rect(filled, radius, Fill::Solid([1.0, 1.0, 1.0, 0.88]));
            }
            overlay.0.submit(list, GlassFrame::default());
        });
        opacity >= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_overlay_sits_centred_and_scales_with_the_window() {
        let a = OverlayLayout::at(1920.0, 1080.0);
        assert!((a.scale - 1.0).abs() < 1e-6);
        assert!((a.bar.x + a.bar.w * 0.5 - 960.0).abs() <= 1.0);
        assert!(a.bar.y > a.caps_center_y);
        let b = OverlayLayout::at(3840.0, 2160.0);
        assert!((b.logo_size / a.logo_size - 2.0).abs() < 1e-3);
        let c = OverlayLayout::at(800.0, 1080.0);
        assert!(
            c.logo_size < a.logo_size,
            "a narrow window shrinks the logo"
        );
    }

    /// No loading-screen options any more (no scene choice,
    /// no minimum time): the mod on is this quick screen, off is vanilla's.
    #[test]
    fn the_loading_screen_has_no_options_left() {
        let aero = mc_mods::catalog::find(MOD_ID).expect("the Aero Title Screen mod");
        for retired in mc_mods::mods::aero_title::RETIRED_OPTION_IDS {
            assert!(aero.option(retired).is_none(), "{retired}");
        }
        assert!(mc_mods::ModSettings::new().enabled(MOD_ID), "on by default");
        assert!(FADE_OUT <= Duration::from_millis(500), "a quick fade");
    }

    #[test]
    fn ease_is_clamped_and_monotonic() {
        assert_eq!(ease(-1.0), 0.0);
        assert_eq!(ease(2.0), 1.0);
        let mut last = 0.0;
        for i in 0..=100 {
            let v = ease(i as f32 / 100.0);
            assert!(v >= last);
            last = v;
        }
    }
}

/// The loading screen through the real client: `Client::new_deferred` on the real assets (the
/// game binary's path), the real `install_loading_ui`, `step_loading`, `build_ui`, `install` and
/// shell-frame hooks, and a headless GPU. Needs the vanilla jar (skips without it; set
/// `MC_JARS_DIR`).
#[cfg(test)]
mod client_tests {
    use std::time::{Duration, Instant};

    use mc_render::loading_scene::CURTAIN_PHASE_NAME;
    use mc_render::{Camera, Gpu, GpuOptions, InputState, PhaseList};

    use super::super::super::{Client, Ui};
    use super::super::nw_menu::OPT_MENU_SCENE;
    use super::{MOD_ID, OVERLAY_PHASE_NAME};
    use crate::app::config::{ClientConfig, LoginMode};

    const SIZE: (u32, u32) = (640, 360);

    /// A deferred client (the binary's start: the loading screen planned by its boot), with the
    /// title screen's scene fixed to the greenhouse (the boot picked one at random) so the tests
    /// see the same title screen every run.
    fn deferred_client() -> Option<(Client, tempfile::TempDir)> {
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
        let mut client = Client::new_deferred(config).expect("a deferred client");
        client.mods.settings.set_choice(MOD_ID, OPT_MENU_SCENE, 1);
        Some((client, dir))
    }

    fn gpu() -> Option<Gpu> {
        match Gpu::headless(
            SIZE.0,
            SIZE.1,
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

    fn has(phases: &PhaseList, name: &str) -> bool {
        phases.names().any(|n| n == name)
    }

    /// One frame the way `ShellApp::frame` runs the hooks: the shell-frame hook, a loading
    /// stage while loading, the UI build (the draw hook and the title screen), the render.
    fn frame(client: &mut Client, gpu: &mut Gpu, phases: &mut PhaseList, input: &mut InputState) {
        client.tw_loading_shell_frame(phases, input);
        if client.is_loading() {
            let now_ms = client.started.elapsed().as_millis() as u64;
            client.step_loading(gpu, phases, SIZE.0, SIZE.1, now_ms);
        }
        client.build_ui(gpu, 0.0);
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(gpu, target.view(), &Camera::default(), 0.0);
    }

    /// Load to the end, one stage per frame, checking the overlay covers every loading frame
    /// (when there is one) and no scene is ever built for it.
    fn load(client: &mut Client, gpu: &mut Gpu, phases: &mut PhaseList, input: &mut InputState) {
        let overlay = client.tw_loading.live.is_some();
        let mut frames = 0;
        while client.is_loading() {
            frame(client, gpu, phases, input);
            frames += 1;
            assert!(frames < 64, "the load must finish");
            assert_eq!(
                has(phases, OVERLAY_PHASE_NAME),
                overlay,
                "the overlay covers every loading frame (when there is one)"
            );
            assert!(
                !has(phases, CURTAIN_PHASE_NAME),
                "no scene is built for the loading screen"
            );
        }
        assert!(client.fatal_error().is_none(), "{:?}", client.fatal_error());
    }

    /// The loading screen covers the load with the overlay alone (no scene), the fade starts on
    /// the frame loading finishes (no hold, no minimum time), the title screen's clock starts with
    /// it, and afterwards nothing is left: no phase, and a resource reload does not bring it back.
    #[test]
    fn the_overlay_covers_the_load_then_fades_at_once_and_frees_everything() {
        let Some((mut client, _dir)) = deferred_client() else {
            return;
        };
        let Some(mut gpu) = gpu() else {
            return;
        };
        let mut phases = PhaseList::new();
        let mut input = InputState::new();
        assert!(client.is_loading());
        client.install_loading_ui(&gpu, &mut phases, SIZE.0, SIZE.1);
        assert!(has(&phases, OVERLAY_PHASE_NAME));
        assert!(!has(&phases, CURTAIN_PHASE_NAME), "no scene");
        assert_eq!(
            phases.names().collect::<Vec<_>>(),
            ["ui", OVERLAY_PHASE_NAME],
            "the UI phase and the overlay: nothing else to build"
        );
        assert!(
            client.tw_loading_frame_cap().is_some(),
            "refresh-rate frame cap"
        );
        // The loading frames: the overlay covers everything (the dirt screen is not drawn).
        frame(&mut client, &mut gpu, &mut phases, &mut input);
        assert!(client.tw_loading_holds_title());

        load(&mut client, &mut gpu, &mut phases, &mut input);
        // `finish_loading` rebuilt the phase list and switched to the title screen, underneath;
        // the frame that finished loading also started the fade (still fully covered on it) and
        // the title screen's clock with it.
        assert!(matches!(client.ui, Ui::MainMenu(_)));
        assert!(
            has(&phases, OVERLAY_PHASE_NAME),
            "back on top after the rebuild"
        );
        assert!(!client.tw_loading_holds_title(), "no hold");
        assert!(client.aero_title_clock_started());
        let fade = |client: &Client| client.tw_loading.live.as_ref().and_then(|live| live.fade);
        assert_eq!(
            fade(&client),
            Some(0.0),
            "the fade starts on the loaded frame"
        );
        std::thread::sleep(Duration::from_millis(20));
        frame(&mut client, &mut gpu, &mut phases, &mut input);
        let step = fade(&client).expect("fading");
        assert!(
            step > 0.0 && step <= mc_render::loading_scene::MAX_STEP_S,
            "the fade advances in capped steps: {step}"
        );
        assert!(has(&phases, OVERLAY_PHASE_NAME), "still fading");

        // The end of the fade: the overlay stops, then leaves the phase list.
        let live = client.tw_loading.live.as_mut().expect("still fading");
        live.fade = Some(10.0);
        frame(&mut client, &mut gpu, &mut phases, &mut input);
        frame(&mut client, &mut gpu, &mut phases, &mut input);
        assert!(client.tw_loading.live.is_none(), "dropped");
        assert!(!has(&phases, OVERLAY_PHASE_NAME));
        assert_eq!(client.tw_loading_frame_cap(), None);
        let report = client.tw_loading_report().expect("a report");
        assert!(report.frames >= 3, "{report:?}");
        assert!(!report.cut_short);

        // A later phase-list rebuild (a resource pack change) does not bring it back.
        client.reinstall(&gpu, &mut phases, SIZE.0, SIZE.1);
        assert!(!has(&phases, OVERLAY_PHASE_NAME));
    }

    /// The earlier screen took every click and key while it held the title screen hidden;
    /// this one hides nothing, so its hook leaves the input alone, while loading and while
    /// fading.
    #[test]
    fn the_loading_screen_leaves_the_input_alone() {
        let Some((mut client, _dir)) = deferred_client() else {
            return;
        };
        let Some(gpu) = gpu() else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install_loading_ui(&gpu, &mut phases, SIZE.0, SIZE.1);
        let _ = client.tw_loading_frame(&gpu);
        let mut input = InputState::new();
        input.on_mouse_button(winit::event::MouseButton::Left, true);
        client.tw_loading_shell_frame(&mut phases, &mut input);
        assert_eq!(
            input.drain_all_mouse_events().len(),
            1,
            "the click is still there for the screen"
        );
    }

    /// The pictures the user sees, through the real client: a loading frame (the bar at 55%) and
    /// two frames of the fade into the title screen (a third and two thirds of the way), at
    /// 1920x1080 and 2560x1440, over each title-screen scene. Writes
    /// `loading-<w>x<h>-*.png` to `MC_MENU_PNG_DIR`:
    ///
    /// `MC_MENU_PNG_DIR=<dir> cargo test -p mc-client --lib loading_screen_pngs -- --ignored`
    #[test]
    #[ignore = "renders full-size PNGs for review; run by hand with MC_MENU_PNG_DIR"]
    fn render_the_loading_screen_pngs() {
        let Some(out) = std::env::var_os("MC_MENU_PNG_DIR").map(std::path::PathBuf::from) else {
            eprintln!("skip: set MC_MENU_PNG_DIR");
            return;
        };
        std::fs::create_dir_all(&out).expect("output folder");
        for full in [(1920_u32, 1080_u32), (2560, 1440)] {
            for (choice, name) in [(1, "greenhouse"), (2, "glass"), (3, "shore")] {
                let Some((mut client, _dir)) = deferred_client() else {
                    return;
                };
                client
                    .mods
                    .settings
                    .set_choice(MOD_ID, OPT_MENU_SCENE, choice);
                let Ok(mut gpu) = Gpu::headless(
                    full.0,
                    full.1,
                    GpuOptions {
                        vsync: false,
                        sample_count: 1,
                        low_power: false,
                    },
                ) else {
                    return;
                };
                let mut phases = PhaseList::new();
                let mut input = InputState::new();
                client.install_loading_ui(&gpu, &mut phases, full.0, full.1);
                let tag = format!("{}x{}", full.0, full.1);
                let shot = |client: &mut Client, gpu: &mut Gpu, phases: &mut PhaseList, what| {
                    client.build_ui(gpu, 0.0);
                    let target = gpu.acquire().expect("offscreen frame");
                    phases.render(gpu, target.view(), &Camera::default(), 0.0);
                    let pixels = gpu.read_pixels().expect("readback");
                    let path = out.join(format!("loading-{tag}-{what}.png"));
                    image::save_buffer(&path, &pixels, full.0, full.1, image::ColorType::Rgba8)
                        .expect("png");
                    eprintln!("wrote {}", path.display());
                };
                if choice == 1 {
                    // Loading, the bar part way (once per size: it is the same for every scene).
                    client.tw_loading_shell_frame(&mut phases, &mut input);
                    let now_ms = client.started.elapsed().as_millis() as u64;
                    client.step_loading(&gpu, &mut phases, full.0, full.1, now_ms);
                    if let Some(live) = client.tw_loading.live.as_mut() {
                        live.progress = 0.55;
                        live.last = Some(Instant::now());
                    }
                    shot(&mut client, &mut gpu, &mut phases, "1-loading".to_owned());
                }
                load(&mut client, &mut gpu, &mut phases, &mut input);
                for (what, fraction) in [("2-fade-third", 0.33_f32), ("3-fade-two-thirds", 0.67)] {
                    if let Some(live) = client.tw_loading.live.as_mut() {
                        live.fade = Some(super::FADE_OUT.as_secs_f32() * fraction);
                        live.last = Some(Instant::now());
                    }
                    client.tw_loading_shell_frame(&mut phases, &mut input);
                    shot(&mut client, &mut gpu, &mut phases, format!("{what}-{name}"));
                }
            }
        }
    }

    /// The Aero Title Screen mod off keeps vanilla's dirt loading screen exactly: nothing is
    /// planned, built or registered, and no hook does anything.
    #[test]
    fn the_mod_off_builds_nothing() {
        let Some((mut client, _dir)) = deferred_client() else {
            return;
        };
        client.mods.settings.set_enabled(MOD_ID, false);
        client.tw_loading = super::TwLoadingState::default();
        client.tw_loading_boot();
        assert!(!client.tw_loading.planned);
        let Some(mut gpu) = gpu() else {
            return;
        };
        let mut phases = PhaseList::new();
        let mut input = InputState::new();
        client.install_loading_ui(&gpu, &mut phases, SIZE.0, SIZE.1);
        assert_eq!(
            phases.names().collect::<Vec<_>>(),
            ["ui"],
            "vanilla's UI phase only"
        );
        assert_eq!(client.tw_loading_frame_cap(), None);
        assert!(!client.tw_loading_frame(&gpu));
        load(&mut client, &mut gpu, &mut phases, &mut input);
        assert!(client.tw_loading.live.is_none());
        assert!(client.tw_loading_report().is_none());
    }

    // ---- A world on screen ends the loading screen ----------
    //
    // The ghost logo: The end-to-end run opened a replay on the first frame after loading,
    // while the curtain was still fading, and its frames carried the glass and the logo at low
    // opacity. These tests render real in-game frames (the SkyWars capture's world, joined by
    // `--server` straight from the load, from the title screen, and as a replay) right after the
    // loading screen and the title screen, and count the pixels the loading screen's and the
    // title screen's layers add to them. The same count over frames where those layers do draw
    // (the fading curtain over the Connecting screen or the title screen, the title screen
    // itself) is the control.
    //
    // `MC_INT_PNG_DIR=<dir>` writes every counted frame as a PNG for review.

    /// Every phase of the loading screen and of the title screen: each logo is drawn by one.
    const MENU_LAYERS: [&str; 6] = [
        CURTAIN_PHASE_NAME,
        OVERLAY_PHASE_NAME,
        "aero_title/background",
        "aero_title",
        "aero_title/finish",
        "aero_title/crisp",
    ];

    /// The SkyWars benchmark capture (a real SkyWars map), relative to this crate.
    fn skywars_capture() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../optimization/scratch/skywars-scene-2026-11-05/captures")
            .join("sw-mega-doubles-mid.mccap");
        if path.is_file() {
            Some(path)
        } else {
            eprintln!("skip: no SkyWars capture at {}", path.display());
            None
        }
    }

    /// The capture's world as a real replay file in the client's library (the writer): the
    /// clientbound Play packets a capture-driven session applies, at the capture's own times.
    fn write_capture_replay(
        game_dir: &std::path::Path,
        capture: &std::path::Path,
    ) -> std::path::PathBuf {
        use mc_parity::capture::{Capture, Direction};
        let capture = Capture::read_file(capture).expect("the capture reads");
        let dir = game_dir.join(mc_replay::REPLAYS_DIR);
        std::fs::create_dir_all(&dir).expect("the replays folder");
        let path = dir.join(format!("skywars.{}", mc_replay::EXTENSION));
        let mut meta = mc_replay::ReplayMeta {
            title: "SkyWars".to_owned(),
            server: "skywars.test".to_owned(),
            ..mc_replay::ReplayMeta::default()
        };
        let mut writer =
            mc_replay::Writer::create(&path, &meta, mc_replay::WriterOptions::default())
                .expect("the writer");
        let mut first = None;
        for frame in capture.frames() {
            // KeepAlive, SetCompression and Disconnect are the network task's own: the replay
            // source drops them too.
            if frame.direction != Direction::Clientbound
                || frame.state != mc_protocol::State::Play
                || matches!(frame.packet_id(), Some(0x00 | 0x40 | 0x46))
            {
                continue;
            }
            let time = frame
                .micros
                .saturating_sub(*first.get_or_insert(frame.micros));
            writer.push(time, mc_replay::StreamId::Packet, &frame.payload);
            meta.duration = time;
        }
        writer
            .finish(meta)
            .wait(Duration::from_secs(30))
            .expect("the file is written");
        path
    }

    /// Hold the loading screen at the very start of its fade (all but opaque: one capped clock
    /// step in, and fading, so input reaches the screen under it) however slowly a test build
    /// draws: where the harness opened its replay. Nothing once it is finished, and nothing
    /// while still loading.
    fn pin_fade_start(client: &mut Client) {
        if let Some(live) = client
            .tw_loading
            .live
            .as_mut()
            .filter(|live| !live.finished && live.fade.is_some())
        {
            live.fade = Some(0.0);
            live.last = Some(Instant::now());
        }
    }

    /// Pixels whose colour differs between `a` and `b`, anywhere or inside `area`
    /// (`[x0, y0, x1, y1)`, physical pixels).
    fn differing(a: &[u8], b: &[u8], area: Option<[usize; 4]>) -> usize {
        let width = SIZE.0 as usize;
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .enumerate()
            .filter(|(i, (p, q))| {
                let (x, y) = (i % width, i / width);
                let inside = area
                    .is_none_or(|[x0, y0, x1, y1]| (x0..x1).contains(&x) && (y0..y1).contains(&y));
                inside && p[..3] != q[..3]
            })
            .count()
    }

    /// The bounding box of the pixels that differ, `[x0, y0, x1, y1)`.
    fn differing_box(a: &[u8], b: &[u8]) -> Option<[usize; 4]> {
        let width = SIZE.0 as usize;
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .enumerate()
            .filter(|(_, (p, q))| p[..3] != q[..3])
            .map(|(i, _)| (i % width, i / width))
            .fold(None, |area, (x, y)| {
                let [x0, y0, x1, y1] = area.unwrap_or([x, y, x + 1, y + 1]);
                Some([x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1)])
            })
    }

    /// The loading logo's box at `SIZE`, `[x0, y0, x1, y1)`: the area `tw_loading_frame` hands the
    /// scene as the logo's (`SceneFrame::logo`), from the same layout and font.
    fn logo_box(client: &Client) -> [usize; 4] {
        let (w, h) = (SIZE.0 as f32, SIZE.1 as f32);
        let layout = super::OverlayLayout::at(w, h);
        let mut font = super::TitleFont::embedded();
        font.set_style(super::LogoStyle::from_settings(&client.mods.settings));
        assert!(
            font.prepare(super::TITLE, layout.logo_size),
            "the logo's font"
        );
        let cap = font.cap_height(layout.logo_size);
        let ink_mid = (font.ink_left() + font.ink_right()) * 0.5;
        let left = (w * 0.5 - ink_mid).round() + font.ink_left();
        let right = left + font.ink_right() - font.ink_left();
        let baseline = (layout.caps_center_y + cap * 0.5).round();
        let top = baseline - cap - 0.2 * layout.logo_size;
        let bottom = top + cap + 0.45 * layout.logo_size;
        [left.floor(), top.floor(), right.ceil(), bottom.ceil()].map(|v| v.max(0.0) as usize)
    }

    /// `MC_INT_PNG_DIR`: write a counted frame for review.
    fn save_png(name: &str, pixels: &[u8]) {
        let Some(dir) = std::env::var_os("MC_INT_PNG_DIR").map(std::path::PathBuf::from) else {
            return;
        };
        std::fs::create_dir_all(&dir).expect("the PNG folder");
        let path = dir.join(format!("{name}.png"));
        image::save_buffer(&path, pixels, SIZE.0, SIZE.1, image::ColorType::Rgba8).expect("png");
        eprintln!("wrote {}", path.display());
    }

    /// A deferred client with the glass loading screen and no minimum time (the fade starts the
    /// frame loading ends), and what the shell would hand it.
    struct Rig {
        client: Client,
        gpu: Gpu,
        phases: PhaseList,
        input: InputState,
        camera: Camera,
        _dir: tempfile::TempDir,
    }

    impl Rig {
        fn new() -> Option<Rig> {
            let (client, dir) = deferred_client()?;
            let gpu = gpu()?;
            let mut camera = crate::app::driver::menu_camera();
            camera.set_viewport(SIZE.0, SIZE.1);
            Some(Rig {
                client,
                gpu,
                phases: PhaseList::new(),
                input: InputState::new(),
                camera,
                _dir: dir,
            })
        }

        /// The loading screen and the load, to the frame the curtain starts to fade.
        fn load(&mut self) {
            self.client
                .install_loading_ui(&self.gpu, &mut self.phases, SIZE.0, SIZE.1);
            load(
                &mut self.client,
                &mut self.gpu,
                &mut self.phases,
                &mut self.input,
            );
        }

        fn in_world(&self) -> bool {
            self.client.machine.state().is_in_world()
        }

        /// One frame as the shell runs one: the shell-frame hook, then (with `step`) one poll and
        /// one tick of the capture's world, then `Client::frame`, which builds the UI and with it
        /// the loading screen's frame. The curtain, while up, stays at the start of its fade.
        fn game_frame(&mut self, step: bool) {
            pin_fade_start(&mut self.client);
            self.client
                .tw_loading_shell_frame(&mut self.phases, &mut self.input);
            if step {
                self.client.poll_in_place();
                Client::tick(&mut self.client, &mc_physics::state::MoveInput::default());
            }
            Client::frame(&mut self.client, &self.gpu, &mut self.camera, 1.0);
        }

        /// One frame of the open replay at `time`, as the exporter and the harness draw one.
        fn replay_frame(&mut self, time: mc_replay::ReplayTime) {
            pin_fade_start(&mut self.client);
            self.client
                .tw_loading_shell_frame(&mut self.phases, &mut self.input);
            let partial = self.client.replay_render_at(time);
            Client::frame(&mut self.client, &self.gpu, &mut self.camera, partial);
        }

        /// Press and release `code` (screens take keys through `handle_screen_input`).
        fn key(&mut self, code: winit::keyboard::KeyCode) {
            for pressed in [true, false] {
                self.client.handle_raw_key(code, pressed, false, None);
                self.client
                    .handle_screen_input(&mut self.input, SIZE.0, SIZE.1);
            }
        }

        fn shoot(&mut self) -> Vec<u8> {
            let target = self.gpu.acquire().expect("offscreen frame");
            self.phases
                .render(&self.gpu, target.view(), &self.camera, 1.0);
            let pixels = target.read_pixels(&self.gpu).expect("readback");
            target.present(&self.gpu.queue);
            pixels
        }

        /// What the loading screen's and the title screen's layers add to the frame the client
        /// just built: it is rendered as it stands, then again with those layers out of the phase
        /// list, and the pixels that differ are counted (anywhere, and inside the loading logo's
        /// box). The layers go back as the client keeps them: the loading screen's while it is
        /// up, the title screen's slots fresh (they fill again when it next draws).
        fn menu_layer_pixels(&mut self, name: &str) -> (usize, usize) {
            // The held item's enchantment glint reads the wall clock as the first-person phase
            // records, so two renders of one frame differ there. It is off for these renders (a
            // latch the driver sets again when the held item next changes).
            if let Some(render) = self.client.render.as_ref() {
                render
                    .first_person
                    .with(|phase| phase.set_item_glint_enabled(false));
            }
            let with = self.shoot();
            save_png(name, &with);
            for layer in MENU_LAYERS {
                self.phases.remove_named(layer);
            }
            let without = self.shoot();
            let again = self.shoot();
            let noise = differing(&without, &again, None);
            if noise > 0 {
                save_png(&format!("{name}-noise-a"), &without);
                save_png(&format!("{name}-noise-b"), &again);
            }
            assert_eq!(
                noise,
                0,
                "{name}: the frame renders the same twice (differing box {:?})",
                differing_box(&without, &again)
            );
            self.client.aero_title_install_phases(&mut self.phases);
            self.client.tw_loading_install(&self.gpu, &mut self.phases);
            let logo = logo_box(&self.client);
            (
                differing(&with, &without, None),
                differing(&with, &without, Some(logo)),
            )
        }

        /// Frames until the capture's world has arrived (its JoinGame).
        fn join(&mut self) {
            let mut frames = 0;
            while !self.in_world() {
                self.game_frame(true);
                frames += 1;
                assert!(frames < 200, "the capture's JoinGame arrives");
            }
        }
    }

    /// `--server` with the capture as the server (the launcher's server field): `finish_loading`
    /// connects straight from the load, so the world arrives while the curtain fades. Over the
    /// Connecting screen the curtain fades as before; the frame the world arrives cuts it, and no
    /// in-game frame carries a pixel of it, of its logo or of the title screen, nor after going
    /// back to the title screen and joining again from it.
    #[test]
    fn joining_a_world_cuts_the_loading_screen_short_and_no_menu_layer_draws_over_the_game() {
        let Some(capture) = skywars_capture() else {
            return;
        };
        let Some(mut rig) = Rig::new() else {
            return;
        };
        rig.client.config.server = Some(("replay.invalid".to_owned(), 25565));
        rig.client.config.replay = Some(capture);
        rig.client.config.replay_step_micros = 50_000;
        rig.load();
        assert!(
            matches!(rig.client.ui, Ui::Connecting(_)),
            "connected from the load"
        );

        // No world yet: the curtain and its logo still fade over the Connecting screen, and the
        // count sees them (the control).
        rig.game_frame(false);
        assert!(!rig.in_world());
        let (all, logo) = rig.menu_layer_pixels("int-server-0-connecting-under-curtain");
        assert!(
            all > 0 && logo > 0,
            "the fading curtain and its logo draw: {all} px, {logo} px in the logo's box"
        );

        // The world arrives: nothing of the loading screen on that frame or after it.
        rig.join();
        assert_eq!(
            rig.menu_layer_pixels("int-server-1-first-world-frame"),
            (0, 0),
            "the world's first frame"
        );
        assert!(
            rig.client
                .tw_loading
                .live
                .as_ref()
                .is_some_and(|live| live.finished),
            "cut short on the world's first frame"
        );
        rig.game_frame(true);
        assert!(rig.client.tw_loading.live.is_none(), "dropped on the next");
        assert!(!has(&rig.phases, CURTAIN_PHASE_NAME));
        assert!(!has(&rig.phases, OVERLAY_PHASE_NAME));
        for _ in 0..150 {
            rig.game_frame(true);
        }
        assert_eq!(
            rig.menu_layer_pixels("int-server-2-in-game"),
            (0, 0),
            "in game"
        );

        // Back to the title screen: its layers draw there (the control)...
        rig.client.disconnect();
        assert!(matches!(rig.client.ui, Ui::MainMenu(_)));
        for _ in 0..3 {
            rig.game_frame(false);
        }
        let (title, _) = rig.menu_layer_pixels("int-server-3-title");
        assert!(title > 0, "the title screen draws its layers: {title} px");
        // ...and none of them over the game joined from it.
        rig.client.connect("replay.invalid", 25565);
        rig.join();
        assert_eq!(
            rig.menu_layer_pixels("int-server-4-rejoin-first-world-frame"),
            (0, 0),
            "the first frame of a world joined from the title screen"
        );
        for _ in 0..150 {
            rig.game_frame(true);
        }
        assert_eq!(
            rig.menu_layer_pixels("int-server-5-rejoin-in-game"),
            (0, 0),
            "in game after the title screen"
        );
    }

    /// A replay opened on the first frame after loading, during the curtain's fade (the harness, where the ghost logo was seen), then the same replay watched through the replay
    /// browser from the title screen: no frame of either carries a pixel of the loading screen,
    /// of its logo or of the title screen.
    #[test]
    fn a_replay_cuts_the_loading_screen_short_and_no_menu_layer_draws_over_it() {
        let Some(capture) = skywars_capture() else {
            return;
        };
        let Some(mut rig) = Rig::new() else {
            return;
        };
        let replay = write_capture_replay(&rig.client.config.game_dir, &capture);
        rig.load();
        assert!(matches!(rig.client.ui, Ui::MainMenu(_)));

        // The title screen under the fading curtain: the count sees the curtain and its logo.
        rig.game_frame(false);
        let (all, logo) = rig.menu_layer_pixels("int-replay-0-title-under-curtain");
        assert!(
            all > 0 && logo > 0,
            "the fading curtain and its logo draw: {all} px, {logo} px in the logo's box"
        );

        // The replay opens on the next frame, as the harness opens it (`replay_open` from
        // `before_ticks`), and plays at 20 frames a second of replay time.
        rig.client.replay_open(&replay).expect("the replay opens");
        let mut time = rig.client.replay_view_range().0 + 3_000_000;
        rig.replay_frame(time);
        assert!(rig.in_world(), "the replay's world is up");
        assert_eq!(
            rig.menu_layer_pixels("int-replay-1-first-frame"),
            (0, 0),
            "the replay's first frame"
        );
        for _ in 0..150 {
            time += 50_000;
            rig.replay_frame(time);
        }
        assert!(rig.client.tw_loading.live.is_none());
        assert_eq!(
            rig.menu_layer_pixels("int-replay-2-playing"),
            (0, 0),
            "playing"
        );

        // Back to the title screen (its layers draw: the control), then the same replay through
        // the browser: Replays, then Enter watches the selected (only) replay.
        rig.client.replay_close();
        assert!(matches!(rig.client.ui, Ui::MainMenu(_)));
        for _ in 0..3 {
            rig.game_frame(false);
        }
        let (title, _) = rig.menu_layer_pixels("int-replay-3-title");
        assert!(title > 0, "the title screen draws its layers: {title} px");
        rig.client.open_replay_browser();
        for _ in 0..500 {
            if rig.client.replay_is_active() {
                break;
            }
            // The browser's listing arrives on a thread; Enter watches once it is in.
            rig.game_frame(false);
            rig.key(winit::keyboard::KeyCode::Enter);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            rig.client.replay_is_active(),
            "the browser opened the replay"
        );
        let mut time = rig.client.replay_view_range().0 + 3_000_000;
        rig.replay_frame(time);
        assert_eq!(
            rig.menu_layer_pixels("int-replay-4-browser-first-frame"),
            (0, 0),
            "the first frame of a replay watched from the browser"
        );
        for _ in 0..150 {
            time += 50_000;
            rig.replay_frame(time);
        }
        assert_eq!(
            rig.menu_layer_pixels("int-replay-5-browser-playing"),
            (0, 0),
            "playing from the browser"
        );
    }
}
