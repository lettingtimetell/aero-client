//! Controls, the client half: Toggle Sprint, Zoom, Dynamic FOV, Raw Mouse Input,
//! Borderless Fullscreen and No Hurt Cam.
//!
//! Every mod here works by changing one *input* to vanilla's own code path, never the path
//! itself, so the packets and the physics stay vanilla's:
//!
//! * `toggle_sprint`: the Sprint (and optionally Sneak) binding reads as held
//!   (edges sampled).
//! * `zoom`: the world FOV multiplier is divided, the look delta scaled,
//!   the wheel consumed (`mouse_wheel`), the smooth camera ORed in (the two
//!   smooth-camera sites), the held item hidden.
//! * `dynamic_fov`: the per-tick target of `updateFovModifierHand` is flattened.
//!   The mod is "Dynamic
//!   FOV", default on: on at 100 % with bow zoom is vanilla and keeps no state; off flattens
//!   everything, as "No Dynamic FOV" at 0 % did.
//! * `raw_input`: OFF reads the OS pointer instead of `WM_INPUT`
//!   (`InputState::set_os_pointer_look`).
//! * `borderless_fullscreen`: OFF asks the shell for an exclusive video mode
//!   (`ShellApp::wants_exclusive_fullscreen`).
//! * `no_hurt_cam`: the hurt roll is scaled (world camera and held item).
//!
//! # Cost
//!
//! Everything reads [`M3State`], a typed snapshot refreshed only when the settings generation
//! moves (one `u64` compare per call otherwise). With every M3 mod at its default, a frame
//! costs a handful of `bool` tests and no allocation.
//!
//! # Toggle sprint and packets
//!
//! The toggle is ORed into `MoveInput::sprint`, the value `move_input` reads from
//! `keyBindSprint.isKeyDown()`. Everything downstream (the hunger, blindness, item-use, forward
//! and wall gates, the double-tap timer, the `C0B` START/STOP_SPRINTING edges) is the unchanged
//! vanilla state machine in `mc-physics`, so a toggled sprint sends exactly the bytes of a held
//! Ctrl. `tests/modwave_m3_controls.rs` compares the two on a loopback server.

use mc_mods::mods::toggle_sprint::SPRINT_TOGGLED_ID;
use mc_mods::ModSettings;
use mc_physics::state::MoveInput;
use mc_render::entity::Mat4;
use mc_render::input::{keys, Input};
use mc_render::{InputState, KeyBindings};
use mc_ui::modhud::elements::toggle_sprint::SprintStatus;

use super::super::{Client, Ui};

const TOGGLE_SPRINT: &str = "toggle_sprint";
const ZOOM: &str = "zoom";
const DYNAMIC_FOV: &str = "dynamic_fov";
const RAW_INPUT: &str = "raw_input";
const BORDERLESS_FULLSCREEN: &str = "borderless_fullscreen";
const NO_HURT_CAM: &str = "no_hurt_cam";

/// Time constant of the smooth zoom, seconds: the divisor closes `1 - 1/e` of the remaining
/// gap every 40 ms, so a 4x zoom settles in about 150 ms at any frame rate.
pub(crate) const ZOOM_EASE_SECONDS: f32 = 0.04;
/// The eased divisor snaps to its goal once this close, so it reaches exactly `1.0` again.
pub(crate) const ZOOM_SNAP: f32 = 0.002;
/// One wheel notch multiplies (up) or divides (down) the zoom by this.
pub(crate) const ZOOM_SCROLL_STEP: f32 = 1.25;
/// The scroll-adjusted zoom stays within this range.
pub(crate) const ZOOM_SCROLL_RANGE: (f32, f32) = (1.5, 50.0);
/// Above this divisor the held item would be magnified off screen, so it is not drawn
/// (research `zoom.md`).
pub(crate) const ZOOM_HIDES_HAND_ABOVE: f32 = 1.01;

/// Everything M3 keeps on the client (`Client::m3`).
#[derive(Debug, Clone)]
pub(crate) struct M3State {
    /// The settings generation the snapshot was read at (`0` = never; the store starts at 1).
    seen: u64,
    pub(crate) toggle: ToggleSprint,
    pub(crate) zoom: Zoom,
    /// Dynamic FOV's flattening, while it is not vanilla (`None` on, at 100 %, with bow zoom).
    fov: Option<StaticFov>,
    /// Dynamic FOV's per-source sliders (`nw_dynamic_fov`), while one of them is not
    /// at 100 % and the mod is on.
    pub(crate) fov_sources: Option<super::nw_dynamic_fov::FovSources>,
    /// Raw Mouse Input is on (the default, and what mc-rs always did).
    pub(crate) raw_input: bool,
    /// Borderless Fullscreen is **off**: F11 asks the shell for an exclusive video mode.
    pub(crate) exclusive_fullscreen: bool,
    /// No Hurt Cam's roll scale (`intensity / 100`), while the mod is on.
    hurt_cam: Option<f32>,
}

impl M3State {
    /// A snapshot of `settings`, already synced.
    #[must_use]
    pub(crate) fn new(settings: &ModSettings) -> M3State {
        let mut state = M3State {
            seen: 0,
            toggle: ToggleSprint::default(),
            zoom: Zoom::default(),
            fov: None,
            fov_sources: None,
            raw_input: true,
            exclusive_fullscreen: false,
            hurt_cam: None,
        };
        state.sync(settings);
        state
    }

    /// Refresh the snapshot when the settings changed. One `u64` compare otherwise.
    pub(crate) fn sync(&mut self, settings: &ModSettings) {
        let generation = settings.generation();
        if self.seen == generation {
            return;
        }
        self.seen = generation;
        self.toggle.sync(settings);
        self.zoom.sync(settings);
        self.fov = StaticFov::from_settings(settings);
        self.fov_sources = super::nw_dynamic_fov::FovSources::from_settings(settings);
        self.raw_input = settings.enabled(RAW_INPUT);
        self.exclusive_fullscreen = !settings.enabled(BORDERLESS_FULLSCREEN);
        self.hurt_cam = settings
            .enabled(NO_HURT_CAM)
            .then(|| settings.f32(NO_HURT_CAM, "intensity") / 100.0);
    }
}

// ---- toggle sprint ----------------------------------------------------------------------

/// Toggle Sprint's runtime state.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToggleSprint {
    enabled: bool,
    /// The `toggle_sneak` option.
    toggle_sneak: bool,
    /// The sprint toggle. Persisted as the hidden `sprint_toggled` option.
    pub(crate) sprint_toggled: bool,
    /// The sneak toggle. Never persisted; any screen releases it.
    pub(crate) sneak_toggled: bool,
    /// The stored `sprint_toggled` as last read from or written to the store; `None` before the
    /// first sync. The store's value is adopted only when it differs from this, so an unrelated
    /// settings change never rolls the live toggle back to a stale stored value.
    stored: Option<bool>,
    sprint_was_down: bool,
    sneak_was_down: bool,
    /// The bindings at the last in-game sample, for the HUD status.
    sprint_down: bool,
    sneak_down: bool,
    /// The Sprint (Sneak) key was physically down outside the game (a
    /// screen was open) and has not been let go since ([`ToggleSprint::note_physical`]): its
    /// binding reading as down in game is that old hold, not a tap.
    sprint_held_over: bool,
    sneak_held_over: bool,
}

impl ToggleSprint {
    fn sync(&mut self, settings: &ModSettings) {
        let enabled = settings.enabled(TOGGLE_SPRINT);
        if enabled && !self.enabled {
            // Start from "not held": the first press after this is a real tap, including one on
            // the very first in-game frame after start-up. A key still held from a screen is
            // told apart by `note_physical`, which the next sample in that screen updates.
            self.sprint_was_down = false;
            self.sneak_was_down = false;
            self.sneak_toggled = false;
            self.sprint_held_over = false;
            self.sneak_held_over = false;
        }
        self.enabled = enabled;
        self.toggle_sneak = settings.bool(TOGGLE_SPRINT, "toggle_sneak");
        if !self.toggle_sneak {
            self.sneak_toggled = false;
        }
        let stored = settings.text(TOGGLE_SPRINT, SPRINT_TOGGLED_ID) == "on";
        if self.stored != Some(stored) {
            self.stored = Some(stored);
            self.sprint_toggled = stored;
        }
    }

    /// The Sprint and Sneak keys' physical state (the keyboard, not the
    /// bindings), sampled each frame before [`ToggleSprint::sample`]. A key held outside the
    /// game is latched until it is let go: Modern Keybinds (on by default) marks a key still
    /// held as a screen closes as held again, and that must not read as a tap. Without the
    /// latch, closing the inventory with Shift still down from a shift-click left the player
    /// sneaking, and Ctrl held for Ctrl+Q (or held through any screen) flipped the sprint
    /// toggle. Vanilla 1.8.9 never re-presses, so with Modern Keybinds off nothing changes.
    pub(crate) fn note_physical(&mut self, sprint_held: bool, sneak_held: bool, in_game: bool) {
        if in_game {
            self.sprint_held_over &= sprint_held;
            self.sneak_held_over &= sneak_held;
        } else {
            self.sprint_held_over = sprint_held;
            self.sneak_held_over = sneak_held;
        }
    }

    /// Whether a key held over from outside the game is still latched (in game,
    /// [`ToggleSprint::note_physical`] needs the keyboard only then).
    #[must_use]
    pub(crate) fn holding_over(&self) -> bool {
        self.sprint_held_over || self.sneak_held_over
    }

    /// One sample of the live Sprint and Sneak bindings. A rising edge flips the matching
    /// toggle (the edge, not `take_pressed`: vanilla never consumes Sprint's press queue, so it
    /// holds presses from long before the mod was on), unless the key is still held from outside
    /// the game ([`ToggleSprint::note_physical`]). Outside the game nothing toggles, and the
    /// sneak toggle is released, so no screen is ever "sneaked" into.
    pub(crate) fn sample(&mut self, sprint_down: bool, sneak_down: bool, in_game: bool) {
        if !self.enabled {
            return;
        }
        if !in_game {
            self.sneak_toggled = false;
            self.sprint_down = false;
            self.sneak_down = false;
            self.sprint_was_down = sprint_down;
            self.sneak_was_down = sneak_down;
            return;
        }
        if sprint_down && !self.sprint_was_down && !self.sprint_held_over {
            self.sprint_toggled = !self.sprint_toggled;
        }
        if self.toggle_sneak && sneak_down && !self.sneak_was_down && !self.sneak_held_over {
            self.sneak_toggled = !self.sneak_toggled;
        }
        self.sprint_was_down = sprint_down;
        self.sneak_was_down = sneak_down;
        self.sprint_down = sprint_down;
        self.sneak_down = sneak_down;
    }

    /// The tick's movement input with the toggles read as held keys. Only in game: with a
    /// screen open vanilla has unpressed every binding, so a held key would not count either.
    #[must_use]
    pub(crate) fn apply(&self, input: MoveInput, in_game: bool) -> MoveInput {
        if !self.enabled || !in_game {
            return input;
        }
        MoveInput {
            sprint: input.sprint || self.sprint_toggled,
            sneak: input.sneak || (self.toggle_sneak && self.sneak_toggled),
            ..input
        }
    }

    /// What the HUD text says, highest priority first: flying, riding, sneaking (toggled, then
    /// held), the sprint toggle, then sprinting with the key held or from vanilla's double tap.
    #[must_use]
    pub(crate) fn status(&self, flying: bool, riding: bool, sprinting: bool) -> SprintStatus {
        if flying {
            SprintStatus::Flying
        } else if riding {
            SprintStatus::Riding
        } else if self.toggle_sneak && self.sneak_toggled {
            SprintStatus::SneakToggled
        } else if self.sneak_down {
            SprintStatus::SneakHeld
        } else if self.sprint_toggled {
            SprintStatus::SprintToggled
        } else if sprinting && self.sprint_down {
            SprintStatus::SprintHeld
        } else if sprinting {
            SprintStatus::SprintVanilla
        } else {
            SprintStatus::Hidden
        }
    }
}

// ---- zoom ---------------------------------------------------------------------------------

/// Zoom's options and its eased FOV divisor.
#[derive(Debug, Clone)]
pub(crate) struct Zoom {
    enabled: bool,
    key: Input,
    factor: f32,
    smooth_zoom: bool,
    scroll: bool,
    smooth_camera: bool,
    /// The `sensitivity` option as a fraction (`1.0` = unchanged).
    sensitivity: f32,
    /// The key is held, in game, with the mod on.
    pub(crate) active: bool,
    /// The divisor the zoom eases toward while active: `factor`, then wheel-adjusted.
    pub(crate) target: f32,
    /// The current FOV divisor. `1.0` means no zoom.
    pub(crate) divisor: f32,
}

impl Default for Zoom {
    fn default() -> Self {
        Zoom {
            enabled: false,
            key: Input::Unbound,
            factor: 4.0,
            smooth_zoom: true,
            scroll: true,
            smooth_camera: true,
            sensitivity: 1.0,
            active: false,
            target: 4.0,
            divisor: 1.0,
        }
    }
}

impl Zoom {
    fn sync(&mut self, settings: &ModSettings) {
        self.enabled = settings.enabled(ZOOM);
        self.key = Input::from_lwjgl(settings.key(ZOOM, "key"));
        self.factor = settings.f32(ZOOM, "factor").max(1.0);
        self.smooth_zoom = settings.bool(ZOOM, "smooth_zoom");
        self.scroll = settings.bool(ZOOM, "scroll");
        self.smooth_camera = settings.bool(ZOOM, "smooth_camera");
        self.sensitivity = settings.f32(ZOOM, "sensitivity") / 100.0;
    }

    /// One frame: `held` is the zoom key's physical state while in game, `dt` the last frame's
    /// duration in seconds. Turning the mod off drops the divisor to 1 at once.
    pub(crate) fn drive(&mut self, held: bool, dt: f32) {
        let held = held && self.enabled;
        if held && !self.active {
            // Every zoom starts from the configured amount, whatever the wheel did last time.
            self.target = self.factor;
        }
        self.active = held;
        if !self.enabled {
            self.divisor = 1.0;
            return;
        }
        let goal = if held { self.target } else { 1.0 };
        self.divisor = if self.smooth_zoom {
            ease_divisor(self.divisor, goal, dt)
        } else {
            goal
        };
    }

    /// A wheel notch while zoomed: zoom further in (up) or out (down), consuming the wheel so
    /// the hotbar does not change and no `C09` is sent. `false` (not consumed) otherwise.
    pub(crate) fn wheel(&mut self, delta: f32) -> bool {
        if !(self.enabled && self.scroll && self.active) {
            return false;
        }
        let (lo, hi) = ZOOM_SCROLL_RANGE;
        self.target = (self.target * ZOOM_SCROLL_STEP.powf(delta)).clamp(lo, hi);
        true
    }

    /// Whether [`Self::wheel`] would take a notch now, without zooming.
    pub(crate) fn takes_wheel(&self) -> bool {
        self.enabled && self.scroll && self.active
    }

    /// The world camera's FOV multiplier, zoomed.
    #[must_use]
    pub(crate) fn fov(&self, f: f32) -> f32 {
        if self.divisor == 1.0 {
            f
        } else {
            f / self.divisor
        }
    }

    /// The `mouseSensitivity` mouse look runs with: `base`, or while the key is held with a
    /// "sensitivity while zoomed" other than 100 %, the vanilla value whose step is that share
    /// of `base`'s.
    ///
    /// java: `EntityRenderer` turns by `(float)deltaX * f1`, `f1 = f * f * f * 8.0F`,
    /// `f = mouseSensitivity * 0.6F + 0.2F`, so every turn is a whole number of `f1` steps.
    /// Scaling the finished float delta instead (what this did before this change) gives steps
    /// no `mouseSensitivity` produces: rotations off vanilla's grid, the aim-check signature.
    /// Solving `f'^3 = share * f^3` for a sensitivity keeps both the frame step and the smooth
    /// camera's tick factor exactly vanilla's for that sensitivity.
    #[must_use]
    pub(crate) fn mouse_sensitivity(&self, base: f32) -> f32 {
        if !(self.active && self.sensitivity != 1.0) {
            return base;
        }
        let f = f64::from(base) * 0.6 + 0.2;
        let zoomed = f * f64::from(self.sensitivity).cbrt();
        ((zoomed - 0.2) / 0.6) as f32
    }

    /// Whether the cinematic camera is forced on by the zoom right now.
    #[must_use]
    pub(crate) fn smooth_camera_active(&self) -> bool {
        self.active && self.smooth_camera
    }

    /// Whether the held item is hidden (it would be magnified off screen).
    #[must_use]
    pub(crate) fn hides_hand(&self) -> bool {
        self.divisor > ZOOM_HIDES_HAND_ABOVE
    }
}

/// Frame-rate independent exponential approach of `current` to `goal`, snapping when close.
#[must_use]
pub(crate) fn ease_divisor(current: f32, goal: f32, dt: f32) -> f32 {
    // A long stall (a world load, a breakpoint) must not overshoot or produce NaN.
    let dt = if dt.is_finite() {
        dt.clamp(0.0, 0.25)
    } else {
        0.0
    };
    let next = goal + (current - goal) * (-dt / ZOOM_EASE_SECONDS).exp();
    if (next - goal).abs() < ZOOM_SNAP {
        goal
    } else {
        next
    }
}

// ---- dynamic FOV ---------------------------------------------------------------------------

/// How Dynamic FOV flattens vanilla's FOV target, while it is not vanilla.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StaticFov {
    /// How much of the movement change (sprint, Speed, Slowness, flying) is kept, `0.0`
    /// (none) to `1.0` (vanilla).
    pub(crate) strength: f32,
    /// The bow's narrowing is kept in full (vanilla); `false` drops it.
    pub(crate) bow_zoom: bool,
}

impl StaticFov {
    /// Off: the FOV never moves, the bow included.
    pub(crate) const OFF: StaticFov = StaticFov {
        strength: 0.0,
        bow_zoom: false,
    };

    /// The flattening `settings` ask for, or `None` when the target stays vanilla's (on, at
    /// 100 %, with bow zoom: the default), so the default costs one `Option` test per tick.
    #[must_use]
    pub(crate) fn from_settings(settings: &ModSettings) -> Option<StaticFov> {
        if !settings.enabled(DYNAMIC_FOV) {
            return Some(StaticFov::OFF);
        }
        let fov = StaticFov {
            strength: settings.f32(DYNAMIC_FOV, "strength") / 100.0,
            bow_zoom: settings.bool(DYNAMIC_FOV, "bow_zoom"),
        };
        (fov.strength < 1.0 || !fov.bow_zoom).then_some(fov)
    }
}

/// Flatten vanilla's per-tick FOV target. `target` is `fov_modifier_target`'s movement part
/// times `bow`, the bow part (`1.0` when no bow is drawn). The movement part keeps `strength` of
/// its distance from `1.0`; the bow part is kept whole or dropped.
#[must_use]
pub(crate) fn static_fov_target(target: f32, bow: f32, fov: StaticFov) -> f32 {
    if fov.strength >= 1.0 && fov.bow_zoom {
        return target;
    }
    let movement = target / bow;
    let movement = if fov.strength >= 1.0 {
        movement
    } else {
        1.0 + (movement - 1.0) * fov.strength
    };
    if fov.bow_zoom {
        movement * bow
    } else {
        movement
    }
}

// ---- no hurt cam ---------------------------------------------------------------------------

/// `hurt_camera_transform` with the roll scaled by `scale`. Both live call sites pass
/// `attacked_at_yaw = 0.0` (1.8.9 never sets it for the local player; see `place_camera`), so
/// the yaw conjugation is the identity and is left out. The arithmetic up to the roll is
/// vanilla's, in the same f32 order, so `scale = 1.0` is the vanilla matrix.
#[must_use]
pub(crate) fn scaled_hurt_camera_transform(
    hurt_time: i32,
    max_hurt_time: i32,
    partial_ticks: f32,
    scale: f32,
) -> Mat4 {
    if scale <= 0.0 {
        return Mat4::IDENTITY;
    }
    let mut hurt = hurt_time as f32 - partial_ticks;
    if hurt < 0.0 {
        return Mat4::IDENTITY;
    }
    hurt /= max_hurt_time as f32;
    hurt = mc_math::sin(hurt * hurt * hurt * hurt * std::f32::consts::PI);
    Mat4::from_rotation_z((-hurt * 14.0 * scale).to_radians())
}

// ---- the hooks -----------------------------------------------------------------------------

impl Client {
    /// Whether the world has the input: no screen open and no diagnostic run.
    fn m3_in_game(&self) -> bool {
        matches!(self.ui, Ui::InGame) && self.diagnostic_progress.is_none()
    }

    /// Once per frame where the live bindings and the shell's input
    /// state are in scope: sync, sample the toggle keys, drive the zoom, and tell the shell which
    /// look path Raw Mouse Input selects.
    pub(crate) fn m3_shell_frame(
        &mut self,
        input: &mut InputState,
        bindings: &KeyBindings,
        dt: f32,
    ) {
        self.m3.sync(&self.mods.settings);
        let in_game = self.m3_in_game();
        // Both lookups below search by name or hash, so they run only while their mod is on.
        if self.m3.toggle.enabled {
            // While a replay is watched, Sprint and Sneak steer its free
            // camera (fast, down): they are not the player's, and a tap there used to flip the
            // saved sprint toggle for the next server.
            let toggle_in_game = in_game && self.tw_play.playback.is_none();
            // The keyboard itself is asked only while a screen is up or a
            // key held over from one is still down, so play costs what it did.
            if !toggle_in_game || self.m3.toggle.holding_over() {
                let held = |name: &str| {
                    bindings
                        .get(name)
                        .is_some_and(|binding| input.is_down(binding.binding()))
                };
                let (sprint_held, sneak_held) = (held(keys::SPRINT), held(keys::SNEAK));
                self.m3
                    .toggle
                    .note_physical(sprint_held, sneak_held, toggle_in_game);
            }
            self.m3.toggle.sample(
                bindings.is_down(keys::SPRINT),
                bindings.is_down(keys::SNEAK),
                toggle_in_game,
            );
        }
        let held = in_game && self.m3.zoom.enabled && input.is_down(self.m3.zoom.key);
        self.m3.zoom.drive(held, dt);
        input.set_os_pointer_look(!self.m3.raw_input);
    }

    /// The tick's movement input with Toggle Sprint's keys held.
    pub(crate) fn toggle_sprint_input(&mut self, input: MoveInput) -> MoveInput {
        self.m3.sync(&self.mods.settings);
        let in_game = self.m3_in_game();
        self.m3.toggle.apply(input, in_game)
    }

    /// At the end of `Client::frame`: hide the held item while zoomed, set
    /// the toggle-sprint HUD status, and persist the toggle from a calm screen.
    pub(crate) fn m3_frame(&mut self) {
        self.m3.sync(&self.mods.settings);
        if self.m3.zoom.hides_hand() {
            // `drive_first_person` rebuilt the visibility earlier this frame, so this override
            // lasts exactly one frame.
            if let Some(render) = self.render.as_ref() {
                render
                    .first_person
                    .with(|phase| phase.render_state.visibility.hide_gui = true);
            }
        }
        if self.m3.toggle.enabled {
            let (flying, riding, sprinting) =
                self.session
                    .as_ref()
                    .map_or((false, false, false), |session| {
                        let state = session.predictor().map(|predictor| &predictor.state);
                        (
                            state.is_some_and(|state| state.flying),
                            session.riding().is_some(),
                            state.is_some_and(|state| state.sprinting),
                        )
                    });
            self.mods.hud_data.sprint_status = if self.session.is_some() {
                self.m3.toggle.status(flying, riding, sprinting)
            } else {
                SprintStatus::Hidden
            };
        }
        if self.m3.toggle.stored != Some(self.m3.toggle.sprint_toggled) && self.m3_calm_screen() {
            self.persist_toggle_state();
        }
    }

    /// Screens where a settings save cannot land in the middle of a fight.
    fn m3_calm_screen(&self) -> bool {
        self.session.is_none()
            || matches!(
                self.ui,
                Ui::Pause(_)
                    | Ui::GameOver(_)
                    | Ui::MainMenu(_)
                    | Ui::Disconnected(_)
                    | Ui::Multiplayer(_)
                    | Ui::Options(_)
            )
    }

    /// Write the sprint toggle into the store if it changed (the framework saves the file
    /// later, off the main thread). Called from calm screens and at shutdown, never from
    /// in-game input.
    pub(crate) fn persist_toggle_state(&mut self) {
        let toggled = self.m3.toggle.sprint_toggled;
        if self.m3.toggle.stored == Some(toggled) {
            return;
        }
        self.m3.toggle.stored = Some(toggled);
        self.mods.settings.set_text(
            TOGGLE_SPRINT,
            SPRINT_TOGGLED_ID,
            if toggled { "on" } else { "off" },
        );
    }

    /// The world camera's FOV multiplier divided by the zoom.
    #[must_use]
    pub(crate) fn zoom_fov(&self, f: f32) -> f32 {
        self.m3.zoom.fov(f)
    }

    /// The `mouseSensitivity` both mouse-look sites use (the frame's
    /// `look_delta_raw` and the tick's smooth-camera filter): vanilla's setting, or the zoom's
    /// vanilla-equivalent one while it is held.
    #[must_use]
    pub(crate) fn look_sensitivity(&self) -> f32 {
        self.m3
            .zoom
            .mouse_sensitivity(self.settings.mouse_sensitivity)
    }

    /// The flag both smooth-camera sites pass: vanilla's setting, or the zoom's cinematic
    /// camera. `settings.smooth_camera` itself is never written, so `options.txt` never
    /// records a zoom.
    #[must_use]
    pub(crate) fn smooth_camera_on(&self) -> bool {
        self.settings.smooth_camera || self.m3.zoom.smooth_camera_active()
    }

    /// The top of `mouse_wheel`: `true` when the zoom consumed the notch.
    pub(crate) fn zoom_wheel(&mut self, delta: f32) -> bool {
        matches!(self.ui, Ui::InGame) && self.m3.zoom.wheel(delta)
    }

    /// Whether [`Client::zoom_wheel`] would consume a notch now, without zooming (the 1.7.10
    /// screen phase keeps a zoom notch in the frame, `app/send_sites_v5.rs`).
    pub(crate) fn zoom_takes_wheel(&self) -> bool {
        matches!(self.ui, Ui::InGame) && self.m3.zoom.takes_wheel()
    }

    /// Dynamic FOV's flattened target. Unchanged at the vanilla
    /// settings (on, 100 %, bow zoom) or when there is no world.
    #[must_use]
    pub(crate) fn static_fov_target(&self, target: f32, bow_elapsed: Option<i32>) -> f32 {
        let Some(fov) = self.m3.fov else {
            return target;
        };
        if self.session.is_none() {
            return target;
        }
        let bow = bow_elapsed.map_or(1.0, crate::app::interact::bow_fov_multiplier);
        static_fov_target(target, bow, fov)
    }

    /// No Hurt Cam's scaled roll. `hurt` is
    /// vanilla's matrix, returned as is while the mod is off or at 100%.
    #[must_use]
    pub(crate) fn hurt_cam(
        &self,
        hurt: Mat4,
        hurt_time: i32,
        max_hurt_time: i32,
        partial_ticks: f32,
    ) -> Mat4 {
        match self.m3.hurt_cam {
            Some(scale) if scale < 1.0 => {
                scaled_hurt_camera_transform(hurt_time, max_hurt_time, partial_ticks, scale)
            }
            _ => hurt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::KeyCode;

    fn enabled(id: &str) -> ModSettings {
        let mut settings = ModSettings::new();
        settings.set_enabled(id, true);
        settings
    }

    fn bindings_with(down: &[&str]) -> KeyBindings {
        let mut keys = KeyBindings::vanilla();
        for name in down {
            let input = keys.get(name).expect("vanilla binding").binding();
            keys.set_state(input, true);
        }
        keys
    }

    #[test]
    fn defaults_are_todays_behaviour() {
        let state = M3State::new(&ModSettings::new());
        assert!(
            state.raw_input,
            "raw input is on by default: the WM_INPUT path"
        );
        assert!(
            !state.exclusive_fullscreen,
            "borderless is on by default: F11 stays borderless"
        );
        assert!(state.fov.is_none() && state.hurt_cam.is_none());
        assert!(!state.toggle.enabled && !state.zoom.enabled);
        assert_eq!(state.zoom.divisor, 1.0);
    }

    #[test]
    fn sync_is_generation_gated_and_reads_every_mod() {
        let mut settings = ModSettings::new();
        let mut state = M3State::new(&settings);
        settings.set_enabled(RAW_INPUT, false);
        settings.set_enabled(BORDERLESS_FULLSCREEN, false);
        settings.set_enabled(DYNAMIC_FOV, true);
        settings.set_f32(DYNAMIC_FOV, "strength", 25.0);
        settings.set_enabled(NO_HURT_CAM, true);
        settings.set_f32(NO_HURT_CAM, "intensity", 40.0);
        state.sync(&settings);
        assert!(!state.raw_input);
        assert!(state.exclusive_fullscreen);
        assert_eq!(
            state.fov,
            Some(StaticFov {
                strength: 0.25,
                bow_zoom: true
            })
        );
        assert_eq!(state.hurt_cam, Some(0.4));
        // Same generation: a stale field is not re-read.
        state.raw_input = true;
        state.sync(&settings);
        assert!(state.raw_input, "no re-read without a generation change");
    }

    #[test]
    fn a_tap_toggles_and_the_input_then_reads_as_a_held_sprint_key() {
        let mut settings = enabled(TOGGLE_SPRINT);
        let mut state = M3State::new(&settings);
        state.toggle.sample(false, false, true);
        state.toggle.sample(true, false, true);
        assert!(state.toggle.sprint_toggled, "rising edge toggles on");
        state.toggle.sample(true, false, true);
        assert!(state.toggle.sprint_toggled, "holding is not another edge");
        state.toggle.sample(false, false, true);
        assert!(state.toggle.sprint_toggled, "release keeps it on");

        // The MoveInput a toggled player builds is exactly the one a held Ctrl builds.
        let walking = bindings_with(&[keys::FORWARD]);
        let holding = bindings_with(&[keys::FORWARD, keys::SPRINT]);
        let toggled = state.toggle.apply(
            crate::app::input::move_input(&walking, 12.5, -3.0, false),
            true,
        );
        let held = crate::app::input::move_input(&holding, 12.5, -3.0, false);
        assert_eq!(toggled, held);
        // Not in game (any screen): nothing is held, exactly as vanilla's unpressed bindings.
        let idle = crate::app::input::move_input(&walking, 0.0, 0.0, false);
        assert_eq!(state.toggle.apply(idle, false), idle);

        // A second tap toggles off; the input is vanilla again.
        state.toggle.sample(true, false, true);
        assert!(!state.toggle.sprint_toggled);
        let plain = crate::app::input::move_input(&walking, 0.0, 0.0, false);
        assert_eq!(state.toggle.apply(plain, true), plain);

        // Turning the mod off stops the OR at once but keeps the toggle for later.
        state.toggle.sample(false, false, true);
        state.toggle.sample(true, false, true);
        assert!(state.toggle.sprint_toggled);
        settings.set_enabled(TOGGLE_SPRINT, false);
        state.sync(&settings);
        assert_eq!(state.toggle.apply(plain, true), plain);
        assert!(state.toggle.sprint_toggled);
    }

    /// The loopback test's case: the mod is on from the settings file and the very first
    /// in-game sample already sees the tap. (The first version treated "down at enable" as
    /// held-before and swallowed exactly this press.)
    #[test]
    fn the_first_press_after_start_up_or_enabling_is_a_tap() {
        let mut state = M3State::new(&enabled(TOGGLE_SPRINT));
        state.toggle.sample(true, false, true);
        assert!(
            state.toggle.sprint_toggled,
            "first in-game frame after start-up"
        );

        let mut settings = ModSettings::new();
        let mut state = M3State::new(&settings);
        state.toggle.sample(false, false, false);
        settings.set_enabled(TOGGLE_SPRINT, true);
        state.sync(&settings);
        state.toggle.sample(true, false, true);
        assert!(
            state.toggle.sprint_toggled,
            "first in-game frame after enabling"
        );
    }

    #[test]
    fn toggle_sneak_needs_its_option_and_any_screen_releases_it() {
        let mut settings = enabled(TOGGLE_SPRINT);
        let mut state = M3State::new(&settings);
        state.toggle.sample(false, false, true);
        state.toggle.sample(false, true, true);
        assert!(
            !state.toggle.sneak_toggled,
            "toggle sneak is off by default"
        );
        settings.set_bool(TOGGLE_SPRINT, "toggle_sneak", true);
        state.sync(&settings);
        state.toggle.sample(false, false, true);
        state.toggle.sample(false, true, true);
        assert!(state.toggle.sneak_toggled);
        let walking = bindings_with(&[keys::FORWARD]);
        let sneaking = bindings_with(&[keys::FORWARD, keys::SNEAK]);
        assert_eq!(
            state.toggle.apply(
                crate::app::input::move_input(&walking, 0.0, 0.0, false),
                true
            ),
            crate::app::input::move_input(&sneaking, 0.0, 0.0, false)
        );
        // Opening a screen releases it, and the screen itself never sneaks.
        state.toggle.sample(false, false, false);
        assert!(!state.toggle.sneak_toggled);
        // Nothing toggles from inside a screen either.
        state.toggle.sample(true, true, false);
        assert!(!state.toggle.sprint_toggled && !state.toggle.sneak_toggled);
    }

    #[test]
    fn the_stored_toggle_is_adopted_only_when_the_store_changes() {
        let mut settings = enabled(TOGGLE_SPRINT);
        settings.set_text(TOGGLE_SPRINT, SPRINT_TOGGLED_ID, "on");
        let mut state = M3State::new(&settings);
        assert!(state.toggle.sprint_toggled, "restored at start-up");
        // Toggled off in game, then an unrelated option changes: the live state stays.
        state.toggle.sample(false, false, true);
        state.toggle.sample(true, false, true);
        assert!(!state.toggle.sprint_toggled);
        settings.set_color(TOGGLE_SPRINT, "text_color", 0xFF12_3456);
        state.sync(&settings);
        assert!(
            !state.toggle.sprint_toggled,
            "a stale stored 'on' must not win"
        );
        // A real change of the stored value (e.g. reset options) is adopted.
        settings.reset_options(TOGGLE_SPRINT);
        state.sync(&settings);
        assert!(!state.toggle.sprint_toggled);
        assert_eq!(state.toggle.stored, Some(false));
    }

    #[test]
    fn the_hud_status_follows_its_priority_order() {
        let mut settings = enabled(TOGGLE_SPRINT);
        settings.set_bool(TOGGLE_SPRINT, "toggle_sneak", true);
        let mut state = M3State::new(&settings);
        let toggle = &mut state.toggle;
        toggle.sample(false, false, true);
        assert_eq!(toggle.status(false, false, false), SprintStatus::Hidden);
        assert_eq!(
            toggle.status(false, false, true),
            SprintStatus::SprintVanilla
        );
        toggle.sample(true, false, true);
        assert_eq!(
            toggle.status(false, false, true),
            SprintStatus::SprintToggled
        );
        assert_eq!(
            toggle.status(false, false, false),
            SprintStatus::SprintToggled,
            "shown while standing still, so the armed toggle is visible"
        );
        toggle.sample(false, false, true);
        toggle.sample(true, false, true);
        assert!(!toggle.sprint_toggled);
        assert_eq!(toggle.status(false, false, true), SprintStatus::SprintHeld);
        toggle.sample(false, true, true);
        assert_eq!(
            toggle.status(false, false, true),
            SprintStatus::SneakToggled
        );
        toggle.sample(false, false, true);
        toggle.sample(false, true, true);
        assert!(!toggle.sneak_toggled);
        assert_eq!(toggle.status(false, false, false), SprintStatus::SneakHeld);
        assert_eq!(toggle.status(false, true, false), SprintStatus::Riding);
        assert_eq!(toggle.status(true, true, false), SprintStatus::Flying);
    }

    #[test]
    fn zoom_divides_the_fov_and_eases_frame_rate_independently() {
        let settings = enabled(ZOOM);
        let mut state = M3State::new(&settings);
        let zoom = &mut state.zoom;
        assert_eq!(
            zoom.key,
            Input::Key(KeyCode::KeyC),
            "the default zoom key is C"
        );
        assert_eq!(zoom.fov(1.15), 1.15, "no zoom: untouched");

        // 150 ms at 1000 fps and at 60 fps both settle on the 4x target.
        let mut fast = zoom.clone();
        for _ in 0..150 {
            fast.drive(true, 0.001);
        }
        let mut slow = zoom.clone();
        for _ in 0..9 {
            slow.drive(true, 1.0 / 60.0);
        }
        assert!((fast.divisor - 4.0).abs() < 0.1, "{}", fast.divisor);
        assert!((slow.divisor - 4.0).abs() < 0.1, "{}", slow.divisor);
        // ...and snap exactly once close enough.
        for _ in 0..200 {
            fast.drive(true, 0.001);
        }
        assert_eq!(fast.divisor, 4.0);
        assert_eq!(fast.fov(1.0), 0.25);
        // Release eases back to exactly 1.
        for _ in 0..400 {
            fast.drive(false, 0.001);
        }
        assert_eq!(fast.divisor, 1.0);
        assert_eq!(fast.fov(1.15), 1.15);

        // One easing step is the closed form.
        zoom.drive(true, 0.01);
        let expected = 4.0 + (1.0 - 4.0) * (-0.01_f32 / ZOOM_EASE_SECONDS).exp();
        assert!((zoom.divisor - expected).abs() < 1e-6);
    }

    #[test]
    fn zoom_without_smoothing_jumps_and_turning_the_mod_off_resets_at_once() {
        let mut settings = enabled(ZOOM);
        settings.set_bool(ZOOM, "smooth_zoom", false);
        settings.set_f32(ZOOM, "factor", 6.0);
        let mut state = M3State::new(&settings);
        state.zoom.drive(true, 0.001);
        assert_eq!(state.zoom.divisor, 6.0);
        assert!(state.zoom.hides_hand());
        settings.set_enabled(ZOOM, false);
        state.sync(&settings);
        state.zoom.drive(true, 0.001);
        assert_eq!(state.zoom.divisor, 1.0);
        assert!(!state.zoom.active && !state.zoom.hides_hand());
    }

    #[test]
    fn the_wheel_adjusts_the_zoom_only_while_zoomed_and_resets_each_zoom() {
        let mut settings = enabled(ZOOM);
        settings.set_bool(ZOOM, "smooth_zoom", false);
        let mut state = M3State::new(&settings);
        assert!(
            !state.zoom.wheel(1.0),
            "not zoomed: the hotbar keeps the wheel"
        );
        state.zoom.drive(true, 0.01);
        assert!(state.zoom.wheel(1.0));
        assert!((state.zoom.target - 5.0).abs() < 1e-5);
        assert!(state.zoom.wheel(-2.0));
        assert!((state.zoom.target - 3.2).abs() < 1e-5);
        for _ in 0..40 {
            state.zoom.wheel(1.0);
        }
        assert_eq!(state.zoom.target, ZOOM_SCROLL_RANGE.1);
        for _ in 0..80 {
            state.zoom.wheel(-1.0);
        }
        assert_eq!(state.zoom.target, ZOOM_SCROLL_RANGE.0);
        // A new zoom starts from the configured factor again.
        state.zoom.drive(false, 0.01);
        state.zoom.drive(true, 0.01);
        assert_eq!(state.zoom.target, 4.0);
        // Scroll off: never consumed.
        settings.set_bool(ZOOM, "scroll", false);
        state.sync(&settings);
        assert!(!state.zoom.wheel(1.0));
    }

    #[test]
    fn zoomed_sensitivity_and_smooth_camera_apply_only_while_held() {
        let mut settings = enabled(ZOOM);
        settings.set_f32(ZOOM, "sensitivity", 50.0);
        let mut state = M3State::new(&settings);
        assert_eq!(state.zoom.mouse_sensitivity(0.5), 0.5);
        assert!(!state.zoom.smooth_camera_active());
        state.zoom.drive(true, 0.001);
        // java: `f = s * 0.6F + 0.2F; f1 = f * f * f * 8.0F` - the zoomed sensitivity is a real
        // `mouseSensitivity` whose step is half of 0.5's (1.0), to float precision.
        let step = |s: f32| {
            let f = s * 0.6 + 0.2;
            f * f * f * 8.0
        };
        let zoomed = state.zoom.mouse_sensitivity(0.5);
        assert!((step(zoomed) - 0.5).abs() < 1e-5, "{}", step(zoomed));
        assert!(state.zoom.smooth_camera_active());
        settings.set_bool(ZOOM, "smooth_camera", false);
        state.sync(&settings);
        assert!(!state.zoom.smooth_camera_active());
    }

    #[test]
    fn static_fov_flattens_movement_and_optionally_the_bow() {
        let sprint = 1.15_f32;
        let bow = crate::app::interact::bow_fov_multiplier(20);
        assert!((bow - 0.85).abs() < 1e-6);
        let off = StaticFov::OFF;
        assert_eq!(static_fov_target(sprint, 1.0, off), 1.0, "fully static");
        assert_eq!(static_fov_target(sprint * bow, bow, off), 1.0);
        let only_bow = StaticFov {
            strength: 0.0,
            bow_zoom: true,
        };
        assert!((static_fov_target(sprint * bow, bow, only_bow) - bow).abs() < 1e-6);
        let half = StaticFov {
            strength: 0.5,
            bow_zoom: true,
        };
        assert!((static_fov_target(sprint, 1.0, half) - 1.075).abs() < 1e-6);
        assert!((static_fov_target(sprint * bow, bow, half) - 1.075 * bow).abs() < 1e-6);
        // Bow zoom off drops the bow's narrowing whatever the strength, even at 100 %.
        let no_bow = StaticFov {
            strength: 1.0,
            bow_zoom: false,
        };
        assert!((static_fov_target(sprint * bow, bow, no_bow) - sprint).abs() < 1e-6);
        assert_eq!(
            static_fov_target(sprint, 1.0, no_bow),
            sprint,
            "no bow drawn"
        );
        let vanilla = StaticFov {
            strength: 1.0,
            bow_zoom: true,
        };
        assert_eq!(static_fov_target(sprint * bow, bow, vanilla), sprint * bow);
    }

    /// "Dynamic FOV", default on. The default is vanilla and keeps no
    /// state; off is the old "No Dynamic FOV" at 0 % without the bow (the user's setting).
    #[test]
    fn dynamic_fov_is_vanilla_by_default_and_off_keeps_the_fov_still() {
        let mut settings = ModSettings::new();
        assert!(settings.enabled(DYNAMIC_FOV));
        assert_eq!(
            StaticFov::from_settings(&settings),
            None,
            "vanilla: no work"
        );
        settings.set_enabled(DYNAMIC_FOV, false);
        assert_eq!(StaticFov::from_settings(&settings), Some(StaticFov::OFF));
        // Off ignores the options: the FOV never moves.
        settings.set_f32(DYNAMIC_FOV, "strength", 60.0);
        assert_eq!(StaticFov::from_settings(&settings), Some(StaticFov::OFF));
        settings.set_enabled(DYNAMIC_FOV, true);
        assert_eq!(
            StaticFov::from_settings(&settings),
            Some(StaticFov {
                strength: 0.6,
                bow_zoom: true
            })
        );
        settings.set_f32(DYNAMIC_FOV, "strength", 100.0);
        settings.set_bool(DYNAMIC_FOV, "bow_zoom", false);
        assert_eq!(
            StaticFov::from_settings(&settings),
            Some(StaticFov {
                strength: 1.0,
                bow_zoom: false
            })
        );
        // The user's saved file ("No Dynamic FOV" on at 0 %, no bow) loads as off: the same
        // per-tick target as before the rename.
        let text = r#"{"mods":{"dynamic_fov":{"enabled":true,
            "options":{"keep_bow":false,"strength":0.0}}}}"#;
        let (users, _) = ModSettings::from_json_str(text);
        assert_eq!(StaticFov::from_settings(&users), Some(StaticFov::OFF));
        // Before the rename, strength 0 without keep_bow gave `1 + (m - 1) * 0` times
        // `1 + (b - 1) * 0`: exactly 1.0 while sprinting with a drawn bow.
        let bow = crate::app::interact::bow_fov_multiplier(12);
        assert_eq!(static_fov_target(1.15 * bow, bow, StaticFov::OFF), 1.0);
    }

    /// A client over the real assets in a temp game dir. Skips without the vanilla jar (set
    /// `MC_JARS_DIR`).
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = crate::app::config::ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: crate::app::config::LoginMode::Offline("Dev".to_owned()),
            ..crate::app::config::ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// The driver hooks, end to end on a real `Client`: the FOV, the wheel (no slot change, so no
    /// C09), the smooth camera flag (never written to the settings), the shell's look path and
    /// fullscreen kind.
    #[test]
    fn the_client_hooks_route_every_mod_through_the_snapshot() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        client.ui = Ui::InGame;
        let bindings = KeyBindings::vanilla();
        let mut input = InputState::new();
        client.m3_shell_frame(&mut input, &bindings, 0.004);
        let unzoomed = client.fov_modifier(mc_math::DVec3::ZERO, 0.0);
        assert!(!input.os_pointer_look(), "raw input on by default");
        assert!(!mc_render::ShellApp::wants_exclusive_fullscreen(&client));

        client.mods.settings.set_enabled(ZOOM, true);
        client.mods.settings.set_bool(ZOOM, "smooth_zoom", false);
        input.on_key(KeyCode::KeyC, true, false);
        client.m3_shell_frame(&mut input, &bindings, 0.004);
        assert_eq!(
            client.fov_modifier(mc_math::DVec3::ZERO, 0.0),
            unzoomed / 4.0
        );
        assert!(client.smooth_camera_on() && !client.settings.smooth_camera);
        assert!(client.m3.zoom.hides_hand());

        client.player_hud.selected_slot = 0;
        client.mouse_wheel(1.0);
        // The hotbar half of the wheel lands in the next tick's input section.
        let mut keys = client.settings.keys.clone();
        client.handle_game_input(&InputState::new(), &mut keys);
        assert_eq!(
            client.player_hud.selected_slot, 0,
            "zoomed: the wheel never reaches the hotbar"
        );
        client.m3_shell_frame(&mut input, &bindings, 0.004);
        assert!((client.fov_modifier(mc_math::DVec3::ZERO, 0.0) - unzoomed / 5.0).abs() < 1e-6);

        input.on_key(KeyCode::KeyC, false, false);
        client.m3_shell_frame(&mut input, &bindings, 0.004);
        assert_eq!(client.fov_modifier(mc_math::DVec3::ZERO, 0.0), unzoomed);
        assert!(!client.smooth_camera_on());
        client.mouse_wheel(1.0);
        client.handle_game_input(&InputState::new(), &mut keys);
        assert_ne!(
            client.player_hud.selected_slot, 0,
            "control: unzoomed, the wheel scrolls"
        );

        client.mods.settings.set_enabled(RAW_INPUT, false);
        client
            .mods
            .settings
            .set_enabled(BORDERLESS_FULLSCREEN, false);
        client.m3_shell_frame(&mut input, &bindings, 0.004);
        assert!(input.os_pointer_look());
        assert!(mc_render::ShellApp::wants_exclusive_fullscreen(&client));
    }

    /// The held item is hidden while zoomed, through the live frame: `drive_first_person`
    /// rebuilds the first-person visibility, then `<<MOD-M3:frame>>` overrides it for that
    /// frame only. F1's own flag is never touched.
    #[test]
    fn zoom_hides_the_held_item_for_exactly_the_zoomed_frames() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let gpu = match mc_render::Gpu::headless(64, 64, mc_render::GpuOptions::default()) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: no GPU adapter: {error}");
                return;
            }
        };
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, 64, 64);
        let mut camera = mc_render::Camera::default();
        client.ui = Ui::InGame;
        client.mods.settings.set_enabled(ZOOM, true);
        client.mods.settings.set_bool(ZOOM, "smooth_zoom", false);
        let bindings = KeyBindings::vanilla();
        let mut input = InputState::new();
        let hand_hidden = |client: &Client| {
            client
                .render
                .as_ref()
                .expect("installed")
                .first_person
                .with(|phase| phase.render_state.visibility.hide_gui)
        };

        client.m3_shell_frame(&mut input, &bindings, 0.004);
        client.frame(&gpu, &mut camera, 0.0);
        assert!(!hand_hidden(&client), "not zoomed: the hand is drawn");
        input.on_key(KeyCode::KeyC, true, false);
        client.m3_shell_frame(&mut input, &bindings, 0.004);
        client.frame(&gpu, &mut camera, 0.0);
        assert!(hand_hidden(&client), "zoomed: no hand");
        assert!(!client.hide_gui, "F1's own flag is untouched");
        input.on_key(KeyCode::KeyC, false, false);
        client.m3_shell_frame(&mut input, &bindings, 0.004);
        client.frame(&gpu, &mut camera, 0.0);
        assert!(
            !hand_hidden(&client),
            "released: the next frame draws it again"
        );
    }

    /// While a replay is watched, Ctrl and Shift steer its free camera
    /// (fast, down). Toggle Sprint took those presses as taps: watching a replay flipped the
    /// player's sprint toggle, which is saved, so the next server started with it changed. The
    /// same taps with the replay closed still toggle (the control).
    #[test]
    fn the_replay_viewers_camera_keys_never_flip_the_toggles() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        client.mods.settings.set_enabled(TOGGLE_SPRINT, true);
        client
            .mods
            .settings
            .set_bool(TOGGLE_SPRINT, "toggle_sneak", true);
        let records: Vec<mc_replay::Record> = (0..10u64)
            .map(|i| mc_replay::Record {
                time: i * 100_000,
                stream: mc_replay::StreamId::Packet,
                payload: vec![0x02, 0, 0],
            })
            .collect();
        let meta = mc_replay::ReplayMeta {
            duration: 1_000_000,
            ..mc_replay::ReplayMeta::default()
        };
        client.replay_open_records(meta, records).expect("opens");
        // The screen a replay is watched under.
        client.ui = Ui::InGame;
        let mut keys = KeyBindings::vanilla();
        let mut raw = InputState::new();
        let tap = |client: &mut Client, raw: &mut InputState, keys: &mut KeyBindings| {
            for code in [KeyCode::ControlLeft, KeyCode::ShiftLeft] {
                raw.on_key(code, false, false);
                keys.set_state(Input::Key(code), false);
            }
            client.m3_shell_frame(raw, keys, 0.004);
            for code in [KeyCode::ControlLeft, KeyCode::ShiftLeft] {
                raw.on_key(code, true, false);
                keys.set_state(Input::Key(code), true);
            }
            client.m3_shell_frame(raw, keys, 0.004);
        };
        tap(&mut client, &mut raw, &mut keys);
        tap(&mut client, &mut raw, &mut keys);
        tap(&mut client, &mut raw, &mut keys);
        assert!(
            !client.m3.toggle.sprint_toggled && !client.m3.toggle.sneak_toggled,
            "camera keys in a replay are not toggle taps"
        );
        client.replay_close();
        client.ui = Ui::InGame;
        tap(&mut client, &mut raw, &mut keys);
        assert!(
            client.m3.toggle.sprint_toggled && client.m3.toggle.sneak_toggled,
            "control: in game the same taps toggle"
        );
    }

    #[test]
    fn hurt_cam_scale_one_is_vanilla_and_zero_is_the_identity() {
        let close = |a: Mat4, b: Mat4| {
            a.to_cols_array()
                .iter()
                .zip(b.to_cols_array())
                .all(|(x, y)| (x - y).abs() < 1e-6)
        };
        for (hurt_time, pt) in [(10, 0.0), (9, 0.37), (5, 0.5), (1, 0.99), (0, 0.5)] {
            let vanilla = super::super::super::hurt_camera_transform(hurt_time, 10, 0.0, pt);
            assert!(close(
                scaled_hurt_camera_transform(hurt_time, 10, pt, 1.0),
                vanilla
            ));
            assert_eq!(
                scaled_hurt_camera_transform(hurt_time, 10, pt, 0.0),
                Mat4::IDENTITY
            );
        }
        // Half intensity is half the roll angle.
        let wave = mc_math::sin(0.5_f32.powi(4) * std::f32::consts::PI);
        let half = scaled_hurt_camera_transform(5, 10, 0.0, 0.5);
        assert!(close(
            half,
            Mat4::from_rotation_z((-wave * 14.0 * 0.5).to_radians())
        ));
    }

    /// A 1.7.10 session's screen pass asks [`Client::zoom_takes_wheel`] whether a wheel notch is
    /// the zoom's before it leaves the notch to the frame (`app/send_sites_v5.rs`), so the copy
    /// must answer what [`Client::zoom_wheel`] then does: on every kind of screen, for every
    /// state of the three flags the two gates read (the mod, its scroll option, the key held).
    #[test]
    fn zoom_takes_wheel_answers_what_zoom_wheel_does() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let mut taken = 0;
        for screen in ["in game", "chat", "title"] {
            for flags in 0..8_u8 {
                match screen {
                    "in game" => client.ui = Ui::InGame,
                    "chat" => client.open_chat(""),
                    _ => client.ui = Ui::MainMenu(client.main_menu(String::new())),
                }
                let zoom = &mut client.m3.zoom;
                zoom.enabled = flags & 1 != 0;
                zoom.scroll = flags & 2 != 0;
                zoom.active = flags & 4 != 0;
                let takes = client.zoom_takes_wheel();
                let took = client.zoom_wheel(1.0);
                assert_eq!(takes, took, "{screen}, flags {flags:03b}");
                taken += usize::from(took);
            }
        }
        // Not vacuous: the zoom takes the notch in exactly one state (in game, all three set).
        assert_eq!(taken, 1);
    }
}
