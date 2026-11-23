//! The Profiles view: save and load whole mod setups (the user's
//! "an option for saving and loading entire mod profiles").
//!
//! A profile is the whole setup: every mod's switch and settings, HUD positions, pins and the
//! shared HUD style (`mc_mods::profiles`). The menu key and Color Grading's own saved looks are
//! not part of it.
//!
//! * **Window 1:** how many profiles are saved.
//! * **Window 2:** a name field and Save, the saved profiles (the one that matches the current
//!   setup says "Current"), then the built-in "Fresh install" and, after a load, "Before last
//!   load".
//! * **Window 3:** the picked profile: Load, Save current here, Rename, Delete, and what loading
//!   it would change ("Turns on: FPS, CPS").
//!
//! **Nothing is lost without a confirm:** Load, saving over a profile and Delete each need a
//! second click (like Reset), and a load first keeps the current setup as "Before last load",
//! so it can be loaded back. File work runs on the library's worker thread, never in a frame.
//!
//! (Color Grading's own per-mod profiles stay in `details.rs`: a different feature.)

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use mc_mods::profiles::{self as setup, Notice, ProfileEntry, ProfileLibrary, SetupDiff};
use mc_mods::ModSettings;

use super::layout::{draw_title, Area};
use super::paint::{black, css, white, Fill, Painter, Rect, Rgba};
use super::sidebar::NavBadge;
use super::state::{NavRow, Pane, Target};
use super::text::{TextStyle, Weight};
use super::widgets::{draw_focus_ring, draw_wrapped, ms_since, shadowed, still_armed};
use super::{MenuEnv, MenuInput, ModMenuScreen};

/// How long a notice ("Saved “Bedwars”") stays up.
const NOTICE_MS: f32 = 6000.0;
/// A profile row in window 2.
const ROW_H: f32 = 44.0;
/// The save row (field and button).
const FIELD_H: f32 = 34.0;
/// Window 3's buttons.
const BUTTON_H: f32 = 28.0;
/// The green of a done notice.
const GOOD: Rgba = css(140, 230, 160, 0.95);
/// The red of a failed one.
const BAD: Rgba = css(255, 140, 120, 0.95);

/// Which profile a row or window 3 stands for.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) enum Pick {
    /// The built-in fresh install.
    #[default]
    Fresh,
    /// The setup from before the last load.
    Backup,
    /// A saved profile, by file.
    Saved(PathBuf),
}

/// A text field of the view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Field {
    /// Window 2's "Name this setup".
    SaveName,
    /// Window 3's new name for the picked profile.
    Rename,
}

/// A button that needs a second click.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Armed {
    Load(Pick),
    /// Save over the profile with this file (window 2's Save with a taken name, or window 3's
    /// "Save current here").
    Replace(PathBuf),
    Delete(PathBuf),
}

/// The Profiles view's own hit targets. Must stay `Copy` and `PartialEq`: it lives inside the
/// shell's `Target`. Rows are positions in the list drawn last frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProfilesTarget {
    /// Pick the fresh install.
    Fresh,
    /// Pick "Before last load".
    Backup,
    /// Pick saved profile `n` (in the list drawn last frame).
    Row(u16),
    /// A text field: focus it.
    Field(FieldId),
    /// Window 2's Save.
    Save,
    /// Window 3: load the picked profile (armed).
    Load,
    /// Window 3: save the current setup over the picked profile (armed).
    Update,
    /// Window 3: rename the picked profile (opens the field; a second click commits).
    Rename,
    /// Window 3: delete the picked profile (armed).
    Delete,
}

/// [`Field`] as a `Copy` id for targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FieldId {
    SaveName,
    Rename,
}

/// The Profiles view's state.
#[derive(Debug, Default)]
pub(super) struct ProfilesState {
    /// The profiles folder and its worker, opened on the first frame with a game folder.
    pub(super) library: Option<ProfileLibrary>,
    /// The library's generation the copies below are from.
    seen: u64,
    /// The saved profiles (a copy, refreshed when the library moves).
    pub(super) entries: Vec<ProfileEntry>,
    /// "Before last load".
    backup: Option<ProfileEntry>,
    /// The settings generation `current` was computed at (0: never).
    current_gen: u64,
    /// Per saved profile: it holds exactly the current setup.
    pub(super) current: Vec<bool>,
    /// A profile being saved or renamed, by name: picked once the library lists it.
    pending_pick: Option<String>,
    /// The fresh install's setup, made once.
    fresh: Option<Arc<ModSettings>>,
    /// The library has read the folder (copied when the library moves).
    scanned: bool,
    /// Per saved profile, and for the backup: the mods it changes from a fresh install (their
    /// switch or settings), computed when the library moves.
    changed: Vec<Vec<usize>>,
    backup_changed: Vec<usize>,
    /// The fresh install is the current setup; the backup is.
    fresh_current: bool,
    backup_current: bool,
    /// What window 3 shows.
    pick: Pick,
    /// What loading the picked profile would change, and for which pick and generation.
    diff: Option<(Pick, u64, u64, SetupDiff)>,
    /// The focused text field, and the two fields' text.
    focus: Option<Field>,
    name: String,
    rename: String,
    /// Window 3's rename field is open (it keeps its text when a click elsewhere takes the
    /// keyboard; "Done" or Enter renames, Esc cancels).
    renaming: bool,
    armed: Option<(Armed, Instant)>,
    /// The last notice and when it came.
    notice: Option<(Notice, Instant)>,
    /// A notice the menu itself made (loads), shown like the library's.
    loaded: Option<(String, Instant)>,
    /// When the view first drew (the caret blinks from it).
    since: Option<Instant>,
}

impl ProfilesState {
    /// The saved profile with this file.
    fn entry(&self, file: &Path) -> Option<&ProfileEntry> {
        self.entries.iter().find(|entry| entry.file == file)
    }

    /// The fresh install's setup (made the first time it is needed).
    fn fresh(&mut self, settings: &ModSettings) -> Arc<ModSettings> {
        Arc::clone(
            self.fresh
                .get_or_insert_with(|| Arc::new(ModSettings::with_catalog(settings.catalog()))),
        )
    }

    /// The setup the pick stands for (`None`: a broken file, or no such profile any more).
    fn setup_of(&mut self, pick: &Pick, settings: &ModSettings) -> Option<Arc<ModSettings>> {
        match pick {
            Pick::Fresh => Some(self.fresh(settings)),
            Pick::Backup => self.backup.as_ref().and_then(|entry| entry.setup.clone()),
            Pick::Saved(file) => self.entry(file).and_then(|entry| entry.setup.clone()),
        }
    }

    fn is_armed(&self, armed: &Armed, now: Instant) -> bool {
        self.armed
            .as_ref()
            .is_some_and(|(what, at)| what == armed && still_armed(*at, now))
    }
}

/// "just now", "5 min ago", "3 h ago", "yesterday", "4 days ago", or the date.
fn ago(saved: Option<u64>) -> String {
    let Some(saved) = saved else {
        return String::new();
    };
    let secs = setup::now_unix().saturating_sub(saved);
    match secs {
        0..=59 => "just now".to_owned(),
        60..=3599 => format!("{} min ago", secs / 60),
        3600..=86_399 => format!("{} h ago", secs / 3600),
        86_400..=172_799 => "yesterday".to_owned(),
        _ if secs < 30 * 86_400 => format!("{} days ago", secs / 86_400),
        _ => date(saved),
    }
}

/// `YYYY-MM-DD` (UTC) for seconds since 1970 (the standard days-to-civil algorithm).
fn date(unix: u64) -> String {
    let days = i64::try_from(unix / 86_400).unwrap_or(0) + 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Mod names joined with commas, at most `max` of them, then "+N more".
fn names(settings: &ModSettings, mods: &[usize], max: usize) -> String {
    let catalog = settings.catalog();
    let shown: Vec<&str> = mods.iter().take(max).map(|&m| catalog[m].name).collect();
    let more = mods.len().saturating_sub(max);
    if more > 0 {
        format!("{} +{more} more", shown.join(", "))
    } else {
        shown.join(", ")
    }
}

/// The mods `setup` changes from a fresh install (`fresh`): their switch or any setting (not
/// where their HUD elements sit), as My setup counts them. In catalog order.
fn changed_from(fresh: &ModSettings, setup: &ModSettings) -> Vec<usize> {
    let diff = fresh.setup_diff(setup);
    let mut mods: Vec<usize> = diff
        .turns_on
        .iter()
        .chain(&diff.turns_off)
        .chain(&diff.settings)
        .copied()
        .collect();
    mods.sort_unstable();
    mods.dedup();
    mods
}

impl ModMenuScreen {
    /// Open the library (first frame with a game folder) and refresh the copies when the
    /// library or the settings moved. A `u64` compare or two on a quiet frame.
    fn sync_profiles(&mut self, env: &MenuEnv<'_>) {
        let state = &mut self.profiles;
        if state.library.is_none() {
            if let Some(dir) = env.game.game_dir {
                state.library = Some(ProfileLibrary::open(dir, env.settings.catalog()));
            }
        }
        let mut moved = false;
        if let Some(library) = &state.library {
            if library.generation() != state.seen {
                // Notices first (taking them moves the generation), then the generation, then
                // the copies: a change that lands while copying moves it again, so the next
                // frame catches it.
                if let Some(notice) = library.take_notices().pop() {
                    state.notice = Some((notice, env.now));
                }
                state.seen = library.generation();
                state.entries = library.entries();
                state.backup = library.backup();
                state.scanned = library.scanned();
                moved = true;
            }
        }
        let settings = env.settings;
        if moved {
            let fresh = state.fresh(settings);
            state.changed = state
                .entries
                .iter()
                .map(|entry| {
                    entry
                        .setup
                        .as_ref()
                        .map_or_else(Vec::new, |setup| changed_from(&fresh, setup))
                })
                .collect();
            state.backup_changed = state
                .backup
                .as_ref()
                .and_then(|entry| entry.setup.as_ref())
                .map_or_else(Vec::new, |setup| changed_from(&fresh, setup));
        }
        if moved || state.current_gen != settings.generation() {
            state.current_gen = settings.generation();
            state.current = state
                .entries
                .iter()
                .map(|entry| entry.setup.as_ref().is_some_and(|s| settings.setup_eq(s)))
                .collect();
            let fresh = state.fresh(settings);
            state.fresh_current = settings.setup_eq(&fresh);
            state.backup_current = state
                .backup
                .as_ref()
                .and_then(|entry| entry.setup.as_ref())
                .is_some_and(|s| settings.setup_eq(s));
        }
        // A profile just saved or renamed is picked once listed; a pick whose profile is gone
        // falls back to the fresh install.
        // (`busy` takes the library's lock: only asked when a pick is in doubt.)
        if let Some(name) = &state.pending_pick {
            let found = state
                .entries
                .iter()
                .find(|entry| entry.name.eq_ignore_ascii_case(name))
                .map(|entry| entry.file.clone());
            if let Some(file) = found {
                state.pick = Pick::Saved(file);
                state.pending_pick = None;
            } else if state.library.as_ref().is_none_or(|l| !l.busy()) {
                state.pending_pick = None;
            }
        }
        if let Pick::Saved(file) = &state.pick {
            if state.entry(file).is_none()
                && state.pending_pick.is_none()
                && state.library.as_ref().is_none_or(|l| !l.busy())
            {
                state.pick = Pick::Fresh;
            }
        }
        if state.pick == Pick::Backup && state.backup.is_none() {
            state.pick = Pick::Fresh;
        }
    }

    /// Window 1's badge for Profiles: how many are saved (none until the folder is read).
    pub(super) fn profiles_badge(&mut self, env: &MenuEnv<'_>) -> NavBadge {
        self.sync_profiles(env);
        NavBadge {
            count: self.profiles.scanned.then_some(self.profiles.entries.len()),
            warn: false,
        }
    }

    /// Window 2 while the Profiles view is open.
    pub(super) fn draw_profiles_list(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) {
        self.sync_profiles(env);
        self.profiles.since.get_or_insert(env.now);
        let saved = self.profiles.entries.len();
        let subtitle = match saved {
            0 => "Save and load whole setups".to_owned(),
            1 => "1 saved".to_owned(),
            n => format!("{n} saved"),
        };
        draw_title(painter, env, rect, "Profiles", &subtitle, 0.0, s);
        let area = self.begin_body(painter, rect, Pane::List, s);
        let mut y = draw_wrapped(
            painter,
            env,
            "A profile is your whole setup: every mod's switch and settings, where HUD \
             elements sit, and your pins.",
            area.x + 4.0 * s,
            area.y,
            area.w - 8.0 * s,
            TextStyle::new(11.5 * s),
            white(0.62),
            s,
        ) + 8.0 * s;
        let has_library = self.profiles.library.is_some();
        if has_library {
            y = self.draw_save_row(painter, env, &area, y);
        }
        y = section(painter, env, &area, y + 6.0 * s, "Saved");
        if !has_library {
            y = draw_wrapped(
                painter,
                env,
                "Profiles are saved in the game folder, next to mc-rs-mods.json.",
                area.x + 4.0 * s,
                y,
                area.w - 8.0 * s,
                TextStyle::new(11.5 * s),
                white(0.5),
                s,
            );
        } else if saved == 0 {
            let text = if self.profiles.scanned {
                "Nothing saved yet. Name your setup above and press Save."
            } else {
                "Reading the profiles folder\u{2026}"
            };
            y = draw_wrapped(
                painter,
                env,
                text,
                area.x + 4.0 * s,
                y,
                area.w - 8.0 * s,
                TextStyle::new(11.5 * s),
                white(0.5),
                s,
            );
        }
        let entries: Vec<(String, String, bool, bool, PathBuf)> = self
            .profiles
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let line = match &entry.setup {
                    Some(_) => {
                        let changed = self.profiles.changed.get(i).map_or(0, Vec::len);
                        let what = match changed {
                            0 => "Same as a fresh install".to_owned(),
                            1 => "1 mod changed".to_owned(),
                            n => format!("{n} mods changed"),
                        };
                        let when = ago(entry.saved_unix);
                        if when.is_empty() {
                            what
                        } else {
                            format!("{what}  \u{b7}  saved {when}")
                        }
                    }
                    None => "Can't be read".to_owned(),
                };
                let current = self.profiles.current.get(i).copied().unwrap_or(false);
                (
                    entry.name.clone(),
                    line,
                    current,
                    entry.setup.is_some(),
                    entry.file.clone(),
                )
            })
            .collect();
        for (i, (name, line, current, ok, file)) in entries.into_iter().enumerate() {
            let picked = self.profiles.pick == Pick::Saved(file);
            let target = ProfilesTarget::Row(u16::try_from(i).unwrap_or(u16::MAX));
            y = self.draw_profile_row(
                painter, env, &area, y, &name, &line, current, ok, picked, target,
            );
        }
        y = section(painter, env, &area, y + 8.0 * s, "Built in");
        let fresh_current = self.profiles.fresh_current;
        let picked = self.profiles.pick == Pick::Fresh;
        y = self.draw_profile_row(
            painter,
            env,
            &area,
            y,
            "Fresh install",
            "Every mod as a new install has it",
            fresh_current,
            true,
            picked,
            ProfilesTarget::Fresh,
        );
        if self.profiles.backup.is_some() {
            let when = ago(self.profiles.backup.as_ref().and_then(|e| e.saved_unix));
            let line = if when.is_empty() {
                "Your setup before the last load".to_owned()
            } else {
                format!("Your setup before the last load ({when})")
            };
            let current = self.profiles.backup_current;
            let picked = self.profiles.pick == Pick::Backup;
            y = self.draw_profile_row(
                painter,
                env,
                &area,
                y,
                setup::BACKUP_NAME,
                &line,
                current,
                true,
                picked,
                ProfilesTarget::Backup,
            );
        }
        let reveal = self.nav_reveal(Pane::List);
        self.end_body(painter, Pane::List, area, y, reveal);
    }

    /// Window 2's save row: the name field and Save (or "Click to replace" while armed over a
    /// taken name).
    fn draw_save_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        area: &Area,
        y: f32,
    ) -> f32 {
        let s = area.s;
        let h = FIELD_H * s;
        let taken = self
            .profiles
            .entries
            .iter()
            .find(|entry| {
                let name = setup::clean_name(&self.profiles.name);
                !name.is_empty() && entry.name.eq_ignore_ascii_case(&name)
            })
            .map(|entry| entry.file.clone());
        let armed = taken.as_ref().is_some_and(|file| {
            self.profiles
                .is_armed(&Armed::Replace(file.clone()), env.now)
        });
        let label = if armed {
            "Click to replace"
        } else if taken.is_some() {
            "Replace\u{2026}"
        } else {
            "Save"
        };
        let style = TextStyle::new(11.5 * s).weight(Weight::Semibold);
        let button_w = env.text.measure(label, style, env.mc_font) + 26.0 * s;
        let button = Rect::new(area.right() - button_w, y, button_w, h);
        let field = Rect::new(area.x, y, button.x - area.x - 8.0 * s, h);
        let text = self.profiles.name.clone();
        self.draw_field(
            painter,
            env,
            area,
            field,
            &text,
            "Name this setup",
            Field::SaveName,
        );
        self.draw_view_button(
            painter,
            env,
            area,
            button,
            label,
            true,
            armed,
            ProfilesTarget::Save,
        );
        if let Some(hit) = area.hit(painter, button) {
            if hit.contains(self.cursor.0, self.cursor.1) {
                let tip = if taken.is_some() {
                    "A profile has this name: a second click saves your setup over it."
                } else {
                    "Save your whole setup under this name (an empty name gets one)."
                };
                self.set_tooltip(tip, self.cursor);
            }
        }
        let list = Pane::List.index();
        self.nav[list].push(NavRow {
            enter: Target::Profiles(ProfilesTarget::Field(FieldId::SaveName)),
            space: None,
            span: area.span(y, y + h),
        });
        y + h + 4.0 * s
    }

    /// One profile row: its name, a line under it, "Current" at the right when it holds the
    /// current setup.
    #[allow(clippy::too_many_arguments)]
    fn draw_profile_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        area: &Area,
        y: f32,
        name: &str,
        line: &str,
        current: bool,
        ok: bool,
        picked: bool,
        target: ProfilesTarget,
    ) -> f32 {
        let s = area.s;
        let row = Rect::new(area.x, y, area.w, ROW_H * s);
        let hit = area.hit(painter, row);
        let hover =
            self.drag.is_none() && hit.is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        let list = Pane::List.index();
        let index = self.nav[list].len();
        let focused = self.keyboard && self.pane == Pane::List && self.cursors[list] == Some(index);
        let radius = 13.0 * s;
        if picked {
            painter.round_rect_shadow(row, radius, [10.0 * s; 4], white(0.06));
            painter.round_rect(
                row,
                radius,
                Fill::css_angle(row, 90.0, white(0.24), black(0.10)),
            );
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.50));
        } else if hover {
            painter.round_rect(row, radius, Fill::Solid(white(0.09)));
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.13));
        }
        if focused {
            draw_focus_ring(painter, row, radius, s);
        }
        let mut right = row.right() - 12.0 * s;
        if current {
            let style = TextStyle::new(10.5 * s).weight(Weight::Semibold);
            let tw = env.text.measure("Current", style, env.mc_font);
            let (_, lh) = env.text.metrics(style);
            let pill = Rect::new(
                right - tw - 14.0 * s,
                row.y + (row.h - lh - 6.0 * s) * 0.5,
                tw + 14.0 * s,
                lh + 6.0 * s,
            );
            painter.round_rect(pill, pill.h * 0.5, Fill::Solid(css(140, 230, 160, 0.16)));
            painter.round_rect_stroke(pill, pill.h * 0.5, 1.0, 0.5, css(140, 230, 160, 0.6));
            shadowed(
                painter,
                env,
                "Current",
                pill.x + 7.0 * s,
                pill.y + 3.0 * s,
                style,
                GOOD,
                s,
            );
            right = pill.x - 8.0 * s;
        }
        let name_style = TextStyle::new(13.0 * s).weight(Weight::Semibold);
        let line_style = TextStyle::new(11.0 * s);
        let saved_clip = painter.clip();
        let clip = Rect::new(row.x + 10.0 * s, row.y, right - row.x - 10.0 * s, row.h);
        painter.set_clip(Some(saved_clip.map_or(clip, |c| c.intersect(&clip))));
        shadowed(
            painter,
            env,
            name,
            row.x + 14.0 * s,
            row.y + 6.0 * s,
            name_style,
            white(if ok { 0.97 } else { 0.6 }),
            s,
        );
        shadowed(
            painter,
            env,
            line,
            row.x + 14.0 * s,
            row.y + 25.0 * s,
            line_style,
            if ok { white(0.58) } else { BAD },
            s,
        );
        painter.set_clip(saved_clip);
        if let Some(r) = hit {
            self.hits.push((r, Target::Profiles(target)));
        }
        self.nav[list].push(NavRow {
            enter: Target::Profiles(target),
            space: None,
            span: area.span(row.y, row.bottom()),
        });
        row.bottom() + 3.0 * s
    }

    /// A text field of the view: its text with a caret while focused, else the placeholder.
    #[allow(clippy::too_many_arguments)]
    fn draw_field(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        area: &Area,
        field: Rect,
        text: &str,
        placeholder: &str,
        which: Field,
    ) {
        let s = area.s;
        let focused = self.profiles.focus == Some(which);
        let hover = area
            .hit(painter, field)
            .is_some_and(|r| r.contains(self.cursor.0, self.cursor.1));
        let radius = 10.0 * s;
        painter.round_rect(
            field,
            radius,
            Fill::Solid(black(if focused || hover { 0.30 } else { 0.22 })),
        );
        painter.round_rect_stroke(
            field,
            radius,
            1.0,
            0.5,
            white(if focused {
                0.70
            } else if hover {
                0.40
            } else {
                0.24
            }),
        );
        let style = TextStyle::new(12.0 * s);
        let (_, lh) = env.text.metrics(style);
        let room = field.w - 20.0 * s;
        let tw = env.text.measure(text, style, env.mc_font);
        let tx = field.x + 10.0 * s - (tw - room).max(0.0);
        let ty = field.y + (field.h - lh) * 0.5;
        let saved = painter.clip();
        let clip = field.inset(3.0 * s);
        painter.set_clip(Some(saved.map_or(clip, |c| c.intersect(&clip))));
        if text.is_empty() && !focused {
            shadowed(
                painter,
                env,
                placeholder,
                field.x + 10.0 * s,
                ty,
                style,
                white(0.42),
                s,
            );
        } else {
            let advance = shadowed(painter, env, text, tx, ty, style, white(0.95), s);
            let since = self.profiles.since.unwrap_or(env.now);
            let blink = (ms_since(since, env.now) as u64 / 500).is_multiple_of(2);
            if focused && blink {
                painter.rect(
                    Rect::new(
                        tx + advance + 1.0,
                        ty + 1.0 * s,
                        1.0 * s.max(1.0),
                        lh - 2.0 * s,
                    ),
                    white(0.9),
                );
            }
        }
        painter.set_clip(saved);
        let id = match which {
            Field::SaveName => FieldId::SaveName,
            Field::Rename => FieldId::Rename,
        };
        if let Some(hit) = area.hit(painter, field) {
            self.hits
                .push((hit, Target::Profiles(ProfilesTarget::Field(id))));
        }
    }

    /// A button of the view (greyed out when not `enabled`, red while `armed`).
    #[allow(clippy::too_many_arguments)]
    fn draw_view_button(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        area: &Area,
        r: Rect,
        label: &str,
        enabled: bool,
        armed: bool,
        target: ProfilesTarget,
    ) {
        let s = area.s;
        let hit = area.hit(painter, r);
        let hover = enabled && hit.is_some_and(|h| h.contains(self.cursor.0, self.cursor.1));
        let fill = if armed {
            css(255, 110, 90, if hover { 0.42 } else { 0.30 })
        } else if !enabled {
            white(0.04)
        } else {
            white(if hover { 0.18 } else { 0.10 })
        };
        let radius = 9.0 * s;
        painter.round_rect(r, radius, Fill::Solid(fill));
        let border = if armed {
            css(255, 150, 130, 0.8)
        } else {
            white(if enabled { 0.32 } else { 0.12 })
        };
        painter.round_rect_stroke(r, radius, 1.0, 0.5, border);
        let style = TextStyle::new(11.5 * s).weight(Weight::Semibold);
        let tw = env.text.measure(label, style, env.mc_font);
        let (_, lh) = env.text.metrics(style);
        shadowed(
            painter,
            env,
            label,
            r.x + (r.w - tw) * 0.5,
            r.y + (r.h - lh) * 0.5,
            style,
            white(if enabled { 0.96 } else { 0.40 }),
            s,
        );
        if enabled {
            if let Some(h) = hit {
                self.hits.push((h, Target::Profiles(target)));
            }
        }
    }

    /// A row of buttons from `x`: each `(label, enabled, armed, target)`. Returns the y below.
    fn draw_button_row(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        area: &Area,
        y: f32,
        buttons: &[(&str, bool, bool, ProfilesTarget)],
    ) -> f32 {
        let s = area.s;
        let style = TextStyle::new(11.5 * s).weight(Weight::Semibold);
        let mut x = area.x;
        let mut y = y;
        let details = Pane::Details.index();
        for &(label, enabled, armed, target) in buttons {
            let w = env.text.measure(label, style, env.mc_font) + 26.0 * s;
            if x + w > area.right() && x > area.x {
                x = area.x;
                y += (BUTTON_H + 8.0) * s;
            }
            let r = Rect::new(x, y, w, BUTTON_H * s);
            let index = self.nav[details].len();
            if enabled
                && self.keyboard
                && self.pane == Pane::Details
                && self.cursors[details] == Some(index)
            {
                draw_focus_ring(painter, r, 9.0 * s, s);
            }
            self.draw_view_button(painter, env, area, r, label, enabled, armed, target);
            if enabled {
                self.nav[details].push(NavRow {
                    enter: Target::Profiles(target),
                    space: None,
                    span: area.span(r.y, r.bottom()),
                });
            }
            x += w + 8.0 * s;
        }
        y + BUTTON_H * s
    }

    /// Window 3 while the Profiles view is open: the picked profile. Always draws the window.
    pub(super) fn draw_profiles_details(
        &mut self,
        painter: &mut Painter<'_>,
        env: &mut MenuEnv<'_>,
        rect: Rect,
        s: f32,
    ) -> bool {
        self.sync_profiles(env);
        let settings = env.settings;
        let now = env.now;
        let pick = self.profiles.pick.clone();
        let (title, subtitle, saved_entry) = match &pick {
            Pick::Fresh => ("Fresh install".to_owned(), "Built in".to_owned(), None),
            Pick::Backup => (
                setup::BACKUP_NAME.to_owned(),
                "Kept when you load a profile".to_owned(),
                None,
            ),
            Pick::Saved(file) => match self.profiles.entry(file) {
                Some(entry) => {
                    let when = ago(entry.saved_unix);
                    let subtitle = if when.is_empty() {
                        "Saved profile".to_owned()
                    } else {
                        format!("Saved {when}")
                    };
                    (entry.name.clone(), subtitle, Some(entry.clone()))
                }
                None => ("Profile".to_owned(), "Saving\u{2026}".to_owned(), None),
            },
        };
        draw_title(painter, env, rect, &title, &subtitle, 0.0, s);
        let area = self.begin_body(painter, rect, Pane::Details, s);
        let mut y = area.y;
        // The last notice, for a few seconds.
        let notice = self
            .profiles
            .notice
            .as_ref()
            .filter(|(_, at)| ms_since(*at, now) < NOTICE_MS)
            .map(|(notice, _)| match notice {
                Notice::Saved(name) => (format!("Saved \u{201c}{name}\u{201d}."), GOOD),
                Notice::Renamed(name) => (format!("Renamed to \u{201c}{name}\u{201d}."), GOOD),
                Notice::Deleted(name) => (format!("Deleted \u{201c}{name}\u{201d}."), GOOD),
                Notice::Failed(what) => (what.clone(), BAD),
            });
        let loaded = self
            .profiles
            .loaded
            .as_ref()
            .filter(|(_, at)| ms_since(*at, now) < NOTICE_MS)
            .map(|(text, _)| (text.clone(), GOOD));
        if let Some((text, color)) = loaded.or(notice) {
            y = draw_wrapped(
                painter,
                env,
                &text,
                area.x,
                y,
                area.w,
                TextStyle::new(12.0 * s).weight(Weight::Semibold),
                color,
                s,
            ) + 8.0 * s;
        }
        let setup = self.profiles.setup_of(&pick, settings);
        let is_current = match &pick {
            Pick::Fresh => self.profiles.fresh_current,
            Pick::Backup => self.profiles.backup_current,
            Pick::Saved(file) => self
                .profiles
                .entries
                .iter()
                .position(|entry| &entry.file == file)
                .and_then(|i| self.profiles.current.get(i).copied())
                .unwrap_or(false),
        };
        let description = match &pick {
            Pick::Fresh => {
                "Every mod as a new install has it: default switches and settings, HUD \
                 elements in their default places, nothing pinned."
            }
            Pick::Backup => {
                "The setup you had before you last loaded a profile. Load it to undo that load."
            }
            Pick::Saved(_) => "",
        };
        if !description.is_empty() {
            y = draw_wrapped(
                painter,
                env,
                description,
                area.x,
                y,
                area.w,
                TextStyle::new(12.0 * s),
                white(0.80),
                s,
            ) + 10.0 * s;
        }
        // The buttons.
        let load_armed = self.profiles.is_armed(&Armed::Load(pick.clone()), now);
        let can_load = setup.is_some() && !is_current;
        let load_label = if load_armed {
            "Click again to load"
        } else {
            "Load"
        };
        let mut buttons: Vec<(&str, bool, bool, ProfilesTarget)> =
            vec![(load_label, can_load, load_armed, ProfilesTarget::Load)];
        let renaming = self.profiles.renaming && saved_entry.is_some();
        if let (Pick::Saved(file), Some(_)) = (&pick, &saved_entry) {
            let update_armed = self.profiles.is_armed(&Armed::Replace(file.clone()), now);
            let delete_armed = self.profiles.is_armed(&Armed::Delete(file.clone()), now);
            buttons.push((
                if update_armed {
                    "Click to save over it"
                } else {
                    "Save current here"
                },
                !is_current,
                update_armed,
                ProfilesTarget::Update,
            ));
            buttons.push((
                if renaming { "Done" } else { "Rename" },
                true,
                false,
                ProfilesTarget::Rename,
            ));
            buttons.push((
                if delete_armed {
                    "Click to delete"
                } else {
                    "Delete"
                },
                true,
                delete_armed,
                ProfilesTarget::Delete,
            ));
        }
        y = self.draw_button_row(painter, env, &area, y, &buttons) + 8.0 * s;
        if renaming {
            let field = Rect::new(area.x, y, area.w, FIELD_H * s);
            let text = self.profiles.rename.clone();
            self.draw_field(painter, env, &area, field, &text, "New name", Field::Rename);
            y = field.bottom() + 4.0 * s;
            let clash = saved_entry.as_ref().is_some_and(|entry| {
                let wanted = setup::clean_name(&self.profiles.rename);
                self.profiles.entries.iter().any(|other| {
                    other.file != entry.file && other.name.eq_ignore_ascii_case(&wanted)
                })
            });
            let hint = if clash {
                "Another profile has this name."
            } else {
                "Enter renames, Esc cancels."
            };
            y = draw_wrapped(
                painter,
                env,
                hint,
                area.x + 2.0 * s,
                y,
                area.w,
                TextStyle::new(11.0 * s),
                if clash { BAD } else { white(0.55) },
                s,
            ) + 6.0 * s;
        }
        let note = if is_current {
            "This is your current setup."
        } else if setup.is_none() {
            "This file is not a settings file, so it can't be loaded. You can delete it."
        } else if self.profiles.library.is_some() {
            "Loading replaces your whole current setup. It is kept as \u{201c}Before last \
             load\u{201d}, so you can load it back."
        } else {
            "Loading replaces your whole current setup."
        };
        y = draw_wrapped(
            painter,
            env,
            note,
            area.x,
            y,
            area.w,
            TextStyle::new(11.0 * s),
            if load_armed {
                css(255, 222, 168, 0.95)
            } else {
                white(0.55)
            },
            s,
        ) + 12.0 * s;
        // What loading it would change.
        if let Some(target) = &setup {
            if !is_current {
                y = section(painter, env, &area, y, "What loading changes");
                let generation = settings.generation();
                let seen = self.profiles.seen;
                let fresh = match &self.profiles.diff {
                    Some((p, g, l, diff)) if *p == pick && *g == generation && *l == seen => {
                        diff.clone()
                    }
                    _ => {
                        let diff = settings.setup_diff(target);
                        self.profiles.diff = Some((pick.clone(), generation, seen, diff.clone()));
                        diff
                    }
                };
                y = draw_diff(painter, env, &area, y, settings, &fresh);
            }
            // What it holds, as My setup counts it: the mods changed from a fresh install.
            let changed: Option<Vec<usize>> = match &pick {
                Pick::Fresh => None,
                Pick::Backup => Some(self.profiles.backup_changed.clone()),
                Pick::Saved(file) => self
                    .profiles
                    .entries
                    .iter()
                    .position(|entry| &entry.file == file)
                    .and_then(|i| self.profiles.changed.get(i).cloned()),
            };
            if let Some(changed) = changed {
                y = section(
                    painter,
                    env,
                    &area,
                    y + 6.0 * s,
                    "Changed from a fresh install",
                );
                let text = if changed.is_empty() {
                    "Nothing: it is a fresh install.".to_owned()
                } else {
                    names(target, &changed, usize::MAX)
                };
                y = draw_wrapped(
                    painter,
                    env,
                    &text,
                    area.x + 4.0 * s,
                    y,
                    area.w - 8.0 * s,
                    TextStyle::new(11.5 * s),
                    white(0.72),
                    s,
                );
            }
        } else if let Some(entry) = &saved_entry {
            for warning in entry.warnings.iter().take(4) {
                y = draw_wrapped(
                    painter,
                    env,
                    warning,
                    area.x,
                    y,
                    area.w,
                    TextStyle::new(11.0 * s),
                    white(0.5),
                    s,
                );
            }
        }
        let reveal = self.nav_reveal(Pane::Details);
        self.end_body(painter, Pane::Details, area, y, reveal);
        true
    }

    /// A key while the view has a text field focused: typing, Backspace, Enter (save or
    /// rename), Esc and Tab (leave the field). `None` when no field has focus.
    pub(super) fn profiles_key(
        &mut self,
        lwjgl: i32,
        typed: Option<char>,
        input: &mut MenuInput<'_>,
    ) -> Option<super::KeyOutcome> {
        let field = self.profiles.focus?;
        match lwjgl {
            super::input::KEY_ESCAPE => {
                self.profiles.focus = None;
                if field == Field::Rename {
                    self.profiles.renaming = false;
                }
            }
            super::input::KEY_TAB => {
                self.profiles.focus = None;
            }
            super::input::KEY_RETURN => {
                self.profiles.focus = None;
                match field {
                    Field::SaveName => self.profiles_save(input),
                    Field::Rename => self.profiles_commit_rename(),
                }
            }
            super::input::KEY_BACK => {
                let text = self.profile_field_mut(field);
                text.pop();
            }
            _ => {
                if let Some(c) = typed.filter(|c| !c.is_control()) {
                    let text = self.profile_field_mut(field);
                    if text.chars().count() < setup::NAME_MAX && !(c == ' ' && text.is_empty()) {
                        text.push(c);
                    }
                }
            }
        }
        Some(super::KeyOutcome::Handled)
    }

    /// Ctrl+V while a field of the view has focus: the text goes there. Returns whether a field
    /// took it.
    pub(super) fn profiles_paste(&mut self, text: &str) -> bool {
        let Some(field) = self.profiles.focus else {
            return false;
        };
        let field = self.profile_field_mut(field);
        for c in text.chars() {
            if field.chars().count() >= setup::NAME_MAX {
                break;
            }
            if !(c == ' ' && field.is_empty()) {
                field.push(c);
            }
        }
        true
    }

    /// Whether a text field of the view has the keyboard (typing then goes there, not to the
    /// search).
    #[must_use]
    pub(super) fn profiles_typing(&self) -> bool {
        self.profiles.focus.is_some()
    }

    /// A click elsewhere: the view's fields lose the keyboard (a click on one gives it back).
    pub(super) fn profiles_blur(&mut self) {
        self.profiles.focus = None;
    }

    fn profile_field_mut(&mut self, field: Field) -> &mut String {
        match field {
            Field::SaveName => &mut self.profiles.name,
            Field::Rename => &mut self.profiles.rename,
        }
    }

    /// Window 2's Save: a new name saves at once; a taken name needs a second click.
    fn profiles_save(&mut self, input: &mut MenuInput<'_>) {
        let Some(library) = &self.profiles.library else {
            return;
        };
        let name = setup::clean_name(&self.profiles.name);
        let taken = (!name.is_empty())
            .then(|| library.find(&name))
            .flatten()
            .map(|entry| entry.file);
        match taken {
            Some(file) => {
                let armed = Armed::Replace(file.clone());
                if self.profiles.is_armed(&armed, input.now) {
                    library.save(&name, input.settings, Some(&file));
                    self.profiles.armed = None;
                    self.profiles.name.clear();
                    self.profiles.pick = Pick::Saved(file);
                } else {
                    self.profiles.armed = Some((armed, input.now));
                }
            }
            None => {
                library.save(&name, input.settings, None);
                self.profiles.armed = None;
                self.profiles.name.clear();
                if !name.is_empty() {
                    self.profiles.pending_pick = Some(name);
                }
            }
        }
    }

    /// Commit window 3's rename (an empty or taken name keeps the field open).
    fn profiles_commit_rename(&mut self) {
        let Pick::Saved(file) = &self.profiles.pick else {
            self.profiles.renaming = false;
            return;
        };
        let wanted = setup::clean_name(&self.profiles.rename);
        let taken = self
            .profiles
            .entries
            .iter()
            .any(|other| &other.file != file && other.name.eq_ignore_ascii_case(&wanted));
        if wanted.is_empty() || taken {
            self.profiles.focus = Some(Field::Rename);
            return;
        }
        self.profiles.renaming = false;
        let same = self
            .profiles
            .entry(file)
            .is_some_and(|entry| entry.name == wanted);
        if let (Some(library), false) = (&self.profiles.library, same) {
            library.rename(file, &wanted);
            self.profiles.pending_pick = Some(wanted);
        }
    }

    /// A click on one of the view's targets (`button`: 0 left, 1 right).
    pub(super) fn activate_profiles(
        &mut self,
        target: ProfilesTarget,
        button: u8,
        input: &mut MenuInput<'_>,
    ) {
        if button != 0 {
            return;
        }
        let now = input.now;
        match target {
            ProfilesTarget::Fresh => self.pick_profile(Pick::Fresh),
            ProfilesTarget::Backup => self.pick_profile(Pick::Backup),
            ProfilesTarget::Row(i) => {
                if let Some(entry) = self.profiles.entries.get(usize::from(i)) {
                    let file = entry.file.clone();
                    self.pick_profile(Pick::Saved(file));
                }
            }
            ProfilesTarget::Field(FieldId::SaveName) => {
                self.profiles.focus = Some(Field::SaveName);
                self.profiles.since = Some(now);
            }
            ProfilesTarget::Field(FieldId::Rename) => {
                self.profiles.focus = Some(Field::Rename);
                self.profiles.since = Some(now);
            }
            ProfilesTarget::Save => self.profiles_save(input),
            ProfilesTarget::Load => {
                let pick = self.profiles.pick.clone();
                let armed = Armed::Load(pick.clone());
                if !self.profiles.is_armed(&armed, now) {
                    self.profiles.armed = Some((armed, now));
                    return;
                }
                self.profiles.armed = None;
                let Some(target) = self.profiles.setup_of(&pick, input.settings) else {
                    return;
                };
                if input.settings.setup_eq(&target) {
                    return;
                }
                // Keep the setup that is there first (memory now, disk soon), then swap the
                // whole setup in one step.
                if let Some(library) = &self.profiles.library {
                    library.keep_backup(input.settings);
                }
                input.settings.apply_setup(&target);
                let name = match &pick {
                    Pick::Fresh => "Fresh install".to_owned(),
                    Pick::Backup => setup::BACKUP_NAME.to_owned(),
                    Pick::Saved(file) => self
                        .profiles
                        .entry(file)
                        .map(|entry| entry.name.clone())
                        .unwrap_or_default(),
                };
                let text = if self.profiles.library.is_some() && pick != Pick::Backup {
                    format!(
                        "Loaded \u{201c}{name}\u{201d}. Your previous setup is in \u{201c}{}\
                         \u{201d}.",
                        setup::BACKUP_NAME
                    )
                } else {
                    format!("Loaded \u{201c}{name}\u{201d}.")
                };
                self.profiles.loaded = Some((text, now));
                self.profiles.notice = None;
            }
            ProfilesTarget::Update => {
                let Pick::Saved(file) = self.profiles.pick.clone() else {
                    return;
                };
                let armed = Armed::Replace(file.clone());
                if !self.profiles.is_armed(&armed, now) {
                    self.profiles.armed = Some((armed, now));
                    return;
                }
                self.profiles.armed = None;
                let name = self.profiles.entry(&file).map(|entry| entry.name.clone());
                if let (Some(library), Some(name)) = (&self.profiles.library, name) {
                    library.save(&name, input.settings, Some(&file));
                }
            }
            ProfilesTarget::Rename => {
                if self.profiles.renaming {
                    self.profiles.focus = None;
                    self.profiles_commit_rename();
                } else if let Pick::Saved(file) = &self.profiles.pick {
                    if let Some(entry) = self.profiles.entry(file) {
                        self.profiles.rename = entry.name.clone();
                        self.profiles.renaming = true;
                        self.profiles.focus = Some(Field::Rename);
                        self.profiles.since = Some(now);
                    }
                }
            }
            ProfilesTarget::Delete => {
                let Pick::Saved(file) = self.profiles.pick.clone() else {
                    return;
                };
                let armed = Armed::Delete(file.clone());
                if !self.profiles.is_armed(&armed, now) {
                    self.profiles.armed = Some((armed, now));
                    return;
                }
                self.profiles.armed = None;
                if let Some(library) = &self.profiles.library {
                    library.delete(&file);
                }
                self.profiles.pick = Pick::Fresh;
            }
        }
    }

    /// Where saved profile `index` (of the list drawn last frame) was drawn in window 2.
    #[must_use]
    pub fn profile_row_rect(&self, index: usize) -> Option<Rect> {
        let row = u16::try_from(index).ok()?;
        self.hit_rect(|target| *target == Target::Profiles(ProfilesTarget::Row(row)))
    }

    /// Where window 3's Load button was drawn last frame (`None` while it is greyed out).
    #[must_use]
    pub fn profile_load_rect(&self) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::Profiles(ProfilesTarget::Load))
    }

    /// Show `pick` in window 3 (its buttons start unarmed, a rename in progress ends).
    fn pick_profile(&mut self, pick: Pick) {
        if self.profiles.pick != pick {
            self.profiles.pick = pick;
            self.profiles.armed = None;
            self.profiles.renaming = false;
            if self.profiles.focus == Some(Field::Rename) {
                self.profiles.focus = None;
            }
            let details = Pane::Details.index();
            self.scroll[details] = 0.0;
            self.scroll_target[details] = 0.0;
            self.cursors[details] = None;
        }
    }
}

/// A section title like window 2's.
fn section(
    painter: &mut Painter<'_>,
    env: &mut MenuEnv<'_>,
    area: &Area,
    y: f32,
    title: &str,
) -> f32 {
    let s = area.s;
    shadowed(
        painter,
        env,
        title,
        area.x + 4.0 * s,
        y + 5.0 * s,
        TextStyle::new(10.0 * s).weight(Weight::Bold).caps(0.16),
        white(0.72),
        s,
    );
    y + 24.0 * s
}

/// "Turns on: FPS, CPS" lines for a diff.
fn draw_diff(
    painter: &mut Painter<'_>,
    env: &mut MenuEnv<'_>,
    area: &Area,
    y: f32,
    settings: &ModSettings,
    diff: &SetupDiff,
) -> f32 {
    let s = area.s;
    let mut y = y;
    let mut lines: Vec<(&str, String)> = Vec::new();
    if !diff.turns_on.is_empty() {
        lines.push(("Turns on", names(settings, &diff.turns_on, 8)));
    }
    if !diff.turns_off.is_empty() {
        lines.push(("Turns off", names(settings, &diff.turns_off, 8)));
    }
    if !diff.settings.is_empty() {
        lines.push(("Settings", names(settings, &diff.settings, 8)));
    }
    if !diff.moved.is_empty() {
        lines.push(("Moves", names(settings, &diff.moved, 8)));
    }
    if diff.hud_style {
        lines.push(("HUD style", "Different".to_owned()));
    }
    if diff.pins {
        lines.push(("Pins", "Different".to_owned()));
    }
    let label_style = TextStyle::new(11.5 * s).weight(Weight::Semibold);
    let text_style = TextStyle::new(11.5 * s);
    let label_w = 78.0 * s;
    for (label, text) in lines {
        shadowed(
            painter,
            env,
            label,
            area.x + 4.0 * s,
            y,
            label_style,
            white(0.86),
            s,
        );
        let bottom = draw_wrapped(
            painter,
            env,
            &text,
            area.x + 4.0 * s + label_w,
            y,
            area.w - label_w - 8.0 * s,
            text_style,
            white(0.70),
            s,
        );
        y = bottom.max(y + 16.0 * s) + 4.0 * s;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_and_ages_read_well() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(951_782_400), "2000-02-29");
        assert_eq!(date(1_762_905_600), "2025-11-12");
        let now = setup::now_unix();
        assert_eq!(ago(Some(now)), "just now");
        assert_eq!(ago(Some(now - 300)), "5 min ago");
        assert_eq!(ago(Some(now - 3 * 3600)), "3 h ago");
        assert_eq!(ago(Some(now - 90_000)), "yesterday");
        assert_eq!(ago(Some(now - 4 * 86_400)), "4 days ago");
        assert_eq!(ago(None), "");
    }
}
