//! The mod menu: three Aero glass windows read left to right: **Where** (search, My setup, Keybinds, Profiles, the categories, All
//! mods), **What** (the mods in what you picked, with a summary line and a switch each) and
//! **How** (the selected mod's settings at full width), plus a taskbar ("Edit HUD") and the HUD
//! editor.
//!
//! The paragraphs below describe the seven-window menu this replaced, where they still apply:
//! the search moved from the taskbar to window 1, the menu key to the Keybinds view, and a mod's
//! options from its row to window 3 ([`View`] and [`MenuPlace`] are the new state).
//!
//! Former summary: seven vertical Aero glass windows, one per [`Category`], with a row per mod,
//! inline options, a taskbar row (search, "Edit HUD", the menu key) and the HUD editor.
//!
//! Plain state plus arithmetic, like every screen in this crate: [`ModMenuScreen::build`]
//! draws one frame into a [`DrawList`] in **physical pixels** and returns the [`GlassFrame`]
//! for [`glass::ModMenuPhase`]; the input methods mutate [`ModSettings`] directly and queue
//! [`ModEvent`]s. Hit regions are recorded while drawing and used by the next input event
//! (immediate-mode), so layout lives in exactly one place.
//!
//! Design source: `Aero/aero-webgl-desktop/index.html` (window, title bar, section labels, nav
//! rows, badges, the white outline pill, the taskbar pill). Sizes below are its CSS pixels
//! ("dip"), scaled by [`px_per_dip`] so seven [`WINDOW_W`]-dip windows, their gaps and a side
//! margin fit the screen width (at 16:9 the width decides, so 1280x720, 1920x1080 and 2560x1440
//! show the same layout at 0.66, 0.98 and 1.31 px per dip).
//!
//! # Features
//!
//! * **Menu key** ([`ModSettings::menu_key`], Right Shift by default): the client opens the menu
//!   with it in game; here it closes the menu (and Esc does). The taskbar's key pill rebinds it.
//! * **Search**: typing goes to the taskbar's search field unless a text field or a key capture
//!   has focus. Terms (split on spaces, all must match, ASCII case-insensitive) are looked up in
//!   each mod's name, description, option labels, choice labels and dynamic-row labels. Rows that
//!   do not match are hidden; a window with none dims and says so; a mod whose options match
//!   shows just those options inline. Esc clears the search first, then closes.
//! * **Reset to defaults** in every expanded mod with options (armed: click twice), and named
//!   **profiles** for [`PROFILE_MODS`] (colour grading).
//! * **Fine slider steps**: Left/Right (Shift: x10) move the hovered or last-dragged slider by
//!   one step, so every value is reachable whatever the pointer resolution.
//!
//! # Modules (the 4,030-line file split)
//!
//! Each module adds its own `impl ModMenuScreen` block; the screen's fields are `pub(super)` in
//! `state`, so every module here reads them.
//!
//! | Module | What it holds | Owner |
//! |---|---|---|
//! | `state` | the screen's fields, [`View`], [`MenuPlace`], hit targets, the accessors | MA |
//! | `layout` | the scale, the three window frames, `build` (animation, glass), the taskbar | MA |
//! | `sidebar` | window 1: the search field and the entries | MA |
//! | `list` | window 2: sections of mod rows, search results | MA |
//! | `details` | window 3: the header and a mod's settings, colour editor, dynamic rows | MA |
//! | `input` | mouse, wheel, keys, clipboard, what each click does | MA |
//! | `widgets` | switch, badge, slider, pills, icons, text shadow, colour maths | MA |
//! | `search` | the menu side of the search engine (`mc_mods::menu_search`) | MB |
//! | `keybinds` | key names and the Keybinds view | MC |
//! | `profiles` | the Profiles view (whole-setup profiles) | MC |
//! | `editor` | the HUD editor, "Move on screen" | MC |
//! | `anim`, `glass`, `paint`, `text` | animation clock, glass pass, painter, Segoe UI text | |

pub mod anim;
mod details;
mod editor;
mod extras;
pub mod glass;
mod input;
mod keybinds;
mod layout;
mod list;
pub mod paint;
mod profiles;
mod search;
mod sidebar;
mod state;
pub mod text;
mod widgets;
// Declare new modmenu modules here.
#[cfg(test)]
mod mc_tests;

use std::time::Instant;

use mc_mods::{DynamicRows, ModEvent, ModSettings};

pub use self::details::PROFILE_MODS;
pub use self::editor::EDITOR_DIM_ARGB;
pub use self::keybinds::key_name;
pub use self::layout::{design_width, px_per_dip, WINDOW_WIDTHS};
pub use self::state::{MenuPlace, ModMenuScreen, View};
pub use self::widgets::{argb_to_hsva, hsva_to_argb};

#[cfg(doc)]
use self::glass::GlassFrame;
use self::text::MenuText;
use crate::font::FontRenderer;
use crate::modhud::ModHud;
#[cfg(doc)]
use crate::widget::DrawList;
#[cfg(doc)]
use mc_mods::Category;

/// What the client hands [`ModMenuScreen::build`] each frame.
pub struct MenuEnv<'a> {
    /// Segoe UI (or Minecraft-font) text.
    pub text: &'a mut MenuText,
    /// The Minecraft font (fallback text and measuring).
    pub mc_font: &'a mut FontRenderer,
    /// The settings store.
    pub settings: &'a ModSettings,
    /// Dynamic rows per catalog index.
    pub rows: &'a [DynamicRows],
    /// The HUD pipeline (the editor reads its slot rectangles).
    pub hud: &'a ModHud,
    /// Window size in physical pixels.
    pub screen: (f32, f32),
    /// Physical pixels per GUI pixel.
    pub gui_scale: f32,
    /// Real time now (the animation clock).
    pub now: Instant,
    /// The glass `uTime` in seconds.
    pub time_s: f32,
    /// What the client knows about the game around the menu.
    pub game: MenuGame<'a>,
}

/// What the client knows about the game around the menu. `MenuGame::default()` is "in a world,
/// no extra facts", which is what tests and harnesses want.
#[derive(Debug, Clone, Copy, Default)]
pub struct MenuGame<'a> {
    /// No world is loaded (the title screen, the replay browser): Edit HUD and "Move on screen"
    /// are greyed out, everything else works.
    pub no_world: bool,
    /// The vanilla controls (`options.txt`), for the Keybinds view's clash check.
    pub vanilla_keys: Option<&'a mc_render::input::KeyBindings>,
    /// The client's `options.txt` settings, for window 3's live warnings ("Does nothing while
    /// MSAA is on"). `None`: no warnings.
    pub options: Option<&'a mc_render::GameSettings>,
    /// The game folder: the Profiles view keeps its profiles in `<game_dir>/mc-rs-mod-profiles`.
    /// `None`: profiles cannot be saved (only "Fresh install" loads).
    pub game_dir: Option<&'a std::path::Path>,
}

/// What the input methods may change.
pub struct MenuInput<'a> {
    /// The settings store.
    pub settings: &'a mut ModSettings,
    /// The HUD pipeline (drag previews).
    pub hud: &'a mut ModHud,
    /// Events for the client.
    pub events: &'a mut Vec<ModEvent>,
    /// The scaled GUI size (HUD placements live in it).
    pub gui_screen: (f32, f32),
    /// Physical pixels per GUI pixel.
    pub gui_scale: f32,
    /// Real time now.
    pub now: Instant,
    /// Alt held: the HUD editor drags without snapping.
    pub alt: bool,
    /// Shift held: finer scroll scaling, ×10 nudges.
    pub shift: bool,
}

/// What a key press did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyOutcome {
    /// The menu used it.
    Handled,
    /// Close the menu (Esc or Tab at the top level).
    Close,
}

#[cfg(test)]
mod tests;
