//! The Accounts screen: the accounts this client can play with, and adding, switching to and
//! removing them. There is no vanilla equivalent (1.8.9 is started with one account already
//! signed in), so this is a design in vanilla's idiom: the server list's scrolling list and
//! 36-pixel rows (a 32x32 head where the server list has its icon), and two rows of buttons.
//!
//! * **The list:** the launcher's session first (when the game was started with one), then the
//!   accounts added in the client, in the order they were added. The account in use says so.
//! * **Use** switches to the selected account (also a double-click). Only with no world loaded:
//!   in a world the button is off and the status line says why. For an account whose saved
//!   sign-in has expired or cannot be opened on this PC the button reads **Sign in again**.
//! * **Add account** opens the Microsoft sign-in ([`LoginScreen`]: the code, the link, Copy code
//!   and Open page) inside this screen. A build without a Microsoft application ID says so on
//!   the status line instead.
//! * **Remove** asks first, then deletes the account and its saved sign-in. The launcher's
//!   session is not stored, so it cannot be removed.
//!
//! The screen only shows state and reports clicks ([`AccountsAction`] inside
//! [`Action::Accounts`]); the client owns the account store, the sign-in task and the switch.
//! The corner on the Aero title screen that opens this screen is [`corner`].

pub mod corner;

use std::time::{Duration, Instant};

use mc_assets::Language;
use mc_auth::{AuthError, DeviceCodePrompt};

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{draw_dirt_background, LABEL_COLOR, TITLE_COLOR};
use crate::screens::login::{LoginScreen, LoginState};
use crate::screens::Action;
use crate::widget::draw::{DrawList, UiTexture};
use crate::widget::keys::Clipboard;
use crate::widget::list::ScrollList;
use crate::widget::{draw_centered_string, draw_string, Button};

/// Which account a row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccountKey {
    /// The session the launcher passed on the command line.
    Launcher,
    /// An account added in the client, by its Minecraft UUID (as `u128`).
    Stored(u128),
}

/// What the screen asks the client to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccountsAction {
    /// Open the Accounts screen (the title screen's account corner, the mod menu's button).
    Open,
    /// Start the Microsoft sign-in for an account to add (the screen already shows it starting).
    Add,
    /// Stop the sign-in in progress (Cancel or Escape on the sign-in page).
    CancelAdd,
    /// Switch to this account.
    Use(AccountKey),
    /// Remove this account and its saved sign-in (the player already confirmed).
    Remove(AccountKey),
    /// Close the screen.
    Done,
}

/// The state of a row's account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowStatus {
    /// Ready to use.
    Ready,
    /// Its saved sign-in expired or cannot be opened here: it needs a new sign-in.
    NeedsSignIn,
}

/// One row of the list.
#[derive(Debug, Clone)]
pub struct AccountRow {
    /// Which account.
    pub key: AccountKey,
    /// The Minecraft name.
    pub name: String,
    /// The skin sheet the 8x8 face (and its hat layer) is taken from.
    pub head: UiTexture,
    /// The sheet's height in texels: 64, or 32 for a legacy sheet.
    pub head_sheet_height: f32,
    /// Whether it can be used as it is.
    pub status: RowStatus,
}

/// How the status line reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    /// Something happened as asked (green).
    Done,
    /// Something to know or do (yellow).
    Notice,
    /// Something failed (red).
    Problem,
}

/// What the status line says.
#[derive(Debug, Clone, PartialEq, Eq)]
enum StatusText {
    /// The client's message.
    Text(String),
    /// [`not_set_up_text`], in the language of the frame it is drawn in.
    NotSetUp,
}

#[derive(Debug)]
enum Mode {
    List,
    /// "Remove <name>?"
    Confirm {
        key: AccountKey,
        name: String,
    },
    /// The Microsoft sign-in for an account to add.
    SignIn(Box<LoginScreen>),
}

mod ids {
    pub const USE: i32 = 0;
    pub const ADD: i32 = 1;
    pub const REMOVE: i32 = 2;
    pub const DONE: i32 = 3;
    pub const CONFIRM_REMOVE: i32 = 10;
    pub const CONFIRM_CANCEL: i32 = 11;
}

/// The list's row height, as the server list's (`ServerSelectionList`: 36).
pub const SLOT_HEIGHT: i32 = 36;
/// The list's content width.
pub const LIST_WIDTH: i32 = 270;
/// Vanilla's double-click window (`GuiSlot`: 250 ms).
const DOUBLE_CLICK: Duration = Duration::from_millis(250);

/// Colours (vanilla's chat palette).
const GREEN: i32 = 0x55_FF55;
const YELLOW: i32 = 0xFF_FF55;
const RED: i32 = 0xFF_5555;
const GREY: i32 = 0x80_8080;

/// A language key with an English fallback: these keys exist in no vanilla language file.
fn text(lang: &Language, key: &str, fallback: &str) -> String {
    if lang.has_key(key) {
        lang.tr(key).to_owned()
    } else {
        fallback.to_owned()
    }
}

fn text_with(lang: &Language, key: &str, fallback: &str, arg: &str) -> String {
    let template = if lang.has_key(key) {
        lang.tr(key)
    } else {
        fallback
    };
    mc_assets::format_template(template, &[arg])
}

/// The Options screen's button that opens this screen ("Accounts...", vanilla's ellipsis for a
/// button that opens a screen).
#[must_use]
pub fn options_button_label(lang: &Language) -> String {
    text(lang, "aero.accounts.button", "Accounts...")
}

/// The status line for a build that has no Microsoft application ID.
#[must_use]
pub fn not_set_up_text(lang: &Language) -> String {
    text(
        lang,
        "aero.accounts.notSetUp",
        "Microsoft sign-in isn't set up in this build of Aero Client.",
    )
}

/// The Accounts screen. See the module documentation.
#[derive(Debug)]
pub struct AccountsScreen {
    rows: Vec<AccountRow>,
    active: Option<AccountKey>,
    selected: Option<usize>,
    list: ScrollList,
    buttons: Vec<Button>,
    mode: Mode,
    status: Option<(StatusText, Tone)>,
    world_loaded: bool,
    sign_in_available: bool,
    last_click: Option<(usize, Instant)>,
}

impl AccountsScreen {
    /// A screen listing `rows`, with `active` the account in use. The account in use starts
    /// selected.
    #[must_use]
    pub fn new(rows: Vec<AccountRow>, active: Option<AccountKey>) -> AccountsScreen {
        let mut screen = AccountsScreen {
            rows: Vec::new(),
            active: None,
            selected: None,
            list: ScrollList::new(0, 0, 32, 0, SLOT_HEIGHT),
            buttons: Vec::new(),
            mode: Mode::List,
            status: None,
            world_loaded: false,
            sign_in_available: true,
            last_click: None,
        };
        screen.list.list_width = LIST_WIDTH;
        screen.set_rows(rows, active);
        screen.selected = screen.active.and_then(|key| screen.index_of(key));
        screen
    }

    /// Replace the rows (after an add, a removal or a switch). The selection follows its account.
    pub fn set_rows(&mut self, rows: Vec<AccountRow>, active: Option<AccountKey>) {
        let selected = self.selected_key();
        self.rows = rows;
        self.active = active;
        self.selected = selected.and_then(|key| self.index_of(key));
        self.list.size = self.rows.len();
        self.refresh_buttons();
    }

    /// The rows.
    #[must_use]
    pub fn rows(&self) -> &[AccountRow] {
        &self.rows
    }

    /// The account in use.
    #[must_use]
    pub fn active(&self) -> Option<AccountKey> {
        self.active
    }

    /// Whether a world is loaded (no switching then).
    pub fn set_world_loaded(&mut self, loaded: bool) {
        self.world_loaded = loaded;
        self.refresh_buttons();
    }

    /// Whether this build can sign in with Microsoft (it has an application ID).
    pub fn set_sign_in_available(&mut self, available: bool) {
        self.sign_in_available = available;
    }

    /// Show `message` on the status line.
    pub fn set_status(&mut self, message: impl Into<String>, tone: Tone) {
        self.status = Some((StatusText::Text(message.into()), tone));
    }

    /// The client's status line message, if one shows.
    #[must_use]
    pub fn status(&self) -> Option<(&str, Tone)> {
        match &self.status {
            Some((StatusText::Text(message), tone)) => Some((message.as_str(), *tone)),
            _ => None,
        }
    }

    /// Whether the status line says that this build cannot sign in with Microsoft.
    #[must_use]
    pub fn says_not_set_up(&self) -> bool {
        matches!(self.status, Some((StatusText::NotSetUp, _)))
    }

    /// The selected row's account.
    #[must_use]
    pub fn selected_key(&self) -> Option<AccountKey> {
        self.selected
            .and_then(|index| self.rows.get(index))
            .map(|row| row.key)
    }

    /// Select `key`'s row.
    pub fn select(&mut self, key: AccountKey) {
        self.selected = self.index_of(key);
        self.refresh_buttons();
    }

    fn index_of(&self, key: AccountKey) -> Option<usize> {
        self.rows.iter().position(|row| row.key == key)
    }

    fn selected_row(&self) -> Option<&AccountRow> {
        self.selected.and_then(|index| self.rows.get(index))
    }

    /// Whether the Microsoft sign-in page is showing.
    #[must_use]
    pub fn is_signing_in(&self) -> bool {
        matches!(self.mode, Mode::SignIn(_))
    }

    /// Whether the removal question is showing.
    #[must_use]
    pub fn is_confirming(&self) -> bool {
        matches!(self.mode, Mode::Confirm { .. })
    }

    /// The sign-in page, while it shows.
    #[must_use]
    pub fn sign_in(&self) -> Option<&LoginScreen> {
        match &self.mode {
            Mode::SignIn(login) => Some(login),
            _ => None,
        }
    }

    /// Show the sign-in page, already asking Microsoft for a code (the client started the task).
    pub fn begin_sign_in(&mut self) {
        let mut login = LoginScreen::new();
        login.state = LoginState::Requesting;
        self.mode = Mode::SignIn(Box::new(login));
        self.status = None;
    }

    /// Microsoft issued the code: show it.
    pub fn sign_in_prompt(&mut self, prompt: DeviceCodePrompt) {
        if let Mode::SignIn(login) = &mut self.mode {
            login.set_prompt(prompt);
        }
    }

    /// The sign-in failed: the page shows why and what to do, and offers to try again.
    pub fn sign_in_failed(&mut self, error: &AuthError, lang: &Language) {
        if let Mode::SignIn(login) = &mut self.mode {
            login.fail(error, lang);
        }
    }

    /// The account was added (and stored): back to the list, with it selected.
    pub fn sign_in_finished(&mut self, key: AccountKey) {
        self.mode = Mode::List;
        self.select(key);
    }

    /// Escape: the sign-in page or the question go back to the list (a sign-in in progress is
    /// cancelled); on the list it closes the screen.
    pub fn back(&mut self) -> Action {
        match std::mem::replace(&mut self.mode, Mode::List) {
            Mode::SignIn(_) => Action::Accounts(AccountsAction::CancelAdd),
            Mode::Confirm { .. } => Action::None,
            Mode::List => Action::Accounts(AccountsAction::Done),
        }
    }

    /// Lay out for this frame (every frame, like the other screens).
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        match &mut self.mode {
            Mode::SignIn(login) => {
                login.init(ctx, lang);
                return;
            }
            Mode::Confirm { .. } => {
                self.buttons = vec![
                    Button::sized(
                        ids::CONFIRM_REMOVE,
                        w / 2 - 154,
                        h / 2 + 10,
                        150,
                        20,
                        text(lang, "aero.accounts.remove", "Remove"),
                    ),
                    Button::sized(
                        ids::CONFIRM_CANCEL,
                        w / 2 + 4,
                        h / 2 + 10,
                        150,
                        20,
                        lang.tr("gui.cancel").to_owned(),
                    ),
                ];
                return;
            }
            Mode::List => {}
        }
        self.list.set_dimensions(w, h, 32, h - 76);
        self.list.list_width = LIST_WIDTH;
        self.list.size = self.rows.len();
        let use_label = if self
            .selected_row()
            .is_some_and(|row| row.status == RowStatus::NeedsSignIn)
        {
            text(lang, "aero.accounts.signInAgain", "Sign in again")
        } else {
            text(lang, "aero.accounts.use", "Use")
        };
        self.buttons = vec![
            Button::sized(ids::USE, w / 2 - 154, h - 52, 150, 20, use_label),
            Button::sized(
                ids::ADD,
                w / 2 + 4,
                h - 52,
                150,
                20,
                text(lang, "aero.accounts.add", "Add account"),
            ),
            Button::sized(
                ids::REMOVE,
                w / 2 - 154,
                h - 28,
                150,
                20,
                text(lang, "aero.accounts.remove", "Remove"),
            ),
            Button::sized(
                ids::DONE,
                w / 2 + 4,
                h - 28,
                150,
                20,
                lang.tr("gui.done").to_owned(),
            ),
        ];
        self.refresh_buttons();
    }

    fn refresh_buttons(&mut self) {
        let selected = self.selected_row().map(|row| (row.key, row.status));
        let active = self.active;
        let world = self.world_loaded;
        for button in &mut self.buttons {
            button.enabled = match button.id {
                ids::USE => match selected {
                    // A dead sign-in is renewed by signing in again, world or not.
                    Some((_, RowStatus::NeedsSignIn)) => true,
                    Some((key, RowStatus::Ready)) => !world && Some(key) != active,
                    None => false,
                },
                ids::REMOVE => match selected {
                    Some((AccountKey::Stored(_), _)) => {
                        !(world && selected.map(|(key, _)| key) == active)
                    }
                    _ => false,
                },
                _ => true,
            };
        }
    }

    /// Handle a click.
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        clipboard: &mut dyn Clipboard,
    ) -> Action {
        self.mouse_clicked_at(x, y, button, clipboard, Instant::now())
    }

    /// [`AccountsScreen::mouse_clicked`] at a given time (double-clicks; tests).
    pub fn mouse_clicked_at(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        clipboard: &mut dyn Clipboard,
        now: Instant,
    ) -> Action {
        if let Mode::SignIn(login) = &mut self.mode {
            return match login.mouse_clicked(x, y, button, clipboard) {
                // "Sign in with Microsoft" after a failure: try again.
                Action::BeginLogin => Action::Accounts(AccountsAction::Add),
                Action::Back => self.back(),
                other => other,
            };
        }
        if button != 0 {
            return Action::None;
        }
        let hit = self
            .buttons
            .iter()
            .find(|b| b.mouse_pressed(x, y))
            .map(|b| b.id);
        if let Mode::Confirm { key, .. } = &self.mode {
            let key = *key;
            return match hit {
                Some(ids::CONFIRM_REMOVE) => {
                    self.mode = Mode::List;
                    Action::Accounts(AccountsAction::Remove(key))
                }
                Some(ids::CONFIRM_CANCEL) => {
                    self.mode = Mode::List;
                    Action::None
                }
                _ => Action::None,
            };
        }
        match hit {
            Some(ids::USE) => return self.use_selected(),
            Some(ids::ADD) => return self.add(),
            Some(ids::REMOVE) => {
                if let Some(row) = self.selected_row() {
                    self.mode = Mode::Confirm {
                        key: row.key,
                        name: row.name.clone(),
                    };
                }
                return Action::None;
            }
            Some(ids::DONE) => return Action::Accounts(AccountsAction::Done),
            _ => {}
        }
        if y >= self.list.top && y <= self.list.bottom {
            if let Some(index) = self.list.slot_index_at(x, y) {
                let double = self.last_click.is_some_and(|(last, at)| {
                    last == index && now.duration_since(at) < DOUBLE_CLICK
                });
                self.selected = Some(index);
                self.last_click = Some((index, now));
                self.refresh_buttons();
                if double {
                    return self.use_selected();
                }
                return Action::None;
            }
            // GuiSlot: a press in the list area may start a scrollbar drag.
            self.list.mouse_pressed(x, y);
        }
        Action::None
    }

    /// The Use button (or a double-click): switch, or sign in again for a dead sign-in.
    fn use_selected(&mut self) -> Action {
        let Some(row) = self.selected_row() else {
            return Action::None;
        };
        let key = row.key;
        if row.status == RowStatus::NeedsSignIn {
            return self.add();
        }
        if Some(key) == self.active {
            return Action::None;
        }
        if self.world_loaded {
            return Action::None;
        }
        Action::Accounts(AccountsAction::Use(key))
    }

    fn add(&mut self) -> Action {
        if !self.sign_in_available {
            // The status line says it; the client never starts a sign-in it cannot finish.
            self.status = Some((StatusText::NotSetUp, Tone::Notice));
            return Action::None;
        }
        Action::Accounts(AccountsAction::Add)
    }

    /// The mouse wheel over the list.
    pub fn mouse_wheel(&mut self, x: i32, y: i32, delta: f32) {
        if matches!(self.mode, Mode::List) {
            self.list.mouse_wheel_at(x, y, delta);
        }
    }

    /// Draw.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
    ) {
        match &mut self.mode {
            Mode::SignIn(login) => {
                login.draw(d, font, ctx, mouse, lang);
                return;
            }
            Mode::Confirm { name, .. } => {
                let question = text_with(lang, "aero.accounts.removeQuestion", "Remove %s?", name);
                draw_dirt_background(d, ctx.width, ctx.height, 0.0);
                draw_centered_string(
                    d,
                    font,
                    &question,
                    ctx.width / 2,
                    ctx.height / 2 - 34,
                    TITLE_COLOR,
                );
                let detail = text(
                    lang,
                    "aero.accounts.removeDetail",
                    "Its saved sign-in is deleted from this PC. You can add it again later.",
                );
                let mut y = ctx.height / 2 - 18;
                for line in font.wrap_to_lines(&detail, ctx.width - 60) {
                    draw_centered_string(d, font, &line, ctx.width / 2, y, LABEL_COLOR);
                    y += font.font_height() + 1;
                }
                for b in &mut self.buttons {
                    b.draw(d, font, mouse.0, mouse.1);
                }
                return;
            }
            Mode::List => {}
        }
        draw_dirt_background(d, ctx.width, ctx.height, 0.0);
        self.list.size = self.rows.len();
        self.list.draw_background(d);
        for index in self.list.visible_slots() {
            let rect = self.list.slot_rect(index);
            if self.selected == Some(index) {
                self.list.draw_selection_box(d, index);
            }
            let Some(row) = self.rows.get(index) else {
                continue;
            };
            draw_row(
                d,
                font,
                lang,
                row,
                self.active == Some(row.key),
                rect.x,
                rect.y,
            );
        }
        self.list.draw_overlay(d);
        draw_centered_string(
            d,
            font,
            &text(lang, "aero.accounts.title", "Accounts"),
            ctx.width / 2,
            16,
            TITLE_COLOR,
        );
        if self.rows.is_empty() {
            let empty = text(
                lang,
                "aero.accounts.empty",
                "No accounts yet. Add your Microsoft account to play on online servers.",
            );
            let mut y = 48;
            for line in font.wrap_to_lines(&empty, LIST_WIDTH) {
                draw_centered_string(d, font, &line, ctx.width / 2, y, LABEL_COLOR);
                y += font.font_height() + 2;
            }
        }
        let status = match &self.status {
            Some((StatusText::Text(message), tone)) => Some((message.clone(), *tone)),
            Some((StatusText::NotSetUp, tone)) => Some((not_set_up_text(lang), *tone)),
            None if self.world_loaded => Some((
                text(
                    lang,
                    "aero.accounts.inWorld",
                    "Leave the server to switch accounts.",
                ),
                Tone::Notice,
            )),
            None => None,
        };
        if let Some((message, tone)) = status {
            let color = match tone {
                Tone::Done => GREEN,
                Tone::Notice => YELLOW,
                Tone::Problem => RED,
            };
            let line = font.trim_to_width(&message, ctx.width - 20);
            draw_centered_string(d, font, &line, ctx.width / 2, ctx.height - 68, color);
        }
        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
    }
}

/// One 32-pixel row: the head where the server list has its icon, the name, what kind of
/// account it is, and its state.
fn draw_row(
    d: &mut DrawList,
    font: &mut FontRenderer,
    lang: &Language,
    row: &AccountRow,
    in_use: bool,
    x: i32,
    y: i32,
) {
    draw_head(d, &row.head, row.head_sheet_height, x, y, 32);
    draw_string(d, font, &row.name, x + 35, y + 1, 0xFF_FFFF);
    let kind = match row.key {
        AccountKey::Launcher => text(lang, "aero.accounts.fromLauncher", "From the launcher"),
        AccountKey::Stored(_) => text(lang, "aero.accounts.microsoft", "Microsoft account"),
    };
    draw_string(d, font, &kind, x + 35, y + 12, GREY);
    let state = match (row.status, in_use) {
        (RowStatus::NeedsSignIn, _) => Some((
            text(lang, "aero.accounts.needsSignIn", "Sign in again to use it"),
            YELLOW,
        )),
        (RowStatus::Ready, true) => Some((text(lang, "aero.accounts.inUse", "In use"), GREEN)),
        (RowStatus::Ready, false) => None,
    };
    if let Some((label, color)) = state {
        draw_string(d, font, &label, x + 35, y + 23, color);
    }
}

/// A player's head from their skin sheet: the 8x8 face, then the hat layer over it.
/// java: `GuiPlayerTabOverlay`'s `drawScaledCustomSizeModalRect(8, 8, 8, 8, ...)` and
/// `(40, 8, 8, 8, ...)`, here at `size` pixels.
pub fn draw_head(
    d: &mut DrawList,
    sheet: &UiTexture,
    sheet_height: f32,
    x: i32,
    y: i32,
    size: i32,
) {
    d.blit_scaled(sheet, x, y, 8.0, 8.0, 8, 8, size, size, 64.0, sheet_height);
    d.blit_scaled(sheet, x, y, 40.0, 8.0, 8, 8, size, size, 64.0, sheet_height);
}

#[cfg(test)]
mod tests;
