//! The in-game menu (Escape).
//!
//! java: `client/gui/GuiIngameMenu.java`
//!
//! Multiplayer-only changes exactly two things from vanilla: button 1 always
//! reads "Disconnect" rather than "Save and Quit to Title", and "Open to LAN" is permanently
//! disabled. Every coordinate is unchanged.
//!
//! Aero adds one button: **Multiplayer**, a
//! full-width button in the row vanilla leaves empty (`height / 4 + 72 - 16`, between
//! Achievements/Statistics and Options/Open to LAN), so no vanilla button moves. It opens the
//! server list over the running game ([`Action::OpenMultiplayer`]); the client keeps the
//! connection alive while the list is open, returns here on Cancel, and disconnects cleanly
//! before joining another server (`app/mods/x3_screens.rs`).
//!
//! Aero puts **Mods** where vanilla's "Open to
//! LAN" is: that button can never work in this multiplayer-only client (the multiplayer-only rule, it was
//! permanently disabled), Mods sits beside Options where players look, and no button moves. It
//! opens the mod menu over the pause menu ([`Action::OpenModMenu`]); closing the menu comes back
//! here.
//!
//! **Confirm Disconnect** (`mc_mods::mods::smart_disconnect`, off by
//! default). While the client says it is on ([`PauseScreen::set_confirm_disconnect`]), Disconnect
//! asks first: a vanilla [`YesNoScreen`] drawn in this screen's place (as `GuiGameOver` embeds its
//! own leave question), armed after `ARM_TICKS`. Its Disconnect returns the unchanged
//! [`Action::Disconnect`]; its Cancel brings the pause menu back. Off, the button acts at once.

use mc_assets::Language;
use mc_mods::mods::smart_disconnect::ARM_TICKS;

use crate::font::FontRenderer;
use crate::screen_stack::{Screen, ScreenCtx};
use crate::screens::common::{draw_world_background, TITLE_COLOR};
use crate::screens::yes_no::YesNoScreen;
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::{draw_centered_string, Button};

/// Button ids, matching vanilla's.
mod ids {
    pub const OPTIONS: i32 = 0;
    pub const DISCONNECT: i32 = 1;
    pub const RETURN: i32 = 4;
    pub const ACHIEVEMENTS: i32 = 5;
    pub const STATS: i32 = 6;
    /// Vanilla's "Open to LAN" id: no button has it any more (Mods took its place).
    #[cfg(test)]
    pub const OPEN_TO_LAN: i32 = 7;
    /// X3: not a vanilla pause button. 2 is `GuiMainMenu`'s multiplayer id, unused here.
    pub const MULTIPLAYER: i32 = 2;
    /// Not a vanilla pause button; in Open to LAN's place.
    pub const MODS: i32 = 8;
}

/// The question's first line.
pub const CONFIRM_DISCONNECT_TITLE: &str = "Are you sure you want to disconnect?";
/// The question's second line: where the question comes from, so it can be turned off.
pub const CONFIRM_DISCONNECT_DETAIL: &str =
    "Confirm Disconnect asks first. You can turn it off in Mods.";

/// The pause menu.
#[derive(Debug, Default)]
pub struct PauseScreen {
    buttons: Vec<Button>,
    /// Disconnect asks first (set by the client before a click).
    confirm_disconnect: bool,
    /// The question, while it is up. It replaces this screen's buttons and drawing.
    confirmation: Option<YesNoScreen>,
    /// The size from the last [`PauseScreen::init`], for laying the question out.
    width: i32,
    height: i32,
    /// The question's buttons: `menu.disconnect` and `gui.cancel`, translated at `init`.
    confirm_text: String,
    cancel_text: String,
}

impl PauseScreen {
    /// A pause menu.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lay out. java: `initGui`, with `i = -16`.
    // java: GuiIngameMenu.java:19
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        let i = -16;
        let tr = |k: &str| lang.tr(k).to_owned();
        // java: the label is `menu.returnToMenu` in single-player and `menu.disconnect`
        // otherwise. There is no "otherwise" here (I1).
        // Mods, in the place of vanilla's "Open to LAN" (never usable
        // here: I1), the same size and position.
        let mods = Button::sized(
            ids::MODS,
            w / 2 + 2,
            h / 4 + 96 + i,
            98,
            20,
            "Mods".to_owned(),
        );
        self.buttons = vec![
            Button::new(
                ids::DISCONNECT,
                w / 2 - 100,
                h / 4 + 120 + i,
                tr("menu.disconnect"),
            ),
            Button::new(
                ids::RETURN,
                w / 2 - 100,
                h / 4 + 24 + i,
                tr("menu.returnToGame"),
            ),
            Button::sized(
                ids::OPTIONS,
                w / 2 - 100,
                h / 4 + 96 + i,
                98,
                20,
                tr("menu.options"),
            ),
            mods,
            Button::sized(
                ids::ACHIEVEMENTS,
                w / 2 - 100,
                h / 4 + 48 + i,
                98,
                20,
                tr("gui.achievements"),
            ),
            Button::sized(
                ids::STATS,
                w / 2 + 2,
                h / 4 + 48 + i,
                98,
                20,
                tr("gui.stats"),
            ),
            // The row vanilla leaves empty.
            Button::new(
                ids::MULTIPLAYER,
                w / 2 - 100,
                h / 4 + 72 + i,
                tr("menu.multiplayer"),
            ),
        ];
        // An open question follows the new size. `GuiYesNo.initGui`
        // makes fresh, enabled buttons; the remaining arm time is put back so a resize cannot
        // skip it.
        self.width = w;
        self.height = h;
        self.confirm_text = tr("menu.disconnect");
        self.cancel_text = tr("gui.cancel");
        if let Some(question) = &mut self.confirmation {
            question.resize_keeping_delay(w, h);
        }
    }

    /// Whether Disconnect asks first. The client sets this before
    /// each click from Confirm Disconnect (and never while a replay plays).
    pub fn set_confirm_disconnect(&mut self, on: bool) {
        self.confirm_disconnect = on;
    }

    /// Whether the disconnect question is up.
    #[must_use]
    pub fn is_confirming_disconnect(&self) -> bool {
        self.confirmation.is_some()
    }

    /// Whether the question's buttons accept clicks yet (`false` while it is not up).
    #[must_use]
    pub fn confirmation_armed(&self) -> bool {
        self.confirmation
            .as_ref()
            .is_some_and(YesNoScreen::buttons_enabled)
    }

    /// One client tick: counts the question's arm time down. java: `GuiYesNo.updateScreen`.
    pub fn tick(&mut self) {
        if let Some(question) = &mut self.confirmation {
            Screen::tick(question);
        }
    }

    fn open_disconnect_confirmation(&mut self) {
        let mut question = YesNoScreen::new(
            CONFIRM_DISCONNECT_TITLE,
            CONFIRM_DISCONNECT_DETAIL,
            self.confirm_text.clone(),
            self.cancel_text.clone(),
            ids::DISCONNECT,
        )
        .with_world_loaded(true);
        question.resize(self.width, self.height);
        question.set_button_delay(ARM_TICKS);
        self.confirmation = Some(question);
    }

    /// Handle a click.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> Action {
        // While the question is up it takes every click.
        if let Some(question) = &mut self.confirmation {
            return match question.click(x, y, button) {
                Some(answer) if answer.confirmed => {
                    self.confirmation = None;
                    self.leave()
                }
                Some(_) => {
                    // Cancel: back to the pause menu, exactly as it was.
                    self.confirmation = None;
                    Action::None
                }
                None => Action::None,
            };
        }
        if button != 0 {
            return Action::None;
        }
        let Some(id) = self
            .buttons
            .iter()
            .find(|b| b.mouse_pressed(x, y))
            .map(|b| b.id)
        else {
            return Action::None;
        };
        match id {
            ids::OPTIONS => Action::OpenOptions,
            ids::DISCONNECT if self.confirm_disconnect => {
                self.open_disconnect_confirmation();
                Action::None
            }
            ids::DISCONNECT => self.leave(),
            ids::RETURN => Action::CloseAll,
            ids::ACHIEVEMENTS => Action::OpenAchievements,
            ids::STATS => Action::OpenStats,
            ids::MULTIPLAYER => Action::OpenMultiplayer,
            ids::MODS => Action::OpenModMenu,
            _ => Action::None,
        }
    }

    /// The Disconnect button's own action. java: disable the button before the network close
    /// and teardown begin.
    fn leave(&mut self) -> Action {
        if let Some(b) = self.buttons.iter_mut().find(|b| b.id == ids::DISCONNECT) {
            b.enabled = false;
        }
        Action::Disconnect
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
        // The question is drawn in this screen's place, as
        // `GuiYesNo` would be after `displayGuiScreen`.
        if let Some(question) = &mut self.confirmation {
            question.draw_dialog(d, font, mouse);
            return;
        }
        draw_world_background(d, ctx.width, ctx.height, true);
        draw_centered_string(
            d,
            font,
            lang.tr("menu.game"),
            ctx.width / 2,
            40,
            TITLE_COLOR,
        );
        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
    }
}

impl Screen for PauseScreen {
    fn name(&self) -> &'static str {
        "pause"
    }
    /// The multiplayer-only rule: multiplayer has no clock to stop, but the flag is what the F3 overlay
    /// reports and what a single-player build would honour.
    fn pauses_game(&self) -> bool {
        true
    }
    fn tick(&mut self) {
        PauseScreen::tick(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn screen() -> PauseScreen {
        let mut s = PauseScreen::new();
        s.init(&ctx(), &Language::new("en_US"));
        s
    }

    #[test]
    fn the_buttons_land_on_vanillas_coordinates() {
        let s = screen();
        let by = |id: i32| s.buttons.iter().find(|b| b.id == id).expect("button");
        let base = 480 / 4 - 16;
        assert_eq!(
            (by(ids::RETURN).rect.x, by(ids::RETURN).rect.y),
            (327, base + 24)
        );
        assert_eq!(by(ids::RETURN).rect.width, 200);
        assert_eq!(by(ids::ACHIEVEMENTS).rect.y, base + 48);
        assert_eq!(by(ids::ACHIEVEMENTS).rect.width, 98);
        assert_eq!(by(ids::STATS).rect.x, 854 / 2 + 2);
        assert_eq!(by(ids::OPTIONS).rect.y, base + 96);
        assert_eq!(by(ids::DISCONNECT).rect.y, base + 120);
    }

    /// X3: the Multiplayer button fills vanilla's empty row and overlaps no vanilla button.
    #[test]
    fn the_multiplayer_button_fills_the_empty_row_without_moving_anything() {
        let s = screen();
        let mp = s
            .buttons
            .iter()
            .find(|b| b.id == ids::MULTIPLAYER)
            .expect("the multiplayer button");
        let base = 480 / 4 - 16;
        assert_eq!((mp.rect.x, mp.rect.y), (854 / 2 - 100, base + 72));
        assert_eq!((mp.rect.width, mp.rect.height), (200, 20));
        assert!(mp.enabled);
        for other in s.buttons.iter().filter(|b| b.id != ids::MULTIPLAYER) {
            let (a, b) = (mp.rect, other.rect);
            let overlap =
                a.x < b.right() && b.x < a.right() && a.y < b.bottom() && b.y < a.bottom();
            assert!(!overlap, "overlaps button {}", other.id);
        }
        let mut lang = Language::new("en_US");
        lang.insert("menu.multiplayer", "Multiplayer");
        let mut s = PauseScreen::new();
        s.init(&ctx(), &lang);
        let mp = s.buttons.iter().find(|b| b.id == ids::MULTIPLAYER).unwrap();
        assert_eq!(mp.text, "Multiplayer");
    }

    /// Mods takes Open to LAN's exact place (that button was
    /// permanently disabled by the multiplayer-only rule), so no vanilla button moves.
    #[test]
    fn mods_takes_the_place_of_open_to_lan_which_can_never_work_here() {
        let s = screen();
        assert!(
            s.buttons.iter().all(|b| b.id != ids::OPEN_TO_LAN),
            "Open to LAN is gone"
        );
        let mods = s.buttons.iter().find(|b| b.id == ids::MODS).unwrap();
        let base = 480 / 4 - 16;
        assert_eq!((mods.rect.x, mods.rect.y), (854 / 2 + 2, base + 96));
        assert_eq!((mods.rect.width, mods.rect.height), (98, 20));
        assert!(mods.enabled);
        assert_eq!(mods.text, "Mods");
        for other in s.buttons.iter().filter(|b| b.id != ids::MODS) {
            let (a, b) = (mods.rect, other.rect);
            let overlap =
                a.x < b.right() && b.x < a.right() && a.y < b.bottom() && b.y < a.bottom();
            assert!(!overlap, "overlaps button {}", other.id);
        }
    }

    #[test]
    fn each_button_produces_its_action() {
        let mut s = screen();
        let hit = |s: &mut PauseScreen, id: i32| {
            let b = s.buttons.iter().find(|b| b.id == id).unwrap();
            let (x, y) = (b.rect.x + 1, b.rect.y + 1);
            s.mouse_clicked(x, y, 0)
        };
        assert_eq!(hit(&mut s, ids::RETURN), Action::CloseAll);
        assert_eq!(hit(&mut s, ids::OPTIONS), Action::OpenOptions);
        assert_eq!(hit(&mut s, ids::DISCONNECT), Action::Disconnect);
        assert_eq!(hit(&mut s, ids::ACHIEVEMENTS), Action::OpenAchievements);
        assert_eq!(hit(&mut s, ids::STATS), Action::OpenStats);
        assert_eq!(hit(&mut s, ids::MULTIPLAYER), Action::OpenMultiplayer);
        assert_eq!(hit(&mut s, ids::MODS), Action::OpenModMenu);
        assert_eq!(
            s.mouse_clicked(0, 0, 0),
            Action::None,
            "a miss does nothing"
        );
    }

    #[test]
    fn the_disconnect_label_never_says_save_and_quit() {
        let mut lang = Language::new("en_US");
        lang.insert("menu.disconnect", "Disconnect");
        lang.insert("menu.returnToMenu", "Save and Quit to Title");
        let mut s = PauseScreen::new();
        s.init(&ctx(), &lang);
        let b = s.buttons.iter().find(|b| b.id == ids::DISCONNECT).unwrap();
        assert_eq!(b.text, "Disconnect");
    }

    /// Where `GuiYesNo` puts its two buttons on an 854x480 screen.
    fn question_point(confirm: bool) -> (i32, i32) {
        let (w, h) = (854, 480);
        let x = if confirm { w / 2 - 155 } else { w / 2 + 5 };
        (x + 1, h / 6 + 96 + 1)
    }

    fn click_disconnect(s: &mut PauseScreen) -> Action {
        let b = s.buttons.iter().find(|b| b.id == ids::DISCONNECT).unwrap();
        let (x, y) = (b.rect.x + 1, b.rect.y + 1);
        s.mouse_clicked(x, y, 0)
    }

    /// Confirm Disconnect: Disconnect asks; a double click, or any click before the arm time, does
    /// nothing; Cancel brings the pause menu back untouched; only the question's Disconnect
    /// returns the button's own action.
    #[test]
    fn smart_disconnect_asks_first_and_only_its_disconnect_leaves() {
        let mut s = screen();
        s.set_confirm_disconnect(true);
        assert_eq!(click_disconnect(&mut s), Action::None, "the question opens");
        assert!(s.is_confirming_disconnect());
        assert!(!s.confirmation_armed(), "armed only after ARM_TICKS");
        // A double click lands on the question, not on a button of it.
        assert_eq!(click_disconnect(&mut s), Action::None);
        let (x, y) = question_point(true);
        assert_eq!(
            s.mouse_clicked(x, y, 0),
            Action::None,
            "too early to answer"
        );
        assert!(s.is_confirming_disconnect());
        for _ in 1..ARM_TICKS {
            s.tick();
        }
        assert!(!s.confirmation_armed(), "one tick short");
        s.tick();
        assert!(s.confirmation_armed(), "armed after {ARM_TICKS} ticks");
        // Cancel: the pause menu again, its Disconnect still live.
        let (x, y) = question_point(false);
        assert_eq!(s.mouse_clicked(x, y, 0), Action::None);
        assert!(!s.is_confirming_disconnect());
        assert!(
            s.buttons
                .iter()
                .find(|b| b.id == ids::DISCONNECT)
                .unwrap()
                .enabled
        );
        // Ask again and confirm: the button's own action, button disabled as before.
        assert_eq!(click_disconnect(&mut s), Action::None);
        for _ in 0..ARM_TICKS {
            s.tick();
        }
        let (x, y) = question_point(true);
        assert_eq!(s.mouse_clicked(x, y, 0), Action::Disconnect);
        assert!(!s.is_confirming_disconnect());
        assert_eq!(
            click_disconnect(&mut s),
            Action::None,
            "the button is off now"
        );
        // A right click never answers.
        let mut s = screen();
        s.set_confirm_disconnect(true);
        click_disconnect(&mut s);
        for _ in 0..ARM_TICKS {
            s.tick();
        }
        let (x, y) = question_point(true);
        assert_eq!(s.mouse_clicked(x, y, 1), Action::None);
        assert!(s.is_confirming_disconnect());
    }

    /// Off (the default) is vanilla: the first click leaves, no question.
    #[test]
    fn without_smart_disconnect_the_button_leaves_at_once() {
        let mut s = screen();
        assert_eq!(click_disconnect(&mut s), Action::Disconnect);
        assert!(!s.is_confirming_disconnect());
        let mut s = screen();
        s.set_confirm_disconnect(true);
        s.set_confirm_disconnect(false);
        assert_eq!(click_disconnect(&mut s), Action::Disconnect);
    }

    /// A resize while the question is up keeps it, laid out for the new size, and keeps the
    /// remaining arm time (vanilla's `initGui` would make the buttons live at once).
    #[test]
    fn a_resize_keeps_the_question_and_its_remaining_arm_time() {
        let mut s = screen();
        s.set_confirm_disconnect(true);
        click_disconnect(&mut s);
        for _ in 0..3 {
            s.tick();
        }
        s.init(&ctx(), &Language::new("en_US"));
        assert!(s.is_confirming_disconnect());
        assert!(!s.confirmation_armed());
        for _ in 3..ARM_TICKS - 1 {
            s.tick();
        }
        assert!(!s.confirmation_armed());
        s.tick();
        assert!(s.confirmation_armed());
    }

    /// The question's Disconnect never lies under the pause menu's Disconnect, at any GUI size
    /// a window can have (the scaled height is at least 240 above GUI scale 1), so the second
    /// half of a double click cannot answer it even after the arm time.
    #[test]
    fn the_questions_disconnect_never_lies_under_the_pause_disconnect() {
        for height in 160..=1440 {
            let width = 854;
            let mut s = PauseScreen::new();
            s.init(
                &ScreenCtx::new(
                    ScaledResolution::new(width as u32, height as u32, 1, false),
                    0.0,
                ),
                &Language::new("en_US"),
            );
            let pause = s
                .buttons
                .iter()
                .find(|b| b.id == ids::DISCONNECT)
                .unwrap()
                .rect;
            let (qx, qy, qw, qh) = (width / 2 - 155, height / 6 + 96, 150, 20);
            let overlap =
                qx < pause.right() && pause.x < qx + qw && qy < pause.bottom() && pause.y < qy + qh;
            assert!(!overlap, "height {height}");
        }
    }

    #[test]
    fn disconnect_disables_its_button_before_returning_the_action() {
        let mut s = screen();
        let b = s.buttons.iter().find(|b| b.id == ids::DISCONNECT).unwrap();
        let point = (b.rect.x + 1, b.rect.y + 1);
        assert_eq!(s.mouse_clicked(point.0, point.1, 0), Action::Disconnect);
        assert_eq!(s.mouse_clicked(point.0, point.1, 0), Action::None);
        assert!(
            !s.buttons
                .iter()
                .find(|b| b.id == ids::DISCONNECT)
                .unwrap()
                .enabled
        );
    }
}
