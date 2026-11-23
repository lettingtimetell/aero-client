//! Reusable two-button confirmation dialog.
//!
//! java: `client/gui/GuiYesNo.java`, `client/gui/GuiYesNoCallback.java`
//!
//! Vanilla passes decisions to a `GuiYesNoCallback`. Rust callers instead poll
//! [`YesNoScreen::take_result`], which carries both the boolean decision and the opaque parent
//! button id. Keeping that id is important: the multiplayer screen, resource-pack prompt and
//! link prompt all reuse this dialog without sharing an action enum.

use crate::font::FontRenderer;
use crate::screen_stack::{Screen, ScreenAction, ScreenCtx};
use crate::screens::common::{draw_world_background, TITLE_COLOR};
use crate::widget::draw::DrawList;
use crate::widget::{draw_centered_string, Button};

mod ids {
    pub const CONFIRM: i32 = 0;
    pub const CANCEL: i32 = 1;
}

/// The value vanilla sends to `GuiYesNoCallback.confirmClicked(result, id)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YesNoResult {
    /// `true` for button id `0` (confirm), `false` for button id `1` (cancel).
    pub confirmed: bool,
    /// The opaque id supplied by the parent when it constructed the dialog.
    pub parent_button_id: i32,
}

/// A faithful `GuiYesNo` state machine without a back-pointer to its parent screen.
#[derive(Debug, Clone)]
pub struct YesNoScreen {
    message_line_1: String,
    message_line_2: String,
    confirm_button_text: String,
    cancel_button_text: String,
    parent_button_id: i32,
    buttons: Vec<Button>,
    /// java: `ticksUntilEnable`.
    ticks_until_enable: i32,
    result: Option<YesNoResult>,
    width: i32,
    height: i32,
    world_loaded: bool,
}

impl YesNoScreen {
    /// Construct the five-string `GuiYesNo` overload.
    #[must_use]
    pub fn new(
        message_line_1: impl Into<String>,
        message_line_2: impl Into<String>,
        confirm_button_text: impl Into<String>,
        cancel_button_text: impl Into<String>,
        parent_button_id: i32,
    ) -> Self {
        Self {
            message_line_1: message_line_1.into(),
            message_line_2: message_line_2.into(),
            confirm_button_text: confirm_button_text.into(),
            cancel_button_text: cancel_button_text.into(),
            parent_button_id,
            buttons: Vec::new(),
            ticks_until_enable: 0,
            result: None,
            width: 0,
            height: 0,
            world_loaded: false,
        }
    }

    /// Select whether `drawDefaultBackground` sees a live world.
    #[must_use]
    pub fn with_world_loaded(mut self, world_loaded: bool) -> Self {
        self.world_loaded = world_loaded;
        self
    }

    /// Lay out exactly the two vanilla `GuiOptionButton`s.
    ///
    /// Just like `initGui`, this creates fresh, enabled buttons. In particular, resizing while a
    /// button delay is counting does *not* re-disable the replacements; that oddity is visible in
    /// the Java client.
    pub fn resize(&mut self, width: i32, height: i32) {
        self.width = width;
        self.height = height;
        let y = height / 6 + 96;
        self.buttons = vec![
            Button::sized(
                ids::CONFIRM,
                width / 2 - 155,
                y,
                150,
                20,
                self.confirm_button_text.clone(),
            ),
            Button::sized(
                ids::CANCEL,
                width / 2 + 5,
                y,
                150,
                20,
                self.cancel_button_text.clone(),
            ),
        ];
    }

    /// [`YesNoScreen::resize`], then the remaining button delay put
    /// back, so a relayout (every frame, or a window resize) cannot wake the buttons early the way
    /// vanilla's `initGui` would. For questions this client arms itself (Confirm Disconnect);
    /// vanilla's own dialogs keep [`YesNoScreen::resize`].
    pub fn resize_keeping_delay(&mut self, width: i32, height: i32) {
        let remaining = self.ticks_until_enable;
        self.resize(width, height);
        if remaining > 0 {
            self.set_button_delay(remaining);
        }
    }

    /// `GuiYesNo.setButtonDelay`: save the countdown and disable the buttons that exist now.
    pub fn set_button_delay(&mut self, ticks: i32) {
        self.ticks_until_enable = ticks;
        for button in &mut self.buttons {
            button.enabled = false;
        }
    }

    /// Current value of vanilla's pre-decremented delay counter.
    #[must_use]
    pub fn ticks_until_enable(&self) -> i32 {
        self.ticks_until_enable
    }

    /// Whether both dialog buttons currently accept clicks.
    #[must_use]
    pub fn buttons_enabled(&self) -> bool {
        !self.buttons.is_empty() && self.buttons.iter().all(|button| button.enabled)
    }

    /// Return one callback value at most once.
    pub fn take_result(&mut self) -> Option<YesNoResult> {
        self.result.take()
    }

    /// Handle the vanilla left-click activation and return the callback value immediately.
    pub fn click(&mut self, x: i32, y: i32, button: i32) -> Option<YesNoResult> {
        if button != 0 {
            return None;
        }
        let id = self
            .buttons
            .iter()
            .find(|candidate| candidate.mouse_pressed(x, y))?
            .id;
        let result = YesNoResult {
            confirmed: id == ids::CONFIRM,
            parent_button_id: self.parent_button_id,
        };
        self.result = Some(result);
        Some(result)
    }

    /// Draw `GuiYesNo.drawScreen`, including the wrapped second message line.
    pub fn draw_dialog(&mut self, d: &mut DrawList, font: &mut FontRenderer, mouse: (i32, i32)) {
        draw_world_background(d, self.width, self.height, self.world_loaded);
        draw_centered_string(
            d,
            font,
            &self.message_line_1,
            self.width / 2,
            70,
            TITLE_COLOR,
        );
        let mut y = 90;
        for line in font.wrap_to_lines(&self.message_line_2, self.width - 50) {
            draw_centered_string(d, font, &line, self.width / 2, y, TITLE_COLOR);
            y += font.font_height();
        }
        for button in &mut self.buttons {
            button.draw(d, font, mouse.0, mouse.1);
        }
    }
}

impl Screen for YesNoScreen {
    fn name(&self) -> &'static str {
        "yes_no"
    }

    fn init(&mut self, ctx: &ScreenCtx) {
        self.resize(ctx.width, ctx.height);
    }

    fn tick(&mut self) {
        // java: `if (--ticksUntilEnable == 0)`. The pre-decrement, equality check and lack of a
        // lower bound are all intentional.
        self.ticks_until_enable -= 1;
        if self.ticks_until_enable == 0 {
            for button in &mut self.buttons {
                button.enabled = true;
            }
        }
    }

    fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        _ctx: &ScreenCtx,
        mouse: (i32, i32),
    ) {
        self.draw_dialog(d, font, mouse);
    }

    fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> ScreenAction {
        self.click(x, y, button);
        ScreenAction::None
    }
}

/// Which button `GuiConfirmOpenLink` activated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenLinkChoice {
    /// Button `0`: open the link.
    Open,
    /// Button `2`: copy the link, then report `false` to the parent callback.
    Copy,
    /// Button `1`: cancel.
    Cancel,
}

/// The link as the dialog draws it: every `§` becomes U+FFFD.
///
/// java: `GuiYesNo` draws the link with the font's formatting codes live, so a server's link
/// could color part of itself black or obfuscate it (`§0`, `§k`) and hide where it really goes
/// in the very dialog that exists to show that. A link without `§` (every real one) is drawn
/// exactly as before; which links open, and what Open and Copy use, do not change.
#[must_use]
pub fn link_as_shown(link: &str) -> String {
    link.replace('\u{a7}', "\u{fffd}")
}

/// The three-button `GuiConfirmOpenLink` specialization.
#[derive(Debug, Clone)]
pub struct ConfirmOpenLinkScreen {
    message: String,
    link: String,
    /// The link as the dialog draws it (see [`link_as_shown`]).
    shown_link: String,
    open_text: String,
    copy_text: String,
    cancel_text: String,
    warning: String,
    show_security_warning: bool,
    parent_button_id: i32,
    buttons: Vec<Button>,
    result: Option<(OpenLinkChoice, i32)>,
    width: i32,
    height: i32,
    world_loaded: bool,
}

impl ConfirmOpenLinkScreen {
    /// Construct from already-localized labels.
    #[must_use]
    pub fn new(
        message: impl Into<String>,
        link: impl Into<String>,
        open_text: impl Into<String>,
        copy_text: impl Into<String>,
        cancel_text: impl Into<String>,
        warning: impl Into<String>,
        parent_button_id: i32,
    ) -> Self {
        let link = link.into();
        Self {
            message: message.into(),
            shown_link: link_as_shown(&link),
            link,
            open_text: open_text.into(),
            copy_text: copy_text.into(),
            cancel_text: cancel_text.into(),
            warning: warning.into(),
            show_security_warning: true,
            parent_button_id,
            buttons: Vec::new(),
            result: None,
            width: 0,
            height: 0,
            world_loaded: false,
        }
    }

    /// Select whether `drawDefaultBackground` sees a live world.
    #[must_use]
    pub fn with_world_loaded(mut self, world_loaded: bool) -> Self {
        self.world_loaded = world_loaded;
        self
    }

    /// java: `GuiConfirmOpenLink.disableSecurityWarning`.
    pub fn disable_security_warning(&mut self) {
        self.show_security_warning = false;
    }

    /// Recreate the three buttons at vanilla's exact coordinates.
    pub fn resize(&mut self, width: i32, height: i32) {
        self.width = width;
        self.height = height;
        let y = height / 6 + 96;
        self.buttons = vec![
            Button::sized(0, width / 2 - 155, y, 100, 20, self.open_text.clone()),
            Button::sized(2, width / 2 - 50, y, 100, 20, self.copy_text.clone()),
            Button::sized(1, width / 2 + 55, y, 100, 20, self.cancel_text.clone()),
        ];
    }

    /// Return the URL this dialog presents, for the clipboard/open handler.
    #[must_use]
    pub fn link(&self) -> &str {
        &self.link
    }

    /// Handle a left click and return the chosen operation plus the opaque parent id.
    pub fn click(&mut self, x: i32, y: i32, button: i32) -> Option<(OpenLinkChoice, i32)> {
        if button != 0 {
            return None;
        }
        let id = self
            .buttons
            .iter()
            .find(|candidate| candidate.mouse_pressed(x, y))?
            .id;
        let choice = match id {
            0 => OpenLinkChoice::Open,
            2 => OpenLinkChoice::Copy,
            _ => OpenLinkChoice::Cancel,
        };
        let result = (choice, self.parent_button_id);
        self.result = Some(result);
        Some(result)
    }

    /// Return one click result at most once.
    pub fn take_result(&mut self) -> Option<(OpenLinkChoice, i32)> {
        self.result.take()
    }

    fn draw_dialog(&mut self, d: &mut DrawList, font: &mut FontRenderer, mouse: (i32, i32)) {
        draw_world_background(d, self.width, self.height, self.world_loaded);
        draw_centered_string(d, font, &self.message, self.width / 2, 70, TITLE_COLOR);
        let mut y = 90;
        for line in font.wrap_to_lines(&self.shown_link, self.width - 50) {
            draw_centered_string(d, font, &line, self.width / 2, y, TITLE_COLOR);
            y += font.font_height();
        }

        // GuiConfirmOpenLink delegates to GuiYesNo.drawScreen first, which calls
        // GuiScreen.drawScreen and emits every button before this overlay.
        for button in &mut self.buttons {
            button.draw(d, font, mouse.0, mouse.1);
        }
        if self.show_security_warning {
            draw_centered_string(d, font, &self.warning, self.width / 2, 110, 16_764_108);
        }
    }
}

impl Screen for ConfirmOpenLinkScreen {
    fn name(&self) -> &'static str {
        "confirm_open_link"
    }

    fn init(&mut self, ctx: &ScreenCtx) {
        self.resize(ctx.width, ctx.height);
    }

    fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        _ctx: &ScreenCtx,
        mouse: (i32, i32),
    ) {
        self.draw_dialog(d, font, mouse);
    }

    fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> ScreenAction {
        self.click(x, y, button);
        ScreenAction::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::ScaledResolution;

    fn dialog() -> YesNoScreen {
        let mut screen = YesNoScreen::new("Question", "Details", "Yes", "No", 42);
        screen.resize(854, 480);
        screen
    }

    #[test]
    fn buttons_use_gui_yes_nos_exact_geometry_and_callback_payload() {
        let mut screen = dialog();
        let y = 480 / 6 + 96;
        assert_eq!(screen.buttons[0].rect.x, 854 / 2 - 155);
        assert_eq!(screen.buttons[1].rect.x, 854 / 2 + 5);
        assert_eq!(screen.buttons[0].rect.y, y);
        assert_eq!(screen.buttons[0].rect.width, 150);

        assert_eq!(
            screen.click(854 / 2 - 154, y + 1, 0),
            Some(YesNoResult {
                confirmed: true,
                parent_button_id: 42,
            })
        );
        assert_eq!(
            screen.take_result(),
            Some(YesNoResult {
                confirmed: true,
                parent_button_id: 42,
            })
        );
        assert_eq!(screen.take_result(), None);
    }

    #[test]
    fn confirm_and_cancel_are_distinct_and_right_click_is_ignored() {
        let mut screen = dialog();
        let y = 480 / 6 + 97;
        assert_eq!(screen.click(854 / 2 + 6, y, 1), None);
        assert_eq!(
            screen.click(854 / 2 + 6, y, 0),
            Some(YesNoResult {
                confirmed: false,
                parent_button_id: 42,
            })
        );
    }

    #[test]
    fn delay_uses_vanillas_predecrement_equality_gate() {
        let mut screen = dialog();
        screen.set_button_delay(20);
        for remaining in (1..=20).rev() {
            assert_eq!(screen.ticks_until_enable(), remaining);
            assert!(!screen.buttons_enabled());
            Screen::tick(&mut screen);
        }
        assert_eq!(screen.ticks_until_enable(), 0);
        assert!(screen.buttons_enabled());
    }

    #[test]
    fn resize_during_delay_recreates_enabled_buttons_without_resetting_countdown() {
        let mut screen = dialog();
        screen.set_button_delay(20);
        Screen::tick(&mut screen);
        screen.resize(1920, 1080);
        assert_eq!(screen.ticks_until_enable(), 19);
        assert!(screen.buttons_enabled());
    }

    /// The client's own armed questions keep the delay across a
    /// relayout, and wake on the same tick as without one.
    #[test]
    fn resize_keeping_delay_keeps_the_remaining_countdown() {
        let mut screen = dialog();
        screen.set_button_delay(3);
        Screen::tick(&mut screen);
        screen.resize_keeping_delay(1920, 1080);
        assert_eq!(screen.ticks_until_enable(), 2);
        assert!(!screen.buttons_enabled(), "still waiting");
        assert_eq!(
            screen.buttons[0].rect.x,
            1920 / 2 - 155,
            "laid out for the new size"
        );
        Screen::tick(&mut screen);
        screen.resize_keeping_delay(1920, 1080);
        assert!(!screen.buttons_enabled());
        Screen::tick(&mut screen);
        assert!(screen.buttons_enabled(), "awake after the third tick");
        screen.resize_keeping_delay(854, 480);
        assert!(screen.buttons_enabled(), "a spent delay is not put back");
        // A question laid out for the first time after its delay was set starts disarmed.
        let mut fresh = YesNoScreen::new("Question", "Details", "Yes", "No", 0);
        fresh.set_button_delay(2);
        fresh.resize_keeping_delay(854, 480);
        assert!(!fresh.buttons_enabled());
    }

    #[test]
    fn second_message_uses_font_renderer_wrapping_width() {
        let mut screen = YesNoScreen::new("Question", "aa bb cc", "Yes", "No", 0);
        screen.resize(80, 100);
        let font = crate::widget::test_support::uniform_font();
        assert_eq!(
            font.wrap_to_lines(&screen.message_line_2, 30),
            ["aa bb", "cc"]
        );
        assert_eq!(screen.width - 50, 30);
    }

    #[test]
    fn screen_trait_escape_inherits_gui_screens_pop_behavior() {
        let mut screen = dialog();
        assert_eq!(
            Screen::key_typed(
                &mut screen,
                None,
                crate::widget::keys::ESCAPE,
                crate::widget::keys::Modifiers::NONE,
            ),
            ScreenAction::Pop
        );
        Screen::init(
            &mut screen,
            &ScreenCtx::new(ScaledResolution::new(640, 480, 1, false), 0.0),
        );
        assert_eq!(screen.width, 640);
    }

    #[test]
    fn open_link_uses_vanillas_three_button_geometry_and_distinct_outcomes() {
        let make = || {
            let mut screen = ConfirmOpenLinkScreen::new(
                "Open?",
                "https://example.invalid",
                "Open",
                "Copy",
                "Cancel",
                "Warning",
                7,
            );
            screen.resize(854, 480);
            screen
        };
        let y = 480 / 6 + 97;
        let mut open = make();
        assert_eq!(open.buttons[0].rect.x, 854 / 2 - 155);
        assert_eq!(open.buttons[1].rect.x, 854 / 2 - 50);
        assert_eq!(open.buttons[2].rect.x, 854 / 2 + 55);
        assert!(open.buttons.iter().all(|button| button.rect.width == 100));
        assert_eq!(
            open.click(854 / 2 - 154, y, 0),
            Some((OpenLinkChoice::Open, 7))
        );

        let mut copy = make();
        assert_eq!(
            copy.click(854 / 2 - 49, y, 0),
            Some((OpenLinkChoice::Copy, 7))
        );
        assert_eq!(copy.link(), "https://example.invalid");

        let mut cancel = make();
        assert_eq!(
            cancel.click(854 / 2 + 56, y, 0),
            Some((OpenLinkChoice::Cancel, 7))
        );
    }

    /// A server's link that colors part of itself black (`§0`) or
    /// obfuscates it (`§k`) is drawn with those codes inert, so the whole destination shows;
    /// Open and Copy still use the link exactly as it was sent.
    #[test]
    fn open_link_dialog_draws_formatting_codes_in_the_link_inert() {
        let link = "https://good.example\u{a7}0\u{a7}k@evil.example/";
        let mut screen =
            ConfirmOpenLinkScreen::new("Open?", link, "Open", "Copy", "Cancel", "Warning", 0);
        screen.resize(854, 480);
        assert_eq!(screen.link(), link, "Open and Copy use the link as sent");
        assert_eq!(
            link_as_shown(link),
            "https://good.example\u{fffd}0\u{fffd}k@evil.example/"
        );
        let plain = "https://example.invalid/a?b=c&d=%41#top";
        assert_eq!(
            link_as_shown(plain),
            plain,
            "a real link is drawn as before"
        );

        // Any pure-black vertex on the link's line (y 90, its shadow one pixel lower).
        let black_on_link_line = |draw: &DrawList| {
            draw.vertices().iter().any(|vertex| {
                (90.0..100.0).contains(&vertex.pos[1]) && vertex.color[..3] == [0.0, 0.0, 0.0]
            })
        };
        let mut font = crate::widget::test_support::uniform_font();
        // Control: drawn as vanilla's dialog draws it, the codes are live.
        let mut raw = DrawList::new();
        draw_centered_string(&mut raw, &mut font, link, 854 / 2, 90, TITLE_COLOR);
        assert!(
            black_on_link_line(&raw),
            "the raw link turns part of itself black"
        );
        let mut dialog = DrawList::new();
        screen.draw_dialog(&mut dialog, &mut font, (0, 0));
        assert!(
            !black_on_link_line(&dialog),
            "the dialog draws every part of the link visibly"
        );
    }

    #[test]
    fn open_link_warning_can_be_disabled_without_changing_the_link() {
        let mut screen = ConfirmOpenLinkScreen::new(
            "Open?",
            "https://example.invalid",
            "Open",
            "Copy",
            "Cancel",
            "Warning",
            0,
        );
        assert!(screen.show_security_warning);
        screen.disable_security_warning();
        assert!(!screen.show_security_warning);
        assert_eq!(screen.link(), "https://example.invalid");
    }

    #[test]
    fn open_link_warning_is_composited_after_buttons_like_vanilla() {
        let mut screen = ConfirmOpenLinkScreen::new(
            "Open?",
            "https://example.invalid",
            "Open",
            "Copy",
            "Cancel",
            "WARN",
            0,
        );
        // At this short height the warning overlaps the button row, making the submission order
        // visible rather than merely an internal command-stream detail.
        screen.resize(854, 60);
        let mut draw = DrawList::new();
        let mut font = crate::widget::test_support::uniform_font();
        screen.draw_dialog(&mut draw, &mut font, (0, 0));

        let widgets = crate::widget::UiTexture::gui(crate::widget::WIDGETS_TEXTURE);
        let first_button_vertex = draw
            .batches()
            .iter()
            .find(|batch| batch.texture == widgets)
            .expect("GuiButton artwork")
            .range
            .start;
        let warning_vertex = draw
            .vertices()
            .iter()
            .position(|vertex| vertex.pos[1] == 110.0)
            .expect("security-warning glyph");

        // GuiConfirmOpenLink calls super.drawScreen first, then draws this warning.  The order is
        // observable when a compact layout makes the warning overlap button artwork.
        assert!(
            first_button_vertex < warning_vertex,
            "button vertex {first_button_vertex} must precede warning vertex {warning_vertex}"
        );
    }
}
