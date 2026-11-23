//! Multiplayer sleep chat.
//!
//! java: `client/gui/GuiSleepMP.java`
//!
//! Vanilla subclasses [`GuiChat`](chat::ChatScreen), but changes the two inputs which would
//! normally close chat. Escape requests `STOP_SLEEPING`; Enter sends and clears the message
//! while leaving this screen open. Keeping that distinction in [`SleepAction`] prevents a caller
//! from accidentally treating Escape as the ordinary chat "Back" action and leaving the player
//! trapped in bed server-side.

use mc_assets::Language;

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::chat::{java_trim, ChatLayout, ChatLog, ChatScreen};
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::keys::{self, Clipboard, Modifiers};
use crate::widget::Button;

const LEAVE_BED_BUTTON_ID: i32 = 1;

/// An intent produced by [`SleepScreen`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SleepAction {
    /// The input was consumed without an application-level effect.
    None,
    /// Send `C0BPacketEntityAction(STOP_SLEEPING)` while keeping sleep state authoritative until
    /// the server updates it.
    StopSleeping,
    /// Send chat without closing this screen. Unlike `GuiChat`, `GuiSleepMP` calls
    /// `EntityPlayerSP.sendChatMessage` directly, so vanilla does not add this message to the
    /// local sent-history list either.
    SendChat(String),
    /// A normal chat intent (send chat, tab-complete, component click, and so on).
    Chat(Action),
}

impl SleepAction {
    /// Wrap a chat action without making callers special-case its no-op variant.
    fn from_chat(action: Action) -> Self {
        if action.is_none() {
            Self::None
        } else {
            Self::Chat(action)
        }
    }
}

/// `GuiSleepMP`: chat input plus the sole multiplayer "Leave Bed" control.
#[derive(Debug)]
pub struct SleepScreen {
    /// The inherited `GuiChat` state. Public so the application can reuse its component-click,
    /// hover and tab-completion plumbing exactly as it does for ordinary chat.
    pub chat: ChatScreen,
    leave_bed: Button,
}

impl SleepScreen {
    /// Construct the screen. `sent_count` is the current chat-history length, matching
    /// `GuiChat.initGui`'s initial history cursor.
    #[must_use]
    pub fn new(sent_count: usize, font: &FontRenderer) -> Self {
        Self {
            chat: ChatScreen::new("", sent_count, font),
            // `init` installs the real translated label and position before first draw.
            leave_bed: Button::new(LEAVE_BED_BUTTON_ID, 0, 0, ""),
        }
    }

    /// Lay out the inherited chat field and vanilla's 200x20 leave-bed button.
    // java: GuiSleepMP.java:18-22
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        self.chat.init(ctx);
        self.leave_bed = Button::new(
            LEAVE_BED_BUTTON_ID,
            ctx.width / 2 - 100,
            ctx.height - 40,
            lang.tr("multiplayer.stopSleeping"),
        );
    }

    /// The leave-bed button, exposed read-only for layout inspection and accessibility adapters.
    #[must_use]
    pub fn leave_bed_button(&self) -> &Button {
        &self.leave_bed
    }

    /// Advance the inherited chat caret blink.
    pub fn tick(&mut self) {
        self.chat.tick();
    }

    /// Handle keyboard input with `GuiSleepMP.keyTyped` semantics.
    #[allow(clippy::too_many_arguments)]
    // java: GuiSleepMP.java:30-52
    pub fn key_typed(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        clipboard: &mut dyn Clipboard,
        font: &FontRenderer,
        log: &mut ChatLog,
        layout: &ChatLayout,
    ) -> SleepAction {
        match key {
            keys::ESCAPE => SleepAction::StopSleeping,
            keys::ENTER | keys::NUMPAD_ENTER => {
                let text = java_trim(&self.chat.input.text()).to_owned();
                // Java clears the field and resets scroll even when it contained only whitespace.
                self.chat.input.set_text("", font);
                log.reset_scroll();
                if text.is_empty() {
                    SleepAction::None
                } else {
                    SleepAction::SendChat(text)
                }
            }
            _ => SleepAction::from_chat(
                self.chat
                    .key_typed(typed, key, mods, clipboard, font, log, layout),
            ),
        }
    }

    /// Handle a mouse press. Only a left click activates vanilla buttons; all other presses are
    /// forwarded to the inherited chat input.
    // java: GuiSleepMP.java:57-61; GuiScreen.java:497-515
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        font: &FontRenderer,
    ) -> SleepAction {
        if button == 0 && self.leave_bed.mouse_pressed(x, y) {
            SleepAction::StopSleeping
        } else {
            SleepAction::from_chat(self.chat.mouse_clicked(x, y, button, font))
        }
    }

    /// Forward vanilla's wheel rule to the inherited chat log.
    pub fn mouse_wheel(&mut self, delta: f32, shift: bool, log: &mut ChatLog, layout: &ChatLayout) {
        self.chat.mouse_wheel(delta, shift, log, layout);
    }

    /// Supply a `S3APacketTabComplete` response to the inherited chat state.
    pub fn on_completions(&mut self, candidates: &[String], font: &FontRenderer) {
        self.chat.on_completions(candidates, font);
    }

    /// Reset chat scroll when the screen is removed, matching `GuiChat.onGuiClosed`.
    pub fn closed(&mut self, log: &mut ChatLog) {
        log.reset_scroll();
    }

    /// Draw the chat input and then the leave-bed button, matching superclass draw order.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
    ) {
        self.chat.draw(d, font, ctx);
        self.leave_bed.draw(d, font, mouse.0, mouse.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screens::clipboard::NullClipboard;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn screen(font: &FontRenderer) -> SleepScreen {
        let mut screen = SleepScreen::new(0, font);
        screen.init(&ctx(), &Language::new("en_US"));
        screen
    }

    #[test]
    fn leave_bed_button_uses_vanillas_id_size_and_coordinates() {
        let font = uniform_font();
        let screen = screen(&font);
        let button = screen.leave_bed_button();
        assert_eq!(button.id, 1);
        assert_eq!((button.rect.x, button.rect.y), (854 / 2 - 100, 480 - 40));
        assert_eq!((button.rect.width, button.rect.height), (200, 20));
    }

    #[test]
    fn escape_and_only_a_left_button_click_request_stop_sleeping() {
        let font = uniform_font();
        let mut screen = screen(&font);
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clipboard = NullClipboard;

        assert_eq!(
            screen.key_typed(
                None,
                keys::ESCAPE,
                Modifiers::NONE,
                &mut clipboard,
                &font,
                &mut log,
                &layout,
            ),
            SleepAction::StopSleeping
        );
        assert_eq!(
            screen.mouse_clicked(854 / 2, 480 - 30, 0, &font),
            SleepAction::StopSleeping
        );
        assert_eq!(
            screen.mouse_clicked(854 / 2, 480 - 30, 1, &font),
            SleepAction::None,
            "right click is a negative control for the button gate"
        );
        assert_eq!(
            screen.mouse_clicked(854 / 2, 480 - 41, 0, &font),
            SleepAction::None,
            "one pixel above is a negative control for the hit box"
        );
    }

    #[test]
    fn enter_sends_trimmed_chat_clears_input_and_keeps_sleep_open() {
        let font = uniform_font();
        let mut screen = screen(&font);
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clipboard = NullClipboard;
        screen.chat.input.set_text("  good night  ", &font);

        let action = screen.key_typed(
            None,
            keys::ENTER,
            Modifiers::NONE,
            &mut clipboard,
            &font,
            &mut log,
            &layout,
        );
        assert_eq!(action, SleepAction::SendChat("good night".to_owned()));
        assert_eq!(screen.chat.input.text(), "");
        assert!(
            log.sent_messages().is_empty(),
            "GuiSleepMP calls EntityPlayerSP directly instead of GuiScreen.sendChatMessage"
        );

        screen.chat.input.set_text("   ", &font);
        assert_eq!(
            screen.key_typed(
                None,
                keys::NUMPAD_ENTER,
                Modifiers::NONE,
                &mut clipboard,
                &font,
                &mut log,
                &layout,
            ),
            SleepAction::None,
            "blank messages neither send nor close the sleep screen"
        );
        assert!(log.sent_messages().is_empty());
    }

    #[test]
    fn non_exit_keys_reuse_chat_editing_and_tab_completion() {
        let font = uniform_font();
        let mut screen = screen(&font);
        let mut log = ChatLog::new();
        let layout = ChatLayout::default();
        let mut clipboard = NullClipboard;

        assert_eq!(
            screen.key_typed(
                Some('h'),
                35,
                Modifiers::NONE,
                &mut clipboard,
                &font,
                &mut log,
                &layout,
            ),
            SleepAction::None
        );
        assert_eq!(screen.chat.input.text(), "h");
        assert_eq!(
            screen.key_typed(
                None,
                keys::TAB,
                Modifiers::NONE,
                &mut clipboard,
                &font,
                &mut log,
                &layout,
            ),
            SleepAction::Chat(Action::RequestTabComplete("h".to_owned()))
        );
    }
}
