//! Minecraft 1.7.10's chat clicks. A child of `chat`, for the chat log's lines and the screen's
//! input field.
//!
//! java17 `GuiChat.mouseClicked` (`client/gui/GuiChat.java:167-252`) acts on a chat component
//! only for a left click with Chat Links on, and only when the piece under the pointer has a
//! click event; then it consumes the click:
//!
//! - with Shift, it writes the piece's own text (`getUnformattedTextForChat`, which for a wrapped
//!   piece is its formatting prefix plus its text, as `GuiNewChat.setChatLine` builds it) into
//!   the input field; there is no `insertion` in 1.7.10;
//! - otherwise it opens the URL, suggests the command, or runs it with `submitChatMessage`, which
//!   also records it in the sent history (the caller's half).
//!
//! 1.8.9 instead gates only `open_url` on Chat Links, runs commands without recording them and
//! inserts a component's `insertion` (java18 `GuiScreen.handleComponentClick`).
//! `change_page` does not exist in 1.7.10, so a piece carrying it acts as if it had no event. A
//! URL with no scheme makes vanilla 1.7.10 throw; here it goes to the client's usual URL
//! handling instead.

use super::{Action, ChatLayout, ChatLog, ChatScreen};
use crate::font::component::{ClickAction, ClickEvent};
use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;

impl ChatLog {
    /// java17 `GuiNewChat.getChatComponent`: the wrapped piece under a GUI point, as its text and
    /// click event, when it has a click event 1.7.10 knows.
    #[must_use]
    pub fn clicked_piece_v5(
        &self,
        gui_x: i32,
        gui_y: i32,
        ctx: &ScreenCtx,
        layout: &ChatLayout,
        font: &FontRenderer,
    ) -> Option<(&str, &ClickEvent)> {
        let line = self.line_at(gui_x, gui_y, ctx, layout, true)?;
        let x = mc_math::floor_f32((gui_x - 3) as f32 / layout.scale);
        let mut right = 0;
        let span = line.spans.iter().find(|span| {
            right += font.string_width(&span.text);
            right > x
        })?;
        let click = span.style.click_event.as_ref()?;
        (click.action != ClickAction::ChangePage).then_some((span.text.as_str(), click))
    }
}

impl ChatScreen {
    /// The component half of java17 `GuiChat.mouseClicked` for a left click: `Some(action)` when
    /// the click is consumed (`Action::None` after a Shift insertion), `None` when it goes on to
    /// the input field. `piece` is [`ChatLog::clicked_piece_v5`]'s result.
    pub fn component_clicked_v5(
        &mut self,
        piece: Option<(&str, &ClickEvent)>,
        shift: bool,
        chat_links: bool,
        font: &FontRenderer,
    ) -> Option<Action> {
        if !chat_links {
            return None;
        }
        let (text, click) = piece?;
        if shift {
            self.ensure_input_layout(font);
            self.input.write_text(text, font);
            return Some(Action::None);
        }
        Some(match click.action {
            ClickAction::OpenUrl => Action::OpenUrl(click.value.clone()),
            ClickAction::RunCommand => Action::RunCommand(click.value.clone()),
            ClickAction::SuggestCommand => Action::SuggestCommand(click.value.clone()),
            // None of these comes from a 1.7.10 server: java17's chat style parser drops
            // `open_file` and `twitch_user_info` (not allowed in chat) and `change_page` (an
            // action 1.7.10 does not have), and `CopyImage` is this client's own.
            ClickAction::OpenFile
            | ClickAction::CopyImage
            | ClickAction::TwitchUserInfo
            | ClickAction::ChangePage => Action::None,
        })
    }
}

#[cfg(test)]
#[path = "chat_v5_tests.rs"]
mod tests;
