//! 1.7.10 chat clicks against java17 `GuiChat.mouseClicked`, with 1.8.9's rules as the negative
//! control.

use mc_assets::Language;

use super::*;
use crate::font::component::{ChatComponent, ChatStyle, ResolvedStyle};
use crate::font::format::TextFormat;
use crate::widget::test_support::uniform_font;
use crate::widget::ScaledResolution;

fn ctx() -> ScreenCtx {
    ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
}

fn click(action: ClickAction, value: &str) -> ClickEvent {
    ClickEvent {
        action,
        value: value.to_owned(),
    }
}

/// A chat log holding one line: `prefix` (no event) then `name` in red with `event`.
fn log_with(event: ClickEvent, font: &FontRenderer) -> (ChatLog, ChatLayout) {
    let layout = ChatLayout::default();
    let mut log = ChatLog::new();
    let name = ChatComponent::text("Steve").with_style(ChatStyle {
        color: Some(TextFormat::Red),
        click_event: Some(event),
        ..ChatStyle::default()
    });
    log.add_message(
        ChatComponent::text("<a> ").with_extra(name),
        0,
        0,
        &layout,
        font,
        &Language::new("en_US"),
        true,
    );
    (log, layout)
}

#[test]
fn the_piece_under_the_pointer_is_its_formatting_prefix_and_text() {
    let font = uniform_font();
    let (log, layout) = log_with(click(ClickAction::RunCommand, "/msg Steve"), &font);
    let y = ctx().height - 30;
    // "<a> " has no click event.
    assert!(log
        .clicked_piece_v5(3 + 5, y, &ctx(), &layout, &font)
        .is_none());
    let (text, event) = log
        .clicked_piece_v5(3 + 40, y, &ctx(), &layout, &font)
        .expect("the name");
    assert_eq!(text, "\u{a7}cSteve");
    assert_eq!(event.value, "/msg Steve");
}

#[test]
fn change_page_is_not_a_1_7_10_click_event() {
    let font = uniform_font();
    let (log, layout) = log_with(click(ClickAction::ChangePage, "2"), &font);
    let y = ctx().height - 30;
    assert!(log
        .clicked_piece_v5(3 + 40, y, &ctx(), &layout, &font)
        .is_none());
}

#[test]
fn with_chat_links_off_no_click_event_acts() {
    let font = uniform_font();
    let run = click(ClickAction::RunCommand, "/spawn");
    let mut screen = ChatScreen::new("", 0, &font);
    for shift in [false, true] {
        assert_eq!(
            screen.component_clicked_v5(Some(("/spawn", &run)), shift, false, &font),
            None
        );
    }
    assert_eq!(screen.input.text(), "");
    // 1.8.9 (negative control): only `open_url` checks Chat Links.
    let style = ResolvedStyle {
        click_event: Some(run.clone()),
        ..ResolvedStyle::default()
    };
    assert_eq!(
        screen.component_clicked(&style, false, false, &font),
        Some(Action::RunCommand("/spawn".into()))
    );
}

#[test]
fn with_chat_links_on_each_action_is_consumed() {
    let font = uniform_font();
    let mut screen = ChatScreen::new("", 0, &font);
    let cases = [
        (ClickAction::RunCommand, Action::RunCommand("/spawn".into())),
        (
            ClickAction::SuggestCommand,
            Action::SuggestCommand("/spawn".into()),
        ),
        (ClickAction::OpenUrl, Action::OpenUrl("/spawn".into())),
        (ClickAction::TwitchUserInfo, Action::None),
    ];
    for (action, expected) in cases {
        let event = click(action, "/spawn");
        assert_eq!(
            screen.component_clicked_v5(Some(("x", &event)), false, true, &font),
            Some(expected)
        );
    }
    // No event under the pointer: the input field gets the click.
    assert_eq!(screen.component_clicked_v5(None, false, true, &font), None);
}

#[test]
fn shift_inserts_the_pieces_text_and_consumes_the_click() {
    let font = uniform_font();
    let event = click(ClickAction::RunCommand, "/must_not_run");
    let mut screen = ChatScreen::new(">", 0, &font);
    assert_eq!(
        screen.component_clicked_v5(Some(("\u{a7}cSteve", &event)), true, true, &font),
        Some(Action::None)
    );
    // `GuiTextField.writeText` drops the section sign and keeps the code letter.
    assert_eq!(screen.input.text(), ">cSteve");
    // 1.8.9 inserts the `insertion` only, and lets the click through.
    let mut screen = ChatScreen::new(">", 0, &font);
    let style = ResolvedStyle {
        click_event: Some(event),
        ..ResolvedStyle::default()
    };
    assert_eq!(screen.component_clicked(&style, true, true, &font), None);
    assert_eq!(screen.input.text(), ">");
}
