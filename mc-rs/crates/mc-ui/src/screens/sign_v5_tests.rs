//! 1.7.10's sign line limit and command-block controls, with 1.8.9 as the negative control.

use mc_math::BlockPos;

use super::*;
use crate::font::component::ChatComponent;
use crate::screen_stack::ScreenCtx;
use crate::screens::sign::{CommandBlockTarget, SignAction, MAX_LINE_WIDTH};
use crate::widget::keys;
use crate::widget::test_support::uniform_font;
use crate::widget::ScaledResolution;

fn editor(version: GameVersion) -> EditSignScreen {
    let mut screen = EditSignScreen::new(BlockPos::new(1, 64, -2), Default::default());
    screen.set_game_version(version);
    screen
}

/// Type `text` on the current line, then press Escape (Done) and return the sent lines.
fn type_and_finish(screen: &mut EditSignScreen, text: &str) -> [String; 4] {
    let font = uniform_font();
    for character in text.chars() {
        screen.key_typed(Some(character), 0, &font);
    }
    screen.key_typed(None, keys::ESCAPE, &font);
    screen
        .take_actions()
        .into_iter()
        .find_map(|action| match action {
            SignAction::SendUpdateSign { lines, .. } => Some(lines),
            _ => None,
        })
        .expect("Done sends the lines")
}

fn plain(json: &str) -> String {
    ChatComponent::from_json(json)
        .expect("component JSON")
        .to_string()
}

#[test]
fn twenty_narrow_characters_stop_at_fifteen_in_1_7_10() {
    let lines = type_and_finish(&mut editor(GameVersion::V1_7_10), &"i".repeat(20));
    assert_eq!(plain(&lines[0]), "i".repeat(15));
    // 1.8.9 measures pixels: this test font advances 5 per glyph, so 18 fit in 90.
    let lines = type_and_finish(&mut editor(GameVersion::V1_8_9), &"i".repeat(20));
    assert_eq!(plain(&lines[0]), "i".repeat(18));
    assert_eq!(MAX_LINE_WIDTH, 90);
}

#[test]
fn fifteen_characters_fit_whatever_their_width_and_backspace_makes_room() {
    let font = uniform_font();
    let mut screen = editor(GameVersion::V1_7_10);
    for _ in 0..16 {
        screen.key_typed(Some('W'), 0, &font);
    }
    assert_eq!(screen.line(0).unwrap(), "W".repeat(15));
    screen.key_typed(None, keys::BACKSPACE, &font);
    screen.key_typed(Some('x'), 0, &font);
    assert_eq!(screen.line(0).unwrap(), format!("{}x", "W".repeat(14)));
    assert!(super::line_accepts_v5(&"a".repeat(15)));
    assert!(!super::line_accepts_v5(&"a".repeat(16)));
}

/// The 1.7.10 server counts UTF-16 units: a character outside the Basic Multilingual Plane is
/// two, so it fits only while the line has two units of room.
#[test]
fn a_1_7_10_sign_line_counts_utf16_units() {
    let font = uniform_font();
    let wide = '\u{1F600}';
    for (typed, expected) in [(13, 15), (14, 14)] {
        let mut screen = editor(GameVersion::V1_7_10);
        for _ in 0..typed {
            screen.key_typed(Some('a'), 0, &font);
        }
        screen.key_typed(Some(wide), 0, &font);
        let line = screen.line(0).unwrap();
        assert_eq!(line.encode_utf16().count(), expected, "{typed} typed first");
        assert_eq!(line.ends_with(wide), typed == 13, "{typed} typed first");
    }
}

#[test]
fn a_1_7_10_command_block_editor_has_no_track_output_toggle() {
    let font = uniform_font();
    let ctx = ScreenCtx::new(ScaledResolution::new(427, 240, 1, false), 0.0);
    let lang = mc_assets::Language::new("en_US");
    let target = CommandBlockTarget::Block {
        position: BlockPos::new(0, 70, 0),
    };
    for (version, shown) in [(GameVersion::V1_8_9, true), (GameVersion::V1_7_10, false)] {
        let mut screen = CommandBlockScreen::new(target, "say hi", true, None);
        screen.set_game_version(version);
        screen.init(&ctx, &lang, &font);
        let toggle = screen.track_output_button().rect;
        assert_eq!(screen.track_output_button().visible, shown, "{version}");
        // A click on it toggles only where it is shown.
        screen.mouse_clicked(toggle.x + 1, toggle.y + 1, 0, &font);
        assert_eq!(screen.track_output(), !shown, "{version}");
    }
}
