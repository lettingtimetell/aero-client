//! 1.7.10's book page fit and plain pages, with 1.8.9 as the negative control.

use mc_nbt::{Nbt, NbtCompound};
use mc_protocol::item::ItemStack;

use super::*;
use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::book::BookAction;
use crate::widget::keys::Modifiers;
use crate::widget::test_support::uniform_font;
use crate::widget::{DrawList, MemoryClipboard, ScaledResolution};

const ENTER: i32 = 28;

fn writable(version: GameVersion) -> BookScreen {
    let mut screen = BookScreen::new(
        ItemStack {
            item_id: 386,
            count: 1,
            damage: 0,
            nbt: None,
        },
        "Player",
        true,
    );
    screen.set_game_version(version);
    screen
}

fn enter(screen: &mut BookScreen, font: &FontRenderer) {
    screen.key_typed(
        None,
        ENTER,
        Modifiers::NONE,
        &MemoryClipboard::default(),
        font,
    );
}

/// The number of lines a page grows to under repeated Enter presses.
fn lines_after_enters(version: GameVersion) -> usize {
    let font = uniform_font();
    let mut screen = writable(version);
    for _ in 0..20 {
        enter(&mut screen, &font);
    }
    screen.current_page().matches('\n').count() + 1
}

#[test]
fn a_1_7_10_page_takes_thirteen_lines_and_1_8_9_fourteen() {
    // With the caret's line: 13 * 9 = 117 <= 118 (14 * 9 = 126 <= 128 in 1.8.9).
    assert_eq!(lines_after_enters(GameVersion::V1_7_10), 13);
    assert_eq!(lines_after_enters(GameVersion::V1_8_9), 14);
    assert_eq!(MAX_PAGE_HEIGHT_V5, 118);
}

/// A signed book whose one page is the JSON of a clickable `/give` command.
fn signed_with_click_page(version: GameVersion) -> BookScreen {
    let page =
        r#"{"text":"click me","clickEvent":{"action":"run_command","value":"/give @p diamond"}}"#;
    let mut tag = NbtCompound::new();
    tag.insert("title", Nbt::String("T".into()));
    tag.insert("author", Nbt::String("A".into()));
    tag.insert("pages", Nbt::list(vec![Nbt::String(page.into())]));
    let mut screen = BookScreen::new(
        ItemStack {
            item_id: 387,
            count: 1,
            damage: 0,
            nbt: Some(tag),
        },
        "Player",
        false,
    );
    screen.set_game_version(version);
    screen
}

#[test]
fn a_1_7_10_page_is_plain_text_with_nothing_to_click() {
    let font = uniform_font();
    let lang = mc_assets::Language::new("en_US");
    let ctx = ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0);
    for (version, runs) in [(GameVersion::V1_8_9, true), (GameVersion::V1_7_10, false)] {
        let mut screen = signed_with_click_page(version);
        screen.init(&ctx, &lang);
        assert_eq!(
            screen.parsed_lines(&font, &lang).is_some(),
            runs,
            "{version}"
        );
        // Click the first glyph of the page's first line.
        let x = (ctx.width - 192) / 2 + 36 + 1;
        screen.mouse_clicked(x, 34 + 1, 0, &font, &lang);
        let ran = screen
            .take_actions()
            .iter()
            .any(|action| matches!(action, BookAction::RunCommand(_)));
        assert_eq!(ran, runs, "{version}");
    }
}

/// A signed book with a page but no title or author: 1.8.9's validity check rejects it, and
/// 1.7.10 has none (java17 `GuiScreenBook.drawScreen` draws the raw page).
#[test]
fn a_1_7_10_book_without_a_title_still_shows_its_page() {
    let mut font = uniform_font();
    let lang = mc_assets::Language::new("en_US");
    let ctx = ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0);
    let mut tag = NbtCompound::new();
    tag.insert("pages", Nbt::list(vec![Nbt::String("hello".into())]));
    // Text on the page, by colour: `§4` dark red (the invalid-book line) and black (a page).
    let page_text = |d: &DrawList, rgb: [f32; 3]| {
        d.vertices().iter().any(|vertex| {
            vertex.pos[1] >= super::super::PAGE_TOP as f32
                && vertex.color[..3]
                    .iter()
                    .zip(rgb)
                    .all(|(have, want)| (have - want).abs() < 1e-3)
        })
    };
    let dark_red = [170.0 / 255.0, 0.0, 0.0];
    for (version, invalid) in [(GameVersion::V1_8_9, true), (GameVersion::V1_7_10, false)] {
        let mut screen = BookScreen::new(
            ItemStack {
                item_id: 387,
                count: 1,
                damage: 0,
                nbt: Some(tag.clone()),
            },
            "Player",
            false,
        );
        screen.set_game_version(version);
        screen.init(&ctx, &lang);
        let mut d = DrawList::new();
        screen.draw(&mut d, &mut font, &ctx, (0, 0), &lang);
        assert_eq!(page_text(&d, dark_red), invalid, "{version}");
        assert_eq!(page_text(&d, [0.0; 3]), !invalid, "{version}");
    }
}
