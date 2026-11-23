//! The chat history at its 5000-line maximum, with the real
//! vanilla font (minigame-server messages: rank chat, coin lines, long lines that wrap, blank
//! spacers, kills).
//!
//! * Adding a message to a full history is cheap (one wrap plus O(1) list work; the lists are
//!   `VecDeque`s, where the old `Vec::insert(0, ..)` moved every older line).
//! * A re-wrap (the chat width, height, scale or opacity slider, which vanilla re-wraps on every
//!   drag event) stays fast at 5000 lines.
//! * With chat closed, a full history draws exactly what a short one draws: only visible rows.
//!
//! Timings print as provisional (the test profile is barely
//! optimised): the bounds below are generous. Skips without the vanilla jar.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mc_assets::{Language, ResourceManager, ZipResourcePack};
use mc_ui::font::component::ChatComponent;
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::chat::{ChatLayout, ChatLog};
use mc_ui::widget::{DrawList, ScaledResolution};

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn font() -> Option<FontRenderer> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        eprintln!("SKIP: 1.8.9.jar not found");
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(FontRenderer::load(&rm, false, false).expect("the vanilla font loads"))
}

fn message(i: usize) -> ChatComponent {
    match i % 5 {
        0 => ChatComponent::text(format!(
            "\u{a7}7[\u{a7}b\u{a7}lMVP\u{a7}c+\u{a7}7] \u{a7}bPlayer{i}\u{a7}f: gg wp, that was close"
        )),
        1 => ChatComponent::text(format!("\u{a7}6+{} coins! (Win)", i % 97)),
        2 => ChatComponent::text(format!(
            "\u{a7}e\u{a7}lCLICK HERE \u{a7}r\u{a7}eto play again! {}",
            "Rewards: winning, kills, assists, survival bonus. ".repeat(1 + i % 3)
        )),
        3 => ChatComponent::text(""),
        _ => ChatComponent::text(format!(
            "\u{a7}c{i}Bot \u{a7}7was killed by \u{a7}a{i}Hunter\u{a7}7. \u{a7}6\u{a7}lFINAL KILL!"
        )),
    }
}

#[test]
fn a_full_five_thousand_line_history_stays_cheap() {
    let Some(font) = font() else { return };
    let lang = Language::new("en_US");
    let layout = ChatLayout::default();
    let mut log = ChatLog::new();
    log.set_max_lines(5000);
    let fill = Instant::now();
    for i in 0..6000 {
        log.add_message(message(i), 0, 0, &layout, &font, &lang, false);
    }
    let fill_ms = fill.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(log.lines().len(), 5000);

    // A burst like a game's end screen, into the full history.
    let burst = Instant::now();
    for i in 0..200 {
        log.add_message(message(i), 0, 0, &layout, &font, &lang, false);
    }
    let per_message_us = burst.elapsed().as_secs_f64() * 1e6 / 200.0;

    // A width slider drag: one re-wrap per drag event, alternating widths.
    let mut rewraps = Vec::new();
    for width in [0.6f32, 1.0, 0.45, 1.0, 0.8] {
        let narrow = ChatLayout {
            width_scale: width,
            ..ChatLayout::default()
        };
        let t = Instant::now();
        log.refresh(&narrow, &font, &lang);
        rewraps.push(t.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(log.lines().len(), 5000);
    }
    let worst_rewrap = rewraps.iter().copied().fold(0.0f64, f64::max);

    // Drawing (chat closed and open) touches only the visible rows.
    let ctx = ScreenCtx::new(ScaledResolution::new(1920, 1080, 3, false), 0.0);
    let mut draw_font = font;
    let draw = Instant::now();
    for _ in 0..100 {
        let mut d = DrawList::new();
        log.draw(&mut d, &mut draw_font, &ctx, 10, &layout, false);
        log.draw(&mut d, &mut draw_font, &ctx, 10, &layout, true);
    }
    let draw_us = draw.elapsed().as_secs_f64() * 1e6 / 100.0;

    eprintln!(
        "B1 history at 5000 lines (provisional): filled 6000 messages in {fill_ms:.0} ms, \
         {per_message_us:.1} us per message when full, re-wraps {rewraps:.1?} ms (worst \
         {worst_rewrap:.1} ms), a closed+open draw {draw_us:.1} us"
    );
    assert!(
        per_message_us < 2000.0,
        "{per_message_us:.1} us per message"
    );
    assert!(worst_rewrap < 500.0, "re-wrap {worst_rewrap:.1} ms");
}

/// With chat closed a 5000-line history draws exactly what a log of only its newest messages
/// draws (the vanilla font this time).
#[test]
fn only_the_visible_rows_are_drawn() {
    let Some(mut font) = font() else { return };
    let lang = Language::new("en_US");
    let layout = ChatLayout::default();
    let mut long = ChatLog::new();
    long.set_max_lines(5000);
    let mut short = ChatLog::new();
    for i in 0..6000 {
        long.add_message(message(i), 0, 100, &layout, &font, &lang, false);
    }
    for i in 5900..6000 {
        short.add_message(message(i), 0, 100, &layout, &font, &lang, false);
    }
    let ctx = ScreenCtx::new(ScaledResolution::new(854, 480, 2, false), 0.0);
    let mut a = DrawList::new();
    long.draw(&mut a, &mut font, &ctx, 110, &layout, false);
    let mut b = DrawList::new();
    short.draw(&mut b, &mut font, &ctx, 110, &layout, false);
    assert!(!a.vertices().is_empty());
    assert_eq!(a, b);
}
