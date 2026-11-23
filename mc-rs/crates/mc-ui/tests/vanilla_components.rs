//! Chat component tests against the real `en_US.lang` and real protocol-47 chat payloads.
//!
//! The JSON in these tests is the shape a vanilla 1.8.9 server actually puts on the wire for
//! `S02PacketChat` — a `chat.type.text` translation whose `with` array holds a styled player
//! name carrying `clickEvent`/`hoverEvent`/`insertion`, plus the message body.

use std::path::{Path, PathBuf};

use mc_assets::{Language, ResourceManager, ZipResourcePack};
use mc_ui::font::component::{ClickAction, HoverAction};
use mc_ui::font::{ChatComponent, FontRenderer, TextFormat};

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn manager() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

fn lang() -> Option<Language> {
    let rm = manager()?;
    let l = Language::load(&rm, &["en_US"]);
    (l.len() > 1000).then_some(l)
}

macro_rules! require {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => {
                eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found; set MC_JARS_DIR");
                return;
            }
        }
    };
}

#[test]
fn a_real_chat_line_resolves_through_the_real_language_table() {
    let lang = require!(lang());
    assert_eq!(lang.tr("chat.type.text"), "<%s> %s");

    let json = r#"{"translate":"chat.type.text","with":[{"text":"Steve"},"hello world"]}"#;
    let c = ChatComponent::from_json(json).expect("parse");
    assert_eq!(c.unformatted_text(&lang), "<Steve> hello world");
    assert_eq!(
        c.formatted_text(&lang),
        "<\u{a7}rSteve\u{a7}r> \u{a7}rhello world\u{a7}r"
    );
}

#[test]
fn positional_arguments_work_against_real_death_messages() {
    let lang = require!(lang());
    assert_eq!(lang.tr("death.attack.player"), "%1$s was slain by %2$s");

    let c =
        ChatComponent::from_json(r#"{"translate":"death.attack.player","with":["Alex","Steve"]}"#)
            .expect("parse");
    assert_eq!(c.unformatted_text(&lang), "Alex was slain by Steve");

    // Three positional arguments, out of order in the format string is legal too.
    assert_eq!(
        lang.tr("death.fell.assist.item"),
        "%1$s was doomed to fall by %2$s using %3$s"
    );
    let c =
        ChatComponent::from_json(r#"{"translate":"death.fell.assist.item","with":["A","B","C"]}"#)
            .expect("parse");
    assert_eq!(
        c.unformatted_text(&lang),
        "A was doomed to fall by B using C"
    );
}

#[test]
fn a_player_name_carries_its_click_hover_and_insertion_through_the_translation() {
    let lang = require!(lang());
    // This is what a vanilla server sends when a player with a display name speaks.
    let json = r#"{
        "translate":"chat.type.text",
        "with":[
            {"text":"Steve","color":"yellow","bold":true,
             "insertion":"Steve",
             "clickEvent":{"action":"suggest_command","value":"/msg Steve "},
             "hoverEvent":{"action":"show_entity","value":{"text":"{name:Steve}"}}},
            "hi"
        ],
        "color":"gray"
    }"#;
    let c = ChatComponent::from_json(json).expect("parse");
    let spans = c.spans(&lang);
    let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, vec!["<", "Steve", "> ", "hi"]);

    // The name span carries the events; the surrounding literals inherit only the colour.
    let name = &spans[1];
    assert_eq!(name.style.color, Some(TextFormat::Yellow));
    assert!(name.style.bold);
    assert_eq!(
        name.style.click_event.as_ref().map(|c| c.action),
        Some(ClickAction::SuggestCommand)
    );
    assert_eq!(
        name.style.click_event.as_ref().map(|c| c.value.as_str()),
        Some("/msg Steve ")
    );
    assert_eq!(
        name.style.hover_event.as_ref().map(|h| h.action),
        Some(HoverAction::ShowEntity)
    );
    assert_eq!(name.style.insertion.as_deref(), Some("Steve"));

    assert_eq!(spans[0].style.color, Some(TextFormat::Gray));
    assert!(spans[0].style.click_event.is_none());
    assert!(!spans[0].style.bold);
}

/// Clicking a chat line has to resolve to the *span* under the cursor, which means summing
/// rendered widths — so it is a font question as much as a component question.
#[test]
fn hit_testing_a_chat_line_picks_the_span_under_the_cursor() {
    let lang = require!(lang());
    let rm = manager().expect("manager");
    let font = FontRenderer::load(&rm, false, false).expect("font");

    let json = r#"{"translate":"chat.type.text","with":[{"text":"Steve","clickEvent":{"action":"suggest_command","value":"/msg Steve "}},"hi"]}"#;
    let c = ChatComponent::from_json(json).expect("parse");
    let spans = c.spans(&lang);

    // Accumulate widths the way GuiNewChat.getChatComponent does.
    let pick = |x: i32| -> Option<usize> {
        let mut acc = 0;
        for (i, s) in spans.iter().enumerate() {
            acc += font.string_width(&s.text);
            if acc > x {
                return Some(i);
            }
        }
        None
    };

    let w_open = font.string_width("<");
    let w_name = font.string_width("Steve");
    assert_eq!(pick(0), Some(0), "the '<' literal");
    assert_eq!(pick(w_open), Some(1), "the name starts right after it");
    assert_eq!(pick(w_open + w_name - 1), Some(1));
    assert_eq!(pick(w_open + w_name), Some(2), "'> ' comes next");
    assert_eq!(pick(10_000), None, "past the end of the line");

    // The clickable span is the one carrying the event.
    let hit = pick(w_open + 2).expect("hit");
    assert_eq!(
        spans[hit]
            .style
            .click_event
            .as_ref()
            .map(|c| c.value.as_str()),
        Some("/msg Steve ")
    );
}

#[test]
fn a_formatted_component_measures_and_wraps_like_a_flat_string() {
    let lang = require!(lang());
    let rm = manager().expect("manager");
    let font = FontRenderer::load(&rm, false, false).expect("font");

    let c = ChatComponent::from_json(
        r#"{"translate":"multiplayer.player.joined","with":[{"text":"Steve","color":"yellow"}]}"#,
    )
    .expect("parse");
    assert_eq!(c.unformatted_text(&lang), "Steve joined the game");

    let flat = c.formatted_text(&lang);
    // The formatting codes are free, so the formatted and unformatted forms measure the same.
    assert_eq!(
        font.string_width(&flat),
        font.string_width(&c.unformatted_text(&lang))
    );
    // And it wraps into the vanilla chat width (320 scaled pixels) without losing the colour.
    let lines = font.wrap_to_lines(&flat, 320);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains('\u{a7}'));
}

#[test]
fn a_disconnect_reason_round_trips_through_json() {
    let lang = require!(lang());
    let json = r#"{"text":"","extra":[{"text":"Kicked: ","color":"red"},{"text":"cheating","bold":true}]}"#;
    let c = ChatComponent::from_json(json).expect("parse");
    assert_eq!(c.unformatted_text(&lang), "Kicked: cheating");
    // Siblings inherit from their *parent*, never from the sibling before them, so "cheating"
    // is bold but not red. The leading §r is the empty root component's own run.
    assert_eq!(
        c.formatted_text(&lang),
        "\u{a7}r\u{a7}cKicked: \u{a7}r\u{a7}lcheating\u{a7}r"
    );

    let round = ChatComponent::from_json(&c.to_json()).expect("reparse");
    assert_eq!(round, c);
    assert_eq!(round.unformatted_text(&lang), c.unformatted_text(&lang));
}

#[test]
fn a_hostile_payload_is_an_error_not_a_crash() {
    let lang = require!(lang());
    // Unterminated, wrong types, unknown shapes, absurd nesting: all must be Err, never panic.
    for bad in [
        "",
        "{",
        "[]",
        "{}",
        "null",
        r#"{"extra":["x"]}"#,
        r#"{"text":"a","extra":[]}"#,
        r#"{"score":{"name":"n"}}"#,
        r#"{"translate":"x","with":{}}"#,
    ] {
        let _ = ChatComponent::from_json(bad);
    }
    assert!(ChatComponent::from_json("{}").is_err());

    // 5000 levels of nesting. This used to assert `is_err()`, which pinned the bug: the error
    // reaches `self.disconnect()`, so it meant one `S02PacketChat` could kick the player, and
    // vanilla renders a depth-1000 tree without complaint. What actually has to hold is that
    // mc-rs neither crashes (the recursive `Drop` on `ChatComponent` makes a stack overflow an
    // abort, not an unwind) nor errors. See `font::component::MAX_DEPTH`.
    let deep = format!(
        "{}{}{}",
        r#"{"text":"","extra":["#.repeat(5000),
        r#""x""#,
        "]}".repeat(5000)
    );
    let parsed = ChatComponent::from_json(&deep).expect("must not disconnect the player");
    // ... and the surviving prefix is renderable, i.e. the fallback did not just give up.
    let _ = parsed.unformatted_text(&lang);
}
