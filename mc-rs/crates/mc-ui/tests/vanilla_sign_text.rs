//! "Cant see text on signs".
//!
//! `TileEntitySignRenderer.renderTileEntityAt:95-96` draws
//! `GuiUtilRenderComponents.splitText(row, 90, fontrenderer, false, true).get(0)
//! .getFormattedText()`. Every expected string below is the output of **the real 1.8.9
//! bytecode** for that expression: a small harness loaded `1.8.9.jar` (plus its guava, gson,
//! commons-lang3 and icu4j libraries), parsed each row with `IChatComponent.Serializer`
//! (`eu$a.a`), allocated a `FontRenderer` (`avn`) without its GL constructor, filled its
//! `charWidth` table with `FontRenderer.readFontTexture`'s own loop over the jar's `ascii.png`,
//! and called `avu.a(row, 90, font, false, true)` then `eu.d()`. The harness transcribes
//! Minecraft code, so its source is not published.
//!
//! The first three rows are verbatim S33 payloads captured from a real 1.8.8 server
//! (`mc-parity/tests/fixtures/spigot-1.8.8-offline-join-walk-chat.mccap`): an empty root with the
//! text in `extra`. A root-only reading of the row returns `"§r"` for all of them — a blank sign.
//!
//! Skips (loudly) without the jar; point `MC_JARS_DIR` at a `jars/` directory to override.

use std::path::{Path, PathBuf};

use mc_assets::{Language, ResourceManager, ZipResourcePack};
use mc_ui::font::component::ChatComponent;
use mc_ui::font::{FontData, FontRenderer};
use mc_ui::screens::chat::sign_row_text;

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
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(FontRenderer::new(
        FontData::load(&rm).expect("load vanilla font"),
        false,
        false,
    ))
}

/// `(S33 row JSON, what vanilla's sign renderer draws for it)`.
const GOLDENS: &[(&str, &str)] = &[
    // Real 1.8.8 server captures.
    (
        r#"{"extra":["-CXXVI"],"text":""}"#,
        "\u{a7}r\u{a7}r-CXXVI\u{a7}r",
    ),
    (
        r#"{"extra":["XLIII"],"text":""}"#,
        "\u{a7}r\u{a7}rXLIII\u{a7}r",
    ),
    (
        r#"{"extra":["ratiooooooooo"],"text":""}"#,
        "\u{a7}r\u{a7}rratiooooooooo\u{a7}r",
    ),
    (r#""""#, "\u{a7}r\u{a7}r"),
    (r#"{"text":"Shop"}"#, "\u{a7}rShop\u{a7}r"),
    // The captured server's coloured form: the style sits on the sibling.
    (
        r#"{"extra":[{"color":"dark_blue","text":"[Shop]"}],"text":""}"#,
        "\u{a7}r\u{a7}r\u{a7}1\u{a7}1[Shop]\u{a7}r",
    ),
    // Nested siblings inherit their parent's resolved style.
    (
        r#"{"text":"base","color":"red","extra":[{"text":" child","color":"blue","extra":[{"text":" deep"}]}]}"#,
        "\u{a7}r\u{a7}c\u{a7}cbase\u{a7}r\u{a7}9\u{a7}9 child\u{a7}r\u{a7}9\u{a7}9 deep\u{a7}r",
    ),
    (
        r#"{"text":"","extra":[{"text":"not visible"}]}"#,
        "\u{a7}r\u{a7}rnot visible\u{a7}r",
    ),
    (r#"["three","four"]"#, "\u{a7}rthree\u{a7}rfour\u{a7}r"),
    (
        r#"{"text":"one","extra":[{"text":"two"}]}"#,
        "\u{a7}rone\u{a7}rtwo\u{a7}r",
    ),
    (
        r#"{"bold":true,"extra":[{"text":"A"},{"text":"B","italic":true}],"text":"x"}"#,
        "\u{a7}r\u{a7}l\u{a7}lx\u{a7}r\u{a7}l\u{a7}lA\u{a7}r\u{a7}l\u{a7}o\u{a7}l\u{a7}oB\u{a7}r",
    ),
    (
        r#"{"extra":[{"strikethrough":true,"color":"gold","text":"Sale"},{"obfuscated":true,"text":"x"}],"text":""}"#,
        "\u{a7}r\u{a7}r\u{a7}6\u{a7}m\u{a7}6\u{a7}mSale\u{a7}r\u{a7}k\u{a7}kx\u{a7}r",
    ),
    // Legacy codes inside the text survive (`splitText`'s fifth argument is `true`).
    (
        r#"{"text":"§cLegacy §lcodes"}"#,
        "\u{a7}r\u{a7}cLegacy \u{a7}lcodes\u{a7}r",
    ),
    // Only the first wrapped line is drawn: overflow is trimmed at 90 px, at the last space when
    // there is one, and a following component that no longer fits leaves an empty piece.
    (
        r#"{"extra":["MMMMMMMMMMMMMMMMMMMMMM"],"text":""}"#,
        "\u{a7}r\u{a7}rMMMMMMMMMMMMMMM\u{a7}r",
    ),
    (
        r#"{"extra":["aaaa bbbb cccc dddd eeee ffff"],"text":""}"#,
        "\u{a7}r\u{a7}raaaa bbbb cccc\u{a7}r",
    ),
    (r#"{"text":"line one\nline two"}"#, "\u{a7}rline one\u{a7}r"),
    (
        r#"{"extra":[{"text":"aaaaaaaaaaaaaaa"},{"text":"bbbbbbb"}],"text":""}"#,
        "\u{a7}r\u{a7}raaaaaaaaaaaaaaa\u{a7}r\u{a7}r",
    ),
    (
        r#"{"extra":[{"text":"aaaaaaaaaaaa"},{"text":"bbb cccccc"}],"text":""}"#,
        "\u{a7}r\u{a7}raaaaaaaaaaaa\u{a7}rbbb\u{a7}r",
    ),
    (
        r#"{"extra":[{"bold":true,"text":"WWWWWWWWWWWWW"}],"text":""}"#,
        "\u{a7}r\u{a7}r\u{a7}l\u{a7}lWWWWWWWWWWWW\u{a7}r",
    ),
];

#[test]
fn sign_rows_match_the_vanilla_split_text_bytecode() {
    let Some(font) = font() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let lang = Language::new("en_US");
    let mut failures = Vec::new();
    for (json, expected) in GOLDENS {
        let component = ChatComponent::from_json(json).expect("vanilla's Gson accepts the row");
        let actual = sign_row_text(&component, 90, &font, &lang);
        if actual != *expected {
            failures.push(format!(
                "{json}\n   java: {expected:?}\n   rust: {actual:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} sign rows differ from the 1.8.9 bytecode:\n{}",
        failures.len(),
        GOLDENS.len(),
        failures.join("\n")
    );
}
