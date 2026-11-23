//! Which chat and book link values may leave the client.
//!
//! java: `GuiScreen.handleComponentClick` parses an `open_url` value with `new URI(value)` and
//! gives up (logging "Can't open url") when that throws, before the scheme check against
//! `GuiScreen.PROTOCOLS` (`http`, `https`) and long before `GuiConfirmOpenLink` or the browser.
//! The client hands an accepted URL to the operating system as a command-line argument
//! (`rundll32 url.dll,FileProtocolHandler <url>` on Windows, `open` or `xdg-open` elsewhere), so
//! refusing what Java's parser refuses also keeps quotes, angle brackets, pipes and control
//! characters out of that command line and the browser's.

/// Whether `url` avoids every character `java.net.URI` refuses in all of a URI's components.
///
/// This is deliberately a subset of Java's grammar: a value it refuses always makes
/// `new URI(value)` throw, so no link that opens in vanilla is refused. It checks:
/// - ISO control characters and Unicode space characters (`Character.isISOControl` and
///   `Character.isSpaceChar`, which `URI.scanEscape` refuses even outside ASCII);
/// - the ASCII characters outside every RFC 2396 character class: `"` `<` `>` `\` `^` `` ` ``
///   `{` `|` `}`;
/// - a `%` that is not followed by two hex digits ("Malformed escape pair");
/// - a second `#`: a fragment may not contain one.
#[must_use]
pub(crate) fn java_uri_characters_allowed(url: &str) -> bool {
    let mut fragment = false;
    let mut chars = url.chars();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_control() || c.is_whitespace() => return false,
            '"' | '<' | '>' | '\\' | '^' | '`' | '{' | '|' | '}' => return false,
            '%' => {
                let mut pair = chars.clone().take(2);
                let hex = pair.next().is_some_and(|c| c.is_ascii_hexdigit())
                    && pair.next().is_some_and(|c| c.is_ascii_hexdigit());
                if !hex {
                    return false;
                }
            }
            '#' if fragment => return false,
            '#' => fragment = true,
            _ => {}
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use mc_ui::screens::Action;

    use super::super::driver::{take_opened_external_targets, Client, Ui};
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};

    /// Values `new URI(value)` accepts stay accepted, including the characters Java allows in
    /// only some components and non-ASCII "other" characters.
    #[test]
    fn links_javas_uri_parser_accepts_stay_allowed() {
        for url in [
            "https://example.invalid/",
            "HTTPS://example.invalid/path?q=1&r=2#top",
            "http://example.invalid/a%20b%2Fc",
            "http://example.invalid/search?q=[x]",
            "http://example.invalid/(it's)*!~-_.",
            "http://example.invalid/caf\u{e9}/\u{65e5}\u{672c}",
            "http://user:pass@example.invalid:8080/;p=1,$+",
        ] {
            assert!(java_uri_characters_allowed(url), "{url}");
        }
    }

    /// Each of these makes `new URI(value)` throw `URISyntaxException`, so vanilla never opens
    /// them.
    #[test]
    fn links_javas_uri_parser_refuses_are_refused() {
        for url in [
            "https://example.invalid/\"",
            "https://example.invalid/\"--new-window",
            "https://example.invalid/<script>",
            "https://example.invalid/a\\b",
            "https://example.invalid/^",
            "https://example.invalid/`",
            "https://example.invalid/{x}",
            "https://example.invalid/a|b",
            "https://example.invalid/%",
            "https://example.invalid/%4",
            "https://example.invalid/%zz",
            "https://example.invalid/#a#b",
            "https://example.invalid/\u{0}",
            "https://example.invalid/\u{7f}",
            "https://example.invalid/\u{85}",
            "https://example.invalid/\u{a0}",
            "https://example.invalid/\u{2028}",
            "https://example.invalid/ x",
        ] {
            assert!(!java_uri_characters_allowed(url), "{url:?}");
        }
    }

    /// `\uXXXX` UTF-16 escapes (the fixture's spelling of everything outside printable ASCII)
    /// back to text.
    fn unescape(escaped: &str) -> String {
        let mut units = Vec::new();
        let mut rest = escaped;
        while let Some(at) = rest.find("\\u") {
            units.extend(rest[..at].encode_utf16());
            units.push(u16::from_str_radix(&rest[at + 2..at + 6], 16).expect("hex escape"));
            rest = &rest[at + 6..];
        }
        units.extend(rest.encode_utf16());
        String::from_utf16(&units).expect("the corpus has no lone surrogates")
    }

    /// The subset claim, checked against `java.net.URI` itself on Java 8: every value the client
    /// refuses, Java refuses too, so no link that opens in vanilla is refused. The fixture is
    /// `javaref/LinkUriRef.java`'s corpus (`javaref/generate-link-uri.ps1`): every character
    /// from U+0000 to U+00FF and others, in each part of a URL, plus escapes and fragments.
    #[test]
    fn every_value_refused_is_also_refused_by_javas_uri_parser() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/link_uri_ref.tsv"
        );
        let fixture = std::fs::read_to_string(path).expect("the link URI fixture");
        let (mut cases, mut refused, mut java_alone) = (0usize, 0usize, 0usize);
        let mut java_accepted = Vec::new();
        for line in fixture.lines() {
            let (verdict, escaped) = line.split_once('\t').expect("verdict TAB value");
            let java_accepts = match verdict {
                "ok" => true,
                "err" => false,
                other => panic!("unknown verdict {other:?}"),
            };
            let value = unescape(escaped);
            let allowed = java_uri_characters_allowed(&value);
            assert!(
                allowed || !java_accepts,
                "refused a value Java accepts: {escaped}"
            );
            cases += 1;
            refused += usize::from(!allowed);
            java_alone += usize::from(allowed && !java_accepts);
            if java_accepts {
                java_accepted.push(value);
            }
        }
        // Teeth: Java accepts values holding characters a stricter rule could wrongly refuse.
        for needle in ["[", "%41", "\u{e9}", "\u{202e}", "\u{1f600}"] {
            assert!(
                java_accepted.iter().any(|value| value.contains(needle)),
                "no accepted value holds {needle:?}"
            );
        }
        eprintln!(
            "{cases} values: the client refuses {refused}, all refused by Java too; \
             Java alone refuses {java_alone} more"
        );
        assert!(cases >= 1700, "the whole corpus was read ({cases})");
        assert!(
            refused >= 400,
            "the corpus exercises the refusals ({refused})"
        );
    }

    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// A chat screen whose links open with no confirmation, the setting where a click goes
    /// straight to the operating system.
    fn chat_with_links_unprompted() -> Option<(Client, tempfile::TempDir)> {
        let (mut client, dir) = offline_client()?;
        client.settings.chat_links = true;
        client.settings.chat_links_prompt = false;
        client.open_chat("");
        let _ = take_opened_external_targets();
        Some((client, dir))
    }

    /// The pack screen's "Open Pack Folder" button used to send
    /// `OpenUrl("resourcepacks")`, so a server's `open_url` link (chat or book) with that value
    /// opened the player's pack folder, with no confirmation. Vanilla refuses the value: it has
    /// no scheme ("Missing protocol").
    #[test]
    fn a_link_valued_resourcepacks_does_not_open_the_pack_folder() {
        let Some((mut client, _dir)) = chat_with_links_unprompted() else {
            return;
        };
        client.handle_action(Action::OpenUrl("resourcepacks".to_owned()));
        assert_eq!(take_opened_external_targets(), Vec::<PathBuf>::new());
        assert!(matches!(client.ui, Ui::Chat(_)), "{}", client.ui.name());

        // The pack screen's own intent still opens the folder.
        client.handle_action(Action::OpenResourcePackFolder);
        assert_eq!(
            take_opened_external_targets(),
            vec![client.config.resource_pack_dir.clone()]
        );
    }

    /// With the prompt off this is the last check before the command
    /// line. A quote would end the argument a browser's registered command wraps the URL in.
    #[test]
    fn a_link_javas_uri_parser_refuses_is_never_opened() {
        let Some((mut client, _dir)) = chat_with_links_unprompted() else {
            return;
        };
        for url in [
            "https://example.invalid/\"--new-window",
            "https://example.invalid/a|b",
            "https://example.invalid/%zz",
        ] {
            client.handle_action(Action::OpenUrl(url.to_owned()));
            assert_eq!(
                take_opened_external_targets(),
                Vec::<PathBuf>::new(),
                "{url}"
            );
        }
        client.handle_action(Action::OpenUrl("https://example.invalid/page".to_owned()));
        assert_eq!(
            take_opened_external_targets(),
            vec![PathBuf::from("https://example.invalid/page")]
        );
    }
}
