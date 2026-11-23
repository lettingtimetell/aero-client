//! The status-ping JSON: MOTD, player counts, version, favicon.
//!
//! java: `client/network/OldServerPinger.java`, `network/ServerStatusResponse.java`
//!
//! S07 gives us [`ping_server`](mc_protocol::connection::ping_server), which returns the raw
//! JSON string and a measured latency and deliberately does not model the schema. This is where
//! it becomes the six fields a server-list row shows.
//!
//! **Everything in here is untrusted**: the payload is whatever a stranger's server sent. Every
//! field is optional, every type is checked, the favicon is size-capped, and nothing panics.

use base64::Engine;
use image::ImageDecoder;
use mc_assets::Language;
use serde_json::Value;
use std::io::Cursor;

use crate::font::component::ChatComponent;
use crate::screens::multiplayer::server_data::{PingState, ServerEntry};

/// Largest favicon payload accepted, in decoded bytes.
///
/// Vanilla checks only the decoded image's dimensions (64×64) and will happily decode a
/// multi-megabyte PNG first. A 64×64 RGBA PNG is at most ~16 KB even stored uncompressed, so
/// 64 KB is generous and bounds the decode.
pub const MAX_FAVICON_BYTES: usize = 64 * 1024;

/// Vanilla requires a square 64-pixel server icon.
pub const FAVICON_SIDE: u32 = 64;

/// The `data:` prefix vanilla's favicons carry.
pub const FAVICON_PREFIX: &str = "data:image/png;base64,";

/// The fields a server-list row reads out of a status response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatusResponse {
    /// `description`, rendered with formatting codes. May be a string or a chat component.
    pub motd: String,
    /// `version.name`, e.g. `"1.8.9"` or `"Server 1.8.8"`; `"Old"` when the `version` object
    /// was absent, which is what `OldServerPinger.handleServerInfo:92` writes into
    /// `ServerData.gameVersion`. Left empty, `ServerEntry::version_mismatch` (protocol `0`
    /// is not 47) rendered an empty dark-red label where vanilla renders a dark-red "Old".
    pub version_name: String,
    /// `version.protocol`. 47 is 1.8.9.
    pub protocol: i32,
    /// `players.online`, or `None` when the whole `players` object was absent.
    ///
    /// java: `ServerStatusResponse.getPlayerCountData()` stays `null` in that case and
    /// `OldServerPinger.handleServerInfo:129` writes `EnumChatFormatting.DARK_GRAY + "???"`
    /// instead of a count - see [`Self::population_info`]. Storing `0` here rendered a
    /// plausible-looking "0/0" lie.
    pub online: Option<i32>,
    /// `players.max`, or `None` when the `players` object was absent.
    pub max: Option<i32>,
    /// `players.sample[].name`, in order.
    pub sample: Vec<String>,
    /// `favicon`, with the `data:image/png;base64,` prefix stripped.
    pub favicon_base64: Option<String>,
    /// Already-validated RGBA pixels supplied by the transport parser, when available.
    ///
    /// The field is optional so tests and non-network callers can still provide only the raw
    /// payload. The live client uses this to avoid decoding the same 64x64 PNG a second time on
    /// the game thread.
    pub favicon_rgba: Option<Vec<u8>>,
    /// Whether the JSON contained a primitive `favicon` field at all. This distinguishes
    /// absent (vanilla clears the old icon) from a present string with an unknown prefix
    /// (vanilla logs and keeps the old icon).
    pub favicon_present: bool,
}

impl StatusResponse {
    /// Parse a status payload.
    ///
    /// Every top-level field is optional, but a field which is present must have the shape
    /// vanilla's Gson adapters require; malformed present fields fail the whole status packet.
    // java: OldServerPinger.java:98 (the ServerStatusResponse handler)
    #[must_use]
    pub fn parse(json: &str) -> Option<Self> {
        let value: Value = serde_json::from_str(json).ok()?;
        let obj = value.as_object()?;
        let mut out = StatusResponse::default();

        if let Some(desc) = obj.get("description") {
            // java: a bare string is a legal `description`; `IChatComponent.Serializer`
            // (`util/IChatComponent.java:63-65`) wraps it in a `ChatComponentText` like any
            // other shape, and `OldServerPinger:78` then calls `getFormattedText()` on it --
            // which appends a trailing reset. The `Value::String` fast path that returned the
            // raw string skipped that, so a bare-string description came out one code short of
            // vanilla's bytes.
            if !desc.is_null() && !matches!(desc, Value::Array(items) if items.is_empty()) {
                out.motd = ChatComponent::from_value(desc, 0)
                    .ok()?
                    .formatted_text(&Language::new("en_US"));
            }
        }
        if let Some(version) = obj.get("version") {
            let v = version.as_object()?;
            out.version_name = gson_string(v.get("name")?)?;
            out.protocol = gson_i32(v.get("protocol")?)?;
        } else {
            // java: `OldServerPinger.handleServerInfo:90-94`'s else branch.
            out.version_name = "Old".to_owned();
            out.protocol = 0;
        }
        if let Some(players) = obj.get("players") {
            let p = players.as_object()?;
            out.online = Some(gson_i32(p.get("online")?)?);
            out.max = Some(gson_i32(p.get("max")?)?);
            if let Some(sample) = p.get("sample").and_then(Value::as_array) {
                let mut names = Vec::with_capacity(sample.len());
                for value in sample {
                    let player = value.as_object()?;
                    let id = gson_string(player.get("id")?)?;
                    if !java_uuid_is_valid(&id) {
                        return None;
                    }
                    names.push(gson_string(player.get("name")?)?);
                }
                out.sample = names;
            }
        }
        if let Some(favicon) = obj.get("favicon") {
            let favicon = gson_string(favicon)?;
            out.favicon_present = true;
            out.favicon_base64 = strip_favicon_prefix(&favicon).map(str::to_owned);
        }
        Some(out)
    }

    /// Vanilla's grey "online/max" population string, in `§7online§8/§7max` form.
    ///
    /// When the server sent no `players` object at all, vanilla shows `\u{a7}8???` instead.
    // java: OldServerPinger.java:98, and its else branch at :129
    #[must_use]
    pub fn population_info(&self) -> String {
        match (self.online, self.max) {
            (Some(online), Some(max)) => format!("\u{a7}7{online}\u{a7}8/\u{a7}7{max}"),
            _ => "\u{a7}8???".to_owned(),
        }
    }

    /// Vanilla's hover text: the sample names, one per line, plus an "and N more..." tail.
    ///
    /// The tail is a translation key, so this takes the language. An empty sample yields an
    /// empty string, which is what suppresses the tooltip entirely.
    // java: OldServerPinger.java:118
    #[must_use]
    pub fn player_list(&self, lang: &Language) -> String {
        if self.sample.is_empty() {
            return String::new();
        }
        let mut out = self.sample.join("\n");
        let hidden = self.online.unwrap_or(0) - self.sample.len() as i32;
        if hidden > 0 {
            out.push('\n');
            // java: `OldServerPinger.java:120-121` appends the **hardcoded English literal**
            // `"... and " + n + " more ..."`. 1.8.9 has no `multiplayer.status.*` translation
            // key at all -- grepping the shipped `en_US.lang`, and all 68 language files in
            // the java tree, for `multiplayer.status` matches nothing -- so routing this
            // through `Language` rendered the literal key `multiplayer.status.and_more` on
            // every server with more players online than its sample carries, i.e. the normal
            // case. The only definition of that key anywhere was the unit test that injected
            // it, which is why the test passed.
            out.push_str(&format!("... and {hidden} more ..."));
        }
        let _ = lang;
        out
    }

    /// Decode the favicon into 64x64 RGBA pixels, rejecting anything oversized, malformed, or
    /// the wrong dimensions.
    ///
    /// Returns `None` rather than an error: vanilla logs "Invalid icon for server …" and falls
    /// back to `unknown_server.png`, and so does the row renderer.
    // java: ServerListEntryNormal.java:244 (prepareServerIcon)
    #[must_use]
    pub fn decode_favicon(&self) -> Option<Vec<u8>> {
        decode_favicon(self.favicon_base64.as_deref()?)
    }

    /// Return validated favicon pixels, preferring the transport parser's one-time decode.
    ///
    /// Raw-payload callers still take the ordinary [`Self::decode_favicon`] path. A caller that
    /// provides pixels directly cannot bypass the exact 64x64 RGBA invariant.
    #[must_use]
    pub fn decoded_favicon(&self) -> Option<Vec<u8>> {
        self.favicon_rgba
            .as_ref()
            .filter(|rgba| rgba.len() == (FAVICON_SIDE * FAVICON_SIDE * 4) as usize)
            .cloned()
            .or_else(|| self.decode_favicon())
    }

    /// Copy this response into a saved list entry, marking it [`PingState::Ok`]. Returns whether
    /// the persisted favicon field changed, which is when vanilla saves `servers.dat` again.
    pub fn apply_to(&self, entry: &mut ServerEntry, latency_ms: i64, lang: &Language) -> bool {
        entry.motd = self.motd.clone();
        entry.population = self.population_info();
        entry.protocol = self.protocol;
        entry.game_version = self.version_name.clone();
        entry.player_list = self.player_list(lang);
        entry.ping_ms = latency_ms;
        entry.state = PingState::Ok;
        let previous_icon = entry.icon_base64.clone();
        // A missing favicon clears the saved one. A valid-prefix value that fails PNG decoding
        // is cleared while preparing the response; an unknown prefix keeps the old icon, matching
        // Java's three-way handling of null, usable PNG data, and unsupported image formats.
        if let Some(icon) = &self.favicon_base64 {
            entry.icon_base64 = self.decoded_favicon().map(|_| icon.clone());
        } else if !self.favicon_present {
            entry.icon_base64 = None;
        }
        previous_icon != entry.icon_base64
    }
}

/// Strip vanilla's `data:image/png;base64,` prefix. Anything else is rejected.
#[must_use]
pub fn strip_favicon_prefix(favicon: &str) -> Option<&str> {
    favicon.strip_prefix(FAVICON_PREFIX)
}

/// Gson's `JsonPrimitive.getAsInt`: numeric JSON primitives are accepted, including integral
/// floats, while strings and booleans are rejected by `JsonUtils.getInt` before coercion.
fn gson_i32(value: &Value) -> Option<i32> {
    let number = value.as_number()?;
    let text = number.to_string();
    text.parse::<i32>()
        .ok()
        .or_else(|| text.parse::<i64>().ok().map(|n| n as i32))
        .or_else(|| text.parse::<u64>().ok().map(|n| n as i32))
        .or_else(|| decimal_to_wrapping_i32(&text))
}

/// `BigDecimal.intValue`: truncate toward zero and retain the low 32 integer bits.
fn decimal_to_wrapping_i32(text: &str) -> Option<i32> {
    let (negative, unsigned) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(index) => (
            &unsigned[..index],
            unsigned[index + 1..].parse::<i64>().ok()?,
        ),
        None => (unsigned, 0),
    };
    let mut digits = Vec::with_capacity(mantissa.len());
    let mut fractional_digits = 0_i64;
    let mut after_decimal = false;
    for byte in mantissa.bytes() {
        match byte {
            b'0'..=b'9' => {
                digits.push(byte - b'0');
                if after_decimal {
                    fractional_digits += 1;
                }
            }
            b'.' if !after_decimal => after_decimal = true,
            _ => return None,
        }
    }
    if digits.is_empty() {
        return None;
    }

    // BigDecimal stores `scale = fractional_digits - exponent` as a signed Java int and throws
    // `NumberFormatException: Exponent overflow` when that derived scale does not fit.
    let scale = fractional_digits.checked_sub(exponent)?;
    i32::try_from(scale).ok()?;
    let integer_digits = i64::try_from(digits.len()).ok()?.checked_sub(scale)?;
    if integer_digits <= 0 {
        return Some(0);
    }
    let kept = usize::try_from(integer_digits)
        .unwrap_or(usize::MAX)
        .min(digits.len());
    let mut low = 0_u32;
    for &digit in &digits[..kept] {
        low = low.wrapping_mul(10).wrapping_add(u32::from(digit));
    }
    let trailing_zeroes = integer_digits.saturating_sub(i64::try_from(kept).ok()?);
    if trailing_zeroes >= 32 {
        low = 0;
    } else {
        for _ in 0..trailing_zeroes {
            low = low.wrapping_mul(10);
        }
    }
    Some(if negative {
        low.wrapping_neg() as i32
    } else {
        low as i32
    })
}

/// `JsonUtils.getString` accepts any JSON primitive and delegates to
/// `JsonPrimitive.getAsString`; only null, arrays, and objects are rejected.
fn gson_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// Java 8's `UUID.fromString` accepts shortened groups but still requires five non-empty hex
/// groups, at most 36 total characters, each parseable as a signed hexadecimal `long`.
fn java_uuid_is_valid(value: &str) -> bool {
    if value.len() > 36 {
        return false;
    }
    let parts: Vec<&str> = value.split('-').collect();
    parts.len() == 5
        && parts
            .iter()
            .all(|part| !part.is_empty() && i64::from_str_radix(part, 16).is_ok())
}

/// Decode a base64 favicon payload into tightly packed RGBA8, capped at
/// [`MAX_FAVICON_BYTES`] and requiring vanilla's exact 64x64 dimensions.
#[must_use]
pub fn decode_favicon(base64_payload: &str) -> Option<Vec<u8>> {
    let compact: String = base64_payload
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();
    // Bound the meaningful encoded length before decoding. Whitespace is ignored by Netty's
    // decoder and therefore must not count against an otherwise valid wrapped icon.
    if compact.len() > MAX_FAVICON_BYTES / 3 * 4 + 4 {
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&compact)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(&compact))
        .ok()?;
    if bytes.len() > MAX_FAVICON_BYTES {
        return None;
    }
    // Read IHDR before allocating the decoded surface. A tiny, highly-compressed PNG can claim
    // enormous dimensions, so decoding first would turn the encoded-size guard into a false
    // sense of safety.
    let decoder = image::codecs::png::PngDecoder::new(Cursor::new(bytes)).ok()?;
    if decoder.dimensions() != (FAVICON_SIDE, FAVICON_SIDE) {
        return None;
    }
    Some(
        image::DynamicImage::from_decoder(decoder)
            .ok()?
            .to_rgba8()
            .into_raw(),
    )
}

/// Mark an entry as having failed to ping, with vanilla's dark-red MOTD.
// java: ServerListEntryNormal.java:64
pub fn mark_failed(entry: &mut ServerEntry, unresolved_host: bool) {
    entry.state = PingState::Failed;
    entry.ping_ms = -1;
    entry.population = String::new();
    entry.player_list.clear();
    entry.motd = if unresolved_host {
        "\u{a7}4Can't resolve hostname".to_owned()
    } else {
        "\u{a7}4Can't connect to server.".to_owned()
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    fn png_base64(width: u32, height: u32) -> String {
        let image = image::RgbaImage::from_pixel(width, height, image::Rgba([1, 2, 3, 255]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode PNG");
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    const REAL: &str = r#"{
        "description": {"text": "A Minecraft Server", "color": "gold"},
        "players": {"max": 20, "online": 2, "sample": [{"name":"alice","id":"00000000-0000-0000-0000-000000000001"},{"name":"bob","id":"00000000-0000-0000-0000-000000000002"}]},
        "version": {"name": "1.8.9", "protocol": 47},
        "favicon": "data:image/png;base64,aGVsbG8="
    }"#;

    #[test]
    fn a_full_status_response_parses_every_field() {
        let s = StatusResponse::parse(REAL).expect("parse");
        // `formatted_text` appends vanilla's trailing `§r`, which is why the MOTD carries one.
        assert_eq!(s.motd, "\u{a7}6A Minecraft Server\u{a7}r");
        assert_eq!(s.version_name, "1.8.9");
        assert_eq!(s.protocol, 47);
        assert_eq!(s.online, Some(2));
        assert_eq!(s.max, Some(20));
        assert_eq!(s.sample, ["alice", "bob"]);
        assert_eq!(s.favicon_base64.as_deref(), Some("aGVsbG8="));
        assert_eq!(s.population_info(), "\u{a7}72\u{a7}8/\u{a7}720");
    }

    /// java: a bare-string `description` goes through `IChatComponent.Serializer` like any
    /// other shape, so `getFormattedText()` appends the same trailing reset an object
    /// description gets. This test used to assert the raw string, i.e. the bug.
    #[test]
    fn a_plain_string_description_still_gets_the_trailing_reset() {
        let s = StatusResponse::parse(r#"{"description":"hello"}"#).expect("parse");
        assert_eq!(s.motd, "hello\u{a7}r");
    }

    /// java: `OldServerPinger.handleServerInfo`'s two else branches - a missing `version`
    /// object gives `gameVersion = "Old"` / `version = 0` (`:92-93`), and a missing `players`
    /// object gives `populationInfo = DARK_GRAY + "???"` (`:129`).
    ///
    /// This test used to assert `StatusResponse::default()` and a literal `0/0`, i.e. exactly
    /// the two lies. `0/0` is the dangerous one: it looks like a real answer.
    #[test]
    fn an_absent_version_reads_old_and_an_absent_players_object_reads_question_marks() {
        let s = StatusResponse::parse("{}").expect("parse");
        assert_eq!(s.version_name, "Old");
        assert_eq!(s.protocol, 0);
        assert_eq!(s.online, None);
        assert_eq!(s.max, None);
        assert_eq!(s.population_info(), "\u{a7}8???");
        assert_eq!(s.player_list(&Language::new("en_US")), "");

        assert!(
            StatusResponse::parse(r#"{"players":{}}"#).is_none(),
            "present players requires numeric max and online fields"
        );
    }

    #[test]
    fn non_json_and_non_object_payloads_are_rejected_not_panicked_on() {
        assert!(StatusResponse::parse("not json").is_none());
        assert!(StatusResponse::parse("[1,2,3]").is_none());
        assert!(StatusResponse::parse("42").is_none());
        assert!(StatusResponse::parse("").is_none());
    }

    #[test]
    fn malformed_present_fields_fail_the_whole_status_like_gson() {
        for json in [
            r#"{"players":{"online":"lots","max":20}}"#,
            r#"{"players":{"online":2,"max":null}}"#,
            r#"{"version":{"protocol":true,"name":"x"}}"#,
            r#"{"version":{"protocol":47,"name":{}}}"#,
            r#"{"favicon":[]}"#,
            r#"{"players":{"online":1,"max":2,"sample":[{"id":"not-a-uuid","name":"x"}]}}"#,
            r#"{"players":{"online":1,"max":2,"sample":[{"id":"0-0-0-0-0"}]}}"#,
            r#"{"players":{"online":1,"max":2,"sample":[{"id":"0-0-0-0-0","name":[]}]}}"#,
        ] {
            assert!(StatusResponse::parse(json).is_none(), "accepted {json}");
        }

        let s =
            StatusResponse::parse(r#"{"players":{"online":1,"max":2,"sample":"not an array"}}"#)
                .expect("JsonUtils.isJsonArray ignores a wrong-typed optional sample");
        assert!(s.sample.is_empty());

        let s = StatusResponse::parse(
            r#"{"players":{"online":1,"max":2,"sample":[{"id":"0-0-0-0-0","name":"short"}]}}"#,
        )
        .expect("Java 8 UUID.fromString accepts shortened groups");
        assert_eq!(s.sample, ["short"]);

        for description in ["null", "[]"] {
            let json =
                format!(r#"{{"description":{description},"players":{{"online":1,"max":2}}}}"#);
            let s = StatusResponse::parse(&json).expect("Gson maps null/empty arrays to null");
            assert_eq!(s.motd, "");
            assert_eq!(s.online, Some(1));
        }
    }

    #[test]
    fn jsonutils_get_string_coerces_every_primitive_before_status_use() {
        let s = StatusResponse::parse(
            r#"{"version":{"name":7,"protocol":47},"players":{"online":1,"max":2,"sample":[{"id":"0-0-0-0-0","name":true}]},"favicon":12}"#,
        )
        .expect("JsonUtils.getString accepts number and boolean primitives");
        assert_eq!(s.version_name, "7");
        assert_eq!(s.sample, ["true"]);
        assert!(s.favicon_present);
        assert!(
            s.favicon_base64.is_none(),
            "a numeric favicon coerces to an unsupported-prefix string"
        );

        let mut entry = ServerEntry::new("saved", "example.test");
        entry.icon_base64 = Some("old-icon".to_owned());
        assert!(!s.apply_to(&mut entry, 1, &Language::new("en_US")));
        assert_eq!(
            entry.icon_base64.as_deref(),
            Some("old-icon"),
            "a present unsupported-format favicon keeps the saved icon"
        );
    }

    #[test]
    fn gson_numeric_int_coercion_accepts_floats_and_wraps_long_values() {
        let s = StatusResponse::parse(
            r#"{"players":{"online":1e100,"max":-2.9},"version":{"name":"x","protocol":99999999999}}"#,
        )
        .expect("parse");
        assert_eq!(s.online, Some(0));
        assert_eq!(s.max, Some(-2));
        assert_eq!(s.protocol, 99_999_999_999_i64 as i32);
        assert_ne!(
            s.protocol,
            i32::MAX,
            "negative control: Rust's saturating float cast is not Gson wrapping"
        );
    }

    #[test]
    fn gson_numeric_coercion_keeps_extreme_fraction_lexemes_exact() {
        // Oracle: bundled Gson 2.2.4 JsonParser + JsonPrimitive.getAsInt. These values are
        // deliberately beyond f64's exact-integer range, so disabling serde_json's
        // arbitrary_precision feature makes this test observe rounded inputs.
        for (lexeme, expected) in [
            ("9007199254740993.5", 1),
            ("-9007199254740993.5", -1),
            ("4294967297.999999999999", 1),
            ("1.5e32", i32::MIN),
            ("1.0e2147483649", 0),
        ] {
            let json = format!(
                r#"{{"players":{{"online":{lexeme},"max":0}},"version":{{"name":"x","protocol":47}}}}"#
            );
            let status = StatusResponse::parse(&json).expect("Gson-compatible numeric primitive");
            assert_eq!(status.online, Some(expected), "lexeme {lexeme}");
        }

        for lexeme in ["1e2147483649", "1e-2147483648"] {
            let json = format!(r#"{{"players":{{"online":{lexeme},"max":0}}}}"#);
            assert!(
                StatusResponse::parse(&json).is_none(),
                "Gson reports exponent overflow for {lexeme}"
            );
        }
    }

    /// java: `OldServerPinger.java:120-121` builds this from a hardcoded English literal,
    /// not a translation key - 1.8.9 ships no `multiplayer.status.*` key.
    ///
    /// The old version of this test inserted `multiplayer.status.and_more` into the language
    /// by hand, so it passed while every real client rendered the bare key.
    #[test]
    fn the_hover_list_appends_the_hidden_player_count_from_a_hardcoded_literal() {
        // Deliberately an *empty* language: vanilla never consults one here, so neither may
        // this, and a `Language::tr` miss would show up immediately as the raw key.
        let lang = Language::new("en_US");
        assert_eq!(
            lang.format("multiplayer.status.and_more", &["3"]),
            "multiplayer.status.and_more",
            "negative control: the key really is absent, so a lookup would render itself"
        );
        let mut s = StatusResponse::parse(REAL).expect("parse");
        assert_eq!(s.player_list(&lang), "alice\nbob");
        s.online = Some(5);
        assert_eq!(s.player_list(&lang), "alice\nbob\n... and 3 more ...");
    }

    #[test]
    fn a_favicon_must_carry_vanillas_exact_data_prefix() {
        assert_eq!(
            strip_favicon_prefix("data:image/png;base64,QQ=="),
            Some("QQ==")
        );
        assert!(strip_favicon_prefix("data:image/jpeg;base64,QQ==").is_none());
        assert!(strip_favicon_prefix("QQ==").is_none());
        assert!(strip_favicon_prefix("javascript:alert(1)").is_none());
    }

    #[test]
    fn an_oversized_favicon_is_refused_before_it_is_decoded() {
        let huge = "A".repeat(MAX_FAVICON_BYTES * 2);
        assert!(decode_favicon(&huge).is_none());
        assert!(decode_favicon("!!!not base64!!!").is_none());
        assert!(
            decode_favicon("aGVsbG8=").is_none(),
            "base64 is not enough: it must be PNG"
        );
    }

    #[test]
    fn a_favicon_must_decode_to_exactly_64_by_64_rgba() {
        let encoded = png_base64(64, 64);
        let decoded = decode_favicon(&encoded).expect("valid icon");
        assert_eq!(decoded.len(), 64 * 64 * 4);
        assert!(decode_favicon(&png_base64(63, 64)).is_none());
        assert!(decode_favicon(&png_base64(64, 65)).is_none());
        let wrapped = encoded
            .as_bytes()
            .chunks(76)
            .map(|line| std::str::from_utf8(line).unwrap())
            .collect::<Vec<_>>()
            .join("\r\n");
        assert!(decode_favicon(&wrapped).is_some());
        assert!(decode_favicon(encoded.trim_end_matches('=')).is_some());
    }

    #[test]
    fn applying_a_response_fills_the_entry_and_drops_an_undecodable_icon() {
        let lang = Language::new("en_US");
        let mut e = ServerEntry::new("n", "a");
        e.icon_base64 = Some("stale".into());
        let mut s = StatusResponse::parse(REAL).expect("parse");
        s.favicon_base64 = Some("!!!".into());
        s.apply_to(&mut e, 123, &lang);
        assert_eq!(e.state, PingState::Ok);
        assert_eq!(e.ping_ms, 123);
        assert_eq!(e.protocol, 47);
        assert_eq!(e.game_version, "1.8.9");
        assert_eq!(e.ping_bars(), 0);
        assert!(e.icon_base64.is_none(), "a bad icon is cleared, not kept");
    }

    #[test]
    fn a_response_without_a_favicon_clears_and_reports_the_saved_icon_change() {
        let lang = Language::new("en_US");
        let mut e = ServerEntry::new("n", "a");
        e.icon_base64 = Some(png_base64(64, 64));
        let s = StatusResponse::parse("{}").expect("parse");
        assert!(s.apply_to(&mut e, 1, &lang));
        assert!(e.icon_base64.is_none());
        assert!(!s.apply_to(&mut e, 2, &lang), "null to null needs no save");
    }

    #[test]
    fn an_unknown_favicon_format_keeps_the_previous_icon() {
        let lang = Language::new("en_US");
        let mut e = ServerEntry::new("n", "a");
        e.icon_base64 = Some("old payload".to_owned());
        let s = StatusResponse::parse(r#"{"favicon":"data:image/jpeg;base64,AAAA"}"#)
            .expect("the status packet is valid even though the icon format is unknown");
        assert!(s.favicon_present);
        assert!(s.favicon_base64.is_none());
        assert!(!s.apply_to(&mut e, 1, &lang));
        assert_eq!(e.icon_base64.as_deref(), Some("old payload"));
    }

    #[test]
    fn a_failed_ping_gets_vanillas_dark_red_message() {
        let mut e = ServerEntry::new("n", "a");
        e.player_list = "stale hover".to_owned();
        mark_failed(&mut e, true);
        assert_eq!(e.state, PingState::Failed);
        assert_eq!(e.ping_ms, -1);
        assert_eq!(e.ping_bars(), 5);
        assert!(e.motd.starts_with('\u{a7}'));
        assert!(e.player_list.is_empty());
        mark_failed(&mut e, false);
        assert!(e.motd.contains("connect"));
    }
}
