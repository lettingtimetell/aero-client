//! Resource packs: the sources a [`ResourceManager`](crate::ResourceManager) stacks.
//!
//! Three implementations ship here, matching the three shapes vanilla ships:
//!
//! | this crate | java | backing store |
//! |---|---|---|
//! | [`ObjectStorePack`] | `DefaultResourcePack` + `ResourceIndex` | `assets/objects/<ab>/<sha1>` |
//! | [`ZipResourcePack`] | `FileResourcePack` | `1.8.9.jar`, user `.zip` packs |
//! | [`FolderResourcePack`] | `FolderResourcePack` | an unpacked directory |
//!
// java: IResourcePack.java, AbstractResourcePack.java

mod folder;
mod object_store;
mod zip_pack;

use std::io::Read;
use std::path::PathBuf;

use serde_json::Value;

pub use folder::FolderResourcePack;
pub use object_store::ObjectStorePack;
pub use zip_pack::{ZipLimits, ZipResourcePack};

use crate::{AssetError, LanguageInfo, ResourceLocation};

/// The `pack_format` 1.8.9 writes and reads. Later Minecraft versions bumped it.
pub const PACK_FORMAT_1_8_9: u32 = 1;

/// The largest `.mcmeta` read (`pack.mcmeta`, a texture's sidecar).
/// Metadata is parsed whole, where every tiny element costs dozens of bytes: at the general
/// 64 MiB entry cap, a `pack.mcmeta` deflating from half a megabyte cost gigabytes whenever the
/// pack was opened (the pack list opens every pack in the folder). Real ones are a few
/// kilobytes; long animation frame lists, tens.
pub const MAX_MCMETA_BYTES: u64 = 1 << 20;

/// A source of assets, addressed by [`ResourceLocation`].
///
/// Implementors are expected to be cheap to query: [`contains`](ResourcePack::contains) is
/// called once per pack per lookup while the manager walks the stack.
// java: IResourcePack.java
pub trait ResourcePack {
    /// Opens a reader for this resource, or `None` if this pack does not provide it.
    ///
    /// Returns `None` rather than an error when the underlying store fails, mirroring the
    /// frozen section 7 signature; failures are logged at `warn` level. Use
    /// [`ResourceManager::load`](crate::ResourceManager::load) when you want a typed error.
    fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>>;

    /// Whether this pack provides the resource.
    fn contains(&self, l: &ResourceLocation) -> bool;

    /// Whether this pack carries the resource, answered from what the pack already holds in
    /// memory (an asset index, a zip's entry list) without touching storage; `None` (the
    /// default) when only a storage call can tell, as for a folder.
    ///
    /// `Some(true)` is a listing, not a promise: the launcher's object store can index a file it
    /// never downloaded, and a zip entry can still fail to inflate. Whoever needs certainty
    /// opens the resource. Added so that loading
    /// `sounds.json` no longer opens every sound file at start-up
    /// ([`ResourceManager::presence`](crate::ResourceManager::presence)).
    fn lists(&self, l: &ResourceLocation) -> Option<bool> {
        let _ = l;
        None
    }

    /// The plain file on disk this pack would read `l` from, worked out without touching
    /// storage (so it may not exist); `None` (the default) when the pack does not keep resources
    /// as plain files, as a zip does. Packs are not `Send`; a path lets another thread do the
    /// read. Added for the sound prewarm
    /// ([`ResourceManager::file_path`](crate::ResourceManager::file_path)).
    fn file_path(&self, l: &ResourceLocation) -> Option<PathBuf> {
        let _ = l;
        None
    }

    /// Human-readable pack name, used in log messages and the resource-pack screen.
    // java: AbstractResourcePack.java:93 (getPackName)
    fn name(&self) -> &str {
        "unnamed"
    }

    /// Namespaces this pack declares, in sorted order.
    ///
    /// Vanilla derives these from the `assets/<namespace>/` directories present in the pack;
    /// the language loader iterates them to find `lang/*.lang` in every namespace.
    // java: IResourcePack.java (getResourceDomains)
    fn namespaces(&self) -> Vec<String> {
        Vec::new()
    }

    /// Parsed `pack.mcmeta`, when the pack has one. The vanilla `1.8.9.jar` does **not**.
    fn metadata(&self) -> Option<&PackMetadata> {
        None
    }

    /// Languages declared by this pack's root `pack.mcmeta`.
    ///
    /// The slice is retained at pack-open time because root metadata cannot be addressed by a
    /// [`ResourceLocation`]. A malformed `"language"` section is isolated from the ordinary
    /// [`PackMetadata`], matching vanilla's per-section metadata parsing.
    fn language_descriptors(&self) -> &[LanguageInfo] {
        &[]
    }

    /// Every file under `assets/<namespace>/<dir_prefix>`, as paths relative to
    /// `assets/<namespace>/` (so each starts with `dir_prefix`), in no particular order.
    ///
    /// Additive: the connected-textures loader finds a pack's
    /// `mcpatcher/ctm/**.properties` rules by listing them. The default lists nothing, which is
    /// right for the vanilla object store.
    fn list_files(&self, namespace: &str, dir_prefix: &str) -> Vec<String> {
        let _ = (namespace, dir_prefix);
        Vec::new()
    }
}

/// Whether a namespace and directory prefix are safe to join under `assets/` for a listing:
/// no traversal, no drive letters, no backslashes. An empty prefix lists the whole namespace.
pub(crate) fn listing_is_safe(namespace: &str, dir_prefix: &str) -> bool {
    let bad = |c: char| c == '\\' || c == ':' || c.is_control();
    !namespace.is_empty()
        && !namespace.contains('/')
        && !namespace.chars().any(bad)
        && !dir_prefix.starts_with('/')
        && !dir_prefix.chars().any(bad)
        && dir_prefix
            .split('/')
            .all(|segment| segment != "." && segment != "..")
}

/// The `"pack"` section of `pack.mcmeta`.
// java: PackMetadataSection.java
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackMetadata {
    /// Declared pack format. Held as `i64` so an out-of-range value produces a typed
    /// [`AssetError::PackFormat`] instead of a parse failure.
    pub pack_format: i64,
    /// Formatted description, including the `\u{a7}` codes emitted by vanilla's
    /// `IChatComponent.getFormattedText()`.
    pub description: String,
}

impl PackMetadata {
    /// Parses `pack.mcmeta` bytes.
    // java: AbstractResourcePack.java:66 (readMetadata)
    pub fn parse(bytes: &[u8]) -> Result<Self, AssetError> {
        let root: Value =
            serde_json::from_slice(bytes).map_err(|e| AssetError::json("pack.mcmeta", e))?;
        let pack = root
            .get("pack")
            .ok_or(AssetError::BadPackMeta("missing \"pack\" section"))?;
        let pack_format =
            pack.get("pack_format")
                .and_then(Value::as_i64)
                .ok_or(AssetError::BadPackMeta(
                    "missing or non-integer \"pack_format\"",
                ))?;
        let description = pack
            .get("description")
            .ok_or(AssetError::BadPackMeta("missing \"description\""))
            .and_then(format_description)?;
        Ok(Self {
            pack_format,
            description,
        })
    }

    /// Whether this client can read the pack.
    pub fn is_supported(&self) -> bool {
        self.pack_format == i64::from(PACK_FORMAT_1_8_9)
    }

    /// Returns [`AssetError::PackFormat`] unless the pack declares format
    /// [`PACK_FORMAT_1_8_9`].
    ///
    /// Kept separate from parsing so a mismatched pack can still be inspected and shown to
    /// the user in the resource-pack screen rather than blowing up the load.
    pub fn require_supported(&self) -> Result<(), AssetError> {
        if self.is_supported() {
            Ok(())
        } else {
            Err(AssetError::PackFormat {
                found: self.pack_format,
                expected: PACK_FORMAT_1_8_9,
            })
        }
    }
}

/// Parses a pack description as the 1.8.9 `IChatComponent.Serializer` and flattens it with
/// `getFormattedText`.  This deliberately validates the component rather than accepting a
/// conveniently stringifiable JSON shape: Gson rejects `{}`, `null`, empty arrays, and a
/// present non-array `extra`, and `PackMetadataSectionSerializer` propagates that rejection.
fn format_description(v: &Value) -> Result<String, AssetError> {
    let mut out = String::new();
    format_component(v, DescriptionStyle::default(), &mut out)?;
    Ok(out)
}

#[derive(Clone, Copy, Default)]
struct DescriptionStyle {
    color: Option<char>,
    bold: bool,
    italic: bool,
    underlined: bool,
    strikethrough: bool,
    obfuscated: bool,
}

impl DescriptionStyle {
    fn formatting_code(self) -> String {
        let mut out = String::new();
        if let Some(color) = self.color {
            out.push('\u{a7}');
            out.push(color);
        }
        for (enabled, code) in [
            (self.bold, 'l'),
            (self.italic, 'o'),
            (self.underlined, 'n'),
            (self.obfuscated, 'k'),
            (self.strikethrough, 'm'),
        ] {
            if enabled {
                out.push('\u{a7}');
                out.push(code);
            }
        }
        out
    }
}

/// `JsonElement.getAsString`: strings, numbers, and booleans are valid Gson primitives;
/// nulls, arrays, and objects are not.
fn gson_primitive_string(value: &Value) -> Result<String, AssetError> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        _ => Err(AssetError::BadPackMeta(
            "invalid \"description\" chat component",
        )),
    }
}

fn format_component(
    value: &Value,
    parent: DescriptionStyle,
    out: &mut String,
) -> Result<(), AssetError> {
    match value {
        Value::String(_) | Value::Number(_) | Value::Bool(_) => {
            append_formatted(out, parent, &gson_primitive_string(value)?);
            Ok(())
        }
        Value::Array(items) => {
            if items.is_empty() {
                return Err(AssetError::BadPackMeta(
                    "invalid \"description\" chat component",
                ));
            }
            for item in items {
                format_component(item, parent, out)?;
            }
            Ok(())
        }
        Value::Object(object) => format_component_object(object, parent, out),
        Value::Null => Err(AssetError::BadPackMeta(
            "invalid \"description\" chat component",
        )),
    }
}

fn format_component_object(
    object: &serde_json::Map<String, Value>,
    parent: DescriptionStyle,
    out: &mut String,
) -> Result<(), AssetError> {
    let text = if let Some(text) = object.get("text") {
        gson_primitive_string(text)?
    } else if let Some(translate) = object.get("translate") {
        // Pack metadata is parsed before there is a selected `Language`.  Keep the translation
        // key (the same fallback used by `Language::tr`) while still validating every `with`
        // component exactly as Gson does.
        if let Some(with) = object.get("with") {
            let Value::Array(args) = with else {
                return Err(AssetError::BadPackMeta(
                    "invalid \"description\" chat component",
                ));
            };
            for argument in args {
                let mut ignored = String::new();
                format_component(argument, DescriptionStyle::default(), &mut ignored)?;
            }
        }
        gson_primitive_string(translate)?
    } else if let Some(score) = object.get("score") {
        let Value::Object(score) = score else {
            return Err(AssetError::BadPackMeta(
                "invalid \"description\" chat component",
            ));
        };
        let (Some(name), Some(objective)) = (score.get("name"), score.get("objective")) else {
            return Err(AssetError::BadPackMeta(
                "invalid \"description\" chat component",
            ));
        };
        // `ChatComponentScore` validates name/objective even though a disconnected client shows
        // its supplied value (or an empty string) rather than resolving a scoreboard score.
        let _ = gson_primitive_string(name)?;
        let _ = gson_primitive_string(objective)?;
        score
            .get("value")
            .map(gson_primitive_string)
            .transpose()?
            .unwrap_or_default()
    } else if let Some(selector) = object.get("selector") {
        gson_primitive_string(selector)?
    } else {
        return Err(AssetError::BadPackMeta(
            "invalid \"description\" chat component",
        ));
    };

    let style = parse_description_style(object, parent)?;
    append_formatted(out, style, &text);

    if let Some(extra) = object.get("extra") {
        let Value::Array(items) = extra else {
            return Err(AssetError::BadPackMeta(
                "invalid \"description\" chat component",
            ));
        };
        if items.is_empty() {
            return Err(AssetError::BadPackMeta(
                "invalid \"description\" chat component",
            ));
        }
        for item in items {
            format_component(item, style, out)?;
        }
    }
    Ok(())
}

fn parse_description_style(
    object: &serde_json::Map<String, Value>,
    mut style: DescriptionStyle,
) -> Result<DescriptionStyle, AssetError> {
    if let Some(color) = object.get("color") {
        style.color = match gson_primitive_string(color)?.as_str() {
            "black" => Some('0'),
            "dark_blue" => Some('1'),
            "dark_green" => Some('2'),
            "dark_aqua" => Some('3'),
            "dark_red" => Some('4'),
            "dark_purple" => Some('5'),
            "gold" => Some('6'),
            "gray" => Some('7'),
            "dark_gray" => Some('8'),
            "blue" => Some('9'),
            "green" => Some('a'),
            "aqua" => Some('b'),
            "red" => Some('c'),
            "light_purple" => Some('d'),
            "yellow" => Some('e'),
            "white" => Some('f'),
            // Gson's enum adapter accepts `reset`, and ChatStyle does not filter it out before
            // appending the colour slot.  It is therefore an explicit `§r`, not “no colour”.
            "reset" => Some('r'),
            _ => style.color,
        };
    }
    for (name, destination) in [
        ("bold", &mut style.bold),
        ("italic", &mut style.italic),
        ("underlined", &mut style.underlined),
        ("strikethrough", &mut style.strikethrough),
        ("obfuscated", &mut style.obfuscated),
    ] {
        if let Some(value) = object.get(name) {
            *destination = gson_primitive_string(value)? == "true";
        }
    }
    // These have no effect on a pack-list row, but Gson still parses their typed members; an
    // invalid one makes the enclosing description invalid.
    if let Some(insertion) = object.get("insertion") {
        let _ = gson_primitive_string(insertion)?;
    }
    validate_description_events(object)?;
    Ok(style)
}

fn validate_description_events(object: &serde_json::Map<String, Value>) -> Result<(), AssetError> {
    if let Some(click) = object.get("clickEvent") {
        let Value::Object(click) = click else {
            return Err(AssetError::BadPackMeta(
                "invalid \"description\" chat component",
            ));
        };
        if let Some(action) = click.get("action") {
            let _ = gson_primitive_string(action)?;
        }
        if let Some(value) = click.get("value") {
            let _ = gson_primitive_string(value)?;
        }
    }
    if let Some(hover) = object.get("hoverEvent") {
        let Value::Object(hover) = hover else {
            return Err(AssetError::BadPackMeta(
                "invalid \"description\" chat component",
            ));
        };
        if let Some(action) = hover.get("action") {
            let _ = gson_primitive_string(action)?;
        }
        if let Some(value) = hover.get("value") {
            let mut ignored = String::new();
            format_component(value, DescriptionStyle::default(), &mut ignored)?;
        }
    }
    Ok(())
}

fn append_formatted(out: &mut String, style: DescriptionStyle, text: &str) {
    out.push_str(&style.formatting_code());
    out.push_str(text);
    out.push_str("\u{a7}r");
}

/// Reads at most `limit` bytes, erroring rather than truncating if there are more.
///
/// This is the zip-bomb guard: a `.zip` pack can declare a small compressed size and expand
/// to gigabytes, so every read through this crate is capped.
pub(crate) fn read_capped(
    mut r: impl Read,
    limit: u64,
    what: impl Into<String>,
) -> Result<Vec<u8>, AssetError> {
    let mut buf = Vec::new();
    let read = (&mut r)
        .take(limit.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| AssetError::Io {
            path: "<resource>".to_owned(),
            source: e,
        })?;
    if read as u64 > limit {
        return Err(AssetError::TooLarge {
            what: what.into(),
            limit,
        });
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_vanilla_shape_with_get_formatted_text_reset() {
        let m = PackMetadata::parse(
            br#"{"pack":{"pack_format":1,"description":"The default look of Minecraft"}}"#,
        )
        .unwrap();
        assert_eq!(m.pack_format, 1);
        assert_eq!(m.description, "The default look of Minecraft\u{a7}r");
        assert!(m.is_supported());
        assert!(m.require_supported().is_ok());
    }

    #[test]
    fn a_newer_pack_format_is_a_typed_error_not_a_panic() {
        let m = PackMetadata::parse(br#"{"pack":{"pack_format":9,"description":"x"}}"#).unwrap();
        assert!(!m.is_supported());
        assert!(matches!(
            m.require_supported(),
            Err(AssetError::PackFormat {
                found: 9,
                expected: 1
            })
        ));
    }

    #[test]
    fn a_chat_component_description_keeps_vanilla_formatting_and_inheritance() {
        let m = PackMetadata::parse(
            br#"{"pack":{"pack_format":1,"description":{"text":"Hi ","color":"gold","bold":true,"extra":[{"text":"there"},{"text":"!","bold":false}]}}}"#,
        )
        .unwrap();
        assert_eq!(
            m.description,
            "\u{a7}6\u{a7}lHi \u{a7}r\u{a7}6\u{a7}lthere\u{a7}r\u{a7}6!\u{a7}r"
        );
    }

    #[test]
    fn malformed_metadata_is_an_error() {
        assert!(PackMetadata::parse(b"nonsense").is_err());
        assert!(matches!(
            PackMetadata::parse(br#"{"nope":1}"#),
            Err(AssetError::BadPackMeta(_))
        ));
        assert!(matches!(
            PackMetadata::parse(br#"{"pack":{"description":"x"}}"#),
            Err(AssetError::BadPackMeta(_))
        ));
    }

    #[test]
    fn missing_null_or_structurally_invalid_descriptions_are_errors() {
        // Biting controls: each shape was previously accepted as an empty or partial string,
        // whereas `IChatComponent.Serializer` / `PackMetadataSectionSerializer` rejects it.
        let bad_descriptions: &[&[u8]] = &[
            br#"{"pack":{"pack_format":1}}"#,
            br#"{"pack":{"pack_format":1,"description":null}}"#,
            br#"{"pack":{"pack_format":1,"description":[]}}"#,
            br#"{"pack":{"pack_format":1,"description":{}}}"#,
            br#"{"pack":{"pack_format":1,"description":{"text":"x","extra":[]}}}"#,
            br#"{"pack":{"pack_format":1,"description":{"text":{}}}}"#,
        ];
        for bad in bad_descriptions {
            assert!(matches!(
                PackMetadata::parse(bad),
                Err(AssetError::BadPackMeta(_))
            ));
        }
    }

    #[test]
    fn description_primitives_match_gsons_get_as_string() {
        let numeric =
            PackMetadata::parse(br#"{"pack":{"pack_format":1,"description":7}}"#).unwrap();
        assert_eq!(numeric.description, "7\u{a7}r");

        let boolean =
            PackMetadata::parse(br#"{"pack":{"pack_format":1,"description":true}}"#).unwrap();
        assert_eq!(boolean.description, "true\u{a7}r");
    }

    #[test]
    fn reset_is_a_real_gson_colour_value_not_an_absent_style() {
        let metadata = PackMetadata::parse(
            br#"{"pack":{"pack_format":1,"description":{"text":"a","color":"gold","extra":[{"text":"b","color":"reset"}]}}}"#,
        )
        .unwrap();
        assert_eq!(metadata.description, "\u{a7}6a\u{a7}r\u{a7}rb\u{a7}r");
    }

    #[test]
    fn read_capped_rejects_oversize() {
        let data = [0u8; 100];
        assert_eq!(read_capped(&data[..], 100, "t").unwrap().len(), 100);
        assert!(matches!(
            read_capped(&data[..], 99, "t"),
            Err(AssetError::TooLarge { .. })
        ));
    }
}
