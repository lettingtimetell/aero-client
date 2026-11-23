//! `ResourceLocation` — the namespaced key every asset is addressed by.
//!
// java: ResourceLocation.java

use std::fmt;
use std::str::FromStr;

use crate::AssetError;

/// The namespace assumed when a location string carries no `namespace:` prefix.
pub const DEFAULT_NAMESPACE: &str = "minecraft";

/// A namespaced asset key, e.g. `minecraft:textures/blocks/stone.png`.
///
/// This is the map key for every asset table downstream (models, sprites, sounds, lang), so
/// it derives `Hash`, `Eq` and `Ord`. Cloning allocates two `String`s; hot loops should hold
/// a reference or key on an interned index rather than cloning per lookup.
///
/// Parsing and [`Display`](fmt::Display) round-trip: `ResourceLocation::parse(&l.to_string())`
/// always yields `l` again for any location this crate will construct.
///
/// # Examples
/// ```
/// use mc_assets::ResourceLocation;
///
/// let l = ResourceLocation::parse("textures/blocks/stone.png").unwrap();
/// assert_eq!(l.namespace, "minecraft");
/// assert_eq!(l.to_string(), "minecraft:textures/blocks/stone.png");
/// ```
// java: ResourceLocation.java:5
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceLocation {
    /// The namespace ("resource domain" in the Java source). Always lower-case, never empty.
    pub namespace: String,
    /// The path within the namespace, e.g. `lang/en_US.lang`. Case is preserved — vanilla
    /// language files really are named `en_US.lang`.
    pub path: String,
}

impl ResourceLocation {
    /// Builds a location from an explicit namespace and path.
    ///
    /// The namespace is lower-cased (vanilla does the same); an empty namespace becomes
    /// [`DEFAULT_NAMESPACE`]. No validation is performed — use
    /// [`ResourceLocation::is_safe`] before turning an untrusted location into a filesystem or
    /// zip-entry path.
    // java: ResourceLocation.java:10
    pub fn new(namespace: impl AsRef<str>, path: impl Into<String>) -> Self {
        let namespace = namespace.as_ref();
        let namespace = if namespace.is_empty() {
            DEFAULT_NAMESPACE.to_owned()
        } else {
            namespace.to_lowercase()
        };
        Self {
            namespace,
            path: path.into(),
        }
    }

    /// Builds a location in the `minecraft` namespace.
    pub fn minecraft(path: impl Into<String>) -> Self {
        Self {
            namespace: DEFAULT_NAMESPACE.to_owned(),
            path: path.into(),
        }
    }

    /// Parses `"namespace:path"`, or `"path"` for the `minecraft` namespace.
    ///
    /// This is deliberately permissive, just like vanilla 1.8.9: every Rust `&str` is accepted.
    /// `splitObjectName` splits at the first colon, but only uses the part before it as a
    /// namespace when that colon is past index 1. Thus `"a:b"` becomes `minecraft:b`, while
    /// `"ab:c"` becomes `ab:c`. The namespace alone is lower-cased.
    ///
    /// Safety is a separate concern. Resource packs call [`ResourceLocation::is_safe`] before
    /// an accepted location can become a filesystem or archive path.
    // java: ResourceLocation.java:31
    pub fn parse(s: &str) -> Result<Self, AssetError> {
        // java: ResourceLocation.splitObjectName. It always removes the prefix through the
        // first colon, but has the historical `i > 1` guard before accepting that prefix as a
        // resource domain. Keep this oddity: packet sound names can reach this constructor.
        let (namespace, path) = match s.find(':') {
            Some(i) if i > 1 => (&s[..i], &s[i + 1..]),
            Some(i) => (DEFAULT_NAMESPACE, &s[i + 1..]),
            None => (DEFAULT_NAMESPACE, s),
        };
        Ok(Self::new(namespace, path))
    }

    /// Whether this location is safe to turn into a filesystem or zip-entry path.
    ///
    /// Parsing intentionally does not enforce this — vanilla accepts arbitrary strings — and
    /// the fields are public, so packs must check before touching the disk.
    pub fn is_safe(&self) -> bool {
        if self.namespace.is_empty() || self.path.is_empty() {
            return false;
        }
        if self
            .namespace
            .chars()
            .any(|c| c == '/' || c == '\\' || c == ':' || c.is_control())
        {
            return false;
        }
        if self.path.starts_with('/') || self.path.ends_with('/') {
            return false;
        }
        if self
            .path
            .chars()
            .any(|c| c == '\\' || c == ':' || c.is_control())
        {
            return false;
        }
        self.path
            .split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
    }

    /// The entry name this location has inside a zip or folder pack: `assets/<ns>/<path>`.
    // java: AbstractResourcePack.java:32 (locationToName)
    pub fn pack_entry_name(&self) -> String {
        format!("assets/{}/{}", self.namespace, self.path)
    }

    /// Returns this location with `suffix` appended to the path.
    ///
    /// Used for the `.mcmeta` sidecar lookup, which is how animated textures and font
    /// metadata are found.
    // java: FallbackResourceManager.java:97 (getLocationMcmeta)
    pub fn with_suffix(&self, suffix: &str) -> Self {
        Self {
            namespace: self.namespace.clone(),
            path: format!("{}{}", self.path, suffix),
        }
    }

    /// Returns this location with `prefix` prepended to the path.
    ///
    /// S09 needs this to turn a model's `blocks/stone` texture reference into
    /// `textures/blocks/stone.png`.
    pub fn with_prefix(&self, prefix: &str) -> Self {
        Self {
            namespace: self.namespace.clone(),
            path: format!("{}{}", prefix, self.path),
        }
    }
}

/// Whether Windows opens a device rather than a file for a path part
/// called `name`: `CON`, `PRN`, `AUX`, `NUL`, `CONIN$`, `CONOUT$`, `COM0`-`COM9` and
/// `LPT0`-`LPT9` (also with a superscript 1-3), whatever the case, and on Windows 10 whatever
/// follows the first dot (`con.png` is the console there).
#[must_use]
pub fn is_windows_device_name(name: &str) -> bool {
    let base = name.split('.').next().unwrap_or_default().trim_end();
    let upper = base.to_ascii_uppercase();
    let numbered = |prefix: &str| {
        upper.strip_prefix(prefix).is_some_and(|rest| {
            let mut chars = rest.chars();
            matches!(
                (chars.next(), chars.next()),
                (Some('0'..='9' | '\u{b9}' | '\u{b2}' | '\u{b3}'), None)
            )
        })
    };
    matches!(
        upper.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || numbered("COM")
        || numbered("LPT")
}

impl fmt::Display for ResourceLocation {
    // java: ResourceLocation.java:59
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.namespace, self.path)
    }
}

impl FromStr for ResourceLocation {
    type Err = AssetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for ResourceLocation {
    type Error = AssetError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_path_defaults_to_minecraft() {
        let l = ResourceLocation::parse("textures/blocks/stone.png").unwrap();
        assert_eq!(l.namespace, "minecraft");
        assert_eq!(l.path, "textures/blocks/stone.png");
    }

    #[test]
    fn explicit_namespace_is_kept_and_lowercased() {
        let l = ResourceLocation::parse("Realms:lang/en_GB.lang").unwrap();
        assert_eq!(l.namespace, "realms");
        assert_eq!(l.path, "lang/en_GB.lang");
    }

    #[test]
    fn path_case_is_preserved() {
        // The vanilla language files really are `en_US.lang`, not `en_us.lang`.
        let l = ResourceLocation::parse("lang/en_US.lang").unwrap();
        assert_eq!(l.path, "lang/en_US.lang");
    }

    #[test]
    fn empty_namespace_falls_back() {
        assert_eq!(
            ResourceLocation::parse(":pack.png").unwrap(),
            ResourceLocation::minecraft("pack.png")
        );
    }

    #[test]
    fn parse_display_round_trips() {
        for s in [
            "minecraft:textures/blocks/stone.png",
            "realms:lang/en_GB.lang",
            "minecraft:pack.png",
        ] {
            let l = ResourceLocation::parse(s).unwrap();
            assert_eq!(l.to_string(), s);
            assert_eq!(ResourceLocation::parse(&l.to_string()).unwrap(), l);
        }
    }

    #[test]
    fn parse_matches_vanillas_permissive_split_and_default_domain() {
        for (input, namespace, path) in [
            ("", "minecraft", ""),
            (":", "minecraft", ""),
            ("minecraft:", "minecraft", ""),
            ("MINE CRAFT:x", "mine craft", "x"),
            (
                "minecraft:../../../etc/passwd",
                "minecraft",
                "../../../etc/passwd",
            ),
            ("x\\y", "minecraft", "x\\y"),
            ("x//y", "minecraft", "x//y"),
            ("\u{1}", "minecraft", "\u{1}"),
            // Java's `i > 1` guard drops one-character resource domains.
            ("a:b", "minecraft", "b"),
            ("_:x", "minecraft", "x"),
            ("ab:c", "ab", "c"),
            ("ab:c:d", "ab", "c:d"),
        ] {
            let location = ResourceLocation::parse(input).unwrap();
            assert_eq!(location.namespace, namespace, "namespace for {input:?}");
            assert_eq!(location.path, path, "path for {input:?}");
        }
    }

    #[test]
    fn unsafe_locations_parse_but_remain_unsafe_for_packs() {
        for input in [
            "minecraft:../../../etc/passwd",
            "minecraft:/etc/passwd",
            "minecraft:a//b",
            "minecraft:a/./b",
            "minecraft:..",
            "minecraft:textures\\blocks\\stone.png",
            "ab:c:d",
            "\u{1}",
        ] {
            let location = ResourceLocation::parse(input).unwrap();
            assert!(!location.is_safe(), "expected {input:?} to remain unsafe");
        }
    }

    #[test]
    fn direct_construction_is_still_checked_by_is_safe() {
        let evil = ResourceLocation {
            namespace: "minecraft".into(),
            path: "../../secrets".into(),
        };
        assert!(!evil.is_safe());
    }

    #[test]
    fn pack_entry_name_matches_java_layout() {
        let l = ResourceLocation::minecraft("models/block/stone.json");
        assert_eq!(
            l.pack_entry_name(),
            "assets/minecraft/models/block/stone.json"
        );
    }

    #[test]
    fn suffix_and_prefix() {
        let l = ResourceLocation::minecraft("textures/blocks/lava_flow.png");
        assert_eq!(
            l.with_suffix(".mcmeta").to_string(),
            "minecraft:textures/blocks/lava_flow.png.mcmeta"
        );
        assert_eq!(
            ResourceLocation::minecraft("blocks/stone")
                .with_prefix("textures/")
                .with_suffix(".png")
                .to_string(),
            "minecraft:textures/blocks/stone.png"
        );
    }

    #[test]
    fn usable_as_a_hash_key() {
        use std::collections::HashMap;
        let mut m = HashMap::new();
        m.insert(ResourceLocation::minecraft("a/b"), 1u8);
        assert_eq!(m.get(&ResourceLocation::parse("a/b").unwrap()), Some(&1));
    }
}
