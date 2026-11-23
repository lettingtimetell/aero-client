//! The launcher asset index (`assets/indexes/1.8.json`) → logical path map.
//!
// java: ResourceIndex.java

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{AssetError, ResourceLocation};

/// One entry of the asset index: the SHA-1 of a blob in `assets/objects/` plus its size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetObject {
    /// Lower-case 40-character hex SHA-1. Validated on parse, because it is concatenated
    /// into a filesystem path.
    pub hash: String,
    /// Size in bytes as declared by the index.
    pub size: u64,
}

impl AssetObject {
    /// The blob's path relative to the `objects/` directory: `<first two hex chars>/<hash>`.
    // java: ResourceIndex.java:49
    pub fn relative_path(&self) -> PathBuf {
        let mut p = PathBuf::from(&self.hash[..2]);
        p.push(&self.hash);
        p
    }

    /// The blob's absolute path given the `objects/` directory.
    pub fn path_in(&self, objects_dir: &Path) -> PathBuf {
        objects_dir.join(self.relative_path())
    }
}

/// A parsed asset index.
///
/// Keys follow vanilla's transform: the index's `minecraft/lang/en_GB.lang` becomes the
/// location string `minecraft:lang/en_GB.lang`, while a key with no `/` (there is exactly
/// one — `pack.mcmeta`) is kept verbatim and is reachable only via
/// [`AssetIndex::get_raw`].
#[derive(Debug, Clone, Default)]
pub struct AssetIndex {
    objects: BTreeMap<String, AssetObject>,
}

#[derive(Deserialize)]
struct RawIndex {
    objects: BTreeMap<String, RawObject>,
}

#[derive(Deserialize)]
struct RawObject {
    hash: String,
    #[serde(default)]
    size: u64,
}

impl AssetIndex {
    /// Parses an index from JSON text.
    // java: ResourceIndex.java:26
    pub fn parse(json: &str) -> Result<Self, AssetError> {
        let raw: RawIndex =
            serde_json::from_str(json).map_err(|e| AssetError::json("asset index", e))?;
        let mut objects = BTreeMap::new();

        for (key, obj) in raw.objects {
            if !is_sha1_hex(&obj.hash) {
                return Err(AssetError::BadAssetIndex(format!(
                    "entry {key:?} has hash {:?}, expected 40 lower-case hex characters",
                    obj.hash
                )));
            }
            if key.is_empty()
                || key.starts_with('/')
                || key.split('/').any(|s| s == ".." || s == ".")
            {
                return Err(AssetError::BadAssetIndex(format!(
                    "entry {key:?} is not a valid logical path"
                )));
            }
            // "minecraft/lang/en_GB.lang" -> "minecraft:lang/en_GB.lang";
            // "pack.mcmeta" (no slash) is left alone.
            let logical = match key.split_once('/') {
                Some((domain, rest)) => format!("{domain}:{rest}"),
                None => key,
            };
            objects.insert(
                logical,
                AssetObject {
                    hash: obj.hash,
                    size: obj.size,
                },
            );
        }

        Ok(Self { objects })
    }

    /// Reads and parses an index file, e.g. `jars/assets/indexes/1.8.json`.
    pub fn load(path: &Path) -> Result<Self, AssetError> {
        let text = fs::read_to_string(path).map_err(|e| AssetError::io(path, e))?;
        Self::parse(&text)
    }

    /// Number of objects in the index.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Looks an object up by resource location.
    pub fn get(&self, location: &ResourceLocation) -> Option<&AssetObject> {
        self.objects.get(&location.to_string())
    }

    /// Looks an object up by its logical key, for the handful of entries that have no
    /// namespace (`pack.mcmeta`).
    pub fn get_raw(&self, key: &str) -> Option<&AssetObject> {
        self.objects.get(key)
    }

    /// Whether the index carries this location.
    pub fn contains(&self, location: &ResourceLocation) -> bool {
        self.objects.contains_key(&location.to_string())
    }

    /// Iterates `(logical key, object)` pairs in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &AssetObject)> {
        self.objects.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The set of namespaces this index provides, derived from the keys.
    pub fn namespaces(&self) -> impl Iterator<Item = &str> {
        self.objects
            .keys()
            .filter_map(|k| k.split_once(':').map(|(ns, _)| ns))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
    }
}

fn is_sha1_hex(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "objects": {
            "minecraft/lang/en_GB.lang": { "hash": "03f31164d234f10a3230611656332f1756e570a9", "size": 117031 },
            "pack.mcmeta":               { "hash": "a1ee4b8b62d984d924e14f40855c3797a4c5c18a", "size": 11097 },
            "realms/lang/en_GB.lang":    { "hash": "176c14c2ce03e7a57725f7ea973d485b2888983a", "size": 7539 }
        }
    }"#;

    #[test]
    fn keys_are_transformed_like_vanilla() {
        let idx = AssetIndex::parse(SAMPLE).unwrap();
        assert_eq!(idx.len(), 3);

        let l = ResourceLocation::parse("minecraft:lang/en_GB.lang").unwrap();
        assert_eq!(idx.get(&l).unwrap().size, 117_031);

        let r = ResourceLocation::parse("realms:lang/en_GB.lang").unwrap();
        assert!(idx.contains(&r));

        // No slash: not addressable as a location, only as a raw key.
        assert!(idx.get_raw("pack.mcmeta").is_some());
    }

    #[test]
    fn object_path_is_sharded_by_the_first_two_hex_chars() {
        let idx = AssetIndex::parse(SAMPLE).unwrap();
        let o = idx.get_raw("pack.mcmeta").unwrap();
        assert_eq!(
            o.path_in(Path::new("/assets/objects")),
            Path::new("/assets/objects/a1/a1ee4b8b62d984d924e14f40855c3797a4c5c18a")
        );
    }

    #[test]
    fn namespaces_are_derived_from_keys() {
        let idx = AssetIndex::parse(SAMPLE).unwrap();
        let ns: Vec<&str> = idx.namespaces().collect();
        assert!(ns.contains(&"minecraft"));
        assert!(ns.contains(&"realms"));
    }

    #[test]
    fn a_hash_that_is_not_sha1_hex_is_rejected() {
        // A hash is pasted into a filesystem path, so anything that is not 40 hex characters
        // is refused rather than sanitised.
        for bad in [
            r#"{"objects":{"a/b":{"hash":"../../../../etc/passwd","size":1}}}"#,
            r#"{"objects":{"a/b":{"hash":"03F31164D234F10A3230611656332F1756E570A9","size":1}}}"#,
            r#"{"objects":{"a/b":{"hash":"short","size":1}}}"#,
        ] {
            assert!(matches!(
                AssetIndex::parse(bad),
                Err(AssetError::BadAssetIndex(_))
            ));
        }
    }

    #[test]
    fn a_traversing_key_is_rejected() {
        let bad = r#"{"objects":{"../evil":{"hash":"03f31164d234f10a3230611656332f1756e570a9","size":1}}}"#;
        assert!(matches!(
            AssetIndex::parse(bad),
            Err(AssetError::BadAssetIndex(_))
        ));
    }

    #[test]
    fn junk_json_is_an_error_not_a_panic() {
        assert!(AssetIndex::parse("not json").is_err());
        assert!(AssetIndex::parse("{}").is_err());
    }
}
