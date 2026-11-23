//! A resource pack backed by the launcher's content-addressed object store.
//!
//! The launcher stores every downloaded asset as `assets/objects/<first two hex>/<sha1>` and
//! maps logical paths onto those hashes with `assets/indexes/<version>.json`. This is where
//! sounds, the non-US language files and the icons live — the version jar does not carry
//! them.
//!
// java: DefaultResourcePack.java, ResourceIndex.java

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::index::AssetIndex;
use crate::pack::{PackMetadata, ResourcePack};
use crate::{parse_language_sections, AssetError, LanguageInfo, ResourceLocation};

/// A resource pack that resolves locations through an [`AssetIndex`] into blobs under an
/// `objects/` directory.
#[derive(Debug)]
pub struct ObjectStorePack {
    index: AssetIndex,
    objects_dir: PathBuf,
    name: String,
    metadata: Option<PackMetadata>,
    language_descriptors: Vec<LanguageInfo>,
    max_resource_size: u64,
}

impl ObjectStorePack {
    /// Default per-resource read cap: 64 MiB.
    pub const DEFAULT_MAX_RESOURCE_SIZE: u64 = 64 * 1024 * 1024;

    /// Opens the store rooted at `assets_root` (the directory holding `indexes/` and
    /// `objects/`) using the named index — `"1.8"` for this project.
    ///
    /// # Examples
    /// ```no_run
    /// # use mc_assets::ObjectStorePack;
    /// let pack = ObjectStorePack::open("jars/assets", "1.8")?;
    /// # Ok::<_, mc_assets::AssetError>(())
    /// ```
    // java: ResourceIndex.java:26
    pub fn open(assets_root: impl AsRef<Path>, index_name: &str) -> Result<Self, AssetError> {
        let assets_root = assets_root.as_ref();
        let index_path = assets_root
            .join("indexes")
            .join(format!("{index_name}.json"));
        let index = AssetIndex::load(&index_path)?;
        Ok(Self::with_index(
            index,
            assets_root.join("objects"),
            format!("assets index {index_name}"),
        ))
    }

    /// Builds a store from an already-parsed index and an explicit `objects/` directory.
    pub fn with_index(
        index: AssetIndex,
        objects_dir: impl Into<PathBuf>,
        name: impl Into<String>,
    ) -> Self {
        let objects_dir = objects_dir.into();
        let name = name.into();
        // java: DefaultResourcePack.java:71 — pack.mcmeta is indexed without a namespace.
        let metadata_bytes = index
            .get_raw("pack.mcmeta")
            .map(|o| o.path_in(&objects_dir))
            .and_then(|p| std::fs::read(p).ok());
        let metadata =
            metadata_bytes
                .as_deref()
                .and_then(|bytes| match PackMetadata::parse(bytes) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        tracing::warn!(pack = %name, error = %e, "ignoring malformed pack.mcmeta");
                        None
                    }
                });
        let language_descriptors = metadata_bytes
            .as_deref()
            .map(parse_language_sections)
            .transpose()
            .unwrap_or_else(|e| {
                tracing::warn!(pack = %name, error = %e, "ignoring malformed language metadata");
                None
            })
            .unwrap_or_default();

        Self {
            index,
            objects_dir,
            name,
            metadata,
            language_descriptors,
            max_resource_size: Self::DEFAULT_MAX_RESOURCE_SIZE,
        }
    }

    /// Overrides the per-resource read cap.
    #[must_use]
    pub fn with_max_resource_size(mut self, bytes: u64) -> Self {
        self.max_resource_size = bytes;
        self
    }

    /// The parsed index behind this pack.
    pub fn index(&self) -> &AssetIndex {
        &self.index
    }

    /// The `objects/` directory blobs are read from.
    pub fn objects_dir(&self) -> &Path {
        &self.objects_dir
    }

    /// Reads the blob for a logical index key that has no namespace (`pack.mcmeta`).
    pub fn read_raw(&self, key: &str) -> Result<Vec<u8>, AssetError> {
        let obj = self
            .index
            .get_raw(key)
            .ok_or_else(|| AssetError::NotFound(ResourceLocation::minecraft(key)))?;
        let path = obj.path_in(&self.objects_dir);
        std::fs::read(&path).map_err(|e| AssetError::io(&path, e))
    }

    /// Path of the blob backing a location, if the index carries it.
    ///
    /// The blob may still be absent from disk — a launcher install can be partial.
    pub fn blob_path(&self, l: &ResourceLocation) -> Option<PathBuf> {
        self.index.get(l).map(|o| o.path_in(&self.objects_dir))
    }
}

impl ResourcePack for ObjectStorePack {
    // java: DefaultResourcePack.java:50 (getInputStreamAssets)
    fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
        let path = self.blob_path(l)?;
        match File::open(&path) {
            Ok(f) => Some(Box::new(f.take(self.max_resource_size))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                tracing::warn!(pack = %self.name, path = %path.display(), error = %e, "failed to open object blob");
                None
            }
        }
    }

    // java: DefaultResourcePack.java:61
    fn contains(&self, l: &ResourceLocation) -> bool {
        // The index can list an object the launcher never downloaded, so `contains` has to
        // check the blob too — otherwise the manager stops falling through to lower packs.
        self.blob_path(l).is_some_and(|p| p.is_file())
    }

    /// The index alone: no disk access. An indexed blob the launcher never downloaded is listed
    /// here and then fails to open.
    fn lists(&self, l: &ResourceLocation) -> Option<bool> {
        Some(self.index.get(l).is_some())
    }

    fn file_path(&self, l: &ResourceLocation) -> Option<PathBuf> {
        self.blob_path(l)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn namespaces(&self) -> Vec<String> {
        // DefaultResourcePack.java:19,66. This is deliberately not derived from the asset
        // index: the launcher index also contains `icons/*`, which is addressable as a
        // resource but is not a resource-pack domain in vanilla.
        vec!["minecraft".to_owned(), "realms".to_owned()]
    }

    fn metadata(&self) -> Option<&PackMetadata> {
        self.metadata.as_ref()
    }

    fn language_descriptors(&self) -> &[LanguageInfo] {
        &self.language_descriptors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "aa00000000000000000000000000000000000000";

    fn index_for_metadata() -> AssetIndex {
        AssetIndex::parse(&format!(
            r#"{{"objects":{{"pack.mcmeta":{{"hash":"{HASH}","size":1}}}}}}"#
        ))
        .expect("test index")
    }

    fn write_metadata(objects: &Path, bytes: &[u8]) {
        let path = objects.join("aa").join(HASH);
        std::fs::create_dir_all(path.parent().expect("object parent")).expect("create objects");
        std::fs::write(path, bytes).expect("write pack.mcmeta object");
    }

    #[test]
    fn malformed_default_pack_metadata_is_none_not_an_open_failure() {
        // `DefaultResourcePack.getPackMetadata` catches RuntimeException and returns null. This
        // control fails if the object-store constructor ever propagates a parse error instead.
        let temp = tempfile::tempdir().expect("tempdir");
        write_metadata(temp.path(), br#"{"pack":{"pack_format":1}}"#);

        let pack = ObjectStorePack::with_index(index_for_metadata(), temp.path(), "test default");
        assert!(pack.metadata().is_none());
    }

    #[test]
    fn valid_default_pack_metadata_is_formatted_like_vanilla() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_metadata(
            temp.path(),
            br#"{"pack":{"pack_format":1,"description":{"text":"Default","color":"gold"}}}"#,
        );

        let pack = ObjectStorePack::with_index(index_for_metadata(), temp.path(), "test default");
        assert_eq!(
            pack.metadata()
                .map(|metadata| metadata.description.as_str()),
            Some("\u{a7}6Default\u{a7}r")
        );
    }

    #[test]
    fn language_metadata_is_retained_and_its_failure_is_section_local() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_metadata(
            temp.path(),
            br#"{"pack":{"pack_format":1,"description":"test"},"language":{"zz_ZZ":{"region":"Test","name":"Synthetic"}}}"#,
        );
        let pack = ObjectStorePack::with_index(index_for_metadata(), temp.path(), "test default");
        assert!(pack.metadata().is_some());
        assert_eq!(pack.language_descriptors()[0].code, "zz_ZZ");

        write_metadata(
            temp.path(),
            br#"{"pack":{"pack_format":1,"description":"test"},"language":{"bad":{}}}"#,
        );
        let pack = ObjectStorePack::with_index(index_for_metadata(), temp.path(), "test default");
        assert!(
            pack.metadata().is_some(),
            "language failure is section-local"
        );
        assert!(pack.language_descriptors().is_empty());
    }

    #[test]
    fn default_pack_domains_are_constant_even_when_the_index_has_icons() {
        // The real 1.8 asset index has all three prefixes below. `icons` is a launcher asset,
        // not a DefaultResourcePack domain; deriving domains from this index was the defect.
        let index = AssetIndex::parse(&format!(
            r#"{{"objects":{{
                "icons/icon_16x16.png":{{"hash":"{HASH}","size":1}},
                "minecraft/lang/en_us.lang":{{"hash":"{HASH}","size":1}},
                "realms/lang/en_us.lang":{{"hash":"{HASH}","size":1}}
            }}}}"#
        ))
        .expect("test index");
        let derived: Vec<_> = index.namespaces().collect();
        assert_eq!(derived, ["icons", "minecraft", "realms"]);

        let pack = ObjectStorePack::with_index(index, "objects", "test default");
        assert_eq!(
            pack.namespaces(),
            ["minecraft".to_owned(), "realms".to_owned()]
        );
    }
}
