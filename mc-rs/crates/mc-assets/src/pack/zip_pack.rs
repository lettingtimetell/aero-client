//! A resource pack backed by a zip archive — the vanilla `1.8.9.jar` or a user `.zip` pack.
//!
//! User packs are attacker-controlled (a server can push a pack URL at the client), so this
//! module is deliberately paranoid: entry names that could escape the archive root are
//! rejected at open time, entry counts are capped, and every read is bounded so a
//! zip bomb cannot exhaust memory.
//!
// java: FileResourcePack.java

use std::collections::{BTreeSet, HashSet};
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use zip::ZipArchive;

use crate::pack::{read_capped, PackMetadata, ResourcePack};
use crate::{parse_language_sections, AssetError, LanguageInfo, ResourceLocation};

/// Budget applied to an untrusted archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZipLimits {
    /// Largest decompressed size accepted for a single entry. Default 64 MiB, comfortably
    /// above the biggest vanilla asset and far below "chews the machine".
    pub max_entry_size: u64,
    /// Largest number of entries accepted in the archive. Default 262,144; the vanilla jar
    /// has about 5,600.
    pub max_entries: usize,
}

impl Default for ZipLimits {
    fn default() -> Self {
        Self {
            max_entry_size: 64 * 1024 * 1024,
            max_entries: 262_144,
        }
    }
}

/// A resource pack read from a zip archive.
///
/// The archive handle is behind a `Mutex` so the pack is `Sync` and can be shared with the
/// background meshing pool S11 will introduce. Entry names are indexed once at open time, so
/// [`contains`](ResourcePack::contains) never takes the lock.
#[derive(Debug)]
pub struct ZipResourcePack {
    path: PathBuf,
    name: String,
    archive: Mutex<ZipArchive<BufReader<File>>>,
    entries: HashSet<String>,
    namespaces: BTreeSet<String>,
    metadata: Option<PackMetadata>,
    language_descriptors: Vec<LanguageInfo>,
    limits: ZipLimits,
}

impl ZipResourcePack {
    /// Opens a zip archive as a resource pack with the default [`ZipLimits`].
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, AssetError> {
        Self::open_with_limits(path, ZipLimits::default())
    }

    /// Opens a zip archive as a resource pack with explicit limits.
    ///
    /// Fails with [`AssetError::UnsafeZipEntry`] if the archive holds an entry name that
    /// could escape its root, or if it exceeds `limits.max_entries`. A missing or malformed
    /// `pack.mcmeta` is *not* fatal — the vanilla `1.8.9.jar` has none — but a present one
    /// that declares an unreadable `pack_format` is surfaced by
    /// [`PackMetadata::require_supported`], which callers should invoke for user packs.
    pub fn open_with_limits(
        path: impl Into<PathBuf>,
        limits: ZipLimits,
    ) -> Result<Self, AssetError> {
        let path = path.into();
        let file = File::open(&path).map_err(|e| AssetError::io(&path, e))?;
        let mut archive = ZipArchive::new(BufReader::new(file)).map_err(|e| AssetError::Zip {
            path: path.display().to_string(),
            source: e,
        })?;

        if archive.len() > limits.max_entries {
            return Err(AssetError::UnsafeZipEntry {
                path: path.display().to_string(),
                entry: format!("<{} entries>", archive.len()),
                reason: "archive exceeds the entry-count limit",
            });
        }

        let mut entries = HashSet::with_capacity(archive.len());
        let mut namespaces = BTreeSet::new();
        for raw in archive.file_names() {
            if let Some(reason) = unsafe_entry_reason(raw) {
                return Err(AssetError::UnsafeZipEntry {
                    path: path.display().to_string(),
                    entry: raw.to_owned(),
                    reason,
                });
            }
            // java: FileResourcePack.java:84 — namespaces come from assets/<ns>/...
            if let Some(rest) = raw.strip_prefix("assets/") {
                // Guava Splitter.omitEmptyStrings().limit(3) sees explicit directory
                // entries as well as files, so `assets/foo/` and even the unusual
                // `assets/foo` both declare `foo` in Java. Do this before the directory
                // marker filter; the directory itself is not a readable resource, but its
                // namespace still participates in Locale's domain scan.
                if let Some(ns) = rest.split('/').find(|part| !part.is_empty()) {
                    namespaces.insert(ns.to_owned());
                }
            }
            if raw.ends_with('/') {
                continue; // directory marker
            }
            entries.insert(raw.to_owned());
        }

        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());

        // java: AbstractResourcePack.logNameNotLowercase
        let bad: Vec<String> = namespaces
            .iter()
            .filter(|ns| **ns != ns.to_lowercase())
            .cloned()
            .collect();
        for ns in &bad {
            tracing::warn!(pack = %name, namespace = %ns, "ignoring non-lowercase namespace");
            namespaces.remove(ns);
        }

        // Metadata is limited to what metadata can need.
        let mcmeta_limits = ZipLimits {
            max_entry_size: limits.max_entry_size.min(crate::pack::MAX_MCMETA_BYTES),
            ..limits
        };
        let metadata_bytes = entries.contains("pack.mcmeta").then(|| {
            read_entry(&mut archive, "pack.mcmeta", &mcmeta_limits, &path).unwrap_or_else(|e| {
                tracing::warn!(pack = %name, error = %e, "ignoring unreadable pack.mcmeta");
                Vec::new()
            })
        });
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

        Ok(Self {
            path,
            name,
            archive: Mutex::new(archive),
            entries,
            namespaces,
            metadata,
            language_descriptors,
            limits,
        })
    }

    /// The archive's path on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of usable (non-directory) entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Whether the archive holds this raw entry name (`assets/minecraft/...`, `pack.png`).
    pub fn has_entry(&self, name: &str) -> bool {
        self.entries.contains(name)
    }

    /// Reads a raw entry by its in-archive name, bounded by [`ZipLimits::max_entry_size`].
    ///
    /// Needed for the handful of files that live outside `assets/` and therefore have no
    /// [`ResourceLocation`]: `pack.mcmeta`, `pack.png`.
    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>, AssetError> {
        if !self.entries.contains(name) {
            return Err(AssetError::NotFound(ResourceLocation::minecraft(name)));
        }
        // Metadata is limited to what metadata can need.
        let limits = if name.ends_with(".mcmeta") {
            ZipLimits {
                max_entry_size: self
                    .limits
                    .max_entry_size
                    .min(crate::pack::MAX_MCMETA_BYTES),
                ..self.limits
            }
        } else {
            self.limits
        };
        let mut archive = self.archive.lock().unwrap_or_else(PoisonError::into_inner);
        read_entry(&mut archive, name, &limits, &self.path)
    }
}

impl ResourcePack for ZipResourcePack {
    // java: FileResourcePack.java:37
    fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
        if !self.contains(l) {
            return None;
        }
        match self.read_raw(&l.pack_entry_name()) {
            Ok(bytes) => Some(Box::new(Cursor::new(bytes))),
            Err(e) => {
                tracing::warn!(pack = %self.name, location = %l, error = %e, "failed to read zip entry");
                None
            }
        }
    }

    // java: FileResourcePack.java:52
    fn contains(&self, l: &ResourceLocation) -> bool {
        l.is_safe() && self.entries.contains(&l.pack_entry_name())
    }

    /// The entry list read at open time. `open` would inflate the whole entry.
    fn lists(&self, l: &ResourceLocation) -> Option<bool> {
        Some(self.contains(l))
    }

    fn name(&self) -> &str {
        &self.name
    }

    // java: FileResourcePack.java:64
    fn namespaces(&self) -> Vec<String> {
        self.namespaces.iter().cloned().collect()
    }

    fn metadata(&self) -> Option<&PackMetadata> {
        self.metadata.as_ref()
    }

    fn language_descriptors(&self) -> &[LanguageInfo] {
        &self.language_descriptors
    }

    // Connected-textures rule discovery. Entry names were validated
    // and indexed at open time, so this never touches the archive.
    fn list_files(&self, namespace: &str, dir_prefix: &str) -> Vec<String> {
        if !crate::pack::listing_is_safe(namespace, dir_prefix) {
            return Vec::new();
        }
        let root = format!("assets/{namespace}/");
        let wanted = format!("{root}{dir_prefix}");
        self.entries
            .iter()
            .filter(|name| name.starts_with(&wanted))
            .map(|name| name[root.len()..].to_owned())
            .collect()
    }
}

/// Reads one entry, refusing anything that decompresses past the cap.
fn read_entry(
    archive: &mut ZipArchive<BufReader<File>>,
    name: &str,
    limits: &ZipLimits,
    path: &Path,
) -> Result<Vec<u8>, AssetError> {
    let entry = archive.by_name(name).map_err(|e| AssetError::Zip {
        path: path.display().to_string(),
        source: e,
    })?;
    // Cheap rejection first: the central directory's declared size. An honest archive is
    // refused here without decompressing a byte; a lying one is caught by `read_capped`.
    if entry.size() > limits.max_entry_size {
        return Err(AssetError::TooLarge {
            what: format!("{}!{name}", path.display()),
            limit: limits.max_entry_size,
        });
    }
    read_capped(
        entry,
        limits.max_entry_size,
        format!("{}!{name}", path.display()),
    )
}

/// Why an entry name is unsafe, or `None` if it is fine.
///
/// The archive is only ever read by exact name, so traversal is not directly exploitable
/// today — but any future code that extracts a pack to disk would inherit these names, and
/// the check is nearly free at open time.
fn unsafe_entry_reason(name: &str) -> Option<&'static str> {
    if name.is_empty() {
        return Some("empty entry name");
    }
    if name.starts_with('/') || name.starts_with('\\') {
        return Some("absolute path");
    }
    if name.contains('\\') {
        return Some("backslash path separator");
    }
    if name.contains(':') {
        return Some("drive letter or alternate data stream");
    }
    if name.chars().any(char::is_control) {
        return Some("control character in entry name");
    }
    if name.split('/').any(|seg| seg == ".." || seg == ".") {
        return Some("path traversal segment");
    }
    None
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use zip::write::SimpleFileOptions;
    use zip::{CompressionMethod, ZipWriter};

    use super::*;

    /// Writes a throwaway zip into the temp dir and returns its path.
    fn temp_zip(tag: &str, entries: &[(&str, &[u8])]) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("mc-assets-s04-{}-{tag}.zip", std::process::id()));
        let file = File::create(&path).expect("create temp zip");
        let mut w = ZipWriter::new(file);
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        for (name, data) in entries {
            w.start_file(*name, opts).expect("start entry");
            w.write_all(data).expect("write entry");
        }
        w.finish().expect("finish zip");
        path
    }

    #[test]
    fn a_traversing_entry_makes_the_whole_pack_unusable() {
        // A hostile pack is refused outright rather than partially trusted.
        let path = temp_zip(
            "evil",
            &[("assets/minecraft/x.txt", b"ok"), ("../../evil", b"no")],
        );
        let err = ZipResourcePack::open(&path).expect_err("traversal must be rejected");
        assert!(
            matches!(err, AssetError::UnsafeZipEntry { .. }),
            "unexpected error: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_decompression_bomb_is_capped() {
        let bomb = vec![0u8; 1 << 20]; // 1 MiB of zeroes, a few hundred bytes compressed
        let path = temp_zip("bomb", &[("assets/minecraft/bomb.bin", &bomb)]);

        let limits = ZipLimits {
            max_entry_size: 4096,
            ..ZipLimits::default()
        };
        let pack = ZipResourcePack::open_with_limits(&path, limits).expect("open bomb zip");
        let l = ResourceLocation::minecraft("bomb.bin");

        assert!(
            pack.contains(&l),
            "the entry exists, it is just too big to read"
        );
        assert!(matches!(
            pack.read_raw(&l.pack_entry_name()),
            Err(AssetError::TooLarge { .. })
        ));
        // The frozen `open` signature has no error channel, so an over-budget entry reads
        // as absent rather than as a panic.
        assert!(pack.open(&l).is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_ordinary_pack_round_trips() {
        let path = temp_zip(
            "good",
            &[
                (
                    "pack.mcmeta",
                    br#"{"pack":{"pack_format":1,"description":"t"},"language":{"zz_ZZ":{"region":"Test","name":"Synthetic"}}}"#,
                ),
                ("assets/minecraft/models/block/x.json", b"{}"),
                ("assets/realms/lang/en_US.lang", b"a=b"),
            ],
        );
        let pack = ZipResourcePack::open(&path).expect("open zip");

        assert_eq!(
            pack.namespaces(),
            vec!["minecraft".to_owned(), "realms".to_owned()]
        );
        assert_eq!(pack.metadata().map(|m| m.pack_format), Some(1));
        assert!(pack.metadata().is_some_and(|m| m.is_supported()));
        assert_eq!(pack.language_descriptors()[0].code, "zz_ZZ");

        let l = ResourceLocation::minecraft("models/block/x.json");
        assert!(pack.contains(&l));
        let mut buf = Vec::new();
        pack.open(&l)
            .expect("open entry")
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(buf, b"{}");

        assert!(!pack.contains(&ResourceLocation::minecraft("models/block/missing.json")));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn explicit_directory_entries_declare_java_resource_domains() {
        // FileResourcePack's Guava splitter examines directory entries too: a pack that
        // contains only `assets/dironly/` still reports `dironly` from getResourceDomains.
        let path = temp_zip(
            "directory-domains",
            &[
                ("assets/dironly/", b""),
                ("assets/UpperCase/", b""),
                ("assets/dironly/lang/", b""),
            ],
        );
        let pack = ZipResourcePack::open(&path).expect("open directory-only zip");
        assert_eq!(pack.namespaces(), vec!["dironly".to_owned()]);
        assert_eq!(pack.entry_count(), 0, "directory markers are not resources");
        assert!(!pack.contains(&ResourceLocation::new("dironly", "lang")));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn malformed_language_metadata_does_not_discard_valid_pack_metadata() {
        let path = temp_zip(
            "bad-language",
            &[(
                "pack.mcmeta",
                br#"{"pack":{"pack_format":1,"description":"t"},"language":{"bad":{}}}"#,
            )],
        );
        let pack = ZipResourcePack::open(&path).expect("open zip");
        assert!(pack.metadata().is_some());
        assert!(pack.language_descriptors().is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_unsafe_location_never_reaches_the_archive() {
        let path = temp_zip("safe", &[("assets/minecraft/a.txt", b"a")]);
        let pack = ZipResourcePack::open(&path).expect("open zip");
        let evil = ResourceLocation {
            namespace: "minecraft".into(),
            path: "../../../../../../etc/passwd".into(),
        };
        assert!(!pack.contains(&evil));
        assert!(pack.open(&evil).is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn traversal_names_are_recognised() {
        for bad in [
            "../evil",
            "assets/../../evil",
            "/etc/passwd",
            "C:/windows/system32",
            "assets\\minecraft\\x.png",
            "assets/./x",
            "",
        ] {
            assert!(
                unsafe_entry_reason(bad).is_some(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn ordinary_names_are_accepted() {
        for good in [
            "pack.mcmeta",
            "pack.png",
            "assets/minecraft/models/block/stone.json",
            "assets/minecraft/lang/en_US.lang",
        ] {
            assert_eq!(
                unsafe_entry_reason(good),
                None,
                "{good:?} should be allowed"
            );
        }
    }

    #[test]
    fn list_files_filters_the_indexed_entries_by_prefix() {
        let path = temp_zip(
            "listing",
            &[
                (
                    "assets/minecraft/mcpatcher/ctm/glass/glass.properties",
                    b"x",
                ),
                ("assets/minecraft/mcpatcher/ctm/glass black/0.png", b"x"),
                ("assets/minecraft/mcpatcher/ctm/", b""),
                ("assets/minecraft/textures/blocks/glass.png", b"x"),
            ],
        );
        let pack = ZipResourcePack::open(&path).expect("open zip");
        let mut listed = pack.list_files("minecraft", "mcpatcher/ctm/");
        listed.sort();
        assert_eq!(
            listed,
            [
                "mcpatcher/ctm/glass black/0.png",
                "mcpatcher/ctm/glass/glass.properties"
            ]
        );
        assert!(pack.list_files("realms", "mcpatcher/ctm/").is_empty());
        assert!(pack.list_files("minecraft", "../").is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn default_limits_are_sane() {
        let l = ZipLimits::default();
        assert!(l.max_entry_size >= 16 * 1024 * 1024);
        assert!(l.max_entries > 5_600); // the vanilla jar's entry count
    }
}
