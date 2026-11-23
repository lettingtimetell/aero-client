//! A resource pack backed by an unpacked directory.
//!
// java: FolderResourcePack.java

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::pack::{read_capped, PackMetadata, ResourcePack};
use crate::{parse_language_sections, AssetError, LanguageInfo, ResourceLocation};

/// A resource pack read straight out of a directory (`<root>/assets/<namespace>/...`).
///
/// Useful for development — point it at an extracted `1.8.9.jar` and edit files in place —
/// and for the `resourcepacks/` folder-style packs vanilla supports.
#[derive(Debug)]
pub struct FolderResourcePack {
    root: PathBuf,
    name: String,
    namespaces: BTreeSet<String>,
    metadata: Option<PackMetadata>,
    language_descriptors: Vec<LanguageInfo>,
    max_resource_size: u64,
}

impl FolderResourcePack {
    /// Default per-resource read cap: 64 MiB.
    pub const DEFAULT_MAX_RESOURCE_SIZE: u64 = 64 * 1024 * 1024;

    /// Opens the directory as a pack.
    ///
    /// Scans `<root>/assets/` once for namespaces and reads `pack.mcmeta` if present. A
    /// missing or malformed `pack.mcmeta` is not fatal (the vanilla jar has none) — it is
    /// logged and [`metadata`](ResourcePack::metadata) stays `None`.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, AssetError> {
        let root = root.into();
        if !root.is_dir() {
            return Err(AssetError::io(
                &root,
                std::io::Error::new(std::io::ErrorKind::NotFound, "not a directory"),
            ));
        }
        let name = root
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| root.display().to_string());

        let mut namespaces = BTreeSet::new();
        let assets = root.join("assets");
        if let Ok(entries) = fs::read_dir(&assets) {
            for entry in entries.flatten() {
                if !entry.path().is_dir() {
                    continue;
                }
                let ns = entry.file_name().to_string_lossy().into_owned();
                // java: AbstractResourcePack.logNameNotLowercase
                if ns != ns.to_lowercase() {
                    tracing::warn!(pack = %name, namespace = %ns, "ignoring non-lowercase namespace");
                    continue;
                }
                namespaces.insert(ns);
            }
        }

        // At most what metadata can need: the
        // folder is someone else's pack, and its `pack.mcmeta` used to be read whole.
        let metadata_bytes = File::open(root.join("pack.mcmeta")).ok().and_then(|file| {
            read_capped(file, crate::pack::MAX_MCMETA_BYTES, "pack.mcmeta")
                .map_err(|e| {
                    tracing::warn!(pack = %name, error = %e, "ignoring unreadable pack.mcmeta");
                })
                .ok()
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
            root,
            name,
            namespaces,
            metadata,
            language_descriptors,
            max_resource_size: Self::DEFAULT_MAX_RESOURCE_SIZE,
        })
    }

    /// Overrides the per-resource read cap.
    #[must_use]
    pub fn with_max_resource_size(mut self, bytes: u64) -> Self {
        self.max_resource_size = bytes;
        self
    }

    /// The directory this pack reads from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolves a location to a file inside the pack, or `None` if the location is unsafe.
    ///
    /// Rejecting unsafe locations here and not only in
    /// [`ResourceLocation::parse`] matters because the struct's fields are public.
    fn resolve(&self, l: &ResourceLocation) -> Option<PathBuf> {
        if !l.is_safe() {
            tracing::warn!(pack = %self.name, location = %l, "refusing unsafe resource location");
            return None;
        }
        // A part named like a Windows device (`con.png` is the console
        // on Windows 10, which the client inherits from its launcher) is never looked up on
        // disk: the folder is someone else's pack. Zip packs read entries by name and keep them.
        if crate::is_windows_device_name(&l.namespace)
            || l.path.split('/').any(crate::is_windows_device_name)
        {
            return None;
        }
        let mut p = self.root.join("assets");
        p.push(&l.namespace);
        for segment in l.path.split('/') {
            p.push(segment);
        }
        Some(p)
    }
}

impl ResourcePack for FolderResourcePack {
    // java: FolderResourcePack.java:20
    fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
        let path = self.resolve(l)?;
        match File::open(&path) {
            // Keep one probe byte so ResourceManager::read_capped can distinguish an
            // over-budget file from a file that ends exactly at the cap.  Taking exactly
            // `max_resource_size` would silently turn a valid Java stream for a larger file
            // into a successful, truncated resource on the Rust path.
            Ok(f) => Some(Box::new(f.take(self.max_resource_size.saturating_add(1)))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                tracing::warn!(pack = %self.name, path = %path.display(), error = %e, "failed to open resource");
                None
            }
        }
    }

    // java: FolderResourcePack.java:25
    fn contains(&self, l: &ResourceLocation) -> bool {
        self.resolve(l).is_some_and(|p| p.is_file())
    }

    fn file_path(&self, l: &ResourceLocation) -> Option<PathBuf> {
        self.resolve(l)
    }

    fn name(&self) -> &str {
        &self.name
    }

    // java: FolderResourcePack.java:30
    fn namespaces(&self) -> Vec<String> {
        self.namespaces.iter().cloned().collect()
    }

    fn metadata(&self) -> Option<&PackMetadata> {
        self.metadata.as_ref()
    }

    fn language_descriptors(&self) -> &[LanguageInfo] {
        &self.language_descriptors
    }

    // Connected-textures rule discovery.
    fn list_files(&self, namespace: &str, dir_prefix: &str) -> Vec<String> {
        /// A hostile folder cannot make a listing walk forever.
        const MAX_DEPTH: usize = 16;
        const MAX_FILES: usize = 65_536;
        if !crate::pack::listing_is_safe(namespace, dir_prefix) {
            return Vec::new();
        }
        let mut start = self.root.join("assets");
        start.push(namespace);
        let mut prefix = String::new();
        for segment in dir_prefix.split('/').filter(|s| !s.is_empty()) {
            start.push(segment);
            prefix.push_str(segment);
            prefix.push('/');
        }
        let mut out = Vec::new();
        let mut stack = vec![(start, prefix, 0usize)];
        while let Some((dir, rel, depth)) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(kind) = entry.file_type() else {
                    continue;
                };
                let name = entry.file_name().to_string_lossy().into_owned();
                if kind.is_dir() {
                    if depth < MAX_DEPTH {
                        stack.push((entry.path(), format!("{rel}{name}/"), depth + 1));
                    }
                } else if kind.is_file() {
                    if out.len() >= MAX_FILES {
                        tracing::warn!(pack = %self.name, "file listing truncated");
                        return out;
                    }
                    out.push(format!("{rel}{name}"));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResourceManager;

    #[test]
    fn list_files_walks_subdirectories_and_keeps_names_verbatim() {
        let temp = tempfile::tempdir().expect("tempdir");
        let ctm = temp.path().join("assets/minecraft/mcpatcher/ctm");
        std::fs::create_dir_all(ctm.join("glass/Black Glass")).expect("dirs");
        std::fs::write(ctm.join("glass/Black Glass/glass_black.properties"), b"x").unwrap();
        std::fs::write(ctm.join("glass/0.png"), b"x").unwrap();
        std::fs::write(temp.path().join("assets/minecraft/other.txt"), b"x").unwrap();
        let pack = FolderResourcePack::open(temp.path()).expect("open folder pack");
        let mut listed = pack.list_files("minecraft", "mcpatcher/ctm/");
        listed.sort();
        assert_eq!(
            listed,
            [
                "mcpatcher/ctm/glass/0.png",
                "mcpatcher/ctm/glass/Black Glass/glass_black.properties"
            ]
        );
        assert!(pack.list_files("minecraft", "../").is_empty());
        assert!(pack.list_files("minecraft", "missing/").is_empty());
    }

    #[test]
    fn language_metadata_is_retained_without_poisoning_pack_metadata() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join("pack.mcmeta"),
            br#"{"pack":{"pack_format":1,"description":"test"},"language":{"zz_ZZ":{"region":"Test","name":"Synthetic","bidirectional":false}}}"#,
        )
        .expect("write metadata");
        let pack = FolderResourcePack::open(temp.path()).expect("open folder pack");
        assert!(pack.metadata().is_some());
        assert_eq!(pack.language_descriptors()[0].code, "zz_ZZ");

        std::fs::write(
            temp.path().join("pack.mcmeta"),
            br#"{"pack":{"pack_format":1,"description":"test"},"language":{"bad":{}}}"#,
        )
        .expect("replace metadata");
        let pack = FolderResourcePack::open(temp.path()).expect("reopen folder pack");
        assert!(
            pack.metadata().is_some(),
            "language failure is section-local"
        );
        assert!(pack.language_descriptors().is_empty());
    }

    #[test]
    fn oversized_files_are_bounded_but_not_silently_accepted() {
        let temp = tempfile::tempdir().expect("tempdir");
        let resource_dir = temp.path().join("assets/minecraft");
        std::fs::create_dir_all(&resource_dir).expect("resource directory");
        std::fs::write(resource_dir.join("large.bin"), b"012345").expect("write resource");

        let location = ResourceLocation::minecraft("large.bin");
        let pack = FolderResourcePack::open(temp.path())
            .expect("open folder pack")
            .with_max_resource_size(4);

        // The reader is bounded, but includes the probe byte needed by the manager to reject
        // the resource instead of returning the first four bytes as if the file ended there.
        let mut bounded = Vec::new();
        pack.open(&location)
            .expect("resource exists")
            .read_to_end(&mut bounded)
            .expect("read bounded resource");
        assert_eq!(bounded, b"01234");

        let mut resources = ResourceManager::new().with_max_resource_size(4);
        resources.push_pack(Box::new(pack));
        assert!(matches!(
            resources.load(&location),
            Err(AssetError::TooLarge { limit: 4, .. })
        ));
    }

    #[test]
    fn manual_unsafe_locations_never_reach_the_filesystem() {
        let temp = tempfile::tempdir().expect("tempdir");
        let outside_dir = tempfile::tempdir_in(temp.path().parent().expect("temp parent"))
            .expect("outside tempdir");
        let resource_dir = temp.path().join("assets/minecraft");
        std::fs::create_dir_all(&resource_dir).expect("resource directory");
        std::fs::write(resource_dir.join("safe.txt"), b"safe").expect("write resource");
        std::fs::write(outside_dir.path().join("outside.txt"), b"outside")
            .expect("write outside resource");

        let pack = FolderResourcePack::open(temp.path()).expect("open folder pack");
        let unsafe_location = ResourceLocation {
            namespace: "minecraft".to_owned(),
            path: format!(
                "../../../{}/outside.txt",
                outside_dir
                    .path()
                    .file_name()
                    .expect("outside directory name")
                    .to_string_lossy()
            ),
        };
        assert!(!unsafe_location.is_safe());
        assert!(!pack.contains(&unsafe_location));
        assert!(pack.open(&unsafe_location).is_none());
    }

    /// A location naming a Windows device is never looked up on disk
    /// (Windows 10 opens the console for `con.png`); ordinary names beside it still load.
    #[test]
    fn device_named_locations_never_reach_the_filesystem() {
        let temp = tempfile::tempdir().expect("tempdir");
        let blocks = temp.path().join("assets/minecraft/textures/blocks");
        std::fs::create_dir_all(&blocks).expect("resource directory");
        std::fs::write(blocks.join("stone.png"), b"png").expect("write resource");
        // Windows 11 lets a file have this name (Windows 10 does not); it is refused either way.
        let _ = std::fs::write(blocks.join("con.png"), b"png");
        let pack = FolderResourcePack::open(temp.path()).expect("open folder pack");

        assert!(pack.contains(&ResourceLocation::minecraft("textures/blocks/stone.png")));
        for path in [
            "textures/blocks/con.png",
            "textures/blocks/nul",
            "textures/COM1/x.png",
            "textures/blocks/aux.json",
            "lpt9.png",
            "conin$.txt",
        ] {
            let location = ResourceLocation::minecraft(path);
            assert!(!pack.contains(&location), "{path}");
            assert!(pack.open(&location).is_none(), "{path}");
            assert!(pack.file_path(&location).is_none(), "{path}");
        }
        for ordinary in [
            "console",
            "com10",
            "lpt",
            "nul_",
            "connector.png",
            "auxiliary",
        ] {
            assert!(!crate::is_windows_device_name(ordinary), "{ordinary}");
        }
    }

    #[test]
    fn java_folder_contract_only_exposes_regular_files_and_immediate_domains() {
        let temp = tempfile::tempdir().expect("tempdir");
        let assets = temp.path().join("assets");
        std::fs::create_dir_all(assets.join("minecraft/models/block"))
            .expect("create minecraft assets");
        std::fs::create_dir_all(assets.join("realms/lang")).expect("create realms assets");
        std::fs::create_dir_all(assets.join("UpperCase")).expect("create uppercase namespace");
        std::fs::create_dir_all(assets.join("minecraft/models/block/directory.json"))
            .expect("create directory with resource-shaped name");
        std::fs::write(
            assets.join("minecraft/models/block/stone.json"),
            br#"{"parent":"block/cube_all"}"#,
        )
        .expect("write regular resource");

        let pack = FolderResourcePack::open(temp.path()).expect("open folder pack");
        assert_eq!(pack.namespaces(), vec!["minecraft", "realms"]);

        let stone = ResourceLocation::minecraft("models/block/stone.json");
        let mut bytes = Vec::new();
        pack.open(&stone)
            .expect("regular file is readable")
            .read_to_end(&mut bytes)
            .expect("read regular file");
        assert_eq!(bytes, br#"{"parent":"block/cube_all"}"#);
        assert!(pack.contains(&stone));

        let directory = ResourceLocation::minecraft("models/block/directory.json");
        assert!(!pack.contains(&directory));

        let missing = ResourceLocation::new("realms", "lang/missing.lang");
        assert!(!pack.contains(&missing));
        assert!(pack.open(&missing).is_none());
    }
}
