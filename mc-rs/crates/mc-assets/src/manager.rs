//! The stacked resource manager: several packs, last one wins, per-resource fallthrough.
//!
// java: SimpleReloadableResourceManager.java, FallbackResourceManager.java

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::PathBuf;

use serde::de::DeserializeOwned;

use crate::pack::{read_capped, ResourcePack};
use crate::sprite::{parse_texture_metadata, TextureMetadata};
use crate::{AssetError, LanguageInfo, ResourceLocation};

/// A resource together with the pack it came from and its optional `.mcmeta` sidecar.
// java: SimpleResource.java
#[derive(Debug, Clone)]
pub struct Resource {
    /// The location that was requested.
    pub location: ResourceLocation,
    /// Name of the pack that supplied the bytes.
    pub pack_name: String,
    /// The resource bytes.
    pub data: Vec<u8>,
    /// Raw `<path>.mcmeta` bytes, when a pack at or above `pack_name` supplies them.
    pub metadata: Option<Vec<u8>>,
}

/// What [`ResourceManager::presence`] found out about one resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    /// No pack provides it, or the pack that does could not open it.
    Missing,
    /// The pack that provides it opened it.
    Readable,
    /// The pack that provides it lists it in memory ([`ResourcePack::lists`]). Nothing was
    /// opened, so reading it can still fail.
    Listed,
}

/// Notified whenever the pack stack changes.
///
/// This is the seam S21 needs for runtime pack swapping: register once, get called on every
/// [`ResourceManager::reload`]. Consumers that would rather poll can read
/// [`ResourceManager::generation`] instead.
// java: IResourceManagerReloadListener.java
pub trait ReloadListener {
    /// Called after the pack stack has been replaced, and once at registration time.
    fn on_reload(&mut self, manager: &ResourceManager);
}

/// A stack of resource packs queried as one.
///
/// Precedence follows vanilla exactly: packs are pushed low-to-high and lookups walk the
/// stack **downwards from the top**, taking the first pack that carries *that specific
/// resource*. A high pack overriding `textures/blocks/stone.png` does not hide the rest of
/// the packs below it — the fallthrough is per resource, not per pack.
///
/// # Examples
/// ```no_run
/// use mc_assets::{ObjectStorePack, ResourceLocation, ResourceManager, ZipResourcePack};
///
/// let mut rm = ResourceManager::new();
/// rm.push_pack(Box::new(ObjectStorePack::open("jars/assets", "1.8")?));  // lowest
/// rm.push_pack(Box::new(ZipResourcePack::open("jars/versions/1.8.9/1.8.9.jar")?));
///
/// let model = rm.load(&ResourceLocation::parse("models/block/stone.json")?)?;
/// assert!(!model.is_empty());
/// # Ok::<_, mc_assets::AssetError>(())
/// ```
pub struct ResourceManager {
    packs: Vec<Box<dyn ResourcePack>>,
    namespaces: BTreeSet<String>,
    listeners: Vec<Box<dyn ReloadListener>>,
    generation: u64,
    max_resource_size: u64,
}

impl std::fmt::Debug for ResourceManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceManager")
            .field(
                "packs",
                &self.packs.iter().map(|p| p.name()).collect::<Vec<_>>(),
            )
            .field("namespaces", &self.namespaces)
            .field("listeners", &self.listeners.len())
            .field("generation", &self.generation)
            .finish()
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceManager {
    /// Default per-resource read cap: 64 MiB.
    pub const DEFAULT_MAX_RESOURCE_SIZE: u64 = 64 * 1024 * 1024;

    /// An empty manager.
    pub fn new() -> Self {
        Self {
            packs: Vec::new(),
            namespaces: BTreeSet::new(),
            listeners: Vec::new(),
            generation: 0,
            max_resource_size: Self::DEFAULT_MAX_RESOURCE_SIZE,
        }
    }

    /// Overrides the per-resource read cap applied by [`ResourceManager::load`].
    #[must_use]
    pub fn with_max_resource_size(mut self, bytes: u64) -> Self {
        self.max_resource_size = bytes;
        self
    }

    /// Pushes a pack on top of the stack. Later pushes win.
    // java: SimpleReloadableResourceManager.java:33 (reloadResourcePack)
    pub fn push_pack(&mut self, pack: Box<dyn ResourcePack>) {
        self.namespaces.extend(pack.namespaces());
        self.packs.push(pack);
    }

    /// Replaces the whole stack, bumps [`generation`](ResourceManager::generation), and
    /// notifies every registered [`ReloadListener`].
    ///
    /// This is the runtime pack-swap hook. Nothing in the client calls it yet; S21 will.
    // java: SimpleReloadableResourceManager.java:89 (reloadResources)
    pub fn reload(&mut self, packs: Vec<Box<dyn ResourcePack>>) {
        mc_core::profile_scope!("assets.reload");
        self.packs.clear();
        self.namespaces.clear();
        for pack in packs {
            self.push_pack(pack);
        }
        self.generation = self.generation.wrapping_add(1);
        tracing::info!(
            generation = self.generation,
            packs = ?self.pack_names(),
            "reloaded resource manager"
        );
        self.notify_listeners();
    }

    /// Monotonic counter bumped by every [`reload`](ResourceManager::reload).
    ///
    /// Cheaper to consume than a listener when the consumer already runs every frame: cache
    /// the value, compare, rebuild when it moved.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Registers a reload listener and calls it immediately, as vanilla does.
    // java: SimpleReloadableResourceManager.java:108
    pub fn add_reload_listener(&mut self, mut listener: Box<dyn ReloadListener>) {
        listener.on_reload(self);
        self.listeners.push(listener);
    }

    fn notify_listeners(&mut self) {
        // Taken out so listeners can borrow the manager immutably; anything a listener
        // registers during the callback is appended rather than dropped.
        let mut listeners = std::mem::take(&mut self.listeners);
        for listener in &mut listeners {
            listener.on_reload(self);
        }
        listeners.append(&mut self.listeners);
        self.listeners = listeners;
    }

    /// Names of the stacked packs, bottom first.
    pub fn pack_names(&self) -> Vec<&str> {
        self.packs.iter().map(|p| p.name()).collect()
    }

    /// Number of stacked packs.
    pub fn pack_count(&self) -> usize {
        self.packs.len()
    }

    /// Every language advertised by the current pack stack, sorted by code.
    ///
    /// Packs are stored in Minecraft's established bottom-to-top order. Java's
    /// `LanguageManager.parseLanguageMetadata` visits that list in order and stores a descriptor
    /// only `if (!this.languageMap.containsKey(code))`, so the **lowest** pack advertising a code
    /// keeps it: the default pack's vanilla languages cannot be renamed, moved to another region
    /// or made bidirectional by a resource pack, which may only add new codes. (PA: mc-rs let the higher pack win.) This is computed from the live stack so removed packs
    /// vanish immediately after [`reload`](Self::reload).
    pub fn language_descriptors(&self) -> Vec<LanguageInfo> {
        let mut by_code = BTreeMap::new();
        for pack in &self.packs {
            for language in pack.language_descriptors() {
                by_code
                    .entry(language.code.clone())
                    .or_insert_with(|| language.clone());
            }
        }
        by_code.into_values().collect()
    }

    /// Every namespace any stacked pack declares, sorted.
    // java: SimpleReloadableResourceManager.java:50 (getResourceDomains)
    pub fn namespaces(&self) -> &BTreeSet<String> {
        &self.namespaces
    }

    /// Index of the topmost pack providing this resource.
    // java: FallbackResourceManager.java:42 — iterate from the top down.
    fn top_pack_index(&self, l: &ResourceLocation) -> Option<usize> {
        (0..self.packs.len())
            .rev()
            .find(|&i| self.packs[i].contains(l))
    }

    /// Whether any pack provides this resource.
    pub fn contains(&self, l: &ResourceLocation) -> bool {
        self.top_pack_index(l).is_some()
    }

    /// Opens a reader from the topmost pack providing this resource.
    pub fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
        self.packs[self.top_pack_index(l)?].open(l)
    }

    /// Whether the resource is there, found as cheaply as each pack allows. The stack is walked
    /// from the top as [`ResourceManager::open`] walks it: a pack that can answer from memory
    /// ([`ResourcePack::lists`]) does so without storage access, and any other pack is asked
    /// through `contains` and then actually opened, which is what this replaced everywhere.
    ///
    /// Added: `sounds.json` names every sound file, and
    /// opening each one at start-up took 1.4 s with a cold cache and 4-9 s with the assets on a
    /// USB disk. [`Presence::Listed`] tells the caller nothing was opened.
    // java: FallbackResourceManager.getResource — resourceExists per pack, top down, then open.
    pub fn presence(&self, l: &ResourceLocation) -> Presence {
        for pack in self.packs.iter().rev() {
            match pack.lists(l) {
                Some(true) => return Presence::Listed,
                Some(false) => {}
                None if pack.contains(l) => {
                    return if pack.open(l).is_some() {
                        Presence::Readable
                    } else {
                        Presence::Missing
                    };
                }
                None => {}
            }
        }
        Presence::Missing
    }

    /// The per-resource read cap [`ResourceManager::load`] applies, for a caller that reads a
    /// [`ResourceManager::file_path`] itself.
    pub fn max_resource_size(&self) -> u64 {
        self.max_resource_size
    }

    /// The plain file [`ResourceManager::load`] would read `l` from, when the pack providing it
    /// keeps it as a plain file (the object store, a folder pack), so that another thread can
    /// read it. The pack is found the way [`ResourceManager::presence`] finds it, from in-memory
    /// listings where a pack has one, so the file may turn out to be missing. `None` when the
    /// providing pack is a zip, or no pack has it.
    pub fn file_path(&self, l: &ResourceLocation) -> Option<PathBuf> {
        for pack in self.packs.iter().rev() {
            match pack.lists(l) {
                Some(true) => return pack.file_path(l),
                Some(false) => {}
                None if pack.contains(l) => return pack.file_path(l),
                None => {}
            }
        }
        None
    }

    /// Reads a resource fully, from the topmost pack providing it.
    ///
    /// Fails with [`AssetError::NotFound`] when no pack has it and
    /// [`AssetError::TooLarge`] when it exceeds the configured cap.
    // java: FallbackResourceManager.java:37 (getResource)
    pub fn load(&self, l: &ResourceLocation) -> Result<Vec<u8>, AssetError> {
        let idx = self
            .top_pack_index(l)
            .ok_or_else(|| AssetError::NotFound(l.clone()))?;
        let reader = self.packs[idx]
            .open(l)
            .ok_or_else(|| AssetError::NotFound(l.clone()))?;
        read_capped(reader, self.max_resource_size, l.to_string())
    }

    /// Reads a resource and decodes it as UTF-8, replacing invalid sequences.
    ///
    /// Lossy on purpose: vanilla reads text assets through `IOUtils.readLines(.., UTF_8)`,
    /// which also substitutes rather than failing, and at least one shipped `.lang` file
    /// relies on that.
    pub fn load_string(&self, l: &ResourceLocation) -> Result<String, AssetError> {
        Ok(String::from_utf8_lossy(&self.load(l)?).into_owned())
    }

    /// Reads a resource and deserialises it as JSON.
    ///
    /// S09 leans on this for blockstate and model files.
    pub fn load_json<T: DeserializeOwned>(&self, l: &ResourceLocation) -> Result<T, AssetError> {
        let bytes = self.load(l)?;
        serde_json::from_slice(&bytes).map_err(|e| AssetError::json(l.to_string(), e))
    }

    /// Reads a resource together with its `.mcmeta` sidecar and originating pack name.
    ///
    /// The sidecar follows vanilla's slightly surprising rule: it comes from the **topmost**
    /// pack that has `<path>.mcmeta`, scanning down only as far as the pack that supplies the
    /// resource itself. A high-priority pack can therefore re-animate a texture supplied by a
    /// lower one, but a sidecar sitting *below* the winning pack is ignored.
    // java: FallbackResourceManager.java:37
    pub fn resource(&self, l: &ResourceLocation) -> Result<Resource, AssetError> {
        let meta_loc = l.with_suffix(".mcmeta");
        let mut meta_pack: Option<usize> = None;

        for i in (0..self.packs.len()).rev() {
            if meta_pack.is_none() && self.packs[i].contains(&meta_loc) {
                meta_pack = Some(i);
            }
            if self.packs[i].contains(l) {
                let data = self.packs[i]
                    .open(l)
                    .ok_or_else(|| AssetError::NotFound(l.clone()))
                    .and_then(|r| read_capped(r, self.max_resource_size, l.to_string()))?;
                // java: `getResource` opens the selected sidecar's stream eagerly, and
                // `SimpleResource.getMetadata` reads it on first use; an `IOException` from
                // either fails the resource (a texture falls back to the missing one), it never
                // quietly loses its animation or blur. PA: mc-rs used to drop
                // an unreadable sidecar and load the resource without it.
                let metadata = match meta_pack {
                    Some(m) => Some(
                        self.packs[m]
                            .open(&meta_loc)
                            .ok_or_else(|| AssetError::NotFound(meta_loc.clone()))
                            .and_then(|r| {
                                // What metadata can need, at most.
                                let cap = self.max_resource_size.min(crate::pack::MAX_MCMETA_BYTES);
                                read_capped(r, cap, meta_loc.to_string())
                            })?,
                    ),
                    None => None,
                };
                return Ok(Resource {
                    location: l.clone(),
                    pack_name: self.packs[i].name().to_owned(),
                    data,
                    metadata,
                });
            }
        }
        Err(AssetError::NotFound(l.clone()))
    }

    /// Reads and decodes one PNG texture from the active pack stack.
    ///
    /// This is the resource-backed half of the live entity texture path. Keeping PNG decoding
    /// beside [`ResourceManager::resource`] preserves per-resource pack precedence and the
    /// matching `.mcmeta` sidecar; renderers can then compose several already-decoded layers
    /// without reaching into pack internals. The returned bytes are tightly packed RGBA8.
    pub fn load_png_rgba(
        &self,
        l: &ResourceLocation,
    ) -> Result<(Vec<u8>, u32, u32, TextureMetadata), AssetError> {
        let resource = self.resource(l)?;
        // A texture of its own: refused if the device cannot hold it.
        let image = crate::texture_size::decode_texture_png(&resource.data)
            .map_err(|error| AssetError::Image {
                location: l.to_string(),
                message: error.to_string(),
            })?
            .into_rgba8();
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            return Err(AssetError::Image {
                location: l.to_string(),
                message: "decoded image has zero dimensions".to_owned(),
            });
        }
        let metadata = parse_texture_metadata(l, resource.metadata.as_deref())?;
        Ok((image.into_raw(), width, height, metadata))
    }

    /// [`ResourcePack::list_files`] for every stacked pack, **highest priority first** (the
    /// reverse of push order), so index 0 is the pack whose resources win.
    ///
    /// Additive: connected-texture rules are collected pack by
    /// pack from the top down, and the rule order is observable.
    pub fn list_files_by_pack(&self, namespace: &str, dir_prefix: &str) -> Vec<Vec<String>> {
        self.packs
            .iter()
            .rev()
            .map(|pack| pack.list_files(namespace, dir_prefix))
            .collect()
    }

    /// Reads `l` from one pack only, ignoring the stack above it. `index` counts from the top,
    /// as in [`ResourceManager::list_files_by_pack`].
    ///
    /// # Errors
    /// [`AssetError::NotFound`] when there is no such pack or it lacks the resource, and
    /// [`AssetError::TooLarge`] past the configured cap.
    pub fn load_from_pack(
        &self,
        index: usize,
        l: &ResourceLocation,
    ) -> Result<Vec<u8>, AssetError> {
        let pack = index
            .checked_add(1)
            .and_then(|n| self.packs.len().checked_sub(n))
            .and_then(|i| self.packs.get(i))
            .ok_or_else(|| AssetError::NotFound(l.clone()))?;
        let reader = pack
            .open(l)
            .ok_or_else(|| AssetError::NotFound(l.clone()))?;
        read_capped(reader, self.max_resource_size, l.to_string())
    }

    /// Reads this resource from **every** pack that has it, bottom pack first.
    ///
    /// Language files are merged this way: each pack's `.lang` contributes, and because the
    /// list runs bottom-to-top a higher pack's key overwrites a lower one's.
    // java: FallbackResourceManager.java:73 (getAllResources)
    pub fn all_resources(&self, l: &ResourceLocation) -> Vec<Resource> {
        let meta_loc = l.with_suffix(".mcmeta");
        let mut out = Vec::new();
        for pack in &self.packs {
            if !pack.contains(l) {
                continue;
            }
            let Some(data) = pack
                .open(l)
                .and_then(|r| read_capped(r, self.max_resource_size, l.to_string()).ok())
            else {
                tracing::warn!(pack = %pack.name(), location = %l, "skipping unreadable resource");
                continue;
            };
            // What metadata can need, at most.
            let meta_cap = self.max_resource_size.min(crate::pack::MAX_MCMETA_BYTES);
            let metadata = pack
                .open(&meta_loc)
                .and_then(|r| read_capped(r, meta_cap, meta_loc.to_string()).ok());
            out.push(Resource {
                location: l.clone(),
                pack_name: pack.name().to_owned(),
                data,
                metadata,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;

    /// An in-memory pack, so precedence can be tested without touching the disk.
    struct MemPack {
        name: String,
        files: HashMap<String, Vec<u8>>,
    }

    impl MemPack {
        fn boxed(name: &str, files: &[(&str, &str)]) -> Box<dyn ResourcePack> {
            Box::new(Self {
                name: name.to_owned(),
                files: files
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), v.as_bytes().to_vec()))
                    .collect(),
            })
        }
    }

    impl ResourcePack for MemPack {
        fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
            self.files
                .get(&l.to_string())
                .map(|v| Box::new(Cursor::new(v.clone())) as Box<dyn Read>)
        }
        fn contains(&self, l: &ResourceLocation) -> bool {
            self.files.contains_key(&l.to_string())
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn namespaces(&self) -> Vec<String> {
            let mut v: Vec<String> = self
                .files
                .keys()
                .filter_map(|k| k.split_once(':').map(|(ns, _)| ns.to_owned()))
                .collect();
            v.sort();
            v.dedup();
            v
        }
    }

    fn loc(s: &str) -> ResourceLocation {
        ResourceLocation::parse(s).unwrap()
    }

    /// A pack with an in-memory listing (`lists`) that disagrees with its storage the way a
    /// partial launcher download does, counting every storage call.
    struct ListingPack {
        listed: Vec<&'static str>,
        stored: Vec<&'static str>,
        storage_calls: Arc<AtomicUsize>,
    }

    impl ResourcePack for ListingPack {
        fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
            self.storage_calls.fetch_add(1, Ordering::Relaxed);
            self.stored
                .contains(&l.to_string().as_str())
                .then(|| Box::new(Cursor::new(b"x".to_vec())) as Box<dyn Read>)
        }
        fn contains(&self, l: &ResourceLocation) -> bool {
            self.storage_calls.fetch_add(1, Ordering::Relaxed);
            self.stored.contains(&l.to_string().as_str())
        }
        fn lists(&self, l: &ResourceLocation) -> Option<bool> {
            Some(self.listed.contains(&l.to_string().as_str()))
        }
    }

    /// `presence` answers from a pack's listing without storage calls, falls through a pack
    /// that does not list the resource, and opens a pack that cannot list (`lists` = `None`).
    #[test]
    fn presence_uses_listings_without_storage_and_opens_the_rest() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed(
            "folder-like",
            &[("minecraft:only/below", "x")],
        ));
        rm.push_pack(Box::new(ListingPack {
            listed: vec![
                "minecraft:indexed/present",
                "minecraft:indexed/not_downloaded",
            ],
            stored: vec!["minecraft:indexed/present"],
            storage_calls: Arc::clone(&calls),
        }));
        assert_eq!(rm.presence(&loc("indexed/present")), Presence::Listed);
        // Listed, not promised: the read is what finds out, and the manager falls through.
        assert_eq!(
            rm.presence(&loc("indexed/not_downloaded")),
            Presence::Listed
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "a listing needs no storage call"
        );
        assert!(rm.load(&loc("indexed/not_downloaded")).is_err());

        assert_eq!(rm.presence(&loc("only/below")), Presence::Readable);
        assert_eq!(rm.presence(&loc("nowhere")), Presence::Missing);
    }

    /// The object store lists from its index, so an indexed blob missing on disk is listed but
    /// neither contained nor openable.
    #[test]
    fn the_object_store_lists_from_its_index_alone() {
        let index = crate::AssetIndex::parse(
            r#"{"objects":{"minecraft/sounds/a.ogg":{"hash":"aa00000000000000000000000000000000000000","size":1}}}"#,
        )
        .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let pack = crate::ObjectStorePack::with_index(index, temp.path(), "partial download");
        let indexed = loc("sounds/a.ogg");
        assert_eq!(pack.lists(&indexed), Some(true));
        assert!(!pack.contains(&indexed) && pack.open(&indexed).is_none());
        assert_eq!(pack.lists(&loc("sounds/b.ogg")), Some(false));
    }

    #[test]
    fn later_packs_win() {
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed("base", &[("minecraft:a", "base-a")]));
        rm.push_pack(MemPack::boxed("top", &[("minecraft:a", "top-a")]));
        assert_eq!(rm.load(&loc("a")).unwrap(), b"top-a");
    }

    #[test]
    fn fallthrough_is_per_resource_not_per_pack() {
        // The top pack overrides only `a`. `b` must still resolve from the base pack —
        // this is the FallbackResourceManager behaviour that is easy to get wrong.
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed(
            "base",
            &[("minecraft:a", "base-a"), ("minecraft:b", "base-b")],
        ));
        rm.push_pack(MemPack::boxed("top", &[("minecraft:a", "top-a")]));

        assert_eq!(rm.load(&loc("a")).unwrap(), b"top-a");
        assert_eq!(rm.load(&loc("b")).unwrap(), b"base-b");
        assert_eq!(rm.resource(&loc("b")).unwrap().pack_name, "base");
    }

    #[test]
    fn three_deep_precedence() {
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed(
            "p1",
            &[
                ("minecraft:x", "1"),
                ("minecraft:y", "1"),
                ("minecraft:z", "1"),
            ],
        ));
        rm.push_pack(MemPack::boxed(
            "p2",
            &[("minecraft:x", "2"), ("minecraft:y", "2")],
        ));
        rm.push_pack(MemPack::boxed("p3", &[("minecraft:x", "3")]));

        assert_eq!(rm.load(&loc("x")).unwrap(), b"3");
        assert_eq!(rm.load(&loc("y")).unwrap(), b"2");
        assert_eq!(rm.load(&loc("z")).unwrap(), b"1");
    }

    #[test]
    fn missing_resource_is_a_typed_error() {
        let rm = ResourceManager::new();
        let error = rm
            .load(&loc("nope"))
            .expect_err("an empty manager must report the missing resource");
        assert!(matches!(&error, AssetError::NotFound(_)));
        assert_eq!(
            error.to_string(),
            "File not found",
            "missing resources use Java JsonException's normalized message"
        );
        assert!(!rm.contains(&loc("nope")));
    }

    #[test]
    fn malformed_json_has_the_java_invalid_file_prefix_and_a_typed_source() {
        use std::error::Error;

        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed(
            "p",
            &[("minecraft:bad.json", r#"{"unterminated":)"#)],
        ));
        let error = rm
            .load_json::<serde_json::Value>(&loc("bad.json"))
            .expect_err("malformed JSON must be rejected");
        assert!(
            error
                .to_string()
                .starts_with("Invalid minecraft:bad.json: "),
            "Java's file entry must be visible: {error}"
        );
        assert!(error.source().is_some(), "serde's typed cause must survive");
    }

    #[test]
    fn json_without_a_filename_uses_the_unknown_file_entry() {
        let source = serde_json::from_str::<serde_json::Value>("{").expect_err("malformed JSON");
        let error = AssetError::json("", source);
        assert!(error.to_string().starts_with("Invalid (Unknown file): "));
    }

    #[test]
    fn mcmeta_comes_from_the_topmost_pack_at_or_above_the_winner() {
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed(
            "base",
            &[
                ("minecraft:t.png", "base-png"),
                ("minecraft:t.png.mcmeta", "base-meta"),
            ],
        ));
        rm.push_pack(MemPack::boxed(
            "top",
            &[("minecraft:t.png.mcmeta", "top-meta")],
        ));

        let r = rm.resource(&loc("t.png")).unwrap();
        assert_eq!(r.pack_name, "base");
        assert_eq!(r.data, b"base-png");
        assert_eq!(r.metadata.as_deref(), Some(&b"top-meta"[..]));
    }

    #[test]
    fn mcmeta_below_the_winning_pack_is_ignored() {
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed(
            "base",
            &[
                ("minecraft:t.png", "base-png"),
                ("minecraft:t.png.mcmeta", "base-meta"),
            ],
        ));
        rm.push_pack(MemPack::boxed("top", &[("minecraft:t.png", "top-png")]));

        let r = rm.resource(&loc("t.png")).unwrap();
        assert_eq!(r.pack_name, "top");
        assert_eq!(r.metadata, None);
    }

    /// java: `FallbackResourceManager.getResource` opens the selected `.mcmeta` eagerly and
    /// `SimpleResource.getMetadata` reads it; an `IOException` in either fails the resource. A
    /// sidecar that a pack lists but cannot open, or one past the size cap, must not silently
    /// leave the texture without its metadata.
    #[test]
    fn an_unreadable_selected_mcmeta_fails_the_resource() {
        struct UnopenableMeta;
        impl ResourcePack for UnopenableMeta {
            fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
                (l.path == "t.png").then(|| Box::new(Cursor::new(b"png".to_vec())) as Box<dyn Read>)
            }
            fn contains(&self, l: &ResourceLocation) -> bool {
                l.path == "t.png" || l.path == "t.png.mcmeta"
            }
            fn name(&self) -> &str {
                "unopenable"
            }
        }
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(UnopenableMeta));
        assert!(rm.resource(&loc("t.png")).is_err(), "unopenable sidecar");
        // Negative control: without a sidecar the resource loads, metadata absent.
        let mut plain = ResourceManager::new();
        plain.push_pack(MemPack::boxed("plain", &[("minecraft:t.png", "png")]));
        assert_eq!(plain.resource(&loc("t.png")).unwrap().metadata, None);

        let capped = |cap: u64| {
            let mut rm = ResourceManager::new().with_max_resource_size(cap);
            rm.push_pack(MemPack::boxed(
                "capped",
                &[
                    ("minecraft:t.png", "png"),
                    ("minecraft:t.png.mcmeta", "{\"animation\":{}}"),
                ],
            ));
            rm
        };
        assert!(capped(1024).resource(&loc("t.png")).is_ok());
        assert!(
            capped(4).resource(&loc("t.png")).is_err(),
            "sidecar past the cap"
        );
    }

    #[test]
    fn all_resources_runs_bottom_to_top() {
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed("base", &[("minecraft:l", "base")]));
        rm.push_pack(MemPack::boxed("mid", &[("minecraft:other", "x")]));
        rm.push_pack(MemPack::boxed("top", &[("minecraft:l", "top")]));

        let all = rm.all_resources(&loc("l"));
        assert_eq!(
            all.iter().map(|r| r.pack_name.as_str()).collect::<Vec<_>>(),
            ["base", "top"]
        );
    }

    #[test]
    fn language_descriptors_are_lowest_pack_wins_and_are_never_stale() {
        struct LanguagePack {
            name: &'static str,
            languages: Vec<LanguageInfo>,
        }
        impl ResourcePack for LanguagePack {
            fn open(&self, _: &ResourceLocation) -> Option<Box<dyn Read>> {
                None
            }
            fn contains(&self, _: &ResourceLocation) -> bool {
                false
            }
            fn name(&self) -> &str {
                self.name
            }
            fn language_descriptors(&self) -> &[LanguageInfo] {
                &self.languages
            }
        }
        fn language(code: &str, region: &str, name: &str, bidirectional: bool) -> LanguageInfo {
            LanguageInfo {
                code: code.to_owned(),
                name: name.to_owned(),
                region: region.to_owned(),
                bidirectional,
            }
        }

        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(LanguagePack {
            name: "base",
            languages: vec![
                language("aa_AA", "Base Region", "Base", false),
                language("bb_BB", "Removed Region", "Removed", false),
            ],
        }));
        rm.push_pack(Box::new(LanguagePack {
            name: "top",
            languages: vec![language("aa_AA", "Top Region", "Top", true)],
        }));
        let languages = rm.language_descriptors();
        assert_eq!(
            languages
                .iter()
                .map(|language| language.code.as_str())
                .collect::<Vec<_>>(),
            ["aa_AA", "bb_BB"]
        );
        // java: `parseLanguageMetadata` keeps the first descriptor for a code, in bottom-to-top
        // order, so the higher pack's duplicate `aa_AA` (and its bidi flag) is ignored.
        assert_eq!(languages[0].name, "Base");
        assert_eq!(languages[0].region, "Base Region");
        assert!(!languages[0].bidirectional);

        rm.reload(vec![Box::new(LanguagePack {
            name: "replacement",
            languages: vec![language("cc_CC", "Fresh Region", "Fresh", false)],
        })]);
        assert_eq!(
            rm.language_descriptors()
                .iter()
                .map(|language| language.code.as_str())
                .collect::<Vec<_>>(),
            ["cc_CC"],
            "an uncached query cannot retain removed-pack descriptors"
        );
    }

    #[test]
    fn namespaces_are_unioned_across_packs() {
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed("a", &[("minecraft:x", "1")]));
        rm.push_pack(MemPack::boxed("b", &[("realms:y", "2")]));
        let ns: Vec<&str> = rm.namespaces().iter().map(String::as_str).collect();
        assert_eq!(ns, ["minecraft", "realms"]);
    }

    #[test]
    fn reload_swaps_the_stack_and_fires_listeners() {
        struct Counter(Arc<AtomicUsize>);
        impl ReloadListener for Counter {
            fn on_reload(&mut self, _m: &ResourceManager) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let hits = Arc::new(AtomicUsize::new(0));
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed("old", &[("minecraft:a", "old")]));
        rm.add_reload_listener(Box::new(Counter(Arc::clone(&hits))));
        assert_eq!(hits.load(Ordering::SeqCst), 1, "fires once at registration");
        assert_eq!(rm.generation(), 0);

        rm.reload(vec![MemPack::boxed("new", &[("minecraft:a", "new")])]);

        assert_eq!(rm.load(&loc("a")).unwrap(), b"new");
        assert_eq!(rm.pack_names(), ["new"]);
        assert_eq!(rm.generation(), 1);
        assert_eq!(hits.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn oversize_resources_are_refused() {
        let mut rm = ResourceManager::new().with_max_resource_size(4);
        rm.push_pack(MemPack::boxed("p", &[("minecraft:big", "0123456789")]));
        assert!(matches!(
            rm.load(&loc("big")),
            Err(AssetError::TooLarge { .. })
        ));
    }

    #[test]
    fn json_loads_through_the_stack() {
        #[derive(serde::Deserialize)]
        struct Model {
            parent: String,
        }
        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed(
            "p",
            &[(
                "minecraft:models/block/stone.json",
                r#"{"parent":"block/cube_all"}"#,
            )],
        ));
        let m: Model = rm.load_json(&loc("models/block/stone.json")).unwrap();
        assert_eq!(m.parent, "block/cube_all");
    }
}
