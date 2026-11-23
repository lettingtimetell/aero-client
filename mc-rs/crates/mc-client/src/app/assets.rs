//! Asset loading and resource-pack reloading.
//!
//! java: the middle third of `Minecraft.startGame` (`:493`–`:562`) plus `refreshResources`
//! (`:788`). Vanilla interleaves "build the pack stack", "reload every listener" and "construct
//! the subsystem that consumes the result"; here the first two are [`GameAssets::load`] and
//! [`GameAssets::reload_packs`], and constructing subsystems is [`crate::app::driver`]'s job.
//!
//! # What reloads — read this before touching pack swapping
//!
//! Everything, now. [`GameAssets::reload_packs`] re-stacks the packs and re-derives *all* of it:
//! the language table, the font, the sound registry, **and** the baked block models, the stitched
//! block atlas, the item models and the biome colormaps. The last four used to be frozen at first
//! install because `mc_assets::install` and `mc_render::terrain::tint::install_colormaps` were
//! `OnceLock`s; they are now [`mc_assets::model::HotSwap`]s with an explicit
//! [`mc_assets::model::reinstall`] / [`mc_render::terrain::tint::reload_colormaps`] alongside the
//! unchanged first-wins `install`.
//!
//! Two things still need the caller's help, and [`crate::app::driver::Client::reload_resource_packs`]
//! does them:
//!
//! * a re-stitched atlas is baked into `TerrainPhase`'s bind group at construction, so the render
//!   phases must be **rebuilt**, not updated — java: `refreshResources` ends with
//!   `RenderGlobal.loadRenderers()` for exactly this reason;
//! * already-uploaded GUI textures are cached by key in `mc_ui::widget::UiRenderer`, so they need
//!   `invalidate`/`clear_textures`.
//!
//! # Deferred loading
//!
//! [`GameAssets::load`] does the whole load in one call, which is what a test wants. The *game*
//! wants the window open first so the user sees a progress screen, so it uses
//! [`GameAssets::boot`] (packs, language, font — everything the loading screen itself needs) plus
//! an [`AssetLoader`] the frame loop steps through. Both paths run the same stages in the same
//! order and report the same progress; `load` is literally `boot` followed by `step` in a loop.

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use mc_assets::cache::{BakeInputs, BakedSnapshot, CacheRead};
use mc_assets::{
    BakeReport, FolderResourcePack, Language, ObjectStorePack, ResourceManager, ResourcePack,
    ZipLimits, ZipResourcePack,
};
use mc_render::settings::GameSettings;
use mc_ui::font::glyphs::FontData;

use crate::app::config::ClientConfig;
use crate::app::loading::{LoadingScreen, StageProgress};
use crate::error::ClientError;

/// The maximum atlas dimension handed to [`mc_assets::bake_all`].
///
/// java: `TextureMap` stitches into whatever `Minecraft.getGLMaximumTextureSize()` reports, which
/// on any modern GPU is at least 16384. 2048 is what S09's own tests use against the real jar and
/// is enough for vanilla's block sprites. It remains the fallback for synchronous or GPU-less
/// loads; once a [`mc_render::Gpu`] exists, its granted limit replaces this value.
pub const MAX_ATLAS_DIMENSION: u32 = 2048;

/// User pack roots in precedence order. The client folder wins over standard Minecraft.
fn user_pack_roots(config: &ClientConfig) -> impl Iterator<Item = &Path> {
    std::iter::once(config.resource_pack_dir.as_path()).chain(
        config
            .resource_pack_fallback_dir
            .as_deref()
            .filter(|fallback| *fallback != config.resource_pack_dir.as_path()),
    )
}

/// Resolve a saved pack name the same way for loading, compatibility checks and its icon.
///
/// `None` when the name cannot be an entry of a pack folder (see [`is_plain_pack_name`]): it
/// then names no pack, and nothing is looked up on disk for it.
pub(crate) fn user_resource_pack_path(config: &ClientConfig, name: &str) -> Option<PathBuf> {
    if !is_plain_pack_name(name) {
        return None;
    }
    Some(
        user_pack_roots(config)
            .map(|root| root.join(name))
            .find(|path| path.exists())
            .unwrap_or_else(|| config.resource_pack_dir.join(name)),
    )
}

/// Whether a saved pack name (from `options.txt`, which travels with a
/// shared game folder) can be the name of an entry directly inside a pack folder.
///
/// java: `ResourcePackRepository`'s constructor only matches saved names against the entries
/// `updateRepositoryEntriesAll` listed in the pack folder, so any other name selects nothing and
/// nothing is looked up for it. `Path::join` would instead follow `..` out of the folder and
/// replace the folder with an absolute name, and on Windows a `\\server\share\...` name makes
/// Windows sign in to that server to look; a device name (`CON`, `COM1.zip`) opens the device on
/// Windows 10. Such names stay in `options.txt` untouched, as vanilla keeps an unmatched name.
fn is_plain_pack_name(name: &str) -> bool {
    use std::path::Component;
    let mut parts = Path::new(name).components();
    let single = matches!(
        (parts.next(), parts.next()),
        (Some(Component::Normal(_)), None)
    );
    // A separator anywhere (`pack/.` is one component to `components`), a NUL, and on Windows
    // the second separator, a drive or stream colon and the device names.
    single
        && !name.contains(['/', '\0'])
        && !(cfg!(windows)
            && (name.contains(['\\', ':']) || mc_assets::is_windows_device_name(name)))
}

/// Discover packs in both folders, retaining the primary entry for duplicate names.
pub(crate) fn user_resource_pack_candidates(config: &ClientConfig) -> Vec<(String, PathBuf)> {
    let mut entries = std::collections::BTreeMap::new();
    for root in user_pack_roots(config) {
        let Ok(directory) = std::fs::read_dir(root) else {
            continue;
        };
        for entry in directory.flatten() {
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            #[cfg(target_os = "windows")]
            let key = name.to_lowercase();
            #[cfg(not(target_os = "windows"))]
            let key = name.clone();
            entries.entry(key).or_insert_with(|| (name, entry.path()));
        }
    }
    entries.into_values().collect()
}

/// Maximum decoded width or height accepted for an untrusted resource-pack icon.
///
/// Vanilla delegates to `ImageIO`, but a server-supplied pack can make this input hostile. 4096
/// still permits an icon 128 times larger than its 32-pixel GUI destination while bounding the
/// decoder's strict dimension limits.
const MAX_PACK_ICON_DIMENSION: u32 = 4096;
const MAX_PACK_ICON_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PACK_ICON_ALLOC_BYTES: u64 = 64 * 1024 * 1024;

/// Vanilla's `ResourcePackRepository.dirServerResourcepacks` child directory.
pub const SERVER_RESOURCE_PACK_DIR: &str = "server-resource-packs";
/// `HttpUtil.downloadResourcePack`'s hard byte ceiling (50 MiB).
pub const SERVER_RESOURCE_PACK_MAX_BYTES: u64 = 50 * 1024 * 1024;
/// `deleteOldServerResourcesPacks` retains the ten newest files.
pub const SERVER_RESOURCE_PACK_CACHE_LIMIT: usize = 10;
const SERVER_RESOURCE_PACK_ACTIVE_MARKER: &str = ".active";
const SERVER_RESOURCE_PACK_PART_SUFFIX: &str = ".part";
const SERVER_RESOURCE_PACK_HTTP_TIMEOUT: Duration = Duration::from_secs(30);
/// What vanilla's pack download says about itself: the defaults of
/// Java's `HttpURLConnection`, under the Java 8 runtime (8u51) the launcher runs 1.8.9 on. The
/// request goes to a host the server names, usually its own, so a client-specific value would
/// tell the server operator which client the player uses.
const SERVER_RESOURCE_PACK_USER_AGENT: &str = "Java/1.8.0_51";
/// java: `sun.net.www.protocol.http.HttpURLConnection.acceptString`.
const SERVER_RESOURCE_PACK_ACCEPT: &str = "text/html, image/gif, image/jpeg, *; q=.2, */*; q=.2";
/// `HttpURLConnection` follows a bounded number of redirects before reporting failure.
///
/// Java 8's implementation defaults to twenty redirects (`http.maxRedirects`). Keeping the
/// bound here also prevents a server from holding a resource-pack worker in a redirect loop
/// forever. HTTPS remains in the same bounded policy because rustls is configured on the client.
const SERVER_RESOURCE_PACK_MAX_REDIRECTS: usize = 20;
/// A `Location` header is metadata, not a resource-pack body. Bound it separately so a redirect
/// cannot use the response-header budget to make the next URL unreasonably large.
const SERVER_RESOURCE_PACK_MAX_REDIRECT_LOCATION_BYTES: usize = 8 * 1024;

/// Decoded `pack.png` pixels ready for `UiRenderer::upload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackIconImage {
    /// Tightly packed RGBA8 pixels.
    pub rgba: Vec<u8>,
    /// Source image width. Resource-pack icons are scaled to 32 pixels by the UI.
    pub width: u32,
    /// Source image height. Resource-pack icons are scaled to 32 pixels by the UI.
    pub height: u32,
}

/// Metadata and optional icon used to construct one resource-pack selection row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePackUiData {
    /// Flattened `pack.mcmeta` description, including vanilla's red fallback on failure.
    pub description: String,
    /// Declared pack format, or `None` when metadata was absent or malformed.
    pub pack_format: Option<i64>,
    /// Decoded root `pack.png`. `None` means the UI must use the built-in default icon.
    pub icon: Option<PackIconImage>,
}

/// Everything loaded from disk that the rest of the client reads.
///
/// java: the cluster of fields `startGame` assigns — `mcResourceManager`, `mcLanguageManager`,
/// `fontRendererObj`'s data, `textureMapBlocks`, `mcSoundHandler`'s registry.
pub struct GameAssets {
    /// The stacked resource packs.
    pub resources: ResourceManager,
    /// The active language table. java: `LanguageManager`'s current `Locale`.
    pub language: Language,
    /// `ascii.png` + `glyph_sizes.bin`, for S15's `FontRenderer` and S12's nameplates.
    pub font: FontData,
    /// The stitched block atlas. java: `textureMapBlocks`.
    pub atlas: mc_assets::Atlas,
    /// What baking found. java: vanilla logs missing models and carries on; so does this.
    pub bake: BakeReport,
    /// `sounds.json`, resolved across the whole pack stack.
    pub sounds: mc_audio::SoundRegistry,
    /// Whether the block models and atlas in this struct are the ones actually installed
    /// globally. True after [`GameAssets::load`] and after every [`GameAssets::reload_packs`];
    /// false in between [`GameAssets::boot`] and the [`AssetLoader`] reaching its model stage.
    pub models_installed: bool,
    /// Whether every stage has run. `false` for a [`GameAssets::boot`] result that an
    /// [`AssetLoader`] is still stepping through — the atlas is a 1×1 placeholder until then, so
    /// nothing may build a `TerrainPhase` from it.
    pub ready: bool,
    /// Bake with the Connected Textures mod's tiles and table. Set
    /// from the mod settings before the first bake; `false` bakes exactly vanilla.
    pub connected_textures: bool,
}

/// The bake-cache key tag of a connected-textures bake: its atlas holds the CTM tiles, so it
/// answers a different question than the vanilla bake of the same pack stack.
const CTM_BAKE_KEY: &[u8] = b"connected-textures v1";

impl std::fmt::Debug for GameAssets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GameAssets")
            .field("packs", &self.resources.pack_names())
            .field("language", &self.language.code())
            .field("atlas", &(self.atlas.w, self.atlas.h))
            .field("sounds", &self.sounds.len())
            .field("models_installed", &self.models_installed)
            .finish()
    }
}

/// java: the first `IResourceManagerReloadListener` pass — building the pack stack itself.
pub const STAGE_PACKS: &str = "Reloading resource packs";
/// java: `LanguageManager.onResourceManagerReload`.
pub const STAGE_LANGUAGE: &str = "Loading language";
/// java: `FontRenderer.onResourceManagerReload`.
pub const STAGE_FONTS: &str = "Loading fonts";
/// java: `ModelManager.onResourceManagerReload` — `ModelBakery` plus `TextureMap.loadSprites`.
pub const STAGE_MODELS: &str = "Baking block models";
/// java: `GrassColorReloadListener` and `FoliageColorReloadListener`.
pub const STAGE_COLORMAPS: &str = "Loading colormaps";
/// java: `SoundHandler.onResourceManagerReload`.
pub const STAGE_SOUNDS: &str = "Loading sounds";

/// The stages the loader walks, with weights measured against the real 1.8.9 assets: baking
/// 1251 block states and stitching the atlas dominates everything else combined.
fn stages() -> StageProgress {
    StageProgress::new(vec![
        (STAGE_PACKS, 2),
        (STAGE_LANGUAGE, 1),
        (STAGE_FONTS, 2),
        (STAGE_MODELS, 60),
        (STAGE_COLORMAPS, 1),
        (STAGE_SOUNDS, 4),
    ])
}

/// A 1×1 transparent atlas, standing in until the model stage runs.
///
/// Not an empty `Vec`: `TerrainRenderer::new` and `ParticlePhase::install_block_atlas` upload
/// `atlas.texture` as a `w × h` RGBA image, and a zero-sized wgpu texture is a validation error.
/// Nothing should reach it — [`GameAssets::ready`] says so — but a placeholder that survives being
/// uploaded turns a mistake into a blank texture instead of a device loss.
fn placeholder_atlas() -> mc_assets::Atlas {
    mc_assets::Atlas {
        texture: vec![0u8; 4],
        mipmaps: Vec::new(),
        w: 1,
        h: 1,
    }
}

/// The expensive half of an asset load, steppable one stage at a time.
///
/// java: the body of `SimpleReloadableResourceManager.reloadResources`' listener loop, which
/// vanilla runs to completion in one call. Splitting it is what lets the frame loop present the
/// progress screen between stages — see [`GameAssets::boot`].
#[derive(Debug)]
pub struct AssetLoader {
    stages: StageProgress,
    /// The texture limit used for both the bake and its cache identity. Deferred loading begins
    /// before a GPU exists, then the driver replaces the fallback with the granted device limit
    /// immediately before the model stage.
    atlas_max_dimension: u32,
    /// Where the last bake was stored and what question it answers, or `None` when the cache is
    /// off (`MC_RS_NO_BAKE_CACHE`) or the pack stack could not be stamped precisely enough to key
    /// one on. See [`bake_cache_inputs`].
    cache: Option<(PathBuf, BakeInputs)>,
}

/// Test builds only: the process-wide model registry's generation moves on every install, and
/// every test that builds a client loads assets. A test that needs the generation to hold still
/// (the atlas batch driver test) takes the write side; each install takes the read side. Added, after the no-assert suite caught that test reading "stale" while
/// a parallel test built a client.
#[cfg(test)]
pub(crate) fn installed_generation_lock() -> &'static std::sync::RwLock<()> {
    static LOCK: std::sync::OnceLock<std::sync::RwLock<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::RwLock::new(()))
}

/// First-wins on a cold process, replacing on a reload. [`mc_assets::install`] is kept for the
/// first call precisely so its documented semantics are unchanged.
///
/// `key` is the bake cache's identity of a clean bake (see
/// [`AssetLoader::bake_key`]): a reload whose inputs match a bake published before reuses that
/// bake instead of leaking another ~12 MB copy. `None` keeps the old one-copy-per-reload path.
fn install_or_reinstall(baked: mc_assets::BakedAssets, key: Option<String>) {
    #[cfg(test)]
    let _generation_guard = installed_generation_lock()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if mc_assets::model::installed_generation() == 0 {
        mc_assets::model::install_keyed(baked, key);
    } else {
        mc_assets::model::reinstall_keyed(baked, key);
    }
}

impl AssetLoader {
    /// What this loader's bake is, exactly: the on-disk cache's key
    /// for this pack stack, atlas limit, mipmap levels and connected-textures choice. `None`
    /// when the cache is off or the stack could not be stamped precisely.
    fn bake_key(&self) -> Option<String> {
        self.cache
            .as_ref()
            .filter(|(_, inputs)| inputs.is_stampable())
            .map(|(_, inputs)| inputs.key())
    }
    /// Try the on-disk bake cache. `None` means "bake it", for any reason at all — a miss is
    /// never an error here, and every reason is logged with the cache's own name for it.
    fn load_cached_bake(&self, assets: &GameAssets) -> Option<mc_assets::BakedAssets> {
        let (path, inputs) = self.cache.as_ref()?;
        let started = std::time::Instant::now();
        match mc_assets::cache::load(path, inputs) {
            CacheRead::Hit(snapshot) => {
                let assets_out = snapshot.into_assets();
                tracing::info!(
                    path = %path.display(),
                    ms = started.elapsed().as_secs_f64() * 1000.0,
                    packs = ?assets.resources.pack_names(),
                    "baked assets restored from cache"
                );
                Some(assets_out)
            }
            CacheRead::Miss(reason) => {
                tracing::info!(
                    path = %path.display(),
                    %reason,
                    "baking block models from scratch"
                );
                None
            }
        }
    }

    /// Store a finished bake for the next start. Failures are logged and dropped: a cache that
    /// cannot be written is a slower start, not a broken game.
    fn save_bake(&self, baked: &mc_assets::BakedAssets) {
        let Some((path, inputs)) = self.cache.as_ref() else {
            return;
        };
        let started = std::time::Instant::now();
        let snapshot = BakedSnapshot::of(baked);
        match mc_assets::cache::save(path, inputs, &snapshot) {
            Ok(()) => tracing::info!(
                path = %path.display(),
                ms = started.elapsed().as_secs_f64() * 1000.0,
                "baked assets cached"
            ),
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "could not cache the bake");
            }
        }
    }
}

impl AssetLoader {
    /// The stage [`AssetLoader::step`] will run next, for the loading screen's second line.
    #[must_use]
    pub fn message(&self) -> &'static str {
        self.stages.current().unwrap_or("Done")
    }

    /// Weighted progress, `0..=100`, for [`LoadingScreen::set_loading_progress`].
    #[must_use]
    pub fn percent(&self) -> i32 {
        self.stages.percent()
    }

    /// Whether every stage has run.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.stages.done()
    }

    /// Run exactly one stage. Cheap stages return in microseconds; the model bake is the one
    /// that stalls a frame, which is why it is announced before it runs.
    ///
    /// # Errors
    ///
    /// Model baking failed outright (the atlas could not be packed). A *partial* bake failure is
    /// recorded in [`GameAssets::bake`] and is not an error, exactly as vanilla substitutes
    /// `MissingModel` and carries on.
    pub fn step(
        &mut self,
        assets: &mut GameAssets,
        atlas_max_dimension: u32,
        config: &ClientConfig,
        settings: &GameSettings,
    ) -> Result<(), ClientError> {
        if self.atlas_max_dimension != atlas_max_dimension {
            self.atlas_max_dimension = atlas_max_dimension;
            self.cache = bake_cache_slot(config, settings, atlas_max_dimension);
        }
        match self.stages.current() {
            Some(STAGE_MODELS) => {
                // A connected-textures bake is keyed apart from the vanilla one.
                // This stage runs once per loader, so the tag is pushed once.
                if assets.connected_textures {
                    if let Some((_, inputs)) = self.cache.as_mut() {
                        inputs.push_opaque("connected_textures", CTM_BAKE_KEY);
                    }
                }
                if let Some(mut baked) = self.load_cached_bake(assets) {
                    if assets.connected_textures {
                        baked.attach_ctm(&assets.resources);
                    }
                    assets.atlas = baked.atlas.clone();
                    assets.bake = BakeReport {
                        failures: Vec::new(),
                        model_failures: Vec::new(),
                        model_states_considered: model_states_considered(),
                    };
                    // Only clean bakes are ever cached, so a cache hit is keyable.
                    install_or_reinstall(baked, self.bake_key());
                    assets.models_installed = true;
                    self.stages.finish_stage();
                    if self.stages.done() {
                        assets.ready = true;
                    }
                    return Ok(());
                }

                let (baked, bake) = mc_assets::model::bake_all_with_ctm(
                    &assets.resources,
                    self.atlas_max_dimension,
                    atlas_mipmap_levels(settings),
                    assets.connected_textures,
                )
                .map_err(assets_err)?;
                assets.atlas = baked.atlas.clone();
                if !bake.is_clean() {
                    // java: `ModelBakery` logs each failure and substitutes the missing model.
                    tracing::warn!(
                        state_failures = bake.failures.len(),
                        model_failures = bake.model_failures.len(),
                        considered = bake.model_states_considered,
                        "some block states or registered models did not bake"
                    );
                }
                // Only a *clean* bake is ever written. Persisting a bake that fell back to the
                // missing-model placeholder for some states would make a transient problem — a
                // pack half-copied, a file briefly locked — permanent, and the next start would
                // have no way to tell.
                // A bake that fell back to the missing model somewhere is never reused (the same
                // rule as the on-disk cache): a later reload with the same inputs bakes again.
                let key = if bake.is_clean() {
                    self.save_bake(&baked);
                    self.bake_key()
                } else {
                    None
                };
                assets.bake = bake;
                install_or_reinstall(baked, key);
                assets.models_installed = true;
            }
            Some(STAGE_COLORMAPS) => {
                if let Err(error) = mc_render::terrain::tint::reload_colormaps(&assets.resources) {
                    // java: the grass and foliage reload listeners catch IOException
                    // independently. `reload_colormaps` likewise publishes a successful sibling;
                    // a failed map retains its previous buffer, or the zero-filled (black)
                    // initial buffer when no successful load has happened yet.
                    tracing::warn!(%error, "one or more colormaps unavailable; failed map retained");
                }
            }
            Some(STAGE_SOUNDS) => {
                assets.sounds =
                    mc_audio::SoundRegistry::load(&assets.resources).unwrap_or_else(|error| {
                        tracing::warn!(%error, "sounds.json unavailable; running silent");
                        mc_audio::SoundRegistry::empty()
                    });
            }
            // The three boot stages are already finished by `GameAssets::boot`; anything else is
            // a stage list that grew without this match growing with it.
            Some(other) => tracing::warn!(stage = other, "no work registered for loading stage"),
            None => return Ok(()),
        }
        self.stages.finish_stage();
        if self.stages.done() {
            assets.ready = true;
        }
        Ok(())
    }
}

/// Build the resource-pack stack, lowest first.
///
/// java: `refreshResources` (`:790`) — `defaultResourcePacks`, then every enabled repository
/// entry in `gameSettings.resourcePacks` order, then the server-supplied pack. Later entries
/// win, which is `SimpleReloadableResourceManager`'s documented behaviour and S04's too.
///
/// The two default packs are this port's equivalent of vanilla's single `DefaultResourcePack`:
/// the launcher's object store (sounds, language files, the assets the jar does not carry) and
/// the version jar (models, blockstates, block textures). Vanilla gets the second for free by
/// running *inside* it.
///
/// A pack that fails to open is logged and skipped rather than fatal — vanilla's
/// `ResourcePackRepository` does the same, and a corrupt pack in a user's folder must not stop
/// the game starting.
#[must_use]
pub fn build_pack_stack(
    config: &ClientConfig,
    settings: &GameSettings,
) -> Vec<Box<dyn ResourcePack>> {
    build_pack_stack_with_sources(config, settings).0
}

/// java: the list `refreshResources`'s `catch (RuntimeException)` reloads
/// with: the default packs only. The server's pack is left out of it as well (it stays active,
/// as `resourcePackInstance` does, so the next refresh tries it again).
fn default_pack_stack(config: &ClientConfig) -> Vec<Box<dyn ResourcePack>> {
    let server = active_server_resource_pack_path(config);
    let (packs, sources) = build_pack_stack_with_sources(config, &GameSettings::default());
    packs
        .into_iter()
        .zip(sources)
        .filter(|(_, source)| server.as_ref() != Some(source))
        .map(|(pack, _)| pack)
        .collect()
}

/// The identity headers vanilla sends with a server-pack HTTP request.
///
/// java: `Minecraft.getSessionInfo()` returns `X-Minecraft-Username`,
/// `X-Minecraft-UUID`, and `X-Minecraft-Version` in addition to the request URL.  The token is
/// intentionally not part of this record: resource-pack downloads do not send it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerResourcePackHeaders {
    /// The current account name, when one is available.
    pub username: Option<String>,
    /// The current dashless account UUID, when one is available.
    pub uuid: Option<String>,
}

/// Return the cache key used by vanilla's `downloadResourcePack`.
///
/// `String.matches("^[a-f0-9]{40}$")` is deliberately narrower than "a string that looks like a
/// digest": uppercase hex, Unicode digits, and a digest with an extra byte all go to the literal
/// `legacy` cache slot.  The returned value is owned so the caller can safely move it into a
/// network task.
#[must_use]
pub fn server_resource_pack_cache_key(hash: &str) -> String {
    if hash.encode_utf16().count() == 40
        && hash
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        hash.to_owned()
    } else {
        "legacy".to_owned()
    }
}

/// `<game_dir>/server-resource-packs`, vanilla's dedicated server-pack cache directory.
#[must_use]
pub fn server_resource_pack_dir(config: &ClientConfig) -> PathBuf {
    config.game_dir.join(SERVER_RESOURCE_PACK_DIR)
}

/// The on-disk cache file for one S48 hash.  Vanilla deliberately has no `.zip` suffix.
#[must_use]
pub fn server_resource_pack_cache_path(config: &ClientConfig, hash: &str) -> PathBuf {
    server_resource_pack_dir(config).join(server_resource_pack_cache_key(hash))
}

fn server_resource_pack_active_marker(config: &ClientConfig) -> PathBuf {
    server_resource_pack_dir(config).join(SERVER_RESOURCE_PACK_ACTIVE_MARKER)
}

/// Resolve the child named by Java's `level://` resource-pack URL.
///
/// java: `NetHandlerPlayClient.handleResourcePack` strips the literal `level://` prefix and
/// constructs `new File(new File(mcDataDir, "saves"), child)`. The prefix is case-sensitive, and
/// the remainder is a file name/path rather than a URL path, so query characters and `#` are
/// intentionally preserved. Reject NUL and line breaks before the value can be written into the
/// active marker; they cannot name a useful pack on the supported filesystems and would make the
/// marker ambiguous.
#[must_use]
pub fn level_resource_pack_path(config: &ClientConfig, url: &str) -> Option<PathBuf> {
    let child = url.strip_prefix("level://")?;
    if child.is_empty() || child.chars().any(|ch| matches!(ch, '\0' | '\r' | '\n')) {
        return None;
    }
    // Only a plain relative path inside `saves`, which is what an
    // integrated server sends (`level://<world>/resources.zip`). Java's `new File(saves, child)`
    // follows `../`, and `Path::join` even replaces the base with an absolute child, so any
    // server could name any file on the PC and learn from the ACCEPTED / FAILED_DOWNLOAD answer
    // whether it exists (then have it opened as a pack). A rejected child answers exactly like a
    // missing file. A remote server never has a reason to send anything else.
    if !Path::new(child)
        .components()
        .all(|part| matches!(part, std::path::Component::Normal(_)))
    {
        return None;
    }
    Some(config.game_dir.join("saves").join(child))
}

fn write_server_resource_pack_active_marker(
    config: &ClientConfig,
    value: &str,
) -> Result<(), ClientError> {
    let marker = server_resource_pack_active_marker(config);
    std::fs::create_dir_all(
        marker
            .parent()
            .expect("server resource-pack marker has a parent"),
    )?;
    let temporary = marker.with_extension("tmp");
    std::fs::write(&temporary, value.as_bytes())?;
    // Windows does not replace an existing destination with rename, so remove the old marker
    // first. A stale/missing marker is harmless: `active_server_resource_pack_path` ignores it.
    let _ = std::fs::remove_file(&marker);
    std::fs::rename(&temporary, &marker)?;
    Ok(())
}

/// Mark a validated cache file as the pack that belongs above all user-selected packs.
///
/// The marker is intentionally separate from the cache file: cache files survive a restart, while
/// the active server instance must be cleared at disconnect and only becomes visible after an S48
/// download/cache hit has passed validation.
pub fn activate_server_resource_pack(
    config: &ClientConfig,
    hash: &str,
) -> Result<PathBuf, ClientError> {
    let key = server_resource_pack_cache_key(hash);
    let path = server_resource_pack_dir(config).join(&key);
    if !path.is_file() {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "server resource-pack cache file {} is missing",
                path.display()
            ),
        )));
    }
    write_server_resource_pack_active_marker(config, &key)?;
    Ok(path)
}

/// Validate and activate a local `level://` resource pack.
///
/// java: `NetHandlerPlayClient.handleResourcePack` accepts an existing save-file, then
/// `ResourcePackRepository.setResourcePackInstance` makes that file the highest-precedence pack
/// without copying it into `server-resource-packs`. The active marker stores the original URL so
/// a reload can reopen the same save file, while startup/disconnect still clears the instance.
pub fn activate_level_resource_pack(
    config: &ClientConfig,
    url: &str,
) -> Result<PathBuf, ClientError> {
    let path = level_resource_pack_path(config, url).ok_or_else(|| {
        ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid level resource-pack URL",
        ))
    })?;
    if !path.is_file() {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("level resource-pack file {} is missing", path.display()),
        )));
    }
    // `setResourcePackInstance` itself is lazy in Java, but a malformed archive would otherwise
    // be reported as SUCCESSFULLY_LOADED here and silently disappear during the later stack
    // build. Opening once keeps the C19 terminal status honest while retaining Java's file
    // ownership (the save file is never copied or deleted).
    ZipResourcePack::open(&path).map_err(|error| {
        ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("level resource-pack is not a readable ZIP archive: {error}"),
        ))
    })?;
    write_server_resource_pack_active_marker(config, url)?;
    Ok(path)
}

/// Clear the currently-active server or local level pack. The cached bytes remain for a later
/// hash hit. Returns whether an active marker was removed so the caller can schedule Java's
/// deferred resource refresh when an existing instance is replaced or disconnected.
pub fn clear_active_server_resource_pack(config: &ClientConfig) -> bool {
    let marker = server_resource_pack_active_marker(config);
    match std::fs::remove_file(&marker) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            tracing::debug!(path = %marker.display(), %error, "could not clear active server resource pack");
            false
        }
    }
}

/// Resolve the active marker into a cache or local-level path for `build_pack_stack`.
fn active_server_resource_pack_path(config: &ClientConfig) -> Option<PathBuf> {
    let marker = server_resource_pack_active_marker(config);
    let key = std::fs::read_to_string(marker).ok()?;
    if let Some(child) = key.strip_prefix("level://") {
        let path = level_resource_pack_path(config, &format!("level://{child}"))?;
        return path.is_file().then_some(path);
    }
    let key = key.trim();
    if key.is_empty()
        || key.contains('/')
        || key.contains('\\')
        || key != server_resource_pack_cache_key(key)
    {
        return None;
    }
    let path = server_resource_pack_dir(config).join(key);
    path.is_file().then_some(path)
}

/// Validate the cache entry for an S48 request.
///
/// java: `downloadResourcePack` only hashes an existing file when `hash.length() == 40`; malformed
/// hashes therefore use the `legacy` cache file without a digest gate.  A strict lowercase SHA-1
/// mismatch, an oversized file, or a malformed ZIP is deleted and treated as a cache miss.  The
/// caller can then perform the HTTP leg and report its eventual terminal C19 result.
pub fn cached_server_resource_pack(
    config: &ClientConfig,
    hash: &str,
) -> Result<Option<PathBuf>, ClientError> {
    let path = server_resource_pack_cache_path(config, hash);
    if !path.is_file() {
        return Ok(None);
    }

    let remove_bad_cache = |reason: &str| {
        tracing::warn!(path = %path.display(), %reason, "discarding server resource-pack cache entry");
        let _ = std::fs::remove_file(&path);
    };
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if metadata.len() > SERVER_RESOURCE_PACK_MAX_BYTES {
        remove_bad_cache("cache file exceeds the 50 MiB limit");
        return Ok(None);
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            remove_bad_cache("cache file could not be read");
            tracing::debug!(%error, "server resource-pack cache read failed");
            return Ok(None);
        }
    };
    if hash.encode_utf16().count() == 40 && sha1_hex(&bytes) != hash {
        remove_bad_cache("cache SHA-1 does not match the server hash");
        return Ok(None);
    }
    if ZipResourcePack::open(&path).is_err() {
        remove_bad_cache("cache file is not a readable ZIP resource pack");
        return Ok(None);
    }
    Ok(Some(path))
}

/// Keep the ten most recent server-pack cache files, matching `deleteOldServerResourcesPacks`.
pub fn prune_server_resource_pack_cache(config: &ClientConfig) {
    let Ok(entries) = std::fs::read_dir(server_resource_pack_dir(config)) else {
        return;
    };
    let mut files = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let is_file = entry.file_type().ok()?.is_file();
            let name = path.file_name()?.to_string_lossy();
            (is_file
                && name != SERVER_RESOURCE_PACK_ACTIVE_MARKER
                && !name.ends_with(SERVER_RESOURCE_PACK_PART_SUFFIX))
            .then(|| {
                let modified = entry
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(std::time::UNIX_EPOCH);
                (modified, path)
            })
        })
        .collect::<Vec<_>>();
    files.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    for (_, path) in files.into_iter().skip(SERVER_RESOURCE_PACK_CACHE_LIMIT) {
        tracing::info!(path = %path.display(), "deleting old server resource-pack cache entry");
        let _ = std::fs::remove_file(path);
    }
}

/// Download a server resource pack, atomically install it in the vanilla-shaped cache, and
/// validate both its size and archive shape before returning.
///
/// Java-shaped 301/302/303/307/308 redirects are followed for up to twenty hops, with absolute,
/// network-path, path-relative, and query-only `Location` values resolved against the current
/// URL. Reqwest supplies rustls TLS and SOCKS5H transport, while this function retains the
/// vanilla cache/hash/ZIP validation and bounded-body policy around it.
pub async fn download_server_resource_pack(
    config: &ClientConfig,
    url: &str,
    hash: &str,
    headers: &ServerResourcePackHeaders,
) -> Result<PathBuf, ClientError> {
    download_server_resource_pack_with_proxy(config, url, hash, headers, None).await
}

/// Download a server pack using an optional Java-compatible SOCKS5H proxy.
pub async fn download_server_resource_pack_with_proxy(
    config: &ClientConfig,
    url: &str,
    hash: &str,
    headers: &ServerResourcePackHeaders,
    proxy: Option<&mc_auth::SocksProxy>,
) -> Result<PathBuf, ClientError> {
    let client = build_server_resource_pack_client(proxy)?;
    let mut current_url = url.to_owned();
    let mut redirects = 0usize;
    let response = loop {
        let request = parse_server_resource_pack_url(&current_url)?;
        let response = request_server_resource_pack(&client, &request, headers).await?;
        if !matches!(response.status, 301 | 302 | 303 | 307 | 308) {
            break response;
        }
        if redirects >= SERVER_RESOURCE_PACK_MAX_REDIRECTS {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "server resource-pack URL exceeded the {SERVER_RESOURCE_PACK_MAX_REDIRECTS}-redirect limit"
                ),
            )));
        }
        let location = response.location.as_deref().ok_or_else(|| {
            ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "server resource-pack HTTP {} redirect has no Location header",
                    response.status
                ),
            ))
        })?;
        current_url = resolve_server_resource_pack_redirect(&request, location)?;
        redirects += 1;
    };

    let cache_dir = server_resource_pack_dir(config);
    // The worker owns the only blocking filesystem calls in this path.  `mc-client` deliberately
    // keeps Tokio's `fs` feature off; the runtime has two network workers, and these short cache
    // operations happen only once per server-pack push.
    std::fs::create_dir_all(&cache_dir)?;
    prune_server_resource_pack_cache(config);

    if !(200..300).contains(&response.status) {
        return Err(ClientError::Io(std::io::Error::other(format!(
            "server resource-pack URL returned HTTP {}",
            response.status
        ))));
    }
    if response.body.len() as u64 > SERVER_RESOURCE_PACK_MAX_BYTES {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "server resource-pack response exceeds the {}-byte limit",
                SERVER_RESOURCE_PACK_MAX_BYTES
            ),
        )));
    }

    let path = server_resource_pack_cache_path(config, hash);
    let temporary = path.with_extension(SERVER_RESOURCE_PACK_PART_SUFFIX.trim_start_matches('.'));
    let _ = std::fs::remove_file(&temporary);
    std::fs::write(&temporary, &response.body)?;

    let strict_hash = hash.encode_utf16().count() == 40;
    if strict_hash && sha1_hex(&response.body) != hash {
        let _ = std::fs::remove_file(&temporary);
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "downloaded server resource-pack SHA-1 does not match the server hash",
        )));
    }
    if ZipResourcePack::open(&temporary).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "downloaded server resource-pack is not a readable ZIP archive",
        )));
    }
    let _ = std::fs::remove_file(&path);
    std::fs::rename(&temporary, &path)?;
    Ok(path)
}

fn build_server_resource_pack_client(
    proxy: Option<&mc_auth::SocksProxy>,
) -> Result<reqwest::Client, ClientError> {
    let mut builder = reqwest::Client::builder()
        .user_agent(SERVER_RESOURCE_PACK_USER_AGENT)
        .timeout(SERVER_RESOURCE_PACK_HTTP_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none());
    if let Some(proxy) = proxy {
        builder = builder
            .proxy(reqwest::Proxy::all(proxy.to_reqwest_url()).map_err(reqwest_client_error)?);
    }
    builder.build().map_err(reqwest_client_error)
}

fn reqwest_client_error(error: reqwest::Error) -> ClientError {
    ClientError::Io(std::io::Error::other(error.to_string()))
}

/// Issue one request through the already-configured rustls/SOCKS client. Redirects remain outside
/// reqwest so each hop passes through our Java-compatible URL parser and limit.
async fn request_server_resource_pack(
    client: &reqwest::Client,
    request: &ServerResourcePackUrl,
    headers: &ServerResourcePackHeaders,
) -> Result<HttpResponse, ClientError> {
    let mut call = client
        .get(render_server_resource_pack_url(request))
        .header(reqwest::header::HOST, &request.host_header)
        .header(reqwest::header::ACCEPT, SERVER_RESOURCE_PACK_ACCEPT)
        .header(reqwest::header::CONNECTION, "close")
        .header("X-Minecraft-Version", "1.8.9");
    if let Some(username) = &headers.username {
        call = call.header("X-Minecraft-Username", username);
    }
    if let Some(uuid) = &headers.uuid {
        call = call.header("X-Minecraft-UUID", uuid);
    }
    let mut response = call.send().await.map_err(reqwest_client_error)?;
    let status = response.status().as_u16();
    let location = response
        .headers()
        .get(reqwest::header::LOCATION)
        .map(|value| {
            value.to_str().map(str::to_owned).map_err(|_| {
                ClientError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "server resource-pack redirect has a non-UTF-8 Location header",
                ))
            })
        })
        .transpose()?;

    // Do not accumulate an untrusted body without a hard cap. A redirect body is discarded and
    // the final body is streamed in chunks so Content-Length lies cannot bypass the 50 MiB cap.
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(reqwest_client_error)? {
        if body.len() as u64 + chunk.len() as u64 > SERVER_RESOURCE_PACK_MAX_BYTES {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "server resource-pack response exceeds the {SERVER_RESOURCE_PACK_MAX_BYTES}-byte limit"
                ),
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(HttpResponse {
        status,
        body,
        location,
    })
}

#[derive(Debug)]
struct ServerResourcePackUrl {
    scheme: String,
    host_header: String,
    path: String,
}

fn parse_server_resource_pack_url(url: &str) -> Result<ServerResourcePackUrl, ClientError> {
    if url.is_empty() || url.chars().any(|ch| ch.is_whitespace() || ch.is_control()) {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "server resource-pack URL contains whitespace or control characters",
        )));
    }
    let (scheme, rest) = url.split_once("://").ok_or_else(|| {
        ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "server resource-pack URL has no scheme",
        ))
    })?;
    let scheme = scheme.to_ascii_lowercase();
    let mut scheme_chars = scheme.chars();
    if !scheme_chars
        .next()
        .is_some_and(|ch| ch.is_ascii_alphabetic())
        || !scheme_chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.'))
    {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "server resource-pack URL has an invalid scheme",
        )));
    }
    if !matches!(scheme.as_str(), "http" | "https") {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("server resource-pack URL scheme is unsupported: {scheme}"),
        )));
    }
    let path_start = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..path_start];
    let suffix = &rest[path_start..];
    let suffix = suffix.split_once('#').map_or(suffix, |(path, _)| path);
    if authority.is_empty() || authority.contains('@') {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "server resource-pack URL has an invalid authority",
        )));
    }

    let (host_header, host_for_connect, _port) = if authority.starts_with('[') {
        let close = authority.find(']').ok_or_else(|| {
            ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "server resource-pack URL has an unterminated IPv6 host",
            ))
        })?;
        let host = &authority[1..close];
        if host.is_empty() || host.contains(['[', ']']) {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "server resource-pack URL has an invalid IPv6 host",
            )));
        }
        let authority_suffix = &authority[close + 1..];
        let port = if authority_suffix.is_empty() {
            if scheme == "https" {
                443
            } else {
                80
            }
        } else if let Some(port) = authority_suffix.strip_prefix(':') {
            port.parse::<u16>().map_err(|_| {
                ClientError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "server resource-pack URL has an invalid port",
                ))
            })?
        } else {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "server resource-pack URL has an invalid IPv6 authority",
            )));
        };
        (authority.to_owned(), host.to_owned(), port)
    } else {
        if authority.contains(['[', ']']) || authority.matches(':').count() > 1 {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "server resource-pack URL has an unbracketed IPv6 host",
            )));
        }
        let (host, port) = if let Some((host, port)) = authority.split_once(':') {
            if host.is_empty() {
                return Err(ClientError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "server resource-pack URL has an empty host",
                )));
            }
            let port = port.parse::<u16>().map_err(|_| {
                ClientError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "server resource-pack URL has an invalid port",
                ))
            })?;
            (host, port)
        } else {
            (authority, if scheme == "https" { 443 } else { 80 })
        };
        if host.is_empty() {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "server resource-pack URL has an empty host",
            )));
        }
        (authority.to_owned(), host.to_owned(), port)
    };
    if host_for_connect.contains(['/', '\\']) {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "server resource-pack URL has an invalid host",
        )));
    }
    let path = if suffix.is_empty() || suffix.starts_with('#') {
        "/".to_owned()
    } else if suffix.starts_with('?') {
        format!("/{suffix}")
    } else {
        suffix.to_owned()
    };
    Ok(ServerResourcePackUrl {
        scheme,
        host_header,
        path,
    })
}

fn render_server_resource_pack_url(request: &ServerResourcePackUrl) -> String {
    format!(
        "{}://{}{}",
        request.scheme, request.host_header, request.path
    )
}

/// Resolve one HTTP `Location` value against the URL that produced it.
///
/// Java delegates this to `URL`/`HttpURLConnection`, which accepts absolute URLs as well as
/// network-path, absolute-path, query-only, and ordinary path-relative references. This helper
/// keeps that useful subset deterministic without pulling a URL crate into `mc-client`; every
/// resulting absolute URL goes back through [`parse_server_resource_pack_url`] so redirects
/// cannot bypass the host, scheme, port, or control-character checks used for the first request.
fn resolve_server_resource_pack_redirect(
    base: &ServerResourcePackUrl,
    location: &str,
) -> Result<String, ClientError> {
    if location.len() > SERVER_RESOURCE_PACK_MAX_REDIRECT_LOCATION_BYTES
        || location
            .chars()
            .any(|ch| ch.is_whitespace() || ch.is_control() || ch == '\\')
    {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "server resource-pack redirect has an invalid Location header",
        )));
    }
    let location = location.split_once('#').map_or(location, |(path, _)| path);
    if location.is_empty() {
        return Ok(render_server_resource_pack_url(base));
    }

    // An explicit scheme must be absolute. Reject `javascript:`, `file:`, and malformed
    // `http:/...` references rather than treating them as a path under the original host.
    if location.contains("://") {
        return canonical_server_resource_pack_url(location);
    }
    let first_path_delimiter = location.find(['/', '?']).unwrap_or(location.len());
    if location
        .find(':')
        .is_some_and(|colon| colon < first_path_delimiter)
    {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "server resource-pack redirect uses an unsupported URL scheme",
        )));
    }

    let authority = format!("{}://{}", base.scheme, base.host_header);
    let base_path = base
        .path
        .split_once('?')
        .map_or(base.path.as_str(), |(path, _)| path);
    let candidate = if location.starts_with("//") {
        format!("{}:{}", base.scheme, location)
    } else if location.starts_with('/') {
        format!("{authority}{location}")
    } else if location.starts_with('?') {
        format!("{authority}{base_path}{location}")
    } else {
        let directory = base_path
            .rfind('/')
            .map_or("/", |slash| &base_path[..=slash]);
        format!("{authority}{directory}{location}")
    };
    canonical_server_resource_pack_url(&candidate)
}

fn canonical_server_resource_pack_url(url: &str) -> Result<String, ClientError> {
    let request = parse_server_resource_pack_url(url)?;
    let path = normalize_server_resource_pack_path(&request.path);
    Ok(format!(
        "{}://{}{}",
        request.scheme, request.host_header, path
    ))
}

/// Normalize dot segments in a redirect path, while retaining the query string verbatim.
///
/// `HttpURLConnection` hands path resolution to Java's URL implementation. The resource-pack
/// request does not need full URI percent-decoding, and decoding here would make `%2F` change
/// authorities/paths, so segments are joined without interpreting escapes.
fn normalize_server_resource_pack_path(path_and_query: &str) -> String {
    let (raw_path, query) = path_and_query
        .split_once('?')
        .map_or((path_and_query, None), |(path, query)| (path, Some(query)));
    let raw_path = if raw_path.starts_with('/') {
        raw_path
    } else {
        "/"
    };
    let trailing_slash = raw_path.ends_with('/');
    let mut segments = Vec::new();
    for segment in raw_path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    let mut normalized = String::from("/");
    normalized.push_str(&segments.join("/"));
    if trailing_slash && !normalized.ends_with('/') {
        normalized.push('/');
    }
    if let Some(query) = query {
        normalized.push('?');
        normalized.push_str(query);
    }
    normalized
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
    location: Option<String>,
}

#[cfg(test)]
fn decode_chunked_body(encoded: &[u8]) -> Result<Vec<u8>, ClientError> {
    let mut decoded = Vec::new();
    let mut cursor = 0usize;
    loop {
        let Some(line_end) = encoded[cursor..]
            .windows(2)
            .position(|window| window == b"\r\n")
        else {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "chunked server resource-pack response has no size line",
            )));
        };
        let line_end = cursor + line_end;
        let size_text = std::str::from_utf8(&encoded[cursor..line_end])
            .ok()
            .and_then(|line| line.split(';').next())
            .ok_or_else(|| {
                ClientError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "chunked server resource-pack size is not ASCII",
                ))
            })?;
        let size = usize::from_str_radix(size_text.trim(), 16).map_err(|_| {
            ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "chunked server resource-pack size is invalid",
            ))
        })?;
        cursor = line_end + 2;
        if size == 0 {
            // RFC 7230 permits optional trailer fields between the zero-sized chunk and the
            // final empty line. Java's `HttpURLConnection` consumes them before exposing the
            // body; doing the same avoids treating a perfectly ordinary `ETag` trailer as a
            // corrupt resource pack.
            loop {
                let Some(trailer_end) = encoded[cursor..]
                    .windows(2)
                    .position(|window| window == b"\r\n")
                else {
                    return Err(ClientError::Io(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "chunked server resource-pack response has no terminator",
                    )));
                };
                if trailer_end == 0 {
                    return Ok(decoded);
                }
                cursor += trailer_end + 2;
            }
        }
        let end = cursor.checked_add(size).ok_or_else(|| {
            ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "chunked server resource-pack size overflows",
            ))
        })?;
        let payload_end = end.checked_add(2).ok_or_else(|| {
            ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "chunked server resource-pack body size overflows",
            ))
        })?;
        if payload_end > encoded.len() || &encoded[end..payload_end] != b"\r\n" {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "chunked server resource-pack body is truncated",
            )));
        }
        decoded.extend_from_slice(&encoded[cursor..end]);
        if decoded.len() as u64 > SERVER_RESOURCE_PACK_MAX_BYTES {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "chunked server resource-pack exceeds the 50 MiB limit",
            )));
        }
        cursor = payload_end;
    }
}

/// SHA-1 in the exact lowercase hexadecimal form Guava's `Hashing.sha1().toString()` uses.
fn sha1_hex(bytes: &[u8]) -> String {
    let mut message = bytes.to_vec();
    let bit_len = (message.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut h = [
        0x6745_2301u32,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ];
    for chunk in message.chunks_exact(64) {
        let mut words = [0u32; 80];
        for (index, word) in words[..16].iter_mut().enumerate() {
            let start = index * 4;
            *word = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (index, &word) in words.iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let next = a
                .rotate_left(5)
                .wrapping_add(function)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

/// [`build_pack_stack`], plus the on-disk path each pack that actually opened came from.
///
/// Added by S23 so the bake cache can be keyed on the packs that were **really stacked** rather
/// than on the ones that were asked for: a jar that failed to open is skipped above, and a cache
/// key that claimed it was present would be a key for a stack that never existed. The two vectors
/// are the same length and in the same order, which
/// `the_cache_key_names_exactly_the_packs_that_opened` asserts.
///
/// One deliberate substitution: the object store's source is its **index file**
/// (`assets/indexes/<id>.json`), not the `assets/` directory. The store is content-addressed —
/// every object file is named by its own SHA-1 — so an unchanged index means an unchanged set of
/// resolvable resources, while stamping the directory means stat'ing 14,108 files on this
/// checkout, which costs several times what the whole cache saves. The residual hole is an object
/// file deleted or truncated on disk with the index left alone; that makes a *fresh* bake worse
/// (a missing texture) and leaves the cached one correct, so it is not a hole in the harmful
/// direction.
#[must_use]
pub fn build_pack_stack_with_sources(
    config: &ClientConfig,
    settings: &GameSettings,
) -> (Vec<Box<dyn ResourcePack>>, Vec<PathBuf>) {
    let mut packs: Vec<Box<dyn ResourcePack>> = Vec::new();
    let mut sources: Vec<PathBuf> = Vec::new();

    if config.assets_dir.join("indexes").is_dir() {
        match ObjectStorePack::open(&config.assets_dir, &config.asset_index) {
            Ok(pack) => {
                packs.push(Box::new(pack));
                sources.push(
                    config
                        .assets_dir
                        .join("indexes")
                        .join(format!("{}.json", config.asset_index)),
                );
            }
            Err(error) => tracing::warn!(
                dir = %config.assets_dir.display(), %error, "asset index unavailable"
            ),
        }
    }
    if let Some(jar) = &config.version_jar {
        match ZipResourcePack::open(jar) {
            Ok(pack) => {
                packs.push(Box::new(pack));
                sources.push(jar.clone());
            }
            Err(error) => tracing::warn!(jar = %jar.display(), %error, "version jar unavailable"),
        }
    }

    for name in &settings.resource_packs {
        let Some(path) = user_resource_pack_path(config, name) else {
            tracing::warn!(
                "a saved resource pack name that is not a pack folder entry was skipped"
            );
            continue;
        };
        match open_selected_user_pack(
            &path,
            settings
                .incompatible_resource_packs
                .iter()
                .any(|saved| saved == name),
        ) {
            Ok(Some(pack)) => {
                packs.push(pack);
                sources.push(path);
            }
            Ok(None) => tracing::warn!(pack = %path.display(), "resource pack skipped"),
            Err(error) => tracing::warn!(pack = %path.display(), %error, "resource pack skipped"),
        }
    }

    // java: `refreshResources` appends `ResourcePackRepository.resourcePackInstance` after the
    // default packs and every user-selected repository entry.  The marker is written only after
    // a cache hit/download has passed hash + ZIP validation, so this final slot has the highest
    // precedence and never exposes a half-written `.part` file.
    if let Some(path) = active_server_resource_pack_path(config) {
        match ZipResourcePack::open(&path) {
            Ok(pack) => {
                packs.push(Box::new(pack));
                sources.push(path);
            }
            Err(error) => {
                tracing::warn!(pack = %path.display(), %error, "active server resource pack unavailable")
            }
        }
    }
    debug_assert_eq!(packs.len(), sources.len());
    (packs, sources)
}

/// Environment variable that turns the on-disk bake cache off entirely.
///
/// Set it to any **non-empty** value to force a full bake and skip writing one; an empty value
/// counts as unset, so `MC_RS_NO_BAKE_CACHE=` in a shell script does not silently disable the
/// cache. It exists so the two startup paths can be measured against each other on the real
/// binary — a cache whose saving is only ever *computed* is not measured — and so a user who
/// suspects the cache can rule it out in one step without deleting files.
pub const NO_BAKE_CACHE_ENV: &str = "MC_RS_NO_BAKE_CACHE";

/// Whether the bake cache is in use for this process.
#[must_use]
pub fn bake_cache_enabled() -> bool {
    std::env::var_os(NO_BAKE_CACHE_ENV).is_none_or(|v| v.is_empty())
}

/// The question this process's cached bake would be an answer to.
///
/// Returns `None` when the stack could not be stamped precisely enough to key a cache on — a pack
/// that cannot be stat'd, or a folder pack with more entries than the stamp budget. That is not
/// an error: it means "bake it, and do not write a cache", which is exactly what
/// [`mc_assets::cache::save`] enforces on its own side too.
#[must_use]
pub fn bake_cache_inputs(config: &ClientConfig, settings: &GameSettings) -> Option<BakeInputs> {
    bake_cache_inputs_for_atlas(config, settings, MAX_ATLAS_DIMENSION)
}

/// The cache question for a particular GPU texture limit.
///
/// The atlas dimensions change the bake's pixel layout, so this must be the same cap passed to
/// [`mc_assets::bake_all`]. GPU-less startup keeps [`MAX_ATLAS_DIMENSION`] as its documented
/// fallback; the deferred and live-reload paths use the actual granted device limit.
#[must_use]
pub fn bake_cache_inputs_for_atlas(
    config: &ClientConfig,
    settings: &GameSettings,
    atlas_max_dimension: u32,
) -> Option<BakeInputs> {
    let (_, sources) = build_pack_stack_with_sources(config, settings);
    let mut inputs = BakeInputs::new(atlas_max_dimension);
    inputs.set_mipmap_levels(atlas_mipmap_levels(settings));
    for source in &sources {
        inputs.push_path(source);
    }
    inputs.is_stampable().then_some(inputs)
}

/// The block atlas's configured mip level count. java: `TextureMap.setMipmapLevels(
/// gameSettings.mipmapLevels)`; the options slider spans 0–4 and the atlas lowers it further to
/// what its smallest sprite supports. A hand-edited negative `options.txt` value bakes level 0.
/// §7.6 / ATLAS-4: this setting was previously read nowhere and the bake was hard-wired to 4.
#[must_use]
pub fn atlas_mipmap_levels(settings: &GameSettings) -> u32 {
    u32::try_from(settings.mipmap_levels).unwrap_or(0)
}

/// How many block states [`mc_assets::bake_all`] would consider, without running it.
///
/// A cache hit does not run the bake, so [`BakeReport::model_states_considered`] has to come from
/// somewhere. This is the definition `bake_all` itself uses — every legal state whose
/// `RenderType` is `Model` — and it cannot drift from the cached file, because
/// `mc_data::STATE_COUNT` is part of [`BakeInputs`]' key: a registry change invalidates every
/// cache before this number could disagree with one.
/// The cache slot an [`AssetLoader`] runs with: `None` when the cache is off or the stack cannot
/// be stamped.
fn bake_cache_slot(
    config: &ClientConfig,
    settings: &GameSettings,
    atlas_max_dimension: u32,
) -> Option<(PathBuf, BakeInputs)> {
    if !bake_cache_enabled() {
        tracing::info!("{NO_BAKE_CACHE_ENV} is set; baking block models every start");
        return None;
    }
    match bake_cache_inputs_for_atlas(config, settings, atlas_max_dimension) {
        Some(inputs) => Some((config.bake_cache_path(), inputs)),
        None => {
            tracing::info!(
                "a resource pack could not be identified precisely enough to key a bake cache on"
            );
            None
        }
    }
}

#[must_use]
fn model_states_considered() -> usize {
    mc_data::legal_states()
        .filter(|s| s.props().render_type == mc_data::RenderType::Model)
        .count()
}

/// Read the built-in row's metadata and icon.
///
/// Java's `DefaultResourcePack` obtains `pack.mcmeta` from the launcher object index but obtains
/// `/pack.png` from the client jar. This port splits those sources into `ObjectStorePack` and
/// `ZipResourcePack`, so the selection screen needs this small join operation to reconstruct the
/// one logical default pack. Missing metadata and image use the exact fallbacks from
/// `ResourcePackListEntryDefault` and `TextureUtil.missingTexture`.
#[must_use]
pub fn load_default_resource_pack_ui(config: &ClientConfig) -> ResourcePackUiData {
    let description = ObjectStorePack::open(&config.assets_dir, &config.asset_index)
        .ok()
        .and_then(|pack| pack.metadata().cloned())
        .map(|metadata| metadata.description)
        .unwrap_or_else(|| "\u{a7}cMissing pack.mcmeta :(".to_owned());

    let icon = config
        .version_jar
        .as_deref()
        .and_then(|jar| read_zip_pack_icon(jar).ok())
        .and_then(|bytes| decode_pack_icon(&bytes).ok())
        .unwrap_or_else(missing_pack_icon);

    ResourcePackUiData {
        description,
        pack_format: Some(i64::from(mc_assets::PACK_FORMAT_1_8_9)),
        icon: Some(icon),
    }
}

/// Read one installed user pack's row metadata and root `pack.png`.
///
/// A missing, malformed, oversized, or undecodable icon is deliberately represented by `None`:
/// `ResourcePackRepository.Entry.updateResourcePack` catches that image failure and substitutes
/// `DefaultResourcePack.getPackImage`. [`mc_ui::screens::options::packs::ResourcePacksScreen`]
/// performs the same substitution from its `default_icon`.
///
/// # Errors
///
/// The pack path itself could not be opened as a folder or zip resource pack. Icon failures are
/// not returned because vanilla keeps the row and falls back to the default image.
pub fn load_user_resource_pack_ui(path: &Path) -> Result<ResourcePackUiData, ClientError> {
    open_user_pack(path)?;
    let metadata = read_user_pack_metadata(path)?;
    let description = metadata
        .as_ref()
        .map(|metadata| metadata.description.clone())
        .unwrap_or_else(|| "\u{a7}cInvalid pack.mcmeta (or missing 'pack' section)".to_owned());
    let pack_format = metadata.map(|metadata| metadata.pack_format);

    let icon = match read_user_pack_icon(path).and_then(|bytes| decode_pack_icon(&bytes)) {
        Ok(icon) => Some(icon),
        Err(error) => {
            tracing::debug!(pack = %path.display(), %error, "resource-pack icon unavailable; using default");
            None
        }
    };

    Ok(ResourcePackUiData {
        description,
        pack_format,
        icon,
    })
}

/// Remove selected packs that vanilla's repository refuses because their declared format is not 1.
///
/// A pack whose metadata cannot be read is deliberately **not** removed: Java drops it from
/// `repositoryEntriesAll`, so the constructor never finds a matching entry to remove from
/// `GameSettings.resourcePacks`.
///
/// Call this immediately after loading settings and before building a resource stack.  It is
/// separate from [`build_pack_stack`] because startup callers that only borrow settings cannot
/// faithfully persist Java's `Iterator.remove()` side effect.
pub fn evict_unconfirmed_incompatible_user_packs(
    config: &ClientConfig,
    settings: &mut GameSettings,
) {
    let confirmed = settings.incompatible_resource_packs.clone();
    settings.resource_packs.retain(|name| {
        // Not an entry name: vanilla finds no entry for it, so it keeps it (see below).
        let Some(path) = user_resource_pack_path(config, name) else {
            return true;
        };
        match read_user_pack_metadata(&path) {
            Ok(Some(metadata)) => {
                metadata.is_supported() || confirmed.iter().any(|saved| saved == name)
            }
            // Java's constructor only removes a name after finding a usable repository entry.
            // Missing, malformed, and no-`pack` metadata never produce one, so retain their
            // options.txt value even though the stack below refuses to load it.
            Ok(None) | Err(_) => true,
        }
    });
}

/// Open a selected pack only when it can be an entry in vanilla's repository and either uses
/// format 1 or was explicitly confirmed through `incompatibleResourcePacks`.
fn open_selected_user_pack(
    path: &Path,
    incompatible_was_confirmed: bool,
) -> Result<Option<Box<dyn ResourcePack>>, ClientError> {
    let Some(metadata) = read_user_pack_metadata(path)? else {
        return Ok(None);
    };
    if !metadata.is_supported() && !incompatible_was_confirmed {
        return Ok(None);
    }
    open_user_pack(path).map(Some)
}

/// Metadata as `ResourcePackRepository.Entry.updateResourcePack` sees it.
///
/// A syntactically valid root with no `pack` member returns `Ok(None)`: Java's
/// `IMetadataSerializer.parseMetadataSection` does the same and the UI displays its red
/// "Invalid pack.mcmeta" description.  Missing input, invalid JSON, a non-object root, a
/// non-object `pack`, or a malformed `pack` section instead makes `updateResourcePack` throw;
/// `updateRepositoryEntriesAll` then omits the entry entirely.
fn read_user_pack_metadata(path: &Path) -> Result<Option<mc_assets::PackMetadata>, ClientError> {
    if !is_user_resource_pack_candidate(path) {
        return Err(not_a_user_resource_pack(path));
    }
    let bytes = if path.is_dir() {
        // The zip branch's cap (1 MiB): the pack list and the start-up
        // eviction read every folder pack's metadata, and a hostile one used to be read whole.
        read_capped_file(&path.join("pack.mcmeta"), mc_assets::MAX_MCMETA_BYTES)?
    } else {
        let pack = ZipResourcePack::open(path).map_err(assets_err)?;
        pack.read_raw("pack.mcmeta").map_err(assets_err)?
    };
    let root: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid pack.mcmeta: {error}"),
        ))
    })?;
    let object = root.as_object().ok_or_else(|| {
        ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid pack.mcmeta: expected a JSON object",
        ))
    })?;
    if !object.contains_key("pack") {
        return Ok(None);
    }
    mc_assets::PackMetadata::parse(&bytes)
        .map(Some)
        .map_err(assets_err)
}

fn read_user_pack_icon(path: &Path) -> Result<Vec<u8>, ClientError> {
    if path.is_dir() {
        read_capped_file(&path.join("pack.png"), MAX_PACK_ICON_FILE_BYTES)
    } else if path.is_file() {
        read_zip_pack_icon(path)
    } else {
        Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("resource pack {} does not exist", path.display()),
        )))
    }
}

fn read_zip_pack_icon(path: &Path) -> Result<Vec<u8>, ClientError> {
    let limits = ZipLimits {
        max_entry_size: MAX_PACK_ICON_FILE_BYTES,
        ..ZipLimits::default()
    };
    let pack = ZipResourcePack::open_with_limits(path, limits).map_err(assets_err)?;
    pack.read_raw("pack.png").map_err(assets_err)
}

fn read_capped_file(path: &Path, limit: u64) -> Result<Vec<u8>, ClientError> {
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{} exceeds the {limit}-byte pack-icon limit",
                path.display()
            ),
        )));
    }
    Ok(bytes)
}

fn decode_pack_icon(bytes: &[u8]) -> Result<PackIconImage, ClientError> {
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), image::ImageFormat::Png);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_PACK_ICON_DIMENSION);
    limits.max_image_height = Some(MAX_PACK_ICON_DIMENSION);
    limits.max_alloc = Some(MAX_PACK_ICON_ALLOC_BYTES);
    reader.limits(limits);
    let image = reader.decode().map_err(|error| {
        ClientError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid pack.png: {error}"),
        ))
    })?;
    let rgba = image.into_rgba8();
    Ok(PackIconImage {
        width: rgba.width(),
        height: rgba.height(),
        rgba: rgba.into_raw(),
    })
}

fn missing_pack_icon() -> PackIconImage {
    const SIZE: u32 = 16;
    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            if (x < SIZE / 2) == (y < SIZE / 2) {
                // Java literal -524040 = ARGB FFF800F8.
                rgba.extend_from_slice(&[0xF8, 0x00, 0xF8, 0xFF]);
            } else {
                rgba.extend_from_slice(&[0x00, 0x00, 0x00, 0xFF]);
            }
        }
    }
    PackIconImage {
        rgba,
        width: SIZE,
        height: SIZE,
    }
}

/// Open a user resource pack, which vanilla allows to be either a `.zip` or a directory.
///
/// java: `ResourcePackRepository.updateRepositoryEntriesAll` filters on
/// `RESOURCE_PACK_FILTER`, which accepts a directory containing `pack.mcmeta` or a `.zip` file.
///
/// # Errors
///
/// The pack is unreadable, or does not pass vanilla's repository file filter.
pub fn open_user_pack(path: &Path) -> Result<Box<dyn ResourcePack>, ClientError> {
    if path.is_dir() && path.join("pack.mcmeta").is_file() {
        Ok(Box::new(
            FolderResourcePack::open(path).map_err(assets_err)?,
        ))
    } else if is_user_resource_pack_candidate(path) {
        Ok(Box::new(ZipResourcePack::open(path).map_err(assets_err)?))
    } else {
        Err(not_a_user_resource_pack(path))
    }
}

/// `ResourcePackRepository.resourcePackFilter` verbatim: a lower-case `.zip` file, or a
/// directory whose root `pack.mcmeta` is itself a file.  The driver uses this same boundary for
/// directory enumeration, so it cannot grow a second approximation of vanilla's filter.
#[must_use]
pub fn is_user_resource_pack_candidate(path: &Path) -> bool {
    (path.is_file()
        && path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().ends_with(".zip")))
        || (path.is_dir() && path.join("pack.mcmeta").is_file())
}

fn not_a_user_resource_pack(path: &Path) -> ClientError {
    ClientError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "resource pack {} is not a vanilla resource-pack candidate",
            path.display()
        ),
    ))
}

/// `mc_assets::AssetError` has no `From` into [`ClientError`] — S14 froze that enum and this
/// section does not own it — so asset failures are carried as I/O errors with the original
/// message preserved.
fn assets_err(error: mc_assets::AssetError) -> ClientError {
    ClientError::Io(std::io::Error::other(error.to_string()))
}

impl GameAssets {
    /// Load everything, reporting progress as it goes.
    ///
    /// `progress` is called with `(message, percent)` before each stage and once at the end.
    /// It is the seam the loading screen plugs into and the reason this function needs no
    /// window: a test passes a closure that records, the game passes one that draws.
    ///
    /// java: `startGame`'s `refreshResources()` + the `TextureMap`/`ModelManager` block, whose
    /// progress vanilla reports through `IProgressUpdate`.
    ///
    /// # Errors
    ///
    /// The vanilla assets are missing or unreadable, or model baking fails outright. A *partial*
    /// bake failure is not an error: `BakeReport::failures` records it and the placeholder model
    /// is used, exactly as vanilla substitutes `MissingModel`.
    pub fn load(
        config: &ClientConfig,
        settings: &GameSettings,
        progress: &mut dyn FnMut(&str, i32),
    ) -> Result<GameAssets, ClientError> {
        mc_core::profile_scope!("assets.load");
        let (mut assets, mut loader) = GameAssets::boot(config, settings, progress)?;
        while !loader.is_done() {
            progress(loader.message(), loader.percent());
            // This synchronous/test path has no GPU yet, so retain the conservative documented
            // fallback. The windowed path replaces it with `Gpu::atlas_max_dimension` before
            // `AssetLoader` reaches `STAGE_MODELS`.
            loader.step(&mut assets, MAX_ATLAS_DIMENSION, config, settings)?;
        }
        progress("Done", 100);
        Ok(assets)
    }

    /// The cheap part of the load: the pack stack, the language table and the font.
    ///
    /// This is everything the loading screen itself needs — a `FontRenderer` and a
    /// `ResourceManager` to pull `gui/options_background.png` from — and it is deliberately the
    /// part that costs milliseconds. The expensive stages come back as an [`AssetLoader`] the
    /// caller steps through while presenting frames.
    ///
    /// java: `Minecraft.startGame` creates the display (`:488`) *before* `refreshResources`
    /// (`:497`), so vanilla's window is open — blank — for the whole resource load too. Showing
    /// the progress bar during it rather than a blank window is this port's deliberate
    /// improvement, on the same footing as the weighted stages.
    ///
    /// The returned [`GameAssets`] has `ready == false`, `models_installed == false` and a 1×1
    /// placeholder [`GameAssets::atlas`]; nothing may build a terrain phase from it yet.
    ///
    /// # Errors
    ///
    /// No resource pack is readable at all, or the font data is missing or malformed.
    pub fn boot(
        config: &ClientConfig,
        settings: &GameSettings,
        progress: &mut dyn FnMut(&str, i32),
    ) -> Result<(GameAssets, AssetLoader), ClientError> {
        mc_core::profile_scope!("assets.boot");
        let mut stages = stages();

        // ---- packs ----
        progress(stages.current().unwrap_or("Reloading resource packs"), 0);
        let mut resources = ResourceManager::new();
        let packs = build_pack_stack(config, settings);
        if packs.is_empty() {
            return Err(ClientError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "no resource packs: neither {} nor a version jar is readable",
                    config.assets_dir.display()
                ),
            )));
        }
        for pack in packs {
            resources.push_pack(pack);
        }
        stages.finish_stage();

        // ---- language ----
        if let Some(name) = stages.current() {
            progress(name, stages.percent());
        }
        let language = load_language(&resources, settings);
        stages.finish_stage();

        // ---- font ----
        if let Some(name) = stages.current() {
            progress(name, stages.percent());
        }
        let font = FontData::load(&resources)
            .map_err(|error| ClientError::Io(std::io::Error::other(format!("font: {error}"))))?;
        stages.finish_stage();

        let assets = GameAssets {
            resources,
            language,
            font,
            atlas: placeholder_atlas(),
            bake: BakeReport {
                failures: Vec::new(),
                model_failures: Vec::new(),
                model_states_considered: 0,
            },
            sounds: mc_audio::SoundRegistry::empty(),
            models_installed: false,
            ready: false,
            connected_textures: false,
        };
        Ok((
            assets,
            AssetLoader {
                stages,
                atlas_max_dimension: MAX_ATLAS_DIMENSION,
                cache: bake_cache_slot(config, settings, MAX_ATLAS_DIMENSION),
            },
        ))
    }

    /// Swap the resource-pack stack and re-derive **everything**, without a restart.
    ///
    /// java: `Minecraft.refreshResources` (`:788`), including its recovery path — a stack that
    /// fails to produce a usable language or font is abandoned, the reference pack selection is
    /// cleared, and the defaults are reloaded.
    ///
    /// Re-derived here: the language table, the font data, the baked block models, the item
    /// models, the stitched block atlas, the biome colormaps and the sound registry. The *render
    /// phases* are not — they are the caller's, and they must be rebuilt because a re-stitched
    /// atlas is baked into `TerrainPhase`'s bind group. See
    /// [`crate::app::driver::Client::reload_resource_packs`].
    ///
    /// # Errors
    ///
    /// The *default* stack is unreadable, which means the game cannot continue at all. A failure
    /// caused by a user pack is recovered from instead, and reported through the returned
    /// [`ReloadOutcome`].
    pub fn reload_packs(
        &mut self,
        config: &ClientConfig,
        settings: &mut GameSettings,
        progress: &mut dyn FnMut(&str, i32),
    ) -> Result<ReloadOutcome, ClientError> {
        self.reload_packs_with_atlas_dimension(config, settings, MAX_ATLAS_DIMENSION, progress)
    }

    /// Reload packs using the texture limit granted to the live GPU.
    ///
    /// [`GameAssets::reload_packs`] remains the GPU-less fallback for tests and early startup;
    /// the windowed driver uses this entry point once it has a [`mc_render::Gpu`].
    pub fn reload_packs_with_atlas_dimension(
        &mut self,
        config: &ClientConfig,
        settings: &mut GameSettings,
        atlas_max_dimension: u32,
        progress: &mut dyn FnMut(&str, i32),
    ) -> Result<ReloadOutcome, ClientError> {
        mc_core::profile_scope!("assets.reload_packs");
        let generation = self.resources.generation();
        let mut reverted = false;

        progress(STAGE_PACKS, 0);
        self.resources.reload(build_pack_stack(config, settings));

        progress(STAGE_LANGUAGE, 2);
        let mut language = load_language(&self.resources, settings);

        progress(STAGE_FONTS, 4);
        let font = match FontData::load(&self.resources) {
            Ok(font) => font,
            Err(error) => {
                // java: the `catch (RuntimeException)` arm — drop every user pack and retry.
                tracing::warn!(%error, "resource pack broke the font; reverting to defaults");
                settings.resource_packs.clear();
                settings.incompatible_resource_packs.clear();
                self.resources.reload(default_pack_stack(config));
                reverted = true;
                language = load_language(&self.resources, settings);
                FontData::load(&self.resources).map_err(|error| {
                    ClientError::Io(std::io::Error::other(format!("font: {error}")))
                })?
            }
        };
        self.language = language;
        self.font = font;

        // The remaining stages are exactly the ones a cold load runs, driven by the same code so
        // a reload and a restart cannot diverge. `ready`/`models_installed` end up true either
        // way, because after this the globals really do describe the new stack.
        let reload_loader = |settings: &GameSettings, reverted: bool| AssetLoader {
            stages: {
                let mut stages = stages();
                for _ in 0..3 {
                    stages.finish_stage();
                }
                stages
            },
            // Recomputed against the *new* selection, which is the whole point: a reload that
            // added a pack must not be answered from the cache the old stack wrote.
            atlas_max_dimension,
            // A reverted stack is the default packs alone, but the slot
            // stamps the server's pack, which stays active (see `default_pack_stack`): the
            // reverted bake would be saved under that stack's key and answered with the model
            // set published for it, while the atlas uploaded is this bake's. Not cached then;
            // without a server pack the slot describes the defaults exactly.
            cache: if reverted && active_server_resource_pack_path(config).is_some() {
                None
            } else {
                bake_cache_slot(config, settings, atlas_max_dimension)
            },
        };
        let mut loader = reload_loader(settings, reverted);
        while !loader.is_done() {
            progress(loader.message(), loader.percent());
            if let Err(error) = loader.step(self, atlas_max_dimension, config, settings) {
                // java: the same `catch (RuntimeException)` covers the
                // stitch (`StitcherException`): a pack with a texture larger than the device's
                // atlas is dropped with every other user pack, instead of leaving the stack half
                // loaded and the selection saved, to fail again at every start.
                if reverted {
                    return Err(error);
                }
                tracing::warn!(%error, "resource pack broke the block atlas; reverting to defaults");
                settings.resource_packs.clear();
                settings.incompatible_resource_packs.clear();
                self.resources.reload(default_pack_stack(config));
                reverted = true;
                self.language = load_language(&self.resources, settings);
                self.font = FontData::load(&self.resources).map_err(|error| {
                    ClientError::Io(std::io::Error::other(format!("font: {error}")))
                })?;
                loader = reload_loader(settings, reverted);
            }
        }
        progress("Done", 100);

        debug_assert_ne!(self.resources.generation(), generation);
        Ok(ReloadOutcome {
            reverted,
            generation: self.resources.generation(),
            packs: self.resources.pack_names().len(),
            model_generation: mc_assets::model::installed_generation(),
        })
    }
}

/// What a pack reload did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReloadOutcome {
    /// The reference pack selection was abandoned because it broke the game.
    /// java: `refreshResources`'s `catch (RuntimeException)` arm.
    pub reverted: bool,
    /// [`ResourceManager::generation`] afterwards, which every reload listener saw.
    pub generation: u64,
    /// How many packs are now stacked.
    pub packs: usize,
    /// [`mc_assets::model::installed_generation`] afterwards. Advances on every reload, which is
    /// how a caller can assert the *block models* really were re-baked rather than just the
    /// language table.
    pub model_generation: u64,
}

/// java: `LanguageManager.onResourceManagerReload` — `en_US` is always loaded first as the
/// fallback, then the raw selected language on top, so a partially-translated pack falls back
/// per key rather than per file. [`Language::load`] separately resolves the returned code to the
/// advertised `en_US` descriptor when the selected code is unknown, which keeps bidi/font
/// direction consistent with `LanguageManager.getCurrentLanguage()`.
fn load_language(resources: &ResourceManager, settings: &GameSettings) -> Language {
    let selected = if settings.language.is_empty() {
        mc_assets::DEFAULT_LANGUAGE
    } else {
        settings.language.as_str()
    };
    if selected == mc_assets::DEFAULT_LANGUAGE {
        Language::load(resources, &[mc_assets::DEFAULT_LANGUAGE])
    } else {
        Language::load(resources, &[mc_assets::DEFAULT_LANGUAGE, selected])
    }
}

/// Drive a [`LoadingScreen`] from a progress report, redrawing at vanilla's rate.
///
/// Returns `true` when the screen wants a frame. This is the glue between
/// [`GameAssets::load`]'s callback and [`crate::app::driver`]'s frame loop, kept here so the
/// loader itself never mentions a screen.
pub fn report_to_screen(
    screen: &mut LoadingScreen,
    message: &str,
    percent: i32,
    now_ms: u64,
) -> bool {
    if screen.message != message {
        screen.display_loading_string(message);
    }
    screen.set_loading_progress(percent, now_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_assets::ResourceLocation;

    #[test]
    fn fallback_resource_packs_are_discovered_loaded_and_checked_with_primary_precedence() {
        let dir = tempfile::tempdir().expect("tempdir");
        let primary = dir.path().join("client/resourcepacks");
        let fallback = dir.path().join(".minecraft/resourcepacks");
        let make = |root: &Path, name: &str, format: u8| {
            let pack = root.join(name);
            std::fs::create_dir_all(&pack).expect("pack directory");
            std::fs::write(
                pack.join("pack.mcmeta"),
                format!(r#"{{"pack":{{"pack_format":{format},"description":"{name}"}}}}"#),
            )
            .expect("metadata");
            pack
        };
        let fallback_only = make(&fallback, "Fallback", 1);
        let shared_name = if cfg!(target_os = "windows") {
            "shared"
        } else {
            "Shared"
        };
        make(&fallback, shared_name, 2);
        let config = ClientConfig {
            resource_pack_dir: primary.clone(),
            resource_pack_fallback_dir: Some(fallback.clone()),
            ..ClientConfig::default()
        };
        // Even when the client pack folder does not exist, the alternative is usable.
        assert_eq!(user_resource_pack_candidates(&config).len(), 2);
        assert_eq!(
            user_resource_pack_path(&config, "Fallback"),
            Some(fallback_only.clone())
        );
        assert_eq!(
            load_user_resource_pack_ui(&fallback_only)
                .unwrap()
                .description,
            "Fallback\u{a7}r"
        );
        let primary_shared = make(&primary, "Shared", 1);
        let candidates = user_resource_pack_candidates(&config);
        assert_eq!(candidates.len(), 2, "duplicate names appear only once");
        assert_eq!(
            user_resource_pack_path(&config, "Shared"),
            Some(primary_shared.clone())
        );
        let mut settings = GameSettings {
            resource_packs: vec!["Fallback".into(), "Shared".into()],
            ..GameSettings::default()
        };
        evict_unconfirmed_incompatible_user_packs(&config, &mut settings);
        assert_eq!(
            settings.resource_packs.len(),
            2,
            "primary compatible pack wins"
        );
        let (packs, sources) = build_pack_stack_with_sources(&config, &settings);
        assert_eq!(packs.len(), 2);
        assert_eq!(sources, [fallback_only, primary_shared]);
        make(&fallback, "Fallback", 2);
        evict_unconfirmed_incompatible_user_packs(&config, &mut settings);
        assert_eq!(
            settings.resource_packs,
            ["Shared"],
            "fallback compatibility checked too"
        );
        let config = ClientConfig {
            resource_pack_fallback_dir: Some(primary),
            ..config
        };
        assert_eq!(
            user_resource_pack_candidates(&config).len(),
            1,
            "same folder is not duplicated"
        );
    }

    /// A saved name is only ever an entry of a pack folder: a name in
    /// `options.txt` that is a path (absolute, `..`, a network share) selects nothing, is never
    /// opened, and stays in the list, as vanilla keeps a name it found no entry for.
    #[test]
    fn a_saved_pack_name_never_leaves_the_pack_folders() {
        let dir = tempfile::tempdir().expect("tempdir");
        let packs = dir.path().join("game/resourcepacks");
        let make = |pack: &Path| {
            std::fs::create_dir_all(pack).expect("pack directory");
            std::fs::write(
                pack.join("pack.mcmeta"),
                r#"{"pack":{"pack_format":1,"description":"x"}}"#,
            )
            .expect("metadata");
        };
        let inside = packs.join("Inside");
        let outside = dir.path().join("elsewhere/Outside");
        make(&inside);
        make(&outside);
        let config = ClientConfig {
            resource_pack_dir: packs.clone(),
            resource_pack_fallback_dir: None,
            ..ClientConfig::default()
        };
        let absolute = outside.to_string_lossy().into_owned();
        let climbing = "../../elsewhere/Outside".to_owned();
        let mut settings = GameSettings {
            resource_packs: vec!["Inside".into(), absolute.clone(), climbing.clone()],
            ..GameSettings::default()
        };
        evict_unconfirmed_incompatible_user_packs(&config, &mut settings);
        assert_eq!(
            settings.resource_packs,
            ["Inside".to_owned(), absolute.clone(), climbing.clone()],
            "names that select nothing stay saved"
        );
        let (loaded, sources) = build_pack_stack_with_sources(&config, &settings);
        assert_eq!(loaded.len(), 1, "only the pack inside the folder loads");
        assert_eq!(sources, [inside]);
        assert_eq!(user_resource_pack_path(&config, &absolute), None);
        assert_eq!(user_resource_pack_path(&config, &climbing), None);

        let mut refused = vec!["", ".", "..", "../x.zip", "a/b.zip", "pack/.", "nul\0.zip"];
        if cfg!(windows) {
            refused.extend([
                r"\\attacker.invalid\share\pack.zip",
                r"\\?\C:\pack.zip",
                r"C:\pack.zip",
                "C:pack.zip",
                r"\pack.zip",
                r"a\b.zip",
                "pack.zip:stream",
                "CON",
                "com1.zip",
                "NUL.zip",
            ]);
        } else {
            refused.push("/etc/pack.zip");
        }
        for name in refused {
            assert!(!is_plain_pack_name(name), "{name:?} must name no pack");
        }
        for plain in [
            "Inside",
            "My Pack.zip",
            "pack.v2.zip",
            "\u{fc}mlaut 32x.zip",
            "CONSOLE.zip",
            "COM10.zip",
            "...x.zip",
        ] {
            assert!(is_plain_pack_name(plain), "{plain:?} is a plain entry name");
        }
    }

    fn config() -> Option<ClientConfig> {
        let config = ClientConfig::default().with_discovered_assets();
        config.version_jar.as_ref()?.is_file().then_some(config)
    }

    /// The stack must be object store first, jar on top — the order every other section's tests
    /// use, and the one that makes the jar's models win over the launcher's copies.
    #[test]
    fn the_default_pack_stack_is_object_store_then_jar() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let settings = GameSettings::default();
        let packs = build_pack_stack(&config, &settings);
        assert_eq!(packs.len(), 2, "object store + version jar");
        assert!(
            packs[0].name().contains("index"),
            "the object store goes underneath, got {:?}",
            packs[0].name()
        );
        assert!(packs[1].name().contains("1.8.9"), "{:?}", packs[1].name());
    }

    /// The full load against the real 1.8.9 assets, with the progress callback recorded.
    #[test]
    fn loading_the_real_assets_reports_monotonic_progress() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let settings = GameSettings::default();
        let mut reports: Vec<(String, i32)> = Vec::new();
        let assets = GameAssets::load(&config, &settings, &mut |message, percent| {
            reports.push((message.to_owned(), percent));
        })
        .expect("the real assets must load");

        assert!(reports.len() >= 7, "{reports:?}");
        assert_eq!(reports[0].1, 0);
        assert_eq!(reports.last().unwrap().1, 100);
        assert!(
            reports.windows(2).all(|w| w[0].1 <= w[1].1),
            "progress must never go backwards: {reports:?}"
        );
        assert!(
            reports.iter().any(|(m, _)| m == "Baking block models"),
            "{reports:?}"
        );

        // The results must be usable, not merely present.
        assert!(assets.atlas.w.is_power_of_two() && assets.atlas.h.is_power_of_two());
        assert_eq!(
            assets.atlas.texture.len(),
            (assets.atlas.w * assets.atlas.h * 4) as usize
        );
        assert_eq!(assets.font.char_widths().len(), 256);
        assert_eq!(assets.language.tr("menu.multiplayer"), "Multiplayer");
        assert!(assets.sounds.len() > 100, "{}", assets.sounds.len());
        assert!(assets.models_installed);
        assert!(
            assets.bake.model_states_considered > 1_000,
            "{}",
            assets.bake.model_states_considered
        );
        // Stone must have a real model, which is the cheapest proof `install` took effect.
        assert!(!mc_assets::model_for(mc_data::BlockState { id: 1 << 4 }).is_empty());
    }

    /// A missing assets directory *and* a missing jar is the one unrecoverable case.
    #[test]
    fn loading_with_no_packs_at_all_is_an_error() {
        let config = ClientConfig {
            assets_dir: PathBufFromStr("/definitely/not/here/assets"),
            version_jar: None,
            ..ClientConfig::default()
        };
        let settings = GameSettings::default();
        let error =
            GameAssets::load(&config, &settings, &mut |_, _| {}).expect_err("no packs must fail");
        assert!(error.to_string().contains("no resource packs"), "{error}");
    }

    #[allow(non_snake_case)]
    fn PathBufFromStr(s: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(s)
    }

    /// The reload path, proven by an override that only a user pack can supply: a folder pack
    /// carrying its own `en_US.lang` must change what `tr` answers, live, and removing it must
    /// change it back.
    #[test]
    fn a_user_pack_hot_reloads_the_language_table() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("Loud");
        let lang_dir = pack.join("assets").join("minecraft").join("lang");
        std::fs::create_dir_all(&lang_dir).expect("mkdir");
        std::fs::write(
            pack.join("pack.mcmeta"),
            br#"{"pack":{"pack_format":1,"description":"test"}}"#,
        )
        .expect("write mcmeta");
        std::fs::write(
            lang_dir.join("en_US.lang"),
            "menu.multiplayer=MULTIPLAYER!!!\n",
        )
        .expect("write lang");

        let mut config = config;
        config.resource_pack_dir = dir.path().to_path_buf();
        let mut settings = GameSettings::default();

        let mut assets =
            GameAssets::load(&config, &settings, &mut |_, _| {}).expect("initial load");
        assert_eq!(assets.language.tr("menu.multiplayer"), "Multiplayer");
        let before = assets.resources.generation();
        let base_packs = assets.resources.pack_names().len();

        settings.resource_packs.push("Loud".to_owned());
        let outcome = assets
            .reload_packs(&config, &mut settings, &mut |_, _| {})
            .expect("reload");
        assert!(!outcome.reverted);
        assert_ne!(outcome.generation, before, "the generation must advance");
        assert_eq!(outcome.packs, base_packs + 1);
        assert_eq!(
            assets.language.tr("menu.multiplayer"),
            "MULTIPLAYER!!!",
            "the pack must be live without a restart"
        );
        // This assertion is the inverse of what it was before this run: the reload re-bakes and
        // re-installs, so the flag comes back true and the models really are the live ones.
        assert!(
            assets.models_installed && assets.ready,
            "a reload must leave the globally-installed models describing the new stack"
        );
        assert!(
            outcome.model_generation >= 2,
            "and the model registry generation must have advanced: {}",
            outcome.model_generation
        );
        // Keys the pack does not override still come from underneath.
        assert_eq!(assets.language.tr("menu.options"), "Options...");

        settings.resource_packs.clear();
        assets
            .reload_packs(&config, &mut settings, &mut |_, _| {})
            .expect("reload back");
        assert_eq!(assets.language.tr("menu.multiplayer"), "Multiplayer");
    }

    #[test]
    fn an_activated_server_pack_reaches_the_gpu_less_reload_stack() {
        const EMPTY_ZIP: &[u8] = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        let Some(mut config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let dir = tempfile::tempdir().expect("tempdir");
        config.game_dir = dir.path().to_path_buf();
        config.resource_pack_dir = dir.path().join("resourcepacks");
        let mut settings = GameSettings::default();
        let mut assets =
            GameAssets::load(&config, &settings, &mut |_, _| {}).expect("initial load");
        let base_packs = assets.resources.pack_names().len();
        let hash = sha1_hex(EMPTY_ZIP);
        let cache_path = server_resource_pack_cache_path(&config, &hash);
        std::fs::create_dir_all(cache_path.parent().expect("cache parent")).expect("cache dir");
        std::fs::write(&cache_path, EMPTY_ZIP).expect("server pack");
        activate_server_resource_pack(&config, &hash).expect("activate server pack");

        let before = assets.resources.generation();
        let outcome = assets
            .reload_packs(&config, &mut settings, &mut |_, _| {})
            .expect("server pack reload");
        assert!(!outcome.reverted);
        assert_ne!(outcome.generation, before);
        assert_eq!(outcome.packs, base_packs + 1);
        assert_eq!(assets.resources.pack_names().len(), base_packs + 1);
        assert!(
            assets
                .resources
                .pack_names()
                .last()
                .is_some_and(|name| name.contains(&hash)),
            "the activated cache pack must be the highest-precedence stack entry: {:?}",
            assets.resources.pack_names()
        );
        assert!(assets.models_installed && assets.ready);
    }

    /// java: `refreshResources`'s `catch (RuntimeException)` arm — a pack that breaks the font
    /// must not brick the client; the selection is dropped and the defaults come back.
    #[test]
    fn a_pack_that_breaks_the_font_is_reverted() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("Broken");
        let font_dir = pack.join("assets").join("minecraft").join("font");
        std::fs::create_dir_all(&font_dir).expect("mkdir");
        std::fs::write(
            pack.join("pack.mcmeta"),
            br#"{"pack":{"pack_format":1,"description":"x"}}"#,
        )
        .expect("mcmeta");
        // `glyph_sizes.bin` must be exactly 65536 bytes; a short one is a UiError::GlyphSizes.
        std::fs::write(font_dir.join("glyph_sizes.bin"), [0u8; 16]).expect("write");

        let mut config = config;
        config.resource_pack_dir = dir.path().to_path_buf();
        let mut settings = GameSettings::default();
        let mut assets =
            GameAssets::load(&config, &settings, &mut |_, _| {}).expect("initial load");

        settings.resource_packs.push("Broken".to_owned());
        let outcome = assets
            .reload_packs(&config, &mut settings, &mut |_, _| {})
            .expect("a broken pack must be recovered from, not propagated");
        assert!(outcome.reverted);
        assert!(
            settings.resource_packs.is_empty(),
            "the selection must be cleared, as vanilla clears it"
        );
        assert_eq!(
            assets.font.char_widths().len(),
            256,
            "the font must work again"
        );
    }

    /// java: the same recovery arm covers the stitch. A pack whose block
    /// texture is larger than the device's atlas (`StitcherException`) used to fail the reload
    /// half way and stay selected (and saved), so every later start failed in the same stitch.
    #[test]
    fn a_pack_whose_texture_cannot_be_stitched_is_reverted() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        const ATLAS: u32 = 1024;
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("Huge");
        let blocks = pack.join("assets/minecraft/textures/blocks");
        std::fs::create_dir_all(&blocks).expect("mkdir");
        std::fs::write(
            pack.join("pack.mcmeta"),
            br#"{"pack":{"pack_format":1,"description":"x"}}"#,
        )
        .expect("mcmeta");
        // Square (not an animation strip), and twice the side the atlas may have.
        let side = ATLAS * 2;
        let stone = image::RgbaImage::from_pixel(side, side, image::Rgba([90, 90, 90, 255]));
        let mut png = std::io::Cursor::new(Vec::new());
        stone
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode");
        std::fs::write(blocks.join("stone.png"), png.into_inner()).expect("write");

        let mut config = config;
        config.resource_pack_dir = dir.path().to_path_buf();
        let mut settings = GameSettings::default();
        let mut assets =
            GameAssets::load(&config, &settings, &mut |_, _| {}).expect("initial load");
        // The control: the default stack stitches within this atlas.
        let control = assets
            .reload_packs_with_atlas_dimension(&config, &mut settings, ATLAS, &mut |_, _| {})
            .expect("the defaults fit");
        assert!(!control.reverted);

        settings.resource_packs.push("Huge".to_owned());
        let outcome = assets
            .reload_packs_with_atlas_dimension(&config, &mut settings, ATLAS, &mut |_, _| {})
            .expect("a pack that cannot be stitched must be recovered from, not propagated");
        assert!(outcome.reverted);
        assert!(
            settings.resource_packs.is_empty(),
            "the selection must be cleared, as vanilla clears it"
        );
        assert_eq!(
            outcome.packs, control.packs,
            "the defaults are the stack again"
        );
        assert!(assets.models_installed && assets.ready);
        assert_eq!(assets.font.char_widths().len(), 256);
    }

    // ---- S23: the on-disk bake cache -------------------------------------------------------

    /// The key must name the packs that **opened**, not the ones that were asked for. A jar that
    /// failed to open is skipped from the stack, and a key that claimed it was there would be a
    /// key for a stack that never existed.
    #[test]
    fn the_cache_key_names_exactly_the_packs_that_opened() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let settings = GameSettings::default();
        let (packs, sources) = build_pack_stack_with_sources(&config, &settings);
        assert_eq!(packs.len(), sources.len());
        assert_eq!(packs.len(), 2, "object store + version jar");
        assert!(sources[0].ends_with("1.8.json"), "{:?}", sources[0]);
        assert!(sources[1].ends_with("1.8.9.jar"), "{:?}", sources[1]);

        let key = bake_cache_inputs(&config, &settings)
            .expect("the real stack must be stampable")
            .key();
        assert!(key.contains("1.8.9.jar"), "{key}");
        assert!(key.contains("1.8.json"), "{key}");
        assert!(
            key.contains(&format!("states={}", mc_data::STATE_COUNT)),
            "{key}"
        );

        // A selected pack that does not exist is skipped from the stack, so it must not appear
        // in the key either — the key describes what was loaded.
        let with_missing = GameSettings {
            resource_packs: vec!["NotThere.zip".to_owned()],
            ..GameSettings::default()
        };
        let (packs, sources) = build_pack_stack_with_sources(&config, &with_missing);
        assert_eq!(packs.len(), 2);
        assert_eq!(sources.len(), 2);
        assert_eq!(
            bake_cache_inputs(&config, &with_missing).map(|i| i.key()),
            Some(key.clone()),
            "a pack that could not be opened must not change the key"
        );
    }

    /// The end-to-end claim: a second start against the same packs must produce **exactly** the
    /// same block models, item models and atlas as a first one, and must do it from the cache.
    ///
    /// This is the correctness bar for the whole task. Both halves are needed: without the
    /// equality a cache hit proves nothing, and without asserting the *hit* the equality would be
    /// satisfied by a cache that never loaded and re-baked instead.
    #[test]
    fn a_second_start_restores_the_same_bake_from_disk() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            ..config
        };
        let settings = GameSettings::default();
        let inputs = bake_cache_inputs(&config, &settings).expect("stampable");
        let path = config.bake_cache_path();
        assert!(!path.exists(), "the scratch game dir starts empty");

        // Cold: a full bake that writes the cache.
        let cold_started = std::time::Instant::now();
        let cold = GameAssets::load(&config, &settings, &mut |_, _| {}).expect("cold load");
        let cold_ms = cold_started.elapsed().as_secs_f64() * 1000.0;
        assert!(path.is_file(), "a clean bake must have written the cache");
        assert!(cold.bake.is_clean());
        assert_eq!(
            cold.bake.model_states_considered,
            model_states_considered(),
            "the recomputed considered-count must match the real bake's"
        );

        // Warm: the same packs, so the cache must be used.
        assert!(
            matches!(mc_assets::cache::load(&path, &inputs), CacheRead::Hit(_)),
            "the file just written must be a hit for the same stack"
        );
        let warm_started = std::time::Instant::now();
        let warm = GameAssets::load(&config, &settings, &mut |_, _| {}).expect("warm load");
        let warm_ms = warm_started.elapsed().as_secs_f64() * 1000.0;

        // The results must be identical, not merely both present.
        assert_eq!(warm.atlas.w, cold.atlas.w);
        assert_eq!(warm.atlas.h, cold.atlas.h);
        assert_eq!(
            warm.atlas.texture, cold.atlas.texture,
            "the stitched atlas must be byte-identical across a cache round trip"
        );
        assert_eq!(
            warm.bake.model_states_considered,
            cold.bake.model_states_considered
        );
        assert!(warm.models_installed && warm.ready);
        assert!(
            !mc_assets::model_for(mc_data::BlockState { id: 1 << 4 }).is_empty(),
            "stone must have geometry"
        );
        println!("bake cache: cold load {cold_ms:.1} ms, warm load {warm_ms:.1} ms");
    }

    /// Adding a resource pack must invalidate the cache. The control is the unchanged reload
    /// afterwards, which must hit — without it, a cache that never hits would pass.
    #[test]
    fn changing_the_pack_stack_invalidates_the_cached_bake() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let packs_dir = dir.path().join("resourcepacks");
        let pack = packs_dir.join("Loud");
        let lang_dir = pack.join("assets").join("minecraft").join("lang");
        std::fs::create_dir_all(&lang_dir).expect("mkdir");
        std::fs::write(
            pack.join("pack.mcmeta"),
            br#"{"pack":{"pack_format":1,"description":"test"}}"#,
        )
        .expect("mcmeta");
        std::fs::write(lang_dir.join("en_US.lang"), "menu.multiplayer=LOUD\n").expect("lang");

        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            resource_pack_dir: packs_dir,
            ..config
        };
        let base = GameSettings::default();
        let mut with_pack = GameSettings::default();
        with_pack.resource_packs.push("Loud".to_owned());

        // Write a cache for the base stack.
        GameAssets::load(&config, &base, &mut |_, _| {}).expect("cold load");
        let path = config.bake_cache_path();
        assert!(path.is_file());

        let base_inputs = bake_cache_inputs(&config, &base).expect("stampable");
        let pack_inputs = bake_cache_inputs(&config, &with_pack).expect("stampable");
        assert_ne!(
            base_inputs.key(),
            pack_inputs.key(),
            "a pack changes the key"
        );
        assert!(
            matches!(
                mc_assets::cache::load(&path, &pack_inputs),
                CacheRead::Miss(mc_assets::cache::CacheMiss::Inputs { .. })
            ),
            "the base stack's cache must not answer for a stack with a pack on top"
        );
        // Control: the base stack still hits, so the miss above is the key check working.
        assert!(matches!(
            mc_assets::cache::load(&path, &base_inputs),
            CacheRead::Hit(_)
        ));
    }

    /// `MC_RS_NO_BAKE_CACHE` must genuinely turn the whole thing off — no read, no write.
    ///
    /// Env vars are process-global, so this is deliberately a *pure* check of the slot rather
    /// than a load: setting the variable around a `GameAssets::load` would race every other test
    /// in this binary.
    #[test]
    fn the_cache_can_be_turned_off() {
        // The default in this process (nothing sets it) is on.
        assert!(
            bake_cache_enabled(),
            "no test may leave {NO_BAKE_CACHE_ENV} set"
        );
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        assert!(bake_cache_slot(&config, &GameSettings::default(), MAX_ATLAS_DIMENSION).is_some());

        // An unstampable stack also produces no slot, which is the other way the cache switches
        // itself off.
        let broken = ClientConfig {
            version_jar: Some(PathBuf::from("/definitely/not/a/jar/anywhere.jar")),
            assets_dir: PathBuf::from("/definitely/not/here"),
            ..ClientConfig::default()
        };
        assert!(
            bake_cache_inputs(&broken, &GameSettings::default())
                .expect("an empty stack is trivially stampable")
                .key()
                .matches('|')
                .count()
                == 2,
            "a stack with nothing in it must key as a stack with nothing in it"
        );
    }

    #[test]
    fn the_gpu_atlas_cap_is_part_of_the_bake_cache_question() {
        let broken = ClientConfig {
            version_jar: Some(PathBuf::from("/definitely/not/a/jar/anywhere.jar")),
            assets_dir: PathBuf::from("/definitely/not/here"),
            ..ClientConfig::default()
        };
        let settings = GameSettings::default();
        let small = bake_cache_inputs_for_atlas(&broken, &settings, 1_024)
            .expect("an empty stack is stampable");
        let large = bake_cache_inputs_for_atlas(&broken, &settings, 16_384)
            .expect("an empty stack is stampable");
        assert_ne!(small.key(), large.key());
        assert!(small.key().contains("atlas<=1024"));
        assert!(large.key().contains("atlas<=16384"));
    }

    /// A selected pack that is not on disk is skipped with a warning, not a crash.
    #[test]
    fn a_missing_pack_is_skipped() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let settings = GameSettings {
            resource_packs: vec!["NotThere.zip".to_owned()],
            ..GameSettings::default()
        };
        let packs = build_pack_stack(&config, &settings);
        assert_eq!(packs.len(), 2, "only the two defaults survive");
    }

    #[test]
    fn open_user_pack_rejects_a_path_that_is_neither_zip_nor_directory() {
        match open_user_pack(Path::new("/no/such/pack")) {
            Ok(_) => panic!("a missing path must not open"),
            Err(error) => assert!(error.to_string().contains("candidate"), "{error}"),
        }
    }

    /// Java's `resourcePackFilter` is intentionally much narrower than "anything that can be
    /// opened as a pack": folders require a real root `pack.mcmeta`, while only lower-case
    /// `.zip` file names pass the file branch.  The controls below fail under both former Rust
    /// shortcuts (every folder and case-insensitive extensions).
    #[test]
    fn user_pack_candidate_filter_matches_vanillas_folder_and_zip_branches() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bare_folder = dir.path().join("bare");
        std::fs::create_dir(&bare_folder).expect("bare folder");
        assert!(open_user_pack(&bare_folder).is_err());

        let metadata_directory = dir.path().join("metadata-directory");
        std::fs::create_dir(&metadata_directory).expect("folder");
        std::fs::create_dir(metadata_directory.join("pack.mcmeta")).expect("metadata dir");
        assert!(open_user_pack(&metadata_directory).is_err());

        let valid_folder = dir.path().join("valid-folder");
        std::fs::create_dir(&valid_folder).expect("folder");
        std::fs::write(
            valid_folder.join("pack.mcmeta"),
            br#"{"pack":{"pack_format":1,"description":"ok"}}"#,
        )
        .expect("metadata");
        assert!(open_user_pack(&valid_folder).is_ok());

        let uppercase_zip = dir.path().join("looks-like-a-pack.ZIP");
        std::fs::write(&uppercase_zip, []).expect("uppercase zip placeholder");
        assert!(open_user_pack(&uppercase_zip).is_err());

        // A file named `.zip` is still a *candidate* to Java's FileFilter. The deliberately
        // minimal archive is then rejected while opening, which is the later
        // `updateResourcePack` phase rather than the filename filter.
        let empty_zip = dir.path().join("empty.zip");
        std::fs::write(&empty_zip, b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")
            .expect("empty zip");
        assert!(is_user_resource_pack_candidate(&empty_zip));
        assert!(load_user_resource_pack_ui(&empty_zip).is_err());
    }

    #[test]
    fn malformed_or_missing_user_metadata_is_omitted_but_no_pack_section_keeps_the_red_row() {
        let dir = tempfile::tempdir().expect("tempdir");
        let malformed = dir.path().join("malformed");
        std::fs::create_dir(&malformed).expect("folder");
        std::fs::write(malformed.join("pack.mcmeta"), b"not json").expect("metadata");
        assert!(
            load_user_resource_pack_ui(&malformed).is_err(),
            "updateRepositoryEntriesAll catches this parser failure and omits the entry"
        );

        let no_section = dir.path().join("no-section");
        std::fs::create_dir(&no_section).expect("folder");
        std::fs::write(no_section.join("pack.mcmeta"), br#"{"other":1}"#).expect("metadata");
        let data = load_user_resource_pack_ui(&no_section).expect("valid root without pack");
        assert_eq!(data.pack_format, None);
        assert_eq!(
            data.description,
            "\u{a7}cInvalid pack.mcmeta (or missing 'pack' section)"
        );
    }

    /// A folder pack's `pack.mcmeta` is read up to the same 1 MiB cap as
    /// a zip pack's (the pack list reads every pack in the folder, the start-up eviction every
    /// selected one). One past the cap is omitted like unreadable metadata; one at the cap with
    /// the same content still lists.
    #[test]
    fn a_folder_packs_metadata_is_read_up_to_the_mcmeta_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cap = usize::try_from(mc_assets::MAX_MCMETA_BYTES).expect("cap fits");
        let metadata = |len: usize| {
            let head = r#"{"pack":{"pack_format":1,"description":"big"},"pad":""#;
            let mut text = head.to_owned();
            text.push_str(&"a".repeat(len - head.len() - 2));
            text.push_str(r#""}"#);
            assert_eq!(text.len(), len);
            text
        };
        let at_cap = dir.path().join("at-cap");
        std::fs::create_dir(&at_cap).expect("folder");
        std::fs::write(at_cap.join("pack.mcmeta"), metadata(cap)).expect("metadata");
        let data = load_user_resource_pack_ui(&at_cap).expect("metadata at the cap lists");
        assert_eq!(data.pack_format, Some(1));

        let past_cap = dir.path().join("past-cap");
        std::fs::create_dir(&past_cap).expect("folder");
        std::fs::write(past_cap.join("pack.mcmeta"), metadata(cap + 1)).expect("metadata");
        assert!(
            load_user_resource_pack_ui(&past_cap).is_err(),
            "metadata past the cap is not read whole"
        );
    }

    #[test]
    fn selection_evicts_only_unconfirmed_incompatible_entries_and_never_stacks_them() {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, format) in [("Compatible", 1), ("Legacy", 2), ("Confirmed", 3)] {
            let pack = dir.path().join(name);
            std::fs::create_dir(&pack).expect("folder");
            std::fs::write(
                pack.join("pack.mcmeta"),
                format!(r#"{{"pack":{{"pack_format":{format},"description":"{name}"}}}}"#),
            )
            .expect("metadata");
        }
        let malformed = dir.path().join("Malformed");
        std::fs::create_dir(&malformed).expect("folder");
        std::fs::write(malformed.join("pack.mcmeta"), b"{").expect("bad metadata");

        let config = ClientConfig {
            resource_pack_dir: dir.path().to_path_buf(),
            assets_dir: dir.path().join("no-default-assets"),
            version_jar: None,
            ..ClientConfig::default()
        };
        let mut settings = GameSettings {
            resource_packs: vec![
                "Compatible".to_owned(),
                "Legacy".to_owned(),
                "Confirmed".to_owned(),
                "Malformed".to_owned(),
                "Missing.zip".to_owned(),
            ],
            incompatible_resource_packs: vec!["Confirmed".to_owned()],
            ..GameSettings::default()
        };

        evict_unconfirmed_incompatible_user_packs(&config, &mut settings);
        assert_eq!(
            settings.resource_packs,
            ["Compatible", "Confirmed", "Malformed", "Missing.zip"],
            "only a usable, unconfirmed non-format-1 entry reaches Java's iterator.remove()"
        );

        let packs = build_pack_stack(&config, &settings);
        let names: Vec<_> = packs.iter().map(|pack| pack.name()).collect();
        assert_eq!(names, ["Compatible", "Confirmed"]);
        assert!(
            !names.contains(&"Legacy"),
            "negative control: an unconfirmed incompatible pack must not be stacked"
        );
    }

    #[test]
    fn folder_pack_ui_data_decodes_rgba_and_falls_back_for_bad_or_oversized_pngs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("Colourful");
        std::fs::create_dir(&pack).expect("pack dir");
        std::fs::write(
            pack.join("pack.mcmeta"),
            br#"{"pack":{"pack_format":1,"description":"Colourful pack"}}"#,
        )
        .expect("metadata");

        let image =
            image::RgbaImage::from_fn(3, 2, |x, y| image::Rgba([x as u8, y as u8, 0xA5, 0xFF]));
        let mut png = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode png");
        std::fs::write(pack.join("pack.png"), png.into_inner()).expect("pack icon");

        let loaded = load_user_resource_pack_ui(&pack).expect("load row");
        assert_eq!(loaded.description, "Colourful pack§r");
        assert_eq!(loaded.pack_format, Some(1));
        let icon = loaded.icon.expect("valid icon");
        assert_eq!((icon.width, icon.height), (3, 2));
        assert_eq!(&icon.rgba[5 * 4..5 * 4 + 4], &[2, 1, 0xA5, 0xFF]);

        std::fs::write(pack.join("pack.png"), b"this is not a png").expect("bad icon");
        assert!(
            load_user_resource_pack_ui(&pack)
                .expect("bad icon still keeps row")
                .icon
                .is_none(),
            "an undecodable icon must select the UI's default-image fallback"
        );

        let too_wide = image::RgbaImage::new(MAX_PACK_ICON_DIMENSION + 1, 1);
        let mut png = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(too_wide)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode oversized-dimension png");
        std::fs::write(pack.join("pack.png"), png.into_inner()).expect("wide icon");
        assert!(
            load_user_resource_pack_ui(&pack)
                .expect("wide icon still keeps row")
                .icon
                .is_none(),
            "the strict 4096-pixel decoder limit must reject width 4097"
        );

        std::fs::remove_file(pack.join("pack.png")).expect("remove icon");
        assert!(load_user_resource_pack_ui(&pack)
            .expect("missing icon still keeps row")
            .icon
            .is_none());

        std::fs::write(pack.join("pack.mcmeta"), b"not json").expect("bad metadata");
        assert!(
            load_user_resource_pack_ui(&pack).is_err(),
            "a parser exception drops the entry before it can reach the UI"
        );
    }

    #[test]
    fn folder_icon_reads_accept_the_byte_cap_and_reject_the_first_byte_past_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pack.png");
        std::fs::write(&path, [0xA5; 16]).expect("at cap");
        assert_eq!(read_capped_file(&path, 16).expect("exact cap").len(), 16);

        std::fs::write(&path, [0xA5; 17]).expect("past cap");
        let error = read_capped_file(&path, 16).expect_err("one byte over must fail");
        assert!(error.to_string().contains("16-byte pack-icon limit"));
    }

    #[test]
    fn server_pack_cache_key_keeps_only_lowercase_sha1_and_sha1_control_bites() {
        assert_eq!(
            server_resource_pack_cache_key("a".repeat(40).as_str()),
            "a".repeat(40)
        );
        assert_eq!(server_resource_pack_cache_key(&"A".repeat(40)), "legacy");
        assert_eq!(server_resource_pack_cache_key(&"a".repeat(41)), "legacy");
        assert_eq!(
            server_resource_pack_cache_key("１２３４５６７８９０"),
            "legacy"
        );
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn server_pack_cache_hit_requires_a_matching_hash_and_readable_archive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            ..ClientConfig::default()
        };
        let bytes = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        let digest = sha1_hex(bytes);
        let path = server_resource_pack_cache_path(&config, &digest);
        std::fs::create_dir_all(path.parent().expect("cache parent")).expect("cache dir");
        std::fs::write(&path, bytes).expect("cache pack");
        assert_eq!(
            cached_server_resource_pack(&config, &digest)
                .expect("cache read")
                .as_deref(),
            Some(path.as_path())
        );
        activate_server_resource_pack(&config, &digest).expect("activate cache hit");
        assert_eq!(
            active_server_resource_pack_path(&config).as_deref(),
            Some(path.as_path())
        );

        // Java's `hash.length() == 40` still hashes an uppercase/malformed request's `legacy`
        // file, and deletes it when the lowercase digest cannot match.  This is the negative
        // control that prevents a malformed hash from silently reusing a different pack.
        let legacy = server_resource_pack_cache_path(&config, &"A".repeat(40));
        std::fs::write(&legacy, bytes).expect("legacy cache pack");
        assert!(cached_server_resource_pack(&config, &"A".repeat(40))
            .expect("malformed hash cache read")
            .is_none());
        assert!(
            !legacy.exists(),
            "a wrong strict-hash legacy cache is deleted"
        );
    }

    #[test]
    fn level_server_pack_resolves_under_saves_and_activates_without_copying() {
        // java: `handleResourcePack`'s `new File(new File(mcDataDir, "saves"), s2)` branch.
        const EMPTY_ZIP: &[u8] = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            ..ClientConfig::default()
        };
        let url = "level://world/resources.zip";
        let path = level_resource_pack_path(&config, url).expect("level URL");
        std::fs::create_dir_all(path.parent().expect("save parent")).expect("save directory");
        std::fs::write(&path, EMPTY_ZIP).expect("local pack");

        assert_eq!(
            activate_level_resource_pack(&config, url).expect("activate local pack"),
            path
        );
        assert_eq!(
            active_server_resource_pack_path(&config).as_deref(),
            Some(path.as_path())
        );
        assert!(
            !server_resource_pack_cache_path(&config, "legacy").exists(),
            "level:// must use the save file directly, not the remote cache"
        );
        assert!(clear_active_server_resource_pack(&config));
        assert!(active_server_resource_pack_path(&config).is_none());

        // The URL branch is Java's literal prefix test; malformed marker-safe children are not
        // allowed to turn into a different active file.
        assert!(level_resource_pack_path(&config, "LEVEL://world/resources.zip").is_none());
        assert!(level_resource_pack_path(&config, "level://world/line\nfeed.zip").is_none());
    }

    #[test]
    fn server_pack_cache_prune_keeps_ten_entries_and_never_deletes_the_active_marker() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            ..ClientConfig::default()
        };
        let cache = server_resource_pack_dir(&config);
        std::fs::create_dir_all(&cache).expect("cache dir");
        for index in 0..12 {
            std::fs::write(cache.join(format!("pack-{index}")), [index as u8])
                .expect("cache entry");
        }
        std::fs::write(cache.join(SERVER_RESOURCE_PACK_ACTIVE_MARKER), b"legacy")
            .expect("active marker");
        prune_server_resource_pack_cache(&config);
        let files = std::fs::read_dir(&cache)
            .expect("cache dir read")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
            .collect::<Vec<_>>();
        assert!(
            files.len() <= SERVER_RESOURCE_PACK_CACHE_LIMIT + 1,
            "ten cache files plus the marker are retained: {files:?}"
        );
        assert!(cache.join(SERVER_RESOURCE_PACK_ACTIVE_MARKER).is_file());
    }

    #[test]
    fn server_pack_download_validates_zip_hash_and_identity_headers() {
        const BODY: &[u8] = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .expect("test runtime");
        runtime.block_on(async {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("listener");
            let address = listener.local_addr().expect("listener address");
            let server = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.expect("request");
                let mut request = Vec::new();
                let mut request_chunk = [0; 512];
                loop {
                    let read = stream
                        .read(&mut request_chunk)
                        .await
                        .expect("request bytes");
                    assert!(read > 0, "request must have headers");
                    request.extend_from_slice(&request_chunk[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&request);
                assert!(request.starts_with("GET /pack.zip?from=test HTTP/1.1\r\n"));
                // HTTP field names are case-insensitive; hyper lowercases them on the wire even
                // though the request builder receives the Java-shaped spellings.
                let request_lower = request.to_ascii_lowercase();
                assert!(request_lower.contains("host: 127.0.0.1:"));
                assert!(request_lower.contains("x-minecraft-username: oomf\r\n"));
                assert!(request_lower.contains("x-minecraft-uuid: 0123456789abcdef\r\n"));
                // Java's own `HttpURLConnection` defaults, and nothing
                // that names this client to the pack's host.
                assert!(
                    request_lower.contains("user-agent: java/1.8.0_51\r\n"),
                    "{request}"
                );
                assert!(
                    request_lower.contains(
                        "accept: text/html, image/gif, image/jpeg, *; q=.2, */*; q=.2\r\n"
                    ),
                    "{request}"
                );
                assert!(!request_lower.contains("mc-rs"), "{request}");

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    BODY.len()
                );
                stream
                    .write_all(response.as_bytes())
                    .await
                    .expect("response headers");
                stream.write_all(BODY).await.expect("response body");
            });

            let dir = tempfile::tempdir().expect("tempdir");
            let config = ClientConfig {
                game_dir: dir.path().to_path_buf(),
                ..ClientConfig::default()
            };
            let hash = "b04f3ee8f5e43fa3b162981b50bb72fe1acabb33";
            let headers = ServerResourcePackHeaders {
                username: Some("oomf".to_owned()),
                uuid: Some("0123456789abcdef".to_owned()),
            };
            let path = download_server_resource_pack(
                &config,
                &format!("http://{address}/pack.zip?from=test"),
                hash,
                &headers,
            )
            .await
            .expect("download");
            server.await.expect("server task");
            assert_eq!(path, server_resource_pack_cache_path(&config, hash));
            assert_eq!(std::fs::read(path).expect("cached body").as_slice(), BODY);
        });
    }

    #[test]
    fn server_pack_download_follows_relative_redirects_and_keeps_request_identity() {
        const BODY: &[u8] = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .expect("test runtime");
        runtime.block_on(async {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("listener");
            let address = listener.local_addr().expect("listener address");
            let server = tokio::spawn(async move {
                for (index, expected_path) in [
                    (0, "/packs/start.zip?old=1"),
                    (1, "/packs/final.zip?new=2"),
                ] {
                    let (mut stream, _) = listener.accept().await.expect("request");
                    let mut request = Vec::new();
                    let mut request_chunk = [0; 512];
                    loop {
                        let read = stream
                            .read(&mut request_chunk)
                            .await
                            .expect("request bytes");
                        assert!(read > 0, "request must have headers");
                        request.extend_from_slice(&request_chunk[..read]);
                        if request.windows(4).any(|window| window == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let request = String::from_utf8_lossy(&request);
                    assert!(
                        request.starts_with(&format!("GET {expected_path} HTTP/1.1\r\n")),
                        "unexpected request: {request}"
                    );
                    let request_lower = request.to_ascii_lowercase();
                    assert!(request_lower.contains("x-minecraft-username: oomf\r\n"));
                    assert!(request_lower.contains("x-minecraft-uuid: 0123456789abcdef\r\n"));

                    if index == 0 {
                        // The base URL is `/packs/start.zip`; this resolves to `/packs/final.zip`.
                        stream
                            .write_all(
                                b"HTTP/1.1 302 Found\r\nLocation: final.zip?new=2\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            )
                            .await
                            .expect("redirect response");
                    } else {
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            BODY.len()
                        );
                        stream
                            .write_all(response.as_bytes())
                            .await
                            .expect("response headers");
                        stream.write_all(BODY).await.expect("response body");
                    }
                }
            });

            let dir = tempfile::tempdir().expect("tempdir");
            let config = ClientConfig {
                game_dir: dir.path().to_path_buf(),
                ..ClientConfig::default()
            };
            let hash = sha1_hex(BODY);
            let headers = ServerResourcePackHeaders {
                username: Some("oomf".to_owned()),
                uuid: Some("0123456789abcdef".to_owned()),
            };
            let path = download_server_resource_pack(
                &config,
                &format!("http://{address}/packs/start.zip?old=1"),
                &hash,
                &headers,
            )
            .await
            .expect("redirected download");
            server.await.expect("server task");
            assert_eq!(path, server_resource_pack_cache_path(&config, &hash));
            assert_eq!(std::fs::read(path).expect("cached body").as_slice(), BODY);
        });
    }

    #[test]
    fn server_pack_http_parser_rejects_bad_hosts_and_decodes_chunked_bodies() {
        let ipv6 = parse_server_resource_pack_url("http://[::1]:25565/pack.zip")
            .expect("bracketed IPv6 URL");
        assert_eq!(ipv6.host_header, "[::1]:25565");
        assert_eq!(ipv6.path, "/pack.zip");
        assert!(parse_server_resource_pack_url("http://::1/pack.zip").is_err());
        assert!(parse_server_resource_pack_url("http://host:not-a-port/pack.zip").is_err());
        assert!(parse_server_resource_pack_url("http://host/pack zip").is_err());
        assert_eq!(
            decode_chunked_body(b"4\r\nPK\x05\x06\r\n0\r\n\r\n").expect("chunked body"),
            b"PK\x05\x06"
        );
        assert_eq!(
            decode_chunked_body(b"4\r\nPK\x05\x06\r\n0\r\nETag: test\r\n\r\n")
                .expect("chunked body with trailers"),
            b"PK\x05\x06"
        );
        assert!(decode_chunked_body(b"4\r\nPK\x05").is_err());
    }

    #[test]
    fn server_pack_socks_proxy_receives_remote_dns_connect_request() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .expect("test runtime");
        runtime.block_on(async {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("SOCKS listener");
            let proxy_address = listener.local_addr().expect("SOCKS address");
            let proxy = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.expect("proxy connection");
                let mut greeting = [0u8; 2];
                stream.read_exact(&mut greeting).await.expect("greeting");
                assert_eq!(greeting[0], 5);
                let mut methods = vec![0u8; usize::from(greeting[1])];
                stream.read_exact(&mut methods).await.expect("methods");
                assert!(methods.contains(&0), "proxy must offer no-auth SOCKS");
                stream.write_all(&[5, 0]).await.expect("method reply");

                let mut request = [0u8; 4];
                stream
                    .read_exact(&mut request)
                    .await
                    .expect("connect header");
                assert_eq!(&request[..3], &[5, 1, 0]);
                assert_eq!(request[3], 3, "socks5h must send the hostname to the proxy");
                let mut length = [0u8; 1];
                stream
                    .read_exact(&mut length)
                    .await
                    .expect("hostname length");
                let mut hostname = vec![0u8; usize::from(length[0])];
                stream.read_exact(&mut hostname).await.expect("hostname");
                assert_eq!(hostname, b"target.invalid");
                let mut port = [0u8; 2];
                stream
                    .read_exact(&mut port)
                    .await
                    .expect("destination port");
                assert_eq!(u16::from_be_bytes(port), 80);
                stream
                    .write_all(&[5, 1, 0, 1, 0, 0, 0, 0, 0, 0])
                    .await
                    .expect("failure reply");
            });

            let dir = tempfile::tempdir().expect("game dir");
            let config = ClientConfig {
                game_dir: dir.path().to_path_buf(),
                ..ClientConfig::default()
            };
            let socks = mc_auth::SocksProxy::new(
                proxy_address.ip().to_string(),
                proxy_address.port(),
                None,
                None,
            )
            .expect("proxy");
            let error = download_server_resource_pack_with_proxy(
                &config,
                "http://target.invalid/pack.zip",
                "legacy",
                &ServerResourcePackHeaders::default(),
                Some(&socks),
            )
            .await
            .expect_err("test proxy deliberately rejects the destination");
            proxy.await.expect("proxy task");
            assert!(error.to_string().contains("I/O") || error.to_string().contains("connection"));
            assert!(!server_resource_pack_dir(&config).exists());
        });
    }

    #[test]
    fn server_pack_redirect_resolution_validates_https_and_reference_forms() {
        let base = parse_server_resource_pack_url("http://example.invalid:8080/dir/pack.zip?old=1")
            .expect("base URL");
        for (location, expected) in [
            ("https://cdn.invalid/new.zip", "https://cdn.invalid/new.zip"),
            ("//cdn.invalid/new.zip", "http://cdn.invalid/new.zip"),
            ("/root.zip#ignored", "http://example.invalid:8080/root.zip"),
            ("?new=2", "http://example.invalid:8080/dir/pack.zip?new=2"),
            ("../other.zip", "http://example.invalid:8080/other.zip"),
            ("", "http://example.invalid:8080/dir/pack.zip?old=1"),
        ] {
            assert_eq!(
                resolve_server_resource_pack_redirect(&base, location).expect("redirect"),
                expected,
                "Location {location:?}"
            );
        }
        assert!(resolve_server_resource_pack_redirect(&base, "javascript:alert(1)").is_err());
        assert!(resolve_server_resource_pack_redirect(&base, "http://bad host/pack.zip").is_err());

        let https = parse_server_resource_pack_url("HTTPS://example.invalid/pack.zip")
            .expect("valid HTTPS URL shape");
        assert_eq!(https.scheme, "https");
    }

    #[test]
    fn server_pack_https_uses_tls_instead_of_plaintext_and_does_not_cache_on_failure() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .expect("test runtime");
        runtime.block_on(async {
            let dir = tempfile::tempdir().expect("tempdir");
            let config = ClientConfig {
                game_dir: dir.path().to_path_buf(),
                ..ClientConfig::default()
            };
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("TLS listener");
            let address = listener.local_addr().expect("TLS listener address");
            let server = tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let (mut stream, _) = listener.accept().await.expect("TLS connection");
                let mut first = [0u8; 3];
                stream.read_exact(&mut first).await.expect("TLS record");
                assert_eq!(first[0], 0x16, "HTTPS must begin a TLS handshake");
                assert_eq!(first[1], 0x03, "HTTPS must use a TLS record version");
            });
            let error = download_server_resource_pack(
                &config,
                &format!("https://{address}/pack.zip"),
                "legacy",
                &ServerResourcePackHeaders::default(),
            )
            .await
            .expect_err("the plain loopback listener is not a TLS server");
            server.await.expect("TLS probe");
            assert!(!error.to_string().is_empty());
            assert!(
                !server_resource_pack_dir(&config).exists(),
                "unsupported HTTPS must not create a cache directory"
            );
        });
    }

    #[test]
    fn default_pack_ui_joins_object_metadata_with_the_exact_missing_texture_fallback() {
        let dir = tempfile::tempdir().expect("tempdir");
        let assets = dir.path().join("assets");
        let indexes = assets.join("indexes");
        let hash = "aa00000000000000000000000000000000000000";
        let object = assets.join("objects").join("aa").join(hash);
        std::fs::create_dir_all(&indexes).expect("indexes");
        std::fs::create_dir_all(object.parent().expect("object parent")).expect("objects");
        let metadata = br#"{"pack":{"pack_format":1,"description":"Built in"}}"#;
        std::fs::write(&object, metadata).expect("metadata object");
        std::fs::write(
            indexes.join("test.json"),
            format!(
                r#"{{"objects":{{"pack.mcmeta":{{"hash":"{hash}","size":{}}}}}}}"#,
                metadata.len()
            ),
        )
        .expect("index");
        let config = ClientConfig {
            assets_dir: assets,
            asset_index: "test".to_owned(),
            version_jar: None,
            ..ClientConfig::default()
        };

        let loaded = load_default_resource_pack_ui(&config);
        assert_eq!(loaded.description, "Built in\u{a7}r");
        assert_eq!(loaded.pack_format, Some(1));
        let icon = loaded.icon.expect("default always has an image");
        assert_eq!((icon.width, icon.height), (16, 16));
        assert_eq!(&icon.rgba[..4], &[0xF8, 0x00, 0xF8, 0xFF]);
        assert_eq!(&icon.rgba[8 * 4..8 * 4 + 4], &[0x00, 0x00, 0x00, 0xFF]);
        assert_eq!(
            &icon.rgba[(8 * 16 * 4)..(8 * 16 * 4 + 4)],
            &[0x00, 0x00, 0x00, 0xFF]
        );

        let missing_metadata = load_default_resource_pack_ui(&ClientConfig {
            assets_dir: config.assets_dir.clone(),
            asset_index: "missing".to_owned(),
            version_jar: None,
            ..ClientConfig::default()
        });
        assert_eq!(
            missing_metadata.description,
            "\u{a7}cMissing pack.mcmeta :("
        );
    }

    #[test]
    fn the_real_default_pack_png_is_decoded_from_the_version_jar_root() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let loaded = load_default_resource_pack_ui(&config);
        let icon = loaded.icon.as_ref().expect("default icon");
        assert!(!loaded.description.is_empty());
        assert!(icon.width > 0 && icon.height > 0);
        assert_eq!(
            icon.rgba.len(),
            (icon.width * icon.height * 4) as usize,
            "the root pack.png must decode to a complete RGBA image"
        );
        assert_ne!(
            icon,
            &missing_pack_icon(),
            "the discovered 1.8.9 jar carries a real root pack.png"
        );

        let jar = config.version_jar.as_deref().expect("discovered jar");
        let same_root_icon = decode_pack_icon(&read_zip_pack_icon(jar).expect("zip root pack.png"))
            .expect("zip root pack.png must decode");
        assert_eq!(
            &same_root_icon, icon,
            "user zip packs and the default jar must share the bounded root-entry decoder"
        );
    }

    /// The screen glue: a new message forces a redraw, a repeated one respects the throttle.
    #[test]
    fn progress_reports_drive_the_loading_screen() {
        let mut screen = LoadingScreen::new();
        assert!(report_to_screen(&mut screen, "Loading language", 10, 1_000));
        assert_eq!(screen.message, "Loading language");
        assert_eq!(screen.progress, 10);
        assert!(!report_to_screen(
            &mut screen,
            "Loading language",
            12,
            1_010
        ));
        assert!(
            report_to_screen(&mut screen, "Baking block models", 20, 1_020),
            "a stage change must redraw immediately"
        );
    }

    /// The language stack must be `en_US` underneath the selected language, so an untranslated
    /// key falls back per key rather than showing the raw key.
    #[test]
    fn a_non_english_language_keeps_english_underneath() {
        let Some(config) = config() else {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        };
        let settings = GameSettings {
            language: "de_DE".to_owned(),
            ..GameSettings::default()
        };
        let mut resources = ResourceManager::new();
        for pack in build_pack_stack(&config, &settings) {
            resources.push_pack(pack);
        }
        let de = load_language(&resources, &settings);
        assert_eq!(de.code(), "de_DE");
        assert_ne!(
            de.tr("menu.multiplayer"),
            "menu.multiplayer",
            "a real translation must be present"
        );
        assert!(
            resources.contains(&ResourceLocation::minecraft("lang/de_DE.lang")),
            "the German lang file must exist in the stack this test is about"
        );
    }

    #[test]
    fn an_unknown_language_uses_the_advertised_english_descriptor() {
        let temp = tempfile::tempdir().expect("tempdir");
        let lang_dir = temp.path().join("assets/minecraft/lang");
        std::fs::create_dir_all(&lang_dir).expect("language directory");
        std::fs::write(
            temp.path().join("pack.mcmeta"),
            br#"{"pack":{"pack_format":1,"description":"test"},"language":{"en_US":{"region":"United States","name":"English","bidirectional":true}}}"#,
        )
        .expect("metadata");
        std::fs::write(lang_dir.join("en_US.lang"), "probe=English\n").expect("english lang");
        std::fs::write(lang_dir.join("unknown.lang"), "probe=Unknown\n").expect("unknown lang");

        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(
            FolderResourcePack::open(temp.path()).expect("open described pack"),
        ));
        let settings = GameSettings {
            language: "unknown".to_owned(),
            ..GameSettings::default()
        };

        let language = load_language(&resources, &settings);
        assert_eq!(language.code(), mc_assets::DEFAULT_LANGUAGE);
        // Java still loads the raw selected file; only the current descriptor falls back.
        assert_eq!(language.tr("probe"), "Unknown");
        let descriptor = resources
            .language_descriptors()
            .into_iter()
            .find(|language| language.code == mc_assets::DEFAULT_LANGUAGE)
            .expect("advertised en_US descriptor");
        assert!(descriptor.bidirectional);
    }
}
