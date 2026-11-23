//! Player skins: fetch, normalise, cache.
//!
//! java: `client/renderer/ImageBufferDownload.java` (the 64x32 to 64x64 conversion and the
//! opaque/transparent fix-ups), `client/resources/DefaultPlayerSkin.java` (the Steve/Alex
//! rule), `client/resources/SkinManager.java` (the async download).
//!
//! # The hazard this module exists to avoid
//!
//! A skin is an HTTP fetch. Doing one inline when a player joins stalls the frame for as long
//! as the round trip takes, and a full server means dozens of joins in a burst. So:
//!
//! * The **game loop never awaits.** [`SkinCache::get`] returns what is resident, queues a
//!   request if not, and returns immediately. Callers draw the default skin meanwhile.
//! * `tokio` is used **for the network only**. The worker owns the runtime and
//!   `reqwest`; it talks to the synchronous cache over `std::sync::mpsc`.
//! * A failed fetch is remembered so it is not retried every frame forever.
//!
//! # Two skin layouts, two arm widths
//!
//! 1.8 introduced the 64x64 skin. The 64x32 legacy layout has no second layer on the limbs and
//! no left arm/leg at all — vanilla builds the missing halves by mirroring the right ones into
//! the lower half of a 64x64 image ([`normalise_skin`]). Independently, a skin is *classic*
//! (4px arms) or *slim* (3px, the "Alex" model); the session server states which, and when it
//! does not, vanilla derives it from the UUID hash ([`SkinModel::default_for_uuid`]).
//!
//! Both are decided from data, never guessed from the pixels — except the size, which is the
//! one thing the image itself tells you.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Mutex, OnceLock};

/// The process-local handoff used by builtin/entity item callers.
///
/// GUI and dropped-item submission live in crates that deliberately cannot borrow the client's
/// [`AppearanceState`].  Keep the handoff beside the cache rather than growing a driver-only
/// parameter: the normal appearance cache publishes completed skins here, and the item adapters
/// can request/consume the same normalized pixels without doing I/O on their render paths.
static SHARED_READY: OnceLock<Mutex<HashMap<String, SkinTexture>>> = OnceLock::new();
static SHARED_REQUESTS: OnceLock<Mutex<HashMap<String, SkinRequest>>> = OnceLock::new();

fn shared_ready() -> &'static Mutex<HashMap<String, SkinTexture>> {
    SHARED_READY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn shared_requests() -> &'static Mutex<HashMap<String, SkinRequest>> {
    SHARED_REQUESTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Publish one normalized skin for item renderers to consume.
///
/// This is intentionally a clone into a process-local map.  `SkinCache` owns its own lifetime
/// and may be dropped/replaced during a client reconnect, while a draw list can outlive the
/// mutable cache borrow for the rest of its frame.
pub fn publish_shared_skin(key: &str, texture: &SkinTexture) {
    if let Ok(mut ready) = shared_ready().lock() {
        ready.insert(key.to_owned(), texture.clone());
    }
}

/// Look up normalized pixels published by the active appearance cache.
#[must_use]
pub fn shared_skin(key: &str) -> Option<SkinTexture> {
    shared_ready().lock().ok()?.get(key).cloned()
}

/// Whether normalized pixels for `key` have reached the process-local item handoff.
#[must_use]
pub fn has_shared_skin(key: &str) -> bool {
    shared_ready()
        .lock()
        .ok()
        .is_some_and(|ready| ready.contains_key(key))
}

/// Register an item-stack skin fetch with the active [`SkinCache`].
///
/// The registration itself never touches the network.  The next [`SkinCache::poll`] drains this
/// queue into its ordinary worker channel, so disk caching, retry suppression, and the existing
/// async worker remain the single implementation of fetching.
pub fn request_shared_skin(key: &str, url: &str, model: SkinModel) {
    if has_shared_skin(key) {
        return;
    }
    let Ok(mut requests) = shared_requests().lock() else {
        return;
    };
    // Same first-writer-wins semantics as `entry(..).or_insert_with`, without allocating the
    // owned key on every frame a still-pending skull asks again.
    if requests.contains_key(key) {
        return;
    }
    requests.insert(
        key.to_owned(),
        SkinRequest {
            key: key.to_owned(),
            url: url.to_owned(),
            model,
            cache_file: None,
        },
    );
}

/// The final path component of a trusted texture URL, restricted to filename-safe ASCII.
///
/// java: `MinecraftProfileTexture.getHash()`, the CDN URL basename that `SkinManager.loadSkin`
/// names its `ResourceLocation("skins/" + hash)` by. The restriction keeps a malformed profile
/// out of the in-memory/disk key space.
#[must_use]
pub fn texture_url_hash(url: &str) -> Option<&str> {
    let path = url.split(['?', '#']).next()?;
    let hash = path.rsplit('/').next()?.trim();
    (!hash.is_empty()
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')))
    .then_some(hash)
}

/// The skin cache identity for a trusted texture URL: one resident skin per texture hash,
/// shared by every profile that names it, as `SkinManager.loadSkin`'s `skins/<hash>`.
#[must_use]
pub fn skin_key_for_texture_url(url: &str) -> Option<String> {
    texture_url_hash(url).map(|hash| format!("mc-render/entity/skin/{hash}"))
}


/// The one host the skin and cape downloaders fetch from: Mojang's texture server.
pub const MOJANG_TEXTURE_HOST: &str = "textures.minecraft.net";

/// The most bytes one skin or cape download may return. Mojang serves 64x64 and 64x32 PNGs of a
/// few kilobytes; the cap stops a broken or hostile response from growing memory and the disk
/// cache without bound.
pub const MAX_TEXTURE_DOWNLOAD_BYTES: usize = 1024 * 1024;

/// The largest width or height a skin or cape PNG may declare. Every texture Mojang serves is at
/// most 64 texels wide. The limit is checked against the PNG header before any pixel buffer is
/// allocated, so a few hundred bytes cannot ask for a surface of hundreds of megabytes.
pub const MAX_TEXTURE_DIMENSION: u32 = 1024;

/// Redirect hops a texture fetch follows, each to a URL [`mojang_texture_url`] accepts.
const MAX_TEXTURE_REDIRECTS: usize = 10;

/// How long a texture fetch may take to connect, and in all. A half-open connection otherwise
/// holds its task and socket forever (the HTTP client has no default); either way the player
/// keeps the default skin, and a real texture takes well under a second.
const TEXTURE_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// The whole fetch's limit (see [`TEXTURE_CONNECT_TIMEOUT`]).
const TEXTURE_FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// The exact URL the skin and cape downloaders may fetch for a profile texture URL, or `None`.
///
/// java: authlib 1.5.21's `YggdrasilMinecraftSessionService.isWhitelistedDomain` accepts any
/// host ending in `.minecraft.net` or `.mojang.com`, on any port. That lets a server make every
/// client that draws a player head send requests to any Mojang service and port, so this is
/// stricter: only `http(s)://textures.minecraft.net/texture/<hash>`, the form of every texture
/// URL Mojang signs, in its canonical spelling. Profiles are still also held to authlib's own
/// check where they are decoded.
///
/// The URL is parsed with the parser the request itself uses (reqwest's WHATWG [`reqwest::Url`])
/// and must serialise back to exactly the input, so the host checked is the host connected to.
/// A hand-written authority split cannot be trusted here: for an `http` URL that parser ends the
/// authority at a backslash, so `http://evil.example\@textures.minecraft.net/texture/a` connects
/// to `evil.example` although its text after the `@` names Mojang's host.
#[must_use]
pub fn mojang_texture_url(url: &str) -> Option<reqwest::Url> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let canonical = parsed.as_str() == url;
    let scheme = matches!(parsed.scheme(), "http" | "https");
    let host = parsed.host_str() == Some(MOJANG_TEXTURE_HOST);
    let plain = parsed.username().is_empty()
        && parsed.password().is_none()
        && parsed.port().is_none()
        && parsed.query().is_none()
        && parsed.fragment().is_none();
    // The hash characters are the disk cache's own file-name rule (`texture_hash`).
    let path = parsed
        .path()
        .strip_prefix("/texture/")
        .is_some_and(cache_file_name_ok);
    (canonical && scheme && host && plain && path).then_some(parsed)
}

/// Whether the skin and cape downloaders may fetch `url` (see [`mojang_texture_url`]).
#[must_use]
pub fn is_mojang_texture_url(url: &str) -> bool {
    mojang_texture_url(url).is_some()
}

/// Texture downloads in flight at once, skins and capes together. A
/// server can name thousands of distinct real skins (a tab list, a wall of heads), and every
/// fetch is its own task with its own connection. The first eight of a join burst start at once;
/// the rest wait for a free slot in request order (the semaphore is fair). Only the request and
/// its body hold a slot: a texture from the disk cache never waits, and decoding runs after the
/// slot is free. Vanilla starts a thread per download, with no bound.
pub const MAX_TEXTURE_FETCHES: usize = 8;

static TEXTURE_FETCH_SLOTS: tokio::sync::Semaphore =
    tokio::sync::Semaphore::const_new(MAX_TEXTURE_FETCHES);

/// Download one texture's bytes from `url` (already checked) while holding one of
/// [`MAX_TEXTURE_FETCHES`] slots.
async fn download_texture(
    client: &reqwest::Client,
    url: reqwest::Url,
) -> Result<Vec<u8>, SkinError> {
    // The semaphore is never closed, so `acquire` cannot fail.
    let _slot = TEXTURE_FETCH_SLOTS
        .acquire()
        .await
        .map_err(|error| SkinError::Network(error.to_string()))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| SkinError::Network(error.to_string()))?
        .error_for_status()
        .map_err(|error| SkinError::Network(error.to_string()))?;
    read_texture_body(response).await
}

/// A cached texture's bytes, read only up to what a download may be
/// ([`MAX_TEXTURE_DOWNLOAD_BYTES`]): a damaged or planted oversized cache file is a miss (the
/// texture is fetched again and the file overwritten), never read whole.
fn read_cached_texture(path: &Path) -> Option<Vec<u8>> {
    use std::io::Read as _;
    let file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(MAX_TEXTURE_DOWNLOAD_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    (bytes.len() <= MAX_TEXTURE_DOWNLOAD_BYTES).then_some(bytes)
}

/// The HTTP client both texture workers use (see [`texture_http_client_builder`]).
fn texture_http_client() -> reqwest::Client {
    // Fails only when the TLS back end cannot start, which `reqwest::Client::new` (what the
    // workers used before) also panics on.
    texture_http_client_builder()
        .build()
        .expect("the texture HTTP client")
}

/// A redirect is followed only to another URL that [`mojang_texture_url`] accepts, so a redirect
/// cannot move a fetch to another host either.
fn texture_http_client_builder() -> reqwest::ClientBuilder {
    let redirects = reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= MAX_TEXTURE_REDIRECTS {
            attempt.error("too many texture redirects")
        } else if is_mojang_texture_url(attempt.url().as_str()) {
            attempt.follow()
        } else {
            attempt.error("texture redirect leaves Mojang's texture host")
        }
    });
    reqwest::Client::builder()
        .redirect(redirects)
        .connect_timeout(TEXTURE_CONNECT_TIMEOUT)
        .timeout(TEXTURE_FETCH_TIMEOUT)
}

/// The error a texture URL off Mojang's texture host gets. It is remembered like any failed
/// fetch, so the player keeps the default skin and the URL is not asked for again.
fn off_host_texture() -> SkinError {
    SkinError::Network(format!(
        "texture URL is not on {MOJANG_TEXTURE_HOST}; not fetched"
    ))
}

/// Read a texture response body, refusing one over [`MAX_TEXTURE_DOWNLOAD_BYTES`] as it streams
/// in, so a missing or lying `Content-Length` cannot get around the cap.
async fn read_texture_body(mut response: reqwest::Response) -> Result<Vec<u8>, SkinError> {
    let too_big = || {
        SkinError::Network(format!(
            "texture is larger than {MAX_TEXTURE_DOWNLOAD_BYTES} bytes"
        ))
    };
    if response
        .content_length()
        .is_some_and(|length| length > MAX_TEXTURE_DOWNLOAD_BYTES as u64)
    {
        return Err(too_big());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| SkinError::Network(error.to_string()))?
    {
        if body.len() + chunk.len() > MAX_TEXTURE_DOWNLOAD_BYTES {
            return Err(too_big());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Run texture decode work so that a panic in it is a failed texture, not the game's crash
/// (`mc_core::panic_scope`: the crash hook ignores contained panics). A receiver thread that
/// panicked would also have ended skin loading for the session.
fn contained_decode<T>(work: impl FnOnce() -> Result<T, SkinError>) -> Result<T, SkinError> {
    mc_core::panic_scope::contain(work)
        .unwrap_or_else(|_| Err(SkinError::Decode("the texture decoder panicked".to_owned())))
}

/// Decode a skin or cape PNG, refusing a declared size over [`MAX_TEXTURE_DIMENSION`] from the
/// header, before the decoder allocates the image.
fn decode_texture_image(bytes: &[u8]) -> Result<image::DynamicImage, SkinError> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| SkinError::Decode(error.to_string()))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_TEXTURE_DIMENSION);
    limits.max_image_height = Some(MAX_TEXTURE_DIMENSION);
    reader.limits(limits);
    reader
        .decode()
        .map_err(|error| SkinError::Decode(error.to_string()))
}


/// java: `SkinManager.skinCacheLoader`, a Guava cache built with `expireAfterAccess(15,
/// SECONDS)` and keyed by `GameProfile`, whose `equals` compares only the id and the name.
/// `TileEntitySkullRenderer.renderSkull` asks it for a profile's textures map every frame a player
/// head draws (placed, held, dropped, worn and inventory icons alike: one `SkinManager`), so the
/// first profile to miss decides the texture for every equal profile until none of them has drawn
/// for 15 s; then the next one to draw is decoded afresh. Previously, mc-rs keyed head
/// skins by UUID alone, forever: a same-UUID profile with a new texture (an `S35` update, a
/// changed menu icon) kept the old skin for the whole session, and two names sharing a UUID shared
/// one skin.
pub const PLAYER_HEAD_PROFILE_TTL: std::time::Duration = std::time::Duration::from_secs(15);

/// Entries kept before expired ones are first swept (Guava cleans up during writes the same
/// way); after a sweep the next one waits until the cache has doubled, so it stays amortised O(1).
const PLAYER_HEAD_SWEEP_AT: usize = 1024;

/// What one player-head profile's textures map resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayerHeadTextures {
    /// The SKIN texture's hash-named cache key ([`skin_key_for_texture_url`]) and its URL;
    /// `None` when the map has no (trusted) SKIN.
    pub skin: Option<(String, String)>,
    /// The profile-declared arm model, or the UUID default.
    pub model: SkinModel,
}

struct PlayerHeadEntry {
    name: Box<str>,
    textures: PlayerHeadTextures,
    last_access: std::time::Instant,
}

/// The `skinCacheLoader` descriptor cache: (profile id, name) to the textures the first profile
/// that missed decoded, expiring [`PLAYER_HEAD_PROFILE_TTL`] after the last access.
#[derive(Default)]
pub struct PlayerHeadProfiles {
    by_id: HashMap<u128, Vec<PlayerHeadEntry>>,
    len: usize,
    /// `len` at which the next sweep runs; `0` means [`PLAYER_HEAD_SWEEP_AT`].
    sweep_at: usize,
}

impl PlayerHeadProfiles {
    /// java: `skinCacheLoader.getUnchecked(profile)`: a fresh entry for the same id and name is
    /// returned and its access time refreshed; a missing or expired one is loaded by `load`
    /// (`MinecraftSessionService.getTextures(profile, false)` for the profile asking now). No
    /// allocation on a hit.
    pub fn get(
        &mut self,
        id: u128,
        name: &str,
        now: std::time::Instant,
        load: impl FnOnce() -> PlayerHeadTextures,
    ) -> &PlayerHeadTextures {
        if self.len >= self.sweep_at.max(PLAYER_HEAD_SWEEP_AT) {
            self.sweep(now);
            self.sweep_at = (self.len * 2).max(PLAYER_HEAD_SWEEP_AT);
        }
        let entries = self.by_id.entry(id).or_default();
        let found = entries.iter().position(|entry| *entry.name == *name);
        let index = match found {
            Some(index)
                if now.saturating_duration_since(entries[index].last_access)
                    < PLAYER_HEAD_PROFILE_TTL =>
            {
                entries[index].last_access = now;
                index
            }
            Some(index) => {
                entries[index] = PlayerHeadEntry {
                    name: name.into(),
                    textures: load(),
                    last_access: now,
                };
                index
            }
            None => {
                entries.push(PlayerHeadEntry {
                    name: name.into(),
                    textures: load(),
                    last_access: now,
                });
                self.len += 1;
                entries.len() - 1
            }
        };
        &entries[index].textures
    }

    /// Drop every entry idle for [`PLAYER_HEAD_PROFILE_TTL`] or longer (Guava's `>=`).
    pub fn sweep(&mut self, now: std::time::Instant) {
        self.by_id.retain(|_, entries| {
            entries.retain(|entry| {
                now.saturating_duration_since(entry.last_access) < PLAYER_HEAD_PROFILE_TTL
            });
            !entries.is_empty()
        });
        self.len = self.by_id.values().map(Vec::len).sum();
    }

    /// Entries held, expired ones included until the next sweep.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no entry is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

thread_local! {
    /// The render thread's one `skinCacheLoader`, shared by every head renderer (the client's
    /// tile, held, dropped and worn heads and the GUI's inventory icons), as vanilla's one
    /// `SkinManager` is.
    static PLAYER_HEADS: std::cell::RefCell<PlayerHeadProfiles> =
        std::cell::RefCell::new(PlayerHeadProfiles::default());
}

/// Resolve a player head's profile through the shared descriptor cache ([`PlayerHeadProfiles`])
/// at the current time, handing the result to `with`. `load` decodes the asking profile on a
/// miss or after expiry. Neither closure may resolve another head.
pub fn player_head_textures<R>(
    id: u128,
    name: &str,
    load: impl FnOnce() -> PlayerHeadTextures,
    with: impl FnOnce(&PlayerHeadTextures) -> R,
) -> R {
    let now = std::time::Instant::now();
    PLAYER_HEADS.with(|cache| with(cache.borrow_mut().get(id, name, now, load)))
}

/// A decode-once memo for base64 `textures` profile properties (SKIN-4 / TILE-2).
///
/// Player-head renderers resolve their `SkullOwner` profile every frame, and decoding the
/// ~1 KiB property (base64, then a `serde_json::Value`, then the authlib host check) costs far
/// more than recognising a value already seen. The decode is a pure function of the property
/// string, so caching it is invisible; callers keep every side effect (skin requests, the
/// fallback model) on their own per-frame path.
///
/// The value is server-chosen, so it is keyed whole in a seeded [`mc_core::FastMap`]: a lookup
/// hashes and compares it once, and a server cannot steer many values into one probe sequence.
/// Values longer than [`Self::MAX_VALUE_BYTES`] (real ones are a few hundred bytes) are decoded
/// on every call instead, so no entry pins a large string. Capped at [`Self::CAPACITY`] values;
/// past that it starts over rather than evicting piecemeal.
#[derive(Debug)]
pub struct TexturesPropertyMemo<T> {
    values: mc_core::FastMap<Box<str>, T>,
}

impl<T> Default for TexturesPropertyMemo<T> {
    fn default() -> Self {
        Self {
            values: mc_core::FastMap::default(),
        }
    }
}

impl<T> TexturesPropertyMemo<T> {
    /// Distinct values kept before the memo starts over.
    pub const CAPACITY: usize = 512;
    /// Longest value memoised, in bytes. Mojang's signed `textures` payloads and custom-head
    /// values are a few hundred bytes; a longer one is decoded per call, as before the memo.
    pub const MAX_VALUE_BYTES: usize = 4096;

    /// An empty memo.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Distinct values currently memoised.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when nothing is memoised.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Run `with` on the decoded form of `value`, decoding it with `decode` only the first time.
    pub fn with<R>(
        &mut self,
        value: &str,
        decode: impl FnOnce(&str) -> T,
        with: impl FnOnce(&T) -> R,
    ) -> R {
        if value.len() > Self::MAX_VALUE_BYTES {
            return with(&decode(value));
        }
        if let Some(decoded) = self.values.get(value) {
            return with(decoded);
        }
        if self.values.len() >= Self::CAPACITY {
            self.values.clear();
        }
        let decoded = decode(value);
        let result = with(&decoded);
        self.values.insert(value.into(), decoded);
        result
    }
}

fn take_shared_requests() -> Vec<SkinRequest> {
    let Ok(mut requests) = shared_requests().lock() else {
        return Vec::new();
    };
    requests.drain().map(|(_, request)| request).collect()
}

/// Which vanilla normalization branch a downloaded skin uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkinLayout {
    /// A 32-pixel-tall source — pre-1.8. Left limbs are mirrored from the right.
    Legacy64x32,
    /// Any other source height — the modern path, without legacy limb mirroring.
    Modern64x64,
}

impl SkinLayout {
    /// Classifies by source height, as `ImageBufferDownload` does.
    ///
    /// java: `ImageBufferDownload.parseUserSkin` branches on `image.getHeight() == 32`.
    #[must_use]
    pub fn classify(_width: u32, height: u32) -> Option<Self> {
        Some(if height == 32 {
            Self::Legacy64x32
        } else {
            Self::Modern64x64
        })
    }
}

/// Arm geometry: which `ModelPlayer` to build.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SkinModel {
    /// 4-pixel arms — the Steve model. java: `"default"`
    #[default]
    Classic,
    /// 3-pixel arms — the Alex model. java: `"slim"`
    Slim,
}

impl SkinModel {
    /// Parses the session server's `metadata.model` property.
    ///
    /// java: `NetworkPlayerInfo.getSkinType` — anything that is not exactly `"slim"` is classic.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        if s == "slim" {
            Self::Slim
        } else {
            Self::Classic
        }
    }

    /// The model a player gets when the profile carries no skin at all.
    ///
    /// java: `DefaultPlayerSkin.isSlimSkin` — `(playerUUID.hashCode() & 1) == 1`, where
    /// `UUID.hashCode` is `(int)(hilo >> 32) ^ (int)hilo` over `msb ^ lsb`. Reproduced exactly:
    /// a different rule gives half of all default skins the wrong arms.
    #[must_use]
    pub fn default_for_uuid(uuid: u128) -> Self {
        if uuid_hash_code(uuid) & 1 == 1 {
            Self::Slim
        } else {
            Self::Classic
        }
    }

    /// The default texture path for a player with no skin.
    ///
    /// java: `DefaultPlayerSkin.getDefaultSkin`
    #[must_use]
    pub fn default_texture(self) -> &'static str {
        match self {
            Self::Slim => "textures/entity/alex.png",
            Self::Classic => "textures/entity/steve.png",
        }
    }
}

/// `java.util.UUID.hashCode`, bit for bit.
#[must_use]
pub fn uuid_hash_code(uuid: u128) -> i32 {
    let msb = (uuid >> 64) as u64;
    let lsb = uuid as u64;
    let hilo = msb ^ lsb;
    ((hilo >> 32) as u32 as i32) ^ (hilo as u32 as i32)
}

/// A decoded, normalised skin: always 64x64 RGBA, whatever arrived over the wire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkinTexture {
    /// 64 * 64 * 4 bytes, RGBA8.
    ///
    /// Shared, not owned: every UI/first-person/item consumer used to clone all 16 KiB per call,
    /// per frame (SKIN-3), and `shared_skin` cloned it again per player head. An `Arc` clone is a
    /// reference-count bump; the pixels are never mutated after normalisation.
    pub rgba: std::sync::Arc<[u8]>,
    /// What the source image was.
    pub layout: SkinLayout,
    /// Classic or slim arms.
    pub model: SkinModel,
}

impl SkinTexture {
    /// Width of every normalised skin.
    pub const WIDTH: u32 = 64;
    /// Height of every normalised skin.
    pub const HEIGHT: u32 = 64;
}

/// A decoded cape image. Unlike a player skin, a cape is bound at its native dimensions:
/// vanilla's `ModelPlayer.bipedCape` owns a 64x32 texture coordinate space and
/// `ThreadDownloadImageData` does not run the skin normaliser for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapeTexture {
    /// RGBA8 pixels in row-major order.
    pub rgba: Vec<u8>,
    /// Source image width in texels.
    pub width: u32,
    /// Source image height in texels.
    pub height: u32,
}

/// Decodes a cape PNG without applying the 64x64 player-skin fix-ups.
///
/// java: `LayerCape` binds the `ResourceLocation` returned by
/// `AbstractClientPlayer.getLocationCape`; it never calls `ImageBufferDownload.parseUserSkin`.
/// Keeping dimensions here is important because the cape model's UVs are authored in a 64x32
/// space while the texture manager itself accepts the decoded image's native size.
pub fn decode_cape(bytes: &[u8]) -> Result<CapeTexture, SkinError> {
    // Bounded decode (header size limit before allocation).
    let image = decode_texture_image(bytes)?;
    let image = image.into_rgba8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(SkinError::BadSize(width, height));
    }
    Ok(CapeTexture {
        rgba: image.into_raw(),
        width,
        height,
    })
}

/// Errors from decoding or normalising a skin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkinError {
    /// The bytes were not a decodable image.
    Decode(String),
    /// The decoded pixel payload does not match its declared dimensions.
    BadSize(u32, u32),
    /// The HTTP fetch failed.
    Network(String),
    /// A downloaded skin could not be persisted to the configured cache.
    Cache(String),
}

impl core::fmt::Display for SkinError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "skin is not a decodable image: {e}"),
            Self::BadSize(w, h) => write!(f, "skin pixel payload does not match {w}x{h}"),
            Self::Network(e) => write!(f, "skin fetch failed: {e}"),
            Self::Cache(e) => write!(f, "skin cache failed: {e}"),
        }
    }
}

impl std::error::Error for SkinError {}

/// Source rectangle to destination rectangle, with the horizontal flip vanilla relies on.
///
/// java: `Graphics.drawImage(img, dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2, obs)`. In every call
/// `ImageBufferDownload` makes, `dx1 > dx2` — the destination is written right-to-left, which is
/// what mirrors the right arm into the left arm's slot.
fn blit_mirrored(px: &mut [u8], dx1: u32, dy1: u32, dx2: u32, dy2: u32, sx1: u32, sy1: u32) {
    let w = dx1 - dx2;
    let h = dy2 - dy1;
    let mut scratch = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let src = (((sy1 + y) * 64 + sx1 + x) * 4) as usize;
            let dst = ((y * w + x) * 4) as usize;
            scratch[dst..dst + 4].copy_from_slice(&px[src..src + 4]);
        }
    }
    for y in 0..h {
        for x in 0..w {
            // Destination x runs from dx2 up to dx1, taking source columns in reverse.
            let src = ((y * w + (w - 1 - x)) * 4) as usize;
            let dst = (((dy1 + y) * 64 + dx2 + x) * 4) as usize;
            px[dst..dst + 4].copy_from_slice(&scratch[src..src + 4]);
        }
    }
}

/// True if any pixel in the rectangle has alpha below 128. java: `hasTransparency`
fn has_transparency(px: &[u8], x1: u32, y1: u32, x2: u32, y2: u32) -> bool {
    for x in x1..x2 {
        for y in y1..y2 {
            if px[((y * 64 + x) * 4 + 3) as usize] < 128 {
                return true;
            }
        }
    }
    false
}

/// java: `setAreaOpaque` — force alpha to 255.
fn set_area_opaque(px: &mut [u8], x1: u32, y1: u32, x2: u32, y2: u32) {
    for x in x1..x2 {
        for y in y1..y2 {
            px[((y * 64 + x) * 4 + 3) as usize] = 255;
        }
    }
}

/// java: `setAreaTransparent` — clear alpha, but only if the whole area was already opaque.
fn set_area_transparent(px: &mut [u8], x1: u32, y1: u32, x2: u32, y2: u32) {
    if has_transparency(px, x1, y1, x2, y2) {
        return;
    }
    for x in x1..x2 {
        for y in y1..y2 {
            px[((y * 64 + x) * 4 + 3) as usize] = 0;
        }
    }
}

/// Turns any legal skin image into the canonical 64x64 RGBA form.
///
/// java: `ImageBufferDownload.parseUserSkin`, in the same order — draw the source into the top
/// of a 64x64 canvas, mirror the legacy limbs down into the 1.8 slots if the source was 32 tall,
/// then run the eight `setAreaOpaque`/`setAreaTransparent` fix-ups. Those fix-ups are what stop
/// a skin drawn without an alpha channel from rendering as a solid block around the head.
///
/// # Errors
/// [`SkinError::BadSize`] if the pixel payload does not match the declared dimensions.
pub fn normalise_skin(
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<(Vec<u8>, SkinLayout), SkinError> {
    let layout = SkinLayout::classify(width, height).expect("every height has a vanilla branch");
    let expected_len = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(SkinError::BadSize(width, height))?;
    if rgba.len() != expected_len {
        return Err(SkinError::BadSize(width, height));
    }

    let mut px = vec![0u8; 64 * 64 * 4];
    // java: `graphics.drawImage(image, 0, 0, null)` into a 64x64 TYPE_INT_ARGB canvas — the
    // untouched remainder stays fully transparent (0,0,0,0).
    let copy_width = width.min(64);
    let copy_height = height.min(64);
    for y in 0..copy_height {
        let src = (y * width * 4) as usize;
        let dst = (y * 64 * 4) as usize;
        let row_len = (copy_width * 4) as usize;
        px[dst..dst + row_len].copy_from_slice(&rgba[src..src + row_len]);
    }

    if layout == SkinLayout::Legacy64x32 {
        // java: the twelve drawImage calls, verbatim (dx1, dy1, dx2, dy2, sx1, sy1).
        blit_mirrored(&mut px, 24, 48, 20, 52, 4, 16);
        blit_mirrored(&mut px, 28, 48, 24, 52, 8, 16);
        blit_mirrored(&mut px, 20, 52, 16, 64, 8, 20);
        blit_mirrored(&mut px, 24, 52, 20, 64, 4, 20);
        blit_mirrored(&mut px, 28, 52, 24, 64, 0, 20);
        blit_mirrored(&mut px, 32, 52, 28, 64, 12, 20);
        blit_mirrored(&mut px, 40, 48, 36, 52, 44, 16);
        blit_mirrored(&mut px, 44, 48, 40, 52, 48, 16);
        blit_mirrored(&mut px, 36, 52, 32, 64, 48, 20);
        blit_mirrored(&mut px, 40, 52, 36, 64, 44, 20);
        blit_mirrored(&mut px, 44, 52, 40, 64, 40, 20);
        blit_mirrored(&mut px, 48, 52, 44, 64, 52, 20);
    }

    set_area_opaque(&mut px, 0, 0, 32, 16);
    set_area_transparent(&mut px, 32, 0, 64, 32);
    set_area_opaque(&mut px, 0, 16, 64, 32);
    set_area_transparent(&mut px, 0, 32, 16, 48);
    set_area_transparent(&mut px, 16, 32, 40, 48);
    set_area_transparent(&mut px, 40, 32, 56, 48);
    set_area_transparent(&mut px, 0, 48, 16, 64);
    set_area_opaque(&mut px, 16, 48, 48, 64);
    set_area_transparent(&mut px, 48, 48, 64, 64);

    Ok((px, layout))
}

/// Decodes a PNG and normalises it in one step.
///
/// # Errors
/// [`SkinError::Decode`] if the bytes are not an image, [`SkinError::BadSize`] if the
/// dimensions do not match the decoded pixel payload.
pub fn decode_skin(bytes: &[u8], model: SkinModel) -> Result<SkinTexture, SkinError> {
    // Bounded decode (header size limit before allocation).
    let img = decode_texture_image(bytes)?;
    let img = img.to_rgba8();
    let (w, h) = img.dimensions();
    let (rgba, layout) = normalise_skin(img.as_raw(), w, h)?;
    Ok(SkinTexture {
        rgba: rgba.into(),
        layout,
        model,
    })
}

/// A queued download. The worker turns this into a [`SkinResponse`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkinRequest {
    /// Cache key — normally the player's UUID as a string.
    pub key: String,
    /// The texture URL from the session server's profile properties.
    pub url: String,
    /// Which arms the profile says this skin has.
    pub model: SkinModel,
    /// Vanilla's `<assetsDir>/skins/<hash[0..2]>/<hash>` file, when disk caching is enabled.
    pub cache_file: Option<PathBuf>,
}

/// A finished download.
#[derive(Debug)]
pub struct SkinResponse {
    /// The key from the matching [`SkinRequest`].
    pub key: String,
    /// The normalised skin, or why it failed.
    pub result: Result<SkinTexture, SkinError>,
}

/// The worker half of the channel pair. Hand this to [`spawn_http_worker`], or drive it by hand
/// in a test.
pub struct SkinChannel {
    /// Requests coming from the game loop.
    pub requests: Receiver<SkinRequest>,
    /// Where to send finished skins.
    pub responses: Sender<SkinResponse>,
}

/// Non-blocking skin cache for the synchronous game loop.
///
/// java: `SkinManager` — but without the `ThreadDownloadImageData`/`Minecraft.addScheduledTask`
/// hop, since here the game loop polls rather than being called back into.
pub struct SkinCache {
    ready: HashMap<String, SkinTexture>,
    inflight: HashSet<String>,
    failed: HashMap<String, SkinError>,
    disk_cache_root: Option<PathBuf>,
    tx: Sender<SkinRequest>,
    rx: Receiver<SkinResponse>,
    requested: u64,
    delivered: u64,
}

impl SkinCache {
    /// Builds a cache and the channel half a worker needs.
    #[must_use]
    pub fn new() -> (Self, SkinChannel) {
        Self::build(None)
    }

    /// Builds a cache backed by vanilla's on-disk `assets/skins` directory.
    ///
    /// `cache_root` is the `skins` directory itself. Files are sharded by the first two
    /// characters of authlib's texture hash exactly as `SkinManager.loadSkin` does.
    #[must_use]
    pub fn with_disk_cache(cache_root: impl Into<PathBuf>) -> (Self, SkinChannel) {
        Self::build(Some(cache_root.into()))
    }

    fn build(disk_cache_root: Option<PathBuf>) -> (Self, SkinChannel) {
        let (req_tx, req_rx) = std::sync::mpsc::channel();
        let (res_tx, res_rx) = std::sync::mpsc::channel();
        (
            Self {
                ready: HashMap::new(),
                inflight: HashSet::new(),
                failed: HashMap::new(),
                disk_cache_root,
                tx: req_tx,
                rx: res_rx,
                requested: 0,
                delivered: 0,
            },
            SkinChannel {
                requests: req_rx,
                responses: res_tx,
            },
        )
    }

    /// Returns the skin if it is resident, queueing a fetch if it is not.
    ///
    /// **Never blocks.** The first call for a key returns `None` and starts the download; a
    /// later frame gets the real thing once [`SkinCache::poll`] has picked it up. Callers draw
    /// [`SkinModel::default_texture`] in the meantime, which is exactly what vanilla does.
    pub fn get(&mut self, key: &str, url: &str, model: SkinModel) -> Option<&SkinTexture> {
        if self.ready.contains_key(key) {
            return self.ready.get(key);
        }
        // A builtin/entity item may have asked for this profile before the ordinary player path
        // reached the cache.  Reuse the completed handoff instead of starting a duplicate fetch.
        if let Some(texture) = shared_skin(key) {
            self.ready.insert(key.to_owned(), texture);
            return self.ready.get(key);
        }
        if !self.inflight.contains(key) && !self.failed.contains_key(key) {
            self.inflight.insert(key.to_string());
            self.requested += 1;
            // A dead worker is not fatal: the entity keeps its default skin.
            let _ = self.tx.send(SkinRequest {
                key: key.to_string(),
                url: url.to_string(),
                model,
                cache_file: self
                    .disk_cache_root
                    .as_deref()
                    .and_then(|root| skin_cache_path(root, url)),
            });
        }
        None
    }

    /// Drains finished downloads. Call once per frame; returns how many arrived.
    ///
    /// This is the only place the cache learns anything, and it does no I/O of its own.
    pub fn poll(&mut self) -> usize {
        // Item adapters cannot borrow the client's appearance state.  Move their requests into
        // this cache before consuming responses so the existing worker remains the only network
        // boundary.  A caller with no worker still gets a deterministic queued request and can
        // inspect it in tests through the normal channel.
        for request in take_shared_requests() {
            if self.ready.contains_key(&request.key)
                || self.inflight.contains(&request.key)
                || self.failed.contains_key(&request.key)
            {
                continue;
            }
            self.inflight.insert(request.key.clone());
            self.requested += 1;
            let _ = self.tx.send(SkinRequest {
                cache_file: self
                    .disk_cache_root
                    .as_deref()
                    .and_then(|root| skin_cache_path(root, &request.url)),
                ..request
            });
        }
        let mut n = 0;
        while let Ok(res) = self.rx.try_recv() {
            self.inflight.remove(&res.key);
            match res.result {
                Ok(tex) => {
                    publish_shared_skin(&res.key, &tex);
                    self.ready.insert(res.key, tex);
                    self.delivered += 1;
                }
                Err(e) => {
                    // Remember the failure so a broken URL is not retried every frame.
                    self.failed.insert(res.key, e);
                }
            }
            n += 1;
        }
        n
    }

    /// Inserts a skin directly — used for the local player, for resource-pack skins, and by
    /// tests that must not touch the network.
    pub fn insert(&mut self, key: &str, tex: SkinTexture) {
        self.inflight.remove(key);
        self.failed.remove(key);
        publish_shared_skin(key, &tex);
        self.ready.insert(key.to_string(), tex);
    }

    /// True if the skin is resident right now.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.ready.contains_key(key) || has_shared_skin(key)
    }

    /// The resident skin for `key`, queueing nothing: 3D Skins reads the pixels
    /// of a texture the body already draws.
    #[must_use]
    pub fn ready(&self, key: &str) -> Option<&SkinTexture> {
        self.ready.get(key)
    }

    /// True if a fetch for this key is outstanding.
    #[must_use]
    pub fn is_inflight(&self, key: &str) -> bool {
        self.inflight.contains(key)
    }

    /// How many fetches are outstanding (a replay export waits for them
    /// before its first frame, so players start in their own skins).
    #[must_use]
    pub fn inflight_len(&self) -> usize {
        self.inflight.len()
    }

    /// Why a key's fetch failed, if it did.
    #[must_use]
    pub fn failure(&self, key: &str) -> Option<&SkinError> {
        self.failed.get(key)
    }

    /// Lets a failed key be tried again — for a manual "reload skins" action.
    pub fn clear_failure(&mut self, key: &str) {
        self.failed.remove(key);
    }

    /// `(requests started, skins delivered)`, for the F3 overlay.
    #[must_use]
    pub fn stats(&self) -> (u64, u64) {
        (self.requested, self.delivered)
    }
}

/// A queued cape download. Cape requests intentionally use a separate type from
/// [`SkinRequest`]: a cape must not accidentally pass through the skin normaliser and become a
/// 64x64 player texture. The key is supplied by the caller and must include account/profile
/// identity as well as the texture kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapeRequest {
    /// Cache key for this profile's CAPE texture.
    pub key: String,
    /// Trusted CAPE URL extracted from the signed profile property.
    pub url: String,
    /// Optional vanilla-style disk-cache path.
    pub cache_file: Option<PathBuf>,
}

/// A finished cape download.
#[derive(Debug)]
pub struct CapeResponse {
    /// The key from the matching [`CapeRequest`].
    pub key: String,
    /// Native-size cape pixels or the decode/network/cache failure.
    pub result: Result<CapeTexture, SkinError>,
}

/// The worker half of a cape cache channel.
pub struct CapeChannel {
    /// Requests coming from the synchronous game loop.
    pub requests: Receiver<CapeRequest>,
    /// Finished responses returned by the async worker.
    pub responses: Sender<CapeResponse>,
}

/// Non-blocking native-size CAPE cache.
///
/// java: `SkinManager`'s CAPE `ResourceLocation`/`ThreadDownloadImageData` path. It is kept
/// separate from [`SkinCache`] so profile replacement can never make a 64x32 cape alias a skin
/// entry, and because `LayerCape` needs the image's original dimensions.
pub struct CapeCache {
    ready: HashMap<String, CapeTexture>,
    inflight: HashSet<String>,
    failed: HashMap<String, SkinError>,
    disk_cache_root: Option<PathBuf>,
    tx: Sender<CapeRequest>,
    rx: Receiver<CapeResponse>,
    requested: u64,
    delivered: u64,
}

impl CapeCache {
    /// Builds a cape cache and its worker channel.
    #[must_use]
    pub fn new() -> (Self, CapeChannel) {
        Self::build(None)
    }

    /// Builds a cape cache backed by the vanilla `<assetsDir>/skins` directory.
    #[must_use]
    pub fn with_disk_cache(cache_root: impl Into<PathBuf>) -> (Self, CapeChannel) {
        Self::build(Some(cache_root.into()))
    }

    fn build(disk_cache_root: Option<PathBuf>) -> (Self, CapeChannel) {
        let (req_tx, req_rx) = std::sync::mpsc::channel();
        let (res_tx, res_rx) = std::sync::mpsc::channel();
        (
            Self {
                ready: HashMap::new(),
                inflight: HashSet::new(),
                failed: HashMap::new(),
                disk_cache_root,
                tx: req_tx,
                rx: res_rx,
                requested: 0,
                delivered: 0,
            },
            CapeChannel {
                requests: req_rx,
                responses: res_tx,
            },
        )
    }

    /// Returns a resident cape, queueing at most one fetch for `key` when absent.
    pub fn get(&mut self, key: &str, url: &str) -> Option<&CapeTexture> {
        if self.ready.contains_key(key) {
            return self.ready.get(key);
        }
        if !self.inflight.contains(key) && !self.failed.contains_key(key) {
            self.inflight.insert(key.to_owned());
            self.requested += 1;
            let _ = self.tx.send(CapeRequest {
                key: key.to_owned(),
                url: url.to_owned(),
                cache_file: self
                    .disk_cache_root
                    .as_deref()
                    .and_then(|root| skin_cache_path(root, url)),
            });
        }
        None
    }

    /// Drains completed cape downloads without blocking.
    pub fn poll(&mut self) -> usize {
        let mut n = 0;
        while let Ok(response) = self.rx.try_recv() {
            self.inflight.remove(&response.key);
            match response.result {
                Ok(texture) => {
                    self.ready.insert(response.key, texture);
                    self.delivered += 1;
                }
                Err(error) => {
                    self.failed.insert(response.key, error);
                }
            }
            n += 1;
        }
        n
    }

    /// Inserts a decoded cape directly, used by tests and local/resource-pack integrations.
    pub fn insert(&mut self, key: &str, texture: CapeTexture) {
        self.inflight.remove(key);
        self.failed.remove(key);
        self.ready.insert(key.to_owned(), texture);
    }

    /// Whether a cape is resident right now.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.ready.contains_key(key)
    }

    /// Whether a fetch is outstanding for `key`.
    #[must_use]
    pub fn is_inflight(&self, key: &str) -> bool {
        self.inflight.contains(key)
    }

    /// How many fetches are outstanding.
    #[must_use]
    pub fn inflight_len(&self) -> usize {
        self.inflight.len()
    }

    /// Why a key's fetch failed, if known.
    #[must_use]
    pub fn failure(&self, key: &str) -> Option<&SkinError> {
        self.failed.get(key)
    }

    /// Allows a failed key to be requested again.
    pub fn clear_failure(&mut self, key: &str) {
        self.failed.remove(key);
    }

    /// `(requests started, capes delivered)` for diagnostics/tests.
    #[must_use]
    pub fn stats(&self) -> (u64, u64) {
        (self.requested, self.delivered)
    }
}

/// Authlib 1.5.21's `MinecraftProfileTexture.getHash()` is
/// `FilenameUtils.getBaseName(url)`. Texture CDN URLs end in the extensionless content hash;
/// accepting only filename-safe hash characters also guarantees a hostile URL cannot escape the
/// configured cache root.
fn texture_hash(url: &str) -> Option<&str> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let filename = path.rsplit(['/', '\\']).next()?;
    let base = filename
        .rsplit_once('.')
        .map_or(filename, |(without_extension, _)| without_extension);
    cache_file_name_ok(base).then_some(base)
}

/// Whether `hash` may name a file in the skin cache: filename-safe characters only, and
/// never a Windows device name. On Windows 10 a path ending in `CON`
/// (whatever folder it is in) opens the console: the cache read on the worker thread then waits
/// for console input forever, and every later skin and cape queues behind it. A server could do
/// that with one head texture `.../texture/CON`; Mojang's hashes are hex, so none is refused.
fn cache_file_name_ok(hash: &str) -> bool {
    !hash.is_empty()
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        && !mc_assets::is_windows_device_name(hash)
}

fn skin_cache_path(root: &Path, url: &str) -> Option<PathBuf> {
    let hash = texture_hash(url)?;
    let shard = if hash.len() > 2 { &hash[..2] } else { "xx" };
    Some(root.join(shard).join(hash))
}

/// Read and decode a cache hit on the request-receiver thread. Any read/decode failure is a miss:
/// vanilla logs it and falls through to `loadTextureFromServer`.
fn cached_response(req: &SkinRequest) -> Option<SkinResponse> {
    let path = req.cache_file.as_deref()?;
    let bytes = read_cached_texture(path)?;
    match contained_decode(|| decode_skin(&bytes, req.model)) {
        Ok(texture) => Some(SkinResponse {
            key: req.key.clone(),
            result: Ok(texture),
        }),
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "could not load cached skin; downloading it again");
            None
        }
    }
}

/// Persist the exact downloaded image before decoding it, matching
/// `FileUtils.copyInputStreamToFile` followed by `ImageIO.read(cacheFile)`.
fn cache_and_decode_download(
    bytes: &[u8],
    cache_file: Option<&Path>,
    model: SkinModel,
) -> Result<SkinTexture, SkinError> {
    if let Some(path) = cache_file {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| SkinError::Cache(error.to_string()))?;
        }
        std::fs::write(path, bytes).map_err(|error| SkinError::Cache(error.to_string()))?;
    }
    decode_skin(bytes, model)
}

/// Runs the download loop on a `tokio` runtime.
///
/// This is the **only** part of `mc-render` that touches the network, and it never touches the
/// frame loop: it owns the `reqwest` client, and its sole output is a [`SkinResponse`] pushed
/// down a `std::sync::mpsc` channel the game loop drains with [`SkinCache::poll`].
///
/// The blocking `recv` runs on a `spawn_blocking` thread so it cannot starve the runtime's
/// worker threads, and each fetch is its own task, so one slow CDN response does not hold up
/// the rest of a join burst.
pub fn spawn_http_worker(channel: SkinChannel, handle: &tokio::runtime::Handle) {
    let client = texture_http_client();
    let handle2 = handle.clone();
    // A plain OS thread, **not** `Handle::spawn_blocking`.
    //
    // This loop parks in `recv()` and only returns when every `SkinRequest` sender is dropped —
    // i.e. when the `Client` goes away. Tokio does not cancel blocking tasks at shutdown: it
    // *waits* for the ones that have already started. So as a `spawn_blocking` task this loop
    // deadlocked `Runtime::drop` against the very sender whose drop would have released it —
    // `Client` declares `runtime` (`app/driver.rs:542`) before the `AppearanceState` that owns
    // the sender, and fields drop in declaration order, so the runtime always waited first.
    //
    // The symptom was a client that rendered a frame and then never shut down: every
    // `mc-client` test that called `Client::frame` hung forever on teardown *after passing its
    // assertions* (four in the lib suite, plus the integration rig), and the real binary had to
    // be killed after its own clean-shutdown log. An OS thread is what a never-returning loop
    // wants anyway, and the runtime no longer has anything to wait for.
    std::thread::spawn(move || {
        while let Ok(req) = channel.requests.recv() {
            if let Some(response) = cached_response(&req) {
                let _ = channel.responses.send(response);
                continue;
            }
            let client = client.clone();
            let responses = channel.responses.clone();
            let SkinRequest {
                key,
                url,
                model,
                cache_file,
            } = req;
            handle2.spawn(async move {
                let result = fetch_skin(&client, &url, model, cache_file).await;
                let _ = responses.send(SkinResponse { key, result });
            });
        }
    });
}

/// Read and decode a cached cape image on the request-receiver thread.
fn cached_cape_response(request: &CapeRequest) -> Option<CapeResponse> {
    let path = request.cache_file.as_deref()?;
    let bytes = read_cached_texture(path)?;
    match contained_decode(|| decode_cape(&bytes)) {
        Ok(texture) => Some(CapeResponse {
            key: request.key.clone(),
            result: Ok(texture),
        }),
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "could not load cached cape; downloading it again");
            None
        }
    }
}

/// Persist one downloaded cape before decoding it, matching the skin worker's cache semantics.
fn cache_and_decode_cape_download(
    bytes: &[u8],
    cache_file: Option<&Path>,
) -> Result<CapeTexture, SkinError> {
    if let Some(path) = cache_file {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| SkinError::Cache(error.to_string()))?;
        }
        std::fs::write(path, bytes).map_err(|error| SkinError::Cache(error.to_string()))?;
    }
    decode_cape(bytes)
}

/// Runs the native-size cape download loop on a Tokio runtime.
///
/// The receiver parks on a plain OS thread for the same shutdown reason as
/// [`spawn_http_worker`]. Individual requests and PNG decoding remain asynchronous/off-thread,
/// and the game loop only ever drains [`CapeCache::poll`].
pub fn spawn_cape_http_worker(channel: CapeChannel, handle: &tokio::runtime::Handle) {
    let client = texture_http_client();
    let handle = handle.clone();
    std::thread::spawn(move || {
        while let Ok(request) = channel.requests.recv() {
            if let Some(response) = cached_cape_response(&request) {
                let _ = channel.responses.send(response);
                continue;
            }
            let client = client.clone();
            let responses = channel.responses.clone();
            handle.spawn(async move {
                let CapeRequest {
                    key,
                    url,
                    cache_file,
                } = request;
                let result = fetch_cape(&client, &url, cache_file).await;
                let _ = responses.send(CapeResponse { key, result });
            });
        }
    });
}

async fn fetch_cape(
    client: &reqwest::Client,
    url: &str,
    cache_file: Option<PathBuf>,
) -> Result<CapeTexture, SkinError> {
    // Checked here, where a cape leaves the process, so every caller is
    // covered; the request uses the parsed URL that was checked.
    let url = mojang_texture_url(url).ok_or_else(off_host_texture)?;
    // At most `MAX_TEXTURE_FETCHES` downloads at once.
    let bytes = download_texture(client, url).await?;
    tokio::task::spawn_blocking(move || {
        contained_decode(|| cache_and_decode_cape_download(&bytes, cache_file.as_deref()))
    })
    .await
    .map_err(|error| SkinError::Cache(error.to_string()))?
}

async fn fetch_skin(
    client: &reqwest::Client,
    url: &str,
    model: SkinModel,
    cache_file: Option<PathBuf>,
) -> Result<SkinTexture, SkinError> {
    // Checked here, where a skin leaves the process, so every caller
    // (tab list, heads, items, replays) is covered; the request uses the parsed URL that was
    // checked.
    let url = mojang_texture_url(url).ok_or_else(off_host_texture)?;
    // At most `MAX_TEXTURE_FETCHES` downloads at once.
    let bytes = download_texture(client, url).await?;
    // Disk and PNG work stay off Tokio's async worker. Unlike the never-ending request receiver,
    // this blocking task is finite, so runtime shutdown can safely wait for it.
    tokio::task::spawn_blocking(move || {
        contained_decode(|| cache_and_decode_download(&bytes, cache_file.as_deref(), model))
    })
    .await
    .map_err(|error| SkinError::Cache(error.to_string()))?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SKIN-4: the memo decodes each distinct value once, answers repeats from the memo, never
    /// confuses two values, and starts over at its cap. Phase 4b review S2: values are keyed
    /// whole, and an over-long value is decoded every time and never stored.
    #[test]
    fn textures_property_memo_decodes_each_value_once() {
        let mut memo: TexturesPropertyMemo<usize> = TexturesPropertyMemo::new();
        let mut decodes = 0;
        let mut decode = |value: &str| {
            decodes += 1;
            value.len()
        };
        assert_eq!(memo.with("abc", &mut decode, |n| *n), 3);
        assert_eq!(memo.with("abc", &mut decode, |n| *n), 3);
        assert_eq!(memo.with("abcd", &mut decode, |n| *n), 4);
        assert_eq!(decodes, 2);
        assert_eq!(memo.len(), 2);

        // Same length, same first 16 and last 64 bytes, different middle: the shape the old
        // end-only fingerprint put in one bucket.
        let left = format!("{}L{}", "h".repeat(16), "t".repeat(64));
        let right = format!("{}R{}", "h".repeat(16), "t".repeat(64));
        let mut memo: TexturesPropertyMemo<String> = TexturesPropertyMemo::new();
        let first = memo.with(&left, |v| v.to_owned(), |v: &String| v.clone());
        let second = memo.with(&right, |v| v.to_owned(), |v: &String| v.clone());
        assert_eq!(first, left);
        assert_eq!(second, right, "values sharing their ends must not alias");
        assert_eq!(memo.len(), 2);

        let long = "x".repeat(TexturesPropertyMemo::<usize>::MAX_VALUE_BYTES + 1);
        let mut memo: TexturesPropertyMemo<usize> = TexturesPropertyMemo::new();
        let mut long_decodes = 0;
        for _ in 0..3 {
            let decoded = memo.with(
                &long,
                |value| {
                    long_decodes += 1;
                    value.len()
                },
                |n| *n,
            );
            assert_eq!(decoded, long.len());
        }
        assert_eq!(long_decodes, 3, "an over-long value is decoded every time");
        assert!(memo.is_empty(), "and never stored");

        let mut memo: TexturesPropertyMemo<u32> = TexturesPropertyMemo::new();
        for n in 0..TexturesPropertyMemo::<u32>::CAPACITY as u32 {
            memo.with(&n.to_string(), |_| n, |_| ());
        }
        assert_eq!(memo.len(), TexturesPropertyMemo::<u32>::CAPACITY);
        memo.with("one more", |_| 0, |_| ());
        assert_eq!(memo.len(), 1, "full memo starts over");
    }

    fn png(w: u32, h: u32, colour: [u8; 4]) -> Vec<u8> {
        let image = image::RgbaImage::from_raw(w, h, solid(w, h, colour)).expect("dimensions");
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .expect("encode png");
        bytes.into_inner()
    }

    fn solid(w: u32, h: u32, colour: [u8; 4]) -> Vec<u8> {
        colour
            .iter()
            .copied()
            .cycle()
            .take((w * h * 4) as usize)
            .collect()
    }

    #[test]
    fn layout_is_classified_by_size_only() {
        assert_eq!(SkinLayout::classify(64, 32), Some(SkinLayout::Legacy64x32));
        assert_eq!(SkinLayout::classify(64, 64), Some(SkinLayout::Modern64x64));
        assert_eq!(
            SkinLayout::classify(128, 128),
            Some(SkinLayout::Modern64x64)
        );
        assert_eq!(SkinLayout::classify(32, 32), Some(SkinLayout::Legacy64x32));
    }

    #[test]
    fn uuid_hash_matches_java() {
        // java.util.UUID.hashCode: (int)(hilo >> 32) ^ (int)hilo over msb ^ lsb.
        // 00000000-0000-0001-0000-000000000001 -> msb = 1, lsb = 1, hilo = 0 -> hash 0.
        assert_eq!(uuid_hash_code(0x0000_0000_0000_0001_0000_0000_0000_0001), 0);
        // msb = 0, lsb = 1 -> hilo = 1 -> (0) ^ (1) = 1.
        assert_eq!(uuid_hash_code(0x0000_0000_0000_0000_0000_0000_0000_0001), 1);
        // msb = 1 << 32, lsb = 0 -> hilo = 1 << 32 -> (1) ^ (0) = 1.
        assert_eq!(uuid_hash_code(0x0000_0001_0000_0000_0000_0000_0000_0000), 1);
        // The sign has to survive: hilo high word 0xFFFFFFFF, low word 0 -> -1.
        assert_eq!(
            uuid_hash_code(0xFFFF_FFFF_0000_0000_0000_0000_0000_0000u128),
            -1
        );
    }

    #[test]
    fn default_model_follows_the_uuid_parity_rule() {
        assert_eq!(SkinModel::default_for_uuid(1), SkinModel::Slim);
        assert_eq!(SkinModel::default_for_uuid(0), SkinModel::Classic);
        assert_eq!(SkinModel::default_for_uuid(2), SkinModel::Classic);
        assert_eq!(SkinModel::default_for_uuid(3), SkinModel::Slim);
        assert_eq!(
            SkinModel::Slim.default_texture(),
            "textures/entity/alex.png"
        );
        assert_eq!(
            SkinModel::Classic.default_texture(),
            "textures/entity/steve.png"
        );
        // A -1 hash is odd in two's complement, so it is slim.
        assert_eq!(
            SkinModel::default_for_uuid(0xFFFF_FFFF_0000_0000_0000_0000_0000_0000u128),
            SkinModel::Slim
        );
    }

    #[test]
    fn model_parse_only_accepts_the_exact_string() {
        assert_eq!(SkinModel::parse("slim"), SkinModel::Slim);
        assert_eq!(SkinModel::parse("default"), SkinModel::Classic);
        assert_eq!(SkinModel::parse("Slim"), SkinModel::Classic);
        assert_eq!(SkinModel::parse(""), SkinModel::Classic);
    }

    #[test]
    fn normalising_always_produces_a_64x64_image() {
        for (w, h) in [(64u32, 32u32), (64, 64)] {
            let (px, layout) = normalise_skin(&solid(w, h, [10, 20, 30, 255]), w, h).unwrap();
            assert_eq!(px.len(), 64 * 64 * 4);
            assert_eq!(layout, SkinLayout::classify(w, h).unwrap());
        }
    }

    #[test]
    fn arbitrary_skin_sizes_are_cropped_or_padded_on_the_top_left_canvas() {
        let mut small = solid(2, 2, [0, 0, 0, 0]);
        small[0..4].copy_from_slice(&[9, 8, 7, 6]);
        let (small_px, small_layout) = normalise_skin(&small, 2, 2).unwrap();
        assert_eq!(small_layout, SkinLayout::Modern64x64);
        assert_eq!(&small_px[0..4], &[9, 8, 7, 255]);
        let padded = ((40 * 64 + 63) * 4) as usize;
        assert_eq!(&small_px[padded..padded + 4], &[0, 0, 0, 0]);

        let mut large = solid(128, 128, [0, 0, 0, 0]);
        let kept = ((63 * 128 + 63) * 4) as usize;
        large[kept..kept + 4].copy_from_slice(&[1, 2, 3, 4]);
        let discarded = ((64 * 128 + 64) * 4) as usize;
        large[discarded..discarded + 4].copy_from_slice(&[99, 98, 97, 96]);
        let (large_px, large_layout) = normalise_skin(&large, 128, 128).unwrap();
        assert_eq!(large_layout, SkinLayout::Modern64x64);
        let last = ((63 * 64 + 63) * 4) as usize;
        assert_eq!(&large_px[last..last + 4], &[1, 2, 3, 4]);
        assert!(!large_px.windows(4).any(|pixel| pixel == [99, 98, 97, 96]));

        assert_eq!(
            normalise_skin(&[0; 15], 2, 2),
            Err(SkinError::BadSize(2, 2))
        );
    }

    #[test]
    fn a_legacy_skin_gets_its_left_arm_mirrored_from_the_right() {
        // Paint a single distinctive column inside the right-arm front face (44..48, 20..32)
        // and check it lands mirrored in the left-arm front slot (36..40, 52..64).
        let mut src = solid(64, 32, [0, 0, 0, 255]);
        let mark = |px: &mut [u8], x: u32, y: u32, c: [u8; 4]| {
            let i = ((y * 64 + x) * 4) as usize;
            px[i..i + 4].copy_from_slice(&c);
        };
        mark(&mut src, 44, 20, [255, 0, 0, 255]); // leftmost column of the face
        mark(&mut src, 47, 20, [0, 255, 0, 255]); // rightmost column

        let (px, _) = normalise_skin(&src, 64, 32).unwrap();
        let at = |x: u32, y: u32| {
            let i = ((y * 64 + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };
        // java: drawImage(bi, 40, 52, 36, 64, 44, 20, 48, 32) — source columns 44..48 land in
        // destination columns 36..40 *reversed*.
        assert_eq!(at(39, 52), [255, 0, 0, 255], "source x=44 -> dest x=39");
        assert_eq!(at(36, 52), [0, 255, 0, 255], "source x=47 -> dest x=36");
    }

    #[test]
    fn a_modern_skin_is_copied_through_unchanged_outside_the_fixups() {
        let mut src = solid(64, 64, [1, 2, 3, 255]);
        // A pixel in the left-leg region (16..48, 48..64) that setAreaOpaque touches only in
        // alpha, so the colour must survive.
        let i = ((50 * 64 + 20) * 4) as usize;
        src[i..i + 4].copy_from_slice(&[9, 8, 7, 255]);
        let (px, layout) = normalise_skin(&src, 64, 64).unwrap();
        assert_eq!(layout, SkinLayout::Modern64x64);
        assert_eq!(&px[i..i + 4], &[9, 8, 7, 255]);
    }

    #[test]
    fn a_fully_opaque_overlay_region_is_cleared() {
        // java: setAreaTransparent(32, 0, 64, 32) — a skin saved without an alpha channel would
        // otherwise render a solid box around the head.
        let src = solid(64, 64, [200, 200, 200, 255]);
        let (px, _) = normalise_skin(&src, 64, 64).unwrap();
        let alpha = |x: u32, y: u32| px[((y * 64 + x) * 4 + 3) as usize];
        assert_eq!(
            alpha(40, 8),
            0,
            "hat layer was opaque everywhere, so it is cleared"
        );
        assert_eq!(alpha(8, 8), 255, "the head itself is forced opaque");
    }

    #[test]
    fn a_partly_transparent_overlay_region_is_left_alone() {
        let mut src = solid(64, 64, [200, 200, 200, 255]);
        // One transparent pixel inside the hat region means the author did use alpha.
        let i = ((5 * 64 + 40) * 4) as usize;
        src[i + 3] = 0;
        let (px, _) = normalise_skin(&src, 64, 64).unwrap();
        let alpha = |x: u32, y: u32| px[((y * 64 + x) * 4 + 3) as usize];
        assert_eq!(alpha(41, 5), 255, "the rest of the hat keeps its alpha");
        assert_eq!(alpha(40, 5), 0);
    }

    #[test]
    fn the_cache_never_blocks_and_queues_exactly_once() {
        let (mut cache, channel) = SkinCache::new();
        assert!(cache
            .get("abc", "http://example/skin.png", SkinModel::Classic)
            .is_none());
        assert!(cache.is_inflight("abc"));
        // Asking again while in flight must not queue a second download.
        assert!(cache
            .get("abc", "http://example/skin.png", SkinModel::Classic)
            .is_none());
        let reqs: Vec<_> = channel.requests.try_iter().collect();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].key, "abc");
        assert_eq!(cache.stats(), (1, 0));

        // Deliver it by hand, standing in for the worker.
        let (rgba, layout) = normalise_skin(&solid(64, 64, [1, 1, 1, 255]), 64, 64).unwrap();
        channel
            .responses
            .send(SkinResponse {
                key: "abc".into(),
                result: Ok(SkinTexture {
                    rgba: rgba.into(),
                    layout,
                    model: SkinModel::Classic,
                }),
            })
            .unwrap();
        assert_eq!(cache.poll(), 1);
        assert!(cache.contains("abc"));
        assert!(!cache.is_inflight("abc"));
        assert_eq!(cache.stats(), (1, 1));
        assert_eq!(
            cache
                .get("abc", "http://example/skin.png", SkinModel::Classic)
                .unwrap()
                .layout,
            SkinLayout::Modern64x64
        );
    }

    #[test]
    fn capes_decode_at_native_size_and_cache_once() {
        let bytes = png(10, 20, [7, 8, 9, 255]);
        let cape = decode_cape(&bytes).expect("valid cape PNG");
        assert_eq!((cape.width, cape.height), (10, 20));
        assert_eq!(cape.rgba.len(), 10 * 20 * 4);

        let (mut cache, channel) = CapeCache::new();
        assert!(cache
            .get(
                "profile/cape",
                "https://textures.minecraft.net/texture/cape-hash"
            )
            .is_none());
        assert!(cache
            .get(
                "profile/cape",
                "https://textures.minecraft.net/texture/cape-hash"
            )
            .is_none());
        let requests: Vec<_> = channel.requests.try_iter().collect();
        assert_eq!(
            requests.len(),
            1,
            "a pending cape must not refetch per frame"
        );
        assert_eq!(requests[0].key, "profile/cape");

        channel
            .responses
            .send(CapeResponse {
                key: "profile/cape".to_owned(),
                result: Ok(cape.clone()),
            })
            .unwrap();
        assert_eq!(cache.poll(), 1);
        assert_eq!(cache.stats(), (1, 1));
        assert_eq!(cache.get("profile/cape", "ignored").unwrap(), &cape);
    }

    #[test]
    fn cape_disk_cache_uses_the_same_safe_profile_hash_shard() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("skins");
        let (mut cache, channel) = CapeCache::with_disk_cache(&root);
        cache.get(
            "profile/cape",
            "https://textures.minecraft.net/texture/abcdef0123456789",
        );
        let request = channel.requests.try_recv().expect("one cape request");
        assert_eq!(
            request.cache_file,
            Some(root.join("ab").join("abcdef0123456789"))
        );
    }

    /// java: `SkinManager.skinCacheLoader` (keys compare id and
    /// name, 15 s expire-after-access): an equal profile keeps the first textures while heads
    /// keep drawing, takes its own once the entry has been idle 15 s, and another name with the
    /// same UUID is another entry. Previously, a UUID key kept the first skin forever.
    #[test]
    fn player_head_profiles_follow_the_java_descriptor_cache() {
        let start = std::time::Instant::now();
        let textures = |hash: &str| PlayerHeadTextures {
            skin: Some((
                format!("mc-render/entity/skin/{hash}"),
                format!("https://textures.minecraft.net/texture/{hash}"),
            )),
            model: SkinModel::Classic,
        };
        let mut cache = PlayerHeadProfiles::default();
        let mut get = |name: &str, hash: &str, seconds: u64| {
            let now = start + std::time::Duration::from_secs(seconds);
            cache.get(7, name, now, || textures(hash)).clone()
        };
        assert_eq!(get("HeadOwner", "first", 0), textures("first"));
        assert_eq!(
            get("HeadOwner", "changed", 10),
            textures("first"),
            "hot: equal profile"
        );
        assert_eq!(
            get("HeadOwner", "changed", 24),
            textures("first"),
            "access restarts 15 s"
        );
        assert_eq!(
            get("HeadOwner", "changed", 39),
            textures("changed"),
            "idle 15 s: reloaded"
        );
        assert_eq!(
            get("HeadOwner", "first", 40),
            textures("changed"),
            "refreshed and shared"
        );
        assert_eq!(
            get("OtherName", "other", 40),
            textures("other"),
            "another name, own entry"
        );
    }

    /// Entries idle for the TTL are dropped when the cache grows past its sweep point, and a
    /// sweep that finds everything fresh moves the next one out to twice the size, so a large
    /// live cache is not swept again on every head lookup.
    #[test]
    fn player_head_profiles_sweep_idle_entries_amortised() {
        let start = std::time::Instant::now();
        let bare = || PlayerHeadTextures {
            skin: None,
            model: SkinModel::Classic,
        };
        let mut cache = PlayerHeadProfiles::default();
        for id in 0..PLAYER_HEAD_SWEEP_AT as u128 {
            cache.get(id, "head", start, bare);
        }
        assert_eq!(cache.len(), PLAYER_HEAD_SWEEP_AT);
        let one_second = start + std::time::Duration::from_secs(1);
        cache.get(1 << 40, "late", one_second, bare);
        assert_eq!(
            cache.len(),
            PLAYER_HEAD_SWEEP_AT + 1,
            "fresh entries survive a sweep"
        );
        assert_eq!(cache.sweep_at, PLAYER_HEAD_SWEEP_AT * 2);
        // At start + TTL every first-batch entry is exactly at the TTL (expired, Guava's `>=`);
        // the late one is 14 s old.
        cache.sweep(start + PLAYER_HEAD_PROFILE_TTL);
        assert_eq!(cache.len(), 1);
        assert!(cache.by_id.contains_key(&(1 << 40)));
    }

    /// `MinecraftProfileTexture.getHash()`: the URL basename, filename-safe ASCII only.
    #[test]
    fn skin_keys_name_the_texture_hash() {
        assert_eq!(
            skin_key_for_texture_url("https://textures.minecraft.net/texture/ab12?x=1#y"),
            Some("mc-render/entity/skin/ab12".to_owned())
        );
        assert_eq!(
            skin_key_for_texture_url("https://textures.minecraft.net/texture/"),
            None
        );
        assert_eq!(
            skin_key_for_texture_url("https://textures.minecraft.net/texture/a b"),
            None
        );
    }

    #[test]
    fn shared_item_requests_join_the_existing_cache_without_network_in_the_test() {
        let key = "mc-render/entity/skin/shared-item-request-test";
        request_shared_skin(
            key,
            "https://textures.minecraft.net/texture/shared-item-test",
            SkinModel::Slim,
        );
        let (mut cache, channel) = SkinCache::new();
        // Poll only transfers the request into the ordinary channel; no worker is started here.
        assert_eq!(cache.poll(), 0);
        let request = channel
            .requests
            .try_recv()
            .expect("item request reaches the normal cache channel");
        assert_eq!(request.key, key);
        assert_eq!(request.model, SkinModel::Slim);
        assert_eq!(cache.stats(), (1, 0));
    }

    #[test]
    fn disk_cache_paths_match_authlibs_hash_sharding() {
        let root = Path::new("skins");
        assert_eq!(
            skin_cache_path(
                root,
                "https://textures.minecraft.net/texture/abcdef0123456789"
            ),
            Some(root.join("ab").join("abcdef0123456789"))
        );
        assert_eq!(
            skin_cache_path(root, "https://example.invalid/a.png?version=1"),
            Some(root.join("xx").join("a")),
            "authlib uses the base name and short hashes go under xx"
        );
        assert_eq!(skin_cache_path(root, "https://example.invalid/"), None);
        assert_eq!(
            skin_cache_path(root, "https://example.invalid/%2e%2e"),
            None,
            "a hostile path segment cannot escape the cache root"
        );
    }

    #[test]
    fn configured_cache_attaches_the_vanilla_file_to_requests() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("skins");
        let (mut cache, channel) = SkinCache::with_disk_cache(&root);
        cache.get(
            "player",
            "https://textures.minecraft.net/texture/abcdef",
            SkinModel::Slim,
        );
        let request = channel.requests.try_recv().expect("one request");
        assert_eq!(request.cache_file, Some(root.join("ab").join("abcdef")));
    }

    #[test]
    fn a_valid_disk_hit_decodes_and_a_corrupt_one_falls_through() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ab").join("abcdef");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, png(64, 64, [5, 6, 7, 255])).unwrap();
        let request = SkinRequest {
            key: "player".into(),
            url: "https://textures.minecraft.net/texture/abcdef".into(),
            model: SkinModel::Slim,
            cache_file: Some(path.clone()),
        };
        let response = cached_response(&request).expect("cache hit avoids HTTP");
        let skin = response.result.expect("valid cached skin");
        assert_eq!(skin.model, SkinModel::Slim);
        assert_eq!(skin.layout, SkinLayout::Modern64x64);

        std::fs::write(&path, b"corrupt").unwrap();
        assert!(
            cached_response(&request).is_none(),
            "decode failure is a cache miss so the worker downloads again"
        );
    }

    #[test]
    fn downloaded_bytes_are_persisted_before_the_skin_is_delivered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ab").join("abcdef");
        let bytes = png(64, 32, [9, 8, 7, 255]);
        let skin = cache_and_decode_download(&bytes, Some(&path), SkinModel::Classic).unwrap();
        assert_eq!(skin.layout, SkinLayout::Legacy64x32);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn a_failed_fetch_is_remembered_and_not_retried() {
        let (mut cache, channel) = SkinCache::new();
        cache.get("bad", "http://example/404.png", SkinModel::Classic);
        let _ = channel.requests.try_iter().count();
        channel
            .responses
            .send(SkinResponse {
                key: "bad".into(),
                result: Err(SkinError::Network("404".into())),
            })
            .unwrap();
        assert_eq!(cache.poll(), 1);
        assert!(cache.failure("bad").is_some());

        cache.get("bad", "http://example/404.png", SkinModel::Classic);
        assert_eq!(channel.requests.try_iter().count(), 0, "no retry storm");

        cache.clear_failure("bad");
        cache.get("bad", "http://example/404.png", SkinModel::Classic);
        assert_eq!(
            channel.requests.try_iter().count(),
            1,
            "explicit retry works"
        );
    }

    #[test]
    fn a_dead_worker_does_not_panic_the_frame_loop() {
        let (mut cache, channel) = SkinCache::new();
        drop(channel);
        assert!(cache
            .get("x", "http://example/x.png", SkinModel::Slim)
            .is_none());
        assert_eq!(cache.poll(), 0);
    }

    #[test]
    fn decode_rejects_non_images() {
        assert!(matches!(
            decode_skin(b"not a png", SkinModel::Classic),
            Err(SkinError::Decode(_))
        ));
    }


    /// A decoder panic becomes a failed texture (the crash hook ignores contained panics), so the
    /// game, and a cache receiver thread, carry on.
    #[test]
    fn a_panicking_texture_decode_is_a_failed_texture() {
        let panicked: Result<(), SkinError> = contained_decode(|| panic!("decoder bug"));
        assert_eq!(
            panicked,
            Err(SkinError::Decode("the texture decoder panicked".to_owned()))
        );
        assert_eq!(contained_decode(|| Ok::<_, SkinError>(7)), Ok(7));
    }

    #[test]
    fn only_mojangs_texture_host_is_fetched() {
        for url in [
            "http://textures.minecraft.net/texture/abcdef0123456789",
            "https://textures.minecraft.net/texture/3b60a1f6d562f52aaebbf1434f1de147933a3affe0e764fa49ea057536623cd3",
            "http://textures.minecraft.net/texture/test-hash_2",
        ] {
            assert!(is_mojang_texture_url(url), "{url}");
        }
        for url in [
            // The request's parser ends an http authority at a backslash: these connect to the
            // host before it.
            "http://evil.example\\@textures.minecraft.net/texture/abc",
            "http://127.0.0.1:8080\\@textures.minecraft.net/texture/abc",
            "http://textures.minecraft.net@evil.example/texture/abc",
            "http://user@textures.minecraft.net/texture/abc",
            "http://textures.minecraft.net:8080/texture/abc",
            "http://textures.minecraft.net:80/texture/abc",
            "http://textures.minecraft.net.evil.example/texture/abc",
            "http://evil.example/textures.minecraft.net/texture/abc",
            // authlib's suffix rule allows these hosts; the client does not.
            "http://api.mojang.com/texture/abc",
            "https://sessionserver.mojang.com/session/minecraft/profile/abc",
            "http://evil.minecraft.net/texture/abc",
            // Not the canonical `/texture/<hash>` form.
            "http://textures.minecraft.net/abc",
            "http://textures.minecraft.net/texture/",
            "http://textures.minecraft.net/texture/abc?x=1",
            "http://textures.minecraft.net/texture/abc#frag",
            "http://textures.minecraft.net/texture/a/../b",
            "http://textures.minecraft.net/texture/a%2Fb",
            "http://textures.minecraft.net/texture/abc.png",
            "http://TEXTURES.minecraft.net/texture/abc",
            " http://textures.minecraft.net/texture/abc",
            "ftp://textures.minecraft.net/texture/abc",
            "file:///C:/texture/abc",
            "//textures.minecraft.net/texture/abc",
            "",
            // Windows device names (the cache file would be a device).
            "http://textures.minecraft.net/texture/CON",
            "http://textures.minecraft.net/texture/con",
            "http://textures.minecraft.net/texture/NUL",
            "http://textures.minecraft.net/texture/COM1",
            "http://textures.minecraft.net/texture/lpt9",
        ] {
            assert!(!is_mojang_texture_url(url), "{url:?}");
        }
        // Nor do they name a cache file, whatever asked for one; near misses still do.
        let root = Path::new("cache");
        for hash in ["CON", "aux", "Prn", "nul", "COM3", "LPT0"] {
            let url = format!("http://textures.minecraft.net/texture/{hash}");
            assert_eq!(skin_cache_path(root, &url), None, "{hash}");
        }
        for hash in ["CONSOLE", "COM10", "auxiliary", "NULL", "abc123"] {
            let url = format!("http://textures.minecraft.net/texture/{hash}");
            assert!(is_mojang_texture_url(&url), "{hash}");
            assert!(skin_cache_path(root, &url).is_some(), "{hash}");
        }
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
    }

    /// A local, non-blocking listener: `accept` tells whether anything connected to it.
    fn quiet_listener() -> (std::net::TcpListener, u16) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1");
        listener
            .set_nonblocking(true)
            .expect("non-blocking listener");
        let port = listener.local_addr().expect("local address").port();
        (listener, port)
    }

    /// Answer one local HTTP request with `response`, after reading its headers.
    fn serve_once(response: Vec<u8>) -> u16 {
        use std::io::{Read as _, Write as _};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1");
        let port = listener.local_addr().expect("local address").port();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => return,
                    Ok(read) => request.extend_from_slice(&buffer[..read]),
                }
            }
            let _ = stream.write_all(&response);
        });
        port
    }

    /// [`serve_once`], also handing back the request's head (request line and headers).
    fn serve_recording(response: Vec<u8>) -> (u16, std::sync::mpsc::Receiver<String>) {
        use std::io::{Read as _, Write as _};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1");
        let port = listener.local_addr().expect("local address").port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => return,
                    Ok(read) => request.extend_from_slice(&buffer[..read]),
                }
            }
            let _ = tx.send(String::from_utf8_lossy(&request).into_owned());
            let _ = stream.write_all(&response);
        });
        (port, rx)
    }

    /// The fix keeps the real path working: a canonical texture URL is fetched (as the parsed URL
    /// that was checked) and decoded. `resolve` points Mojang's texture host at a local mock, so
    /// nothing leaves the machine.
    #[test]
    fn a_mojang_texture_url_is_still_fetched_and_decoded() {
        let ok = |body: Vec<u8>| {
            let mut response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .into_bytes();
            response.extend_from_slice(&body);
            response
        };
        let mock = |port: u16| {
            // No proxy: a system proxy would otherwise be asked for Mojang's host.
            texture_http_client_builder()
                .no_proxy()
                .resolve(
                    MOJANG_TEXTURE_HOST,
                    std::net::SocketAddr::from(([127, 0, 0, 1], port)),
                )
                .build()
                .expect("texture client")
        };
        let runtime = runtime();

        let (port, requests) = serve_recording(ok(png(64, 64, [10, 20, 30, 255])));
        let skin = runtime
            .block_on(fetch_skin(
                &mock(port),
                "http://textures.minecraft.net/texture/abc123",
                SkinModel::Slim,
                None,
            ))
            .expect("a Mojang texture URL loads");
        assert_eq!(
            (skin.layout, skin.model),
            (SkinLayout::Modern64x64, SkinModel::Slim)
        );
        let request = requests
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the mock saw the request");
        assert!(
            request.starts_with("GET /texture/abc123 HTTP/1.1\r\n"),
            "{request}"
        );
        assert!(
            request
                .to_ascii_lowercase()
                .contains("host: textures.minecraft.net\r\n"),
            "{request}"
        );

        let (port, _requests) = serve_recording(ok(png(64, 32, [40, 50, 60, 255])));
        let cape = runtime
            .block_on(fetch_cape(
                &mock(port),
                "http://textures.minecraft.net/texture/cape-1",
                None,
            ))
            .expect("a Mojang cape URL loads");
        assert_eq!((cape.width, cape.height), (64, 32));
    }

    /// A local texture host that answers every request with `body` after `delay`, each on its
    /// own thread, and counts the requests it holds at once: (port, most at once, answered).
    fn serve_slowly(
        body: Vec<u8>,
        delay: std::time::Duration,
    ) -> (
        u16,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use std::io::{Read as _, Write as _};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1");
        let port = listener.local_addr().expect("local address").port();
        let peak = Arc::new(AtomicUsize::new(0));
        let answered = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let mut response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(&body);
        let response = Arc::new(response);
        let (peak_out, answered_out) = (Arc::clone(&peak), Arc::clone(&answered));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else {
                    return;
                };
                let (peak, answered, in_flight, response) = (
                    Arc::clone(&peak),
                    Arc::clone(&answered),
                    Arc::clone(&in_flight),
                    Arc::clone(&response),
                );
                std::thread::spawn(move || {
                    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
                    let mut request = Vec::new();
                    let mut buffer = [0u8; 1024];
                    while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                        match stream.read(&mut buffer) {
                            Ok(0) | Err(_) => return,
                            Ok(read) => request.extend_from_slice(&buffer[..read]),
                        }
                    }
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(delay);
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    answered.fetch_add(1, Ordering::SeqCst);
                    let _ = stream.write_all(&response);
                });
            }
        });
        (port, peak_out, answered_out)
    }

    /// A join burst of 20 distinct skins and capes: at most
    /// `MAX_TEXTURE_FETCHES` requests reach the texture host at once (the first ones at once,
    /// the others as slots free up), and every texture still arrives. Control: the same 20
    /// requests sent without the bound all reach the host together, so the bound is the client's
    /// and not the host's or the HTTP stack's.
    #[test]
    fn texture_downloads_are_bounded_and_every_texture_still_arrives() {
        use std::sync::atomic::Ordering;
        const BURST: usize = 20;
        let delay = std::time::Duration::from_millis(200);
        let client_for = |port: u16| {
            texture_http_client_builder()
                .no_proxy()
                .resolve(
                    MOJANG_TEXTURE_HOST,
                    std::net::SocketAddr::from(([127, 0, 0, 1], port)),
                )
                .build()
                .expect("texture client")
        };
        let runtime = runtime();

        let (port, peak, answered) = serve_slowly(png(64, 64, [10, 20, 30, 255]), delay);
        let client = client_for(port);
        let arrived = runtime.block_on(async {
            let tasks: Vec<_> = (0..BURST)
                .map(|index| {
                    let client = client.clone();
                    tokio::spawn(async move {
                        let url = format!("http://textures.minecraft.net/texture/burst{index}");
                        if index % 4 == 3 {
                            fetch_cape(&client, &url, None).await.map(|_| ())
                        } else {
                            fetch_skin(&client, &url, SkinModel::Classic, None)
                                .await
                                .map(|_| ())
                        }
                    })
                })
                .collect();
            let mut arrived = 0;
            for task in tasks {
                if task.await.expect("fetch task").is_ok() {
                    arrived += 1;
                }
            }
            arrived
        });
        assert_eq!(arrived, BURST, "every texture of the burst arrives");
        assert_eq!(answered.load(Ordering::SeqCst), BURST);
        let peak = peak.load(Ordering::SeqCst);
        // Another test may hold a slot for a moment while this one runs.
        assert!(
            peak <= MAX_TEXTURE_FETCHES && peak + 2 >= MAX_TEXTURE_FETCHES,
            "{peak} requests at the host at once"
        );

        let (port, control_peak, _) = serve_slowly(png(64, 64, [10, 20, 30, 255]), delay);
        let client = client_for(port);
        runtime.block_on(async {
            let tasks: Vec<_> = (0..BURST)
                .map(|index| {
                    let client = client.clone();
                    tokio::spawn(async move {
                        let url = format!("http://textures.minecraft.net/texture/control{index}");
                        let response = client.get(url).send().await.expect("control request");
                        let _ = response.bytes().await;
                    })
                })
                .collect();
            for task in tasks {
                task.await.expect("control task");
            }
        });
        let control_peak = control_peak.load(Ordering::SeqCst);
        eprintln!("requests at the host at once: {peak} bounded, {control_peak} unbounded");
        assert!(
            control_peak > MAX_TEXTURE_FETCHES + 4,
            "without the bound the burst reaches the host together: {control_peak}"
        );
    }

    /// A cache file larger than any download may be is a miss, never read
    /// whole: here a real skin followed by padding past the cap, which the decoder alone still
    /// accepts (the control), so the old whole-file read would have used it. A normal cache file
    /// still loads, skins and capes alike.
    #[test]
    fn an_oversized_cache_file_is_a_miss() {
        let dir = tempfile::tempdir().expect("temp dir");
        let skin_request = |name: &str| SkinRequest {
            key: "key".to_owned(),
            url: String::new(),
            model: SkinModel::Classic,
            cache_file: Some(dir.path().join(name)),
        };
        let skin = png(64, 64, [10, 20, 30, 255]);
        std::fs::write(dir.path().join("normal"), &skin).expect("write");
        assert!(cached_response(&skin_request("normal")).is_some_and(|hit| hit.result.is_ok()));

        let mut padded = skin.clone();
        padded.resize(MAX_TEXTURE_DOWNLOAD_BYTES + 1, 0);
        assert!(
            decode_skin(&padded, SkinModel::Classic).is_ok(),
            "control: the decoder alone takes the padded file"
        );
        std::fs::write(dir.path().join("padded"), &padded).expect("write");
        assert!(cached_response(&skin_request("padded")).is_none());

        let cape_request = |name: &str| CapeRequest {
            key: "key".to_owned(),
            url: String::new(),
            cache_file: Some(dir.path().join(name)),
        };
        let mut cape = png(64, 32, [40, 50, 60, 255]);
        std::fs::write(dir.path().join("cape"), &cape).expect("write");
        assert!(cached_cape_response(&cape_request("cape")).is_some());
        cape.resize(MAX_TEXTURE_DOWNLOAD_BYTES + 1, 0);
        std::fs::write(dir.path().join("cape-padded"), &cape).expect("write");
        assert!(cached_cape_response(&cape_request("cape-padded")).is_none());
    }

    /// Give a request that will never be answered a moment to connect, then give up on it.
    fn send_briefly(runtime: &tokio::runtime::Runtime, client: &reqwest::Client, url: &str) {
        // `send` starts the client's timeout timer, so it runs inside the runtime.
        let _ = runtime.block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(2), client.get(url).send()).await
        });
    }

    /// The old hand-written gate accepted `http://<host>\@textures.minecraft.net/...` because
    /// the text after the `@` names Mojang's host, but the request's parser connects to
    /// `<host>`. A local listener stands in for that host.
    #[test]
    fn a_backslash_cannot_move_a_texture_fetch_to_another_host() {
        let (listener, port) = quiet_listener();
        let url = format!("http://127.0.0.1:{port}\\@textures.minecraft.net/texture/abcdef");
        let parsed = reqwest::Url::parse(&url).expect("the request parser accepts the URL");
        assert_eq!(
            parsed.host_str(),
            Some("127.0.0.1"),
            "and connects to the listener"
        );

        let runtime = runtime();
        let client = texture_http_client();
        let skin = runtime.block_on(fetch_skin(&client, &url, SkinModel::Classic, None));
        assert!(
            matches!(&skin, Err(error) if *error == off_host_texture()),
            "{:?}",
            skin.err()
        );
        let cape = runtime.block_on(fetch_cape(&client, &url, None));
        assert!(
            matches!(&cape, Err(error) if *error == off_host_texture()),
            "{:?}",
            cape.err()
        );
        assert!(listener.accept().is_err(), "the fetchers never connected");

        // Negative control: the same URL requested without the gate does reach the listener.
        send_briefly(&runtime, &client, &url);
        assert!(
            listener.accept().is_ok(),
            "without the gate the request goes to the listener"
        );
    }

    #[test]
    fn a_texture_redirect_cannot_leave_mojangs_texture_host() {
        let (target, target_port) = quiet_listener();
        let redirect = format!(
            "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{target_port}/texture/abc\r\n\
             Content-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let runtime = runtime();
        let origin = serve_once(redirect.clone().into_bytes());
        let result = runtime.block_on(async {
            texture_http_client()
                .get(format!("http://127.0.0.1:{origin}/texture/abc"))
                .send()
                .await
        });
        assert!(
            result.as_ref().is_err_and(reqwest::Error::is_redirect),
            "{result:?}"
        );
        assert!(
            target.accept().is_err(),
            "the redirect target never saw a connection"
        );

        // Negative control: a client with the default redirect policy follows it.
        let origin = serve_once(redirect.into_bytes());
        send_briefly(
            &runtime,
            &reqwest::Client::new(),
            &format!("http://127.0.0.1:{origin}/texture/abc"),
        );
        assert!(
            target.accept().is_ok(),
            "the default policy follows the redirect"
        );
    }

    #[test]
    fn a_texture_body_is_capped_whatever_its_length_header_says() {
        let runtime = runtime();
        let client = texture_http_client();
        let fetch_body = |response: Vec<u8>| {
            let port = serve_once(response);
            runtime.block_on(async {
                let response = client
                    .get(format!("http://127.0.0.1:{port}/texture/abc"))
                    .send()
                    .await
                    .expect("local response");
                read_texture_body(response).await
            })
        };
        let over = MAX_TEXTURE_DOWNLOAD_BYTES + 1;

        // An honest length over the cap is refused before the body is read.
        let declared = format!("HTTP/1.1 200 OK\r\nContent-Length: {over}\r\n\r\n");
        assert!(fetch_body(declared.into_bytes()).is_err());

        // No length at all: the body runs until the connection closes, and is cut at the cap.
        let mut streamed = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_vec();
        streamed.resize(streamed.len() + over, 0);
        assert!(fetch_body(streamed).is_err());

        // A real skin's size is unaffected.
        let skin = png(64, 64, [1, 2, 3, 255]);
        let mut ok = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            skin.len()
        )
        .into_bytes();
        ok.extend_from_slice(&skin);
        assert_eq!(fetch_body(ok).expect("a normal skin"), skin);
    }

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFF_u32;
        for &byte in bytes {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    /// A PNG of a few dozen bytes whose header claims `width` x `height` RGBA pixels.
    fn png_claiming(width: u32, height: u32) -> Vec<u8> {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut chunk = |kind: &[u8; 4], data: &[u8]| {
            let mut body = kind.to_vec();
            body.extend_from_slice(data);
            png.extend_from_slice(&u32::try_from(data.len()).unwrap().to_be_bytes());
            png.extend_from_slice(&body);
            png.extend_from_slice(&crc32(&body).to_be_bytes());
        };
        let mut header = Vec::new();
        header.extend_from_slice(&width.to_be_bytes());
        header.extend_from_slice(&height.to_be_bytes());
        header.extend_from_slice(&[8, 6, 0, 0, 0]);
        chunk(b"IHDR", &header);
        chunk(b"IDAT", &[0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
        chunk(b"IEND", &[]);
        png
    }

    /// The size check runs on the PNG header, so a tiny file cannot make the decoder allocate the
    /// surface it claims (11000 x 11000 RGBA is 484 MB).
    #[test]
    fn a_texture_header_claiming_a_huge_image_is_refused_before_decoding() {
        let limit_error = image::ImageError::Limits(image::error::LimitError::from_kind(
            image::error::LimitErrorKind::DimensionError,
        ))
        .to_string();
        let huge = png_claiming(11_000, 11_000);
        assert!(huge.len() < 100);
        assert_eq!(
            decode_skin(&huge, SkinModel::Classic),
            Err(SkinError::Decode(limit_error.clone()))
        );
        assert_eq!(
            decode_cape(&huge),
            Err(SkinError::Decode(limit_error.clone()))
        );
        for (width, height) in [
            (MAX_TEXTURE_DIMENSION + 1, 1),
            (1, MAX_TEXTURE_DIMENSION + 1),
        ] {
            let bytes = png(width, height, [1, 2, 3, 255]);
            assert_eq!(
                decode_cape(&bytes),
                Err(SkinError::Decode(limit_error.clone()))
            );
        }
        // Up to the limit, decoding is unchanged.
        for (width, height) in [(MAX_TEXTURE_DIMENSION, 1), (1, MAX_TEXTURE_DIMENSION)] {
            let cape = decode_cape(&png(width, height, [1, 2, 3, 255])).expect("at the limit");
            assert_eq!((cape.width, cape.height), (width, height));
        }
        let skin = decode_skin(&png(64, 32, [4, 5, 6, 255]), SkinModel::Slim).expect("64x32");
        assert_eq!(skin.layout, SkinLayout::Legacy64x32);
    }
}
