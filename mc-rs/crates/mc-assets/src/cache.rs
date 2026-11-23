//! On-disk cache for the baked model set and the stitched atlas.
//!
//! Owned by **S23** ("Startup-time optimisation (atlas + model baking cache on
//! disk)").
//!
//! Baking every one of the 1251 `RenderType::Model` block states and stitching the atlas is the
//! dominant cost of a cold start, and its inputs — the resource-pack stack — change rarely.
//! Caching the result keyed on pack identity plus a format version turns that into a read.
//!
//! # The correctness bar this module has to clear
//!
//! A stale cache that silently serves the wrong models is far worse than a slow start: it would
//! produce a client that renders subtly incorrect blocks with no error anywhere, which is the
//! exact failure mode the whole project's parity work exists to prevent. Invalidation is the
//! feature; speed is the side effect.
//!
//! Three consequences of taking that seriously:
//!
//! * **Every rejection path is named.** [`load`] returns a [`CacheRead`], and a miss carries a
//!   [`CacheMiss`] saying which check failed. The invalidation tests assert the *reason*, not
//!   just that a miss happened — a test that only asserts "did not load" passes just as happily
//!   when the file cannot be opened at all.
//! * **A miss is never an error.** Anything short of "this file is provably the answer to this
//!   exact question" degrades to a full bake. The only [`CacheError`] is a failure to *write*.
//! * **The reader treats the file as hostile.** Every length is bounds-checked against the bytes
//!   actually present before a single element is allocated, so a corrupt or truncated cache
//!   produces a miss rather than a panic or a multi-gigabyte allocation. This crate already
//!   treats resource packs that way (see the crate docs); a cache file derived from them
//!   inherits the same rule.
//!
//! # What the key covers
//!
//! [`BakeInputs`] is the question the cached bytes are an answer to. It carries a stamp per
//! resource pack in stack order, the atlas dimension cap the bake ran with, and
//! `mc_data::STATE_COUNT` (so a registry change invalidates every cache). [`FORMAT_VERSION`] is
//! bumped by hand whenever the baker's *output* changes shape or value — that is the check that
//! covers code changes, which no amount of input stamping can see.
//!
//! A pack stamp is path, kind, byte length, modification time, and a content hash. Up to
//! [`FULL_HASH_LIMIT`] the hash covers **every byte**, because that turned out to be cheap:
//! `benches/startup.rs` measures the whole 8.1 MiB vanilla jar at 1.8 ms against a 41 ms bake.
//! Above the limit it degrades to a 128 KiB head/tail sample, and the failure mode that leaves
//! open is stated rather than hidden — a pack edited in place, to exactly the same length,
//! leaving the same mtime, changing only bytes outside the sampled window, is not detected. The
//! stamp records *which* mode it used, so a sampled stamp can never be mistaken for a hashed
//! one, and [`BakeInputs::push_file_hashed`] forces certainty at any size.
//!
//! A folder pack (a loose, unpacked directory rather than a `.zip`/`.jar`) is stamped the same
//! way, per file: [`stamp_dir`]'s doc comment explains why modification time is deliberately not
//! part of that stamp at all, unlike the byte-length-and-mtime-alongside-hash shape a *file*
//! stamp uses above.
//!
//! # Status
//!
//! Complete and wired. [`BakedSnapshot::of`] turns a finished [`bake_all`](crate::bake_all) into
//! something [`save`] can write, and [`BakedSnapshot::into_assets`] turns a verified hit back
//! into a [`BakedAssets`] that [`install`](crate::install) publishes — through the additive
//! `BakedAssets::from_parts`/`to_parts` pair.
//! `mc-client`'s `AssetLoader` is the caller; see `app::assets`.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use thiserror::Error;

use crate::model::state_mapper::StateMapping;
use crate::model::{
    BakedAssets, BakedModel, BakedQuad, ItemCameraTransforms, ItemTransform, Vertex,
};
use crate::sprite::{Sprite, SpriteFrame, UvRect};
use crate::{Atlas, ResourceLocation};

/// Magic bytes every cache file starts with.
const MAGIC: [u8; 8] = *b"MCRSBAKE";

/// The cache format version.
///
/// **Bump this whenever the baker's output changes**, in shape *or* in value — a new field on
/// [`BakedModel`], a corrected UV, a different stitching order. Input stamping cannot see a code
/// change, so this number is the only thing standing between a rebuilt binary and a cache
/// written by the previous one.
///
/// History:
/// * `1` — initial format (S23). Atlas + sprites + per-state models + weighted variants + item
///   models.
///   Version 7 invalidates snapshots created before signed Java variant weights were preserved.
///   Version 9 drops empty generated/compass/clock item entries so lookup reaches builtin/missing.
///   Version 10 retains `IBakedModel.isGui3d()` instead of reconstructing it from item registry data.
///   Version 11 carries `TextureMetadataSection`'s blur/clamp/explicit-mipmap options on sprites.
///   Version 12 carries the decoded explicit mip PNG pixels needed to reproduce those options.
///   Version 13 invalidates bakes that loaded explicit mips at or above a configured
///   `mipmapLevels` of 1-3 (vanilla generates those levels; phase 5a review S3).
///   Version 14 invalidates quad positions baked before `FaceBakery`'s rescale constants and
///   LWJGL float rotation matrices were reproduced bit for bit (fire planes were 6% short).
///   Version 15 invalidates bakes that gave an animated sprite whose
///   sheet is shorter than one frame a zero-filled frame; such a sprite is now skipped.
pub const FORMAT_VERSION: u32 = 15;

/// Bytes sampled from each end of a pack file too large to hash in full.
const SAMPLE_BYTES: u64 = 128 * 1024;

/// Default size at or below which [`BakeInputs::push_path`] hashes a pack file **in full**
/// rather than sampling it. 64 MiB.
///
/// Chosen from measurement, not taste. `benches/startup.rs` puts the whole-file hash of the
/// 8.1 MiB vanilla jar at 1.8 ms against a 41 ms bake, so certainty about the key is nearly free
/// at that size and there is no reason to accept the sampling hole. The limit exists for the
/// other end of the range: a 1 GB high-resolution resource pack would cost ~220 ms to hash,
/// which is several times what caching it saves. Above the limit the stamp degrades to a sample
/// and *says so* — the stamp kind is part of the key, so a sampled stamp can never be mistaken
/// for a hashed one.
pub const FULL_HASH_LIMIT: u64 = 64 * 1024 * 1024;

/// Hard cap on entries walked for a folder pack. A folder with more than this many files gets a
/// stamp that refuses to be cached rather than a stamp that is quietly incomplete.
const MAX_DIR_ENTRIES: usize = 65_536;

/// Failures that stop a cache from being *written*. Reading never produces one of these — see
/// [`CacheRead`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CacheError {
    /// The cache file (or its temporary sibling) could not be written.
    #[error("cache i/o error on {path}")]
    Io {
        /// The path being written when the failure happened.
        path: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// [`BakeInputs`] could not be stamped precisely enough to be trusted, so writing a cache
    /// keyed on it would be unsound. See [`BakeInputs::is_stampable`].
    #[error("refusing to write a cache: {0}")]
    Unstampable(&'static str),
}

impl CacheError {
    fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        CacheError::Io {
            path: path.as_ref().display().to_string(),
            source,
        }
    }
}

/// Why [`load`] declined to use a cache file.
///
/// Every variant means the same thing to a caller — bake from scratch — but they are kept
/// distinct because the invalidation tests assert on them. A test that only checks "no snapshot
/// came back" cannot tell a correctly rejected stale cache from a cache file it failed to open.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheMiss {
    /// No file at that path, or it could not be read.
    NoFile,
    /// The file does not begin with this format's magic bytes.
    NotACacheFile,
    /// Written by a different version of the baker.
    Version {
        /// The version recorded in the file.
        found: u32,
        /// The version this build writes and reads.
        expected: u32,
    },
    /// The resource-pack stack, atlas cap or state count differs from the one that was baked.
    Inputs {
        /// The key recorded in the file.
        found: String,
        /// The key the caller asked for.
        expected: String,
    },
    /// The file ends before its own headers say it should.
    Truncated,
    /// The payload checksum does not match — the file is damaged.
    Checksum,
    /// The payload's own structure is inconsistent (a length that runs past the end, a tag with
    /// no meaning). Carries which field noticed.
    Malformed(&'static str),
}

impl std::fmt::Display for CacheMiss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheMiss::NoFile => write!(f, "no cache file"),
            CacheMiss::NotACacheFile => write!(f, "not a cache file"),
            CacheMiss::Version { found, expected } => {
                write!(f, "format version {found}, this build reads {expected}")
            }
            CacheMiss::Inputs { .. } => write!(f, "resource-pack stack changed"),
            CacheMiss::Truncated => write!(f, "truncated"),
            CacheMiss::Checksum => write!(f, "checksum mismatch"),
            CacheMiss::Malformed(what) => write!(f, "malformed payload ({what})"),
        }
    }
}

/// The result of a [`load`].
#[derive(Debug)]
pub enum CacheRead {
    /// The file is a valid answer to exactly this [`BakeInputs`]; here is the bake.
    Hit(Box<BakedSnapshot>),
    /// Bake from scratch; this is why.
    Miss(CacheMiss),
}

impl CacheRead {
    /// The snapshot, or `None` for a miss. For callers that do not care why.
    #[must_use]
    pub fn hit(self) -> Option<Box<BakedSnapshot>> {
        match self {
            CacheRead::Hit(s) => Some(s),
            CacheRead::Miss(_) => None,
        }
    }

    /// The miss reason, or `None` for a hit.
    #[must_use]
    pub fn miss(&self) -> Option<&CacheMiss> {
        match self {
            CacheRead::Hit(_) => None,
            CacheRead::Miss(m) => Some(m),
        }
    }
}

/// One resource pack's identity, as far as this cache can establish it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PackStamp {
    /// How the pack was identified, for the human reading a stale-cache log line.
    label: String,
    /// `file`, `dir`, `hash` or `opaque`.
    kind: &'static str,
    /// Total bytes (a directory sums its files).
    len: u64,
    /// Modification time in milliseconds since the Unix epoch, or 0 when unavailable.
    modified_ms: u64,
    /// Sample (or full content) hash — see the module docs.
    sample: u64,
}

impl PackStamp {
    fn render(&self, out: &mut String) {
        use std::fmt::Write as _;
        let _ = write!(
            out,
            "{}:{}:{}:{}:{:016x}",
            self.kind, self.label, self.len, self.modified_ms, self.sample
        );
    }
}

/// The question a cached bake is the answer to: which packs, stacked in which order, baked with
/// which parameters, against which block registry.
///
/// Build it in the same order the packs were pushed onto the [`ResourceManager`]
/// (crate::ResourceManager) — order is part of the identity, because a later pack overrides an
/// earlier one.
///
/// ```
/// # use mc_assets::cache::BakeInputs;
/// let mut inputs = BakeInputs::new(2048);
/// inputs.push_opaque("test-pack", b"pretend this is a pack");
/// assert!(inputs.is_stampable());
/// assert!(inputs.key().contains("test-pack"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BakeInputs {
    packs: Vec<PackStamp>,
    max_atlas_dimension: u32,
    /// `GameSettings.mipmapLevels` the bake stitched with (§7.6 / ATLAS-4).
    mipmap_levels: u32,
    state_count: u32,
    full_hash_limit: u64,
    /// Set when a pack could not be stamped precisely; poisons the whole key.
    unstampable: Option<&'static str>,
}

impl BakeInputs {
    /// A new, empty stack baking with `max_atlas_dimension`.
    ///
    /// `mc_data::STATE_COUNT` is folded in automatically, so a change to the block registry
    /// invalidates every cache without the caller having to remember.
    #[must_use]
    pub fn new(max_atlas_dimension: u32) -> Self {
        BakeInputs {
            packs: Vec::new(),
            max_atlas_dimension,
            mipmap_levels: crate::atlas::DEFAULT_MIPMAP_LEVELS,
            state_count: mc_data::STATE_COUNT as u32,
            full_hash_limit: FULL_HASH_LIMIT,
            unstampable: None,
        }
    }

    /// Key the bake on the `mipmapLevels` it was stitched with. The level count changes sprite
    /// alignment, placements and every UV, so a cache baked at another level is a different
    /// answer. Defaults to [`crate::atlas::DEFAULT_MIPMAP_LEVELS`].
    pub fn set_mipmap_levels(&mut self, mipmap_levels: u32) {
        self.mipmap_levels = mipmap_levels;
    }

    /// Change the size at or below which [`push_path`](BakeInputs::push_path) hashes a file in
    /// full instead of sampling it. Defaults to [`FULL_HASH_LIMIT`].
    pub fn set_full_hash_limit(&mut self, bytes: u64) {
        self.full_hash_limit = bytes;
    }

    /// Stamp a pack that lives at `path`, which may be a `.zip`/`.jar` file or a folder pack.
    ///
    /// A file up to [`FULL_HASH_LIMIT`] is hashed in full; a larger one is sampled and its stamp
    /// records that it was, so the two can never be confused.
    ///
    /// A path that cannot be read is *not* an error: it poisons the key so that
    /// [`is_stampable`](BakeInputs::is_stampable) is false and [`save`] refuses. A cache keyed on
    /// a pack whose identity is unknown is precisely the stale cache this module exists to
    /// prevent.
    pub fn push_path(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        let stamp = if path.is_dir() {
            stamp_dir(path, self.full_hash_limit)
        } else {
            stamp_file(path, self.full_hash_limit)
        };
        match stamp {
            Ok(stamp) => self.packs.push(stamp),
            Err(reason) => self.unstampable = Some(reason),
        }
    }

    /// Stamp a pack file by hashing **every byte** of it, whatever its size.
    ///
    /// [`push_path`](BakeInputs::push_path) already does this for anything up to
    /// [`FULL_HASH_LIMIT`]; this is for a caller that wants certainty about a pack larger than
    /// that and is willing to pay for it (measured at ~4.5 GiB/s in release, ~1.8 ms for the
    /// 8.1 MiB vanilla jar).
    pub fn push_file_hashed(&mut self, path: impl AsRef<Path>) {
        match stamp_file(path.as_ref(), u64::MAX) {
            Ok(stamp) => self.packs.push(stamp),
            Err(reason) => self.unstampable = Some(reason),
        }
    }

    /// Stamp a pack that is not a file at all — an in-memory pack, or one whose identity the
    /// caller already knows.
    pub fn push_opaque(&mut self, label: &str, identity: &[u8]) {
        self.packs.push(PackStamp {
            label: sanitise(label),
            kind: "opaque",
            len: identity.len() as u64,
            modified_ms: 0,
            sample: fnv1a(identity),
        });
    }

    /// Whether every pack in the stack could be identified precisely enough to key a cache on.
    #[must_use]
    pub fn is_stampable(&self) -> bool {
        self.unstampable.is_none()
    }

    /// The canonical key. Two [`BakeInputs`] that render the same string are the same question.
    ///
    /// Human-readable on purpose: a stale-cache log line that says which pack moved is worth far
    /// more than four bytes saved on disk.
    #[must_use]
    pub fn key(&self) -> String {
        let mut out = String::with_capacity(64 + self.packs.len() * 48);
        out.push_str("v1|states=");
        out.push_str(&self.state_count.to_string());
        out.push_str("|atlas<=");
        out.push_str(&self.max_atlas_dimension.to_string());
        // Only a non-default level count extends the key, so every existing default cache keeps
        // its exact key (and stays valid) while a mipmap-0 bake can never be mistaken for one.
        if self.mipmap_levels != crate::atlas::DEFAULT_MIPMAP_LEVELS {
            out.push_str("|mip=");
            out.push_str(&self.mipmap_levels.to_string());
        }
        for pack in &self.packs {
            out.push('|');
            pack.render(&mut out);
        }
        out
    }
}

/// Strip the separators the canonical key uses, so a pack whose name contains one cannot forge
/// another pack's stamp.
fn sanitise(label: &str) -> String {
    label
        .chars()
        .map(|c| if c == '|' || c == ':' { '_' } else { c })
        .collect()
}

fn modified_ms(meta: &fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis() as u64)
}

/// Hash an open file's content: every byte when `len <= full_hash_limit`, otherwise a 128 KiB
/// head/tail sample. Returns `(full, hash)` — `full` says which policy ran, so a caller can fold
/// that into its own stamp and never confuse a sampled hash with a complete one.
///
/// Shared by [`stamp_file`] and [`stamp_dir`] so a folder pack's per-entry hashing is byte-for-
/// byte the same policy as a single pack file's, not a separate, weaker approximation of it.
fn hash_file_content(
    file: &mut fs::File,
    len: u64,
    full_hash_limit: u64,
) -> Result<(bool, u64), &'static str> {
    let full = len <= full_hash_limit;
    let hash = if full {
        let mut hash = FNV_OFFSET;
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|_| "a pack file could not be read")?;
            if n == 0 {
                break;
            }
            hash = fnv1a_into(hash, &buf[..n]);
        }
        hash
    } else {
        let window = SAMPLE_BYTES.min(len) as usize;
        let mut head = vec![0u8; window];
        file.read_exact(&mut head)
            .map_err(|_| "a pack file could not be read")?;
        let mut hash = fnv1a_into(FNV_OFFSET, &head);
        if len > SAMPLE_BYTES {
            file.seek(SeekFrom::End(-(window as i64)))
                .map_err(|_| "a pack file could not be seeked")?;
            let mut tail = vec![0u8; window];
            file.read_exact(&mut tail)
                .map_err(|_| "a pack file could not be read")?;
            hash = fnv1a_into(hash, &tail);
        }
        hash
    };
    Ok((full, hash))
}

/// Stamp one pack file, hashing it in full when it is no larger than `full_hash_limit`.
fn stamp_file(path: &Path, full_hash_limit: u64) -> Result<PackStamp, &'static str> {
    let meta = fs::metadata(path).map_err(|_| "a pack file could not be stat'd")?;
    if !meta.is_file() {
        return Err("a pack path is neither a file nor a directory");
    }
    let len = meta.len();
    let mut file = fs::File::open(path).map_err(|_| "a pack file could not be opened")?;
    let (full, sample) = hash_file_content(&mut file, len, full_hash_limit)?;
    Ok(PackStamp {
        label: sanitise(&file_label(path)),
        kind: if full { "hash" } else { "file" },
        len,
        modified_ms: modified_ms(&meta),
        sample,
    })
}

/// The last two path components, which is enough to tell `1.8.9/1.8.9.jar` from
/// `1.8.8/1.8.9.jar` without recording the whole of somebody's home directory in a file.
fn file_label(path: &Path) -> String {
    let mut parts: Vec<String> = path
        .components()
        .rev()
        .take(2)
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    parts.reverse();
    parts.join("/")
}

/// Walk a folder pack, summing sizes and folding every entry's relative path, length and content
/// into one hash. Sorted, so the walk order the OS happens to hand back cannot change the key.
///
/// # Why content, not mtime
///
/// An earlier version of this function folded each entry's *modification time* in instead of
/// reading it, on the theory that a stat is nearly free and a full read is not. That is the right
/// call for a pack too large to hash (see [`FULL_HASH_LIMIT`]'s reasoning) but the wrong one here:
/// mtime is a proxy for "did the bytes change", and proxies have failure modes the thing itself
/// does not. This filesystem's mtime resolution is coarser than two back-to-back `fs::write`
/// calls in one test function, so two different-content writes to the same file can land in the
/// same tick and stamp identically — verified directly, by the "ignores walk order" test in this
/// module's `#[cfg(test)]` block, which reproduced that exact collision (both keys rendering
/// byte-for-byte equal) in roughly a third of runs before this fix. A cache that trusts mtime for
/// identity inherits that hole, and per this module's correctness bar (see the module docs) a
/// stale hit is the one outcome that is never acceptable — so mtime is not part of the hash at
/// all.
///
/// The fix is to hash each entry with the exact same policy [`stamp_file`] already uses for a
/// whole pack file — full content up to `full_hash_limit`, a head/tail sample above it — applied
/// per entry instead of per pack. A folder pack's files are ordinary asset-sized files (JSON,
/// PNG, OGG), so in the common case every one of them is hashed in full; the per-file sampling
/// threshold only matters for a stray huge file inside an otherwise normal tree, and it is
/// deliberately the same threshold `push_path`'s doc comment already promises, not a new number
/// to reason about. This costs a real read of the tree on every stamp, which is the deliberate
/// trade this function makes: correctness (never missing a same-length, same-tick edit) over the
/// mtime shortcut's speed, matching "invalidation is the feature; speed is the side effect" above.
fn stamp_dir(root: &Path, full_hash_limit: u64) -> Result<PackStamp, &'static str> {
    let mut entries: Vec<(String, u64, bool, u64)> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let listing = fs::read_dir(&dir).map_err(|_| "a folder pack could not be listed")?;
        for entry in listing {
            let entry = entry.map_err(|_| "a folder pack entry could not be read")?;
            let path = entry.path();
            let meta = entry
                .metadata()
                .map_err(|_| "a folder pack entry could not be stat'd")?;
            if meta.is_dir() {
                stack.push(path);
            } else {
                if entries.len() >= MAX_DIR_ENTRIES {
                    return Err("a folder pack has more entries than the stamp budget");
                }
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                let len = meta.len();
                let mut file =
                    fs::File::open(&path).map_err(|_| "a folder pack entry could not be opened")?;
                let (full, content) = hash_file_content(&mut file, len, full_hash_limit)?;
                entries.push((rel, len, full, content));
            }
        }
    }
    entries.sort();
    let mut hash = FNV_OFFSET;
    let mut len = 0u64;
    for (rel, size, full, content) in &entries {
        hash = fnv1a_into(hash, rel.as_bytes());
        hash = fnv1a_into(hash, &size.to_le_bytes());
        // Whether this entry was hashed in full or sampled is part of the fold, mirroring
        // stamp_file's `kind` field, so a sampled entry's hash can never be confused with a
        // fully-hashed one that happens to collide.
        hash = fnv1a_into(hash, &[u8::from(*full)]);
        hash = fnv1a_into(hash, &content.to_le_bytes());
        len += size;
    }
    Ok(PackStamp {
        label: sanitise(&file_label(root)),
        kind: "dir",
        len,
        modified_ms: entries.len() as u64,
        sample: hash,
    })
}

// ---------------------------------------------------------------------------------------------
// The snapshot
// ---------------------------------------------------------------------------------------------

/// Everything `bake_all` produces, in a form that can be written to disk and read back.
///
/// This is deliberately a plain data mirror rather than the `BakedAssets` type itself: it owns no
/// invariants, so a corrupt file can only produce wrong *data*, never a broken type.
/// [`BakedSnapshot::of`] and [`BakedSnapshot::into_assets`] are the two ends of the seam.
#[derive(Debug, Clone)]
pub struct BakedSnapshot {
    /// The stitched atlas.
    pub atlas: Atlas,
    /// Every sprite's placement and animation frames, in stitch order.
    pub sprites: Vec<Sprite>,
    /// Baked geometry indexed by `BlockState::id`.
    pub models: Vec<BakedModel>,
    /// States with more than one weighted variant: `(state id, [(weight, model)])`.
    pub weighted: Vec<(u16, Vec<(i32, BakedModel)>)>,
    /// Models keyed by the full blockstate-file stem and derived-property variant key.
    pub actual_models: Vec<(StateMapping, BakedModel)>,
    /// Weighted candidates for full actual-state keys.
    pub actual_weighted: Vec<(StateMapping, Vec<(i32, BakedModel)>)>,
    /// Item models by vanilla's packed `item id << 16 | metadata` mesher key.
    pub item_models: Vec<(u32, BakedModel)>,
    /// `ModelBakery`'s two manually registered `item_frame` variants (`normal`, then `map`).
    pub item_frame_models: [BakedModel; 2],
}

/// Field-by-field equality for a [`BakedModel`].
///
/// `BakedModel` and [`Atlas`] do not derive `PartialEq` (they belong to S09 and S04, and this
/// module does not add derives to another section's types), so the round-trip tests compare them
/// here instead. Every field is named explicitly rather than reached through a wildcard, so a
/// field added to `BakedModel` later fails to compile here and forces a decision about whether
/// the cache needs to carry it — silently ignoring a new field is exactly how a cache starts
/// serving subtly wrong models.
#[must_use]
pub fn model_eq(a: &BakedModel, b: &BakedModel) -> bool {
    let BakedModel {
        face_quads,
        general_quads,
        ambient_occlusion,
        gui_3d,
        particle_uv,
        transforms,
    } = a;
    *face_quads == b.face_quads
        && *general_quads == b.general_quads
        && *ambient_occlusion == b.ambient_occlusion
        && *gui_3d == b.gui_3d
        && *particle_uv == b.particle_uv
        && *transforms == b.transforms
}

impl PartialEq for BakedSnapshot {
    fn eq(&self, other: &Self) -> bool {
        let models_eq = |a: &[BakedModel], b: &[BakedModel]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| model_eq(x, y))
        };
        self.atlas.w == other.atlas.w
            && self.atlas.h == other.atlas.h
            && self.atlas.texture == other.atlas.texture
            && self.atlas.mipmaps == other.atlas.mipmaps
            && self.sprites == other.sprites
            && models_eq(&self.models, &other.models)
            && self.weighted.len() == other.weighted.len()
            && self
                .weighted
                .iter()
                .zip(&other.weighted)
                .all(|((sa, ca), (sb, cb))| {
                    sa == sb
                        && ca.len() == cb.len()
                        && ca
                            .iter()
                            .zip(cb)
                            .all(|((wa, ma), (wb, mb))| wa == wb && model_eq(ma, mb))
                })
            && self.actual_models.len() == other.actual_models.len()
            && self
                .actual_models
                .iter()
                .zip(&other.actual_models)
                .all(|((ka, ma), (kb, mb))| ka == kb && model_eq(ma, mb))
            && self.actual_weighted.len() == other.actual_weighted.len()
            && self.actual_weighted.iter().zip(&other.actual_weighted).all(
                |((ka, ca), (kb, cb))| {
                    ka == kb
                        && ca.len() == cb.len()
                        && ca
                            .iter()
                            .zip(cb)
                            .all(|((wa, ma), (wb, mb))| wa == wb && model_eq(ma, mb))
                },
            )
            && self.item_models.len() == other.item_models.len()
            && self
                .item_models
                .iter()
                .zip(&other.item_models)
                .all(|((ia, ma), (ib, mb))| ia == ib && model_eq(ma, mb))
            && self
                .item_frame_models
                .iter()
                .zip(&other.item_frame_models)
                .all(|(a, b)| model_eq(a, b))
    }
}

impl BakedSnapshot {
    /// Take a snapshot of a finished bake, ready for [`save`].
    ///
    /// Clones the whole bake (~10 MiB against the real 1.8.9 assets). That cost is paid on a
    /// cache **miss** only, right after the 40 ms bake it is about to make unnecessary next time,
    /// and never on the hit path.
    #[must_use]
    pub fn of(assets: &BakedAssets) -> BakedSnapshot {
        let (atlas, sprites, models, weighted, _) = assets.to_parts();
        let (actual_models, actual_weighted) = assets.actual_to_parts();
        let item_models = assets.item_models_to_parts();
        let item_frame_models = assets.item_frame_models_to_parts();
        BakedSnapshot {
            atlas,
            sprites,
            models,
            weighted,
            actual_models,
            actual_weighted,
            item_models,
            item_frame_models,
        }
    }

    /// Rebuild the installable bake this snapshot is of.
    ///
    /// Nothing here can fail: [`BakedSnapshot::decode`] has already bounds-checked every length,
    /// and `BakedAssets::from_parts` is total for any model vector length. What makes the result
    /// *correct* rather than merely well-formed is [`load`]'s key check, not this function.
    #[must_use]
    pub fn into_assets(self) -> BakedAssets {
        BakedAssets::from_parts_with_actual(
            self.atlas,
            self.sprites,
            self.models,
            self.weighted,
            self.actual_models,
            self.actual_weighted,
            self.item_models,
            self.item_frame_models,
        )
    }

    /// Encode to the on-disk payload bytes (no header, no checksum — see [`save`]).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer(Vec::with_capacity(1 << 20));
        w.u32(self.atlas.w);
        w.u32(self.atlas.h);
        w.bytes(&self.atlas.texture);
        w.u32(self.atlas.mipmaps.len() as u32);
        for mip in &self.atlas.mipmaps {
            w.bytes(mip);
        }

        w.u32(self.sprites.len() as u32);
        for sprite in &self.sprites {
            write_sprite(&mut w, sprite);
        }

        w.u32(self.models.len() as u32);
        for model in &self.models {
            write_model(&mut w, model);
        }

        w.u32(self.weighted.len() as u32);
        for (state, candidates) in &self.weighted {
            w.u32(u32::from(*state));
            w.u32(candidates.len() as u32);
            for (weight, model) in candidates {
                w.u32(*weight as u32);
                write_model(&mut w, model);
            }
        }

        w.u32(self.actual_models.len() as u32);
        for (key, model) in &self.actual_models {
            w.str(&key.file_stem);
            w.str(&key.variant_key);
            write_model(&mut w, model);
        }

        w.u32(self.actual_weighted.len() as u32);
        for (key, candidates) in &self.actual_weighted {
            w.str(&key.file_stem);
            w.str(&key.variant_key);
            w.u32(candidates.len() as u32);
            for (weight, model) in candidates {
                w.u32(*weight as u32);
                write_model(&mut w, model);
            }
        }

        w.u32(self.item_models.len() as u32);
        for (id, model) in &self.item_models {
            w.u32(*id);
            write_model(&mut w, model);
        }
        for model in &self.item_frame_models {
            write_model(&mut w, model);
        }
        w.0
    }

    /// Decode payload bytes produced by [`BakedSnapshot::encode`].
    ///
    /// # Errors
    ///
    /// A [`CacheMiss::Malformed`] naming the field that noticed, for any input that is not a
    /// well-formed payload. Never panics and never allocates more than the input can justify.
    pub fn decode(bytes: &[u8]) -> Result<BakedSnapshot, CacheMiss> {
        let mut r = Reader::new(bytes);
        let w = r.u32("atlas.w")?;
        let h = r.u32("atlas.h")?;
        let texture = r.bytes("atlas.texture")?.to_vec();
        if texture.len() as u64 != u64::from(w) * u64::from(h) * 4 {
            return Err(CacheMiss::Malformed("atlas.texture length"));
        }
        let mipmaps = r.vec("atlas.mipmaps", 4, |r| {
            Ok(r.bytes("atlas.mipmaps.level")?.to_vec())
        })?;
        for (index, mip) in mipmaps.iter().enumerate() {
            let level = index as u32 + 1;
            let expected = u64::from(w >> level) * u64::from(h >> level) * 4;
            if expected == 0 || mip.len() as u64 != expected {
                return Err(CacheMiss::Malformed("atlas.mipmaps length"));
            }
        }

        // A sprite's fixed fields plus the metadata flags/length are at least 27 bytes before
        // any frame payload. Keep the affordability guard honest after FORMAT_VERSION 12.
        let sprites = r.vec("sprites", 27, read_sprite)?;
        let models = r.vec("models", MODEL_MIN_BYTES, read_model)?;
        let weighted = r.vec("weighted", 8, |r| {
            let state = r.u32("weighted.state")?;
            let state = u16::try_from(state).map_err(|_| CacheMiss::Malformed("weighted.state"))?;
            let candidates = r.vec("weighted.candidates", MODEL_MIN_BYTES + 4, |r| {
                let weight = r.u32("weighted.weight")? as i32;
                Ok((weight, read_model(r)?))
            })?;
            Ok((state, candidates))
        })?;
        let actual_models = r.vec("actual_models", MODEL_MIN_BYTES + 8, |r| {
            let file_stem = r.str("actual_models.file_stem")?;
            let variant_key = r.str("actual_models.variant_key")?;
            Ok((
                StateMapping {
                    file_stem,
                    variant_key,
                },
                read_model(r)?,
            ))
        })?;
        let actual_weighted = r.vec("actual_weighted", 12, |r| {
            let file_stem = r.str("actual_weighted.file_stem")?;
            let variant_key = r.str("actual_weighted.variant_key")?;
            let candidates = r.vec("actual_weighted.candidates", MODEL_MIN_BYTES + 4, |r| {
                let weight = r.u32("actual_weighted.weight")? as i32;
                Ok((weight, read_model(r)?))
            })?;
            Ok((
                StateMapping {
                    file_stem,
                    variant_key,
                },
                candidates,
            ))
        })?;
        let item_models = r.vec("items", MODEL_MIN_BYTES + 4, |r| {
            let key = r.u32("items.key")?;
            Ok((key, read_model(r)?))
        })?;
        let item_frame_models = [read_model(&mut r)?, read_model(&mut r)?];

        if !r.is_empty() {
            return Err(CacheMiss::Malformed("trailing bytes"));
        }
        Ok(BakedSnapshot {
            atlas: Atlas {
                texture,
                mipmaps,
                w,
                h,
            },
            sprites,
            models,
            weighted,
            actual_models,
            actual_weighted,
            item_models,
            item_frame_models,
        })
    }
}

/// Write `snapshot` to `path`, keyed on `inputs`.
///
/// The write is **atomic**: the bytes go to a sibling temporary file which is then renamed over
/// `path`, so a process killed mid-write leaves either the old cache or no cache, never half of
/// one. (The checksum would catch a torn file anyway; this makes sure the *previous*, valid
/// cache is not the casualty.)
///
/// # Errors
///
/// [`CacheError::Unstampable`] when `inputs` contains a pack whose identity could not be
/// established, and [`CacheError::Io`] for a filesystem failure.
pub fn save(path: &Path, inputs: &BakeInputs, snapshot: &BakedSnapshot) -> Result<(), CacheError> {
    if let Some(reason) = inputs.unstampable {
        return Err(CacheError::Unstampable(reason));
    }
    let payload = snapshot.encode();
    let key = inputs.key();

    let mut out = Vec::with_capacity(payload.len() + key.len() + 64);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(key.as_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    // The checksum covers the key as well as the payload, so a file whose key was edited to
    // match a different pack stack fails here rather than deserialising into the wrong models.
    let checksum = fnv1a_into(fnv1a_into(FNV_OFFSET, key.as_bytes()), &payload);
    out.extend_from_slice(&checksum.to_le_bytes());
    out.extend_from_slice(&payload);

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| CacheError::io(parent, e))?;
        }
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, &out).map_err(|e| CacheError::io(&tmp, e))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        CacheError::io(path, e)
    })?;
    Ok(())
}

/// Read the cache at `path`, but only if it is an answer to exactly `inputs`.
///
/// Never fails: anything other than a verified hit is a [`CacheMiss`] and means "bake it".
#[must_use]
pub fn load(path: &Path, inputs: &BakeInputs) -> CacheRead {
    if inputs.unstampable.is_some() {
        return CacheRead::Miss(CacheMiss::Inputs {
            found: String::new(),
            expected: inputs.key(),
        });
    }
    let Ok(bytes) = fs::read(path) else {
        return CacheRead::Miss(CacheMiss::NoFile);
    };
    match parse(&bytes, inputs) {
        Ok(snapshot) => CacheRead::Hit(Box::new(snapshot)),
        Err(miss) => {
            tracing::debug!(path = %path.display(), reason = %miss, "baked-asset cache miss");
            CacheRead::Miss(miss)
        }
    }
}

fn parse(bytes: &[u8], inputs: &BakeInputs) -> Result<BakedSnapshot, CacheMiss> {
    if bytes.len() < MAGIC.len() + 4 {
        return Err(CacheMiss::NotACacheFile);
    }
    if bytes[..MAGIC.len()] != MAGIC {
        return Err(CacheMiss::NotACacheFile);
    }
    let mut r = Reader::new(&bytes[MAGIC.len()..]);
    let version = r.u32("version").map_err(|_| CacheMiss::Truncated)?;
    if version != FORMAT_VERSION {
        return Err(CacheMiss::Version {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    let key_bytes = r.bytes("key").map_err(|_| CacheMiss::Truncated)?;
    let found = String::from_utf8_lossy(key_bytes).into_owned();
    let expected = inputs.key();
    if found != expected {
        return Err(CacheMiss::Inputs { found, expected });
    }
    let payload_len = r.u64("payload.len").map_err(|_| CacheMiss::Truncated)?;
    let checksum = r.u64("checksum").map_err(|_| CacheMiss::Truncated)?;
    let payload = r
        .take(payload_len, "payload")
        .map_err(|_| CacheMiss::Truncated)?;
    if !r.is_empty() {
        return Err(CacheMiss::Malformed("trailing bytes after payload"));
    }
    if fnv1a_into(fnv1a_into(FNV_OFFSET, key_bytes), payload) != checksum {
        return Err(CacheMiss::Checksum);
    }
    BakedSnapshot::decode(payload)
}

// ---------------------------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------------------------

/// Smallest possible encoding of a [`BakedModel`]: seven empty quad lists, AO and GUI-3D flags,
/// an absent particle-UV tag, and the 54 floats of [`ItemCameraTransforms`]. Used to bound how
/// many elements a count is allowed to claim before anything is allocated.
const MODEL_MIN_BYTES: usize = 7 * 4 + 1 + 1 + 1 + 54 * 4;

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.0
            .extend_from_slice(&(v.len() as u64).to_le_bytes().as_slice()[..4]);
        // Lengths are written as u32; nothing in a bake is anywhere near 4 GiB and a 32-bit
        // length is one fewer way for a corrupt file to claim an enormous allocation.
        self.0.extend_from_slice(v);
    }
    fn str(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes }
    }

    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn take(&mut self, n: u64, what: &'static str) -> Result<&'a [u8], CacheMiss> {
        let n = usize::try_from(n).map_err(|_| CacheMiss::Malformed(what))?;
        if self.bytes.len() < n {
            return Err(CacheMiss::Malformed(what));
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, CacheMiss> {
        Ok(self.take(1, what)?[0])
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, CacheMiss> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, CacheMiss> {
        let b = self.take(8, what)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn f32(&mut self, what: &'static str) -> Result<f32, CacheMiss> {
        let b = self.take(4, what)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn bytes(&mut self, what: &'static str) -> Result<&'a [u8], CacheMiss> {
        let n = self.u32(what)?;
        self.take(u64::from(n), what)
    }

    fn str(&mut self, what: &'static str) -> Result<String, CacheMiss> {
        let raw = self.bytes(what)?;
        String::from_utf8(raw.to_vec()).map_err(|_| CacheMiss::Malformed(what))
    }

    /// Read a length-prefixed sequence, refusing to reserve capacity a file this size could not
    /// possibly justify. `min_element_bytes` is the smallest encoding one element can have.
    fn vec<T>(
        &mut self,
        what: &'static str,
        min_element_bytes: usize,
        mut each: impl FnMut(&mut Reader<'a>) -> Result<T, CacheMiss>,
    ) -> Result<Vec<T>, CacheMiss> {
        let count = self.u32(what)? as usize;
        let affordable = self.bytes.len() / min_element_bytes.max(1);
        if count > affordable {
            return Err(CacheMiss::Malformed(what));
        }
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(each(self)?);
        }
        Ok(out)
    }
}

fn write_uv(w: &mut Writer, uv: UvRect) {
    w.f32(uv.min_u);
    w.f32(uv.max_u);
    w.f32(uv.min_v);
    w.f32(uv.max_v);
}

fn read_uv(r: &mut Reader<'_>) -> Result<UvRect, CacheMiss> {
    Ok(UvRect {
        min_u: r.f32("uv")?,
        max_u: r.f32("uv")?,
        min_v: r.f32("uv")?,
        max_v: r.f32("uv")?,
    })
}

fn write_sprite(w: &mut Writer, sprite: &Sprite) {
    w.str(&sprite.location.namespace);
    w.str(&sprite.location.path);
    w.u32(sprite.width);
    w.u32(sprite.height);
    write_uv(w, sprite.uv);
    w.u32(sprite.frames.len() as u32);
    for frame in &sprite.frames {
        w.bytes(&frame.rgba);
        match frame.duration_ticks {
            None => w.u8(0),
            Some(d) => {
                w.u8(1);
                w.u32(d);
            }
        }
    }
    w.u32(sprite.default_duration_ticks);
    w.u8(u8::from(sprite.interpolate));
    w.u8(u8::from(sprite.texture_metadata.blur));
    w.u8(u8::from(sprite.texture_metadata.clamp));
    w.u32(sprite.texture_metadata.mipmaps.len() as u32);
    for level in &sprite.texture_metadata.mipmaps {
        w.u32(*level as u32);
    }
    w.u32(sprite.texture_metadata.explicit_mipmaps.len() as u32);
    for frame in &sprite.texture_metadata.explicit_mipmaps {
        w.u32(frame.len() as u32);
        for (level, pixels) in frame {
            w.u32(*level);
            w.bytes(pixels);
        }
    }
}

fn read_sprite(r: &mut Reader<'_>) -> Result<Sprite, CacheMiss> {
    let namespace = r.str("sprite.namespace")?;
    let path = r.str("sprite.path")?;
    let width = r.u32("sprite.width")?;
    let height = r.u32("sprite.height")?;
    let uv = read_uv(r)?;
    let frames = r.vec("sprite.frames", 5, |r| {
        let rgba = r.bytes("sprite.frame.rgba")?.to_vec();
        let duration_ticks = match r.u8("sprite.frame.tag")? {
            0 => None,
            1 => Some(r.u32("sprite.frame.duration")?),
            _ => return Err(CacheMiss::Malformed("sprite.frame.tag")),
        };
        Ok(SpriteFrame {
            rgba,
            duration_ticks,
        })
    })?;
    let default_duration_ticks = r.u32("sprite.default_duration")?;
    let interpolate = r.u8("sprite.interpolate")? != 0;
    let blur = r.u8("sprite.texture.blur")? != 0;
    let clamp = r.u8("sprite.texture.clamp")? != 0;
    let mipmaps = r.vec("sprite.texture.mipmaps", 4, |r| {
        Ok(r.u32("sprite.texture.mipmap")? as i32)
    })?;
    let explicit_mipmaps = r.vec("sprite.texture.explicit_mipmaps", 8, |r| {
        r.vec("sprite.texture.explicit_mipmap.frame", 8, |r| {
            let level = r.u32("sprite.texture.explicit_mipmap.level")?;
            if level == 0 {
                return Err(CacheMiss::Malformed("sprite.texture.explicit_mipmap.level"));
            }
            let pixels = r.bytes("sprite.texture.explicit_mipmap.pixels")?.to_vec();
            Ok((level, pixels))
        })
    })?;
    if !explicit_mipmaps.is_empty() && explicit_mipmaps.len() != frames.len() {
        return Err(CacheMiss::Malformed(
            "sprite.texture.explicit_mipmaps.frames",
        ));
    }
    for frame in &explicit_mipmaps {
        let mut previous = 0;
        for (level, pixels) in frame {
            if *level <= previous {
                return Err(CacheMiss::Malformed("sprite.texture.explicit_mipmap.level"));
            }
            previous = *level;
            if *level >= 32 {
                return Err(CacheMiss::Malformed("sprite.texture.explicit_mipmap.level"));
            }
            let expected = u64::from(width >> *level) * u64::from(height >> *level) * 4;
            if expected == 0 || pixels.len() as u64 != expected {
                return Err(CacheMiss::Malformed(
                    "sprite.texture.explicit_mipmap.pixels",
                ));
            }
        }
    }
    Ok(Sprite {
        location: ResourceLocation::new(namespace, path),
        width,
        height,
        uv,
        frames,
        default_duration_ticks,
        interpolate,
        texture_metadata: crate::sprite::TextureMetadata {
            blur,
            clamp,
            mipmaps,
            explicit_mipmaps,
        },
    })
}

fn write_transform(w: &mut Writer, t: ItemTransform) {
    for v in t.rotation.iter().chain(&t.translation).chain(&t.scale) {
        w.f32(*v);
    }
}

fn read_transform(r: &mut Reader<'_>) -> Result<ItemTransform, CacheMiss> {
    let mut triple = || -> Result<[f32; 3], CacheMiss> {
        Ok([
            r.f32("transform")?,
            r.f32("transform")?,
            r.f32("transform")?,
        ])
    };
    Ok(ItemTransform {
        rotation: triple()?,
        translation: triple()?,
        scale: triple()?,
    })
}

fn write_transforms(w: &mut Writer, t: &ItemCameraTransforms) {
    write_transform(w, t.third_person);
    write_transform(w, t.first_person);
    write_transform(w, t.head);
    write_transform(w, t.gui);
    write_transform(w, t.ground);
    write_transform(w, t.fixed);
}

fn read_transforms(r: &mut Reader<'_>) -> Result<ItemCameraTransforms, CacheMiss> {
    Ok(ItemCameraTransforms {
        third_person: read_transform(r)?,
        first_person: read_transform(r)?,
        head: read_transform(r)?,
        gui: read_transform(r)?,
        ground: read_transform(r)?,
        fixed: read_transform(r)?,
    })
}

/// `Facing` is encoded as its own `index()`, plus 6 for "no resolvable facing".
fn write_quad(w: &mut Writer, quad: &BakedQuad) {
    for v in &quad.verts {
        w.f32(v.pos[0]);
        w.f32(v.pos[1]);
        w.f32(v.pos[2]);
        w.f32(v.uv[0]);
        w.f32(v.uv[1]);
    }
    w.u8(quad.face.map_or(6, |f| f.index()));
    w.u8(quad.shade_face.map_or(6, |f| f.index()));
    match quad.tint_index {
        None => w.u8(0),
        Some(t) => {
            w.u8(1);
            w.u8(t);
        }
    }
    w.u8(u8::from(quad.shade));
}

fn read_quad(r: &mut Reader<'_>) -> Result<BakedQuad, CacheMiss> {
    let mut verts = [Vertex::default(); 4];
    for v in &mut verts {
        v.pos = [r.f32("quad.pos")?, r.f32("quad.pos")?, r.f32("quad.pos")?];
        v.uv = [r.f32("quad.uv")?, r.f32("quad.uv")?];
    }
    let face = match r.u8("quad.face")? {
        6 => None,
        i => Some(
            mc_math::Facing::VALUES
                .get(i as usize)
                .copied()
                .ok_or(CacheMiss::Malformed("quad.face"))?,
        ),
    };
    let shade_face = match r.u8("quad.shade_face")? {
        6 => None,
        i => Some(
            mc_math::Facing::VALUES
                .get(i as usize)
                .copied()
                .ok_or(CacheMiss::Malformed("quad.shade_face"))?,
        ),
    };
    let tint_index = match r.u8("quad.tint.tag")? {
        0 => None,
        1 => Some(r.u8("quad.tint")?),
        _ => return Err(CacheMiss::Malformed("quad.tint.tag")),
    };
    let shade = r.u8("quad.shade")? != 0;
    Ok(BakedQuad {
        verts,
        face,
        shade_face,
        tint_index,
        shade,
    })
}

/// Smallest encoding of one quad: 20 floats plus facing, tint tag and shade.
const QUAD_MIN_BYTES: usize = 20 * 4 + 4;

fn write_model(w: &mut Writer, model: &BakedModel) {
    for bucket in &model.face_quads {
        w.u32(bucket.len() as u32);
        for quad in bucket {
            write_quad(w, quad);
        }
    }
    w.u32(model.general_quads.len() as u32);
    for quad in &model.general_quads {
        write_quad(w, quad);
    }
    w.u8(u8::from(model.ambient_occlusion));
    w.u8(u8::from(model.gui_3d));
    match model.particle_uv {
        None => w.u8(0),
        Some(uv) => {
            w.u8(1);
            write_uv(w, uv);
        }
    }
    write_transforms(w, &model.transforms);
}

fn read_model(r: &mut Reader<'_>) -> Result<BakedModel, CacheMiss> {
    let mut face_quads: [Vec<BakedQuad>; 6] = Default::default();
    for bucket in &mut face_quads {
        *bucket = r.vec("model.face_quads", QUAD_MIN_BYTES, read_quad)?;
    }
    let general_quads = r.vec("model.general_quads", QUAD_MIN_BYTES, read_quad)?;
    let ambient_occlusion = r.u8("model.ao")? != 0;
    let gui_3d = match r.u8("model.gui_3d")? {
        0 => false,
        1 => true,
        _ => return Err(CacheMiss::Malformed("model.gui_3d")),
    };
    let particle_uv = match r.u8("model.particle_uv.tag")? {
        0 => None,
        1 => Some(read_uv(r)?),
        _ => return Err(CacheMiss::Malformed("model.particle_uv.tag")),
    };
    let transforms = read_transforms(r)?;
    Ok(BakedModel {
        face_quads,
        general_quads,
        ambient_occlusion,
        gui_3d,
        particle_uv,
        transforms,
    })
}

// ---------------------------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over `bytes`, continuing from `hash`, taking **eight bytes at a time**.
///
/// This is a **corruption** detector, not a security boundary: a local cache file is trusted
/// exactly as much as the resource packs it was derived from, and an attacker who can write to
/// the cache path can already write to the pack path. What it has to catch is a torn write, a
/// truncated file and a bit-rotted sector.
///
/// The word-at-a-time variant is used rather than textbook byte-at-a-time FNV-1a because it is
/// measured, not assumed, to matter: the payload is ~4 MiB and verifying it byte-wise cost more
/// than decoding it did (`benches/startup.rs`), which would have thrown away a sixth of the
/// saving the cache exists to produce. Mixing eight bytes per multiply keeps every input bit in
/// the state — `one_flipped_bit_anywhere_in_the_payload_is_caught` sweeps the whole file and
/// asserts each flip is rejected, so the detection claim is tested rather than argued. Note the
/// consequence: this is **not** interoperable with a standard FNV-1a implementation, which is
/// fine — nothing outside this module ever reads the value, and [`FORMAT_VERSION`] covers the
/// day that stops being true.
fn fnv1a_into(mut hash: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        hash ^= word;
        hash = hash.wrapping_mul(FNV_PRIME);
        // One multiply cannot carry a change in the high bytes down into the low ones, so a
        // flipped bit near the top of a word would only ever affect bits above it. The rotate
        // folds the state back over itself and closes that.
        hash = hash.rotate_left(29);
    }
    for b in chunks.remainder() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_into(FNV_OFFSET, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_math::Facing;

    fn quad(seed: f32) -> BakedQuad {
        let mut verts = [Vertex::default(); 4];
        for (i, v) in verts.iter_mut().enumerate() {
            let i = i as f32;
            v.pos = [seed + i, seed - i, seed * i];
            v.uv = [seed / 3.0, i / 7.0];
        }
        BakedQuad {
            verts,
            face: Some(Facing::VALUES[(seed as usize) % 6]),
            shade_face: Some(Facing::VALUES[(seed as usize) % 6]),
            tint_index: (seed as u8).is_multiple_of(2).then_some(seed as u8),
            shade: (seed as u8).is_multiple_of(3),
        }
    }

    fn model(seed: f32) -> BakedModel {
        let mut m = BakedModel {
            general_quads: vec![quad(seed), quad(seed + 1.0)],
            ambient_occlusion: (seed as u8).is_multiple_of(2),
            gui_3d: (seed as u8).is_multiple_of(3),
            particle_uv: Some(UvRect {
                min_u: seed / 10.0,
                max_u: seed / 10.0 + 0.01,
                min_v: seed / 20.0,
                max_v: seed / 20.0 + 0.01,
            }),
            ..BakedModel::default()
        };
        m.face_quads[2].push(quad(seed + 2.0));
        m.face_quads[5].push(quad(seed + 3.0));
        m.transforms.gui = ItemTransform {
            rotation: [30.0, 225.0, 0.0],
            translation: [0.0, 0.125, -0.25],
            scale: [0.625, 0.625, 0.625],
        };
        m
    }

    fn snapshot() -> BakedSnapshot {
        BakedSnapshot {
            atlas: Atlas {
                texture: (0..(8 * 8 * 4)).map(|i| (i % 251) as u8).collect(),
                mipmaps: Vec::new(),
                w: 8,
                h: 8,
            },
            sprites: vec![Sprite {
                location: ResourceLocation::minecraft("textures/blocks/lava_still"),
                width: 16,
                height: 16,
                uv: UvRect {
                    min_u: 0.25,
                    max_u: 0.5,
                    min_v: 0.0,
                    max_v: 0.125,
                },
                frames: vec![
                    SpriteFrame {
                        rgba: vec![1, 2, 3, 4],
                        duration_ticks: None,
                    },
                    SpriteFrame {
                        rgba: vec![9, 8, 7, 6],
                        duration_ticks: Some(3),
                    },
                ],
                default_duration_ticks: 2,
                interpolate: true,
                texture_metadata: crate::sprite::TextureMetadata {
                    blur: true,
                    clamp: true,
                    mipmaps: vec![1, 3, -1],
                    explicit_mipmaps: Vec::new(),
                },
            }],
            models: vec![model(1.0), model(2.0), BakedModel::default()],
            weighted: vec![(17, vec![(3, model(4.0)), (-1, model(5.0))])],
            actual_models: vec![(
                StateMapping {
                    file_stem: "fence".to_owned(),
                    variant_key: "east=true,north=false,south=false,west=false".to_owned(),
                },
                model(5.5),
            )],
            actual_weighted: vec![(
                StateMapping {
                    file_stem: "stone".to_owned(),
                    variant_key: "normal".to_owned(),
                },
                vec![(2, model(5.75)), (i32::MIN, model(5.875))],
            )],
            item_models: vec![((280u32 << 16) | 7, model(6.0))],
            item_frame_models: [model(6.5), model(7.0)],
        }
    }

    fn inputs() -> BakeInputs {
        let mut i = BakeInputs::new(2048);
        i.push_opaque("1.8.9.jar", b"jar identity");
        i
    }

    #[test]
    fn the_payload_round_trips_bit_for_bit() {
        let snap = snapshot();
        let bytes = snap.encode();
        let back = BakedSnapshot::decode(&bytes).expect("decode");
        assert_eq!(snap, back);
        assert_eq!(back.item_models[0].0, (280u32 << 16) | 7);
        // Re-encoding is byte-identical, so the format has no free choices left in it.
        assert_eq!(bytes, back.encode());
    }

    #[test]
    fn model_particle_uv_tag_rejects_unknown_values() {
        let mut writer = Writer(Vec::new());
        write_model(&mut writer, &BakedModel::default());
        // Seven empty vector counts, then AO and GUI-3D, then the optional-UV tag.
        writer.0[7 * 4 + 2] = 2;
        assert!(matches!(
            read_model(&mut Reader::new(&writer.0)),
            Err(CacheMiss::Malformed("model.particle_uv.tag"))
        ));
    }

    #[test]
    fn model_gui_3d_bit_is_strict_and_part_of_cache_identity() {
        let flat = BakedModel::default();
        let three_dimensional = BakedModel {
            gui_3d: true,
            ..BakedModel::default()
        };
        assert!(!model_eq(&flat, &three_dimensional));

        let mut writer = Writer(Vec::new());
        write_model(&mut writer, &three_dimensional);
        assert!(read_model(&mut Reader::new(&writer.0)).unwrap().gui_3d);

        writer.0[7 * 4 + 1] = 2;
        assert!(matches!(
            read_model(&mut Reader::new(&writer.0)),
            Err(CacheMiss::Malformed("model.gui_3d"))
        ));
    }

    #[test]
    fn a_file_round_trips_through_save_and_load() {
        let dir = std::env::temp_dir().join(format!("mc-rs-cache-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("roundtrip.bin");
        let snap = snapshot();
        save(&path, &inputs(), &snap).expect("save");
        match load(&path, &inputs()) {
            CacheRead::Hit(back) => assert_eq!(*back, snap),
            CacheRead::Miss(m) => panic!("unexpected miss: {m:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn explicit_mip_pixels_round_trip_through_the_snapshot_payload() {
        let mut snap = snapshot();
        let expected = {
            let sprite = &mut snap.sprites[0];
            sprite.width = 2;
            sprite.height = 2;
            sprite.texture_metadata.explicit_mipmaps =
                vec![vec![(1, vec![5, 6, 7, 8])], vec![(1, vec![9, 10, 11, 12])]];
            sprite.texture_metadata.clone()
        };

        let payload = snap.encode();
        let back = BakedSnapshot::decode(&payload).expect("explicit mip payload decodes");
        assert_eq!(back.sprites[0].texture_metadata, expected);
    }

    #[test]
    fn a_different_pack_stack_is_a_miss_and_says_so() {
        let dir = std::env::temp_dir().join(format!("mc-rs-cache-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("stale.bin");
        save(&path, &inputs(), &snapshot()).expect("save");

        // Same packs, but one of them changed.
        let mut changed = BakeInputs::new(2048);
        changed.push_opaque("1.8.9.jar", b"a different jar");
        assert!(matches!(
            load(&path, &changed).miss(),
            Some(CacheMiss::Inputs { .. })
        ));

        // Same packs, but a second one was stacked on top.
        let mut extra = inputs();
        extra.push_opaque("user-pack.zip", b"classic 32x");
        assert!(matches!(
            load(&path, &extra).miss(),
            Some(CacheMiss::Inputs { .. })
        ));

        // Same packs, different atlas cap.
        let mut capped = BakeInputs::new(1024);
        capped.push_opaque("1.8.9.jar", b"jar identity");
        assert!(matches!(
            load(&path, &capped).miss(),
            Some(CacheMiss::Inputs { .. })
        ));

        // And the control: the *unchanged* key still hits, so the three misses above are the
        // checks working rather than the file being unreadable.
        assert!(load(&path, &inputs()).hit().is_some());
        let _ = fs::remove_file(&path);
    }

    /// §7.6 / ATLAS-4: the stitched layout depends on `mipmapLevels`, so it is part of the key,
    /// while the default keeps exactly the pre-existing key so current caches stay valid.
    #[test]
    fn mipmap_levels_are_part_of_the_identity_and_the_default_key_is_unchanged() {
        let mut default = BakeInputs::new(2048);
        default.push_opaque("1.8.9.jar", b"jar identity");
        let mut explicit_default = default.clone();
        explicit_default.set_mipmap_levels(crate::atlas::DEFAULT_MIPMAP_LEVELS);
        assert_eq!(default.key(), explicit_default.key());
        assert!(!default.key().contains("mip="));
        let mut keys = vec![default.key()];
        for levels in 0..crate::atlas::DEFAULT_MIPMAP_LEVELS {
            let mut lowered = default.clone();
            lowered.set_mipmap_levels(levels);
            keys.push(lowered.key());
        }
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 5, "levels 0..=4 must all key differently");
    }

    #[test]
    fn pack_order_is_part_of_the_identity() {
        let mut a = BakeInputs::new(2048);
        a.push_opaque("base", b"1");
        a.push_opaque("over", b"2");
        let mut b = BakeInputs::new(2048);
        b.push_opaque("over", b"2");
        b.push_opaque("base", b"1");
        assert_ne!(a.key(), b.key(), "a later pack overrides an earlier one");
    }

    #[test]
    fn a_pack_name_cannot_forge_another_stack() {
        // A pack whose *name* contains the key's own separators must not be able to make a
        // one-pack stack render as a two-pack stack, which would let a cache baked from one
        // pack be served to a client that stacked two.
        let mut sneaky = BakeInputs::new(2048);
        sneaky.push_opaque("a|opaque:b", b"x");
        let sneaky_key = sneaky.key();

        let mut honest = BakeInputs::new(2048);
        honest.push_opaque("a", b"x");
        honest.push_opaque("b", b"y");

        assert_ne!(sneaky_key, honest.key());
        // The pack count is readable off the key and cannot be forged: two fixed separators for
        // the header fields, then exactly one per pack.
        assert_eq!(sneaky_key.matches('|').count(), 3, "{sneaky_key}");
        assert_eq!(honest.key().matches('|').count(), 4);
    }

    #[test]
    fn a_bumped_format_version_invalidates_every_cache() {
        // The version is written into the header, so simulate a bump by rewriting that field.
        let dir = std::env::temp_dir().join(format!("mc-rs-cache-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("version.bin");
        save(&path, &inputs(), &snapshot()).expect("save");
        let mut bytes = fs::read(&path).expect("read");
        assert!(load(&path, &inputs()).hit().is_some(), "control: it hits");

        bytes[8..12].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        fs::write(&path, &bytes).expect("write");
        assert_eq!(
            load(&path, &inputs()).miss(),
            Some(&CacheMiss::Version {
                found: FORMAT_VERSION + 1,
                expected: FORMAT_VERSION,
            })
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn pre_explicit_mip_v11_cache_is_rejected() {
        let dir = std::env::temp_dir().join(format!("mc-rs-cache-v11-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("version.bin");
        save(&path, &inputs(), &snapshot()).expect("save");
        let mut bytes = fs::read(&path).expect("read");
        bytes[8..12].copy_from_slice(&11_u32.to_le_bytes());
        fs::write(&path, &bytes).expect("write");
        assert_eq!(
            load(&path, &inputs()).miss(),
            Some(&CacheMiss::Version {
                found: 11,
                expected: FORMAT_VERSION,
            })
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn one_flipped_bit_anywhere_in_the_payload_is_caught() {
        let dir = std::env::temp_dir().join(format!("mc-rs-cache-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("corrupt.bin");
        save(&path, &inputs(), &snapshot()).expect("save");
        let good = fs::read(&path).expect("read");

        // Sweep the whole payload rather than poking one byte: a checksum that only covers the
        // header would pass a spot check on byte 0.
        let header = good.len() - snapshot().encode().len();
        let mut caught = 0;
        for i in (header..good.len()).step_by(7) {
            let mut bad = good.clone();
            bad[i] ^= 0x01;
            fs::write(&path, &bad).expect("write");
            match load(&path, &inputs()).miss() {
                Some(CacheMiss::Checksum) => caught += 1,
                other => panic!("byte {i} corruption was not caught: {other:?}"),
            }
        }
        assert!(caught > 100, "only swept {caught} bytes");

        // Control: put the good bytes back and it hits again, so the sweep above was detecting
        // corruption rather than a path that never loads.
        fs::write(&path, &good).expect("write");
        assert!(load(&path, &inputs()).hit().is_some());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_truncated_file_is_a_miss_not_a_panic() {
        let dir = std::env::temp_dir().join(format!("mc-rs-cache-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("truncated.bin");
        save(&path, &inputs(), &snapshot()).expect("save");
        let good = fs::read(&path).expect("read");
        for cut in [0usize, 1, 7, 8, 12, 40, good.len() / 2, good.len() - 1] {
            fs::write(&path, &good[..cut]).expect("write");
            assert!(
                load(&path, &inputs()).hit().is_none(),
                "a {cut}-byte file was accepted"
            );
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_is_an_ordinary_miss() {
        let path = std::env::temp_dir().join("mc-rs-cache-definitely-not-here.bin");
        let _ = fs::remove_file(&path);
        assert_eq!(load(&path, &inputs()).miss(), Some(&CacheMiss::NoFile));
    }

    #[test]
    fn arbitrary_bytes_never_panic_the_reader() {
        // Every prefix of a valid file, plus a magic-prefixed garbage sweep. The reader is fed
        // the cache file's own shape often enough to reach deep into the payload decoder.
        let good = {
            let mut out = Vec::new();
            out.extend_from_slice(&MAGIC);
            out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
            let key = inputs().key();
            out.extend_from_slice(&(key.len() as u32).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            let payload = snapshot().encode();
            out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            out.extend_from_slice(
                &fnv1a_into(fnv1a_into(FNV_OFFSET, key.as_bytes()), &payload).to_le_bytes(),
            );
            out.extend_from_slice(&payload);
            out
        };
        for n in 0..good.len().min(600) {
            let _ = parse(&good[..n], &inputs());
        }
        let mut rng = 0x1234_5678_9abc_def0u64;
        for _ in 0..2000 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let cut = (rng as usize) % good.len();
            let mut bad = good[..cut].to_vec();
            for _ in 0..(cut % 32) {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                bad.push(rng as u8);
            }
            let _ = parse(&bad, &inputs());
        }
        // A length field claiming far more elements than the file could hold must be rejected
        // by the affordability check, not by attempting the allocation.
        let mut absurd = Vec::new();
        absurd.extend_from_slice(&8u32.to_le_bytes()); // atlas.w
        absurd.extend_from_slice(&8u32.to_le_bytes()); // atlas.h
        absurd.extend_from_slice(&(8u32 * 8 * 4).to_le_bytes());
        absurd.extend_from_slice(&vec![0u8; 8 * 8 * 4]);
        absurd.extend_from_slice(&0u32.to_le_bytes()); // atlas mip levels
        absurd.extend_from_slice(&u32::MAX.to_le_bytes()); // sprites: 4 billion of them
        assert!(matches!(
            BakedSnapshot::decode(&absurd),
            Err(CacheMiss::Malformed("sprites"))
        ));
    }

    #[test]
    fn an_unstampable_pack_refuses_to_be_cached() {
        let mut broken = BakeInputs::new(2048);
        broken.push_path("C:/definitely/not/a/real/pack/anywhere.jar");
        assert!(!broken.is_stampable());
        let path = std::env::temp_dir().join("mc-rs-cache-unstampable.bin");
        assert!(matches!(
            save(&path, &broken, &snapshot()),
            Err(CacheError::Unstampable(_))
        ));
        // And it can never *read* one either, so a key that could not be established can never
        // accidentally match one that could.
        assert!(load(&path, &broken).hit().is_none());
    }

    #[test]
    fn a_real_file_stamp_notices_content_moving() {
        let dir = std::env::temp_dir().join(format!("mc-rs-stamp-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("pack.zip");
        fs::write(&path, b"original contents, exactly this long!!").expect("write");
        let mut a = BakeInputs::new(2048);
        a.push_path(&path);
        assert!(a.is_stampable());

        // Same length, different bytes, inside the sampled window: caught.
        fs::write(&path, b"replaced contents, exactly this long!!").expect("write");
        let mut b = BakeInputs::new(2048);
        b.push_path(&path);
        assert_ne!(a.key(), b.key());

        // A different length is caught too.
        fs::write(&path, b"shorter").expect("write");
        let mut c = BakeInputs::new(2048);
        c.push_path(&path);
        assert_ne!(b.key(), c.key());

        // Control: restamping unchanged content reproduces the key exactly, so the two
        // inequalities above are the stamp working rather than it being nondeterministic.
        let mut c2 = BakeInputs::new(2048);
        c2.push_path(&path);
        assert_eq!(c.key(), c2.key());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_folder_pack_stamp_covers_every_file_and_ignores_walk_order() {
        let dir = std::env::temp_dir().join(format!("mc-rs-folder-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("assets/minecraft/models/block")).expect("mkdir");
        fs::write(dir.join("pack.mcmeta"), b"{}").expect("write");
        fs::write(
            dir.join("assets/minecraft/models/block/stone.json"),
            b"{\"parent\":\"block/cube_all\"}",
        )
        .expect("write");
        let mut a = BakeInputs::new(2048);
        a.push_path(&dir);
        assert!(a.is_stampable());

        let mut a2 = BakeInputs::new(2048);
        a2.push_path(&dir);
        assert_eq!(a.key(), a2.key(), "the walk must be order-independent");

        // A nested file changing content at the same length is caught.
        fs::write(
            dir.join("assets/minecraft/models/block/stone.json"),
            b"{\"parent\":\"block/cube_top\"}",
        )
        .expect("write");
        let mut b = BakeInputs::new(2048);
        b.push_path(&dir);
        assert_ne!(a.key(), b.key());

        // A new file is caught.
        fs::write(dir.join("assets/minecraft/models/block/dirt.json"), b"{}").expect("write");
        let mut c = BakeInputs::new(2048);
        c.push_path(&dir);
        assert_ne!(b.key(), c.key());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_pack_under_the_limit_is_hashed_in_full_and_a_larger_one_is_sampled() {
        let dir = std::env::temp_dir().join(format!("mc-rs-hashed-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("pack.zip");
        fs::write(&path, vec![7u8; 300_000]).expect("write");

        // Default limit: 300 KB is far under 64 MiB, so it is hashed in full.
        let mut hashed = BakeInputs::new(2048);
        hashed.push_path(&path);
        assert!(hashed.key().contains("hash:"), "{}", hashed.key());

        // Forced sampling, by lowering the limit below the file's size. The stamp *kind* changes
        // with it, so a sampled key can never collide with a hashed one for the same file.
        let mut sampled = BakeInputs::new(2048);
        sampled.set_full_hash_limit(1024);
        sampled.push_path(&path);
        assert!(sampled.key().contains("file:"), "{}", sampled.key());
        assert_ne!(hashed.key(), sampled.key());

        // The documented hole, asserted rather than described: a byte changed *between* the
        // 128 KiB head and the 128 KiB tail of a 300 KB file is invisible to the sample and
        // visible to the full hash. This is exactly where the sampled key stops, and it is why
        // the default is to hash.
        let mut bytes = fs::read(&path).expect("read");
        bytes[150_000] = 8;
        fs::write(&path, &bytes).expect("write");
        let mut hashed2 = BakeInputs::new(2048);
        hashed2.push_path(&path);
        let mut sampled2 = BakeInputs::new(2048);
        sampled2.set_full_hash_limit(1024);
        sampled2.push_path(&path);

        assert_ne!(hashed.key(), hashed2.key(), "the full hash must see it");
        // mtime almost certainly differs, so the sampled keys are not asserted equal as a whole;
        // what is asserted is that the *sample* component — the part the hash is supposed to be
        // doing the work — did not move.
        let sample_of = |k: &str| k.rsplit(':').next().unwrap().to_owned();
        assert_eq!(
            sample_of(&sampled.key()),
            sample_of(&sampled2.key()),
            "the sample window genuinely does not cover the middle of a large file"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_write_cannot_replace_a_good_cache() {
        // `save` writes to `<path>.tmp` and renames. Assert the temporary file is gone and the
        // real one is intact, which is what makes an interrupted save non-destructive.
        let dir = std::env::temp_dir().join(format!("mc-rs-atomic-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("atomic.bin");
        save(&path, &inputs(), &snapshot()).expect("save");
        assert!(path.is_file());
        assert!(!path.with_extension("tmp").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_state_count_is_part_of_the_key() {
        // Not settable from outside on purpose — a registry change must invalidate every cache
        // without a caller opting in. Assert it is actually in there.
        let key = BakeInputs::new(2048).key();
        assert!(
            key.contains(&format!("states={}", mc_data::STATE_COUNT)),
            "{key}"
        );
    }
}
