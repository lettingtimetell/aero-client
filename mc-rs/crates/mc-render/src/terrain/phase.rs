//! The GPU side: pipelines, per-section buffers, and the [`RenderPhase`] implementations.
//!
//! Vanilla draws terrain as four passes over the same visible-chunk list, in a fixed order
//! interleaved with entities and particles (`EntityRenderer.renderWorldPass`). Opaque layers
//! submit that list near-to-far; `TRANSLUCENT` submits it in reverse (far-to-near). S10's
//! [`PhaseOrder`] reserves the same slots, so terrain registers **four** phases sharing one
//! [`TerrainRenderer`]:
//!
//! | order | layer | vanilla state |
//! |---|---|---|
//! | `TERRAIN_SOLID` (200) | `SOLID` | opaque, depth write, no blend |
//! | `TERRAIN_CUTOUT` (250) | `CUTOUT_MIPPED` then `CUTOUT` | alpha test at 0.5, depth write |
//! | `TERRAIN_TRANSLUCENT` (500) | `TRANSLUCENT` | `glDepthMask(false)`, src-alpha blend, sorted |
//!
//! The solid phase runs first each frame, so it is where the per-frame work lives: the frustum
//! is rebuilt, the chunk-graph walk runs, and the per-chunk camera-relative offsets are
//! uploaded. The other three phases reuse that result.
//
// java: client/renderer/RenderGlobal.java renderBlockLayer/renderBlockLayer0,
//       client/renderer/EntityRenderer.java renderWorldPass (the state for each layer)

use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

use mc_assets::{Atlas, AtlasAnimationUpload};
use mc_data::RenderLayer;
use mc_math::{BlockPos, ChunkPos, ChunkSectionPos, DVec3};
use wgpu::util::{DeviceExt, DrawIndexedIndirectArgs};

use super::cull::{SectionSource, SectionWalk, VisibleSection};
use super::frustum::Frustum;
use super::jobs::{
    next_mesh_revision, GpuBudget, GpuMemoryReport, GpuMemoryWindow, GpuResidency, MeshRevision,
    MeshTaskResult, UploadPlan,
};
use super::mesher::ChunkMesh;
use super::sort::{needs_resort, sort_translucent};
use super::vertex::{
    pack_bytes, Buffer, PackedTerrainVertex, TerrainVertex, PACKED_VERTEX_SIZE, VERTEX_SIZE,
};
use super::visgraph::{CameraReach, SectionOcclusion, SetVisibility};
use super::walk_grid::{self, WalkGrid};
use crate::sky::fog::{FogDistance, FogMode, FogState};
use crate::sky::lightmap::{Lightmap, LightmapParams};
use crate::{Camera, DepthAttachment, Gpu, PhaseList, PhaseOrder, RenderPhase};

// Anisotropic Filtering: a child module so it
// can reuse this file's private vertex-offset layout and layer helpers.
#[path = "aniso.rs"]
pub mod aniso;
// Shader packs' terrain variant: a child module
// for the same reason as `aniso`.
#[path = "pack.rs"]
pub mod pack;

/// Build-time knobs for [`TerrainRenderer::new_with`]. Additive; [`TerrainRenderer::new`] uses
/// [`TerrainOptions::default`], which is vanilla.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainOptions {
    /// Back-face culling. Vanilla runs terrain with `glEnable(GL_CULL_FACE)` and
    /// `glCullFace(GL_BACK)` (java: `Minecraft.startGame`'s `cullFace(1029)`, plus
    /// `EntityRenderer.renderWorldPass`' `enableCull()` around every terrain layer), so the
    /// default is `true`.
    ///
    /// This exists as a knob **only** so `s11_cull.rs` can render the same scene both ways and
    /// assert the frames are identical — the direct evidence that S09's baked winding is
    /// vanilla's. Turning it off is not a supported configuration, just a test instrument.
    pub cull_backfaces: bool,
    /// Draw through the old, lossy 20-byte `Unorm16` layout rather than vanilla's 28-byte float
    /// `DefaultVertexFormats.BLOCK` layout, packing each upload on the way to the GPU.
    ///
    /// The same kind of knob as [`cull_backfaces`](TerrainOptions::cull_backfaces), and for the
    /// same reason: `s23_batching.rs` renders one scene both ways in one process and asserts the
    /// framebuffers are byte-identical. A sound A/B wants an interleaved same-process
    /// control and a whole-format change cannot be interleaved, so the control is to keep the
    /// format T1.1 replaced alive and runnable.
    ///
    /// **Not a supported configuration.** It destroys sub-1/2048 position and sub-1/32768 UV
    /// detail, and defaults to `false`.
    pub packed_vertices: bool,
}

impl Default for TerrainOptions {
    fn default() -> Self {
        Self {
            cull_backfaces: true,
            packed_vertices: false,
        }
    }
}

/// What the F3 overlay wants to know about the last frame.
///
/// S23 added [`TerrainStats::vertex_buffer_binds`]. That is not purely additive — it would break
/// an *exhaustive* struct literal — so every construction site in the workspace was audited
/// first: there is exactly one, in this module, and nothing outside `mc-render` builds one.
/// `#[non_exhaustive]` was considered and rejected as the larger change: it would forbid external
/// construction and exhaustive matching forever, to protect a struct nobody constructs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TerrainStats {
    /// Sections with a mesh resident on the GPU.
    pub loaded_sections: usize,
    /// Sections with geometry the culling walk selected this frame: the numerator of vanilla's
    /// F3 `C:` (`RenderGlobal.getDebugInfoRenders` skips `DUMMY` and empty compiled chunks).
    /// All-air records the walk passed through are not counted.
    pub visible_sections: usize,
    /// Draw calls issued this frame, across all four layers.
    pub draw_calls: usize,
    /// Logical section draw items submitted this frame. With native indirect submission this can
    /// exceed [`TerrainStats::draw_calls`], which counts command-recording API calls.
    pub draw_items: usize,
    /// Quads submitted this frame, across all four layers.
    pub quads: usize,
    /// `set_vertex_buffer` calls issued this frame.
    ///
    /// Added by S23. Before the batched upload this was exactly `2 × draw_calls` — every draw
    /// rebound both the section's own vertex buffer and a 16-byte window of the chunk-offset
    /// buffer. Sections that share a batched buffer now share one binding, and the offset buffer
    /// is bound once per layer and indexed with `first_instance`, so this drops to roughly
    /// `2 × (layers drawn)` when everything came from one upload. It is counted rather than
    /// asserted because "the draw loop got cheaper" is a claim that needs a number.
    pub vertex_buffer_binds: usize,
}

/// Whether terrain draws through native multi-draw on this adapter.
///
/// On when it was measured to help: NVIDIA under Vulkan, drawing from the vertex arena. Other
/// adapters keep the portable direct path until someone measures them there. `MC_RS_TERRAIN_MDI`
/// set to `1` or anything else overrides this either way, for measurements.
fn multi_draw_wanted(adapter: &wgpu::AdapterInfo) -> bool {
    const NVIDIA_VENDOR_ID: u32 = 0x10DE;
    match std::env::var_os("MC_RS_TERRAIN_MDI") {
        Some(value) => value == "1",
        None => adapter.backend == wgpu::Backend::Vulkan && adapter.vendor == NVIDIA_VENDOR_ID,
    }
}

/// Bytes in a newly opened terrain vertex arena page.
///
/// Large enough that a real world needs only a handful, because a bind and its resource tracking
/// are per buffer; a section is far smaller than this. Unused pages are not created.
const ARENA_PAGE_BYTES: u64 = 128 * 1024 * 1024;

/// The size page 0 may grow to before the arena opens a second page (phase 3b; see
/// [`VertexArena`]). Growing copies the page once, so it briefly holds the old and the new
/// buffer together; above this, extra pages trade a few multi-draw groups for that peak. Dense
/// terrain at render distance 16 needs about 440 MiB, so this still fits it in one page.
const ARENA_GROWN_PAGE_MAX_BYTES: u64 = 512 * 1024 * 1024;

/// The size page 0 opens at when it can grow (see
/// [`VertexArena::new_page_bytes`]): about three times a SkyWars map's terrain at render
/// distance 8 (11.4 MiB in the SkyWars bench). Denser worlds grow it by doubling, as they already
/// grew past [`ARENA_PAGE_BYTES`].
const ARENA_FIRST_PAGE_BYTES: u64 = 32 * 1024 * 1024;

/// Kill switch for sizing the per-section maps up front (see
/// [`TerrainRenderer::reserve_section_maps`]): `MC_NW_F5_SECTION_RESERVE=0` turns it off. Read
/// once.
fn section_reserve_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("MC_NW_F5_SECTION_RESERVE")
                .as_deref()
                .map(str::trim),
            Ok("0")
        )
    })
}

/// The render distance the section maps are sized for at most. Above it the
/// maps still grow on demand, as before; up to it a session never rehashes them while loading.
const SECTION_RESERVE_MAX_RENDER_DISTANCE: u32 = 16;

/// Sections in the render-distance square, `(2r + 1)^2 * 16`, with `r` capped at
/// [`SECTION_RESERVE_MAX_RENDER_DISTANCE`]: render distance 8 is 4624 (exactly what a SkyWars
/// world holds once loaded), 16 is 17424.
fn section_reserve_target(render_distance: u32) -> usize {
    let r = render_distance.clamp(2, SECTION_RESERVE_MAX_RENDER_DISTANCE) as usize;
    (2 * r + 1) * (2 * r + 1) * 16
}

/// Kill switch for removing a whole world's sections in bulk (see
/// [`TerrainRenderer::remove_sections`]): `MC_NW_F5_BULK_DROP=0` removes them one by one as
/// before. Read once.
fn bulk_drop_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("MC_NW_F5_BULK_DROP")
                .as_deref()
                .map(str::trim),
            Ok("0")
        )
    })
}

/// Kill switch for forgetting the mesh-revision tombstones when a world goes
/// away (see [`TerrainRenderer::remove_world_sections`]): `MC_W18_OC_PRUNE_REVISIONS=0` keeps
/// them, as before. Read once.
fn prune_revisions_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("MC_W18_OC_PRUNE_REVISIONS")
                .as_deref()
                .map(str::trim),
            Ok("0")
        )
    })
}

/// Kill switch for merging released arena ranges in one pass before the
/// next allocation instead of one sorted insert per release (see [`FreeRanges::release`]):
/// `MC_W18_OC_LAZY_RELEASE=0` inserts each one as it is released, as before. Read once.
fn lazy_release_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("MC_W18_OC_LAZY_RELEASE")
                .as_deref()
                .map(str::trim),
            Ok("0")
        )
    })
}

/// Which byte ranges of one arena page are unused. Separate from the buffer so it can be tested
/// without a GPU.
struct FreeRanges {
    capacity: u64,
    /// Unused `(offset, len)` ranges, sorted by offset and never adjacent.
    free: Vec<(u64, u64)>,
    /// Ranges released since `free` was last read, in release order.
    /// Merged into `free` (see [`FreeRanges::settle`]) before an allocation or a growth reads it.
    pending: Vec<(u64, u64)>,
    /// The merge's output buffer, kept so a merge never allocates once
    /// warm.
    scratch: Vec<(u64, u64)>,
    /// Bytes handed out and not released yet: the page is unused exactly
    /// when this is 0, with or without pending releases.
    used: u64,
    /// Whether releases wait in `pending` (the default) or are inserted
    /// into `free` one at a time (`MC_W18_OC_LAZY_RELEASE=0`).
    lazy: bool,
}

impl FreeRanges {
    /// A page in this process's release mode (tests; the arena passes its own).
    #[cfg(test)]
    fn new(capacity: u64) -> Self {
        Self::with_mode(capacity, lazy_release_enabled())
    }

    fn with_mode(capacity: u64, lazy: bool) -> Self {
        Self {
            capacity,
            free: vec![(0, capacity)],
            pending: Vec::new(),
            scratch: Vec::new(),
            used: 0,
            lazy,
        }
    }

    /// First fit. Every length is a whole number of quads, so every offset stays a whole number
    /// of vertices, which is what `base_vertex` needs.
    fn allocate(&mut self, len: u64) -> Option<u64> {
        self.settle();
        let index = self.free.iter().position(|&(_, free)| free >= len)?;
        let (offset, free) = self.free[index];
        if free == len {
            self.free.remove(index);
        } else {
            self.free[index] = (offset + len, free - len);
        }
        self.used += len;
        Some(offset)
    }

    /// Return a range. Inserting into the sorted list costs a move of
    /// everything after the insertion point; a world change or a far teleport releases thousands
    /// of ranges in one frame, in hash order, which fragments the list and made each insert
    /// O(n) (measured 87% of a one-by-one world change in these inserts). Now a
    /// release only records the range (and resets the page outright when it empties); the next
    /// allocation or growth merges every recorded range in one sort and one linear pass. The
    /// sorted, coalesced list describes a set of free bytes, and that set does not depend on the
    /// order the ranges came back in, so every later allocation gets the same offset as before;
    /// emptiness comes from `used`, so pages are dropped at the same releases as before.
    fn release(&mut self, offset: u64, len: u64) {
        debug_assert!(self.used >= len);
        self.used -= len;
        if !self.lazy {
            self.insert_free(offset, len);
        } else if self.used == 0 {
            // Every range is back: the merge would give exactly this.
            self.free.clear();
            self.free.push((0, self.capacity));
            self.pending.clear();
        } else {
            self.pending.push((offset, len));
        }
    }

    /// Merge the pending releases into `free`: one sort of the pending
    /// ranges and one linear merge of two sorted lists of disjoint ranges, coalescing touching
    /// neighbors, which is the list one sorted insert per release would have built.
    fn settle(&mut self) {
        match self.pending.len() {
            0 => {}
            1 => {
                let (offset, len) = self.pending[0];
                self.pending.clear();
                self.insert_free(offset, len);
            }
            _ => {
                self.pending.sort_unstable_by_key(|&(start, _)| start);
                let mut merged = std::mem::take(&mut self.scratch);
                merged.clear();
                merged.reserve(self.free.len() + self.pending.len());
                let (mut free, mut pending) = (0, 0);
                while free < self.free.len() || pending < self.pending.len() {
                    let next = if pending == self.pending.len()
                        || (free < self.free.len() && self.free[free].0 < self.pending[pending].0)
                    {
                        free += 1;
                        self.free[free - 1]
                    } else {
                        pending += 1;
                        self.pending[pending - 1]
                    };
                    match merged.last_mut() {
                        Some((start, len)) if *start + *len == next.0 => *len += next.1,
                        _ => merged.push(next),
                    }
                }
                self.pending.clear();
                self.scratch = std::mem::replace(&mut self.free, merged);
            }
        }
    }

    /// The pending releases merged in, without changing anything (tests compare pages by this).
    #[cfg(test)]
    fn settled(&self) -> Vec<(u64, u64)> {
        let mut copy = Self {
            capacity: self.capacity,
            free: self.free.clone(),
            pending: self.pending.clone(),
            scratch: Vec::new(),
            used: self.used,
            lazy: self.lazy,
        };
        copy.settle();
        copy.free
    }

    /// Insert one free range into the sorted list, merging it with touching neighbors.
    fn insert_free(&mut self, offset: u64, len: u64) {
        let index = self.free.partition_point(|&(start, _)| start < offset);
        self.free.insert(index, (offset, len));
        if let Some(&(next_start, next_len)) = self.free.get(index + 1) {
            if offset + len == next_start {
                self.free[index].1 += next_len;
                self.free.remove(index + 1);
            }
        }
        if index > 0 {
            let (previous_start, previous_len) = self.free[index - 1];
            if previous_start + previous_len == offset {
                self.free[index - 1].1 += self.free[index].1;
                self.free.remove(index);
            }
        }
    }

    fn is_unused(&self) -> bool {
        if self.lazy {
            // Exact: allocated and free bytes partition the page.
            self.used == 0
        } else {
            self.free == [(0, self.capacity)]
        }
    }

    /// Extend the range to `capacity` bytes; the new tail is free, merged with a free tail.
    fn grow(&mut self, capacity: u64) {
        debug_assert!(capacity > self.capacity);
        self.settle();
        let added = capacity - self.capacity;
        match self.free.last_mut() {
            Some((start, len)) if *start + *len == self.capacity => *len += added,
            _ => self.free.push((self.capacity, added)),
        }
        self.capacity = capacity;
    }
}

/// One large vertex buffer, carved into section-sized ranges.
struct ArenaPage {
    buffer: Rc<wgpu::Buffer>,
    ranges: FreeRanges,
}

/// Sub-allocates live section geometry from a few large vertex buffers.
///
/// A separate `wgpu::Buffer` per section layer means a bind per draw, and every buffer a pass
/// touches is tracked and validated individually by wgpu on each frame, which made recording the
/// largest cost of a real-terrain frame. Sharing buffers leaves a bind per page and lets
/// consecutive draws collapse into one multi-draw.
///
/// Reusing a range straight away is safe: `Queue::write_buffer` is ordered after earlier
/// submissions by a barrier, and removing a section invalidates the draw plans that referenced it.
///
/// Phase 3b: when every page is full, page 0 grows in place (into a buffer at least twice its
/// size, filled by one GPU copy) before a second page opens, up to
/// [`ARENA_GROWN_PAGE_MAX_BYTES`]. Multi-draw groups are maximal runs of same-page draws in walk
/// order, so with one page they stay whole whatever order the mesher finished sections in; with
/// several, placement decided how often consecutive draws switched pages.
struct VertexArena {
    pages: Vec<Option<ArenaPage>>,
    /// Size of a newly opened page ([`ARENA_PAGE_BYTES`]; smaller in tests).
    page_bytes: u64,
    /// The size page 0 grows to before another page opens ([`ARENA_GROWN_PAGE_MAX_BYTES`]).
    grow_limit: u64,
    /// The size page 0 opens at where it can grow, at most `page_bytes`
    /// ([`ARENA_FIRST_PAGE_BYTES`]; smaller in tests).
    first_page_bytes: u64,
    /// A growth allocation ran out of device memory once; later full arenas open pages of the
    /// usual size instead of retrying.
    growth_failed: bool,
    /// Growth is only worth its copy when native multi-draw groups same-page draws; the direct
    /// path draws every section on its own whatever page it is in. The renderer mirrors its
    /// multi-draw state here (a debug replay test drawing directly measured growth doubling the
    /// load-phase frame time for no draw-call gain).
    grow_enabled: bool,
    /// When opening a new page last failed (the GPU had no memory for it, or the
    /// device is lost). For [`ARENA_PAGE_RETRY`] after that, a section that fits no page waits
    /// in the ready queue instead of retrying a 128 MiB allocation every frame.
    page_failed_at: Option<Instant>,
    /// Set while every lease of this arena is being dropped at once
    /// ([`TerrainRenderer::remove_sections`]): releases are skipped and
    /// [`VertexArena::finish_bulk_release`] resets the pages afterwards.
    bulk_release: bool,
    /// The release mode of the pages this arena opens (see
    /// [`FreeRanges::release`]); `MC_W18_OC_LAZY_RELEASE=0` turns it off. Tests set both.
    lazy_release: bool,
}

/// How long the arena waits after a failed page before trying another.
const ARENA_PAGE_RETRY: std::time::Duration = std::time::Duration::from_secs(2);

impl Default for VertexArena {
    fn default() -> Self {
        Self {
            pages: Vec::new(),
            page_bytes: ARENA_PAGE_BYTES,
            grow_limit: ARENA_GROWN_PAGE_MAX_BYTES,
            first_page_bytes: ARENA_FIRST_PAGE_BYTES,
            growth_failed: false,
            grow_enabled: false,
            page_failed_at: None,
            bulk_release: false,
            lazy_release: lazy_release_enabled(),
        }
    }
}

/// Where [`VertexArena::allocate`] put a range.
struct ArenaAllocation {
    page: usize,
    buffer: Rc<wgpu::Buffer>,
    offset: u64,
    /// `(old, new)` when this allocation grew page 0: every slice still naming page 0's previous
    /// buffer must be pointed at its new one, and plans holding the old one dropped. Separate
    /// from `buffer`, which names a different page when the grown page still had no room.
    replaced: Option<(Rc<wgpu::Buffer>, Rc<wgpu::Buffer>)>,
}

/// A terrain arena buffer. `COPY_SRC` because page 0 is copied into its replacement when it
/// grows.
fn new_arena_buffer(gpu: &Gpu, size: u64) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain vertex arena"),
        size,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Copy `page` into a new `capacity`-byte buffer, make that the page's buffer and return the old
/// one. One immediate submission: every earlier `write_buffer` into the old buffer is applied
/// before it (queue writes run ahead of the next submission), and every later write targets the
/// new buffer. The old buffer lives until the copy has run.
///
/// `None`, with the page untouched, when the device is out of memory for the larger buffer: one
/// big allocation can fail where pages of the usual size still fit (low-VRAM GPUs), and wgpu's
/// default handler would otherwise turn that into a panic.
fn grow_page(gpu: &Gpu, page: &mut ArenaPage, capacity: u64) -> Option<Rc<wgpu::Buffer>> {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buffer = new_arena_buffer(gpu, capacity);
    if pollster::block_on(scope.pop()).is_some() {
        return None;
    }
    let buffer = Rc::new(buffer);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain vertex arena growth"),
        });
    encoder.copy_buffer_to_buffer(&page.buffer, 0, &buffer, 0, page.ranges.capacity);
    gpu.queue.submit([encoder.finish()]);
    page.ranges.grow(capacity);
    Some(std::mem::replace(&mut page.buffer, buffer))
}

/// [`VertexArena::allocate`] found no room and could not open a page. `replaced`
/// is page 0's `(old, new)` buffer pair when it grew on the way (see [`ArenaAllocation`]): the
/// caller must still repoint those slices.
struct ArenaFull {
    replaced: Option<(Rc<wgpu::Buffer>, Rc<wgpu::Buffer>)>,
}

impl VertexArena {
    /// The size of a page about to be opened. Where page 0 can grow
    /// (`may_grow`: native multi-draw, no failed growth) it starts at
    /// [`ARENA_FIRST_PAGE_BYTES`] and doubles as the world fills it (the phase 3b growth), so a
    /// SkyWars map (about 11 MiB of terrain) no longer holds a 128 MiB page; every other page,
    /// and every page without growth, is [`Self::page_bytes`] as before.
    /// `MC_W21_MR_SMALL_FIRST_PAGE=0` opens page 0 at the full size too.
    fn new_page_bytes(&self, opens_page_zero: bool, may_grow: bool) -> u64 {
        if opens_page_zero && may_grow && crate::w21_mr::small_first_page_enabled() {
            self.page_bytes.min(self.first_page_bytes)
        } else {
            self.page_bytes
        }
    }

    /// A range of `len` bytes, or [`ArenaFull`] when no page has room and a new page cannot be
    /// opened (the GPU is out of memory or the device is lost). The caller keeps
    /// the section for a later frame; this used to be a panic in wgpu's error handler.
    fn allocate(&mut self, gpu: &Gpu, len: u64) -> Result<ArenaAllocation, ArenaFull> {
        for (index, page) in self.pages.iter_mut().enumerate() {
            if let Some(page) = page {
                if let Some(offset) = page.ranges.allocate(len) {
                    return Ok(ArenaAllocation {
                        page: index,
                        buffer: Rc::clone(&page.buffer),
                        offset,
                        replaced: None,
                    });
                }
            }
        }
        // Every page is full. Nothing can be created on a lost device, and after
        // a failed page the next try waits a little (see `page_failed_at`).
        if gpu.is_device_lost()
            || self
                .page_failed_at
                .is_some_and(|at| at.elapsed() < ARENA_PAGE_RETRY)
        {
            return Err(ArenaFull { replaced: None });
        }
        let max_buffer = gpu.device.limits().max_buffer_size;
        let mut replaced = None;
        let may_grow = self.grow_enabled && !self.growth_failed;
        if let (true, Some(Some(first))) = (may_grow, self.pages.first_mut()) {
            let current = first.ranges.capacity;
            let target = current
                .saturating_mul(2)
                .max(current.saturating_add(len))
                .min(self.grow_limit.min(max_buffer));
            if target > current {
                replaced = grow_page(gpu, first, target).map(|old| (old, Rc::clone(&first.buffer)));
                self.growth_failed = replaced.is_none();
                if let Some(offset) = replaced.as_ref().and_then(|_| first.ranges.allocate(len)) {
                    return Ok(ArenaAllocation {
                        page: 0,
                        buffer: Rc::clone(&first.buffer),
                        offset,
                        replaced,
                    });
                }
            }
        }
        let opens_page_zero = self.pages.first().is_none_or(Option::is_none);
        let capacity = len.max(
            self.new_page_bytes(opens_page_zero, may_grow)
                .min(max_buffer),
        );
        // A new page is the one large terrain allocation of normal play: an
        // out-of-memory GPU (a 4 GB card at a high render distance) refuses it here instead of
        // panicking in wgpu's default error handler. Page 0 may have grown just above without
        // room for `len`; its old buffer still has to be repointed, so that pair goes back too.
        let buffer = match crate::gpu_health::try_create_buffer(
            &gpu.device,
            &wgpu::BufferDescriptor {
                label: Some("terrain vertex arena"),
                size: capacity,
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                // The first refusal at WARN with the allocator's numbers; the retries every
                // `ARENA_PAGE_RETRY` at DEBUG.
                if self.page_failed_at.is_none() {
                    tracing::warn!(
                        %error,
                        bytes = capacity,
                        memory = %crate::gpu_health::GpuMemory::of(&gpu.device)
                            .map_or_else(String::new, |m| m.to_string()),
                        "terrain: no GPU memory for another vertex page; far sections wait"
                    );
                } else {
                    tracing::debug!(%error, bytes = capacity, "terrain: vertex page refused again");
                }
                self.page_failed_at = Some(Instant::now());
                return Err(ArenaFull { replaced });
            }
        };
        self.page_failed_at = None;
        let mut page = ArenaPage {
            buffer: Rc::new(buffer),
            ranges: FreeRanges::with_mode(capacity, self.lazy_release),
        };
        let offset = page
            .ranges
            .allocate(len)
            .expect("a new page fits its first range");
        let buffer = Rc::clone(&page.buffer);
        let index = match self.pages.iter().position(Option::is_none) {
            Some(index) => {
                self.pages[index] = Some(page);
                index
            }
            None => {
                self.pages.push(Some(page));
                self.pages.len() - 1
            }
        };
        Ok(ArenaAllocation {
            page: index,
            buffer,
            offset,
            replaced,
        })
    }

    /// Every lease was dropped while `bulk_release` was set, so every page is
    /// unused. Leave the pages as releasing each range one by one would have: a grown page goes,
    /// and one page of the usual size stays for the next upload (`release` keeps the first page
    /// to empty and gives back any other that empties after it).
    fn finish_bulk_release(&mut self) {
        self.bulk_release = false;
        let mut kept = false;
        // A page is "grown" above the size its slot opens at: for page 0
        // where it can grow, that is the small first page, so a world change after a dense world
        // starts the next one small again (exactly `page_bytes` otherwise, as before).
        let may_grow = self.grow_enabled && !self.growth_failed;
        let (first_fresh, fresh) = (self.new_page_bytes(true, may_grow), self.page_bytes);
        for (index, slot) in self.pages.iter_mut().enumerate() {
            let Some(page) = slot.as_mut() else {
                continue;
            };
            let fresh = if index == 0 { first_fresh } else { fresh };
            if kept || page.ranges.capacity > fresh {
                *slot = None;
            } else {
                page.ranges = FreeRanges::with_mode(page.ranges.capacity, self.lazy_release);
                kept = true;
            }
        }
    }

    fn release(&mut self, page_index: usize, offset: u64, len: u64) {
        // The whole arena is being emptied; `finish_bulk_release` resets it.
        if self.bulk_release {
            return;
        }
        // The size this slot opens at (see `finish_bulk_release`).
        let fresh = self.new_page_bytes(page_index == 0, self.grow_enabled && !self.growth_failed);
        let Some(page) = self.pages[page_index].as_mut() else {
            return;
        };
        page.ranges.release(offset, len);
        // Keep one empty page for the next upload, and give any others back. A page grown past
        // the normal size goes back even when it is the only one (phase 3b review S3): after a
        // dense world, a lobby or SkyWars map must not keep holding a 512 MiB page.
        let grown = page.ranges.capacity > fresh;
        if page.ranges.is_unused()
            && (grown
                || self.pages.iter().enumerate().any(|(index, other)| {
                    index != page_index
                        && other.as_ref().is_some_and(|other| other.ranges.is_unused())
                }))
        {
            self.pages[page_index] = None;
        }
    }
}

/// A section layer's range of an arena page, returned when the section is removed or replaced.
struct ArenaLease {
    arena: Rc<RefCell<VertexArena>>,
    page: usize,
    offset: u64,
    len: u64,
}

impl Drop for ArenaLease {
    fn drop(&mut self) {
        self.arena
            .borrow_mut()
            .release(self.page, self.offset, self.len);
    }
}

/// One render layer's slice of a vertex buffer, which may be shared with other sections.
///
/// Before S23 every section owned one `wgpu::Buffer` per non-empty layer. [`UploadPlan`]
/// concatenates a whole batch of sections per layer into one buffer, so a section now names a
/// *range* inside a shared allocation. `offset` is always a multiple of
/// `wgpu::COPY_BUFFER_ALIGNMENT`, asserted at upload.
struct LayerSlice {
    buffer: Rc<wgpu::Buffer>,
    /// Byte offset of this section's first vertex within `buffer`.
    offset: u64,
    /// Length in bytes.
    len: u64,
    /// Quads in the run.
    quads: u32,
    /// Bytes per vertex in `buffer`: [`VERTEX_SIZE`] normally or [`PACKED_VERTEX_SIZE`] in the
    /// explicit precision-regression control.
    stride: u64,
    /// Present when this range belongs to the vertex arena rather than to its own buffer.
    _lease: Option<ArenaLease>,
}

struct DrawItem {
    buffer: Rc<wgpu::Buffer>,
    quads: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct IndirectGroup {
    buffer: Rc<wgpu::Buffer>,
    command_start: usize,
    command_count: usize,
}

/// Draw arguments cached for one layer while the visible section order and resident buffers
/// remain unchanged. The indirect command bytes are written only when this plan is rebuilt;
/// camera motion still updates the instance-offset buffer, which the commands index by slot.
struct LayerDrawPlan {
    visible_signature: u64,
    /// [`TerrainRenderer::layer_generations`] for this plan's layer when it was built.
    layer_generation: u64,
    use_indirect: bool,
    items: Vec<DrawItem>,
    groups: Vec<IndirectGroup>,
    quads: usize,
}

impl LayerSlice {
    /// The value to pass as `draw_indexed`'s `base_vertex`, so the whole batch buffer can stay
    /// bound across sections instead of being re-sliced per draw.
    fn base_vertex(&self) -> i32 {
        // `offset` is a whole number of vertices by construction (a quad is 4 × the stride),
        // and a batch is capped by `GpuBudget`, so this cannot overflow `i32` for any budget
        // that fits in GPU memory: i32::MAX vertices at 20 bytes is 40 GiB.
        (self.offset / self.stride) as i32
    }
}

/// One section's uploaded geometry.
struct ChunkBuffers {
    /// `None` for a layer with no geometry.
    layers: [Option<LayerSlice>; 4],
    visibility: SetVisibility,
    /// Compact mesh-time opacity data for the camera section's `getVisibleFacings` query.
    occlusion: SectionOcclusion,
    /// Kept so the translucent layer can be re-sorted without re-meshing.
    /// java: `CompiledChunk.setState` + `RenderChunk.resortTransparency`.
    translucent_cpu: Buffer,
}

impl ChunkBuffers {
    /// Bit `i` is set when [`layer_index`] `i` has geometry. An all-air record is `0`: it is
    /// resident only so the culling walk can read its visibility matrix, and it can never
    /// produce a draw. java: `CompiledChunk.isEmpty` / `isLayerEmpty`.
    fn layer_mask(&self) -> u8 {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| layer.is_some())
            .fold(0, |mask, (index, _)| mask | 1 << index)
    }

    /// The parts of this record each per-frame cache reads; see [`ResidentChange`].
    fn resident_key(&self) -> ResidentKey<'_> {
        ResidentKey {
            visibility: self.visibility,
            occlusion: &self.occlusion,
            layer_mask: self.layer_mask(),
        }
    }
}

/// What the culling walk and the draw plans can observe of one resident record.
#[derive(Clone, Copy, Debug)]
struct ResidentKey<'a> {
    visibility: SetVisibility,
    occlusion: &'a SectionOcclusion,
    layer_mask: u8,
}

/// Which per-frame caches one resident-set mutation (insert, replace, remove, evict) can
/// invalidate. #30: this used to be one global generation, so re-uploading an all-air record
/// over an identical all-air record threw away the walk, all four plans and the resort.
///
/// * The walk ([`SectionWalk`]) reads a record's presence and visibility matrix, and the camera
///   section's opacity snapshot ([`AuthoritativeSectionSource`]). Nothing else.
/// * A layer's [`LayerDrawPlan`] holds buffer ranges of the layer's resident slices. A replaced
///   record's ranges go back to the arena when it drops, so every layer the old *or* the new
///   record has must be rebuilt. A layer neither has cannot reference this section.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ResidentChange {
    walk: bool,
    /// Bit `i` is [`layer_index`] `i`.
    layers: u8,
}

impl ResidentChange {
    fn between(old: Option<ResidentKey<'_>>, new: Option<ResidentKey<'_>>) -> Self {
        match (old, new) {
            (None, None) => Self::default(),
            (Some(key), None) | (None, Some(key)) => Self {
                walk: true,
                layers: key.layer_mask,
            },
            (Some(old), Some(new)) => Self {
                walk: old.visibility != new.visibility || old.occlusion != new.occlusion,
                layers: old.layer_mask | new.layer_mask,
            },
        }
    }
}

/// A translucent resort is intentionally bounded. Sorting every resident section on a camera
/// move made the render thread do all of the CPU work in `RenderGlobal.renderBlockLayer`; the
/// live path only snapshots the visible prefix and sorts that snapshot on a worker.
pub const MAX_TRANSLUCENT_RESORT_SECTIONS: usize = 15;

#[derive(Debug)]
struct TranslucentResortJob {
    resident_version: u64,
    camera: DVec3,
    sections: Vec<(ChunkSectionPos, Buffer)>,
}

#[derive(Debug)]
struct TranslucentResortResult {
    resident_version: u64,
    sections: Vec<(ChunkSectionPos, Buffer)>,
}

/// Select only resident, visible translucent buffers for one bounded CPU job. Keeping this pure
/// makes the visible-only and 15-section cap independently testable without a GPU or a thread.
/// Takes the visible list by iterator so the live caller does not copy it first.
fn translucent_resort_job<'a>(
    visible: impl IntoIterator<Item = &'a VisibleSection>,
    chunks: &HashMap<ChunkSectionPos, ChunkBuffers>,
    resident_version: u64,
    camera: DVec3,
) -> Option<TranslucentResortJob> {
    let sections = visible
        .into_iter()
        .filter_map(|info| {
            let chunk = chunks.get(&info.pos)?;
            (!chunk.translucent_cpu.is_empty()).then(|| (info.pos, chunk.translucent_cpu.clone()))
        })
        .take(MAX_TRANSLUCENT_RESORT_SECTIONS)
        .collect::<Vec<_>>();
    (!sections.is_empty()).then_some(TranslucentResortJob {
        resident_version,
        camera,
        sections,
    })
}

fn run_translucent_resort_job(
    mut job: TranslucentResortJob,
    cancelled: &AtomicBool,
) -> Option<TranslucentResortResult> {
    for (pos, buffer) in &mut job.sections {
        if cancelled.load(Ordering::Relaxed) {
            return None;
        }
        sort_translucent(buffer, *pos, job.camera);
    }
    (!cancelled.load(Ordering::Relaxed)).then_some(TranslucentResortResult {
        resident_version: job.resident_version,
        sections: job.sections,
    })
}

#[must_use]
fn accepts_translucent_resort(current_version: u64, result_version: u64) -> bool {
    current_version == result_version
}

/// One resort handed to the persistent worker: the job, its cooperative cancellation flag and
/// the render thread's one-slot result channel.
struct TranslucentResortTask {
    job: TranslucentResortJob,
    cancelled: Arc<AtomicBool>,
    result: mpsc::SyncSender<TranslucentResortResult>,
}

impl TranslucentResortTask {
    fn run(self) {
        if let Some(result) = run_translucent_resort_job(self.job, &self.cancelled) {
            // The render thread may have dropped its receiver (cancellation); that is fine.
            let _ = self.result.send(result);
        }
    }
}

/// Run a resort on the long-lived "Terrain Resort" thread instead of a new OS thread per job.
///
/// java: `ChunkRenderDispatcher.updateTransparencyLater` hands resort tasks to its persistent
/// `ChunkRenderWorker`s; it never creates a thread per resort. A dedicated thread (rather than
/// the shared terrain mesh pool) keeps the old latency: a resort never waits behind up to
/// [`super::jobs::MAX_PENDING`] queued section builds. The worker owns only the task's cloned
/// CPU buffers and never sees a [`Gpu`], exactly like the detached thread it replaces. If a sort
/// ever panics, that task's result channel disconnects (which
/// [`TerrainRenderer::poll_translucent_resort`] already handles) and the worker goes on with the
/// next task; the panic runs inside `mc_core::panic_scope::contain`, so this
/// recovered failure is not recorded as the game's crash. Should the thread itself end, the next
/// submission restarts it.
fn submit_translucent_resort(task: TranslucentResortTask) {
    static WORKER: std::sync::Mutex<Option<mpsc::Sender<TranslucentResortTask>>> =
        std::sync::Mutex::new(None);
    let mut worker = WORKER
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let task = match worker.as_ref() {
        Some(sender) => match sender.send(task) {
            Ok(()) => return,
            Err(mpsc::SendError(task)) => task,
        },
        None => task,
    };
    let (sender, receiver) = mpsc::channel::<TranslucentResortTask>();
    let spawned = std::thread::Builder::new()
        .name("Terrain Resort".to_owned())
        .spawn(move || {
            for task in receiver {
                let _ = mc_core::panic_scope::contain(|| task.run());
            }
        });
    match spawned {
        Ok(_) => {
            // Cannot fail: the new thread owns the receiver until this sender drops.
            let _ = sender.send(task);
            *worker = Some(sender);
        }
        // Dropping the task drops its result sender, so the render thread observes a
        // disconnected job, exactly as it would after a worker panic.
        Err(_) => *worker = None,
    }
}

/// The resident section source used by the terrain walk. Keeping this as a real
/// [`SectionSource`] rather than a visibility-only closure is what carries the camera-cell
/// `VisGraph` result into `RenderGlobal.setupTerrain`'s special seed branch.
impl SectionSource for HashMap<ChunkSectionPos, ChunkBuffers> {
    fn visibility(&self, pos: ChunkSectionPos) -> Option<SetVisibility> {
        self.get(&pos).map(|chunk| chunk.visibility)
    }

    fn camera_visible_facings(
        &self,
        pos: ChunkSectionPos,
        local_x: i32,
        local_y: i32,
        local_z: i32,
    ) -> Option<Vec<mc_math::Facing>> {
        self.get(&pos).map(|chunk| {
            chunk
                .occlusion
                .visible_facings_from(local_x, local_y, local_z)
        })
    }

    fn camera_cell_opaque(
        &self,
        pos: ChunkSectionPos,
        local_x: i32,
        local_y: i32,
        local_z: i32,
    ) -> bool {
        self.get(&pos)
            .is_some_and(|chunk| chunk.occlusion.is_opaque(local_x, local_y, local_z))
    }
}

/// Section source used when the client supplies authoritative world-column membership.
///
/// A missing section in a loaded column is still an unmeshed/DUMMY hard stop. A missing
/// section in a column the world explicitly says is absent is settled all-air, which lets the
/// smart walk cross an unloaded corridor without retaining historical renderer tombstones.
/// Resident buffers always win so cache state cannot override world lifecycle state.
struct AuthoritativeSectionSource<'a> {
    chunks: &'a HashMap<ChunkSectionPos, ChunkBuffers>,
    world_columns: Option<&'a HashSet<ChunkPos>>,
}

impl SectionSource for AuthoritativeSectionSource<'_> {
    fn visibility(&self, pos: ChunkSectionPos) -> Option<SetVisibility> {
        // One definition of the rule, shared with the dense walk index that caches it.
        walk_grid::authoritative(
            self.chunks.get(&pos).map(|chunk| chunk.visibility),
            self.world_columns,
            pos,
        )
    }

    fn camera_visible_facings(
        &self,
        pos: ChunkSectionPos,
        local_x: i32,
        local_y: i32,
        local_z: i32,
    ) -> Option<Vec<mc_math::Facing>> {
        self.chunks.get(&pos).map(|chunk| {
            chunk
                .occlusion
                .visible_facings_from(local_x, local_y, local_z)
        })
    }

    fn camera_cell_opaque(
        &self,
        pos: ChunkSectionPos,
        local_x: i32,
        local_y: i32,
        local_z: i32,
    ) -> bool {
        self.chunks
            .get(&pos)
            .is_some_and(|chunk| chunk.occlusion.is_opaque(local_x, local_y, local_z))
    }
}

/// The last camera-cell answer (#34), keyed by everything `getVisibleFacings` is a function of
/// here: the camera section's retained mesh-time opacity bits and the eye's cell. Keying on
/// the bits themselves rather than on a revision counter means no upload path can forget to
/// invalidate it, and an unrelated upload elsewhere does not.
#[derive(Clone, Copy)]
struct CameraReachMemo {
    occlusion: SectionOcclusion,
    cell: usize,
    reach: CameraReach,
}

/// The live walk source: the per-section query from the dense [`WalkGrid`] (#7), everything
/// else from [`AuthoritativeSectionSource`], whose answers it must reproduce exactly. Debug
/// builds check each grid answer the walk reads against the hashed maps.
struct GridSectionSource<'a> {
    grid: &'a WalkGrid,
    inner: AuthoritativeSectionSource<'a>,
    camera_memo: &'a Cell<Option<CameraReachMemo>>,
}

impl SectionSource for GridSectionSource<'_> {
    fn visibility(&self, pos: ChunkSectionPos) -> Option<SetVisibility> {
        self.inner.visibility(pos)
    }

    #[inline]
    fn visibility_pairs(&self, pos: ChunkSectionPos) -> Option<u16> {
        let pairs = match self.grid.get(pos) {
            Some(cell) => walk_grid::decode(cell),
            // Outside the window: never on the live path (the walk stays within the render
            // distance of the camera column the window is centred on), but still exact.
            None => self.inner.visibility(pos).map(SetVisibility::pair_mask),
        };
        debug_assert_eq!(
            pairs,
            self.inner.visibility(pos).map(SetVisibility::pair_mask),
            "walk grid out of date at {pos:?}"
        );
        pairs
    }

    fn camera_visible_facings(
        &self,
        pos: ChunkSectionPos,
        local_x: i32,
        local_y: i32,
        local_z: i32,
    ) -> Option<Vec<mc_math::Facing>> {
        self.inner
            .camera_visible_facings(pos, local_x, local_y, local_z)
    }

    fn camera_reach(
        &self,
        pos: ChunkSectionPos,
        local_x: i32,
        local_y: i32,
        local_z: i32,
    ) -> Option<CameraReach> {
        // `None` exactly when `camera_visible_facings` is `None`: the section is not resident.
        let occlusion = &self.inner.chunks.get(&pos)?.occlusion;
        if occlusion.is_empty() {
            // All air, the usual void-map camera section: no fill and no memo needed.
            return Some(CameraReach::Many);
        }
        let cell = super::visgraph::index(local_x, local_y, local_z);
        if let Some(memo) = self.camera_memo.get() {
            if memo.cell == cell && memo.occlusion == *occlusion {
                return Some(memo.reach);
            }
        }
        let reach = occlusion.camera_reach(local_x, local_y, local_z);
        self.camera_memo.set(Some(CameraReachMemo {
            occlusion: *occlusion,
            cell,
            reach,
        }));
        Some(reach)
    }

    fn camera_cell_opaque(
        &self,
        pos: ChunkSectionPos,
        local_x: i32,
        local_y: i32,
        local_z: i32,
    ) -> bool {
        self.inner
            .camera_cell_opaque(pos, local_x, local_y, local_z)
    }
}

/// Fill one [`WalkGrid`] column from the renderer's maps.
fn fill_walk_column(
    chunks: &HashMap<ChunkSectionPos, ChunkBuffers>,
    world_columns: Option<&HashSet<ChunkPos>>,
    col: ChunkPos,
    cells: &mut [u16],
) {
    walk_grid::fill_column(col, cells, world_columns, |pos| {
        chunks.get(&pos).map(|chunk| chunk.visibility)
    });
}

const LAYER_COUNT: usize = 4;

/// java: `EntityRenderer.renderWorld` seeds `alphaFunc(GL_GREATER, 0.5)`. In
/// `renderWorldPass`, `SOLID` disables alpha, then `CUTOUT_MIPPED` and `CUTOUT` re-enable it
/// without changing that reference. Java resets the reference to 0.1 only after both cutout
/// layers, immediately before `TRANSLUCENT`.
fn layer_alpha_cutoff(layer_index: usize) -> f32 {
    match layer_index {
        0 => 0.0,
        1 | 2 => 0.5,
        3 => 0.1,
        _ => unreachable!("invalid terrain layer index: {layer_index}"),
    }
}

fn layer_index(layer: RenderLayer) -> usize {
    match layer {
        RenderLayer::Solid => 0,
        RenderLayer::CutoutMipped => 1,
        RenderLayer::Cutout => 2,
        RenderLayer::Translucent => 3,
    }
}

/// Return the visible-list slot for one draw position in a layer pass.
///
/// `RenderGlobal.renderBlockLayer` walks its BFS-produced list forward for the opaque layers,
/// but walks it backwards for `TRANSLUCENT` (`i1 = size - 1`, `j1 = -1`). The translucent pipeline
/// has depth writes disabled, so preserving that far-to-near submission order is observable when
/// translucent sections overlap. The index arithmetic keeps the hot draw loop allocation-free.
#[inline]
fn submission_index(draw_position: usize, visible_len: usize, layer: RenderLayer) -> usize {
    debug_assert!(draw_position < visible_len);
    if layer == RenderLayer::Translucent {
        visible_len - 1 - draw_position
    } else {
        draw_position
    }
}

/// Java switches the atlas to level zero only for the second cutout pass:
/// `CUTOUT_MIPPED` then `setBlurMipmap(false, false)` then `CUTOUT`.
fn layer_uses_mipmaps(layer_index: usize) -> bool {
    layer_index != 2
}

/// The post-task gate used by Java's `ChunkRenderDispatcher.runChunkUploads`.
///
/// Vanilla always runs the first queued upload, even when `finishTimeNano` has already passed;
/// only after that task does it compare the deadline, and an exact deadline is still accepted
/// (`finishTimeNano - System.nanoTime() < 0`). `None` models Java's `finishTimeNano == 0` shortcut,
/// which runs at most that one task, and is useful as a CPU-only control for this otherwise
/// GPU-facing scheduling seam.
#[derive(Debug, Clone, Copy)]
struct UploadTaskDeadline {
    finish_time: Option<Instant>,
    completed_task: bool,
}

impl UploadTaskDeadline {
    fn timed(finish_time: Instant) -> Self {
        Self {
            finish_time: Some(finish_time),
            completed_task: false,
        }
    }

    #[cfg(test)]
    fn zero_finish_time() -> Self {
        Self {
            finish_time: None,
            completed_task: false,
        }
    }

    fn permits_next(&self, now: Instant) -> bool {
        if !self.completed_task {
            return true;
        }
        match self.finish_time {
            None => false,
            Some(finish_time) => now <= finish_time,
        }
    }

    fn complete_task(&mut self) {
        self.completed_task = true;
    }
}

/// Completed CPU meshes waiting for their render-thread upload.
///
/// Java keeps these as `ListenableFutureTask`s in `ChunkRenderDispatcher.queueChunkUploads`.
/// A deadline only postpones running a task; it does not turn a completed task back into a
/// `RenderChunk` rebuild. Keeping the whole [`ChunkMesh`] here is the renderer-side equivalent.
///
/// The queue also owns the small amount of cancellation that matters at this seam. A newer
/// result for the same section supersedes an older retained result, just as
/// `RenderChunk.makeCompileTaskChunk` finishes the old compile task before installing a new one.
#[derive(Debug, Default)]
struct ReadyMeshQueue {
    tasks: VecDeque<ChunkMesh>,
    /// Lifecycle revision matching each retained task.  Kept beside the frozen `ChunkMesh`
    /// payload so the pre-revision CPU-only tests and compatibility methods can continue to
    /// inspect the mesh directly.
    revisions: HashMap<ChunkSectionPos, MeshRevision>,
}

impl ReadyMeshQueue {
    /// Number of completed mesh tasks retained for a later render-thread upload.
    #[must_use]
    fn len(&self) -> usize {
        self.tasks.len()
    }

    fn push_latest_revisioned(&mut self, result: MeshTaskResult) {
        let pos = result.pos();
        if let Some(&current) = self.revisions.get(&pos) {
            if result.revision() < current {
                return;
            }
        }
        self.cancel(pos);
        self.revisions.insert(pos, result.revision());
        self.tasks.push_back(result.into_mesh());
    }

    /// Append a completed mesh, cancelling any older task for that section first.
    #[cfg(test)]
    fn push_latest(&mut self, mesh: ChunkMesh) {
        self.cancel(mesh.pos);
        self.revisions.insert(mesh.pos, 0);
        self.tasks.push_back(mesh);
    }

    /// Cancel a completed task for `pos`, returning whether one was present.
    fn cancel(&mut self, pos: ChunkSectionPos) -> bool {
        // `revisions` holds an entry exactly while a task for `pos` is queued (every push
        // cancels first and then inserts; a pop removes it and a rejected pop restores it), so
        // a position with no entry has no task. Every push calls this, and almost every push is
        // for a new position: without the probe each one scanned the whole queue, O(n^2) over a
        // join burst (TDRAW-8).
        if !self.revisions.contains_key(&pos) {
            return false;
        }
        let mut cancelled = false;
        self.tasks.retain(|task| {
            let keep = task.pos != pos;
            cancelled |= !keep;
            keep
        });
        self.revisions.remove(&pos);
        cancelled
    }

    /// Poll the next task if Java's post-task deadline gate permits it.
    ///
    /// The caller marks the task complete with [`UploadTaskDeadline::complete_task`] only after
    /// the actual upload has run. If the gate rejects the task, it is put back at the front so
    /// the completed mesh remains the first task on the next frame.
    #[cfg(test)]
    fn pop_if_permitted(&mut self, deadline: &UploadTaskDeadline) -> Option<ChunkMesh> {
        let task = self.pop_task_if_permitted(deadline)?;
        Some(task.mesh)
    }

    fn pop_task_if_permitted(&mut self, deadline: &UploadTaskDeadline) -> Option<ReadyMeshTask> {
        let mesh = self.tasks.pop_front()?;
        let revision = self.revisions.remove(&mesh.pos).unwrap_or(0);
        if deadline.permits_next(Instant::now()) {
            Some(ReadyMeshTask { mesh, revision })
        } else {
            self.revisions.insert(mesh.pos, revision);
            self.tasks.push_front(mesh);
            None
        }
    }

    /// Section origins still waiting for upload, in dispatcher queue order.
    fn positions(&self) -> Vec<ChunkSectionPos> {
        self.tasks.iter().map(|task| task.pos).collect()
    }
}

#[derive(Debug)]
struct ReadyMeshTask {
    mesh: ChunkMesh,
    revision: MeshRevision,
}

/// Result of one Java-shaped ready-upload dispatch.
///
/// `retained_positions` are informational: the corresponding complete meshes remain owned by
/// [`TerrainRenderer`] and must not be re-enqueued for meshing. The position list is exposed so a
/// caller can account for deferred work without stealing the retained task itself.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UploadDeadlineResult {
    uploaded_positions: Vec<ChunkSectionPos>,
    retained_positions: Vec<ChunkSectionPos>,
    /// Revisioned worker results rejected before entering the retained ready queue.
    input_revision_rejections: usize,
    /// Retained ready tasks rejected after dequeue because a newer revision or tombstone won.
    retained_revision_rejections: usize,
    /// Number of complete tasks still retained after this dispatch.
    retained_ready_count: usize,
    /// Whether the post-first-task deadline gate prevented another retained task from running.
    deadline_blocked: bool,
}

impl UploadDeadlineResult {
    /// Sections whose complete meshes were uploaded during this dispatch.
    #[must_use]
    pub fn uploaded_positions(&self) -> &[ChunkSectionPos] {
        &self.uploaded_positions
    }

    /// Sections whose complete meshes remain queued for a later dispatch.
    #[must_use]
    pub fn retained_positions(&self) -> &[ChunkSectionPos] {
        &self.retained_positions
    }

    /// Number of completed mesh tasks uploaded during this dispatch.
    #[must_use]
    pub fn uploaded_count(&self) -> usize {
        self.uploaded_positions.len()
    }

    /// Number of input results rejected by the renderer's current lifecycle revision.
    #[must_use]
    pub fn input_revision_rejections(&self) -> usize {
        self.input_revision_rejections
    }

    /// Number of retained ready tasks discarded after dequeue as stale or invalid.
    #[must_use]
    pub fn retained_revision_rejections(&self) -> usize {
        self.retained_revision_rejections
    }

    /// Number of complete tasks still waiting for a later upload dispatch.
    #[must_use]
    pub fn retained_ready_count(&self) -> usize {
        self.retained_ready_count
    }

    /// Whether the post-first-task deadline gate blocked another retained upload.
    #[must_use]
    pub fn deadline_blocked(&self) -> bool {
        self.deadline_blocked
    }
}

/// Shared terrain rendering state: pipelines, atlas, and every resident section's buffers.
///
/// Held behind an `Rc<RefCell<_>>` by the four [`TerrainPhase`]s. The arena-index rule's ban on
/// `Rc<RefCell<_>>` covers `mc-world`/`mc-physics`/`mc-protocol`, not the renderer, and the
/// alternative here — four phases each owning a copy of the atlas and the pipelines — would be
/// strictly worse.
pub struct TerrainRenderer {
    pipelines: [wgpu::RenderPipeline; LAYER_COUNT],
    bind_groups: [wgpu::BindGroup; LAYER_COUNT],
    uniforms: [wgpu::Buffer; LAYER_COUNT],
    index_buffer: wgpu::Buffer,
    index_capacity_quads: u32,
    /// One retained indirect argument buffer per layer, grown geometrically with visible groups.
    indirect_buffers: [wgpu::Buffer; LAYER_COUNT],
    indirect_capacity: [u32; LAYER_COUNT],
    /// Native multi-draw was granted by the adapter/device and may be disabled for an A/B control.
    multi_draw_indirect_supported: bool,
    multi_draw_indirect: bool,
    /// Retained atlas allocation for S09's per-tick animated-sprite dirty rectangles.
    atlas_texture: wgpu::Texture,
    /// Raw RGBA8 bytes for level zero and every resident mip. Java applies its anaglyph
    /// transform to the temporary texture-upload array, so a live setting change must replay
    /// these source levels rather than converting whatever bytes currently happen to be on the
    /// GPU a second time. Shared by the owners of the shared atlas.
    atlas_source_levels: std::rc::Rc<RefCell<Vec<Vec<u8>>>>,
    atlas_width: u32,
    atlas_height: u32,
    atlas_mip_level_count: u32,
    /// The Anisotropic Filtering mod's level and, while on, its pipelines
    /// ([`aniso`]). Off draws with `pipelines` exactly as before.
    aniso: aniso::AnisoState,
    /// The shader packs' terrain variant ([`pack`]): `None` core (the default)
    /// draws with `pipelines` exactly as before.
    pack: pack::PackState,
    /// The group-1 layout `bind_groups` were built with and the build options, kept so the AF
    /// pipelines can be built later against the same bind groups and layer state.
    group1_layout: wgpu::BindGroupLayout,
    options: TerrainOptions,
    /// Java's global upload-time `GameSettings.anaglyph` switch.
    anaglyph: bool,
    /// `Uint16` while a single layer stays under [`MAX_U16_QUADS`], `Uint32` above it. T1.2.
    index_format: wgpu::IndexFormat,
    /// Bytes per vertex in every buffer this renderer owns. [`VERTEX_SIZE`] normally.
    vertex_stride: u64,
    offsets: wgpu::Buffer,
    offsets_capacity: u32,
    /// The 16x16 `EntityRenderer.updateLightmap` texture every layer samples. Raw bytes live in
    /// [`Self::lightmap_source_rgba`]; this allocation contains the current upload representation.
    lightmap_texture: wgpu::Texture,
    /// The parameters `lightmap_texture` was last generated from, so `begin_frame` can notice
    /// [`TerrainRenderer::sky_brightness`] moving without regenerating 256 texels every frame.
    lightmap_params: LightmapParams,
    /// Raw bytes last supplied to the dynamic lightmap. Keeping this beside the texture lets a
    /// live anaglyph toggle rewrite the current table exactly, without double conversion.
    lightmap_source_rgba: Vec<u8>,
    fog: Option<FogState>,
    /// The per-draw fog of the TRANSLUCENT layer alone, with its
    /// distance metric; see [`TerrainRenderer::set_translucent_fog`].
    translucent_fog: Option<(FogState, FogDistance)>,
    chunks: HashMap<ChunkSectionPos, ChunkBuffers>,
    /// Authoritative current-world column membership, when the client supplies it. `None`
    /// preserves the standalone renderer's historical behavior: absent sections are hard stops.
    /// This is lifecycle state, not GPU residency; uploads, removals, and budget eviction never
    /// mutate it.
    world_columns: Option<HashSet<ChunkPos>>,
    /// Completed meshes waiting for the render-thread upload deadline. This is the CPU analogue
    /// of Java's `ChunkRenderDispatcher.queueChunkUploads`; deadline-deferred work stays ready
    /// instead of returning to [`MeshQueue`](super::jobs::MeshQueue) for another rebuild.
    ready_uploads: ReadyMeshQueue,
    /// Highest revision accepted for each section, including tombstones left by remove/cancel.
    /// A result only reaches the GPU when it is newer than an invalidation tombstone, so a worker
    /// that races an unload or re-enqueue cannot resurrect old geometry.
    mesh_revisions: HashMap<ChunkSectionPos, MeshRevision>,
    invalid_mesh_revisions: HashSet<ChunkSectionPos>,
    /// Version of what the culling walk can observe: bumped when a section is added or removed,
    /// when a replacement changes its visibility matrix or camera-cell opacity, and when the
    /// authoritative world-column set changes. [`SectionWalk`] keys its cached result on this,
    /// so anything that changes what the culling walk would answer has to move it. See
    /// [`SectionWalk::run`] and [`ResidentChange`]. A geometry-only re-mesh, or an all-air record
    /// replaced by an identical one, does not move it.
    chunk_generation: u64,
    /// Per-layer version of the resident geometry the draw plans reference. Bumped (and that
    /// layer's plan dropped) whenever a record that has, or had, geometry in the layer is
    /// inserted, replaced, removed or evicted. Uploads that only touch other layers keep a
    /// layer's plan.
    layer_generations: [u64; LAYER_COUNT],
    /// The reusable culling walk, with its cached result. S23; see [`super::cull`].
    walk: SectionWalk,
    /// Dense camera-centred cache of what the walk reads per section (#7). Every `chunks`
    /// insert/remove and every `world_columns` change below re-derives the affected cells.
    walk_grid: WalkGrid,
    /// The camera section's last `getVisibleFacings` answer (#34).
    camera_reach_memo: Cell<Option<CameraReachMemo>>,
    /// How much terrain vertex memory may live on the GPU, and what is using it.
    residency: GpuResidency,
    budget: GpuBudget,
    /// Whether the optional S23 GPU-memory cap is active. The live Java-shaped upload path
    /// drains ready tasks by deadline and leaves this optimization disabled; the legacy
    /// budgeted APIs and explicit `set_gpu_budget` opt in so their existing controls remain
    /// meaningful.
    enforce_gpu_budget: bool,
    uniform_writes: [crate::context::ChangedWrite; LAYER_COUNT],
    arena: Rc<RefCell<VertexArena>>,
    /// What the lightmap texture currently holds, when it came from parameters.
    uploaded_lightmap_params: Option<LightmapParams>,
    /// Sections `evict_to_budget` dropped and the caller has not been told about yet. Drained by
    /// [`TerrainRenderer::take_evicted`].
    evicted: Vec<ChunkSectionPos>,
    /// Whether the "the budget bit" line has been logged. See `begin_frame`.
    warned_evicting: bool,
    /// Terrain buffers the GPU refused ([`TerrainRenderer::refused_uploads`]).
    refused_uploads: u64,
    /// How often to log a [`GpuMemoryWindow`], or `None` (the default) for never. Set from the
    /// profiler's own `MC_PROFILE`/`MC_PROFILE_INTERVAL` so the residency table lines up with the
    /// scope table in the same log.
    memory_log_interval: Option<std::time::Duration>,
    /// When the last [`GpuMemoryWindow`] was logged.
    memory_log_last: std::time::Instant,
    /// Selected by the last [`TerrainRenderer::begin_frame`], with the instance slot each one
    /// was assigned in `offsets`. Only sections with geometry: all-air records stay resident for
    /// the walk but never enter this list (#2).
    visible: Vec<(VisibleSection, u32)>,
    /// Whether the last [`TerrainRenderer::begin_frame`] walk reached any resident section, with
    /// or without geometry. This is what `visible` being non-empty meant before all-air records
    /// left it, and it still decides whether the terrain passes open.
    walk_reached_resident: bool,
    /// Bumped by every resident change the walk or a draw plan can observe (see
    /// [`ResidentChange`]); an all-air record replaced by an identical one does not move it.
    resident_epoch: u64,
    /// `(walk misses, resident_epoch)` that `visible`, `walk_reached_resident`, the uploaded
    /// offsets and `visible_signature` were built from. While both are unchanged the walk has
    /// answered every call since from its cache, i.e. for bit-identical camera inputs (so every
    /// camera-relative offset is bit-identical too), and no record gained or lost anything the
    /// list reads, so all four are still exact and `begin_frame` reuses them.
    draw_list_key: Option<(u64, u64)>,
    /// Per-layer submission plans. A cache hit avoids rebuilding `Rc` item lists, indirect
    /// argument vectors, and the corresponding queue write; a change to the drawn order or to
    /// any resident geometry in that layer changes the key before a draw can reuse one.
    draw_plans: [Option<LayerDrawPlan>; LAYER_COUNT],
    draw_plan_rebuilds: u64,
    draw_plan_cache_hits: u64,
    /// Hash of the selected section order and instance slots from the last begin-frame walk.
    visible_signature: u64,
    /// At most one bounded translucent CPU resort is in flight. The receiver is polled from the
    /// render thread; the worker never receives a [`Gpu`] and therefore cannot write GPU state.
    translucent_resort: Option<Receiver<TranslucentResortResult>>,
    /// Cooperative cancellation for the in-flight worker, used when the whole job is abandoned
    /// (the synchronous [`TerrainRenderer::resort_translucent`]). Its receiver is dropped too,
    /// so a result that races the cancellation is never polled.
    translucent_resort_cancel: Option<Arc<AtomicBool>>,
    /// The sections the in-flight resort still owns (at most
    /// [`MAX_TRANSLUCENT_RESORT_SECTIONS`]). Replacing or removing one unlists only that section,
    /// as vanilla cancels only that chunk's resort; the job runs on and its result is written
    /// only into the sections still listed here.
    translucent_resort_sections: Vec<ChunkSectionPos>,
    /// Identifies the in-flight resort; a result is accepted only for the current one.
    translucent_resort_version: u64,
    /// Test seam ([`TerrainRenderer::set_translucent_near_to_far_for_test`]): submit
    /// `TRANSLUCENT` in the walk's near-to-far order, the reverse of vanilla's. Never set outside
    /// order-sensitivity controls.
    translucent_near_to_far: bool,
    /// Staging for `offsets`, kept between frames so the per-frame `Vec` is one `clear` rather
    /// than an allocation of up to `visible.len()` entries.
    offset_scratch: Vec<ChunkOffset>,
    /// Render distance in chunks. java: `GameSettings.renderDistanceChunks`.
    pub render_distance: u32,
    /// Vanilla's `Minecraft.renderChunksMany` — the occlusion-culling toggle.
    pub smart_culling: bool,
    /// Whether the render-view player is spectating. Java uses this only to bypass the
    /// camera-section opaque-cell shortcut; the ordinary smart-culling walk remains enabled.
    pub player_spectator: bool,
    /// `0..1` sun brightness, java: `World.getSunBrightness(1.0F)` /
    /// [`crate::sky::environment::Environment::sun_brightness`].
    ///
    /// Feeds [`LightmapParams::sun_brightness`]: setting it regenerates the lightmap texture on
    /// the next frame, which is how the time of day reaches terrain. For anything richer than
    /// the sun term — night vision, gamma, the torch flicker, the Nether's ambient floor —
    /// call [`TerrainRenderer::set_lightmap_params`] instead and leave this alone.
    pub sky_brightness: f32,
    last_sort_pos: DVec3,
    stats: TerrainStats,
}

/// The instance attribute: a section's origin minus the eye position.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ChunkOffset {
    offset: [f32; 4],
}

impl ChunkOffset {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ChunkOffset>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![4 => Float32x4],
    };
}

/// Mirrors `terrain.wgsl`'s `TerrainUniform`. The fog half uses the same packing as
/// `sky/fog.wgsl`'s uniform on purpose, so the screen-space pass and this one cannot drift.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainUniform {
    /// `x` = alpha cutoff, `y` = fog enabled.
    params: [f32; 4],
    /// `rgb` = fog colour, `a` = 0 linear / 1 exp.
    fog_color: [f32; 4],
    /// `x` = linear start, `y` = linear end, `z` = exp density.
    fog_params: [f32; 4],
}

impl TerrainUniform {
    fn new(cutoff: f32, fog: Option<FogState>) -> Self {
        let (enabled, color, params) = match fog {
            None => (0.0, [0.0; 4], [0.0; 4]),
            Some(f) => {
                let (mode_flag, p) = match f.mode {
                    FogMode::Linear { start, end } => (0.0, [start, end, 0.0, 0.0]),
                    FogMode::Exp { density } => (1.0, [0.0, 0.0, density, 0.0]),
                };
                (1.0, [f.color.x, f.color.y, f.color.z, mode_flag], p)
            }
        };
        Self {
            params: [cutoff, enabled, 0.0, 0.0],
            fog_color: color,
            fog_params: params,
        }
    }

    /// [`Self::new`] with the fog distance metric in `params.z`
    /// (`terrain.wgsl`'s vertex stage reads it): 0 planar, 1 radial.
    fn with_distance(cutoff: f32, fog: Option<FogState>, distance: FogDistance) -> Self {
        let mut uniform = Self::new(cutoff, fog);
        uniform.params[2] = match distance {
            FogDistance::Planar => 0.0,
            FogDistance::Radial => 1.0,
        };
        uniform
    }
}

/// java: `EntityRenderer.lightmapTexture` is 16x16 RGBA.
const LIGHTMAP_SIDE: u32 = 16;

/// Java's `TextureUtil.allocateTextureImpl` allocates the terrain atlas with `GL_RGBA`.
///
/// `GL_RGBA` is plain normalized byte storage: the bytes stitched by `TextureMap` are sampled
/// as-is before vanilla multiplies them by the terrain vertex colour and lightmap. Keep this
/// contract explicit so an sRGB texture allocation cannot silently add a second colour-space
/// conversion to the ordinary terrain path.
const JAVA_TERRAIN_ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Build the temporary RGBA8 array that Java hands to `TextureUtil.uploadTexture`.
///
/// `TextureUtil.anaglyphColor` mixes only RGB with integer arithmetic and carries alpha through
/// unchanged.  Keeping this boundary local to terrain makes every atlas/lightmap write obey the
/// same rule while the disabled path remains an exact byte-for-byte clone.
fn terrain_upload_rgba(source_rgba: &[u8], anaglyph: bool) -> Vec<u8> {
    mc_assets::sprite::anaglyph_rgba(source_rgba, anaglyph)
}

/// Dimensions of one resident atlas mip. Atlas levels are only valid while both dimensions are
/// non-zero; checked shifts turn malformed animation records into a rejected upload instead of a
/// panic from an oversized shift count.
fn atlas_mip_dimensions(width: u32, height: u32, level: u32) -> (u32, u32) {
    (
        width.checked_shr(level).unwrap_or(0),
        height.checked_shr(level).unwrap_or(0),
    )
}

fn create_lightmap_texture_with_anaglyph(
    gpu: &Gpu,
    lightmap: &Lightmap,
    anaglyph: bool,
) -> wgpu::Texture {
    assert_eq!(
        lightmap.rgba.len(),
        (LIGHTMAP_SIDE * LIGHTMAP_SIDE * 4) as usize,
        "invalid terrain lightmap payload"
    );
    let upload_rgba = terrain_upload_rgba(&lightmap.rgba, anaglyph);
    gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: Some("terrain lightmap"),
            size: wgpu::Extent3d {
                width: LIGHTMAP_SIDE,
                height: LIGHTMAP_SIDE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Deliberately **not** `Rgba8UnormSrgb`: `Lightmap::generate` already produces a
            // linear multiplier per channel (vanilla uploads it with `GL_RGBA`/no sRGB
            // conversion and modulates it in fixed-function), so re-linearising it here would
            // darken every lit surface.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &upload_rgba,
    )
}

/// Write one complete 16x16 lightmap to the retained dynamic texture. The caller supplies the
/// already-selected upload bytes so both initial allocation and live rewrites use one exact path.
fn upload_lightmap_texture(gpu: &Gpu, texture: &wgpu::Texture, rgba: &[u8]) {
    assert_eq!(
        rgba.len(),
        (LIGHTMAP_SIDE * LIGHTMAP_SIDE * 4) as usize,
        "invalid terrain lightmap payload"
    );
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LIGHTMAP_SIDE * 4),
            rows_per_image: Some(LIGHTMAP_SIDE),
        },
        wgpu::Extent3d {
            width: LIGHTMAP_SIDE,
            height: LIGHTMAP_SIDE,
            depth_or_array_layers: 1,
        },
    );
}

/// Upload one already-stitched atlas mip level. `Queue::write_texture` accepts tight rows, so
/// the final 1px level needs no artificial 256-byte padding.
fn upload_atlas_level_with_anaglyph(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    level: u32,
    rgba: &[u8],
    width: u32,
    height: u32,
    anaglyph: bool,
) {
    assert_eq!(
        rgba.len(),
        (width * height * 4) as usize,
        "invalid terrain atlas mip {level}"
    );
    let upload_rgba = terrain_upload_rgba(rgba, anaglyph);
    let mut destination = texture.as_image_copy();
    destination.mip_level = level;
    gpu.queue.write_texture(
        destination,
        &upload_rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Validate one dirty rectangle before handing it to wgpu. Animation records are produced from
/// pack data, so treating their coordinates and byte length as untrusted keeps a malformed
/// `.mcmeta`/sprite from turning a client tick into a device validation error.
fn atlas_animation_rect_valid(
    upload: &AtlasAnimationUpload,
    atlas_width: u32,
    atlas_height: u32,
    mip_level_count: u32,
) -> bool {
    let Some(expected_bytes) = u64::from(upload.width)
        .checked_mul(u64::from(upload.height))
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return false;
    };
    if upload.mip_level >= mip_level_count
        || upload.width == 0
        || upload.height == 0
        || upload.rgba.len() as u64 != expected_bytes
    {
        return false;
    }
    let (width, height) = atlas_mip_dimensions(atlas_width, atlas_height, upload.mip_level);
    width > 0
        && height > 0
        && upload
            .x
            .checked_add(upload.width)
            .is_some_and(|end| end <= width)
        && upload
            .y
            .checked_add(upload.height)
            .is_some_and(|end| end <= height)
}

fn upload_atlas_rect_with_anaglyph(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    upload: &AtlasAnimationUpload,
    anaglyph: bool,
) {
    let upload_rgba = terrain_upload_rgba(&upload.rgba, anaglyph);
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: upload.mip_level,
            origin: wgpu::Origin3d {
                x: upload.x,
                y: upload.y,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &upload_rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(upload.width * 4),
            rows_per_image: Some(upload.height),
        },
        wgpu::Extent3d {
            width: upload.width,
            height: upload.height,
            depth_or_array_layers: 1,
        },
    );
}

impl TerrainRenderer {
    /// Build the pipelines and upload the atlas.
    ///
    /// S09's [`Atlas`] carries sprite-local stitched mip levels. Upload each supplied level as
    /// its own source image rather than generating mips from the completed atlas, which would
    /// bleed colour across sprite boundaries. `CUTOUT_MIPPED` therefore samples the same
    /// sprite-local chain as vanilla while `CUTOUT` remains pinned to level zero.
    #[must_use]
    pub fn new(gpu: &Gpu, atlas: &Atlas) -> Self {
        Self::new_with(gpu, atlas, TerrainOptions::default())
    }

    /// Supply the current world's authoritative loaded-column membership.
    ///
    /// This is deliberately separate from mesh residency: a loaded column may have no compiled
    /// section yet and must remain a DUMMY hard stop, while a column absent from this set is
    /// treated as settled all-air for culling. Replacing the set also bounds lifecycle metadata
    /// to the actual current world; JoinGame, dimension reset, and world teardown can clear it
    /// with an empty iterator. Calling this once opts the renderer into authoritative behavior.
    pub fn set_world_columns<I>(&mut self, columns: I)
    where
        I: IntoIterator<Item = ChunkPos>,
    {
        let next: HashSet<ChunkPos> = columns.into_iter().collect();
        let changed = self.world_columns.as_ref() != Some(&next);
        self.world_columns = Some(next);
        if changed {
            self.walk_grid.invalidate();
            self.bump_walk_generation();
        }
    }

    /// Mark one column loaded when authoritative membership has been enabled.
    ///
    /// The default standalone renderer has no world-membership authority, so this is a no-op
    /// until [`Self::set_world_columns`] has been called.
    pub fn mark_world_column_loaded(&mut self, pos: ChunkPos) {
        let changed = self
            .world_columns
            .as_mut()
            .is_some_and(|columns| columns.insert(pos));
        if changed {
            self.note_walk_column(pos);
            self.bump_walk_generation();
        }
    }

    /// Mark one column absent when authoritative membership has been enabled.
    ///
    /// This changes culling semantics only. The caller still owns cancellation/removal of any
    /// real mesh and must not use this method for renderer-cache eviction.
    pub fn mark_world_column_unloaded(&mut self, pos: ChunkPos) {
        let changed = self
            .world_columns
            .as_mut()
            .is_some_and(|columns| columns.remove(&pos));
        if changed {
            self.note_walk_column(pos);
            self.bump_walk_generation();
        }
    }

    /// Re-derive the walk grid's cell for `pos` after `chunks` changed there (#7).
    fn note_walk_section(&mut self, pos: ChunkSectionPos) {
        let resident = self.chunks.get(&pos).map(|chunk| chunk.visibility);
        let cell = walk_grid::encode(walk_grid::authoritative(
            resident,
            self.world_columns.as_ref(),
            pos,
        ));
        self.walk_grid.set_section(pos, cell);
    }

    /// Re-derive the walk grid's cells for `col` after its world membership changed (#7).
    fn note_walk_column(&mut self, col: ChunkPos) {
        let (chunks, world_columns) = (&self.chunks, self.world_columns.as_ref());
        self.walk_grid.refresh_column(col, |col, cells| {
            fill_walk_column(chunks, world_columns, col, cells);
        });
    }

    /// Columns the walk grid has filled from the hashed maps: what recentring has cost. For
    /// diagnostics and tests; a column boundary crossing costs one strip of `2r + 1`.
    #[must_use]
    pub fn walk_grid_column_fills(&self) -> u64 {
        self.walk_grid.column_fills()
    }

    /// [`TerrainRenderer::new`] with explicit [`TerrainOptions`].
    #[must_use]
    #[allow(clippy::too_many_lines)] // One pipeline-construction function, not a mixed bag.
    pub fn new_with(gpu: &Gpu, atlas: &Atlas, options: TerrainOptions) -> Self {
        Self::new_with_options_and_anaglyph(gpu, atlas, options, false)
    }

    /// Build the renderer with Java's global upload-time anaglyph switch selected.
    ///
    /// The ordinary [`Self::new`] and [`Self::new_with`] constructors remain disabled for
    /// compatibility.  A client that owns `GameSettings` can seed the terrain copy here, or
    /// call [`Self::set_anaglyph`] before its first atlas/lightmap upload.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, atlas: &Atlas, anaglyph: bool) -> Self {
        Self::new_with_options_and_anaglyph(gpu, atlas, TerrainOptions::default(), anaglyph)
    }

    /// [`Self::new_with`] with an explicit startup anaglyph switch.
    #[must_use]
    #[allow(clippy::too_many_lines)] // One pipeline-construction function, not a mixed bag.
    pub fn new_with_options_and_anaglyph(
        gpu: &Gpu,
        atlas: &Atlas,
        options: TerrainOptions,
        anaglyph: bool,
    ) -> Self {
        let mip_level_count = atlas.mipmaps.len() as u32 + 1;
        let mut atlas_source_levels = Vec::with_capacity(mip_level_count as usize);
        atlas_source_levels.push(atlas.texture.clone());
        atlas_source_levels.extend(atlas.mipmaps.iter().cloned());
        // Inside the client's install sweep this is the first owner of the
        // one shared GPU block atlas (`atlas_transfer::AtlasShareScope`); a plain create otherwise.
        let texture = crate::atlas_transfer::share_atlas_texture(
            gpu,
            &wgpu::TextureDescriptor {
                label: Some("terrain atlas"),
                size: wgpu::Extent3d {
                    width: atlas.w,
                    height: atlas.h,
                    depth_or_array_layers: 1,
                },
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: JAVA_TERRAIN_ATLAS_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            &atlas_source_levels,
            anaglyph,
        );
        upload_atlas_level_with_anaglyph(
            gpu,
            &texture,
            0,
            &atlas_source_levels[0],
            atlas.w,
            atlas.h,
            anaglyph,
        );
        for (index, pixels) in atlas_source_levels.iter().enumerate().skip(1) {
            let level = index as u32;
            let (width, height) = atlas_mip_dimensions(atlas.w, atlas.h, level);
            upload_atlas_level_with_anaglyph(gpu, &texture, level, pixels, width, height, anaglyph);
        }
        // One raw shadow for every owner of the shared GPU atlas.
        let atlas_source_levels =
            crate::atlas_transfer::share_atlas_shadow(&texture, atlas_source_levels);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // java: TextureMap uses GL_NEAREST for magnification — Minecraft's look depends on it.
        let mipmapped_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain atlas mipmapped sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let level_zero_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain atlas level-zero sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_max_clamp: 0.0,
            ..Default::default()
        });

        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain group1"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // java: `EntityRenderer` builds the lightmap with `setTextureFilter(GL_LINEAR)` and
        // clamped wrapping, so the 16x16 table reads as a smooth gradient rather than 256 flat
        // cells, and level 15 does not wrap round to level 0 at the edge.
        let lightmap_params = LightmapParams::default();
        let lightmap = Lightmap::generate(lightmap_params);
        let lightmap_source_rgba = lightmap.rgba.clone();
        let lightmap_texture = create_lightmap_texture_with_anaglyph(gpu, &lightmap, anaglyph);
        let lightmap_view = lightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let lightmap_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain lightmap sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // One uniform buffer per layer, because the alpha cutoff differs per layer and the
        // bind group is baked into the pipeline's layout.
        let uniforms: [wgpu::Buffer; LAYER_COUNT] = std::array::from_fn(|i| {
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("terrain uniform"),
                    contents: bytemuck::bytes_of(&TerrainUniform::new(layer_alpha_cutoff(i), None)),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
        });
        let bind_groups: [wgpu::BindGroup; LAYER_COUNT] = std::array::from_fn(|i| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain group1"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(if layer_uses_mipmaps(i) {
                            &mipmapped_sampler
                        } else {
                            &level_zero_sampler
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniforms[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&lightmap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&lightmap_sampler),
                    },
                ],
            })
        });

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("terrain"),
                source: wgpu::ShaderSource::Wgsl(include_str!("terrain.wgsl").into()),
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain layout"),
                bind_group_layouts: &[Some(gpu.camera_layout()), Some(&layout)],
                immediate_size: 0,
            });

        let pipelines = std::array::from_fn(|i| {
            let translucent = i == 3;
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("terrain"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(if options.packed_vertices {
                            "vs_packed"
                        } else {
                            "vs"
                        }),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[
                            Some(if options.packed_vertices {
                                PackedTerrainVertex::layout()
                            } else {
                                TerrainVertex::layout()
                            }),
                            Some(ChunkOffset::LAYOUT),
                        ],
                    },
                    primitive: wgpu::PrimitiveState {
                        // java: Minecraft.startGame's `cullFace(1029)` = GL_BACK, with
                        // `EntityRenderer.renderWorldPass` calling `enableCull()` around the
                        // terrain layers. `wgpu::FrontFace::Ccw` (the `PrimitiveState` default)
                        // matches the winding `FaceBakery` emits — every baked quad's `0,1,2`
                        // triangle is counter-clockwise seen from outside, asserted in
                        // `mc_assets`' `every_baked_quad_winds_counter_clockwise_about_its_own_facing`
                        // and re-proved on the GPU by `s11_cull.rs`.
                        cull_mode: options.cull_backfaces.then_some(wgpu::Face::Back),
                        ..wgpu::PrimitiveState::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: crate::DEPTH_FORMAT,
                        // java: EntityRenderer.renderWorldPass — glDepthMask(false) for the
                        // translucent pass only.
                        depth_write_enabled: Some(!translucent),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(if i == layer_index(RenderLayer::Solid) {
                            "fs_solid"
                        } else {
                            "fs"
                        }),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: translucent.then_some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        });

        // T1.2: 4096 quads is 24 576 indices, whose largest value is 16 387 — a `u16`, and half
        // the bytes. See `grow_indices` for the `Uint32` fallback.
        let index_capacity_quads = 4096;
        let index_format = wgpu::IndexFormat::Uint16;
        let index_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain indices"),
                contents: bytemuck::cast_slice(&quad_indices_u16(index_capacity_quads)),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });

        let indirect_capacity = [1; LAYER_COUNT];
        let indirect_buffers = std::array::from_fn(|layer| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match layer {
                    0 => "terrain solid indirect draws",
                    1 => "terrain cutout-mipped indirect draws",
                    2 => "terrain cutout indirect draws",
                    3 => "terrain translucent indirect draws",
                    _ => unreachable!(),
                }),
                size: std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let offsets_capacity = 1024;
        let offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain chunk offsets"),
            size: u64::from(offsets_capacity) * std::mem::size_of::<ChunkOffset>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let budget = budget_from_env();
        let multi_draw_indirect =
            gpu.supports_multi_draw_indirect() && multi_draw_wanted(&gpu.adapter_info());
        Self {
            pipelines,
            bind_groups,
            uniforms,
            index_buffer,
            index_capacity_quads,
            indirect_buffers,
            indirect_capacity,
            multi_draw_indirect_supported: gpu.supports_multi_draw_indirect(),
            multi_draw_indirect,
            atlas_texture: texture,
            atlas_source_levels,
            atlas_width: atlas.w,
            atlas_height: atlas.h,
            atlas_mip_level_count: mip_level_count,
            aniso: aniso::AnisoState::default(),
            pack: pack::PackState::default(),
            group1_layout: layout,
            options,
            anaglyph,
            index_format,
            vertex_stride: if options.packed_vertices {
                PACKED_VERTEX_SIZE as u64
            } else {
                VERTEX_SIZE as u64
            },
            offsets,
            offsets_capacity,
            lightmap_texture,
            lightmap_params,
            lightmap_source_rgba,
            fog: None,
            translucent_fog: None,
            chunks: HashMap::new(),
            world_columns: None,
            ready_uploads: ReadyMeshQueue::default(),
            mesh_revisions: HashMap::new(),
            invalid_mesh_revisions: HashSet::new(),
            chunk_generation: 0,
            layer_generations: [0; LAYER_COUNT],
            walk: SectionWalk::new(),
            walk_grid: WalkGrid::new(),
            camera_reach_memo: Cell::new(None),
            residency: GpuResidency::new(budget),
            budget,
            enforce_gpu_budget: false,
            uniform_writes: Default::default(),
            arena: Rc::new(RefCell::new(VertexArena {
                grow_enabled: multi_draw_indirect,
                ..VertexArena::default()
            })),
            uploaded_lightmap_params: None,
            evicted: Vec::new(),
            warned_evicting: false,
            refused_uploads: 0,
            memory_log_interval: memory_log_interval_from_env(),
            memory_log_last: std::time::Instant::now(),
            visible: Vec::new(),
            walk_reached_resident: false,
            resident_epoch: 0,
            draw_list_key: None,
            draw_plans: [None, None, None, None],
            draw_plan_rebuilds: 0,
            draw_plan_cache_hits: 0,
            visible_signature: 0,
            translucent_resort: None,
            translucent_resort_cancel: None,
            translucent_resort_sections: Vec::new(),
            translucent_resort_version: 0,
            translucent_near_to_far: false,
            offset_scratch: Vec::new(),
            render_distance: 8,
            smart_culling: true,
            player_spectator: false,
            sky_brightness: 1.0,
            last_sort_pos: DVec3::new(f64::MAX, f64::MAX, f64::MAX),
            stats: TerrainStats::default(),
        }
    }

    /// Apply one S09 animated-atlas dirty rectangle to the retained terrain texture.
    ///
    /// java: `TextureMap.updateAnimations` -> `TextureUtil.uploadTextureMipmap`, which updates
    /// the sprite rectangle at every available mip level rather than rebuilding the atlas. The
    /// CPU animator has already generated the level-specific bytes; this method only validates
    /// the record and queues its subresource write.
    pub fn upload_atlas_animation(&self, gpu: &Gpu, upload: &AtlasAnimationUpload) -> bool {
        self.upload_atlas_animation_with_sink(upload, |texture, anaglyph| {
            upload_atlas_rect_with_anaglyph(gpu, texture, upload, anaglyph);
        })
    }

    /// Experimental transfer seam: validate and retain raw pixels before invoking the sink.
    /// The sink receives this owner's exact texture and current transform setting.
    #[doc(hidden)]
    pub fn upload_atlas_animation_with_sink(
        &self,
        upload: &AtlasAnimationUpload,
        sink: impl FnOnce(&wgpu::Texture, bool),
    ) -> bool {
        if !atlas_animation_rect_valid(
            upload,
            self.atlas_width,
            self.atlas_height,
            self.atlas_mip_level_count,
        ) {
            return false;
        }
        // Atlas animator records are raw CPU pixels. Keep the latest frame in the retained
        // source level, then apply Java's transform only to the upload-owned rectangle. This is
        // what makes both mip-specific animation and a later false -> true -> false toggle exact.
        let (level_width, _) =
            atlas_mip_dimensions(self.atlas_width, self.atlas_height, upload.mip_level);
        {
            let mut source_levels = self.atlas_source_levels.borrow_mut();
            let Some(source) = source_levels.get_mut(upload.mip_level as usize) else {
                return false;
            };
            for row in 0..upload.height {
                let src_start = (row * upload.width * 4) as usize;
                let dst_start = (((upload.y + row) * level_width + upload.x) * 4) as usize;
                let count = (upload.width * 4) as usize;
                source[dst_start..dst_start + count]
                    .copy_from_slice(&upload.rgba[src_start..src_start + count]);
            }
        }
        sink(&self.atlas_texture, self.anaglyph);
        true
    }

    /// Whether this renderer applies Java's global upload-time anaglyph transform.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global texture-upload transform and rebuild the resident atlas levels and
    /// dynamic lightmap from retained raw source bytes. Replaying raw pixels avoids applying the
    /// integer transform twice and includes the latest animated frame at every mip.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;

        let source_levels = self.atlas_source_levels.borrow().clone();
        for (level, source_rgba) in source_levels.iter().enumerate() {
            let level = level as u32;
            let (width, height) = atlas_mip_dimensions(self.atlas_width, self.atlas_height, level);
            upload_atlas_level_with_anaglyph(
                gpu,
                &self.atlas_texture,
                level,
                source_rgba,
                width,
                height,
                enabled,
            );
        }

        let upload_rgba = terrain_upload_rgba(&self.lightmap_source_rgba, enabled);
        upload_lightmap_texture(gpu, &self.lightmap_texture, &upload_rgba);
    }

    /// Set the transform for subsequent atlas/lightmap uploads without touching resident GPU
    /// bytes. Use [`Self::set_anaglyph_with_gpu`] when an already-installed renderer must change
    /// immediately.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Upload (or replace) one section's geometry. The translucent layer is sorted for
    /// `camera` on the way in, the way `RenderChunk.rebuildChunk` sorts before it uploads.
    ///
    /// Since S23 this is [`TerrainRenderer::upload_all`] with a one-section batch, which for a
    /// single section produces exactly what it always did: one buffer per non-empty layer, each
    /// holding that section alone at offset 0. The budget is deliberately **not** applied here —
    /// a caller asking for one section to be resident gets it, and the cap is enforced by
    /// `evict_to_budget` on the next frame instead, so this function's contract is unchanged.
    ///
    /// Returns whether the section is now resident: `false` only when the GPU refused its buffer
    /// (a lost device, no memory, over the buffer limit), never for the budget,
    /// since a single-section upload runs with an unlimited packing budget.
    pub fn upload(&mut self, gpu: &Gpu, mesh: ChunkMesh, camera: DVec3) -> bool {
        self.enforce_gpu_budget = true;
        let deferred = self.upload_with_budget(gpu, vec![mesh], camera, GpuBudget::new(u64::MAX));
        deferred.is_empty()
    }

    /// Upload a whole batch of sections as **one GPU buffer per render layer** instead of one
    /// per section per layer.
    ///
    /// "Chunk mesh upload batching + GPU memory budget". Measured on 100 real meshed
    /// sections: **150 buffers on the per-section path, 4 on this one.** The packing itself is
    /// [`UploadPlan::plan`], which is pure data and tested exhaustively without a GPU; all this
    /// function adds is the `create_buffer_init` per batch and the bookkeeping.
    ///
    /// Returns the sections that did not fit under [`TerrainRenderer::gpu_budget`] and were left
    /// for a later frame — requeue them. A section is never split across frames: half a
    /// section's geometry is a visible hole.
    ///
    /// # Panics
    ///
    /// If a planned slice is not 4-byte aligned, which would make `buffer.slice(..)` invalid on
    /// some backends. `UploadPlan` guarantees it; the assertion is here because it is cheap and
    /// because a misaligned bind is a hard-to-diagnose backend-specific failure.
    pub fn upload_all(
        &mut self,
        gpu: &Gpu,
        meshes: Vec<ChunkMesh>,
        camera: DVec3,
    ) -> Vec<ChunkSectionPos> {
        self.enforce_gpu_budget = true;
        let budget = self.budget;
        self.upload_with_budget(gpu, meshes, camera, budget)
    }

    fn accepts_mesh_revision(&self, pos: ChunkSectionPos, revision: MeshRevision) -> bool {
        match self.mesh_revisions.get(&pos).copied() {
            None => true,
            Some(current) if self.invalid_mesh_revisions.contains(&pos) => revision > current,
            Some(current) => revision >= current,
        }
    }

    fn accept_mesh_revision(&mut self, pos: ChunkSectionPos, revision: MeshRevision) -> bool {
        if !self.accepts_mesh_revision(pos, revision) {
            return false;
        }
        self.mesh_revisions
            .entry(pos)
            .and_modify(|current| *current = (*current).max(revision))
            .or_insert(revision);
        self.invalid_mesh_revisions.remove(&pos);
        true
    }

    fn invalidate_mesh_revision(&mut self, pos: ChunkSectionPos) {
        // Draw from the same process-wide authority as `MeshQueue` rather than adding one to a
        // local value.  This keeps a freshly-created queue (for a new dimension) newer than a
        // renderer tombstone left by the previous world's queue.
        let tombstone = next_mesh_revision();
        self.mesh_revisions.insert(pos, tombstone);
        self.invalid_mesh_revisions.insert(pos);
    }

    /// Reject a blank-section result unless it is newer than the section's current lifecycle
    /// revision, then remove the old GPU allocation.  This is the empty/tile-preserving half of
    /// the worker handoff: a stale blank result must not erase a newer non-empty upload.
    pub fn remove_if_revision(&mut self, pos: ChunkSectionPos, revision: MeshRevision) -> bool {
        if !self.accept_mesh_revision(pos, revision) {
            return false;
        }
        self.ready_uploads.cancel(pos);
        let removed = self.remove_resident(pos);
        if removed {
            self.residency.remove(pos);
        }
        self.invalidate_mesh_revision(pos);
        removed
    }

    /// Upload ready terrain tasks using Java 1.8.9's absolute finish-time gate.
    ///
    /// This compatibility wrapper preserves the earlier position-list return type. Deferred
    /// positions are now retained as complete meshes by this renderer, so callers using this
    /// wrapper must treat the returned list as informational and must not re-enqueue it. New
    /// integrations should use [`Self::upload_all_with_deadline_retaining`] to also account for
    /// which tasks actually uploaded.
    pub fn upload_all_with_deadline(
        &mut self,
        gpu: &Gpu,
        meshes: Vec<ChunkMesh>,
        camera: DVec3,
        finish_time: Instant,
    ) -> Vec<ChunkSectionPos> {
        self.upload_all_with_deadline_retaining(gpu, meshes, camera, finish_time)
            .retained_positions
    }

    /// Upload ready terrain tasks and retain deadline-deferred meshes for the next frame.
    ///
    /// `ChunkRenderDispatcher.runChunkUploads` has no vertex-byte packing budget: it runs one
    /// queued upload task, then keeps polling while `System.nanoTime()` has not passed the
    /// caller's `finishTimeNano`. The first task is unconditional, including an already-expired
    /// deadline. Rust's [`ChunkBuffers`] installs a section atomically, so one [`ChunkMesh`] is
    /// the smallest faithful task here; its upload uses an unlimited packing budget. Tasks
    /// rejected by the post-task gate stay in the renderer's ready queue instead of going back to
    /// [`MeshQueue`](super::jobs::MeshQueue) and being meshed a second time.
    ///
    /// A newly completed mesh for a section replaces an older retained task for that section.
    /// This mirrors `RenderChunk.makeCompileTaskChunk` finishing the superseded compile task,
    /// including the cancellation of its eventual upload future.
    pub fn upload_all_with_deadline_retaining(
        &mut self,
        gpu: &Gpu,
        meshes: Vec<ChunkMesh>,
        camera: DVec3,
        finish_time: Instant,
    ) -> UploadDeadlineResult {
        let meshes = meshes
            .into_iter()
            .map(MeshTaskResult::unversioned)
            .collect();
        self.upload_all_with_deadline_retaining_revisioned(gpu, meshes, camera, finish_time)
    }

    /// Upload revisioned worker results and retain deadline-deferred tasks for a later frame.
    ///
    /// A result that loses a cancellation/re-enqueue race is dropped before it reaches
    /// [`Self::upload_with_budget`].  Once accepted, a section's complete mesh remains in the
    /// ready queue when the absolute upload deadline expires; it is never sent back to the CPU
    /// mesher merely because the render thread ran out of time.
    pub fn upload_all_with_deadline_retaining_revisioned(
        &mut self,
        gpu: &Gpu,
        meshes: Vec<MeshTaskResult>,
        camera: DVec3,
        finish_time: Instant,
    ) -> UploadDeadlineResult {
        let mut input_revision_rejections = 0;
        for result in meshes {
            let pos = result.pos();
            if self.accept_mesh_revision(pos, result.revision()) {
                self.ready_uploads.push_latest_revisioned(result);
            } else {
                input_revision_rejections += 1;
            }
        }

        let mut deadline = UploadTaskDeadline::timed(finish_time);
        let mut uploaded_positions = Vec::new();
        let mut retained_revision_rejections = 0;
        while let Some(task) = self.ready_uploads.pop_task_if_permitted(&deadline) {
            let pos = task.mesh.pos;
            // `remove`/a newer result can invalidate a task after it was retained.  The check is
            // cheap and keeps the render-thread boundary defensive even when a caller interleaves
            // lifecycle operations between dispatches.
            if self
                .mesh_revisions
                .get(&pos)
                .is_some_and(|&current| current != task.revision)
                || self.invalid_mesh_revisions.contains(&pos)
            {
                retained_revision_rejections += 1;
                deadline.complete_task();
                continue;
            }
            // No GPU memory for the section (or a lost device): it stays first
            // in the ready queue for a later frame, and this frame uploads nothing more.
            if let Err(mesh) = self.upload_one_to_arena(gpu, task.mesh, camera) {
                self.ready_uploads.revisions.insert(pos, task.revision);
                self.ready_uploads.tasks.push_front(*mesh);
                break;
            }
            deadline.complete_task();
            uploaded_positions.push(pos);
        }

        let deadline_blocked = !self.ready_uploads.tasks.is_empty();
        let retained_ready_count = self.ready_uploads.len();

        UploadDeadlineResult {
            uploaded_positions,
            retained_positions: self.ready_uploads.positions(),
            input_revision_rejections,
            retained_revision_rejections,
            retained_ready_count,
            deadline_blocked,
        }
    }

    /// Upload one section into the shared vertex arena. java: `RenderChunk.rebuildChunk`'s upload,
    /// with the section's previous ranges returned to the arena when its old entry is replaced.
    /// Phase 3b: arena page 0 grew from `old` into `new` (same offsets). Point every slice that
    /// named `old` (resident sections, and the layers of the mesh being uploaded) at `new`, and
    /// drop every cached draw plan, since plans hold buffer references.
    fn repoint_arena_buffer(
        &mut self,
        old: &Rc<wgpu::Buffer>,
        new: &Rc<wgpu::Buffer>,
        pending: &mut [Option<LayerSlice>],
    ) {
        let slices = self
            .chunks
            .values_mut()
            .flat_map(|chunk| chunk.layers.iter_mut())
            .chain(pending.iter_mut())
            .flatten();
        for slice in slices {
            if Rc::ptr_eq(&slice.buffer, old) {
                slice.buffer = Rc::clone(new);
            }
        }
        for (layer, generation) in self.layer_generations.iter_mut().enumerate() {
            *generation = generation.wrapping_add(1);
            self.draw_plans[layer] = None;
        }
    }

    /// `Err(mesh)`when the GPU had no memory for it or the device is lost:
    /// nothing of it is resident and the caller keeps the mesh for a later frame. Boxed so the
    /// success path stays a plain `Ok(())` (the box is made only on a refusal).
    fn upload_one_to_arena(
        &mut self,
        gpu: &Gpu,
        mut mesh: ChunkMesh,
        camera: DVec3,
    ) -> Result<(), Box<ChunkMesh>> {
        mc_core::profile_scope!("terrain.upload");
        // java: `RenderChunk.rebuildChunk` sorts the translucent layer before it uploads.
        sort_translucent(&mut mesh.translucent, mesh.pos, camera);
        let mut max_quads = 0u32;
        for (_, buf) in mesh.layers() {
            max_quads = max_quads.max(buf.quad_count() as u32);
        }
        if !self.grow_indices(gpu, max_quads) {
            return Err(Box::new(mesh));
        }

        let stride = self.vertex_stride;
        let mut layers: [Option<LayerSlice>; LAYER_COUNT] = std::array::from_fn(|_| None);
        let mut bytes = 0u64;
        let mut full = false;
        for (layer, buf) in mesh.layers() {
            if buf.is_empty() {
                continue;
            }
            let packed;
            let contents: &[u8] = if stride == VERTEX_SIZE as u64 {
                buf.as_bytes()
            } else {
                packed = pack_bytes(buf.as_bytes());
                &packed
            };
            let len = contents.len() as u64;
            let allocated = self.arena.borrow_mut().allocate(gpu, len);
            let ArenaAllocation {
                page,
                buffer,
                offset,
                replaced,
            } = match allocated {
                Ok(allocation) => allocation,
                Err(ArenaFull { replaced }) => {
                    if let Some((old, new)) = replaced {
                        self.repoint_arena_buffer(&old, &new, &mut layers);
                    }
                    full = true;
                    break;
                }
            };
            if let Some((old, new)) = replaced {
                self.repoint_arena_buffer(&old, &new, &mut layers);
            }
            gpu.queue.write_buffer(&buffer, offset, contents);
            bytes += len;
            layers[layer_index(layer)] = Some(LayerSlice {
                buffer,
                offset,
                len,
                quads: buf.quad_count() as u32,
                stride,
                _lease: Some(ArenaLease {
                    arena: Rc::clone(&self.arena),
                    page,
                    offset,
                    len,
                }),
            });
        }
        if full {
            // The layers placed so far go back to the arena with their leases.
            drop(layers);
            return Err(Box::new(mesh));
        }
        // An all-air section is still recorded: it has no buffers to draw, but the culling walk
        // reads its visibility matrix on the way past.
        self.install_resident(
            mesh.pos,
            ChunkBuffers {
                layers,
                visibility: mesh.visibility,
                occlusion: mesh.occlusion,
                translucent_cpu: mesh.translucent,
            },
        );
        if bytes == 0 {
            self.residency.remove(mesh.pos);
        } else {
            self.residency.insert(mesh.pos, bytes);
        }
        Ok(())
    }

    fn upload_with_budget(
        &mut self,
        gpu: &Gpu,
        mut meshes: Vec<ChunkMesh>,
        camera: DVec3,
        budget: GpuBudget,
    ) -> Vec<ChunkSectionPos> {
        mc_core::profile_scope!("terrain.upload");
        if meshes.is_empty() {
            return Vec::new();
        }

        let mut max_quads = 0u32;
        for mesh in &mut meshes {
            // java: `RenderChunk.rebuildChunk` sorts the translucent layer before it uploads.
            sort_translucent(&mut mesh.translucent, mesh.pos, camera);
            for (_, buf) in mesh.layers() {
                max_quads = max_quads.max(buf.quad_count() as u32);
            }
        }
        if !self.grow_indices(gpu, max_quads) {
            // No index buffer for these quads: nothing is uploaded.
            return meshes.iter().map(|mesh| mesh.pos).collect();
        }

        let plan = UploadPlan::plan(&meshes, budget);

        // One `create_buffer_init` per layer that had any geometry — the whole point.
        let mut slices: HashMap<ChunkSectionPos, [Option<LayerSlice>; LAYER_COUNT]> =
            HashMap::with_capacity(meshes.len());
        // Normally the identity; the packed regression control scales offsets and lengths by
        // 20/28 while quantising the faithful CPU bytes on upload.
        let stride = self.vertex_stride;
        let rescale = |packed: u64| packed / VERTEX_SIZE as u64 * stride;
        // Sections whose batch the GPU refused. Empty (no allocation) unless one
        // was refused.
        let mut refused: HashSet<ChunkSectionPos> = HashSet::new();
        for batch in &plan.batches {
            let packed;
            let contents: &[u8] = if stride == VERTEX_SIZE as u64 {
                &batch.bytes
            } else {
                packed = pack_bytes(&batch.bytes);
                &packed
            };
            // The friend's AMD crash was here: `create_buffer_init` on a lost
            // device panics in its mapping ("Buffer with 'terrain batch' label is invalid").
            // Same GPU work, but a refusal (lost device, no memory, over the buffer limit) leaves
            // the batch's sections out instead.
            let created = crate::gpu_health::try_create_buffer_init(
                &gpu.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("terrain batch"),
                    contents,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            );
            let buffer = match created {
                Ok(buffer) => Rc::new(buffer),
                Err(error) => {
                    self.note_refused_upload(gpu, &error, contents.len());
                    refused.extend(batch.slices.iter().map(|slice| slice.pos));
                    continue;
                }
            };
            let li = layer_index(batch.layer);
            for slice in &batch.slices {
                let offset = rescale(slice.offset);
                assert_eq!(
                    offset % wgpu::COPY_BUFFER_ALIGNMENT,
                    0,
                    "an upload slice must be copy-aligned to be bindable"
                );
                let entry = slices
                    .entry(slice.pos)
                    .or_insert_with(|| std::array::from_fn(|_| None));
                entry[li] = Some(LayerSlice {
                    buffer: Rc::clone(&buffer),
                    offset,
                    len: rescale(slice.len),
                    quads: slice.quads,
                    stride,
                    _lease: None,
                });
            }
        }

        let deferred: HashSet<ChunkSectionPos> = plan.deferred.iter().copied().collect();
        let mut not_uploaded = plan.deferred;
        for mesh in meshes {
            if deferred.contains(&mesh.pos) {
                continue;
            }
            if refused.contains(&mesh.pos) {
                // Not resident (a resident older version stays as it was).
                not_uploaded.push(mesh.pos);
                continue;
            }
            let layers = slices
                .remove(&mesh.pos)
                .unwrap_or_else(|| std::array::from_fn(|_| None));
            let bytes: u64 = layers.iter().flatten().map(|s| s.len).sum();
            // An all-air section is still recorded: it has no buffers to draw, but the culling
            // walk reads its visibility matrix on the way past. A load or replacement cancels a
            // CPU resort snapshot of this section (see `install_resident`): the worker owns only
            // cloned CPU buffers, but the render-thread result must not write them into a newer
            // section allocation.
            self.install_resident(
                mesh.pos,
                ChunkBuffers {
                    layers,
                    visibility: mesh.visibility,
                    occlusion: mesh.occlusion,
                    translucent_cpu: mesh.translucent,
                },
            );
            if bytes == 0 {
                self.residency.remove(mesh.pos);
            } else {
                self.residency.insert(mesh.pos, bytes);
            }
        }
        not_uploaded
    }

    /// Log a refused terrain buffer: the first one of a kind at WARN with the
    /// allocator's numbers, later ones at DEBUG (an export can refuse thousands in a row), and
    /// record a lost device found this way so the frame loop sees it at once.
    fn note_refused_upload(
        &mut self,
        gpu: &Gpu,
        error: &crate::gpu_health::AllocError,
        len: usize,
    ) {
        if matches!(error, crate::gpu_health::AllocError::DeviceLost) {
            gpu.device_health().mark_lost(
                wgpu::DeviceLostReason::Unknown,
                "a terrain buffer came back invalid: the device was lost",
            );
        }
        self.refused_uploads += 1;
        if self.refused_uploads == 1 {
            tracing::warn!(
                %error,
                bytes = len,
                memory = %crate::gpu_health::GpuMemory::of(&gpu.device)
                    .map_or_else(String::new, |m| m.to_string()),
                "terrain: the GPU refused a vertex buffer; the section waits"
            );
        } else {
            tracing::debug!(%error, bytes = len, refused = self.refused_uploads, "terrain buffer refused");
        }
    }

    /// Terrain buffers the GPU refused since the renderer was made (0 on a
    /// healthy device). The export reads it to fail instead of writing a video with holes.
    #[must_use]
    pub fn refused_uploads(&self) -> u64 {
        self.refused_uploads
    }

    /// Drop a section's GPU buffers and cancel any ready upload for it.
    /// java: `RenderChunk.deleteGlResources` / `RenderChunk.stopCompileTask`.
    ///
    /// With batched uploads the buffer itself only dies once every section sharing it has gone,
    /// because they hold it through an `Rc`. That is the one real cost of batching: a single
    /// re-meshed section keeps its whole batch alive until its siblings are replaced too. It is
    /// bounded — a batch is one frame's worth of meshing, at most `DEFAULT_BUDGET` sections —
    /// and it is the same trade `RenderChunk`'s fixed display-list allocation made.
    pub fn remove(&mut self, pos: ChunkSectionPos) {
        self.ready_uploads.cancel(pos);
        if self.remove_resident(pos) {
            self.residency.remove(pos);
        }
        self.invalidate_mesh_revision(pos);
    }

    /// [`Self::remove`] for every position of a set, in the set's order.
    ///
    /// A world change (join, respawn into another dimension) removes every resident section in
    /// one frame. One at a time, each removal returns its arena ranges to a sorted free list in
    /// hash order, which fragments it: 2.4 ms at render distance 8 and 21.5 ms at 16 (measured,
    /// `nw_f5_probes::probe_world_change_removal_breakdown`), 87% of it in those releases. When
    /// the set is exactly the resident sections, this does once what the removals did each time
    /// (one walk/epoch/layer invalidation, every cached plan dropped, the resort list emptied, the
    /// walk grid refilled on its next use), and empties the arena in one step. Mesh-revision
    /// tombstones and residency are still written per position, exactly as `remove` writes them.
    /// Any other set takes the one-at-a-time path.
    pub fn remove_sections<S: std::hash::BuildHasher>(
        &mut self,
        positions: &HashSet<ChunkSectionPos, S>,
    ) {
        self.remove_sections_with(positions, true);
    }

    /// [`Self::remove_sections`] for a world that is going away, then forget
    /// every mesh-revision entry and tombstone (the bulk path skips writing the ones it would
    /// forget straight away).
    ///
    /// Only for a world change, after the client reset its mesh queue (`MeshQueue::reset_world`
    /// replaces the result channel, so no result meshed before can arrive any more) and called
    /// [`Self::cancel_all_pending_uploads`]. Every later result carries a revision drawn from the
    /// process-wide counter after the entries forgotten here, so it is accepted exactly as it
    /// would have been against them (a tombstone only rejects older revisions). Before this, the
    /// renderer kept an entry for every section position the session had ever seen, in every
    /// world (about 165 KB per new map at render distance 8, never freed). The maps keep their
    /// capacity (sized for the render distance), so the next world fills them
    /// without rehashing. `MC_W18_OC_PRUNE_REVISIONS=0` removes exactly as `remove_sections`.
    pub fn remove_world_sections<S: std::hash::BuildHasher>(
        &mut self,
        positions: &HashSet<ChunkSectionPos, S>,
    ) {
        if !prune_revisions_enabled() {
            self.remove_sections(positions);
            return;
        }
        self.remove_sections_with(positions, false);
        self.mesh_revisions.clear();
        self.invalid_mesh_revisions.clear();
    }

    /// [`Self::remove_sections`]; `tombstones: false` skips the bulk path's tombstone writes
    /// (only for [`Self::remove_world_sections`], which forgets them right after).
    fn remove_sections_with<S: std::hash::BuildHasher>(
        &mut self,
        positions: &HashSet<ChunkSectionPos, S>,
        tombstones: bool,
    ) {
        let every_resident = bulk_drop_enabled()
            && !self.chunks.is_empty()
            && positions.len() == self.chunks.len()
            && self.chunks.keys().all(|pos| positions.contains(pos));
        if !every_resident {
            for &pos in positions {
                self.remove(pos);
            }
            return;
        }
        for &pos in positions {
            self.ready_uploads.cancel(pos);
            self.residency.remove(pos);
            if tombstones {
                self.invalidate_mesh_revision(pos);
            }
        }
        self.resident_epoch = self.resident_epoch.wrapping_add(1);
        self.bump_walk_generation();
        for (layer, generation) in self.layer_generations.iter_mut().enumerate() {
            *generation = generation.wrapping_add(1);
            self.draw_plans[layer] = None;
        }
        self.translucent_resort_sections.clear();
        self.arena.borrow_mut().bulk_release = true;
        // Dropping the records drops their arena leases; their releases are skipped and the
        // pages reset together below.
        self.chunks.clear();
        self.arena.borrow_mut().finish_bulk_release();
        self.walk_grid.invalidate();
    }

    /// Cancel a completed mesh that has not reached the GPU yet.
    ///
    /// This is the renderer-side equivalent of `ChunkCompileTaskGenerator.finish()` cancelling
    /// the upload future when a section is unloaded or superseded. It is separate from
    /// [`Self::remove`] so a client can cancel a non-resident task without changing the loaded
    /// section generation.
    pub fn cancel_pending_upload(&mut self, pos: ChunkSectionPos) -> bool {
        let cancelled = self.ready_uploads.cancel(pos);
        self.invalidate_mesh_revision(pos);
        cancelled
    }

    /// Cancel every completed-but-not-uploaded task before the owning client world is replaced.
    ///
    /// A ready task is not represented in `uploaded_sections`, so a world teardown that only
    /// removes resident buffers would otherwise leave one old-world result to upload after the
    /// next dimension's first frame.
    pub fn cancel_all_pending_uploads(&mut self) -> usize {
        let positions = self.ready_uploads.positions();
        // Every task goes, so drop the queue whole instead of one O(n) cancel per task.
        self.ready_uploads = ReadyMeshQueue::default();
        for pos in &positions {
            self.invalidate_mesh_revision(*pos);
        }
        positions.len()
    }

    /// The sections whose finished meshes wait for upload: what
    /// [`Self::cancel_all_pending_uploads`] drops. A caller that cancels them to re-mesh queues
    /// these again (a section meshed for the first time has no buffer to be re-meshed from).
    #[must_use]
    pub fn pending_upload_positions(&self) -> Vec<ChunkSectionPos> {
        self.ready_uploads.positions()
    }

    /// The GPU vertex-memory cap. Defaults to [`GpuBudget::DEFAULT_BYTES`] (512 MiB).
    #[must_use]
    pub fn gpu_budget(&self) -> GpuBudget {
        self.budget
    }

    /// Change the cap. Takes effect on the next frame's eviction pass.
    pub fn set_gpu_budget(&mut self, budget: GpuBudget) {
        self.enforce_gpu_budget = true;
        self.budget = budget;
        self.residency = {
            let mut r = GpuResidency::new(budget);
            for (pos, chunk) in &self.chunks {
                let bytes: u64 = chunk.layers.iter().flatten().map(|s| s.len).sum();
                if bytes > 0 {
                    r.insert(*pos, bytes);
                }
            }
            r
        };
    }

    /// Resident sections, bytes, the cap and the eviction count — vanilla's `Mem:` line, but for
    /// terrain and true. See the S23 → S16 cross-section request.
    #[must_use]
    pub fn gpu_memory(&self) -> GpuMemoryReport {
        self.residency.report()
    }

    /// [`TerrainRenderer::gpu_memory`] plus what changed since this was last called, and resets
    /// the window counters.
    ///
    /// This is the instrument. The cumulative report cannot separate
    /// "the budget is evicting every window" from "the budget never fires and the queue is
    /// refilling for some other reason"; a windowed one can, and `peak_bytes` catches a working
    /// set that crosses the cap *inside* a window and is back under it when sampled.
    ///
    /// A caller with its own profiling cadence should use this. A caller with none gets the same
    /// data logged automatically whenever `MC_PROFILE` and `MC_PROFILE_INTERVAL` are set — see
    /// [`GPU_BUDGET_MB_ENV`] and `begin_frame` — so no caller change is required to read it.
    pub fn take_gpu_memory_window(&mut self) -> GpuMemoryWindow {
        let elapsed = self.memory_log_last.elapsed().as_secs_f64();
        let mut w = self.residency.take_window();
        w.seconds = elapsed;
        self.memory_log_last = std::time::Instant::now();
        w
    }

    /// How many distinct `wgpu::Buffer`s the resident sections' geometry lives in.
    ///
    /// This is the number the batching task exists to move: on the per-section path it is up to
    /// four per section, on the batched path it is four per upload call. Exposed because a
    /// claimed reduction that nothing counts is not a measurement.
    #[must_use]
    pub fn gpu_buffer_count(&self) -> usize {
        let mut seen: HashSet<usize> = HashSet::new();
        for chunk in self.chunks.values() {
            for slice in chunk.layers.iter().flatten() {
                seen.insert(Rc::as_ptr(&slice.buffer) as usize);
            }
        }
        seen.len()
    }

    /// `(cache hits, walks)` for the culling walk since this renderer was built.
    #[must_use]
    pub fn cull_cache_counts(&self) -> (u64, u64) {
        self.walk.cache_counts()
    }

    /// Sections the budget dropped since this was last called, and forgets them.
    ///
    /// **The caller must drain this every frame.** A section evicted here is no longer resident
    /// and no longer known to this renderer, so a caller that tracks what it uploaded (which
    /// `mc-client` does, for its leak check and for rebuilding phases after a pack reload) would
    /// otherwise believe a dropped section is still on the GPU — and never ask for it again,
    /// which is a permanent hole rather than a one-frame one.
    ///
    /// Re-queueing what comes out of here is the right response, and does **not** thrash:
    /// eviction drops the *farthest* sections and
    /// [`MeshQueue::drain`](super::jobs::MeshQueue::drain) rebuilds the *nearest* first, so the
    /// working set settles on "the nearest sections that fit". The wasted work in the meantime is
    /// bounded by one frame's meshing budget, and only ever paid by content pathological enough
    /// to exceed [`GpuBudget::DEFAULT_BYTES`] in the first place.
    pub fn take_evicted(&mut self) -> Vec<ChunkSectionPos> {
        std::mem::take(&mut self.evicted)
    }

    /// Regenerate and upload the lightmap from S20's [`LightmapParams`].
    ///
    /// This is the whole of `EntityRenderer.updateLightmap` reaching terrain: gamma, night
    /// vision, the torch flicker, the lightning-flash bypass, the End's override and the
    /// Nether's ambient floor all arrive through here. Cheap enough to call every frame (256
    /// texels), which is what vanilla does.
    pub fn set_lightmap_params(&mut self, gpu: &Gpu, params: LightmapParams) {
        self.sky_brightness = params.sun_brightness;
        if self.uploaded_lightmap_params == Some(params) {
            return;
        }
        self.uploaded_lightmap_params = Some(params);
        self.lightmap_params = params;
        self.upload_lightmap(gpu, &Lightmap::generate_cached(params));
    }

    /// Upload an already-generated [`Lightmap`], for a caller that shares one table across
    /// terrain, entities and particles the way vanilla shares one texture unit.
    pub fn set_lightmap(&mut self, gpu: &Gpu, lightmap: &Lightmap) {
        self.uploaded_lightmap_params = None;
        self.upload_lightmap(gpu, lightmap);
    }

    fn upload_lightmap(&mut self, gpu: &Gpu, lightmap: &Lightmap) {
        self.lightmap_source_rgba = lightmap.rgba.clone();
        let upload_rgba = terrain_upload_rgba(&self.lightmap_source_rgba, self.anaglyph);
        upload_lightmap_texture(gpu, &self.lightmap_texture, &upload_rgba);
    }

    /// The per-draw fog term, or `None` (the default) for no fog.
    ///
    /// java: `EntityRenderer.setupFog`, whose `GL_FOG` applied to every terrain draw. This is
    /// the fix for the limitation [`crate::sky::fog::FogPhase`] documents: that pass is
    /// screen-space, so it can only fog the nearest **depth-writing** surface and gets water,
    /// stained glass and ice wrong. `FogState` is S20's, computed by
    /// [`FogState::compute`](crate::sky::fog::FogState::compute), and the uniform packing here
    /// is deliberately identical to `sky/fog.wgsl`'s.
    ///
    /// **Left off by default on purpose.** A caller that registers `FogPhase` *and* sets this
    /// fogs opaque terrain twice; the two are alternatives, not layers. Wiring the choice is
    /// S21's call — see the new cross-section request.
    pub fn set_fog(&mut self, fog: Option<FogState>) {
        self.fog = fog;
    }

    /// The fog term currently applied to terrain draws.
    #[must_use]
    pub fn fog(&self) -> Option<FogState> {
        self.fog
    }

    /// Fog the TRANSLUCENT layer per draw, each fragment at its own
    /// distance, with `distance`'s metric ("layers of
    /// translucent blocks aren't one to one with vanilla").
    ///
    /// java: `renderWorldPass` calls `setupFog(0)` right before
    /// `renderBlockLayer(TRANSLUCENT)` and only disables `GL_FOG` after it, so water, stained
    /// glass and ice fade into the fog at their own depth and each layer of a stack is fogged on
    /// its own before it blends. A host that fogs everything else with the screen-space
    /// [`crate::sky::fog::FogPhase`] must register that pass *before* this layer
    /// ([`crate::sky::fog::FogPhase::BEFORE_TRANSLUCENT_ORDER`]): translucent terrain writes no
    /// depth, so a later screen-space pass fogs a window at the depth of what is behind it, and
    /// not at all against the sky. Pass the screen-space pass's metric so the two agree.
    ///
    /// `None` (the default) leaves the layer unfogged here. Ignored while [`Self::set_fog`] fogs
    /// every layer.
    pub fn set_translucent_fog(&mut self, fog: Option<(FogState, FogDistance)>) {
        self.translucent_fog = fog;
    }

    /// The fog term currently applied to the TRANSLUCENT layer alone.
    #[must_use]
    pub fn translucent_fog(&self) -> Option<(FogState, FogDistance)> {
        self.translucent_fog
    }

    /// Number of sections resident on the GPU.
    #[must_use]
    pub fn loaded_sections(&self) -> usize {
        self.chunks.len()
    }

    /// Iterate the exact section identities whose geometry is resident on the GPU.
    ///
    /// The iterator borrows the renderer and traverses the backing map directly: it allocates
    /// nothing and does not impose an order. Diagnostic callers that need deterministic output
    /// should copy and sort the yielded positions at their own sampling boundary.
    pub fn resident_section_positions(&self) -> impl Iterator<Item = ChunkSectionPos> + '_ {
        self.chunks.keys().copied()
    }

    /// Iterate the ordered section identities selected by the latest culling walk.
    ///
    /// This is the same order used by the layer submission loops. The iterator is a read-only
    /// view over retained `self.visible` state, so it allocates nothing and preserves the actual
    /// near-to-far order (with the translucent pass applying its existing reverse index).
    ///
    /// This reports every *resident* section the last walk reached, all-air records included,
    /// because callers use it to observe reachability through air. The per-frame draw list
    /// holds only sections with geometry, so this is derived from the walk itself (a cold path:
    /// one map probe per walked section, nothing per frame).
    pub fn visible_section_positions(&self) -> impl Iterator<Item = ChunkSectionPos> + '_ {
        self.walk
            .sections()
            .iter()
            .map(|section| section.pos)
            .filter(|pos| self.chunks.contains_key(pos))
    }

    /// Number of completed mesh tasks waiting in the retained ready-upload queue.
    ///
    /// This is the queue that survives an expired upload deadline; it is distinct from the
    /// worker-side mesh queue and therefore reports work that is ready to upload rather than work
    /// that still needs CPU meshing.
    #[must_use]
    pub fn pending_ready_upload_count(&self) -> usize {
        self.ready_uploads.len()
    }

    /// Last frame's counters, for the F3 overlay.
    #[must_use]
    pub fn stats(&self) -> TerrainStats {
        self.stats
    }

    /// Whether this renderer is currently using native grouped indirect submission.
    #[must_use]
    pub fn multi_draw_indirect_enabled(&self) -> bool {
        self.multi_draw_indirect
    }

    /// Select the S23 indirect path or the per-section control path. Enabling is harmless on an
    /// adapter without native support: the renderer keeps the direct fallback selected.
    pub fn set_multi_draw_indirect(&mut self, enabled: bool) {
        self.multi_draw_indirect = enabled && self.multi_draw_indirect_supported;
        self.arena.borrow_mut().grow_enabled = self.multi_draw_indirect;
    }

    /// Test seam for order-sensitivity controls: submit `TRANSLUCENT` in the walk's own
    /// near-to-far order instead of vanilla's far-to-near (see [`submission_index`]). This is
    /// wrong on purpose, so a pixel test can show its fixture would catch a mis-ordered
    /// submission. Nothing outside tests sets it.
    #[doc(hidden)]
    pub fn set_translucent_near_to_far_for_test(&mut self, near_to_far: bool) {
        self.translucent_near_to_far = near_to_far;
        self.draw_plans[layer_index(RenderLayer::Translucent)] = None;
    }

    /// Test seam for arena growth (phase 3b): the size of a newly opened arena page and the size
    /// page 0 may grow to. Takes effect for pages opened afterwards, so set it before uploading.
    #[doc(hidden)]
    pub fn set_arena_page_bytes_for_test(&mut self, page_bytes: u64, grow_limit: u64) {
        let mut arena = self.arena.borrow_mut();
        arena.page_bytes = page_bytes;
        arena.grow_limit = grow_limit;
    }

    /// `(rebuilds, cache hits)` for the per-layer S23 submission plans.
    #[must_use]
    pub fn draw_plan_cache_counts(&self) -> (u64, u64) {
        (self.draw_plan_rebuilds, self.draw_plan_cache_hits)
    }

    fn compute_visible_signature(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.visible.len().hash(&mut hasher);
        for (info, slot) in &self.visible {
            info.pos.hash(&mut hasher);
            slot.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// The visibility matrix of a resident section, for the culling walk.
    #[must_use]
    pub fn visibility(&self, pos: ChunkSectionPos) -> Option<SetVisibility> {
        self.chunks.get(&pos).map(|c| c.visibility)
    }

    /// Invalidate the culling walk only: the authoritative world-column set changed, or a
    /// resident change reached what the walk reads (see [`ResidentChange`]).
    fn bump_walk_generation(&mut self) {
        self.chunk_generation = self.chunk_generation.wrapping_add(1);
    }

    /// Insert or replace one resident record and invalidate exactly what the change can reach.
    fn install_resident(&mut self, pos: ChunkSectionPos, buffers: ChunkBuffers) {
        // One capacity compare per install once the maps are sized.
        self.reserve_section_maps();
        let change = ResidentChange::between(
            self.chunks.get(&pos).map(ChunkBuffers::resident_key),
            Some(buffers.resident_key()),
        );
        self.apply_resident_change(pos, change);
        // Dropping the replaced record returns its arena ranges; every plan that could name
        // them was dropped above.
        self.chunks.insert(pos, buffers);
        self.note_walk_section(pos);
    }

    /// Size the per-section maps for the whole render distance before they
    /// fill, so loading a world never rehashes them mid-load.
    ///
    /// `chunks` holds every resident section inline (~850 bytes each, most of it the 512-byte
    /// occlusion bits), so each doubling copied the whole map into freshly faulted memory in one
    /// frame: 2.4-3.4 ms at 3,585 sections (render distance 8, every SkyWars join; the map keeps
    /// its capacity across worlds, so it is the session's first join) and 7.3-7.9 ms at 14,337
    /// sections (render distance 16), the largest frame of the whole terrain load. Reserving
    /// while the map is still empty costs only the allocation: pages are touched by the inserts
    /// that would have touched them anyway, spread over the load. A render-distance increase
    /// reserves once at the next install, replacing the later doublings with one smaller copy.
    ///
    /// Nothing observable changes: these maps are hashed with a per-process random seed, so no
    /// output may depend on their iteration order, and lookups do not depend on capacity.
    fn reserve_section_maps(&mut self) {
        if !section_reserve_enabled() {
            return;
        }
        let target = section_reserve_target(self.render_distance);
        if self.chunks.capacity() >= target {
            return;
        }
        self.chunks.reserve(target - self.chunks.len());
        self.mesh_revisions
            .reserve(target.saturating_sub(self.mesh_revisions.len()));
        self.invalid_mesh_revisions
            .reserve(target.saturating_sub(self.invalid_mesh_revisions.len()));
        self.residency.reserve(target);
    }

    /// Remove one resident record (unload, cache removal or eviction), invalidating exactly what
    /// it can reach. Returns whether a record was resident.
    fn remove_resident(&mut self, pos: ChunkSectionPos) -> bool {
        let change =
            ResidentChange::between(self.chunks.get(&pos).map(ChunkBuffers::resident_key), None);
        self.apply_resident_change(pos, change);
        let removed = self.chunks.remove(&pos).is_some();
        self.note_walk_section(pos);
        removed
    }

    fn apply_resident_change(&mut self, pos: ChunkSectionPos, change: ResidentChange) {
        if change.walk || change.layers != 0 {
            self.resident_epoch = self.resident_epoch.wrapping_add(1);
        }
        if change.walk {
            self.bump_walk_generation();
        }
        for (layer, generation) in self.layer_generations.iter_mut().enumerate() {
            if change.layers & (1 << layer) != 0 {
                *generation = generation.wrapping_add(1);
                // Cached plans own buffer references. Drop them at invalidation, including when
                // the final section is removed and an empty frame never enters the rebuild.
                self.draw_plans[layer] = None;
            }
        }
        // Only a change to a section the in-flight resort cloned can make its result stale, and
        // only that section's part of it: the worker sorted a clone of a mesh that is now gone.
        // java: a rebuild cancels just that chunk's own pending resort task
        // (`RenderChunk.makeCompileTaskChunk` -> `finishCompileTask`); the other chunks' resorts
        // run on, and `prevRenderSort*` is written only by the >1-block check in
        // `RenderGlobal.renderBlockLayer`, so the job keeps running and `last_sort_pos` stays
        // put. The poll writes only the sections still listed. The uploaded mesh itself was
        // sorted for the current camera on its way in.
        self.translucent_resort_sections
            .retain(|section| *section != pos);
    }

    fn cancel_translucent_resort(&mut self) {
        if let Some(cancelled) = self.translucent_resort_cancel.take() {
            cancelled.store(true, Ordering::Relaxed);
        }
        self.translucent_resort = None;
        self.translucent_resort_sections.clear();
    }

    /// Poll a completed CPU resort and perform its GPU write on the render thread only. A
    /// worker result carries the version of the job it answers, and only the sections the job
    /// still owns are written: one replaced, removed or evicted since scheduling has left
    /// `translucent_resort_sections`, so nothing stale is written.
    fn poll_translucent_resort(&mut self, gpu: &Gpu) {
        let Some(receiver) = self.translucent_resort.take() else {
            return;
        };
        match receiver.try_recv() {
            Ok(result) => {
                self.translucent_resort_cancel.take();
                if !accepts_translucent_resort(
                    self.translucent_resort_version,
                    result.resident_version,
                ) {
                    self.translucent_resort_sections.clear();
                    return;
                }
                for (pos, buffer) in result.sections {
                    // Still listed means unchanged since the worker cloned it: every resident
                    // change goes through `apply_resident_change`, which unlists the section.
                    if !self.translucent_resort_sections.contains(&pos) {
                        continue;
                    }
                    let Some(chunk) = self.chunks.get_mut(&pos) else {
                        continue;
                    };
                    let Some(slice) = &chunk.layers[3] else {
                        continue;
                    };
                    let packed;
                    let bytes: &[u8] = if slice.stride == VERTEX_SIZE as u64 {
                        buffer.as_bytes()
                    } else {
                        packed = pack_bytes(buffer.as_bytes());
                        &packed
                    };
                    debug_assert_eq!(bytes.len() as u64, slice.len);
                    // This is deliberately the only GPU write in the worker-result path. The
                    // worker sorted a cloned Buffer and never had access to `gpu` or `queue`.
                    gpu.queue.write_buffer(&slice.buffer, slice.offset, bytes);
                    chunk.translucent_cpu = buffer;
                }
                self.translucent_resort_sections.clear();
            }
            Err(TryRecvError::Empty) => {
                self.translucent_resort = Some(receiver);
            }
            Err(TryRecvError::Disconnected) => {
                self.translucent_resort_cancel.take();
                self.translucent_resort_sections.clear();
            }
        }
    }

    /// Schedule a bounded visible-only translucent resort on a CPU worker. The actual buffer
    /// writes are deferred to [`Self::poll_translucent_resort`] on the render thread.
    fn schedule_translucent_resort(&mut self, camera: DVec3) {
        if self.translucent_resort.is_some() {
            return;
        }
        let version = self.translucent_resort_version.wrapping_add(1);
        let Some(job) = translucent_resort_job(
            self.visible.iter().map(|(info, _)| info),
            &self.chunks,
            version,
            camera,
        ) else {
            self.last_sort_pos = camera;
            return;
        };
        let (sender, receiver) = mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        self.translucent_resort_version = version;
        self.translucent_resort_sections.clear();
        self.translucent_resort_sections
            .extend(job.sections.iter().map(|(pos, _)| *pos));
        self.translucent_resort = Some(receiver);
        self.translucent_resort_cancel = Some(Arc::clone(&cancelled));
        self.last_sort_pos = camera;
        // Cancellation drops the receiver and sets the flag; the worker skips the rest of the
        // job at its next section boundary without holding renderer state.
        submit_translucent_resort(TranslucentResortTask {
            job,
            cancelled,
            result: sender,
        });
    }

    /// Re-sort every resident translucent layer for a new camera position and re-upload it.
    /// java: `RenderGlobal.renderBlockLayer`'s deferred `resortTransparency`.
    ///
    /// This explicit API remains synchronous for callers that request an immediate full-resident
    /// update; the live frame path uses the bounded worker above.
    pub fn resort_translucent(&mut self, gpu: &Gpu, camera: DVec3) {
        self.cancel_translucent_resort();
        for (pos, chunk) in &mut self.chunks {
            if chunk.translucent_cpu.is_empty() {
                continue;
            }
            sort_translucent(&mut chunk.translucent_cpu, *pos, camera);
            if let Some(slice) = &chunk.layers[3] {
                // Writes into this section's own run of the shared batch buffer. The sort only
                // ever permutes whole quads, so the length is unchanged and no neighbour's
                // vertices can be trampled.
                let packed;
                let bytes: &[u8] = if slice.stride == VERTEX_SIZE as u64 {
                    chunk.translucent_cpu.as_bytes()
                } else {
                    packed = pack_bytes(chunk.translucent_cpu.as_bytes());
                    &packed
                };
                debug_assert_eq!(bytes.len() as u64, slice.len);
                gpu.queue.write_buffer(&slice.buffer, slice.offset, bytes);
            }
        }
        self.last_sort_pos = camera;
    }

    /// Per-frame setup, run by the solid phase before anything is drawn.
    fn begin_frame(&mut self, gpu: &Gpu, cam: &Camera) {
        mc_core::profile_scope!("terrain.begin_frame");

        // Poll before culling and eviction so a completed worker result is installed by the
        // render thread, while the frame's resident-set mutations can invalidate the next job.
        self.poll_translucent_resort(gpu);

        // Enforce the optional GPU memory cap before the walk, so an evicted section cannot be selected
        // this frame. Returns empty and costs nothing while under budget, which is every frame
        // of ordinary play — the cap is a backstop against pathological content, not a working
        // set target.
        self.log_memory_window();

        let dropped = if self.enforce_gpu_budget {
            self.residency.evict_to_budget(cam.pos)
        } else {
            Vec::new()
        };
        // Churn diagnostic. The leading hypothesis is that the cap bites at rd 16 and
        // the client's re-queue turns that into a loop — but nothing in the client has ever said
        // whether the cap bites *at all*, so the hypothesis could not be told apart from a
        // dirty-set defect from the outside. Logged once per renderer, at the first eviction, with
        // the numbers that decide it. Silence in a session's log is now evidence too: it means the
        // working set stayed inside the budget and eviction churn cannot be the mechanism.
        if !dropped.is_empty() && !self.warned_evicting {
            self.warned_evicting = true;
            let r = self.residency.report();
            tracing::info!(
                sections_dropped = dropped.len(),
                resident_sections = r.sections,
                resident_bytes = r.bytes,
                limit_bytes = r.limit,
                "terrain gpu budget evicted for the first time; \
                 raise it with {GPU_BUDGET_MB_ENV}"
            );
        }
        for pos in dropped {
            if self.remove_resident(pos) {
                self.ready_uploads.cancel(pos);
                self.invalidate_mesh_revision(pos);
                self.evicted.push(pos);
            }
        }

        self.prepare_culling_walk(cam);
        // A still camera with nothing resident changing (AFK, a GUI open over the world, a
        // spectator pause) re-probes nothing: same walk, same records, same camera bits.
        let draw_list_key = (self.walk.cache_counts().1, self.resident_epoch);
        if self.draw_list_key == Some(draw_list_key) {
            self.finish_begin_frame(gpu, cam, false);
            return;
        }
        let walk = std::mem::take(&mut self.walk);
        {
            let selected = walk.sections();

            self.visible.clear();
            self.walk_reached_resident = false;
            self.offset_scratch.clear();
            self.offset_scratch.reserve(selected.len());
            for info in selected {
                // #2: an all-air record stays resident for the walk, which has already read its
                // visibility matrix to get here, but it can never produce a draw. Keeping it out
                // of the draw list keeps it out of the signature, the offsets upload, every plan
                // rebuild and the resort scan. The relative order of the sections that remain is
                // unchanged, and each keeps its own offset through its own slot, so the submitted
                // draws (including the reversed translucent order) are identical.
                let Some(chunk) = self.chunks.get(&info.pos) else {
                    continue;
                };
                self.walk_reached_resident = true;
                if chunk.layer_mask() == 0 {
                    continue;
                }
                let origin = info.pos.origin();
                let rel = cam.relative(DVec3::new(
                    f64::from(origin.x),
                    f64::from(origin.y),
                    f64::from(origin.z),
                ));
                let slot = self.offset_scratch.len() as u32;
                self.offset_scratch.push(ChunkOffset {
                    offset: [rel.x, rel.y, rel.z, 0.0],
                });
                self.visible.push((*info, slot));
            }
        }
        self.walk = walk;
        self.draw_list_key = Some(draw_list_key);
        self.finish_begin_frame(gpu, cam, true);
    }

    /// Run this frame's culling walk. java: `RenderGlobal.setupTerrain`, which vanilla runs before
    /// `renderEntities` reads `renderInfos`. The client calls this before its entity passes, and
    /// the terrain pass then reuses the cached answer.
    pub fn prepare_culling_walk(&mut self, cam: &Camera) {
        let frustum = Frustum::from_camera(cam);
        let render_distance = self.render_distance;
        // Java retains render slots after a world-column unload and rebuilds them as
        // visible-through air. Authoritative world membership supplies that settled-air answer
        // without retaining renderer tombstones; loaded-but-unmeshed sections still return None
        // and therefore remain DUMMY hard stops. With no authority, retain the legacy missing-
        // camera frustum fallback and ordinary None semantics.
        let camera_section = BlockPos::from_vec3(cam.pos).section_pos();
        let missing_camera =
            (0..16).contains(&camera_section.y) && !self.chunks.contains_key(&camera_section);
        // With authoritative world membership, an absent camera column is an all-air source and
        // the normal smart walk remains safe. The legacy renderer retains its conservative
        // frustum fallback when no world set was supplied.
        let smart_culling = if self.world_columns.is_some() {
            self.smart_culling
        } else {
            self.smart_culling && !missing_camera
        };
        let spectator = self.player_spectator;
        let generation = self.chunk_generation;
        // #7: the walk never leaves the render-distance square around the camera's column, so
        // centring the dense index there makes every per-section query an array load. O(1)
        // unless the camera changed column (one strip) or the render distance changed.
        let (chunks, world_columns) = (&self.chunks, self.world_columns.as_ref());
        self.walk_grid
            .recenter(camera_section.chunk_pos(), render_distance, |col, cells| {
                fill_walk_column(chunks, world_columns, col, cells);
            });
        let source = GridSectionSource {
            grid: &self.walk_grid,
            inner: AuthoritativeSectionSource {
                chunks: &self.chunks,
                world_columns: self.world_columns.as_ref(),
            },
            camera_memo: &self.camera_reach_memo,
        };
        self.walk.run_with_view(
            &source,
            &frustum,
            cam.pos,
            Some(cam.look_vec()),
            render_distance,
            smart_culling,
            spectator,
            generation,
        );
    }

    /// Whether this frame's walk reached `pos`, which gates the entity and tile entity passes the
    /// way `renderInfos` does. True before the first walk, so a caller that draws without a
    /// terrain walk is unaffected.
    #[must_use]
    pub fn section_reached_by_walk(&self, pos: ChunkSectionPos) -> bool {
        self.walk.reached(pos).unwrap_or(true)
    }

    /// The rest of the per-frame setup, once the walk's result is in `self.visible`.
    /// `rebuilt` is false when `begin_frame` reused last frame's draw list, whose signature and
    /// uploaded offsets are then still exact.
    fn finish_begin_frame(&mut self, gpu: &Gpu, cam: &Camera, rebuilt: bool) {
        if rebuilt {
            self.visible_signature = self.compute_visible_signature();

            let offsets_len = self.offset_scratch.len() as u32;
            if offsets_len > self.offsets_capacity {
                self.offsets_capacity = offsets_len.next_power_of_two();
                self.offsets = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("terrain chunk offsets"),
                    size: u64::from(self.offsets_capacity)
                        * std::mem::size_of::<ChunkOffset>() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            if !self.offset_scratch.is_empty() {
                gpu.write_buffer_staged(
                    &self.offsets,
                    0,
                    bytemuck::cast_slice(&self.offset_scratch),
                );
            }
        }

        // The time of day reaches terrain through the lightmap, so a moved `sky_brightness`
        // means a regenerated table. Compared rather than unconditionally regenerated because
        // it is otherwise a per-frame 256-texel upload for nothing.
        if self.sky_brightness != self.lightmap_params.sun_brightness {
            let params = LightmapParams {
                sun_brightness: self.sky_brightness,
                ..self.lightmap_params
            };
            self.lightmap_params = params;
            self.uploaded_lightmap_params = Some(params);
            self.upload_lightmap(gpu, &Lightmap::generate(params));
        }

        for (i, (buf, write)) in self
            .uniforms
            .iter()
            .zip(self.uniform_writes.iter_mut())
            .enumerate()
        {
            // The translucent layer's own fog, unless `set_fog` fogs
            // every layer. Both `None` (the default): the old uniform, byte for byte.
            let uniform = match (self.fog, self.translucent_fog) {
                (None, Some((fog, distance))) if i == layer_index(RenderLayer::Translucent) => {
                    TerrainUniform::with_distance(layer_alpha_cutoff(i), Some(fog), distance)
                }
                (fog, _) => TerrainUniform::new(layer_alpha_cutoff(i), fog),
            };
            write.write(gpu, buf, bytemuck::bytes_of(&uniform));
        }

        self.stats = TerrainStats {
            loaded_sections: self.chunks.len(),
            visible_sections: self.visible.len(),
            ..TerrainStats::default()
        };

        if needs_resort(self.last_sort_pos, cam.pos) {
            self.schedule_translucent_resort(cam.pos);
        }
    }

    fn ensure_indirect_capacity(&mut self, gpu: &Gpu, layer: usize, count: usize) {
        if count <= usize::try_from(self.indirect_capacity[layer]).unwrap_or(usize::MAX) {
            return;
        }
        let capacity = count.next_power_of_two().max(1);
        self.indirect_capacity[layer] = u32::try_from(capacity).expect("visible groups fit u32");
        self.indirect_buffers[layer] = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain indirect draw arguments"),
            size: (capacity * std::mem::size_of::<DrawIndexedIndirectArgs>()) as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    fn draw_layer(
        &mut self,
        gpu: &Gpu,
        out: LayerPass<'_>,
        layers: &[RenderLayer],
        label: &'static str,
    ) {
        // Keyed on what the walk reached rather than on the draw list, so a view of nothing but
        // all-air records still opens (and, with MSAA, resolves) exactly the passes it always did.
        if !self.walk_reached_resident {
            return;
        }

        // Prepare (or reuse) each layer's commands before opening the pass. Each group is a
        // maximal run of sections sharing one uploaded vertex buffer; the cache key makes the
        // queue write conditional on visibility, resident geometry, and the direct/indirect A/B
        // mode actually changing.
        let visible_signature = self.visible_signature;
        for &layer in layers {
            let li = layer_index(layer);
            let layer_generation = self.layer_generations[li];
            // Translucent too (#42/TDRAW-5): draws inside one multi-draw execute in command order
            // (Vulkan primitive order, D3D12 `ExecuteIndirect`, and wgpu's GL/Metal loops), and
            // this plan writes translucent commands far-to-near via `submission_index` and groups
            // only consecutive same-buffer runs, so blending sees the same sequence as the direct
            // path.
            let use_indirect = self.multi_draw_indirect;
            let cache_valid = self.draw_plans[li].as_ref().is_some_and(|plan| {
                plan.visible_signature == visible_signature
                    && plan.layer_generation == layer_generation
                    && plan.use_indirect == use_indirect
            });
            if cache_valid {
                self.draw_plan_cache_hits += 1;
                continue;
            }

            let mut items = Vec::new();
            let mut groups = Vec::new();
            let mut commands = Vec::new();
            let mut quads = 0usize;
            let near_to_far = self.translucent_near_to_far && layer == RenderLayer::Translucent;
            for draw_position in 0..self.visible.len() {
                let visible_index = if near_to_far {
                    draw_position
                } else {
                    submission_index(draw_position, self.visible.len(), layer)
                };
                let (info, slot) = &self.visible[visible_index];
                let Some(chunk) = self.chunks.get(&info.pos) else {
                    continue;
                };
                let Some(slice) = &chunk.layers[li] else {
                    continue;
                };
                let item = DrawItem {
                    buffer: slice.buffer.clone(),
                    quads: slice.quads,
                    base_vertex: slice.base_vertex(),
                    first_instance: *slot,
                };
                quads += slice.quads as usize;
                if use_indirect {
                    let command_start = commands.len();
                    let same_buffer = groups.last().is_some_and(|group: &IndirectGroup| {
                        Rc::ptr_eq(&group.buffer, &item.buffer)
                            && group.command_start + group.command_count == command_start
                    });
                    if same_buffer {
                        groups.last_mut().unwrap().command_count += 1;
                    } else {
                        groups.push(IndirectGroup {
                            buffer: item.buffer.clone(),
                            command_start,
                            command_count: 1,
                        });
                    }
                    commands.push(DrawIndexedIndirectArgs {
                        index_count: item.quads * 6,
                        instance_count: 1,
                        first_index: 0,
                        base_vertex: item.base_vertex,
                        first_instance: item.first_instance,
                    });
                }
                items.push(item);
            }
            if use_indirect {
                self.ensure_indirect_capacity(gpu, li, commands.len());
                if !commands.is_empty() {
                    gpu.write_buffer_staged(
                        &self.indirect_buffers[li],
                        0,
                        bytemuck::cast_slice(&commands),
                    );
                }
            }
            self.draw_plans[li] = Some(LayerDrawPlan {
                visible_signature,
                layer_generation,
                use_indirect,
                items,
                groups,
                quads,
            });
            self.draw_plan_rebuilds += 1;
        }

        // One `Option` test while no shader pack was ever attached. A running
        // pack's pipelines take precedence over AF's (the pack shader samples the atlas itself).
        let pack = self.prepare_pack(gpu);
        let pack_scene = if pack && layers == [RenderLayer::Translucent] {
            self.take_pack_scene_group(gpu)
        } else {
            None
        };
        // One bool test while the Anisotropic Filtering mod is off.
        let aniso = !pack && self.prepare_aniso(gpu);
        // This layer's own pass (dropped, so ended, when this returns), or
        // the frame's shared world pass. `forget_lifetime` changes nothing that is recorded.
        let mut own;
        let pass: &mut wgpu::RenderPass<'static> = match out {
            LayerPass::Shared(pass) => pass,
            LayerPass::Open { enc, target } => {
                own = if pack_scene.is_some() {
                    Self::begin_pack_scene_pass(gpu, enc, target, label)
                } else {
                    gpu.begin_pass(enc, target, label, DepthAttachment::Shared)
                }
                .forget_lifetime();
                &mut own
            }
        };
        pass.set_index_buffer(self.index_buffer.slice(..), self.index_format);
        for &layer in layers {
            let li = layer_index(layer);
            let plan = self.draw_plans[li].as_ref().unwrap();
            self.stats.draw_items += plan.items.len();
            self.stats.quads += plan.quads;
            let pack_pipeline = if pack {
                self.pack.pipeline(li, pack_scene.is_some())
            } else {
                None
            };
            let aniso_pipeline = if aniso { self.aniso.pipeline(li) } else { None };
            pass.set_pipeline(
                pack_pipeline
                    .or(aniso_pipeline)
                    .unwrap_or(&self.pipelines[li]),
            );
            pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
            pass.set_bind_group(1, &self.bind_groups[li], &[]);
            if let (Some(_), Some(group)) = (pack_pipeline, self.pack.group()) {
                pass.set_bind_group(2, group, &[]);
                if let Some(scene_group) = &pack_scene {
                    pass.set_bind_group(3, scene_group, &[]);
                }
            } else if let (Some(_), Some(group)) = (aniso_pipeline, self.aniso.group()) {
                pass.set_bind_group(2, group, &[]);
            }

            pass.set_vertex_buffer(1, self.offsets.slice(..));
            self.stats.vertex_buffer_binds += 1;
            if plan.use_indirect {
                for group in &plan.groups {
                    pass.set_vertex_buffer(0, group.buffer.slice(..));
                    self.stats.vertex_buffer_binds += 1;
                    pass.multi_draw_indexed_indirect(
                        &self.indirect_buffers[li],
                        (group.command_start * std::mem::size_of::<DrawIndexedIndirectArgs>())
                            as u64,
                        group.command_count as u32,
                    );
                    self.stats.draw_calls += 1;
                }
            } else {
                let mut bound: Option<usize> = None;
                for item in &plan.items {
                    let id = Rc::as_ptr(&item.buffer) as usize;
                    if bound != Some(id) {
                        pass.set_vertex_buffer(0, item.buffer.slice(..));
                        self.stats.vertex_buffer_binds += 1;
                        bound = Some(id);
                    }
                    pass.draw_indexed(
                        0..item.quads * 6,
                        item.base_vertex,
                        item.first_instance..item.first_instance + 1,
                    );
                    self.stats.draw_calls += 1;
                }
            }
        }
    }

    /// The index format the draw loop is currently binding. `Uint16` unless some layer has ever
    /// exceeded [`MAX_U16_QUADS`]. Exposed so `s23_batching.rs` can assert the fallback fires —
    /// without that assertion T1.2 would be a latent corruption bug rather than a saving.
    #[must_use]
    pub fn index_format(&self) -> wgpu::IndexFormat {
        self.index_format
    }

    /// Log one [`GpuMemoryWindow`] if the profiler's interval has elapsed. No-op otherwise, and
    /// no-op entirely unless `MC_PROFILE` and `MC_PROFILE_INTERVAL` are both set.
    fn log_memory_window(&mut self) {
        let Some(interval) = self.memory_log_interval else {
            return;
        };
        if self.memory_log_last.elapsed() < interval {
            return;
        }
        let w = self.take_gpu_memory_window();
        tracing::info!(
            seconds = format_args!("{:.1}", w.seconds),
            resident_sections = w.now.sections,
            resident_mib = format_args!("{:.1}", w.now.bytes as f64 / (1024.0 * 1024.0)),
            peak_mib = format_args!("{:.1}", w.peak_bytes as f64 / (1024.0 * 1024.0)),
            limit_mib = w.now.limit / (1024 * 1024),
            used_pct = format_args!("{:.1}", w.now.used_fraction() * 100.0),
            evicted_in_window = w.evicted,
            evicted_total = w.now.evicted,
            uploaded_in_window = w.inserted,
            "terrain gpu memory"
        );
    }

    /// Grow (and if necessary widen) the shared quad index buffer.
    ///
    /// `quads` is the largest single **layer** in the batch, not the batch total, because
    /// `draw_indexed` is given a 0-based range plus `base_vertex` — every section indexes from
    /// its own first vertex. A dense section measures ~814 quads, so `Uint16`'s 16 383-quad
    /// ceiling has ~20× headroom (T1.2).
    ///
    /// The ceiling is nonetheless enforced rather than assumed: a 16³ section holds 4096 blocks
    /// and there is no theorem that caps a pathological one below 16 384 quads in one layer, and
    /// a `u16` index buffer that silently wrapped would draw a section's geometry out of another
    /// section's vertices. Above the threshold this falls back to `Uint32` for the rest of the
    /// session — the upgrade is deliberately sticky, so a run of small batches after one large
    /// one cannot reintroduce the ceiling or thrash the buffer.
    ///
    /// Returns whether the buffer now covers `quads`: `false`, with the old
    /// buffer kept, when the GPU refused the new one; the caller then uploads nothing.
    fn grow_indices(&mut self, gpu: &Gpu, quads: u32) -> bool {
        let wide = quads > MAX_U16_QUADS || self.index_format == wgpu::IndexFormat::Uint32;
        let format = if wide {
            wgpu::IndexFormat::Uint32
        } else {
            wgpu::IndexFormat::Uint16
        };
        if quads <= self.index_capacity_quads && format == self.index_format {
            return true;
        }
        let capacity = if wide {
            quads.next_power_of_two()
        } else {
            // `next_power_of_two` would step past the ceiling at 8193..16 383 quads.
            quads.next_power_of_two().min(MAX_U16_QUADS)
        }
        .max(self.index_capacity_quads);
        let contents: Vec<u8> = if wide {
            bytemuck::cast_slice::<u32, u8>(&quad_indices(capacity)).to_vec()
        } else {
            bytemuck::cast_slice::<u16, u8>(&quad_indices_u16(capacity)).to_vec()
        };
        // `create_buffer_init` panics on a lost device; this refuses instead.
        let created = crate::gpu_health::try_create_buffer_init(
            &gpu.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain indices"),
                contents: &contents,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            },
        );
        match created {
            Ok(buffer) => {
                self.index_capacity_quads = capacity;
                self.index_format = format;
                self.index_buffer = buffer;
                true
            }
            Err(error) => {
                self.note_refused_upload(gpu, &error, contents.len());
                false
            }
        }
    }
}

/// Environment override for the terrain GPU vertex-memory cap, in **mebibytes**.
///
/// Documented here rather than in `mc_core::profile` because it is a renderer knob, but it
/// follows the same rules as `MC_PROFILE`/`MC_PROFILE_INTERVAL`: read once at renderer
/// construction, and **ignored** — falling back to [`GpuBudget::DEFAULT_BYTES`] — when unset,
/// empty, unparseable or zero. A typo must never silently shrink the cap to nothing.
///
/// | | |
/// |---|---|
/// | unset / empty / not a number / `0` | 512 MiB, i.e. [`GpuBudget::DEFAULT_BYTES`] |
/// | `MC_GPU_BUDGET_MB=2048` | 2 GiB |
///
/// It exists for the churn diagnosis, which needs the cap to be a variable rather than a
/// constant to tell two hypotheses apart. At rd 16 on real terrain the chunk pipeline never
/// reaches steady state, and the leading explanation is eviction churn against this cap
/// (F3 projects 424 MiB resident at rd 16 against 512 MiB). Raising the cap and re-running
/// separates "the budget is too small" (a capacity problem, which T1.1 fixes properly) from
/// "the re-queue path is wrong" (a correctness bug). Neither can be told from the other while the
/// only way to change the cap is to recompile.
pub const GPU_BUDGET_MB_ENV: &str = "MC_GPU_BUDGET_MB";

/// [`GPU_BUDGET_MB_ENV`], parsed, or [`GpuBudget::default`].
///
/// Read on every [`TerrainRenderer::new_with`] rather than cached in a `OnceLock`, so a test can
/// exercise both branches in one process.
#[must_use]
fn budget_from_env() -> GpuBudget {
    match std::env::var(GPU_BUDGET_MB_ENV) {
        Ok(v) => parse_budget_mb(&v).unwrap_or_default(),
        Err(_) => GpuBudget::default(),
    }
}

/// The parse half of [`budget_from_env`], split out so it can be tested without touching process
/// environment (which is global, and which other tests in this binary run in parallel with).
fn parse_budget_mb(raw: &str) -> Option<GpuBudget> {
    let mb: u64 = raw.trim().parse().ok()?;
    // Zero is rejected rather than honoured: a budget of nothing evicts every section every
    // frame, which looks exactly like the defect this knob exists to diagnose.
    (mb > 0).then(|| GpuBudget::new(mb.saturating_mul(1024 * 1024)))
}

/// How often [`TerrainRenderer`] logs a [`GpuMemoryWindow`] by itself: the profiler's own
/// `MC_PROFILE_INTERVAL`, and only when `MC_PROFILE` is on.
///
/// Deliberately piggy-backed on the profiler rather than given its own knob, so the residency line
/// appears in the same log, on the same cadence, as the scope table it has to be read against.
fn memory_log_interval_from_env() -> Option<std::time::Duration> {
    // Spelled out rather than taken from `mc_core::profile`'s constants: that module is behind
    // `mc-core`'s `subscriber` feature, which `mc-render` does not enable and should not start
    // enabling for two `&str`s. `mc_core::profile`'s own tests pin the names.
    let on = std::env::var("MC_PROFILE").is_ok_and(|v| {
        let v = v.trim();
        !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
    });
    if !on {
        return None;
    }
    let secs: f64 = std::env::var("MC_PROFILE_INTERVAL")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (secs > 0.0).then(|| std::time::Duration::from_secs_f64(secs))
}

/// The most quads one layer may hold and still be indexed with `u16`.
///
/// A quad's last vertex is `4q + 3`, so `q = 16 383` reaches exactly `u16::MAX`. 16 384 would
/// need index 65 539 and wrap. T1.2.
pub const MAX_U16_QUADS: u32 = (u16::MAX as u32 - 3) / 4;

/// `0,1,2, 0,2,3` per quad — the index pattern that turns `Buffer`'s quad runs into triangles.
#[must_use]
pub fn quad_indices(quads: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(quads as usize * 6);
    for q in 0..quads {
        let b = q * 4;
        out.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    out
}

/// [`quad_indices`] as `u16` — the same numbers in half the bytes. T1.2.
///
/// # Panics
///
/// If `quads` exceeds [`MAX_U16_QUADS`], which would wrap. Callers pick the format with
/// `grow_indices`; this asserts rather than truncating so a miscount is a crash and not silent
/// geometry corruption.
#[must_use]
pub fn quad_indices_u16(quads: u32) -> Vec<u16> {
    assert!(
        quads <= MAX_U16_QUADS,
        "{quads} quads need more than a u16 index (max {MAX_U16_QUADS})"
    );
    let mut out = Vec::with_capacity(quads as usize * 6);
    for q in 0..quads as u16 {
        let b = q * 4;
        out.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    out
}

// The layer lists `TerrainPhase::register` gives the solid and cutout
// phases, and the merged list the solid phase draws while the switch is on.
const ONE_PASS_SOLID: &[RenderLayer] = &[RenderLayer::Solid];
const ONE_PASS_CUTOUT: &[RenderLayer] = &[RenderLayer::CutoutMipped, RenderLayer::Cutout];
const ONE_PASS_SOLID_AND_CUTOUT: &[RenderLayer] = &[
    RenderLayer::Solid,
    RenderLayer::CutoutMipped,
    RenderLayer::Cutout,
];

/// Where [`TerrainRenderer::draw_layer`] records: a pass of its own on
/// `enc`, or the frame's shared world pass (see [`RenderPhase::shares_world_pass`]).
enum LayerPass<'a> {
    Open {
        enc: &'a mut wgpu::CommandEncoder,
        target: &'a wgpu::TextureView,
    },
    Shared(&'a mut wgpu::RenderPass<'static>),
}

/// Whether solid and cutout terrain share one render pass. Read once;
/// `MC_NW_F2_TERRAIN_ONE_PASS=0` restores a pass each.
fn terrain_one_pass_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("MC_NW_F2_TERRAIN_ONE_PASS").is_none_or(|value| value != "0")
    })
}

/// One terrain render pass. Frozen S11 API.
///
/// Four of these share a [`TerrainRenderer`]; use [`TerrainPhase::install`] rather than
/// constructing them by hand.
pub struct TerrainPhase {
    shared: Rc<RefCell<TerrainRenderer>>,
    layers: &'static [RenderLayer],
    label: &'static str,
    /// The solid phase owns the once-per-frame culling and upload work.
    leads_frame: bool,
}

/// Alias kept for readers who expect the per-layer phase to be named after what it draws.
pub type TerrainLayerPhase = TerrainPhase;

impl TerrainPhase {
    /// Build the renderer and register all four terrain phases on `phases`, returning the
    /// shared state so the caller can upload meshes into it.
    pub fn install(
        gpu: &Gpu,
        atlas: &Atlas,
        phases: &mut PhaseList,
    ) -> Rc<RefCell<TerrainRenderer>> {
        Self::install_with(gpu, atlas, TerrainOptions::default(), phases)
    }

    /// [`TerrainPhase::install`] with explicit [`TerrainOptions`].
    pub fn install_with(
        gpu: &Gpu,
        atlas: &Atlas,
        options: TerrainOptions,
        phases: &mut PhaseList,
    ) -> Rc<RefCell<TerrainRenderer>> {
        Self::install_with_options_and_anaglyph(gpu, atlas, options, false, phases)
    }

    /// Build and register all terrain phases with Java's startup anaglyph switch selected.
    #[must_use]
    pub fn install_with_anaglyph(
        gpu: &Gpu,
        atlas: &Atlas,
        anaglyph: bool,
        phases: &mut PhaseList,
    ) -> Rc<RefCell<TerrainRenderer>> {
        Self::install_with_options_and_anaglyph(
            gpu,
            atlas,
            TerrainOptions::default(),
            anaglyph,
            phases,
        )
    }

    /// [`Self::install_with`] with an explicit startup anaglyph switch.
    #[must_use]
    pub fn install_with_options_and_anaglyph(
        gpu: &Gpu,
        atlas: &Atlas,
        options: TerrainOptions,
        anaglyph: bool,
        phases: &mut PhaseList,
    ) -> Rc<RefCell<TerrainRenderer>> {
        let shared = Rc::new(RefCell::new(
            TerrainRenderer::new_with_options_and_anaglyph(gpu, atlas, options, anaglyph),
        ));
        Self::register(&shared, phases);
        shared
    }

    /// Register the four phases against an existing [`TerrainRenderer`].
    pub fn register(shared: &Rc<RefCell<TerrainRenderer>>, phases: &mut PhaseList) {
        phases.insert(
            PhaseOrder::TERRAIN_SOLID,
            Box::new(TerrainPhase {
                shared: Rc::clone(shared),
                layers: &[RenderLayer::Solid],
                label: "terrain/solid",
                leads_frame: true,
            }),
        );
        phases.insert(
            PhaseOrder::TERRAIN_CUTOUT,
            Box::new(TerrainPhase {
                shared: Rc::clone(shared),
                layers: &[RenderLayer::CutoutMipped, RenderLayer::Cutout],
                label: "terrain/cutout",
                leads_frame: false,
            }),
        );
        phases.insert(
            PhaseOrder::TERRAIN_TRANSLUCENT,
            Box::new(TerrainPhase {
                shared: Rc::clone(shared),
                layers: &[RenderLayer::Translucent],
                label: "terrain/translucent",
                leads_frame: false,
            }),
        );
    }
}

impl RenderPhase for TerrainPhase {
    fn name(&self) -> &'static str {
        self.label
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        let mut r = self.shared.borrow_mut();
        if self.leads_frame {
            r.begin_frame(gpu, cam);
        }
        // Solid and cutout record into one render pass: the same
        // draws, pipelines and bind groups in the same order, minus one pass boundary (wgpu
        // builds a command buffer plus a barrier buffer per pass). Nothing records between the
        // two orders, so nothing can land between them. `MC_NW_F2_TERRAIN_ONE_PASS=0` keeps two.
        if terrain_one_pass_enabled() {
            if self.layers == ONE_PASS_SOLID {
                let out = LayerPass::Open { enc, target: view };
                r.draw_layer(gpu, out, ONE_PASS_SOLID_AND_CUTOUT, self.label);
                return;
            }
            if self.layers == ONE_PASS_CUTOUT {
                return;
            }
        }
        r.draw_layer(
            gpu,
            LayerPass::Open { enc, target: view },
            self.layers,
            self.label,
        );
    }

    // The opaque layers draw into the frame's shared world pass: their own
    // pass is exactly that configuration. Translucent terrain records after the screen-space fog
    // (which reads depth) and may need its shader-pack scene pass, so it keeps its own.
    fn shares_world_pass(&self) -> bool {
        self.layers != [RenderLayer::Translucent]
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, cam: &Camera) {
        let mut r = self.shared.borrow_mut();
        if self.leads_frame {
            r.begin_frame(gpu, cam);
        }
        r.draw_layer(gpu, LayerPass::Shared(pass), self.layers, self.label);
    }

    // Translucent terrain only tests depth (every translucent pipeline,
    // vanilla and shader pack, has depth writes off: `glDepthMask(false)`), so it can draw in the
    // read-only-depth pass it shares with the screen-space fog recorded just before it. While a
    // shader pack's water reads the frame, `PhaseList` gives it its own pass as before.
    fn shares_depth_read_pass(&self) -> bool {
        self.layers == [RenderLayer::Translucent]
    }

    fn record_into_depth_read_pass(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass<'static>,
        cam: &Camera,
    ) {
        self.record_into_pass(gpu, pass, cam);
    }

    // Only the translucent layer, and only while a shader pack's water runs,
    // reads the frame so far (refraction depth and reflections). Otherwise `false`, and the
    // phase records exactly as before.
    fn needs_scene_color(&self) -> bool {
        self.layers.first() == Some(&RenderLayer::Translucent)
            && self.shared.borrow().pack_wants_scene()
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &crate::SceneTextures<'_>,
    ) {
        self.shared.borrow_mut().set_pack_scene(scene.color);
        self.record(gpu, enc, view, cam);
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn released_ranges_merge_back_into_a_whole_page_in_any_order() {
        const QUAD: u64 = 112;
        for order in [
            [0usize, 1, 2],
            [2, 1, 0],
            [1, 0, 2],
            [1, 2, 0],
            [0, 2, 1],
            [2, 0, 1],
        ] {
            let mut ranges = FreeRanges::new(QUAD * 100);
            let lens = [QUAD * 10, QUAD * 20, QUAD * 30];
            let offsets: Vec<u64> = lens
                .iter()
                .map(|&len| ranges.allocate(len).expect("fits"))
                .collect();
            assert_eq!(
                offsets,
                [0, QUAD * 10, QUAD * 30],
                "first fit is contiguous"
            );
            for index in order {
                ranges.release(offsets[index], lens[index]);
            }
            assert!(ranges.is_unused(), "order {order:?}");
        }
    }

    /// Phase 3b: growing a page extends its free list, merging with a free tail.
    #[test]
    fn a_grown_range_merges_its_new_tail_with_a_free_tail() {
        let mut ranges = FreeRanges::new(400);
        assert_eq!(ranges.allocate(100), Some(0));
        ranges.grow(800);
        assert_eq!(ranges.free, [(100, 700)], "free tail and new space merge");
        assert_eq!(ranges.allocate(700), Some(100));
        ranges.grow(1000);
        assert_eq!(ranges.free, [(800, 200)], "a full page gains one new range");
        ranges.release(0, 100);
        ranges.release(100, 700);
        assert!(ranges.is_unused());
    }

    /// Phase 3b: when every arena page is full, page 0 grows in place (one GPU copy) instead of
    /// a second page opening, so sections keep sharing one buffer whatever order they arrived
    /// in. Every slice is re-pointed at the grown buffer and every section's bytes survive the
    /// copies; past the growth limit a second page opens as before.
    #[test]
    fn a_full_arena_grows_page_zero_and_keeps_every_sections_bytes() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let atlas = Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let one_quad_mesh = |pos: ChunkSectionPos, marker: f32| {
            let mut mesh = ChunkMesh::empty(pos);
            let vertex = TerrainVertex::new([marker, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
            mesh.solid.push_quad([vertex; 4]);
            mesh
        };
        // The live path (`runChunkUploads`) is the one that sub-allocates from the arena.
        let live_upload = |renderer: &mut TerrainRenderer, mesh: ChunkMesh| {
            let deadline = Instant::now() + std::time::Duration::from_secs(3600);
            let retained =
                renderer.upload_all_with_deadline(&gpu, vec![mesh], DVec3::ZERO, deadline);
            assert!(retained.is_empty());
        };
        let mut renderer = TerrainRenderer::new(&gpu, &atlas);
        {
            let mut arena = renderer.arena.borrow_mut();
            arena.page_bytes = 4 * 1024;
            arena.grow_limit = 16 * 1024;
            arena.grow_enabled = true;
        }
        let positions: Vec<ChunkSectionPos> =
            (0..100).map(|x| ChunkSectionPos::new(x, 0, 0)).collect();
        for (marker, &pos) in positions.iter().enumerate() {
            live_upload(&mut renderer, one_quad_mesh(pos, marker as f32));
        }
        let buffer = {
            let arena = renderer.arena.borrow();
            assert_eq!(
                arena.pages.iter().flatten().count(),
                1,
                "page 0 grew instead"
            );
            Rc::clone(&arena.pages[0].as_ref().expect("page 0").buffer)
        };
        assert_eq!(
            buffer.size(),
            16 * 1024,
            "4 KiB doubled twice, to the limit"
        );

        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("arena growth readback"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, buffer.size());
        gpu.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).expect("map result receiver");
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("readback poll");
        rx.recv().expect("map callback").expect("readback map");
        let bytes = readback
            .get_mapped_range(..)
            .expect("readback view")
            .to_vec();
        readback.unmap();

        for (marker, &pos) in positions.iter().enumerate() {
            let slice = renderer.chunks[&pos].layers[0]
                .as_ref()
                .expect("solid layer");
            assert!(Rc::ptr_eq(&slice.buffer, &buffer), "section {marker}");
            let expected = one_quad_mesh(pos, marker as f32).solid;
            let start = slice.offset as usize;
            assert_eq!(
                &bytes[start..start + slice.len as usize],
                expected.as_bytes(),
                "section {marker}"
            );
        }

        for x in 100..250 {
            let pos = ChunkSectionPos::new(x, 0, 0);
            live_upload(&mut renderer, one_quad_mesh(pos, x as f32));
        }
        {
            let arena = renderer.arena.borrow();
            assert!(arena.pages.iter().flatten().count() >= 2, "past the limit");
            assert_eq!(
                arena.pages[0].as_ref().expect("page 0").ranges.capacity,
                16 * 1024
            );
        }

        // Unloading everything gives the grown page back (review S3): a later world starts
        // from a page of the normal size, not the 16 KiB one.
        for x in 0..250 {
            renderer.remove(ChunkSectionPos::new(x, 0, 0));
        }
        let arena = renderer.arena.borrow();
        assert!(
            arena
                .pages
                .iter()
                .flatten()
                .all(|page| page.ranges.capacity <= 4 * 1024),
            "no grown page survives emptying"
        );
    }

    #[test]
    fn a_freed_hole_is_reused_and_a_full_page_refuses() {
        const QUAD: u64 = 112;
        let mut ranges = FreeRanges::new(QUAD * 10);
        let first = ranges.allocate(QUAD * 4).unwrap();
        let second = ranges.allocate(QUAD * 6).unwrap();
        assert_eq!(ranges.allocate(QUAD), None);
        ranges.release(first, QUAD * 4);
        assert_eq!(ranges.allocate(QUAD * 5), None, "the hole is too small");
        assert_eq!(ranges.allocate(QUAD * 3), Some(0));
        assert_eq!(ranges.allocate(QUAD), Some(QUAD * 3));
        ranges.release(second, QUAD * 6);
        // The list as the next allocation will read it.
        assert_eq!(ranges.settled(), vec![(QUAD * 4, QUAD * 6)]);
    }

    /// Lazy releases build exactly the free list one sorted insert per
    /// release builds: the same churn drives a lazy and an eager list in lockstep, with bursts of
    /// many releases in a row (as a world change or a far teleport makes), and every allocation
    /// must get the same offset (or the same refusal), every growth the same tail, and the two
    /// pages must agree on being unused after every step.
    #[test]
    fn lazy_releases_allocate_exactly_like_one_insert_per_release() {
        const STRIDE: u64 = 28;
        const QUAD: u64 = 4 * STRIDE;
        let mut eager = FreeRanges::with_mode(QUAD * 3000, false);
        let mut lazy = FreeRanges::with_mode(QUAD * 3000, true);
        let mut live: Vec<(u64, u64)> = Vec::new();
        let mut random = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let (mut bursts, mut emptied, mut grown) = (0, 0, 0);
        for step in 0..6_000 {
            let roll = next() % 100;
            if roll < 2 && !live.is_empty() {
                // A burst: most or all of the live ranges come back at once, in random order.
                bursts += 1;
                let keep = if next() % 3 == 0 { 0 } else { live.len() / 4 };
                while live.len() > keep {
                    let index = (next() % live.len() as u64) as usize;
                    let (offset, len) = live.swap_remove(index);
                    eager.release(offset, len);
                    lazy.release(offset, len);
                    assert_eq!(lazy.is_unused(), eager.is_unused(), "step {step}");
                }
                emptied += usize::from(live.is_empty());
            } else if roll < 3 {
                let capacity = eager.capacity + QUAD * (1 + next() % 500);
                eager.grow(capacity);
                lazy.grow(capacity);
                grown += 1;
            } else if live.is_empty() || roll < 58 {
                let len = QUAD * (1 + next() % 150);
                let offset = eager.allocate(len);
                assert_eq!(lazy.allocate(len), offset, "step {step}");
                if let Some(offset) = offset {
                    live.push((offset, len));
                }
            } else {
                let index = (next() % live.len() as u64) as usize;
                let (offset, len) = live.swap_remove(index);
                eager.release(offset, len);
                lazy.release(offset, len);
            }
            assert_eq!(lazy.is_unused(), eager.is_unused(), "step {step}");
            assert_eq!(lazy.settled(), eager.free, "step {step}");
            assert_eq!(lazy.used, eager.used, "step {step}");
        }
        assert!(
            bursts > 50 && emptied > 5 && grown > 20,
            "{bursts} {emptied} {grown}"
        );
    }

    /// A long churn of mixed sizes, as chunk loading and block edits produce. A live range that
    /// overlapped another would draw one section with another's vertices, so after every step:
    /// live ranges never overlap, every byte is live or free exactly once, free ranges stay
    /// sorted and merged, and every offset is a whole number of vertices for `base_vertex`.
    #[test]
    fn churn_never_overlaps_and_accounts_for_every_byte() {
        const STRIDE: u64 = 28;
        const QUAD: u64 = 4 * STRIDE;
        const CAPACITY: u64 = QUAD * 4000;
        let mut ranges = FreeRanges::new(CAPACITY);
        let mut live: Vec<(u64, u64)> = Vec::new();
        let mut random = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let (mut allocations, mut refusals) = (0, 0);
        for _ in 0..20_000 {
            if live.is_empty() || next() % 100 < 55 {
                let len = QUAD * (1 + next() % 200);
                match ranges.allocate(len) {
                    Some(offset) => {
                        assert_eq!(offset % STRIDE, 0);
                        assert!(offset + len <= CAPACITY);
                        live.push((offset, len));
                        allocations += 1;
                    }
                    None => refusals += 1,
                }
            } else {
                let index = (next() % live.len() as u64) as usize;
                let (offset, len) = live.swap_remove(index);
                ranges.release(offset, len);
            }

            let mut sorted = live.clone();
            sorted.sort_unstable();
            for pair in sorted.windows(2) {
                assert!(pair[0].0 + pair[0].1 <= pair[1].0, "live ranges overlap");
            }
            // The list as the next allocation will read it.
            let free_list = ranges.settled();
            for pair in free_list.windows(2) {
                assert!(
                    pair[0].0 + pair[0].1 < pair[1].0,
                    "free ranges must stay sorted and merged"
                );
            }
            let used: u64 = live.iter().map(|&(_, len)| len).sum();
            let free: u64 = free_list.iter().map(|&(_, len)| len).sum();
            assert_eq!(used + free, CAPACITY);
            for &(offset, len) in &sorted {
                assert!(
                    free_list
                        .iter()
                        .all(|&(start, size)| offset + len <= start || start + size <= offset),
                    "a live range is also free"
                );
            }
        }
        assert!(
            allocations > 1000 && refusals > 0,
            "the churn must fill the page"
        );
        for (offset, len) in live {
            ranges.release(offset, len);
        }
        assert!(ranges.is_unused());
    }
}

/// The per-section maps are sized for the render distance before a load fills
/// them, so no install in a world load rehashes them.
#[cfg(test)]
mod nw_f5_reserve_tests {
    use super::*;

    #[test]
    fn the_reserve_target_is_the_render_distance_square_capped_at_sixteen() {
        assert_eq!(section_reserve_target(8), 17 * 17 * 16, "SkyWars: 4624");
        assert_eq!(section_reserve_target(16), 33 * 33 * 16);
        assert_eq!(section_reserve_target(32), 33 * 33 * 16, "capped at 16");
        assert_eq!(
            section_reserve_target(0),
            5 * 5 * 16,
            "clamped to 2 like the setting"
        );
    }

    #[test]
    fn a_world_load_never_reallocates_the_section_maps() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let atlas = Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let mut renderer = TerrainRenderer::new(&gpu, &atlas);
        renderer.render_distance = 2;
        let target = section_reserve_target(2);
        // Every section of a render-distance-2 square: 5 x 5 columns of 16.
        let positions: Vec<ChunkSectionPos> = (-2..=2)
            .flat_map(|x| (-2..=2).flat_map(move |z| (0..16).map(move |y| (x, y, z))))
            .map(|(x, y, z)| ChunkSectionPos::new(x, y, z))
            .collect();
        assert_eq!(positions.len(), target);

        // Negative control: the same inserts into a map that is not reserved reallocate it
        // several times (256 -> 512 buckets at 225 entries, and so on).
        let mut plain: HashMap<ChunkSectionPos, ()> = HashMap::new();
        let mut reallocations = 0;
        for &pos in &positions {
            let before = plain.capacity();
            plain.insert(pos, ());
            reallocations += usize::from(plain.capacity() != before);
        }
        assert!(
            reallocations >= 3,
            "the control must cross growth steps: {reallocations}"
        );

        let deadline = Instant::now() + std::time::Duration::from_secs(3600);
        let upload = |renderer: &mut TerrainRenderer, pos: ChunkSectionPos, marker: f32| {
            let mut mesh = ChunkMesh::empty(pos);
            let vertex = TerrainVertex::new([marker, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
            mesh.solid.push_quad([vertex; 4]);
            let retained =
                renderer.upload_all_with_deadline(&gpu, vec![mesh], DVec3::ZERO, deadline);
            assert!(retained.is_empty());
        };
        upload(&mut renderer, positions[0], 0.0);
        let capacity = renderer.chunks.capacity();
        let revisions = renderer.mesh_revisions.capacity();
        assert!(
            capacity >= target,
            "sized for the square at the first install: {capacity}"
        );
        for (marker, &pos) in positions.iter().enumerate().skip(1) {
            upload(&mut renderer, pos, marker as f32);
            assert_eq!(
                renderer.chunks.capacity(),
                capacity,
                "rehashed at install {marker}"
            );
        }
        assert_eq!(renderer.loaded_sections(), target);
        assert_eq!(renderer.mesh_revisions.capacity(), revisions);
        assert!(renderer.mesh_revisions.len() <= revisions);
    }
}

/// Removing every resident section at once leaves the renderer exactly as
/// removing them one by one does.
#[cfg(test)]
mod nw_f5_bulk_drop_tests {
    use super::*;

    fn atlas() -> Atlas {
        Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        }
    }

    fn mesh(pos: ChunkSectionPos, quads: usize) -> ChunkMesh {
        let mut mesh = ChunkMesh::empty(pos);
        let vertex = TerrainVertex::new([pos.x as f32, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
        for _ in 0..quads {
            mesh.solid.push_quad([vertex; 4]);
        }
        if quads.is_multiple_of(2) {
            mesh.cutout.push_quad([vertex; 4]);
        }
        mesh
    }

    /// A renderer with small pages (so a load opens several, and page 0 grows) holding
    /// `count` sections, two of them all-air.
    fn loaded(gpu: &Gpu, count: i32) -> (TerrainRenderer, Vec<ChunkSectionPos>) {
        let mut renderer = TerrainRenderer::new(gpu, &atlas());
        {
            let mut arena = renderer.arena.borrow_mut();
            arena.page_bytes = 4 * 1024;
            arena.grow_limit = 8 * 1024;
            arena.grow_enabled = true;
        }
        let deadline = Instant::now() + std::time::Duration::from_secs(3600);
        let positions: Vec<ChunkSectionPos> = (0..count)
            .map(|x| ChunkSectionPos::new(x, x % 5, -x))
            .collect();
        let meshes = positions
            .iter()
            .map(|&pos| {
                if pos.x < 2 {
                    ChunkMesh::empty(pos)
                } else {
                    mesh(pos, 1 + pos.x as usize % 4)
                }
            })
            .collect();
        let retained = renderer.upload_all_with_deadline(gpu, meshes, DVec3::ZERO, deadline);
        assert!(retained.is_empty());
        (renderer, positions)
    }

    /// What the next uploads see: each live page's size and free list, in page order.
    fn arena_state(renderer: &TerrainRenderer) -> Vec<(u64, Vec<(u64, u64)>)> {
        renderer
            .arena
            .borrow()
            .pages
            .iter()
            .flatten()
            .map(|page| (page.ranges.capacity, page.ranges.settled()))
            .collect()
    }

    fn slices(renderer: &TerrainRenderer) -> Vec<(ChunkSectionPos, usize, u64, u64)> {
        let mut out: Vec<_> = renderer
            .chunks
            .iter()
            .flat_map(|(pos, chunk)| {
                chunk
                    .layers
                    .iter()
                    .enumerate()
                    .filter_map(move |(i, s)| s.as_ref().map(|s| (*pos, i, s.offset, s.len)))
            })
            .collect();
        out.sort_unstable_by_key(|&(pos, i, ..)| (pos.x, pos.y, pos.z, i));
        out
    }

    #[test]
    fn bulk_removal_of_every_section_equals_removing_them_one_by_one() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let (mut one_by_one, positions) = loaded(&gpu, 120);
        let (mut bulk, _) = loaded(&gpu, 120);
        assert!(
            one_by_one.arena.borrow().pages.iter().flatten().count() >= 2,
            "the load must span pages for this to test anything"
        );
        let set: HashSet<ChunkSectionPos> = positions.iter().copied().collect();
        for &pos in &set {
            one_by_one.remove(pos);
        }
        bulk.remove_sections(&set);

        for renderer in [&one_by_one, &bulk] {
            assert_eq!(renderer.loaded_sections(), 0);
            assert_eq!(renderer.gpu_memory().sections, 0);
            assert_eq!(renderer.gpu_memory().bytes, 0);
            assert!(renderer.draw_plans.iter().all(Option::is_none));
            assert!(renderer.translucent_resort_sections.is_empty());
        }
        assert_eq!(arena_state(&bulk), arena_state(&one_by_one));
        assert_eq!(arena_state(&bulk), vec![(4 * 1024, vec![(0, 4 * 1024)])]);
        let keys = |m: &HashMap<ChunkSectionPos, MeshRevision>| {
            let mut k: Vec<_> = m.keys().map(|p| (p.x, p.y, p.z)).collect();
            k.sort_unstable();
            k
        };
        assert_eq!(keys(&bulk.mesh_revisions), keys(&one_by_one.mesh_revisions));
        assert_eq!(
            bulk.invalid_mesh_revisions,
            one_by_one.invalid_mesh_revisions
        );
        // A tombstoned section is accepted again only with a newer revision, in both.
        for renderer in [&bulk, &one_by_one] {
            let current = renderer.mesh_revisions[&positions[5]];
            assert!(!renderer.accepts_mesh_revision(positions[5], current));
        }

        // The next world loads into identical arena ranges. (Its meshes carry newer revisions in
        // the client; these unversioned test meshes need the tombstones, already compared above,
        // out of the way in both renderers alike.)
        let deadline = Instant::now() + std::time::Duration::from_secs(3600);
        for renderer in [&mut one_by_one, &mut bulk] {
            renderer.mesh_revisions.clear();
            renderer.invalid_mesh_revisions.clear();
            let meshes = positions.iter().map(|&pos| mesh(pos, 3)).collect();
            let retained = renderer.upload_all_with_deadline(&gpu, meshes, DVec3::ZERO, deadline);
            assert!(retained.is_empty());
        }
        assert_eq!(slices(&bulk), slices(&one_by_one));
        assert_eq!(arena_state(&bulk), arena_state(&one_by_one));
        assert_eq!(bulk.loaded_sections(), positions.len());
    }

    #[test]
    fn a_partial_set_takes_the_one_by_one_path() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let (mut one_by_one, positions) = loaded(&gpu, 60);
        let (mut bulk, _) = loaded(&gpu, 60);
        let set: HashSet<ChunkSectionPos> = positions.iter().step_by(3).copied().collect();
        for &pos in &set {
            one_by_one.remove(pos);
        }
        bulk.remove_sections(&set);
        assert_eq!(bulk.loaded_sections(), 60 - set.len());
        assert_eq!(slices(&bulk), slices(&one_by_one));
        assert_eq!(arena_state(&bulk), arena_state(&one_by_one));
    }
}

/// Forgetting the
/// mesh-revision tombstones when a world goes away; lazy arena releases.
#[cfg(test)]
mod w18_oc_tests {
    use super::*;

    fn atlas() -> Atlas {
        Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        }
    }

    fn mesh(pos: ChunkSectionPos, quads: usize) -> ChunkMesh {
        let mut mesh = ChunkMesh::empty(pos);
        let vertex = TerrainVertex::new([pos.x as f32, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
        for _ in 0..quads {
            mesh.solid.push_quad([vertex; 4]);
        }
        mesh
    }

    /// Worker results for `positions`, with revisions drawn now (as a queue reset after a world
    /// change draws them for the next world).
    fn results(positions: &[ChunkSectionPos]) -> Vec<(ChunkSectionPos, usize, MeshRevision)> {
        positions
            .iter()
            .map(|&pos| {
                (
                    pos,
                    1 + pos.x.unsigned_abs() as usize % 4,
                    next_mesh_revision(),
                )
            })
            .collect()
    }

    fn upload(gpu: &Gpu, renderer: &mut TerrainRenderer, batch: &[(ChunkSectionPos, usize, u64)]) {
        let meshes = batch
            .iter()
            .map(|&(pos, quads, revision)| {
                let mesh = if quads == 1 {
                    ChunkMesh::empty(pos)
                } else {
                    mesh(pos, quads)
                };
                MeshTaskResult::with_revision(mesh, revision)
            })
            .collect();
        let deadline = Instant::now() + std::time::Duration::from_secs(3600);
        let upload = renderer.upload_all_with_deadline_retaining_revisioned(
            gpu,
            meshes,
            DVec3::ZERO,
            deadline,
        );
        assert_eq!(upload.input_revision_rejections(), 0);
        assert_eq!(upload.retained_revision_rejections(), 0);
        assert_eq!(upload.retained_ready_count(), 0);
    }

    fn renderer(gpu: &Gpu) -> TerrainRenderer {
        let renderer = TerrainRenderer::new(gpu, &atlas());
        {
            let mut arena = renderer.arena.borrow_mut();
            arena.page_bytes = 4 * 1024;
            arena.grow_limit = 8 * 1024;
            arena.grow_enabled = true;
        }
        renderer
    }

    fn arena_state(renderer: &TerrainRenderer) -> Vec<(u64, Vec<(u64, u64)>)> {
        renderer
            .arena
            .borrow()
            .pages
            .iter()
            .flatten()
            .map(|page| (page.ranges.capacity, page.ranges.settled()))
            .collect()
    }

    fn slices(renderer: &TerrainRenderer) -> Vec<(ChunkSectionPos, usize, u64, u64)> {
        let mut out: Vec<_> = renderer
            .chunks
            .iter()
            .flat_map(|(pos, chunk)| {
                chunk
                    .layers
                    .iter()
                    .enumerate()
                    .filter_map(move |(i, s)| s.as_ref().map(|s| (*pos, i, s.offset, s.len)))
            })
            .collect();
        out.sort_unstable_by_key(|&(pos, i, ..)| (pos.x, pos.y, pos.z, i));
        out
    }

    #[test]
    fn forgetting_a_world_changes_nothing_a_later_result_sees() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let world_a: Vec<ChunkSectionPos> = (0..120)
            .map(|x| ChunkSectionPos::new(x, x % 5, -x))
            .collect();
        let load_a = results(&world_a);
        let mut kept = renderer(&gpu);
        let mut pruned = renderer(&gpu);
        for renderer in [&mut kept, &mut pruned] {
            upload(&gpu, renderer, &load_a);
            // A section unloaded earlier in world A leaves a tombstone of its own.
            renderer.remove(world_a[7]);
        }
        assert!(kept.arena.borrow().pages.iter().flatten().count() >= 2);
        let resident: HashSet<ChunkSectionPos> = kept.chunks.keys().copied().collect();
        let capacity = pruned.mesh_revisions.capacity();
        kept.remove_sections(&resident);
        pruned.remove_world_sections(&resident);

        // The old behavior keeps an entry for every position world A ever had; now none stay,
        // and the maps keep their capacity for the next world.
        assert_eq!(kept.mesh_revisions.len(), world_a.len());
        assert_eq!(kept.invalid_mesh_revisions.len(), world_a.len());
        assert!(pruned.mesh_revisions.is_empty());
        assert!(pruned.invalid_mesh_revisions.is_empty());
        assert!(pruned.mesh_revisions.capacity() >= capacity);
        // Everything else the removal leaves is the same.
        for renderer in [&kept, &pruned] {
            assert_eq!(renderer.loaded_sections(), 0);
            assert_eq!(renderer.gpu_memory().bytes, 0);
            assert!(renderer.draw_plans.iter().all(Option::is_none));
        }
        assert_eq!(arena_state(&pruned), arena_state(&kept));

        // World B overlaps world A's positions (60..120, and world A's removed 7) and adds new
        // ones. Its results are meshed after the reset, so their revisions are newer than every
        // entry forgotten above: both renderers accept them all and place them identically.
        let world_b: Vec<ChunkSectionPos> = (60..200)
            .chain([7])
            .map(|x| ChunkSectionPos::new(x, x % 5, -x))
            .collect();
        let load_b = results(&world_b);
        for renderer in [&mut kept, &mut pruned] {
            upload(&gpu, renderer, &load_b);
        }
        assert_eq!(slices(&pruned), slices(&kept));
        assert_eq!(arena_state(&pruned), arena_state(&kept));
        assert_eq!(pruned.loaded_sections(), kept.loaded_sections());
        for &(pos, _, revision) in &load_b {
            assert_eq!(pruned.mesh_revisions.get(&pos), Some(&revision));
            assert_eq!(kept.mesh_revisions.get(&pos), Some(&revision));
            for candidate in [revision - 1, revision, next_mesh_revision()] {
                assert_eq!(
                    pruned.accepts_mesh_revision(pos, candidate),
                    kept.accepts_mesh_revision(pos, candidate)
                );
            }
        }
        // A result from world A could only differ here (a tombstone rejects it); none can
        // arrive, because `MeshQueue::reset_world` drops the old result channel (see the queue
        // test `reset_world_replaces_work_preserves_options_and_keeps_diagnostic_frame`).
        assert!(!kept.accepts_mesh_revision(world_a[3], load_a[3].2));
        assert!(pruned.accepts_mesh_revision(world_a[3], load_a[3].2));
    }

    /// A far teleport removes every section one by one (column unload
    /// events, not the world-change bulk path), and a near one removes some of them; then the
    /// new area loads. With lazy releases the arena must end up exactly as with one insert per
    /// release: same pages, same free lists, and the next uploads at the same offsets.
    #[test]
    fn lazy_releases_leave_every_later_upload_where_eager_ones_did() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let area = |from: i32, to: i32| -> Vec<ChunkSectionPos> {
            (from..to)
                .flat_map(|x| (0..4).map(move |y| ChunkSectionPos::new(x, y, -x)))
                .collect()
        };
        let mut eager = renderer(&gpu);
        let mut lazy = renderer(&gpu);
        eager.arena.borrow_mut().lazy_release = false;
        lazy.arena.borrow_mut().lazy_release = true;
        let first = results(&area(0, 60));
        for renderer in [&mut eager, &mut lazy] {
            upload(&gpu, renderer, &first);
        }
        assert!(lazy.arena.borrow().pages.iter().flatten().count() >= 2);
        assert!(lazy
            .arena
            .borrow()
            .pages
            .iter()
            .flatten()
            .all(|page| page.ranges.lazy));
        assert!(eager
            .arena
            .borrow()
            .pages
            .iter()
            .flatten()
            .all(|page| !page.ranges.lazy));

        // A near teleport: the columns behind unload one section at a time, in hash order.
        let behind: HashSet<ChunkSectionPos> = area(0, 25).into_iter().collect();
        for renderer in [&mut eager, &mut lazy] {
            for &pos in &behind {
                renderer.remove(pos);
            }
        }
        assert_eq!(arena_state(&lazy), arena_state(&eager));
        let ahead = results(&area(60, 90));
        for renderer in [&mut eager, &mut lazy] {
            upload(&gpu, renderer, &ahead);
        }
        assert_eq!(slices(&lazy), slices(&eager));
        assert_eq!(arena_state(&lazy), arena_state(&eager));

        // A far teleport: everything unloads one section at a time, then a new area loads.
        let everything: HashSet<ChunkSectionPos> = eager.chunks.keys().copied().collect();
        for renderer in [&mut eager, &mut lazy] {
            for &pos in &everything {
                renderer.remove(pos);
            }
            assert_eq!(renderer.loaded_sections(), 0);
        }
        assert_eq!(arena_state(&lazy), arena_state(&eager));
        let far = results(&area(500, 580));
        for renderer in [&mut eager, &mut lazy] {
            upload(&gpu, renderer, &far);
        }
        assert_eq!(slices(&lazy), slices(&eager));
        assert_eq!(arena_state(&lazy), arena_state(&eager));
        assert_eq!(lazy.loaded_sections(), eager.loaded_sections());
    }

    /// Measurement, not a check (run with `--ignored --nocapture`): a world
    /// change at render distance 8 and 16 through the bulk path, with tombstones written
    /// (`remove_sections`) and forgotten (`remove_world_sections`), alternating.
    #[test]
    #[ignore = "timing probe; prints numbers"]
    fn probe_world_change_prune() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let vertex = TerrainVertex::new([1.0, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
        let deadline = Instant::now() + std::time::Duration::from_secs(3600);
        for r in [8i32, 16] {
            for prune in [false, true, false, true, false, true] {
                let mut renderer = TerrainRenderer::new(&gpu, &atlas());
                renderer.render_distance = r as u32;
                let positions: Vec<ChunkSectionPos> = (-r..=r)
                    .flat_map(|x| (-r..=r).flat_map(move |z| (0..16).map(move |y| (x, y, z))))
                    .map(|(x, y, z)| ChunkSectionPos::new(x, y, z))
                    .collect();
                for chunk in positions.chunks(512) {
                    let meshes = chunk
                        .iter()
                        .map(|&pos| {
                            let mut mesh = ChunkMesh::empty(pos);
                            mesh.solid.push_quad([vertex; 4]);
                            mesh
                        })
                        .collect();
                    let retained =
                        renderer.upload_all_with_deadline(&gpu, meshes, DVec3::ZERO, deadline);
                    assert!(retained.is_empty());
                }
                let set: HashSet<ChunkSectionPos> = positions.iter().copied().collect();
                let start = Instant::now();
                if prune {
                    renderer.remove_world_sections(&set);
                } else {
                    renderer.remove_sections(&set);
                }
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(renderer.loaded_sections(), 0);
                println!(
                    "rd {r} world change, {} sections, tombstones {}: {ms:.3} ms",
                    positions.len(),
                    if prune { "forgotten" } else { "written" }
                );
            }
        }
    }

    /// Measurement, not a check (run with `--ignored --nocapture`): a
    /// teleport at render distance 8 and 16 removes all (far) or half (near) of the sections one
    /// at a time, in hash order, as column unloads do; time that removal with eager and lazy
    /// releases, and the next upload (which merges the lazy releases). Real page sizes.
    #[test]
    #[ignore = "timing probe; prints numbers"]
    fn probe_teleport_removal() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let vertex = TerrainVertex::new([1.0, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
        let mesh = |pos: ChunkSectionPos| {
            let mut mesh = ChunkMesh::empty(pos);
            for _ in 0..6 {
                mesh.solid.push_quad([vertex; 4]);
            }
            mesh.cutout.push_quad([vertex; 4]);
            mesh
        };
        let deadline = Instant::now() + std::time::Duration::from_secs(3600);
        for r in [8i32, 16] {
            for far in [true, false] {
                for lazy in [false, true, false, true] {
                    let mut renderer = TerrainRenderer::new(&gpu, &atlas());
                    renderer.render_distance = r as u32;
                    renderer.arena.borrow_mut().lazy_release = lazy;
                    let positions: Vec<ChunkSectionPos> = (-r..=r)
                        .flat_map(|x| (-r..=r).flat_map(move |z| (0..16).map(move |y| (x, y, z))))
                        .map(|(x, y, z)| ChunkSectionPos::new(x, y, z))
                        .collect();
                    for chunk in positions.chunks(512) {
                        let meshes = chunk.iter().map(|&pos| mesh(pos)).collect();
                        let retained =
                            renderer.upload_all_with_deadline(&gpu, meshes, DVec3::ZERO, deadline);
                        assert!(retained.is_empty());
                    }
                    let order: HashSet<ChunkSectionPos> = positions
                        .iter()
                        .copied()
                        .filter(|pos| far || pos.x < 0)
                        .collect();
                    let start = Instant::now();
                    for &pos in &order {
                        renderer.remove(pos);
                    }
                    let removal = start.elapsed().as_secs_f64() * 1000.0;
                    let next: Vec<ChunkMesh> = (0..256)
                        .map(|i| mesh(ChunkSectionPos::new(1000 + i / 16, i % 16, 1000)))
                        .collect();
                    let start = Instant::now();
                    let retained =
                        renderer.upload_all_with_deadline(&gpu, next, DVec3::ZERO, deadline);
                    let upload = start.elapsed().as_secs_f64() * 1000.0;
                    assert!(retained.is_empty());
                    println!(
                        "rd {r} {} teleport, {} releases: removing {} sections {removal:.3} ms, \
                         next 256-section upload {upload:.3} ms",
                        if far { "far" } else { "near" },
                        if lazy { "lazy" } else { "eager" },
                        order.len()
                    );
                }
            }
        }
    }
}

/// Timing probes for the terrain's worst frames (all ignored;
/// run with `--ignored --nocapture`). They print numbers and check nothing.
#[cfg(test)]
mod nw_f5_probes {
    use super::*;

    /// Measurement, not a check (run with `--ignored --nocapture`): GPU time of the first pass
    /// that binds a fresh 256 MiB vertex buffer whole (wgpu zero-fills it first) against one
    /// that binds only a written 1 MiB prefix.
    #[test]
    #[ignore = "timing probe; prints numbers"]
    fn probe_whole_page_bind_clear_cost() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("probe target"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let page = 256 * 1024 * 1024u64;
        let written = 1024 * 1024u64;
        let run = |prefix: bool| -> f64 {
            let buffer = new_arena_buffer(&gpu, page);
            gpu.queue
                .write_buffer(&buffer, 0, &vec![7u8; written as usize]);
            gpu.queue.submit([]);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("idle");
            let start = Instant::now();
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("probe"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let len = if prefix { written } else { page };
                pass.set_vertex_buffer(0, buffer.slice(..len));
            }
            gpu.queue.submit([encoder.finish()]);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("idle");
            start.elapsed().as_secs_f64() * 1000.0
        };
        for round in 0..4 {
            let whole = run(false);
            let prefix = run(true);
            println!("round {round}: whole-page bind {whole:.3} ms, prefix bind {prefix:.3} ms");
        }
    }

    /// Measurement, not a check (run with `--ignored --nocapture`): the CPU cost of growing a
    /// full 256 MiB page to 512 MiB, split into buffer creation and the copy's submission.
    #[test]
    #[ignore = "timing probe; prints numbers"]
    fn probe_page_growth_cpu_cost() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        for round in 0..3 {
            let old = new_arena_buffer(&gpu, 256 * 1024 * 1024);
            gpu.queue.write_buffer(&old, 0, &vec![1u8; 4 * 1024 * 1024]);
            gpu.queue.submit([]);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("idle");
            let mut page = ArenaPage {
                buffer: Rc::new(old),
                ranges: FreeRanges::new(256 * 1024 * 1024),
            };
            let start = Instant::now();
            let created = new_arena_buffer(&gpu, 512 * 1024 * 1024);
            let create_ms = start.elapsed().as_secs_f64() * 1000.0;
            drop(created);
            let start = Instant::now();
            let replaced = grow_page(&gpu, &mut page, 512 * 1024 * 1024);
            let grow_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert!(replaced.is_some());
            let start = Instant::now();
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("idle");
            let gpu_ms = start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "round {round}: create 512 MiB {create_ms:.3} ms; grow_page (create + copy \
                 submit) {grow_ms:.3} ms; then GPU wait {gpu_ms:.3} ms"
            );
        }
    }

    /// Measurement, not a check (run with `--ignored --nocapture`): a world change removes every
    /// resident section in one frame (`drop_all_chunk_meshes`, in the client's hash-set order).
    /// Times that for the section counts of render distance 8 and 16, two layers each.
    #[test]
    #[ignore = "timing probe; prints numbers"]
    fn probe_world_change_removal_cost() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let atlas = Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        for r in [8i32, 16] {
            let mut renderer = TerrainRenderer::new(&gpu, &atlas);
            renderer.render_distance = r as u32;
            let positions: Vec<ChunkSectionPos> = (-r..=r)
                .flat_map(|x| (-r..=r).flat_map(move |z| (0..16).map(move |y| (x, y, z))))
                .map(|(x, y, z)| ChunkSectionPos::new(x, y, z))
                .collect();
            let deadline = Instant::now() + std::time::Duration::from_secs(3600);
            let vertex = TerrainVertex::new([1.0, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
            for chunk in positions.chunks(512) {
                let meshes = chunk
                    .iter()
                    .map(|&pos| {
                        let mut mesh = ChunkMesh::empty(pos);
                        for _ in 0..6 {
                            mesh.solid.push_quad([vertex; 4]);
                        }
                        mesh.cutout.push_quad([vertex; 4]);
                        mesh
                    })
                    .collect();
                let retained =
                    renderer.upload_all_with_deadline(&gpu, meshes, DVec3::ZERO, deadline);
                assert!(retained.is_empty());
            }
            // The client's `uploaded_sections` is a hash set: removal order is its iteration
            // order, effectively random.
            let order: HashSet<ChunkSectionPos> = positions.iter().copied().collect();
            let start = Instant::now();
            for pos in order {
                renderer.remove(pos);
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(renderer.loaded_sections(), 0);
            println!(
                "render distance {r}: removing {} sections took {ms:.3} ms",
                positions.len()
            );
        }
    }

    /// Measurement, not a check: where a world change's removal time goes. Drops the same
    /// resident set three ways: through `remove` (hash order), by dropping the records in hash
    /// order (arena releases only), and by dropping them in arena offset order.
    #[test]
    #[ignore = "timing probe; prints numbers"]
    fn probe_world_change_removal_breakdown() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let atlas = Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let vertex = TerrainVertex::new([1.0, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
        let filled = |r: i32| {
            let mut renderer = TerrainRenderer::new(&gpu, &atlas);
            renderer.render_distance = r as u32;
            let positions: Vec<ChunkSectionPos> = (-r..=r)
                .flat_map(|x| (-r..=r).flat_map(move |z| (0..16).map(move |y| (x, y, z))))
                .map(|(x, y, z)| ChunkSectionPos::new(x, y, z))
                .collect();
            let deadline = Instant::now() + std::time::Duration::from_secs(3600);
            for chunk in positions.chunks(512) {
                let meshes = chunk
                    .iter()
                    .map(|&pos| {
                        let mut mesh = ChunkMesh::empty(pos);
                        for _ in 0..6 {
                            mesh.solid.push_quad([vertex; 4]);
                        }
                        mesh.cutout.push_quad([vertex; 4]);
                        mesh
                    })
                    .collect();
                let retained =
                    renderer.upload_all_with_deadline(&gpu, meshes, DVec3::ZERO, deadline);
                assert!(retained.is_empty());
            }
            (renderer, positions)
        };
        for r in [8i32, 16] {
            let (mut renderer, positions) = filled(r);
            let order: HashSet<ChunkSectionPos> = positions.iter().copied().collect();
            let start = Instant::now();
            for pos in order {
                renderer.remove(pos);
            }
            let remove_ms = start.elapsed().as_secs_f64() * 1000.0;

            let (mut renderer, _) = filled(r);
            let start = Instant::now();
            let records: Vec<ChunkBuffers> = renderer.chunks.drain().map(|(_, c)| c).collect();
            drop(records);
            let hash_drop_ms = start.elapsed().as_secs_f64() * 1000.0;

            let (mut renderer, _) = filled(r);
            let start = Instant::now();
            let mut records: Vec<(u64, ChunkBuffers)> = renderer
                .chunks
                .drain()
                .map(|(_, c)| {
                    let key = c
                        .layers
                        .iter()
                        .flatten()
                        .map(|s| s.offset)
                        .min()
                        .unwrap_or(0);
                    (key, c)
                })
                .collect();
            records.sort_unstable_by_key(|(key, _)| *key);
            drop(records);
            let ordered_drop_ms = start.elapsed().as_secs_f64() * 1000.0;

            let (mut renderer, positions) = filled(r);
            let set: HashSet<ChunkSectionPos> = positions.iter().copied().collect();
            let start = Instant::now();
            renderer.remove_sections(&set);
            let bulk_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(renderer.loaded_sections(), 0);
            println!(
                "render distance {r}: remove() {remove_ms:.3} ms; drop records in hash order \
                 {hash_drop_ms:.3} ms; drop in offset order {ordered_drop_ms:.3} ms; \
                 remove_sections {bulk_ms:.3} ms"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn visible_info(pos: ChunkSectionPos) -> VisibleSection {
        VisibleSection {
            pos,
            entered_through: None,
            depth: 0,
        }
    }

    fn translucent_chunk(marker: f32) -> ChunkBuffers {
        let vertex = TerrainVertex::new([marker, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
        let mut translucent_cpu = Buffer::new();
        translucent_cpu.push_quad([vertex; 4]);
        ChunkBuffers {
            layers: std::array::from_fn(|_| None),
            visibility: SetVisibility::all(true),
            occlusion: SectionOcclusion::default(),
            translucent_cpu,
        }
    }

    #[test]
    fn translucent_resort_job_is_visible_only_and_carries_resident_version() {
        let visible = ChunkSectionPos::new(0, 0, 0);
        let hidden = ChunkSectionPos::new(1, 0, 0);
        let mut chunks = HashMap::new();
        chunks.insert(visible, translucent_chunk(1.0));
        chunks.insert(hidden, translucent_chunk(2.0));

        let job = translucent_resort_job(
            &[visible_info(visible)],
            &chunks,
            42,
            DVec3::new(3.0, 4.0, 5.0),
        )
        .expect("the visible section has translucent geometry");
        assert_eq!(job.resident_version, 42);
        assert_eq!(job.sections.len(), 1);
        assert_eq!(job.sections[0].0, visible);
        assert!(job.sections.iter().all(|(pos, _)| *pos != hidden));
    }

    #[test]
    fn translucent_resort_job_caps_visible_sections_at_fifteen() {
        let positions: Vec<_> = (0..MAX_TRANSLUCENT_RESORT_SECTIONS + 1)
            .map(|x| ChunkSectionPos::new(x as i32, 0, 0))
            .collect();
        let visible: Vec<_> = positions.iter().copied().map(visible_info).collect();
        let mut chunks = HashMap::new();
        for (index, pos) in positions.iter().copied().enumerate() {
            chunks.insert(pos, translucent_chunk(index as f32));
        }

        let job = translucent_resort_job(&visible, &chunks, 9, DVec3::default())
            .expect("the visible prefix has translucent geometry");
        assert_eq!(job.sections.len(), MAX_TRANSLUCENT_RESORT_SECTIONS);
        assert_eq!(
            job.sections.first().map(|section| section.0),
            Some(positions[0])
        );
        assert_eq!(
            job.sections.last().map(|section| section.0),
            Some(positions[MAX_TRANSLUCENT_RESORT_SECTIONS - 1])
        );
    }

    fn air_record() -> ChunkBuffers {
        ChunkBuffers {
            layers: std::array::from_fn(|_| None),
            visibility: SetVisibility::all(true),
            occlusion: SectionOcclusion::default(),
            translucent_cpu: Buffer::new(),
        }
    }

    fn key(visibility: SetVisibility, occlusion: &SectionOcclusion, layers: u8) -> ResidentKey<'_> {
        ResidentKey {
            visibility,
            occlusion,
            layer_mask: layers,
        }
    }

    #[test]
    fn an_all_air_record_has_an_empty_layer_mask() {
        // `begin_frame` keeps a walked section in the draw list only when this is non-zero, and
        // a record without any GPU slice can never yield a `DrawItem` in any layer.
        assert_eq!(air_record().layer_mask(), 0);
        assert_eq!(air_record().resident_key().layer_mask, 0);
        // A CPU translucent copy without a GPU slice draws nothing either.
        assert_eq!(translucent_chunk(1.0).layer_mask(), 0);
    }

    #[test]
    fn resident_changes_invalidate_only_the_caches_they_can_reach() {
        let air = SectionOcclusion::default();
        let all = SetVisibility::all(true);

        // #30: an all-air record replaced by an identical one (light-only re-dirties, bridging
        // over the void, join bursts) changes nothing the walk or any plan can observe.
        let none = ResidentChange::default();
        assert_eq!(
            ResidentChange::between(Some(key(all, &air, 0)), Some(key(all, &air, 0))),
            none
        );
        assert_eq!(ResidentChange::between(None, None), none);

        // Presence always reaches the walk (`None` is a DUMMY hard stop for it), plus every
        // layer the record draws.
        assert_eq!(
            ResidentChange::between(None, Some(key(all, &air, 0))),
            ResidentChange {
                walk: true,
                layers: 0
            }
        );
        assert_eq!(
            ResidentChange::between(Some(key(all, &air, 0b0101)), None),
            ResidentChange {
                walk: true,
                layers: 0b0101
            }
        );

        // A geometry-only re-mesh keeps the walk and rebuilds the layers either version draws:
        // the old ranges are released, the new ones are new.
        assert_eq!(
            ResidentChange::between(Some(key(all, &air, 0b0001)), Some(key(all, &air, 0b1000))),
            ResidentChange {
                walk: false,
                layers: 0b1001
            }
        );

        // A different visibility matrix, or a different camera-cell opacity snapshot, re-walks.
        let mut tunnel = SetVisibility::all(false);
        tunnel.set_visible(mc_math::Facing::North, mc_math::Facing::South);
        assert_eq!(
            ResidentChange::between(Some(key(all, &air, 0)), Some(key(tunnel, &air, 0))),
            ResidentChange {
                walk: true,
                layers: 0
            }
        );
        let mut graph = crate::terrain::VisGraph::new();
        graph.set_opaque(3, 4, 5);
        let one_cell = graph.occlusion();
        assert_eq!(
            ResidentChange::between(
                Some(key(all, &air, 0b0001)),
                Some(key(all, &one_cell, 0b0001))
            ),
            ResidentChange {
                walk: true,
                layers: 0b0001
            }
        );
    }

    #[test]
    fn the_persistent_resort_worker_returns_exactly_the_inline_result() {
        // Several sections with out-of-order quads, so a sort really permutes them.
        let positions = [
            ChunkSectionPos::new(0, 0, 0),
            ChunkSectionPos::new(1, 0, 0),
            ChunkSectionPos::new(0, 1, 2),
        ];
        let mut chunks = HashMap::new();
        for (index, pos) in positions.iter().enumerate() {
            let mut chunk = translucent_chunk(index as f32);
            for y in [3.0f32, 9.0, 1.0, 14.0, 6.0] {
                let v =
                    |x: f32, z: f32| TerrainVertex::new([x, y, z], [0.0, 0.0], [255; 4], [0; 4]);
                chunk.translucent_cpu.push_quad([
                    v(0.0, 0.0),
                    v(1.0, 0.0),
                    v(1.0, 1.0),
                    v(0.0, 1.0),
                ]);
            }
            chunks.insert(*pos, chunk);
        }
        let visible: Vec<_> = positions.iter().copied().map(visible_info).collect();
        let camera = DVec3::new(4.5, 30.0, 7.25);

        let inline = run_translucent_resort_job(
            translucent_resort_job(&visible, &chunks, 5, camera).expect("job"),
            &AtomicBool::new(false),
        )
        .expect("not cancelled");

        // Several submissions in a row go through the same long-lived worker; each must return
        // the inline answer bit for bit.
        for _ in 0..3 {
            let (sender, receiver) = mpsc::sync_channel(1);
            submit_translucent_resort(TranslucentResortTask {
                job: translucent_resort_job(&visible, &chunks, 5, camera).expect("job"),
                cancelled: Arc::new(AtomicBool::new(false)),
                result: sender,
            });
            let threaded = receiver
                .recv_timeout(Duration::from_secs(10))
                .expect("the resort worker answers");
            assert_eq!(threaded.resident_version, inline.resident_version);
            assert_eq!(threaded.sections, inline.sections);
        }

        // A cancelled task never sends a result; its sender is dropped instead.
        let (sender, receiver) = mpsc::sync_channel(1);
        submit_translucent_resort(TranslucentResortTask {
            job: translucent_resort_job(&visible, &chunks, 6, camera).expect("job"),
            cancelled: Arc::new(AtomicBool::new(true)),
            result: sender,
        });
        assert!(matches!(
            receiver.recv_timeout(Duration::from_secs(10)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ));
    }

    #[test]
    fn stale_translucent_resort_result_is_rejected_after_resident_change() {
        assert!(accepts_translucent_resort(7, 7));
        assert!(!accepts_translucent_resort(8, 7));
        assert!(!accepts_translucent_resort(7, 8));
    }

    #[test]
    fn cancelled_translucent_resort_worker_drops_before_sorting() {
        let pos = ChunkSectionPos::new(0, 0, 0);
        let mut chunks = HashMap::new();
        chunks.insert(pos, translucent_chunk(1.0));
        let job = translucent_resort_job(&[visible_info(pos)], &chunks, 11, DVec3::default())
            .expect("visible translucent section");
        let cancelled = AtomicBool::new(true);
        assert!(run_translucent_resort_job(job, &cancelled).is_none());
    }

    /// java: rebuilding a chunk cancels only that chunk's own pending resort
    /// (`RenderChunk.makeCompileTaskChunk` -> `finishCompileTask`); the other chunks' resorts
    /// complete, and `prevRenderSort*` is written only by `renderBlockLayer`'s >1-block check.
    #[test]
    fn a_rebuilt_section_drops_only_its_own_part_of_an_in_flight_resort() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let atlas = Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let one_quad = |marker: f32| {
            let vertex = TerrainVertex::new([marker, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
            let mut buffer = Buffer::new();
            buffer.push_quad([vertex; 4]);
            buffer
        };
        let translucent_mesh = |pos, marker| {
            let mut mesh = ChunkMesh::empty(pos);
            mesh.translucent = one_quad(marker);
            mesh
        };
        let rebuilt = ChunkSectionPos::new(0, 0, 0);
        let kept = ChunkSectionPos::new(1, 0, 0);
        let mut renderer = TerrainRenderer::new(&gpu, &atlas);
        renderer.upload(&gpu, translucent_mesh(rebuilt, 1.0), DVec3::ZERO);
        renderer.upload(&gpu, translucent_mesh(kept, 2.0), DVec3::ZERO);

        // A resort in flight over both sections, as `schedule_translucent_resort` leaves it.
        let sort_camera = DVec3::new(0.5, 0.5, 0.5);
        let (sender, receiver) = mpsc::sync_channel(1);
        renderer.translucent_resort = Some(receiver);
        renderer.translucent_resort_cancel = Some(Arc::new(AtomicBool::new(false)));
        renderer.translucent_resort_version = 4;
        renderer.translucent_resort_sections = vec![rebuilt, kept];
        renderer.last_sort_pos = sort_camera;

        // One of its sections is rebuilt before the worker answers.
        renderer.upload(&gpu, translucent_mesh(rebuilt, 3.0), DVec3::ZERO);
        let rebuilt_mesh = renderer.chunks[&rebuilt]
            .translucent_cpu
            .as_bytes()
            .to_vec();
        assert!(
            renderer.translucent_resort.is_some(),
            "a rebuild must not cancel the other sections' resort"
        );
        assert_eq!(renderer.translucent_resort_sections, [kept]);
        assert_eq!(
            renderer.last_sort_pos, sort_camera,
            "a rebuild must not force a resort the >1-block check would not"
        );

        // The worker sorted clones taken at scheduling; the markers make any write visible.
        let sorted_kept = one_quad(8.0);
        sender
            .send(TranslucentResortResult {
                resident_version: 4,
                sections: vec![(rebuilt, one_quad(7.0)), (kept, sorted_kept.clone())],
            })
            .expect("receiver is held by the renderer");
        renderer.poll_translucent_resort(&gpu);

        assert_eq!(
            renderer.chunks[&kept].translucent_cpu.as_bytes(),
            sorted_kept.as_bytes(),
            "the untouched section takes its sorted buffer"
        );
        assert_eq!(
            renderer.chunks[&rebuilt].translucent_cpu.as_bytes(),
            rebuilt_mesh,
            "the sort of the replaced mesh is never written over the new one"
        );
        assert!(renderer.translucent_resort.is_none());
        assert!(renderer.translucent_resort_sections.is_empty());
    }

    #[test]
    fn java_upload_deadline_runs_the_first_task_even_after_expiry() {
        let expired = Instant::now() - Duration::from_nanos(1);
        let mut deadline = UploadTaskDeadline::timed(expired);

        // `runChunkUploads` polls and runs its first queued FutureTask before checking the
        // deadline. A mutant that checks the clock before the first task would reject this.
        assert!(deadline.permits_next(Instant::now()));
        deadline.complete_task();
        assert!(!deadline.permits_next(Instant::now()));
    }

    #[test]
    fn java_upload_deadline_accepts_exact_finish_time_but_not_after() {
        let finish = Instant::now();
        let mut deadline = UploadTaskDeadline::timed(finish);
        deadline.complete_task();

        // Java tests `finishTimeNano - System.nanoTime() < 0`, so equality still drains another
        // task. Changing `<=` to `<` makes this exact-boundary control fail.
        assert!(deadline.permits_next(finish));
        assert!(!deadline.permits_next(finish + Duration::from_nanos(1)));
    }

    #[test]
    fn java_upload_zero_finish_time_runs_at_most_one_task() {
        let mut deadline = UploadTaskDeadline::zero_finish_time();
        assert!(deadline.permits_next(Instant::now()));
        deadline.complete_task();
        assert!(!deadline.permits_next(Instant::now()));
    }

    #[test]
    fn revisioned_upload_reports_input_stale_retained_stale_and_deadline_counts() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let atlas = Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let input_pos = ChunkSectionPos::new(0, 0, 0);
        let retained_stale_pos = ChunkSectionPos::new(1, 0, 0);
        let retained_deadline_pos = ChunkSectionPos::new(2, 0, 0);
        let mut renderer = TerrainRenderer::new(&gpu, &atlas);

        // A worker result carrying revision zero loses to an already accepted revision one
        // before it can enter the retained ready queue.
        assert!(renderer.accept_mesh_revision(input_pos, 1));

        // A retained revision-zero task is stale at dequeue time. The second retained task is
        // valid but remains queued because the expired deadline permits only the first task.
        assert!(renderer.accept_mesh_revision(retained_stale_pos, 1));
        renderer
            .ready_uploads
            .push_latest_revisioned(MeshTaskResult::unversioned(marked_mesh(
                retained_stale_pos,
                1.0,
            )));
        renderer
            .ready_uploads
            .push_latest(marked_mesh(retained_deadline_pos, 2.0));

        let result = renderer.upload_all_with_deadline_retaining_revisioned(
            &gpu,
            vec![MeshTaskResult::unversioned(marked_mesh(input_pos, 3.0))],
            DVec3::ZERO,
            Instant::now() - Duration::from_nanos(1),
        );

        assert_eq!(result.input_revision_rejections(), 1);
        assert_eq!(result.retained_revision_rejections(), 1);
        assert_eq!(result.retained_ready_count(), 1);
        assert!(result.deadline_blocked());
        assert!(result.uploaded_positions().is_empty());
        assert_eq!(
            result.retained_positions(),
            &[retained_deadline_pos],
            "deadline-deferred work remains the same completed mesh"
        );
    }

    fn marked_mesh(pos: ChunkSectionPos, marker: f32) -> ChunkMesh {
        let mut mesh = ChunkMesh::empty(pos);
        let vertex = TerrainVertex::new([marker, 0.0, 0.0], [0.0, 0.0], [255; 4], [0; 4]);
        mesh.solid.push_quad([vertex; 4]);
        mesh
    }

    #[test]
    fn deadline_deferred_mesh_stays_ready_instead_of_becoming_a_remesh() {
        let first_pos = ChunkSectionPos::new(0, 0, 0);
        let second_pos = ChunkSectionPos::new(1, 0, 0);
        let mut ready = ReadyMeshQueue::default();
        ready.push_latest(marked_mesh(first_pos, 1.0));
        ready.push_latest(marked_mesh(second_pos, 2.0));

        // An expired finish time still permits Java's first queued task. The second task must
        // remain as the same completed mesh for the next frame; a mutant that drops the task
        // instead of pushing it back would make this assertion fail.
        let mut deadline = UploadTaskDeadline::timed(Instant::now() - Duration::from_nanos(1));
        let first = ready
            .pop_if_permitted(&deadline)
            .expect("the first queued task is unconditional");
        assert_eq!(first.pos, first_pos);
        deadline.complete_task();
        assert!(ready.pop_if_permitted(&deadline).is_none());
        assert_eq!(ready.positions(), vec![second_pos]);
    }

    #[test]
    fn a_new_mesh_cancels_an_older_retained_result_for_the_same_section() {
        let pos = ChunkSectionPos::new(3, 2, -1);
        let mut ready = ReadyMeshQueue::default();
        ready.push_latest(marked_mesh(pos, 1.0));
        ready.push_latest(marked_mesh(pos, 2.0));

        assert_eq!(ready.tasks.len(), 1);
        let latest = ready.tasks.front().expect("latest task");
        assert_eq!(latest.solid.vertices()[0].pos[0], 2.0);
        assert!(ready.cancel(pos));
        assert!(ready.tasks.is_empty());
        assert!(!ready.cancel(pos));
    }

    /// #2 + #30 end to end: air records carry the walk to remote geometry but never enter the
    /// draw list, and an air-over-air upload invalidates neither the walk nor any plan. On the
    /// pre-#30 renderer the second frame below re-walks and rebuilds all four plans.
    #[test]
    fn air_records_are_walked_not_drawn_and_air_over_air_invalidates_nothing() {
        let Ok(mut gpu) = Gpu::headless(32, 32, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let atlas = Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let mut phases = PhaseList::new();
        let shared = TerrainPhase::install(&gpu, &atlas, &mut phases);
        let camera_section = ChunkSectionPos::new(0, 4, 0);
        let corridor = ChunkSectionPos::new(0, 4, 1);
        let island = ChunkSectionPos::new(0, 4, 2);
        let mut camera = Camera {
            pos: DVec3::new(8.0, 72.0, 8.0),
            yaw: 0.0,
            pitch: 0.0,
            fov: 70.0,
            near: 0.05,
            far: 256.0,
            aspect: 1.0,
            ..Camera::default()
        };
        camera.update();
        {
            let mut r = shared.borrow_mut();
            r.render_distance = 4;
            r.upload(&gpu, ChunkMesh::empty(camera_section), camera.pos);
            r.upload(&gpu, ChunkMesh::empty(corridor), camera.pos);
            r.upload(&gpu, marked_mesh(island, 1.0), camera.pos);
        }
        let frame = |gpu: &mut Gpu, phases: &mut PhaseList| {
            let target = gpu.acquire().expect("acquire");
            phases.render(gpu, target.view(), &camera, 0.0);
        };

        frame(&mut gpu, &mut phases);
        {
            let r = shared.borrow();
            assert_eq!(
                r.visible_section_positions().collect::<Vec<_>>(),
                vec![camera_section, corridor, island],
                "the walk crosses both air records"
            );
            assert_eq!(r.stats().visible_sections, 1, "only the island is drawn");
            assert_eq!(r.stats().draw_items, 1);
            assert_eq!(r.cull_cache_counts(), (0, 1));
            assert_eq!(r.draw_plan_cache_counts(), (4, 0));
        }
        let first_list = shared.borrow().draw_list_key;

        // Air over identical air (a light-only re-dirty, a void-map bridge): nothing moves.
        shared
            .borrow_mut()
            .upload(&gpu, ChunkMesh::empty(corridor), camera.pos);
        frame(&mut gpu, &mut phases);
        {
            let r = shared.borrow();
            assert_eq!(r.cull_cache_counts(), (1, 1), "air over air keeps the walk");
            assert_eq!(r.draw_plan_cache_counts(), (4, 4), "and every layer plan");
            assert_eq!(r.stats().draw_items, 1);
            assert_eq!(
                r.draw_list_key, first_list,
                "and the still camera's draw list"
            );
        }

        // A solid-only re-mesh with the same matrix keeps the walk and rebuilds one plan.
        shared
            .borrow_mut()
            .upload(&gpu, marked_mesh(island, 2.0), camera.pos);
        frame(&mut gpu, &mut phases);
        {
            let r = shared.borrow();
            assert_eq!(r.cull_cache_counts(), (2, 1));
            assert_eq!(r.draw_plan_cache_counts(), (5, 7));
            assert_ne!(
                r.draw_list_key, first_list,
                "a geometry change rebuilds the list"
            );
            assert_eq!(r.stats().draw_items, 1);
        }

        // Removing the corridor is a presence change: re-walk, and the island is cut off (no
        // world-column authority, so a missing record is a DUMMY hard stop). The camera's air
        // record is still reached, so the passes still open, now with nothing to draw.
        shared.borrow_mut().remove(corridor);
        frame(&mut gpu, &mut phases);
        {
            let r = shared.borrow();
            assert_eq!(r.cull_cache_counts(), (2, 2));
            assert_eq!(r.stats().visible_sections, 0);
            assert_eq!(r.stats().draw_items, 0);
            assert!(r.walk_reached_resident);
        }
    }

    #[test]
    fn ready_queue_cancel_finds_exactly_the_queued_tasks_through_its_revision_index() {
        let mut ready = ReadyMeshQueue::default();
        for x in 0..64 {
            ready.push_latest(marked_mesh(ChunkSectionPos::new(x, 0, 0), x as f32));
        }
        assert_eq!(
            ready.len(),
            64,
            "distinct positions never cancel each other"
        );
        assert!(!ready.cancel(ChunkSectionPos::new(99, 0, 0)));
        assert_eq!(ready.len(), 64);

        // A task put back by an expired deadline is still found and cancelled.
        let mut deadline = UploadTaskDeadline::timed(Instant::now() - Duration::from_nanos(1));
        let first = ready.pop_if_permitted(&deadline).expect("first task");
        assert_eq!(first.pos, ChunkSectionPos::new(0, 0, 0));
        deadline.complete_task();
        assert!(ready.pop_if_permitted(&deadline).is_none());
        assert!(ready.cancel(ChunkSectionPos::new(1, 0, 0)));
        assert!(!ready.cancel(ChunkSectionPos::new(1, 0, 0)));
        assert!(
            !ready.cancel(ChunkSectionPos::new(0, 0, 0)),
            "already popped"
        );
        assert_eq!(ready.len(), 62);
        assert_eq!(ready.revisions.len(), ready.tasks.len());
    }

    #[test]
    fn diagnostic_section_iterators_report_upload_visible_and_remove_identities() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let atlas = Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let pos = ChunkSectionPos::new(2, 3, -1);
        let mut renderer = TerrainRenderer::new(&gpu, &atlas);

        renderer.upload(&gpu, marked_mesh(pos, 1.0), DVec3::ZERO);
        assert_eq!(
            renderer.resident_section_positions().collect::<Vec<_>>(),
            vec![pos]
        );
        assert_eq!(renderer.pending_ready_upload_count(), 0);

        // The diagnostic iterator reports the walk's resident sections (all-air records included),
        // so drive the production walk from a camera inside the section: the camera's own
        // section is always the walk's first entry.
        let origin = pos.origin();
        let mut camera = crate::Camera {
            pos: DVec3::new(
                f64::from(origin.x) + 8.0,
                f64::from(origin.y) + 8.0,
                f64::from(origin.z) + 8.0,
            ),
            ..crate::Camera::default()
        };
        camera.update();
        renderer.prepare_culling_walk(&camera);
        assert_eq!(
            renderer.visible_section_positions().collect::<Vec<_>>(),
            vec![pos]
        );

        renderer.remove(pos);
        assert!(renderer.resident_section_positions().next().is_none());
        assert_eq!(renderer.pending_ready_upload_count(), 0);
    }

    /// The friend's AMD export crash. The export's
    /// deterministic mesher uploads through `upload` ("terrain batch"); once the device is lost,
    /// that was `create_buffer_init`'s panic "Buffer with 'terrain batch' label is invalid"
    /// (`gpu_health`'s test reproduces the message with wgpu's own call). Now the upload reports
    /// `false`, nothing is installed, the loss is recorded, and the arena path (normal play)
    /// keeps its mesh for a later frame instead of dropping it.
    #[test]
    fn a_lost_device_refuses_terrain_uploads_without_panicking() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let atlas = Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let mut renderer = TerrainRenderer::new(&gpu, &atlas);
        // Made before the loss: a renderer's own construction is not this test's subject.
        let mut live = TerrainRenderer::new(&gpu, &atlas);
        let before = ChunkSectionPos::new(0, 4, 0);
        assert!(renderer.upload(&gpu, marked_mesh(before, 1.0), DVec3::ZERO));
        assert_eq!(renderer.refused_uploads(), 0);

        gpu.device.destroy();
        let after = ChunkSectionPos::new(1, 4, 0);
        assert!(
            !renderer.upload(&gpu, marked_mesh(after, 2.0), DVec3::ZERO),
            "refused, not a panic"
        );
        assert!(renderer.refused_uploads() >= 1);
        assert!(gpu.is_device_lost(), "the refusal recorded the loss");
        assert_eq!(
            renderer.resident_section_positions().collect::<Vec<_>>(),
            vec![before],
            "only the section uploaded before the loss is resident"
        );

        // The arena (live play): a fresh renderer has no page yet, so the lost device refuses
        // one and the mesh stays queued.
        let result = live.upload_all_with_deadline_retaining(
            &gpu,
            vec![marked_mesh(after, 3.0)],
            DVec3::ZERO,
            Instant::now() + std::time::Duration::from_secs(1),
        );
        assert_eq!(result.uploaded_count(), 0);
        assert_eq!(
            live.pending_ready_upload_count(),
            1,
            "kept for a later frame"
        );
        assert!(live.resident_section_positions().next().is_none());
    }

    #[test]
    fn animated_atlas_rect_validation_matches_mip_bounds_and_payload() {
        let valid = AtlasAnimationUpload {
            mip_level: 1,
            x: 4,
            y: 2,
            width: 4,
            height: 3,
            rgba: vec![0xAB; 4 * 3 * 4],
        };
        assert!(atlas_animation_rect_valid(&valid, 16, 16, 3));

        let mut edge = valid.clone();
        edge.x = 5;
        assert!(!atlas_animation_rect_valid(&edge, 16, 16, 3));
        let mut wrong_level = valid.clone();
        wrong_level.mip_level = 3;
        assert!(!atlas_animation_rect_valid(&wrong_level, 16, 16, 3));
        let mut wrong_payload = valid;
        wrong_payload.rgba.pop();
        assert!(!atlas_animation_rect_valid(&wrong_payload, 16, 16, 3));

        let oversized = AtlasAnimationUpload {
            mip_level: 0,
            x: 0,
            y: 0,
            width: u32::MAX,
            height: u32::MAX,
            rgba: Vec::new(),
        };
        assert!(!atlas_animation_rect_valid(&oversized, 16, 16, 1));
    }

    #[test]
    fn terrain_anaglyph_upload_mixes_colours_once_and_preserves_alpha() {
        let source = [
            255, 0, 0, 7, // red, translucent
            0, 255, 0, 19, // green
            0, 0, 255, 31, // blue
            128, 128, 128, 64,
        ];
        assert_eq!(terrain_upload_rgba(&source, false), source);
        assert_eq!(
            terrain_upload_rgba(&source, true),
            [76, 76, 76, 7, 178, 150, 0, 19, 0, 28, 178, 31, 128, 128, 128, 64,],
            "Java's integer anaglyph transform mixes RGB only"
        );
    }

    #[test]
    fn java_rgba_atlas_bytes_survive_gpu_vertex_and_lightmap_modulation() {
        let Ok(mut gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };

        // A non-white texel makes an sRGB allocation observable: sampling the old
        // `Rgba8UnormSrgb` texture would linearise these bytes before the multipliers below.
        let atlas_byte = [200u8, 120, 60, 255];
        let atlas = Atlas {
            texture: atlas_byte.repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        };
        let renderer = TerrainRenderer::new(&gpu, &atlas);
        assert_eq!(JAVA_TERRAIN_ATLAS_FORMAT, wgpu::TextureFormat::Rgba8Unorm);

        // This tiny probe uses the same normalized texture sample and byte-valued factors as
        // `terrain.wgsl`: Java's BLOCK vertex colour and EntityRenderer lightmap are both
        // multiplied after sampling the plain GL_RGBA atlas.
        const PROBE_WGSL: &str = r#"
@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  3.0),
        vec2<f32>( 3.0, -1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    let atlas_sample = textureSampleLevel(atlas, atlas_sampler, vec2<f32>(0.5, 0.5), 0.0);
    let vertex_colour = vec4<f32>(128.0, 160.0, 192.0, 255.0) / 255.0;
    let lightmap = vec4<f32>(224.0, 192.0, 160.0, 255.0) / 255.0;
    return atlas_sample * vertex_colour * lightmap;
}
"#;

        let view = renderer
            .atlas_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain atlas byte-contract probe sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let bind_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain atlas byte-contract probe layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain atlas byte-contract probe bind group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("terrain atlas byte-contract probe shader"),
                source: wgpu::ShaderSource::Wgsl(PROBE_WGSL.into()),
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain atlas byte-contract probe pipeline layout"),
                bind_group_layouts: &[Some(&bind_layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("terrain atlas byte-contract probe pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let frame = gpu.acquire().expect("acquire atlas contract probe target");
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain atlas byte-contract probe encoder"),
            });
        drop(gpu.begin_clear_pass(&mut encoder, frame.view(), wgpu::Color::BLACK));
        {
            let mut pass = gpu.begin_pass(
                &mut encoder,
                frame.view(),
                "terrain atlas byte-contract probe pass",
                crate::DepthAttachment::None,
            );
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        gpu.queue.submit(Some(encoder.finish()));

        let pixels = gpu.read_pixels().expect("read atlas contract probe pixels");
        let center = ((4 * 8 + 4) * 4) as usize;
        let actual = [
            pixels[center],
            pixels[center + 1],
            pixels[center + 2],
            pixels[center + 3],
        ];
        let expected = [
            ((u32::from(atlas_byte[0]) * 128 * 224) / (255 * 255)) as u8,
            ((u32::from(atlas_byte[1]) * 160 * 192) / (255 * 255)) as u8,
            ((u32::from(atlas_byte[2]) * 192 * 160) / (255 * 255)) as u8,
            255,
        ];
        for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                actual.abs_diff(expected) <= 1,
                "Java GL_RGBA byte contract channel {channel}: expected {expected}, got {actual}; full pixel {actual:?}"
            );
        }
    }

    #[test]
    fn terrain_anaglyph_animation_updates_the_raw_mip_source() {
        let Ok(gpu) = Gpu::headless(16, 16, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let base = [255u8, 0, 0, 7].repeat(4 * 4);
        let mip = [0u8, 0, 255, 19].repeat(2 * 2);
        let atlas = Atlas {
            texture: base.clone(),
            mipmaps: vec![mip.clone()],
            w: 4,
            h: 4,
        };
        let renderer = TerrainRenderer::new(&gpu, &atlas);
        let animated = AtlasAnimationUpload {
            mip_level: 1,
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            rgba: [0u8, 255, 0, 23].repeat(2),
        };
        assert!(renderer.upload_atlas_animation(&gpu, &animated));

        let mut expected_mip = mip;
        expected_mip[4..8].copy_from_slice(&animated.rgba[..4]);
        expected_mip[12..16].copy_from_slice(&animated.rgba[4..8]);
        assert_eq!(
            renderer.atlas_source_levels.borrow().as_slice(),
            &[base, expected_mip],
            "mip animation must retain raw source bytes, not transformed GPU bytes"
        );
    }

    #[test]
    fn terrain_live_anaglyph_toggle_replays_atlas_mips_and_lightmap_from_raw_bytes() {
        let Ok(gpu) = Gpu::headless(16, 16, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let base = [255u8, 0, 0, 7].repeat(4 * 4);
        let mip = [0u8, 0, 255, 19].repeat(2 * 2);
        let atlas = Atlas {
            texture: base.clone(),
            mipmaps: vec![mip.clone()],
            w: 4,
            h: 4,
        };
        let mut renderer = TerrainRenderer::new(&gpu, &atlas);
        let lightmap = Lightmap {
            rgba: [0u8, 255, 0, 37].repeat((LIGHTMAP_SIDE * LIGHTMAP_SIDE) as usize),
        };
        renderer.set_lightmap(&gpu, &lightmap);
        let raw_levels = renderer.atlas_source_levels.borrow().clone();
        let raw_lightmap = renderer.lightmap_source_rgba.clone();

        renderer.set_anaglyph_with_gpu(&gpu, true);
        assert!(renderer.anaglyph_enabled());
        assert_eq!(renderer.atlas_source_levels.borrow().as_slice(), raw_levels);
        assert_eq!(renderer.lightmap_source_rgba, raw_lightmap);

        renderer.set_anaglyph_with_gpu(&gpu, false);
        assert!(!renderer.anaglyph_enabled());
        assert_eq!(renderer.atlas_source_levels.borrow().as_slice(), raw_levels);
        assert_eq!(renderer.lightmap_source_rgba, raw_lightmap);
    }

    #[test]
    fn only_the_plain_cutout_pass_disables_atlas_mipmaps() {
        // This discriminates the two passes. Mutating the old one-sampler implementation
        // (`true` for all four) makes the CUTOUT assertion fail.
        assert!(layer_uses_mipmaps(layer_index(RenderLayer::Solid)));
        assert!(layer_uses_mipmaps(layer_index(RenderLayer::CutoutMipped)));
        assert!(!layer_uses_mipmaps(layer_index(RenderLayer::Cutout)));
        assert!(layer_uses_mipmaps(layer_index(RenderLayer::Translucent)));
    }

    #[test]
    fn the_gpu_budget_env_override_parses_or_falls_back() {
        // The good case, in mebibytes — the unit F5's experiment is written in.
        assert_eq!(parse_budget_mb("2048"), Some(GpuBudget::new(2048 << 20)));
        assert_eq!(parse_budget_mb(" 512 "), Some(GpuBudget::new(512 << 20)));
        assert_eq!(parse_budget_mb("1"), Some(GpuBudget::new(1 << 20)));

        // Every way of being wrong falls back rather than shrinking the cap. A knob that
        // silently set the budget to zero on a typo would manufacture the exact eviction churn
        // it exists to diagnose.
        for bad in ["", "  ", "0", "-1", "2048MB", "2.5", "abc", "0x400", "1e3"] {
            assert_eq!(parse_budget_mb(bad), None, "{bad:?} must not parse");
        }
        assert_eq!(
            parse_budget_mb("junk").unwrap_or_default(),
            GpuBudget::default(),
            "the fallback is the compiled-in 512 MiB"
        );
        assert_eq!(GpuBudget::default().limit(), GpuBudget::DEFAULT_BYTES);

        // And it cannot overflow into a small number on a huge input.
        assert_eq!(
            parse_budget_mb(&u64::MAX.to_string()),
            Some(GpuBudget::new(u64::MAX))
        );
    }

    #[test]
    fn quad_indices_are_two_triangles_per_quad() {
        assert_eq!(quad_indices(1), vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(quad_indices(2)[6..], [4, 5, 6, 4, 6, 7]);
        assert_eq!(quad_indices(100).len(), 600);
    }

    #[test]
    fn layer_indices_follow_vanilla_ordinals() {
        assert_eq!(layer_index(RenderLayer::Solid), 0);
        assert_eq!(layer_index(RenderLayer::CutoutMipped), 1);
        assert_eq!(layer_index(RenderLayer::Cutout), 2);
        assert_eq!(layer_index(RenderLayer::Translucent), 3);
    }

    #[test]
    fn translucent_submission_reverses_the_bfs_visible_list_only() {
        let positions = [0usize, 1, 2, 3];
        let solid: Vec<_> = positions
            .iter()
            .map(|&position| submission_index(position, positions.len(), RenderLayer::Solid))
            .collect();
        let cutout: Vec<_> = positions
            .iter()
            .map(|&position| submission_index(position, positions.len(), RenderLayer::Cutout))
            .collect();
        let translucent: Vec<_> = positions
            .iter()
            .map(|&position| submission_index(position, positions.len(), RenderLayer::Translucent))
            .collect();

        assert_eq!(solid, [0, 1, 2, 3]);
        assert_eq!(cutout, [0, 1, 2, 3]);
        assert_eq!(translucent, [3, 2, 1, 0]);
    }

    #[test]
    fn the_instance_stride_matches_the_slice_size_the_draw_loop_uses() {
        assert_eq!(std::mem::size_of::<ChunkOffset>(), 16);
    }

    #[test]
    fn chunk_vertices_expand_about_the_local_center_before_world_translation() {
        const SCALE: f32 = 1.000_001;
        let expanded = |local: f32| (local - 8.0) * SCALE + 8.0;
        assert!((expanded(0.0) - -0.000_008).abs() < 0.000_001);
        assert_eq!(expanded(8.0), 8.0);
        assert!((expanded(16.0) - 16.000_008).abs() < 0.000_001);

        let shader = include_str!("terrain.wgsl");
        assert!(shader
            .contains("(local - vec3<f32>(8.0)) * 1.000001 + vec3<f32>(8.0) + chunk_offset.xyz"));
    }

    #[test]
    fn terrain_alpha_cutoffs_follow_the_java_render_world_state_machine() {
        // Java: renderWorld seeds GL_GREATER 0.5; renderWorldPass re-enables that state for both
        // cutout layers, then changes it to 0.1 only before TRANSLUCENT.
        let layers = [
            RenderLayer::Solid,
            RenderLayer::CutoutMipped,
            RenderLayer::Cutout,
            RenderLayer::Translucent,
        ];
        let cutoffs = layers.map(|layer| layer_alpha_cutoff(layer_index(layer)));
        assert_eq!(cutoffs, [0.0, 0.5, 0.5, 0.1]);

        // CPU oracle for `terrain.wgsl`: GL_GREATER rejects equality. The semi-alpha sample is
        // the negative control: the old 0.1 cutout mutant would incorrectly retain it.
        let passes = |alpha: f32| cutoffs.map(|cutoff| alpha > cutoff);
        assert_eq!(passes(0.25), [true, false, false, true]);
        assert_eq!(passes(0.5), [true, false, false, true]);
        assert_eq!(passes(0.51), [true, true, true, true]);
    }

    #[test]
    fn the_fog_uniform_packs_the_way_sky_fog_wgsl_does() {
        // Off: the enable flag is the only thing the shader looks at, and it is 0.
        let off = TerrainUniform::new(0.1, None);
        assert_eq!(off.params, [0.1, 0.0, 0.0, 0.0]);

        // Linear: mode flag 0 in `fog_color.a`, start/end in x/y.
        let linear = TerrainUniform::new(
            0.0,
            Some(FogState {
                color: glam::Vec3::new(0.25, 0.5, 0.75),
                mode: FogMode::Linear {
                    start: 12.0,
                    end: 48.0,
                },
            }),
        );
        assert_eq!(linear.params[1], 1.0);
        assert_eq!(linear.fog_color, [0.25, 0.5, 0.75, 0.0]);
        assert_eq!(linear.fog_params, [12.0, 48.0, 0.0, 0.0]);

        // Exp: mode flag 1, density in z.
        let exp = TerrainUniform::new(
            0.0,
            Some(FogState {
                color: glam::Vec3::ZERO,
                mode: FogMode::Exp { density: 2.0 },
            }),
        );
        assert_eq!(exp.fog_color[3], 1.0);
        assert_eq!(exp.fog_params, [0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn the_terrain_uniform_stays_std140_compatible() {
        // Three `vec4<f32>`, matching `terrain.wgsl`.
        assert_eq!(std::mem::size_of::<TerrainUniform>(), 48);
        assert_eq!(std::mem::align_of::<TerrainUniform>(), 4);
    }

    #[test]
    fn the_default_lightmap_is_a_full_sixteen_by_sixteen_rgba_table() {
        let lm = Lightmap::generate(LightmapParams::default());
        assert_eq!(
            lm.rgba.len(),
            (LIGHTMAP_SIDE * LIGHTMAP_SIDE * 4) as usize,
            "the upload path assumes exactly this many bytes"
        );
    }

    #[test]
    fn the_live_resident_cache_feeds_compiled_visgraph_data_to_the_section_walk() {
        // The detailed sealed-wall/open-tunnel behavior lives in `terrain::cull`. This test
        // bites the production resident-cache adapter instead of only closure-backed fixtures.
        let pos = ChunkSectionPos::new(3, 4, -2);
        let mut compiled = SetVisibility::all(false);
        compiled.set_visible(mc_math::Facing::North, mc_math::Facing::South);
        let mut chunks = HashMap::new();
        chunks.insert(
            pos,
            ChunkBuffers {
                layers: std::array::from_fn(|_| None),
                visibility: compiled,
                occlusion: SectionOcclusion::default(),
                translucent_cpu: Buffer::new(),
            },
        );

        assert_eq!(SectionSource::visibility(&chunks, pos), Some(compiled));
        assert_eq!(
            SectionSource::visibility(&chunks, ChunkSectionPos::new(4, 4, -2)),
            None,
            "an unmeshed section must remain a hard stop for the live walk"
        );
    }

    #[test]
    fn authoritative_world_membership_distinguishes_air_from_loaded_unmeshed() {
        let loaded = ChunkSectionPos::new(0, 4, 0);
        let absent = ChunkSectionPos::new(1, 4, 0);
        let chunks = HashMap::new();
        let mut columns = HashSet::new();
        columns.insert(loaded.chunk_pos());
        let source = AuthoritativeSectionSource {
            chunks: &chunks,
            world_columns: Some(&columns),
        };

        assert_eq!(
            source.visibility(loaded),
            None,
            "loaded but unmeshed columns retain DUMMY hard-stop semantics"
        );
        assert_eq!(
            source.visibility(absent),
            Some(SetVisibility::all(true)),
            "a column absent from authoritative world membership is settled air"
        );
        assert_eq!(
            source.visibility(ChunkSectionPos::new(1, 16, 0)),
            None,
            "world-column authority covers the client's sixteen render sections only"
        );

        let resident_visibility = SetVisibility::all(false);
        let mut resident_chunks = HashMap::new();
        resident_chunks.insert(
            loaded,
            ChunkBuffers {
                layers: std::array::from_fn(|_| None),
                visibility: resident_visibility,
                occlusion: SectionOcclusion::default(),
                translucent_cpu: Buffer::new(),
            },
        );
        let resident_source = AuthoritativeSectionSource {
            chunks: &resident_chunks,
            world_columns: Some(&columns),
        };
        assert_eq!(
            resident_source.visibility(loaded),
            Some(resident_visibility),
            "resident compiled visibility wins over lifecycle membership"
        );
    }
}

#[cfg(test)]
mod atlas_batch_owner_contract_tests {
    use super::*;

    // Fixed three-level atlas and explicit byte-offset expectations below form the oracle;
    // no production rectangle validator or source-copy helper computes expected output.
    fn atlas() -> mc_assets::Atlas {
        mc_assets::Atlas {
            texture: [17, 29, 43, 7].repeat(16),
            mipmaps: vec![[53, 67, 79, 19].repeat(4), vec![83, 97, 109, 31]],
            w: 4,
            h: 4,
        }
    }

    fn records() -> Vec<mc_assets::AtlasAnimationUpload> {
        let base = mc_assets::AtlasAnimationUpload {
            mip_level: 0,
            x: 1,
            y: 1,
            width: 2,
            height: 2,
            rgba: vec![
                151, 13, 79, 23, 2, 173, 61, 37, 83, 7, 199, 47, 113, 157, 11, 59,
            ],
        };
        let mip = mc_assets::AtlasAnimationUpload {
            mip_level: 1,
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            rgba: vec![3, 211, 41, 67, 227, 17, 89, 71],
        };
        let tail = mc_assets::AtlasAnimationUpload {
            mip_level: 2,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            rgba: vec![251, 101, 5, 83],
        };
        let mut short = base.clone();
        short.rgba.pop();
        let mut right = mip.clone();
        right.x = 2;
        let mut zero_width = base.clone();
        zero_width.width = 0;
        zero_width.rgba.clear();
        let mut level = tail.clone();
        level.mip_level = 3;
        let mut overflow = tail.clone();
        overflow.x = u32::MAX;
        let mut overlap = base.clone();
        overlap.rgba.reverse();
        let mut bottom = mip.clone();
        bottom.y = 1;
        let mut zero_height = base.clone();
        zero_height.height = 0;
        zero_height.rgba.clear();
        let mut long = tail.clone();
        long.rgba.push(1);
        vec![
            base,
            short,
            mip,
            right,
            zero_width,
            tail,
            level,
            overflow,
            overlap,
            bottom,
            zero_height,
            long,
        ]
    }

    #[test]
    fn actual_owner_sink_stream_shadow_settings_and_standalone_contract() {
        let gpu = Gpu::headless(16, 16, crate::context::GpuOptions::default())
            .expect("atlas owner qualification requires a working GPU adapter");
        let atlas = atlas();
        let mut direct = TerrainRenderer::new(&gpu, &atlas);
        let mut captured = TerrainRenderer::new(&gpu, &atlas);

        let mut expected = vec![
            atlas.texture.clone(),
            atlas.mipmaps[0].clone(),
            atlas.mipmaps[1].clone(),
        ];
        let stream = records();
        let mut receipts = Vec::new();
        for enabled in [false, true, false] {
            direct.set_anaglyph_with_gpu(&gpu, enabled);
            captured.set_anaglyph_with_gpu(&gpu, enabled);
            assert_eq!(
                *direct.atlas_source_levels.borrow(),
                expected,
                "toggle must retain raw bytes"
            );
            assert_eq!(
                *captured.atlas_source_levels.borrow(),
                expected,
                "toggle must retain raw bytes"
            );
            for (index, upload) in stream.iter().enumerate() {
                let accepted = matches!(index, 0 | 2 | 5 | 8);
                match index {
                    0 | 8 => {
                        expected[0][20..28].copy_from_slice(&upload.rgba[..8]);
                        expected[0][36..44].copy_from_slice(&upload.rgba[8..]);
                    }
                    2 => {
                        expected[1][4..8].copy_from_slice(&upload.rgba[..4]);
                        expected[1][12..16].copy_from_slice(&upload.rgba[4..]);
                    }
                    5 => expected[2].copy_from_slice(&upload.rgba),
                    _ => {}
                }
                let before_count = receipts.len();
                let actual =
                    captured.upload_atlas_animation_with_sink(upload, |texture, setting| {
                        assert!(
                            std::ptr::eq(texture, &captured.atlas_texture),
                            "sink must receive this owner's texture"
                        );
                        assert_eq!(
                            setting, enabled,
                            "sink must receive the owner's current setting"
                        );
                        // This borrow also proves the mutation borrow ended before the callback.
                        assert_eq!(
                            *captured.atlas_source_levels.borrow(),
                            expected,
                            "raw shadow must be ready before transfer"
                        );
                        receipts.push((enabled, index, upload.mip_level));
                    });
                assert_eq!(actual, accepted, "capture acceptance at record {index}");
                assert_eq!(
                    direct.upload_atlas_animation(&gpu, upload),
                    accepted,
                    "standalone acceptance at record {index}"
                );
                assert_eq!(
                    receipts.len() - before_count,
                    usize::from(accepted),
                    "one sink call per accepted record"
                );
                assert_eq!(
                    *captured.atlas_source_levels.borrow(),
                    expected,
                    "capture raw bytes at record {index}"
                );
                assert_eq!(
                    *direct.atlas_source_levels.borrow(),
                    expected,
                    "standalone raw bytes at record {index}"
                );
            }
        }
        assert_eq!(
            receipts,
            vec![
                (false, 0, 0),
                (false, 2, 1),
                (false, 5, 2),
                (false, 8, 0),
                (true, 0, 0),
                (true, 2, 1),
                (true, 5, 2),
                (true, 8, 0),
                (false, 0, 0),
                (false, 2, 1),
                (false, 5, 2),
                (false, 8, 0),
            ]
        );
    }
}

#[cfg(test)]
mod walk_grid_source_tests {
    //! #7 / #34 parity: the live walk source (dense grid + camera memo) must make the walk
    //! answer exactly what the hashed `AuthoritativeSectionSource` makes it answer.
    use super::*;
    use crate::terrain::cull::visible_sections_with_view;
    use crate::terrain::visgraph::VisGraph;
    use mc_math::Facing;

    fn splitmix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn buffers(visibility: SetVisibility, occlusion: SectionOcclusion) -> ChunkBuffers {
        ChunkBuffers {
            layers: std::array::from_fn(|_| None),
            visibility,
            occlusion,
            translucent_cpu: Buffer::new(),
        }
    }

    /// A void-map-like world: a loaded square of columns (server view distance) holding mostly
    /// all-air sections, a few islands with real connectivity and opacity, some loaded but
    /// unmeshed sections, and everything beyond the loaded square absent.
    fn void_world(
        seed: u64,
        view_distance: i32,
    ) -> (HashMap<ChunkSectionPos, ChunkBuffers>, HashSet<ChunkPos>) {
        let mut chunks = HashMap::new();
        let mut columns = HashSet::new();
        let mut island = VisGraph::new();
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..5 {
                    island.set_opaque(x, y, z);
                }
            }
        }
        let island_occlusion = island.occlusion();
        for cx in -view_distance..=view_distance {
            for cz in -view_distance..=view_distance {
                columns.insert(ChunkPos::new(cx, cz));
                for y in 0..16 {
                    let pos = ChunkSectionPos::new(cx, y, cz);
                    let r = splitmix(
                        seed ^ splitmix(
                            ((cx as u64) << 32) ^ ((cz as u32) as u64) ^ ((y as u64) << 58),
                        ),
                    );
                    match r % 40 {
                        // Loaded but not meshed yet: a DUMMY hard stop.
                        0 => {}
                        1..=3 => {
                            let mut vis = SetVisibility::all(false);
                            let group: Vec<Facing> = Facing::VALUES
                                .into_iter()
                                .filter(|f| (r >> (8 + f.index())) & 1 == 1)
                                .collect();
                            vis.set_many_visible(&group);
                            chunks.insert(pos, buffers(vis, island_occlusion));
                        }
                        4 => {
                            chunks
                                .insert(pos, buffers(SetVisibility::all(false), island_occlusion));
                        }
                        _ => {
                            chunks.insert(
                                pos,
                                buffers(SetVisibility::all(true), SectionOcclusion::default()),
                            );
                        }
                    }
                }
            }
        }
        (chunks, columns)
    }

    fn camera(pos: DVec3, yaw: f32, pitch: f32) -> Camera {
        let mut cam = Camera {
            pos,
            yaw,
            pitch,
            aspect: 16.0 / 9.0,
            far: 1024.0,
            ..Camera::default()
        };
        cam.update();
        cam
    }

    #[test]
    fn the_grid_source_walks_exactly_like_the_hashed_source() {
        let (mut chunks, mut columns) = void_world(7, 5);
        let mut grid = WalkGrid::new();
        let memo = Cell::new(None);
        let mut walk = SectionWalk::new();
        let mut generation = 0u64;
        let mut nontrivial = 0;
        for step in 0..160u64 {
            let r = splitmix(step ^ 0xabcd);
            // The world changes under the grid the way the renderer's hooks see it change.
            if step % 5 == 4 {
                let pos = ChunkSectionPos::new(
                    (r % 15) as i32 - 7,
                    ((r >> 8) % 16) as i32,
                    ((r >> 16) % 15) as i32 - 7,
                );
                if (r >> 24).is_multiple_of(2) {
                    chunks.remove(&pos);
                } else {
                    chunks.insert(
                        pos,
                        buffers(
                            SetVisibility::all((r >> 25).is_multiple_of(2)),
                            SectionOcclusion::default(),
                        ),
                    );
                }
                let resident = chunks.get(&pos).map(|chunk| chunk.visibility);
                grid.set_section(
                    pos,
                    walk_grid::encode(walk_grid::authoritative(resident, Some(&columns), pos)),
                );
            }
            if step % 11 == 10 {
                let col = ChunkPos::new((r % 17) as i32 - 8, ((r >> 8) % 17) as i32 - 8);
                let changed = if (r >> 16).is_multiple_of(2) {
                    columns.insert(col)
                } else {
                    columns.remove(&col)
                };
                if changed {
                    grid.refresh_column(col, |col, cells| {
                        fill_walk_column(&chunks, Some(&columns), col, cells)
                    });
                }
            }
            // A camera wandering over the islands, crossing column boundaries, looking around.
            let pos = DVec3::new(
                ((r >> 32) % 2400) as f64 / 10.0 - 120.0,
                40.0 + ((r >> 44) % 900) as f64 / 10.0,
                ((r >> 20) % 2400) as f64 / 10.0 - 120.0,
            );
            let cam = camera(
                pos,
                ((r >> 8) % 360) as f32,
                ((r >> 52) % 120) as f32 - 60.0,
            );
            let frustum = Frustum::from_camera(&cam);
            let rd = [4u32, 8, 12][(step % 3) as usize];
            let start = BlockPos::from_vec3(cam.pos).section_pos();
            grid.recenter(start.chunk_pos(), rd, |col, cells| {
                fill_walk_column(&chunks, Some(&columns), col, cells);
            });
            let hashed = AuthoritativeSectionSource {
                chunks: &chunks,
                world_columns: Some(&columns),
            };
            let expected =
                visible_sections_with_view(&hashed, &frustum, cam.pos, cam.look_vec(), rd, true);
            let live = GridSectionSource {
                grid: &grid,
                inner: AuthoritativeSectionSource {
                    chunks: &chunks,
                    world_columns: Some(&columns),
                },
                camera_memo: &memo,
            };
            generation += 1;
            let got = walk.run_with_view(
                &live,
                &frustum,
                cam.pos,
                Some(cam.look_vec()),
                rd,
                true,
                false,
                generation,
            );
            assert_eq!(got, expected.as_slice(), "step {step}");
            if expected.len() > 1 {
                nontrivial += 1;
            }
        }
        assert!(nontrivial > 100, "{nontrivial}");
    }

    /// #34: the memo answers like a fresh query for every cell of a partly-solid camera section,
    /// in an order that hits, misses, revisits cells and changes the section's bits under it.
    #[test]
    fn the_camera_memo_answers_like_a_fresh_query() {
        let pos = ChunkSectionPos::new(0, 4, 0);
        let mut ground = VisGraph::new();
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..6 {
                    if (x + z) % 7 != 0 {
                        ground.set_opaque(x, y, z);
                    }
                }
            }
        }
        let mut sealed = VisGraph::new();
        for i in 0..4096 {
            if i % 61 != 0 {
                sealed.set_opaque(i & 15, (i >> 8) & 15, (i >> 4) & 15);
            }
        }
        let grid = WalkGrid::new();
        let memo = Cell::new(None);
        let columns = HashSet::new();
        for occlusion in [
            ground.occlusion(),
            sealed.occlusion(),
            SectionOcclusion::default(),
            ground.occlusion(),
        ] {
            let mut chunks = HashMap::new();
            chunks.insert(pos, buffers(SetVisibility::all(true), occlusion));
            let live = GridSectionSource {
                grid: &grid,
                inner: AuthoritativeSectionSource {
                    chunks: &chunks,
                    world_columns: Some(&columns),
                },
                camera_memo: &memo,
            };
            for i in (0..4096).chain((0..4096).rev()).step_by(3) {
                let (x, y, z) = (i & 15, (i >> 8) & 15, (i >> 4) & 15);
                for _ in 0..2 {
                    assert_eq!(
                        live.camera_reach(pos, x, y, z),
                        Some(CameraReach::of(&occlusion.visible_facings_from(x, y, z))),
                        "cell ({x},{y},{z})"
                    );
                }
            }
            assert_eq!(
                live.camera_reach(ChunkSectionPos::new(1, 4, 0), 0, 0, 0),
                None,
                "not resident"
            );
        }
    }
}

/// Only page 0, and only where it can grow,
/// opens small; never bigger than the arena's own page size (the growth tests' 4 KiB pages keep
/// their sizes).
#[cfg(test)]
mod w21_mr_first_page_tests {
    use super::*;

    #[test]
    fn w21_mr_page_zero_opens_small_only_where_it_can_grow() {
        let arena = VertexArena::default();
        let first = if crate::w21_mr::small_first_page_enabled() {
            ARENA_FIRST_PAGE_BYTES
        } else {
            ARENA_PAGE_BYTES
        };
        assert_eq!(arena.new_page_bytes(true, true), first);
        assert_eq!(
            arena.new_page_bytes(true, false),
            ARENA_PAGE_BYTES,
            "no growth"
        );
        assert_eq!(
            arena.new_page_bytes(false, true),
            ARENA_PAGE_BYTES,
            "later pages"
        );
        assert_eq!(arena.new_page_bytes(false, false), ARENA_PAGE_BYTES);
        let tiny = VertexArena {
            page_bytes: 4 * 1024,
            ..VertexArena::default()
        };
        assert_eq!(tiny.new_page_bytes(true, true), 4 * 1024);
    }

    // Page 0 starts at a quarter of a full page.
    const _: () = assert!(ARENA_FIRST_PAGE_BYTES * 4 == ARENA_PAGE_BYTES);

    fn atlas() -> Atlas {
        Atlas {
            texture: [255u8, 255, 255, 255].repeat(4),
            mipmaps: Vec::new(),
            w: 2,
            h: 2,
        }
    }

    fn world(count: i32, quads: usize) -> Vec<ChunkMesh> {
        (0..count)
            .map(|x| {
                let pos = ChunkSectionPos::new(x, x % 5, -x);
                let mut mesh = ChunkMesh::empty(pos);
                let vertex = TerrainVertex::new([x as f32, 0.0, 0.0], [0.0; 2], [255; 4], [0; 4]);
                for _ in 0..quads {
                    mesh.solid.push_quad([vertex; 4]);
                }
                mesh
            })
            .collect()
    }

    fn page_sizes(renderer: &TerrainRenderer) -> Vec<u64> {
        let arena = renderer.arena.borrow();
        arena
            .pages
            .iter()
            .map(|page| page.as_ref().map_or(0, |page| page.ranges.capacity))
            .collect()
    }

    /// A dense world grows the small page 0 (4 KiB here, pages of 16 KiB); after it is unloaded
    /// (in bulk, as a world change does) the next, small world starts with a small page 0 again
    /// instead of keeping the grown one. Without growth (the direct path) nothing changes: pages
    /// open at the full size and one stays as the spare.
    #[test]
    fn w21_mr_a_world_change_after_a_dense_world_starts_page_zero_small_again() {
        let Ok(gpu) = Gpu::headless(8, 8, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let small_on = crate::w21_mr::small_first_page_enabled();
        let deadline = Instant::now() + std::time::Duration::from_secs(3600);
        for grow in [true, false] {
            let mut renderer = TerrainRenderer::new(&gpu, &atlas());
            {
                let mut arena = renderer.arena.borrow_mut();
                arena.page_bytes = 16 * 1024;
                arena.first_page_bytes = 4 * 1024;
                arena.grow_limit = 64 * 1024;
                arena.grow_enabled = grow;
            }
            // About 8 KiB of terrain: page 0 grows 4 -> 8 -> 16 KiB, not past a full page. That
            // is the case the earlier rule kept as the spare (only pages above `page_bytes`
            // counted as grown), so the next world would have started at 16 KiB.
            let dense = world(25, 3);
            let positions: HashSet<ChunkSectionPos> = dense.iter().map(|m| m.pos).collect();
            let retained = renderer.upload_all_with_deadline(&gpu, dense, DVec3::ZERO, deadline);
            assert!(retained.is_empty());
            let loaded = page_sizes(&renderer);
            if grow && small_on {
                assert!(
                    loaded[0] > 4 * 1024 && loaded[0] <= 16 * 1024,
                    "page 0 grew, within a full page: {loaded:?}"
                );
            }
            renderer.remove_sections(&positions);
            let after = page_sizes(&renderer);
            let kept: Vec<u64> = after.iter().copied().filter(|&size| size > 0).collect();
            if grow && small_on {
                assert!(
                    kept.is_empty(),
                    "the grown page 0 goes with the world: {after:?}"
                );
            } else if !grow {
                assert_eq!(kept, vec![16 * 1024], "one full-size spare page stays");
            }

            renderer.mesh_revisions.clear();
            renderer.invalid_mesh_revisions.clear();
            let small = world(2, 1);
            let retained = renderer.upload_all_with_deadline(&gpu, small, DVec3::ZERO, deadline);
            assert!(retained.is_empty());
            let next = page_sizes(&renderer);
            let first = next.iter().copied().find(|&size| size > 0).unwrap();
            if grow && small_on {
                assert_eq!(first, 4 * 1024, "the next world starts small: {next:?}");
            } else if !grow {
                assert_eq!(
                    first,
                    16 * 1024,
                    "the direct path keeps full pages: {next:?}"
                );
            }
        }
    }
}
