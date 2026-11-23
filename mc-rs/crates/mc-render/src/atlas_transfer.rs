//! Experimental, flush-local staging for independently owned RGBA8 atlas textures.
//!
//! The driver must preflight a common anaglyph setting and preserve owner call order.
//! Each owner still validates and updates its own raw shadow before calling the sink.
//! Finish the batch before returning from the flush: later queue writes must win.
//!
//! # One GPU block atlas
//!
//! Vanilla has one block atlas. mc-rs used to keep a private GPU copy per render owner, so every
//! animated sprite level was copied five times per tick (terrain, particles, block overlay,
//! first person, screen overlay). Inside an [`AtlasShareScope`] (the client's render-stack
//! install), [`share_atlas_texture`] hands every owner whose copy would be byte-identical a clone
//! of one texture handle, and [`AtlasTransferBatch::copy_to`] records one copy per record and
//! texture instead of one per owner. Owners keep their own views, samplers, bind groups and raw
//! shadows, so what they sample is unchanged. `MC_SHARED_ATLAS=0` restores private copies.

use crate::Gpu;
use mc_assets::AtlasAnimationUpload;
use std::cell::RefCell;

// Experimental safety ceilings, not observed workload maxima or tuning results.
const MAX_PACKED_BYTES: u64 = 8 * 1024 * 1024;
const MAX_RECORDS: usize = 8192;

/// Preflight failures select the entire original flush before owner mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchFallback {
    /// Empty input needs no transfer allocation.
    Empty,
    /// Dimensions, byte length, coordinates or arithmetic cannot be represented.
    Layout,
    /// Experimental byte/record ceiling or device buffer limit exceeded.
    Limit,
    /// CPU layout storage could not be reserved.
    Allocation,
    /// The newly created mapped buffer could not expose a writable range.
    Mapping,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RecordLayout {
    offset: u64,
    row_bytes: u32,
    stride: u32,
    span: u64,
    staged: bool,
}

fn record_layout(upload: &AtlasAnimationUpload, offset: u64) -> Option<RecordLayout> {
    if upload.width == 0 || upload.height == 0 || !offset.is_multiple_of(4) {
        return None;
    }
    upload.x.checked_add(upload.width)?;
    upload.y.checked_add(upload.height)?;
    let row_bytes = upload.width.checked_mul(4)?;
    let logical_bytes = u64::from(row_bytes).checked_mul(u64::from(upload.height))?;
    if usize::try_from(logical_bytes).ok()? != upload.rgba.len() {
        return None;
    }
    let stride = if upload.height == 1 {
        row_bytes
    } else {
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        row_bytes.checked_add(alignment - 1)? / alignment * alignment
    };
    let span = u64::from(stride)
        .checked_mul(u64::from(upload.height - 1))?
        .checked_add(u64::from(row_bytes))?;
    usize::try_from(offset.checked_add(span)?).ok()?;
    Some(RecordLayout {
        offset,
        row_bytes,
        stride,
        span,
        staged: false,
    })
}

fn plan(
    uploads: &[AtlasAnimationUpload],
    device_max_buffer_size: u64,
) -> Result<(Vec<RecordLayout>, u64), BatchFallback> {
    if uploads.is_empty() {
        return Err(BatchFallback::Empty);
    }
    if uploads.len() > MAX_RECORDS {
        return Err(BatchFallback::Limit);
    }
    let cap = MAX_PACKED_BYTES.min(device_max_buffer_size);
    let mut layouts = Vec::new();
    layouts
        .try_reserve_exact(uploads.len())
        .map_err(|_| BatchFallback::Allocation)?;
    let mut total = 0_u64;
    for upload in uploads {
        let layout = record_layout(upload, total).ok_or(BatchFallback::Layout)?;
        total = total
            .checked_add(layout.span)
            .ok_or(BatchFallback::Layout)?;
        if total > cap {
            return Err(BatchFallback::Limit);
        }
        layouts.push(layout);
    }
    Ok((layouts, total))
}

fn pack_record(
    mapped: &mut wgpu::WriteOnly<'_, [u8]>,
    upload: &AtlasAnimationUpload,
    layout: RecordLayout,
    anaglyph: bool,
) {
    // Layout planning checked these conversions and every referenced byte in advance.
    let offset = layout.offset as usize;
    let row_bytes = layout.row_bytes as usize;
    let stride = layout.stride as usize;
    for (row, source) in upload.rgba.chunks_exact(row_bytes).enumerate() {
        let start = offset + row * stride;
        let mut destination = mapped.slice(start..start + row_bytes);
        if anaglyph {
            // wgpu 30 mappings are write-only. Transform stack pixels directly into
            // the mapped destination, never read/modify mapped memory or copy padding.
            let (pixels, remainder) = destination.into_chunks::<4>();
            debug_assert!(remainder.is_empty());
            pixels.write_iter(source.chunks_exact(4).map(|pixel| {
                mc_assets::sprite::anaglyph_color([pixel[0], pixel[1], pixel[2], pixel[3]])
            }));
        } else {
            destination.copy_from_slice(source);
        }
    }
}

/// ATLAS-5 / PRESENT-10: persistent staging memory for [`AtlasTransferBatch::new_with_staging`].
///
/// A `wgpu::util::StagingBelt` whose chunks are re-mapped once the GPU has consumed them, so a
/// steady-state flush neither creates nor zero-fills a buffer. One chunk holds a vanilla 16×
/// tick-frame's packed span (about 55 KB at 4 mip levels, ATLAS-1); a larger flush (catch-up
/// before coalescing, 64× packs) gets a chunk sized to fit, which is then recycled too.
pub struct AtlasStaging {
    belt: wgpu::util::StagingBelt,
}

impl AtlasStaging {
    /// Default chunk size; see the type docs.
    pub const CHUNK_BYTES: u64 = 128 * 1024;

    /// A belt on `gpu`'s device. No GPU memory is allocated until the first flush.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            belt: wgpu::util::StagingBelt::new(gpu.device.clone(), Self::CHUNK_BYTES),
        }
    }
}

impl std::fmt::Debug for AtlasStaging {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.belt, f)
    }
}

/// Environment switch: `0` makes [`AtlasShareScope::begin`] inactive, so every owner creates and
/// uploads its own copy of the block atlas again (A/B and rollback).
pub const SHARED_ATLAS_ENV: &str = "MC_SHARED_ATLAS";

/// The texture shape every full-mip block-atlas owner creates: plain RGBA8 (1.8.9 does no colour
/// space conversion), sampled and written by uploads only.
const SHARED_ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SHARED_ATLAS_USAGE: wgpu::TextureUsages =
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST);

thread_local! {
    /// The active [`AtlasShareScope`] on this thread, if any.
    static ATLAS_SHARE: RefCell<Option<ShareState>> = const { RefCell::new(None) };
}

struct ShareState {
    width: u32,
    height: u32,
    /// Raw (untransformed) base level and mip levels of the scope's atlas.
    levels: Vec<Vec<u8>>,
    anaglyph: bool,
    /// The first matching owner's texture, on the device it was created on.
    shared: Option<(wgpu::Device, wgpu::Texture)>,
    adopted: u32,
    /// The raw shadow every owner of `shared` keeps (see
    /// [`share_atlas_shadow`]); made from the first such owner's levels.
    shadow: Option<std::rc::Rc<RefCell<Vec<Vec<u8>>>>>,
}

impl ShareState {
    fn matches(
        &self,
        descriptor: &wgpu::TextureDescriptor<'_>,
        levels: &[Vec<u8>],
        anaglyph: bool,
    ) -> bool {
        descriptor.size
            == (wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            })
            && descriptor.mip_level_count as usize == self.levels.len()
            && descriptor.sample_count == 1
            && descriptor.dimension == wgpu::TextureDimension::D2
            && descriptor.format == SHARED_ATLAS_FORMAT
            && descriptor.usage == SHARED_ATLAS_USAGE
            && descriptor.view_formats.is_empty()
            && anaglyph == self.anaglyph
            // Exact bytes, every level: only an owner about to upload this very atlas may share.
            && levels == self.levels.as_slice()
    }
}

/// Install-time sharing of the stitched block atlas between render owners (X14).
///
/// While a scope is alive on this thread, [`share_atlas_texture`] gives the first owner that asks
/// for a texture holding exactly this atlas (same size, level count, raw bytes and anaglyph
/// setting, plain RGBA8 sampled/copy-destination usage) a fresh texture as before, and every later
/// such owner a clone of that same handle. The client wraps its render-stack install in one scope,
/// so terrain, particles, block overlay, first person and the screen overlay end up sampling one
/// texture, and an animation flush copies each dirty rectangle once instead of five times.
///
/// Why sharing is exact: every owner uploads the same raw levels through the same transform
/// (`mc_assets::sprite::anaglyph_rgba`), applies every animation record, and replays the same
/// shadow on an anaglyph toggle; the client changes the setting on all owners together. Each
/// owner keeps its own views and samplers, so sampled texels are unchanged. An owner that later
/// rebuilds its texture outside a scope (the particle and screen-overlay anaglyph replays) simply
/// gets a private texture again.
///
/// Dropping the scope ends sharing for later creations; textures already shared stay shared.
/// A nested `begin` replaces the outer scope's state, so creations after it no longer share
/// with the outer scope's texture (safe, just no saving).
#[must_use = "atlas textures are shared only while the scope is alive"]
pub struct AtlasShareScope {
    active: bool,
    /// Thread-local state: the guard must end on the thread that began it.
    _thread_bound: std::marker::PhantomData<*const ()>,
}

impl AtlasShareScope {
    /// Begin sharing `atlas` (raw bytes, uploaded with `anaglyph`) on this thread, unless
    /// [`SHARED_ATLAS_ENV`] is `0`.
    pub fn begin(atlas: &mc_assets::Atlas, anaglyph: bool) -> Self {
        let enabled = std::env::var(SHARED_ATLAS_ENV).map_or(true, |value| value.trim() != "0");
        Self::begin_if(enabled, atlas, anaglyph)
    }

    /// [`Self::begin`] with an explicit switch instead of the environment (tests, A/B).
    pub fn begin_if(enabled: bool, atlas: &mc_assets::Atlas, anaglyph: bool) -> Self {
        let active = enabled && atlas.w > 0 && atlas.h > 0;
        if active {
            let mut levels = Vec::with_capacity(1 + atlas.mipmaps.len());
            levels.push(atlas.texture.clone());
            levels.extend(atlas.mipmaps.iter().cloned());
            ATLAS_SHARE.with(|share| {
                *share.borrow_mut() = Some(ShareState {
                    width: atlas.w,
                    height: atlas.h,
                    levels,
                    anaglyph,
                    shared: None,
                    adopted: 0,
                    shadow: None,
                });
            });
        }
        Self {
            active,
            _thread_bound: std::marker::PhantomData,
        }
    }

    /// Whether this scope shares at all (`false` when disabled or for an empty atlas).
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// How many owners adopted the first owner's texture so far.
    #[must_use]
    pub fn adopted(&self) -> u32 {
        if !self.active {
            return 0;
        }
        ATLAS_SHARE.with(|share| share.borrow().as_ref().map_or(0, |state| state.adopted))
    }
}

impl Drop for AtlasShareScope {
    fn drop(&mut self) {
        if self.active {
            ATLAS_SHARE.with(|share| share.borrow_mut().take());
        }
    }
}

impl std::fmt::Debug for AtlasShareScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtlasShareScope")
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}

/// Create a full-mip block-atlas owner's texture, or adopt the shared one (X14).
///
/// `levels` are the raw source levels the caller will upload into the returned texture, every
/// level through `mc_assets::sprite::anaglyph_rgba(level, anaglyph)`. Outside an
/// [`AtlasShareScope`], or when the request differs from the scope's atlas in any way, this is
/// exactly `gpu.device.create_texture(descriptor)`. An adopting owner's own uploads rewrite the
/// same bytes into the shared texture (idempotent; install time only).
pub fn share_atlas_texture(
    gpu: &Gpu,
    descriptor: &wgpu::TextureDescriptor<'_>,
    levels: &[Vec<u8>],
    anaglyph: bool,
) -> wgpu::Texture {
    let shared = ATLAS_SHARE.with(|share| {
        let mut share = share.borrow_mut();
        let state = share.as_mut()?;
        if !state.matches(descriptor, levels, anaglyph) {
            return None;
        }
        match &state.shared {
            Some((device, texture)) => {
                if *device != gpu.device {
                    return None;
                }
                let texture = texture.clone();
                state.adopted += 1;
                Some(texture)
            }
            None => {
                let texture = gpu.device.create_texture(descriptor);
                state.shared = Some((gpu.device.clone(), texture.clone()));
                Some(texture)
            }
        }
    });
    shared.unwrap_or_else(|| gpu.device.create_texture(descriptor))
}

/// The raw shadow (every level's untransformed bytes, kept for the
/// anaglyph replay and written by every animation record) for an owner whose texture is
/// `texture` and whose raw levels are `levels`.
///
/// Inside an [`AtlasShareScope`], every owner holding the scope's shared texture gets one
/// handle to the same shadow, made from the first such owner's levels: their levels are
/// byte-identical by construction ([`share_atlas_texture`] shares only an exact match), every
/// owner applies every animation record to its shadow in the same flush (rewriting the same
/// bytes), and the client toggles anaglyph on all of them together. Before, each kept a private
/// copy of the whole atlas mip chain (five in all). Anywhere else, or with
/// `MC_W21_MR_SHARED_SHADOW=0`, the owner keeps its own levels as before.
pub fn share_atlas_shadow(
    texture: &wgpu::Texture,
    levels: Vec<Vec<u8>>,
) -> std::rc::Rc<RefCell<Vec<Vec<u8>>>> {
    if !crate::w21_mr::shared_shadow_enabled() {
        return std::rc::Rc::new(RefCell::new(levels));
    }
    ATLAS_SHARE.with(|share| {
        let mut share = share.borrow_mut();
        let Some(state) = share.as_mut().filter(|state| {
            state
                .shared
                .as_ref()
                .is_some_and(|(_, shared)| shared == texture)
        }) else {
            return std::rc::Rc::new(RefCell::new(levels));
        };
        match &state.shadow {
            Some(shadow) if *shadow.borrow() == levels => std::rc::Rc::clone(shadow),
            // Never true while the shadow only follows the scope's own installs (same atlas);
            // a mismatch keeps the owner's levels private rather than sharing wrong bytes.
            Some(_) => std::rc::Rc::new(RefCell::new(levels)),
            None => {
                let shadow = std::rc::Rc::new(RefCell::new(levels));
                state.shadow = Some(std::rc::Rc::clone(&shadow));
                shadow
            }
        }
    })
}

/// Mechanism counts, separate from each owner's acceptance/logical-byte accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchStats {
    /// One fresh mapped staging buffer for a successful nonempty preflight.
    pub buffer_creations: u64,
    /// Full allocated span, including alignment gaps and unaccepted reserved records.
    pub packed_span_bytes: u64,
    /// Logical bytes packed once for each record accepted by at least one owner.
    pub staged_payload_bytes: u64,
    /// Copy commands, one per accepting destination texture and record.
    pub copy_commands: u64,
    /// Owner calls that named a texture this record had just been copied to (owners sharing one
    /// GPU atlas, see [`share_atlas_texture`]); they record nothing.
    pub deduplicated_copies: u64,
    /// Zero when no owner accepted, otherwise one flush-local submission.
    pub submissions: u64,
}

/// One flush's mapped buffer, ordered copies and original borrowed upload records.
///
/// Call `finish` before any subsequent writes or rendering. No buffer is reused across
/// flushes. Standalone phase upload APIs retain their original queue-write behavior.
#[must_use = "finish the batch before returning from the atlas flush"]
pub struct AtlasTransferBatch<'a> {
    uploads: &'a [AtlasAnimationUpload],
    layouts: Vec<RecordLayout>,
    anaglyph: bool,
    buffer: wgpu::Buffer,
    /// Where this flush's span starts in `buffer`: zero for a fresh buffer, the belt slice's
    /// offset for [`Self::new_with_staging`].
    base: u64,
    /// The persistent belt this flush borrowed its span from, if any.
    staging: Option<&'a mut AtlasStaging>,
    mapped: wgpu::BufferViewMut,
    encoder: wgpu::CommandEncoder,
    stats: BatchStats,
    /// The record the latest `copy_to` calls named, and the textures it was copied to since that
    /// record began (X14 dedupe).
    recorded_index: Option<usize>,
    recorded_textures: Vec<wgpu::Texture>,
}

impl<'a> AtlasTransferBatch<'a> {
    /// Preflight every record and allocate/map before mutating any owner.
    ///
    /// A returned error permits the entire original flush. GPU allocation/device errors
    /// follow wgpu's existing error policy; they are not caught and retried after mutation.
    pub fn new(
        gpu: &Gpu,
        uploads: &'a [AtlasAnimationUpload],
        anaglyph: bool,
    ) -> Result<Self, BatchFallback> {
        let (layouts, total) = plan(uploads, gpu.device.limits().max_buffer_size)?;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("experimental atlas transfer staging"),
            size: total,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let mut mapped = match buffer.get_mapped_range_mut(..) {
            Ok(mapped) => mapped,
            Err(_) => {
                buffer.unmap();
                return Err(BatchFallback::Mapping);
            }
        };
        // Explicit initialization includes gaps and spans rejected by every owner.
        mapped.slice(..).fill(0);
        let encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("experimental atlas transfer copies"),
            });
        Ok(Self {
            uploads,
            layouts,
            anaglyph,
            buffer,
            base: 0,
            staging: None,
            mapped,
            encoder,
            stats: BatchStats {
                buffer_creations: 1,
                packed_span_bytes: total,
                ..BatchStats::default()
            },
            recorded_index: None,
            recorded_textures: Vec::new(),
        })
    }

    /// [`Self::new`], staging into a span of the persistent `staging` belt instead of a fresh
    /// mapped buffer (ATLAS-5). The span is not zero-filled: copies read only each record's
    /// `row_bytes` per row, never the alignment gaps or spans no owner accepted, so stale bytes
    /// there are unobservable. `buffer_creations` stays zero; the belt allocates a chunk only
    /// when none of its recycled chunks has room.
    ///
    /// The span starts at a 512-byte boundary (D3D12's placement alignment), so each record
    /// keeps the same offset alignment it had at the start of a fresh buffer.
    pub fn new_with_staging(
        gpu: &Gpu,
        staging: &'a mut AtlasStaging,
        uploads: &'a [AtlasAnimationUpload],
        anaglyph: bool,
    ) -> Result<Self, BatchFallback> {
        let (layouts, total) = plan(uploads, gpu.device.limits().max_buffer_size)?;
        // Every record span is a multiple of 4 bytes, so `total` meets the belt's alignment.
        let size = wgpu::BufferSize::new(total).ok_or(BatchFallback::Layout)?;
        let alignment = wgpu::BufferSize::new(512).expect("nonzero");
        let slice = staging.belt.allocate(size, alignment);
        let (buffer, base) = (slice.buffer().clone(), slice.offset());
        let mapped = slice
            .get_mapped_range_mut()
            .map_err(|_| BatchFallback::Mapping)?;
        let encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("atlas transfer copies"),
            });
        Ok(Self {
            uploads,
            layouts,
            anaglyph,
            buffer,
            base,
            staging: Some(staging),
            mapped,
            encoder,
            stats: BatchStats {
                buffer_creations: 0,
                packed_span_bytes: total,
                ..BatchStats::default()
            },
            recorded_index: None,
            recorded_textures: Vec::new(),
        })
    }

    /// Record an already validated owner's exact texture after its raw shadow update.
    ///
    /// `index` must name the same record given to that owner. Owner state must match
    /// the whole-flush preflight; a mismatch is a caller bug, never a partial retry.
    ///
    /// X14: a call naming a texture this record was already copied to since the caller moved on
    /// to this record (owners sharing one GPU atlas, [`share_atlas_texture`]) records nothing.
    /// That skip is exact: since that copy, only copies of this same record (to other textures)
    /// were recorded, so the repeat would write the same bytes over the same texels. A record
    /// named again after another record (`0, 1, 0`) is copied again, because the record between
    /// may overlap it.
    pub fn copy_to(&mut self, index: usize, texture: &wgpu::Texture, anaglyph: bool) {
        assert_eq!(anaglyph, self.anaglyph, "atlas batch owner state changed");
        if self.recorded_index == Some(index) {
            if self.recorded_textures.contains(texture) {
                self.stats.deduplicated_copies += 1;
                return;
            }
        } else {
            self.recorded_index = Some(index);
            self.recorded_textures.clear();
        }
        self.recorded_textures.push(texture.clone());
        let upload = &self.uploads[index];
        let layout = &mut self.layouts[index];
        if !layout.staged {
            pack_record(&mut self.mapped.slice(..), upload, *layout, anaglyph);
            layout.staged = true;
            self.stats.staged_payload_bytes += upload.rgba.len() as u64;
        }
        self.encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: self.base + layout.offset,
                    bytes_per_row: (upload.height > 1).then_some(layout.stride),
                    rows_per_image: None,
                },
            },
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
            wgpu::Extent3d {
                width: upload.width,
                height: upload.height,
                depth_or_array_layers: 1,
            },
        );
        self.stats.copy_commands += 1;
    }

    /// Unmap and submit immediately, before the caller queues later writes.
    ///
    /// Existing pending replay/entity/UI writes run before these commands. Later
    /// queue writes run after them, preserving last-writer precedence per texture.
    pub fn finish(self, gpu: &Gpu) -> BatchStats {
        let Self {
            buffer,
            mapped,
            encoder,
            mut stats,
            mut staging,
            ..
        } = self;
        drop(mapped);
        match staging.as_mut() {
            // The belt unmaps its own chunks; the buffer handle is a chunk, not ours to unmap.
            Some(staging) => staging.belt.finish(),
            None => buffer.unmap(),
        }
        if stats.copy_commands != 0 {
            let _ = gpu.queue.submit([encoder.finish()]);
            stats.submissions = 1;
        }
        if let Some(staging) = staging {
            // Re-map closed chunks once the GPU is done with them; never blocks.
            staging.belt.recall();
        }
        // Dropping this handle is safe: submitted commands retain their resources.
        // Explicit destroy() here would violate the submission lifetime contract.
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upload(width: u32, height: u32) -> AtlasAnimationUpload {
        AtlasAnimationUpload {
            mip_level: 0,
            x: 0,
            y: 0,
            width,
            height,
            rgba: (0..u64::from(width) * u64::from(height) * 4)
                .map(|index| (index.wrapping_mul(73).wrapping_add(31) % 256) as u8)
                .collect(),
        }
    }

    #[test]
    fn layout_boundary_cases_use_minimum_last_row_span() {
        for (width, height, stride, span) in [
            (1, 1, 4, 4),
            (3, 1, 12, 12),
            (2, 2, 256, 264),
            (63, 2, 256, 508),
            (64, 2, 256, 512),
            (65, 2, 512, 772),
            (3, 5, 256, 1036),
        ] {
            let actual = record_layout(&upload(width, height), 4).unwrap();
            assert_eq!(
                (actual.offset, actual.stride, actual.span),
                (4, stride, span)
            );
        }
    }

    #[test]
    fn malformed_overflow_and_caps_select_preflight_fallback() {
        let mut bad = upload(1, 1);
        bad.width = u32::MAX;
        assert_eq!(record_layout(&bad, 0), None);
        bad = upload(1, 1);
        bad.x = u32::MAX;
        assert_eq!(record_layout(&bad, 0), None);
        bad = upload(1, 1);
        bad.rgba.pop();
        assert_eq!(record_layout(&bad, 0), None);
        assert_eq!(record_layout(&upload(1, 1), u64::MAX - 3), None);
        assert_eq!(record_layout(&upload(1, 1), 2), None);
        assert_eq!(record_layout(&upload(0, 1), 0), None);
        assert_eq!(record_layout(&upload(1, 0), 0), None);
        assert_eq!(plan(&[], u64::MAX), Err(BatchFallback::Empty));
        assert_eq!(plan(&[upload(2, 2)], 263), Err(BatchFallback::Limit));
        assert_eq!(
            plan(&vec![upload(1, 1); MAX_RECORDS + 1], u64::MAX),
            Err(BatchFallback::Limit)
        );
        assert_eq!(
            plan(&[upload(1, 32770)], u64::MAX),
            Err(BatchFallback::Limit)
        );
    }

    #[test]
    fn packed_pixels_match_legacy_transform_without_touching_padding_or_raw() {
        let records = [upload(3, 5), upload(1, 1), upload(65, 2)];
        let original = records.clone();
        let (layouts, total) = plan(&records, u64::MAX).unwrap();
        for enabled in [false, true] {
            let mut mapped = vec![0_u8; total as usize];
            for (record, layout) in records.iter().zip(&layouts) {
                pack_record(
                    &mut wgpu::WriteOnly::from(mapped.as_mut_slice()),
                    record,
                    *layout,
                    enabled,
                );
            }
            for (record, layout) in records.iter().zip(&layouts) {
                let legacy = mc_assets::sprite::anaglyph_rgba(&record.rgba, enabled);
                for (row, expected) in legacy.chunks_exact(layout.row_bytes as usize).enumerate() {
                    let start = layout.offset as usize + row * layout.stride as usize;
                    let end = start + layout.row_bytes as usize;
                    assert_eq!(&mapped[start..end], expected);
                    if row + 1 < record.height as usize {
                        assert!(mapped[end..start + layout.stride as usize]
                            .iter()
                            .all(|byte| *byte == 0));
                    }
                }
            }
            assert_eq!(records, original);
        }
    }
}

#[cfg(test)]
mod gpu_oracle_tests {
    // Test-only sibling of atlas_transfer's production implementation.
    // Synthetic destinations exercise transfer mechanics, not real owner acceptance.
    use super::*;

    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 16;
    const MIPS: u32 = 5;

    fn extent(mip: u32) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: (WIDTH >> mip).max(1),
            height: (HEIGHT >> mip).max(1),
            depth_or_array_layers: 1,
        }
    }

    fn texture(gpu: &Gpu) -> wgpu::Texture {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas transfer synthetic GPU oracle"),
            size: extent(0),
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn record(mip: u32, x: u32, y: u32, width: u32, height: u32, seed: u8) -> AtlasAnimationUpload {
        AtlasAnimationUpload {
            mip_level: mip,
            x,
            y,
            width,
            height,
            rgba: (0..width * height * 4)
                .map(|i| seed.wrapping_add((i as u8).wrapping_mul(73)))
                .collect(),
        }
    }

    fn write(gpu: &Gpu, texture: &wgpu::Texture, upload: &AtlasAnimationUpload, enabled: bool) {
        let pixels = mc_assets::sprite::anaglyph_rgba(&upload.rgba, enabled);
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
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload.width * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: upload.width,
                height: upload.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn apply(pixels: &mut [u8], upload: &AtlasAnimationUpload, enabled: bool) {
        let source = mc_assets::sprite::anaglyph_rgba(&upload.rgba, enabled);
        let pitch = extent(upload.mip_level).width as usize * 4;
        let row_bytes = upload.width as usize * 4;
        for (row, bytes) in source.chunks_exact(row_bytes).enumerate() {
            let start = (upload.y as usize + row) * pitch + upload.x as usize * 4;
            pixels[start..start + row_bytes].copy_from_slice(bytes);
        }
    }

    fn read(gpu: &Gpu, texture: &wgpu::Texture, mip: u32) -> Vec<u8> {
        let size = extent(mip);
        let row_bytes = size.width * 4;
        let stride = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atlas oracle readback"),
            size: u64::from(stride) * u64::from(size.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        gpu.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).expect("GPU map result receiver");
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU readback poll");
        rx.recv()
            .expect("GPU map callback")
            .expect("GPU readback map");
        let mapped = buffer.get_mapped_range(..).expect("GPU readback view");
        let mut pixels = Vec::with_capacity((row_bytes * size.height) as usize);
        for row in 0..size.height as usize {
            let start = row * stride as usize;
            pixels.extend_from_slice(&mapped[start..start + row_bytes as usize]);
        }
        drop(mapped);
        buffer.unmap();
        pixels
    }

    #[test]
    fn atlas_transfer_gpu_records_mips_overlap_later_acceptance_and_toggle_cycle() {
        let gpu = Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let legacy: Vec<_> = (0..5).map(|_| texture(&gpu)).collect();
        let batched: Vec<_> = (0..5).map(|_| texture(&gpu)).collect();
        let records = vec![
            record(0, 0, 0, 1, 1, 1),
            record(0, 2, 1, 3, 1, 7),
            record(0, 3, 2, 3, 5, 19),
            record(0, 0, 8, 63, 2, 29),
            record(0, 32, 8, 64, 2, 37),
            record(0, 5, 8, 65, 2, 43),
            record(0, 5, 8, 65, 2, 47),
            record(1, 3, 1, 3, 5, 53),
            record(4, 2, 0, 1, 1, 59),
            record(2, 1, 1, 3, 1, 61),
        ];
        let raw_records = records.clone();
        // Index 8 is first accepted by the fourth synthetic destination; index 9 by none.
        let accepts = |owner: usize, index: usize| {
            if index == 9 {
                false
            } else if index == 8 {
                owner >= 3
            } else {
                owner != 1 || index.is_multiple_of(2)
            }
        };
        for enabled in [false, true, false] {
            let mut expected = Vec::new();
            for owner in 0..5 {
                let mut levels = Vec::new();
                for mip in 0..MIPS {
                    let size = extent(mip);
                    let initial = record(
                        mip,
                        0,
                        0,
                        size.width,
                        size.height,
                        (owner * 31 + mip as usize * 7) as u8,
                    );
                    write(&gpu, &legacy[owner], &initial, enabled);
                    write(&gpu, &batched[owner], &initial, enabled);
                    levels.push(mc_assets::sprite::anaglyph_rgba(&initial.rgba, enabled));
                }
                expected.push(levels);
            }
            let mut batch =
                AtlasTransferBatch::new(&gpu, &records, enabled).expect("valid transfer plan");
            let mut copy_count = 0;
            for (index, upload) in records.iter().enumerate() {
                for owner in 0..5 {
                    if accepts(owner, index) {
                        write(&gpu, &legacy[owner], upload, enabled);
                        batch.copy_to(index, &batched[owner], enabled);
                        apply(
                            &mut expected[owner][upload.mip_level as usize],
                            upload,
                            enabled,
                        );
                        copy_count += 1;
                    }
                }
            }
            let stats = batch.finish(&gpu);
            assert_eq!(stats.buffer_creations, 1);
            assert_eq!(stats.copy_commands, copy_count);
            assert_eq!(stats.submissions, 1);
            assert_eq!(
                stats.staged_payload_bytes,
                records[..9]
                    .iter()
                    .map(|r| r.rgba.len() as u64)
                    .sum::<u64>()
            );
            assert!(stats.packed_span_bytes > stats.staged_payload_bytes);
            for owner in 0..5 {
                for mip in 0..MIPS {
                    let baseline = read(&gpu, &legacy[owner], mip);
                    let candidate = read(&gpu, &batched[owner], mip);
                    assert_eq!(
                        baseline, expected[owner][mip as usize],
                        "legacy owner={owner} mip={mip} enabled={enabled}"
                    );
                    assert_eq!(
                        candidate, baseline,
                        "batch owner={owner} mip={mip} enabled={enabled}"
                    );
                }
            }
            assert_eq!(
                records, raw_records,
                "packing must not modify raw upload pixels"
            );
        }
        let rejected = [record(0, 0, 0, 3, 5, 91)];
        let before = read(&gpu, &batched[0], 0);
        let batch =
            AtlasTransferBatch::new(&gpu, &rejected, false).expect("rejected-record preflight");
        let stats = batch.finish(&gpu);
        assert_eq!(stats.buffer_creations, 1);
        assert!(stats.packed_span_bytes > 0);
        assert_eq!(
            (
                stats.copy_commands,
                stats.staged_payload_bytes,
                stats.submissions
            ),
            (0, 0, 0)
        );
        assert_eq!(read(&gpu, &batched[0], 0), before);
        assert!(
            pollster::block_on(errors.pop()).is_none(),
            "unexpected wgpu validation error"
        );
    }

    // Deliberately broken test-only finish: release the mapping, retain the command buffer,
    // and delay submission across a later queue.write_texture. Never a production API.
    fn finish_deferred(batch: AtlasTransferBatch<'_>) -> (wgpu::Buffer, wgpu::CommandBuffer) {
        let AtlasTransferBatch {
            buffer,
            mapped,
            encoder,
            ..
        } = batch;
        drop(mapped);
        buffer.unmap();
        (buffer, encoder.finish())
    }

    #[test]
    fn atlas_transfer_gpu_immediate_submit_beats_deferred_negative_control() {
        let gpu = Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        // Include both full-atlas toggle replays and a partial later write.
        for (animation_enabled, later_enabled, full_replay) in [
            (false, false, false),
            (false, true, true),
            (true, false, true),
        ] {
            let legacy = texture(&gpu);
            let immediate = texture(&gpu);
            let deferred = texture(&gpu);
            let initial = record(0, 0, 0, WIDTH, HEIGHT, 13);
            let animation = record(0, 2, 2, 7, 5, 83);
            let later = if full_replay {
                record(0, 0, 0, WIDTH, HEIGHT, 157)
            } else {
                record(0, 4, 3, 3, 2, 157)
            };
            for destination in [&legacy, &immediate, &deferred] {
                write(&gpu, destination, &initial, animation_enabled);
            }
            let uploads = [animation.clone()];
            write(&gpu, &legacy, &animation, animation_enabled);
            let mut batch = AtlasTransferBatch::new(&gpu, &uploads, animation_enabled)
                .expect("immediate preflight");
            batch.copy_to(0, &immediate, animation_enabled);
            assert_eq!(batch.finish(&gpu).submissions, 1);
            let mut broken = AtlasTransferBatch::new(&gpu, &uploads, animation_enabled)
                .expect("deferred preflight");
            broken.copy_to(0, &deferred, animation_enabled);
            let (retained_buffer, commands) = finish_deferred(broken);
            for destination in [&legacy, &immediate, &deferred] {
                write(&gpu, destination, &later, later_enabled);
            }
            // wgpu flushes pending queue writes before explicit submitted commands.
            gpu.queue.submit([commands]);
            let mut expected = mc_assets::sprite::anaglyph_rgba(&initial.rgba, animation_enabled);
            apply(&mut expected, &animation, animation_enabled);
            apply(&mut expected, &later, later_enabled);
            let mut wrong_order = expected.clone();
            apply(&mut wrong_order, &animation, animation_enabled);
            assert_ne!(
                expected, wrong_order,
                "negative control must discriminate the two orders"
            );
            assert_eq!(read(&gpu, &legacy, 0), expected);
            assert_eq!(read(&gpu, &immediate, 0), expected);
            let observed_wrong_order = read(&gpu, &deferred, 0);
            assert_ne!(
                observed_wrong_order, expected,
                "deferred negative control must fail the positive oracle"
            );
            assert_eq!(
                observed_wrong_order, wrong_order,
                "negative control must specifically expose animation overwriting later writes"
            );
            drop(retained_buffer);
        }
        assert!(
            pollster::block_on(errors.pop()).is_none(),
            "unexpected wgpu validation error"
        );
    }

    #[test]
    fn atlas_transfer_gpu_completed_flushes_retain_staging_and_destination_identity() {
        let gpu = Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let legacy = texture(&gpu);
        let original = texture(&gpu);
        let initial = record(0, 0, 0, WIDTH, HEIGHT, 71);
        write(&gpu, &legacy, &initial, false);
        write(&gpu, &original, &initial, false);
        let mut expected = initial.rgba.clone();
        // Each production finish drops its staging handle before the next allocation.
        // No explicit wait or readback occurs between these overlapping flushes.
        for seed in [17, 97, 213] {
            let records = [record(0, 3, 2, 65, 3, seed)];
            write(&gpu, &legacy, &records[0], false);
            let mut batch =
                AtlasTransferBatch::new(&gpu, &records, false).expect("successive preflight");
            batch.copy_to(0, &original, false);
            assert_eq!(batch.finish(&gpu).submissions, 1);
            apply(&mut expected, &records[0], false);
        }
        let replacement = texture(&gpu);
        let replacement_pixels = record(0, 0, 0, WIDTH, HEIGHT, 231);
        write(&gpu, &replacement, &replacement_pixels, true);
        assert_eq!(read(&gpu, &original, 0), expected);
        assert_eq!(read(&gpu, &legacy, 0), expected);
        assert_eq!(
            read(&gpu, &replacement, 0),
            mc_assets::sprite::anaglyph_rgba(&replacement_pixels.rgba, true)
        );
        assert!(
            pollster::block_on(errors.pop()).is_none(),
            "unexpected wgpu validation error"
        );
    }

    /// ATLAS-5 (U2): the persistent belt path produces exactly the fresh-buffer path's texels across
    /// several flushes that reuse its chunks, at every mip level and through an anaglyph toggle cycle,
    /// with no buffer creation and the same copy, payload and span accounting.
    #[test]
    fn atlas_transfer_gpu_retained_staging_matches_fresh_buffer_across_flushes() {
        let gpu = Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fresh = texture(&gpu);
        let retained = texture(&gpu);
        let mut staging = AtlasStaging::new(&gpu);
        for enabled in [false, true, false] {
            let mut expected = Vec::new();
            for mip in 0..MIPS {
                let size = extent(mip);
                let initial = record(
                    mip,
                    0,
                    0,
                    size.width,
                    size.height,
                    (mip as u8).wrapping_mul(29),
                );
                write(&gpu, &fresh, &initial, enabled);
                write(&gpu, &retained, &initial, enabled);
                expected.push(mc_assets::sprite::anaglyph_rgba(&initial.rgba, enabled));
            }
            for seed in [11_u8, 97, 213, 5] {
                let records = vec![
                    record(0, 3, 2, 65, 3, seed),
                    record(1, 1, 1, 3, 5, seed ^ 0x55),
                    record(4, 2, 0, 1, 1, seed.wrapping_add(7)),
                    record(0, 0, 8, 63, 2, seed.wrapping_add(3)),
                    record(2, 0, 0, 32, 4, seed.wrapping_add(1)),
                ];
                let mut baseline =
                    AtlasTransferBatch::new(&gpu, &records, enabled).expect("fresh preflight");
                let mut candidate =
                    AtlasTransferBatch::new_with_staging(&gpu, &mut staging, &records, enabled)
                        .expect("retained preflight");
                for index in 0..records.len() {
                    baseline.copy_to(index, &fresh, enabled);
                    candidate.copy_to(index, &retained, enabled);
                }
                let fresh_stats = baseline.finish(&gpu);
                let retained_stats = candidate.finish(&gpu);
                assert_eq!(retained_stats.buffer_creations, 0);
                assert_eq!(
                    (
                        retained_stats.copy_commands,
                        retained_stats.submissions,
                        retained_stats.staged_payload_bytes,
                        retained_stats.packed_span_bytes
                    ),
                    (
                        fresh_stats.copy_commands,
                        fresh_stats.submissions,
                        fresh_stats.staged_payload_bytes,
                        fresh_stats.packed_span_bytes
                    ),
                );
                for upload in &records {
                    apply(&mut expected[upload.mip_level as usize], upload, enabled);
                }
            }
            for mip in 0..MIPS {
                let baseline = read(&gpu, &fresh, mip);
                assert_eq!(
                    baseline, expected[mip as usize],
                    "fresh mip={mip} enabled={enabled}"
                );
                assert_eq!(
                    read(&gpu, &retained, mip),
                    baseline,
                    "retained mip={mip} enabled={enabled}"
                );
            }
        }
        assert!(
            pollster::block_on(errors.pop()).is_none(),
            "unexpected wgpu validation error"
        );
    }

    /// X14: owners sharing one texture name it in consecutive `copy_to` calls for a record. Only the
    /// first records a copy; a record named again after another record is copied again (records 0
    /// and 1 overlap, so skipping that repeat would leave record 1's texels: the negative control).
    /// Every destination ends byte-identical to plain writes for every call, at every mip.
    #[test]
    fn atlas_transfer_gpu_dedupes_repeated_textures_within_one_record_only() {
        let gpu = Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shared = texture(&gpu);
        let private = texture(&gpu);
        let legacy_shared = texture(&gpu);
        let legacy_private = texture(&gpu);
        let mut expected_shared = Vec::new();
        let mut expected_private = Vec::new();
        for mip in 0..MIPS {
            let size = extent(mip);
            let initial = record(
                mip,
                0,
                0,
                size.width,
                size.height,
                (mip as u8).wrapping_mul(41),
            );
            for destination in [&shared, &private, &legacy_shared, &legacy_private] {
                write(&gpu, destination, &initial, false);
            }
            expected_shared.push(initial.rgba.clone());
            expected_private.push(initial.rgba.clone());
        }
        let records = vec![
            record(0, 2, 1, 9, 4, 11),
            record(0, 5, 2, 9, 4, 97),
            record(1, 1, 1, 3, 3, 53),
        ];
        // (record, destination is the shared texture)
        let calls = [
            (0, true),
            (0, true),
            (0, false),
            (0, true),
            (1, true),
            (1, false),
            (1, true),
            (0, true),
            (2, true),
            (2, true),
        ];
        let mut batch =
            AtlasTransferBatch::new(&gpu, &records, false).expect("valid transfer plan");
        for (index, to_shared) in calls {
            let upload = &records[index];
            let (destination, legacy, expected) = if to_shared {
                (&shared, &legacy_shared, &mut expected_shared)
            } else {
                (&private, &legacy_private, &mut expected_private)
            };
            batch.copy_to(index, destination, false);
            write(&gpu, legacy, upload, false);
            apply(&mut expected[upload.mip_level as usize], upload, false);
        }
        let stats = batch.finish(&gpu);
        assert_eq!(
            (
                stats.copy_commands,
                stats.deduplicated_copies,
                stats.submissions
            ),
            (6, 4, 1)
        );
        // Negative control: without the second record-0 copy the overlap would hold record 1.
        let mut skipped_repeat = expected_shared[0].clone();
        for upload in [&records[0], &records[1]] {
            apply(&mut skipped_repeat, upload, false);
        }
        assert_ne!(
            skipped_repeat, expected_shared[0],
            "records 0 and 1 must overlap"
        );
        for mip in 0..MIPS {
            let index = mip as usize;
            assert_eq!(
                read(&gpu, &legacy_shared, mip),
                expected_shared[index],
                "legacy shared mip={mip}"
            );
            assert_eq!(
                read(&gpu, &legacy_private, mip),
                expected_private[index],
                "legacy private mip={mip}"
            );
            assert_eq!(
                read(&gpu, &shared, mip),
                expected_shared[index],
                "batched shared mip={mip}"
            );
            assert_eq!(
                read(&gpu, &private, mip),
                expected_private[index],
                "batched private mip={mip}"
            );
        }
        assert!(
            pollster::block_on(errors.pop()).is_none(),
            "unexpected wgpu validation error"
        );
    }

    fn share_atlas() -> mc_assets::Atlas {
        let level = |width: u32, height: u32, seed: u8| -> Vec<u8> {
            (0..width * height * 4)
                .map(|i| seed.wrapping_add((i as u8).wrapping_mul(29)))
                .collect()
        };
        mc_assets::Atlas {
            texture: level(32, 16, 3),
            mipmaps: vec![level(16, 8, 5), level(8, 4, 7)],
            w: 32,
            h: 16,
        }
    }

    fn share_levels(atlas: &mc_assets::Atlas) -> Vec<Vec<u8>> {
        std::iter::once(atlas.texture.clone())
            .chain(atlas.mipmaps.iter().cloned())
            .collect()
    }

    fn share_descriptor(
        levels: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::TextureDescriptor<'static> {
        wgpu::TextureDescriptor {
            label: Some("atlas share test"),
            size: wgpu::Extent3d {
                width: 32,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        }
    }

    /// X14: inside a scope only requests for exactly the scope's atlas (bytes, shape, usage, format,
    /// anaglyph) share one texture; anything else, any request outside a scope, after it ends or with
    /// the switch off is a plain create.
    #[test]
    fn atlas_share_scope_adopts_only_byte_identical_requests() {
        let gpu = Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let atlas = share_atlas();
        let levels = share_levels(&atlas);
        let plain = share_descriptor(3, SHARED_ATLAS_FORMAT, SHARED_ATLAS_USAGE);
        let outside = share_atlas_texture(&gpu, &plain, &levels, false);
        assert_ne!(
            outside,
            share_atlas_texture(&gpu, &plain, &levels, false),
            "no scope, no sharing"
        );
        {
            let scope = AtlasShareScope::begin_if(true, &atlas, false);
            assert!(scope.is_active());
            let first = share_atlas_texture(&gpu, &plain, &levels, false);
            assert_ne!(first, outside);
            assert_eq!(first, share_atlas_texture(&gpu, &plain, &levels, false));
            assert_eq!(first, share_atlas_texture(&gpu, &plain, &levels, false));
            assert_eq!(scope.adopted(), 2);
            let mut changed_mip = levels.clone();
            changed_mip[2][5] ^= 1;
            let with_src = SHARED_ATLAS_USAGE | wgpu::TextureUsages::COPY_SRC;
            let misses = [
                share_atlas_texture(&gpu, &plain, &levels, true),
                share_atlas_texture(&gpu, &plain, &changed_mip, false),
                share_atlas_texture(
                    &gpu,
                    &share_descriptor(2, SHARED_ATLAS_FORMAT, SHARED_ATLAS_USAGE),
                    &levels[..2],
                    false,
                ),
                share_atlas_texture(
                    &gpu,
                    &share_descriptor(3, wgpu::TextureFormat::Rgba8UnormSrgb, SHARED_ATLAS_USAGE),
                    &levels,
                    false,
                ),
                share_atlas_texture(
                    &gpu,
                    &share_descriptor(3, SHARED_ATLAS_FORMAT, with_src),
                    &levels,
                    false,
                ),
            ];
            for (case, texture) in misses.iter().enumerate() {
                assert_ne!(
                    *texture, first,
                    "mismatch case {case} must get a private texture"
                );
            }
            assert_eq!(scope.adopted(), 2, "a mismatch never counts as adopted");
        }
        let after = share_atlas_texture(&gpu, &plain, &levels, false);
        assert_ne!(
            after,
            share_atlas_texture(&gpu, &plain, &levels, false),
            "ended scope shares nothing"
        );
        {
            let scope = AtlasShareScope::begin_if(false, &atlas, false);
            assert!(!scope.is_active());
            let a = share_atlas_texture(&gpu, &plain, &levels, false);
            assert_ne!(
                a,
                share_atlas_texture(&gpu, &plain, &levels, false),
                "disabled scope"
            );
            assert_eq!(scope.adopted(), 0);
        }
        assert!(
            pollster::block_on(errors.pop()).is_none(),
            "unexpected wgpu validation error"
        );
    }

    /// Owners holding the scope's shared texture get one shadow (a
    /// write through one is the other's); a private texture in the scope, a request outside it,
    /// or levels that no longer match the shared shadow keep their own.
    #[test]
    fn w21_mr_owners_of_the_shared_texture_share_one_shadow() {
        use std::rc::Rc;
        let Ok(gpu) = Gpu::headless(4, 4, crate::GpuOptions::default()) else {
            eprintln!("SKIP: no GPU adapter");
            return;
        };
        let atlas = share_atlas();
        let levels = share_levels(&atlas);
        let plain = share_descriptor(3, SHARED_ATLAS_FORMAT, SHARED_ATLAS_USAGE);
        let shared_on = crate::w21_mr::shared_shadow_enabled();

        let outside_texture = share_atlas_texture(&gpu, &plain, &levels, false);
        let outside = share_atlas_shadow(&outside_texture, levels.clone());
        let scope = AtlasShareScope::begin_if(true, &atlas, false);
        let first_texture = share_atlas_texture(&gpu, &plain, &levels, false);
        let first = share_atlas_shadow(&first_texture, levels.clone());
        let second_texture = share_atlas_texture(&gpu, &plain, &levels, false);
        assert_eq!(first_texture, second_texture);
        let second = share_atlas_shadow(&second_texture, levels.clone());
        assert_eq!(Rc::ptr_eq(&first, &second), shared_on);
        assert_eq!(*second.borrow(), levels);

        // A texture that is not the shared one (anaglyph differs) keeps its own shadow.
        let private_texture = share_atlas_texture(&gpu, &plain, &levels, true);
        assert_ne!(private_texture, first_texture);
        let private = share_atlas_shadow(&private_texture, levels.clone());
        assert!(!Rc::ptr_eq(&private, &first));

        // An animation record written through one owner is what the other replays.
        first.borrow_mut()[1][3] ^= 0x5A;
        let seen = second.borrow()[1][3] == first.borrow()[1][3];
        assert_eq!(seen, shared_on);
        if shared_on {
            // Levels that no longer match the shared shadow are kept private, never mixed.
            let third_texture = share_atlas_texture(&gpu, &plain, &levels, false);
            assert_eq!(third_texture, first_texture);
            let third = share_atlas_shadow(&third_texture, levels.clone());
            assert!(!Rc::ptr_eq(&third, &first));
            assert_eq!(*third.borrow(), levels);
        }
        drop(scope);
        assert!(!Rc::ptr_eq(&outside, &first));
        let after = share_atlas_shadow(&first_texture, levels.clone());
        assert!(!Rc::ptr_eq(&after, &first), "an ended scope shares nothing");
    }
}
