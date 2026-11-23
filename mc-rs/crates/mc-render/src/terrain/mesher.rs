//! `mesh_section` — the per-section mesher.
//!
//! This is the port of `RenderChunk.rebuildChunk` plus `BlockRendererDispatcher.renderBlock`
//! and `BlockModelRenderer`. It walks the 4096 blocks of one 16³ section, asks `mc_assets` for
//! each one's baked model, culls the faces whose neighbour hides them, lights every vertex
//! (flat or with [smooth lighting](super::ao)), tints it, and appends it to whichever of the
//! four render layers the block belongs to.
//!
//! It is a **pure function of `&World`** — no GPU, no globals it writes, no interior
//! mutability — which is what lets [`super::jobs`] run it on a `rayon` pool and what lets the
//! tests assert exact quad counts instead of eyeballing screenshots.
//!
//! # Differences from vanilla, deliberately
//!
//! * Vanilla builds a `RegionRenderCache` that memoises `getCombinedLight` **including the
//!   `lightValue` argument it was first called with**, so a later call for a different block
//!   at the same position can get a stale answer. [`super::light`] recomputes instead, which
//!   is the same value in every non-pathological case and never stale.
//! * Vanilla's client-only `Block.getActualState` pass is evaluated while meshing, without
//!   mutating the packed `id << 4 | meta` state stored by the world and network layers. The
//!   [`actual_state`] step below preserves that split; for stairs, its neighbour-derived shape
//!   selects straight, inner-corner, or outer-corner geometry before the widened model lookup.
//
// java: client/renderer/chunk/RenderChunk.java:115-222, client/renderer/BlockModelRenderer.java,
//       client/renderer/BlockRendererDispatcher.java:53-89

// Release builds compile a `debug_assert!` out together with any mutation in its condition.
// That once disabled the whole `actual_state` pass in the installed client (see `set_property`).
#![warn(clippy::debug_assert_with_mut_call)]

use std::any::Any;
use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};

use mc_assets::model::{
    model_for, model_for_actual, model_for_actual_at, model_for_at, BakedModel, BakedQuad,
};
use mc_data::block::ActualBlockState;
use mc_data::{
    block_info, block_props, can_catch_fire, state_properties, BlockState, Material, RenderLayer,
    RenderType,
};
use mc_math::{BlockPos, ChunkPos, ChunkSectionPos, Facing};
use mc_world::World;

use super::ao::{fill_quad_bounds, quad_shade, update_vertex_brightness};
use super::cull_face::{should_side_be_rendered, MeshSettings};
use super::light::{block_byte, mixed_brightness, sky_byte};
use super::tint::{color_multiplier, NO_TINT};
use super::vertex::{Buffer, TerrainVertex};
use super::visgraph::{SectionOcclusion, SetVisibility, VisGraph};

thread_local! {
    /// True only while a per-block renderer panic is being caught and enriched below.
    ///
    /// The client panic hook reads this to ignore the original, unannotated payload.  The
    /// replacement [`BlockMeshingPanic`] is raised after the catch and is the one that reaches
    /// the crash report, just as vanilla catches the renderer's throwable before it leaves
    /// `BlockRendererDispatcher.renderBlock`.
    static WRAPPING_BLOCK_PANIC: Cell<bool> = const { Cell::new(false) };
}

/// The per-block context preserved when terrain rendering panics.
///
/// java: `BlockRendererDispatcher.renderBlock` catches a `Throwable`, adds the exact packed
/// block and position to a `"Block being tesselated"` category, then throws a
/// `ReportedException`.  `mc-render` cannot depend on the client crash-report crate, so it
/// transports the same inputs in this payload; `mc-client` is the owner that turns it into the
/// rendered category.
#[derive(Debug, Clone)]
pub struct BlockMeshingPanic {
    pos: BlockPos,
    state: BlockState,
    cause: String,
}

impl BlockMeshingPanic {
    /// Construct the enriched renderer-panic payload from an already-rendered cause string.
    ///
    /// The normal mesher path uses its private [`Self::from_payload`] helper so ordinary Rust
    /// panic payloads retain their conventional rendering.  This constructor keeps the transport
    /// type testable by the client crash-report boundary without giving `mc-render` a dependency
    /// on that crate.
    #[must_use]
    pub fn new(pos: BlockPos, state: BlockState, cause: impl Into<String>) -> Self {
        Self {
            pos,
            state,
            cause: cause.into(),
        }
    }

    fn from_payload(pos: BlockPos, state: BlockState, payload: &(dyn Any + Send)) -> Self {
        let cause = if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_owned()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "Box<dyn Any>".to_owned()
        };
        Self::new(pos, state, cause)
    }

    /// The world coordinate of the block whose renderer panicked.
    #[must_use]
    pub fn pos(&self) -> BlockPos {
        self.pos
    }

    /// The packed block state presented to the renderer.
    #[must_use]
    pub fn state(&self) -> BlockState {
        self.state
    }

    /// The original Rust panic payload, rendered for the report's cause text.
    #[must_use]
    pub fn cause(&self) -> &str {
        &self.cause
    }
}

/// Whether the panic hook is observing the original panic a block renderer is about to enrich.
///
/// This is public for `mc-client`'s process-wide crash hook only.  It must not record the first
/// panic: after the catch, [`catch_block_meshing_panic`] raises [`BlockMeshingPanic`], whose
/// state and position let the hook produce vanilla's useful crash category.
#[must_use]
pub fn block_meshing_panic_is_being_wrapped() -> bool {
    WRAPPING_BLOCK_PANIC.with(Cell::get)
}

/// Run one block's renderer with vanilla's `BlockRendererDispatcher` catch boundary.
///
/// `panic_any` intentionally starts a new panic rather than `resume_unwind`: Rust invokes the
/// panic hook for the former, and the client hook is where the renderer payload is converted to a
/// crash report.  The original panic has already passed that hook while
/// [`block_meshing_panic_is_being_wrapped`] was true.
fn catch_block_meshing_panic(pos: BlockPos, state: BlockState, render: impl FnOnce()) {
    let result = WRAPPING_BLOCK_PANIC.with(|wrapping| {
        debug_assert!(
            !wrapping.get(),
            "block renderer catch boundaries must not nest"
        );
        wrapping.set(true);
        let result = panic::catch_unwind(AssertUnwindSafe(render));
        wrapping.set(false);
        result
    });
    if let Err(payload) = result {
        panic::panic_any(BlockMeshingPanic::from_payload(
            pos,
            state,
            payload.as_ref(),
        ));
    }
}

/// The four render layers of one section, plus the data the renderer needs to draw and cull it.
///
/// The frozen S11 contract is `ChunkMesh { solid, cutout, translucent }`; `cutout_mipped`,
/// `pos` and `visibility` are additive. `cutout_mipped` exists because S11's own task list
/// asks for all four of vanilla's `EnumWorldBlockLayer`s as separate buffers, and vanilla
/// really does draw leaves (mipped) and glass (unmipped) with different samplers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkMesh {
    /// Which section this was built from.
    pub pos: ChunkSectionPos,
    /// `EnumWorldBlockLayer.SOLID`.
    pub solid: Buffer,
    /// `EnumWorldBlockLayer.CUTOUT_MIPPED` — leaves.
    pub cutout_mipped: Buffer,
    /// `EnumWorldBlockLayer.CUTOUT` — glass, plants, rails.
    pub cutout: Buffer,
    /// `EnumWorldBlockLayer.TRANSLUCENT` — water, ice, stained glass. Sorted back to front by
    /// [`super::sort::sort_translucent`] before upload and re-sorted when the camera moves.
    pub translucent: Buffer,
    /// The section's face-to-face visibility graph, for vanilla's occlusion culling.
    pub visibility: SetVisibility,
    /// The mesh-time opaque-cell snapshot used for `RenderGlobal.getVisibleFacings` when this
    /// section contains the camera. It is 512 bytes, not a second copy of the section's blocks.
    pub occlusion: SectionOcclusion,
}

impl ChunkMesh {
    /// An empty mesh for `pos` (an all-air section, or one whose chunk is not loaded).
    #[must_use]
    pub fn empty(pos: ChunkSectionPos) -> Self {
        Self {
            pos,
            visibility: SetVisibility::all(true),
            occlusion: SectionOcclusion::default(),
            ..Self::default()
        }
    }

    /// Whether the section produced no geometry at all. java: `CompiledChunk.isEmpty`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers().all(|(_, b)| b.is_empty())
    }

    /// Total quads across all four layers.
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.layers().map(|(_, b)| b.quad_count()).sum()
    }

    /// Bytes this section will occupy in GPU vertex memory, across all four layers.
    ///
    /// Exactly what gets uploaded — `quad_count() * 4 * VERTEX_SIZE` — so it can be summed into
    /// a real memory figure rather than an estimate. Added by S23 for the GPU budget
    /// ([`GpuResidency`](super::jobs::GpuResidency)) and the batched upload planner.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.layers().map(|(_, b)| b.byte_size()).sum()
    }

    /// Every layer with its vanilla `EnumWorldBlockLayer`, in vanilla's ordinal order.
    pub fn layers(&self) -> impl Iterator<Item = (RenderLayer, &Buffer)> {
        [
            (RenderLayer::Solid, &self.solid),
            (RenderLayer::CutoutMipped, &self.cutout_mipped),
            (RenderLayer::Cutout, &self.cutout),
            (RenderLayer::Translucent, &self.translucent),
        ]
        .into_iter()
    }

    fn buffer_mut(&mut self, layer: RenderLayer) -> &mut Buffer {
        match layer {
            RenderLayer::Solid => &mut self.solid,
            RenderLayer::CutoutMipped => &mut self.cutout_mipped,
            RenderLayer::Cutout => &mut self.cutout,
            RenderLayer::Translucent => &mut self.translucent,
        }
    }
}

/// Mesh one 16³ section. Pure and thread-safe; frozen S11 API.
///
/// Uses vanilla's default graphics settings; [`mesh_section_with`] takes them explicitly.
#[must_use]
pub fn mesh_section(world: &World, pos: ChunkSectionPos) -> ChunkMesh {
    mesh_section_with(world, pos, MeshSettings::default())
}

/// [`mesh_section`] with explicit graphics settings (fancy graphics, smooth lighting).
///
/// # The vacant-section fast path (S23)
///
/// A section with no blocks in it at all — an unloaded chunk, a section the server never sent,
/// or one that is entirely air — returns [`ChunkMesh::empty`] without walking its 4096 block
/// positions. This is not a micro-optimisation: at render distance 12 the client holds
/// 25 × 25 × 16 = 10,000 sections and the great majority of them are sky, so the walk it skips
/// is most of the work of loading a world.
///
/// It is exact rather than approximate. The walk over an all-air section emits no quads (air's
/// `render_type` is `Invisible`) and marks nothing opaque, so its `VisGraph` resolves to
/// all-connected — which is precisely what `ChunkMesh::empty` carries. `mesh_walk` is kept
/// separately and `the_vacant_fast_path_agrees_with_the_full_walk` asserts the two produce
/// **equal** meshes, including for a section that is air but whose storage exists.
#[must_use]
pub fn mesh_section_with(world: &World, pos: ChunkSectionPos, settings: MeshSettings) -> ChunkMesh {
    mesh_section_with_alternatives(world, pos, settings, true)
}

/// [`mesh_section_with`] with vanilla's `allowBlockAlternatives` model-selection toggle.
///
/// With alternatives disabled, a weighted blockstate uses its deterministic base model rather
/// than hashing the block position. java: `BlockRendererDispatcher.java:100,126`.
#[must_use]
pub fn mesh_section_with_alternatives(
    world: &World,
    pos: ChunkSectionPos,
    settings: MeshSettings,
    allow_block_alternatives: bool,
) -> ChunkMesh {
    if section_is_vacant(world, pos) {
        return ChunkMesh::empty(pos);
    }
    mesh_walk_with_alternatives(world, pos, settings, allow_block_alternatives)
}

/// Whether `pos` holds no blocks at all, so meshing it cannot produce geometry.
///
/// Three ways to be vacant, cheapest first: the chunk is not loaded, the vertical index is
/// outside `0..16` or its section was never sent, or the section exists and every block in it is
/// air (`ExtendedBlockStorage.isEmpty`, an 8 KiB scan against a 4096-position walk).
fn section_is_vacant(world: &World, pos: ChunkSectionPos) -> bool {
    let Some(chunk) = world.chunk(ChunkPos::new(pos.x, pos.z)) else {
        return true;
    };
    let Ok(index) = u8::try_from(pos.y) else {
        return true;
    };
    match chunk.section(index) {
        None => true,
        Some(section) => section.is_empty(),
    }
}

/// Whether `pos` has no block storage at all: its column is unloaded, its vertical index is outside
/// `0..16`, or the server never sent that section. This is the constant-time half of
/// [`section_is_vacant`], cheap enough to test on the game thread before dispatching a task.
pub(super) fn section_is_absent(world: &World, pos: ChunkSectionPos) -> bool {
    let Some(chunk) = world.chunk(ChunkPos::new(pos.x, pos.z)) else {
        return true;
    };
    let Ok(index) = u8::try_from(pos.y) else {
        return true;
    };
    chunk.section(index).is_none()
}

/// The full 4096-position walk. Split out of [`mesh_section_with`] so the fast path above can be
/// tested *against* it rather than merely believed.
#[must_use]
#[cfg(test)]
fn mesh_walk(world: &World, pos: ChunkSectionPos, settings: MeshSettings) -> ChunkMesh {
    mesh_walk_with_alternatives(world, pos, settings, true)
}

fn mesh_walk_with_alternatives(
    world: &World,
    pos: ChunkSectionPos,
    settings: MeshSettings,
    allow_block_alternatives: bool,
) -> ChunkMesh {
    mesh_walk_impl::<true>(world, pos, settings, allow_block_alternatives)
}

/// The mesher with the per-block fast paths **disabled**: the string-keyed
/// actual-state model lookup for every block, a fresh `color_multiplier` for every tinted quad,
/// and the general `should_side_be_rendered` that re-reads the block for its bounds.
///
/// It exists only so tests (including `tests/mesher_fast_path_parity.rs`, which needs real
/// baked models in its own process) can prove the fast paths produce byte-identical meshes.
/// Production code must call [`mesh_section_with_alternatives`].
#[doc(hidden)]
#[must_use]
pub fn mesh_section_reference(
    world: &World,
    pos: ChunkSectionPos,
    settings: MeshSettings,
    allow_block_alternatives: bool,
) -> ChunkMesh {
    if section_is_vacant(world, pos) {
        return ChunkMesh::empty(pos);
    }
    mesh_walk_impl::<false>(world, pos, settings, allow_block_alternatives)
}

/// `FAST` selects the per-block fast paths (see [`mesh_section_reference`]). It is a const
/// generic so the production walk carries no runtime switch at all.
fn mesh_walk_impl<const FAST: bool>(
    world: &World,
    pos: ChunkSectionPos,
    settings: MeshSettings,
    allow_block_alternatives: bool,
) -> ChunkMesh {
    mc_core::profile_scope!("terrain.mesh_section");

    let mut mesh = ChunkMesh {
        pos,
        ..ChunkMesh::default()
    };
    let mut vis = VisGraph::new();
    let origin = pos.origin();
    // Connected textures: `None` (the vanilla path) unless the mod is on and
    // the installed bake carries a CTM table. Read once per section.
    let ctm = super::ctm::CtmSection::new(settings.connected_textures);

    for y in 0..16i32 {
        for z in 0..16i32 {
            for x in 0..16i32 {
                let bp = origin.add(x, y, z);
                let state = world.block_at(bp);
                if state == BlockState::AIR {
                    // The overwhelmingly common case even in a non-vacant section. Air is never
                    // opaque and never renders, so both calls below are known answers —
                    // `air_is_never_opaque_under_either_graphics_setting` pins the first of them.
                    continue;
                }
                let props = block_props(state);

                // java: RenderChunk.rebuildChunk feeds `block.isOpaqueCube()` to the VisGraph,
                // which for leaves depends on the graphics setting (see `is_opaque_cube`).
                if super::cull_face::is_opaque_cube(state, settings) {
                    vis.set_opaque(x, y, z);
                }

                // The Tall Grass mod turned off: grass and ferns get no geometry.
                // One `bool` test per block while it is on (vanilla). Grass is never opaque, so
                // the visibility graph above is the same either way, and the world is untouched.
                if settings.hide_tall_grass
                    && super::tall_grass::is_hidden_tall_grass(world, state, bp)
                {
                    continue;
                }

                // java: `BlockRendererDispatcher.renderBlock` catches `Throwable` around each
                // state it dispatches, rather than letting one bad model lose the identity of
                // the other 4,095 blocks in this section.  The cross-crate payload this emits is
                // consumed by mc-client's crash hook as `Block being tesselated`.
                catch_block_meshing_panic(bp, state, || match props.render_type {
                    // java: BlockRendererDispatcher.renderBlock — -1 draws nothing, 2 is a
                    // tile-entity renderer and contributes no chunk geometry.
                    RenderType::Invisible | RenderType::TileEntity => {}
                    RenderType::Liquid => {
                        let layer = super::cull_face::render_layer_of(state, settings);
                        let local = [x as f32, y as f32, z as f32];
                        super::fluid::render_fluid(
                            world,
                            state,
                            bp,
                            local,
                            settings,
                            mesh.buffer_mut(layer),
                        );
                    }
                    RenderType::Model => {
                        let layer = super::cull_face::render_layer_of(state, settings);
                        let local = [x as f32, y as f32, z as f32];
                        render_model::<FAST>(
                            world,
                            state,
                            bp,
                            local,
                            settings,
                            allow_block_alternatives,
                            ctm,
                            mesh.buffer_mut(layer),
                        );
                    }
                });
            }
        }
    }

    mesh.visibility = vis.compute_visibility();
    mesh.occlusion = vis.occlusion();
    mesh
}

/// java: `BlockModelRenderer.renderModel` — pick the AO or the flat path, then walk the six
/// culled face lists plus the uncullable general quads.
#[allow(clippy::too_many_arguments)]
fn render_model<const FAST: bool>(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    local: [f32; 3],
    settings: MeshSettings,
    allow_block_alternatives: bool,
    ctm: Option<super::ctm::CtmSection>,
    out: &mut Buffer,
) {
    let props = block_props(state);
    let model: &BakedModel =
        if FAST && !super::cull_face::class_info(state).actual_state && !world.is_debug_world() {
            // MESH-1 fast path. `actual_state` has no arm for this class, so it would return the
            // stored property tuple unchanged; `resolve_actual` of an unchanged tuple equals
            // `resolve(state)` (`resolve_actual_of_an_unchanged_stored_state_is_resolve`), and that
            // mapping is exactly what bake projected into `models[state.id]` and
            // `weighted[state.id]`. Same model, without the four allocations and two
            // `String`-keyed hash lookups per block.
            match model_lookup(allow_block_alternatives) {
                ModelLookup::Base => model_for(state),
                ModelLookup::PositionWeighted => model_for_at(state, [pos.x, pos.y, pos.z]),
            }
        } else {
            let actual = actual_state(world, state, pos);
            // `getActualState` often changes nothing: grass and mycelium
            // without snow above, every non-podzol dirt (about a quarter of a terrain section). An
            // unchanged tuple resolves exactly like the stored state (pinned for every bakeable
            // state by `resolve_actual_of_an_unchanged_stored_state_is_resolve`), so it takes the
            // same dense lookup as MESH-1 above instead of building and hashing two `String`s.
            if FAST && !world.is_debug_world() && actual.properties() == state_properties(state) {
                match model_lookup(allow_block_alternatives) {
                    ModelLookup::Base => model_for(state),
                    ModelLookup::PositionWeighted => model_for_at(state, [pos.x, pos.y, pos.z]),
                }
            } else {
                match model_lookup(allow_block_alternatives) {
                    ModelLookup::Base => model_for_actual(&actual),
                    ModelLookup::PositionWeighted => {
                        model_for_actual_at(&actual, [pos.x, pos.y, pos.z])
                    }
                }
            }
        };
    if model.is_empty() {
        return;
    }

    // java: `Minecraft.isAmbientOcclusionEnabled() && block.getLightValue() == 0
    //        && model.isAmbientOcclusion()` — a block that emits light never gets smooth
    // lighting, which is why glowstone and lava look flat.
    let smooth = settings.ambient_occlusion && props.light_emission == 0 && model.ambient_occlusion;

    // java: Block.getOffsetType — grass, flowers and double plants are nudged by a hash of
    // their position so a field of them is not a perfect grid.
    let offset = if FAST {
        position_offset_for(super::cull_face::class_info(state), pos)
    } else {
        position_offset(state, pos)
    };
    let local = [
        local[0] + offset[0],
        local[1] + offset[1],
        local[2] + offset[2],
    ];
    // MESH-6: every tinted quad of one block asks `color_multiplier` the same question.
    let mut tint = BlockTint::new(FAST);
    // One array load per block while connected textures are on, else `None`.
    let ctm = ctm.and_then(|section| section.block(state));
    // Shader-pack meshing for this block: one compare while no pack runs.
    let pack = super::phase::pack::PackQuad::for_block(state, settings.pack);

    for facing in Facing::VALUES {
        let quads = &model.face_quads[facing.index() as usize];
        if quads.is_empty() {
            continue;
        }
        let visible = if FAST {
            // `state` was read from `world` at `pos` by the walk, which is this variant's
            // precondition.
            super::cull_face::should_side_be_rendered_for_world_block(
                world, state, pos, facing, settings,
            )
        } else {
            should_side_be_rendered(world, state, pos, facing, settings)
        };
        if !visible {
            continue;
        }
        // Connected textures may move a quad onto a CTM tile or drop a merged
        // pane edge (`ctm_quad`); with `ctm == None` every quad passes through untouched.
        if smooth {
            for quad in quads {
                let mut remapped = None;
                let Some(quad) = ctm_quad(ctm, world, state, pos, quad, &mut remapped) else {
                    continue;
                };
                emit_smooth(
                    world, state, pos, local, quad, settings, pack, &mut tint, out,
                );
            }
        } else {
            // java: renderModelStandard passes the *neighbour's* brightness for face quads.
            let brightness = mixed_brightness(world, state, pos.relative(facing));
            for quad in quads {
                let mut remapped = None;
                let Some(quad) = ctm_quad(ctm, world, state, pos, quad, &mut remapped) else {
                    continue;
                };
                emit_flat(
                    world, state, pos, local, quad, brightness, pack, &mut tint, out,
                );
            }
        }
    }

    if !model.general_quads.is_empty() {
        // Glass panes' flat faces and edges are general quads.
        if smooth {
            for quad in &model.general_quads {
                let mut remapped = None;
                let Some(quad) = ctm_quad(ctm, world, state, pos, quad, &mut remapped) else {
                    continue;
                };
                emit_smooth(
                    world, state, pos, local, quad, settings, pack, &mut tint, out,
                );
            }
        } else {
            for quad in &model.general_quads {
                let mut remapped = None;
                let Some(quad) = ctm_quad(ctm, world, state, pos, quad, &mut remapped) else {
                    continue;
                };
                // java: renderModelStandardQuads with ownBrightness == true — a quad flush
                // with a face samples through that face, otherwise it samples its own block.
                let brightness = match quad.face {
                    Some(f) => {
                        let qb = fill_quad_bounds(quad, f, props.full_cube);
                        if qb.flush {
                            mixed_brightness(world, state, pos.relative(f))
                        } else {
                            mixed_brightness(world, state, pos)
                        }
                    }
                    None => mixed_brightness(world, state, pos),
                };
                emit_flat(
                    world, state, pos, local, quad, brightness, pack, &mut tint, out,
                );
            }
        }
    }
}

/// Connected textures for one quad: the quad itself when the block has no CTM
/// (always, with the mod off), a copy with tile UVs in `remapped`, or `None` for a merged pane
/// edge.
#[inline(always)]
fn ctm_quad<'q>(
    ctm: Option<super::ctm::CtmBlock>,
    world: &World,
    state: BlockState,
    pos: BlockPos,
    quad: &'q BakedQuad,
    remapped: &'q mut Option<BakedQuad>,
) -> Option<&'q BakedQuad> {
    match ctm {
        None => Some(quad),
        Some(ctm) => ctm.apply(world, state, pos, quad, remapped),
    }
}

/// One block's `colorMultiplier`, asked at most once.
///
/// java: `Block.colorMultiplier(world, pos, tintIndex)` — in 1.8.9 no block reads the tint
/// index (see [`color_multiplier`]), so every tinted quad of a block gets the same answer from
/// the same `(world, state, pos)`. The old path recomputed it per quad: up to 9 biome lookups
/// and 9 simplex samples each, ×6 for a fancy leaf, ×5 for grass with its side overlays.
struct BlockTint {
    memoize: bool,
    value: Option<u32>,
}

impl BlockTint {
    fn new(memoize: bool) -> Self {
        Self {
            memoize,
            value: None,
        }
    }

    fn of(&mut self, world: &World, state: BlockState, pos: BlockPos, quad: &BakedQuad) -> u32 {
        if quad.tint_index.is_none() {
            return NO_TINT;
        }
        if !self.memoize {
            return color_multiplier(world, state, pos);
        }
        *self
            .value
            .get_or_insert_with(|| color_multiplier(world, state, pos))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelLookup {
    Base,
    PositionWeighted,
}

/// Vanilla only enters `WeightedBakedModel.getAlternativeModel` when this setting is enabled.
const fn model_lookup(allow_block_alternatives: bool) -> ModelLookup {
    if allow_block_alternatives {
        ModelLookup::PositionWeighted
    } else {
        ModelLookup::Base
    }
}

/// Vanilla's `BlockRendererDispatcher.getModelFromBlockState` calls `getActualState` before
/// indexing `BlockModelShapes`. The packed chunk state remains untouched; this value exists only
/// for the model lookup.
pub fn actual_state(world: &World, stored: BlockState, pos: BlockPos) -> ActualBlockState {
    let mut state = ActualBlockState::from_stored(stored);
    // java: BlockRendererDispatcher.getModelFromBlockState skips getActualState for the
    // DEBUG_WORLD state grid. `World::block_at` has already supplied that grid's synthetic
    // state; use its complete tuple because the packed metadata representative is allowed to
    // collide for connection/shape/content properties.
    if world.is_debug_world() {
        return world.debug_world_actual_state_at(pos).unwrap_or(state);
    }
    let class = super::cull_face::class_of(stored);

    match class {
        "BlockBed" => {
            if state.get("part") == Some("foot") {
                let facing = state.get("facing").unwrap_or("north");
                let other = world.block_at(horizontal_offset(pos, facing));
                if other.block() == stored.block() {
                    copy_property(&mut state, other, "occupied");
                }
            }
        }
        "BlockDirt" => {
            if state.get("variant") == Some("podzol") {
                set_bool(&mut state, "snowy", is_snow(world.block_at(pos.up())));
            }
        }
        "BlockGrass" | "BlockMycelium" => {
            set_bool(&mut state, "snowy", is_snow(world.block_at(pos.up())));
        }
        "BlockDoor" => actual_door(world, pos, &mut state),
        "BlockDoublePlant" => {
            if state.get("half") == Some("upper") {
                let below = world.block_at(pos.down());
                if below.block() == stored.block() {
                    copy_property(&mut state, below, "variant");
                }
            }
        }
        "BlockFence" => {
            let material = stored.props().material;
            for (name, facing) in HORIZONTAL {
                set_bool(
                    &mut state,
                    name,
                    fence_can_connect(world, pos.relative(facing), material),
                );
            }
        }
        "BlockFenceGate" => {
            let axis_z = matches!(state.get("facing"), Some("north" | "south"));
            let wall = |p| block_class(world.block_at(p)) == "BlockWall";
            let in_wall = if axis_z {
                wall(pos.west()) || wall(pos.east())
            } else {
                wall(pos.north()) || wall(pos.south())
            };
            // Java only writes true; the stored state's default is false, but spelling the
            // complete answer makes this total even for hostile frozen state tuples.
            set_bool(&mut state, "in_wall", in_wall);
        }
        "BlockFire" => actual_fire(world, pos, &mut state),
        "BlockFlowerPot" => {
            let contents = world
                .tile_entity(pos)
                .and_then(|tile| tile.flower_pot())
                .map_or("empty", |pot| pot.contents());
            set_property(&mut state, "contents", contents);
        }
        "BlockPane" | "BlockStainedGlassPane" => {
            for (name, facing) in HORIZONTAL {
                set_bool(
                    &mut state,
                    name,
                    pane_can_connect(world, pos.relative(facing), stored.block()),
                );
            }
        }
        "BlockRedstoneRepeater" => {
            let locked = repeater_is_locked(world, pos, &state);
            set_bool(&mut state, "locked", locked);
        }
        "BlockRedstoneWire" => {
            for (name, facing) in HORIZONTAL {
                let value = wire_attach_position(world, pos, facing);
                set_property(&mut state, name, value);
            }
        }
        "BlockStairs" => {
            let shape = stair_shape(world, pos, &state);
            set_property(&mut state, "shape", shape);
        }
        "BlockStem" => {
            let crop = if stored.block().info().name == "minecraft:pumpkin_stem" {
                "minecraft:pumpkin"
            } else {
                "minecraft:melon_block"
            };
            let mut facing = "up";
            // EnumFacing.Plane.HORIZONTAL is N-E-S-W in 1.8.9.
            for (name, direction) in HORIZONTAL {
                if world.block_at(pos.relative(direction)).block().info().name == crop {
                    facing = name;
                    break;
                }
            }
            set_property(&mut state, "facing", facing);
        }
        "BlockTripWire" => {
            for (name, facing) in HORIZONTAL {
                let connected = tripwire_connected(world, pos, &state, facing);
                set_bool(&mut state, name, connected);
            }
        }
        "BlockTripWireHook" => {
            set_bool(
                &mut state,
                "suspended",
                !world.does_block_have_solid_top_surface(pos.down()),
            );
        }
        "BlockVine" => {
            set_bool(
                &mut state,
                "up",
                is_block_normal_cube(world.block_at(pos.up())),
            );
        }
        "BlockWall" => {
            set_bool(&mut state, "up", !world.block_at(pos.up()).props().is_air());
            for (name, facing) in HORIZONTAL {
                set_bool(
                    &mut state,
                    name,
                    wall_can_connect(world, pos.relative(facing), stored.block()),
                );
            }
        }
        _ => {}
    }
    state
}

const HORIZONTAL: [(&str, Facing); 4] = [
    ("north", Facing::North),
    ("east", Facing::East),
    ("south", Facing::South),
    ("west", Facing::West),
];

fn set_property(state: &mut ActualBlockState, name: &'static str, value: &'static str) {
    // The write must happen in every build. It used to be the `debug_assert!` condition itself,
    // which release builds compile out together with its side effect: the installed client
    // never applied a single actual-state property, so fences, walls, panes and wire drew as
    // posts and dots, every stair as an outer corner, door tops with the default facing and
    // every double-plant top as a peony. The module-level `debug_assert_with_mut_call` lint
    // keeps that from coming back.
    let declared = state.set(name, value);
    debug_assert!(declared, "{name} is not a property of {:?}", state.stored());
}

fn set_bool(state: &mut ActualBlockState, name: &'static str, value: bool) {
    set_property(state, name, if value { "true" } else { "false" });
}

fn copy_property(state: &mut ActualBlockState, source: BlockState, name: &'static str) {
    if let Some(value) = stored_property(source, name) {
        set_property(state, name, value);
    }
}

/// One property of a neighbour's stored tuple. Same answer as
/// `ActualBlockState::from_stored(state).get(name)` without copying the tuple into a `Vec`: the
/// neighbour reads below run for every stair, door and wire block the mesher visits.
fn stored_property(state: BlockState, name: &str) -> Option<&'static str> {
    let properties = mc_data::state_properties(state);
    properties
        .binary_search_by_key(&name, |(property, _)| *property)
        .ok()
        .map(|index| properties[index].1)
}

fn horizontal_offset(pos: BlockPos, facing: &str) -> BlockPos {
    match facing {
        "south" => pos.south(),
        "west" => pos.west(),
        "east" => pos.east(),
        _ => pos.north(),
    }
}

fn is_snow(state: BlockState) -> bool {
    matches!(
        state.block().info().name,
        "minecraft:snow" | "minecraft:snow_layer"
    )
}

fn is_block_normal_cube(state: BlockState) -> bool {
    state.props().material.blocks_movement() && state.props().full_cube
}

fn block_class(state: BlockState) -> &'static str {
    let class = block_info(state.block()).java_class;
    class.rsplit('.').next().unwrap_or(class)
}

fn actual_door(world: &World, pos: BlockPos, state: &mut ActualBlockState) {
    let stored = state.stored();
    if state.get("half") == Some("lower") {
        let upper = world.block_at(pos.up());
        if upper.block() == stored.block() {
            copy_property(state, upper, "hinge");
            copy_property(state, upper, "powered");
        }
    } else {
        let lower = world.block_at(pos.down());
        if lower.block() == stored.block() {
            copy_property(state, lower, "facing");
            copy_property(state, lower, "open");
        }
    }
}

fn fence_can_connect(world: &World, pos: BlockPos, self_material: Material) -> bool {
    let neighbour = world.block_at(pos);
    let props = neighbour.props();
    let class = block_class(neighbour);
    if class == "BlockBarrier" {
        return false;
    }
    if (class == "BlockFence" && props.material == self_material) || class == "BlockFenceGate" {
        return true;
    }
    props.material.is_opaque() && props.full_cube && props.material != Material::Gourd
}

fn pane_can_connect(world: &World, pos: BlockPos, self_block: mc_data::BlockId) -> bool {
    let props = world.block_at(pos).props();
    let info = block_info(props.block);
    let class = block_class(world.block_at(pos));
    props.full_block
        || props.block == self_block
        || matches!(
            info.name,
            "minecraft:glass" | "minecraft:stained_glass" | "minecraft:stained_glass_pane"
        )
        || matches!(class, "BlockPane" | "BlockStainedGlassPane")
}

fn wall_can_connect(world: &World, pos: BlockPos, self_block: mc_data::BlockId) -> bool {
    let props = world.block_at(pos).props();
    let class = block_class(world.block_at(pos));
    if class == "BlockBarrier" {
        return false;
    }
    if props.block == self_block || class == "BlockFenceGate" {
        return true;
    }
    props.material.is_opaque() && props.full_cube && props.material != Material::Gourd
}

fn actual_fire(world: &World, pos: BlockPos, state: &mut ActualBlockState) {
    let below = pos.down();
    if world.does_block_have_solid_top_surface(below)
        || can_catch_fire(world.block_at(below).block())
    {
        // `return getDefaultState()`: unlike the other overrides, fire discards AGE too.
        for name in ["alt", "east", "flip", "north", "south", "west"] {
            set_bool(state, name, false);
        }
        set_property(state, "age", "0");
        set_property(state, "upper", "0");
        return;
    }

    let alt = (pos.x + pos.y + pos.z) & 1 == 1;
    let flip = (pos.x / 2 + pos.y / 2 + pos.z / 2) & 1 == 1;
    set_bool(state, "alt", alt);
    set_bool(state, "flip", flip);
    for (name, facing) in HORIZONTAL {
        set_bool(
            state,
            name,
            can_catch_fire(world.block_at(pos.relative(facing)).block()),
        );
    }
    let upper = if can_catch_fire(world.block_at(pos.up()).block()) {
        if alt {
            "1"
        } else {
            "2"
        }
    } else {
        "0"
    };
    set_property(state, "upper", upper);
}

fn repeater_is_locked(world: &World, pos: BlockPos, state: &ActualBlockState) -> bool {
    let facing = state.get("facing").unwrap_or("north");
    let (left, right) = match facing {
        "south" | "north" => (Facing::West, Facing::East),
        _ => (Facing::North, Facing::South),
    };
    [left, right].into_iter().any(|side| {
        // java: `getPowerOnSide` asks the side diode's `getStrongPower`, non-zero only when it is
        // powered and faces `side`. A comparator's signal is its tile entity's output, which the
        // 1.8.9 server never sends (S35 has no comparator action), so on the client it is 0 and
        // only the powered repeater (94) locks.
        let neighbour = world.block_at(pos.relative(side));
        neighbour.block().0 == 94 && stored_property(neighbour, "facing") == Some(facing_name(side))
    })
}

fn facing_name(facing: Facing) -> &'static str {
    match facing {
        Facing::North => "north",
        Facing::South => "south",
        Facing::West => "west",
        Facing::East => "east",
        Facing::Down => "down",
        Facing::Up => "up",
    }
}

fn wire_attach_position(world: &World, pos: BlockPos, direction: Facing) -> &'static str {
    let neighbour_pos = pos.relative(direction);
    let neighbour = world.block_at(neighbour_pos);
    if wire_can_connect(neighbour, Some(direction))
        || (!is_block_normal_cube(neighbour)
            && wire_can_connect(world.block_at(neighbour_pos.down()), None))
    {
        return "side";
    }
    let above = world.block_at(pos.up());
    if !is_block_normal_cube(above)
        && is_block_normal_cube(neighbour)
        && wire_can_connect(world.block_at(neighbour_pos.up()), None)
    {
        "up"
    } else {
        "none"
    }
}

fn wire_can_connect(state: BlockState, side: Option<Facing>) -> bool {
    match state.block().0 {
        55 => true,
        // java: `Blocks.unpowered_repeater.isAssociated(block)` matches the two repeaters only.
        // Comparators (149/150) fall through to `canProvidePower() && side != null` below, so
        // wire joins a comparator from every side.
        93 | 94 => {
            let facing = stored_property(state, "facing").unwrap_or("north");
            side.is_some_and(|side| {
                facing == facing_name(side) || facing == facing_name(side.opposite())
            })
        }
        _ => side.is_some() && block_can_provide_power(state),
    }
}

fn block_can_provide_power(state: BlockState) -> bool {
    // java: `Block.canProvidePower` is overridden by exactly the classes listed below.
    // BlockEndPortalFrame is not one of them (its eye feeds `getComparatorInputOverride`, not
    // power), so an End Portal Frame never provides power, with or without the eye.

    // java: `BlockChest.canProvidePower` returns `this.chestType == 1`, so the shared
    // `BlockChest` class is not sufficient to identify a power source.  The two registered
    // instances are Blocks.chest (54) and Blocks.trapped_chest (146).
    if block_class(state) == "BlockChest" {
        return state.block().0 == 146;
    }
    matches!(
        block_class(state),
        "BlockBasePressurePlate"
            | "BlockPressurePlate"
            | "BlockPressurePlateWeighted"
            | "BlockButtonStone"
            | "BlockButtonWood"
            | "BlockCompressedPowered"
            | "BlockDaylightDetector"
            | "BlockLever"
            | "BlockRedstoneWire"
            | "BlockRedstoneTorch"
            | "BlockRailDetector"
            | "BlockRedstoneRepeater"
            | "BlockRedstoneComparator"
            | "BlockTripWireHook"
    )
}

fn tripwire_connected(
    world: &World,
    pos: BlockPos,
    state: &ActualBlockState,
    direction: Facing,
) -> bool {
    let neighbour = world.block_at(pos.relative(direction));
    match neighbour.block().0 {
        131 => stored_property(neighbour, "facing") == Some(facing_name(direction.opposite())),
        132 => stored_property(neighbour, "suspended") == state.get("suspended"),
        _ => false,
    }
}

fn stair_shape(world: &World, pos: BlockPos, state: &ActualBlockState) -> &'static str {
    let facing = state.get("facing").unwrap_or("north");
    let top = state.get("half") == Some("top");
    let outer = stair_corner(world, pos, facing, top, true);
    match outer {
        1 => "outer_right",
        2 => "outer_left",
        _ => match stair_corner(world, pos, facing, top, false) {
            1 => "inner_right",
            2 => "inner_left",
            _ => "straight",
        },
    }
}

/// Ports `BlockStairs.func_176307_f` (front/outer) and `func_176305_g` (back/inner).
fn stair_corner(world: &World, pos: BlockPos, facing: &str, top: bool, outer: bool) -> u8 {
    let (probe, first, first_guard, second, second_guard) = match (facing, outer) {
        ("east", true) => (pos.east(), "north", pos.south(), "south", pos.north()),
        ("west", true) => (pos.west(), "north", pos.south(), "south", pos.north()),
        ("south", true) => (pos.south(), "west", pos.east(), "east", pos.west()),
        ("north", true) => (pos.north(), "west", pos.east(), "east", pos.west()),
        ("east", false) => (pos.west(), "north", pos.north(), "south", pos.south()),
        ("west", false) => (pos.east(), "north", pos.north(), "south", pos.south()),
        ("south", false) => (pos.north(), "west", pos.west(), "east", pos.east()),
        _ => (pos.south(), "west", pos.west(), "east", pos.east()),
    };
    let Some(probe_state) = stair_at(world, probe) else {
        return 0;
    };
    if stored_property(probe_state, "half") == Some(if top { "top" } else { "bottom" }) {
        let probe_facing = stored_property(probe_state, "facing").unwrap_or("north");
        if probe_facing == first && !same_stair(world, first_guard, facing, top) {
            return stair_corner_code(facing, first, top);
        }
        if probe_facing == second && !same_stair(world, second_guard, facing, top) {
            return stair_corner_code(facing, second, top);
        }
    }
    0
}

fn stair_corner_code(facing: &str, neighbour: &str, top: bool) -> u8 {
    let base = match (facing, neighbour) {
        ("east", "north") | ("west", "south") | ("south", "east") | ("north", "west") => 2,
        _ => 1,
    };
    if top {
        3 - base
    } else {
        base
    }
}

fn stair_at(world: &World, pos: BlockPos) -> Option<BlockState> {
    let state = world.block_at(pos);
    (block_class(state) == "BlockStairs").then_some(state)
}

fn same_stair(world: &World, pos: BlockPos, facing: &str, top: bool) -> bool {
    stair_at(world, pos).is_some_and(|state| {
        stored_property(state, "facing") == Some(facing)
            && stored_property(state, "half") == Some(if top { "top" } else { "bottom" })
    })
}

/// java: `renderModelAmbientOcclusionQuads` for one quad.
#[allow(clippy::too_many_arguments)]
fn emit_smooth(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    local: [f32; 3],
    quad: &BakedQuad,
    settings: MeshSettings,
    pack: super::phase::pack::PackQuad,
    tint: &mut BlockTint,
    out: &mut Buffer,
) {
    let Some(face) = quad.face else {
        // A quad with no resolvable facing cannot index the neighbour tables; vanilla never
        // produces one from the 1.8.9 assets (see BakedQuad::face's docs). Fall back to flat.
        let brightness = mixed_brightness(world, state, pos);
        emit_flat(world, state, pos, local, quad, brightness, pack, tint, out);
        return;
    };
    let qb = fill_quad_bounds(quad, face, block_props(state).full_cube);
    let ao = update_vertex_brightness(world, state, pos, face, &qb, Some(&quad.verts), settings);
    let tint = tint.of(world, state, pos, quad);
    // `pack` is `PackQuad::VANILLA` (both calls are identities) unless a shader
    // pack runs.
    let shade = pack.shade(shade_byte(quad));

    let mut verts = [TerrainVertex::default(); 4];
    for (i, v) in quad.verts.iter().enumerate() {
        let [material, weight] = pack.light_meta(v.pos[1]);
        verts[i] = TerrainVertex::new(
            [
                local[0] + v.pos[0],
                local[1] + v.pos[1],
                local[2] + v.pos[2],
            ],
            v.uv,
            modulate(shade, tint, ao.color_multiplier[i]),
            [
                sky_byte(ao.brightness[i]),
                block_byte(ao.brightness[i]),
                material,
                weight,
            ],
        );
    }
    out.push_quad(verts);
}

/// java: `renderModelStandardQuads` for one quad — one brightness for all four vertices.
#[allow(clippy::too_many_arguments)]
fn emit_flat(
    world: &World,
    state: BlockState,
    pos: BlockPos,
    local: [f32; 3],
    quad: &BakedQuad,
    brightness: u32,
    pack: super::phase::pack::PackQuad,
    tint: &mut BlockTint,
    out: &mut Buffer,
) {
    let tint = tint.of(world, state, pos, quad);
    // java: `renderQuadsFlat` has no AO term — with a tint index it calls
    // `putColorMultiplier(f, f1, f2, ...)` with the tint channels alone, and without one it
    // leaves the baked shade byte untouched.
    // `pack` is `PackQuad::VANILLA` (both calls are identities) unless a shader
    // pack runs.
    let color = modulate(pack.shade(shade_byte(quad)), tint, 1.0);
    let (sky, block) = (sky_byte(brightness), block_byte(brightness));

    let mut verts = [TerrainVertex::default(); 4];
    for (i, v) in quad.verts.iter().enumerate() {
        let [material, weight] = pack.light_meta(v.pos[1]);
        verts[i] = TerrainVertex::new(
            [
                local[0] + v.pos[0],
                local[1] + v.pos[1],
                local[2] + v.pos[2],
            ],
            v.uv,
            color,
            [sky, block, material, weight],
        );
    }
    out.push_quad(verts);
}

/// The face shade as vanilla **stores** it on the baked quad, `0..=255`.
///
/// java: `FaceBakery.getFaceShadeColor(f)` (`FaceBakery.java:48-53`) —
/// `MathHelper.clamp_int((int)(f * 255.0F), 0, 255)`, packed into the quad's vertex colour at
/// bake time. The quantisation is observable: DOWN's `0.5F` becomes **127**, not `127.5`, and
/// everything downstream multiplies that integer. Keeping the shade as a float until the final
/// `as u8` makes every tinted or ambient-occluded DOWN face one 1/255 step too bright (16.9% of
/// sampled `ao x tint` combinations).
fn shade_byte(quad: &BakedQuad) -> f32 {
    ((quad_shade(quad) * 255.0) as i32).clamp(0, 255) as f32
}

/// java: `WorldRenderer.putColorMultiplier` (`WorldRenderer.java:297-325`) —
/// `(int)((float)(bakedByte & 255) * channel)`, where `channel` is
/// `aoFace.colorMultiplier[i] * (tintByte / 255.0F)`. The association matters: the baked byte is
/// multiplied by the *product* of the two floats, in that order.
fn modulate(shade_byte: f32, tint: u32, ao: f32) -> [u8; 4] {
    let ch = |shift: u32| {
        let c = ((tint >> shift) & 255) as f32 / 255.0;
        (shade_byte * (ao * c)).clamp(0.0, 255.0) as u8
    };
    [ch(16), ch(8), ch(0), 255]
}

/// java: `BlockModelRenderer`'s `EnumOffsetType` block — flowers, tall grass and double
/// plants are displaced by a hash of their column so a meadow does not look like a grid.
fn position_offset(state: BlockState, pos: BlockPos) -> [f32; 3] {
    let class = super::cull_face::class_of(state);
    let xyz = match class {
        // java: BlockTallGrass.getOffsetType -> XYZ
        "BlockTallGrass" => true,
        // java: BlockFlower (BlockYellowFlower/BlockRedFlower), BlockDoublePlant -> XZ
        "BlockYellowFlower" | "BlockRedFlower" | "BlockDoublePlant" => false,
        _ => return [0.0, 0.0, 0.0],
    };
    position_offset_hash(pos, xyz)
}

/// [`position_offset`] with the offset type from the per-block class table instead of a class
/// string match per block.
fn position_offset_for(info: super::cull_face::BlockClassInfo, pos: BlockPos) -> [f32; 3] {
    if info.offset_xyz {
        position_offset_hash(pos, true)
    } else if info.offset_xz {
        position_offset_hash(pos, false)
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn position_offset_hash(pos: BlockPos, xyz: bool) -> [f32; 3] {
    // java: `BlockModelRenderer.renderModelStandardQuads` (1.8.9:253-266), not
    // `MathHelper.getPositionRandom`.  The standard renderer's offset path deliberately hashes
    // the block's x/z column only; its inline expression has no `^ (long)y`.  This is a distinct
    // seam from the weighted-model picker above, which does call `getPositionRandom` and must
    // use `mc_math::coordinate_random`.  Every vanilla offset model has `ambientocclusion:false`,
    // so this standard path is the one that renders flowers, grass and double plants even when
    // smooth lighting is enabled.
    let mut i =
        i64::from(pos.x.wrapping_mul(3_129_871)) ^ i64::from(pos.z).wrapping_mul(116_129_781);
    i = i
        .wrapping_mul(i)
        .wrapping_mul(42_317_861)
        .wrapping_add(i.wrapping_mul(11));
    let dx = ((((i >> 16) & 15) as f32 / 15.0) - 0.5) * 0.5;
    let dz = ((((i >> 24) & 15) as f32 / 15.0) - 0.5) * 0.5;
    let dy = if xyz {
        ((((i >> 20) & 15) as f32 / 15.0) - 1.0) * 0.2
    } else {
        0.0
    };
    [dx, dy, dz]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::fixture;

    #[test]
    fn per_block_panic_preserves_the_exact_state_and_coordinate() {
        let pos = BlockPos::new(-12, 70, 33);
        let state = fixture::state_of("minecraft:oak_stairs");
        let panic = std::panic::catch_unwind(|| {
            catch_block_meshing_panic(pos, state, || panic!("broken baked model"));
        })
        .expect_err("the renderer panic must remain fatal");
        let panic = panic
            .downcast::<BlockMeshingPanic>()
            .expect("the client must receive the enriched block payload");
        assert_eq!(panic.pos(), pos);
        assert_eq!(panic.state(), state);
        assert_eq!(panic.cause(), "broken baked model");
        assert!(
            !block_meshing_panic_is_being_wrapped(),
            "the guard must be clear before the replacement panic reaches the client hook"
        );
    }

    #[test]
    fn blocks_without_an_actual_state_arm_actualise_to_their_stored_tuple() {
        // `render_model`'s dense model fast path skips `actual_state` for every class not in
        // `cull_face::ACTUAL_STATE_CLASSES`. Check that, for every legal state of every other
        // class and a spread of neighbourhoods, `actual_state` really is the stored tuple — so
        // an override arm added without listing its class fails here instead of silently
        // rendering the stored-state model.
        let centre = BlockPos::new(8, 64, 8);
        let neighbourhoods = [
            "minecraft:air",
            "minecraft:stone",
            "minecraft:snow_layer",
            "minecraft:fence",
            "minecraft:cobblestone_wall",
            "minecraft:glass_pane",
            "minecraft:redstone_wire",
            "minecraft:oak_stairs",
            "minecraft:grass",
            "minecraft:netherrack",
        ];
        let mut checked = 0;
        for name in neighbourhoods {
            let neighbour = fixture::state_of(name);
            let mut w = fixture::blank_world(0);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        w.set_block(centre.add(dx, dy, dz), neighbour);
                    }
                }
            }
            for state in mc_data::legal_states() {
                if super::super::cull_face::class_info(state).actual_state {
                    continue;
                }
                w.set_block(centre, state);
                assert_eq!(
                    actual_state(&w, state, centre),
                    ActualBlockState::from_stored(state),
                    "{state:?} among {name}"
                );
                checked += 1;
            }
        }
        assert!(checked > 1_000);
    }

    #[test]
    fn the_class_table_lists_exactly_the_classes_actual_state_rewrites() {
        // The converse direction: a listed class whose states never change would only cost the
        // fast path, but a typo would silently disable it, so pin the list against `mc_data`.
        for class in super::super::cull_face::ACTUAL_STATE_CLASSES {
            assert!(
                mc_data::legal_states().any(|s| super::super::cull_face::class_of(s) == class),
                "{class} names no registered block"
            );
        }
    }

    #[test]
    fn an_empty_section_produces_nothing() {
        let w = fixture::blank_world(1);
        let mesh = mesh_section(&w, ChunkSectionPos::new(0, 4, 0));
        assert!(mesh.is_empty());
        assert_eq!(mesh.quad_count(), 0);
        assert_eq!(mesh.pos, ChunkSectionPos::new(0, 4, 0));
    }

    #[test]
    fn a_section_of_an_unloaded_chunk_produces_nothing() {
        let w = fixture::blank_world(0);
        let mesh = mesh_section(&w, ChunkSectionPos::new(50, 4, 50));
        assert!(mesh.is_empty());
    }

    #[test]
    fn layers_are_listed_in_vanilla_ordinal_order() {
        // java: EnumWorldBlockLayer — SOLID, CUTOUT_MIPPED, CUTOUT, TRANSLUCENT.
        let mesh = ChunkMesh::empty(ChunkSectionPos::new(0, 0, 0));
        let order: Vec<_> = mesh.layers().map(|(l, _)| l).collect();
        assert_eq!(
            order,
            [
                RenderLayer::Solid,
                RenderLayer::CutoutMipped,
                RenderLayer::Cutout,
                RenderLayer::Translucent
            ]
        );
    }

    #[test]
    fn alternate_blocks_setting_selects_weighted_or_base_lookup() {
        assert_eq!(model_lookup(true), ModelLookup::PositionWeighted);
        assert_eq!(model_lookup(false), ModelLookup::Base);
    }

    #[test]
    fn modulate_scales_every_channel() {
        assert_eq!(modulate(255.0, NO_TINT, 1.0), [255, 255, 255, 255]);
        assert_eq!(modulate(127.0, NO_TINT, 1.0), [127, 127, 127, 255]);
        assert_eq!(modulate(255.0, 0x00FF_0000, 1.0), [255, 0, 0, 255]);
    }

    /// java: `FaceBakery.getFaceShadeColor` bakes the shade to an integer byte
    /// (`(int)(0.5F * 255.0F) == 127`) and `putColorMultiplier` multiplies *that* by
    /// `ao * tint`. Folding the raw `0.5F` through instead and quantising at the end uses
    /// `127.5` as the base, which rounds up on 16.9% of `ao x tint` combinations.
    ///
    /// The oracle below is the pair of formulas evaluated over `ao` in 1/1000 steps across
    /// `[0.2, 1.0]` (`AmbientOcclusionFace.colorMultiplier`'s real range) x tint bytes
    /// `{255, 200, 128, 63}` = 3,204 combinations per shade. Where they differ it is always by
    /// exactly one 1/255 step, in either direction: the two forms also *associate* the three
    /// multiplies differently, and vanilla's association (`bakedByte * (ao * tint)`) is the one
    /// `new` reproduces.
    ///
    /// Negative control: `shade = 1.0` must differ on **zero** combinations, because
    /// `1.0f * 255` is exact and the quantisation is then a no-op. If the sweep reported
    /// differences there, the two formulas would be disagreeing for some reason other than the
    /// quantisation and the 0.5 count would prove nothing.
    ///
    /// Note the counts: `P2-client.renderer-02`'s report claims 0.6 and 0.8 differ on zero
    /// combinations. On this tint sample they differ on 4 and 6 — the report's "0" was true of
    /// its own four tint bytes, not of the formula. The 0.5 face is still where essentially all
    /// of it lives.
    #[test]
    fn the_face_shade_is_quantised_to_a_byte_before_the_ao_and_tint_multiply() {
        assert_eq!(((0.5f32 * 255.0) as i32).clamp(0, 255), 127);
        let old = |tint_byte: u32, shade: f32, ao: f32| {
            let c = tint_byte as f32 / 255.0;
            ((c * (ao * shade)) * 255.0) as u8
        };
        let new = |tint_byte: u32, shade: f32, ao: f32| {
            let baked = ((shade * 255.0) as i32).clamp(0, 255) as f32;
            let c = tint_byte as f32 / 255.0;
            (baked * (ao * c)).clamp(0.0, 255.0) as u8
        };
        for (shade, want_diffs) in [(0.5f32, 643), (0.6, 4), (0.8, 6), (1.0, 0)] {
            let mut diffs = 0;
            for tint_byte in [255u32, 200, 128, 63] {
                for step in 200..=1000 {
                    let ao = step as f32 / 1000.0;
                    let (o, n) = (old(tint_byte, shade, ao), new(tint_byte, shade, ao));
                    if o != n {
                        assert_eq!(
                            (i32::from(o) - i32::from(n)).abs(),
                            1,
                            "only ever one step, at shade {shade} tint {tint_byte} ao {ao}"
                        );
                        diffs += 1;
                    }
                }
            }
            assert_eq!(diffs, want_diffs, "shade {shade}");
        }
    }

    // ---- S23: the vacant-section fast path -------------------------------------------------

    #[test]
    fn the_vacant_fast_path_agrees_with_the_full_walk() {
        // The fast path is only sound if it returns *exactly* what the walk would have. Assert
        // that on every shape of section the fast path can see, with the walk as the reference.
        let settings = MeshSettings::default();
        let mut w = fixture::blank_world(1);
        // Water rather than stone, because these are unit tests: no resource pack is installed,
        // so `model_for` answers with the empty model and a stone section produces no quads.
        // `terrain::fluid` does not go through the model registry, so it is the one block that
        // draws in a bare unit test — the same reason `fluid.rs`'s own tests use it.
        let stone = fixture::state_of("minecraft:stone");
        let water = fixture::state_of("minecraft:water");
        fixture::fill(
            &mut w,
            BlockPos::new(0, 16, 0),
            BlockPos::new(15, 17, 15),
            stone,
        );
        fixture::fill(
            &mut w,
            BlockPos::new(0, 18, 0),
            BlockPos::new(15, 19, 15),
            water,
        );

        // 1. A section whose storage exists and is entirely air: the fast path fires.
        let air = ChunkSectionPos::new(0, 4, 0);
        assert!(section_is_vacant(&w, air));
        assert_eq!(
            mesh_section_with(&w, air, settings),
            mesh_walk(&w, air, settings)
        );

        // 2. A section in an unloaded chunk.
        let unloaded = ChunkSectionPos::new(40, 4, 40);
        assert!(section_is_vacant(&w, unloaded));
        assert_eq!(
            mesh_section_with(&w, unloaded, settings),
            mesh_walk(&w, unloaded, settings)
        );

        // 3. A vertical index outside the world.
        let above = ChunkSectionPos::new(0, 20, 0);
        assert!(section_is_vacant(&w, above));
        assert_eq!(
            mesh_section_with(&w, above, settings),
            mesh_walk(&w, above, settings)
        );

        // 4. **The control**: a section with real geometry must NOT take the fast path, and the
        // two must still agree. Without this the test would pass just as happily if the fast
        // path fired for everything and returned nothing.
        let solid = ChunkSectionPos::new(0, 1, 0);
        assert!(!section_is_vacant(&w, solid));
        let fast = mesh_section_with(&w, solid, settings);
        let walked = mesh_walk(&w, solid, settings);
        assert_eq!(fast, walked);
        assert!(fast.quad_count() > 0, "the control section must draw");
        assert_ne!(
            fast,
            ChunkMesh::empty(solid),
            "if the fast path had fired here it would have been wrong"
        );
        assert_ne!(
            fast.visibility,
            SetVisibility::all(true),
            "a section with opaque blocks must not read as fully connected"
        );

        // 5. And one block is enough to disqualify a section, so the emptiness test is not a
        // "mostly empty" heuristic.
        let mut w2 = fixture::blank_world(0);
        fixture::place(&mut w2, BlockPos::new(3, 70, 9), "minecraft:stone");
        let one = ChunkSectionPos::new(0, 4, 0);
        assert!(!section_is_vacant(&w2, one));
        assert_eq!(
            mesh_section_with(&w2, one, settings),
            mesh_walk(&w2, one, settings)
        );
    }

    #[test]
    fn air_is_never_opaque_under_either_graphics_setting() {
        // The walk skips air before asking. That is only correct because the answer is known.
        for fancy in [false, true] {
            let settings = MeshSettings {
                fancy_graphics: fancy,
                ..MeshSettings::default()
            };
            assert!(!super::super::cull_face::is_opaque_cube(
                BlockState::AIR,
                settings
            ));
        }
    }

    #[test]
    fn byte_size_is_exactly_what_gets_uploaded() {
        let mut w = fixture::blank_world(0);
        // Water: see the note in `the_vacant_fast_path_agrees_with_the_full_walk`.
        let water = fixture::state_of("minecraft:water");
        fixture::fill(
            &mut w,
            BlockPos::new(0, 16, 0),
            BlockPos::new(15, 17, 15),
            water,
        );
        let mesh = mesh_section(&w, ChunkSectionPos::new(0, 1, 0));
        let summed: usize = mesh.layers().map(|(_, b)| b.as_bytes().len()).sum();
        assert_eq!(mesh.byte_size(), summed);
        assert_eq!(
            mesh.byte_size(),
            mesh.quad_count() * 4 * crate::terrain::VERTEX_SIZE
        );
        assert!(mesh.byte_size() > 0);
        assert_eq!(
            ChunkMesh::empty(ChunkSectionPos::new(0, 0, 0)).byte_size(),
            0
        );
    }

    #[test]
    fn only_offset_blocks_are_displaced() {
        let p = BlockPos::new(7, 64, -3);
        assert_eq!(
            position_offset(fixture::state_of("minecraft:stone"), p),
            [0.0; 3]
        );
        let grass = fixture::state_of("minecraft:tallgrass");
        let o = position_offset(grass, p);
        assert!(o[0] != 0.0 || o[2] != 0.0, "tall grass must be displaced");
        assert!(o[1] <= 0.0, "XYZ offset only ever moves down: {o:?}");
        assert!(o[0].abs() <= 0.25 && o[2].abs() <= 0.25);
        // XZ blocks never move vertically.
        let flower = fixture::state_of("minecraft:yellow_flower");
        assert_eq!(position_offset(flower, p)[1], 0.0);
    }

    /// `BlockModelRenderer.renderModelStandardQuads` hashes only x/z for offset blocks.  Double
    /// plants are deliberately XZ-offset only, which makes two otherwise identical plants at
    /// distinct heights the cleanest negative control for accidentally routing this path through
    /// `MathHelper.getPositionRandom`: a y-inclusive hash would move them differently.
    ///
    /// The literal long result transcribes the standard renderer's inline expression at
    /// `BlockModelRenderer.java:256-257`, including Java's int-width x product before widening.
    #[test]
    fn tall_plant_offsets_hash_its_xz_column_not_y() {
        let plant = fixture::state_of("minecraft:double_plant");
        let low = BlockPos::new(17, 64, -31);
        let high = BlockPos::new(17, 65, -31);

        assert_eq!(
            standard_offset_random(low.x, low.z),
            -5_865_527_175_364_811_454,
            "Java BlockModelRenderer standard offset hash (17, -31)"
        );

        let low_offset = position_offset(plant, low);
        let high_offset = position_offset(plant, high);
        assert_eq!(
            low_offset,
            [((10.0 / 15.0) - 0.5) * 0.5, 0.0, ((3.0 / 15.0) - 0.5) * 0.5],
            "the column hash uses nibbles 16=10 and 24=3"
        );
        assert_ne!(
            standard_offset_random(low.x, low.z),
            coordinate_random_oracle(low.x, low.y, low.z),
            "negative control: the y-inclusive weighted-model hash must differ"
        );
        assert_eq!(
            low_offset, high_offset,
            "changing only y must not move a standard-path double plant"
        );
    }

    /// `long`.  ±687 are the first ordinary coordinates either side of that overflow boundary.
    /// Independent spellings of the two Java hashes keep the standard offset seam from silently
    /// being unified with weighted-model selection.  The shared helper itself is tested in
    /// `mc-assets`; this local oracle is for the renderer's x/z-only path.
    fn standard_offset_random(x: i32, z: i32) -> i64 {
        let mut hash =
            i64::from(x.wrapping_mul(3_129_871)) ^ i64::from(z).wrapping_mul(116_129_781);
        hash = hash
            .wrapping_mul(hash)
            .wrapping_mul(42_317_861)
            .wrapping_add(hash.wrapping_mul(11));
        hash
    }

    fn coordinate_random_oracle(x: i32, y: i32, z: i32) -> i64 {
        let mut hash = standard_offset_random_seed(x, z) ^ i64::from(y);
        hash = hash
            .wrapping_mul(hash)
            .wrapping_mul(42_317_861)
            .wrapping_add(hash.wrapping_mul(11));
        hash
    }

    fn standard_offset_random_seed(x: i32, z: i32) -> i64 {
        i64::from(x.wrapping_mul(3_129_871)) ^ i64::from(z).wrapping_mul(116_129_781)
    }

    #[test]
    fn position_offsets_use_java_standard_xz_hash_at_overflow_boundary() {
        let plant = fixture::state_of("minecraft:double_plant");
        let cases = [
            (
                BlockPos::new(0, 64, 0),
                0_i64,
                [((0.0 / 15.0) - 0.5) * 0.5, 0.0, ((0.0 / 15.0) - 0.5) * 0.5],
            ),
            (
                BlockPos::new(686, 64, -31),
                8_324_486_923_028_287_170_i64,
                [((12.0 / 15.0) - 0.5) * 0.5, 0.0, ((0.0 / 15.0) - 0.5) * 0.5],
            ),
            (
                BlockPos::new(687, 64, -31),
                6_283_420_133_724_176_300_i64,
                [((10.0 / 15.0) - 0.5) * 0.5, 0.0, ((5.0 / 15.0) - 0.5) * 0.5],
            ),
            (
                BlockPos::new(-686, -20, 17),
                -1_885_603_651_383_866_762_i64,
                [((5.0 / 15.0) - 0.5) * 0.5, 0.0, ((12.0 / 15.0) - 0.5) * 0.5],
            ),
            (
                BlockPos::new(-687, -20, 17),
                -2_471_465_404_312_760_846_i64,
                [((5.0 / 15.0) - 0.5) * 0.5, 0.0, ((5.0 / 15.0) - 0.5) * 0.5],
            ),
        ];

        for (pos, expected_hash, expected_offset) in cases {
            assert_eq!(
                standard_offset_random(pos.x, pos.z),
                expected_hash,
                "Java BlockModelRenderer standard hash ({}, {})",
                pos.x,
                pos.z
            );
            assert_eq!(position_offset(plant, pos), expected_offset);
        }
    }

    #[test]
    fn fence_actual_state_tracks_each_neighbour_independently() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:fence");
        fixture::place(&mut world, pos.north(), "minecraft:stone");

        let connected = actual_state(&world, world.block_at(pos), pos);
        assert_eq!(connected.get("north"), Some("true"));
        assert_eq!(connected.get("east"), Some("false"));
        assert_eq!(connected.get("south"), Some("false"));
        assert_eq!(connected.get("west"), Some("false"));

        world.set_block(pos.north(), BlockState::AIR);
        let isolated = actual_state(&world, world.block_at(pos), pos);
        assert_eq!(isolated.get("north"), Some("false"));
        assert_ne!(connected.properties(), isolated.properties());
    }

    #[test]
    fn door_halves_copy_only_the_properties_the_other_half_owns() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        let door = mc_data::block_by_name("wooden_door").unwrap();
        let lower = BlockState::new(door, 0);
        let upper = BlockState::new(door, 8); // upper, left hinge, not powered
        world.set_block(pos, lower);
        world.set_block(pos.up(), upper);

        let lower_stored = ActualBlockState::from_stored(lower);
        let upper_stored = ActualBlockState::from_stored(upper);
        let lower_actual = actual_state(&world, lower, pos);
        let upper_actual = actual_state(&world, upper, pos.up());
        assert_eq!(lower_actual.get("hinge"), upper_stored.get("hinge"));
        assert_eq!(lower_actual.get("powered"), upper_stored.get("powered"));
        assert_eq!(upper_actual.get("facing"), lower_stored.get("facing"));
        assert_eq!(upper_actual.get("open"), lower_stored.get("open"));

        world.set_block(pos.up(), BlockState::AIR);
        assert_ne!(
            actual_state(&world, lower, pos).get("hinge"),
            upper_stored.get("hinge"),
            "negative control: an orphan lower half keeps its frozen property"
        );
    }

    #[test]
    fn stairs_resolve_outer_corners_and_the_neighbour_removal_bites() {
        let stairs = mc_data::block_by_name("oak_stairs").unwrap();
        let find = |facing: &str| {
            (0..8)
                .map(|meta| BlockState::new(stairs, meta))
                .find(|state| {
                    let p = ActualBlockState::from_stored(*state);
                    p.get("facing") == Some(facing) && p.get("half") == Some("bottom")
                })
                .expect("every horizontal bottom stair exists")
        };
        let centre = find("east");
        let corner = find("north");
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        world.set_block(pos, centre);
        world.set_block(pos.east(), corner);
        assert_eq!(
            actual_state(&world, centre, pos).get("shape"),
            Some("outer_left")
        );

        world.set_block(pos.east(), BlockState::AIR);
        assert_eq!(
            actual_state(&world, centre, pos).get("shape"),
            Some("straight")
        );
    }

    #[test]
    fn debug_world_keeps_the_synthetic_stair_state_stored() {
        let mut debug = World::new();
        debug.set_dimension_and_level_type(0, "debug_all_block_states");

        // The DEBUG_WORLD grid deliberately includes stair tuples whose shape is not encoded in
        // their packed metadata. With ordinary getActualState, the empty cardinal neighbours at
        // a grid cell would resolve that shape back to straight.
        let mut grid_stair = None;
        for x in (1..=77).step_by(2) {
            for z in (1..=77).step_by(2) {
                let pos = BlockPos::new(x, 70, z);
                let stored = debug.block_at(pos);
                if let Some(state) = debug.debug_world_actual_state_at(pos) {
                    if block_class(stored) == "BlockStairs"
                        && state.get("shape") != Some("straight")
                    {
                        grid_stair = Some((pos, stored, state));
                        break;
                    }
                }
            }
            if grid_stair.is_some() {
                break;
            }
        }
        let (debug_pos, stored, stored_state) =
            grid_stair.expect("DEBUG_WORLD grid must contain a non-straight stair state");

        assert_eq!(actual_state(&debug, stored, debug_pos), stored_state);

        // Negative control: the same packed state in an ordinary world still runs the
        // neighbour-dependent actual-state pass and loses its frozen corner shape.
        let mut normal = fixture::blank_world(0);
        let normal_pos = BlockPos::new(8, 70, 8);
        normal.set_block(normal_pos, stored);
        let normal_state = actual_state(&normal, stored, normal_pos);
        assert_eq!(normal_state.get("shape"), Some("straight"));
        assert_ne!(normal_state, stored_state);
    }

    #[test]
    fn wire_actual_state_distinguishes_side_up_and_none() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:redstone_wire");
        fixture::place(&mut world, pos.north(), "minecraft:redstone_wire");
        assert_eq!(
            actual_state(&world, world.block_at(pos), pos).get("north"),
            Some("side")
        );

        world.set_block(pos.north(), fixture::state_of("minecraft:stone"));
        world.set_block(
            pos.north().up(),
            fixture::state_of("minecraft:redstone_wire"),
        );
        assert_eq!(
            actual_state(&world, world.block_at(pos), pos).get("north"),
            Some("up")
        );

        world.set_block(pos.north().up(), BlockState::AIR);
        assert_eq!(
            actual_state(&world, world.block_at(pos), pos).get("north"),
            Some("none")
        );
    }

    #[test]
    fn redstone_wire_uses_chest_instance_power_semantics() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:redstone_wire");

        // java: BlockChest.canProvidePower() is `chestType == 1`; both registry entries share
        // BlockChest, but only the trapped instance is the redstone source.
        fixture::place(&mut world, pos.east(), "minecraft:chest");
        let ordinary = world.block_at(pos.east());
        assert_eq!(ordinary.block().0, 54);
        assert!(!block_can_provide_power(ordinary));
        assert_eq!(
            actual_state(&world, world.block_at(pos), pos).get("east"),
            Some("none"),
            "ordinary chest is the negative control"
        );

        fixture::place(&mut world, pos.east(), "minecraft:trapped_chest");
        let trapped = world.block_at(pos.east());
        assert_eq!(trapped.block().0, 146);
        assert!(block_can_provide_power(trapped));
        assert_eq!(
            actual_state(&world, world.block_at(pos), pos).get("east"),
            Some("side")
        );
    }

    #[test]
    fn redstone_wire_never_connects_to_end_portal_frames() {
        // java: BlockEndPortalFrame overrides `hasComparatorInputOverride`, not
        // `canProvidePower` (checked in the real 1.8.9.jar: class `ago` declares no `i()Z`), so
        // `Block.canProvidePower` answers false with or without the eye and
        // `BlockRedstoneWire.canConnectTo` never joins the frame's side. This test used to
        // assert that an eyed frame connects; the vanilla-bytecode oracle
        // (`tests/actual_state_java_oracle.rs`) showed otherwise.
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:redstone_wire");

        let frame = mc_data::block_by_name("end_portal_frame").unwrap();
        let empty_frame = BlockState::new(frame, 0);
        let eyed_frame = BlockState::new(frame, 4);
        let stone = fixture::state_of("minecraft:stone");

        for direction in [Facing::North, Facing::East, Facing::South, Facing::West] {
            let neighbour = pos.relative(direction);
            world.set_block(neighbour, eyed_frame);
            assert!(!block_can_provide_power(eyed_frame));
            assert_eq!(
                actual_state(&world, world.block_at(pos), pos).get(facing_name(direction)),
                Some("none"),
                "an eyed End Portal Frame does not connect on {direction:?}"
            );
            // The frame is still a normal cube, so wire on top of it climbs: `up`.
            world.set_block(neighbour.up(), fixture::state_of("minecraft:redstone_wire"));
            assert_eq!(
                actual_state(&world, world.block_at(pos), pos).get(facing_name(direction)),
                Some("up"),
                "wire on an End Portal Frame climbs on {direction:?}"
            );
            world.set_block(neighbour.up(), BlockState::AIR);

            world.set_block(neighbour, empty_frame);
            assert!(!block_can_provide_power(empty_frame));
            assert_eq!(
                actual_state(&world, world.block_at(pos), pos).get(facing_name(direction)),
                Some("none"),
                "an empty End Portal Frame does not connect on {direction:?}"
            );

            world.set_block(neighbour, stone);
            assert!(!block_can_provide_power(stone));
            assert_eq!(
                actual_state(&world, world.block_at(pos), pos).get(facing_name(direction)),
                Some("none"),
                "ordinary stone remains the non-power control on {direction:?}"
            );
        }
    }

    #[test]
    fn fire_actual_state_uses_identity_fire_catch_table() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:fire");
        fixture::place(&mut world, pos.north(), "minecraft:planks");
        fixture::place(&mut world, pos.east(), "minecraft:acacia_stairs");
        fixture::place(&mut world, pos.up(), "minecraft:acacia_stairs");

        let actual = actual_state(&world, world.block_at(pos), pos);
        assert_eq!(actual.get("north"), Some("true"));
        assert_eq!(
            actual.get("east"),
            Some("false"),
            "acacia stairs are deliberately absent from BlockFire.init"
        );
        assert_eq!(actual.get("upper"), Some("0"));

        // Positive control for the same upper-model branch: oak stairs are one of the four
        // stair instances listed by Java's BlockFire.init, while acacia stairs are not.
        fixture::place(&mut world, pos.up(), "minecraft:oak_stairs");
        let supported = actual_state(&world, world.block_at(pos), pos);
        assert!(matches!(supported.get("upper"), Some("1" | "2")));

        // A solid floor takes the early `getDefaultState()` path and clears the directional
        // actual-state values, independent of flammable neighbours.
        fixture::place(&mut world, pos.down(), "minecraft:stone");
        let grounded = actual_state(&world, world.block_at(pos), pos);
        assert_eq!(grounded.get("age"), Some("0"));
        assert_eq!(grounded.get("north"), Some("false"));
        assert_eq!(grounded.get("upper"), Some("0"));
    }

    // ---- the actual state reaches the model lookup ----------
    //
    // These name the exact `ModelResourceLocation` (`<blockstates file>#<variant>`) the mesher's
    // model lookup receives, as `assets/minecraft/blockstates/*.json` in the 1.8.9 jar spells it.
    // They must also hold with debug assertions off: the installed client is a release build, and
    // `set_property` once wrote the actual state only inside a `debug_assert!`.

    /// `"<blockstates file>#<variant>"` for the block at `pos`, after `getActualState`.
    fn model_location(world: &World, pos: BlockPos) -> String {
        let actual = actual_state(world, world.block_at(pos), pos);
        let mapping = mc_assets::model::state_mapper::resolve_actual(&actual);
        format!("{}#{}", mapping.file_stem, mapping.variant_key)
    }

    /// The first legal packed state of `name` whose stored tuple has every `wanted` value.
    fn stored_with(name: &str, wanted: &[(&str, &str)]) -> BlockState {
        let block = mc_data::block_by_name(name).unwrap();
        (0..16)
            .map(|meta| BlockState::new(block, meta))
            .filter(|state| state.is_legal())
            .find(|state| {
                let stored = ActualBlockState::from_stored(*state);
                wanted
                    .iter()
                    .all(|(property, value)| stored.get(property) == Some(*value))
            })
            .unwrap_or_else(|| panic!("{name} stores no state with {wanted:?}"))
    }

    #[test]
    fn fence_connects_to_a_solid_block_and_to_another_fence() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:fence");
        // Alone, the post: the only fence model a raw-metadata lookup can ever name.
        assert_eq!(
            model_location(&world, pos),
            "fence#east=false,north=false,south=false,west=false"
        );

        fixture::place(&mut world, pos.east(), "minecraft:stone");
        assert_eq!(
            model_location(&world, pos),
            "fence#east=true,north=false,south=false,west=false"
        );

        fixture::place(&mut world, pos.north(), "minecraft:fence");
        assert_eq!(
            model_location(&world, pos),
            "fence#east=true,north=true,south=false,west=false"
        );
        assert_eq!(
            model_location(&world, pos.north()),
            "fence#east=false,north=false,south=true,west=false",
            "the neighbouring fence connects back"
        );

        // Negative control: java `BlockFence.canConnectTo` wants the same material, so a
        // nether-brick fence (rock) does not join an oak fence (wood).
        fixture::place(&mut world, pos.west(), "minecraft:nether_brick_fence");
        assert_eq!(
            model_location(&world, pos),
            "fence#east=true,north=true,south=false,west=false"
        );
    }

    #[test]
    fn stairs_select_their_inner_and_outer_corner_models() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        let east = stored_with("oak_stairs", &[("facing", "east"), ("half", "bottom")]);
        let north = stored_with("oak_stairs", &[("facing", "north"), ("half", "bottom")]);
        // Chunk storage resolves stair metadata to the last registered tuple, the outer-right
        // corner, so a renderer that skips `getActualState` draws every stair as that corner.
        assert_eq!(
            ActualBlockState::from_stored(east).get("shape"),
            Some("outer_right")
        );
        world.set_block(pos, east);
        assert_eq!(
            model_location(&world, pos),
            "oak_stairs#facing=east,half=bottom,shape=straight"
        );

        // java: BlockStairs.func_176307_f — a north-facing stair in front: outer corner.
        world.set_block(pos.east(), north);
        assert_eq!(
            model_location(&world, pos),
            "oak_stairs#facing=east,half=bottom,shape=outer_left"
        );

        // java: BlockStairs.func_176305_g — the same stair behind instead: inner corner.
        world.set_block(pos.east(), BlockState::AIR);
        world.set_block(pos.west(), north);
        assert_eq!(
            model_location(&world, pos),
            "oak_stairs#facing=east,half=bottom,shape=inner_left"
        );

        // Upside-down stairs swap left and right (`flag ? 1 : 2`).
        world.set_block(
            pos,
            stored_with("oak_stairs", &[("facing", "east"), ("half", "top")]),
        );
        world.set_block(
            pos.west(),
            stored_with("oak_stairs", &[("facing", "north"), ("half", "top")]),
        );
        assert_eq!(
            model_location(&world, pos),
            "oak_stairs#facing=east,half=top,shape=inner_right"
        );
    }

    #[test]
    fn a_pane_between_two_blocks_spans_them() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:glass_pane");
        // Alone, 1.8.9 draws the four-armed cross (`glass_pane_nsew`).
        assert_eq!(
            model_location(&world, pos),
            "glass_pane#east=false,north=false,south=false,west=false"
        );

        fixture::place(&mut world, pos.west(), "minecraft:stone");
        fixture::place(&mut world, pos.east(), "minecraft:glass");
        assert_eq!(
            model_location(&world, pos),
            "glass_pane#east=true,north=false,south=false,west=true"
        );

        // A stained pane takes its file from its colour and joins the plain pane beside it.
        world.set_block(
            pos.north(),
            stored_with("stained_glass_pane", &[("color", "red")]),
        );
        assert_eq!(
            model_location(&world, pos.north()),
            "red_stained_glass_pane#east=false,north=false,south=true,west=false"
        );
    }

    #[test]
    fn a_wall_takes_its_post_from_the_block_above() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:cobblestone_wall");
        fixture::place(&mut world, pos.north(), "minecraft:cobblestone_wall");
        fixture::place(&mut world, pos.south(), "minecraft:stone");
        // A straight run with air above has no post (`cobblestone_wall_ns`)...
        assert_eq!(
            model_location(&world, pos),
            "cobblestone_wall#east=false,north=true,south=true,up=false,west=false"
        );
        // ...and gets one under anything that is not air (`cobblestone_wall_ns_above`).
        fixture::place(&mut world, pos.up(), "minecraft:glass");
        assert_eq!(
            model_location(&world, pos),
            "cobblestone_wall#east=false,north=true,south=true,up=true,west=false"
        );

        // The mossy wall renames its file by variant.
        world.set_block(
            pos,
            stored_with("cobblestone_wall", &[("variant", "mossy_cobblestone")]),
        );
        assert_eq!(
            model_location(&world, pos),
            "mossy_cobblestone_wall#east=false,north=true,south=true,up=true,west=false"
        );
    }

    #[test]
    fn redstone_wire_turns_a_corner() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:redstone_wire");
        assert_eq!(
            model_location(&world, pos),
            "redstone_wire#east=none,north=none,south=none,west=none"
        );

        fixture::place(&mut world, pos.north(), "minecraft:redstone_wire");
        fixture::place(&mut world, pos.east(), "minecraft:redstone_wire");
        // `redstone_ne`. POWER only tints, so the mapper drops it from the key.
        assert_eq!(
            model_location(&world, pos),
            "redstone_wire#east=side,north=side,south=none,west=none"
        );
    }

    #[test]
    fn both_door_halves_share_facing_hinge_and_open() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        // java: BlockDoor stores FACING and OPEN in the lower half and HINGE and POWERED in the
        // upper half; `getActualState` copies the missing ones from the other half.
        let lower = stored_with(
            "wooden_door",
            &[("half", "lower"), ("facing", "north"), ("open", "true")],
        );
        let upper_right = stored_with(
            "wooden_door",
            &[("half", "upper"), ("hinge", "right"), ("powered", "false")],
        );
        world.set_block(pos, lower);
        world.set_block(pos.up(), upper_right);
        assert_eq!(
            model_location(&world, pos),
            "wooden_door#facing=north,half=lower,hinge=right,open=true"
        );
        assert_eq!(
            model_location(&world, pos.up()),
            "wooden_door#facing=north,half=upper,hinge=right,open=true"
        );

        let upper_left = stored_with(
            "wooden_door",
            &[("half", "upper"), ("hinge", "left"), ("powered", "false")],
        );
        world.set_block(pos.up(), upper_left);
        assert_eq!(
            model_location(&world, pos),
            "wooden_door#facing=north,half=lower,hinge=left,open=true"
        );
        assert_eq!(
            model_location(&world, pos.up()),
            "wooden_door#facing=north,half=upper,hinge=left,open=true"
        );
    }

    #[test]
    fn both_halves_of_double_tall_grass_select_the_grass_models() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        let lower = stored_with(
            "double_plant",
            &[("half", "lower"), ("variant", "double_grass")],
        );
        let upper = stored_with("double_plant", &[("half", "upper")]);
        // The upper half's metadata has no room for the variant; chunk storage resolves it to
        // the last registered tuple, the peony. That flower top is what a renderer that skips
        // `getActualState` puts on every double plant.
        assert_eq!(
            ActualBlockState::from_stored(upper).get("variant"),
            Some("paeonia")
        );
        world.set_block(pos, lower);
        world.set_block(pos.up(), upper);
        assert_eq!(model_location(&world, pos), "double_grass#half=lower");
        assert_eq!(model_location(&world, pos.up()), "double_grass#half=upper");

        // java: BlockDoublePlant.getActualState copies VARIANT from the lower half, whatever it is.
        world.set_block(
            pos,
            stored_with(
                "double_plant",
                &[("half", "lower"), ("variant", "sunflower")],
            ),
        );
        assert_eq!(model_location(&world, pos.up()), "sunflower#half=upper");
    }

    #[test]
    fn redstone_wire_joins_a_comparator_from_any_side() {
        // java: BlockRedstoneWire.canConnectTo — `Blocks.unpowered_repeater.isAssociated` matches
        // only the two repeater blocks, so a comparator falls through to
        // `canProvidePower() && side != null`, which BlockRedstoneDiode answers true for every
        // horizontal side (real 1.8.9.jar: `ajb.a(alz, cq)` tests `afi.bb`, the repeater).
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        fixture::place(&mut world, pos, "minecraft:redstone_wire");
        for comparator in ["unpowered_comparator", "powered_comparator"] {
            for facing in ["north", "east", "south", "west"] {
                world.set_block(pos.east(), stored_with(comparator, &[("facing", facing)]));
                assert_eq!(
                    actual_state(&world, world.block_at(pos), pos).get("east"),
                    Some("side"),
                    "{comparator} facing {facing}"
                );
            }
        }

        // Control: a repeater still joins only along its own axis.
        world.set_block(
            pos.east(),
            stored_with("unpowered_repeater", &[("facing", "north")]),
        );
        assert_eq!(
            actual_state(&world, world.block_at(pos), pos).get("east"),
            Some("none")
        );
        world.set_block(
            pos.east(),
            stored_with("unpowered_repeater", &[("facing", "east")]),
        );
        assert_eq!(
            actual_state(&world, world.block_at(pos), pos).get("east"),
            Some("side")
        );
    }

    #[test]
    fn a_comparator_never_locks_a_repeater_on_the_client() {
        // java: BlockRedstoneDiode.getPowerOnSide asks the side block's getStrongPower. A
        // comparator answers with its tile entity's output signal (BlockRedstoneComparator
        // .getActiveSignal), which 1.8.9 never sends to the client (S35 has no comparator
        // action), so it reads 0 and the repeater is not drawn locked.
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        let repeater = stored_with("unpowered_repeater", &[("facing", "north")]);
        world.set_block(pos, repeater);

        // Control: a powered repeater pointing into the side does lock it.
        world.set_block(
            pos.east(),
            stored_with("powered_repeater", &[("facing", "east")]),
        );
        assert_eq!(
            actual_state(&world, repeater, pos).get("locked"),
            Some("true")
        );

        world.set_block(
            pos.east(),
            stored_with("powered_comparator", &[("facing", "east")]),
        );
        assert_eq!(
            actual_state(&world, repeater, pos).get("locked"),
            Some("false")
        );
    }
}
