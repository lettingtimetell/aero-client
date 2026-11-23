//! The block-breaking overlay and the selection-box outline.
//!
//! Both draw at [`PhaseOrder::BLOCK_OVERLAY`] (350), between entities and particles, which is
//! where vanilla draws them.
//!
//! * **Selection box** — the black wireframe around the block under the crosshair. Vanilla
//!   uses the block's *render* bounds expanded by `0.002`, blended at 40 % alpha with depth
//!   writes off, which is why it shows through its own edges.
//! * **Breaking overlay** — `destroy_stage_0..9` drawn over the block with a
//!   `GL_DST_COLOR`/`GL_SRC_COLOR` blend and a polygon offset, so the cracks darken the block
//!   underneath instead of replacing it.
//!
//! # Documented divergence
//!
//! `glLineWidth(2.0)` has no portable wgpu equivalent; the selection outline is one pixel wide.
//!
//! # The breaking overlay re-bakes the real model (closes a §13 cross-section request)
//!
//! Vanilla re-bakes the broken block's *model* with the crack texture substituted
//! (`SimpleBakedModel.Builder(bakedModel, destroyStageSprite)`, each quad wrapped in a
//! `BreakingFour`), so cracks follow a stair or a fence exactly rather than a bounding box.
//! [`crack_vertices`] does the same: it walks
//! [`mc_assets::model::model_for_actual`]'s real quads and
//! remaps each vertex's UV onto the `destroy_stage_N` sprite that is now stitched into the same
//! atlas terrain samples (closing an earlier §13 row), so this phase no longer loads its own
//! `destroy_stage_0..9` strip texture at all.
//!
//! The natural-sounding design for the remap — store each quad's *original* atlas sprite rect
//! on [`mc_assets::model::BakedQuad`] and re-normalise its baked UV through it — turned out to
//! be a wrong premise, caught by reading `BreakingFour.remapVert` (java) rather than guessing:
//! vanilla does not touch the quad's original UV at all. It re-derives the crack UV purely from
//! each vertex's **position** projected onto the quad's own facing plane — exactly
//! `BreakingFour.remapVert`, ported here as
//! [`mc_assets::model::facebakery::breaking_vertex_uv`] — then interpolates that through the
//! *destination* sprite's rect. Reusing that existing, already-tested function is what this
//! phase does, so no new field or memory cost was needed on `BakedQuad` after all: a face whose
//! JSON gives it a non-default `"uv"` sub-rect (common on stairs, slabs, fences) would render
//! its crack differently under the two designs, and only the position-based one agrees with
//! vanilla.
//!
//! The live phase keeps a compact six-neighbour snapshot for each active break, so vanilla's
//! checked-sides damage path can run without retaining a borrow of the mutable client world.
//! The phase snapshots the states needed by the live caller; [`crack_vertices`] remains the
//! no-world/model-only helper used by asset tests, while
//! [`crack_vertices_with_neighbors`] is the checked-sides path. With no snapshot supplied, the
//! helper still draws every baked quad, preserving its standalone model-only contract.
//
// java: client/renderer/RenderGlobal.java drawSelectionBox / drawSelectionBoundingBox /
//       drawBlockDamageTexture / preRenderDamagedBlocks, client/renderer/DestroyBlockProgress.java,
//       client/resources/model/SimpleBakedModel.java (Builder), client/renderer/block/model/BreakingFour.java

use std::cell::RefCell;

use crate::pipeline::PassTarget;
use crate::{Camera, Gpu, RenderPhase};
use mc_assets::model::facebakery::{breaking_vertex_uv, facing_from_vertex_data};
use mc_assets::{Atlas, AtlasAnimationUpload, ResourceLocation};
use mc_data::block::ActualBlockState;
use mc_data::BlockState;
use mc_math::{Aabb, BlockPos, DVec3, Facing};
use mc_world::World;

/// How many crack textures vanilla ships. java: `RenderGlobal.destroyBlockIcons`.
pub const DESTROY_STAGES: u32 = 10;

/// Java's `TextureMap` uploads the block/destroy-stage atlas with `GL_RGBA`.
///
/// The crack pass re-uploads the same raw atlas bytes into a private texture, so it must retain
/// those normalized bytes when sampled rather than applying an implicit sRGB decode.
const JAVA_BLOCK_OVERLAY_ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One block being broken. java: `DestroyBlockProgress`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BreakProgress {
    /// The block position.
    pub pos: BlockPos,
    /// Damage stage `0..=9`. java: `getPartialBlockDamage`.
    pub stage: u8,
}

/// The outline around the block under the crosshair, in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectionBox {
    /// Minimum corner (already the block's render bounds offset to world space).
    pub min: DVec3,
    /// Maximum corner.
    pub max: DVec3,
}

impl SelectionBox {
    /// A state-local/preview selection box, expanded by `0.002` so the lines clear the block's
    /// own surface.  Live rendering must use [`Self::for_block_in_world`], because a door's
    /// selected bounds depend on the other half.
    /// java: `Block.getSelectedBoundingBox(...).expand(0.002)`.
    ///
    /// Nine blocks override `getSelectedBoundingBox`; eight of them are no-ops at the live call
    /// site, because `RenderGlobal.drawSelectionBox:1891` has already run
    /// `setBlockBoundsBasedOnState` and the override just re-derives the same box (`BlockBanner`,
    /// `BlockCocoa`, `BlockDoor`, `BlockLadder`, `BlockSign`, `BlockTrapDoor` literally so;
    /// `BlockCake` returns its collision box, which is the same numbers; `BlockStairs` forwards
    /// to a `modelBlock` that is always a full cube outside `collisionRayTrace`).
    /// `BlockCactus` is the one that is not — see below.
    #[must_use]
    pub fn for_block(pos: BlockPos, state: mc_data::BlockState) -> Self {
        const E: f64 = 0.002_000_000_094_994_902_6;
        Self::from_selected_bounds(static_selected_bounding_box(pos, state), E)
    }

    /// The live outline for a block in a world.  This is the `for_block` counterpart used by
    /// `RenderGlobal.drawSelectionBox`: `BlockDoor.setBlockBoundsBasedOnState` first combines
    /// facing/open from the lower cell with hinge from the upper cell, so a stored door state's
    /// registry bounds are not sufficient.
    #[must_use]
    pub fn for_block_in_world(pos: BlockPos, state: mc_data::BlockState, world: &World) -> Self {
        const E: f64 = 0.002_000_000_094_994_902_6;
        Self::from_selected_bounds(selected_bounding_box(world, pos, state), E)
    }

    fn from_selected_bounds(bounds: Aabb, expansion: f64) -> Self {
        // `Block.getSelectedBoundingBox` constructs a fresh Java `AxisAlignedBB` from the
        // block's six mutable bounds.  That constructor normalises each axis before
        // `RenderGlobal` expands it.  Keep that step here even though the live raytrace path
        // deliberately receives the raw corners: an unattached vine is the observable case
        // where `setBlockBoundsBasedOnState` leaves min > max, and the outline must still be a
        // full cube rather than an inside-out wireframe.
        let bounds = Aabb::new(bounds.min, bounds.max).expand(expansion, expansion, expansion);
        Self {
            min: bounds.min,
            max: bounds.max,
        }
    }

    /// The twelve edges as twenty-four line endpoints, camera-relative.
    #[must_use]
    pub fn line_vertices(&self, cam: &Camera) -> Vec<OverlayVertex> {
        let lo = cam.relative(self.min);
        let hi = cam.relative(self.max);
        let c = |x: f32, y: f32, z: f32| OverlayVertex {
            pos: [x, y, z],
            uv: [0.0, 0.0],
        };
        let corners = [
            (lo.x, lo.y, lo.z),
            (hi.x, lo.y, lo.z),
            (hi.x, lo.y, hi.z),
            (lo.x, lo.y, hi.z),
            (lo.x, hi.y, lo.z),
            (hi.x, hi.y, lo.z),
            (hi.x, hi.y, hi.z),
            (lo.x, hi.y, hi.z),
        ];
        // Bottom loop, top loop, four uprights.
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        let mut out = Vec::with_capacity(24);
        for (a, b) in EDGES {
            let (x, y, z) = corners[a];
            out.push(c(x, y, z));
            let (x, y, z) = corners[b];
            out.push(c(x, y, z));
        }
        out
    }
}

/// The world-space box left in `Block.minX..maxZ` by
/// `Block.setBlockBoundsBasedOnState`, before `getSelectedBoundingBox` expands it by 0.002.
///
/// Doors and fences are the relevant neighbour-dependent variants here.  The collision system
/// already transcribes both the door metadata combination and `BlockFence.canConnectTo`, so use
/// those oracles instead of duplicating subtly different state rules in the renderer.  Walls and
/// panes have *selection* bounds which deliberately differ from collision, so they are resolved
/// through the shared `mc-world` world-aware oracles; chests and vines are resolved below. Every
/// other block is resolved from its state-local render bounds (with cactus's selected-box override
/// below).
// java: BlockDoor.java:74-116, 295-308; BlockFence.java:127-159; BlockWall.java:67-112;
//       BlockPane.java:129-173; BlockChest.java:66-87; BlockVine.java:81-146; Block.java:679
#[must_use]
pub fn selected_bounding_box(world: &World, pos: BlockPos, state: mc_data::BlockState) -> Aabb {
    let class = mc_data::block_info(state.block()).java_class;
    if class.ends_with("BlockDoor") {
        // `collision_boxes_at` applies the same combined lower/upper state as the door's
        // selected bounds.  This deliberately tiny mask is guaranteed to intersect any one
        // block-local door slab, even at extreme coordinates, without accidentally querying a
        // neighbouring block.
        let x = f64::from(pos.x);
        let y = f64::from(pos.y);
        let z = f64::from(pos.z);
        let mask = Aabb::from_bounds(x - 1.0, y - 1.0, z - 1.0, x + 2.0, y + 2.0, z + 2.0);
        if let Some(bounds) = mc_physics::collision_boxes_at(world, pos, &mask)
            .into_iter()
            .next()
        {
            return bounds;
        }
    }
    if let Some(bounds) = mc_world::collision::fence_selection_bounds_at(world, pos) {
        return bounds;
    }
    match class.rsplit('.').next().unwrap_or(class) {
        "BlockWall" => mc_world::collision::wall_selection_bounds_at(world, pos)
            .unwrap_or_else(|| static_selected_bounding_box(pos, state)),
        "BlockPane" | "BlockStainedGlassPane" => {
            // `BlockPane.setBlockBoundsBasedOnState` reads the four live neighbours.  Keep this
            // consumer on the shared world oracle; the outline expands and normalises the
            // returned Java bounds in `from_selected_bounds` below.
            mc_world::collision::pane_selection_bounds_at(world, pos)
                .unwrap_or_else(|| static_selected_bounding_box(pos, state))
        }
        "BlockChest" => mc_world::collision::chest_selection_bounds_at(world, pos)
            .unwrap_or_else(|| static_selected_bounding_box(pos, state)),
        "BlockVine" => mc_world::collision::vine_selection_bounds_at(world, pos)
            .unwrap_or_else(|| static_selected_bounding_box(pos, state)),
        _ => static_selected_bounding_box(pos, state),
    }
}

#[cfg(test)]
fn bounds_at(pos: BlockPos, min: (f64, f64, f64), max: (f64, f64, f64)) -> Aabb {
    let base = pos.to_vec3();
    // `Block.setBlockBounds` stores these six fields verbatim.  In particular,
    // `BlockVine.setBlockBoundsBasedOnState` leaves a faceless, unattached vine at
    // `min = 1, max = 0`; `getSelectedBoundingBox` passes those same inverted corners to
    // `new AxisAlignedBB`, whose six-double constructor sorts them.  This helper intentionally
    // keeps the raw fields so source-oracle tests can distinguish them from the selected outline.
    Aabb {
        min: base + DVec3::new(min.0, min.1, min.2),
        max: base + DVec3::new(max.0, max.1, max.2),
    }
}

/// State-local selected bounds for preview/test callers which do not have a world.  Live code
/// must use [`selected_bounding_box`] so doors see their other half.
fn static_selected_bounding_box(pos: BlockPos, state: mc_data::BlockState) -> Aabb {
    let base = DVec3::new(f64::from(pos.x), f64::from(pos.y), f64::from(pos.z));

    // java: `BlockCactus.getSelectedBoundingBox` (`BlockCactus.java:69-73`) —
    // `(x + 0.0625, y, z + 0.0625) -> (x + 1 - 0.0625, y + 1, z + 1 - 0.0625)`: inset 1/16 in
    // X and Z but **full height** in Y. `BlockCactus` never overrides
    // `setBlockBoundsBasedOnState`, so its `blocks.json` bounds are the constructor's plain
    // `0..1`.  The corner arithmetic is a `float` add on a `float` coordinate in Java.
    if mc_data::block_info(state.block())
        .java_class
        .ends_with("BlockCactus")
    {
        const F: f32 = 0.0625;
        return Aabb::new(
            DVec3::new(
                f64::from(pos.x as f32 + F),
                f64::from(pos.y),
                f64::from(pos.z as f32 + F),
            ),
            DVec3::new(
                f64::from((pos.x + 1) as f32 - F),
                f64::from(pos.y + 1),
                f64::from((pos.z + 1) as f32 - F),
            ),
        );
    }

    // java: `BlockTorch.collisionRayTrace` sets the torch's bounds just
    // before `getMouseOver`'s pick, and `getSelectedBoundingBox` reads them back for the outline.
    // The registry row is the constructor's full cube, so the outline was a whole block (item 17).
    if let Some(torch) = mc_world::collision::torch_bounds(state) {
        return Aabb::new(base + torch.min, base + torch.max);
    }

    let b = mc_data::block_props(state).bounds;
    Aabb::new(
        base + DVec3::new(b.min_x, b.min_y, b.min_z),
        base + DVec3::new(b.max_x, b.max_y, b.max_z),
    )
}

/// Position + UV, the overlay phase's vertex.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayVertex {
    /// Camera-relative position.
    pub pos: [f32; 3],
    /// UV into the crack strip (unused by the outline).
    pub uv: [f32; 2],
}

impl OverlayVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Build the crack quads for one broken block, camera-relative, by re-baking the block's real
/// [`BakedModel`](mc_assets::model::BakedModel) with the `destroy_stage_N` sprite substituted
/// for every quad's texture. java: `SimpleBakedModel.Builder(bakedModel, destroyStageSprite)` /
/// `BreakingFour`; see the module docs for why the UV remap is position-based rather than a
/// re-normalisation through each quad's original sprite rect.
///
/// Total and non-panicking even before any bake has been installed —
/// [`mc_assets::model::model_for_actual`] degrades to the empty model and this returns an
/// empty `Vec`, the same "no jar, no panic"
/// convention every other consumer of it in this crate follows (see [`super`]'s module docs).
#[must_use]
pub fn crack_vertices(
    progress: BreakProgress,
    state: &ActualBlockState,
    cam: &Camera,
) -> Vec<OverlayVertex> {
    // java: `BlockRendererDispatcher.renderBlockDamage` calls `getActualState`, then
    // `BlockModelShapes.getModelForState` directly. It does not choose a position-weighted
    // alternative as ordinary terrain rendering does.
    let model = mc_assets::model::model_for_actual(state);
    crack_vertices_from_model(progress, model, cam, None)
}

/// Build the damage overlay with vanilla's checked-sides face culling.
///
/// `neighbors` is indexed by [`Facing::VALUES`]. The render phase uses a snapshot rather than a
/// borrowed world because `RenderPhase::record` can outlive the client-world borrow. Keeping this
/// helper separate also leaves [`crack_vertices`] useful for model-only asset tests and callers
/// which intentionally have no world context.
#[must_use]
pub fn crack_vertices_with_neighbors(
    progress: BreakProgress,
    state: &ActualBlockState,
    neighbors: &[BlockState; 6],
    cam: &Camera,
) -> Vec<OverlayVertex> {
    let model = mc_assets::model::model_for_actual(state);
    let visible_faces = checked_face_visibility(progress.pos, state.stored(), neighbors);
    crack_vertices_from_model(progress, model, cam, Some(&visible_faces))
}

/// Build the damage overlay with checked-side culling under the caller's live mesh settings.
///
/// Java's `BlockLeavesBase.shouldSideBeRendered` follows the graphics level installed by
/// `RenderGlobal.loadRenderers`; fast graphics therefore culls adjacent leaf faces while fancy
/// graphics keeps them visible. The model-only public helper retains vanilla's default settings,
/// while the live phase uses this variant with the client's current settings.
#[must_use]
pub fn crack_vertices_with_neighbors_settings(
    progress: BreakProgress,
    state: &ActualBlockState,
    neighbors: &[BlockState; 6],
    cam: &Camera,
    settings: crate::terrain::cull_face::MeshSettings,
) -> Vec<OverlayVertex> {
    let model = mc_assets::model::model_for_actual(state);
    let visible_faces =
        checked_face_visibility_with_settings(progress.pos, state.stored(), neighbors, settings);
    crack_vertices_from_model(progress, model, cam, Some(&visible_faces))
}

/// Apply the ordinary terrain checked-sides oracle to a six-state snapshot.
fn checked_face_visibility(
    pos: BlockPos,
    state: BlockState,
    neighbors: &[BlockState; 6],
) -> [bool; 6] {
    checked_face_visibility_with_settings(
        pos,
        state,
        neighbors,
        crate::terrain::cull_face::MeshSettings::default(),
    )
}

/// Apply the ordinary terrain checked-sides oracle with explicit graphics settings.
fn checked_face_visibility_with_settings(
    pos: BlockPos,
    state: BlockState,
    neighbors: &[BlockState; 6],
    settings: crate::terrain::cull_face::MeshSettings,
) -> [bool; 6] {
    let mut world = World::new();
    // `World::set_block` intentionally ignores unloaded columns (matching the client's
    // EmptyChunk provider), so make the tiny oracle world resident before installing the
    // snapshot. Without this, every neighbour reads as air and even an opaque cube appears to
    // have all six damage faces visible.
    world.load_empty_chunk(pos.chunk_pos());
    for facing in Facing::VALUES {
        world.load_empty_chunk(pos.relative(facing).chunk_pos());
    }
    world.set_block(pos, state);
    for (facing, neighbor) in Facing::VALUES.into_iter().zip(neighbors.iter().copied()) {
        world.set_block(pos.relative(facing), neighbor);
    }
    Facing::VALUES.map(|facing| {
        crate::terrain::cull_face::should_side_be_rendered(&world, state, pos, facing, settings)
    })
}

/// One damaged block's checked-sides inputs and answer, captured by
/// [`BlockOverlayPhase::update_breaking_neighbors`] while the live world is borrowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BreakingSides {
    pos: BlockPos,
    /// The stored state `visible` was computed for (`ActualBlockState::stored`).
    stored: BlockState,
    settings: crate::terrain::cull_face::MeshSettings,
    /// Six-neighbour snapshot indexed by [`Facing::VALUES`]; only read by the oracle fallback.
    neighbors: [BlockState; 6],
    /// `should_side_be_rendered` per [`Facing::VALUES`].
    visible: [bool; 6],
}

/// [`checked_face_visibility_with_settings`] answered from the live world (BOVL-1).
///
/// The oracle builds a throwaway `World` holding exactly seven blocks — `stored` at `pos` and
/// the six neighbour snapshots — with air everywhere else, then asks
/// `cull_face::should_side_be_rendered`. That predicate reads block states only (no light, no
/// AO, no tile entities or border), and for every class but one it reads nothing outside
/// those seven cells:
///
/// * `world.block_at(pos.relative(side))` (the neighbour) and, in `slab`, `world.block_at(pos)`;
/// * `base` -> `selected_bounding_box(world, pos, state)`, whose world reads are `pos` itself
///   plus: the door halves `pos.up()`/`pos.down()` (`door_combined_meta`), the four horizontal
///   neighbours for fences, walls, panes and chests, and `pos.up()` for vines.
///
/// The exception is `BlockPortal`, whose edge test reads cells two away
/// (`neighbour.west().west()` etc.), which the oracle sees as air. Those keep the oracle so the
/// output stays identical (the live answer would be closer to vanilla, which reads the real
/// world — a visible change, so not taken here).
///
/// So when the live world agrees with the snapshot on those seven cells — the neighbours are
/// read from this same world in the same call, and `block_at(pos) == stored` is checked — the
/// two answers are identical. Cost: six predicate calls instead of `World::new()` (an ~80-node
/// GenLayer graph), seven column loads (~1,000 dirty-set inserts) and seven lit `set_block`s.
fn live_checked_face_visibility(
    world: &World,
    pos: BlockPos,
    stored: BlockState,
    neighbors: &[BlockState; 6],
    settings: crate::terrain::cull_face::MeshSettings,
) -> [bool; 6] {
    if world.block_at(pos) != stored || crate::terrain::cull_face::class_of(stored) == "BlockPortal"
    {
        return checked_face_visibility_with_settings(pos, stored, neighbors, settings);
    }
    Facing::VALUES.map(|facing| {
        crate::terrain::cull_face::should_side_be_rendered(world, stored, pos, facing, settings)
    })
}

/// The common BreakingFour projection and triangle emission, with an optional checked-sides mask.
fn crack_vertices_from_model(
    progress: BreakProgress,
    model: &mc_assets::model::BakedModel,
    cam: &Camera,
    visible_faces: Option<&[bool; 6]>,
) -> Vec<OverlayVertex> {
    let stage = progress.stage.min((DESTROY_STAGES - 1) as u8);
    let dest_loc = ResourceLocation::minecraft(format!("textures/blocks/destroy_stage_{stage}"));
    let dest = mc_assets::model::sprite(&dest_loc).uv;

    let base = DVec3::new(
        f64::from(progress.pos.x),
        f64::from(progress.pos.y),
        f64::from(progress.pos.z),
    );

    let mut out = Vec::new();
    let mut emit_quad = |quad: &mc_assets::BakedQuad| {
        // java: `BreakingFour`'s constructor recomputes the facing fresh from the quad's own
        // vertex positions (`FaceBakery.getFacingFromVertexData`) rather than trusting a stored
        // one; `Facing::Up` is vanilla's own fallback for the rare quad whose vertices are not
        // axis-aligned.
        let facing = facing_from_vertex_data(&quad.verts).unwrap_or(Facing::Up);
        let mut verts = [OverlayVertex::default(); 4];
        for (i, v) in quad.verts.iter().enumerate() {
            let world = base
                + DVec3::new(
                    f64::from(v.pos[0]),
                    f64::from(v.pos[1]),
                    f64::from(v.pos[2]),
                );
            let rel = cam.relative(world);
            let (u16, v16) = breaking_vertex_uv(v.pos, facing);
            verts[i] = OverlayVertex {
                pos: [rel.x, rel.y, rel.z],
                uv: [dest.interpolate_u(u16), dest.interpolate_v(v16)],
            };
        }
        // Two triangles per quad (0,1,2)/(0,2,3), so the phase can draw without an index
        // buffer — the same fan the terrain mesher's own upload uses.
        out.push(verts[0]);
        out.push(verts[1]);
        out.push(verts[2]);
        out.push(verts[0]);
        out.push(verts[2]);
        out.push(verts[3]);
    };
    // java: the `generalQuads` overload is never checked against a neighbour.
    for quad in &model.general_quads {
        emit_quad(quad);
    }
    // java: `renderModel` walks each checked face list only when
    // `Block.shouldSideBeRendered(world, pos.offset(face), face)` returns true.
    for (index, quads) in model.face_quads.iter().enumerate() {
        if visible_faces.is_none_or(|visible| visible[index]) {
            for quad in quads {
                emit_quad(quad);
            }
        }
    }
    out
}

/// Build the temporary RGBA8 upload array used by Java's `TextureUtil.uploadTextureMipmap`.
///
/// `TextureUtil.anaglyphColor` mixes only RGB with integer arithmetic and carries alpha through
/// unchanged.  The source atlas levels remain raw; only this upload-owned copy is transformed so
/// a later setting toggle can replay the exact bytes without converting an already-converted
/// texture a second time.
fn overlay_upload_rgba(source_rgba: &[u8], anaglyph: bool) -> Vec<u8> {
    mc_assets::sprite::anaglyph_rgba(source_rgba, anaglyph)
}

/// Dimensions of one resident atlas mip level.
fn atlas_mip_dimensions(width: u32, height: u32, level: u32) -> (u32, u32) {
    (
        width.checked_shr(level).unwrap_or(0),
        height.checked_shr(level).unwrap_or(0),
    )
}

fn rgba_len(width: u32, height: u32) -> Option<usize> {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
}

fn valid_source_levels(levels: &[Vec<u8>], width: u32, height: u32) -> bool {
    !levels.is_empty()
        && levels.iter().enumerate().all(|(level, rgba)| {
            let (level_width, level_height) = atlas_mip_dimensions(width, height, level as u32);
            rgba_len(level_width, level_height).is_some_and(|expected| rgba.len() == expected)
        })
}

/// Validate one dirty rectangle before handing it to wgpu. Animation records are produced from
/// pack data, so malformed coordinates or payload lengths must be rejected before a device
/// validation error can occur.
fn atlas_animation_rect_valid(
    upload: &AtlasAnimationUpload,
    atlas_width: u32,
    atlas_height: u32,
    mip_level_count: u32,
) -> bool {
    if upload.mip_level >= mip_level_count
        || upload.width == 0
        || upload.height == 0
        || rgba_len(upload.width, upload.height)
            .is_none_or(|expected| upload.rgba.len() != expected)
    {
        return false;
    }
    let (level_width, level_height) =
        atlas_mip_dimensions(atlas_width, atlas_height, upload.mip_level);
    level_width > 0
        && level_height > 0
        && upload
            .x
            .checked_add(upload.width)
            .is_some_and(|end| end <= level_width)
        && upload
            .y
            .checked_add(upload.height)
            .is_some_and(|end| end <= level_height)
}

/// Upload one complete atlas mip level. The caller retains the raw source bytes separately.
fn upload_crack_atlas_level(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    level: u32,
    rgba: &[u8],
    width: u32,
    height: u32,
    anaglyph: bool,
) {
    assert_eq!(
        rgba_len(width, height),
        Some(rgba.len()),
        "invalid block overlay atlas mip {level}"
    );
    let upload_rgba = overlay_upload_rgba(rgba, anaglyph);
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

/// Upload one animated atlas dirty rectangle, applying the transform only to its temporary copy.
fn upload_crack_atlas_rect(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    upload: &AtlasAnimationUpload,
    anaglyph: bool,
) {
    let upload_rgba = overlay_upload_rgba(&upload.rgba, anaglyph);
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

/// The crack pass's atlas sampler: nearest filtering, **base level only**.
///
/// java: `EntityRenderer.renderWorldPass:1434` wraps `drawBlockDamageTexture` in
/// `setBlurMipmap(false, false)`, i.e. `GL_NEAREST` for both filters, so this pass never reads
/// the atlas's mip levels. Those levels are not crack-shaped: every `destroy_stage_N` sprite is
/// white at alpha 1/255 outside its crack, none of its first texels is fully transparent, so
/// `TextureUtil.generateMipmapData` takes the plain average and every 2x2 block touching a crack
/// texel becomes a bright texel at about half alpha, which passes the 0.1 alpha test and the
/// `DST_COLOR, SRC_COLOR` blend doubles towards white. A face seen nearly edge-on samples those
/// levels, which is the white crack of the 2026-11-06 report.
fn crack_sampler_descriptor() -> wgpu::SamplerDescriptor<'static> {
    wgpu::SamplerDescriptor {
        label: Some("block overlay atlas"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        lod_max_clamp: 0.0,
        ..Default::default()
    }
}

/// java: `RenderGlobal.preRenderDamagedBlocks:1802`, `doPolygonOffset(-3.0F, -3.0F)`, i.e.
/// `glPolygonOffset(factor = -3, units = -3)`; the same pair as `RenderTNTPrimed` and the world
/// border. The previous `-1` units / `-10` factor pulled a face seen edge-on more than three
/// times as far towards the camera as vanilla.
const CRACK_DEPTH_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: -3,
    slope_scale: -3.0,
    clamp: 0.0,
};

/// java: `EntityRenderer.renderWorldPass:1328` enables `GL_CULL_FACE` (`Minecraft.startGame`
/// selected `cullFace(GL_BACK)`) before the damage pass, so a broken block's faces turned away
/// from the camera get no crack. The crack quads keep `FaceBakery`'s counter-clockwise winding
/// (they are the baked quads' own vertices in the terrain mesher's 0-1-2 / 0-2-3 fan).
fn crack_primitive_state() -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        cull_mode: Some(wgpu::Face::Back),
        ..wgpu::PrimitiveState::default()
    }
}

/// The `BLOCK_OVERLAY` phase: selection outline plus every block currently being broken.
pub struct BlockOverlayPhase {
    line_pipeline: wgpu::RenderPipeline,
    crack_pipeline: Option<wgpu::RenderPipeline>,
    crack_bind_group: Option<wgpu::BindGroup>,
    crack_texture: Option<wgpu::Texture>,
    /// Raw base and resident mip bytes used to replay the GPU texture on anaglyph toggles.
    /// Shared by the owners of the shared atlas.
    crack_source_levels: Option<std::rc::Rc<RefCell<Vec<Vec<u8>>>>>,
    crack_atlas_width: u32,
    crack_atlas_height: u32,
    crack_atlas_mip_level_count: u32,
    /// Java's global `GameSettings.anaglyph`, applied at each atlas upload boundary.
    anaglyph: bool,
    vertex_buffer: wgpu::Buffer,
    capacity: usize,
    /// The block under the crosshair, or `None`. Set this from the ray-trace result each tick.
    pub selection: Option<SelectionBox>,
    /// The Block Overlay mod's fill and wide outline for [`Self::selection`].
    /// `None` draws vanilla's one-pixel black line exactly as before.
    pub selection_style: Option<super::styled_outline::SelectionStyle>,
    /// The styled pipelines, built the first time a style is drawn.
    styled: Option<super::styled_outline::StyledOutline>,
    /// Blocks currently being broken, after Java's world-aware `getActualState` pass.
    pub breaking: Vec<(BreakProgress, ActualBlockState)>,
    /// Checked-sides masks (plus their six-neighbour snapshots) for [`Self::breaking`].
    ///
    /// `RenderPhase::record` cannot retain a borrow of the client world, so the client resolves
    /// the checked-sides damage mask against the live world immediately before rendering.
    /// Missing entries deliberately preserve the historical model-only behavior.
    breaking_neighbors: Vec<BreakingSides>,
    /// Current terrain culling settings, including Java's fancy/fast leaf seam behavior.
    breaking_mesh_settings: crate::terrain::cull_face::MeshSettings,
}

impl BlockOverlayPhase {
    /// Build the phase. `atlas` is S09's stitched block/item atlas (`GameAssets::atlas`); pass
    /// `None` to get the selection outline only, with the breaking overlay disabled.
    ///
    /// The crack texture used to be loaded here into a dedicated, private ten-frame strip
    /// (`textures/blocks/destroy_stage_0..9.png`, independent of S09's atlas). That closed
    /// §13's `S09 <-> S11/S23` cross-section request: `bake_all` now stitches
    /// `destroy_stage_0..9` into the same atlas terrain samples
    /// (`mc_assets::model::BUILTIN_TEXTURES`), and [`crack_vertices`] re-bakes the block's real
    /// model against it (see the module docs), so this phase reuses the atlas bytes the caller
    /// already has rather than issuing its own resource-manager loads — the same "re-upload the
    /// CPU-side atlas into a private GPU texture" pattern `mc-render`'s particle phase already
    /// uses for its own block-atlas needs (`ParticlePhase::install_block_atlas`), not a new one.
    #[must_use]
    pub fn new(gpu: &Gpu, atlas: Option<&Atlas>) -> Self {
        Self::new_with_anaglyph(gpu, atlas, false)
    }

    /// Build the phase with Java's global upload-time anaglyph switch selected.
    ///
    /// The compatibility [`Self::new`] constructor keeps the transform disabled. A caller
    /// which owns `GameSettings` can seed the phase here, or call [`Self::set_anaglyph`] before
    /// its first atlas upload.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, atlas: Option<&Atlas>, anaglyph: bool) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("block overlay"),
                source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
            });
        let line_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("overlay line layout"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let line_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("overlay lines"),
                layout: Some(&line_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_line"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(OverlayVertex::layout())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::DEPTH_FORMAT,
                    // java: GlStateManager.depthMask(false) around the outline.
                    depth_write_enabled: Some(false),
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
                    entry_point: Some("fs_line"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let (
            crack_pipeline,
            crack_bind_group,
            crack_texture,
            crack_source_levels,
            crack_atlas_mip_level_count,
        ) = match atlas {
            None => (None, None, None, None, 0),
            Some(atlas) => {
                // Re-upload the same CPU-side atlas bytes `TerrainRenderer`/`ParticlePhase`
                // already uploaded into their own textures — this crate has no shared-GPU-atlas
                // cache (each consumer independently owns its texture), so this follows that
                // existing precedent rather than inventing a new sharing mechanism.
                let mut source_levels = Vec::with_capacity(1 + atlas.mipmaps.len());
                source_levels.push(atlas.texture.clone());
                source_levels.extend(atlas.mipmaps.iter().cloned());
                assert!(
                    valid_source_levels(&source_levels, atlas.w, atlas.h),
                    "invalid block overlay atlas source levels"
                );
                // Inside the client's install sweep this adopts the one
                // shared GPU block atlas (`atlas_transfer::AtlasShareScope`); otherwise a plain
                // private texture as before.
                let texture = crate::atlas_transfer::share_atlas_texture(
                    gpu,
                    &wgpu::TextureDescriptor {
                        label: Some("block overlay atlas"),
                        size: wgpu::Extent3d {
                            width: atlas.w,
                            height: atlas.h,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: source_levels.len() as u32,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: JAVA_BLOCK_OVERLAY_ATLAS_FORMAT,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    &source_levels,
                    anaglyph,
                );
                for (level, rgba) in source_levels.iter().enumerate() {
                    let level = level as u32;
                    let (width, height) = atlas_mip_dimensions(atlas.w, atlas.h, level);
                    upload_crack_atlas_level(gpu, &texture, level, rgba, width, height, anaglyph);
                }
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let sampler = gpu.device.create_sampler(&crack_sampler_descriptor());
                let bgl = gpu
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("block overlay atlas"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
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
                    label: Some("block overlay atlas"),
                    layout: &bgl,
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
                let layout = gpu
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("overlay crack layout"),
                        bind_group_layouts: &[Some(gpu.camera_layout()), Some(&bgl)],
                        immediate_size: 0,
                    });
                let pipeline = gpu
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("overlay cracks"),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs_crack"),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            buffers: &[Some(OverlayVertex::layout())],
                        },
                        primitive: crack_primitive_state(),
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: crate::DEPTH_FORMAT,
                            depth_write_enabled: Some(false),
                            depth_compare: Some(wgpu::CompareFunction::LessEqual),
                            stencil: wgpu::StencilState::default(),
                            bias: CRACK_DEPTH_BIAS,
                        }),
                        multisample: wgpu::MultisampleState {
                            count: gpu.sample_count(),
                            ..Default::default()
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some("fs_crack"),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: gpu.config.format,
                                // java: tryBlendFuncSeparate(GL_DST_COLOR, GL_SRC_COLOR, 1, 0)
                                blend: Some(wgpu::BlendState {
                                    color: wgpu::BlendComponent {
                                        src_factor: wgpu::BlendFactor::Dst,
                                        dst_factor: wgpu::BlendFactor::Src,
                                        operation: wgpu::BlendOperation::Add,
                                    },
                                    alpha: wgpu::BlendComponent::OVER,
                                }),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    });
                // One raw shadow for every owner of the shared atlas.
                let source_levels =
                    crate::atlas_transfer::share_atlas_shadow(&texture, source_levels);
                (
                    Some(pipeline),
                    Some(bind_group),
                    Some(texture),
                    Some(source_levels),
                    (1 + atlas.mipmaps.len()) as u32,
                )
            }
        };

        let capacity = 1024;
        let vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block overlay"),
            size: (capacity * std::mem::size_of::<OverlayVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            line_pipeline,
            crack_pipeline,
            crack_bind_group,
            crack_texture,
            crack_source_levels,
            crack_atlas_width: atlas.map_or(0, |atlas| atlas.w),
            crack_atlas_height: atlas.map_or(0, |atlas| atlas.h),
            crack_atlas_mip_level_count,
            anaglyph,
            vertex_buffer,
            capacity,
            selection: None,
            selection_style: None,
            styled: None,
            breaking: Vec::new(),
            breaking_neighbors: Vec::new(),
            breaking_mesh_settings: crate::terrain::cull_face::MeshSettings::default(),
        }
    }

    /// Whether the atlas was supplied and the breaking overlay will draw.
    #[must_use]
    pub fn cracks_available(&self) -> bool {
        self.crack_pipeline.is_some()
    }

    /// Set the live terrain culling settings used by the next damage-overlay snapshot.
    ///
    /// Java stores the equivalent graphics level on the leaf block instances when renderers
    /// reload; this phase receives the setting explicitly because its renderer is independent of
    /// the terrain mesher.
    pub fn set_mesh_settings(&mut self, settings: crate::terrain::cull_face::MeshSettings) {
        self.breaking_mesh_settings = settings;
    }

    /// Refresh the neighbour snapshots used by the checked-sides damage overlay.
    ///
    /// Call this after assigning [`Self::breaking`] and before the phase records its pass. The
    /// snapshot is intentionally keyed by position: Java's damage map can contain multiple
    /// breakers at different positions, while the baked model/cull decision is position-local.
    pub fn update_breaking_neighbors(&mut self, world: &World) {
        mc_core::profile_scope!("render.block_overlay.breaking_sides");
        self.breaking_neighbors.clear();
        self.breaking_neighbors.reserve(self.breaking.len());
        let settings = self.breaking_mesh_settings;
        for (progress, state) in &self.breaking {
            let pos = progress.pos;
            let stored = state.stored();
            let neighbors = Facing::VALUES.map(|facing| world.block_at(pos.relative(facing)));
            // BOVL-1: resolve the mask here, against the live world, instead of rebuilding and
            // lighting an oracle `World` per crack per frame in `record`.
            let visible = live_checked_face_visibility(world, pos, stored, &neighbors, settings);
            self.breaking_neighbors.push(BreakingSides {
                pos,
                stored,
                settings,
                neighbors,
                visible,
            });
        }
    }

    /// Apply one S09 animated-atlas dirty rectangle to the crack overlay's private atlas copy.
    /// Resident mip levels are accepted at their own dimensions, and the retained source is
    /// updated before the upload so a later anaglyph replay includes the animated frame.
    pub fn upload_atlas_animation(&self, gpu: &Gpu, upload: &AtlasAnimationUpload) -> bool {
        self.upload_atlas_animation_with_sink(upload, |texture, anaglyph| {
            upload_crack_atlas_rect(gpu, texture, upload, anaglyph);
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
        let (Some(texture), Some(source_levels)) = (&self.crack_texture, &self.crack_source_levels)
        else {
            return false;
        };
        if !atlas_animation_rect_valid(
            upload,
            self.crack_atlas_width,
            self.crack_atlas_height,
            self.crack_atlas_mip_level_count,
        ) {
            return false;
        }
        let (level_width, _) = atlas_mip_dimensions(
            self.crack_atlas_width,
            self.crack_atlas_height,
            upload.mip_level,
        );
        {
            let mut source_levels = source_levels.borrow_mut();
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
        sink(texture, self.anaglyph);
        true
    }

    /// Whether this phase applies Java's global upload-time anaglyph transform.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global texture-upload transform and replay every resident atlas level from
    /// retained raw bytes. Replaying raw pixels avoids double conversion and includes the latest
    /// animated frame in every mip level.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;
        let Some(source_levels) = &self.crack_source_levels else {
            return;
        };
        let source_levels = source_levels.borrow().clone();
        let Some(texture) = &self.crack_texture else {
            return;
        };
        for (level, rgba) in source_levels.iter().enumerate() {
            let level = level as u32;
            let (width, height) =
                atlas_mip_dimensions(self.crack_atlas_width, self.crack_atlas_height, level);
            upload_crack_atlas_level(gpu, texture, level, rgba, width, height, enabled);
        }
    }

    /// Set the transform for subsequent atlas uploads without touching resident GPU bytes.
    /// Use [`Self::set_anaglyph_with_gpu`] when an installed phase must change immediately.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }
}

impl RenderPhase for BlockOverlayPhase {
    fn name(&self) -> &'static str {
        "terrain/block-overlay"
    }

    fn enabled(&self) -> bool {
        self.selection.is_some() || !self.breaking.is_empty()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.record_to(gpu, PassTarget::Own { enc, view }, cam);
    }

    // Uploads, then one pass: it joins the world pass.
    fn shares_world_pass(&self) -> bool {
        true
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, cam: &Camera) {
        self.record_to(gpu, PassTarget::Shared(pass), cam);
    }
}

impl BlockOverlayPhase {
    /// The phase's work, drawing into a pass of its own or the shared
    /// world pass.
    fn record_to(&mut self, gpu: &Gpu, target: PassTarget<'_>, cam: &Camera) {
        mc_core::profile_scope!("render.block_overlay.record");
        // With the Block Overlay mod on, the selection goes through the styled
        // fill/outline pipelines below instead of the vanilla line list.
        let styled = self.selection.zip(self.selection_style);
        let mut lines = match &self.selection {
            Some(sel) if styled.is_none() => sel.line_vertices(cam),
            _ => Vec::new(),
        };
        let line_count = lines.len();

        let mut cracks: Vec<OverlayVertex> = Vec::new();
        if self.crack_pipeline.is_some() {
            for (progress, state) in &self.breaking {
                if let Some(sides) = self
                    .breaking_neighbors
                    .iter()
                    .find(|sides| sides.pos == progress.pos)
                {
                    if sides.stored == state.stored()
                        && sides.settings == self.breaking_mesh_settings
                    {
                        cracks.extend(crack_vertices_from_model(
                            *progress,
                            mc_assets::model::model_for_actual(state),
                            cam,
                            Some(&sides.visible),
                        ));
                    } else {
                        // `breaking` or the settings changed after the snapshot: the old
                        // per-frame oracle over the stored neighbours, exactly as before.
                        cracks.extend(crack_vertices_with_neighbors_settings(
                            *progress,
                            state,
                            &sides.neighbors,
                            cam,
                            self.breaking_mesh_settings,
                        ));
                    }
                } else {
                    cracks.extend(crack_vertices(*progress, state, cam));
                }
            }
        }
        lines.extend_from_slice(&cracks);
        if lines.is_empty() && styled.is_none() {
            return;
        }
        if lines.len() > self.capacity {
            self.capacity = lines.len().next_power_of_two();
            self.vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("block overlay"),
                size: (self.capacity * std::mem::size_of::<OverlayVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !lines.is_empty() {
            gpu.write_buffer_staged(&self.vertex_buffer, 0, bytemuck::cast_slice(&lines));
        }
        if let Some((selection, style)) = &styled {
            self.styled
                .get_or_insert_with(|| super::styled_outline::StyledOutline::new(gpu))
                .prepare(gpu, selection, style, cam);
        }

        let mut own = None;
        let pass = target.begin(gpu, "terrain/block-overlay", &mut own);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        if line_count > 0 {
            pass.set_pipeline(&self.line_pipeline);
            pass.draw(0..line_count as u32, 0..1);
        }
        // java: drawSelectionBox runs before drawBlockDamageTexture, so the styled box (like the
        // vanilla line above) precedes the cracks. `draw` rebinds vertex slot 0 and group 1,
        // which the crack draw below sets again.
        if let (Some(_), Some(outline)) = (&styled, &self.styled) {
            outline.draw(pass);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        }
        if let (Some(pipeline), Some(bg)) = (&self.crack_pipeline, &self.crack_bind_group) {
            if !cracks.is_empty() {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(1, bg, &[]);
                let start = line_count as u32;
                pass.draw(start..start + cracks.len() as u32, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::fixture;

    fn cam() -> Camera {
        let mut c = Camera {
            pos: DVec3::new(0.0, 0.0, 0.0),
            ..Camera::default()
        };
        c.update();
        c
    }

    fn actual(state: mc_data::BlockState) -> ActualBlockState {
        ActualBlockState::from_stored(state)
    }

    #[test]
    fn a_full_cube_selection_box_is_the_block_plus_two_thousandths() {
        let state = fixture::state_of("minecraft:stone");
        let b = SelectionBox::for_block(BlockPos::new(3, 4, 5), state);
        assert!((b.min.x - 2.998).abs() < 1e-6, "{:?}", b.min);
        assert!((b.max.y - 5.002).abs() < 1e-6, "{:?}", b.max);
    }

    #[test]
    fn a_partial_block_selection_box_follows_its_render_bounds() {
        // A slab is half height, so the outline must be too.
        let state = fixture::state_of("minecraft:stone_slab");
        let b = SelectionBox::for_block(BlockPos::ORIGIN, state);
        assert!(
            b.max.y < 0.6,
            "slab outline should be half height: {:?}",
            b.max
        );
    }

    /// java: `BlockCactus.getSelectedBoundingBox` — the one `getSelectedBoundingBox` override
    /// that is not equal to `setBlockBoundsBasedOnState`'s box. Inset 1/16 in X and Z, full
    /// height in Y.
    ///
    /// Negative control: `blocks.json` records cactus's `bounds` as the full cube, so if the
    /// override were dropped every assertion below would read `0.0`/`1.0`; and the collision box
    /// (`0.0625..0.9375` in **all three** axes) must not be the answer either, which the
    /// `max.y` assertion pins down.
    /// `RenderGlobal.drawSelectionBox` calls `setBlockBoundsBasedOnState` before reading the
    /// selected box. Exercise every lower/upper observation and every combined
    /// facing/open/hinge tuple; a static door `BlockProps::bounds` is a full cube.
    #[test]
    fn a_door_outline_uses_combined_live_bounds_for_both_halves() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        const F: f64 = 0.1875;
        const G: f64 = 1.0 - F;
        let lower = BlockPos::new(3, 64, 5);
        let door = mc_data::block_by_name("wooden_door").expect("wooden door");

        for facing in 0u8..4 {
            for open in [false, true] {
                for hinge_right in [false, true] {
                    let mut world = fixture::blank_world(0);
                    world.set_block(
                        lower,
                        mc_data::BlockState::new(door, facing | if open { 4 } else { 0 }),
                    );
                    world.set_block(
                        lower.up(),
                        mc_data::BlockState::new(door, 8 | if hinge_right { 1 } else { 0 }),
                    );
                    let (min_x, min_z, max_x, max_z) = if open {
                        match (facing, hinge_right) {
                            (0, false) => (0.0, 0.0, 1.0, F),
                            (0, true) => (0.0, G, 1.0, 1.0),
                            (1, false) => (G, 0.0, 1.0, 1.0),
                            (1, true) => (0.0, 0.0, F, 1.0),
                            (2, false) => (0.0, G, 1.0, 1.0),
                            (2, true) => (0.0, 0.0, 1.0, F),
                            (3, false) => (0.0, 0.0, F, 1.0),
                            (3, true) => (G, 0.0, 1.0, 1.0),
                            _ => unreachable!(),
                        }
                    } else {
                        match facing {
                            0 => (0.0, 0.0, F, 1.0),
                            1 => (0.0, 0.0, 1.0, F),
                            2 => (G, 0.0, 1.0, 1.0),
                            3 => (0.0, G, 1.0, 1.0),
                            _ => unreachable!(),
                        }
                    };

                    for pos in [lower, lower.up()] {
                        let live =
                            SelectionBox::for_block_in_world(pos, world.block_at(pos), &world);
                        assert_eq!(
                            live.min,
                            DVec3::new(
                                f64::from(pos.x) + min_x - E,
                                f64::from(pos.y) - E,
                                f64::from(pos.z) + min_z - E,
                            ),
                            "facing={facing}, open={open}, hinge_right={hinge_right}, pos={pos:?}"
                        );
                        assert_eq!(
                            live.max,
                            DVec3::new(
                                f64::from(pos.x) + max_x + E,
                                f64::from(pos.y + 1) + E,
                                f64::from(pos.z) + max_z + E,
                            ),
                            "facing={facing}, open={open}, hinge_right={hinge_right}, pos={pos:?}"
                        );
                    }
                }
            }
        }

        // Isolated negative control: a state-local lower half remains the static full cube;
        // only the live API may incorporate its upper-half hinge.
        let mut world = fixture::blank_world(0);
        world.set_block(lower, mc_data::BlockState::new(door, 4));
        // A right-hinged upper half is deliberately chosen: the extracted lower-state render
        // bounds can already represent the left-hinged open slab, but only the live path may
        // read this upper-half hinge and move it to the opposite side.
        world.set_block(lower.up(), mc_data::BlockState::new(door, 9));
        assert_ne!(
            SelectionBox::for_block(lower, world.block_at(lower)),
            SelectionBox::for_block_in_world(lower, world.block_at(lower), &world)
        );
    }

    /// `BlockFence.setBlockBoundsBasedOnState` makes one one-block-high envelope for all four
    /// neighbour arms.  This is intentionally not its collision shape: collision is 1.5 high
    /// and can comprise two overlapping bars.  Cover every connection mask for both fence
    /// materials, then pin the `canConnectTo` exceptions which a naive "solid neighbour" rule
    /// gets wrong.
    #[test]
    fn a_fence_outline_uses_every_live_connection_mask() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        let pos = BlockPos::new(3, 64, 5);

        for fence_name in ["minecraft:fence", "minecraft:nether_brick_fence"] {
            let fence = fixture::state_of(fence_name);
            for mask in 0_u8..16 {
                let mut world = fixture::blank_world(0);
                world.set_block(pos, fence);
                for (bit, neighbour) in [
                    (1, pos.north()),
                    (2, pos.south()),
                    (4, pos.west()),
                    (8, pos.east()),
                ] {
                    if mask & bit != 0 {
                        world.set_block(neighbour, fence);
                    }
                }

                let (north, south, west, east) =
                    (mask & 1 != 0, mask & 2 != 0, mask & 4 != 0, mask & 8 != 0);
                let live = selected_bounding_box(&world, pos, fence);
                assert_eq!(
                    live,
                    Aabb::new(
                        DVec3::new(
                            f64::from(pos.x) + if west { 0.0 } else { 0.375 },
                            f64::from(pos.y),
                            f64::from(pos.z) + if north { 0.0 } else { 0.375 },
                        ),
                        DVec3::new(
                            f64::from(pos.x) + if east { 1.0 } else { 0.625 },
                            f64::from(pos.y + 1),
                            f64::from(pos.z) + if south { 1.0 } else { 0.625 },
                        ),
                    ),
                    "{fence_name}, mask={mask:04b}"
                );

                let outline = SelectionBox::for_block_in_world(pos, fence, &world);
                assert_eq!(outline.min, live.min - DVec3::splat(E));
                assert_eq!(outline.max, live.max + DVec3::splat(E));
            }
        }

        let fence = fixture::state_of("minecraft:fence");
        let mut world = fixture::blank_world(0);
        world.set_block(pos, fence);
        // Different-material fence, barrier, and gourd all fail `canConnectTo`.
        fixture::place(&mut world, pos.north(), "minecraft:nether_brick_fence");
        fixture::place(&mut world, pos.south(), "minecraft:barrier");
        fixture::place(&mut world, pos.east(), "minecraft:pumpkin");
        // Gates connect irrespective of their own fence material/open state.
        fixture::place(&mut world, pos.west(), "minecraft:fence_gate");
        let live = selected_bounding_box(&world, pos, fence);
        assert_eq!(live.min, DVec3::new(3.0, 64.0, 5.375));
        assert_eq!(live.max, DVec3::new(3.625, 65.0, 5.625));
        assert_ne!(
            SelectionBox::for_block(pos, fence),
            SelectionBox::for_block_in_world(pos, fence, &world),
            "state-local bounds must not hide the live west gate connection"
        );
    }

    /// The outline consumer must read the shared world-owned fence envelope on every call.  The
    /// connected result catches the old state-local/static path, while the removal checks that a
    /// previously connected arm is not retained after the world changes.  This also pins the
    /// caller's Java `getSelectedBoundingBox(...).expand(0.002)` step, rather than only checking
    /// the unexpanded source helper.
    #[test]
    fn fence_outline_reads_connected_neighbour_and_refreshes_after_removal() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        let pos = BlockPos::new(3, 64, 5);
        let fence = fixture::state_of("minecraft:fence");
        let mut world = fixture::blank_world(0);
        world.set_block(pos, fence);
        fixture::place(&mut world, pos.east(), "minecraft:stone");

        let connected = SelectionBox::for_block_in_world(pos, fence, &world);
        assert_eq!(
            connected.min,
            DVec3::new(
                f64::from(pos.x) + 0.375 - E,
                f64::from(pos.y) - E,
                f64::from(pos.z) + 0.375 - E,
            )
        );
        assert_eq!(
            connected.max,
            DVec3::new(
                f64::from(pos.x + 1) + E,
                f64::from(pos.y + 1) + E,
                f64::from(pos.z) + 0.625 + E,
            )
        );
        assert_ne!(
            connected,
            SelectionBox::for_block(pos, fence),
            "negative control: the old isolated path misses the live east arm"
        );

        world.set_block(pos.east(), mc_data::BlockState::AIR);
        let isolated = SelectionBox::for_block_in_world(pos, fence, &world);
        assert_eq!(
            isolated.min,
            DVec3::new(
                f64::from(pos.x) + 0.375 - E,
                f64::from(pos.y) - E,
                f64::from(pos.z) + 0.375 - E,
            )
        );
        assert_eq!(
            isolated.max,
            DVec3::new(
                f64::from(pos.x) + 0.625 + E,
                f64::from(pos.y + 1) + E,
                f64::from(pos.z) + 0.625 + E,
            )
        );
        assert_ne!(
            connected, isolated,
            "neighbour removal must refresh the fence outline"
        );
    }

    /// `fence_selection_bounds_at` takes the centre state from the world, as Java's block
    /// methods do.  Passing a stale, different-material state is a caller-level control that
    /// distinguishes that shared oracle from the old physics duplicate, which used the caller's
    /// material and would incorrectly reject the wood neighbour.
    #[test]
    fn fence_outline_uses_the_world_centre_state_for_connection_material() {
        let pos = BlockPos::new(3, 64, 5);
        let wood = fixture::state_of("minecraft:fence");
        let stale_nether = fixture::state_of("minecraft:nether_brick_fence");
        let mut world = fixture::blank_world(0);
        world.set_block(pos, wood);
        world.set_block(pos.east(), wood);

        let live = selected_bounding_box(&world, pos, stale_nether);
        assert_eq!(
            live.min,
            DVec3::new(
                f64::from(pos.x) + 0.375,
                f64::from(pos.y),
                f64::from(pos.z) + 0.375,
            )
        );
        assert_eq!(
            live.max,
            DVec3::new(
                f64::from(pos.x + 1),
                f64::from(pos.y + 1),
                f64::from(pos.z) + 0.625,
            )
        );
        assert_ne!(
            live,
            static_selected_bounding_box(pos, stale_nether),
            "negative control: a static or caller-material path misses the live east arm"
        );
    }

    /// `BlockWall.setBlockBoundsBasedOnState` is not its 1.5-block-high collision box.  Its
    /// two straight-line cases shrink the transverse axis and lower the outline to 13/16.
    #[test]
    fn a_wall_outline_uses_every_live_connection_mask() {
        let pos = BlockPos::new(3, 64, 5);
        for wall_name in ["minecraft:cobblestone_wall"] {
            let wall = fixture::state_of(wall_name);
            for mask in 0_u8..16 {
                let mut world = fixture::blank_world(0);
                world.set_block(pos, wall);
                for (bit, neighbour) in [
                    (1, pos.north()),
                    (2, pos.south()),
                    (4, pos.west()),
                    (8, pos.east()),
                ] {
                    if mask & bit != 0 {
                        world.set_block(neighbour, wall);
                    }
                }
                let (north, south, west, east) =
                    (mask & 1 != 0, mask & 2 != 0, mask & 4 != 0, mask & 8 != 0);
                let (min_x, max_x, min_z, max_z, max_y) = if north && south && !west && !east {
                    (0.3125, 0.6875, 0.0, 1.0, 0.8125)
                } else if !north && !south && west && east {
                    (0.0, 1.0, 0.3125, 0.6875, 0.8125)
                } else {
                    (
                        if west { 0.0 } else { 0.25 },
                        if east { 1.0 } else { 0.75 },
                        if north { 0.0 } else { 0.25 },
                        if south { 1.0 } else { 0.75 },
                        1.0,
                    )
                };
                assert_eq!(
                    selected_bounding_box(&world, pos, wall),
                    bounds_at(pos, (min_x, 0.0, min_z), (max_x, max_y, max_z)),
                    "{wall_name}, mask={mask:04b}"
                );
            }
        }
    }

    /// The outline caller must use the shared live wall oracle, observe a neighbour removal on
    /// the next frame, and retain the Java selected-box expansion.  The state-local call is the
    /// old isolated-path mutant: it must disagree with both the connected arm and the refreshed
    /// isolated result.
    #[test]
    fn wall_outline_reads_live_neighbour_removal_and_expands_the_connected_arm() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        let pos = BlockPos::new(3, 64, 5);
        let wall = fixture::state_of("minecraft:cobblestone_wall");
        let mut world = fixture::blank_world(0);
        world.set_block(pos, wall);
        fixture::place(&mut world, pos.east(), "minecraft:stone");

        let connected = SelectionBox::for_block_in_world(pos, wall, &world);
        assert_eq!(
            connected.min,
            DVec3::new(
                f64::from(pos.x) + 0.25 - E,
                f64::from(pos.y) - E,
                f64::from(pos.z) + 0.25 - E,
            )
        );
        assert_eq!(
            connected.max,
            DVec3::new(
                f64::from(pos.x + 1) + E,
                f64::from(pos.y + 1) + E,
                f64::from(pos.z) + 0.75 + E,
            )
        );
        assert_ne!(
            connected,
            SelectionBox::for_block(pos, wall),
            "negative control: the old isolated path misses the live east arm"
        );

        // A live outline must not retain a stale neighbour-derived box after the arm is removed.
        world.set_block(pos.east(), mc_data::BlockState::AIR);
        let isolated = SelectionBox::for_block_in_world(pos, wall, &world);
        assert_eq!(
            isolated.min,
            DVec3::new(
                f64::from(pos.x) + 0.25 - E,
                f64::from(pos.y) - E,
                f64::from(pos.z) + 0.25 - E,
            )
        );
        assert_eq!(
            isolated.max,
            DVec3::new(
                f64::from(pos.x) + 0.75 + E,
                f64::from(pos.y + 1) + E,
                f64::from(pos.z) + 0.75 + E,
            )
        );
        assert_ne!(
            connected, isolated,
            "neighbour removal must refresh the wall outline"
        );
    }

    /// Java lowers and narrows a wall outline for a north/south straight run.  This caller-level
    /// control catches the tempting collision-box mutant (`maxY = 1.5`) as well as the old
    /// isolated state-local path, which has neither the arm nor the 13/16 height.
    #[test]
    fn wall_outline_uses_the_straight_run_selection_envelope() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        let pos = BlockPos::new(3, 64, 5);
        let wall = fixture::state_of("minecraft:cobblestone_wall");
        let mut world = fixture::blank_world(0);
        world.set_block(pos, wall);
        world.set_block(pos.north(), wall);
        world.set_block(pos.south(), wall);

        let straight = SelectionBox::for_block_in_world(pos, wall, &world);
        assert_eq!(
            straight.min,
            DVec3::new(
                f64::from(pos.x) + 0.3125 - E,
                f64::from(pos.y) - E,
                f64::from(pos.z) - E,
            )
        );
        assert_eq!(
            straight.max,
            DVec3::new(
                f64::from(pos.x) + 0.6875 + E,
                f64::from(pos.y) + 0.8125 + E,
                f64::from(pos.z + 1) + E,
            )
        );
        assert_ne!(
            straight,
            SelectionBox::for_block(pos, wall),
            "negative control: the old isolated path cannot produce a straight run"
        );
        assert!(
            straight.max.y < f64::from(pos.y) + 1.0,
            "selection must keep Java's 13/16 straight-run height, not collision maxY=1.5"
        );
    }

    /// `BlockPane.setBlockBoundsBasedOnState` has the unusual isolated full-cube result and
    /// then collapses to a 1/8-wide arm for a one-sided connection.  Cover all masks for both
    /// base classes; this is deliberately selection geometry rather than the cross collision
    /// volume in `mc-physics`.
    #[test]
    fn pane_outlines_follow_every_live_connection_mask() {
        let pos = BlockPos::new(3, 64, 5);
        for pane_name in ["minecraft:glass_pane", "minecraft:stained_glass_pane"] {
            let pane = fixture::state_of(pane_name);
            for mask in 0_u8..16 {
                let mut world = fixture::blank_world(0);
                world.set_block(pos, pane);
                for (bit, neighbour) in [
                    (1, pos.north()),
                    (2, pos.south()),
                    (4, pos.west()),
                    (8, pos.east()),
                ] {
                    if mask & bit != 0 {
                        world.set_block(neighbour, pane);
                    }
                }
                let (north, south, west, east) =
                    (mask & 1 != 0, mask & 2 != 0, mask & 4 != 0, mask & 8 != 0);
                let any = north || south || west || east;
                let (min_x, max_x) = if (!west || !east) && any {
                    if west {
                        (0.0, 0.5625)
                    } else if east {
                        (0.4375, 1.0)
                    } else {
                        (0.4375, 0.5625)
                    }
                } else {
                    (0.0, 1.0)
                };
                let (min_z, max_z) = if (!north || !south) && any {
                    if north {
                        (0.0, 0.5625)
                    } else if south {
                        (0.4375, 1.0)
                    } else {
                        (0.4375, 0.5625)
                    }
                } else {
                    (0.0, 1.0)
                };
                assert_eq!(
                    selected_bounding_box(&world, pos, pane),
                    bounds_at(pos, (min_x, 0.0, min_z), (max_x, 1.0, max_z)),
                    "{pane_name}, mask={mask:04b}"
                );
            }
        }
    }

    /// The outline caller must consume the live world oracle, then observe neighbour removal on
    /// the next call.  The connected arm is intentionally only a 1/8-wide selection envelope;
    /// the isolated result is Java's unusual full cube.  This is a caller-level control for the
    /// old state-local/duplicate pane path, not merely another assertion of the oracle's table.
    #[test]
    fn pane_outline_reads_connected_neighbour_and_refreshes_after_removal() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        let pos = BlockPos::new(3, 64, 5);

        for pane_name in ["minecraft:glass_pane", "minecraft:stained_glass_pane"] {
            let pane = fixture::state_of(pane_name);
            let mut world = fixture::blank_world(0);
            world.set_block(pos, pane);
            // Stone is a full block, so it must connect under BlockPane.canPaneConnectToBlock.
            fixture::place(&mut world, pos.east(), "minecraft:stone");

            let connected = SelectionBox::for_block_in_world(pos, pane, &world);
            assert_eq!(
                connected.min,
                DVec3::new(
                    f64::from(pos.x) + 0.4375 - E,
                    f64::from(pos.y) - E,
                    f64::from(pos.z) + 0.4375 - E,
                ),
                "{pane_name}: connected outline must use the east-arm envelope"
            );
            assert_eq!(
                connected.max,
                DVec3::new(
                    f64::from(pos.x + 1) + E,
                    f64::from(pos.y + 1) + E,
                    f64::from(pos.z) + 0.5625 + E,
                ),
                "{pane_name}: connected outline must retain Java's 0.002 expansion"
            );

            // A live consumer must not retain a stale neighbour-derived box after the block is
            // removed.  The refreshed isolated pane intentionally expands to a full cube.
            world.set_block(pos.east(), mc_data::BlockState::AIR);
            let isolated = SelectionBox::for_block_in_world(pos, pane, &world);
            assert_eq!(
                isolated.min,
                pos.to_vec3() - DVec3::splat(E),
                "{pane_name}: removal must refresh the minimum to the isolated full cube"
            );
            assert_eq!(
                isolated.max,
                pos.to_vec3() + DVec3::splat(1.0 + E),
                "{pane_name}: removal must refresh the maximum to the isolated full cube"
            );
            assert_ne!(
                connected, isolated,
                "{pane_name}: neighbour removal must change outline"
            );
        }
    }

    /// The outline consumer must use the live chest pair envelope, independently of FACING
    /// metadata, for both Java `BlockChest` singleton types.  The explicit expanded coordinates
    /// catch a consumer that calls the shared source helper but drops RenderGlobal's `0.002`
    /// expansion; the state-local comparison is the static-bounds negative control.
    #[test]
    fn chest_outlines_use_live_pairs_in_all_directions_and_keep_expansion() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        let pos = BlockPos::new(3, 64, 5);
        for chest_name in ["minecraft:chest", "minecraft:trapped_chest"] {
            let chest_id = mc_data::block_by_name(chest_name).expect("chest block");
            for (centre_meta, neighbour_meta) in [(2, 5), (3, 4), (4, 3), (5, 2)] {
                let chest = mc_data::BlockState::new(chest_id, centre_meta);
                let neighbour_state = mc_data::BlockState::new(chest_id, neighbour_meta);
                assert!(chest.is_legal(), "{chest_name} centre meta {centre_meta}");
                assert!(
                    neighbour_state.is_legal(),
                    "{chest_name} neighbour meta {neighbour_meta}"
                );
                for (neighbour, min, max) in [
                    (pos.north(), (0.0625, 0.0, 0.0), (0.9375, 0.875, 0.9375)),
                    (pos.south(), (0.0625, 0.0, 0.0625), (0.9375, 0.875, 1.0)),
                    (pos.west(), (0.0, 0.0, 0.0625), (0.9375, 0.875, 0.9375)),
                    (pos.east(), (0.0625, 0.0, 0.0625), (1.0, 0.875, 0.9375)),
                ] {
                    let mut world = fixture::blank_world(0);
                    world.set_block(pos, chest);
                    world.set_block(neighbour, neighbour_state);

                    assert_eq!(
                        selected_bounding_box(&world, pos, neighbour_state),
                        bounds_at(pos, min, max),
                        "{chest_name}, centre_meta={centre_meta}, neighbour_meta={neighbour_meta}, neighbour={neighbour:?}"
                    );

                    let outline = SelectionBox::for_block_in_world(pos, neighbour_state, &world);
                    assert_eq!(
                        outline.min,
                        DVec3::new(
                            f64::from(pos.x) + min.0 - E,
                            f64::from(pos.y) + min.1 - E,
                            f64::from(pos.z) + min.2 - E,
                        ),
                        "{chest_name}: live pair minimum must retain Java expansion"
                    );
                    assert_eq!(
                        outline.max,
                        DVec3::new(
                            f64::from(pos.x) + max.0 + E,
                            f64::from(pos.y) + max.1 + E,
                            f64::from(pos.z) + max.2 + E,
                        ),
                        "{chest_name}: live pair maximum must retain Java expansion"
                    );
                    assert_ne!(
                        outline,
                        SelectionBox::for_block(pos, neighbour_state),
                        "{chest_name}: static isolated bounds must not hide the live pair"
                    );
                }
            }
        }
    }

    /// The source checks north, south, west, east as an `if/else if` chain.  Every mask is a
    /// caller-level precedence control, so an `any`/union or last-match mutant cannot survive.
    #[test]
    fn chest_outlines_preserve_java_precedence_for_every_pair_mask() {
        let pos = BlockPos::new(3, 64, 5);
        for chest_name in ["minecraft:chest", "minecraft:trapped_chest"] {
            let chest = mc_data::block_by_name(chest_name)
                .expect("chest block")
                .default_state();
            assert!(chest.is_legal(), "{chest_name} default is registered");
            for mask in 0_u8..16 {
                let mut world = fixture::blank_world(0);
                world.set_block(pos, chest);
                for (bit, neighbour) in [
                    (1, pos.north()),
                    (2, pos.south()),
                    (4, pos.west()),
                    (8, pos.east()),
                ] {
                    if mask & bit != 0 {
                        world.set_block(neighbour, chest);
                    }
                }
                let expected = if mask & 1 != 0 {
                    ((0.0625, 0.0, 0.0), (0.9375, 0.875, 0.9375))
                } else if mask & 2 != 0 {
                    ((0.0625, 0.0, 0.0625), (0.9375, 0.875, 1.0))
                } else if mask & 4 != 0 {
                    ((0.0, 0.0, 0.0625), (0.9375, 0.875, 0.9375))
                } else if mask & 8 != 0 {
                    ((0.0625, 0.0, 0.0625), (1.0, 0.875, 0.9375))
                } else {
                    ((0.0625, 0.0, 0.0625), (0.9375, 0.875, 0.9375))
                };
                assert_eq!(
                    selected_bounding_box(&world, pos, chest),
                    bounds_at(pos, expected.0, expected.1),
                    "{chest_name}, neighbour mask={mask:04b}"
                );
            }
        }
    }

    /// Pairing is keyed by the live centre's Java singleton identity, not by a stale state
    /// argument or merely by the shared `BlockChest` class.  The second half also verifies that
    /// removing a live neighbour refreshes the outline instead of retaining a cached extension.
    #[test]
    fn chest_outlines_use_world_centre_type_and_refresh_after_removal() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        let pos = BlockPos::new(3, 64, 5);
        let ordinary = fixture::state_of("minecraft:chest");
        let trapped = fixture::state_of("minecraft:trapped_chest");

        for (centre, neighbour, stale_argument) in
            [(ordinary, trapped, trapped), (trapped, ordinary, ordinary)]
        {
            let mut world = fixture::blank_world(0);
            world.set_block(pos, centre);
            world.set_block(pos.east(), neighbour);
            assert_eq!(
                selected_bounding_box(&world, pos, stale_argument),
                bounds_at(pos, (0.0625, 0.0, 0.0625), (0.9375, 0.875, 0.9375)),
                "mixed ordinary/trapped pair must not use stale argument type"
            );
        }

        let mut world = fixture::blank_world(0);
        world.set_block(pos, ordinary);
        world.set_block(pos.east(), ordinary);
        let connected = SelectionBox::for_block_in_world(pos, ordinary, &world);
        assert_eq!(
            connected.min,
            DVec3::new(
                f64::from(pos.x) + 0.0625 - E,
                f64::from(pos.y) - E,
                f64::from(pos.z) + 0.0625 - E,
            )
        );
        assert_eq!(
            connected.max,
            DVec3::new(
                f64::from(pos.x + 1) + E,
                f64::from(pos.y) + 0.875 + E,
                f64::from(pos.z) + 0.9375 + E,
            )
        );
        assert_ne!(
            connected,
            SelectionBox::for_block(pos, ordinary),
            "negative control: isolated/static path cannot produce the east extension"
        );

        world.set_block(pos.east(), mc_data::BlockState::AIR);
        let isolated = SelectionBox::for_block_in_world(pos, ordinary, &world);
        assert_eq!(
            isolated,
            SelectionBox::for_block(pos, ordinary),
            "removal must refresh the live outline to isolated chest bounds"
        );
        assert_ne!(
            connected, isolated,
            "neighbour removal must change the outline"
        );
    }

    #[test]
    fn vine_outlines_follow_stored_faces_or_the_live_ceiling_fallback() {
        let pos = BlockPos::new(3, 64, 5);
        let vine_id = mc_data::block_by_name("vine").expect("vine");
        for (meta, expected) in [
            (1, ((0.0, 0.0, 0.9375), (1.0, 1.0, 1.0))), // south
            (2, ((0.0, 0.0, 0.0), (0.0625, 1.0, 1.0))), // west
            (4, ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0625))), // north
            (8, ((0.9375, 0.0, 0.0), (1.0, 1.0, 1.0))), // east
        ] {
            let vine = mc_data::BlockState::new(vine_id, meta);
            let mut world = fixture::blank_world(0);
            world.set_block(pos, vine);
            assert_eq!(
                selected_bounding_box(&world, pos, vine),
                bounds_at(pos, expected.0, expected.1)
            );
        }
        let vine = mc_data::BlockState::new(vine_id, 0);
        let mut world = fixture::blank_world(0);
        world.set_block(pos, vine);
        fixture::place(&mut world, pos.up(), "minecraft:stone");
        assert_eq!(
            selected_bounding_box(&world, pos, vine),
            bounds_at(pos, (0.0, 0.9375, 0.0), (1.0, 1.0, 1.0)),
            "a faceless vine falls back to the solid block above"
        );
    }

    /// The live outline must consume the centre block and support from the world, not a stale
    /// state snapshot supplied by a caller.  This is deliberately one test with three controls:
    /// the world metadata selects EAST while the argument selects SOUTH; support then supplies
    /// Java's 1/16 ceiling slab for metadata zero; removing support must expose the raw inverted
    /// source bounds, which the selected-box constructor normalises back to a full cube.
    #[test]
    fn vine_live_outline_reads_world_metadata_support_and_removal() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        let pos = BlockPos::new(3, 64, 5);
        let vine_id = mc_data::block_by_name("vine").expect("vine");
        let world_vine = mc_data::BlockState::new(vine_id, 8); // EAST
        let stale_vine = mc_data::BlockState::new(vine_id, 1); // SOUTH
        let mut world = fixture::blank_world(0);
        world.set_block(pos, world_vine);

        let east = SelectionBox::for_block_in_world(pos, stale_vine, &world);
        assert_eq!(
            east.min,
            DVec3::new(
                f64::from(pos.x) + 0.9375 - E,
                f64::from(pos.y) - E,
                f64::from(pos.z) - E,
            ),
            "live vine metadata must come from the world centre, not the stale argument"
        );
        assert_eq!(
            east.max,
            DVec3::new(
                f64::from(pos.x + 1) + E,
                f64::from(pos.y + 1) + E,
                f64::from(pos.z + 1) + E,
            )
        );

        let world_vine = mc_data::BlockState::new(vine_id, 0);
        world.set_block(pos, world_vine);
        fixture::place(&mut world, pos.up(), "minecraft:stone");
        let supported = SelectionBox::for_block_in_world(pos, stale_vine, &world);
        assert_eq!(
            supported.min,
            DVec3::new(
                f64::from(pos.x) - E,
                f64::from(pos.y) + 0.9375 - E,
                f64::from(pos.z) - E,
            ),
            "metadata zero must use the live full-cube support"
        );
        assert_eq!(
            supported.max,
            DVec3::new(
                f64::from(pos.x + 1) + E,
                f64::from(pos.y + 1) + E,
                f64::from(pos.z + 1) + E,
            )
        );

        world.set_block(pos.up(), mc_data::BlockState::AIR);
        let detached = SelectionBox::for_block_in_world(pos, stale_vine, &world);
        assert_eq!(
            detached.min,
            pos.to_vec3() - DVec3::splat(E),
            "support removal must refresh the raw meta-zero source to normalized full-cube min"
        );
        assert_eq!(
            detached.max,
            pos.to_vec3() + DVec3::splat(1.0 + E),
            "support removal must refresh the raw meta-zero source to normalized full-cube max"
        );
    }

    /// `BlockVine.setBlockBoundsBasedOnState` can leave its mutable fields inverted when a
    /// metadata-0 vine has neither a stored face nor a solid block above.  The raytrace must
    /// preserve those raw corners (so it passes through), but `Block.getSelectedBoundingBox`
    /// constructs an `AxisAlignedBB` that sorts them before the outline expands it.
    #[test]
    fn an_unattached_vine_outline_normalises_java_selected_bounds() {
        const E: f64 = 0.002_000_000_094_994_902_6;
        let pos = BlockPos::new(3, 64, 5);
        let vine = mc_data::BlockState::new(mc_data::block_by_name("vine").expect("vine"), 0);
        let mut world = fixture::blank_world(0);
        world.set_block(pos, vine);

        let raw = selected_bounding_box(&world, pos, vine);
        assert!(
            raw.min.x > raw.max.x && raw.min.y > raw.max.y && raw.min.z > raw.max.z,
            "negative control: the live vine helper must retain Java's raw inverted fields"
        );

        let outline = SelectionBox::for_block_in_world(pos, vine, &world);
        assert_eq!(
            outline.min,
            pos.to_vec3() - DVec3::splat(E),
            "AxisAlignedBB's selected-box constructor sorts min/max before expand"
        );
        assert_eq!(
            outline.max,
            pos.to_vec3() + DVec3::splat(1.0 + E),
            "the outline must be the full normalized cube, not a shrunken inverted box"
        );
    }

    #[test]
    fn a_cactus_outlines_narrower_than_its_block_but_full_height() {
        let state = fixture::state_of("minecraft:cactus");
        assert_eq!(
            mc_data::block_props(state).bounds.max_x,
            1.0,
            "the recorded render bounds are still the full cube"
        );
        let b = SelectionBox::for_block(BlockPos::new(3, 4, 5), state);
        assert!((b.min.x - (3.0625 - 0.002)).abs() < 1e-6, "{:?}", b.min);
        assert!((b.min.z - (5.0625 - 0.002)).abs() < 1e-6, "{:?}", b.min);
        assert!((b.max.x - (3.9375 + 0.002)).abs() < 1e-6, "{:?}", b.max);
        assert!((b.max.z - (5.9375 + 0.002)).abs() < 1e-6, "{:?}", b.max);
        // Full height, unlike the collision box.
        assert!((b.min.y - (4.0 - 0.002)).abs() < 1e-6, "{:?}", b.min);
        assert!((b.max.y - (5.0 + 0.002)).abs() < 1e-6, "{:?}", b.max);
    }

    #[test]
    fn the_outline_is_twelve_edges() {
        let state = fixture::state_of("minecraft:stone");
        let b = SelectionBox::for_block(BlockPos::ORIGIN, state);
        assert_eq!(b.line_vertices(&cam()).len(), 24);
    }

    /// Installs a synthetic, fully-controlled bake — one deliberately non-cube state plus two
    /// `destroy_stage` sprites at distinctive, non-overlapping atlas rects — so
    /// [`crack_vertices`] can be exercised without the real 1.8.9 jar, the same "src-level
    /// tests don't need a jar" convention [`crate::terrain`]'s module docs establish.
    ///
    /// Built via [`mc_assets::model::install`], the same additive, first-wins entry point
    /// `tests/s23_install.rs` (a separate binary) exercises directly. Within *this* binary
    /// `cargo test` runs every test in parallel on one process, so whichever call across every
    /// `#[test]` here actually wins the race is unobservable: every call builds byte-for-byte
    /// identical content at the same state id, so losing the race changes nothing.
    fn install_synthetic_crack_fixture() -> mc_data::BlockState {
        use mc_assets::model::{BakedAssets, BakedModel, Vertex};
        use mc_assets::sprite::{Sprite, SpriteFrame, UvRect};
        use mc_assets::{Atlas, BakedQuad};

        // A single up-facing quad at local y = 0.5 spanning the full x/z square. Nothing like
        // the render-bounds cube the old bounding-box code drew (y in {0, 1}), so a test that
        // still saw those values would prove the old code path was still running.
        let v = |x: f32, z: f32| Vertex {
            pos: [x, 0.5, z],
            // Deliberately meaningless: the real remap is position-based (java:
            // `BreakingFour.remapVert`), so the crack UV must not depend on this at all.
            uv: [0.0, 0.0],
        };
        let quad = BakedQuad {
            verts: [v(0.0, 0.0), v(0.0, 1.0), v(1.0, 1.0), v(1.0, 0.0)],
            // Deliberately `None`: proves `crack_vertices` recomputes the facing itself
            // (`facing_from_vertex_data`) instead of trusting a stored one, matching
            // `BreakingFour`'s own constructor.
            face: None,
            shade_face: None,
            tint_index: None,
            shade: true,
        };
        let model = BakedModel {
            general_quads: vec![quad],
            ..BakedModel::default()
        };

        let state_id = (mc_data::STATE_COUNT - 1) as u16;
        let mut models = vec![BakedModel::default(); mc_data::STATE_COUNT];
        models[state_id as usize] = model;

        let sprite = |loc: &str, uv: UvRect| Sprite {
            location: ResourceLocation::minecraft(loc.to_string()),
            width: 16,
            height: 16,
            uv,
            frames: vec![SpriteFrame {
                rgba: vec![0u8; 16 * 16 * 4],
                duration_ticks: None,
            }],
            default_duration_ticks: 1,
            interpolate: false,
            texture_metadata: Default::default(),
        };
        let sprites = vec![
            sprite(
                "textures/blocks/destroy_stage_3",
                UvRect {
                    min_u: 0.1,
                    max_u: 0.2,
                    min_v: 0.4,
                    max_v: 0.5,
                },
            ),
            sprite(
                "textures/blocks/destroy_stage_9",
                UvRect {
                    min_u: 0.6,
                    max_u: 0.7,
                    min_v: 0.75,
                    max_v: 0.85,
                },
            ),
        ];

        let assets = BakedAssets::from_parts(
            Atlas {
                texture: vec![0u8; 4],
                mipmaps: Vec::new(),
                w: 1,
                h: 1,
            },
            sprites,
            models,
            Vec::new(),
            Vec::new(),
        );
        mc_assets::install(assets);
        mc_data::BlockState { id: state_id }
    }

    #[test]
    fn crack_vertices_follow_the_real_model_not_a_bounding_box() {
        let state = install_synthetic_crack_fixture();
        let v = crack_vertices(
            BreakProgress {
                pos: BlockPos::new(2, 3, 4),
                stage: 3,
            },
            &actual(state),
            &cam(),
        );
        assert_eq!(v.len(), 6, "one quad -> two triangles -> six vertices");
        // The synthetic model's quad sits at local y = 0.5, not the 0/1 a bounding-box
        // approximation would use — proves this is the real re-baked geometry.
        for vert in &v {
            assert!((vert.pos[1] - 3.5).abs() < 1e-5, "{vert:?}");
        }
        // And the UV lands inside destroy_stage_3's own (distinctive, synthetic) atlas rect —
        // proves the crack samples the shared atlas by index, not a private strip.
        let dest = mc_assets::model::sprite(&ResourceLocation::minecraft(
            "textures/blocks/destroy_stage_3".to_string(),
        ))
        .uv;
        for vert in &v {
            assert!(
                vert.uv[0] >= dest.min_u - 1e-6 && vert.uv[0] <= dest.max_u + 1e-6,
                "{vert:?} outside {dest:?}"
            );
            assert!(
                vert.uv[1] >= dest.min_v - 1e-6 && vert.uv[1] <= dest.max_v + 1e-6,
                "{vert:?} outside {dest:?}"
            );
        }
    }

    #[test]
    fn damage_overlay_checked_sides_match_six_direction_and_non_opaque_controls() {
        use mc_assets::model::{BakedModel, Vertex};

        let pos = BlockPos::new(2, 64, 3);
        let state = fixture::state_of("minecraft:stone");
        let air = [BlockState::AIR; 6];
        assert_eq!(
            checked_face_visibility(pos, state, &air),
            [true; 6],
            "air must leave all six checked damage faces visible"
        );

        let stone = fixture::state_of("minecraft:stone");
        let opaque = [stone; 6];
        assert_eq!(
            checked_face_visibility(pos, state, &opaque),
            [false; 6],
            "an opaque cube must cull every shared face"
        );

        let glass = fixture::state_of("minecraft:glass");
        assert_eq!(
            checked_face_visibility(pos, state, &[glass; 6]),
            [true; 6],
            "transparent glass must not cull the damage face"
        );
        let partial = fixture::state_of("minecraft:stone_slab");
        assert_eq!(
            checked_face_visibility(pos, state, &[partial; 6]),
            [true; 6],
            "a partial slab must not cull the full cube's shared face"
        );

        // java: BlockLeavesBase.shouldSideBeRendered — fast graphics makes leaves opaque and
        // culls adjacent same-block faces; fancy graphics keeps their seam transparent.
        let leaves = fixture::state_of("minecraft:leaves");
        let fancy = crate::terrain::cull_face::MeshSettings {
            fancy_graphics: true,
            ambient_occlusion: true,
            ..crate::terrain::cull_face::MeshSettings::default()
        };
        let fast = crate::terrain::cull_face::MeshSettings {
            fancy_graphics: false,
            ambient_occlusion: true,
            ..crate::terrain::cull_face::MeshSettings::default()
        };
        assert_eq!(
            checked_face_visibility_with_settings(pos, leaves, &[leaves; 6], fancy),
            [true; 6],
            "fancy leaves keep adjacent crack faces visible"
        );
        assert_eq!(
            checked_face_visibility_with_settings(pos, leaves, &[leaves; 6], fast),
            [false; 6],
            "fast leaves cull adjacent crack faces like Java"
        );

        let vertex = |p| Vertex {
            pos: p,
            uv: [0.0, 0.0],
        };
        let quad = |facing| {
            let verts = match facing {
                Facing::Down => [
                    vertex([0.0, 0.0, 0.0]),
                    vertex([1.0, 0.0, 0.0]),
                    vertex([1.0, 0.0, 1.0]),
                    vertex([0.0, 0.0, 1.0]),
                ],
                Facing::Up => [
                    vertex([0.0, 1.0, 1.0]),
                    vertex([1.0, 1.0, 1.0]),
                    vertex([1.0, 1.0, 0.0]),
                    vertex([0.0, 1.0, 0.0]),
                ],
                Facing::North => [
                    vertex([1.0, 0.0, 0.0]),
                    vertex([0.0, 0.0, 0.0]),
                    vertex([0.0, 1.0, 0.0]),
                    vertex([1.0, 1.0, 0.0]),
                ],
                Facing::South => [
                    vertex([0.0, 0.0, 1.0]),
                    vertex([1.0, 0.0, 1.0]),
                    vertex([1.0, 1.0, 1.0]),
                    vertex([0.0, 1.0, 1.0]),
                ],
                Facing::West => [
                    vertex([0.0, 0.0, 0.0]),
                    vertex([0.0, 0.0, 1.0]),
                    vertex([0.0, 1.0, 1.0]),
                    vertex([0.0, 1.0, 0.0]),
                ],
                Facing::East => [
                    vertex([1.0, 0.0, 1.0]),
                    vertex([1.0, 0.0, 0.0]),
                    vertex([1.0, 1.0, 0.0]),
                    vertex([1.0, 1.0, 1.0]),
                ],
            };
            mc_assets::BakedQuad {
                verts,
                face: Some(facing),
                shade_face: Some(facing),
                tint_index: None,
                shade: true,
            }
        };
        let mut model = BakedModel::default();
        for facing in Facing::VALUES {
            model.face_quads[facing.index() as usize].push(quad(facing));
        }
        let progress = BreakProgress { pos, stage: 0 };
        let all_faces = crack_vertices_from_model(progress, &model, &cam(), Some(&[true; 6]));
        let no_faces = crack_vertices_from_model(progress, &model, &cam(), Some(&[false; 6]));
        assert_eq!(
            all_faces.len(),
            36,
            "six face quads emit two triangles each"
        );
        assert!(
            no_faces.is_empty(),
            "opaque-neighbour mask removes all six face quads"
        );
    }

    #[test]
    fn a_stage_above_the_last_is_clamped_to_stage_nine() {
        let state = install_synthetic_crack_fixture();
        let v = crack_vertices(
            BreakProgress {
                pos: BlockPos::ORIGIN,
                stage: 200,
            },
            &actual(state),
            &cam(),
        );
        assert!(!v.is_empty());
        let dest = mc_assets::model::sprite(&ResourceLocation::minecraft(
            "textures/blocks/destroy_stage_9".to_string(),
        ))
        .uv;
        for vert in &v {
            assert!(vert.uv[0] >= dest.min_u - 1e-6 && vert.uv[0] <= dest.max_u + 1e-6);
            assert!(vert.uv[1] >= dest.min_v - 1e-6 && vert.uv[1] <= dest.max_v + 1e-6);
        }
    }

    /// A state nothing baked geometry for (id `0`, deliberately left untouched by the synthetic
    /// fixture above) must degrade to no crack quads rather than panicking — the same "no jar,
    /// no panic" contract every other [`mc_assets::model::model_for`] consumer in this crate
    /// follows.
    #[test]
    fn a_state_with_no_baked_geometry_produces_no_crack_quads() {
        let v = crack_vertices(
            BreakProgress {
                pos: BlockPos::ORIGIN,
                stage: 0,
            },
            &actual(mc_data::BlockState { id: 0 }),
            &cam(),
        );
        assert!(v.is_empty());
    }

    #[test]
    fn overlay_anaglyph_upload_mixes_rgb_once_and_preserves_alpha() {
        let source = [
            255, 0, 0, 7, // red, translucent
            0, 255, 0, 19, // green
            0, 0, 255, 31, // blue
            128, 128, 128, 64,
        ];
        assert_eq!(overlay_upload_rgba(&source, false), source);
        assert_eq!(
            overlay_upload_rgba(&source, true),
            [76, 76, 76, 7, 178, 150, 0, 19, 0, 28, 178, 31, 128, 128, 128, 64,],
            "Java's integer anaglyph transform mixes RGB only"
        );
    }

    #[test]
    fn overlay_animation_rect_validation_covers_resident_mips() {
        let valid = AtlasAnimationUpload {
            mip_level: 1,
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            rgba: vec![0xAB; 8],
        };
        assert!(atlas_animation_rect_valid(&valid, 4, 4, 2));

        let mut edge = valid.clone();
        edge.x = 2;
        assert!(!atlas_animation_rect_valid(&edge, 4, 4, 2));
        let mut wrong_level = valid.clone();
        wrong_level.mip_level = 2;
        assert!(!atlas_animation_rect_valid(&wrong_level, 4, 4, 2));
        let mut wrong_payload = valid;
        wrong_payload.rgba.pop();
        assert!(!atlas_animation_rect_valid(&wrong_payload, 4, 4, 2));
    }

    #[test]
    fn overlay_animation_retains_raw_mips_across_anaglyph_toggles() {
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
        let mut phase = BlockOverlayPhase::new_with_anaglyph(&gpu, Some(&atlas), false);
        let animated = AtlasAnimationUpload {
            mip_level: 1,
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            rgba: [0u8, 255, 0, 23].repeat(2),
        };
        assert!(phase.upload_atlas_animation(&gpu, &animated));

        let mut expected_mip = mip;
        expected_mip[4..8].copy_from_slice(&animated.rgba[..4]);
        expected_mip[12..16].copy_from_slice(&animated.rgba[4..8]);
        let raw_levels = vec![base, expected_mip];
        assert_eq!(
            phase
                .crack_source_levels
                .as_ref()
                .expect("atlas source levels")
                .borrow()
                .as_slice(),
            raw_levels.as_slice(),
            "animation must retain raw bytes rather than converted GPU bytes"
        );

        phase.set_anaglyph_with_gpu(&gpu, true);
        assert!(phase.anaglyph_enabled());
        assert_eq!(
            phase
                .crack_source_levels
                .as_ref()
                .expect("atlas source levels")
                .borrow()
                .as_slice(),
            raw_levels.as_slice()
        );
        phase.set_anaglyph_with_gpu(&gpu, false);
        assert!(!phase.anaglyph_enabled());
        assert_eq!(
            phase
                .crack_source_levels
                .as_ref()
                .expect("atlas source levels")
                .borrow()
                .as_slice(),
            raw_levels.as_slice()
        );
    }

    #[test]
    fn java_rgba_overlay_texel_survives_crack_sampling_and_blend() {
        let Ok(mut gpu) = Gpu::headless(96, 96, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let state = install_synthetic_crack_fixture();
        let atlas = Atlas {
            texture: vec![200, 120, 60, 255],
            mipmaps: Vec::new(),
            w: 1,
            h: 1,
        };
        let mut phase = BlockOverlayPhase::new(&gpu, Some(&atlas));
        phase.breaking.push((
            BreakProgress {
                pos: BlockPos::ORIGIN,
                stage: 3,
            },
            actual(state),
        ));

        let mut phases = crate::PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: 0.25,
            g: 0.25,
            b: 0.25,
            a: 1.0,
        };
        phases.insert(crate::PhaseOrder::BLOCK_OVERLAY, Box::new(phase));
        let mut camera = cam();
        camera.pos = DVec3::new(0.5, 3.0, -2.5);
        camera.pitch = 45.0;
        camera.aspect = 1.0;
        camera.update();

        let frame = gpu.acquire().expect("acquire overlay byte-contract target");
        phases.render(&gpu, frame.view(), &camera, 0.0);
        frame.present(&gpu.queue);
        let pixels = gpu
            .read_pixels()
            .expect("read overlay byte-contract pixels");
        // GL_DST_COLOR / GL_SRC_COLOR gives 2 * texel * destination. The 0.25
        // clear is stored as 64/255; allow only final UNORM rounding, not a gamma shift.
        let expected = [100_u8, 60, 30];
        assert!(
            pixels.chunks_exact(4).any(|pixel| {
                pixel[..3]
                    .iter()
                    .zip(expected)
                    .all(|(&actual, expected)| actual.abs_diff(expected) <= 1)
            }),
            "the Java GL_RGBA crack texel and blend should produce {expected:?}"
        );
    }

    /// The crack pass's fixed-function state, against `RenderGlobal.preRenderDamagedBlocks` and
    /// `EntityRenderer.renderWorldPass` (see each item's docs for the Java lines).
    #[test]
    fn crack_pass_state_matches_the_java_damage_pass() {
        let sampler = crack_sampler_descriptor();
        assert_eq!(sampler.mag_filter, wgpu::FilterMode::Nearest);
        assert_eq!(sampler.min_filter, wgpu::FilterMode::Nearest);
        assert_eq!(
            sampler.lod_max_clamp, 0.0,
            "setBlurMipmap(false, false) is GL_NEAREST: base level only"
        );
        assert_eq!(CRACK_DEPTH_BIAS.constant, -3, "doPolygonOffset units");
        assert_eq!(CRACK_DEPTH_BIAS.slope_scale, -3.0, "doPolygonOffset factor");
        assert_eq!(CRACK_DEPTH_BIAS.clamp, 0.0);
        let primitive = crack_primitive_state();
        assert_eq!(primitive.cull_mode, Some(wgpu::Face::Back));
        assert_eq!(primitive.front_face, wgpu::FrontFace::Ccw);
    }

    /// A vanilla-shaped crack sprite: white at alpha 1 everywhere except opaque dark-grey crack
    /// texels, one in every 2x2 block (vanilla's `destroy_stage_N.png` use exactly these two
    /// kinds of texel; stage 9 has a crack texel in most 2x2 blocks).
    fn vanilla_shaped_crack_tile() -> Vec<u8> {
        let mut tile = Vec::with_capacity(16 * 16 * 4);
        for y in 0..16 {
            for x in 0..16 {
                if x % 2 == 0 && y % 2 == 0 {
                    tile.extend_from_slice(&[61, 61, 61, 255]);
                } else {
                    tile.extend_from_slice(&[255, 255, 255, 1]);
                }
            }
        }
        tile
    }

    /// A 256x256 atlas with four mip levels, every level tiled with the given 16x16 sprite's own
    /// mip chain, so whatever atlas rect a crack UV lands in samples the same sprite.
    fn tiled_crack_atlas(tile: &[u8]) -> Atlas {
        const SIZE: u32 = 256;
        let chain = mc_assets::sprite::generate_mipmap_chain(tile, 16, 16, 4);
        assert_eq!(chain.len(), 5);
        let level = |index: usize| {
            let size = SIZE >> index;
            let tile_size = 16 >> index;
            let mut out = Vec::with_capacity((size * size * 4) as usize);
            for y in 0..size {
                for x in 0..size {
                    let at = (((y % tile_size) * tile_size + x % tile_size) * 4) as usize;
                    out.extend_from_slice(&chain[index][at..at + 4]);
                }
            }
            out
        };
        Atlas {
            texture: level(0),
            mipmaps: (1..5).map(level).collect(),
            w: SIZE,
            h: SIZE,
        }
    }

    /// Render the synthetic crack fixture (one up-facing quad at y = 0.5, see
    /// `install_synthetic_crack_fixture`) over a 0.25 grey clear and return the RGBA pixels.
    fn render_crack(gpu: &mut Gpu, atlas: &Atlas, camera: &Camera) -> Vec<u8> {
        let state = install_synthetic_crack_fixture();
        let mut phase = BlockOverlayPhase::new(gpu, Some(atlas));
        phase.breaking.push((
            BreakProgress {
                pos: BlockPos::ORIGIN,
                stage: 3,
            },
            actual(state),
        ));
        let mut phases = crate::PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: 0.25,
            g: 0.25,
            b: 0.25,
            a: 1.0,
        };
        phases.insert(crate::PhaseOrder::BLOCK_OVERLAY, Box::new(phase));
        let frame = gpu.acquire().expect("acquire crack target");
        phases.render(gpu, frame.view(), camera, 0.0);
        frame.present(&gpu.queue);
        gpu.read_pixels().expect("read crack pixels")
    }

    fn camera_at(pos: DVec3, pitch: f32) -> Camera {
        let mut camera = cam();
        camera.pos = pos;
        camera.pitch = pitch;
        camera.aspect = 1.0;
        camera.update();
        camera
    }

    /// `(pixels brighter than the 64/255 clear, pixels clearly darker than it)`.
    fn brighter_and_darker(pixels: &[u8]) -> (usize, usize) {
        let brighter = pixels
            .chunks_exact(4)
            .filter(|p| p[..3].iter().any(|&c| c > 66))
            .count();
        let darker = pixels
            .chunks_exact(4)
            .filter(|p| p[..3].iter().all(|&c| c < 54))
            .count();
        (brighter, darker)
    }

    /// The 2026-11-06 report: "a not in frame or barely in frame face of the block will display
    /// a white breaking texture". A face seen nearly edge-on has a large UV derivative, so a
    /// mipmapped sampler reads the crack sprite's bright averaged levels. Java's damage pass
    /// reads level 0 only, where the white texels are alpha 1/255 and discarded; the only effect
    /// left is the dark crack (`2 * 61/255 * dst` darkens).
    ///
    /// Negative control: with the old sampler (no `lod_max_clamp`) the far part of the band
    /// samples levels 1-2 and turns white, so `brighter` is nonzero.
    #[test]
    fn a_crack_seen_edge_on_never_turns_white() {
        let Ok(mut gpu) = Gpu::headless(128, 128, crate::context::GpuOptions::default()) else {
            panic!("this qualification needs the real GPU adapter");
        };
        let tile = vanilla_shaped_crack_tile();
        let chain = mc_assets::sprite::generate_mipmap_chain(&tile, 16, 16, 4);
        // The premise, from the real mip filter: level 1 of a vanilla-shaped crack is bright and
        // well above the 0.1 alpha test, which is why sampling it draws white.
        assert!(
            chain[1].chunks_exact(4).all(|t| t[0] > 200 && t[3] > 100),
            "level 1 of the crack sprite must be bright and visible: {:?}",
            &chain[1][..8]
        );
        let atlas = tiled_crack_atlas(&tile);
        // 0.2 above the quad's plane, 1 to 2 blocks in front: the face spans about ten rows.
        let grazing = camera_at(DVec3::new(0.5, 0.7, -1.0), 8.5);
        let pixels = render_crack(&mut gpu, &atlas, &grazing);
        let (brighter, darker) = brighter_and_darker(&pixels);
        eprintln!("edge-on crack: brighter={brighter} darker={darker}");
        assert_eq!(
            brighter, 0,
            "an edge-on crack must never brighten the block (white crack)"
        );
        assert!(
            darker > 0,
            "positive control: the edge-on crack must draw its dark texels"
        );
    }

    /// java: `preRenderDamagedBlocks` draws with `color(1, 1, 1, 0.5)` under
    /// `alphaFunc(GL_GREATER, 0.1)`, and fixed-function GL tests the *combined* alpha: a crack
    /// texel needs alpha above 0.2. A resource pack's soft crack texel at 38/255 (0.149) is
    /// therefore discarded; mc-rs tested the bare texel and drew it (H7 2026-11-11). Control:
    /// 77/255 (0.302, combined 0.151) draws.
    #[test]
    fn a_crack_texel_must_clear_the_alpha_test_after_the_half_colour_alpha() {
        let Ok(mut gpu) = Gpu::headless(128, 128, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let camera = camera_at(DVec3::new(0.5, 1.5, -0.5), 45.0);
        let soft = tiled_crack_atlas(&[61, 61, 61, 38].repeat(16 * 16));
        let (_, soft_drawn) = brighter_and_darker(&render_crack(&mut gpu, &soft, &camera));
        let firm = tiled_crack_atlas(&[61, 61, 61, 77].repeat(16 * 16));
        let (_, firm_drawn) = brighter_and_darker(&render_crack(&mut gpu, &firm, &camera));
        eprintln!("crack alpha test: 38/255 drawn={soft_drawn}, 77/255 drawn={firm_drawn}");
        assert_eq!(soft_drawn, 0, "38/255 * 0.5 is under the 0.1 alpha test");
        assert!(firm_drawn > 100, "control: 77/255 * 0.5 clears it");
    }

    /// java: `GL_CULL_FACE` is enabled for the damage pass, so a crack quad seen from behind is
    /// not drawn. Control: the same quad seen from its front is drawn.
    #[test]
    fn a_crack_quad_seen_from_behind_is_culled() {
        let Ok(mut gpu) = Gpu::headless(128, 128, crate::context::GpuOptions::default()) else {
            panic!("this qualification needs the real GPU adapter");
        };
        // Every level an opaque crack texel, so any drawn fragment darkens the clear.
        let atlas = tiled_crack_atlas(&[61, 61, 61, 255].repeat(16 * 16));
        let front = render_crack(
            &mut gpu,
            &atlas,
            &camera_at(DVec3::new(0.5, 1.5, -0.5), 45.0),
        );
        let back = render_crack(
            &mut gpu,
            &atlas,
            &camera_at(DVec3::new(0.5, -0.5, -0.5), -45.0),
        );
        let (_, front_drawn) = brighter_and_darker(&front);
        let (_, back_drawn) = brighter_and_darker(&back);
        eprintln!("crack quad: front drawn={front_drawn} back drawn={back_drawn}");
        assert!(front_drawn > 100, "control: the front face must be drawn");
        assert_eq!(back_drawn, 0, "the back face must be culled");
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
        let mut direct = BlockOverlayPhase::new(&gpu, Some(&atlas));
        let mut captured = BlockOverlayPhase::new(&gpu, Some(&atlas));

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
                *direct
                    .crack_source_levels
                    .as_ref()
                    .expect("raw crack atlas")
                    .borrow(),
                expected,
                "toggle must retain raw bytes"
            );
            assert_eq!(
                *captured
                    .crack_source_levels
                    .as_ref()
                    .expect("raw crack atlas")
                    .borrow(),
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
                            std::ptr::eq(
                                texture,
                                captured.crack_texture.as_ref().expect("crack atlas")
                            ),
                            "sink must receive this owner's texture"
                        );
                        assert_eq!(
                            setting, enabled,
                            "sink must receive the owner's current setting"
                        );
                        // This borrow also proves the mutation borrow ended before the callback.
                        assert_eq!(
                            *captured
                                .crack_source_levels
                                .as_ref()
                                .expect("raw crack atlas")
                                .borrow(),
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
                    *captured
                        .crack_source_levels
                        .as_ref()
                        .expect("raw crack atlas")
                        .borrow(),
                    expected,
                    "capture raw bytes at record {index}"
                );
                assert_eq!(
                    *direct
                        .crack_source_levels
                        .as_ref()
                        .expect("raw crack atlas")
                        .borrow(),
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
        // Exercise the actual optional-owner absence gate, not a synthetic validator.
        let absent = BlockOverlayPhase::new(&gpu, None);
        assert!(
            !absent.upload_atlas_animation_with_sink(&stream[0], |_, _| panic!(
                "absent owner called sink"
            ))
        );
        assert!(!absent.upload_atlas_animation(&gpu, &stream[0]));
    }
}

/// BOVL-1 parity: the live-world checked-sides mask against the seven-block oracle `World`.
#[cfg(test)]
mod live_checked_sides_tests {
    use super::*;
    use crate::terrain::cull_face::MeshSettings;

    fn state(name: &str, meta: u8) -> Option<BlockState> {
        mc_data::block_by_name(name).map(|block| BlockState::new(block, meta))
    }

    /// Every `shouldSideBeRendered` override family plus world-aware selection bounds (doors,
    /// fences, walls, panes, chests, vines), random 5x5x5 neighbourhoods so cells two away are
    /// populated (the oracle sees air there), both graphics settings, at mid-height and at both
    /// world edges, straddling a chunk corner.
    #[test]
    fn live_world_mask_matches_the_seven_block_oracle_for_every_cull_override() {
        let centres: Vec<BlockState> = [
            ("stone", 0),
            ("glass", 0),
            ("stained_glass", 3),
            ("stained_glass", 5),
            ("ice", 0),
            ("packed_ice", 0),
            ("slime", 0),
            ("glass_pane", 0),
            ("stained_glass_pane", 3),
            ("leaves", 0),
            ("leaves2", 1),
            ("stone_slab", 0),
            ("stone_slab", 8),
            ("wooden_slab", 2),
            ("wooden_slab", 10),
            ("double_stone_slab", 0),
            ("snow_layer", 2),
            ("carpet", 4),
            ("farmland", 0),
            ("fence", 0),
            ("fence_gate", 1),
            ("cobblestone_wall", 0),
            ("unpowered_repeater", 0),
            ("unpowered_comparator", 0),
            ("end_portal", 0),
            ("wooden_door", 1),
            ("wooden_door", 8),
            ("vine", 0),
            ("vine", 5),
            ("cactus", 0),
            ("water", 0),
            ("flowing_lava", 3),
            ("hopper", 0),
            ("anvil", 0),
            ("dragon_egg", 0),
            ("oak_stairs", 2),
            ("chest", 2),
            ("portal", 1),
            ("bed", 0),
            ("torch", 5),
        ]
        .iter()
        .filter_map(|&(name, meta)| state(name, meta))
        .collect();
        assert!(
            centres.len() >= 35,
            "block registry names drifted: {}",
            centres.len()
        );
        let mut palette = centres.clone();
        palette.extend([BlockState::default(); 12]); // plenty of air

        let mut rng = mc_math::JavaRandom::new(0xC4AC);
        let fast = MeshSettings {
            fancy_graphics: false,
            ambient_occlusion: true,
            ..MeshSettings::default()
        };
        let mut culled_masks = 0;
        for round in 0..15 {
            let centre = BlockPos::new(16, [64, 255, 0][round % 3], 15);
            let mut world = World::new();
            for (x, z) in [(14, 13), (18, 13), (14, 17), (18, 17)] {
                world.load_empty_chunk(BlockPos::new(x, 0, z).chunk_pos());
            }
            for dx in -2..=2 {
                for dy in -2..=2 {
                    for dz in -2..=2 {
                        if (dx, dy, dz) == (0, 0, 0) {
                            continue;
                        }
                        let pick = rng.next_i32_bound(palette.len() as i32) as usize;
                        world.set_block(
                            BlockPos::new(centre.x + dx, centre.y + dy, centre.z + dz),
                            palette[pick],
                        );
                    }
                }
            }
            for &candidate in &centres {
                world.set_block(centre, candidate);
                let stored = world.block_at(centre);
                let neighbors = Facing::VALUES.map(|f| world.block_at(centre.relative(f)));
                for settings in [MeshSettings::default(), fast] {
                    let live =
                        live_checked_face_visibility(&world, centre, stored, &neighbors, settings);
                    let oracle =
                        checked_face_visibility_with_settings(centre, stored, &neighbors, settings);
                    assert_eq!(live, oracle, "round {round}: {stored:?} {settings:?}");
                    if live.contains(&false) {
                        culled_masks += 1;
                    }
                }
                // A stale `stored` (the world changed after the breaking list was built) must
                // fall back to the oracle rather than read the live centre.
                let stale = centres[(round * 7) % centres.len()];
                assert_eq!(
                    live_checked_face_visibility(&world, centre, stale, &neighbors, fast),
                    checked_face_visibility_with_settings(centre, stale, &neighbors, fast),
                );
            }
        }
        assert!(
            culled_masks > 100,
            "the sweep must exercise culling: {culled_masks}"
        );
    }

    /// BOVL-1 timing: one crack's mask per frame, oracle `World` vs live reads, for stone at
    /// y = 70 inside six stone neighbours. Printed, not asserted.
    #[test]
    #[ignore = "microbenchmark: run with --include-ignored --nocapture"]
    fn microbench_bovl1_checked_sides() {
        use std::time::Instant;
        let stone = state("stone", 0).expect("stone");
        let pos = BlockPos::new(8, 70, 8);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        world.set_block(pos, stone);
        for facing in Facing::VALUES {
            world.set_block(pos.relative(facing), stone);
        }
        let neighbors = Facing::VALUES.map(|f| world.block_at(pos.relative(f)));
        let settings = MeshSettings::default();
        const N: u32 = 200;
        let t = Instant::now();
        for _ in 0..N {
            std::hint::black_box(checked_face_visibility_with_settings(
                pos, stone, &neighbors, settings,
            ));
        }
        let oracle = t.elapsed() / N;
        let t = Instant::now();
        for _ in 0..N {
            std::hint::black_box(live_checked_face_visibility(
                &world, pos, stone, &neighbors, settings,
            ));
        }
        let live = t.elapsed() / N;
        assert_eq!(
            live_checked_face_visibility(&world, pos, stone, &neighbors, settings),
            [false; 6]
        );
        eprintln!("BOVL-1 per crack per frame: oracle World {oracle:?}, live reads {live:?}");
    }
}
