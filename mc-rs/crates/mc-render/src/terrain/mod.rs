//! Terrain meshing and the chunk renderer.
//!
//! S12 owns [`crate::entity`]; do not edit
//! across the boundary.
//!
//! # What this module is
//!
//! One pure function and one render phase.
//!
//! ```no_run
//! use mc_render::terrain::{mesh_section, ChunkMesh, TerrainPhase};
//! use mc_math::ChunkSectionPos;
//! # fn demo(world: &mc_world::World) {
//! let mesh: ChunkMesh = mesh_section(world, ChunkSectionPos::new(0, 4, 0));
//! # }
//! ```
//!
//! [`mesh_section`] turns one 16³ section of `mc_world::World` into four CPU vertex buffers —
//! vanilla's `EnumWorldBlockLayer` split of solid / cutout-mipped / cutout / translucent — and
//! [`TerrainPhase`] uploads them, culls them, and draws them. Because meshing is a pure
//! function of `&World` it runs on a `rayon` pool ([`jobs`]) and, more importantly, it is
//! *testable*: every claim this module makes about culling, lighting and layering is asserted
//! against an exact expected number rather than against a screenshot.
//!
//! # Layout
//!
//! | module | covers | java |
//! |---|---|---|
//! | [`mesher`] | the 4096-block walk, face culling, quad emission | `RenderChunk`, `BlockModelRenderer` |
//! | [`ao`] | smooth lighting / ambient occlusion (hazard **H8**) | `BlockModelRenderer.AmbientOcclusionFace` |
//! | [`light`] | packed sky/block brightness | `World.getCombinedLight`, `Block.getMixedBrightnessForBlock` |
//! | [`cull_face`] | `shouldSideBeRendered` and its nineteen overrides | `Block` + subclasses |
//! | [`tint`] | grass/foliage/water colormaps | `ColorizerGrass`, `BiomeColorHelper` |
//! | [`fluid`] | flowing/still water and lava | `BlockFluidRenderer` |
//! | [`visgraph`] | per-section face connectivity | `VisGraph`, `SetVisibility` |
//! | [`frustum`] · [`cull`] | frustum planes and the chunk-graph walk | `ClippingHelper`, `RenderGlobal.setupTerrain` |
//! | [`sort`] | back-to-front translucent ordering | `WorldRenderer.sortVertexData` |
//! | [`jobs`] | background meshing, dirty-set driven | `ChunkRenderDispatcher` |
//! | [`phase`] | the wgpu pipelines and the [`RenderPhase`](crate::RenderPhase) impl | `RenderGlobal.renderBlockLayer` |
//! | [`overlay`] | block-breaking progress and the selection box | `RenderGlobal.drawSelectionBox` |
//! | [`fixture`] | test-support world builders | — |
//!
//! # Startup
//!
//! Two installs have to happen before terrain looks right, both mirroring vanilla's own
//! one-shot global initialisation:
//!
//! 1. `mc_assets::model::install(bake_all(..))` — models and the atlas (S09's job).
//! 2. [`tint::load_colormaps`] — `textures/colormap/{grass,foliage}.png`.
//!
//! Neither is required for [`mesh_section`] to be *callable*: without them it produces empty
//! or untinted geometry rather than panicking, which keeps a headless test from needing the
//! 1.8.9 jar just to exercise the culling logic.

pub mod ao;
// Connected textures in the mesher.
pub mod ctm;
pub mod cull;
pub mod cull_face;
pub mod fixture;
pub mod fluid;
pub mod frustum;
pub mod jobs;
pub mod light;
pub mod mesher;
pub mod overlay;
pub mod phase;
pub mod sort;
// The Block Overlay mod's fill and wide outline.
pub mod styled_outline;
// The grass the Tall Grass mod hides when off.
pub mod tall_grass;
pub mod tint;
pub mod vertex;
pub mod visgraph;
mod walk_grid;

pub use ctm::CtmMode;
pub use cull::{
    visible_sections, visible_sections_with_view, SectionSource, SectionWalk, VisibleSection,
};
pub use cull_face::{should_side_be_rendered, MeshSettings};
pub use frustum::Frustum;
pub use jobs::{mesh_worker_panics, MeshQueue};
pub use mesher::{
    block_meshing_panic_is_being_wrapped, mesh_section, mesh_section_with, BlockMeshingPanic,
    ChunkMesh,
};
pub use overlay::{selected_bounding_box, BlockOverlayPhase, BreakProgress, SelectionBox};
pub use phase::{TerrainLayerPhase, TerrainOptions, TerrainPhase, TerrainRenderer, TerrainStats};
pub use sort::{needs_resort, sort_translucent};
pub use styled_outline::SelectionStyle;
pub use tall_grass::is_hidden_tall_grass;
pub use vertex::{Buffer, TerrainVertex, VERTEX_SIZE};
pub use visgraph::{SectionOcclusion, SetVisibility, VisGraph};
