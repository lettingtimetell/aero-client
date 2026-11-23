//! Orchestration: blockstate resolution + model baking + atlas assembly, wired into the
//! frozen `model_for`/`sprite` free functions.
//!
// java: ModelBakery.java (the overall load/bake sequence), BlockModelShapes.reloadModels

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use mc_data::{legal_states, BlockState, RenderType};
use mc_math::{coordinate_random, Facing as BlockFacing};

use super::blockmodel::{RawElement, RawFace, ResolvedModel};
use super::blockstate::{BlockStateJson, Variant};
use super::facebakery::{bake_quad, BakeRotation};
use super::item::{
    expand_builtin_generated, install_item_models, item_model_key, missing_item_model,
    registrations_for, texture_location,
};
use super::state_mapper;
use super::{BakedModel, BakedQuad, HotSwap};
use crate::atlas::{
    build_atlas_with_mipmap_levels, missing_texture_location, missing_texture_pixels, Atlas,
};
use crate::sprite::{Sprite, UvRect};
use crate::{AssetError, ResourceLocation, ResourceManager};

/// Everything baked once per resource-pack load: the atlas, every sprite's placement, and
/// every legal block state's baked geometry (plus item models, additively).
pub struct BakedAssets {
    /// The stitched block/item texture atlas.
    pub atlas: Atlas,
    sprites: HashMap<ResourceLocation, Sprite>,
    /// Indexed by `BlockState::id`, length `mc_data::STATE_COUNT`.
    models: Vec<BakedModel>,
    /// States with more than one weighted variant candidate — `model_for` always returns the
    /// highest-weight one; `model_for_at` picks among these by block position, matching
    /// vanilla's `WeightedBakedModel`.
    weighted: HashMap<u16, Vec<(i32, BakedModel)>>,
    /// Full `ModelResourceLocation` index. Unlike `models`, this can name properties supplied
    /// only by `Block.getActualState` (fence connections, stair shape, door hinge, ...).
    actual_models: HashMap<state_mapper::StateMapping, BakedModel>,
    /// Weighted candidates for the widened index.
    actual_weighted: HashMap<state_mapper::StateMapping, Vec<(i32, BakedModel)>>,
    /// This bake's item models. [`bake_all`] still publishes them itself through
    /// `install_item_models`, exactly as it always did — this copy exists only so [`reinstall`]
    /// can swap the item registry in the same step as the block registry, instead of a reload
    /// leaving items pointing at the previous pack's geometry.
    item_models: HashMap<u32, BakedModel>,
    /// The two `ModelBakery` registrations that are consumed by `RenderItemFrame`, not by a
    /// block state.  Index zero is `item_frame#normal`, index one `item_frame#map`.
    item_frame_models: [BakedModel; 2],
    /// Connected textures: `Some` exactly when the bake was asked for
    /// them ([`bake_all_with_ctm`] or [`BakedAssets::attach_ctm`]), even if the pack has no
    /// rules. Never cached; rebuilt from the pack on a cache hit.
    ctm: Option<crate::ctm::CtmTable>,
}

impl BakedAssets {
    /// The connected-textures table this bake carries, if it was baked with them.
    #[must_use]
    pub fn ctm(&self) -> Option<&crate::ctm::CtmTable> {
        self.ctm.as_ref()
    }

    /// Build the connected-textures table for a bake restored from the cache (M8).
    ///
    /// The cache stores the atlas, including any CTM tiles a CTM bake stitched, but not the
    /// table; parsing the ~50 rule files again is cheap. A rule whose tiles are not in this
    /// atlas is dropped, so a cache written without tiles yields only the pane-edge merge.
    pub fn attach_ctm(&mut self, rm: &ResourceManager) {
        let textures: HashSet<ResourceLocation> = self.sprites.keys().cloned().collect();
        let rules = crate::ctm::CtmRules::load(rm, &textures);
        self.ctm = Some(crate::ctm::CtmTable::build(
            &rules,
            &self.sprites,
            &self.models,
            &self.weighted,
            &self.actual_models,
        ));
    }

    /// Number of full blockstate-file/variant keys published by this bake.
    #[must_use]
    pub fn actual_model_count(&self) -> usize {
        self.actual_models.len()
    }

    /// Number of packed item-ID/metadata models published by this bake.
    #[must_use]
    pub fn item_model_count(&self) -> usize {
        self.item_models.len()
    }

    /// Number of usable manually registered item-frame variants (`normal`, `map`).
    #[must_use]
    pub fn item_frame_model_count(&self) -> usize {
        self.item_frame_models
            .iter()
            .filter(|model| !model.is_empty())
            .count()
    }

    /// Create fresh per-tick playback state for this bake's stitched atlas.
    ///
    /// The returned controller owns only animation state and frame/mip bytes; the caller keeps
    /// the mutable [`Atlas`] and applies its dirty rectangles to each GPU atlas it owns.
    #[must_use]
    pub fn atlas_animator(&self) -> crate::atlas::AtlasAnimator {
        crate::atlas::AtlasAnimator::new(&self.atlas, self.sprites.values())
    }

    /// Create playback state for the currently installed atlas, if a bake has been published.
    ///
    /// The client owns the mutable [`crate::atlas::Atlas`] and calls the returned animator once
    /// per 20 Hz tick. Keeping this accessor beside the registry means a resource-pack reload
    /// cannot accidentally pair a new atlas with a controller built from an older bake.
    #[must_use]
    pub fn installed_atlas_animator() -> Option<crate::atlas::AtlasAnimator> {
        registry().map(BakedAssets::atlas_animator)
    }

    /// Assemble a [`BakedAssets`] from parts that were baked earlier — the on-disk bake cache's
    /// constructor.
    ///
    /// Additive API for the startup cache: before it, the only way to obtain
    /// a `BakedAssets` was to run [`bake_all`], which made a cached bake unusable however
    /// correctly it had been stored. The argument shapes are deliberately the flat, ordered ones
    /// [`crate::cache::BakedSnapshot`] carries rather than this type's internal maps, so nothing
    /// about the storage layout leaks into the cache format.
    ///
    /// * `sprites` is keyed by each [`Sprite`]'s own `location`, so a duplicate location keeps
    ///   the **last** entry — the same rule stacking a resource pack follows.
    /// * `models` is indexed by [`mc_data::BlockState::id`]. A short vector is padded with the
    ///   empty model and a long one is truncated, so a cache written against a different block
    ///   registry cannot index out of bounds. (It cannot be *read* either — `mc_data::STATE_COUNT`
    ///   is part of [`crate::cache::BakeInputs`]'s key — but this constructor is public and must
    ///   be total on its own.)
    #[must_use]
    pub fn from_parts(
        atlas: Atlas,
        sprites: Vec<Sprite>,
        models: Vec<BakedModel>,
        weighted: Vec<(u16, Vec<(i32, BakedModel)>)>,
        item_models: Vec<(u16, BakedModel)>,
    ) -> BakedAssets {
        let mut models = models;
        models.resize_with(mc_data::STATE_COUNT, empty_model);
        BakedAssets {
            atlas,
            sprites: sprites
                .into_iter()
                .map(|s| (s.location.clone(), s))
                .collect(),
            models,
            weighted: weighted.into_iter().collect(),
            actual_models: HashMap::new(),
            actual_weighted: HashMap::new(),
            item_models: item_models
                .into_iter()
                .map(|(id, m)| (item_model_key(mc_data::ItemId(id), 0), m))
                .collect(),
            item_frame_models: std::array::from_fn(|_| empty_model()),
            ctm: None,
        }
    }

    /// Cache constructor including the widened actual-state model index.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts_with_actual(
        atlas: Atlas,
        sprites: Vec<Sprite>,
        models: Vec<BakedModel>,
        weighted: Vec<(u16, Vec<(i32, BakedModel)>)>,
        actual_models: Vec<(state_mapper::StateMapping, BakedModel)>,
        actual_weighted: Vec<(state_mapper::StateMapping, Vec<(i32, BakedModel)>)>,
        item_models: Vec<(u32, BakedModel)>,
        item_frame_models: [BakedModel; 2],
    ) -> BakedAssets {
        let mut assets = Self::from_parts(atlas, sprites, models, weighted, Vec::new());
        assets.actual_models = actual_models.into_iter().collect();
        assets.actual_weighted = actual_weighted.into_iter().collect();
        assets.item_models = item_models.into_iter().collect();
        assets.item_frame_models = item_frame_models;
        assets
    }

    /// The inverse of [`BakedAssets::from_parts`]: everything this bake holds, in the flat shape
    /// the cache stores.
    ///
    /// **Sorted** — sprites by location, weighted variants by state id, item models by item id —
    /// so two runs that produce the same bake produce the same cache file. That matters because
    /// `bake_all` walks `HashMap`s, and without the sort the file would differ byte for byte
    /// between runs of identical input for no reason. Since the `S21 → S09` quad-order request
    /// closed (`bake_resolved_model` now sorts each element's faces by `Facing` before baking,
    /// rather than trusting `RawElement::faces`' `HashMap` iteration), the quad sequence inside
    /// each `BakedModel` is pinned to the input too — so this sort is what makes the cache file
    /// genuinely byte-for-byte canonical across runs of identical input, not an approximation
    /// of it.
    ///
    /// Clones the whole bake, ~10 MiB against the real 1.8.9 assets. That is paid only on a cache
    /// *miss*, immediately after a 40 ms bake, and never on the hit path.
    #[must_use]
    #[allow(clippy::type_complexity)] // The five parts of a bake; naming a struct for them would
                                      // duplicate `BakedSnapshot`, which is exactly what this feeds.
    pub fn to_parts(
        &self,
    ) -> (
        Atlas,
        Vec<Sprite>,
        Vec<BakedModel>,
        Vec<(u16, Vec<(i32, BakedModel)>)>,
        Vec<(u16, BakedModel)>,
    ) {
        let mut sprites: Vec<Sprite> = self.sprites.values().cloned().collect();
        sprites.sort_by(|a, b| a.location.cmp(&b.location));
        let mut weighted: Vec<(u16, Vec<(i32, BakedModel)>)> =
            self.weighted.iter().map(|(k, v)| (*k, v.clone())).collect();
        weighted.sort_by_key(|(k, _)| *k);
        let mut item_models: Vec<(u16, BakedModel)> = self
            .item_models
            .iter()
            .filter(|(key, _)| (**key & 0xFFFF) == 0)
            .map(|(key, model)| ((*key >> 16) as u16, model.clone()))
            .collect();
        item_models.sort_by_key(|(k, _)| *k);
        (
            self.atlas.clone(),
            sprites,
            self.models.clone(),
            weighted,
            item_models,
        )
    }

    /// Every item model, including metadata variants, in deterministic packed-key order.
    pub(crate) fn item_models_to_parts(&self) -> Vec<(u32, BakedModel)> {
        let mut models: Vec<_> = self
            .item_models
            .iter()
            .map(|(key, model)| (*key, model.clone()))
            .collect();
        models.sort_by_key(|(key, _)| *key);
        models
    }

    /// The two manually registered item-frame models, in `normal`, `map` order.
    pub(crate) fn item_frame_models_to_parts(&self) -> [BakedModel; 2] {
        self.item_frame_models.clone()
    }

    /// The widened actual-state pieces in a deterministic cache-friendly order.
    #[allow(clippy::type_complexity)]
    pub(crate) fn actual_to_parts(
        &self,
    ) -> (
        Vec<(state_mapper::StateMapping, BakedModel)>,
        Vec<(state_mapper::StateMapping, Vec<(i32, BakedModel)>)>,
    ) {
        let mut models: Vec<_> = self
            .actual_models
            .iter()
            .map(|(key, model)| (key.clone(), model.clone()))
            .collect();
        models.sort_by(|(a, _), (b, _)| a.cmp(b));
        let mut weighted: Vec<_> = self
            .actual_weighted
            .iter()
            .map(|(key, candidates)| (key.clone(), candidates.clone()))
            .collect();
        weighted.sort_by(|(a, _), (b, _)| a.cmp(b));
        (models, weighted)
    }
}

/// One state that did not bake cleanly, and why. A clean bake against the real vanilla jar
/// produces an empty list — this exists for the acceptance test and for surfacing a hostile
/// or broken resource pack's damage without panicking.
/// One model-resolution failure with the exact registry identity that requested it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelBakeFailure {
    /// A model referenced by one blockstate variant could not be resolved.
    Block {
        /// The blockstate JSON resource.
        blockstate: ResourceLocation,
        /// The exact variant key within that resource.
        variant: String,
        /// The referenced model resource.
        model: ResourceLocation,
        /// The resolver error.
        reason: String,
    },
    /// A registered item/metadata model could not be resolved.
    Item {
        /// The numeric item registry ID.
        item_id: mc_data::ItemId,
        /// The metadata registered by `ItemModelMesher`.
        metadata: u16,
        /// The requested item-model resource.
        model: ResourceLocation,
        /// The resolver error.
        reason: String,
    },
    /// The manually registered item-frame blockstate file could not be loaded.
    ItemFrameDefinition {
        /// The exact blockstate resource.
        blockstate: ResourceLocation,
        /// The loader error.
        reason: String,
    },
    /// One manually registered item-frame variant was absent or its model failed to resolve.
    ItemFrameVariant {
        /// `normal` or `map`.
        variant: String,
        /// The requested model, or `None` when the variant key itself was absent.
        model: Option<ResourceLocation>,
        /// The resolution error.
        reason: String,
    },
}

/// Every state and registered-model identity that did not bake cleanly, plus the total legal
/// model-state coverage considered by the bake.
#[derive(Debug, Clone)]
pub struct BakeReport {
    /// States whose blockstate file or variant key could not be resolved.
    pub failures: Vec<(BlockState, String)>,
    /// Block, item and item-frame model references that failed after their registry identity was
    /// known. Kept separately so the long-standing state-failure API remains source-compatible.
    pub model_failures: Vec<ModelBakeFailure>,
    /// Total legal, `RenderType::Model` states considered.
    pub model_states_considered: usize,
}

impl BakeReport {
    /// True when every considered state baked cleanly.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failures.is_empty() && self.model_failures.is_empty()
    }
}

/// The textures vanilla stitches into the block atlas even though **no JSON model references
/// them**. java: `ModelBakery.LOCATIONS_BUILTIN_TEXTURES`, added to the texture set by
/// `getVariantsTextureLocations` right before `loadSprites` runs.
///
/// Each is bound by name at draw time by a renderer that has no model to go through:
///
/// * `blocks/{water,lava}_{still,flow}` — `BlockFluidRenderer.initAtlasSprites`. Without these
///   `sprite()` answers with its placeholder and every fluid quad's UV is wrong.
/// * `blocks/destroy_stage_0..9` — `RenderGlobal.loadRenderers`' `destroyBlockIcons`.
/// * `items/empty_armor_slot_*` — `GuiContainer`'s empty-armour-slot ghosts (S17).
///
/// Paths are given the way vanilla writes them (`blocks/…`, no `textures/` prefix and no
/// `.png`); [`super::item::texture_location`] adds the prefix, exactly as it does for a
/// texture that *did* come out of a model's `textures` table.
pub const BUILTIN_TEXTURES: [&str; 18] = [
    "blocks/water_flow",
    "blocks/water_still",
    "blocks/lava_flow",
    "blocks/lava_still",
    "blocks/destroy_stage_0",
    "blocks/destroy_stage_1",
    "blocks/destroy_stage_2",
    "blocks/destroy_stage_3",
    "blocks/destroy_stage_4",
    "blocks/destroy_stage_5",
    "blocks/destroy_stage_6",
    "blocks/destroy_stage_7",
    "blocks/destroy_stage_8",
    "blocks/destroy_stage_9",
    "items/empty_armor_slot_helmet",
    "items/empty_armor_slot_chestplate",
    "items/empty_armor_slot_leggings",
    "items/empty_armor_slot_boots",
];

/// Vanilla block-atlas locations reached by `ModelBakery`'s *complete* blockstate-variant and
/// item-model enumerations, but not by this port's state/model bake subset.
///
/// Java calls `getVariantsTextureLocations` for all 2,557 variant entries and
/// `getItemsTextureLocations` for all 513 registered item model locations before registering
/// sprites. `bake_all` only needs the smaller executable subset for geometry, but the atlas is
/// also a name-addressable registry (`sprite()`), so omitting these 45 real locations leaves
/// renderer-owned and metadata-selected textures at the blank fallback. Keep this separate from
/// [`BUILTIN_TEXTURES`]: that is Java's literal `LOCATIONS_BUILTIN_TEXTURES` constant (18),
/// whereas this is the measured difference caused by the intentionally narrower Rust discovery
/// pass. Paths are exactly the `ResourceLocation` names passed to `registerSprite` (no
/// `textures/` prefix or `.png` suffix).
pub const EXTRA_VANILLA_ATLAS_TEXTURES: [&str; 45] = [
    // Block variants Java loads but `legal_states()` does not reach as a distinct bake input.
    "blocks/double_plant_fern_top",
    "blocks/double_plant_grass_top",
    "blocks/double_plant_rose_top",
    "blocks/double_plant_sunflower_back",
    "blocks/double_plant_sunflower_front",
    "blocks/double_plant_sunflower_top",
    "blocks/double_plant_syringa_top",
    "blocks/fire_layer_1",
    "blocks/grass_side_snowed",
    "blocks/itemframe_background",
    "blocks/redstone_dust_line",
    "blocks/redstone_dust_line_overlay",
    // Flat/generated and metadata-selected item models Java registers regardless of whether
    // this port currently bakes their model entry.
    "items/clock",
    "items/compass",
    "items/charcoal",
    "items/door_wood",
    "items/fishing_rod_cast",
    "items/bow_pulling_0",
    "items/bow_pulling_1",
    "items/bow_pulling_2",
    "items/dye_powder_black",
    "items/dye_powder_blue",
    "items/dye_powder_brown",
    "items/dye_powder_cyan",
    "items/dye_powder_gray",
    "items/dye_powder_green",
    "items/dye_powder_light_blue",
    "items/dye_powder_lime",
    "items/dye_powder_magenta",
    "items/dye_powder_orange",
    "items/dye_powder_pink",
    "items/dye_powder_purple",
    "items/dye_powder_red",
    "items/dye_powder_silver",
    "items/dye_powder_white",
    "items/dye_powder_yellow",
    "items/fish_clownfish_raw",
    "items/fish_cod_cooked",
    "items/fish_cod_raw",
    "items/fish_pufferfish_raw",
    "items/fish_salmon_cooked",
    "items/fish_salmon_raw",
    "items/potion_bottle_drinkable",
    "items/potion_bottle_splash",
    "items/potion_overlay",
];

/// Loads every copy of a blockstate definition and overlays their variant maps in pack order.
///
/// `ModelBakery.getModelBlockDefinition` asks `getAllResources`, which is bottom-to-top. Its
/// `ModelBlockDefinition(List)` then calls `putAll` for each parsed definition, so a higher pack
/// replaces a duplicate variant key wholesale while inheriting keys it does not mention. This is
/// deliberately unlike model JSON loading, which uses the one winning resource.
///
/// Parse every supplied resource before returning a merged result. A bad lower-priority resource
/// is not made harmless by a valid override: vanilla fails the enclosing definition load when any
/// resource returned by `getAllResources` cannot be parsed.
fn load_blockstate_definition(
    rm: &ResourceManager,
    location: &ResourceLocation,
) -> Result<BlockStateJson, AssetError> {
    let resources = rm.all_resources(location);
    if resources.is_empty() {
        return Err(AssetError::NotFound(location.clone()));
    }

    let mut merged = BlockStateJson::default();
    for resource in resources {
        let definition =
            serde_json::from_slice::<BlockStateJson>(&resource.data).map_err(|error| {
                AssetError::json(format!("{location} in {}", resource.pack_name), error)
            })?;
        // Java's `mapVariants.putAll` replaces the entire `Variants` object; it does not append
        // alternatives from two packs with the same variant key.
        merged.variants.extend(definition.variants);
    }
    Ok(merged)
}

fn empty_model() -> BakedModel {
    BakedModel::default()
}

fn missing_block_model(missing_uv: UvRect) -> BakedModel {
    let element = RawElement {
        from: [0.0, 0.0, 0.0],
        to: [16.0, 16.0, 16.0],
        rotation: None,
        faces: HashMap::new(),
        shade: true,
    };
    let mut model = BakedModel {
        gui_3d: true,
        particle_uv: Some(missing_uv),
        ..BakedModel::default()
    };
    for (name, facing) in [
        ("down", BlockFacing::Down),
        ("up", BlockFacing::Up),
        ("north", BlockFacing::North),
        ("south", BlockFacing::South),
        ("west", BlockFacing::West),
        ("east", BlockFacing::East),
    ] {
        let face = RawFace {
            uv: None,
            texture: "missingno".to_owned(),
            cullface: Some(name.to_owned()),
            rotation: 0,
            tintindex: -1,
        };
        model.face_quads[facing.index() as usize].push(bake_quad(
            &element,
            facing,
            &face,
            missing_uv,
            BakeRotation::default(),
        ));
    }
    model
}

/// Bakes every legal block state plus every registered item's default model, and stitches
/// the atlas from exactly the textures those bakes reference. This is the real constructor;
/// [`install`] publishes the result for [`model_for`]/[`sprite`] to read.
///
/// # Errors
/// Only for the atlas packing itself failing outright (a texture too large for
/// `max_atlas_dimension` even alone) — a missing or malformed individual texture, model, or
/// blockstate file degrades that one state/item to the missing-texture placeholder and is
/// reported through the returned [`BakeReport`], not propagated.
pub fn bake_all(
    rm: &ResourceManager,
    max_atlas_dimension: u32,
) -> Result<(BakedAssets, BakeReport), AssetError> {
    bake_all_with_mipmap_levels(rm, max_atlas_dimension, crate::atlas::DEFAULT_MIPMAP_LEVELS)
}

/// [`bake_all`] honouring the user's `GameSettings.mipmapLevels` (0–4 in the options slider).
/// See [`crate::atlas::build_atlas_with_mipmap_levels`]; §7.6 / ATLAS-4.
///
/// # Errors
/// As [`bake_all`].
pub fn bake_all_with_mipmap_levels(
    rm: &ResourceManager,
    max_atlas_dimension: u32,
    mipmap_levels: u32,
) -> Result<(BakedAssets, BakeReport), AssetError> {
    bake_all_with_ctm(rm, max_atlas_dimension, mipmap_levels, false)
}

/// [`bake_all_with_mipmap_levels`], optionally with **connected textures**: the pack's `mcpatcher/ctm` tiles are stitched into the atlas and
/// [`BakedAssets::ctm`] carries the mesher's table. With `connected_textures == false` this is
/// byte-for-byte the vanilla bake.
///
/// A CTM pack can never break the bake: if the tiles do not fit `max_atlas_dimension`, the atlas
/// is stitched without them (logged) and only the pane-edge merge remains.
///
/// # Errors
/// As [`bake_all`].
pub fn bake_all_with_ctm(
    rm: &ResourceManager,
    max_atlas_dimension: u32,
    mipmap_levels: u32,
    connected_textures: bool,
) -> Result<(BakedAssets, BakeReport), AssetError> {
    bake_all_inner(
        rm,
        max_atlas_dimension,
        mipmap_levels,
        connected_textures,
        true,
    )
}

/// [`bake_all_with_mipmap_levels`] that publishes **nothing**
/// process-global: no block registry (as always, that is [`install`]'s job) and, unlike every
/// other `bake_all*`, no item models either.
///
/// For a private bake that must never touch the game's registries, such as the Aero title
/// screen's vanilla-textured background. The other entry points
/// publish this bake's item models first-wins, which is harmless after the game's own install
/// but would let a private bake that happened to run first win the slot for the whole process.
/// Read the result through [`with_thread_registry`].
///
/// # Errors
/// As [`bake_all`].
pub fn bake_all_unpublished(
    rm: &ResourceManager,
    max_atlas_dimension: u32,
    mipmap_levels: u32,
) -> Result<(BakedAssets, BakeReport), AssetError> {
    bake_all_inner(rm, max_atlas_dimension, mipmap_levels, false, false)
}

/// The body of every `bake_all*`. `publish_item_models` is `false` only for
/// [`bake_all_unpublished`].
fn bake_all_inner(
    rm: &ResourceManager,
    max_atlas_dimension: u32,
    mipmap_levels: u32,
    connected_textures: bool,
    publish_item_models: bool,
) -> Result<(BakedAssets, BakeReport), AssetError> {
    mc_core::profile_scope!("assets.bake_all");

    let mut resolved_cache: HashMap<ResourceLocation, ResolvedModel> = HashMap::new();
    let mut blockstate_cache: HashMap<String, Option<BlockStateJson>> = HashMap::new();
    let mut failures: Vec<(BlockState, String)> = Vec::new();
    let mut model_failures = Vec::new();
    let mut considered = 0usize;

    // Pass 1: discover the blockstate files reachable through vanilla's state mapper. Each
    // file is then enumerated in full: `getValidStates()` includes the cartesian product of
    // derived properties, not merely the 12-bit metadata representatives in `legal_states()`.
    struct Pending {
        mapping: state_mapper::StateMapping,
        candidates: Vec<Variant>,
    }
    let mut pending: Vec<Pending> = Vec::new();
    let mut raw_states: Vec<(BlockState, state_mapper::StateMapping)> = Vec::new();

    for state in legal_states() {
        if state.props().render_type != RenderType::Model {
            continue;
        }
        considered += 1;
        let mapping = state_mapper::resolve(state);
        raw_states.push((state, mapping.clone()));
        let bs = blockstate_cache.entry(mapping.file_stem.clone()).or_insert_with(|| {
            let loc = ResourceLocation::minecraft(format!("blockstates/{}.json", mapping.file_stem));
            match load_blockstate_definition(rm, &loc) {
                Ok(j) => Some(j),
                Err(e) => {
                    tracing::warn!(file = %mapping.file_stem, error = %e, "blockstate file failed to load");
                    None
                }
            }
        });
        let Some(bs) = bs else {
            failures.push((
                state,
                format!("blockstates/{}.json failed to load", mapping.file_stem),
            ));
            continue;
        };
        if !bs.variants.contains_key(&mapping.variant_key) {
            failures.push((
                state,
                format!(
                    "blockstates/{}.json has no variant {:?}",
                    mapping.file_stem, mapping.variant_key
                ),
            ));
            continue;
        }
    }

    // java: `ModelBakery.loadVariantItemModels:92-95` explicitly registers this pair even
    // though `minecraft:item_frame` is an entity, never a block.  Keep the candidates separate
    // from `pending`: putting them in `actual_models` would pretend a `BlockState` can address
    // them and corrupt the 2,555-state block-model invariant.
    let mut item_frame_candidates: [Vec<Variant>; 2] = std::array::from_fn(|_| Vec::new());
    let item_frame_location = ResourceLocation::minecraft("blockstates/item_frame.json");
    match load_blockstate_definition(rm, &item_frame_location) {
        Ok(blockstate) => {
            for (index, key) in ["normal", "map"].iter().enumerate() {
                let Some(candidates) = blockstate.variants.get(*key) else {
                    tracing::warn!(
                        variant = *key,
                        "item_frame blockstate lacks manually registered variant"
                    );
                    model_failures.push(ModelBakeFailure::ItemFrameVariant {
                        variant: (*key).to_owned(),
                        model: None,
                        reason: "item_frame blockstate lacks manually registered variant"
                            .to_owned(),
                    });
                    continue;
                };
                for variant in candidates {
                    if resolved_cache.contains_key(&variant.model) {
                        continue;
                    }
                    match ResolvedModel::resolve(rm, &variant.model) {
                        Ok(model) => {
                            resolved_cache.insert(variant.model.clone(), model);
                        }
                        Err(error) => {
                            let reason = error.to_string();
                            tracing::warn!(
                                variant = *key,
                                model = %variant.model,
                                error = %error,
                                "item-frame model failed to resolve"
                            );
                            model_failures.push(ModelBakeFailure::ItemFrameVariant {
                                variant: (*key).to_owned(),
                                model: Some(variant.model.clone()),
                                reason,
                            });
                        }
                    }
                }
                item_frame_candidates[index] = candidates.clone();
            }
        }
        Err(error) => {
            let reason = error.to_string();
            tracing::warn!(error = %error, "item_frame blockstate failed to load");
            model_failures.push(ModelBakeFailure::ItemFrameDefinition {
                blockstate: item_frame_location,
                reason,
            });
        }
    }

    for (file_stem, blockstate) in &blockstate_cache {
        let Some(blockstate) = blockstate else {
            continue;
        };
        for (variant_key, candidates) in &blockstate.variants {
            let mapping = state_mapper::StateMapping {
                file_stem: file_stem.clone(),
                variant_key: variant_key.clone(),
            };
            for v in candidates {
                if !resolved_cache.contains_key(&v.model) {
                    match ResolvedModel::resolve(rm, &v.model) {
                        Ok(rm_model) => {
                            resolved_cache.insert(v.model.clone(), rm_model);
                        }
                        Err(e) => {
                            tracing::warn!(model = %v.model, error = %e, "model failed to resolve");
                            model_failures.push(ModelBakeFailure::Block {
                                blockstate: ResourceLocation::minecraft(format!(
                                    "blockstates/{file_stem}.json"
                                )),
                                variant: variant_key.clone(),
                                model: v.model.clone(),
                                reason: e.to_string(),
                            });
                        }
                    }
                }
            }
            pending.push(Pending {
                mapping,
                candidates: candidates.clone(),
            });
        }
    }

    // Item models: default (undamaged) appearance for every registered item.
    struct PendingItem {
        id: mc_data::ItemId,
        metadata: u16,
        model_loc: ResourceLocation,
    }
    let mut pending_items: Vec<PendingItem> = Vec::new();
    for props in mc_data::items() {
        if !props.registered {
            continue;
        }
        let short = props.name.strip_prefix("minecraft:").unwrap_or(props.name);
        for registration in registrations_for(short) {
            let model_loc = ResourceLocation::minecraft(format!("item/{}", registration.model));
            if !resolved_cache.contains_key(&model_loc) {
                match ResolvedModel::resolve(rm, &model_loc) {
                    Ok(rm_model) => {
                        resolved_cache.insert(model_loc.clone(), rm_model);
                    }
                    Err(e) => {
                        tracing::warn!(
                            item = %props.name,
                            metadata = registration.metadata,
                            model = %model_loc,
                            error = %e,
                            "item model failed to resolve"
                        );
                        model_failures.push(ModelBakeFailure::Item {
                            item_id: props.id,
                            metadata: registration.metadata,
                            model: model_loc,
                            reason: e.to_string(),
                        });
                        continue;
                    }
                }
            }
            pending_items.push(PendingItem {
                id: props.id,
                metadata: registration.metadata,
                model_loc,
            });
        }
    }

    // Pass 2: collect every texture any resolved model references, and stitch the atlas.
    let mut texture_set: HashSet<ResourceLocation> = HashSet::new();
    for model in resolved_cache.values() {
        // java: `getTextureLocations`/`getItemsTextureLocations` both add
        // `resolveTextureName("particle")` unconditionally, *alongside* (not instead of) the
        // face-texture scan below, for every model -- block or item, builtin or not. Closes the
        // `S09 -> S09/S21` request: this pass used to collect only face textures, so a model
        // whose particle texture was not also a face texture baked to the missing-texture
        // placeholder wherever something looked it up by name (e.g. a block-break particle
        // system reading `sprite()`). Reuses `resolve_texture`, the same `#var` indirection
        // face textures already go through, so `particle` gets exactly the same parent-chain
        // resolution.
        if let Some(loc) = texture_location(model.resolve_texture("#particle")) {
            texture_set.insert(loc);
        }

        if let Some(builtin) = &model.builtin {
            if matches!(builtin.as_str(), "generated" | "compass" | "clock") {
                for layer in super::item::LAYERS {
                    if let Some(v) = model.textures.get(layer) {
                        if let Some(loc) = texture_location(v) {
                            texture_set.insert(loc);
                        }
                    }
                }
            }
            continue;
        }
        for element in &model.elements {
            for face in element.faces.values() {
                let resolved = model.resolve_texture(&face.texture);
                if let Some(loc) = texture_location(resolved) {
                    texture_set.insert(loc);
                }
            }
        }
    }
    // java: `set.addAll(LOCATIONS_BUILTIN_TEXTURES)` in `getVariantsTextureLocations`.
    for path in BUILTIN_TEXTURES {
        if let Some(loc) = texture_location(path) {
            texture_set.insert(loc);
        }
    }
    // `getVariantsTextureLocations` / `getItemsTextureLocations` see a wider registry than the
    // geometry bake above. Register their known delta too so name-based clients resolve the same
    // vanilla atlas sprites without making model discovery load unrelated geometry.
    for path in EXTRA_VANILLA_ATLAS_TEXTURES {
        if let Some(loc) = texture_location(path) {
            texture_set.insert(loc);
        }
    }
    // `TextureMap.initMissingImage` creates and registers this synthetic checkerboard before
    // loading the model texture set. A face whose `#texture` chain resolves to `missingno`
    // must use this *stitched* rectangle, not `UvRect::FULL` (which addresses every sprite in
    // a multi-sprite atlas).
    texture_set.insert(missing_texture_location());
    // Connected textures read their rules against this texture set.
    let ctm_rules = connected_textures.then(|| crate::ctm::CtmRules::load(rm, &texture_set));
    let mut texture_list: Vec<ResourceLocation> = texture_set.into_iter().collect();
    texture_list.sort(); // deterministic atlas layout across runs

    let (atlas, sprites) = match ctm_rules.as_ref().filter(|rules| !rules.is_empty()) {
        None => {
            build_atlas_with_mipmap_levels(rm, &texture_list, max_atlas_dimension, mipmap_levels)?
        }
        Some(rules) => {
            stitch_with_ctm_tiles(rm, &texture_list, rules, max_atlas_dimension, mipmap_levels)?
        }
    };
    // Fallback UV for a texture reference that resolved to nothing (a broken/hostile pack) —
    // not a real atlas region, just enough to bake *some* quad instead of dropping it.
    let missing_uv = sprites
        .get(&missing_texture_location())
        .expect("synthetic missingno sprite is included before stitching")
        .uv;
    let missing_block = missing_block_model(missing_uv);

    // Pass 3: bake. Cache by (model location, rotation, uvlock) since many states share the
    // same base model.
    let mut baked_cache: HashMap<(ResourceLocation, u8, u8, bool), BakedModel> = HashMap::new();
    let bake_variant = |v: &Variant,
                        cache: &mut HashMap<(ResourceLocation, u8, u8, bool), BakedModel>|
     -> BakedModel {
        let key = (
            v.model.clone(),
            v.rotation.quarters_x,
            v.rotation.quarters_y,
            v.uvlock,
        );
        if let Some(m) = cache.get(&key) {
            return m.clone();
        }
        let baked = match resolved_cache.get(&v.model) {
            Some(resolved) => bake_resolved_model(
                resolved,
                &sprites,
                missing_uv,
                v.rotation.quarters_x,
                v.rotation.quarters_y,
                v.uvlock,
            ),
            None => missing_block.clone(),
        };
        cache.insert(key, baked.clone());
        baked
    };

    let mut models = vec![empty_model(); mc_data::STATE_COUNT];
    let mut weighted: HashMap<u16, Vec<(i32, BakedModel)>> = HashMap::new();
    let mut actual_models: HashMap<state_mapper::StateMapping, BakedModel> = HashMap::new();
    let mut actual_weighted: HashMap<state_mapper::StateMapping, Vec<(i32, BakedModel)>> =
        HashMap::new();

    let item_frame_models = std::array::from_fn(|index| {
        let mut baked: Vec<(i32, BakedModel)> = item_frame_candidates[index]
            .iter()
            .filter(|variant| resolved_cache.contains_key(&variant.model))
            .map(|variant| (variant.weight, bake_variant(variant, &mut baked_cache)))
            .collect();
        baked.sort_by(|a, b| {
            b.0.cmp(&a.0).then_with(|| {
                let quad_count = |model: &BakedModel| {
                    model.general_quads.len()
                        + model
                            .face_quads
                            .iter()
                            .map(std::vec::Vec::len)
                            .sum::<usize>()
                };
                quad_count(&a.1).cmp(&quad_count(&b.1))
            })
        });
        baked
            .into_iter()
            .next()
            .map_or_else(empty_model, |(_, model)| model)
    });

    for p in &pending {
        let mut baked: Vec<(i32, BakedModel)> = p
            .candidates
            .iter()
            // java: `loadModelsCheck` drops a variant whose model failed to load. Keeping an
            // empty candidate changes both the base model and weighted selection.
            .filter(|v| resolved_cache.contains_key(&v.model))
            .map(|v| (v.weight, bake_variant(v, &mut baked_cache)))
            .collect();
        if baked.is_empty() {
            actual_models.insert(p.mapping.clone(), missing_block.clone());
            continue;
        }
        // java: WeightedBakedModel picks the base model as highest weight first, ties broken
        // by fewer total quads.
        baked.sort_by(|a, b| {
            b.0.cmp(&a.0).then_with(|| {
                let qa: usize = a.1.general_quads.len()
                    + a.1.face_quads.iter().map(std::vec::Vec::len).sum::<usize>();
                let qb: usize = b.1.general_quads.len()
                    + b.1.face_quads.iter().map(std::vec::Vec::len).sum::<usize>();
                qa.cmp(&qb)
            })
        });
        actual_models.insert(p.mapping.clone(), baked[0].1.clone());
        if baked.len() > 1 {
            actual_weighted.insert(p.mapping.clone(), baked);
        }
    }

    for (_, mapping) in &raw_states {
        actual_models
            .entry(mapping.clone())
            .or_insert_with(|| missing_block.clone());
    }

    // Preserve the frozen packed-state API by projecting each stored state through the wider
    // index. `model_for_actual_at` uses the latter directly after the mesher resolves neighbours.
    for (state, mapping) in raw_states {
        if let Some(model) = actual_models.get(&mapping) {
            models[state.id as usize] = model.clone();
        }
        if let Some(candidates) = actual_weighted.get(&mapping) {
            weighted.insert(state.id, candidates.clone());
        }
    }

    let mut item_models: HashMap<u32, BakedModel> = HashMap::new();
    for pi in &pending_items {
        let Some(resolved) = resolved_cache.get(&pi.model_loc) else {
            continue;
        };
        let baked = bake_item_model(resolved, &sprites, missing_uv);
        if should_register_item_model(resolved, &baked) {
            item_models.insert(item_model_key(pi.id, pi.metadata), baked);
        }
    }
    if publish_item_models {
        install_item_models(item_models.clone(), missing_item_model(missing_uv));
    }

    let ctm = ctm_rules.map(|rules| {
        crate::ctm::CtmTable::build(&rules, &sprites, &models, &weighted, &actual_models)
    });
    Ok((
        BakedAssets {
            atlas,
            sprites,
            models,
            weighted,
            actual_models,
            actual_weighted,
            item_models,
            item_frame_models,
            ctm,
        },
        BakeReport {
            failures,
            model_failures,
            model_states_considered: considered,
        },
    ))
}

/// Bake the three `ModelBakery.hasItemModel` markers, or the one custom-renderer marker.
///
/// java: `ModelBakery.hasItemModel` admits `MODEL_GENERATED`, `MODEL_COMPASS`, and
/// `MODEL_CLOCK`; all three pass through `ItemModelGenerator`. `MODEL_ENTITY` instead takes
/// the `isCustomRenderer` branch in `bakeBlockModels`, yielding `BuiltInModel`: it deliberately
/// has no JSON quads, disables AO, and preserves the resolved item-camera transforms for the
/// tile-entity item renderer. `BakedModel` has no built-in-renderer flag, so its empty geometry
/// is the faithful representation available at this seam — do not invent stand-in entity quads.
fn bake_item_model(
    resolved: &ResolvedModel,
    sprites: &HashMap<ResourceLocation, Sprite>,
    missing_uv: UvRect,
) -> BakedModel {
    let mut baked = match resolved.builtin.as_deref() {
        Some("generated" | "compass" | "clock") => {
            expand_builtin_generated(&resolved.textures, sprites, missing_uv)
        }
        Some("entity") => BakedModel {
            // `BuiltInModel.isAmbientOcclusion()` is hard-coded false, rather than inherited
            // from the leaf JSON as a normal `SimpleBakedModel` would be.
            ambient_occlusion: false,
            gui_3d: true,
            ..BakedModel::default()
        },
        // Unknown `builtin/*` parents cannot occur in vanilla 1.8.9. Treat a hostile pack's
        // value as an ordinary leaf so its own elements remain usable rather than silently
        // converting arbitrary geometry into a custom-renderer sentinel.
        _ => bake_resolved_model(resolved, sprites, missing_uv, 0, 0, false),
    };

    // Both Java paths retain `modelBlockIn.getAllTransforms()`: `SimpleBakedModel.Builder` for
    // generated/compass/clock and `new BuiltInModel(modelBlock.getAllTransforms())` for entity.
    baked.transforms = resolved.transforms;
    baked
}

/// Whether `ModelBakery` should publish one baked item under its requested key.
///
/// An empty generated/compass/clock result means `ItemModelGenerator` had no usable layers.
/// Leaving that empty value in the registry suppresses `ModelManager`'s shared missing-model
/// fallback. `builtin/entity` is deliberately different: its empty `BuiltInModel` is the marker
/// that delegates banners, chests, and skulls to the custom item renderer.
fn should_register_item_model(resolved: &ResolvedModel, baked: &BakedModel) -> bool {
    !baked.is_empty()
        || !matches!(
            resolved.builtin.as_deref(),
            Some("generated" | "compass" | "clock")
        )
}

fn bake_resolved_model(
    resolved: &ResolvedModel,
    sprites: &HashMap<ResourceLocation, Sprite>,
    missing_uv: UvRect,
    quarters_x: u8,
    quarters_y: u8,
    uvlock: bool,
) -> BakedModel {
    let mut model = BakedModel {
        ambient_occlusion: resolved.ambient_occlusion,
        gui_3d: true,
        particle_uv: Some(
            texture_location(resolved.resolve_texture("#particle"))
                .and_then(|location| sprites.get(&location))
                .map_or(missing_uv, |sprite| sprite.uv),
        ),
        transforms: resolved.transforms,
        ..BakedModel::default()
    };
    let rot = BakeRotation {
        quarters_x,
        quarters_y,
        uvlock,
    };
    for element in &resolved.elements {
        // Deterministic quad order (closes the S21 -> S09 request): `element.faces` is a
        // `HashMap<String, RawFace>`, so trusting its iteration order would make the quad
        // *sequence* within a bucket a property of two independently-hashed `HashMap`s'
        // random seeds rather than of the model itself. Harmless today — draw order inside a
        // bucket is unobservable, since the depth test settles opaque geometry and
        // `terrain::sort` orders translucent quads by distance — but it meant no byte-exact
        // golden could ever be taken over a `BakedModel`'s quad sequence. Sorting by `Facing`
        // pins the order to the input alone: its discriminant is already vanilla's own wire
        // ordinal order (Down-Up-North-South-West-East), so this needs no new table.
        let mut faces: Vec<_> = element
            .faces
            .iter()
            .filter_map(|(name, face)| {
                super::blockmodel::parse_facing(name).map(|facing| (facing, face))
            })
            .collect();
        faces.sort_by_key(|(facing, _)| *facing);

        for (facing, face) in faces {
            let resolved_tex = resolved.resolve_texture(&face.texture);
            let uv = texture_location(resolved_tex)
                .and_then(|loc| sprites.get(&loc))
                .map_or(missing_uv, |s| s.uv);
            let quad: BakedQuad = bake_quad(element, facing, face, uv, rot);
            match &face.cullface {
                Some(cull_name) => {
                    if let Some(cull_facing) = super::blockmodel::parse_facing(cull_name) {
                        let bucket =
                            super::facebakery::rotate_face(cull_facing, quarters_x, quarters_y);
                        model.face_quads[bucket.index() as usize].push(quad);
                    } else {
                        model.general_quads.push(quad);
                    }
                }
                None => model.general_quads.push(quad),
            }
        }
    }
    model
}

/// Stitch the vanilla texture list plus the connected-texture tiles (M8). When they do not fit
/// the atlas cap, stitch the vanilla list alone: a CTM pack must never break the bake.
fn stitch_with_ctm_tiles(
    rm: &ResourceManager,
    texture_list: &[ResourceLocation],
    rules: &crate::ctm::CtmRules,
    max_atlas_dimension: u32,
    mipmap_levels: u32,
) -> Result<(Atlas, HashMap<ResourceLocation, Sprite>), AssetError> {
    let known: HashSet<&ResourceLocation> = texture_list.iter().collect();
    let tiles: Vec<ResourceLocation> = rules
        .atlas_textures()
        .into_iter()
        .filter(|location| !known.contains(location))
        .collect();
    let mut with_tiles: Vec<ResourceLocation> = texture_list.to_vec();
    with_tiles.extend(tiles.iter().cloned());
    with_tiles.sort();
    match build_atlas_with_mipmap_levels(rm, &with_tiles, max_atlas_dimension, mipmap_levels) {
        Ok(stitched) => {
            tracing::info!(
                tiles = tiles.len(),
                width = stitched.0.w,
                height = stitched.0.h,
                "connected-texture tiles stitched"
            );
            Ok(stitched)
        }
        Err(error) => {
            tracing::warn!(
                %error,
                tiles = tiles.len(),
                max_atlas_dimension,
                "connected-texture tiles do not fit the atlas; baking without them"
            );
            build_atlas_with_mipmap_levels(rm, texture_list, max_atlas_dimension, mipmap_levels)
        }
    }
}

static REGISTRY: HotSwap<BakedAssets> = HotSwap::new();

std::thread_local! {
    /// Per-thread copy of `REGISTRY`, valid while the generation it was taken at still stands.
    ///
    /// The mesher runs `model_for` on a `rayon` pool, thousands of calls per chunk section. A
    /// `RwLock` read per call would put every worker on the same reader-count cache line; this
    /// reduces the steady state to one relaxed atomic *load* of a read-mostly line plus a
    /// thread-local compare, which is the same shape of work `OnceLock::get` did.
    static CACHED_REGISTRY: std::cell::Cell<(u64, Option<&'static BakedAssets>)> =
        const { std::cell::Cell::new((0, None)) };
}

/// The generation [`with_thread_registry`] parks in
/// `CACHED_REGISTRY` while its override is active. A real generation is a count of installs and
/// never reaches it, so the ordinary path's `cached_generation == generation` compare can never
/// accept it: the override costs that path nothing, and is only looked at on a cache miss.
const OVERRIDE_GENERATION: u64 = u64::MAX;

/// The installed assets, going through the per-thread cache.
fn registry() -> Option<&'static BakedAssets> {
    let generation = REGISTRY.generation();
    CACHED_REGISTRY.with(|cache| {
        let (cached_generation, cached) = cache.get();
        if cached_generation == generation || cached_generation == OVERRIDE_GENERATION {
            return cached;
        }
        let current = REGISTRY.load();
        cache.set((generation, current));
        current
    })
}

/// Run `f` with **this thread's** [`model_for`]-family lookups, [`sprite`] and [`ctm_table`]
/// reading `assets` instead of the installed registry, then restore whatever this thread saw
/// before (also when `f` panics). Nests.
///
/// Added for the Aero title screen, which meshes its background
/// with a private vanilla bake ([`bake_all_unpublished`]) while the game's registry holds the
/// user's resource packs. Nothing global changes: other threads, [`installed_generation`] and
/// the item-model registry ([`super::item_model_for`], which never went through this cache) are
/// untouched. Work that `f` hands to another thread (a `rayon` job) does **not** inherit the
/// override; run each job's lookups inside its own call.
///
/// Per-thread caches elsewhere that are keyed by [`installed_generation`] (the terrain mesher's
/// fluid UVs) cannot see an override by themselves; such a cache must bypass itself while
/// [`thread_registry_overridden`] is true.
pub fn with_thread_registry<R>(assets: &'static BakedAssets, f: impl FnOnce() -> R) -> R {
    struct Restore((u64, Option<&'static BakedAssets>));
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0;
            // `try_with`: a guard dropped while the thread's locals are being torn down must not
            // turn into a second panic.
            let _ = CACHED_REGISTRY.try_with(|cache| cache.set(previous));
        }
    }
    let previous = CACHED_REGISTRY.with(|cache| cache.replace((OVERRIDE_GENERATION, Some(assets))));
    let _restore = Restore(previous);
    f()
}

/// Whether [`with_thread_registry`] is active on this thread.
#[must_use]
pub fn thread_registry_overridden() -> bool {
    CACHED_REGISTRY.with(|cache| cache.get().0 == OVERRIDE_GENERATION)
}

/// Publishes `assets` for [`model_for`]/[`sprite`] to read. **Only the first call wins**, which
/// is the behaviour every caller since S09 has coded against; use [`reinstall`] to replace a
/// previous bake on a resource-pack reload.
///
/// Since S23 this also publishes the bake's **item** models for [`item_model_for`], with the same
/// first-wins rule. [`bake_all`] already published them itself on its way out, so for the
/// original path this call is a no-op and nothing changes; it exists so a `BakedAssets` that did
/// *not* come from `bake_all` — one rebuilt from [`crate::cache`] by
/// [`BakedAssets::from_parts`] — installs a complete registry rather than block models with the
/// item models silently missing. That is the "item-model install from outside `bake`" half of
/// the startup cache's needs.
pub fn install(assets: BakedAssets) {
    install_keyed(assets, None);
}

/// [`install`], remembering what produced `assets`: the bake cache's
/// identity of a **clean** bake ([`crate::cache::BakeInputs::key`]), or `None` when there is
/// none. A later [`reinstall_keyed`] with the same key reuses this bake.
pub fn install_keyed(assets: BakedAssets, key: Option<String>) {
    let missing_uv = assets
        .sprites
        .get(&missing_texture_location())
        .map_or(UvRect::FULL, |sprite| sprite.uv);
    let missing = missing_item_model(missing_uv);
    let item_models = assets.item_models.clone();
    REGISTRY.install_keyed(key.clone(), assets);
    super::item::install_item_models_keyed(item_models, missing, key);
}

/// Replace the installed assets, for a resource-pack reload (S21). Returns the new generation.
///
/// java: `SimpleReloadableResourceManager.reloadResources` re-runs every
/// `IResourceManagerReloadListener`, and `ModelManager.onResourceManagerReload` re-bakes the
/// whole model set — vanilla simply overwrites `ModelManager.modelRegistry`, which is what this
/// is the `&'static`-safe equivalent of.
///
/// **The previous generation is leaked**, deliberately and unavoidably: [`model_for`] returns
/// `&'static BakedModel`, so a mesher job already holding one must stay sound. See
/// [`super::HotSwap`] for the full argument. Cost is one baked-asset set per pack swap, not per
/// frame; [`reinstall_keyed`] bounds it to one per distinct pack stack.
pub fn reinstall(assets: BakedAssets) -> u64 {
    reinstall_keyed(assets, None)
}

/// [`reinstall`] for a reload whose inputs are known. When an earlier
/// install or reload published a bake under the same `key`, that bake and its item models are
/// published again and `assets` is dropped, instead of leaking one more baked-asset set (about
/// 12 MB against the 1.8.9 assets) on every reload. A server resource pack reloads on join and
/// again on leave, so without this every visit to such a server leaked two sets; with it, a
/// session leaks one set per distinct pack stack it ever used. `None` behaves as [`reinstall`].
///
/// `key` must identify the bake exactly: pass [`crate::cache::BakeInputs::key`] only for a
/// stampable stack and a clean bake (the rule the on-disk cache already follows).
pub fn reinstall_keyed(assets: BakedAssets, key: Option<String>) -> u64 {
    let missing_uv = assets
        .sprites
        .get(&missing_texture_location())
        .map_or(UvRect::FULL, |sprite| sprite.uv);
    let missing = missing_item_model(missing_uv);
    let item_models = assets.item_models.clone();
    let generation = REGISTRY.reinstall_keyed(key.clone(), assets);
    super::item::reinstall_item_models_keyed(item_models, missing, key);
    generation
}

/// How many baked-asset sets the registry has published, and so keeps
/// (each is leaked; see [`super::HotSwap`]).
#[must_use]
pub fn distinct_generations() -> usize {
    REGISTRY.distinct_published()
}

/// The `ModelBakery` entry that `RenderItemFrame` asks for by its manually registered
/// `ModelResourceLocation`: `false` is `item_frame#normal`, `true` is `item_frame#map`.
///
/// This is deliberately independent of [`model_for`]: item frames are entities, so no packed
/// `BlockState` can name either variant.  As with the other model lookup functions, an early
/// query is total and returns empty geometry.
#[must_use]
pub fn item_frame_model(map: bool) -> &'static BakedModel {
    let index = usize::from(map);
    registry()
        .map(|assets| &assets.item_frame_models[index])
        .unwrap_or_else(empty_model_static)
}

/// How many times block models have been installed: `0` before [`install`], `1` after it, one
/// more per [`reinstall`]. Lets a caller assert a reload actually took effect.
#[must_use]
pub fn installed_generation() -> u64 {
    REGISTRY.generation()
}

/// The installed bake's connected-textures table, or `None` when the
/// bake had them off. Read through the same per-thread registry cache as [`model_for`], so the
/// table and the models a mesher job sees come from one generation.
#[must_use]
pub fn ctm_table() -> Option<&'static crate::ctm::CtmTable> {
    registry().and_then(|assets| assets.ctm.as_ref())
}

fn empty_model_static() -> &'static BakedModel {
    static EMPTY: OnceLock<BakedModel> = OnceLock::new();
    EMPTY.get_or_init(empty_model)
}

/// The baked model for a block state. Frozen S09 API.
///
/// Total and non-panicking even before [`install`] has run (returns the empty model, the
/// same value a real `render_type != Model` state gets — never the missing-texture
/// placeholder, since that would misrepresent "not baked yet" as "vanilla has no model").
#[must_use]
pub fn model_for(state: BlockState) -> &'static BakedModel {
    match registry() {
        Some(r) if (state.id as usize) < r.models.len() => &r.models[state.id as usize],
        _ => empty_model_static(),
    }
}

/// Position-aware variant selection for multi-variant blockstates (java:
/// `WeightedBakedModel.getAlternativeModel`, keyed by `MathHelper.getPositionRandom`).
/// Additive S09 API: the frozen `model_for(state)` contract has no position parameter, so it
/// always returns the deterministic highest-weight candidate; this is for callers (the
/// terrain mesher) that want vanilla's per-position random pick (grass tuft rotation, stone
/// brick variety, ...).
#[must_use]
pub fn model_for_at(state: BlockState, pos: [i32; 3]) -> &'static BakedModel {
    let Some(r) = registry() else {
        return empty_model_static();
    };
    let Some(candidates) = r.weighted.get(&state.id) else {
        return model_for(state);
    };
    let total = java_weight_total(candidates);
    let hash = coordinate_random(pos[0], pos[1], pos[2]);
    let idx = weighted_index(hash, total);
    let selected = weighted_candidate_index(candidates, idx)
        .expect("Java WeightedRandom selection produced no model");
    &candidates[selected].1
}

fn actual_base_for<'a>(
    assets: &'a BakedAssets,
    state: &mc_data::block::ActualBlockState,
) -> Option<&'a BakedModel> {
    assets
        .actual_models
        .get(&state_mapper::resolve_actual(state))
}

/// Deterministic base-model lookup after vanilla's client-only `Block.getActualState` pass.
///
/// Unlike [`model_for_actual_at`], this deliberately does not choose a position-weighted
/// alternative. Vanilla's block-damage and digging-particle paths call
/// `BlockModelShapes.getModelForState` directly after actualising the state; only terrain chunk
/// baking enters `WeightedBakedModel.getAlternativeModel`.
#[must_use]
pub fn model_for_actual(state: &mc_data::block::ActualBlockState) -> &'static BakedModel {
    let Some(r) = registry() else {
        return empty_model_static();
    };
    actual_base_for(r, state).unwrap_or_else(|| model_for(state.stored()))
}

/// Position-aware lookup after vanilla's client-only `Block.getActualState` pass.
///
/// The widened key is the real `(blockstate file, property tuple)`
/// `ModelResourceLocation`; it can therefore address neighbour-derived variants that have no
/// packed metadata id. A synthetic/cached asset set made through the legacy [`from_parts`](BakedAssets::from_parts)
/// constructor has no widened index and deliberately falls back to [`model_for_at`].
#[must_use]
pub fn model_for_actual_at(
    state: &mc_data::block::ActualBlockState,
    pos: [i32; 3],
) -> &'static BakedModel {
    let Some(r) = registry() else {
        return empty_model_static();
    };
    let mapping = state_mapper::resolve_actual(state);
    let Some(base) = r.actual_models.get(&mapping) else {
        return model_for_at(state.stored(), pos);
    };
    let Some(candidates) = r.actual_weighted.get(&mapping) else {
        return base;
    };
    let total = java_weight_total(candidates);
    let hash = coordinate_random(pos[0], pos[1], pos[2]);
    let idx = weighted_index(hash, total);
    let selected = weighted_candidate_index(candidates, idx)
        .expect("Java WeightedRandom selection produced no model");
    &candidates[selected].1
}

/// Java `WeightedBakedModel.getAlternativeModel`'s
/// `Math.abs((int)positionRandom >> 16) % totalWeight`.
///
/// The cast is deliberately before the shift: Java discards the high 32 bits of the `long`,
/// then arithmetic-shifts that signed `int`. Shifting the full 64-bit hash chooses a different
/// variant for roughly a quarter of ordinary multi-variant block positions.
fn java_weight_total(candidates: &[(i32, BakedModel)]) -> i32 {
    candidates
        .iter()
        .fold(0_i32, |total, (weight, _)| total.wrapping_add(*weight))
}

/// `WeightedRandom.getRandomItem(collection, weight)`: subtract each signed item weight with
/// Java overflow and select the first candidate that makes the running value negative.
fn weighted_candidate_index(candidates: &[(i32, BakedModel)], mut index: i32) -> Option<usize> {
    for (candidate_index, (weight, _)) in candidates.iter().enumerate() {
        index = index.wrapping_sub(*weight);
        if index < 0 {
            return Some(candidate_index);
        }
    }
    None
}

fn weighted_index(position_hash: i64, total_weight: i32) -> i32 {
    let shifted = (position_hash as i32) >> 16;
    let numerator = shifted.wrapping_abs(); // Java Math.abs(i32::MIN) remains negative.
    if total_weight == 0 {
        panic!("Java WeightedBakedModel divides by zero total weight");
    }
    numerator.wrapping_rem(total_weight)
}

/// The sprite for a texture location (e.g. `minecraft:textures/blocks/stone`). Frozen S09
/// API. Returns a static placeholder sprite (the missing-texture checkerboard, at `(0,0)`
/// with a full `0..1` UV rect) before [`install`] has run or for a location nothing baked
/// referenced.
#[must_use]
pub fn sprite(l: &ResourceLocation) -> &Sprite {
    static PLACEHOLDER: OnceLock<Sprite> = OnceLock::new();
    match registry().and_then(|r| r.sprites.get(l)) {
        Some(s) => s,
        None => PLACEHOLDER.get_or_init(missing_sprite_placeholder),
    }
}

/// Java's `TextureMap.getAtlasSprite` returns its single shared `missingImage` for every
/// unregistered key. Keep the pre-install/unknown-key API on that same checkerboard instead of
/// manufacturing a transparent black image (or retaining the first requested key as metadata).
fn missing_sprite_placeholder() -> Sprite {
    Sprite {
        location: missing_texture_location(),
        width: 16,
        height: 16,
        uv: UvRect::FULL,
        frames: vec![crate::sprite::SpriteFrame {
            rgba: missing_texture_pixels(),
            duration_ticks: None,
        }],
        default_duration_ticks: 1,
        interpolate: false,
        texture_metadata: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::blockmodel::{RawElement, RawFace, RawRotation};
    use crate::ResourcePack;
    use mc_math::Facing;
    use std::collections::HashMap;
    use std::io::{Cursor, Read};
    use std::path::PathBuf;

    /// Tiny in-memory pack for blockstate stack tests. `ResourceManager` owns the ordering;
    /// this intentionally has no precedence behaviour of its own.
    struct BlockstatePack {
        name: String,
        files: HashMap<String, Vec<u8>>,
    }

    impl BlockstatePack {
        fn boxed(name: &str, files: &[(&str, &str)]) -> Box<dyn ResourcePack> {
            Box::new(Self {
                name: name.to_owned(),
                files: files
                    .iter()
                    .map(|(path, contents)| ((*path).to_owned(), contents.as_bytes().to_vec()))
                    .collect(),
            })
        }
    }

    impl ResourcePack for BlockstatePack {
        fn open(&self, location: &ResourceLocation) -> Option<Box<dyn Read>> {
            self.files
                .get(&location.to_string())
                .cloned()
                .map(|bytes| Box::new(Cursor::new(bytes)) as Box<dyn Read>)
        }

        fn contains(&self, location: &ResourceLocation) -> bool {
            self.files.contains_key(&location.to_string())
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn missing_block_fallback_is_a_six_face_cube_inside_the_stitched_rect() {
        let uv = UvRect {
            min_u: 0.25,
            max_u: 0.5,
            min_v: 0.125,
            max_v: 0.375,
        };
        let model = missing_block_model(uv);
        assert_eq!(model.particle_uv, Some(uv));
        assert!(model.general_quads.is_empty());
        assert_eq!(model.face_quads.iter().map(Vec::len).sum::<usize>(), 6);
        for facing in Facing::VALUES {
            let quads = &model.face_quads[facing.index() as usize];
            assert_eq!(quads.len(), 1, "{facing:?}");
            assert_eq!(quads[0].face, Some(facing));
            for vertex in &quads[0].verts {
                assert!((uv.min_u..=uv.max_u).contains(&vertex.uv[0]));
                assert!((uv.min_v..=uv.max_v).contains(&vertex.uv[1]));
            }
        }
    }

    #[test]
    fn absent_ordinary_models_fall_back_while_air_and_tile_blocks_stay_empty() {
        let manager = ResourceManager::new();
        let (assets, report) = bake_all(&manager, 512).expect("fallback-only bake");
        assert!(
            !report.is_clean(),
            "the empty pack is the missing-model fixture"
        );

        let model = |name: &str| {
            let state = mc_data::block_by_name(name)
                .unwrap_or_else(|| panic!("registered block {name}"))
                .default_state();
            &assets.models[state.id as usize]
        };
        assert_eq!(
            model("stone")
                .face_quads
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
            6,
            "ordinary missing Model state uses the cube fallback"
        );
        assert!(model("air").is_empty(), "air has no JSON model pass");
        assert!(
            model("chest").is_empty(),
            "tile-entity-rendered blocks remain intentionally empty"
        );
    }

    /// F-P2-client.resources.model-02-13: the report must identify failures outside the legacy
    /// packed-state list. In particular, an item failure has to make the bake non-clean so the
    /// client's clean-only cache gate cannot turn a hostile cold bake into a silent warm hit.
    #[test]
    fn model_resolution_failures_retain_block_item_and_item_frame_identities() {
        let mut manager = ResourceManager::new();
        manager.push_pack(BlockstatePack::boxed(
            "hostile-missing-model",
            &[(
                "minecraft:blockstates/stone.json",
                r#"{"variants":{"normal":{"model":"stone_missing_for_report"}}}"#,
            )],
        ));

        let (_, report) = bake_all(&manager, 512).expect("missing models use fallbacks");
        assert!(report.model_failures.iter().any(|failure| {
            matches!(
                failure,
                ModelBakeFailure::Block {
                    blockstate,
                    variant,
                    model,
                    ..
                } if *blockstate == ResourceLocation::minecraft("blockstates/stone.json")
                    && variant == "normal"
                    && *model == ResourceLocation::minecraft("block/stone_missing_for_report")
            )
        }));

        let stone = mc_data::item_by_name("stone").expect("registered stone item");
        assert!(report.model_failures.iter().any(|failure| {
            matches!(
                failure,
                ModelBakeFailure::Item {
                    item_id,
                    metadata: 0,
                    model,
                    ..
                } if *item_id == stone
                    && *model == ResourceLocation::minecraft("item/stone")
            )
        }));
        assert!(report.model_failures.iter().any(|failure| {
            matches!(
                failure,
                ModelBakeFailure::ItemFrameDefinition { blockstate, .. }
                    if *blockstate
                        == ResourceLocation::minecraft("blockstates/item_frame.json")
            )
        }));

        let mut model_failures_only = report.clone();
        model_failures_only.failures.clear();
        assert!(
            !model_failures_only.is_clean(),
            "model identities alone must suppress a clean-cache write"
        );
    }

    #[test]
    fn blockstate_definitions_merge_pack_variants_bottom_to_top() {
        let location = ResourceLocation::minecraft("blockstates/test.json");
        let mut manager = ResourceManager::new();
        manager.push_pack(BlockstatePack::boxed(
            "base",
            &[(
                "minecraft:blockstates/test.json",
                r#"{"variants":{"normal":{"model":"base_normal"},"facing=north":{"model":"base_north"}}}"#,
            )],
        ));
        manager.push_pack(BlockstatePack::boxed(
            "override",
            &[(
                "minecraft:blockstates/test.json",
                r#"{"variants":{"normal":{"model":"override_normal"},"facing=south":{"model":"override_south"}}}"#,
            )],
        ));

        let merged = load_blockstate_definition(&manager, &location).expect("merged blockstate");
        assert_eq!(merged.variants.len(), 3);
        assert_eq!(
            merged.variants["facing=north"][0].model.to_string(),
            "minecraft:block/base_north",
            "a key absent from the higher pack inherits the lower definition"
        );
        assert_eq!(
            merged.variants["facing=south"][0].model.to_string(),
            "minecraft:block/override_south"
        );
        assert_eq!(
            merged.variants["normal"].len(),
            1,
            "negative control: duplicate keys replace, rather than combine, candidate lists"
        );
        assert_eq!(
            merged.variants["normal"][0].model.to_string(),
            "minecraft:block/override_normal",
            "negative control: a top-pack duplicate must win"
        );
    }

    #[test]
    fn malformed_blockstate_in_any_pack_rejects_the_merged_definition() {
        let location = ResourceLocation::minecraft("blockstates/test.json");
        let valid = r#"{"variants":{"normal":{"model":"stone"}}}"#;

        for (bad_pack, base, top) in [("base", "{", valid), ("override", valid, "{")] {
            let mut manager = ResourceManager::new();
            manager.push_pack(BlockstatePack::boxed(
                "base",
                &[("minecraft:blockstates/test.json", base)],
            ));
            manager.push_pack(BlockstatePack::boxed(
                "override",
                &[("minecraft:blockstates/test.json", top)],
            ));
            let error = load_blockstate_definition(&manager, &location)
                .expect_err("one malformed participating resource rejects the definition");
            assert!(
                error.to_string().contains(bad_pack),
                "the parse error identifies the malformed {bad_pack} resource: {error}"
            );
        }
    }

    /// Closes the S21 -> S09 request: two independent bakes of identical input must emit the
    /// same quad **sequence**, not just the same set, and the sequence must be pinned to the
    /// input rather than to whichever order a `HashMap` happened to iterate.
    ///
    /// The element below is built with a deliberately non-canonical insertion order (not
    /// D-U-N-S-W-E), and `build_resolved` is called twice so the two bakes really do walk two
    /// independently-allocated `HashMap<String, RawFace>`s — `std`'s `RandomState` seeds each
    /// instance separately, so before the `bake_resolved_model` fix these two calls had no
    /// reason to agree (this is exactly the non-determinism `s21_reinstall.rs`'s old
    /// `fingerprint` comment measured: the stitched atlas matched across two bakes, the
    /// unsorted per-bucket quad hash did not).
    fn build_resolved_with_six_general_faces() -> ResolvedModel {
        let mut faces = HashMap::new();
        // Insertion order is West/South/Down/East/Up/North on purpose -- the opposite of
        // Facing's Down-Up-North-South-West-East ordinal order this test expects the baked
        // output to land on.
        for name in ["west", "south", "down", "east", "up", "north"] {
            faces.insert(
                name.to_string(),
                RawFace {
                    uv: None,
                    texture: "#all".to_string(),
                    cullface: None, // no cullface: every face lands in `general_quads`, where
                    // insertion order (not bucket assignment) is what's observable.
                    rotation: 0,
                    tintindex: -1,
                },
            );
        }
        let mut textures = HashMap::new();
        textures.insert("all".to_string(), "blocks/stone".to_string());
        ResolvedModel {
            elements: vec![RawElement {
                from: [0.0, 0.0, 0.0],
                to: [16.0, 16.0, 16.0],
                rotation: None,
                faces,
                shade: true,
            }],
            textures,
            texture_layers: vec![],
            ambient_occlusion: true,
            builtin: None,
            transforms: crate::model::transforms::ItemCameraTransforms::DEFAULT,
        }
    }

    fn build_resolved_with_named_face(
        face_name: &str,
        cullface: &str,
        axis: &str,
    ) -> ResolvedModel {
        let mut faces = HashMap::new();
        faces.insert(
            face_name.to_string(),
            RawFace {
                uv: None,
                texture: "#all".to_string(),
                cullface: Some(cullface.to_string()),
                rotation: 0,
                tintindex: -1,
            },
        );
        let mut textures = HashMap::new();
        textures.insert("all".to_string(), "blocks/stone".to_string());
        ResolvedModel {
            elements: vec![RawElement {
                from: [0.0, 0.0, 0.0],
                to: [16.0, 16.0, 16.0],
                rotation: Some(RawRotation {
                    origin: [8.0, 8.0, 8.0],
                    axis: axis.to_string(),
                    angle: 45.0,
                    rescale: false,
                }),
                faces,
                shade: true,
            }],
            textures,
            texture_layers: vec![],
            ambient_occlusion: true,
            builtin: None,
            transforms: crate::model::transforms::ItemCameraTransforms::DEFAULT,
        }
    }

    #[test]
    fn builtin_item_markers_generate_icons_but_entity_remains_a_renderer_sentinel() {
        let mut flat_icon = ResolvedModel {
            textures: HashMap::from([("layer0".to_string(), "items/test_icon".to_string())]),
            ambient_occlusion: true,
            transforms: crate::model::transforms::ItemCameraTransforms::DEFAULT,
            ..ResolvedModel::default()
        };

        // Java `ModelBakery.hasItemModel` uses root-model identity, not only the spelling of
        // `builtin/generated`: compass and clock must take exactly the same ItemModelGenerator
        // route. An empty sprite map deliberately exercises the stitched-missing fallback;
        // the generated core/perimeter quads must still exist.
        for marker in ["generated", "compass", "clock"] {
            flat_icon.builtin = Some(marker.to_string());
            let baked = bake_item_model(&flat_icon, &HashMap::new(), UvRect::FULL);
            assert!(
                !baked.is_empty(),
                "builtin/{marker} must use ItemModelGenerator"
            );
            assert!(
                !baked.gui_3d,
                "builtin/{marker} is ItemModelGenerator output"
            );
        }

        let mut entity = build_resolved_with_named_face("north", "north", "y");
        entity.builtin = Some("entity".to_string());
        entity.transforms.gui.rotation = [30.0, 45.0, 0.0];

        let ordinary = bake_resolved_model(&entity, &HashMap::new(), UvRect::FULL, 0, 0, false);
        assert!(
            !ordinary.is_empty(),
            "negative control: the leaf element would produce ordinary JSON geometry"
        );
        assert!(ordinary.gui_3d);

        let sentinel = bake_item_model(&entity, &HashMap::new(), UvRect::FULL);
        assert!(
            sentinel.is_empty(),
            "builtin/entity delegates drawing to TileEntityItemStackRenderer; do not fake quads"
        );
        assert!(
            !sentinel.ambient_occlusion,
            "BuiltInModel hard-codes isAmbientOcclusion() to false"
        );
        assert!(sentinel.gui_3d, "BuiltInModel hard-codes isGui3d() to true");
        assert_eq!(
            sentinel.transforms, entity.transforms,
            "BuiltInModel retains modelBlock.getAllTransforms()"
        );
        assert!(
            ordinary.ambient_occlusion,
            "negative control: treating entity as an ordinary SimpleBakedModel changes AO"
        );

        let empty_generated = ResolvedModel {
            builtin: Some("generated".to_string()),
            ..ResolvedModel::default()
        };
        let empty_bake = bake_item_model(&empty_generated, &HashMap::new(), UvRect::FULL);
        assert!(empty_bake.is_empty());
        assert!(
            !should_register_item_model(&empty_generated, &empty_bake),
            "an unusable generated model must remain absent so lookup selects builtin/missing"
        );
        assert!(
            should_register_item_model(
                &flat_icon,
                &bake_item_model(&flat_icon, &HashMap::new(), UvRect::FULL)
            ),
            "a generated model with layer0 remains registered"
        );
        assert!(
            should_register_item_model(&entity, &sentinel),
            "builtin/entity's empty model is an intentional custom-renderer sentinel"
        );
    }

    #[test]
    fn vanilla_banner_model_resolves_to_the_entity_renderer_sentinel_with_item_transforms() {
        // This is the shipped `assets/minecraft/models/item/banner.json`, kept inline so this
        // unit test exercises the real parent-marker resolution rather than only constructing a
        // `ResolvedModel` by hand.  Java's ModelBakery stops at `builtin/entity`; the item
        // renderer, not JSON quads, then supplies the banner cloth and its layered texture.
        let mut resources = ResourceManager::new();
        resources.push_pack(BlockstatePack::boxed(
            "vanilla-banner-model",
            &[(
                "minecraft:models/item/banner.json",
                r#"{
                    "parent": "builtin/entity",
                    "display": {
                        "thirdperson": {
                            "rotation": [0, 90, -90],
                            "translation": [0, 0, -4],
                            "scale": [0.5, 0.5, 0.5]
                        },
                        "firstperson": { "rotation": [0, 225, 0] },
                        "gui": {
                            "rotation": [0, -65, 0],
                            "translation": [0, -3, 0],
                            "scale": [0.85, 0.85, 0.85]
                        },
                        "head": {
                            "translation": [0, 8, 3.5],
                            "scale": [1.5, 1.5, 1.5]
                        }
                    }
                }"#,
            )],
        ));

        let resolved =
            ResolvedModel::resolve(&resources, &ResourceLocation::minecraft("item/banner"))
                .expect("the vanilla banner item model should resolve");
        assert_eq!(resolved.builtin.as_deref(), Some("entity"));
        assert!(
            resolved.elements.is_empty(),
            "builtin/entity has no JSON geometry for the item renderer to replay"
        );

        let baked = bake_item_model(&resolved, &HashMap::new(), UvRect::FULL);
        assert!(
            baked.is_empty(),
            "banner geometry belongs to TileEntityItemStackRenderer, not the atlas mesher"
        );
        assert!(!baked.ambient_occlusion);
        assert!(
            baked.particle_uv.is_none(),
            "builtin/entity has no SimpleBakedModel particle sprite"
        );
        assert_eq!(baked.transforms.first_person.rotation, [0.0, 225.0, 0.0]);
        assert_eq!(
            baked.transforms.third_person.translation,
            [0.0, 0.0, -4.0 / 16.0]
        );
        assert_eq!(baked.transforms.gui.translation, [0.0, -3.0 / 16.0, 0.0]);
        assert_eq!(baked.transforms.head.scale, [1.5, 1.5, 1.5]);
    }

    #[test]
    fn unresolved_face_uses_the_stitched_missingno_rect_in_a_multi_sprite_atlas() {
        // Deliberately give the atlas two other locations: `UvRect::FULL` would sample those
        // neighbours too, whereas the synthetic atlas sprite has its own 16x16 checker rect.
        let resources = ResourceManager::new();
        let missing = missing_texture_location();
        let locations = [
            ResourceLocation::minecraft("textures/blocks/first"),
            missing.clone(),
            ResourceLocation::minecraft("textures/blocks/second"),
        ];
        let (_atlas, sprites) = crate::atlas::build_atlas(&resources, &locations, 64).unwrap();
        let missing_uv = sprites[&missing].uv;
        assert_ne!(
            missing_uv,
            UvRect::FULL,
            "missingno must occupy one atlas rect"
        );

        let mut faces = HashMap::new();
        faces.insert(
            "north".to_string(),
            RawFace {
                uv: None,
                texture: "#undefined".to_string(),
                cullface: None,
                rotation: 0,
                tintindex: -1,
            },
        );
        let unresolved = ResolvedModel {
            elements: vec![RawElement {
                from: [0.0, 0.0, 0.0],
                to: [16.0, 16.0, 16.0],
                rotation: None,
                faces,
                shade: true,
            }],
            textures: HashMap::new(), // `#undefined` resolves to vanilla's `missingno`.
            texture_layers: vec![],
            ambient_occlusion: true,
            builtin: None,
            transforms: crate::model::transforms::ItemCameraTransforms::DEFAULT,
        };
        let baked = bake_resolved_model(&unresolved, &sprites, missing_uv, 0, 0, false);
        let quad = baked
            .general_quads
            .first()
            .expect("unresolved face still bakes");
        for vertex in quad.verts {
            assert!(
                vertex.uv[0] >= missing_uv.min_u && vertex.uv[0] <= missing_uv.max_u,
                "U {:?} escaped missingno rect {:?}",
                vertex.uv,
                missing_uv
            );
            assert!(
                vertex.uv[1] >= missing_uv.min_v && vertex.uv[1] <= missing_uv.max_v,
                "V {:?} escaped missingno rect {:?}",
                vertex.uv,
                missing_uv
            );
        }
    }

    #[test]
    fn resolved_particle_texture_is_retained_even_when_faces_use_another_sprite() {
        let particle_uv = UvRect {
            min_u: 0.1,
            max_u: 0.2,
            min_v: 0.3,
            max_v: 0.4,
        };
        let face_uv = UvRect {
            min_u: 0.6,
            max_u: 0.7,
            min_v: 0.8,
            max_v: 0.9,
        };
        let mut particle = missing_sprite_placeholder();
        particle.location = ResourceLocation::minecraft("textures/blocks/particle_only");
        particle.uv = particle_uv;
        let mut face = missing_sprite_placeholder();
        face.location = ResourceLocation::minecraft("textures/blocks/face_only");
        face.uv = face_uv;
        let sprites = HashMap::from([
            (particle.location.clone(), particle),
            (face.location.clone(), face),
        ]);
        let resolved = ResolvedModel {
            elements: vec![RawElement {
                from: [0.0, 0.0, 0.0],
                to: [16.0, 16.0, 16.0],
                rotation: None,
                faces: HashMap::from([(
                    "north".to_owned(),
                    RawFace {
                        uv: None,
                        texture: "#face".to_owned(),
                        cullface: None,
                        rotation: 0,
                        tintindex: -1,
                    },
                )]),
                shade: true,
            }],
            textures: HashMap::from([
                ("particle".to_owned(), "blocks/particle_only".to_owned()),
                ("face".to_owned(), "blocks/face_only".to_owned()),
            ]),
            texture_layers: vec![],
            ambient_occlusion: true,
            builtin: None,
            transforms: crate::model::transforms::ItemCameraTransforms::DEFAULT,
        };

        let baked = bake_resolved_model(&resolved, &sprites, UvRect::FULL, 0, 0, false);
        assert_eq!(baked.particle_uv, Some(particle_uv));
        assert!(baked.general_quads[0]
            .verts
            .iter()
            .all(|vertex| (face_uv.min_u..=face_uv.max_u).contains(&vertex.uv[0])));
    }

    #[test]
    fn bake_resolved_model_routes_mixed_case_json_names_through_java_by_name() {
        let sprites: HashMap<ResourceLocation, Sprite> = HashMap::new();
        let lower = bake_resolved_model(
            &build_resolved_with_named_face("north", "south", "x"),
            &sprites,
            UvRect::FULL,
            0,
            0,
            false,
        );
        let mixed = bake_resolved_model(
            &build_resolved_with_named_face("NoRtH", "SOUTH", "X"),
            &sprites,
            UvRect::FULL,
            0,
            0,
            false,
        );

        // This passes through all three live names: element `faces` key, `cullface`, and
        // `rotation.axis`.  A lowercase-only parser drops the mixed-case face, mis-buckets
        // the cullface, or falls back to Y for the rotation, respectively.
        assert_eq!(mixed.face_quads, lower.face_quads);
        assert_eq!(mixed.general_quads, lower.general_quads);
        assert_eq!(mixed.face_quads[Facing::South.index() as usize].len(), 1);
        assert!(mixed.general_quads.is_empty());
    }

    #[test]
    fn bake_resolved_model_emits_quads_in_a_deterministic_facing_order() {
        let sprites: HashMap<ResourceLocation, Sprite> = HashMap::new();
        let a = bake_resolved_model(
            &build_resolved_with_six_general_faces(),
            &sprites,
            UvRect::FULL,
            0,
            0,
            false,
        );
        let b = bake_resolved_model(
            &build_resolved_with_six_general_faces(),
            &sprites,
            UvRect::FULL,
            0,
            0,
            false,
        );

        assert_eq!(a.general_quads.len(), 6, "all six faces are cullface-free");
        assert_eq!(
            a.general_quads, b.general_quads,
            "two bakes of identical input must emit the same quad sequence, not just the same set"
        );

        // And it isn't merely "the two runs happened to agree by luck": the sequence is pinned
        // to Facing's vanilla ordinal order, which is NOT the insertion order used above, so a
        // bake that still trusted HashMap iteration would have no reliable reason to land here.
        let expected_facings = Facing::VALUES; // Down, Up, North, South, West, East
        let actual_facings: Vec<Facing> = a
            .general_quads
            .iter()
            .map(|q| q.face.expect("axis-aligned quad always has a facing"))
            .collect();
        assert_eq!(actual_facings, expected_facings.to_vec());
    }

    #[test]
    fn model_for_is_total_before_install() {
        // Cannot easily reset the OnceLock between tests, so just assert it never panics and
        // returns *some* model for an arbitrary state.
        let m = model_for(BlockState::AIR);
        assert!(m.is_empty() || !m.general_quads.is_empty());
    }

    #[test]
    fn weighted_model_hash_uses_the_shared_java_coordinate_random() {
        // `x * 3129871` is an `int` operation in Java.  ±687 cross that multiplication's
        // overflow edge; changing the shared call back to a widened `long` multiply chooses
        // different WeightedBakedModel alternatives at ordinary far-world coordinates.
        assert_eq!(
            coordinate_random(0, 64, 0),
            173_333_959_360,
            "ordinary control coordinate"
        );
        assert_eq!(
            coordinate_random(686, 64, -31),
            5_273_562_806_968_822_402,
            "last x before Java int multiplication wraps"
        );
        assert_eq!(
            coordinate_random(687, 64, -31),
            3_219_645_010_886_526_060,
            "first positive x whose Java multiplication wraps"
        );
        assert_eq!(
            coordinate_random(-686, -20, 17),
            -2_090_763_108_050_131_774,
            "last negative x before Java int multiplication wraps"
        );
        assert_eq!(
            coordinate_random(-687, -20, 17),
            -2_709_820_035_785_812_506,
            "first negative x whose Java multiplication wraps"
        );
        assert_eq!(
            coordinate_random(i32::MIN, 255, i32::MAX),
            7_254_248_326_551_643_980
        );
    }

    /// Independent spelling of Java 1.8.9's `MathHelper.getCoordinateRandom` (source:
    /// `MCP-919-main/src/minecraft/net/minecraft/util/MathHelper.java:390-397`). The x term is
    /// intentionally narrowed before widening: Java evaluates `x * 3129871` as an `int`, while
    /// the z term is promoted to `long` before its multiply.
    fn java_coordinate_random_oracle(x: i32, y: i32, z: i32) -> i64 {
        let mut hash = i64::from(x.wrapping_mul(3_129_871))
            ^ i64::from(z).wrapping_mul(116_129_781)
            ^ i64::from(y);
        hash = hash
            .wrapping_mul(hash)
            .wrapping_mul(42_317_861)
            .wrapping_add(hash.wrapping_mul(11));
        hash
    }

    /// Deliberately wrong control for the old `bake.rs::position_random` implementation. It
    /// widens x before multiplying, which only diverges once the Java int product overflows.
    fn widened_x_coordinate_random(x: i32, y: i32, z: i32) -> i64 {
        let mut hash = i64::from(x).wrapping_mul(3_129_871)
            ^ i64::from(z).wrapping_mul(116_129_781)
            ^ i64::from(y);
        hash = hash
            .wrapping_mul(hash)
            .wrapping_mul(42_317_861)
            .wrapping_add(hash.wrapping_mul(11));
        hash
    }

    #[test]
    fn weighted_model_hash_x_multiply_matches_java_at_overflow_edges() {
        // The vectors are generated from the Java 1.8.9 expression above and deliberately
        // straddle the first positive/negative i32-product overflow at |x| = 687.
        let cases = [
            (0, 64, 0, 173_333_959_360_i64),
            (686, 64, -31, 5_273_562_806_968_822_402_i64),
            (687, 64, -31, 3_219_645_010_886_526_060_i64),
            (-686, -20, 17, -2_090_763_108_050_131_774_i64),
            (-687, -20, 17, -2_709_820_035_785_812_506_i64),
            (i32::MIN, 255, i32::MAX, 7_254_248_326_551_643_980_i64),
        ];
        for (x, y, z, expected) in cases {
            assert_eq!(
                java_coordinate_random_oracle(x, y, z),
                expected,
                "Java oracle vector ({x}, {y}, {z})"
            );
            assert_eq!(
                coordinate_random(x, y, z),
                expected,
                "shared coordinate_random diverged at ({x}, {y}, {z})"
            );
        }

        // Negative controls prove the sample reaches the divergent region rather than merely
        // exercising coordinates whose x product still fits in i32.
        assert_eq!(
            widened_x_coordinate_random(686, 64, -31),
            java_coordinate_random_oracle(686, 64, -31),
            "the pre-overflow boundary is a non-divergent control"
        );
        assert_ne!(
            widened_x_coordinate_random(687, 64, -31),
            java_coordinate_random_oracle(687, 64, -31),
            "the first overflowing positive x must distinguish a widened multiply"
        );
        assert_ne!(
            widened_x_coordinate_random(-687, -20, 17),
            java_coordinate_random_oracle(-687, -20, 17),
            "the first overflowing negative x must distinguish a widened multiply"
        );
    }

    /// Independent spelling of Java's bytecode order: retain the low 32 bits of the `long`,
    /// reinterpret them as signed, arithmetic-shift, then abs/modulo.
    fn java_weighted_index_oracle(hash: i64, total: i32) -> i32 {
        let narrowed = i32::from_ne_bytes((hash as u32).to_ne_bytes());
        (narrowed >> 16).wrapping_abs().wrapping_rem(total)
    }

    #[test]
    fn weighted_variant_index_narrows_to_java_int_before_shifting() {
        let hash = coordinate_random(1, 64, 0);
        assert_eq!(hash, 8_738_263_850_476_356_458);
        assert_eq!(weighted_index(hash, 3), 0);
        assert_eq!(weighted_index(hash, 5), 0);
        assert_eq!(weighted_index(hash, 13), 9);

        let coordinates = [
            i32::MIN,
            -2_000_000,
            -1_000_000,
            -1,
            0,
            1,
            1_000_000,
            i32::MAX,
        ];
        for x in coordinates {
            for y in [-64, 0, 64, 255] {
                for z in coordinates {
                    let hash = coordinate_random(x, y, z);
                    for total in 1..=32_i32 {
                        assert_eq!(
                            weighted_index(hash, total),
                            java_weighted_index_oracle(hash, total),
                            "position ({x}, {y}, {z}), total {total}, hash {hash}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn signed_weight_totals_and_selection_keep_java_failure_modes() {
        let empty = BakedModel::default();
        let zero_total = vec![(0, empty.clone()), (0, empty.clone())];
        assert_eq!(java_weight_total(&zero_total), 0);
        assert!(
            std::panic::catch_unwind(|| weighted_index(173_333_959_360, 0)).is_err(),
            "negative control: Java WeightedBakedModel evaluates modulo zero rather than falling back"
        );

        let negative_total = vec![(-3, empty.clone()), (-2, empty)];
        assert_eq!(java_weight_total(&negative_total), -5);
        let index = weighted_index(173_333_959_360, -5);
        assert!((0..5).contains(&index));
        assert_eq!(weighted_candidate_index(&negative_total, index), None);

        // A zero candidate does not consume any of the cumulative index. This would select the
        // second model in Java too; changing `weight <= 0` into one makes this control fail.
        let mixed = vec![(0, BakedModel::default()), (2, BakedModel::default())];
        assert_eq!(weighted_candidate_index(&mixed, 0), Some(1));
        assert_eq!(weighted_candidate_index(&mixed, 1), Some(1));
    }

    #[test]
    fn actual_base_lookup_ignores_weighted_alternatives_for_damage_and_particles() {
        let stored = mc_data::block_by_name("fence").unwrap().default_state();
        let actual = mc_data::block::ActualBlockState::from_stored(stored);
        let mapping = state_mapper::resolve_actual(&actual);

        let base = BakedModel {
            ambient_occlusion: false,
            ..BakedModel::default()
        };
        let alternative = BakedModel {
            ambient_occlusion: true,
            ..BakedModel::default()
        };
        let assets = BakedAssets::from_parts_with_actual(
            Atlas {
                texture: Vec::new(),
                mipmaps: Vec::new(),
                w: 1,
                h: 1,
            },
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![(mapping.clone(), base)],
            vec![(mapping, vec![(1, alternative)])],
            Vec::new(),
            std::array::from_fn(|_| BakedModel::default()),
        );

        assert!(
            !actual_base_for(&assets, &actual)
                .expect("actual-state base model")
                .ambient_occlusion,
            "base-only lookup must not enter WeightedBakedModel alternative selection"
        );
        assert!(
            assets
                .actual_weighted
                .get(&state_mapper::resolve_actual(&actual))
                .expect("negative control has a weighted alternative")[0]
                .1
                .ambient_occlusion,
            "negative control: the weighted candidate differs from the returned base"
        );
    }

    #[test]
    fn atlas_registers_the_complete_java_variant_and_item_texture_delta() {
        // `ModelBakery.loadSprites` registers these from every variant/item model, even though
        // this port's executable geometry pass deliberately reaches fewer of those entries.
        // Test against the real jar: a synthetic fallback has a UV rectangle too, so compare
        // the decoded frame as well as the registry/count to catch a typo or a dropped name.
        assert_eq!(EXTRA_VANILLA_ATLAS_TEXTURES.len(), 45);
        let unique: HashSet<_> = EXTRA_VANILLA_ATLAS_TEXTURES.iter().copied().collect();
        assert_eq!(
            unique.len(),
            45,
            "the Java-registration delta has no aliases"
        );

        let jar = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../MCP-919-main/jars/versions/1.8.9/1.8.9.jar");
        assert!(
            jar.is_file(),
            "vanilla acceptance jar missing: {}",
            jar.display()
        );
        let mut manager = ResourceManager::new();
        manager.push_pack(Box::new(
            crate::ZipResourcePack::open(&jar).expect("open vanilla jar"),
        ));
        let (assets, report) = bake_all(&manager, 2048).expect("bake vanilla models");
        assert!(report.is_clean(), "{:?}", report.failures);

        for path in EXTRA_VANILLA_ATLAS_TEXTURES {
            let location = texture_location(path).expect("constant is a valid texture path");
            let stitched = assets
                .sprites
                .get(&location)
                .unwrap_or_else(|| panic!("Java-registered atlas texture missing: {path}"));
            assert_ne!(
                stitched.uv,
                UvRect::FULL,
                "{path} fell through sprite()'s blank placeholder"
            );
            let source = crate::sprite::load_sprite(&manager, &location)
                .unwrap_or_else(|error| panic!("{path} is not a vanilla atlas texture: {error}"));
            assert_eq!(
                (stitched.width, stitched.height, &stitched.frames[0].rgba),
                (source.width, source.height, &source.frames[0].rgba),
                "{path} was replaced by a fallback rather than its jar sprite"
            );
        }
    }

    #[test]
    fn unknown_sprite_uses_vanillas_shared_missing_checkerboard() {
        let sprite = missing_sprite_placeholder();
        assert_eq!(sprite.location, missing_texture_location());
        assert_eq!((sprite.width, sprite.height), (16, 16));
        assert_eq!(sprite.frames.len(), 1);
        assert_eq!(&sprite.frames[0].rgba[0..4], &[0xF8, 0x00, 0xF8, 0xFF]);
        let right_half = (8 * 4) as usize;
        assert_eq!(
            &sprite.frames[0].rgba[right_half..right_half + 4],
            &[0x00, 0x00, 0x00, 0xFF]
        );
    }

    #[test]
    fn the_real_bake_addresses_every_vanilla_block_model_and_cache_keeps_them() {
        let jar = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../MCP-919-main/jars/versions/1.8.9/1.8.9.jar");
        assert!(
            jar.is_file(),
            "vanilla acceptance jar missing: {}",
            jar.display()
        );
        let mut manager = ResourceManager::new();
        manager.push_pack(Box::new(
            crate::ZipResourcePack::open(&jar).expect("open vanilla jar"),
        ));
        let (assets, report) = bake_all(&manager, 2048).expect("bake vanilla models");
        assert!(report.is_clean(), "{:?}", report.failures);
        assert_eq!(
            assets.actual_models.len(),
            2_555,
            "all vanilla block ModelResourceLocations except item_frame's two manual entries"
        );
        assert!(
            assets
                .item_frame_models
                .iter()
                .all(|model| !model.is_empty()),
            "ModelBakery hand-registers item_frame#normal and #map for RenderItemFrame"
        );
        assert!(
            !crate::cache::model_eq(&assets.item_frame_models[0], &assets.item_frame_models[1]),
            "the map frame's background model must not alias the ordinary item frame"
        );

        let fence = mc_data::block_by_name("fence").unwrap().default_state();
        let mut connected = mc_data::block::ActualBlockState::from_stored(fence);
        assert!(connected.set("east", "true"));
        let key = state_mapper::resolve_actual(&connected);
        let fresh = assets
            .actual_models
            .get(&key)
            .expect("east-connected fence has its own baked model");
        let raw_key = state_mapper::resolve(fence);
        assert_ne!(
            key, raw_key,
            "negative control reaches beyond packed metadata"
        );
        assert!(
            !crate::cache::model_eq(
                fresh,
                assets
                    .actual_models
                    .get(&raw_key)
                    .expect("raw isolated fence model")
            ),
            "negative control: the derived lookup must select different geometry"
        );

        let snapshot = crate::cache::BakedSnapshot::of(&assets);
        let cached = crate::cache::BakedSnapshot::decode(&snapshot.encode())
            .expect("decode cache payload")
            .into_assets();
        assert_eq!(cached.actual_models.len(), 2_555);
        assert!(crate::cache::model_eq(
            cached
                .actual_models
                .get(&key)
                .expect("cached connected fence"),
            fresh
        ));
        for index in 0..2 {
            assert!(crate::cache::model_eq(
                &cached.item_frame_models[index],
                &assets.item_frame_models[index]
            ));
        }

        let item = |name: &str, metadata: u16| {
            let id = mc_data::item_by_name(name).expect("registered item");
            item_model_key(id, metadata)
        };
        let oak_door = assets
            .item_models
            .get(&item("wooden_door", 0))
            .expect("wooden_door must select item/oak_door");
        assert!(!oak_door.is_empty());

        let oak_planks = assets
            .item_models
            .get(&item("planks", 0))
            .expect("oak planks model");
        let dark_oak_planks = assets
            .item_models
            .get(&item("planks", 5))
            .expect("dark oak planks model");
        assert!(
            !crate::cache::model_eq(oak_planks, dark_oak_planks),
            "negative control: metadata 0 and 5 must not alias one registry-name model"
        );

        let black_dye = assets
            .item_models
            .get(&item("dye", 0))
            .expect("black dye model");
        let white_dye = assets
            .item_models
            .get(&item("dye", 15))
            .expect("white dye model");
        assert!(!black_dye.is_empty() && !white_dye.is_empty());
        assert!(
            !crate::cache::model_eq(black_dye, white_dye),
            "negative control: dye damage uses black-to-white order, not one shared model"
        );

        assert!(
            !assets.item_models.contains_key(&item("stone_slab", 2)),
            "vanilla ModelBakery never loads legacy item/old_wood_slab"
        );
        assert_eq!(
            cached.item_models.len(),
            assets.item_models.len(),
            "cache must retain every metadata-keyed item model"
        );
        assert!(crate::cache::model_eq(
            cached
                .item_models
                .get(&item("planks", 5))
                .expect("cached dark-oak planks"),
            dark_oak_planks
        ));
    }

    /// Connected textures over a stack with no CTM rules stitch
    /// nothing extra, so the bake is the vanilla one, plus a table with the pane-edge slots.
    #[test]
    fn a_ctm_bake_without_rules_is_the_vanilla_bake_plus_a_pane_table() {
        let jar = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../MCP-919-main/jars/versions/1.8.9/1.8.9.jar");
        assert!(
            jar.is_file(),
            "vanilla acceptance jar missing: {}",
            jar.display()
        );
        let mut manager = ResourceManager::new();
        manager.push_pack(Box::new(
            crate::ZipResourcePack::open(&jar).expect("open vanilla jar"),
        ));
        let (vanilla, _) = bake_all_with_ctm(&manager, 2048, 4, false).expect("vanilla bake");
        let (with_ctm, report) = bake_all_with_ctm(&manager, 2048, 4, true).expect("CTM bake");
        assert!(report.is_clean(), "{:?}", report.failures);
        assert!(vanilla.ctm().is_none(), "off bakes no table");
        let table = with_ctm.ctm().expect("on always bakes a table");
        assert_eq!(table.rule_count(), 0);
        let state = |name: &str| mc_data::block_by_name(name).unwrap().default_state();
        assert!(
            table.slot(state("glass_pane")).is_some(),
            "panes keep the edge merge"
        );
        assert!(table.slot(state("glass")).is_none(), "no rule, no slot");
        assert!(
            crate::cache::BakedSnapshot::of(&vanilla) == crate::cache::BakedSnapshot::of(&with_ctm),
            "the same atlas, sprites and models"
        );
    }

    /// The override is this thread's
    /// alone, nests, restores what the thread saw before (also on a panic), and never moves the
    /// installed registry's generation. No test in this binary installs the global registry, so
    /// "what the thread saw before" is stable here.
    #[test]
    fn a_thread_registry_override_is_thread_scoped_and_restores() {
        let stone = mc_data::block_by_name("minecraft:stone")
            .expect("stone")
            .default_state();
        let marked = |mark: f32| -> &'static BakedAssets {
            let mut models = vec![empty_model(); mc_data::STATE_COUNT];
            models[stone.id as usize].particle_uv = Some(UvRect {
                min_u: mark,
                max_u: mark,
                min_v: mark,
                max_v: mark,
            });
            let atlas = Atlas {
                texture: vec![0; 4],
                mipmaps: Vec::new(),
                w: 1,
                h: 1,
            };
            Box::leak(Box::new(BakedAssets::from_parts(
                atlas,
                Vec::new(),
                models,
                Vec::new(),
                Vec::new(),
            )))
        };
        let (a, b) = (marked(0.123_45), marked(0.678_9));
        let mark = move || model_for(stone).particle_uv.map(|uv| uv.min_u);
        let generation = installed_generation();
        let global = mark();
        assert!(!thread_registry_overridden());

        let seen = with_thread_registry(a, || {
            assert!(thread_registry_overridden());
            let outer = mark();
            let other_thread = std::thread::spawn(move || (thread_registry_overridden(), mark()))
                .join()
                .expect("probe thread");
            assert_eq!(
                other_thread,
                (false, global),
                "another thread saw the override"
            );
            let nested = with_thread_registry(b, mark);
            (outer, nested, mark())
        });
        assert_eq!(seen, (Some(0.123_45), Some(0.678_9), Some(0.123_45)));
        assert!(
            !thread_registry_overridden(),
            "the override outlived its scope"
        );
        assert_eq!(
            mark(),
            global,
            "the thread did not get the global registry back"
        );

        let caught = std::panic::catch_unwind(|| with_thread_registry(a, || panic!("boom")));
        assert!(caught.is_err());
        assert!(
            !thread_registry_overridden(),
            "a panic left the override in place"
        );
        assert_eq!(mark(), global);
        assert_eq!(
            installed_generation(),
            generation,
            "the override moved the generation"
        );
    }
}
