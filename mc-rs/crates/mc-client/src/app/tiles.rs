//! The tile-entity seam: packets in, geometry out.
//!
//! Everything here is an `impl Client { .. }` block, so the tile-entity seam reads as one file
//! instead of as scattered arms of [`super::driver`].
//!
//! # The three things that were missing
//!
//! 1. **A store.** [`mc_world::tileent`] now holds it. The *identity* of a tile entity comes
//!    from the block (`Chunk.createNewTileEntity`), so a chest draws with nothing in the store;
//!    only decoration (sign text, banner patterns, skull owner, spawner mob, flower-pot plant,
//!    beacon levels, and command-block logic for the editor) arrives on the wire, as description
//!    packets pushed behind the chunk.
//! 2. **A pass.** [`mc_render::entity::tilepass::TilePhase`] and the long-dead
//!    [`mc_render::entity::tile::BeaconPhase`] are registered in [`Client::install`] at
//!    [`PhaseOrder::ENTITIES`], right after the entity phase — which is where
//!    `RenderGlobal.renderEntities` runs the TESR loop, i.e. after opaque terrain and before
//!    translucent.
//! 3. **The wire.** `TileEntityUpdated`, `SignUpdated`, `OpenSignEditor` and `BlockAction` were
//!    all in `driver.rs`'s "no reader yet" list. They have real arms now.
//!
//! # Why a client-side index
//!
//! Vanilla collects the visible tile entities from the render chunks the mesher already walked
//! (`RenderChunk.getCompiledChunk().getTileEntities()`, `RenderGlobal.java:695`). `mc-render`'s
//! mesher has no such collection and belongs to another Wave-1 author, so [`TileIndex`] answers
//! the same question by scanning chunk columns for tile-entity blocks and caching the result per
//! column. The refresh policy is in [`TileIndex::refresh`]: newly loaded columns are indexed on
//! a budget, the nine columns around the player are re-scanned every tick so a block the player
//! just placed appears immediately, and everything else rotates through on a rolling cursor.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use mc_data::block::ActualBlockState;
use mc_data::{BlockId, BlockState};
use mc_math::{BlockPos, ChunkPos, DVec3, Facing, Vec3};
use mc_render::entity::tile::{
    self, enchant_book_matrix, enchant_page_flips, sign_text_matrix, spawner_matrix, BeaconPhase,
};
use mc_render::entity::tilepass::{
    builtin_tile_item_kind, builtin_tile_item_kind_for_model, emit_baked_block_model_lit,
    light_brightness, lit_into, remap_destroy_stage_uvs, BuiltinSkullOwner, BuiltinTileItem,
    BuiltinTileItemKind, TileBlend, TileModels, TilePhase, TileVertex,
};
use mc_render::entity::{EntityVertex, NameplateGlyphVertex, NameplatePass, NameplateVertex};
use mc_render::{Camera, Gpu};
use mc_ui::font::{GlyphQuad, SolidRect, TextSink};
use mc_ui::screens::sign::{EditSignScreen, SignAction};
use mc_world::tileent::{
    piston_facing, PistonState, PistonWorldBorder, TileData, TileKind, PISTON_HEAD_ID,
    STICKY_PISTON_ID,
};
use mc_world::World;

use super::driver::{Client, Ui};

/// How many not-yet-indexed columns [`TileIndex::refresh`] scans per tick.
///
/// A column scan walks its non-empty sections' block arrays; at eight per tick a fresh join's
/// several hundred columns are all indexed inside a couple of seconds, which is well under the
/// time the terrain mesher needs for the same columns.
const INDEX_BUDGET: usize = 8;

/// How many already-indexed columns are re-scanned per tick, on a rolling cursor.
///
/// This is the backstop for a block change outside the player's own nine columns — another
/// player placing a chest across the street. `mc-client` has no block-change event to hook (the
/// dispatcher applies `S23`/`S22` straight into the world), and the dirty set that would be the
/// exact signal is drained by the mesher before [`Client::tick`]'s tail runs.
const REFRESH_BUDGET: usize = 4;

/// `System.currentTimeMillis()` sample used while applying a client-local piston block action.
/// The environment owns S44 transition endpoints; this module only snapshots their current
/// diameter before passing the minimal border projection into `mc-world`.
#[must_use]
fn piston_border_clock_millis() -> i64 {
    let millis = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as i128,
        Err(error) => -(error.duration().as_millis() as i128),
    };
    millis.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// The client-only identity sample for a mob spawner's cached miniature.
///
/// java: `MobSpawnerBaseLogic.func_180612_a` keeps one entity instance per spawner and hands it
/// to `RendererLivingEntity`.  The play protocol does not carry this entity's render fields, so
/// the tile pass owns the small cache instead of borrowing a server-authoritative world entity.
/// Crucially, `spawnNewEntity(entity, false)` does **not** add that cached entity to the world, and
/// `TileEntityMobSpawnerRenderer.renderMob` only calls `setLocationAndAngles` before rendering.
/// No `onUpdate` ever reaches the cached entity: its constructor pose is therefore age/limb/head
/// zero on every frame, with only the caller's ordinary `partialTicks` contributing to `age`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SpawnerRenderAnimation {
    /// The Java entity id is not sent in tile payloads; keep the render adapter deterministic.
    entity_id: i32,
}

/// The few renderer-facing entity fields that are carried by a raw `SpawnData` compound but are
/// intentionally not part of `mc_world::SpawnerSpawnData`'s NBT-free shared state.
///
/// `MobSpawnerBaseLogic.spawnNewEntity` overlays *every* SpawnData key on a freshly serialized
/// entity before `readFromNBT`.  Keep these values beside the client render snapshot so a packet
/// can affect a miniature's texture/placement without turning the world store into an NBT
/// dependency or pretending that a client-only cached entity is gameplay state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SpawnerRenderAppearance {
    /// `Entity.getCustomNameTag`, used by RenderRabbit (Toast), LayerSheepWool (jeb_) and the
    /// Dinnerbone/Grumm upside-down renderer branch.
    custom_name: Option<String>,
    /// `EntityWither.getInvulTime`, which selects the invulnerable texture and spawn scale.
    wither_invul_time: i32,
    /// `EntityBat.getIsBatHanging`, restored from the entity's persisted `BatFlags` byte.
    bat_hanging: bool,
    /// `EntityZombie.isConverting`, restored from `ConversionTime > -1`.
    converting: bool,
}

/// One visible spawner miniature collected before the tile render pass borrows its models.
///
/// Keeping the Java-shaped tuple behind a local alias makes the collection boundary readable
/// without changing the order or ownership of any renderer inputs.
type SpawnerUpload = (
    BlockPos,
    String,
    f64,
    f64,
    Option<mc_world::tileent::SpawnerSpawnData>,
    SpawnerRenderAppearance,
    SpawnerRenderAnimation,
);

#[derive(Clone, Debug, PartialEq)]
struct SpawnerAnimationCache {
    entity_id: String,
    animation: SpawnerRenderAnimation,
}

impl SpawnerAnimationCache {
    fn new(entity_id: &str) -> Self {
        Self {
            entity_id: entity_id.to_owned(),
            animation: SpawnerRenderAnimation::default(),
        }
    }
}

/// The stitched atlas key used by baked block models in every renderer-owned phase.
///
/// java: `TextureMap.locationBlocksTexture`, which `TileEntityPistonRenderer` binds before it
/// dispatches the carried state to `BlockModelRenderer`. `TilePhase` owns its own texture table,
/// so the atlas is uploaded lazily when the first piston reaches this path.
const BLOCK_ATLAS_TEXTURE: &str = "mc-render/entity/block-atlas";

/// Convert an item stack that selected vanilla's `builtin/entity` model into the small typed
/// payload consumed by [`mc_render::entity::tilepass::TileModels::emit_builtin_item`].
///
/// java: `TileEntityItemStackRenderer.renderByItem` and `TileEntityBanner.setItemValues`.  The
/// three chest families carry no item-specific NBT; skull damage selects its head type; banners
/// take their base dye and ordered pattern list from `BlockEntityTag` when present.  Keeping this
/// adapter in `mc-client` avoids making the render crate depend on the wire `ItemStack` type and
/// leaves the eventual GUI/held-item caller free to choose its own phase and texture upload.
#[must_use]
pub fn builtin_tile_item_from_stack(stack: &mc_protocol::ItemStack) -> Option<BuiltinTileItem> {
    // Keep the original no-context entry point for callers that
    // only have an item stack.  The driver supplies the live tab-list below when it has one.
    builtin_tile_item_from_stack_with_players(stack, None)
}

/// Decode a builtin item while allowing a legacy player-skull name to use a profile already
/// present in the client tab list.
///
/// Vanilla's multiplayer `TileEntitySkull.updateGameprofile` cannot resolve a name on its own:
/// `MinecraftServer.getServer()` is null on the client, so the name-only profile falls through
/// to `DefaultPlayerSkin.getDefaultSkinLegacy()` (Steve).  The player list is the one local,
/// already-authoritative profile source available to this port.  A missing row deliberately
/// remains an incomplete [`BuiltinSkullOwner`] rather than causing a name lookup or inventing a
/// UUID.
#[must_use]
pub fn builtin_tile_item_from_stack_with_players(
    stack: &mc_protocol::ItemStack,
    players: Option<&crate::sync::PlayerList>,
) -> Option<BuiltinTileItem> {
    let item_id = u16::try_from(stack.item_id).ok()?;
    builtin_tile_item_from_kind(stack, builtin_tile_item_kind(item_id)?, players)
}

/// Decode a builtin item when the caller has also resolved the selected baked model.
///
/// This is the model-aware counterpart to [`builtin_tile_item_from_stack_with_players`].  Once a
/// model is known to be `builtin/entity`, Java's renderer sends *any* unexpected item family to
/// the ordinary chest instance.  A normal model keeps the strict five-family table and returns
/// `None`, so an unknown network item cannot silently become a chest merely because its id was
/// unrecognised.
#[must_use]
pub fn builtin_tile_item_from_stack_with_players_and_model(
    stack: &mc_protocol::ItemStack,
    players: Option<&crate::sync::PlayerList>,
    model: &mc_assets::BakedModel,
) -> Option<BuiltinTileItem> {
    let startup_empty_model = model.is_empty() && !model.gui_3d && model.particle_uv.is_none();
    if !mc_render::entity::tilepass::is_builtin_entity_model(model) && !startup_empty_model {
        return None;
    }
    let item_id = u16::try_from(stack.item_id).ok()?;
    builtin_tile_item_from_kind(
        stack,
        builtin_tile_item_kind_for_model(item_id, model)?,
        players,
    )
}

fn builtin_tile_item_from_kind(
    stack: &mc_protocol::ItemStack,
    kind: BuiltinTileItemKind,
    players: Option<&crate::sync::PlayerList>,
) -> Option<BuiltinTileItem> {
    match kind {
        BuiltinTileItemKind::Chest => Some(BuiltinTileItem::Chest { trapped: false }),
        BuiltinTileItemKind::TrappedChest => Some(BuiltinTileItem::Chest { trapped: true }),
        BuiltinTileItemKind::EnderChest => Some(BuiltinTileItem::EnderChest),
        BuiltinTileItemKind::Banner => Some(BuiltinTileItem::Banner {
            state: banner_item_state(stack),
        }),
        BuiltinTileItemKind::Skull => {
            // TileEntityItemStackRenderer only consults SkullOwner for player heads.  A malformed
            // owner follows the owner-less renderer's safe Steve path here; a complete profile or
            // a locally-known legacy name can be handed to SkinCache without a blocking lookup.
            let skull_type = u8::try_from(stack.damage.max(0)).unwrap_or(u8::MAX);
            if skull_type == 3 {
                if let Some(owner) = player_skull_owner(stack, players) {
                    return Some(BuiltinTileItem::PlayerSkull { skull_type, owner });
                }
            }
            Some(BuiltinTileItem::Skull { skull_type })
        }
    }
}

/// Promote a legacy string `SkullOwner` on a mutable item stack using an already-known local
/// profile when one exists.
///
/// java: `TileEntityItemStackRenderer.renderByItem` replaces the root string with
/// `NBTUtil.writeGameProfile(new NBTTagCompound(), profile)` after
/// `TileEntitySkull.updateGameprofile`.  The client cannot use the server's
/// `PlayerProfileCache`, so this helper only consults the supplied tab-list/local profile and
/// otherwise writes the same name-only compound that Java leaves behind on a multiplayer client.
/// Empty, missing, wrong-typed, and already-compound owners are controls rather than errors: they
/// return `None` (or the already decoded profile) and leave unrelated stack NBT untouched.
#[must_use]
pub fn normalize_legacy_skull_owner_in_stack(
    stack: &mut mc_protocol::ItemStack,
    players: Option<&crate::sync::PlayerList>,
) -> Option<mc_auth::GameProfile> {
    let root = stack.nbt.as_mut()?;
    normalize_legacy_skull_owner_in_nbt(root, players)
}

/// The NBT half of [`normalize_legacy_skull_owner_in_stack`].  Keeping this separate lets the
/// immutable render adapters work on a short-lived clone while the inventory/held-item owners
/// that have a mutable stack can persist Java's in-place promotion.
fn normalize_legacy_skull_owner_in_nbt(
    root: &mut mc_nbt::NbtCompound,
    players: Option<&crate::sync::PlayerList>,
) -> Option<mc_auth::GameProfile> {
    let name = match root.get("SkullOwner") {
        Some(mc_nbt::Nbt::String(name)) if !name.is_empty() => name.clone(),
        Some(mc_nbt::Nbt::Compound(owner)) => return mc_auth::read_game_profile_from_nbt(owner),
        _ => return None,
    };
    let resolved = players.and_then(|players| legacy_player_profile(&name, players));
    mc_auth::normalize_legacy_skull_owner(root, resolved.as_ref())
}

/// Resolve an item-stack `SkullOwner` compound into the existing appearance skin cache.
///
/// java: `TileEntityItemStackRenderer.renderByItem:31-48` reads the compound with
/// `NBTUtil.readGameProfileFromNBT`; `TileEntitySkullRenderer.renderSkull:73-94` then asks the
/// skin manager for the profile's cached texture and uses the UUID-derived default until it is
/// ready.  The item renderers in `mc-ui` and `entrender` cannot borrow `AppearanceState`, so the
/// request goes through [`mc_render::entity::skin::request_shared_skin`].  The ordinary client
/// `SkinCache::poll` drains that queue, preserving one async fetch/cache implementation.
fn player_skull_owner(
    stack: &mc_protocol::ItemStack,
    players: Option<&crate::sync::PlayerList>,
) -> Option<BuiltinSkullOwner> {
    // Render callers borrow the wire stack immutably.  Work on a clone so the same normalizer
    // drives both the persistent mutable seam above and this allocation-free descriptor path;
    // Java's observable string-to-compound mutation is therefore retained whenever a caller can
    // hand us `&mut ItemStack`, without forcing every UI draw call to take mutable ownership.
    //
    // A compound owner is only *read* by the normaliser, so the clone is skipped for it (SKIN-4).
    if let Some(mc_nbt::Nbt::Compound(owner)) = stack.nbt.as_ref()?.get("SkullOwner") {
        // (`MC_W18_OA_HEAD_PROFILES=0` turns it off) The renderer reads
        // three fields of the profile; take them borrowed instead of copying the whole profile.
        if super::driver::mods::w18_oa::head_profiles() {
            return Some(player_skull_owner_from_parts(head_parts_from_nbt(owner)?));
        }
        let profile = mc_auth::read_game_profile_from_nbt(owner)?;
        return Some(player_skull_owner_from_profile(&profile));
    }
    let mut normalized = stack.clone();
    let profile = normalize_legacy_skull_owner_in_stack(&mut normalized, players)?;
    Some(player_skull_owner_from_profile(&profile))
}

/// The three fields of a player-head owner profile that the head
/// renderer reads ([`player_skull_owner_from_parts`]): the parsed id, the name, and the value of
/// the first `textures` property.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeadProfileParts<'a> {
    id: Option<uuid::Uuid>,
    name: &'a str,
    textures: Option<&'a str>,
}

/// [`HeadProfileParts`] of an already decoded profile.
fn head_parts_from_profile(profile: &mc_auth::GameProfile) -> HeadProfileParts<'_> {
    HeadProfileParts {
        id: profile.id,
        name: &profile.name,
        textures: profile
            .properties
            .iter()
            .find(|property| property.name == "textures")
            .map(|property| property.value.as_str()),
    }
}

/// [`HeadProfileParts`] read straight out of a `SkullOwner` compound,
/// borrowed, instead of `mc_auth::read_game_profile_from_nbt` copying the name and every
/// property's value and signature for each head every frame. Rule for rule the reader's: `None`
/// when the name and the id are both missing or empty, Java's `UUID.fromString` for the id, and
/// for the first `textures` property the reader's walk over `Properties` in insertion order
/// (non-list values, lists of another element type and non-compound elements skipped, a missing
/// `Value` read as empty).
fn head_parts_from_nbt(owner: &mc_nbt::NbtCompound) -> Option<HeadProfileParts<'_>> {
    let name = owner.get_str("Name").unwrap_or_default();
    let id_text = owner.get_str("Id");
    if name.is_empty() && !id_text.is_some_and(|value| !value.is_empty()) {
        return None;
    }
    let textures = owner.get_compound("Properties").and_then(|by_name| {
        by_name.iter().find_map(|(key, values)| {
            let mc_nbt::Nbt::List(values) = values else {
                return None;
            };
            if key != "textures"
                || (!values.is_empty() && values.tag_type() != mc_nbt::TagType::Compound)
            {
                return None;
            }
            values
                .into_iter()
                .find_map(mc_nbt::Nbt::as_compound)
                .map(|value| value.get_str("Value").unwrap_or_default())
        })
    });
    Some(HeadProfileParts {
        id: id_text.and_then(mc_auth::java_uuid_from_string),
        name,
        textures,
    })
}

/// Find a locally-known profile for a legacy owner name, matching `PlayerProfileCache`'s
/// case-insensitive username key while preferring an exact spelling when a malformed list has
/// duplicate rows.  This never performs I/O and never guesses a UUID.
fn legacy_player_profile(
    name: &str,
    players: &crate::sync::PlayerList,
) -> Option<mc_auth::GameProfile> {
    let info = players
        .iter()
        .find(|info| info.username == name)
        .or_else(|| {
            players
                .iter()
                .find(|info| info.username.eq_ignore_ascii_case(name))
        })?;
    Some(mc_auth::GameProfile {
        id: Some(info.uuid),
        name: info.username.clone(),
        properties: info
            .properties
            .iter()
            .map(|property| mc_auth::GameProfileProperty {
                name: property.name.clone(),
                value: property.value.clone(),
                signature: property.signature.clone(),
            })
            .collect(),
    })
}

thread_local! {
    /// SKIN-4 / TILE-2: every visible player head resolves its owner profile every frame, and the
    /// base64 + JSON decode of the `textures` property dominated that. The decode is pure, so it
    /// is done once per distinct property value; the skin request below still runs per frame.
    static SKULL_TEXTURES: std::cell::RefCell<
        mc_render::entity::skin::TexturesPropertyMemo<Option<super::appearance::ProfileTextures>>,
    > = std::cell::RefCell::new(mc_render::entity::skin::TexturesPropertyMemo::new());
}

/// java: `MinecraftSessionService.getTextures(profile, false)`: decode the profile's first
/// `textures` property (memoised by value, SKIN-4) into what `skinCacheLoader` caches, the SKIN's
/// hash-named key and URL plus the arm model. `textures` is that property's value
/// ([`HeadProfileParts::textures`]).
fn skull_profile_textures(
    textures: Option<&str>,
    fallback: mc_render::entity::SkinModel,
) -> mc_render::entity::skin::PlayerHeadTextures {
    let (skin, model) = match textures {
        Some(value) => SKULL_TEXTURES.with(|memo| {
            memo.borrow_mut().with(
                value,
                super::appearance::decode_textures_property,
                |textures| {
                    let model = textures
                        .as_ref()
                        .map_or(fallback, |textures| textures.model);
                    let skin = textures
                        .as_ref()
                        .and_then(|textures| textures.skin_url.as_deref())
                        .and_then(|url| {
                            super::appearance::skin_texture_key_for_url(url)
                                .map(|key| (key, url.to_owned()))
                        });
                    (skin, model)
                },
            )
        }),
        None => (None, fallback),
    };
    mc_render::entity::skin::PlayerHeadTextures { skin, model }
}

/// <<W18-PA:skull-descriptor>> java: `TileEntitySkullRenderer.renderSkull`'s
/// `getSkinManager().loadSkinFromCache(profile)` then `loadSkin(texture)`: the profile goes
/// through the one descriptor cache every head renderer shares
/// ([`mc_render::entity::skin::PlayerHeadProfiles`], 15 s expire-after-access, keyed by id and
/// name), and the skin is named by its texture hash.
fn player_skull_owner_from_profile(profile: &mc_auth::GameProfile) -> BuiltinSkullOwner {
    player_skull_owner_from_parts(head_parts_from_profile(profile))
}

/// [`player_skull_owner_from_profile`] on the three profile fields it reads (split out so a head can hand them over borrowed from its item NBT).
fn player_skull_owner_from_parts(parts: HeadProfileParts<'_>) -> BuiltinSkullOwner {
    let Some(id) = parts.id else {
        // Java's client-side legacy promotion starts with GameProfile(null, name). The renderer
        // then calls EntityPlayer.getUUID, which derives UUID.nameUUIDFromBytes from
        // `OfflinePlayer:<name>` before selecting the default Steve/Alex sheet.
        let offline_id = mc_auth::offline::offline_uuid(parts.name);
        let fallback = mc_render::entity::SkinModel::default_for_uuid(offline_id.as_u128());
        return BuiltinSkullOwner::legacy(parts.name, fallback);
    };
    let fallback = mc_render::entity::SkinModel::default_for_uuid(id.as_u128());
    mc_render::entity::skin::player_head_textures(
        id.as_u128(),
        parts.name,
        || skull_profile_textures(parts.textures, fallback),
        |textures| match &textures.skin {
            Some((cache_key, url)) => {
                mc_render::entity::skin::request_shared_skin(cache_key, url, textures.model);
                BuiltinSkullOwner::new(cache_key.as_str(), textures.model, fallback)
            }
            // java: no SKIN in the textures map: `DefaultPlayerSkin.getDefaultSkin(uuid)`.
            None => BuiltinSkullOwner::new(
                super::appearance::skin_texture_key(id),
                textures.model,
                fallback,
            ),
        },
    )
}

/// Emit a builtin tile item after decoding its wire stack.
///
/// This is intentionally a reusable render-side entry point rather than a live UI call: the
/// inventory renderer's glint/overlay ownership is active elsewhere.  The returned pair gives a
/// future caller both the decoded descriptor (needed by the banner compositor) and the texture
/// key expected by the emitted geometry.
#[must_use]
pub fn emit_builtin_tile_item(
    models: &mut TileModels,
    stack: &mc_protocol::ItemStack,
    parent: mc_render::entity::Mat4,
    out: &mut Vec<EntityVertex>,
) -> Option<(BuiltinTileItem, String)> {
    let item = builtin_tile_item_from_stack(stack)?;
    let texture = models.emit_builtin_item(&item, parent, out)?;
    Some((item, texture))
}

/// `TileEntityBanner.setItemValues`'s state extraction from an item stack.
///
/// `Base` uses Java's broad numeric `hasKey(..., 99)`/`getInteger` pair and therefore accepts all
/// primitive numeric NBT tags.  Pattern entries still require a compound and a known pattern id,
/// while the renderer itself applies the later sixteen-pattern cap.
#[must_use]
fn banner_item_state(stack: &mc_protocol::ItemStack) -> mc_world::BannerState {
    let mut state = mc_world::BannerState {
        base_color: stack.damage.max(0) as i32 & 15,
        patterns: Vec::new(),
        initialized: true,
    };
    let Some(tag) = stack
        .nbt
        .as_ref()
        .and_then(|root| root.get_compound("BlockEntityTag"))
    else {
        return state;
    };

    // `hasKey("Base", 99)` is numeric-only, but getInteger then performs Java primitive
    // conversion. `get_i32` is the matching mc-nbt accessor.
    if tag.contains_numeric_type("Base") {
        state.base_color = tag.get_i32("Base").unwrap_or(0);
    }

    // Java's getTagList(key, 10) returns an empty list when the value is not a compound list.
    // mc-nbt exposes the list elements directly, so requiring each element to be a compound is
    // the equivalent type gate here.
    if let Some(patterns) = tag.get_list("Patterns") {
        for entry in patterns {
            let Some(pattern) = entry.as_compound() else {
                continue;
            };
            let Some(id) = pattern.get_str("Pattern") else {
                continue;
            };
            if mc_world::tileent::banner_pattern_name(id).is_none() {
                continue;
            }
            state
                .patterns
                .push((id.to_owned(), pattern.get_i32("Color").unwrap_or(0)));
        }
    }
    state
}

/// Builds the client-only piston-head state used by `TileEntityPistonRenderer` while a head is
/// moving. `short` is not represented in the packed metadata; it is the property Java mutates
/// immediately before the model lookup.
///
/// java: `BlockPistonExtension.TYPE`, `FACING` and `SHORT` assignments at
/// `TileEntityPistonRenderer.renderTileEntityAt:54-64`.
#[must_use]
fn piston_head_state(sticky: bool, facing: Facing, short: bool) -> ActualBlockState {
    let metadata = facing.index() | (u8::from(sticky) << 3);
    let stored = BlockState::new(BlockId(PISTON_HEAD_ID), metadata);
    let mut state = ActualBlockState::from_stored(stored);
    // Every registered piston-head state declares this property. Keep the check debug-only so a
    // generated registry mismatch is visible to tests/debuggers without making a custom pack's
    // malformed state able to crash the render loop. The write itself must stay outside the
    // assertion: release builds compile the whole `debug_assert!` condition out, which left
    // `short` at the stored representative's `false` in the installed client.
    let declared = state.set("short", if short { "true" } else { "false" });
    debug_assert!(declared);
    state
}

/// Emits one moving piston through the same three branches as the vanilla TESR.
///
/// The world has already installed the moving carrier block and its [`PistonState`], so this
/// helper only performs the renderer's progress guard, offset interpolation, carried-model lookup
/// and retracting-head synthesis. `rel` is the camera-relative origin of the carrier block; baked
/// block geometry itself stays in `0..1` local coordinates.
///
/// java: `TileEntityPistonRenderer.renderTileEntityAt:27-72`.
fn visit_piston_models(
    piston: PistonState,
    rel: Vec3,
    partial_ticks: f32,
    mut emit: impl FnMut(BlockState, &mc_assets::BakedModel, Vec3, bool),
) {
    let progress = piston.progress_at(partial_ticks);
    // java: `block.getMaterial() != Material.air && te.getProgress(partialTicks) < 1.0F`.
    if piston.piston_state.props().is_air() || progress >= 1.0 {
        return;
    }

    let (offset_x, offset_y, offset_z) = piston.offset_at(partial_ticks);
    let moving = rel + Vec3::new(offset_x, offset_y, offset_z);
    let carried = piston.piston_state.block().0;

    if carried == PISTON_HEAD_ID && progress < 0.5 {
        // Java preserves the carried head's sticky/facing properties and only sets SHORT here.
        let sticky = piston.piston_state.meta() & 8 != 0;
        // `iblockstate.withProperty(SHORT, true)` leaves FACING untouched.  The carried head
        // can have been pushed by a differently-oriented piston, so using the carrier's travel
        // direction here would rotate that reachable head before it is half extended.
        let head_facing = piston_facing(piston.piston_state.meta()).unwrap_or(piston.facing);
        let head = piston_head_state(sticky, head_facing, true);
        emit(
            head.stored(),
            mc_assets::model::model_for_actual(&head),
            moving,
            true,
        );
    } else if piston.should_head_be_rendered && !piston.extending {
        // Retraction draws a synthetic head at the moving offset, then resets the translation
        // before drawing the carried piston body. This intentionally keeps Java's ignored
        // `iblockstate.withProperty(EXTENDED, true)` call out of the body state.
        let sticky = carried == STICKY_PISTON_ID;
        let head = piston_head_state(sticky, piston.facing, progress >= 0.5);
        emit(
            head.stored(),
            mc_assets::model::model_for_actual(&head),
            moving,
            true,
        );
        emit(
            piston.piston_state,
            mc_assets::model_for(piston.piston_state),
            rel,
            true,
        );
    } else {
        emit(
            piston.piston_state,
            mc_assets::model_for(piston.piston_state),
            moving,
            false,
        );
    }
}

#[cfg(test)]
fn emit_piston(piston: PistonState, rel: Vec3, partial_ticks: f32, out: &mut Vec<EntityVertex>) {
    visit_piston_models(piston, rel, partial_ticks, |_, model, translation, _| {
        mc_render::entity::tilepass::emit_baked_block_model(
            model,
            mc_render::entity::Mat4::from_translation(translation),
            out,
        );
    });
}

/// Piston counterpart to [`emit_piston`] that keeps Java's per-quad packed light/AO state.
#[allow(clippy::too_many_arguments)]
fn emit_piston_lit(
    world: &World,
    pos: BlockPos,
    piston: PistonState,
    rel: Vec3,
    partial_ticks: f32,
    settings: mc_render::terrain::MeshSettings,
    anaglyph: bool,
    out: &mut Vec<TileVertex>,
) {
    visit_piston_models(
        piston,
        rel,
        partial_ticks,
        |state, model, translation, check_sides| {
            emit_baked_block_model_lit(
                world,
                state,
                pos,
                model,
                mc_render::entity::Mat4::from_translation(translation),
                check_sides,
                settings,
                anaglyph,
                out,
            );
        },
    );
}

/// The exact client-side `EnumParticleTypes.NOTE` request made by
/// `BlockNote.onBlockEventReceived`.  `eventParam` is a hue fraction here, not a movement
/// velocity: `EntityNoteFX.Factory` passes its `xSpeedIn` through to `EntityNoteFX`, whose three
/// phase-shifted sine channels turn it into the rainbow note colour.
///
/// Keeping the coordinate conversion beside the block-action handler makes the two unusual
/// constants (`+1.2` and `/24.0`) independently testable without constructing a socket-backed
/// [`Client`].
#[must_use]
fn note_particle_parameters(position: BlockPos, note: u8) -> (DVec3, DVec3) {
    (
        DVec3::new(
            f64::from(position.x) + 0.5,
            f64::from(position.y) + 1.2,
            f64::from(position.z) + 0.5,
        ),
        DVec3::new(f64::from(note) / 24.0, 0.0, 0.0),
    )
}

/// Whether today is one of the three days `TileEntityChestRenderer` swaps the chest texture.
///
/// java: its constructor (`:24-31`) — `calendar.get(2) + 1 == 12 && calendar.get(5) >= 24 &&
/// calendar.get(5) <= 26`, sampled once at construction, which is what this is called at.
///
/// The civil-calendar arithmetic is the standard `civil_from_days`, written out rather than
/// pulled in: this is one boolean per launch, and the wave's rules freeze every `Cargo.toml`.
#[must_use]
pub fn christmas_today() -> bool {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let (month, day) = civil_from_days(secs.div_euclid(86_400));
    mc_render::entity::tile::is_christmas(month, day)
}

/// Read the absolute millisecond clock used by the end-portal texture scroll.
///
/// java: `Minecraft.getSystemTime() % 700000L` in
/// `TileEntityEndPortalRenderer.renderTileEntityAt:78`.  This must not be derived from an
/// [`std::time::Instant`] created when the Rust process first reaches the renderer: that makes
/// the visible phase depend on which frame happened to initialise the phase and cannot replay a
/// Java capture.  `SystemTime` supplies the process-independent millisecond domain; the render
/// pass samples it once so every portal in a frame advances in lockstep, just like Java's loop.
#[must_use]
fn end_portal_system_time_millis() -> u64 {
    end_portal_system_time_millis_with(|| {
        let live = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        // A benchmark's pinned clock, else the wall clock.
        u128::from(mc_render::entity::item::animation_clock_ms(
            live.min(u128::from(u64::MAX - 1)) as u64,
        ))
    })
}

/// Convert one injected clock reading into the Java millisecond domain.
///
/// The closure is deliberately the seam rather than a global mutable test clock: production
/// code reads `SystemTime`, while tests can provide an exact value (including the 700,000 ms
/// wrap boundary) without racing other tests or changing process-wide state.
#[must_use]
fn end_portal_system_time_millis_with(read: impl FnOnce() -> u128) -> u64 {
    read().min(u128::from(u64::MAX)) as u64
}

/// `(month, day)` for a count of days since 1970-01-01. Proleptic Gregorian, valid for any
/// `i64` a system clock can produce.
fn civil_from_days(days: i64) -> (u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (month, day)
}

/// Which columns hold tile entities, and where.
///
/// java: no direct equivalent — see this module's header for why it exists.
#[derive(Debug, Default, Clone)]
pub struct TileIndex {
    columns: HashMap<ChunkPos, Vec<(BlockPos, TileKind)>>,
    /// The packet/header key remains the store identity, while Java's `TileEntity.readFromNBT`
    /// keeps a separate render position read from the description compound.  This side table
    /// preserves that distinction without changing the public `TileEntity` literals used by the
    /// interaction and driver lanes.
    payload_positions: HashMap<BlockPos, BlockPos>,
    /// Generation-local animation for block-spawner miniatures.  This is deliberately separate
    /// from `World::TileEntity`: the server owns the tile payload, while the cached render entity
    /// is a client visual and must not become a gameplay/world authority.
    spawner_animations: HashMap<BlockPos, SpawnerAnimationCache>,
    /// Raw SpawnData fields consumed by the entity renderer but not represented in the shared
    /// NBT-free world snapshot. Keyed by the effective TileEntity render position, matching the
    /// animation cache and the position read from the S35 payload.
    spawner_appearances: HashMap<BlockPos, SpawnerRenderAppearance>,
    order: Vec<ChunkPos>,
    cursor: usize,
    /// The section `Arc`s each of the player's 3x3 columns had at its
    /// last scan (at most nine entries); see [`TileIndex::near_column_unchanged`].
    near_sections: Vec<(
        ChunkPos,
        [Option<std::sync::Arc<mc_world::ChunkSection>>; 16],
    )>,
}

impl TileIndex {
    /// An empty index.
    #[must_use]
    pub fn new() -> TileIndex {
        TileIndex::default()
    }

    /// How many columns are indexed.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// How many tile entities are indexed, across every column.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.columns.values().map(Vec::len).sum()
    }

    /// The tile entities in one column.
    #[must_use]
    pub fn column(&self, pos: ChunkPos) -> &[(BlockPos, TileKind)] {
        self.columns.get(&pos).map_or(&[], Vec::as_slice)
    }

    /// Every indexed tile entity.
    pub fn iter(&self) -> impl Iterator<Item = (BlockPos, TileKind)> + '_ {
        self.columns.values().flat_map(|v| v.iter().copied())
    }

    /// The effective `TileEntity.pos` used by the renderer for a packet/header key.
    #[must_use]
    pub fn render_position(&self, header: BlockPos) -> BlockPos {
        self.payload_positions
            .get(&header)
            .copied()
            .unwrap_or(header)
    }

    /// Retain the payload position separately from the world-store key.
    fn set_render_position(&mut self, header: BlockPos, payload: BlockPos) {
        if header == payload {
            self.payload_positions.remove(&header);
        } else {
            self.payload_positions.insert(header, payload);
        }
    }

    /// Observe one cached miniature's entity selection without ticking the cached entity.
    ///
    /// An entity-id change is a Java cache replacement boundary: `cachedEntity` is discarded and
    /// `func_180612_a` constructs a fresh entity.  Do not carry render identity across that
    /// boundary, even when the block position is unchanged.  There is intentionally no age
    /// advance here: the cached entity is never inserted into the world and hence never receives
    /// `Entity.onUpdate`.
    fn observe_spawner_animation(&mut self, pos: BlockPos, entity_id: &str) {
        let cache = self
            .spawner_animations
            .entry(pos)
            .or_insert_with(|| SpawnerAnimationCache::new(entity_id));
        if cache.entity_id != entity_id {
            *cache = SpawnerAnimationCache::new(entity_id);
        }
    }

    /// Read a miniature's constructor render state without advancing it.  The frame partial is
    /// applied later by `spawner_anim_params`; an unknown/missing id still gets a harmless cache
    /// entry, and the renderer's registry gate decides whether it can emit a body at all.
    fn spawner_animation(&mut self, pos: BlockPos, entity_id: &str) -> SpawnerRenderAnimation {
        self.observe_spawner_animation(pos, entity_id);
        let cache = self
            .spawner_animations
            .get(&pos)
            .expect("spawner cache entry");
        cache.animation
    }

    /// Read the client-only appearance overlay for a miniature. Missing data means the freshly
    /// constructed Java entity's ordinary defaults.
    fn spawner_appearance(&self, pos: BlockPos) -> SpawnerRenderAppearance {
        self.spawner_appearances
            .get(&pos)
            .cloned()
            .unwrap_or_default()
    }

    /// Replace the appearance overlay after an accepted S35 spawner description packet. This is
    /// deliberately not merged: Java's `readFromNBT` gives absent/malformed SpawnData fields the
    /// newly constructed entity's defaults on each cache replacement.
    fn set_spawner_appearance(&mut self, pos: BlockPos, appearance: SpawnerRenderAppearance) {
        self.spawner_appearances.insert(pos, appearance);
    }

    /// Forgets everything. Used on world change.
    pub fn clear(&mut self) {
        self.columns.clear();
        self.payload_positions.clear();
        self.spawner_animations.clear();
        self.spawner_appearances.clear();
        self.order.clear();
        self.cursor = 0;
        self.near_sections.clear();
    }

    /// One tick of index maintenance. See this module's header for the policy.
    pub fn refresh(&mut self, world: &World, around: Option<BlockPos>) {
        let loaded: HashSet<ChunkPos> = world.loaded_chunks().map(|(pos, _)| pos).collect();
        self.columns.retain(|pos, _| loaded.contains(pos));
        self.payload_positions
            .retain(|header, _| loaded.contains(&header.chunk_pos()));
        self.spawner_animations
            .retain(|pos, _| loaded.contains(&pos.chunk_pos()));
        self.spawner_appearances
            .retain(|pos, _| loaded.contains(&pos.chunk_pos()));

        // New columns first: an unindexed column is an invisible chest, and a re-scan of an
        // already-correct column is only a refresh. Nearest-first, and **sorted** — a
        // `HashSet`'s order is not stable across runs, and `--replay` determinism is a standing
        // requirement of this crate's tests.
        let mut fresh: Vec<ChunkPos> = loaded
            .iter()
            .copied()
            .filter(|pos| !self.columns.contains_key(pos))
            .collect();
        let centre = around.map_or(ChunkPos::new(0, 0), BlockPos::chunk_pos);
        fresh.sort_by_key(|p| {
            // Server packets can load a column anywhere in the `int` range.
            // An `i32` square overflowed (a debug-build panic, garbage order in release) for any
            // column ~46k chunks from the player; widen it so nearest-first stays exact.
            let dx = i64::from(p.x) - i64::from(centre.x);
            let dz = i64::from(p.z) - i64::from(centre.z);
            (
                dx.saturating_mul(dx).saturating_add(dz.saturating_mul(dz)),
                p.x,
                p.z,
            )
        });
        for pos in fresh.into_iter().take(INDEX_BUDGET) {
            self.columns
                .insert(pos, world.scan_chunk_tile_entities(pos));
        }

        // The player's own 3x3 of columns, every tick, so a placed block is never stale on
        // screen. Nine column scans is the cost of one mesher rebuild.
        if let Some(centre) = around {
            let cp = centre.chunk_pos();
            // (`MC_W18_OA_NEAR_RESCAN=0` turns it off)
            let skip_unchanged = super::driver::mods::w18_oa::near_rescan();
            if skip_unchanged {
                self.near_sections
                    .retain(|(pos, _)| pos.x.abs_diff(cp.x) <= 1 && pos.z.abs_diff(cp.z) <= 1);
            } else {
                self.near_sections.clear();
            }
            for dx in -1..=1 {
                for dz in -1..=1 {
                    let pos = ChunkPos::new(cp.x + dx, cp.z + dz);
                    if loaded.contains(&pos) {
                        if skip_unchanged && self.near_column_unchanged(world, pos) {
                            continue;
                        }
                        self.columns
                            .insert(pos, world.scan_chunk_tile_entities(pos));
                        if skip_unchanged {
                            self.remember_near_column(world, pos);
                        }
                    }
                }
            }
        }

        // Everything else, on a rolling cursor.
        if self.order.len() != loaded.len() || self.cursor >= self.order.len() {
            self.order = loaded.iter().copied().collect();
            self.order.sort_by_key(|p| (p.x, p.z));
            self.cursor = 0;
        }
        for _ in 0..REFRESH_BUDGET {
            if self.order.is_empty() {
                break;
            }
            let pos = self.order[self.cursor % self.order.len()];
            self.cursor = (self.cursor + 1) % self.order.len();
            if loaded.contains(&pos) {
                self.columns
                    .insert(pos, world.scan_chunk_tile_entities(pos));
            }
        }
    }

    /// Whether `pos`'s indexed scan is still the scan of the column's
    /// current blocks, so the every-tick 3x3 rescan may keep it.
    ///
    /// [`TileIndex::remember_near_column`] keeps a clone of every section `Arc` the last scan
    /// read. While those clones live, the world can no longer change a section in place:
    /// `mc-world` writes blocks (and light) only through `Arc::make_mut`, which copies a shared
    /// section first, and it adds or drops a section by replacing the slot. So when all 16 slots
    /// still hold the very same `Arc`s (or the same `None`), no block of the column changed since
    /// that scan, and `scan_chunk_tile_entities`, which reads nothing but those sections' blocks,
    /// would return the indexed vector again. Any write, a new chunk object, an unload or a
    /// cleared index fails the test and the column is scanned as before.
    fn near_column_unchanged(&self, world: &World, pos: ChunkPos) -> bool {
        let Some(chunk) = world.chunk(pos) else {
            return false;
        };
        self.columns.contains_key(&pos)
            && self
                .near_sections
                .iter()
                .find(|(near, _)| *near == pos)
                .is_some_and(|(_, held)| {
                    held.iter()
                        .zip(chunk.sections().iter())
                        .all(|(held, live)| match (held, live) {
                            (None, None) => true,
                            (Some(held), Some(live)) => std::sync::Arc::ptr_eq(held, live),
                            _ => false,
                        })
                })
    }

    /// Hold the sections a 3x3 scan of `pos` just read (see
    /// [`TileIndex::near_column_unchanged`]). At most the nine current columns are held.
    fn remember_near_column(&mut self, world: &World, pos: ChunkPos) {
        let Some(chunk) = world.chunk(pos) else {
            return;
        };
        let sections = chunk.sections().clone();
        match self.near_sections.iter_mut().find(|(near, _)| *near == pos) {
            Some((_, held)) => *held = sections,
            None => self.near_sections.push((pos, sections)),
        }
    }
}

/// Everything the renderer needs about one visible tile entity, read off the world in one
/// immutable pass so the pose pass can borrow `Client::tile_models` mutably.
///
/// java: the per-render-chunk `List<TileEntity>` `RenderGlobal.renderEntities` (`:695`) collects
/// before running the dispatcher over it — same two-phase shape, for a different reason.
#[derive(Debug)]
struct TileDraw {
    pos: BlockPos,
    kind: TileKind,
    rel: Vec3,
    metadata: u8,
    light: f32,
    /// java: `adjacentChestXPos != null || adjacentChestZPos != null`
    double: bool,
    /// java: the `translate(1, 0, 0)` / `translate(0, 0, -1)` a large chest's far half gets.
    double_offset: Vec3,
    /// java: `(prevLidAngle, lidAngle)`
    lid: (f32, f32),
    /// java: the `-X`/`-Z` neighbours' interpolated lids, whose larger value wins.
    neighbour_lids: Vec<f32>,
    /// java: `TileEntitySign.signText` plus `lineBeingEdited`.
    sign: Option<([String; 4], i32)>,
    /// java: `(getSkullType(), getSkullRotation(), getPlayerProfile())`; the owner descriptor is
    /// prepared while the immutable world snapshot is live so the skin request is non-blocking.
    skull: (u8, u8, Option<BuiltinSkullOwner>),
    /// java: `TileEntityBanner`'s decoded base dye and pattern list.
    banner: Option<mc_world::BannerState>,
    /// java: `TileEntityEnchantmentTable`'s six paired animation fields.
    enchant: mc_world::EnchantState,
    /// java: `(mobID, prevMobRotation, mobRotation, SpawnData appearance)`
    spawner: Option<(
        String,
        f64,
        f64,
        Option<mc_world::tileent::SpawnerSpawnData>,
    )>,
    /// The raw SpawnData appearance fields that do not belong in the shared world snapshot.
    spawner_appearance: SpawnerRenderAppearance,
    /// java: `TileEntityBeacon`'s segment list and fade.
    beacon: Option<mc_world::BeaconState>,
    /// java: `TileEntityPiston`'s carried state and interpolation fields.
    piston: Option<PistonState>,
    /// Java BlockModelRenderer output for the moving carrier, including culling/AO/light.
    piston_geometry: Option<Vec<TileVertex>>,
    /// Destroy-stage sprites currently attached to this tile position. A double chest's
    /// positive half is folded into the single `-X`/`-Z` draw below, so damage sent for either
    /// block still lands on the anchored large-chest geometry.
    destroy_stages: Vec<u8>,
}

impl Default for TileDraw {
    fn default() -> Self {
        TileDraw {
            pos: BlockPos::ORIGIN,
            // Overwritten by every construction site; `TileKind` has no meaningful default of
            // its own, and a placeholder here is cheaper than a nine-argument constructor.
            kind: TileKind::Chest,
            rel: Vec3::ZERO,
            metadata: 0,
            light: 1.0,
            double: false,
            double_offset: Vec3::ZERO,
            lid: (0.0, 0.0),
            neighbour_lids: Vec::new(),
            sign: None,
            skull: (0, 0, None),
            banner: None,
            enchant: mc_world::EnchantState::default(),
            spawner: None,
            spawner_appearance: SpawnerRenderAppearance::default(),
            beacon: None,
            piston: None,
            piston_geometry: None,
            destroy_stages: Vec::new(),
        }
    }
}

/// Texture-matrix scale used by the tile-entity destroy-stage replay.
///
/// Java excludes chest, ender-chest, sign, and skull blocks from the ordinary terrain crack
/// pass, then each matching TESR binds `destroy_stage_N` and scales its texture coordinates to
/// the model. Banners are deliberately absent: `TileEntityBannerRenderer` ignores
/// `destroyStage`, so their damage remains an ordinary block overlay.
#[must_use]
fn destroy_stage_scale(kind: TileKind, double: bool) -> Option<(f32, f32)> {
    match kind {
        TileKind::Chest | TileKind::TrappedChest => {
            if double {
                Some((8.0, 4.0))
            } else {
                Some((4.0, 4.0))
            }
        }
        TileKind::EnderChest => Some((4.0, 4.0)),
        TileKind::StandingSign | TileKind::WallSign | TileKind::Skull => Some((4.0, 2.0)),
        _ => None,
    }
}

/// `MC_NW_F1_TILE_FAST=0` restores the
/// previous tile-entity snapshot exactly. Read once per process.
fn nw_f1_tile_fast() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_NW_F1_TILE_FAST").is_none_or(|value| value != "0"))
}

/// Kinds with no `TileEntitySpecialRenderer`: their pose arm in
/// `upload_tile_entities_with_clock` yields no texture (so no draw and no destroy-stage replay),
/// and the snapshot loop gives them no side effect. Exhaustive on purpose: a new kind must decide.
#[must_use]
fn nw_f1_tile_draws_nothing(kind: TileKind) -> bool {
    match kind {
        TileKind::Dispenser
        | TileKind::Furnace
        | TileKind::Jukebox
        | TileKind::BrewingStand
        | TileKind::Comparator
        | TileKind::DaylightDetector
        | TileKind::Hopper
        | TileKind::Dropper
        | TileKind::Note
        | TileKind::CommandBlock
        | TileKind::FlowerPot => true,
        TileKind::Chest
        | TileKind::TrappedChest
        | TileKind::EnderChest
        | TileKind::StandingSign
        | TileKind::WallSign
        | TileKind::StandingBanner
        | TileKind::WallBanner
        | TileKind::Skull
        | TileKind::Beacon
        | TileKind::MobSpawner
        | TileKind::EnchantmentTable
        | TileKind::EndPortal
        | TileKind::Piston => false,
    }
}

/// The TESR draw-list order (06-entity-batch U3): piston carriers first, position-free (a
/// stable sort keeps their relative order); then opaque kinds grouped by the sheet they bind
/// (kind, then the variant that selects the sheet), then block position for a total order that
/// no longer depends on `TileIndex`'s std `HashMap` iteration; then kinds that never push
/// ordinary tile geometry, sharing one position-free rank.
///
/// Pistons first (phase 4c review S1): a carrier is blended but writes depth, and vanilla draws
/// `renderInfos` nearest chunk first, so a moving glass, ice or slime block in front of a chest
/// or sign hides it. Drawing carriers first reproduces that whenever the two are in different
/// sections; behind an opaque tile entity either order gives the same pixels. Within one
/// section vanilla's block order decides, which this order does not model.
#[must_use]
fn tile_draw_order_key(draw: &TileDraw) -> (u8, u8, i32, i32, i32) {
    let positioned = |rank: u8, variant: u8| (rank, variant, draw.pos.x, draw.pos.y, draw.pos.z);
    match draw.kind {
        TileKind::Piston => (0, 0, 0, 0, 0),
        TileKind::Chest => positioned(1, u8::from(draw.double)),
        TileKind::TrappedChest => positioned(2, u8::from(draw.double)),
        TileKind::EnderChest => positioned(3, 0),
        TileKind::StandingSign | TileKind::WallSign => positioned(4, 0),
        TileKind::Skull => positioned(5, draw.skull.0),
        TileKind::EnchantmentTable => positioned(6, 0),
        TileKind::StandingBanner | TileKind::WallBanner => positioned(7, 0),
        _ => (8, 0, 0, 0, 0),
    }
}

/// Keep only the ten legal TESR destroy-stage sprites, in deterministic draw order.
#[must_use]
fn legal_destroy_stages(stages: &[u8]) -> Vec<u8> {
    let mut stages: Vec<u8> = stages.iter().copied().filter(|stage| *stage < 10).collect();
    stages.sort_unstable();
    stages.dedup();
    stages
}

/// Returns the state that `TileEntityBannerRenderer.func_178463_a` is allowed to render.
///
/// java: `TileEntityBanner.field_175119_g` starts false on the lazily-created tile and becomes
/// true only after `readFromNBT`/`setItemValues`.  The vanilla renderer turns that false state
/// into an empty resource key and skips `ModelBanner.renderBanner` entirely; keeping this as a
/// small shared predicate makes the packet/store/render seam directly testable without a GPU.
#[must_use]
fn banner_state_for_render(stored: Option<&mc_world::TileEntity>) -> Option<mc_world::BannerState> {
    stored
        .and_then(mc_world::TileEntity::banner)
        .filter(|banner| banner.initialized)
        .cloned()
}

fn should_render_sign_line(line: &str, selected: bool) -> bool {
    selected || !line.is_empty()
}

/// The fixed-function state a TESR leaves its geometry in.
///
/// java: `TileEntityPistonRenderer` enables `blendFunc(770, 771)` for the carried block, and
/// `TileEntityEnchantmentTableRenderer.renderTileEntityAt:64` enables back-face culling for
/// `ModelBook`, whose zero-depth covers z-fight without it. Every other TESR here draws closed
/// boxes (or, like the skull, disables culling itself), which the cull-less opaque pipeline
/// draws identically.
#[must_use]
fn tile_blend(kind: TileKind) -> TileBlend {
    match kind {
        TileKind::Piston => TileBlend::Translucent,
        TileKind::EnchantmentTable => TileBlend::OpaqueCulled,
        _ => TileBlend::Opaque,
    }
}

/// The colour factor vanilla's fixed-function lighting leaves on a sign's text.
///
/// java: `TileEntitySignRenderer.renderTileEntityAt:75-84` — under `scale(f3, -f3, f3)` the text
/// gets `glNormal3f(0, 0, -1 * f3)` with `enableRescaleNormal()` on, while the standard item
/// lighting `RenderGlobal.renderEntities:685` enabled for the TESR loop is still on
/// (`FontRenderer` never turns it off). `GL_RESCALE_NORMAL` restores the normal's own length,
/// `f3 = 0.0104`, and nothing enables `GL_NORMALIZE`, so both diffuse terms shrink to about 1%:
/// the glyph colour keeps the 0.4 scene ambient plus a sliver, times the tile's lightmap
/// brightness exactly as the board. Black text is unaffected; coloured text is drawn at ~40%.
#[must_use]
fn sign_text_shade(board: mc_render::entity::Mat4, light: f32) -> f32 {
    // java: `RenderHelper.LIGHT0_POS`/`LIGHT1_POS`, fixed in world space like the tile shader's.
    let light0 = Vec3::new(0.2, 1.0, -0.7).normalize();
    let light1 = Vec3::new(-0.2, 1.0, 0.7).normalize();
    let normal = board.transform_vector3(Vec3::NEG_Z) * tile::SIGN_TEXT_SCALE;
    let diffuse = 0.4 + 0.6 * normal.dot(light0).max(0.0) + 0.6 * normal.dot(light1).max(0.0);
    diffuse.min(1.0) * light
}

/// Lays glyphs out under an arbitrary world matrix.
///
/// java: `TileEntitySignRenderer` calls `fontrenderer.drawString` with the sign's own matrix on
/// the stack, so the glyphs land on the board rather than facing the camera.
/// [`mc_ui::nameplate`]'s `BillboardSink` does the same job for a *billboard*; this is the same
/// seam with the billboard swapped for a fixed matrix.
struct SignTextSink {
    matrix: mc_render::entity::Mat4,
    /// [`sign_text_shade`] for this sign: the lighting factor on every glyph and bar colour.
    shade: f32,
    /// Batched by sheet in emission order, keeping the `FontTexture` rather than its key string
    /// so the caller can *upload* a missing unicode page rather than only notice it is absent.
    batches: Vec<(mc_ui::font::FontTexture, Vec<NameplateGlyphVertex>)>,
    /// Vanilla renders strikethrough/underline decorations as solid quads. These use the same
    /// fixed sign matrix and depth-write-off pipeline as the glyphs.
    bars: Vec<NameplateVertex>,
}

impl SignTextSink {
    fn place(&self, x: f32, y: f32) -> [f32; 3] {
        self.matrix.transform_point3(Vec3::new(x, y, 0.0)).into()
    }

    /// `FontRenderer`'s ARGB under the sign's fixed-function lighting; alpha is untouched.
    fn lit(&self, argb: u32) -> [f32; 4] {
        let [r, g, b, a] = mc_render::entity::argb_to_rgba(argb);
        [r * self.shade, g * self.shade, b * self.shade, a]
    }
}

impl TextSink for SignTextSink {
    fn glyph(&mut self, quad: GlyphQuad) {
        let rgba = self.lit(quad.color);
        let world: [[f32; 3]; 4] =
            core::array::from_fn(|i| self.place(quad.pos[i][0], quad.pos[i][1]));
        // GL_TRIANGLE_STRIP over [0,1,2,3] is triangles (0,1,2) and (2,1,3) — the same
        // expansion `Nameplate::glyph_vertices` documents.
        const STRIP: [usize; 6] = [0, 1, 2, 2, 1, 3];
        let verts = STRIP.map(|i| NameplateGlyphVertex {
            pos: world[i],
            uv: quad.uv[i],
            color: rgba,
        });
        match self.batches.last_mut() {
            Some((texture, batch)) if *texture == quad.texture => batch.extend_from_slice(&verts),
            _ => self.batches.push((quad.texture, verts.to_vec())),
        }
    }

    fn bar(&mut self, rect: SolidRect) {
        let rgba = self.lit(rect.color);
        let corners = [
            (rect.x0, rect.y0),
            (rect.x1, rect.y0),
            (rect.x1, rect.y1),
            (rect.x0, rect.y1),
        ];
        for i in [0usize, 1, 2, 0, 2, 3] {
            let (x, y) = corners[i];
            self.bars.push(NameplateVertex {
                pos: self.matrix.transform_point3(Vec3::new(x, y, 0.0)).into(),
                color: rgba,
            });
        }
    }
}

/// R2-20 / TILE-3: each visible sign row's glyphs, as `FontRenderer` emitted them in sign-local
/// pixels, so a sign is wrapped, measured and run through the `§` state machine once rather than
/// every frame.
///
/// A row's quads are a pure function of the stored line, its row index (the y origin), whether
/// the edit cursor is on it (`"> " + s + " <"`), and the font's layout state, so that is the key
/// (font changes drop everything via [`mc_ui::font::FontRenderer::layout_generation`]). The sign's
/// matrix is applied per frame by feeding the cached quads through the very same
/// [`SignTextSink`], so the vertices are bit-identical to drawing the row again. Rows that can
/// reach `§k` ([`mc_ui::font::FontRenderer::may_obfuscate`]) are never cached: they are drawn
/// every frame as before, so the shared obfuscation stream advances exactly as it did.
#[derive(Debug, Default)]
pub(crate) struct SignTextCache {
    /// Stored line text → slot `row * 2 + selected`.
    rows: HashMap<String, [Option<(mc_ui::font::TextMesh, u64)>; 8]>,
    generation: u64,
    frame: u64,
}

impl SignTextCache {
    /// Frames a row may go unseen before it is dropped.
    const EVICT_AFTER_FRAMES: u64 = 120;

    fn begin_frame(&mut self, font: &mc_ui::font::FontRenderer) {
        if self.generation != font.layout_generation() {
            self.rows.clear();
            self.generation = font.layout_generation();
        }
        self.frame += 1;
        if self.frame.is_multiple_of(Self::EVICT_AFTER_FRAMES) {
            let horizon = self.frame - Self::EVICT_AFTER_FRAMES;
            self.rows.retain(|_, slots| {
                for slot in slots.iter_mut() {
                    if slot.as_ref().is_some_and(|(_, used)| *used < horizon) {
                        *slot = None;
                    }
                }
                slots.iter().any(Option::is_some)
            });
        }
    }

    fn slot(index: usize, selected: bool) -> usize {
        index * 2 + usize::from(selected)
    }

    /// The cached row, marked used this frame.
    fn get(&mut self, line: &str, index: usize, selected: bool) -> Option<&mc_ui::font::TextMesh> {
        let frame = self.frame;
        let (mesh, used) = self.rows.get_mut(line)?[Self::slot(index, selected)].as_mut()?;
        *used = frame;
        Some(&*mesh)
    }

    fn insert(&mut self, line: &str, index: usize, selected: bool, mesh: mc_ui::font::TextMesh) {
        let frame = self.frame;
        self.rows.entry(line.to_owned()).or_default()[Self::slot(index, selected)] =
            Some((mesh, frame));
    }
}

/// Flatten one S33 sign row into the text `TileEntitySignRenderer` draws for it.
///
/// java: `TileEntitySignRenderer.renderTileEntityAt:95-96` —
/// `GuiUtilRenderComponents.splitText(row, 90, fontrenderer, false, true)` and the first line's
/// `getFormattedText()`, ported as [`mc_ui::screens::chat::sign_row_text`]. `splitText` walks the
/// root **and its siblings** (`Lists.newArrayList(Iterable)` over the component, bytecode `avu.a`
/// offset 20), which matters as many servers send every sign row as `{"extra":[..],"text":""}`;
/// the former root-only reading drew all of those signs blank.
///
/// The live S33 packet has already passed Gson validation by the network task.  Keeping the raw
/// line on a direct malformed call preserves the existing non-panicking fallback and makes the
/// impossible-after-decode case observable to the caller instead of silently losing a sign row.
fn sign_line_formatted_text(
    line: &str,
    font: &mc_ui::font::FontRenderer,
    language: &mc_assets::Language,
) -> String {
    match mc_ui::font::component::ChatComponent::from_json_nullable(line) {
        Ok(Some(component)) => mc_ui::screens::chat::sign_row_text(&component, 90, font, language),
        // TileEntitySignRenderer checks each component for null before drawing it.
        Ok(None) => String::new(),
        Err(_) => line.to_owned(),
    }
}

impl Client {
    // -------------------------------------------------------------------------------------
    // Small shared helpers
    // -------------------------------------------------------------------------------------

    // `play_world_sound` used to live here, wrapping `World.playSoundEffect`. It is gone,
    // because on a client that method plays nothing: it walks the world's `IWorldAccess` list,
    // whose only member is `RenderGlobal` (`client/renderer/RenderGlobal.java:474`), and
    // `RenderGlobal.playSound` is an empty body (`:2025-2027`). Every sound that reached it —
    // the chest lid, the note block — is produced by the *server* running the same code and
    // sent as `S29PacketSoundEffect`. Its two callers here were doubling those sounds.
    //
    // The client-local sound path that *does* exist is the seven-argument
    // `WorldClient.playSound` (`client/multiplayer/WorldClient.java:439`), which
    // `EntityPlayerSP.playSound` overrides onto; nothing in this file reaches it.

    /// One plain client-side line in the chat box. java:
    /// `thePlayer.addChatMessage(new ChatComponentText(...))`.
    fn chat_system_message(&mut self, text: String) {
        let layout = mc_ui::screens::chat::ChatLayout::from_settings(&self.settings);
        let open = matches!(self.ui, super::driver::Ui::Chat(_));
        self.chat.add_message(
            mc_ui::font::component::ChatComponent::text(text),
            0,
            self.tick_count() as i32,
            &layout,
            &self.font,
            &self.assets.language,
            open,
        );
    }

    /// Which loaded columns hold tile entities, and where. Test/debug read-out.
    #[must_use]
    pub fn tile_index(&self) -> &TileIndex {
        &self.tile_index
    }

    /// Vertices the tile-entity pass uploaded last frame, and the draw calls they batched into.
    ///
    /// `(0, 0)` before [`Client::install`]. This is the number the A05 verdict was about: it was
    /// structurally incapable of being anything but zero, because no tile-entity phase existed.
    #[must_use]
    pub fn tile_geometry(&self) -> (usize, usize) {
        self.tiles.as_ref().map_or((0, 0), |p| {
            p.with(|phase| (phase.vertex_count(), phase.batch_count()))
        })
    }

    /// Beacon-beam vertices uploaded last frame, `(inner, outer)`.
    #[must_use]
    pub fn beacon_geometry(&self) -> (usize, usize) {
        self.beacons
            .as_ref()
            .map_or((0, 0), |p| p.with(|phase| phase.vertex_counts()))
    }

    // -------------------------------------------------------------------------------------
    // Packets in
    // -------------------------------------------------------------------------------------

    /// java: `NetHandlerPlayClient.handleUpdateTileEntity` (`:1267-1281`).
    ///
    /// The action byte is a **type check**, not a constructor — vanilla looks the tile entity up
    /// from the block first and only reads the compound when the action and the instance's class
    /// agree. `mc_world::World::apply_tile_entity_nbt` enforces that; this function's whole job
    /// is the `readFromNBT` half, which lives here because `mc-world` has no `mc-nbt`
    /// dependency (its manifest is frozen for the wave).
    pub(crate) fn react_to_tile_entity_update(
        &mut self,
        position: BlockPos,
        action: u8,
        nbt: Option<&mc_nbt::NbtCompound>,
    ) {
        let Some(nbt) = nbt else { return };
        // java: `TileEntity.readFromNBT` unconditionally overwrites `pos` from these three
        // `getInteger` calls. `NBTTagCompound.getInteger` returns zero for a missing or
        // non-numeric tag, so the packet/header position is not a fallback for a partial payload.
        let payload_position = tile_entity_payload_position(nbt);
        let applied = {
            let Some(session) = self.session_mut() else {
                return;
            };
            let world = &mut session.state_mut().world;
            // java: `theWorld.isBlockLoaded(packetIn.getPos())` — a description packet for a chunk
            // that has not arrived is dropped, not queued.
            let Some(kind) = world.tile_kind_at(position) else {
                return;
            };
            // `TileEntityCommandBlock.readDataFromNBT` only overwrites its optional fields when the
            // exact NBT tag is present. Seed the decode from the live instance so a partial S35 has
            // the same preserve-on-absence behaviour instead of manufacturing Java's defaults.
            let command_block = world
                .tile_entity(position)
                .and_then(mc_world::TileEntity::command_block)
                .cloned();
            let data = match kind {
                // java: `MobSpawnerBaseLogic.readFromNBT` (`:249-296`).
                //
                // The two conditional blocks matter and were missing: `MinSpawnDelay` gates
                // `minSpawnDelay`/`maxSpawnDelay`/`spawnCount`, and `MaxNearbyEntities` gates
                // `maxNearbyEntities`/**`RequiredPlayerRange`** — the radius `isActivated()` uses to
                // decide whether the spawner animates at all. Both survive `getDescriptionPacket`,
                // which strips only `SpawnPotentials`.
                //
                // `mobRotation`/`prevMobRotation` are not in the compound and are preserved across a
                // repeat packet by `World::apply_tile_entity_nbt`; the defaults here are only used
                // when there is no previous state.
                //
                // `SpawnData` is decoded into the render-visible overlay applied to the cached
                // miniature by `spawnNewEntity` (a charged creeper, an armoured skeleton). The
                // species is not affected: the two-argument `WeightedRandomMinecart` takes its type
                // from `mobID`.
                TileKind::MobSpawner => TileData::Spawner(read_spawner(nbt)),
                // java: `TileEntityBeacon.readFromNBT` — `Levels` is the only field the renderer
                // reads; the segment list is recomputed client-side by `updateSegmentColors`.
                TileKind::Beacon => TileData::Beacon(mc_world::BeaconState {
                    levels: nbt.get_i32("Levels").unwrap_or(0),
                    ..mc_world::BeaconState::default()
                }),
                // java: `TileEntitySkull.readFromNBT` (`:33-55`).
                TileKind::Skull => TileData::Skull(read_skull(nbt)),
                // java: `TileEntityFlowerPot.readFromNBT` (`:32-44`). The description packet rewrites
                // `Item` from a string to an integer (`:55`), so the int form is the one on the wire.
                TileKind::FlowerPot => TileData::FlowerPot(mc_world::FlowerPotState {
                    item_id: nbt.get_i32("Item").unwrap_or(0),
                    data: nbt.get_i32("Data").unwrap_or(0),
                }),
                // java: `TileEntityBanner.readFromNBT` (`:78+`).
                TileKind::StandingBanner | TileKind::WallBanner => {
                    TileData::Banner(read_banner(nbt))
                }
                // java: `TileEntityCommandBlock` — accepted so the action-2 payload is not reported
                // as a decode failure; `CommandBlockLogic.readDataFromNBT` retains the typed state
                // even though this block has no TESR. The GUI/editor is a separate scope.
                TileKind::CommandBlock => {
                    TileData::CommandBlock(read_command_block(nbt, command_block.as_ref()))
                }
                _ => return,
            };
            let applied = world.apply_tile_entity_nbt(position, action, data);
            if applied && kind == TileKind::FlowerPot {
                // The pot's *contents* change its block model, so the section has to be
                // re-meshed. The terrain actual-state hook consumes the fresh tile payload on its
                // next pass, just as `BlockRendererDispatcher` calls `getActualState` per lookup.
                world.mark_dirty_box(position);
            }
            applied
        };
        if applied {
            self.tile_index
                .set_render_position(position, payload_position);
            // Keep renderer-only SpawnData keys beside the index. The world store receives the
            // typed appearance subset above, while this local overlay retains exact name/bat/
            // wither/conversion inputs without making gameplay depend on raw NBT.
            if action == 1 {
                let spawner = read_spawner(nbt);
                self.tile_index.set_spawner_appearance(
                    payload_position,
                    read_spawner_render_appearance(
                        &spawner.entity_id,
                        nbt.get_compound("SpawnData"),
                    ),
                );
            }
        }
    }

    /// java: `NetHandlerPlayClient.handleUpdateSign` (`:1234-1261`), including the chat warning
    /// when the position is loaded but is not a sign.
    pub(crate) fn react_to_sign_update(&mut self, position: BlockPos, lines: &[String; 4]) {
        // java: `System.arraycopy(packetIn.getLines(), ...)` copies `IChatComponent`s, and the
        // renderer's `GuiUtilRenderComponents.splitText` (siblings included) runs on them every
        // frame; it is resolved once here with the current font.
        let text: [String; 4] = core::array::from_fn(|i| {
            sign_line_formatted_text(&lines[i], &self.font, &self.assets.language)
        });
        let found = match self.session_mut() {
            Some(session) => {
                session
                    .state_mut()
                    .world
                    .apply_sign_components(position, text, lines.clone())
            }
            None => false,
        };
        if !found {
            // java: `"Unable to locate sign at " + x + ", " + y + ", " + z`, straight into the
            // chat box as a plain `ChatComponentText`.
            self.chat_system_message(format!(
                "Unable to locate sign at {}, {}, {}",
                position.x, position.y, position.z
            ));
        }
    }

    /// java: `NetHandlerPlayClient.handleSignEditorOpen` (`:1216-1229`).
    ///
    /// Vanilla uses the live sign components when the target is a sign, but fabricates an
    /// unregistered bare `TileEntitySign` for every other target.  The screen model owns that
    /// fabricated state; the live path mirrors its raw rows and blinking caret back to the
    /// world renderer every input/frame lifecycle step.
    pub(crate) fn react_to_open_sign_editor(&mut self, position: BlockPos) {
        let components = self.session_mut().and_then(|session| {
            let world = &mut session.state_mut().world;
            world
                .tile_entity_or_create(position)
                .and_then(|tile| tile.sign())
                .map(|sign| sign.components.clone())
        });
        let screen = match components {
            Some(components) => {
                EditSignScreen::from_component_json(position, components, &self.assets.language)
            }
            // `NetHandlerPlayClient.handleSignEditorOpen` does not insert its fabricated tile
            // into the world; it exists solely until this modal closes.
            None => EditSignScreen::new(position, Default::default()),
        };
        self.show_screen(Some(Ui::EditSign(Box::new(screen))));
    }

    /// Write the editor's raw rows and blink state to a live sign immediately after an input,
    /// tick, or layout.  A fabricated S36 target has no world tile and intentionally becomes a
    /// no-op here, matching vanilla's unregistered temporary `TileEntitySign`.
    pub(crate) fn sync_sign_editor_render_state(&mut self) {
        let state = match &self.ui {
            Ui::EditSign(screen) => Some((
                screen.position(),
                screen.raw_lines(),
                screen.line_being_edited(),
            )),
            _ => None,
        };
        let Some((position, lines, line_being_edited)) = state else {
            return;
        };
        if let Some(session) = self.session_mut() {
            session.state_mut().world.set_sign_editor_render_state(
                position,
                lines,
                line_being_edited,
            );
        }
    }

    /// Apply the narrow host side of `GuiEditSign`: tile editability/dirtiness and C12 belong to
    /// the client, while component serialisation belongs to `mc-ui`.  `Close` is deliberately
    /// handled only after the complete action stream, so C12 precedes the screen replacement.
    pub(crate) fn drain_sign_actions(&mut self) {
        let actions = match &mut self.ui {
            Ui::EditSign(screen) => screen.take_actions(),
            _ => return,
        };
        let mut close = false;
        for action in actions {
            match action {
                SignAction::SetEditable(editable) => {
                    let position = match &self.ui {
                        Ui::EditSign(screen) => screen.position(),
                        _ => continue,
                    };
                    if let Some(session) = self.session_mut() {
                        session
                            .state_mut()
                            .world
                            .set_sign_editable(position, editable);
                    }
                }
                SignAction::MarkDirty => {
                    let position = match &self.ui {
                        Ui::EditSign(screen) => screen.position(),
                        _ => continue,
                    };
                    if let Some(session) = self.session_mut() {
                        session.state_mut().world.mark_dirty_box(position);
                    }
                }
                SignAction::SendUpdateSign { position, lines } => {
                    if let Some(session) = self.session_mut() {
                        session
                            .state_mut()
                            .world
                            .set_sign_components(position, lines.clone());
                        session.send(mc_protocol::play::C2SPlay::UpdateSign(
                            mc_protocol::play::c2s::UpdateSign { position, lines },
                        ));
                    }
                }
                SignAction::Close => close = true,
            }
        }
        if close {
            self.show_screen(None);
        }
    }

    /// java: `NetHandlerPlayClient.handleBlockAction` (`:1322-1326`) → `World.addBlockEvent` →
    /// `Block.onBlockEventReceived`.
    pub(crate) fn react_to_block_action(&mut self, action: &mc_protocol::play::s2c::BlockAction) {
        let Some(session) = self.session_mut() else {
            return;
        };
        let border = {
            let state = session.state();
            let border = state.env.border.dimension_view(state.local.dimension);
            PistonWorldBorder::new(
                border.center_x,
                border.center_z,
                border.current_diameter(piston_border_clock_millis()),
                border.world_size,
            )
        };
        let effect = session.state_mut().world.apply_block_action_with_border(
            action.position,
            action.block_id,
            action.data1,
            action.data2,
            Some(border),
        );
        let Some(effect) = effect else { return };
        match effect {
            // The chest's own state is already applied; the lid animates from the next tick.
            mc_world::BlockActionEffect::ChestUsers { .. } => {}
            // java: `BlockNote.onBlockEventReceived` (`block/BlockNote.java:108-113`) —
            // `note.<instrument>` at `2^((note - 12) / 12)`, plus a NOTE particle at `y + 1.2`.
            // **The sound is deliberately not played here.** `BlockNote.onBlockEventReceived`
            // reaches it through `worldIn.playSoundEffect`, which walks the world's
            // `IWorldAccess` list; on a client the only entry is `RenderGlobal`
            // (`client/renderer/RenderGlobal.java:474`) and its `playSound` body is **empty**
            // (`:2025-2027`). The same handler runs on the server, where `WorldManager.playSound`
            // (`world/WorldManager.java:52-55`) turns it into an `S29PacketSoundEffect` — which
            // `Client::react_to_sound_effect` already plays. Synthesising it here as well made
            // every note block strike twice.
            //
            // The NOTE *particle* on the next line **is** genuinely client-side, and is spawned
            // here. This is the asymmetry that makes `BlockNote.onBlockEventReceived` worth
            // reading carefully: its two statements go to two different places.
            // `worldIn.playSoundEffect` reaches `RenderGlobal.playSound`, an empty body, so the
            // client contributes nothing; `worldIn.spawnParticle` reaches
            // `RenderGlobal.spawnParticle` (`:2064`), which is fully implemented, so the client
            // contributes everything — the server never sends a note particle.
            //
            // ```java
            // worldIn.spawnParticle(EnumParticleTypes.NOTE, pos.getX() + 0.5D, pos.getY() + 1.2D,
            //                       pos.getZ() + 0.5D, (double)eventParam / 24.0D, 0.0D, 0.0D,
            //                       new int[0]);
            // ```
            //
            // `eventParam` is `data2`, the note — which `EntityNoteFX` reuses as a *hue*, not a
            // velocity (`spawn::note`'s three phase-shifted sines), hence `/ 24.0` on `xSpeed`
            // and zeroes on the other two axes. `y + 1.2`, not `+ 0.5`: the note floats out of
            // the block's top face.
            mc_world::BlockActionEffect::Note { note, .. } => {
                let (pos, speed) = note_particle_parameters(action.position, note);
                self.spawn_effect_particle(mc_render::particle::packet::NOTE, pos, speed, &[]);
            }
            // The whole move — `doMove`, the `piston_extension` writes, the `TileEntityPiston`
            // creations and the `EXTENDED` flag — has already been applied by
            // `World::apply_block_action`, because vanilla's client runs it locally
            // (`doMove` writes with flag 4, so the server never sends those changes).
            //
            // Nothing is played here on purpose: `BlockPistonBase.onBlockEventReceived`'s
            // `worldIn.playSoundEffect` reaches only `RenderGlobal.playSound`, whose body is
            // empty (`client/renderer/RenderGlobal.java:2025-2027`); the `tile.piston.out` /
            // `tile.piston.in` the player hears is the server's `S29PacketSoundEffect`.
            //
            // The tile pass owns the moving-block TESR. It snapshots these carrier tile entities
            // from the refreshed index and emits their baked block models in the same entity
            // render phase; this arm only records the already-applied world-side event.
            mc_world::BlockActionEffect::Piston { extending, facing } => {
                tracing::trace!(
                    extending,
                    facing,
                    "piston move applied; moving-block TESR state is live"
                );
            }
        }
    }

    // -------------------------------------------------------------------------------------
    // Per tick
    // -------------------------------------------------------------------------------------

    /// One 20 Hz tick of tile-entity animation.
    ///
    /// java: `World.updateEntities`' `tickableTileEntities` loop, which on a client runs
    /// `TileEntityChest.update`, `MobSpawnerBaseLogic.updateSpawner` and
    /// `TileEntityEnchantmentTable.update`.
    pub(crate) fn tick_tile_entities(&mut self) {
        mc_core::profile_scope!("client.tick_tile_entities");
        let Some(session) = self.session() else {
            self.tile_index.clear();
            return;
        };
        // java: both consumers of this position filter the player list through
        // `EntitySelectors.NOT_SPECTATING` — `World.isAnyPlayerWithinRangeAt:3325` for the mob
        // spawner and `World.getClosestPlayer:3304` for the enchantment table. The predicate is
        // *only* the spectator test (`util/EntitySelectors.java:35-41`); there is no
        // `isEntityAlive` in it, so a dead player still counts and only a spectating one does
        // not. A spectator walking past a spawner leaves it frozen and the book shut.
        let state = session.state();
        let local_spectating = state
            .players
            .get(&state.local.uuid)
            .is_some_and(|info| info.game_mode == mc_protocol::play::s2c::GameMode::Spectator);
        let local_player = session
            .predictor()
            .map(|predictor| (predictor.state.pos, local_spectating));
        let players = state.entities.non_spectator_player_positions(
            &state.world,
            local_player,
            &state.players,
        );
        // The *index* centre is the render viewpoint, not a game-rule player: a spectator still
        // needs tile entities loaded around them to see anything at all. Deliberately taken from
        // the predictor directly rather than from `player` above.
        let centre = session
            .predictor()
            .map(|p| BlockPos::from_f64(p.state.pos.x, p.state.pos.y, p.state.pos.z));
        let tick = self.tick_count();

        // java: `MobSpawnerBaseLogic.updateSpawner` emits its co-located smoke/flame pair at
        // the start of the remote branch, before decrementing `spawnDelay` and advancing the
        // render rotation. Keep both the activation snapshot and particle submission before the
        // world tick; recomputing the same range answer afterward reverses the producer order.
        let active_spawners = session
            .state()
            .world
            .activated_spawners_for_players(&players);
        for pos in active_spawners {
            let at = DVec3::new(
                f64::from(pos.x) + f64::from(self.spawner_rng.next_f32()),
                f64::from(pos.y) + f64::from(self.spawner_rng.next_f32()),
                f64::from(pos.z) + f64::from(self.spawner_rng.next_f32()),
            );
            self.spawn_effect_particle(
                mc_render::particle::packet::SMOKE_NORMAL,
                at,
                DVec3::ZERO,
                &[],
            );
            self.spawn_effect_particle(mc_render::particle::packet::FLAME, at, DVec3::ZERO, &[]);
        }

        let sounds = {
            let Some(session) = self.session_mut() else {
                return;
            };
            let world = &mut session.state_mut().world;
            // Stale entries (a chest broken and replaced by a sign) would hand the renderer a
            // `TileData` of the wrong shape. java: `Chunk.setBlockState` does this inline.
            world.prune_tile_entities();
            let sounds = world.tick_tile_entities_for_players(&players);
            // java: `TileEntityBeacon.update` re-runs `updateSegmentColors` every 80 ticks, but
            if tick.is_multiple_of(80) {
                world.refresh_beacons();
            }
            sounds
        };

        // Taken out and put back so the index can be refreshed against a `&World` borrowed from
        // `self` — `Client::session` is a method, so it borrows the whole struct.
        let mut index = std::mem::take(&mut self.tile_index);
        if let Some(session) = self.session() {
            index.refresh(&session.state().world, centre);
        }
        // Four tile entities need an instance the wire will not necessarily give them. The
        // enchantment table is not one of `handleUpdateTileEntity`'s six action bytes, so no
        // `S35` can ever create it; the beacon's `updateSegmentColors` runs on both sides; and a
        // **mob spawner that was never described still renders a pig**, because
        // `MobSpawnerBaseLogic.mobID` is initialised to `"Pig"`
        // (`tileentity/MobSpawnerBaseLogic.java:24`) and `Chunk.getTileEntity` hands the TESR a
        // default-constructed instance. A **flower pot that was never described must likewise
        // resolve its legacy block metadata before terrain `getActualState`**, or the immutable
        // mesher lookup falls back to `contents=empty`. Vanilla gets all four for free from that
        // lazy create; here the store is only written by a packet unless something asks, so this
        // is that ask.
        let pending: Vec<(BlockPos, TileKind)> = index
            .iter()
            .filter(|(_, kind)| {
                matches!(
                    kind,
                    TileKind::Beacon
                        | TileKind::EnchantmentTable
                        | TileKind::MobSpawner
                        | TileKind::FlowerPot
                )
            })
            .collect();
        if let Some(session) = self.session_mut() {
            let world = &mut session.state_mut().world;
            for (pos, kind) in pending {
                let missing_flower_pot =
                    kind == TileKind::FlowerPot && world.tile_entity(pos).is_none();
                world.tile_entity_or_create(pos);
                // A chunk-loaded pot's default state is derived from its legacy block metadata,
                // but that state is invisible to the mesher until this section is rebuilt. Do
                // not dirty already-described pots every tick: S35 already owns that invalidation.
                if missing_flower_pot {
                    world.mark_dirty_box(pos);
                }
            }
        }
        // `TileEntityMobSpawnerRenderer` receives a cached entity that was constructed with
        // `spawnNewEntity(entity, false)`.  That entity is never inserted into the world, so no
        // client tick reaches its age/limb/head fields.  Keep the activation gate for the real
        // smoke/flame/Delay/rotation work above, but do not invent a render-age tick here.
        self.tile_index = index;

        // java: `TileEntityChest.update`'s two `worldObj.playSoundEffect` calls (`:372`, `:410`)
        // are **not** `isRemote`-gated, but on a client `World.playSoundEffect` walks a
        // `worldAccesses` list whose only member is `RenderGlobal`
        // (`client/renderer/RenderGlobal.java:474`), and `RenderGlobal.playSound` is an **empty
        // body** (`:2025-2027`). The chest sound a vanilla player hears is the *server's*
        // `S29PacketSoundEffect`, produced by the same `update()` running there and dispatched
        // through `world/WorldManager.java:52-55`. Playing it here as well doubled it, and
        // reproducing the empty body is the faithful port.
        //
        // `World::tick_tile_entities` still returns the requests, because they are the record
        // of where vanilla calls `playSoundEffect` and its lid-animation test uses them as the
        // oracle for the open/close transitions.
        let _unplayed_on_the_client = sounds;

        // java: the *other* two statements inside `MobSpawnerBaseLogic.updateSpawner`'s
        // `isRemote` branch (`tileentity/MobSpawnerBaseLogic.java:90-94`):
        //
        // ```java
        // double d3 = blockpos.getX() + rand.nextFloat();   // and d4, d5 for y and z
        // spawnParticle(EnumParticleTypes.SMOKE_NORMAL, d3, d4, d5, 0, 0, 0, new int[0]);
        // spawnParticle(EnumParticleTypes.FLAME,        d3, d4, d5, 0, 0, 0, new int[0]);
        // ```
        //
        // Genuinely client-side, unlike the chest and note sounds above: `spawnParticle` reaches
        // the implemented `RenderGlobal.spawnParticle`, and no packet carries these.
        //
        // Three notes on the transcription, each of which looks like a bug and is not:
        //
        // 1. **One offset, two particles.** `d3`/`d4`/`d5` are drawn once and both spawns reuse
        //    them, so the smoke and the flame are always exactly co-located. Drawing per
        //    particle would be six `nextFloat()`s and a visibly different effect.
        // 2. **No `+ 0.5`.** The offset is `blockpos.getX() + nextFloat()`, i.e. uniform over the
        //    whole block, not jittered around its centre.
        // 3. **Inside the `isActivated()` guard**, so a spawner outside
        //    `RequiredPlayerRange` is completely inert — not smoking-but-not-spawning. That is
        //    why the position list comes from `World::activated_spawners` rather than from
        //    `tile_index`: the two answers must be the same one.
        // java: `TileEntityPiston.update`, run from `World.updateEntities`' `tickableTileEntities`
        // loop, which carries no `isRemote` guard. The tile pass consumes the interpolated state
        // below during the same render lifetime. GAP `A05-15` is closed by the live TESR route.
        self.tick_piston_tile_entities();
    }

    /// Advances every moving block and applies the shoves it hands out.
    ///
    /// java: `TileEntityPiston.update` (`tileentity/TileEntityPiston.java:166-199`) and
    /// `launchWithSlimeBlock` (`:97-145`). The launch is the mechanism by which a piston moves
    /// the **local player**: it runs `worldObj.getEntitiesWithinAABBExcludingEntity(null, box)`
    /// and calls `entity.moveEntity(...)` on everything it finds, on the client, every tick the
    /// block is extending.
    fn tick_piston_tile_entities(&mut self) {
        let Some(session) = self.session_mut() else {
            return;
        };
        let pushes = session.state_mut().world.tick_pistons();
        if pushes.is_empty() {
            return;
        }
        // The shove needs `&World` for the sweep and `&mut PhysicsState` for the player, and
        // `Session` hands those out through two methods that both borrow the whole struct
        // (`session.rs` belongs to another lane this wave, so no combined accessor is added).
        // `PhysicsState` is `Copy`, so it is taken out, moved against an immutable world, and
        // written back.
        let Some(session) = self.session() else {
            return;
        };
        let Some(mut player) = session.predictor().map(|p| p.state) else {
            return;
        };
        let world = &session.state().world;
        let mut moved = false;
        for push in pushes {
            // java: `getEntitiesWithinAABBExcludingEntity(null, aabb)` — a box test against
            // `Entity.getEntityBoundingBox`. Only the local player is modelled here; every other
            // entity is server-driven, so its shove arrives as a position update anyway.
            if !player.aabb().intersects_with(&push.area) {
                continue;
            }
            moved = true;
            let (ox, oy, oz) = push.facing.offset();
            if push.slime {
                // java: the `switch (pistonFacing.getAxis())` arm — `motion` on that one axis is
                // *assigned*, not added, and the other two are left alone.
                match push.facing.axis() {
                    mc_math::Axis::X => player.vel.x = f64::from(ox),
                    mc_math::Axis::Y => player.vel.y = f64::from(oy),
                    mc_math::Axis::Z => player.vel.z = f64::from(oz),
                }
            } else {
                // java: `entity.moveEntity(p2 * frontOffsetX, p2 * frontOffsetY, p2 * frontOffsetZ)`
                let d = f64::from(push.amount);
                mc_physics::movement::move_entity_full(
                    world,
                    &mut player,
                    DVec3::new(d * f64::from(ox), d * f64::from(oy), d * f64::from(oz)),
                );
            }
        }
        if moved {
            if let Some(predictor) = self.session_mut().and_then(|s| s.predictor_mut()) {
                predictor.state = player;
            }
        }
    }

    // -------------------------------------------------------------------------------------
    // Per frame
    // -------------------------------------------------------------------------------------

    /// Submit every visible tile entity.
    ///
    /// java: the `TileEntityRendererDispatcher` half of `RenderGlobal.renderEntities`
    /// (`:695-735`), including the per-TE distance cull and lightmap
    /// (`TileEntityRendererDispatcher.java:107-116`).
    pub(crate) fn upload_tile_entities(&mut self, gpu: &Gpu, camera: &Camera, partial_ticks: f32) {
        self.upload_tile_entities_with_clock(
            gpu,
            camera,
            partial_ticks,
            end_portal_system_time_millis,
        );
    }

    /// Submit visible tile entities with an injectable Java system clock.
    ///
    /// Keeping the clock at this seam makes end-portal phase tests deterministic without a
    /// process-global override, while the live wrapper above remains the only production caller.
    fn upload_tile_entities_with_clock(
        &mut self,
        gpu: &Gpu,
        camera: &Camera,
        partial_ticks: f32,
        end_portal_clock: fn() -> u64,
    ) {
        mc_core::profile_scope!("client.upload_tile_entities");
        let (Some(tiles), Some(beacons)) = (self.tiles.clone(), self.beacons.clone()) else {
            return;
        };
        let terrain = self
            .render
            .as_ref()
            .map(|render| std::rc::Rc::clone(&render.terrain));
        // TilePhase owns a separate GPU texture table from terrain/entities.  Synchronize the
        // persisted/live GameSettings switch at the same pre-upload boundary as the other tile
        // consumers so lazy atlas, banner, skin, portal, and destroy-stage uploads inherit it and
        // resident textures replay from their retained raw bytes on false -> true -> false.
        tiles.with(|phase| phase.set_anaglyph_with_gpu(gpu, self.settings.anaglyph));
        tiles.with(TilePhase::begin);
        beacons.with(BeaconPhase::begin);
        self.ensure_beacon_texture(gpu, &beacons);

        // Two passes, because the world read and the model pose want opposite borrows of
        // `Client`: everything the renderer needs about one tile entity is snapshotted here
        // (immutably, off the session) and posed below (mutably, off `tile_models`). That split
        // is vanilla's own shape too — `RenderGlobal` collects the visible tile entities into a
        // list first and only then runs the dispatcher over it.
        let mut draws: Vec<TileDraw> = Vec::new();
        // `RenderGlobal.damagedBlocks` is keyed by breaker id rather than block position. Take a
        // position index before borrowing the session so the TESR pass can consume every active
        // stage without keeping a mutable `Client` borrow alive. Stages outside 0..=9 are
        // removed by the driver's Java-compatible clear path and are defensively ignored here.
        let mut damaged_tile_stages: HashMap<BlockPos, Vec<u8>> = HashMap::new();
        for damage in self.damaged_blocks.values() {
            if damage.stage < 10 {
                damaged_tile_stages
                    .entry(damage.pos)
                    .or_default()
                    .push(damage.stage);
            }
        }
        for stages in damaged_tile_stages.values_mut() {
            stages.sort_unstable();
            stages.dedup();
        }
        // The non-vanilla `fullbright` option (default off). These draws take a CPU brightness
        // rather than a lightmap coordinate, so the all-white table cannot reach them; give them
        // the value a white lightmap texel would. Client light is still what the world holds.
        // The Fullbright mod's Full takes the same path.
        let fullbright = self.settings.fullbright || self.mod_tile_fullbright();
        // Snapshot the index before taking the mutable session/world borrow required by
        // `TileEntityBeacon.shouldBeamRender`'s renderer-side state transition.
        // (`MC_NW_F1_TILE_FAST=0` turns it off)
        let tile_fast = nw_f1_tile_fast();
        let indexed_tiles: Vec<(BlockPos, BlockPos, TileKind, SpawnerRenderAppearance)> = self
            .tile_index
            .iter()
            .filter_map(|(header_pos, kind)| {
                // A kind with no TESR (furnace, flower pot, hopper...) reaches
                // neither the pose loop nor any side effect below: skip its snapshot, world reads
                // and sort slot.
                if tile_fast && nw_f1_tile_draws_nothing(kind) {
                    return None;
                }
                let render_pos = self.tile_index.render_position(header_pos);
                // 06-entity-batch U6: the 64-block (beacon 256-block) range gate in the loop below
                // is a pure filter, so apply it while snapshotting too. Out-of-range tile
                // entities (most of a loaded map) then skip the spawner-appearance lookup and
                // clone, the per-TE section-walk `RefCell` borrow and the snapshot entry;
                // survivors keep their order and the loop's own gate (kept verbatim) still runs.
                if render_pos.distance_sq_to_center(camera.pos.x, camera.pos.y, camera.pos.z)
                    >= kind.max_render_distance_sq()
                {
                    return None;
                }
                // Only a spawner reads its appearance; every other position
                // misses the map and gets the same default.
                let spawner_appearance = if tile_fast && kind != TileKind::MobSpawner {
                    SpawnerRenderAppearance::default()
                } else {
                    self.tile_index.spawner_appearance(render_pos)
                };
                Some((header_pos, render_pos, kind, spawner_appearance))
            })
            .collect();
        let world_time;
        // The beam's `(totalWorldTime, partialTicks)`: vanilla's own pair
        // while the client and server clocks agree, smoothed across `S03` resync corrections so
        // the beam never turns backwards. `world_time` above stays vanilla's for the fade state.
        let beam_clock;
        let system_time_millis = end_portal_clock();
        let piston_mesh_settings = mc_render::terrain::MeshSettings {
            fancy_graphics: self.settings.fancy_graphics,
            ambient_occlusion: self.settings.ambient_occlusion > 0,
            // Piston-moving blocks never get connected textures (they apply to chunk meshing only).
            ..mc_render::terrain::MeshSettings::default()
        };
        let anaglyph = self.settings.anaglyph;
        {
            let Some(session) = self.session_mut() else {
                tiles.with(|phase| phase.finish(gpu));
                beacons.with(|phase| phase.finish(gpu));
                return;
            };
            let state = session.state_mut();
            world_time = state.env.total_world_time;
            beam_clock = state.env.render_world_time(partial_ticks);
            let world = &mut state.world;
            for (header_pos, pos, kind, spawner_appearance) in indexed_tiles {
                // java: `RenderGlobal.renderEntities` reads tile entities from the compiled chunks
                // of `renderInfos`, and a compile finds them by block position, so this is the map
                // key rather than the instance's own position. Beacons are the exception, because
                // `forceTileEntityRender` puts them in the global `setTileEntities` list. Damage
                // stages come from a separate loop over `damagedBlocks` with no section check, and
                // a double chest draws them from its `-X`/`-Z` half whichever half they name.
                let damaged = damaged_tile_stages.contains_key(&pos)
                    || (matches!(kind, TileKind::Chest | TileKind::TrappedChest)
                        && (damaged_tile_stages.contains_key(&pos.east())
                            || damaged_tile_stages.contains_key(&pos.south())));
                if kind != TileKind::Beacon
                    && !damaged
                    && terrain.as_ref().is_some_and(|terrain| {
                        !terrain
                            .borrow()
                            .section_reached_by_walk(header_pos.section_pos())
                    })
                {
                    continue;
                }
                // The map key is where the packet found the instance; Java's renderer uses the
                // instance's own `TileEntity.pos`, which S35's compound may have overwritten.
                // java: `renderTileEntity:109` — `getDistanceSq(entityX, entityY, entityZ) <
                // getMaxRenderDistanceSquared()`, which is 4096 for everything except the
                // beacon's 65536.
                if pos.distance_sq_to_center(camera.pos.x, camera.pos.y, camera.pos.z)
                    >= kind.max_render_distance_sq()
                {
                    continue;
                }
                // java: `TileEntityBeaconRenderer.renderTileEntityAt` calls the mutating
                // `shouldBeamRender()` only after `TileEntityRendererDispatcher`'s distance
                // check. This position-scoped call preserves out-of-range re-entry decay.
                if kind == TileKind::Beacon {
                    world.advance_beacon_render_frame(header_pos, world_time);
                }
                let stored = world.tile_entity(header_pos);
                let mut draw = TileDraw {
                    pos,
                    kind,
                    spawner_appearance,
                    rel: camera.relative(pos.to_vec3()),
                    metadata: world.block_at(pos).meta(),
                    // java: `world.getCombinedLight(pos, 0)` — the per-TE lightmap coordinate.
                    light: if fullbright {
                        1.0
                    } else {
                        light_brightness(world.light_at(pos))
                    },
                    destroy_stages: damaged_tile_stages.get(&pos).cloned().unwrap_or_default(),
                    ..TileDraw::default()
                };
                match kind {
                    TileKind::Chest | TileKind::TrappedChest => {
                        let adj = world.chest_neighbours(pos);
                        // java: the whole draw is inside `if (adjacentChestZNeg == null &&
                        // adjacentChestXNeg == null)`, so a double chest is drawn exactly once.
                        if !adj.draws() {
                            continue;
                        }
                        draw.double = adj.is_double();
                        let (prev, lid, neighbours) =
                            mc_world::chest_lid_inputs(world, pos, partial_ticks);
                        draw.lid = (prev, lid);
                        draw.neighbour_lids = neighbours;
                        // java: `if (i == 2 && adjacentChestXPos != null) translate(1, 0, 0);`
                        // and `if (i == 5 && adjacentChestZPos != null) translate(0, 0, -1);`
                        draw.double_offset = if draw.metadata == 2 && adj.x_pos {
                            Vec3::new(1.0, 0.0, 0.0)
                        } else if draw.metadata == 5 && adj.z_pos {
                            Vec3::new(0.0, 0.0, -1.0)
                        } else {
                            Vec3::ZERO
                        };
                        // Vanilla draws one large chest from its `-X`/`-Z` half. Damage packets
                        // may name the other half, so anchor those stages to the same model.
                        if adj.x_pos {
                            draw.destroy_stages.extend(
                                damaged_tile_stages
                                    .get(&pos.east())
                                    .into_iter()
                                    .flatten()
                                    .copied(),
                            );
                        }
                        if adj.z_pos {
                            draw.destroy_stages.extend(
                                damaged_tile_stages
                                    .get(&pos.south())
                                    .into_iter()
                                    .flatten()
                                    .copied(),
                            );
                        }
                        draw.destroy_stages.sort_unstable();
                        draw.destroy_stages.dedup();
                    }
                    TileKind::EnderChest => {
                        if let Some(chest) = stored.and_then(mc_world::TileEntity::chest) {
                            draw.lid = (chest.prev_lid_angle, chest.lid_angle);
                        }
                    }
                    TileKind::StandingSign | TileKind::WallSign => {
                        draw.sign = stored
                            .and_then(mc_world::TileEntity::sign)
                            .map(|s| (s.lines.clone(), s.line_being_edited));
                    }
                    TileKind::StandingBanner | TileKind::WallBanner => {
                        // java: `TileEntityBannerRenderer.func_178463_a` returns null until
                        // `readFromNBT`/`setItemValues` has set `field_175119_g`; a lazily created
                        // banner therefore contributes no geometry before its S35 action-6.
                        draw.banner = banner_state_for_render(stored);
                    }
                    TileKind::Skull => {
                        if let Some(skull) = stored.and_then(mc_world::TileEntity::skull) {
                            draw.skull = (
                                skull.skull_type,
                                skull.rotation,
                                skull
                                    .owner_profile
                                    .as_ref()
                                    .map(player_skull_owner_from_profile),
                            );
                        }
                    }
                    TileKind::EnchantmentTable => {
                        draw.enchant = stored
                            .and_then(mc_world::TileEntity::enchant)
                            .copied()
                            .unwrap_or_default();
                    }
                    TileKind::MobSpawner => {
                        draw.spawner = stored
                            .and_then(mc_world::TileEntity::spawner)
                            .filter(|s| !s.entity_id.is_empty())
                            .map(|s| {
                                (
                                    s.entity_id.clone(),
                                    s.prev_mob_rotation,
                                    s.mob_rotation,
                                    s.spawn_data.clone(),
                                )
                            });
                    }
                    TileKind::Beacon => {
                        draw.beacon = stored.and_then(mc_world::TileEntity::beacon).cloned();
                    }
                    TileKind::Piston => {
                        // java: the moving block has no lazily-created tile entity; the piston
                        // event installs its state explicitly, and the renderer skips a carrier
                        // whose state has not arrived yet.
                        draw.piston = stored.and_then(mc_world::TileEntity::piston).copied();
                        if let Some(piston) = draw.piston {
                            let mut geometry = Vec::new();
                            emit_piston_lit(
                                world,
                                pos,
                                piston,
                                draw.rel,
                                partial_ticks,
                                piston_mesh_settings,
                                anaglyph,
                                &mut geometry,
                            );
                            draw.piston_geometry = Some(geometry);
                        }
                    }
                    _ => {}
                }
                draws.push(draw);
            }
        }

        let mut verts: Vec<EntityVertex> = Vec::new();
        let mut sign_text: Vec<(mc_render::entity::Mat4, [String; 4], i32, f32)> = Vec::new();
        // Snapshot the cache before borrowing `tile_models` mutably.  The body/layer pass below
        // is intentionally a pure consumer of this vector, just like the entity phase's
        // render-state snapshot.
        let spawners: Vec<SpawnerUpload> = draws
            .iter()
            .filter_map(|draw| {
                let (entity_id, previous_rotation, rotation, spawn_data) = draw.spawner.as_ref()?;
                Some((
                    draw.pos,
                    entity_id.clone(),
                    *previous_rotation,
                    *rotation,
                    spawn_data.clone(),
                    draw.spawner_appearance.clone(),
                    self.tile_index.spawner_animation(draw.pos, entity_id),
                ))
            })
            .collect();
        // 06-entity-batch U3: a deterministic, texture-grouped order lets consecutive pushes
        // share one batch (previously std `HashMap` order: about one draw per tile entity in a
        // mixed scene, and a different order on every run). The ordinary TESR pipeline has no
        // blending, writes depth with LEQUAL and discards by alpha, so each pixel keeps the
        // nearest surviving fragment in any order; only exact depth ties depend on order, and
        // those were already decided by hash order. Each crack replay is still pushed straight
        // after its own base. The spawner snapshot above is taken before sorting, so miniature
        // mobs keep their previous order.
        draws.sort_by_key(tile_draw_order_key);
        let mut lit_scratch: Vec<TileVertex> = Vec::new();
        // Disjoint field borrows, taken once: the pose loop needs `tile_models` mutably while
        // still reading the resource manager for texture loads.
        let models = &mut self.tile_models;
        let resources = &self.assets.resources;
        let atlas = &self.assets.atlas;

        for draw in &draws {
            verts.clear();
            let texture: Option<String> = match draw.kind {
                TileKind::Chest | TileKind::TrappedChest => Some(
                    models
                        .emit_chest(
                            draw.rel,
                            draw.metadata,
                            draw.kind == TileKind::TrappedChest,
                            draw.double,
                            draw.double_offset,
                            draw.lid.0,
                            draw.lid.1,
                            &draw.neighbour_lids,
                            partial_ticks,
                            &mut verts,
                        )
                        .to_owned(),
                ),
                TileKind::EnderChest => Some(
                    models
                        .emit_ender_chest(
                            draw.rel,
                            draw.metadata,
                            draw.lid.0,
                            draw.lid.1,
                            partial_ticks,
                            &mut verts,
                        )
                        .to_owned(),
                ),
                TileKind::StandingSign | TileKind::WallSign => {
                    let standing = draw.kind == TileKind::StandingSign;
                    let (texture, board) =
                        models.emit_sign(draw.rel, standing, draw.metadata, &mut verts);
                    // java: `RenderGlobal.renderEntities:692-705` draws every tile entity with
                    // `destroyStage = -1` first, so the sign's text loop always runs. Only the
                    // separate damaged-block replay (`:709-735`) passes a stage, and that call
                    // skips the text; here it is the crack pass below, which replays geometry.
                    if let Some((lines, editing)) = &draw.sign {
                        sign_text.push((
                            sign_text_matrix(board),
                            lines.clone(),
                            *editing,
                            sign_text_shade(board, draw.light),
                        ));
                    }
                    Some(texture.to_owned())
                }
                TileKind::StandingBanner | TileKind::WallBanner => {
                    let Some(banner) = draw.banner.as_ref() else {
                        // java: `func_178463_a` returns null while field_175119_g is false, and
                        // the renderer wraps the whole model draw in that null check.
                        continue;
                    };
                    models.emit_banner(
                        draw.rel,
                        draw.kind == TileKind::StandingBanner,
                        draw.metadata,
                        draw.pos.x,
                        draw.pos.y,
                        draw.pos.z,
                        world_time,
                        partial_ticks,
                        &mut verts,
                    );
                    tiles.with(|phase| phase.load_banner_texture(gpu, resources, banner))
                }
                TileKind::Skull => {
                    let owner = draw.skull.2.as_ref();
                    let texture = models.emit_skull_with_owner(
                        draw.rel,
                        draw.metadata,
                        draw.skull.0,
                        draw.skull.1,
                        owner,
                        &mut verts,
                    );
                    if let Some(owner) = owner {
                        // `SkinCache` publishes normalized pixels through the shared handoff.
                        // Uploading into the tile phase here keeps the normal render loop
                        // non-blocking: before the download arrives, bind the Java Steve/Alex
                        // fallback; on the next frame, bind the profile skin under its stable key.
                        let fallback = owner.fallback_model.default_texture();
                        if texture != fallback {
                            let ready = tiles.with(|phase| {
                                if phase.has_texture(&texture) {
                                    return true;
                                }
                                let Some(skin) = mc_render::entity::skin::shared_skin(&texture)
                                else {
                                    return false;
                                };
                                phase.upload_texture(
                                    gpu,
                                    &texture,
                                    &skin.rgba,
                                    mc_render::entity::SkinTexture::WIDTH,
                                    mc_render::entity::SkinTexture::HEIGHT,
                                );
                                true
                            });
                            if ready {
                                Some(texture)
                            } else {
                                Some(fallback.to_owned())
                            }
                        } else {
                            Some(texture)
                        }
                    } else {
                        // Non-player skull types keep their fixed texture. Type 3 without an
                        // owner follows Java's legacy Steve fallback through the render helper.
                        Some(texture)
                    }
                }
                TileKind::EnchantmentTable => {
                    let e = draw.enchant;
                    let base = enchant_book_matrix(
                        draw.rel,
                        e.tick_count,
                        e.book_rotation_prev,
                        e.book_rotation,
                        partial_ticks,
                    );
                    let flips = enchant_page_flips(e.page_flip_prev, e.page_flip, partial_ticks);
                    let spread =
                        e.book_spread_prev + (e.book_spread - e.book_spread_prev) * partial_ticks;
                    Some(
                        models
                            .emit_book(base, e.tick_count, partial_ticks, flips, spread, &mut verts)
                            .to_owned(),
                    )
                }
                TileKind::MobSpawner => None,
                TileKind::Beacon => {
                    if let Some(beacon) = &draw.beacon {
                        let (beam_ticks, beam_partial) = beam_clock;
                        emit_beacon(&beacons, draw.rel, beacon, beam_ticks, beam_partial);
                    }
                    None
                }
                TileKind::Piston => {
                    let Some(piston) = draw.piston.as_ref() else {
                        // java: a BlockPistonMoving carrier only gets a TileEntity when the
                        // piston block event installs it; without that state there is nothing
                        // for TileEntityPistonRenderer to dispatch.
                        continue;
                    };
                    // Geometry was lit while the immutable world snapshot was live. Keep this
                    // state check in the pose arm so a stale index cannot submit a carrier.
                    if draw.piston_geometry.is_none() {
                        continue;
                    }
                    let _ = piston;
                    // Baked piston quads carry stitched-atlas UVs, while TilePhase's normal tile
                    // textures are per-entity files. Keep the same key as EntityPhase and upload
                    // this phase's binding lazily because Client::install cannot be edited in
                    // this lane.
                    tiles.with(|phase| {
                        if !phase.has_texture(BLOCK_ATLAS_TEXTURE) {
                            phase.upload_texture(
                                gpu,
                                BLOCK_ATLAS_TEXTURE,
                                &atlas.texture,
                                atlas.w,
                                atlas.h,
                            );
                        }
                    });
                    Some(BLOCK_ATLAS_TEXTURE.to_owned())
                }
                // The end portal uses the dedicated sixteen-layer path above. No TESR exists in
                // vanilla for the note or command block; flower pots still need a block-model
                // variant selected from tile data.
                TileKind::EndPortal => {
                    // java: `TileEntityEndPortalRenderer.renderTileEntityAt:22-110` binds
                    // `end_sky.png` for layer zero and `end_portal.png` for layers one through
                    // fifteen. The tile phase owns both uploads and expands the fixed-function
                    // texgen into explicit projective UVs before recording its two blend states.
                    let loaded = tiles.with(|phase| phase.load_end_portal_textures(gpu, resources));
                    if loaded {
                        let _ =
                            tiles.with(|phase| phase.push_end_portal(draw.rel, system_time_millis));
                    }
                    None
                }
                TileKind::Dispenser
                | TileKind::Furnace
                | TileKind::Jukebox
                | TileKind::BrewingStand
                | TileKind::Comparator
                | TileKind::DaylightDetector
                | TileKind::Hopper
                | TileKind::Dropper
                | TileKind::Note
                | TileKind::CommandBlock
                | TileKind::FlowerPot => None,
            };

            if let Some(texture) = texture {
                let is_piston = draw.kind == TileKind::Piston;
                let geometry: &[TileVertex] = if is_piston {
                    draw.piston_geometry.as_deref().unwrap_or_default()
                } else {
                    lit_into(&verts, draw.light, &mut lit_scratch);
                    &lit_scratch
                };
                if geometry.is_empty() {
                    continue;
                }
                tiles.with(|phase| {
                    let key = if phase.load_texture(gpu, resources, &texture) {
                        texture.as_str()
                    } else {
                        TilePhase::FALLBACK_TEXTURE
                    };
                    phase.push_blended(key, tile_blend(draw.kind), geometry);

                    // `RenderGlobal.drawBlockDamageTexture` excludes tile-entity blocks from the
                    // baked terrain overlay. Their destroy stages are still visible in vanilla:
                    // the TESR geometry is replayed under the matching `destroy_stage_N` sprite.
                    // Keep this after the ordinary tile draw so the crack has equal-depth access
                    // to the chest/sign/skull surface, and clamp defensively to the ten legal
                    // stages even if a malformed remote packet reaches this loop.
                    // The banner TESR intentionally ignores `destroyStage`; unlike chests,
                    // signs, and skulls it remains on the ordinary terrain-damage path in Java.
                    if let Some((scale_u, scale_v)) = destroy_stage_scale(draw.kind, draw.double) {
                        for stage in legal_destroy_stages(&draw.destroy_stages) {
                            let crack = format!("textures/blocks/destroy_stage_{stage}.png");
                            if phase.load_texture(gpu, resources, &crack) {
                                let crack_geometry =
                                    remap_destroy_stage_uvs(geometry, scale_u, scale_v);
                                phase.push(&crack, &crack_geometry);
                            }
                        }
                    }
                });
            }
        }

        tiles.with(|phase| phase.finish(gpu));
        beacons.with(|phase| phase.finish(gpu));

        // The miniature mobs go through the *entity* phase, because that is literally what they
        // are: `TileEntityMobSpawnerRenderer.renderMob` calls
        // `RenderManager.renderEntityWithPosYaw`.
        self.upload_spawner_mobs(gpu, camera, partial_ticks, &spawners);
        self.upload_sign_text(gpu, &sign_text);
    }

    /// java: `TileEntityBeaconRenderer` binds `beacon_beam.png` on every draw; here it is a
    /// one-off upload the first frame the phase is used.
    fn ensure_beacon_texture(&self, gpu: &Gpu, beacons: &super::driver::Shared<BeaconPhase>) {
        beacons.with(|phase| {
            if phase.has_texture() {
                return;
            }
            let Ok(location) =
                mc_assets::ResourceLocation::parse("textures/entity/beacon_beam.png")
            else {
                return;
            };
            let Ok(bytes) = self.assets.resources.load(&location) else {
                return;
            };
            // <<W21-SL:texture-size>> refused (missing) when the device cannot hold it.
            let Ok(image) = mc_assets::decode_texture_png(&bytes) else {
                return;
            };
            let image = image.into_rgba8();
            let (w, h) = image.dimensions();
            phase.upload_texture(gpu, &image.into_raw(), w, h);
        });
    }

    /// java: `TileEntityMobSpawnerRenderer.renderMob` (`:22-36`).
    fn upload_spawner_mobs(
        &mut self,
        gpu: &Gpu,
        camera: &Camera,
        partial_ticks: f32,
        spawners: &[SpawnerUpload],
    ) {
        if spawners.is_empty() {
            return;
        }
        let Some(render) = self.render.as_ref() else {
            return;
        };
        let mut verts = Vec::new();
        let mut layer_verts = Vec::new();
        for (pos, entity_id, prev, cur, spawn_data, appearance, animation) in spawners {
            let Some(entity_type) = mc_data::entity_by_name(entity_id) else {
                continue;
            };
            let variant = spawn_data.as_ref().map_or(0, |data| data.variant);
            let Some(desc) = mc_render::entity::renderer_for_variant(entity_type, variant) else {
                continue;
            };
            let mut params = spawner_anim_params(*animation, partial_ticks);
            apply_spawner_spawn_data(&mut params, entity_type, spawn_data.as_ref());
            apply_spawner_render_appearance(&mut params, appearance);
            let rel = camera.relative(pos.to_vec3());
            // RenderManager's ordinary living placement includes the species callbacks that can
            // still be selected by SpawnData (bat hanging and villager-zombie conversion). The
            // spawner's own matrix remains the outer Java TESR placement and keeps its exact
            // rotation/0.4375 miniature scale.
            let dynamic_scale = mc_render::entity::manager::entity_pre_render_scale(
                entity_type,
                variant,
                params.entity.child,
                params.entity.squish_factor,
                0.0,
            );
            let entity_scale = if entity_type == mc_data::EntityType::WitherBoss
                && appearance.wither_invul_time > 0
            {
                // RenderWither replaces its settled descriptor scale during the 220-tick spawn
                // shield; multiplying by the normal 2.0 descriptor would double-scale it.
                mc_render::entity::wither_pre_render_scale(
                    appearance.wither_invul_time,
                    partial_ticks,
                )
            } else {
                desc.scale
            };
            let transform = mc_render::entity::render::EntityTransform {
                scale: entity_scale,
                upside_down: spawner_upside_down(entity_type, appearance),
                upside_down_height: entity_type.info().height
                    * if params.entity.child { 0.5 } else { 1.0 },
                ..Default::default()
            };
            let entity_base = if matches!(
                entity_type,
                mc_data::EntityType::Slime
                    | mc_data::EntityType::LavaSlime
                    | mc_data::EntityType::Creeper
            ) || (entity_type == mc_data::EntityType::Villager
                && params.entity.child)
            {
                mc_render::entity::render::entity_matrix_scaled(&transform, dynamic_scale)
            } else {
                mc_render::entity::manager::placement_matrix_for(entity_type, &transform, &params)
            };
            let base = spawner_matrix(rel, *prev, *cur, partial_ticks) * entity_base;
            // `MobSpawnerBaseLogic.func_180612_a` supplies a cached entity to the normal renderer,
            // but `spawnNewEntity(entity, false)` never inserts it into the world.  Therefore its
            // constructor age/limb/head pose is still zero; only the renderer's partial tick is
            // retained by `spawner_anim_params` below.  The tile's activation gate belongs to
            // smoke/flame/Delay/rotation, not to a fabricated cached-entity tick.
            verts.clear();
            if entity_type == mc_data::EntityType::EntityHorse {
                let data = spawn_data.as_ref();
                let state = mc_render::entity::models::passive::HorseRenderState {
                    horse_type: variant,
                    adult: !params.entity.child,
                    saddled: data.is_some_and(|data| data.saddled),
                    chested: data.is_some_and(|data| data.horse_chested),
                    // `Riding` means this outer horse is the rider, not that it has a rider.
                    // The nested mount is attached by Java for pose state but is not submitted by
                    // TileEntityMobSpawnerRenderer, so the horse's own reins remain hidden.
                    ridden: false,
                    ..mc_render::entity::models::passive::HorseRenderState::default()
                };
                self.tile_models
                    .emit_horse_with_state(base, &params, &state, &mut verts);
            } else {
                self.models.emit_with(desc.model, base, &params, &mut verts);
            }
            if verts.is_empty() {
                continue;
            }
            let horse_selection = (entity_type == mc_data::EntityType::EntityHorse).then(|| {
                let data = spawn_data.as_ref();
                mc_render::entity::manager::horse_texture_selection(
                    variant,
                    data.map_or(0, |data| data.horse_variant),
                    data.map_or(0, |data| data.horse_armor),
                )
            });
            let named = spawner_unformatted_name(appearance);
            let texture = if entity_type == mc_data::EntityType::WitherBoss
                && spawner_wither_uses_invulnerable_texture(appearance.wither_invul_time)
            {
                mc_render::entity::manager::texture_for(entity_type, 1)
            } else {
                mc_render::entity::manager::texture_for_named(
                    entity_type,
                    variant,
                    named.as_deref(),
                )
            };
            let resources = &self.assets.resources;
            let atlas = &self.assets.atlas;
            render.entities.with(|phase| {
                let key = if let Some(selection) = horse_selection.as_ref() {
                    if phase.load_horse_texture(gpu, resources, selection) {
                        selection.upload_key()
                    } else {
                        selection.fallback
                    }
                } else if phase.load_texture(gpu, resources, texture) {
                    texture
                } else {
                    mc_render::entity::EntityPhase::FALLBACK_TEXTURE
                };
                phase.push(key, &verts);

                // Java's RenderManager runs the cached entity through the same ordered layer
                // list as a world entity. SpawnData now supplies the ordinary appearance fields
                // (saddle, dye, carried block and equipment) that those layers read.
                for &layer in desc.layers {
                    if !spawner_layer_should_render_with_spawn_data(
                        entity_type,
                        layer,
                        &params,
                        spawn_data.as_ref(),
                    ) {
                        continue;
                    }
                    if matches!(
                        layer,
                        mc_render::entity::LayerKind::HeldItem
                            | mc_render::entity::LayerKind::WitchHeldItem
                            | mc_render::entity::LayerKind::CustomHead
                    ) {
                        if let Some(data) = spawn_data.as_ref() {
                            let _ = emit_spawner_equipment_layer(
                                &mut self.models,
                                &mut self.tile_models,
                                phase,
                                gpu,
                                resources,
                                atlas,
                                entity_type,
                                layer,
                                desc.model,
                                base,
                                &params,
                                data,
                                &mut layer_verts,
                            );
                        }
                        continue;
                    }
                    if layer == mc_render::entity::LayerKind::Armor {
                        emit_spawner_armor_layers(
                            &mut self.models,
                            phase,
                            gpu,
                            resources,
                            entity_type,
                            desc.model,
                            base,
                            &params,
                            spawn_data.as_ref(),
                            &mut layer_verts,
                        );
                        continue;
                    }
                    if emit_spawner_dynamic_layer(
                        &mut self.models,
                        phase,
                        gpu,
                        atlas,
                        entity_type,
                        layer,
                        desc.model,
                        base,
                        &params,
                        &mut layer_verts,
                    ) {
                        continue;
                    }
                    let Some(layer_texture) = layer.texture().or_else(|| {
                        (layer == mc_render::entity::LayerKind::SlimeGel).then_some(texture)
                    }) else {
                        continue;
                    };
                    layer_verts.clear();
                    self.models
                        .emit_layer(layer, &verts, &params, base, &mut layer_verts);
                    if layer_verts.is_empty() {
                        continue;
                    }
                    mc_render::entity::tint(
                        &mut layer_verts,
                        layer.resolved_tint(spawner_layer_tint(
                            entity_type,
                            spawn_data.as_ref(),
                            &params,
                            appearance,
                            partial_ticks,
                        )),
                    );
                    if let Some(offset) = layer.texture_scroll(params.age_in_ticks) {
                        mc_render::entity::offset_uv(&mut layer_verts, offset);
                    }
                    let key = if phase.load_texture(gpu, resources, layer_texture) {
                        layer_texture
                    } else {
                        mc_render::entity::EntityPhase::FALLBACK_TEXTURE
                    };
                    phase.push_blended(key, layer.blend_for(params.entity.invisible), &layer_verts);
                }
                // An enderman's held block is the one SpawnData layer whose geometry does not
                // come from `LayerKind::model`; emit it against the shared stitched atlas.
                if entity_type == mc_data::EntityType::Enderman {
                    if let Some(block_state) =
                        spawn_data.as_ref().and_then(|data| data.carried_block)
                    {
                        emit_spawner_carried_block(
                            &mut self.models,
                            phase,
                            gpu,
                            resources,
                            atlas,
                            desc.model,
                            base,
                            block_state,
                            &mut layer_verts,
                        );
                    }
                }
            });
        }
        // Client::frame finalizes the shared entity phase after all producers.
    }

    /// java: `TileEntitySignRenderer.java:104-133` — the four lines, each wrapped to 90 px with
    /// only the first wrapped line kept, centred, under `depthMask(false)`.
    fn upload_sign_text(
        &mut self,
        gpu: &Gpu,
        signs: &[(mc_render::entity::Mat4, [String; 4], i32, f32)],
    ) {
        if signs.is_empty() {
            return;
        }
        let Some(render) = self.render.as_ref() else {
            return;
        };
        let mut pending: Vec<(
            mc_ui::font::FontTexture,
            Vec<NameplateGlyphVertex>,
            Vec<NameplateVertex>,
        )> = Vec::new();
        self.sign_text_cache.begin_frame(&self.font);
        for (matrix, lines, editing, shade) in signs {
            for (index, line) in lines.iter().enumerate() {
                let selected = *editing == index as i32;
                if !should_render_sign_line(line, selected) {
                    continue;
                }
                let mut sink = SignTextSink {
                    matrix: *matrix,
                    shade: *shade,
                    batches: Vec::new(),
                    bars: Vec::new(),
                };
                if let Some(mesh) = self.sign_text_cache.get(line, index, selected) {
                    // The row's font output is unchanged since it was recorded; replaying it
                    // through the same sink is the same as drawing it again.
                    for quad in &mesh.glyphs {
                        sink.glyph(*quad);
                    }
                    for rect in &mesh.bars {
                        sink.bar(*rect);
                    }
                } else {
                    // java: `GuiUtilRenderComponents.splitText(component, 90, fontrenderer,
                    // false, true)` and then `list.get(0)` — only the *first* wrapped line is
                    // drawn, which is why a long sign line is silently truncated rather than
                    // overflowing.
                    let wrapped = self.font.wrap_to_lines(line, 90);
                    let mut text = wrapped.first().cloned().unwrap_or_default();
                    // java: `if (j == te.lineBeingEdited) s = "> " + s + " <";`
                    if selected {
                        text = format!("> {text} <");
                    }
                    let width = self.font.string_width(&text);
                    let x = (-width / 2) as f32;
                    let y = tile::sign_line_y(index);
                    // java: `drawString(s, -getStringWidth(s) / 2, j * 10 - signText.length *
                    // 5, 0)` — colour `0`, i.e. black, and no drop shadow.
                    if self.font.may_obfuscate(&text) {
                        // `§k` re-rolls every draw from the shared stream: never cached.
                        self.font.draw_string(&mut sink, &text, x, y, 0, false);
                    } else {
                        let mut mesh = mc_ui::font::TextMesh::new();
                        self.font.draw_string(&mut mesh, &text, x, y, 0, false);
                        for quad in &mesh.glyphs {
                            sink.glyph(*quad);
                        }
                        for rect in &mesh.bars {
                            sink.bar(*rect);
                        }
                        self.sign_text_cache.insert(line, index, selected, mesh);
                    }
                }
                for (texture, verts) in sink.batches.drain(..) {
                    pending.push((texture, verts, Vec::new()));
                }
                if let Some((_, _, bars)) = pending.last_mut() {
                    bars.append(&mut sink.bars);
                }
            }
        }
        if pending.is_empty() {
            return;
        }
        // The ascii sheet is uploaded at install; a unicode page is not, so ask for whatever
        // this frame's glyphs need before pushing them. A page that will not load is not fatal —
        // `NameplatePhase` falls back to a 1x1 white texel, so the line draws as boxes rather
        // than vanishing, which is S12's stated policy for a texture that is not resident.
        render.nameplates.with(|phase| {
            for (texture, _, _) in &pending {
                let key = mc_ui::nameplate::font_texture_key(*texture);
                if phase.has_font_texture(&key) {
                    continue;
                }
                match texture {
                    mc_ui::font::FontTexture::Ascii => {
                        mc_ui::nameplate::upload_ascii_font(phase, gpu, &self.assets.font);
                    }
                    mc_ui::font::FontTexture::StandardGalactic => {
                        let _ = mc_ui::nameplate::upload_standard_galactic(
                            phase,
                            gpu,
                            &self.assets.resources,
                        );
                    }
                    mc_ui::font::FontTexture::UnicodePage(page) => {
                        let _ = mc_ui::nameplate::upload_unicode_page(
                            phase,
                            gpu,
                            &self.assets.resources,
                            *page,
                        );
                    }
                }
            }
            for (texture, verts, bars) in &pending {
                if !bars.is_empty() {
                    // `NameplatePass::SignText` uses the same Less/depth-write-off state for
                    // solid decorations as for textured glyphs.
                    phase.push(NameplatePass::SignText, bars);
                }
                // java: `TileEntitySignRenderer:110` — `depthMask(false)` with the depth *test*
                // left on. The dedicated `SignText` pass is depth-tested without writing depth;
                // the generic `InFront` pass remains reserved for entity nameplates.
                // This matches vanilla's `depthMask(false)` exactly.
                // The same key, without a `String` per sign line.
                phase.push_glyphs(
                    NameplatePass::SignText,
                    mc_ui::nameplate::font_texture_key_str(*texture),
                    verts,
                );
            }
            // `upload_entities` already called `finish` this frame; re-running it re-uploads
            // the buffer with the glyphs just appended.
            phase.finish(gpu);
        });
    }
}

/// java: `TileEntityBeaconRenderer.renderTileEntityAt` (`:30-131`) — one inner and one outer
/// shell per segment, stacked from the beacon upward.
fn emit_beacon(
    beacons: &super::driver::Shared<BeaconPhase>,
    rel: Vec3,
    beacon: &mc_world::BeaconState,
    world_time: i64,
    partial_ticks: f32,
) {
    // java: `float f = te.shouldBeamRender(); if (f > 0.0F) { ... }`
    if beacon.beam_fade <= 0.0 || !beacon.complete {
        return;
    }
    let mut inner = Vec::new();
    let mut outer = Vec::new();
    let mut base_height = 0;
    for segment in &beacon.segments {
        if segment.height <= 0 {
            base_height += segment.height;
            continue;
        }
        let top = base_height + segment.height;
        let beam = tile::beacon_beam(world_time, partial_ticks, segment.height, beacon.beam_fade);
        let xz = [rel.x, rel.z];
        tile::emit_beacon_inner_shell(
            xz,
            rel.y + base_height as f32,
            rel.y + top as f32,
            &beam,
            segment.colors,
            &mut inner,
        );
        tile::emit_beacon_outer_shell(
            xz,
            rel.y + base_height as f32,
            rel.y + top as f32,
            &beam,
            segment.colors,
            &mut outer,
        );
        base_height = top;
    }
    if inner.is_empty() && outer.is_empty() {
        return;
    }
    beacons.with(|phase| {
        phase.push_inner(&inner);
        phase.push_outer(&outer);
    });
}

/// java: `TileEntitySkull.readFromNBT` (`:33-55`) plus the `Owner` compound's embedded texture
/// property, which carries the skin URL directly — no session-server round trip.
/// `TileEntity.readFromNBT` uses the same numeric `getInteger` convention for its own position;
/// keep that wire rule in one helper so a packet header is never mistaken for the stored render
/// position when the compound deliberately disagrees.
#[must_use]
fn tile_entity_payload_position(nbt: &mc_nbt::NbtCompound) -> BlockPos {
    BlockPos::new(
        nbt.get_i32("x").unwrap_or(0),
        nbt.get_i32("y").unwrap_or(0),
        nbt.get_i32("z").unwrap_or(0),
    )
}

fn read_skull(nbt: &mc_nbt::NbtCompound) -> mc_world::SkullState {
    let mut state = mc_world::SkullState {
        skull_type: nbt.get_i8("SkullType").unwrap_or(0) as u8,
        rotation: nbt.get_i8("Rot").unwrap_or(0) as u8,
        owner_name: None,
        owner_profile: None,
        skin_url: None,
    };
    if state.skull_type != 3 {
        return state;
    }
    if let Some(owner) = nbt.get_compound("Owner") {
        state.owner_profile = mc_auth::read_game_profile_from_nbt(owner);
        if let Some(profile) = &state.owner_profile {
            // NBTUtil.readGameProfileFromNBT preserves a null/missing Name when an Id-only
            // profile is received; SkullState uses None for that Java-visible null rather than
            // exposing Rust's empty-string representation.
            state.owner_name = (!profile.name.is_empty()).then(|| profile.name.clone());
            state.skin_url = profile
                .properties
                .iter()
                .find(|property| property.name == "textures")
                .and_then(|property| super::appearance::decode_textures_property(&property.value))
                .and_then(|textures| textures.skin_url);
        }
    } else if let Some(legacy) = nbt.get_str("ExtraType") {
        // java: `new GameProfile(null, s)` then `updateGameprofile`, which needs the session
        // server. Offline, the name is all there is. Keep the profile object as Java does, so
        // write/read and the renderer's name-only fallback observe the same state.
        if !legacy.is_empty() {
            state.owner_name = Some(legacy.to_owned());
            state.owner_profile = Some(mc_auth::GameProfile {
                id: None,
                name: legacy.to_owned(),
                properties: Vec::new(),
            });
        }
    }
    state
}

/// java: `MobSpawnerBaseLogic.readFromNBT` (`:249-296`). `Delay` is read unconditionally,
/// so a missing or non-numeric tag overwrites the field initializer with Java's getter default 0.
fn read_spawner(nbt: &mc_nbt::NbtCompound) -> mc_world::SpawnerState {
    let default = mc_world::SpawnerState::default();
    let entity_id = nbt
        .get_str("EntityId")
        .map_or_else(|| default.entity_id.clone(), str::to_owned);
    mc_world::SpawnerState {
        entity_id: entity_id.clone(),
        spawn_delay: i32::from(nbt.get_i16("Delay").unwrap_or(0)),
        min_spawn_delay: nbt
            .get_i16("MinSpawnDelay")
            .map_or(default.min_spawn_delay, i32::from),
        activating_range: nbt
            .get_i16("RequiredPlayerRange")
            .map_or(default.activating_range, i32::from),
        mob_rotation: default.mob_rotation,
        prev_mob_rotation: default.prev_mob_rotation,
        // java: `readFromNBT` installs a fresh `WeightedRandomMinecart` whenever the exact
        // `SpawnData` compound is present, and clears it when absent.  The world crate cannot
        // depend on mc-nbt, so retain the render-visible overlay as a typed value here.
        spawn_data: nbt
            .get_compound("SpawnData")
            .map(|spawn_data| read_spawner_spawn_data(&entity_id, spawn_data)),
    }
}

/// Decode only the raw `SpawnData` keys that the current entity renderer reads directly and that
/// are not represented by the shared typed snapshot. Wrong/missing tags deliberately resolve to
/// Java's freshly-constructed defaults; an absent `SpawnData` compound clears the overlay.
fn read_spawner_render_appearance(
    entity_id: &str,
    spawn_data: Option<&mc_nbt::NbtCompound>,
) -> SpawnerRenderAppearance {
    let Some(nbt) = spawn_data else {
        return SpawnerRenderAppearance::default();
    };
    SpawnerRenderAppearance {
        custom_name: nbt
            .get_str("CustomName")
            .filter(|name| !name.is_empty())
            .map(str::to_owned),
        wither_invul_time: if entity_id == "WitherBoss" {
            nbt.get_i32("Invul").unwrap_or(0)
        } else {
            0
        },
        bat_hanging: if entity_id == "Bat" {
            nbt.get_i8("BatFlags").unwrap_or(0) & 1 != 0
        } else {
            false
        },
        converting: if entity_id == "Zombie" {
            nbt.get_i32("ConversionTime").is_some_and(|time| time > -1)
        } else {
            false
        },
    }
}

/// Decode the appearance subset of `MobSpawnerBaseLogic.WeightedRandomMinecart.nbtData`.
///
/// `spawnNewEntity` overlays this compound on a freshly serialized entity and invokes that
/// entity's ordinary `readFromNBT`.  The entity id is intentionally supplied separately: the
/// two-argument `WeightedRandomMinecart` uses `mobID` as the entity class, so a SpawnData `id`
/// tag does not replace the spawner's species (the `id` is only relevant to a nested `Riding`
/// chain, which the miniature renderer does not draw).
fn read_spawner_spawn_data(
    entity_id: &str,
    nbt: &mc_nbt::NbtCompound,
) -> mc_world::tileent::SpawnerSpawnData {
    use mc_world::tileent::SpawnerSpawnData;

    // java: `EntityTameable.readEntityFromNBT` first tests the *presence and type* of
    // `OwnerUUID`; only when that exact string key is absent does it fall back to the legacy
    // `Owner` name.  An empty OwnerUUID therefore intentionally masks a non-empty Owner tag.
    let tame_owner = if nbt.contains_type("OwnerUUID", mc_nbt::TagType::String) {
        nbt.get_str("OwnerUUID")
    } else {
        nbt.get_str("Owner")
    };

    let mut data = SpawnerSpawnData {
        child: nbt.get_bool("IsBaby").unwrap_or(false)
            || nbt.get_i32("Age").is_some_and(|age| age < 0),
        invisible: nbt.get_bool("Invisible").unwrap_or(false),
        sheared: nbt.get_bool("Sheared").unwrap_or(false),
        powered: nbt.get_bool("powered").unwrap_or(false),
        saddled: if entity_id == "EntityHorse" {
            read_spawner_horse_saddle(nbt)
        } else {
            data_bool(nbt, "Saddle")
        },
        // `EntityTameable.readEntityFromNBT` does not read a generic `Tame` key; its
        // tame watcher changes only when an OwnerUUID/legacy Owner resolves. `Tame` is
        // an EntityHorse field and must not make a Wolf/Ocelot preview look tamed.
        tamed: tame_owner.is_some_and(|owner| !owner.is_empty()),
        angry: nbt.get_bool("Angry").unwrap_or(false),
        sitting: nbt.get_bool("Sitting").unwrap_or(false),
        horse_variant: nbt.get_i32("Variant").unwrap_or(0),
        horse_armor: read_spawner_horse_armor(nbt),
        horse_chested: nbt.get_bool("ChestedHorse").unwrap_or(false),
        // java: `spawnNewEntity` only calls `mountEntity` when `EntityList.createEntityByName`
        // succeeds for the nested compound's string id. The nested entity is not recursively
        // drawn by the spawner TESR, but the successful mount still reaches `ModelBase.isRiding`.
        riding: read_spawner_riding(nbt),
        wither_armored: read_spawner_wither_armored(entity_id, nbt),
        ..SpawnerSpawnData::default()
    };

    // These are the exact NBT fields consumed by the renderer-specific getters.  Keep the
    // entity-specific interpretation at this boundary rather than overloading one generic byte
    // on the world side (e.g. `SkeletonType == 1` and `Pig.Saddle == true` both become variant 1
    // but select entirely different render branches).
    data.variant = match entity_id {
        "Zombie" => {
            if data_bool(nbt, "IsVillager") {
                1
            } else {
                0
            }
        }
        "Skeleton" => nbt.get_i8("SkeletonType").map_or(0, i32::from),
        "Guardian" => {
            if data_bool(nbt, "Elder") {
                1
            } else {
                0
            }
        }
        // `EntityOcelot.setTameSkin`/`EntityRabbit.setRabbitType` store the integer through a
        // signed-byte watcher, just like `EntitySkeleton`'s type. Preserve that narrowing for
        // custom SpawnData values instead of exposing an impossible unbounded renderer variant.
        "Ozelot" => read_spawner_signed_byte(nbt, "CatType"),
        "Rabbit" => read_spawner_signed_byte(nbt, "RabbitType"),
        "Villager" => nbt.get_i32("Profession").unwrap_or(0),
        // `EntityHorse.readEntityFromNBT` feeds `getInteger("Type")` through
        // `setHorseType`, whose watcher is a signed byte. Keep the same low-byte
        // narrowing for malformed/extended SpawnData instead of treating Type as an
        // unbounded renderer variant.
        "EntityHorse" => read_spawner_horse_type(nbt),
        // EntitySlime stores its size zero-based and reads `Size + 1` into the watcher.
        "Slime" | "LavaSlime" => nbt.get_i32("Size").unwrap_or(0).max(0) + 1,
        "Wolf" => {
            if data.tamed {
                1
            } else if data.angry {
                2
            } else {
                0
            }
        }
        "Pig" => i32::from(data.saddled),
        _ => 0,
    };

    // Sheep stores dye in metadata order. Wolf stores dye damage, whose enum lookup reverses the
    // sixteen-entry table and falls back to black (metadata 15) for invalid input.
    data.dye_color = match entity_id {
        "Sheep" => Some(nbt.get_i8("Color").map_or(0, |color| {
            u8::try_from(color)
                .ok()
                .filter(|&color| color < 16)
                .unwrap_or(0)
        })),
        // A fresh cached wolf starts with RED (metadata 14). `spawnNewEntity` first serializes
        // that constructor state and then overlays SpawnData, so an omitted key stays red.
        "Wolf" => Some(nbt.get_i8("CollarColor").map_or(14, |damage| {
            u8::try_from(damage)
                .ok()
                .filter(|&damage| damage < 16)
                .map_or(15, |damage| 15 - damage)
        })),
        _ => None,
    };

    // EntityEnderman's old numeric carried block format is the one written by 1.8.9.  Name-form
    // carried blocks are accepted too when the registry knows them; malformed ids simply leave
    // the block layer absent rather than turning an untrusted S35 into a panic.
    data.carried_block = read_spawner_carried_block(nbt);

    if let Some(items) = nbt.get_list("Equipment") {
        data.equipment = core::array::from_fn(|index| {
            items.get(index).and_then(|item| match item {
                mc_nbt::Nbt::Compound(item) => read_spawner_equipment(item),
                _ => None,
            })
        });
    }

    data
}

/// Java's `NBTTagCompound.getBoolean`: wrong/missing tags read as false.
fn data_bool(nbt: &mc_nbt::NbtCompound, key: &str) -> bool {
    nbt.get_bool(key).unwrap_or(false)
}

/// Java's `DataWatcher.updateObject(..., Byte.valueOf((byte) value))` narrowing used by
/// Ocelot/Rabbit variant setters.
fn read_spawner_signed_byte(nbt: &mc_nbt::NbtCompound, key: &str) -> i32 {
    i32::from(nbt.get_i8(key).unwrap_or(0))
}

fn read_spawner_carried_block(nbt: &mc_nbt::NbtCompound) -> Option<mc_data::BlockState> {
    let block_id = nbt
        .get_str("carried")
        .and_then(mc_data::block_by_name)
        .map(|id| id.0)
        .or_else(|| nbt.get_i16("carried").and_then(|id| u16::try_from(id).ok()))?;
    let meta = nbt
        .get_i16("carriedData")
        .and_then(|value| u8::try_from(value).ok())
        .unwrap_or(0);
    Some(mc_data::BlockState::new(mc_data::BlockId(block_id), meta))
}

fn read_spawner_equipment(
    nbt: &mc_nbt::NbtCompound,
) -> Option<mc_world::tileent::SpawnerEquipment> {
    let item_id = nbt
        .get_str("id")
        .and_then(mc_data::item_by_name)
        .or_else(|| {
            nbt.get_i16("id")
                .and_then(|id| u16::try_from(id).ok())
                .map(mc_data::ItemId)
                .filter(|&id| mc_data::try_item_props(id).is_some())
        })?;
    let props = mc_data::try_item_props(item_id)?;
    let tag = nbt.get_compound("tag");
    let leather_color = (props.armor_material == Some(mc_data::ArmorMaterial::Leather))
        .then(|| {
            tag.and_then(|tag| tag.get_compound("display"))
                .and_then(|display| display.get_i32("color"))
                .map(|color| color as u32)
        })
        .flatten();
    // java: ItemStack.isItemEnchanted checks only for the `ench` list key (even an
    // empty list); `StoredEnchantments` belongs to ItemEnchantedBook and must not make
    // an armor/held-item stack glint by itself.
    let enchanted = tag.is_some_and(|tag| tag.contains_type("ench", mc_nbt::TagType::List));
    // java: `LayerCustomHead` reads `ItemStack.tag.SkullOwner` in compound form first and then
    // accepts the old string form.  A name-only profile is deliberately kept name-only; the
    // client-side 1.8.9 path cannot ask a server-side `PlayerProfileCache` to promote it.
    let skull_owner = tag.and_then(|tag| {
        if let Some(owner) = tag.get_compound("SkullOwner") {
            mc_auth::read_game_profile_from_nbt(owner)
        } else {
            tag.get_str("SkullOwner")
                .filter(|name| !name.is_empty())
                .map(|name| mc_auth::GameProfile {
                    id: None,
                    name: name.to_owned(),
                    properties: Vec::new(),
                })
        }
    });
    let banner_state = (item_id.0 == mc_render::entity::tilepass::BANNER_ITEM_ID)
        .then(|| read_spawner_banner_state(tag, nbt.get_i16("Damage").unwrap_or(0)));
    Some(mc_world::tileent::SpawnerEquipment {
        item_id: item_id.0,
        damage: nbt.get_i16("Damage").unwrap_or(0).max(0),
        leather_color,
        enchanted,
        skull_owner,
        banner_state,
    })
}

/// java: `EntityHorse.readEntityFromNBT`'s `SaddleItem` branch. The boolean `Saddle` fallback is
/// used only when the exact compound key is absent; a malformed present compound does not revive
/// the legacy flag because vanilla's `else if` is keyed on presence, not successful decoding.
fn read_spawner_horse_saddle(nbt: &mc_nbt::NbtCompound) -> bool {
    if nbt.contains_type("SaddleItem", mc_nbt::TagType::Compound) {
        return nbt
            .get_compound("SaddleItem")
            .and_then(read_spawner_equipment)
            .is_some_and(|item| item.item_id == 329);
    }
    data_bool(nbt, "Saddle")
}

/// Java's `EntityHorse.setHorseType` stores the integer in a `Byte` watcher.
fn read_spawner_horse_type(nbt: &mc_nbt::NbtCompound) -> i32 {
    i32::from(nbt.get_i8("Type").unwrap_or(0))
}

/// java: `EntityHorse.getHorseArmorIndex` accepts only the three horse-armor item singletons.
fn read_spawner_horse_armor(nbt: &mc_nbt::NbtCompound) -> i32 {
    // java: `EntityHorse.updateHorseSlots` only synchronizes slot 1 when `canWearArmor()` is
    // true, and that predicate is `getHorseType() == 0`; ArmorItem on donkeys, mules, and undead
    // horses remains in the inventory slot but never reaches `getHorseArmorIndexSynced()`.
    if read_spawner_horse_type(nbt) != 0 {
        return 0;
    }
    let Some(item) = nbt
        .get_compound("ArmorItem")
        .and_then(read_spawner_equipment)
    else {
        return 0;
    };
    match item.item_id {
        417 => 1,
        418 => 2,
        419 => 3,
        _ => 0,
    }
}

fn read_spawner_riding(nbt: &mc_nbt::NbtCompound) -> bool {
    nbt.get_compound("Riding")
        .and_then(|mount| mount.get_str("id"))
        .and_then(mc_data::entity_by_name)
        .is_some()
}

/// Resolve the health-dependent wither aura from the overlay that reaches the cached entity.
///
/// java: `EntityLivingBase.readEntityFromNBT` prefers numeric `HealF`, otherwise accepts a float
/// or short `Health`; `EntityWither.isArmored` compares that value with half of its constructor
/// max health (300). SpawnData may carry custom attributes too, but this typed tile snapshot does
/// not become a gameplay attribute store, so the vanilla default bound is the honest one here.
fn read_spawner_wither_armored(entity_id: &str, nbt: &mc_nbt::NbtCompound) -> bool {
    if entity_id != "WitherBoss" {
        return false;
    }
    let health = nbt.get_f32("HealF").or_else(|| match nbt.get("Health") {
        // java: EntityLivingBase.readEntityFromNBT only accepts TAG_Float or TAG_Short
        // for the legacy Health fallback; other numeric tag kinds leave constructor health
        // untouched even though NBTTagCompound.getFloat would coerce them for HealF.
        Some(mc_nbt::Nbt::Float(value)) => Some(*value),
        Some(mc_nbt::Nbt::Short(value)) => Some(f32::from(*value)),
        _ => None,
    });
    health.is_some_and(|health| health <= 150.0)
}

/// `TileEntityBanner.setItemValues` starts from the banner item's dye metadata, then lets a
/// numeric `BlockEntityTag.Base` override it and appends the known pattern ids. The block-tile
/// reader below has a zero base default, so restore the item metadata when that override is
/// absent, exactly as the item renderer's Java helper does.
fn read_spawner_banner_state(
    tag: Option<&mc_nbt::NbtCompound>,
    damage: i16,
) -> mc_world::BannerState {
    let metadata = i32::from(damage.max(0)) & 15;
    let Some(block_entity) = tag.and_then(|tag| tag.get_compound("BlockEntityTag")) else {
        return mc_world::BannerState {
            base_color: metadata,
            patterns: Vec::new(),
            initialized: true,
        };
    };
    let mut state = read_banner(block_entity);
    if !block_entity.contains_numeric_type("Base") {
        state.base_color = metadata;
    }
    state
}

/// Build the animation arguments for the cached entity used by
/// `TileEntityMobSpawnerRenderer.renderMob`.
///
/// java: `RendererLivingEntity.doRender` derives the model inputs from the cached entity before it
/// runs the body and every `LayerRenderer`.  `TileEntityMobSpawnerRenderer.renderMob` constructs
/// that entity once with `spawnNewEntity(entity, false)` and never ticks it, so every cached
/// living field is constructor-zero.  Keeping this adapter on top of `LivingRenderState::params`
/// preserves the exact frame partial (`age = 0 + partialTicks`) and the normal body/layer call
/// site without fabricating live age, limb swing or head motion.
#[must_use]
fn spawner_anim_params(
    animation: SpawnerRenderAnimation,
    partial_ticks: f32,
) -> mc_render::entity::AnimParams {
    let state = mc_render::entity::LivingRenderState {
        entity: mc_render::entity::EntityAnimState {
            entity_id: animation.entity_id,
            ..mc_render::entity::EntityAnimState::default()
        },
        ..mc_render::entity::LivingRenderState::default()
    };
    state.params(partial_ticks)
}

/// Apply the entity fields that `spawnNewEntity` would have populated from `SpawnData` before
/// `RendererLivingEntity` asks for its model arguments.
fn apply_spawner_spawn_data(
    params: &mut mc_render::entity::AnimParams,
    entity_type: mc_data::EntityType,
    spawn_data: Option<&mc_world::tileent::SpawnerSpawnData>,
) {
    let Some(data) = spawn_data else {
        return;
    };
    params.entity.variant = data.variant;
    params.entity.child = data.child;
    params.entity.invisible = data.invisible;
    params.entity.sheared = data.sheared;
    params.entity.sitting = data.sitting;
    params.entity.riding = data.riding;
    // java: EntityWolf.isAngry() is read directly by ModelWolf's pose path.  The texture
    // variant (2) is not a substitute: a malformed/partial snapshot can still retain the
    // constructor skin while the arm/head pose must follow the explicit Angry watcher value.
    params.entity.aggressive = entity_type == mc_data::EntityType::Wolf && data.angry;
    // java: ModelSkeleton.setLivingAnimations unconditionally assigns aimedBow from
    // getSkeletonType() == 1.  This matters for the copied armour model even though the
    // skeleton body rewrites its own arms later in setRotationAngles.
    params.entity.aimed_bow = entity_type == mc_data::EntityType::Skeleton && data.variant == 1;
    params.entity.carrying_block = data.carried_block.is_some();
    // `heldItemRight` is set by the living renderer's equipment path; ordinary items use 1 and
    // full block items use 3. The exact block-model branch is resolved by the layer adapter.
    if data.equipment[0].is_some() {
        params.entity.held_item_right = 1;
        // java: `RenderWitch.doRender` writes `ModelWitch.field_82900_g` from
        // `entity.getHeldItem() != null` before `RendererLivingEntity` poses the body.  The
        // generic held-item latch is also retained for biped models; only the witch model reads
        // this separate per-entity flag.
        params.entity.witch_holding_item = true;
    }
}

/// Apply the renderer-facing fields that cannot be stored in the NBT-free world snapshot.
fn apply_spawner_render_appearance(
    params: &mut mc_render::entity::AnimParams,
    appearance: &SpawnerRenderAppearance,
) {
    params.entity.bat_hanging = appearance.bat_hanging;
    params.entity.converting = appearance.converting;
}

/// Whether the cached wither uses `wither_invulnerable.png` for this frame.
///
/// java: `RenderWither.getEntityTexture` (`:26-28`) — the 220-tick spawn shield only forces
/// the invulnerable sheet after tick 80; before that it intentionally flickers on five-tick
/// windows where `(invulTime / 5) % 2 == 1` selects the ordinary sheet.
#[must_use]
fn spawner_wither_uses_invulnerable_texture(invul_time: i32) -> bool {
    invul_time > 0 && (invul_time > 80 || invul_time / 5 % 2 != 1)
}

/// Java's name-keyed render easter eggs compare the formatting-stripped custom name. Keep the
/// exact section-code rule local to this packet seam so malformed/dangling formatting remains a
/// literal name instead of being silently normalised.
fn spawner_unformatted_name(appearance: &SpawnerRenderAppearance) -> Option<String> {
    let name = appearance.custom_name.as_deref()?;
    let mut out = String::with_capacity(name.len());
    let mut chars = name.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{00a7}' {
            if let Some(&code) = chars.peek() {
                if matches!(
                    code.to_ascii_lowercase(),
                    '0'..='9' | 'a'..='f' | 'k'..='o' | 'r'
                ) {
                    chars.next();
                    continue;
                }
            }
        }
        out.push(ch);
    }
    Some(out)
}

/// The cached spawner entity has no player/profile cape gate; living mob names still select the
/// Dinnerbone/Grumm transform exactly as the normal living renderer does.
fn spawner_upside_down(
    entity_type: mc_data::EntityType,
    appearance: &SpawnerRenderAppearance,
) -> bool {
    let Some(name) = spawner_unformatted_name(appearance) else {
        return false;
    };
    if !entity_type.info().living {
        return false;
    }
    matches!(name.as_str(), "Dinnerbone" | "Grumm")
}

/// The texture source for one spawner-held/custom-head item submission. Baked items use the
/// stitched atlas, fixed/builtin items use a resource-pack path, banners are generated from
/// their layered pixels, and player skulls may use the shared asynchronous skin cache.
enum SpawnerItemTexture {
    Atlas,
    Resource,
    Banner(mc_world::BannerState),
    Skin(BuiltinSkullOwner),
}

/// Resolve the same two mutually-exclusive branches as `LayerCustomHead`: any `ItemBlock` is a
/// block head, while only `Items.skull` enters the skull renderer. Other equipment is not drawn
/// by vanilla's layer even when it happens to occupy the helmet slot.
fn spawner_custom_head_item_kind(
    item: &mc_world::tileent::SpawnerEquipment,
) -> Option<mc_render::entity::layers::CustomHeadItem> {
    let props = mc_data::try_item_props(mc_data::ItemId(item.item_id))?;
    if props.block.is_some() {
        Some(mc_render::entity::layers::CustomHeadItem::Block)
    } else if props.name == "minecraft:skull" {
        Some(mc_render::entity::layers::CustomHeadItem::Skull)
    } else {
        None
    }
}

/// Decode a spawner equipment stack's `builtin/entity` branch. Unlike an ordinary baked model,
/// chest/banner/skull item ids intentionally have no quads in the item atlas and must go through
/// `TileEntityItemStackRenderer`'s typed tile-model equivalent.
fn spawner_builtin_item(
    item: &mc_world::tileent::SpawnerEquipment,
    model: &mc_assets::BakedModel,
) -> Option<BuiltinTileItem> {
    match builtin_tile_item_kind_for_model(item.item_id, model)? {
        BuiltinTileItemKind::Chest => Some(BuiltinTileItem::Chest { trapped: false }),
        BuiltinTileItemKind::TrappedChest => Some(BuiltinTileItem::Chest { trapped: true }),
        BuiltinTileItemKind::EnderChest => Some(BuiltinTileItem::EnderChest),
        BuiltinTileItemKind::Banner => Some(BuiltinTileItem::Banner {
            state: item
                .banner_state
                .clone()
                .unwrap_or_else(|| mc_world::BannerState {
                    base_color: i32::from(item.damage.max(0)) & 15,
                    patterns: Vec::new(),
                    initialized: true,
                }),
        }),
        BuiltinTileItemKind::Skull => {
            let skull_type = u8::try_from(item.damage.max(0)).unwrap_or(u8::MAX);
            if skull_type == 3 {
                if let Some(profile) = item.skull_owner.as_ref() {
                    return Some(BuiltinTileItem::PlayerSkull {
                        skull_type,
                        owner: player_skull_owner_from_profile(profile),
                    });
                }
            }
            Some(BuiltinTileItem::Skull { skull_type })
        }
    }
}

/// Select how a builtin item's returned texture key should be made resident in the entity phase.
fn spawner_builtin_texture_source(item: &BuiltinTileItem, texture: &str) -> SpawnerItemTexture {
    match item {
        BuiltinTileItem::Banner { state } => SpawnerItemTexture::Banner(state.clone()),
        BuiltinTileItem::PlayerSkull { owner, .. } if texture == owner.cache_key => {
            SpawnerItemTexture::Skin(owner.clone())
        }
        _ => SpawnerItemTexture::Resource,
    }
}

/// Push an item/builtin attachment into the entity phase, preserving its atlas/resource/generated
/// texture route. The phase intentionally falls back to its resident 1x1 texture on a missing
/// resource; Java's `TextureManager` has the same non-fatal draw-time behavior.
#[allow(clippy::too_many_arguments)]
fn push_spawner_item_geometry(
    phase: &mut mc_render::entity::EntityPhase,
    gpu: &Gpu,
    resources: &mc_assets::ResourceManager,
    atlas: &mc_assets::Atlas,
    texture: &str,
    source: SpawnerItemTexture,
    blend: mc_render::entity::LayerBlend,
    vertices: &[EntityVertex],
) {
    if vertices.is_empty() {
        return;
    }
    let resident = match source {
        SpawnerItemTexture::Atlas => {
            if !phase.has_texture(BLOCK_ATLAS_TEXTURE) {
                phase.upload_texture(gpu, BLOCK_ATLAS_TEXTURE, &atlas.texture, atlas.w, atlas.h);
            }
            true
        }
        SpawnerItemTexture::Resource => phase.load_texture(gpu, resources, texture),
        SpawnerItemTexture::Banner(state) => {
            if phase.has_texture(texture) {
                true
            } else {
                mc_render::entity::tilepass::banner_texture_pixels(resources, &state)
                    .filter(|(key, _, _, _)| key == texture)
                    .map(|(_, pixels, width, height)| {
                        phase.upload_texture(gpu, texture, &pixels, width, height);
                        true
                    })
                    .unwrap_or(false)
            }
        }
        SpawnerItemTexture::Skin(owner) => {
            if phase.has_texture(texture) {
                true
            } else if let Some(skin) = mc_render::entity::skin::shared_skin(&owner.cache_key) {
                phase.upload_texture(
                    gpu,
                    texture,
                    &skin.rgba,
                    mc_render::entity::SkinTexture::WIDTH,
                    mc_render::entity::SkinTexture::HEIGHT,
                );
                true
            } else {
                false
            }
        }
    };
    let key = if resident {
        texture
    } else {
        mc_render::entity::EntityPhase::FALLBACK_TEXTURE
    };
    phase.push_blended(key, blend, vertices);
}

/// Item quads use `Item.getColorFromItemStack` tint indices. The wire-free spawner snapshot keeps
/// the appearance inputs that matter here (damage, leather colour, and banner state); this covers
/// the 1.8.9 potion/colored-block/spawn-egg branches without retaining the whole NBT compound.
fn spawner_item_quad_tint(item: &mc_world::tileent::SpawnerEquipment, tint_index: u8) -> u32 {
    let props = mc_data::item_props(mc_data::ItemId(item.item_id));
    if tint_index == 0 && props.name == "minecraft:potion" {
        return mc_data::potion_liquid_color_from_damage(i32::from(item.damage));
    }
    if tint_index == 0 && props.armor_material == Some(mc_data::ArmorMaterial::Leather) {
        return item.leather_color.unwrap_or(0xA0_65_40) & 0x00FF_FFFF;
    }
    if props.name == "minecraft:spawn_egg" {
        if let Some(entity) = u8::try_from(item.damage)
            .ok()
            .and_then(mc_data::entity_type)
        {
            let info = entity.info();
            if tint_index == 0 {
                return info.egg_primary.unwrap_or(0xFF_FF_FF);
            }
            return info.egg_secondary.unwrap_or(0xFF_FF_FF);
        }
    }
    // java: ItemColored/ItemLeaves/ItemLilyPad ask their block's position-independent render
    // colour. Tallgrass normalises invalid metadata to DEAD_BUSH before that lookup.
    if let Some(block) = props.block {
        let metadata = if props.java_class == "net.minecraft.item.ItemColored" && block.0 == 31 {
            match i32::from(item.damage) {
                0..=2 => item.damage as u8,
                _ => 0,
            }
        } else {
            item.damage as u8
        };
        if matches!(
            props.java_class,
            "net.minecraft.item.ItemColored"
                | "net.minecraft.item.ItemLeaves"
                | "net.minecraft.item.ItemLilyPad"
        ) {
            return mc_render::terrain::tint::render_color(mc_data::BlockState::new(
                block, metadata,
            ));
        }
    }
    0xFF_FF_FF
}

/// `LayerHeldItem`/`LayerHeldItemWitch` choose translucent blending only for a GUI-3D block whose
/// block render layer is translucent. Everything else is opaque, including flat potion sprites.
fn spawner_item_layer_blend(
    item: &mc_world::tileent::SpawnerEquipment,
    model: &mc_assets::BakedModel,
) -> mc_render::entity::LayerBlend {
    let translucent = model.gui_3d
        && mc_data::item_props(mc_data::ItemId(item.item_id))
            .block
            .is_some_and(|block| {
                mc_data::block_props(block.default_state()).render_layer
                    == mc_data::RenderLayer::Translucent
            });
    if translucent {
        mc_render::entity::LayerBlend::Translucent
    } else {
        mc_render::entity::LayerBlend::Opaque
    }
}

/// Replay the equipment-bearing layer branches for one spawner entity. Returning `true` means
/// the caller must not try the ordinary fixed-texture layer path, even when Java emitted no
/// geometry (for example a malformed banner or a non-biped renderer).
#[allow(clippy::too_many_arguments)]
fn emit_spawner_equipment_layer(
    models: &mut mc_render::entity::RenderManager,
    tile_models: &mut TileModels,
    phase: &mut mc_render::entity::EntityPhase,
    gpu: &Gpu,
    resources: &mc_assets::ResourceManager,
    atlas: &mc_assets::Atlas,
    entity_type: mc_data::EntityType,
    layer: mc_render::entity::LayerKind,
    body_kind: mc_render::entity::ModelKind,
    base: mc_render::entity::Mat4,
    params: &mc_render::entity::AnimParams,
    data: &mc_world::tileent::SpawnerSpawnData,
    scratch: &mut Vec<EntityVertex>,
) -> bool {
    use mc_render::entity::layers::{CustomHeadItem, LayerBlend, LayerKind};

    let slot = match layer {
        LayerKind::HeldItem | LayerKind::WitchHeldItem => 0,
        LayerKind::CustomHead => 4,
        _ => return false,
    };
    let Some(item) = data.equipment[slot].as_ref() else {
        return true;
    };

    match layer {
        LayerKind::HeldItem => {
            let item_id = mc_data::ItemId(item.item_id);
            let full_block_pose = mc_data::item_props(item_id).block.is_some_and(|block| {
                mc_data::block_props(block.default_state()).render_type
                    == mc_data::RenderType::TileEntity
            });
            let Some(hand) = models.held_item_matrix_for(
                body_kind,
                params.scale,
                base,
                params.entity.child,
                full_block_pose,
                params.entity.sneaking,
            ) else {
                return true;
            };
            let model = mc_assets::item_model_override_for(item_id, item.damage).model;
            scratch.clear();
            if let Some(builtin) = spawner_builtin_item(item, model) {
                let parent = hand
                    * mc_render::entity::objects::item_renderer_display_transform(
                        model.gui_3d,
                        &model.transforms.third_person,
                    );
                let Some(texture) = tile_models.emit_builtin_item(&builtin, parent, scratch) else {
                    return true;
                };
                let source = spawner_builtin_texture_source(&builtin, &texture);
                push_spawner_item_geometry(
                    phase,
                    gpu,
                    resources,
                    atlas,
                    &texture,
                    source,
                    LayerBlend::Opaque,
                    scratch,
                );
            } else {
                let matrix = hand
                    * mc_render::entity::objects::item_renderer_model_transform(
                        model.gui_3d,
                        &model.transforms.third_person,
                    );
                mc_render::entity::item::emit_baked_model_tinted(model, matrix, scratch, |index| {
                    spawner_item_quad_tint(item, index)
                });
                let blend = spawner_item_layer_blend(item, model);
                push_spawner_item_geometry(
                    phase,
                    gpu,
                    resources,
                    atlas,
                    BLOCK_ATLAS_TEXTURE,
                    SpawnerItemTexture::Atlas,
                    blend,
                    scratch,
                );
                if item.enchanted {
                    scratch.clear();
                    mc_render::entity::item::emit_baked_model_glint(
                        model,
                        matrix,
                        mc_render::entity::item::item_glint_system_time_ms(),
                        scratch,
                    );
                    phase.push_blended(BLOCK_ATLAS_TEXTURE, LayerBlend::Glint, scratch);
                }
            }
        }
        LayerKind::WitchHeldItem => {
            if entity_type != mc_data::EntityType::Witch {
                return true;
            }
            let item_id = mc_data::ItemId(item.item_id);
            let Some(witch) = models.witch_held_item_matrix_for(
                body_kind,
                params.scale,
                base,
                params.entity.child,
                super::appearance::witch_pose(item_id),
            ) else {
                return true;
            };
            let model = mc_assets::item_model_override_for(item_id, item.damage).model;
            scratch.clear();
            if let Some(builtin) = spawner_builtin_item(item, model) {
                let parent = witch
                    * mc_render::entity::objects::item_renderer_display_transform(
                        model.gui_3d,
                        &model.transforms.third_person,
                    );
                let Some(texture) = tile_models.emit_builtin_item(&builtin, parent, scratch) else {
                    return true;
                };
                let source = spawner_builtin_texture_source(&builtin, &texture);
                push_spawner_item_geometry(
                    phase,
                    gpu,
                    resources,
                    atlas,
                    &texture,
                    source,
                    LayerBlend::Opaque,
                    scratch,
                );
            } else {
                let matrix = witch
                    * mc_render::entity::objects::item_renderer_model_transform(
                        model.gui_3d,
                        &model.transforms.third_person,
                    );
                mc_render::entity::item::emit_baked_model_tinted(model, matrix, scratch, |index| {
                    spawner_item_quad_tint(item, index)
                });
                let blend = spawner_item_layer_blend(item, model);
                push_spawner_item_geometry(
                    phase,
                    gpu,
                    resources,
                    atlas,
                    BLOCK_ATLAS_TEXTURE,
                    SpawnerItemTexture::Atlas,
                    blend,
                    scratch,
                );
                if item.enchanted {
                    scratch.clear();
                    mc_render::entity::item::emit_baked_model_glint(
                        model,
                        matrix,
                        mc_render::entity::item::item_glint_system_time_ms(),
                        scratch,
                    );
                    phase.push_blended(BLOCK_ATLAS_TEXTURE, LayerBlend::Glint, scratch);
                }
            }
        }
        LayerKind::CustomHead => {
            let Some(item_kind) = spawner_custom_head_item_kind(item) else {
                return true;
            };
            let custom_head = {
                let model = models.model_mut(body_kind).model();
                let Some(head_part) = model.find("head") else {
                    return true;
                };
                let villager_like = entity_type == mc_data::EntityType::Villager
                    || (entity_type == mc_data::EntityType::Zombie && params.entity.variant == 1);
                let custom_head = mc_render::entity::layers::custom_head_matrix(
                    base,
                    model,
                    head_part,
                    params.scale,
                    params.entity.sneaking,
                    params.entity.child,
                    villager_like,
                    item_kind,
                );
                // `RenderItem` applies the selected HEAD display transform after the head part;
                // builtin/entity's own half-scale remains inside `TileModels::emit_builtin_item`.
                custom_head
            };
            let item_id = mc_data::ItemId(item.item_id);
            let model = mc_assets::item_model_override_for(item_id, item.damage).model;
            scratch.clear();
            match item_kind {
                CustomHeadItem::Skull => {
                    let skull_model = mc_render::entity::tile::SkullVariant::from_id(
                        u8::try_from(item.damage.max(0)).unwrap_or(u8::MAX),
                    )
                    .model();
                    // java: `renderSkull(-0.5F, 0.0F, -0.5F, EnumFacing.UP, 180.0F, ...)` in
                    // LayerCustomHead: the `scale(-1, -1, 1)` flip and a fixed 180-degree yaw
                    // with no pitch, which the shared helper owns.
                    models.emit_custom_head_skull(skull_model, custom_head, scratch);
                    let (texture, source) = if let Some(profile) = item.skull_owner.as_ref() {
                        let owner = player_skull_owner_from_profile(profile);
                        let texture = owner.texture_key();
                        let source = if texture == owner.cache_key {
                            SpawnerItemTexture::Skin(owner)
                        } else {
                            SpawnerItemTexture::Resource
                        };
                        (texture, source)
                    } else {
                        (
                            super::appearance::skull_texture(item.damage).to_owned(),
                            SpawnerItemTexture::Resource,
                        )
                    };
                    push_spawner_item_geometry(
                        phase,
                        gpu,
                        resources,
                        atlas,
                        &texture,
                        source,
                        LayerBlend::Opaque,
                        scratch,
                    );
                }
                CustomHeadItem::Block => {
                    if let Some(builtin) = spawner_builtin_item(item, model) {
                        let parent = custom_head
                            * mc_render::entity::objects::item_renderer_display_transform(
                                model.gui_3d,
                                &model.transforms.head,
                            );
                        let Some(texture) =
                            tile_models.emit_builtin_item(&builtin, parent, scratch)
                        else {
                            return true;
                        };
                        let source = spawner_builtin_texture_source(&builtin, &texture);
                        push_spawner_item_geometry(
                            phase,
                            gpu,
                            resources,
                            atlas,
                            &texture,
                            source,
                            LayerBlend::Opaque,
                            scratch,
                        );
                    } else {
                        let matrix = custom_head
                            * mc_render::entity::objects::item_renderer_model_transform(
                                model.gui_3d,
                                &model.transforms.head,
                            );
                        mc_render::entity::item::emit_baked_model_tinted(
                            model,
                            matrix,
                            scratch,
                            |index| spawner_item_quad_tint(item, index),
                        );
                        push_spawner_item_geometry(
                            phase,
                            gpu,
                            resources,
                            atlas,
                            BLOCK_ATLAS_TEXTURE,
                            SpawnerItemTexture::Atlas,
                            LayerBlend::Opaque,
                            scratch,
                        );
                        if item.enchanted {
                            scratch.clear();
                            mc_render::entity::item::emit_baked_model_glint(
                                model,
                                matrix,
                                mc_render::entity::item::item_glint_system_time_ms(),
                                scratch,
                            );
                            phase.push_blended(BLOCK_ATLAS_TEXTURE, LayerBlend::Glint, scratch);
                        }
                    }
                }
            }
        }
        _ => unreachable!("equipment layer was filtered above"),
    }
    true
}

/// Whether a descriptor layer can be replayed for the spawner's cached default entity.
///
/// The cached entity has no equipment/profile payload in the current tile state, so the layers
/// whose Java implementations dereference those values stay suppressed.  Fixed texture/model
/// layers still run and preserve the descriptor order; the species gates are the same ones used
/// by the live entity adapter for constructor-default state.
#[must_use]
#[cfg(test)]
fn spawner_layer_should_render(
    entity_type: mc_data::EntityType,
    layer: mc_render::entity::LayerKind,
    params: &mc_render::entity::AnimParams,
) -> bool {
    spawner_layer_should_render_with_spawn_data(entity_type, layer, params, None)
}

/// The same layer gate with the optional `SpawnData` overlay available.
fn spawner_layer_should_render_with_spawn_data(
    entity_type: mc_data::EntityType,
    layer: mc_render::entity::LayerKind,
    params: &mc_render::entity::AnimParams,
    spawn_data: Option<&mc_world::tileent::SpawnerSpawnData>,
) -> bool {
    use mc_render::entity::LayerKind;
    match layer {
        LayerKind::CreeperCharge => {
            spawn_data.map_or(params.entity.variant != 0, |data| data.powered)
        }
        LayerKind::EndermanEyes | LayerKind::SpiderEyes | LayerKind::DragonEyes => true,
        // java: `LayerWitherAura` delegates to `EntityWither.isArmored()`; SpawnData's numeric
        // Health/HealF overlay is reduced to `wither_armored` at the NBT boundary. A missing tag
        // keeps the constructor's full-health default, so the aura stays off in that case.
        LayerKind::WitherAura => {
            entity_type == mc_data::EntityType::WitherBoss
                && spawn_data.is_some_and(|data| data.wither_armored)
        }
        LayerKind::SheepWool => {
            entity_type == mc_data::EntityType::Sheep
                && !params.entity.sheared
                && !params.entity.invisible
        }
        LayerKind::SlimeGel => {
            entity_type == mc_data::EntityType::Slime && !params.entity.invisible
        }
        LayerKind::Saddle => {
            entity_type == mc_data::EntityType::Pig
                && spawn_data.map_or(params.entity.variant != 0, |data| data.saddled)
        }
        LayerKind::WolfCollar => {
            entity_type == mc_data::EntityType::Wolf
                && params.entity.variant == 1
                && !params.entity.invisible
        }
        LayerKind::Armor => spawn_data.is_some_and(|data| {
            data.equipment[1..].iter().flatten().any(|item| {
                mc_data::item_props(mc_data::ItemId(item.item_id))
                    .armor_material
                    .is_some()
            })
        }),
        LayerKind::HeldItem => spawn_data.is_some_and(|data| data.equipment[0].is_some()),
        LayerKind::WitchHeldItem => {
            entity_type == mc_data::EntityType::Witch
                && spawn_data.is_some_and(|data| data.equipment[0].is_some())
        }
        LayerKind::CustomHead => spawn_data.is_some_and(|data| {
            data.equipment[4]
                .as_ref()
                .and_then(spawner_custom_head_item_kind)
                .is_some()
        }),
        LayerKind::MooshroomMushroom => {
            entity_type == mc_data::EntityType::MushroomCow
                && !params.entity.child
                && !params.entity.invisible
        }
        LayerKind::SnowmanHead => {
            entity_type == mc_data::EntityType::SnowMan && !params.entity.invisible
        }
        LayerKind::HeldBlock => {
            entity_type == mc_data::EntityType::Enderman
                && spawn_data.is_some_and(|data| data.carried_block.is_some())
        }
        // The block-backed mooshroom/snowman layers are emitted by `emit_spawner_dynamic_layer`.
        // Profile/cape/arrows and the iron-golem flower remain suppressed: a spawner EntityList
        // entry is never a Player, no profile/history is serialized for those player-only layers,
        // and `LayerIronGolemFlower` reads a client-only 400-tick status timer rather than NBT.
        LayerKind::Deadmau5Head
        | LayerKind::Cape
        | LayerKind::Arrow
        | LayerKind::IronGolemFlower
        | LayerKind::DragonDeath => false,
        _ => false,
    }
}

/// Supply the dye tint used by the wool/collar layers from the SpawnData metadata.
fn spawner_layer_tint(
    entity_type: mc_data::EntityType,
    spawn_data: Option<&mc_world::tileent::SpawnerSpawnData>,
    params: &mc_render::entity::AnimParams,
    appearance: &SpawnerRenderAppearance,
    partial_ticks: f32,
) -> mc_render::entity::layers::LayerTintInput {
    let dye = match entity_type {
        mc_data::EntityType::Sheep => crate::app::appearance::jeb_sheep_dye(
            appearance.custom_name.as_deref(),
            &params.entity,
            partial_ticks,
        )
        .or_else(|| {
            let meta = spawn_data.and_then(|data| data.dye_color).unwrap_or(0);
            let rgb = mc_world::dye_rgb(meta);
            Some([rgb[0], rgb[1], rgb[2], 1.0])
        }),
        mc_data::EntityType::Wolf => {
            let meta = spawn_data.and_then(|data| data.dye_color).unwrap_or(0);
            let rgb = mc_world::dye_rgb(meta);
            Some([rgb[0], rgb[1], rgb[2], 1.0])
        }
        _ => None,
    };
    mc_render::entity::layers::LayerTintInput { dye }
}

/// Emit the SpawnData-reachable block-backed layer branches that do not have a fixed entity
/// texture/model pair.
///
/// java: `LayerMooshroomMushroom` submits three `red_mushroom` block models and
/// `LayerSnowmanHead` submits a pumpkin item model at the posed head. Returning `true` means the
/// caller must not fall through to `LayerKind::texture`/`emit_layer`; both layers intentionally
/// report no fixed texture because they bind the block atlas in vanilla.
// Java's layer dispatcher passes these independent render resources separately; preserving that
// call shape keeps the block-backed branches ordered exactly like the vanilla TESR path.
#[allow(clippy::too_many_arguments)]
fn emit_spawner_dynamic_layer(
    models: &mut mc_render::entity::RenderManager,
    phase: &mut mc_render::entity::EntityPhase,
    gpu: &Gpu,
    atlas: &mc_assets::Atlas,
    entity_type: mc_data::EntityType,
    layer: mc_render::entity::LayerKind,
    body_kind: mc_render::entity::ModelKind,
    base: mc_render::entity::Mat4,
    params: &mc_render::entity::AnimParams,
    scratch: &mut Vec<EntityVertex>,
) -> bool {
    use mc_render::entity::LayerKind;

    let mut ensure_atlas = || {
        if !phase.has_texture(BLOCK_ATLAS_TEXTURE) {
            phase.upload_texture(gpu, BLOCK_ATLAS_TEXTURE, &atlas.texture, atlas.w, atlas.h);
        }
    };

    match layer {
        LayerKind::SnowmanHead => {
            if entity_type != mc_data::EntityType::SnowMan || params.entity.invisible {
                return true;
            }
            let Some(head) = models.part_matrix_for(body_kind, "head", params.scale, base) else {
                return true;
            };
            let Some(model) = super::appearance::block_item_model(86) else {
                return true;
            };
            scratch.clear();
            mc_render::entity::item::emit_baked_model(
                model,
                super::appearance::snowman_head_matrix(head),
                scratch,
            );
            if !scratch.is_empty() {
                ensure_atlas();
                phase.push_blended(
                    BLOCK_ATLAS_TEXTURE,
                    layer.blend_for(params.entity.invisible),
                    scratch,
                );
            }
            true
        }
        LayerKind::MooshroomMushroom => {
            if entity_type != mc_data::EntityType::MushroomCow
                || params.entity.child
                || params.entity.invisible
            {
                return true;
            }
            let block_state = mc_data::BlockState::new(mc_data::BlockId(40), 0);
            let Some(model) = super::appearance::block_item_model(block_state) else {
                return true;
            };
            let head = models.part_matrix_for(body_kind, "head", params.scale, base);
            scratch.clear();
            for matrix in super::appearance::mooshroom_mushroom_matrices(base, head) {
                mc_render::entity::item::emit_block_model_brightness(
                    model,
                    matrix,
                    1.0,
                    mc_render::terrain::tint::render_color(block_state),
                    scratch,
                );
            }
            if !scratch.is_empty() {
                ensure_atlas();
                phase.push_blended(
                    BLOCK_ATLAS_TEXTURE,
                    layer.blend_for(params.entity.invisible),
                    scratch,
                );
            }
            true
        }
        _ => false,
    }
}

/// Emit the four `LayerArmorBase` passes for equipment carried by the spawner's cached entity.
// Java's `LayerArmorBase.doRenderLayer` inputs are intentionally kept explicit here so the base,
// overlay, and glint submissions remain in their vanilla order.
#[allow(clippy::too_many_arguments)]
fn emit_spawner_armor_layers(
    models: &mut mc_render::entity::RenderManager,
    phase: &mut mc_render::entity::EntityPhase,
    gpu: &Gpu,
    resources: &mc_assets::ResourceManager,
    entity_type: mc_data::EntityType,
    body_kind: mc_render::entity::ModelKind,
    base: mc_render::entity::Mat4,
    params: &mc_render::entity::AnimParams,
    spawn_data: Option<&mc_world::tileent::SpawnerSpawnData>,
    scratch: &mut Vec<EntityVertex>,
) {
    let Some(data) = spawn_data else { return };
    for slot in mc_render::entity::layers::ArmorSlot::DRAW_ORDER {
        let index = match slot {
            mc_render::entity::layers::ArmorSlot::Boots => 1,
            mc_render::entity::layers::ArmorSlot::Leggings => 2,
            mc_render::entity::layers::ArmorSlot::Chestplate => 3,
            mc_render::entity::layers::ArmorSlot::Helmet => 4,
        };
        let Some(item) = data.equipment[index].as_ref() else {
            continue;
        };
        let props = mc_data::item_props(mc_data::ItemId(item.item_id));
        let Some(material) = props.armor_material else {
            continue;
        };
        scratch.clear();
        models.emit_armor_slot_for(body_kind, slot, params, base, scratch);
        if scratch.is_empty() {
            continue;
        }
        if let Some(color) = item.leather_color {
            let rgba = [
                ((color >> 16) & 255) as f32 / 255.0,
                ((color >> 8) & 255) as f32 / 255.0,
                (color & 255) as f32 / 255.0,
                1.0,
            ];
            mc_render::entity::tint(scratch, rgba);
        }
        let texture = mc_render::entity::layers::armor_texture(
            mc_render::entity::layers::armor_material_name(material),
            slot,
            None,
        );
        let key = if phase.load_texture(gpu, resources, &texture) {
            texture.as_str()
        } else {
            mc_render::entity::EntityPhase::FALLBACK_TEXTURE
        };
        phase.push_blended(key, mc_render::entity::LayerBlend::Opaque, scratch);
        // Leather armour has a second, untinted overlay texture. Java always submits this pass
        // after the dyed base whenever the item has the leather material.
        if material == mc_data::ArmorMaterial::Leather {
            scratch.clear();
            models.emit_armor_slot_for(body_kind, slot, params, base, scratch);
            let overlay = mc_render::entity::layers::armor_texture(
                mc_render::entity::layers::armor_material_name(material),
                slot,
                Some("overlay"),
            );
            let key = if phase.load_texture(gpu, resources, &overlay) {
                overlay.as_str()
            } else {
                mc_render::entity::EntityPhase::FALLBACK_TEXTURE
            };
            phase.push_blended(key, mc_render::entity::LayerBlend::Opaque, scratch);
        }
        // java: `LayerArmorBase.renderGlint` replays the posed armour under the standalone
        // enchanted-item sheet after the leather base/overlay passes. The source must be the
        // untinted armour geometry (the overlay re-emission above leaves exactly that in `scratch`).
        if item.enchanted {
            let mut glint = Vec::with_capacity(scratch.len() * 2);
            mc_render::entity::item::emit_armor_glint(scratch, params.age_in_ticks, &mut glint);
            if !glint.is_empty()
                && phase.load_texture(
                    gpu,
                    resources,
                    mc_render::entity::item::ENCHANTED_ITEM_GLINT_TEXTURE,
                )
            {
                phase.push_blended(
                    mc_render::entity::item::ENCHANTED_ITEM_GLINT_TEXTURE,
                    mc_render::entity::LayerBlend::Glint,
                    &glint,
                );
            }
        }
    }
    let _ = entity_type;
}

/// Replay an Enderman's `carried`/`carriedData` block from SpawnData through the same baked block
/// model route as live entities.
// The renderer APIs deliberately receive the same phase/model/resource bundle as the other
// entity layers; collapsing it would risk changing texture upload and submission ordering.
#[allow(clippy::too_many_arguments)]
fn emit_spawner_carried_block(
    models: &mut mc_render::entity::RenderManager,
    phase: &mut mc_render::entity::EntityPhase,
    gpu: &Gpu,
    resources: &mc_assets::ResourceManager,
    atlas: &mc_assets::Atlas,
    body_kind: mc_render::entity::ModelKind,
    base: mc_render::entity::Mat4,
    block_state: mc_data::BlockState,
    scratch: &mut Vec<EntityVertex>,
) {
    let Some(model) = super::appearance::block_item_model(block_state) else {
        return;
    };
    scratch.clear();
    let _ = models;
    mc_render::entity::item::emit_block_model_brightness(
        model,
        mc_render::entity::layers::enderman_held_block_matrix(base),
        1.0,
        mc_render::terrain::tint::render_color(block_state),
        scratch,
    );
    if scratch.is_empty() {
        return;
    }
    if !phase.has_texture(BLOCK_ATLAS_TEXTURE) {
        phase.upload_texture(gpu, BLOCK_ATLAS_TEXTURE, &atlas.texture, atlas.w, atlas.h);
    }
    let _ = (body_kind, resources);
    phase.push_blended(
        BLOCK_ATLAS_TEXTURE,
        mc_render::entity::LayerBlend::Opaque,
        scratch,
    );
}

/// java: `CommandBlockLogic.readDataFromNBT` (`command/server/CommandBlockLogic.java:69-89`).
///
/// The two required-looking fields (`Command` and `SuccessCount`) use NBT getters directly and
/// therefore reset to Java's getter defaults when absent or mistyped. `CustomName` and
/// `TrackOutput` are guarded by exact tag checks, so absence (or a wrong tag) preserves the live
/// value. `LastOutput` is similarly conditional on an exact string tag and the *resulting*
/// `track_output` flag. Keeping this merge in the reader makes a replacement in
/// `World::apply_tile_entity_nbt` atomic while preserving all of those read semantics.
fn read_command_block(
    nbt: &mc_nbt::NbtCompound,
    previous: Option<&mc_world::tileent::CommandBlockState>,
) -> mc_world::tileent::CommandBlockState {
    let mut state = previous.cloned().unwrap_or_default();
    state.command = nbt.get_str("Command").unwrap_or_default().to_owned();
    state.success_count = nbt.get_i32("SuccessCount").unwrap_or(0);

    if nbt.contains_type("CustomName", mc_nbt::TagType::String) {
        state.custom_name = nbt.get_str("CustomName").unwrap_or_default().to_owned();
    }
    if nbt.contains_type("TrackOutput", mc_nbt::TagType::Byte) {
        state.track_output = nbt.get_bool("TrackOutput").unwrap_or(false);
    }
    if nbt.contains_type("LastOutput", mc_nbt::TagType::String) && state.track_output {
        state.last_output = Some(nbt.get_str("LastOutput").unwrap_or_default().to_owned());
    }

    state
}

/// The skin URL inside an authlib `textures` property, which is a base64 JSON blob.
///
/// java: `NBTUtil.readGameProfileFromNBT` builds the `GameProfile`; `SkinManager.loadSkin` then
/// reads the same property. The URL is embedded, so this needs no network — but *fetching* it
/// does, and there is none in this sandbox, so the download is not wired.
#[cfg(test)]
fn skin_url_from_profile(profile: &mc_auth::GameProfile) -> Option<String> {
    profile
        .properties
        .iter()
        .find(|property| property.name == "textures")
        .and_then(|property| super::appearance::decode_textures_property(&property.value))
        .and_then(|textures| textures.skin_url)
}

/// Standard base64, decoded without pulling the `base64` crate's API into this module (it is a
/// dependency of `mc-client` for server-list favicons and its version's API differs by release).
#[cfg(test)]
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for byte in input.bytes() {
        if byte == b'=' || byte.is_ascii_whitespace() {
            continue;
        }
        let value = ALPHABET.iter().position(|&c| c == byte)? as u32;
        acc = (acc << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xFF) as u8);
        }
    }
    Some(out)
}

/// java: `TileEntityBanner.readFromNBT` — `Base` plus the `Patterns` list of
/// `{Pattern: <hashname>, Color: <dye>}` compounds.
fn read_banner(nbt: &mc_nbt::NbtCompound) -> mc_world::BannerState {
    let mut state = mc_world::BannerState {
        base_color: nbt.get_i32("Base").unwrap_or(0),
        patterns: Vec::new(),
        initialized: true,
    };
    if let Some(list) = nbt.get_list("Patterns") {
        for entry in list {
            let Some(compound) = entry.as_compound() else {
                continue;
            };
            let Some(pattern) = compound.get_str("Pattern") else {
                continue;
            };
            // java: `EnumBannerPattern.getPatternByID(...) != null` guards the append.  Keeping
            // unknown ids in the state would make them look like valid masks at render time.
            if mc_world::tileent::banner_pattern_name(pattern).is_none() {
                continue;
            }
            state
                .patterns
                .push((pattern.to_owned(), compound.get_i32("Color").unwrap_or(0)));
        }
    }
    state
}

#[cfg(test)]
mod tests {
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::NetHandle;
    use crate::session::{JoinSettings, Session};
    use mc_data::{BlockId, BlockState};
    use mc_math::ChunkPos;
    use mc_nbt::{Nbt, NbtCompound};
    use mc_protocol::play::s2c;
    use mc_render::entity::tilepass::{BANNER_ITEM_ID, SKULL_ITEM_ID};
    use mc_render::particle::packet;
    use mc_render::particle::{ParticleKind, ParticleSystem};

    use super::*;

    #[test]
    fn visible_beacon_upload_owns_the_per_frame_fade_after_the_distance_gate() {
        let source = include_str!("tiles.rs");
        let test_module = source
            .find("#[cfg(test)]\r\nmod tests")
            .or_else(|| source.find("#[cfg(test)]\nmod tests"))
            .expect("production/test boundary");
        let production = &source[..test_module];
        let distance_gate = production
            .find(">= kind.max_render_distance_sq()")
            .expect("tile distance gate");
        let frame_step = production
            .find("world.advance_beacon_render_frame(header_pos, world_time);")
            .expect("visible beacon frame step");
        let snapshot = production[frame_step..]
            .find("world.tile_entity(header_pos)")
            .map(|offset| frame_step + offset)
            .expect("post-step tile snapshot");

        assert!(
            distance_gate < frame_step,
            "out-of-range beacons do not advance"
        );
        assert!(
            frame_step < snapshot,
            "the draw snapshots the advanced fade"
        );
        assert!(
            !production.contains("tick_beacon_fade"),
            "the 20 Hz tick path must not own renderer fade"
        );
    }

    /// The beam animates on the resync-smoothed render clock (proven
    /// frame by frame in `tests/w7_beacon_resync.rs`), while the fade keeps vanilla's raw
    /// `getTotalWorldTime()`: `shouldBeamRender` is beacon state, not an animation.
    #[test]
    fn the_beam_animates_on_the_render_clock_and_the_fade_on_vanillas() {
        let source = include_str!("tiles.rs");
        let test_module = source
            .find("#[cfg(test)]\r\nmod tests")
            .or_else(|| source.find("#[cfg(test)]\nmod tests"))
            .expect("production/test boundary");
        let production = &source[..test_module];
        assert!(production.contains("world_time = state.env.total_world_time;"));
        assert!(production.contains("beam_clock = state.env.render_world_time(partial_ticks);"));
        assert!(production.contains("world.advance_beacon_render_frame(header_pos, world_time);"));
        let emit = production
            .find("emit_beacon(&beacons, draw.rel, beacon, beam_ticks, beam_partial);")
            .expect("the beam reads the render clock");
        assert!(production[..emit].contains("let (beam_ticks, beam_partial) = beam_clock;"));
    }

    #[test]
    fn live_destroy_stage_attachment_matches_tesr_families_and_rejects_banner_or_bad_packets() {
        assert_eq!(
            destroy_stage_scale(TileKind::Chest, false),
            Some((4.0, 4.0))
        );
        assert_eq!(
            destroy_stage_scale(TileKind::Chest, true),
            Some((8.0, 4.0)),
            "large chest replay uses Java's 8x4 texture matrix"
        );
        assert_eq!(
            destroy_stage_scale(TileKind::EnderChest, false),
            Some((4.0, 4.0))
        );
        assert_eq!(
            destroy_stage_scale(TileKind::StandingSign, false),
            Some((4.0, 2.0))
        );
        assert_eq!(
            destroy_stage_scale(TileKind::Skull, false),
            Some((4.0, 2.0))
        );
        assert_eq!(
            destroy_stage_scale(TileKind::StandingBanner, false),
            None,
            "banner TESR ignores destroyStage and stays on ordinary block overlay"
        );

        // The S22/S23 consumer may receive repeated stages or a malformed stage >= 10.  Java's
        // renderer has exactly ten destroy sprites; malformed values must not bind an arbitrary
        // texture and repeated updates should not duplicate the crack geometry.
        assert_eq!(
            legal_destroy_stages(&[9, 2, 9, 10, 0, 255, 2]),
            vec![0, 2, 9]
        );
    }

    /// 06-entity-batch U3: the TESR draw order is a pure function of the visible set (not of
    /// `TileIndex`'s hash order), groups each sheet into one contiguous run, keeps position-free
    /// kinds in their relative order, and puts translucent pistons after every opaque TESR.
    #[test]
    fn tesr_draw_order_is_hash_independent_and_groups_sheets() {
        let draw = |kind: TileKind, x: i32, double: bool, skull_type: u8| TileDraw {
            pos: BlockPos::new(x, 64, 0),
            kind,
            double,
            skull: (skull_type, 0, None),
            ..TileDraw::default()
        };
        let visible = || {
            vec![
                draw(TileKind::Piston, 9, false, 0),
                draw(TileKind::WallSign, 4, false, 0),
                draw(TileKind::Chest, 7, false, 0),
                draw(TileKind::EndPortal, 3, false, 0),
                draw(TileKind::Skull, 2, false, 1),
                draw(TileKind::Chest, 1, true, 0),
                draw(TileKind::Piston, 0, false, 0),
                draw(TileKind::StandingSign, 8, false, 0),
                draw(TileKind::Chest, 5, false, 0),
                draw(TileKind::EndPortal, 6, false, 0),
                draw(TileKind::Skull, 10, false, 0),
            ]
        };
        let order = |mut draws: Vec<TileDraw>| {
            draws.sort_by_key(tile_draw_order_key);
            draws.iter().map(|d| (d.kind, d.pos.x)).collect::<Vec<_>>()
        };
        let sorted = order(visible());
        assert_eq!(
            sorted,
            vec![
                (TileKind::Piston, 9),
                (TileKind::Piston, 0),
                (TileKind::Chest, 5),
                (TileKind::Chest, 7),
                (TileKind::Chest, 1),
                (TileKind::WallSign, 4),
                (TileKind::StandingSign, 8),
                (TileKind::Skull, 10),
                (TileKind::Skull, 2),
                (TileKind::EndPortal, 3),
                (TileKind::EndPortal, 6),
            ],
            "pistons first, then single chests, the double-chest sheet, signs, skulls by type; \
             portals and pistons keep their relative order"
        );
        // Any permutation of the opaque TESRs gives the same order: no hash-order dependence.
        let opaque = || {
            visible()
                .into_iter()
                .filter(|d| !matches!(d.kind, TileKind::Piston | TileKind::EndPortal))
                .collect::<Vec<TileDraw>>()
        };
        let mut reversed = opaque();
        reversed.reverse();
        assert_eq!(order(reversed), order(opaque()));
    }

    #[test]
    fn live_tile_upload_consumes_packet_stages_in_tesr_order_with_banner_negative_control() {
        // Production code only: every pattern below also appears in this test, so a stale
        // pattern must fail to match instead of matching itself.
        let (source, _) = include_str!("tiles.rs")
            .split_once("#[cfg(test)]\nmod tests {")
            .expect("tiles keeps its inline test module");
        let attach = source
            .find("destroy_stage_scale(draw.kind, draw.double)")
            .expect("live tile upload stage gate");
        let replay = source[attach..]
            .find("remap_destroy_stage_uvs(geometry, scale_u, scale_v)")
            .map(|offset| attach + offset)
            .expect("live tile geometry receives the stage UV matrix");
        let packet_snapshot = source
            .find("damaged_tile_stages.get(&pos).cloned()")
            .expect("live packet/state snapshot reaches TileDraw");
        assert!(
            packet_snapshot < attach && attach < replay,
            "S25 state must be snapshotted before the TESR replay"
        );
        let stage_path = &source[attach..replay];
        assert!(stage_path.contains("destroy_stage_{stage}.png"));
        assert!(
            !stage_path.contains("TileKind::StandingBanner")
                && !stage_path.contains("TileKind::WallBanner"),
            "banner remains on ordinary terrain damage"
        );
    }

    #[test]
    fn tile_phase_anaglyph_is_seeded_before_lazy_tesr_uploads() {
        let source = include_str!("tiles.rs");
        let sync = source
            .find("phase.set_anaglyph_with_gpu(gpu, self.settings.anaglyph)")
            .expect("tile phase owns a public live anaglyph hook");
        let begin = source
            .find("tiles.with(TilePhase::begin);")
            .expect("tile phase frame reset");
        let first_upload = source
            .find("phase.upload_texture(")
            .expect("lazy TESR texture upload");
        assert!(
            sync < begin && begin < first_upload,
            "GameSettings.anaglyph must be replayed before any tile atlas/builtin upload"
        );
        assert!(
            source.contains("false -> true -> false"),
            "the tile caller documents raw-byte replay instead of double conversion"
        );
    }

    fn item_stack(item_id: u16, damage: i16) -> mc_protocol::ItemStack {
        mc_protocol::ItemStack {
            item_id: item_id as i16,
            count: 1,
            damage,
            nbt: None,
        }
    }

    #[test]
    fn builtin_item_adapter_routes_the_five_vanilla_entity_item_families() {
        assert_eq!(
            builtin_tile_item_from_stack(&item_stack(mc_world::tileent::CHEST_ID, 0)),
            Some(BuiltinTileItem::Chest { trapped: false })
        );
        assert_eq!(
            builtin_tile_item_from_stack(&item_stack(mc_world::tileent::TRAPPED_CHEST_ID, 0)),
            Some(BuiltinTileItem::Chest { trapped: true })
        );
        assert_eq!(
            builtin_tile_item_from_stack(&item_stack(mc_world::tileent::ENDER_CHEST_ID, 0)),
            Some(BuiltinTileItem::EnderChest)
        );
        assert_eq!(
            builtin_tile_item_from_stack(&item_stack(BANNER_ITEM_ID, 15)),
            Some(BuiltinTileItem::Banner {
                state: mc_world::BannerState {
                    base_color: 15,
                    patterns: Vec::new(),
                    initialized: true,
                },
            })
        );
        assert_eq!(
            builtin_tile_item_from_stack(&item_stack(SKULL_ITEM_ID, 2)),
            Some(BuiltinTileItem::Skull { skull_type: 2 })
        );

        // Negative controls: an ordinary block item and a negative registry id never enter the
        // custom renderer.  The latter matters because ItemStack ids are signed on the wire.
        assert_eq!(builtin_tile_item_from_stack(&item_stack(1, 0)), None);
        assert_eq!(
            builtin_tile_item_from_stack(&mc_protocol::ItemStack {
                item_id: -1,
                ..item_stack(1, 0)
            }),
            None
        );
    }

    #[test]
    fn player_skull_item_owner_uses_skin_cache_and_uuid_fallback_without_http() {
        let uuid = uuid::Uuid::from_u128(0xfeed_beef_cafe_babe);
        let payload = r#"{"textures":{"SKIN":{"url":"https://textures.minecraft.net/texture/player-skull-test","metadata":{"model":"slim"}}}}"#;
        let mut property = NbtCompound::new();
        property.insert("Value", Nbt::String(encode(payload.as_bytes())));
        let mut properties = NbtCompound::new();
        properties.insert("textures", Nbt::list(vec![Nbt::Compound(property)]));
        let mut owner = NbtCompound::new();
        owner.insert("Id", Nbt::String(uuid.to_string()));
        owner.insert("Name", Nbt::String("ProfileHead".to_owned()));
        owner.insert("Properties", Nbt::Compound(properties));
        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::Compound(owner));
        let mut stack = item_stack(SKULL_ITEM_ID, 3);
        stack.nbt = Some(root);

        let item = builtin_tile_item_from_stack(&stack).expect("player skull route");
        let BuiltinTileItem::PlayerSkull { skull_type, owner } = &item else {
            panic!("valid SkullOwner must retain its player profile");
        };
        assert_eq!(*skull_type, 3);
        // java: `SkinManager.loadSkin` names the texture by its URL hash.
        assert_eq!(
            Some(owner.cache_key.clone()),
            crate::app::appearance::skin_texture_key_for_url(
                "https://textures.minecraft.net/texture/player-skull-test"
            )
        );
        assert_eq!(owner.model, mc_render::entity::SkinModel::Slim);
        assert_eq!(
            owner.texture_key(),
            mc_render::entity::SkinModel::default_for_uuid(uuid.as_u128()).default_texture()
        );

        // A cache poll only transfers the request to its ordinary worker channel.  No runtime,
        // socket, or GPU is involved in this deterministic check. The handoff queue is
        // process-wide, so other tests in this binary may have queued heads of their own: check
        // this head's request, not the total.
        let mut appearance = crate::app::appearance::AppearanceState::new();
        assert_eq!(appearance.skins.poll(), 0);
        assert!(
            appearance.skins.is_inflight(&owner.cache_key),
            "the poll handed this head's skin to the worker channel"
        );
        assert_eq!(appearance.skins.stats().1, 0, "nothing was delivered");

        appearance.skins.insert(
            &owner.cache_key,
            mc_render::entity::SkinTexture {
                rgba: vec![255; 64 * 64 * 4].into(),
                layout: mc_render::entity::SkinLayout::Modern64x64,
                model: mc_render::entity::SkinModel::Slim,
            },
        );
        let ready = builtin_tile_item_from_stack(&stack).expect("player skull route remains");
        let BuiltinTileItem::PlayerSkull {
            owner: ready_owner, ..
        } = ready
        else {
            panic!("valid SkullOwner changed family after cache completion");
        };
        assert_eq!(ready_owner.texture_key(), owner.cache_key);
    }

    /// <<W18-PA:skull-descriptor>> The client's head path goes through the shared
    /// `skinCacheLoader` descriptor cache (`mc_render::entity::skin::PlayerHeadProfiles`, whose
    /// own tests pin the 15 s expiry): an equal profile (same id and name) drawn right after keeps
    /// the first texture, another name with the same UUID gets its own, and every key is the
    /// texture hash. The skins are published first, so nothing is queued in the process-wide
    /// handoff that other tests in this binary read.
    #[test]
    fn player_skull_owners_share_the_profile_descriptor_cache() {
        let uuid = uuid::Uuid::from_u128(0x5a11_0000_0000_4000_8000_0000_0000_0001);
        let profile = |name: &str, hash: &str| {
            let payload = format!(
                r#"{{"textures":{{"SKIN":{{"url":"https://textures.minecraft.net/texture/{hash}"}}}}}}"#
            );
            mc_auth::GameProfile {
                id: Some(uuid),
                name: name.to_owned(),
                properties: vec![mc_auth::GameProfileProperty {
                    name: "textures".to_owned(),
                    value: encode(payload.as_bytes()),
                    signature: None,
                }],
            }
        };
        let key = |hash: &str| {
            crate::app::appearance::skin_texture_key_for_url(&format!(
                "https://textures.minecraft.net/texture/{hash}"
            ))
            .expect("safe hash")
        };
        let hashes = [
            "w18-pa-head-first",
            "w18-pa-head-changed",
            "w18-pa-head-other",
        ];
        for hash in hashes {
            mc_render::entity::skin::publish_shared_skin(
                &key(hash),
                &mc_render::entity::SkinTexture {
                    rgba: vec![255; 64 * 64 * 4].into(),
                    layout: mc_render::entity::SkinLayout::Modern64x64,
                    model: mc_render::entity::SkinModel::Classic,
                },
            );
        }
        let owner = |profile: &mc_auth::GameProfile| player_skull_owner_from_profile(profile);

        let first = owner(&profile("HeadOwner", hashes[0]));
        assert_eq!(first.cache_key, key(hashes[0]));
        assert_eq!(
            first.texture_key(),
            key(hashes[0]),
            "a published skin binds at once"
        );
        // java: the cached textures map of the equal profile wins while it is hot.
        let changed = owner(&profile("HeadOwner", hashes[1]));
        assert_eq!(
            changed.cache_key,
            key(hashes[0]),
            "equal profile, cached textures"
        );
        // Another name is another `GameProfile` key, even with the same UUID.
        let other = owner(&profile("OtherName", hashes[2]));
        assert_eq!(other.cache_key, key(hashes[2]));
        // A profile without a SKIN keeps the UUID-derived default.
        let bare = owner(&mc_auth::GameProfile {
            id: Some(uuid),
            name: "Bare".to_owned(),
            properties: Vec::new(),
        });
        assert_eq!(
            bare.texture_key(),
            mc_render::entity::SkinModel::default_for_uuid(uuid.as_u128()).default_texture()
        );
    }

    #[test]
    fn legacy_string_skull_owner_uses_name_derived_default_without_network() {
        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::String("jeb_".to_owned()));
        let mut stack = item_stack(SKULL_ITEM_ID, 3);
        stack.nbt = Some(root);

        let item = builtin_tile_item_from_stack(&stack).expect("legacy player skull route");
        let BuiltinTileItem::PlayerSkull { owner, .. } = item else {
            panic!("a non-empty legacy SkullOwner must retain the player-skull family");
        };
        assert_eq!(owner.legacy_name.as_deref(), Some("jeb_"));
        let expected = mc_render::entity::SkinModel::default_for_uuid(
            mc_auth::offline::offline_uuid("jeb_").as_u128(),
        );
        assert_eq!(expected, mc_render::entity::SkinModel::Slim);
        assert_eq!(owner.model, expected);
        assert_eq!(owner.fallback_model, expected);
        assert_eq!(owner.texture_key(), expected.default_texture());
        assert_eq!(owner.cache_key, "mc-render/entity/skin/legacy/6a65625f");

        // Empty names and wrong NBT types follow TileEntityItemStackRenderer's ordinary skull
        // branch instead of manufacturing a profile or a cache request.
        let mut empty = item_stack(SKULL_ITEM_ID, 3);
        let mut empty_root = NbtCompound::new();
        empty_root.insert("SkullOwner", Nbt::String(String::new()));
        empty.nbt = Some(empty_root);
        assert_eq!(
            builtin_tile_item_from_stack(&empty),
            Some(BuiltinTileItem::Skull { skull_type: 3 })
        );
        let mut wrong = item_stack(SKULL_ITEM_ID, 3);
        let mut wrong_root = NbtCompound::new();
        wrong_root.insert("SkullOwner", Nbt::Int(9));
        wrong.nbt = Some(wrong_root);
        assert_eq!(
            builtin_tile_item_from_stack(&wrong),
            Some(BuiltinTileItem::Skull { skull_type: 3 })
        );
    }

    #[test]
    fn legacy_string_skull_owner_uses_case_insensitive_live_player_profile() {
        let uuid = uuid::Uuid::from_u128(1);
        let mut players = crate::sync::PlayerList::new();
        players
            .apply(&s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
                uuid,
                username: "legacynamE".to_owned(),
                properties: Vec::new(),
                game_mode: s2c::GameMode::Survival,
                ping: 0,
                display_name: None,
            }]))
            .expect("synthetic player-list row");

        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::String("LegacyName".to_owned()));
        let mut stack = item_stack(SKULL_ITEM_ID, 3);
        stack.nbt = Some(root);

        let item = builtin_tile_item_from_stack_with_players(&stack, Some(&players))
            .expect("legacy player skull route");
        let BuiltinTileItem::PlayerSkull { owner, .. } = item else {
            panic!("legacy player-list profile must retain the player-skull family");
        };
        assert_eq!(owner.legacy_name, None);
        assert_eq!(
            owner.cache_key,
            crate::app::appearance::skin_texture_key(uuid)
        );
        assert_eq!(owner.fallback_model, mc_render::entity::SkinModel::Slim);
        assert_eq!(owner.texture_key(), "textures/entity/alex.png");
    }

    #[test]
    fn legacy_string_skull_owner_mutates_to_name_only_compound_when_no_local_profile_exists() {
        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::String("LegacyName".to_owned()));
        root.insert("Unrelated", Nbt::Int(17));
        let mut stack = item_stack(SKULL_ITEM_ID, 3);
        stack.nbt = Some(root);

        let profile = normalize_legacy_skull_owner_in_stack(&mut stack, None)
            .expect("a non-empty legacy name is a valid name-only profile");
        assert_eq!(profile.id, None);
        assert_eq!(profile.name, "LegacyName");
        let owner = stack
            .nbt
            .as_ref()
            .and_then(|root| root.get_compound("SkullOwner"))
            .expect("the mutable seam promotes the root string");
        assert_eq!(owner.get_str("Name"), Some("LegacyName"));
        assert_eq!(owner.get_str("Id"), None);
        assert_eq!(
            stack
                .nbt
                .as_ref()
                .and_then(|root| root.get_i32("Unrelated")),
            Some(17),
            "normalization does not discard unrelated root item NBT"
        );
    }

    #[test]
    fn legacy_string_skull_owner_mutation_uses_local_profile_and_rejects_malformed_controls() {
        let uuid = uuid::Uuid::from_u128(9);
        let mut players = crate::sync::PlayerList::new();
        players
            .apply(&s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
                uuid,
                username: "LegacyName".to_owned(),
                properties: Vec::new(),
                game_mode: s2c::GameMode::Survival,
                ping: 0,
                display_name: None,
            }]))
            .expect("synthetic local profile");

        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::String("legacynamE".to_owned()));
        let mut stack = item_stack(SKULL_ITEM_ID, 3);
        stack.nbt = Some(root);
        let profile = normalize_legacy_skull_owner_in_stack(&mut stack, Some(&players))
            .expect("local tab-list name resolves without network");
        assert_eq!(profile.id, Some(uuid));
        assert_eq!(profile.name, "LegacyName");
        let owner = stack
            .nbt
            .as_ref()
            .and_then(|root| root.get_compound("SkullOwner"))
            .expect("resolved profile is written back as a compound");
        assert_eq!(owner.get_str("Name"), Some("LegacyName"));
        assert_eq!(owner.get_str("Id"), Some(uuid.to_string().as_str()));

        for malformed in [
            None,
            Some(Nbt::String(String::new())),
            Some(Nbt::Int(4)),
            Some(Nbt::Compound(NbtCompound::new())),
        ] {
            let mut malformed_root = NbtCompound::new();
            if let Some(value) = malformed {
                malformed_root.insert("SkullOwner", value);
            }
            let before = malformed_root.clone();
            let mut malformed_stack = item_stack(SKULL_ITEM_ID, 3);
            malformed_stack.nbt = Some(malformed_root);
            let result =
                normalize_legacy_skull_owner_in_stack(&mut malformed_stack, Some(&players));
            assert_eq!(result, None);
            assert_eq!(malformed_stack.nbt, Some(before));
        }
    }

    #[test]
    fn model_aware_builtin_adapter_keeps_strict_ids_for_ordinary_models_and_chest_fallback_for_sentinel(
    ) {
        let stack = item_stack(1, 0); // stone: not one of vanilla's five builtin/entity ids
        let sentinel = mc_assets::BakedModel {
            ambient_occlusion: false,
            gui_3d: true,
            particle_uv: None,
            ..mc_assets::BakedModel::default()
        };
        assert_eq!(
            builtin_tile_item_from_stack_with_players_and_model(&stack, None, &sentinel),
            Some(BuiltinTileItem::Chest { trapped: false })
        );
        assert_eq!(
            builtin_tile_item_from_stack_with_players_and_model(
                &stack,
                None,
                &mc_assets::BakedModel::default()
            ),
            None,
            "an ordinary empty/default model is not the Java builtin/entity gate"
        );
        let chest = item_stack(mc_world::tileent::CHEST_ID, 0);
        let mut ordinary_known = sentinel.clone();
        ordinary_known.particle_uv = Some(mc_assets::sprite::UvRect::FULL);
        assert_eq!(
            builtin_tile_item_from_stack_with_players_and_model(&chest, None, &ordinary_known),
            None,
            "known ids still obey RenderItem's built-in-model gate"
        );
    }

    #[test]
    fn banner_item_adapter_matches_set_item_values_numeric_and_pattern_gates() {
        let mut block_entity = NbtCompound::new();
        block_entity.insert("Base", Nbt::Byte(14));
        block_entity.insert(
            "Patterns",
            Nbt::list(vec![
                Nbt::Compound({
                    let mut pattern = NbtCompound::new();
                    pattern.insert("Pattern", Nbt::String("cr".to_owned()));
                    pattern.insert("Color", Nbt::Short(1));
                    pattern
                }),
                Nbt::Compound({
                    let mut pattern = NbtCompound::new();
                    pattern.insert("Pattern", Nbt::String("not_a_pattern".to_owned()));
                    pattern.insert("Color", Nbt::Int(2));
                    pattern
                }),
                // TileEntityBanner.getTagList(key, 10) ignores a non-compound entry.
                Nbt::String("wrong element type".to_owned()),
            ]),
        );
        let mut stack = item_stack(BANNER_ITEM_ID, 3);
        let mut root = NbtCompound::new();
        root.insert("BlockEntityTag", Nbt::Compound(block_entity));
        stack.nbt = Some(root);

        let item = builtin_tile_item_from_stack(&stack).expect("banner item route");
        let BuiltinTileItem::Banner { state } = item else {
            panic!("banner item must produce Banner state");
        };
        assert_eq!(
            state.base_color, 14,
            "numeric Base overrides stack metadata"
        );
        assert_eq!(state.patterns, vec![("cr".to_owned(), 1)]);
        assert!(state.initialized);

        for base in [
            Nbt::Byte(6),
            Nbt::Short(6),
            Nbt::Int(6),
            Nbt::Long(6),
            Nbt::Float(6.0),
            Nbt::Double(6.0),
        ] {
            let mut block_entity = NbtCompound::new();
            block_entity.insert("Base", base);
            let mut root = NbtCompound::new();
            root.insert("BlockEntityTag", Nbt::Compound(block_entity));
            let mut numeric = item_stack(BANNER_ITEM_ID, 3);
            numeric.nbt = Some(root);
            assert_eq!(banner_item_state(&numeric).base_color, 6);
        }

        let mut string_base = NbtCompound::new();
        string_base.insert("Base", Nbt::String("6".into()));
        let mut string_root = NbtCompound::new();
        string_root.insert("BlockEntityTag", Nbt::Compound(string_base));
        let mut nonnumeric = item_stack(BANNER_ITEM_ID, 3);
        nonnumeric.nbt = Some(string_root);
        assert_eq!(
            banner_item_state(&nonnumeric).base_color,
            3,
            "setItemValues rejects present nonnumeric Base"
        );

        // Java checks BlockEntityTag's compound type before reading either field.  A malformed
        // root therefore keeps the stack metadata and contributes no patterns.
        let mut malformed = item_stack(BANNER_ITEM_ID, 7);
        let mut nbt = NbtCompound::new();
        nbt.insert("BlockEntityTag", Nbt::String("not a compound".to_owned()));
        malformed.nbt = Some(nbt);
        assert_eq!(
            builtin_tile_item_from_stack(&malformed),
            Some(BuiltinTileItem::Banner {
                state: mc_world::BannerState {
                    base_color: 7,
                    patterns: Vec::new(),
                    initialized: true,
                },
            })
        );

        // Damage values outside the skull switch are Java's default skeleton, not a wrapped
        // subtype.  This catches a lossy i16 -> u8 cast in the adapter.
        let skull = builtin_tile_item_from_stack(&item_stack(SKULL_ITEM_ID, 258));
        assert_eq!(
            skull,
            Some(BuiltinTileItem::Skull {
                skull_type: u8::MAX
            })
        );
    }

    #[test]
    fn banner_render_policy_requires_the_s35_description_latch() {
        let lazy = mc_world::TileEntity::new(mc_world::TileKind::StandingBanner);
        assert!(
            banner_state_for_render(None).is_none(),
            "a missing tile entity cannot produce banner geometry"
        );
        assert!(
            banner_state_for_render(Some(&lazy)).is_none(),
            "the Java default field_175119_g=false skips the whole model"
        );

        let described = mc_world::TileEntity {
            kind: mc_world::TileKind::StandingBanner,
            data: mc_world::TileData::Banner(mc_world::BannerState {
                base_color: 14,
                patterns: vec![("cr".to_owned(), 1)],
                initialized: true,
            }),
        };
        let rendered = banner_state_for_render(Some(&described)).expect("S35 makes it visible");
        assert_eq!(rendered.base_color, 14);
        assert_eq!(rendered.patterns, vec![("cr".to_owned(), 1)]);

        // Biting mutant: removing the `initialized` filter would make the lazily-created banner
        // above return `Some` and fail the assertion before this positive path is reached.
    }

    /// java: `TileEntitySignRenderer.renderTileEntityAt:95-96` draws
    /// `splitText(row, 90, fontrenderer, false, true).get(0).getFormattedText()`, and `splitText`
    /// walks the row's siblings: bytecode `avu.a` offset 20 is `Lists.newArrayList(Iterable)` over
    /// the component. Every expected string is the real 1.8.9 bytecode's output for the same row
    /// with the jar's `en_US.lang` and `ascii.png` (harness:
    /// The full table is
    /// `mc-ui/tests/vanilla_sign_text.rs`). This replaced a test that asserted the opposite —
    /// `{"text":"","extra":[{"text":"not visible"}]}` decoding to a bare `§r` — which is exactly
    /// how every `extra`-only sign (`{"extra":["-CXXVI"],"text":""}`, captured) went blank.
    #[test]
    fn sign_rows_decode_like_vanillas_split_text_including_extra_siblings() {
        let Some((_dir, client)) = real_client() else {
            return;
        };
        let (font, language) = (&client.font, &client.assets.language);
        let decode = |row: &str| sign_line_formatted_text(row, font, language);

        // A real 1.8.8 server's S33 row: an empty root, the text in `extra`.
        assert_eq!(
            decode(r#"{"extra":["-CXXVI"],"text":""}"#),
            "\u{a7}r\u{a7}r-CXXVI\u{a7}r"
        );
        assert_eq!(
            decode(r#"{"text":"","extra":[{"text":"not visible"}]}"#),
            "\u{a7}r\u{a7}rnot visible\u{a7}r"
        );
        // Each sibling keeps its own resolved style.
        assert_eq!(
            decode(
                r#"{"text":"base","color":"red","extra":[{"text":" child","color":"blue","extra":[{"text":" deep"}]}]}"#
            ),
            "\u{a7}r\u{a7}c\u{a7}cbase\u{a7}r\u{a7}9\u{a7}9 child\u{a7}r\u{a7}9\u{a7}9 deep\u{a7}r"
        );
        // A translation yields its expanded pieces, argument siblings, then the root's `extra`.
        assert_eq!(
            decode(r#"{"translate":"container.chest"}"#),
            "\u{a7}rChest\u{a7}r"
        );
        assert_eq!(
            decode(
                r#"{"translate":"chat.type.text","with":[{"text":"A","extra":[{"text":"B"}]}],"extra":[{"text":"C"}]}"#
            ),
            "\u{a7}r<\u{a7}rA\u{a7}rB\u{a7}r> \u{a7}rC\u{a7}r"
        );
        // Only the first 90 px line is drawn.
        assert_eq!(
            decode(r#"{"extra":["aaaa bbbb cccc dddd eeee ffff"],"text":""}"#),
            "\u{a7}r\u{a7}raaaa bbbb cccc\u{a7}r"
        );
        // S33 owns exactly four slots; an array row is one component with a sibling.
        assert_eq!(
            decode(r#"["three","four"]"#),
            "\u{a7}rthree\u{a7}rfour\u{a7}r"
        );
        assert_eq!(decode(r#""""#), "\u{a7}r\u{a7}r");
        // A malformed direct call keeps the raw fallback (the live S33 decoder rejects it
        // before dispatch), and must not panic.
        assert_eq!(decode("{malformed"), "{malformed");
    }

    #[test]
    fn a_null_sign_line_is_blank_and_keeps_raw_component() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let position = BlockPos::new(7, 65, 4);
        session
            .state_mut()
            .world
            .load_empty_chunk(position.chunk_pos());
        session
            .state_mut()
            .world
            .set_block(position, BlockState::new(BlockId(63), 0));
        client.session = Some(session);
        let lines = [
            "\"Have fun\"".to_owned(),
            "\"at\"".to_owned(),
            "\"play.test  !\"".to_owned(),
            "null".to_owned(),
        ];
        client.react_to_sign_update(position, &lines);
        let sign = client
            .session()
            .unwrap()
            .state()
            .world
            .tile_entity(position)
            .and_then(mc_world::TileEntity::sign)
            .expect("updated sign");
        assert_eq!(
            sign.lines[3], "",
            "nullable row must not draw the word null"
        );
        assert!(sign.lines[0].contains("Have fun"));
        assert_eq!(
            sign.components, lines,
            "retain wire components for sign editing"
        );
        for nullable in ["null", "[]", "", " \n\t"] {
            assert_eq!(
                sign_line_formatted_text(nullable, &client.font, &client.assets.language),
                ""
            );
        }
    }

    #[test]
    fn selected_empty_sign_line_survives_the_cursor_marker_render_gate() {
        assert!(should_render_sign_line("", true));
        assert!(!should_render_sign_line("", false));
        assert!(should_render_sign_line("text", false));

        let source = include_str!("tiles.rs");
        let upload = source
            .split("fn upload_sign_text")
            .nth(1)
            .and_then(|tail| tail.split("/// java: `TileEntityBeaconRenderer").next())
            .expect("bounded upload_sign_text source");
        assert!(upload.contains("should_render_sign_line(line, selected)"));
        assert!(
            !upload.contains("if line.is_empty()"),
            "an unconditional empty-line guard would discard the selected cursor markers"
        );
    }

    #[test]
    fn sign_text_decorations_use_the_fixed_sign_matrix_and_depth_pass() {
        let matrix = mc_render::entity::Mat4::from_translation(Vec3::new(3.0, 4.0, 5.0));
        let mut sink = SignTextSink {
            matrix,
            shade: 1.0,
            batches: Vec::new(),
            bars: Vec::new(),
        };
        sink.bar(SolidRect {
            x0: -2.0,
            y0: 1.0,
            x1: 6.0,
            y1: 2.0,
            color: 0xFF11_2233,
        });
        assert_eq!(sink.bars.len(), 6);
        assert_eq!(sink.bars[0].pos, [1.0, 5.0, 5.0]);
        assert_eq!(sink.bars[1].pos, [9.0, 5.0, 5.0]);
        assert_eq!(
            sink.bars[2].color,
            mc_render::entity::argb_to_rgba(0xFF11_2233)
        );
    }

    /// java: `TileEntitySignRenderer.renderTileEntityAt:83-84` — `glNormal3f(0, 0, -f3)` under
    /// `scale(f3, -f3, f3)` with `GL_RESCALE_NORMAL`, so the lit text keeps the 0.4 scene ambient
    /// plus `0.6 * f3 * cos` of diffuse, then the tile brightness, like the board.
    #[test]
    fn sign_text_keeps_vanillas_ambient_lighting_and_the_tile_brightness() {
        let f3 = tile::SIGN_TEXT_SCALE;
        // Standing, rotation 0: the text faces south, its normal points north into the board,
        // and only LIGHT0 (0.2, 1, -0.7) is on that side.
        let south = tile::sign_matrix(Vec3::ZERO, true, 0);
        let expected = 0.4 + 0.6 * f3 * 0.7 / Vec3::new(0.2, 1.0, -0.7).length();
        assert!((sign_text_shade(south, 1.0) - expected).abs() < 1e-6);
        assert!((sign_text_shade(south, 1.0) - 0.4035).abs() < 1e-3);
        // A wall sign facing east (metadata 5): its text normal points west, where only LIGHT1
        // (-0.2, 1, 0.7) has a component, and only its 0.2.
        let east = tile::sign_matrix(Vec3::ZERO, false, 5);
        let expected = 0.4 + 0.6 * f3 * 0.2 / Vec3::new(0.2, 1.0, -0.7).length();
        assert!((sign_text_shade(east, 1.0) - expected).abs() < 1e-6);
        // The tile's lightmap brightness scales it exactly as it scales the board.
        assert!((sign_text_shade(south, 0.25) - 0.25 * sign_text_shade(south, 1.0)).abs() < 1e-6);

        // The sink applies it to glyph RGB and leaves alpha alone.
        let mut sink = SignTextSink {
            matrix: south,
            shade: 0.5,
            batches: Vec::new(),
            bars: Vec::new(),
        };
        assert_eq!(sink.lit(0xFF80_4020), {
            let [r, g, b, a] = mc_render::entity::argb_to_rgba(0xFF80_4020);
            [r * 0.5, g * 0.5, b * 0.5, a]
        });
        sink.bar(SolidRect {
            x0: 0.0,
            y0: 0.0,
            x1: 1.0,
            y1: 1.0,
            color: 0xFFFF_FFFF,
        });
        assert_eq!(sink.bars[0].color, [0.5, 0.5, 0.5, 1.0]);
    }

    /// java: only `TileEntityEnchantmentTableRenderer.renderTileEntityAt:64` turns the cull on
    /// for its model, and only `TileEntityPistonRenderer` blends.
    #[test]
    fn only_the_enchantment_book_draws_with_the_back_face_cull() {
        assert_eq!(
            tile_blend(TileKind::EnchantmentTable),
            TileBlend::OpaqueCulled
        );
        assert_eq!(tile_blend(TileKind::Piston), TileBlend::Translucent);
        for kind in [
            TileKind::Chest,
            TileKind::TrappedChest,
            TileKind::EnderChest,
            TileKind::StandingSign,
            TileKind::WallSign,
            TileKind::StandingBanner,
            TileKind::WallBanner,
            TileKind::Skull,
        ] {
            assert_eq!(tile_blend(kind), TileBlend::Opaque, "{kind:?}");
        }
    }

    #[test]
    fn live_s33_sign_gate_locks_during_editor_and_restores_on_close() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let sign = mc_math::BlockPos::new(7, 65, 4);
        let stone = mc_math::BlockPos::new(8, 65, 4);
        let missing = mc_math::BlockPos::new(40, 65, 4);
        let initial_components = [
            r#"{"color":"red","text":"old"}"#.to_owned(),
            r#"{"bold":true,"text":"keep"}"#.to_owned(),
            "\"\"".to_owned(),
            "\"\"".to_owned(),
        ];
        {
            let world = &mut session.state_mut().world;
            world.load_empty_chunk(sign.chunk_pos());
            world.set_block(sign, BlockState::new(BlockId(63), 4));
            world.set_block(stone, BlockState::new(BlockId(1), 0));
            assert!(world.apply_sign_components(
                sign,
                [
                    "old".to_owned(),
                    "keep".to_owned(),
                    String::new(),
                    String::new(),
                ],
                initial_components.clone(),
            ));
        }
        client.session = Some(session);

        // S36 opens from the packet handler, then GuiEditSign.initGui emits setEditable(false).
        client.react_to_open_sign_editor(sign);
        let ctx = mc_ui::screen_stack::ScreenCtx::new(client.resolution(854, 480), 0.0);
        let language = client.assets.language.clone();
        assert!(matches!(&client.ui, Ui::EditSign(_)));
        if let Ui::EditSign(screen) = &mut client.ui {
            screen.init(&ctx, &language);
        }
        client.sync_sign_editor_render_state();
        client.drain_sign_actions();
        let sign_state = client
            .session()
            .and_then(|session| session.state().world.tile_entity(sign))
            .and_then(mc_world::TileEntity::sign)
            .expect("the sign block owns a live tile");
        assert!(!sign_state.editable);

        let incoming = [
            r#"{"text":"server"}"#.to_owned(),
            r#"{"text":"replacement"}"#.to_owned(),
            r#"{"text":"new"}"#.to_owned(),
            "\"\"".to_owned(),
        ];
        client.react_to_sign_update(sign, &incoming);
        let locked = client
            .session()
            .and_then(|session| session.state().world.tile_entity(sign))
            .and_then(mc_world::TileEntity::sign)
            .expect("the locked sign remains present");
        assert_eq!(locked.components, initial_components);
        assert_eq!(locked.lines[0], "old");

        // A real non-sign and an unloaded position take Java's flag=false warning path; neither
        // may be turned into a sign as a side effect of S33 dispatch.
        client.react_to_sign_update(stone, &incoming);
        client.react_to_sign_update(missing, &incoming);
        assert!(client
            .session()
            .and_then(|session| session.state().world.tile_entity(missing))
            .is_none());

        // Minecraft.displayGuiScreen(null) runs GuiEditSign.onGuiClosed: C12 is staged, then
        // setEditable(true) restores acceptance for subsequent server updates.
        client.show_screen(None);
        let closed = client
            .session()
            .and_then(|session| session.state().world.tile_entity(sign))
            .and_then(mc_world::TileEntity::sign)
            .expect("the closed sign remains present");
        assert!(closed.editable);
        assert_eq!(closed.components[0], initial_components[0]);
        client.react_to_sign_update(sign, &incoming);
        let updated = client
            .session()
            .and_then(|session| session.state().world.tile_entity(sign))
            .and_then(mc_world::TileEntity::sign)
            .expect("the sign remains present after reopening the gate");
        // java: `splitText(...).get(0).getFormattedText()` is the line root's `§r` followed by
        // the row's own piece — `{"text":"Shop"}` gives `§rShop§r` in the 1.8.9 bytecode.
        assert_eq!(updated.lines[0], "\u{a7}rserver\u{a7}r");
        assert_eq!(updated.components, incoming);
    }

    #[test]
    fn builtin_item_adapter_and_renderer_share_the_same_descriptor() {
        let stack = item_stack(mc_world::tileent::ENDER_CHEST_ID, 0);
        let item = builtin_tile_item_from_stack(&stack).expect("ender chest item route");
        assert_eq!(item.kind(), BuiltinTileItemKind::EnderChest);

        let mut models = TileModels::new(false);
        let mut vertices = Vec::new();
        let (returned, texture) = emit_builtin_tile_item(
            &mut models,
            &stack,
            mc_render::entity::Mat4::IDENTITY,
            &mut vertices,
        )
        .expect("render-side adapter route");
        assert_eq!(returned, item);
        assert_eq!(texture, "textures/entity/chest/ender.png");
        assert_eq!(vertices.len(), 3 * 6 * 6);
    }

    #[test]
    fn piston_head_state_selects_vanilla_facing_sticky_and_short_variants() {
        let normal = piston_head_state(false, Facing::East, false);
        assert_eq!(normal.get("facing"), Some("east"));
        assert_eq!(normal.get("type"), Some("normal"));
        assert_eq!(normal.get("short"), Some("false"));

        let sticky_short = piston_head_state(true, Facing::East, true);
        assert_eq!(sticky_short.get("facing"), Some("east"));
        assert_eq!(sticky_short.get("type"), Some("sticky"));
        assert_eq!(sticky_short.get("short"), Some("true"));
        assert_ne!(normal, sticky_short);

        // `SHORT` is the only property that changes across the half-way branch; facing and type
        // remain stable while the head's extension geometry changes.
        let sticky_long = piston_head_state(true, Facing::East, false);
        assert_eq!(sticky_long.get("facing"), Some("east"));
        assert_eq!(sticky_long.get("type"), Some("sticky"));
        assert_eq!(sticky_long.get("short"), Some("false"));
        assert_ne!(sticky_short, sticky_long);
    }

    #[test]
    fn moving_head_branch_preserves_carried_facing_when_travel_direction_differs() {
        // Java's first renderer branch mutates only SHORT on the carried head state.  A piston
        // moving east can therefore carry a north-facing head; rebuilding the head from the
        // carrier direction is the tempting but wrong visual rotation.
        let piston = PistonState {
            piston_state: BlockState::new(
                BlockId(PISTON_HEAD_ID),
                Facing::North.index() | 8, // sticky, north-facing carried head
            ),
            facing: Facing::East, // the moving carrier's travel direction
            extending: true,
            should_head_be_rendered: false,
            progress: 0.25,
            last_progress: 0.0,
        };
        let mut observed = None;
        visit_piston_models(piston, Vec3::ZERO, 1.0, |state, _, _, check_sides| {
            observed = Some((state.meta() & 7, state.meta() & 8 != 0, check_sides));
        });
        assert_eq!(
            observed,
            Some((Facing::North.index(), true, true)),
            "the <0.5 carried-head branch must retain its own facing"
        );
    }

    /// The old A05-15 audit finding described this as an absent renderer.  Keep a real baked
    /// model assertion here: a moving carrier must submit block geometry while interpolated
    /// progress is below one, and must submit nothing after Java's strict `< 1.0F` guard.
    #[test]
    fn moving_piston_tesr_emits_baked_geometry_and_honours_the_progress_guard() {
        let Some((_dir, _client)) = real_client() else {
            return;
        };

        let mut moving = PistonState {
            piston_state: BlockState::new(BlockId(1), 0),
            facing: Facing::East,
            extending: true,
            should_head_be_rendered: false,
            progress: 0.5,
            last_progress: 0.0,
        };
        let mut vertices = Vec::new();
        emit_piston(moving, Vec3::new(10.0, 20.0, 30.0), 1.0, &mut vertices);
        assert!(
            !vertices.is_empty(),
            "a live carrier must emit baked block quads"
        );
        let min_x = vertices
            .iter()
            .map(|vertex| vertex.pos[0])
            .fold(f32::INFINITY, f32::min);
        let max_x = vertices
            .iter()
            .map(|vertex| vertex.pos[0])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((min_x - 9.5).abs() < 1e-5, "east extension offset: {min_x}");
        assert!((max_x - 10.5).abs() < 1e-5, "unit carrier width: {max_x}");

        moving.last_progress = 0.5;
        moving.progress = 1.0;
        vertices.clear();
        emit_piston(moving, Vec3::new(10.0, 20.0, 30.0), 1.0, &mut vertices);
        assert!(
            vertices.is_empty(),
            "Java skips a carrier at interpolated progress 1.0"
        );
    }

    #[test]
    fn piston_extension_and_retraction_keep_java_face_cull_controls() {
        let Some((_dir, _client)) = real_client() else {
            return;
        };
        for name in ["minecraft:stained_glass", "minecraft:stone_slab"] {
            let state = mc_data::block_by_name(name)
                .expect("piston translucent/slab state is registered")
                .default_state();
            let extending = PistonState {
                piston_state: state,
                facing: Facing::East,
                extending: true,
                should_head_be_rendered: false,
                progress: 0.5,
                last_progress: 0.0,
            };
            let mut extension_flags = Vec::new();
            visit_piston_models(extending, Vec3::ZERO, 1.0, |_, _, _, check_sides| {
                extension_flags.push(check_sides)
            });
            assert_eq!(extension_flags.as_slice(), &[false], "extension {name}");

            let retracting = PistonState {
                extending: false,
                should_head_be_rendered: true,
                ..extending
            };
            let mut retraction_flags = Vec::new();
            visit_piston_models(retracting, Vec3::ZERO, 1.0, |_, _, _, check_sides| {
                retraction_flags.push(check_sides)
            });
            assert_eq!(
                retraction_flags.as_slice(),
                &[true, true],
                "retraction {name}"
            );
        }
    }

    /// Drive the actual S24 handler, then prove the refreshed tile index exposes the carrier to
    /// the render snapshot. The source-order assertions pin the final live handoff: removing the
    /// `TileKind::Piston` arm or its lit piston snapshot makes this regression fail even if the
    /// world movement tests continue to pass.
    #[test]
    fn s24_piston_dispatch_reaches_the_live_tesr_snapshot() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let base = BlockPos::new(2, 64, 2);
        let push = base.east();
        let dest = push.east();
        {
            let world = &mut session.state_mut().world;
            world.load_empty_chunk(base.chunk_pos());
            world.set_block(
                base,
                BlockState::new(BlockId(mc_world::tileent::PISTON_ID), Facing::East.index()),
            );
            world.set_block(push, BlockState::new(BlockId(1), 0));
            world.set_block(dest, BlockState::AIR);
        }
        client.session = Some(session);

        client.react_to_block_action(&s2c::BlockAction {
            position: base,
            data1: 0,
            data2: Facing::East.index(),
            block_id: i32::from(mc_world::tileent::PISTON_ID),
        });

        let session = client.session().expect("live session");
        let world = &session.state().world;
        let carrier = world
            .tile_entity(dest)
            .and_then(mc_world::TileEntity::piston)
            .copied()
            .expect("S24 installs the carried tile entity at the destination");
        assert_eq!(carrier.piston_state, BlockState::new(BlockId(1), 0));
        assert!(carrier.extending);

        let mut index = TileIndex::new();
        index.refresh(world, Some(base));
        assert!(
            index
                .iter()
                .any(|(position, kind)| position == dest && kind == TileKind::Piston),
            "the refreshed render index must expose the moving carrier"
        );

        let source = include_str!("tiles.rs");
        let arm = source
            .find("TileKind::Piston => {")
            .expect("live piston draw arm");
        let call = source[arm..]
            .find("draw.piston_geometry")
            .map(|offset| arm + offset)
            .expect("live piston TESR call");
        let body_push = source[call..]
            .find("phase.push_blended(")
            .map(|offset| call + offset)
            .expect("piston geometry enters the tile phase");
        assert!(arm < call && call < body_push);
    }

    /// java: `handleBlockAction` -> `World.addBlockEvent` runs
    /// `Block.onBlockEventReceived` inline (only `WorldServer` queues block events), so a piston
    /// move lands in packet order and an S23 later in the same frame overwrites its moving block.
    /// Reacting to a frame's events only after its whole batch re-ran the piston on top of the
    /// S23 and left a moving block (and, two ticks later, a solid one) the server never sent.
    /// The live path is `Client::poll_in_place`, which now reacts after each message.
    #[test]
    fn an_s24_piston_move_then_an_s23_in_one_frame_apply_in_packet_order() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let base = BlockPos::new(2, 64, 2);
        let push = base.east();
        let dest = push.east();
        // A session whose world holds an east-facing piston with stone in front of it, and whose
        // inbound queue holds the S24 extend followed by an S23 clearing the stone's destination.
        let queued = || {
            let (handle, to_game, to_net) = NetHandle::new_pair();
            let mut session = Session::new(handle, JoinSettings::default());
            let world = &mut session.state_mut().world;
            world.load_empty_chunk(base.chunk_pos());
            world.set_block(
                base,
                BlockState::new(BlockId(mc_world::tileent::PISTON_ID), Facing::East.index()),
            );
            world.set_block(push, BlockState::new(BlockId(1), 0));
            for packet in [
                mc_protocol::play::S2CPlay::BlockAction(s2c::BlockAction {
                    position: base,
                    data1: 0,
                    data2: Facing::East.index(),
                    block_id: i32::from(mc_world::tileent::PISTON_ID),
                }),
                mc_protocol::play::S2CPlay::BlockChange(s2c::BlockChange {
                    position: dest,
                    block_state_id: 0,
                }),
            ] {
                to_game
                    .try_send(crate::net::ToGame::Packet(packet))
                    .expect("inbound queue");
            }
            (session, to_game, to_net)
        };

        // Negative control: the old shape, the whole batch applied first and its events reacted
        // to afterwards, replays the piston on top of the S23 and leaves a moving block there.
        let (mut session, _to_game, _to_net) = queued();
        let events = session.poll_all();
        client.session = Some(session);
        for event in &events {
            client.react(event);
        }
        let world = &client.session().expect("live session").state().world;
        assert_eq!(
            world.block_at(dest).block().0,
            mc_world::tileent::PISTON_EXTENSION_ID,
            "batch-then-react resurrects the moving block the S23 cleared"
        );

        let (session, _to_game, _to_net) = queued();
        client.session = Some(session);
        client.poll_in_place();

        let world = &client.session().expect("live session").state().world;
        assert_eq!(
            world.block_at(base).meta() & 8,
            8,
            "the S24 ran: the piston base is extended"
        );
        assert_eq!(
            world.block_at(dest),
            BlockState::AIR,
            "the S23 after it wins over the piston's moving block"
        );
        // The orphaned carrier finishes without resurrecting the block: `TileEntityPiston.update`
        // only writes back over a cell that is still `piston_extension`.
        for _ in 0..3 {
            client.tick_tile_entities();
        }
        let world = &client.session().expect("live session").state().world;
        assert_eq!(world.block_at(dest), BlockState::AIR);
    }

    /// `F-P0-network.play.server-09-5`: the S24 block-action path has no particle packet; the
    /// client must manufacture this one `NOTE` itself. This drives the request through the same
    /// `EntityNoteFX.Factory` used by `RenderGlobal.spawnParticle`, so the position, hue input,
    /// and resulting three-channel colour are one oracle rather than three disconnected checks.
    #[test]
    fn a_note_block_action_spawns_a_note_particle_with_the_event_hue() {
        let block = BlockPos::new(-4, 70, 9);
        let note = 23;
        let (pos, speed) = note_particle_parameters(block, note);
        assert_eq!(pos, DVec3::new(-3.5, 71.2, 9.5));
        assert_eq!(speed, DVec3::new(23.0 / 24.0, 0.0, 0.0));

        let world = World::new();
        let mut particles = ParticleSystem::with_seed(91);
        assert!(packet::spawn_one(
            &mut particles,
            &world,
            pos,
            packet::NOTE,
            pos,
            speed,
            &[],
            &crate::app::particles::AtlasIcons,
        ));
        assert_eq!(
            particles.len(),
            1,
            "World.spawnParticle's one NOTE call creates one EntityNoteFX"
        );
        let particle = particles
            .particles()
            .first()
            .expect("the NOTE factory spawned");
        assert_eq!(particle.pos, pos);
        assert!(matches!(particle.kind, ParticleKind::Note { .. }));

        let hue = 23.0_f32 / 24.0;
        let channel =
            |phase: f32| mc_math::sin((hue + phase) * std::f32::consts::PI * 2.0) * 0.65 + 0.35;
        let expected = [
            channel(0.0),
            channel(0.333_333_34),
            channel(0.666_666_7),
            1.0,
        ];
        for (actual, expected) in particle.color.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "EntityNoteFX colour must derive from data2 / 24: {actual} != {expected}"
            );
        }

        // Negative controls for the two deceptively plausible transcriptions: y + 0.5 puts the
        // note inside the block, and /12 makes its hue cycle twice as fast. Both differ at this
        // sample, so either mutation makes a preceding exact assertion fail.
        assert_ne!(pos.y, f64::from(block.y) + 0.5);
        assert_ne!(speed.x, f64::from(note) / 12.0);
    }

    /// The pure parameter oracle above proves the factory inputs; this drives the actual S24
    /// handler so a regression that computes the right constants but drops the call still bites.
    /// Java's client `RenderGlobal.playSound` is empty, so the action must add exactly one local
    /// NOTE particle and leave the audio request stream untouched.  The two packet/state
    /// controls at the end cover the dispatch seam as well: a foreign packet block is ignored,
    /// while a note packet still runs when the client's world state is stale.
    #[test]
    fn note_block_action_emits_one_particle_without_local_sound() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let position = BlockPos::new(0, 0, 0);
        {
            let world = &mut session.state_mut().world;
            world.load_empty_chunk(position.chunk_pos());
            world.set_block(position, BlockState::new(BlockId(25), 0));
        }
        client.session = Some(session);
        client.particles.clear();
        client.audio.clear_recent();

        let action = s2c::BlockAction {
            position,
            data1: 4,
            data2: 23,
            block_id: 25,
        };
        client.react_to_block_action(&action);
        assert_eq!(
            client.particles.len(),
            1,
            "one BlockNote spawnParticle call must create one NOTE particle"
        );
        {
            let particle = client
                .particles
                .particles()
                .first()
                .expect("the S24 action must reach the NOTE factory");
            assert_eq!(particle.pos, DVec3::new(0.5, 1.2, 0.5));
            assert!(matches!(particle.kind, ParticleKind::Note { .. }));
            let hue = 23.0_f32 / 24.0;
            let channel =
                |phase: f32| mc_math::sin((hue + phase) * std::f32::consts::PI * 2.0) * 0.65 + 0.35;
            for (actual, expected) in particle.color.into_iter().zip([
                channel(0.0),
                channel(0.333_333_34),
                channel(0.666_666_7),
                1.0,
            ]) {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "S24 data2 must reach EntityNoteFX as its hue: {actual} != {expected}"
                );
            }
        }
        assert!(
            client.audio.last_play().is_none(),
            "the client's RenderGlobal.playSound path is empty; S29 owns the sound"
        );

        // Negative control: a block action for a non-note packet block has no BlockNote handler,
        // even when the world currently contains a note.  This catches an unconditional local
        // particle spawn moved outside the packet-block dispatch arm.
        client.particles.clear();
        client.react_to_block_action(&s2c::BlockAction {
            block_id: 1,
            ..action
        });
        assert!(
            client.particles.is_empty(),
            "a foreign packet block must not manufacture a NOTE particle"
        );

        // Positive guard control: Java's client dispatch does not compare the packet block with
        // the stale state handed to the virtual handler.  The NOTE must still appear after the
        // local copy changes to stone before the action arrives.
        client
            .session_mut()
            .expect("session")
            .state_mut()
            .world
            .set_block(position, BlockState::new(BlockId(1), 0));
        client.react_to_block_action(&action);
        assert_eq!(
            client.particles.len(),
            1,
            "packet block dispatch must survive a stale world block"
        );
    }

    #[test]
    fn command_block_reader_matches_java_field_gates() {
        let previous = mc_world::tileent::CommandBlockState {
            command: "/old".to_owned(),
            custom_name: "OldName".to_owned(),
            success_count: 4,
            track_output: true,
            last_output: Some(r#"{"text":"old"}"#.to_owned()),
        };

        // Java's two direct getters reset when absent, while its optional fields retain the
        // existing value when the exact tag is absent.
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("Command", mc_nbt::Nbt::String("/new".to_owned()));
        nbt.insert("SuccessCount", mc_nbt::Nbt::Int(8));
        let state = read_command_block(&nbt, Some(&previous));
        assert_eq!(state.command, "/new");
        assert_eq!(state.success_count, 8);
        assert_eq!(state.custom_name, "OldName");
        assert!(state.track_output);
        assert_eq!(state.last_output.as_deref(), Some(r#"{"text":"old"}"#));

        // Wrong tags follow NBTTagCompound.hasKey(key, type): Command/SuccessCount use getter
        // defaults, but CustomName/TrackOutput/LastOutput do not enter their guarded branches.
        nbt.insert("Command", mc_nbt::Nbt::Int(9));
        nbt.insert("SuccessCount", mc_nbt::Nbt::String("not an int".to_owned()));
        nbt.insert("CustomName", mc_nbt::Nbt::Int(10));
        nbt.insert("TrackOutput", mc_nbt::Nbt::Int(0));
        nbt.insert("LastOutput", mc_nbt::Nbt::Int(11));
        let malformed = read_command_block(&nbt, Some(&previous));
        assert_eq!(malformed.command, "");
        assert_eq!(malformed.success_count, 0);
        assert_eq!(malformed.custom_name, "OldName");
        assert!(malformed.track_output);
        assert_eq!(malformed.last_output.as_deref(), Some(r#"{"text":"old"}"#));

        // LastOutput is gated by the resulting TrackOutput value, so disabling output does not
        // clear an already-read last message (the Java reader leaves the field untouched).
        let mut disabled_nbt = mc_nbt::NbtCompound::new();
        disabled_nbt.insert("TrackOutput", mc_nbt::Nbt::Byte(0));
        disabled_nbt.insert(
            "LastOutput",
            mc_nbt::Nbt::String(r#"{"text":"ignored"}"#.to_owned()),
        );
        let disabled = read_command_block(&disabled_nbt, Some(&previous));
        assert!(!disabled.track_output);
        assert_eq!(disabled.last_output.as_deref(), Some(r#"{"text":"old"}"#));

        // Once tracking is enabled, a string LastOutput replaces the previous component JSON.
        disabled_nbt.insert("TrackOutput", mc_nbt::Nbt::Byte(1));
        let enabled = read_command_block(&disabled_nbt, Some(&previous));
        assert!(enabled.track_output);
        assert_eq!(
            enabled.last_output.as_deref(),
            Some(r#"{"text":"ignored"}"#)
        );
    }

    /// A client over the actual 1.8.9 sound registry, or `None` in a source-only checkout.
    ///
    /// This deliberately does not use a synthetic `AudioEngine`: the two negative assertions
    /// below need to prove that a *real, playable* chest sound was not accidentally funneled
    /// into it, rather than merely observing an engine with no registered sounds.
    fn real_client() -> Option<(tempfile::TempDir, Client)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config
            .version_jar
            .as_ref()
            .is_some_and(|path| path.is_file())
        {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((dir, client))
    }

    #[test]
    fn remote_player_snapshot_drives_the_live_enchanting_tick_and_filters_spectators() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let table = BlockPos::new(0, 64, 0);
        let remote_uuid = uuid::Uuid::from_u128(0x0E11_CA17);
        {
            let state = session.state_mut();
            state.world.load_empty_chunk(table.chunk_pos());
            state
                .world
                .set_block(table, BlockState::new(BlockId(116), 0));
            state.world.tile_entity_or_create(table);
            state.entities.spawn(
                &mut state.world,
                41,
                crate::sync::TrackedKind::Player,
                Some(remote_uuid),
                [32, 64 * 32, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::entity::Metadata::new(),
            );
        }
        client.session = Some(session);

        assert!(
            client.session().unwrap().predictor().is_none(),
            "negative control: this fixture has no local player candidate"
        );
        client.tick_tile_entities();
        let spread = client
            .session()
            .unwrap()
            .state()
            .world
            .tile_entity(table)
            .and_then(mc_world::TileEntity::enchant)
            .unwrap()
            .book_spread;
        assert!(
            spread > 0.0,
            "a remote player alone must open the live book"
        );

        client
            .session_mut()
            .unwrap()
            .state_mut()
            .players
            .apply(&s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
                uuid: remote_uuid,
                username: "spectator".to_owned(),
                properties: Vec::new(),
                game_mode: s2c::GameMode::Spectator,
                ping: 0,
                display_name: None,
            }]))
            .expect("valid tab row");
        client.tick_tile_entities();
        let closed = client
            .session()
            .unwrap()
            .state()
            .world
            .tile_entity(table)
            .and_then(mc_world::TileEntity::enchant)
            .unwrap()
            .book_spread;
        assert_eq!(closed, 0.0, "the same remote spectator must be excluded");
    }

    #[test]
    fn active_spawner_submits_one_colocated_pair_before_advancing_animation() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let spawner_pos = BlockPos::new(0, 0, 0);
        {
            let state = session.state_mut();
            state.world.load_empty_chunk(spawner_pos.chunk_pos());
            state
                .world
                .set_block(spawner_pos, BlockState::new(BlockId(52), 0));
            let entity = state
                .world
                .tile_entity_or_create(spawner_pos)
                .expect("the spawner block owns its animation state");
            let mc_world::TileData::Spawner(spawner) = &mut entity.data else {
                panic!("the spawner block owns SpawnerState");
            };
            spawner.spawn_delay = 2;
            state.entities.spawn(
                &mut state.world,
                42,
                crate::sync::TrackedKind::Player,
                Some(uuid::Uuid::from_u128(0x05A0_A1E2)),
                [32, 0, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                mc_world::entity::Metadata::new(),
            );
        }
        client.session = Some(session);
        client.particles.clear();

        client.tick_tile_entities();

        let particles = client.particles.particles();
        assert_eq!(particles.len(), 2, "one active tick emits smoke plus flame");
        assert_eq!(
            particles[0].prev_pos, particles[1].prev_pos,
            "both factories must receive the same three Java RNG offsets"
        );
        assert!(particles
            .iter()
            .any(|particle| matches!(particle.kind, ParticleKind::Smoke { .. })));
        assert!(particles
            .iter()
            .any(|particle| matches!(particle.kind, ParticleKind::Flame { .. })));
        let spawner = client
            .session()
            .expect("session")
            .state()
            .world
            .tile_entity(spawner_pos)
            .and_then(mc_world::TileEntity::spawner)
            .expect("spawner remains live after its client tick");
        assert_eq!(spawner.spawn_delay, 1);
        assert_ne!(spawner.mob_rotation, 0.0);
        assert_eq!(
            client.tile_index.spawner_animation(spawner_pos, "Pig"),
            SpawnerRenderAnimation::default(),
            "the cached miniature is not a world entity and remains at constructor pose"
        );

        // The entire branch is under `isActivated()`: after the only player becomes a
        // spectator, neither factory runs and the countdown freezes.
        let remote_uuid = uuid::Uuid::from_u128(0x05A0_A1E2);
        client
            .session_mut()
            .expect("session")
            .state_mut()
            .players
            .apply(&s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
                uuid: remote_uuid,
                username: "spectator".to_owned(),
                properties: Vec::new(),
                game_mode: s2c::GameMode::Spectator,
                ping: 0,
                display_name: None,
            }]))
            .expect("valid tab row");
        client.particles.clear();
        client.tick_tile_entities();
        assert!(client.particles.is_empty());
        assert_eq!(
            client
                .session()
                .expect("session")
                .state()
                .world
                .tile_entity(spawner_pos)
                .and_then(mc_world::TileEntity::spawner)
                .expect("spawner")
                .spawn_delay,
            1
        );
        assert_eq!(
            client.tile_index.spawner_animation(spawner_pos, "Pig"),
            SpawnerRenderAnimation::default(),
            "spectator-only/out-of-range ticks leave the constructor pose unchanged"
        );
    }

    #[test]
    fn command_block_s35_updates_live_typed_state_and_rejects_wrong_targets() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let command_pos = BlockPos::new(0, 0, 0);
        let stone_pos = BlockPos::new(1, 0, 0);
        {
            let world = &mut session.state_mut().world;
            world.load_empty_chunk(command_pos.chunk_pos());
            world.set_block(command_pos, BlockState::new(BlockId(137), 0));
            world.set_block(stone_pos, BlockState::new(BlockId(1), 0));
        }
        client.session = Some(session);

        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("Command", mc_nbt::Nbt::String("/say hello".to_owned()));
        nbt.insert("CustomName", mc_nbt::Nbt::String("Builder".to_owned()));
        nbt.insert("SuccessCount", mc_nbt::Nbt::Int(7));
        nbt.insert("TrackOutput", mc_nbt::Nbt::Byte(1));
        nbt.insert(
            "LastOutput",
            mc_nbt::Nbt::String(r#"{"text":"ok"}"#.to_owned()),
        );
        client.react_to_tile_entity_update(command_pos, 2, Some(&nbt));

        let expected = mc_world::tileent::CommandBlockState {
            command: "/say hello".to_owned(),
            custom_name: "Builder".to_owned(),
            success_count: 7,
            track_output: true,
            last_output: Some(r#"{"text":"ok"}"#.to_owned()),
        };
        assert_eq!(
            client
                .session()
                .expect("session")
                .state()
                .world
                .tile_entity(command_pos)
                .and_then(mc_world::TileEntity::command_block),
            Some(&expected)
        );

        // Optional fields omitted from a later read stay as they were in Java; this also proves
        // the live reader seeds its typed decode from the existing tile rather than resetting it.
        let mut partial = mc_nbt::NbtCompound::new();
        partial.insert("Command", mc_nbt::Nbt::String("/say again".to_owned()));
        partial.insert("SuccessCount", mc_nbt::Nbt::Int(8));
        client.react_to_tile_entity_update(command_pos, 2, Some(&partial));
        let after_partial = client
            .session()
            .expect("session")
            .state()
            .world
            .tile_entity(command_pos)
            .and_then(mc_world::TileEntity::command_block)
            .expect("command block state");
        assert_eq!(after_partial.command, "/say again");
        assert_eq!(after_partial.success_count, 8);
        assert_eq!(after_partial.custom_name, "Builder");
        assert!(after_partial.track_output);
        assert_eq!(
            after_partial.last_output.as_deref(),
            Some(r#"{"text":"ok"}"#)
        );

        // A wrong action is rejected before replacement and therefore cannot partially mutate a
        // previously decoded command block.
        let mut wrong_action = mc_nbt::NbtCompound::new();
        wrong_action.insert("Command", mc_nbt::Nbt::String("/say wrong".to_owned()));
        wrong_action.insert("SuccessCount", mc_nbt::Nbt::Int(99));
        client.react_to_tile_entity_update(command_pos, 3, Some(&wrong_action));
        let after_wrong_action = client
            .session()
            .expect("session")
            .state()
            .world
            .tile_entity(command_pos)
            .and_then(mc_world::TileEntity::command_block)
            .expect("command block state");
        assert_eq!(after_wrong_action.command, "/say again");
        assert_eq!(after_wrong_action.success_count, 8);

        // Action 2 on a non-command block is the other half of Java's action/type conjunction.
        client.react_to_tile_entity_update(stone_pos, 2, Some(&wrong_action));
        assert!(client
            .session()
            .expect("session")
            .state()
            .world
            .tile_entity(stone_pos)
            .is_none());
    }

    #[test]
    fn banner_s35_action_saves_design_and_only_described_state_can_render() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let banner_pos = BlockPos::new(0, 64, 0);
        let wall_banner_pos = BlockPos::new(1, 64, 0);
        let stone_pos = BlockPos::new(2, 64, 0);
        {
            let world = &mut session.state_mut().world;
            world.load_empty_chunk(banner_pos.chunk_pos());
            world.set_block(banner_pos, BlockState::new(BlockId(176), 3));
            world.set_block(wall_banner_pos, BlockState::new(BlockId(177), 4));
            world.set_block(stone_pos, BlockState::new(BlockId(1), 0));
        }
        client.session = Some(session);

        // Java's S35 action 6 reaches TileEntityBanner.readFromNBT.  Use numeric-but-not-Int
        // values and malformed pattern entries so both the NBT coercion and EnumBannerPattern
        // type/id gates are exercised on the live packet-to-store path.
        let mut valid = NbtCompound::new();
        valid.insert("Pattern", Nbt::String("cr".to_owned()));
        valid.insert("Color", Nbt::Short(1));
        let mut unknown = NbtCompound::new();
        unknown.insert("Pattern", Nbt::String("not_real".to_owned()));
        unknown.insert("Color", Nbt::Int(2));
        let mut nbt = NbtCompound::new();
        nbt.insert("x", Nbt::Int(41));
        nbt.insert("y", Nbt::Short(72));
        nbt.insert("z", Nbt::Long(-37));
        nbt.insert("Base", Nbt::Byte(14));
        nbt.insert(
            "Patterns",
            Nbt::list(vec![
                Nbt::Compound(valid),
                Nbt::Compound(unknown),
                Nbt::String("not a compound".to_owned()),
            ]),
        );
        // A wrong action never reaches TileEntity.readFromNBT, so it cannot overwrite the
        // header-keyed instance's effective render position.
        client.react_to_tile_entity_update(banner_pos, 5, Some(&nbt));
        assert_eq!(client.tile_index.render_position(banner_pos), banner_pos);
        client.react_to_tile_entity_update(banner_pos, 6, Some(&nbt));
        assert_eq!(
            client.tile_index.render_position(banner_pos),
            BlockPos::new(41, 72, -37)
        );

        let world = &client.session().expect("session").state().world;
        let stored = world
            .tile_entity(banner_pos)
            .and_then(mc_world::TileEntity::banner)
            .expect("S35 action 6 stores the banner payload");
        assert!(stored.initialized);
        assert_eq!(stored.base_color, 14);
        assert_eq!(stored.patterns, vec![("cr".to_owned(), 1)]);
        assert!(banner_state_for_render(world.tile_entity(banner_pos)).is_some());
        assert!(
            banner_state_for_render(world.tile_entity(wall_banner_pos)).is_none(),
            "an undescribed wall banner remains invisible too"
        );

        // Missing payloads are ignored by the packet handler, and a different action is rejected
        // by the same six-way action/type conjunction as vanilla.  Neither may clear the design.
        client.react_to_tile_entity_update(banner_pos, 6, None);
        client.react_to_tile_entity_update(banner_pos, 5, Some(&NbtCompound::new()));
        let world = &client.session().expect("session").state().world;
        let retained = world
            .tile_entity(banner_pos)
            .and_then(mc_world::TileEntity::banner)
            .expect("rejected descriptions preserve the previous banner");
        assert_eq!(retained.base_color, 14);
        assert_eq!(retained.patterns, vec![("cr".to_owned(), 1)]);

        // A matching action on a non-banner block cannot manufacture a tile entity.
        client.react_to_tile_entity_update(stone_pos, 6, Some(&nbt));
        assert!(client
            .session()
            .expect("session")
            .state()
            .world
            .tile_entity(stone_pos)
            .is_none());
    }

    #[test]
    fn flower_pot_s35_updates_actual_model_state_and_rejects_wrong_actions() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let pot_pos = BlockPos::new(2, 64, 2);
        let stone_pos = BlockPos::new(3, 64, 2);
        {
            let world = &mut session.state_mut().world;
            world.load_empty_chunk(pot_pos.chunk_pos());
            world.set_block(pot_pos, BlockState::new(BlockId(140), 0));
            world.set_block(stone_pos, BlockState::new(BlockId(1), 0));
            world.drain_dirty_sections();
        }
        client.session = Some(session);

        // java: `TileEntityFlowerPot.getDescriptionPacket` rewrites Item to an integer, and
        // action 5 is the only S35 arm that accepts a flower pot. This is the wire-positive
        // control for the terrain path, not just a direct store insertion.
        let mut fern_nbt = mc_nbt::NbtCompound::new();
        fern_nbt.insert("Item", mc_nbt::Nbt::Int(31));
        fern_nbt.insert("Data", mc_nbt::Nbt::Int(2));
        client.react_to_tile_entity_update(pot_pos, 5, Some(&fern_nbt));
        {
            let world = &client.session().expect("session").state().world;
            let pot = world
                .tile_entity(pot_pos)
                .and_then(mc_world::TileEntity::flower_pot)
                .expect("S35 action 5 stores the flower-pot payload");
            assert_eq!((pot.item_id, pot.data), (31, 2));
            assert_eq!(pot.contents(), "fern");

            // This is the same call the terrain mesher makes before model lookup. The stored
            // metadata says nothing about the plant; the tile payload must supply `contents`.
            let actual =
                mc_render::terrain::mesher::actual_state(world, world.block_at(pot_pos), pot_pos);
            assert_eq!(actual.get("contents"), Some("fern"));
            let mapping = mc_assets::model::state_mapper::resolve_actual(&actual);
            assert_eq!(mapping.file_stem, "flower_pot");
            assert_eq!(mapping.variant_key, "contents=fern");
            assert!(
                world.dirty_sections().next().is_some(),
                "changing contents invalidates the terrain section"
            );
        }

        // Negative control: action 4 belongs to a skull, so vanilla's instanceof/action gate
        // drops this plausible cactus payload and does not dirty or replace the fern.
        client
            .session_mut()
            .expect("session")
            .state_mut()
            .world
            .drain_dirty_sections();
        let mut cactus_nbt = mc_nbt::NbtCompound::new();
        cactus_nbt.insert("Item", mc_nbt::Nbt::Int(81));
        cactus_nbt.insert("Data", mc_nbt::Nbt::Int(0));
        client.react_to_tile_entity_update(pot_pos, 4, Some(&cactus_nbt));
        {
            let world = &client.session().expect("session").state().world;
            let pot = world
                .tile_entity(pot_pos)
                .and_then(mc_world::TileEntity::flower_pot)
                .expect("wrong action cannot remove the existing pot");
            assert_eq!((pot.item_id, pot.data), (31, 2));
            assert!(
                world.dirty_sections().next().is_none(),
                "a rejected S35 does not invalidate terrain"
            );
        }

        // A matching action on a non-provider block is the other type-check negative: the
        // packet cannot manufacture a flower pot where the world has stone.
        client.react_to_tile_entity_update(stone_pos, 5, Some(&fern_nbt));
        assert!(client
            .session()
            .expect("session")
            .state()
            .world
            .tile_entity(stone_pos)
            .is_none());
    }

    #[test]
    fn chunk_loaded_flower_pot_is_lazily_created_before_actual_state_rendering() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let pos = BlockPos::new(6, 64, 6);
        {
            let world = &mut session.state_mut().world;
            world.load_empty_chunk(pos.chunk_pos());
            // Legacy metadata 11 is fern. Removing the inline entity models a full chunk packet,
            // whose block arrays arrive before the follow-up S35 description packets.
            world.set_block(pos, BlockState::new(BlockId(140), 11));
            world.remove_tile_entity(pos);
            world.drain_dirty_sections();
        }
        client.session = Some(session);

        // `Chunk.getTileEntity(pos, IMMEDIATE)` performs this create when the render path asks;
        // the client tick is the bounded equivalent because the terrain mesher intentionally
        // reads the immutable store. Without FlowerPot in `pending`, this stays empty forever.
        client.tick_tile_entities();
        let world = &client.session().expect("session").state().world;
        let pot = world
            .tile_entity(pos)
            .and_then(mc_world::TileEntity::flower_pot)
            .expect("a loaded pot gets its legacy tile before model lookup");
        assert_eq!((pot.item_id, pot.data), (31, 2));
        let actual = mc_render::terrain::mesher::actual_state(world, world.block_at(pos), pos);
        assert_eq!(actual.get("contents"), Some("fern"));
        assert!(
            world.dirty_sections().next().is_some(),
            "first lazy pot creation requests a remesh"
        );

        // Negative control: the default stored state is not trusted as a plant identity; it is
        // only the legacy block metadata that supplies the fern through the created tile.
        assert_ne!(
            mc_assets::model::state_mapper::resolve_actual(&actual).variant_key,
            "contents=cactus"
        );
    }

    /// `F-P1-block-06-2` + `F-P1-block-23-1`: client-side `World.playSoundEffect` reaches
    /// RenderGlobal's empty `playSound` method. The server's matching invocation becomes S29,
    /// so locally playing either path would make the player hear it twice.
    ///
    /// The chest transition and note action are both deliberately driven far enough to produce
    /// their respective world-side effects. If `tick_tile_entities` starts forwarding its
    /// returned `TileEntitySound`, or the note arm starts calling `play_sound`, either negative
    /// assertion bites. The final S29 is a positive control: silence here must mean "defer to
    /// the server packet", never "the client lost its audio funnel".
    #[test]
    fn chest_and_note_world_sounds_wait_for_the_servers_single_s29() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let chest = BlockPos::new(2, 64, 3);
        let note = BlockPos::new(-2, 64, 3);
        let world = &mut session.state_mut().world;
        world.load_empty_chunk(chest.chunk_pos());
        world.load_empty_chunk(note.chunk_pos());
        // Chest metadata 0 is not a legal 1.8.9 facing state and therefore reads back as air.
        // Use north (2), as a real S21 column would.
        world.set_block(chest, BlockState::new(BlockId(54), 2));
        world.set_block(note, BlockState::new(BlockId(25), 0));
        assert!(matches!(
            world.apply_block_action(chest, 54, 1, 1),
            Some(mc_world::BlockActionEffect::ChestUsers { players: 1 })
        ));
        client.session = Some(session);

        // This is the `TileEntityChest.update` branch that creates `random.chestopen` in the
        // world layer. It must update the lid but must *not* hand a local request to AudioEngine.
        client.audio.clear_recent();
        client.tick_tile_entities();
        assert!(
            client.audio.last_play().is_none(),
            "the client-side chest World.playSoundEffect path is RenderGlobal's empty method"
        );

        // `BlockNote.onBlockEventReceived` likewise creates the NOTE particle locally but has
        // no local audio output. The event is intentionally received at a real note block so
        // this is not a no-op or unknown-block false positive.
        client.react_to_block_action(&s2c::BlockAction {
            position: note,
            data1: 0,
            data2: 12,
            block_id: 25,
        });
        assert!(
            client.audio.last_play().is_none(),
            "the note particle is client-local, but its sound waits for the server S29"
        );

        // Positive control: the actual S29 for the chest still reaches the one audio funnel.
        client.react_to_sound_effect(&s2c::SoundEffect {
            sound_name: "random.chestopen".to_owned(),
            x_fixed: chest.x * 8,
            y_fixed: chest.y * 8,
            z_fixed: chest.z * 8,
            volume: 0.5,
            pitch_quantized: 63,
        });
        let played = client
            .audio
            .last_play()
            .expect("the server S29 must supply the one audible chest sound");
        assert_eq!(played.request.event.path, "random.chestopen");
    }

    /// The index has to find every tile-entity block in a column and drop a column that
    /// unloads — the two failure modes are an invisible chest and a leak across a long session.
    #[test]
    fn the_index_finds_tile_entity_blocks_and_follows_the_loaded_set() {
        let mut world = World::new();
        let column = ChunkPos::new(0, 0);
        world.load_empty_chunk(column);
        // A chest, a wall sign and a beacon, spread across three sections so the scan cannot
        // pass by only ever looking at one.
        let chest = BlockPos::new(3, 20, 5);
        let sign = BlockPos::new(9, 70, 11);
        let beacon = BlockPos::new(1, 130, 2);
        world.set_block(chest, BlockState::new(BlockId(54), 3));
        world.set_block(sign, BlockState::new(BlockId(68), 2));
        world.set_block(beacon, BlockState::new(BlockId(138), 0));
        // ... and one block that is not a tile entity at all.
        world.set_block(BlockPos::new(4, 20, 5), BlockState::new(BlockId(1), 0));

        let mut index = TileIndex::new();
        index.refresh(&world, Some(chest));
        let mut found: Vec<(BlockPos, TileKind)> = index.iter().collect();
        found.sort_by_key(|(p, _)| p.y);
        assert_eq!(
            found,
            vec![
                (chest, TileKind::Chest),
                (sign, TileKind::WallSign),
                (beacon, TileKind::Beacon),
            ]
        );
        assert_eq!(index.column_count(), 1);
        assert_eq!(index.tile_count(), 3);

        // Breaking the chest removes it on the next refresh — the player's own 3x3 is re-scanned
        // every tick precisely so a block they just changed is never stale on screen.
        world.set_block(chest, BlockState::AIR);
        index.refresh(&world, Some(chest));
        assert_eq!(index.tile_count(), 2);

        world.unload_chunk(column);
        index.refresh(&world, Some(chest));
        assert_eq!(index.tile_count(), 0);
        assert_eq!(index.column_count(), 0);
    }

    /// S21/S26 can load a column anywhere in the `int` range. The fresh-column
    /// key squared an `i32` distance: a debug-build panic for any column ~46k chunks from the
    /// player (found by the packet fuzz) and a scrambled order in release. Now exact.
    #[test]
    fn far_columns_rank_after_near_ones_in_the_fresh_index_order() {
        let mut world = World::new();
        // Eight columns two to three chunks out (outside the player's always-scanned 3x3), plus
        // three far ones; the old key overflowed on all three.
        let near: Vec<ChunkPos> = [
            (2, 0),
            (-2, 0),
            (0, 2),
            (0, -2),
            (2, 2),
            (-2, -2),
            (3, 0),
            (0, 3),
        ]
        .into_iter()
        .map(|(x, z)| ChunkPos::new(x, z))
        .collect();
        assert_eq!(near.len(), INDEX_BUDGET);
        let far = [
            ChunkPos::new(i32::MIN + 1, 0),
            ChunkPos::new(50_000, -50_000),
            ChunkPos::new(i32::MAX, i32::MIN),
        ];
        for &pos in near.iter().chain(&far) {
            world.load_empty_chunk(pos);
        }
        let mut index = TileIndex::new();
        index.refresh(&world, Some(BlockPos::new(8, 64, 8)));
        // The fresh budget goes to the nearest columns; only the rolling cursor (one column of
        // any distance per refresh) may have reached a far one.
        assert!(near.iter().all(|pos| index.columns.contains_key(pos)));
        assert!(
            far.iter()
                .filter(|pos| index.columns.contains_key(pos))
                .count()
                <= 1
        );
    }

    #[test]
    fn tile_entity_payload_position_matches_java_numeric_and_zero_defaults() {
        let header = BlockPos::new(7, 64, 9);
        let mut nbt = NbtCompound::new();
        nbt.insert("x", Nbt::Short(41));
        nbt.insert("y", Nbt::String("wrong type".to_owned()));
        nbt.insert("z", Nbt::Long(-37));
        assert_eq!(
            tile_entity_payload_position(&nbt),
            BlockPos::new(41, 0, -37)
        );

        let mut index = TileIndex::new();
        assert_eq!(index.render_position(header), header);
        let out_of_chunk = BlockPos::new(1000, 200, -1000);
        index.set_render_position(header, out_of_chunk);
        assert_eq!(
            index.render_position(header),
            out_of_chunk,
            "the payload position survives even when it is outside the header chunk"
        );

        // A header-only mutant fails the first assertion above; retaining a header after an
        // incomplete NBT payload would fail this Java `getInteger` zero-default control.
        let missing = NbtCompound::new();
        assert_eq!(
            tile_entity_payload_position(&missing),
            BlockPos::ORIGIN,
            "missing x/y/z are Java's zero-valued getters"
        );
        index.set_render_position(header, tile_entity_payload_position(&missing));
        assert_eq!(index.render_position(header), BlockPos::ORIGIN);
        index.clear();
        assert_eq!(index.render_position(header), header);
    }

    /// java: `TileEntityChestRenderer`'s constructor reads `Calendar` once. The date arithmetic
    /// is the only part worth pinning; `is_christmas` itself is tested in `mc-render`.
    #[test]
    fn the_civil_calendar_conversion_lands_on_the_right_day() {
        // 1970-01-01, 2000-03-01 (a leap-year boundary), and 2024-12-26.
        assert_eq!(civil_from_days(0), (1, 1));
        assert_eq!(civil_from_days(11017), (3, 1));
        assert_eq!(civil_from_days(20083), (12, 26));
        assert!(mc_render::entity::tile::is_christmas(
            civil_from_days(20083).0,
            civil_from_days(20083).1
        ));
    }

    #[test]
    fn end_portal_clock_preserves_absolute_milliseconds_and_wraps_only_in_render_math() {
        let absolute = 4_200_000_000_u128 + 1_234;
        assert_eq!(
            end_portal_system_time_millis_with(|| absolute),
            absolute as u64,
            "the injected clock must not be rebased to a process-local Instant"
        );
        let scroll = mc_render::entity::tilepass::end_portal_scroll(absolute as u64);
        assert!((scroll - 1234.0 / 700_000.0).abs() < f32::EPSILON);
        assert_eq!(
            end_portal_system_time_millis_with(|| u128::MAX),
            u64::MAX,
            "a platform clock wider than u64 must saturate instead of wrapping"
        );
    }

    #[test]
    fn base64_round_trips_a_known_profile_blob() {
        // The exact shape a `SkullOwner` property carries.
        let json =
            r#"{"textures":{"SKIN":{"url":"http:\/\/textures.minecraft.net\/texture/abc"}}}"#;
        let encoded = encode(json.as_bytes());
        let decoded = base64_decode(&encoded).expect("decodes");
        assert_eq!(String::from_utf8(decoded).unwrap(), json);
    }

    #[test]
    fn a_profile_texture_property_yields_a_skin_url() {
        let json = r#"{"timestamp":1,"textures":{"SKIN":{"url":"http:\/\/textures.minecraft.net\/texture\/abc"}}}"#;
        let mut properties = mc_nbt::NbtCompound::new();
        let mut texture = mc_nbt::NbtCompound::new();
        texture.insert("Value", mc_nbt::Nbt::String(encode(json.as_bytes())));
        texture.insert(
            "Signature",
            mc_nbt::Nbt::String("skin-signature".to_owned()),
        );
        properties.insert(
            "textures",
            mc_nbt::Nbt::list(vec![mc_nbt::Nbt::Compound(texture)]),
        );
        let mut custom = mc_nbt::NbtCompound::new();
        custom.insert("Value", mc_nbt::Nbt::String("custom-value".to_owned()));
        properties.insert(
            "custom",
            mc_nbt::Nbt::list(vec![mc_nbt::Nbt::Compound(custom)]),
        );
        let mut owner = mc_nbt::NbtCompound::new();
        owner.insert(
            "Id",
            mc_nbt::Nbt::String("123e4567-e89b-12d3-a456-426614174000".to_owned()),
        );
        owner.insert("Name", mc_nbt::Nbt::String("Notch".to_owned()));
        owner.insert("Properties", mc_nbt::Nbt::Compound(properties));
        let profile = mc_auth::read_game_profile_from_nbt(&owner).expect("complete profile");
        assert_eq!(
            skin_url_from_profile(&profile).as_deref(),
            Some("http://textures.minecraft.net/texture/abc")
        );
        let skull = {
            let mut nbt = mc_nbt::NbtCompound::new();
            nbt.insert("SkullType", mc_nbt::Nbt::Byte(3));
            nbt.insert("Rot", mc_nbt::Nbt::Byte(5));
            nbt.insert("Owner", mc_nbt::Nbt::Compound(owner));
            read_skull(&nbt)
        };
        assert_eq!(skull.skull_type, 3);
        assert_eq!(skull.rotation, 5);
        assert_eq!(skull.owner_name.as_deref(), Some("Notch"));
        assert_eq!(
            skull.skin_url.as_deref(),
            Some("http://textures.minecraft.net/texture/abc")
        );
        let profile = skull
            .owner_profile
            .expect("skull retains its complete profile");
        assert_eq!(
            profile.id.unwrap().to_string(),
            "123e4567-e89b-12d3-a456-426614174000"
        );
        assert_eq!(profile.properties.len(), 2);
        assert_eq!(
            profile.properties[0].signature.as_deref(),
            Some("skin-signature")
        );
        assert_eq!(profile.properties[1].name, "custom");
    }

    #[test]
    fn a_profile_texture_property_selects_skin_not_the_first_cape_url() {
        // Authlib indexes the decoded `textures` map by `SKIN`; scanning for the first `url`
        // would incorrectly bind the CAPE when the proxy serializes that entry first.
        let json = r#"{"textures":{
            "CAPE":{"url":"https://textures.minecraft.net/texture/cape-first"},
            "SKIN":{"url":"https://textures.minecraft.net/texture/actual-skin"}}}"#;
        let mut properties = mc_nbt::NbtCompound::new();
        let mut texture = mc_nbt::NbtCompound::new();
        texture.insert("Value", mc_nbt::Nbt::String(encode(json.as_bytes())));
        properties.insert(
            "textures",
            mc_nbt::Nbt::list(vec![mc_nbt::Nbt::Compound(texture)]),
        );
        let mut owner = mc_nbt::NbtCompound::new();
        owner.insert(
            "Id",
            mc_nbt::Nbt::String("123e4567-e89b-12d3-a456-426614174000".to_owned()),
        );
        owner.insert("Name", mc_nbt::Nbt::String("Notch".to_owned()));
        owner.insert("Properties", mc_nbt::Nbt::Compound(properties));

        let profile = mc_auth::read_game_profile_from_nbt(&owner).expect("complete profile");
        assert_eq!(
            skin_url_from_profile(&profile).as_deref(),
            Some("https://textures.minecraft.net/texture/actual-skin")
        );

        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("SkullType", mc_nbt::Nbt::Byte(3));
        nbt.insert("Owner", mc_nbt::Nbt::Compound(owner));
        let skull = read_skull(&nbt);
        assert_eq!(
            skull.skin_url.as_deref(),
            Some("https://textures.minecraft.net/texture/actual-skin")
        );
        assert!(skull.owner_profile.is_some());
    }

    #[test]
    fn a_profile_without_skin_or_with_malformed_textures_keeps_safe_fallback_state() {
        let read = |value: &str| {
            let mut properties = mc_nbt::NbtCompound::new();
            let mut texture = mc_nbt::NbtCompound::new();
            texture.insert("Value", mc_nbt::Nbt::String(value.to_owned()));
            properties.insert(
                "textures",
                mc_nbt::Nbt::list(vec![mc_nbt::Nbt::Compound(texture)]),
            );
            let mut owner = mc_nbt::NbtCompound::new();
            owner.insert(
                "Id",
                mc_nbt::Nbt::String("123e4567-e89b-12d3-a456-426614174000".to_owned()),
            );
            owner.insert("Name", mc_nbt::Nbt::String("Notch".to_owned()));
            owner.insert("Properties", mc_nbt::Nbt::Compound(properties));
            let mut nbt = mc_nbt::NbtCompound::new();
            nbt.insert("SkullType", mc_nbt::Nbt::Byte(3));
            nbt.insert("Owner", mc_nbt::Nbt::Compound(owner));
            read_skull(&nbt)
        };

        let cape_only = encode(
            br#"{"textures":{"CAPE":{"url":"https://textures.minecraft.net/texture/cape"}}}"#,
        );
        let no_skin = read(&cape_only);
        // A valid profile with no SKIN has no URL, allowing the live consumer's Alex/Steve
        // fallback rather than accidentally using the cape as a head texture.
        assert!(no_skin.owner_profile.is_some());
        assert_eq!(no_skin.skin_url, None);

        let malformed = read("not-base64");
        assert!(malformed.owner_profile.is_some());
        assert_eq!(malformed.skin_url, None);
    }

    #[test]
    fn a_malformed_owner_id_keeps_the_name_only_profile() {
        let mut owner = mc_nbt::NbtCompound::new();
        owner.insert("Id", mc_nbt::Nbt::String("not-a-uuid".to_owned()));
        owner.insert("Name", mc_nbt::Nbt::String("Notch".to_owned()));
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("SkullType", mc_nbt::Nbt::Byte(3));
        nbt.insert("Owner", mc_nbt::Nbt::Compound(owner));

        let skull = read_skull(&nbt);
        assert_eq!(skull.owner_name.as_deref(), Some("Notch"));
        assert_eq!(
            skull.owner_profile.as_ref().and_then(|profile| profile.id),
            None
        );
        assert_eq!(skull.skin_url, None);
    }

    #[test]
    fn legacy_extra_type_keeps_java_name_only_profile_and_id_only_name_is_null() {
        // java: TileEntitySkull.readFromNBT creates GameProfile(null, ExtraType) and
        // updatePlayerProfile leaves it intact on the multiplayer client. The later write path
        // therefore emits Owner{Name}, not a state with only a display-name side channel.
        let mut legacy = mc_nbt::NbtCompound::new();
        legacy.insert("SkullType", mc_nbt::Nbt::Byte(3));
        legacy.insert("ExtraType", mc_nbt::Nbt::String("LegacyOwner".to_owned()));
        let skull = read_skull(&legacy);
        assert_eq!(
            skull.owner_profile,
            Some(mc_auth::GameProfile {
                id: None,
                name: "LegacyOwner".to_owned(),
                properties: Vec::new(),
            })
        );
        assert_eq!(skull.owner_name.as_deref(), Some("LegacyOwner"));

        // java: NBTUtil.readGameProfileFromNBT accepts an Id-only profile, whose Name remains
        // null; do not turn that null into Some("") in the renderer-facing snapshot.
        let mut id_only_owner = mc_nbt::NbtCompound::new();
        id_only_owner.insert(
            "Id",
            mc_nbt::Nbt::String("123e4567-e89b-12d3-a456-426614174000".to_owned()),
        );
        let mut id_only = mc_nbt::NbtCompound::new();
        id_only.insert("SkullType", mc_nbt::Nbt::Byte(3));
        id_only.insert("Owner", mc_nbt::Nbt::Compound(id_only_owner));
        assert_eq!(read_skull(&id_only).owner_name, None);
    }

    #[test]
    fn banner_decode_marks_description_and_filters_unknown_masks() {
        let mut valid = mc_nbt::NbtCompound::new();
        valid.insert("Pattern", mc_nbt::Nbt::String("cre".to_owned()));
        valid.insert("Color", mc_nbt::Nbt::Int(1));
        let mut unknown = mc_nbt::NbtCompound::new();
        unknown.insert("Pattern", mc_nbt::Nbt::String("not_real".to_owned()));
        unknown.insert("Color", mc_nbt::Nbt::Int(2));
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("Base", mc_nbt::Nbt::Int(14));
        nbt.insert(
            "Patterns",
            mc_nbt::Nbt::list(vec![
                mc_nbt::Nbt::Compound(valid),
                mc_nbt::Nbt::Compound(unknown),
            ]),
        );

        let state = read_banner(&nbt);
        assert!(
            state.initialized,
            "an S35/readFromNBT makes the banner renderable"
        );
        assert_eq!(state.base_color, 14);
        assert_eq!(state.patterns, vec![("cre".to_owned(), 1)]);

        // Negative control: the lazily created Java tile starts with field_175119_g=false;
        // this is what prevents a pre-description banner from drawing a white placeholder.
        assert!(!mc_world::BannerState::default().initialized);
    }

    #[test]
    fn spawner_delay_uses_numeric_conversion_and_java_zero_default() {
        let missing = read_spawner(&mc_nbt::NbtCompound::new());
        assert_eq!(missing.spawn_delay, 0);

        let mut command_nbt = mc_nbt::NbtCompound::new();
        command_nbt.insert("Delay", mc_nbt::Nbt::Int(100));
        assert_eq!(read_spawner(&command_nbt).spawn_delay, 100);

        let mut wrong_type = mc_nbt::NbtCompound::new();
        wrong_type.insert("Delay", mc_nbt::Nbt::String("100".to_owned()));
        assert_eq!(read_spawner(&wrong_type).spawn_delay, 0);
    }

    #[test]
    fn spawner_spawn_data_overlays_visible_state_on_java_constructor_defaults() {
        let wolf = read_spawner_spawn_data("Wolf", &mc_nbt::NbtCompound::new());
        assert_eq!(wolf.dye_color, Some(14), "a fresh wolf's collar is red");
        assert!(!wolf.tamed);
        assert!(!wolf.sitting);

        let mut creeper_nbt = mc_nbt::NbtCompound::new();
        creeper_nbt.insert("powered", mc_nbt::Nbt::Byte(1));
        creeper_nbt.insert("Invisible", mc_nbt::Nbt::Byte(1));
        let creeper = read_spawner_spawn_data("Creeper", &creeper_nbt);
        assert!(creeper.powered);
        assert!(creeper.invisible);

        let mut skeleton_nbt = mc_nbt::NbtCompound::new();
        skeleton_nbt.insert("SkeletonType", mc_nbt::Nbt::Byte(1));
        let skeleton = read_spawner_spawn_data("Skeleton", &skeleton_nbt);
        assert_eq!(skeleton.variant, 1);

        let mut skeleton_params = spawner_anim_params(SpawnerRenderAnimation::default(), 0.0);
        apply_spawner_spawn_data(
            &mut skeleton_params,
            mc_data::EntityType::Skeleton,
            Some(&skeleton),
        );
        assert!(
            skeleton_params.entity.aimed_bow,
            "the wither-skeleton variant raises the cached armour model's bow pose"
        );

        let mut angry_wolf_nbt = mc_nbt::NbtCompound::new();
        angry_wolf_nbt.insert("Angry", mc_nbt::Nbt::Byte(1));
        let angry_wolf = read_spawner_spawn_data("Wolf", &angry_wolf_nbt);
        let mut angry_params = spawner_anim_params(SpawnerRenderAnimation::default(), 0.0);
        apply_spawner_spawn_data(
            &mut angry_params,
            mc_data::EntityType::Wolf,
            Some(&angry_wolf),
        );
        assert!(angry_wolf.angry);
        assert!(
            angry_params.entity.aggressive,
            "wolf aggression is a model pose input, not only a texture variant"
        );

        let mut byte_backed_variants = mc_nbt::NbtCompound::new();
        byte_backed_variants.insert("CatType", mc_nbt::Nbt::Int(257));
        assert_eq!(
            read_spawner_spawn_data("Ozelot", &byte_backed_variants).variant,
            1,
            "EntityOcelot.setTameSkin narrows CatType through its byte watcher"
        );
        byte_backed_variants.insert("RabbitType", mc_nbt::Nbt::Int(257));
        assert_eq!(
            read_spawner_spawn_data("Rabbit", &byte_backed_variants).variant,
            1,
            "EntityRabbit.setRabbitType narrows RabbitType through its byte watcher"
        );
    }

    #[test]
    fn spawner_render_appearance_snapshot_carries_name_status_and_malformed_defaults() {
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("CustomName", mc_nbt::Nbt::String("§aDinnerbone".to_owned()));
        nbt.insert("Invul", mc_nbt::Nbt::Int(120));
        nbt.insert("BatFlags", mc_nbt::Nbt::Byte(1));
        nbt.insert("ConversionTime", mc_nbt::Nbt::Int(0));
        let appearance = read_spawner_render_appearance("WitherBoss", Some(&nbt));
        assert_eq!(appearance.custom_name.as_deref(), Some("§aDinnerbone"));
        assert_eq!(appearance.wither_invul_time, 120);
        assert!(!appearance.bat_hanging, "BatFlags is species-gated");
        assert!(!appearance.converting, "ConversionTime is species-gated");
        assert_eq!(
            spawner_unformatted_name(&appearance).as_deref(),
            Some("Dinnerbone")
        );
        assert!(spawner_upside_down(
            mc_data::EntityType::WitherBoss,
            &appearance
        ));

        let bat = read_spawner_render_appearance("Bat", Some(&nbt));
        assert!(bat.bat_hanging);
        let zombie = read_spawner_render_appearance("Zombie", Some(&nbt));
        assert!(zombie.converting);

        let mut malformed = mc_nbt::NbtCompound::new();
        malformed.insert("CustomName", mc_nbt::Nbt::Int(7));
        malformed.insert("Invul", mc_nbt::Nbt::String("120".to_owned()));
        malformed.insert("BatFlags", mc_nbt::Nbt::String("true".to_owned()));
        malformed.insert("ConversionTime", mc_nbt::Nbt::String("0".to_owned()));
        assert_eq!(
            read_spawner_render_appearance("Zombie", Some(&malformed)),
            SpawnerRenderAppearance::default(),
            "wrong tags must retain constructor defaults"
        );
        assert_eq!(
            read_spawner_render_appearance("Zombie", None),
            SpawnerRenderAppearance::default(),
            "missing SpawnData must clear the local overlay"
        );
    }

    #[test]
    fn spawner_wither_texture_keeps_java_spawn_shield_flicker() {
        // RenderWither.getEntityTexture uses the ordinary sheet for the odd five-tick windows
        // while Invul is 1..=80, then stays on the invulnerable sheet through tick 81 onward.
        assert!(spawner_wither_uses_invulnerable_texture(1));
        assert!(!spawner_wither_uses_invulnerable_texture(5));
        assert!(!spawner_wither_uses_invulnerable_texture(9));
        assert!(spawner_wither_uses_invulnerable_texture(10));
        assert!(spawner_wither_uses_invulnerable_texture(80));
        assert!(spawner_wither_uses_invulnerable_texture(81));
        assert!(!spawner_wither_uses_invulnerable_texture(0));
        assert!(!spawner_wither_uses_invulnerable_texture(-1));
    }

    #[test]
    fn spawner_horse_spawn_data_preserves_packed_variant_mount_state_and_model_parts() {
        let mut saddle = mc_nbt::NbtCompound::new();
        saddle.insert("id", mc_nbt::Nbt::Short(329));
        let mut armor = mc_nbt::NbtCompound::new();
        armor.insert("id", mc_nbt::Nbt::Short(419));
        let mut riding = mc_nbt::NbtCompound::new();
        riding.insert("id", mc_nbt::Nbt::String("Pig".to_owned()));

        let mut spawn = mc_nbt::NbtCompound::new();
        spawn.insert("Type", mc_nbt::Nbt::Int(0));
        spawn.insert("Variant", mc_nbt::Nbt::Int(2 | (3 << 8)));
        spawn.insert("SaddleItem", mc_nbt::Nbt::Compound(saddle));
        spawn.insert("ArmorItem", mc_nbt::Nbt::Compound(armor));
        spawn.insert("ChestedHorse", mc_nbt::Nbt::Byte(1));
        spawn.insert("Riding", mc_nbt::Nbt::Compound(riding));

        let normal = read_spawner_spawn_data("EntityHorse", &spawn);
        assert_eq!(normal.variant, 0);
        assert_eq!(normal.horse_variant, 2 | (3 << 8));
        assert_eq!(normal.horse_armor, 3, "diamond ArmorItem selects index 3");
        assert!(normal.saddled);

        // `EntityHorse.setHorseType` stores the NBT integer in a signed-byte watcher. A
        // malformed 256 therefore wraps to normal horse type 0, which still accepts armor.
        spawn.insert("Type", mc_nbt::Nbt::Int(256));
        let wrapped_type = read_spawner_spawn_data("EntityHorse", &spawn);
        assert_eq!(wrapped_type.variant, 0);
        assert_eq!(wrapped_type.horse_armor, 3);

        // `updateHorseSlots` refuses to synchronize ArmorItem for a mule, although the item is
        // still accepted into its inventory slot by `readEntityFromNBT`.
        spawn.insert("Type", mc_nbt::Nbt::Int(2));
        let data = read_spawner_spawn_data("EntityHorse", &spawn);
        assert_eq!(data.variant, 2, "Type selects the mule model/scale");
        assert_eq!(data.horse_armor, 0);
        assert!(data.saddled);
        assert!(data.horse_chested);
        assert!(data.riding);
        spawn.insert("Riding", mc_nbt::Nbt::Compound(mc_nbt::NbtCompound::new()));
        assert!(
            !read_spawner_spawn_data("EntityHorse", &spawn).riding,
            "Java only mounts a nested compound when EntityList resolves its id"
        );
        spawn.insert(
            "Riding",
            mc_nbt::Nbt::Compound({
                let mut nested = mc_nbt::NbtCompound::new();
                nested.insert("id", mc_nbt::Nbt::String("Pig".to_owned()));
                nested
            }),
        );

        let mut params = spawner_anim_params(SpawnerRenderAnimation::default(), 0.0);
        apply_spawner_spawn_data(&mut params, mc_data::EntityType::EntityHorse, Some(&data));
        assert!(
            params.entity.riding,
            "nested Riding reaches ModelBase.isRiding"
        );

        let state = mc_render::entity::models::passive::HorseRenderState {
            horse_type: data.variant,
            adult: !data.child,
            saddled: data.saddled,
            chested: data.horse_chested,
            ridden: false,
            ..Default::default()
        };
        let mut models = TileModels::new(false);
        let mut vertices = Vec::new();
        models.emit_horse_with_state(
            mc_render::entity::Mat4::IDENTITY,
            &params,
            &state,
            &mut vertices,
        );
        assert_eq!(
            vertices.len(),
            35 * 36,
            "mule saddle/chest parts are submitted"
        );

        let selection = mc_render::entity::manager::horse_texture_selection(
            normal.variant,
            normal.horse_variant,
            normal.horse_armor,
        );
        assert_eq!(selection.upload_key(), "horse/hchwdodio");
    }

    #[test]
    fn spawner_dynamic_layers_follow_java_visibility_and_wither_health_gates() {
        let params = mc_render::entity::AnimParams::still();
        use mc_render::entity::LayerKind;

        assert!(spawner_layer_should_render_with_spawn_data(
            mc_data::EntityType::MushroomCow,
            LayerKind::MooshroomMushroom,
            &params,
            None,
        ));
        assert!(spawner_layer_should_render_with_spawn_data(
            mc_data::EntityType::SnowMan,
            LayerKind::SnowmanHead,
            &params,
            None,
        ));

        let mut baby = params.clone();
        baby.entity.child = true;
        assert!(!spawner_layer_should_render_with_spawn_data(
            mc_data::EntityType::MushroomCow,
            LayerKind::MooshroomMushroom,
            &baby,
            None,
        ));

        let mut health = mc_nbt::NbtCompound::new();
        health.insert("Health", mc_nbt::Nbt::Short(150));
        let armored = read_spawner_spawn_data("WitherBoss", &health);
        assert!(armored.wither_armored);
        assert!(spawner_layer_should_render_with_spawn_data(
            mc_data::EntityType::WitherBoss,
            LayerKind::WitherAura,
            &params,
            Some(&armored),
        ));

        health.insert("Health", mc_nbt::Nbt::Short(151));
        let unarmored = read_spawner_spawn_data("WitherBoss", &health);
        assert!(!unarmored.wither_armored);
        assert!(!spawner_layer_should_render_with_spawn_data(
            mc_data::EntityType::WitherBoss,
            LayerKind::WitherAura,
            &params,
            Some(&unarmored),
        ));

        // EntityLivingBase's Health fallback checks the exact Float/Short tag ids. A
        // malformed Int does not replace the constructor's full health and therefore cannot
        // enable the wither aura.
        health.insert("Health", mc_nbt::Nbt::Int(100));
        let wrong_health_tag = read_spawner_spawn_data("WitherBoss", &health);
        assert!(!wrong_health_tag.wither_armored);
    }

    #[test]
    fn spawner_spawn_data_retains_profile_heads_and_exact_tame_owner_precedence() {
        let mut owner = mc_nbt::NbtCompound::new();
        owner.insert(
            "Id",
            mc_nbt::Nbt::String("01234567-89ab-cdef-0123-456789abcdef".to_owned()),
        );
        owner.insert("Name", mc_nbt::Nbt::String("profile-head".to_owned()));
        let mut item_tag = mc_nbt::NbtCompound::new();
        item_tag.insert("SkullOwner", mc_nbt::Nbt::Compound(owner));
        let mut helmet = mc_nbt::NbtCompound::new();
        helmet.insert("id", mc_nbt::Nbt::String("minecraft:skull".to_owned()));
        helmet.insert("Damage", mc_nbt::Nbt::Short(3));
        helmet.insert("tag", mc_nbt::Nbt::Compound(item_tag));
        let equipment = vec![
            mc_nbt::Nbt::Compound(mc_nbt::NbtCompound::new()),
            mc_nbt::Nbt::Compound(mc_nbt::NbtCompound::new()),
            mc_nbt::Nbt::Compound(mc_nbt::NbtCompound::new()),
            mc_nbt::Nbt::Compound(mc_nbt::NbtCompound::new()),
            mc_nbt::Nbt::Compound(helmet),
        ];
        let mut spawn = mc_nbt::NbtCompound::new();
        spawn.insert("Equipment", mc_nbt::Nbt::List(equipment.into()));
        let data = read_spawner_spawn_data("Zombie", &spawn);
        let profile = data.equipment[4]
            .as_ref()
            .and_then(|item| item.skull_owner.as_ref())
            .expect("SkullOwner compound survives the typed spawner snapshot");
        assert_eq!(profile.name, "profile-head");
        assert!(profile.id.is_some());
        assert!(spawner_layer_should_render_with_spawn_data(
            mc_data::EntityType::Zombie,
            mc_render::entity::LayerKind::CustomHead,
            &mc_render::entity::AnimParams::still(),
            Some(&data),
        ));

        // java: EntityTameable.readEntityFromNBT checks an exact OwnerUUID string key first; an
        // empty UUID masks the legacy Owner string instead of promoting it.
        let mut owner_precedence = mc_nbt::NbtCompound::new();
        owner_precedence.insert("OwnerUUID", mc_nbt::Nbt::String(String::new()));
        owner_precedence.insert("Owner", mc_nbt::Nbt::String("legacy-owner".to_owned()));
        assert!(!read_spawner_spawn_data("Wolf", &owner_precedence).tamed);
        owner_precedence.remove("OwnerUUID");
        assert!(read_spawner_spawn_data("Wolf", &owner_precedence).tamed);

        // `Tame` belongs to EntityHorse; EntityTameable never reads it for wolves.
        owner_precedence.remove("Owner");
        owner_precedence.insert("Tame", mc_nbt::Nbt::Byte(1));
        assert!(!read_spawner_spawn_data("Wolf", &owner_precedence).tamed);
    }

    #[test]
    fn spawner_equipment_paths_gate_witch_items_and_decode_banner_layers() {
        let mut block_entity = mc_nbt::NbtCompound::new();
        block_entity.insert("Base", mc_nbt::Nbt::Int(14));
        let mut pattern = mc_nbt::NbtCompound::new();
        pattern.insert("Pattern", mc_nbt::Nbt::String("cre".to_owned()));
        pattern.insert("Color", mc_nbt::Nbt::Int(1));
        block_entity.insert(
            "Patterns",
            mc_nbt::Nbt::List(vec![mc_nbt::Nbt::Compound(pattern)].into()),
        );
        let mut tag = mc_nbt::NbtCompound::new();
        tag.insert("BlockEntityTag", mc_nbt::Nbt::Compound(block_entity));
        let mut banner = mc_nbt::NbtCompound::new();
        banner.insert("id", mc_nbt::Nbt::String("minecraft:banner".to_owned()));
        banner.insert("Damage", mc_nbt::Nbt::Short(3));
        banner.insert("tag", mc_nbt::Nbt::Compound(tag));
        let item = read_spawner_equipment(&banner).expect("banner item id is registered");
        let state = item
            .banner_state
            .as_ref()
            .expect("BlockEntityTag is retained for item rendering");
        assert_eq!(state.base_color, 14);
        assert_eq!(state.patterns, vec![("cre".to_owned(), 1)]);

        let mut data = mc_world::tileent::SpawnerSpawnData::default();
        data.equipment[0] = Some(item);
        let params = mc_render::entity::AnimParams::still();
        assert!(spawner_layer_should_render_with_spawn_data(
            mc_data::EntityType::Witch,
            mc_render::entity::LayerKind::WitchHeldItem,
            &params,
            Some(&data),
        ));
        assert!(!spawner_layer_should_render_with_spawn_data(
            mc_data::EntityType::Zombie,
            mc_render::entity::LayerKind::WitchHeldItem,
            &params,
            Some(&data),
        ));

        // `LayerArmorBase.renderGlint` is driven by `ItemStack.hasEffect()`; preserve that
        // payload bit and the renderer's two-pass replay contract at this seam.
        let mut enchant = mc_nbt::NbtCompound::new();
        enchant.insert(
            "id",
            mc_nbt::Nbt::String("minecraft:diamond_helmet".to_owned()),
        );
        let mut enchant_tag = mc_nbt::NbtCompound::new();
        enchant_tag.insert(
            "ench",
            mc_nbt::Nbt::list(vec![mc_nbt::Nbt::Compound(mc_nbt::NbtCompound::new())]),
        );
        enchant.insert("tag", mc_nbt::Nbt::Compound(enchant_tag));
        let enchanted = read_spawner_equipment(&enchant).expect("diamond helmet is registered");
        assert!(enchanted.enchanted);

        // Java's ItemStack.isItemEnchanted is key-presence based, even for an empty `ench`
        // list, while an unrelated StoredEnchantments list does not make armor glint.
        let mut empty_ench = mc_nbt::NbtCompound::new();
        empty_ench.insert(
            "id",
            mc_nbt::Nbt::String("minecraft:diamond_helmet".to_owned()),
        );
        empty_ench.insert(
            "tag",
            mc_nbt::Nbt::Compound({
                let mut tag = mc_nbt::NbtCompound::new();
                tag.insert("ench", mc_nbt::Nbt::List(Vec::new().into()));
                tag
            }),
        );
        assert!(
            read_spawner_equipment(&empty_ench)
                .expect("empty ench list still keeps the item")
                .enchanted
        );
        let mut stored_only = empty_ench.clone();
        stored_only.insert(
            "tag",
            mc_nbt::Nbt::Compound({
                let mut tag = mc_nbt::NbtCompound::new();
                tag.insert(
                    "StoredEnchantments",
                    mc_nbt::Nbt::list(vec![mc_nbt::Nbt::Compound(mc_nbt::NbtCompound::new())]),
                );
                tag
            }),
        );
        assert!(
            !read_spawner_equipment(&stored_only)
                .expect("stored enchantment item remains registered")
                .enchanted
        );
        let source = vec![EntityVertex::default()];
        let mut glint = Vec::new();
        mc_render::entity::item::emit_armor_glint(&source, 0.0, &mut glint);
        assert_eq!(glint.len(), 2, "armour glint replays both Java passes");
    }

    /// Encoder for the tests only — production only ever decodes.
    /// Finding F-P0-network.play.server-13-6 is the live S35 seam, not merely the NBT helper:
    /// `NetHandlerPlayClient.handleUpdateTileEntity` replaces the spawner's descriptive fields
    /// while `MobSpawnerBaseLogic.readFromNBT` leaves the accumulated animation fields alone.
    /// The packet's absent `Delay` is an unconditional `getShort` read, hence Java's zero getter
    /// default, while a present value must still land. Wrong actions are a type-gated no-op.
    #[test]
    fn spawner_cached_entity_keeps_constructor_pose_and_only_safe_layers() {
        let animation = SpawnerRenderAnimation { entity_id: 17 };
        let params = spawner_anim_params(animation, 0.375);
        assert_eq!(params.partial_ticks, 0.375);
        assert_eq!(params.age_in_ticks, 0.375);
        assert_eq!(params.limb_swing, 0.0);
        assert_eq!(params.limb_swing_amount, 0.0);
        assert_eq!(params.net_head_yaw, 0.0);
        assert_eq!(params.head_pitch, 0.0);
        assert_eq!(params.entity.ticks_existed, 0);
        assert_eq!(params.entity.entity_id, 17);

        use mc_render::entity::LayerKind;
        assert!(spawner_layer_should_render(
            mc_data::EntityType::Sheep,
            LayerKind::SheepWool,
            &params
        ));
        assert!(spawner_layer_should_render(
            mc_data::EntityType::Enderman,
            LayerKind::EndermanEyes,
            &params
        ));
        assert!(spawner_layer_should_render(
            mc_data::EntityType::Slime,
            LayerKind::SlimeGel,
            &params
        ));
        assert!(!spawner_layer_should_render(
            mc_data::EntityType::Pig,
            LayerKind::Saddle,
            &params
        ));
        assert!(!spawner_layer_should_render(
            mc_data::EntityType::Zombie,
            LayerKind::HeldItem,
            &params
        ));
        assert!(!spawner_layer_should_render(
            mc_data::EntityType::Creeper,
            LayerKind::CreeperCharge,
            &params
        ));
        assert!(!spawner_layer_should_render(
            mc_data::EntityType::WitherBoss,
            LayerKind::WitherAura,
            &params
        ));

        let mut charged = params.clone();
        charged.entity.variant = 1;
        assert!(spawner_layer_should_render(
            mc_data::EntityType::Creeper,
            LayerKind::CreeperCharge,
            &charged
        ));

        let mut invisible = params.clone();
        invisible.entity.invisible = true;
        assert!(!spawner_layer_should_render(
            mc_data::EntityType::Sheep,
            LayerKind::SheepWool,
            &invisible
        ));
        assert!(!spawner_layer_should_render(
            mc_data::EntityType::Slime,
            LayerKind::SlimeGel,
            &invisible
        ));
        // Eyes keep their Java layer even for an invisible cached entity; the entity renderer's
        // depth-mask branch, not the layer selection, owns that distinction.
        assert!(spawner_layer_should_render(
            mc_data::EntityType::Enderman,
            LayerKind::EndermanEyes,
            &invisible
        ));
    }

    #[test]
    fn spawner_animation_cache_is_render_generation_safe_and_never_ticks() {
        let pos = BlockPos::new(4, 64, -2);
        let mut index = TileIndex::new();

        // A frame-only read creates the neutral cache.  This is the negative control for ticking
        // every cached miniature as though it were a world entity.
        assert_eq!(
            index.spawner_animation(pos, "Pig"),
            SpawnerRenderAnimation::default(),
            "the default cached entity starts at constructor pose"
        );
        assert_eq!(
            index.spawner_animation(pos, "Pig"),
            SpawnerRenderAnimation::default()
        );

        // EntityId changes replace Java's cached entity rather than inheriting its pose/age.
        index.observe_spawner_animation(pos, "Zombie");
        assert_eq!(
            index.spawner_animation(pos, "Zombie"),
            SpawnerRenderAnimation::default()
        );
        assert_eq!(
            index.spawner_animation(pos, "Pig"),
            SpawnerRenderAnimation::default()
        );

        // Missing EntityId follows MobSpawnerBaseLogic's field initializer; an unknown explicit
        // id remains a safe renderer no-op and must not be rewritten to Pig.
        assert_eq!(read_spawner(&mc_nbt::NbtCompound::new()).entity_id, "Pig");
        assert!(mc_data::entity_by_name("definitely_not_a_1_8_9_entity").is_none());
    }

    #[test]
    fn spawner_rotation_still_uses_the_partial_interpolated_outer_matrix() {
        let still = spawner_matrix(Vec3::ZERO, 0.0, 0.0, 0.5);
        let halfway = spawner_matrix(Vec3::ZERO, 0.0, 90.0, 0.5);
        let finished = spawner_matrix(Vec3::ZERO, 0.0, 90.0, 1.0);
        assert_ne!(
            still, halfway,
            "a live rotation must reach the miniature body"
        );
        assert_ne!(halfway, finished, "partial ticks must interpolate rotation");
        let midpoint = halfway.transform_point3(Vec3::new(1.0, 0.4, 0.0));
        assert!(
            midpoint.x.is_finite() && midpoint.y.is_finite() && midpoint.z.is_finite(),
            "the matrix remains finite under the ordinary transform chain"
        );
    }

    #[test]
    fn spawner_s35_replaces_description_without_rebuilding_animation_state() {
        let Some((_dir, mut client)) = real_client() else {
            return;
        };
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        let pos = BlockPos::new(2, 64, 2);
        {
            let world = &mut session.state_mut().world;
            world.load_empty_chunk(pos.chunk_pos());
            world.set_block(pos, BlockState::new(BlockId(52), 0));
            let entity = world
                .tile_entity_or_create(pos)
                .expect("the spawner block owns a tile entity");
            let mc_world::TileData::Spawner(spawner) = &mut entity.data else {
                panic!("spawner block must carry SpawnerState");
            };
            spawner.mob_rotation = 123.5;
            spawner.prev_mob_rotation = 98.25;
            spawner.spawn_delay = 77;
        }
        client.session = Some(session);

        let mut present = mc_nbt::NbtCompound::new();
        present.insert("EntityId", mc_nbt::Nbt::String("Zombie".to_owned()));
        present.insert("Delay", mc_nbt::Nbt::Int(37));
        client.react_to_tile_entity_update(pos, 1, Some(&present));
        let state = client
            .session()
            .and_then(|session| session.state().world.tile_entity(pos))
            .and_then(mc_world::TileEntity::spawner)
            .expect("present S35 keeps the spawner live");
        assert_eq!(state.entity_id, "Zombie");
        assert_eq!(state.spawn_delay, 37);
        assert_eq!(state.mob_rotation, 123.5);
        assert_eq!(state.prev_mob_rotation, 98.25);

        // A repeat description without Delay must not fabricate the old port's 20-tick value,
        // and it still must not rewind either side of the render interpolation pair.
        let mut missing_delay = mc_nbt::NbtCompound::new();
        missing_delay.insert("EntityId", mc_nbt::Nbt::String("Skeleton".to_owned()));
        client.react_to_tile_entity_update(pos, 1, Some(&missing_delay));
        let state = client
            .session()
            .and_then(|session| session.state().world.tile_entity(pos))
            .and_then(mc_world::TileEntity::spawner)
            .expect("missing-Delay S35 keeps the spawner live");
        assert_eq!(state.entity_id, "Skeleton");
        assert_eq!(state.spawn_delay, 0);
        assert_eq!(state.mob_rotation, 123.5);
        assert_eq!(state.prev_mob_rotation, 98.25);

        // Action 2 belongs to command blocks; it cannot replace an existing spawner, even with a
        // plausible EntityId/Delay payload.
        let mut wrong_action = mc_nbt::NbtCompound::new();
        wrong_action.insert("EntityId", mc_nbt::Nbt::String("Creeper".to_owned()));
        wrong_action.insert("Delay", mc_nbt::Nbt::Int(55));
        client.react_to_tile_entity_update(pos, 2, Some(&wrong_action));
        let state = client
            .session()
            .and_then(|session| session.state().world.tile_entity(pos))
            .and_then(mc_world::TileEntity::spawner)
            .expect("wrong action cannot remove the spawner");
        assert_eq!(state.entity_id, "Skeleton");
        assert_eq!(state.spawn_delay, 0);
        assert_eq!(state.mob_rotation, 123.5);
        assert_eq!(state.prev_mob_rotation, 98.25);
    }

    fn encode(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            for i in 0..4 {
                if i <= chunk.len() {
                    out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }
}

/// The kinds the snapshot skips must be
/// exactly the pose loop's no-TESR arm, and the snapshot loop must give none of them an effect.
#[cfg(test)]
mod nw_f1_tile_fast_tests {
    use super::*;

    const ALL: [TileKind; 24] = [
        TileKind::Dispenser,
        TileKind::Chest,
        TileKind::TrappedChest,
        TileKind::EnderChest,
        TileKind::Furnace,
        TileKind::Jukebox,
        TileKind::StandingSign,
        TileKind::WallSign,
        TileKind::StandingBanner,
        TileKind::WallBanner,
        TileKind::Skull,
        TileKind::Beacon,
        TileKind::MobSpawner,
        TileKind::EnchantmentTable,
        TileKind::EndPortal,
        TileKind::FlowerPot,
        TileKind::BrewingStand,
        TileKind::Comparator,
        TileKind::DaylightDetector,
        TileKind::Hopper,
        TileKind::Dropper,
        TileKind::Piston,
        TileKind::Note,
        TileKind::CommandBlock,
    ];

    fn upload_body() -> &'static str {
        let source = include_str!("tiles.rs");
        let body = source
            .split_once("fn upload_tile_entities_with_clock(")
            .expect("the upload function")
            .1;
        body.split_once("\n    fn ensure_beacon_texture(")
            .expect("the next function")
            .0
    }

    #[test]
    fn skipped_kinds_are_exactly_the_pose_loops_no_tesr_arm() {
        let body = upload_body();
        let arm_end = body
            .find("| TileKind::FlowerPot =>")
            .expect("the pose loop's no-TESR arm");
        let arm_start = body[..arm_end]
            .rfind("TileKind::Dispenser")
            .expect("the arm's first kind");
        let arm = &body[arm_start..arm_end + "| TileKind::FlowerPot".len()];
        let mut named: Vec<String> = arm
            .split("TileKind::")
            .skip(1)
            .map(|rest| {
                rest.chars()
                    .take_while(char::is_ascii_alphanumeric)
                    .collect()
            })
            .collect();
        named.sort();
        let mut skipped: Vec<String> = ALL
            .iter()
            .filter(|&&kind| nw_f1_tile_draws_nothing(kind))
            .map(|kind| format!("{kind:?}"))
            .collect();
        skipped.sort();
        assert_eq!(named, skipped);
        // The arm (the match's last) yields no texture: no draw and no destroy-stage replay.
        let after = &body[arm_end..];
        let arm_body = &after[..after.find("\n            };").expect("the match's end")];
        assert!(
            arm_body.contains("None") && !arm_body.contains("Some("),
            "{arm_body}"
        );
    }

    #[test]
    fn the_snapshot_loop_gives_skipped_kinds_no_side_effect() {
        let body = upload_body();
        let start = body
            .find("let mut draw = TileDraw {")
            .expect("snapshot loop");
        let end = body.find("draws.push(draw);").expect("snapshot push");
        let snapshot = &body[start..end];
        for kind in ALL.iter().filter(|&&kind| nw_f1_tile_draws_nothing(kind)) {
            let name = format!("TileKind::{kind:?}");
            assert!(!snapshot.contains(&name), "{name} has a snapshot arm");
        }
        // Only the beacon has a world side effect before the snapshot, and it is not skipped.
        assert!(!nw_f1_tile_draws_nothing(TileKind::Beacon));
        assert!(!nw_f1_tile_draws_nothing(TileKind::MobSpawner));
    }
}

/// The 3x3 rescan that keeps an unchanged column's scan against the old
/// every-tick rescan, over random block writes, unloads and player moves.
#[cfg(test)]
mod w18_oa_near_rescan_tests {
    use super::*;

    /// A fixed-seed splitmix64 stream: deterministic cases without a dependency.
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, bound: u64) -> i32 {
            i32::try_from(self.next() % bound).expect("small")
        }
    }

    /// Chest, sign, skull, banner, spawner, furnace, beacon (tile providers) and stone, glass
    /// and air (not).
    const BLOCKS: [(u16, u8); 10] = [
        (54, 2),
        (63, 4),
        (144, 1),
        (176, 0),
        (52, 0),
        (61, 3),
        (138, 0),
        (1, 0),
        (20, 0),
        (0, 0),
    ];

    /// One indexed column: its chunk coordinates and its tile entities.
    type Column = (i32, i32, Vec<(BlockPos, TileKind)>);

    fn sorted_columns(index: &TileIndex) -> Vec<Column> {
        let mut columns: Vec<_> = index
            .columns
            .iter()
            .map(|(pos, tiles)| (pos.x, pos.z, tiles.clone()))
            .collect();
        columns.sort_by_key(|(x, z, _)| (*x, *z));
        columns
    }

    #[test]
    fn keeping_unchanged_near_scans_indexes_exactly_what_rescanning_does() {
        let mut stream = Stream(0x0A18_0003);
        let mut world = World::new();
        for cx in -3..=3 {
            for cz in -3..=3 {
                world.load_empty_chunk(ChunkPos::new(cx, cz));
            }
        }
        let mut keeping = TileIndex::new();
        let mut rescanning = TileIndex::new();
        // Negative control: an index whose held sections are swapped for the live ones before
        // each tick, so it wrongly calls every near column unchanged, must be caught.
        let mut broken = TileIndex::new();
        let mut broken_caught = false;
        let mut player = BlockPos::new(8, 64, 8);
        let mut provider_writes = 0;
        for step in 0..400 {
            // Zero to three writes, mostly near the player, sometimes anywhere in the area.
            for _ in 0..stream.below(4) {
                let spread: i32 = if stream.below(4) == 0 { 56 } else { 20 };
                let width = u64::from((2 * spread + 1).unsigned_abs());
                let pos = BlockPos::new(
                    player.x + stream.below(width) - spread,
                    stream.below(256),
                    player.z + stream.below(width) - spread,
                );
                let (id, meta) = BLOCKS[stream.below(BLOCKS.len() as u64) as usize];
                if world.is_chunk_loaded(pos.chunk_pos()) {
                    world.set_block(pos, BlockState::new(BlockId(id), meta));
                    provider_writes += usize::from(id != 1 && id != 20 && id != 0);
                }
            }
            // Now and then a column near the player unloads, and later comes back empty.
            if step % 37 == 11 {
                world.unload_chunk(ChunkPos::new(
                    player.chunk_pos().x + 1,
                    player.chunk_pos().z,
                ));
            }
            if step % 37 == 30 {
                for cx in -3..=3 {
                    for cz in -3..=3 {
                        if !world.is_chunk_loaded(ChunkPos::new(cx, cz)) {
                            world.load_empty_chunk(ChunkPos::new(cx, cz));
                        }
                    }
                }
            }
            // The player wanders across column borders inside the loaded area.
            if step % 5 == 0 {
                player = BlockPos::new(
                    (player.x + stream.below(25) - 12).clamp(-40, 56),
                    64,
                    (player.z + stream.below(25) - 12).clamp(-40, 56),
                );
            }
            keeping.refresh(&world, Some(player));
            // The old rule: no column is ever known to be unchanged, so all nine are rescanned.
            rescanning.near_sections.clear();
            rescanning.refresh(&world, Some(player));
            assert_eq!(
                sorted_columns(&keeping),
                sorted_columns(&rescanning),
                "step {step}"
            );
            // Every near column the keeping index would skip next tick is really current.
            for (pos, tiles) in &keeping.columns {
                if keeping.near_column_unchanged(&world, *pos) {
                    assert_eq!(*tiles, world.scan_chunk_tile_entities(*pos), "step {step}");
                }
            }
            for (pos, held) in &mut broken.near_sections {
                if let Some(chunk) = world.chunk(*pos) {
                    *held = chunk.sections().clone();
                }
            }
            broken.refresh(&world, Some(player));
            broken_caught |= sorted_columns(&broken) != sorted_columns(&rescanning);
        }
        assert!(provider_writes > 100, "{provider_writes} provider writes");
        assert!(broken_caught, "the comparison must see a stale near column");
    }

    #[test]
    fn one_block_write_ends_the_reuse_of_its_column() {
        let mut world = World::new();
        let centre = BlockPos::new(8, 70, 8);
        for cx in -1..=1 {
            for cz in -1..=1 {
                world.load_empty_chunk(ChunkPos::new(cx, cz));
            }
        }
        world.set_block(centre, BlockState::new(BlockId(54), 2));
        let mut index = TileIndex::new();
        index.refresh(&world, Some(centre));
        let column = centre.chunk_pos();
        assert!(index.near_column_unchanged(&world, column));
        // Negative control: a write anywhere in the column, even of a non-provider, is seen.
        world.set_block(BlockPos::new(3, 12, 3), BlockState::new(BlockId(1), 0));
        assert!(!index.near_column_unchanged(&world, column));
        index.refresh(&world, Some(centre));
        assert!(index.near_column_unchanged(&world, column));
        // A provider placed in a near column is indexed on the next tick.
        let sign = BlockPos::new(-5, 64, 20);
        world.set_block(sign, BlockState::new(BlockId(63), 4));
        index.refresh(&world, Some(centre));
        assert!(index
            .column(sign.chunk_pos())
            .contains(&(sign, TileKind::StandingSign)));
        // An unload drops the reuse; a reloaded column is a new chunk and is scanned afresh.
        world.unload_chunk(column);
        assert!(!index.near_column_unchanged(&world, column));
        world.load_empty_chunk(column);
        index.refresh(&world, Some(centre));
        assert!(index.column(column).is_empty());
    }
}

/// The borrowed owner fields against `read_game_profile_from_nbt`, and
/// a worn head through a skin change on both paths.
#[cfg(test)]
mod w18_oa_head_profile_tests {
    use super::*;
    use base64::Engine as _;
    use mc_nbt::{Nbt, NbtCompound, NbtList};

    /// What the head renderer reads from the reader's profile, owned for comparison.
    fn reader_parts(owner: &NbtCompound) -> Option<(Option<uuid::Uuid>, String, Option<String>)> {
        let profile = mc_auth::read_game_profile_from_nbt(owner)?;
        let parts = head_parts_from_profile(&profile);
        Some((
            parts.id,
            parts.name.to_owned(),
            parts.textures.map(str::to_owned),
        ))
    }

    fn borrowed_parts(owner: &NbtCompound) -> Option<(Option<uuid::Uuid>, String, Option<String>)> {
        let parts = head_parts_from_nbt(owner)?;
        Some((
            parts.id,
            parts.name.to_owned(),
            parts.textures.map(str::to_owned),
        ))
    }

    fn property(value: Option<&str>, signature: Option<&str>) -> Nbt {
        let mut entry = NbtCompound::new();
        if let Some(value) = value {
            entry.insert("Value", Nbt::String(value.to_owned()));
        }
        if let Some(signature) = signature {
            entry.insert("Signature", Nbt::String(signature.to_owned()));
        }
        Nbt::Compound(entry)
    }

    fn owner(id: Option<Nbt>, name: Option<Nbt>, properties: Option<Nbt>) -> NbtCompound {
        let mut owner = NbtCompound::new();
        if let Some(id) = id {
            owner.insert("Id", id);
        }
        if let Some(name) = name {
            owner.insert("Name", name);
        }
        if let Some(properties) = properties {
            owner.insert("Properties", properties);
        }
        owner
    }

    fn by_name(entries: Vec<(&str, Nbt)>) -> Nbt {
        let mut compound = NbtCompound::new();
        for (key, value) in entries {
            compound.insert(key, value);
        }
        Nbt::Compound(compound)
    }

    fn list(items: Vec<Nbt>) -> Nbt {
        Nbt::List(NbtList::from(items))
    }

    fn text(value: &str) -> Option<Nbt> {
        Some(Nbt::String(value.to_owned()))
    }

    #[test]
    fn borrowed_owner_fields_equal_the_profile_readers_for_every_shape() {
        let uuid = "5a110000-0000-4000-8000-000000000001";
        let textures = |items: Vec<Nbt>| Some(by_name(vec![("textures", list(items))]));
        let mut cases = vec![
            // The shapes servers send: id, name, one signed or unsigned textures value.
            owner(
                text(uuid),
                text("Head"),
                textures(vec![property(Some("dmFsdWU="), Some("c2ln"))]),
            ),
            owner(
                text(uuid),
                text("Head"),
                textures(vec![property(Some("dmFsdWU="), None)]),
            ),
            // Two values: the first is the one the renderer reads.
            owner(
                text(uuid),
                text("Head"),
                textures(vec![
                    property(Some("Zmlyc3Q="), None),
                    property(Some("c2Vjb25k"), None),
                ]),
            ),
            // A missing `Value` reads as empty; an empty list or one of another type gives none.
            owner(
                text(uuid),
                text("Head"),
                textures(vec![property(None, Some("c2ln"))]),
            ),
            owner(text(uuid), text("Head"), textures(Vec::new())),
            owner(
                text(uuid),
                text("Head"),
                textures(vec![Nbt::String("x".to_owned())]),
            ),
            owner(
                text(uuid),
                text("Head"),
                textures(vec![property(Some("Zmlyc3Q="), None), Nbt::Int(3)]),
            ),
            // Other properties before and after, and a non-list `textures`.
            owner(
                text(uuid),
                text("Head"),
                Some(by_name(vec![
                    ("other", list(vec![property(Some("b3RoZXI="), None)])),
                    ("textures", list(vec![property(Some("dGV4"), None)])),
                    ("later", list(vec![property(Some("bGF0ZXI="), None)])),
                ])),
            ),
            owner(
                text(uuid),
                text("Head"),
                Some(by_name(vec![("textures", Nbt::String("dGV4".to_owned()))])),
            ),
            owner(
                text(uuid),
                text("Head"),
                Some(Nbt::String("Properties".to_owned())),
            ),
            owner(text(uuid), text("Head"), None),
            // Id and name edge cases: Java's lenient UUID spelling, a malformed id, missing or
            // empty parts (both missing or empty is no profile at all), wrong tag types.
            owner(text("1-2-3-4-5"), text("Head"), None),
            owner(text("not-a-uuid"), text("Head"), None),
            owner(text(uuid), None, None),
            owner(text(uuid), text(""), None),
            owner(None, text("Head"), None),
            owner(text(""), text("Head"), None),
            owner(text(""), text(""), None),
            owner(None, None, textures(vec![property(Some("dGV4"), None)])),
            owner(Some(Nbt::Int(7)), text("Head"), None),
            owner(text(uuid), Some(Nbt::Int(7)), None),
            owner(Some(Nbt::Int(7)), Some(Nbt::Int(7)), None),
        ];
        // And a deterministic spread of mixed shapes.
        let mut seed = 0x0A18_0005_u64;
        let mut next = |bound: u64| {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) % bound
        };
        let values = ["", "dGV4", "QQ==", "Zmlyc3Q="];
        for _ in 0..1_500 {
            let id = match next(5) {
                0 => None,
                1 => text(uuid),
                2 => text(""),
                3 => text("1-2-3-4-5"),
                _ => Some(Nbt::Int(1)),
            };
            let name = match next(4) {
                0 => None,
                1 => text("Head"),
                2 => text(""),
                _ => Some(Nbt::Byte(1)),
            };
            let mut entries = Vec::new();
            for _ in 0..next(4) {
                let key = ["textures", "other", "textures2"][next(3) as usize];
                let items = (0..next(4))
                    .map(|_| match next(4) {
                        0 => Nbt::Int(1),
                        1 => property(None, None),
                        _ => property(Some(values[next(4) as usize]), None),
                    })
                    .collect();
                let value = if next(6) == 0 {
                    Nbt::String("s".to_owned())
                } else {
                    list(items)
                };
                entries.push((key, value));
            }
            let properties = (next(5) != 0).then(|| by_name(entries));
            cases.push(owner(id, name, properties));
        }
        let mut some = 0;
        for (index, case) in cases.iter().enumerate() {
            let expected = reader_parts(case);
            some += usize::from(expected.as_ref().is_some_and(|parts| parts.2.is_some()));
            assert_eq!(borrowed_parts(case), expected, "case {index}: {case:?}");
        }
        assert!(some > 50, "{some} cases carry a textures value");
        // Negative control: the comparison sees a different textures value.
        let first = owner(
            text(uuid),
            text("Head"),
            textures(vec![property(Some("QQ=="), None)]),
        );
        let second = owner(
            text(uuid),
            text("Head"),
            textures(vec![property(Some("Qg=="), None)]),
        );
        assert_ne!(borrowed_parts(&first), reader_parts(&second));
    }

    fn textures_value(hash: &str) -> String {
        let payload = format!(
            r#"{{"textures":{{"SKIN":{{"url":"https://textures.minecraft.net/texture/{hash}"}}}}}}"#
        );
        base64::engine::general_purpose::STANDARD.encode(payload)
    }

    fn worn_head(name: &str, hash: Option<&str>) -> mc_protocol::ItemStack {
        let properties = hash.map(|hash| {
            let entries = vec![property(Some(&textures_value(hash)), Some("c2ln"))];
            by_name(vec![("textures", list(entries))])
        });
        let owner = owner(
            text("5a110000-0000-4000-8000-0000000000a5"),
            text(name),
            properties,
        );
        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::Compound(owner));
        mc_protocol::ItemStack {
            item_id: 397,
            count: 1,
            damage: 3,
            nbt: Some(root),
        }
    }

    /// The owner one frame resolves for `stack`, on the borrowed or the reader path (the two
    /// branches of `player_skull_owner`'s compound arm).
    fn owner_of(stack: &mc_protocol::ItemStack, borrowed: bool) -> Option<BuiltinSkullOwner> {
        let Some(Nbt::Compound(owner)) = stack.nbt.as_ref()?.get("SkullOwner") else {
            return None;
        };
        if borrowed {
            Some(player_skull_owner_from_parts(head_parts_from_nbt(owner)?))
        } else {
            Some(player_skull_owner_from_profile(
                &mc_auth::read_game_profile_from_nbt(owner)?,
            ))
        }
    }

    #[test]
    fn a_worn_heads_skin_change_resolves_identically_on_both_paths() {
        let hashes = ["w18-oa-head-old", "w18-oa-head-new"];
        let key = |hash: &str| {
            crate::app::appearance::skin_texture_key_for_url(&format!(
                "https://textures.minecraft.net/texture/{hash}"
            ))
            .expect("safe hash")
        };
        // Publish the skins first, so no request reaches the process-wide handoff other tests
        // in this binary read.
        for hash in hashes {
            mc_render::entity::skin::publish_shared_skin(
                &key(hash),
                &mc_render::entity::SkinTexture {
                    rgba: vec![255; 64 * 64 * 4].into(),
                    layout: mc_render::entity::SkinLayout::Modern64x64,
                    model: mc_render::entity::SkinModel::Classic,
                },
            );
        }
        // The frames of one worn head: its skin, then the server changes the skin, then the
        // owner name, then drops the textures.
        let frames = vec![
            worn_head("WornHead", Some(hashes[0])),
            worn_head("WornHead", Some(hashes[0])),
            worn_head("WornHead", Some(hashes[1])),
            worn_head("WornHead", Some(hashes[1])),
            worn_head("RenamedHead", Some(hashes[1])),
            worn_head("BareHead", None),
        ];
        // Each path in a thread of its own: each starts with an empty descriptor cache.
        let run = |borrowed: bool| {
            let frames = frames.clone();
            std::thread::spawn(move || {
                frames
                    .iter()
                    .map(|stack| {
                        let owner = owner_of(stack, borrowed).expect("a complete profile");
                        let texture = owner.texture_key();
                        (owner, texture)
                    })
                    .collect::<Vec<_>>()
            })
            .join()
            .expect("the path's thread")
        };
        let borrowed = run(true);
        let reader = run(false);
        assert_eq!(borrowed, reader);
        // And the sequence is vanilla's: the equal profile keeps its cached skin while it is
        // hot; another name is another profile and gets the new skin.
        let keys: Vec<&str> = borrowed
            .iter()
            .map(|(owner, _)| owner.cache_key.as_str())
            .collect();
        assert_eq!(keys[..4], [key(hashes[0]).as_str(); 4]);
        assert_eq!(keys[4], key(hashes[1]));
        assert_eq!(
            borrowed[2].1,
            key(hashes[0]),
            "a published skin binds at once"
        );
        assert_ne!(keys[5], key(hashes[1]), "no textures: the UUID default");
    }
}
