//! `RenderManager` — entity type to renderer, and the per-renderer constants.
//!
//! java: `client/renderer/entity/RenderManager.java` (the `entityRenderMap` built in the
//! constructor) plus each `Render*` subclass's own constructor and `getEntityTexture`.
//!
//! # Why a table and not a trait object per entity
//!
//! Vanilla keys a `HashMap<Class<? extends Entity>, Render>` by Java class, which is how it
//! dispatches. There are no entity classes here — an entity is an `hecs` id with an
//! [`EntityKind`](mc_world::EntityKind) component — so the equivalent is a
//! `match` from [`mc_data::EntityType`] to a static [`EntityRenderer`] descriptor. That makes
//! the whole dispatch table a `const`-shaped, exhaustively testable thing rather than a
//! registry built at startup.
//!
//! [`RenderManager`] then owns one built model per [`ModelKind`], created lazily and shared by
//! every entity using it — exactly what vanilla does, since a `Render` instance holds one
//! `ModelBase` no matter how many mobs it draws. That sharing is why posing and emitting are
//! two steps: the model is mutated, drawn, and then reused for the next entity of that type.

use glam::{Mat4, Vec3};
use mc_data::EntityType;
use mc_math::JavaRandom;
use mc_world::EntityKind;

use super::anim::{AnimParams, EntityModel};
use super::layers::{emit_cape, ArmorSlot, CapeState, LayerKind};
use super::models::biped::{BipedModel, PlayerModel};
use super::models::object::ArmorStandArmorModel;
use super::models::passive::{HorseModel, HorseRenderState};
use super::objects::{boat_matrix, ender_crystal_matrix, leash_knot_matrix, minecart_matrix};
use super::render::{entity_matrix, entity_matrix_scaled, slime_pre_render_scale, EntityTransform};
use super::skin::SkinModel;
use super::EntityVertex;

pub use super::models::ModelKind;

/// Everything the renderer needs to know about one entity type.
///
/// java: the constructor arguments and overridden methods of one `Render*` class.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityRenderer {
    /// The model to build. java: the `new Model*()` argument.
    pub model: ModelKind,
    /// Default texture. java: `getEntityTexture` — see [`texture_for`] for the variants.
    pub texture: &'static str,
    /// java: the `shadowSize` constructor argument.
    pub shadow_size: f32,
    /// The **static** part of `preRenderCallback`'s scale.
    ///
    /// Renderers whose scale depends on live entity state — the slime's squish, the magma
    /// cube's size, the guardian's spikes — are `1.0` here and are the caller's job; their
    /// scale is non-uniform, so it does not fit this field at all.
    pub scale: f32,
    /// java: `getDeathMaxRotation` — 90 for everything except the guardian (180).
    /// Java exceptions: spiders, silverfish, and endermites use `180`, ArmorStand uses `0`,
    /// and ordinary living renderers use `90`.
    pub death_max_rotation: f32,
    /// java: the `addLayer` calls in the renderer's constructor, in order.
    pub layers: &'static [LayerKind],
}

/// The layer stack a `RenderBiped` subclass that adds armour ends up with.
///
/// java, and the order is not the obvious one: `RenderZombie`'s constructor (`:27`) calls the
/// **four**-argument `RenderBiped` super, which adds only `new LayerCustomHead(bipedHead)`
/// (`RenderBiped:27`); the three-argument overload is what adds `LayerHeldItem` (`RenderBiped:19`).
/// `RenderZombie` then adds `LayerHeldItem` (`:31`) and `LayerBipedArmor` (`:40`) itself. So the
/// sequence is head, held item, armour — the exact reverse of what this constant used to say.
/// `RenderPigZombie` (`:15-17`) is identical.
///
/// `RenderSkeleton` (`:17-19`) takes the *three*-argument super and then adds `LayerHeldItem` a
/// second time, so its layer list is intentionally not shared with the otherwise identical
/// zombie/pig-zombie stack. The duplicate is part of vanilla's registration order; callers that
/// replay the descriptor list must preserve both held-item passes.
const BIPED_LAYERS: &[LayerKind] = &[LayerKind::CustomHead, LayerKind::HeldItem, LayerKind::Armor];
/// java: `RenderSkeleton.<init>` (`:17-19`) inherits the first `LayerHeldItem` from the
/// three-argument `RenderBiped` constructor, then registers a second one before armour.
const SKELETON_LAYERS: &[LayerKind] = &[
    LayerKind::CustomHead,
    LayerKind::HeldItem,
    LayerKind::HeldItem,
    LayerKind::Armor,
];
/// `RenderZombie` removes its ordinary head/armour layers, adds their villager forms, then saves
/// that list.  `List.remove`/`add` means the resulting order is held item, villager head, villager
/// armour — not the ordinary biped order.
///
/// java: `RenderZombie.<init>` (`field_177121_n`).  The layer enum stays the same because the
/// differing head attachment follows the selected body model and the differing armour geometry
/// is selected by [`RenderManager::emit_armor_slot_for`].
const ZOMBIE_VILLAGER_LAYERS: &[LayerKind] =
    &[LayerKind::HeldItem, LayerKind::CustomHead, LayerKind::Armor];
/// java: `ArmorStandRenderer`'s constructor (`:30-32`) adds armour, held item, then head — which
/// is the opposite of [`BIPED_LAYERS`], because it extends `RendererLivingEntity` directly and
/// does its own three `addLayer` calls rather than inheriting `RenderBiped`'s.
const ARMOR_STAND_LAYERS: &[LayerKind] =
    &[LayerKind::Armor, LayerKind::HeldItem, LayerKind::CustomHead];
/// java: `RenderGiantZombie`'s constructor (`:22-23`) adds `LayerHeldItem` then `LayerBipedArmor`
/// and nothing else — it extends `RenderLiving`, not `RenderBiped`, so a giant has **no**
/// `LayerCustomHead` and cannot wear a skull.
const GIANT_LAYERS: &[LayerKind] = &[LayerKind::HeldItem, LayerKind::Armor];
const NO_LAYERS: &[LayerKind] = &[];
const DRAGON_LAYERS: &[LayerKind] = &[LayerKind::DragonEyes, LayerKind::DragonDeath];

// java: `EntityHorse`'s four texture tables (`:53-59`) and the abbreviations used by
// `setHorseTexturePaths` (`:712-782`).  A horse's normal colour/marking/armour sheets are
// composed by `LayeredTexture`, so they cannot be represented by the one static descriptor
// string used by ordinary entity renderers.  Keep the table here, beside `texture_for`, and
// expose the three source paths for the eventual resource-loader seam.
const HORSE_TEXTURES: [&str; 7] = [
    "textures/entity/horse/horse_white.png",
    "textures/entity/horse/horse_creamy.png",
    "textures/entity/horse/horse_chestnut.png",
    "textures/entity/horse/horse_brown.png",
    "textures/entity/horse/horse_black.png",
    "textures/entity/horse/horse_gray.png",
    "textures/entity/horse/horse_darkbrown.png",
];
const HORSE_TEXTURE_ABBR: [&str; 7] = ["hwh", "hcr", "hch", "hbr", "hbl", "hgr", "hdb"];
const HORSE_MARKINGS: [Option<&str>; 5] = [
    None,
    Some("textures/entity/horse/horse_markings_white.png"),
    Some("textures/entity/horse/horse_markings_whitefield.png"),
    Some("textures/entity/horse/horse_markings_whitedots.png"),
    Some("textures/entity/horse/horse_markings_blackdots.png"),
];
const HORSE_MARKING_ABBR: [&str; 5] = ["", "wo_", "wmo", "wdo", "bdo"];
const HORSE_ARMOR: [Option<&str>; 4] = [
    None,
    Some("textures/entity/horse/armor/horse_armor_iron.png"),
    Some("textures/entity/horse/armor/horse_armor_gold.png"),
    Some("textures/entity/horse/armor/horse_armor_diamond.png"),
];
const HORSE_ARMOR_ABBR: [&str; 4] = ["", "meo", "goo", "dio"];

/// The source sheets selected by vanilla's `EntityHorse.setHorseTexturePaths`.
///
/// `RenderHorse` binds a generated `LayeredTexture` when [`Self::valid`] is true.  The current
/// entity texture upload path only accepts one static path, so [`Self::fallback`] is the safe
/// base/type sheet for the descriptor path while callers that own texture composition can use
/// [`Self::paths`] and [`Self::cache_key`].  Invalid metadata is intentionally reported instead
/// of indexing past a table; vanilla's `field_175508_bO` takes the same invalid branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HorseTextureSelection {
    /// The colour/base sheet, or `None` for an armour-only non-normal horse layer.
    pub base: Option<&'static str>,
    /// The markings mask, or `None` for marking id zero.
    pub markings: Option<&'static str>,
    /// The armour sheet, or `None` when unarmoured.
    pub armor: Option<&'static str>,
    /// Static renderer fallback when no layered source can be built.
    pub fallback: &'static str,
    /// Whether the source triple is within vanilla's accepted table ranges.
    pub valid: bool,
    cache_key: Option<String>,
}

impl HorseTextureSelection {
    /// Source paths in `LayeredTexture` order: base, markings, armour.
    #[must_use]
    pub fn paths(&self) -> [Option<&'static str>; 3] {
        [self.base, self.markings, self.armor]
    }

    /// The generated texture key used by vanilla's `RenderHorse.field_110852_a` cache.
    #[must_use]
    pub fn cache_key(&self) -> Option<&str> {
        self.cache_key.as_deref()
    }

    /// The texture key a caller should bind after resolving this selection.
    ///
    /// Valid selections name the generated `LayeredTexture` cache entry.  Invalid/non-layered
    /// selections intentionally fall back to the ordinary type sheet, matching
    /// `RenderHorse.getEntityTexture` when it cannot build a composite.
    #[must_use]
    pub fn upload_key(&self) -> &str {
        self.cache_key.as_deref().unwrap_or(self.fallback)
    }

    /// The resource-loader contract for one horse texture upload.
    ///
    /// `layers` is ordered exactly as Java's `LayeredTexture`: base colour, markings, armour.
    /// When [`HorseTextureUploadPlan::layered`] is false, bind [`Self::fallback`] directly and do
    /// not call the compositor. When it is true, load each non-`None` path at the same dimensions,
    /// pass the raw RGBA bytes to [`Self::compose_rgba`], then upload the returned bytes under the
    /// plan's [`HorseTextureUploadPlan::key`].
    #[must_use]
    pub fn upload_plan(&self) -> HorseTextureUploadPlan {
        HorseTextureUploadPlan {
            key: self.upload_key().to_owned(),
            layers: self.paths(),
            fallback: self.fallback,
            layered: self.valid,
        }
    }

    /// Composes loaded RGBA layers using the source-over operation used by `LayeredTexture`.
    ///
    /// The byte arrays are deliberately supplied by the resource loader: this module only owns
    /// selection and cache identity, so no filesystem/GPU work is hidden in the render manager.
    /// A missing/extra layer or a dimension mismatch returns `None` instead of producing a cache
    /// entry with a subtly corrupted sheet.
    #[must_use]
    pub fn compose_rgba(
        &self,
        layers: [Option<&[u8]>; 3],
        width: u32,
        height: u32,
    ) -> Option<Vec<u8>> {
        compose_horse_texture(self, layers, width, height)
    }
}

/// The resource-loader/upload boundary for a selected horse texture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HorseTextureUploadPlan {
    /// Generated cache key for the composed image, or the static fallback path when layered is
    /// false.
    pub key: String,
    /// Source paths in base, markings, armour order.
    pub layers: [Option<&'static str>; 3],
    /// Static type-sheet fallback for invalid or non-layered selections.
    pub fallback: &'static str,
    /// Whether the loader should compose `layers` and upload under `key`.
    pub layered: bool,
}

/// Composes a selected horse's loaded source sheets into one RGBA upload payload.
///
/// This is the CPU half of Java's `LayeredTexture.loadTexture`: every non-`None` source is drawn
/// at `(0, 0)` in base/markings/armour order with source-over alpha.  The caller owns decoding
/// and the eventual `EntityPhase::upload_texture` call.
#[must_use]
pub fn compose_horse_texture(
    selection: &HorseTextureSelection,
    layers: [Option<&[u8]>; 3],
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    if !selection.valid || width == 0 || height == 0 {
        return None;
    }
    let pixels = usize::try_from(width)
        .ok()?
        .checked_mul(usize::try_from(height).ok()?)?;
    let expected_len = pixels.checked_mul(4)?;
    let mut composed = image::RgbaImage::from_pixel(width, height, image::Rgba([0, 0, 0, 0]));

    for (expected_path, layer) in selection.paths().into_iter().zip(layers) {
        if expected_path.is_some() != layer.is_some() {
            return None;
        }
        let Some(raw) = layer else {
            continue;
        };
        if raw.len() != expected_len {
            return None;
        }
        let source = image::RgbaImage::from_raw(width, height, raw.to_vec())?;
        image::imageops::overlay(&mut composed, &source, 0, 0);
    }
    Some(composed.into_raw())
}

fn horse_type_fallback(horse_type: i32) -> &'static str {
    match horse_type {
        1 => "textures/entity/horse/donkey.png",
        2 => "textures/entity/horse/mule.png",
        3 => "textures/entity/horse/horse_zombie.png",
        4 => "textures/entity/horse/horse_skeleton.png",
        // java: RenderHorse's `default` switch arm is the white horse sheet.
        _ => HORSE_TEXTURES[0],
    }
}

fn invalid_horse_texture_selection(horse_type: i32) -> HorseTextureSelection {
    HorseTextureSelection {
        base: None,
        markings: None,
        armor: None,
        fallback: horse_type_fallback(horse_type),
        valid: false,
        cache_key: None,
    }
}

/// Selects the exact horse colour/marking/armour source sheets and cache key.
///
/// java: `EntityHorse.setHorseTexturePaths` plus `RenderHorse.getEntityTexture`.
/// `packed_variant` is `getHorseVariant()` (low byte = colour, high byte = markings); `armor`
/// is `getHorseArmorIndexSynced()` (0 none, 1 iron, 2 gold, 3 diamond).  The normal horse always
/// uses a layered source, including the unarmoured case.  Donkeys, mules and undead horses use a
/// type sheet without armour, and an armour value switches them to an armour-only layer exactly
/// as vanilla's `horseTexturesArray[0] = ""` does.
#[must_use]
pub fn horse_texture_selection(
    horse_type: i32,
    packed_variant: i32,
    armor: i32,
) -> HorseTextureSelection {
    let fallback = horse_type_fallback(horse_type);
    if horse_type == 0 {
        let colour = (packed_variant & 255) as usize;
        let marking = ((packed_variant & 65_280) >> 8) as usize;
        if colour >= HORSE_TEXTURES.len()
            || marking >= HORSE_MARKINGS.len()
            || !(0..HORSE_ARMOR.len() as i32).contains(&armor)
        {
            return invalid_horse_texture_selection(horse_type);
        }
        return HorseTextureSelection {
            base: Some(HORSE_TEXTURES[colour]),
            markings: HORSE_MARKINGS[marking],
            armor: HORSE_ARMOR[armor as usize],
            fallback,
            valid: true,
            cache_key: Some(format!(
                "horse/{}{}{}",
                HORSE_TEXTURE_ABBR[colour],
                HORSE_MARKING_ABBR[marking],
                HORSE_ARMOR_ABBR[armor as usize]
            )),
        };
    }

    if !(1..=4).contains(&horse_type) || !(0..HORSE_ARMOR.len() as i32).contains(&armor) {
        return invalid_horse_texture_selection(horse_type);
    }
    if armor == 0 {
        // java: `func_110239_cn()` is false for non-normal unarmoured horses, so the type sheet
        // is returned directly and no LayeredTexture cache entry is created.
        return invalid_horse_texture_selection(horse_type);
    }

    HorseTextureSelection {
        base: None,
        markings: None,
        armor: HORSE_ARMOR[armor as usize],
        fallback,
        valid: true,
        cache_key: Some(format!(
            "horse/_{horse_type}_{}",
            HORSE_ARMOR_ABBR[armor as usize]
        )),
    }
}

macro_rules! r {
    ($model:expr, $tex:literal, $shadow:expr, $scale:expr, $layers:expr) => {
        r!($model, $tex, $shadow, $scale, $layers, 90.0)
    };
    ($model:expr, $tex:literal, $shadow:expr, $scale:expr, $layers:expr, $death:expr) => {
        EntityRenderer {
            model: $model,
            texture: $tex,
            shadow_size: $shadow,
            scale: $scale,
            death_max_rotation: $death,
            layers: $layers,
        }
    };
}

/// The renderer for an entity type, or `None` for types this section does not draw.
///
/// java: `RenderManager.getEntityRenderObject`. `None` is not a bug: `Mob` and `Monster` are
/// abstract `EntityList` entries no server ever spawns. The dragon, Wither and horse have real
/// model/renderer descriptors; their remaining state-dependent details are separate per-frame
/// seams rather than missing registrations. The dragon's eye/death layers, beam, and corpse
/// rotation still need renderer/client state that does not belong in this static table.
#[must_use]
pub fn renderer_for(t: EntityType) -> Option<&'static EntityRenderer> {
    use EntityType as E;
    use ModelKind as M;
    // Values are from RenderManager's constructor and each Render* class's own `super(...)`.
    // `FastMap`: probed 2+ times per rendered entity per frame with a
    // one-byte key, where SipHash's fixed cost dwarfed the probe. Never iterated.
    static TABLE: std::sync::OnceLock<mc_core::FastMap<EntityType, EntityRenderer>> =
        std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut m = mc_core::FastMap::default();
        // --- hostile ---
        m.insert(
            E::Creeper,
            r!(
                M::Creeper,
                "textures/entity/creeper/creeper.png",
                0.5,
                1.0,
                &[LayerKind::CreeperCharge]
            ),
        );
        m.insert(
            E::Spider,
            r!(
                M::Spider,
                "textures/entity/spider/spider.png",
                1.0,
                1.0,
                &[LayerKind::SpiderEyes],
                // java: `RenderSpider.getDeathMaxRotation` (`:18`) returns 180 — a dying spider
                // flips right onto its back rather than falling on its side.
                180.0
            ),
        );
        m.insert(
            E::CaveSpider,
            r!(
                M::Spider,
                "textures/entity/spider/cave_spider.png",
                // java: `RenderCaveSpider`'s constructor is `super(renderManagerIn)` followed by
                // `this.shadowSize *= 0.7F`, so it inherits `RenderSpider`'s 1.0 and scales it —
                // the shadow shrinks with the body, it does not stay spider-sized.
                0.7,
                0.7,
                &[LayerKind::SpiderEyes],
                // java: inherited from `RenderSpider`, which `RenderCaveSpider` extends.
                180.0
            ),
        );
        m.insert(
            E::Zombie,
            r!(
                M::Zombie,
                "textures/entity/zombie/zombie.png",
                0.5,
                1.0,
                BIPED_LAYERS
            ),
        );
        m.insert(
            E::Giant,
            r!(
                M::Zombie,
                "textures/entity/zombie/zombie.png",
                // java: `RenderManager:162` constructs `new RenderGiantZombie(this,
                // new ModelZombie(), 0.5F, 6.0F)` and `RenderGiantZombie:20` passes
                // `shadowSizeIn * scaleIn` — 0.5 x 6.0 — to its super, *not* `shadowSizeIn`. The
                // shadow is sized for the six-block body, not for a normal zombie.
                3.0,
                6.0,
                GIANT_LAYERS
            ),
        );
        m.insert(
            E::PigZombie,
            r!(
                M::Zombie,
                "textures/entity/zombie_pigman.png",
                0.5,
                1.0,
                BIPED_LAYERS
            ),
        );
        m.insert(
            E::Skeleton,
            r!(
                M::Skeleton,
                "textures/entity/skeleton/skeleton.png",
                0.5,
                1.0,
                SKELETON_LAYERS
            ),
        );
        m.insert(
            E::Enderman,
            r!(
                M::Enderman,
                "textures/entity/enderman/enderman.png",
                0.5,
                1.0,
                &[LayerKind::EndermanEyes, LayerKind::HeldBlock]
            ),
        );
        m.insert(
            E::Endermite,
            r!(
                M::Endermite,
                "textures/entity/endermite.png",
                0.3,
                1.0,
                NO_LAYERS,
                // java: `RenderEndermite.getDeathMaxRotation` (`:16`) returns 180.
                180.0
            ),
        );
        m.insert(
            E::Silverfish,
            r!(
                M::Silverfish,
                "textures/entity/silverfish.png",
                0.3,
                1.0,
                NO_LAYERS,
                // java: `RenderSilverfish.getDeathMaxRotation` (`:16`) returns 180.
                180.0
            ),
        );
        m.insert(
            E::Blaze,
            r!(M::Blaze, "textures/entity/blaze.png", 0.5, 1.0, NO_LAYERS),
        );
        m.insert(
            E::Slime,
            r!(
                M::Slime,
                "textures/entity/slime/slime.png",
                0.25,
                1.0,
                &[LayerKind::SlimeGel]
            ),
        );
        m.insert(
            E::LavaSlime,
            r!(
                M::MagmaCube,
                "textures/entity/slime/magmacube.png",
                0.25,
                1.0,
                NO_LAYERS
            ),
        );
        m.insert(
            E::Ghast,
            r!(
                M::Ghast,
                "textures/entity/ghast/ghast.png",
                0.5,
                // java: `RenderGhast.preRenderCallback` (`:30`) — `f = 1.0F`, so both
                // `f1 = (8 + f) / 2` and `f2 = (8 + 1/f) / 2` come out at 4.5 and the
                // `scale(f2, f1, f2)` is uniform. A ghast is a 4-block mob; at 1.0 it drew as a
                // single cube adrift inside its own hitbox.
                4.5,
                NO_LAYERS
            ),
        );
        m.insert(
            E::Witch,
            r!(
                M::Witch,
                "textures/entity/witch.png",
                0.5,
                0.9375,
                &[LayerKind::WitchHeldItem]
            ),
        );
        m.insert(
            E::Guardian,
            EntityRenderer {
                model: M::Guardian,
                texture: "textures/entity/guardian.png",
                shadow_size: 0.5,
                scale: 1.0,
                // java: `RenderGuardian` does **not** override `getDeathMaxRotation` — grepping
                // the whole `client/renderer/entity/` package finds exactly three overrides
                // (`RenderSpider`, `RenderSilverfish`, `RenderEndermite`, plus `RenderCaveSpider`
                // by inheritance), so the guardian takes `RendererLivingEntity`'s 90
                // (`RendererLivingEntity.java:473`). The 180 that used to be here, and the
                // comment asserting the override existed, were both wrong.
                death_max_rotation: 90.0,
                layers: NO_LAYERS,
            },
        );
        m.insert(
            E::EnderDragon,
            r!(
                M::Dragon,
                "textures/entity/enderdragon/dragon.png",
                0.5,
                1.0,
                DRAGON_LAYERS
            ),
        );
        m.insert(
            E::WitherBoss,
            r!(
                M::Wither,
                "textures/entity/wither/wither.png",
                1.0,
                // java: `RenderWither.preRenderCallback` starts at a twofold scale. The small
                // invulnerability shrink needs the live watcher/partial-tick seam; this is the
                // settled (and overwhelmingly common) value.
                2.0,
                &[LayerKind::WitherAura]
            ),
        );
        // --- passive ---
        m.insert(
            E::Pig,
            r!(
                M::Pig,
                "textures/entity/pig/pig.png",
                0.7,
                1.0,
                &[LayerKind::Saddle]
            ),
        );
        m.insert(
            E::Sheep,
            r!(
                M::Sheep,
                "textures/entity/sheep/sheep.png",
                0.7,
                1.0,
                &[LayerKind::SheepWool]
            ),
        );
        m.insert(
            E::Cow,
            r!(M::Cow, "textures/entity/cow/cow.png", 0.7, 1.0, NO_LAYERS),
        );
        m.insert(
            E::MushroomCow,
            r!(
                M::Cow,
                "textures/entity/cow/mooshroom.png",
                0.7,
                1.0,
                &[LayerKind::MooshroomMushroom]
            ),
        );
        m.insert(
            E::Chicken,
            r!(
                M::Chicken,
                "textures/entity/chicken.png",
                0.3,
                1.0,
                NO_LAYERS
            ),
        );
        m.insert(
            E::Wolf,
            r!(
                M::Wolf,
                "textures/entity/wolf/wolf.png",
                0.5,
                1.0,
                &[LayerKind::WolfCollar]
            ),
        );
        m.insert(
            E::Ozelot,
            r!(
                M::Ocelot,
                "textures/entity/cat/ocelot.png",
                0.4,
                1.0,
                NO_LAYERS
            ),
        );
        m.insert(
            E::Rabbit,
            r!(
                M::Rabbit,
                "textures/entity/rabbit/brown.png",
                0.3,
                1.0,
                NO_LAYERS
            ),
        );
        m.insert(
            E::Squid,
            r!(M::Squid, "textures/entity/squid.png", 0.7, 1.0, NO_LAYERS),
        );
        m.insert(
            E::Bat,
            // java: `RenderBat.preRenderCallback` (`:32`) — `scale(0.35F, 0.35F, 0.35F)`. At 1.0
            // a bat drew nearly as large as a player.
            r!(M::Bat, "textures/entity/bat.png", 0.25, 0.35, NO_LAYERS),
        );
        m.insert(
            E::SnowMan,
            r!(
                M::SnowMan,
                "textures/entity/snowman.png",
                0.5,
                1.0,
                &[LayerKind::SnowmanHead]
            ),
        );
        m.insert(
            E::VillagerGolem,
            r!(
                M::IronGolem,
                "textures/entity/iron_golem.png",
                0.5,
                1.0,
                &[LayerKind::IronGolemFlower]
            ),
        );
        m.insert(
            E::Villager,
            r!(
                M::Villager,
                "textures/entity/villager/villager.png",
                0.5,
                0.9375,
                // java: `RenderVillager`'s constructor (`:19-20`) has exactly one `addLayer`, and
                // it is `new LayerCustomHead(getMainModel().villagerHead)`. A villager has **no**
                // armour layer at all — `ModelVillager` is not a `ModelBiped`, so drawing a biped
                // armour pass over it put a visibly mismatched mesh on every villager.
                &[LayerKind::CustomHead]
            ),
        );
        m.insert(
            E::EntityHorse,
            r!(
                M::Horse,
                "textures/entity/horse/horse_white.png",
                0.75,
                1.0,
                NO_LAYERS
            ),
        );
        // --- objects ---
        m.insert(
            E::ArmorStand,
            r!(
                M::ArmorStand,
                "textures/entity/armorstand/wood.png",
                0.0,
                1.0,
                ARMOR_STAND_LAYERS,
                // java: `ArmorStandRenderer.rotateCorpse` (`:47-50`) overrides the base method
                // with `rotate(180 - bodyYaw, Y)` and **nothing else** — no `deathTime` branch,
                // so `getDeathMaxRotation`'s 90 degrees is never read for a stand and a broken
                // one stays upright while it fades. This row fell through to the macro's 90.0
                // default and rolled every dying armour stand onto its side.
                0.0
            ),
        );
        m.insert(
            E::Boat,
            r!(M::Boat, "textures/entity/boat.png", 0.5, 1.0, NO_LAYERS),
        );
        m.insert(
            E::LeashKnot,
            r!(
                M::LeashKnot,
                "textures/entity/lead_knot.png",
                0.0,
                1.0,
                NO_LAYERS
            ),
        );
        m.insert(
            E::EnderCrystal,
            r!(
                M::EnderCrystal,
                "textures/entity/endercrystal/endercrystal.png",
                0.5,
                1.0,
                NO_LAYERS
            ),
        );
        for kind in [
            E::MinecartRideable,
            E::MinecartChest,
            E::MinecartFurnace,
            E::MinecartTNT,
            E::MinecartHopper,
            E::MinecartSpawner,
            E::MinecartCommandBlock,
        ] {
            m.insert(
                kind,
                r!(
                    M::Minecart,
                    "textures/entity/minecart.png",
                    0.5,
                    1.0,
                    NO_LAYERS
                ),
            );
        }
        m
    });
    table.get(&t)
}

/// The renderer descriptor after applying an entity's watched render variant.
///
/// Most 1.8.9 renderers only switch texture, so [`renderer_for`] remains the descriptor for
/// those variants. The stateful exceptions are kept here as well: `RenderZombie.func_82427_a`
/// selects a different main model/layer stack for villagers, `RenderGuardian.preRenderCallback`
/// applies the elder scale, `RenderSkeleton.preRenderCallback` enlarges wither skeletons, and
/// `RenderOcelot.preRenderCallback` shrinks tamed cats. Keeping those decisions beside the
/// descriptor table prevents a texture-only caller from drawing the right pixels with the wrong
/// geometry.
///
/// `variant == 1` is `mc_client`'s decoded `Watched::is_zombie_villager` (the villager flag in
/// `EntityZombie`'s `DataWatcher`).
#[must_use]
pub fn renderer_for_variant(t: EntityType, variant: i32) -> Option<&'static EntityRenderer> {
    use EntityType as E;
    use ModelKind as M;

    static ZOMBIE_VILLAGER: std::sync::OnceLock<EntityRenderer> = std::sync::OnceLock::new();
    static ELDER_GUARDIAN: std::sync::OnceLock<EntityRenderer> = std::sync::OnceLock::new();
    static WITHER_SKELETON: std::sync::OnceLock<EntityRenderer> = std::sync::OnceLock::new();
    static TAMED_OCELOT: std::sync::OnceLock<EntityRenderer> = std::sync::OnceLock::new();
    static DONKEY: std::sync::OnceLock<EntityRenderer> = std::sync::OnceLock::new();
    static MULE: std::sync::OnceLock<EntityRenderer> = std::sync::OnceLock::new();
    match (t, variant) {
        (E::Zombie, 1) => Some(ZOMBIE_VILLAGER.get_or_init(|| {
            r!(
                M::ZombieVillager,
                "textures/entity/zombie/zombie_villager.png",
                0.5,
                1.0,
                ZOMBIE_VILLAGER_LAYERS
            )
        })),
        // java: `RenderGuardian.preRenderCallback` (`:33-35`) scales an elder guardian by
        // `2.35F` before the ordinary model draw. `EntityGuardian.isElder()` is decoded by the
        // watcher seam as render variant 1; the ordinary guardian remains at 1.0.
        (E::Guardian, 1) => Some(ELDER_GUARDIAN.get_or_init(|| {
            r!(
                M::Guardian,
                "textures/entity/guardian.png",
                0.5,
                2.35,
                NO_LAYERS
            )
        })),
        // java: `RenderSkeleton.preRenderCallback` (`:43-46`) scales skeleton type 1 by
        // `1.2F`; `EntitySkeleton.getSkeletonType()` is the watcher value represented here.
        // The wither texture is selected independently by `texture_for`.
        (E::Skeleton, 1) => Some(WITHER_SKELETON.get_or_init(|| {
            r!(
                M::Skeleton,
                "textures/entity/skeleton/wither_skeleton.png",
                0.5,
                1.2,
                SKELETON_LAYERS
            )
        })),
        // java: `RenderOcelot.preRenderCallback` (`:37-40`) scales a tamed cat by `0.8F`.
        // The three non-zero skin values all imply the tamed branch in vanilla's normal
        // metadata flow; wild skin 0 deliberately falls through to the 1.0 descriptor.
        (E::Ozelot, 1..=3) => Some(TAMED_OCELOT.get_or_init(|| {
            r!(
                M::Ocelot,
                "textures/entity/cat/ocelot.png",
                0.4,
                0.8,
                NO_LAYERS
            )
        })),
        // java: `RenderHorse.preRenderCallback` (`:30-45`) scales donkeys by 0.87 and mules by
        // 0.92.  The horse type is the watched byte represented by `variant` at this boundary;
        // zombie and skeleton horses retain the ordinary 1.0 scale.
        (E::EntityHorse, 1) => Some(DONKEY.get_or_init(|| {
            r!(
                M::Horse,
                "textures/entity/horse/donkey.png",
                0.75,
                0.87,
                NO_LAYERS
            )
        })),
        (E::EntityHorse, 2) => Some(MULE.get_or_init(|| {
            r!(
                M::Horse,
                "textures/entity/horse/mule.png",
                0.75,
                0.92,
                NO_LAYERS
            )
        })),
        _ => renderer_for(t),
    }
}

/// The renderer for a remote player, keyed by arm width.
///
/// java: `RenderManager`'s two player entries — `playerRenderers.put("default", new
/// RenderPlayer(this, false))` and `.put("slim", new RenderPlayer(this, true))` — which vanilla
/// reaches through a *separate* map from `entityRenderMap`, consulted first by
/// `RenderManager.getEntityRenderObject`'s `instanceof AbstractClientPlayer` check, because
/// `EntityList` never registers the player class at all (`NetHandlerPlayClient.
/// handleSpawnPlayer` constructs `EntityOtherPlayerMP` directly). [`EntityKind::Player`] is this
/// crate's equivalent of that fork: there is no [`EntityType`] to hand [`renderer_for`], so this
/// is a parallel entry point keyed on [`SkinModel`] (arm width) instead.
///
/// Unlike [`renderer_for`] this never misses — every player has arms of one width or the
/// other — so the return is not an `Option`.
///
/// This resolves only the static arm-geometry choice, never a texture: the actual skin image is
/// [`super::skin::SkinCache`]'s job entirely, fetched and cached per player and substituted for
/// [`SkinModel::default_texture`] once it lands, exactly as vanilla's `SkinManager` does. That
/// split already exists elsewhere in this module tree —
/// [`LayerKind::Deadmau5Head`](super::layers::LayerKind::Deadmau5Head) keys its own body-shaped
/// layer purely off `ModelKind::Player`/`PlayerSlim` and likewise leaves the texture to the
/// caller.
///
/// # Wiring in `mc-client`
///
/// `mc-client`'s `driver.rs::upload_entities` resolves a remote player's profile model before
/// calling this function, and keeps [`SkinModel::default_for_uuid`] as the descriptor fallback
/// until the same asynchronous skin callback makes the profile pixels ready. This preserves the
/// Java ordering: the `NetworkPlayerInfo` model and `locationSkin` become visible together, while
/// the renderer cache remains a two-entry classic/slim map.
#[must_use]
pub fn renderer_for_player(model: SkinModel) -> &'static EntityRenderer {
    use ModelKind as M;
    // java: `RenderPlayer`'s constructor, in the order it calls `addLayer`: armour, held item,
    // stuck arrows, a custom (skull/block) head, the Deadmau5 easter egg, then the cape.
    // `RenderPlayer` never overrides `shadowSize` beyond the `0.5F` it passes `super(...)`. It
    // *does* apply a `preRenderCallback` scale — `RenderPlayer.java:133-137` is
    // `float f = 0.9375F; GlStateManager.scale(f, f, f)` — which this table used to record as
    // `1.0`, with a comment claiming there was no such callback. Both were wrong; every player
    // body drew 6.7% too large.
    // java: `RenderPlayer`'s constructor (`:35-40`) — armour, held item, arrow, deadmau5 head,
    // cape, custom head, in that order. `CustomHead` used to sit third from last here; it is the
    // *last* `addLayer`, which matters as soon as blending exists, since a worn skull
    // must draw over the cape rather than under it.
    const PLAYER_LAYERS: &[LayerKind] = &[
        LayerKind::Armor,
        LayerKind::HeldItem,
        LayerKind::Arrow,
        LayerKind::Deadmau5Head,
        LayerKind::Cape,
        LayerKind::CustomHead,
    ];
    static CLASSIC: std::sync::OnceLock<EntityRenderer> = std::sync::OnceLock::new();
    static SLIM: std::sync::OnceLock<EntityRenderer> = std::sync::OnceLock::new();
    match model {
        SkinModel::Classic => CLASSIC.get_or_init(|| {
            r!(
                M::Player,
                "textures/entity/steve.png",
                0.5,
                0.9375,
                PLAYER_LAYERS
            )
        }),
        SkinModel::Slim => SLIM.get_or_init(|| {
            r!(
                M::PlayerSlim,
                "textures/entity/alex.png",
                0.5,
                0.9375,
                PLAYER_LAYERS
            )
        }),
    }
}

/// One dispatch point for every drawable entity, mob or player.
///
/// java: conceptually `RenderManager.getEntityRenderObject`, which forks on `instanceof
/// AbstractClientPlayer` *before* ever consulting `entityRenderMap` — the same fork
/// [`mc_world::EntityKind`] already makes explicit at the type level instead of a runtime
/// `instanceof`. This function is that fork, reunified: [`EntityKind::Type`] goes to
/// [`renderer_for`], [`EntityKind::Player`] goes to [`renderer_for_player`].
///
/// `player_model` is only consulted for [`EntityKind::Player`]; it is ignored for
/// [`EntityKind::Type`], so a caller walking a mixed set of tracked entities (mobs and players
/// together) does not need to special-case which kind it has before calling this — see the doc
/// on [`renderer_for_player`] for exactly how `mc-client` should eventually supply it.
#[must_use]
pub fn renderer_for_kind(
    kind: EntityKind,
    player_model: SkinModel,
) -> Option<&'static EntityRenderer> {
    match kind {
        EntityKind::Type(t) => renderer_for(t),
        EntityKind::Player => Some(renderer_for_player(player_model)),
        // EntityFishHook is emitted by the dedicated object path, not a descriptor-backed
        // living renderer.
        EntityKind::FishingBobber => None,
    }
}

/// The texture for an entity type given its variant byte.
///
/// java: each renderer's `getEntityTexture`, which switches on a metadata field —
/// `EntityVillager.getProfession`, `EntityOcelot.getTameSkin`, `EntityRabbit.getRabbitType`,
/// `EntitySkeleton.getSkeletonType`, `EntityWolf.isAngry`/`isTamed` and
/// `EntityHorse.getHorseType`. Anything unrecognised falls back to the descriptor's default,
/// which is also what vanilla's `default` arms do.  A horse's packed colour/marking/armour
/// composite is exposed separately by [`horse_texture_selection`].
#[must_use]
pub fn texture_for(t: EntityType, variant: i32) -> &'static str {
    use EntityType as E;
    match (t, variant) {
        (E::Villager, 0) => "textures/entity/villager/farmer.png",
        (E::Villager, 1) => "textures/entity/villager/librarian.png",
        (E::Villager, 2) => "textures/entity/villager/priest.png",
        (E::Villager, 3) => "textures/entity/villager/smith.png",
        (E::Villager, 4) => "textures/entity/villager/butcher.png",
        (E::Ozelot, 1) => "textures/entity/cat/black.png",
        (E::Ozelot, 2) => "textures/entity/cat/red.png",
        (E::Ozelot, 3) => "textures/entity/cat/siamese.png",
        (E::Rabbit, 0) => "textures/entity/rabbit/brown.png",
        (E::Rabbit, 1) => "textures/entity/rabbit/white.png",
        (E::Rabbit, 2) => "textures/entity/rabbit/black.png",
        (E::Rabbit, 3) => "textures/entity/rabbit/white_splotched.png",
        (E::Rabbit, 4) => "textures/entity/rabbit/gold.png",
        (E::Rabbit, 5) => "textures/entity/rabbit/salt.png",
        // java: `RenderRabbit.getEntityTexture` (`:57-58`) — type 99 is `CAERBANNOG`, the Killer
        // Bunny. `TOAST` is not a type at all: it is a *name* check
        // (`getName().equals("Toast")`, `:30`) applied before the switch, which is why it lives in
        // [`texture_for_named`] instead of here.
        (E::Rabbit, 99) => "textures/entity/rabbit/caerbannog.png",
        (E::Skeleton, 1) => "textures/entity/skeleton/wither_skeleton.png",
        // java: RenderWolf binds wolf_angry when angry and wolf_tame when tamed.
        (E::Wolf, 1) => "textures/entity/wolf/wolf_tame.png",
        (E::Wolf, 2) => "textures/entity/wolf/wolf_angry.png",
        // A02-14's four missing variants. Each is a `getEntityTexture` switch on a watched value
        // that had no arm here at all, so the entity always drew its default sheet.
        //
        // java: `RenderZombie.getEntityTexture` (`:59`) — `isVillager()`. **The texture is only
        // half of it**: the same test picks `ModelZombieVillager` over `ModelZombie` and a
        // different layer list (`:46-50`, `LayerCustomHead` on the villager head plus
        // `LayerVillagerArmor`), which needs its own `ModelKind`; this arm fixes the skin only.
        (E::Zombie, 1) => "textures/entity/zombie/zombie_villager.png",
        // java: `RenderGhast.getEntityTexture` (`:20`) — `entity.isAttacking()`, the charging
        // ghast's red-eyed sheet, which is the only warning a fireball is coming.
        (E::Ghast, 1) => "textures/entity/ghast/ghast_shooting.png",
        // java: `RenderGuardian.getEntityTexture` (`:129`) — `isElder()`.
        (E::Guardian, 1) => "textures/entity/guardian_elder.png",
        // java: `RenderWither.getEntityTexture` (`:26`) — `getInvulTime() > 0`, the spawn
        // animation's skeletal sheet. The watched invulnerability value is still a caller-owned
        // seam, so this arm remains reachable to a variant-aware caller even though the generic
        // descriptor defaults to the ordinary sheet.
        (E::WitherBoss, 1) => "textures/entity/wither/wither_invulnerable.png",
        // java: `RenderHorse.getEntityTexture` (`:50-75`) when no generated layered texture is
        // requested.  The generic variant at this boundary is `getHorseType()`; colour,
        // markings and armour are selected by [`horse_texture_selection`].
        (E::EntityHorse, 1) => "textures/entity/horse/donkey.png",
        (E::EntityHorse, 2) => "textures/entity/horse/mule.png",
        (E::EntityHorse, 3) => "textures/entity/horse/horse_zombie.png",
        (E::EntityHorse, 4) => "textures/entity/horse/horse_skeleton.png",
        _ => renderer_for(t).map_or("textures/entity/steve.png", |r| r.texture),
    }
}

/// The texture for an entity type, including the two vanilla renderers that switch on the
/// entity's **custom name** rather than on a watched value.
///
/// java: `RenderRabbit.getEntityTexture` (`:29-33`) tests
/// `EnumChatFormatting.getTextWithoutFormattingCodes(entity.getName()).equals("Toast")` *before*
/// the `getRabbitType()` switch, so a rabbit named Toast wears `toast.png` whatever its type; and
/// `RendererLivingEntity.renderModel` applies the Dinnerbone/Grumm easter egg from the same
/// `getName()`. Everything else routes straight through [`texture_for`].
///
/// `name` is the entity's custom name with formatting codes already stripped — vanilla strips
/// them with `EnumChatFormatting.getTextWithoutFormattingCodes`, which is the caller's job here
/// because `mc-render` does not own the chat-formatting parser.
#[must_use]
pub fn texture_for_named(t: EntityType, variant: i32, name: Option<&str>) -> &'static str {
    if t == EntityType::Rabbit && name == Some("Toast") {
        return "textures/entity/rabbit/toast.png";
    }
    texture_for(t, variant)
}

/// Selects the outer placement matrix for a renderer-table entity.
///
/// The descriptor table contains four model-backed objects whose renderers do **not** extend
/// `RendererLivingEntity`: boats, every minecart subtype, leash knots, and end crystals.  They
/// must therefore not inherit [`entity_matrix`]'s corpse turn, model flip, static scale, or
/// `-1.5078125` living-model lift.  Their Java renderers each have a small bespoke transform,
/// represented by the helpers in [`super::objects`].
///
/// `EntityTransform` supplies only the common packet-derived position and yaw.  The object
/// renderers' additional live inputs (boat/cart damage rocking and minecart rail pitch) belong
/// to the client entity submission path, so this manager-level dispatch uses their settled
/// values.  A caller with those inputs may call the corresponding object helper directly.
///
/// The fallback is intentionally the living transform: this function is the safe replacement
/// for the ordinary descriptor-table path after its caller has applied the descriptor's static
/// scale.  Slimes and magma cubes still require their separate non-uniform scale branch.
#[must_use]
pub fn placement_matrix(t: EntityType, transform: &EntityTransform) -> Mat4 {
    use EntityType as E;

    match t {
        // java: `RenderBoat.doRender`.
        E::Boat => boat_matrix(transform.rel, transform.body_yaw, 0.0, 0.0, 1, 0.0),
        // java: all seven `RenderMinecart` subclasses inherit `RenderMinecart.doRender`.
        E::MinecartRideable
        | E::MinecartChest
        | E::MinecartFurnace
        | E::MinecartTNT
        | E::MinecartHopper
        | E::MinecartSpawner
        | E::MinecartCommandBlock => {
            minecart_matrix(transform.rel, transform.body_yaw, 0.0, 0.0, 0.0, 1, 0.0)
        }
        // java: `RenderLeashKnot.doRender` has neither a yaw turn nor model lift.
        E::LeashKnot => leash_knot_matrix(transform.rel),
        // java: `RenderEnderCrystal.doRender` is a translation only.
        E::EnderCrystal => ender_crystal_matrix(transform.rel),
        // java: `ArmorStandRenderer.rotateCorpse` overrides the living implementation and
        // never applies the death roll. Keep this true even for a direct helper caller that has
        // not first selected the descriptor (whose death_max_rotation is already zero).
        E::ArmorStand => {
            let mut stand_transform = *transform;
            stand_transform.death_rotation = 0.0;
            entity_matrix(&stand_transform)
        }
        // java: `RenderSquid.rotateCorpse` adds the fixed translations and rotation order. The
        // dynamic squid pitch/yaw are supplied by `squid_placement_matrix`; a neutral call here
        // is still more faithful than inheriting the ordinary corpse transform.
        E::Squid => squid_placement_matrix(transform, 0.0, 0.0),
        _ => entity_matrix(transform),
    }
}

/// Builds `RenderSquid.rotateCorpse` followed by the ordinary living renderer tail.
///
/// java: `RenderSquid.rotateCorpse` (`:25-34`) translates up by `0.5F`, rotates by
/// `180 - renderYawOffset`, then by the interpolated `squidPitch` and `squidYaw`, and translates
/// down by `1.2F`. `RendererLivingEntity.doRender` then applies the model flip, pre-render scale,
/// and the normal `-1.5078125` model lift. The two squid angles are degrees, matching the Java
/// `GlStateManager.rotate` calls. Death rotation is intentionally ignored because this override
/// never calls `super.rotateCorpse`.
#[must_use]
pub fn squid_placement_matrix(transform: &EntityTransform, pitch: f32, yaw: f32) -> Mat4 {
    let mut m = Mat4::from_translation(transform.rel);
    m *= Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0));
    m *= Mat4::from_rotation_y((180.0 - transform.body_yaw).to_radians());
    m *= Mat4::from_rotation_x(pitch.to_radians());
    m *= Mat4::from_rotation_y(yaw.to_radians());
    m *= Mat4::from_translation(Vec3::new(0.0, -1.2, 0.0));
    m *= Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0));
    if transform.scale != 1.0 {
        m *= Mat4::from_scale(Vec3::splat(transform.scale));
    }
    m *= Mat4::from_translation(Vec3::new(0.0, transform.y_offset, 0.0));
    m
}

/// Selects a species-aware placement matrix for callers that have per-frame animation facts.
///
/// `RenderBat.rotateCorpse` translates by `cos(ageInTicks * 0.3) * 0.1` while flying and by
/// `-0.1` while hanging, before it delegates to `RendererLivingEntity`.
/// `RenderIronGolem.rotateCorpse` follows the common corpse rotation with a phase-shifted
/// walking roll around Z. `RenderZombie.rotateCorpse` perturbs the interpolated body yaw while
/// a villager zombie is converting. The ordinary
/// [`placement_matrix`] entry point deliberately remains state-free for object submissions;
/// this seam carries living transforms whose inputs are represented by [`AnimParams`], including
/// the Squid pitch/yaw samples forwarded to [`squid_placement_matrix`].
#[must_use]
pub fn placement_matrix_for(
    t: EntityType,
    transform: &EntityTransform,
    params: &AnimParams,
) -> Mat4 {
    match t {
        EntityType::EnderDragon => dragon_placement_matrix(
            transform,
            params.entity.dragon.as_ref(),
            params.partial_ticks,
        ),
        EntityType::Squid => squid_placement_matrix(
            transform,
            params.entity.squid_pitch,
            params.entity.squid_yaw,
        ),
        EntityType::Bat => {
            let bob = if params.entity.bat_hanging {
                -0.1
            } else {
                mc_math::cos(params.age_in_ticks * 0.3) * 0.1
            };
            let mut bat_transform = *transform;
            bat_transform.rel.y += bob;
            placement_matrix(t, &bat_transform)
        }
        EntityType::VillagerGolem if params.limb_swing_amount >= 0.01 => {
            // java: `RenderIronGolem.rotateCorpse`. `params.limb_swing` is already
            // `limbSwing - limbSwingAmount * (1 - partialTicks)`, so only Java's `+ 6` phase
            // remains here. Adding to `death_rotation` composes this second Z rotation after
            // the common corpse turn and before the model flip, exactly where Java does it.
            let phase = params.limb_swing + 6.0;
            let wave = ((phase % 13.0 - 6.5).abs() - 3.25) / 3.25;
            let mut golem_transform = *transform;
            golem_transform.death_rotation += 6.5 * wave;
            placement_matrix(t, &golem_transform)
        }
        EntityType::Zombie if params.entity.converting => {
            // java: `RenderZombie.rotateCorpse` adds this *radian-looking numeric value* to a
            // body-yaw argument measured in degrees. The tiny ±PI/4-degree shake is therefore
            // intentional, and the cosine uses the integer `ticksExisted`, not partial ticks.
            let shake = ((f64::from(params.entity.ticks_existed) * 3.25).cos()
                * core::f64::consts::PI
                * 0.25) as f32;
            let mut zombie_transform = *transform;
            zombie_transform.body_yaw += shake;
            placement_matrix(t, &zombie_transform)
        }
        _ => placement_matrix(t, transform),
    }
}

/// `RenderDragon.rotateCorpse` followed by the common living-renderer flip/lift.
///
/// Dragon rotation comes from its ring buffer, not `renderYawOffset`; the renderer's
/// `deathTime` is also intentionally ignored because `EntityDragon` owns the separate
/// 200-tick `deathTicks` sequence and never increments the inherited timer.
#[must_use]
pub fn dragon_placement_matrix(
    transform: &EntityTransform,
    state: Option<&super::anim::DragonRenderState>,
    partial_ticks: f32,
) -> Mat4 {
    let (yaw, pitch) = state.map_or((0.0, 0.0), |dragon| {
        (
            dragon.movement_offset(7, partial_ticks)[0] as f32,
            (dragon.movement_offset(5, partial_ticks)[1]
                - dragon.movement_offset(10, partial_ticks)[1]) as f32
                * 10.0,
        )
    });
    let mut m = Mat4::from_translation(transform.rel);
    m *= Mat4::from_rotation_y(-yaw.to_radians());
    m *= Mat4::from_rotation_x(pitch.to_radians());
    m *= Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0));
    m *= Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0));
    if transform.scale != 1.0 {
        m *= Mat4::from_scale(Vec3::splat(transform.scale));
    }
    m *= Mat4::from_translation(Vec3::new(0.0, transform.y_offset, 0.0));
    m
}

/// `RenderCreeper.preRenderCallback`'s exact non-uniform flash scale.
///
/// java: `RenderCreeper.preRenderCallback` (`:28-37`) computes the sine pulse from the
/// *unclamped* flash intensity, clamps and squares the intensity twice, then applies reciprocal
/// XZ/Y factors. `flash_intensity` is the interpolated result of
/// `EntityCreeper.getCreeperFlashIntensity(partialTicks)` supplied by the caller.
#[must_use]
pub fn creeper_pre_render_scale(flash_intensity: f32) -> Vec3 {
    let f1 = 1.0 + mc_math::sin(flash_intensity * 100.0) * flash_intensity * 0.01;
    let f = flash_intensity.clamp(0.0, 1.0);
    let f = f * f;
    let f = f * f;
    let f2 = (1.0 + f * 0.4) * f1;
    let f3 = (1.0 + f * 0.1) / f1;
    Vec3::new(f2, f3, f2)
}

/// The dynamic, state-dependent part of a renderer's `preRenderCallback` scale.
///
/// The static descriptor scale is selected by [`renderer_for_variant`] and must be composed
/// separately. This function covers the callbacks whose input is available at the render seam:
/// Creeper flash, slime/magma size and squish, and the baby villager multiplier. The exact Java
/// sources are `RenderCreeper.preRenderCallback` (`:28-37`), `RenderSlime.preRenderCallback`
/// (`:29-35`), `RenderMagmaCube.preRenderCallback` (`:29-37`), and `RenderVillager` (`:31-36`).
#[must_use]
pub fn entity_pre_render_scale(
    t: EntityType,
    variant: i32,
    child: bool,
    squish: f32,
    creeper_flash: f32,
) -> Vec3 {
    match t {
        EntityType::Creeper => creeper_pre_render_scale(creeper_flash),
        EntityType::Slime | EntityType::LavaSlime => slime_pre_render_scale(variant, squish),
        EntityType::Villager if child => Vec3::splat(0.5),
        _ => Vec3::ONE,
    }
}

/// Returns the per-frame shadow size for a descriptor, including `RenderSlime`'s size update.
///
/// java: `RenderSlime.doRender` assigns `shadowSize = 0.25F * getSlimeSize()` before delegating
/// to the common renderer. `RenderMagmaCube` has no `doRender` override: its constructor passes
/// `0.25F` to `RenderLiving`, so a magma cube's shadow stays at the descriptor constant even
/// though its body callback scales with size. The ordinary descriptor value is returned for all
/// other types; child-specific shadow policy remains the caller's common `Render` path, just as
/// in Java.
#[must_use]
pub fn shadow_size_for(t: EntityType, variant: i32, child: bool) -> Option<f32> {
    let renderer = renderer_for_variant(t, variant)?;
    Some(match t {
        EntityType::Slime => 0.25 * variant.max(1) as f32,
        EntityType::Villager if child => 0.25,
        _ => renderer.shadow_size,
    })
}

/// One cached model entry, with the concrete horse branch retained for stateful rendering.
enum CachedModel {
    /// The horse has a concrete stateful entry point in addition to the trait object path.
    Horse(HorseModel),
    /// Player bodies retain their concrete model because `LayerCape` renders a part on the same
    /// posed instance as the body (java: `RenderPlayer.mainModel.bipedCape`).
    Player(PlayerModel),
    /// Alex/slim player bodies use the same cape path, with their own arm geometry.
    PlayerSlim(PlayerModel),
    /// All other models use the common trait contract.
    Other(Box<dyn EntityModel>),
}

impl CachedModel {
    fn new(kind: ModelKind) -> Self {
        match kind {
            ModelKind::Horse => Self::Horse(HorseModel::new()),
            ModelKind::Player => Self::Player(PlayerModel::new(0.0, false)),
            ModelKind::PlayerSlim => Self::PlayerSlim(PlayerModel::new(0.0, true)),
            _ => Self::Other(kind.build_default()),
        }
    }

    fn as_entity_model(&mut self) -> &mut dyn EntityModel {
        match self {
            Self::Horse(model) => model,
            Self::Player(model) | Self::PlayerSlim(model) => model,
            Self::Other(model) => model.as_mut(),
        }
    }

    fn as_player_model(&mut self) -> Option<&mut PlayerModel> {
        match self {
            Self::Player(model) | Self::PlayerSlim(model) => Some(model),
            _ => None,
        }
    }
}

/// Owns one built model per [`ModelKind`] and turns entities into vertices.
///
/// java: `RenderManager` — one `Render` per entity class, each holding one `ModelBase`.
#[derive(Default)]
pub struct RenderManager {
    /// This and the three maps below are [`mc_core::FastMap`]s: one
    /// or more lookups per emitted model per frame, small trusted keys, never iterated in an
    /// observable order.
    models: mc_core::FastMap<ModelKind, CachedModel>,
    /// Layer bodies, keyed by `(kind, inflation.to_bits())` — see [`Self::layer_model_mut`].
    layer_models: mc_core::FastMap<(ModelKind, u32), Box<dyn EntityModel>>,
    /// java: `LayerArmorBase`'s two `ModelBiped` instances — `modelArmor` at `1.0F` and
    /// `modelLeggings` at `0.5F`, keyed here by [`ArmorSlot::model_inflation`]'s bit pattern for
    /// the same reason [`Self::layer_models`] is. These cannot live in `layer_models`: that map
    /// is `dyn EntityModel` and [`crate::entity::layers::emit_armor`] needs the concrete
    /// [`BipedModel`] so it can flip part visibility per slot.
    armor_models: mc_core::FastMap<(ArmorModelKind, u32), BipedModel>,
    /// `ArmorStandRenderer.initArmor`'s dedicated `ModelArmorStandArmor` pair.
    armor_stand_models: mc_core::FastMap<u32, ArmorStandArmorModel>,
}

/// One `LayerArrow` sample after selecting a model part and cube with Java's seeded RNG.
#[derive(Clone, Copy, Debug)]
pub struct StuckArrowAttachment {
    /// Index in `ModelBase.boxList`, retained for exact parity tests and diagnostics.
    pub part_index: usize,
    /// Index in the selected `ModelRenderer.cubeList`.
    pub cube_index: usize,
    /// Sampled point inside the cube, in rendered model units.
    pub point: Vec3,
    /// Arrow yaw computed from the negated sample vector, in degrees.
    pub yaw: f32,
    /// Arrow pitch computed from the negated sample vector, in degrees.
    pub pitch: f32,
    /// Entity base, `postRender(0.0625)`, then the sampled local translation.
    pub matrix: Mat4,
}

/// Which of one entity's two armour models [`RenderManager::emit_armor_slot_for_entity`] has
/// posed so far. Start each entity with [`Default`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ArmorPoses {
    posed: [bool; 2],
}

impl ArmorPoses {
    /// `true` the first time `slot`'s model (by [`ArmorSlot::model_inflation`]) is asked for.
    fn first_use(&mut self, slot: ArmorSlot) -> bool {
        let index = usize::from(slot == ArmorSlot::Leggings);
        !core::mem::replace(&mut self.posed[index], true)
    }
}

/// The three `LayerBipedArmor` model families vanilla instantiates.
///
/// `ModelZombie` and `ModelSkeleton(true)` have the standard biped geometry but the zombie arm
/// pose.  `LayerVillagerArmor` additionally lifts the helmet cube to match the villager head.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ArmorModelKind {
    Biped,
    Zombie,
    ZombieVillager,
}

impl ArmorModelKind {
    fn for_body(kind: ModelKind) -> Self {
        match kind {
            ModelKind::ZombieVillager => Self::ZombieVillager,
            ModelKind::Zombie | ModelKind::Skeleton => Self::Zombie,
            _ => Self::Biped,
        }
    }

    fn build(self, inflation: f32) -> BipedModel {
        match self {
            Self::ZombieVillager => super::models::biped::ZombieVillagerModel::armor(inflation),
            Self::Biped | Self::Zombie => BipedModel::new(inflation),
        }
    }
}

impl RenderManager {
    /// An empty manager. Models are built the first time an entity of that kind is drawn.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The shared model for a kind, building it if this is the first use.
    ///
    /// Always built at zero inflation — an entity's own body is never inflated in vanilla, only
    /// a layer's second copy is (see [`Self::layer_model_mut`]).
    pub fn model_mut(&mut self, kind: ModelKind) -> &mut dyn EntityModel {
        self.models
            .entry(kind)
            .or_insert_with(|| CachedModel::new(kind))
            .as_entity_model()
    }

    /// The one shared concrete horse model, building it if this manager has not drawn a horse
    /// yet.  The enum-backed cache keeps this entry identical to [`Self::model_mut`] rather than
    /// creating a second `HorseModel` for the stateful path.
    fn horse_model_mut(&mut self) -> &mut HorseModel {
        let model = self
            .models
            .entry(ModelKind::Horse)
            .or_insert_with(|| CachedModel::new(ModelKind::Horse));
        match model {
            CachedModel::Horse(model) => model,
            CachedModel::Other(_) | CachedModel::Player(_) | CachedModel::PlayerSlim(_) => {
                unreachable!("ModelKind::Horse must cache HorseModel")
            }
        }
    }

    /// The shared model for a [`LayerKind`] that owns separate geometry — the charged-creeper
    /// aura, the sheep's fleece, and so on — built once per `(kind, inflation)` pair and reused,
    /// exactly like [`Self::model_mut`] shares the body model.
    ///
    /// `inflation` is keyed by its bit pattern rather than compared with an epsilon: every real
    /// caller passes one of a handful of `const` literals
    /// ([`LayerKind::model_inflation`](super::layers::LayerKind::model_inflation)), so bit
    /// equality is exact equality here, not a source of cache-key drift.
    pub fn layer_model_mut(&mut self, kind: ModelKind, inflation: f32) -> &mut dyn EntityModel {
        self.layer_models
            .entry((kind, inflation.to_bits()))
            .or_insert_with(|| kind.build(inflation))
            .as_mut()
    }

    /// Emits one [`LayerKind`]'s geometry for the entity currently being drawn.
    ///
    /// Layers whose [`LayerKind::model`] is `Some` own separate geometry — java: `LayerCreeper
    /// Charge`, `LayerSheepWool`, `LayerSnowmanHead`/`LayerCustomHead` and `LayerDeadmau5Head`
    /// each hold a second `ModelBase` of their own — built at [`LayerKind::model_inflation`] via
    /// [`Self::layer_model_mut`] and posed with the same `params`/`base` the body used this
    /// frame, which is exactly what each of those Java layers does with its own model instance.
    ///
    /// Layers whose [`LayerKind::model`] is `None` but which still bind a fixed
    /// [`LayerKind::texture`] (glowing eyes, a saddle, a wolf's collar, a mooshroom's mushroom,
    /// the wither's aura) draw the entity's *own* already-posed geometry a second time under a
    /// different texture — java: `LayerEndermanEyes`, `LayerSaddle` and the rest call
    /// `getMainModel().render(...)` again rather than owning a model of their own — so `body`
    /// (the vertices [`Self::emit`]/[`Self::emit_with`] already produced this frame) is copied
    /// verbatim rather than re-walking the part tree.
    ///
    /// Armour, held items, held blocks, the witch's item and the cape are **not** handled here:
    /// each needs data this manager does not have (an equipped item's material or baked model, a
    /// player's cape state). See [`crate::entity::layers::emit_armor`] and
    /// [`crate::entity::layers::emit_cape`]; held items/blocks are the open `mc-assets` seam the
    /// module docs describe.
    pub fn emit_layer(
        &mut self,
        layer: LayerKind,
        body: &[EntityVertex],
        params: &AnimParams,
        base: Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        match layer.model() {
            Some(kind) => {
                self.layer_model_mut(kind, layer.model_inflation())
                    .pose_and_render(params, base, out);
            }
            // `LayerEnderDragonDeath` owns untextured shard fans, which the client submits
            // through `emit_dragon_death_shards` after this manager has emitted the body. It must
            // not duplicate the normal textured body when a caller walks the descriptor list.
            None if layer == LayerKind::DragonDeath => {}
            None => out.extend_from_slice(body),
        }
    }

    /// Emits the two skin-textured ears from `LayerDeadmau5Head`.
    ///
    /// Java draws `ModelPlayer.bipedDeadmau5Head` twice from the already-established player
    /// body matrix.  The layer first follows the interpolated head yaw/pitch, translates one
    /// ear to either side and up, then cancels those rotations before rendering the flat box at
    /// `4/3` scale.  The account-name, loaded-skin and invisibility gates belong to the client
    /// caller because this geometry manager deliberately has no profile state.
    pub fn emit_deadmau5_ears(
        &mut self,
        kind: ModelKind,
        params: &AnimParams,
        base: Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        if !matches!(kind, ModelKind::Player | ModelKind::PlayerSlim) {
            return;
        }

        let model = self.layer_model_mut(kind, 0.0);
        let Some(ear) = model.model().find("deadmau5") else {
            return;
        };
        model.model_mut().get_mut(ear).show_model = true;

        let yaw = params.net_head_yaw.to_radians();
        let pitch = params.head_pitch.to_radians();
        for side in [-1.0_f32, 1.0] {
            let matrix = base
                * Mat4::from_rotation_y(yaw)
                * Mat4::from_rotation_x(pitch)
                * Mat4::from_translation(Vec3::new(0.375 * side, -0.375, 0.0))
                * Mat4::from_rotation_x(-pitch)
                * Mat4::from_rotation_y(-yaw)
                * Mat4::from_scale(Vec3::splat(4.0 / 3.0));
            model.model().emit_part(ear, params.scale, matrix, out);
        }
    }

    /// Emits a player's cape from the same concrete, already posed body model used for this
    /// frame.  Keeping the player model typed in [`CachedModel`] is important: unlike ordinary
    /// layers, vanilla's cape is a hidden `ModelPlayer` part and not a second model instance.
    pub fn emit_cape(
        &mut self,
        kind: ModelKind,
        state: &CapeState,
        base: Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        if !matches!(kind, ModelKind::Player | ModelKind::PlayerSlim) {
            return;
        }
        let Some(model) = self
            .models
            .entry(kind)
            .or_insert_with(|| CachedModel::new(kind))
            .as_player_model()
        else {
            return;
        };
        emit_cape(model, state, 0.0625, base, out);
    }

    /// The shared, concrete player model for `kind` (built on first use),
    /// or `None` for a non-player kind. 3D Skins (`super::skin3d::emit_player_body`) poses and
    /// emits it with some wear parts drawn as voxels; the cape, held items and every other
    /// layer then read the same posed instance, as after [`Self::emit_with`].
    pub fn player_model_mut(&mut self, kind: ModelKind) -> Option<&mut PlayerModel> {
        if !matches!(kind, ModelKind::Player | ModelKind::PlayerSlim) {
            return None;
        }
        self.models
            .entry(kind)
            .or_insert_with(|| CachedModel::new(kind))
            .as_player_model()
    }

    /// Emits one armour pass, building (and thereafter sharing) the `ModelBiped` that pass needs.
    ///
    /// java: `LayerArmorBase.doRenderLayer` — the four `renderLayer(..., 4/3/2/1)` calls, each of
    /// which picks `modelArmor` or `modelLeggings`, copies `isChild` off the entity, poses it
    /// with the body's own animation arguments and hides every part the slot does not cover.
    /// [`crate::entity::layers::emit_armor`] is that per-pass body; this wraps it with the model
    /// cache, since vanilla likewise keeps exactly two armour models per renderer rather than one
    /// per wearer.
    ///
    /// The caller supplies the texture: it depends on the equipped item's material, which is
    /// `mc-data`'s, not this module's. See
    /// [`armor_texture`](crate::entity::layers::armor_texture).
    pub fn emit_armor_slot(
        &mut self,
        slot: ArmorSlot,
        params: &AnimParams,
        base: Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        self.emit_armor_slot_for(ModelKind::Biped, slot, params, base, out);
    }

    /// Emits an armour slot whose model matches the selected living body.
    ///
    /// The ordinary biped helper is retained for callers that have no concrete renderer.  Entity
    /// submission must use this variant so `RenderZombie`'s `LayerVillagerArmor` can substitute
    /// its raised helmet cube when the body is `ModelZombieVillager`.
    pub fn emit_armor_slot_for(
        &mut self,
        body_kind: ModelKind,
        slot: ArmorSlot,
        params: &AnimParams,
        base: Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        let mut fresh = ArmorPoses::default();
        self.emit_armor_slot_for_entity(body_kind, slot, params, base, &mut fresh, out);
    }

    /// [`Self::emit_armor_slot_for`] across one entity's slots, posing each of the two armour
    /// models (`modelArmor` 1.0, `modelLeggings` 0.5) at most once instead of once per slot —
    /// a full set used to pose the 1.0 model three times with identical inputs (ENTM-5).
    ///
    /// Exact: every pose (`pose`, `pose_as_zombie`, `pose_stand`) writes each value it later
    /// reads within the same call, so it is a pure function of `params` and construction-time
    /// state, and the per-slot work between two slots only flips `show_model` and draws, which no
    /// pose reads. Re-posing would rewrite the same bits. `posed` must start fresh
    /// ([`ArmorPoses::default`]) for every entity.
    pub fn emit_armor_slot_for_entity(
        &mut self,
        body_kind: ModelKind,
        slot: ArmorSlot,
        params: &AnimParams,
        base: Mat4,
        posed: &mut ArmorPoses,
        out: &mut Vec<EntityVertex>,
    ) {
        let inflation = slot.model_inflation();
        let pose = posed.first_use(slot);
        if body_kind == ModelKind::ArmorStand {
            let model = self
                .armor_stand_models
                .entry(inflation.to_bits())
                .or_insert_with(|| ArmorStandArmorModel::new(inflation));
            model.model.is_child = params.entity.child;
            if pose {
                super::layers::pose_armor_stand(model, params);
            }
            super::layers::emit_armor_stand_posed(model, slot, params, base, out);
            return;
        }
        let kind = ArmorModelKind::for_body(body_kind);
        let model = self
            .armor_models
            .entry((kind, inflation.to_bits()))
            .or_insert_with(|| kind.build(inflation));
        // java: `LayerArmorBase.renderLayer` calls `t.setModelAttributes(renderer.getMainModel())`
        // before posing, and `ModelBase.setModelAttributes` copies `swingProgress`, `isRiding` and
        // `isChild` (`ModelBiped`'s held-item/sneak/bow extras reach `pose` through `params`).
        // Without the first two a chestplate's sleeves never followed an attack swing and
        // leggings never took the riding pose (phase 4c review N5).
        model.model.swing_progress = params.entity.swing_progress;
        model.model.is_riding = params.entity.riding;
        model.model.is_child = params.entity.child;
        if pose {
            match kind {
                ArmorModelKind::Zombie | ArmorModelKind::ZombieVillager => {
                    model.pose_as_zombie(params);
                }
                ArmorModelKind::Biped => model.pose(params),
            }
        }
        super::layers::emit_armor_posed(model, slot, params, base, out);
    }

    /// How many distinct `ArmorSlot::model_inflation()` armour models have been built. At most
    /// two, matching vanilla's `modelArmor`/`modelLeggings` pair.
    #[must_use]
    pub fn built_armor_models(&self) -> usize {
        self.armor_models.len()
    }

    /// The matrix a held item hangs off this entity's right hand, or `None` if the model has no
    /// hand to hang it from.
    ///
    /// java: `LayerHeldItem.doRenderLayer` → `ModelBiped.postRenderArm`. **Must be called after
    /// the body has been posed and drawn this frame** — `postRender` reads the arm's live
    /// rotation, so a matrix taken before [`Self::emit`]/[`Self::emit_with`] would hang the item
    /// off last frame's arm. The shared model instance is the same one either of those posed, so
    /// no re-pose happens here.
    ///
    /// `is_full_block` is vanilla's `Block.getRenderType() == 2` test on the held stack — a real
    /// cube rather than a flat icon, which gets an extra tilt and shrink.
    pub fn held_item_matrix_for(
        &mut self,
        kind: ModelKind,
        scale: f32,
        base: Mat4,
        child: bool,
        is_full_block: bool,
        sneaking: bool,
    ) -> Option<Mat4> {
        let model = self.model_mut(kind);
        let arm = model.held_item_arm()?;
        if kind == ModelKind::PlayerSlim {
            // java: ModelPlayer.postRenderArm increments bipedRightArm.rotationPointX by one
            // model unit for small arms, calls postRender, then immediately restores the pivot.
            // The player model is shared by every entity of this arm width, so keeping the
            // correction local to this attachment transform is essential: changing the cached
            // part would move the rendered arm and leak into the next player.
            let model = model.model_mut();
            model.get_mut(arm).rotation_point[0] += 1.0;
            let matrix = super::layers::held_item_matrix(
                model,
                arm,
                scale,
                base,
                child,
                is_full_block,
                sneaking,
            );
            model.get_mut(arm).rotation_point[0] -= 1.0;
            Some(matrix)
        } else if kind == ModelKind::ArmorStand {
            // java: ModelArmorStand.postRenderArm temporarily forces
            // bipedRightArm.showModel = true before delegating to
            // ModelBiped.postRenderArm.  A stand normally has its arms hidden,
            // but a held item still hangs from the stored, posed right-arm
            // transform.  Keep the mutation local: the body has already been
            // emitted, and leaking this bit would make the next stand draw a
            // visible arm.  `is_hidden` is intentionally untouched; Java only
            // changes showModel here.
            let model = model.model_mut();
            let was_visible = model.get(arm).show_model;
            model.get_mut(arm).show_model = true;
            let matrix = super::layers::held_item_matrix(
                model,
                arm,
                scale,
                base,
                child,
                is_full_block,
                sneaking,
            );
            model.get_mut(arm).show_model = was_visible;
            Some(matrix)
        } else {
            Some(super::layers::held_item_matrix(
                model.model(),
                arm,
                scale,
                base,
                child,
                is_full_block,
                sneaking,
            ))
        }
    }

    /// The matrix a named model part contributes, for a layer that hangs geometry off a part
    /// other than an arm.
    ///
    /// java: `ModelRenderer.postRender(0.0625F)`, which `LayerCustomHead`, `LayerSnowmanHead`,
    /// `LayerMooshroomMushroom` and `LayerHeldItemWitch` each call on a part they were handed at
    /// construction (`bipedHead`, `ModelSnowMan.head`, `ModelQuadruped.head`, `villagerNose`).
    /// Like [`Self::held_item_matrix_for`] this **must be called after the body was posed this
    /// frame** — `postRender` reads the part's live rotation, exactly as vanilla's does, and
    /// exactly like vanilla it applies only that one part's own transform and never walks its
    /// parent chain.
    ///
    /// `None` if this model has no part by that name, which is the honest answer for a layer
    /// attached to the wrong renderer.
    pub fn part_matrix_for(
        &mut self,
        kind: ModelKind,
        part: &str,
        scale: f32,
        base: Mat4,
    ) -> Option<Mat4> {
        let model = self.model_mut(kind).model();
        let id = model.find(part)?;
        Some(base * model.get(id).post_render_matrix(scale))
    }

    /// Samples the posed attachment transforms for `LayerArrow`.
    ///
    /// Java seeds a fresh `Random` from the entity id on every draw, then consumes one bounded
    /// draw for `ModelBase.boxList`, one for the selected part's `cubeList` (even when its length
    /// is one), and three floats. The shared body model must already have been posed this frame.
    pub fn stuck_arrow_attachments(
        &mut self,
        kind: ModelKind,
        entity_id: i32,
        count: i32,
        scale: f32,
        base: Mat4,
    ) -> Vec<StuckArrowAttachment> {
        if count <= 0 {
            return Vec::new();
        }
        let model = self.model_mut(kind).model();
        let parts = model.parts();
        let part_bound = i32::try_from(parts.len()).expect("model part count fits in i32");
        let mut random = JavaRandom::new(i64::from(entity_id));
        let mut attachments = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let part_index = random.next_i32_bound(part_bound) as usize;
            let part = &parts[part_index];
            let cube_bound = i32::try_from(part.cubes.len()).expect("model cube count fits in i32");
            let cube_index = random.next_i32_bound(cube_bound) as usize;
            let cube = &part.cubes[cube_index];
            let mut x = random.next_f32();
            let mut y = random.next_f32();
            let mut z = random.next_f32();
            let point = Vec3::new(
                (cube.pos1[0] + (cube.pos2[0] - cube.pos1[0]) * x) / 16.0,
                (cube.pos1[1] + (cube.pos2[1] - cube.pos1[1]) * y) / 16.0,
                (cube.pos1[2] + (cube.pos2[2] - cube.pos1[2]) * z) / 16.0,
            );
            x = -(x * 2.0 - 1.0);
            y = -(y * 2.0 - 1.0);
            z = -(z * 2.0 - 1.0);
            let horizontal = mc_math::sqrt_f(x * x + z * z);
            let yaw = (f64::from(x).atan2(f64::from(z)) * 180.0 / std::f64::consts::PI) as f32;
            let pitch =
                (f64::from(y).atan2(f64::from(horizontal)) * 180.0 / std::f64::consts::PI) as f32;
            attachments.push(StuckArrowAttachment {
                part_index,
                cube_index,
                point,
                yaw,
                pitch,
                matrix: base * part.post_render_matrix(scale) * Mat4::from_translation(point),
            });
        }
        attachments
    }

    /// Applies a player's `EnumPlayerModelParts` visibility mask.
    ///
    /// java: `RenderPlayer.setModelVisibilities` (`:44-72`) — `bipedHeadwear.showModel =
    /// isWearing(HAT)` and the same for the jacket, both sleeves and both pant legs, read from
    /// `AbstractClientPlayer.getModelFlags()` (watched index 10 on `EntityPlayer`). Nothing set
    /// these, so every player showed all five overlays to everyone regardless of their own
    /// settings.
    ///
    /// The mask bits are `EnumPlayerModelParts`' own ids: cape 0, jacket 1, left sleeve 2, right
    /// sleeve 3, left pants 4, right pants 5, hat 6. **Must be called before the body is posed
    /// this frame**, since one model instance is shared by every player on the server.
    ///
    /// A `ModelKind` that is not a player model is a no-op, which is what a `RenderBiped` doing
    /// this would be.
    pub fn set_player_model_parts(&mut self, kind: ModelKind, mask: i8) {
        self.set_player_model_visibility(kind, false, mask);
    }

    /// Applies `RenderPlayer.setModelVisibilities`, including its spectator-only head branch.
    ///
    /// java: a spectator first calls `ModelBiped.setInvisible(false)`, then forces only
    /// `bipedHead` and `bipedHeadwear` visible. A non-spectator restores every base part before
    /// applying the ordinary skin-parts mask. The explicit restore matters because each arm
    /// width has one model instance shared by every player rendered in the frame.
    pub fn set_player_model_visibility(&mut self, kind: ModelKind, spectator: bool, mask: i8) {
        if !matches!(kind, ModelKind::Player | ModelKind::PlayerSlim) {
            return;
        }
        const PARTS: [(&str, u8); 6] = [
            ("cape", 0),
            ("bodywear", 1),
            ("left_armwear", 2),
            ("right_armwear", 3),
            ("left_legwear", 4),
            ("right_legwear", 5),
        ];
        let model = self.model_mut(kind).model_mut();
        for name in [
            "head",
            "headwear",
            "body",
            "right_arm",
            "left_arm",
            "right_leg",
            "left_leg",
        ] {
            if let Some(id) = model.find(name) {
                model.get_mut(id).show_model = !spectator;
            }
        }
        for (name, bit) in PARTS {
            if let Some(id) = model.find(name) {
                model.get_mut(id).show_model = !spectator && mask & (1 << bit) != 0;
            }
        }
        // java: `bipedHeadwear` is bit 6, and it lives on `ModelBiped` rather than `ModelPlayer`,
        // so it is named differently here.
        if let Some(id) = model.find("headwear") {
            model.get_mut(id).show_model = spectator || mask & (1 << 6) != 0;
        }
        if spectator {
            if let Some(id) = model.find("head") {
                model.get_mut(id).show_model = true;
            }
        }
    }

    /// The matrix a **witch's** held item hangs off, or `None` if this model has no nose part.
    ///
    /// java: `LayerHeldItemWitch.doRenderLayer` → `((ModelWitch)getMainModel()).villagerNose
    /// .postRender(0.0625F)`. Same "must run after the body was posed" contract as
    /// [`Self::held_item_matrix_for`]; see [`crate::entity::layers::witch_held_item_matrix`] for
    /// the two vanilla quirks it preserves.
    pub fn witch_held_item_matrix_for(
        &mut self,
        kind: ModelKind,
        scale: f32,
        base: Mat4,
        child: bool,
        item: super::layers::WitchHeldItem,
    ) -> Option<Mat4> {
        let model = self.model_mut(kind).model();
        let nose = model.find("nose")?;
        Some(super::layers::witch_held_item_matrix(
            model,
            nose,
            scale,
            base,
            child,
            item,
            super::layers::WITCH_FULL_3D_LIFT,
        ))
    }

    /// A posed part's pitch, in radians — the one thing
    /// [`crate::entity::layers::iron_golem_flower_matrix`] needs that a matrix cannot supply.
    ///
    /// java: `((ModelIronGolem)getMainModel()).ironGolemRightArm.rotateAngleX`, read by
    /// `LayerIronGolemFlower.doRenderLayer` *after* the body pose.
    pub fn part_angle_x(&mut self, kind: ModelKind, part: &str) -> Option<f32> {
        let model = self.model_mut(kind).model();
        let id = model.find(part)?;
        Some(model.get(id).rotate_angle[0])
    }

    /// How many distinct body models have been built.
    #[must_use]
    pub fn built_models(&self) -> usize {
        self.models.len()
    }

    /// How many distinct `(kind, inflation)` layer models have been built.
    #[must_use]
    pub fn built_layer_models(&self) -> usize {
        self.layer_models.len()
    }

    /// Poses and emits one entity, returning the texture key to draw it with.
    ///
    /// `variant` selects between a type's texture variants — see [`texture_for`]. Returns
    /// `None` for a type with no renderer, in which case nothing is emitted, which is what
    /// vanilla does when `getEntityRenderObject` misses.
    pub fn emit(
        &mut self,
        t: EntityType,
        variant: i32,
        transform: &EntityTransform,
        params: &AnimParams,
        out: &mut Vec<EntityVertex>,
    ) -> Option<&'static str> {
        self.emit_with_state(t, variant, transform, params, 0.0, out)
    }

    /// Poses and emits one entity with the per-frame callback input that cannot be inferred from
    /// [`AnimParams`] alone.
    ///
    /// `creeper_flash` is the interpolated result of Java's
    /// `EntityCreeper.getCreeperFlashIntensity(partialTicks)`. Passing `0.0` is the neutral
    /// value used by [`Self::emit`] for callers that do not track that watcher-derived sample;
    /// callers that do have it should use this method so the non-uniform Creeper scale reaches
    /// the matrix before the model lift. Slime/MagmaCube size and squish, Bat bob, and the
    /// state-selected renderer variants are all handled by the same seam.
    pub fn emit_with_state(
        &mut self,
        t: EntityType,
        variant: i32,
        transform: &EntityTransform,
        params: &AnimParams,
        creeper_flash: f32,
        out: &mut Vec<EntityVertex>,
    ) -> Option<&'static str> {
        let desc = renderer_for_variant(t, variant)?;
        let mut transform = *transform;
        // The descriptor's static scale composes with whatever dynamic scale the caller set.
        transform.scale *= desc.scale;
        let base = match t {
            // java: `RenderSlime`/`RenderMagmaCube.preRenderCallback` is non-uniform and must
            // be composed before the model lift. `variant` is the watched slime size and
            // `squish_factor` is the interpolated callback input.
            EntityType::Slime | EntityType::LavaSlime => entity_matrix_scaled(
                &transform,
                entity_pre_render_scale(
                    t,
                    variant,
                    params.entity.child,
                    params.entity.squish_factor,
                    creeper_flash,
                ),
            ),
            EntityType::Creeper => entity_matrix_scaled(
                &transform,
                entity_pre_render_scale(
                    t,
                    variant,
                    params.entity.child,
                    params.entity.squish_factor,
                    creeper_flash,
                ),
            ),
            EntityType::Villager if params.entity.child => entity_matrix_scaled(
                &transform,
                entity_pre_render_scale(
                    t,
                    variant,
                    params.entity.child,
                    params.entity.squish_factor,
                    creeper_flash,
                ),
            ),
            _ => placement_matrix_for(t, &transform, params),
        };
        self.model_mut(desc.model)
            .pose_and_render(params, base, out);
        Some(texture_for(t, variant))
    }

    /// Poses and emits a horse with its watched/client-local state, returning the layered texture
    /// selection the resource loader must resolve for this frame.
    ///
    /// `packed_variant` is the horse colour/marking pair and `armor` is the synced armour index;
    /// they are intentionally separate from [`HorseRenderState::horse_type`], matching the three
    /// independent `EntityHorse` watcher values. The method uses the same cached `HorseModel` as
    /// ordinary [`Self::emit`] calls, so a manager never has two horse part forests racing to hold
    /// different poses.
    pub fn emit_horse_with_state(
        &mut self,
        transform: &EntityTransform,
        params: &AnimParams,
        state: &HorseRenderState,
        packed_variant: i32,
        armor: i32,
        out: &mut Vec<EntityVertex>,
    ) -> Option<HorseTextureSelection> {
        let desc = renderer_for_variant(EntityType::EntityHorse, state.horse_type)?;
        let mut transform = *transform;
        transform.scale *= desc.scale;
        let base = placement_matrix_for(EntityType::EntityHorse, &transform, params);
        self.horse_model_mut()
            .pose_and_render_with_state(params, state, base, out);
        Some(horse_texture_selection(
            state.horse_type,
            packed_variant,
            armor,
        ))
    }

    /// The same, but under a caller-supplied matrix — for layers and tile entities, which do
    /// their own placement.
    pub fn emit_with(
        &mut self,
        kind: ModelKind,
        base: Mat4,
        params: &AnimParams,
        out: &mut Vec<EntityVertex>,
    ) {
        self.model_mut(kind).pose_and_render(params, base, out);
    }

    /// Emits the skull `LayerCustomHead` draws for an `Items.skull` helmet, under the matrix
    /// [`custom_head_matrix`](super::layers::custom_head_matrix) built for its skull branch.
    ///
    /// java: `LayerCustomHead.doRenderLayer` (`:106`) ends with
    /// `TileEntitySkullRenderer.renderSkull(-0.5F, 0.0F, -0.5F, EnumFacing.UP, 180.0F, meta,
    /// profile, -1)`. Its `UP` branch adds `translate(x + 0.5, y, z + 0.5)` (zero here) and
    /// `scale(-1, -1, 1)`, then calls `modelbase.render(null, 0, 0, 0, 180.0F, 0, 0.0625F)`: the
    /// skull is posed with a fixed 180-degree yaw and no pitch, never with the wearer's head
    /// angles, which `bipedHead.postRender` has already put in the matrix. Together with the
    /// layer's `scale(1.1875, -1.1875, -1.1875)` the flip and the half turn cancel, leaving the
    /// head box enlarged 1.1875 times, so a worn skull sits over the head. Posing the skull with
    /// the wearer's own [`AnimParams`] and no flip hung it upside down below the neck, inside the
    /// torso, where it could not be seen.
    ///
    /// `model` is [`SkullVariant::model`](super::tile::SkullVariant::model).
    pub fn emit_custom_head_skull(
        &mut self,
        model: ModelKind,
        layer_matrix: Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        let (skull, yaw) = super::tile::skull_transform(
            Vec3::new(-0.5, 0.0, -0.5),
            super::tile::SkullMount::Floor,
            180.0,
        );
        let params = AnimParams {
            net_head_yaw: yaw,
            ..AnimParams::still()
        };
        self.emit_with(model, layer_matrix * skull, &params, out);
    }

    /// [`Self::emit_with`] for a body that draws nothing this frame (`BodyDraw::Skip`): leaves
    /// the shared model in exactly the pose `emit_with` would, so layers that read it
    /// ([`Self::held_item_matrix_for`], head attachments) stay correct, but emits no vertices.
    /// See [`EntityModel::pose_only`].
    pub fn pose_with(&mut self, kind: ModelKind, params: &AnimParams) {
        self.model_mut(kind).pose_only(params);
    }

    /// Emits a `RenderWitherSkull` head through its own cached model instance.
    ///
    /// java: `RenderWitherSkull` owns `new ModelSkeletonHead()`, whose no-argument constructor
    /// uses `(0, 35, 64, 64)`. This dedicated path keeps that 64×64 model separate from
    /// [`ModelKind::SkeletonHead`], the tile skull's `(0, 0, 64, 32)` model. The caller supplies
    /// the same translated/flipped matrix and interpolated head angles that Java passes to
    /// `ModelSkeletonHead.render`.
    pub fn emit_wither_skull(
        &mut self,
        base: Mat4,
        params: &AnimParams,
        out: &mut Vec<EntityVertex>,
    ) {
        self.model_mut(ModelKind::WitherSkull)
            .pose_and_render(params, base, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::anim::DragonRenderState;
    use glam::Vec3;
    use mc_data::EntityType as E;

    #[test]
    fn deadmau5_layer_emits_two_mirrored_ear_boxes_and_follows_the_head() {
        let mut manager = RenderManager::new();
        let mut params = AnimParams::still();
        let mut neutral = Vec::new();
        manager.emit_deadmau5_ears(ModelKind::Player, &params, Mat4::IDENTITY, &mut neutral);

        // One one-cube ModelRenderer is six quads / twelve triangles / 36 vertices. Java calls
        // renderDeadmau5Head exactly twice, once at each 0.375-block X offset.
        assert_eq!(neutral.len(), 72);
        let left_x = neutral[..36]
            .iter()
            .map(|vertex| vertex.pos[0])
            .sum::<f32>()
            / 36.0;
        let right_x = neutral[36..]
            .iter()
            .map(|vertex| vertex.pos[0])
            .sum::<f32>()
            / 36.0;
        assert!((left_x + 0.375).abs() < 1.0e-6, "{left_x}");
        assert!((right_x - 0.375).abs() < 1.0e-6, "{right_x}");

        let mut cape = Vec::new();
        manager.emit_cape(
            ModelKind::Player,
            &CapeState::default(),
            Mat4::IDENTITY,
            &mut cape,
        );
        assert_eq!(cape.len(), 36, "LayerCape submits the one hidden cape box");
        let trailing = CapeState {
            chase_dx: 0.25,
            ..Default::default()
        };
        let mut trailing_vertices = Vec::new();
        manager.emit_cape(
            ModelKind::Player,
            &trailing,
            Mat4::IDENTITY,
            &mut trailing_vertices,
        );
        assert_ne!(cape, trailing_vertices, "cape geometry follows chase input");

        params.net_head_yaw = 37.0;
        params.head_pitch = -21.0;
        let mut looking = Vec::new();
        manager.emit_deadmau5_ears(ModelKind::PlayerSlim, &params, Mat4::IDENTITY, &mut looking);
        assert_eq!(looking.len(), 72);
        assert_ne!(looking[0].pos, neutral[0].pos);

        let mut non_player = Vec::new();
        manager.emit_deadmau5_ears(ModelKind::Zombie, &params, Mat4::IDENTITY, &mut non_player);
        assert!(non_player.is_empty());
    }

    /// Exercises the public renderer dispatch rather than merely comparing the object helper:
    /// a regression that changes `RenderManager::emit` back to `entity_matrix` makes each of
    /// these equality checks fail.  `death_rotation`, `scale`, and `y_offset` are all inputs
    /// unique to `RendererLivingEntity`; mutating them must leave an object renderer untouched.
    fn assert_object_emit_ignores_living_only_transform(
        kind: E,
        expected: fn(&EntityTransform) -> Mat4,
    ) {
        let still = EntityTransform {
            rel: Vec3::new(3.0, 5.0, 7.0),
            body_yaw: 37.0,
            ..EntityTransform::default()
        };
        let living_mutation = EntityTransform {
            death_rotation: 63.0,
            scale: 2.5,
            y_offset: 11.0,
            ..still
        };

        assert_eq!(placement_matrix(kind, &still), expected(&still));
        assert_eq!(placement_matrix(kind, &living_mutation), expected(&still));
        assert_ne!(
            entity_matrix(&still),
            entity_matrix(&living_mutation),
            "the oracle must mutate the living-only matrix inputs"
        );

        let emit = |transform: &EntityTransform| {
            let mut manager = RenderManager::new();
            let mut vertices = Vec::new();
            manager
                .emit(kind, 0, transform, &AnimParams::still(), &mut vertices)
                .expect("all four objects have renderer descriptors");
            vertices
        };
        assert_eq!(
            emit(&still),
            emit(&living_mutation),
            "{kind:?} must not inherit RendererLivingEntity placement"
        );
    }

    /// java: `RenderBoat` is a model-backed object renderer, not `RendererLivingEntity`.
    #[test]
    fn a_boat_emit_uses_its_object_matrix_not_living_mutations() {
        assert_object_emit_ignores_living_only_transform(E::Boat, |t| {
            boat_matrix(t.rel, t.body_yaw, 0.0, 0.0, 1, 0.0)
        });
    }

    /// Every minecart subtype shares `RenderMinecart.doRender`; the rideable representative
    /// verifies that the table dispatch reaches that shared object path.
    #[test]
    fn a_minecart_emit_uses_its_object_matrix_not_living_mutations() {
        assert_object_emit_ignores_living_only_transform(E::MinecartRideable, |t| {
            minecart_matrix(t.rel, t.body_yaw, 0.0, 0.0, 0.0, 1, 0.0)
        });
    }

    /// java: `RenderLeashKnot.doRender` is a translation plus model flip, never a corpse turn.
    #[test]
    fn a_leash_knot_emit_uses_its_object_matrix_not_living_mutations() {
        assert_object_emit_ignores_living_only_transform(E::LeashKnot, |t| {
            leash_knot_matrix(t.rel)
        });
    }

    /// java: `RenderEnderCrystal.doRender` is a pure translation; its animation stays inside
    /// `ModelEnderCrystal` and must not be confused with a living renderer's outer pose.
    #[test]
    fn an_ender_crystal_emit_uses_its_object_matrix_not_living_mutations() {
        assert_object_emit_ignores_living_only_transform(E::EnderCrystal, |t| {
            ender_crystal_matrix(t.rel)
        });
    }

    #[test]
    fn armor_stand_transform_ignores_a_direct_death_rotation_mutation() {
        let upright = EntityTransform {
            rel: Vec3::new(3.0, 5.0, 7.0),
            body_yaw: 37.0,
            ..EntityTransform::default()
        };
        let mut dying = upright;
        dying.death_rotation = 89.0;

        assert_eq!(
            placement_matrix(E::ArmorStand, &upright),
            placement_matrix(E::ArmorStand, &dying)
        );
        assert_ne!(
            entity_matrix(&upright),
            entity_matrix(&dying),
            "the negative control must prove a normal living renderer would roll"
        );
    }

    #[test]
    fn squid_transform_keeps_java_rotation_order_and_ignores_death_roll() {
        let transform = EntityTransform {
            rel: Vec3::new(3.0, 5.0, 7.0),
            body_yaw: 37.0,
            scale: 1.25,
            ..EntityTransform::default()
        };
        let angled = squid_placement_matrix(&transform, 17.0, -23.0);
        let mut params = AnimParams::still();
        params.entity.squid_pitch = 17.0;
        params.entity.squid_yaw = -23.0;
        let mut changed = transform;
        changed.death_rotation = 89.0;

        assert_eq!(
            placement_matrix_for(E::Squid, &transform, &params),
            angled,
            "the live typed placement adapter must consume the interpolated squid angles"
        );
        assert_eq!(
            angled,
            squid_placement_matrix(&changed, 17.0, -23.0),
            "RenderSquid.rotateCorpse never calls the death-spin branch"
        );
        assert_ne!(
            angled,
            entity_matrix(&transform),
            "the normal living transform is the negative control"
        );
        assert_eq!(
            placement_matrix(E::Squid, &transform),
            squid_placement_matrix(&transform, 0.0, 0.0)
        );
    }

    #[test]
    fn bat_transform_dispatches_flying_bob_and_hanging_offset() {
        let transform = EntityTransform {
            rel: Vec3::new(3.0, 5.0, 7.0),
            body_yaw: 37.0,
            ..EntityTransform::default()
        };
        let mut flying = AnimParams::still();
        flying.age_in_ticks = 5.0;
        let mut expected_flying = transform;
        expected_flying.rel.y += mc_math::cos(1.5) * 0.1;
        assert_eq!(
            placement_matrix_for(E::Bat, &transform, &flying),
            placement_matrix(E::Bat, &expected_flying)
        );

        let mut hanging = flying.clone();
        hanging.entity.bat_hanging = true;
        let mut expected_hanging = transform;
        expected_hanging.rel.y -= 0.1;
        assert_eq!(
            placement_matrix_for(E::Bat, &transform, &hanging),
            placement_matrix(E::Bat, &expected_hanging)
        );
        assert_ne!(
            placement_matrix_for(E::Bat, &transform, &flying),
            placement_matrix_for(E::Bat, &transform, &hanging),
            "a mutant that drops the bat state would collapse both placements"
        );
    }

    #[test]
    fn iron_golem_placement_carries_the_phase_shifted_walking_sway() {
        let transform = EntityTransform {
            rel: Vec3::new(3.0, 5.0, 7.0),
            body_yaw: 37.0,
            death_rotation: 2.0,
            ..EntityTransform::default()
        };
        let mut walking = AnimParams::still();
        walking.limb_swing = 0.5;
        walking.limb_swing_amount = 1.0;

        // `limb_swing + 6 == 6.5` is the trough of Java's triangle wave: exactly -6.5 degrees.
        let mut expected = transform;
        expected.death_rotation -= 6.5;
        let placed = placement_matrix_for(E::VillagerGolem, &transform, &walking);
        assert_eq!(placed, placement_matrix(E::VillagerGolem, &expected));
        assert_ne!(
            placed,
            placement_matrix(E::VillagerGolem, &transform),
            "dropping the walking sway must change the matrix"
        );

        let mut below_threshold = walking.clone();
        below_threshold.limb_swing_amount = 0.009;
        assert_eq!(
            placement_matrix_for(E::VillagerGolem, &transform, &below_threshold),
            placement_matrix(E::VillagerGolem, &transform)
        );
        assert_eq!(
            placement_matrix_for(E::Zombie, &transform, &walking),
            placement_matrix(E::Zombie, &transform),
            "the stateful sway is golem-only"
        );
    }

    #[test]
    fn converting_zombie_placement_uses_the_tiny_integer_tick_yaw_shake() {
        let transform = EntityTransform {
            rel: Vec3::new(3.0, 5.0, 7.0),
            body_yaw: 37.0,
            ..EntityTransform::default()
        };
        let mut converting = AnimParams::still();
        converting.entity.converting = true;
        converting.entity.ticks_existed = 17;
        converting.age_in_ticks = 17.75;

        let shake = ((17.0_f64 * 3.25).cos() * core::f64::consts::PI * 0.25) as f32;
        let mut expected = transform;
        expected.body_yaw += shake;
        let placed = placement_matrix_for(E::Zombie, &transform, &converting);
        assert_eq!(placed, placement_matrix(E::Zombie, &expected));
        assert_ne!(
            placed,
            placement_matrix(E::Zombie, &transform),
            "dropping the conversion flag must change the zombie matrix"
        );

        let mut ordinary = converting.clone();
        ordinary.entity.converting = false;
        assert_eq!(
            placement_matrix_for(E::Zombie, &transform, &ordinary),
            placement_matrix(E::Zombie, &transform)
        );
        assert_eq!(
            placement_matrix_for(E::PigZombie, &transform, &converting),
            placement_matrix(E::PigZombie, &transform),
            "index 14 must not shake a PigZombie"
        );
    }

    #[test]
    fn stateful_renderer_variants_carry_java_pre_render_scales() {
        let guardian = renderer_for_variant(E::Guardian, 0).unwrap();
        let elder = renderer_for_variant(E::Guardian, 1).unwrap();
        assert_eq!(guardian.scale, 1.0);
        assert_eq!(elder.scale, 2.35);
        assert_ne!(guardian.scale, elder.scale);
        assert_eq!(
            texture_for(E::Guardian, 1),
            "textures/entity/guardian_elder.png"
        );

        let skeleton = renderer_for_variant(E::Skeleton, 0).unwrap();
        let wither_skeleton = renderer_for_variant(E::Skeleton, 1).unwrap();
        assert_eq!(skeleton.scale, 1.0);
        assert_eq!(wither_skeleton.scale, 1.2);
        assert_ne!(skeleton.scale, wither_skeleton.scale);
        assert_eq!(
            texture_for(E::Skeleton, 1),
            "textures/entity/skeleton/wither_skeleton.png"
        );

        let wild_cat = renderer_for_variant(E::Ozelot, 0).unwrap();
        let tame_cat = renderer_for_variant(E::Ozelot, 1).unwrap();
        assert_eq!(wild_cat.scale, 1.0);
        assert_eq!(tame_cat.scale, 0.8);
        assert_ne!(wild_cat.scale, tame_cat.scale);
        assert_eq!(renderer_for_variant(E::Ozelot, 4).unwrap().scale, 1.0);
    }

    #[test]
    fn horse_renderer_registers_type_scales_and_texture_forks() {
        let horse = renderer_for(E::EntityHorse).unwrap();
        assert_eq!(horse.model, ModelKind::Horse);
        assert_eq!(horse.texture, "textures/entity/horse/horse_white.png");
        assert_eq!(horse.shadow_size, 0.75);
        assert_eq!(horse.scale, 1.0);

        let donkey = renderer_for_variant(E::EntityHorse, 1).unwrap();
        assert_eq!(donkey.model, ModelKind::Horse);
        assert_eq!(donkey.texture, "textures/entity/horse/donkey.png");
        assert_eq!(donkey.scale, 0.87);
        let mule = renderer_for_variant(E::EntityHorse, 2).unwrap();
        assert_eq!(mule.texture, "textures/entity/horse/mule.png");
        assert_eq!(mule.scale, 0.92);
        assert_eq!(renderer_for_variant(E::EntityHorse, 3).unwrap().scale, 1.0);
        assert_eq!(
            texture_for(E::EntityHorse, 3),
            "textures/entity/horse/horse_zombie.png"
        );
        assert_eq!(
            texture_for(E::EntityHorse, 4),
            "textures/entity/horse/horse_skeleton.png"
        );
    }

    #[test]
    fn horse_texture_selection_matches_vanilla_layered_tables() {
        let normal = horse_texture_selection(0, 2 | (3 << 8), 3);
        assert!(normal.valid);
        assert_eq!(
            normal.paths(),
            [
                Some("textures/entity/horse/horse_chestnut.png"),
                Some("textures/entity/horse/horse_markings_whitedots.png"),
                Some("textures/entity/horse/armor/horse_armor_diamond.png"),
            ]
        );
        assert_eq!(normal.cache_key(), Some("horse/hchwdodio"));

        let unarmoured = horse_texture_selection(0, 0, 0);
        assert!(unarmoured.valid);
        assert_eq!(
            unarmoured.paths(),
            [Some("textures/entity/horse/horse_white.png"), None, None]
        );
        assert_eq!(unarmoured.cache_key(), Some("horse/hwh"));

        // Non-normal horses use their type sheet until armour is present; with armour, vanilla
        // gives LayeredTexture an empty base plus the armour sheet.
        let bare_donkey = horse_texture_selection(1, 0, 0);
        assert!(!bare_donkey.valid);
        assert_eq!(bare_donkey.fallback, "textures/entity/horse/donkey.png");
        assert_eq!(bare_donkey.paths(), [None, None, None]);
        let gold_mule = horse_texture_selection(2, 0, 2);
        assert!(gold_mule.valid);
        assert_eq!(
            gold_mule.paths(),
            [
                None,
                None,
                Some("textures/entity/horse/armor/horse_armor_gold.png")
            ]
        );
        assert_eq!(gold_mule.cache_key(), Some("horse/_2_goo"));

        let invalid = horse_texture_selection(0, 7, 0);
        assert!(!invalid.valid);
        assert_eq!(invalid.fallback, "textures/entity/horse/horse_white.png");
        assert_eq!(invalid.cache_key(), None);
    }

    #[test]
    fn horse_dispatch_emits_visible_geometry_for_horse_and_donkey() {
        let mut manager = RenderManager::new();
        let transform = EntityTransform::default();
        let mut p = AnimParams::still();
        p.entity.child = false;
        p.entity.variant = 0;
        let mut horse_vertices = Vec::new();
        assert_eq!(
            manager.emit(E::EntityHorse, 0, &transform, &p, &mut horse_vertices),
            Some("textures/entity/horse/horse_white.png")
        );
        assert_eq!(horse_vertices.len(), 23 * 36);
        assert!(horse_vertices.iter().any(|v| v.pos != [0.0; 3]));
        assert_eq!(manager.built_models(), 1);

        p.entity.variant = 1;
        let mut donkey_vertices = Vec::new();
        assert_eq!(
            manager.emit(E::EntityHorse, 1, &transform, &p, &mut donkey_vertices),
            Some("textures/entity/horse/donkey.png")
        );
        assert_eq!(donkey_vertices.len(), horse_vertices.len());
        assert_ne!(donkey_vertices, horse_vertices);
        assert_eq!(
            manager.built_models(),
            1,
            "all horse types share ModelHorse"
        );
    }

    #[test]
    fn horse_stateful_dispatch_emits_optional_parts_through_the_shared_cache() {
        let mut manager = RenderManager::new();
        let state = HorseRenderState {
            horse_type: 2,
            adult: true,
            saddled: true,
            chested: true,
            ridden: true,
            ..HorseRenderState::default()
        };
        let mut params = AnimParams::still();
        params.entity.child = false;
        let mut vertices = Vec::new();
        let selection = manager
            .emit_horse_with_state(
                &EntityTransform::default(),
                &params,
                &state,
                2 | (3 << 8),
                3,
                &mut vertices,
            )
            .expect("mule has a horse renderer descriptor");
        // 23 base + 10 saddle/bridle + 2 reins + 2 mule chest parts.
        assert_eq!(vertices.len(), 37 * 36);
        assert_eq!(selection.upload_key(), "horse/_2_dio");
        assert_eq!(selection.upload_plan().layers[2], selection.armor);
        assert_eq!(manager.built_models(), 1);

        let mut generic = Vec::new();
        params.entity.variant = 0;
        manager.emit(
            E::EntityHorse,
            0,
            &EntityTransform::default(),
            &params,
            &mut generic,
        );
        assert_eq!(generic.len(), 23 * 36);
        assert_eq!(
            manager.built_models(),
            1,
            "stateful and generic paths share HorseModel"
        );
    }

    #[test]
    fn dragon_dispatch_emits_the_reused_spine_model() {
        let mut manager = RenderManager::new();
        let mut params = AnimParams::still();
        params.partial_ticks = 0.25;
        params.entity.dragon = Some(DragonRenderState::neutral(0.5));
        let mut vertices = Vec::new();

        assert_eq!(
            manager.emit(
                E::EnderDragon,
                0,
                &EntityTransform::default(),
                &params,
                &mut vertices,
            ),
            Some("textures/entity/enderdragon/dragon.png")
        );
        // 7 head/jaw + 5 neck + 4 body + 20 mirrored wing/leg trees + 12 tail segments,
        // each box contributing six quads and six triangle vertices.
        assert_eq!(vertices.len(), 65 * 36);
        assert!(vertices.iter().any(|v| v.pos != [0.0; 3]));
        assert!(vertices.iter().all(|v| v.pos.iter().all(|c| c.is_finite())));
        assert_eq!(manager.built_models(), 1);
    }

    #[test]
    fn horse_layered_upload_contract_composes_in_source_over_order() {
        let selection = horse_texture_selection(0, 0, 3);
        let plan = selection.upload_plan();
        assert!(plan.layered);
        assert_eq!(plan.key, "horse/hwhdio");
        assert_eq!(
            plan.layers,
            [
                Some("textures/entity/horse/horse_white.png"),
                None,
                Some("textures/entity/horse/armor/horse_armor_diamond.png")
            ]
        );

        // Java's LayeredTexture draws the later armour sheet over the base with SrcOver. Opaque
        // 1x1 fixtures make the layer order unambiguous without a filesystem or GPU.
        let base = [10, 20, 30, 255];
        let armor = [200, 150, 100, 255];
        assert_eq!(
            selection.compose_rgba([Some(&base), None, Some(&armor)], 1, 1),
            Some(armor.to_vec())
        );

        let translucent_armor = [200, 150, 100, 128];
        let blended = selection
            .compose_rgba([Some(&base), None, Some(&translucent_armor)], 1, 1)
            .expect("transparent armour still composes");
        assert!(
            blended[3] >= 254,
            "source-over keeps the opaque base alpha (with image's integer rounding): {blended:?}"
        );
        assert!(
            blended[0] > base[0] && blended[0] < translucent_armor[0],
            "armour RGB is alpha-composited over the base: {blended:?}"
        );

        // A malformed resource set must not silently enter the generated-texture cache.
        assert_eq!(
            selection.compose_rgba([Some(&base), Some(&base), Some(&armor)], 1, 1),
            None
        );
        assert_eq!(
            selection.compose_rgba([Some(&[1, 2, 3]), None, Some(&armor)], 1, 1),
            None
        );
        let invalid = horse_texture_selection(0, 7, 0);
        assert!(!invalid.upload_plan().layered);
        assert_eq!(invalid.compose_rgba([None, None, None], 1, 1), None);
    }

    #[test]
    fn creeper_flash_scale_matches_the_java_clamp_square_and_reciprocal() {
        for flash in [-0.25_f32, 0.0, 0.371, 1.0, 1.25] {
            let f1 = 1.0 + mc_math::sin(flash * 100.0) * flash * 0.01;
            let f = flash.clamp(0.0, 1.0);
            let f = f * f;
            let f = f * f;
            let expected = Vec3::new(
                (1.0 + f * 0.4) * f1,
                (1.0 + f * 0.1) / f1,
                (1.0 + f * 0.4) * f1,
            );
            assert_eq!(creeper_pre_render_scale(flash), expected, "flash={flash}");
        }
        let pulsing = creeper_pre_render_scale(0.371);
        assert_eq!(pulsing.x, pulsing.z);
        assert_ne!(
            pulsing.x, pulsing.y,
            "the negative control catches replacing the non-uniform Java scale with one factor"
        );
        assert_eq!(creeper_pre_render_scale(0.0), Vec3::ONE);
    }

    #[test]
    fn dynamic_scale_and_shadow_seams_keep_size_state() {
        assert_eq!(
            entity_pre_render_scale(E::Villager, 0, true, 0.0, 0.0),
            Vec3::splat(0.5)
        );
        assert_eq!(
            entity_pre_render_scale(E::Villager, 0, false, 0.0, 0.0),
            Vec3::ONE
        );
        assert_ne!(
            entity_pre_render_scale(E::Slime, 4, false, 0.25, 0.0),
            Vec3::ONE,
            "size and squish must reach the non-uniform slime callback"
        );
        assert_eq!(shadow_size_for(E::Slime, 1, false), Some(0.25));
        assert_eq!(shadow_size_for(E::Slime, 4, false), Some(1.0));
        assert_ne!(
            shadow_size_for(E::Slime, 1, false),
            shadow_size_for(E::Slime, 4, false)
        );
        // RenderMagmaCube only supplies the 0.25F constructor shadow to RenderLiving; unlike
        // RenderSlime it does not override doRender to replace that field per size. Its body
        // still uses the same dynamic non-uniform callback, so this is an intentional negative
        // control for the shadow seam rather than a typo in the shared slime geometry path.
        assert_eq!(shadow_size_for(E::LavaSlime, 1, false), Some(0.25));
        assert_eq!(shadow_size_for(E::LavaSlime, 4, false), Some(0.25));
        assert_eq!(shadow_size_for(E::EnderDragon, 0, false), Some(0.5));
        assert_eq!(shadow_size_for(E::Villager, 0, false), Some(0.5));
        assert_eq!(shadow_size_for(E::Villager, 0, true), Some(0.25));
        assert_eq!(shadow_size_for(E::Zombie, 1, true), Some(0.5));
    }

    #[test]
    fn every_mob_a_server_can_spawn_has_a_renderer() {
        // `Mob` and `Monster` are abstract EntityList entries. The dragon, Wither and horse are
        // intentionally included as negative controls because they now have real descriptors.
        let unported = [E::Mob, E::Monster];
        for t in mc_data::entity_types() {
            if !t.is_living() || unported.contains(&t) {
                continue;
            }
            assert!(
                renderer_for(t).is_some(),
                "{} ({}) has no renderer",
                t.name(),
                t.net_id()
            );
        }
    }

    #[test]
    fn shadow_sizes_match_the_render_manager_constructor() {
        // java: RenderManager.java:142-198 and each Render*'s own super(...) call.
        for (t, want) in [
            (E::Pig, 0.7),
            (E::Sheep, 0.7),
            (E::Cow, 0.7),
            (E::Wolf, 0.5),
            (E::Chicken, 0.3),
            (E::Ozelot, 0.4),
            (E::Rabbit, 0.3),
            (E::Squid, 0.7),
            (E::Bat, 0.25),
            (E::Slime, 0.25),
            (E::LavaSlime, 0.25),
            (E::Silverfish, 0.3),
            (E::Endermite, 0.3),
            (E::Spider, 1.0),
            (E::Creeper, 0.5),
            (E::EntityHorse, 0.75),
            (E::ArmorStand, 0.0),
            // `RenderGiantZombie:20` passes `shadowSizeIn * scaleIn` to its super, so the
            // giant's shadow is 0.5 x 6.0 and not the 0.5 this table used to carry.
            (E::Giant, 3.0),
            // java: `RenderCaveSpider:14` — `this.shadowSize *= 0.7F` over `RenderSpider`'s 1.0.
            (E::CaveSpider, 0.7),
        ] {
            assert_eq!(renderer_for(t).unwrap().shadow_size, want, "{}", t.name());
        }
    }

    #[test]
    fn the_giant_is_a_zombie_at_six_times_scale() {
        let g = renderer_for(E::Giant).unwrap();
        let z = renderer_for(E::Zombie).unwrap();
        assert_eq!(g.model, z.model);
        assert_eq!(g.texture, z.texture);
        assert_eq!(g.scale, 6.0);
        assert_eq!(z.scale, 1.0);
        // ...but not a zombie's layer stack: `RenderGiantZombie` extends `RenderLiving`, so it
        // never gets `RenderBiped`'s `LayerCustomHead`.
        assert_eq!(g.layers, &[LayerKind::HeldItem, LayerKind::Armor]);
        assert!(!g.layers.contains(&LayerKind::CustomHead));
    }

    /// Vanilla's `addLayer` order is the order the passes draw in, and
    /// three of these stacks disagreed with it.
    #[test]
    fn layer_stacks_follow_vanillas_add_layer_order() {
        // java: `RenderBiped:27` (four-arg super, `LayerCustomHead`) then `RenderZombie:31`
        // (`LayerHeldItem`) then `:40` (`LayerBipedArmor`).
        for t in [E::Zombie, E::PigZombie] {
            assert_eq!(
                renderer_for(t).unwrap().layers,
                &[LayerKind::CustomHead, LayerKind::HeldItem, LayerKind::Armor],
                "{}",
                t.name()
            );
        }
        // java: `RenderBiped:19` registers the first held-item layer for the three-argument
        // super, then `RenderSkeleton:18` registers another one before `LayerBipedArmor`.
        let skeleton_layers = renderer_for(E::Skeleton).unwrap().layers;
        assert_eq!(skeleton_layers, SKELETON_LAYERS);
        assert_eq!(
            skeleton_layers
                .iter()
                .filter(|&&layer| layer == LayerKind::HeldItem)
                .count(),
            2,
            "skeleton must preserve both vanilla held-item registrations"
        );
        assert_eq!(
            renderer_for_variant(E::Skeleton, 1).unwrap().layers,
            SKELETON_LAYERS,
            "wither skeleton is the same Java renderer with a stateful scale/texture"
        );
        // java: `ArmorStandRenderer:30-32` — the opposite order, because it does its own three
        // `addLayer` calls rather than inheriting `RenderBiped`'s.
        assert_eq!(
            renderer_for(E::ArmorStand).unwrap().layers,
            &[LayerKind::Armor, LayerKind::HeldItem, LayerKind::CustomHead]
        );
        // java: `RenderVillager:20` — one `addLayer`, and it is `LayerCustomHead`. A villager has
        // no armour layer; this used to say `[Armor]`, which is both the wrong layer and a mesh
        // mismatch (`ModelVillager` is not a `ModelBiped`).
        assert_eq!(
            renderer_for(E::Villager).unwrap().layers,
            &[LayerKind::CustomHead]
        );
        // java: `RenderPlayer:35-40` — `CustomHead` is the *last* `addLayer`, not the fourth.
        let player = renderer_for_player(SkinModel::Classic);
        assert_eq!(
            player.layers,
            &[
                LayerKind::Armor,
                LayerKind::HeldItem,
                LayerKind::Arrow,
                LayerKind::Deadmau5Head,
                LayerKind::Cape,
                LayerKind::CustomHead,
            ]
        );
        assert_eq!(player.layers, renderer_for_player(SkinModel::Slim).layers);
    }

    #[test]
    fn cave_spiders_share_the_spider_model_at_seven_tenths() {
        let c = renderer_for(E::CaveSpider).unwrap();
        assert_eq!(c.model, ModelKind::Spider);
        assert_eq!(c.scale, 0.7);
        assert_eq!(c.texture, "textures/entity/spider/cave_spider.png");
    }

    /// java: `getDeathMaxRotation` is overridden in exactly three files under
    /// `client/renderer/entity/` — `RenderSpider.java:18`, `RenderSilverfish.java:16` and
    /// `RenderEndermite.java:16`, all returning 180 — plus `RenderCaveSpider`, which inherits
    /// `RenderSpider`'s. Everything else takes `RendererLivingEntity.java:473`'s 90.
    ///
    /// This test previously asserted the *guardian* was the 180 case and that the spiders were
    /// 90; both were backwards. `RenderGuardian` has no such override, and `RenderDragon` only
    /// *calls* `getDeathMaxRotation` from its own `rotateCorpse` — it does not override it
    /// either, so the dragon descriptor is 90 as well.
    #[test]
    fn the_four_mobs_that_flip_onto_their_backs_are_the_spiders_and_the_burrowers() {
        for t in [E::Spider, E::CaveSpider, E::Silverfish, E::Endermite] {
            assert_eq!(
                renderer_for(t).unwrap().death_max_rotation,
                180.0,
                "{}",
                t.name()
            );
        }
        for t in [E::Zombie, E::Creeper, E::Pig, E::Villager, E::Guardian] {
            assert_eq!(
                renderer_for(t).unwrap().death_max_rotation,
                90.0,
                "{}",
                t.name()
            );
        }
        // Nothing else in the table may claim 180: the override set is closed. The armour stand
        // is the one row that is neither 90 nor 180 — java: `ArmorStandRenderer.rotateCorpse`
        // (`:47-50`) replaces the base method with `rotate(180 - bodyYaw, Y)` and drops the whole
        // `deathTime` branch, so `getDeathMaxRotation` is unreachable for a stand. It is the only
        // 1.8.9 renderer besides `RenderSquid` that does not call `super.rotateCorpse`
        // (`RenderBat`, `RenderIronGolem`, `RenderZombie` and `RenderPlayer` all do).
        for t in mc_data::entity_types() {
            let Some(r) = renderer_for(t) else { continue };
            let expected = match t {
                E::Spider | E::CaveSpider | E::Silverfish | E::Endermite => 180.0,
                E::ArmorStand => 0.0,
                _ => 90.0,
            };
            assert_eq!(r.death_max_rotation, expected, "{}", t.name());
        }
    }

    /// java: three `preRenderCallback` scales this table had at `1.0`. `RenderBat.java:32` is
    /// `scale(0.35F, ...)`; `RenderGhast.java:30-35` works out to a uniform 4.5 because its `f`
    /// is the literal `1.0F`; `RenderPlayer.java:133-137` is `0.9375F`.
    #[test]
    fn the_three_pre_render_scales_that_were_wrong() {
        assert_eq!(renderer_for(E::Bat).unwrap().scale, 0.35);
        assert_eq!(renderer_for(E::Ghast).unwrap().scale, 4.5);
        assert_eq!(renderer_for_player(SkinModel::Classic).scale, 0.9375);
        assert_eq!(renderer_for_player(SkinModel::Slim).scale, 0.9375);
    }

    /// java: `RenderCaveSpider`'s constructor multiplies the inherited `shadowSize` by 0.7 as
    /// well as scaling the body, so both numbers move together.
    #[test]
    fn the_cave_spiders_shadow_shrinks_with_its_body() {
        let c = renderer_for(E::CaveSpider).unwrap();
        assert_eq!(c.shadow_size, 0.7);
        assert_eq!(renderer_for(E::Spider).unwrap().shadow_size, 1.0);
    }

    #[test]
    fn layers_are_attached_to_the_right_mobs() {
        assert!(renderer_for(E::Creeper)
            .unwrap()
            .layers
            .contains(&LayerKind::CreeperCharge));
        assert!(renderer_for(E::Sheep)
            .unwrap()
            .layers
            .contains(&LayerKind::SheepWool));
        assert!(renderer_for(E::Pig)
            .unwrap()
            .layers
            .contains(&LayerKind::Saddle));
        assert!(renderer_for(E::Wolf)
            .unwrap()
            .layers
            .contains(&LayerKind::WolfCollar));
        assert!(renderer_for(E::Enderman)
            .unwrap()
            .layers
            .contains(&LayerKind::HeldBlock));
        assert!(renderer_for(E::Zombie)
            .unwrap()
            .layers
            .contains(&LayerKind::Armor));
        assert!(renderer_for(E::Chicken).unwrap().layers.is_empty());
        // A mooshroom is a cow plus a mushroom layer.
        let moo = renderer_for(E::MushroomCow).unwrap();
        assert_eq!(moo.model, ModelKind::Cow);
        assert!(moo.layers.contains(&LayerKind::MooshroomMushroom));
    }

    #[test]
    fn every_minecart_variant_uses_the_one_model() {
        for t in [
            E::MinecartRideable,
            E::MinecartChest,
            E::MinecartFurnace,
            E::MinecartTNT,
            E::MinecartHopper,
            E::MinecartSpawner,
            E::MinecartCommandBlock,
        ] {
            let r = renderer_for(t).unwrap();
            assert_eq!(r.model, ModelKind::Minecart, "{}", t.name());
            assert_eq!(r.texture, "textures/entity/minecart.png");
        }
    }

    #[test]
    fn variant_textures_resolve_and_fall_back() {
        assert_eq!(
            texture_for(E::Villager, 2),
            "textures/entity/villager/priest.png"
        );
        assert_eq!(texture_for(E::Ozelot, 3), "textures/entity/cat/siamese.png");
        // Type 99 is the Killer Bunny (`CAERBANNOG`), not Toast. `RenderRabbit:57` is the
        // `case 99:` arm; Toast is the name check at `:30`, above the switch.
        assert_eq!(
            texture_for(E::Rabbit, 99),
            "textures/entity/rabbit/caerbannog.png"
        );
        assert_eq!(
            texture_for_named(E::Rabbit, 99, Some("Toast")),
            "textures/entity/rabbit/toast.png",
            "the name wins over the type, as it does above the switch in vanilla"
        );
        assert_eq!(
            texture_for_named(E::Rabbit, 3, Some("Toast")),
            "textures/entity/rabbit/toast.png"
        );
        assert_eq!(
            texture_for_named(E::Rabbit, 3, Some("toast")),
            "textures/entity/rabbit/white_splotched.png",
            "java: String.equals, not equalsIgnoreCase"
        );
        assert_eq!(
            texture_for_named(E::Rabbit, 1, None),
            "textures/entity/rabbit/white.png"
        );
        // A02-14's four previously-missing variants.
        assert_eq!(
            texture_for(E::Zombie, 1),
            "textures/entity/zombie/zombie_villager.png"
        );
        assert_eq!(
            texture_for(E::Ghast, 1),
            "textures/entity/ghast/ghast_shooting.png"
        );
        assert_eq!(
            texture_for(E::Guardian, 1),
            "textures/entity/guardian_elder.png"
        );
        assert_eq!(
            texture_for(E::WitherBoss, 1),
            "textures/entity/wither/wither_invulnerable.png"
        );
        assert_eq!(
            texture_for(E::Skeleton, 1),
            "textures/entity/skeleton/wither_skeleton.png"
        );
        assert_eq!(
            texture_for(E::Skeleton, 0),
            "textures/entity/skeleton/skeleton.png"
        );
        // An out-of-range variant must not panic; it takes the default.
        assert_eq!(
            texture_for(E::Villager, 99),
            "textures/entity/villager/villager.png"
        );
        assert_eq!(texture_for(E::Ozelot, -5), "textures/entity/cat/ocelot.png");
    }

    /// `LayerCustomHead`, `LayerSnowmanHead`,
    /// `LayerMooshroomMushroom` and `LayerHeldItemWitch` each hang geometry off a named model
    /// part via `postRender`; before this there was no way to ask a `ModelKind` for one.
    #[test]
    fn a_named_part_yields_a_matrix_only_for_a_model_that_has_it() {
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());

        // Every biped-shaped model has a head; a creeper does too, and a boat does not.
        for kind in [ModelKind::Player, ModelKind::Zombie, ModelKind::Villager] {
            assert!(
                rm.part_matrix_for(kind, "head", 0.0625, base).is_some(),
                "{kind:?} has a head part"
            );
        }
        assert_eq!(
            rm.part_matrix_for(ModelKind::Boat, "head", 0.0625, base),
            None
        );
        // A part no model has is `None` rather than a panic or the identity.
        assert_eq!(
            rm.part_matrix_for(ModelKind::Player, "tentacle", 0.0625, base),
            None
        );
        // The matrix really carries the part's *live* pose, which is the whole reason
        // `postRender` has to run after the body. A player's head box pivots at the origin, so
        // at rest its local matrix is the identity and `part_matrix_for` returns `base`
        // unchanged; once the head is turned it must not.
        let at_rest = rm
            .part_matrix_for(ModelKind::Player, "head", 0.0625, base)
            .unwrap();
        assert_eq!(at_rest, base, "an unrotated head box pivots at the origin");
        let mut params = AnimParams::still();
        params.net_head_yaw = 45.0;
        params.head_pitch = 20.0;
        let mut out = Vec::new();
        rm.emit_with(ModelKind::Player, base, &params, &mut out);
        let turned = rm
            .part_matrix_for(ModelKind::Player, "head", 0.0625, base)
            .unwrap();
        assert_ne!(turned, base, "a turned head carries its own rotation");
    }

    #[test]
    fn a_named_part_matrix_uses_post_render_not_its_geometry_offset() {
        // `part_matrix_for` backs the Java layer callers that invoke `part.postRender`.  This
        // deliberately nonzero offset is a mutation oracle: `local_matrix` would move the
        // attachment even though Java's `postRender` omits `offsetX/Y/Z`.
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());
        let before = rm
            .part_matrix_for(ModelKind::Witch, "nose", 0.0625, base)
            .expect("witch has a nose");
        let model = rm.model_mut(ModelKind::Witch).model_mut();
        let nose = model.find("nose").expect("witch has a nose");
        model.get_mut(nose).offset = [0.25, -0.5, 0.125];
        assert_eq!(
            rm.part_matrix_for(ModelKind::Witch, "nose", 0.0625, base),
            Some(before),
            "a render-only offset must not move a postRender attachment"
        );
    }

    /// java: `LayerIronGolemFlower` reads `ironGolemRightArm.rotateAngleX` *after* the pose.
    #[test]
    fn a_posed_parts_pitch_is_readable_and_reflects_the_pose() {
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());
        let mut out = Vec::new();

        // Un-posed, the golem's arms sit at zero.
        assert_eq!(
            rm.part_angle_x(ModelKind::IronGolem, "right_arm"),
            Some(0.0)
        );
        assert_eq!(rm.part_angle_x(ModelKind::IronGolem, "no_such_part"), None);

        // Pose it mid-stride and the arm has swung.
        let mut params = AnimParams::still();
        params.limb_swing = 3.0;
        params.limb_swing_amount = 1.0;
        rm.emit_with(ModelKind::IronGolem, base, &params, &mut out);
        let pitch = rm
            .part_angle_x(ModelKind::IronGolem, "right_arm")
            .expect("the golem has a right arm");
        assert!(pitch != 0.0, "a walking golem's arm is not at rest");
    }

    /// java: `LayerSlimeGel` holds `new ModelSlime(0)` while `RenderSlime`'s main model is
    /// `new ModelSlime(16)` — a plain eight-unit shell over a six-unit body with a face.
    ///
    /// `LayerKind::SlimeGel` used to resolve to `ModelKind::Slime`, the body's own geometry, and
    /// `driver.rs` recorded the layer as unwirable because "drawing it would duplicate the body
    /// exactly". That was a consequence of the wrong arm, not a missing model:
    /// `ModelKind::SlimeGel` was already built and tested, with no caller. This asserts the two
    /// really are different geometry, which is what makes the driver arm a one-liner.
    #[test]
    fn the_slime_gel_layer_is_a_different_model_from_the_slime_body() {
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());
        let params = AnimParams::still();

        assert_eq!(
            LayerKind::SlimeGel.model(),
            Some(ModelKind::SlimeGel),
            "the gel is `ModelSlime(0)`, not the main model"
        );

        let mut body = Vec::new();
        rm.emit_with(ModelKind::Slime, base, &params, &mut body);
        let mut gel = Vec::new();
        rm.emit_layer(LayerKind::SlimeGel, &body, &params, base, &mut gel);
        assert_ne!(body, gel, "the gel must not be a copy of the body");

        // java: `ModelSlime(16)` renders `slimeBodies` plus the two eyes and the mouth — four
        // boxes; `ModelSlime(0)` renders `slimeBodies` alone, since the other three renderers
        // are null. Six faces of six vertices each.
        assert_eq!(
            body.len(),
            4 * 6 * 6,
            "body: one cube plus two eyes and a mouth"
        );
        assert_eq!(gel.len(), 6 * 6, "gel: one bare cube");

        // The shell really is the larger of the two: -4..4 against the body's -3..3.
        let span = |v: &[EntityVertex], axis: usize| {
            v.iter().fold(f32::MIN, |a, x| a.max(x.pos[axis]))
                - v.iter().fold(f32::MAX, |a, x| a.min(x.pos[axis]))
        };
        assert!(
            span(&gel, 0) > span(&body, 0),
            "the gel shell wraps the body, not the other way round"
        );
    }

    /// java: `ArmorStandRenderer.rotateCorpse` (`:47`) is `rotate(180 - bodyYaw, Y)` alone: the
    /// `deathTime` branch of `RendererLivingEntity.rotateCorpse` (`:419-430`) is overridden away,
    /// so `getDeathMaxRotation` never runs for a stand.
    #[test]
    fn an_armour_stand_has_no_death_roll() {
        assert_eq!(
            renderer_for(EntityType::ArmorStand)
                .expect("the stand is drawn")
                .death_max_rotation,
            0.0,
            "`rotateCorpse` is overridden with no deathTime branch"
        );
        // The negative control: a mob that does *not* override it keeps the base 90 degrees.
        assert_eq!(
            renderer_for(EntityType::Zombie).unwrap().death_max_rotation,
            90.0
        );
        // java: `RenderSilverfish.getDeathMaxRotation` (`:16`) returns 180 — an override that
        // raises it, proving this column is not a constant.
        assert_eq!(
            renderer_for(EntityType::Silverfish)
                .unwrap()
                .death_max_rotation,
            180.0
        );
    }

    /// **The shared-model defect class, pinned.** `RenderManager` keeps exactly *one* model per
    /// [`ModelKind`], so any per-entity value parked on a model struct is "whichever entity was
    /// posed last". That is invisible in every single-entity test and wrong the moment two of
    /// the species are on screen, which is why nothing caught it for two waves.
    ///
    /// Four fields had it: `WolfModel::shake_time`, `GuardianModel::spikes_animation`,
    /// `IronGolemModel::hold_rose_ticks` and `WitchModel::{holding_item, entity_id}`. All are
    /// now on [`AnimParams`]. This test poses each species *twice through one manager* with
    /// different per-entity inputs and asserts the second pose owes nothing to the first — the
    /// shape that regresses the instant someone adds another such field.
    #[test]
    fn one_shared_model_per_kind_must_not_carry_state_between_two_entities() {
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());
        let mut out = Vec::new();

        // A pose taken through the manager, with a per-entity input applied.
        let mut pose =
            |rm: &mut RenderManager, kind: ModelKind, part: &str, f: &dyn Fn(&mut AnimParams)| {
                let mut p = AnimParams::still();
                f(&mut p);
                out.clear();
                rm.emit_with(kind, base, &p, &mut out);
                rm.part_angle_x(kind, part).expect("the part exists")
            };

        // java: `ModelIronGolem.setLivingAnimations` — `getHoldRoseTick() > 0` lifts only the
        // right arm; zero falls through to the walking swing.
        let poppy = pose(&mut rm, ModelKind::IronGolem, "right_arm", &|p| {
            p.entity.hold_rose_ticks = 40;
        });
        let walking = pose(&mut rm, ModelKind::IronGolem, "right_arm", &|p| {
            p.limb_swing = 3.0;
            p.limb_swing_amount = 1.0;
        });
        assert_ne!(
            poppy, walking,
            "a walking golem must not inherit the previous golem's poppy pose"
        );

        // java: `ModelWitch.setRotationAngles` — `0.01F * (getEntityId() % 10)` is the nose
        // frequency, so two witches with different ids twitch out of phase.
        let witch_a = pose(&mut rm, ModelKind::Witch, "nose", &|p| {
            p.entity.entity_id = 7;
            p.entity.ticks_existed = 25;
        });
        let witch_b = pose(&mut rm, ModelKind::Witch, "nose", &|p| {
            p.entity.entity_id = 3;
            p.entity.ticks_existed = 25;
        });
        assert_ne!(witch_a, witch_b, "two witches must twitch out of phase");
        // And the drinking dip belongs to the witch holding the bottle, not to the next one.
        let drinking = pose(&mut rm, ModelKind::Witch, "nose", &|p| {
            p.entity.entity_id = 3;
            p.entity.ticks_existed = 25;
            p.entity.witch_holding_item = true;
        });
        assert!((drinking - -0.9).abs() < 1e-6, "the drinker dips her nose");
        let after = pose(&mut rm, ModelKind::Witch, "nose", &|p| {
            p.entity.entity_id = 3;
            p.entity.ticks_existed = 25;
        });
        assert_eq!(after, witch_b, "the next witch is not still drinking");

        // java: `ModelArmorStandArmor.setRotationAngles` reads the six `Rotations` slots off the
        // entity; a stand the server never posed holds `DEFAULT_*_ROTATION`.
        let posed = pose(&mut rm, ModelKind::ArmorStand, "left_arm", &|p| {
            p.entity.armor_stand = Some(super::super::models::object::ArmorStandPose {
                left_arm: [80.0, 0.0, 0.0],
                ..Default::default()
            });
        });
        let unposed = pose(&mut rm, ModelKind::ArmorStand, "left_arm", &|_| {});
        assert_ne!(
            posed, unposed,
            "a default stand must not inherit the previous stand's arm"
        );

        // `ArmorStandRenderer` installs `LayerHeldItem`, and `ModelArmorStand` inherits
        // `ModelBiped.postRenderArm`. The matrix must exist, follow the live watched right-arm
        // pose, and retain LayerHeldItem's child transform for a small stand.
        let held_matrix = |rm: &mut RenderManager, pose: [f32; 3], child: bool| {
            let mut p = AnimParams::still();
            p.entity.child = child;
            p.entity.armor_stand = Some(super::super::models::object::ArmorStandPose {
                right_arm: pose,
                // `EntityArmorStand.getShowArms()` gates the arm's `showModel` bit.  The
                // held-item layer can only inherit a posed hand when that arm is actually
                // visible; a plain stand's default is false and Java's `postRenderArm` then
                // leaves the item at the caller's base matrix by design.
                show_arms: true,
                ..Default::default()
            });
            let mut vertices = Vec::new();
            rm.emit_with(ModelKind::ArmorStand, base, &p, &mut vertices);
            rm.held_item_matrix_for(ModelKind::ArmorStand, p.scale, base, child, false, false)
                .expect("ArmorStand inherits the biped right-arm held-item hook")
        };
        let default_item = held_matrix(&mut rm, [-10.0, 0.0, -10.0], false);
        let posed_item = held_matrix(&mut rm, [80.0, 25.0, -35.0], false);
        let small_item = held_matrix(&mut rm, [80.0, 25.0, -35.0], true);
        assert_ne!(
            default_item, posed_item,
            "the held item must follow the watched right-arm rotation"
        );
        assert_ne!(
            posed_item, small_item,
            "a small stand must keep LayerHeldItem's child scale and lift"
        );

        // java: ModelArmorStand.postRenderArm forces bipedRightArm.showModel=true around the
        // inherited ModelBiped.postRenderArm call.  A stand's arms are hidden by default, but a
        // held item must still use the stored right-arm pose rather than the identity matrix.
        let mut hidden = AnimParams::still();
        hidden.entity.armor_stand = Some(super::super::models::object::ArmorStandPose {
            right_arm: [80.0, 25.0, -35.0],
            show_arms: false,
            ..Default::default()
        });
        let mut vertices = Vec::new();
        rm.emit_with(ModelKind::ArmorStand, base, &hidden, &mut vertices);
        let hidden_arm_item = rm
            .held_item_matrix_for(
                ModelKind::ArmorStand,
                hidden.scale,
                base,
                false,
                false,
                false,
            )
            .expect("hidden-arm ArmorStand still has a held-item attachment");

        let mut visible = hidden;
        visible.entity.armor_stand.as_mut().unwrap().show_arms = true;
        rm.emit_with(ModelKind::ArmorStand, base, &visible, &mut vertices);
        let visible_arm_item = rm
            .held_item_matrix_for(
                ModelKind::ArmorStand,
                visible.scale,
                base,
                false,
                false,
                false,
            )
            .expect("visible-arm ArmorStand has a held-item attachment");
        assert_eq!(
            hidden_arm_item, visible_arm_item,
            "ModelArmorStand.postRenderArm must preserve the hand transform when showArms=false"
        );

        // The two wave-1 cases, kept here so the whole class lives in one test. Both move
        // geometry rather than a single named part's pitch, so they are compared on vertices.
        let geometry = |rm: &mut RenderManager, kind: ModelKind, f: &dyn Fn(&mut AnimParams)| {
            let mut p = AnimParams::still();
            f(&mut p);
            let mut v = Vec::new();
            rm.emit_with(kind, base, &p, &mut v);
            v
        };

        // java: `ModelWolf.render` — `getShakeAngle(partialTicks, ..)` off
        // `EntityWolf.timeWolfIsShaking`, applied as a roll to the whole body.
        let shaking = geometry(&mut rm, ModelKind::Wolf, &|p| {
            p.entity.wolf_shake_time = 0.4;
        });
        let dry = geometry(&mut rm, ModelKind::Wolf, &|_| {});
        assert_ne!(shaking, dry, "a dry wolf must not inherit the shake");

        // java: `ModelGuardian.setRotationAngles` — `func_175472_n(partialTicks)` spike extension.
        let spiked = geometry(&mut rm, ModelKind::Guardian, &|p| {
            p.entity.spikes_animation = 1.0;
        });
        let retracted = geometry(&mut rm, ModelKind::Guardian, &|_| {});
        assert_ne!(
            spiked, retracted,
            "a retracted guardian must not inherit the previous one's spikes"
        );
    }

    /// java: `RenderPlayer.setModelVisibilities` (`:44-72`) — the five
    /// overlay parts and the cape, from `EnumPlayerModelParts`' bit mask.
    #[test]
    fn the_player_model_parts_mask_hides_the_overlays() {
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());
        let params = AnimParams::still();

        let count = |rm: &mut RenderManager, mask: i8| {
            rm.set_player_model_parts(ModelKind::Player, mask);
            let mut out = Vec::new();
            rm.emit_with(ModelKind::Player, base, &params, &mut out);
            out.len()
        };

        let all_on = count(&mut rm, !0);
        let all_off = count(&mut rm, 0);
        assert!(
            all_off < all_on,
            "hiding every overlay draws fewer vertices: {all_off} vs {all_on}"
        );

        // Bit 6 is the hat. Turning only it off must remove exactly one box (6 faces x 6 verts).
        let no_hat = count(&mut rm, !(1 << 6));
        assert_eq!(
            all_on - no_hat,
            6 * 6,
            "the hat is one box: {all_on} -> {no_hat}"
        );

        // A model with no wear parts at all is untouched rather than broken.
        rm.set_player_model_parts(ModelKind::Creeper, 0);
        let mut out = Vec::new();
        rm.emit_with(ModelKind::Creeper, base, &params, &mut out);
        assert!(!out.is_empty(), "a creeper has no overlays to hide");
    }

    #[test]
    fn spectator_player_visibility_keeps_only_the_head_and_hat_and_resets_afterward() {
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());
        let params = AnimParams::still();
        let geometry = |rm: &mut RenderManager, spectator| {
            rm.set_player_model_visibility(ModelKind::Player, spectator, !0);
            let mut out = Vec::new();
            rm.emit_with(ModelKind::Player, base, &params, &mut out);
            out
        };

        let ordinary = geometry(&mut rm, false);
        let spectator = geometry(&mut rm, true);
        assert_eq!(
            spectator.len(),
            2 * 6 * 6,
            "vanilla's spectator body is exactly the head and inflated headwear cubes"
        );
        assert!(spectator.len() < ordinary.len());
        assert_eq!(
            geometry(&mut rm, false).len(),
            ordinary.len(),
            "a spectator must not hide parts on the next player sharing this model"
        );
    }

    /// java: `LayerHeldItemWitch` hangs the item off `villagerNose`, which only the witch and the
    /// villager have.
    #[test]
    fn the_witchs_item_matrix_needs_a_nose() {
        use super::super::layers::WitchHeldItem;
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());
        assert!(rm
            .witch_held_item_matrix_for(ModelKind::Witch, 0.0625, base, false, WitchHeldItem::Flat)
            .is_some());
        assert_eq!(
            rm.witch_held_item_matrix_for(
                ModelKind::Zombie,
                0.0625,
                base,
                false,
                WitchHeldItem::Flat
            ),
            None
        );
        // The four poses are genuinely different matrices, not one with a flag.
        let flat = rm
            .witch_held_item_matrix_for(ModelKind::Witch, 0.0625, base, false, WitchHeldItem::Flat)
            .unwrap();
        let bow = rm
            .witch_held_item_matrix_for(ModelKind::Witch, 0.0625, base, false, WitchHeldItem::Bow)
            .unwrap();
        assert_ne!(flat, bow);
    }

    #[test]
    fn slim_player_held_item_nudges_the_arm_pivot_without_mutating_the_shared_model() {
        let base = entity_matrix(&EntityTransform::default())
            * Mat4::from_translation(Vec3::new(0.13, -0.07, 0.21));
        let mut params = AnimParams::still();
        params.entity.child = true;
        params.entity.swing_progress = 0.37;

        // Classic arms inherit ModelBiped.postRenderArm unchanged.
        let mut classic = RenderManager::new();
        let mut classic_vertices = Vec::new();
        classic.emit_with(ModelKind::Player, base, &params, &mut classic_vertices);
        let (classic_arm, classic_raw) = {
            let model = classic.model_mut(ModelKind::Player).model();
            let arm = model.find("right_arm").expect("classic right arm");
            (
                arm,
                super::super::layers::held_item_matrix(
                    model,
                    arm,
                    params.scale,
                    base,
                    params.entity.child,
                    false,
                    false,
                ),
            )
        };
        let classic_pivot_before = classic
            .model_mut(ModelKind::Player)
            .model()
            .get(classic_arm)
            .rotation_point;
        let classic_matrix = classic
            .held_item_matrix_for(
                ModelKind::Player,
                params.scale,
                base,
                params.entity.child,
                false,
                false,
            )
            .expect("classic player right arm");
        assert_eq!(
            classic_matrix, classic_raw,
            "classic arms have no slim nudge"
        );
        assert_eq!(
            classic
                .model_mut(ModelKind::Player)
                .model()
                .get(classic_arm)
                .rotation_point,
            classic_pivot_before,
            "classic attachment leaves the shared pivot untouched"
        );

        // The expected slim matrix is the Java postRenderArm sequence: increment the right-arm
        // X pivot by exactly one model unit for the call, then use the ordinary held-item path.
        let mut slim = RenderManager::new();
        let mut slim_vertices = Vec::new();
        slim.emit_with(ModelKind::PlayerSlim, base, &params, &mut slim_vertices);
        let (slim_arm, expected) = {
            let model = slim.model_mut(ModelKind::PlayerSlim).model();
            let arm = model.find("right_arm").expect("slim right arm");
            let mut shifted = model.clone();
            shifted.get_mut(arm).rotation_point[0] += 1.0;
            (
                arm,
                super::super::layers::held_item_matrix(
                    &shifted,
                    arm,
                    params.scale,
                    base,
                    params.entity.child,
                    false,
                    false,
                ),
            )
        };
        let pivot_before = slim
            .model_mut(ModelKind::PlayerSlim)
            .model()
            .get(slim_arm)
            .rotation_point;
        let matrix = slim
            .held_item_matrix_for(
                ModelKind::PlayerSlim,
                params.scale,
                base,
                params.entity.child,
                false,
                false,
            )
            .expect("slim player right arm");
        assert_eq!(
            matrix, expected,
            "slim attachment applies Java's +1 pivot unit"
        );
        assert_eq!(
            slim.model_mut(ModelKind::PlayerSlim)
                .model()
                .get(slim_arm)
                .rotation_point,
            pivot_before,
            "the temporary Java pivot adjustment must be restored"
        );
        assert_eq!(
            slim.held_item_matrix_for(
                ModelKind::PlayerSlim,
                params.scale,
                base,
                params.entity.child,
                false,
                false,
            )
            .expect("repeat slim player right arm"),
            matrix,
            "repeated attachments are stable and do not accumulate the nudge"
        );
    }

    #[test]
    fn armor_stand_hidden_held_item_uses_the_posed_arm_and_restores_visibility() {
        let base = entity_matrix(&EntityTransform::default())
            * Mat4::from_translation(Vec3::new(0.17, -0.09, 0.23));
        let pose = [80.0, 25.0, -35.0];

        let attachment = |show_arms: bool| {
            let mut rm = RenderManager::new();
            let mut params = AnimParams::still();
            params.entity.armor_stand = Some(super::super::models::object::ArmorStandPose {
                right_arm: pose,
                show_arms,
                ..Default::default()
            });
            let mut vertices = Vec::new();
            rm.emit_with(ModelKind::ArmorStand, base, &params, &mut vertices);
            let arm = rm
                .model_mut(ModelKind::ArmorStand)
                .model()
                .find("right_arm")
                .expect("armor stand right arm");
            let matrix = rm
                .held_item_matrix_for(
                    ModelKind::ArmorStand,
                    params.scale,
                    base,
                    params.entity.child,
                    false,
                    false,
                )
                .expect("armor stands have a right-hand attachment");
            let visible_after = rm
                .model_mut(ModelKind::ArmorStand)
                .model()
                .get(arm)
                .show_model;
            (matrix, visible_after)
        };

        // Java's ModelArmorStand.postRenderArm must retain the nonzero pose even
        // though setRotationAngles hid the physical arm for a plain stand.
        let (hidden_matrix, hidden_after) = attachment(false);
        let (visible_matrix, visible_after) = attachment(true);
        assert_ne!(
            hidden_matrix, base,
            "the hidden hand still has a posed attachment"
        );
        assert_eq!(
            hidden_matrix, visible_matrix,
            "showArms only hides geometry, not the hand transform"
        );
        assert!(
            !hidden_after,
            "the temporary showModel=true must be restored"
        );
        assert!(visible_after, "a visible stand arm must remain visible");

        // Reusing one hidden stand must not accumulate a visibility mutation or
        // change the matrix on the second LayerHeldItem pass.
        let mut rm = RenderManager::new();
        let mut params = AnimParams::still();
        params.entity.armor_stand = Some(super::super::models::object::ArmorStandPose {
            right_arm: pose,
            show_arms: false,
            ..Default::default()
        });
        let mut vertices = Vec::new();
        rm.emit_with(ModelKind::ArmorStand, base, &params, &mut vertices);
        let first = rm
            .held_item_matrix_for(
                ModelKind::ArmorStand,
                params.scale,
                base,
                params.entity.child,
                false,
                false,
            )
            .expect("first hidden-arm attachment");
        let second = rm
            .held_item_matrix_for(
                ModelKind::ArmorStand,
                params.scale,
                base,
                params.entity.child,
                false,
                false,
            )
            .expect("repeat hidden-arm attachment");
        assert_eq!(first, second, "repeated hidden-arm attachments are stable");
        let arm = rm
            .model_mut(ModelKind::ArmorStand)
            .model()
            .find("right_arm")
            .expect("armor stand right arm after repeat");
        assert!(
            !rm.model_mut(ModelKind::ArmorStand)
                .model()
                .get(arm)
                .show_model
        );
    }

    #[test]
    fn models_are_built_once_and_shared() {
        let mut rm = RenderManager::new();
        assert_eq!(rm.built_models(), 0);
        let mut out = Vec::new();
        let p = AnimParams::still();
        let t = EntityTransform::default();
        rm.emit(E::Zombie, 0, &t, &p, &mut out);
        assert_eq!(rm.built_models(), 1);
        rm.emit(E::Zombie, 0, &t, &p, &mut out);
        rm.emit(E::PigZombie, 0, &t, &p, &mut out);
        assert_eq!(rm.built_models(), 1, "all three share ModelZombie");
        rm.emit(E::Creeper, 0, &t, &p, &mut out);
        assert_eq!(rm.built_models(), 2);
    }

    /// `RenderWitherSkull` owns a no-argument `ModelSkeletonHead` (64×64, V offset 35), while
    /// tile skulls own the explicitly-sized 64×32 model. The manager must cache those as two
    /// model instances, and the UV comparison is the negative control that catches accidentally
    /// routing the projectile back through `ModelKind::SkeletonHead`.
    #[test]
    fn wither_skull_emission_uses_a_distinct_cached_uv_layout() {
        let mut rm = RenderManager::new();
        let params = AnimParams::still();
        let mut tile = Vec::new();
        rm.emit_with(ModelKind::SkeletonHead, Mat4::IDENTITY, &params, &mut tile);
        assert_eq!(rm.built_models(), 1);

        let mut wither = Vec::new();
        rm.emit_wither_skull(Mat4::IDENTITY, &params, &mut wither);
        assert_eq!(
            rm.built_models(),
            2,
            "the projectile gets its own cache key"
        );
        assert_eq!(tile.len(), wither.len(), "both heads are one 8-cube model");
        assert!(tile.iter().any(|v| v.uv == [8.0 / 64.0, 8.0 / 32.0]));
        assert!(wither.iter().any(|v| v.uv == [8.0 / 64.0, 43.0 / 64.0]));
        assert_ne!(
            tile, wither,
            "tile and wither heads must not share their UV/texture-size constructor"
        );

        let mut wither_again = Vec::new();
        rm.emit_wither_skull(Mat4::IDENTITY, &params, &mut wither_again);
        assert_eq!(rm.built_models(), 2, "the dedicated path is cached");
        assert_eq!(wither_again, wither);
    }

    /// `RenderZombie.func_82427_a` changes more than the texture: a watched villager zombie
    /// uses `ModelZombieVillager` and the layer list `RenderZombie` saved after replacing its
    /// ordinary head/armour layers.  Keeping the assertion at the public dispatch boundary makes
    /// a texture-only regression fail before it reaches a live client submitter.
    #[test]
    fn zombie_villager_variant_selects_its_body_and_layer_renderer() {
        let ordinary = renderer_for_variant(E::Zombie, 0).expect("zombies render");
        let villager = renderer_for_variant(E::Zombie, 1).expect("villager zombies render");

        assert_eq!(ordinary.model, ModelKind::Zombie);
        assert_eq!(ordinary.texture, "textures/entity/zombie/zombie.png");
        assert_eq!(ordinary.layers, BIPED_LAYERS);

        assert_eq!(villager.model, ModelKind::ZombieVillager);
        assert_eq!(
            villager.texture,
            "textures/entity/zombie/zombie_villager.png"
        );
        assert_eq!(villager.layers, ZOMBIE_VILLAGER_LAYERS);
        assert_ne!(
            ordinary.layers, villager.layers,
            "villager swaps layer order too"
        );

        // Only EntityZombie's `isVillager()` bit takes this branch.  A pig zombie happens to
        // share the ordinary body model but must not inherit the villager renderer by value.
        assert_eq!(
            renderer_for_variant(E::PigZombie, 1)
                .expect("pig zombies render")
                .model,
            ModelKind::Zombie
        );
    }

    /// The specialized head has one extra nose cube.  `RenderManager::emit` must select it from
    /// the same watched variant that selects the texture, while ordinary zombies remain on the
    /// original cached `ModelZombie`.
    #[test]
    fn zombie_villager_emit_builds_the_specialized_model_without_mutating_zombies() {
        let mut rm = RenderManager::new();
        let params = AnimParams::still();
        let transform = EntityTransform::default();
        let mut ordinary = Vec::new();
        let mut villager = Vec::new();

        assert_eq!(
            rm.emit(E::Zombie, 0, &transform, &params, &mut ordinary),
            Some("textures/entity/zombie/zombie.png")
        );
        assert_eq!(rm.built_models(), 1);
        assert_eq!(
            rm.emit(E::Zombie, 1, &transform, &params, &mut villager),
            Some("textures/entity/zombie/zombie_villager.png")
        );
        assert_eq!(
            rm.built_models(),
            2,
            "the villager body is a distinct cache key"
        );
        assert_eq!(
            villager.len(),
            ordinary.len() + 6 * 6,
            "the villager nose is one additional cube"
        );

        let mut ordinary_again = Vec::new();
        rm.emit(E::Zombie, 0, &transform, &params, &mut ordinary_again);
        assert_eq!(ordinary_again, ordinary, "ordinary zombies stay ordinary");
    }

    #[test]
    fn zombie_skeleton_and_pigman_armour_keeps_the_arms_out_pose() {
        let mut rm = RenderManager::new();
        let params = AnimParams::still();
        let base = entity_matrix(&EntityTransform::default());

        let armour_for = |rm: &mut RenderManager, entity, variant| {
            let body_kind = renderer_for_variant(entity, variant)
                .expect("the living entity has a renderer")
                .model;
            let mut out = Vec::new();
            rm.emit_armor_slot_for(body_kind, ArmorSlot::Chestplate, &params, base, &mut out);
            out
        };

        let zombie = armour_for(&mut rm, E::Zombie, 0);
        let pigman = armour_for(&mut rm, E::PigZombie, 0);
        let skeleton = armour_for(&mut rm, E::Skeleton, 0);
        assert_eq!(
            zombie, pigman,
            "pig-zombie sleeves share ModelZombie's pose"
        );
        assert_eq!(zombie, skeleton, "skeleton sleeves share the arms-out pose");

        // java: RenderSkeleton's armour uses `ModelSkeleton(inflation, true)`. That class still
        // extends ModelZombie: ModelBiped applies aimedBow first, then ModelZombie's final tail
        // overwrites both arms. A Wither Skeleton's armour therefore remains arms-out too; a
        // skeleton-specific generic-biped bow pose would be a plausible-looking divergence.
        let mut aimed = AnimParams::still();
        aimed.entity.aimed_bow = true;
        let mut aimed_skeleton = Vec::new();
        rm.emit_armor_slot_for(
            ModelKind::Skeleton,
            ArmorSlot::Chestplate,
            &aimed,
            base,
            &mut aimed_skeleton,
        );
        assert_eq!(
            skeleton, aimed_skeleton,
            "ModelSkeleton armour inherits ModelZombie's post-bow arms-out tail"
        );

        let mut player = Vec::new();
        rm.emit_armor_slot_for(
            ModelKind::Biped,
            ArmorSlot::Chestplate,
            &params,
            base,
            &mut player,
        );
        assert_ne!(
            zombie, player,
            "a plain biped pose must not masquerade as zombie-family armour"
        );
    }

    #[test]
    fn armor_stand_worn_armor_uses_the_stands_stored_limb_pose() {
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());
        let mut posed = AnimParams::still();
        posed.entity.armor_stand = Some(super::super::models::object::ArmorStandPose {
            head: [20.0, -30.0, 15.0],
            body: [-25.0, 35.0, -10.0],
            right_arm: [70.0, 25.0, -35.0],
            left_arm: [-55.0, 10.0, 40.0],
            right_leg: [45.0, -20.0, 30.0],
            left_leg: [-40.0, 15.0, -25.0],
            ..Default::default()
        });

        for slot in ArmorSlot::DRAW_ORDER {
            let mut stand_posed = Vec::new();
            rm.emit_armor_slot_for(ModelKind::ArmorStand, slot, &posed, base, &mut stand_posed);
            let mut stand_default = Vec::new();
            rm.emit_armor_slot_for(
                ModelKind::ArmorStand,
                slot,
                &AnimParams::still(),
                base,
                &mut stand_default,
            );
            assert_ne!(
                stand_posed, stand_default,
                "{slot:?} armor must follow the stand's stored rotations"
            );
        }
        assert_eq!(
            rm.armor_stand_models.len(),
            2,
            "stand renderer caches Java's 1.0 armor and 0.5 leggings models"
        );

        let mut biped_posed = Vec::new();
        rm.emit_armor_slot_for(
            ModelKind::Biped,
            ArmorSlot::Chestplate,
            &posed,
            base,
            &mut biped_posed,
        );
        let mut biped_default = Vec::new();
        rm.emit_armor_slot_for(
            ModelKind::Biped,
            ArmorSlot::Chestplate,
            &AnimParams::still(),
            base,
            &mut biped_default,
        );
        assert_eq!(
            biped_posed, biped_default,
            "the stand-only pose must not leak into ordinary biped armor"
        );
    }

    /// Animation inputs that exercise every branch of the biped, zombie and stand poses.
    fn busy_params() -> Vec<AnimParams> {
        let mut out = vec![AnimParams::still()];
        let mut walking = AnimParams::still();
        walking.limb_swing = 12.5;
        walking.limb_swing_amount = 0.7;
        walking.age_in_ticks = 40.25;
        walking.net_head_yaw = 25.0;
        walking.head_pitch = -15.0;
        walking.entity.swing_progress = 0.35;
        walking.entity.held_item_right = 1;
        out.push(walking.clone());
        let mut sneaking = walking.clone();
        sneaking.entity.sneaking = true;
        sneaking.entity.aimed_bow = true;
        sneaking.entity.held_item_right = 3;
        sneaking.entity.held_item_left = 1;
        out.push(sneaking);
        let mut child = walking.clone();
        child.entity.child = true;
        out.push(child);
        let mut stand = walking;
        stand.entity.armor_stand = Some(super::super::models::object::ArmorStandPose {
            head: [20.0, -30.0, 15.0],
            body: [-25.0, 35.0, -10.0],
            right_arm: [70.0, 25.0, -35.0],
            left_arm: [-55.0, 10.0, 40.0],
            right_leg: [45.0, -20.0, 30.0],
            left_leg: [-40.0, 15.0, -25.0],
            ..Default::default()
        });
        out.push(stand);
        out
    }

    /// #1 step 1 (d): posing each armour model once per entity emits the same four-slot stream,
    /// bit for bit, as re-posing per slot — for a player, a zombie, a zombie villager and an
    /// armour stand, with entities interleaved so a stale pose would show up in the next one.
    #[test]
    fn armor_posed_once_per_entity_matches_per_slot_posing_bit_for_bit() {
        let base = entity_matrix(&EntityTransform {
            rel: glam::Vec3::new(2.5, -1.0, 6.0),
            body_yaw: 37.5,
            ..EntityTransform::default()
        });
        let (mut per_slot, mut once) = (RenderManager::new(), RenderManager::new());
        for body in [
            ModelKind::Player,
            ModelKind::Zombie,
            ModelKind::ZombieVillager,
            ModelKind::ArmorStand,
        ] {
            for params in busy_params() {
                let (mut want, mut got) = (Vec::new(), Vec::new());
                let mut poses = ArmorPoses::default();
                for slot in ArmorSlot::DRAW_ORDER {
                    per_slot.emit_armor_slot_for(body, slot, &params, base, &mut want);
                    once.emit_armor_slot_for_entity(
                        body, slot, &params, base, &mut poses, &mut got,
                    );
                }
                assert!(!want.is_empty());
                assert_eq!(
                    bytemuck::cast_slice::<_, u32>(&got[..]),
                    bytemuck::cast_slice::<_, u32>(&want[..]),
                    "{body:?}"
                );
            }
        }
    }

    /// java: `LayerArmorBase.renderLayer` copies the main model's `swingProgress` and `isRiding`
    /// into the armour model before posing it (`ModelBase.setModelAttributes`), so a chestplate's
    /// sleeves follow an attack swing and leggings take the riding pose (phase 4c review N5).
    #[test]
    fn armour_follows_the_attack_swing_and_the_riding_pose() {
        let base = entity_matrix(&EntityTransform::default());
        let emit = |body: ModelKind, slot: ArmorSlot, params: &AnimParams| {
            let mut out = Vec::new();
            RenderManager::new().emit_armor_slot_for_entity(
                body,
                slot,
                params,
                base,
                &mut ArmorPoses::default(),
                &mut out,
            );
            assert!(!out.is_empty());
            bytemuck::cast_slice::<_, u32>(&out[..]).to_vec()
        };
        for body in [ModelKind::Player, ModelKind::Zombie] {
            let still = AnimParams::still();
            let mut swinging = still.clone();
            swinging.entity.swing_progress = 0.35;
            assert_ne!(
                emit(body, ArmorSlot::Chestplate, &swinging),
                emit(body, ArmorSlot::Chestplate, &still),
                "{body:?}: the chestplate sleeve must follow the swing"
            );
            let mut riding = still.clone();
            riding.entity.riding = true;
            assert_ne!(
                emit(body, ArmorSlot::Leggings, &riding),
                emit(body, ArmorSlot::Leggings, &still),
                "{body:?}: leggings must take the riding pose"
            );
        }
    }

    /// #11 (f): `pose_with` leaves every model in exactly the state `emit_with` leaves, so a
    /// skipped invisible body still hands layers (held item, head attachments) its own pose.
    #[test]
    fn pose_with_leaves_the_same_model_state_as_emit_with() {
        let base = entity_matrix(&EntityTransform::default());
        for &kind in ModelKind::all() {
            let (mut emitted, mut posed) = (RenderManager::new(), RenderManager::new());
            for params in busy_params() {
                let mut sink = Vec::new();
                emitted.emit_with(kind, base, &params, &mut sink);
                posed.pose_with(kind, &params);
                let (a, b) = (
                    emitted.model_mut(kind).model(),
                    posed.model_mut(kind).model(),
                );
                assert_eq!(
                    a.swing_progress.to_bits(),
                    b.swing_progress.to_bits(),
                    "{kind:?}"
                );
                assert_eq!(
                    (a.is_riding, a.is_child),
                    (b.is_riding, b.is_child),
                    "{kind:?}"
                );
                for (pa, pb) in a.parts().iter().zip(b.parts()) {
                    let state = |p: &crate::entity::model::ModelPart| {
                        let f = |v: [f32; 3]| v.map(f32::to_bits);
                        (
                            f(p.rotate_angle),
                            f(p.rotation_point),
                            f(p.offset),
                            p.show_model,
                            p.is_hidden,
                        )
                    };
                    assert_eq!(state(pa), state(pb), "{kind:?} part {}", pa.name);
                }
            }
        }
    }

    /// `LayerVillagerArmor.initArmor` creates the two `ModelZombieVillager(..., true)` copies,
    /// whose helmet cube starts two model units higher.  Verify the model-family selection rather
    /// than merely the emitted vertex count, which would miss a normal armour model at the wrong
    /// height.
    #[test]
    fn zombie_villager_armour_uses_the_raised_helmet_model() {
        let mut rm = RenderManager::new();
        let params = AnimParams::still();
        let base = entity_matrix(&EntityTransform::default());
        let mut ordinary = Vec::new();
        let mut villager = Vec::new();

        rm.emit_armor_slot_for(
            ModelKind::Zombie,
            ArmorSlot::Helmet,
            &params,
            base,
            &mut ordinary,
        );
        rm.emit_armor_slot_for(
            ModelKind::ZombieVillager,
            ArmorSlot::Helmet,
            &params,
            base,
            &mut villager,
        );
        assert_eq!(
            rm.built_armor_models(),
            2,
            "each body family owns vanilla's armour pair"
        );

        let helmet_top = |kind| {
            let armour = rm
                .armor_models
                .get(&(kind, 1.0_f32.to_bits()))
                .expect("the selected armour family was built");
            armour.model.get(armour.parts.head).cubes[0].pos1[1]
        };
        let ordinary_head = helmet_top(ArmorModelKind::Zombie);
        let villager_head = helmet_top(ArmorModelKind::ZombieVillager);
        assert_eq!(ordinary_head, -8.0);
        assert_eq!(
            villager_head, -10.0,
            "the helmet caps the ten-unit villager head"
        );
        assert_ne!(
            ordinary, villager,
            "the raised helmet changes emitted geometry"
        );
    }

    #[test]
    fn emit_scales_by_the_descriptor_and_reports_the_texture() {
        let mut rm = RenderManager::new();
        let p = AnimParams::still();
        let t = EntityTransform::default();

        let mut small = Vec::new();
        let tex = rm.emit(E::Zombie, 0, &t, &p, &mut small).unwrap();
        assert_eq!(tex, "textures/entity/zombie/zombie.png");

        let mut big = Vec::new();
        rm.emit(E::Giant, 0, &t, &p, &mut big).unwrap();
        assert_eq!(small.len(), big.len(), "same model, same vertex count");

        // The giant's geometry really is six times the size. Measure the bounding box, which
        // scales linearly whatever the lift does.
        let extent = |v: &[EntityVertex]| {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for x in v {
                lo = lo.min(x.pos[1]);
                hi = hi.max(x.pos[1]);
            }
            hi - lo
        };
        let ratio = extent(&big) / extent(&small);
        assert!((ratio - 6.0).abs() < 0.001, "{ratio}");
    }

    #[test]
    fn emit_composes_size_and_squish_state_before_both_slime_model_lifts() {
        let mut rm = RenderManager::new();
        let t = EntityTransform::default();
        // Cover both the normal slime and its magma subclass, with size and squish changing
        // independently. A fixed descriptor scale or a stale `tail_rotation` transcription can
        // otherwise make one of these pairs look correct by accident.
        for kind in [E::Slime, E::LavaSlime] {
            let mut size_one_idle = Vec::new();
            rm.emit(kind, 1, &t, &AnimParams::still(), &mut size_one_idle)
                .expect("slime-family renderer");

            let mut size_four_landing = AnimParams::still();
            size_four_landing.entity.squish_factor = -0.165;
            let mut size_four = Vec::new();
            rm.emit(kind, 4, &t, &size_four_landing, &mut size_four)
                .expect("slime-family renderer");
            assert_eq!(size_one_idle.len(), size_four.len());
            assert_ne!(
                size_one_idle, size_four,
                "{kind:?}: size callback must affect emitted vertices"
            );

            let mut size_four_takeoff = AnimParams::still();
            size_four_takeoff.entity.squish_factor = 0.33;
            let mut takeoff = Vec::new();
            rm.emit(kind, 4, &t, &size_four_takeoff, &mut takeoff)
                .expect("slime-family renderer");
            assert_ne!(
                size_four, takeoff,
                "{kind:?}: squish callback must affect emitted vertices"
            );
        }
    }

    #[test]
    fn emit_with_state_composes_creeper_flash_before_the_model_lift() {
        let mut rm = RenderManager::new();
        let t = EntityTransform::default();
        let p = AnimParams::still();
        let mut neutral = Vec::new();
        rm.emit_with_state(E::Creeper, 0, &t, &p, 0.0, &mut neutral)
            .expect("creeper renderer");
        let mut flashing = Vec::new();
        rm.emit_with_state(E::Creeper, 0, &t, &p, 0.75, &mut flashing)
            .expect("creeper renderer");
        assert_eq!(neutral.len(), flashing.len());
        assert_ne!(
            neutral, flashing,
            "a mutant that drops RenderCreeper.preRenderCallback would emit identical vertices"
        );
    }

    /// java: `LayerArmorBase.doRenderLayer` draws four passes with two model instances —
    /// `modelLeggings` at `0.5F` for slot 2, `modelArmor` at `1.0F` for the other three — and
    /// each pass shows only the parts that slot covers.
    #[test]
    fn armour_passes_share_two_models_and_show_only_their_own_parts() {
        let mut rm = RenderManager::new();
        let p = AnimParams::still();
        let base = entity_matrix(&EntityTransform::default());

        let mut per_slot = Vec::new();
        for slot in ArmorSlot::DRAW_ORDER {
            let mut out = Vec::new();
            rm.emit_armor_slot(slot, &p, base, &mut out);
            assert!(!out.is_empty(), "{slot:?} drew nothing");
            per_slot.push((slot, out.len()));
        }
        assert_eq!(
            rm.built_armor_models(),
            2,
            "one model per inflation, exactly as vanilla keeps modelArmor and modelLeggings"
        );

        // The helmet shows head + headwear, the chestplate body + both arms, the leggings body +
        // both legs, the boots both legs. Every part is one box, so vertex counts rank the same
        // way the part counts do: boots (2) < helmet (2, but headwear is an inflated copy of the
        // head, same box count) < chestplate (3) == leggings (3).
        let count = |want: ArmorSlot| {
            per_slot
                .iter()
                .find(|(slot, _)| *slot == want)
                .expect("emitted above")
                .1
        };
        assert_eq!(count(ArmorSlot::Boots), count(ArmorSlot::Helmet));
        assert!(count(ArmorSlot::Chestplate) > count(ArmorSlot::Boots));
        assert_eq!(count(ArmorSlot::Chestplate), count(ArmorSlot::Leggings));
    }

    /// The leggings model is built at `0.5F` and the rest at `1.0F`, so the leggings really are
    /// the thinner copy. java: `LayerArmorBase.getArmorModel`.
    #[test]
    fn the_leggings_pass_is_the_thinner_model() {
        let mut rm = RenderManager::new();
        let p = AnimParams::still();
        let base = entity_matrix(&EntityTransform::default());
        let extent = |slot| {
            let mut out = Vec::new();
            RenderManager::new().emit_armor_slot(slot, &p, base, &mut out);
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for v in &out {
                lo = lo.min(v.pos[0]);
                hi = hi.max(v.pos[0]);
            }
            hi - lo
        };
        assert!(
            extent(ArmorSlot::Leggings) < extent(ArmorSlot::Chestplate),
            "leggings 0.5 inflation vs chestplate 1.0"
        );
        let _ = &mut rm;
    }

    /// java: `LayerArmorBase.doRenderLayer` copies `isChild` onto the armour model before every
    /// pass; `emit_armor` cannot do it itself because `BipedModel` has no entity handle.
    #[test]
    fn a_baby_wears_baby_armour() {
        let mut rm = RenderManager::new();
        let base = entity_matrix(&EntityTransform::default());
        let mut adult = Vec::new();
        rm.emit_armor_slot(ArmorSlot::Helmet, &AnimParams::still(), base, &mut adult);

        let child_params = AnimParams {
            entity: super::super::anim::EntityAnimState {
                child: true,
                ..Default::default()
            },
            ..AnimParams::still()
        };
        let mut child = Vec::new();
        rm.emit_armor_slot(ArmorSlot::Helmet, &child_params, base, &mut child);
        assert_ne!(adult, child, "the child branch must change the geometry");
    }

    #[test]
    fn an_unrenderable_type_emits_nothing_rather_than_panicking() {
        let mut rm = RenderManager::new();
        let mut out = Vec::new();
        assert!(rm
            .emit(
                E::Mob,
                0,
                &EntityTransform::default(),
                &AnimParams::still(),
                &mut out
            )
            .is_none());
        assert!(out.is_empty());
    }

    #[test]
    fn layer_models_are_built_once_per_kind_and_inflation() {
        let mut rm = RenderManager::new();
        assert_eq!(rm.built_layer_models(), 0);
        rm.layer_model_mut(ModelKind::Creeper, 2.0);
        assert_eq!(rm.built_layer_models(), 1);
        rm.layer_model_mut(ModelKind::Creeper, 2.0);
        assert_eq!(rm.built_layer_models(), 1, "same (kind, inflation) reuses");
        rm.layer_model_mut(ModelKind::Creeper, 0.0);
        assert_eq!(
            rm.built_layer_models(),
            2,
            "a different inflation is a separate cache slot"
        );
        rm.layer_model_mut(ModelKind::Biped, 0.5);
        assert_eq!(rm.built_layer_models(), 3);
        // The layer cache is independent of the body cache.
        assert_eq!(rm.built_models(), 0);
    }

    /// java: `LayerCreeperCharge` draws a second, fatter `ModelCreeper` under the same pose.
    #[test]
    fn emit_layer_with_a_model_builds_an_inflated_second_copy() {
        let mut rm = RenderManager::new();
        let p = AnimParams::still();

        let mut plain = Vec::new();
        rm.model_mut(ModelKind::Creeper)
            .pose_and_render(&p, Mat4::IDENTITY, &mut plain);

        let mut aura = Vec::new();
        rm.emit_layer(
            LayerKind::CreeperCharge,
            &plain,
            &p,
            Mat4::IDENTITY,
            &mut aura,
        );
        assert!(!aura.is_empty());
        assert_eq!(aura.len(), plain.len(), "same part tree, same vertex count");

        let extent = |v: &[EntityVertex]| {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for x in v {
                lo = lo.min(x.pos[0]);
                hi = hi.max(x.pos[0]);
            }
            hi - lo
        };
        assert!(
            extent(&aura) > extent(&plain),
            "the charge aura ({}) must be fatter than the body ({})",
            extent(&aura),
            extent(&plain)
        );
        assert_eq!(rm.built_layer_models(), 1);
    }

    /// java: `LayerEndermanEyes`/... call `getMainModel().render(...)` again — no separate
    /// model, just the body's own already-posed geometry under another texture. `LayerSaddle`
    /// owns an inflated `ModelPig(0.5F)` and is tested separately below.
    #[test]
    fn emit_layer_without_a_model_reuses_the_bodys_own_geometry() {
        let mut rm = RenderManager::new();
        let p = AnimParams::still();
        let mut body = Vec::new();
        rm.model_mut(ModelKind::Pig)
            .pose_and_render(&p, Mat4::IDENTITY, &mut body);
        assert!(!body.is_empty());

        for layer in [
            LayerKind::EndermanEyes,
            LayerKind::SpiderEyes,
            LayerKind::WolfCollar,
            LayerKind::MooshroomMushroom,
        ] {
            assert_eq!(layer.model(), None, "{layer:?} should reuse the body");
            let mut out = Vec::new();
            rm.emit_layer(layer, &body, &p, Mat4::IDENTITY, &mut out);
            assert_eq!(out, body, "{layer:?} must copy the body verbatim");
        }
        // Reusing the body draws no new model.
        assert_eq!(rm.built_layer_models(), 0);
    }

    // ---- player dispatch (bodies for remote players) -------------------------------------

    /// java: `LayerSaddle` owns `new ModelPig(0.5F)`, copies the main pose onto it, and draws
    /// that expanded shell under `pig_saddle.png`.
    #[test]
    fn pig_saddle_uses_an_inflated_second_pig_model() {
        let mut rm = RenderManager::new();
        let p = AnimParams::still();
        let mut body = Vec::new();
        rm.model_mut(ModelKind::Pig)
            .pose_and_render(&p, Mat4::IDENTITY, &mut body);
        let mut saddle = Vec::new();
        rm.emit_layer(LayerKind::Saddle, &body, &p, Mat4::IDENTITY, &mut saddle);

        assert_eq!(saddle.len(), body.len(), "same pig box tree");
        assert_ne!(
            saddle, body,
            "the saddle must not be coplanar with the body"
        );
        let x_extent = |vertices: &[EntityVertex]| {
            let (mut min, mut max) = (f32::MAX, f32::MIN);
            for vertex in vertices {
                min = min.min(vertex.pos[0]);
                max = max.max(vertex.pos[0]);
            }
            max - min
        };
        assert!(
            x_extent(&saddle) > x_extent(&body),
            "the half-unit saddle model must be wider than the body"
        );
        assert_eq!(rm.built_layer_models(), 1);
    }

    #[test]
    fn renderer_for_player_picks_the_matching_arm_model() {
        let classic = renderer_for_player(SkinModel::Classic);
        assert_eq!(classic.model, ModelKind::Player);
        assert_eq!(classic.texture, "textures/entity/steve.png");

        let slim = renderer_for_player(SkinModel::Slim);
        assert_eq!(slim.model, ModelKind::PlayerSlim);
        assert_eq!(slim.texture, "textures/entity/alex.png");
    }

    #[test]
    fn stuck_arrow_sampling_keeps_java_box_list_order_and_rng_draws() {
        let mut models = RenderManager::new();
        let classic =
            models.stuck_arrow_attachments(ModelKind::Player, 0, 3, 0.0625, Mat4::IDENTITY);
        assert_eq!(
            classic.iter().map(|a| a.part_index).collect::<Vec<_>>(),
            [11, 4, 5]
        );
        assert!(classic.iter().all(|a| a.cube_index == 0));
        assert!((classic[0].yaw - 117.906_63).abs() < 1.0e-4);
        assert!((classic[0].pitch - -19.910_53).abs() < 1.0e-4);

        let slim =
            models.stuck_arrow_attachments(ModelKind::PlayerSlim, 0, 3, 0.0625, Mat4::IDENTITY);
        assert_eq!(
            slim.iter().map(|a| a.part_index).collect::<Vec<_>>(),
            [1, 16, 9]
        );
        assert_eq!(slim[2].point.x, 0.009_722_97);
        assert!(models
            .stuck_arrow_attachments(ModelKind::Player, 7, 0, 0.0625, Mat4::IDENTITY)
            .is_empty());
    }

    #[test]
    fn renderer_for_player_matches_vanillas_render_player_constants() {
        // java: `RenderManager`'s player entries both pass shadowSize 0.5 to `super(...)` and
        // `RenderPlayer` does not override `getDeathMaxRotation`. It *does* override
        // `preRenderCallback` (`RenderPlayer.java:133-137`, `float f = 0.9375F`) — this
        // assertion used to read `1.0` with a comment saying there was no such override, which
        // is what kept every player body 6.7% oversized.
        for model in [SkinModel::Classic, SkinModel::Slim] {
            let r = renderer_for_player(model);
            assert_eq!(r.shadow_size, 0.5, "{model:?}");
            assert_eq!(r.scale, 0.9375, "{model:?}");
            assert_eq!(r.death_max_rotation, 90.0, "{model:?}");
        }
    }

    #[test]
    fn renderer_for_player_carries_the_render_player_layer_stack() {
        // java: `RenderPlayer`'s constructor `addLayer` calls, in order.
        for model in [SkinModel::Classic, SkinModel::Slim] {
            let layers = renderer_for_player(model).layers;
            for want in [
                LayerKind::Armor,
                LayerKind::HeldItem,
                LayerKind::Arrow,
                LayerKind::CustomHead,
                LayerKind::Deadmau5Head,
                LayerKind::Cape,
            ] {
                assert!(layers.contains(&want), "{model:?} is missing {want:?}");
            }
            // Layers that belong to specific mobs must not leak onto a player.
            for absent in [
                LayerKind::CreeperCharge,
                LayerKind::SheepWool,
                LayerKind::WolfCollar,
                LayerKind::HeldBlock,
            ] {
                assert!(
                    !layers.contains(&absent),
                    "{model:?} must not carry {absent:?}"
                );
            }
        }
    }

    #[test]
    fn renderer_for_player_is_cached_like_every_other_renderer() {
        // Two calls for the same arm width must be the exact same static, the same way
        // `renderer_for`'s table entries are built once and shared.
        let a = renderer_for_player(SkinModel::Classic);
        let b = renderer_for_player(SkinModel::Classic);
        assert!(std::ptr::eq(a, b));
        let slim_a = renderer_for_player(SkinModel::Slim);
        let slim_b = renderer_for_player(SkinModel::Slim);
        assert!(std::ptr::eq(slim_a, slim_b));
        assert!(
            !std::ptr::eq(a, slim_a),
            "classic and slim are distinct statics"
        );
    }

    #[test]
    fn renderer_for_player_actually_builds_and_poses_a_sane_model() {
        // Not just a descriptor: the ModelKind it names must really build and emit geometry,
        // same as every mob renderer's model does.
        for (model, kind) in [
            (SkinModel::Classic, ModelKind::Player),
            (SkinModel::Slim, ModelKind::PlayerSlim),
        ] {
            let desc = renderer_for_player(model);
            assert_eq!(desc.model, kind);
            let mut built = desc.model.build_default();
            let mut out = Vec::new();
            built.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
            assert!(!out.is_empty(), "{model:?} model emitted no geometry");
            assert!(
                out.iter().all(|v| v.pos.iter().all(|c| c.is_finite())),
                "{model:?} model emitted a non-finite vertex"
            );
        }
    }

    #[test]
    fn renderer_for_kind_reaches_a_player_body_by_arm_width() {
        let classic = renderer_for_kind(EntityKind::Player, SkinModel::Classic).unwrap();
        assert_eq!(classic.model, ModelKind::Player);
        let slim = renderer_for_kind(EntityKind::Player, SkinModel::Slim).unwrap();
        assert_eq!(slim.model, ModelKind::PlayerSlim);
    }

    #[test]
    fn renderer_for_kind_still_resolves_ordinary_mobs_and_ignores_the_player_model_hint() {
        // The `player_model` argument must be irrelevant to a `Type` entity — a zombie is a
        // zombie whether it is passed `Classic` or `Slim`.
        let via_kind = renderer_for_kind(EntityKind::Type(E::Zombie), SkinModel::Slim).unwrap();
        let direct = renderer_for(E::Zombie).unwrap();
        assert_eq!(via_kind, direct);
        let via_kind2 = renderer_for_kind(EntityKind::Type(E::Zombie), SkinModel::Classic).unwrap();
        assert_eq!(via_kind, via_kind2);
    }

    #[test]
    fn renderer_for_kind_reaches_both_boss_model_descriptors() {
        let dragon = renderer_for_kind(EntityKind::Type(E::EnderDragon), SkinModel::Classic)
            .expect("ModelDragon has a normal living renderer descriptor");
        assert_eq!(dragon.model, ModelKind::Dragon);
        assert_eq!(dragon.texture, "textures/entity/enderdragon/dragon.png");
        assert_eq!(dragon.shadow_size, 0.5);
        assert_eq!(dragon.scale, 1.0);
        assert_eq!(
            dragon.layers,
            &[LayerKind::DragonEyes, LayerKind::DragonDeath]
        );
        let wither = renderer_for_kind(EntityKind::Type(E::WitherBoss), SkinModel::Classic)
            .expect("ModelWither is a normal living renderer now");
        assert_eq!(wither.model, ModelKind::Wither);
        assert_eq!(wither.scale, 2.0);
        assert_eq!(wither.layers, &[LayerKind::WitherAura]);
    }
}
