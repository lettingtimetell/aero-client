//! The layer system: armour, held items, capes, glowing eyes, saddles, wool, collars.
//!
//! java: `client/renderer/entity/layers/**` (24 files) and
//! `RendererLivingEntity.renderLayers`.
//!
//! # What a layer is
//!
//! A layer is a second pass over the same posed model, run after the body with its own texture
//! and its own GL state. `RendererLivingEntity` keeps an ordered `layerRenderers` list and
//! replays each one with the same animation arguments the main model got, so a layer never
//! re-poses anything — it inherits the pose and only changes what is drawn and how.
//!
//! That inheritance is the whole design, and it is why [`LayerKind`] here carries only *what
//! differs*: the model to draw ([`LayerKind::model`]), the texture, the blend/depth state
//! ([`LayerBlend`]), and any extra transform. The pose comes from the caller.
//!
//! # Armour is four passes, not one
//!
//! `LayerArmorBase.doRenderLayer` calls `renderLayer` four times, **4, 3, 2, 1** — helmet,
//! chestplate, leggings, boots. Each pass hides every part of the armour model except the ones
//! that slot covers ([`ArmorSlot::visible_parts`]), and the texture is `_layer_2` for leggings
//! and `_layer_1` for everything else ([`ArmorSlot::texture_layer`]) — that is why leggings are
//! a separate, thinner model in the first place. Leather draws its dyed base sheet first, then
//! adds a second, untinted pass over the `_overlay` texture ([`armor_texture`]'s `overlay`
//! argument).

use glam::{Mat4, Vec3};

use super::anim::AnimParams;
pub use super::cape_cloth::{CapeCloth, CapeClothStyle};
use super::model::{Model, PartId};
use super::models::biped::{BipedModel, PlayerModel};
use super::models::object::{ArmorStandArmorModel, ArmorStandParts};

/// Per-entity colour supplied to a layer at submission time.
///
/// Most layer colours are constants owned by [`LayerKind`]. The wool and collar layers are the
/// exception: Java resolves them from the entity immediately before its model is replayed. This
/// input keeps that entity-side fact out of the shared model cache, so one `RenderManager` can
/// draw several differently coloured sheep in one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayerTintInput {
    /// `EntitySheep.getDyeRgb(...)`, when this particular entity has a dye-driven layer.
    pub dye: Option<[f32; 4]>,
}
use super::models::ModelKind;
use super::EntityVertex;

/// Which of the four armour slots a pass is drawing.
///
/// java: `LayerArmorBase`'s `armorSlot`, which is `getCurrentArmor(slot - 1)` — so 1 is the
/// *boots* and 4 is the helmet, the opposite of the inventory's own ordering. Getting this
/// backwards silently swaps a player's helmet and boots textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArmorSlot {
    /// java: `armorSlot == 1`
    Boots,
    /// java: `armorSlot == 2` — the only slot that uses `_layer_2`.
    Leggings,
    /// java: `armorSlot == 3`
    Chestplate,
    /// java: `armorSlot == 4`
    Helmet,
}

/// A biped part an armour pass can show. java: the `ModelBiped` fields `setModelPartVisible`
/// flips on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArmorPart {
    /// java: `bipedHead`
    Head,
    /// java: `bipedHeadwear`
    Headwear,
    /// java: `bipedBody`
    Body,
    /// java: `bipedRightArm`
    RightArm,
    /// java: `bipedLeftArm`
    LeftArm,
    /// java: `bipedRightLeg`
    RightLeg,
    /// java: `bipedLeftLeg`
    LeftLeg,
}

impl ArmorSlot {
    /// The four passes in the order vanilla draws them: helmet first, boots last.
    ///
    /// java: `LayerArmorBase.doRenderLayer` — `renderLayer(..., 4)`, `3`, `2`, `1`.
    pub const DRAW_ORDER: [ArmorSlot; 4] =
        [Self::Helmet, Self::Chestplate, Self::Leggings, Self::Boots];

    /// Vanilla's `armorSlot` integer.
    #[must_use]
    pub fn slot_index(self) -> u8 {
        match self {
            Self::Boots => 1,
            Self::Leggings => 2,
            Self::Chestplate => 3,
            Self::Helmet => 4,
        }
    }

    /// `1` or `2` — which armour texture sheet this slot reads.
    ///
    /// java: `LayerArmorBase.isSlotForLeggings` — `armorSlot == 2`.
    #[must_use]
    pub fn texture_layer(self) -> u8 {
        if self == Self::Leggings {
            2
        } else {
            1
        }
    }

    /// Which biped parts this pass shows. Everything else is hidden first.
    ///
    /// java: `LayerBipedArmor.setModelPartVisible` — note the chestplate shows the *arms* and
    /// the body, while the leggings show the *legs* and the body, so the two overlap at the
    /// waist on purpose.
    #[must_use]
    pub fn visible_parts(self) -> &'static [ArmorPart] {
        match self {
            Self::Boots => &[ArmorPart::RightLeg, ArmorPart::LeftLeg],
            Self::Leggings => &[ArmorPart::Body, ArmorPart::RightLeg, ArmorPart::LeftLeg],
            Self::Chestplate => &[ArmorPart::Body, ArmorPart::RightArm, ArmorPart::LeftArm],
            Self::Helmet => &[ArmorPart::Head, ArmorPart::Headwear],
        }
    }

    /// Which armour model this pass uses.
    ///
    /// java: `LayerArmorBase.getArmorModel` — leggings get `modelLeggings`, everything else
    /// `modelArmor`; for a biped those are `ModelBiped(0.5F)` and `ModelBiped(1.0F)`.
    #[must_use]
    pub fn model_inflation(self) -> f32 {
        if self == Self::Leggings {
            0.5
        } else {
            1.0
        }
    }
}

/// The name an armour material contributes to its texture path.
///
/// java: `ItemArmor.ArmorMaterial`'s first constructor argument (`ItemArmor.java:237-241`) —
/// `getName()`, which `LayerArmorBase.getArmorResource` interpolates. It is **not** the enum
/// name: `CHAIN` is spelled `chainmail` on disk, so mapping through
/// [`mc_data::ArmorMaterialInfo::name`] (which carries `CHAIN`) would 404 every chainmail
/// texture.
#[must_use]
pub fn armor_material_name(material: mc_data::ArmorMaterial) -> &'static str {
    match material {
        mc_data::ArmorMaterial::Leather => "leather",
        mc_data::ArmorMaterial::Chain => "chainmail",
        mc_data::ArmorMaterial::Iron => "iron",
        mc_data::ArmorMaterial::Gold => "gold",
        mc_data::ArmorMaterial::Diamond => "diamond",
    }
}

/// Builds an armour texture path.
///
/// java: `LayerArmorBase.getArmorResource` —
/// `String.format("textures/models/armor/%s_layer_%d%s.png", material, slot == 2 ? 2 : 1,
/// overlay == null ? "" : "_" + overlay)`.
#[must_use]
pub fn armor_texture(material: &str, slot: ArmorSlot, overlay: Option<&str>) -> String {
    match overlay {
        Some(o) => format!(
            "textures/models/armor/{material}_layer_{}_{o}.png",
            slot.texture_layer()
        ),
        None => format!(
            "textures/models/armor/{material}_layer_{}.png",
            slot.texture_layer()
        ),
    }
}

/// [`armor_texture`] for a vanilla material without the per-call `format!` (ENTM-6: the armour
/// layer built one or two `String`s per worn slot per frame). Same paths, as static keys.
#[must_use]
pub fn armor_texture_key(
    material: mc_data::ArmorMaterial,
    slot: ArmorSlot,
    overlay: bool,
) -> &'static str {
    macro_rules! sheets {
        ($name:literal) => {
            [
                concat!("textures/models/armor/", $name, "_layer_1.png"),
                concat!("textures/models/armor/", $name, "_layer_2.png"),
                concat!("textures/models/armor/", $name, "_layer_1_overlay.png"),
                concat!("textures/models/armor/", $name, "_layer_2_overlay.png"),
            ]
        };
    }
    let sheets = match material {
        mc_data::ArmorMaterial::Leather => sheets!("leather"),
        mc_data::ArmorMaterial::Chain => sheets!("chainmail"),
        mc_data::ArmorMaterial::Iron => sheets!("iron"),
        mc_data::ArmorMaterial::Gold => sheets!("gold"),
        mc_data::ArmorMaterial::Diamond => sheets!("diamond"),
    };
    sheets[usize::from(slot.texture_layer() == 2) + 2 * usize::from(overlay)]
}

/// How a layer's pass is blended and depth-tested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerBlend {
    /// Normal opaque draw with the alpha cutout — armour, saddles, wool, collars.
    Opaque,
    /// Unlit opaque draw with the ordinary world alpha cutout and depth writes.  This is
    /// `Render.renderEntityOnFire`: it calls `disableLighting()` but leaves blending disabled
    /// and the world pass's `alphaFunc(GL_GREATER, 0.1)` / depth mask intact.
    OpaqueUnlit,
    /// [`Self::OpaqueUnlit`] with back faces culled: the world pass's own `GL_CULL_FACE`
    /// (`EntityRenderer:1328`) for a renderer that never disables it, as
    /// `RendererLivingEntity.doRender` (`:92`) does.
    ///
    /// `RenderArrow` is the user (see [`super::objects::ARROW_BLEND`]). Each of its six quads is
    /// one of a coplanar pair wound opposite ways, and with culling only the camera-facing one of
    /// each pair draws; two-sided, the pairs z-fight. Its fixed-function lighting is folded into
    /// the vertex colour on the CPU ([`super::objects::light_arrow`]) because its normal is
    /// constant per quad and deliberately not unit length, so the pipeline itself is unlit.
    OpaqueUnlitCulled,
    /// Additive and unlit with the ordinary visible-layer depth write — glowing eyes and the
    /// charged-creeper aura. java: `blendFunc(1, 1)` with `disableLighting()`.
    /// Invisible eye layers need a separate no-depth-write submission state.
    Additive,
    /// Additive and unlit with depth testing retained but depth writes disabled.
    ///
    /// java: `RendererLivingEntity.renderModel` calls `depthMask(!entity.isInvisible())` before
    /// the layer pass. Spider/enderman eyes still render for an invisible entity, but they must
    /// not leave the invisible body's silhouette in the depth buffer.
    AdditiveNoDepthWrite,
    /// Straight alpha with depth writes off — contact shadows and translucent held blocks.
    ///
    /// java: the entity pass's `alphaFunc(GL_GREATER, 0.1)` tests the *combined* fragment alpha
    /// (texel times primary colour), so a shadow cell whose vertex alpha has faded to 0.1 is
    /// discarded outright rather than drawn as a faint disc.
    Translucent,
    /// [`Self::Translucent`]'s state for the invisible-teammate ghost body.
    ///
    /// java: `RendererLivingEntity.renderModel` draws it at `color(1, 1, 1, 0.15)` with
    /// `depthMask(false)` and a temporarily lowered `alphaFunc(GL_GREATER, 0.003921569)`, which
    /// is still tested against `texel * 0.15`.
    Ghost,
    /// Straight alpha with depth writes still enabled — `LayerSlimeGel`'s outer shell, and
    /// `MobAppearance`'s custom layer-3 renderer after it calls `depthMask(true)`.
    TranslucentDepthWrite,
    /// Unlit map-style transparency with depth writes. java: `MapItemRenderer.renderMap`'s
    /// `disableLighting`, `blendFuncSeparate(ONE, ONE_MINUS_SRC_ALPHA, ZERO, ONE)` and unchanged
    /// world depth mask. This is deliberately separate from ordinary alpha blending: the map's
    /// transparent unexplored checkerboard is not source-alpha-premultiplied in vanilla.
    MapTranslucentDepthWrite,
    /// The enchanted-item foil replay.  Vanilla uses `blendFunc(768, 1)`
    /// (`GL_SRC_COLOR, GL_ONE`), disables lighting, writes no depth, and accepts only fragments
    /// at the model's existing depth (`GL_EQUAL`).
    Glint,
    /// A dropped item's baked model. java: `RenderEntityItem.doRender` (`:103-106`) enables
    /// blending with `tryBlendFuncSeparate(770, 771, 1, 0)` and keeps the world pass's depth mask
    /// and alpha test; unlike `RendererLivingEntity.doRender` (`:92`) it never disables
    /// `GL_CULL_FACE`, which the world pass enabled (`EntityRenderer:1328`), so back faces are
    /// culled. It calls `RenderItem.renderItem(stack, model)` directly, which has no
    /// `isThereOneNegativeScale` face switch, so the culled face is always `GL_BACK`.
    DroppedItem,
    /// [`Self::Glint`] for a dropped item: the same foil state with back faces culled, so of a
    /// coplanar pair of opposite-facing `ItemModelGenerator` side quads only the one facing the
    /// camera receives the `GL_EQUAL` foil.
    DroppedItemGlint,
    /// `RendererLivingEntity.setBrightness`'s hurt/death texture combiner for an ordinary
    /// depth-writing model pass. Unlike [`Self::WhiteFlash`], this samples the entity texture,
    /// preserves its cutout, and applies the red affine mix before the framebuffer write.
    LivingHurt,
    /// The same hurt/death combiner while retaining a blended source pass's alpha blend and
    /// enabled depth writes (`LayerSlimeGel`, [`Self::TranslucentDepthWrite`]).
    LivingHurtTranslucent,
    /// The invisible-teammate body variant. It retains the translucent state above while using
    /// Java's temporarily lowered `1/255` alpha-test threshold.
    LivingHurtGhost,
    /// An enchanted custom-head glint while its combine-true layer keeps living brightness
    /// active: glint texture and blend/depth state, with the affine red combiner retained.
    LivingHurtGlint,
    /// `RendererLivingEntity.setBrightness`'s colour-multiplier replay. The entity texture is
    /// sampled to retain its cutout/source alpha, while the multiplier supplies lightmapped
    /// white and ordinary alpha-over blending at the body depth.
    WhiteFlash,
    /// Primed TNT's intermittent white replay. Vanilla disables texture and lighting, enables
    /// `SRC_ALPHA, ONE_MINUS_SRC_COLOR`, keeps depth writes/LEQUAL, and applies polygon offset
    /// `-3, -3` before drawing the same block geometry a second time.
    TntFlash,
    /// Ender Dragon's exploding-texture body replay. The alpha-test threshold is the interpolated
    /// 200-tick death phase, while the body remains lit and depth-writing like the normal model.
    DragonDeathBody,
    /// The untextured magenta/white shard fans emitted by `LayerEnderDragonDeath`: additive
    /// (`SRC_ALPHA, ONE`), no depth writes, back faces culled, lightmapped at the dragon's light.
    DragonDeathShards,
    /// The ender-crystal recharge beam: opaque (blending off), the 0.1 alpha test, unlit, the
    /// lightmap stage at the dragon's own light, depth writes, two-sided.
    DragonBeam,
    /// The untextured fishing line, rendered as a Java `GL_LINE_STRIP` with depth writes on.
    FishingLine,
    /// The two untextured leash strips, rendered as `GL_TRIANGLE_STRIP` with depth writes on.
    Leash,
    /// Guardian's textured laser: opaque (blending off), the 0.1 alpha test, unlit, the
    /// full-bright lightmap stage, depth writes, two-sided.
    GuardianBeam,
}

impl LayerBlend {
    /// Selects the in-draw hurt/death combiner while preserving the source pass's framebuffer
    /// and depth-write policy. Only living-model source states can participate.
    #[must_use]
    pub const fn living_hurt(self) -> Option<Self> {
        match self {
            Self::Opaque => Some(Self::LivingHurt),
            // `LayerSlimeGel` is the only combining layer with a blended source pass, and it
            // keeps depth writes on; nothing that combines draws with `Translucent`.
            Self::TranslucentDepthWrite => Some(Self::LivingHurtTranslucent),
            _ => None,
        }
    }
}

/// The material state for `Render.renderEntityOnFire`.
///
/// Fire disables lighting but leaves blending disabled, alpha cutout active, and depth writes on.
/// Keeping the selection beside [`LayerBlend`] gives the client caller and renderer tests one
/// biting contract instead of letting the live upload silently drift back to additive blending.
pub const FIRE_OVERLAY_BLEND: LayerBlend = LayerBlend::OpaqueUnlit;

/// One layer in a renderer's list.
///
/// java: one `LayerRenderer` implementation. Only the layers whose behaviour is fully
/// determined by entity state are here; the two that need an item model
/// ([`LayerKind::HeldItem`], [`LayerKind::Armor`]) carry the *policy* and leave the geometry to
/// the caller, since baked item models live in `mc-assets` and armour materials in `mc-data`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LayerKind {
    /// java: `LayerBipedArmor` / `LayerVillagerArmor` — four passes, see [`ArmorSlot`].
    Armor,
    /// java: `LayerHeldItem`
    HeldItem,
    /// java: `LayerHeldBlock` — the enderman's carried block.
    HeldBlock,
    /// java: `LayerCape`
    Cape,
    /// java: `LayerCreeperCharge`
    CreeperCharge,
    /// java: `LayerEndermanEyes`
    EndermanEyes,
    /// java: `LayerSpiderEyes`
    SpiderEyes,
    /// java: `LayerSaddle` — the pig's saddle.
    Saddle,
    /// java: `LayerSheepWool`
    SheepWool,
    /// java: `LayerWolfCollar`
    WolfCollar,
    /// java: `LayerMooshroomMushroom`
    MooshroomMushroom,
    /// java: `LayerIronGolemFlower`
    IronGolemFlower,
    /// java: `LayerSnowmanHead`
    SnowmanHead,
    /// java: `LayerSlimeGel`
    SlimeGel,
    /// java: `LayerWitherAura`
    WitherAura,
    /// java: `LayerArrow` — arrows stuck in a mob.
    Arrow,
    /// java: `LayerCustomHead` — a skull or block worn in the helmet slot.
    CustomHead,
    /// java: `LayerDeadmau5Head`
    Deadmau5Head,
    /// java: `LayerHeldItemWitch`
    WitchHeldItem,
    /// java: `LayerEnderDragonEyes`
    DragonEyes,
    /// java: `LayerEnderDragonDeath`
    DragonDeath,
}

impl LayerKind {
    /// Whether `RendererLivingEntity.renderLayers` leaves the entity brightness combiner active
    /// while this layer draws. java: `LayerRenderer.shouldCombineTextures()`.
    ///
    /// This is observable during hurt/death: combining layers receive the same affine red flash
    /// as the body, while additive eyes, armour, held items and other `false` layers do not.
    #[must_use]
    pub const fn combines_brightness(self) -> bool {
        matches!(
            self,
            Self::SheepWool
                | Self::WolfCollar
                | Self::MooshroomMushroom
                | Self::SnowmanHead
                | Self::SlimeGel
                | Self::CustomHead
                | Self::Deadmau5Head
        )
    }

    /// The model this layer draws, if it draws one of ours.
    ///
    /// `None` means the geometry comes from somewhere else — a baked item model
    /// ([`LayerKind::HeldItem`], [`LayerKind::WitchHeldItem`], [`LayerKind::IronGolemFlower`]),
    /// a block model ([`LayerKind::HeldBlock`]), or the caller's armour model
    /// ([`LayerKind::Armor`], which needs the material to pick a texture).
    #[must_use]
    pub fn model(self) -> Option<ModelKind> {
        match self {
            Self::CreeperCharge => Some(ModelKind::Creeper),
            // java: `LayerSaddle` owns `new ModelPig(0.5F)`, rather than replaying
            // `RenderPig`'s uninflated main model.
            Self::Saddle => Some(ModelKind::Pig),
            Self::SheepWool => Some(ModelKind::SheepWool),
            // java: `LayerSlimeGel`'s own field is `new ModelSlime(0)` — the plain
            // eight-unit shell — while `RenderSlime`'s main model is
            // `new ModelSlime(16)`, the smaller body that carries the eyes. This arm
            // resolved to `ModelKind::Slime`, i.e. the body's own geometry, so the gel was a
            // pixel-exact duplicate of the body and the outer shell was never drawn. Both
            // models were already built and tested; only this arm was wrong.
            Self::SlimeGel => Some(ModelKind::SlimeGel),
            // java: `LayerWitherAura` owns `new ModelWither(0.5F)`, rather than replaying the
            // main model's uninflated geometry.
            Self::WitherAura => Some(ModelKind::Wither),
            Self::SnowmanHead | Self::CustomHead => Some(ModelKind::HumanoidHead),
            Self::Deadmau5Head => Some(ModelKind::Player),
            _ => None,
        }
    }

    /// Whether this layer redraws the entity's *own* posed body under its fixed texture — the
    /// `body` argument [`RenderManager::emit_layer`](super::manager::RenderManager::emit_layer)
    /// copies in its `None` arm (java: `LayerEndermanEyes`, `LayerSpiderEyes`,
    /// `LayerEnderDragonEyes`, `LayerWolfCollar` call `getMainModel().render(...)` again).
    ///
    /// Every other layer either owns a model, draws item/block geometry, or draws nothing from
    /// the body, so a caller only needs to keep a copy of the posed body when some layer in the
    /// renderer's list answers `true` (R2-1/ENTM-6: players, villagers and armour stands never do).
    #[must_use]
    pub fn replays_body(self) -> bool {
        self.model().is_none() && self.texture().is_some()
    }

    /// The fixed texture this layer binds, if it has one.
    #[must_use]
    pub fn texture(self) -> Option<&'static str> {
        match self {
            Self::CreeperCharge => Some("textures/entity/creeper/creeper_armor.png"),
            Self::EndermanEyes => Some("textures/entity/enderman/enderman_eyes.png"),
            Self::SpiderEyes => Some("textures/entity/spider_eyes.png"),
            Self::DragonEyes => Some("textures/entity/enderdragon/dragon_eyes.png"),
            Self::Saddle => Some("textures/entity/pig/pig_saddle.png"),
            Self::SheepWool => Some("textures/entity/sheep/sheep_fur.png"),
            Self::WolfCollar => Some("textures/entity/wolf/wolf_collar.png"),
            // `MooshroomMushroom` deliberately has **no** entity texture: java
            // `LayerMooshroomMushroom.doRenderLayer` binds `TextureMap.locationBlocksTexture`
            // and calls `renderBlockBrightness(Blocks.red_mushroom, 1.0F)` three times — two
            // mushrooms on the back and one on the head. It is a block-model layer like
            // `LayerHeldBlock`, not a re-skin of the body, so returning the mooshroom's *own*
            // texture here (as this used to) would make a caller draw the cow a second time on
            // top of itself.
            Self::WitherAura => Some("textures/entity/wither/wither_armor.png"),
            _ => None,
        }
    }

    /// How this layer is blended.
    ///
    /// java: the eyes layers and `LayerCreeperCharge` use `blendFunc(1, 1)` with lighting off,
    /// which is why they glow in the dark; `LayerWitherAura` uses straight alpha.
    #[must_use]
    pub fn blend(self) -> LayerBlend {
        match self {
            // java: all four call `GlStateManager.blendFunc(1, 1)` — GL_ONE/GL_ONE.
            // `LayerWitherAura:45` is one of them: it was classified `Translucent` here (and
            // described as "straight alpha" in the audit), but its `blendFunc` is byte-for-byte
            // `LayerCreeperCharge`'s, right down to the `color(0.5, 0.5, 0.5, 1.0)` grey and the
            // `disableLighting()` on either side of it. The wither's shield glows; it does not
            // fade.
            Self::CreeperCharge
            | Self::EndermanEyes
            | Self::SpiderEyes
            | Self::WitherAura
            | Self::DragonEyes => LayerBlend::Additive,
            // java: `LayerSlimeGel:26` — `blendFunc(770, 771)`, GL_SRC_ALPHA /
            // GL_ONE_MINUS_SRC_ALPHA, and **no** `depthMask(false)`:
            // `RendererLivingEntity.doRender` (`:174`) sets `depthMask(true)` right before
            // `renderLayers`, so the gel writes depth.
            // Vanilla draws entities before translucent terrain (`EntityRenderer:1402` vs
            // `:1467`), so water behind a slime fails the depth test wherever the gel is, and
            // the gel's back face (drawn last) is hidden behind its front face. With depth writes
            // off, water drew over the gel and the back face blended on top of the front.
            Self::SlimeGel => LayerBlend::TranslucentDepthWrite,
            Self::DragonDeath => LayerBlend::DragonDeathShards,
            // java: `LayerArrow.doRenderLayer` brackets its `RenderArrow` calls with
            // `RenderHelper.disableStandardItemLighting()`/`enableStandardItemLighting()`, inside
            // `RendererLivingEntity.doRender`'s `disableCull()`: stuck arrows are unlit and
            // two-sided, unlike a free arrow (`objects::ARROW_BLEND`). Measured on the real GL by
            // `ArrowRenderRef` (`GL_LIGHTING` and `GL_CULL_FACE` both off at every arrow draw).
            Self::Arrow => LayerBlend::OpaqueUnlit,
            _ => LayerBlend::Opaque,
        }
    }

    /// Select the entity-state-dependent blend used by the live layer submission.
    ///
    /// java: every additive layer turns depth writes off for an invisible entity — the eyes
    /// layers, `LayerCreeperCharge.doRenderLayer` (`depthMask(!flag)`, `flag = isInvisible()`)
    /// and `LayerWitherAura.doRenderLayer` (`depthMask(!isInvisible())`). This mattered once
    /// invisible bodies replay their layers (19-0 U2; phase 5a review S1).
    #[must_use]
    pub fn blend_for(self, invisible: bool) -> LayerBlend {
        if invisible
            && matches!(
                self,
                Self::EndermanEyes
                    | Self::SpiderEyes
                    | Self::DragonEyes
                    | Self::CreeperCharge
                    | Self::WitherAura
            )
        {
            LayerBlend::AdditiveNoDepthWrite
        } else {
            self.blend()
        }
    }

    /// The constant colour vanilla sets before drawing this layer, if it sets one.
    ///
    /// java: the `GlStateManager.color` immediately before the layer's `render` call.
    /// `LayerCreeperCharge:36` and `LayerWitherAura:41` both use `float f = 0.5F; color(f, f, f,
    /// 1.0F)` — half-brightness, which is what stops an additive aura blowing out to white. The
    /// eyes layers and every opaque layer set `color(1, 1, 1, 1)`, the identity.
    ///
    /// Layers whose colour depends on entity state — the sheep's fleece and the wolf's collar,
    /// both `EntitySheep.getDyeRgb(...)` — return [`None`] here and are the caller's, because the
    /// dye index is entity metadata this module does not read.
    #[must_use]
    pub fn tint(self) -> Option<[f32; 4]> {
        match self {
            Self::CreeperCharge | Self::WitherAura => Some([0.5, 0.5, 0.5, 1.0]),
            Self::SheepWool | Self::WolfCollar => None,
            _ => Some([1.0, 1.0, 1.0, 1.0]),
        }
    }

    /// The primary colour for this concrete layer submission.
    ///
    /// java: `GlStateManager.color` in each `LayerRenderer`. Dynamic colours apply only to the
    /// two layers that read `EntitySheep.getDyeRgb`; passing a sheep's colour to another layer
    /// must not accidentally dye armour, eyes, or a charged-creeper aura.
    #[must_use]
    pub fn resolved_tint(self, input: LayerTintInput) -> [f32; 4] {
        match self {
            Self::SheepWool | Self::WolfCollar => input.dye.unwrap_or([1.0, 1.0, 1.0, 1.0]),
            _ => self.tint().unwrap_or([1.0, 1.0, 1.0, 1.0]),
        }
    }

    /// Whether this layer's texture scrolls with time.
    ///
    /// java: `LayerCreeperCharge` translates the *texture* matrix by
    /// `(ticksExisted + partialTicks) * 0.01` on both axes — the aura crawls across the model.
    #[must_use]
    pub fn texture_scroll(self, age_in_ticks: f32) -> Option<[f32; 2]> {
        match self {
            Self::CreeperCharge => Some([age_in_ticks * 0.01, age_in_ticks * 0.01]),
            // java: `LayerWitherAura:36-38` — `f1 = MathHelper.cos(f * 0.02F) * 3.0F` on X and
            // `f2 = f * 0.01F` on Y, i.e. the shield *sways* horizontally over a +/-3 range while
            // it creeps vertically. This used to be the creeper's symmetric `0.01, 0.01`.
            Self::WitherAura => {
                Some([mc_math::cos(age_in_ticks * 0.02) * 3.0, age_in_ticks * 0.01])
            }
            _ => None,
        }
    }

    /// Extra uniform inflation applied to the layer's model.
    ///
    /// java: `LayerCreeperCharge`'s `new ModelCreeper(2.0F)` and `LayerSlimeGel`'s
    /// `new ModelSlime(0)` — the aura is a fatter copy of the same box tree.
    #[must_use]
    pub fn model_inflation(self) -> f32 {
        match self {
            Self::CreeperCharge => 2.0,
            // java: `LayerSaddle`'s separate `ModelPig(0.5F)` shell sits 0.5 model
            // units (1/32 block) proud of the pig skin on every face.
            Self::Saddle => 0.5,
            Self::WitherAura => 0.5,
            _ => 0.0,
        }
    }
}

/// The per-frame inputs `LayerCape` needs, all already interpolated.
///
/// java: `LayerCape.doRenderLayer`'s `d0`/`d1`/`d2` — the difference between the player's
/// interpolated position and its lagging "chasing" position, which is what makes a cape trail
/// behind a sprinting player and settle when they stop.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CapeState {
    /// Optional connected cloth surface. `None` keeps the exact vanilla cape geometry.
    pub cloth: Option<CapeCloth>,
    /// Chasing minus real position, X. java: `d0`
    pub chase_dx: f64,
    /// Chasing minus real position, Y. java: `d1`
    pub chase_dy: f64,
    /// Chasing minus real position, Z. java: `d2`
    pub chase_dz: f64,
    /// Interpolated body yaw in degrees. java: `f`
    pub body_yaw: f32,
    /// Interpolated `cameraYaw` — the walk bob amount. java: `f4`
    pub camera_yaw: f32,
    /// Interpolated `distanceWalkedModified`.
    pub distance_walked: f32,
    /// Sneaking adds a flat 25 degrees. java: `if (isSneaking()) f1 += 25`
    pub sneaking: bool,
}

/// The three cape rotations, in degrees: `(x, z, y)`.
///
/// java: `LayerCape.doRenderLayer` — `rotate(6 + f2/2 + f1, X)`, `rotate(f3/2, Z)`,
/// `rotate(-f3/2, Y)`, then a flat `rotate(180, Y)`.
#[must_use]
pub fn cape_angles(s: &CapeState) -> (f32, f32, f32) {
    let d3 = f64::from(mc_math::sin(s.body_yaw * core::f32::consts::PI / 180.0));
    let d4 = f64::from(-mc_math::cos(s.body_yaw * core::f32::consts::PI / 180.0));
    let mut f1 = s.chase_dy as f32 * 10.0;
    f1 = mc_math::clamp_f32(f1, -6.0, 32.0);
    let mut f2 = (s.chase_dx * d3 + s.chase_dz * d4) as f32 * 100.0;
    let f3 = (s.chase_dx * d4 - s.chase_dz * d3) as f32 * 100.0;
    if f2 < 0.0 {
        f2 = 0.0;
    }
    f1 += mc_math::sin(s.distance_walked * 6.0) * 32.0 * s.camera_yaw;
    if s.sneaking {
        f1 += 25.0;
    }
    (6.0 + f2 / 2.0 + f1, f3 / 2.0, -f3 / 2.0)
}

/// The matrix `LayerCape` puts the cape part under.
///
/// java: `translate(0, 0, 0.125)`, then the three rotations, then `rotate(180, Y)`.
#[must_use]
pub fn cape_matrix(s: &CapeState) -> Mat4 {
    let (x, z, y) = cape_angles(s);
    Mat4::from_translation(Vec3::new(0.0, 0.0, 0.125))
        * Mat4::from_rotation_x(x.to_radians())
        * Mat4::from_rotation_z(z.to_radians())
        * Mat4::from_rotation_y(y.to_radians())
        * Mat4::from_rotation_y(core::f32::consts::PI)
}

/// Which of the two concrete item branches [`LayerCustomHead`] can draw.
///
/// java: `LayerCustomHead.doRenderLayer` checks `item instanceof ItemBlock` first and
/// `item == Items.skull` second.  Other items still enter the layer, but have no geometry branch;
/// callers should simply omit them rather than manufacturing a head transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CustomHeadItem {
    /// `ItemBlock`, rendered with `TransformType.HEAD`.
    Block,
    /// `Items.skull`, rendered by `TileEntitySkullRenderer`.
    Skull,
}

/// The matrix used by `LayerCustomHead` after the body has been posed.
///
/// java: `LayerCustomHead.doRenderLayer` (`:39-90`).  The caller supplies the entity-space
/// `base` matrix and the selected head part.  This helper deliberately keeps the pre-head
/// adjustments separate from the part transform: vanilla pushes the sneak/child operations,
/// then calls `ModelRenderer.postRender(0.0625F)`.  Pre-multiplying those operations onto an
/// already-composed world matrix would apply them in world space and move a rotated entity's
/// helmet sideways, which is the exact failure this seam prevents.
///
/// `villager_like` is vanilla's `EntityVillager || (EntityZombie && isVillager())` branch.  It
/// suppresses the generic child shrink/lift and adds the small item-specific offset after the
/// block/skull scale.  `head_scale` is normally `0.0625`, but remains an argument so tests and
/// specialized callers cannot accidentally smuggle a second hard-coded model scale into the
/// operation.
#[must_use]
// These inputs are the explicit Java LayerCustomHead call shape; a context struct would hide the
// order-sensitive transform gates this helper is intended to make reviewable.
#[allow(clippy::too_many_arguments)]
pub fn custom_head_matrix(
    base: Mat4,
    head: &Model,
    head_part: PartId,
    head_scale: f32,
    sneaking: bool,
    child: bool,
    villager_like: bool,
    item: CustomHeadItem,
) -> Mat4 {
    let mut m = base;
    if sneaking {
        // java: GlStateManager.translate(0.0F, 0.2F, 0.0F)
        m *= Mat4::from_translation(Vec3::new(0.0, 0.2, 0.0));
    }
    if child && !villager_like {
        // java: scale(1.4F / 2.0F), then translate(0.0F, 16.0F * scale, 0.0F).
        m *= Mat4::from_scale(Vec3::splat(1.4 / 2.0));
        m *= Mat4::from_translation(Vec3::new(0.0, 16.0 * head_scale, 0.0));
    }
    m *= head.get(head_part).post_render_matrix(head_scale);

    match item {
        CustomHeadItem::Block => {
            // java: f2 = 0.625F; translate, rotate, scale; villager offset is last.
            m *= Mat4::from_translation(Vec3::new(0.0, -0.25, 0.0));
            m *= Mat4::from_rotation_y(core::f32::consts::PI);
            m *= Mat4::from_scale(Vec3::new(0.625, -0.625, -0.625));
            if villager_like {
                m *= Mat4::from_translation(Vec3::new(0.0, 0.1875, 0.0));
            }
        }
        CustomHeadItem::Skull => {
            // java: f3 = 1.1875F; scale; villager offset is last.
            m *= Mat4::from_scale(Vec3::new(1.1875, -1.1875, -1.1875));
            if villager_like {
                m *= Mat4::from_translation(Vec3::new(0.0, 0.0625, 0.0));
            }
        }
    }
    m
}

// ---------------------------------------------------------------------------------------------
// Geometry emission
// ---------------------------------------------------------------------------------------------
//
// `LayerKind`s whose geometry is "some other posed model" (`LayerKind::model` is `Some`, or the
// body's own geometry is reused) are wired end to end by
// [`super::manager::RenderManager::emit_layer`]. Armour and the cape are not: vanilla draws
// armour with a *plain* `ModelBiped`/`ModelBiped(0.5F)` regardless of the mob wearing it
// (`LayerArmorBase.getArmorModel`), and the cape lives on the *same* `ModelPlayer` instance the
// body already used this frame (`RenderPlayer.getMainModel().renderCape`), so neither fits the
// "build a `ModelKind` and pose it fresh" shape `emit_layer` covers. Both are free functions
// here instead.

/// The seven [`ArmorPart`]s in the order [`BipedParts`](super::models::biped::BipedParts) keeps
/// them, paired with the field each name reads.
fn armor_part_ids(parts: &super::models::biped::BipedParts) -> [(ArmorPart, PartId); 7] {
    [
        (ArmorPart::Head, parts.head),
        (ArmorPart::Headwear, parts.headwear),
        (ArmorPart::Body, parts.body),
        (ArmorPart::RightArm, parts.right_arm),
        (ArmorPart::LeftArm, parts.left_arm),
        (ArmorPart::RightLeg, parts.right_leg),
        (ArmorPart::LeftLeg, parts.left_leg),
    ]
}

/// The standalone part handles used by `ModelArmorStandArmor`, in the same order as a biped.
fn armor_stand_part_ids(parts: &ArmorStandParts) -> [(ArmorPart, PartId); 7] {
    [
        (ArmorPart::Head, parts.head),
        (ArmorPart::Headwear, parts.headwear),
        (ArmorPart::Body, parts.body),
        (ArmorPart::RightArm, parts.right_arm),
        (ArmorPart::LeftArm, parts.left_arm),
        (ArmorPart::RightLeg, parts.right_leg),
        (ArmorPart::LeftLeg, parts.left_leg),
    ]
}

/// Emits one armour slot's visible geometry.
///
/// java: `LayerBipedArmor.doRenderLayer`'s per-slot call — pose `model` with the same
/// [`AnimParams`] the body used this frame, hide every part [`ArmorSlot::visible_parts`] does
/// not cover (java: `setModelPartVisible`), and draw.
///
/// `model` must already be built at [`ArmorSlot::model_inflation`] — `BipedModel::new(slot.
/// model_inflation())` for the helmet/chestplate/boots pass and the leggings pass respectively,
/// mirroring vanilla's two armour model instances (`modelArmor` and `modelLeggings`). Its
/// `is_child` flag is the caller's responsibility to copy from the body model being armoured —
/// java: `LayerArmorBase.doRenderLayer` sets `modelArmor.isChild = entitylivingbaseIn.isChild()`
/// itself, which this function cannot do since [`BipedModel`] has no entity handle.
pub fn emit_armor(
    model: &mut BipedModel,
    slot: ArmorSlot,
    p: &AnimParams,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    model.pose(p);
    emit_armor_posed(model, slot, p, base, out);
}

/// Emits one armour slot on an armor stand using its six stored rotations.
///
/// java: `ArmorStandRenderer.initArmor` installs `ModelArmorStandArmor` for both armour model
/// sizes, while `LayerBipedArmor` still owns the ordinary per-slot visibility mask.
pub fn emit_armor_stand(
    model: &mut ArmorStandArmorModel,
    slot: ArmorSlot,
    p: &AnimParams,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    pose_armor_stand(model, p);
    emit_armor_stand_posed(model, slot, p, base, out);
}

/// The pose half of [`emit_armor_stand`].
pub(crate) fn pose_armor_stand(model: &mut ArmorStandArmorModel, p: &AnimParams) {
    model.pose = p.entity.armor_stand.unwrap_or_default();
    model.pose_stand();
}

/// The per-slot half of [`emit_armor_stand`]: the slot's visibility mask and the draw.
pub(crate) fn emit_armor_stand_posed(
    model: &mut ArmorStandArmorModel,
    slot: ArmorSlot,
    p: &AnimParams,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    let visible = slot.visible_parts();
    let parts = model.parts;
    for (part, id) in armor_stand_part_ids(&parts) {
        model.model.get_mut(id).show_model = visible.contains(&part);
    }
    model.draw(p, base, out);
}

/// Emits one armour pass using `ModelZombie`'s arm-forward pose.
///
/// java: the anonymous `LayerBipedArmor` in `RenderZombie`/`RenderPigZombie`,
/// `RenderSkeleton`'s corresponding layer, and `LayerVillagerArmor`: every one holds a
/// `ModelZombie`-derived armour model, rather than a plain `ModelBiped`.
pub fn emit_zombie_armor(
    model: &mut BipedModel,
    slot: ArmorSlot,
    p: &AnimParams,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    model.pose_as_zombie(p);
    emit_armor_posed(model, slot, p, base, out);
}

/// Applies `LayerBipedArmor.setModelPartVisible` after the selected armour model is posed.
pub(crate) fn emit_armor_posed(
    model: &mut BipedModel,
    slot: ArmorSlot,
    p: &AnimParams,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    let visible = slot.visible_parts();
    let parts = model.parts;
    for (part, id) in armor_part_ids(&parts) {
        model.model.get_mut(id).show_model = visible.contains(&part);
    }
    model.draw(p, base, out);
}

/// Emits the player's cape.
///
/// java: `LayerCape.doRenderLayer` — `translate(0, 0, 0.125)`, the three chase/sneak rotations
/// from [`cape_matrix`], then `renderCape`, which is `this.bipedCape.render(0.0625F)` with no
/// re-pose. `model` must be the **same** [`PlayerModel`] instance the body was already posed and
/// drawn with this frame — `bipedCape`'s only per-frame input, the sneaking pivot lift, is set
/// by [`PlayerModel`]'s own [`super::anim::EntityModel::set_rotation_angles`], not by this
/// function. `base` is the same matrix the body was drawn under; `scale` is the model draw scale
/// (`0.0625` for every living entity).
///
/// The cape part is hidden by default ([`PlayerModel::new`](super::models::biped::PlayerModel::new))
/// because the body's own draw pass never includes it in its part list; this function is what
/// makes it visible, exactly as vanilla's separate `renderCape` call does.
pub fn emit_cape(
    model: &mut PlayerModel,
    state: &CapeState,
    scale: f32,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    let cape = model.wear.cape;
    model.biped.model.get_mut(cape).show_model = true;
    if let Some(cloth) = state.cloth {
        super::cape_cloth::emit(model.biped.model.get(cape), state, cloth, scale, base, out);
        return;
    }
    let m = base * cape_matrix(state);
    model.biped.model.emit_part(cape, scale, m, out);
}

/// The matrix a held item is drawn under, attached to a biped's hand.
///
/// java: `LayerHeldItem.doRenderLayer`. `ModelBiped.postRenderArm(scale)` is
/// `bipedRightArm.postRender(scale)` — every biped arm is a tree root, not nested under another
/// part, so that is exactly [`ModelPart::post_render_matrix`] applied on top of `base`, with no walk
/// needed. What follows is transcribed in the Java's own order: the child scale-and-tilt (pushed
/// *before* the arm's own matrix, so it wraps the whole rest of the stack), the fixed
/// `(-0.0625, 0.4375, 0.0625)` offset that centres an item in an open hand, the extra pose
/// vanilla applies only when the held item is a full block model (`Block.getRenderType() == 2`
/// — a real cube rather than a flat icon), and the sneaking lift.
///
/// `arm` is [`BipedParts::right_arm`](super::models::biped::BipedParts::right_arm) for
/// `heldItemRight` (the common case) or `left_arm` for `heldItemLeft` — vanilla's off-hand item
/// (`LayerHeldItem` special-cases neither arm; both slots share this one method).
///
/// The caller combines the returned matrix with the equipped stack's baked model — see
/// [`super::item::emit_baked_model`] — since the item geometry itself is `mc-assets`' (S09), not
/// this section's.
#[must_use]
pub fn held_item_matrix(
    model: &Model,
    arm: PartId,
    scale: f32,
    base: Mat4,
    child: bool,
    is_full_block: bool,
    sneaking: bool,
) -> Mat4 {
    let mut m = base;
    if child {
        // java: `translate(0, 0.625, 0)`, `rotate(-20, -1, 0, 0)`, `scale(0.5)`. Rotating -20
        // degrees about the axis (-1, 0, 0) is the same rotation as +20 degrees about (1, 0, 0).
        m *= Mat4::from_translation(Vec3::new(0.0, 0.625, 0.0));
        m *= Mat4::from_rotation_x(20.0_f32.to_radians());
        m *= Mat4::from_scale(Vec3::splat(0.5));
    }
    // java: `ModelRenderer.postRender` deliberately omits `offsetX/Y/Z`.  Those move the
    // arm geometry in `render`, but must not leak into an attachment's hand pivot.
    m *= model.get(arm).post_render_matrix(scale);
    m *= Mat4::from_translation(Vec3::new(-0.0625, 0.4375, 0.0625));
    if is_full_block {
        m *= Mat4::from_translation(Vec3::new(0.0, 0.1875, -0.3125));
        m *= Mat4::from_rotation_x(20.0_f32.to_radians());
        m *= Mat4::from_rotation_y(45.0_f32.to_radians());
        m *= Mat4::from_scale(Vec3::new(-0.375, -0.375, 0.375));
    }
    if sneaking {
        m *= Mat4::from_translation(Vec3::new(0.0, 0.203_125, 0.0));
    }
    m
}

/// The matrix an **enderman's carried block** is drawn under.
///
/// java: `LayerHeldBlock.doRenderLayer`. Unlike every other held-item layer this one never
/// touches a model part: the pose is a fixed offset from the entity's own matrix, because an
/// enderman carries the block in front of its chest rather than in a hand.
///
/// The `scale(-f, -f, f)` at the end is the *same* X/Y flip
/// [`super::render::entity_matrix`] already applied, so it cancels back to un-flipped block
/// space — which is exactly what `renderBlockBrightness` expects, since a baked block model is
/// authored in world orientation, not in the Y-down model space a `ModelRenderer` lives in.
///
/// The caller supplies the block's own geometry (`mc_assets::BakedModel` through
/// [`super::item::emit_baked_model`]) and binds the block atlas, since neither is this
/// module's. Vanilla's `getBrightnessForRender` lightmap coordinates are not applied: the
/// entity shader has no lightmap input, which is the same gap every other layer here has.
#[must_use]
pub fn enderman_held_block_matrix(base: Mat4) -> Mat4 {
    // java: translate(0, 0.6875, -0.75), rotate(20, X), rotate(45, Y),
    // translate(0.25, 0.1875, 0.25), scale(-0.5, -0.5, 0.5).
    base * Mat4::from_translation(Vec3::new(0.0, 0.6875, -0.75))
        * Mat4::from_rotation_x(20.0_f32.to_radians())
        * Mat4::from_rotation_y(45.0_f32.to_radians())
        * Mat4::from_translation(Vec3::new(0.25, 0.1875, 0.25))
        * Mat4::from_scale(Vec3::new(-0.5, -0.5, 0.5))
}

/// The matrix an **iron golem's poppy** is drawn under while it is offering one.
///
/// java: `LayerIronGolemFlower.doRenderLayer`, drawn only when
/// `EntityIronGolem.getHoldRoseTick() != 0` — which
/// [`super::models::passive::IronGolemModel::hold_rose_tick`] already carries.
///
/// `arm_rotate_x` is the golem's **right arm** pitch in radians, after posing
/// (`ModelIronGolem.ironGolemRightArm.rotateAngleX`); the flower tracks the arm rather than
/// hanging off it, which is why this takes an angle instead of a [`PartId`]. Vanilla writes the
/// term as `5 + 180 * rotateAngleX / PI` degrees, i.e. five degrees plus the arm's own pitch
/// converted to degrees — reproduced here in radians without the round trip.
///
/// Like the enderman's block this ends in a flip (`scale(0.5, -0.5, 0.5)`) that undoes the
/// model-space Y inversion, because a poppy is a baked block model.
#[must_use]
pub fn iron_golem_flower_matrix(base: Mat4, arm_rotate_x: f32) -> Mat4 {
    base * Mat4::from_rotation_x(5.0_f32.to_radians() + arm_rotate_x)
        * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
        * Mat4::from_translation(Vec3::new(-0.9375, -0.625, -0.9375))
        * Mat4::from_scale(Vec3::new(0.5, -0.5, 0.5))
}

/// Which of `LayerHeldItemWitch`'s four mutually exclusive item poses applies.
///
/// java: the `if / else if / else if / else` chain in `LayerHeldItemWitch.doRenderLayer`. The
/// branch is chosen from the item, not the entity, and the four poses are genuinely different —
/// a witch holds a potion very differently from a bow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WitchHeldItem {
    /// An `ItemBlock` whose block renders as a chest (`isRenderTypeChest`). Chests are drawn
    /// from a tile-entity model rather than a baked one, hence their own branch.
    Chest,
    /// `Items.bow`, which vanilla lays almost flat and swings back.
    Bow,
    /// Any `Item.isFull3D()` — swords, tools, rods. `rotate_around` is
    /// `Item.shouldRotateAroundWhenRendering()`, true only for the fishing rod and the carrot on
    /// a stick, which vanilla spins 180° about Z so the tip points the right way.
    Full3D {
        /// java: `shouldRotateAroundWhenRendering()`
        rotate_around: bool,
    },
    /// Everything else — the flat sprite case, which is what a potion actually takes.
    Flat,
}

/// The matrix a **witch's held item** is drawn under.
///
/// java: `LayerHeldItemWitch.doRenderLayer`. A witch holds its item at the *nose*, not in a
/// hand, so this is the one held-item pose that hangs off a head part.
///
/// Two things here look like bugs and are not:
///
/// * `villagerNose.postRender(0.0625F)` applies **only the nose's own local transform**, never
///   its parent chain — `ModelRenderer.postRender` walks nothing. The nose is a child of
///   `villagerHead` in the model tree, so vanilla is composing a head-relative pivot onto the
///   *entity* matrix. Reproduced exactly ([`Model::get`] +
///   [`super::model::ModelPart::post_render_matrix`]),
///   because "fixing" it would move every witch's potion.
/// * The trailing `rotate(0, X)` / `rotate(0, Z)` in the `isFull3D` branch are no-ops in the
///   Java and are simply omitted.
///
/// `full_3d_layer_lift` is the renderer's `transformHeldFull3DItemLayer` hook, which for
/// `RenderWitch` is `translate(0, 0.1875, 0)` and for the `RendererLivingEntity` default is
/// nothing; it is a parameter so the same function serves both.
#[must_use]
pub fn witch_held_item_matrix(
    model: &Model,
    nose: PartId,
    scale: f32,
    base: Mat4,
    child: bool,
    item: WitchHeldItem,
    full_3d_layer_lift: Vec3,
) -> Mat4 {
    let mut m = base;
    if child {
        // java: translate(0, 0.625, 0), rotate(-20, -1, 0, 0) == +20 about X, scale(0.5).
        m *= Mat4::from_translation(Vec3::new(0.0, 0.625, 0.0));
        m *= Mat4::from_rotation_x(20.0_f32.to_radians());
        m *= Mat4::from_scale(Vec3::splat(0.5));
    }
    // Like every Java `postRender` caller, a witch-held item uses the pivot/rotation only;
    // `offsetX/Y/Z` belong exclusively to the geometry-render path.
    m *= model.get(nose).post_render_matrix(scale);
    m *= Mat4::from_translation(Vec3::new(-0.0625, 0.53125, 0.21875));
    match item {
        WitchHeldItem::Chest => {
            m *= Mat4::from_translation(Vec3::new(0.0, 0.0625, -0.25));
            m *= Mat4::from_rotation_x(30.0_f32.to_radians());
            m *= Mat4::from_rotation_y((-5.0_f32).to_radians());
            m *= Mat4::from_scale(Vec3::new(0.375, -0.375, 0.375));
        }
        WitchHeldItem::Bow => {
            m *= Mat4::from_translation(Vec3::new(0.0, 0.125, -0.125));
            m *= Mat4::from_rotation_y((-45.0_f32).to_radians());
            m *= Mat4::from_scale(Vec3::new(0.625, -0.625, 0.625));
            m *= Mat4::from_rotation_x((-100.0_f32).to_radians());
            m *= Mat4::from_rotation_y((-20.0_f32).to_radians());
        }
        WitchHeldItem::Full3D { rotate_around } => {
            if rotate_around {
                m *= Mat4::from_rotation_z(std::f32::consts::PI);
                m *= Mat4::from_translation(Vec3::new(0.0, -0.0625, 0.0));
            }
            m *= Mat4::from_translation(full_3d_layer_lift);
            m *= Mat4::from_translation(Vec3::new(0.0625, -0.125, 0.0));
            m *= Mat4::from_scale(Vec3::new(0.625, -0.625, 0.625));
        }
        WitchHeldItem::Flat => {
            m *= Mat4::from_translation(Vec3::new(0.1875, 0.1875, 0.0));
            m *= Mat4::from_scale(Vec3::splat(0.875));
            m *= Mat4::from_rotation_z((-20.0_f32).to_radians());
            m *= Mat4::from_rotation_x((-60.0_f32).to_radians());
            m *= Mat4::from_rotation_z((-30.0_f32).to_radians());
        }
    }
    // java: applied after the branch, to every case.
    m *= Mat4::from_rotation_x((-15.0_f32).to_radians());
    m *= Mat4::from_rotation_z(40.0_f32.to_radians());
    m
}

/// `RenderWitch.transformHeldFull3DItemLayer` — the lift a witch's full-3D item gets.
///
/// java: `GlStateManager.translate(0.0F, 0.1875F, 0.0F)`. `RendererLivingEntity`'s default is
/// `Vec3::ZERO`, so pass that for any renderer that does not override the hook.
pub const WITCH_FULL_3D_LIFT: Vec3 = Vec3::new(0.0, 0.1875, 0.0);

#[cfg(test)]
mod tests {
    use super::super::anim::EntityModel;
    use super::*;

    #[test]
    fn brightness_combine_policy_matches_every_live_java_layer() {
        for layer in [
            LayerKind::SheepWool,
            LayerKind::WolfCollar,
            LayerKind::MooshroomMushroom,
            LayerKind::SnowmanHead,
            LayerKind::SlimeGel,
            LayerKind::CustomHead,
            LayerKind::Deadmau5Head,
        ] {
            assert!(layer.combines_brightness(), "{layer:?}");
        }
        for layer in [
            LayerKind::Armor,
            LayerKind::HeldItem,
            LayerKind::HeldBlock,
            LayerKind::Cape,
            LayerKind::CreeperCharge,
            LayerKind::EndermanEyes,
            LayerKind::SpiderEyes,
            LayerKind::Saddle,
            LayerKind::IronGolemFlower,
            LayerKind::WitherAura,
            LayerKind::Arrow,
            LayerKind::WitchHeldItem,
        ] {
            assert!(!layer.combines_brightness(), "{layer:?}");
        }

        assert_eq!(
            LayerBlend::Opaque.living_hurt(),
            Some(LayerBlend::LivingHurt)
        );
        // java: the slime gel is the one combining layer with a blended source pass, and it
        // keeps `depthMask(true)` (`RendererLivingEntity:174`), so its hurt variant does too.
        assert_eq!(
            LayerBlend::TranslucentDepthWrite.living_hurt(),
            Some(LayerBlend::LivingHurtTranslucent)
        );
        for blend in [
            LayerBlend::OpaqueUnlit,
            LayerBlend::Additive,
            LayerBlend::AdditiveNoDepthWrite,
            LayerBlend::Translucent,
            LayerBlend::MapTranslucentDepthWrite,
            LayerBlend::Glint,
            LayerBlend::LivingHurt,
            LayerBlend::LivingHurtTranslucent,
            LayerBlend::LivingHurtGhost,
            LayerBlend::LivingHurtGlint,
            LayerBlend::WhiteFlash,
            LayerBlend::TntFlash,
        ] {
            assert_eq!(blend.living_hurt(), None, "{blend:?}");
        }
    }

    #[test]
    fn armour_is_drawn_helmet_first_boots_last() {
        assert_eq!(
            ArmorSlot::DRAW_ORDER.map(ArmorSlot::slot_index),
            [4, 3, 2, 1]
        );
    }

    #[test]
    fn only_leggings_use_the_second_texture_sheet() {
        for s in ArmorSlot::DRAW_ORDER {
            let want = if s == ArmorSlot::Leggings { 2 } else { 1 };
            assert_eq!(s.texture_layer(), want, "{s:?}");
        }
    }

    #[test]
    fn armour_texture_paths_match_the_java_format_string() {
        assert_eq!(
            armor_texture("diamond", ArmorSlot::Helmet, None),
            "textures/models/armor/diamond_layer_1.png"
        );
        assert_eq!(
            armor_texture("diamond", ArmorSlot::Leggings, None),
            "textures/models/armor/diamond_layer_2.png"
        );
        assert_eq!(
            armor_texture("leather", ArmorSlot::Chestplate, Some("overlay")),
            "textures/models/armor/leather_layer_1_overlay.png"
        );
        assert_eq!(
            armor_texture("leather", ArmorSlot::Leggings, Some("overlay")),
            "textures/models/armor/leather_layer_2_overlay.png"
        );
    }

    #[test]
    fn static_armour_keys_equal_the_formatted_paths() {
        use mc_data::ArmorMaterial as M;
        for material in [M::Leather, M::Chain, M::Iron, M::Gold, M::Diamond] {
            for slot in ArmorSlot::DRAW_ORDER {
                for overlay in [false, true] {
                    assert_eq!(
                        armor_texture_key(material, slot, overlay),
                        armor_texture(
                            armor_material_name(material),
                            slot,
                            overlay.then_some("overlay")
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn chestplate_covers_arms_and_leggings_cover_legs_and_both_cover_the_body() {
        let chest = ArmorSlot::Chestplate.visible_parts();
        let legs = ArmorSlot::Leggings.visible_parts();
        assert!(chest.contains(&ArmorPart::RightArm) && chest.contains(&ArmorPart::LeftArm));
        assert!(!chest.contains(&ArmorPart::RightLeg));
        assert!(legs.contains(&ArmorPart::RightLeg) && legs.contains(&ArmorPart::LeftLeg));
        assert!(!legs.contains(&ArmorPart::RightArm));
        assert!(chest.contains(&ArmorPart::Body) && legs.contains(&ArmorPart::Body));
        // Boots are legs only, helmet is head plus the hat layer.
        assert_eq!(
            ArmorSlot::Boots.visible_parts(),
            &[ArmorPart::RightLeg, ArmorPart::LeftLeg]
        );
        assert_eq!(
            ArmorSlot::Helmet.visible_parts(),
            &[ArmorPart::Head, ArmorPart::Headwear]
        );
    }

    #[test]
    fn leggings_use_the_thinner_armour_model() {
        assert_eq!(ArmorSlot::Leggings.model_inflation(), 0.5);
        for s in [ArmorSlot::Helmet, ArmorSlot::Chestplate, ArmorSlot::Boots] {
            assert_eq!(s.model_inflation(), 1.0, "{s:?}");
        }
    }

    #[test]
    fn glowing_layers_are_additive_and_the_rest_are_not() {
        for k in [
            LayerKind::CreeperCharge,
            LayerKind::EndermanEyes,
            LayerKind::SpiderEyes,
            // java: `LayerWitherAura:45` is `blendFunc(1, 1)`, exactly `LayerCreeperCharge:40`.
            // This assertion used to say `Translucent`, matching neither the Java nor the wither's
            // actual appearance.
            LayerKind::WitherAura,
        ] {
            assert_eq!(k.blend(), LayerBlend::Additive, "{k:?}");
        }
        assert_eq!(LayerKind::Saddle.blend(), LayerBlend::Opaque);
        assert_eq!(LayerKind::SheepWool.blend(), LayerBlend::Opaque);
        // java: `LayerSlimeGel:26` — `blendFunc(770, 771)` with the depth mask left on. The one
        // genuinely alpha-blended layer, and it still writes depth.
        assert_eq!(
            LayerKind::SlimeGel.blend(),
            LayerBlend::TranslucentDepthWrite
        );
        assert_eq!(FIRE_OVERLAY_BLEND, LayerBlend::OpaqueUnlit);
    }

    #[test]
    fn invisible_additive_layers_keep_depth_testing_without_depth_writes() {
        for layer in [
            LayerKind::EndermanEyes,
            LayerKind::SpiderEyes,
            LayerKind::DragonEyes,
            LayerKind::CreeperCharge,
            LayerKind::WitherAura,
        ] {
            assert_eq!(layer.blend_for(false), LayerBlend::Additive, "{layer:?}");
            assert_eq!(
                layer.blend_for(true),
                LayerBlend::AdditiveNoDepthWrite,
                "{layer:?}"
            );
        }
        assert_eq!(
            LayerKind::SlimeGel.blend_for(true),
            LayerBlend::TranslucentDepthWrite
        );
    }

    /// java: the `GlStateManager.color` each layer sets before its own draw.
    #[test]
    fn the_two_aura_layers_draw_at_half_brightness() {
        assert_eq!(LayerKind::CreeperCharge.tint(), Some([0.5, 0.5, 0.5, 1.0]));
        assert_eq!(LayerKind::WitherAura.tint(), Some([0.5, 0.5, 0.5, 1.0]));
        assert_eq!(LayerKind::SpiderEyes.tint(), Some([1.0, 1.0, 1.0, 1.0]));
        assert_eq!(LayerKind::Armor.tint(), Some([1.0, 1.0, 1.0, 1.0]));
        // The two dye-driven layers have no constant colour at all.
        assert_eq!(LayerKind::SheepWool.tint(), None);
        assert_eq!(LayerKind::WolfCollar.tint(), None);
    }

    #[test]
    fn dynamic_dye_input_only_reaches_the_two_dye_layers() {
        let dye = LayerTintInput {
            dye: Some([0.6, 0.2, 0.2, 1.0]),
        };
        assert_eq!(
            LayerKind::SheepWool.resolved_tint(dye),
            [0.6, 0.2, 0.2, 1.0]
        );
        assert_eq!(
            LayerKind::WolfCollar.resolved_tint(dye),
            [0.6, 0.2, 0.2, 1.0]
        );
        assert_eq!(LayerKind::Armor.resolved_tint(dye), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            LayerKind::CreeperCharge.resolved_tint(dye),
            [0.5, 0.5, 0.5, 1.0]
        );
        assert_eq!(
            LayerKind::SheepWool.resolved_tint(LayerTintInput::default()),
            [1.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn the_charged_creeper_aura_scrolls_and_inflates() {
        assert_eq!(LayerKind::CreeperCharge.model_inflation(), 2.0);
        assert_eq!(
            LayerKind::CreeperCharge.texture_scroll(100.0),
            Some([1.0, 1.0]),
            "java: translate(f * 0.01, f * 0.01, 0)"
        );
        // java: `LayerWitherAura:36-38` — `cos(f * 0.02) * 3.0` on X, `f * 0.01` on Y. The two
        // axes are *not* the same expression, which is what this used to assume.
        let wither = LayerKind::WitherAura.texture_scroll(100.0).unwrap();
        assert!(
            (wither[0] - mc_math::cos(2.0) * 3.0).abs() < 1e-6,
            "x sways: {}",
            wither[0]
        );
        assert!((wither[1] - 1.0).abs() < 1e-6, "y creeps: {}", wither[1]);
        assert_eq!(LayerKind::Saddle.texture_scroll(100.0), None);
        assert_eq!(
            LayerKind::CreeperCharge.texture(),
            Some("textures/entity/creeper/creeper_armor.png")
        );
        assert_eq!(LayerKind::CreeperCharge.model(), Some(ModelKind::Creeper));
    }

    #[test]
    fn pig_saddle_has_its_own_half_unit_inflated_pig_model() {
        assert_eq!(LayerKind::Saddle.model(), Some(ModelKind::Pig));
        assert_eq!(LayerKind::Saddle.model_inflation(), 0.5);
        assert_eq!(
            LayerKind::Saddle.texture(),
            Some("textures/entity/pig/pig_saddle.png")
        );
    }

    #[test]
    fn item_layers_carry_no_model_of_their_own() {
        for k in [
            LayerKind::HeldItem,
            LayerKind::HeldBlock,
            LayerKind::WitchHeldItem,
            LayerKind::IronGolemFlower,
            LayerKind::Armor,
        ] {
            assert_eq!(k.model(), None, "{k:?} geometry comes from elsewhere");
        }
    }

    #[test]
    fn a_still_player_has_a_hanging_cape() {
        // Nothing moving: f1 = 0, f2 = 0, f3 = 0, so only the constant 6 degrees remain.
        let s = CapeState::default();
        let (x, z, y) = cape_angles(&s);
        assert!((x - 6.0).abs() < 1e-5, "{x}");
        assert_eq!(z, 0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn sneaking_lifts_the_cape_by_twenty_five_degrees() {
        let s = CapeState {
            sneaking: true,
            ..Default::default()
        };
        let (x, _, _) = cape_angles(&s);
        assert!((x - 31.0).abs() < 1e-5, "6 + 25, got {x}");
    }

    #[test]
    fn a_backwards_chase_offset_cannot_push_the_cape_through_the_body() {
        // java: `if (f2 < 0) f2 = 0` — the forward term is clamped at zero, so running
        // backwards does not fold the cape into the player's chest.
        let s = CapeState {
            chase_dz: 1.0,
            body_yaw: 0.0,
            ..Default::default()
        };
        // yaw 0: d3 = sin(0) = 0, d4 = -cos(0) = -1, so f2 = (0 + 1 * -1) * 100 = -100 -> 0.
        let (x, _, _) = cape_angles(&s);
        assert!((x - 6.0).abs() < 1e-4, "clamped, got {x}");
        // The opposite offset is not clamped.
        let s2 = CapeState {
            chase_dz: -1.0,
            ..Default::default()
        };
        let (x2, _, _) = cape_angles(&s2);
        assert!(x2 > 50.0, "a forward chase offset lifts the cape: {x2}");
    }

    #[test]
    fn vertical_chase_is_clamped_both_ways() {
        // java: MathHelper.clamp_float(d1 * 10, -6, 32)
        let up = CapeState {
            chase_dy: 100.0,
            ..Default::default()
        };
        let (x, _, _) = cape_angles(&up);
        assert!((x - 38.0).abs() < 1e-4, "6 + 32, got {x}");
        let down = CapeState {
            chase_dy: -100.0,
            ..Default::default()
        };
        let (x, _, _) = cape_angles(&down);
        assert!((x - 0.0).abs() < 1e-4, "6 - 6, got {x}");
    }

    #[test]
    fn the_cape_matrix_pushes_the_cape_behind_the_body() {
        let m = cape_matrix(&CapeState::default());
        let o = m.transform_point3(Vec3::ZERO);
        assert!((o.z - 0.125).abs() < 1e-6, "java: translate(0, 0, 0.125)");
        // The trailing 180-degree Y rotation turns the cape to face backwards.
        let f = m.transform_vector3(Vec3::Z).normalize();
        assert!(f.z < 0.0, "{f}");
    }

    fn assert_matrix_close(got: Mat4, want: Mat4) {
        for (a, b) in got.to_cols_array().iter().zip(want.to_cols_array()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn custom_head_child_and_sneak_adjustments_stay_in_entity_space() {
        let mut model = Model::new(64, 32);
        let head = model
            .part("head", 0, 0)
            .at(2.0, 4.0, -3.0)
            .angles(0.3, -0.2, 0.4)
            .id();
        let base = Mat4::from_translation(Vec3::new(11.0, 7.0, -5.0)) * Mat4::from_rotation_y(0.6);
        let scale = 0.0625;

        let got = custom_head_matrix(
            base,
            &model,
            head,
            scale,
            true,
            true,
            false,
            CustomHeadItem::Block,
        );
        // Java's push/pop stack is base * sneak * child-scale * child-translate * postRender,
        // followed by the ItemBlock branch.  This explicit chain is the negative control for the
        // tempting (and wrong) `translate * (base * postRender)` implementation.
        let want = base
            * Mat4::from_translation(Vec3::new(0.0, 0.2, 0.0))
            * Mat4::from_scale(Vec3::splat(0.7))
            * Mat4::from_translation(Vec3::new(0.0, 16.0 * scale, 0.0))
            * model.get(head).post_render_matrix(scale)
            * Mat4::from_translation(Vec3::new(0.0, -0.25, 0.0))
            * Mat4::from_rotation_y(core::f32::consts::PI)
            * Mat4::from_scale(Vec3::new(0.625, -0.625, -0.625));
        assert_matrix_close(got, want);

        let wrong_world_space = Mat4::from_translation(Vec3::new(0.0, 0.2, 0.0))
            * Mat4::from_scale(Vec3::splat(0.7))
            * Mat4::from_translation(Vec3::new(0.0, 16.0 * scale, 0.0))
            * base
            * model.get(head).post_render_matrix(scale)
            * Mat4::from_translation(Vec3::new(0.0, -0.25, 0.0))
            * Mat4::from_rotation_y(core::f32::consts::PI)
            * Mat4::from_scale(Vec3::new(0.625, -0.625, -0.625));
        assert_ne!(
            got.transform_point3(Vec3::ZERO),
            wrong_world_space.transform_point3(Vec3::ZERO),
            "child/sneak offsets must follow the entity basis, not the world basis"
        );
    }

    #[test]
    fn custom_head_villager_branch_suppresses_child_lift_and_adds_item_offset() {
        let mut model = Model::new(64, 32);
        let head = model.part("head", 0, 0).at(0.0, 8.0, 0.0).id();
        let base = Mat4::IDENTITY;
        let scale = 0.0625;

        let block = custom_head_matrix(
            base,
            &model,
            head,
            scale,
            false,
            true,
            true,
            CustomHeadItem::Block,
        );
        let skull = custom_head_matrix(
            base,
            &model,
            head,
            scale,
            false,
            true,
            true,
            CustomHeadItem::Skull,
        );
        let head_post = model.get(head).post_render_matrix(scale);
        let want_block = head_post
            * Mat4::from_translation(Vec3::new(0.0, -0.25, 0.0))
            * Mat4::from_rotation_y(core::f32::consts::PI)
            * Mat4::from_scale(Vec3::new(0.625, -0.625, -0.625))
            * Mat4::from_translation(Vec3::new(0.0, 0.1875, 0.0));
        let want_skull = head_post
            * Mat4::from_scale(Vec3::new(1.1875, -1.1875, -1.1875))
            * Mat4::from_translation(Vec3::new(0.0, 0.0625, 0.0));
        assert_matrix_close(block, want_block);
        assert_matrix_close(skull, want_skull);

        // The same child flag on an ordinary biped must still take the generic .7/one-block
        // branch; this catches accidentally applying the villager exception to every child.
        let ordinary_child = custom_head_matrix(
            base,
            &model,
            head,
            scale,
            false,
            true,
            false,
            CustomHeadItem::Skull,
        );
        assert_ne!(
            ordinary_child.transform_point3(Vec3::ZERO),
            skull.transform_point3(Vec3::ZERO),
            "villager-like is a real renderer-family gate"
        );
    }

    #[test]
    fn a_helmet_pass_shows_only_the_head_and_hides_the_rest() {
        let mut model = BipedModel::new(ArmorSlot::Helmet.model_inflation());
        let p = AnimParams::still();
        let mut out = Vec::new();
        emit_armor(&mut model, ArmorSlot::Helmet, &p, Mat4::IDENTITY, &mut out);
        // 2 visible parts (head, headwear), one box each, 6 faces, 6 vertices.
        assert_eq!(out.len(), 2 * 6 * 6);

        // Boots is the mirror image: only the two legs.
        let mut out2 = Vec::new();
        emit_armor(&mut model, ArmorSlot::Boots, &p, Mat4::IDENTITY, &mut out2);
        assert_eq!(out2.len(), 2 * 6 * 6);
    }

    #[test]
    fn a_chestplate_pass_shows_the_body_and_both_arms() {
        let mut model = BipedModel::new(ArmorSlot::Chestplate.model_inflation());
        let p = AnimParams::still();
        let mut out = Vec::new();
        emit_armor(
            &mut model,
            ArmorSlot::Chestplate,
            &p,
            Mat4::IDENTITY,
            &mut out,
        );
        // body + right_arm + left_arm, one box each.
        assert_eq!(out.len(), 3 * 6 * 6);
    }

    #[test]
    fn armor_is_inflated_to_the_slots_scale() {
        // java: `getArmorModel` hands the leggings pass a `ModelBiped(0.5F)` and every other
        // pass a `ModelBiped(1.0F)`. Compare the *same* slot's visible parts (chestplate: body +
        // both arms) at the two real inflation constants, so the only variable is the inflation
        // — comparing different slots would also change which anatomical parts are visible,
        // which have different natural extents regardless of inflation.
        let extent_x = |inflation: f32| {
            let mut model = BipedModel::new(inflation);
            let p = AnimParams::still();
            let mut out = Vec::new();
            emit_armor(
                &mut model,
                ArmorSlot::Chestplate,
                &p,
                Mat4::IDENTITY,
                &mut out,
            );
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for v in &out {
                lo = lo.min(v.pos[0]);
                hi = hi.max(v.pos[0]);
            }
            hi - lo
        };
        assert!(
            extent_x(ArmorSlot::Leggings.model_inflation())
                < extent_x(ArmorSlot::Helmet.model_inflation()),
            "0.5 inflation must be thinner than 1.0 on the same slot's visible parts"
        );
    }

    #[test]
    fn armor_is_posed_the_same_way_the_body_is() {
        // The armour model must swing its arms/legs with the same AnimParams the body used —
        // otherwise a walking player's sleeves would not track their arms.
        let mut model = BipedModel::new(1.0);
        let mut p = AnimParams::still();
        p.limb_swing = 5.0;
        p.limb_swing_amount = 1.0;
        let mut out = Vec::new();
        emit_armor(
            &mut model,
            ArmorSlot::Chestplate,
            &p,
            Mat4::IDENTITY,
            &mut out,
        );

        let mut still = Vec::new();
        let mut still_model = BipedModel::new(1.0);
        emit_armor(
            &mut still_model,
            ArmorSlot::Chestplate,
            &AnimParams::still(),
            Mat4::IDENTITY,
            &mut still,
        );
        assert_ne!(out, still, "a swinging pose must move the armour geometry");
    }

    #[test]
    fn the_cape_is_hidden_until_the_layer_draws_it() {
        let mut model = PlayerModel::new(0.0, false);
        // The body's own draw pass never includes the cape part.
        let mut body_out = Vec::new();
        let mut p = AnimParams::still();
        p.entity.child = false;
        model.model_mut().is_child = false;
        model.pose_and_render(&p, Mat4::IDENTITY, &mut body_out);
        assert_eq!(
            body_out.len(),
            12 * 6 * 6,
            "7 biped parts + 5 wear parts, no cape"
        );

        let mut cape_out = Vec::new();
        emit_cape(
            &mut model,
            &CapeState::default(),
            p.scale,
            Mat4::IDENTITY,
            &mut cape_out,
        );
        assert_eq!(cape_out.len(), 6 * 6, "one cape box");
        assert!(cape_out.iter().all(|v| v.pos.iter().all(|c| c.is_finite())));
    }

    #[test]
    fn sneaking_moves_the_cape_geometry() {
        let mut model = PlayerModel::new(0.0, false);
        let mut p = AnimParams::still();
        p.entity.child = false;
        model.model_mut().is_child = false;
        let mut out = Vec::new();
        model.pose_and_render(&p, Mat4::IDENTITY, &mut out);

        let mut standing = Vec::new();
        emit_cape(
            &mut model,
            &CapeState::default(),
            p.scale,
            Mat4::IDENTITY,
            &mut standing,
        );

        p.entity.sneaking = true;
        let mut out2 = Vec::new();
        model.pose_and_render(&p, Mat4::IDENTITY, &mut out2);
        let sneaking_state = CapeState {
            sneaking: true,
            ..Default::default()
        };
        let mut sneaking = Vec::new();
        emit_cape(
            &mut model,
            &sneaking_state,
            p.scale,
            Mat4::IDENTITY,
            &mut sneaking,
        );

        assert_ne!(
            standing, sneaking,
            "the sneak pivot lift must move the cape"
        );
    }

    // ---------------------------------------------- the three species-specific held poses

    /// Column magnitudes of the linear part, which are the net scale factors when everything
    /// after the scale is a pure rotation — the case in all three poses below.
    fn axis_lengths(m: Mat4) -> [f32; 3] {
        [
            m.x_axis.truncate().length(),
            m.y_axis.truncate().length(),
            m.z_axis.truncate().length(),
        ]
    }

    fn determinant3(m: Mat4) -> f32 {
        glam::Mat3::from_mat4(m).determinant()
    }

    #[test]
    fn the_enderman_holds_its_block_in_front_of_and_above_its_origin() {
        let m = enderman_held_block_matrix(Mat4::IDENTITY);
        let o = m.transform_point3(Vec3::ZERO);
        // java: translate(0, 0.6875, -0.75) dominates — the block sits forward of the chest.
        assert!(o.z < -0.6, "the block is carried in front: {o}");
        assert!(o.y > 0.8, "and above the origin: {o}");
        // scale(-f, -f, f) with f = 0.5: a half-size block, X and Y flipped.
        for (i, len) in axis_lengths(m).into_iter().enumerate() {
            assert!((len - 0.5).abs() < 1e-6, "axis {i} is {len}, expected 0.5");
        }
        assert!(
            determinant3(m) > 0.0,
            "two negated axes leave the handedness alone — this is what un-does \
             entity_matrix's own scale(-1, -1, 1)"
        );
        assert!(
            m.x_axis.x < 0.0,
            "the X flip must survive the two rotations"
        );
        assert!(m.y_axis.y < 0.0, "and so must the Y flip");
    }

    #[test]
    fn the_iron_golems_poppy_tracks_its_right_arm_plus_five_degrees() {
        // java: `rotate(5 + 180 * rotateAngleX / PI, X)` then `rotate(90, X)`. Feeding the arm
        // -5 degrees makes the total exactly a quarter turn, which pins both terms at once.
        let m = iron_golem_flower_matrix(Mat4::IDENTITY, (-5.0_f32).to_radians());
        let y = m.transform_vector3(Vec3::Y);
        assert!(y.x.abs() < 1e-6, "{y}");
        assert!(y.y.abs() < 1e-6, "a quarter turn puts model +Y flat: {y}");
        assert!((y.z + 0.5).abs() < 1e-6, "{y}");
        // The arm angle really drives it.
        let swung = iron_golem_flower_matrix(Mat4::IDENTITY, 0.5);
        assert_ne!(
            m.transform_vector3(Vec3::Y),
            swung.transform_vector3(Vec3::Y)
        );
        // scale(f, -f, f) with f = 0.5 — one negated axis, so handedness *does* flip.
        for len in axis_lengths(m) {
            assert!((len - 0.5).abs() < 1e-6, "{len}");
        }
        assert!(determinant3(m) < 0.0);
    }

    fn witch_pose(item: WitchHeldItem, child: bool) -> Mat4 {
        let w = super::super::models::passive::WitchModel::new(0.0);
        witch_held_item_matrix(
            &w.model,
            w.parts.nose,
            0.0625,
            Mat4::IDENTITY,
            child,
            item,
            WITCH_FULL_3D_LIFT,
        )
    }

    #[test]
    fn each_witch_item_branch_has_its_own_scale() {
        // java: 0.375 for a chest, 0.625 for a bow and for full-3D items, 0.875 otherwise.
        for (item, want) in [
            (WitchHeldItem::Chest, 0.375),
            (WitchHeldItem::Bow, 0.625),
            (
                WitchHeldItem::Full3D {
                    rotate_around: false,
                },
                0.625,
            ),
            (WitchHeldItem::Flat, 0.875),
        ] {
            for len in axis_lengths(witch_pose(item, false)) {
                assert!(
                    (len - want).abs() < 1e-5,
                    "{item:?} scaled {len}, want {want}"
                );
            }
        }
    }

    #[test]
    fn only_the_flat_witch_pose_keeps_its_handedness() {
        // java: the first three branches spell `scale(f, -f, f)`; the flat one is `scale(f)`.
        assert!(determinant3(witch_pose(WitchHeldItem::Chest, false)) < 0.0);
        assert!(determinant3(witch_pose(WitchHeldItem::Bow, false)) < 0.0);
        assert!(
            determinant3(witch_pose(
                WitchHeldItem::Full3D {
                    rotate_around: false
                },
                false
            )) < 0.0
        );
        assert!(determinant3(witch_pose(WitchHeldItem::Flat, false)) > 0.0);
    }

    #[test]
    fn the_four_witch_poses_are_all_different() {
        let poses = [
            witch_pose(WitchHeldItem::Chest, false),
            witch_pose(WitchHeldItem::Bow, false),
            witch_pose(
                WitchHeldItem::Full3D {
                    rotate_around: false,
                },
                false,
            ),
            witch_pose(WitchHeldItem::Flat, false),
        ];
        for i in 0..poses.len() {
            for j in (i + 1)..poses.len() {
                assert_ne!(poses[i], poses[j], "branches {i} and {j} collapsed");
            }
        }
    }

    #[test]
    fn a_rod_that_rotates_around_is_spun_half_a_turn_and_dropped() {
        // java: `rotate(180, 0, 0, 1)` then `translate(0, -0.0625, 0)`, only for
        // `shouldRotateAroundWhenRendering()`.
        let plain = witch_pose(
            WitchHeldItem::Full3D {
                rotate_around: false,
            },
            false,
        );
        let spun = witch_pose(
            WitchHeldItem::Full3D {
                rotate_around: true,
            },
            false,
        );
        assert_ne!(plain, spun);
        // A half turn about Z leaves the scale alone.
        for (a, b) in axis_lengths(plain).into_iter().zip(axis_lengths(spun)) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn a_baby_witch_halves_the_whole_stack() {
        // java: the `isChild` block is pushed *before* the nose's own matrix, so the 0.5 wraps
        // everything after it, scale included.
        let grown = axis_lengths(witch_pose(WitchHeldItem::Flat, false));
        let baby = axis_lengths(witch_pose(WitchHeldItem::Flat, true));
        for (g, b) in grown.into_iter().zip(baby) {
            assert!((b - g * 0.5).abs() < 1e-5, "{b} is not half of {g}");
        }
    }

    #[test]
    fn the_witchs_item_hangs_off_the_nose_part() {
        // java: `villagerNose.postRender(0.0625F)` — the nose's own local transform, and
        // deliberately *not* its parent chain, which `ModelRenderer.postRender` never walks.
        let mut w = super::super::models::passive::WitchModel::new(0.0);
        let nose = w.parts.nose;
        let pose = |w: &super::super::models::passive::WitchModel| {
            witch_held_item_matrix(
                &w.model,
                nose,
                0.0625,
                Mat4::IDENTITY,
                false,
                WitchHeldItem::Flat,
                WITCH_FULL_3D_LIFT,
            )
        };
        let before = pose(&w);
        w.model.get_mut(nose).rotate_angle[0] += 0.4;
        assert_ne!(before, pose(&w), "twitching the nose must move the potion");
    }

    #[test]
    fn held_item_matrix_sits_at_the_hand_offset_from_the_arms_own_pivot() {
        let biped = BipedModel::new(0.0);
        let scale = 0.0625;
        let m = held_item_matrix(
            &biped.model,
            biped.parts.right_arm,
            scale,
            Mat4::IDENTITY,
            false,
            false,
            false,
        );
        let arm_post_render = biped
            .model
            .get(biped.parts.right_arm)
            .post_render_matrix(scale);
        let want = arm_post_render.transform_point3(Vec3::new(-0.0625, 0.4375, 0.0625));
        let got = m.transform_point3(Vec3::ZERO);
        assert!((got - want).length() < 1e-6, "{got} vs {want}");
    }

    #[test]
    fn held_item_post_render_ignores_an_arms_geometry_offset() {
        // Mutation oracle: replacing `post_render_matrix` above with `local_matrix` moves the
        // item by this unscaled offset and makes this fail.  Java `postRender` never applies
        // `ModelRenderer.offsetX/Y/Z`.
        let mut biped = BipedModel::new(0.0);
        let scale = 0.0625;
        let pose = |biped: &BipedModel| {
            held_item_matrix(
                &biped.model,
                biped.parts.right_arm,
                scale,
                Mat4::IDENTITY,
                false,
                false,
                false,
            )
        };
        let before = pose(&biped);
        biped.model.get_mut(biped.parts.right_arm).offset = [0.25, -0.5, 0.125];
        assert_eq!(
            pose(&biped),
            before,
            "an arm's render-only offset must not displace its held item"
        );
    }

    #[test]
    fn witch_held_item_post_render_ignores_the_noses_geometry_offset() {
        // `ModelWitch.setRotationAngles` changes this very offset while the witch is sneaking;
        // Java attaches the item with `villagerNose.postRender`, not `render`.
        let mut witch = super::super::models::passive::WitchModel::new(0.0);
        let nose = witch.parts.nose;
        let pose = |witch: &super::super::models::passive::WitchModel| {
            witch_held_item_matrix(
                &witch.model,
                nose,
                0.0625,
                Mat4::IDENTITY,
                false,
                WitchHeldItem::Flat,
                WITCH_FULL_3D_LIFT,
            )
        };
        let before = pose(&witch);
        witch.model.get_mut(nose).offset = [0.25, 0.1875, -0.09375];
        assert_eq!(
            pose(&witch),
            before,
            "the witch nose's render-only offset must not displace its held item"
        );
    }

    #[test]
    fn a_childs_held_item_is_lifted_tilted_and_halved() {
        let biped = BipedModel::new(0.0);
        let scale = 0.0625;
        let adult = held_item_matrix(
            &biped.model,
            biped.parts.right_arm,
            scale,
            Mat4::IDENTITY,
            false,
            false,
            false,
        );
        let child = held_item_matrix(
            &biped.model,
            biped.parts.right_arm,
            scale,
            Mat4::IDENTITY,
            true,
            false,
            false,
        );
        // The child branch pushes an extra scale(0.5), so a unit vector along the matrix's own
        // local axis is half as long.
        let adult_scale = adult.transform_vector3(Vec3::X).length();
        let child_scale = child.transform_vector3(Vec3::X).length();
        assert!(
            (child_scale - adult_scale * 0.5).abs() < 1e-5,
            "{child_scale} vs {adult_scale}"
        );
        assert_ne!(
            adult.transform_point3(Vec3::ZERO),
            child.transform_point3(Vec3::ZERO),
            "the child branch also translates and rotates before the arm's own matrix"
        );
    }

    #[test]
    fn a_full_block_item_gets_the_extra_pose_a_flat_item_does_not() {
        let biped = BipedModel::new(0.0);
        let scale = 0.0625;
        let flat = held_item_matrix(
            &biped.model,
            biped.parts.right_arm,
            scale,
            Mat4::IDENTITY,
            false,
            false,
            false,
        );
        let block = held_item_matrix(
            &biped.model,
            biped.parts.right_arm,
            scale,
            Mat4::IDENTITY,
            false,
            true,
            false,
        );
        assert_ne!(
            flat.transform_point3(Vec3::ZERO),
            block.transform_point3(Vec3::ZERO)
        );
        // java: `scale(-f1, -f1, f1)` — the block branch mirrors X and Y.
        let flipped = block.transform_vector3(Vec3::X);
        assert!(
            flipped.x < 0.0 || flipped.y < 0.0 || flipped.z < 0.0,
            "{flipped}"
        );
    }

    #[test]
    fn sneaking_lifts_a_held_item_too() {
        let biped = BipedModel::new(0.0);
        let scale = 0.0625;
        let standing = held_item_matrix(
            &biped.model,
            biped.parts.right_arm,
            scale,
            Mat4::IDENTITY,
            false,
            false,
            false,
        );
        let sneaking = held_item_matrix(
            &biped.model,
            biped.parts.right_arm,
            scale,
            Mat4::IDENTITY,
            false,
            false,
            true,
        );
        assert_ne!(
            standing.transform_point3(Vec3::ZERO),
            sneaking.transform_point3(Vec3::ZERO)
        );
    }

    #[test]
    fn left_and_right_hands_use_their_own_arms_pivot() {
        let biped = BipedModel::new(0.0);
        let scale = 0.0625;
        let right = held_item_matrix(
            &biped.model,
            biped.parts.right_arm,
            scale,
            Mat4::IDENTITY,
            false,
            false,
            false,
        );
        let left = held_item_matrix(
            &biped.model,
            biped.parts.left_arm,
            scale,
            Mat4::IDENTITY,
            false,
            false,
            false,
        );
        let rp = right.transform_point3(Vec3::ZERO);
        let lp = left.transform_point3(Vec3::ZERO);
        assert!(rp.x < 0.0, "the right arm sits at -x: {rp}");
        assert!(lp.x > 0.0, "the left arm sits at +x: {lp}");
    }
}
