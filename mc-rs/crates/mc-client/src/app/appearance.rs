//! The client's entity-appearance seam.
//!
//! A second `impl Client` block, so the entity-appearance seam reads as one file instead of as
//! scattered arms of [`super::driver`]: the equipment reader, the per-entity layer stack
//! submission, the fire overlay and shadow passes, and the skin cache.
//!
//! Everything here answers one of vanilla's four "how does this entity look right now" questions,
//! each of which was computed correctly somewhere in `mc-render` and then read by nobody:
//!
//! | Question | java | Here |
//! |---|---|---|
//! | What colour is it multiplied by? | `RendererLivingEntity.setBrightness`, `getColorMultiplier`, `LayerSheepWool`'s `getDyeRgb` | [`Tints`] |
//! | Is it drawn at all? | `RendererLivingEntity.renderModel`'s `isInvisible` fork | [`BodyDraw`] |
//! | Is it on fire? | `Render.renderEntityOnFire` | [`emit_fire_overlay`] |
//! | Does it cast a shadow? | `Render.renderShadow`/`renderShadowBlock` | [`emit_shadow`] |
//!
//! plus the skin pipeline ([`ProfileTextures`]), which is the one piece of this seam that
//! genuinely cannot be finished without a network.

use std::collections::HashMap;

use mc_math::{BlockPos, DVec3, Vec3};
use mc_render::entity::{EntityVertex, Mat4};
use mc_render::Camera;
use mc_world::World;
use uuid::Uuid;

use crate::sync::entities::TrackedEntity;
use crate::sync::watched::DyeColour;

// -------------------------------------------------------------------------------------------
// The watched values this seam consumes (A10-01's consumer side)
// -------------------------------------------------------------------------------------------
//
// Every reader here delegates to [`crate::sync::watched::Watched`], the DataWatcher
// interpretation the A10-01 lane produced in parallel with this one. Nothing in this file
// indexes a metadata slot by number: the whole point of that module is that the index-to-meaning
// mapping lives in exactly one place.

/// java: `Entity.isBurning`.
#[must_use]
pub fn is_burning(world: &World, tracked: &TrackedEntity) -> bool {
    tracked
        .with_watched(world, |w| {
            use mc_data::EntityType as E;
            match tracked.kind.entity_type() {
                // EntityBlaze overrides Entity.isBurning rather than extending it: its private
                // attack-state byte is the whole answer, even if the shared flag disagrees.
                Some(E::Blaze) => w.is_blaze_burning(),
                // Entity.isBurning begins with `!isImmuneToFire`. These five inherit that method;
                // LavaSlime additionally hard-overrides it to false. Blaze is deliberately not in
                // this arm because its override above is the visible fireball-charge cue.
                Some(E::Ghast | E::PigZombie | E::LavaSlime | E::EnderDragon | E::WitherBoss) => {
                    false
                }
                _ => w.is_burning(),
            }
        })
        .unwrap_or(false)
}

/// java: `Entity.isInvisible`.
#[must_use]
pub fn is_invisible(world: &World, tracked: &TrackedEntity) -> bool {
    tracked
        .with_watched(world, |w| w.is_invisible())
        .unwrap_or(false)
}

/// The entity's custom name.
///
/// java: `Entity.getCustomNameTag`. Two renderers switch on it rather than on a watched number:
/// `RenderRabbit.getEntityTexture` (`:29`) checks for `"Toast"`, and
/// `RendererLivingEntity.renderModel` (`:433`) checks for `"Dinnerbone"`/`"Grumm"`.
///
/// Owned rather than borrowed because the `Metadata` component lives behind the ECS's borrow
/// guard, which `with_watched` closes when it returns.
#[must_use]
pub fn custom_name(world: &World, tracked: &TrackedEntity) -> Option<String> {
    tracked.with_watched(world, |w| w.custom_name().map(str::to_owned))?
}

/// The display name spelling used by vanilla's renderer easter eggs.
///
/// java: `EnumChatFormatting.getTextWithoutFormattingCodes` removes only a section sign followed
/// by one of `[0-9A-FK-OR]`, case-insensitively. Invalid and dangling section signs remain part of
/// the name, and neither case nor whitespace is otherwise normalized.
#[must_use]
pub fn unformatted_custom_name(world: &World, tracked: &TrackedEntity) -> Option<String> {
    custom_name(world, tracked).map(|name| strip_formatting_codes(&name))
}

/// The entry of `names` that [`unformatted_custom_name`] equals, compared in place without the
/// two `String` copies — the per-frame easter-egg tests only ever ask "is it exactly X".
#[must_use]
pub fn unformatted_custom_name_among(
    world: &World,
    tracked: &TrackedEntity,
    names: &[&'static str],
) -> Option<&'static str> {
    tracked.with_watched(world, |w| {
        let raw = w.custom_name()?;
        names
            .iter()
            .copied()
            .find(|name| formatting_stripped(raw).eq(name.chars()))
    })?
}

fn strip_formatting_codes(input: &str) -> String {
    formatting_stripped(input).collect()
}

/// `strip_formatting_codes` as a borrowing iterator, so comparisons need not allocate.
fn formatting_stripped(input: &str) -> impl Iterator<Item = char> + '_ {
    let mut chars = input.chars().peekable();
    core::iter::from_fn(move || loop {
        let ch = chars.next()?;
        if ch == '§' {
            if let Some(&code) = chars.peek() {
                if matches!(code.to_ascii_lowercase(), '0'..='9' | 'a'..='f' | 'k'..='o' | 'r') {
                    chars.next();
                    continue;
                }
            }
        }
        return Some(ch);
    })
}

/// java: `RenderPlayer.setModelVisibilities`' `EnumPlayerModelParts` mask (watched index 10).
///
/// The byte is returned exactly, including zero: vanilla registers zero as the watched default,
/// and zero is also a valid client setting meaning that every optional overlay is hidden.
#[must_use]
pub fn skin_parts_mask(world: &World, tracked: &TrackedEntity) -> i8 {
    tracked
        .with_watched(world, |w| w.skin_parts().bits())
        .unwrap_or(0)
}

/// The texture variant [`mc_render::entity::manager::texture_for`] switches on.
///
/// [`Watched::variant`] covers the six species whose variant is a number on the wire. The four
/// remaining `getEntityTexture` switches are *booleans* — `RenderGhast:20` (`isAttacking`),
/// `RenderGuardian:129` (`isElder`), `RenderZombie:59` (`isVillager`) and `RenderWither:26`
/// (`getInvulTime() > 0`) — and they map onto the same `variant` slot as `1`, which is what the
/// four arms this pass added to `texture_for` expect.
#[must_use]
pub fn render_variant(world: &World, tracked: &TrackedEntity) -> i32 {
    tracked
        .with_watched(world, |w| match tracked.kind.entity_type() {
            Some(mc_data::EntityType::Ghast) => i32::from(w.is_ghast_attacking()),
            Some(mc_data::EntityType::Guardian) => i32::from(w.is_elder_guardian()),
            Some(mc_data::EntityType::Zombie) => i32::from(w.is_zombie_villager()),
            _ => w.variant(),
        })
        .unwrap_or(0)
}

/// java: `LayerWolfCollar:30` — `EntitySheep.getDyeRgb(wolf.getCollarColor())`, watched index 20.
#[must_use]
pub fn collar_dye(world: &World, tracked: &TrackedEntity) -> Option<[f32; 4]> {
    let rgb = tracked.with_watched(world, |w| w.collar_colour().map(DyeColour::fleece_rgb))??;
    Some([rgb[0], rgb[1], rgb[2], 1.0])
}

/// The enderman's carried block state.
///
/// java: `EntityEnderman.getHeldBlockState` reads watched index 16 as an unsigned Java state id:
/// `block_id | metadata << 12`. Index 17 is registered but never read or written. The slightly
/// stale function name is retained because this is the existing layer-context API, but the value
/// deliberately remains a full [`mc_data::BlockState`] all the way to the block-model lookup.
#[must_use]
pub fn carried_block_id(world: &World, tracked: &TrackedEntity) -> Option<mc_data::BlockState> {
    tracked.with_watched(world, |w| {
        w.carried_block()
            .map(|(id, meta)| mc_data::BlockState::new(mc_data::BlockId(id as u16), meta as u8))
    })?
}

/// Whether an iron golem is currently offering a poppy. **Always `false` today** — see below.
///
/// java: `LayerIronGolemFlower.doRenderLayer` (`:24`) gates on
/// `EntityIronGolem.getHoldRoseTick() != 0`. `holdRoseTick` is a plain `private int`
/// (`EntityIronGolem.java:43`) that is **not on the DataWatcher at all**: it is latched to `400`
/// by `handleStatusUpdate((byte) 11)` (`:190`-`:199`) and decremented in `onLivingUpdate`
/// (`:128`-`:130`). So the poppy is a 20-second timer driven by an entity-status packet, exactly
/// like the attack swing that opcode `4` already drives.
///
/// This function first shipped reading **watched index 16, bit 0**, which is
/// `isPlayerCreated()` (`:273`) — a different fact entirely, read only by `canAttackClass`
/// (`:153`) and the village-reputation branch (`:297`), with no render effect in vanilla. GAP
/// `A02-01`'s write-up named only the arm pitch as the blocker and did not mention the gate,
/// and the index-16 bit was assumed by analogy with the other mobs that do key off it.
///
/// The consequence was visible and wrong in the other direction: **every player-built iron golem
/// held a poppy permanently**, since `isPlayerCreated` is set once and never cleared. Returning
/// `false` restores vanilla's far more common case (a golem holds a poppy only in the seconds
/// after it offers one to a villager) until the timer exists.
///
/// **Done (wave 1).** `EntityAnimState::hold_rose_ticks` is latched to `400` by
/// `S19PacketEntityStatus` opcode `11` beside the opcode-`4` `attack_timer` in
/// [`crate::sync::entities`] and decremented once per tick by `advance_render_state`, so the
/// gate below is now vanilla's own `getHoldRoseTick() != 0` rather than a stub. The `world`
/// argument is kept because the signature is part of the layer-context contract in `driver.rs`
/// and a golem's poppy needs nothing from the world.
#[must_use]
pub fn is_holding_rose(_world: &World, tracked: &TrackedEntity) -> bool {
    // java: `LayerIronGolemFlower.doRenderLayer` (`:24`) —
    // `if (entity.getHoldRoseTick() != 0)`.
    tracked.render.entity.hold_rose_ticks != 0
}

// -------------------------------------------------------------------------------------------
// Is the body drawn at all
// -------------------------------------------------------------------------------------------

/// What `RendererLivingEntity.renderModel` does with the body this frame.
///
/// java: `renderModel` (`:246-278`) — `flag = !isInvisible()`, `flag1 = !flag &&
/// !isInvisibleToPlayer(thePlayer)`. Three outcomes, and the middle one is the interesting one:
/// an invisible entity a *teammate* can still see is drawn as a 15%-alpha ghost with depth
/// writing off, not skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyDraw {
    /// Not drawn at all. java: both `flag` and `flag1` false.
    Skip,
    /// The ordinary opaque draw.
    Solid,
    /// `color(1, 1, 1, 0.15)`, `depthMask(false)`, `blendFunc(770, 771)`,
    /// `alphaFunc(516, 0.003921569)`.
    Ghost,
}

impl BodyDraw {
    /// java: `renderModel`'s two-flag fork.
    ///
    /// `visible_to_viewer` is `!isInvisibleToPlayer(thePlayer)` — vanilla's team/spectator rule.
    /// This client has no per-viewer invisibility exemption to feed it (`EntityLivingBase
    /// .isInvisibleToPlayer` consults the scoreboard team's `getSeeFriendlyInvisibles`), so the
    /// caller passes `false` and an invisible mob simply vanishes, which is the majority case and
    /// the one that matters for gameplay.
    #[must_use]
    pub fn decide(invisible: bool, visible_to_viewer: bool) -> Self {
        if !invisible {
            Self::Solid
        } else if visible_to_viewer {
            Self::Ghost
        } else {
            Self::Skip
        }
    }

    /// The tint this draw applies to the body's vertices.
    #[must_use]
    pub fn tint(self) -> [f32; 4] {
        match self {
            // java: `GlStateManager.color(1.0F, 1.0F, 1.0F, 0.15F)`.
            Self::Ghost => [1.0, 1.0, 1.0, 0.15],
            _ => EntityVertex::WHITE,
        }
    }

    /// Which of [`mc_render::entity::LayerBlend`]'s pipelines draws it.
    #[must_use]
    pub fn blend(self) -> mc_render::entity::LayerBlend {
        match self {
            // java: `renderModel`'s ghost keeps its own `alphaFunc(516, 0.003921569)` (H7).
            Self::Ghost => mc_render::entity::LayerBlend::Ghost,
            _ => mc_render::entity::LayerBlend::Opaque,
        }
    }
}

// -------------------------------------------------------------------------------------------
// The tints
// -------------------------------------------------------------------------------------------

/// java: `EntitySheep.DYE_TO_RGB` (`EntitySheep.java:372-387`), the sixteen fleece/collar colours.
///
/// The index is `EnumDyeColor.getMetadata()`, i.e. the low nibble of the sheep's watched byte 16,
/// which is exactly what `sync::entities::decode_entity_state` already puts in `variant`.
/// `LayerWolfCollar` uses the *same* table with the wolf's collar colour.
const DYE_TO_RGB: [[f32; 3]; 16] = [
    [1.0, 1.0, 1.0],   // WHITE
    [0.85, 0.5, 0.2],  // ORANGE
    [0.7, 0.3, 0.85],  // MAGENTA
    [0.4, 0.6, 0.85],  // LIGHT_BLUE
    [0.9, 0.9, 0.2],   // YELLOW
    [0.5, 0.8, 0.1],   // LIME
    [0.95, 0.5, 0.65], // PINK
    [0.3, 0.3, 0.3],   // GRAY
    [0.6, 0.6, 0.6],   // SILVER
    [0.3, 0.5, 0.6],   // CYAN
    [0.5, 0.25, 0.7],  // PURPLE
    [0.2, 0.3, 0.7],   // BLUE
    [0.4, 0.3, 0.2],   // BROWN
    [0.4, 0.5, 0.2],   // GREEN
    [0.6, 0.2, 0.2],   // RED
    [0.1, 0.1, 0.1],   // BLACK
];

/// java: `EntitySheep.getDyeRgb`. Out-of-range indices take white, which is what a `null` lookup
/// would have coloured with (vanilla's `GlStateManager.color` would have been left at the
/// identity).
#[must_use]
pub fn dye_rgb(index: i32) -> [f32; 4] {
    let rgb = DYE_TO_RGB
        .get(usize::try_from(index).unwrap_or(usize::MAX))
        .copied()
        .unwrap_or([1.0, 1.0, 1.0]);
    [rgb[0], rgb[1], rgb[2], 1.0]
}

/// The animated fleece colour of a sheep named exactly `jeb_`.
///
/// java: `LayerSheepWool.doRenderLayer` (`1.8.9`). The name comparison is deliberately raw and
/// case-sensitive: `EntitySheep.hasCustomName() && "jeb_".equals(getCustomNameTag())` does not
/// call `EnumChatFormatting.getTextWithoutFormattingCodes`, so a styled `"§ajeb_"` or a name
/// with a reset suffix remains an ordinary dyed sheep. `ticks_existed` and `entity_id` retain
/// Java's signed-`int` arithmetic, including its overflow, before the metadata lookup.
///
/// `None` says to retain the normal watched fleece colour. The caller still owns the wool
/// layer's sheared/invisible gate; vanilla checks that before it chooses either colour path.
#[must_use]
pub fn jeb_sheep_dye(
    custom_name: Option<&str>,
    entity: &mc_render::entity::EntityAnimState,
    partial_ticks: f32,
) -> Option<[f32; 4]> {
    if custom_name != Some("jeb_") {
        return None;
    }

    // `Entity.ticksExisted` and `Entity.getEntityId()` are Java `int`s. The tracker stores the
    // monotonically incremented tick counter as `u32`, so reinterpret it rather than saturating
    // at the signed boundary.
    let ticks = i32::from_ne_bytes(entity.ticks_existed.to_ne_bytes());
    let phase = (ticks / 25).wrapping_add(entity.entity_id);
    let current = phase % 16; // `EnumDyeColor.values().length`
    let next = phase.wrapping_add(1) % 16;
    let progress = ((ticks % 25) as f32 + partial_ticks) / 25.0;
    let a = dye_rgb(current);
    let b = dye_rgb(next);
    // `LayerSheepWool` passes only the three RGB components to `GlStateManager.color`; alpha is
    // not part of its interpolation and stays at the default `1.0`.
    Some([
        a[0] * (1.0 - progress) + b[0] * progress,
        a[1] * (1.0 - progress) + b[1] * progress,
        a[2] * (1.0 - progress) + b[2] * progress,
        1.0,
    ])
}

/// java: `RendererLivingEntity.setBrightness` (`:284-334`) — the red damage flash.
///
/// The condition is `hurtTime > 0 || deathTime > 0` and the colour it writes into the
/// `brightnessBuffer` is literally `(1.0, 0.0, 0.0, 0.3)`. That buffer is a `GL_CONSTANT`
/// interpolated against the texel on the lightmap unit rather than a plain modulate, so a hurt
/// mob goes *toward* red rather than becoming pure red — the 0.3 is the interpolation weight.
/// The renderer selects a textured in-draw `LivingHurt` material from this gate; multiplying
/// `(1, 0.7, 0.7)` into the body would incorrectly erase blue texels instead of retaining 70%.
#[must_use]
pub fn hurt_flash(hurt_time: u32, death_time: u32) -> Option<[f32; 4]> {
    if hurt_time == 0 && death_time == 0 {
        return None;
    }
    Some([1.0, 0.0, 0.0, 0.3])
}

/// java: `RenderCreeper.getColorMultiplier` (`:39-52`) — the white flash before the explosion.
///
/// `flash` is `EntityCreeper.getCreeperFlashIntensity(partialTicks)`, the fuse ramp in `0..=1`.
/// The strobe is `(int)(f * 10) % 2 == 0 -> no flash`, so it blinks five times a second, and the
/// alpha byte is `clamp((int)(f * 0.2 * 255), 0, 255)` over an RGB of `0xFFFFFF`.
///
/// Vanilla feeds that ARGB through the same `GL_INTERPOLATE` combiner as the hurt flash, with
/// alpha as the weight — so this returns the interpolated multiplier directly, and `None` when
/// the strobe is in an off frame.
#[must_use]
pub fn creeper_flash(flash: f32) -> Option<[f32; 4]> {
    if (flash * 10.0) as i32 % 2 == 0 {
        return None;
    }
    let weight = ((flash * 0.2 * 255.0) as i32).clamp(0, 255) as f32 / 255.0;
    if weight <= 0.0 {
        return None;
    }
    // The constant colour is 0xFFFFFF. The live WhiteFlash replay consumes this alpha with
    // ordinary source-alpha blending; multiplying an ordinary body vertex by white is an
    // identity and cannot represent fixed-function GL_INTERPOLATE.
    Some([1.0, 1.0, 1.0, weight])
}

/// The RGB multiplier `RenderWolf.doRender` installs while the wolf is wet.
///
/// java: `EntityWolf.getShadingWhileWet(partialTicks)` (`:292-295`) —
/// `0.75F + interpolatedTimeWolfIsShaking / 2.0F * 0.25F`. The world brightness term from
/// `RenderWolf` is deliberately not part of this helper: mc-rs submits the same packed light
/// to the body and its layers, so multiplying that brightness into vertex RGB would apply the
/// lightmap twice. A dry wolf keeps the identity multiplier, exactly as vanilla's outer `if
/// (entity.isWolfWet())` leaves the primary colour alone.
#[must_use]
pub fn wolf_wet_shading(is_wet: bool, interpolated_shake_time: f32) -> f32 {
    if is_wet {
        0.75 + interpolated_shake_time / 2.0 * 0.25
    } else {
        1.0
    }
}

/// Everything that multiplies an entity's colour this frame, resolved once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tints {
    /// Applied to the body and to every layer that does not have a colour of its own.
    pub body: [f32; 4],
    /// The sheep's fleece / wolf's collar dye, if the entity has one.
    pub dye: Option<[f32; 4]>,
}

impl Tints {
    /// Resolves the whole-entity colour from the state the tracker already keeps.
    ///
    /// java: `renderModel` applies the invisibility alpha. Hurt/death and creeper brightness are
    /// separate affine replays in the live renderer, not multiplicative body colours.
    #[must_use]
    pub fn resolve(
        kind: Option<mc_data::EntityType>,
        draw: BodyDraw,
        hurt_time: u32,
        death_time: u32,
        state: &mc_render::entity::EntityAnimState,
        _partial_ticks: f32,
    ) -> Self {
        let body = draw.tint();
        let _ = (hurt_time, death_time);
        let dye = match kind {
            // java: `LayerSheepWool:31` — `EntitySheep.getDyeRgb(sheep.getFleeceColor())`.
            // `decode_entity_state` puts the fleece nibble in `variant`.
            Some(mc_data::EntityType::Sheep) => Some(dye_rgb(state.variant)),
            // java: `LayerWolfCollar:30` — `getDyeRgb(wolf.getCollarColor())`, watched index 20.
            // `EntityWolf.entityInit` writes RED's *metadata* (14) into the dye-damage slot;
            // `getCollarColor` reads it through `byDyeDamage`, where 14 is ORANGE (metadata 1).
            Some(mc_data::EntityType::Wolf) => {
                Some(dye_rgb(DyeColour::from_dye_damage(14).metadata()))
            }
            _ => None,
        };
        Self { body, dye }
    }

    /// Adds `RenderWolf`'s wet body shading without touching the collar/layer dye or body alpha.
    ///
    /// The collar is a separate Java layer replay. Its geometry is emitted from the untinted
    /// body batch and receives `ctx.dye` later in the live driver, so this operation must remain
    /// body-only. Keeping alpha intact also preserves the invisible-ghost multiplier while the
    /// textured hurt/death material remains the owner of its affine red flash.
    #[must_use]
    pub fn with_wolf_wet(self, is_wet: bool, interpolated_shake_time: f32) -> Self {
        let shading = wolf_wet_shading(is_wet, interpolated_shake_time);
        if shading == 1.0 {
            return self;
        }

        let mut body = self.body;
        body[0] *= shading;
        body[1] *= shading;
        body[2] *= shading;
        Self { body, ..self }
    }
}

// -------------------------------------------------------------------------------------------
// The fire overlay
// -------------------------------------------------------------------------------------------

/// The two atlas sprites `renderEntityOnFire` alternates between.
pub const FIRE_SPRITES: [&str; 2] = [
    "minecraft:blocks/fire_layer_0",
    "minecraft:blocks/fire_layer_1",
];

/// Emits the camera-facing stack of quads drawn over a burning entity.
///
/// java: `Render.renderEntityOnFire` (`Render.java:105-158`). The transform is
/// `translate(x, y, z)`, `scale(f, f, f)` where `f = width * 1.4`, `rotate(-playerViewY, Y)`, then
/// `translate(0, 0, -0.3 + (int)(height / f) * 0.02)`; the loop then walks `f3 = height / f` down
/// in `0.45` steps, shrinking the half-width by `0.9x` and stepping `0.03` in Z per layer, and
/// mirrors U every second *pair* of layers.
///
/// `uv` is the atlas rectangle `(min_u, min_v, max_u, max_v)` for the sprite this layer wants;
/// the caller resolves it because the stitched atlas is `mc-assets`'. The geometry is atlas-UV'd,
/// so it must be pushed under the block-atlas texture key, not an entity texture.
///
/// `y_offset` is vanilla's `f4` seed, `posY - boundingBox.minY` — zero for every entity whose
/// origin is its feet, which is all of them in 1.8.9 except the ones this client does not draw.
pub fn emit_fire_overlay(
    rel: Vec3,
    width: f32,
    height: f32,
    view_yaw: f32,
    mut sprite_uv: impl FnMut(usize) -> [f32; 4],
    out: &mut Vec<EntityVertex>,
) {
    let f = width * 1.4;
    if f <= 0.0 {
        return;
    }
    let mut f3 = height / f;
    // java: `translate(0, 0, -0.3 + (int)f3 * 0.02)` — the integer truncation is vanilla's, and it
    // is what keeps a tall mob's flames from z-fighting with a short one's.
    let z_bias = f3 as i32 as f32 * 0.02 - 0.3;
    let base = Mat4::from_translation(rel)
        * Mat4::from_scale(Vec3::splat(f))
        * Mat4::from_rotation_y((-view_yaw).to_radians())
        * Mat4::from_translation(Vec3::new(0.0, 0.0, z_bias));

    let mut half_width = 0.5_f32;
    let mut f4 = 0.0_f32;
    let mut z = 0.0_f32;
    let mut i = 0_usize;
    while f3 > 0.0 {
        let [min_u, min_v, max_u, max_v] = sprite_uv(i % 2);
        // java: `if (i / 2 % 2 == 0) { swap(minU, maxU); }` — the flames mirror every other pair.
        let (u_left, u_right) = if (i / 2).is_multiple_of(2) {
            (max_u, min_u)
        } else {
            (min_u, max_u)
        };
        // java's quad winding is (+x, bottom), (-x, bottom), (-x, top), (+x, top).
        let quad = [
            ([half_width, -f4, z], [u_right, max_v]),
            ([-half_width, -f4, z], [u_left, max_v]),
            ([-half_width, 1.4 - f4, z], [u_left, min_v]),
            ([half_width, 1.4 - f4, z], [u_right, min_v]),
        ];
        for &idx in &[0usize, 1, 2, 0, 2, 3] {
            let (pos, uv) = quad[idx];
            out.push(EntityVertex {
                pos: base.transform_point3(Vec3::from(pos)).into(),
                uv,
                // java: `disableLighting()` for the whole overlay. The additive pipeline is
                // unlit, so the normal is only along for the ride; +Z keeps it well-defined.
                normal: [0.0, 0.0, 1.0],
                color: EntityVertex::WHITE,
            });
        }
        f3 -= 0.45;
        f4 -= 0.45;
        half_width *= 0.9;
        z += 0.03;
        i += 1;
        // Vanilla's loop is bounded by the entity's height; a corrupt height must not hang a
        // frame. 64 layers is 28 blocks of flame, well past anything 1.8.9 can spawn.
        if i >= 64 {
            break;
        }
    }
}

// -------------------------------------------------------------------------------------------
// The contact shadow
// -------------------------------------------------------------------------------------------

/// java: `Render.shadowTextures`.
pub const SHADOW_TEXTURE: &str = "textures/misc/shadow.png";

/// java: `Render.shadowOpaque`, which is `1.0` for every renderer except `RenderEntityItem` and
/// `RenderXPOrb` (`0.75`).
pub const SHADOW_OPAQUE: f32 = 1.0;

/// The dynamic light inputs that `Render.renderShadowBlock` gets through its `World` query.
///
/// Terrain uploads raw sky and block-light separately and lets the lightmap apply day/night
/// colour later. Entity shadows are different: `World.getLightBrightness(pos)` reads
/// `getLight(pos, true)`, so it first subtracts `skylightSubtracted` from *only* the sky
/// channel, then compares that result with block light. Keeping those inputs together avoids
/// darkening torches at night or leaving moonlit shadows at noon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowLighting {
    /// java: `World.skylightSubtracted`, normally `0..=11`.
    pub skylight_subtracted: u8,
    /// java: `WorldProvider.generateLightBrightnessTable`'s ambient floor: `0.0` in the
    /// Overworld/End and `0.1` in the Nether.
    pub ambient_light: f32,
}

impl ShadowLighting {
    /// Builds the render-frame snapshot from the same `Environment` that drives the lightmap.
    #[must_use]
    pub fn from_environment(environment: mc_render::sky::Environment, partial_ticks: f32) -> Self {
        Self {
            skylight_subtracted: environment.skylight_subtracted(partial_ticks),
            ambient_light: environment.dimension.ambient_light(),
        }
    }
}

impl Default for ShadowLighting {
    fn default() -> Self {
        Self {
            skylight_subtracted: 0,
            ambient_light: 0.0,
        }
    }
}

/// Calculates one projected shadow quad's alpha, if vanilla would emit that quad at all.
///
/// java: `Render.renderShadowBlock` (`1.8.9`, `:224-240`): its `getLightFromNeighbors(pos) >
/// 3` gate and its subsequent `getLightBrightness(pos)` both use `World.getLight(pos, true)`.
/// That query subtracts [`ShadowLighting::skylight_subtracted`] from sky light before it takes
/// the maximum with block light; `World.getLightBrightness` then indexes the provider's
/// `lightBrightnessTable`, rather than linearly dividing the winning nibble by 15.
#[must_use]
pub fn shadow_block_alpha(
    shadow_alpha: f32,
    entity_y: f64,
    block_y: i32,
    sky_light: u8,
    block_light: u8,
    lighting: ShadowLighting,
) -> Option<f32> {
    let level = sky_light
        .saturating_sub(lighting.skylight_subtracted)
        .max(block_light)
        .min(15);
    if level <= 3 {
        return None;
    }
    let brightness = mc_render::sky::lightmap::light_brightness_table(lighting.ambient_light)
        [usize::from(level)];
    let alpha = (f64::from(shadow_alpha) - (entity_y - f64::from(block_y)) / 2.0)
        * 0.5
        * f64::from(brightness);
    // `renderShadow` only supplies `0..=1` and its loop never visits a block above the entity,
    // so the production result is already at most 0.5. Do not add a defensive clamp here:
    // `renderShadowBlock` passes this float straight to `color(1, 1, 1, alpha)`.
    (alpha >= 0.0).then_some(alpha as f32)
}

/// Emits an entity's projected contact shadow.
///
/// java: `Render.doRenderShadowAndFire` (`:299`) gates on `options.entityShadows && shadowSize >
/// 0 && !isInvisible()` and computes `alpha = (1 - distance/256) * shadowOpaque`; `renderShadow`
/// (`:164`) then walks the block box `floor(pos +/- f)` on XZ and one block down, and
/// `renderShadowBlock` (`:224`) emits one quad per full cube whose per-block alpha is
/// `(alpha - (y - blockY)/2) * 0.5 * getLightBrightness(pos)`, with UVs that project the entity's
/// footprint onto that block's top face.
///
/// `shadow_size` is the descriptor's, already multiplied by `getRenderSizeModifier()` and halved
/// for a baby by the caller. Returns the number of quads emitted, so a test can assert a shadow
/// exists without a pixel readback.
///
/// The `+ 0.015625` on the quad's Y is vanilla's own z-fight guard (`renderShadowBlock:239`), a
/// quarter of a pixel above the block's top face.
#[allow(clippy::too_many_arguments)]
pub fn emit_shadow(
    world: &World,
    camera: &Camera,
    position: DVec3,
    shadow_size: f32,
    alpha: f32,
    out: &mut Vec<EntityVertex>,
) -> usize {
    emit_shadow_lit(
        world,
        camera,
        position,
        shadow_size,
        alpha,
        ShadowLighting::default(),
        out,
    )
}

/// [`emit_shadow`] with the frame's Java `World.getLightBrightness` inputs.
///
/// The compatibility wrapper above uses noon Overworld lighting. Production callers should use
/// this entry point with [`ShadowLighting::from_environment`].
#[allow(clippy::too_many_arguments)]
pub fn emit_shadow_lit(
    world: &World,
    camera: &Camera,
    position: DVec3,
    shadow_size: f32,
    alpha: f32,
    lighting: ShadowLighting,
    out: &mut Vec<EntityVertex>,
) -> usize {
    if shadow_size <= 0.0 || alpha <= 0.0 {
        return 0;
    }
    let f = f64::from(shadow_size);
    let (x0, x1) = (
        (position.x - f).floor() as i32,
        (position.x + f).floor() as i32,
    );
    let (y0, y1) = ((position.y - f).floor() as i32, position.y.floor() as i32);
    let (z0, z1) = (
        (position.z - f).floor() as i32,
        (position.z + f).floor() as i32,
    );
    let mut quads = 0;

    for bx in x0..=x1 {
        for by in y0..=y1 {
            for bz in z0..=z1 {
                let pos = BlockPos::new(bx, by, bz);
                let below = BlockPos::new(bx, by - 1, bz);
                let state = world.block_at(below);
                let props = mc_data::block_props(state);
                // java: `block.getRenderType() != -1 && getLightFromNeighbors(pos) > 3`, then
                // `blockIn.isFullCube()` inside `renderShadowBlock`. `RenderType::None` is the
                // `-1`.
                if props.render_type == mc_data::RenderType::Invisible || !props.full_cube {
                    continue;
                }
                let Some(d0) = shadow_block_alpha(
                    alpha,
                    position.y,
                    by,
                    world.sky_light_at(pos),
                    world.block_light_at(pos),
                    lighting,
                ) else {
                    continue;
                };
                // java: `World.getLightBrightness` — the lightmap ramp, `lightBrightnessTable`.

                let (min_x, max_x) = (f64::from(bx), f64::from(bx) + 1.0);
                let (min_z, max_z) = (f64::from(bz), f64::from(bz) + 1.0);
                // java: `Render.renderShadowBlock:241` — `pos.getY() + blockIn.getBlockBoundsMinY()
                // + 0.015625`, where `pos` is the loop's `blockpos` (`:206` passes `blockpos`, not
                // `blockpos.down()`) and `blockIn` is the block *below* it. For a full cube
                // `getBlockBoundsMinY()` is `0.0`, so the quad sits on `blockpos`'s own floor —
                // which is the top face of the solid block underneath. An extra `+ 1.0` here put
                // every shadow a whole block above the ground it belongs to.
                let y = f64::from(by) + 0.015_625;
                // java: `f = (posX - d1) / 2 / f + 0.5` — the entity's footprint projected onto
                // this block's top face, in the shadow texture's 0..1 space.
                let u = |wx: f64| ((position.x - wx) / 2.0 / f + 0.5) as f32;
                let v = |wz: f64| ((position.z - wz) / 2.0 / f + 0.5) as f32;

                let corners = [
                    (min_x, min_z, u(min_x), v(min_z)),
                    (min_x, max_z, u(min_x), v(max_z)),
                    (max_x, max_z, u(max_x), v(max_z)),
                    (max_x, min_z, u(max_x), v(min_z)),
                ];
                for &idx in &[0usize, 1, 2, 0, 2, 3] {
                    let (wx, wz, tu, tv) = corners[idx];
                    let rel = camera.relative(DVec3::new(wx, y, wz));
                    out.push(EntityVertex {
                        pos: rel.into(),
                        uv: [tu, tv],
                        // Straight up: the shadow lies on a block's top face. With this normal
                        // the two-light diffuse term saturates to 1.0, so the shadow texture's
                        // own colour reaches the framebuffer unmodulated, which is what vanilla's
                        // untextured-lighting draw does.
                        normal: [0.0, 1.0, 0.0],
                        color: [1.0, 1.0, 1.0, d0],
                    });
                }
                quads += 1;
            }
        }
    }
    quads
}

/// java: `doRenderShadowAndFire` (`:306-307`) — `f = (1 - d0 / 256) * shadowOpaque`.
///
/// **`d0` is the squared distance.** `RenderManager.getDistanceToCamera` (`:478-484`) is named
/// as though it returns a distance and returns `d0*d0 + d1*d1 + d2*d2` with no `sqrt`, so the
/// 256 is 16 blocks squared: a vanilla 1.8.9 entity shadow is gone at 16 blocks and is already
/// down to 0.12 alpha at 15. This function used to take the `sqrt` first, which made the fade
/// reach 256 *blocks* — every entity in render distance kept a near-opaque shadow (0.94 at 16
/// blocks, where vanilla draws none at all).
#[must_use]
pub fn shadow_alpha(distance_sq: f64) -> f32 {
    ((1.0 - distance_sq / 256.0) * f64::from(SHADOW_OPAQUE)).max(0.0) as f32
}

// -------------------------------------------------------------------------------------------
// Skins
// -------------------------------------------------------------------------------------------

/// The decoded `textures` profile property.
///
/// java: `NetworkPlayerInfo.loadPlayerTextures` hands the property to
/// `SkinManager.loadProfileTextures`, which base64-decodes it into a
/// `MinecraftProfileTexture` map keyed `SKIN`/`CAPE`/`ELYTRA`; the `SKIN` entry's
/// `metadata.model` property is `"slim"` for an Alex-shaped account and absent otherwise.
///
/// The payload looks like:
/// ```json
/// {"timestamp":0,"profileId":"…","profileName":"…","textures":{
///   "SKIN":{"url":"http://textures.minecraft.net/texture/…","metadata":{"model":"slim"}},
///   "CAPE":{"url":"…"}}}
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProfileTextures {
    /// Where the skin PNG lives. `None` for an account that has never set one.
    pub skin_url: Option<String>,
    /// Where the cape PNG lives, for `LayerCape`.
    pub cape_url: Option<String>,
    /// Arm width, from `SKIN.metadata.model`.
    pub model: mc_render::entity::SkinModel,
}

/// Decodes a tab-list `textures` property.
///
/// Returns `None` for anything that is not valid base64, not the expected JSON shape, or that
/// names a texture outside authlib's allowed domains. A malformed property from a proxy must
/// leave the player as Steve, never fail a frame.
///
/// authlib 1.5.21's `YggdrasilMinecraftSessionService.getTextures` applies the domain check to
/// **every** entry in the decoded map and returns an empty map when any one fails. In particular,
/// a valid skin plus an off-domain cape is not a partially usable profile. The signature gate is
/// enforced earlier by [`crate::sync::PlayerInfo::textures`]; RSA validation of a nonempty
/// signature is deliberately kept at that security seam rather than making this parser guess.
#[must_use]
pub fn decode_textures_property(value: &str) -> Option<ProfileTextures> {
    use base64::Engine as _;

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(value.trim())
        .ok()?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let textures = json.get("textures")?;
    let skin = textures.get("SKIN");
    let decoded = ProfileTextures {
        skin_url: skin
            .and_then(|s| s.get("url"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        cape_url: textures
            .get("CAPE")
            .and_then(|c| c.get("url"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        // java: `MinecraftProfileTexture.getMetadata("model")`, compared against `"slim"` in
        // `AbstractClientPlayer.getSkinType`.
        model: skin
            .and_then(|s| s.get("metadata"))
            .and_then(|m| m.get("model"))
            .and_then(serde_json::Value::as_str)
            .map_or(mc_render::entity::SkinModel::Classic, |m| {
                mc_render::entity::SkinModel::parse(m)
            }),
    };
    // authlib rejects the *entire* map for one untrusted entry, not merely the URL the current
    // caller happens to use. Preserve that before either SKIN or CAPE reaches a downloader.
    // The downloader's own host rule applies here as well, so a texture
    // it would refuse leaves the profile on its default skin rather than a failed fetch.
    decoded
        .skin_url
        .iter()
        .chain(decoded.cape_url.iter())
        .all(|url| {
            is_authlib_texture_domain(url) && mc_render::entity::skin::is_mojang_texture_url(url)
        })
        .then_some(decoded)
}

/// authlib 1.5.21's `YggdrasilMinecraftSessionService.isWhitelistedDomain`.
///
/// The original constructs a `java.net.URI`, reads its host, then uses a case-sensitive
/// `endsWith` against `.minecraft.net` and `.mojang.com`. It intentionally permits all schemes
/// and ports; the host suffix, not HTTPS, is the security policy in this version. This small
/// authority parser accepts the URI forms that can carry a host and rejects malformed/opaque
/// values before reproducing that suffix comparison.
fn is_authlib_texture_domain(url: &str) -> bool {
    if url.is_empty()
        || !url.is_ascii()
        || url
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ')
    {
        return false;
    }
    let authority = if let Some(rest) = url.strip_prefix("//") {
        rest
    } else {
        let Some(colon) = url.find(':') else {
            return false;
        };
        let scheme = &url[..colon];
        if scheme.is_empty()
            || !scheme.bytes().enumerate().all(|(index, byte)| {
                byte.is_ascii_alphabetic() || (index != 0 && matches!(byte, b'+' | b'-' | b'.'))
            })
        {
            return false;
        }
        let Some(rest) = url[colon + 1..].strip_prefix("//") else {
            return false;
        };
        rest
    };
    let Some(authority) = authority
        .split(['/', '?', '#'])
        .next()
        .filter(|authority| !authority.is_empty())
    else {
        return false;
    };
    let host_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    // IP-literals cannot end in either Java whitelist suffix. Refuse them before colon handling.
    if host_port.starts_with('[') {
        return false;
    }
    let host = host_port
        .split_once(':')
        .map_or(host_port, |(host, _)| host);
    !host.is_empty() && (host.ends_with(".minecraft.net") || host.ends_with(".mojang.com"))
}

/// The [`mc_render::entity::EntityPhase`] texture key a fetched skin is uploaded under.
///
/// Namespaced away from resource paths so it can never collide with `textures/entity/steve.png`,
/// and keyed by the account uuid so two players wearing the same skin URL still get one upload
/// each — which is what vanilla's `SkinManager` cache does (it keys on the texture *hash*, but
/// the uuid is what the render path has in hand and the difference is one extra upload).
#[must_use]
pub fn skin_texture_key(uuid: uuid::Uuid) -> String {
    format!("mc-render/entity/skin/{uuid}")
}

/// Extract authlib's safe texture hash (the final URL path component).
///
/// java: `SkinManager.loadSkin` keys its `ResourceLocation` by
/// `MinecraftProfileTexture.getHash()`, which is the CDN URL basename. Restricting this to
/// filename-safe ASCII keeps a malformed profile from escaping the in-memory/disk key space.
/// One rule for every caller: the GUI's player-head icons use it from `mc-render` too.
fn texture_hash(url: &str) -> Option<&str> {
    mc_render::entity::skin::texture_url_hash(url)
}

/// The skin cache identity for a trusted texture URL, shared safely across account UUIDs.
#[must_use]
pub fn skin_texture_key_for_url(url: &str) -> Option<String> {
    mc_render::entity::skin::skin_key_for_texture_url(url)
}

/// The cape cache identity for a trusted texture URL, shared safely across account UUIDs while
/// remaining separate from the native-size cape cache's item kind.
#[must_use]
pub fn cape_texture_key_for_url(url: &str) -> Option<String> {
    texture_hash(url).map(|hash| format!("mc-render/entity/cape/{hash}"))
}

// -------------------------------------------------------------------------------------------
// The layers that needed a model-local seam
// -------------------------------------------------------------------------------------------

/// What a layer's dispatch needs beyond the entity itself.
///
/// Bundled rather than passed as four more arguments so `emit_entity_layer`'s signature stays
/// readable while five other authors are editing the same file.
pub struct LayerContext<'a> {
    /// For the watched values `decode_entity_state` reduces to a bool — the enderman's carried
    /// block id, the iron golem's rose bit.
    pub world: &'a World,
    /// The entity-state colour for the two layers that have one ([`Tints::dye`]).
    pub dye: Option<[f32; 4]>,
    /// The account name, for the layers that key off it (`LayerDeadmau5Head`).
    pub username: Option<&'a str>,
    /// Whether the player's trusted custom skin has completed loading. Java's Deadmau5 layer
    /// checks `AbstractClientPlayer.hasSkin()` and does not draw over the UUID fallback.
    pub skin_ready: bool,
}

/// A block-model lookup requested either by numeric block id or by a complete state.
///
/// The numeric form is used by layers with a fixed metadata-zero block (the golem's poppy, the
/// snowman's pumpkin and the mooshroom's mushroom); the state form is used by the enderman.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockModelInput {
    /// A fixed block id whose layer uses the metadata-zero default state.
    BlockId(i32),
    /// A complete state supplied by the entity, including its metadata.
    State(mc_data::BlockState),
}

impl From<i16> for BlockModelInput {
    fn from(value: i16) -> Self {
        Self::BlockId(i32::from(value))
    }
}

impl From<i32> for BlockModelInput {
    fn from(value: i32) -> Self {
        Self::BlockId(value)
    }
}

impl From<mc_data::BlockState> for BlockModelInput {
    fn from(value: mc_data::BlockState) -> Self {
        Self::State(value)
    }
}

fn block_model_state(block: BlockModelInput) -> Option<mc_data::BlockState> {
    let state = match block {
        BlockModelInput::BlockId(id) => {
            mc_data::BlockState::new(mc_data::BlockId(u16::try_from(id).ok()?), 0)
        }
        BlockModelInput::State(state) => state,
    };
    (mc_data::block_props(state).render_type == mc_data::RenderType::Model).then_some(state)
}

/// The baked block-state model for a layer that calls `renderBlockBrightness`.
///
/// java: `BlockRendererDispatcher.renderBlockBrightness(state, 1.0F)`, which `LayerHeldBlock`,
/// `LayerIronGolemFlower` and `LayerMooshroomMushroom` all use. That route resolves
/// `BlockModelShapes.getModelForState`; it does not use the inventory item model. Returns `None`
/// for non-model render types, matching `renderBlockBrightness`'s render-type gate.
#[must_use]
pub fn block_item_model(
    block: impl Into<BlockModelInput>,
) -> Option<&'static mc_assets::BakedModel> {
    let model = mc_assets::model_for(block_model_state(block.into())?);
    if model.face_quads.iter().all(Vec::is_empty) && model.general_quads.is_empty() {
        return None;
    }
    Some(model)
}

/// java: `LayerHeldItemWitch.doRenderLayer`'s four-way `if/else if/else if/else`.
#[must_use]
pub fn witch_pose(item: mc_data::ItemId) -> mc_render::entity::layers::WitchHeldItem {
    use mc_render::entity::layers::WitchHeldItem as W;
    let props = mc_data::item_props(item);
    // java: `itemstack.getItem() instanceof ItemBlock && Block.getBlockFromItem(...)
    // .getRenderType() == 2`... but the witch layer's first test is `isRenderTypeChest`, which in
    // 1.8.9 is `getRenderType() == 2 && getBlockLayer() == CUTOUT`-adjacent; the only blocks it is
    // true for are the three chests. Matched by name, since `mc-data` models the render type but
    // not `isRenderTypeChest`.
    if matches!(
        props.name,
        "minecraft:chest" | "minecraft:trapped_chest" | "minecraft:ender_chest"
    ) {
        return W::Chest;
    }
    // java: `Items.bow`.
    if props.name == "minecraft:bow" {
        return W::Bow;
    }
    // java: `Item.isFull3D()` — swords, tools and rods, i.e. everything vanilla holds like a
    // stick. `shouldRotateAroundWhenRendering` is true only for the fishing rod and the carrot on
    // a stick.
    if props.full_3d {
        return W::Full3D {
            rotate_around: matches!(
                props.name,
                "minecraft:fishing_rod" | "minecraft:carrot_on_a_stick"
            ),
        };
    }
    W::Flat
}

/// Whether `LayerCustomHead` takes its villager-like child branch for this entity.
///
/// java: `LayerCustomHead.doRenderLayer` (`:47`) —
/// `entity instanceof EntityVillager || entity instanceof EntityZombie &&
/// ((EntityZombie)entity).isVillager()`. The first arm is a class test, not the child flag, and
/// the second arm reads `EntityZombie`'s watched index 13 (`isVillager()`, `== 1`). Keeping this
/// predicate beside the caller prevents an ordinary baby biped from accidentally inheriting the
/// villager helmet offsets.
#[must_use]
pub fn is_villager_like(world: &World, tracked: &TrackedEntity) -> bool {
    use mc_data::EntityType as E;

    match tracked.kind.entity_type() {
        Some(E::Villager) => true,
        Some(E::Zombie) => tracked
            .with_watched(world, |watched| watched.is_zombie_villager())
            .unwrap_or(false),
        _ => false,
    }
}

/// The production `LayerCustomHead` adapter, and whether it selected the skull branch.
///
/// java: `LayerCustomHead.doRenderLayer` (`:34-107`). The body has already been posed when this
/// is called, so the adapter borrows that same [`RenderManager`] model and hands its entity-space
/// `base` plus the named head part to [`mc_render::entity::layers::custom_head_matrix`]. This is
/// important: the sneak/child operations must be composed between `base` and `postRender`, not
/// pre-multiplied onto a matrix that already contains both (which applies the offsets in world
/// space for a rotated entity).
///
/// `params` is the exact layer argument source for `scale`, `isSneaking`, and `isChild`. The
/// villager-like gate is derived here from the entity kind and watched state, so a caller cannot
/// apply it to every child biped by accident. `None` means the item is neither an `ItemBlock` nor
/// `Items.skull`, matching vanilla's two mutually exclusive geometry branches.
#[must_use]
pub fn custom_head_matrix_for_entity(
    models: &mut mc_render::entity::RenderManager,
    world: &World,
    tracked: &TrackedEntity,
    model_kind: mc_render::entity::ModelKind,
    base: Mat4,
    params: &mc_render::entity::AnimParams,
    item: mc_data::ItemId,
) -> Option<(Mat4, bool)> {
    custom_head_matrix_for_model(
        models,
        model_kind,
        is_villager_like(world, tracked),
        base,
        params,
        item,
    )
}

/// [`custom_head_matrix_for_entity`] with the villager-like gate already decided.
///
/// The local player's F5 body has no [`TrackedEntity`]; it is an `EntityPlayerSP`, which java's
/// `LayerCustomHead` (`:47`) never treats as villager-like, so its caller passes `false`.
#[must_use]
pub fn custom_head_matrix_for_model(
    models: &mut mc_render::entity::RenderManager,
    model_kind: mc_render::entity::ModelKind,
    villager_like: bool,
    base: Mat4,
    params: &mc_render::entity::AnimParams,
    item: mc_data::ItemId,
) -> Option<(Mat4, bool)> {
    use mc_render::entity::layers::{custom_head_matrix as render_head_matrix, CustomHeadItem};

    let props = mc_data::item_props(item);
    // Keep Java's branch order: ItemBlock is tested before Items.skull. In 1.8.9 the skull item
    // is not an ItemBlock, but preserving the order makes this adapter faithful if registry data
    // ever exposes both properties for one item.
    let (item_kind, skull) = if props.block.is_some() {
        (CustomHeadItem::Block, false)
    } else if props.name == "minecraft:skull" {
        (CustomHeadItem::Skull, true)
    } else {
        return None;
    };

    let model = models.model_mut(model_kind).model();
    let head_part = model.find("head")?;
    let matrix = render_head_matrix(
        base,
        model,
        head_part,
        params.scale,
        params.entity.sneaking,
        params.entity.child,
        villager_like,
        item_kind,
    );
    Some((matrix, skull))
}

/// java: `TileEntitySkullRenderer.SKELETON_TEXTURES` — the five skull sheets, indexed by the
/// stack's damage value.
#[must_use]
pub fn skull_texture(damage: i16) -> &'static str {
    match damage {
        1 => "textures/entity/skeleton/wither_skeleton.png",
        2 => "textures/entity/zombie/zombie.png",
        3 => "textures/entity/steve.png",
        4 => "textures/entity/creeper/creeper.png",
        _ => "textures/entity/skeleton/skeleton.png",
    }
}

/// java: `LayerSnowmanHead.doRenderLayer` — `translate(0, -0.34375, 0)`, `rotate(180, Y)`,
/// `scale(f, -f, -f)` with `f = 0.625`, over `head.postRender(0.0625F)`.
#[must_use]
pub fn snowman_head_matrix(head: Mat4) -> Mat4 {
    let f = 0.625;
    head * Mat4::from_translation(Vec3::new(0.0, -0.343_75, 0.0))
        * Mat4::from_rotation_y(std::f32::consts::PI)
        * Mat4::from_scale(Vec3::new(f, -f, -f))
}

/// Adapts a matrix that ends where vanilla calls `renderBlockBrightness` into one
/// [`mc_render::entity::item::emit_baked_model`] can use.
///
/// java: `BlockRendererDispatcher.renderBlockBrightness` (`:134`) forwards a render-type-3 block
/// to `BlockModelRenderer.renderModelBrightness` (`:310`), whose **first** statement is
/// `GlStateManager.rotate(90.0F, 0.0F, 1.0F, 0.0F)` and which then draws the baked quads in
/// their own `0..1` block space with no recentring at all. `emit_baked_model` instead applies
/// `translate(-0.5, -0.5, -0.5)` — the convention `RenderEntityItem` wants, since a dropped item
/// spins about its own middle. So the three layers that draw a *block* rather than an item need
/// the quarter turn put back and the recentring taken back out.
///
/// The three are `LayerHeldBlock` (the enderman's block), `LayerIronGolemFlower` (the poppy) and
/// `LayerMooshroomMushroom`. `LayerCustomHead` and `LayerSnowmanHead` go the *other* way — they
/// call `ItemRenderer.renderItem(..., TransformType.HEAD)`, whose `RenderItem.renderModel` does
/// its own `translate(-0.5, -0.5, -0.5)` and no rotate — so those two pair with
/// `emit_baked_model` unadapted.
#[must_use]
pub fn block_brightness_matrix(m: Mat4) -> Mat4 {
    m * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2)
        * Mat4::from_translation(Vec3::splat(0.5))
}

/// The three matrices `LayerMooshroomMushroom` draws `Blocks.red_mushroom` under.
///
/// java: `LayerMooshroomMushroom.doRenderLayer` (`:24-52`). Two hang off the *entity* matrix —
/// vanilla opens with `scale(1, -1, 1)`, `translate(0.2, 0.35, 0.5)`, `rotate(42, Y)` and then
/// pushes twice inside that — and the third hangs off the head part. Every one ends with
/// `translate(-0.5, -0.5, 0.5)` immediately before `renderBlockBrightness`. These matrices stop
/// at that dispatcher boundary; the block-brightness emitter owns the following quarter turn.
#[must_use]
pub fn mooshroom_mushroom_matrices(base: Mat4, head: Option<Mat4>) -> Vec<Mat4> {
    // java: the `translate(-0.5, -0.5, 0.5)` every one of the three shares.
    let shift = Mat4::from_translation(Vec3::new(-0.5, -0.5, 0.5));
    let body = base
        * Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
        * Mat4::from_translation(Vec3::new(0.2, 0.35, 0.5))
        * Mat4::from_rotation_y(42.0_f32.to_radians());
    let mut out = vec![
        body * shift,
        body * Mat4::from_translation(Vec3::new(0.1, 0.0, -0.6))
            * Mat4::from_rotation_y(42.0_f32.to_radians())
            * shift,
    ];
    if let Some(head) = head {
        out.push(
            head * Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
                * Mat4::from_translation(Vec3::new(0.0, 0.7, -0.2))
                * Mat4::from_rotation_y(12.0_f32.to_radians())
                * shift,
        );
    }
    out
}

/// java: `ItemArmor.getColor(stack)` — the dye stored in `display.color`, defaulting to
/// `0xA06540` for undyed leather, split into an RGB multiplier.
#[must_use]
pub fn leather_dye(nbt: Option<&mc_nbt::NbtCompound>) -> [f32; 4] {
    /// java: `ItemArmor.getColor`'s `return 10511680` when the tag is absent.
    const UNDYED: i32 = 10_511_680;
    let color = nbt
        .and_then(|tag| tag.get("display"))
        .and_then(|display| match display {
            mc_nbt::Nbt::Compound(c) => c.get("color"),
            _ => None,
        })
        .and_then(mc_nbt::Nbt::as_i32)
        .unwrap_or(UNDYED);
    [
        ((color >> 16) & 255) as f32 / 255.0,
        ((color >> 8) & 255) as f32 / 255.0,
        (color & 255) as f32 / 255.0,
        1.0,
    ]
}

/// Everything the appearance seam keeps between frames — one field on [`crate::app::Client`].
///
/// java: `Minecraft.skinManager` plus the `RenderManager`'s implicit "which textures are bound"
/// state. Held as one struct so the six concurrent Wave-1 authors each add exactly one field to
/// `Client`.
pub struct AppearanceState {
    /// java: `SkinManager`'s texture cache. Non-blocking by construction — see
    /// [`mc_render::entity::SkinCache`].
    pub skins: mc_render::entity::SkinCache,
    /// Native-size CAPE images. Kept beside skins but in a distinct cache so a profile's CAPE
    /// can never alias its 64x64 SKIN entry.
    pub capes: mc_render::entity::CapeCache,
    /// The worker half of the skin channel, taken exactly once when the runtime spawns it.
    channel: Option<mc_render::entity::skin::SkinChannel>,
    /// The CAPE worker half, taken exactly once when the runtime spawns it.
    cape_channel: Option<mc_render::entity::skin::CapeChannel>,
    /// Whether [`spawn_worker`](Self::spawn_worker) has run.
    worker_running: bool,
    /// Whether the download worker may start at all. Only benchmarks
    /// turn it off ([`crate::app::Client::set_skin_downloads`]): they must be offline and
    /// deterministic, so profile skins and capes stay the defaults, nothing is fetched and
    /// nothing is read from or written to the skin cache.
    downloads_enabled: bool,
    /// Skin keys already uploaded to the entity phase this session, so a resident skin costs one
    /// `HashSet` probe a frame rather than a texture re-upload.
    uploaded: std::collections::HashSet<String>,
    /// The profile payload that produced each player's current skin key.  Java puts the
    /// one-shot `playerTexturesLoaded` guard on `NetworkPlayerInfo`, so a replacement
    /// `ADD_PLAYER` gets a fresh profile even when its UUID is unchanged.  Keeping the payload
    /// beside the generated key gives this seam the same lifetime without needing to mutate the
    /// shared player-list state.
    player_skin_keys: HashMap<Uuid, (String, String)>,
    /// Monotonic suffix for profile replacements.  The key is deliberately generation based
    /// rather than a lossy hash: a new `textures` value can never alias a resident upload.
    next_skin_generation: u64,
    /// Monotonic suffix for profile-specific CAPE replacements.
    next_cape_generation: u64,
    /// Profile payload associated with each CAPE key. A changed ADD_PLAYER profile gets a fresh
    /// key and resets the chase history, matching NetworkPlayerInfo replacement semantics.
    player_cape_keys: HashMap<Uuid, (String, String)>,
    /// EntityPlayer's chasingPos/prevChasingPos state, keyed by account rather than network id so
    /// local F5 and remote players share one profile lifetime while a UUID remains online.
    cape_chases: HashMap<Uuid, CapeChase>,
    /// The Cape Physics mod's cloth settings, applied to every sampled
    /// [`CapeState`](mc_render::entity::layers::CapeState). [`CapeMotion::Vanilla`] (the mod off)
    /// hands the state back untouched. Written by `app/mods/x5_cosmetics.rs` on a settings
    /// change only.
    pub(crate) cape_motion: CapeMotion,
}

/// Optional segmented-cloth settings. Off preserves vanilla's chase and rigid cape exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CapeMotion {
    /// The unmodified vanilla rigid cape while the module is off.
    #[default]
    Vanilla,
    /// User-selected cloth style and controls; sample time/phase are filled at render time.
    Cloth(mc_render::entity::layers::CapeCloth),
}

impl CapeMotion {
    /// Attach deterministic animation inputs without advancing physics while rendering.
    #[must_use]
    pub fn apply(
        self,
        mut state: mc_render::entity::layers::CapeState,
        uuid: Uuid,
        tick: u64,
        partial_ticks: f32,
    ) -> mc_render::entity::layers::CapeState {
        if let Self::Cloth(mut cloth) = self {
            let partial = if partial_ticks.is_finite() {
                partial_ticks.clamp(0.0, 1.0)
            } else {
                0.0
            };
            cloth.time_seconds = (tick as f64 + f64::from(partial)) * 0.05;
            // Stable per-account phase: nearby capes do not all ripple in lockstep. No RNG,
            // per-frame history, extra player map, or wall-clock/frame-rate dependency.
            let bits = uuid.as_u128();
            let seed = ((bits as u64) ^ ((bits >> 64) as u64)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            cloth.phase = (seed >> 40) as f32 * (core::f32::consts::TAU / 16_777_216.0);
            state.cloth = Some(cloth);
        }
        state
    }
}

/// `LayerCape`'s body yaw `f`: `prevRenderYawOffset + (renderYawOffset - prevRenderYawOffset) *
/// partialTicks`, a plain lerp of the player's own body yaw. Unlike the body transform
/// (`interpolateRotation`, and the vehicle's yaw for a rider) there is no shortest-arc wrap and
/// no substitution. The local F5 cape used the head yaw here.
#[must_use]
pub(crate) fn cape_body_yaw(
    render: &mc_render::entity::LivingRenderState,
    partial_ticks: f32,
) -> f32 {
    render.prev_body_yaw + (render.body_yaw - render.prev_body_yaw) * partial_ticks
}

/// `LayerCape`'s interpolated `distanceWalkedModified` for the local player, from the
/// predictor's Java-parity accumulator (`Entity.moveEntity`). Remote players pass `0.0`:
/// `EntityOtherPlayerMP` moves by `setPosition` only, so vanilla never advances theirs.
#[must_use]
pub(crate) fn local_cape_walk(physics: &mc_physics::PhysicsState, partial_ticks: f32) -> f32 {
    physics.prev_distance_walked_modified
        + (physics.distance_walked_modified - physics.prev_distance_walked_modified) * partial_ticks
}

/// One player's `EntityPlayer` chase history (X5: advanced by the
/// client tick for every player, sampled by the render paths).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CapeChase {
    /// java: `prevChasingPosX/Y/Z`
    pub(crate) previous: DVec3,
    /// java: `chasingPosX/Y/Z`
    pub(crate) current: DVec3,
    last_tick: u64,
}

impl CapeChase {
    /// A history settled on `position`.
    ///
    /// Deliberate deviation: vanilla starts `chasingPos` at the origin, so a player who spawns
    /// far from 0,0,0 flails its cape for a few ticks while the ten-block snap converges. mc-rs
    /// starts the history where the player is.
    pub(crate) fn new(position: DVec3, tick: u64) -> Self {
        Self {
            previous: position,
            current: position,
            last_tick: tick,
        }
    }

    /// The chase block at the end of `EntityPlayer.onUpdate`, run once per client tick after
    /// the player's movement (`position` is the post-move `posX/Y/Z`).
    ///
    /// java: `prevChasingPos = chasingPos`; per axis, a delta beyond ten blocks snaps both the
    /// previous and the current sample to the position (and the quarter step below still applies
    /// on top, which is vanilla's teleport overshoot); then `chasingPos += delta * 0.25`.
    pub(crate) fn tick(&mut self, position: DVec3, tick: u64) {
        if self.last_tick == tick {
            return;
        }
        self.previous = self.current;
        let delta = position - self.current;
        if delta.x > 10.0 || delta.x < -10.0 {
            self.previous.x = position.x;
            self.current.x = position.x;
        }
        if delta.y > 10.0 || delta.y < -10.0 {
            self.previous.y = position.y;
            self.current.y = position.y;
        }
        if delta.z > 10.0 || delta.z < -10.0 {
            self.previous.z = position.z;
            self.current.z = position.z;
        }
        self.current += delta * 0.25;
        self.last_tick = tick;
    }

    /// `LayerCape.doRenderLayer`'s inputs at `partial_ticks`. `position` is the interpolated
    /// `prevPos + (pos - prevPos) * partialTicks`, `body_yaw` the interpolated
    /// `renderYawOffset`, `camera_yaw` the interpolated `cameraYaw` and `distance_walked` the
    /// interpolated `distanceWalkedModified`.
    pub(crate) fn state(
        &self,
        position: DVec3,
        partial_ticks: f32,
        body_yaw: f32,
        camera_yaw: f32,
        distance_walked: f32,
        sneaking: bool,
    ) -> mc_render::entity::layers::CapeState {
        let chase = self.previous + (self.current - self.previous) * f64::from(partial_ticks);
        let offset = chase - position;
        mc_render::entity::layers::CapeState {
            chase_dx: offset.x,
            chase_dy: offset.y,
            chase_dz: offset.z,
            body_yaw,
            camera_yaw,
            distance_walked,
            sneaking,
            cloth: None,
        }
    }
}

impl Default for AppearanceState {
    fn default() -> Self {
        Self::new()
    }
}

/// `MC_NO_SKIN_DOWNLOADS=1` keeps the skin and cape download worker off
/// for a whole run of the client. Benchmark tools set it (`optimization/quick_ab.py`, and the
/// PGO training's `pgo_train.py`): their runs must never reach the network and must draw the
/// same frames every time, so profile skins stay the defaults and the `<assetsDir>/skins` cache
/// is neither read nor written. Players never set it. Read once.
fn skin_downloads_disabled_by_env() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("MC_NO_SKIN_DOWNLOADS").is_some_and(|value| value == "1"))
}

impl AppearanceState {
    /// A cache with nothing in it and no worker yet.
    #[must_use]
    pub fn new() -> Self {
        let (skins, channel) = mc_render::entity::SkinCache::new();
        let (capes, cape_channel) = mc_render::entity::CapeCache::new();
        Self::from_parts(skins, channel, capes, cape_channel)
    }

    /// A cache backed by vanilla's `<assetsDir>/skins` disk layout.
    #[must_use]
    pub fn with_disk_cache(cache_root: impl Into<std::path::PathBuf>) -> Self {
        let cache_root = cache_root.into();
        let (skins, channel) = mc_render::entity::SkinCache::with_disk_cache(&cache_root);
        let (capes, cape_channel) = mc_render::entity::CapeCache::with_disk_cache(&cache_root);
        Self::from_parts(skins, channel, capes, cape_channel)
    }

    fn from_parts(
        skins: mc_render::entity::SkinCache,
        channel: mc_render::entity::skin::SkinChannel,
        capes: mc_render::entity::CapeCache,
        cape_channel: mc_render::entity::skin::CapeChannel,
    ) -> Self {
        Self {
            skins,
            capes,
            channel: Some(channel),
            cape_channel: Some(cape_channel),
            worker_running: false,
            downloads_enabled: !skin_downloads_disabled_by_env(),
            uploaded: std::collections::HashSet::new(),
            player_skin_keys: HashMap::new(),
            next_skin_generation: 0,
            next_cape_generation: 0,
            player_cape_keys: HashMap::new(),
            cape_chases: HashMap::new(),
            cape_motion: CapeMotion::Vanilla,
        }
    }

    /// Starts the download worker on `handle`. Idempotent; a second call does nothing.
    ///
    /// java: `SkinManager`'s `ExecutorService`. Separated from [`Self::new`] because `Client::new`
    /// runs in tests and headless tools that have no runtime and must not start one.
    pub fn spawn_worker(&mut self, handle: &tokio::runtime::Handle) {
        // A benchmark never starts it (one more `bool` test per frame).
        if self.worker_running || !self.downloads_enabled {
            return;
        }
        if let Some(channel) = self.channel.take() {
            mc_render::entity::skin::spawn_http_worker(channel, handle);
        }
        if let Some(channel) = self.cape_channel.take() {
            mc_render::entity::skin::spawn_cape_http_worker(channel, handle);
        }
        self.worker_running = true;
    }

    /// Whether the download worker has been started.
    #[must_use]
    pub fn worker_running(&self) -> bool {
        self.worker_running
    }

    /// Marks a skin key as resident on the GPU.
    pub fn mark_uploaded(&mut self, key: &str) {
        // Called for every visible player every frame (SKIN-2): only allocate the owned key the
        // first time, the set's contents are identical either way.
        if !self.uploaded.contains(key) {
            self.uploaded.insert(key.to_owned());
        }
    }

    /// Whether a skin key has already been uploaded to the entity phase.
    #[must_use]
    pub fn is_uploaded(&self, key: &str) -> bool {
        self.uploaded.contains(key)
    }

    /// Resolve the stable in-session cache key for one exact profile payload.
    ///
    /// java: `NetHandlerPlayClient` constructs a new `NetworkPlayerInfo` for every
    /// `ADD_PLAYER`, while `SkinManager` keeps image entries separate by texture hash.  The
    /// generation suffix reproduces the important observable property here: a changed payload
    /// cannot be hidden behind the old UUID-only `uploaded` short-circuit.
    fn player_skin_key(&mut self, uuid: Uuid, profile_value: &str) -> String {
        if let Some((previous, key)) = self.player_skin_keys.get(&uuid) {
            if previous == profile_value {
                return key.clone();
            }
        }
        // Keep the public `Client::insert_skin` seam useful for a profile that has not yet
        // requested a generation key. Once this UUID has observed a real profile, replacements
        // always take the generation path below, so a stale injected/base entry cannot mask one.
        if !self.player_skin_keys.contains_key(&uuid) {
            let legacy = skin_texture_key(uuid);
            if self.skins.contains(&legacy) {
                self.player_skin_keys
                    .insert(uuid, (profile_value.to_owned(), legacy.clone()));
                return legacy;
            }
        }
        let generation = self.next_skin_generation;
        self.next_skin_generation = self.next_skin_generation.wrapping_add(1);
        let key = format!("{}/profile-{generation}", skin_texture_key(uuid));
        self.player_skin_keys
            .insert(uuid, (profile_value.to_owned(), key.clone()));
        key
    }

    /// Resolve a skin key from the texture URL's hash, retaining a UUID/profile association only
    /// to observe profile replacement. Equal hashes intentionally reuse one resident skin across
    /// UUIDs, matching `SkinManager`'s `ResourceLocation("skins/" + hash)` cache identity.
    fn player_skin_key_for_url(&mut self, uuid: Uuid, profile_value: &str, url: &str) -> String {
        if let Some((previous, key)) = self.player_skin_keys.get(&uuid) {
            if previous == profile_value {
                return key.clone();
            }
        }
        if let Some(key) = skin_texture_key_for_url(url) {
            self.player_skin_keys
                .insert(uuid, (profile_value.to_owned(), key.clone()));
            return key;
        }
        self.player_skin_key(uuid, profile_value)
    }

    fn player_cape_key(&mut self, uuid: Uuid, profile_value: &str) -> String {
        if let Some((previous, key)) = self.player_cape_keys.get(&uuid) {
            if previous == profile_value {
                return key.clone();
            }
            // An ADD_PLAYER replacement is a new NetworkPlayerInfo. Do not let its first cape
            // frame inherit a chase offset or texture state from the old profile generation.
            self.cape_chases.remove(&uuid);
        }
        if !self.player_cape_keys.contains_key(&uuid) {
            let legacy = format!("{}/cape", skin_texture_key(uuid));
            if self.capes.contains(&legacy) {
                self.player_cape_keys
                    .insert(uuid, (profile_value.to_owned(), legacy.clone()));
                return legacy;
            }
        }
        let generation = self.next_cape_generation;
        self.next_cape_generation = self.next_cape_generation.wrapping_add(1);
        let key = format!("{}/cape/profile-{generation}", skin_texture_key(uuid));
        self.player_cape_keys
            .insert(uuid, (profile_value.to_owned(), key.clone()));
        key
    }

    /// Resolve a CAPE key by the URL hash, with a UUID/profile association for replacement
    /// detection. The separate `cape/` namespace preserves the distinct native-size cache.
    fn player_cape_key_for_url(&mut self, uuid: Uuid, profile_value: &str, url: &str) -> String {
        if let Some((previous, key)) = self.player_cape_keys.get(&uuid) {
            if previous == profile_value {
                return key.clone();
            }
            self.cape_chases.remove(&uuid);
        }
        if let Some(key) = cape_texture_key_for_url(url) {
            self.player_cape_keys
                .insert(uuid, (profile_value.to_owned(), key.clone()));
            return key;
        }
        self.player_cape_key(uuid, profile_value)
    }

    /// Advance one player's chase history for the client tick that just ran: the chase block at
    /// the end of `EntityPlayer.onUpdate`, which vanilla runs every tick for every player in the
    /// world (the local player in first person included) whether or not anything draws its cape.
    /// `position` is the player's post-move position. Idempotent within one tick.
    ///
    /// This used to happen lazily inside
    /// [`Self::cape_state`], i.e. only while a render path asked. In first person nothing asked,
    /// so the history froze where F5 was last used and the first F5 frame stepped from that stale
    /// point: the cape flicked up ("going into F5 while jumping makes the cape flick up").
    pub fn tick_cape_chase(&mut self, uuid: Uuid, tick: u64, position: DVec3) {
        self.cape_chases
            .entry(uuid)
            .or_insert_with(|| CapeChase::new(position, tick))
            .tick(position, tick);
    }

    /// Sample one player's chase history for this frame (java: `LayerCape.doRenderLayer`'s
    /// inputs), with optional cloth animation. Never advances the history: the client tick
    /// does ([`Self::tick_cape_chase`]). A player the tick has not seen yet starts a history
    /// settled on `current_position`.
    // Keep the renderer's independent camera/body/chase inputs explicit to match Java's call
    // boundary; a context object would make the public appearance API less parity-shaped.
    #[allow(clippy::too_many_arguments)]
    pub fn cape_state(
        &mut self,
        uuid: Uuid,
        tick: u64,
        current_position: DVec3,
        render_position: DVec3,
        partial_ticks: f32,
        body_yaw: f32,
        camera_yaw: f32,
        distance_walked: f32,
        sneaking: bool,
    ) -> mc_render::entity::layers::CapeState {
        let state = self
            .cape_chases
            .entry(uuid)
            .or_insert_with(|| CapeChase::new(current_position, tick))
            .state(
                render_position,
                partial_ticks,
                body_yaw,
                camera_yaw,
                distance_walked,
                sneaking,
            );
        self.cape_motion.apply(state, uuid, tick, partial_ticks)
    }

    /// The raw chase history for one player, for tests.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn cape_chase(&self, uuid: Uuid) -> Option<CapeChase> {
        self.cape_chases.get(&uuid).copied()
    }

    /// Reset all chase state, matching a disconnect/world replacement.
    pub fn reset_capes(&mut self) {
        self.cape_chases.clear();
        self.player_cape_keys.clear();
    }

    /// Forget the per-player state of every UUID `keep` rejects: its
    /// cape chase history and the profile payload behind its skin and cape keys.
    ///
    /// java: `chasingPos*` live on the `EntityPlayer` and `playerTexturesLoaded` on the
    /// `NetworkPlayerInfo`, so both die with their player. Here they are keyed by UUID beside
    /// the caches, and nothing removed them when a player left: the chase map and the cape keys
    /// grew with every player seen until the next world, the skin keys (about 0.5-1 KB of
    /// profile payload each) for the whole run. A player who comes back gets the same key again
    /// (it is derived from the texture URL), so the resident texture is reused. Keys that came
    /// from the generation fallback (a URL without a texture hash) cannot be derived again, so
    /// those are kept rather than minting a second upload later.
    pub(crate) fn retain_players(&mut self, keep: impl Fn(&Uuid) -> bool) {
        let generation_key = |key: &str| key.contains("/profile-");
        self.cape_chases.retain(|uuid, _| keep(uuid));
        self.player_cape_keys
            .retain(|uuid, (_, key)| keep(uuid) || generation_key(key));
        self.player_skin_keys
            .retain(|uuid, (_, key)| keep(uuid) || generation_key(key));
    }

    /// Entries across the per-player maps [`Self::retain_players`]
    /// bounds (for its caller's cheap "anything to do?" test and for tests).
    pub(crate) fn per_player_entries(&self) -> usize {
        self.cape_chases.len() + self.player_cape_keys.len() + self.player_skin_keys.len()
    }

    /// Copy the capes' tick state (every player's chase history, and the profile
    /// cape keys that go with it) for a replay checkpoint:
    /// vanilla keeps `chasingPos` on the player entity, which a checkpoint copies, but here it
    /// lives beside the skin caches, so a restore had reset it (a caped player's cape swung
    /// differently for a second or two after a jump).
    pub(crate) fn cape_snapshot(&self) -> CapeSnapshot {
        CapeSnapshot {
            chases: self.cape_chases.clone(),
            keys: self.player_cape_keys.clone(),
        }
    }

    /// Put a [`Self::cape_snapshot`] back.
    pub(crate) fn restore_capes(&mut self, snapshot: CapeSnapshot) {
        self.cape_chases = snapshot.chases;
        self.player_cape_keys = snapshot.keys;
    }
}

/// A copy of the capes' tick state ([`AppearanceState::cape_snapshot`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct CapeSnapshot {
    chases: HashMap<Uuid, CapeChase>,
    keys: HashMap<Uuid, (String, String)>,
}

/// Resolve the trusted profile payload and its decoded texture map for one tab-list entry.
///
/// java: `NetworkPlayerInfo.loadPlayerTextures` only reaches `SkinManager` after the signed
/// `textures` property has been accepted. Keeping that gate in
/// [`crate::sync::PlayerInfo::textures`] and this
/// decoder means the tab overlay cannot accidentally fetch a URL supplied by an unsigned proxy.
fn player_profile(
    players: &crate::sync::PlayerList,
    uuid: Uuid,
) -> Option<(&str, std::sync::Arc<ProfileTextures>)> {
    let info = players.get(&uuid)?;
    // Render/UI only: never an RSA check inside a frame. `None` while the background verifier
    // works is the same "not ready yet" every caller already handles (HITCH-1).
    info.trusted_textures.try_profile(&info.properties)
}

fn tracked_player_profile(
    tracked: &TrackedEntity,
) -> Option<(&str, std::sync::Arc<ProfileTextures>)> {
    let profile = tracked.player_profile()?;
    profile.trusted_textures.try_profile(&profile.properties)
}

/// Resolve the arm model used by a player body or tab-list face.
///
/// java: `NetworkPlayerInfo.getSkinType` returns the session profile's exact `metadata.model`
/// after the SKIN callback and otherwise falls back to `DefaultPlayerSkin.isSlimSkin(uuid)`. This
/// profile-only helper resolves the declared value; callers that also need Java's asynchronous
/// readiness boundary (such as [`player_skin_for_ui`]) apply the UUID fallback until pixels land.
#[must_use]
pub(crate) fn player_model(
    players: &crate::sync::PlayerList,
    uuid: Uuid,
) -> mc_render::entity::SkinModel {
    player_profile(players, uuid)
        .filter(|(_, profile)| profile.skin_url.is_some())
        .map_or_else(
            || mc_render::entity::SkinModel::default_for_uuid(uuid.as_u128()),
            |(_, profile)| profile.model,
        )
}

/// Resolve the arm model for a tracked remote player from the profile copied at spawn.
///
/// The UUID-derived default remains the pre-download value; once a valid snapshot texture is
/// available its declared `metadata.model` belongs to that entity generation, even if the tab
/// row has since been replaced or removed.
pub(crate) fn player_model_for_tracked(
    players: &crate::sync::PlayerList,
    tracked: &TrackedEntity,
) -> mc_render::entity::SkinModel {
    let uuid = tracked.uuid.unwrap_or_default();
    if tracked.player_profile().is_some() {
        tracked_player_profile(tracked)
            .filter(|(_, profile)| profile.skin_url.is_some())
            .map_or_else(
                || mc_render::entity::SkinModel::default_for_uuid(uuid.as_u128()),
                |(_, profile)| profile.model,
            )
    } else {
        player_model(players, uuid)
    }
}

/// A resolved skin's RGBA pixels, shared rather than copied: callers mostly discard pixels whose
/// texture key is already resident.
pub(crate) type SkinPixels = std::sync::Arc<[u8]>;

/// Ask the skin cache for a tab-list face without requiring the UI renderer to know about the
/// entity phase's cache type.
///
/// The returned pixels are already normalised to 64×64 by
/// [`mc_render::entity::SkinCache::get`]. `None` means the
/// profile was absent/invalid, or the asynchronous download has not completed; callers should
/// use [`SkinModel::default_texture`] for the vanilla fallback until the next frame.
pub(crate) fn player_skin_for_ui(
    state: &mut AppearanceState,
    players: &crate::sync::PlayerList,
    uuid: Uuid,
) -> (mc_render::entity::SkinModel, Option<(String, SkinPixels)>) {
    let fallback = mc_render::entity::SkinModel::default_for_uuid(uuid.as_u128());
    // The model comes from the same profile lookup as the URL below (it used to be a second,
    // identical lookup): with off-thread verification the verdict can land between two lookups,
    // and a ready skin must never be paired with the pre-verification default model.
    let Some((profile_value, profile)) = player_profile(players, uuid) else {
        return (fallback, None);
    };
    let Some(url) = profile.skin_url.as_deref() else {
        return (fallback, None);
    };
    let model = profile.model;
    let key = state.player_skin_key_for_url(uuid, profile_value, url);
    let Some(skin) = state.skins.get(&key, url, model) else {
        // Java does not expose `metadata.model` until SkinManager's callback fires; before that
        // `NetworkPlayerInfo.getSkinType` is still the UUID-derived default.
        return (fallback, None);
    };
    // An `Arc` clone: the callers mostly discard pixels whose key is already resident.
    (model, Some((key, skin.rgba.clone())))
}

/// [`player_skin_for_ui`] without the result: the same profile-key cache update and the same
/// `SkinCache::get` (whose first call per key starts the fetch), in the same order, but nothing
/// handed back. For a caller that must keep those side effects but will not draw the skin — a
/// container whose art has no player preview.
pub(crate) fn touch_player_skin_for_ui(
    state: &mut AppearanceState,
    players: &crate::sync::PlayerList,
    uuid: Uuid,
) {
    let Some((profile_value, profile)) = player_profile(players, uuid) else {
        return;
    };
    let Some(url) = profile.skin_url.as_deref() else {
        return;
    };
    let key = state.player_skin_key_for_url(uuid, profile_value, url);
    let _ = state.skins.get(&key, url, profile.model);
}

/// The atlas UV rectangle for a sprite name, as `[min_u, min_v, max_u, max_v]`.
///
/// java: `TextureMap.getAtlasSprite(name).getMinU()` and friends, which is how
/// `Render.renderEntityOnFire` gets at `blocks/fire_layer_0`.
///
/// `None` for a name nothing baked. [`mc_assets::sprite`] answers a miss with a placeholder whose
/// rect is the **whole atlas**, which for a billboard would draw every block texture in the game
/// stretched over the mob — so the full rect is treated as a miss here. A real 16-pixel sprite in
/// a 512-pixel-or-larger atlas can never legitimately occupy `0..1`.
#[must_use]
pub fn sprite_uv(name: &str) -> Option<[f32; 4]> {
    let location = mc_assets::ResourceLocation::parse(name).ok()?;
    // java: the atlas is keyed by the *texture* path, and `getAtlasSprite` is handed
    // `minecraft:blocks/fire_layer_0` — the `textures/` prefix and `.png` suffix are the
    // stitcher's, not the caller's.
    let location = mc_assets::ResourceLocation::new(
        &location.namespace,
        format!("textures/{}", location.path),
    );
    let uv = mc_assets::sprite(&location).uv;
    let rect = [uv.min_u, uv.min_v, uv.max_u, uv.max_v];
    (rect != [0.0, 0.0, 1.0, 1.0]).then_some(rect)
}

/// The [`mc_render::entity::EntityPhase`] texture key a remote player's body draws under.
///
/// java: `RenderPlayer.getEntityTexture` → `AbstractClientPlayer.getLocationSkin`, which asks
/// `SkinManager` for the profile's `SKIN` texture and returns
/// `DefaultPlayerSkin.getDefaultSkin(uuid)` until it arrives.
///
/// `None` means "use the descriptor's default", i.e. Steve or Alex. That is the answer for every
/// non-player, for a player whose server never sent a `textures` property, and for the frames
/// between the first request and the download landing.
///
/// Takes its three pieces of `Client` separately rather than `&mut self`: the caller is inside
/// `upload_entities`, which is already holding `&self.render` and `&self.session` across the
/// whole entity loop, and disjoint field borrows are the only way to reach a `&mut` on the cache
/// from in there.
///
/// **The download half of this is unverifiable in this environment** — there is no network.
/// Everything from a completed fetch onward is exercised by [`crate::app::Client::insert_skin`],
/// which is the same code path a real download's callback takes.
pub(crate) fn player_skin(
    state: &mut AppearanceState,
    players: &crate::sync::PlayerList,
    gpu: &mc_render::Gpu,
    render: &super::driver::RenderStack,
    tracked: &TrackedEntity,
) -> Option<String> {
    if !matches!(tracked.kind, crate::sync::entities::TrackedKind::Player) {
        return None;
    }
    let uuid = tracked.uuid?;
    let (profile_value, decoded) = if tracked.player_profile().is_some() {
        tracked_player_profile(tracked)?
    } else {
        player_profile(players, uuid)?
    };
    let url = decoded.skin_url.as_deref()?;
    player_skin_for_profile(state, gpu, render, uuid, profile_value, &decoded, url)
}

/// Upload a ready profile skin for a player that is not represented by a tracked entity.
///
/// The local `EntityPlayerSP` is deliberately not in [`TrackedEntity`], but it still resolves
/// through the same `NetworkPlayerInfo` row as a remote player. Keeping the uuid-only entry point
/// beside [`player_skin`] makes the F5 body use the exact same cache key and replacement lifetime.
/// `None` means that no trusted profile skin is available yet, so the caller must bind the
/// descriptor's UUID-derived Steve/Alex fallback.
pub(crate) fn player_skin_for_player(
    state: &mut AppearanceState,
    players: &crate::sync::PlayerList,
    gpu: &mc_render::Gpu,
    render: &super::driver::RenderStack,
    uuid: Uuid,
) -> Option<String> {
    let (profile_value, decoded) = player_profile(players, uuid)?;
    let url = decoded.skin_url.as_deref()?;
    player_skin_for_profile(state, gpu, render, uuid, profile_value, &decoded, url)
}

fn player_skin_for_profile(
    state: &mut AppearanceState,
    gpu: &mc_render::Gpu,
    render: &super::driver::RenderStack,
    uuid: Uuid,
    profile_value: &str,
    decoded: &ProfileTextures,
    url: &str,
) -> Option<String> {
    let key = state.player_skin_key_for_url(uuid, profile_value, url);
    let skin = state.skins.get(&key, url, decoded.model)?;
    render.entities.with(|phase| {
        // The phase is rebuilt by a resource-pack refresh, while the appearance cache survives;
        // checking the phase itself keeps a previously uploaded key from suppressing the upload
        // after that rebuild.
        if !phase.has_texture(&key) {
            phase.upload_texture(
                gpu,
                &key,
                &skin.rgba,
                mc_render::entity::SkinTexture::WIDTH,
                mc_render::entity::SkinTexture::HEIGHT,
            );
        }
    });
    state.mark_uploaded(&key);
    Some(key)
}

/// Upload a ready profile CAPE for one player, returning its profile/kind-safe entity texture key.
///
/// java: `NetworkPlayerInfo.loadPlayerTextures` stores CAPE separately from SKIN and
/// `LayerCape.doRenderLayer` binds it only after the asynchronous image callback completes. The
/// first call queues a non-blocking fetch and returns `None`, so callers retain the normal body
/// and simply omit the cape layer until the next frame.
pub(crate) fn player_cape_for_player(
    state: &mut AppearanceState,
    players: &crate::sync::PlayerList,
    gpu: &mc_render::Gpu,
    render: &super::driver::RenderStack,
    uuid: Uuid,
) -> Option<String> {
    let (profile_value, decoded) = player_profile(players, uuid)?;
    player_cape_for_profile(state, gpu, render, uuid, profile_value, &decoded)
}

/// Upload a ready CAPE for a tracked remote player using its spawn-time profile snapshot. A
/// missing snapshot is only a compatibility fallback for synthetic/directly-spawned fixtures;
/// real network `SpawnPlayer` records always install one at dispatch time.
pub(crate) fn player_cape_for_tracked(
    state: &mut AppearanceState,
    players: &crate::sync::PlayerList,
    gpu: &mc_render::Gpu,
    render: &super::driver::RenderStack,
    tracked: &TrackedEntity,
) -> Option<String> {
    let uuid = tracked.uuid?;
    let (profile_value, decoded) = if tracked.player_profile().is_some() {
        tracked_player_profile(tracked)?
    } else {
        player_profile(players, uuid)?
    };
    player_cape_for_profile(state, gpu, render, uuid, profile_value, &decoded)
}

fn player_cape_for_profile(
    state: &mut AppearanceState,
    gpu: &mc_render::Gpu,
    render: &super::driver::RenderStack,
    uuid: Uuid,
    profile_value: &str,
    decoded: &ProfileTextures,
) -> Option<String> {
    // Observe profile replacement even when the new payload has no CAPE. This clears the old
    // generation's chase/texture association instead of allowing a stale cape to reappear if a
    // later profile reuses the same UUID.
    let Some(url) = decoded.cape_url.as_deref() else {
        if state
            .player_cape_keys
            .get(&uuid)
            .is_some_and(|(previous, _)| previous != profile_value)
        {
            state.player_cape_keys.remove(&uuid);
            state.cape_chases.remove(&uuid);
        }
        return None;
    };
    let key = state.player_cape_key_for_url(uuid, profile_value, url);
    let cape = state.capes.get(&key, url)?;
    render.entities.with(|phase| {
        // Resource reloads rebuild EntityPhase while the async cache survives, so inspect the
        // phase itself rather than a client-side uploaded set.
        if !phase.has_texture(&key) {
            phase.upload_texture(gpu, &key, &cape.rgba, cape.width, cape.height);
        }
    });
    Some(key)
}

impl crate::app::Client {
    /// Inserts a decoded skin directly, bypassing the network — the seam a local file, a
    /// resource pack, or a test uses.
    ///
    /// java: `SkinManager.loadSkin`'s cache-hit path, which likewise never touches HTTP.
    pub fn insert_skin(&mut self, uuid: uuid::Uuid, skin: mc_render::entity::SkinTexture) {
        let key = self
            .appearance
            .player_skin_keys
            .get(&uuid)
            .map_or_else(|| skin_texture_key(uuid), |(_, key)| key.clone());
        self.appearance.skins.insert(&key, skin);
        // `SkinCache::insert` moves no counter in the spectator menu's cache key.
        self.spectator_entries_key = None;
    }

    /// Whether a skin is resident for this account.
    #[must_use]
    pub fn has_skin(&self, uuid: uuid::Uuid) -> bool {
        self.appearance.skins.contains(&skin_texture_key(uuid))
            || self
                .appearance
                .player_skin_keys
                .get(&uuid)
                .is_some_and(|(_, key)| self.appearance.skins.contains(key))
    }

    /// Inserts a decoded native-size cape directly, bypassing HTTP (tests/resource-pack tools).
    pub fn insert_cape(&mut self, uuid: uuid::Uuid, cape: mc_render::entity::CapeTexture) {
        let profile_key = self
            .appearance
            .player_cape_keys
            .get(&uuid)
            .map(|(_, key)| key.clone())
            .unwrap_or_else(|| {
                // Keep direct insertion useful before a profile row exists. The first real
                // profile still receives a generation-specific key and cannot alias this entry.
                format!("{}/cape", skin_texture_key(uuid))
            });
        self.appearance.capes.insert(&profile_key, cape);
    }

    /// Whether any profile-generation CAPE is resident for this account.
    #[must_use]
    pub fn has_cape(&self, uuid: uuid::Uuid) -> bool {
        self.appearance
            .player_cape_keys
            .get(&uuid)
            .is_some_and(|(_, key)| self.appearance.capes.contains(key))
            || self
                .appearance
                .capes
                .contains(&format!("{}/cape", skin_texture_key(uuid)))
    }

    /// `(requests started, capes delivered)` for diagnostics/tests.
    #[must_use]
    pub fn cape_stats(&self) -> (u64, u64) {
        self.appearance.capes.stats()
    }

    /// Starts the skin download worker, if it is not already running.
    pub fn start_skin_worker(&mut self, handle: &tokio::runtime::Handle) {
        self.appearance.spawn_worker(handle);
    }

    /// Allow or forbid the skin and cape download worker. Benchmarks
    /// (`skywars_bench`, and so the PGO training that runs it) call this with `false` before
    /// the first frame: they must never reach the network and must draw the same frames every
    /// run, so every player keeps the default skin (vanilla's Steve/Alex by UUID) and nothing is
    /// read from or written to the `<assetsDir>/skins` cache. The client itself never calls it.
    /// Turning it off after the worker started has no effect.
    pub fn set_skin_downloads(&mut self, enabled: bool) {
        self.appearance.downloads_enabled = enabled;
    }

    /// Whether the skin download worker has started (benchmarks check it stays off).
    #[must_use]
    pub fn skin_worker_running(&self) -> bool {
        self.appearance.worker_running()
    }

    /// `(requests started, skins delivered)` — for the F3 overlay and for tests.
    #[must_use]
    pub fn skin_stats(&self) -> (u64, u64) {
        self.appearance.skins.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With downloads off (benchmarks) `spawn_worker` never starts the
    /// download worker, so nothing can reach the network; with them on (players) it starts.
    #[test]
    fn an_offline_run_never_starts_the_skin_worker() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime");
        let mut offline = AppearanceState::new();
        offline.downloads_enabled = false;
        for _ in 0..3 {
            offline.spawn_worker(runtime.handle());
        }
        assert!(!offline.worker_running());
        assert!(offline.channel.is_some() && offline.cape_channel.is_some());

        let mut online = AppearanceState::new();
        online.downloads_enabled = true;
        online.spawn_worker(runtime.handle());
        assert!(online.worker_running());
        assert!(online.channel.is_none() && online.cape_channel.is_none());
    }

    /// A container whose art has no player preview calls
    /// [`touch_player_skin_for_ui`] instead of [`player_skin_for_ui`]. Everything that outlives
    /// the frame must match: the remembered profile key and the one fetch the first lookup
    /// starts. (Mojang-signed Notch profile, the same fixture as `sync::players`' tests.)
    #[test]
    fn touching_the_preview_skin_keeps_the_lookups_lasting_effects() {
        use mc_protocol::play::s2c::{AddPlayerEntry, GameMode, PlayerListItem, ProfileProperty};
        const NOTCH_TEXTURE_VALUE: &str = "ewogICJ0aW1lc3RhbXAiIDogMTc4NzY5MTA3NTE3MiwKICAicHJvZmlsZUlkIiA6ICIwNjlhNzlmNDQ0ZTk0NzI2YTViZWZjYTkwZTM4YWFmNSIsCiAgInByb2ZpbGVOYW1lIiA6ICJOb3RjaCIsCiAgInNpZ25hdHVyZVJlcXVpcmVkIiA6IHRydWUsCiAgInRleHR1cmVzIiA6IHsKICAgICJTS0lOIiA6IHsKICAgICAgInVybCIgOiAiaHR0cDovL3RleHR1cmVzLm1pbmVjcmFmdC5uZXQvdGV4dHVyZS8yOTIwMDlhNDkyNWI1OGYwMmM3N2RhZGMzZWNlZjA3ZWE0Yzc0NzJmNjRlMGZkYzMyY2U1NTIyNDg5MzYyNjgwIgogICAgfQogIH0KfQ==";
        const NOTCH_TEXTURE_SIGNATURE: &str = "oqgTiH1ZWCjCcU9a5bOJpskVzloygYgbdNZ16Xtn1RgKrRsPLutWL9Uok/rkg4re1O+TbsKL0lKYzQtZH0+OfWdfWIoWE4bBanHuhejGWu69EWVvcxlpEiNhZptzER9KfPGP5Q7SPsT+4MaSj/AbJP+yKWNjriYZZFV00bWSW/aexY6PJSoXMD/R57gn0TucdaZp1nrbfgL9DXf+6R6T8ou2Lz4qsPAoB2CJI2I8F9HPsLJ/dUinrp8gmjfKnvvJUf+a6XeG+BESWqqChuiD+XJjvnZSyM3Fs1zdb+RkqvFNqT4W14ohyh3mxyX3Q2S/8S+Q2axT47nnjvy9WMVpaDay1o8pquzy+zRqPps1tRFTScrp4w4jVdSzm8xL4FhNqm2H/QUoAYczVUyAfBx4Q8w9daelERlNOcQAIEeauBzan/728P/F4Rxc24Wl9iV5Qb+XeE/858QHNplwLIcSv4Wfp6G29++0XVljN7K2Vtbilp9LipVrDG028XMFY6eY/lb1n9C4hiMWF/N4YTXz8Tm+IBIBqKpgeUCzb3ZmB7bvVubPUgWdL3K7piiClRQ8ssIAN67DEOm/sTgyv0eFitWiZopPhBvgizOBPDv2/GSa3WdFRiNAtaxrqOZVO3W4HUn62XNz+leI640BMjAYMvIcpNU2x0R+S8ILf7v0zcM=";

        let uuid = Uuid::from_u128(0x069a_79f4_44e9_4726_a5be_fca9_0e38_aaf5);
        let mut players = crate::sync::PlayerList::new();
        players
            .apply(&PlayerListItem::AddPlayer(vec![AddPlayerEntry {
                uuid,
                username: "Notch".to_owned(),
                properties: vec![ProfileProperty {
                    name: "textures".into(),
                    value: NOTCH_TEXTURE_VALUE.into(),
                    signature: Some(NOTCH_TEXTURE_SIGNATURE.into()),
                }],
                game_mode: GameMode::Survival,
                ping: 0,
                display_name: None,
            }]))
            .expect("signed row decodes");
        // 14-hitches-2 verifies signatures off the frame path, so the UI's `try_profile` is
        // `None` until a verdict exists. Resolve it now, as any frame after the background
        // verifier finished would find it.
        let info = players.get(&uuid).expect("row added");
        assert!(info.trusted_textures.profile(&info.properties).is_some());

        let mut touched = AppearanceState::new();
        let mut resolved = AppearanceState::new();
        touch_player_skin_for_ui(&mut touched, &players, uuid);
        let _ = player_skin_for_ui(&mut resolved, &players, uuid);

        // Same fetches started (exactly one on first sight, unless another test in this process
        // already published this texture through the shared handoff), same remembered key.
        assert_eq!(touched.skins.stats(), resolved.skins.stats());
        assert_eq!(
            touched.player_skin_keys.get(&uuid),
            resolved.player_skin_keys.get(&uuid)
        );
        let (_, key) = touched.player_skin_keys.get(&uuid).expect("key remembered");
        assert!(touched.skins.is_inflight(key) || touched.skins.contains(key));

        // A second frame behaves like the full path's second frame: no new fetch.
        let after_first = touched.skins.stats();
        touch_player_skin_for_ui(&mut touched, &players, uuid);
        assert_eq!(touched.skins.stats(), after_first);
    }

    /// java: `EntitySheep.DYE_TO_RGB` — sixteen entries, white first, black last.
    #[test]
    fn the_dye_table_is_vanillas() {
        assert_eq!(dye_rgb(0), [1.0, 1.0, 1.0, 1.0], "WHITE");
        assert_eq!(dye_rgb(4), [0.9, 0.9, 0.2, 1.0], "YELLOW");
        assert_eq!(dye_rgb(14), [0.6, 0.2, 0.2, 1.0], "RED");
        assert_eq!(dye_rgb(15), [0.1, 0.1, 0.1, 1.0], "BLACK");
        // Out of range must not panic and must not tint.
        assert_eq!(dye_rgb(16), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(dye_rgb(-1), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn jeb_sheep_cycles_with_entity_tick_id_and_partial_tick() {
        let mut entity = mc_render::entity::EntityAnimState::default();
        assert_eq!(jeb_sheep_dye(Some("jeb_"), &entity, 0.0), Some(dye_rgb(0)));

        // Java: f = (ticksExisted % 25 + partialTicks) / 25. At the half tick we are halfway
        // through the first 1/25th step from WHITE (metadata 0) to ORANGE (metadata 1).
        assert_eq!(
            jeb_sheep_dye(Some("jeb_"), &entity, 0.5),
            Some([0.99700004, 0.99, 0.984, 1.0])
        );

        entity.ticks_existed = 25;
        assert_eq!(jeb_sheep_dye(Some("jeb_"), &entity, 0.0), Some(dye_rgb(1)));

        entity.ticks_existed = 0;
        entity.entity_id = 15;
        assert_eq!(jeb_sheep_dye(Some("jeb_"), &entity, 0.0), Some(dye_rgb(15)));
        entity.entity_id = 16;
        assert_eq!(jeb_sheep_dye(Some("jeb_"), &entity, 0.0), Some(dye_rgb(0)));
    }

    /// The `25` in `LayerSheepWool` is a period, not a rate: its final sub-tick joins the
    /// following phase exactly. This also pins the interpolation to f32 arithmetic and guards
    /// the 15 -> 0 rainbow wrap.
    #[test]
    fn jeb_sheep_joins_each_25_tick_boundary_and_keeps_opaque_alpha() {
        let mut entity = mc_render::entity::EntityAnimState {
            ticks_existed: 24,
            ..Default::default()
        };
        assert_eq!(
            jeb_sheep_dye(Some("jeb_"), &entity, 1.0),
            Some(dye_rgb(1)),
            "tick 24 plus its final partial tick is phase 1"
        );
        entity.ticks_existed = 25;
        assert_eq!(
            jeb_sheep_dye(Some("jeb_"), &entity, 0.0),
            Some(dye_rgb(1)),
            "the next whole tick starts at the same colour"
        );

        entity.ticks_existed = 399;
        let at_wrap = jeb_sheep_dye(Some("jeb_"), &entity, 1.0).expect("named sheep");
        assert_eq!(at_wrap, dye_rgb(0), "phase 15 joins phase 0");
        assert_eq!(at_wrap[3], 1.0, "LayerSheepWool never interpolates alpha");
    }

    #[test]
    fn jeb_sheep_uses_raw_name_and_java_int_wraps() {
        let mut entity = mc_render::entity::EntityAnimState::default();
        for name in [
            None,
            Some("Jeb_"),
            Some("§ajeb_"),
            Some("jeb_§r"),
            Some("jeb"),
        ] {
            assert_eq!(jeb_sheep_dye(name, &entity, 0.0), None, "{name:?}");
        }

        // `(Integer.MAX_VALUE + 1)` wraps to MIN_VALUE before `% 16`; both Java metadata
        // lookups therefore use WHITE (the negative next index is EnumDyeColor's white fallback).
        entity.ticks_existed = 25;
        entity.entity_id = i32::MAX;
        assert_eq!(jeb_sheep_dye(Some("jeb_"), &entity, 0.0), Some(dye_rgb(0)));

        // `u32::MAX` is Java's signed `-1`. Java integer division truncates toward zero, while
        // the negative remainder remains in the interpolation expression rather than wrapping.
        entity.ticks_existed = u32::MAX;
        entity.entity_id = 0;
        let wrapped = jeb_sheep_dye(Some("jeb_"), &entity, 0.0).expect("named sheep");
        // Java `float`: `1.0F * 1.04F + 0.85F * -0.04F` has raw bits `0x3f80c49b`,
        // which is 1.0059999 when rendered as Rust's shortest round-trippable f32 literal.
        assert_eq!(wrapped, [1.005_999_9, 1.02, 1.032, 1.0]);
    }

    #[test]
    fn jeb_sheep_uses_the_signed_entity_id_phase() {
        let entity = mc_render::entity::EntityAnimState {
            // Java computes `ticksExisted / 25 + getEntityId()` as signed ints. A negative id
            // therefore moves the phase backwards rather than becoming a huge u32 phase.
            ticks_existed: 50,
            entity_id: -1,
            ..Default::default()
        };
        assert_eq!(
            jeb_sheep_dye(Some("jeb_"), &entity, 0.0),
            Some(dye_rgb(1)),
            "2 + (-1) is the orange metadata phase"
        );
    }

    #[test]
    fn a_wolf_without_watched_collar_data_uses_vanillas_accidental_orange_default() {
        let state = mc_render::entity::EntityAnimState::default();
        let wolf = Tints::resolve(
            Some(mc_data::EntityType::Wolf),
            BodyDraw::Solid,
            0,
            0,
            &state,
            0.0,
        );
        assert_eq!(wolf.dye, Some([0.85, 0.5, 0.2, 1.0]));
        assert_ne!(
            wolf.dye,
            Some(dye_rgb(14)),
            "metadata 14 is red, not the fallback"
        );
        assert_eq!(DyeColour::from_dye_damage(14), DyeColour::Orange);
    }

    /// java: `RenderWolf.doRender`'s fixed-light body colour and
    /// `EntityWolf.getShadingWhileWet(partialTicks)`. The packed-light input is held constant
    /// by this pure tint oracle: only the wet RGB multiplier may change, never alpha or layers.
    #[test]
    fn wolf_wet_shading_interpolates_body_rgb_but_dry_is_identity() {
        assert_eq!(wolf_wet_shading(false, 0.0), 1.0);
        assert_eq!(wolf_wet_shading(false, 1.5), 1.0);
        assert_eq!(wolf_wet_shading(true, 0.0), 0.75);
        let interpolated = 0.5 + (1.5 - 0.5) * 0.5;
        assert_eq!(wolf_wet_shading(true, interpolated), 0.875);
        assert_eq!(wolf_wet_shading(true, 2.0), 1.0);

        let dry = Tints {
            body: [0.8, 0.7, 0.6, 0.15],
            dye: Some([0.85, 0.5, 0.2, 1.0]),
        };
        let wet = dry.with_wolf_wet(true, 1.0);
        assert_eq!(wet.body, [0.8 * 0.875, 0.7 * 0.875, 0.6 * 0.875, 0.15]);
        assert_eq!(
            wet.dye, dry.dye,
            "wet body shading must not tint the collar input"
        );
        assert_eq!(dry.with_wolf_wet(false, 1.0), dry);
    }

    /// java: `setBrightness`'s `hurtTime > 0 || deathTime > 0`.
    #[test]
    fn the_hurt_flash_fires_on_either_timer_and_never_otherwise() {
        assert_eq!(hurt_flash(0, 0), None);
        assert_eq!(hurt_flash(1, 0), Some([1.0, 0.0, 0.0, 0.3]));
        assert_eq!(hurt_flash(0, 1), Some([1.0, 0.0, 0.0, 0.3]));
    }

    /// java: `RenderCreeper.getColorMultiplier` — a strobe, not a ramp.
    #[test]
    fn the_creeper_flash_strobes_and_brightens() {
        // (int)(0.05 * 10) % 2 == 0 -> off.
        assert_eq!(creeper_flash(0.05), None);
        // (int)(0.15 * 10) == 1 -> on, weight = (int)(0.15 * 0.2 * 255) / 255.
        let on = creeper_flash(0.15).expect("odd decile flashes");
        assert_eq!(&on[..3], &[1.0, 1.0, 1.0]);
        assert!(
            on[3] > 0.0,
            "the replay has a nonzero affine weight: {on:?}"
        );
        // Full fuse: (int)(1.0*10) = 10, even -> off, which is vanilla's own last-frame quirk.
        assert_eq!(creeper_flash(1.0), None);
        let strong = creeper_flash(0.95).expect("odd decile");
        let weak = creeper_flash(0.15).expect("odd decile");
        assert!(strong[3] > weak[3], "later in the fuse flashes harder");
        assert_ne!(
            on,
            EntityVertex::WHITE,
            "multiplying by opaque white is an identity, not the Java flash"
        );
    }

    /// java: `renderModel`'s two flags. The middle case is the one that is easy to lose.
    #[test]
    fn invisibility_has_three_outcomes_not_two() {
        assert_eq!(BodyDraw::decide(false, false), BodyDraw::Solid);
        assert_eq!(BodyDraw::decide(false, true), BodyDraw::Solid);
        assert_eq!(BodyDraw::decide(true, false), BodyDraw::Skip);
        assert_eq!(BodyDraw::decide(true, true), BodyDraw::Ghost);
        assert_eq!(BodyDraw::Ghost.tint(), [1.0, 1.0, 1.0, 0.15]);
        assert_eq!(BodyDraw::Solid.tint(), EntityVertex::WHITE);
        assert_eq!(
            BodyDraw::Ghost.blend(),
            mc_render::entity::LayerBlend::Ghost
        );
    }

    /// java: `Render.renderEntityOnFire`'s `while (f3 > 0)` loop.
    #[test]
    fn the_fire_overlay_stacks_one_quad_per_forty_five_hundredths_of_height() {
        let mut out = Vec::new();
        // A player: width 0.6, height 1.8. f = 0.84, f3 = 1.8 / 0.84 = 2.142..., so the loop runs
        // ceil(2.142 / 0.45) = 5 times.
        emit_fire_overlay(
            Vec3::ZERO,
            0.6,
            1.8,
            0.0,
            |_| [0.0, 0.0, 1.0, 1.0],
            &mut out,
        );
        assert_eq!(out.len(), 5 * 6, "five layers, six vertices each");
        // Each successive layer is narrower (0.9x) and further forward (+0.03 in Z).
        let first_half_width = out[0].pos[0];
        let second_half_width = out[6].pos[0];
        assert!(
            second_half_width.abs() < first_half_width.abs(),
            "{first_half_width} -> {second_half_width}"
        );
        assert!(out[6].pos[2] > out[0].pos[2], "each layer steps forward");
    }

    /// A zero-width entity would divide by zero into an unbounded loop.
    #[test]
    fn a_degenerate_entity_emits_no_flames() {
        let mut out = Vec::new();
        emit_fire_overlay(Vec3::ZERO, 0.0, 1.8, 0.0, |_| [0.0; 4], &mut out);
        assert!(out.is_empty());
    }

    /// java: `doRenderShadowAndFire` (`:306`) — `(1 - d0 / 256) * shadowOpaque`, where `d0` is
    /// `getDistanceToCamera` (`:478`), i.e. the **squared** distance. The old name and body of
    /// this test asserted a fade over 256 blocks, which is the bug, not the behaviour.
    #[test]
    fn shadow_alpha_fades_to_nothing_at_sixteen_blocks_because_the_256_is_squared() {
        assert!((shadow_alpha(0.0) - 1.0).abs() < 1e-6);
        // Half alpha at sqrt(128) = 11.31 blocks, not at 128.
        assert!((shadow_alpha(128.0) - 0.5).abs() < 1e-5);
        assert_eq!(shadow_alpha(256.0), 0.0, "gone at exactly sixteen blocks");
        assert_eq!(shadow_alpha(300.0), 0.0, "clamped, never negative");
        // The negative control that bites: with the `sqrt` back in, a mob 15 blocks away keeps
        // 0.94 alpha instead of 0.12.
        let at_fifteen = shadow_alpha(15.0 * 15.0);
        assert!(
            (at_fifteen - 0.121_093_75).abs() < 1e-5,
            "15 blocks must be nearly faded out, got {at_fifteen}"
        );
    }

    /// The per-block part of `Render.renderShadowBlock` uses `getLight(pos, true)`, not raw
    /// combined light and not a linear `level / 15` factor. These three vectors intentionally
    /// separate noon sky, midnight sky, and a torch which must remain bright after sky
    /// darkening. Restoring either old shortcut makes at least one assertion fail.
    #[test]
    fn shadow_block_alpha_applies_skylight_subtraction_and_the_java_brightness_table() {
        let noon = ShadowLighting {
            skylight_subtracted: 0,
            ambient_light: 0.0,
        };
        let midnight = ShadowLighting {
            skylight_subtracted: 11,
            ambient_light: 0.0,
        };

        let daylight = shadow_block_alpha(1.0, 65.0, 65, 15, 0, noon).unwrap();
        assert!(
            (daylight - 0.5).abs() < 1e-6,
            "full daylight uses table[15]"
        );

        // Raw sky 15 falls to level 4 at midnight. Table[4] is 1/12, not the old 4/15 ramp.
        let moonlight = shadow_block_alpha(1.0, 65.0, 65, 15, 0, midnight).unwrap();
        assert!(
            (moonlight - 1.0 / 24.0).abs() < 1e-6,
            "night must darken the sky channel"
        );
        assert!(moonlight < daylight * 0.11);

        // Block light wins after subtraction, exactly as Chunk.getLightSubtracted does.
        let torch = shadow_block_alpha(1.0, 65.0, 65, 0, 14, midnight).unwrap();
        let expected = mc_render::sky::lightmap::light_brightness_table(0.0)[14] * 0.5;
        assert!(
            (torch - expected).abs() < 1e-6,
            "night must not subtract torchlight"
        );
        assert!(
            torch > moonlight,
            "a torch-lit shadow stays brighter than moonlight"
        );

        // The same darkened query is also the >3 emission gate. Sky 14 becomes 3 at midnight,
        // while block light 4 keeps a quad eligible.
        assert_eq!(shadow_block_alpha(1.0, 65.0, 65, 14, 0, midnight), None);
        assert!(shadow_block_alpha(1.0, 65.0, 65, 0, 4, midnight).is_some());
    }

    /// java: `renderShadow` walks the block box under the entity and `renderShadowBlock` emits a
    /// quad only for a full cube. A mob standing over air casts nothing.
    #[test]
    fn a_shadow_needs_a_full_cube_under_the_entity() {
        use mc_math::ChunkPos;

        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let camera = Camera::default();
        let pos = DVec3::new(8.0, 65.0, 8.0);

        // Nothing but air below: `getRenderType() != -1` fails for air, so no quads.
        let mut air = Vec::new();
        let quads = emit_shadow(&world, &camera, pos, 0.5, 1.0, &mut air);
        assert_eq!(quads, 0, "a mob over air casts no shadow");
        assert!(air.is_empty());

        // A stone floor at y = 64: the mob stands on it and a shadow appears.
        for x in 5..=11 {
            for z in 5..=11 {
                world.set_block(
                    BlockPos::new(x, 64, z),
                    mc_data::BlockState::new(mc_data::BlockId(1), 0),
                );
            }
        }
        let mut stone = Vec::new();
        let quads = emit_shadow(&world, &camera, pos, 0.5, 1.0, &mut stone);
        assert!(quads > 0, "a mob over stone casts a shadow");
        assert_eq!(stone.len(), quads * 6, "six vertices per quad");
        // The alpha is the shadow's whole visual content — `shadow.png` is pure black with a
        // radial alpha ramp, so a zero here would draw nothing at all.
        assert!(stone.iter().all(|v| v.color[3] > 0.0));
        assert!(stone.iter().all(|v| v.color[3] <= 1.0));
        // Every vertex faces straight up and sits vanilla's `+0.015625` above the top face.
        assert!(stone.iter().all(|v| v.normal == [0.0, 1.0, 0.0]));
        assert!(stone.iter().all(|v| (v.pos[1] - 65.015_625).abs() < 1e-3));

        // A bigger mob covers more blocks, which is what `shadow_size` is *for* — the assertion
        // that could not pass while the descriptor field had no consumer at all.
        let mut big = Vec::new();
        let big_quads = emit_shadow(&world, &camera, pos, 1.5, 1.0, &mut big);
        assert!(
            big_quads > quads,
            "a 1.5-block shadow covers more blocks than a 0.5-block one: {big_quads} vs {quads}"
        );

        // A zero shadow size is `Render.shadowSize == 0` — the armour stand, and every entity
        // once `entityShadows` is off.
        let mut none = Vec::new();
        assert_eq!(emit_shadow(&world, &camera, pos, 0.0, 1.0, &mut none), 0);
    }

    /// java: `Entity.getFlag(0)` and `getFlag(5)`, off the shared watched byte.
    #[test]
    fn the_shared_flag_byte_drives_burning_and_invisibility() {
        use crate::sync::entities::{EntityTracker, TrackedKind};
        use mc_world::entity::MetadataValue as M;

        let spawn = |flags: Option<i8>| {
            let mut world = World::new();
            let mut tracker = EntityTracker::new();
            let mut meta = mc_world::entity::Metadata::new();
            if let Some(f) = flags {
                meta.set(0, M::Byte(f));
            }
            tracker.spawn(
                &mut world,
                1,
                TrackedKind::Type(mc_data::EntityType::Zombie),
                None,
                [0, 0, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                meta,
            );
            let tracked = tracker.get(1).unwrap().clone();
            (world, tracked)
        };

        let (world, t) = spawn(None);
        assert!(!is_burning(&world, &t), "no metadata means not burning");
        assert!(!is_invisible(&world, &t));

        let (world, t) = spawn(Some(0x01));
        assert!(is_burning(&world, &t), "bit 0 is Entity.isBurning");
        assert!(!is_invisible(&world, &t));

        let (world, t) = spawn(Some(0x20));
        assert!(!is_burning(&world, &t));
        assert!(is_invisible(&world, &t), "bit 5 is Entity.isInvisible");

        // Both at once, which is what a burning invisible mob sends.
        let (world, t) = spawn(Some(0x21));
        assert!(is_burning(&world, &t) && is_invisible(&world, &t));
    }

    #[test]
    fn a_zero_player_model_parts_byte_really_hides_every_optional_layer() {
        use crate::sync::entities::{EntityTracker, TrackedKind};
        use mc_world::entity::MetadataValue as M;

        let mask = |wire: Option<i8>| {
            let mut world = World::new();
            let mut tracker = EntityTracker::new();
            let mut meta = mc_world::entity::Metadata::new();
            if let Some(wire) = wire {
                meta.set(10, M::Byte(wire));
            }
            tracker.spawn(
                &mut world,
                1,
                TrackedKind::Player,
                None,
                [0, 0, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                meta,
            );
            let tracked = tracker.get(1).unwrap().clone();
            skin_parts_mask(&world, &tracked)
        };

        assert_eq!(mask(None), 0, "EntityPlayer.entityInit registers byte zero");
        assert_eq!(mask(Some(0)), 0, "zero is a valid all-layers-hidden mask");
        assert_eq!(mask(Some(0x40)), 0x40, "the hat bit survives alone");
        assert_eq!(mask(Some(0x7f)), 0x7f, "all seven vanilla bits survive");
        assert_ne!(
            mask(Some(0)),
            !0,
            "negative control: zero must not become all-on"
        );
    }

    /// java: `EntityBlaze.isBurning` overrides the base method with watched index 16 bit 0.
    #[test]
    fn a_blazes_attack_byte_replaces_the_shared_burning_flag() {
        use crate::sync::entities::{EntityTracker, TrackedKind};
        use mc_world::entity::MetadataValue as M;

        let burning = |shared_flags: i8, blaze_flags: i8| {
            let mut world = World::new();
            let mut tracker = EntityTracker::new();
            let mut meta = mc_world::entity::Metadata::new();
            meta.set(0, M::Byte(shared_flags));
            meta.set(16, M::Byte(blaze_flags));
            tracker.spawn(
                &mut world,
                1,
                TrackedKind::Type(mc_data::EntityType::Blaze),
                None,
                [0, 0, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                meta,
            );
            let tracked = tracker.get(1).unwrap().clone();
            is_burning(&world, &tracked)
        };

        assert!(
            burning(0, 1),
            "the fireball-charge byte enables the overlay"
        );
        assert!(!burning(0, 2), "only bit zero is EntityBlaze's fire bit");
        assert!(
            !burning(1, 0),
            "negative control: the blaze override does not OR in Entity's shared bit"
        );
    }

    #[test]
    fn fire_immune_species_suppress_the_shared_burning_flag() {
        use crate::sync::entities::{EntityTracker, TrackedKind};
        use mc_world::entity::MetadataValue as M;

        for kind in [
            mc_data::EntityType::Ghast,
            mc_data::EntityType::PigZombie,
            mc_data::EntityType::LavaSlime,
            mc_data::EntityType::EnderDragon,
            mc_data::EntityType::WitherBoss,
        ] {
            let mut world = World::new();
            let mut tracker = EntityTracker::new();
            let mut meta = mc_world::entity::Metadata::new();
            meta.set(0, M::Byte(1));
            tracker.spawn(
                &mut world,
                1,
                TrackedKind::Type(kind),
                None,
                [0, 0, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                meta,
            );
            let tracked = tracker.get(1).unwrap().clone();
            assert!(
                !is_burning(&world, &tracked),
                "{kind:?} inherits the isImmuneToFire guard"
            );
        }
    }

    /// java: `EntityEnderman.setHeldBlockState` writes `Block.getStateId(state)` to watched index
    /// 16, and `getHeldBlockState` reads that short back unsigned through `Block.getStateById`.
    #[test]
    fn an_endermans_carried_block_keeps_the_metadata_until_the_block_model_lookup() {
        use crate::sync::entities::{EntityTracker, TrackedKind};
        use mc_world::entity::MetadataValue as M;

        let decode = |java_state_id: u16| {
            let mut world = World::new();
            let mut tracker = EntityTracker::new();
            let mut meta = mc_world::entity::Metadata::new();
            meta.set(16, M::Short(java_state_id as i16));
            tracker.spawn(
                &mut world,
                1,
                TrackedKind::Type(mc_data::EntityType::Enderman),
                None,
                [0, 0, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                meta,
            );
            let tracked = tracker.get(1).unwrap().clone();
            carried_block_id(&world, &tracked)
        };

        // Java oracle: red sand is block 12, metadata 1 => 12 | (1 << 12) = 4108.
        let red_sand = decode(4_108).expect("red sand is not air");
        assert_eq!(red_sand.block(), mc_data::BlockId(12));
        assert_eq!(red_sand.meta(), 1);
        assert_eq!(red_sand.to_java_state_id(), 4_108);
        assert_ne!(
            red_sand,
            mc_data::BlockState::new(mc_data::BlockId(12), 0),
            "discarding metadata silently turns red sand into ordinary sand"
        );
        assert_eq!(block_model_state(red_sand.into()), Some(red_sand));

        // Java oracle: black wool sets the short's sign bit: 35 | (15 << 12) = 61475/-4061.
        let black_wool = decode(61_475).expect("signed watched shorts are read unsigned");
        assert_eq!(black_wool.block(), mc_data::BlockId(35));
        assert_eq!(black_wool.meta(), 15);
        assert_eq!(black_wool.to_java_state_id(), 61_475);
        assert_eq!(block_model_state(black_wool.into()), Some(black_wool));

        assert_eq!(decode(0), None, "the default air state means empty hands");
    }

    #[test]
    fn render_block_brightness_uses_block_states_not_inventory_item_models() {
        let poppy = block_model_state(BlockModelInput::from(38_i32)).expect("model block");
        assert_eq!(poppy.block(), mc_data::BlockId(38));
        assert_eq!(poppy.meta(), 0);

        let red_sand = mc_data::BlockState::new(mc_data::BlockId(12), 1);
        assert_eq!(block_model_state(red_sand.into()), Some(red_sand));
        assert_ne!(
            block_model_state(red_sand.into()),
            block_model_state(BlockModelInput::from(12_i32)),
            "a state-keyed lookup must not collapse metadata to the block id"
        );

        assert_eq!(
            block_model_state(BlockModelInput::from(-1_i32)),
            None,
            "invalid numeric ids are rejected"
        );
        assert_eq!(
            block_model_state(mc_data::BlockState::AIR.into()),
            None,
            "renderBlockBrightness skips a non-model render type"
        );

        // The shipped resources make the registry choice observable: red_flower has a crossed
        // planes block model, while its item registration resolves the separate "poppy" generated
        // model.
        let Some(jar) = crate::app::config::discover_jars_dir()
            .map(|dir| dir.join("versions").join("1.8.9").join("1.8.9.jar"))
            .filter(|jar| jar.is_file())
        else {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return;
        };
        let mut manager = mc_assets::ResourceManager::new();
        manager.push_pack(Box::new(
            mc_assets::ZipResourcePack::open(&jar).expect("open vanilla jar"),
        ));
        let (assets, report) = mc_assets::bake_all(&manager, 2_048).expect("bake vanilla models");
        assert!(report.is_clean(), "{:?}", report.failures);
        mc_assets::model::reinstall(assets);

        let selected = block_item_model(poppy).expect("the poppy block model has geometry");
        let block_model = mc_assets::model_for(poppy);
        let inventory_model = mc_assets::item_model_for(mc_data::ItemId(38));
        assert!(
            std::ptr::eq(selected, block_model),
            "renderBlockBrightness must select BlockModelShapes.getModelForState"
        );
        assert!(
            !std::ptr::eq(selected, inventory_model),
            "negative control: the registered poppy inventory model is a different registry entry"
        );
        assert!(
            !selected.is_empty(),
            "the crossed-planes block model must have geometry"
        );
    }

    /// java: `LayerCustomHead.doRenderLayer`'s `EntityVillager ||
    /// (EntityZombie && isVillager())` predicate. The zombie watcher must be the exact `== 1`
    /// check; a generic child zombie uses the ordinary shrink/lift branch.
    #[test]
    fn custom_head_villager_gate_matches_the_entity_class_and_watcher() {
        use crate::sync::entities::{EntityTracker, TrackedKind};
        use mc_world::entity::MetadataValue as M;

        let spawn = |kind: mc_data::EntityType, metadata: &[(u8, M)]| {
            let mut world = World::new();
            let mut tracker = EntityTracker::new();
            let mut watched = mc_world::entity::Metadata::new();
            for (index, value) in metadata {
                watched.set(*index, value.clone());
            }
            tracker.spawn(
                &mut world,
                1,
                TrackedKind::Type(kind),
                None,
                [0, 0, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                watched,
            );
            (world, tracker.get(1).expect("just spawned").clone())
        };

        let (world, villager) = spawn(mc_data::EntityType::Villager, &[]);
        assert!(is_villager_like(&world, &villager));

        let (world, zombie_villager) = spawn(mc_data::EntityType::Zombie, &[(13, M::Byte(1))]);
        assert!(is_villager_like(&world, &zombie_villager));

        let (world, ordinary_zombie) = spawn(mc_data::EntityType::Zombie, &[(13, M::Byte(0))]);
        assert!(!is_villager_like(&world, &ordinary_zombie));

        // The same metadata byte has no meaning on another entity class.
        let (world, skeleton) = spawn(mc_data::EntityType::Skeleton, &[(13, M::Byte(1))]);
        assert!(!is_villager_like(&world, &skeleton));
    }

    /// A caller-level negative control for both layer findings. The body is posed on a rotated
    /// entity base before the adapter runs, so a legacy world-space pre-multiply changes the
    /// origin. The second half proves the zombie-villager child exception reaches the render-side
    /// helper and is not applied to an ordinary baby zombie.
    #[test]
    fn custom_head_caller_routes_entity_space_and_zombie_villager_child_policy() {
        use crate::sync::entities::{EntityTracker, TrackedKind};
        use mc_world::entity::MetadataValue as M;

        let spawn = |kind: mc_data::EntityType, metadata: &[(u8, M)]| {
            let mut world = World::new();
            let mut tracker = EntityTracker::new();
            let mut watched = mc_world::entity::Metadata::new();
            for (index, value) in metadata {
                watched.set(*index, value.clone());
            }
            tracker.spawn(
                &mut world,
                1,
                TrackedKind::Type(kind),
                None,
                [0, 0, 0],
                0.0,
                0.0,
                DVec3::ZERO,
                watched,
            );
            (world, tracker.get(1).expect("just spawned").clone())
        };
        let assert_matrix_close = |got: Mat4, want: Mat4| {
            for (a, b) in got.to_cols_array().iter().zip(want.to_cols_array()) {
                assert!((a - b).abs() < 1e-6, "{a} != {b}");
            }
        };

        let (ordinary_world, ordinary) = spawn(
            mc_data::EntityType::Zombie,
            &[(12, M::Byte(1)), (13, M::Byte(0))],
        );
        let mut ordinary_params = mc_render::entity::AnimParams::still();
        ordinary_params.entity.child = true;
        ordinary_params.entity.sneaking = true;
        ordinary_params.net_head_yaw = 37.0;
        ordinary_params.head_pitch = -23.0;
        let base = Mat4::from_translation(Vec3::new(11.0, 7.0, -5.0)) * Mat4::from_rotation_y(0.6);
        let mut ordinary_models = mc_render::entity::RenderManager::new();
        let mut body = Vec::new();
        ordinary_models
            .model_mut(mc_render::entity::ModelKind::Zombie)
            .pose_and_render(&ordinary_params, base, &mut body);

        let (ordinary_matrix, ordinary_skull) = custom_head_matrix_for_entity(
            &mut ordinary_models,
            &ordinary_world,
            &ordinary,
            mc_render::entity::ModelKind::Zombie,
            base,
            &ordinary_params,
            mc_data::ItemId(1),
        )
        .expect("stone is an ItemBlock");
        assert!(!ordinary_skull);

        let ordinary_model = ordinary_models
            .model_mut(mc_render::entity::ModelKind::Zombie)
            .model();
        let ordinary_head = ordinary_model.find("head").expect("zombie has a head");
        let expected = mc_render::entity::layers::custom_head_matrix(
            base,
            ordinary_model,
            ordinary_head,
            ordinary_params.scale,
            true,
            true,
            false,
            mc_render::entity::layers::CustomHeadItem::Block,
        );
        assert_matrix_close(ordinary_matrix, expected);

        let head_post = ordinary_model
            .get(ordinary_head)
            .post_render_matrix(ordinary_params.scale);
        let wrong_world_space = Mat4::from_translation(Vec3::new(0.0, 0.2, 0.0))
            * Mat4::from_scale(Vec3::splat(0.7))
            * Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))
            * base
            * head_post
            * Mat4::from_translation(Vec3::new(0.0, -0.25, 0.0))
            * Mat4::from_rotation_y(core::f32::consts::PI)
            * Mat4::from_scale(Vec3::new(0.625, -0.625, -0.625));
        assert_ne!(
            ordinary_matrix.transform_point3(Vec3::ZERO),
            wrong_world_space.transform_point3(Vec3::ZERO),
            "sneak/child offsets must follow the entity basis"
        );
        assert_eq!(
            custom_head_matrix_for_entity(
                &mut ordinary_models,
                &ordinary_world,
                &ordinary,
                mc_render::entity::ModelKind::Zombie,
                base,
                &ordinary_params,
                mc_data::ItemId(276),
            ),
            None,
            "a sword is neither an ItemBlock nor Items.skull"
        );

        let (villager_world, zombie_villager) = spawn(
            mc_data::EntityType::Zombie,
            &[(12, M::Byte(1)), (13, M::Byte(1))],
        );
        let mut villager_params = mc_render::entity::AnimParams::still();
        villager_params.entity.child = true;
        villager_params.net_head_yaw = -31.0;
        villager_params.head_pitch = 17.0;
        let mut villager_models = mc_render::entity::RenderManager::new();
        let mut villager_body = Vec::new();
        villager_models
            .model_mut(mc_render::entity::ModelKind::ZombieVillager)
            .pose_and_render(&villager_params, base, &mut villager_body);

        let (skull_matrix, is_skull) = custom_head_matrix_for_entity(
            &mut villager_models,
            &villager_world,
            &zombie_villager,
            mc_render::entity::ModelKind::ZombieVillager,
            base,
            &villager_params,
            mc_data::ItemId(397),
        )
        .expect("skull selects the skull branch");
        assert!(is_skull);
        let villager_model = villager_models
            .model_mut(mc_render::entity::ModelKind::ZombieVillager)
            .model();
        let villager_head = villager_model
            .find("head")
            .expect("zombie villager has a head");
        let villager_expected = mc_render::entity::layers::custom_head_matrix(
            base,
            villager_model,
            villager_head,
            villager_params.scale,
            false,
            true,
            true,
            mc_render::entity::layers::CustomHeadItem::Skull,
        );
        assert_matrix_close(skull_matrix, villager_expected);

        let ordinary_child = mc_render::entity::layers::custom_head_matrix(
            base,
            villager_model,
            villager_head,
            villager_params.scale,
            false,
            true,
            false,
            mc_render::entity::layers::CustomHeadItem::Skull,
        );
        assert_ne!(
            skull_matrix.transform_point3(Vec3::ZERO),
            ordinary_child.transform_point3(Vec3::ZERO),
            "a zombie-villager child suppresses the generic child lift"
        );
    }

    /// java: `ItemArmor.getColor` — `display.color`, defaulting to `10511680` (`0xA06540`).
    #[test]
    fn undyed_leather_takes_vanillas_default_brown() {
        let undyed = leather_dye(None);
        assert!((undyed[0] - 160.0 / 255.0).abs() < 1e-5, "{undyed:?}");
        assert!((undyed[1] - 101.0 / 255.0).abs() < 1e-5, "{undyed:?}");
        assert!((undyed[2] - 64.0 / 255.0).abs() < 1e-5, "{undyed:?}");
        assert_eq!(undyed[3], 1.0);
        // It is a real tint, not the identity: undyed leather is visibly brown, and drawing it
        // white was the bug this closes.
        assert_ne!(undyed, EntityVertex::WHITE);
    }

    /// java: `TileEntitySkullRenderer`'s five sheets, indexed by the stack damage.
    #[test]
    fn skull_textures_cover_the_five_vanilla_variants() {
        assert_eq!(skull_texture(0), "textures/entity/skeleton/skeleton.png");
        assert_eq!(
            skull_texture(1),
            "textures/entity/skeleton/wither_skeleton.png"
        );
        assert_eq!(skull_texture(2), "textures/entity/zombie/zombie.png");
        assert_eq!(skull_texture(3), "textures/entity/steve.png");
        assert_eq!(skull_texture(4), "textures/entity/creeper/creeper.png");
        // An out-of-range damage value falls back rather than panicking.
        assert_eq!(skull_texture(99), "textures/entity/skeleton/skeleton.png");
    }

    /// java: `LayerMooshroomMushroom` draws three mushrooms — two off the body, one off the head.
    #[test]
    fn a_mooshroom_wears_three_mushrooms_or_two_without_a_head_part() {
        let base = Mat4::IDENTITY;
        let with_head = mooshroom_mushroom_matrices(base, Some(Mat4::IDENTITY));
        assert_eq!(with_head.len(), 3);
        // The two body mushrooms are at different places — vanilla's second one adds
        // `translate(0.1, 0, -0.6)` and another 42-degree turn.
        assert_ne!(with_head[0], with_head[1]);
        assert_ne!(with_head[1], with_head[2]);
        // A model with no head part loses only the third.
        assert_eq!(mooshroom_mushroom_matrices(base, None).len(), 2);
    }

    /// java: `LayerHeldItemWitch`'s four-way branch, chosen from the item.
    #[test]
    fn the_witchs_pose_forks_on_what_it_is_holding() {
        use mc_render::entity::layers::WitchHeldItem as W;
        // 373 is `minecraft:potion`, the flat sprite case and what a witch actually holds.
        assert_eq!(witch_pose(mc_data::ItemId(373)), W::Flat);
        // 261 is the bow.
        assert_eq!(witch_pose(mc_data::ItemId(261)), W::Bow);
        // 54 is the chest.
        assert_eq!(witch_pose(mc_data::ItemId(54)), W::Chest);
        // 346 is the fishing rod: full 3D *and* the one item vanilla spins about Z.
        assert_eq!(
            witch_pose(mc_data::ItemId(346)),
            W::Full3D {
                rotate_around: true
            }
        );
    }

    /// java: `SkinManager.loadProfileTextures`' base64 payload.
    #[test]
    fn a_textures_property_yields_a_url_and_an_arm_width() {
        use base64::Engine as _;
        let payload = r#"{"timestamp":1,"profileId":"a","profileName":"b","textures":{
            "SKIN":{"url":"http://textures.minecraft.net/texture/abc",
                    "metadata":{"model":"slim"}},
            "CAPE":{"url":"http://textures.minecraft.net/texture/def"}}}"#;
        let encoded = base64::engine::general_purpose::STANDARD.encode(payload);
        let decoded = decode_textures_property(&encoded).expect("valid payload");
        assert_eq!(
            decoded.skin_url.as_deref(),
            Some("http://textures.minecraft.net/texture/abc")
        );
        assert_eq!(
            decoded.cape_url.as_deref(),
            Some("http://textures.minecraft.net/texture/def")
        );
        assert_eq!(decoded.model, mc_render::entity::SkinModel::Slim);

        // No metadata at all is the classic-arm majority case, not an error.
        let plain =
            r#"{"textures":{"SKIN":{"url":"https://textures.minecraft.net/texture/plain"}}}"#;
        let plain = base64::engine::general_purpose::STANDARD.encode(plain);
        let decoded = decode_textures_property(&plain).unwrap();
        assert_eq!(decoded.model, mc_render::entity::SkinModel::Classic);
        assert_eq!(decoded.cape_url, None);
    }

    /// A proxy that rewrites the tab list can send anything at all here; none of it may panic.
    #[test]
    fn a_malformed_textures_property_is_none_not_a_panic() {
        assert_eq!(decode_textures_property(""), None);
        assert_eq!(decode_textures_property("not base64 !!!"), None);
        // Valid base64, not JSON.
        assert_eq!(decode_textures_property("aGVsbG8="), None);
        // Valid JSON, wrong shape.
        use base64::Engine as _;
        let wrong = base64::engine::general_purpose::STANDARD.encode(r#"{"nope":1}"#);
        assert_eq!(decode_textures_property(&wrong), None);
    }

    /// authlib's URI-host suffix gate is intentionally more specific than a substring check.
    /// The allowed values cover both bundled 1.5.21 suffixes, while the negative cases reach
    /// the authority/userinfo/whole-map branches that a `contains("minecraft.net")` mutation
    /// would accidentally accept.
    #[test]
    fn texture_urls_must_use_an_authlib_whitelisted_host() {
        for url in [
            "http://textures.minecraft.net/texture/abc",
            "https://sessionserver.mojang.com/session/minecraft/skin",
            "ftp://cdn.minecraft.net/legacy/skin",
            "//textures.minecraft.net/texture/protocol-relative",
        ] {
            assert!(is_authlib_texture_domain(url), "must accept {url}");
        }
        for url in [
            "https://minecraft.net/texture/no-leading-dot",
            "https://textures.minecraft.net.evil.example/texture/abc",
            "https://evil.example@textures.minecraft.net.evil.example/texture/abc",
            "https://textures.minecraft.net@evil.example/texture/abc",
            "https://[::1]/texture/abc",
            "mailto:textures.minecraft.net",
            "not a URL",
        ] {
            assert!(!is_authlib_texture_domain(url), "must reject {url}");
        }
    }

    /// `getTextures` iterates every map entry before returning it. One unsafe CAPE therefore
    /// prevents use of an otherwise trusted SKIN too; accepting only the field the caller wants
    /// would let a future cape consumer re-open the SSRF/authlib parity hole.
    #[test]
    fn one_off_domain_texture_rejects_the_entire_profile_map() {
        use base64::Engine as _;
        let payload = r#"{"textures":{
            "SKIN":{"url":"https://textures.minecraft.net/texture/skin"},
            "CAPE":{"url":"https://textures.minecraft.net.evil.example/texture/cape"}}}"#;
        let encoded = base64::engine::general_purpose::STANDARD.encode(payload);
        assert_eq!(decode_textures_property(&encoded), None);
    }

    /// An unsigned skull texture (a placed head, an item, a replay) is
    /// the one textures map a server writes freely. authlib's suffix rule, done by hand, read the
    /// host after the `@`; the request's parser ends an http authority at the backslash and
    /// connects to the host before it. Other Mojang hosts are refused too.
    #[test]
    fn a_textures_map_naming_another_host_gets_the_default_skin() {
        use base64::Engine as _;
        let encode = |url: &str| {
            let payload = format!(r#"{{"textures":{{"SKIN":{{"url":"{url}"}}}}}}"#);
            base64::engine::general_purpose::STANDARD.encode(payload)
        };
        let smuggled = r"http://evil.example\@textures.minecraft.net/texture/abc";
        // Negative control: authlib's rule alone (the gate before this change) accepts it.
        assert!(is_authlib_texture_domain(smuggled));
        // The payload carries exactly that URL (the backslash is JSON-escaped), so a `None`
        // below comes from the host rule, not from a JSON error.
        let payload = base64::engine::general_purpose::STANDARD
            .decode(encode(&smuggled.replace('\\', "\\\\")))
            .expect("base64");
        let payload: serde_json::Value = serde_json::from_slice(&payload).expect("JSON");
        assert_eq!(payload["textures"]["SKIN"]["url"], smuggled);
        for url in [
            smuggled.replace('\\', "\\\\"),
            "https://api.mojang.com/texture/abc".to_owned(),
            "http://textures.minecraft.net:8080/texture/abc".to_owned(),
        ] {
            assert_eq!(decode_textures_property(&encode(&url)), None, "{url}");
        }
        let fine = decode_textures_property(&encode("http://textures.minecraft.net/texture/abc"))
            .expect("Mojang's texture host");
        assert_eq!(
            fine.skin_url.as_deref(),
            Some("http://textures.minecraft.net/texture/abc")
        );
    }

    #[test]
    fn skin_keys_are_namespaced_away_from_resource_paths() {
        let key = skin_texture_key(uuid::Uuid::nil());
        assert!(key.starts_with("mc-render/entity/skin/"));
        assert!(!key.ends_with(".png"));
    }

    /// java: every `ADD_PLAYER` creates a fresh `NetworkPlayerInfo`, so a changed signed profile
    /// cannot remain hidden behind the previous UUID's already-uploaded texture.
    #[test]
    fn replacement_profiles_get_distinct_skin_keys_but_repeats_are_stable() {
        let mut state = AppearanceState::new();
        let uuid = uuid::Uuid::from_u128(7);
        let first = state.player_skin_key(uuid, "profile-a");
        assert_eq!(state.player_skin_key(uuid, "profile-a"), first);
        let replacement = state.player_skin_key(uuid, "profile-b");
        assert_ne!(replacement, first);
        assert_eq!(state.player_skin_key(uuid, "profile-b"), replacement);
    }

    #[test]
    fn equal_texture_hashes_reuse_skin_keys_across_uuids() {
        let mut state = AppearanceState::new();
        let first = state.player_skin_key_for_url(
            uuid::Uuid::from_u128(1),
            "profile-a",
            "https://textures.minecraft.net/texture/shared-hash",
        );
        let second = state.player_skin_key_for_url(
            uuid::Uuid::from_u128(2),
            "profile-b",
            "https://textures.minecraft.net/texture/shared-hash",
        );
        assert_eq!(first, second, "SkinManager identity is the texture hash");

        let changed = state.player_skin_key_for_url(
            uuid::Uuid::from_u128(2),
            "profile-c",
            "https://textures.minecraft.net/texture/other-hash",
        );
        assert_ne!(second, changed, "a new texture hash is a new skin identity");
        assert_eq!(
            skin_texture_key_for_url("https://textures.minecraft.net/texture/shared-hash"),
            Some(first)
        );
        assert_eq!(
            skin_texture_key_for_url("https://textures.minecraft.net/texture/../"),
            None
        );
        assert_eq!(
            cape_texture_key_for_url("https://textures.minecraft.net/texture/shared-hash"),
            Some("mc-render/entity/cape/shared-hash".to_owned())
        );
    }

    /// Players who left take their chase history and profile keys
    /// with them; players still here keep theirs untouched; a returning player gets the same
    /// URL-derived key back; generation keys (not derivable again) are kept.
    #[test]
    fn departed_players_leave_no_per_player_state_behind() {
        let mut state = AppearanceState::new();
        let url = |n: u128| format!("https://textures.minecraft.net/texture/hash{n}");
        for n in 0..50u128 {
            let uuid = uuid::Uuid::from_u128(n);
            state.tick_cape_chase(uuid, 1, DVec3::new(n as f64, 64.0, 0.0));
            state.player_skin_key_for_url(uuid, &format!("profile-{n}"), &url(n));
            state.player_cape_key_for_url(uuid, &format!("profile-{n}"), &url(n));
        }
        // A UUID whose skin URL had no texture hash got a generation key.
        let odd = uuid::Uuid::from_u128(1000);
        let generation_key = state.player_skin_key(odd, "profile-odd");
        assert_eq!(state.per_player_entries(), 50 * 3 + 1);
        let stayed_chase = state.cape_chase(uuid::Uuid::from_u128(3)).expect("history");

        // Only players 0..5 are still online.
        state.retain_players(|uuid| uuid.as_u128() < 5);
        assert_eq!(
            state.per_player_entries(),
            5 * 3 + 1,
            "the departed 45 left nothing"
        );
        assert_eq!(
            state.cape_chase(uuid::Uuid::from_u128(3)),
            Some(stayed_chase),
            "a player still here keeps their history exactly"
        );
        assert!(state.cape_chase(uuid::Uuid::from_u128(30)).is_none());
        assert_eq!(
            state.player_skin_key(odd, "profile-odd"),
            generation_key,
            "a generation key is kept, so a return cannot mint a second upload"
        );

        // A departed player who comes back resolves the same texture key as before.
        let back = uuid::Uuid::from_u128(30);
        assert_eq!(
            state.player_skin_key_for_url(back, "profile-30", &url(30)),
            skin_texture_key_for_url(&url(30)).expect("hash key")
        );
    }

    /// java: `NetworkPlayerInfo.getSkinType` falls back to `DefaultPlayerSkin.isSlimSkin` until
    /// a trusted profile supplies a loaded skin model.
    #[test]
    fn player_model_uses_the_uuid_default_without_a_profile() {
        let uuid = uuid::Uuid::from_u128(7);
        assert_eq!(
            player_model(&crate::sync::PlayerList::new(), uuid),
            mc_render::entity::SkinModel::default_for_uuid(uuid.as_u128())
        );
    }

    #[test]
    fn cape_chase_interpolates_remote_trail_and_resets_on_profile_replacement() {
        let mut state = AppearanceState::new();
        let uuid = uuid::Uuid::from_u128(17);
        let origin = DVec3::ZERO;
        let first = state.cape_state(uuid, 0, origin, origin, 0.0, 0.0, 0.0, 0.0, false);
        assert_eq!(first, mc_render::entity::layers::CapeState::default());

        // EntityPlayer's chase spring moves a quarter of the one-block tick delta; the render
        // sample is half-way through that same tick, so the cape trails the body behind it.
        state.tick_cape_chase(uuid, 1, DVec3::new(1.0, 0.0, 0.0));
        let moved = state.cape_state(
            uuid,
            1,
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.5, 0.0, 0.0),
            0.5,
            0.0,
            0.0,
            0.0,
            false,
        );
        // chase = 0 + (0.25 - 0) * 0.5 = 0.125; render x = 0.5.
        assert_eq!(moved.chase_dx, 0.125 - 0.5);

        let old_key = state.player_cape_key(uuid, "profile-a");
        state.tick_cape_chase(uuid, 2, DVec3::new(2.0, 0.0, 0.0));
        let new_key = state.player_cape_key(uuid, "profile-b");
        assert_ne!(new_key, old_key);
        let reset = state.cape_state(
            uuid,
            3,
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            1.0,
            0.0,
            0.0,
            0.0,
            false,
        );
        assert_eq!(
            reset.chase_dx, 0.0,
            "replacement starts a fresh chase history"
        );
    }

    #[test]
    fn cape_state_passes_the_walk_distance_through_and_reset_clears_the_history() {
        let mut state = AppearanceState::new();
        let uuid = uuid::Uuid::from_u128(18);
        state.tick_cape_chase(uuid, 10, DVec3::ZERO);
        state.tick_cape_chase(uuid, 11, DVec3::new(1.0, 0.0, 0.0));
        let walked = state.cape_state(
            uuid,
            11,
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            1.0,
            0.0,
            0.0,
            1.5,
            false,
        );
        // The interpolated distanceWalkedModified is the caller's (the predictor owns Java's).
        assert_eq!(walked.distance_walked, 1.5);
        assert_eq!(walked.chase_dx, 0.25 - 1.0);
        state.reset_capes();
        let reset = state.cape_state(
            uuid,
            12,
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            1.0,
            0.0,
            0.0,
            0.0,
            false,
        );
        assert_eq!(reset.chase_dx, 0.0);
    }

    /// The chase is a per-tick spring: a second advance within one tick is ignored, and sampling
    /// (any number of frames) never moves it.
    #[test]
    fn the_cape_chase_steps_once_per_tick_and_sampling_never_steps_it() {
        let mut state = AppearanceState::new();
        let uuid = uuid::Uuid::from_u128(19);
        state.tick_cape_chase(uuid, 1, DVec3::ZERO);
        state.tick_cape_chase(uuid, 2, DVec3::new(4.0, 0.0, 0.0));
        let after_one = state.cape_chase(uuid).expect("history");
        state.tick_cape_chase(uuid, 2, DVec3::new(8.0, 0.0, 0.0));
        assert_eq!(
            state.cape_chase(uuid),
            Some(after_one),
            "same tick: no second step"
        );
        for frame in 0..10 {
            let _ = state.cape_state(
                uuid,
                2,
                DVec3::new(4.0, 0.0, 0.0),
                DVec3::new(4.0, 0.0, 0.0),
                frame as f32 / 10.0,
                0.0,
                0.0,
                0.0,
                false,
            );
        }
        assert_eq!(
            state.cape_chase(uuid),
            Some(after_one),
            "frames do not step it"
        );
        assert_eq!(after_one.current.x, 1.0);
        assert_eq!(after_one.previous.x, 0.0);
    }

    /// Item 30, "going into F5 while jumping makes the cape flick up". The client tick advances
    /// the history every tick in first person, so the first F5 frame mid-jump shows the settled
    /// trailing offset. The negative control replays the old lazy behaviour (the history only
    /// stepped when a render path asked, i.e. never in first person): from a history frozen
    /// where F5 was last used, the same frame swings the cape far up.
    #[test]
    fn entering_f5_mid_jump_shows_the_settled_cape_not_a_flick() {
        use mc_render::entity::layers::cape_angles;
        // Sprint-jumping along +x at 0.28 blocks/tick: vanilla jump arc (0.42 up, then
        // (v - 0.08) * 0.98 per tick), landing on y = 64.
        let mut positions = Vec::new();
        let (mut x, mut y, mut vy) = (0.5_f64, 64.0_f64, 0.0_f64);
        for tick in 0..40 {
            if y <= 64.0 && tick % 12 == 2 {
                vy = 0.42;
            }
            x += 0.28;
            y += vy;
            vy = (vy - 0.08) * 0.98;
            if y < 64.0 {
                y = 64.0;
                vy = 0.0;
            }
            positions.push(DVec3::new(x, y, 0.5));
        }
        let uuid = uuid::Uuid::from_u128(20);
        // Facing +x: java yaw -90 (sin(-90) = -1 -> d3 = -1, d4 = -cos = 0).
        let body_yaw = -90.0;
        let mut ticked = AppearanceState::new();
        // F5 was last used at tick 20, 5.3 blocks back (under the ten-block snap).
        let mut lazy = CapeChase::new(positions[20], 20);
        for (tick, &position) in positions.iter().enumerate() {
            ticked.tick_cape_chase(uuid, tick as u64, position);
        }
        // F5 pressed at tick 39 (mid-air), frame half-way through the tick.
        let f5 = 39;
        let pt = 0.5_f32;
        let render = positions[f5 - 1] + (positions[f5] - positions[f5 - 1]) * f64::from(pt);
        let settled = ticked.cape_state(
            uuid,
            f5 as u64,
            positions[f5],
            render,
            pt,
            body_yaw,
            0.0,
            0.0,
            false,
        );
        lazy.tick(positions[f5], f5 as u64);
        let flicked = lazy.state(render, pt, body_yaw, 0.0, 0.0, false);
        let (settled_x, _, _) = cape_angles(&settled);
        let (flicked_x, _, _) = cape_angles(&flicked);
        // A 0.28 b/t sprint trails the chase by ~0.84 blocks: f2/2 ~ 42 degrees.
        assert!(
            (20.0..70.0).contains(&settled_x),
            "settled sprint-jump cape angle: {settled_x}"
        );
        assert!(
            flicked_x > 200.0,
            "the stale-history negative control must flick the cape: {flicked_x}"
        );
    }

    #[test]
    fn cape_cloth_keeps_chase_inputs_and_samples_stable_tick_time() {
        use mc_render::entity::layers::{CapeCloth, CapeClothStyle, CapeState};
        let state = CapeState {
            chase_dx: -0.6,
            chase_dy: 0.3,
            chase_dz: 0.45,
            body_yaw: 37.0,
            camera_yaw: 0.08,
            distance_walked: 3.3,
            sneaking: true,
            ..CapeState::default()
        };
        let uuid = Uuid::from_u128(42);
        assert_eq!(CapeMotion::Vanilla.apply(state, uuid, 120, 0.5), state);
        let motion = CapeMotion::Cloth(CapeCloth {
            style: CapeClothStyle::Wavy,
            time_seconds: 0.0,
            phase: 0.0,
            strength: 0.6,
            speed: 1.0,
            stiffness: 0.5,
        });
        let sample = motion.apply(state, uuid, 120, 0.5);
        let cloth = sample.cloth.unwrap();
        assert!((cloth.time_seconds - 6.025).abs() < 1e-9);
        assert_eq!(sample, motion.apply(state, uuid, 120, 0.5));
        assert_ne!(
            cloth.phase,
            motion
                .apply(state, Uuid::from_u128(43), 120, 0.5)
                .cloth
                .unwrap()
                .phase
        );
        assert_eq!(
            CapeState {
                cloth: None,
                ..sample
            },
            state
        );
        let next = motion.apply(state, uuid, 121, 0.0).cloth.unwrap();
        assert!((next.time_seconds - cloth.time_seconds - 0.025).abs() < 1e-9);
        assert_eq!(next.phase, cloth.phase);
        assert!(motion
            .apply(state, uuid, 1, f32::NAN)
            .cloth
            .unwrap()
            .time_seconds
            .is_finite());
    }
}
#[test]
fn renderer_names_strip_only_vanilla_formatting_codes() {
    assert_eq!(strip_formatting_codes("§aToast"), "Toast");
    assert_eq!(strip_formatting_codes("§lToast§r"), "Toast");
    assert_eq!(strip_formatting_codes("§ATOAST"), "TOAST");
    assert_eq!(strip_formatting_codes(" Toast "), " Toast ");
    assert_eq!(strip_formatting_codes("§xToast§"), "§xToast§");
}

/// ENTM-6: the borrowing stripper (now also behind `strip_formatting_codes`) agrees with the
/// previous owned loop, kept here verbatim as the oracle, on every edge it has (valid, invalid
/// and dangling codes, case, multibyte text) — both as a string and as an in-place comparison.
#[test]
fn borrowed_formatting_strip_compares_like_the_owned_strip() {
    fn strip_formatting_codes_reference(input: &str) -> String {
        let mut output = String::with_capacity(input.len());
        let mut chars = input.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '§' {
                if let Some(&code) = chars.peek() {
                    if matches!(code.to_ascii_lowercase(), '0'..='9' | 'a'..='f' | 'k'..='o' | 'r')
                    {
                        chars.next();
                        continue;
                    }
                }
            }
            output.push(ch);
        }
        output
    }
    let raws = [
        "Dinnerbone",
        "§aDinnerbone",
        "§lGrumm§r",
        "§ADinner§Bbone",
        "§xDinnerbone",
        "Dinnerbone§",
        "§§aDinnerbone",
        "dinnerbone",
        " Grumm",
        "Grumm ",
        "Gr§kumm",
        "§r",
        "",
        "Dinnérbone",
        "§6§lGRUMM",
    ];
    for raw in raws {
        let reference = strip_formatting_codes_reference(raw);
        assert_eq!(strip_formatting_codes(raw), reference, "{raw:?}");
        for name in ["Dinnerbone", "Grumm", "GRUMM", "", "§xDinnerbone"] {
            assert_eq!(
                formatting_stripped(raw).eq(name.chars()),
                reference == name,
                "{raw:?} vs {name:?}"
            );
        }
    }
}
