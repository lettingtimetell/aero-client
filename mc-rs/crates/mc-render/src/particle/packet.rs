//! `S2APacketParticles` — server-triggered particles.
//!
//! java: `network/play/server/S2APacketParticles.java`, `util/EnumParticleTypes.java`,
//! `client/network/NetHandlerPlayClient.handleParticles`.
//!
//! `mc-render` does not depend on `mc-protocol` (not in this crate's dependency list — see the
//! Wire framing is owned by `mc-protocol`; this module contains the render-side model.
use glam::{DVec3, Vec3};
use mc_data::BlockState;
use mc_math::{BlockPos, Facing, JavaRandom};
use mc_world::World;

use super::spawn::{self, SpellVariant};
use super::system::ParticleSystem;
use super::tuning::ParticleGroup;
use super::types::{Particle, ParticleIcon};

/// One `EnumParticleTypes` entry. java: `util/EnumParticleTypes.java`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParticleTypeInfo {
    /// Wire id.
    pub id: i32,
    /// java: `getParticleName()`
    pub name: &'static str,
    /// java: `getShouldIgnoreRange()` — bypasses both the 16-block range check and the density
    /// gate's "minimal" cutoff. [`ParticleSystem::should_spawn`].
    pub ignore_range: bool,
    /// Extra `VarInt` arguments following the packet's fixed fields.
    pub arg_count: u8,
}

macro_rules! particle_types {
    ($($id:literal => $konst:ident, $name:literal, $ignore:literal, $args:literal;)*) => {
        /// Every vanilla particle type, in wire-id order.
        pub const PARTICLE_TYPES: &[ParticleTypeInfo] = &[
            $(ParticleTypeInfo { id: $id, name: $name, ignore_range: $ignore, arg_count: $args },)*
        ];
        $(
            #[doc = concat!("Wire id for `", $name, "`.")]
            #[allow(dead_code)]
            pub const $konst: i32 = $id;
        )*
    };
}

particle_types! {
    0  => EXPLOSION_NORMAL, "explode", true, 0;
    1  => EXPLOSION_LARGE, "largeexplode", true, 0;
    2  => EXPLOSION_HUGE, "hugeexplosion", true, 0;
    3  => FIREWORKS_SPARK, "fireworksSpark", false, 0;
    4  => WATER_BUBBLE, "bubble", false, 0;
    5  => WATER_SPLASH, "splash", false, 0;
    6  => WATER_WAKE, "wake", false, 0;
    7  => SUSPENDED, "suspended", false, 0;
    8  => SUSPENDED_DEPTH, "depthsuspend", false, 0;
    9  => CRIT, "crit", false, 0;
    10 => CRIT_MAGIC, "magicCrit", false, 0;
    11 => SMOKE_NORMAL, "smoke", false, 0;
    12 => SMOKE_LARGE, "largesmoke", false, 0;
    13 => SPELL, "spell", false, 0;
    14 => SPELL_INSTANT, "instantSpell", false, 0;
    15 => SPELL_MOB, "mobSpell", false, 0;
    16 => SPELL_MOB_AMBIENT, "mobSpellAmbient", false, 0;
    17 => SPELL_WITCH, "witchMagic", false, 0;
    18 => DRIP_WATER, "dripWater", false, 0;
    19 => DRIP_LAVA, "dripLava", false, 0;
    20 => VILLAGER_ANGRY, "angryVillager", false, 0;
    21 => VILLAGER_HAPPY, "happyVillager", false, 0;
    22 => TOWN_AURA, "townaura", false, 0;
    23 => NOTE, "note", false, 0;
    24 => PORTAL, "portal", false, 0;
    25 => ENCHANTMENT_TABLE, "enchantmenttable", false, 0;
    26 => FLAME, "flame", false, 0;
    27 => LAVA, "lava", false, 0;
    28 => FOOTSTEP, "footstep", false, 0;
    29 => CLOUD, "cloud", false, 0;
    30 => REDSTONE, "reddust", false, 0;
    31 => SNOWBALL, "snowballpoof", false, 0;
    32 => SNOW_SHOVEL, "snowshovel", false, 0;
    33 => SLIME, "slime", false, 0;
    34 => HEART, "heart", false, 0;
    35 => BARRIER, "barrier", false, 0;
    36 => ITEM_CRACK, "iconcrack_", false, 2;
    37 => BLOCK_CRACK, "blockcrack_", false, 1;
    38 => BLOCK_DUST, "blockdust_", false, 1;
    39 => WATER_DROP, "droplet", false, 0;
    40 => ITEM_TAKE, "take", false, 0;
    41 => MOB_APPEARANCE, "mobappearance", true, 0;
}

/// Looks up a particle type by wire id. java: `EnumParticleTypes.getParticleFromId`
#[must_use]
pub fn type_by_id(id: i32) -> Option<&'static ParticleTypeInfo> {
    PARTICLE_TYPES.iter().find(|t| t.id == id)
}

/// The decoded packet. java: `S2APacketParticles`'s fields.
#[derive(Debug, Clone, PartialEq)]
pub struct S2APacketParticles {
    /// Falls back to `BARRIER` for an unrecognised id, exactly as `readPacketData` does.
    pub particle_id: i32,
    /// java: `longDistance` — relaxes the client's normal "don't bother if far away" culling.
    /// Distinct from [`ParticleTypeInfo::ignore_range`], which is per-*type*.
    pub long_distance: bool,
    /// Spawn position.
    pub pos: DVec3,
    /// Position/velocity spread. java: `xOffset`/`yOffset`/`zOffset`
    pub offset: Vec3,
    /// java: `particleSpeed`
    pub speed: f32,
    /// `0` means "one particle, offset used as an exact velocity"; `> 0` means "this many,
    /// Gaussian-scattered". java: `particleCount`
    pub count: i32,
    /// Block/item ids and metadata for the parameterised types (`ITEM_CRACK`, `BLOCK_CRACK`,
    /// `BLOCK_DUST`). Length is always [`ParticleTypeInfo::arg_count`] for the packet's type.
    pub args: Vec<i32>,
}

/// java: `S2APacketParticles.writePacketData` — for round-trip tests, and for anything that
/// Resolves the atlas rect for the terrain-layer types (`ITEM_CRACK`, `BLOCK_CRACK`,
/// `BLOCK_DUST`). `particle/**` does not itself decide how a block or item's particle texture is
/// found (that is `mc_assets`/S09 territory — see [`super::phase`]); a caller supplies one.
pub trait IconLookup {
    /// `args[0]` decoded as `Block.getStateId`'s packing. java: `Block.getStateById`
    fn block_icon(&self, state: BlockState) -> Option<ParticleIcon>;
    /// `args[0]` is the item id, `args[1]` (if present) the damage/metadata.
    fn item_icon(&self, item_id: i32, damage: i32) -> Option<ParticleIcon>;
}

/// Builds the [`Particle`] this packet's type describes for one spawn attempt. `speed` is the
/// already-resolved velocity/RGB triple (see [`apply`] for the two ways a packet produces one).
/// `None` means "no dedicated family and no icon available" — dropped silently, matching
/// vanilla's own `catch (Throwable)` around a factory that can legitimately return `null`.
fn build(
    rng: &mut JavaRandom,
    particle_id: i32,
    pos: DVec3,
    speed: DVec3,
    args: &[i32],
    icons: &dyn IconLookup,
) -> Option<Particle> {
    Some(match particle_id {
        EXPLOSION_NORMAL => spawn::explode(rng, pos, speed),
        EXPLOSION_LARGE => spawn::explode_large(rng, pos, speed.x as f32),
        EXPLOSION_HUGE => spawn::huge_explode(rng, pos),
        WATER_BUBBLE => spawn::bubble(rng, pos, speed),
        WATER_SPLASH => spawn::splash(rng, pos, speed),
        CRIT => spawn::crit(rng, pos, speed, 1.0, false),
        CRIT_MAGIC => spawn::crit(rng, pos, speed, 1.0, true),
        SMOKE_NORMAL => spawn::smoke(rng, pos, speed, 1.0),
        SMOKE_LARGE => spawn::smoke(rng, pos, speed, 2.5),
        SPELL => spawn::spell(rng, pos, speed, SpellVariant::Normal),
        SPELL_INSTANT => spawn::spell(rng, pos, speed, SpellVariant::Instant),
        SPELL_MOB => spawn::spell(rng, pos, speed, SpellVariant::Mob),
        SPELL_MOB_AMBIENT => spawn::spell(rng, pos, speed, SpellVariant::AmbientMob),
        SPELL_WITCH => spawn::spell(rng, pos, speed, SpellVariant::Witch),
        DRIP_WATER => spawn::drip(rng, pos, true),
        DRIP_LAVA => spawn::drip(rng, pos, false),
        PORTAL => spawn::portal(rng, pos, speed),
        ENCHANTMENT_TABLE => spawn::enchant_glyph(rng, pos, speed),
        FLAME => spawn::flame(rng, pos, speed),
        LAVA => spawn::lava(rng, pos),
        CLOUD => spawn::cloud(rng, pos, speed),
        REDSTONE => spawn::reddust(
            rng,
            pos,
            speed.x as f32,
            speed.y as f32,
            speed.z as f32,
            1.0,
        ),
        WATER_DROP => spawn::rain(rng, pos),
        FIREWORKS_SPARK => {
            spawn::firework_spark(rng, pos, speed, [1.0, 1.0, 1.0], None, false, false)
        }
        ITEM_CRACK => {
            // The Java factory indexes argument 0 directly; `spawnEffectParticle` catches the
            // resulting exception and returns null when a malformed packet omits it. Metadata
            // is the one optional item argument (`length > 1 ? args[1] : 0`).
            let item_id = args.first().copied()?;
            let damage = args.get(1).copied().unwrap_or(0);
            let icon = icons.item_icon(item_id, damage)?;
            spawn::breaking(rng, pos, speed, icon)
        }
        // java: both factories end in `.func_174845_l()`, which folds `Block.getRenderColor`
        // into the flat `0.6` grey. Without it every leaf, vine, lily pad, tall-grass and stem
        // particle came out plain grey instead of green. `F-P2-client.particle-01-1`,
        // `F-P1-block-19-2`, `F-P1-block-24-5`.
        BLOCK_CRACK => {
            // `Block.getStateById` throws for a metadata the block rejects;
            // `handleParticles` catches it ("Could not spawn particle effect"): no particle.
            let state = BlockState::try_from_java_state_id(args.first().copied()?)?;
            let icon = icons.block_icon(state)?;
            let tint = crate::terrain::tint::digging_tint_from_render_color(state);
            // java: the `BLOCK_CRACK` factory ends in `.func_174845_l()`, whose `sourcePos` is
            // `new BlockPos(this.posX, this.posY, this.posZ)` — the particle's own cell.
            spawn::digging(rng, pos, speed, icon, tint, false, BlockPos::from_vec3(pos))
        }
        BLOCK_DUST => {
            // As BLOCK_CRACK: a rejected metadata throws in the factory and spawns nothing.
            let state = BlockState::try_from_java_state_id(args.first().copied()?)?;
            // EntityBlockDustFX.Factory is the one parameterised block factory that rejects
            // `getRenderType() == -1` before constructing anything.  EntityDiggingFX.Factory
            // (BLOCK_CRACK) does not make this check, so keep the guard local to BLOCK_DUST.
            if mc_data::block_props(state).render_type == mc_data::RenderType::Invisible {
                return None;
            }
            let icon = icons.block_icon(state)?;
            let tint = crate::terrain::tint::digging_tint_from_render_color(state);
            spawn::digging(rng, pos, speed, icon, tint, true, BlockPos::from_vec3(pos))
        }
        SUSPENDED => spawn::suspend(rng, pos),
        SUSPENDED_DEPTH | TOWN_AURA => spawn::aura(rng, pos, speed, false),
        VILLAGER_HAPPY => spawn::aura(rng, pos, speed, true),
        WATER_WAKE => spawn::fish_wake(rng, pos, speed),
        // java: both `EntityHeartFX` factories pass vanilla's `2.0F` default scale.
        HEART => spawn::heart(rng, pos, 2.0, false),
        VILLAGER_ANGRY => spawn::heart(rng, pos, 2.0, true),
        // java: `EntityNoteFX.Factory` — `xSpeedIn` is the note's pitch fraction, not a velocity.
        NOTE => spawn::note(rng, pos, speed.x as f32, 2.0),
        SNOW_SHOVEL => spawn::snow_shovel(rng, pos, speed, 1.0),
        // java: `EntityBreakingFX.SnowballFactory`/`SlimeFactory` — fixed items, five-arg ctor.
        SNOWBALL => spawn::breaking_item(rng, pos, icons.item_icon(SNOWBALL_ITEM_ID, 0)?),
        SLIME => spawn::breaking_item(rng, pos, icons.item_icon(SLIME_BALL_ITEM_ID, 0)?),
        // java: `Barrier.Factory` — `Item.getItemFromBlock(Blocks.barrier)`, item id 166.
        BARRIER => spawn::barrier(rng, pos, icons.item_icon(BARRIER_ITEM_ID, 0)?),
        // `EntityFootStepFX.Factory`: packet speeds are ignored; the particle is a fixed,
        // ground-aligned footprint through its own texture/render path.
        FOOTSTEP => spawn::footstep(rng, pos),
        MOB_APPEARANCE => spawn::mob_appearance(rng, pos),
        // java: `EffectRenderer.registerVanillaParticles` registers **41** of the 42
        // `EnumParticleTypes` ids — `ITEM_TAKE` (40) has no factory, so `spawnEntityFX`'s
        // `particleTypes.get(id)` returns null and `spawnParticle` produces nothing at all. A
        // client that spawns a puff here shows a particle where vanilla shows none.
        // `F-P0-network.play.server-10-5`, `F-P2-client.particle-SWEEP-14`.
        ITEM_TAKE => return None,
        _ => spawn::generic(rng, pos, speed),
    })
}

/// java: `Items.snowball` — `EntityBreakingFX.SnowballFactory`'s fixed item.
const SNOWBALL_ITEM_ID: i32 = 332;
/// java: `Items.slime_ball` — `EntityBreakingFX.SlimeFactory`'s fixed item.
const SLIME_BALL_ITEM_ID: i32 = 341;
/// java: `Item.getItemFromBlock(Blocks.barrier)` — the barrier block's item form.
const BARRIER_ITEM_ID: i32 = 166;

/// One spawn attempt at an exact position and velocity, through the same density/range gate a
/// packet-driven spawn uses. `adjust` runs on the constructed particle before it enters the
/// arena, which is how the two vanilla call sites that mutate a freshly-built `EntityFX`
/// (`RenderGlobal.playAuxSFX`'s potion-smash colour and velocity multiply) are expressed. It runs
/// once per particle actually built: the particles mod ([`super::tuning`]) may build several
/// copies of one attempt, or none.
///
/// Returns `true` if a particle was actually added.
///
/// java: `RenderGlobal.spawnParticle(int, boolean, double...)` -> `spawnEntityFX` (`:2069`).
/// This is the entry point every **client-generated** effect uses — `playAuxSFX`,
/// `randomDisplayTick`, entity trails — as distinct from [`apply`], which is the packet's.
// The argument list mirrors `RenderGlobal.spawnParticle`'s own; bundling it into a struct
// would only move the same nine values one indirection away.
#[allow(clippy::too_many_arguments)]
pub fn spawn_one_with(
    system: &mut ParticleSystem,
    world: &World,
    cam_pos: DVec3,
    particle_id: i32,
    ignore_range: bool,
    pos: DVec3,
    speed: DVec3,
    args: &[i32],
    icons: &dyn IconLookup,
    mut adjust: impl FnMut(&mut Particle),
) -> bool {
    if !system.should_spawn(ignore_range, cam_pos, pos) {
        return false;
    }
    // Without the particles mod's table this is exactly build, adjust, `spawn`, on the system
    // RNG; with it, the table picks how many copies to build and their render size.
    system.spawn_attempt(world, particle_id, |rng| {
        let mut particle = build(rng, particle_id, pos, speed, args, icons)?;
        adjust(&mut particle);
        Some(particle)
    })
}

/// [`spawn_one_with`] with no post-construction adjustment, and with `ignore_range` taken from
/// the type table the way `RenderGlobal`'s private
/// `spawnParticle(EnumParticleTypes, ...)` overload does (`:2064-2067`).
// The argument list mirrors `RenderGlobal.spawnParticle`'s own; bundling it into a struct
// would only move the same nine values one indirection away.
#[allow(clippy::too_many_arguments)]
pub fn spawn_one(
    system: &mut ParticleSystem,
    world: &World,
    cam_pos: DVec3,
    particle_id: i32,
    pos: DVec3,
    speed: DVec3,
    args: &[i32],
    icons: &dyn IconLookup,
) -> bool {
    let ignore_range = type_by_id(particle_id).is_some_and(|t| t.ignore_range);
    spawn_one_with(
        system,
        world,
        cam_pos,
        particle_id,
        ignore_range,
        pos,
        speed,
        args,
        icons,
        |_| {},
    )
}

/// java: `EffectRenderer.addBlockDestroyEffects` (`:369`) — the 4x4x4 = 64-particle burst a
/// broken block leaves behind.
///
/// **This deliberately does not go through [`ParticleSystem::should_spawn`].** Vanilla calls
/// `addEffect` directly here, not `spawnParticle`, so neither the 16-block range check nor the
/// `particleSetting` density gate applies: a block break is visible at any distance and at every
/// particle setting. (`mc-render`'s own `block_destroy_burst_respects_density` integration test
/// asserts the opposite; it is testing the arena, not this function.)
///
/// `tint` is the already-resolved `setBlockPos` colour — `0.6` grey times
/// `Block.colorMultiplier(world, pos)`, or plain [`spawn::DEFAULT_DIGGING_TINT`] for
/// `Blocks.grass`, which vanilla exempts. Returns the number of particles added.
pub fn add_block_destroy_effects(
    system: &mut ParticleSystem,
    world: &World,
    pos: mc_math::BlockPos,
    state: BlockState,
    tint: [f32; 3],
    icons: &dyn IconLookup,
) -> usize {
    if mc_data::block_props(state).is_air() {
        return 0;
    }
    let Some(icon) = icons.block_icon(state) else {
        return 0;
    };
    let mut added = 0;
    for j in 0..4 {
        for k in 0..4 {
            for l in 0..4 {
                let d0 = f64::from(pos.x) + (f64::from(j) + 0.5) / 4.0;
                let d1 = f64::from(pos.y) + (f64::from(k) + 0.5) / 4.0;
                let d2 = f64::from(pos.z) + (f64::from(l) + 0.5) / 4.0;
                let spawn_pos = DVec3::new(d0, d1, d2);
                let speed = DVec3::new(
                    d0 - f64::from(pos.x) - 0.5,
                    d1 - f64::from(pos.y) - 0.5,
                    d2 - f64::from(pos.z) - 0.5,
                );
                // Particles mod: the block group's amount and size. Without its table
                // the vanilla lines below run unchanged.
                if let Some(tuned) = system.tune_group(ParticleGroup::Block, 64) {
                    let made = system.spawn_tuned_deferred(world, tuned, |rng| {
                        Some(spawn::digging(
                            rng, spawn_pos, speed, icon, tint, false, pos,
                        ))
                    });
                    added += made as usize;
                    continue;
                }
                let rng = system.rng_mut();
                // java: `EffectRenderer.addBlockDestroyEffects` calls `.setBlockPos(pos)` with
                // the *block's* position, not the particle's.
                let particle = spawn::digging(rng, spawn_pos, speed, icon, tint, false, pos);
                system.spawn_deferred(world, particle);
                added += 1;
            }
        }
    }
    // One compaction for the 64-particle burst; nothing above reads the arena.
    system.flush_evictions();
    added
}

/// Adds the one shard produced by `EffectRenderer.addBlockHitEffects` while the local player is
/// still mining a block.  This is intentionally a separate path from
/// [`add_block_destroy_effects`]: Java's hit effect is one ranged client-local particle per
/// successful damage tick, while a destroy effect is the direct 64-particle burst and must not
/// be density/range gated or otherwise changed by this helper.
///
/// The random coordinates follow the 1.8.9 implementation exactly.  The two free axes are
/// sampled from the state's render bounds after leaving a `0.1` margin; the struck axis is put
/// just outside the actual face by that same margin.  `EntityDiggingFX.multiplyVelocity(0.2F)`
/// and `multipleParticleScaleBy(0.6F)` are applied after construction, so the inherited
/// constructor jitter still consumes the normal `ParticleSystem` Java RNG stream.
///
/// Returns `true` when a particle was added (the particles mod, [`super::tuning`], may add
/// several or none).  Air, invisible blocks, missing icons, and failed factories all return
/// `false` without consuming particle-constructor RNG.
pub fn add_block_hit_effects(
    system: &mut ParticleSystem,
    world: &World,
    pos: BlockPos,
    face: Facing,
    icons: &dyn IconLookup,
) -> bool {
    let state = world.block_at(pos);
    let props = mc_data::block_props(state);
    if props.is_air() || props.render_type == mc_data::RenderType::Invisible {
        return false;
    }
    let Some(icon) = icons.block_icon(state) else {
        return false;
    };

    const MARGIN: f64 = 0.1;
    let bounds = props.bounds;
    // The tint reads no RNG, so computing it before the position draws keeps vanilla's order of
    // draws: three position doubles, then the digging constructor's jitter.
    let tint = crate::terrain::tint::digging_tint_from_color_multiplier(world, state, pos);
    let make = |rng: &mut JavaRandom| {
        let mut x = f64::from(pos.x)
            + rng.next_f64() * (bounds.max_x - bounds.min_x - 2.0 * MARGIN)
            + MARGIN
            + bounds.min_x;
        let mut y = f64::from(pos.y)
            + rng.next_f64() * (bounds.max_y - bounds.min_y - 2.0 * MARGIN)
            + MARGIN
            + bounds.min_y;
        let mut z = f64::from(pos.z)
            + rng.next_f64() * (bounds.max_z - bounds.min_z - 2.0 * MARGIN)
            + MARGIN
            + bounds.min_z;
        match face {
            Facing::Down => y = f64::from(pos.y) + bounds.min_y - MARGIN,
            Facing::Up => y = f64::from(pos.y) + bounds.max_y + MARGIN,
            Facing::North => z = f64::from(pos.z) + bounds.min_z - MARGIN,
            Facing::South => z = f64::from(pos.z) + bounds.max_z + MARGIN,
            Facing::West => x = f64::from(pos.x) + bounds.min_x - MARGIN,
            Facing::East => x = f64::from(pos.x) + bounds.max_x + MARGIN,
        }
        let spawn_pos = DVec3::new(x, y, z);
        let mut particle = spawn::digging(rng, spawn_pos, DVec3::ZERO, icon, tint, false, pos);
        // java: EntityFX.multiplyVelocity(0.2F).  Y is biased around the inherited 0.1 launch
        // offset rather than scaled around zero.
        particle.motion.x *= 0.2;
        particle.motion.y = (particle.motion.y - 0.1) * 0.2 + 0.1;
        particle.motion.z *= 0.2;
        // java: EntityFX.multipleParticleScaleBy(0.6F) updates both the collision size and the
        // render scale; keeping both fields in lockstep matters to the particle sweep and
        // billboard.
        particle.size.0 *= 0.6;
        particle.size.1 *= 0.6;
        particle.scale *= 0.6;
        particle
    };
    // Particles mod: the block group's amount and size, each copy at its own random
    // point on the face. Without its table the vanilla lines below run unchanged.
    if let Some(tuned) = system.tune_group(ParticleGroup::Block, 1) {
        let added = system.spawn_tuned_deferred(world, tuned, |rng| Some(make(rng))) > 0;
        system.flush_evictions();
        return added;
    }
    let particle = make(system.rng_mut());
    system.spawn(world, particle);
    true
}

/// Applies a decoded packet: the density/range gate, then one or `count` spawn attempts.
/// java: `NetHandlerPlayClient.handleParticles` + `RenderGlobal.spawnParticle`.
pub fn apply(
    system: &mut ParticleSystem,
    world: &World,
    cam_pos: DVec3,
    packet: &S2APacketParticles,
    icons: &dyn IconLookup,
) {
    let Some(info) = type_by_id(packet.particle_id) else {
        return;
    };
    // java: `World.spawnParticle(EnumParticleTypes, boolean, ...)` (`World.java:1126`) passes
    // `particleType.getShouldIgnoreRange() | p_175682_2_` down to `RenderGlobal.spawnParticle`,
    // where `p_175682_2_` is the packet's `longDistance` flag. Until the client actually called
    // `apply`nothing could notice that the `|` was missing here, so an explosion or
    // firework a server marked long-distance was still culled at 16 blocks.
    let ignore_range = info.ignore_range || packet.long_distance;

    if packet.count == 0 {
        let speed = DVec3::new(
            f64::from(packet.speed * packet.offset.x),
            f64::from(packet.speed * packet.offset.y),
            f64::from(packet.speed * packet.offset.z),
        );
        if !system.should_spawn(ignore_range, cam_pos, packet.pos) {
            return;
        }
        // Without the particles mod's table: build, then `spawn`, on the system RNG.
        system.spawn_attempt(world, packet.particle_id, |rng| {
            build(
                rng,
                packet.particle_id,
                packet.pos,
                speed,
                &packet.args,
                icons,
            )
        });
    } else {
        // The particles mod limits how far one burst is multiplied (`tuning::BURST_LIMIT`).
        let burst = usize::try_from(packet.count).unwrap_or(1);
        for _ in 0..packet.count {
            let rng = system.rng_mut();
            let dpos = DVec3::new(
                rng.next_gaussian() * f64::from(packet.offset.x),
                rng.next_gaussian() * f64::from(packet.offset.y),
                rng.next_gaussian() * f64::from(packet.offset.z),
            );
            let speed = DVec3::new(
                rng.next_gaussian() * f64::from(packet.speed),
                rng.next_gaussian() * f64::from(packet.speed),
                rng.next_gaussian() * f64::from(packet.speed),
            );
            let pos = packet.pos + dpos;
            if !system.should_spawn(ignore_range, cam_pos, pos) {
                continue;
            }
            // Without the particles mod's table: build, then `spawn_deferred`.
            system.spawn_attempt_deferred(world, packet.particle_id, burst, |rng| {
                build(rng, packet.particle_id, pos, speed, &packet.args, icons)
            });
        }
        // One compaction per packet burst; the loop above never reads the arena.
        system.flush_evictions();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::types::ParticleKind;

    struct NoIcons;
    impl IconLookup for NoIcons {
        fn block_icon(&self, _state: BlockState) -> Option<ParticleIcon> {
            None
        }
        fn item_icon(&self, _item_id: i32, _damage: i32) -> Option<ParticleIcon> {
            None
        }
    }

    /// java: the BLOCK_CRACK/BLOCK_DUST factories call `Block.getStateById`,
    /// which throws for a metadata the block rejects (powered rail 6 here), and `handleParticles`
    /// catches it ("Could not spawn particle effect"). The old decode panicked: one S2A packet
    /// crashed the client. Found by the live-client packet fuzz.
    #[test]
    fn a_block_particle_with_rejected_metadata_spawns_nothing() {
        let golden_rail = i32::from(mc_data::block_by_name("golden_rail").unwrap().0);
        let mut rng = JavaRandom::new(1);
        let lookup = FixedIcon(full_icon());
        for id in [BLOCK_CRACK, BLOCK_DUST] {
            let rejected = build(
                &mut rng,
                id,
                DVec3::ZERO,
                DVec3::ZERO,
                &[golden_rail | (6 << 12)],
                &lookup,
            );
            assert!(rejected.is_none(), "particle {id}");
        }
    }

    struct FixedIcon(ParticleIcon);
    impl IconLookup for FixedIcon {
        fn block_icon(&self, _state: BlockState) -> Option<ParticleIcon> {
            Some(self.0)
        }
        fn item_icon(&self, _item_id: i32, _damage: i32) -> Option<ParticleIcon> {
            Some(self.0)
        }
    }

    fn full_icon() -> ParticleIcon {
        ParticleIcon {
            min_u: 0.0,
            max_u: 1.0,
            min_v: 0.0,
            max_v: 1.0,
        }
    }

    #[test]
    fn count_zero_spawns_exactly_one_particle_with_exact_velocity() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(1);
        let packet = S2APacketParticles {
            particle_id: FLAME,
            long_distance: false,
            pos: DVec3::new(0.0, 90.0, 0.0),
            offset: Vec3::new(1.0, 0.0, 0.0),
            speed: 0.5,
            count: 0,
            args: vec![],
        };
        // Within the 16-block range check: camera sits right next to the spawn point.
        apply(
            &mut sys,
            &world,
            DVec3::new(0.0, 90.0, 0.0),
            &packet,
            &NoIcons,
        );
        assert_eq!(sys.len(), 1);
    }

    #[test]
    fn count_above_zero_spawns_that_many_gaussian_scattered_particles() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(2);
        let packet = S2APacketParticles {
            particle_id: FLAME,
            long_distance: false,
            pos: DVec3::new(0.0, 90.0, 0.0),
            offset: Vec3::new(0.5, 0.5, 0.5),
            speed: 0.1,
            count: 20,
            args: vec![],
        };
        apply(
            &mut sys,
            &world,
            DVec3::new(0.0, 90.0, 0.0),
            &packet,
            &NoIcons,
        );
        assert_eq!(sys.len(), 20);
        // Spread should not all land on the exact same point.
        let xs: Vec<f64> = sys.particles().iter().map(|p| p.pos.x).collect();
        assert!(xs.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-9));
    }

    #[test]
    fn ignore_range_type_spawns_even_when_minimal_density_and_far_away() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(3);
        sys.set_density(super::super::system::ParticleDensity::Minimal);
        let packet = S2APacketParticles {
            particle_id: EXPLOSION_HUGE,
            long_distance: false,
            pos: DVec3::new(500.0, 90.0, 500.0),
            offset: Vec3::ZERO,
            speed: 0.0,
            count: 0,
            args: vec![],
        };
        apply(&mut sys, &world, DVec3::ZERO, &packet, &NoIcons);
        assert_eq!(
            sys.len(),
            1,
            "EXPLOSION_HUGE ignores both range and density"
        );
    }

    /// java: `World.spawnParticle` (`World.java:1126`) ORs the packet's `longDistance` into the
    /// per-type `getShouldIgnoreRange()`. `FLAME` is a range-respecting type, so the same far-away
    /// packet must be dropped without the flag and kept with it.
    #[test]
    fn long_distance_flag_bypasses_the_range_check_for_a_ranged_type() {
        let world = World::new();
        let far = DVec3::new(0.0, 90.0, 400.0);
        let mut base = S2APacketParticles {
            particle_id: FLAME,
            long_distance: false,
            pos: far,
            offset: Vec3::ZERO,
            speed: 0.0,
            count: 0,
            args: vec![],
        };

        let mut near_sighted = ParticleSystem::with_seed(21);
        apply(&mut near_sighted, &world, DVec3::ZERO, &base, &NoIcons);
        assert_eq!(near_sighted.len(), 0, "400 blocks away, longDistance unset");

        base.long_distance = true;
        let mut far_sighted = ParticleSystem::with_seed(21);
        apply(&mut far_sighted, &world, DVec3::ZERO, &base, &NoIcons);
        assert_eq!(
            far_sighted.len(),
            1,
            "longDistance must bypass the 16-block gate"
        );
    }

    #[test]
    fn block_crack_without_an_icon_is_silently_dropped() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(4);
        let packet = S2APacketParticles {
            particle_id: BLOCK_CRACK,
            long_distance: false,
            pos: DVec3::new(0.0, 90.0, 0.0),
            offset: Vec3::ZERO,
            speed: 0.0,
            count: 0,
            args: vec![
                BlockState::new(mc_data::block_by_name("stone").unwrap(), 0).to_java_state_id()
            ],
        };
        apply(
            &mut sys,
            &world,
            DVec3::new(0.0, 90.0, 0.0),
            &packet,
            &NoIcons,
        );
        assert_eq!(sys.len(), 0);
    }

    #[test]
    fn block_crack_with_an_icon_spawns_a_terrain_layer_particle() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(4);
        let packet = S2APacketParticles {
            particle_id: BLOCK_CRACK,
            long_distance: false,
            pos: DVec3::new(0.0, 90.0, 0.0),
            offset: Vec3::ZERO,
            speed: 0.0,
            count: 0,
            args: vec![
                BlockState::new(mc_data::block_by_name("stone").unwrap(), 0).to_java_state_id()
            ],
        };
        apply(
            &mut sys,
            &world,
            DVec3::new(0.0, 90.0, 0.0),
            &packet,
            &FixedIcon(full_icon()),
        );
        assert_eq!(sys.len(), 1);
        assert_eq!(
            sys.particles()[0].layer,
            super::super::types::FxLayer::Terrain
        );
    }

    #[test]
    fn block_dust_rejects_invisible_render_types_like_java_factory() {
        let world = World::new();
        let air = BlockState::new(mc_data::BlockId::AIR, 0).to_java_state_id();
        let packet = S2APacketParticles {
            particle_id: BLOCK_DUST,
            long_distance: false,
            pos: DVec3::ZERO,
            offset: Vec3::ZERO,
            speed: 0.0,
            count: 0,
            args: vec![air],
        };

        let mut dust = ParticleSystem::with_seed(4);
        apply(
            &mut dust,
            &world,
            DVec3::ZERO,
            &packet,
            &FixedIcon(full_icon()),
        );
        assert_eq!(
            dust.len(),
            0,
            "EntityBlockDustFX returns null for renderType -1"
        );

        // Negative control: EntityDiggingFX.Factory has no render-type guard at this call site;
        // an icon provider can still construct its BLOCK_CRACK particle for the same state.
        let mut crack = ParticleSystem::with_seed(4);
        let mut crack_packet = packet;
        crack_packet.particle_id = BLOCK_CRACK;
        apply(
            &mut crack,
            &world,
            DVec3::ZERO,
            &crack_packet,
            &FixedIcon(full_icon()),
        );
        assert_eq!(crack.len(), 1);
    }

    #[test]
    fn parameterised_factories_drop_packets_with_missing_required_arguments() {
        let world = World::new();
        for particle_id in [ITEM_CRACK, BLOCK_CRACK, BLOCK_DUST] {
            let packet = S2APacketParticles {
                particle_id,
                long_distance: false,
                pos: DVec3::ZERO,
                offset: Vec3::ZERO,
                speed: 0.0,
                count: 0,
                args: vec![],
            };
            let mut system = ParticleSystem::with_seed(12);
            apply(
                &mut system,
                &world,
                DVec3::ZERO,
                &packet,
                &FixedIcon(full_icon()),
            );
            assert_eq!(system.len(), 0, "malformed particle id {particle_id}");
        }

        // Negative control: supplying the required state makes BLOCK_CRACK construct normally,
        // so this is testing the Java factory's missing-array behavior, not a globally disabled
        // parameterised path.
        let valid = S2APacketParticles {
            particle_id: BLOCK_CRACK,
            long_distance: false,
            pos: DVec3::ZERO,
            offset: Vec3::ZERO,
            speed: 0.0,
            count: 0,
            args: vec![BlockState::new(mc_data::BlockId::AIR, 0).to_java_state_id()],
        };
        let mut system = ParticleSystem::with_seed(12);
        apply(
            &mut system,
            &world,
            DVec3::ZERO,
            &valid,
            &FixedIcon(full_icon()),
        );
        assert_eq!(system.len(), 1);
    }

    // ---- the previously-familyless types (W-PARTICLE) --------------------------------------

    /// Spawns one particle of `id` through the real dispatch and hands back its kind.
    fn kind_of(id: i32) -> ParticleKind {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(11);
        let packet = S2APacketParticles {
            particle_id: id,
            long_distance: false,
            pos: DVec3::new(0.0, 90.0, 0.0),
            offset: Vec3::ZERO,
            speed: 1.0,
            count: 0,
            args: vec![],
        };
        apply(
            &mut sys,
            &world,
            DVec3::new(0.0, 90.0, 0.0),
            &packet,
            &FixedIcon(full_icon()),
        );
        assert_eq!(sys.len(), 1, "id {id} produced no particle");
        sys.particles()[0].kind.clone()
    }

    /// The headline of `F-P0-network.play.server-10-4` / `F-P1-util-06-4` /
    /// `F-P2-client.particle-SWEEP-1`: fourteen ids fell through `build`'s catch-all to
    /// `ParticleKind::Generic`, i.e. a white puff at misc-atlas cell (0, 0).
    ///
    /// Every registered packet id reaches its matching family; `ITEM_TAKE` is tested separately
    /// because vanilla deliberately registers no factory for it.
    #[test]
    fn every_wired_particle_type_now_reaches_its_own_family() {
        type Check = fn(&ParticleKind) -> bool;
        let cases: &[(i32, Check, &str)] = &[
            (
                SUSPENDED,
                |k| matches!(k, ParticleKind::Suspend),
                "EntitySuspendFX",
            ),
            (
                SUSPENDED_DEPTH,
                |k| matches!(k, ParticleKind::Aura),
                "EntityAuraFX",
            ),
            (
                TOWN_AURA,
                |k| matches!(k, ParticleKind::Aura),
                "EntityAuraFX",
            ),
            (
                VILLAGER_HAPPY,
                |k| matches!(k, ParticleKind::Aura),
                "EntityAuraFX.HappyVillager",
            ),
            (
                WATER_WAKE,
                |k| matches!(k, ParticleKind::FishWake),
                "EntityFishWakeFX",
            ),
            (
                HEART,
                |k| matches!(k, ParticleKind::Heart { .. }),
                "EntityHeartFX",
            ),
            (
                VILLAGER_ANGRY,
                |k| matches!(k, ParticleKind::Heart { .. }),
                "EntityHeartFX.Angry",
            ),
            (
                NOTE,
                |k| matches!(k, ParticleKind::Note { .. }),
                "EntityNoteFX",
            ),
            (
                SNOW_SHOVEL,
                |k| matches!(k, ParticleKind::SnowShovel { .. }),
                "EntitySnowShovelFX",
            ),
            (
                SNOWBALL,
                |k| matches!(k, ParticleKind::Breaking),
                "EntityBreakingFX.Snowball",
            ),
            (
                SLIME,
                |k| matches!(k, ParticleKind::Breaking),
                "EntityBreakingFX.Slime",
            ),
            (BARRIER, |k| matches!(k, ParticleKind::Barrier), "Barrier"),
            (
                FOOTSTEP,
                |k| matches!(k, ParticleKind::Footstep),
                "EntityFootStepFX",
            ),
            (
                MOB_APPEARANCE,
                |k| matches!(k, ParticleKind::MobAppearance),
                "MobAppearance (elder guardian)",
            ),
        ];
        for &(id, ok, what) in cases {
            let kind = kind_of(id);
            assert!(ok(&kind), "id {id} ({what}) built {kind:?}");
        }
    }

    /// `EntityFootStepFX.Factory` deliberately ignores all three packet speed components: its
    /// custom renderer places a fixed ground quad and its constructor clears the inherited
    /// randomised motion.  This goes through the real S2A dispatch rather than testing
    /// `spawn::footstep` in isolation, so a future generic-factory fallback cannot silently
    /// reintroduce the white-puff behaviour.  `F-P2-client.particle-02-6`.
    #[test]
    fn footstep_packet_ignores_velocity_and_builds_the_fixed_footprint_family() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(23);
        let packet = S2APacketParticles {
            particle_id: FOOTSTEP,
            long_distance: false,
            pos: DVec3::new(12.25, 64.5, -3.75),
            offset: Vec3::ZERO,
            speed: 7.0,
            count: 0,
            args: vec![],
        };
        apply(
            &mut sys,
            &world,
            DVec3::new(12.25, 64.5, -3.75),
            &packet,
            &FixedIcon(full_icon()),
        );

        assert_eq!(sys.len(), 1);
        let footstep = &sys.particles()[0];
        assert!(matches!(&footstep.kind, ParticleKind::Footstep));
        assert_eq!(footstep.pos, packet.pos);
        assert_eq!(footstep.motion, DVec3::ZERO);
        assert_eq!(footstep.max_age, 200);
        assert_eq!(footstep.layer, super::super::types::FxLayer::Footprint);
    }

    /// The `VILLAGER_HAPPY` / `SUSPENDED_DEPTH` split is a texture index and a colour, not a
    /// family, so `every_wired_particle_type_now_reaches_its_own_family` cannot see it.
    /// java: `EntityAuraFX.HappyVillagerFactory` — index 82 and white.
    #[test]
    fn happy_villager_is_a_white_aura_at_texture_index_82() {
        let happy = {
            let world = World::new();
            let mut sys = ParticleSystem::with_seed(11);
            let packet = S2APacketParticles {
                particle_id: VILLAGER_HAPPY,
                long_distance: false,
                pos: DVec3::new(0.0, 90.0, 0.0),
                offset: Vec3::ZERO,
                speed: 1.0,
                count: 0,
                args: vec![],
            };
            apply(
                &mut sys,
                &world,
                DVec3::new(0.0, 90.0, 0.0),
                &packet,
                &NoIcons,
            );
            sys.particles()[0].clone()
        };
        assert_eq!((happy.tex_x, happy.tex_y), (2, 5), "82 == row 5, column 2");
        assert_eq!(happy.color, [1.0, 1.0, 1.0, 1.0]);

        // Negative control: plain TOWN_AURA keeps index 0 and the grey the ctor drew.
        let plain = {
            let world = World::new();
            let mut sys = ParticleSystem::with_seed(11);
            let packet = S2APacketParticles {
                particle_id: TOWN_AURA,
                long_distance: false,
                pos: DVec3::new(0.0, 90.0, 0.0),
                offset: Vec3::ZERO,
                speed: 1.0,
                count: 0,
                args: vec![],
            };
            apply(
                &mut sys,
                &world,
                DVec3::new(0.0, 90.0, 0.0),
                &packet,
                &NoIcons,
            );
            sys.particles()[0].clone()
        };
        assert_eq!((plain.tex_x, plain.tex_y), (0, 0));
        assert!(
            plain.color[0] >= 0.2 && plain.color[0] < 0.3,
            "rand * 0.1 + 0.2"
        );
    }

    /// java: `EffectRenderer.registerVanillaParticles` registers 41 factories and skips
    /// `ITEM_TAKE` (40), so `spawnEntityFX` finds none and adds nothing.
    /// `F-P0-network.play.server-10-5`, `F-P2-client.particle-SWEEP-14`.
    ///
    /// The negative control is `SNOWBALL` run through the identical call: a type whose factory
    /// *does* exist still spawns, so this cannot pass because `apply` stopped working.
    #[test]
    fn item_take_spawns_nothing_because_vanilla_registers_no_factory_for_it() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(3);
        for (id, expected) in [(ITEM_TAKE, 0usize), (SNOWBALL, 1usize)] {
            sys.clear();
            let packet = S2APacketParticles {
                particle_id: id,
                long_distance: false,
                pos: DVec3::new(0.0, 90.0, 0.0),
                offset: Vec3::ZERO,
                speed: 0.0,
                count: 0,
                args: vec![],
            };
            apply(
                &mut sys,
                &world,
                DVec3::new(0.0, 90.0, 0.0),
                &packet,
                &FixedIcon(full_icon()),
            );
            assert_eq!(sys.len(), expected, "particle id {id}");
        }
    }

    /// `F-P2-client.particle-SWEEP-3` / `F-P2-client.particle-03-4`: `EntityReddustFX:24` tests
    /// the parameter that later multiplies `particleRed`, so an `offsetX == 0` packet is the one
    /// vanilla reads as "colour unset".
    ///
    /// Both directions are asserted, which is what makes the old code fail this: with the clamp
    /// on green, case 1 came out green-dominant with red at 0.
    #[test]
    fn reddust_forces_a_zero_red_channel_to_one_and_leaves_green_alone() {
        let mut rng = mc_math::JavaRandom::new(5);
        let p = spawn::reddust(&mut rng, DVec3::ZERO, 0.0, 0.0, 0.0, 1.0);
        assert!(
            p.color[0] > 0.4,
            "a zero red channel must be forced to 1.0, got {:?}",
            p.color
        );
        assert_eq!(p.color[1], 0.0, "green must stay exactly as given");
        assert_eq!(p.color[2], 0.0, "blue must stay exactly as given");

        let mut rng = mc_math::JavaRandom::new(5);
        let p = spawn::reddust(&mut rng, DVec3::ZERO, 1.0, 0.0, 0.0, 1.0);
        assert!(p.color[0] > 0.4);
        assert_eq!(p.color[1], 0.0);
    }

    /// java: `EffectRenderer.addBlockDestroyEffects` calls `addEffect` directly, so neither the
    /// 16-block range check nor `particleSetting` can suppress it. Both are forced hostile here:
    /// `Minimal` density and a camera 1000 blocks away.
    #[test]
    fn a_block_destroy_burst_is_64_particles_and_ignores_range_and_density() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(9);
        sys.set_density(crate::particle::ParticleDensity::Minimal);
        let state = BlockState::new(mc_data::block_by_name("stone").unwrap(), 0);
        let added = add_block_destroy_effects(
            &mut sys,
            &world,
            mc_math::BlockPos::new(1000, 64, 1000),
            state,
            spawn::DEFAULT_DIGGING_TINT,
            &FixedIcon(full_icon()),
        );
        assert_eq!(added, 64, "4x4x4");
        assert_eq!(sys.len(), 64);
        // Negative control: the *gated* path with the same hostile settings adds nothing, so the
        // 64 above cannot be an artefact of the gate having been disabled globally.
        assert!(!spawn_one(
            &mut sys,
            &world,
            DVec3::ZERO,
            SMOKE_NORMAL,
            DVec3::new(1000.0, 64.0, 1000.0),
            DVec3::ZERO,
            &[],
            &FixedIcon(full_icon()),
        ));
    }

    /// Air must produce nothing: java guards on `getMaterial() != Material.air`.
    #[test]
    fn a_block_destroy_burst_on_air_adds_nothing() {
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(9);
        let added = add_block_destroy_effects(
            &mut sys,
            &world,
            mc_math::BlockPos::new(0, 64, 0),
            BlockState::new(mc_data::BlockId(0), 0),
            spawn::DEFAULT_DIGGING_TINT,
            &FixedIcon(full_icon()),
        );
        assert_eq!(added, 0);
    }

    #[test]
    fn block_hit_emits_one_seeded_shard_on_each_face_with_java_modifiers() {
        let pos = BlockPos::new(4, 64, -3);
        let state = BlockState::new(mc_data::block_by_name("stone").unwrap(), 0);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        world.set_block(pos, state);
        let icon = FixedIcon(full_icon());

        for face in Facing::VALUES {
            let seed = 0x5eed_i64 + i64::from(face.index());
            let mut sys = ParticleSystem::with_seed(seed);
            assert!(add_block_hit_effects(&mut sys, &world, pos, face, &icon));
            assert_eq!(sys.len(), 1, "one shard for {face:?}");
            let particle = &sys.particles()[0];
            assert!((particle.size.0 - 0.12).abs() < 1e-6);
            assert!((particle.size.1 - 0.12).abs() < 1e-6);
            assert!((0.3..=0.6).contains(&particle.scale));
            assert!(matches!(
                particle.kind,
                ParticleKind::Digging {
                    is_dust: false,
                    source_pos
                } if source_pos == pos
            ));

            let (min, max) = (pos.to_vec3(), pos.to_vec3() + DVec3::ONE);
            let margin = 0.1;
            match face {
                Facing::Down => {
                    assert_eq!(particle.pos.y, min.y - margin);
                    assert!((min.x + margin..=max.x - margin).contains(&particle.pos.x));
                    assert!((min.z + margin..=max.z - margin).contains(&particle.pos.z));
                }
                Facing::Up => {
                    assert_eq!(particle.pos.y, max.y + margin);
                    assert!((min.x + margin..=max.x - margin).contains(&particle.pos.x));
                    assert!((min.z + margin..=max.z - margin).contains(&particle.pos.z));
                }
                Facing::North => {
                    assert_eq!(particle.pos.z, min.z - margin);
                    assert!((min.x + margin..=max.x - margin).contains(&particle.pos.x));
                    assert!((min.y + margin..=max.y - margin).contains(&particle.pos.y));
                }
                Facing::South => {
                    assert_eq!(particle.pos.z, max.z + margin);
                    assert!((min.x + margin..=max.x - margin).contains(&particle.pos.x));
                    assert!((min.y + margin..=max.y - margin).contains(&particle.pos.y));
                }
                Facing::West => {
                    assert_eq!(particle.pos.x, min.x - margin);
                    assert!((min.y + margin..=max.y - margin).contains(&particle.pos.y));
                    assert!((min.z + margin..=max.z - margin).contains(&particle.pos.z));
                }
                Facing::East => {
                    assert_eq!(particle.pos.x, max.x + margin);
                    assert!((min.y + margin..=max.y - margin).contains(&particle.pos.y));
                    assert!((min.z + margin..=max.z - margin).contains(&particle.pos.z));
                }
            }

            // Re-run the same seeded constructor stream to make the velocity transform an exact
            // oracle rather than merely checking that the shard is non-stationary.
            let bounds = state.props().bounds;
            let mut rng = JavaRandom::new(seed);
            let mut x = f64::from(pos.x)
                + rng.next_f64() * (bounds.max_x - bounds.min_x - 2.0 * margin)
                + margin
                + bounds.min_x;
            let mut y = f64::from(pos.y)
                + rng.next_f64() * (bounds.max_y - bounds.min_y - 2.0 * margin)
                + margin
                + bounds.min_y;
            let mut z = f64::from(pos.z)
                + rng.next_f64() * (bounds.max_z - bounds.min_z - 2.0 * margin)
                + margin
                + bounds.min_z;
            match face {
                Facing::Down => y = f64::from(pos.y) + bounds.min_y - margin,
                Facing::Up => y = f64::from(pos.y) + bounds.max_y + margin,
                Facing::North => z = f64::from(pos.z) + bounds.min_z - margin,
                Facing::South => z = f64::from(pos.z) + bounds.max_z + margin,
                Facing::West => x = f64::from(pos.x) + bounds.min_x - margin,
                Facing::East => x = f64::from(pos.x) + bounds.max_x + margin,
            }
            let mut expected = spawn::digging(
                &mut rng,
                DVec3::new(x, y, z),
                DVec3::ZERO,
                icon.0,
                crate::terrain::tint::digging_tint_from_color_multiplier(&world, state, pos),
                false,
                pos,
            );
            expected.motion.x *= 0.2;
            expected.motion.y = (expected.motion.y - 0.1) * 0.2 + 0.1;
            expected.motion.z *= 0.2;
            expected.size.0 *= 0.6;
            expected.size.1 *= 0.6;
            expected.scale *= 0.6;
            assert_eq!(particle.pos, expected.pos);
            assert_eq!(particle.motion, expected.motion);
            assert_eq!(particle.scale, expected.scale);
        }
    }

    #[test]
    fn block_hit_uses_non_full_bounds_and_rejects_air_invisible_and_missing_icons() {
        let pos = BlockPos::new(0, 64, 0);
        let slab = BlockState::new(mc_data::block_by_name("stone_slab").unwrap(), 0);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        world.set_block(pos, slab);
        let mut sys = ParticleSystem::with_seed(12);
        assert!(add_block_hit_effects(
            &mut sys,
            &world,
            pos,
            Facing::Up,
            &FixedIcon(full_icon())
        ));
        assert_eq!(sys.particles()[0].pos.y, f64::from(pos.y) + 0.5 + 0.1);

        let mut air = World::new();
        air.load_empty_chunk(pos.chunk_pos());
        let mut air_sys = ParticleSystem::with_seed(12);
        assert!(!add_block_hit_effects(
            &mut air_sys,
            &air,
            pos,
            Facing::Up,
            &FixedIcon(full_icon())
        ));
        assert_eq!(air_sys.len(), 0);

        // Piston moving is a legal, non-air state whose Java render type is -1.
        let invisible = BlockState::new(mc_data::BlockId(36), 0);
        air.set_block(pos, invisible);
        let mut invisible_sys = ParticleSystem::with_seed(12);
        assert!(!add_block_hit_effects(
            &mut invisible_sys,
            &air,
            pos,
            Facing::Up,
            &FixedIcon(full_icon())
        ));
        assert_eq!(invisible_sys.len(), 0);

        let mut no_icon_sys = ParticleSystem::with_seed(12);
        assert!(!add_block_hit_effects(
            &mut no_icon_sys,
            &world,
            pos,
            Facing::Up,
            &NoIcons
        ));
        assert_eq!(no_icon_sys.len(), 0);
    }

    #[test]
    fn block_hit_does_not_change_the_separate_destroy_burst() {
        let pos = BlockPos::new(0, 64, 0);
        let state = BlockState::new(mc_data::block_by_name("stone").unwrap(), 0);
        let world = World::new();
        let mut sys = ParticleSystem::with_seed(17);
        assert_eq!(
            add_block_destroy_effects(
                &mut sys,
                &world,
                pos,
                state,
                spawn::DEFAULT_DIGGING_TINT,
                &FixedIcon(full_icon())
            ),
            64
        );
        assert_eq!(
            sys.len(),
            64,
            "destroy remains a 4x4x4 burst, not one hit shard"
        );
    }

    /// `F-P2-client.particle-01-1` / `F-P1-block-19-2` / `F-P1-block-24-5`: both digging
    /// factories end in `func_174845_l()`, which folds `Block.getRenderColor` into the flat grey.
    /// This covers each of the seven `getRenderColor` override families (and both packet
    /// factories), so replacing that call with the default grey cannot quietly regress leaves,
    /// vines, lily pads, tall grass, or stems.
    #[test]
    fn block_particles_fold_get_render_color_into_the_digging_tint() {
        let states = [
            // `BlockOldLeaf`: basic foliage, pine, and birch variants.
            BlockState::new(mc_data::block_by_name("leaves").unwrap(), 0),
            BlockState::new(mc_data::block_by_name("leaves").unwrap(), 1),
            BlockState::new(mc_data::block_by_name("leaves").unwrap(), 2),
            BlockState::new(mc_data::block_by_name("leaves2").unwrap(), 0),
            BlockState::new(mc_data::block_by_name("vine").unwrap(), 0),
            BlockState::new(mc_data::block_by_name("waterlily").unwrap(), 0),
            BlockState::new(mc_data::block_by_name("tallgrass").unwrap(), 1),
            BlockState::new(mc_data::block_by_name("pumpkin_stem").unwrap(), 7),
            BlockState::new(mc_data::block_by_name("melon_stem").unwrap(), 3),
            // `BlockGrass` has an override too, but `EntityDiggingFX` explicitly leaves it grey.
            BlockState::new(mc_data::block_by_name("grass").unwrap(), 0),
        ];
        let stone = BlockState::new(mc_data::block_by_name("stone").unwrap(), 0);
        let world = World::new();
        let make = |particle_id, state: BlockState| S2APacketParticles {
            particle_id,
            long_distance: false,
            pos: DVec3::new(0.0, 90.0, 0.0),
            offset: Vec3::ZERO,
            speed: 0.0,
            count: 0,
            args: vec![state.to_java_state_id()],
        };

        for particle_id in [BLOCK_CRACK, BLOCK_DUST] {
            for state in states {
                let mut sys = ParticleSystem::with_seed(2);
                apply(
                    &mut sys,
                    &world,
                    DVec3::new(0.0, 90.0, 0.0),
                    &make(particle_id, state),
                    &FixedIcon(full_icon()),
                );
                let color = sys.particles()[0].color;
                let expect = crate::terrain::tint::digging_tint_from_render_color(state);
                assert_eq!(
                    [color[0], color[1], color[2]],
                    expect,
                    "particle {particle_id} state {state:?} must retain its render tint"
                );
            }

            let mut sys = ParticleSystem::with_seed(2);
            apply(
                &mut sys,
                &world,
                DVec3::new(0.0, 90.0, 0.0),
                &make(particle_id, stone),
                &FixedIcon(full_icon()),
            );
            let stone_color = sys.particles()[0].color;
            // Negative control: a block with no `getRenderColor` override keeps the flat grey,
            // so this cannot pass by applying an arbitrary tint to every digging particle.
            assert_eq!(
                [stone_color[0], stone_color[1], stone_color[2]],
                spawn::DEFAULT_DIGGING_TINT
            );
        }
    }
}
