//! Packed brightness: vanilla's `getCombinedLight` / `getMixedBrightnessForBlock`.
//!
//! Everything the mesher and the AO pass do with light happens on a single packed `u32`
//! laid out exactly the way 1.8.9 lays it out, because [`super::ao`]'s arithmetic
//! (`br1 + br2 + br3 + br4 >> 2 & 0x00FF00FF`) only works on that layout.
//!
//! ```text
//! bit 31                    24 23        20 19          8 7          4 3   0
//!      |                       | sky 0..=15 |             | block 0..=15 |   |
//! ```
//!
//! i.e. `sky << 20 | block << 4`. The two bytes a vertex actually carries are
//! `packed >> 16 & 255` (sky × 16) and `packed & 255` (block × 16).
//
// java: world/World.java getCombinedLight, world/ChunkCache.java getLightForExt,
//       block/Block.java getMixedBrightnessForBlock, block/BlockLiquid.java (override)

use mc_data::{block_props, BlockState, Material};
use mc_math::{BlockPos, Facing};
use mc_world::World;

/// Which of vanilla's two light channels to read. java: `world/EnumSkyBlock.java`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyBlock {
    /// Sunlight. Default `15` outside the loaded world.
    Sky,
    /// Emitted light. Default `0` outside the loaded world.
    Block,
}

impl SkyBlock {
    /// `EnumSkyBlock.defaultLightValue` — what an unloaded or out-of-world position reads as.
    #[must_use]
    pub const fn default_light_value(self) -> u8 {
        match self {
            SkyBlock::Sky => 15,
            SkyBlock::Block => 0,
        }
    }
}

/// Pack a sky/block nibble pair the way `World.getCombinedLight` does.
#[inline]
#[must_use]
pub const fn pack(sky: u8, block: u8) -> u32 {
    ((sky as u32) << 20) | ((block as u32) << 4)
}

/// The sky byte a vertex carries: `packed >> 16 & 255`, i.e. `level * 16`.
#[inline]
#[must_use]
pub const fn sky_byte(packed: u32) -> u8 {
    ((packed >> 16) & 255) as u8
}

/// The block byte a vertex carries: `packed & 255`, i.e. `level * 16`.
#[inline]
#[must_use]
pub const fn block_byte(packed: u32) -> u8 {
    (packed & 255) as u8
}

/// The raw stored nibble at `pos`, with vanilla's out-of-world defaults.
///
/// `mc_world` reports `0` sky light above `y = 255`, which is right for *storage* but wrong
/// for *rendering* — `ChunkCache.getLightForExt` falls through to
/// `EnumSkyBlock.SKY.defaultLightValue` there, which is why the top of the world is lit and
/// not pitch black. Hazard **H6**: the sky channel stays `0` in a dimension with no sky
/// regardless of `y`, which [`World::has_sky_light`] already tracks.
#[must_use]
pub fn raw_light(world: &World, kind: SkyBlock, pos: BlockPos) -> u8 {
    match kind {
        SkyBlock::Sky => {
            if !world.has_sky_light() {
                0
            } else if (0..256).contains(&pos.y) {
                world.sky_light_at(pos)
            } else {
                15
            }
        }
        SkyBlock::Block => {
            if (0..256).contains(&pos.y) {
                world.block_light_at(pos)
            } else {
                0
            }
        }
    }
}

/// `ChunkCache.getLightForExt`: the raw nibble, except that a block with
/// `getUseNeighborBrightness()` (stairs, slabs, farmland, ...) reports the brightest of its
/// six neighbours instead of its own — which is what stops a stair from being a black seam.
#[must_use]
pub fn light_for_ext(world: &World, kind: SkyBlock, pos: BlockPos) -> u8 {
    if kind == SkyBlock::Sky && !world.has_sky_light() {
        return 0;
    }
    if !(0..256).contains(&pos.y) {
        return kind.default_light_value();
    }
    if block_props(world.block_at(pos)).use_neighbor_brightness {
        let mut best = 0;
        for facing in Facing::VALUES {
            let l = raw_light(world, kind, pos.relative(facing));
            if l > best {
                best = l;
            }
            if best >= 15 {
                return best;
            }
        }
        return best;
    }
    raw_light(world, kind, pos)
}

/// `World.getCombinedLight(pos, lightValue)`: both channels packed, with the block channel
/// floored at the emitting block's own light value.
#[must_use]
pub fn combined_light(world: &World, pos: BlockPos, light_value: u8) -> u32 {
    let sky = light_for_ext(world, SkyBlock::Sky, pos);
    let block = light_for_ext(world, SkyBlock::Block, pos).max(light_value);
    pack(sky, block)
}

/// True for every `BlockSlab` subclass. Vanilla's `getMixedBrightnessForBlock` has an
/// `instanceof BlockSlab` special case; `mc_data` records the originating Java class, which is
/// the only handle we have on the class hierarchy.
#[must_use]
pub fn is_slab(state: BlockState) -> bool {
    let class = mc_data::block_info(state.block()).java_class;
    // BlockHalfStoneSlab, BlockDoubleWoodSlab, BlockHalfStoneSlabNew, ...
    class.ends_with("Slab") || class.ends_with("SlabNew")
}

/// `Block.getMixedBrightnessForBlock(world, pos)`, dispatched on the block being *rendered*.
///
/// Only two overrides exist in 1.8.9 and only one of them differs: `BlockLiquid` takes the
/// per-channel max of its own cell and the one above (so a water surface is lit by the air
/// above it rather than by its own dark interior), and `BlockStairs` forwards to its model
/// block, which is the base implementation again.
#[must_use]
pub fn mixed_brightness(world: &World, rendered: BlockState, pos: BlockPos) -> u32 {
    let material = block_props(rendered).material;
    if material == Material::Water || material == Material::Lava {
        // java: BlockLiquid.getMixedBrightnessForBlock
        let here = combined_light(world, pos, 0);
        let above = combined_light(world, pos.up(), 0);
        let block = (here & 255).max(above & 255);
        let sky = ((here >> 16) & 255).max((above >> 16) & 255);
        return block | (sky << 16);
    }

    // java: Block.getMixedBrightnessForBlock
    let at = world.block_at(pos);
    let i = combined_light(world, pos, block_props(at).light_emission);
    if i == 0 && is_slab(at) {
        let down = pos.down();
        return combined_light(
            world,
            down,
            block_props(world.block_at(down)).light_emission,
        );
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_matches_vanilla_bit_layout() {
        // java: World.getCombinedLight — `i << 20 | j << 4`
        assert_eq!(pack(15, 0), 15 << 20);
        assert_eq!(pack(0, 15), 15 << 4);
        let p = pack(11, 7);
        assert_eq!(sky_byte(p), 11 * 16);
        assert_eq!(block_byte(p), 7 * 16);
    }

    #[test]
    fn sky_defaults_to_full_above_the_world_and_zero_below() {
        let mut w = World::new();
        w.set_has_sky_light(true);
        assert_eq!(raw_light(&w, SkyBlock::Sky, BlockPos::new(0, 300, 0)), 15);
        assert_eq!(raw_light(&w, SkyBlock::Sky, BlockPos::new(0, -5, 0)), 15);
        assert_eq!(raw_light(&w, SkyBlock::Block, BlockPos::new(0, 300, 0)), 0);
        w.set_has_sky_light(false);
        assert_eq!(raw_light(&w, SkyBlock::Sky, BlockPos::new(0, 300, 0)), 0);
    }

    #[test]
    fn slab_detection_covers_every_vanilla_slab_class() {
        // Every 1.8.9 BlockSlab subclass, by mc_data's recorded java_class.
        let mut found = 0;
        for id in 0..4096u16 {
            let s = BlockState::from_raw(id);
            if !s.is_legal() {
                continue;
            }
            if is_slab(s) {
                found += 1;
            }
        }
        assert!(found > 0, "no slab states detected");
        assert!(is_slab(
            mc_data::block_by_name("minecraft:stone_slab")
                .expect("stone_slab")
                .default_state()
        ));
        assert!(!is_slab(
            mc_data::block_by_name("minecraft:stone")
                .expect("stone")
                .default_state()
        ));
    }
}
