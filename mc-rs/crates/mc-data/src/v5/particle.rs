//! 1.7.10 particle names (`S2APacketParticles` carries a string) resolved to the client's 1.8.9
//! particle ids (`EnumParticleTypes`).
//!
//! 1.7.10 picks the particle with an `equals` chain over 35 plain names and three prefixed
//! families, `iconcrack_<item>[_<meta>]`, `blockcrack_<block>_<meta>` and
//! `blockdust_<block>_<meta>`, whose arguments are inside the name (java17:
//! `RenderGlobal.doSpawnParticle`, `client/renderer/RenderGlobal.java:1836-2078`). Every plain
//! name is the `getParticleName()` of a 1.8.9 constant. A name outside the chain spawns nothing
//! (1.8.9 would show its barrier particle for an unknown id, java18:
//! `S2APacketParticles.readPacketData`). A prefixed name 1.7.10 cannot parse throws out of the
//! packet handler and crashes vanilla whenever it gets as far as the parse (java17:
//! `RenderGlobal.spawnParticle` rethrows it as a `ReportedException`,
//! `NetHandlerPlayClient.handleParticles` does not catch it); a particle further than 16 blocks
//! away, or skipped by the particle setting (always at "minimal", one time in three at
//! "decreased"), returns before the parse (java17: `RenderGlobal.doSpawnParticle`,
//! `RenderGlobal.java:1840-1880`). The client drops the packet either way, which looks the same
//! as every outcome that does not crash.

use super::block::state_from_v5_id_meta;
use super::item::is_item_registered_v5;
use crate::item::ItemId;

/// What a 1.7.10 particle name spawns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleV5 {
    /// A particle: its 1.8.9 id and arguments.
    Spawn(Particle5),
    /// A name 1.7.10 does not know. It spawns nothing.
    Unknown,
    /// A prefixed name 1.7.10 cannot parse (a missing or non-numeric field, or an item id it
    /// does not register). Vanilla 1.7.10 crashes on it when the particle passes its distance
    /// and setting checks; the packet is dropped instead.
    Malformed,
}

/// A particle in the 1.8.9 model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Particle5 {
    /// The `EnumParticleTypes` id, `0..=38`.
    pub id: i32,
    args: [i32; 2],
    arg_count: u8,
    /// Whether 1.7.10 adds it without the 16-block distance and particle-setting checks: only
    /// `hugeexplosion`, `largeexplode` and `fireworksSpark` (java17:
    /// `RenderGlobal.doSpawnParticle:1840-1880`), which is not 1.8.9's `getShouldIgnoreRange`
    /// set. A protocol-5 packet has no long-distance flag.
    pub ignore_range: bool,
}

impl Particle5 {
    /// The arguments 1.8.9 carries after the id: `[item, meta]` for `ITEM_CRACK`, the block
    /// state id (`Block.getStateId` packing) for `BLOCK_CRACK` and `BLOCK_DUST`, none otherwise.
    #[inline]
    #[must_use]
    pub fn args(&self) -> &[i32] {
        &self.args[..usize::from(self.arg_count)]
    }

    const fn plain(id: i32) -> Self {
        Self {
            id,
            args: [0; 2],
            arg_count: 0,
            ignore_range: ignores_range_v5(id),
        }
    }
}

/// `ITEM_CRACK`, `BLOCK_CRACK`, `BLOCK_DUST`.
const ITEM_CRACK: i32 = 36;
const BLOCK_CRACK: i32 = 37;
const BLOCK_DUST: i32 = 38;

/// The 35 plain names of java17 `RenderGlobal.doSpawnParticle`, with their 1.8.9 ids (java18:
/// `EnumParticleTypes`).
const PLAIN: [(&str, i32); 35] = [
    ("explode", 0),
    ("largeexplode", 1),
    ("hugeexplosion", 2),
    ("fireworksSpark", 3),
    ("bubble", 4),
    ("splash", 5),
    ("wake", 6),
    ("suspended", 7),
    ("depthsuspend", 8),
    ("crit", 9),
    ("magicCrit", 10),
    ("smoke", 11),
    ("largesmoke", 12),
    ("spell", 13),
    ("instantSpell", 14),
    ("mobSpell", 15),
    ("mobSpellAmbient", 16),
    ("witchMagic", 17),
    ("dripWater", 18),
    ("dripLava", 19),
    ("angryVillager", 20),
    ("happyVillager", 21),
    ("townaura", 22),
    ("note", 23),
    ("portal", 24),
    ("enchantmenttable", 25),
    ("flame", 26),
    ("lava", 27),
    ("footstep", 28),
    ("cloud", 29),
    ("reddust", 30),
    ("snowballpoof", 31),
    ("snowshovel", 32),
    ("slime", 33),
    ("heart", 34),
];

/// Whether a 1.7.10 client adds particle `id` without the distance and particle-setting
/// checks, for particles it spawns itself as well as for packets: `largeexplode` (1),
/// `hugeexplosion` (2) and `fireworksSpark` (3).
#[inline]
#[must_use]
pub const fn ignores_range_v5(id: i32) -> bool {
    matches!(id, 1..=3)
}

/// Resolves a 1.7.10 particle name.
///
/// A block particle's state goes through the 1.7.10 block decode
/// ([`super::state_from_v5_id_meta`]), so its texture is the block 1.7.10 would draw.
///
/// # Example
/// ```
/// use mc_data::v5::{particle_from_v5_name, ParticleV5};
///
/// let ParticleV5::Spawn(p) = particle_from_v5_name("iconcrack_276_3") else { panic!() };
/// assert_eq!((p.id, p.args()), (36, &[276, 3][..]));
/// assert_eq!(particle_from_v5_name("barrier"), ParticleV5::Unknown); // 1.8.9 only
/// assert_eq!(particle_from_v5_name("blockcrack_1"), ParticleV5::Malformed);
/// ```
#[must_use]
pub fn particle_from_v5_name(name: &str) -> ParticleV5 {
    if let Some(&(_, id)) = PLAIN.iter().find(|&&(plain, _)| plain == name) {
        return ParticleV5::Spawn(Particle5::plain(id));
    }
    let parsed = if name.starts_with("iconcrack_") {
        icon_crack(name)
    } else if name.starts_with("blockcrack_") {
        block_particle(name, BLOCK_CRACK)
    } else if name.starts_with("blockdust_") {
        block_particle(name, BLOCK_DUST)
    } else {
        return ParticleV5::Unknown;
    };
    parsed.map_or(ParticleV5::Malformed, ParticleV5::Spawn)
}

/// `iconcrack_<item>[_<meta>]`: `split("_", 3)`, the item from `Item.getItemById` (null for an
/// unregistered id, which 1.7.10 then dereferences), the meta 0 when absent.
fn icon_crack(name: &str) -> Option<Particle5> {
    let parts = java_split3(name);
    let item = java_parse_int(parts.get(1)?)?;
    let meta = match parts.get(2) {
        Some(meta) => java_parse_int(meta)?,
        None => 0,
    };
    let item_id = u16::try_from(item).ok()?;
    if !is_item_registered_v5(ItemId(item_id)) {
        return None;
    }
    Some(Particle5 {
        id: ITEM_CRACK,
        args: [item, meta],
        arg_count: 2,
        ignore_range: false,
    })
}

/// `blockcrack_<block>_<meta>` / `blockdust_<block>_<meta>`: `split("_", 3)`, both fields
/// required (1.7.10 reads `[2]` without a length check), the block from the defaulted registry
/// (an unknown id is air).
fn block_particle(name: &str, id: i32) -> Option<Particle5> {
    let parts = java_split3(name);
    let block = java_parse_int(parts.get(1)?)?;
    let meta = java_parse_int(parts.get(2)?)?;
    let state = state_from_v5_id_meta(block, meta);
    Some(Particle5 {
        id,
        args: [state.to_java_state_id(), 0],
        arg_count: 1,
        ignore_range: false,
    })
}

/// Java's `s.split("_", 3)`: at most three fields, the last keeping any further `_`, and empty
/// fields kept (a positive limit does not drop trailing empty strings).
fn java_split3(s: &str) -> Vec<&str> {
    s.splitn(3, '_').collect()
}

/// Java's `Integer.parseInt` for the ASCII forms: an optional `+` or `-` and decimal digits,
/// within `i32`. (Java also accepts other Unicode decimal digits; a 1.7.10 server never sends a
/// prefixed name at all, so those count as malformed here.)
fn java_parse_int(s: &str) -> Option<i32> {
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockId, BlockState};

    fn spawn(name: &str) -> Particle5 {
        match particle_from_v5_name(name) {
            ParticleV5::Spawn(p) => p,
            other => panic!("{name}: {other:?}"),
        }
    }

    #[test]
    fn plain_names_map_to_the_1_8_9_ids() {
        for (name, id) in PLAIN {
            let p = spawn(name);
            assert_eq!((p.id, p.args()), (id, &[][..]), "{name}");
        }
        assert!(spawn("hugeexplosion").ignore_range);
        assert!(spawn("fireworksSpark").ignore_range);
        // 1.8.9 ignores range for `explode`; 1.7.10 does not.
        assert!(!spawn("explode").ignore_range);
        // Names are case-sensitive and must match exactly.
        assert_eq!(particle_from_v5_name("Explode"), ParticleV5::Unknown);
        assert_eq!(particle_from_v5_name("explode "), ParticleV5::Unknown);
    }

    #[test]
    fn names_only_1_8_9_has_are_unknown() {
        for name in [
            "barrier",
            "droplet",
            "take",
            "mobappearance",
            "",
            "iconcrack",
        ] {
            assert_eq!(particle_from_v5_name(name), ParticleV5::Unknown, "{name}");
        }
    }

    #[test]
    fn icon_crack() {
        assert_eq!(spawn("iconcrack_276").args(), &[276, 0]);
        assert_eq!(spawn("iconcrack_351_4").args(), &[351, 4]);
        assert_eq!(spawn("iconcrack_+1_-2").args(), &[1, -2]);
        // A 1.7.10-only block item is a registered 1.7.10 item.
        assert_eq!(spawn("iconcrack_9").args(), &[9, 0]);
        for bad in [
            "iconcrack_",
            "iconcrack_x",
            "iconcrack_5_",
            "iconcrack__5",
            "iconcrack_5_1_2",
            "iconcrack_99999999999",
            "iconcrack_409", // prismarine shard: 1.8.9 only, so null in 1.7.10
            "iconcrack_0",
        ] {
            assert_eq!(particle_from_v5_name(bad), ParticleV5::Malformed, "{bad}");
        }
    }

    #[test]
    fn block_particles_go_through_the_block_decode() {
        let p = spawn("blockcrack_1_0");
        assert_eq!((p.id, p.args()), (BLOCK_CRACK, &[1][..]));
        // Torch meta 0 has no 1.8.9 state; 1.7.10 draws a standing torch.
        let torch = BlockState::new(BlockId(50), 5).to_java_state_id();
        assert_eq!(spawn("blockdust_50_0").args(), &[torch]);
        assert_eq!(spawn("blockdust_50_0").id, BLOCK_DUST);
        // An unknown block is air, not an error.
        assert_eq!(spawn("blockcrack_4000_0").args(), &[0]);
        assert_eq!(spawn("blockcrack_-7_3").args(), &[0]);
        for bad in [
            "blockcrack_1",
            "blockdust_1_",
            "blockcrack_a_1",
            "blockdust_1_2_3",
        ] {
            assert_eq!(particle_from_v5_name(bad), ParticleV5::Malformed, "{bad}");
        }
    }

    #[test]
    fn java_int_parsing() {
        assert_eq!(java_parse_int("0"), Some(0));
        assert_eq!(java_parse_int("-2147483648"), Some(i32::MIN));
        assert_eq!(java_parse_int("2147483648"), None);
        assert_eq!(java_parse_int("+"), None);
        assert_eq!(java_parse_int("--1"), None);
        assert_eq!(java_parse_int(" 1"), None);
        assert_eq!(java_split3("a_b_c_d"), vec!["a", "b", "c_d"]);
        assert_eq!(java_split3("a_"), vec!["a", ""]);
    }
}
