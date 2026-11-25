//! 1.7.10 statistic ids (`S37PacketStatistics`) resolved to the client's 1.8.9 statistics.
//!
//! The packet layout is the same in both versions; the ids are not. 1.7.10 names block and item
//! statistics by numeric id, `"stat.mineBlock." + id` (java17: `StatList.initMiningStats`,
//! `initCraftableStats`, `initStats`, `initItemDepleteStats`, `stats/StatList.java:98-165`),
//! where 1.8.9 uses the registry name, `stat.mineBlock.minecraft.stone`, and keeps no numeric
//! alias (java18: `StatList`). Without the translation every block and item statistic from a
//! 1.7.10 server would be dropped as unknown (java18: `S37PacketStatistics.readPacketData` keeps
//! only ids `StatList.getOneShotStat` knows).
//!
//! Every other id (general statistics, achievements, kill counts) has the same string in both
//! versions; 1.7.10 has a subset of them, and an id it does not have is unknown to it and dropped
//! (java17: `S37PacketStatistics.readPacketData` keeps only `StatList.getOneShotStat` hits).

use crate::stat::{stat_by_id, StatInfo, StatKind};

use super::generated::{
    V5_STATS_BREAK_ITEM, V5_STATS_CRAFT_ITEM, V5_STATS_MINE_BLOCK, V5_STATS_USE_ITEM,
};

/// The 23 general statistics 1.7.10 has (java17: `StatList` fields, `stats/StatList.java:27-49`);
/// 1.8.9 adds 25 more.
const GENERAL_1_7_10: [&str; 23] = [
    "stat.leaveGame",
    "stat.playOneMinute",
    "stat.walkOneCm",
    "stat.swimOneCm",
    "stat.fallOneCm",
    "stat.climbOneCm",
    "stat.flyOneCm",
    "stat.diveOneCm",
    "stat.minecartOneCm",
    "stat.boatOneCm",
    "stat.pigOneCm",
    "stat.horseOneCm",
    "stat.jump",
    "stat.drop",
    "stat.damageDealt",
    "stat.damageTaken",
    "stat.deaths",
    "stat.mobKills",
    "stat.animalsBred",
    "stat.playerKills",
    "stat.fishCaught",
    "stat.junkFished",
    "stat.treasureFished",
];

/// The `EntityList` names with spawn eggs, and so kill statistics, in 1.7.10 (java17:
/// `EntityList` static init); 1.8.9 adds the endermite, guardian and rabbit.
const EGG_ENTITIES_1_7_10: [&str; 24] = [
    "Creeper",
    "Skeleton",
    "Spider",
    "Zombie",
    "Slime",
    "Ghast",
    "PigZombie",
    "Enderman",
    "CaveSpider",
    "Silverfish",
    "Blaze",
    "LavaSlime",
    "Bat",
    "Witch",
    "Pig",
    "Sheep",
    "Cow",
    "Chicken",
    "Squid",
    "Wolf",
    "MushroomCow",
    "Ozelot",
    "EntityHorse",
    "Villager",
];

/// The 1.8.9 statistic a 1.7.10 statistic id names, `None` when 1.7.10 has no such id or 1.8.9
/// has no statistic for it.
///
/// A numeric block or item id resolves to the statistic 1.8.9 keeps in the same array slot,
/// after 1.8.9's `replaceAllSimilarBlocks` merges (so `stat.mineBlock.91`, the jack o'lantern,
/// is 1.8.9's pumpkin counter, as both versions merge the two). The 1.7.10-only block items
/// (water, portal, cocoa, carrots, potatoes, ...) have no 1.8.9 statistic and resolve to `None`.
///
/// # Example
/// ```
/// use mc_data::v5::stat_from_v5;
///
/// assert_eq!(stat_from_v5("stat.mineBlock.1").unwrap().id, "stat.mineBlock.minecraft.stone");
/// assert_eq!(stat_from_v5("stat.jump").unwrap().id, "stat.jump");
/// // 1.8.9's own spelling is not a 1.7.10 id.
/// assert!(stat_from_v5("stat.mineBlock.minecraft.stone").is_none());
/// ```
#[must_use]
pub fn stat_from_v5(id: &str) -> Option<&'static StatInfo> {
    if let Some(rest) = id.strip_prefix("stat.") {
        for (prefix, table) in [
            ("mineBlock.", &V5_STATS_MINE_BLOCK[..]),
            ("craftItem.", &V5_STATS_CRAFT_ITEM[..]),
            ("useItem.", &V5_STATS_USE_ITEM[..]),
            ("breakItem.", &V5_STATS_BREAK_ITEM[..]),
        ] {
            if let Some(number) = rest.strip_prefix(prefix) {
                let n = decimal_id(number)?;
                let i = table.binary_search_by_key(&n, |&(id, _)| id).ok()?;
                return stat_by_id(table[i].1);
            }
        }
    }
    let stat = stat_by_id(id)?;
    let known_in_1_7_10 = match stat.kind {
        StatKind::General => GENERAL_1_7_10.contains(&id),
        StatKind::Achievement => id != "achievement.overpowered",
        StatKind::KillEntity | StatKind::EntityKilledBy => id
            .rsplit('.')
            .next()
            .is_some_and(|name| EGG_ENTITIES_1_7_10.contains(&name)),
        // Block and item statistics exist only under their numeric 1.7.10 ids.
        StatKind::MineBlock | StatKind::CraftItem | StatKind::UseItem | StatKind::BreakItem => {
            false
        }
    };
    known_in_1_7_10.then_some(stat)
}

/// A block or item id as `Integer.toString` writes it: `0`, or digits without a leading zero
/// or a sign. 1.7.10 looks ids up by exact string, so any other spelling is unknown.
fn decimal_id(s: &str) -> Option<u16> {
    let canonical =
        !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) && (s == "0" || !s.starts_with('0'));
    if canonical {
        s.parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stat::{all_stats, general_stats};

    #[test]
    fn numeric_block_and_item_ids() {
        let id = |s| stat_from_v5(s).map(|stat| stat.id);
        assert_eq!(
            id("stat.mineBlock.1"),
            Some("stat.mineBlock.minecraft.stone")
        );
        assert_eq!(
            id("stat.useItem.276"),
            Some("stat.useItem.minecraft.diamond_sword")
        );
        assert_eq!(
            id("stat.breakItem.276"),
            Some("stat.breakItem.minecraft.diamond_sword")
        );
        assert_eq!(
            id("stat.craftItem.58"),
            Some("stat.craftItem.minecraft.crafting_table")
        );
        // Item 350's name differs between the versions; the numeric id does not care.
        assert_eq!(
            id("stat.useItem.350"),
            Some("stat.useItem.minecraft.cooked_fish")
        );
        // Merged slots resolve as 1.8.9 leaves them.
        assert_eq!(
            id("stat.mineBlock.91"),
            Some("stat.mineBlock.minecraft.pumpkin")
        );
        assert_eq!(
            id("stat.mineBlock.2"),
            Some("stat.mineBlock.minecraft.dirt")
        );
        assert_eq!(
            id("stat.mineBlock.75"),
            Some("stat.mineBlock.minecraft.redstone_torch")
        );
        // 1.7.10-only block items have no 1.8.9 statistic.
        for missing in [
            "stat.mineBlock.127",
            "stat.mineBlock.141",
            "stat.mineBlock.90",
        ] {
            assert_eq!(id(missing), None, "{missing}");
        }
        // Spellings Integer.toString never produces.
        for bad in [
            "stat.mineBlock.01",
            "stat.mineBlock.+1",
            "stat.mineBlock.",
            "stat.mineBlock.x",
        ] {
            assert_eq!(id(bad), None, "{bad}");
        }
        assert_eq!(id("stat.mineBlock.70000"), None);
    }

    #[test]
    fn shared_and_1_8_9_only_string_ids() {
        for stat in general_stats() {
            let known = GENERAL_1_7_10.contains(&stat.id);
            assert_eq!(stat_from_v5(stat.id).is_some(), known, "{}", stat.id);
        }
        assert_eq!(general_stats().count(), 48);
        assert!(stat_from_v5("stat.crouchOneCm").is_none());
        assert!(stat_from_v5("achievement.openInventory").is_some());
        assert!(stat_from_v5("achievement.overpowered").is_none());
        assert!(stat_from_v5("stat.killEntity.Creeper").is_some());
        assert!(stat_from_v5("stat.entityKilledBy.Villager").is_some());
        assert!(stat_from_v5("stat.killEntity.Rabbit").is_none());
        assert!(stat_from_v5("stat.entityKilledBy.Guardian").is_none());
        assert!(stat_from_v5("stat.notAStat").is_none());
    }

    #[test]
    fn every_table_entry_names_a_registered_1_8_9_statistic_of_its_kind() {
        for (table, kind) in [
            (&V5_STATS_MINE_BLOCK[..], StatKind::MineBlock),
            (&V5_STATS_CRAFT_ITEM[..], StatKind::CraftItem),
            (&V5_STATS_USE_ITEM[..], StatKind::UseItem),
            (&V5_STATS_BREAK_ITEM[..], StatKind::BreakItem),
        ] {
            assert!(table.windows(2).all(|w| w[0].0 < w[1].0), "sorted, unique");
            for &(n, target) in table {
                let stat = stat_by_id(target).unwrap_or_else(|| panic!("{target}"));
                assert_eq!(stat.kind, kind, "{n} -> {target}");
            }
        }
        // Every 1.8.9 block and item statistic is reachable from some numeric id.
        for stat in all_stats() {
            let table: &[(u16, &str)] = match stat.kind {
                StatKind::MineBlock => &V5_STATS_MINE_BLOCK,
                StatKind::CraftItem => &V5_STATS_CRAFT_ITEM,
                StatKind::UseItem => &V5_STATS_USE_ITEM,
                StatKind::BreakItem => &V5_STATS_BREAK_ITEM,
                _ => continue,
            };
            assert!(table.iter().any(|&(_, t)| t == stat.id), "{}", stat.id);
        }
    }
}
