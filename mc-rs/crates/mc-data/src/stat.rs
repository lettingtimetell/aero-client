//! Vanilla 1.8.9's immutable `StatList` registry.
//!
//! The table is extracted after `Bootstrap.register()`, when recipe outputs, merged block aliases,
//! achievements, and spawn-egg entity statistics have all registered. The strings here are the
//! exact keys carried by `S37PacketStatistics`; the association tables preserve the row order used
//! by `GuiStats`.

use crate::block::BlockId;
use crate::item::ItemId;

#[path = "generated/stats.rs"]
mod table;

use table::{ALL_STATS, BLOCK_STATS, GENERAL_STAT_INDEXES, ITEM_STATS, MOB_STATS, STAT_IDS_SORTED};

/// The family that registered a statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatKind {
    /// One of `StatList.generalStats`' 48 counters.
    General,
    /// A mined-block counter.
    MineBlock,
    /// A crafted-item counter.
    CraftItem,
    /// A used/placed-item counter.
    UseItem,
    /// A depleted durability counter.
    BreakItem,
    /// Kills of a spawn-egg entity.
    KillEntity,
    /// Deaths caused by a spawn-egg entity.
    EntityKilledBy,
    /// An `AchievementList` one-shot statistic.
    Achievement,
}

/// Vanilla's four `IStatType` value formatters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatFormat {
    /// Locale-US grouped integer.
    Simple,
    /// Ticks rendered as seconds, minutes, hours, days, or years.
    Time,
    /// Centimetres rendered as centimetres, metres, or kilometres.
    Distance,
    /// Damage stored in tenths and rendered with two decimal places.
    Damage,
}

impl StatFormat {
    /// Format an `S37PacketStatistics` value exactly like `StatBase.format`.
    #[must_use]
    pub fn format(self, value: i32) -> String {
        match self {
            Self::Simple => format_grouped_i32(value),
            Self::Time => {
                let seconds = f64::from(value) / 20.0;
                let minutes = seconds / 60.0;
                let hours = minutes / 60.0;
                let days = hours / 24.0;
                let years = days / 365.0;
                if years > 0.5 {
                    format_two_decimals(years, " y")
                } else if days > 0.5 {
                    format_two_decimals(days, " d")
                } else if hours > 0.5 {
                    format_two_decimals(hours, " h")
                } else if minutes > 0.5 {
                    format_two_decimals(minutes, " m")
                } else {
                    format!("{} s", java_plain_double(seconds))
                }
            }
            Self::Distance => {
                let metres = f64::from(value) / 100.0;
                let kilometres = metres / 1_000.0;
                if kilometres > 0.5 {
                    format_two_decimals(kilometres, " km")
                } else if metres > 0.5 {
                    format_two_decimals(metres, " m")
                } else {
                    format!("{value} cm")
                }
            }
            Self::Damage => format_two_decimals(f64::from(value) * 0.1, ""),
        }
    }
}

/// Registry metadata for one `StatBase`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatInfo {
    /// Exact packet/registry id, such as `stat.mineBlock.minecraft.stone`.
    pub id: &'static str,
    /// Translation key used by `StatBase.statName`.
    pub translation_key: &'static str,
    /// Extracted English display text, useful as a deterministic fallback.
    pub display_en_us: &'static str,
    /// Value formatter.
    pub format: StatFormat,
    /// `StatBase.isIndependent`.
    pub independent: bool,
    /// Registry family.
    pub kind: StatKind,
    /// Associated item for crafting/mining/use/break statistics.
    pub item: Option<ItemId>,
}

/// One row in `GuiStats.StatsBlock`, in vanilla registry order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockStatAssociation {
    /// Canonical block after `replaceAllSimilarBlocks` alias merging.
    pub block: BlockId,
    /// Item whose icon/name identifies the row.
    pub item: ItemId,
    /// Mined counter; every block row has one.
    pub mined: &'static str,
    /// Crafted counter when this block item is a recipe/smelting output.
    pub crafted: Option<&'static str>,
    /// Used/placed counter when the block has an item form.
    pub used: Option<&'static str>,
}

/// One row in `GuiStats.StatsItem`, in vanilla item-registry order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemStatAssociation {
    /// Item registry id and row icon.
    pub item: ItemId,
    /// Durability-depletion counter for damageable items.
    pub depleted: Option<&'static str>,
    /// Crafted counter when the item is a recipe/smelting output.
    pub crafted: Option<&'static str>,
    /// Used counter; every item row has one.
    pub used: &'static str,
}

/// One row in `GuiStats.StatsMobsList`, in spawn-egg registry order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MobStatAssociation {
    /// `EntityList` numeric id.
    pub entity_id: u8,
    /// `EntityList` save name used in translation/stat suffixes.
    pub entity_name: &'static str,
    /// `stat.killEntity.<name>`.
    pub killed: &'static str,
    /// `stat.entityKilledBy.<name>`.
    pub killed_by: &'static str,
}

/// Every registered `StatBase`, in live `StatList.allStats` order.
#[must_use]
pub fn all_stats() -> &'static [StatInfo] {
    &ALL_STATS
}

/// Resolve an exact packet stat id without allocation.
#[must_use]
pub fn stat_by_id(id: &str) -> Option<&'static StatInfo> {
    STAT_IDS_SORTED
        .binary_search_by(|(candidate, _)| candidate.cmp(&id))
        .ok()
        .map(|position| &ALL_STATS[STAT_IDS_SORTED[position].1])
}

/// The 48 general statistics in their unsorted on-screen order.
pub fn general_stats() -> impl ExactSizeIterator<Item = &'static StatInfo> + Clone {
    GENERAL_STAT_INDEXES.iter().map(|&index| &ALL_STATS[index])
}

/// Visible block rows in `StatList.objectMineStats` order.
#[must_use]
pub fn block_stats() -> &'static [BlockStatAssociation] {
    &BLOCK_STATS
}

/// Visible non-block item rows in `StatList.itemStats` order.
#[must_use]
pub fn item_stats() -> &'static [ItemStatAssociation] {
    &ITEM_STATS
}

/// Spawn-egg mob rows in `EntityList.entityEggs` order.
#[must_use]
pub fn mob_stats() -> &'static [MobStatAssociation] {
    &MOB_STATS
}

fn format_grouped_i32(value: i32) -> String {
    let negative = value < 0;
    let digits = i64::from(value).abs().to_string();
    let first = digits.len() % 3;
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + usize::from(negative));
    if negative {
        out.push('-');
    }
    for (index, byte) in digits.bytes().enumerate() {
        if index > 0 && index % 3 == first % 3 {
            out.push(',');
        }
        out.push(char::from(byte));
    }
    out
}

fn format_two_decimals(value: f64, suffix: &str) -> String {
    format!("{value:.2}{suffix}")
}

fn java_plain_double(value: f64) -> String {
    let rendered = value.to_string();
    if value.is_finite()
        && value.fract() == 0.0
        && !rendered.contains('.')
        && !rendered.contains('e')
        && !rendered.contains('E')
    {
        format!("{rendered}.0")
    } else {
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_cardinalities_and_special_general_metadata_match_vanilla() {
        assert_eq!(all_stats().len(), 883);
        assert_eq!(general_stats().len(), 48);
        assert_eq!(block_stats().len(), 141);
        assert_eq!(item_stats().len(), 186);
        assert_eq!(mob_stats().len(), 27);
        assert_eq!(general_stats().filter(|stat| stat.independent).count(), 17);

        let workbench = stat_by_id("stat.craftingTableInteraction").expect("workbench stat");
        assert_eq!(workbench.translation_key, "stat.workbenchInteraction");
        assert_eq!(workbench.kind, StatKind::General);
        assert_eq!(workbench.format, StatFormat::Simple);
        assert!(stat_by_id("stat.mineBlock.minecraft:stone").is_none());
        assert!(stat_by_id("stat.mineBlock.minecraft.stone").is_some());
    }

    #[test]
    fn row_associations_reference_registered_stats_and_expected_registry_objects() {
        for id in block_stats().iter().flat_map(|row| {
            [Some(row.mined), row.crafted, row.used]
                .into_iter()
                .flatten()
        }) {
            assert!(stat_by_id(id).is_some(), "block row references {id}");
        }
        for id in item_stats().iter().flat_map(|row| {
            [row.depleted, row.crafted, Some(row.used)]
                .into_iter()
                .flatten()
        }) {
            assert!(stat_by_id(id).is_some(), "item row references {id}");
        }
        for row in mob_stats() {
            assert!(stat_by_id(row.killed).is_some());
            assert!(stat_by_id(row.killed_by).is_some());
        }

        let stone = block_stats()
            .iter()
            .find(|row| row.block == BlockId(1))
            .expect("stone row");
        assert_eq!(stone.item, ItemId(1));
        assert_eq!(stone.mined, "stat.mineBlock.minecraft.stone");
        let shovel = item_stats()
            .iter()
            .find(|row| row.item == ItemId(256))
            .expect("iron shovel row");
        assert_eq!(
            shovel.depleted,
            Some("stat.breakItem.minecraft.iron_shovel")
        );
        let zombie = mob_stats()
            .iter()
            .find(|row| row.entity_id == 54)
            .expect("zombie row");
        assert_eq!(zombie.entity_name, "Zombie");
        assert_eq!(zombie.killed, "stat.killEntity.Zombie");
    }

    #[test]
    fn formatters_keep_vanillas_strict_thresholds_and_raw_seconds_branch() {
        assert_eq!(StatFormat::Simple.format(1_234_567), "1,234,567");
        assert_eq!(StatFormat::Simple.format(i32::MIN), "-2,147,483,648");
        assert_eq!(StatFormat::Time.format(20), "1.0 s");
        assert_eq!(StatFormat::Time.format(600), "30.0 s");
        assert_eq!(StatFormat::Time.format(620), "0.52 m");
        assert_eq!(StatFormat::Distance.format(50), "50 cm");
        assert_eq!(StatFormat::Distance.format(51), "0.51 m");
        assert_eq!(StatFormat::Distance.format(50_000), "500.00 m");
        assert_eq!(StatFormat::Distance.format(50_001), "0.50 km");
        assert_eq!(StatFormat::Damage.format(12), "1.20");
    }
}
