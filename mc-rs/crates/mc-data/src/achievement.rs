//! Achievement table: the static prerequisite tree the (not-yet-ported) `AchievementsScreen`
//! / `AchievementGrid` will render.
//!
//! Java input: `net/minecraft/stats/AchievementList.java` and `net/minecraft/stats/Achievement.java`,
//! read live through the extractor rather than transcribed. `AchievementList`'s static
//! initializer builds each `Achievement` on a chain of mutable calls
//! (`(new Achievement(...)).setSpecial().registerStat()`), so pulling the rows straight out of
//! the registered `AchievementList.achievementList` after `Bootstrap.register()` runs (which
//! reaches `AchievementList.init()` via `StatList.init()`) is both easier and more faithful
//! than parsing the constructor calls out of the source.
//!
//! # This is purely client-side decoration
//!
//! Achievements never cross the network as a distinct wire type: the "Update Statistics"
//! packet server->client carries `(stat id, value)` pairs for *all* stats (achievements and
//! plain counters alike), already handled by the statistics path elsewhere. Nothing here
//! needs to serialise; this module exists only so the achievement screen has real grid
//! positions, icons and a real prerequisite tree to draw, instead of placeholder data.
//!
//! Wiring this into `mc-ui`'s `AchievementsScreen` / `mc-client`'s `Ui` enum is a separate,
//! larger follow-up (plan row `S16 | S21`) - this crate only carries the data.

use crate::generated::achievements::ACHIEVEMENT_INFO;
use crate::item::ItemId;

pub use crate::generated::achievements::Achievement;

/// Display/tree data for one achievement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AchievementInfo {
    /// This row's own enum variant.
    pub achievement: Achievement,
    /// Full stat id (`StatBase.statId`), e.g. `achievement.openInventory`. This is the same
    /// string used as the map key when a "Update Statistics" packet reports unlock progress.
    pub stat_id: &'static str,
    /// Short name (`stat_id` with the `achievement.` prefix stripped), e.g. `openInventory`.
    /// Also the stem `AchievementList` builds the description translation key from
    /// (`achievement.<name>.desc`).
    pub name: &'static str,
    /// Column in the achievement grid, in 24px units, relative to the grid's center.
    /// `Achievement.displayColumn`.
    pub display_column: i8,
    /// Row in the achievement grid, in 24px units, relative to the grid's center.
    /// `Achievement.displayRow`.
    pub display_row: i8,
    /// Prerequisite achievement that must be earned first. `None` only for the tree's root,
    /// `openInventory`.
    pub parent: Option<Achievement>,
    /// Icon item drawn in the achievement grid.
    pub item: ItemId,
    /// Icon item's damage/metadata value, e.g. `1` for the wither skeleton skull
    /// (`spawnWither`) and the enchanted golden apple (`overpowered`) - both share a base item
    /// id with a plain variant, and only the metadata tells them apart.
    pub item_damage: u16,
    /// "Hard" achievements (`Achievement.setSpecial()`): rendered with a spiked frame instead
    /// of the normal square one.
    pub special: bool,
    /// `StatBase.isIndependent`: true only for `openInventory`, the one achievement not gated
    /// behind earning a prerequisite first.
    pub independent: bool,
}

impl Achievement {
    /// Display/tree data for this achievement. `O(1)`, no allocation.
    #[inline]
    #[must_use]
    pub fn info(self) -> &'static AchievementInfo {
        &ACHIEVEMENT_INFO[self as usize]
    }

    /// Prerequisite achievement, or `None` for the root (`openInventory`).
    #[inline]
    #[must_use]
    pub fn parent(self) -> Option<Achievement> {
        self.info().parent
    }

    /// True for "hard" achievements (spiked frame).
    #[inline]
    #[must_use]
    pub fn is_special(self) -> bool {
        self.info().special
    }

    /// Every achievement that lists `self` as its direct parent, in table order.
    pub fn children(self) -> impl Iterator<Item = Achievement> {
        ACHIEVEMENT_INFO
            .iter()
            .filter(move |a| a.parent == Some(self))
            .map(|a| a.achievement)
    }

    /// Walks up the prerequisite chain from `self` to the root, inclusive of `self`.
    pub fn ancestors(self) -> impl Iterator<Item = Achievement> {
        std::iter::successors(Some(self), |a| a.parent())
    }
}

/// Looks up an achievement by its short name (`openInventory`, not `achievement.openInventory`).
///
/// # Example
/// ```
/// use mc_data::achievement_by_name;
/// assert!(achievement_by_name("openInventory").is_some());
/// assert!(achievement_by_name("achievement.openInventory").is_none());
/// assert!(achievement_by_name("notAnAchievement").is_none());
/// ```
#[must_use]
pub fn achievement_by_name(name: &str) -> Option<Achievement> {
    ACHIEVEMENT_INFO
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.achievement)
}

/// Looks up an achievement by its full stat id (`achievement.openInventory`), the string
/// carried as the key of a per-stat entry in the "Update Statistics" packet.
#[must_use]
pub fn achievement_by_stat_id(stat_id: &str) -> Option<Achievement> {
    ACHIEVEMENT_INFO
        .iter()
        .find(|a| a.stat_id == stat_id)
        .map(|a| a.achievement)
}

/// Every achievement, in `AchievementList`'s declaration (topological) order.
#[must_use]
pub fn achievements() -> &'static [AchievementInfo] {
    &ACHIEVEMENT_INFO
}

/// Achievements with no prerequisite. 1.8.9 has exactly one: `openInventory`.
pub fn achievement_roots() -> impl Iterator<Item = &'static AchievementInfo> {
    ACHIEVEMENT_INFO.iter().filter(|a| a.parent.is_none())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_achievement_resolves_without_panic() {
        for a in achievements() {
            let info = a.achievement.info();
            assert_eq!(info.stat_id, a.stat_id);
            assert!(a.stat_id.starts_with("achievement."));
            assert_eq!(&a.stat_id["achievement.".len()..], a.name);
        }
    }

    #[test]
    fn root_is_open_inventory() {
        let roots: Vec<_> = achievement_roots().collect();
        assert_eq!(roots.len(), 1, "1.8.9 has exactly one root achievement");
        assert_eq!(roots[0].name, "openInventory");
        assert!(roots[0].independent);
        assert_eq!(roots[0].parent, None);
    }

    #[test]
    fn known_achievement_round_trips() {
        let open_inv = achievement_by_name("openInventory").expect("openInventory");
        assert_eq!(open_inv.parent(), None);
        assert!(!open_inv.is_special());

        // mineWood's parent is openInventory - a real, non-root parent link.
        let mine_wood = achievement_by_name("mineWood").expect("mineWood");
        assert_eq!(mine_wood.parent(), Some(open_inv));

        // onARail is a "special" (hard) achievement.
        let on_a_rail = achievement_by_name("onARail").expect("onARail");
        assert!(on_a_rail.is_special());
    }

    #[test]
    fn parent_tree_is_internally_consistent() {
        // No dangling parent references, and no self-parenting cycles at depth 1.
        for a in achievements() {
            if let Some(parent) = a.parent {
                assert_ne!(parent, a.achievement, "{} parents itself", a.name);
                let _ = parent.info(); // must not panic
            }
        }
        // Every achievement is reachable from a root within a bounded number of hops -
        // proves there are no parent cycles anywhere in the table.
        for a in achievements() {
            let mut hops = 0;
            let mut cur = a.achievement;
            while let Some(p) = cur.parent() {
                cur = p;
                hops += 1;
                assert!(hops <= achievements().len(), "cycle involving {}", a.name);
            }
        }
    }

    #[test]
    fn children_are_the_inverse_of_parent() {
        let open_inv = achievement_by_name("openInventory").expect("openInventory");
        let kids: Vec<_> = open_inv.children().collect();
        assert!(kids.contains(&achievement_by_name("mineWood").expect("mineWood")));
        for child in kids {
            assert_eq!(child.parent(), Some(open_inv));
        }
    }
}
