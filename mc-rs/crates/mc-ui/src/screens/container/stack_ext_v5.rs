//! The [`super`] helpers for a container of a given game version: the item properties a click
//! prediction reads come from that version's item registry.
//!
//! 1.7.10's registry ([`mc_data::v5::try_item_props_v5`]) differs from 1.8.9's on what a click
//! can observe: doors stack to one (java17: `ItemDoor.<init>`), stone, grass and sponge have no
//! subtypes, twelve blocks such as water and fire have items, and the 34 items 1.8.9 added do not
//! exist. The stack rules themselves are the same in both versions (java17 and java18
//! `ItemStack.getMaxStackSize`, `isItemStackDamageable`, `isItemDamaged`, `isStackable`,
//! `getHasSubtypes`), so each 1.7.10 arm applies the 1.8.9 rule to the 1.7.10 row, and each 1.8.9
//! arm is the 1.8.9 helper itself.

use mc_data::{item_props, ItemId, ItemProps};
use mc_version::GameVersion;

use super::{
    has_subtypes, is_item_damaged, is_stack_damageable, is_stackable, max_stack_size, props,
    same_item_respecting_subtypes, try_props, unbreakable, ItemStack,
};

/// [`try_props`] in `version`'s registry: `None` where that version registers no such item.
#[must_use]
pub fn try_props_in(version: GameVersion, stack: &ItemStack) -> Option<&'static ItemProps> {
    match version {
        GameVersion::V1_8_9 => try_props(stack),
        GameVersion::V1_7_10 => mc_data::v5::try_item_props_v5(ItemId(stack.item_id as u16)),
    }
}

/// [`props`] in `version`'s registry: the unregistered sentinel where that version registers no
/// such item.
#[must_use]
pub fn props_in(version: GameVersion, stack: &ItemStack) -> &'static ItemProps {
    match version {
        GameVersion::V1_8_9 => props(stack),
        // Item ids stop far below `u16::MAX`, so this is `item_props`'s unregistered sentinel.
        GameVersion::V1_7_10 => {
            try_props_in(version, stack).unwrap_or_else(|| item_props(ItemId(u16::MAX)))
        }
    }
}

/// [`max_stack_size`] in `version`'s registry.
#[must_use]
pub fn max_stack_size_in(version: GameVersion, stack: &ItemStack) -> i32 {
    match version {
        GameVersion::V1_8_9 => max_stack_size(stack),
        GameVersion::V1_7_10 => i32::from(props_in(version, stack).max_stack_size),
    }
}

/// [`is_stack_damageable`] in `version`'s registry.
#[must_use]
pub fn is_stack_damageable_in(version: GameVersion, stack: &ItemStack) -> bool {
    match version {
        GameVersion::V1_8_9 => is_stack_damageable(stack),
        GameVersion::V1_7_10 => props_in(version, stack).max_damage != 0 && !unbreakable(stack),
    }
}

/// [`is_item_damaged`] in `version`'s registry.
#[must_use]
pub fn is_item_damaged_in(version: GameVersion, stack: &ItemStack) -> bool {
    match version {
        GameVersion::V1_8_9 => is_item_damaged(stack),
        GameVersion::V1_7_10 => is_stack_damageable_in(version, stack) && stack.damage > 0,
    }
}

/// [`is_stackable`] in `version`'s registry.
#[must_use]
pub fn is_stackable_in(version: GameVersion, stack: &ItemStack) -> bool {
    match version {
        GameVersion::V1_8_9 => is_stackable(stack),
        GameVersion::V1_7_10 => {
            max_stack_size_in(version, stack) > 1
                && (!is_stack_damageable_in(version, stack) || !is_item_damaged_in(version, stack))
        }
    }
}

/// [`has_subtypes`] in `version`'s registry.
#[must_use]
pub fn has_subtypes_in(version: GameVersion, stack: &ItemStack) -> bool {
    match version {
        GameVersion::V1_8_9 => has_subtypes(stack),
        GameVersion::V1_7_10 => props_in(version, stack).has_subtypes,
    }
}

/// [`same_item_respecting_subtypes`] in `version`'s registry.
#[must_use]
pub fn same_item_respecting_subtypes_in(
    version: GameVersion,
    stack: &ItemStack,
    other: &ItemStack,
) -> bool {
    match version {
        GameVersion::V1_8_9 => same_item_respecting_subtypes(stack, other),
        GameVersion::V1_7_10 => {
            stack.item_id == other.item_id
                && (!has_subtypes_in(version, stack) || stack.damage == other.damage)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(item_id: i16, count: i32, damage: i16) -> ItemStack {
        ItemStack {
            item_id,
            count,
            damage,
            nbt: None,
        }
    }

    #[test]
    fn the_1_8_9_arm_is_the_1_8_9_helper_for_every_id_and_some_damages() {
        let v = GameVersion::V1_8_9;
        for id in -2i16..=2300 {
            for damage in [0, 1, 3, 15, 100] {
                let s = stack(id, 1, damage);
                let other = stack(id, 1, 0);
                assert_eq!(try_props_in(v, &s), try_props(&s));
                assert!(std::ptr::eq(props_in(v, &s), props(&s)));
                assert_eq!(max_stack_size_in(v, &s), max_stack_size(&s));
                assert_eq!(is_stack_damageable_in(v, &s), is_stack_damageable(&s));
                assert_eq!(is_item_damaged_in(v, &s), is_item_damaged(&s));
                assert_eq!(is_stackable_in(v, &s), is_stackable(&s));
                assert_eq!(has_subtypes_in(v, &s), has_subtypes(&s));
                assert_eq!(
                    same_item_respecting_subtypes_in(v, &s, &other),
                    same_item_respecting_subtypes(&s, &other)
                );
            }
        }
    }

    #[test]
    fn doors_stack_to_one_in_1_7_10_and_sixty_four_in_1_8_9() {
        for door in [324, 330] {
            let s = stack(door, 1, 0);
            assert_eq!(max_stack_size_in(GameVersion::V1_7_10, &s), 1);
            assert!(!is_stackable_in(GameVersion::V1_7_10, &s));
            assert_eq!(max_stack_size_in(GameVersion::V1_8_9, &s), 64);
            assert!(is_stackable_in(GameVersion::V1_8_9, &s));
        }
    }

    #[test]
    fn stone_of_two_damages_is_one_item_only_in_1_7_10() {
        let (stone, granite) = (stack(1, 1, 0), stack(1, 1, 1));
        assert!(same_item_respecting_subtypes_in(
            GameVersion::V1_7_10,
            &stone,
            &granite
        ));
        assert!(!same_item_respecting_subtypes_in(
            GameVersion::V1_8_9,
            &stone,
            &granite
        ));
    }

    #[test]
    fn each_registry_knows_its_own_items() {
        let water = stack(9, 1, 0);
        let prismarine_shard = stack(409, 1, 0);
        assert!(try_props_in(GameVersion::V1_7_10, &water).is_some());
        assert!(try_props_in(GameVersion::V1_8_9, &water).is_none());
        assert!(try_props_in(GameVersion::V1_7_10, &prismarine_shard).is_none());
        // The unregistered sentinel, as `props` gives for an id 1.8.9 does not register.
        assert!(!props_in(GameVersion::V1_7_10, &prismarine_shard).registered);
        assert!(std::ptr::eq(
            props_in(GameVersion::V1_7_10, &prismarine_shard),
            props(&stack(-1, 1, 0))
        ));
        assert!(try_props_in(GameVersion::V1_8_9, &prismarine_shard).is_some());
    }
}
