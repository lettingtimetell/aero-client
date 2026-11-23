//! Small `ItemStack` helpers the click protocol needs.
//!
//! `mc-protocol`'s [`ItemStack`] is deliberately a thin wire type (S06 owns it, and it must not
//! grow an `mc-data` dependency). Everything here is a free function or extension trait layered
//! on top from this crate instead, mirroring one Java `ItemStack` method each.
//!
//! java: `net/minecraft/item/ItemStack.java`

pub use mc_protocol::item::ItemStack;

use mc_data::{item_props, try_item_props, ItemId};

/// This item's registry properties, or `None` when its signed wire id does not name a
/// registered 1.8.9 item. Java's item registry lookup is nullable; presentation callers use
/// this strict form so an unknown server value cannot become fabricated missing-item geometry.
#[must_use]
pub fn try_props(stack: &ItemStack) -> Option<&'static mc_data::ItemProps> {
    try_item_props(ItemId(stack.item_id as u16))
}

/// This item's registry properties. Unregistered/out-of-range ids resolve to the same
/// zeroed sentinel `mc_data::item_props` returns for any unknown id — never a panic, because
/// `item_id` arrives from the network.
#[must_use]
pub fn props(stack: &ItemStack) -> &'static mc_data::ItemProps {
    item_props(ItemId(stack.item_id as u16))
}

/// java: `ItemStack.getMaxStackSize` — `Item.getItemStackLimit()` verbatim, not slot-adjusted.
#[must_use]
pub fn max_stack_size(stack: &ItemStack) -> i32 {
    i32::from(props(stack).max_stack_size)
}

/// java: `ItemStack.isItemStackDamageable` (`item/ItemStack.java:250-253`) —
/// `getMaxDamage() <= 0 ? false : !hasTagCompound() || !getTagCompound().getBoolean("Unbreakable")`.
///
/// The `Unbreakable` conjunct is unobservable through this crate's *current* callers
/// (`is_stackable` short-circuits on `max_stack_size > 1`, and every damageable 1.8.9 item
/// stacks to 1), but it selects between `ContainerRepair.updateRepairOutput`'s material-repair
/// and combine branches, so it becomes load-bearing the moment that lands.
///
/// Note the comparison: java's is `getMaxDamage() <= 0`, on a signed `int`. `mc_data`'s
/// `max_damage` is a `u16`, so the `< 0` half of that is unreachable here - and it is
/// unreachable in 1.8.9 too: every `setMaxDamage` call site in the tree passes a literal
/// `0`/`25`/`64`/`238`/`384`, `ToolMaterial.getMaxUses` (`32..1561`), or
/// `ArmorMaterial.getDurability` (`maxDamageArray[type] * maxDamageFactor`, both positive), and
/// there is no negative anywhere. So `== 0` is the whole of the java guard, not a narrowing of
/// it. Written as `== 0` rather than `<= 0` because the latter is a lint (and a lie).
///
/// `ItemProps::damageable` is deliberately **not** consulted: that field is `Item.isDamageable`
/// (`Item.java:209`), which is `maxDamage > 0 && !hasSubtypes` - a *different* method with an
/// extra conjunct `isItemStackDamageable` does not have. The two agree on every registered
/// 1.8.9 item (checked: no item has `max_damage > 0` together with `has_subtypes`), so this is
/// not a behaviour change today, but keying off `max_damage` is the faithful one.
#[must_use]
pub fn is_stack_damageable(stack: &ItemStack) -> bool {
    if props(stack).max_damage == 0 {
        return false;
    }
    !unbreakable(stack)
}

/// java: `NBTTagCompound.getBoolean("Unbreakable")`, which is `getByte(key) != 0` and whose
/// `hasKey(key, 99)` accepts **any** numeric primitive, not only `TAG_Byte`
/// (`nbt/NBTTagCompound.java:212-222,387-390`). `mc_nbt`'s `get_bool` reads `TAG_Byte` alone,
/// so the other numeric tags are matched here rather than silently dropped.
fn unbreakable(stack: &ItemStack) -> bool {
    let Some(nbt) = stack.nbt.as_ref() else {
        return false;
    };
    match nbt.get("Unbreakable") {
        // Each arm is that tag's own `NBTPrimitive.getByte()`: `(byte)(data & 255)` for the
        // integral tags, and `(byte)(MathHelper.floor_* (data) & 255)` — a *floor*, not a
        // truncation — for the two floating ones.
        Some(mc_nbt::Nbt::Byte(v)) => *v != 0,
        Some(mc_nbt::Nbt::Short(v)) => *v as i8 != 0,
        Some(mc_nbt::Nbt::Int(v)) => *v as i8 != 0,
        Some(mc_nbt::Nbt::Long(v)) => *v as i8 != 0,
        Some(mc_nbt::Nbt::Float(v)) => v.floor() as i32 as i8 != 0,
        Some(mc_nbt::Nbt::Double(v)) => v.floor() as i32 as i8 != 0,
        _ => false,
    }
}

/// java: `ItemStack.isItemDamaged` — `isItemStackDamageable() && damage > 0`.
#[must_use]
pub fn is_item_damaged(stack: &ItemStack) -> bool {
    is_stack_damageable(stack) && stack.damage > 0
}

/// java: `ItemStack.isStackable` — `getMaxStackSize() > 1 && (!isItemStackDamageable() ||
/// !isItemDamaged())`.
#[must_use]
pub fn is_stackable(stack: &ItemStack) -> bool {
    max_stack_size(stack) > 1 && (!is_stack_damageable(stack) || !is_item_damaged(stack))
}

/// java: `ItemStack.getHasSubtypes` (`Item.getHasSubtypes()`).
#[must_use]
pub fn has_subtypes(stack: &ItemStack) -> bool {
    props(stack).has_subtypes
}

/// java: `ItemStack.areItemStackTagsEqual` — two absent *stacks* are equal, one absent stack is
/// never equal to a present stack, and two present stacks compare only their optional compounds.
/// `NbtCompound` already derives structural [`PartialEq`], exactly vanilla's `NBTBase.equals`.
#[must_use]
pub fn tags_equal(a: Option<&ItemStack>, b: Option<&ItemStack>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.nbt == b.nbt,
        _ => false,
    }
}

/// java: `ItemStack.areItemStacksEqual` / the private `isItemStackEqual` it delegates to.
///
/// This deliberately includes the stack count.  `is_item_equal` below is the narrower
/// item-and-metadata predicate used for merge eligibility, whereas vanilla uses this full
/// predicate when deciding whether a slot snapshot has changed.  As in Java, two absent stacks
/// compare equal; a missing stack and a present tagless stack do not.
#[must_use]
pub fn stacks_equal(a: Option<&ItemStack>, b: Option<&ItemStack>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.count == b.count && is_item_equal(a, b) && tags_equal(Some(a), Some(b))
        }
        _ => false,
    }
}

/// java: `ItemStack.isItemEqual` — same registry item and same metadata. **Not** subtype-aware
/// (unlike [`same_item_respecting_subtypes`]) and does **not** compare NBT — callers that need
/// that call [`tags_equal`] separately, exactly as every Java call site does.
#[must_use]
pub fn is_item_equal(a: &ItemStack, b: &ItemStack) -> bool {
    a.item_id == b.item_id && a.damage == b.damage
}

/// The `mergeItemStack`/pickup-combine check: same registry item, and metadata only has to
/// match when the item `getHasSubtypes()` (java: `!stack.getHasSubtypes() ||
/// stack.getMetadata() == itemstack.getMetadata()`).
#[must_use]
pub fn same_item_respecting_subtypes(stack: &ItemStack, other: &ItemStack) -> bool {
    stack.item_id == other.item_id && (!has_subtypes(stack) || stack.damage == other.damage)
}

/// java: `ItemStack.splitStack` — makes a new stack whose count is exactly `amount`, deep-copies
/// its tag, and subtracts `amount` from the source.  It deliberately does **not** clamp to the
/// available count: Java accepts zero, negative, and oversized signed `int` amounts.  The
/// subtraction wraps as Java `int` arithmetic does, rather than panicking on malformed
/// network-derived counts in Rust debug builds.
#[must_use]
pub fn split_stack(stack: &mut ItemStack, amount: i32) -> ItemStack {
    stack.count = stack.count.wrapping_sub(amount);
    ItemStack {
        item_id: stack.item_id,
        count: amount,
        damage: stack.damage,
        nbt: stack.nbt.clone(),
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
    fn diamond_sword_is_not_stackable_once_damaged() {
        // 276 = minecraft:diamond_sword, single-stack, damageable.
        let fresh = stack(276, 1, 0);
        assert_eq!(max_stack_size(&fresh), 1);
        assert!(!is_stackable(&fresh), "max stack size 1 is never stackable");
    }

    #[test]
    fn cobblestone_stacks_to_sixty_four() {
        // 4 = minecraft:cobblestone.
        let s = stack(4, 1, 0);
        assert_eq!(max_stack_size(&s), 64);
        assert!(is_stackable(&s));
    }

    #[test]
    fn split_stack_leaves_the_remainder_behind() {
        let mut s = stack(1, 10, 0);
        let taken = split_stack(&mut s, 4);
        assert_eq!(taken.count, 4);
        assert_eq!(s.count, 6);
    }

    #[test]
    fn split_stack_does_not_clamp_an_oversized_amount() {
        let mut s = stack(1, 3, 0);
        let taken = split_stack(&mut s, 10);
        assert_eq!(taken.count, 10);
        assert_eq!(s.count, -7);
    }

    #[test]
    fn split_stack_zero_returns_an_empty_stack_without_changing_the_source() {
        let mut s = stack(1, 3, 0);
        let taken = split_stack(&mut s, 0);
        assert_eq!(taken.count, 0);
        assert_eq!(s.count, 3);
    }

    #[test]
    fn split_stack_keeps_java_signed_int_arithmetic_for_negative_amounts() {
        // A negative wire count or request must not be reinterpreted as a large unsigned byte.
        let mut s = stack(1, 5, 0);
        let taken = split_stack(&mut s, -1);
        assert_eq!(taken.count, -1);
        assert_eq!(s.count, 6);
    }

    #[test]
    fn split_stack_wraps_like_java_int_subtraction_at_the_signed_boundary() {
        let mut s = stack(1, -1, 0);
        let taken = split_stack(&mut s, i32::MIN);
        assert_eq!(taken.count, i32::MIN);
        assert_eq!(s.count, i32::MAX);
    }

    #[test]
    fn split_stack_deep_copies_nbt_instead_of_aliasing_the_source() {
        let mut s = stack(1, 5, 2);
        let mut tag = mc_nbt::NbtCompound::new();
        let mut nested = mc_nbt::NbtCompound::new();
        nested.insert("inner", mc_nbt::Nbt::Int(1));
        tag.insert("nested", mc_nbt::Nbt::Compound(nested));
        s.nbt = Some(tag);

        let mut taken = split_stack(&mut s, 2);
        taken
            .nbt
            .as_mut()
            .expect("split stack preserves a present tag")
            .get_mut("nested")
            .and_then(mc_nbt::Nbt::as_compound_mut)
            .expect("nested compound")
            .insert("inner", mc_nbt::Nbt::Int(9));

        assert_eq!(s.count, 3);
        assert_eq!(taken.count, 2);
        assert_eq!(
            s.nbt
                .as_ref()
                .and_then(|tag| tag.get_compound("nested"))
                .and_then(|tag| match tag.get("inner") {
                    Some(mc_nbt::Nbt::Int(value)) => Some(*value),
                    _ => None,
                }),
            Some(1),
            "mutating the returned tag must not mutate the source tag"
        );
    }

    #[test]
    fn tags_equal_treats_both_absent_as_equal() {
        let a = stack(1, 1, 0);
        let b = stack(1, 1, 0);
        assert!(tags_equal(Some(&a), Some(&b)));
    }

    #[test]
    fn tags_equal_is_false_when_only_one_side_has_a_tag() {
        let a = stack(1, 1, 0);
        let mut b = stack(1, 1, 0);
        b.nbt = Some(mc_nbt::NbtCompound::new());
        assert!(!tags_equal(Some(&a), Some(&b)));
    }

    /// Java distinguishes a null `ItemStack` from a non-null stack whose tag compound is null.
    /// Projecting both through `and_then(stack.nbt)` erases that distinction.
    #[test]
    fn tags_equal_preserves_the_full_stack_and_compound_presence_matrix() {
        let tagless = stack(1, 1, 0);
        let mut tagged = stack(1, 1, 0);
        let mut tag = mc_nbt::NbtCompound::new();
        tag.insert("marker", mc_nbt::Nbt::Int(1));
        tagged.nbt = Some(tag);

        let states = [None, Some(&tagless), Some(&tagged)];
        for (a_index, a) in states.into_iter().enumerate() {
            for (b_index, b) in states.into_iter().enumerate() {
                assert_eq!(
                    tags_equal(a, b),
                    a_index == b_index,
                    "matrix cell ({a_index}, {b_index})"
                );
            }
        }
    }

    /// `areItemStacksEqual` is a complete slot-snapshot comparison, unlike
    /// `areItemStackTagsEqual` and `areItemsEqual`: it includes amount, id, metadata and tag.
    #[test]
    fn stacks_equal_matches_java_snapshot_comparison_and_null_rules() {
        let plain = stack(1, 4, 2);
        let mut tagged = plain.clone();
        let mut tag = mc_nbt::NbtCompound::new();
        tag.insert("marker", mc_nbt::Nbt::Int(7));
        tagged.nbt = Some(tag);

        assert!(stacks_equal(None, None));
        assert!(!stacks_equal(None, Some(&plain)));
        assert!(!stacks_equal(Some(&plain), None));
        assert!(stacks_equal(Some(&plain), Some(&plain.clone())));

        for changed in [
            stack(1, 3, 2), // stackSize
            stack(2, 4, 2), // item
            stack(1, 4, 3), // itemDamage
            tagged,         // stackTagCompound presence/content
        ] {
            assert!(
                !stacks_equal(Some(&plain), Some(&changed)),
                "different stack snapshot must not compare equal: {changed:?}"
            );
        }
    }

    /// The Java comparator is exactly the structural `Option<ItemStack>` snapshot predicate:
    /// null presence, count, item id, metadata, and the compound's type-sensitive contents all
    /// participate. Keeping this matrix beside the helper makes a future "compare only the item"
    /// or tag-projection mutant fail at the first changed field.
    #[test]
    fn stacks_equal_matches_structural_option_snapshot_equality() {
        let plain = stack(1, 4, 2);
        let mut tagged = plain.clone();
        let mut tag = mc_nbt::NbtCompound::new();
        tag.insert("marker", mc_nbt::Nbt::Int(7));
        tagged.nbt = Some(tag);
        let count_changed = stack(1, 5, 2);
        let item_changed = stack(2, 4, 2);
        let damage_changed = stack(1, 4, 3);
        let states = [
            None,
            Some(&plain),
            Some(&tagged),
            Some(&count_changed),
            Some(&item_changed),
            Some(&damage_changed),
        ];

        for (a_index, a) in states.into_iter().enumerate() {
            for (b_index, b) in states.into_iter().enumerate() {
                assert_eq!(
                    stacks_equal(a, b),
                    a == b,
                    "snapshot cell ({a_index}, {b_index})"
                );
            }
        }
    }

    /// `NBTTagCompound.equals` is a map comparison: insertion order is irrelevant, while each
    /// NBT tag's concrete type and value remain significant. `NbtCompound`/`Nbt` provide the
    /// same structural equality used by `stacks_equal`; these two controls guard both halves.
    #[test]
    fn stacks_equal_preserves_java_nbt_type_and_map_equality() {
        let mut first_root = mc_nbt::NbtCompound::new();
        first_root.insert("first", mc_nbt::Nbt::Int(1));
        first_root.insert("second", mc_nbt::Nbt::String("same".to_owned()));

        let mut reordered_root = mc_nbt::NbtCompound::new();
        reordered_root.insert("second", mc_nbt::Nbt::String("same".to_owned()));
        reordered_root.insert("first", mc_nbt::Nbt::Int(1));

        let mut different_type_root = reordered_root.clone();
        different_type_root.insert("first", mc_nbt::Nbt::Short(1));

        let mut first = stack(1, 1, 0);
        first.nbt = Some(first_root);
        let mut reordered = stack(1, 1, 0);
        reordered.nbt = Some(reordered_root);
        let mut different_type = stack(1, 1, 0);
        different_type.nbt = Some(different_type_root);

        assert!(stacks_equal(Some(&first), Some(&reordered)));
        assert!(!stacks_equal(Some(&first), Some(&different_type)));
    }

    /// The `getMaxDamage() <= 0 -> false` guard itself. Nothing covered it: every existing
    /// caller reaches the same answer with or without it, so removing it was a silent mutation
    /// that made **every** subtyped block "damaged" and therefore unstackable.
    #[test]
    fn an_item_with_no_durability_is_never_damageable_whatever_its_metadata() {
        let wool = mc_data::item_by_name("minecraft:wool")
            .expect("known item")
            .0 as i16;
        // Metadata 5 is lime wool, not "5 damage": `maxDamage == 0`, so `isItemStackDamageable`
        // returns false at the first test and the stack stays stackable.
        let lime = stack(wool, 64, 5);
        assert_eq!(props(&lime).max_damage, 0);
        assert!(!is_stack_damageable(&lime));
        assert!(!is_item_damaged(&lime));
        assert!(is_stackable(&lime), "a coloured wool still stacks to 64");

        // Negative control: a real durability item at the same metadata is both.
        let sword = mc_data::item_by_name("minecraft:diamond_sword")
            .expect("known item")
            .0 as i16;
        let used = stack(sword, 1, 5);
        assert!(is_stack_damageable(&used));
        assert!(is_item_damaged(&used));
    }

    /// java: `ItemStack.isItemStackDamageable`'s second conjunct.
    #[test]
    fn an_unbreakable_tool_is_not_damageable() {
        let sword_id = mc_data::item_by_name("minecraft:diamond_sword")
            .expect("known item")
            .0 as i16;
        let mut s = stack(sword_id, 1, 10);
        assert!(is_stack_damageable(&s), "plain diamond sword");
        assert!(is_item_damaged(&s), "plain used diamond sword");
        let mut tag = mc_nbt::NbtCompound::new();
        tag.insert("Unbreakable", mc_nbt::Nbt::Byte(1));
        s.nbt = Some(tag);
        assert!(!is_stack_damageable(&s));
        assert!(!is_item_damaged(&s), "and therefore never 'damaged'");
    }

    /// `NBTTagCompound.getByte`'s `hasKey(key, 99)` accepts every numeric primitive, and each
    /// tag's own `getByte()` is `(byte)(data & 255)` — so `Unbreakable:256` is byte `0`, i.e.
    /// **false**, and an `Unbreakable:0b` is false too.
    #[test]
    fn unbreakable_follows_nbt_get_byte_and_not_mere_presence() {
        let sword_id = mc_data::item_by_name("minecraft:diamond_sword")
            .expect("known item")
            .0 as i16;
        for (tag, unbreakable) in [
            (mc_nbt::Nbt::Byte(0), false),
            (mc_nbt::Nbt::Byte(1), true),
            (mc_nbt::Nbt::Int(1), true),
            (mc_nbt::Nbt::Int(256), false), // & 255 == 0
            (mc_nbt::Nbt::Short(2), true),
            (mc_nbt::Nbt::Long(256), false),
            (mc_nbt::Nbt::Float(-0.5), true), // Java floors to -1 before narrowing to byte
            (mc_nbt::Nbt::Float(0.5), false), // Java floors to 0
            (mc_nbt::Nbt::Double(-0.5), true),
            (mc_nbt::Nbt::String("1".into()), false), // not a primitive: hasKey(_, 99) fails
        ] {
            let mut s = stack(sword_id, 1, 0);
            let mut c = mc_nbt::NbtCompound::new();
            c.insert("Unbreakable", tag.clone());
            s.nbt = Some(c);
            assert_eq!(
                is_stack_damageable(&s),
                !unbreakable,
                "Unbreakable = {tag:?}"
            );
        }
    }

    #[test]
    fn subtype_check_ignores_metadata_for_items_without_subtypes() {
        // Cobblestone has no subtypes, so different "metadata" (there isn't really one) still
        // matches on item id alone.
        let a = stack(4, 1, 0);
        let b = stack(4, 1, 5);
        assert!(same_item_respecting_subtypes(&a, &b));
    }
}
