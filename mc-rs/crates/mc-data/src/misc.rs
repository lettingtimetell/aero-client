//! The smaller registries: potion effects, enchantments, sound events.
//!
//! Java inputs: `net/minecraft/potion/Potion.java`,
//! `net/minecraft/enchantment/Enchantment.java`. Sound events are not a Java registry in
//! 1.8.9 at all - `S29PacketSoundEffect` carries a name string - so that table comes from
//! `assets/minecraft/sounds.json` in the vanilla asset store.

use crate::generated::enchantments::ENCHANTMENTS;
use crate::generated::potions::POTIONS;
use crate::generated::sounds::SOUND_EVENTS;

pub use crate::generated::enchantments::EnchantmentTarget;

impl EnchantmentTarget {
    /// Java 1.8.9 `EnumEnchantmentType.canEnchantItem`.
    ///
    /// This is the base target predicate only. Individual enchantments may further restrict
    /// applicability in their own `canApply` override.
    #[must_use]
    pub fn can_enchant_item(self, item: &crate::ItemProps) -> bool {
        use crate::ItemKind;

        // Registry lookup is nullable in Java. The total Rust registry uses an inert sentinel
        // for hostile wire ids, so reject that sentinel before `All` can accept it.
        if !item.registered {
            return false;
        }

        match self {
            Self::All => true,
            // Java tests BREAKABLE before any instanceof branch. `damageable` is the extracted
            // `Item.isDamageable()` result (`maxDamage > 0 && !hasSubtypes`).
            Self::Breakable => item.damageable,
            Self::Armor => item.kind == ItemKind::Armor,
            Self::ArmorHead => item.kind == ItemKind::Armor && item.armor_slot == Some(0),
            Self::ArmorTorso => item.kind == ItemKind::Armor && item.armor_slot == Some(1),
            Self::ArmorLegs => item.kind == ItemKind::Armor && item.armor_slot == Some(2),
            Self::ArmorFeet => item.kind == ItemKind::Armor && item.armor_slot == Some(3),
            Self::Weapon => item.kind == ItemKind::Sword,
            Self::Digger => matches!(
                item.kind,
                ItemKind::Pickaxe | ItemKind::Axe | ItemKind::Shovel
            ),
            Self::FishingRod => item.java_class == "net.minecraft.item.ItemFishingRod",
            Self::Bow => item.kind == ItemKind::Bow,
        }
    }
}

/// A potion effect type, as carried by `S1DPacketEntityEffect`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotionEffect {
    /// Effect id, `1..=23` in 1.8.9.
    pub id: u8,
    /// Translation key, e.g. `potion.moveSpeed`.
    pub name: &'static str,
    /// `en_US` display name.
    pub display_name: &'static str,
    /// Whether the effect is harmful (drawn with a red-tinted border).
    pub bad: bool,
    /// Packed RGB particle/liquid colour.
    pub liquid_color: u32,
    /// `Potion.getEffectiveness()`: duration multiplier used by
    /// `PotionHelper.getPotionEffects` for non-instant effects.
    pub effectiveness: f64,
    /// Whether the effect can appear on a brewed potion.
    pub usable: bool,
    /// Whether the effect applies instantly instead of over a duration.
    pub instant: bool,
    /// `Potion.getStatusIconIndex()`: the cell in `inventory.png`'s 8-wide, `v = 198` icon grid
    /// (`col + row * 8`), or `None` when the effect has no HUD/beacon icon (`heal`, `harm`,
    /// `saturation` — the three instant effects vanilla never shows a status icon for).
    /// The beacon's power buttons and the HUD's potion list
    /// both need this to blit the right 18×18 cell.
    pub status_icon_index: Option<i32>,
}

/// A potion effect by id, or `None` for ids vanilla does not register.
///
/// # Example
/// ```
/// use mc_data::potion;
/// assert_eq!(potion(1).map(|p| p.name), Some("potion.moveSpeed"));
/// assert!(potion(0).is_none());
/// ```
#[inline]
#[must_use]
pub fn potion(id: u8) -> Option<&'static PotionEffect> {
    POTIONS.get(id as usize).and_then(Option::as_ref)
}

/// Every registered potion effect, ascending by id.
pub fn potions() -> impl Iterator<Item = &'static PotionEffect> {
    POTIONS.iter().flatten()
}

/// One effect decoded from an `ItemPotion` damage value.
///
/// This is Java 1.8.9's `PotionEffect` value as constructed by
/// `PotionHelper.getPotionEffects`: all metadata-derived effects are non-ambient, show their
/// particles, and carry the potion item's splash flag.  Item NBT's `CustomPotionEffects` is a
/// separate `ItemPotion` layer and intentionally belongs to the item/NBT consumer rather than
/// this static-data crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PotionStackEffect {
    /// Registered potion-effect id.
    pub id: u8,
    /// Remaining duration in 20 Hz game ticks; instant effects use `1`.
    pub duration: i32,
    /// Zero-based strength (`0` is level I, `1` is II).
    pub amplifier: i32,
    /// Metadata-derived effects are never ambient.
    pub ambient: bool,
    /// Metadata-derived effects always show particles.
    pub show_particles: bool,
    /// Whether bit 14 of the item damage says this is a splash potion.
    pub splash: bool,
}

/// Decode `PotionHelper.getPotionEffects(damage, false)`.
///
/// The decoder intentionally accepts every `i32`, just as Java does.  The effect requirement
/// grammar reads bits 0 through 6 and the splash adjustment reads bit 14; all other bits are
/// inert.  Thus malformed, negative, or high-bit wire metadata remains total instead of being
/// normalised before the vanilla bit tests can observe it.
#[must_use]
pub fn potion_effects_from_damage(damage: i32) -> Vec<PotionStackEffect> {
    potion_effects_from_damage_cached(damage).to_vec()
}

/// The damage bits PotionHelper's live 1.8.9 tables read: requirement bits 0–3 and 6
/// (`"1+6"` etc.), amplifier bit 5 (`"5"`) and the splash bit 14. Every other bit is inert, so
/// two damage values agreeing on these decode identically. Pinned exhaustively against the
/// expression evaluator by `memoized_potion_decoders_match_the_expression_parser_for_all_i16`.
const POTION_DAMAGE_BITS: i32 = 0b0100_0000_0110_1111;

/// Dense index of the [`POTION_DAMAGE_BITS`] of `damage`: bits 0–6 as-is, bit 14 as bit 7.
fn potion_memo_index(damage: i32) -> usize {
    let bits = damage & POTION_DAMAGE_BITS;
    let low = (bits & 0x7F) as usize;
    let splash = usize::from(bits & 16_384 != 0) << 7;
    low | splash
}

/// A representative damage value for a memo slot: the inverse of [`potion_memo_index`].
fn potion_memo_damage(index: usize) -> i32 {
    ((index & 0x7F) as i32) | (((index >> 7) & 1) as i32) << 14
}

/// [`potion_effects_from_damage`] without the allocation, memoized per relevant-bit pattern.
///
/// The decoder runs PotionHelper's string-expression grammar (13 requirement expressions,
/// `split`/`trim`/`parse` per term); GUI item tinting asks for it once per tinted quad per frame
/// (C2-7 / CONT-2), and dropped/thrown potions per entity. Each of the 256 patterns is decoded
/// once, on first use, by the unchanged evaluator.
#[must_use]
pub fn potion_effects_from_damage_cached(damage: i32) -> &'static [PotionStackEffect] {
    static MEMO: [std::sync::OnceLock<Box<[PotionStackEffect]>>; 256] =
        [const { std::sync::OnceLock::new() }; 256];
    let index = potion_memo_index(damage);
    MEMO[index].get_or_init(|| decode_potion_effects(potion_memo_damage(index)).into_boxed_slice())
}

/// The uncached `PotionHelper.getPotionEffects(damage, false)` decoder.
fn decode_potion_effects(damage: i32) -> Vec<PotionStackEffect> {
    let mut effects = Vec::new();
    for info in potions() {
        let Some(requirement) = potion_requirement(info.id) else {
            continue;
        };
        let mut duration = potion_expression_value(requirement, damage);
        if duration <= 0 {
            continue;
        }
        let amplifier = potion_amplifier(info.id)
            .map_or(0, |expr| potion_expression_value(expr, damage).max(0));
        if info.instant {
            duration = 1;
        } else {
            // PotionHelper: 1200 * (n * 3 + (n - 1) * 2), then a right shift for strength.
            duration = 1_200 * (duration * 3 + (duration - 1) * 2);
            duration >>= amplifier;
            duration = (f64::from(duration) * info.effectiveness).round() as i32;
            if damage & 16_384 != 0 {
                // Java float arithmetic: (int)(duration * 0.75F + 0.5F).  All reachable
                // duration values are exactly representable, so this f64 form is equivalent.
                duration = (f64::from(duration) * 0.75 + 0.5).round() as i32;
            }
        }
        effects.push(PotionStackEffect {
            id: info.id,
            duration,
            amplifier,
            ambient: false,
            show_particles: true,
            splash: damage & 16_384 != 0,
        });
    }
    effects
}

/// Decode `PotionHelper.getLiquidColor(damage, false)`.
///
/// This is metadata-only on purpose: `ItemPotion.getColorFromItemStack` calls the damage-value
/// overload even when the stack has `CustomPotionEffects`.
#[must_use]
pub fn potion_liquid_color_from_damage(damage: i32) -> u32 {
    // Memoized per relevant-bit pattern, like the effect list it is derived from (C2-7). Colours
    // are 24-bit, so `u32::MAX` marks a slot that has not been decoded yet.
    static MEMO: [std::sync::atomic::AtomicU32; 256] =
        [const { std::sync::atomic::AtomicU32::new(u32::MAX) }; 256];
    let slot = &MEMO[potion_memo_index(damage)];
    let cached = slot.load(std::sync::atomic::Ordering::Relaxed);
    if cached != u32::MAX {
        return cached;
    }
    let color = decode_potion_liquid_color(damage);
    slot.store(color, std::sync::atomic::Ordering::Relaxed);
    color
}

/// The uncached `PotionHelper.getLiquidColor(damage, false)` decoder.
fn decode_potion_liquid_color(damage: i32) -> u32 {
    let effects = potion_effects_from_damage_cached(damage);
    if effects.is_empty() {
        return 3_694_022;
    }

    let mut red = 0.0_f32;
    let mut green = 0.0_f32;
    let mut blue = 0.0_f32;
    let mut weight = 0.0_f32;
    for effect in effects {
        let color = potion(effect.id)
            .expect("potion_effects_from_damage only emits registered ids")
            .liquid_color;
        for _ in 0..=effect.amplifier {
            red += ((color >> 16) & 255) as f32 / 255.0;
            green += ((color >> 8) & 255) as f32 / 255.0;
            blue += (color & 255) as f32 / 255.0;
            weight += 1.0;
        }
    }
    // This cannot be zero for metadata effects, but retaining PotionHelper's guard makes the
    // total API resilient if a future extracted effect intentionally hides particles.
    if weight == 0.0 {
        return 0;
    }
    let red = (red / weight * 255.0) as u32;
    let green = (green / weight * 255.0) as u32;
    let blue = (blue / weight * 255.0) as u32;
    (red << 16) | (green << 8) | blue
}

/// The potion metadata values exposed by `ItemPotion.getSubItems` in the 1.8.9 brewing tab.
///
/// Java probes the 16 low recipe bits with the two bottle forms (extended and splash) and the
/// two amplifier bits, then inserts each non-empty effect list into a `LinkedHashMap`.  The map
/// key is the complete `List<PotionEffect>`, so duplicate effect lists replace their metadata
/// value while retaining their first insertion position.  Replaying that finite 16 × 2 × 3 probe against the authoritative
/// decoder keeps the creative catalog coupled to the same requirements, durations, amplifier,
/// and splash semantics used by names and tinting.  The returned values are in Java insertion
/// order and are safe for callers to own or mutate.
#[must_use]
pub fn potion_creative_damage_values() -> Vec<i16> {
    let mut entries = Vec::<(Vec<PotionStackEffect>, i16)>::with_capacity(62);

    for base in 0..=15_i32 {
        for bottle_flag in [8_192_i32, 16_384] {
            let bottle_damage = base | bottle_flag;
            for amplifier_flag in [0_i32, 32, 64] {
                let damage = bottle_damage | amplifier_flag;
                let effects = potion_effects_from_damage(damage);
                if effects.is_empty() {
                    continue;
                }
                // Every candidate is at most 16_463 and therefore fits the signed protocol
                // metadata field.  Use an explicit checked conversion so a future table change
                // cannot silently wrap a catalog entry.
                let damage = i16::try_from(damage).expect("vanilla potion metadata fits i16");
                if let Some((_prior, value)) =
                    entries.iter_mut().find(|(prior, _)| prior == &effects)
                {
                    // `LinkedHashMap.put` replaces the value for an equal key while retaining
                    // that key's original insertion position.  This is why e.g. 8195 becomes
                    // 8227 when bit 5 is toggled without changing the decoded effect list.
                    *value = damage;
                } else {
                    entries.push((effects, damage));
                }
            }
        }
    }
    entries.into_iter().map(|(_, damage)| damage).collect()
}

/// The thirteen `PotionHelper.potionRequirements` entries, keyed by potion id.
fn potion_requirement(id: u8) -> Option<&'static str> {
    Some(match id {
        1 => "!0 & 1 & !2 & !3 & 1+6",
        2 => "!0 & 1 & !2 & 3 & 3+6",
        5 => "0 & !1 & !2 & 3 & 3+6",
        6 => "0 & !1 & 2 & !3",
        7 => "!0 & !1 & 2 & 3",
        8 => "0 & 1 & !2 & 3 & 3+6",
        10 => "0 & !1 & !2 & !3 & 0+6",
        12 => "0 & 1 & !2 & !3 & 0+6",
        13 => "0 & !1 & 2 & 3 & 2+6",
        14 => "!0 & 1 & 2 & 3 & 2+6",
        16 => "!0 & 1 & 2 & !3 & 2+6",
        18 => "!0 & !1 & !2 & 3 & 3+6",
        19 => "!0 & !1 & 2 & !3 & 2+6",
        _ => return None,
    })
}

fn potion_amplifier(id: u8) -> Option<&'static str> {
    matches!(id, 1 | 3 | 5 | 6 | 7 | 8 | 10 | 11 | 19).then_some("5")
}

/// Evaluate the subset of PotionHelper's bit-expression grammar used by its live 1.8.9 tables.
fn potion_expression_value(expression: &str, damage: i32) -> i32 {
    for alternative in expression.split('|') {
        let mut conjunction = 0;
        let mut matched = true;
        for term in alternative.split('&') {
            let value: i32 = term
                .trim()
                .split('+')
                .map(|atom| {
                    let atom = atom.trim();
                    let (negated, bit) = atom
                        .strip_prefix('!')
                        .map_or((false, atom), |bit| (true, bit));
                    let bit: u32 = bit.parse().expect("static potion bit expression");
                    let set = damage & (1_i32 << bit) != 0;
                    i32::from(if negated { !set } else { set })
                })
                .sum();
            if value <= 0 {
                matched = false;
                break;
            }
            conjunction = conjunction.max(value);
        }
        if matched && conjunction > 0 {
            return conjunction;
        }
    }
    0
}

/// An enchantment type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnchantmentInfo {
    /// Enchantment id as stored in item NBT.
    pub id: u8,
    /// Translation key, e.g. `enchantment.damage.all`.
    pub name: &'static str,
    /// `en_US` display name.
    pub display_name: &'static str,
    /// Selection weight in the enchanting table.
    pub weight: u8,
    /// Lowest level this enchantment has.
    pub min_level: u8,
    /// Highest level this enchantment has.
    pub max_level: u8,
    /// What the enchantment can be applied to.
    pub target: EnchantmentTarget,
}

/// An enchantment by id, or `None` for ids vanilla does not register.
#[inline]
#[must_use]
pub fn enchantment(id: u8) -> Option<&'static EnchantmentInfo> {
    ENCHANTMENTS.get(id as usize).and_then(Option::as_ref)
}

/// Every registered enchantment, ascending by id.
pub fn enchantments() -> impl Iterator<Item = &'static EnchantmentInfo> {
    ENCHANTMENTS.iter().flatten()
}

/// A named sound event from `sounds.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoundEvent {
    /// Event name, e.g. `mob.zombie.say`.
    pub name: &'static str,
    /// Mixer category: `master`, `music`, `record`, `weather`, `block`, `hostile`,
    /// `neutral`, `player`, `ambient`.
    pub category: &'static str,
    /// How many weighted sound files the event picks between.
    pub variants: usize,
}

/// Looks up a sound event by name. `O(log n)`, no allocation.
///
/// # Example
/// ```
/// use mc_data::sound_event;
/// assert!(sound_event("random.click").is_some());
/// assert!(sound_event("nope.not.a.sound").is_none());
/// ```
#[must_use]
pub fn sound_event(name: &str) -> Option<&'static SoundEvent> {
    SOUND_EVENTS
        .binary_search_by(|e| e.name.cmp(name))
        .ok()
        .map(|i| &SOUND_EVENTS[i])
}

/// Every sound event vanilla 1.8.9 ships, sorted by name.
#[must_use]
pub fn sound_events() -> &'static [SoundEvent] {
    &SOUND_EVENTS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_potion_and_enchantment_id_resolves_without_panic() {
        for id in 0..=u8::MAX {
            let _ = potion(id);
            let _ = enchantment(id);
        }
    }

    #[test]
    fn known_potions() {
        assert_eq!(potion(1).map(|p| p.name), Some("potion.moveSpeed"));
        assert_eq!(potion(2).map(|p| p.bad), Some(true)); // slowness
        assert_eq!(potion(6).map(|p| p.instant), Some(true)); // instant health
        assert!(potion(0).is_none());
    }

    #[test]
    fn extracted_effectiveness_preserves_vanillas_overrides_and_bad_default() {
        // Potion.java's constructor gives bad effects 0.5 and good effects 1.0, while these
        // explicit setEffectiveness calls are the non-default cases.  This both protects the
        // extractor column and prevents the duration decoder from quietly regressing to an
        // id-specific Rust fallback.
        assert_eq!(potion(1).unwrap().effectiveness, 1.0); // good default
        assert_eq!(potion(2).unwrap().effectiveness, 0.5); // bad default
        assert_eq!(potion(3).unwrap().effectiveness, 1.5); // haste override
        assert_eq!(potion(9).unwrap().effectiveness, 0.25); // nausea override
        assert_eq!(potion(15).unwrap().effectiveness, 0.25); // blindness override
        assert_ne!(
            potion(3).unwrap().effectiveness,
            potion(1).unwrap().effectiveness
        );
        assert_ne!(
            potion(9).unwrap().effectiveness,
            potion(2).unwrap().effectiveness
        );
    }

    /// Oracle vectors captured from a bootstrapped vanilla 1.8.9 `PotionHelper`, including the
    /// base/extended/amplified/splash branches which are easiest to accidentally conflate.
    #[test]
    fn potion_damage_decoder_matches_vanilla_oracle_vectors() {
        type PotionCase<'a> = (i32, &'a [(u8, i32, i32)]);
        let cases: &[PotionCase<'_>] = &[
            (0, &[]),
            (1, &[(10, 900, 0)]),
            (16, &[]),
            (8_193, &[(10, 900, 0)]),
            (8_194, &[(1, 3_600, 0)]),
            (8_195, &[(12, 3_600, 0)]),
            (8_196, &[(19, 900, 0)]),
            (8_197, &[(6, 1, 0)]),
            (8_200, &[(18, 1_800, 0)]),
            (8_201, &[(5, 3_600, 0)]),
            (8_225, &[(10, 450, 1)]),
            (8_257, &[(10, 2_400, 0)]),
            (16_386, &[(1, 2_701, 0)]),
            (16_420, &[(19, 338, 1)]),
            (16_424, &[(18, 1_351, 0)]),
            (32_767, &[]),
        ];
        for (damage, expected) in cases {
            let actual: Vec<_> = potion_effects_from_damage(*damage)
                .into_iter()
                .map(|effect| (effect.id, effect.duration, effect.amplifier))
                .collect();
            assert_eq!(&actual, expected, "metadata {damage}");
        }
    }

    #[test]
    fn creative_potion_values_match_vanilla_linked_hash_map_order() {
        // ItemPotion.getSubItems: the real 1.8.9 SUB_ITEMS_CACHE values after the
        // List<PotionEffect> equality collapse.  The first metadata-zero stack is added by
        // Item.getSubItems and is tested by mc-ui; this table is the potion override itself.
        let expected = [
            8193, 8225, 8257, 16385, 16417, 16449, 8194, 8226, 8258, 16386, 16418, 16450, 8227,
            8259, 16419, 16451, 8196, 8228, 8260, 16388, 16420, 16452, 8261, 8229, 16453, 16421,
            8230, 8262, 16422, 16454, 8232, 8264, 16424, 16456, 8201, 8233, 8265, 16393, 16425,
            16457, 8234, 8266, 16426, 16458, 8203, 8235, 8267, 16395, 16427, 16459, 8268, 8236,
            16460, 16428, 8237, 8269, 16429, 16461, 8238, 8270, 16430, 16462,
        ];
        assert_eq!(potion_creative_damage_values(), expected);
        assert!(expected
            .iter()
            .all(|&damage| { !potion_effects_from_damage(i32::from(damage)).is_empty() }));
    }

    /// C2-7: the memo tables must be invisible. Every 16-bit metadata value plus i32 extremes
    /// (sign and high bits are inert to PotionHelper) decodes identically through the memo and
    /// through the unchanged expression evaluator, and only `POTION_DAMAGE_BITS` matter.
    #[test]
    fn memoized_potion_decoders_match_the_expression_parser_for_all_i16() {
        let extremes = [
            i32::MIN,
            i32::MAX,
            -1,
            0x4000_0000,
            0x7FFF_0000,
            0x0001_0000 | 8_194,
        ];
        for damage in (i32::from(i16::MIN)..=i32::from(u16::MAX)).chain(extremes) {
            let direct = decode_potion_effects(damage);
            assert_eq!(
                potion_effects_from_damage_cached(damage),
                direct.as_slice(),
                "{damage}"
            );
            assert_eq!(potion_effects_from_damage(damage), direct, "{damage}");
            assert_eq!(
                decode_potion_effects(damage & POTION_DAMAGE_BITS),
                direct,
                "bits outside the mask must be inert: {damage}"
            );
            assert_eq!(
                potion_liquid_color_from_damage(damage),
                decode_potion_liquid_color(damage),
                "{damage}"
            );
        }
    }

    /// All 15 on-wire potion metadata bits, frozen as an output fingerprint generated from the
    /// same vanilla 1.8.9 probe as the vectors above.  This catches a one-bit requirement,
    /// splash, amplifier, duration, order, or liquid-colour mutant without making a 32K-line
    /// fixture part of the source tree.
    #[test]
    fn every_wire_potion_metadata_value_matches_vanilla_golden_fingerprint() {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64; // FNV-1a
        for damage in 0..=32_767 {
            hash_byte(&mut hash, 0xFF);
            for effect in potion_effects_from_damage(damage) {
                hash_byte(&mut hash, effect.id);
                hash_i32(&mut hash, effect.duration);
                hash_i32(&mut hash, effect.amplifier);
                hash_byte(&mut hash, u8::from(effect.splash));
            }
            hash_i32(&mut hash, potion_liquid_color_from_damage(damage) as i32);
        }
        assert_eq!(hash, 0xD6B3_8B13_5E7B_F925);
    }

    #[test]
    fn corrupt_high_bits_are_total_and_inert_except_for_the_splash_bit() {
        for damage in 0..=32_767 {
            assert_eq!(
                potion_effects_from_damage(damage | 0x4000_0000),
                potion_effects_from_damage(damage),
                "high wire garbage must not alter PotionHelper's low-bit expressions"
            );
        }
        let splash = potion_effects_from_damage(8_194 | 16_384);
        assert!(splash.iter().all(|effect| effect.splash));
        assert_eq!(
            potion_effects_from_damage(-1),
            potion_effects_from_damage(32_767)
        );
    }

    fn hash_byte(hash: &mut u64, byte: u8) {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }

    fn hash_i32(hash: &mut u64, value: i32) {
        for byte in value.to_le_bytes() {
            hash_byte(hash, byte);
        }
    }

    #[test]
    fn status_icon_index_matches_potion_java_set_icon_index_calls() {
        // java: `Potion.java`'s `setIconIndex(col, row)` -> `col + row * 8`. Values hand-checked
        // against the source, then confirmed by the extractor reading the real `Potion` class.
        assert_eq!(potion(1).unwrap().status_icon_index, Some(0)); // speed        (0, 0)
        assert_eq!(potion(8).unwrap().status_icon_index, Some(10)); // jump boost (2, 1)
        assert_eq!(potion(10).unwrap().status_icon_index, Some(7)); // regeneration (7, 0)
        assert_eq!(potion(16).unwrap().status_icon_index, Some(12)); // night vision (4, 1)
                                                                     // Health boost and absorption share a cell in vanilla — a real quirk, not a bug.
        assert_eq!(potion(21).unwrap().status_icon_index, Some(18));
        assert_eq!(potion(22).unwrap().status_icon_index, Some(18));
        // The three instant effects (heal, harm, saturation) never call `setIconIndex`, so
        // `Potion.hasStatusIcon()` is false and there is no HUD/beacon icon for them.
        assert_eq!(potion(6).unwrap().status_icon_index, None);
        assert_eq!(potion(7).unwrap().status_icon_index, None);
        assert_eq!(potion(23).unwrap().status_icon_index, None);
    }

    #[test]
    fn known_enchantments() {
        assert_eq!(
            enchantment(0).map(|e| e.name),
            Some("enchantment.protect.all")
        );
        assert_eq!(
            enchantment(16).map(|e| e.name),
            Some("enchantment.damage.all")
        );
        assert_eq!(
            enchantment(34).map(|e| e.name),
            Some("enchantment.durability")
        );
        assert!(enchantments().count() >= 22);
    }

    #[test]
    fn enchantment_ids_and_max_levels_match_java_registrations() {
        let expected = [
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 3),
            (6, 1),
            (7, 3),
            (8, 3),
            (16, 5),
            (17, 5),
            (18, 5),
            (19, 2),
            (20, 2),
            (21, 3),
            (32, 5),
            (33, 1),
            (34, 3),
            (35, 3),
            (48, 5),
            (49, 2),
            (50, 1),
            (51, 1),
            (61, 3),
            (62, 3),
        ];
        for (id, max_level) in expected {
            let info = enchantment(id).expect("Java registered enchantment");
            assert_eq!(info.id, id);
            assert_eq!(info.min_level, 1);
            assert_eq!(info.max_level, max_level, "enchantment id {id}");
        }
        for id in [9, 10, 15, 22, 31, 36, 47, 52, 60, 63, 255] {
            assert!(
                enchantment(id).is_none(),
                "Java leaves id {id} unregistered"
            );
        }
        assert_eq!(enchantments().count(), expected.len());
    }

    fn item(name: &str) -> &'static crate::ItemProps {
        crate::item_props(crate::item_by_name(name).expect("registered 1.8.9 item"))
    }

    #[test]
    fn enchantment_target_enum_keeps_all_java_variants_in_values_order() {
        let values = [
            EnchantmentTarget::All,
            EnchantmentTarget::Armor,
            EnchantmentTarget::ArmorFeet,
            EnchantmentTarget::ArmorLegs,
            EnchantmentTarget::ArmorTorso,
            EnchantmentTarget::ArmorHead,
            EnchantmentTarget::Weapon,
            EnchantmentTarget::Digger,
            EnchantmentTarget::FishingRod,
            EnchantmentTarget::Breakable,
            EnchantmentTarget::Bow,
        ];
        for (ordinal, value) in values.into_iter().enumerate() {
            assert_eq!(value as usize, ordinal);
        }
    }

    #[test]
    fn enchantment_target_predicates_cover_every_java_branch() {
        assert!(EnchantmentTarget::All.can_enchant_item(item("apple")));

        for (target, accepted, rejected) in [
            (EnchantmentTarget::Armor, "diamond_helmet", "apple"),
            (
                EnchantmentTarget::ArmorFeet,
                "diamond_boots",
                "diamond_helmet",
            ),
            (EnchantmentTarget::ArmorLegs, "iron_leggings", "iron_boots"),
            (
                EnchantmentTarget::ArmorTorso,
                "chainmail_chestplate",
                "chainmail_leggings",
            ),
            (
                EnchantmentTarget::ArmorHead,
                "golden_helmet",
                "golden_chestplate",
            ),
            (EnchantmentTarget::Weapon, "wooden_sword", "wooden_axe"),
            (EnchantmentTarget::Digger, "diamond_pickaxe", "wooden_hoe"),
            (
                EnchantmentTarget::FishingRod,
                "fishing_rod",
                "carrot_on_a_stick",
            ),
            (EnchantmentTarget::Breakable, "shears", "apple"),
            (EnchantmentTarget::Bow, "bow", "fishing_rod"),
        ] {
            assert!(
                target.can_enchant_item(item(accepted)),
                "{target:?} should accept {accepted}"
            );
            assert!(
                !target.can_enchant_item(item(rejected)),
                "{target:?} should reject {rejected}"
            );
        }

        // `ItemTool` includes these three classes, but not hoes or shears. Shears still satisfy
        // BREAKABLE because that Java branch calls Item.isDamageable before class dispatch.
        assert!(EnchantmentTarget::Digger.can_enchant_item(item("diamond_axe")));
        assert!(EnchantmentTarget::Digger.can_enchant_item(item("diamond_shovel")));
        assert!(!EnchantmentTarget::Digger.can_enchant_item(item("shears")));
        assert!(EnchantmentTarget::Breakable.can_enchant_item(item("diamond_chestplate")));
    }

    #[test]
    fn enchantment_targets_reject_the_unknown_item_sentinel() {
        let unknown = crate::item_props(crate::ItemId(32_767));
        assert!(!unknown.registered);
        for target in [
            EnchantmentTarget::All,
            EnchantmentTarget::Armor,
            EnchantmentTarget::ArmorFeet,
            EnchantmentTarget::ArmorLegs,
            EnchantmentTarget::ArmorTorso,
            EnchantmentTarget::ArmorHead,
            EnchantmentTarget::Weapon,
            EnchantmentTarget::Digger,
            EnchantmentTarget::FishingRod,
            EnchantmentTarget::Breakable,
            EnchantmentTarget::Bow,
        ] {
            assert!(
                !target.can_enchant_item(unknown),
                "{target:?} accepted the unknown-item sentinel"
            );
        }
    }

    #[test]
    fn sound_table_is_sorted_and_searchable() {
        let all = sound_events();
        assert!(!all.is_empty(), "sounds.json did not load");
        assert!(all.windows(2).all(|w| w[0].name < w[1].name));
        assert!(sound_event("random.click").is_some());
        assert!(sound_event("mob.zombie.say").is_some());
        assert!(sound_event("").is_none());
    }
}
