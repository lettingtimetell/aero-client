//! Item tooltip text: name, enchantments, lore, durability, attributes. java:
//! `ItemStack.getTooltip`.
//!
//! Every line is returned as plain text with vanilla's own `§`-formatting codes embedded
//! (matching how `ItemStack.getTooltip` itself returns a `List<String>` full of
//! `EnumChatFormatting` codes) — [`crate::font::FontRenderer`]'s `§` state machine (S15) parses
//! these directly, so a tooltip built here needs no separate "colour per line index" convention
//! at the render call site.
//!
//! # Coverage limits
//!
//! - **Item-specific information**: potion, firework, written-book, banner, record, enchanted
//!   book, spawn-egg display names, and filled-map information are reproduced below. `EntityTag`
//!   does not alter `ItemMonsterPlacer.getItemStackDisplayName` in 1.8.9, so there is no
//!   additional spawn-egg tooltip line to infer from it. Other item classes have no normal-player
//!   `addInformation` text in the audited 1.8.9 sources.
//! - **Attribute modifiers**: vanilla merges an item's *default* modifiers (a per-`Item`-class
//!   Java override — a sword's `damageVsEntity`, etc.) with any `AttributeModifiers` NBT
//!   override, per equipment slot. This port renders the `AttributeModifiers` NBT tag verbatim
//!   when present (the general, data-driven case), and additionally synthesises a baseline
//!   "Attack Damage" line for tools/swords from `mc_data::ItemProps::attack_damage` when no NBT
//!   override exists. The 1.8.9 Sharpness contribution is applied to Minecraft's built-in
//!   weapon/tool modifier UUID, including when that modifier is supplied through NBT. 1.8.9 has
//!   no default attack-speed modifier on the audited tool/sword classes. See `attribute_lines`
//!   below.
//! - **Localisation**: [`build_localized`] resolves every tooltip translation through the active
//!   [`Language`]. [`build`] keeps exact en_US fallbacks for the built-in attribute, book,
//!   banner, colour, and adventure-mode strings, while unrecognised translation keys visibly
//!   retain their Java fallback keys.

use super::stack_ext::{self, ItemStack};
use mc_assets::Language;
use mc_nbt::NbtCompound;
use mc_world::MapData;

const RESET: &str = "\u{a7}r";
const ITALIC: &str = "\u{a7}o";
const GRAY: &str = "\u{a7}7";
const DARK_GRAY: &str = "\u{a7}8";
const DARK_PURPLE: &str = "\u{a7}5";
const BLUE: &str = "\u{a7}9";
const RED: &str = "\u{a7}c";
const WHITE: &str = "\u{a7}f";
const YELLOW: &str = "\u{a7}e";
const AQUA: &str = "\u{a7}b";
const LIGHT_PURPLE: &str = "\u{a7}d";

/// java: `ItemStack.getTooltip(player, advanced)`. `advanced` is vanilla's F3+H "advanced
/// tooltips" setting: it adds the durability line, the registry id, and (for subtyped items)
/// the metadata.
#[must_use]
pub fn build(stack: &ItemStack, advanced: bool) -> Vec<String> {
    build_with_map_data(stack, advanced, None)
}

/// Build a tooltip with the `map_<metadata>` record that `ItemMap.getMapData` resolved for a
/// filled map.
///
/// `ItemStack.getTooltip` receives the player/world, whereas this UI-only builder ordinarily
/// does not.  Callers that have the live world must pass
/// `world.map_data(i32::from(stack.damage))` so advanced filled-map tooltips can reproduce
/// `ItemMap.addInformation`: an absent record says `Unknown map`; a present record gives its
/// scale.  Passing `None` is therefore the exact no-record branch, not a suppressed line.
#[must_use]
pub fn build_with_map_data(
    stack: &ItemStack,
    advanced: bool,
    map_data: Option<&MapData>,
) -> Vec<String> {
    let Some(props) = stack_ext::try_props(stack) else {
        return Vec::new();
    };
    let name = written_book_display_name(stack)
        .or_else(|| player_skull_name(stack, None))
        .or_else(|| spawn_egg_display_name(stack, None, props.display_name))
        .or_else(|| potion_display_name(stack, None))
        .or_else(|| banner_display_name(stack, None))
        .unwrap_or_else(|| props.display_name_for(stack.damage).to_owned());
    build_with_base_name(stack, advanced, &name, None, map_data)
}

/// Build a tooltip using the active language and the item's metadata-aware unlocalized name.
///
/// java: `ItemStack.getDisplayName` -> `Item.getItemStackDisplayName`; custom `display.Name`
/// still wins, while ordinary names resolve `getUnlocalizedName(stack) + ".name"` at draw time.
#[must_use]
pub fn build_localized(stack: &ItemStack, advanced: bool, language: &Language) -> Vec<String> {
    build_localized_with_map_data(stack, advanced, language, None)
}

/// Localized [`build_with_map_data`]. Map's two advanced information lines are intentionally
/// not localized: `ItemMap.addInformation` appends its English literals directly in 1.8.9.
#[must_use]
pub fn build_localized_with_map_data(
    stack: &ItemStack,
    advanced: bool,
    language: &Language,
    map_data: Option<&MapData>,
) -> Vec<String> {
    if stack_ext::try_props(stack).is_none() {
        return Vec::new();
    }
    let (name, _) = display_name_localized(stack, language);
    build_with_base_name(stack, advanced, &name, Some(language), map_data)
}

/// Resolve `ItemStack.getDisplayName()` and whether it came from `display.Name`.
/// Hosts use the second value for anvil's ordinary-name-to-empty normalization.
#[must_use]
pub fn display_name_localized(stack: &ItemStack, language: &Language) -> (String, bool) {
    let Some(props) = stack_ext::try_props(stack) else {
        return (String::new(), false);
    };
    let custom = stack
        .nbt
        .as_ref()
        .and_then(|nbt| nbt.get_compound("display"))
        .and_then(|display| display.get_str("Name"));
    if let Some(name) = custom {
        return (name.to_owned(), true);
    }
    let name = written_book_display_name(stack)
        .or_else(|| player_skull_name(stack, Some(language)))
        .or_else(|| spawn_egg_display_name(stack, Some(language), props.display_name))
        .or_else(|| potion_display_name(stack, Some(language)))
        .or_else(|| banner_display_name(stack, Some(language)))
        .unwrap_or_else(|| {
            let key = format!("{}.name", props.unlocalized_name(stack.damage));
            language.tr(&key).to_owned()
        });
    (name, false)
}

/// `ItemBanner.getItemStackDisplayName`: the base colour in `BlockEntityTag` wins over the
/// stack metadata, and hostile values follow `EnumDyeColor.byDyeDamage`'s black fallback.
fn banner_display_name(stack: &ItemStack, language: Option<&Language>) -> Option<String> {
    if stack_ext::props(stack).name != "minecraft:banner" {
        return None;
    }
    let base = stack
        .nbt
        .as_ref()
        .and_then(|root| root.get_compound("BlockEntityTag"))
        .and_then(|tag| tag.get("Base").map(|_| tag.get_i32("Base").unwrap_or(0)))
        .unwrap_or_else(|| i32::from(stack.damage));
    let color = banner_dye_name(base);
    let key = format!("item.banner.{color}.name");
    let fallback = format!("{} Banner", banner_dye_display_name(color));
    Some(tooltip_translation(language, &key, &fallback))
}

/// Java `ItemWrittenBook.getItemStackDisplayName`.
///
/// The title is a root-level `TAG_String`, not `display.Name`; a blank or wrong-typed title
/// leaves the ordinary localized item name intact.  This belongs in the `getDisplayName` core,
/// rather than the written-book tooltip information block: the HUD and anvil consume the former
/// without ever asking for the latter.  `display.Name` is deliberately checked by the caller
/// first, matching `ItemStack.getDisplayName`'s post-override custom-name precedence.
fn written_book_display_name(stack: &ItemStack) -> Option<String> {
    (stack_ext::props(stack).name == "minecraft:written_book")
        .then(|| {
            stack
                .nbt
                .as_ref()
                .and_then(|root| root.get_str("title"))
                .filter(|title| !title.is_empty())
                .map(str::to_owned)
        })
        .flatten()
}

fn build_with_base_name(
    stack: &ItemStack,
    advanced: bool,
    base_name: &str,
    language: Option<&Language>,
    map_data: Option<&MapData>,
) -> Vec<String> {
    let mut lines = Vec::new();
    let props = stack_ext::props(stack);
    let display = stack.nbt.as_ref().and_then(|n| n.get_compound("display"));
    let custom_name = display.and_then(|d| d.get_str("Name"));

    let mut name = custom_name.unwrap_or(base_name).to_string();
    if custom_name.is_some() {
        name = format!("{ITALIC}{name}");
    }
    name.push_str(RESET);
    if advanced {
        let id = props.id.0;
        if props.has_subtypes {
            name.push_str(&format!(" (#{id:04}/{})", stack.damage));
        } else {
            name.push_str(&format!(" (#{id:04})"));
        }
    } else if custom_name.is_none() && props.name == "minecraft:filled_map" {
        name.push_str(&format!(" #{}", stack.damage));
    }
    lines.push(name);

    let hide_flags = stack
        .nbt
        .as_ref()
        .and_then(|n| n.get_i32("HideFlags"))
        .unwrap_or(0);

    if hide_flags & 32 == 0 {
        // java: Item.addInformation runs before the generic enchantment/display/attribute blocks.
        lines.extend(item_information_lines(stack, language));
        lines.extend(filled_map_information_lines(stack, advanced, map_data));
    }

    if hide_flags & 1 == 0 {
        for line in enchantment_lines(stack) {
            lines.push(line);
        }
    }

    if let Some(display) = display {
        // java: ItemStack.getTooltip:720-730. This is intentionally before Lore.
        // `hasKey("color", 3)` accepts only TAG_Int, not the numeric coercion used by the
        // compound's `getInteger` after that guard succeeds.
        if let Some(mc_nbt::Nbt::Int(color)) = display.get("color") {
            if advanced {
                // `Integer.toHexString(int).toUpperCase()` renders its two's-complement value,
                // so a hostile negative colour becomes eight hex digits rather than `-000001`.
                lines.push(format!("Color: #{:X}", *color as u32));
            } else {
                lines.push(format!(
                    "{ITALIC}{}",
                    tooltip_translation(language, "item.dyed", "Dyed")
                ));
            }
        }
        if let Some(lore) = display.get_list("Lore") {
            for entry in lore {
                if let Some(text) = entry.as_str() {
                    lines.push(format!("{DARK_PURPLE}{ITALIC}{text}"));
                }
            }
        }
    }

    // java: `ItemStack.getTooltip:749` gates the whole attribute block — the blank separator
    // line included — on `(i1 & 2) == 0`. Hiding "+7 Attack Damage" with `HideFlags: 2` is the
    // standard way a server dresses up a cosmetic or menu item, so an unconditional block draws
    // two lines vanilla would not and makes the tooltip box the wrong height.
    if hide_flags & 2 == 0 {
        // java: `if (!multimap.isEmpty() && (i1 & 2) == 0) { list.add(""); ...`
        // -- the separator follows the map, not the rendered lines, so a valid modifier whose
        // amount is 0 (or NaN) still leaves the blank line behind.
        if let Some(attrs) = attribute_lines(stack, language) {
            lines.push(String::new());
            lines.extend(attrs);
        }
    }

    // java: `ItemStack.getTooltip:786` gates the Unbreakable line on `(i1 & 4) == 0`.
    if hide_flags & 4 == 0
        && stack
            .nbt
            .as_ref()
            .is_some_and(|n| n.get_bool("Unbreakable").unwrap_or(false))
    {
        lines.push(format!(
            "{BLUE}{}",
            tooltip_translation(language, "item.unbreakable", "Unbreakable")
        ));
    }

    if hide_flags & 8 == 0 {
        append_adventure_block_lines(&mut lines, stack, "CanDestroy", "item.canBreak", language);
    }
    if hide_flags & 16 == 0 {
        append_adventure_block_lines(&mut lines, stack, "CanPlaceOn", "item.canPlace", language);
    }

    if advanced {
        if stack_ext::is_item_damaged(stack) {
            // java: `getMaxDamage() - getItemDamage()` in `int` arithmetic, so a
            // stack damaged past its maximum shows a negative remainder.
            let max = props.max_damage;
            let cur = i32::from(max) - i32::from(stack.damage);
            lines.push(format!("{GRAY}Durability: {cur} / {max}"));
        }
        lines.push(format!("{DARK_GRAY}{}", props.name));
        if let Some(nbt) = &stack.nbt {
            lines.push(format!("{DARK_GRAY}NBT: {} tag(s)", nbt.len()));
        }
    }

    lines
}

/// java: `ItemMap.addInformation`.
///
/// Java shifts an `int` by `mapdata.scale`; the JVM masks that count to five bits. Normally the
/// server sends scale `0..=4`, but retaining the signed byte and `wrapping_shl` also preserves
/// the observable result for malformed packet values held in client map state.
fn filled_map_information_lines(
    stack: &ItemStack,
    advanced: bool,
    map_data: Option<&MapData>,
) -> Vec<String> {
    if !advanced || stack_ext::props(stack).name != "minecraft:filled_map" {
        return Vec::new();
    }
    let Some(map_data) = map_data else {
        return vec!["Unknown map".to_owned()];
    };
    // Java resolves exactly `"map_" + stack.getMetadata()` inside ItemMap.getMapData.  The
    // caller normally obtains this record through `World::map_data`, but keeping the identity
    // check here prevents a stale record for another hovered map from leaking its scale into
    // this tooltip if a host supplies the wrong optional record.
    if map_data.id != i32::from(stack.damage) {
        return vec!["Unknown map".to_owned()];
    }
    let scale = i32::from(map_data.scale);
    let ratio = 1_i32.wrapping_shl(scale as u32);
    vec![
        format!("Scaling at 1:{ratio}"),
        format!("(Level {scale}/4)"),
    ]
}

/// Java's `StatCollector.translateToLocal`, with the English literals used by [`build`] when no
/// resource language is available. A missing key through an actual [`Language`] deliberately
/// stays a raw key: that is `StringTranslate.tryTranslateKey`'s observable fallback.
fn tooltip_translation(language: Option<&Language>, key: &str, en_us: &str) -> String {
    language.map_or_else(|| en_us.to_owned(), |lang| lang.tr(key).to_owned())
}

/// java: the `CanDestroy` / `CanPlaceOn` tail of `ItemStack.getTooltip`.
///
/// The key must be a string list. The NBT reader's `get_list` is deliberately type-agnostic, so
/// filtering to strings here mirrors `getTagList(key, 8)`: a list of compounds is an empty list
/// to Java and emits neither the separator nor the header.
fn append_adventure_block_lines(
    lines: &mut Vec<String>,
    stack: &ItemStack,
    nbt_key: &str,
    heading_key: &str,
    language: Option<&Language>,
) {
    let Some(entries) = stack.nbt.as_ref().and_then(|nbt| nbt.get_list(nbt_key)) else {
        return;
    };
    let entries: Vec<_> = entries.iter().filter_map(|entry| entry.as_str()).collect();
    if entries.is_empty() {
        return;
    }

    let heading = match heading_key {
        "item.canBreak" => "Can break:",
        "item.canPlace" => "Can be placed on:",
        _ => unreachable!("fixed Java key"),
    };
    lines.push(String::new());
    lines.push(format!(
        "{GRAY}{}",
        tooltip_translation(language, heading_key, heading)
    ));
    lines.extend(entries.into_iter().map(|name| {
        let translated = mc_data::block_by_name(name).map_or_else(
            || "missingno".to_owned(),
            |block| {
                let info = block.info();
                let key = format!("{}.name", info.unlocalized);
                tooltip_translation(language, &key, info.display_name)
            },
        );
        format!("{DARK_GRAY}{translated}")
    }));
}

/// `ItemSkull.getItemStackDisplayName`: player heads use SkullOwner's name when present.
///
/// Java checks the NBT *type* before reading either representation: a root `TAG_String` wins over
/// a compound, while a compound only participates when its `Name` is also a `TAG_String`.  An
/// empty but correctly typed value is still present and is formatted (`"'s Head"` in en_US); only
/// a missing or wrong-typed value falls back to the ordinary metadata-specific skull name.
fn player_skull_name(stack: &ItemStack, language: Option<&Language>) -> Option<String> {
    let props = stack_ext::props(stack);
    if props.name != "minecraft:skull" || stack.damage != 3 {
        return None;
    }
    let tag = stack.nbt.as_ref()?;
    let owner = tag.get_str("SkullOwner").or_else(|| {
        tag.get_compound("SkullOwner")
            .and_then(|owner| owner.get_str("Name"))
    })?;
    Some(language.map_or_else(
        || format!("{owner}'s Head"),
        |lang| lang.format("item.skull.player.name", &[owner]),
    ))
}

/// `ItemMonsterPlacer.getItemStackDisplayName`.
///
/// Java asks `EntityList`, not the spawn-egg colour table, so any registered entity id gains a
/// suffix even when that entity has no obtainable egg. Unknown, negative, and out-of-range ids
/// keep only the translated/trimmed base item name.
fn spawn_egg_display_name(
    stack: &ItemStack,
    language: Option<&Language>,
    base_item_en_us: &str,
) -> Option<String> {
    if stack_ext::props(stack).name != "minecraft:spawn_egg" {
        return None;
    }

    let base = java_string_trim(
        language.map_or(base_item_en_us, |lang| lang.tr("item.monsterPlacer.name")),
    )
    .to_owned();
    let entity = u8::try_from(stack.damage)
        .ok()
        .and_then(mc_data::entity_type);
    let Some(entity) = entity else {
        return Some(base);
    };

    let entity_key = format!("entity.{}.name", entity.name());
    let entity_name = language.map_or_else(
        || entity.name().to_owned(),
        |lang| lang.tr(&entity_key).to_owned(),
    );
    Some(format!("{base} {entity_name}"))
}

/// `ItemPotion.getItemStackDisplayName`.
///
/// The splash word is only used when an effect supplies a postfix. An effect-less splash damage
/// value still takes the ordinary `<prefix> Potion` fallback, exactly like Java's unused `s`
/// local in that branch.
fn potion_display_name(stack: &ItemStack, language: Option<&Language>) -> Option<String> {
    if stack_ext::props(stack).name != "minecraft:potion" {
        return None;
    }
    if stack.damage == 0 {
        return Some(potion_translation(
            language,
            "item.emptyPotion.name",
            "Water Bottle",
        ));
    }

    let effects = potion_effects(stack).expect("potion item");
    if let Some(effect) = effects.first() {
        let info = mc_data::potion(effect.id).expect("potion_effects validates ids");
        let key = format!("{}.postfix", info.name);
        let postfix = potion_translation(language, &key, potion_postfix_en_us(effect.id));
        if stack.damage & 16_384 != 0 {
            let splash = potion_translation(language, "potion.prefix.grenade", "Splash");
            return Some(format!("{splash} {postfix}"));
        }
        return Some(postfix);
    }

    let (key, prefix_en_us) = POTION_PREFIXES[potion_prefix_index(i32::from(stack.damage))];
    let prefix = potion_translation(language, key, prefix_en_us);
    // `ItemPotion` calls `super.getItemStackDisplayName(stack)` in this branch.  That is the
    // ordinary `item.potion.name` translation, not the metadata-zero `item.emptyPotion.name`
    // string stored in mc-data's generic potion item record.
    let base = language.map_or("Potion", |lang| lang.tr("item.potion.name"));
    Some(format!("{prefix} {}", java_string_trim(base)))
}

fn potion_prefix_index(meta: i32) -> usize {
    [5_u32, 4, 3, 2, 1].into_iter().fold(0, |index, bit| {
        (index << 1) | usize::from(meta & (1_i32 << bit) != 0)
    })
}

const POTION_PREFIXES: [(&str, &str); 32] = [
    ("potion.prefix.mundane", "Mundane"),
    ("potion.prefix.uninteresting", "Uninteresting"),
    ("potion.prefix.bland", "Bland"),
    ("potion.prefix.clear", "Clear"),
    ("potion.prefix.milky", "Milky"),
    ("potion.prefix.diffuse", "Diffuse"),
    ("potion.prefix.artless", "Artless"),
    ("potion.prefix.thin", "Thin"),
    ("potion.prefix.awkward", "Awkward"),
    ("potion.prefix.flat", "Flat"),
    ("potion.prefix.bulky", "Bulky"),
    ("potion.prefix.bungling", "Bungling"),
    ("potion.prefix.buttered", "Buttered"),
    ("potion.prefix.smooth", "Smooth"),
    ("potion.prefix.suave", "Suave"),
    ("potion.prefix.debonair", "Debonair"),
    ("potion.prefix.thick", "Thick"),
    ("potion.prefix.elegant", "Elegant"),
    ("potion.prefix.fancy", "Fancy"),
    ("potion.prefix.charming", "Charming"),
    ("potion.prefix.dashing", "Dashing"),
    ("potion.prefix.refined", "Refined"),
    ("potion.prefix.cordial", "Cordial"),
    ("potion.prefix.sparkling", "Sparkling"),
    ("potion.prefix.potent", "Potent"),
    ("potion.prefix.foul", "Foul"),
    ("potion.prefix.odorless", "Odorless"),
    ("potion.prefix.rank", "Rank"),
    ("potion.prefix.harsh", "Harsh"),
    ("potion.prefix.acrid", "Acrid"),
    ("potion.prefix.gross", "Gross"),
    ("potion.prefix.stinky", "Stinky"),
];

fn potion_postfix_en_us(id: u8) -> &'static str {
    match id {
        1 => "Potion of Swiftness",
        2 => "Potion of Slowness",
        3 => "Potion of Haste",
        4 => "Potion of Dullness",
        5 => "Potion of Strength",
        6 => "Potion of Healing",
        7 => "Potion of Harming",
        8 => "Potion of Leaping",
        9 => "Potion of Nausea",
        10 => "Potion of Regeneration",
        11 => "Potion of Resistance",
        12 => "Potion of Fire Resistance",
        13 => "Potion of Water Breathing",
        14 => "Potion of Invisibility",
        15 => "Potion of Blindness",
        16 => "Potion of Night Vision",
        17 => "Potion of Hunger",
        18 => "Potion of Weakness",
        19 => "Potion of Poison",
        20 => "Potion of Decay",
        21 => "Potion of Health Boost",
        22 => "Potion of Absorption",
        23 => "Potion of Saturation",
        _ => "Potion",
    }
}

/// One effect decoded from a potion stack.
///
/// Metadata decoding lives in [`mc_data::PotionStackEffect`]; this re-export preserves the UI
/// stack API, which additionally handles the `CustomPotionEffects` NBT override.
pub use mc_data::PotionStackEffect;

/// Decode `ItemPotion.getEffects(ItemStack)` for a potion stack.
///
/// `None` means the item is not a potion; a potion with no effects is `Some(vec![])`. A present
/// `CustomPotionEffects` list replaces the metadata-derived list, including when it is empty.
#[must_use]
pub fn potion_effects(stack: &ItemStack) -> Option<Vec<PotionStackEffect>> {
    if !stack_ext::try_props(stack).is_some_and(|props| props.name == "minecraft:potion") {
        return None;
    }
    if let Some(custom) = stack
        .nbt
        .as_ref()
        .and_then(|nbt| nbt.get_list("CustomPotionEffects"))
    {
        return Some(
            custom
                .iter()
                .filter_map(|entry| entry.as_compound())
                .filter_map(|effect| {
                    let id = i32::from(effect.get_i8("Id").unwrap_or(0));
                    if !(0..32).contains(&id) || mc_data::potion(id as u8).is_none() {
                        return None;
                    }
                    Some(PotionStackEffect {
                        id: id as u8,
                        amplifier: i32::from(effect.get_i8("Amplifier").unwrap_or(0)),
                        duration: effect.get_i32("Duration").unwrap_or(0),
                        ambient: effect.get_bool("Ambient").unwrap_or(false),
                        // `PotionEffect.readCustomPotionEffectFromNBT` checks
                        // `hasKey("ShowParticles", 1)` before reading the flag.  Do not let the
                        // numeric-coercing getter treat TAG_Int/Float/etc. as an explicitly
                        // supplied byte: malformed or wrong-typed values default to true.
                        show_particles: if effect
                            .contains_type("ShowParticles", mc_nbt::TagType::Byte)
                        {
                            effect.get_bool("ShowParticles").unwrap_or(false)
                        } else {
                            true
                        },
                        splash: stack.damage & 16_384 != 0,
                    })
                })
                .collect(),
        );
    }

    Some(mc_data::potion_effects_from_damage(i32::from(stack.damage)))
}

/// `ItemPotion.getColorFromItemStack(stack, 0)` for the liquid-overlay tint.
///
/// Java deliberately calls `getColorFromDamage(stack.getMetadata())`, not `getEffects(stack)`,
/// so `CustomPotionEffects` (including its `ShowParticles` flag) changes `hasEffect` and tooltip
/// lines but does not change item tint. `CustomPotionColor` is not an `ItemPotion` feature in
/// 1.8.9 and is likewise ignored. `None` means the stack is not a potion. The glass-bottle layer
/// (`tint_index > 0`) is always white and remains the render caller's responsibility.
#[must_use]
pub fn potion_liquid_color(stack: &ItemStack) -> Option<u32> {
    if !stack_ext::try_props(stack).is_some_and(|props| props.name == "minecraft:potion") {
        return None;
    }
    Some(potion_liquid_color_from_damage(i32::from(stack.damage)))
}

/// `PotionHelper.getLiquidColor(damage, false)` without requiring an `ItemStack` wrapper.
/// RenderPotion consumes the spawn object's i32 damage directly, so this preserves Java's
/// decision before any protocol-stack i16 narrowing.
#[must_use]
pub fn potion_liquid_color_from_damage(damage: i32) -> u32 {
    mc_data::potion_liquid_color_from_damage(damage)
}

/// `ItemPotion.hasEffect`: custom effects count, including effects on otherwise empty metadata.
#[must_use]
pub fn potion_has_effect(stack: &ItemStack) -> bool {
    potion_effects(stack).is_some_and(|effects| !effects.is_empty())
}

fn item_information_lines(stack: &ItemStack, language: Option<&Language>) -> Vec<String> {
    let props = stack_ext::props(stack);
    if let Some(record) = mc_data::record_info(props.id) {
        // java: `ItemRecord.addInformation` appends the localised `getRecordNameLocal()` line
        // after the title, before ItemStack's generic enchantment/NBT information blocks. It
        // does not add a colour itself; `GuiScreen.renderToolTip` supplies continuation gray.
        return vec![tooltip_translation(
            language,
            record.description_key,
            record.description_en_us,
        )];
    }
    match props.name {
        "minecraft:potion" => potion_information_lines(stack, language),
        "minecraft:fireworks" => firework_rocket_information_lines(stack, language),
        "minecraft:firework_charge" => firework_star_information_lines(stack, language),
        "minecraft:written_book" => written_book_information_lines(stack, language),
        // `ItemEnchantedBook.addInformation` reads its private
        // `StoredEnchantments` list here, before ItemStack's ordinary `ench` block below.
        // A book can therefore contain enchantments without having an `ench` tag at all.
        "minecraft:enchanted_book" => enchantment_lines_for_key(stack, "StoredEnchantments"),
        "minecraft:banner" => banner_information_lines(stack, language),
        _ => Vec::new(),
    }
}

/// `ItemEditableBook.addInformation`. A present root tag always adds generation, even when it
/// lacks the field: `NBTTagCompound.getInteger` yields zero and Java translates
/// `book.generation.0`.
fn written_book_information_lines(stack: &ItemStack, language: Option<&Language>) -> Vec<String> {
    let Some(root) = stack.nbt.as_ref() else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    if let Some(author) = root.get_str("author").filter(|author| !author.is_empty()) {
        let by = language.map_or_else(
            || format!("by {author}"),
            |lang| lang.format("book.byAuthor", &[author]),
        );
        lines.push(format!("{GRAY}{by}"));
    }
    let generation = root.get_i32("generation").unwrap_or(0);
    let key = format!("book.generation.{generation}");
    let fallback = match generation {
        0 => "Original",
        1 => "Copy of original",
        2 => "Copy of a copy",
        3 => "Tattered",
        _ => key.as_str(),
    };
    lines.push(format!(
        "{GRAY}{}",
        tooltip_translation(language, &key, fallback)
    ));
    lines
}

/// `ItemBanner.addInformation`: its NBT list is compound-typed, only the first six entries are
/// shown, and unknown pattern ids are skipped. `EnumDyeColor.byDyeDamage` maps invalid values to
/// black, which is why the helper deliberately does not reject hostile Color integers.
fn banner_information_lines(stack: &ItemStack, language: Option<&Language>) -> Vec<String> {
    let Some(patterns) = stack
        .nbt
        .as_ref()
        .and_then(|root| root.get_compound("BlockEntityTag"))
        .and_then(|tag| tag.get_list("Patterns"))
    else {
        return Vec::new();
    };
    patterns
        .iter()
        .filter_map(|entry| entry.as_compound())
        .take(6)
        .filter_map(|entry| {
            let pattern = banner_pattern_name(entry.get_str("Pattern").unwrap_or(""))?;
            let color = banner_dye_name(entry.get_i32("Color").unwrap_or(0));
            let key = format!("item.banner.{pattern}.{color}");
            let fallback = format!(
                "{} {}",
                banner_dye_display_name(color),
                banner_pattern_en_us(pattern)
            );
            Some(tooltip_translation(language, &key, &fallback))
        })
        .collect()
}

/// `TileEntityBanner.EnumBannerPattern.getPatternByID`, excluding `BASE`: that id is valid for
/// hostile NBT too, and vanilla renders its translation exactly like every other pattern.
fn banner_pattern_name(id: &str) -> Option<&'static str> {
    Some(match id {
        "b" => "base",
        "bl" => "square_bottom_left",
        "br" => "square_bottom_right",
        "tl" => "square_top_left",
        "tr" => "square_top_right",
        "bs" => "stripe_bottom",
        "ts" => "stripe_top",
        "ls" => "stripe_left",
        "rs" => "stripe_right",
        "cs" => "stripe_center",
        "ms" => "stripe_middle",
        "drs" => "stripe_downright",
        "dls" => "stripe_downleft",
        "ss" => "small_stripes",
        "cr" => "cross",
        "sc" => "straight_cross",
        "bt" => "triangle_bottom",
        "tt" => "triangle_top",
        "bts" => "triangles_bottom",
        "tts" => "triangles_top",
        "ld" => "diagonal_left",
        "rd" => "diagonal_up_right",
        "lud" => "diagonal_up_left",
        "rud" => "diagonal_right",
        "mc" => "circle",
        "mr" => "rhombus",
        "vh" => "half_vertical",
        "hh" => "half_horizontal",
        "vhr" => "half_vertical_right",
        "hhb" => "half_horizontal_bottom",
        "bo" => "border",
        "cbo" => "curly_border",
        "cre" => "creeper",
        "gra" => "gradient",
        "gru" => "gradient_up",
        "bri" => "bricks",
        "sku" => "skull",
        "flo" => "flower",
        "moj" => "mojang",
        _ => return None,
    })
}

/// `EnumDyeColor.byDyeDamage(...).getUnlocalizedName()`.
fn banner_dye_name(damage: i32) -> &'static str {
    match damage {
        0 => "black",
        1 => "red",
        2 => "green",
        3 => "brown",
        4 => "blue",
        5 => "purple",
        6 => "cyan",
        7 => "silver",
        8 => "gray",
        9 => "pink",
        10 => "lime",
        11 => "yellow",
        12 => "lightBlue",
        13 => "magenta",
        14 => "orange",
        15 => "white",
        _ => "black",
    }
}

fn banner_dye_display_name(name: &str) -> &'static str {
    match name {
        "black" => "Black",
        "red" => "Red",
        "green" => "Green",
        "brown" => "Brown",
        "blue" => "Blue",
        "purple" => "Purple",
        "cyan" => "Cyan",
        "silver" => "Light Gray",
        "gray" => "Gray",
        "pink" => "Pink",
        "lime" => "Lime",
        "yellow" => "Yellow",
        "lightBlue" => "Light Blue",
        "magenta" => "Magenta",
        "orange" => "Orange",
        "white" => "White",
        _ => unreachable!("banner_dye_name returns a known dye"),
    }
}

fn banner_pattern_en_us(name: &str) -> &'static str {
    match name {
        "base" => "Base",
        "square_bottom_left" => "Base Dexter Canton",
        "square_bottom_right" => "Base Sinister Canton",
        "square_top_left" => "Chief Dexter Canton",
        "square_top_right" => "Chief Sinister Canton",
        "stripe_bottom" => "Base Fess",
        "stripe_top" => "Chief Fess",
        "stripe_left" => "Pale Dexter",
        "stripe_right" => "Pale Sinister",
        "stripe_center" => "Pale",
        "stripe_middle" => "Fess",
        "stripe_downright" => "Bend",
        "stripe_downleft" => "Bend Sinister",
        "small_stripes" => "Paly",
        "cross" => "Saltire",
        "straight_cross" => "Cross",
        "triangle_bottom" => "Chevron",
        "triangle_top" => "Inverted Chevron",
        "triangles_bottom" => "Base Indented",
        "triangles_top" => "Chief Indented",
        "diagonal_left" => "Per Bend Sinister",
        "diagonal_up_right" => "Per Bend",
        "diagonal_up_left" => "Per Bend Inverted",
        "diagonal_right" => "Per Bend Sinister Inverted",
        "circle" => "Roundel",
        "rhombus" => "Lozenge",
        "half_vertical" => "Per Pale",
        "half_horizontal" => "Per Fess",
        "half_vertical_right" => "Per Pale Inverted",
        "half_horizontal_bottom" => "Per Fess Inverted",
        "border" => "Bordure",
        "curly_border" => "Bordure Indented",
        "creeper" => "Creeper Charge",
        "gradient" => "Gradient",
        "gradient_up" => "Base Gradient",
        "bricks" => "Field Masoned",
        "skull" => "Skull Charge",
        "flower" => "Flower Charge",
        "mojang" => "Thing",
        _ => unreachable!("banner_pattern_name returns a known pattern"),
    }
}

/// `ItemFirework.addInformation`.
fn firework_rocket_information_lines(
    stack: &ItemStack,
    language: Option<&Language>,
) -> Vec<String> {
    let Some(root) = stack.nbt.as_ref() else {
        return Vec::new();
    };
    // Java's getCompoundTag returns an empty compound on a miss/wrong type. That distinction is
    // unobservable for rockets because both optional children then read empty.
    let Some(fireworks) = root.get_compound("Fireworks") else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    if let Some(flight) = fireworks.get_i8("Flight") {
        let label = firework_translation(language, "item.fireworks.flight", "Flight Duration:");
        lines.push(format!("{label} {flight}"));
    }
    if let Some(explosions) = fireworks.get_list("Explosions") {
        for explosion in explosions.iter().filter_map(|entry| entry.as_compound()) {
            let mut explosion_lines =
                firework_explosion_information_lines(Some(explosion), language);
            for line in explosion_lines.iter_mut().skip(1) {
                line.insert_str(0, "  ");
            }
            lines.extend(explosion_lines);
        }
    }
    lines
}

/// `ItemFireworkCharge.addInformation`.
fn firework_star_information_lines(stack: &ItemStack, language: Option<&Language>) -> Vec<String> {
    let root = match stack.nbt.as_ref() {
        Some(root) => root,
        None => return Vec::new(),
    };
    // Unlike rockets, the empty-compound fallback is visible here: any root tag without an
    // Explosion child still emits Type=0 (Small Ball).
    firework_explosion_information_lines(root.get_compound("Explosion"), language)
}

fn firework_explosion_information_lines(
    explosion: Option<&NbtCompound>,
    language: Option<&Language>,
) -> Vec<String> {
    let kind = explosion
        .and_then(|compound| compound.get_i8("Type"))
        .unwrap_or(0);
    let (kind_key, kind_en_us) = match kind {
        0 => ("item.fireworksCharge.type.0", "Small Ball"),
        1 => ("item.fireworksCharge.type.1", "Large Ball"),
        2 => ("item.fireworksCharge.type.2", "Star-shaped"),
        3 => ("item.fireworksCharge.type.3", "Creeper-shaped"),
        4 => ("item.fireworksCharge.type.4", "Burst"),
        _ => ("item.fireworksCharge.type", "Unknown Shape"),
    };
    let kind_name = firework_translation(language, kind_key, kind_en_us);
    let mut lines = vec![java_string_trim(&kind_name).to_owned()];

    let colors = explosion
        .and_then(|compound| compound.get_ints("Colors"))
        .unwrap_or(&[]);
    if !colors.is_empty() {
        lines.push(
            colors
                .iter()
                .map(|color| firework_color_name(*color, language))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    let fade_colors = explosion
        .and_then(|compound| compound.get_ints("FadeColors"))
        .unwrap_or(&[]);
    if !fade_colors.is_empty() {
        let label = firework_translation(language, "item.fireworksCharge.fadeTo", "Fade to");
        let colors = fade_colors
            .iter()
            .map(|color| firework_color_name(*color, language))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("{label} {colors}"));
    }

    if explosion
        .and_then(|compound| compound.get_bool("Trail"))
        .unwrap_or(false)
    {
        lines.push(firework_translation(
            language,
            "item.fireworksCharge.trail",
            "Trail",
        ));
    }
    if explosion
        .and_then(|compound| compound.get_bool("Flicker"))
        .unwrap_or(false)
    {
        lines.push(firework_translation(
            language,
            "item.fireworksCharge.flicker",
            "Twinkle",
        ));
    }
    lines
}

/// `ItemFireworkCharge.getColorFromItemStack(stack, 1)` for the overlay layer.
///
/// Returns `None` for every other item. Missing, malformed, and explicitly empty Colors arrays
/// use vanilla's neutral star colour. Vanilla itself divides by zero for the explicitly-empty
/// array case; accepting hostile server NBT without crashing is the sole deliberate robustness
/// extension here.
#[must_use]
pub fn firework_star_color(stack: &ItemStack) -> Option<u32> {
    if !stack_ext::try_props(stack).is_some_and(|props| props.name == "minecraft:firework_charge") {
        return None;
    }
    let colors = stack
        .nbt
        .as_ref()
        .and_then(|root| root.get_compound("Explosion"))
        .and_then(|explosion| explosion.get_ints("Colors"));
    let Some(colors) = colors.filter(|colors| !colors.is_empty()) else {
        return Some(0x8A_8A_8A);
    };
    if colors.len() == 1 {
        return Some(colors[0] as u32);
    }

    let mut red = 0_i32;
    let mut green = 0_i32;
    let mut blue = 0_i32;
    for color in colors {
        let color = *color as u32;
        red = red.wrapping_add(((color & 0xFF_00_00) >> 16) as i32);
        green = green.wrapping_add(((color & 0x00_FF_00) >> 8) as i32);
        blue = blue.wrapping_add((color & 0x00_00_FF) as i32);
    }
    let count = colors.len() as i32;
    Some((((red / count) << 16) | ((green / count) << 8) | (blue / count)) as u32)
}

// `ItemDye.dyeColors` in dye-damage order, paired with
// `EnumDyeColor.byDyeDamage(j).getUnlocalizedName()`.
const FIREWORK_DYE_COLORS: [(i32, &str, &str); 16] = [
    (1_973_019, "black", "Black"),
    (11_743_532, "red", "Red"),
    (3_887_386, "green", "Green"),
    (5_320_730, "brown", "Brown"),
    (2_437_522, "blue", "Blue"),
    (8_073_150, "purple", "Purple"),
    (2_651_799, "cyan", "Cyan"),
    (11_250_603, "silver", "Light Gray"),
    (4_408_131, "gray", "Gray"),
    (14_188_952, "pink", "Pink"),
    (4_312_372, "lime", "Lime"),
    (14_602_026, "yellow", "Yellow"),
    (6_719_955, "lightBlue", "Light Blue"),
    (12_801_229, "magenta", "Magenta"),
    (15_435_844, "orange", "Orange"),
    (15_790_320, "white", "White"),
];

fn firework_color_name(color: i32, language: Option<&Language>) -> String {
    if let Some((_, suffix, en_us)) = FIREWORK_DYE_COLORS.iter().find(|entry| entry.0 == color) {
        let key = format!("item.fireworksCharge.{suffix}");
        return firework_translation(language, &key, en_us);
    }
    firework_translation(language, "item.fireworksCharge.customColor", "Custom")
}

fn firework_translation(language: Option<&Language>, key: &str, en_us: &str) -> String {
    language.map_or_else(|| en_us.to_owned(), |lang| lang.tr(key).to_owned())
}

/// Java 8 `String.trim`: remove only code points at or below U+0020. Rust's Unicode-aware
/// `str::trim` would incorrectly remove resource-pack NBSP and other non-ASCII spacing.
fn java_string_trim(value: &str) -> &str {
    value.trim_matches(|character| character <= '\u{20}')
}

fn potion_information_lines(stack: &ItemStack, language: Option<&Language>) -> Vec<String> {
    if stack.damage == 0 {
        return Vec::new();
    }
    let Some(effects) = potion_effects(stack) else {
        return Vec::new();
    };
    if effects.is_empty() {
        return vec![format!(
            "{GRAY}{}",
            potion_translation(language, "potion.empty", "No Effects")
        )];
    }

    let mut lines = Vec::new();
    let mut attributes = Vec::new();
    for effect in effects {
        let info = mc_data::potion(effect.id).expect("potion_effects validates ids");
        let mut text = potion_translation(language, info.name, info.display_name);
        if effect.amplifier > 0 {
            let key = format!("potion.potency.{}", effect.amplifier);
            let fallback = match effect.amplifier {
                1 => "II",
                2 => "III",
                3 => "IV",
                _ => key.as_str(),
            };
            text.push(' ');
            text.push_str(&potion_translation(language, &key, fallback));
        }
        if effect.duration > 20 {
            let seconds = effect.duration / 20;
            text.push_str(&format!(" ({}:{:02})", seconds / 60, seconds % 60));
        }
        lines.push(format!("{}{text}", if info.bad { RED } else { GRAY }));
        if let Some(attribute) = potion_attribute(effect.id, effect.amplifier) {
            attributes.push(attribute);
        }
    }

    if !attributes.is_empty() {
        lines.push(String::new());
        lines.push(format!(
            "{DARK_PURPLE}{}",
            tooltip_translation(language, "potion.effects.whenDrank", "When Applied:")
        ));
        lines.extend(
            attributes
                .into_iter()
                .map(|attribute| potion_attribute_line(language, attribute)),
        );
    }
    lines
}

#[derive(Debug, Clone, Copy)]
struct PotionAttribute {
    name_key: &'static str,
    name_en_us: &'static str,
    amount: f64,
    operation: i32,
}

fn potion_attribute(id: u8, amplifier: i32) -> Option<PotionAttribute> {
    let scale = f64::from(amplifier + 1);
    let (name_key, name_en_us, amount, operation) = match id {
        1 => (
            "attribute.name.generic.movementSpeed",
            "Speed",
            0.200_000_002_980_232_24 * scale,
            2,
        ),
        2 => (
            "attribute.name.generic.movementSpeed",
            "Speed",
            -0.150_000_005_960_464_48 * scale,
            2,
        ),
        // PotionAttackDamage overrides the registered 2.5 amount with 1.3 per level.
        5 => (
            "attribute.name.generic.attackDamage",
            "Attack Damage",
            1.3 * scale,
            2,
        ),
        // The same override makes Weakness -0.5 per level with operation zero.
        18 => (
            "attribute.name.generic.attackDamage",
            "Attack Damage",
            -0.5 * scale,
            0,
        ),
        21 => (
            "attribute.name.generic.maxHealth",
            "Max Health",
            4.0 * scale,
            0,
        ),
        _ => return None,
    };
    // Java only emits a modifier line when its scaled value is strictly positive or negative.
    // A custom effect may legally carry amplifier -1, which scales its modifier to exact zero.
    if amount == 0.0 {
        return None;
    }
    Some(PotionAttribute {
        name_key,
        name_en_us,
        amount,
        operation,
    })
}

fn potion_attribute_line(language: Option<&Language>, attribute: PotionAttribute) -> String {
    let positive = attribute.amount > 0.0;
    let display_amount = if matches!(attribute.operation, 1 | 2) {
        attribute.amount.abs() * 100.0
    } else {
        attribute.amount.abs()
    };
    let amount = format_amount(display_amount);
    // Unlike potion effect names/potencies, Java does not call `trim()` for the attribute
    // name or the `When Applied:` header. Preserve resource-pack whitespace exactly.
    let name = tooltip_translation(language, attribute.name_key, attribute.name_en_us);
    let key = format!(
        "attribute.modifier.{}.{}",
        if positive { "plus" } else { "take" },
        attribute.operation
    );
    let fallback = match (positive, attribute.operation) {
        (true, 1 | 2) => format!("+{amount}% {name}"),
        (true, _) => format!("+{amount} {name}"),
        (false, 1 | 2) => format!("-{amount}% {name}"),
        (false, _) => format!("-{amount} {name}"),
    };
    let rendered = language.map_or(fallback, |lang| lang.format(&key, &[&amount, &name]));
    format!("{}{rendered}", if positive { BLUE } else { RED })
}

fn potion_translation(language: Option<&Language>, key: &str, en_us: &str) -> String {
    language.map_or_else(
        || en_us.to_owned(),
        |lang| java_string_trim(lang.tr(key)).to_owned(),
    )
}

/// java: `EnumRarity` (`item/EnumRarity.java:7-10`) - four values, each carrying the
/// `EnumChatFormatting` that colours a tooltip's **first** line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rarity {
    /// `EnumChatFormatting.WHITE`.
    Common,
    /// `EnumChatFormatting.YELLOW`.
    Uncommon,
    /// `EnumChatFormatting.AQUA`.
    Rare,
    /// `EnumChatFormatting.LIGHT_PURPLE`.
    Epic,
}

impl Rarity {
    /// The section-sign code `EnumRarity.rarityColor` resolves to.
    ///
    /// `Common` is an explicit white and not "no code": the two look identical against
    /// [`super::tooltip_box`]'s `-1` default, but vanilla *prefixes* the line with it, and on a
    /// first line that already carries formatting of its own (a `display.Name` with a colour in
    /// it) that prefix is a real reset to white.
    #[must_use]
    pub const fn color_code(self) -> &'static str {
        match self {
            Rarity::Common => WHITE,
            Rarity::Uncommon => YELLOW,
            Rarity::Rare => AQUA,
            Rarity::Epic => LIGHT_PURPLE,
        }
    }
}

/// java: `ItemStack.getRarity()` (`:864-867`) delegating to `Item.getRarity(stack)`
/// (`Item.java:424-427`), which is `isItemEnchanted() ? RARE : COMMON`.
///
/// 1.8.9 has exactly three overrides - verified by grepping `EnumRarity getRarity` across
/// `net/minecraft`:
///
/// * `ItemRecord.getRarity` (`:76-79`) - `RARE`, unconditionally, for all twelve discs;
/// * `ItemAppleGold.getRarity` (`:26-29`) - `RARE` at metadata `0`, `EPIC` otherwise (the
///   "enchanted" apple), so a golden apple is **never** `COMMON`;
/// * `ItemEnchantedBook.getRarity` (`:32-35`) - `UNCOMMON` when `StoredEnchantments` holds at
///   least one entry, else `super`.
#[must_use]
pub fn rarity(stack: &ItemStack) -> Rarity {
    let Some(props) = stack_ext::try_props(stack) else {
        return Rarity::Common;
    };
    let name = props.name;
    if props.kind == mc_data::ItemKind::Record {
        return Rarity::Rare;
    }
    if name == "minecraft:golden_apple" {
        return if stack.damage > 0 {
            Rarity::Epic
        } else {
            Rarity::Rare
        };
    }
    if name == "minecraft:enchanted_book"
        && stack
            .nbt
            .as_ref()
            .and_then(|n| n.get_list("StoredEnchantments"))
            .is_some_and(|l| !l.is_empty())
    {
        return Rarity::Uncommon;
    }
    if is_item_enchanted(stack) {
        Rarity::Rare
    } else {
        Rarity::Common
    }
}

/// java: `ItemStack.isItemEnchanted` (`:902-905`) - `hasKey("ench", 9)`, i.e. the key is
/// present **and** is a `TAG_List`. Its contents are never inspected, so an empty `ench` list
/// still counts as enchanted.
#[must_use]
pub fn is_item_enchanted(stack: &ItemStack) -> bool {
    stack
        .nbt
        .as_ref()
        .and_then(|n| n.get_list("ench"))
        .is_some()
}

/// Java `ItemStack.hasEffect()` / `Item.hasEffect(ItemStack)`.
///
/// The base implementation is [`is_item_enchanted`], but 1.8.9 has four item-class overrides
/// relevant to the renderer: potions use their decoded effects; enchanted books and experience
/// bottles foil unconditionally; and only the metadata-one golden apple foils. This predicate is
/// deliberately separate from [`rarity`]: an empty enchanted book is common in the tooltip yet
/// still has its always-on `ItemEnchantedBook.hasEffect` glint.
#[must_use]
pub fn has_effect(stack: &ItemStack) -> bool {
    let Some(props) = stack_ext::try_props(stack) else {
        return false;
    };
    match props.name {
        "minecraft:potion" => potion_has_effect(stack),
        "minecraft:enchanted_book" | "minecraft:experience_bottle" => true,
        "minecraft:golden_apple" => stack.damage > 0,
        _ => is_item_enchanted(stack),
    }
}

/// java: the `ench` NBT-list branch of `getTooltip`.
///
/// `ItemStack.getTooltip` returns these lines without a colour code. `GuiScreen.renderToolTip`
/// prefixes every continuation line with gray later; this crate's equivalent is
/// `tooltip_box::recolour_tooltip_lines`. Embedding gray here too would produce duplicate `§7§7`
/// prefixes on the live container and creative paths.
fn enchantment_lines(stack: &ItemStack) -> Vec<String> {
    enchantment_lines_for_key(stack, "ench")
}

/// Java `EnchantmentHelper.getEnchantments`: the item-specific enchanted-book hook uses
/// `StoredEnchantments`, whereas `ItemStack.getTooltip` itself uses `ench`.
fn enchantment_lines_for_key(stack: &ItemStack, key: &str) -> Vec<String> {
    let Some(list) = stack.nbt.as_ref().and_then(|n| n.get_list(key)) else {
        return Vec::new();
    };
    list.iter()
        .filter_map(|entry| entry.as_compound())
        .filter_map(|c| {
            let id = c.get_i16("id")?;
            let level = c.get_i16("lvl").unwrap_or(0);
            if !(0..=255).contains(&id) {
                return None;
            }
            let info = mc_data::enchantment(id as u8)?;
            Some(enchantment_display_name(info, level))
        })
        .collect()
}

/// java: `Enchantment.getTranslatedName` (`enchantment/Enchantment.java:217-221`):
///
/// ```text
/// String s = StatCollector.translateToLocal(this.getName());
/// return s + " " + StatCollector.translateToLocal("enchantment.level." + level);
/// ```
///
/// **Unconditional.** The `level == 1 && getMaxLevel() == 1` short-circuit that drops the
/// numeral for Silk Touch / Infinity / Flame / Aqua Affinity is a **1.9** addition; 1.8.9
/// renders "Silk Touch I". This function used to implement the 1.9 rule and its own doc
/// asserted it was vanilla.
fn enchantment_display_name(info: &mc_data::EnchantmentInfo, level: i16) -> String {
    format!("{} {}", info.display_name, enchantment_level_token(level))
}

/// java: `StatCollector.translateToLocal("enchantment.level." + level)`, resolved against
/// `assets/minecraft/lang/en_US.lang`.
///
/// The lang file defines `enchantment.level.1` .. `enchantment.level.10` and **nothing else**
/// (read out of `jars/versions/1.8.9/1.8.9.jar`), and `StringTranslate.tryTranslateKey`
/// (`util/StringTranslate.java:110-114`) returns the *key itself* when a lookup misses. So a
/// server's `lvl: 100` renders as the literal `enchantment.level.100`, and `lvl: 0` as
/// `enchantment.level.0` - a recognisable signature that this port previously replaced with an
/// invented numeral (`XXXXXXXXXX` for 100, a decimal for anything over 3999).
///
/// The ten values are inlined rather than looked up: this crate's tooltip builder has no
/// `Language` (see the module doc's localisation note), and hard-coding the en_US strings is
/// what every other string in this file already does. The *fallback* is the part that matters
/// and it is exact.
#[must_use]
fn enchantment_level_token(level: i16) -> String {
    const EN_US: [&str; 10] = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
    match level {
        1..=10 => EN_US[level as usize - 1].to_string(),
        other => format!("enchantment.level.{other}"),
    }
}

/// See the module doc's "Attribute modifiers" note: renders an explicit `AttributeModifiers`
/// NBT tag verbatim when present, otherwise synthesises a baseline Attack Damage line for
/// tools/swords.
///
/// `None` is Java's empty `getAttributeModifiers()` map; `Some` may still hold no line when every
/// valid modifier's amount is zero (see the caller).
fn attribute_lines(stack: &ItemStack, language: Option<&Language>) -> Option<Vec<String>> {
    let creature_modifier = modifier_for_undefined_creature(stack);
    if let Some(list) = stack
        .nbt
        .as_ref()
        .and_then(|n| n.get_list("AttributeModifiers"))
    {
        let mut in_map = false;
        let mut lines = Vec::new();
        for modifier in list.iter().filter_map(|entry| entry.as_compound()) {
            if let Some(line) = attribute_line_from_nbt(modifier, creature_modifier, language) {
                in_map = true;
                lines.extend(line);
            }
        }
        return in_map.then_some(lines);
    }
    let props = stack_ext::props(stack);
    if props.is_tool() && props.attack_damage > 0.0 && props.tool_material.is_some() {
        // java: `ItemStack.getTooltip` prints `attributemodifier.getAmount()` **unmodified**
        // (`ItemStack.java:756,775`), and `ItemTool`/`ItemSword` already put the *total* on the
        // map -- `ItemSword.attackDamage = 4.0F + material.getDamageVsEntity()`
        // (`ItemSword.java:29`), `ItemTool.damageVsEntity = n + material.getDamageVsEntity()`.
        // `mc_data`'s `attack_damage` holds that same total (diamond_sword `7.0`,
        // iron_axe `5.0`), so there is nothing to add: a `1.0 +` here read `+8`/`+6` where
        // vanilla reads `+7`/`+5`. The only term vanilla *does* add is
        // `EnchantmentHelper.getModifierForCreature` for the `itemModifierUUID` modifier, i.e.
        // Sharpness. `EnumCreatureAttribute.UNDEFINED` makes Smite and Bane contribute zero.
        return Some(vec![format_attribute_line(
            "generic.attackDamage",
            f64::from(props.attack_damage) + creature_modifier,
            0,
            language,
        )]);
    }
    None
}

/// java: `EnchantmentHelper.getModifierForCreature(stack, UNDEFINED)` iterates every `ench`
/// compound and sums `EnchantmentDamage.calcDamageByCreature`. Only Sharpness (id 16) is
/// non-zero for `UNDEFINED`, at `level * 1.25`; duplicate tags therefore deliberately add twice.
fn modifier_for_undefined_creature(stack: &ItemStack) -> f64 {
    stack
        .nbt
        .as_ref()
        .and_then(|n| n.get_list("ench"))
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.as_compound())
        .filter(|enchantment| enchantment.get_i16("id") == Some(16))
        .map(|enchantment| f64::from(enchantment.get_i16("lvl").unwrap_or(0)) * 1.25)
        .sum()
}

/// `Item.itemModifierUUID`, split exactly as Java's `UUIDMost` / `UUIDLeast` NBT pair.
const ITEM_MODIFIER_UUID_MOST: i64 = -3_801_225_194_067_177_672;
const ITEM_MODIFIER_UUID_LEAST: i64 = -6_586_624_321_849_018_929;

/// `None` when Java's reader rejects the modifier (it never enters the map); `Some(None)` for a
/// valid modifier that renders no line.
fn attribute_line_from_nbt(
    c: &mc_nbt::NbtCompound,
    creature_modifier: f64,
    language: Option<&Language>,
) -> Option<Option<String>> {
    // java: `SharedMonsterAttributes.readAttributeModifierFromNBT` constructs an
    // `AttributeModifier` first. Its constructor rejects an empty `Name` and operations outside
    // 0..=2; `ItemStack.getAttributeModifiers` then rejects a UUID with either half zero.
    let modifier_name = c.get_str("Name").unwrap_or("");
    let operation = c.get_i32("Operation").unwrap_or(0);
    let uuid_most = c.get_i64("UUIDMost").unwrap_or(0);
    let uuid_least = c.get_i64("UUIDLeast").unwrap_or(0);
    if modifier_name.is_empty()
        || !(0..=2).contains(&operation)
        || uuid_most == 0
        || uuid_least == 0
    {
        return None;
    }
    // `NBTTagCompound.getString` returns `""` for a missing/non-string AttributeName; unlike
    // the modifier's own `Name`, Java does not validate this map key.
    let name = c.get_str("AttributeName").unwrap_or("");
    let mut amount = c.get_f64("Amount").unwrap_or(0.0);
    if uuid_most == ITEM_MODIFIER_UUID_MOST && uuid_least == ITEM_MODIFIER_UUID_LEAST {
        amount += creature_modifier;
    }
    // Java's two rendering branches are `d0 > 0` / `d0 < 0`; NaN satisfies neither.
    if amount == 0.0 || amount.is_nan() {
        return Some(None);
    }
    Some(Some(format_attribute_line(
        name, amount, operation, language,
    )))
}

/// The attribute section of `ItemStack.getTooltip`: translates both the attribute-map key and
/// the plus/take template. The fallback names cover vanilla's five shared attributes; unknown
/// server keys retain `attribute.name.<key>` as Java does when a language table has no entry.
fn format_attribute_line(
    attribute: &str,
    amount: f64,
    operation: i32,
    language: Option<&Language>,
) -> String {
    let positive = amount > 0.0;
    let shown = if matches!(operation, 1 | 2) {
        amount * 100.0
    } else {
        amount
    };
    let shown = format_amount(if positive { shown } else { -shown });
    let name_key = format!("attribute.name.{attribute}");
    let name_fallback = match attribute {
        "generic.maxHealth" => "Max Health",
        "generic.followRange" => "Follow Range",
        "generic.knockbackResistance" => "Knockback Resistance",
        "generic.movementSpeed" => "Speed",
        "generic.attackDamage" => "Attack Damage",
        _ => name_key.as_str(),
    };
    let name = tooltip_translation(language, &name_key, name_fallback);
    let modifier_key = format!(
        "attribute.modifier.{}.{}",
        if positive { "plus" } else { "take" },
        operation
    );
    let fallback = match (positive, operation) {
        (true, 1 | 2) => format!("+{shown}% {name}"),
        (true, _) => format!("+{shown} {name}"),
        (false, 1 | 2) => format!("-{shown}% {name}"),
        (false, _) => format!("-{shown} {name}"),
    };
    let text = language.map_or(fallback, |lang| {
        lang.format(&modifier_key, &[&shown, &name])
    });
    format!("{}{text}", if positive { BLUE } else { RED })
}

fn format_amount(v: f64) -> String {
    // java: `DECIMALFORMAT` is `#.###` - up to three fraction digits, trimmed, HALF_EVEN.
    // Formatting directly to three decimal places uses ties-to-even rounding. Do not pre-round
    // with `f64::round`: it is half-away-from-zero and changes exact ties such as 0.0625
    // (Java's `DecimalFormat("#.###")` renders that as `0.062`).
    if v.is_nan() {
        return "NaN".to_owned();
    }
    if v.is_infinite() {
        return if v.is_sign_negative() {
            "-\u{221e}".to_owned()
        } else {
            "\u{221e}".to_owned()
        };
    }
    let mut s = format!("{v:.3}");
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_nbt::{Nbt, NbtCompound};

    fn stack(name: &str) -> ItemStack {
        let id = mc_data::item_by_name(name).unwrap();
        ItemStack {
            item_id: id.0 as i16,
            count: 1,
            damage: 0,
            nbt: None,
        }
    }

    #[test]
    fn unregistered_item_ids_have_no_tooltip() {
        for raw_id in [8_u16, 32_767, 60_000] {
            let stack = ItemStack {
                item_id: raw_id as i16,
                count: 1,
                damage: 7,
                nbt: None,
            };
            assert!(build(&stack, false).is_empty(), "unregistered id {raw_id}");
            assert!(build(&stack, true).is_empty(), "unregistered id {raw_id}");
        }
    }

    #[test]
    fn unknown_item_public_presentation_helpers_are_inert() {
        let mut unknown = ItemStack {
            item_id: 32_767,
            count: 1,
            damage: 7,
            nbt: None,
        };
        let language = Language::new("en_US");

        assert_eq!(
            display_name_localized(&unknown, &language),
            (String::new(), false)
        );
        assert_eq!(potion_effects(&unknown), None);
        assert_eq!(potion_liquid_color(&unknown), None);
        assert_eq!(firework_star_color(&unknown), None);
        assert_eq!(rarity(&unknown), Rarity::Common);
        assert!(!has_effect(&unknown));

        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(Vec::new()));
        unknown.nbt = Some(root);
        assert_eq!(rarity(&unknown), Rarity::Common);
        assert!(!has_effect(&unknown));
    }

    fn potion_stack(meta: i16) -> ItemStack {
        let mut stack = stack("potion");
        stack.damage = meta;
        stack
    }

    fn custom_potion(meta: i16, id: i8, amplifier: i8, duration: i32) -> ItemStack {
        custom_potion_sequence(meta, &[(id, amplifier, duration)])
    }

    fn custom_potion_sequence(meta: i16, effects: &[(i8, i8, i32)]) -> ItemStack {
        let effects = effects
            .iter()
            .map(|&(id, amplifier, duration)| {
                let mut effect = NbtCompound::new();
                effect.insert("Id", Nbt::Byte(id));
                effect.insert("Amplifier", Nbt::Byte(amplifier));
                effect.insert("Duration", Nbt::Int(duration));
                Nbt::Compound(effect)
            })
            .collect();
        let mut root = NbtCompound::new();
        root.insert("CustomPotionEffects", Nbt::list(effects));
        let mut stack = potion_stack(meta);
        stack.nbt = Some(root);
        stack
    }

    fn empty_custom_potion(meta: i16) -> ItemStack {
        let mut root = NbtCompound::new();
        root.insert("CustomPotionEffects", Nbt::list(vec![]));
        let mut stack = potion_stack(meta);
        stack.nbt = Some(root);
        stack
    }

    #[test]
    fn plain_item_tooltip_is_just_the_display_name() {
        // "stone" is deliberately avoided here: `mc_data`'s extractor leaves a handful of
        // block-derived items' `display_name` as the raw unlocalised key (a pre-existing S01
        // gap, not something this section can fix without touching `mc-data`) — "apple" is a
        // plain `Item` (not `ItemBlock`) and resolves normally.
        let s = stack("apple");
        let lines = build(&s, false);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("Apple"), "{:?}", lines[0]);
        assert!(lines[0].ends_with(RESET));
    }

    /// `ItemMap.addInformation` is only an advanced-tooltip override. Its state comes from the
    /// world's `map_<metadata>` record rather than the stack NBT, and `HideFlags: 32` suppresses
    /// the entire override because `ItemStack.getTooltip` owns that gate.
    #[test]
    fn filled_map_advanced_tooltip_uses_its_live_map_record() {
        let mut map = stack("filled_map");
        map.damage = 37;
        let mut data = MapData::new(37);
        data.scale = 3;

        let ordinary = build_with_map_data(&map, false, Some(&data));
        assert_eq!(
            ordinary.len(),
            1,
            "ordinary map keeps only its #metadata name"
        );

        let known = build_with_map_data(&map, true, Some(&data));
        assert_eq!(
            &known[1..3],
            ["Scaling at 1:8", "(Level 3/4)"],
            "ItemMap.addInformation appends both exact hard-coded lines"
        );

        let unknown = build_with_map_data(&map, true, None);
        assert!(unknown.iter().any(|line| line == "Unknown map"));
        assert!(!unknown.iter().any(|line| line.starts_with("Scaling at")));

        let mut wrong_record = MapData::new(36);
        wrong_record.scale = 0;
        let wrong = build_with_map_data(&map, true, Some(&wrong_record));
        assert_eq!(
            wrong[1], "Unknown map",
            "ItemMap.getMapData must not consume another map_<metadata> record"
        );
        assert!(!wrong.iter().any(|line| line.starts_with("Scaling at")));

        let mut hidden = map.clone();
        let mut nbt = NbtCompound::new();
        nbt.insert("HideFlags", Nbt::Int(32));
        hidden.nbt = Some(nbt);
        let hidden_lines = build_with_map_data(&hidden, true, Some(&data));
        assert!(
            !hidden_lines
                .iter()
                .any(|line| line.starts_with("Scaling at") || line.starts_with("(Level")),
            "bit 32 suppresses Item.addInformation, including ItemMap's override"
        );
    }

    #[test]
    fn filled_map_scale_uses_java_int_shift_masking() {
        let map = stack("filled_map");
        let mut data = MapData::new(0);
        data.scale = -1;
        let lines = build_with_map_data(&map, true, Some(&data));
        assert!(lines.iter().any(|line| line == "Scaling at 1:-2147483648"));
        assert!(lines.iter().any(|line| line == "(Level -1/4)"));
    }

    /// `ItemMap.addInformation` is selected by the filled-map item identity, while the map
    /// suffix belongs only to the ordinary (non-advanced, non-custom-name) title branch.  The
    /// enclosing `ItemStack.getTooltip` gate reads `HideFlags` with `hasKey(..., 99)` and then
    /// `getInteger`, so every numeric NBT type must suppress bit 32 just like TAG_Int; a textual
    /// value must not accidentally become a flag.
    #[test]
    fn filled_map_tooltip_preserves_name_guard_and_java_numeric_hide_flags() {
        let mut map = stack("filled_map");
        map.damage = 9;
        let mut data = MapData::new(9);
        data.scale = 2;

        let mut custom = map.clone();
        let mut display = NbtCompound::new();
        display.insert("Name", Nbt::String("Atlas".to_owned()));
        let mut custom_root = NbtCompound::new();
        custom_root.insert("display", Nbt::Compound(display));
        custom.nbt = Some(custom_root);

        let ordinary = build_with_map_data(&custom, false, Some(&data));
        assert_eq!(ordinary, vec![format!("{ITALIC}Atlas{RESET}")]);

        let advanced = build_with_map_data(&custom, true, Some(&data));
        assert_eq!(
            advanced[0],
            format!("{ITALIC}Atlas{RESET} (#0358/9)"),
            "a custom name suppresses only the ordinary map suffix"
        );
        assert_eq!(&advanced[1..3], ["Scaling at 1:4", "(Level 2/4)"]);

        let numeric_hide_flags = [
            Nbt::Byte(32),
            Nbt::Short(32),
            Nbt::Int(32),
            Nbt::Long(32),
            Nbt::Float(32.9),
            Nbt::Double(32.9),
        ];
        for hide_flags in numeric_hide_flags {
            let mut hidden = map.clone();
            let mut root = NbtCompound::new();
            root.insert("HideFlags", hide_flags.clone());
            hidden.nbt = Some(root);
            let lines = build_with_map_data(&hidden, true, Some(&data));
            assert!(
                !lines
                    .iter()
                    .any(|line| line.starts_with("Scaling at") || line.starts_with("(Level")),
                "numeric HideFlags bit 32 must hide ItemMap information: {hide_flags:?}"
            );
        }

        let mut textual = map.clone();
        let mut root = NbtCompound::new();
        root.insert("HideFlags", Nbt::String("32".to_owned()));
        textual.nbt = Some(root);
        let lines = build_with_map_data(&textual, true, Some(&data));
        assert_eq!(&lines[1..3], ["Scaling at 1:4", "(Level 2/4)"]);
    }

    /// Every `ItemRecord` has the shared `Music Disc` title plus the subclass-specific
    /// `getRecordNameLocal()` line. The ids matter: the 12 names cannot be inferred from the
    /// shared `item.record` unlocalised key. The raw line is uncoloured; the render seam adds
    /// continuation gray, just as `GuiScreen.renderToolTip` does.
    #[test]
    fn music_disc_tooltips_use_every_exact_record_description() {
        let expected = [
            ("record_13", "C418 - 13"),
            ("record_cat", "C418 - cat"),
            ("record_blocks", "C418 - blocks"),
            ("record_chirp", "C418 - chirp"),
            ("record_far", "C418 - far"),
            ("record_mall", "C418 - mall"),
            ("record_mellohi", "C418 - mellohi"),
            ("record_stal", "C418 - stal"),
            ("record_strad", "C418 - strad"),
            ("record_ward", "C418 - ward"),
            ("record_11", "C418 - 11"),
            ("record_wait", "C418 - wait"),
        ];
        for (name, description) in expected {
            let disc = stack(name);
            let lines = build(&disc, false);
            assert_eq!(
                lines,
                [format!("Music Disc{RESET}"), description.to_owned()]
            );
            let rendered = crate::screens::container::tooltip_box::recolour_tooltip_lines(
                lines,
                rarity(&disc),
            );
            assert_eq!(rendered[1], format!("{GRAY}{description}"));
            assert_eq!(rarity(&disc), Rarity::Rare, "{name} is an ItemRecord");
        }

        let apple = build(&stack("apple"), false);
        assert_eq!(apple.len(), 1, "ordinary items do not acquire disc info");
    }

    #[test]
    fn music_disc_description_uses_the_active_language_desc_key() {
        let mut language = Language::new("en_US");
        language.merge_lang_str(
            "item.record.name=Disque de musique\nitem.record.wait.desc=Attendre par C418\n",
        );
        let lines = build_localized(&stack("record_wait"), false, &language);
        assert_eq!(
            lines,
            [
                format!("Disque de musique{RESET}"),
                "Attendre par C418".to_owned()
            ]
        );
        let rendered = crate::screens::container::tooltip_box::recolour_tooltip_lines(
            lines,
            rarity(&stack("record_wait")),
        );
        assert_eq!(rendered[1], format!("{GRAY}Attendre par C418"));
    }

    #[test]
    fn display_name_core_honors_metadata_book_titles_and_custom_name_precedence() {
        let mut language = Language::new("xx_TEST");
        language.insert("item.writtenBook.name", "Localized Written Book");
        language.insert("tile.cloth.red.name", "Localized Red Wool");

        // The ordinary branch is `getUnlocalizedName(stack) + ".name"`, not the one-name-per-id
        // extractor field.  This is deliberately a subtype whose static `display_name` is just
        // "Wool", so changing the resolver back to that field fails this oracle.
        let mut wool = stack("wool");
        wool.damage = 14;
        assert_eq!(
            display_name_localized(&wool, &language),
            ("Localized Red Wool".into(), false)
        );

        let mut book = stack("written_book");
        let mut root = NbtCompound::new();
        root.insert("title", Nbt::String("A Very Real Book".into()));
        book.nbt = Some(root);
        assert_eq!(
            display_name_localized(&book, &language),
            ("A Very Real Book".into(), false),
            "ItemWrittenBook's nonempty root title is an item display-name override"
        );
        assert_eq!(
            build_localized(&book, false, &language)[0],
            format!("A Very Real Book{RESET}"),
            "tooltip line zero adds its own reset; getDisplayName itself does not"
        );
        assert_eq!(build(&book, false)[0], format!("A Very Real Book{RESET}"));

        book.nbt
            .as_mut()
            .unwrap()
            .insert("title", Nbt::String(String::new()));
        assert_eq!(
            display_name_localized(&book, &language),
            ("Localized Written Book".into(), false),
            "an empty title falls through to Item.getItemStackDisplayName"
        );

        let mut display = NbtCompound::new();
        display.insert("Name", Nbt::String("Server Override".into()));
        book.nbt
            .as_mut()
            .unwrap()
            .insert("title", Nbt::String("Book Title Loses".into()));
        book.nbt
            .as_mut()
            .unwrap()
            .insert("display", Nbt::Compound(display));
        assert_eq!(
            display_name_localized(&book, &language),
            ("Server Override".into(), true),
            "ItemStack.display.Name runs after ItemWrittenBook and therefore wins"
        );
    }

    /// Oracle: `ItemMonsterPlacer.getItemStackDisplayName` translates and trims the base, then
    /// appends `entity.<EntityList save name>.name` whenever the metadata is any registered
    /// entity id. The lookup deliberately does not consult `EntityList.entityEggs`.
    #[test]
    fn spawn_egg_name_uses_entity_list_and_active_language() {
        let mut language = Language::new("xx_TEST");
        language.insert("item.monsterPlacer.name", "  Summon  ");
        language.insert("entity.Creeper.name", "Hisser");
        language.insert("entity.PigZombie.name", "Undead Pig");
        language.insert("entity.EnderDragon.name", "Dragon");

        let named = |damage| {
            let mut egg = stack("spawn_egg");
            egg.damage = damage;
            display_name_localized(&egg, &language).0
        };
        assert_eq!(named(50), "Summon Hisser");
        assert_eq!(named(57), "Summon Undead Pig");

        let dragon = mc_data::entity_type(63).expect("registered EnderDragon");
        assert!(
            dragon.info().egg_primary.is_none(),
            "this vector proves the name lookup is EntityList, not the egg-colour table"
        );
        assert_eq!(named(63), "Summon Dragon");

        for invalid in [-1, 3, 256, 306, i16::MAX] {
            assert_eq!(named(invalid), "Summon", "invalid metadata {invalid}");
        }
    }

    #[test]
    fn spawn_egg_missing_entity_translation_keeps_the_raw_java_language_key() {
        let mut language = Language::new("xx_TEST");
        language.insert("item.monsterPlacer.name", "Spawn");
        let mut egg = stack("spawn_egg");
        egg.damage = 54;
        assert_eq!(
            display_name_localized(&egg, &language).0,
            "Spawn entity.Zombie.name"
        );

        let mut display = NbtCompound::new();
        display.insert("Name", Nbt::String("Server Pet".into()));
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        egg.nbt = Some(root);
        assert_eq!(
            display_name_localized(&egg, &language),
            ("Server Pet".into(), true),
            "ItemStack display.Name still overrides the item subclass name"
        );
    }

    #[test]
    fn firework_star_information_matches_java_order_colors_and_empty_compound_fallback() {
        let star = stack("firework_charge");
        assert!(
            item_information_lines(&star, None).is_empty(),
            "ItemFireworkCharge returns before getCompoundTag when the root tag is absent"
        );

        let mut empty_root_star = star.clone();
        empty_root_star.nbt = Some(NbtCompound::new());
        assert_eq!(
            item_information_lines(&empty_root_star, None),
            ["Small Ball"],
            "Java getCompoundTag returns an empty compound; missing Type reads byte zero"
        );

        let mut explosion = NbtCompound::new();
        explosion.insert("Type", Nbt::Byte(3));
        explosion.insert(
            "Colors",
            Nbt::IntArray(vec![11_743_532, 0x12_3456, 1_973_019]),
        );
        explosion.insert("FadeColors", Nbt::IntArray(vec![15_790_320, 6_719_955]));
        explosion.insert("Trail", Nbt::Int(1));
        explosion.insert("Flicker", Nbt::Byte(1));
        let mut root = NbtCompound::new();
        root.insert("Explosion", Nbt::Compound(explosion));
        let mut decorated = star;
        decorated.nbt = Some(root);
        assert_eq!(
            item_information_lines(&decorated, None),
            [
                "Creeper-shaped",
                "Red, Custom, Black",
                "Fade to White, Light Blue",
                "Trail",
                "Twinkle",
            ]
        );
    }

    #[test]
    fn firework_rocket_information_preserves_explosion_boundaries_and_indentation() {
        let mut first = NbtCompound::new();
        first.insert("Type", Nbt::Byte(1));
        first.insert("Colors", Nbt::IntArray(vec![2_437_522]));
        first.insert("Trail", Nbt::Byte(1));
        let mut second = NbtCompound::new();
        second.insert("Type", Nbt::Byte(-1));
        second.insert("Flicker", Nbt::Byte(1));

        let mut fireworks = NbtCompound::new();
        fireworks.insert("Flight", Nbt::Int(257));
        fireworks.insert(
            "Explosions",
            Nbt::list(vec![Nbt::Compound(first), Nbt::Compound(second)]),
        );
        let mut root = NbtCompound::new();
        root.insert("Fireworks", Nbt::Compound(fireworks));
        let mut rocket = stack("fireworks");
        rocket.nbt = Some(root);

        assert_eq!(
            item_information_lines(&rocket, None),
            [
                "Flight Duration: 1",
                "Large Ball",
                "  Blue",
                "  Trail",
                "Unknown Shape",
                "  Twinkle",
            ]
        );
    }

    #[test]
    fn firework_information_localizes_every_component_and_obeys_hide_additional_flags() {
        let mut language = Language::new("xx_TEST");
        for (key, value) in [
            ("item.fireworksCharge.type.0", "  Orb  "),
            ("item.fireworksCharge.red", "Rouge"),
            ("item.fireworksCharge.fadeTo", "Into"),
            ("item.fireworksCharge.black", "Noir"),
            ("item.fireworksCharge.trail", "Tail"),
            ("item.fireworksCharge.flicker", "Sparkle"),
        ] {
            language.insert(key, value);
        }
        let mut explosion = NbtCompound::new();
        explosion.insert("Colors", Nbt::IntArray(vec![11_743_532]));
        explosion.insert("FadeColors", Nbt::IntArray(vec![1_973_019]));
        explosion.insert("Trail", Nbt::Byte(1));
        explosion.insert("Flicker", Nbt::Byte(1));
        let mut root = NbtCompound::new();
        root.insert("Explosion", Nbt::Compound(explosion));
        let mut star = stack("firework_charge");
        star.nbt = Some(root);
        assert_eq!(
            item_information_lines(&star, Some(&language)),
            ["Orb", "Rouge", "Into Noir", "Tail", "Sparkle"]
        );

        let mut trim_language = Language::new("trim_TEST");
        trim_language.insert("item.fireworksCharge.type.0", " \u{a0}Orb\u{a0} ");
        assert_eq!(
            firework_explosion_information_lines(None, Some(&trim_language)),
            ["\u{a0}Orb\u{a0}"],
            "Java String.trim removes ASCII spaces but preserves non-breaking spaces"
        );

        let visible = build_localized(&star, false, &language);
        assert_eq!(
            &visible[1..],
            ["Orb", "Rouge", "Into Noir", "Tail", "Sparkle"]
        );
        star.nbt.as_mut().unwrap().insert("HideFlags", Nbt::Int(32));
        assert_eq!(
            build_localized(&star, false, &language).len(),
            1,
            "HideFlags bit 5 gates Item.addInformation as one block"
        );
    }

    #[test]
    fn all_sixteen_exact_dye_rgb_values_are_named_and_nearby_values_are_custom() {
        let expected = [
            (1_973_019, "Black"),
            (11_743_532, "Red"),
            (3_887_386, "Green"),
            (5_320_730, "Brown"),
            (2_437_522, "Blue"),
            (8_073_150, "Purple"),
            (2_651_799, "Cyan"),
            (11_250_603, "Light Gray"),
            (4_408_131, "Gray"),
            (14_188_952, "Pink"),
            (4_312_372, "Lime"),
            (14_602_026, "Yellow"),
            (6_719_955, "Light Blue"),
            (12_801_229, "Magenta"),
            (15_435_844, "Orange"),
            (15_790_320, "White"),
        ];
        for (rgb, name) in expected {
            assert_eq!(firework_color_name(rgb, None), name, "RGB {rgb:#08x}");
            assert_eq!(
                firework_color_name(rgb ^ 1, None),
                "Custom",
                "matching is exact, not nearest-colour"
            );
        }
    }

    #[test]
    fn firework_star_overlay_color_matches_single_average_and_default_java_vectors() {
        let with_colors = |colors: Option<Nbt>| {
            let mut explosion = NbtCompound::new();
            if let Some(colors) = colors {
                explosion.insert("Colors", colors);
            }
            let mut root = NbtCompound::new();
            root.insert("Explosion", Nbt::Compound(explosion));
            let mut star = stack("firework_charge");
            star.nbt = Some(root);
            star
        };

        assert_eq!(firework_star_color(&stack("apple")), None);
        assert_eq!(
            firework_star_color(&stack("firework_charge")),
            Some(0x8A_8A_8A)
        );
        assert_eq!(
            firework_star_color(&with_colors(Some(Nbt::String("red".into())))),
            Some(0x8A_8A_8A),
            "only TAG_Int_Array is accepted"
        );
        assert_eq!(
            firework_star_color(&with_colors(Some(Nbt::IntArray(vec![0x12_3456])))),
            Some(0x12_3456),
            "a single color is returned verbatim"
        );
        assert_eq!(
            firework_star_color(&with_colors(Some(Nbt::IntArray(vec![
                0xFF_0000, 0x00_FF00, 0x00_00FF,
            ])))),
            Some(0x55_55_55),
            "three channels are summed and integer-divided independently"
        );
        assert_eq!(
            firework_star_color(&with_colors(Some(Nbt::IntArray(Vec::new())))),
            Some(0x8A_8A_8A),
            "safe extension: hostile empty arrays use neutral grey instead of Java's /0 crash"
        );
    }

    /// Oracle: real 1.8.9 `PotionHelper.getPotionEffects(meta, false)`, covering ordinary,
    /// extended, amplified, instant, splash, no-effect, and metadata-without-the-8192 base bit.
    #[test]
    fn potion_metadata_decoder_matches_java_oracle_vectors() {
        type PotionCase<'a> = (i16, &'a [(u8, i32, i32)]);
        let cases: &[PotionCase<'_>] = &[
            (0, &[]),
            (1, &[(10, 900, 0)]),
            (16, &[]),
            (8193, &[(10, 900, 0)]),
            (8194, &[(1, 3600, 0)]),
            (8195, &[(12, 3600, 0)]),
            (8196, &[(19, 900, 0)]),
            (8197, &[(6, 1, 0)]),
            (8200, &[(18, 1800, 0)]),
            (8201, &[(5, 3600, 0)]),
            (8225, &[(10, 450, 1)]),
            (8257, &[(10, 2400, 0)]),
            (16386, &[(1, 2701, 0)]),
            (16420, &[(19, 338, 1)]),
            (16424, &[(18, 1351, 0)]),
        ];
        for (meta, expected) in cases {
            let actual: Vec<_> = potion_effects(&potion_stack(*meta))
                .expect("potion")
                .into_iter()
                .map(|effect| (effect.id, effect.duration, effect.amplifier))
                .collect();
            assert_eq!(&actual, expected, "metadata {meta}");
        }
        assert!(potion_effects(&stack("apple")).is_none());
    }

    /// Oracle: direct Java 1.8.9 `ItemPotion.getColorFromItemStack(stack, 0)` and `hasEffect`
    /// probes after `Bootstrap.register()`.
    #[test]
    fn potion_liquid_color_and_effect_glint_policy_match_java_oracle_vectors() {
        let cases = [
            (0, 0x38_5D_C6, false),
            (16, 0x38_5D_C6, false),
            (8193, 0xCD_5C_AB, true),
            (8194, 0x7C_AF_C6, true),
            (8196, 0x4E_93_31, true),
            (8197, 0xF8_24_23, true),
            (8200, 0x48_4D_48, true),
            (8201, 0x93_24_23, true),
            (8225, 0xCD_5C_AB, true),
            (16386, 0x7C_AF_C6, true),
            (16420, 0x4E_93_31, true),
        ];
        for (meta, color, glint) in cases {
            let stack = potion_stack(meta);
            assert_eq!(potion_liquid_color(&stack), Some(color), "metadata {meta}");
            assert_eq!(potion_has_effect(&stack), glint, "metadata {meta}");
        }
        assert_eq!(potion_liquid_color(&stack("apple")), None);
        assert!(!potion_has_effect(&stack("apple")));
    }

    #[test]
    fn custom_effects_change_potion_glint_but_java_tint_still_comes_from_metadata() {
        let custom_on_empty = custom_potion(16, 1, 0, 1200);
        assert!(potion_has_effect(&custom_on_empty));
        assert_eq!(potion_liquid_color(&custom_on_empty), Some(0x38_5D_C6));

        let empty_on_speed = empty_custom_potion(8194);
        assert!(!potion_has_effect(&empty_on_speed));
        assert_eq!(potion_liquid_color(&empty_on_speed), Some(0x7C_AF_C6));
    }

    /// Negative controls for the precise 1.8.9 `ItemPotion.getColorFromItemStack` route:
    /// `getColorFromDamage(stack.getMetadata())` reaches `PotionHelper.getLiquidColor`, not the
    /// stack's custom-effect list. `calcPotionLiquidColor` does filter hidden particles, but only
    /// when it is given such effects (for example, living-entity particles), which this item path
    /// never does. `CustomPotionColor` is a later-version tag and absent from 1.8.9's path.
    #[test]
    fn potion_item_tint_ignores_custom_effect_visibility_and_custom_color_nbt() {
        let mut hidden_custom_effect = NbtCompound::new();
        hidden_custom_effect.insert("Id", Nbt::Byte(1));
        hidden_custom_effect.insert("Amplifier", Nbt::Byte(0));
        hidden_custom_effect.insert("Duration", Nbt::Int(1_200));
        hidden_custom_effect.insert("ShowParticles", Nbt::Byte(0));

        let mut root = NbtCompound::new();
        root.insert(
            "CustomPotionEffects",
            Nbt::list(vec![Nbt::Compound(hidden_custom_effect)]),
        );
        root.insert("CustomPotionColor", Nbt::Int(0x12_3456));
        let mut potion = potion_stack(8_194); // Speed: Java's metadata tint is 0x7C_AF_C6.
        potion.nbt = Some(root);

        assert!(
            potion_has_effect(&potion),
            "the custom list remains reachable elsewhere"
        );
        assert!(
            !potion_effects(&potion).unwrap()[0].show_particles,
            "negative control: this would affect calcPotionLiquidColor if the item passed custom effects"
        );
        assert_eq!(
            potion_liquid_color(&potion),
            Some(0x7C_AF_C6),
            "only stack damage reaches ItemPotion's liquid-colour calculation"
        );
    }

    /// Oracle: direct Java 1.8.9 `Items.potionitem.getItemStackDisplayName` probes after
    /// `Bootstrap.register()`. This spans water, raw prefix, effect postfix, extended/amplified,
    /// splash-effect, effect-less splash, and all-five-prefix-bits paths.
    #[test]
    fn potion_display_names_match_java_oracle_vectors() {
        let cases = [
            (0, "Water Bottle"),
            (1, "Potion of Regeneration"),
            (16, "Awkward Potion"),
            (32, "Thick Potion"),
            (64, "Mundane Potion"),
            (8193, "Potion of Regeneration"),
            (8194, "Potion of Swiftness"),
            (8195, "Potion of Fire Resistance"),
            (8196, "Potion of Poison"),
            (8197, "Potion of Healing"),
            (8200, "Potion of Weakness"),
            (8201, "Potion of Strength"),
            (8225, "Potion of Regeneration"),
            (8257, "Potion of Regeneration"),
            (16384, "Mundane Potion"),
            (16385, "Splash Potion of Regeneration"),
            (16386, "Splash Potion of Swiftness"),
            (16420, "Splash Potion of Poison"),
            (16424, "Splash Potion of Weakness"),
            (32767, "Stinky Potion"),
        ];
        for (meta, expected) in cases {
            assert_eq!(
                build(&potion_stack(meta), false)[0],
                format!("{expected}{RESET}"),
                "metadata {meta}"
            );
        }
    }

    #[test]
    fn potion_display_name_uses_custom_effect_order_empty_override_and_custom_stack_name() {
        let ordered = custom_potion_sequence(16, &[(21, 0, 1200), (8, 0, 1200)]);
        assert_eq!(
            build(&ordered, false)[0],
            format!("Potion of Health Boost{RESET}"),
            "ItemPotion names a custom-effect stack from list.get(0), not the last effect"
        );
        assert_eq!(
            build(&custom_potion(16384, 8, 0, 1200), false)[0],
            format!("Splash Potion of Leaping{RESET}")
        );
        assert_eq!(
            build(&empty_custom_potion(8194), false)[0],
            format!("Uninteresting Potion{RESET}")
        );
        assert_eq!(
            build(&empty_custom_potion(16384), false)[0],
            format!("Mundane Potion{RESET}"),
            "Java does not apply the splash word in its no-effect fallback branch"
        );

        let mut named = potion_stack(8194);
        let mut display = NbtCompound::new();
        display.insert("Name", Nbt::String("Custom Brew".to_owned()));
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        named.nbt = Some(root);
        assert_eq!(
            build(&named, false)[0],
            format!("{ITALIC}Custom Brew{RESET}")
        );
    }

    #[test]
    fn potion_display_name_uses_the_active_language_and_trims_each_java_component() {
        let mut language = Language::new("xx_TEST");
        language.merge_lang_str(
            "item.emptyPotion.name=  Flacon  \n\
             item.potion.name=  Breuvage  \n\
             potion.moveSpeed.postfix=  Elixir Rapide  \n\
             potion.prefix.grenade=  Jet  \n\
             potion.prefix.awkward=  Maladroit  \n\
             potion.prefix.thick=  Epais  \n\
             potion.prefix.mundane=  Ordinaire  \n\
             potion.prefix.uninteresting=  Curieux  \n",
        );
        assert_eq!(
            display_name_localized(&potion_stack(0), &language),
            ("Flacon".to_owned(), false)
        );
        assert_eq!(
            display_name_localized(&potion_stack(16386), &language),
            ("Jet Elixir Rapide".to_owned(), false)
        );
        // These no-effect values each select a different bit-ordered PotionHelper prefix.  They
        // all append `item.potion.name`, whereas metadata zero is the one and only branch that
        // selects `item.emptyPotion.name` above.
        for (meta, expected) in [
            (16, "Maladroit Breuvage"),
            (32, "Epais Breuvage"),
            (64, "Ordinaire Breuvage"),
        ] {
            assert_eq!(
                display_name_localized(&potion_stack(meta), &language),
                (expected.to_owned(), false),
                "metadata {meta}"
            );
        }
        assert_eq!(
            display_name_localized(&empty_custom_potion(8194), &language),
            ("Curieux Breuvage".to_owned(), false)
        );
    }

    #[test]
    fn standard_potion_tooltips_match_java_effect_and_attribute_lines() {
        let cases: &[(i16, &[&str])] = &[
            (0, &[]),
            (16, &["§7No Effects"]),
            (8193, &["§7Regeneration (0:45)"]),
            (
                8194,
                &["§7Speed (3:00)", "", "§5When Applied:", "§9+20% Speed"],
            ),
            (8197, &["§7Instant Health"]),
            (
                8200,
                &[
                    "§cWeakness (1:30)",
                    "",
                    "§5When Applied:",
                    "§c-0.5 Attack Damage",
                ],
            ),
            (
                8201,
                &[
                    "§7Strength (3:00)",
                    "",
                    "§5When Applied:",
                    "§9+130% Attack Damage",
                ],
            ),
            (8225, &["§7Regeneration II (0:22)"]),
            (8257, &["§7Regeneration (2:00)"]),
            (
                16386,
                &["§7Speed (2:15)", "", "§5When Applied:", "§9+20% Speed"],
            ),
            (16420, &["§cPoison II (0:16)"]),
        ];
        for (meta, expected) in cases {
            let lines = build(&potion_stack(*meta), false);
            assert_eq!(&lines[1..], *expected, "metadata {meta}: {lines:?}");
        }
    }

    #[test]
    fn custom_potion_effects_override_metadata_and_preserve_nbt_order_and_defaults() {
        let potion = custom_potion(8194, 2, 1, 1220);
        let effects = potion_effects(&potion).unwrap();
        assert_eq!(effects.len(), 1, "custom list replaces metadata Speed");
        assert_eq!(effects[0].id, 2);
        assert!(!effects[0].ambient);
        assert!(
            effects[0].show_particles,
            "missing ShowParticles defaults true"
        );
        assert_eq!(
            &build(&potion, false)[1..],
            [
                "§cSlowness II (1:01)",
                "",
                "§5When Applied:",
                "§c-30% Speed"
            ]
        );

        let mut empty_override = potion_stack(8194);
        let mut root = NbtCompound::new();
        root.insert("CustomPotionEffects", Nbt::list(vec![]));
        empty_override.nbt = Some(root);
        assert!(potion_effects(&empty_override).unwrap().is_empty());
        assert_eq!(&build(&empty_override, false)[1..], ["§7No Effects"]);

        let zero_modifier = custom_potion(8194, 1, -1, 1200);
        assert_eq!(
            &build(&zero_modifier, false)[1..],
            ["§7Speed (1:00)"],
            "Java suppresses a modifier whose (amplifier + 1) scale is zero"
        );
    }

    #[test]
    fn custom_potion_show_particles_requires_java_byte_tag() {
        // `PotionEffect.readCustomPotionEffectFromNBT` only reads ShowParticles when
        // `hasKey("ShowParticles", 1)` succeeds.  The shared NBT numeric getter is intentionally
        // broader (matching getBoolean), so this test protects the ItemPotion-specific type gate.
        let stack_with = |value| {
            let mut effect = NbtCompound::new();
            effect.insert("Id", Nbt::Byte(1));
            effect.insert("Duration", Nbt::Int(1_200));
            effect.insert("ShowParticles", value);
            let mut root = NbtCompound::new();
            root.insert(
                "CustomPotionEffects",
                Nbt::list(vec![Nbt::Compound(effect)]),
            );
            let mut stack = potion_stack(16);
            stack.nbt = Some(root);
            stack
        };

        assert!(potion_effects(&stack_with(Nbt::Byte(1))).unwrap()[0].show_particles);
        assert!(!potion_effects(&stack_with(Nbt::Byte(0))).unwrap()[0].show_particles);
        assert!(
            potion_effects(&stack_with(Nbt::Int(0))).unwrap()[0].show_particles,
            "wrong-typed numeric tags default true in Java"
        );
        assert!(
            potion_effects(&stack_with(Nbt::String("false".to_owned()))).unwrap()[0].show_particles,
            "wrong-typed string tags default true in Java"
        );
    }

    #[test]
    fn potion_lines_use_the_active_language_for_names_headers_and_attributes() {
        let mut language = Language::new("xx_TEST");
        language.merge_lang_str(
            "item.potion.name=Potion\n\
             potion.moveSpeed=Vitesse\n\
             potion.effects.whenDrank=Applique :\n\
             attribute.name.generic.movementSpeed=Rapidite\n\
             attribute.modifier.plus.2=plus %s%% %s\n",
        );
        language.insert("potion.effects.whenDrank", "  Applique :  ");
        language.insert("attribute.name.generic.movementSpeed", "  Rapidite  ");
        let lines = build_localized(&potion_stack(8194), false, &language);
        assert_eq!(
            &lines[1..],
            [
                "§7Vitesse (3:00)",
                "",
                "§5  Applique :  ",
                "§9plus 20%   Rapidite  "
            ]
        );
    }

    #[test]
    fn potion_information_obeys_hide_flags_and_precedes_enchantments() {
        let mut potion = custom_potion(8194, 1, 0, 1200);
        let root = potion.nbt.as_mut().unwrap();
        let mut enchantment = NbtCompound::new();
        enchantment.insert("id", Nbt::Short(16));
        enchantment.insert("lvl", Nbt::Short(1));
        root.insert("ench", Nbt::list(vec![Nbt::Compound(enchantment)]));

        let lines = build(&potion, false);
        let effect = lines
            .iter()
            .position(|line| line.contains("Speed (1:00)"))
            .unwrap();
        let enchantment = lines
            .iter()
            .position(|line| line.contains("Sharpness I"))
            .unwrap();
        assert!(
            effect < enchantment,
            "Item.addInformation runs first: {lines:?}"
        );

        potion
            .nbt
            .as_mut()
            .unwrap()
            .insert("HideFlags", Nbt::Int(32));
        let hidden = build(&potion, false);
        assert!(!hidden.iter().any(|line| line.contains("Speed")));
        assert!(hidden.iter().any(|line| line.contains("Sharpness I")));
    }

    #[test]
    fn localized_tooltip_uses_the_metadata_specific_unlocalized_name() {
        let mut s = stack("sapling");
        s.damage = 5;
        let mut language = Language::new("en_US");
        language.merge_lang_str(
            "tile.sapling.name=Sapling\ntile.sapling.big_oak.name=Dark Oak Sapling\n",
        );
        let lines = build_localized(&s, false, &language);
        assert!(lines[0].contains("Dark Oak Sapling"), "{:?}", lines[0]);
        assert!(!lines[0].contains("tile.sapling"));
    }

    #[test]
    fn coal_tooltip_distinguishes_charcoal_and_falls_back_to_coal() {
        let mut language = Language::new("xx_TEST");
        language.insert("item.coal.name", "translated coal");
        language.insert("item.charcoal.name", "translated charcoal");

        let mut coal = stack("coal");
        assert_eq!(build(&coal, false), [format!("Coal{RESET}")]);
        assert_eq!(
            build_localized(&coal, false, &language),
            [format!("translated coal{RESET}")]
        );

        coal.damage = 1;
        assert_eq!(build(&coal, false), [format!("Charcoal{RESET}")]);
        assert_eq!(
            build_localized(&coal, false, &language),
            [format!("translated charcoal{RESET}")]
        );

        coal.damage = i16::MAX;
        assert_eq!(build(&coal, false), [format!("Coal{RESET}")]);
        assert_eq!(
            build_localized(&coal, false, &language),
            [format!("translated coal{RESET}")]
        );
    }

    #[test]
    fn metadata_identity_tooltips_cover_prismarine_red_flower_and_red_sandstone() {
        let cases: &[(&str, &[(&str, &str)])] = &[
            (
                "prismarine",
                &[
                    ("tile.prismarine.rough", "Prismarine"),
                    ("tile.prismarine.bricks", "Prismarine Bricks"),
                    ("tile.prismarine.dark", "Dark Prismarine"),
                ],
            ),
            (
                "red_flower",
                &[
                    ("tile.flower2.poppy", "Poppy"),
                    ("tile.flower2.blueOrchid", "Blue Orchid"),
                    ("tile.flower2.allium", "Allium"),
                    ("tile.flower2.houstonia", "Azure Bluet"),
                    ("tile.flower2.tulipRed", "Red Tulip"),
                    ("tile.flower2.tulipOrange", "Orange Tulip"),
                    ("tile.flower2.tulipWhite", "White Tulip"),
                    ("tile.flower2.tulipPink", "Pink Tulip"),
                    ("tile.flower2.oxeyeDaisy", "Oxeye Daisy"),
                ],
            ),
            (
                "red_sandstone",
                &[
                    ("tile.redSandStone.default", "Red Sandstone"),
                    ("tile.redSandStone.chiseled", "Chiseled Red Sandstone"),
                    ("tile.redSandStone.smooth", "Smooth Red Sandstone"),
                ],
            ),
        ];
        let mut language = Language::new("xx_TEST");
        for &(item, variants) in cases {
            for (damage, &(key, en_us)) in variants.iter().enumerate() {
                language.insert(format!("{key}.name"), format!("translated {item} {damage}"));
                let mut s = stack(item);
                s.damage = damage as i16;
                assert_eq!(
                    build(&s, false),
                    [format!("{en_us}{RESET}")],
                    "metadata {damage} must use the vanilla display name for {item}"
                );
                assert_eq!(
                    build_localized(&s, false, &language),
                    [format!("translated {item} {damage}{RESET}")],
                    "metadata {damage} must select {key} for {item}"
                );
            }

            let mut invalid = stack(item);
            invalid.damage = i16::MAX;
            let (fallback_key, fallback_name) = variants[0];
            assert_eq!(build(&invalid, false), [format!("{fallback_name}{RESET}")]);
            assert_eq!(
                build_localized(&invalid, false, &language),
                [format!("translated {item} 0{RESET}")],
                "invalid metadata must select the metadata-zero key {fallback_key}"
            );
        }
    }

    #[test]
    fn cloth_family_tooltips_use_metadata_names_and_white_enum_fallbacks() {
        let cases = [
            (
                "wool",
                "tile.cloth.red",
                "tile.cloth.white",
                "Red Wool",
                "Wool",
            ),
            (
                "stained_glass",
                "tile.stainedGlass.red",
                "tile.stainedGlass.white",
                "Red Stained Glass",
                "White Stained Glass",
            ),
            (
                "stained_hardened_clay",
                "tile.clayHardenedStained.red",
                "tile.clayHardenedStained.white",
                "Red Stained Clay",
                "White Stained Clay",
            ),
            (
                "stained_glass_pane",
                "tile.thinStainedGlass.red",
                "tile.thinStainedGlass.white",
                "Red Stained Glass Pane",
                "White Stained Glass Pane",
            ),
            (
                "carpet",
                "tile.woolCarpet.red",
                "tile.woolCarpet.white",
                "Red Carpet",
                "Carpet",
            ),
        ];
        let mut language = Language::new("xx_TEST");

        for &(item, red_key, white_key, red_en_us, white_en_us) in &cases {
            language.insert(format!("{red_key}.name"), format!("translated {item}"));
            language.insert(format!("{white_key}.name"), format!("fallback {item}"));

            let id = mc_data::item_by_name(item).expect("cloth-family item");
            let mut red = stack(item);
            red.damage = 14;
            assert_eq!(
                build(&red, false),
                [format!("{red_en_us}{RESET}")],
                "no-language fallback must resolve the metadata-specific en_US name for {item}"
            );
            assert_eq!(
                build_localized(&red, false, &language),
                [format!("translated {item}{RESET}")],
                "localized tooltip must use the metadata-specific key for {item}"
            );

            let mut invalid = ItemStack {
                item_id: id.0 as i16,
                count: 1,
                damage: i16::MAX,
                nbt: None,
            };
            assert_eq!(
                build(&invalid, false),
                [format!("{white_en_us}{RESET}")],
                "no-language invalid metadata must follow EnumDyeColor.byMetadata for {item}"
            );
            assert_eq!(
                build_localized(&invalid, false, &language),
                [format!("fallback {item}{RESET}")],
                "localized invalid metadata must use the white variant key for {item}"
            );
            invalid.damage = -1;
            assert_eq!(
                build_localized(&invalid, false, &language),
                [format!("fallback {item}{RESET}")],
                "negative metadata must take the same white fallback for {item}"
            );
        }
    }

    #[test]
    fn localized_player_head_formats_the_skull_owner_name() {
        let mut s = stack("skull");
        s.damage = 3;
        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::String("Dinnerbone".to_owned()));
        s.nbt = Some(root);
        let mut language = Language::new("en_US");
        language.merge_lang_str("item.skull.player.name=%s's Head\nitem.skull.char.name=Head\n");
        let lines = build_localized(&s, false, &language);
        assert!(lines[0].contains("Dinnerbone's Head"), "{:?}", lines[0]);
    }

    #[test]
    fn skull_tooltips_cover_all_metadata_names_and_java_owner_type_gates() {
        let mut language = Language::new("xx_TEST");
        language.merge_lang_str(
            "item.skull.skeleton.name=Skeleton Variant\n\
             item.skull.wither.name=Wither Variant\n\
             item.skull.zombie.name=Zombie Variant\n\
             item.skull.char.name=Player Variant\n\
             item.skull.creeper.name=Creeper Variant\n\
             item.skull.player.name=%s's Head\n",
        );
        let variants = [
            (0, "Skeleton Skull", "Skeleton Variant"),
            (1, "Wither Skeleton Skull", "Wither Variant"),
            (2, "Zombie Head", "Zombie Variant"),
            (3, "Head", "Player Variant"),
            (4, "Creeper Head", "Creeper Variant"),
        ];

        for (damage, en_us, localized) in variants {
            let mut skull = stack("skull");
            skull.damage = damage;
            assert_eq!(build(&skull, false), [format!("{en_us}{RESET}")]);
            assert_eq!(
                build_localized(&skull, false, &language),
                [format!("{localized}{RESET}")],
                "metadata {damage} must select ItemSkull's key"
            );
        }

        // ItemSkull only enters its owner branch for metadata exactly 3.  The invalid values
        // below must use EnumDyeColor-style metadata-zero fallback rather than leaking the owner
        // into a skeleton/creeper name.
        for damage in [-1, 5, i16::MAX] {
            let mut skull = stack("skull");
            skull.damage = damage;
            let mut root = NbtCompound::new();
            root.insert("SkullOwner", Nbt::String("Dinnerbone".to_owned()));
            skull.nbt = Some(root);
            assert_eq!(build(&skull, false), [format!("Skeleton Skull{RESET}")]);
            assert_eq!(
                build_localized(&skull, false, &language),
                [format!("Skeleton Variant{RESET}")],
                "invalid metadata {damage} must not activate SkullOwner"
            );
        }

        let mut player = stack("skull");
        player.damage = 3;
        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::String("Dinnerbone".to_owned()));
        player.nbt = Some(root);
        assert_eq!(build(&player, false), [format!("Dinnerbone's Head{RESET}")]);
        assert_eq!(
            build_localized(&player, false, &language),
            [format!("Dinnerbone's Head{RESET}")]
        );

        // `hasKey("SkullOwner", 10)` is the second Java branch.  Empty strings remain valid
        // typed values, while wrong-typed Name values fall through to the ordinary char name.
        let mut owner_compound = NbtCompound::new();
        owner_compound.insert("Name", Nbt::String("Alex".to_owned()));
        player
            .nbt
            .as_mut()
            .unwrap()
            .insert("SkullOwner", Nbt::Compound(owner_compound));
        assert_eq!(build(&player, false), [format!("Alex's Head{RESET}")]);
        assert_eq!(
            display_name_localized(&player, &language),
            ("Alex's Head".to_owned(), false)
        );

        player
            .nbt
            .as_mut()
            .unwrap()
            .insert("SkullOwner", Nbt::String(String::new()));
        assert_eq!(build(&player, false), [format!("'s Head{RESET}")]);
        assert_eq!(
            display_name_localized(&player, &language),
            ("'s Head".to_owned(), false),
            "an empty TAG_String is present and must still be formatted"
        );

        let mut wrong_name = NbtCompound::new();
        wrong_name.insert("Name", Nbt::Int(1));
        player
            .nbt
            .as_mut()
            .unwrap()
            .insert("SkullOwner", Nbt::Compound(wrong_name));
        assert_eq!(
            display_name_localized(&player, &language),
            ("Player Variant".to_owned(), false),
            "a non-string compound Name must fall through"
        );

        player
            .nbt
            .as_mut()
            .unwrap()
            .insert("SkullOwner", Nbt::Int(1));
        assert_eq!(
            display_name_localized(&player, &language),
            ("Player Variant".to_owned(), false),
            "a non-string/non-compound SkullOwner must fall through"
        );
    }

    #[test]
    fn custom_named_item_is_shown_italic() {
        let mut s = stack("apple");
        let mut display = NbtCompound::new();
        display.insert("Name", Nbt::String("My Rock".into()));
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        s.nbt = Some(root);
        let lines = build(&s, false);
        assert!(lines[0].starts_with(ITALIC), "{:?}", lines[0]);
        assert!(lines[0].contains("My Rock"));
    }

    #[test]
    fn advanced_tooltip_appends_the_registry_id() {
        let s = stack("stone");
        let advanced = build(&s, true);
        let plain = build(&s, false);
        assert!(advanced[0].len() > plain[0].len());
        assert!(advanced[0].contains('#'));
    }

    /// java: `Item.getRarity` is `isItemEnchanted() ? RARE : COMMON`, and there are exactly
    /// three overrides in 1.8.9.
    #[test]
    fn rarity_covers_the_default_and_all_three_overrides() {
        assert_eq!(rarity(&stack("stone")), Rarity::Common);
        assert_eq!(
            rarity(&stack("diamond_sword")),
            Rarity::Common,
            "unenchanted"
        );

        let mut enchanted = stack("diamond_sword");
        let mut ench = NbtCompound::new();
        ench.insert("id", Nbt::Short(16));
        ench.insert("lvl", Nbt::Short(1));
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(vec![Nbt::Compound(ench)]));
        enchanted.nbt = Some(root);
        assert_eq!(rarity(&enchanted), Rarity::Rare);

        // `ItemRecord.getRarity` ignores everything and returns RARE.
        assert_eq!(rarity(&stack("record_13")), Rarity::Rare);
        assert_eq!(rarity(&stack("record_wait")), Rarity::Rare);

        // `ItemAppleGold`: metadata 0 is RARE and metadata 1 is EPIC, so neither is COMMON.
        let mut apple = stack("golden_apple");
        assert_eq!(rarity(&apple), Rarity::Rare);
        apple.damage = 1;
        assert_eq!(rarity(&apple), Rarity::Epic);

        // `ItemEnchantedBook`: UNCOMMON only when `StoredEnchantments` is non-empty; an empty
        // list falls through to `super`, which is COMMON for an unenchanted stack.
        let mut book = stack("enchanted_book");
        assert_eq!(rarity(&book), Rarity::Common, "no StoredEnchantments");
        let mut root = NbtCompound::new();
        root.insert("StoredEnchantments", Nbt::list(vec![]));
        book.nbt = Some(root);
        assert_eq!(rarity(&book), Rarity::Common, "tagCount() > 0 is false");
        let mut stored = NbtCompound::new();
        stored.insert("id", Nbt::Short(16));
        stored.insert("lvl", Nbt::Short(1));
        let mut root = NbtCompound::new();
        root.insert("StoredEnchantments", Nbt::list(vec![Nbt::Compound(stored)]));
        book.nbt = Some(root);
        assert_eq!(rarity(&book), Rarity::Uncommon);
    }

    /// `isItemEnchanted` is `hasKey("ench", 9)`: present **and** a list, contents unread.
    #[test]
    fn an_empty_ench_list_still_counts_as_enchanted_but_a_non_list_does_not() {
        let mut s = stack("diamond_sword");
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(vec![]));
        s.nbt = Some(root);
        assert!(is_item_enchanted(&s));
        assert_eq!(rarity(&s), Rarity::Rare);

        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::Byte(1));
        s.nbt = Some(root);
        assert!(!is_item_enchanted(&s), "hasKey(_, 9) wants a TAG_List");
    }

    /// `Item.hasEffect` is not merely the `ench` NBT predicate: its 1.8.9 overrides include
    /// always-foiled books/bottles and metadata/effect-driven foils. The controls deliberately
    /// separate this from `getRarity` (an empty enchanted book is common but still glints).
    #[test]
    fn has_effect_covers_base_nbt_and_every_vanilla_override() {
        assert!(!has_effect(&stack("diamond_sword")));
        assert!(!has_effect(&stack("potion")));
        assert!(has_effect(&potion_stack(8194)), "Speed potion");
        assert!(has_effect(&stack("enchanted_book")), "always foiled");
        assert!(has_effect(&stack("experience_bottle")), "always foiled");

        let mut apple = stack("golden_apple");
        assert!(!has_effect(&apple), "ordinary golden apple");
        apple.damage = 1;
        assert!(has_effect(&apple), "enchanted golden apple");

        let mut enchanted = stack("diamond_sword");
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(vec![]));
        enchanted.nbt = Some(root);
        assert!(
            has_effect(&enchanted),
            "an empty TAG_List is still enchanted"
        );
    }

    /// The four `EnumRarity` colours, pinned to their `EnumChatFormatting` codes.
    #[test]
    fn the_rarity_colour_codes_are_vanillas() {
        assert_eq!(Rarity::Common.color_code(), "\u{a7}f");
        assert_eq!(Rarity::Uncommon.color_code(), "\u{a7}e");
        assert_eq!(Rarity::Rare.color_code(), "\u{a7}b");
        assert_eq!(Rarity::Epic.color_code(), "\u{a7}d");
    }

    #[test]
    fn enchantments_render_as_name_plus_roman_numeral() {
        let mut s = stack("diamond_sword");
        let mut ench = NbtCompound::new();
        ench.insert("id", Nbt::Short(16)); // Sharpness (enchantment.damage.all)
        ench.insert("lvl", Nbt::Short(3));
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(vec![Nbt::Compound(ench)]));
        s.nbt = Some(root);
        let lines = build(&s, false);
        let sharpness = mc_data::enchantment(16).unwrap();
        assert!(
            lines
                .iter()
                .any(|l| l.contains(sharpness.display_name) && l.contains("III")),
            "{lines:?}"
        );
    }

    #[test]
    fn enchantment_colour_is_added_once_by_the_tooltip_renderer() {
        let mut s = stack("diamond_sword");
        let mut enchantment = NbtCompound::new();
        enchantment.insert("id", Nbt::Short(16));
        enchantment.insert("lvl", Nbt::Short(3));
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(vec![Nbt::Compound(enchantment)]));
        s.nbt = Some(root);

        let raw = build(&s, false);
        let line = raw
            .iter()
            .find(|line| line.contains("Sharpness III"))
            .expect("enchantment line");
        assert_eq!(line, "Sharpness III", "ItemStack.getTooltip adds no gray");

        let coloured =
            crate::screens::container::tooltip_box::recolour_tooltip_lines(raw, rarity(&s));
        let line = coloured
            .iter()
            .find(|line| line.contains("Sharpness III"))
            .expect("recoloured enchantment line");
        assert!(line.starts_with(GRAY));
        assert!(
            !line.starts_with(&format!("{GRAY}{GRAY}")),
            "GuiScreen.renderToolTip supplies exactly one continuation colour: {line:?}"
        );
    }

    /// This test used to assert the opposite - that a max-level-1 enchantment drops its
    /// numeral. That is `Enchantment.getTranslatedName` in **1.9**; 1.8.9's is unconditional
    /// (`Enchantment.java:217-221`), so a Silk Touch pickaxe reads "Silk Touch I".
    #[test]
    fn a_max_level_one_enchantment_still_shows_its_numeral() {
        let mut s = stack("diamond_pickaxe");
        let mut ench = NbtCompound::new();
        ench.insert("id", Nbt::Short(33)); // Silk Touch, max level 1
        ench.insert("lvl", Nbt::Short(1));
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(vec![Nbt::Compound(ench)]));
        s.nbt = Some(root);
        let lines = build(&s, false);
        let silk_touch = mc_data::enchantment(33).unwrap();
        let line = lines
            .iter()
            .find(|l| l.contains(silk_touch.display_name))
            .unwrap();
        assert!(line.ends_with(" I"), "{line:?}");
    }

    /// An out-of-range `lvl` - which servers do send - renders as the untranslated lang key,
    /// end to end through `build`.
    #[test]
    fn an_out_of_range_enchantment_level_renders_as_its_lang_key() {
        let mut s = stack("diamond_sword");
        let mut ench = NbtCompound::new();
        ench.insert("id", Nbt::Short(16)); // Sharpness
        ench.insert("lvl", Nbt::Short(100));
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(vec![Nbt::Compound(ench)]));
        s.nbt = Some(root);
        let lines = build(&s, false);
        assert!(
            lines
                .iter()
                .any(|l| l.ends_with("Sharpness enchantment.level.100")),
            "{lines:?}"
        );
    }

    #[test]
    fn lore_lines_are_dark_purple_and_italic() {
        let mut s = stack("stone");
        let mut display = NbtCompound::new();
        display.insert(
            "Lore",
            Nbt::list(vec![Nbt::String("A humble rock.".into())]),
        );
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        s.nbt = Some(root);
        let lines = build(&s, false);
        assert!(lines
            .iter()
            .any(|l| l == &format!("{DARK_PURPLE}{ITALIC}A humble rock.")));
    }

    /// java: ItemStack.getTooltip:720-742 puts the generic display colour before Lore, and its
    /// `Integer.toHexString` branch exposes a negative server colour as an eight-digit hex word.
    #[test]
    fn display_color_precedes_lore_and_has_the_exact_advanced_hex_form() {
        let mut s = stack("leather_chestplate");
        let mut display = NbtCompound::new();
        display.insert("color", Nbt::Int(-1));
        display.insert("Lore", Nbt::list(vec![Nbt::String("dyed lore".into())]));
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        s.nbt = Some(root);

        let normal = build(&s, false);
        let dyed = normal
            .iter()
            .position(|line| line == &format!("{ITALIC}Dyed"));
        let lore = normal
            .iter()
            .position(|line| line == &format!("{DARK_PURPLE}{ITALIC}dyed lore"));
        assert!(
            dyed.is_some_and(|dyed| lore.is_some_and(|lore| dyed < lore)),
            "{normal:?}"
        );

        let advanced = build(&s, true);
        assert!(
            advanced.iter().any(|line| line == "Color: #FFFFFFFF"),
            "{advanced:?}"
        );
        assert!(!advanced.iter().any(|line| line.contains("Dyed")));
    }

    /// `CanDestroy` and `CanPlaceOn` are separate HideFlags blocks: each owns its separator,
    /// translated header, registered block lookup, and `missingno` fallback.
    #[test]
    fn adventure_block_lists_render_in_java_order_and_obey_their_independent_hide_flags() {
        let mut s = stack("apple");
        let mut root = NbtCompound::new();
        root.insert(
            "CanDestroy",
            Nbt::list(vec![
                Nbt::String("minecraft:stone".into()),
                Nbt::String("minecraft:no_such_block".into()),
            ]),
        );
        root.insert("CanPlaceOn", Nbt::list(vec![Nbt::String("dirt".into())]));
        s.nbt = Some(root);
        assert_eq!(
            build(&s, false),
            vec![
                format!("Apple{RESET}"),
                String::new(),
                format!("{GRAY}Can break:"),
                format!("{DARK_GRAY}Stone"),
                format!("{DARK_GRAY}missingno"),
                String::new(),
                format!("{GRAY}Can be placed on:"),
                format!("{DARK_GRAY}Dirt"),
            ]
        );

        s.nbt.as_mut().unwrap().insert("HideFlags", Nbt::Int(8));
        let no_destroy = build(&s, false);
        assert!(
            !no_destroy.iter().any(|line| line.contains("Can break")),
            "{no_destroy:?}"
        );
        assert!(no_destroy
            .iter()
            .any(|line| line.contains("Can be placed on")));

        s.nbt.as_mut().unwrap().insert("HideFlags", Nbt::Int(16));
        let no_place = build(&s, false);
        assert!(no_place.iter().any(|line| line.contains("Can break")));
        assert!(
            !no_place
                .iter()
                .any(|line| line.contains("Can be placed on")),
            "{no_place:?}"
        );
    }

    #[test]
    fn written_book_and_banner_information_follow_java_nbt_rules() {
        let mut book = stack("written_book");
        let mut book_tag = NbtCompound::new();
        book_tag.insert("author", Nbt::String("Ari".into()));
        book_tag.insert("generation", Nbt::Int(2));
        book.nbt = Some(book_tag);
        assert!(build(&book, false)
            .ends_with(&[format!("{GRAY}by Ari"), format!("{GRAY}Copy of a copy"),]));

        let mut banner = stack("banner");
        let pattern = |id: &str, color: i32| {
            let mut entry = NbtCompound::new();
            entry.insert("Pattern", Nbt::String(id.into()));
            entry.insert("Color", Nbt::Int(color));
            Nbt::Compound(entry)
        };
        let mut block_entity = NbtCompound::new();
        block_entity.insert(
            "Patterns",
            Nbt::list(vec![
                pattern("bs", 1),
                pattern("unknown", 4),
                pattern("flo", 12),
                pattern("bo", -1),
                pattern("cre", 15),
                pattern("moj", 0),
                pattern("ts", 2), // Java stops before the seventh entry.
            ]),
        );
        let mut banner_tag = NbtCompound::new();
        banner_tag.insert("BlockEntityTag", Nbt::Compound(block_entity));
        banner.nbt = Some(banner_tag);
        assert_eq!(
            banner_information_lines(&banner, None),
            vec![
                "Red Base Fess",
                "Light Blue Flower Charge",
                "Black Bordure",
                "White Creeper Charge",
                "Black Thing",
            ]
        );
    }

    #[test]
    fn banner_and_anvil_class_names_follow_metadata_and_class_overrides() {
        let mut language = Language::new("xx_TEST");
        language.insert("item.banner.red.name", "Translated Red Banner");
        language.insert(
            "tile.anvil.veryDamaged.name",
            "Translated Very Damaged Anvil",
        );

        let mut banner = stack("banner");
        banner.damage = 14; // orange in EnumDyeColor.byDyeDamage order
        assert_eq!(build(&banner, false), [format!("Orange Banner{RESET}")]);

        let mut block_entity = NbtCompound::new();
        block_entity.insert("Base", Nbt::Int(1)); // BlockEntityTag wins over stack metadata
        let mut root = NbtCompound::new();
        root.insert("BlockEntityTag", Nbt::Compound(block_entity));
        banner.nbt = Some(root);
        assert_eq!(build(&banner, false), [format!("Red Banner{RESET}")]);
        assert_eq!(
            build_localized(&banner, false, &language),
            [format!("Translated Red Banner{RESET}")]
        );

        let mut wrong_block_entity = NbtCompound::new();
        wrong_block_entity.insert("Base", Nbt::String("wrong type".into()));
        let mut wrong_root = NbtCompound::new();
        wrong_root.insert("BlockEntityTag", Nbt::Compound(wrong_block_entity));
        banner.nbt = Some(wrong_root);
        assert_eq!(
            build(&banner, false),
            [format!("Black Banner{RESET}")],
            "a present wrong-typed Base follows Java getInteger's zero fallback"
        );

        let mut anvil = stack("anvil");
        anvil.damage = 2;
        assert_eq!(build(&anvil, false), [format!("Very Damaged Anvil{RESET}")]);
        assert_eq!(
            build_localized(&anvil, false, &language),
            [format!("Translated Very Damaged Anvil{RESET}")]
        );
        anvil.damage = i16::MAX;
        assert_eq!(build(&anvil, false), [format!("Anvil{RESET}")]);

        let mut apple = stack("golden_apple");
        assert_eq!(build(&apple, false), [format!("Golden Apple{RESET}")]);
        assert_eq!(rarity(&apple), Rarity::Rare);
        apple.damage = 1;
        assert_eq!(
            build(&apple, false),
            [format!("Golden Apple{RESET}")],
            "ItemAppleGold keeps the shared appleGold display key"
        );
        assert_eq!(rarity(&apple), Rarity::Epic);
    }

    #[test]
    fn enchanted_book_stored_enchantments_precede_regular_enchantments_and_respect_hide_info() {
        let mut book = stack("enchanted_book");
        let mut stored = NbtCompound::new();
        stored.insert("id", Nbt::Short(16)); // Sharpness
        stored.insert("lvl", Nbt::Short(3));
        let mut ordinary = NbtCompound::new();
        ordinary.insert("id", Nbt::Short(34)); // Unbreaking
        ordinary.insert("lvl", Nbt::Short(1));
        let mut root = NbtCompound::new();
        root.insert("StoredEnchantments", Nbt::list(vec![Nbt::Compound(stored)]));
        root.insert("ench", Nbt::list(vec![Nbt::Compound(ordinary)]));
        book.nbt = Some(root);

        let lines = build(&book, false);
        assert_eq!(
            lines,
            vec![
                format!("Enchanted Book{RESET}"),
                "Sharpness III".to_owned(),
                "Unbreaking I".to_owned(),
            ]
        );

        // HideFlags bit 32 gates Item.addInformation only; the generic `ench` list is separate.
        book.nbt.as_mut().unwrap().insert("HideFlags", Nbt::Int(32));
        assert_eq!(
            build(&book, false),
            vec![format!("Enchanted Book{RESET}"), "Unbreaking I".to_owned()]
        );
    }

    #[test]
    fn attribute_block_uses_active_language_for_both_the_name_and_template() {
        let mut s = stack("diamond_sword");
        let mut modifier = NbtCompound::new();
        modifier.insert("Name", Nbt::String("server modifier".into()));
        modifier.insert("AttributeName", Nbt::String("generic.attackDamage".into()));
        modifier.insert("Amount", Nbt::Double(2.5));
        modifier.insert("Operation", Nbt::Int(0));
        modifier.insert("UUIDMost", Nbt::Long(1));
        modifier.insert("UUIDLeast", Nbt::Long(2));
        let mut root = NbtCompound::new();
        root.insert(
            "AttributeModifiers",
            Nbt::list(vec![Nbt::Compound(modifier)]),
        );
        s.nbt = Some(root);
        let mut language = Language::new("xx_TEST");
        language.insert("attribute.name.generic.attackDamage", "Dégâts");
        language.insert("attribute.modifier.plus.0", "PLUS %2$s (%1$s)");
        let lines = build_localized(&s, false, &language);
        assert!(
            lines
                .iter()
                .any(|line| line == &format!("{BLUE}PLUS Dégâts (2.5)")),
            "{lines:?}"
        );
    }

    #[test]
    fn durability_line_only_appears_in_advanced_mode_and_only_when_damaged() {
        let mut s = stack("diamond_sword");
        s.damage = 100;
        let plain = build(&s, false);
        assert!(!plain.iter().any(|l| l.contains("Durability")));
        let advanced = build(&s, true);
        assert!(advanced
            .iter()
            .any(|l| l.contains("Durability: ") && l.contains("1461")));
    }

    #[test]
    fn undamaged_tool_shows_no_durability_line_even_when_advanced() {
        let s = stack("diamond_sword");
        let advanced = build(&s, true);
        assert!(!advanced.iter().any(|l| l.contains("Durability")));
    }

    #[test]
    fn a_sword_gets_a_synthesised_attack_damage_line() {
        // The *number* matters and this test used to assert only the substring, which is why a
        // `1.0 +` survived here. java: `ItemSword.attackDamage = 4.0F +
        // ToolMaterial.EMERALD.getDamageVsEntity() (3.0F)` = 7, printed as-is.
        let s = stack("diamond_sword");
        let lines = build(&s, false);
        assert!(
            lines.iter().any(|l| l.contains("+7 Attack Damage")),
            "diamond sword is +7, not +8: {lines:?}"
        );
        // `ItemTool.damageVsEntity` for an axe is `3.0F + material` -> iron axe 5, and for a
        // pickaxe `2.0F + material` -> diamond pickaxe 5. Two more materials and two more
        // tool classes, so the assertion is not pinned to one row of the table.
        for (name, want) in [
            ("iron_axe", "+5 Attack Damage"),
            ("diamond_pickaxe", "+5 Attack Damage"),
            ("wooden_sword", "+4 Attack Damage"),
        ] {
            let lines = build(&stack(name), false);
            assert!(
                lines.iter().any(|l| l.contains(want)),
                "{name} should read {want}: {lines:?}"
            );
        }
    }

    /// java: `ItemStack.getTooltip:758-761` adds
    /// `EnchantmentHelper.getModifierForCreature(stack, UNDEFINED)` to the built-in weapon
    /// modifier. `EnchantmentDamage.calcDamageByCreature` makes that Sharpness-only here:
    /// level * 1.25, while Smite and Bane require a concrete creature attribute.
    #[test]
    fn sharpness_but_not_smite_increases_the_attack_damage_tooltip() {
        let with_enchantment = |id: i16, level: i16| {
            let mut s = stack("diamond_sword");
            let mut enchantment = NbtCompound::new();
            enchantment.insert("id", Nbt::Short(id));
            enchantment.insert("lvl", Nbt::Short(level));
            let mut root = NbtCompound::new();
            root.insert("ench", Nbt::list(vec![Nbt::Compound(enchantment)]));
            s.nbt = Some(root);
            build(&s, false)
        };

        let sharpness = with_enchantment(16, 3);
        assert!(
            sharpness
                .iter()
                .any(|line| line.contains("+10.75 Attack Damage")),
            "diamond sword 7 + Sharpness III 3.75: {sharpness:?}"
        );

        let smite = with_enchantment(17, 3);
        assert!(
            smite.iter().any(|line| line.contains("+7 Attack Damage")),
            "Smite is zero against UNDEFINED: {smite:?}"
        );
        assert!(
            !smite
                .iter()
                .any(|line| line.contains("+10.75 Attack Damage")),
            "the damage-enchantment id, not merely its level, gates the bonus: {smite:?}"
        );
    }

    /// The enchantment term is UUID-gated, not attribute-name-gated. A server-provided modifier
    /// gets the term only when it carries `Item.itemModifierUUID`; this is the exact comparison
    /// at `ItemStack.getTooltip:758`.
    #[test]
    fn nbt_attribute_gets_sharpness_only_for_the_builtin_item_modifier_uuid() {
        let with_uuid = |uuid_most: i64, uuid_least: i64| {
            let mut s = stack("diamond_sword");
            let mut sharpness = NbtCompound::new();
            sharpness.insert("id", Nbt::Short(16));
            sharpness.insert("lvl", Nbt::Short(2));
            let mut modifier = NbtCompound::new();
            modifier.insert("Name", Nbt::String("Weapon modifier".into()));
            modifier.insert("AttributeName", Nbt::String("generic.attackDamage".into()));
            modifier.insert("Amount", Nbt::Double(9.0));
            modifier.insert("Operation", Nbt::Int(0));
            modifier.insert("UUIDMost", Nbt::Long(uuid_most));
            modifier.insert("UUIDLeast", Nbt::Long(uuid_least));
            let mut root = NbtCompound::new();
            root.insert("ench", Nbt::list(vec![Nbt::Compound(sharpness)]));
            root.insert(
                "AttributeModifiers",
                Nbt::list(vec![Nbt::Compound(modifier)]),
            );
            s.nbt = Some(root);
            build(&s, false)
        };

        let builtin = with_uuid(ITEM_MODIFIER_UUID_MOST, ITEM_MODIFIER_UUID_LEAST);
        assert!(
            builtin.iter().any(|line| line.contains("+11.5")),
            "NBT amount 9 + Sharpness II 2.5: {builtin:?}"
        );

        let unrelated = with_uuid(1, 2);
        assert!(
            unrelated.iter().any(|line| line.contains("+9")),
            "an unrelated modifier UUID keeps its original amount: {unrelated:?}"
        );
        assert!(
            !unrelated.iter().any(|line| line.contains("+11.5")),
            "attribute name alone must not trigger the enchantment term: {unrelated:?}"
        );
    }

    /// java: `new DecimalFormat("#.###")`, whose default rounding mode is HALF_EVEN. The first
    /// two exact ties distinguish it from both the former two-digit formatter and `f64::round`.
    #[test]
    fn attribute_amounts_use_three_digits_and_half_even_rounding() {
        for (value, expected) in [
            (0.0625, "0.062"),
            (0.0635, "0.064"),
            (0.125, "0.125"),
            (1.2345, "1.234"),
            (1.2355, "1.236"),
            (-1.2345, "-1.234"),
            (2.5, "2.5"),
            (0.0, "0"),
            (-0.0, "-0"),
            (f64::INFINITY, "∞"),
            (f64::NEG_INFINITY, "-∞"),
        ] {
            assert_eq!(format_amount(value), expected, "{value:?}");
        }
        assert_eq!(format_amount(f64::NAN), "NaN");
    }

    #[test]
    fn explicit_attribute_modifiers_nbt_wins_over_the_synthesised_default() {
        let mut s = stack("diamond_sword");
        let mut modifier = NbtCompound::new();
        modifier.insert("Name", Nbt::String("Weapon modifier".into()));
        modifier.insert("AttributeName", Nbt::String("generic.attackDamage".into()));
        modifier.insert("Amount", Nbt::Double(9.0));
        modifier.insert("Operation", Nbt::Int(0));
        modifier.insert("UUIDMost", Nbt::Long(1));
        modifier.insert("UUIDLeast", Nbt::Long(2));
        let mut root = NbtCompound::new();
        root.insert(
            "AttributeModifiers",
            Nbt::list(vec![Nbt::Compound(modifier)]),
        );
        s.nbt = Some(root);
        let lines = build(&s, false);
        assert!(lines
            .iter()
            .any(|l| l.contains("+9") && l.contains("Attack Damage")));
    }

    #[test]
    fn malformed_attribute_modifiers_are_dropped_at_vanillas_three_gates() {
        let modifier = |modifier_name: Option<&str>, operation: i32, most: i64, least: i64| {
            let mut nbt = NbtCompound::new();
            if let Some(name) = modifier_name {
                nbt.insert("Name", Nbt::String(name.to_owned()));
            }
            nbt.insert("AttributeName", Nbt::String("generic.attackDamage".into()));
            nbt.insert("Amount", Nbt::Double(1.0));
            nbt.insert("Operation", Nbt::Int(operation));
            nbt.insert("UUIDMost", Nbt::Long(most));
            nbt.insert("UUIDLeast", Nbt::Long(least));
            nbt
        };

        assert!(attribute_line_from_nbt(&modifier(Some("valid"), 0, 1, 2), 0.0, None).is_some());
        for invalid in [
            modifier(None, 0, 1, 2),
            modifier(Some(""), 0, 1, 2),
            modifier(Some("valid"), -1, 1, 2),
            modifier(Some("valid"), 3, 1, 2),
            modifier(Some("valid"), 0, 0, 2),
            modifier(Some("valid"), 0, 1, 0),
        ] {
            assert_eq!(
                attribute_line_from_nbt(&invalid, 0.0, None),
                None,
                "{invalid:?}"
            );
        }

        // `AttributeName` is the multimap key, not AttributeModifier.name. Java's
        // NBTTagCompound.getString returns an empty string when it is absent and does not reject
        // that key, so it is deliberately not a fourth validity gate.
        let mut no_attribute_name = modifier(Some("valid"), 0, 1, 2);
        no_attribute_name.remove("AttributeName");
        assert!(attribute_line_from_nbt(&no_attribute_name, 0.0, None).is_some());

        // A valid NaN modifier is in the map (so the separator stays), but renders no line.
        let mut nan_amount = modifier(Some("valid"), 0, 1, 2);
        nan_amount.insert("Amount", Nbt::Double(f64::NAN));
        assert_eq!(
            attribute_line_from_nbt(&nan_amount, 0.0, None),
            Some(None),
            "Java's d0 > 0 / d0 < 0 branches both reject NaN"
        );
    }

    /// java: the attribute block's blank separator follows `!multimap.isEmpty()`,
    /// not the rendered lines, so a valid zero-amount modifier leaves the separator on its own at
    /// the end. Negative control: the old builder dropped it together with the modifier.
    #[test]
    fn a_zero_amount_attribute_modifier_still_adds_the_separator_line() {
        let mut s = stack("diamond_sword");
        let mut modifier = NbtCompound::new();
        modifier.insert("Name", Nbt::String("hidden".into()));
        modifier.insert("AttributeName", Nbt::String("generic.attackDamage".into()));
        modifier.insert("Amount", Nbt::Double(0.0));
        modifier.insert("Operation", Nbt::Int(0));
        modifier.insert("UUIDMost", Nbt::Long(1));
        modifier.insert("UUIDLeast", Nbt::Long(2));
        let mut root = NbtCompound::new();
        root.insert(
            "AttributeModifiers",
            Nbt::list(vec![Nbt::Compound(modifier)]),
        );
        s.nbt = Some(root);
        let lines = build(&s, false);
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[1], "");
    }

    /// java: F3+H's `"Durability: " + (getMaxDamage() - getItemDamage())` is
    /// `int` arithmetic, negative past the maximum. Negative control: `saturating_sub` printed 0.
    #[test]
    fn advanced_durability_goes_negative_for_an_over_damaged_stack() {
        let mut s = stack("diamond_sword");
        s.damage = 2000;
        let lines = build(&s, true);
        assert!(
            lines
                .iter()
                .any(|line| line.ends_with("Durability: -439 / 1561")),
            "{lines:?}"
        );
    }

    #[test]
    fn unbreakable_flag_adds_a_line() {
        let mut s = stack("diamond_sword");
        let mut root = NbtCompound::new();
        root.insert("Unbreakable", Nbt::Byte(1));
        s.nbt = Some(root);
        let lines = build(&s, false);
        assert!(lines.iter().any(|l| l.contains("Unbreakable")));
    }

    /// java: `ItemStack.getTooltip:749` gates the whole attribute block on
    /// `(i1 & 2) == 0` and `:786` gates the Unbreakable line on `(i1 & 4) == 0`. Both bits were
    /// read out of the NBT and then discarded, so `HideFlags: 2` — the standard way a hub server
    /// dresses a cosmetic item — still drew "+7 Attack Damage" *and* its blank separator, which
    /// also makes the tooltip box the wrong height.
    #[test]
    fn hide_flags_two_and_four_suppress_the_attribute_block_and_the_unbreakable_line() {
        let with_flags = |flags: i32| {
            let mut s = stack("diamond_sword");
            let mut root = NbtCompound::new();
            root.insert("Unbreakable", Nbt::Byte(1));
            root.insert("HideFlags", Nbt::Int(flags));
            s.nbt = Some(root);
            build(&s, false)
        };

        let shown = with_flags(0);
        assert!(shown.iter().any(|l| l.contains("Attack Damage")));
        assert!(shown.iter().any(|l| l.contains("Unbreakable")));
        assert!(
            shown.iter().any(String::is_empty),
            "the attribute block is preceded by a blank separator line"
        );

        let no_attrs = with_flags(2);
        assert!(
            !no_attrs.iter().any(|l| l.contains("Attack Damage")),
            "{no_attrs:?}"
        );
        assert!(
            !no_attrs.iter().any(String::is_empty),
            "the separator goes with the block it separates: {no_attrs:?}"
        );
        assert!(
            no_attrs.iter().any(|l| l.contains("Unbreakable")),
            "bit 2 must not take bit 4's line with it"
        );

        let no_unbreakable = with_flags(4);
        assert!(!no_unbreakable.iter().any(|l| l.contains("Unbreakable")));
        assert!(no_unbreakable.iter().any(|l| l.contains("Attack Damage")));

        let neither = with_flags(2 | 4);
        assert_eq!(neither.len(), 1, "just the name left: {neither:?}");
    }

    /// The en_US table is `enchantment.level.1` .. `.10` and nothing else; every other level
    /// renders as the untranslated key, which is what `StringTranslate.tryTranslateKey` returns
    /// on a miss.
    #[test]
    fn the_level_token_is_the_lang_table_and_then_the_raw_key() {
        for (level, want) in [
            (1i16, "I"),
            (2, "II"),
            (3, "III"),
            (4, "IV"),
            (5, "V"),
            (9, "IX"),
            (10, "X"),
        ] {
            assert_eq!(enchantment_level_token(level), want);
        }
        // Out of range in both directions, and the two a server actually sends.
        assert_eq!(enchantment_level_token(11), "enchantment.level.11");
        assert_eq!(enchantment_level_token(100), "enchantment.level.100");
        assert_eq!(enchantment_level_token(0), "enchantment.level.0");
        assert_eq!(enchantment_level_token(-1), "enchantment.level.-1");
    }

    /// java: `getTranslatedName` has no `getMaxLevel() == 1` short-circuit in 1.8.9 - that
    /// arrived in 1.9. Silk Touch (`max_level == 1`) still reads "Silk Touch I".
    #[test]
    fn a_single_level_enchantment_still_carries_its_numeral_in_one_eight_nine() {
        let silk = mc_data::enchantment(33).expect("silk touch");
        assert_eq!(
            silk.max_level, 1,
            "the case the 1.9 rule would short-circuit"
        );
        assert_eq!(enchantment_display_name(silk, 1), "Silk Touch I");
        let sharpness = mc_data::enchantment(16).expect("sharpness");
        assert_eq!(enchantment_display_name(sharpness, 3), "Sharpness III");
    }
}
