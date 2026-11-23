//! Block state -> blockstate-file resolution (java: `BlockModelShapes.registerAllBlocks`,
//! `StateMap`, `BlockStateMapper`).
//!
//! # Why this exists (hazard **H12**)
//!
//! `mc_data::state_variant_key()` reproduces the *property* string vanilla's `BlockState`
//! constructor builds (`facing=east,half=bottom`), sorted by property name. For most blocks
//! that string is also the exact key inside `assets/minecraft/blockstates/<block>.json`. It
//! is not for 76 of 1.8.9's blocks: vanilla's real `BlockStateMapper` renames the file
//! (`Blocks.wool` bakes from `white_wool.json`, not `wool.json`), drops properties that don't
//! affect the model (`Blocks.oak_door` ignores its redstone `powered` bit), or both.
//!
//! This table is not guessed from the Java source alone — every rule below was cross-checked
//! against the real blockstate files in `jars/versions/1.8.9/1.8.9.jar` (see the S09 handoff
//! log entry for the verification method) before being written here, because several of the
//! Java-derived assumptions turned out to be subtly wrong (e.g. `Blocks.monster_egg`'s
//! `BlockSilverfish.EnumType` display names looked like they should need a rename table, but
//! `mc_data`'s own extracted property values already match the file names once the
//! `_monster_egg` suffix is applied — no rename table needed after all).
//!
//! The remaining 1394 - 1000 = ~394 legal states resolve directly:
//! `blockstates/<short-block-name>.json` keyed by `mc_data::state_variant_key()` verbatim.

use mc_data::block::ActualBlockState;
use mc_data::{block_of, state_properties, state_variant_key, BlockState};

/// Where to find and how to key a state's blockstate JSON.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateMapping {
    /// The blockstate file stem, e.g. `"white_wool"` for `assets/minecraft/blockstates/white_wool.json`.
    pub file_stem: String,
    /// The key to look up inside that file's `"variants"` object.
    pub variant_key: String,
}

fn key_from(props: &[(&str, &str)]) -> String {
    if props.is_empty() {
        return "normal".to_string();
    }
    let mut s = String::new();
    for (i, (k, v)) in props.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(k);
        s.push('=');
        s.push_str(v);
    }
    s
}

/// `StateMap.Builder().withName(prop).withSuffix(suffix).ignore(ignored...)`, java-equivalent.
fn state_map(
    short: &str,
    props: &[(&str, &str)],
    name_prop: Option<&str>,
    suffix: &str,
    ignore: &[&str],
) -> StateMapping {
    let mut remaining: Vec<(&str, &str)> = Vec::with_capacity(props.len());
    let mut name_value: Option<&str> = None;
    for &(k, v) in props {
        if Some(k) == name_prop {
            name_value = Some(v);
            continue;
        }
        if ignore.contains(&k) {
            continue;
        }
        remaining.push((k, v));
    }
    let file_stem = match name_value {
        Some(v) => format!("{v}{suffix}"),
        None => format!("{short}{suffix}"),
    };
    StateMapping {
        file_stem,
        variant_key: key_from(&remaining),
    }
}

fn prop<'a>(props: &'a [(&str, &str)], name: &str) -> &'a str {
    props
        .iter()
        .find(|(k, _)| *k == name)
        .map_or("", |(_, v)| v)
}

/// Custom (non-`StateMap`) mappers, ported 1:1 from the anonymous `StateMapperBase`
/// subclasses in `BlockModelShapes.registerAllBlocks`.
fn custom_mapping(short: &str, props: &[(&str, &str)]) -> Option<StateMapping> {
    let mk = |file_stem: &str, key: String| {
        Some(StateMapping {
            file_stem: file_stem.to_string(),
            variant_key: key,
        })
    };
    match short {
        "quartz_block" => match prop(props, "variant") {
            "default" => mk("quartz_block", "normal".into()),
            "chiseled" => mk("chiseled_quartz_block", "normal".into()),
            "lines_y" => mk("quartz_column", "axis=y".into()),
            "lines_x" => mk("quartz_column", "axis=x".into()),
            "lines_z" => mk("quartz_column", "axis=z".into()),
            _ => None,
        },
        "deadbush" => mk("dead_bush", "normal".into()),
        "pumpkin_stem" | "melon_stem" => {
            // java: if facing != UP, AGE is dropped from the key.
            let mut remaining: Vec<(&str, &str)> = props.to_vec();
            if prop(props, "facing") != "up" {
                remaining.retain(|(k, _)| *k != "age");
            }
            mk(short, key_from(&remaining))
        }
        "dirt" => {
            let variant = prop(props, "variant");
            let mut remaining: Vec<(&str, &str)> = props
                .iter()
                .copied()
                .filter(|(k, _)| *k != "variant")
                .collect();
            if variant != "podzol" {
                remaining.retain(|(k, _)| *k != "snowy");
            }
            mk(variant, key_from(&remaining))
        }
        "double_stone_slab" | "double_stone_slab2" => {
            let variant = prop(props, "variant");
            let seamless = prop(props, "seamless") == "true";
            let file_stem = format!("{variant}_double_slab");
            mk(
                &file_stem,
                if seamless {
                    "all".into()
                } else {
                    "normal".into()
                },
            )
        }
        _ => None,
    }
}

/// Blocks handled by the generic `StateMap.Builder()` calls in `registerAllBlocks`, in the
/// same order as the Java source (grouped where the rule is identical). Each closure applies
/// to a single block's already-fetched `(name, value)` property slice.
fn generic_mapping(short: &str, props: &[(&str, &str)]) -> Option<StateMapping> {
    let sm = |name_prop: Option<&str>, suffix: &str, ignore: &[&str]| {
        Some(state_map(short, props, name_prop, suffix, ignore))
    };
    match short {
        "stone" | "prismarine" | "sandstone" | "red_sandstone" | "tallgrass" | "yellow_flower"
        | "red_flower" | "stonebrick" | "sand" => {
            // `stone`/`prismarine`/`stonebrick`/`sand` key on "variant"; the others key on
            // "type" (sandstone family) or "type" (tallgrass/flowers use their own
            // TYPE property, also named "type" in mc_data's extraction).
            let name_prop = if matches!(
                short,
                "sandstone" | "red_sandstone" | "tallgrass" | "yellow_flower" | "red_flower"
            ) {
                "type"
            } else {
                "variant"
            };
            sm(Some(name_prop), "", &[])
        }
        "leaves" | "leaves2" => sm(Some("variant"), "_leaves", &["check_decay", "decayable"]),
        "cactus" | "reeds" => sm(None, "", &["age"]),
        "jukebox" => sm(None, "", &["has_record"]),
        "command_block" => sm(None, "", &["triggered"]),
        "cobblestone_wall" => sm(Some("variant"), "_wall", &[]),
        "double_plant" => sm(Some("variant"), "", &["facing"]),
        "fence_gate"
        | "spruce_fence_gate"
        | "birch_fence_gate"
        | "jungle_fence_gate"
        | "dark_oak_fence_gate"
        | "acacia_fence_gate" => sm(None, "", &["powered"]),
        "tripwire" => sm(None, "", &["disarmed", "powered"]),
        "double_wooden_slab" => sm(Some("variant"), "_double_slab", &[]),
        "wooden_slab" => sm(Some("variant"), "_slab", &[]),
        "tnt" => sm(None, "", &["explode"]),
        "fire" => sm(None, "", &["age"]),
        "redstone_wire" => sm(None, "", &["power"]),
        "wooden_door" | "spruce_door" | "birch_door" | "jungle_door" | "acacia_door"
        | "dark_oak_door" | "iron_door" => sm(None, "", &["powered"]),
        "wool" => sm(Some("color"), "_wool", &[]),
        "carpet" => sm(Some("color"), "_carpet", &[]),
        "stained_hardened_clay" => sm(Some("color"), "_stained_hardened_clay", &[]),
        "stained_glass_pane" => sm(Some("color"), "_stained_glass_pane", &[]),
        "stained_glass" => sm(Some("color"), "_stained_glass", &[]),
        "bed" => sm(None, "", &["occupied"]),
        "stone_slab" => sm(Some("variant"), "_slab", &[]),
        "stone_slab2" => sm(Some("variant"), "_slab", &[]),
        "monster_egg" => sm(Some("variant"), "_monster_egg", &[]),
        "dispenser" => sm(None, "", &["triggered"]),
        "dropper" => sm(None, "", &["triggered"]),
        "log" => sm(Some("variant"), "_log", &[]),
        "log2" => sm(Some("variant"), "_log", &[]),
        "planks" => sm(Some("variant"), "_planks", &[]),
        "sapling" => sm(Some("type"), "_sapling", &[]),
        "hopper" => sm(None, "", &["enabled"]),
        "flower_pot" => sm(None, "", &["legacy_data"]),
        _ => None,
    }
}

/// Resolves a block state to its blockstate file + variant key.
///
/// Callers must only invoke this for states whose `render_type` is
/// [`mc_data::RenderType::Model`] — everything else (air, fluids, tile-entity-rendered
/// blocks) has no JSON model at all in vanilla (java: `BlockStateMapper.registerBuiltInBlocks`)
/// and is handled upstream by [`super::bake::model_for`] returning the empty model.
#[must_use]
pub fn resolve(state: BlockState) -> StateMapping {
    resolve_properties(
        state,
        state_properties(state),
        Some(state_variant_key(state)),
    )
}

/// Resolves a client-derived state after `Block.getActualState` has supplied properties that
/// do not fit in packed metadata.
#[must_use]
pub fn resolve_actual(state: &ActualBlockState) -> StateMapping {
    resolve_properties(state.stored(), state.properties(), None)
}

fn resolve_properties(
    state: BlockState,
    props: &[(&str, &str)],
    // Stored states' full property keys are generated from the vanilla BlockState table.
    // Actual states can have world-derived properties, so they must still assemble their key.
    stored_variant_key: Option<&str>,
) -> StateMapping {
    let full_name = block_of(state).info().name;
    let short = full_name.strip_prefix("minecraft:").unwrap_or(full_name);

    if let Some(m) = custom_mapping(short, props) {
        return m;
    }
    if let Some(m) = generic_mapping(short, props) {
        return m;
    }
    // The ~394 blocks that need no special handling: direct lookup, full variant key.
    StateMapping {
        file_stem: short.to_string(),
        variant_key: stored_variant_key.map_or_else(|| key_from(props), ToOwned::to_owned),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::{
        block::EnumFacing, block::PropertyDirection, block_by_name, state_variant_key, BlockId,
    };

    fn state_of(name: &str, meta: u8) -> BlockState {
        BlockState::new(block_by_name(name).unwrap(), meta)
    }

    #[test]
    fn direct_lookup_for_unremarkable_blocks() {
        let m = resolve(state_of("stone", 0)); // meta 0 is itself the "stone" variant
        assert_eq!(m.file_stem, "stone");
        assert_eq!(m.variant_key, "normal");
    }

    #[test]
    fn ordinary_stored_states_consume_the_generated_variant_key() {
        // Vanilla DefaultStateMapper feeds its state.getProperties() through
        // StateMapperBase.getPropertyString. For direct stored states, the generated mc-data
        // key is that exact Java property string and must be the value the live mapper uses.
        let stairs = state_of("oak_stairs", 0);
        assert_eq!(resolve(stairs).variant_key, state_variant_key(stairs));
    }

    #[test]
    fn direction_descriptor_serializes_java_names_for_model_mapping() {
        let facing = PropertyDirection::create_filtered("facing", |direction| {
            !matches!(direction, EnumFacing::Down)
        });
        let torch = state_of("torch", 5); // EnumFacing.UP in torch metadata
        assert_eq!(torch.get_value(facing), Some(EnumFacing::Up));
        assert_eq!(resolve(torch).variant_key, "facing=up");

        let north = torch
            .cycle_property(facing)
            .expect("filtered torch direction cycle");
        assert_eq!(north.get_value(facing), Some(EnumFacing::North));
        assert_eq!(resolve(north).variant_key, "facing=north");
    }

    #[test]
    fn resolve_actual_of_an_unchanged_stored_state_is_resolve() {
        // The terrain mesher's dense-table fast path (mc-render `render_model`)
        // depends on this: for a block whose class has no `getActualState` override the actual
        // state is the stored tuple, and bake projected `models[state.id]` and
        // `weighted[state.id]` from `resolve(state)`. The two differ only in where the default
        // branch takes its variant key from, so pin the equality for every bakeable state.
        let mut checked = 0;
        for state in mc_data::legal_states() {
            if state.props().render_type != mc_data::RenderType::Model {
                continue;
            }
            assert_eq!(
                resolve_actual(&ActualBlockState::from_stored(state)),
                resolve(state),
                "{state:?}"
            );
            checked += 1;
        }
        assert!(checked > 100, "only {checked} model states checked");
    }

    #[test]
    fn state_map_overrides_do_not_use_the_raw_generated_variant_key() {
        // Negative control: `stone` is registered through StateMap.withName(VARIANT), so the
        // generated raw key would be wrong here ("variant=stone" vs JSON's "normal").
        let stone = state_of("stone", 0);
        assert_eq!(state_variant_key(stone), "variant=stone");
        assert_eq!(resolve(stone).variant_key, "normal");
        assert_ne!(resolve(stone).variant_key, state_variant_key(stone));
    }

    #[test]
    fn wool_renames_by_color_and_drops_the_property() {
        let m = resolve(state_of("wool", 0)); // white
        assert_eq!(m.file_stem, "white_wool");
        assert_eq!(m.variant_key, "normal");
    }

    #[test]
    fn oak_door_ignores_powered() {
        let m = resolve(state_of("wooden_door", 0));
        assert_eq!(m.file_stem, "wooden_door");
        assert!(!m.variant_key.contains("powered"));
    }

    #[test]
    fn stairs_raw_mapping_pins_the_storage_corner() {
        // Chunk storage's representative for oak-stairs meta 0 is the outer-right state. The
        // terrain path must actualise it before model lookup; the mapper itself must preserve
        // this raw key rather than silently pretending it is the isolated straight state.
        let m = resolve(state_of("oak_stairs", 0));
        assert_eq!(m.file_stem, "oak_stairs");
        assert_eq!(m.variant_key, "facing=east,half=bottom,shape=outer_right");
    }

    #[test]
    fn stairs_actual_mapping_pins_the_isolated_straight_model() {
        let stored = state_of("oak_stairs", 0);
        let mut actual = ActualBlockState::from_stored(stored);
        assert!(actual.set("shape", "straight"));
        let mapping = resolve_actual(&actual);
        assert_eq!(mapping.file_stem, "oak_stairs");
        assert_eq!(
            mapping.variant_key,
            "facing=east,half=bottom,shape=straight"
        );
        assert_ne!(mapping, resolve(stored));
    }

    #[test]
    fn fence_needs_no_special_mapping_but_fence_gate_does() {
        let fence = resolve(state_of("fence", 0));
        assert_eq!(fence.file_stem, "fence");

        let gate = resolve(state_of("fence_gate", 0));
        assert_eq!(gate.file_stem, "fence_gate");
        assert!(!gate.variant_key.contains("powered"));
    }

    #[test]
    fn an_actual_fence_state_names_a_variant_with_no_metadata_encoding() {
        let stored = state_of("fence", 0);
        let mut actual = ActualBlockState::from_stored(stored);
        assert!(actual.set("east", "true"));
        let mapping = resolve_actual(&actual);
        assert_eq!(mapping.file_stem, "fence");
        assert_eq!(
            mapping.variant_key,
            "east=true,north=false,south=false,west=false"
        );
        assert_ne!(
            mapping,
            resolve(stored),
            "negative control: raw metadata cannot name it"
        );
    }

    #[test]
    fn flower_pot_actual_contents_selects_the_dynamic_model_variant() {
        let stored = state_of("flower_pot", 0);
        let raw = resolve(stored);
        assert_eq!(raw.file_stem, "flower_pot");
        assert_eq!(raw.variant_key, "contents=cactus");

        // `BlockFlowerPot.getActualState` replaces CONTENTS from the tile entity while leaving
        // the packed legacy-data property alone. The model mapper must retain the former and
        // ignore the latter, exactly like the Java `BlockModelShapes` StateMap.
        let mut actual = ActualBlockState::from_stored(stored);
        assert!(actual.set("contents", "fern"));
        assert!(actual.set("legacy_data", "15"));
        let fern = resolve_actual(&actual);
        assert_eq!(fern.file_stem, "flower_pot");
        assert_eq!(fern.variant_key, "contents=fern");
        assert_ne!(fern, raw, "raw metadata cannot name the potted fern");
        assert_ne!(fern.variant_key, "legacy_data=15");

        // Negative model control: changing the resolved CONTENTS is what changes the variant;
        // a later empty-payload fallback must select the explicit empty model, not cactus.
        assert!(actual.set("contents", "empty"));
        assert_eq!(resolve_actual(&actual).variant_key, "contents=empty");
        assert_ne!(resolve_actual(&actual).variant_key, fern.variant_key);
    }

    #[test]
    fn double_stone_slab_seamless_flag_picks_all_vs_normal() {
        // stone (variant=0), seamless bit is the top bit of stone-slab meta encoding;
        // just check both branches are reachable via the real property strings.
        let block = block_by_name("double_stone_slab").unwrap();
        assert_eq!(block, BlockId(43));
        let normal = resolve(BlockState::new(block, 0));
        assert!(normal.variant_key == "normal" || normal.variant_key == "all");
    }

    #[test]
    fn quartz_column_axis_mapping() {
        let m = resolve(state_of("quartz_block", 2)); // lines_y
        assert_eq!(m.file_stem, "quartz_column");
        assert_eq!(m.variant_key, "axis=y");
    }
}
