//! Golden test: the generated Rust tables must agree with the extractor's JSON, row for
//! row, for every legal state.
//!
//! The JSON in `extractor/out/` is the recorded output of running the real 1.8.9 classes,
//! so this catches any codegen bug - a dropped field, a bad dedup key, a mangled float -
//! without needing a JDK or the Minecraft jars in CI. If this fails, `codegen.py` is wrong;
//! if the JSON itself is wrong, re-run `extractor/run.ps1`.

use mc_data::{
    achievement_by_stat_id, biome, block_info, block_props, entity_type, item_props,
    state_properties, state_variant_key, BlockId, BlockState, CollisionShape, ItemId, Material,
    RenderLayer, RenderType,
};
use serde_json::Value;

const BLOCKS: &str = include_str!("../extractor/out/blocks.json");
const ITEMS: &str = include_str!("../extractor/out/items.json");
const ENTITIES: &str = include_str!("../extractor/out/entities.json");
const BIOMES: &str = include_str!("../extractor/out/biomes.json");
const ACHIEVEMENTS: &str = include_str!("../extractor/out/achievements.json");
const MATERIALS: &str = include_str!("../extractor/out/materials.json");

// `Material` is repr(u8) in extraction order.  Keep the variants explicit so this test
// proves that codegen has neither reordered nor dropped a Java static material field.
const ALL_MATERIALS: [Material; 35] = [
    Material::Air,
    Material::Grass,
    Material::Ground,
    Material::Wood,
    Material::Rock,
    Material::Iron,
    Material::Anvil,
    Material::Water,
    Material::Lava,
    Material::Leaves,
    Material::Plants,
    Material::Vine,
    Material::Sponge,
    Material::Cloth,
    Material::Fire,
    Material::Sand,
    Material::Circuits,
    Material::Carpet,
    Material::Glass,
    Material::RedstoneLight,
    Material::Tnt,
    Material::Coral,
    Material::Ice,
    Material::PackedIce,
    Material::Snow,
    Material::CraftedSnow,
    Material::Cactus,
    Material::Clay,
    Material::Gourd,
    Material::DragonEgg,
    Material::Portal,
    Material::Cake,
    Material::Web,
    Material::Piston,
    Material::Barrier,
];

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn s(v: &Value) -> &str {
    v.as_str().expect("string")
}

fn b(v: &Value) -> bool {
    v.as_bool().expect("bool")
}

#[test]
fn materials_match_the_extractor() {
    let doc: Value = serde_json::from_str(MATERIALS).expect("materials.json parses");
    let rows = doc["materials"].as_array().expect("materials array");
    assert_eq!(rows.len(), ALL_MATERIALS.len());

    for (row, material) in rows.iter().zip(ALL_MATERIALS) {
        let info = material.info();
        let name = s(&row["name"]);
        assert_eq!(info.name, name, "{name} info name");
        assert_eq!(material.name(), name, "{name} accessor name");
        assert_eq!(info.liquid, b(&row["liquid"]), "{name} liquid");
        assert_eq!(material.is_liquid(), info.liquid, "{name} liquid accessor");
        assert_eq!(info.solid, b(&row["solid"]), "{name} solid");
        assert_eq!(material.is_solid(), info.solid, "{name} solid accessor");
        assert_eq!(info.blocks_light, b(&row["blocks_light"]), "{name} light");
        assert_eq!(
            material.blocks_light(),
            info.blocks_light,
            "{name} light accessor"
        );
        assert_eq!(
            info.blocks_movement,
            b(&row["blocks_movement"]),
            "{name} movement"
        );
        assert_eq!(
            material.blocks_movement(),
            info.blocks_movement,
            "{name} movement accessor"
        );
        assert_eq!(info.can_burn, b(&row["can_burn"]), "{name} burn");
        assert_eq!(material.can_burn(), info.can_burn, "{name} burn accessor");
        assert_eq!(
            info.replaceable,
            b(&row["replaceable"]),
            "{name} replaceable"
        );
        assert_eq!(
            material.is_replaceable(),
            info.replaceable,
            "{name} replaceable accessor"
        );
        assert_eq!(info.opaque, b(&row["opaque"]), "{name} opaque");
        assert_eq!(material.is_opaque(), info.opaque, "{name} opaque accessor");
        assert_eq!(
            info.tool_not_required,
            b(&row["tool_not_required"]),
            "{name} tool requirement"
        );
        assert_eq!(
            material.tool_not_required(),
            info.tool_not_required,
            "{name} tool requirement accessor"
        );
        assert_eq!(
            info.mobility_flag,
            row["mobility_flag"].as_u64().expect("mobility flag") as u8,
            "{name} mobility"
        );
        assert_eq!(
            material.mobility_flag(),
            info.mobility_flag,
            "{name} mobility accessor"
        );
        assert_eq!(
            info.map_color,
            row["map_color"].as_u64().expect("map color") as u32,
            "{name} map colour"
        );
        assert_eq!(
            material.map_color(),
            info.map_color,
            "{name} map colour accessor"
        );
    }
}

#[test]
fn blocks_match_the_extractor() {
    let doc: Value = serde_json::from_str(BLOCKS).expect("blocks.json parses");
    let blocks = doc["blocks"].as_array().expect("blocks array");
    let mut checked = 0usize;

    for blk in blocks {
        let bid = blk["id"].as_u64().unwrap() as u16;
        let info = block_info(BlockId(bid));
        assert_eq!(
            info.step_sound,
            s(&blk["step_sound"]),
            "{} step sound",
            s(&blk["name"])
        );
        assert_eq!(
            info.break_sound,
            s(&blk["break_sound"]),
            "{} break sound",
            s(&blk["name"])
        );
        for st in blk["states"].as_array().unwrap() {
            let state = BlockState::from_raw(st["state_id"].as_u64().unwrap() as u16);
            let props = block_props(state);

            if !b(&st["legal"]) {
                assert!(!props.legal, "state {:#06x} should be illegal", state.id);
                assert!(props.is_air());
                continue;
            }
            checked += 1;

            assert!(props.legal, "state {:#06x} should be legal", state.id);
            assert_eq!(props.block.0, bid, "state {:#06x} block id", state.id);
            assert_eq!(
                props.light_emission as u64,
                blk["light_value"].as_u64().unwrap(),
                "{} light",
                s(&blk["name"])
            );
            assert_eq!(
                props.light_opacity as u64,
                blk["light_opacity"].as_u64().unwrap(),
                "{} opacity",
                s(&blk["name"])
            );
            assert_eq!(
                props.hardness,
                f(&st["hardness_state"]) as f32,
                "{} hardness",
                s(&blk["name"])
            );
            assert_eq!(
                props.slipperiness,
                f(&blk["slipperiness"]) as f32,
                "{} slipperiness",
                s(&blk["name"])
            );
            assert_eq!(props.material.name(), s(&blk["material"]));
            assert_eq!(
                props.render_type,
                match blk["render_type"].as_i64().unwrap() {
                    -1 => RenderType::Invisible,
                    1 => RenderType::Liquid,
                    2 => RenderType::TileEntity,
                    _ => RenderType::Model,
                }
            );
            assert_eq!(
                props.render_layer,
                match s(&blk["render_layer"]) {
                    "SOLID" => RenderLayer::Solid,
                    "CUTOUT_MIPPED" => RenderLayer::CutoutMipped,
                    "CUTOUT" => RenderLayer::Cutout,
                    _ => RenderLayer::Translucent,
                }
            );
            assert_eq!(props.opaque_cube, b(&blk["opaque_cube"]));
            assert_eq!(props.full_cube, b(&blk["full_cube"]));
            assert_eq!(props.translucent, b(&blk["translucent"]));
            assert_eq!(
                props.use_neighbor_brightness,
                b(&blk["use_neighbor_brightness"])
            );
            assert_eq!(props.has_tile_entity, b(&blk["tile_entity"]));
            // `Block.hasTileEntity()` (`isBlockContainer`) and
            // `block instanceof ITileEntityProvider` are two different spellings that vanilla
            // uses in different places, and it is tempting to assume they differ for
            // `BlockRedstoneComparator` — which implements the interface without extending
            // `BlockContainer`. It does not: `BlockRedstoneComparator.java:38` sets
            // `isBlockContainer = true` by hand. The extractor emits both from the live
            // registry so this equality is *checked*, which is what lets every caller in the
            // workspace reproduce the `instanceof` with `has_tile_entity` and no id patch.
            assert_eq!(
                b(&blk["tile_entity"]),
                b(&blk["tile_entity_provider"]),
                "{}: hasTileEntity() and instanceof ITileEntityProvider disagree; every \
                 caller using `has_tile_entity` for the `instanceof` is now wrong",
                blk["name"]
            );
            assert_eq!(
                props.collision_depends_on_neighbors,
                b(&st["collision_depends_on_neighbors"])
            );

            // Bounds.
            let bounds = st["bounds"].as_array().unwrap();
            assert_eq!(props.bounds.min_x, f(&bounds[0]));
            assert_eq!(props.bounds.min_y, f(&bounds[1]));
            assert_eq!(props.bounds.min_z, f(&bounds[2]));
            assert_eq!(props.bounds.max_x, f(&bounds[3]));
            assert_eq!(props.bounds.max_y, f(&bounds[4]));
            assert_eq!(props.bounds.max_z, f(&bounds[5]));

            // Collision, box for box, exact doubles.
            let want = st["collision"].as_array().unwrap();
            let got = props.collision.boxes();
            if want.is_empty() {
                assert_eq!(
                    props.collision,
                    CollisionShape::Empty,
                    "{}",
                    s(&blk["name"])
                );
            }
            assert_eq!(got.len(), want.len(), "{} box count", s(&blk["name"]));
            for (g, w) in got.iter().zip(want) {
                let w = w.as_array().unwrap();
                assert_eq!(g.min_x, f(&w[0]), "{}", s(&blk["name"]));
                assert_eq!(g.min_y, f(&w[1]), "{}", s(&blk["name"]));
                assert_eq!(g.min_z, f(&w[2]), "{}", s(&blk["name"]));
                assert_eq!(g.max_x, f(&w[3]), "{}", s(&blk["name"]));
                assert_eq!(g.max_y, f(&w[4]), "{}", s(&blk["name"]));
                assert_eq!(g.max_z, f(&w[5]), "{}", s(&blk["name"]));
            }

            // Blockstate variant key.
            let want_props = st["properties"].as_object().unwrap();
            let got_props = state_properties(state);
            assert_eq!(got_props.len(), want_props.len(), "{}", s(&blk["name"]));
            for (k, v) in want_props {
                assert!(
                    got_props.contains(&(k.as_str(), s(v))),
                    "{} missing {k}={}",
                    s(&blk["name"]),
                    s(v)
                );
            }
            let expected_key = if want_props.is_empty() {
                "normal".to_string()
            } else {
                want_props
                    .iter()
                    .map(|(k, v)| format!("{k}={}", s(v)))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            assert_eq!(
                state_variant_key(state),
                expected_key,
                "{}",
                s(&blk["name"])
            );
        }
    }

    assert_eq!(checked, 1394, "1.8.9 registers 1394 legal block states");
}

#[test]
fn items_match_the_extractor() {
    let doc: Value = serde_json::from_str(ITEMS).expect("items.json parses");
    let items = doc["items"].as_array().expect("items array");
    for it in items {
        let id = ItemId(it["id"].as_u64().unwrap() as u16);
        let p = item_props(id);
        assert!(p.registered, "item {} missing", s(&it["name"]));
        assert_eq!(p.name, s(&it["name"]));
        assert_eq!(p.unlocalized, s(&it["unlocalized"]));
        assert_eq!(p.display_name, s(&it["display_name"]));
        assert_eq!(
            p.max_stack_size as u64,
            it["max_stack_size"].as_u64().unwrap()
        );
        assert_eq!(p.max_damage as u64, it["max_damage"].as_u64().unwrap());
        assert_eq!(p.damageable, b(&it["damageable"]));
        assert_eq!(p.has_subtypes, b(&it["has_subtypes"]));
        assert_eq!(
            p.enchantability as u64,
            it["enchantability"].as_u64().unwrap()
        );
        assert_eq!(p.food_heal as u64, it["food_heal"].as_u64().unwrap());
        for variant in it["food_variants"].as_array().unwrap() {
            let damage = variant["damage"].as_i64().unwrap() as i16;
            let stats = p.food_stats(damage);
            assert_eq!(stats.heal as u64, variant["heal"].as_u64().unwrap());
            assert_eq!(stats.saturation, f(&variant["saturation"]) as f32);
        }
        assert_eq!(p.attack_damage, f(&it["attack_damage"]) as f32);
        let block_id = it["block_id"].as_i64().unwrap();
        assert_eq!(
            p.block.map(|b| i64::from(b.0)),
            (block_id >= 0).then_some(block_id)
        );
    }
    assert_eq!(items.len(), mc_data::items().len());
}

#[test]
fn slab_display_names_follow_vanilla_default_itemstacks() {
    // The two red-sandstone block registry rows have no item counterpart for the double slab;
    // ItemSlab(182) is the stackable single slab and dispatches its display name through the
    // default ItemStack metadata.  Keep the adjacent old-slab item here as a control.
    for id in [BlockId(181), BlockId(182)] {
        assert_eq!(
            block_info(id).display_name,
            "Red Sandstone Slab",
            "block {id:?}"
        );
    }
    assert_eq!(item_props(ItemId(44)).display_name, "Stone Slab");
    assert_eq!(item_props(ItemId(182)).display_name, "Red Sandstone Slab");
}

#[test]
fn entities_match_the_extractor() {
    let doc: Value = serde_json::from_str(ENTITIES).expect("entities.json parses");
    for e in doc["entities"].as_array().unwrap() {
        let net_id = e["net_id"].as_u64().unwrap() as u8;
        let t = entity_type(net_id).unwrap_or_else(|| panic!("entity {net_id} missing"));
        assert_eq!(t.name(), s(&e["name"]));
        assert_eq!(t.is_living(), b(&e["living"]));
        assert_eq!(t.info().width, f(&e["width"]) as f32);
        assert_eq!(t.info().height, f(&e["height"]) as f32);
        assert_eq!(t.info().java_class, s(&e["class"]));
    }
}

#[test]
fn biomes_match_the_extractor() {
    let doc: Value = serde_json::from_str(BIOMES).expect("biomes.json parses");
    for bi in doc["biomes"].as_array().unwrap() {
        let id = bi["id"].as_u64().unwrap() as u8;
        let got = biome(id);
        assert!(got.registered, "biome {id} missing");
        assert_eq!(got.name, s(&bi["name"]));
        assert_eq!(got.temperature, f(&bi["temperature"]) as f32);
        assert_eq!(got.rainfall, f(&bi["rainfall"]) as f32);
        assert_eq!(got.snow, b(&bi["snow"]));
        assert_eq!(got.rain, b(&bi["rain"]));
        assert_eq!(
            got.water_color,
            bi["water_color"].as_u64().unwrap() as u32 & 0xFF_FFFF
        );
    }
}

#[test]
fn achievements_match_the_extractor() {
    let doc: Value = serde_json::from_str(ACHIEVEMENTS).expect("achievements.json parses");
    let rows = doc["achievements"].as_array().expect("achievements array");

    for row in rows {
        let stat_id = s(&row["stat_id"]);
        let a = achievement_by_stat_id(stat_id)
            .unwrap_or_else(|| panic!("achievement {stat_id} missing"));
        let info = a.info();

        assert_eq!(info.name, s(&row["name"]));
        assert_eq!(
            i64::from(info.display_column),
            row["display_column"].as_i64().unwrap()
        );
        assert_eq!(
            i64::from(info.display_row),
            row["display_row"].as_i64().unwrap()
        );
        assert_eq!(
            info.parent.map(|p| p.info().stat_id),
            row["parent"].as_str()
        );
        assert_eq!(u64::from(info.item.0), row["item_id"].as_u64().unwrap());
        assert_eq!(
            u64::from(info.item_damage),
            row["item_damage"].as_u64().unwrap()
        );
        assert_eq!(info.special, b(&row["special"]));
        assert_eq!(info.independent, b(&row["independent"]));
    }

    assert_eq!(rows.len(), mc_data::achievements().len());
}
