//! Exhaustive golden comparison of the generated StatList tables against the live extractor dump.

use mc_data::stat::{
    all_stats, block_stats, general_stats, item_stats, mob_stats, StatFormat, StatKind,
};
use serde_json::Value;

const STATS: &str = include_str!("../extractor/out/stats.json");

fn s(value: &Value) -> &str {
    value.as_str().expect("string")
}

fn optional_s(value: &Value) -> Option<&str> {
    value.as_str()
}

#[test]
fn every_generated_stat_and_row_association_matches_the_live_registry_dump() {
    let doc: Value = serde_json::from_str(STATS).expect("stats.json parses");
    assert_eq!(doc["format"], "mc-data/stats/1");

    let extracted = doc["all"].as_array().expect("all stats");
    assert_eq!(all_stats().len(), extracted.len());
    for (actual, expected) in all_stats().iter().zip(extracted) {
        assert_eq!(actual.id, s(&expected["id"]));
        assert_eq!(actual.translation_key, s(&expected["translation_key"]));
        assert_eq!(actual.display_en_us, s(&expected["display_en_us"]));
        assert_eq!(
            actual.independent,
            expected["independent"].as_bool().unwrap()
        );
        assert_eq!(
            actual.item.map(|item| u64::from(item.0)),
            expected["item_id"].as_u64(),
            "{} item association",
            actual.id
        );
        assert_eq!(
            actual.format,
            match s(&expected["formatter"]) {
                "simple" => StatFormat::Simple,
                "time" => StatFormat::Time,
                "distance" => StatFormat::Distance,
                "damage" => StatFormat::Damage,
                other => panic!("unknown formatter {other}"),
            }
        );
        assert_eq!(
            actual.kind,
            match s(&expected["kind"]) {
                "general" => StatKind::General,
                "mine_block" => StatKind::MineBlock,
                "craft_item" => StatKind::CraftItem,
                "use_item" => StatKind::UseItem,
                "break_item" => StatKind::BreakItem,
                "kill_entity" => StatKind::KillEntity,
                "entity_killed_by" => StatKind::EntityKilledBy,
                "achievement" => StatKind::Achievement,
                other => panic!("unknown stat kind {other}"),
            }
        );
    }

    let extracted_general = doc["general"].as_array().expect("general stats");
    let actual_general = general_stats().collect::<Vec<_>>();
    assert_eq!(actual_general.len(), extracted_general.len());
    for (actual, expected) in actual_general.iter().zip(extracted_general) {
        assert_eq!(actual.id, s(expected));
    }

    let extracted_blocks = doc["blocks"].as_array().expect("block rows");
    assert_eq!(block_stats().len(), extracted_blocks.len());
    for (actual, expected) in block_stats().iter().zip(extracted_blocks) {
        assert_eq!(
            u64::from(actual.block.0),
            expected["block_id"].as_u64().unwrap()
        );
        assert_eq!(
            u64::from(actual.item.0),
            expected["item_id"].as_u64().unwrap()
        );
        assert_eq!(actual.mined, s(&expected["mined"]));
        assert_eq!(actual.crafted, optional_s(&expected["crafted"]));
        assert_eq!(actual.used, optional_s(&expected["used"]));
    }

    let extracted_items = doc["items"].as_array().expect("item rows");
    assert_eq!(item_stats().len(), extracted_items.len());
    for (actual, expected) in item_stats().iter().zip(extracted_items) {
        assert_eq!(
            u64::from(actual.item.0),
            expected["item_id"].as_u64().unwrap()
        );
        assert_eq!(actual.depleted, optional_s(&expected["depleted"]));
        assert_eq!(actual.crafted, optional_s(&expected["crafted"]));
        assert_eq!(actual.used, s(&expected["used"]));
    }

    let extracted_mobs = doc["mobs"].as_array().expect("mob rows");
    assert_eq!(mob_stats().len(), extracted_mobs.len());
    for (actual, expected) in mob_stats().iter().zip(extracted_mobs) {
        assert_eq!(
            u64::from(actual.entity_id),
            expected["entity_id"].as_u64().unwrap()
        );
        assert_eq!(actual.entity_name, s(&expected["entity_name"]));
        assert_eq!(actual.killed, s(&expected["killed"]));
        assert_eq!(actual.killed_by, s(&expected["killed_by"]));
    }
}
