//! The protocol-5 (1.7.10) tables checked against dumps of the real 1.7.10 and 1.8.9 classes.
//!
//! The dumps (block geometry in 15,746 world scenarios per version, the 1.8.9 state table,
//! every item, and the name tables) are Mojang-derived and never part of this repository. Set
//! `MC_V5_JAVA_DUMPS_DIR` to the directory that holds them (`java/out/*.tsv` and
//! `names/*.tsv`) to run these tests; without it every test returns early and says so.
//!
//! The block test is the central one: every 1.7.10 cell of every scenario is decoded with
//! [`state_from_v5`], and the 1.8.9 geometry of the decoded world (collision boxes, stale-bounds
//! collision, ray-trace bounds, selection box, 18 rays) must equal 1.7.10's, except for the
//! differences [`expected_deltas`] lists with the 1.7.10 code behind each: the movement and
//! interaction code implements them, or they only occur at metas no vanilla code writes.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

use mc_data::v5::{
    enchantment_v5, particle_from_v5_name, potion_v5, sound_event_v5, sound_files_v5, stat_from_v5,
    state_from_v5, try_item_props_v5, ParticleV5,
};
use mc_data::{sound_event, stat_by_id, BlockId, BlockState, ItemId};

/// The dump directory, or `None` (the test then does nothing).
fn dumps_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("MC_V5_JAVA_DUMPS_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("MC_V5_JAVA_DUMPS_DIR is not set: skipping the 1.7.10 oracle comparison");
    }
    dir
}

fn read_rows(dir: &std::path::Path, file: &str) -> Vec<Vec<String>> {
    let path = dir.join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .map(|line| line.split('\t').map(str::to_owned).collect())
        .collect()
}

fn decoded(block: u16, meta: u8) -> BlockState {
    state_from_v5((block << 4) | u16::from(meta))
}

/// The 1.8.9 state table of the dump (`S` records) against mc-data's own: legality of every
/// packed value and the `getStateFromMeta` image. The decode table is built from mc-data's tables,
/// so this anchors it to the independent dump.
#[test]
fn mc_data_state_table_matches_the_1_8_9_dump() {
    let Some(dir) = dumps_dir() else { return };
    let mut rows = 0;
    for row in read_rows(&dir, "java/out/blocks18.tsv") {
        if row[0] != "S" {
            continue;
        }
        let block: u16 = row[1].parse().unwrap();
        let meta: u8 = row[2].parse().unwrap();
        let state = BlockState::new(BlockId(block), meta);
        assert_eq!(state.is_legal(), row[3] == "1", "{block}:{meta} legality");
        let java = BlockState::try_from_java_state_id(i32::from(block) | (i32::from(meta) << 12));
        match row[4].as_str() {
            "-" | "EXC" => assert!(java.is_none(), "{block}:{meta} throws in Java"),
            canonical => {
                let canonical: u8 = canonical.parse().unwrap();
                assert_eq!(
                    java,
                    Some(BlockState::new(BlockId(block), canonical)),
                    "{block}:{meta}"
                );
            }
        }
        rows += 1;
    }
    assert_eq!(rows, 198 * 16, "every 1.8.9 block, every meta");
}

/// `(block, field)` -> the 1.7.10 metas whose geometry differs from the decoded 1.8.9 state,
/// with the 1.7.10 code that explains each.
fn expected_deltas() -> BTreeMap<(u16, &'static str), Vec<u8>> {
    let all: Vec<u8> = (0..16).collect();
    let non_vanilla_ladder = vec![0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let button = vec![0, 5, 6, 7, 8, 13, 14, 15];
    let portal = vec![0, 3, 4, 7, 8, 11, 12, 15];
    let mut deltas = BTreeMap::new();
    let mut put = |block: u16, fields: &[&'static str], metas: &[u8]| {
        for &field in fields {
            deltas.insert((block, field), metas.to_vec());
        }
    };
    let ray_fields = ["bounds", "selected", "rays"];
    let all_fields = ["collision", "poisoned", "bounds", "selected", "rays"];
    // Ascending powered, detector and activator rails trace a flat box in 1.7.10: java17
    // `BlockRailBase.setBlockBoundsBasedOnState` (block/BlockRailBase.java:60-72) tests the raw
    // meta for 2-5, so the powered bit makes 10-13 flat; java18 reads the shape property. The
    // decode keeps metas 10-13, so the interaction code can apply the raw-meta rule.
    for rail in [27, 28, 157] {
        put(rail, &ray_fields, &[10, 11, 12, 13]);
    }
    // A moving piston can be targeted in 1.7.10: java17 `BlockPistonMoving` has no
    // `collisionRayTrace` override (its bounds come from the moving tile entity,
    // block/BlockPistonMoving.java:136); java18 `BlockPistonMoving.collisionRayTrace` returns null.
    put(36, &["rays"], &all);
    // Ladders and wall signs at metas no vanilla code writes: java17 `BlockLadder.func_149797_b`
    // (block/BlockLadder.java:37) sets bounds only for metas 2-5 and otherwise keeps whatever the
    // shared block held; java17 `BlockSign.setBlockBoundsBasedOnState` (block/BlockSign.java:47)
    // leaves a wall sign outside 2-5 a full cube.
    put(65, &all_fields, &non_vanilla_ladder);
    put(68, &ray_fields, &non_vanilla_ladder);
    // Buttons at metas 0, 5-8, 13-15: java17 `BlockButton.setBlockBoundsBasedOnState`
    // (block/BlockButton.java:142) sets bounds only for facings 1-4 and keeps stale ones
    // otherwise; java18 has floor and ceiling buttons there.
    put(77, &ray_fields, &button);
    put(143, &ray_fields, &button);
    // The snow layer collides with the shared block bounds in both versions (java17
    // `BlockSnow.getCollisionBoundingBoxFromPool`, block/BlockSnow.java:36); with those bounds
    // poisoned before the call the two differ at these metas only.
    put(78, &["poisoned"], &[0, 1, 8, 9]);
    // Portals whose axis bits are 0 or 3: java17 `BlockPortal.setBlockBoundsBasedOnState`
    // (block/BlockPortal.java:56-90) derives axis 0 from the neighbours and makes 3 thin on both
    // axes; java18 decodes both to the X axis.
    put(90, &ray_fields, &portal);
    // Cake: java17 `BlockCake.getCollisionBoundingBoxFromPool` (block/BlockCake.java:43-50) tops
    // the box at 0.5 - 1/16 for every meta, and from meta 7 on its x range is empty or inverted.
    put(92, &["collision", "poisoned"], &all);
    put(92, &ray_fields, &[7, 8, 9, 10, 11, 12, 13, 14, 15]);
    // Stems at 8-15: java17 `BlockStem.setBlockBoundsBasedOnState` (block/BlockStem.java:190-195)
    // traces up to (2 * meta + 2) / 16; age 7 is the tallest 1.8.9 box.
    for stem in [104, 105] {
        put(stem, &ray_fields, &[8, 9, 10, 11, 12, 13, 14, 15]);
    }
    // The outline of a vine with no faces: java17 `BlockVine.setBlockBoundsBasedOnState`
    // (block/BlockVine.java:49) leaves it empty; java18 draws a full cube.
    put(106, &["selected"], &[0]);
    // A closed fence gate with bit 3: java17 `BlockFenceGate.getCollisionBoundingBoxFromPool`
    // (block/BlockFenceGate.java:36-40) picks the axis from the raw meta (`meta != 2 && meta !=
    // 0`); java18 from the facing. The decode keeps the meta, so the movement code can apply it.
    put(107, &["collision", "poisoned"], &[8, 10]);
    // Cocoa at 12-15: java17 `BlockCocoa` sizes the box for age `(meta & 12) >> 2` = 3
    // (block/BlockCocoa.java:88-107); 1.8.9's largest is age 2.
    put(127, &all_fields, &[12, 13, 14, 15]);
    // Carpet: java17 `BlockCarpet.getCollisionBoundingBoxFromPool` (block/BlockCarpet.java:33-38)
    // returns a box of height 0; java18 collides with the 1/16 bounds.
    put(171, &["collision", "poisoned"], &all);
    deltas
}

#[test]
fn decoded_geometry_matches_1_7_10_except_the_listed_deltas() {
    let Some(dir) = dumps_dir() else { return };
    // The meta the 1.8.9 dump's world holds for a raw meta: `getStateFromMeta`'s choice, or the
    // default state where it throws (`None`).
    let mut naive: HashMap<(u16, u8), Option<u8>> = HashMap::new();
    type Records = HashMap<(String, u16, u8), Vec<String>>;
    let load = |file: &str, naive: Option<&mut HashMap<(u16, u8), Option<u8>>>| {
        let mut c: Records = HashMap::new();
        let mut r: Records = HashMap::new();
        let mut naive = naive;
        for row in read_rows(&dir, file) {
            match row[0].as_str() {
                "C" | "R" => {
                    let key = (
                        row[1].clone(),
                        row[2].parse().unwrap(),
                        row[3].parse().unwrap(),
                    );
                    let map = if row[0] == "C" { &mut c } else { &mut r };
                    map.insert(key, row[4..].to_vec());
                }
                "S" => {
                    if let Some(naive) = naive.as_deref_mut() {
                        let key = (row[1].parse().unwrap(), row[2].parse().unwrap());
                        naive.insert(key, row[4].parse().ok());
                    }
                }
                _ => {}
            }
        }
        (c, r)
    };
    let (c17, r17) = load("java/out/blocks17.tsv", None);
    let (c18, r18) = load("java/out/blocks18.tsv", Some(&mut naive));
    assert_eq!(c17.len(), 15_746);

    // The single-block scenarios repeat the target's own (id, meta) around it; the others hold
    // fixed neighbours whose decode equals the dump's (checked below).
    let per_cell = ["iso", "east_same", "ns_same", "stone_wsdu", "vine_up"];
    // The 1.8.9 record standing for the decoded 1.7.10 world: the same record when the dump
    // decoded the cell to the same state (it then also has the same run history, which the
    // stair bounds depend on), else the record of the decoded meta.
    let target = |scenario: &str, block: u16, meta: u8| -> u8 {
        let decoded = decoded(block, meta).meta();
        if !per_cell.contains(&scenario) || naive[&(block, meta)] == Some(decoded) {
            meta
        } else {
            decoded
        }
    };
    let fields = ["collision", "poisoned", "bounds", "selected"];
    let mut seen: BTreeMap<(u16, &'static str), BTreeSet<u8>> = BTreeMap::new();
    let mut cells: BTreeSet<(u16, u8)> = BTreeSet::new();
    for ((scenario, block, meta), v17) in &c17 {
        let v18 = &c18[&(scenario.clone(), *block, target(scenario, *block, *meta))];
        for (k, &field) in fields.iter().enumerate() {
            if v17[k] != v18[k] {
                seen.entry((*block, field)).or_default().insert(*meta);
            }
        }
        if per_cell.contains(&scenario.as_str()) {
            cells.insert((*block, *meta));
        }
    }
    for ((scenario, block, meta), v17) in &r17 {
        if v17 != &r18[&(scenario.clone(), *block, target(scenario, *block, *meta))] {
            seen.entry((*block, "rays")).or_default().insert(*meta);
        }
    }
    // Every decode-table entry of every 1.7.10 block was exercised.
    assert_eq!(cells.len(), 171 * 16);

    let expected: BTreeMap<(u16, &str), BTreeSet<u8>> = expected_deltas()
        .into_iter()
        .map(|(key, metas)| (key, metas.into_iter().collect()))
        .collect();
    let seen_flat: Vec<_> = seen.iter().collect();
    let expected_flat: Vec<_> = expected.iter().collect();
    // Both directions: no unlisted difference, and no listed one that is not there.
    assert_eq!(seen_flat, expected_flat);

    // The scenarios with fixed neighbours rely on the decode agreeing with the dump's for them:
    // meta 0 of every block (fence/pane/wall neighbours), stairs, doors, chests.
    for block in (0..=164).chain(170..=175) {
        assert_eq!(
            naive[&(block, 0)],
            Some(decoded(block, 0).meta()),
            "{block}:0"
        );
    }
    for block in [53, 64, 71, 54, 146] {
        for meta in 0..16 {
            assert_eq!(
                naive[&(block, meta)],
                Some(decoded(block, meta).meta()),
                "{block}"
            );
        }
    }
}

/// The negative control of the geometry test: with the raw decode a 1.8.9 lookup would apply
/// (`id << 4 | meta` as is), cells vanish into air.
#[test]
fn the_raw_decode_loses_blocks_the_table_keeps() {
    let Some(dir) = dumps_dir() else { return };
    let mut lost = 0;
    for row in read_rows(&dir, "java/out/blocks18.tsv") {
        if row[0] != "S" {
            continue;
        }
        let block: u16 = row[1].parse().unwrap();
        let meta: u8 = row[2].parse().unwrap();
        if block > 175 || (165..=169).contains(&block) || block == 0 {
            continue;
        }
        if row[3] == "0" {
            // Not in BLOCK_STATE_IDS: a raw 1.8.9 lookup reads air; the table keeps the block.
            assert!(BlockState::new(BlockId(block), meta).props().is_air());
            assert_eq!(decoded(block, meta).block(), BlockId(block));
            lost += 1;
        }
    }
    // The metas 1.8.9's state table does not register (java18: `Block.registerBlocks` maps only
    // `getMetaFromState` of each valid state), spread over 139 of the 171 ids.
    assert_eq!(lost, 1_545);
}

#[test]
fn items_match_the_1_7_10_dump() {
    let Some(dir) = dumps_dir() else { return };
    let mut ids = BTreeSet::new();
    for row in read_rows(&dir, "java/out/items17.tsv") {
        match row[0].as_str() {
            "I" => {
                let id: u16 = row[1].parse().unwrap();
                ids.insert(id);
                let props = try_item_props_v5(ItemId(id)).unwrap_or_else(|| panic!("item {id}"));
                assert_eq!(props.name, row[2], "{id}");
                assert_eq!(
                    props.java_class.rsplit('.').next(),
                    Some(row[3].as_str()),
                    "{id}"
                );
                assert_eq!(props.max_stack_size.to_string(), row[4], "{id} stack");
                assert_eq!(props.max_damage.to_string(), row[5], "{id} damage");
                assert_eq!(props.has_subtypes, row[6] == "1", "{id} subtypes");
                assert_eq!(props.full_3d, row[7] == "1", "{id} full 3d");
                assert_eq!(
                    props.enchantability.to_string(),
                    row[10],
                    "{id} enchantability"
                );
                assert_eq!(props.damageable, row[11] == "1", "{id} damageable");
            }
            "F" => {
                let id: u16 = row[1].parse().unwrap();
                let meta: i16 = row[2].parse().unwrap();
                let food = try_item_props_v5(ItemId(id)).unwrap().food_stats(meta);
                assert_eq!(food.heal.to_string(), row[3], "{id}:{meta} heal");
                assert!((f64::from(food.saturation) - row[4].parse::<f64>().unwrap()).abs() < 1e-6);
            }
            _ => {}
        }
    }
    assert_eq!(ids.len(), 315);
    for id in 0..=2300 {
        assert_eq!(
            try_item_props_v5(ItemId(id)).is_some(),
            ids.contains(&id),
            "{id}"
        );
    }
}

#[test]
fn particle_names_match_the_1_7_10_table() {
    let Some(dir) = dumps_dir() else { return };
    let mut plain = 0;
    for row in read_rows(&dir, "names/particles.tsv").into_iter().skip(1) {
        let (name, bypass, id) = (row[0].as_str(), row[2].as_str(), row[4].as_str());
        if name.contains('<') || name.starts_with('(') {
            continue; // the prefixed families and the 1.8.9-only rows
        }
        let ParticleV5::Spawn(p) = particle_from_v5_name(name) else {
            panic!("{name}");
        };
        assert_eq!(p.id.to_string(), id, "{name}");
        assert_eq!(p.ignore_range, bypass == "yes", "{name}");
        plain += 1;
    }
    assert_eq!(plain, 35);
}

#[test]
fn sound_events_match_the_1_7_10_index() {
    let Some(dir) = dumps_dir() else { return };
    let mut events = BTreeSet::new();
    for row in read_rows(&dir, "names/sounds_events_1_7_10.tsv")
        .into_iter()
        .skip(1)
    {
        let name = row[0].as_str();
        let files: Vec<&str> = row[4].split(' ').collect();
        let event = sound_event_v5(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(event.category, row[1], "{name}");
        match sound_files_v5(name) {
            Some(own) => assert_eq!(own, &files[..], "{name}"),
            None => assert_eq!(event.variants, files.len(), "{name}"),
        }
        events.insert(name.to_owned());
    }
    let known: BTreeSet<String> = mc_data::sound_events()
        .iter()
        .filter(|e| sound_event_v5(e.name).is_some())
        .map(|e| e.name.to_owned())
        .collect();
    assert_eq!(known, events);
    assert!(sound_event("mob.rabbit.hop").is_some());
}

#[test]
fn statistic_ids_match_the_name_table() {
    let Some(dir) = dumps_dir() else { return };
    // Ids whose slot 1.8.9 merges with another block's (java18: replaceAllSimilarBlocks).
    let merged = [
        8, 9, 10, 11, 91, 86, 62, 61, 74, 73, 94, 93, 150, 149, 76, 75, 124, 123,
    ];
    let merged: BTreeSet<u16> = merged
        .into_iter()
        .chain([43, 44, 125, 126, 2, 3, 60])
        .collect();
    let mut resolved = 0;
    for row in read_rows(&dir, "names/stat_names.tsv").into_iter().skip(1) {
        let (id, v17, v18) = (
            row[1].parse::<u16>().unwrap(),
            row[4].as_str(),
            row[5].as_str(),
        );
        if v17 == "-" {
            continue; // 1.8.9 only
        }
        let got = stat_from_v5(v17);
        if merged.contains(&id) {
            // Resolves to the 1.8.9 counter of the slot, which may be the partner's.
            continue;
        }
        // The name table pairs ids by registry name; an item id below 256 takes its block's
        // name there, which can be another item's (`bed` is block 26 but item 355). Only a
        // 1.8.9 statistic of this very id is the expected answer.
        match stat_by_id(v18) {
            Some(expected) if expected.item == Some(ItemId(id)) => {
                assert_eq!(got.map(|s| s.id), Some(v18), "{v17}");
                resolved += 1;
            }
            _ => assert!(got.is_none(), "{v17}: 1.8.9 has no statistic for id {id}"),
        }
    }
    assert_eq!(resolved, 647);
}

#[test]
fn potions_and_enchantments_match_the_id_tables() {
    let Some(dir) = dumps_dir() else { return };
    for row in read_rows(&dir, "names/potions.tsv").into_iter().skip(1) {
        let id: u8 = row[0].parse().unwrap();
        let colour: u32 = row[5].parse().unwrap();
        assert_eq!(
            potion_v5(id).map(|p| p.liquid_color),
            Some(colour),
            "potion {id}"
        );
    }
    for row in read_rows(&dir, "names/enchantments.tsv")
        .into_iter()
        .skip(1)
    {
        let id: u8 = row[0].parse().unwrap();
        assert_eq!(
            enchantment_v5(id).is_some(),
            row[8] == "yes",
            "enchantment {id}"
        );
    }
}
