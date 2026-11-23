//! Differential test: the terrain mesher's `Block.getActualState` port and the
//! `BlockModelShapes` state mapper, against the real 1.8.9 client bytecode.
//!
//! `tests/actual_state_oracle/ActualStateOracle.java` runs vanilla's own `getActualState` for
//! every block class that overrides it (found by reflection) over seeded random 3x3x3
//! neighbourhoods, and records per case the actual property tuple, the `ModelResourceLocation`
//! vanilla's state mapper gives it, and the bounds `setBlockBoundsBasedOnState` leaves (what the
//! selection outline draws). This test rebuilds each neighbourhood in an `mc_world::World` and
//! requires mc-rs's `actual_state`, `state_mapper::resolve_actual` and `selected_bounding_box`
//! to reproduce all three.
//!
//! The case file is ~5 MB, so it is generated rather than committed:
//!
//! ```text
//! bash crates/mc-render/tests/actual_state_oracle/run_oracle.sh <cases.tsv> [cases-per-meta]
//! MC_ACTUAL_STATE_ORACLE=<cases.tsv> cargo test -p mc-render --test actual_state_java_oracle -- --ignored
//! ```
//!
//! The test is `#[ignore]`d because it needs a JDK and the recompiled vanilla classes; run with
//! `--ignored` and without the variable, it fails loudly rather than passing vacuously.

use mc_assets::model::state_mapper::resolve_actual;
use mc_data::{BlockId, BlockState};
use mc_math::BlockPos;
use mc_render::terrain::{fixture, mesher::actual_state};

fn parse_state(text: &str) -> BlockState {
    let (id, meta) = text
        .split_once(':')
        .unwrap_or_else(|| panic!("bad state {text:?}"));
    BlockState::new(
        BlockId(id.parse().expect("block id")),
        meta.parse().expect("metadata"),
    )
}

#[test]
#[ignore = "needs MC_ACTUAL_STATE_ORACLE from tests/actual_state_oracle/run_oracle.sh"]
fn actual_states_and_model_locations_match_vanilla_bytecode() {
    let path = std::env::var_os("MC_ACTUAL_STATE_ORACLE").expect(
        "set MC_ACTUAL_STATE_ORACLE to the file tests/actual_state_oracle/run_oracle.sh wrote",
    );
    let text = std::fs::read_to_string(&path).expect("read the oracle case file");

    let mut world = fixture::blank_world(0);
    let centre = BlockPos::new(8, 64, 8);
    let mut cases = 0usize;
    let mut mismatches = Vec::new();
    let mut per_block: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut bounds_mismatches = Vec::new();
    let mut bounds_per_block: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for line in text.lines() {
        let columns: Vec<&str> = line.split('\t').collect();
        assert_eq!(columns.len(), 5, "malformed oracle line {line:?}");
        let target = parse_state(columns[0]);
        let mut neighbours = columns[1].split(',').map(parse_state);
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if (dx, dy, dz) == (0, 0, 0) {
                        continue;
                    }
                    let state = neighbours.next().expect("26 neighbours");
                    world.set_block(centre.add(dx, dy, dz), state);
                }
            }
        }
        assert!(neighbours.next().is_none(), "more than 26 neighbours");
        world.set_block(centre, target);
        assert_eq!(world.block_at(centre), target);

        let actual = actual_state(&world, target, centre);
        let tuple = actual
            .properties()
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(",");
        let tuple = if tuple.is_empty() {
            "-".to_owned()
        } else {
            tuple
        };
        let mapping = resolve_actual(&actual);
        let location = format!("minecraft:{}#{}", mapping.file_stem, mapping.variant_key);
        cases += 1;
        let name = target.block().info().name.to_owned();
        if tuple != columns[2] || location != columns[3] {
            *per_block.entry(name.clone()).or_default() += 1;
            if mismatches.len() < 40 {
                mismatches.push(format!("{line}\n    mc-rs: {tuple}\t{location}"));
            }
        }

        // The live selection outline (`SelectionBox::for_block_in_world`) before its 0.002
        // expansion, block-local, against the bounds vanilla's setBlockBoundsBasedOnState left.
        let java: Vec<f64> = columns[4]
            .split(',')
            .map(|v| v.parse().expect("bound"))
            .collect();
        let aabb = mc_render::terrain::selected_bounding_box(&world, centre, target);
        let origin = [
            f64::from(centre.x),
            f64::from(centre.y),
            f64::from(centre.z),
        ];
        let ours = [
            aabb.min.x - origin[0],
            aabb.min.y - origin[1],
            aabb.min.z - origin[2],
            aabb.max.x - origin[0],
            aabb.max.y - origin[1],
            aabb.max.z - origin[2],
        ];
        if java.iter().zip(ours).any(|(j, o)| (j - o).abs() > 1e-6) {
            *bounds_per_block.entry(name).or_default() += 1;
            if bounds_mismatches.len() < 40 {
                bounds_mismatches.push(format!("{line}\n    mc-rs bounds: {ours:?}"));
            }
        }
    }
    assert!(cases > 1_000, "only {cases} oracle cases");
    assert!(
        per_block.is_empty(),
        "{} of {cases} cases differ from vanilla, by block {per_block:?}:\n{}",
        per_block.values().sum::<usize>(),
        mismatches.join("\n")
    );
    assert!(
        bounds_per_block.is_empty(),
        "{} of {cases} selection bounds differ from vanilla, by block {bounds_per_block:?}:\n{}",
        bounds_per_block.values().sum::<usize>(),
        bounds_mismatches.join("\n")
    );
    eprintln!("{cases} actual-state cases and selection bounds match vanilla 1.8.9");
}
