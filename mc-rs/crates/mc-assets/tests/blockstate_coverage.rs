//! **The W-ACTUALSTATE oracle.** Every `ModelResourceLocation` vanilla 1.8.9 *declares*,
//! measured against every one mc-rs can *address*.
//!
//! # Why this file exists, and why it is not in `codegen.py`
//!
//! `extractor/codegen.py::validate_blockstates` looks like this measurement and is not: it
//! keys on `mc_data::state_variant_key()` — the raw stored-metadata property string — with no
//! state mapper, so it scores 645 of 2,557 and reports the shortfall as "N blocks need a
//! custom state mapper". It also *skipped entirely* whenever the extractor was run against a
//! scratch output directory, because it derived the repo root from that directory. Both are
//! fixed there, but the real oracle cannot live there: the resolution it has to measure is
//! `mc_assets::model::state_mapper::resolve`, ~200 lines of ported `BlockModelShapes`
//! registrations. A Python re-implementation would be a *second* implementation, and an
//! oracle that agrees with itself measures nothing.
//!
//! So the oracle runs the real path, over the real registry, against the real jar.
//!
//! # What it measures
//!
//! * **declared** — `(file stem, variant key)` for every key in every
//!   `assets/minecraft/blockstates/*.json` inside `1.8.9.jar`. This is exactly the set of
//!   `ModelResourceLocation`s `ModelBakery` can look up. **2,557 across 340 files.**
//! * **addressable** — `state_mapper::resolve(s)` for every `s` in `mc_data::legal_states()`
//!   whose `render_type` is [`RenderType::Model`], which is precisely the loop
//!   `mc_assets::model::bake::bake_all` runs.
//!
//! The `RenderType::Model` filter is not an approximation: `BlockModelShapes.registerAllBlocks`
//! (`client/renderer/BlockModelShapes.java:161`) hands sixteen blocks to
//! `registerBuiltInBlocks`, whose states `BlockStateMapper.putAllStateModelLocations` skips
//! outright — and those sixteen are *exactly* the blocks whose extracted `render_type` is not
//! 3, and *none of them ships a blockstates file*. See [`BUILT_IN_BLOCKS`] and
//! [`the_built_in_blocks_are_exactly_the_non_model_render_types`], which pins that coincidence
//! so it cannot rot.
//!
//! # 2,555 or 2,557 — both, and here is which is which
//!
//! Three prior executions of this measurement disagreed. They were counting two different
//! things and both were right.
//!
//! `ModelBakery.loadVariantItemModels` (`client/resources/model/ModelBakery.java:88-97`) loads
//! `putAllStateModelLocations().values()` — the block-state locations — and then, on lines
//! 92-95, **hand-registers two more that belong to no block at all**:
//!
//! ```java
//! ResourceLocation resourcelocation = new ResourceLocation("item_frame");
//! this.registerVariant(modelblockdefinition, new ModelResourceLocation(resourcelocation, "normal"));
//! this.registerVariant(modelblockdefinition, new ModelResourceLocation(resourcelocation, "map"));
//! ```
//!
//! They are consumed by `RenderItemFrame` (`:37-38`), a *tile-entity/entity* renderer, and
//! `minecraft:item_frame` is an `ItemHangingEntity` (`Item.java:899`), never a block. So:
//!
//! | count | meaning |
//! |---|---|
//! | **2,557** | keys under `"variants"` across the 340 blockstates files — what `ModelBakery` loads |
//! | **2** | `item_frame#normal`, `item_frame#map` — hand-registered for the entity renderer |
//! | **2,555** | the block-state target: what `putAllStateModelLocations` must produce |
//!
//! **2,555 == 2,555 is the correct target for this oracle**, and the shortfall it measures
//! today is exactly the audit's **1,513**. See
//! [`the_two_item_frame_locations_belong_to_no_block`].
//!
//! # The failure mode this file is built around
//!
//! A validator that cannot find its input and passes anyway is worse than no validator,
//! because it reads as a pass. Nothing here skips. [`locate_jar`] returns `Err`, and every
//! test that needs the jar panics with the path it looked at.
//! [`the_oracle_fails_loudly_when_the_jar_is_absent`] is the negative control on that.
//!
//! # Reading a failure
//!
//! The coverage numbers are **pinned, not asserted at the target**, because the gap is the
//! known W-ACTUALSTATE defect and a permanently-red test in a shared tree teaches lanes to
//! ignore it. The pin bites in both directions: any change to `state_mapper`, to the extracted
//! state tables, or to the eventual `getActualState` port moves a number and fails here with a
//! diff. When the work lands, `ADDRESSABLE_TODAY` becomes `BLOCK_TARGET` (2,555) and the pin
//! becomes the equality.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use mc_assets::model::state_mapper;
use mc_data::{block_of, legal_states, BlockState, RenderType};

// --------------------------------------------------------------------- pins
//
// Every number below was measured by this file. Change one only with the measurement that
// justifies it in hand.

/// `(stem, key)` pairs across `assets/minecraft/blockstates/*.json` in 1.8.9.jar — every
/// `ModelResourceLocation` `ModelBakery` loads.
const DECLARED_TOTAL: usize = 2_557;

/// Files those keys are spread across.
const DECLARED_FILES: usize = 340;

/// The two `ModelBakery` hand-registers for `RenderItemFrame` rather than deriving from a
/// block state (`ModelBakery.java:92-95`). Not a defect, not addressable from a `BlockState`,
/// and the entire 2,555-vs-2,557 discrepancy.
const ENTITY_RENDERER_LOCATIONS: [(&str, &str); 2] =
    [("item_frame", "normal"), ("item_frame", "map")];

/// **The target.** `DECLARED_TOTAL` minus the item-frame pair: what
/// `BlockStateMapper.putAllStateModelLocations` must produce.
const BLOCK_TARGET: usize = DECLARED_TOTAL - ENTITY_RENDERER_LOCATIONS.len();

/// Declared locations reachable today from some legal `(id, meta)` via `state_mapper::resolve`.
///
/// The `BLOCK_TARGET - ADDRESSABLE_TODAY` = **1,513**-location shortfall is W-ACTUALSTATE:
/// vanilla keys its blockstate files by `Block.getActualState(state, world, pos)` — a value
/// recomputed per lookup from the *neighbourhood* and never stored in the 12 bits on the wire —
/// while mc-rs keys them by the stored metadata alone.
///
/// This reproduces the pass-2 audit's "1,513 of 2,555" to the unit. A later recomputation put
/// it at 1,153/2,557; that one applied a hand-rolled approximation of the property string
/// instead of the real `state_mapper::resolve`, and overstates reach by 111. Running the real
/// path is the whole reason this file exists.
const ADDRESSABLE_TODAY: usize = 1_042;

/// Blockstate files that declare more variant keys than there are packed states resolving into
/// them, and the number of keys that excess accounts for.
///
/// This is the half of W-ACTUALSTATE that porting `getActualState` alone does **not** fix.
/// `mc_assets::model::bake` stores one baked model per packed state in a 4,096-entry array
/// indexed by `state.id` (`bake.rs::model_for`), so a file reached by one packed state has one
/// slot no matter what resolves into it. `cobblestone_wall.json` declares 32 keys and is
/// reached by one state; `fence.json` declares 16 and is reached by one.
///
/// Measured **per file**, which is the unit the cache actually starves at. A per-*block*
/// count that assumes `file stem == block short name` reports 43 files / 1,102 keys, because
/// it silently skips the 38 blocks the state mapper renames — `fence` and the 22 pane/fence
/// files above are among them.
const CACHE_STARVED_FILES: usize = 66;
const CACHE_STARVED_KEYS: usize = 1_380;

/// The blocks `BlockModelShapes.registerAllBlocks` marks built-in
/// (`client/renderer/BlockModelShapes.java:161`). Their states get no `ModelResourceLocation`
/// at all, and none of them ships a blockstates file.
const BUILT_IN_BLOCKS: [&str; 16] = [
    "air",
    "flowing_water",
    "water",
    "flowing_lava",
    "lava",
    "piston_extension",
    "chest",
    "ender_chest",
    "trapped_chest",
    "standing_sign",
    "skull",
    "end_portal",
    "barrier",
    "wall_sign",
    "wall_banner",
    "standing_banner",
];

// ----------------------------------------------------------------- locating the jar

/// Where the vanilla jar should be under a `jars/` directory, or why it is not usable.
///
/// Split out from [`vanilla_jar`] so [`the_oracle_fails_loudly_when_the_jar_is_absent`] can
/// exercise the absent branch without touching process-wide environment.
fn locate_jar(jars: &Path) -> Result<PathBuf, String> {
    if !jars.is_dir() {
        return Err(format!("{} is not a directory", jars.display()));
    }
    let p = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    if !p.is_file() {
        return Err(format!("{} is not a file", p.display()));
    }
    Ok(p)
}

fn jars_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        return PathBuf::from(dir);
    }
    // crates/mc-assets -> crates -> mc-rs -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("CARGO_MANIFEST_DIR should have three ancestors")
        .join("MCP-919-main")
        .join("jars")
}

/// The vanilla jar, or a panic naming the path.
///
/// **Deliberately not a skip.** The sibling suites in this directory print `SKIP:` and pass,
/// which `cargo test` swallows for a passing test, so a machine without the jar reports a
/// clean run of tests that examined nothing. This oracle exists to measure a defect; a green
/// run that measured nothing is the exact failure it is built to rule out.
fn vanilla_jar() -> PathBuf {
    let jars = jars_dir();
    locate_jar(&jars).unwrap_or_else(|why| {
        panic!(
            "blockstate coverage oracle cannot run: {why}\n\
             looked under {}\n\
             set MC_JARS_DIR to a jars/ directory containing versions/1.8.9/1.8.9.jar.\n\
             this test does not skip on purpose -- a validator that cannot find its input and\n\
             passes anyway reads as a pass, which is worse than no validator.",
            jars.display()
        )
    })
}

// ----------------------------------------------------------------- the two sets

/// `stem -> {variant key}` for every `assets/minecraft/blockstates/*.json` in the jar.
fn declared() -> BTreeMap<String, BTreeSet<String>> {
    let jar = vanilla_jar();
    let file = std::fs::File::open(&jar).unwrap_or_else(|e| panic!("open {}: {e}", jar.display()));
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .unwrap_or_else(|e| panic!("read {} as a zip: {e}", jar.display()));

    let stems: Vec<(String, String)> = zip
        .file_names()
        .filter_map(|n| {
            let rest = n.strip_prefix("assets/minecraft/blockstates/")?;
            let stem = rest.strip_suffix(".json")?;
            (!stem.contains('/')).then(|| (n.to_string(), stem.to_string()))
        })
        .collect();

    assert!(
        !stems.is_empty(),
        "{} contains no assets/minecraft/blockstates/*.json -- wrong jar?",
        jar.display()
    );

    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (entry, stem) in stems {
        let mut buf = String::new();
        zip.by_name(&entry)
            .unwrap_or_else(|e| panic!("open {entry}: {e}"))
            .read_to_string(&mut buf)
            .unwrap_or_else(|e| panic!("read {entry}: {e}"));
        let json: serde_json::Value =
            serde_json::from_str(&buf).unwrap_or_else(|e| panic!("parse {entry}: {e}"));
        let variants = json
            .get("variants")
            .and_then(serde_json::Value::as_object)
            .unwrap_or_else(|| panic!("{entry} has no \"variants\" object"));
        out.insert(stem, variants.keys().cloned().collect());
    }
    out
}

/// Every `ModelResourceLocation` mc-rs can reach, and which states reach it.
///
/// This is `bake_all`'s pass-1 loop verbatim: legal states, `RenderType::Model` only,
/// `state_mapper::resolve`.
fn addressable() -> BTreeMap<(String, String), Vec<BlockState>> {
    let mut out: BTreeMap<(String, String), Vec<BlockState>> = BTreeMap::new();
    for state in legal_states() {
        if state.props().render_type != RenderType::Model {
            continue;
        }
        let m = state_mapper::resolve(state);
        out.entry((m.file_stem, m.variant_key))
            .or_default()
            .push(state);
    }
    out
}

fn flat(declared: &BTreeMap<String, BTreeSet<String>>) -> BTreeSet<(String, String)> {
    declared
        .iter()
        .flat_map(|(stem, keys)| keys.iter().map(move |k| (stem.clone(), k.clone())))
        .collect()
}

fn short_name(state: BlockState) -> &'static str {
    let n = block_of(state).info().name;
    n.strip_prefix("minecraft:").unwrap_or(n)
}

fn is_entity_renderer_location(stem: &str, key: &str) -> bool {
    ENTITY_RENDERER_LOCATIONS
        .iter()
        .any(|(s, k)| *s == stem && *k == key)
}

// ----------------------------------------------------------------- the tests

/// **Negative control on the skip.** The whole point of this file is that it cannot pass
/// without its input; this proves the detection actually fires rather than being assumed.
#[test]
fn the_oracle_fails_loudly_when_the_jar_is_absent() {
    let empty = tempfile::tempdir().expect("tempdir");
    let err = locate_jar(empty.path()).expect_err("an empty directory must not yield a jar");
    assert!(err.contains("1.8.9.jar"), "unhelpful message: {err}");

    let missing = empty.path().join("does-not-exist");
    let err = locate_jar(&missing).expect_err("a missing directory must not yield a jar");
    assert!(err.contains("not a directory"), "unhelpful message: {err}");

    // ...and the real one is found, so the control is not passing for the wrong reason.
    let jars = jars_dir();
    assert!(
        locate_jar(&jars).is_ok(),
        "the real jar was not found under {} -- every other test in this file will panic",
        jars.display()
    );
}

/// Pins the coincidence the `RenderType::Model` filter rests on: mc-rs skips exactly the
/// sixteen blocks vanilla marks built-in, and none of the sixteen has a blockstates file.
///
/// Without this, a future extractor change to `render_type` would silently redefine what
/// "addressable" means and the coverage number would move for a reason unrelated to the defect.
#[test]
fn the_built_in_blocks_are_exactly_the_non_model_render_types() {
    let skipped: BTreeSet<&str> = legal_states()
        .filter(|s| s.props().render_type != RenderType::Model)
        .map(short_name)
        .collect();
    let expected: BTreeSet<&str> = BUILT_IN_BLOCKS.into_iter().collect();
    assert_eq!(
        skipped, expected,
        "the set of blocks with render_type != Model no longer matches \
         BlockModelShapes.registerAllBlocks' built-in list"
    );

    let declared = declared();
    for b in BUILT_IN_BLOCKS {
        assert!(
            !declared.contains_key(b),
            "{b} is a built-in block but ships blockstates/{b}.json -- \
             its keys would be permanently unaddressable and the 2,557 target would be wrong"
        );
    }
}

/// The declared side, on its own. Independent of everything mc-rs does.
#[test]
fn vanilla_declares_two_thousand_five_hundred_and_fifty_seven_model_locations() {
    let declared = declared();
    let total: usize = declared.values().map(BTreeSet::len).sum();
    assert_eq!(
        declared.len(),
        DECLARED_FILES,
        "blockstates file count moved"
    );
    assert_eq!(
        total, DECLARED_TOTAL,
        "declared ModelResourceLocation count moved -- this is the count, from the jar, of \
         keys under \"variants\" across assets/minecraft/blockstates/*.json"
    );
}

/// **Resolves the 2,555-vs-2,557 disagreement.** `item_frame#normal` and `item_frame#map` are
/// declared in `blockstates/item_frame.json`, loaded by `ModelBakery` — and reachable from no
/// block state, because `minecraft:item_frame` is an entity
/// (`Item.java:899`, `ItemHangingEntity`), not a block. `ModelBakery.java:92-95` registers them
/// by hand, outside `putAllStateModelLocations`, for `RenderItemFrame` (`:37-38`).
///
/// So the two locations are **correctly** unaddressable from a `BlockState` and must be carved
/// out before the block-side target is stated. Everything else in the 2,557 is a block's.
#[test]
fn the_two_item_frame_locations_belong_to_no_block() {
    let declared = declared();
    let keys = declared
        .get("item_frame")
        .expect("blockstates/item_frame.json should exist in 1.8.9.jar");
    let got: BTreeSet<&str> = keys.iter().map(String::as_str).collect();
    let want: BTreeSet<&str> = ENTITY_RENDERER_LOCATIONS.iter().map(|(_, k)| *k).collect();
    assert_eq!(got, want, "item_frame.json's variant set moved");

    assert!(
        mc_data::block_by_name("item_frame").is_none()
            && mc_data::block_by_name("minecraft:item_frame").is_none(),
        "item_frame is registered as a block -- the carve-out below is then wrong"
    );

    // No legal state resolves into that file, which is what makes the carve-out exact rather
    // than an assumption: subtracting 2 removes exactly the locations nothing can reach.
    let reached = addressable();
    let into_item_frame: Vec<&(String, String)> =
        reached.keys().filter(|(s, _)| s == "item_frame").collect();
    assert!(
        into_item_frame.is_empty(),
        "some block state resolves into item_frame.json: {into_item_frame:?}"
    );

    assert_eq!(BLOCK_TARGET, 2_555, "the block-side target is 2,557 - 2");
}

/// **Nothing mc-rs addresses may be undeclared.** A resolution that names a location vanilla
/// does not ship is a state-mapper bug, not a `getActualState` gap, and shows up at runtime as
/// a missing-model block rather than a wrong-looking one. This half is expected to be *clean*.
#[test]
fn no_state_addresses_a_model_location_vanilla_does_not_declare() {
    let declared = flat(&declared());
    let reached = addressable();

    let dangling: Vec<String> = reached
        .keys()
        .filter(|k| !declared.contains(*k))
        .map(|(stem, key)| {
            let example = reached[&(stem.clone(), key.clone())][0];
            format!(
                "{stem}#{key} (e.g. {} meta {})",
                short_name(example),
                example.meta()
            )
        })
        .collect();

    assert!(
        dangling.is_empty(),
        "{} model locations are addressed but not declared:\n  {}",
        dangling.len(),
        dangling.join("\n  ")
    );
}

/// **The headline measurement.** How many of vanilla's 2,557 declared model locations can any
/// legal `(id, meta)` reach?
#[test]
fn declared_model_locations_that_no_block_state_can_address() {
    let declared_map = declared();
    let declared = flat(&declared_map);
    let reached: BTreeSet<(String, String)> = addressable().into_keys().collect();

    let covered = reached.intersection(&declared).count();
    let unreachable: BTreeSet<&(String, String)> = declared
        .difference(&reached)
        .filter(|(s, k)| !is_entity_renderer_location(s, k))
        .collect();

    // Per-file breakdown of the shortfall, worst first -- this is the work list.
    let mut by_file: BTreeMap<&str, usize> = BTreeMap::new();
    for (stem, _) in &unreachable {
        *by_file.entry(stem.as_str()).or_default() += 1;
    }
    let mut worst: Vec<(&str, usize)> = by_file.iter().map(|(s, n)| (*s, *n)).collect();
    worst.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

    eprintln!("W-ACTUALSTATE oracle");
    eprintln!(
        "  declared   {DECLARED_TOTAL} locations across {DECLARED_FILES} files \
         ({} of them RenderItemFrame's, not a block's)",
        ENTITY_RENDERER_LOCATIONS.len()
    );
    eprintln!("  target     {BLOCK_TARGET} addressable from a BlockState");
    eprintln!("  addressed  {covered}");
    eprintln!(
        "  shortfall  {} across {} files",
        unreachable.len(),
        by_file.len()
    );
    for (stem, n) in worst.iter().take(20) {
        eprintln!("    {n:>4}  {stem}");
    }

    assert_eq!(
        covered + unreachable.len(),
        BLOCK_TARGET,
        "arithmetic: covered + unreachable must be the block-side target"
    );
    assert_eq!(
        covered, ADDRESSABLE_TODAY,
        "\nblockstate coverage moved: {covered} of {BLOCK_TARGET} addressable, \
         pin says {ADDRESSABLE_TODAY}.\n\
         If this went UP, W-ACTUALSTATE progressed -- move the pin (target {BLOCK_TARGET}).\n\
         If it went DOWN, something regressed in state_mapper or the extracted state tables."
    );
}

/// **The half that porting `getActualState` alone cannot fix.**
///
/// `bake_all` stores one baked model per packed state in a 4,096-entry array indexed by
/// `state.id` (`bake.rs::model_for`). For 43 blocks the blockstate file declares more variant
/// keys than the block has legal metadata values, so those keys have nowhere to be *held* even
/// if something recomputed them correctly. `cobblestone_wall` is the extreme: 32 keys, 2 metas.
///
/// This is why the two defects are symmetrically blocking, and why the fix cannot be
/// "port `getActualState`" alone.
#[test]
fn the_twelve_bit_model_cache_has_fewer_slots_than_vanilla_declares_variants() {
    let declared = declared();
    let reached = addressable();

    // stem -> how many distinct legal states resolve into that file at all.
    let mut slots: BTreeMap<&str, usize> = BTreeMap::new();
    for ((stem, _), states) in &reached {
        *slots.entry(stem.as_str()).or_default() += states.len();
    }

    let mut starved: Vec<(&str, usize, usize)> = Vec::new();
    for (stem, keys) in &declared {
        // item_frame.json is RenderItemFrame's, not starved -- see the module docs.
        if stem == "item_frame" {
            continue;
        }
        let have = slots.get(stem.as_str()).copied().unwrap_or(0);
        if keys.len() > have {
            starved.push((stem.as_str(), keys.len(), have));
        }
    }
    starved.sort_by(|a, b| (b.1 - b.2).cmp(&(a.1 - a.2)).then(a.0.cmp(b.0)));
    let excess: usize = starved.iter().map(|(_, k, h)| k - h).sum();

    eprintln!("12-bit model cache starvation");
    eprintln!(
        "  {} files declare {excess} more variant keys than there are packed states \
         resolving into them",
        starved.len()
    );
    for (stem, keys, have) in starved.iter().take(20) {
        eprintln!("    {stem}: {keys} keys, {have} state slots");
    }

    assert_eq!(
        starved.len(),
        CACHE_STARVED_FILES,
        "count of cache-starved blockstate files moved"
    );
    assert_eq!(
        excess, CACHE_STARVED_KEYS,
        "count of variant keys with no 12-bit slot to live in moved"
    );

    // The decisive pair, spelled out: porting getActualState perfectly would still leave
    // 31 of cobblestone_wall's 32 models unnameable, because it has one slot.
    let wall = starved
        .iter()
        .find(|(s, _, _)| *s == "cobblestone_wall")
        .expect("cobblestone_wall.json must be starved");
    assert_eq!((wall.1, wall.2), (32, 1));
    let fence = starved
        .iter()
        .find(|(s, _, _)| *s == "fence")
        .expect("fence.json must be starved");
    assert_eq!((fence.1, fence.2), (16, 1));
}
