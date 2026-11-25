//! The protocol-5 (1.7.10) entity tables checked against dumps of the real 1.7.10 and 1.8.9
//! classes: every `EntityList` class built against a client world, with its size, `yOffset`, eye
//! height and `DataWatcher` registrations.
//!
//! The dumps are Mojang-derived and never part of this repository. Set `MC_V5_JAVA_DUMPS_DIR` to
//! the directory that holds them (`java/out/ents17.tsv`, `ents18.tsv`; the same variable as
//! mc-data's `v5_java_dump_oracle` test) to run these tests; without it they return early and say
//! so.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use mc_data::{entity_type, EntityType};
use mc_protocol::{ItemStack, MetadataEntry, MetadataValue};
use mc_world::entity::EntityKind;
use mc_world::v5::{
    exists_in_v5, geometry_v5, mob_type_v5, registered_type_tag_v5, remap_metadata_v5,
};
use mc_world::Metadata;

/// The dump directory, or `None` (the test then does nothing).
fn dumps_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("MC_V5_JAVA_DUMPS_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("MC_V5_JAVA_DUMPS_DIR is not set: skipping the 1.7.10 entity oracle comparison");
    }
    dir
}

/// One `E` record: class name -> (width, height, yOffset, eye height, watcher entries).
struct Class {
    width: f64,
    height: f64,
    /// `-` in the 1.8.9 dump, which has no `yOffset` field.
    y_offset: Option<f64>,
    eye: f64,
    watchers: Vec<(u8, u8, String)>,
}

struct Dump {
    /// `EntityList` id -> class name.
    ids: BTreeMap<u8, String>,
    classes: BTreeMap<String, Class>,
}

fn load(dir: &std::path::Path, file: &str) -> Dump {
    let path = dir.join("java/out").join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut ids = BTreeMap::new();
    let mut classes = BTreeMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "L" => {
                ids.insert(f[1].parse().unwrap(), f[3].to_owned());
            }
            "E" if f.len() >= 8 => {
                let watchers = f[7]
                    .split(';')
                    .map(|w| {
                        // `NN:tT:Class=value`
                        let mut parts = w.splitn(3, ':');
                        let index = parts.next().unwrap().parse().unwrap();
                        let tag = parts.next().unwrap()[1..].parse().unwrap();
                        let value = parts.next().unwrap().split_once('=').unwrap().1.to_owned();
                        (index, tag, value)
                    })
                    .collect();
                let num = |s: &str| s.parse::<f64>().unwrap();
                classes.insert(
                    f[1].to_owned(),
                    Class {
                        width: num(f[2]),
                        height: num(f[3]),
                        y_offset: f[4].parse().ok(),
                        eye: num(f[5]),
                        watchers,
                    },
                );
            }
            _ => {}
        }
    }
    Dump { ids, classes }
}

/// The client kind of a dumped class: by its `EntityList` id (the 57 ids 1.7.10 registers name the
/// same classes in java17 and java18 `EntityList`), or the fishing hook.
fn kinds(dump: &Dump) -> BTreeMap<String, EntityKind> {
    let mut out: BTreeMap<String, EntityKind> = dump
        .ids
        .iter()
        .map(|(&id, class)| (class.clone(), EntityKind::Type(entity_type(id).unwrap())))
        .collect();
    out.insert("EntityFishHook".into(), EntityKind::FishingBobber);
    out
}

fn close(a: f32, b: f64) -> bool {
    // The dump prints `Math.round(x * 1e6) / 1e6` of the float.
    (f64::from(a) - b).abs() < 1e-6
}

/// The 1.7.10 dump's `yOffset` column.
fn y_offset(c: &Class) -> f64 {
    c.y_offset.expect("the 1.7.10 dump records yOffset")
}

#[test]
fn mob_ids_are_the_dumped_entity_list() {
    let Some(dir) = dumps_dir() else { return };
    let dump = load(&dir, "ents17.tsv");
    let dumped: BTreeSet<u8> = dump.ids.keys().copied().collect();
    let ours: BTreeSet<u8> = (0..=u8::MAX)
        .filter(|&id| mob_type_v5(id).is_some())
        .collect();
    assert_eq!(ours, dumped);
    // Every 1.8.9 class the 1.7.10 dump lacks is a kind 1.7.10 does not have (the thrown egg is
    // the exception: no EntityList id, but object type 62 spawns it).
    let v18 = load(&dir, "ents18.tsv");
    for (&id, class) in &v18.ids {
        let t = entity_type(id).unwrap();
        let expected = dump.ids.contains_key(&id) || t == EntityType::ThrownEgg;
        assert_eq!(exists_in_v5(EntityKind::Type(t)), expected, "{class}");
    }
}

#[test]
fn geometry_matches_the_dump() {
    let Some(dir) = dumps_dir() else { return };
    let dump = load(&dir, "ents17.tsv");
    let mut checked = 0;
    for (class, kind) in kinds(&dump) {
        let Some(c) = dump.classes.get(&class) else {
            continue; // abstract `Mob`/`Monster`
        };
        let g = geometry_v5(kind).unwrap_or_else(|| panic!("{class}"));
        match kind {
            // The dump builds `(World)` constructors; the client spawns these two with others
            // (EntityXPOrb(World, x, y, z, xp) and EntityFallingBlock(World, x, y, z, Block,
            // int)), which the table follows.
            EntityKind::Type(EntityType::XPOrb) => {
                assert!(close(0.25, c.width) && close(0.125, y_offset(c)), "{class}");
                assert_eq!((g.width, g.y_offset), (0.5, 0.25));
            }
            EntityKind::Type(EntityType::FallingSand) => {
                assert!(close(0.6, c.width) && close(0.0, y_offset(c)), "{class}");
                assert_eq!((g.width, g.y_offset), (0.98, 0.49));
            }
            // A slime's constructor size is random (1, 2 or 4 times 0.6) until the first tick.
            EntityKind::Type(EntityType::Slime | EntityType::LavaSlime) => {
                assert!(
                    [1.0, 2.0, 4.0].iter().any(|k| close(0.6 * k, c.width)),
                    "{class}"
                );
                assert!(close(c.height as f32 * 0.85, c.eye), "{class}");
                assert!(close(0.0, y_offset(c)), "{class}");
            }
            _ => {
                assert!(
                    close(g.width, c.width),
                    "{class} width {} vs {}",
                    g.width,
                    c.width
                );
                assert!(close(g.height, c.height), "{class} height");
                assert!(
                    close(g.eye_height, c.eye),
                    "{class} eye {} vs {}",
                    g.eye_height,
                    c.eye
                );
                assert!(close(g.y_offset, y_offset(c)), "{class} yOffset");
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 56, "55 EntityList classes and the fishing hook");
}

#[test]
fn watcher_registrations_match_the_dump() {
    let Some(dir) = dumps_dir() else { return };
    let dump = load(&dir, "ents17.tsv");
    let mut checked = 0;
    for (class, kind) in kinds(&dump) {
        let Some(c) = dump.classes.get(&class) else {
            continue;
        };
        let dumped: BTreeSet<(u8, u8)> = c.watchers.iter().map(|w| (w.0, w.1)).collect();
        let ours: BTreeSet<(u8, u8)> = (0..32)
            .filter_map(|i| registered_type_tag_v5(kind, i).map(|t| (i, t)))
            .collect();
        assert_eq!(ours, dumped, "{class}");
        checked += 1;
    }
    assert_eq!(checked, 56);
}

/// The full 1.7.10 watcher of every class (its constructor values) remapped: every output entry
/// must be a slot the 1.8.9 class registers, with the 1.8.9 type.
#[test]
fn remapped_watchers_land_on_1_8_9_registrations() {
    let Some(dir) = dumps_dir() else { return };
    let v17 = load(&dir, "ents17.tsv");
    let v18 = load(&dir, "ents18.tsv");
    let class18: BTreeMap<u8, &String> = v18.ids.iter().map(|(&id, c)| (id, c)).collect();
    let mut checked = 0;
    for (&id, class) in &v17.ids {
        let Some(c) = v17.classes.get(class) else {
            continue;
        };
        let kind = EntityKind::Type(entity_type(id).unwrap());
        let registered18: BTreeSet<(u8, u8)> = v18.classes[class18[&id]]
            .watchers
            .iter()
            .map(|w| (w.0, w.1))
            .collect();
        let list: Vec<MetadataEntry> = c
            .watchers
            .iter()
            .map(|(index, tag, value)| MetadataEntry {
                index: *index,
                value: value_of(*tag, value),
            })
            .collect();
        let out = remap_metadata_v5(kind, &Metadata::new(), &list);
        for entry in &out {
            let slot = (entry.index, entry.value.type_tag());
            assert!(
                registered18.contains(&slot),
                "{class}: {slot:?} -> {:?}",
                entry.value
            );
        }
        // Nothing registered in 1.7.10 is lost unless 1.8.9 has no slot for its meaning.
        assert!(out.len() >= list.len(), "{class}");
        checked += 1;
    }
    assert_eq!(checked, 55);
}

fn value_of(tag: u8, text: &str) -> MetadataValue {
    match tag {
        0 => MetadataValue::Byte(text.parse().unwrap()),
        1 => MetadataValue::Short(text.parse().unwrap()),
        2 => MetadataValue::Int(text.parse().unwrap()),
        3 => MetadataValue::Float(text.parse().unwrap()),
        4 => MetadataValue::String(text.to_owned()),
        5 => MetadataValue::ItemStack((text != "null").then_some(ItemStack {
            item_id: 1,
            count: 1,
            damage: 0,
            nbt: None,
        })),
        other => panic!("type {other}"),
    }
}
