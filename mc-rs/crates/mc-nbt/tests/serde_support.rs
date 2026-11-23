//! `serde` integration tests. Only built with `--features serde`.

use mc_nbt::{from_nbt, to_nbt, Nbt, NbtCompound, NbtLimits, TagType};
use serde::{Deserialize, Serialize};

/// The shape S16's multiplayer screen needs out of `servers.dat`.
///
/// java: `client/multiplayer/ServerData.java`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ServerData {
    name: String,
    ip: String,
    #[serde(rename = "acceptTextures", skip_serializing_if = "Option::is_none")]
    accept_textures: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Player {
    #[serde(rename = "Pos")]
    pos: [f64; 3],
    #[serde(rename = "Health")]
    health: f32,
    #[serde(rename = "SelectedItemSlot")]
    slot: u8,
    #[serde(rename = "Inventory")]
    inventory: Vec<Item>,
    #[serde(rename = "UUID")]
    uuid: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Item {
    id: i16,
    #[serde(rename = "Count")]
    count: u8,
    #[serde(rename = "Damage")]
    damage: i16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum GameMode {
    Survival,
    Creative,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Shape {
    Cube(f64),
    Box { w: f64, h: f64 },
    Verts(f64, f64),
}

fn sample_player() -> Player {
    Player {
        pos: [8.5, 64.0, -12.25],
        health: 20.0,
        slot: 200, // > i8::MAX on purpose: NBT stores it as a signed TAG_Byte
        inventory: vec![
            Item {
                id: 278,
                count: 1,
                damage: 0,
            },
            Item {
                id: 1,
                count: 64,
                damage: 0,
            },
        ],
        uuid: 0xDEAD_BEEF_CAFE_F00D,
    }
}

#[test]
fn structs_round_trip_through_nbt() {
    let player = sample_player();
    let nbt = to_nbt(&player).expect("serialize");
    let back: Player = from_nbt(&nbt).expect("deserialize");
    assert_eq!(back, player);
}

#[test]
fn structs_round_trip_through_the_wire_format_too() {
    let player = sample_player();
    let nbt = to_nbt(&player).expect("serialize");
    let bytes = mc_nbt::to_vec("", &nbt).expect("encode");
    let (_, decoded) =
        mc_nbt::from_slice_with_limits(&bytes, NbtLimits::UNLIMITED).expect("decode");
    let back: Player = from_nbt(&decoded).expect("deserialize");
    assert_eq!(back, player);
    assert_eq!(mc_nbt::to_vec("", &decoded).expect("re-encode"), bytes);
}

#[test]
fn field_names_and_tag_types_are_what_vanilla_writes() {
    let nbt = to_nbt(&sample_player()).expect("serialize");
    let c = nbt.as_compound().expect("compound");

    assert_eq!(c.keys().collect::<Vec<_>>().len(), 5);
    assert_eq!(c.get("Pos").map(Nbt::tag_type), Some(TagType::List));
    assert_eq!(
        c.get_list("Pos").map(|l| l[0].tag_type()),
        Some(TagType::Double)
    );
    assert_eq!(c.get("Health").map(Nbt::tag_type), Some(TagType::Float));
    // u8 200 is stored in a signed TAG_Byte, exactly as the game does.
    assert_eq!(c.get_i8("SelectedItemSlot"), Some(-56));
    assert_eq!(c.get("UUID").map(Nbt::tag_type), Some(TagType::Long));
    assert_eq!(c.get_i64("UUID"), Some(0xDEAD_BEEF_CAFE_F00Du64 as i64));
}

#[test]
fn bools_are_bytes_the_way_vanilla_stores_them() {
    let server = ServerData {
        name: "MCP Test Server".into(),
        ip: "127.0.0.1".into(),
        accept_textures: Some(true),
    };
    let nbt = to_nbt(&server).expect("serialize");
    assert_eq!(nbt.as_compound().unwrap().get_i8("acceptTextures"), Some(1));
    assert_eq!(from_nbt::<ServerData>(&nbt).expect("deserialize"), server);
}

#[test]
fn absent_optional_fields_are_omitted_and_come_back_as_none() {
    let server = ServerData {
        name: "n".into(),
        ip: "i".into(),
        accept_textures: None,
    };
    let nbt = to_nbt(&server).expect("serialize");
    assert_eq!(nbt.as_compound().unwrap().len(), 2);
    assert_eq!(from_nbt::<ServerData>(&nbt).expect("deserialize"), server);
}

#[test]
fn nbt_has_no_null() {
    // Serialising a bare `None` has nowhere to go; the message says what to do instead.
    let err = to_nbt(&Option::<i32>::None).unwrap_err();
    assert!(
        err.to_string().contains("skip_serializing_if"),
        "unhelpful message: {err}"
    );
}

#[test]
fn enums_map_to_strings_and_single_entry_compounds() {
    let unit = to_nbt(&GameMode::Creative).expect("unit variant");
    assert_eq!(unit, Nbt::String("Creative".into()));
    assert_eq!(from_nbt::<GameMode>(&unit).unwrap(), GameMode::Creative);

    for shape in [
        Shape::Cube(2.0),
        Shape::Box { w: 1.0, h: 2.0 },
        Shape::Verts(3.0, 4.0),
    ] {
        let nbt = to_nbt(&shape).expect("variant");
        assert_eq!(nbt.as_compound().map(NbtCompound::len), Some(1));
        assert_eq!(from_nbt::<Shape>(&nbt).expect("round trip"), shape);
    }
}

#[test]
fn maps_become_compounds() {
    use std::collections::BTreeMap;
    let mut rules = BTreeMap::new();
    rules.insert("keepInventory".to_owned(), "false".to_owned());
    rules.insert("mobGriefing".to_owned(), "true".to_owned());

    let nbt = to_nbt(&rules).expect("serialize");
    let c = nbt.as_compound().expect("compound");
    assert_eq!(c.get_str("keepInventory"), Some("false"));
    assert_eq!(
        from_nbt::<BTreeMap<String, String>>(&nbt).expect("deserialize"),
        rules
    );
}

#[test]
fn int_arrays_deserialize_as_sequences() {
    let nbt = Nbt::IntArray(vec![i32::MIN, 0, i32::MAX]);
    assert_eq!(
        from_nbt::<Vec<i32>>(&nbt).expect("int array as a seq"),
        vec![i32::MIN, 0, i32::MAX]
    );
}

#[test]
fn unsigned_targets_reinterpret_java_sign_extended_numeric_values() {
    assert_eq!(from_nbt::<u16>(&Nbt::Byte(-1)).unwrap(), u16::MAX);
    assert_eq!(from_nbt::<u32>(&Nbt::Byte(-1)).unwrap(), u32::MAX);
    assert_eq!(from_nbt::<u32>(&Nbt::Short(-2)).unwrap(), u32::MAX - 1);
    assert_eq!(from_nbt::<u64>(&Nbt::Byte(-1)).unwrap(), u64::MAX);
    assert_eq!(from_nbt::<u64>(&Nbt::Short(-2)).unwrap(), u64::MAX - 1);
    assert_eq!(from_nbt::<u64>(&Nbt::Int(-3)).unwrap(), u64::MAX - 2);

    // Negative control: widening a positive value is unchanged.
    assert_eq!(from_nbt::<u64>(&Nbt::Byte(127)).unwrap(), 127);
}

#[test]
fn strings_borrow_out_of_the_tag() {
    let nbt = Nbt::String("borrowed".into());
    let s: &str = from_nbt(&nbt).expect("zero-copy str");
    assert_eq!(s, "borrowed");
}

#[test]
fn type_mismatches_are_errors_not_panics() {
    let nbt = Nbt::String("not a number".into());
    assert!(from_nbt::<i32>(&nbt).is_err());
    assert!(from_nbt::<Player>(&Nbt::Byte(0)).is_err());
    assert!(from_nbt::<GameMode>(&Nbt::String("Nonexistent".into())).is_err());
}

/// [`Nbt`] itself is `Serialize`/`Deserialize`, so a tag tree can go through JSON.
#[test]
fn nbt_values_travel_through_a_self_describing_format() {
    let mut c = NbtCompound::new();
    c.insert("LevelName", Nbt::String("mcpworld".into()));
    c.insert("Time", Nbt::Long(1620));
    c.insert("Pos", Nbt::list(vec![Nbt::Double(1.5), Nbt::Double(-2.5)]));
    let value = Nbt::Compound(c);

    let json = serde_json::to_string(&value).expect("to json");
    assert!(json.contains("\"LevelName\":\"mcpworld\""), "{json}");

    // JSON is lossy about integer width — it has one number type — so the tags widen.
    // The structure and the strings survive, which is all a debug dump needs.
    let back: Nbt = serde_json::from_str(&json).expect("from json");
    let c = back.as_compound().expect("compound");
    assert_eq!(c.get_str("LevelName"), Some("mcpworld"));
    assert_eq!(c.get_i64("Time"), Some(1620));
    assert_eq!(c.get_list("Pos").map(<[Nbt]>::len), Some(2));
}

/// The real file, through the real API S16 will use.
#[test]
fn servers_dat_deserializes_into_typed_entries() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap()
        .join("MCP-919-main/jars/servers.dat");
    let bytes = std::fs::read(&path).expect("servers.dat");
    let (_, root) = mc_nbt::read_compound_with_limits(&mut &bytes[..], NbtLimits::UNLIMITED)
        .expect("parse servers.dat");

    let entries: Vec<ServerData> = root
        .get_list("servers")
        .expect("servers list")
        .iter()
        .map(|e| from_nbt(e).expect("ServerData"))
        .collect();

    assert_eq!(
        entries,
        vec![ServerData {
            name: "MCP Test Server".into(),
            ip: "127.0.0.1".into(),
            accept_textures: None,
        }]
    );
}
