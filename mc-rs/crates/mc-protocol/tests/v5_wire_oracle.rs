//! The protocol-5 codec against bytes the real 1.7.10 classes wrote.
//!
//! `parity/wire/java/WireOracle17.java` (run by `parity/tools/wire_capture_v5.sh`) builds packets
//! from fixed values with the 1.7.10 client and server classes and writes each frame (VarInt id,
//! then `writePacketData`) to `parity/wire/fixtures-v5/`. Nobody on this project wrote those
//! bytes. For every frame this test builds the packet from the **same values** independently and
//! checks, in both directions:
//!
//! * encoding the expected packet gives the Java bytes (a field written in the wrong order or
//!   width fails here even if `read` makes the same mistake);
//! * decoding the Java bytes gives the expected packet and consumes the frame exactly;
//! * decoding then encoding reproduces the Java bytes.
//!
//! Gzip NBT (item stacks, S35) is compared as Java writes it except for the deflate body: the
//! ten-byte gzip header, the CRC-32/size trailer and the inflated NBT bytes must be identical,
//! and every byte outside the blob too. Only the compressed body may differ.
//!
//! Chunk packets also check the inflated arrays against what Java's `readPacketData` produced
//! (`*.inflated.bin`). `decode.tsv` holds hand-written bodies with Java's verdict and Java's own
//! re-encode of what it read, pinning what `readPacketData` does with values a vanilla server
//! never sends.
//!
//! A missing fixture directory is a failure, not a skip.

mod v5_support;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use mc_math::BlockPos;
use mc_nbt::{Nbt, NbtCompound};
use mc_protocol::handshake::{Handshake, NextState};
use mc_protocol::login::{Disconnect, LoginStart, LoginSuccess};
use mc_protocol::v5::c2s as c5;
use mc_protocol::v5::chunk::{ChunkData, MapChunkBulk};
use mc_protocol::v5::login::{EncryptionRequest, EncryptionResponse};
use mc_protocol::v5::metadata::{MetadataEntry5, MetadataValue5};
use mc_protocol::v5::s2c as s5;
use mc_protocol::v5::{C2SPlay5, S2CPlay5};
use mc_protocol::{Angle, ItemStack, Packet, PacketBuffer};
use uuid::Uuid;
use v5_support::unhex;

// ---- fixtures ---------------------------------------------------------------------------------

fn fixtures_dir() -> PathBuf {
    let mut dir: &Path = Path::new(env!("CARGO_MANIFEST_DIR"));
    loop {
        let candidate = dir.join("parity").join("wire").join("fixtures-v5");
        if candidate.join("manifest.tsv").is_file() {
            return candidate;
        }
        dir = dir.parent().unwrap_or_else(|| {
            panic!(
                "parity/wire/fixtures-v5/manifest.tsv not found above {}. \
                 Run: bash parity/tools/wire_capture_v5.sh",
                env!("CARGO_MANIFEST_DIR")
            )
        });
    }
}

struct Frame {
    state: String,
    dir: String,
    id: i32,
    bytes: Vec<u8>,
}

fn parse_id(s: &str) -> i32 {
    i32::from_str_radix(s.trim_start_matches("0x"), 16).expect("id")
}

fn load_frames() -> BTreeMap<String, Frame> {
    let dir = fixtures_dir();
    let manifest = fs::read_to_string(dir.join("manifest.tsv")).expect("manifest.tsv");
    let mut out = BTreeMap::new();
    for line in manifest.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert!(cols.len() >= 8, "manifest row {line:?}");
        let bytes = fs::read(dir.join(format!("{}.bin", cols[0]))).expect("fixture .bin");
        assert_eq!(
            bytes,
            unhex(cols[5]),
            "{}: .bin and manifest hex disagree",
            cols[0]
        );
        out.insert(
            cols[0].to_owned(),
            Frame {
                state: cols[1].to_owned(),
                dir: cols[2].to_owned(),
                id: parse_id(cols[3]),
                bytes,
            },
        );
    }
    assert!(out.len() > 100, "manifest has only {} frames", out.len());
    out
}

// ---- comparison ---------------------------------------------------------------------------

/// [`v5_support::assert_wire_eq`], reporting (in `--nocapture` runs) which frames rely on the
/// deflate-body exemption.
fn assert_wire_eq(name: &str, ours: &[u8], java: &[u8]) {
    let relaxed = v5_support::assert_wire_eq(name, ours, java);
    if relaxed > 0 {
        println!("{name}: {relaxed} gzip blob(s) equal after inflating");
    }
}

fn encode_frame<P: Packet>(packet: &P) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    buf.write_varint(P::ID);
    packet.write(&mut buf);
    buf.finish_write().expect("encodable");
    buf.as_slice().to_vec()
}

fn body(frame: &Frame) -> PacketBuffer {
    let mut buf = PacketBuffer::from_slice(&frame.bytes);
    assert_eq!(buf.read_varint().unwrap(), frame.id);
    buf
}

// ---- the values WireOracle17 used ---------------------------------------------------------

/// `String.hashCode` over UTF-16 units.
fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, unit| {
        h.wrapping_mul(31).wrapping_add(i32::from(unit))
    })
}

/// Reorder a compound the way `NBTTagCompound.write` iterates its `java.util.HashMap` (JDK 8):
/// by bucket `(h ^ h >>> 16) & (capacity - 1)`, entries of one bucket in insertion order. Every
/// compound here has at most 12 keys, so the table keeps its default 16 buckets. The order on
/// the wire is vanilla's, not something these tests get to choose; building the expected tags
/// in the order the Java code inserts them and then applying this reproduces it independently.
fn java_order(compound: NbtCompound) -> NbtCompound {
    assert!(compound.len() <= 12, "a larger table would resize");
    let mut entries: Vec<(String, Nbt)> = compound
        .iter()
        .map(|(k, v)| (k.to_owned(), java_order_value(v.clone())))
        .collect();
    entries.sort_by_key(|(k, _)| {
        let h = java_string_hash(k);
        (h ^ ((h as u32) >> 16) as i32) & 15
    });
    let mut out = NbtCompound::new();
    for (k, v) in entries {
        out.insert(k, v);
    }
    out
}

fn java_order_value(value: Nbt) -> Nbt {
    match value {
        Nbt::Compound(c) => Nbt::Compound(java_order(c)),
        Nbt::List(items) => Nbt::list(items.iter().cloned().map(java_order_value).collect()),
        other => other,
    }
}

fn stone() -> ItemStack {
    ItemStack {
        item_id: 1,
        count: 64,
        damage: 0,
        nbt: None,
    }
}

fn sword_tag() -> NbtCompound {
    let mut ench = NbtCompound::new();
    ench.insert("id", Nbt::Short(16));
    ench.insert("lvl", Nbt::Short(5));
    let mut display = NbtCompound::new();
    display.insert("Name", Nbt::String("Excalibur \u{a7}c".into()));
    display.insert("Lore", Nbt::list(vec![Nbt::String("Sharp".into())]));
    let mut tag = NbtCompound::new();
    tag.insert("ench", Nbt::list(vec![Nbt::Compound(ench)]));
    tag.insert("display", Nbt::Compound(display));
    tag.insert("RepairCost", Nbt::Int(3));
    java_order(tag)
}

fn sword() -> ItemStack {
    ItemStack {
        item_id: 276,
        count: 1,
        damage: 7,
        nbt: Some(sword_tag()),
    }
}

fn book() -> ItemStack {
    let pages = (0..50)
        .map(|i| {
            Nbt::String(
                (0..200)
                    .map(|j| char::from(b'a' + ((i * 7 + j * 3) % 26) as u8))
                    .collect(),
            )
        })
        .collect();
    let mut tag = NbtCompound::new();
    tag.insert("pages", Nbt::list(pages));
    tag.insert("title", Nbt::String("Notes".into()));
    tag.insert("author", Nbt::String("Steve".into()));
    ItemStack {
        item_id: 387,
        count: 1,
        damage: 0,
        nbt: Some(java_order(tag)),
    }
}

fn skull() -> NbtCompound {
    let mut tag = NbtCompound::new();
    tag.insert("id", Nbt::String("Skull".into()));
    tag.insert("x", Nbt::Int(10));
    tag.insert("y", Nbt::Int(64));
    tag.insert("z", Nbt::Int(-10));
    tag.insert("SkullType", Nbt::Byte(3));
    tag.insert("Rot", Nbt::Byte(4));
    tag.insert("ExtraType", Nbt::String("Steve".into()));
    java_order(tag)
}

fn entry(index: u8, value: MetadataValue5) -> MetadataEntry5 {
    MetadataEntry5 { index, value }
}

/// `WireOracle17.watcher()`: a HashMap of small keys, written in ascending order.
fn watcher() -> Vec<MetadataEntry5> {
    vec![
        entry(0, MetadataValue5::Byte(0)),
        entry(1, MetadataValue5::Short(300)),
        entry(6, MetadataValue5::Float(20.0)),
        entry(8, MetadataValue5::Int(0)),
        entry(10, MetadataValue5::String("Zed".into())),
        entry(14, MetadataValue5::ItemStack(Some(stone()))),
    ]
}

fn angle(b: i8) -> Angle {
    Angle(b as u8)
}

fn teams(mode: i8) -> s5::Teams {
    let info = mode == 0 || mode == 2;
    let players = mode == 0 || mode == 3 || mode == 4;
    s5::Teams {
        team_name: "red".into(),
        mode,
        display_name: if info {
            "Red Team".into()
        } else {
            String::new()
        },
        prefix: if info { "[R] ".into() } else { String::new() },
        suffix: if info { " :)".into() } else { String::new() },
        friendly_flags: if info { 3 } else { 0 },
        players: if players {
            vec!["Steve".into(), "Alex".into()]
        } else {
            Vec::new()
        },
    }
}

enum Expected {
    Handshake(Handshake),
    LoginStart(LoginStart),
    EncryptionResponse(EncryptionResponse),
    LoginDisconnect(Disconnect),
    EncryptionRequest(EncryptionRequest),
    LoginSuccess(LoginSuccess),
    C2S(C2SPlay5),
    S2C(S2CPlay5),
}

fn expected() -> Vec<(&'static str, Expected)> {
    use Expected::{C2S, S2C};
    let key: Vec<u8> = (0..162u32).map(|i| (0x30 + i) as u8).collect();
    let mut beacon = vec![0u8; 256];
    beacon[3] = 1;
    beacon[7] = 10;
    let chat_100: String = (0..100)
        .map(|i| char::from(b'a' + (i % 26) as u8))
        .collect();
    let mut out = vec![
        (
            "hs_login",
            Expected::Handshake(Handshake {
                protocol_version: 5,
                server_address: "localhost".into(),
                server_port: 25565,
                next_state: NextState::Login,
            }),
        ),
        (
            "hs_status",
            Expected::Handshake(Handshake {
                protocol_version: 5,
                server_address: "mc.example.com".into(),
                server_port: 25566,
                next_state: NextState::Status,
            }),
        ),
        (
            "login_start",
            Expected::LoginStart(LoginStart {
                username: "Steve".into(),
            }),
        ),
        (
            "login_encryption_response",
            Expected::EncryptionResponse(EncryptionResponse {
                shared_secret_encrypted: vec![0xAB; 128],
                verify_token_encrypted: vec![0xCD; 128],
            }),
        ),
        (
            "login_disconnect",
            Expected::LoginDisconnect(Disconnect {
                reason: "\"Bye\"".into(),
            }),
        ),
        (
            "login_encryption_request",
            Expected::EncryptionRequest(EncryptionRequest {
                server_id: String::new(),
                public_key_der: key,
                verify_token: vec![1, 2, 3, 4],
            }),
        ),
        (
            "login_success",
            Expected::LoginSuccess(LoginSuccess {
                uuid: Uuid::from_u64_pair(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
                username: "Steve".into(),
            }),
        ),
        // ---- serverbound
        (
            "c00_keepalive_negative",
            C2S(C2SPlay5::KeepAlive(c5::KeepAlive { key: -1 })),
        ),
        (
            "c00_keepalive",
            C2S(C2SPlay5::KeepAlive(c5::KeepAlive { key: 0x1234_5678 })),
        ),
        (
            "c01_chat_ascii",
            C2S(C2SPlay5::ChatMessage(c5::ChatMessage {
                message: "hello world".into(),
            })),
        ),
        (
            "c01_chat_unicode",
            C2S(C2SPlay5::ChatMessage(c5::ChatMessage {
                message: "\u{e9}\u{20ac}\u{a7}a\u{1F600}".into(),
            })),
        ),
        (
            "c01_chat_100",
            C2S(C2SPlay5::ChatMessage(c5::ChatMessage { message: chat_100 })),
        ),
        (
            "c02_interact",
            C2S(C2SPlay5::UseEntity(c5::UseEntity {
                entity_id: 0x0102_0304,
                action: c5::UseEntityAction::Interact,
            })),
        ),
        (
            "c02_attack",
            C2S(C2SPlay5::UseEntity(c5::UseEntity {
                entity_id: -5,
                action: c5::UseEntityAction::Attack,
            })),
        ),
        (
            "c03_ground",
            C2S(C2SPlay5::Player(c5::Player { on_ground: true })),
        ),
        (
            "c03_air",
            C2S(C2SPlay5::Player(c5::Player { on_ground: false })),
        ),
        (
            "c04_position",
            C2S(C2SPlay5::PlayerPosition(c5::PlayerPosition {
                x: 1.5,
                y: 64.0,
                stance: 65.62,
                z: -2.5,
                on_ground: true,
            })),
        ),
        (
            "c04_position_negzero",
            C2S(C2SPlay5::PlayerPosition(c5::PlayerPosition {
                x: -0.0,
                y: 70.0,
                stance: 71.62,
                z: -0.0,
                on_ground: false,
            })),
        ),
        (
            "c04_position_sneak",
            C2S(C2SPlay5::PlayerPosition(c5::PlayerPosition {
                x: 10.5,
                y: 64.0,
                stance: 64.0 + 1.54,
                z: -3.5,
                on_ground: true,
            })),
        ),
        (
            "c05_look",
            C2S(C2SPlay5::PlayerLook(c5::PlayerLook {
                yaw: -90.5,
                pitch: 45.25,
                on_ground: false,
            })),
        ),
        (
            "c06_poslook",
            C2S(C2SPlay5::PlayerPositionAndLook(c5::PlayerPositionAndLook {
                x: 1.5,
                y: 64.0,
                stance: 65.62,
                z: -2.5,
                yaw: 90.0,
                pitch: -10.0,
                on_ground: true,
            })),
        ),
        (
            "c06_poslook_far",
            C2S(C2SPlay5::PlayerPositionAndLook(c5::PlayerPositionAndLook {
                x: 30_000_000.5,
                y: 255.0,
                stance: 255.0 + 1.62,
                z: -30_000_000.5,
                yaw: -0.0,
                pitch: 90.0,
                on_ground: false,
            })),
        ),
    ];
    let digging = [
        ("c07_start", 0, -1234, 70, 5678, 1),
        ("c07_abort_reset", 1, 10, 64, -10, 0xFF),
        ("c07_release", 5, 0, 0, 0, 255),
        ("c07_drop", 4, 0, 0, 0, 0),
        ("c07_drop_all", 3, 0, 0, 0, 0),
        ("c07_y255", 2, 30_000_000, 255, -30_000_000, 5),
    ];
    for (name, status, x, y, z, face) in digging {
        out.push((
            name,
            C2S(C2SPlay5::PlayerDigging(c5::PlayerDigging {
                status,
                x,
                y,
                z,
                face,
            })),
        ));
    }
    let placement = |x, y, z, direction, stack, cursor: (u8, u8, u8)| {
        C2S(C2SPlay5::PlayerBlockPlacement(c5::PlayerBlockPlacement {
            x,
            y,
            z,
            direction,
            stack,
            cursor_x: cursor.0,
            cursor_y: cursor.1,
            cursor_z: cursor.2,
        }))
    };
    out.push((
        "c08_place",
        placement(10, 64, -10, 1, Some(stone()), (8, 16, 4)),
    ));
    out.push((
        "c08_place_nbt",
        placement(10, 64, -10, 3, Some(sword()), (0, 8, 15)),
    ));
    out.push((
        "c08_air_use",
        placement(-1, 255, -1, 255, Some(stone()), (0, 0, 0)),
    ));
    out.push((
        "c08_air_use_empty",
        placement(-1, 255, -1, 255, None, (0, 0, 0)),
    ));
    out.push((
        "c09_held",
        C2S(C2SPlay5::HeldItemChange(c5::HeldItemChange { slot_id: 5 })),
    ));
    out.push((
        "c0a_swing",
        C2S(C2SPlay5::Animation(c5::Animation {
            entity_id: 12345,
            animation_type: 1,
        })),
    ));
    for (name, action_id, aux_data) in [
        ("c0b_sneak", 1, 0),
        ("c0b_action_2", 2, 0),
        ("c0b_action_3", 3, 0),
        ("c0b_action_4", 4, 0),
        ("c0b_action_5", 5, 0),
        ("c0b_horse_jump", 6, 77),
        ("c0b_action_7", 7, 0),
    ] {
        out.push((
            name,
            C2S(C2SPlay5::EntityAction(c5::EntityAction {
                entity_id: 12345,
                action_id,
                aux_data,
            })),
        ));
    }
    out.extend([
        (
            "c0c_steer",
            C2S(C2SPlay5::SteerVehicle(c5::SteerVehicle {
                strafe: 0.98,
                forward: -0.5,
                jump: true,
                sneak: false,
            })),
        ),
        (
            "c0d_close",
            C2S(C2SPlay5::CloseWindow(c5::CloseWindow { window_id: 1 })),
        ),
        (
            "c0d_close_player",
            C2S(C2SPlay5::CloseWindow(c5::CloseWindow { window_id: 0 })),
        ),
        (
            "c0e_click_left",
            C2S(C2SPlay5::ClickWindow(c5::ClickWindow {
                window_id: 1,
                slot_id: 36,
                used_button: 0,
                action_number: 5,
                mode: 0,
                clicked_item: Some(stone()),
            })),
        ),
        (
            "c0e_click_outside",
            C2S(C2SPlay5::ClickWindow(c5::ClickWindow {
                window_id: 0,
                slot_id: -999,
                used_button: 0,
                action_number: -2,
                mode: 4,
                clicked_item: None,
            })),
        ),
        (
            "c0e_click_nbt",
            C2S(C2SPlay5::ClickWindow(c5::ClickWindow {
                window_id: 2,
                slot_id: 3,
                used_button: 1,
                action_number: 300,
                mode: 1,
                clicked_item: Some(sword()),
            })),
        ),
        (
            "c0f_confirm_accept",
            C2S(C2SPlay5::ConfirmTransaction(c5::ConfirmTransaction {
                window_id: 1,
                action_number: 5,
                accepted: true,
            })),
        ),
        (
            "c0f_confirm_reject",
            C2S(C2SPlay5::ConfirmTransaction(c5::ConfirmTransaction {
                window_id: 0,
                action_number: -32768,
                accepted: false,
            })),
        ),
        (
            "c10_creative_nbt",
            C2S(C2SPlay5::CreativeInventoryAction(
                c5::CreativeInventoryAction {
                    slot_id: 36,
                    stack: Some(sword()),
                },
            )),
        ),
        (
            "c10_creative_book",
            C2S(C2SPlay5::CreativeInventoryAction(
                c5::CreativeInventoryAction {
                    slot_id: 37,
                    stack: Some(book()),
                },
            )),
        ),
        (
            "c10_creative_clear",
            C2S(C2SPlay5::CreativeInventoryAction(
                c5::CreativeInventoryAction {
                    slot_id: -1,
                    stack: None,
                },
            )),
        ),
        (
            "c11_enchant",
            C2S(C2SPlay5::EnchantItem(c5::EnchantItem {
                window_id: 3,
                button: 2,
            })),
        ),
        (
            "c12_sign",
            C2S(C2SPlay5::UpdateSign(c5::UpdateSign {
                x: 10,
                y: 64,
                z: -10,
                lines: [
                    "Hello".into(),
                    String::new(),
                    "line \u{e9}".into(),
                    "fifteen chars!!".into(),
                ],
            })),
        ),
        (
            "c13_abilities",
            C2S(C2SPlay5::PlayerAbilities(c5::PlayerAbilities {
                invulnerable: true,
                flying: true,
                allow_flying: true,
                creative_mode: true,
                fly_speed: 0.05,
                walk_speed: 0.1,
            })),
        ),
        (
            "c14_tab",
            C2S(C2SPlay5::TabComplete(c5::TabComplete {
                message: "/tp St".into(),
            })),
        ),
    ]);
    for (name, locale, view_distance, chat_visibility, chat_colors, difficulty, show_cape) in [
        ("c15_settings", "en_US", 12, 0, true, 2, true),
        ("c15_settings_hidden", "de_DE", 2, 2, false, 0, false),
        ("c15_settings_system_hard", "en_US", 8, 1, true, 3, false),
        ("c15_settings_easy", "en_GB", 16, 0, false, 1, true),
    ] {
        out.push((
            name,
            C2S(C2SPlay5::ClientSettings(c5::ClientSettings {
                locale: locale.into(),
                view_distance,
                chat_visibility,
                chat_colors,
                difficulty,
                show_cape,
            })),
        ));
    }
    for (name, status) in [("c16_respawn", 0), ("c16_stats", 1), ("c16_achievement", 2)] {
        out.push((
            name,
            C2S(C2SPlay5::ClientStatus(c5::ClientStatus { status })),
        ));
    }
    for (name, channel, data) in [
        ("c17_brand", "MC|Brand", b"vanilla".to_vec()),
        ("c17_brand_aero", "MC|Brand", b"Aero".to_vec()),
        ("c17_itemname_ab", "MC|ItemName", b"ab".to_vec()),
        ("c17_itemname_empty", "MC|ItemName", Vec::new()),
        ("c17_null", "REGISTER", Vec::new()),
        ("c17_binary", "MC|BEdit", (0..=255u8).collect()),
        ("c17_beacon_backing_array", "MC|Beacon", beacon),
    ] {
        out.push((
            name,
            C2S(C2SPlay5::PluginMessage(c5::PluginMessage {
                channel: channel.into(),
                data,
            })),
        ));
    }

    // ---- clientbound
    out.extend([
        (
            "s00_keepalive",
            S2C(S2CPlay5::KeepAlive(s5::KeepAlive { id: -559_038_737 })),
        ),
        (
            "s01_join",
            S2C(S2CPlay5::JoinGame(s5::JoinGame {
                entity_id: 25,
                hardcore_mode: true,
                game_mode: 1,
                dimension: -1,
                difficulty: 3,
                max_players: 20,
                level_type: "largeBiomes".into(),
            })),
        ),
        (
            "s02_chat",
            S2C(S2CPlay5::Chat(s5::Chat {
                component: "\"Hello \u{a7}aworld\"".into(),
            })),
        ),
        (
            "s03_time",
            S2C(S2CPlay5::TimeUpdate(s5::TimeUpdate {
                total_world_time: 123_456_789,
                world_time: -6000,
            })),
        ),
        (
            "s04_equipment",
            S2C(S2CPlay5::EntityEquipment(s5::EntityEquipment {
                entity_id: 77,
                equipment_slot: 4,
                item: Some(sword()),
            })),
        ),
        (
            "s05_spawn",
            S2C(S2CPlay5::SpawnPosition(s5::SpawnPosition {
                x: -100,
                y: 64,
                z: 200,
            })),
        ),
        (
            "s06_health",
            S2C(S2CPlay5::UpdateHealth(s5::UpdateHealth {
                health: 19.5,
                food_level: 17,
                saturation_level: 4.25,
            })),
        ),
        (
            "s07_respawn",
            S2C(S2CPlay5::Respawn(s5::Respawn {
                dimension: 1,
                difficulty: 1,
                game_mode: 2,
                level_type: "flat".into(),
            })),
        ),
        (
            "s08_poslook",
            S2C(S2CPlay5::PlayerPosLook(s5::PlayerPosLook {
                x: 1.5,
                y: 65.62,
                z: -2.5,
                yaw: 90.5,
                pitch: -12.25,
                on_ground: true,
            })),
        ),
        (
            "s09_held",
            S2C(S2CPlay5::HeldItemChange(s5::HeldItemChange { slot: 3 })),
        ),
        (
            "s0a_usebed",
            S2C(S2CPlay5::UseBed(s5::UseBed {
                player_id: 99,
                x: -100,
                y: 70,
                z: 200,
            })),
        ),
        (
            "s0a_usebed_high",
            S2C(S2CPlay5::UseBed(s5::UseBed {
                player_id: 99,
                x: 1,
                y: -56,
                z: 2,
            })),
        ),
        (
            "s0b_animation",
            S2C(S2CPlay5::Animation(s5::Animation {
                entity_id: 300,
                animation_type: 4,
            })),
        ),
        (
            "s0c_spawn_player",
            S2C(S2CPlay5::SpawnPlayer(s5::SpawnPlayer {
                entity_id: 401,
                uuid: Uuid::from_u64_pair(0x0123_4567_89ab_cdef, 0x0123_4567_89ab_cdef),
                name: "Alex".into(),
                properties: vec![s5::ProfileProperty {
                    name: "textures".into(),
                    value: "e30=".into(),
                    signature: "c2ln".into(),
                }],
                x: 320,
                y: 2048,
                z: -160,
                yaw: angle(64),
                pitch: angle(-10),
                current_item: 276,
                metadata: watcher(),
            })),
        ),
        (
            "s0d_collect",
            S2C(S2CPlay5::CollectItem(s5::CollectItem {
                collected_entity_id: 501,
                collector_entity_id: 502,
            })),
        ),
        (
            "s0e_falling_block",
            S2C(S2CPlay5::SpawnObject(s5::SpawnObject {
                entity_id: 500,
                object_type: 70,
                x: -32,
                y: 2048,
                z: 64,
                pitch: angle(0),
                yaw: angle(-128),
                data: 12 | 2 << 16,
                velocity: Some((10, -20, 30)),
            })),
        ),
        (
            "s0e_no_velocity",
            S2C(S2CPlay5::SpawnObject(s5::SpawnObject {
                entity_id: 501,
                object_type: 10,
                x: 32,
                y: 64,
                z: 96,
                pitch: angle(5),
                yaw: angle(6),
                data: 0,
                velocity: None,
            })),
        ),
        (
            "s0f_spawn_mob",
            S2C(S2CPlay5::SpawnMob(s5::SpawnMob {
                entity_id: 601,
                entity_type: 54,
                x: 100,
                y: 2100,
                z: -100,
                yaw: angle(64),
                pitch: angle(-20),
                head_yaw: angle(70),
                velocity_x: 100,
                velocity_y: -200,
                velocity_z: 3999,
                metadata: watcher(),
            })),
        ),
        (
            "s10_painting",
            S2C(S2CPlay5::SpawnPainting(s5::SpawnPainting {
                entity_id: 700,
                title: "Kebab".into(),
                x: 10,
                y: 64,
                z: -10,
                direction: 2,
            })),
        ),
        (
            "s11_xp_orb",
            S2C(S2CPlay5::SpawnExperienceOrb(s5::SpawnExperienceOrb {
                entity_id: 800,
                x: 320,
                y: 2050,
                z: -320,
                xp_value: 7,
            })),
        ),
        (
            "s12_velocity",
            S2C(S2CPlay5::EntityVelocity(s5::EntityVelocity {
                entity_id: 900,
                motion_x: 100,
                motion_y: -100,
                motion_z: 8000,
            })),
        ),
        (
            "s13_destroy",
            S2C(S2CPlay5::DestroyEntities(s5::DestroyEntities {
                entity_ids: vec![1, -2, 300_000],
            })),
        ),
        (
            "s14_entity",
            S2C(S2CPlay5::Entity(s5::Entity { entity_id: 1000 })),
        ),
        (
            "s15_relmove",
            S2C(S2CPlay5::EntityRelMove(s5::EntityRelMove {
                entity_id: 1001,
                delta_x: 1,
                delta_y: -2,
                delta_z: 127,
            })),
        ),
        (
            "s16_look",
            S2C(S2CPlay5::EntityLook(s5::EntityLook {
                entity_id: 1002,
                yaw: angle(64),
                pitch: angle(-64),
            })),
        ),
        (
            "s17_lookmove",
            S2C(S2CPlay5::EntityLookAndRelMove(s5::EntityLookAndRelMove {
                entity_id: 1003,
                delta_x: -1,
                delta_y: 0,
                delta_z: 1,
                yaw: angle(2),
                pitch: angle(-3),
            })),
        ),
        (
            "s18_teleport",
            S2C(S2CPlay5::EntityTeleport(s5::EntityTeleport {
                entity_id: 1004,
                x: -3200,
                y: 2048,
                z: 3200,
                yaw: angle(10),
                pitch: angle(-10),
            })),
        ),
        (
            "s19_headlook",
            S2C(S2CPlay5::EntityHeadLook(s5::EntityHeadLook {
                entity_id: 1005,
                yaw: angle(33),
            })),
        ),
        (
            "s1a_status",
            S2C(S2CPlay5::EntityStatus(s5::EntityStatus {
                entity_id: 1006,
                opcode: 2,
            })),
        ),
        (
            "s1b_attach",
            S2C(S2CPlay5::EntityAttach(s5::EntityAttach {
                entity_id: 1007,
                vehicle_entity_id: -1,
                leash: 1,
            })),
        ),
        (
            "s1c_metadata",
            S2C(S2CPlay5::EntityMetadata(s5::EntityMetadata {
                entity_id: 1008,
                metadata: vec![
                    entry(0, MetadataValue5::Byte(5)),
                    entry(2, MetadataValue5::Short(-1)),
                    entry(12, MetadataValue5::Int(-24000)),
                    entry(6, MetadataValue5::Float(1.5)),
                    entry(10, MetadataValue5::String("Name \u{e9}".into())),
                    entry(2, MetadataValue5::ItemStack(Some(sword()))),
                    entry(3, MetadataValue5::ItemStack(None)),
                    entry(31, MetadataValue5::Coordinates(BlockPos::new(-1, 64, 100))),
                ],
            })),
        ),
        (
            "s1d_effect",
            S2C(S2CPlay5::EntityEffect(s5::EntityEffect {
                entity_id: 1009,
                effect_id: 1,
                amplifier: 2,
                duration: 600,
            })),
        ),
        (
            "s1d_effect_long",
            S2C(S2CPlay5::EntityEffect(s5::EntityEffect {
                entity_id: 1009,
                effect_id: 14,
                amplifier: 0,
                duration: 32767,
            })),
        ),
        (
            "s1e_remove_effect",
            S2C(S2CPlay5::RemoveEntityEffect(s5::RemoveEntityEffect {
                entity_id: 1010,
                effect_id: 14,
            })),
        ),
        (
            "s1f_experience",
            S2C(S2CPlay5::SetExperience(s5::SetExperience {
                experience_bar: 0.5,
                level: 30,
                total_experience: 1234,
            })),
        ),
        (
            "s20_properties",
            S2C(S2CPlay5::EntityProperties(s5::EntityProperties {
                entity_id: 1011,
                properties: vec![
                    s5::AttributeSnapshot {
                        key: "generic.movementSpeed".into(),
                        base_value: 0.1,
                        modifiers: vec![s5::AttributeModifier {
                            uuid: Uuid::from_u64_pair(0x662a_6b8d_da3e_4c1c, 0x8813_969a_2b2c_b27e),
                            amount: 0.300_000_011_920_928_96,
                            operation: 2,
                        }],
                    },
                    s5::AttributeSnapshot {
                        key: "generic.maxHealth".into(),
                        base_value: 20.0,
                        modifiers: Vec::new(),
                    },
                ],
            })),
        ),
        (
            "s22_multiblock",
            S2C(S2CPlay5::MultiBlockChange(s5::MultiBlockChange {
                chunk_x: 3,
                chunk_z: -2,
                record_count: 2,
                data: Some(vec![0x34, 70, 0x00, 0x12, 0xF0, 0xFF, 0x23, 0x4F]),
            })),
        ),
        (
            "s23_blockchange",
            S2C(S2CPlay5::BlockChange(s5::BlockChange {
                x: -5,
                y: 255,
                z: 7,
                block_id: 35,
                metadata: 14,
            })),
        ),
        (
            "s24_blockaction",
            S2C(S2CPlay5::BlockAction(s5::BlockAction {
                x: 10,
                y: 64,
                z: -10,
                data1: 1,
                data2: 2,
                block_id: 54,
            })),
        ),
        (
            "s25_breakanim",
            S2C(S2CPlay5::BlockBreakAnimation(s5::BlockBreakAnimation {
                breaker_id: 1012,
                x: 1,
                y: 2,
                z: 3,
                progress: 9,
            })),
        ),
        (
            "s27_explosion",
            S2C(S2CPlay5::Explosion(s5::Explosion {
                x: -1.5,
                y: 64.9,
                z: 0.5,
                strength: 4.0,
                offsets: vec![(1, -1, 0), (-1, 0, 1)],
                motion_x: 0.1,
                motion_y: 0.2,
                motion_z: -0.3,
            })),
        ),
        (
            "s28_effect",
            S2C(S2CPlay5::Effect(s5::Effect {
                effect_id: 2001,
                x: 10,
                y: 200,
                z: -10,
                data: 1 | 3 << 12,
                disable_relative_volume: false,
            })),
        ),
        (
            "s29_sound",
            S2C(S2CPlay5::SoundEffect(s5::SoundEffect {
                sound_name: "random.click".into(),
                x: 80,
                y: 520,
                z: -80,
                volume: 0.5,
                pitch: 63,
            })),
        ),
        (
            "s2a_particles",
            S2C(S2CPlay5::Particles(s5::Particles {
                particle_name: "iconcrack_276_0".into(),
                x: 1.5,
                y: 64.0,
                z: -1.5,
                offset_x: 0.1,
                offset_y: 0.2,
                offset_z: 0.3,
                speed: 0.05,
                count: 10,
            })),
        ),
        (
            "s2b_gamestate",
            S2C(S2CPlay5::ChangeGameState(s5::ChangeGameState {
                reason: 7,
                value: 0.5,
            })),
        ),
        (
            "s2c_global",
            S2C(S2CPlay5::SpawnGlobalEntity(s5::SpawnGlobalEntity {
                entity_id: 1013,
                kind: 1,
                x: 320,
                y: 2048,
                z: -320,
            })),
        ),
        (
            "s2d_openwindow",
            S2C(S2CPlay5::OpenWindow(s5::OpenWindow {
                window_id: 1,
                inventory_type: 0,
                title: "Chest".into(),
                slot_count: 27,
                use_provided_title: false,
                entity_id: 0,
            })),
        ),
        (
            "s2d_openwindow_horse",
            S2C(S2CPlay5::OpenWindow(s5::OpenWindow {
                window_id: 2,
                inventory_type: 11,
                title: "Horse".into(),
                slot_count: 2,
                use_provided_title: true,
                entity_id: 1014,
            })),
        ),
        (
            "s2e_closewindow",
            S2C(S2CPlay5::CloseWindow(s5::CloseWindow { window_id: 1 })),
        ),
        (
            "s2f_setslot_cursor",
            S2C(S2CPlay5::SetSlot(s5::SetSlot {
                window_id: -1,
                slot: -1,
                item: Some(sword()),
            })),
        ),
        (
            "s30_windowitems",
            S2C(S2CPlay5::WindowItems(s5::WindowItems {
                window_id: 0,
                items: vec![None, Some(stone()), Some(sword())],
            })),
        ),
        (
            "s31_property",
            S2C(S2CPlay5::WindowProperty(s5::WindowProperty {
                window_id: 1,
                property: 0,
                value: 200,
            })),
        ),
        (
            "s32_confirm",
            S2C(S2CPlay5::ConfirmTransaction(s5::ConfirmTransaction {
                window_id: 1,
                action_number: -5,
                accepted: false,
            })),
        ),
        (
            "s33_updatesign",
            S2C(S2CPlay5::UpdateSign(s5::UpdateSign {
                x: 10,
                y: 64,
                z: -10,
                lines: [
                    "Hello".into(),
                    String::new(),
                    "line three".into(),
                    "fifteen chars!!".into(),
                ],
            })),
        ),
        (
            "s34_maps",
            S2C(S2CPlay5::Maps(s5::Maps {
                map_id: 5,
                data: vec![2, 3],
            })),
        ),
        (
            "s35_tileentity",
            S2C(S2CPlay5::UpdateTileEntity(s5::UpdateTileEntity {
                x: 10,
                y: 64,
                z: -10,
                action: 4,
                nbt: Some(skull()),
            })),
        ),
        (
            "s36_signeditor",
            S2C(S2CPlay5::SignEditorOpen(s5::SignEditorOpen {
                x: 10,
                y: 64,
                z: -10,
            })),
        ),
        (
            "s37_statistics",
            S2C(S2CPlay5::Statistics(s5::Statistics {
                entries: vec![
                    ("stat.leaveGame".into(), 5),
                    ("stat.mineBlock.1".into(), 300),
                ],
            })),
        ),
        (
            "s38_playerlist",
            S2C(S2CPlay5::PlayerListItem(s5::PlayerListItem {
                name: "Steve".into(),
                online: true,
                ping: 42,
            })),
        ),
        (
            "s39_abilities",
            S2C(S2CPlay5::PlayerAbilities(s5::PlayerAbilities {
                invulnerable: true,
                flying: true,
                allow_flying: true,
                creative_mode: false,
                fly_speed: 0.05,
                walk_speed: 0.1,
            })),
        ),
        (
            "s3a_tabcomplete",
            S2C(S2CPlay5::TabComplete(s5::TabComplete {
                matches: vec!["/tp".into(), "/tell".into()],
            })),
        ),
        (
            "s3b_objective",
            S2C(S2CPlay5::ScoreboardObjective(s5::ScoreboardObjective {
                objective_name: "kills".into(),
                display_name: "Player Kills".into(),
                mode: 0,
            })),
        ),
        (
            "s3c_score_change",
            S2C(S2CPlay5::UpdateScore(s5::UpdateScore {
                score_name: "Steve".into(),
                action: 0,
                objective_name: "kills".into(),
                value: 12,
            })),
        ),
        (
            "s3c_score_remove",
            S2C(S2CPlay5::UpdateScore(s5::UpdateScore {
                score_name: "Steve".into(),
                action: 1,
                objective_name: String::new(),
                value: 0,
            })),
        ),
        ("s3e_teams_create", S2C(S2CPlay5::Teams(teams(0)))),
        ("s3e_teams_remove", S2C(S2CPlay5::Teams(teams(1)))),
        ("s3e_teams_update", S2C(S2CPlay5::Teams(teams(2)))),
        ("s3e_teams_add_players", S2C(S2CPlay5::Teams(teams(3)))),
        (
            "s3d_display",
            S2C(S2CPlay5::DisplayScoreboard(s5::DisplayScoreboard {
                position: 1,
                score_name: "kills".into(),
            })),
        ),
        (
            "s3f_custompayload",
            S2C(S2CPlay5::PluginMessage(s5::PluginMessage {
                channel: "MC|TrList".into(),
                data: vec![0, 0, 0, 1, 2],
            })),
        ),
        (
            "s40_disconnect",
            S2C(S2CPlay5::Disconnect(s5::Disconnect {
                reason: "\"Kicked\"".into(),
            })),
        ),
    ]);
    out
}

/// Frames checked by [`chunk_frames_match_javas_inflate`] instead of an expected struct: their
/// zlib bytes come from the JDK's `Deflater`, which the test cannot reproduce independently.
const CHUNK_FRAMES: [&str; 5] = [
    "s21_chunk_full",
    "s21_chunk_partial",
    "s21_chunk_unload",
    "s21_chunk_nether",
    "s26_bulk",
];

fn check_both_ways<P: Packet + PartialEq + std::fmt::Debug>(
    name: &str,
    frame: &Frame,
    expected: &P,
) {
    assert_wire_eq(name, &encode_frame(expected), &frame.bytes);
    let mut buf = body(frame);
    let decoded = P::read(&mut buf).unwrap_or_else(|e| panic!("{name}: decode failed: {e}"));
    assert!(buf.is_empty(), "{name}: {} trailing bytes", buf.remaining());
    assert_eq!(&decoded, expected, "{name}: decoded fields");
    assert_wire_eq(name, &encode_frame(&decoded), &frame.bytes);
}

fn check_play<E: PartialEq + std::fmt::Debug>(
    name: &str,
    frame: &Frame,
    expected: &E,
    id: i32,
    write: impl Fn(&E, &mut PacketBuffer),
    read: impl Fn(i32, &mut PacketBuffer) -> Result<E, mc_protocol::ProtoError>,
) {
    assert_eq!(id, frame.id, "{name}: packet id");
    let encode = |packet: &E| {
        let mut buf = PacketBuffer::new();
        buf.write_varint(id);
        write(packet, &mut buf);
        buf.finish_write().expect("encodable");
        buf.as_slice().to_vec()
    };
    assert_wire_eq(name, &encode(expected), &frame.bytes);
    let mut buf = body(frame);
    let decoded = read(frame.id, &mut buf).unwrap_or_else(|e| panic!("{name}: decode failed: {e}"));
    assert!(buf.is_empty(), "{name}: {} trailing bytes", buf.remaining());
    assert_eq!(&decoded, expected, "{name}: decoded fields");
    assert_wire_eq(name, &encode(&decoded), &frame.bytes);
}

#[test]
fn every_frame_matches_the_1_7_10_classes_both_ways() {
    let frames = load_frames();
    let expected = expected();
    let mut covered: Vec<&str> = CHUNK_FRAMES.to_vec();
    for (name, want) in &expected {
        let frame = frames
            .get(*name)
            .unwrap_or_else(|| panic!("{name}: no such fixture"));
        match want {
            Expected::Handshake(p) => check_both_ways(name, frame, p),
            Expected::LoginStart(p) => check_both_ways(name, frame, p),
            Expected::EncryptionResponse(p) => check_both_ways(name, frame, p),
            Expected::LoginDisconnect(p) => check_both_ways(name, frame, p),
            Expected::EncryptionRequest(p) => check_both_ways(name, frame, p),
            Expected::LoginSuccess(p) => check_both_ways(name, frame, p),
            Expected::C2S(p) => {
                assert_eq!((frame.state.as_str(), frame.dir.as_str()), ("PLAY", "C2S"));
                check_play(name, frame, p, p.id(), C2SPlay5::write, C2SPlay5::read);
            }
            Expected::S2C(p) => {
                assert_eq!((frame.state.as_str(), frame.dir.as_str()), ("PLAY", "S2C"));
                check_play(name, frame, p, p.id(), S2CPlay5::write, S2CPlay5::read);
            }
        }
        covered.push(name);
    }
    let missing: Vec<&String> = frames
        .keys()
        .filter(|name| !covered.contains(&name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "fixtures without an expectation: {missing:?}"
    );
}

#[test]
fn every_play_packet_has_a_vector() {
    let frames = load_frames();
    let mut c2s: Vec<i32> = Vec::new();
    let mut s2c: Vec<i32> = Vec::new();
    for frame in frames.values().filter(|f| f.state == "PLAY") {
        let ids = if frame.dir == "C2S" {
            &mut c2s
        } else {
            &mut s2c
        };
        if !ids.contains(&frame.id) {
            ids.push(frame.id);
        }
    }
    c2s.sort_unstable();
    s2c.sort_unstable();
    assert_eq!(
        c2s,
        (0x00..=0x17).collect::<Vec<_>>(),
        "every serverbound packet"
    );
    assert_eq!(
        s2c,
        (0x00..=0x40).collect::<Vec<_>>(),
        "every clientbound packet"
    );
}

/// `WireOracle17.column`: the uncompressed bytes the oracle deflated.
fn raw_column(primary: u16, add: u16, sky: bool, ground_up: bool, seed: i32) -> Vec<u8> {
    let n = primary.count_ones() as i32;
    let mut out = Vec::new();
    for s in 0..n {
        out.extend((0..4096).map(|i: i32| (i * 7 + s * 13 + seed) as u8));
    }
    for s in 0..n {
        out.extend((0..2048).map(|i: i32| (i * 5 + s + seed) as u8));
    }
    for s in 0..n {
        out.extend((0..2048).map(|i: i32| 0xF0 | ((i + s) & 0x0F) as u8));
    }
    if sky {
        out.extend(std::iter::repeat_n(0xFFu8, 2048 * n as usize));
    }
    for _ in 0..add.count_ones() {
        out.extend((0..2048).map(|i: i32| ((i >> 4) & 0x11) as u8));
    }
    if ground_up {
        out.extend((0..256).map(|i: i32| (i % 24) as u8));
    }
    out
}

#[test]
fn chunk_frames_match_javas_inflate() {
    let frames = load_frames();
    let dir = fixtures_dir();
    for (name, x, z, ground_up, primary, add, sky) in [
        ("s21_chunk_full", 3, -4, true, 0x0005u16, 0x0001u16, true),
        ("s21_chunk_partial", -1, 7, false, 0x0010, 0x0000, true),
        ("s21_chunk_unload", 8, 8, true, 0x0000, 0x0000, false),
        ("s21_chunk_nether", 0, 0, true, 0x8001, 0x0000, false),
    ] {
        let frame = &frames[name];
        let mut buf = body(frame);
        let packet = ChunkData::read(&mut buf).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(buf.is_empty(), "{name}: trailing bytes");
        assert_eq!(
            (
                packet.chunk_x,
                packet.chunk_z,
                packet.ground_up_continuous,
                packet.primary_bit_mask,
                packet.add_bit_mask
            ),
            (x, z, ground_up, primary, add),
            "{name}: header fields"
        );
        let java = fs::read(dir.join(format!("{name}.inflated.bin"))).expect("inflated dump");
        assert_eq!(
            packet.data, java,
            "{name}: inflated array differs from readPacketData's"
        );
        let raw = raw_column(primary, add, sky, ground_up, x * 31 + z);
        assert_eq!(
            &packet.data[..raw.len()],
            &raw[..],
            "{name}: inflated content"
        );
        assert_eq!(encode_frame(&packet), frame.bytes, "{name}: re-encode");
        // The client-side split reads the same regions Chunk.fillChunk would.
        let column = packet.column(sky).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(column.sections.len(), primary.count_ones() as usize);
        if let Some(first) = column.sections.first() {
            assert_eq!(u16::from(first.blocks[1]), (7 + x * 31 + z) as u8 as u16);
        }
        assert_eq!(column.biomes.is_some(), ground_up);
    }

    let frame = &frames["s26_bulk"];
    let mut buf = body(frame);
    let bulk = MapChunkBulk::read(&mut buf).expect("s26 decodes");
    assert!(buf.is_empty());
    assert!(bulk.has_sky_light);
    let java = fs::read(dir.join("s26_bulk.inflated.bin")).expect("inflated dump");
    let ours: Vec<u8> = bulk.columns.iter().flat_map(|c| c.data.clone()).collect();
    assert_eq!(
        ours, java,
        "s26: column slices differ from readPacketData's"
    );
    let headers: Vec<_> = bulk
        .columns
        .iter()
        .map(|c| (c.chunk_x, c.chunk_z, c.primary_bit_mask, c.add_bit_mask))
        .collect();
    assert_eq!(
        headers,
        vec![(1, 2, 0x0003, 0x0000), (-1, -2, 0x0008, 0x0008)]
    );
    assert_eq!(bulk.columns[0].data, raw_column(0x0003, 0, true, true, 0));
    assert_eq!(
        bulk.columns[1].data,
        raw_column(0x0008, 0x0008, true, true, 1)
    );
    assert_eq!(encode_frame(&bulk), frame.bytes, "s26: re-encode");
}

// ---- decode vectors -----------------------------------------------------------------------

/// Vectors whose Java re-encode is not compared: `C17PacketCustomPayload` keeps the raw length it
/// read as a separate field and writes it back (`ffff`, `7fff`) with no payload; the model keeps
/// only the payload. Only a server reading a client's packet sees this.
const REENCODE_EXEMPT: [&str; 2] = ["c17_negative_length", "c17_length_32767"];

#[test]
fn decode_vectors_follow_javas_verdict_and_re_encode() {
    let manifest = fs::read_to_string(fixtures_dir().join("decode.tsv")).expect("decode.tsv");
    let mut checked = 0;
    for line in manifest.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let (name, dir, id, body, verdict, reencoded) = (
            cols[0],
            cols[2],
            parse_id(cols[3]),
            unhex(cols[4]),
            cols[5],
            cols[6],
        );
        let mut buf = PacketBuffer::from_slice(&body);
        // Our re-encode must be encodable wherever Java's was (finish_write).
        let encode = |write: &dyn Fn(&mut PacketBuffer)| {
            let mut out = PacketBuffer::new();
            write(&mut out);
            out.finish_write().map(|()| out.as_slice().to_vec())
        };
        let ours: Result<Result<Vec<u8>, _>, String> = if dir == "S2C" {
            S2CPlay5::read(id, &mut buf)
                .map(|p| encode(&|out| p.write(out)))
                .map_err(|e| e.to_string())
        } else {
            C2SPlay5::read(id, &mut buf)
                .map(|p| encode(&|out| p.write(out)))
                .map_err(|e| e.to_string())
        };
        match verdict.split(':').next().unwrap() {
            "error" => assert!(ours.is_err(), "{name}: java threw ({verdict}), we decoded"),
            "ok" | "ok-noreencode" => {
                let bytes = ours.unwrap_or_else(|e| panic!("{name}: java accepted, we said {e}"));
                assert!(buf.is_empty(), "{name}: {} trailing bytes", buf.remaining());
                if verdict == "ok" && !REENCODE_EXEMPT.contains(&name) {
                    let bytes = bytes.unwrap_or_else(|e| {
                        panic!("{name}: java re-encodes, our write failed: {e}")
                    });
                    // Gzip NBT may differ only in its deflate body.
                    assert_wire_eq(name, &bytes, &unhex(reencoded));
                }
            }
            other => panic!("{name}: unexpected verdict {other}"),
        }
        checked += 1;
    }
    assert!(checked >= 41, "only {checked} decode vectors");
}
