//! W-CAP: every serverbound packet mc-rs can encode, diffed byte-for-byte against the real
//! 1.8.9 client's output.
//!
//! # Why this test is different from every other codec test in the tree
//!
//! The round-trip tests elsewhere prove that `write` and `read` agree *with each other*. That is
//! a real property, and it is also exactly the property a shared wrong constant preserves: encode
//! a field at the wrong offset and decode it back from the wrong offset and the test is green
//! forever. Pass 1 of the parity audit called this out as its own largest limitation -- it proved
//! self-consistency, not vanilla-agreement, for most packet bodies.
//!
//! The bytes on the right-hand side of these assertions were not written by anyone on this
//! project. They were produced by `parity/wire/java/WireOracle.java` running against
//! `MCP-919-main/bin/minecraft`, the compiled 1.8.9 client, and committed under
//! `parity/wire/fixtures/`. Regenerate them with `bash parity/tools/wire_capture.sh`; verify the
//! committed set still matches vanilla with `bash parity/tools/wire_capture.sh --check`.
//!
//! # What a fixture contains
//!
//! The **packet frame**: a VarInt packet id followed by `writePacketData`'s output. That is the
//! layer the plan asked for -- below compression and below encryption, both of which wrap this
//! frame without looking inside it. The id is included on purpose: a packet encoded perfectly
//! under the wrong id is still a packet the server rejects.
//!
//! # What this test does NOT prove
//!
//! That mc-rs *sends* these packets, at the right time, with the right field values. Every
//! finding in W-PICK, W-RCLICK, W-SWING, W-SETTINGS and friends is about the caller, not the
//! codec, and a green run here says nothing about any of them. What it does give is the thing
//! the plan says must exist before those fixes land: if a cadence fix accidentally changes the
//! bytes, this goes red.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use mc_math::{BlockPos, Facing};
use mc_nbt::{Nbt, NbtCompound};
use mc_protocol::handshake::{Handshake, NextState};
use mc_protocol::item::ItemStack;
use mc_protocol::login::{EncryptionResponse, LoginStart};
use mc_protocol::packet::Packet;
use mc_protocol::play::c2s::*;
use mc_protocol::PacketBuffer;
use uuid::Uuid;

/// Locate `parity/wire/fixtures` by walking up from this crate.
///
/// A missing fixture directory is a hard failure, never a skip. A parity test that quietly
/// passes when its oracle is absent is the same defect shape the audit catalogued 27 times:
/// structurally incapable of failing.
fn fixtures_dir() -> PathBuf {
    let mut dir: &Path = Path::new(env!("CARGO_MANIFEST_DIR"));

    loop {
        let candidate = dir.join("parity").join("wire").join("fixtures");

        if candidate.join("manifest.tsv").is_file() {
            return candidate;
        }

        dir = match dir.parent() {
            Some(parent) => parent,
            None => panic!(
                "parity/wire/fixtures/manifest.tsv not found above {}. \
                 Run: bash parity/tools/wire_capture.sh",
                env!("CARGO_MANIFEST_DIR")
            ),
        };
    }
}

/// One row of the oracle's manifest.
struct Fixture {
    bytes: Vec<u8>,
    hex: String,
    note: String,
}

fn load_fixtures() -> BTreeMap<String, Fixture> {
    let dir = fixtures_dir();
    let manifest = fs::read_to_string(dir.join("manifest.tsv")).expect("read manifest.tsv");
    let mut out = BTreeMap::new();

    for (lineno, line) in manifest.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }

        let cols: Vec<&str> = line.split('\t').collect();
        assert!(
            cols.len() >= 7,
            "manifest.tsv line {} has {} columns, expected 7",
            lineno + 1,
            cols.len()
        );

        let name = cols[0].to_string();
        let bytes = fs::read(dir.join(format!("{name}.bin")))
            .unwrap_or_else(|e| panic!("read fixture {name}.bin: {e}"));

        out.insert(
            name,
            Fixture {
                bytes,
                hex: cols[4].to_string(),
                note: cols[6].to_string(),
            },
        );
    }

    assert!(!out.is_empty(), "manifest.tsv declared no fixtures");
    out
}

/// Encode one packet the way the frame codec does: VarInt id, then the body.
fn frame<P: Packet>(packet: &P) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    buf.write_varint(P::ID);
    packet.write(&mut buf);
    buf.as_slice().to_vec()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The item stack the oracle used for the NBT-carrying fixtures.
///
/// The key ORDER matters and is not ours to choose. Vanilla serialises an `NBTTagCompound` by
/// iterating a `java.util.HashMap`, so the order on the wire is neither insertion order nor
/// sorted order -- it is whatever the JDK's string hashing produced. The oracle observed
/// `RepairCost` before `display`, and `Lore` before `Name`, even though the Java code inserts
/// them the other way round in both cases.
///
/// `mc_nbt::NbtCompound` keeps entries in a `Vec`, so it writes back exactly the order it was
/// given. That is the right behaviour for the case that actually happens in a session -- the
/// client echoes back a stack the SERVER sent, and preserving the received order reproduces the
/// server's bytes exactly. It also means byte-identity here is achievable only by inserting in
/// vanilla's observed order, which is what this function does.
fn nbt_sword() -> NbtCompound {
    let lore = vec![Nbt::String("line one".to_string())];

    let mut display = NbtCompound::new();
    display.insert("Lore", Nbt::list(lore));
    display.insert("Name", Nbt::String("Parity Blade".to_string()));

    let mut root = NbtCompound::new();
    root.insert("RepairCost", Nbt::Int(7));
    root.insert("display", Nbt::Compound(display));
    root
}

fn stone_64() -> ItemStack {
    ItemStack {
        item_id: 1,
        count: 64,
        damage: 0,
        nbt: None,
    }
}

fn sword_plain() -> ItemStack {
    ItemStack {
        item_id: 276,
        count: 1,
        damage: 0,
        nbt: None,
    }
}

/// Every case, keyed by the same name the oracle used, split across a few functions only to
/// keep each one readable.
///
/// Adding a case here without adding it to `WireOracle.java` (or the reverse) fails
/// `fixture_coverage_is_complete`, which is deliberate: a fixture nobody checks and a check with
/// no fixture are both worse than nothing.
fn cases() -> Vec<(&'static str, Vec<u8>)> {
    let mut all = cases_login_and_movement();
    all.extend(cases_interaction());
    all.extend(cases_windows());
    all.extend(cases_settings());
    all.extend(cases_status_and_payload());
    all
}

fn cases_login_and_movement() -> Vec<(&'static str, Vec<u8>)> {
    let mut c: Vec<(&'static str, Vec<u8>)> = Vec::with_capacity(32);

    // ---- HANDSHAKING and LOGIN -----------------------------------------------------------
    c.push((
        "hs_login",
        frame(&Handshake {
            protocol_version: 47,
            server_address: "mc.example.com".to_string(),
            server_port: 25565,
            next_state: NextState::Login,
        }),
    ));
    c.push((
        "hs_status",
        frame(&Handshake {
            protocol_version: 47,
            server_address: "localhost".to_string(),
            server_port: 25565,
            next_state: NextState::Status,
        }),
    ));
    c.push((
        "hs_login_forwarded",
        frame(&Handshake {
            protocol_version: 47,
            server_address: "host.example\u{0}127.0.0.1\u{0}f84c6a79-0a4e-45e0-879b-cd49ebd4c4e2"
                .to_string(),
            server_port: 25565,
            next_state: NextState::Login,
        }),
    ));
    c.push((
        "login_start",
        frame(&LoginStart {
            username: "Parity_Tester".to_string(),
        }),
    ));
    c.push((
        "login_encryption_response",
        frame(&EncryptionResponse {
            shared_secret_encrypted: vec![0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF, 0x10, 0x20],
            verify_token_encrypted: vec![0xDE, 0xAD, 0xBE, 0xEF],
        }),
    ));

    // ---- keep-alive and chat -------------------------------------------------------------
    c.push(("keepalive_zero", frame(&KeepAlive { key: 0 })));
    c.push(("keepalive_small", frame(&KeepAlive { key: 127 })));
    c.push(("keepalive_two_byte", frame(&KeepAlive { key: 128 })));
    c.push(("keepalive_max", frame(&KeepAlive { key: i32::MAX })));
    c.push(("keepalive_negative", frame(&KeepAlive { key: -1 })));
    c.push(("keepalive_min", frame(&KeepAlive { key: i32::MIN })));

    c.push((
        "chat_ascii",
        frame(&ChatMessage {
            message: "hello world".to_string(),
        }),
    ));
    c.push((
        "chat_command",
        frame(&ChatMessage {
            message: "/gamemode creative Parity_Tester".to_string(),
        }),
    ));
    c.push((
        "chat_unicode",
        frame(&ChatMessage {
            message: "caf\u{e9} \u{2666} \u{a7}c red \u{1f600}".to_string(),
        }),
    ));
    c.push((
        "chat_100_chars",
        frame(&ChatMessage {
            message: "0123456789".repeat(10),
        }),
    ));

    // ---- use entity ----------------------------------------------------------------------
    c.push((
        "useentity_attack",
        frame(&UseEntity {
            entity_id: 1,
            action: UseEntityAction::Attack,
        }),
    ));
    c.push((
        "useentity_interact_max_id",
        frame(&UseEntity {
            entity_id: i32::MAX,
            action: UseEntityAction::Interact,
        }),
    ));
    c.push((
        "useentity_interact_at",
        frame(&UseEntity {
            entity_id: 300,
            action: UseEntityAction::InteractAt {
                target_x: 0.5,
                target_y: 1.25,
                target_z: -0.75,
            },
        }),
    ));

    // ---- movement --------------------------------------------------------------------------
    c.push(("player_onground", frame(&Player { on_ground: true })));
    c.push(("player_airborne", frame(&Player { on_ground: false })));
    c.push((
        "playerposition",
        frame(&PlayerPosition {
            x: 8.5,
            y: 64.0,
            z: -32.25,
            on_ground: true,
        }),
    ));
    c.push((
        "playerposition_extreme",
        frame(&PlayerPosition {
            x: -29999999.5,
            y: 255.99609375,
            z: 29999999.5,
            on_ground: false,
        }),
    ));
    c.push((
        "playerposition_negzero",
        frame(&PlayerPosition {
            x: -0.0,
            y: 0.0,
            z: -0.0,
            on_ground: true,
        }),
    ));
    c.push((
        "playerlook",
        frame(&PlayerLook {
            yaw: -179.5,
            pitch: 90.0,
            on_ground: true,
        }),
    ));
    c.push((
        "playerlook_wrapped",
        frame(&PlayerLook {
            yaw: 720.25,
            pitch: -90.0,
            on_ground: false,
        }),
    ));
    c.push((
        "playerposlook",
        frame(&PlayerPositionAndLook {
            x: 8.5,
            y: 64.0,
            z: -32.25,
            yaw: -179.5,
            pitch: 22.5,
            on_ground: true,
        }),
    ));

    c
}

fn cases_interaction() -> Vec<(&'static str, Vec<u8>)> {
    let mut c: Vec<(&'static str, Vec<u8>)> = Vec::with_capacity(32);

    // ---- digging ---------------------------------------------------------------------------
    c.push((
        "digging_start",
        frame(&PlayerDigging {
            status: PlayerDiggingStatus::StartDestroyBlock,
            position: BlockPos::new(10, 64, -20),
            face: Facing::Up,
        }),
    ));
    c.push((
        "digging_abort",
        frame(&PlayerDigging {
            status: PlayerDiggingStatus::AbortDestroyBlock,
            position: BlockPos::new(10, 64, -20),
            face: Facing::North,
        }),
    ));
    c.push((
        "digging_stop",
        frame(&PlayerDigging {
            status: PlayerDiggingStatus::StopDestroyBlock,
            position: BlockPos::new(10, 64, -20),
            face: Facing::East,
        }),
    ));
    c.push((
        "digging_drop_item",
        frame(&PlayerDigging {
            status: PlayerDiggingStatus::DropItem,
            position: BlockPos::ORIGIN,
            face: Facing::Down,
        }),
    ));
    c.push((
        "digging_drop_all",
        frame(&PlayerDigging {
            status: PlayerDiggingStatus::DropAllItems,
            position: BlockPos::ORIGIN,
            face: Facing::Down,
        }),
    ));
    c.push((
        "digging_release_use",
        frame(&PlayerDigging {
            status: PlayerDiggingStatus::ReleaseUseItem,
            position: BlockPos::ORIGIN,
            face: Facing::Down,
        }),
    ));
    c.push((
        "digging_negative_pos",
        frame(&PlayerDigging {
            status: PlayerDiggingStatus::StartDestroyBlock,
            position: BlockPos::new(-1, -1, -1),
            face: Facing::Down,
        }),
    ));
    c.push((
        "digging_far_pos",
        frame(&PlayerDigging {
            status: PlayerDiggingStatus::StartDestroyBlock,
            position: BlockPos::new(-33554432, 0, 33554431),
            face: Facing::Down,
        }),
    ));

    // ---- placement -------------------------------------------------------------------------
    c.push((
        "place_on_block",
        frame(&PlayerBlockPlacement {
            position: BlockPos::new(10, 64, -20),
            placed_block_direction: 1,
            stack: Some(stone_64()),
            cursor_x: 8,
            cursor_y: 8,
            cursor_z: 8,
        }),
    ));
    c.push((
        "place_untargeted",
        frame(&PlayerBlockPlacement {
            position: BlockPos::new(-1, -1, -1),
            placed_block_direction: 255,
            stack: Some(stone_64()),
            cursor_x: 0,
            cursor_y: 0,
            cursor_z: 0,
        }),
    ));
    c.push((
        "place_untargeted_empty_hand",
        frame(&PlayerBlockPlacement {
            position: BlockPos::new(-1, -1, -1),
            placed_block_direction: 255,
            stack: None,
            cursor_x: 0,
            cursor_y: 0,
            cursor_z: 0,
        }),
    ));
    c.push((
        "place_cursor_extremes",
        frame(&PlayerBlockPlacement {
            position: BlockPos::ORIGIN,
            placed_block_direction: 0,
            stack: Some(stone_64()),
            cursor_x: 16,
            cursor_y: 0,
            cursor_z: 15,
        }),
    ));
    c.push((
        "place_stack_with_nbt",
        frame(&PlayerBlockPlacement {
            position: BlockPos::new(1, 2, 3),
            placed_block_direction: 2,
            stack: Some(ItemStack {
                item_id: 276,
                count: 1,
                damage: 42,
                nbt: Some(nbt_sword()),
            }),
            cursor_x: 4,
            cursor_y: 12,
            cursor_z: 8,
        }),
    ));
    c.push((
        "place_stack_nbt_nondamageable",
        frame(&PlayerBlockPlacement {
            position: BlockPos::new(1, 2, 3),
            placed_block_direction: 2,
            stack: Some(ItemStack {
                item_id: 1,
                count: 1,
                damage: 0,
                nbt: Some(nbt_sword()),
            }),
            cursor_x: 4,
            cursor_y: 12,
            cursor_z: 8,
        }),
    ));

    // ---- held item, animation, entity action, steering ---------------------------------------
    c.push(("helditem_slot0", frame(&HeldItemChange { slot_id: 0 })));
    c.push(("helditem_slot8", frame(&HeldItemChange { slot_id: 8 })));
    c.push(("animation", frame(&Animation)));

    c.push((
        "entityaction_start_sprinting",
        frame(&EntityAction {
            entity_id: 1,
            action: EntityActionType::StartSprinting,
            aux_data: 0,
        }),
    ));
    c.push((
        "entityaction_stop_sneaking",
        frame(&EntityAction {
            entity_id: 1,
            action: EntityActionType::StopSneaking,
            aux_data: 0,
        }),
    ));
    c.push((
        "entityaction_riding_jump",
        frame(&EntityAction {
            entity_id: 1,
            action: EntityActionType::RidingJump,
            aux_data: 100,
        }),
    ));
    c.push((
        "entityaction_open_inventory",
        frame(&EntityAction {
            entity_id: 1,
            action: EntityActionType::OpenInventory,
            aux_data: 0,
        }),
    ));

    c.push((
        "input_forward",
        frame(&SteerVehicle {
            sideways: 0.0,
            forward: 0.98,
            flags: 0,
        }),
    ));
    c.push((
        "input_all_flags",
        frame(&SteerVehicle {
            sideways: -0.98,
            forward: 0.98,
            flags: 3,
        }),
    ));

    c
}

fn cases_windows() -> Vec<(&'static str, Vec<u8>)> {
    let mut c: Vec<(&'static str, Vec<u8>)> = Vec::with_capacity(32);

    c.push(("closewindow_player", frame(&CloseWindow { window_id: 0 })));
    c.push(("closewindow_chest", frame(&CloseWindow { window_id: 3 })));

    c.push((
        "clickwindow_left",
        frame(&ClickWindow {
            window_id: 1,
            slot_id: 36,
            used_button: 0,
            action_number: 1,
            mode: 0,
            clicked_item: Some(sword_plain()),
        }),
    ));
    c.push((
        "clickwindow_shift_right",
        frame(&ClickWindow {
            window_id: 1,
            slot_id: 0,
            used_button: 1,
            action_number: 2,
            mode: 1,
            clicked_item: None,
        }),
    ));
    c.push((
        "clickwindow_outside_drop",
        frame(&ClickWindow {
            window_id: 0,
            slot_id: -999,
            used_button: 0,
            action_number: 3,
            mode: 0,
            clicked_item: None,
        }),
    ));
    c.push((
        "clickwindow_number_key",
        frame(&ClickWindow {
            window_id: 2,
            slot_id: 30,
            used_button: 4,
            action_number: 4,
            mode: 2,
            clicked_item: Some(sword_plain()),
        }),
    ));

    c.push((
        "confirmtransaction_accept",
        frame(&ConfirmTransaction {
            window_id: 0,
            action_id: 1,
            accepted: true,
        }),
    ));
    c.push((
        "confirmtransaction_reject",
        frame(&ConfirmTransaction {
            window_id: 3,
            action_id: -4,
            accepted: false,
        }),
    ));

    c.push((
        "creative_set_slot",
        frame(&CreativeInventoryAction {
            slot_id: 36,
            stack: Some(sword_plain()),
        }),
    ));
    c.push((
        "creative_clear_slot",
        frame(&CreativeInventoryAction {
            slot_id: 0,
            stack: None,
        }),
    ));
    c.push((
        "creative_drop_outside",
        frame(&CreativeInventoryAction {
            slot_id: -1,
            stack: Some(sword_plain()),
        }),
    ));

    c.push((
        "enchantitem",
        frame(&EnchantItem {
            window_id: 3,
            button: 2,
        }),
    ));

    c
}

fn cases_settings() -> Vec<(&'static str, Vec<u8>)> {
    let mut c: Vec<(&'static str, Vec<u8>)> = Vec::with_capacity(32);

    // Vanilla does NOT send raw sign text. Each line goes through
    // IChatComponent.Serializer.componentToJson, and gson collapses an unstyled component to a
    // bare JSON string -- `"line one"`, quotes included -- rather than `{"text":"line one"}`.
    // mc-rs's UpdateSign carries already-serialised JSON, so the codec's job is just to write
    // these through; producing them is the caller's, and the shapes below are what it has to
    // produce.
    c.push((
        "updatesign",
        frame(&UpdateSign {
            position: BlockPos::new(12, 70, -3),
            lines: [
                "\"line one\"".to_string(),
                "\"\"".to_string(),
                "{\"bold\":true,\"color\":\"red\",\"text\":\"styled\"}".to_string(),
                "\"caf\u{e9} \u{2666}\"".to_string(),
            ],
        }),
    ));

    c.push((
        "abilities_creative_flying",
        frame(&PlayerAbilities {
            invulnerable: true,
            flying: true,
            allow_flying: true,
            creative_mode: true,
            fly_speed: 0.05,
            walk_speed: 0.1,
        }),
    ));
    c.push((
        "abilities_survival",
        frame(&PlayerAbilities {
            invulnerable: false,
            flying: false,
            allow_flying: false,
            creative_mode: false,
            fly_speed: 0.05,
            walk_speed: 0.1,
        }),
    ));

    c.push((
        "tabcomplete_plain",
        frame(&TabComplete {
            message: "/gamemo".to_string(),
            target_block: None,
        }),
    ));
    c.push((
        "tabcomplete_with_block",
        frame(&TabComplete {
            message: "/setblock ".to_string(),
            target_block: Some(BlockPos::new(-8, 4, 16)),
        }),
    ));

    c.push((
        "clientsettings_default",
        frame(&ClientSettings {
            locale: "en_US".to_string(),
            view_distance: 12,
            chat_visibility: ChatVisibility::Full,
            chat_colors: true,
            displayed_skin_parts: 0x7F,
        }),
    ));
    c.push((
        "clientsettings_hidden",
        frame(&ClientSettings {
            locale: "de_DE".to_string(),
            view_distance: 2,
            chat_visibility: ChatVisibility::Hidden,
            chat_colors: false,
            displayed_skin_parts: 0,
        }),
    ));
    c.push((
        "clientsettings_system",
        frame(&ClientSettings {
            locale: "fr_FR".to_string(),
            view_distance: 32,
            chat_visibility: ChatVisibility::System,
            chat_colors: true,
            displayed_skin_parts: 0x2A,
        }),
    ));

    c
}

fn cases_status_and_payload() -> Vec<(&'static str, Vec<u8>)> {
    let mut c: Vec<(&'static str, Vec<u8>)> = Vec::with_capacity(32);

    c.push((
        "clientstatus_respawn",
        frame(&ClientStatus {
            action: ClientStatusAction::PerformRespawn,
        }),
    ));
    c.push((
        "clientstatus_request_stats",
        frame(&ClientStatus {
            action: ClientStatusAction::RequestStats,
        }),
    ));
    c.push((
        "clientstatus_achievements",
        frame(&ClientStatus {
            action: ClientStatusAction::OpenInventoryAchievement,
        }),
    ));

    // The payload is the remainder of the frame and carries no length prefix of its own -- an
    // encoder that writes one produces a packet the server mis-parses from the first byte.
    c.push((
        "pluginmessage_brand",
        frame(&PluginMessage {
            channel: "MC|Brand".to_string(),
            data: vec![0x07, b'v', b'a', b'n', b'i', b'l', b'l', b'a'],
        }),
    ));
    c.push((
        "pluginmessage_binary",
        frame(&PluginMessage {
            channel: "MC|AdvCmd".to_string(),
            data: vec![0x00, 0xFF, 0x7F, 0x80],
        }),
    ));
    c.push((
        "pluginmessage_empty",
        frame(&PluginMessage {
            channel: "MC|PickItem".to_string(),
            data: Vec::new(),
        }),
    ));

    c.push((
        "spectate",
        frame(&Spectate {
            target: Uuid::parse_str("f84c6a79-0a4e-45e0-879b-cd49ebd4c4e2").unwrap(),
        }),
    ));

    let hash = "0123456789abcdef0123456789abcdef01234567";
    c.push((
        "resourcepack_loaded",
        frame(&ResourcePackStatus {
            hash: hash.to_string(),
            result: ResourcePackStatusResult::SuccessfullyLoaded,
        }),
    ));
    c.push((
        "resourcepack_declined",
        frame(&ResourcePackStatus {
            hash: hash.to_string(),
            result: ResourcePackStatusResult::Declined,
        }),
    ));
    c.push((
        "resourcepack_failed",
        frame(&ResourcePackStatus {
            hash: hash.to_string(),
            result: ResourcePackStatusResult::FailedDownload,
        }),
    ));
    c.push((
        "resourcepack_accepted",
        frame(&ResourcePackStatus {
            hash: hash.to_string(),
            result: ResourcePackStatusResult::Accepted,
        }),
    ));

    c
}

// ---- the tests -------------------------------------------------------------------------------

/// The whole point: mc-rs's bytes, vanilla's bytes, no third opinion.
///
/// Reports every mismatch in one run rather than stopping at the first. A codec regression
/// usually breaks a family of packets at once, and seeing the family is what identifies the
/// cause.
#[test]
fn c2s_frames_match_vanilla_byte_for_byte() {
    let fixtures = load_fixtures();
    let mut failures: Vec<String> = Vec::new();

    for (name, actual) in cases() {
        let expected = match fixtures.get(name) {
            Some(f) => f,
            // Coverage is a separate test; skip quietly here so its message is the one you read.
            None => continue,
        };

        if actual == expected.bytes {
            continue;
        }

        let offset = actual
            .iter()
            .zip(expected.bytes.iter())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| actual.len().min(expected.bytes.len()));

        failures.push(format!(
            "\n  {name}\n    note:    {}\n    vanilla: {} ({} bytes)\n    mc-rs:   {} ({} bytes)\n    \
             first difference at byte {offset}",
            expected.note,
            expected.hex,
            expected.bytes.len(),
            hex(&actual),
            actual.len(),
        ));
    }

    assert!(
        failures.is_empty(),
        "{} of {} serverbound frames differ from the real 1.8.9 client:{}",
        failures.len(),
        fixtures.len(),
        failures.join("")
    );
}

/// Neither side may drift ahead of the other.
#[test]
fn fixture_coverage_is_complete() {
    let fixtures = load_fixtures();
    let cases = cases();

    let mut unchecked: Vec<&str> = fixtures
        .keys()
        .map(String::as_str)
        .filter(|name| !cases.iter().any(|(c, _)| c == name))
        .collect();
    unchecked.sort_unstable();

    let mut unfixtured: Vec<&str> = cases
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !fixtures.contains_key(*name))
        .collect();
    unfixtured.sort_unstable();

    assert!(
        unchecked.is_empty() && unfixtured.is_empty(),
        "fixture/case mismatch.\n  fixtures with no case (nothing checks these): {unchecked:?}\n  \
         cases with no fixture (nothing to compare against): {unfixtured:?}"
    );
}

/// The fixtures are vanilla's output. If a `.bin` no longer matches the hex the oracle recorded
/// beside it, someone edited one by hand, and every assertion above is measuring the wrong thing.
#[test]
fn fixtures_are_internally_consistent() {
    for (name, fixture) in load_fixtures() {
        assert_eq!(
            hex(&fixture.bytes),
            fixture.hex,
            "{name}.bin does not match the hex recorded in manifest.tsv -- \
             regenerate with parity/tools/wire_capture.sh, never edit fixtures by hand"
        );
    }
}
