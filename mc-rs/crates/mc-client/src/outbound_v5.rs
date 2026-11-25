//! Minecraft 1.7.10 (protocol 5): the serverbound packets whose content differs from 1.8.9's.
//!
//! A 1.7.10 connection sends every packet that means the same in both versions as the 1.8.9
//! packet, which the network task converts with [`C2SPlay5::from_v47`]. The packets here carry
//! data the 1.8.9 structs lack, or a different payload format, so the 1.7.10 paths build them
//! directly:
//!
//! | Packet | 1.7.10 |
//! |---|---|
//! | C0B sprint and sneak edges | action ids 4/5 and 1/2, before the movement packet |
//! | C04, C06 | the feet `y` and the eye `stance` (`posY`) |
//! | C07 drop | status 4 (3 with Ctrl) at `(0, 0, 0)`, face 0 |
//! | C12 | four plain lines |
//! | C14 | the text alone, no looked-at block |
//! | C15 | the client's difficulty option and show-cape, where 1.8.9 has skin parts |
//! | C17 `MC\|Brand`, `MC\|ItemName` | raw UTF-8, no length prefix |
//! | C17 `MC\|TrSel`, `MC\|Beacon` | the Netty buffer's whole backing array ([`backing_array`]) |
//! | C17 `MC\|AdvCdm`, `MC\|BEdit`, `MC\|BSign` | the same; a signed book's pages as typed |
//!
//! java17: `client/entity/EntityClientPlayerMP.java`, `client/settings/GameSettings.java`,
//! `client/network/NetHandlerPlayClient.java` and the screens named on each builder.

use mc_math::BlockPos;
use mc_nbt::Nbt;
use mc_physics::v5::OutboundV5;
use mc_protocol::play::c2s as v47;
use mc_protocol::v5::c2s;
use mc_protocol::v5::{C2SPlay5, PacketBufferV5};
use mc_protocol::{ItemStack, PacketBuffer};
use mc_ui::font::component::ChatComponent;
use mc_ui::screens::sign::CommandBlockTarget;

use super::OutboundState;
use crate::session::JoinSettings;

/// The most bytes a payload written into a Netty buffer may hold and still be sent.
///
/// `C17PacketCustomPayload(String, ByteBuf)` keeps `buf.array()`, and its `(String, byte[])`
/// constructor throws for `length >= 32767` (java17
/// `network/play/client/C17PacketCustomPayload.java:19-38`). More than 16384 written bytes grow
/// the buffer to 32768, so the constructor throws and the screen's `catch` sends nothing.
pub const MAX_BUFFER_PAYLOAD: usize = 16_384;

/// `Unpooled.buffer()`'s initial capacity: Netty 4.0.10's `heapBuffer(256, Integer.MAX_VALUE)`.
const INITIAL_BUFFER_CAPACITY: usize = 256;

/// `ByteBuf.array()` of an `Unpooled.buffer()` holding `written`: the written bytes, then zeros
/// up to the buffer's capacity. The capacity starts at 256 and each growth takes the smallest
/// power of two that fits (`AbstractByteBuf.ensureWritable`'s `calculateNewCapacity`), so it is
/// `max(256, written.len().next_power_of_two())`. `None` when vanilla sends nothing
/// ([`MAX_BUFFER_PAYLOAD`]).
#[must_use]
pub fn backing_array(written: &[u8]) -> Option<Vec<u8>> {
    if written.len() > MAX_BUFFER_PAYLOAD {
        return None;
    }
    let capacity = written
        .len()
        .next_power_of_two()
        .max(INITIAL_BUFFER_CAPACITY);
    let mut array = Vec::with_capacity(capacity);
    array.extend_from_slice(written);
    array.resize(capacity, 0);
    Some(array)
}

impl OutboundState {
    /// Everything `EntityClientPlayerMP.sendMotionUpdates` sends this tick: the sprint edge
    /// (action 4 start, 5 stop), the sneak edge (1 start, 2 stop), then the movement packet
    /// `movement` names (java17 `client/entity/EntityClientPlayerMP.java:86-167`).
    ///
    /// `sprinting` is the watched sprint flag and `sneaking` is `isSneaking()`, the sneak key
    /// and not sleeping, as in 1.8.9; the edge fields are the ones [`Self::walking_update`]
    /// keeps (`wasSprinting`/`wasSneaking`, reset the same way when a respawn builds a new
    /// player). There is no view-entity gate in 1.7.10.
    pub fn walking_update_v5(
        &mut self,
        entity_id: i32,
        movement: OutboundV5,
        sprinting: bool,
        sneaking: bool,
        out: &mut Vec<C2SPlay5>,
    ) {
        if sprinting != self.server_sprinting {
            out.push(entity_action(entity_id, if sprinting { 4 } else { 5 }));
            self.server_sprinting = sprinting;
        }
        if sneaking != self.server_sneaking {
            out.push(entity_action(entity_id, if sneaking { 1 } else { 2 }));
            self.server_sneaking = sneaking;
        }
        out.push(movement_packet(movement));
    }
}

/// The C03 family packet `sendMotionUpdates` chose: C04 and C06 carry the box bottom as `y`
/// and the eye `posY` as the stance (java17 `client/entity/EntityClientPlayerMP.java:135-147`).
#[must_use]
pub fn movement_packet(movement: OutboundV5) -> C2SPlay5 {
    match movement {
        OutboundV5::Ground { on_ground } => C2SPlay5::Player(c2s::Player { on_ground }),
        OutboundV5::Position {
            x,
            y,
            stance,
            z,
            on_ground,
        } => C2SPlay5::PlayerPosition(c2s::PlayerPosition {
            x,
            y,
            stance,
            z,
            on_ground,
        }),
        OutboundV5::Look {
            yaw,
            pitch,
            on_ground,
        } => C2SPlay5::PlayerLook(c2s::PlayerLook {
            yaw,
            pitch,
            on_ground,
        }),
        OutboundV5::PositionLook {
            x,
            y,
            stance,
            z,
            yaw,
            pitch,
            on_ground,
        } => C2SPlay5::PlayerPositionAndLook(c2s::PlayerPositionAndLook {
            x,
            y,
            stance,
            z,
            yaw,
            pitch,
            on_ground,
        }),
    }
}

/// `C0BPacketEntityAction(this, action)`: the player's id, a 1-based action, aux 0.
fn entity_action(entity_id: i32, action_id: i8) -> C2SPlay5 {
    C2SPlay5::EntityAction(c2s::EntityAction {
        entity_id,
        action_id,
        aux_data: 0,
    })
}

/// The drop key: `EntityClientPlayerMP.dropOneItem` sends `C07PacketPlayerDigging(3 or 4, 0, 0,
/// 0, 0)`, 3 being the whole stack (java17 `client/entity/EntityClientPlayerMP.java:169-174`).
#[must_use]
pub fn drop_item(whole_stack: bool) -> C2SPlay5 {
    C2SPlay5::PlayerDigging(c2s::PlayerDigging {
        status: if whole_stack { 3 } else { 4 },
        x: 0,
        y: 0,
        z: 0,
        face: 0,
    })
}

/// `GameSettings.sendSettingsToServer`'s `C15PacketClientSettings(language,
/// renderDistanceChunks, chatVisibility, chatColours, difficulty, showCape)` (java17
/// `client/settings/GameSettings.java:1163-1169`). The difficulty is the client's own option,
/// which 1.7.10's `handleJoinGame` never overwrites from the server. The view distance is the
/// stored one as it is: 1.7.10 limits only its slider to 2..=16, and keeps a larger value read
/// from the options file (`:752`) and sends it.
#[must_use]
pub fn client_settings(settings: &JoinSettings) -> c2s::ClientSettings {
    c2s::ClientSettings {
        locale: settings.locale.clone(),
        view_distance: settings.view_distance,
        // java17 `EntityPlayer.EnumChatVisibility`: FULL 0, SYSTEM 1, HIDDEN 2.
        chat_visibility: match settings.chat_visibility {
            v47::ChatVisibility::Full => 0,
            v47::ChatVisibility::System => 1,
            v47::ChatVisibility::Hidden => 2,
        },
        chat_colors: settings.chat_colors,
        difficulty: settings.difficulty,
        show_cape: settings.show_cape,
    }
}

/// `MC|Brand` as `handleJoinGame` sends it: `getClientModName().getBytes(UTF_8)`, raw, with no
/// length prefix (java17 `client/network/NetHandlerPlayClient.java:241`).
#[must_use]
pub fn brand(brand: &str) -> C2SPlay5 {
    plugin_message("MC|Brand", brand.as_bytes().to_vec())
}

/// `MC|ItemName` as `GuiRepair.renameItem` sends it: the name's raw UTF-8 bytes, empty for "no
/// name" (java17 `client/gui/GuiRepair.java:120-132`).
#[must_use]
pub fn anvil_name(name: &str) -> C2SPlay5 {
    plugin_message("MC|ItemName", name.as_bytes().to_vec())
}

/// `MC|TrSel` as `GuiMerchant.actionPerformed` sends it: the recipe index written into an
/// `Unpooled.buffer()`, so 256 bytes (java17 `client/gui/GuiMerchant.java:86-100`).
#[must_use]
pub fn trade_select(index: i32) -> C2SPlay5 {
    let array = backing_array(&index.to_be_bytes()).expect("four bytes always fit");
    plugin_message("MC|TrSel", array)
}

/// `MC|Beacon` as `GuiBeacon.actionPerformed` (Done) sends it: the two effect ids in an
/// `Unpooled.buffer()`, so 256 bytes (java17 `client/gui/inventory/GuiBeacon.java:131-148`).
#[must_use]
pub fn beacon(primary: i32, secondary: i32) -> C2SPlay5 {
    let mut written = [0_u8; 8];
    written[..4].copy_from_slice(&primary.to_be_bytes());
    written[4..].copy_from_slice(&secondary.to_be_bytes());
    let array = backing_array(&written).expect("eight bytes always fit");
    plugin_message("MC|Beacon", array)
}

/// `MC|AdvCdm` as `GuiCommandBlock.actionPerformed` (Done) sends it: the logic's type (0 block,
/// 1 minecart), its position (the block's three `int`s or the minecart's entity id), the
/// command string, and no track-output flag (1.8.9 adds one), in a Netty buffer (java17
/// `client/gui/GuiCommandBlock.java:70-86`, `TileEntityCommandBlock.func_145757_a`,
/// `EntityMinecartCommandBlock.func_145757_a`). `None` when vanilla sends nothing: a command
/// over 32767 UTF-8 bytes (`writeStringToBuffer` throws) or a payload over
/// [`MAX_BUFFER_PAYLOAD`].
#[must_use]
pub fn command_block(target: CommandBlockTarget, command: &str) -> Option<C2SPlay5> {
    let mut written = PacketBuffer::new();
    match target {
        CommandBlockTarget::Block { position } => {
            written.write_u8(0);
            written.write_i32(position.x);
            written.write_i32(position.y);
            written.write_i32(position.z);
        }
        CommandBlockTarget::Minecart { entity_id } => {
            written.write_u8(1);
            written.write_i32(entity_id);
        }
    }
    written.write_string(command);
    written.finish_write().ok()?;
    Some(plugin_message(
        "MC|AdvCdm",
        backing_array(written.as_slice())?,
    ))
}

/// `MC|BEdit` or `MC|BSign` as `GuiScreenBook.sendBookToServer` sends it: the book stack in
/// protocol 5's slot format (gzip NBT) in a Netty buffer (java17
/// `client/gui/GuiScreenBook.java:128-185`). `None` when the payload is over
/// [`MAX_BUFFER_PAYLOAD`] and vanilla sends nothing ("Couldn't send book info"). A signed book's
/// pages must be the typed text ([`signed_book_pages_as_typed`]).
#[must_use]
pub fn book(channel: &str, stack: &ItemStack) -> Option<C2SPlay5> {
    let mut written = PacketBuffer::new();
    written.write_item_v5(Some(stack));
    written.finish_write().ok()?;
    Some(plugin_message(channel, backing_array(written.as_slice())?))
}

/// The signed book 1.7.10 sends: pages as typed. When signing, 1.8.9 turns every page into a
/// JSON text component (java18 `client/gui/GuiScreenBook.java:196-202`) and the book screen
/// builds that form; 1.7.10 keeps the plain strings
/// (java17 `client/gui/GuiScreenBook.java:158-166`). Each page the screen builds is a JSON string
/// literal of the typed text, which decodes back to exactly that text. `None` when a page is not
/// one: it would reach a 1.7.10 server as JSON, so the book is not sent.
#[must_use]
pub fn signed_book_pages_as_typed(stack: &ItemStack) -> Option<ItemStack> {
    let mut stack = stack.clone();
    let Some(tag) = stack.nbt.as_mut() else {
        return Some(stack);
    };
    let Some(pages) = tag.get_list("pages") else {
        return Some(stack);
    };
    if pages.is_empty() {
        return Some(stack);
    }
    let pages = pages
        .iter()
        .map(|page| {
            let typed = serde_json::from_str::<String>(page.as_str()?).ok()?;
            Some(Nbt::String(typed))
        })
        .collect::<Option<Vec<Nbt>>>()?;
    tag.insert("pages", Nbt::list(pages));
    Some(stack)
}

/// A container screen's plugin message (as the screen builds it, in 1.8.9's payload format)
/// in its 1.7.10 form: `MC|ItemName` raw ([`anvil_name`]), `MC|TrSel` and `MC|Beacon` as the
/// 256-byte backing array ([`trade_select`], [`beacon`]). `None` for any other channel or a
/// payload that is not exactly that format.
#[must_use]
pub fn container_plugin_message(message: &v47::PluginMessage) -> Option<C2SPlay5> {
    let mut payload = PacketBuffer::from_slice(&message.data);
    let packet = match message.channel.as_str() {
        "MC|ItemName" => anvil_name(&payload.read_string(32767).ok()?),
        "MC|TrSel" => trade_select(payload.read_i32().ok()?),
        "MC|Beacon" => {
            let primary = payload.read_i32().ok()?;
            beacon(primary, payload.read_i32().ok()?)
        }
        _ => return None,
    };
    payload.is_empty().then_some(packet)
}

/// `GuiEditSign.onGuiClosed`'s `C12PacketUpdateSign(x, y, z, signText)`: four plain lines
/// (java17 `client/gui/inventory/GuiEditSign.java:37-48`). `lines` are the sign editor's
/// serialized text components; 1.7.10 sends their text.
#[must_use]
pub fn update_sign(position: BlockPos, lines: &[String; 4]) -> C2SPlay5 {
    C2SPlay5::UpdateSign(c2s::UpdateSign {
        x: position.x,
        // java17: `writeShort(this.y)`.
        y: position.y as i16,
        z: position.z,
        lines: lines.clone().map(|json| {
            ChatComponent::from_json(&json).map_or(json, |component| component.to_string())
        }),
    })
}

/// `GuiChat.sendAutocompleteRequest`'s `C14PacketTabComplete(text)`: the text before the
/// cursor and nothing else (java17 `client/gui/GuiChat.java:333-340`).
#[must_use]
pub fn tab_complete(text: &str) -> C2SPlay5 {
    C2SPlay5::TabComplete(c2s::TabComplete {
        message: text.to_owned(),
    })
}

fn plugin_message(channel: &str, data: Vec<u8>) -> C2SPlay5 {
    C2SPlay5::PluginMessage(c2s::PluginMessage {
        channel: channel.to_owned(),
        data,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use mc_nbt::NbtCompound;
    use mc_protocol::packet::Packet;
    use mc_protocol::v5::NbtBudget;

    use super::*;

    /// The frame the wire fixtures hold: VarInt id, then `writePacketData`'s bytes.
    fn frame(packet: &C2SPlay5) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.finish_write().expect("encodable");
        buf.as_slice().to_vec()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn unhex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
            .collect()
    }

    /// `parity/wire/fixtures-v5`, the frames the real 1.7.10 classes wrote
    /// (`parity/tools/wire_capture_v5.sh`). A missing directory fails: a parity test that passes
    /// without its oracle cannot fail.
    fn wire_fixtures() -> BTreeMap<String, String> {
        let mut dir: &Path = Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest: PathBuf = loop {
            let candidate = dir.join("parity/wire/fixtures-v5/manifest.tsv");
            if candidate.is_file() {
                break candidate;
            }
            dir = dir.parent().unwrap_or_else(|| {
                panic!("parity/wire/fixtures-v5/manifest.tsv not found above the crate")
            });
        };
        let text = std::fs::read_to_string(&manifest).expect("read the manifest");
        text.lines()
            .skip(1)
            .map(|line| {
                let cells: Vec<&str> = line.split('\t').collect();
                (cells[0].to_owned(), cells[5].to_owned())
            })
            .collect()
    }

    fn assert_wire(fixtures: &BTreeMap<String, String>, name: &str, packet: &C2SPlay5) {
        let vanilla = fixtures
            .get(name)
            .unwrap_or_else(|| panic!("no fixture {name}"));
        assert_eq!(&hex(&frame(packet)), vanilla, "{name}: {packet:?}");
    }

    /// `javaref/v17_payload_ref.txt` (`V17PayloadRef.java`): name -> (sent, frame, raw NBT).
    fn payload_refs() -> BTreeMap<&'static str, (bool, &'static str, &'static str)> {
        include_str!("../javaref/v17_payload_ref.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| {
                let cells: Vec<&str> = line.split('\t').collect();
                (cells[0], (cells[1] == "1", cells[2], cells[3]))
            })
            .collect()
    }

    fn settings(
        locale: &str,
        view_distance: i8,
        chat_visibility: v47::ChatVisibility,
        chat_colors: bool,
        difficulty: u8,
        show_cape: bool,
    ) -> JoinSettings {
        JoinSettings {
            locale: locale.to_owned(),
            view_distance,
            chat_visibility,
            chat_colors,
            difficulty,
            show_cape,
            ..JoinSettings::default()
        }
    }

    #[test]
    fn movement_packets_carry_the_feet_y_and_the_eye_stance_as_1_7_10_writes_them() {
        let wire = wire_fixtures();
        let cases = [
            ("c03_ground", OutboundV5::Ground { on_ground: true }),
            ("c03_air", OutboundV5::Ground { on_ground: false }),
            (
                "c04_position",
                OutboundV5::Position {
                    x: 1.5,
                    y: 64.0,
                    stance: 65.62,
                    z: -2.5,
                    on_ground: true,
                },
            ),
            (
                "c04_position_negzero",
                OutboundV5::Position {
                    x: -0.0,
                    y: 70.0,
                    stance: 71.62,
                    z: -0.0,
                    on_ground: false,
                },
            ),
            (
                "c04_position_sneak",
                OutboundV5::Position {
                    x: 10.5,
                    y: 64.0,
                    stance: 64.0 + 1.54,
                    z: -3.5,
                    on_ground: true,
                },
            ),
            (
                "c05_look",
                OutboundV5::Look {
                    yaw: -90.5,
                    pitch: 45.25,
                    on_ground: false,
                },
            ),
            (
                "c06_poslook",
                OutboundV5::PositionLook {
                    x: 1.5,
                    y: 64.0,
                    stance: 65.62,
                    z: -2.5,
                    yaw: 90.0,
                    pitch: -10.0,
                    on_ground: true,
                },
            ),
            (
                "c06_poslook_far",
                OutboundV5::PositionLook {
                    x: 30_000_000.5,
                    y: 255.0,
                    stance: 255.0 + 1.62,
                    z: -30_000_000.5,
                    yaw: -0.0,
                    pitch: 90.0,
                    on_ground: false,
                },
            ),
        ];
        for (name, movement) in cases {
            assert_wire(&wire, name, &movement_packet(movement));
        }
        // Negative control: the same position without the stance is the 1.8.9 C04, which a
        // 1.7.10 server reads as `stance = z` and the next double as `z`.
        let v47 = mc_protocol::play::C2SPlay::PlayerPosition(v47::PlayerPosition {
            x: 1.5,
            y: 64.0,
            z: -2.5,
            on_ground: true,
        });
        let mut buf = PacketBuffer::new();
        buf.write_varint(v47.id());
        v47.write(&mut buf);
        assert_ne!(&hex(buf.as_slice()), &wire["c04_position"]);
    }

    #[test]
    fn sprint_then_sneak_edges_use_1_based_ids_and_precede_the_movement_packet() {
        let wire = wire_fixtures();
        let still = OutboundV5::Ground { on_ground: true };
        let mut state = OutboundState::new();
        let mut out = Vec::new();
        state.walking_update_v5(12345, still, true, true, &mut out);
        assert_eq!(out.len(), 3, "{out:?}");
        assert_wire(&wire, "c0b_action_4", &out[0]);
        assert_wire(&wire, "c0b_sneak", &out[1]);
        assert_wire(&wire, "c03_ground", &out[2]);

        // No edge, no C0B: only the movement packet.
        out.clear();
        state.walking_update_v5(12345, still, true, true, &mut out);
        assert_eq!(out, vec![movement_packet(still)]);

        out.clear();
        state.walking_update_v5(12345, still, false, false, &mut out);
        assert_wire(&wire, "c0b_action_5", &out[0]);
        assert_wire(&wire, "c0b_action_2", &out[1]);
        assert_eq!(out.len(), 3);

        // The edge fields are 1.8.9's: an edge reported by either version's builder is not
        // reported again by the other.
        out.clear();
        let mut v47_out = Vec::new();
        state.walking_edges(12345, true, false, &mut v47_out);
        state.walking_update_v5(12345, still, true, false, &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn the_drop_key_sends_status_4_or_3_at_the_origin_with_face_0() {
        let wire = wire_fixtures();
        assert_wire(&wire, "c07_drop", &drop_item(false));
        assert_wire(&wire, "c07_drop_all", &drop_item(true));
    }

    #[test]
    fn client_settings_carry_the_difficulty_and_show_cape() {
        let wire = wire_fixtures();
        use v47::ChatVisibility::{Full, Hidden, System};
        let cases = [
            ("c15_settings", settings("en_US", 12, Full, true, 2, true)),
            (
                "c15_settings_hidden",
                settings("de_DE", 2, Hidden, false, 0, false),
            ),
            (
                "c15_settings_system_hard",
                settings("en_US", 8, System, true, 3, false),
            ),
            (
                "c15_settings_easy",
                settings("en_GB", 16, Full, false, 1, true),
            ),
        ];
        for (name, join) in cases {
            let packet = C2SPlay5::ClientSettings(client_settings(&join));
            assert_wire(&wire, name, &packet);
        }
        // Negative control: 1.8.9's skin-part byte in place of difficulty and cape.
        let v47 = mc_protocol::play::C2SPlay::ClientSettings(v47::ClientSettings {
            locale: "en_US".into(),
            view_distance: 12,
            chat_visibility: Full,
            chat_colors: true,
            displayed_skin_parts: 0x7F,
        });
        let mut buf = PacketBuffer::new();
        buf.write_varint(v47.id());
        v47.write(&mut buf);
        assert_ne!(&hex(buf.as_slice()), &wire["c15_settings"]);
    }

    /// java17 sends the stored render distance, also one past its slider's 2..=16 (an options
    /// file 1.8.9 wrote): nothing here narrows it.
    #[test]
    fn the_stored_view_distance_goes_out_as_it_is() {
        use v47::ChatVisibility::Full;
        for view in [0, 2, 16, 17, 32] {
            let join = settings("en_GB", view, Full, false, 1, true);
            assert_eq!(client_settings(&join).view_distance, view);
        }
    }

    #[test]
    fn brand_and_anvil_names_are_raw_utf8() {
        let wire = wire_fixtures();
        assert_wire(&wire, "c17_brand_aero", &brand("Aero"));
        assert_wire(&wire, "c17_brand", &brand("vanilla"));
        assert_wire(&wire, "c17_itemname_ab", &anvil_name("ab"));
        assert_wire(&wire, "c17_itemname_empty", &anvil_name(""));
        // Negative control: 1.8.9's VarInt-prefixed brand.
        let mut prefixed = PacketBuffer::new();
        prefixed.write_string("Aero");
        let v47_brand = plugin_message("MC|Brand", prefixed.as_slice().to_vec());
        assert_ne!(hex(&frame(&v47_brand)), wire["c17_brand_aero"]);
    }

    #[test]
    fn container_payloads_convert_from_the_screens_1_8_9_form() {
        let wire = wire_fixtures();
        let mut name = PacketBuffer::new();
        name.write_string("ab");
        let anvil = v47::PluginMessage {
            channel: "MC|ItemName".into(),
            data: name.as_slice().to_vec(),
        };
        assert_wire(
            &wire,
            "c17_itemname_ab",
            &container_plugin_message(&anvil).expect("MC|ItemName"),
        );
        let mut empty = PacketBuffer::new();
        empty.write_string("");
        let cleared = v47::PluginMessage {
            channel: "MC|ItemName".into(),
            data: empty.as_slice().to_vec(),
        };
        assert_wire(
            &wire,
            "c17_itemname_empty",
            &container_plugin_message(&cleared).expect("MC|ItemName"),
        );
        let mut effects = Vec::new();
        effects.extend_from_slice(&1_i32.to_be_bytes());
        effects.extend_from_slice(&10_i32.to_be_bytes());
        let beacon_v47 = v47::PluginMessage {
            channel: "MC|Beacon".into(),
            data: effects,
        };
        assert_wire(
            &wire,
            "c17_beacon_backing_array",
            &container_plugin_message(&beacon_v47).expect("MC|Beacon"),
        );
        assert_wire(&wire, "c17_beacon_backing_array", &beacon(1, 10));

        let refs = payload_refs();
        let trade = v47::PluginMessage {
            channel: "MC|TrSel".into(),
            data: 2_i32.to_be_bytes().to_vec(),
        };
        let converted = container_plugin_message(&trade).expect("MC|TrSel");
        assert_eq!(hex(&frame(&converted)), refs["trsel_2"].1);

        // A channel or a payload this conversion does not know is refused, never guessed.
        let unknown = v47::PluginMessage {
            channel: "MC|Other".into(),
            data: vec![1, 2, 3],
        };
        assert_eq!(container_plugin_message(&unknown), None);
        let short = v47::PluginMessage {
            channel: "MC|Beacon".into(),
            data: vec![0, 0, 0, 1],
        };
        assert_eq!(container_plugin_message(&short), None);
        let long = v47::PluginMessage {
            channel: "MC|TrSel".into(),
            data: vec![0, 0, 0, 1, 0],
        };
        assert_eq!(container_plugin_message(&long), None);
    }

    #[test]
    fn buffer_payloads_are_the_whole_backing_array_as_the_1_7_10_screens_send_them() {
        let refs = payload_refs();
        for (name, index) in [("trsel_0", 0), ("trsel_2", 2), ("trsel_9", 9)] {
            assert!(refs[name].0);
            assert_eq!(hex(&frame(&trade_select(index))), refs[name].1, "{name}");
        }
        for (name, primary, secondary) in [("beacon_1_10", 1, 10), ("beacon_0_0", 0, 0)] {
            assert_eq!(
                hex(&frame(&beacon(primary, secondary))),
                refs[name].1,
                "{name}"
            );
        }
        let block = |x, y, z| CommandBlockTarget::Block {
            position: BlockPos::new(x, y, z),
        };
        let commands = [
            ("advcdm_block", block(-5, 64, 300), "say hi".to_owned()),
            (
                "advcdm_minecart",
                CommandBlockTarget::Minecart { entity_id: 42 },
                "tp @p ~ ~1 ~".to_owned(),
            ),
            (
                "advcdm_unicode",
                block(1, 2, 3),
                "say \u{e9}\u{20ac}\u{a7}a\u{1F600}".to_owned(),
            ),
            ("advcdm_crosses_256", block(0, 0, 0), "a".repeat(242)),
            ("advcdm_16384", block(0, 0, 0), "b".repeat(16369)),
            ("advcdm_16385", block(0, 0, 0), "b".repeat(16370)),
        ];
        for (name, target, command) in commands {
            let (sent, vanilla, _) = refs[name];
            match command_block(target, &command) {
                Some(packet) => {
                    assert!(sent, "{name}: vanilla sends nothing");
                    assert_eq!(hex(&frame(&packet)), vanilla, "{name}");
                }
                None => assert!(!sent, "{name}: vanilla sends it"),
            }
        }
        // The boundary itself, independently of the fixture: 16384 written bytes are sent as a
        // 16384-byte array, one more is not sent.
        assert_eq!(backing_array(&[7; 16_384]).map(|a| a.len()), Some(16_384));
        assert_eq!(backing_array(&[7; 16_385]), None);
        assert_eq!(backing_array(&[7; 257]).map(|a| a.len()), Some(512));
        assert_eq!(backing_array(&[]).map(|a| a.len()), Some(256));
    }

    /// The page text `V17PayloadRef.noise` writes: a fixed LCG over the 94 printable ASCII
    /// characters, so gzip barely shrinks it.
    fn noise(n: usize, seed: i64) -> String {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                char::from(b'!' + (((x as u64) >> 33) % 94) as u8)
            })
            .collect()
    }

    fn book_stack(item_id: i16, pages: &[&str], author_title: Option<(&str, &str)>) -> ItemStack {
        let mut tag = NbtCompound::new();
        tag.insert(
            "pages",
            Nbt::list(pages.iter().map(|p| Nbt::String((*p).to_owned())).collect()),
        );
        if let Some((author, title)) = author_title {
            tag.insert("author", Nbt::String(author.to_owned()));
            tag.insert("title", Nbt::String(title.to_owned()));
        }
        ItemStack {
            item_id,
            count: 1,
            damage: 0,
            nbt: Some(tag),
        }
    }

    /// The item a book payload carries, read back the way the 1.7.10 server reads it.
    fn read_book(data: &[u8]) -> ItemStack {
        let mut buf = PacketBuffer::from_slice(data);
        buf.read_item_v5(&mut NbtBudget::per_packet())
            .expect("a protocol-5 slot")
            .expect("a book")
    }

    /// The book's tag as `CompressedStreamTools.write` writes it, before gzip.
    fn uncompressed(stack: &ItemStack) -> String {
        let mut raw = Vec::new();
        mc_nbt::write_compound(&mut raw, "", stack.nbt.as_ref().expect("a tag")).expect("NBT");
        hex(&raw)
    }

    fn plugin_data(packet: &C2SPlay5) -> &[u8] {
        match packet {
            C2SPlay5::PluginMessage(message) => &message.data,
            other => panic!("not a plugin message: {other:?}"),
        }
    }

    #[test]
    fn book_payloads_carry_vanillas_book_in_a_padded_array() {
        let refs = payload_refs();
        // The book the server reads and the uncompressed NBT first (gzip's deflate body is the
        // encoder's own), then the padding rule on this encoder's length.
        let cases = [
            (
                "bedit_two_pages",
                "MC|BEdit",
                book_stack(386, &["Hello", "world"], None),
            ),
            (
                "bedit_trailing_empty",
                "MC|BEdit",
                book_stack(386, &["one"], None),
            ),
            (
                "bedit_crosses_256",
                "MC|BEdit",
                book_stack(386, &[&noise(600, 7)], None),
            ),
        ];
        for (name, channel, stack) in cases {
            let (sent, vanilla, nbt) = refs[name];
            assert!(sent, "{name}");
            let packet = book(channel, &stack).expect("sent");
            let mut vanilla_frame = PacketBuffer::from_slice(&unhex(vanilla));
            assert_eq!(vanilla_frame.read_varint().unwrap(), 0x17);
            let vanilla = c2s::PluginMessage::read(&mut vanilla_frame).expect("vanilla's C17");
            let ours = plugin_data(&packet);
            assert_eq!(read_book(ours), read_book(&vanilla.data), "{name}");
            assert_eq!(uncompressed(&stack), nbt, "{name}: uncompressed NBT");
            // Zero padding to the power of two above what this encoder wrote.
            let mut written = PacketBuffer::new();
            written.write_item_v5(Some(&stack));
            let used = written.as_slice().len();
            assert_eq!(ours.len(), used.next_power_of_two().max(256), "{name}");
            assert!(ours[used..].iter().all(|b| *b == 0), "{name}");
            assert_eq!(ours.len(), vanilla.data.len(), "{name}: same array size");
            // And today the bytes themselves: this build's deflate writes what Java's `Deflater`
            // writes for these books. Only the inflated NBT matters to a server, so a different
            // deflate here would be a change to look at, not a protocol fault.
            assert_eq!(ours, vanilla.data.as_slice(), "{name}: the whole payload");
        }
        let (sent, ..) = refs["bedit_over_16384"];
        assert!(!sent);
        let pages = [noise(9000, 1), noise(9000, 2), noise(9000, 3)];
        let pages: Vec<&str> = pages.iter().map(String::as_str).collect();
        assert_eq!(book("MC|BEdit", &book_stack(386, &pages, None)), None);
    }

    #[test]
    fn a_signed_book_keeps_its_pages_as_typed() {
        let refs = payload_refs();
        let (sent, vanilla, nbt) = refs["bsign_plain_pages"];
        assert!(sent);
        // What the book screen builds when signing: 1.8.9's JSON string literals.
        let typed = ["a < b & c's \"q\"", "\u{e9}\u{20ac}"];
        let screen = book_stack(
            387,
            &[
                "\"a \\u003c b \\u0026 c\\u0027s \\\"q\\\"\"",
                "\"\u{e9}\u{20ac}\"",
            ],
            Some(("Player", "Notes")),
        );
        let sent_stack = signed_book_pages_as_typed(&screen).expect("string literal pages");
        assert_eq!(
            sent_stack,
            book_stack(387, &typed, Some(("Player", "Notes")))
        );
        let packet = book("MC|BSign", &sent_stack).expect("sent");
        let mut vanilla_frame = PacketBuffer::from_slice(&unhex(vanilla));
        assert_eq!(vanilla_frame.read_varint().unwrap(), 0x17);
        let vanilla = c2s::PluginMessage::read(&mut vanilla_frame).expect("vanilla's C17");
        assert_eq!(vanilla.channel, "MC|BSign");
        assert_eq!(read_book(plugin_data(&packet)), read_book(&vanilla.data));
        assert_eq!(
            plugin_data(&packet),
            vanilla.data.as_slice(),
            "the whole payload"
        );
        assert_eq!(uncompressed(&sent_stack), nbt, "uncompressed NBT");
        // Negative control: the screen's JSON pages are not what 1.7.10 signs.
        assert_ne!(
            read_book(plugin_data(&book("MC|BSign", &screen).unwrap())),
            read_book(&vanilla.data)
        );
    }

    /// A page that is not a JSON string literal (another encoding of the screen's, say) is never
    /// sent as it is: the whole book is refused.
    #[test]
    fn a_signed_book_page_that_is_not_a_string_literal_refuses_the_book() {
        let signer = Some(("Player", "Notes"));
        for pages in [
            &["\"fine\"", "{\"text\":\"json\"}"][..],
            &["plain"][..],
            &["\"open"][..],
        ] {
            let screen = book_stack(387, pages, signer);
            assert_eq!(signed_book_pages_as_typed(&screen), None, "{pages:?}");
        }
        let mut not_a_string = book_stack(387, &["\"a\""], signer);
        if let Some(tag) = not_a_string.nbt.as_mut() {
            tag.insert("pages", Nbt::list(vec![Nbt::Int(1)]));
        }
        assert_eq!(signed_book_pages_as_typed(&not_a_string), None);
        // Literal pages convert, and a book without pages is left alone.
        let fine = book_stack(387, &["\"fine\"", "\"\""], signer);
        let converted = signed_book_pages_as_typed(&fine).expect("literals");
        assert_eq!(converted, book_stack(387, &["fine", ""], signer));
        let bare = ItemStack {
            nbt: None,
            ..fine.clone()
        };
        assert_eq!(signed_book_pages_as_typed(&bare), Some(bare));
    }

    #[test]
    fn sign_lines_are_the_plain_text_of_the_editors_components() {
        let wire = wire_fixtures();
        let lines = ["Hello", "", "line \u{e9}", "fifteen chars!!"]
            .map(|line| ChatComponent::text(line).to_json());
        assert_wire(
            &wire,
            "c12_sign",
            &update_sign(BlockPos::new(10, 64, -10), &lines),
        );
        // Plain text keeps a section sign as it is (`getUnformattedText`), never a style.
        let styled = [
            r#"{"text":"§cRed"}"#.to_owned(),
            String::new(),
            String::new(),
            String::new(),
        ];
        let C2SPlay5::UpdateSign(sign) = update_sign(BlockPos::new(0, 0, 0), &styled) else {
            unreachable!()
        };
        assert_eq!(sign.lines[0], "\u{a7}cRed");
    }

    #[test]
    fn tab_completion_sends_the_text_alone() {
        let wire = wire_fixtures();
        assert_wire(&wire, "c14_tab", &tab_complete("/tp St"));
    }
}
