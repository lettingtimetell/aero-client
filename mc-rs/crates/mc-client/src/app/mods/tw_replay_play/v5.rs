//! Watching a replay recorded on a 1.7.10 connection.
//!
//! The recorder stamps the connection's protocol on the file (`ReplayMeta::protocol`, 5 for
//! 1.7.10) and keeps each packet's exact protocol-5 bytes. Playback then reads them with the
//! protocol-5 decoder into a 1.7.10 session ([`Session::with_version`]), whose packets are
//! applied through the 1.7.10 path (`sync::dispatch_v5`) when playback flushes them.

use mc_protocol::v5::S2CPlay5;
use mc_protocol::PacketBuffer;
use mc_version::GameVersion;

use super::super::super::Client;
use crate::net::ToGame;

/// The version a replay was recorded in, from its header: 1.8.9 for any protocol the client
/// does not play (a file from before the header named the version says 47).
pub(super) fn recorded_version(protocol: i32) -> GameVersion {
    GameVersion::from_protocol(protocol).unwrap_or_default()
}

/// Decode one recorded protocol-5 clientbound packet: id VarInt, then the body, all of it.
pub(super) fn decode_packet_v5(payload: &[u8]) -> Result<S2CPlay5, String> {
    let mut buf = PacketBuffer::from_slice(payload);
    let id = buf.read_varint().map_err(|error| error.to_string())?;
    let packet = S2CPlay5::read(id, &mut buf).map_err(|error| error.to_string())?;
    if buf.is_empty() {
        Ok(packet)
    } else {
        Err(format!(
            "packet {id:#04x} has {} bytes left over",
            buf.remaining()
        ))
    }
}

/// Whether playback applies a recorded 1.7.10 packet. As for 1.8.9, the transport packets and
/// the ones that would leave the replay are skipped: the keep-alive, the kick, and the server
/// resource pack offer, which 1.7.10 sends as the `MC|RPack` plugin message.
fn replays_packet_v5(packet: &S2CPlay5) -> bool {
    match packet {
        S2CPlay5::KeepAlive(_) | S2CPlay5::Disconnect(_) => false,
        S2CPlay5::PluginMessage(message) => message.channel != "MC|RPack",
        _ => true,
    }
}

impl Client {
    /// [`Client::tw_rb_apply_packet`] for a 1.7.10 recording.
    pub(super) fn tw_rb_apply_packet_v5(&mut self, decoded: Result<S2CPlay5, String>) {
        let Some(playback) = self.tw_play.playback.as_deref_mut() else {
            return;
        };
        match decoded {
            Ok(packet) if replays_packet_v5(&packet) => {
                playback.stats.packets += 1;
                playback.packets_pending = true;
                // A recorded container screen is drawn from it, as for a 1.8.9 replay. A window
                // type 1.7.10 does not open leaves the previous one, as 1.7.10's `switch` does.
                if let S2CPlay5::OpenWindow(open) = &packet {
                    if let Some(open) = mc_ui::screens::container::v5::open_window_v47(open) {
                        playback.gui.open_window = Some(open);
                    }
                }
                if let Some(session) = self.session.as_mut() {
                    session.push_driven(ToGame::PacketV5(Box::new(packet)));
                }
            }
            Ok(_) => playback.stats.packets_skipped += 1,
            Err(error) => {
                playback.stats.packet_errors += 1;
                tracing::warn!(%error, "replay: a recorded packet did not decode; skipped");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use mc_protocol::v5::s2c;

    use super::*;

    fn bytes(packet: &S2CPlay5) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    #[test]
    fn recorded_protocol_5_packets_decode_and_the_transport_ones_are_skipped() {
        let chat = S2CPlay5::Chat(s2c::Chat {
            component: r#"{"text":"hi"}"#.into(),
        });
        let decoded = decode_packet_v5(&bytes(&chat)).expect("decodes");
        assert!(replays_packet_v5(&decoded));
        let keep_alive = decode_packet_v5(&bytes(&S2CPlay5::KeepAlive(s2c::KeepAlive { id: 4 })))
            .expect("decodes");
        assert!(!replays_packet_v5(&keep_alive));
        let pack = S2CPlay5::PluginMessage(s2c::PluginMessage {
            channel: "MC|RPack".into(),
            data: b"http://example.invalid/pack.zip".to_vec(),
        });
        assert!(!replays_packet_v5(&pack));
        // Leftover bytes and an id 1.7.10 does not have are errors, never a guess.
        let mut long = bytes(&chat);
        long.push(0);
        assert!(decode_packet_v5(&long).is_err());
        assert!(decode_packet_v5(&[0x41]).is_err());
        assert!(decode_packet_v5(&[]).is_err());
    }

    #[test]
    fn the_version_comes_from_the_header() {
        assert_eq!(recorded_version(5), GameVersion::V1_7_10);
        assert_eq!(recorded_version(47), GameVersion::V1_8_9);
        assert_eq!(recorded_version(0), GameVersion::V1_8_9);
    }

    /// A watched Aero replay recorded on a 1.7.10 connection (protocol 5 in its header, each
    /// packet's protocol-5 bytes, as the recorder writes them) plays back through a 1.7.10
    /// session into the world, and keeps the recorded container window for its screen as a
    /// 1.8.9 replay does. Skipped without the vanilla jar.
    #[test]
    fn a_1_7_10_aero_replay_plays_back_into_the_world() {
        use crate::app::config::{ClientConfig, LoginMode};
        use crate::session::Phase;
        use mc_replay::{Record, ReplayMeta, StreamId};

        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");

        let mut raw = vec![0u8; 4096];
        raw[..256].fill(1); // stone at y 64
        raw.extend(std::iter::repeat_n(0u8, 2048 * 2));
        raw.extend(std::iter::repeat_n(0xFFu8, 2048));
        raw.extend(std::iter::repeat_n(1u8, 256));
        let packets = [
            S2CPlay5::JoinGame(s2c::JoinGame {
                entity_id: 27,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 2,
                max_players: 20,
                level_type: "flat".to_owned(),
            }),
            S2CPlay5::MapChunkBulk(
                mc_protocol::v5::chunk::MapChunkBulk::from_raw(true, &[(0, 0, 1 << 4, 0, &raw)])
                    .expect("bulk"),
            ),
            S2CPlay5::PlayerPosLook(s2c::PlayerPosLook {
                x: 8.5,
                y: 65.620_000_004_768_37,
                z: 8.5,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: false,
            }),
            S2CPlay5::OpenWindow(s2c::OpenWindow {
                window_id: 1,
                inventory_type: 0,
                title: "container.chest".to_owned(),
                slot_count: 27,
                use_provided_title: false,
                entity_id: 0,
            }),
        ];
        let records = (0u64..)
            .zip(&packets)
            .map(|(index, packet)| Record {
                time: index * 10,
                stream: StreamId::Packet,
                payload: bytes(packet),
            })
            .collect();
        let meta = ReplayMeta {
            protocol: GameVersion::V1_7_10.protocol(),
            duration: 1_000_000,
            ..ReplayMeta::default()
        };
        client.replay_open_records(meta, records).expect("opens");
        client.replay_render_at(500_000);

        let session = client.session.as_ref().expect("the replay's session");
        assert_eq!(session.version(), GameVersion::V1_7_10);
        assert_eq!(*session.phase(), Phase::Playing);
        assert_eq!(
            session
                .state()
                .world
                .block_at(mc_math::BlockPos::new(3, 64, 5)),
            mc_data::v5::state_from_v5(1 << 4),
            "the recorded chunk was applied"
        );
        let window = client
            .tw_play
            .playback
            .as_deref()
            .and_then(|playback| playback.gui.open_window.clone())
            .expect("the recorded window is kept");
        assert_eq!(window.inventory_type, "minecraft:chest");
        client.replay_close();
    }
}
