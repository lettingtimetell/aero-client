//! The client name servers are told when you join: always `Aero`.
//!
//! java: `NetHandlerPlayClient.handleJoinGame` (`:290-291`) runs
//! `gameSettings.sendSettingsToServer()` (C15) and then sends
//! `C17PacketCustomPayload("MC|Brand", writeString(ClientBrandRetriever.getClientModName()))`,
//! where `getClientModName()` is `"vanilla"`. [`crate::session::Session`] sends exactly that
//! pair at every JoinGame, with [`JoinSettings::brand`]; Aero Client puts its own name, `Aero`,
//! in that same packet at that same moment. Nothing else differs: no other packet, no plugin
//! channel registration. F3's first line (`Minecraft 1.8.9 (<version>/<brand>)`, vanilla's
//! `GuiOverlayDebug`, which also reads `getClientModName()`) shows the same name
//! ([`Client::shown_brand`]).
//!
//! It isn't an option: a toggle existed for one build on 2026-11-22 and was retired
//! (`mc_mods::mods::aero_title::RETIRED_OPTION_IDS`); a settings file that still holds it loads
//! as an unknown key and changes nothing.
//!
//! Cost: none (a constant).
//!
//! [`JoinSettings::brand`]: crate::session::JoinSettings::brand

use super::super::Client;

/// The brand sent at every JoinGame, in vanilla's `getClientModName()` slot.
pub(crate) const AERO_BRAND: &str = "Aero";

/// The `MC|Brand` value for a connection opened now.
#[must_use]
pub(crate) const fn client_brand() -> &'static str {
    AERO_BRAND
}

impl Client {
    /// The brand F3 shows: the one the current connection sends at its JoinGames, or, with no
    /// connection, the one a connection opened now would send.
    #[must_use]
    pub(crate) fn shown_brand(&self) -> &str {
        match &self.session {
            Some(session) => session.brand(),
            None => client_brand(),
        }
    }
}

#[cfg(test)]
mod tests {
    use mc_protocol::play::{s2c, C2SPlay, S2CPlay};
    use mc_protocol::PacketBuffer;

    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::{ToGame, ToNet};
    use crate::session::Session;

    /// The brand is always `Aero`; vanilla's own value is only the session default that the
    /// client never uses.
    #[test]
    fn the_brand_is_always_aero() {
        assert_eq!(client_brand(), "Aero");
        assert_eq!(crate::session::JoinSettings::default().brand, "vanilla");
    }

    /// A real client on the real assets in a temp game dir, or `None` (skip) without the
    /// vanilla jar (`MC_JARS_DIR`).
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    fn join_game() -> S2CPlay {
        S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 1,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        })
    }

    /// A packet as it goes on the wire: VarInt id, then the body.
    fn wire(packet: &C2SPlay) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    /// Everything the session put on the wire since the last call, encoded.
    fn sent(to_net: &mut tokio::sync::mpsc::UnboundedReceiver<ToNet>) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Ok(message) = to_net.try_recv() {
            if let ToNet::Send(packet) = message {
                out.push(wire(&packet));
            }
        }
        out
    }

    /// End to end through the client's own join settings and a socket-less session: a JoinGame
    /// sends exactly two packets, C15 then C17 `MC|Brand` = `Aero`, as `handleJoinGame` does.
    /// A second JoinGame of the same connection (a proxy's server switch) sends the same, a
    /// settings file that still holds the retired toggle (off) changes nothing, and F3 shows
    /// `Aero` with and without a connection.
    #[test]
    fn a_join_sends_client_settings_then_the_aero_brand_and_nothing_else() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        // java: the C17 vanilla writes, `MC|Brand` + a VarInt length + the name.
        let c17 = |brand: &[u8]| -> Vec<u8> {
            let length = u8::try_from(brand.len()).expect("short");
            [&[0x17_u8, 0x08][..], &b"MC|Brand"[..], &[length][..], brand].concat()
        };
        let aero_c17 = c17(b"Aero");

        // The retired toggle, off, as the one build that had it would have saved it.
        client
            .mods
            .settings
            .set_bool("aero_title", "client_brand", false);

        let (handle, to_game, mut to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, client.join_settings());
        to_game
            .try_send(ToGame::Packet(join_game()))
            .expect("channel");
        session.poll(16);
        let first = sent(&mut to_net);
        assert_eq!(first.len(), 2, "C15 and C17 only: {first:02X?}");
        assert_eq!(first[0][0], 0x15, "client settings first");
        assert_eq!(first[1], aero_c17, "then MC|Brand = Aero");
        client.session = Some(session);
        assert_eq!(client.shown_brand(), "Aero");

        to_game
            .try_send(ToGame::Packet(join_game()))
            .expect("channel");
        client.session.as_mut().expect("session").poll(16);
        let again = sent(&mut to_net);
        assert_eq!(again.len(), 2, "{again:02X?}");
        assert_eq!(again[1], aero_c17, "the same name at every JoinGame");

        client.session = None;
        assert_eq!(client.shown_brand(), "Aero");
    }
}
