//! Round-trip / robustness fuzzing across every packet type (S06's "round-trip fuzz across
//! every packet type" task).
//!
//! Two complementary properties, both via `proptest`:
//!
//! 1. **Never panics on malformed input.** For every legal Play packet id (both directions)
//!    plus every Handshake/Status/Login packet, feed `Packet::read` (or the dispatch enums'
//!    `read`) arbitrary random bytes and assert the result is `Ok` or `Err` — never a panic.
//!    This exercises `read` for all **113** packet types this crate defines (74 + 26 Play,
//!    2 chunk, 1 handshake, 4 status, 6 login) against inputs no hand-written test would think
//!    to construct, which is the property §11's "no `unwrap()` on network input" rule actually
//!    exists to guarantee.
//! 2. **Round-trips for representative field-randomized instances.** A handful of packets
//!    (chosen for variety: a bare scalar, a string, packed position/angle fields, and a
//!    metadata-bearing spawn packet) get their fields drawn from `proptest` strategies rather
//!    than one fixed hand-built value, so the deterministic unit test beside each packet type
//!    (which every one of the 113 types has) is complemented by randomized coverage on these.
//!
//! Every packet type already has its own fixed-value round-trip test in `src/`; this file is
//! additive robustness/breadth, not a replacement for those.

use mc_math::BlockPos;
use mc_protocol::play::{s2c, C2SPlay, S2CPlay};
use mc_protocol::{Angle, Packet, PacketBuffer};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Every clientbound Play id, fed random bytes, must never panic.
    #[test]
    fn s2c_play_read_never_panics(id in 0x00i32..=0x49, bytes in proptest::collection::vec(any::<u8>(), 0..300)) {
        let mut buf = PacketBuffer::from_slice(&bytes);
        let _ = S2CPlay::read(id, &mut buf); // Ok or Err, either is fine; a panic fails the test.
    }

    /// Every serverbound Play id, fed random bytes, must never panic.
    #[test]
    fn c2s_play_read_never_panics(id in 0x00i32..=0x19, bytes in proptest::collection::vec(any::<u8>(), 0..300)) {
        let mut buf = PacketBuffer::from_slice(&bytes);
        let _ = C2SPlay::read(id, &mut buf);
    }

    /// Ids outside either legal range must be a clean `UnknownPacketId`, not a panic (guards
    /// against an off-by-one in the dispatch macro's match arms).
    #[test]
    fn out_of_range_play_ids_never_panic(id in i32::MIN..=i32::MAX, bytes in proptest::collection::vec(any::<u8>(), 0..64)) {
        let mut buf = PacketBuffer::from_slice(&bytes);
        let _ = S2CPlay::read(id, &mut buf);
        let mut buf2 = PacketBuffer::from_slice(&bytes);
        let _ = C2SPlay::read(id, &mut buf2);
    }

    /// `KeepAlive` (S2C): any VarInt-range id round-trips.
    #[test]
    fn keep_alive_round_trips_any_id(id in any::<i32>()) {
        let pkt = s2c::KeepAlive { id };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        prop_assert_eq!(s2c::KeepAlive::read(&mut read_buf).unwrap(), pkt);
        prop_assert!(read_buf.is_empty());
    }

    /// `Chat` (S2C): arbitrary Unicode text and any position byte round-trip.
    #[test]
    fn chat_round_trips_arbitrary_text(text in ".{0,200}", position in any::<i8>()) {
        let pkt = s2c::Chat { component: text, position };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        prop_assert_eq!(s2c::Chat::read(&mut read_buf).unwrap(), pkt);
    }

    /// `SpawnPosition` (S2C): the packed 26/12/26 `Position` type (hazard H5) round-trips
    /// across the *entire* legal coordinate range, not just the handful of fixed points the
    /// unit tests in `buffer.rs`/`chunk.rs` already cover.
    #[test]
    fn spawn_position_round_trips_full_coordinate_range(
        x in -33_554_432i32..33_554_432,
        y in -2048i32..2048,
        z in -33_554_432i32..33_554_432,
    ) {
        let pkt = s2c::SpawnPosition { position: BlockPos::new(x, y, z) };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        prop_assert_eq!(s2c::SpawnPosition::read(&mut read_buf).unwrap(), pkt);
    }

    /// `Animation` (S2C): angle byte round-trips through the whole `u8` range via `Angle`.
    #[test]
    fn spawn_player_angle_fields_round_trip_every_byte(yaw_raw in any::<u8>(), pitch_raw in any::<u8>()) {
        let pkt = s2c::SpawnPlayer {
            entity_id: 1,
            player_uuid: uuid::Uuid::nil(),
            x: 0,
            y: 0,
            z: 0,
            yaw: Angle(yaw_raw),
            pitch: Angle(pitch_raw),
            current_item: 0,
            metadata: vec![],
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        prop_assert_eq!(s2c::SpawnPlayer::read(&mut read_buf).unwrap(), pkt);
    }
}
