//! Hostile input for the protocol-5 codec: random and mutated bytes never panic, length fields
//! never size an allocation, and the limits hold (1.7.10's own NBT accounting and depth, plus the
//! per-packet NBT budget and the chunk-bulk caps that keep a frame of compressed data from
//! expanding without bound).

use mc_nbt::{Nbt, NbtCompound};
use mc_protocol::v5::chunk::{
    MapChunkBulk, BULK_COLUMN_BYTES, MAX_BULK_COLUMNS, MAX_BULK_COLUMN_BYTES,
};
use mc_protocol::v5::login::{EncryptionRequest, EncryptionResponse};
use mc_protocol::v5::nbt::{read_gzip_compound, write_gzip_compound, NbtBudget};
use mc_protocol::v5::{metadata, s2c, C2SPlay5, PacketBufferV5, S2CPlay5};
use mc_protocol::{ItemStack, Packet, PacketBuffer, ProtoError};
use proptest::prelude::*;

fn sample_tag() -> NbtCompound {
    let mut tag = NbtCompound::new();
    tag.insert("RepairCost", Nbt::Int(3));
    let mut display = NbtCompound::new();
    display.insert("Name", Nbt::String("Sword".into()));
    tag.insert("display", Nbt::Compound(display));
    tag.insert("ench", Nbt::list(vec![Nbt::Short(1), Nbt::Short(2)]));
    tag
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn s2c_read_never_panics(id in 0x00i32..=0x49, bytes in proptest::collection::vec(any::<u8>(), 0..400)) {
        let mut buf = PacketBuffer::from_slice(&bytes);
        let _ = S2CPlay5::read(id, &mut buf);
    }

    #[test]
    fn c2s_read_never_panics(id in 0x00i32..=0x19, bytes in proptest::collection::vec(any::<u8>(), 0..400)) {
        let mut buf = PacketBuffer::from_slice(&bytes);
        let _ = C2SPlay5::read(id, &mut buf);
    }

    #[test]
    fn login_reads_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..400)) {
        let _ = EncryptionRequest::read(&mut PacketBuffer::from_slice(&bytes));
        let _ = EncryptionResponse::read(&mut PacketBuffer::from_slice(&bytes));
    }

    #[test]
    fn metadata_and_slots_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..400)) {
        let _ = metadata::read(&mut PacketBuffer::from_slice(&bytes), &mut NbtBudget::per_packet());
        let _ = PacketBuffer::from_slice(&bytes).read_item_v5(&mut NbtBudget::per_packet());
    }

    #[test]
    fn random_gzip_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..300)) {
        let _ = read_gzip_compound(&bytes, &mut NbtBudget::per_packet());
    }

    /// A valid gzip NBT blob with some bytes changed: any verdict, never a panic.
    #[test]
    fn mutated_gzip_never_panics(flips in proptest::collection::vec((any::<usize>(), any::<u8>()), 1..8)) {
        let mut blob = write_gzip_compound(&sample_tag()).unwrap();
        for (at, value) in flips {
            let len = blob.len();
            blob[at % len] = value;
        }
        let _ = read_gzip_compound(&blob, &mut NbtBudget::per_packet());
    }

    /// A clientbound packet with a slot round-trips for any stack fields.
    #[test]
    fn set_slot_round_trips(window in any::<i8>(), slot in any::<i16>(), id in 0i16..=i16::MAX, count in any::<i8>(), damage in 0i16..=i16::MAX, tagged in any::<bool>()) {
        let packet = s2c::SetSlot {
            window_id: window,
            slot,
            item: Some(ItemStack { item_id: id, count: i32::from(count), damage, nbt: tagged.then(sample_tag) }),
        };
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        let mut read = PacketBuffer::from_slice(buf.as_slice());
        prop_assert_eq!(s2c::SetSlot::read(&mut read).unwrap(), packet);
        prop_assert!(read.is_empty());
    }

    /// Entity movement without onGround round-trips over the whole byte range.
    #[test]
    fn look_move_round_trips(entity_id in any::<i32>(), d in any::<[i8; 3]>(), yaw in any::<u8>(), pitch in any::<u8>()) {
        let packet = s2c::EntityLookAndRelMove {
            entity_id,
            delta_x: d[0],
            delta_y: d[1],
            delta_z: d[2],
            yaw: mc_protocol::Angle(yaw),
            pitch: mc_protocol::Angle(pitch),
        };
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        prop_assert_eq!(buf.as_slice().len(), 9);
        let mut read = PacketBuffer::from_slice(buf.as_slice());
        prop_assert_eq!(s2c::EntityLookAndRelMove::read(&mut read).unwrap(), packet);
    }
}

#[test]
fn out_of_range_ids_are_unknown() {
    for id in [-1, 0x41, 0x49, 0x7F, i32::MAX] {
        assert!(matches!(
            S2CPlay5::read(id, &mut PacketBuffer::new()),
            Err(ProtoError::UnknownPacketId { .. })
        ));
    }
    for id in [-1, 0x18, 0x19, i32::MIN] {
        assert!(matches!(
            C2SPlay5::read(id, &mut PacketBuffer::new()),
            Err(ProtoError::UnknownPacketId { .. })
        ));
    }
}

/// Length fields claiming gigabytes cost a failed read, not an allocation.
#[test]
fn hostile_lengths_do_not_allocate() {
    let max_int = [0x7F, 0xFF, 0xFF, 0xFF];
    // S21 compressed length: x, z, ground-up, primary, add, then the length.
    let mut s21 = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0xFF, 0xFF, 0, 0];
    s21.extend_from_slice(&max_int);
    assert!(matches!(
        S2CPlay5::read(0x21, &mut PacketBuffer::from_slice(&s21)),
        Err(ProtoError::Eof(_))
    ));
    // S22 record length.
    let mut s22 = vec![0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF];
    s22.extend_from_slice(&max_int);
    assert!(S2CPlay5::read(0x22, &mut PacketBuffer::from_slice(&s22)).is_err());
    // S30 with 32767 slots and nothing behind them.
    assert!(S2CPlay5::read(0x30, &mut PacketBuffer::from_slice(&[0, 0x7F, 0xFF])).is_err());
    // S20 claiming 2^31-1 properties.
    let mut s20 = vec![0, 0, 0, 1];
    s20.extend_from_slice(&max_int);
    assert!(S2CPlay5::read(0x20, &mut PacketBuffer::from_slice(&s20)).is_err());
    // S37 claiming 2^31-1 statistics.
    assert!(S2CPlay5::read(
        0x37,
        &mut PacketBuffer::from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0x07])
    )
    .is_err());
    // A slot whose NBT claims 32767 bytes that are not there.
    let mut slot = vec![0, 1, 1, 0, 0];
    slot.extend_from_slice(&[0x7F, 0xFF]);
    assert!(matches!(
        PacketBuffer::from_slice(&slot).read_item_v5(&mut NbtBudget::per_packet()),
        Err(ProtoError::Eof(_))
    ));
}

/// Java allocates `196864 * count` bytes for S26 before reading anything else, so a frame of a
/// few hundred bytes with an empty stream demands megabytes of zero-filled columns. The column
/// cap bounds that before anything is allocated.
#[test]
fn bulk_caps_hold_before_inflating() {
    let bulk = |count: u16, primary: u16| {
        let mut wire = Vec::new();
        wire.extend_from_slice(&count.to_be_bytes());
        wire.extend_from_slice(&0i32.to_be_bytes());
        wire.push(1);
        for i in 0..count {
            wire.extend_from_slice(&i32::from(i).to_be_bytes());
            wire.extend_from_slice(&0i32.to_be_bytes());
            wire.extend_from_slice(&primary.to_be_bytes());
            // The add mask equals the primary mask: a full column slices all 196864 bytes.
            wire.extend_from_slice(&primary.to_be_bytes());
        }
        (
            wire.len(),
            MapChunkBulk::read(&mut PacketBuffer::from_slice(&wire)),
        )
    };
    let over_count = u16::try_from(MAX_BULK_COLUMNS + 1).unwrap();
    assert!(matches!(
        bulk(over_count, 1).1,
        Err(ProtoError::PayloadTooLong { .. })
    ));
    assert!(matches!(
        bulk(400, 0xFFFF).1,
        Err(ProtoError::PayloadTooLong { .. })
    ));
    // At the cap, the worst case: every column full, no stream at all. Java decodes it to
    // zero-filled columns too; this is the most one bulk can cost.
    let at_cap = u16::try_from(MAX_BULK_COLUMNS).unwrap();
    let (wire_len, full) = bulk(at_cap, 0xFFFF);
    let full = full.expect("64 full columns");
    let total: usize = full.columns.iter().map(|c| c.data.len()).sum();
    assert_eq!(total, MAX_BULK_COLUMN_BYTES);
    assert_eq!(MAX_BULK_COLUMN_BYTES, MAX_BULK_COLUMNS * BULK_COLUMN_BYTES);
    assert!(
        wire_len < 1024 && total < 13 << 20,
        "{wire_len} bytes -> {total}"
    );
    // A modest bulk with no data at all decodes to zero-filled columns, as in Java.
    let small = bulk(2, 1).1.expect("two one-section columns");
    assert!(small.columns.iter().all(|c| c.data.iter().all(|&b| b == 0)));
}

/// A compound nested 513 deep is rejected at 1.7.10's limit, without exhausting the stack.
#[test]
fn deep_nbt_is_depth_limited() {
    let mut raw = vec![10, 0, 0];
    for _ in 0..600 {
        raw.extend_from_slice(&[10, 0, 0]);
    }
    raw.extend(std::iter::repeat_n(0, 601));
    let blob = mc_protocol::v5::nbt::gzip_java(&raw).unwrap();
    assert!(matches!(
        read_gzip_compound(&blob, &mut NbtBudget::per_packet()),
        Err(ProtoError::Nbt(mc_nbt::NbtError::DepthLimit { limit: 512 }))
    ));
}

/// A deflate bomb of values 1.7.10 does not charge for: a list of 20 M empty compounds in one
/// 20 KiB blob. The per-packet value budget stops it after a few hundred kilobytes.
#[test]
fn uncharged_values_are_bounded_per_packet() {
    let count: i32 = 20 << 20;
    let mut raw = vec![10, 0, 0, 9, 0, 0, 10];
    raw.extend_from_slice(&count.to_be_bytes());
    raw.extend(std::iter::repeat_n(0u8, count as usize + 1));
    let blob = mc_protocol::v5::nbt::gzip_java(&raw).unwrap();
    assert!(
        blob.len() < 32 * 1024,
        "the bomb fits an i16 length: {}",
        blob.len()
    );
    let error = read_gzip_compound(&blob, &mut NbtBudget::per_packet()).unwrap_err();
    assert!(error.to_string().contains("NBT values"), "{error}");
}

/// Tags that each pass 1.7.10's own 2 MiB tracker but together inflate past the per-packet
/// budget: an S30 whose slots each carry a 2 MiB byte array (2 KiB of gzip apiece).
#[test]
fn inflated_bytes_are_bounded_per_packet() {
    // root { "": byte array of 2097152 } charges exactly the tracker's limit (an empty key is
    // free), and inflates to 2097163 bytes.
    let mut raw = vec![10, 0, 0, 7, 0, 0];
    raw.extend_from_slice(&2_097_152i32.to_be_bytes());
    raw.resize(raw.len() + 2_097_152, 0);
    raw.push(0);
    let blob = mc_protocol::v5::nbt::gzip_java(&raw).unwrap();
    assert!(blob.len() < 4096, "{}", blob.len());
    // One such tag is fine on its own...
    assert!(read_gzip_compound(&blob, &mut NbtBudget::per_packet()).is_ok());
    // ...eight or more in one packet are over the 16 MiB budget (8 * 2097163 > 16777216).
    let slots = 9i16;
    let mut wire = vec![0u8];
    wire.extend_from_slice(&slots.to_be_bytes());
    for _ in 0..slots {
        wire.extend_from_slice(&[0, 1, 1, 0, 0]); // stone, 1, damage 0
        wire.extend_from_slice(&(blob.len() as i16).to_be_bytes());
        wire.extend_from_slice(&blob);
    }
    let error = S2CPlay5::read(0x30, &mut PacketBuffer::from_slice(&wire)).unwrap_err();
    assert!(error.to_string().contains("inflated more than"), "{error}");
    // Negative control: seven stay under it.
    let mut seven = vec![0u8, 0, 7];
    for _ in 0..7 {
        seven.extend_from_slice(&[0, 1, 1, 0, 0]);
        seven.extend_from_slice(&(blob.len() as i16).to_be_bytes());
        seven.extend_from_slice(&blob);
    }
    assert!(S2CPlay5::read(0x30, &mut PacketBuffer::from_slice(&seven)).is_ok());
}
