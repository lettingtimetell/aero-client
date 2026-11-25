//! Protocol-5 field encodings on top of [`PacketBuffer`].
//!
//! Most primitives are the same in both protocols and stay on [`PacketBuffer`] itself: the
//! fixed-width integers and floats, VarInt (`readVarIntFromBuffer` is byte-for-byte the 1.8.9
//! loop), strings (`readStringFromBuffer`: the same `4 * max` byte cap, Java's lossy UTF-8 and the
//! UTF-16 length check), and the one-byte angle. What protocol 5 does differently lives in
//! [`PacketBufferV5`]:
//!
//! | Encoding | java17 | here |
//! |---|---|---|
//! | `i16` length + bytes, negative rejected | `Packet.readBlob` / `writeBlob` | [`PacketBufferV5::read_blob`] |
//! | optional compound: `i16` length (`< 0` = none) + gzip | `PacketBuffer.readNBTTagCompoundFromBuffer` | [`PacketBufferV5::read_nbt_v5`] |
//! | item stack with that NBT | `PacketBuffer.readItemStackFromBuffer` | [`PacketBufferV5::read_item_v5`] |
//!
//! Block coordinates are separate `x`, `y`, `z` fields whose `y` width differs per packet
//! (`int`, `short`, signed or unsigned `byte`); every packet struct stores them at their wire
//! width and offers a `block_pos()` helper.
//!
//! java17: `network/PacketBuffer.java`, `network/Packet.java`

use mc_nbt::NbtCompound;

use super::nbt::{gzip_java, read_gzip_compound, write_compound_java17, NbtBudget};
use crate::item::ItemStack;
use crate::{PacketBuffer, ProtoError};

/// The longest blob an `i16` length prefix can describe.
///
/// `Packet.writeBlob` and `writeNBTTagCompoundToBuffer` narrow the length with `writeShort`
/// without a check, so a longer blob would go out with a wrapped length that no reader could
/// parse. The writers here refuse it instead (a deferred [`ProtoError::PayloadTooLong`]).
pub const MAX_SHORT_BLOB: usize = i16::MAX as usize;

/// Protocol-5 encodings for [`PacketBuffer`]. Readers return [`ProtoError`] like every other
/// read; writers are infallible and defer failures to [`PacketBuffer::finish_write`], like
/// [`crate::Packet::write`].
pub trait PacketBufferV5 {
    /// java17: `Packet.readBlob`: an `i16` length, `IOException` if negative, then the bytes.
    ///
    /// # Errors
    ///
    /// [`ProtoError::InvalidByteArrayLen`] for a negative length, [`ProtoError::Eof`] if the
    /// bytes are not there.
    fn read_blob(&mut self) -> Result<Vec<u8>, ProtoError>;

    /// java17: `Packet.writeBlob`. More than [`MAX_SHORT_BLOB`] bytes is a deferred error.
    fn write_blob(&mut self, blob: &[u8]);

    /// java17: `PacketBuffer.readNBTTagCompoundFromBuffer`: an `i16` length, any negative value
    /// meaning "no compound", else that many bytes of gzip read with 1.7.10's tag reader. A
    /// length of 0 is an empty gzip stream, which is an error, not "no compound".
    ///
    /// # Errors
    ///
    /// [`ProtoError::Eof`] if the blob is short, plus everything
    /// [`super::nbt::read_gzip_compound`] reports.
    fn read_nbt_v5(&mut self, budget: &mut NbtBudget) -> Result<Option<NbtCompound>, ProtoError>;

    /// java17: `PacketBuffer.writeNBTTagCompoundToBuffer`: `-1` for none, else the gzip blob's
    /// length and bytes. An unencodable compound or a blob over [`MAX_SHORT_BLOB`] bytes is a
    /// deferred error and writes nothing.
    fn write_nbt_v5(&mut self, nbt: Option<&NbtCompound>);

    /// java17: `PacketBuffer.readItemStackFromBuffer`: `i16` id (negative = empty), signed byte
    /// count, `i16` damage clamped to `>= 0` by `new ItemStack(Item, int, int)`, then the NBT.
    ///
    /// # Errors
    ///
    /// As [`Self::read_nbt_v5`].
    fn read_item_v5(&mut self, budget: &mut NbtBudget) -> Result<Option<ItemStack>, ProtoError>;

    /// java17: `PacketBuffer.writeItemStackToBuffer`. The tag is always written: the guard
    /// `isDamageable() || getShareTag()` is always true because `Item.getShareTag` returns
    /// `true` and its only overrides (`ItemWritableBook`, `ItemEditableBook`) do too.
    fn write_item_v5(&mut self, stack: Option<&ItemStack>);
}

impl PacketBufferV5 for PacketBuffer {
    fn read_blob(&mut self) -> Result<Vec<u8>, ProtoError> {
        let len = self.read_i16()?;
        if len < 0 {
            // java17: "Key was smaller than nothing!  Weird key!"
            return Err(ProtoError::InvalidByteArrayLen(i32::from(len)));
        }
        self.read_bytes(len as usize)
    }

    fn write_blob(&mut self, blob: &[u8]) {
        if blob.len() > MAX_SHORT_BLOB {
            // Records the error and writes nothing.
            self.write_bounded_payload(blob, MAX_SHORT_BLOB);
            return;
        }
        self.write_i16(blob.len() as i16);
        self.write_bytes(blob);
    }

    fn read_nbt_v5(&mut self, budget: &mut NbtBudget) -> Result<Option<NbtCompound>, ProtoError> {
        let len = self.read_i16()?;
        if len < 0 {
            return Ok(None);
        }
        let blob = self.read_bytes(len as usize)?;
        read_gzip_compound(&blob, budget).map(Some)
    }

    fn write_nbt_v5(&mut self, nbt: Option<&NbtCompound>) {
        let Some(compound) = nbt else {
            self.write_i16(-1);
            return;
        };
        let Ok(raw) = write_compound_java17(compound) else {
            // The 1.8.9 writer refuses every compound this one refuses (it accepts a strict
            // subset), so it records the deferred encoder error, without writing a byte.
            self.write_nbt_compound_opt(Some(compound));
            return;
        };
        match gzip_java(&raw) {
            Ok(blob) if blob.len() <= MAX_SHORT_BLOB => {
                self.write_i16(blob.len() as i16);
                self.write_bytes(&blob);
            }
            // Records PayloadTooLong and writes nothing.
            Ok(blob) => self.write_bounded_payload(&blob, MAX_SHORT_BLOB),
            // An in-memory deflate does not fail; if it ever did, refuse the packet the same way.
            Err(_) => self.write_bounded_payload(&raw, 0),
        }
    }

    fn read_item_v5(&mut self, budget: &mut NbtBudget) -> Result<Option<ItemStack>, ProtoError> {
        let item_id = self.read_i16()?;
        if item_id < 0 {
            return Ok(None);
        }
        let count = i32::from(self.read_i8()?);
        // java17: ItemStack(Item, int, int) clamps a negative metadata to 0.
        let damage = self.read_i16()?.max(0);
        let nbt = self.read_nbt_v5(budget)?;
        Ok(Some(ItemStack {
            item_id,
            count,
            damage,
            nbt,
        }))
    }

    fn write_item_v5(&mut self, stack: Option<&ItemStack>) {
        match stack {
            None => self.write_i16(-1),
            Some(stack) => {
                self.write_i16(stack.item_id);
                self.write_i8(stack.count as i8);
                self.write_i16(stack.damage);
                self.write_nbt_v5(stack.nbt.as_ref());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_nbt::Nbt;

    fn tag() -> NbtCompound {
        let mut tag = NbtCompound::new();
        tag.insert("RepairCost", Nbt::Int(1));
        tag
    }

    #[test]
    fn blob_is_i16_prefixed_and_rejects_negative_lengths() {
        let mut buf = PacketBuffer::new();
        buf.write_blob(&[1, 2, 3]);
        assert_eq!(buf.as_slice(), &[0, 3, 1, 2, 3]);
        let mut read = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(read.read_blob().unwrap(), vec![1, 2, 3]);

        let mut negative = PacketBuffer::from_slice(&[0xFF, 0xFF]);
        assert!(matches!(
            negative.read_blob(),
            Err(ProtoError::InvalidByteArrayLen(-1))
        ));
        let mut lying = PacketBuffer::from_slice(&[0x7F, 0xFF, 1]);
        assert!(matches!(lying.read_blob(), Err(ProtoError::Eof(_))));
    }

    #[test]
    fn oversized_blob_is_refused_not_wrapped() {
        let mut buf = PacketBuffer::new();
        buf.write_blob(&vec![0; MAX_SHORT_BLOB + 1]);
        assert!(matches!(
            buf.finish_write(),
            Err(ProtoError::PayloadTooLong {
                max: MAX_SHORT_BLOB,
                ..
            })
        ));
        assert!(buf.as_slice().is_empty());

        let mut buf = PacketBuffer::new();
        buf.write_blob(&vec![7; MAX_SHORT_BLOB]);
        assert!(buf.finish_write().is_ok());
        assert_eq!(&buf.as_slice()[..2], &[0x7F, 0xFF]);
    }

    #[test]
    fn nbt_none_is_minus_one_and_any_negative_reads_as_none() {
        let mut buf = PacketBuffer::new();
        buf.write_nbt_v5(None);
        assert_eq!(buf.as_slice(), &[0xFF, 0xFF]);
        for raw in [[0xFF, 0xFF], [0x80, 0x00], [0xFF, 0xFE]] {
            let mut read = PacketBuffer::from_slice(&raw);
            assert_eq!(
                read.read_nbt_v5(&mut NbtBudget::per_packet()).unwrap(),
                None
            );
            assert!(read.is_empty());
        }
        // Zero is an empty gzip stream: java's GZIPInputStream throws EOFException.
        let mut zero = PacketBuffer::from_slice(&[0, 0]);
        assert!(zero.read_nbt_v5(&mut NbtBudget::per_packet()).is_err());
    }

    #[test]
    fn nbt_round_trips_through_gzip() {
        let mut buf = PacketBuffer::new();
        buf.write_nbt_v5(Some(&tag()));
        let len = i16::from_be_bytes([buf.as_slice()[0], buf.as_slice()[1]]) as usize;
        assert_eq!(buf.as_slice().len(), 2 + len);
        assert_eq!(&buf.as_slice()[2..4], &[0x1F, 0x8B]);
        let mut read = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(
            read.read_nbt_v5(&mut NbtBudget::per_packet()).unwrap(),
            Some(tag())
        );
        assert!(read.is_empty());
    }

    /// Every reason the 1.7.10 writer refuses a compound: the refusal is reported through the
    /// 1.8.9 writer's deferred error, so this also pins that it refuses each of them.
    #[test]
    fn unencodable_nbt_is_a_deferred_error_that_writes_nothing() {
        let mut long_key = NbtCompound::new();
        long_key.insert("x".repeat(65_536), Nbt::Byte(1));
        let mut mixed = NbtCompound::new();
        mixed.insert("m", Nbt::list(vec![Nbt::Byte(1), Nbt::Short(2)]));
        let mut end_value = NbtCompound::new();
        end_value.insert("e", Nbt::End);
        let mut deep = NbtCompound::new();
        for _ in 0..=crate::v5::nbt::MAX_DEPTH {
            let mut outer = NbtCompound::new();
            outer.insert("c", Nbt::Compound(deep));
            deep = outer;
        }
        for bad in [long_key, mixed, end_value, deep] {
            let mut buf = PacketBuffer::new();
            buf.write_nbt_v5(Some(&bad));
            assert!(matches!(buf.finish_write(), Err(ProtoError::NbtEncode(_))));
            assert!(buf.as_slice().is_empty());
        }
    }

    #[test]
    fn item_stack_layout_matches_java() {
        let stack = ItemStack {
            item_id: 276,
            count: 1,
            damage: 3,
            nbt: None,
        };
        let mut buf = PacketBuffer::new();
        buf.write_item_v5(Some(&stack));
        // id, count, damage, then FF FF for "no tag" (1.8.9 writes a single 00).
        assert_eq!(buf.as_slice(), &[0x01, 0x14, 0x01, 0x00, 0x03, 0xFF, 0xFF]);
        let mut read = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(
            read.read_item_v5(&mut NbtBudget::per_packet()).unwrap(),
            Some(stack)
        );

        let mut empty = PacketBuffer::new();
        empty.write_item_v5(None);
        assert_eq!(empty.as_slice(), &[0xFF, 0xFF]);
        // Any negative id is an empty slot and reads nothing more.
        let mut read = PacketBuffer::from_slice(&[0x80, 0x00]);
        assert_eq!(
            read.read_item_v5(&mut NbtBudget::per_packet()).unwrap(),
            None
        );
        assert!(read.is_empty());
    }

    #[test]
    fn negative_damage_is_clamped_and_count_is_signed() {
        let mut read = PacketBuffer::from_slice(&[0x00, 0x01, 0xC8, 0xFF, 0xFF, 0xFF, 0xFF]);
        let stack = read
            .read_item_v5(&mut NbtBudget::per_packet())
            .unwrap()
            .unwrap();
        assert_eq!(stack.count, -56);
        assert_eq!(stack.damage, 0);
    }

    #[test]
    fn item_with_nbt_round_trips() {
        let stack = ItemStack {
            item_id: 1,
            count: 64,
            damage: 0,
            nbt: Some(tag()),
        };
        let mut buf = PacketBuffer::new();
        buf.write_item_v5(Some(&stack));
        let mut read = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(
            read.read_item_v5(&mut NbtBudget::per_packet()).unwrap(),
            Some(stack)
        );
        assert!(read.is_empty());
    }
}
