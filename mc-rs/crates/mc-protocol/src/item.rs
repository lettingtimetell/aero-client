//! [`ItemStack`] wire codec.
//!
//! java: `PacketBuffer.writeItemStackToBuffer` / `readItemStackFromBuffer`

use mc_nbt::NbtCompound;

use crate::buffer::PacketBuffer;
use crate::ProtoError;

/// A stack of items exactly as it appears on the wire: registry id, count, damage/metadata,
/// and an optional NBT tag.
///
/// An empty item slot is `Option<ItemStack>::None`, written as the sentinel id `-1`
/// (`writeShort(-1)`) — there is no "empty" `ItemStack` value, matching vanilla's `null`.
///
/// Resolving `item_id` to a game item, and any item-specific behaviour, is `mc-data`'s job;
/// this crate only carries the raw wire fields (`mc-protocol` does not depend on `mc-data`).
#[derive(Debug, Clone, PartialEq)]
pub struct ItemStack {
    /// The item's registry id (`Item.getIdFromItem`).
    pub item_id: i16,
    /// Stack size (`ItemStack.stackSize`). The wire representation is a signed byte, widened
    /// to Java's `int` on read; it is narrowed back to its low eight bits on write.
    pub count: i32,
    /// Damage value / metadata (`ItemStack.getMetadata`).
    pub damage: i16,
    /// The stack's NBT tag, if any.
    ///
    /// Vanilla reads this field unconditionally (`readItemStackFromBuffer`) and writes it
    /// unconditionally too, despite appearances — see the `write` doc for why the gate that
    /// looks like an elision is dead code in 1.8.9.
    pub nbt: Option<NbtCompound>,
}

impl ItemStack {
    /// Decode an optional item stack. java: `readItemStackFromBuffer`
    pub fn read(buf: &mut PacketBuffer) -> Result<Option<ItemStack>, ProtoError> {
        let item_id = buf.read_i16()?;
        if item_id < 0 {
            return Ok(None);
        }
        let count = i32::from(buf.read_i8()?);
        // PacketBuffer passes the signed short to ItemStack's constructor, which clamps a
        // negative itemDamage to zero before the packet reaches its handler.
        let damage = buf.read_i16()?.max(0);
        let nbt = buf.read_nbt_compound_opt()?;
        Ok(Some(ItemStack {
            item_id,
            count,
            damage,
            nbt,
        }))
    }

    /// Encode an optional item stack.
    ///
    /// Writes `nbt` exactly as given, which is what vanilla does — **this is not a deviation,
    /// and an earlier version of this comment said it was.**
    ///
    /// `writeItemStackToBuffer` guards the tag with
    /// `stack.getItem().isDamageable() || stack.getItem().getShareTag()`, which reads like an
    /// elision for plain items. It is not: `Item.getShareTag()` is `return true`
    /// (`Item.java:316-319`) and **nothing in the 1.8.9 tree overrides it** — the only other
    /// occurrence of the name in `src/minecraft` is that call site. The `||` is therefore always
    /// true and the tag is always written. Adding a gate here to "match vanilla" would create a
    /// divergence rather than remove one.
    ///
    /// Pinned by the W-CAP fixtures `place_stack_with_nbt` (damageable) and
    /// `place_stack_nbt_nondamageable` (not damageable, tag still present), both byte-identical
    /// to the real client's output. java: `writeItemStackToBuffer`
    pub fn write(stack: Option<&ItemStack>, buf: &mut PacketBuffer) {
        match stack {
            None => buf.write_i16(-1),
            Some(s) => {
                buf.write_i16(s.item_id);
                buf.write_i8(s.count as i8);
                buf.write_i16(s.damage);
                buf.write_nbt_compound_opt(s.nbt.as_ref());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stack_round_trips() {
        let mut buf = PacketBuffer::new();
        ItemStack::write(None, &mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(ItemStack::read(&mut read_buf).unwrap(), None);
    }

    #[test]
    fn stack_without_nbt_round_trips() {
        let stack = ItemStack {
            item_id: 1,
            count: 64,
            damage: 0,
            nbt: None,
        };
        let mut buf = PacketBuffer::new();
        ItemStack::write(Some(&stack), &mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(ItemStack::read(&mut read_buf).unwrap(), Some(stack));
    }

    #[test]
    fn negative_wire_damage_is_clamped_to_zero() {
        let mut buf = PacketBuffer::new();
        buf.write_i16(1);
        buf.write_u8(1);
        buf.write_i16(-1);
        buf.write_nbt_compound_opt(None);

        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let stack = ItemStack::read(&mut read_buf).unwrap().unwrap();
        assert_eq!(stack.damage, 0);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn wire_count_is_a_signed_byte_widened_to_java_int() {
        // Java's `readByte()` sign-extends all 256 byte patterns into its `int` stackSize.
        // Check the full domain, including both sides of the sign boundary, rather than only
        // valid server-produced stack sizes (0..=64).
        for wire in u8::MIN..=u8::MAX {
            let expected = i32::from(wire as i8);
            let mut buf = PacketBuffer::new();
            buf.write_i16(1);
            buf.write_u8(wire);
            buf.write_i16(0);
            buf.write_nbt_compound_opt(None);

            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let stack = ItemStack::read(&mut read_buf).unwrap().unwrap();
            assert_eq!(stack.count, expected, "wire byte {wire:#04x}");
            assert!(read_buf.is_empty());

            let mut written = PacketBuffer::new();
            ItemStack::write(Some(&stack), &mut written);
            assert_eq!(written.as_slice()[2], wire, "wire byte {wire:#04x}");
        }
    }

    #[test]
    fn signed_count_oracle_rejects_the_old_unsigned_decode() {
        // Negative control: these high-bit patterns are precisely where `read_u8()` diverges
        // from Java's `readByte()`. This must not be narrowed to ordinary 0..=64 stack sizes.
        for wire in [0x80, 0xc8, 0xff] {
            let expected = i32::from(wire as i8);
            let old_unsigned_decode = wire;
            assert_ne!(expected, old_unsigned_decode, "wire byte {wire:#04x}");
        }
    }

    /// java: `readItemStackFromBuffer` reads the tag's strings with `readUTF`, which keeps a lone
    /// surrogate (half an emoji cut off by the 30-character anvil name limit, for instance).
    /// Decoding one used to fail the whole packet, which disconnected the client. The stack
    /// must decode and write back the very bytes the server sent.
    #[test]
    fn a_lone_surrogate_in_a_display_name_decodes_and_writes_back_identically() {
        let mut wire = vec![0x01, 0x14, 0x01, 0x00, 0x00]; // id 276, count 1, damage 0
        wire.extend_from_slice(&[0x0A, 0x00, 0x00]); // root compound, empty name
        wire.extend_from_slice(&[0x0A, 0x00, 0x07]);
        wire.extend_from_slice(b"display");
        wire.extend_from_slice(&[0x08, 0x00, 0x04]);
        wire.extend_from_slice(b"Name");
        wire.extend_from_slice(&[0x00, 0x09]);
        wire.extend_from_slice(b"Sword ");
        wire.extend_from_slice(&[0xED, 0xA0, 0xBD]); // lone U+D83D, as writeUTF emits it
        wire.extend_from_slice(&[0x00, 0x00]); // end of display, end of root

        let mut read_buf = PacketBuffer::from_slice(&wire);
        let stack = ItemStack::read(&mut read_buf).unwrap().expect("a stack");
        assert!(read_buf.is_empty());
        let name = stack
            .nbt
            .as_ref()
            .and_then(|tag| tag.get_compound("display"))
            .and_then(|display| display.get_str("Name"));
        assert_eq!(name, Some("Sword \u{FD83D}"));

        let mut written = PacketBuffer::new();
        ItemStack::write(Some(&stack), &mut written);
        assert_eq!(written.as_slice(), wire.as_slice());
    }

    #[test]
    fn stack_with_nbt_round_trips() {
        let mut tag = NbtCompound::new();
        tag.insert("ench", mc_nbt::Nbt::list(vec![]));
        let stack = ItemStack {
            item_id: 278,
            count: 1,
            damage: 0,
            nbt: Some(tag),
        };
        let mut buf = PacketBuffer::new();
        ItemStack::write(Some(&stack), &mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(ItemStack::read(&mut read_buf).unwrap(), Some(stack));
    }
}
