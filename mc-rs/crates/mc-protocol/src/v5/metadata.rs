//! Protocol-5 entity metadata (`DataWatcher` lists).
//!
//! The framing is 1.8.9's: entries of one header byte `(type << 5 | index & 31) & 255` and a
//! typed payload, ended by `0x7F`. The differences are in the types:
//!
//! * there are seven, `0`-`6` (`DataWatcher.dataTypes`); type 6 is `ChunkCoordinates`, three
//!   ints, the same bytes as 1.8.9's `BlockPos` slot;
//! * the item stack of type 5 carries gzip NBT ([`super::buffer::PacketBufferV5`]);
//! * a header with type 7 has no case in `readWatchedListFromPacketBuffer`: Java appends a
//!   `null` entry and consumes nothing else, and the handler later dereferences it. That entry
//!   is kept here as [`MetadataValue5::Null`] so nothing is silently dropped;
//! * a list that is just the terminator decodes as `null`; it is an empty [`Metadata5`] here,
//!   which every handler treats the same way (each one skips a `null` list).
//!
//! java17: `entity/DataWatcher.java` (`writeWatchableObjectToPacketBuffer`,
//! `readWatchedListFromPacketBuffer`, `writeWatchedListToPacketBuffer`, `func_151509_a`)

use mc_math::BlockPos;

use super::buffer::PacketBufferV5;
use super::nbt::NbtBudget;
use crate::item::ItemStack;
use crate::metadata::{MetadataEntry, MetadataValue};
use crate::{PacketBuffer, ProtoError};

/// One watched value as protocol 5 carries it.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue5 {
    /// Type 0.
    Byte(i8),
    /// Type 1.
    Short(i16),
    /// Type 2.
    Int(i32),
    /// Type 3.
    Float(f32),
    /// Type 4, `readStringFromBuffer(32767)`.
    String(String),
    /// Type 5; `None` is an empty slot.
    ItemStack(Option<ItemStack>),
    /// Type 6, `ChunkCoordinates`: three ints.
    Coordinates(BlockPos),
    /// A type-7 header: Java's `null` entry, no payload.
    Null,
}

impl MetadataValue5 {
    /// The wire type tag, the top three bits of the header byte.
    #[must_use]
    pub fn type_tag(&self) -> u8 {
        match self {
            MetadataValue5::Byte(_) => 0,
            MetadataValue5::Short(_) => 1,
            MetadataValue5::Int(_) => 2,
            MetadataValue5::Float(_) => 3,
            MetadataValue5::String(_) => 4,
            MetadataValue5::ItemStack(_) => 5,
            MetadataValue5::Coordinates(_) => 6,
            MetadataValue5::Null => 7,
        }
    }

    /// The same value in the 1.8.9 model, which has every protocol-5 type but `Null`. The index
    /// it belongs to is a separate question: 1.7.10 and 1.8.9 register different indices per
    /// entity.
    #[must_use]
    pub fn to_v47(&self) -> Option<MetadataValue> {
        Some(match self {
            MetadataValue5::Byte(v) => MetadataValue::Byte(*v),
            MetadataValue5::Short(v) => MetadataValue::Short(*v),
            MetadataValue5::Int(v) => MetadataValue::Int(*v),
            MetadataValue5::Float(v) => MetadataValue::Float(*v),
            MetadataValue5::String(v) => MetadataValue::String(v.clone()),
            MetadataValue5::ItemStack(v) => MetadataValue::ItemStack(v.clone()),
            MetadataValue5::Coordinates(p) => MetadataValue::BlockPos(*p),
            MetadataValue5::Null => return None,
        })
    }
}

/// One entry: its index (`0..=31`) and value.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataEntry5 {
    /// `WatchableObject.dataValueId`, the low five bits of the header.
    pub index: u8,
    /// The value.
    pub value: MetadataValue5,
}

impl MetadataEntry5 {
    /// The entry in the 1.8.9 model with the same index, `None` for [`MetadataValue5::Null`].
    #[must_use]
    pub fn to_v47(&self) -> Option<MetadataEntry> {
        Some(MetadataEntry {
            index: self.index,
            value: self.value.to_v47()?,
        })
    }
}

/// A metadata list in wire order. Empty is Java's `null` list.
pub type Metadata5 = Vec<MetadataEntry5>;

/// java17: `DataWatcher.readWatchedListFromPacketBuffer`
///
/// # Errors
///
/// Whatever a field read reports.
pub fn read(buf: &mut PacketBuffer, budget: &mut NbtBudget) -> Result<Metadata5, ProtoError> {
    let mut out = Vec::new();
    loop {
        let header = buf.read_u8()?;
        if header == 0x7F {
            return Ok(out);
        }
        let index = header & 0x1F;
        let value = match header >> 5 {
            0 => MetadataValue5::Byte(buf.read_i8()?),
            1 => MetadataValue5::Short(buf.read_i16()?),
            2 => MetadataValue5::Int(buf.read_i32()?),
            3 => MetadataValue5::Float(buf.read_f32()?),
            4 => MetadataValue5::String(buf.read_string(32767)?),
            5 => MetadataValue5::ItemStack(buf.read_item_v5(budget)?),
            6 => {
                let x = buf.read_i32()?;
                let y = buf.read_i32()?;
                let z = buf.read_i32()?;
                MetadataValue5::Coordinates(BlockPos::new(x, y, z))
            }
            _ => MetadataValue5::Null,
        };
        out.push(MetadataEntry5 { index, value });
    }
}

/// java17: `DataWatcher.writeWatchedListToPacketBuffer` (and `func_151509_a`, which writes a
/// watcher's full list the same way): every entry, then `0x7F`.
pub fn write(entries: &[MetadataEntry5], buf: &mut PacketBuffer) {
    for entry in entries {
        buf.write_u8((entry.value.type_tag() << 5) | (entry.index & 0x1F));
        match &entry.value {
            MetadataValue5::Byte(v) => buf.write_i8(*v),
            MetadataValue5::Short(v) => buf.write_i16(*v),
            MetadataValue5::Int(v) => buf.write_i32(*v),
            MetadataValue5::Float(v) => buf.write_f32(*v),
            MetadataValue5::String(v) => buf.write_string(v),
            MetadataValue5::ItemStack(v) => buf.write_item_v5(v.as_ref()),
            MetadataValue5::Coordinates(p) => {
                buf.write_i32(p.x);
                buf.write_i32(p.y);
                buf.write_i32(p.z);
            }
            MetadataValue5::Null => {}
        }
    }
    buf.write_u8(0x7F);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(entries: &Metadata5) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        write(entries, &mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(
            &read(&mut read_buf, &mut NbtBudget::per_packet()).unwrap(),
            entries
        );
        assert!(read_buf.is_empty());
        buf.as_slice().to_vec()
    }

    #[test]
    fn every_type_round_trips() {
        round_trip(&vec![
            MetadataEntry5 {
                index: 0,
                value: MetadataValue5::Byte(-1),
            },
            MetadataEntry5 {
                index: 1,
                value: MetadataValue5::Short(300),
            },
            MetadataEntry5 {
                index: 12,
                value: MetadataValue5::Int(-24000),
            },
            MetadataEntry5 {
                index: 6,
                value: MetadataValue5::Float(20.0),
            },
            MetadataEntry5 {
                index: 10,
                value: MetadataValue5::String("Grumm".into()),
            },
            MetadataEntry5 {
                index: 2,
                value: MetadataValue5::ItemStack(Some(ItemStack {
                    item_id: 1,
                    count: 1,
                    damage: 0,
                    nbt: None,
                })),
            },
            MetadataEntry5 {
                index: 31,
                value: MetadataValue5::Coordinates(BlockPos::new(-1, 64, 100)),
            },
        ]);
    }

    #[test]
    fn empty_list_is_the_terminator() {
        assert_eq!(round_trip(&Vec::new()), vec![0x7F]);
    }

    #[test]
    fn type_7_is_a_null_entry_that_consumes_nothing() {
        // type 7 index 4, then a byte entry: 1.8.9 would read three floats after 0xE4.
        let wire = [0xE4, 0x00, 0x05, 0x7F];
        let mut buf = PacketBuffer::from_slice(&wire);
        let entries = read(&mut buf, &mut NbtBudget::per_packet()).unwrap();
        assert_eq!(
            entries,
            vec![
                MetadataEntry5 {
                    index: 4,
                    value: MetadataValue5::Null
                },
                MetadataEntry5 {
                    index: 0,
                    value: MetadataValue5::Byte(5)
                },
            ]
        );
        assert_eq!(round_trip(&entries), wire);
        assert!(entries[0].to_v47().is_none());
    }

    #[test]
    fn truncated_payload_errors() {
        let mut buf = PacketBuffer::from_slice(&[0x40]);
        assert!(read(&mut buf, &mut NbtBudget::per_packet()).is_err());
        let mut buf = PacketBuffer::from_slice(&[]);
        assert!(read(&mut buf, &mut NbtBudget::per_packet()).is_err());
    }
}
