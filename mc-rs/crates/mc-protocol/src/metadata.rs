//! Entity metadata (`DataWatcher`) codec — the fiddliest part of S06.
//!
//! Eight typed slots (`0..=31`), each packed as one header byte `(type << 5) | index`
//! followed by a type-specific payload, terminated by a lone `0x7F` header byte.
//!
//! java: `entity/DataWatcher.java` — `writeWatchableObjectToPacketBuffer`/
//! `readWatchedListFromPacketBuffer`. Used inline by `S0CPacketSpawnPlayer`,
//! `S0FPacketSpawnMob`, and standalone by `S1CPacketEntityMetadata`.

use mc_math::BlockPos;

use crate::item::ItemStack;
use crate::{PacketBuffer, ProtoError};

/// One decoded metadata value — `DataWatcher.WatchableObject`'s eight legal payload types
/// (`DataWatcher.dataTypes`), in their wire type-tag order.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// Type 0.
    Byte(i8),
    /// Type 1.
    Short(i16),
    /// Type 2.
    Int(i32),
    /// Type 3.
    Float(f32),
    /// Type 4.
    String(String),
    /// Type 5. `None` is a legal watched value (e.g. an empty equipment slot).
    ItemStack(Option<ItemStack>),
    /// Type 6. Written as three raw big-endian `i32`s — **not** the packed 26/12/26
    /// `Position` encoding `read_block_pos`/`write_block_pos` use elsewhere (hazard H5 does
    /// not apply to this field).
    BlockPos(BlockPos),
    /// Type 7 (`Rotations`: pitch/yaw/roll, e.g. an `ArmorStand` body part, in degrees).
    Rotations(f32, f32, f32),
}

impl MetadataValue {
    /// This value's wire type tag (`0..=7`), the top 3 bits of the header byte.
    #[must_use]
    pub fn type_tag(&self) -> u8 {
        match self {
            MetadataValue::Byte(_) => 0,
            MetadataValue::Short(_) => 1,
            MetadataValue::Int(_) => 2,
            MetadataValue::Float(_) => 3,
            MetadataValue::String(_) => 4,
            MetadataValue::ItemStack(_) => 5,
            MetadataValue::BlockPos(_) => 6,
            MetadataValue::Rotations(..) => 7,
        }
    }
}

/// One watched-object slot: its index (`0..=31`, the low 5 bits of the header byte) and
/// current value.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataEntry {
    /// The data-value id (`DataWatcher.WatchableObject.dataValueId`), `0..=31`.
    pub index: u8,
    /// The slot's current value.
    pub value: MetadataValue,
}

/// A full metadata list, in wire order. Empty is legal and common (most entities watch a
/// handful of slots; an unchanged list between updates can be empty).
pub type Metadata = Vec<MetadataEntry>;

/// Decode a metadata list, stopping at the `0x7F` terminator.
/// java: `DataWatcher.readWatchedListFromPacketBuffer`
pub fn read(buf: &mut PacketBuffer) -> Result<Metadata, ProtoError> {
    let mut out = Vec::new();
    loop {
        let header = buf.read_u8()?;
        if header == 0x7F {
            return Ok(out);
        }
        let type_tag = (header & 0xE0) >> 5;
        let index = header & 0x1F;
        let value = match type_tag {
            0 => MetadataValue::Byte(buf.read_i8()?),
            1 => MetadataValue::Short(buf.read_i16()?),
            2 => MetadataValue::Int(buf.read_i32()?),
            3 => MetadataValue::Float(buf.read_f32()?),
            4 => MetadataValue::String(buf.read_string(32767)?),
            5 => MetadataValue::ItemStack(buf.read_item_stack()?),
            6 => {
                let x = buf.read_i32()?;
                let y = buf.read_i32()?;
                let z = buf.read_i32()?;
                MetadataValue::BlockPos(BlockPos::new(x, y, z))
            }
            7 => {
                let x = buf.read_f32()?;
                let y = buf.read_f32()?;
                let z = buf.read_f32()?;
                MetadataValue::Rotations(x, y, z)
            }
            other => {
                return Err(ProtoError::InvalidEnum {
                    what: "data-watcher type",
                    value: i32::from(other),
                })
            }
        };
        out.push(MetadataEntry { index, value });
    }
}

/// Encode a metadata list, appending the `0x7F` terminator.
/// java: `DataWatcher.writeWatchedListToPacketBuffer` / `writeWatchableObjectToPacketBuffer`
pub fn write(entries: &Metadata, buf: &mut PacketBuffer) {
    for entry in entries {
        let header = (entry.value.type_tag() << 5) | (entry.index & 0x1F);
        buf.write_u8(header);
        match &entry.value {
            MetadataValue::Byte(v) => buf.write_i8(*v),
            MetadataValue::Short(v) => buf.write_i16(*v),
            MetadataValue::Int(v) => buf.write_i32(*v),
            MetadataValue::Float(v) => buf.write_f32(*v),
            MetadataValue::String(v) => buf.write_string(v),
            MetadataValue::ItemStack(v) => buf.write_item_stack(v.as_ref()),
            MetadataValue::BlockPos(p) => {
                buf.write_i32(p.x);
                buf.write_i32(p.y);
                buf.write_i32(p.z);
            }
            MetadataValue::Rotations(x, y, z) => {
                buf.write_f32(*x);
                buf.write_f32(*y);
                buf.write_f32(*z);
            }
        }
    }
    buf.write_u8(0x7F);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Metadata {
        vec![
            MetadataEntry {
                index: 0,
                value: MetadataValue::Byte(-1),
            },
            MetadataEntry {
                index: 6,
                value: MetadataValue::Float(1.5),
            },
            MetadataEntry {
                index: 10,
                value: MetadataValue::String("hi".into()),
            },
            MetadataEntry {
                index: 31,
                value: MetadataValue::ItemStack(None),
            },
            MetadataEntry {
                index: 5,
                value: MetadataValue::BlockPos(BlockPos::new(-1, 64, 100)),
            },
            MetadataEntry {
                index: 2,
                value: MetadataValue::Rotations(1.0, -2.5, 0.0),
            },
        ]
    }

    #[test]
    fn round_trips() {
        let entries = sample();
        let mut buf = PacketBuffer::new();
        write(&entries, &mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(read(&mut read_buf).unwrap(), entries);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn empty_list_is_just_the_terminator() {
        let mut buf = PacketBuffer::new();
        write(&Metadata::new(), &mut buf);
        assert_eq!(buf.as_slice(), &[0x7F]);
    }

    #[test]
    fn truncated_payload_errors_not_panics() {
        // Header claims type 2 (Int, a 4-byte payload) but no payload bytes follow.
        let mut buf = PacketBuffer::from_slice(&[0x40]);
        assert!(read(&mut buf).is_err());
    }
}
