//! Chunk data packets (`0x21 ChunkData`, `0x26 MapChunkBulk`).
//!
//! These carry each chunk's section data as one opaque byte blob; unpacking it into block
//! state / light / biome arrays is `mc-world`'s job (S08's own task list: "`S21PacketChunkData`
//! unpack: bitmask, blocks LE, block light, sky light, biomes"). This module's only
//! responsibility is getting the blob's *boundaries* right, which for `MapChunkBulk` is not
//! trivial: unlike `ChunkData`, a bulk chunk's per-column data length isn't sent — it's
//! *computed* from the section-count popcount of its bitmask, exactly as vanilla computes it
//! before slicing the shared trailing blob.
//!
//! java: `network/play/server/S21PacketChunkData.java`, `S26PacketMapChunkBulk.java`

use crate::packet::Packet;
use crate::{PacketBuffer, ProtoError};

/// The size, in bytes, of one chunk column's extracted section data: `section_count` sections
/// of packed block-state shorts, plus block light, plus (conditionally) sky light nibbles,
/// plus (conditionally) one 256-byte biome array for the whole column.
///
/// java: `S21PacketChunkData.func_180737_a`
#[must_use]
pub fn extracted_data_size(section_count: u32, has_sky_light: bool, has_biomes: bool) -> usize {
    let section_count = section_count as usize;
    let blocks = section_count * 2 * 16 * 16 * 16; // char[4096] per section, 2 bytes each
    let block_light = section_count * 16 * 16 * 16 / 2; // nibble array
    let sky_light = if has_sky_light {
        section_count * 16 * 16 * 16 / 2
    } else {
        0
    };
    let biomes = if has_biomes { 256 } else { 0 };
    blocks + block_light + sky_light + biomes
}

/// `0x21 Chunk Data`. Carries one chunk column's section data as an explicit length-prefixed
/// blob — unlike `MapChunkBulk`, this packet *does* send its own size.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkData {
    /// Chunk column X (in chunks, not blocks).
    pub chunk_x: i32,
    /// Chunk column Z.
    pub chunk_z: i32,
    /// `true` for a full column send (unload-and-replace); `false` for a partial update to
    /// only the sections named in `primary_bit_mask`. A bitmask of `0` with this `true` means
    /// "unload this column."
    pub ground_up_continuous: bool,
    /// One bit per section (`0..16`, LSB = Y section 0) naming which sections are present in
    /// `data`.
    pub primary_bit_mask: u16,
    /// The extracted section/light/biome blob; see [`extracted_data_size`] for its layout.
    pub data: Vec<u8>,
}

impl Packet for ChunkData {
    const ID: i32 = 0x21;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let chunk_x = buf.read_i32()?;
        let chunk_z = buf.read_i32()?;
        let ground_up_continuous = buf.read_bool()?;
        let primary_bit_mask = buf.read_u16()?;
        let data = buf.read_byte_array()?;
        Ok(ChunkData {
            chunk_x,
            chunk_z,
            ground_up_continuous,
            primary_bit_mask,
            data,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.chunk_x);
        buf.write_i32(self.chunk_z);
        buf.write_bool(self.ground_up_continuous);
        buf.write_u16(self.primary_bit_mask);
        buf.write_byte_array(&self.data);
    }
}

/// One column within a [`MapChunkBulk`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkBulkColumn {
    /// Chunk column X.
    pub chunk_x: i32,
    /// Chunk column Z.
    pub chunk_z: i32,
    /// Section presence bitmask, same meaning as [`ChunkData::primary_bit_mask`].
    pub primary_bit_mask: u16,
    /// The extracted section/light/biome blob for this column (always `ground_up_continuous`,
    /// so it always includes the trailing 256-byte biome array).
    pub data: Vec<u8>,
}

/// `0x26 Map Chunk Bulk`: the initial-join burst of every loaded chunk column, sent as one
/// packet. Always `ground_up_continuous` per column, so every [`ChunkBulkColumn::data`]
/// includes a biome array.
#[derive(Debug, Clone, PartialEq)]
pub struct MapChunkBulk {
    /// `true` in the overworld (sky light present in every column's data); `false` in the
    /// Nether/End (hazard H6 — reading sky light unconditionally there desyncs the buffer,
    /// which is exactly what this flag exists to prevent).
    pub has_sky_light: bool,
    /// The columns, in wire order.
    pub columns: Vec<ChunkBulkColumn>,
}

impl Packet for MapChunkBulk {
    const ID: i32 = 0x26;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let has_sky_light = buf.read_bool()?;
        let count = buf.read_varint()?;
        if count < 0 {
            return Err(ProtoError::NegativeLength(count));
        }
        let count = count as usize;

        // Pass 1: read every column's fixed-size header (position + bitmask), and from each
        // bitmask compute how many data bytes that column will occupy — vanilla does not send
        // this length, only the bitmask it's derived from.
        struct Header {
            chunk_x: i32,
            chunk_z: i32,
            primary_bit_mask: u16,
            data_len: usize,
        }
        let mut headers = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            let chunk_x = buf.read_i32()?;
            let chunk_z = buf.read_i32()?;
            let primary_bit_mask = buf.read_u16()?;
            let data_len = extracted_data_size(primary_bit_mask.count_ones(), has_sky_light, true);
            headers.push(Header {
                chunk_x,
                chunk_z,
                primary_bit_mask,
                data_len,
            });
        }

        // Pass 2: the data blobs follow *all* the headers, concatenated in the same order.
        let mut columns = Vec::with_capacity(headers.len());
        for h in headers {
            let data = buf.read_bytes(h.data_len)?;
            columns.push(ChunkBulkColumn {
                chunk_x: h.chunk_x,
                chunk_z: h.chunk_z,
                primary_bit_mask: h.primary_bit_mask,
                data,
            });
        }

        Ok(MapChunkBulk {
            has_sky_light,
            columns,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_bool(self.has_sky_light);
        buf.write_varint(self.columns.len() as i32);
        for c in &self.columns {
            buf.write_i32(c.chunk_x);
            buf.write_i32(c.chunk_z);
            buf.write_u16(c.primary_bit_mask);
        }
        for c in &self.columns {
            buf.write_bytes(&c.data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_size_matches_known_formula() {
        // One full section (16 present), overworld, with the trailing biome array: vanilla's
        // 8192 (blocks) + 2048 (block light) + 2048 (sky light) + 256 (biomes) = 12544.
        assert_eq!(extracted_data_size(1, true, true), 12_544);
        // Nether (no sky light), no biome array (partial update).
        assert_eq!(extracted_data_size(1, false, false), 8192 + 2048);
        assert_eq!(extracted_data_size(0, true, true), 256);
    }

    #[test]
    fn chunk_data_round_trips() {
        let pkt = ChunkData {
            chunk_x: -3,
            chunk_z: 12,
            ground_up_continuous: true,
            primary_bit_mask: 0b0000_0000_0010_0101,
            data: vec![1, 2, 3, 4, 5],
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(ChunkData::read(&mut read_buf).unwrap(), pkt);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn map_chunk_bulk_round_trips_with_mixed_bitmasks() {
        let make_column = |x: i32, z: i32, mask: u16, sky: bool| {
            let len = extracted_data_size(mask.count_ones(), sky, true);
            ChunkBulkColumn {
                chunk_x: x,
                chunk_z: z,
                primary_bit_mask: mask,
                data: (0..len).map(|i| (i % 251) as u8).collect(),
            }
        };
        let pkt = MapChunkBulk {
            has_sky_light: true,
            columns: vec![
                make_column(0, 0, 0xFFFF, true),  // every section present
                make_column(1, 0, 0x0001, true),  // just the bottom section
                make_column(-2, 5, 0x0000, true), // no sections, just biomes
            ],
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(MapChunkBulk::read(&mut read_buf).unwrap(), pkt);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn map_chunk_bulk_nether_omits_sky_light() {
        let len = extracted_data_size(1, false, true);
        let pkt = MapChunkBulk {
            has_sky_light: false,
            columns: vec![ChunkBulkColumn {
                chunk_x: 0,
                chunk_z: 0,
                primary_bit_mask: 0x0001,
                data: vec![0xAB; len],
            }],
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(MapChunkBulk::read(&mut read_buf).unwrap(), pkt);
    }
}
