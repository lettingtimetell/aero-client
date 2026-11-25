//! 1.7.10 chunk packets in the client's 1.8.9 chunk form.
//!
//! A protocol-5 `S21`/`S26` arrives inflated (`mc_protocol::v5::chunk` inflates while decoding,
//! on the network thread, exactly as `readPacketData` does) and is rewritten here into the 1.8.9
//! `S21`/`S26` the client already applies: for every section its block states (each cell through
//! `mc_data::v5`'s block decode), then every section's block light, every section's sky light
//! when the dimension has sky, then the biomes of a full column. That is the layout
//! `Chunk::fill_from_packet` reads. The light and biome bytes are copied unchanged: both versions'
//! `NibbleArray` index `y << 8 | z << 4 | x` with the low nibble first, and the handlers of both
//! packets are the 1.8.9 ones (java17 `NetHandlerPlayClient.handleChunkData` and
//! `handleMapChunkBulk` = java18's), so the existing apply path, the background
//! column unpack (`sync::prebuild`) and the renderer see nothing new.
//!
//! java17: `Chunk.fillChunk(byte[], int, int, boolean)`, `S21PacketChunkData`,
//! `S26PacketMapChunkBulk`.
//!
//! # The sky flag
//!
//! `fillChunk` reads sky light by the client's dimension (`!worldObj.provider.hasNoSky`), not by
//! the packet. The network thread knows the dimension from `JoinGame`/`Respawn` in stream order,
//! the same way the 1.8.9 path does, and the game thread rechecks it before it installs a full
//! `S21` (`sync::dispatch::apply_prebuilt_chunks`). The two can only disagree after the game
//! thread rejected a `JoinGame`/`Respawn` for a dimension without a provider, which vanilla
//! 1.7.10 does not survive (the `WorldClient` constructor throws), so the converted bytes are not
//! kept for a second conversion. Partial `S21`s are converted on the game thread instead, with
//! the world's own flag ([`chunk_data_v47`] from `sync::dispatch_v5`).
//!
//! # What a vanilla 1.7.10 server never sends, handled as the 1.8.9 layout can
//!
//! Block ids from 256 up (the add nibbles) are air in 1.7.10 and here alike. Two `fillChunk`
//! effects on sections the packet does not rewrite are not reproduced: add nibbles for a section
//! outside the primary mask (copied into an existing section), and a partial update keeping a
//! section's previous add nibbles. Both only change ids from 256 up, which no 1.7.10 block has.

use mc_protocol::chunk as c47;
use mc_protocol::play::S2CPlay;
use mc_protocol::v5::chunk::{ChunkData, ColumnView, MapChunkBulk};
use mc_protocol::v5::S2CPlay5;
use mc_protocol::ProtoError;

use crate::net::channel::ToGame;
use crate::sync::prebuild::PrebuildingChunkPacket;

/// Bytes per section in the 1.8.9 block region: two per block.
const SECTION_BLOCK_BYTES: usize = 8192;

/// The message the network thread queues for one decoded protocol-5 packet.
///
/// A full-column chunk packet (`S21` ground-up with sections, or any `S26`) is converted now and
/// starts unpacking on the background thread ([`ToGame::ChunkPacketV5`]), in the same queue
/// position. Anything else, including a chunk packet that cannot be converted with this layout
/// or when no background thread exists, travels as [`ToGame::PacketV5`]: the game thread then
/// converts it, or reports why it cannot, at the packet's turn.
#[must_use]
pub(crate) fn message_v5(packet: S2CPlay5, dimension_has_sky: bool) -> ToGame {
    let converted = match &packet {
        S2CPlay5::ChunkData(chunk) if chunk.ground_up_continuous && chunk.primary_bit_mask != 0 => {
            chunk_data_v47(chunk, dimension_has_sky).map(S2CPlay::ChunkData)
        }
        S2CPlay5::MapChunkBulk(bulk) => {
            map_chunk_bulk_v47(bulk, dimension_has_sky).map(S2CPlay::MapChunkBulk)
        }
        _ => return ToGame::PacketV5(Box::new(packet)),
    };
    let Ok(v47) = converted else {
        return ToGame::PacketV5(Box::new(packet));
    };
    let bytes = match &v47 {
        S2CPlay::ChunkData(chunk) => chunk.data.len(),
        S2CPlay::MapChunkBulk(bulk) => bulk.columns.iter().map(|column| column.data.len()).sum(),
        _ => 0,
    };
    match PrebuildingChunkPacket::spawn(v47, dimension_has_sky) {
        Ok(job) => ToGame::ChunkPacketV5(job, bytes),
        Err(_) => ToGame::PacketV5(Box::new(packet)),
    }
}

/// The 1.8.9 `S21` with the same effect as a protocol-5 one, laid out for a dimension with or
/// without sky.
///
/// # Errors
///
/// [`ProtoError::Eof`] where `Chunk.fillChunk` would run off the inflated array (only a packet
/// whose masks the data does not fill).
pub(crate) fn chunk_data_v47(
    chunk: &ChunkData,
    dimension_has_sky: bool,
) -> Result<c47::ChunkData, ProtoError> {
    let column = chunk.column(dimension_has_sky)?;
    Ok(c47::ChunkData {
        chunk_x: chunk.chunk_x,
        chunk_z: chunk.chunk_z,
        ground_up_continuous: chunk.ground_up_continuous,
        primary_bit_mask: chunk.primary_bit_mask,
        data: column_blob(&column, dimension_has_sky)?,
    })
}

/// The 1.8.9 `S26` with the same effect as a protocol-5 one. Its sky flag is the layout written,
/// the dimension's: each column was sliced by the packet's own flag and is read by the
/// dimension's, as `fillChunk` reads it.
///
/// # Errors
///
/// As [`chunk_data_v47`], for any column.
pub(crate) fn map_chunk_bulk_v47(
    bulk: &MapChunkBulk,
    dimension_has_sky: bool,
) -> Result<c47::MapChunkBulk, ProtoError> {
    let columns = bulk
        .columns
        .iter()
        .map(|column| {
            Ok(c47::ChunkBulkColumn {
                chunk_x: column.chunk_x,
                chunk_z: column.chunk_z,
                primary_bit_mask: column.primary_bit_mask,
                data: column_blob(&column.column(dimension_has_sky)?, dimension_has_sky)?,
            })
        })
        .collect::<Result<Vec<_>, ProtoError>>()?;
    Ok(c47::MapChunkBulk {
        has_sky_light: dimension_has_sky,
        columns,
    })
}

/// One parsed protocol-5 column as the 1.8.9 blob: blocks, block light, sky light (with sky),
/// biomes (full columns).
fn column_blob(column: &ColumnView<'_>, has_sky: bool) -> Result<Vec<u8>, ProtoError> {
    let sections = column.sections.len();
    let light = sections * 2048;
    let mut blob = Vec::with_capacity(
        sections * SECTION_BLOCK_BYTES
            + light
            + if has_sky { light } else { 0 }
            + column.biomes.map_or(0, <[u8]>::len),
    );
    let mut blocks = [0u8; SECTION_BLOCK_BYTES];
    for section in &column.sections {
        let short = || ProtoError::Eof("chunk section");
        let lsb = <&[u8; 4096]>::try_from(section.blocks).map_err(|_| short())?;
        let meta = <&[u8; 2048]>::try_from(section.metadata).map_err(|_| short())?;
        let add = section
            .add
            .map(<&[u8; 2048]>::try_from)
            .transpose()
            .map_err(|_| short())?;
        mc_data::v5::write_section_blob_v5(lsb, meta, add, &mut blocks);
        blob.extend_from_slice(&blocks);
    }
    for section in &column.sections {
        blob.extend_from_slice(section.block_light);
    }
    if has_sky {
        for section in &column.sections {
            let sky = section
                .sky_light
                .ok_or(ProtoError::Eof("chunk section sky light"))?;
            blob.extend_from_slice(sky);
        }
    }
    if let Some(biomes) = column.biomes {
        blob.extend_from_slice(biomes);
    }
    Ok(blob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::BlockState;
    use mc_world::Chunk;

    /// Uncompressed protocol-5 column bytes: section `y` holds block id `ids[y]` with meta 2
    /// everywhere, block light 0x54, sky light 0x76 (with sky), add nibbles 0x01 for sections in
    /// `add`, biome 7.
    fn raw_column(ids: &[(u8, u8)], add: u16, sky: bool, ground_up: bool) -> (u16, Vec<u8>) {
        let primary = ids.iter().fold(0u16, |mask, &(y, _)| mask | 1 << y);
        let mut raw = Vec::new();
        for &(_, id) in ids {
            raw.extend(std::iter::repeat_n(id, 4096));
        }
        for _ in ids {
            raw.extend(std::iter::repeat_n(0x22u8, 2048));
        }
        for _ in ids {
            raw.extend(std::iter::repeat_n(0x54u8, 2048));
        }
        if sky {
            for _ in ids {
                raw.extend(std::iter::repeat_n(0x76u8, 2048));
            }
        }
        raw.extend(std::iter::repeat_n(
            0x11u8,
            2048 * add.count_ones() as usize,
        ));
        if ground_up {
            raw.extend(std::iter::repeat_n(7u8, 256));
        }
        (primary, raw)
    }

    /// Every cell of a converted column is the 1.8.9 state `mc_data::v5` decodes the 1.7.10
    /// cell to, and the light and biomes are the packet's own bytes.
    #[test]
    fn a_full_column_becomes_the_1_8_9_blob_with_decoded_cells() {
        // Stone (1:2 granite in 1.8.9, plain stone in 1.7.10) and a torch (50:2).
        let (primary, raw) = raw_column(&[(0, 1), (4, 50)], 0, true, true);
        let packet = ChunkData::from_raw(3, -4, true, primary, 0, &raw);
        let v47 = chunk_data_v47(&packet, true).expect("converts");
        assert_eq!((v47.chunk_x, v47.chunk_z), (3, -4));
        assert!(v47.ground_up_continuous);
        assert_eq!(v47.primary_bit_mask, primary);
        assert_eq!(
            v47.data.len(),
            mc_world::expected_chunk_data_len(primary, true, true)
        );

        let chunk = Chunk::from_full_column_packet(primary, true, &v47.data).expect("unpacks");
        let stone = mc_data::v5::state_from_v5(1 << 4 | 2);
        let torch = mc_data::v5::state_from_v5(50 << 4 | 2);
        assert_eq!(
            stone,
            BlockState::new(mc_data::BlockId(1), 0),
            "1.7.10 stone ignores meta"
        );
        assert_eq!(chunk.block_at(0, 0, 0), stone);
        assert_eq!(chunk.block_at(15, 15, 15), stone);
        assert_eq!(chunk.block_at(3, 64, 9), torch);
        assert_eq!(
            chunk.block_at(0, 16, 0),
            BlockState::AIR,
            "section 1 was not sent"
        );
        // Light: the nibble bytes, unchanged.
        assert_eq!(&v47.data[2 * 8192..2 * 8192 + 2048], &[0x54; 2048][..]);
        let sky_at = 2 * 8192 + 2 * 2048;
        assert_eq!(&v47.data[sky_at..sky_at + 2048], &[0x76; 2048][..]);
        assert_eq!(&v47.data[v47.data.len() - 256..], &[7u8; 256][..]);
    }

    /// Without sky the 1.8.9 layout has no sky region, and the add nibbles make ids 256+ (air).
    #[test]
    fn no_sky_layout_and_add_nibbles() {
        let (primary, raw) = raw_column(&[(2, 1)], 0b100, false, true);
        let packet = ChunkData::from_raw(0, 0, true, primary, 0b100, &raw);
        let v47 = chunk_data_v47(&packet, false).expect("converts");
        assert_eq!(
            v47.data.len(),
            mc_world::expected_chunk_data_len(primary, false, true)
        );
        let chunk = Chunk::from_full_column_packet(primary, false, &v47.data).expect("unpacks");
        assert_eq!(
            chunk.block_at(0, 32, 0),
            BlockState::AIR,
            "id 0x101 is not a 1.7.10 block"
        );
    }

    /// The bulk keeps its columns in order, one blob each, flagged with the layout written.
    #[test]
    fn a_bulk_converts_every_column() {
        let (pa, a) = raw_column(&[(0, 1)], 0, true, true);
        let (pb, b) = raw_column(&[(1, 2), (3, 3)], 0, true, true);
        let bulk =
            MapChunkBulk::from_raw(true, &[(1, 2, pa, 0, &a), (-1, -2, pb, 0, &b)]).expect("bulk");
        let v47 = map_chunk_bulk_v47(&bulk, true).expect("converts");
        assert!(v47.has_sky_light);
        assert_eq!(v47.columns.len(), 2);
        assert_eq!((v47.columns[1].chunk_x, v47.columns[1].chunk_z), (-1, -2));
        assert_eq!(v47.columns[1].primary_bit_mask, pb);
        let column = Chunk::from_full_column_packet(pb, true, &v47.columns[1].data).expect("ok");
        assert_eq!(
            column.block_at(5, 16, 5),
            mc_data::v5::state_from_v5(2 << 4 | 2)
        );
        assert_eq!(
            column.block_at(5, 48, 5),
            mc_data::v5::state_from_v5(3 << 4 | 2)
        );
    }

    /// A column whose masks the data does not fill is not converted: it travels as the
    /// protocol-5 packet for the game thread to report. A full column starts unpacking.
    #[test]
    fn unreadable_columns_stay_protocol_5_packets() {
        let (primary, raw) = raw_column(&[(0, 1)], 0, true, true);
        let bulk = MapChunkBulk::from_raw(true, &[(0, 0, primary, 0, &raw)]).expect("bulk");
        assert!(matches!(
            message_v5(S2CPlay5::MapChunkBulk(bulk.clone()), true),
            ToGame::ChunkPacketV5(..)
        ));
        // Sliced without sky but read with sky: the slice is too short, as in `fillChunk`.
        let (primary, raw) = raw_column(&[(0, 1)], 0, false, true);
        let short = MapChunkBulk::from_raw(false, &[(0, 0, primary, 0, &raw)]).expect("bulk");
        assert!(map_chunk_bulk_v47(&short, true).is_err());
        assert!(matches!(
            message_v5(S2CPlay5::MapChunkBulk(short), true),
            ToGame::PacketV5(_)
        ));
        // A partial S21 is the game thread's to convert, with the world's own flag.
        let partial = ChunkData::from_raw(0, 0, false, 1, 0, &raw[..raw.len() - 256]);
        assert!(matches!(
            message_v5(S2CPlay5::ChunkData(partial), true),
            ToGame::PacketV5(_)
        ));
    }
}
