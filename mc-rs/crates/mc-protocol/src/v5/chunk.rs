//! Protocol-5 chunk packets: `0x21 ChunkData` and `0x26 MapChunkBulk`.
//!
//! Both carry zlib data that `readPacketData` inflates on the network thread, so decoding here
//! inflates too, exactly as Java does (and fails where Java throws
//! `IOException("Bad compressed data format")`):
//!
//! * `new Inflater()` (zlib wrapper) over the received bytes, **one** `inflate` call into a
//!   zero-filled array of a fixed size, the returned count ignored. A short stream leaves zeros,
//!   a long one is cut off, and nothing past the array is looked at. A dictionary-flagged stream
//!   makes `inflate` return 0 without an exception, so the array stays all zeros. An array of
//!   length 0 still has its stream parsed up to the first byte it would write (header, block
//!   headers), so a broken stream fails there too.
//! * S21's array is `12288` bytes per section in the primary mask, `+ 256` when ground-up
//!   (`S21PacketChunkData.readPacketData`): room for blocks, metadata, both lights and an add
//!   array per section whether or not they are sent.
//! * S26 inflates all columns into one `196864 * count` array, then slices each column's exact
//!   size, which depends on the packet's sky flag (`S26PacketMapChunkBulk.readPacketData`). The
//!   per-column headers follow the compressed data on the wire.
//!
//! The compressed bytes are kept, so a decoded packet re-encodes to the bytes it came from.
//!
//! [`ColumnView::parse`] then splits an inflated array the way `Chunk.fillChunk` reads it:
//! block ids (low 8 bits) of every primary section, then their metadata nibbles, block light,
//! sky light (only when the **client's dimension** has sky, `!worldObj.provider.hasNoSky`, not
//! the packet's flag), the add nibbles of every section in the add mask, then 256 biome bytes
//! when ground-up.
//!
//! # Limits beyond Java's
//!
//! Java allocates `196864 * count` bytes for S26 before looking at the data (6.4 GB for the
//! largest count), and a bulk whose stream is empty still decodes to zero-filled columns: a frame
//! of a few hundred bytes can demand megabytes. A vanilla 1.7.10 server sends at most five
//! columns per bulk (`S26PacketMapChunkBulk.func_149258_c`, used by
//! `EntityPlayerMP.onUpdate`), a 1.8 server at most ten (java18 `EntityPlayerMP.onUpdate`), which
//! a protocol translator in front of it can pass on. Here a bulk may carry at most
//! [`MAX_BULK_COLUMNS`] (64) columns, so its column data never exceeds [`MAX_BULK_COLUMN_BYTES`]
//! (64 full columns, about 12 MiB, the array Java itself would allocate for 64): six times the
//! largest vanilla bulk, for servers that pack more, and still bounded per packet. The columns are
//! inflated straight into their own buffers (no second copy), and the part of Java's array nobody
//! slices is inflated through a small scratch buffer, so a corrupt stream there is still rejected
//! without being stored.
//!
//! java17: `network/play/server/S21PacketChunkData.java`, `S26PacketMapChunkBulk.java`,
//! `world/chunk/Chunk.java` (`fillChunk`), `world/chunk/NibbleArray.java`

use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status};

use crate::packet::Packet;
use crate::{PacketBuffer, ProtoError};

/// Bytes per section in S21's inflate array: 4096 block ids, then four nibble arrays
/// (metadata, block light, sky light, add) of 2048 each. java17: `12288 * var2`
pub const SECTION_BYTES: usize = 12288;

/// Bytes S26 reserves per column: 16 full sections plus biomes.
/// java17: `S21PacketChunkData.func_149275_c()`
pub const BULK_COLUMN_BYTES: usize = 196_864;

/// Biome bytes in a ground-up column.
pub const BIOME_BYTES: usize = 256;

/// Most columns accepted in one S26 (vanilla 1.7.10 sends at most 5, 1.8 at most 10). See the
/// module docs.
pub const MAX_BULK_COLUMNS: usize = 64;

/// The most column data one S26 can decode to: [`MAX_BULK_COLUMNS`] full columns (every section
/// and every add array present), 12,599,296 bytes. Implied by the column cap, never checked on
/// its own.
pub const MAX_BULK_COLUMN_BYTES: usize = MAX_BULK_COLUMNS * BULK_COLUMN_BYTES;

const NIBBLES: usize = 2048;
const BLOCKS: usize = 4096;

/// java17: `S21PacketChunkData.readPacketData`'s `new byte[12288 * n (+ 256)]`.
#[must_use]
pub fn chunk_data_buffer_len(primary_bit_mask: u16, ground_up_continuous: bool) -> usize {
    SECTION_BYTES * primary_bit_mask.count_ones() as usize
        + if ground_up_continuous { BIOME_BYTES } else { 0 }
}

/// java17: the per-column size `S26PacketMapChunkBulk.readPacketData` slices:
/// `2048 * 4 * n + 256 + 2048 * m`, plus `2048 * n` with sky light.
#[must_use]
pub fn bulk_column_len(primary_bit_mask: u16, add_bit_mask: u16, has_sky_light: bool) -> usize {
    let n = primary_bit_mask.count_ones() as usize;
    let m = add_bit_mask.count_ones() as usize;
    NIBBLES * 4 * n + BIOME_BYTES + NIBBLES * m + if has_sky_light { NIBBLES * n } else { 0 }
}

/// Whether zlib stops at the header with `Z_NEED_DICT`: a valid header with FDICT set.
fn zlib_wants_dictionary(input: &[u8]) -> bool {
    match input {
        [cmf, flg, ..] => {
            (u16::from(*cmf) << 8 | u16::from(*flg)) % 31 == 0
                && cmf & 0x0F == 8
                && (cmf >> 4) <= 7
                && flg & 0x20 != 0
        }
        _ => false,
    }
}

fn bad_compressed_data() -> ProtoError {
    ProtoError::Zlib("Bad compressed data format".to_owned())
}

/// java17: `Inflater.inflate(new byte[output_len])` once over `input`.
///
/// Java's array is handed out as `regions`, zero-filled buffers taken in order from its start
/// (together at most `output_len` bytes; they keep zeros where the stream ran out). Whatever lies
/// past them is inflated through scratch space, so the stream is checked exactly as far as Java's
/// array reaches without being stored.
fn inflate_java(
    input: &[u8],
    output_len: usize,
    regions: &mut [Vec<u8>],
) -> Result<(), ProtoError> {
    debug_assert!(regions.iter().map(Vec::len).sum::<usize>() <= output_len);
    if zlib_wants_dictionary(input) {
        return Ok(());
    }
    let mut inflate = Decompress::new(true);
    if output_len == 0 {
        // zlib still parses the header and block headers up to the first byte it would write.
        inflate
            .decompress(input, &mut [], FlushDecompress::None)
            .map_err(|_| bad_compressed_data())?;
        return Ok(());
    }
    let mut scratch = Vec::new();
    let (mut region, mut filled) = (0usize, 0usize);
    let mut produced = 0usize;
    while produced < output_len {
        while region < regions.len() && filled == regions[region].len() {
            region += 1;
            filled = 0;
        }
        let in_before = inflate.total_in() as usize;
        let out_before = inflate.total_out();
        let target: &mut [u8] = match regions.get_mut(region) {
            Some(buffer) => &mut buffer[filled..],
            None => {
                scratch.resize((output_len - produced).min(64 << 10), 0);
                &mut scratch
            }
        };
        let status = inflate
            .decompress(&input[in_before..], target, FlushDecompress::None)
            .map_err(|_| bad_compressed_data())?;
        let out_n = (inflate.total_out() - out_before) as usize;
        produced += out_n;
        if region < regions.len() {
            filled += out_n;
        }
        if matches!(status, Status::StreamEnd)
            || (out_n == 0 && inflate.total_in() as usize == in_before)
        {
            break;
        }
    }
    Ok(())
}

/// [`inflate_java`] into a single array of `len` bytes (S21).
fn inflate_into_one(input: &[u8], len: usize) -> Result<Vec<u8>, ProtoError> {
    let mut regions = [vec![0u8; len]];
    inflate_java(input, len, &mut regions)?;
    let [data] = regions;
    Ok(data)
}

/// zlib-compress at the JDK's default level (`new Deflater(-1)`), for building packets.
fn deflate_zlib(input: &[u8]) -> Vec<u8> {
    let mut deflate = Compress::new(Compression::new(6), true);
    let mut out = Vec::with_capacity(input.len() / 4 + 64);
    loop {
        out.reserve(input.len() / 4 + 64);
        let in_before = deflate.total_in() as usize;
        match deflate.compress_vec(&input[in_before..], &mut out, FlushCompress::Finish) {
            Ok(Status::StreamEnd) => return out,
            Ok(_) => {}
            // An in-memory deflate with output space does not fail.
            Err(_) => return out,
        }
    }
}

fn read_compressed(buf: &mut PacketBuffer) -> Result<Vec<u8>, ProtoError> {
    let len = buf.read_i32()?;
    if len < 0 {
        // java: Netty's readBytes rejects a negative length.
        return Err(ProtoError::InvalidByteArrayLen(len));
    }
    buf.read_bytes(len as usize)
}

/// `0x21 Chunk Data`. java17: `S21PacketChunkData`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkData {
    /// Chunk column X.
    pub chunk_x: i32,
    /// Chunk column Z.
    pub chunk_z: i32,
    /// A full column (unload and replace); with a zero primary mask it unloads the column.
    pub ground_up_continuous: bool,
    /// Sections present (bit `y` for section `y`).
    pub primary_bit_mask: u16,
    /// Sections whose add (block id bits 8-11) nibbles are present.
    pub add_bit_mask: u16,
    /// The zlib stream exactly as received.
    pub compressed: Vec<u8>,
    /// The inflated array exactly as `readPacketData` leaves it:
    /// [`chunk_data_buffer_len`] bytes, zero past whatever the stream produced.
    pub data: Vec<u8>,
}

impl ChunkData {
    /// Build the packet from uncompressed column bytes (the layout [`ColumnView::parse`]
    /// reads), compressing them as the server does and inflating them back the way
    /// `readPacketData` would, so the result equals a decoded packet.
    #[must_use]
    pub fn from_raw(
        chunk_x: i32,
        chunk_z: i32,
        ground_up_continuous: bool,
        primary_bit_mask: u16,
        add_bit_mask: u16,
        raw: &[u8],
    ) -> ChunkData {
        let compressed = deflate_zlib(raw);
        let len = chunk_data_buffer_len(primary_bit_mask, ground_up_continuous);
        // A stream deflate_zlib just wrote always inflates.
        let data = inflate_into_one(&compressed, len).unwrap_or_else(|_| vec![0; len]);
        ChunkData {
            chunk_x,
            chunk_z,
            ground_up_continuous,
            primary_bit_mask,
            add_bit_mask,
            compressed,
            data,
        }
    }

    /// Split [`Self::data`] the way `Chunk.fillChunk` reads it. `has_sky` is the client
    /// dimension's (`!provider.hasNoSky`), which S21 does not carry.
    ///
    /// # Errors
    ///
    /// As [`ColumnView::parse`].
    pub fn column(&self, has_sky: bool) -> Result<ColumnView<'_>, ProtoError> {
        ColumnView::parse(
            &self.data,
            self.primary_bit_mask,
            self.add_bit_mask,
            has_sky,
            self.ground_up_continuous,
        )
    }
}

impl Packet for ChunkData {
    const ID: i32 = 0x21;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let chunk_x = buf.read_i32()?;
        let chunk_z = buf.read_i32()?;
        let ground_up_continuous = buf.read_bool()?;
        let primary_bit_mask = buf.read_u16()?;
        let add_bit_mask = buf.read_u16()?;
        let compressed = read_compressed(buf)?;
        let len = chunk_data_buffer_len(primary_bit_mask, ground_up_continuous);
        let data = inflate_into_one(&compressed, len)?;
        Ok(ChunkData {
            chunk_x,
            chunk_z,
            ground_up_continuous,
            primary_bit_mask,
            add_bit_mask,
            compressed,
            data,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.chunk_x);
        buf.write_i32(self.chunk_z);
        buf.write_bool(self.ground_up_continuous);
        buf.write_u16(self.primary_bit_mask);
        buf.write_u16(self.add_bit_mask);
        buf.write_i32(self.compressed.len() as i32);
        buf.write_bytes(&self.compressed);
    }
}

/// One column of a [`MapChunkBulk`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkBulkColumn {
    /// Chunk column X.
    pub chunk_x: i32,
    /// Chunk column Z.
    pub chunk_z: i32,
    /// Sections present.
    pub primary_bit_mask: u16,
    /// Sections whose add nibbles are present.
    pub add_bit_mask: u16,
    /// This column's slice of the inflated array ([`bulk_column_len`] bytes, always ground-up).
    pub data: Vec<u8>,
}

impl ChunkBulkColumn {
    /// Split [`Self::data`] the way `Chunk.fillChunk` reads it (always ground-up). `has_sky` is
    /// the client dimension's; when it disagrees with the packet's flag Java misreads the slice
    /// or runs off its end, which [`ColumnView::parse`] reports as an error.
    ///
    /// # Errors
    ///
    /// As [`ColumnView::parse`].
    pub fn column(&self, has_sky: bool) -> Result<ColumnView<'_>, ProtoError> {
        ColumnView::parse(
            &self.data,
            self.primary_bit_mask,
            self.add_bit_mask,
            has_sky,
            true,
        )
    }
}

/// `0x26 Map Chunk Bulk`. java17: `S26PacketMapChunkBulk`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapChunkBulk {
    /// The packet's sky-light flag; it sizes each column's slice.
    pub has_sky_light: bool,
    /// One zlib stream for all columns, exactly as received.
    pub compressed: Vec<u8>,
    /// The columns in wire order.
    pub columns: Vec<ChunkBulkColumn>,
}

/// One column for [`MapChunkBulk::from_raw`]: position, masks and uncompressed bytes.
pub type RawBulkColumn<'a> = (i32, i32, u16, u16, &'a [u8]);

impl MapChunkBulk {
    /// Build the packet from uncompressed column bytes, compressing them as one stream as the
    /// server does and slicing them back the way `readPacketData` would.
    ///
    /// # Errors
    ///
    /// The same limits [`Packet::read`] applies.
    pub fn from_raw(
        has_sky_light: bool,
        columns: &[RawBulkColumn<'_>],
    ) -> Result<MapChunkBulk, ProtoError> {
        let mut raw = Vec::new();
        for (_, _, _, _, data) in columns {
            raw.extend_from_slice(data);
        }
        let compressed = deflate_zlib(&raw);
        let headers: Vec<(i32, i32, u16, u16)> = columns
            .iter()
            .map(|&(x, z, primary, add, _)| (x, z, primary, add))
            .collect();
        Self::assemble(has_sky_light, compressed, &headers)
    }

    fn assemble(
        has_sky_light: bool,
        compressed: Vec<u8>,
        headers: &[(i32, i32, u16, u16)],
    ) -> Result<MapChunkBulk, ProtoError> {
        if headers.len() > MAX_BULK_COLUMNS {
            return Err(ProtoError::PayloadTooLong {
                len: headers.len(),
                max: MAX_BULK_COLUMNS,
            });
        }
        // Each column's slice of Java's array, inflated in place; at most 64 full columns
        // (MAX_BULK_COLUMN_BYTES) by the cap above.
        let mut slices: Vec<Vec<u8>> = headers
            .iter()
            .map(|&(_, _, primary, add)| vec![0; bulk_column_len(primary, add, has_sky_light)])
            .collect();
        inflate_java(&compressed, BULK_COLUMN_BYTES * headers.len(), &mut slices)?;
        let columns = headers
            .iter()
            .zip(slices)
            .map(
                |(&(chunk_x, chunk_z, primary_bit_mask, add_bit_mask), data)| ChunkBulkColumn {
                    chunk_x,
                    chunk_z,
                    primary_bit_mask,
                    add_bit_mask,
                    data,
                },
            )
            .collect();
        Ok(MapChunkBulk {
            has_sky_light,
            compressed,
            columns,
        })
    }
}

impl Packet for MapChunkBulk {
    const ID: i32 = 0x26;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let count = buf.read_i16()?;
        let len = buf.read_i32()?;
        let has_sky_light = buf.read_bool()?;
        if count < 0 {
            // java: `new int[count]` throws NegativeArraySizeException.
            return Err(ProtoError::NegativeLength(i32::from(count)));
        }
        if len < 0 {
            return Err(ProtoError::InvalidByteArrayLen(len));
        }
        let compressed = buf.read_bytes(len as usize)?;
        // The headers follow the data; java reads them after inflating, and only an inflate
        // failure comes first, which reading them early does not change.
        let count = count as usize;
        if count > MAX_BULK_COLUMNS {
            return Err(ProtoError::PayloadTooLong {
                len: count,
                max: MAX_BULK_COLUMNS,
            });
        }
        let mut headers = Vec::with_capacity(count);
        for _ in 0..count {
            let chunk_x = buf.read_i32()?;
            let chunk_z = buf.read_i32()?;
            let primary = buf.read_u16()?;
            let add = buf.read_u16()?;
            headers.push((chunk_x, chunk_z, primary, add));
        }
        MapChunkBulk::assemble(has_sky_light, compressed, &headers)
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i16(self.columns.len() as i16);
        buf.write_i32(self.compressed.len() as i32);
        buf.write_bool(self.has_sky_light);
        buf.write_bytes(&self.compressed);
        for column in &self.columns {
            buf.write_i32(column.chunk_x);
            buf.write_i32(column.chunk_z);
            buf.write_u16(column.primary_bit_mask);
            buf.write_u16(column.add_bit_mask);
        }
    }
}

/// One section of a parsed column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionView<'a> {
    /// Section index (block y / 16).
    pub y: u8,
    /// Block id bits 0-7, index `y << 8 | z << 4 | x` (`ExtendedBlockStorage.blockLSBArray`).
    pub blocks: &'a [u8],
    /// Metadata nibbles, low nibble first (`NibbleArray`).
    pub metadata: &'a [u8],
    /// Block light nibbles.
    pub block_light: &'a [u8],
    /// Sky light nibbles, when the dimension has sky.
    pub sky_light: Option<&'a [u8]>,
    /// Block id bits 8-11 as nibbles, when the add mask names this section.
    pub add: Option<&'a [u8]>,
}

fn nibble(array: &[u8], index: usize) -> u8 {
    let byte = array[index >> 1];
    if index & 1 == 0 {
        byte & 0x0F
    } else {
        byte >> 4
    }
}

impl SectionView<'_> {
    /// The 12-bit block id at `index` (`y << 8 | z << 4 | x`): java17
    /// `ExtendedBlockStorage.getBlockByExtId`, `lsb | msb << 8`.
    #[must_use]
    pub fn block_id(&self, index: usize) -> u16 {
        let msb = self.add.map_or(0, |add| nibble(add, index));
        u16::from(self.blocks[index]) | u16::from(msb) << 8
    }

    /// The metadata nibble at `index`.
    #[must_use]
    pub fn block_metadata(&self, index: usize) -> u8 {
        nibble(self.metadata, index)
    }
}

/// An inflated chunk array split the way `Chunk.fillChunk` reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnView<'a> {
    /// The primary-mask sections, lowest first.
    pub sections: Vec<SectionView<'a>>,
    /// Add nibbles for sections in the add mask but not the primary mask. `fillChunk` copies
    /// these into a section the client already has and steps over them otherwise (with
    /// ground-up, such a section has just been removed). `None` when the region lies past the end
    /// of the array: harmless for a section the client lacks, Java's
    /// `ArrayIndexOutOfBoundsException` for one it has.
    pub orphan_add: Vec<(u8, Option<&'a [u8]>)>,
    /// The 256 biome bytes of a ground-up column.
    pub biomes: Option<&'a [u8]>,
}

impl<'a> ColumnView<'a> {
    /// java17: the read order of `Chunk.fillChunk(data, primary, add, groundUp)`.
    ///
    /// Besides the regions read, `fillChunk` has two effects on sections the packet does not
    /// describe, left to the caller: with ground-up it removes every section outside the primary
    /// mask, and it clears the add array of every primary section outside the add mask (without
    /// ground-up an existing add array stays as it was).
    ///
    /// # Errors
    ///
    /// [`ProtoError::Eof`] if `data` is shorter than a region `fillChunk` copies (its
    /// `ArrayIndexOutOfBoundsException`); an orphan add region it may step over is
    /// [`Self::orphan_add`]'s `None` instead.
    pub fn parse(
        data: &'a [u8],
        primary_bit_mask: u16,
        add_bit_mask: u16,
        has_sky: bool,
        ground_up: bool,
    ) -> Result<ColumnView<'a>, ProtoError> {
        fn take<'d>(
            data: &'d [u8],
            offset: &mut usize,
            len: usize,
        ) -> Result<&'d [u8], ProtoError> {
            let slice = data
                .get(*offset..*offset + len)
                .ok_or(ProtoError::Eof("chunk column data"))?;
            *offset += len;
            Ok(slice)
        }

        let mut offset = 0usize;
        let primary: Vec<u8> = (0..16u8)
            .filter(|y| primary_bit_mask & 1 << y != 0)
            .collect();
        let mut blocks = Vec::with_capacity(primary.len());
        for _ in &primary {
            blocks.push(take(data, &mut offset, BLOCKS)?);
        }
        let mut metadata = Vec::with_capacity(primary.len());
        for _ in &primary {
            metadata.push(take(data, &mut offset, NIBBLES)?);
        }
        let mut block_light = Vec::with_capacity(primary.len());
        for _ in &primary {
            block_light.push(take(data, &mut offset, NIBBLES)?);
        }
        let mut sky_light = Vec::with_capacity(primary.len());
        if has_sky {
            for _ in &primary {
                sky_light.push(take(data, &mut offset, NIBBLES)?);
            }
        }
        // A section in the add mask but not the primary mask may not exist on the client, and
        // then fillChunk steps over its 2048 bytes without reading them, so its region may lie
        // past the end of the array. A primary section's add region is copied: it must be there.
        let mut add = Vec::new();
        let mut orphan_add = Vec::new();
        for y in (0..16u8).filter(|y| add_bit_mask & 1 << y != 0) {
            if primary_bit_mask & 1 << y != 0 {
                add.push((y, take(data, &mut offset, NIBBLES)?));
            } else {
                orphan_add.push((y, data.get(offset..offset + NIBBLES)));
                offset += NIBBLES;
            }
        }
        let biomes = if ground_up {
            Some(take(data, &mut offset, BIOME_BYTES)?)
        } else {
            None
        };

        let sections = primary
            .iter()
            .enumerate()
            .map(|(i, &y)| SectionView {
                y,
                blocks: blocks[i],
                metadata: metadata[i],
                block_light: block_light[i],
                sky_light: sky_light.get(i).copied(),
                add: add.iter().find(|(ay, _)| *ay == y).map(|(_, a)| *a),
            })
            .collect();
        Ok(ColumnView {
            sections,
            orphan_add,
            biomes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uncompressed bytes for one column: every section filled with distinct markers.
    fn raw_column(primary: u16, add: u16, sky: bool, ground_up: bool) -> Vec<u8> {
        let n = primary.count_ones() as usize;
        let mut raw = Vec::new();
        raw.extend(std::iter::repeat_n(1u8, BLOCKS * n));
        raw.extend(std::iter::repeat_n(0x32u8, NIBBLES * n));
        raw.extend(std::iter::repeat_n(0x54u8, NIBBLES * n));
        if sky {
            raw.extend(std::iter::repeat_n(0x76u8, NIBBLES * n));
        }
        raw.extend(std::iter::repeat_n(
            0x21u8,
            NIBBLES * add.count_ones() as usize,
        ));
        if ground_up {
            raw.extend(std::iter::repeat_n(7u8, BIOME_BYTES));
        }
        raw
    }

    fn round_trip<P: Packet + PartialEq + std::fmt::Debug>(packet: &P) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        let mut read = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(&P::read(&mut read).unwrap(), packet);
        assert!(read.is_empty());
        buf.as_slice().to_vec()
    }

    #[test]
    fn chunk_data_inflates_into_javas_fixed_array() {
        let raw = raw_column(0b101, 0b001, true, true);
        let packet = ChunkData::from_raw(3, -4, true, 0b101, 0b001, &raw);
        assert_eq!(packet.data.len(), 2 * SECTION_BYTES + BIOME_BYTES);
        assert_eq!(&packet.data[..raw.len()], &raw[..]);
        assert!(packet.data[raw.len()..].iter().all(|&b| b == 0));
        let bytes = round_trip(&packet);
        // x, z, ground-up, primary, add, then an int length and the zlib bytes.
        assert_eq!(
            &bytes[..13],
            &[0, 0, 0, 3, 0xFF, 0xFF, 0xFF, 0xFC, 1, 0, 5, 0, 1]
        );
        assert_eq!(
            &bytes[13..17],
            &(packet.compressed.len() as i32).to_be_bytes()
        );

        let column = packet.column(true).unwrap();
        assert_eq!(column.sections.len(), 2);
        assert_eq!(column.sections[0].y, 0);
        assert_eq!(column.sections[1].y, 2);
        assert_eq!(column.sections[0].block_id(0), 1 | 1 << 8);
        assert_eq!(
            column.sections[1].block_id(1),
            1,
            "no add nibbles for section 2"
        );
        assert_eq!(column.sections[0].block_metadata(0), 2);
        assert_eq!(column.sections[0].block_metadata(1), 3);
        assert!(column.sections[0].sky_light.is_some());
        assert_eq!(column.biomes.map(<[u8]>::len), Some(256));
        assert!(column.orphan_add.is_empty());
    }

    #[test]
    fn a_short_stream_leaves_zeros_and_bad_data_is_rejected() {
        let raw = raw_column(1, 0, true, false);
        let mut packet = ChunkData::from_raw(0, 0, false, 1, 0, &raw[..100]);
        assert_eq!(packet.data.len(), SECTION_BYTES);
        assert!(packet.data[100..].iter().all(|&b| b == 0));

        // Truncating the zlib stream is not an error either: one inflate call, count ignored.
        packet.compressed.truncate(packet.compressed.len() / 2);
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        assert!(ChunkData::read(&mut PacketBuffer::from_slice(buf.as_slice())).is_ok());

        // A broken header is "Bad compressed data format".
        let bad = ChunkData {
            compressed: vec![0x78, 0x9D, 1, 2, 3],
            ..packet.clone()
        };
        let mut buf = PacketBuffer::new();
        bad.write(&mut buf);
        assert!(matches!(
            ChunkData::read(&mut PacketBuffer::from_slice(buf.as_slice())),
            Err(ProtoError::Zlib(_))
        ));

        // A dictionary-flagged stream makes Inflater.inflate return 0: all zeros, no error.
        let dict = ChunkData {
            compressed: vec![0x78, 0xBB, 0, 0, 0, 1, 0xFF],
            ..packet
        };
        let mut buf = PacketBuffer::new();
        dict.write(&mut buf);
        let read = ChunkData::read(&mut PacketBuffer::from_slice(buf.as_slice())).unwrap();
        assert!(read.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn negative_compressed_length_is_rejected() {
        let mut buf = PacketBuffer::new();
        buf.write_i32(0);
        buf.write_i32(0);
        buf.write_bool(true);
        buf.write_u16(1);
        buf.write_u16(0);
        buf.write_i32(-1);
        assert!(ChunkData::read(&mut PacketBuffer::from_slice(buf.as_slice())).is_err());
    }

    #[test]
    fn bulk_slices_each_column_and_puts_headers_after_the_data() {
        let a = raw_column(0b11, 0, true, true);
        let b = raw_column(0b1000, 0b1000, true, true);
        let packet =
            MapChunkBulk::from_raw(true, &[(1, 2, 0b11, 0, &a), (-1, -2, 0b1000, 0b1000, &b)])
                .unwrap();
        assert_eq!(packet.columns[0].data, a);
        assert_eq!(packet.columns[1].data, b);
        let bytes = round_trip(&packet);
        // count (short), length (int), sky flag, data, then 12-byte headers.
        assert_eq!(&bytes[..2], &[0, 2]);
        let n = bytes.len();
        assert_eq!(
            &bytes[n - 12..],
            &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0, 8, 0, 8]
        );
        let column = packet.columns[1].column(true).unwrap();
        assert_eq!(column.sections[0].y, 3);
        // id bits 0-7 from the block array (1), bits 8-11 from the add nibble (low half of 0x21)
        assert_eq!(column.sections[0].block_id(0), 0x101);
        assert_eq!(column.sections[0].block_id(1), 0x201);

        // The client's dimension without sky reads fewer bytes from the same slice and so finds
        // the add nibbles where sky light was: Java misreads it the same way.
        assert!(packet.columns[1].column(false).is_ok());
    }

    #[test]
    fn bulk_rejects_negative_and_oversized_counts() {
        let mut buf = PacketBuffer::new();
        buf.write_i16(-1);
        buf.write_i32(0);
        buf.write_bool(true);
        assert!(MapChunkBulk::read(&mut PacketBuffer::from_slice(buf.as_slice())).is_err());

        let mut buf = PacketBuffer::new();
        buf.write_i16((MAX_BULK_COLUMNS + 1) as i16);
        buf.write_i32(0);
        buf.write_bool(true);
        assert!(matches!(
            MapChunkBulk::read(&mut PacketBuffer::from_slice(buf.as_slice())),
            Err(ProtoError::PayloadTooLong { .. })
        ));
    }

    /// Columns are inflated straight into their own buffers: a stream spanning several columns
    /// lands in the right slices, and the part of Java's array past the slices stays unstored.
    #[test]
    fn bulk_columns_are_filled_in_place_across_boundaries() {
        // Three one-section columns without sky: 8192 + 256 bytes each.
        let raws: Vec<Vec<u8>> = (0..3u8)
            .map(|i| {
                (0..8448u32)
                    .map(|j| (j as u8).wrapping_mul(i + 1))
                    .collect()
            })
            .collect();
        let columns: Vec<RawBulkColumn<'_>> =
            raws.iter().map(|r| (0, 0, 1u16, 0u16, &r[..])).collect();
        let bulk = MapChunkBulk::from_raw(false, &columns).unwrap();
        for (column, raw) in bulk.columns.iter().zip(&raws) {
            assert_eq!(&column.data, raw);
        }
        // A stream with more data than the slices (Java's array is 196864 per column) still
        // decodes; the surplus only passes through the scratch buffer.
        let mut long = raws.concat();
        long.extend(std::iter::repeat_n(7u8, 100_000));
        let surplus = MapChunkBulk {
            compressed: deflate_zlib(&long),
            ..bulk.clone()
        };
        let mut buf = PacketBuffer::new();
        surplus.write(&mut buf);
        let read = MapChunkBulk::read(&mut PacketBuffer::from_slice(buf.as_slice())).unwrap();
        assert_eq!(read.columns, bulk.columns);
    }

    #[test]
    fn column_parse_follows_fill_chunk_offsets() {
        // Sky light read only with sky; an add bit outside the primary mask is an orphan.
        let raw = raw_column(0b1, 0b10, false, false);
        let column = ColumnView::parse(&raw, 0b1, 0b10, false, false).unwrap();
        assert_eq!(column.sections.len(), 1);
        assert!(column.sections[0].sky_light.is_none());
        assert_eq!(column.orphan_add.len(), 1);
        assert_eq!(column.orphan_add[0].0, 1);
        assert_eq!(column.orphan_add[0].1.map(<[u8]>::len), Some(2048));
        assert!(column.biomes.is_none());
        // fillChunk steps over an orphan region without reading it, so a short one is no error...
        let short = ColumnView::parse(&raw[..raw.len() - 1], 0b1, 0b10, false, false).unwrap();
        assert_eq!(short.orphan_add, vec![(1, None)]);
        // ...but a region it copies is (ArrayIndexOutOfBoundsException).
        let own = raw_column(0b1, 0b1, false, false);
        assert!(ColumnView::parse(&own[..own.len() - 1], 0b1, 0b1, false, false).is_err());
        assert!(ColumnView::parse(&raw[..4096 + 2048], 0b1, 0b10, false, false).is_err());
        // A ground-up read needs the biomes after the orphan region too.
        assert!(ColumnView::parse(&raw, 0b1, 0b10, false, true).is_err());
    }

    /// An S21 with more add bits than primary bits overruns `12288 * n`: java17 still applies it
    /// when the extra sections are absent on the client (`fillChunk` only steps over them).
    #[test]
    fn orphan_add_regions_past_the_array_are_not_an_error() {
        let raw = raw_column(0b1, 0b1110, true, false);
        let packet = ChunkData::from_raw(0, 0, false, 0b1, 0b1110, &raw);
        assert_eq!(packet.data.len(), SECTION_BYTES);
        let column = packet.column(true).unwrap();
        // 10240 bytes of section 0, then three orphan regions: 10240..12288 is inside the array.
        assert_eq!(
            column
                .orphan_add
                .iter()
                .map(|(y, a)| (*y, a.is_some()))
                .collect::<Vec<_>>(),
            vec![(1, true), (2, false), (3, false)]
        );
    }

    /// java17 runs `inflate(new byte[0])` too: the stream is parsed until it would write.
    #[test]
    fn an_empty_array_still_checks_the_stream() {
        let empty = |compressed: Vec<u8>| ChunkData {
            chunk_x: 0,
            chunk_z: 0,
            ground_up_continuous: false,
            primary_bit_mask: 0,
            add_bit_mask: 0,
            compressed,
            data: Vec::new(),
        };
        let decode = |packet: &ChunkData| {
            let mut buf = PacketBuffer::new();
            packet.write(&mut buf);
            ChunkData::read(&mut PacketBuffer::from_slice(buf.as_slice()))
        };
        // A bad header, a valid header with an invalid block type: "Bad compressed data format".
        assert!(decode(&empty(vec![0x78, 0x9D, 1, 2])).is_err());
        assert!(decode(&empty(vec![0x78, 0x9C, 0x07])).is_err());
        // A stream whose first block starts with a literal stops before writing it: no error.
        assert_eq!(
            decode(&empty(deflate_zlib(b"a"))).unwrap(),
            empty(deflate_zlib(b"a"))
        );
        assert!(decode(&empty(Vec::new())).is_ok());
    }

    #[test]
    fn s21_array_size_matches_java() {
        assert_eq!(chunk_data_buffer_len(0xFFFF, true), 196_864);
        assert_eq!(chunk_data_buffer_len(0, false), 0);
        assert_eq!(bulk_column_len(0xFFFF, 0xFFFF, true), BULK_COLUMN_BYTES);
        assert_eq!(bulk_column_len(1, 0, false), 8192 + 256);
    }
}
