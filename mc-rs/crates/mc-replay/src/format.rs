//! The byte layout of an `.aeroreplay` file: constants, the block header, the index, the header
//! and footer, and the record encoding inside a block. Everything here is pure (no I/O) so the
//! writer, the reader and the clip cutter share one definition.

use crate::{ReplayError, ReplayMeta, ReplayTime, StreamId};

/// First 8 bytes of every replay file.
pub(crate) const FILE_MAGIC: &[u8; 8] = b"AEROREPL";
/// The container version written by this build.
pub(crate) const CONTAINER_VERSION: u16 = 1;
/// Bytes before the header's metadata JSON: magic, version, flags, length.
pub(crate) const FILE_HEADER_FIXED: usize = 16;

/// First 4 bytes of every block.
pub(crate) const BLOCK_MAGIC: &[u8; 4] = b"ARBK";
/// Size of a block header.
pub(crate) const BLOCK_HEADER_LEN: usize = 44;

/// First 4 bytes of the footer.
pub(crate) const INDEX_MAGIC: &[u8; 4] = b"ARIX";
/// Size of one index entry in the footer.
pub(crate) const INDEX_ENTRY_LEN: usize = 40;
/// Last 8 bytes of a finished file.
pub(crate) const END_MAGIC: &[u8; 8] = b"AEROEND1";
/// The fixed tail: footer checksum, footer offset, end magic.
pub(crate) const TAIL_LEN: usize = 20;

/// Block codec: the payload is stored as is.
pub(crate) const CODEC_STORED: u8 = 0;
/// Block codec: the payload is one zstd frame (with its content checksum).
pub(crate) const CODEC_ZSTD: u8 = 1;

/// Block flag: the block's first record is a world reset (`JoinGame`, or a `Respawn` into
/// another dimension). A clip or a seek can start here.
pub(crate) const FLAG_KEYFRAME: u8 = 1;
/// Block flag: the block is part of a state prefix (records that rebuild what a world reset
/// does not clear: the tab list, the scoreboard, the first `JoinGame`).
pub(crate) const FLAG_PREFIX: u8 = 2;

/// The largest uncompressed block a reader accepts. The writer seals blocks at about 1 MiB; one
/// record can be larger (a chunk packet is at most 2 MiB), so this leaves a wide margin while
/// still refusing a damaged length that would ask for gigabytes.
pub(crate) const MAX_BLOCK_RAW: usize = 64 << 20;
/// The largest metadata JSON a reader accepts.
pub(crate) const MAX_META_LEN: usize = 16 << 20;

/// Record tags.
const TAG_PACKET: u8 = 0;
const TAG_CLIENT: u8 = 1;
const TAG_MIC: u8 = 2;
const TAG_MARKER: u8 = 3;
/// Desktop audio. Tag 5 is reserved.
const TAG_DESKTOP: u8 = 4;

/// Append `value` as an unsigned LEB128 varint.
#[inline]
pub(crate) fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Read an unsigned LEB128 varint at `*pos`, advancing it. `None` if the bytes run out or the
/// value does not fit in 64 bits.
#[inline]
pub(crate) fn get_varint(bytes: &[u8], pos: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes.get(*pos)?;
        *pos += 1;
        if shift == 63 && byte > 1 {
            return None;
        }
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

/// Append one record to a block's raw payload. `prev` is the previous record's time in the same
/// block (the block's first time for its first record).
#[inline]
pub(crate) fn put_record(
    out: &mut Vec<u8>,
    prev: ReplayTime,
    time: ReplayTime,
    stream: StreamId,
    payload: &[u8],
) {
    put_varint(out, time - prev);
    match stream {
        StreamId::Packet => out.push(TAG_PACKET),
        StreamId::Client(kind) => {
            out.push(TAG_CLIENT);
            out.push(kind);
        }
        StreamId::Mic => out.push(TAG_MIC),
        StreamId::Marker => out.push(TAG_MARKER),
        StreamId::Desktop => out.push(TAG_DESKTOP),
    }
    put_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// Decode the record at `*pos` in a block's raw payload, advancing `*pos`. `prev` is the
/// previous record's time. Returns `(time, stream, payload range)`.
#[inline]
pub(crate) fn get_record(
    raw: &[u8],
    pos: &mut usize,
    prev: ReplayTime,
) -> Result<(ReplayTime, StreamId, std::ops::Range<usize>), ReplayError> {
    let bad = |what: &str| ReplayError::Corrupt(format!("record {what}"));
    let delta = get_varint(raw, pos).ok_or_else(|| bad("time"))?;
    let time = prev
        .checked_add(delta)
        .filter(|time| *time <= crate::MAX_META_TIME)
        .ok_or_else(|| bad("time past the longest recording"))?;
    let tag = *raw.get(*pos).ok_or_else(|| bad("tag"))?;
    *pos += 1;
    let stream = match tag {
        TAG_PACKET => StreamId::Packet,
        TAG_CLIENT => {
            let kind = *raw.get(*pos).ok_or_else(|| bad("client kind"))?;
            *pos += 1;
            StreamId::Client(kind)
        }
        TAG_MIC => StreamId::Mic,
        TAG_MARKER => StreamId::Marker,
        TAG_DESKTOP => StreamId::Desktop,
        _ => return Err(bad("tag value")),
    };
    let len = get_varint(raw, pos).ok_or_else(|| bad("length"))?;
    let len = usize::try_from(len).map_err(|_| bad("length"))?;
    let start = *pos;
    let end = start.checked_add(len).ok_or_else(|| bad("length"))?;
    if end > raw.len() {
        return Err(bad("payload past the block end"));
    }
    *pos = end;
    Ok((time, stream, start..end))
}

/// What a block header says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockHeader {
    pub stored_len: u32,
    pub raw_len: u32,
    pub records: u32,
    pub first_time: ReplayTime,
    pub last_time: ReplayTime,
    pub codec: u8,
    pub flags: u8,
    pub streams: u16,
    pub payload_crc: u32,
}

impl BlockHeader {
    /// The 44 header bytes, its own checksum included.
    pub(crate) fn encode(&self) -> [u8; BLOCK_HEADER_LEN] {
        let mut out = [0u8; BLOCK_HEADER_LEN];
        out[0..4].copy_from_slice(BLOCK_MAGIC);
        out[4..8].copy_from_slice(&self.stored_len.to_le_bytes());
        out[8..12].copy_from_slice(&self.raw_len.to_le_bytes());
        out[12..16].copy_from_slice(&self.records.to_le_bytes());
        out[16..24].copy_from_slice(&self.first_time.to_le_bytes());
        out[24..32].copy_from_slice(&self.last_time.to_le_bytes());
        out[32] = self.codec;
        out[33] = self.flags;
        out[34..36].copy_from_slice(&self.streams.to_le_bytes());
        out[36..40].copy_from_slice(&self.payload_crc.to_le_bytes());
        let crc = crc32fast::hash(&out[0..40]);
        out[40..44].copy_from_slice(&crc.to_le_bytes());
        out
    }

    /// Parse and check a block header. `None` if the magic or the header checksum is wrong
    /// (a torn or damaged write), or a field is out of range.
    pub(crate) fn decode(bytes: &[u8]) -> Option<BlockHeader> {
        let bytes = bytes.get(..BLOCK_HEADER_LEN)?;
        if &bytes[0..4] != BLOCK_MAGIC {
            return None;
        }
        if crc32fast::hash(&bytes[0..40]) != u32_at(bytes, 40) {
            return None;
        }
        let header = BlockHeader {
            stored_len: u32_at(bytes, 4),
            raw_len: u32_at(bytes, 8),
            records: u32_at(bytes, 12),
            first_time: u64_at(bytes, 16),
            last_time: u64_at(bytes, 24),
            codec: bytes[32],
            flags: bytes[33],
            streams: u16_at(bytes, 34),
            payload_crc: u32_at(bytes, 36),
        };
        // No block of a recording is later than the longest recording.
        let sane = header.raw_len as usize <= MAX_BLOCK_RAW
            && header.stored_len as usize <= MAX_BLOCK_RAW
            && matches!(header.codec, CODEC_STORED | CODEC_ZSTD)
            && header.first_time <= header.last_time
            && header.last_time <= crate::MAX_META_TIME;
        sane.then_some(header)
    }
}

/// One footer index entry: where a block is and what it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexEntry {
    pub offset: u64,
    pub header: BlockHeader,
}

impl IndexEntry {
    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.header.stored_len.to_le_bytes());
        out.extend_from_slice(&self.header.raw_len.to_le_bytes());
        out.extend_from_slice(&self.header.records.to_le_bytes());
        out.extend_from_slice(&self.header.first_time.to_le_bytes());
        out.extend_from_slice(&self.header.last_time.to_le_bytes());
        out.push(self.header.codec);
        out.push(self.header.flags);
        out.extend_from_slice(&self.header.streams.to_le_bytes());
    }

    /// The payload checksum is not in the index (the block header has it); it is filled in when
    /// the block itself is read.
    pub(crate) fn decode(bytes: &[u8]) -> IndexEntry {
        IndexEntry {
            offset: u64_at(bytes, 0),
            header: BlockHeader {
                stored_len: u32_at(bytes, 8),
                raw_len: u32_at(bytes, 12),
                records: u32_at(bytes, 16),
                first_time: u64_at(bytes, 20),
                last_time: u64_at(bytes, 28),
                codec: bytes[36],
                flags: bytes[37],
                streams: u16_at(bytes, 38),
                payload_crc: 0,
            },
        }
    }
}

/// The file header: magic, version, flags and the metadata as known when recording began.
pub(crate) fn encode_file_header(meta: &ReplayMeta) -> Vec<u8> {
    let json = meta.to_json();
    let mut out = Vec::with_capacity(FILE_HEADER_FIXED + json.len() + 4);
    out.extend_from_slice(FILE_MAGIC);
    out.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(&crc32fast::hash(json.as_bytes()).to_le_bytes());
    out
}

/// The fixed part of the file header: `(container version, metadata length)`.
pub(crate) fn decode_file_header_fixed(bytes: &[u8]) -> Result<(u16, usize), ReplayError> {
    let fixed = bytes
        .get(..FILE_HEADER_FIXED)
        .ok_or(ReplayError::NotAReplay)?;
    if &fixed[0..8] != FILE_MAGIC {
        return Err(ReplayError::NotAReplay);
    }
    let version = u16_at(fixed, 8);
    if version > CONTAINER_VERSION || version == 0 {
        return Err(ReplayError::UnsupportedVersion(version));
    }
    let meta_len = u32_at(fixed, 12) as usize;
    if meta_len > MAX_META_LEN {
        return Err(ReplayError::Meta("header metadata is too long".into()));
    }
    Ok((version, meta_len))
}

/// The header's metadata and where the first block starts, from the whole header's bytes.
pub(crate) fn decode_file_header(bytes: &[u8]) -> Result<(ReplayMeta, usize), ReplayError> {
    let (_, meta_len) = decode_file_header_fixed(bytes)?;
    let json_end = FILE_HEADER_FIXED + meta_len;
    let json = bytes
        .get(FILE_HEADER_FIXED..json_end)
        .ok_or_else(|| ReplayError::Corrupt("the header is cut short".into()))?;
    let crc = bytes
        .get(json_end..json_end + 4)
        .ok_or_else(|| ReplayError::Corrupt("the header is cut short".into()))?;
    if crc32fast::hash(json) != u32_at(crc, 0) {
        return Err(ReplayError::Corrupt("header metadata checksum".into()));
    }
    let text = std::str::from_utf8(json).map_err(|e| ReplayError::Meta(e.to_string()))?;
    Ok((ReplayMeta::from_json(text)?, json_end + 4))
}

/// The footer: the index, the final metadata and the tail. `footer_offset` is where it starts
/// (the end of the last block).
pub(crate) fn encode_footer(
    entries: &[IndexEntry],
    meta: &ReplayMeta,
    footer_offset: u64,
) -> Vec<u8> {
    let json = meta.to_json();
    let mut out = Vec::with_capacity(16 + entries.len() * INDEX_ENTRY_LEN + json.len() + TAIL_LEN);
    out.extend_from_slice(INDEX_MAGIC);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        entry.encode_into(&mut out);
    }
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&footer_offset.to_le_bytes());
    out.extend_from_slice(END_MAGIC);
    out
}

/// Parse the tail (the last [`TAIL_LEN`] bytes of a file): `(footer checksum, footer offset)`,
/// or `None` if the file has no finished footer.
pub(crate) fn decode_tail(tail: &[u8]) -> Option<(u32, u64)> {
    if tail.len() != TAIL_LEN || &tail[12..20] != END_MAGIC {
        return None;
    }
    Some((u32_at(tail, 0), u64_at(tail, 4)))
}

/// Parse a footer body (from its magic up to, not including, the tail) checked against `crc`.
pub(crate) fn decode_footer(body: &[u8], crc: u32) -> Option<(Vec<IndexEntry>, ReplayMeta)> {
    if body.len() < 12 || &body[0..4] != INDEX_MAGIC || crc32fast::hash(body) != crc {
        return None;
    }
    let count = u32_at(body, 4) as usize;
    let index_end = 8usize.checked_add(count.checked_mul(INDEX_ENTRY_LEN)?)?;
    let entries = body
        .get(8..index_end)?
        .chunks_exact(INDEX_ENTRY_LEN)
        .map(IndexEntry::decode)
        .collect();
    let meta_len = u32_at(body.get(index_end..index_end + 4)?, 0) as usize;
    let json = body.get(index_end + 4..index_end + 4 + meta_len)?;
    let meta = ReplayMeta::from_json(std::str::from_utf8(json).ok()?).ok()?;
    Some((entries, meta))
}

/// Unpacks block payloads, keeping one zstd decompression context for every block it reads.
#[derive(Default)]
pub(crate) struct Unpacker {
    dctx: Option<zstd::bulk::Decompressor<'static>>,
}

impl std::fmt::Debug for Unpacker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Unpacker")
    }
}

impl Unpacker {
    /// Decompress (or copy) a block payload into `out`, checking its checksum and its declared
    /// size. `out` is cleared first and reused, so a reader that keeps one buffer allocates once.
    /// The zstd frame's own content checksum is verified by the decompressor as well.
    pub(crate) fn unpack(
        &mut self,
        header: &BlockHeader,
        stored: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), ReplayError> {
        if stored.len() != header.stored_len as usize {
            return Err(ReplayError::Corrupt("block payload is cut short".into()));
        }
        if crc32fast::hash(stored) != header.payload_crc {
            return Err(ReplayError::Corrupt("block payload checksum".into()));
        }
        out.clear();
        match header.codec {
            CODEC_STORED => out.extend_from_slice(stored),
            CODEC_ZSTD => {
                let capacity = header.raw_len as usize;
                out.reserve(capacity);
                if self.dctx.is_none() {
                    self.dctx = Some(
                        zstd::bulk::Decompressor::new()
                            .map_err(|e| ReplayError::Corrupt(format!("zstd: {e}")))?,
                    );
                }
                let Some(dctx) = self.dctx.as_mut() else {
                    return Err(ReplayError::Corrupt("zstd context".into()));
                };
                // Writes into the spare capacity and fails rather than growing past it, so a
                // damaged block cannot decompress into more memory than its header declared.
                let written = dctx
                    .decompress_to_buffer(stored, out)
                    .map_err(|e| ReplayError::Corrupt(format!("zstd: {e}")))?;
                if written != capacity {
                    return Err(ReplayError::Corrupt(
                        "block size disagrees with its header".into(),
                    ));
                }
            }
            other => return Err(ReplayError::Corrupt(format!("unknown block codec {other}"))),
        }
        if out.len() != header.raw_len as usize {
            return Err(ReplayError::Corrupt(
                "block size disagrees with its header".into(),
            ));
        }
        Ok(())
    }
}

#[inline]
pub(crate) fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

#[inline]
pub(crate) fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

#[inline]
pub(crate) fn u64_at(bytes: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_round_trip_across_every_length_boundary() {
        let mut values = vec![
            0u64,
            1,
            127,
            128,
            255,
            16_383,
            16_384,
            u64::MAX,
            u64::MAX - 1,
        ];
        for shift in 0..64 {
            values.push(1u64 << shift);
            values.push((1u64 << shift).wrapping_sub(1));
        }
        for value in values {
            let mut out = Vec::new();
            put_varint(&mut out, value);
            let mut pos = 0;
            assert_eq!(get_varint(&out, &mut pos), Some(value), "{value}");
            assert_eq!(pos, out.len());
        }
        // Cut short, and an 11-byte overlong one, are refused.
        assert_eq!(get_varint(&[0x80], &mut 0), None);
        assert_eq!(get_varint(&[0xFF; 11], &mut 0), None);
    }

    #[test]
    fn records_round_trip_with_every_stream() {
        let mut raw = Vec::new();
        let records = [
            (10u64, StreamId::Packet, b"\x01abc".as_slice()),
            (10, StreamId::Client(7), b"".as_slice()),
            (5_000_000, StreamId::Mic, &[0u8; 300][..]),
            (5_000_001, StreamId::Marker, b"kill".as_slice()),
            (5_000_002, StreamId::Desktop, &[7u8; 248][..]),
        ];
        let mut prev = 10;
        for (time, stream, payload) in records {
            put_record(&mut raw, prev, time, stream, payload);
            prev = time;
        }
        let mut pos = 0;
        let mut prev = 10;
        for (time, stream, payload) in records {
            let (t, s, range) = get_record(&raw, &mut pos, prev).expect("record");
            assert_eq!((t, s, &raw[range]), (time, stream, payload));
            prev = t;
        }
        assert_eq!(pos, raw.len());
    }

    #[test]
    fn a_block_header_detects_a_flipped_bit() {
        let header = BlockHeader {
            stored_len: 10,
            raw_len: 20,
            records: 3,
            first_time: 5,
            last_time: 9,
            codec: CODEC_ZSTD,
            flags: FLAG_KEYFRAME,
            streams: 3,
            payload_crc: 0xDEAD_BEEF,
        };
        let bytes = header.encode();
        assert_eq!(BlockHeader::decode(&bytes), Some(header));
        for bit in 0..(40 * 8) {
            let mut damaged = bytes;
            damaged[bit / 8] ^= 1 << (bit % 8);
            assert_eq!(BlockHeader::decode(&damaged), None, "bit {bit}");
        }
    }
}
