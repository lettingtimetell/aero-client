//! The reader: the whole file in memory, the block index from the footer (or rebuilt by scanning
//! the blocks when a crash left no footer), and cursors that decompress one block at a time.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::format::{
    self, BlockHeader, IndexEntry, Unpacker, BLOCK_HEADER_LEN, FILE_HEADER_FIXED, FLAG_KEYFRAME,
    FLAG_PREFIX, TAIL_LEN,
};
use crate::{Record, ReplayError, ReplayMeta, ReplayTime, StreamId};

/// [`Reader::all_records`]: the memory its records may take per byte of the file.
pub const RECORDS_MEMORY_PER_FILE_BYTE: u64 = 128;
/// [`Reader::all_records`]: the memory any file's records may take on top of that.
pub const RECORDS_MEMORY_BASE: u64 = 256 << 20;
/// What one held record costs besides its payload: the [`Record`] and its allocation.
const RECORD_MEMORY_OVERHEAD: u64 = 64;
/// The largest footer (index and final metadata) [`read_info`] reads; a larger one is treated as
/// missing and the blocks are walked instead. Real footers are a few kilobytes.
const MAX_FOOTER_LEN: u64 = 64 << 20;

/// What the index says about one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockInfo {
    /// Where the block's header starts in the file.
    pub offset: u64,
    /// Compressed payload size.
    pub stored_len: u32,
    /// Uncompressed payload size.
    pub raw_len: u32,
    /// Records in the block.
    pub records: u32,
    /// Time of its first record.
    pub first_time: ReplayTime,
    /// Time of its last record.
    pub last_time: ReplayTime,
    /// Its first record is a world reset.
    pub keyframe: bool,
    /// It belongs to a state prefix.
    pub prefix: bool,
    /// Which streams it holds ([`StreamId::mask`] bits).
    pub streams: u16,
}

impl BlockInfo {
    fn from_entry(entry: &IndexEntry) -> BlockInfo {
        BlockInfo {
            offset: entry.offset,
            stored_len: entry.header.stored_len,
            raw_len: entry.header.raw_len,
            records: entry.header.records,
            first_time: entry.header.first_time,
            last_time: entry.header.last_time,
            keyframe: entry.header.flags & FLAG_KEYFRAME != 0,
            prefix: entry.header.flags & FLAG_PREFIX != 0,
            streams: entry.header.streams,
        }
    }
}

/// One record, borrowed from the cursor's decompressed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordRef<'a> {
    /// When it was recorded.
    pub time: ReplayTime,
    /// Its stream.
    pub stream: StreamId,
    /// Its bytes.
    pub payload: &'a [u8],
}

impl RecordRef<'_> {
    /// An owned copy.
    #[must_use]
    pub fn to_record(&self) -> Record {
        Record {
            time: self.time,
            stream: self.stream,
            payload: self.payload.to_vec(),
        }
    }
}

/// The cheap facts about a file (no block is decompressed): for the replay browser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayInfo {
    /// Its metadata (the footer's; the header's with a scanned duration if the file was not
    /// finished).
    pub meta: ReplayMeta,
    /// Its size on disk.
    pub size_bytes: u64,
    /// The file has no footer (the game stopped while recording); it is read up to its last
    /// complete block.
    pub recovered: bool,
}

/// A whole replay file, read and indexed.
#[derive(Debug)]
pub struct Reader {
    bytes: Vec<u8>,
    meta: ReplayMeta,
    entries: Vec<IndexEntry>,
    blocks: Vec<BlockInfo>,
    recovered: bool,
}

impl Reader {
    /// Read and index a file.
    ///
    /// # Errors
    ///
    /// The file cannot be read, is not a replay, is from a newer client, or its header is
    /// damaged. Damaged or missing data *after* the header is not an error: the reader keeps
    /// the blocks before it and reports [`Reader::recovered`].
    pub fn open(path: impl AsRef<Path>) -> Result<Reader, ReplayError> {
        Reader::from_bytes(std::fs::read(path)?)
    }

    /// Index a file already in memory.
    ///
    /// # Errors
    ///
    /// As [`Reader::open`].
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Reader, ReplayError> {
        let (header_meta, data_start) = format::decode_file_header(&bytes)?;
        if let Some((entries, meta)) = footer_from_bytes(&bytes, data_start) {
            let blocks = entries.iter().map(BlockInfo::from_entry).collect();
            return Ok(Reader {
                bytes,
                meta,
                entries,
                blocks,
                recovered: false,
            });
        }
        // No footer, or a damaged one: walk the blocks.
        let (entries, _) = scan_bytes(&bytes, data_start);
        let mut meta = header_meta;
        meta.duration = entries
            .iter()
            .map(|e| e.header.last_time)
            .max()
            .unwrap_or(0)
            .max(meta.duration);
        let blocks = entries.iter().map(BlockInfo::from_entry).collect();
        Ok(Reader {
            bytes,
            meta,
            entries,
            blocks,
            recovered: true,
        })
    }

    /// The metadata.
    #[must_use]
    pub fn meta(&self) -> &ReplayMeta {
        &self.meta
    }

    /// The file's size in bytes (what a memory budget for holding its records is based on).
    #[must_use]
    pub fn file_len(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Whether the file had no (good) footer and was read by scanning its blocks.
    #[must_use]
    pub fn recovered(&self) -> bool {
        self.recovered
    }

    /// The block index, in file (and time) order.
    #[must_use]
    pub fn blocks(&self) -> &[BlockInfo] {
        &self.blocks
    }

    /// The time of the last record.
    #[must_use]
    pub fn duration(&self) -> ReplayTime {
        self.meta.duration
    }

    /// The index of the block to start reading at to see every record at or after `time`: the
    /// first block whose last record is at or after `time` (the number of blocks if none is).
    #[must_use]
    pub fn block_for(&self, time: ReplayTime) -> usize {
        // Blocks are in time order, so `last_time < time` holds for a prefix of them.
        self.blocks.partition_point(|b| b.last_time < time)
    }

    /// The last block flagged as a world reset whose first record is at or before `time`.
    #[must_use]
    pub fn keyframe_before(&self, time: ReplayTime) -> Option<usize> {
        self.blocks
            .iter()
            .rposition(|b| b.keyframe && b.first_time <= time)
    }

    /// A cursor at the first record.
    #[must_use]
    pub fn cursor(&self) -> Cursor<'_> {
        Cursor {
            reader: self,
            next_block: 0,
            raw: Vec::new(),
            pos: 0,
            prev: 0,
            remaining: 0,
            unpacker: Unpacker::default(),
        }
    }

    /// Every record, owned, for a viewer that holds the whole file in
    /// memory. Refused with [`ReplayError::TooLarge`] once holding them would take more than
    /// [`RECORDS_MEMORY_PER_FILE_BYTE`] times the file's size plus [`RECORDS_MEMORY_BASE`]:
    /// recordings take 8 to 14 times their file size (packets compress about tenfold, and every
    /// record has its bookkeeping), while a block of a few kilobytes that decompresses into
    /// millions of empty records would take gigabytes and end the process.
    ///
    /// # Errors
    ///
    /// A damaged block (as [`Cursor::next`]), or the memory budget is spent.
    pub fn all_records(&self) -> Result<Vec<Record>, ReplayError> {
        let budget = (self.bytes.len() as u64)
            .saturating_mul(RECORDS_MEMORY_PER_FILE_BYTE)
            .saturating_add(RECORDS_MEMORY_BASE);
        let mut used = 0u64;
        let mut records = Vec::new();
        let mut cursor = self.cursor();
        while let Some(record) = cursor.next()? {
            used = used.saturating_add(record.payload.len() as u64 + RECORD_MEMORY_OVERHEAD);
            if used > budget {
                return Err(ReplayError::TooLarge(format!(
                    "its records would take more than {} MiB in memory (from a {} KiB file)",
                    budget >> 20,
                    self.bytes.len() >> 10
                )));
            }
            records.push(record.to_record());
        }
        Ok(records)
    }

    /// Every record, owned (simple, allocates per record; use [`Reader::cursor`] on hot paths).
    pub fn records(&self) -> impl Iterator<Item = Result<Record, ReplayError>> + '_ {
        let mut cursor = self.cursor();
        let mut done = false;
        std::iter::from_fn(move || {
            if done {
                return None;
            }
            match cursor.next() {
                Ok(Some(record)) => Some(Ok(record.to_record())),
                Ok(None) => {
                    done = true;
                    None
                }
                Err(error) => {
                    done = true;
                    Some(Err(error))
                }
            }
        })
    }

    /// Block `index`'s records, decompressed and checked, as stored (the record encoding of the
    /// module docs). For tools and measurements; reading records goes through [`Reader::cursor`].
    ///
    /// # Errors
    ///
    /// The block is damaged, or `index` is out of range.
    pub fn block_raw(&self, index: usize) -> Result<Vec<u8>, ReplayError> {
        if index >= self.entries.len() {
            return Err(ReplayError::Corrupt(format!("no block {index}")));
        }
        let (header, stored) = self.block_bytes(index)?;
        let mut raw = Vec::new();
        Unpacker::default().unpack(&header, stored, &mut raw)?;
        Ok(raw)
    }

    /// The block's verified header and its stored payload.
    pub(crate) fn block_bytes(&self, index: usize) -> Result<(BlockHeader, &[u8]), ReplayError> {
        let entry = &self.entries[index];
        let start = usize::try_from(entry.offset)
            .map_err(|_| ReplayError::Corrupt("block offset".into()))?;
        let header = self
            .bytes
            .get(start..)
            .and_then(BlockHeader::decode)
            .ok_or_else(|| ReplayError::Corrupt(format!("block {index} header")))?;
        let expected = BlockHeader {
            payload_crc: header.payload_crc,
            ..entry.header
        };
        if header != expected {
            return Err(ReplayError::Corrupt(format!(
                "block {index} disagrees with the index"
            )));
        }
        let payload_start = start + BLOCK_HEADER_LEN;
        let payload = self
            .bytes
            .get(payload_start..payload_start + header.stored_len as usize)
            .ok_or_else(|| ReplayError::Corrupt(format!("block {index} is cut short")))?;
        Ok((header, payload))
    }
}

/// Reads records in order, one decompressed block at a time, without allocating per record.
///
/// ```no_run
/// # fn demo(reader: &mc_replay::Reader) -> Result<(), mc_replay::ReplayError> {
/// let mut cursor = reader.cursor();
/// while let Some(record) = cursor.next()? {
///     let _ = (record.time, record.stream, record.payload);
/// }
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct Cursor<'r> {
    reader: &'r Reader,
    /// The block the next load reads.
    next_block: usize,
    raw: Vec<u8>,
    pos: usize,
    prev: ReplayTime,
    remaining: u32,
    unpacker: Unpacker,
}

impl Cursor<'_> {
    /// Go to the start of the block holding `time` ([`Reader::block_for`]). Records before
    /// `time` in that block come first; skip them if they were already applied.
    pub fn seek(&mut self, time: ReplayTime) {
        let block = self.reader.block_for(time);
        self.seek_block(block);
    }

    /// Go to the start of block `index`.
    pub fn seek_block(&mut self, index: usize) {
        self.next_block = index.min(self.reader.blocks.len());
        self.remaining = 0;
        self.raw.clear();
        self.pos = 0;
    }

    /// The index of the block the cursor is in (the one the last record came from).
    #[must_use]
    pub fn block(&self) -> usize {
        self.next_block.saturating_sub(1)
    }

    /// The next record, or `None` at the end.
    ///
    /// # Errors
    ///
    /// A block is damaged (a checksum or a record does not add up). The cursor then stays at
    /// the end; the records before it were good.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<RecordRef<'_>>, ReplayError> {
        while self.remaining == 0 {
            if self.next_block >= self.reader.blocks.len() {
                return Ok(None);
            }
            let index = self.next_block;
            self.next_block += 1;
            let (header, stored) = match self.reader.block_bytes(index) {
                Ok(found) => found,
                Err(error) => {
                    self.next_block = self.reader.blocks.len();
                    return Err(error);
                }
            };
            if let Err(error) = self.unpacker.unpack(&header, stored, &mut self.raw) {
                self.next_block = self.reader.blocks.len();
                return Err(error);
            }
            self.pos = 0;
            self.prev = header.first_time;
            self.remaining = header.records;
        }
        let (time, stream, range) = match format::get_record(&self.raw, &mut self.pos, self.prev) {
            Ok(record) => record,
            Err(error) => {
                self.remaining = 0;
                self.next_block = self.reader.blocks.len();
                return Err(error);
            }
        };
        self.prev = time;
        self.remaining -= 1;
        Ok(Some(RecordRef {
            time,
            stream,
            payload: &self.raw[range],
        }))
    }
}

/// The footer of a file in memory, if it has a good one.
fn footer_from_bytes(bytes: &[u8], data_start: usize) -> Option<(Vec<IndexEntry>, ReplayMeta)> {
    let tail = bytes.get(bytes.len().checked_sub(TAIL_LEN)?..)?;
    let (crc, offset) = format::decode_tail(tail)?;
    let offset = usize::try_from(offset).ok()?;
    if offset < data_start {
        return None;
    }
    let body = bytes.get(offset..bytes.len() - TAIL_LEN)?;
    format::decode_footer(body, crc)
}

/// Walk the blocks from `data_start`, keeping every complete, checksummed one; stop at the first
/// that is not. Returns the blocks and where the last good one ends.
fn scan_bytes(bytes: &[u8], data_start: usize) -> (Vec<IndexEntry>, u64) {
    let mut entries = Vec::new();
    let mut pos = data_start;
    let mut last_time = 0;
    while let Some(header) = bytes.get(pos..).and_then(BlockHeader::decode) {
        let payload_start = pos + BLOCK_HEADER_LEN;
        let Some(payload) = bytes.get(payload_start..payload_start + header.stored_len as usize)
        else {
            break;
        };
        if crc32fast::hash(payload) != header.payload_crc || header.first_time < last_time {
            break;
        }
        last_time = header.last_time;
        entries.push(IndexEntry {
            offset: pos as u64,
            header,
        });
        pos = payload_start + header.stored_len as usize;
    }
    (entries, pos as u64)
}

/// The layout of a file on disk, read with seeks (no payload is decompressed).
#[derive(Debug)]
pub(crate) struct Layout {
    pub meta: ReplayMeta,
    pub entries: Vec<IndexEntry>,
    /// Where the last good block ends (where the footer is, or goes).
    pub blocks_end: u64,
    pub recovered: bool,
    pub size: u64,
}

/// Read a file's layout: the footer if it has a good one, otherwise a scan of the block headers
/// (payload checksums are checked too in that case, so the layout ends at the last good block).
pub(crate) fn read_layout(path: &Path) -> Result<Layout, ReplayError> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut fixed = [0u8; FILE_HEADER_FIXED];
    file.read_exact(&mut fixed)
        .map_err(|_| ReplayError::NotAReplay)?;
    let (_, meta_len) = format::decode_file_header_fixed(&fixed)?;
    // A length past the file's end is refused before it is allocated.
    if (FILE_HEADER_FIXED + meta_len + 4) as u64 > size {
        return Err(ReplayError::Corrupt("the header is cut short".into()));
    }
    let mut header = vec![0u8; FILE_HEADER_FIXED + meta_len + 4];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut header)
        .map_err(|_| ReplayError::Corrupt("the header is cut short".into()))?;
    let (header_meta, data_start) = format::decode_file_header(&header)?;
    let data_start = data_start as u64;

    // The footer, if the tail says there is one.
    if size >= data_start + TAIL_LEN as u64 {
        let mut tail = [0u8; TAIL_LEN];
        file.seek(SeekFrom::Start(size - TAIL_LEN as u64))?;
        file.read_exact(&mut tail)?;
        if let Some((crc, offset)) = format::decode_tail(&tail) {
            if offset >= data_start
                && offset < size - TAIL_LEN as u64
                && size - TAIL_LEN as u64 - offset <= MAX_FOOTER_LEN
            {
                let mut body = vec![0u8; (size - TAIL_LEN as u64 - offset) as usize];
                file.seek(SeekFrom::Start(offset))?;
                file.read_exact(&mut body)?;
                if let Some((entries, meta)) = format::decode_footer(&body, crc) {
                    return Ok(Layout {
                        meta,
                        entries,
                        blocks_end: offset,
                        recovered: false,
                        size,
                    });
                }
            }
        }
    }

    // No footer: walk the blocks.
    let mut entries = Vec::new();
    let mut pos = data_start;
    let mut header_bytes = [0u8; BLOCK_HEADER_LEN];
    let mut payload = Vec::new();
    let mut last_time = 0;
    loop {
        if pos + BLOCK_HEADER_LEN as u64 > size {
            break;
        }
        file.seek(SeekFrom::Start(pos))?;
        if file.read_exact(&mut header_bytes).is_err() {
            break;
        }
        let Some(block) = BlockHeader::decode(&header_bytes) else {
            break;
        };
        let end = pos + BLOCK_HEADER_LEN as u64 + u64::from(block.stored_len);
        if end > size || block.first_time < last_time {
            break;
        }
        payload.resize(block.stored_len as usize, 0);
        if file.read_exact(&mut payload).is_err() || crc32fast::hash(&payload) != block.payload_crc
        {
            break;
        }
        last_time = block.last_time;
        entries.push(IndexEntry {
            offset: pos,
            header: block,
        });
        pos = end;
    }
    let mut meta = header_meta;
    meta.duration = meta.duration.max(last_time);
    Ok(Layout {
        meta,
        entries,
        blocks_end: pos,
        recovered: true,
        size,
    })
}

/// The cheap facts about a replay file: its metadata, size and whether it was finished. Reads
/// the header and the footer (or, for a file a crash left unfinished, walks its blocks).
///
/// # Errors
///
/// The file cannot be read, is not a replay or its header is damaged.
pub fn read_info(path: impl AsRef<Path>) -> Result<ReplayInfo, ReplayError> {
    let layout = read_layout(path.as_ref())?;
    Ok(ReplayInfo {
        meta: layout.meta,
        size_bytes: layout.size,
        recovered: layout.recovered,
    })
}

/// Give a file that a crash left without a footer its footer (the header's metadata, the real
/// duration), cutting off any partial block at its end. A finished file is left alone.
///
/// # Errors
///
/// The file cannot be read or written.
pub(crate) fn repair(path: &Path) -> Result<bool, ReplayError> {
    let layout = read_layout(path)?;
    if !layout.recovered {
        return Ok(false);
    }
    let mut meta = layout.meta;
    // The markers (bookmarks) were in the footer the crash lost, but each one was also written
    // as a marker record: take them back from the blocks that hold one.
    let marked = layout
        .entries
        .iter()
        .any(|entry| entry.header.streams & StreamId::Marker.mask() != 0);
    if meta.markers.is_empty() && marked {
        if let Ok(reader) = Reader::open(path) {
            meta.markers = markers_in(&reader);
        }
    }
    crate::writer::rewrite_footer(path, &layout.entries, layout.blocks_end, &meta)?;
    Ok(true)
}

/// Every marker record of a file, in time order. Only the blocks whose stream mask says they
/// hold one are decompressed.
///
/// It runs on its own when the client starts (finishing files a crash
/// left without a footer), so a crafted file must not make it hold or scan without end: at most
/// [`crate::MAX_META_MARKERS`] markers (what the metadata keeps), labels cut to
/// [`crate::MAX_META_TEXT_CHARS`], and at most [`MAX_MARKER_SCAN_RECORDS`] records looked at.
#[must_use]
pub fn markers_in(reader: &Reader) -> Vec<crate::Marker> {
    let mut markers = Vec::new();
    let mut examined = 0u64;
    let mut cursor = reader.cursor();
    'blocks: for (index, block) in reader.blocks().iter().enumerate() {
        if block.streams & StreamId::Marker.mask() == 0 {
            continue;
        }
        cursor.seek_block(index);
        for _ in 0..block.records {
            examined += 1;
            if examined > MAX_MARKER_SCAN_RECORDS {
                break 'blocks;
            }
            match cursor.next() {
                Ok(Some(record)) => {
                    if record.stream == StreamId::Marker {
                        // A character is at most four bytes: never more than the label keeps.
                        let bytes = record.payload.len().min(4 * crate::MAX_META_TEXT_CHARS);
                        let mut label =
                            String::from_utf8_lossy(&record.payload[..bytes]).into_owned();
                        crate::truncate_chars(&mut label, crate::MAX_META_TEXT_CHARS);
                        markers.push(crate::Marker {
                            time: record.time,
                            label,
                        });
                        if markers.len() >= crate::MAX_META_MARKERS {
                            break 'blocks;
                        }
                    }
                }
                Ok(None) | Err(_) => break,
            }
        }
    }
    markers
}

/// [`markers_in`]: the most records it looks at (a recording of many hours holds fewer).
pub const MAX_MARKER_SCAN_RECORDS: u64 = 16_000_000;
