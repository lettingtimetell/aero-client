//! A replay's records in memory, packed.
//!
//! The viewer keeps every record of the replay it shows in memory, so that seeks, the exporter's
//! audio and the camera queries can read any of them. As a `Vec<Record>`, each record cost a
//! 40-byte entry plus a heap allocation of its own, and most records are small (an entity moving
//! is a packet of 10 to 20 bytes), so the records took about twice their payload bytes.
//!
//! [`RecordStore`] holds the same records in the same order with no allocation per record: the
//! payloads back to back in one buffer, and per record its time (8 bytes), its stream (2 bytes)
//! and where its payload ends (8 bytes). Reading one gives a [`RecordRef`] that borrows its
//! payload from the store.
//!
//! **Payloads left in the file** ([`RecordStore::read_all_in_file`]): chunk packets are about
//! four fifths of a recording's record bytes and compress about tenfold, so the viewer keeps the
//! file (its compressed blocks) and leaves large packet payloads there; a 12-byte locator takes
//! their place, and [`RecordStore::payload_with`] reads one back through a [`BlockCache`] holding
//! one decompressed block. Playback walks the records in order, so each block is decompressed
//! about once per pass.

use crate::format::{self, Unpacker};
use crate::{
    Reader, Record, RecordRef, ReplayError, ReplayTime, StreamId, RECORDS_MEMORY_BASE,
    RECORDS_MEMORY_PER_FILE_BYTE,
};

/// How far a file's claimed record bytes are trusted when room is reserved up front: at most this
/// many times its stored (compressed) bytes. A recording compresses far less than this; a damaged
/// or crafted file that claims more still loads within the budget (the buffers then grow as
/// needed), it is only not given the room in advance.
const MAX_RESERVE_RATIO: u64 = 64;

/// The fewest bytes a record takes inside a block: a one-byte time delta, the stream tag and a
/// one-byte length (see the crate docs).
const MIN_RECORD_HEADER: u64 = 3;

/// What one held record costs besides its payload: its time, stream and end offset.
const RECORD_INDEX_BYTES: u64 = (std::mem::size_of::<ReplayTime>()
    + std::mem::size_of::<StreamId>()
    + std::mem::size_of::<u64>()) as u64;

/// Marks an `ends` entry whose payload was left in the file. Payload offsets never come near it.
const IN_FILE: u64 = 1 << 63;

/// What an in-file payload leaves in the buffer: its block, its offset in the block's records and
/// its length, each a little-endian `u32`.
const LOCATOR_BYTES: usize = 12;

/// Records in file order, packed. See the module docs.
#[derive(Default)]
pub struct RecordStore {
    times: Vec<ReplayTime>,
    streams: Vec<StreamId>,
    /// Where record `i`'s payload (or locator, with [`IN_FILE`] set) ends in `bytes`. It starts
    /// where record `i - 1`'s ends (at 0 for the first record).
    ends: Vec<u64>,
    bytes: Vec<u8>,
    /// The file that in-file payloads are read back from.
    file: Option<Reader>,
}

/// A summary: a store can hold hundreds of megabytes, and its file's bytes never go to a log.
impl std::fmt::Debug for RecordStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordStore")
            .field("records", &self.len())
            .field("held_bytes", &self.bytes.len())
            .field("file_bytes", &self.file.as_ref().map(Reader::file_len))
            .finish()
    }
}

/// One decompressed block of a [`RecordStore`]'s file, for reading the payloads left there.
#[derive(Default)]
pub struct BlockCache {
    block: Option<usize>,
    raw: Vec<u8>,
    unpacker: Unpacker,
}

impl std::fmt::Debug for BlockCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockCache")
            .field("block", &self.block)
            .field("raw_bytes", &self.raw.len())
            .finish()
    }
}

fn too_large(budget: u64, file_len: u64) -> ReplayError {
    ReplayError::TooLarge(format!(
        "its records would take more than {} MiB in memory (from a {} KiB file)",
        budget >> 20,
        file_len >> 10
    ))
}

/// `Reader::all_records`'s memory budget for `reader`'s records.
fn budget(reader: &Reader) -> u64 {
    reader
        .file_len()
        .saturating_mul(RECORDS_MEMORY_PER_FILE_BYTE)
        .saturating_add(RECORDS_MEMORY_BASE)
}

/// The record count and record bytes `reader`'s block index claims, each clamped to what its
/// stored bytes and `budget` make believable.
fn claimed(reader: &Reader, budget: u64) -> (u64, u64) {
    let blocks = reader.blocks();
    let records: u64 = blocks.iter().map(|block| u64::from(block.records)).sum();
    let raw: u64 = blocks.iter().map(|block| u64::from(block.raw_len)).sum();
    // The index is read before any block is checked, so its stored
    // bytes are only a claim: never more than the file holds.
    let stored = blocks
        .iter()
        .fold(0u64, |sum, block| {
            sum.saturating_add(u64::from(block.stored_len))
        })
        .min(reader.file_len());
    let trusted = stored.saturating_mul(MAX_RESERVE_RATIO).min(budget);
    let records = records.min(trusted / (MIN_RECORD_HEADER + RECORD_INDEX_BYTES));
    (records, raw.min(trusted))
}

impl RecordStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> RecordStore {
        RecordStore::default()
    }

    /// An empty store with room for `records` records holding `payload_bytes` bytes in all.
    #[must_use]
    pub fn with_capacity(records: usize, payload_bytes: usize) -> RecordStore {
        RecordStore {
            times: Vec::with_capacity(records),
            streams: Vec::with_capacity(records),
            ends: Vec::with_capacity(records),
            bytes: Vec::with_capacity(payload_bytes),
            file: None,
        }
    }

    /// [`Self::with_capacity`] for room claimed by a file: a
    /// reservation the allocator refuses is skipped (the buffers then grow as records arrive)
    /// instead of aborting the process.
    fn with_claimed_capacity(records: usize, payload_bytes: usize) -> RecordStore {
        let mut store = RecordStore::new();
        if store.times.try_reserve_exact(records).is_err()
            || store.streams.try_reserve_exact(records).is_err()
            || store.ends.try_reserve_exact(records).is_err()
            || store.bytes.try_reserve_exact(payload_bytes).is_err()
        {
            return RecordStore::new();
        }
        store
    }

    /// Every record of `reader`, in file order: what [`Reader::records`] yields, without an
    /// allocation per record. Room is reserved once from the block index, so the buffers do not
    /// grow (and briefly double) while the file is read.
    ///
    /// Holds to the same memory budget as [`Reader::all_records`]: refused once the records
    /// would take more than [`RECORDS_MEMORY_PER_FILE_BYTE`] times the file's size plus
    /// [`RECORDS_MEMORY_BASE`] (counting each record's payload and its index bytes here).
    ///
    /// # Errors
    ///
    /// A damaged block, as [`Reader::records`] reports it, or [`ReplayError::TooLarge`] once the
    /// budget is spent.
    pub fn read_all(reader: &Reader) -> Result<RecordStore, ReplayError> {
        let budget = budget(reader);
        let (records, raw) = claimed(reader, budget);
        let payload = raw.saturating_sub(records.saturating_mul(MIN_RECORD_HEADER));
        let mut store = RecordStore::with_claimed_capacity(
            usize::try_from(records).unwrap_or(0),
            usize::try_from(payload).unwrap_or(0),
        );
        let mut used = 0u64;
        let mut cursor = reader.cursor();
        while let Some(record) = cursor.next()? {
            used = used.saturating_add(record.payload.len() as u64 + RECORD_INDEX_BYTES);
            if used > budget {
                return Err(too_large(budget, reader.file_len()));
            }
            store.push(record.time, record.stream, record.payload);
        }
        Ok(store)
    }

    /// [`Self::read_all`], but the payload of every packet record of at least `min_len` bytes
    /// stays in the file: the store keeps `reader` (the file's compressed bytes) and a 12-byte
    /// locator per such record, and [`Self::payload_with`] reads the payload back. Every other
    /// record is held as by [`Self::read_all`], under the same budget (a locator counts as its 12
    /// bytes).
    ///
    /// # Errors
    ///
    /// As [`Self::read_all`].
    pub fn read_all_in_file(reader: Reader, min_len: usize) -> Result<RecordStore, ReplayError> {
        let budget = budget(&reader);
        let (records, _) = claimed(&reader, budget);
        let mut store =
            RecordStore::with_claimed_capacity(usize::try_from(records).unwrap_or(0), 0);
        // The held bytes are not known before the blocks are walked: grow, then keep an exact copy.
        let mut held: Vec<u8> = Vec::new();
        let mut used = 0u64;
        let mut unpacker = Unpacker::default();
        let mut raw = Vec::new();
        let corrupt = |what: &str| ReplayError::Corrupt(format!("in-file record {what}"));
        for index in 0..reader.blocks().len() {
            let (header, stored) = reader.block_bytes(index)?;
            unpacker.unpack(&header, stored, &mut raw)?;
            let block = u32::try_from(index).map_err(|_| corrupt("block"))?;
            let mut pos = 0usize;
            let mut prev = header.first_time;
            for _ in 0..header.records {
                let (time, stream, range) = format::get_record(&raw, &mut pos, prev)?;
                prev = time;
                let in_file = stream == StreamId::Packet && range.len() >= min_len;
                let held_len = if in_file { LOCATOR_BYTES } else { range.len() };
                used = used.saturating_add(held_len as u64 + RECORD_INDEX_BYTES);
                if used > budget {
                    return Err(too_large(budget, reader.file_len()));
                }
                store.times.push(time);
                store.streams.push(stream);
                if in_file {
                    let offset = u32::try_from(range.start).map_err(|_| corrupt("offset"))?;
                    let len = u32::try_from(range.len()).map_err(|_| corrupt("length"))?;
                    held.extend_from_slice(&block.to_le_bytes());
                    held.extend_from_slice(&offset.to_le_bytes());
                    held.extend_from_slice(&len.to_le_bytes());
                    store.ends.push(held.len() as u64 | IN_FILE);
                } else {
                    held.extend_from_slice(&raw[range]);
                    store.ends.push(held.len() as u64);
                }
            }
        }
        store.bytes = if held.capacity() > held.len() + held.len() / 8 {
            held.as_slice().to_vec()
        } else {
            held
        };
        store.file = Some(reader);
        Ok(store)
    }

    /// The records of `records`, in their order. Each payload is copied in and freed as the
    /// vector is consumed.
    #[must_use]
    pub fn from_records(records: Vec<Record>) -> RecordStore {
        let payload_bytes = records.iter().map(|record| record.payload.len()).sum();
        let mut store = RecordStore::with_capacity(records.len(), payload_bytes);
        for record in records {
            store.push(record.time, record.stream, &record.payload);
        }
        store
    }

    /// Append a record (held in memory).
    pub fn push(&mut self, time: ReplayTime, stream: StreamId, payload: &[u8]) {
        self.bytes.extend_from_slice(payload);
        self.times.push(time);
        self.streams.push(stream);
        self.ends.push(self.bytes.len() as u64);
    }

    /// How many records it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether it holds no record.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Where record `index`'s held bytes are, and whether they are an in-file locator.
    #[inline]
    fn span(&self, index: usize) -> (usize, usize, bool) {
        let start = match index.checked_sub(1) {
            Some(previous) => (self.ends[previous] & !IN_FILE) as usize,
            None => 0,
        };
        let end = self.ends[index];
        (start, (end & !IN_FILE) as usize, end & IN_FILE != 0)
    }

    /// Record `index`, or `None` past the end. The payload of a record left in the file
    /// ([`Self::in_file`]) is empty here: read it with [`Self::payload_with`].
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<RecordRef<'_>> {
        let time = *self.times.get(index)?;
        let (start, end, in_file) = self.span(index);
        Some(RecordRef {
            time,
            stream: self.streams[index],
            payload: if in_file {
                &[]
            } else {
                &self.bytes[start..end]
            },
        })
    }

    /// Whether record `index`'s payload was left in the file.
    #[must_use]
    pub fn in_file(&self, index: usize) -> bool {
        self.ends.get(index).is_some_and(|end| end & IN_FILE != 0)
    }

    /// Record `index`'s payload, held or read back from the file through `cache`.
    ///
    /// # Errors
    ///
    /// `index` is past the end, or the payload's block cannot be read back.
    pub fn payload_with<'a>(
        &'a self,
        index: usize,
        cache: &'a mut BlockCache,
    ) -> Result<&'a [u8], ReplayError> {
        if index >= self.len() {
            return Err(ReplayError::Corrupt(format!("no record {index}")));
        }
        let (start, end, in_file) = self.span(index);
        if !in_file {
            return Ok(&self.bytes[start..end]);
        }
        let corrupt = |what: &str| ReplayError::Corrupt(format!("in-file record {what}"));
        let locator = self
            .bytes
            .get(start..end)
            .ok_or_else(|| corrupt("locator"))?;
        let field = |at: usize| {
            locator
                .get(at..at + 4)
                .and_then(|bytes| bytes.try_into().ok())
                .map(|bytes: [u8; 4]| u32::from_le_bytes(bytes) as usize)
                .ok_or_else(|| corrupt("locator"))
        };
        let (block, offset, len) = (field(0)?, field(4)?, field(8)?);
        let file = self.file.as_ref().ok_or_else(|| corrupt("file"))?;
        if cache.block != Some(block) {
            cache.block = None;
            let (header, stored) = file.block_bytes(block)?;
            cache.unpacker.unpack(&header, stored, &mut cache.raw)?;
            cache.block = Some(block);
        }
        cache
            .raw
            .get(offset..offset + len)
            .ok_or_else(|| corrupt("past its block"))
    }

    /// The last record (see [`Self::get`] for one left in the file).
    #[must_use]
    pub fn last(&self) -> Option<RecordRef<'_>> {
        self.len().checked_sub(1).and_then(|index| self.get(index))
    }

    /// Every record's time, in order.
    #[must_use]
    pub fn times(&self) -> &[ReplayTime] {
        &self.times
    }

    /// The index of the first record at or after `time` (the length if there is none), for
    /// records in time order (as a recording's are).
    #[must_use]
    pub fn first_at_or_after(&self, time: ReplayTime) -> usize {
        self.times.partition_point(|&at| at < time)
    }

    /// The records from index `start` on, in order (see [`Self::get`] for those left in the
    /// file).
    pub fn iter_from(&self, start: usize) -> impl Iterator<Item = RecordRef<'_>> + '_ {
        (start..self.len()).filter_map(move |index| self.get(index))
    }

    /// Every record, in order (see [`Self::get`] for those left in the file).
    pub fn iter(&self) -> impl Iterator<Item = RecordRef<'_>> + '_ {
        self.iter_from(0)
    }

    /// The bytes it holds for payloads (and in-file locators).
    #[must_use]
    pub fn payload_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// The heap bytes its buffers have reserved, the kept file included, for memory reports.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.times.capacity() * std::mem::size_of::<ReplayTime>()
            + self.streams.capacity() * std::mem::size_of::<StreamId>()
            + self.ends.capacity() * std::mem::size_of::<u64>()
            + self.bytes.capacity()
            + self.file.as_ref().map_or(0, |file| {
                usize::try_from(file.file_len()).unwrap_or(usize::MAX)
            })
    }

    /// An owned copy of every record, in-file payloads read back (tests and tools).
    ///
    /// # Errors
    ///
    /// As [`Self::payload_with`].
    pub fn to_records(&self) -> Result<Vec<Record>, ReplayError> {
        let mut cache = BlockCache::default();
        let mut out = Vec::with_capacity(self.len());
        for index in 0..self.len() {
            let payload = self.payload_with(index, &mut cache)?.to_vec();
            out.push(Record {
                time: self.times[index],
                stream: self.streams[index],
                payload,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{ReplayMeta, Writer, WriterOptions};

    /// A deterministic mix of streams, sizes (empty ones and large packets included) and times.
    fn sample(count: usize) -> Vec<Record> {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut time = 0u64;
        (0..count)
            .map(|i| {
                if i % 4 == 0 {
                    time += next() % 50_000;
                }
                let stream = match next() % 8 {
                    0..=4 => StreamId::Packet,
                    5 => StreamId::Client((next() % 6) as u8),
                    6 => StreamId::Mic,
                    _ => StreamId::Marker,
                };
                let len = match next() % 16 {
                    0 => 0,
                    1 => 3000 + (next() % 9000) as usize,
                    _ => (next() % 40) as usize,
                };
                Record {
                    time,
                    stream,
                    payload: (0..len).map(|j| (j as u64 ^ next()) as u8).collect(),
                }
            })
            .collect()
    }

    /// `records` written to a file in a temporary folder (small blocks: many of them), read back.
    fn written(records: &[Record]) -> (tempfile::TempDir, Reader) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.aeroreplay");
        let meta = ReplayMeta {
            title: "store".to_owned(),
            ..ReplayMeta::default()
        };
        let options = WriterOptions {
            level: 3,
            block_bytes: 8192,
            block_interval: Duration::from_secs(3600),
            sync_interval: None,
        };
        let mut writer = Writer::create(&path, &meta, options).expect("create");
        for record in records {
            writer.push(record.time, record.stream, &record.payload);
        }
        writer
            .finish(meta)
            .wait(Duration::from_secs(30))
            .expect("finish");
        let reader = Reader::open(&path).expect("open");
        (dir, reader)
    }

    #[test]
    fn every_record_comes_back_in_order() {
        let input = sample(2000);
        let store = RecordStore::from_records(input.clone());
        assert_eq!(store.len(), input.len());
        assert!(!store.is_empty());
        for (index, record) in input.iter().enumerate() {
            let got = store.get(index).expect("in range");
            assert_eq!(got.time, record.time);
            assert_eq!(got.stream, record.stream);
            assert_eq!(got.payload, record.payload.as_slice());
            assert!(!store.in_file(index));
        }
        assert!(store.get(input.len()).is_none());
        assert_eq!(store.last().map(|r| r.to_record()), input.last().cloned());
        assert_eq!(store.to_records().expect("records"), input);
        assert_eq!(
            store.payload_bytes(),
            input.iter().map(|r| r.payload.len()).sum::<usize>()
        );
        // The iterators agree with indexing, from any start.
        assert_eq!(store.iter().count(), input.len());
        let from = store
            .iter_from(1500)
            .map(|r| r.to_record())
            .collect::<Vec<_>>();
        assert_eq!(from, input[1500..].to_vec());
        assert_eq!(store.iter_from(input.len()).count(), 0);
        assert_eq!(store.iter_from(input.len() + 7).count(), 0);
    }

    #[test]
    fn first_at_or_after_matches_a_partition_point_over_the_records() {
        let input = sample(600);
        let store = RecordStore::from_records(input.clone());
        let last = input.last().map_or(0, |r| r.time);
        for time in (0..=last + 10).step_by(997).chain([0, 1, last, last + 1]) {
            assert_eq!(
                store.first_at_or_after(time),
                input.partition_point(|record| record.time < time),
                "time {time}"
            );
        }
        assert_eq!(store.times().len(), input.len());
    }

    #[test]
    fn an_empty_store_has_nothing() {
        let store = RecordStore::from_records(Vec::new());
        assert!(store.is_empty());
        assert!(store.get(0).is_none());
        assert!(store.last().is_none());
        assert_eq!(store.first_at_or_after(5), 0);
        assert_eq!(store.iter().count(), 0);
        assert!(store.payload_with(0, &mut BlockCache::default()).is_err());
        assert_eq!(RecordStore::new().len(), 0);
    }

    #[test]
    fn read_all_holds_what_the_reader_yields_and_reserves_once() {
        let input = sample(5000);
        let (_dir, reader) = written(&input);
        assert!(
            reader.blocks().len() > 10,
            "{} blocks",
            reader.blocks().len()
        );
        let expected = reader
            .records()
            .collect::<Result<Vec<_>, _>>()
            .expect("records");
        assert_eq!(expected, input);
        let store = RecordStore::read_all(&reader).expect("read");
        assert_eq!(store.to_records().expect("records"), expected);
        // Reserved from the block index: no buffer grew past it, and the payload buffer's slack
        // is only the record headers the estimate could not see.
        assert_eq!(store.times.capacity(), input.len());
        assert_eq!(store.ends.capacity(), input.len());
        let payload = store.payload_bytes();
        assert!(store.bytes.capacity() >= payload);
        assert!(
            store.bytes.capacity() - payload <= input.len() * 4,
            "slack {} for {} records",
            store.bytes.capacity() - payload,
            input.len()
        );
    }

    #[test]
    fn payloads_left_in_the_file_read_back_exactly_in_any_order() {
        let input = sample(5000);
        let (_dir, reader) = written(&input);
        let blocks = reader.blocks().len();
        assert!(blocks > 10, "{blocks} blocks");
        let min_len = 256;
        let store = RecordStore::read_all_in_file(reader, min_len).expect("read");
        assert_eq!(store.len(), input.len());
        let mut cache = BlockCache::default();
        let mut left = 0usize;
        for (index, record) in input.iter().enumerate() {
            let in_file = record.stream == StreamId::Packet && record.payload.len() >= min_len;
            assert_eq!(store.in_file(index), in_file, "record {index}");
            left += usize::from(in_file);
            let got = store.get(index).expect("in range");
            assert_eq!((got.time, got.stream), (record.time, record.stream));
            if in_file {
                assert!(got.payload.is_empty(), "an in-file payload is not held");
            } else {
                assert_eq!(got.payload, record.payload.as_slice());
            }
            assert_eq!(
                store.payload_with(index, &mut cache).expect("payload"),
                record.payload.as_slice(),
                "record {index}"
            );
        }
        assert!(left > 100, "{left} records left in the file");
        // Backwards, so the cache changes block on almost every in-file record.
        for (index, record) in input.iter().enumerate().rev() {
            assert_eq!(
                store.payload_with(index, &mut cache).expect("payload"),
                record.payload.as_slice()
            );
        }
        assert_eq!(store.to_records().expect("records"), input);
        // Held: everything but the large packets, plus a locator for each of those, exactly.
        let held: usize = input
            .iter()
            .map(|record| {
                if record.stream == StreamId::Packet && record.payload.len() >= min_len {
                    LOCATOR_BYTES
                } else {
                    record.payload.len()
                }
            })
            .sum();
        assert_eq!(store.payload_bytes(), held);
        assert!(store.bytes.capacity() <= held + held / 8);
        assert!(store.payload_with(input.len(), &mut cache).is_err());
        assert!(
            format!("{store:?}").len() < 200,
            "the debug form is a summary"
        );
    }
}
