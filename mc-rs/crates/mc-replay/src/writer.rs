//! The writer.
//!
//! **Game thread:** [`Writer::push`] appends a record to the open block (a varint, a tag and a
//! `memcpy`); nothing is allocated once the buffers are warm. A block is *sealed* when it holds
//! about [`WriterOptions::block_bytes`] or has been open for [`WriterOptions::block_interval`]
//! ([`Writer::tick`], once per frame), and handed to the worker over a channel.
//!
//! **Worker thread:** compresses each sealed block with zstd (one reused context), writes it
//! after its 44-byte header, and syncs the file to the disk at most every
//! [`WriterOptions::sync_interval`], so a crash or a power cut loses at most the last few
//! seconds and leaves a file the reader recovers. [`Writer::finish`] writes the footer (the
//! block index and the final metadata). Emptied block buffers go back to the game thread
//! through a return channel.
//!
//! **Memory writer** ([`Writer::memory`]): the same, but the compressed blocks stay in memory,
//! from the last world reset ([`Writer::keyframe`]) on, bounded in size. This is the session
//! buffer behind clips ([`Writer::save_clip`]) and behind a recording started in the middle of a
//! world ([`Writer::create_with_prelude`]).

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::format::{
    self, BlockHeader, IndexEntry, CODEC_STORED, CODEC_ZSTD, FLAG_KEYFRAME, FLAG_PREFIX,
};
use crate::{Record, ReplayError, ReplayKind, ReplayMeta, ReplayTime, StreamId};

/// The most a record adds besides its payload: a 10-byte time varint, 2 tag bytes and a 10-byte
/// length varint.
const RECORD_OVERHEAD: usize = 22;

/// How a writer cuts, compresses and syncs blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriterOptions {
    /// The zstd level. 1 is fastest, 19 smallest; the default (9) is measured in the lane-RA
    /// report: within a few percent of level 19's size at a small fraction of its time.
    pub level: i32,
    /// Seal a block once its uncompressed records reach this many bytes.
    pub block_bytes: usize,
    /// ...or once it has been open this long (checked by [`Writer::tick`]).
    pub block_interval: Duration,
    /// Sync the file to the disk at most this often (`None`: never; the OS writes it out in its
    /// own time). A sync makes the blocks written so far survive a power cut, not only a crash.
    pub sync_interval: Option<Duration>,
}

impl Default for WriterOptions {
    fn default() -> WriterOptions {
        WriterOptions {
            level: 9,
            block_bytes: 1 << 20,
            block_interval: Duration::from_secs(2),
            sync_interval: Some(Duration::from_secs(2)),
        }
    }
}

/// Counters a writer keeps (read with [`Writer::stats`]; cheap atomics).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WriterStats {
    /// Records pushed so far.
    pub records: u64,
    /// Blocks the worker has finished.
    pub blocks: u64,
    /// Uncompressed record bytes the worker has finished.
    pub raw_bytes: u64,
    /// Bytes on disk (file writer) or held in memory (memory writer).
    pub stored_bytes: u64,
    /// A memory writer went over its size cap; it holds nothing until the next world reset.
    pub overflowed: bool,
    /// A memory writer holds a world from its reset on (a clip or a mid-world start can be cut
    /// from it). False until the first world reset, and after an overflow.
    pub ready: bool,
    /// The worker hit an error (the message is in [`Writer::error`]); records are dropped.
    pub failed: bool,
}

/// The block being filled on the game thread.
#[derive(Debug)]
struct OpenBlock {
    raw: Vec<u8>,
    records: u32,
    first_time: ReplayTime,
    prev_time: ReplayTime,
    streams: u16,
    flags: u8,
    opened_at: Option<Instant>,
}

impl OpenBlock {
    fn new(capacity: usize) -> OpenBlock {
        OpenBlock {
            raw: Vec::with_capacity(capacity),
            records: 0,
            first_time: 0,
            prev_time: 0,
            streams: 0,
            flags: 0,
            opened_at: None,
        }
    }
}

/// A sealed block on its way to the worker.
#[derive(Debug)]
struct Sealed {
    raw: Vec<u8>,
    records: u32,
    first_time: ReplayTime,
    last_time: ReplayTime,
    streams: u16,
    flags: u8,
}

/// A compressed block held by a memory writer.
#[derive(Debug, Clone)]
pub(crate) struct StoredBlock {
    pub header: BlockHeader,
    pub payload: Arc<[u8]>,
}

/// What a recording started in the middle of a world begins with: the state prefix, then the
/// session buffer's blocks (from the last world reset), retimed so `base` becomes 0.
#[derive(Debug)]
struct Prelude {
    prefix: Vec<Record>,
    blocks: mpsc::Receiver<Vec<StoredBlock>>,
}

/// A clip request for a memory writer's worker.
#[derive(Debug)]
struct ClipJob {
    path: PathBuf,
    meta: ReplayMeta,
    prefix: Vec<Record>,
    window: (ReplayTime, ReplayTime),
    /// Records kept outside the buffer (the microphone's ring), merged in by time.
    extra: Vec<Record>,
}

#[derive(Debug)]
enum Job {
    Block(Sealed),
    Prelude(Prelude),
    Snapshot(mpsc::Sender<Vec<StoredBlock>>),
    Clip(ClipJob, mpsc::Sender<Result<Finished, String>>),
    Clear,
    Finish(Option<ReplayMeta>, mpsc::Sender<Result<Finished, String>>),
    /// Tests only: run this on the worker thread (to make the worker panic).
    #[cfg(test)]
    Run(TestWork),
}

/// Work a test runs on the worker thread.
#[cfg(test)]
struct TestWork(Box<dyn FnOnce() + Send>);

#[cfg(test)]
impl std::fmt::Debug for TestWork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TestWork")
    }
}

/// What a finished file turned out to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finished {
    /// Where it is.
    pub path: PathBuf,
    /// Its size on disk.
    pub bytes: u64,
    /// Its final metadata.
    pub meta: ReplayMeta,
}

/// The counters shared with the worker.
#[derive(Debug, Default)]
struct Shared {
    blocks: AtomicU64,
    raw_bytes: AtomicU64,
    stored_bytes: AtomicU64,
    overflowed: AtomicBool,
    ready: AtomicBool,
    failed: AtomicBool,
    error: Mutex<Option<String>>,
}

impl Shared {
    fn fail(&self, message: String) {
        tracing::warn!(%message, "replay writer failed; the rest of this recording is dropped");
        self.failed.store(true, Ordering::Relaxed);
        let mut slot = self
            .error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(message);
        }
    }
}

/// A finish (or a clip save) in progress on a worker thread.
#[derive(Debug)]
pub struct Finishing {
    rx: mpsc::Receiver<Result<Finished, String>>,
    thread: Option<JoinHandle<()>>,
    result: Option<Result<Finished, String>>,
}

impl Finishing {
    /// Whether the worker is done (successfully or not). Never blocks.
    pub fn is_done(&mut self) -> bool {
        self.poll();
        self.result.is_some()
    }

    /// The outcome, if the worker is done. Never blocks.
    pub fn try_result(&mut self) -> Option<&Result<Finished, String>> {
        self.poll();
        self.result.as_ref()
    }

    fn poll(&mut self) {
        if self.result.is_none() {
            match self.rx.try_recv() {
                Ok(result) => self.result = Some(result),
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.result = Some(Err("the replay writer stopped".into()));
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    /// Wait up to `timeout` for the worker.
    ///
    /// # Errors
    ///
    /// The worker's error message, or a timeout. After a timeout the worker keeps going on its
    /// own; the file is finished when it is done.
    pub fn wait(mut self, timeout: Duration) -> Result<Finished, String> {
        if let Some(result) = self.result.take() {
            return result;
        }
        let result = match self.rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(format!(
                    "the replay writer did not finish within {timeout:?}"
                ));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err("the replay writer stopped".into()),
        };
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        result
    }
}

/// Writes one replay (or keeps the session buffer). See the module documentation.
#[derive(Debug)]
pub struct Writer {
    block: OpenBlock,
    /// Flags for the next block to open ([`FLAG_KEYFRAME`] after [`Writer::keyframe`]).
    next_flags: u8,
    prefix_mode: bool,
    tx: mpsc::Sender<Job>,
    pool: mpsc::Receiver<Vec<u8>>,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    options: WriterOptions,
    last_time: ReplayTime,
    records: u64,
    finished: bool,
    path: Option<PathBuf>,
}

impl Writer {
    /// Start a worker that creates `path` (and its directory; the file must not exist) and
    /// writes the replay into it. `meta` goes into the header as the metadata known now;
    /// [`Writer::finish`] writes the final one.
    ///
    /// Never blocks: the file is created on the worker thread (a file create on a slow disk can
    /// take tens of milliseconds). If it cannot be created, the writer fails
    /// ([`WriterStats::failed`], [`Writer::error`]) and records pushed to it are dropped.
    ///
    /// # Errors
    ///
    /// None today; the `Result` leaves room for checks that can be made without I/O.
    pub fn create(
        path: impl Into<PathBuf>,
        meta: &ReplayMeta,
        options: WriterOptions,
    ) -> Result<Writer, ReplayError> {
        let sink = Sink::File(Box::new(FileSink {
            file: None,
            path: path.into(),
            offset: 0,
            entries: Vec::new(),
            meta: meta.clone(),
            last_sync: Instant::now(),
            last_time: 0,
        }));
        Ok(Writer::spawn(sink, options))
    }

    /// The path a file writer writes (`None` for a memory writer).
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Like [`Writer::create`], but the file begins with `prefix` (a state prefix block at time
    /// 0) and then a memory writer's blocks from its last world reset, retimed so that reset is
    /// at 0 (`snapshot`, from [`Writer::snapshot`] on that memory writer). Records pushed here
    /// afterwards must use the same clock (the memory writer's time minus the first block's
    /// first time; [`Snapshot`] tells it).
    ///
    /// # Errors
    ///
    /// As [`Writer::create`].
    pub fn create_with_prelude(
        path: impl Into<PathBuf>,
        meta: &ReplayMeta,
        options: WriterOptions,
        prefix: Vec<Record>,
        snapshot: Snapshot,
    ) -> Result<Writer, ReplayError> {
        let writer = Writer::create(path, meta, options)?;
        let _ = writer.tx.send(Job::Prelude(Prelude {
            prefix,
            blocks: snapshot.rx,
        }));
        Ok(writer)
    }

    /// A writer that keeps its compressed blocks in memory, from the last world reset
    /// ([`Writer::keyframe`]) on, up to `max_bytes` of compressed data. Blocks before its first
    /// world reset are not kept (they cannot rebuild a world). Past the cap it holds nothing
    /// until the next world reset ([`WriterStats::overflowed`]).
    #[must_use]
    pub fn memory(options: WriterOptions, max_bytes: u64) -> Writer {
        let options = WriterOptions {
            sync_interval: None,
            ..options
        };
        Writer::spawn(
            Sink::Memory(MemorySink {
                blocks: Vec::new(),
                bytes: 0,
                max_bytes,
                armed: false,
            }),
            options,
        )
    }

    fn spawn(sink: Sink, options: WriterOptions) -> Writer {
        let path = match &sink {
            Sink::File(file) => Some(file.path.clone()),
            Sink::Memory(_) => None,
        };
        let (tx, rx) = mpsc::channel::<Job>();
        let (pool_tx, pool_rx) = mpsc::channel::<Vec<u8>>();
        let shared = Arc::new(Shared::default());
        let worker_shared = Arc::clone(&shared);
        // A panic in the worker is this recording's failure ("Recording
        // failed"), not the game's crash; without this the recorder never learnt of it.
        let thread = std::thread::Builder::new()
            .name("mc-rs replay writer".into())
            .spawn(move || {
                crate::contained::run_helper(
                    "replay writer",
                    || worker(sink, &rx, &pool_tx, &worker_shared, options),
                    |message| worker_shared.fail(message),
                );
            });
        let thread = match thread {
            Ok(thread) => Some(thread),
            Err(error) => {
                shared.fail(format!("could not start the writer thread: {error}"));
                None
            }
        };
        Writer {
            block: OpenBlock::new(options.block_bytes + 4096),
            next_flags: 0,
            prefix_mode: false,
            tx,
            pool: pool_rx,
            shared,
            thread,
            options,
            last_time: 0,
            records: 0,
            finished: false,
            path,
        }
    }

    /// Append a record. Times must not go backwards; an earlier time is raised to the last one
    /// (records keep their push order either way).
    #[inline]
    pub fn push(&mut self, time: ReplayTime, stream: StreamId, payload: &[u8]) {
        let time = time.max(self.last_time);
        // Seal first rather than grow the buffer: recycled buffers keep their capacity, so a
        // block only reallocates for a record bigger than a whole buffer.
        let needed = payload.len() + RECORD_OVERHEAD;
        if self.block.records > 0 && self.block.raw.len() + needed > self.block.raw.capacity() {
            self.seal();
        }
        let block = &mut self.block;
        if block.records == 0 {
            block.first_time = time;
            block.prev_time = time;
            block.flags = std::mem::take(&mut self.next_flags)
                | if self.prefix_mode { FLAG_PREFIX } else { 0 };
        }
        format::put_record(&mut block.raw, block.prev_time, time, stream, payload);
        block.prev_time = time;
        block.records += 1;
        block.streams |= stream.mask();
        self.last_time = time;
        self.records += 1;
        if block.raw.len() >= self.options.block_bytes {
            self.seal();
        }
    }

    /// The next record is a world reset: it starts a new block flagged as a keyframe. A memory
    /// writer drops everything before it once that block arrives.
    pub fn keyframe(&mut self) {
        self.seal();
        self.next_flags |= FLAG_KEYFRAME;
    }

    /// While on, blocks are flagged as part of the state prefix.
    pub fn set_prefix_mode(&mut self, on: bool) {
        if self.prefix_mode != on {
            self.seal();
            self.prefix_mode = on;
        }
    }

    /// Once per frame: seals the open block once it has been open for
    /// [`WriterOptions::block_interval`]. One comparison on most frames.
    #[inline]
    pub fn tick(&mut self, now: Instant) {
        if self.block.records == 0 {
            return;
        }
        match self.block.opened_at {
            None => self.block.opened_at = Some(now),
            Some(at) if now.saturating_duration_since(at) >= self.options.block_interval => {
                self.seal();
            }
            Some(_) => {}
        }
    }

    /// Seal the open block now (it reaches the file or the memory store shortly after).
    pub fn flush(&mut self) {
        self.seal();
    }

    fn seal(&mut self) {
        if self.block.records == 0 {
            return;
        }
        let fresh = self
            .pool
            .try_recv()
            .unwrap_or_else(|_| Vec::with_capacity(self.options.block_bytes + 4096));
        let block = &mut self.block;
        let sealed = Sealed {
            raw: std::mem::replace(&mut block.raw, fresh),
            records: block.records,
            first_time: block.first_time,
            last_time: block.prev_time,
            streams: block.streams,
            flags: block.flags,
        };
        block.records = 0;
        block.streams = 0;
        block.flags = 0;
        block.opened_at = None;
        // A dead worker has already recorded why in `shared`; the block is dropped.
        let _ = self.tx.send(Job::Block(sealed));
    }

    /// The time of the last record pushed.
    #[must_use]
    pub fn last_time(&self) -> ReplayTime {
        self.last_time
    }

    /// Counters (atomics; cheap).
    #[must_use]
    pub fn stats(&self) -> WriterStats {
        WriterStats {
            records: self.records,
            blocks: self.shared.blocks.load(Ordering::Relaxed),
            raw_bytes: self.shared.raw_bytes.load(Ordering::Relaxed),
            stored_bytes: self.shared.stored_bytes.load(Ordering::Relaxed),
            overflowed: self.shared.overflowed.load(Ordering::Relaxed),
            ready: self.shared.ready.load(Ordering::Relaxed),
            failed: self.shared.failed.load(Ordering::Relaxed),
        }
    }

    /// Whether the worker hit an error (one atomic load; [`Writer::error`] has the message).
    #[must_use]
    pub fn failed(&self) -> bool {
        self.shared.failed.load(Ordering::Relaxed)
    }

    /// The worker's error, if it hit one.
    #[must_use]
    pub fn error(&self) -> Option<String> {
        self.shared
            .error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Memory writer: everything it holds (from the last world reset), once the worker has
    /// stored every block pushed before this call. Never blocks; the snapshot arrives through
    /// the returned handle (for [`Writer::create_with_prelude`]).
    pub fn snapshot(&mut self) -> Snapshot {
        self.seal();
        let (tx, rx) = mpsc::channel();
        let _ = self.tx.send(Job::Snapshot(tx));
        Snapshot { rx }
    }

    /// Memory writer: write a clip to `path`: `prefix` (a state prefix block at time 0), then
    /// every block held (from the last world reset), retimed so that reset is at 0. `window` is
    /// the part to show, in this writer's clock; it is clamped to what is held and stored in the
    /// clip's metadata (`kind`, `duration` and `view_start` are filled in). Never blocks.
    ///
    /// `extra` are records kept outside this writer, in its clock (the client's microphone
    /// ring): those inside the held world are merged into the clip in time order (a block they
    /// fall in is recompressed; the others are copied as they are).
    pub fn save_clip(
        &mut self,
        path: impl Into<PathBuf>,
        meta: ReplayMeta,
        prefix: Vec<Record>,
        window: (ReplayTime, ReplayTime),
        extra: Vec<Record>,
    ) -> Finishing {
        self.seal();
        let (tx, rx) = mpsc::channel();
        let job = ClipJob {
            path: path.into(),
            meta,
            prefix,
            window,
            extra,
        };
        if let Err(mpsc::SendError(Job::Clip(_, reply))) = self.tx.send(Job::Clip(job, tx)) {
            let _ = reply.send(Err("the replay buffer is not running".into()));
        }
        Finishing {
            rx,
            thread: None,
            result: None,
        }
    }

    /// Memory writer: drop everything held (a disconnect).
    pub fn clear(&mut self) {
        self.block.raw.clear();
        self.block.records = 0;
        self.block.streams = 0;
        self.block.opened_at = None;
        self.next_flags = 0;
        let _ = self.tx.send(Job::Clear);
    }

    /// Seal the last block and have the worker write the footer with `meta` as the final
    /// metadata (its duration is raised to the last record's time if it is shorter). Never
    /// blocks: wait on the returned handle if the file must be complete before going on (quit).
    pub fn finish(mut self, meta: ReplayMeta) -> Finishing {
        self.finish_inner(Some(meta))
    }

    fn finish_inner(&mut self, meta: Option<ReplayMeta>) -> Finishing {
        self.seal();
        self.finished = true;
        let (tx, rx) = mpsc::channel();
        if let Err(mpsc::SendError(Job::Finish(_, reply))) = self.tx.send(Job::Finish(meta, tx)) {
            let _ = reply.send(Err("the replay writer is not running".into()));
        }
        Finishing {
            rx,
            thread: self.thread.take(),
            result: None,
        }
    }
}

impl Drop for Writer {
    /// A writer dropped without [`Writer::finish`] (a panic unwinding, a test) still gets its
    /// footer, with the header's metadata and the real duration. The worker does it on its own
    /// time; nothing here waits.
    fn drop(&mut self) {
        if !self.finished {
            drop(self.finish_inner(None));
        }
    }
}

/// A memory writer's blocks on their way (see [`Writer::snapshot`]).
#[derive(Debug)]
pub struct Snapshot {
    rx: mpsc::Receiver<Vec<StoredBlock>>,
}

// ---- the worker -------------------------------------------------------------------------------

#[derive(Debug)]
enum Sink {
    // Boxed: the file sink carries the header's metadata and is much the larger.
    File(Box<FileSink>),
    Memory(MemorySink),
}

#[derive(Debug)]
struct FileSink {
    /// Opened (and the header written) by the worker's first step.
    file: Option<File>,
    path: PathBuf,
    offset: u64,
    entries: Vec<IndexEntry>,
    meta: ReplayMeta,
    last_sync: Instant,
    last_time: ReplayTime,
}

impl FileSink {
    /// Create the file (it must not exist) and its directory, and write the header.
    fn open(&mut self) -> Result<(), ReplayError> {
        if self.file.is_some() {
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)?;
        let header = format::encode_file_header(&self.meta);
        file.write_all(&header)?;
        self.offset = header.len() as u64;
        self.file = Some(file);
        Ok(())
    }

    fn file(&mut self) -> Result<&mut File, ReplayError> {
        self.file
            .as_mut()
            .ok_or_else(|| ReplayError::Io(std::io::Error::other("the replay file is not open")))
    }
}

#[derive(Debug)]
struct MemorySink {
    blocks: Vec<StoredBlock>,
    bytes: u64,
    max_bytes: u64,
    /// Holding a world: set by a keyframe block, cleared by an overflow.
    armed: bool,
}

/// zstd plus its reused output buffer.
pub(crate) struct Packer {
    cctx: Option<zstd::bulk::Compressor<'static>>,
    out: Vec<u8>,
}

impl std::fmt::Debug for Packer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Packer")
    }
}

/// The most hash-table bits a writer's zstd context gets (2^18 entries).
///
/// zstd sizes its match-finder tables for the level, not for the input: at level 9 that is 2^21
/// entries, a 10.5 MiB context that a recording holds from start to finish (two of them with
/// the session buffer on). Blocks are at most about 1 MiB, so 2^18 still covers them: on a real
/// 165 s minigame recording (113 blocks, 66.7 MiB of records) it measured a 1.7 MiB context,
/// +0.02 % file size and no slower. Levels below [`HASH_LOG_CAP_FROM_LEVEL`] already use fewer
/// bits and are left alone. Files stay plain zstd frames; any reader decodes them.
const MAX_HASH_LOG: u32 = 18;

/// The lowest level whose own hash table is larger than [`MAX_HASH_LOG`] for a 1 MiB block.
const HASH_LOG_CAP_FROM_LEVEL: i32 = 5;

/// `MC_W21_MW_REPLAY_HASHLOG=0` keeps zstd's own table size (read once).
fn hash_log_cap_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MC_W21_MW_REPLAY_HASHLOG").map_or(true, |v| v != "0"))
}

impl Packer {
    pub(crate) fn new(level: i32) -> Packer {
        let cctx = zstd::bulk::Compressor::new(level)
            .ok()
            .and_then(|mut cctx| {
                // The frame's own checksum, verified on every decompression.
                cctx.include_checksum(true).ok()?;
                if level >= HASH_LOG_CAP_FROM_LEVEL && hash_log_cap_enabled() {
                    // A context that refuses the setting keeps zstd's own size.
                    let _ = cctx.set_parameter(zstd::zstd_safe::CParameter::HashLog(MAX_HASH_LOG));
                }
                Some(cctx)
            });
        Packer {
            cctx,
            out: Vec::new(),
        }
    }

    /// Compress `raw` into `self.out`; `(codec, payload)`. Falls back to storing it if zstd is
    /// unavailable or would not make it smaller.
    pub(crate) fn pack<'a>(&'a mut self, raw: &'a [u8]) -> (u8, &'a [u8]) {
        if let Some(cctx) = self.cctx.as_mut() {
            self.out.clear();
            self.out.reserve(zstd::zstd_safe::compress_bound(raw.len()));
            if cctx.compress_to_buffer(raw, &mut self.out).is_ok() && self.out.len() < raw.len() {
                return (CODEC_ZSTD, &self.out);
            }
        }
        (CODEC_STORED, raw)
    }
}

fn worker(
    mut sink: Sink,
    rx: &mpsc::Receiver<Job>,
    pool: &mpsc::Sender<Vec<u8>>,
    shared: &Shared,
    options: WriterOptions,
) {
    let mut packer = Packer::new(options.level);
    // A file writer creates its file first, here rather than on the caller's (game) thread.
    if let Sink::File(file) = &mut sink {
        if let Err(error) = file.open() {
            // The file name, not its folder (the account name).
            shared.fail(format!(
                "{}: {error}",
                crate::library::file_name(&file.path)
            ));
        }
    }
    while let Ok(job) = rx.recv() {
        match job {
            Job::Block(sealed) => {
                if !shared.failed.load(Ordering::Relaxed) {
                    let stored = store_block(&mut sink, &mut packer, &sealed, shared, options);
                    if let Err(error) = stored {
                        shared.fail(error.to_string());
                    }
                }
                let mut raw = sealed.raw;
                raw.clear();
                let _ = pool.send(raw);
            }
            Job::Prelude(prelude) => {
                if let Err(error) = write_prelude(&mut sink, &mut packer, prelude, shared) {
                    shared.fail(error.to_string());
                }
            }
            Job::Snapshot(reply) => {
                let blocks = match &sink {
                    Sink::Memory(memory) => memory.blocks.clone(),
                    Sink::File(_) => Vec::new(),
                };
                let _ = reply.send(blocks);
            }
            Job::Clip(clip, reply) => {
                let result = match &sink {
                    Sink::Memory(memory) => write_clip(&memory.blocks, &mut packer, clip),
                    Sink::File(_) => Err("clips are cut from the session buffer".into()),
                };
                let _ = reply.send(result);
            }
            Job::Clear => {
                if let Sink::Memory(memory) = &mut sink {
                    memory.blocks.clear();
                    memory.bytes = 0;
                    memory.armed = false;
                    shared.stored_bytes.store(0, Ordering::Relaxed);
                    shared.overflowed.store(false, Ordering::Relaxed);
                    shared.ready.store(false, Ordering::Relaxed);
                }
            }
            Job::Finish(meta, reply) => {
                let result = match &mut sink {
                    Sink::File(file) => finish_file(file, meta, shared).map_err(|e| e.to_string()),
                    Sink::Memory(_) => Err("a memory writer has no file".into()),
                };
                if let Err(error) = &result {
                    if matches!(sink, Sink::File(_)) {
                        shared.fail(error.clone());
                    }
                }
                let _ = reply.send(result);
                return;
            }
            #[cfg(test)]
            Job::Run(TestWork(work)) => work(),
        }
    }
    // Every handle is gone without a finish job (cannot happen through `Drop`, which sends
    // one); leave what was written for the reader to recover.
}

fn store_block(
    sink: &mut Sink,
    packer: &mut Packer,
    sealed: &Sealed,
    shared: &Shared,
    options: WriterOptions,
) -> Result<(), ReplayError> {
    let (codec, payload) = packer.pack(&sealed.raw);
    let header = BlockHeader {
        stored_len: payload.len() as u32,
        raw_len: sealed.raw.len() as u32,
        records: sealed.records,
        first_time: sealed.first_time,
        last_time: sealed.last_time,
        codec,
        flags: sealed.flags,
        streams: sealed.streams,
        payload_crc: crc32fast::hash(payload),
    };
    shared.blocks.fetch_add(1, Ordering::Relaxed);
    shared
        .raw_bytes
        .fetch_add(sealed.raw.len() as u64, Ordering::Relaxed);
    match sink {
        Sink::File(file) => {
            append_block(file, &header, payload)?;
            shared.stored_bytes.store(file.offset, Ordering::Relaxed);
            if let Some(interval) = options.sync_interval {
                if file.last_sync.elapsed() >= interval {
                    file.file()?.sync_data()?;
                    file.last_sync = Instant::now();
                }
            }
        }
        Sink::Memory(memory) => {
            if header.flags & FLAG_KEYFRAME != 0 {
                // A world reset: what came before cannot rebuild the new world.
                memory.blocks.clear();
                memory.bytes = 0;
                memory.armed = true;
                shared.overflowed.store(false, Ordering::Relaxed);
            }
            if memory.armed {
                memory.bytes += (format::BLOCK_HEADER_LEN + payload.len()) as u64;
                memory.blocks.push(StoredBlock {
                    header,
                    payload: Arc::from(payload),
                });
                if memory.bytes > memory.max_bytes {
                    tracing::info!(
                        bytes = memory.bytes,
                        "replay session buffer full; clips resume at the next world"
                    );
                    memory.blocks.clear();
                    memory.bytes = 0;
                    memory.armed = false;
                    shared.overflowed.store(true, Ordering::Relaxed);
                }
            }
            shared.ready.store(memory.armed, Ordering::Relaxed);
            shared.stored_bytes.store(memory.bytes, Ordering::Relaxed);
        }
    }
    Ok(())
}

/// Write one block (header + payload) at the end of a file sink.
fn append_block(
    file: &mut FileSink,
    header: &BlockHeader,
    payload: &[u8],
) -> Result<(), ReplayError> {
    let bytes = header.encode();
    let handle = file.file()?;
    handle.write_all(&bytes)?;
    handle.write_all(payload)?;
    file.entries.push(IndexEntry {
        offset: file.offset,
        header: *header,
    });
    file.offset += (bytes.len() + payload.len()) as u64;
    file.last_time = file.last_time.max(header.last_time);
    Ok(())
}

/// Pack `records` (all at time 0) into one prefix block.
fn prefix_block(packer: &mut Packer, records: &[Record]) -> Option<(BlockHeader, Vec<u8>)> {
    if records.is_empty() {
        return None;
    }
    let mut raw = Vec::new();
    let mut streams = 0u16;
    for record in records {
        format::put_record(&mut raw, 0, 0, record.stream, &record.payload);
        streams |= record.stream.mask();
    }
    let (codec, payload) = packer.pack(&raw);
    let header = BlockHeader {
        stored_len: payload.len() as u32,
        raw_len: raw.len() as u32,
        records: records.len() as u32,
        first_time: 0,
        last_time: 0,
        codec,
        flags: FLAG_PREFIX,
        streams,
        payload_crc: crc32fast::hash(payload),
    };
    Some((header, payload.to_vec()))
}

/// The retimed header of a stored block (`base` becomes 0). The payload's record deltas are
/// relative to the block's first time, so only the header changes.
fn retimed(header: &BlockHeader, base: ReplayTime) -> BlockHeader {
    BlockHeader {
        first_time: header.first_time.saturating_sub(base),
        last_time: header.last_time.saturating_sub(base),
        ..*header
    }
}

fn write_prelude(
    sink: &mut Sink,
    packer: &mut Packer,
    prelude: Prelude,
    shared: &Shared,
) -> Result<(), ReplayError> {
    let Sink::File(file) = sink else {
        return Ok(());
    };
    if let Some((header, payload)) = prefix_block(packer, &prelude.prefix) {
        append_block(file, &header, &payload)?;
    }
    // The memory writer answers as soon as it has stored what was pushed before the request.
    let blocks = prelude
        .blocks
        .recv_timeout(Duration::from_secs(30))
        .unwrap_or_default();
    if let Some(base) = blocks.first().map(|b| b.header.first_time) {
        for block in &blocks {
            append_block(file, &retimed(&block.header, base), &block.payload)?;
        }
    }
    shared.stored_bytes.store(file.offset, Ordering::Relaxed);
    Ok(())
}

fn write_clip(
    blocks: &[StoredBlock],
    packer: &mut Packer,
    clip: ClipJob,
) -> Result<Finished, String> {
    let Some(first) = blocks.first() else {
        return Err("the session buffer is empty (it starts at the next world)".into());
    };
    let base = first.header.first_time;
    let last = blocks
        .iter()
        .map(|b| b.header.last_time)
        .max()
        .unwrap_or(base);
    // The window ends where it was asked to (the key press), even when the last record is a
    // little older: "the last 30 s" is 30 s. It cannot start before the world does.
    let end = clip.window.1.max(base);
    let start = clip.window.0.clamp(base, end);
    let mut meta = clip.meta;
    meta.kind = ReplayKind::Clip {
        start: start - base,
        end: end - base,
    };
    meta.view_start = start - base;
    meta.duration = last.max(end) - base;
    for time in &mut meta.world_resets {
        *time = time.saturating_sub(base);
    }
    meta.world_resets.dedup();
    for marker in &mut meta.markers {
        marker.time = marker.time.saturating_sub(base);
    }
    // Extra records from before the world began cannot be shown; the rest are merged in.
    let mut extra = clip.extra;
    extra.retain(|record| record.time >= base);
    extra.sort_by_key(|record| record.time);
    write_assembled(
        &clip.path,
        &meta,
        packer,
        &clip.prefix,
        blocks.iter().map(|b| (b.header, &b.payload[..])),
        base,
        &extra,
    )
    .map_err(|error| error.to_string())
}

/// Write a new file: the header (with `meta`), `prefix` as one state-prefix block at time 0,
/// then `blocks` copied as they are (still compressed) and retimed so `base` is 0, then the
/// footer. The clip cutters' common back end.
///
/// `extra` (sorted by time, in the blocks' clock) is merged in by time: records falling before
/// a block go into a block of their own ahead of it; a block they fall inside is recompressed
/// with them, its own records first at equal times (so a world-reset block still starts with
/// the reset); what is left after the last block goes into a final block. Blocks no extra
/// record touches are copied untouched.
pub(crate) fn write_assembled<'a>(
    path: &Path,
    meta: &ReplayMeta,
    packer: &mut Packer,
    prefix: &[Record],
    blocks: impl IntoIterator<Item = (BlockHeader, &'a [u8])>,
    base: ReplayTime,
    extra: &[Record],
) -> Result<Finished, ReplayError> {
    let mut sink = FileSink {
        file: None,
        path: path.to_path_buf(),
        offset: 0,
        entries: Vec::new(),
        meta: meta.clone(),
        last_sync: Instant::now(),
        last_time: 0,
    };
    sink.open()?;
    if let Some((header, payload)) = prefix_block(packer, prefix) {
        append_block(&mut sink, &header, &payload)?;
    }
    let mut extra = extra;
    let mut unpacker = format::Unpacker::default();
    let mut raw = Vec::new();
    for (header, payload) in blocks {
        // Extra records before this block: a block of their own.
        let before = extra.partition_point(|r| r.time < header.first_time);
        if before > 0 {
            let records = extra[..before]
                .iter()
                .map(|r| (r.time, r.stream, r.payload.as_slice()));
            write_records(&mut sink, packer, records, base, 0)?;
            extra = &extra[before..];
        }
        // Extra records inside it: recompress it with them.
        let inside = extra.partition_point(|r| r.time <= header.last_time);
        if inside == 0 {
            append_block(&mut sink, &retimed(&header, base), payload)?;
            continue;
        }
        unpacker.unpack(&header, payload, &mut raw)?;
        let mut own = Vec::with_capacity(header.records as usize);
        let (mut pos, mut prev) = (0, header.first_time);
        for _ in 0..header.records {
            let (time, stream, range) = format::get_record(&raw, &mut pos, prev)?;
            own.push((time, stream, range));
            prev = time;
        }
        let mut merged = Vec::with_capacity(own.len() + inside);
        let (mut a, mut b) = (0, 0);
        while a < own.len() || b < inside {
            let take_own = b == inside || (a < own.len() && own[a].0 <= extra[b].time);
            if take_own {
                let (time, stream, range) = &own[a];
                merged.push((*time, *stream, &raw[range.clone()]));
                a += 1;
            } else {
                let record = &extra[b];
                merged.push((record.time, record.stream, record.payload.as_slice()));
                b += 1;
            }
        }
        write_records(&mut sink, packer, merged.into_iter(), base, header.flags)?;
        extra = &extra[inside..];
    }
    if !extra.is_empty() {
        let records = extra
            .iter()
            .map(|r| (r.time, r.stream, r.payload.as_slice()));
        write_records(&mut sink, packer, records, base, 0)?;
    }
    finish_file(&mut sink, Some(meta.clone()), &Shared::default())
}

/// Encode `records` (absolute times in the source clock, in order) retimed so `base` is 0, into
/// blocks of about a megabyte; the first block gets `flags`.
fn write_records<'a>(
    sink: &mut FileSink,
    packer: &mut Packer,
    records: impl Iterator<Item = (ReplayTime, StreamId, &'a [u8])>,
    base: ReplayTime,
    flags: u8,
) -> Result<(), ReplayError> {
    const BLOCK_BYTES: usize = 1 << 20;
    let mut raw = Vec::new();
    let (mut count, mut first, mut prev, mut streams) = (0u32, 0, 0, 0u16);
    let mut flags = flags;
    let mut flush = |raw: &mut Vec<u8>,
                     count: &mut u32,
                     first: ReplayTime,
                     last: ReplayTime,
                     streams: &mut u16,
                     flags: &mut u8|
     -> Result<(), ReplayError> {
        if *count == 0 {
            return Ok(());
        }
        let (codec, payload) = packer.pack(raw);
        let header = BlockHeader {
            stored_len: payload.len() as u32,
            raw_len: raw.len() as u32,
            records: *count,
            first_time: first,
            last_time: last,
            codec,
            flags: *flags,
            streams: *streams,
            payload_crc: crc32fast::hash(payload),
        };
        let payload = payload.to_vec();
        append_block(sink, &header, &payload)?;
        raw.clear();
        *count = 0;
        *streams = 0;
        *flags = 0;
        Ok(())
    };
    for (time, stream, payload) in records {
        let time = time.saturating_sub(base).max(prev);
        if count == 0 {
            first = time;
            prev = time;
        }
        format::put_record(&mut raw, prev, time, stream, payload);
        prev = time;
        count += 1;
        streams |= stream.mask();
        if raw.len() >= BLOCK_BYTES {
            flush(&mut raw, &mut count, first, prev, &mut streams, &mut flags)?;
        }
    }
    flush(&mut raw, &mut count, first, prev, &mut streams, &mut flags)
}

fn finish_file(
    file: &mut FileSink,
    meta: Option<ReplayMeta>,
    shared: &Shared,
) -> Result<Finished, ReplayError> {
    let mut meta = meta.unwrap_or_else(|| file.meta.clone());
    meta.duration = meta.duration.max(file.last_time);
    let footer = format::encode_footer(&file.entries, &meta, file.offset);
    let handle = file.file()?;
    handle.write_all(&footer)?;
    handle.sync_all()?;
    let bytes = file.offset + footer.len() as u64;
    shared.stored_bytes.store(bytes, Ordering::Relaxed);
    Ok(Finished {
        path: file.path.clone(),
        bytes,
        meta,
    })
}

/// Rewrite the footer of a finished (or crash-recovered) file with new metadata: the file is cut
/// at the end of its last good block and a fresh footer is appended. Used by renames and by the
/// repair of files a crash left without a footer.
pub(crate) fn rewrite_footer(
    path: &Path,
    entries: &[IndexEntry],
    blocks_end: u64,
    meta: &ReplayMeta,
) -> Result<(), ReplayError> {
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.set_len(blocks_end)?;
    use std::io::Seek;
    file.seek(std::io::SeekFrom::Start(blocks_end))?;
    file.write_all(&format::encode_footer(entries, meta, blocks_end))?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod w21_mw_tests {
    use super::*;

    /// After a full 1 MiB block, a level-9 writer's context holds about a
    /// 2^18-entry table, where zstd's own choice for the same level and block (the control) holds
    /// 2^21: more than 8 MiB. The block still round-trips.
    #[test]
    fn a_level_9_writer_context_stays_small() {
        let mut raw = Vec::with_capacity(1 << 20);
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        while raw.len() < 1 << 20 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let len = 8 + (seed % 48) as usize;
            raw.extend((0..len).map(|i| (seed >> (i % 8 * 8)) as u8 & 0x3F));
        }
        raw.truncate(1 << 20);

        let mut packer = Packer::new(9);
        let (codec, payload) = packer.pack(&raw);
        assert_eq!(codec, CODEC_ZSTD);
        let payload = payload.to_vec();
        let back = zstd::bulk::decompress(&payload, raw.len()).expect("decompress");
        assert_eq!(back, raw);
        let size = packer
            .cctx
            .as_mut()
            .expect("a zstd context")
            .context_mut()
            .sizeof();

        let mut own = zstd::bulk::Compressor::new(9).expect("a zstd context");
        let mut out = Vec::with_capacity(zstd::zstd_safe::compress_bound(raw.len()));
        own.compress_to_buffer(raw.as_slice(), &mut out)
            .expect("compress");
        let own_size = own.context_mut().sizeof();
        assert!(
            own_size > 8 << 20,
            "zstd's own level-9 context: {own_size} B"
        );
        if hash_log_cap_enabled() {
            assert!(size < 3 << 20, "the writer's level-9 context: {size} B");
        } else {
            assert_eq!(size, own_size);
        }
    }
}

#[cfg(test)]
mod worker_panic_tests {
    use super::*;

    /// A panic on the worker thread is the recording's failure, not the
    /// game's crash: the thread is contained when it panics (what the client's hook checks), the
    /// recorder sees `failed()` with a message (its "Recording failed" notice), records pushed
    /// afterwards are dropped, and a finish comes back with an error instead of waiting forever.
    #[test]
    fn a_worker_panic_fails_the_recording_and_is_contained() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("worker-panic.aeroreplay");
        let mut writer = Writer::create(&path, &ReplayMeta::default(), WriterOptions::default())
            .expect("writer");
        writer.push(0, StreamId::Packet, &[1, 2, 3]);
        writer.flush();
        let contained = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&contained);
        let _ = writer.tx.send(Job::Run(TestWork(Box::new(move || {
            seen.store(mc_core::panic_scope::is_contained(), Ordering::SeqCst);
            panic!("test: the worker fails");
        }))));
        let deadline = Instant::now() + Duration::from_secs(10);
        while !writer.failed() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            contained.load(Ordering::SeqCst),
            "the worker panicked outside panic_scope: the client would record it as the run's \
             crash"
        );
        assert!(
            writer.failed(),
            "the recorder must learn that the writer stopped"
        );
        assert_eq!(
            writer.error().as_deref(),
            Some("the replay writer stopped after an internal error")
        );
        writer.push(1_000, StreamId::Packet, &[4, 5, 6]);
        let finished = writer
            .finish(ReplayMeta::default())
            .wait(Duration::from_secs(10));
        assert!(finished.is_err(), "{finished:?}");
    }
}
