//! What a broken or hostile server can do to the client.
//!
//! A deterministic mutation fuzzer with no dependencies beyond the workspace. Its seeds are real
//! packets: every `.mccap` capture under `crates/mc-parity/captures/` and
//! `crates/mc-parity/tests/fixtures/`. Each round takes one real packet, damages its bytes a few
//! ways (bit flips, boundary integers and VarInts, truncation, growth, deleted and duplicated
//! ranges, crossover with another packet of the same id) and runs the result through the steps a
//! received frame takes in the client:
//!
//! 1. decode: `S2CPlay::read` plus the trailing-bytes check, exactly as `net::task::decode`;
//! 2. the chat components vanilla parses while decoding (`PacketBuffer.readChatComponent`),
//!    then flattened the way the UI draws them;
//! 3. the off-thread chunk unpack (`Chunk::from_full_column_packet`, as `sync::prebuild` does);
//! 4. `sync::apply` (or `apply_prebuilt_chunks`) on a `ClientState` that the unmutated captures
//!    built up first, so handlers meet a real world, real entities and a real tab list.
//!
//! Separate targets damage NBT, chat JSON, chunk payloads, the login packets and the server-list
//! status JSON directly, where byte-level damage to a whole packet rarely reaches.
//!
//! # What counts as a failure
//!
//! A decode or handler `Err` is fine: vanilla closes the connection on a decode exception and
//! logs a handler exception (`Util.runTask`). A failure is:
//! - a panic (a crash in the user's build);
//! - one step running longer than [`SLOW_LIMIT`]; a step that never returns trips the watchdog,
//!   which prints the input and aborts the test binary after [`HANG_LIMIT`];
//! - one allocation over [`ALLOC_LIMIT`] or live heap growing by more than [`GROWTH_LIMIT`] within
//!   one step. Requests over [`REFUSE_ABOVE`] are refused outright (the binary then aborts with
//!   "memory allocation of N bytes failed"), so a bug can never commit gigabytes on this PC.
//!
//! Every step runs on a thread with the stack of the thread it runs on in the client: 2 MiB for
//! decoding (the `mc-net` Tokio workers) and 1 MiB for handlers and UI text (the Windows main
//! thread). Debug frames are larger than release ones, so passing here is the stricter test.
//!
//! # Bounded on purpose
//!
//! The default rounds finish in seconds per target. Exploratory runs scale them with
//! `Z1_FUZZ_ROUNDS=<multiplier>` and add seed packets from replays with
//! `Z1_FUZZ_REPLAYS=<folder of .aeroreplay files>` (local only: replays hold other players' data,
//! so they never become committed fixtures). `Z1_FUZZ_TRACE=1` prints every input before it runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[path = "z1/one_of_each.rs"]
mod one_of_each;

use mc_client::sync::{apply, ClientEvent, ClientState};
use mc_parity::capture::{Capture, Direction};
use mc_protocol::play::S2CPlay;
use mc_protocol::{Packet, PacketBuffer, State};
use mc_ui::font::component::ChatComponent;

// ---- allocation guard ------------------------------------------------------------------------

/// One allocation larger than this is a finding: nothing the client legitimately builds from one
/// packet comes close (a full chunk column is about 80 KiB; the frame limit is 2 MiB).
const ALLOC_LIMIT: usize = 64 << 20;
/// Live heap growth within one step larger than this is a finding.
const GROWTH_LIMIT: isize = 256 << 20;
/// Larger requests are refused (null), which aborts the binary with Rust's allocation-failure
/// message instead of committing that much memory on a PC that hard-resets under load.
const REFUSE_ABOVE: usize = 1 << 30;
/// One step slower than this (in this unoptimised build) is a finding.
const SLOW_LIMIT: Duration = Duration::from_secs(10);
/// One step slower than this is a hang: the watchdog prints the input and aborts.
const HANG_LIMIT: Duration = Duration::from_secs(60);

thread_local! {
    static LARGEST: Cell<usize> = const { Cell::new(0) };
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn note_alloc(size: usize) {
    let _ = LARGEST.try_with(|largest| {
        if size > largest.get() {
            largest.set(size);
        }
    });
    let _ = LIVE.try_with(|live| {
        let now = live.get() + size as isize;
        live.set(now);
        let _ = PEAK.try_with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

fn note_free(size: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - size as isize));
}

/// Per-thread allocation bookkeeping around `System`. Thread-local, because libtest runs the
/// targets side by side and one target's chunk burst must not be blamed on another's input.
struct Guard;

// SAFETY: every call is forwarded to `System` unchanged, except that requests over
// `REFUSE_ABOVE` return null, which `GlobalAlloc` permits. The bookkeeping touches only
// const-initialised thread-locals without destructors, which never allocate.
unsafe impl GlobalAlloc for Guard {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSE_ABOVE {
            return std::ptr::null_mut();
        }
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            note_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSE_ABOVE {
            return std::ptr::null_mut();
        }
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            note_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note_free(layout.size());
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > REFUSE_ABOVE {
            return std::ptr::null_mut();
        }
        // SAFETY: same contract as the caller's.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            note_free(layout.size());
            note_alloc(new_size);
        }
        new
    }
}

#[global_allocator]
static GLOBAL: Guard = Guard;

/// What one step cost: its largest single allocation and its peak live-heap growth.
#[derive(Debug, Clone, Copy, Default)]
struct Cost {
    largest: usize,
    growth: isize,
    elapsed: Duration,
}

/// Run `step`, measuring its allocations on this thread.
fn measured<T>(step: impl FnOnce() -> T) -> (T, Cost) {
    let base = LIVE.with(Cell::get);
    LARGEST.with(|largest| largest.set(0));
    PEAK.with(|peak| peak.set(base));
    let started = Instant::now();
    let out = step();
    let elapsed = started.elapsed();
    let cost = Cost {
        largest: LARGEST.with(Cell::get),
        growth: PEAK.with(Cell::get) - base,
        elapsed,
    };
    (out, cost)
}

// ---- deterministic randomness and mutation ---------------------------------------------------

/// xorshift64*: every failure reproduces from its target and round number.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    fn one_in(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

/// Values that sit on a boundary of some length, count, index or bitmask check.
const INTERESTING: [i64; 24] = [
    0,
    1,
    -1,
    2,
    15,
    16,
    127,
    128,
    255,
    256,
    4095,
    4096,
    32_767,
    32_768,
    65_535,
    65_536,
    (1 << 21) - 1,
    1 << 21,
    0x0FFF_FFFF,
    i32::MAX as i64,
    i32::MIN as i64,
    i64::MAX,
    i64::MIN,
    -129,
];

fn varint_bytes(value: i32) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    buf.write_varint(value);
    buf.as_slice().to_vec()
}

/// Damage `data` once.
fn mutate_once(rng: &mut Rng, data: &mut Vec<u8>, donors: &[Vec<u8>]) {
    let len = data.len();
    let at = rng.below(len + 1);
    match rng.below(13) {
        0 if len > 0 => {
            let i = rng.below(len);
            data[i] ^= 1 << rng.below(8);
        }
        1 if len > 0 => {
            let i = rng.below(len);
            data[i] = *rng.pick(&[0x00, 0x01, 0x7F, 0x80, 0xFF, 0x08, 0x09, 0x0A, 0x0B, 0x0C]);
        }
        2 if len > 0 => {
            let i = rng.below(len);
            data[i] = rng.next() as u8;
        }
        3..=5 => {
            // A big-endian integer of 2, 4 or 8 bytes, overwritten in place.
            let width = [2usize, 4, 8][rng.below(3)];
            let value = *rng.pick(&INTERESTING);
            let bytes = value.to_be_bytes();
            let bytes = &bytes[8 - width..];
            for (k, byte) in bytes.iter().enumerate() {
                if let Some(slot) = data.get_mut(at + k) {
                    *slot = *byte;
                }
            }
        }
        6 => {
            // A VarInt over 1-5 bytes at `at`: lengths, counts and ids are VarInts.
            let value = *rng.pick(&INTERESTING) as i32;
            let encoded = varint_bytes(value);
            let replace = rng.below(5).min(len - at.min(len));
            data.splice(at..at + replace, encoded);
        }
        7 => data.truncate(at),
        8 => {
            let n = 1 + rng.below(16);
            for _ in 0..n {
                data.push(rng.next() as u8);
            }
        }
        9 if len > 0 => {
            let end = (at + 1 + rng.below(16)).min(len);
            let start = at.min(end);
            data.drain(start..end);
        }
        10 if len > 0 => {
            // Duplicate a range: repeats nesting openers in NBT and JSON.
            let start = rng.below(len);
            let end = (start + 1 + rng.below(64)).min(len);
            let copy = data[start..end].to_vec();
            let times = 1 + rng.below(4);
            let insert_at = rng.below(data.len() + 1);
            for _ in 0..times {
                data.splice(insert_at..insert_at, copy.iter().copied());
            }
        }
        11 if !donors.is_empty() => {
            // Crossover: our head, another packet's tail.
            let donor = rng.pick(donors);
            let from = rng.below(donor.len() + 1);
            data.truncate(at);
            data.extend_from_slice(&donor[from..]);
        }
        _ => {
            // Insert a few random bytes.
            let n = 1 + rng.below(8);
            let bytes: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            data.splice(at..at, bytes);
        }
    }
    // Keep growth bounded: the frame limit is 2 MiB anyway.
    data.truncate(1 << 21);
}

fn mutate(rng: &mut Rng, seed: &[u8], donors: &[Vec<u8>]) -> Vec<u8> {
    let mut data = seed.to_vec();
    if rng.one_in(2) && !data.is_empty() {
        // Half the rounds keep every length and offset: only values change, so the packet still
        // decodes and its damaged fields reach the handlers.
        for _ in 0..1 + rng.below(3) {
            let i = rng.below(data.len());
            match rng.below(3) {
                0 => data[i] ^= 1 << rng.below(8),
                1 => data[i] = rng.next() as u8,
                _ => {
                    let width = [2usize, 4, 8][rng.below(3)];
                    let bytes = rng.pick(&INTERESTING).to_be_bytes();
                    for (k, byte) in bytes[8 - width..].iter().enumerate() {
                        if let Some(slot) = data.get_mut(i + k) {
                            *slot = *byte;
                        }
                    }
                }
            }
        }
        return data;
    }
    for _ in 0..1 + rng.below(4) {
        mutate_once(rng, &mut data, donors);
    }
    data
}

// ---- seeds -------------------------------------------------------------------------------------

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates directory")
        .to_path_buf()
}

fn mccap_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<_> = entries.flatten().map(|entry| entry.path()).collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "mccap") {
                out.push(path);
            }
        }
    }
    let parity = crates_dir().join("mc-parity");
    let mut out = Vec::new();
    walk(&parity.join("captures"), &mut out);
    walk(&parity.join("tests").join("fixtures"), &mut out);
    out
}

/// One clientbound packet stream (id VarInt + body per payload), in arrival order.
struct Stream {
    name: String,
    play: Vec<Vec<u8>>,
}

/// Every seed, grouped by state and packet id.
struct Seeds {
    streams: Vec<Stream>,
    play_by_id: BTreeMap<i32, Vec<Vec<u8>>>,
    login: Vec<Vec<u8>>,
    status: Vec<Vec<u8>>,
}

/// Keep at most this many distinct payloads per packet id (the smallest and largest first).
const SEEDS_PER_ID: usize = 48;

fn split_id(payload: &[u8]) -> Option<(i32, &[u8])> {
    let mut buf = PacketBuffer::from_slice(payload);
    let id = buf.read_varint().ok()?;
    let used = payload.len() - buf.remaining();
    Some((id, &payload[used..]))
}

fn load_seeds() -> Seeds {
    let mut streams = Vec::new();
    let mut login = Vec::new();
    let mut status = Vec::new();
    for path in mccap_files() {
        let capture =
            Capture::read_file(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let mut play = Vec::new();
        for frame in capture.frames_in(Direction::Clientbound) {
            match frame.state {
                State::Play => play.push(frame.payload.clone()),
                State::Login => login.push(frame.payload.clone()),
                State::Status => status.push(frame.payload.clone()),
                State::Handshaking => {}
            }
        }
        streams.push(Stream {
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            play,
        });
    }
    if let Some(dir) = std::env::var_os("Z1_FUZZ_REPLAYS") {
        streams.extend(replay_streams(Path::new(&dir)));
    }
    let mut play_by_id: BTreeMap<i32, Vec<Vec<u8>>> = BTreeMap::new();
    let mut seen = HashSet::new();
    for stream in &streams {
        for payload in &stream.play {
            if let Some((id, _)) = split_id(payload) {
                if seen.insert(payload.clone()) {
                    play_by_id.entry(id).or_default().push(payload.clone());
                }
            }
        }
    }
    // One valid instance of every packet type as well: the captures lack some types entirely
    // (explosions, titles, world borders, maps...).
    for packet in one_of_each::one_of_each() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        play_by_id
            .entry(packet.id())
            .or_default()
            .push(buf.as_slice().to_vec());
    }
    for payloads in play_by_id.values_mut() {
        if payloads.len() > SEEDS_PER_ID {
            payloads.sort_by_key(Vec::len);
            let step = payloads.len() as f64 / SEEDS_PER_ID as f64;
            let kept: Vec<_> = (0..SEEDS_PER_ID)
                .map(|k| payloads[((k as f64 * step) as usize).min(payloads.len() - 1)].clone())
                .collect();
            *payloads = kept;
        }
    }
    Seeds {
        streams,
        play_by_id,
        login,
        status,
    }
}

/// Extra seed streams from local replays (exploratory runs only).
fn replay_streams(dir: &Path) -> Vec<Stream> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut paths: Vec<_> = entries.flatten().map(|entry| entry.path()).collect();
    paths.sort();
    for path in paths {
        if path
            .extension()
            .is_none_or(|ext| ext != mc_replay::EXTENSION)
        {
            continue;
        }
        let Ok(reader) = mc_replay::Reader::open(&path) else {
            continue;
        };
        let play = reader
            .records()
            .map_while(Result::ok)
            .filter(|record| record.stream == mc_replay::StreamId::Packet)
            .map(|record| record.payload)
            .collect();
        out.push(Stream {
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            play,
        });
    }
    out
}

fn rounds(default: usize) -> usize {
    let multiplier = std::env::var("Z1_FUZZ_ROUNDS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.0);
    ((default as f64) * multiplier).max(1.0) as usize
}

fn trace() -> bool {
    std::env::var_os("Z1_FUZZ_TRACE").is_some()
}

// ---- findings, the watchdog and stack-sized runners ------------------------------------------

fn hex(bytes: &[u8]) -> String {
    let shown: String = bytes.iter().take(400).map(|b| format!("{b:02x}")).collect();
    if bytes.len() > 400 {
        format!("{shown}.. ({} bytes)", bytes.len())
    } else {
        shown
    }
}

fn panic_text(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "a non-string panic".to_owned())
}

/// The input currently running on one target, for the watchdog.
#[derive(Default)]
struct Slot {
    started: Option<Instant>,
    label: String,
    input: Vec<u8>,
}

/// One fuzz target's bookkeeping.
struct Target {
    name: &'static str,
    slot: Arc<Mutex<Slot>>,
    done: Arc<AtomicBool>,
    findings: Vec<String>,
    steps: usize,
    accepted: usize,
    slowest: Duration,
    largest: usize,
}

impl Target {
    fn new(name: &'static str) -> Target {
        let slot = Arc::new(Mutex::new(Slot::default()));
        let done = Arc::new(AtomicBool::new(false));
        {
            let slot = Arc::clone(&slot);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(250));
                    let slot = slot
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if let Some(started) = slot.started {
                        if started.elapsed() > HANG_LIMIT {
                            eprintln!(
                                "Z1 HANG in {name}: {} has run {:?}; input {}",
                                slot.label,
                                started.elapsed(),
                                hex(&slot.input)
                            );
                            std::process::abort();
                        }
                    }
                }
            });
        }
        Target {
            name,
            slot,
            done,
            findings: Vec::new(),
            steps: 0,
            accepted: 0,
            slowest: Duration::ZERO,
            largest: 0,
        }
    }

    /// Run one step on `input`, catching panics and judging its cost.
    fn step<T>(&mut self, label: &str, input: &[u8], step: impl FnOnce() -> T) -> Option<T> {
        if trace() {
            eprintln!("Z1 {} {label}: {}", self.name, hex(input));
        }
        {
            let mut slot = self
                .slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            slot.started = Some(Instant::now());
            slot.label.clear();
            slot.label.push_str(label);
            slot.input.clear();
            slot.input.extend_from_slice(input);
        }
        let (result, cost) =
            measured(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(step)));
        self.slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .started = None;
        self.steps += 1;
        self.slowest = self.slowest.max(cost.elapsed);
        self.largest = self.largest.max(cost.largest);
        let mut problems = Vec::new();
        if cost.elapsed > SLOW_LIMIT {
            problems.push(format!("took {:?}", cost.elapsed));
        }
        if cost.largest > ALLOC_LIMIT {
            problems.push(format!("allocated {} bytes at once", cost.largest));
        }
        if cost.growth > GROWTH_LIMIT {
            problems.push(format!("grew the heap by {} bytes", cost.growth));
        }
        let out = match result {
            Ok(out) => Some(out),
            Err(panic) => {
                problems.push(format!("panicked: {}", panic_text(&*panic)));
                None
            }
        };
        if !problems.is_empty() && self.findings.len() < 40 {
            self.findings.push(format!(
                "{} {label}: {}\n    input {}",
                self.name,
                problems.join("; "),
                hex(input)
            ));
        }
        out
    }

    fn finish(self) {
        self.done.store(true, Ordering::Relaxed);
        println!(
            "{}: {} steps, {} accepted, slowest {:?}, largest allocation {} bytes, {} findings",
            self.name,
            self.steps,
            self.accepted,
            self.slowest,
            self.largest,
            self.findings.len()
        );
        assert!(
            self.findings.is_empty(),
            "{} findings in {}:\n{}",
            self.findings.len(),
            self.name,
            self.findings.join("\n")
        );
    }
}

/// Run `body` on a thread with `stack` bytes of stack, as the client's own thread has.
fn on_stack(stack: usize, body: impl FnOnce() + Send + 'static) {
    let joined = std::thread::Builder::new()
        .stack_size(stack)
        .spawn(body)
        .expect("spawn a fuzz thread")
        .join();
    if let Err(panic) = joined {
        std::panic::resume_unwind(panic);
    }
}

/// The Windows main thread's default stack: the game loop, packet handlers and UI text.
const MAIN_STACK: usize = 1 << 20;
/// Tokio's default worker stack: the `mc-net` threads that decode packets.
const NET_STACK: usize = 2 << 20;

// ---- the pipeline a received frame takes -----------------------------------------------------

/// `net::task::decode`: the id, the body, and nothing left over.
fn decode(payload: &[u8]) -> Option<S2CPlay> {
    let mut buf = PacketBuffer::from_slice(payload);
    let id = buf.read_varint().ok()?;
    let packet = S2CPlay::read(id, &mut buf).ok()?;
    buf.is_empty().then_some(packet)
}

/// Every chat component a packet carries (the fields vanilla reads with `readChatComponent`).
fn components(packet: &S2CPlay) -> Vec<&str> {
    use mc_protocol::play::s2c;
    match packet {
        S2CPlay::Chat(chat) => vec![&chat.component],
        S2CPlay::OpenWindow(open) => vec![&open.window_title],
        S2CPlay::UpdateSign(update) => update.lines.iter().map(String::as_str).collect(),
        S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(entries)) => entries
            .iter()
            .filter_map(|entry| entry.display_name.as_deref())
            .collect(),
        S2CPlay::PlayerListItem(s2c::PlayerListItem::UpdateDisplayName(entries)) => entries
            .iter()
            .filter_map(|entry| entry.display_name.as_deref())
            .collect(),
        S2CPlay::Disconnect(disconnect) => vec![&disconnect.reason],
        S2CPlay::Title(s2c::Title::SetTitle { text } | s2c::Title::SetSubtitle { text }) => {
            vec![text]
        }
        S2CPlay::PlayerListHeaderFooter(header_footer) => {
            vec![&header_footer.header, &header_footer.footer]
        }
        _ => Vec::new(),
    }
}

/// Parse a component as the decoder does, then flatten it as the UI does. `Err` when the
/// decoder would have refused it (vanilla's `JsonSyntaxException` disconnect).
fn render_component(json: &str, lang: &mc_assets::Language) -> Result<usize, ()> {
    match ChatComponent::from_json_nullable(json) {
        Ok(Some(component)) => {
            let spans = component.spans(lang).len();
            let _ = component.formatted_text(lang);
            let _ = component.unformatted_text(lang);
            // The client writes components back (chat copy, replays): that must not fail either.
            let _ = ChatComponent::from_json(&component.to_json());
            Ok(spans)
        }
        Ok(None) => Ok(0),
        Err(_) => Err(()),
    }
}

/// `sync::prebuild::unpack_columns`: the columns of a full-column chunk packet.
fn unpack_columns(packet: &S2CPlay, has_sky_light: bool) -> Option<Vec<mc_world::Chunk>> {
    match packet {
        S2CPlay::ChunkData(chunk) if chunk.ground_up_continuous && chunk.primary_bit_mask != 0 => {
            mc_world::Chunk::from_full_column_packet(
                chunk.primary_bit_mask,
                has_sky_light,
                &chunk.data,
            )
            .ok()
            .map(|column| vec![column])
        }
        S2CPlay::MapChunkBulk(bulk) => bulk
            .columns
            .iter()
            .map(|column| {
                mc_world::Chunk::from_full_column_packet(
                    column.primary_bit_mask,
                    has_sky_light,
                    &column.data,
                )
                .ok()
            })
            .collect(),
        _ => None,
    }
}

/// Apply one decoded packet the way `Session::apply_packet` does, through the prebuilt-column
/// path when the packet has columns and the plain `apply` otherwise.
fn apply_like_session(state: &mut ClientState, packet: S2CPlay, flip_sky: bool) {
    let mut events: Vec<ClientEvent> = Vec::new();
    let assumed_sky = match &packet {
        S2CPlay::MapChunkBulk(bulk) => bulk.has_sky_light,
        _ => state.world.has_sky_light(),
    } ^ flip_sky;
    match unpack_columns(&packet, assumed_sky) {
        Some(columns) => {
            let _ = mc_client::sync::dispatch::apply_prebuilt_chunks(
                state,
                packet,
                assumed_sky,
                columns,
                &mut events,
            );
        }
        None => {
            let _ = apply(state, packet, &mut events);
        }
    }
}

/// A `ClientState` after every unmutated packet of `stream`.
fn replayed_state(stream: &Stream) -> ClientState {
    let mut state = ClientState::new();
    for payload in &stream.play {
        if let Some(packet) = decode(payload) {
            apply_like_session(&mut state, packet, false);
        }
    }
    state
}

// ---- targets -----------------------------------------------------------------------------------

/// Every unmutated seed must decode in full: the corpus is real traffic, so a refusal here is a
/// decoder that disconnects from a real server.
#[test]
fn every_captured_packet_decodes_in_full() {
    let seeds = load_seeds();
    let mut refused = Vec::new();
    let mut total = 0usize;
    for stream in &seeds.streams {
        for payload in &stream.play {
            total += 1;
            if decode(payload).is_none() {
                refused.push(format!("{}: {}", stream.name, hex(payload)));
            }
        }
    }
    println!("{total} captured clientbound Play packets");
    let captured: HashSet<i32> = seeds
        .streams
        .iter()
        .flat_map(|stream| &stream.play)
        .filter_map(|payload| split_id(payload).map(|(id, _)| id))
        .collect();
    let missing: Vec<String> = (0x00..=0x49)
        .filter(|id| !captured.contains(id))
        .map(|id| format!("{id:#04x}"))
        .collect();
    // These ids are seeded from `z1/one_of_each.rs` instead.
    println!("packet ids without a captured seed: {missing:?}");
    assert!((0x00..=0x49).all(|id| seeds.play_by_id.contains_key(&id)));
    assert!(
        total > 1_000,
        "the seed corpus went missing ({total} packets)"
    );
    assert!(
        refused.is_empty(),
        "{} refused:\n{}",
        refused.len(),
        refused.join("\n")
    );
}

/// Mutated real packets through decode, the decoder's component parse, the UI's flattening, the
/// chunk unpack and the handlers, on a live state built from the captures.
#[test]
fn damaged_play_packets_never_crash_hang_or_balloon() {
    on_stack(MAIN_STACK, || {
        let seeds = load_seeds();
        let lang = mc_assets::Language::new("en_US");
        let ids: Vec<i32> = seeds.play_by_id.keys().copied().collect();
        let mut target = Target::new("play packets");
        let mut rng = Rng::new(0x21_0001);
        // The richest stream first: it has the most chunks, entities and players.
        let base = seeds
            .streams
            .iter()
            .max_by_key(|stream| stream.play.len())
            .expect("a capture");
        let mut state = replayed_state(base);
        let total = rounds(30_000);
        for round in 0..total {
            if round % 1_500 == 1_499 {
                // Mutated JoinGame/Respawn packets wipe the world now and then; start again
                // from a populated one.
                state = replayed_state(rng.pick(&seeds.streams));
            }
            // Round-robin over packet ids, so rare packets get as many rounds as chunk bursts.
            let id = ids[round % ids.len()];
            let donors = &seeds.play_by_id[&id];
            let seed = rng.pick(donors);
            let mut input = mutate(&mut rng, seed, donors);
            if rng.one_in(40) {
                // The same body under another id: a different decoder meets these bytes.
                let other = *rng.pick(&ids);
                if let Some((_, body)) = split_id(&input) {
                    let mut swapped = varint_bytes(other);
                    swapped.extend_from_slice(body);
                    input = swapped;
                }
            }
            let flip_sky = rng.one_in(8);
            let label = format!("round {round} id {id:#04x}");
            let decoded = target
                .step(&format!("{label} decode"), &input, || decode(&input))
                .flatten();
            let Some(packet) = decoded else {
                continue;
            };
            target.accepted += 1;
            let fields: Vec<String> = components(&packet).into_iter().map(str::to_owned).collect();
            let mut refused = false;
            for field in &fields {
                let parsed = target.step(&format!("{label} component"), field.as_bytes(), || {
                    render_component(field, &lang)
                });
                refused |= matches!(parsed, Some(Err(())));
            }
            if refused {
                // net::task disconnects here, before any handler runs.
                continue;
            }
            let applied = target.step(&format!("{label} apply"), &input, || {
                apply_like_session(&mut state, packet, flip_sky);
            });
            if applied.is_none() {
                // A panic may leave the state half-updated.
                state = replayed_state(base);
            }
        }
        target.finish();
    });
}

/// NBT exactly as `readNBTTagCompoundFromBuffer` meets it (2 MiB size budget, depth 512), seeded
/// with every item and tile-entity tag in the captures.
#[test]
fn damaged_nbt_never_crashes_or_balloons() {
    on_stack(NET_STACK, || {
        let seeds = load_seeds();
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        let mut seen = HashSet::new();
        for payloads in seeds.play_by_id.values() {
            for payload in payloads {
                for tag in nbt_of(decode(payload).as_ref()) {
                    let mut bytes = Vec::new();
                    if mc_nbt::write_compound(&mut bytes, "", &tag).is_ok()
                        && seen.insert(bytes.clone())
                    {
                        blobs.push(bytes);
                    }
                }
            }
        }
        blobs.extend(handmade_nbt());
        let mut target = Target::new("nbt");
        let mut rng = Rng::new(0x21_0002);
        for round in 0..rounds(60_000) {
            let seed = rng.pick(&blobs);
            let input = mutate(&mut rng, seed, &blobs);
            let read = target.step(&format!("round {round}"), &input, || {
                let mut buf = PacketBuffer::from_slice(&input);
                buf.read_nbt_compound_opt()
            });
            if let Some(Ok(Some(tag))) = read {
                target.accepted += 1;
                // The client writes item tags back unchanged in window clicks.
                target.step(&format!("round {round} rewrite"), &input, || {
                    let mut out = PacketBuffer::new();
                    out.write_nbt_compound_opt(Some(&tag));
                    let _ = out.finish_write();
                });
            }
        }
        target.finish();
    });
}

/// The NBT trees a packet carries: item tags and tile-entity data.
fn nbt_of(packet: Option<&S2CPlay>) -> Vec<mc_nbt::NbtCompound> {
    let Some(packet) = packet else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut item = |stack: &Option<mc_protocol::ItemStack>| {
        if let Some(tag) = stack.as_ref().and_then(|stack| stack.nbt.clone()) {
            out.push(tag);
        }
    };
    match packet {
        S2CPlay::EntityEquipment(equipment) => item(&equipment.item),
        S2CPlay::SetSlot(set) => item(&set.item),
        S2CPlay::WindowItems(items) => items.items.iter().for_each(&mut item),
        _ => {}
    }
    if let S2CPlay::UpdateTileEntity(update) = packet {
        if let Some(tag) = &update.nbt {
            out.push(tag.clone());
        }
    }
    out
}

/// Tags the captures lack: deep nesting, lists of lists, arrays, long strings.
fn handmade_nbt() -> Vec<Vec<u8>> {
    use mc_nbt::{Nbt, NbtCompound};
    let mut deep = NbtCompound::new();
    for depth in 0..40 {
        let mut outer = NbtCompound::new();
        outer.insert(format!("d{depth}"), Nbt::Compound(deep));
        deep = outer;
    }
    let mut display = NbtCompound::new();
    display.insert("Name", Nbt::String("\u{a7}6Sword".into()));
    display.insert("Lore", Nbt::list(vec![Nbt::String("line".into())]));
    let mut tag = NbtCompound::new();
    tag.insert("display", Nbt::Compound(display));
    tag.insert("ints", Nbt::IntArray(vec![1, 2, 3]));
    tag.insert("bytes", Nbt::ByteArray(vec![1, 2, 3]));
    tag.insert("long", Nbt::Long(-1));
    tag.insert("double", Nbt::Double(f64::NAN));
    [deep, tag]
        .iter()
        .filter_map(|compound| {
            let mut bytes = Vec::new();
            mc_nbt::write_compound(&mut bytes, "", compound).ok()?;
            Some(bytes)
        })
        .collect()
}

/// Chat JSON as the decoder parses it and the UI flattens it, seeded with every component in the
/// captures plus the shapes they lack (translations with arguments, hovers, scores, selectors).
#[test]
fn damaged_chat_json_never_crashes_hangs_or_balloons() {
    on_stack(MAIN_STACK, || {
        let seeds = load_seeds();
        let lang = mc_assets::Language::new("en_US");
        let mut corpus: Vec<Vec<u8>> = Vec::new();
        let mut seen = HashSet::new();
        for payloads in seeds.play_by_id.values() {
            for payload in payloads {
                if let Some(packet) = decode(payload) {
                    for field in components(&packet) {
                        if seen.insert(field.to_owned()) {
                            corpus.push(field.as_bytes().to_vec());
                        }
                    }
                }
            }
        }
        corpus.extend(HANDMADE_JSON.iter().map(|json| json.as_bytes().to_vec()));
        let tokens: Vec<Vec<u8>> = JSON_TOKENS.iter().map(|t| t.as_bytes().to_vec()).collect();
        let mut target = Target::new("chat json");
        let mut rng = Rng::new(0x21_0003);
        for round in 0..rounds(40_000) {
            let seed = rng.pick(&corpus);
            let mut input = if rng.one_in(2) {
                mutate(&mut rng, seed, &corpus)
            } else {
                seed.clone()
            };
            // Structure-aware damage: JSON tokens dropped in anywhere, a few times.
            for _ in 0..rng.below(4) {
                let token = rng.pick(&tokens);
                let at = rng.below(input.len() + 1);
                input.splice(at..at, token.iter().copied());
            }
            // The decoder's string read replaces invalid UTF-8 as Java does.
            let json = String::from_utf8_lossy(&input).into_owned();
            let parsed = target.step(&format!("round {round}"), json.as_bytes(), || {
                render_component(&json, &lang)
            });
            if matches!(parsed, Some(Ok(_))) {
                target.accepted += 1;
            }
        }
        target.finish();
    });
}

const HANDMADE_JSON: [&str; 14] = [
    r#"{"text":"plain"}"#,
    r#""bare string""#,
    r#"[{"text":"a"},{"text":"b","color":"red"}]"#,
    r#"{"translate":"chat.type.text","with":[{"text":"Name"},{"text":"hello"}]}"#,
    r#"{"translate":"%1$s and %2$s and %1$s","with":["x",{"text":"y","bold":true}]}"#,
    r#"{"translate":"%s%%%3$s","with":["a"]}"#,
    r#"{"text":"","extra":[{"text":"x","hoverEvent":{"action":"show_text","value":{"text":"t"}}}]}"#,
    r#"{"text":"i","hoverEvent":{"action":"show_item","value":"{id:276,Count:1b,tag:{display:{Name:\"S\",Lore:[0:\"a\",1:\"b\"]},ench:[0:{id:16s,lvl:5s}]}}"}}"#,
    r#"{"text":"c","clickEvent":{"action":"run_command","value":"/help"}}"#,
    r#"{"score":{"name":"p","objective":"o","value":"7"}}"#,
    r#"{"selector":"@p"}"#,
    r#"{text:unquoted,color:gold,extra:['single',"double"]}"#,
    r#"{"text":"😀 §cred"}"#,
    "null",
];

/// Status responses of every shape the server list reads, for the status oracle.
const HANDMADE_STATUS: [&str; 8] = [
    r#"{"description":{"text":"A ","extra":[{"text":"server","color":"aqua"}]},"players":{"max":20,"online":1,"sample":[{"name":"x","id":"00000000-0000-0000-0000-000000000001"}]},"version":{"name":"1.8.8","protocol":47}}"#,
    r#"{"description":"plain","players":{"max":0,"online":0},"version":{"name":"x","protocol":5}}"#,
    r#"{"description":null,"players":null,"version":null,"favicon":null}"#,
    r#"{}"#,
    r#"{"description":{"translate":"%1$s","with":["a"]},"players":{"max":"20","online":"1"},"version":{"name":1,"protocol":"47"}}"#,
    r#"{"players":{"max":1,"online":1,"sample":[]}}"#,
    r#"{"favicon":"data:image/png;base64,AAAA"}"#,
    r#"{description:lenient,players:{max:1,online:1},version:{name:v,protocol:47}}"#,
];

/// Gson 2.2.4's lenient spellings, for the chat oracle.
const LENIENT_JSON: [&str; 10] = [
    "{text:plain}",
    "{'text':'single'}",
    "{\"text\"=\"equals\"}",
    "{\"text\"=>\"arrow\"}",
    "{\"text\":\"a\";\"color\":\"red\"}",
    "// comment\n{\"text\":\"x\"}",
    "# comment\n{\"text\":\"x\"}",
    "/* block */ {\"text\":\"x\"}",
    ")]}'\n{\"text\":\"x\"}",
    "[\"a\",,\"b\"]",
];

/// Lenient-grammar fragments to splice into chat JSON for the oracle.
const LENIENT_TOKENS: [&str; 16] = [
    "'", "=", "=>", ";", "#", "//", "/*", "*/", "\n", "\r", "\t", "\u{c}", " ", "\u{a0}",
    "\u{feff}", ")]}'\n",
];

/// Fragments that change JSON structure when spliced in anywhere.
const JSON_TOKENS: [&str; 24] = [
    "{",
    "}",
    "[",
    "]",
    "\"",
    "\\",
    ",",
    ":",
    "\\u",
    "\\ud800",
    "null",
    "\"text\":",
    "\"extra\":[",
    "\"with\":[",
    "\"translate\":\"%1$s%1$s%1$s%1$s\",",
    "{\"translate\":\"%1$s%1$s\",\"with\":[",
    "\"%999999999999$s\"",
    "\"hoverEvent\":{\"action\":\"show_item\",\"value\":\"{",
    "\"clickEvent\":{\"action\":\"open_url\",\"value\":\"",
    "\"score\":{\"name\":\"*\",\"objective\":\"",
    "1e999999",
    "-0",
    "/*",
    "\u{0}",
];

/// Chunk payloads straight into the unpacker and the partial-update path, with every bitmask
/// shape and both sky-light layouts.
#[test]
fn damaged_chunk_payloads_never_crash_or_balloon() {
    on_stack(NET_STACK, || {
        let seeds = load_seeds();
        let mut payloads: Vec<(u16, Vec<u8>)> = Vec::new();
        for id in [0x21, 0x26] {
            for payload in seeds.play_by_id.get(&id).into_iter().flatten() {
                match decode(payload) {
                    Some(S2CPlay::ChunkData(chunk)) => {
                        payloads.push((chunk.primary_bit_mask, chunk.data));
                    }
                    Some(S2CPlay::MapChunkBulk(bulk)) => payloads.extend(
                        bulk.columns
                            .into_iter()
                            .take(4)
                            .map(|column| (column.primary_bit_mask, column.data)),
                    ),
                    _ => {}
                }
            }
        }
        assert!(!payloads.is_empty(), "the captures hold chunk packets");
        let donors: Vec<Vec<u8>> = payloads.iter().map(|(_, data)| data.clone()).collect();
        let mut target = Target::new("chunk payloads");
        let mut rng = Rng::new(0x21_0004);
        for round in 0..rounds(8_000) {
            let (mask, data) = rng.pick(&payloads);
            let data = if rng.one_in(4) {
                data.clone()
            } else {
                mutate(&mut rng, data, &donors)
            };
            let mask = match rng.below(4) {
                0 => *mask,
                1 => *mask ^ (1 << rng.below(16)),
                2 => rng.next() as u16,
                _ => [0u16, 1, 0x8000, 0xFFFF][rng.below(4)],
            };
            let sky = rng.one_in(2);
            let ground_up = !rng.one_in(3);
            let built = target.step(&format!("round {round} mask {mask:#06x}"), &data, || {
                let mut column = mc_world::Chunk::empty();
                // A partial update lands on an existing column.
                if !ground_up {
                    let _ = column.fill_from_packet(true, 0xFFFF, sky, &vec![1u8; 196_864]);
                }
                column.fill_from_packet(ground_up, mask, sky, &data).is_ok()
            });
            if built == Some(true) {
                target.accepted += 1;
            }
        }
        target.finish();
    });
}

/// The login packets and the server-list status JSON: both arrive before any world exists, and
/// the status JSON comes from every server in the list just by opening the multiplayer screen.
#[test]
fn damaged_login_and_status_never_crash_hang_or_balloon() {
    on_stack(MAIN_STACK, || {
        let seeds = load_seeds();
        let lang = mc_assets::Language::new("en_US");
        let mut target = Target::new("login and status");
        let mut rng = Rng::new(0x21_0005);
        let mut login = seeds.login.clone();
        login.extend(handmade_login());
        for round in 0..rounds(20_000) {
            let seed = rng.pick(&login);
            let input = mutate(&mut rng, seed, &login);
            target.step(&format!("login round {round}"), &input, || {
                let mut buf = PacketBuffer::from_slice(&input);
                let Ok(id) = buf.read_varint() else {
                    return;
                };
                match id {
                    0x00 => {
                        if let Ok(packet) = mc_protocol::login::Disconnect::read(&mut buf) {
                            let _ = render_component(&packet.reason, &lang);
                        }
                    }
                    0x01 => {
                        let _ = mc_protocol::login::EncryptionRequest::read(&mut buf);
                    }
                    0x02 => {
                        let _ = mc_protocol::login::LoginSuccess::read(&mut buf);
                    }
                    0x03 => {
                        let _ = mc_protocol::login::SetCompression::read(&mut buf);
                    }
                    _ => {}
                }
            });
        }
        let mut statuses: Vec<Vec<u8>> = seeds
            .status
            .iter()
            .filter_map(|payload| {
                let (id, body) = split_id(payload)?;
                (id == 0x00).then(|| body.to_vec())
            })
            .filter_map(|body| {
                mc_protocol::status::Response::read(&mut PacketBuffer::from_slice(&body))
                    .ok()
                    .map(|response| response.json.into_bytes())
            })
            .collect();
        statuses.push(
            br#"{"description":{"text":"A ","extra":[{"text":"server","color":"aqua"}]},"players":{"max":20,"online":1,"sample":[{"name":"x","id":"00000000-0000-0000-0000-000000000001"}]},"version":{"name":"1.8.8","protocol":47}}"#
                .to_vec(),
        );
        let tokens: Vec<Vec<u8>> = JSON_TOKENS.iter().map(|t| t.as_bytes().to_vec()).collect();
        for round in 0..rounds(20_000) {
            let seed = rng.pick(&statuses);
            let mut input = mutate(&mut rng, seed, &statuses);
            if rng.one_in(2) {
                let token = rng.pick(&tokens);
                let at = rng.below(input.len() + 1);
                input.splice(at..at, token.iter().copied());
            }
            let json = String::from_utf8_lossy(&input).into_owned();
            let parsed = target.step(&format!("status round {round}"), json.as_bytes(), || {
                mc_client::net::ping::parse_status(&json, Duration::ZERO)
                    .map(|ping| ping.motd.spans(&lang).len())
            });
            if matches!(parsed, Some(Ok(_))) {
                target.accepted += 1;
            }
        }
        target.finish();
    });
}

/// The layers under the packet decoder: length prefixes and zlib streams. Real packets are
/// compressed and framed as a server sends them (threshold 64), a few frames per stream, then
/// the stream bytes are damaged and read back through `Connection::read_payload`, the same reader
/// the network task uses.
#[test]
fn damaged_frames_and_zlib_streams_never_crash_hang_or_balloon() {
    on_stack(NET_STACK, || {
        use mc_protocol::compress::CompressionCodec;
        use tokio::io::AsyncWriteExt;
        let seeds = load_seeds();
        let payloads: Vec<&Vec<u8>> = seeds
            .play_by_id
            .values()
            .flatten()
            .filter(|payload| payload.len() < 65_536)
            .collect();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime");
        let mut codec = CompressionCodec::with_threshold(64);
        let mut target = Target::new("frames and zlib");
        let mut rng = Rng::new(0x21_0007);
        for round in 0..rounds(6_000) {
            let mut wire = Vec::new();
            for _ in 0..1 + rng.below(3) {
                let mut compressed = bytes::BytesMut::new();
                codec
                    .encode(rng.pick(&payloads), &mut compressed)
                    .expect("compress a real packet");
                wire.extend(varint_bytes(compressed.len() as i32));
                wire.extend_from_slice(&compressed);
            }
            let wire = mutate(&mut rng, &wire, &[]);
            let read = target.step(&format!("round {round}"), &wire, || {
                runtime.block_on(async {
                    let (mut server, client) = tokio::io::duplex(wire.len() + 1);
                    server.write_all(&wire).await.expect("in-memory write");
                    drop(server);
                    let mut connection = mc_protocol::connection::Connection::new(client);
                    connection.set_compression_threshold(64);
                    let mut packets = 0usize;
                    while let Ok(payload) = connection.read_payload().await {
                        packets += 1;
                        let _ = decode(payload.as_slice());
                    }
                    packets
                })
            });
            if read.is_some_and(|packets| packets > 0) {
                target.accepted += 1;
            }
        }
        target.finish();
    });
}

// ---- worst-case shapes at the size limits --------------------------------------------------

/// A frame's whole budget: the 21-bit length prefix allows `2^21 - 1` bytes.
const FRAME_LIMIT: usize = (1 << 21) - 1;

fn payload(id: i32, body: impl FnOnce(&mut PacketBuffer)) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    buf.write_varint(id);
    body(&mut buf);
    buf.as_slice().to_vec()
}

/// A full chunk column of stone at (0, 0), so block updates land in a loaded chunk.
fn stone_column() -> Vec<u8> {
    payload(0x21, |buf| {
        buf.write_i32(0);
        buf.write_i32(0);
        buf.write_bool(true);
        buf.write_u16(0xFFFF);
        let mut data = Vec::with_capacity(196_864);
        for _ in 0..16 * 4096 {
            data.extend_from_slice(&[0x10, 0x00]); // stone, little-endian block state
        }
        data.resize(196_864, 0xFF);
        buf.write_byte_array(&data);
    })
}

/// Chat JSON shapes at the 32767-character limit: deep, wide, escaped, unterminated.
fn huge_chat_shapes() -> Vec<(&'static str, String)> {
    const N: usize = 32_000;
    let nest = |open: &str, inner: &str, close: &str, max: usize| {
        let levels = (max - inner.len()) / (open.len() + close.len());
        format!("{}{inner}{}", open.repeat(levels), close.repeat(levels))
    };
    vec![
        ("open brackets", "[".repeat(N)),
        ("closed brackets", nest("[", "\"x\"", "]", N)),
        (
            "extra tower",
            nest("{\"text\":\"\",\"extra\":[", "\"x\"", "]}", N),
        ),
        (
            "with tower",
            nest("{\"translate\":\"%s\",\"with\":[", "\"x\"", "]}", N),
        ),
        (
            "hover tower",
            nest(
                "{\"text\":\"x\",\"hoverEvent\":{\"action\":\"show_text\",\"value\":",
                "\"x\"",
                "}}",
                N,
            ),
        ),
        (
            "unicode escapes",
            format!("\"{}\"", "\\u0041".repeat(N / 6)),
        ),
        (
            "lone surrogates",
            format!("\"{}\"", "\\ud800".repeat(N / 6)),
        ),
        (
            "section signs",
            format!("{{\"text\":\"{}\"}}", "\u{a7}k".repeat(N / 2)),
        ),
        ("siblings", format!("[\"\"{}]", ",\"a\"".repeat(N / 4))),
        (
            "object siblings",
            format!(
                "{{\"text\":\"\",\"extra\":[{}\"x\"]}}",
                "{\"text\":\"b\"},".repeat(N / 13)
            ),
        ),
        ("unclosed comment", format!("/*{}", "a".repeat(N))),
        ("unquoted keys", "{a:".repeat(N / 3)),
        ("unclosed single quote", format!("'{}", "a".repeat(N))),
        ("long number", "1".repeat(N)),
        (
            "long exponent",
            format!("-{}e{}", "9".repeat(N / 2), "9".repeat(N / 2)),
        ),
        (
            "long score name",
            format!(
                "{{\"score\":{{\"name\":\"{}\",\"objective\":\"o\"}}}}",
                "x".repeat(N)
            ),
        ),
        (
            "implicit arguments",
            format!(
                "{{\"translate\":\"{}\",\"with\":[\"a\"]}}",
                "%s".repeat(N / 2)
            ),
        ),
        (
            "reused arguments",
            format!(
                "{{\"translate\":\"{}\",\"with\":[{}]}}",
                "%1$s".repeat(N / 8),
                doubling_translation(30)
            ),
        ),
        (
            "percent soup",
            format!("{{\"translate\":\"{}\"}}", "%".repeat(N)),
        ),
        (
            "show_item nesting",
            format!(
                "{{\"text\":\"x\",\"hoverEvent\":{{\"action\":\"show_item\",\"value\":\"{}\"}}}}",
                "{a:[".repeat(N / 4)
            ),
        ),
        ("non-execute prefix", format!(")]}}'\n{}", "[".repeat(N))),
        (
            "keys",
            format!(
                "{{{}\"text\":\"x\"}}",
                (0..N / 12)
                    .map(|i| format!("\"k{i:05}\":1,"))
                    .collect::<String>()
            ),
        ),
    ]
}

/// Every chat shape at the packet limit parses (or is refused) and flattens in bounded time and
/// memory: the parse runs on the network thread, where a stall delays keep-alive answers.
#[test]
fn chat_components_at_the_size_limit_stay_bounded() {
    on_stack(MAIN_STACK, || {
        let lang = mc_assets::Language::new("en_US");
        let mut problems = Vec::new();
        let mut slowest = (Duration::ZERO, "");
        for (name, json) in huge_chat_shapes() {
            assert!(
                json.encode_utf16().count() <= 32_767,
                "{name} fits a chat packet"
            );
            let (_, cost) = measured(|| {
                let _ = render_component(&json, &lang);
                let _ = mc_nbt::parse_snbt(&json);
            });
            if cost.elapsed > slowest.0 {
                slowest = (cost.elapsed, name);
            }
            if cost.elapsed > SLOW_LIMIT / 5 || cost.growth > GROWTH_LIMIT / 4 {
                problems.push(format!("{name}: {:?}, heap +{}", cost.elapsed, cost.growth));
            }
        }
        println!("slowest chat shape: {} ({:?})", slowest.1, slowest.0);
        if std::env::var_os("Z1_TRACE_STEPS").is_some() {
            for (name, json) in huge_chat_shapes() {
                let t = Instant::now();
                let parsed = ChatComponent::from_json_nullable(&json);
                let parse = t.elapsed();
                let component = parsed.as_ref().ok().and_then(Option::as_ref);
                let t = Instant::now();
                let spans = component.map(|c| c.spans(&lang).len());
                let flatten = t.elapsed();
                let t = Instant::now();
                let _ = component.map(|c| c.formatted_text(&lang));
                let formatted = t.elapsed();
                let t = Instant::now();
                let _ = component.map(ChatComponent::to_json);
                let written = t.elapsed();
                let t = Instant::now();
                let snbt = mc_nbt::parse_snbt(&json).is_ok();
                let snbt_time = t.elapsed();
                println!(
                    "{name}: parse {parse:?} ({}), spans {flatten:?} ({spans:?}), \
                     formatted {formatted:?}, to_json {written:?}, snbt {snbt_time:?} ({snbt})",
                    if parsed.is_ok() { "ok" } else { "refused" }
                );
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    });
}

/// Packets at the frame limit through decode and the handlers: each must finish in bounded time
/// and memory. Vanilla does the same work for every one of these, so the bound is generous; what
/// it catches is work or memory out of proportion to the bytes.
#[test]
fn packets_at_the_frame_limit_stay_bounded() {
    on_stack(MAIN_STACK, || {
        let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
        // S22 with as many 3-byte records as fit.
        let records = (FRAME_LIMIT - 16) / 3;
        cases.push((
            "multi-block change",
            payload(0x22, |buf| {
                buf.write_i32(0);
                buf.write_i32(0);
                buf.write_varint(records as i32);
                for i in 0..records {
                    buf.write_i16((i % 65_536) as i16);
                    buf.write_varint(if i % 2 == 0 { 0 } else { 0x10 });
                }
            }),
        ));
        // S27 with as many 3-byte records as fit, around a point in the stone column.
        let records = (FRAME_LIMIT - 40) / 3;
        cases.push((
            "explosion",
            payload(0x27, |buf| {
                for v in [8.0f32, 64.0, 8.0, 4.0] {
                    buf.write_f32(v);
                }
                buf.write_i32(records as i32);
                for i in 0..records {
                    buf.write_i8((i % 256) as u8 as i8);
                    buf.write_i8(((i / 256) % 256) as u8 as i8);
                    buf.write_i8(((i / 65_536) % 256) as u8 as i8);
                }
                for v in [0.0f32, 0.0, 0.0] {
                    buf.write_f32(v);
                }
            }),
        ));
        // S26 with as many biome-only columns as fit (10 header bytes + 256 biome bytes each).
        let columns = (FRAME_LIMIT - 8) / 266;
        cases.push((
            "chunk bulk of empty columns",
            payload(0x26, |buf| {
                buf.write_bool(true);
                buf.write_varint(columns as i32);
                for i in 0..columns as i32 {
                    buf.write_i32(i % 89);
                    buf.write_i32(i / 89);
                    buf.write_u16(0);
                }
                buf.write_bytes(&vec![1u8; columns * 256]);
            }),
        ));
        // S38 adding as many players as fit, each with a 16-character name.
        let entries = (FRAME_LIMIT - 16) / 38;
        cases.push((
            "player list",
            payload(0x38, |buf| {
                buf.write_varint(0);
                buf.write_varint(entries as i32);
                for i in 0..entries as u64 {
                    buf.write_u64(i);
                    buf.write_u64(i.wrapping_mul(0x9E37_79B9));
                    buf.write_string(&format!("p{i:015}"));
                    buf.write_varint(0);
                    buf.write_varint(0);
                    buf.write_varint(0);
                    buf.write_bool(false);
                }
            }),
        ));
        // S13 destroying as many entities as fit.
        let ids = (FRAME_LIMIT - 8) / 5;
        cases.push((
            "destroy entities",
            payload(0x13, |buf| {
                buf.write_varint(ids as i32);
                for i in 0..ids as i32 {
                    buf.write_varint(i32::MAX - i);
                }
            }),
        ));
        // S30 with the most slots a window can name, each an enchanted, named item.
        cases.push((
            "window items",
            payload(0x30, |buf| {
                buf.write_u8(0);
                buf.write_i16(i16::MAX);
                let mut tag = mc_nbt::NbtCompound::new();
                let mut display = mc_nbt::NbtCompound::new();
                display.insert("Name", mc_nbt::Nbt::String("x".repeat(32)));
                tag.insert("display", mc_nbt::Nbt::Compound(display));
                let stack = mc_protocol::ItemStack {
                    item_id: 276,
                    count: 1,
                    damage: 0,
                    nbt: Some(tag),
                };
                for _ in 0..i16::MAX {
                    buf.write_item_stack(Some(&stack));
                }
            }),
        ));
        let mut problems = Vec::new();
        let base = decode(&stone_column()).expect("a stone column");
        for (name, bytes) in cases {
            assert!(
                bytes.len() <= FRAME_LIMIT,
                "{name} fits a frame ({} bytes)",
                bytes.len()
            );
            let mut state = ClientState::new();
            let mut events = Vec::new();
            let _ = apply(&mut state, fuzz_join(), &mut events);
            apply_like_session(&mut state, base.clone(), false);
            let (decoded, cost) = measured(|| decode(&bytes));
            let packet = decoded.unwrap_or_else(|| panic!("{name} decodes"));
            let decode_cost = cost;
            let (_, cost) = measured(|| apply_like_session(&mut state, packet, false));
            println!(
                "{name}: {} bytes, decode {:?} (heap +{}), apply {:?} (heap +{})",
                bytes.len(),
                decode_cost.elapsed,
                decode_cost.growth,
                cost.elapsed,
                cost.growth
            );
            for (what, cost) in [("decode", decode_cost), ("apply", cost)] {
                if cost.elapsed > SLOW_LIMIT || cost.growth > GROWTH_LIMIT {
                    problems.push(format!(
                        "{name} {what}: {:?}, heap +{}",
                        cost.elapsed, cost.growth
                    ));
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    });
}

/// Login packets of every id, which the captures do not all have.
fn handmade_login() -> Vec<Vec<u8>> {
    use mc_protocol::login::{Disconnect, EncryptionRequest, SetCompression};
    vec![
        payload(0x00, |buf| {
            Disconnect {
                reason: r#"{"translate":"multiplayer.disconnect.kicked","with":["x"]}"#.into(),
            }
            .write(buf);
        }),
        payload(0x01, |buf| {
            EncryptionRequest {
                server_id: String::new(),
                public_key_der: vec![0x30; 162],
                verify_token: vec![1, 2, 3, 4],
            }
            .write(buf);
        }),
        payload(0x02, |buf| {
            buf.write_string("00000000-0000-0000-0000-000000000001");
            buf.write_string("Player");
        }),
        payload(0x02, |buf| {
            buf.write_string("1-2-3-4-5");
            buf.write_string("P");
        }),
        payload(0x03, |buf| SetCompression { threshold: 256 }.write(buf)),
    ]
}

/// Measurement for the report (not a gate): how long one collision scan takes for the swept box
/// of a huge server-set velocity (an S27 explosion's motion is an unbounded float). Vanilla's
/// `getCollidingBoundingBoxes` walks the same columns, so this is a vanilla-shared freeze; see the
/// Z1 report's recommendation.
#[test]
#[ignore = "a measurement, not a gate"]
fn measure_the_collision_scan_of_a_huge_sweep() {
    let world = mc_world::World::new();
    for span in [1e3, 1e4, 1e5, 1e6] {
        let mask = mc_math::Aabb::new(
            mc_math::DVec3::new(0.2, 64.0, 0.2),
            mc_math::DVec3::new(0.8 + span, 65.8, 0.8),
        );
        let mut out = Vec::new();
        let started = Instant::now();
        mc_physics::collision::collision_boxes_in(&world, &mask, &mut out);
        println!("a {span}-block sweep along x: {:?}", started.elapsed());
    }
}

fn fuzz_join() -> S2CPlay {
    S2CPlay::JoinGame(mc_protocol::play::s2c::JoinGame {
        entity_id: 1_000,
        hardcore_mode: false,
        game_mode: 0,
        dimension: 0,
        difficulty: 2,
        max_players: 20,
        level_type: "default".into(),
        reduced_debug_info: false,
    })
}

// ---- the Java decode oracle ------------------------------------------------------------------

/// What the client's network task does with one clientbound Play payload: `OK` when it reaches
/// the game, otherwise `ERR <why>`, which disconnects (as a vanilla decode exception does).
fn rust_verdict(payload: &[u8]) -> String {
    if payload.is_empty() {
        // `connection.rs` skips an empty frame, as vanilla's decoders do.
        return "OK".to_owned();
    }
    let mut buf = PacketBuffer::from_slice(payload);
    let id = match buf.read_varint() {
        Ok(id) => id,
        Err(error) => return format!("ERR {error:?}"),
    };
    let packet = match S2CPlay::read(id, &mut buf) {
        Ok(packet) => packet,
        Err(error) => {
            let shown = format!("{error:?}");
            let kind = shown
                .split(['(', ' ', '{'])
                .next()
                .unwrap_or_default()
                .to_owned();
            return format!("ERR {kind}");
        }
    };
    if !buf.is_empty() {
        return "ERR TrailingBytes".to_owned();
    }
    for field in components(&packet) {
        if ChatComponent::from_json_nullable(field).is_err() {
            return "ERR ChatJson".to_owned();
        }
    }
    "OK".to_owned()
}

/// The login-state twin of [`rust_verdict`]: what the client's login exchange does with one
/// clientbound Login payload.
fn rust_login_verdict(payload: &[u8]) -> String {
    use mc_protocol::login::{Disconnect, EncryptionRequest, LoginSuccess, SetCompression};
    if payload.is_empty() {
        return "OK".to_owned();
    }
    let mut buf = PacketBuffer::from_slice(payload);
    let Ok(id) = buf.read_varint() else {
        return "ERR id".to_owned();
    };
    let read = match id {
        0x00 => Disconnect::read(&mut buf)
            .map(|packet| ChatComponent::from_json_nullable(&packet.reason).is_ok()),
        0x01 => EncryptionRequest::read(&mut buf).map(|_| true),
        0x02 => LoginSuccess::read(&mut buf).map(|_| true),
        0x03 => SetCompression::read(&mut buf).map(|_| true),
        _ => return "ERR bad-id".to_owned(),
    };
    match read {
        Ok(true) if buf.is_empty() => "OK".to_owned(),
        Ok(true) => "ERR TrailingBytes".to_owned(),
        Ok(false) => "ERR ChatJson".to_owned(),
        Err(error) => format!("ERR {error}"),
    }
}

/// The status twin of [`rust_verdict`]: what the server-list ping does with one clientbound
/// Status payload. An `ERR` is a server the list shows as unreachable.
fn rust_status_verdict(payload: &[u8]) -> String {
    if payload.is_empty() {
        return "OK".to_owned();
    }
    let mut buf = PacketBuffer::from_slice(payload);
    let Ok(id) = buf.read_varint() else {
        return "ERR id".to_owned();
    };
    match id {
        0x00 => match mc_protocol::status::Response::read(&mut buf) {
            Ok(_) if !buf.is_empty() => "ERR TrailingBytes".to_owned(),
            Ok(response) => {
                match mc_client::net::ping::parse_status(&response.json, Duration::ZERO) {
                    Ok(_) => "OK".to_owned(),
                    Err(error) => format!("ERR {error}"),
                }
            }
            Err(error) => format!("ERR {error}"),
        },
        0x01 => match mc_protocol::status::Pong::read(&mut buf) {
            Ok(_) if buf.is_empty() => "OK".to_owned(),
            Ok(_) => "ERR TrailingBytes".to_owned(),
            Err(error) => format!("ERR {error}"),
        },
        _ => "ERR bad-id".to_owned(),
    }
}

/// `Z1_ORACLE_STATE=LOGIN` switches the oracle tests to the login packets.
fn oracle_is_login() -> bool {
    std::env::var("Z1_ORACLE_STATE").is_ok_and(|state| state == "LOGIN")
}

/// `Z1_ORACLE_STATE=STATUS` switches them to the server-list status packets.
fn oracle_is_status() -> bool {
    std::env::var("Z1_ORACLE_STATE").is_ok_and(|state| state == "STATUS")
}

/// The oracle's input: damaged real packets, one hex payload per line, written to
/// `$Z1_ORACLE_DIR/cases.txt` with the client's verdicts in `rust.txt`. Run
/// `javaref/generate-z1-decode.ps1` on the folder, then `compare_with_the_java_decode_oracle`.
#[test]
#[ignore = "writes the Java oracle's input; needs Z1_ORACLE_DIR"]
fn dump_cases_for_the_java_decode_oracle() {
    let Some(dir) = std::env::var_os("Z1_ORACLE_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("oracle folder");
    let mut seeds = load_seeds();
    if oracle_is_login() {
        // The same driver over the login packets, grouped by id like the play ones.
        let mut by_id: BTreeMap<i32, Vec<Vec<u8>>> = BTreeMap::new();
        for payload in seeds.login.iter().chain(&handmade_login()) {
            if let Some((id, _)) = split_id(payload) {
                by_id.entry(id).or_default().push(payload.clone());
            }
        }
        seeds.play_by_id = by_id;
    }
    let verdict = if oracle_is_login() {
        rust_login_verdict
    } else {
        rust_verdict
    };
    if oracle_is_status() {
        // Status JSON with the same structure-aware damage as chat, as S00 status responses.
        let mut corpus: Vec<Vec<u8>> = seeds
            .status
            .iter()
            .filter_map(|payload| {
                let (id, body) = split_id(payload)?;
                let response =
                    mc_protocol::status::Response::read(&mut PacketBuffer::from_slice(body))
                        .ok()?;
                (id == 0x00).then(|| response.json.into_bytes())
            })
            .collect();
        corpus.extend(HANDMADE_STATUS.iter().map(|json| json.as_bytes().to_vec()));
        let tokens: Vec<&str> = JSON_TOKENS
            .iter()
            .chain(LENIENT_TOKENS.iter())
            .copied()
            .collect();
        let mut rng = Rng::new(0x21_0009);
        let mut cases = String::new();
        let mut verdicts = String::new();
        let mut written = HashSet::new();
        let total = rounds(20_000);
        while written.len() < total {
            let mut json = rng.pick(&corpus).clone();
            if rng.one_in(3) {
                json = mutate(&mut rng, &json, &corpus);
            }
            for _ in 0..rng.below(3) {
                let token = rng.pick(&tokens);
                let at = rng.below(json.len() + 1);
                json.splice(at..at, token.bytes());
            }
            let text = String::from_utf8_lossy(&json).into_owned();
            if text.encode_utf16().count() > 32_767 {
                continue;
            }
            let input = payload(0x00, |buf| buf.write_string(&text));
            if !written.insert(input.clone()) {
                continue;
            }
            cases.push_str(&input.iter().map(|b| format!("{b:02x}")).collect::<String>());
            cases.push('\n');
            verdicts.push_str(&rust_status_verdict(&input));
            verdicts.push('\n');
        }
        std::fs::write(dir.join("cases.txt"), cases).expect("write cases");
        std::fs::write(dir.join("rust.txt"), verdicts).expect("write verdicts");
        println!("{} status cases in {}", written.len(), dir.display());
        return;
    }
    if std::env::var("Z1_ORACLE_STATE").is_ok_and(|state| state == "CHAT") {
        // Chat JSON with structure-aware damage, as S02 packets: Gson's lenient grammar is where
        // a stricter parser would disconnect players. The Java side runs with state PLAY.
        let mut corpus: Vec<Vec<u8>> = Vec::new();
        for payload in seeds.play_by_id.values().flatten() {
            if let Some(packet) = decode(payload) {
                corpus.extend(components(&packet).iter().map(|c| c.as_bytes().to_vec()));
            }
        }
        corpus.extend(HANDMADE_JSON.iter().map(|json| json.as_bytes().to_vec()));
        corpus.extend(LENIENT_JSON.iter().map(|json| json.as_bytes().to_vec()));
        let tokens: Vec<&str> = JSON_TOKENS
            .iter()
            .chain(LENIENT_TOKENS.iter())
            .copied()
            .collect();
        let mut rng = Rng::new(0x21_0008);
        let mut cases = String::new();
        let mut verdicts = String::new();
        let mut written = HashSet::new();
        let total = rounds(40_000);
        while written.len() < total {
            let mut json = rng.pick(&corpus).clone();
            if rng.one_in(3) {
                json = mutate(&mut rng, &json, &corpus);
            }
            for _ in 0..rng.below(3) {
                let token = rng.pick(&tokens);
                let at = rng.below(json.len() + 1);
                json.splice(at..at, token.bytes());
            }
            let text = String::from_utf8_lossy(&json).into_owned();
            if text.encode_utf16().count() > 32_767 {
                continue;
            }
            let input = payload(0x02, |buf| {
                buf.write_string(&text);
                buf.write_i8(0);
            });
            if !written.insert(input.clone()) {
                continue;
            }
            cases.push_str(&input.iter().map(|b| format!("{b:02x}")).collect::<String>());
            cases.push('\n');
            verdicts.push_str(&rust_verdict(&input));
            verdicts.push('\n');
        }
        std::fs::write(dir.join("cases.txt"), cases).expect("write cases");
        std::fs::write(dir.join("rust.txt"), verdicts).expect("write verdicts");
        println!("{} chat cases in {}", written.len(), dir.display());
        return;
    }
    let ids: Vec<i32> = seeds.play_by_id.keys().copied().collect();
    let mut rng = Rng::new(0x21_0006);
    let mut cases = String::new();
    let mut verdicts = String::new();
    let mut written = HashSet::new();
    let total = rounds(40_000);
    let mut round = 0usize;
    while written.len() < total && round < total * 4 {
        let id = ids[round % ids.len()];
        round += 1;
        let donors = &seeds.play_by_id[&id];
        let seed = rng.pick(donors);
        // Unmutated seeds are cases too: both decoders must accept every real packet.
        let input = if rng.one_in(10) {
            seed.clone()
        } else {
            mutate(&mut rng, seed, donors)
        };
        // Chunk-sized payloads make the file huge and are covered by the fuzz targets.
        if input.len() > 4096 || !written.insert(input.clone()) {
            continue;
        }
        cases.push_str(&input.iter().map(|b| format!("{b:02x}")).collect::<String>());
        cases.push('\n');
        verdicts.push_str(&verdict(&input));
        verdicts.push('\n');
    }
    std::fs::write(dir.join("cases.txt"), cases).expect("write cases");
    std::fs::write(dir.join("rust.txt"), verdicts).expect("write verdicts");
    println!("{} cases in {}", written.len(), dir.display());
}

/// Every damaged payload the real 1.8.9 decoder accepts, as recorded by the Java oracle in
/// `tests/fixtures/z1_vanilla_accepts.txt`, reaches the game here too. Refusing one would be a
/// disconnect vanilla does not have (last night's lone surrogate in an item name was one).
#[test]
fn every_payload_vanilla_accepts_reaches_the_game() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("z1_vanilla_accepts.txt");
    let text = std::fs::read_to_string(&path).expect("the oracle fixture");
    let seeds = load_seeds();
    let originals: HashSet<&Vec<u8>> = seeds
        .streams
        .iter()
        .flat_map(|stream| &stream.play)
        .chain(&seeds.login)
        .collect();
    let (mut total, mut damaged, mut replaced) = (0usize, 0usize, 0usize);
    let mut refused = Vec::new();
    for line in text
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let (state, hex) = line.split_once(' ').expect("a state and a payload");
        let payload: Vec<u8> = (0..hex.len() / 2)
            .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex"))
            .collect();
        // Negative control: cases whose strings only decode through Java's U+FFFD replacement.
        // A decoder that refused malformed text (the class of last night's surrogate bug) fails
        // on each of them.
        if state == "PLAY"
            && decode(&payload).is_some_and(|p| format!("{p:?}").contains('\u{fffd}'))
        {
            replaced += 1;
        }
        let verdict = match state {
            "PLAY" => rust_verdict(&payload),
            "LOGIN" => rust_login_verdict(&payload),
            other => panic!("unknown state {other}"),
        };
        total += 1;
        damaged += usize::from(!originals.contains(&payload));
        if !verdict.starts_with("OK") {
            refused.push(format!("{state} {hex}: {verdict}"));
        }
    }
    // 1000 damaged packets, 300 damaged chat components (61 of them refused before
    // `<<W19-Z1:gson-null-prefix>>`), 200 damaged login packets.
    assert_eq!(total, 1_500, "the fixture lost lines");
    // Most cases are damaged copies, not the captured packets themselves.
    assert!(damaged > 900, "only {damaged} damaged cases");
    assert!(
        replaced > 0,
        "no case exercises Java's lenient string decoding"
    );
    println!("{total} cases, {damaged} damaged, {replaced} with replaced malformed text");
    assert!(
        refused.is_empty(),
        "{} refused:\n{}",
        refused.len(),
        refused.join("\n")
    );
}

/// Compare `rust.txt` with the oracle's `java.txt` in `$Z1_ORACLE_DIR`: a payload vanilla
/// accepts and the client refuses is a disconnect vanilla would not have.
#[test]
#[ignore = "reads the Java oracle's output; needs Z1_ORACLE_DIR"]
fn compare_with_the_java_decode_oracle() {
    let Some(dir) = std::env::var_os("Z1_ORACLE_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let read = |name: &str| {
        std::fs::read_to_string(dir.join(name))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let (cases, java) = (read("cases.txt"), read("java.txt"));
    assert_eq!(cases.len(), java.len(), "one Java verdict per case");
    let mut only_rust_refuses: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut only_java_refuses: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut agree = 0usize;
    for (case, java) in cases.iter().zip(&java) {
        let payload: Vec<u8> = (0..case.len() / 2)
            .map(|i| u8::from_str_radix(&case[2 * i..2 * i + 2], 16).expect("hex"))
            .collect();
        let rust = if oracle_is_login() {
            rust_login_verdict(&payload)
        } else if oracle_is_status() {
            rust_status_verdict(&payload)
        } else {
            rust_verdict(&payload)
        };
        let id = split_id(&payload).map_or(-1, |(id, _)| id);
        match (rust.starts_with("OK"), java.starts_with("OK")) {
            (true, true) | (false, false) => agree += 1,
            (false, true) => only_rust_refuses
                .entry(format!("id {id:#04x} rust {rust}"))
                .or_default()
                .push(case.clone()),
            (true, false) => only_java_refuses
                .entry(format!("id {id:#04x} java {java}"))
                .or_default()
                .push(case.clone()),
        }
    }
    println!("{agree} of {} agree", cases.len());
    for (title, map) in [
        (
            "refused by the client only (a disconnect vanilla would not have)",
            &only_rust_refuses,
        ),
        ("refused by vanilla only", &only_java_refuses),
    ] {
        println!("== {title}: {} kinds", map.len());
        for (kind, examples) in map {
            println!("  {kind}: {} cases, e.g. {}", examples.len(), examples[0]);
        }
    }
    assert!(
        only_rust_refuses.is_empty(),
        "the client refuses payloads vanilla accepts"
    );
}

// ---- regressions for what the targets found --------------------------------------------------

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 0 {
                crc >> 1
            } else {
                (crc >> 1) ^ 0xEDB8_8320
            };
        }
    }
    !crc
}

/// A real 1x1 PNG whose header then claims `width` x `height`.
fn png_claiming(width: u32, height: u32) -> Vec<u8> {
    let mut png = Vec::new();
    image::RgbaImage::new(1, 1)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode a 1x1 PNG");
    // The 8-byte signature, then IHDR: length, "IHDR", width, height, 5 more bytes, CRC.
    assert_eq!(&png[12..16], b"IHDR");
    png[16..20].copy_from_slice(&width.to_be_bytes());
    png[20..24].copy_from_slice(&height.to_be_bytes());
    let crc = crc32(&png[12..29]);
    png[29..33].copy_from_slice(&crc.to_be_bytes());
    png
}

fn favicon_uri(png: &[u8]) -> String {
    use base64::Engine as _;
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    )
}

/// Every server in the list is pinged when the multiplayer screen
/// opens. A 90-byte icon claiming 11000x11000 made the ping allocate that whole surface (484 MB)
/// before the 64x64 check dropped it; the decoder now refuses it from the header.
#[test]
fn a_server_icon_cannot_make_the_ping_allocate_its_claimed_size() {
    let uri = favicon_uri(&png_claiming(11_000, 11_000));
    let (icon, cost) = measured(|| mc_client::net::ping::parse_favicon(&uri));
    assert!(icon.is_err(), "an 11000x11000 icon is not 64x64");
    assert!(
        cost.largest < 1 << 20,
        "a {}-byte icon allocated {} bytes at once",
        uri.len(),
        cost.largest
    );
    // The same path through a whole status response keeps the MOTD and drops the icon.
    let json = format!(r#"{{"description":"motd","favicon":"{uri}"}}"#);
    let (ping, cost) = measured(|| mc_client::net::ping::parse_status(&json, Duration::ZERO));
    let ping = ping.expect("a bad icon does not fail the ping");
    assert!(ping.favicon.is_none() && ping.favicon_present);
    assert!(
        cost.largest < 1 << 20,
        "allocated {} bytes at once",
        cost.largest
    );

    // Negative control (opt-in, it commits ~480 MB): the plain decode the ping used to do.
    if std::env::var_os("Z1_NEGATIVE_CONTROLS").is_some() {
        let png = png_claiming(11_000, 11_000);
        let (_, cost) =
            measured(|| image::load_from_memory_with_format(&png, image::ImageFormat::Png));
        assert!(
            cost.largest > 400 << 20,
            "the old decode allocated {}",
            cost.largest
        );
    }

    // Control: a genuine 64x64 icon still decodes.
    let mut real = Vec::new();
    image::RgbaImage::from_pixel(64, 64, image::Rgba([1, 2, 3, 255]))
        .write_to(
            &mut std::io::Cursor::new(&mut real),
            image::ImageFormat::Png,
        )
        .expect("encode a 64x64 PNG");
    let icon = mc_client::net::ping::parse_favicon(&favicon_uri(&real)).expect("a real icon");
    assert_eq!((icon.width, icon.height), (64, 64));
    assert_eq!(&icon.rgba[..4], &[1, 2, 3, 255]);
}

/// One S2A with a count of `i32::MAX` was a loop of 2^31 spawn
/// attempts (vanilla's own freeze). The burst now stops at `MAX_BURST_PARTICLES`.
#[test]
fn a_particle_packet_count_cannot_freeze_the_client() {
    use mc_client::app::particles::{to_render_packet, AtlasIcons, MAX_BURST_PARTICLES};
    let packet = |count: i32| mc_protocol::play::s2c::Particles {
        particle_id: 0,
        long_distance: true,
        x: 0.0,
        y: 64.0,
        z: 0.0,
        offset_x: 1.0,
        offset_y: 1.0,
        offset_z: 1.0,
        particle_speed: 0.1,
        particle_count: count,
        extra_data: Vec::new(),
    };
    // Every count a server plausibly sends is untouched; negative ones spawn nothing, as before.
    for count in [i32::MIN, -1, 0, 1, 4000, MAX_BURST_PARTICLES] {
        assert_eq!(to_render_packet(&packet(count)).count, count);
    }
    let hostile = to_render_packet(&packet(i32::MAX));
    assert_eq!(hostile.count, MAX_BURST_PARTICLES);
    let world = mc_world::World::new();
    let mut system = mc_render::particle::ParticleSystem::new();
    let (_, cost) = measured(|| {
        mc_render::particle::packet::apply(
            &mut system,
            &world,
            mc_math::DVec3::new(0.0, 64.0, 0.0),
            &hostile,
            &AtlasIcons,
        );
    });
    assert!(
        cost.elapsed < SLOW_LIMIT,
        "one packet took {:?}",
        cost.elapsed
    );
    assert!(
        cost.growth < GROWTH_LIMIT,
        "one packet grew the heap by {}",
        cost.growth
    );
    // The layer is full, exactly as the uncapped burst would leave it.
    assert!(system.len() >= 4000, "{} live particles", system.len());
    println!(
        "a capped burst took {:?}; the uncapped one is {}x longer",
        cost.elapsed,
        i32::MAX / MAX_BURST_PARTICLES
    );
}

/// An S2B rain level of 1e18 made every tick sample `i32::MAX` columns
/// and collect an effect for most of them. The pass now samples at most `MAX_RAIN_SAMPLES`.
#[test]
fn a_rain_level_cannot_freeze_the_client() {
    use mc_render::sky::weather::{add_rain_particles, MAX_RAIN_SAMPLES};
    let world = mc_world::World::new();
    let camera = mc_math::DVec3::new(0.5, 1.0, 0.5);
    let (pass, cost) = measured(|| add_rain_particles(&world, camera, 7, true, 0, 1e18, 0));
    assert!(pass.effects.len() <= MAX_RAIN_SAMPLES as usize);
    assert!(
        cost.elapsed < SLOW_LIMIT,
        "one tick took {:?}",
        cost.elapsed
    );
    assert!(
        cost.growth < GROWTH_LIMIT,
        "one tick grew the heap by {}",
        cost.growth
    );
    // Real levels are untouched: the cap is far above the 100 samples of full rain.
    const { assert!(MAX_RAIN_SAMPLES > 100 * 18 * 18) };
    let full = add_rain_particles(&world, camera, 7, true, 0, 1.0, 0);
    assert!(full.effects.len() <= 100);
}

/// Status responses that vanilla lists (its Gson decoder accepts each
/// one) made the server list show the server as unreachable: the client parsed strict JSON,
/// refused a JSON-null `version` or `players`, and capped sample ids at 36 characters, which Java
/// 8's `UUID.fromString` does not.
#[test]
fn status_responses_vanilla_accepts_list_the_server() {
    for json in [
        "{\"description\":\"line one\nline two\"}",
        "{\"description\":\"tab\there\",\"players\":{\"max\":1,\"online\":0}}",
        "{description:lenient,players:{max:1,online:1},version:{name:v,protocol:47}}",
        "{'description':'single quoted'}",
        "/* a comment */ {\"description\":\"x\"}",
        "{\"description\":\"x\";\"players\":{\"max\":1;\"online\":1}}",
        "{\"version\":null,\"players\":null,\"description\":null}",
        "{\"players\":{\"max\":2,\"online\":1,\"sample\":[{\"name\":\"x\",\"id\":\"0000000000000001-0000000000000002-3-4-5\"}]}}",
    ] {
        let ping = mc_client::net::ping::parse_status(json, Duration::ZERO);
        assert!(ping.is_ok(), "{json:?}: {:?}", ping.err());
    }
    let ping =
        mc_client::net::ping::parse_status("{\"version\":null,\"players\":null}", Duration::ZERO)
            .expect("null members are absent members");
    assert_eq!((ping.version_name.as_str(), ping.protocol), ("Old", 0));
    assert_eq!((ping.online_players, ping.max_players), (None, None));
    // Negative controls: what vanilla's decoder or its status handler refuses stays refused.
    for json in [
        "",
        "null",
        "[]",
        "{\"favicon\":null}",
        "{\"players\":{\"max\":\"x\",\"online\":1}}",
        "{\"players\":{\"max\":1,\"online\":1,\"sample\":[{\"name\":\"x\",\"id\":\"1-2-3-4\"}]}}",
    ] {
        assert!(
            mc_client::net::ping::parse_status(json, Duration::ZERO).is_err(),
            "{json:?}"
        );
    }
}

/// An S48 `level://` URL could name any file on the PC (`../`, or an
/// absolute path), and the ACCEPTED or FAILED_DOWNLOAD reply told the server whether it existed.
/// Only plain paths inside `saves` resolve now; everything else answers like a missing file.
#[test]
fn a_level_resource_pack_url_cannot_probe_files_outside_saves() {
    use mc_client::app::assets::level_resource_pack_path;
    let dir = tempfile::tempdir().expect("tempdir");
    let config = mc_client::ClientConfig {
        game_dir: dir.path().to_path_buf(),
        ..mc_client::ClientConfig::default()
    };
    std::fs::write(dir.path().join("secret.txt"), b"x").expect("a file outside saves");
    std::fs::create_dir_all(dir.path().join("saves").join("world")).expect("a world");
    std::fs::write(
        dir.path().join("saves").join("world").join("resources.zip"),
        b"PK",
    )
    .expect("a world pack");
    // What an integrated server sends still resolves.
    assert!(
        level_resource_pack_path(&config, "level://world/resources.zip")
            .is_some_and(|path| path.is_file())
    );
    let absolute = format!("level://{}", dir.path().join("secret.txt").display());
    for url in [
        "level://../secret.txt",
        "level://world/../../secret.txt",
        "level://./../secret.txt",
        absolute.as_str(),
    ] {
        assert!(level_resource_pack_path(&config, url).is_none(), "{url}");
    }
    // Negative control: the same `../` joined the old way does reach the file.
    assert!(dir.path().join("saves").join("../secret.txt").is_file());
}

/// `levels` translations of a missing key that names its argument twice, around one letter:
/// unbudgeted, `2^levels` spans.
fn doubling_translation(levels: usize) -> String {
    let mut json = r#"{"text":"a"}"#.to_owned();
    for _ in 0..levels {
        json = format!(r#"{{"translate":"%1$s%1$s","with":[{json}]}}"#);
    }
    json
}

/// A server-list MOTD, a chat line, a title or a sign whose
/// translation arguments are reused at every level used to flatten to 2^40 spans, which hangs
/// the client (and opening the multiplayer screen is enough to receive a MOTD). Flattening now
/// stops at `MAX_FLATTEN_WORK`.
#[test]
fn reused_translation_arguments_cannot_hang_the_motd_chat_or_signs() {
    on_stack(MAIN_STACK, || {
        let lang = mc_assets::Language::new("en_US");
        let tower = doubling_translation(40);
        let json = format!(r#"{{"description":{tower}}}"#);
        let (spans, cost) = measured(|| {
            let ping = mc_client::net::ping::parse_status(&json, Duration::ZERO).expect("valid");
            ping.motd.spans(&lang).len()
        });
        assert!(spans <= mc_ui::font::component::MAX_FLATTEN_WORK);
        assert!(
            cost.elapsed < SLOW_LIMIT,
            "the MOTD took {:?}",
            cost.elapsed
        );
        assert!(
            cost.growth < GROWTH_LIMIT,
            "the MOTD grew the heap by {}",
            cost.growth
        );
        // The same tower as a chat packet: decoded, parsed and flattened as the client does.
        let mut buf = PacketBuffer::new();
        buf.write_varint(0x02);
        mc_protocol::play::s2c::Chat {
            component: tower,
            position: 0,
        }
        .write(&mut buf);
        let packet = decode(buf.as_slice()).expect("a chat packet under 32767 characters");
        let (_, cost) = measured(|| {
            for field in components(&packet) {
                assert!(render_component(field, &lang).is_ok());
            }
            let mut state = ClientState::new();
            apply_like_session(&mut state, packet.clone(), false);
        });
        assert!(
            cost.elapsed < SLOW_LIMIT,
            "the chat line took {:?}",
            cost.elapsed
        );
        assert!(
            cost.growth < GROWTH_LIMIT,
            "the chat line grew the heap by {}",
            cost.growth
        );
    });
}
