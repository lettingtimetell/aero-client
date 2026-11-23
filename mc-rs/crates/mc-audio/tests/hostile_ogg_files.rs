//! Sound files come from resource packs: players share them and any server can offer one. These
//! tests drive the Ogg Vorbis entry points with hand-built streams (a bit-exact generator below,
//! so nothing here needs Minecraft's files) and with seeded random damage to them.
//!
//! The fuzz rounds are bounded (a few seconds). For a longer run by hand:
//!
//! ```text
//! OGG_FUZZ_ROUNDS=20 OGG_FUZZ_SEED=12345 cargo test -p mc-audio --test hostile_ogg_files -- --nocapture
//! ```
//!
//! With `MC_JARS_DIR` set, the fuzzer also starts from a few vanilla sounds, and the ignored test
//! `every_vanilla_sound_passes_the_stream_limits` checks every vanilla sound against the limits.
//!
//! This binary counts the bytes live on each thread and refuses any single allocation over 1 GiB,
//! so a regression fails here instead of committing gigabytes on the machine running the suite.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::panic::AssertUnwindSafe;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use kira::sound::streaming::Decoder as _;
use mc_audio::ogg::{
    check_stream_limits, count_frames_within, decode_all, decode_all_within, frames_from_last_page,
    identification, VorbisDecoder, MAX_DECODED_SAMPLES,
};

// ---------------------------------------------------------------------------------------------
// Measuring.

struct Counting;

const REFUSED_ALLOCATION: usize = 1 << 30;

thread_local! {
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

fn grew(bytes: usize) {
    let _ = LIVE.try_with(|live| {
        live.set(live.get() + bytes as i64);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(live.get())));
    });
}

fn shrank(bytes: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - bytes as i64));
}

// SAFETY: forwards every call to the system allocator unchanged, except that a request over
// `REFUSED_ALLOCATION` gets a null pointer, which every caller must already handle (the standard
// library turns it into an allocation-failure abort).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(layout.size());
        // SAFETY: the caller's contract is passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(layout.size());
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(new_size);
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const MIB: i64 = 1 << 20;

fn peak_during<T>(run: impl FnOnce() -> T) -> (T, i64) {
    let before = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(before));
    let out = run();
    (out, PEAK.with(Cell::get) - before)
}

// ---------------------------------------------------------------------------------------------
// A bit-exact Vorbis stream generator: Vorbis I headers packed least significant bit first, in
// Ogg pages with their CRC-32 (polynomial 0x04C11DB7, no reflection).

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    bit: usize,
}

impl Bits {
    fn put(&mut self, value: u64, count: u32) -> &mut Self {
        for index in 0..count {
            if self.bit.is_multiple_of(8) {
                self.bytes.push(0);
            }
            if (value >> index) & 1 == 1 {
                *self.bytes.last_mut().expect("pushed above") |= 1 << (self.bit % 8);
            }
            self.bit += 1;
        }
        self
    }
}

const SHORT_BLOCK: u8 = 8; // 256 samples
const LONG_BLOCK: u8 = 13; // 8192 samples, the largest Vorbis allows

fn ident_header(channels: u8, rate: u32) -> Vec<u8> {
    let mut packet = vec![1];
    packet.extend_from_slice(b"vorbis");
    packet.extend_from_slice(&0u32.to_le_bytes());
    packet.push(channels);
    packet.extend_from_slice(&rate.to_le_bytes());
    for _ in 0..3 {
        packet.extend_from_slice(&0i32.to_le_bytes());
    }
    packet.push(SHORT_BLOCK | (LONG_BLOCK << 4));
    packet.push(1);
    packet
}

/// A comment header whose length fields are `vendor_field`, `count_field` and, for each
/// comment, `comment_fields[i]`, around the actual bytes given.
fn comment_header(vendor_field: u32, count_field: u32, comment_fields: &[u32]) -> Vec<u8> {
    let mut packet = vec![3];
    packet.extend_from_slice(b"vorbis");
    packet.extend_from_slice(&vendor_field.to_le_bytes());
    if vendor_field < 64 {
        packet.extend(std::iter::repeat_n(b'v', vendor_field as usize));
    }
    packet.extend_from_slice(&count_field.to_le_bytes());
    for &field in comment_fields {
        packet.extend_from_slice(&field.to_le_bytes());
        if field < 64 {
            packet.extend(std::iter::repeat_n(b'c', field as usize));
        }
    }
    packet.push(1);
    packet
}

fn plain_comment() -> Vec<u8> {
    comment_header(4, 1, &[7])
}

/// One codebook: two entries of one-bit codewords, one dimension, no lookup table.
fn plain_codebook(bits: &mut Bits) {
    bits.put(0x56_4342, 24).put(1, 16).put(2, 24); // sync, dimensions, entries
    bits.put(0, 1).put(0, 1).put(0, 5).put(0, 5); // not ordered, not sparse, lengths 1 and 1
    bits.put(0, 4); // no lookup table
}

/// A setup header: `codebooks` codebooks written by `write_codebooks`, then one empty floor,
/// residue and mapping, a short mode and a long mode.
fn setup_header(codebooks: u64, write_codebooks: impl Fn(&mut Bits)) -> Vec<u8> {
    let mut bits = Bits::default();
    bits.put(codebooks - 1, 8);
    write_codebooks(&mut bits);
    bits.put(0, 6).put(0, 16); // one time-domain transform
    bits.put(0, 6).put(1, 16); // one floor, type 1
    bits.put(0, 5).put(0, 2).put(7, 4); // no partitions, multiplier 1, range bits 7
    bits.put(0, 6).put(0, 16); // one residue, type 0
    bits.put(0, 24).put(0, 24).put(0, 24); // begin, end, partition size 1
    bits.put(0, 6).put(0, 8).put(0, 3).put(0, 1); // one classification, classbook 0, cascade 0
    bits.put(0, 6).put(0, 16); // one mapping, type 0
    bits.put(0, 1).put(0, 1).put(0, 2); // one submap, no coupling, reserved
    bits.put(0, 8).put(0, 8).put(0, 8); // submap: time, floor 0, residue 0
    bits.put(1, 6); // two modes
    bits.put(0, 1).put(0, 16).put(0, 16).put(0, 8); // mode 0: short block, mapping 0
    bits.put(1, 1).put(0, 16).put(0, 16).put(0, 8); // mode 1: long block, mapping 0
    bits.put(1, 1); // framing
    let mut packet = vec![5];
    packet.extend_from_slice(b"vorbis");
    packet.extend_from_slice(&bits.bytes);
    packet
}

fn plain_setup() -> Vec<u8> {
    setup_header(1, plain_codebook)
}

/// A long-block audio packet whose floors are all unused: one byte that the decoder turns into
/// 4096 samples per channel (after the first packet, which only primes the overlap).
const LONG_SILENCE: u8 = 0b0000_1110;
/// The same in a short block: 128 samples per channel.
const SHORT_SILENCE: u8 = 0b0000_0000;
const FRAMES_PER_LONG_PACKET: usize = 1 << (LONG_BLOCK - 1);

fn ogg_crc(page: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &byte in page {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

const SERIAL: u32 = 0x5EED;

fn page(flags: u8, granule: u64, serial: u32, sequence: u32, packets: &[&[u8]]) -> Vec<u8> {
    let mut lacing = Vec::new();
    for packet in packets {
        lacing.extend(std::iter::repeat_n(255u8, packet.len() / 255));
        lacing.push((packet.len() % 255) as u8);
    }
    assert!(lacing.len() <= 255, "too many segments for one page");
    let mut out = b"OggS".to_vec();
    out.push(0);
    out.push(flags);
    out.extend_from_slice(&granule.to_le_bytes());
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&sequence.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.push(lacing.len() as u8);
    out.extend_from_slice(&lacing);
    for packet in packets {
        out.extend_from_slice(packet);
    }
    let crc = ogg_crc(&out);
    out[22..26].copy_from_slice(&crc.to_le_bytes());
    out
}

/// A whole stream: the three headers, then `pages` pages of `per_page` copies of `packet`, the
/// last page flagged end-of-stream with the true granule position.
fn stream(
    channels: u8,
    comment: &[u8],
    setup: &[u8],
    pages: usize,
    per_page: usize,
    packet: u8,
) -> Vec<u8> {
    stream_with_serial(SERIAL, channels, comment, setup, pages, per_page, packet)
}

fn stream_with_serial(
    serial: u32,
    channels: u8,
    comment: &[u8],
    setup: &[u8],
    pages: usize,
    per_page: usize,
    packet: u8,
) -> Vec<u8> {
    let mut out = page(0x02, 0, serial, 0, &[&ident_header(channels, 44_100)]);
    out.extend(page(0, 0, serial, 1, &[comment, setup]));
    let packet = [packet];
    let mut packets = 0usize;
    for index in 0..pages {
        packets += per_page;
        let frames = (packets.saturating_sub(1) * FRAMES_PER_LONG_PACKET) as u64;
        let flags = if index + 1 == pages { 0x04 } else { 0 };
        let list: Vec<&[u8]> = vec![&packet[..]; per_page];
        out.extend(page(flags, frames, serial, 2 + index as u32, &list));
    }
    out
}

fn ordinary_stream() -> Vec<u8> {
    stream(2, &plain_comment(), &plain_setup(), 1, 9, LONG_SILENCE)
}

/// Rewrites every page's CRC, as far as the pages parse, so a damaged packet reaches the
/// decoder instead of failing the page checksum.
fn fix_crcs(bytes: &mut [u8]) {
    let mut at = 0;
    while at + 27 <= bytes.len() && &bytes[at..at + 4] == b"OggS" {
        let segments = usize::from(bytes[at + 26]);
        let Some(lacing) = bytes.get(at + 27..at + 27 + segments) else {
            return;
        };
        let end = at + 27 + segments + lacing.iter().map(|&l| usize::from(l)).sum::<usize>();
        if end > bytes.len() {
            return;
        }
        bytes[at + 22..at + 26].copy_from_slice(&[0; 4]);
        let crc = ogg_crc(&bytes[at..end]);
        bytes[at + 22..at + 26].copy_from_slice(&crc.to_le_bytes());
        at = end;
    }
}

// ---------------------------------------------------------------------------------------------
// The generator is real Vorbis.

#[test]
fn the_generated_stream_is_ordinary_vorbis() {
    let bytes = ordinary_stream();
    check_stream_limits(&bytes).expect("within the limits");
    assert_eq!(identification(&bytes), Some((2, 44_100)));
    let decoded = decode_all(&bytes, "test:ordinary").expect("decodes");
    assert_eq!(decoded.channels, 2);
    assert_eq!(decoded.samples.len(), 8 * FRAMES_PER_LONG_PACKET * 2);
    assert!(decoded.samples.iter().all(|&sample| sample == 0), "silence");
    // Far more frames per byte than real audio has: the last page is not trusted, so a play of
    // such a file goes through the counting decode.
    assert_eq!(frames_from_last_page(&bytes), None);
    let shared: Arc<[u8]> = Arc::from(bytes);
    assert_eq!(
        count_frames_within(&shared, "test:ordinary", usize::MAX).expect("counts"),
        8 * FRAMES_PER_LONG_PACKET
    );
}

// ---------------------------------------------------------------------------------------------
// Regressions.

/// A file of a few kilobytes can decode to hours: each one-byte packet
/// stands for 4096 samples per channel. `decode_all` kept every sample (16 pages, 9 KB: 33
/// million samples, 64 MB); it now stops at MAX_DECODED_SAMPLES, and the frame count stops at its
/// limit.
#[test]
fn a_stream_that_decodes_to_hours_is_refused_with_bounded_memory() {
    let bomb = stream(2, &plain_comment(), &plain_setup(), 16, 255, LONG_SILENCE);
    eprintln!("bomb: {} bytes", bomb.len());
    assert!(bomb.len() < 10_000);
    check_stream_limits(&bomb).expect("its headers are ordinary");

    // Negative control: the decoder itself (no limit) really produces that much audio.
    let mut raw = lewton::inside_ogg::OggStreamReader::new(std::io::Cursor::new(&bomb))
        .expect("the raw decoder opens it");
    let mut total = 0usize;
    while let Some(packet) = raw.read_dec_packet_itl().expect("decodes") {
        total += packet.len();
        if total > MAX_DECODED_SAMPLES {
            break;
        }
    }
    eprintln!(
        "raw decoder: over {total} samples from {} bytes",
        bomb.len()
    );
    assert!(total > MAX_DECODED_SAMPLES);

    let began = Instant::now();
    let (result, peak) = peak_during(|| decode_all(&bomb, "test:bomb"));
    eprintln!(
        "decode_all: {:?} in {:?}, peak {} MiB",
        result.as_ref().err(),
        began.elapsed(),
        peak / MIB
    );
    assert!(result.is_err(), "refused");
    assert!(peak < 128 * MIB, "held {peak} bytes");

    let shared: Arc<[u8]> = Arc::from(bomb);
    let counted = count_frames_within(&shared, "test:bomb", 1_000_000);
    assert!(counted.is_err(), "the count stops at its limit");
}

/// Hostile header fields, each of which made the decoder reserve gigabytes to terabytes before
/// reading the data they describe (see `hostile_headers_make_the_raw_decoder_request_gigabytes`).
fn hostile_stream(case: &str) -> Vec<u8> {
    let (comment, setup) = match case {
        // A vendor string of 4 GB: `vec![0; length]` before reading it.
        "vendor" => (comment_header(0xFFFF_FFF0, 0, &[]), plain_setup()),
        // Four billion comments: `Vec::with_capacity(count)` of strings.
        "count" => (comment_header(0, 0xFFFF_FFFF, &[]), plain_setup()),
        // One comment of 4 GB.
        "comment" => (comment_header(0, 1, &[0xFFFF_FFF0]), plain_setup()),
        // 2^24 - 1 entries of 65535 dimensions with a lookup table: 2^40 floats.
        "codebook" => (
            plain_comment(),
            setup_header(1, |bits| {
                bits.put(0x56_4342, 24).put(0xFFFF, 16).put(0xFF_FFFF, 24);
                bits.put(1, 1).put(0, 5).put(0xFF_FFFF, 24); // ordered: every entry one length
                bits.put(1, 4).put(0, 32).put(0, 32).put(0, 4).put(0, 1); // lookup type 1
                bits.put(0, 1); // its one multiplicand
            }),
        ),
        // A lookup table on a codebook of no dimensions: 2^32 multiplicands.
        "dimensionless" => (
            plain_comment(),
            setup_header(1, |bits| {
                bits.put(0x56_4342, 24).put(0, 16).put(2, 24);
                bits.put(0, 1).put(0, 1).put(0, 5).put(0, 5);
                bits.put(1, 4).put(0, 32).put(0, 32).put(0, 4).put(0, 1);
            }),
        ),
        // Forty codebooks of 2^17 entries each: 5 million codeword lengths and tree nodes.
        "many-entries" => (
            plain_comment(),
            setup_header(40, |bits| {
                for _ in 0..40 {
                    bits.put(0x56_4342, 24).put(1, 16).put(1 << 17, 24);
                    bits.put(1, 1).put(16, 5).put(1 << 17, 18); // ordered: all of length 17
                    bits.put(0, 4);
                }
            }),
        ),
        other => panic!("unknown case {other}"),
    };
    stream(2, &comment, &setup, 1, 4, LONG_SILENCE)
}

const HOSTILE_HEADERS: [&str; 6] = [
    "vendor",
    "count",
    "comment",
    "codebook",
    "dimensionless",
    "many-entries",
];

#[test]
fn hostile_vorbis_headers_are_refused_before_the_decoder_sees_them() {
    for case in HOSTILE_HEADERS {
        let bytes = hostile_stream(case);
        let (verdict, peak) = peak_during(|| check_stream_limits(&bytes));
        assert!(verdict.is_err(), "{case}: accepted");
        assert!(peak < MIB, "{case}: the check held {peak} bytes");
        let (decoded, peak) = peak_during(|| decode_all(&bytes, "test:hostile"));
        assert!(decoded.is_err(), "{case}: decoded");
        assert!(peak < 16 * MIB, "{case}: decode_all held {peak} bytes");
        let shared: Arc<[u8]> = Arc::from(bytes);
        let (streamed, peak) = peak_during(|| {
            VorbisDecoder::new(Arc::clone(&shared), "test:hostile")
                .and_then(|mut decoder| decoder.decode().map(|_| ()))
        });
        assert!(streamed.is_err(), "{case}: streamed");
        assert!(peak < 16 * MIB, "{case}: streaming held {peak} bytes");
        eprintln!("{case}: refused ({})", verdict.unwrap_err());
    }
}

/// Files the decoder has always played keep playing: junk before the first page (an ID3 tag
/// from a tagging tool; the Ogg reader skips up to 150 KiB of it) and chained streams (each link
/// with its own three headers), whose every link's headers are checked as the first's are.
#[test]
fn leading_junk_and_chained_streams_still_decode_and_every_link_is_checked() {
    let ordinary = ordinary_stream();
    let expected = decode_all(&ordinary, "test:ordinary")
        .expect("decodes")
        .samples
        .len();

    let mut tagged = b"ID3\x04\x00\x00\x00\x00\x00\x20".to_vec();
    tagged.extend(std::iter::repeat_n(0u8, 32));
    tagged.extend_from_slice(&ordinary);
    check_stream_limits(&tagged).expect("junk before the first page is skipped");
    let decoded = decode_all(&tagged, "test:tagged").expect("decodes");
    assert_eq!(decoded.samples.len(), expected);

    let second = stream_with_serial(7, 2, &plain_comment(), &plain_setup(), 1, 9, LONG_SILENCE);
    let chained = [ordinary.clone(), second].concat();
    check_stream_limits(&chained).expect("a chain of ordinary links");
    let decoded = decode_all(&chained, "test:chained").expect("decodes");
    assert!(decoded.samples.len() > expected, "both links play");

    // A chain whose second link carries a hostile setup header: lewton would read it in the
    // middle of decoding and reserve 2^40 values.
    let hostile = hostile_stream("codebook");
    let mut relinked = hostile.clone();
    for page_start in (0..relinked.len().saturating_sub(4))
        .filter(|&at| &hostile[at..at + 4] == b"OggS")
        .collect::<Vec<_>>()
    {
        relinked[page_start + 14..page_start + 18].copy_from_slice(&9u32.to_le_bytes());
    }
    fix_crcs(&mut relinked);
    let chained = [ordinary.clone(), relinked].concat();
    assert!(
        check_stream_limits(&chained).is_err(),
        "the second link is checked"
    );
    assert!(decode_all(&chained, "test:hostile-link").is_err());

    // A second link of another codec: its packets are not Vorbis headers, which the decoder
    // refuses by their type before reading a length, so the file is not refused here (the first
    // link plays, as before). Its comment packet's "length" would be 4 GB if read as Vorbis.
    let mut tags = b"OpusTags".to_vec();
    tags.extend_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
    let mut other_codec = page(
        0x02,
        0,
        11,
        0,
        &[b"OpusHead\x01\x02\x38\x01\x80\xbb\x00\x00\x00\x00\x00"],
    );
    other_codec.extend(page(0, 0, 11, 1, &[&tags]));
    other_codec.extend(page(0x04, 960, 11, 2, &[&[0xFC, 0xFF, 0xFE]]));
    let mixed = [ordinary.clone(), other_codec].concat();
    check_stream_limits(&mixed).expect("a link of another codec is the decoder's to refuse");

    // Two streams interleaved page by page: refused (not one file a pack ships).
    let other = stream_with_serial(8, 2, &plain_comment(), &plain_setup(), 1, 9, LONG_SILENCE);
    let first_page = 27 + 1 + 30;
    let interleaved = [&ordinary[..first_page], &other[..], &ordinary[first_page..]].concat();
    assert!(check_stream_limits(&interleaved).is_err());
}

/// Found by the fuzzer (vanilla-seeded run, seed 99): a last page whose granule position is near
/// 2^64 makes `lewton`'s end-of-stream trim multiply past `usize`, a panic in builds with
/// overflow checks. Every decode entry point now turns a decoder panic into an error, the replay
/// export's included.
#[test]
fn a_last_granule_near_two_to_the_64_is_never_a_panic() {
    const INPUT: &[u8] = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\xed^\x00\x00\x00\x00\x00\x00\xbbo\x8a=\x01\x1e\x01vorbis\x00\x00\x00\x00\x02D\xac\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\xd8\x01OggS\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xed^\x00\x00\x01\x00\x00\x00\xcd\xf0g^\x02\x1e9\x03vorbis\x03\x00\x00\x00vvv\x02\x00\x00\x00\x01\x00\x00\x00c\x02\x00\x00\x00cc\x01\x05vorbis\x00BCV\x01\x00\x02\x00\x00\x00\x00\x00\x00\x00\x10\x00\x008\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x02OggS\x00\x00\x000\x00\x00\x00\x00\x00\x00\xed^\x00\x00\x02\x00\x00\x00z\x99\xc8a\x04\x01\x01\x01\x01\x0e\x0e\x0e\x0eOggS\x00\x00\x00p\x00\x00\x00\x00\x00\x00\xed^\x00\x00\x03\x00\x00\x00\xe7~D*\x04\x01\x01\x01\x01\x0e\x0e\x0e\x0eOggS\x00\x04\x00\xb0\x00\x00\x00\x00\x00\xfe\xed^\x00\x00\x04\x00\x00\x00_\xb5D)\x04\x01\x01\x01\x01\x0e\x0e\x0e\x0e";
    // Control: the decoder by itself panics on it where overflow checks are on (test builds).
    let raw = std::panic::catch_unwind(|| {
        let mut reader =
            lewton::inside_ogg::OggStreamReader::new(std::io::Cursor::new(INPUT)).ok()?;
        while let Ok(Some(_)) = reader.read_dec_packet_itl() {}
        Some(())
    });
    if cfg!(debug_assertions) {
        assert!(raw.is_err(), "the raw decoder panics on this input");
    }
    let _ = decode_all(INPUT, "test:granule");
    let _ = decode_all_within(INPUT, "test:granule", 1 << 28);
    let shared: Arc<[u8]> = Arc::from(INPUT);
    let _ = count_frames_within(&shared, "test:granule", usize::MAX);
}

/// A replay export decodes music and records whole too, so it passes its own, larger limit.
#[test]
fn a_caller_can_allow_more_samples_than_the_game_decodes_whole() {
    let long = stream(2, &plain_comment(), &plain_setup(), 16, 255, LONG_SILENCE);
    assert!(decode_all(&long, "test:long").is_err());
    let decoded = decode_all_within(&long, "test:long", 1 << 28).expect("within 2^28");
    assert!(decoded.samples.len() > MAX_DECODED_SAMPLES);
}

/// Run in a child process by the negative control below: hands one hostile stream straight to
/// the decoder, without this crate's check.
#[test]
#[ignore = "run in a child process by hostile_headers_make_the_raw_decoder_request_gigabytes"]
fn raw_decoder_on_a_hostile_stream() {
    let Ok(case) = std::env::var("OGG_RAW_DECODER_CASE") else {
        return;
    };
    let bytes = hostile_stream(&case);
    let opened = lewton::inside_ogg::OggStreamReader::new(std::io::Cursor::new(bytes));
    println!("raw decoder returned: {}", opened.is_ok());
}

/// Negative control: without the check, the decoder asks for more memory than any machine has
/// (this binary refuses requests over 1 GiB, and the standard library then ends the process).
#[test]
fn hostile_headers_make_the_raw_decoder_request_gigabytes() {
    let exe = std::env::current_exe().expect("this test binary");
    for case in ["vendor", "count", "comment", "codebook", "dimensionless"] {
        let output = std::process::Command::new(&exe)
            .args(["raw_decoder_on_a_hostile_stream", "--exact", "--ignored"])
            .env("OGG_RAW_DECODER_CASE", case)
            .output()
            .expect("run the child");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let request = stderr
            .lines()
            .find(|line| line.contains("memory allocation of"))
            .map(str::to_owned);
        eprintln!("{case}: {request:?}");
        assert!(
            !output.status.success() && request.is_some(),
            "{case}: the raw decoder did not fail an allocation: {stderr}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The fuzzer.

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

fn mutate(rng: &mut Rng, data: &mut Vec<u8>) {
    for _ in 0..=rng.below(6) {
        let len = data.len();
        if len == 0 {
            data.push(rng.next() as u8);
            continue;
        }
        let at = rng.below(len);
        match rng.below(7) {
            0 => data[at] ^= 1 << rng.below(8),
            1 => data[at] = [0, 0xFF, 0x7F, 0x80, 1, 0xFE][rng.below(6)],
            2 => {
                let end = (at + 1 + rng.below(8)).min(len);
                for byte in &mut data[at..end] {
                    *byte = rng.next() as u8;
                }
            }
            3 => {
                let end = (at + 1 + rng.below(32)).min(len);
                data.drain(at..end);
            }
            4 => {
                let end = (at + 1 + rng.below(64)).min(len);
                let piece = data[at..end].to_vec();
                let to = rng.below(data.len() + 1);
                data.splice(to..to, piece);
            }
            5 => data.truncate(at),
            _ => {
                let bytes: Vec<u8> = (0..1 + rng.below(4)).map(|_| rng.next() as u8).collect();
                data.splice(at..at, bytes);
            }
        }
    }
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// A few small vanilla sounds as extra seeds, when `MC_JARS_DIR` points at the vanilla files.
fn vanilla_seeds() -> Vec<Vec<u8>> {
    let Some(jars) = std::env::var_os("MC_JARS_DIR").map(std::path::PathBuf::from) else {
        return Vec::new();
    };
    let Ok(pack) = mc_assets::ObjectStorePack::open(jars.join("assets"), "1.8") else {
        return Vec::new();
    };
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(pack));
    [
        "sounds/random/click.ogg",
        "sounds/random/pop.ogg",
        "sounds/step/stone1.ogg",
    ]
    .iter()
    .filter_map(|path| rm.load(&mc_assets::ResourceLocation::minecraft(*path)).ok())
    .collect()
}

#[test]
fn damaged_vorbis_streams_never_panic_hang_or_balloon() {
    let mut seeds = vec![
        ordinary_stream(),
        stream(1, &plain_comment(), &plain_setup(), 2, 5, SHORT_SILENCE),
        stream(
            2,
            &comment_header(3, 2, &[1, 2]),
            &plain_setup(),
            3,
            4,
            LONG_SILENCE,
        ),
    ];
    let vanilla = vanilla_seeds();
    eprintln!("vanilla seeds: {}", vanilla.len());
    seeds.extend(vanilla);
    let rounds = 1500 * env_number("OGG_FUZZ_ROUNDS", 1).max(1) as usize;
    let seed = env_number("OGG_FUZZ_SEED", 0x0060_FACE_D00D);
    let current = Arc::new(Mutex::new(Vec::new()));
    let watched = Arc::clone(&current);
    let (done, finished) = mpsc::channel();
    std::thread::Builder::new()
        .name("ogg fuzz".to_owned())
        .stack_size(1 << 20)
        .spawn(move || {
            let mut rng = Rng(seed);
            let started = Instant::now();
            for round in 0..rounds {
                let mut input = seeds[rng.below(seeds.len())].clone();
                mutate(&mut rng, &mut input);
                fix_crcs(&mut input);
                if let Ok(mut slot) = current.lock() {
                    slot.clone_from(&input);
                }
                let began = Instant::now();
                let (outcome, peak) = peak_during(|| {
                    std::panic::catch_unwind(AssertUnwindSafe(|| {
                        let _ = check_stream_limits(&input);
                        let _ = identification(&input);
                        let _ = frames_from_last_page(&input);
                        let _ = decode_all(&input, "fuzz");
                        let shared: Arc<[u8]> = Arc::from(input.as_slice());
                        let _ = count_frames_within(&shared, "fuzz", 4_000_000);
                        if let Ok(mut decoder) = VorbisDecoder::new(shared, "fuzz") {
                            for _ in 0..8 {
                                if decoder.decode().is_err() {
                                    break;
                                }
                            }
                            let _ = decoder.seek(100);
                        }
                    }))
                });
                let took = began.elapsed();
                let shown = || {
                    let text: String = input
                        .escape_ascii()
                        .to_string()
                        .chars()
                        .take(3000)
                        .collect();
                    format!("{} bytes: \"{text}\"", input.len())
                };
                assert!(
                    outcome.is_ok(),
                    "round {round} (seed {seed:#x}) panicked on {}",
                    shown()
                );
                assert!(
                    took < Duration::from_secs(10),
                    "round {round} (seed {seed:#x}) took {took:?} on {}",
                    shown()
                );
                assert!(
                    peak < 256 * MIB,
                    "round {round} (seed {seed:#x}) held {peak} bytes on {}",
                    shown()
                );
            }
            eprintln!(
                "ogg fuzz: {rounds} rounds, seed {seed:#x}, {:?}",
                started.elapsed()
            );
            let _ = done.send(());
        })
        .expect("spawn");
    match finished.recv_timeout(Duration::from_secs(600)) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let input = watched.lock().map(|i| i.len()).unwrap_or_default();
            panic!("an input of {input} bytes never returned")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("the fuzz worker failed"),
    }
}

/// Every vanilla 1.8 sound passes the limits (the check refuses nothing real).
#[test]
#[ignore = "needs the vanilla assets: MC_JARS_DIR=<jars> ... -- --ignored every_vanilla"]
fn every_vanilla_sound_passes_the_stream_limits() {
    let jars = std::env::var_os("MC_JARS_DIR")
        .map(std::path::PathBuf::from)
        .expect("MC_JARS_DIR");
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(jars.join("assets/indexes/1.8.json")).expect("the asset index"),
    )
    .expect("index JSON");
    let objects = index["objects"].as_object().expect("objects");
    let mut checked = 0;
    for (name, object) in objects {
        if !name.ends_with(".ogg") {
            continue;
        }
        let hash = object["hash"].as_str().expect("hash");
        let path = jars.join("assets/objects").join(&hash[..2]).join(hash);
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        check_stream_limits(&bytes).unwrap_or_else(|why| panic!("{name}: {why}"));
        checked += 1;
    }
    eprintln!("{checked} vanilla sounds within the limits");
    assert!(checked > 500, "only {checked} sounds found");
}
