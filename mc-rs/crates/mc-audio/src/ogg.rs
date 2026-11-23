//! Ogg Vorbis decoding, via `lewton`.
//!
//! Two entry points, matching the buffered-vs-streaming split S19 asks for:
//! - [`decode_all`] fully decodes to interleaved PCM up front — short effects.
//! - [`VorbisDecoder`] implements [`kira`]'s [`streaming::Decoder`](kira::sound::streaming::Decoder)
//!   trait, feeding the mixer one packet at a time — music and records.
//!
//! Neither talks to `mc-assets` directly; both take the already-loaded bytes. Resolving a
//! `sounds.json` variant to bytes is [`crate::AudioEngine`]'s job.

use std::io::Cursor;
use std::sync::Arc;

use kira::sound::streaming::Decoder;
use kira::Frame;
use lewton::inside_ogg::OggStreamReader;
use lewton::VorbisError;

use crate::AudioError;

fn vorbis_err(location: &str, e: VorbisError) -> AudioError {
    AudioError::Decode {
        location: location.to_owned(),
        reason: e.to_string(),
    }
}

fn refused(location: &str, reason: String) -> AudioError {
    AudioError::Decode {
        location: location.to_owned(),
        reason,
    }
}

// What a stream may make the decoder hold. Sounds come from resource
// packs (players share them, and a server can offer one), so every field read here is hostile.

/// The most interleaved samples [`decode_all`] produces: 24 Mi, about 4.75 minutes of 44.1 kHz
/// stereo (a clip that long is streamed, never decoded whole). A file of a few kilobytes can
/// decode to hours: one byte of Vorbis packet can stand for 4096 samples per channel.
pub const MAX_DECODED_SAMPLES: usize = 24 << 20;

/// The most frames [`count_frames`] counts before it refuses the file: 2^26, over 25 minutes at
/// 44.1 kHz. It runs on the game thread, for a stream whose last page cannot be trusted.
pub const MAX_COUNTED_FRAMES: usize = 1 << 26;

/// Codebook entries one setup header may declare, all its codebooks together (the decoder keeps
/// a codeword length and a tree node per entry).
const MAX_SETUP_ENTRIES: u64 = 1 << 22;

/// Vector-quantisation values (entries times dimensions, for codebooks with a lookup table) one
/// setup header may declare, all its codebooks together (the decoder expands each into a table of
/// `f32`s while it reads the header).
const MAX_SETUP_VQ_VALUES: u64 = 1 << 22;

/// Checks, before `lewton` reads a byte, that `bytes` is one plain Ogg Vorbis stream whose
/// headers stay within what a decoder may hold.
///
/// `lewton` sizes buffers from header fields before it reads the data they describe: a comment
/// header's 32-bit lengths and count, a codebook's entries times dimensions (up to 2^40 values)
/// and a lookup table's value count (2^32 for a codebook of no dimensions). A file of a few
/// dozen bytes made it ask for terabytes, and a failed allocation ends the process on the spot,
/// with no unwinding and no crash report. The reference decoder refuses a codebook with
/// `ilog(dimensions) + ilog(entries) > 24`; this does the same, refuses a table whose data the
/// packet cannot hold (`lewton` would fail on it after reserving its size) and bounds the totals.
///
/// The pages are walked as the Ogg reader under `lewton` walks them (it looks for the next page's
/// capture pattern up to 150 KiB on, so junk before or between pages is skipped), and the
/// headers of every logical stream are checked: `lewton` reads a chained stream's three headers
/// again in the middle of decoding. Streams interleaved with each other are refused, so the
/// packets checked here are exactly the ones `lewton` parses.
///
/// # Errors
/// Why the stream is refused.
pub fn check_stream_limits(bytes: &[u8]) -> Result<(), String> {
    const HEADER_LEN: usize = 27;
    const CONTINUED: u8 = 0x01;
    let mut links: Vec<Link> = Vec::new();
    let mut at = 0usize;
    while let Some(start) = next_page_start(bytes, at) {
        let page = &bytes[start..];
        // A cut or unknown page ends what the reader can parse (it fails there).
        if page.len() < HEADER_LEN || page[4] != 0 {
            break;
        }
        let serial = u32::from_le_bytes([page[14], page[15], page[16], page[17]]);
        let segments = usize::from(page[26]);
        let body_start = HEADER_LEN + segments;
        let Some(lacing) = page.get(HEADER_LEN..body_start) else {
            break;
        };
        let body_len: usize = lacing.iter().map(|&lace| usize::from(lace)).sum();
        let Some(body) = page.get(body_start..body_start + body_len) else {
            break;
        };
        match links.iter().position(|(link, _, _)| *link == serial) {
            Some(index) if index + 1 == links.len() => {}
            Some(_) => return Err("interleaved logical streams".to_owned()),
            None => links.push((serial, Vec::with_capacity(3), None)),
        }
        let (_, packets, pending) = links.last_mut().expect("pushed above");
        if packets.len() < 3 {
            if (page[5] & CONTINUED != 0) != pending.is_some() {
                return Err(format!(
                    "the Ogg page at byte {start} breaks a header packet"
                ));
            }
            let mut offset = 0;
            for &lace in lacing {
                let end = offset + usize::from(lace);
                pending
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&body[offset..end]);
                offset = end;
                if lace < 255 {
                    packets.extend(pending.take());
                    if packets.len() == 3 {
                        check_comment_header(&packets[1])?;
                        check_setup_header(&packets[2])?;
                        break;
                    }
                }
            }
        }
        at = start + body_start + body_len;
    }
    // A later link cut short is read up to its end and no further; the first must be whole.
    match links.first() {
        Some((_, packets, _)) if packets.len() == 3 => Ok(()),
        _ => Err("fewer than three Vorbis headers".to_owned()),
    }
}

/// One chained stream as [`check_stream_limits`] walks it: its serial, its header packets so
/// far, and a packet still open at the end of the last page.
type Link = (u32, Vec<Vec<u8>>, Option<Vec<u8>>);

/// How far to look for the next page's capture pattern: a little past the 150 KiB the Ogg reader
/// under `lewton` searches, so every page it can find is checked here too.
const OGG_RESYNC_BYTES: usize = 160 * 1024;

/// Where the reader finds the next page at or after `from`: the first capture pattern within
/// [`OGG_RESYNC_BYTES`] (none: the stream ends there).
fn next_page_start(bytes: &[u8], from: usize) -> Option<usize> {
    let window = bytes.get(from..)?;
    let window = &window[..window.len().min(OGG_RESYNC_BYTES + 4)];
    window
        .windows(4)
        .position(|w| w == b"OggS")
        .map(|offset| from + offset)
}

/// The comment header's lengths and count fit the packet (`lewton` reserves them first).
fn check_comment_header(packet: &[u8]) -> Result<(), String> {
    let refuse = || Err("a comment header length runs past the header".to_owned());
    fn le_u32(rest: &mut &[u8]) -> Option<usize> {
        let (head, tail) = rest.split_first_chunk::<4>()?;
        *rest = tail;
        usize::try_from(u32::from_le_bytes(*head)).ok()
    }
    // Not a Vorbis comment header (another codec's link of a chain, damage): `lewton` checks
    // the type and magic first and stops there, before reserving anything; so it does for a
    // packet too short for a field.
    if packet.len() < 7 || packet[0] != 3 || &packet[1..7] != b"vorbis" {
        return Ok(());
    }
    let mut rest = &packet[7..];
    let Some(vendor) = le_u32(&mut rest) else {
        return Ok(());
    };
    if vendor > rest.len() {
        return refuse();
    }
    rest = &rest[vendor..];
    let Some(count) = le_u32(&mut rest) else {
        return Ok(());
    };
    // Each comment takes at least its four-byte length.
    if count > rest.len() / 4 {
        return refuse();
    }
    for _ in 0..count {
        let Some(length) = le_u32(&mut rest) else {
            return refuse();
        };
        if length > rest.len() {
            return refuse();
        }
        rest = &rest[length..];
    }
    Ok(())
}

/// Vorbis I `ilog`: the bits needed to write `value`.
fn ilog(value: u64) -> u32 {
    u64::BITS - value.leading_zeros()
}

/// `lewton`'s `lookup1_values`: the greatest `r` with `r^dimensions <= entries`; for no
/// dimensions, `u32::MAX` (any entries) or 0 (none).
fn lookup1_values(entries: u64, dimensions: u64) -> u64 {
    match dimensions {
        0 if entries == 0 => 0,
        0 => u64::from(u32::MAX),
        32.. => u64::from(entries != 0),
        _ => {
            let fits = |root: u64| {
                root.checked_pow(dimensions as u32)
                    .is_some_and(|p| p <= entries)
            };
            let (mut low, mut high) = (0u64, entries);
            while low < high {
                let middle = low + (high - low).div_ceil(2);
                if fits(middle) {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            low
        }
    }
}

/// Least-significant-bit-first reader, as Vorbis packs its headers.
struct HeaderBits<'a> {
    data: &'a [u8],
    bit: u64,
}

impl HeaderBits<'_> {
    fn remaining(&self) -> u64 {
        (self.data.len() as u64 * 8).saturating_sub(self.bit)
    }

    fn read(&mut self, count: u32) -> Option<u64> {
        if u64::from(count) > self.remaining() {
            return None;
        }
        let mut value = 0u64;
        for index in 0..count {
            let byte = self.data[(self.bit / 8) as usize];
            value |= u64::from((byte >> (self.bit % 8)) & 1) << index;
            self.bit += 1;
        }
        Some(value)
    }

    fn skip(&mut self, count: u64) -> Option<()> {
        (count <= self.remaining()).then(|| self.bit += count)
    }
}

/// Walks the setup header's codebooks exactly as `lewton` reads them, refusing sizes no real
/// stream has before `lewton` reserves them. Whatever `lewton` refuses by itself (a bad sync
/// pattern, a short packet) passes here: it stops there without reserving more.
fn check_setup_header(packet: &[u8]) -> Result<(), String> {
    if packet.len() < 7 || packet[0] != 5 || &packet[1..7] != b"vorbis" {
        return Ok(());
    }
    let mut bits = HeaderBits {
        data: &packet[7..],
        bit: 0,
    };
    let Some(count) = bits.read(8) else {
        return Ok(());
    };
    let (mut entries_total, mut values_total) = (0u64, 0u64);
    for _ in 0..=count {
        if bits.read(24) != Some(0x56_4342) {
            return Ok(());
        }
        let (Some(dimensions), Some(entries)) = (bits.read(16), bits.read(24)) else {
            return Ok(());
        };
        // libvorbis `vorbis_staticbook_unpack`.
        if ilog(dimensions) + ilog(entries) > 24 {
            return Err("a codebook larger than Vorbis allows".to_owned());
        }
        entries_total += entries;
        if entries_total > MAX_SETUP_ENTRIES {
            return Err("codebooks with more entries than a sound needs".to_owned());
        }
        let Some(ordered) = bits.read(1) else {
            return Ok(());
        };
        if ordered == 0 {
            let Some(sparse) = bits.read(1) else {
                return Ok(());
            };
            for _ in 0..entries {
                let used = if sparse == 1 { bits.read(1) } else { Some(1) };
                let read = match used {
                    None => false,
                    Some(0) => true,
                    Some(_) => bits.read(5).is_some(),
                };
                if !read {
                    return Ok(());
                }
            }
        } else {
            if bits.read(5).is_none() {
                return Ok(());
            }
            let mut current = 0u64;
            while current < entries {
                let Some(number) = bits.read(ilog(entries - current)) else {
                    return Ok(());
                };
                current += number;
            }
            if current > entries {
                return Ok(());
            }
        }
        let Some(lookup) = bits.read(4) else {
            return Ok(());
        };
        if lookup == 0 {
            continue;
        }
        if lookup > 2 {
            return Ok(());
        }
        // Minimum and delta (32 bits each), value bits, sequence flag.
        let Some(value_bits) = bits.skip(64).and_then(|()| bits.read(4)) else {
            return Ok(());
        };
        if bits.read(1).is_none() {
            return Ok(());
        }
        let values = if lookup == 1 {
            lookup1_values(entries, dimensions)
        } else {
            entries * dimensions
        };
        // `lewton` reserves `values` before reading them, so a table the packet cannot hold is
        // refused here rather than failing after the reservation.
        let table_bits = values.saturating_mul(value_bits + 1);
        if table_bits > bits.remaining() {
            return Err("a codebook table longer than its header".to_owned());
        }
        values_total += entries * dimensions;
        if values_total > MAX_SETUP_VQ_VALUES {
            return Err("codebook tables larger than a sound needs".to_owned());
        }
        if bits.skip(table_bits).is_none() {
            return Ok(());
        }
    }
    Ok(())
}

/// A fully-decoded clip: interleaved PCM samples at their native sample rate/channel count.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    /// Samples per second.
    pub sample_rate: u32,
    /// `1` (mono) or `2` (stereo) for every real vanilla sound asset. Vorbis itself allows more.
    pub channels: u8,
    /// Interleaved `i16` samples: `[L, R, L, R, ...]` for stereo, `[S, S, ...]` for mono.
    pub samples: Vec<i16>,
}

impl DecodedAudio {
    /// Number of sample frames (i.e. `samples.len() / channels`).
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    /// Duration in seconds, from `num_frames` and `sample_rate`.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.num_frames() as f64 / self.sample_rate as f64
        }
    }

    /// Converts to kira's stereo [`Frame`]s (mono is duplicated to both channels, matching
    /// [`Frame::from_mono`]), ready to hand `kira` as a [`kira::sound::static_sound::StaticSoundData`].
    pub fn to_frames(&self) -> Vec<Frame> {
        interleaved_to_frames(&self.samples, self.channels)
    }
}

/// Converts interleaved `i16` PCM to kira [`Frame`]s, scaling to `[-1.0, 1.0]`.
fn interleaved_to_frames(samples: &[i16], channels: u8) -> Vec<Frame> {
    const SCALE: f32 = 1.0 / 32768.0;
    match channels {
        0 => Vec::new(),
        1 => samples
            .iter()
            .map(|&s| Frame::from_mono(s as f32 * SCALE))
            .collect(),
        _ => samples
            .chunks_exact(channels as usize)
            .map(|frame| {
                // Anything beyond stereo (vanilla never ships >2 channels) is folded down to
                // the first two channels rather than panicking or silently dropping the sound.
                Frame::new(frame[0] as f32 * SCALE, frame[1] as f32 * SCALE)
            })
            .collect(),
    }
}

/// Fully decodes an in-memory Ogg Vorbis stream to PCM.
///
/// `location` is only used to label errors; pass the resource location being decoded.
pub fn decode_all(bytes: &[u8], location: &str) -> Result<DecodedAudio, AudioError> {
    decode_all_within(bytes, location, MAX_DECODED_SAMPLES)
}

/// [`decode_all`] with its own limit on the interleaved samples it produces. The game decodes
/// whole only clips it would not stream ([`MAX_DECODED_SAMPLES`]); a replay export decodes
/// every recorded sound, music included (vanilla's longest track is about 80 million samples).
///
/// # Errors
/// As [`decode_all`], and when the stream decodes to more than `max_samples` samples.
pub fn decode_all_within(
    bytes: &[u8],
    location: &str,
    max_samples: usize,
) -> Result<DecodedAudio, AudioError> {
    check_stream_limits(bytes).map_err(|reason| refused(location, reason))?;
    contained(location, || decode_checked(bytes, location, max_samples))
}

/// Runs one `lewton` decode, turning a panic inside it into an error
/// for every caller (the engine's decodes, the replay export, the frame count). The fuzzer found
/// `lewton` panicking on damaged streams (an overflowing multiply in its end-of-stream trim, in
/// builds with overflow checks).
fn contained<T>(
    location: &str,
    decode: impl FnOnce() -> Result<T, AudioError>,
) -> Result<T, AudioError> {
    crate::engine::contain_prewarm_panic(|| Some(decode()))
        .unwrap_or_else(|| Err(refused(location, "the Vorbis decoder panicked".to_owned())))
}

fn decode_checked(
    bytes: &[u8],
    location: &str,
    max_samples: usize,
) -> Result<DecodedAudio, AudioError> {
    let mut reader =
        OggStreamReader::new(Cursor::new(bytes)).map_err(|e| vorbis_err(location, e))?;
    let sample_rate = reader.ident_hdr.audio_sample_rate;
    let channels = reader.ident_hdr.audio_channels;
    let mut samples = Vec::new();
    while let Some(packet) = reader
        .read_dec_packet_itl()
        .map_err(|e| vorbis_err(location, e))?
    {
        // Checked before the samples are added.
        if samples.len() + packet.len() > max_samples {
            return Err(refused(
                location,
                format!("decodes to more than {max_samples} samples"),
            ));
        }
        samples.extend(packet);
    }
    Ok(DecodedAudio {
        sample_rate,
        channels,
        samples,
    })
}

/// Feeds an Ogg Vorbis stream to kira one packet at a time, rather than decoding the whole clip
/// up front — used for music and records (`sounds.json`'s `"stream": true`).
///
/// The compressed bytes are still held resident (`mc-assets`'s `ResourceManager::open` returns
/// `Read`, not `Read + Seek`, and `lewton` needs `Seek`; buffering once is far cheaper than
/// re-fetching), but PCM — roughly 10x larger — is produced incrementally as kira's mixer asks
/// for it, which is the memory/CPU trade-off "streaming" is actually about for a resident game
/// asset. `bytes` is reference-counted so re-opening the stream (see `num_frames`, `seek`) is a
/// cheap clone rather than a copy.
pub struct VorbisDecoder {
    bytes: Arc<[u8]>,
    /// Opened by the first [`Decoder::decode`] or seek past the start, which kira runs on the
    /// stream's own thread, not in [`Self::new`] on the game thread: reading the setup header
    /// (the codebooks) costs about 0.6 ms even for a tiny effect. `Some` from the start only
    /// when [`identification`] could not read the rate and channels by itself.
    reader: Option<OggStreamReader<Cursor<Arc<[u8]>>>>,
    sample_rate: u32,
    channels: u8,
    num_frames: usize,
    /// Frames handed to kira so far, which is what EOF padding is measured against.
    delivered: usize,
    location: String,
}

impl VorbisDecoder {
    /// Builds a streaming decoder. kira needs the total frame count up front
    /// (`kira::sound::streaming::Decoder::num_frames`) and Vorbis's headers don't carry one, but
    /// the final Ogg page does, so this does not decode any audio. This runs on the game thread:
    /// a full decode pass here froze the whole client for hundreds of milliseconds per track.
    /// A file without a usable final page falls back to [`count_frames`].
    ///
    /// The sample rate and channel count come from the
    /// identification header, read by hand; the full Vorbis reader opens at the first decode, on
    /// kira's thread. A file whose later headers are broken therefore fails there, which kira
    /// reports by stopping the sound, instead of here.
    pub fn new(bytes: Arc<[u8]>, location: impl Into<String>) -> Result<Self, AudioError> {
        let location = location.into();
        let num_frames = match frames_from_last_page(&bytes) {
            Some(frames) => frames,
            None => count_frames(&bytes, &location)?,
        };
        let (reader, channels, sample_rate) = match identification(&bytes) {
            Some((channels, sample_rate)) => (None, channels, sample_rate),
            None => {
                let reader = open_reader(&bytes, &location)?;
                let channels = reader.ident_hdr.audio_channels;
                let sample_rate = reader.ident_hdr.audio_sample_rate;
                (Some(reader), channels, sample_rate)
            }
        };
        Ok(Self {
            bytes,
            reader,
            sample_rate,
            channels,
            num_frames,
            delivered: 0,
            location,
        })
    }

    /// The Vorbis reader, opened on first use.
    fn reader(&mut self) -> Result<&mut OggStreamReader<Cursor<Arc<[u8]>>>, AudioError> {
        if self.reader.is_none() {
            self.reader = Some(open_reader(&self.bytes, &self.location)?);
        }
        Ok(self.reader.as_mut().expect("opened just above"))
    }
}

fn open_reader(
    bytes: &Arc<[u8]>,
    location: &str,
) -> Result<OggStreamReader<Cursor<Arc<[u8]>>>, AudioError> {
    check_stream_limits(bytes).map_err(|reason| refused(location, reason))?;
    OggStreamReader::new(Cursor::new(Arc::clone(bytes))).map_err(|e| vorbis_err(location, e))
}

/// `(channels, sample rate)` from the Vorbis identification header, without parsing the rest.
///
/// Vorbis I spec, 4.2.2: the identification header is the only packet on the stream's first
/// Ogg page and is exactly 30 bytes: type `1`, `"vorbis"`, version `0` (u32), channels (u8),
/// sample rate (u32 LE), three bitrates, the two block sizes (4 bits each, 2^6..=2^13, the first
/// not larger) and a set framing bit. `None` unless every one of those checks out, as
/// `lewton`'s own reader requires; the caller then lets `lewton` read the headers.
#[must_use]
pub fn identification(bytes: &[u8]) -> Option<(u8, u32)> {
    const HEADER_LEN: usize = 27;
    const PACKET_LEN: usize = 30;
    if bytes.len() < HEADER_LEN || !bytes.starts_with(b"OggS") || bytes[4] != 0 {
        return None;
    }
    let segments = usize::from(bytes[26]);
    let table = bytes.get(HEADER_LEN..HEADER_LEN + segments)?;
    // A packet ends at the first lacing value below 255; 30 bytes is a single lacing value.
    if table.first() != Some(&(PACKET_LEN as u8)) {
        return None;
    }
    let start = HEADER_LEN + segments;
    let packet = bytes.get(start..start + PACKET_LEN)?;
    let block_sizes = packet[28];
    let (small, large) = (block_sizes & 0x0f, block_sizes >> 4);
    let valid = packet[0] == 1
        && &packet[1..7] == b"vorbis"
        && packet[7..11] == [0, 0, 0, 0]
        && (6..=13).contains(&small)
        && (6..=13).contains(&large)
        && small <= large
        && packet[29] & 1 == 1;
    let channels = packet[11];
    let sample_rate = u32::from_le_bytes(packet[12..16].try_into().ok()?);
    (valid && channels > 0 && sample_rate > 0).then_some((channels, sample_rate))
}

/// The stream's total sample frames, read from its last Ogg page without decoding any audio.
///
/// RFC 3533 page header: `OggS`, version, flags, granule position (u64 LE at 6), stream serial
/// (u32 LE at 14), sequence, checksum, segment count at 26, then the segment table. For Vorbis, the
/// final page's granule position is the stream's total sample count.
///
/// Returns `None` unless the last page is unambiguous: it must parse, end exactly at the end of
/// the file, be flagged end-of-stream, and belong to the same logical stream as the first page.
/// The pattern can occur by chance inside audio data, and a chained file has several streams;
/// both fall back to the decode pass rather than guessing.
#[must_use]
pub fn frames_from_last_page(bytes: &[u8]) -> Option<usize> {
    const CAPTURE: &[u8; 4] = b"OggS";
    const HEADER_LEN: usize = 27;
    const END_OF_STREAM: u8 = 0x04;
    if bytes.len() < HEADER_LEN || !bytes.starts_with(CAPTURE) {
        return None;
    }
    let start = bytes.windows(CAPTURE.len()).rposition(|w| w == CAPTURE)?;
    let page = &bytes[start..];
    if page.len() < HEADER_LEN || page[4] != 0 || page[5] & END_OF_STREAM == 0 {
        return None;
    }
    if page[14..18] != bytes[14..18] {
        return None;
    }
    let segments = usize::from(page[26]);
    let table = page.get(HEADER_LEN..HEADER_LEN + segments)?;
    let body: usize = table.iter().map(|&lacing| usize::from(lacing)).sum();
    if HEADER_LEN + segments + body != page.len() {
        return None;
    }
    let granule = u64::from_le_bytes(page[6..14].try_into().ok()?);
    // All-ones means no packet finishes on this page; zero is not a playable stream.
    if granule == 0 || granule == u64::MAX {
        return None;
    }
    // A track cut from a longer stream keeps that stream's positions, so its last page can claim
    // far more audio than the file could hold. Decoding gives the true length.
    const MOST_FRAMES_PER_BYTE: u64 = 64;
    if granule > bytes.len() as u64 * MOST_FRAMES_PER_BYTE {
        return None;
    }
    usize::try_from(granule).ok()
}

/// The total sample frames found by decoding the whole stream. Slow: for a file whose last page
/// cannot be trusted, and for checking that page against the decoder.
pub fn count_frames(bytes: &Arc<[u8]>, location: &str) -> Result<usize, AudioError> {
    count_frames_within(bytes, location, MAX_COUNTED_FRAMES)
}

/// [`count_frames`] with its own limit: refuses the file once the count passes `max_frames`.
///
/// # Errors
/// As [`count_frames`].
pub fn count_frames_within(
    bytes: &Arc<[u8]>,
    location: &str,
    max_frames: usize,
) -> Result<usize, AudioError> {
    contained(location, || {
        let mut reader = open_reader(bytes, location)?;
        let channels = reader.ident_hdr.audio_channels.max(1) as usize;
        let mut total = 0usize;
        while let Some(packet) = reader
            .read_dec_packet_itl()
            .map_err(|e| vorbis_err(location, e))?
        {
            total += packet.len() / channels;
            // The count decodes every packet, on the game thread.
            if total > max_frames {
                return Err(refused(
                    location,
                    format!("decodes to more than {max_frames} frames"),
                ));
            }
        }
        Ok(total)
    })
}

impl Decoder for VorbisDecoder {
    type Error = AudioError;

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn num_frames(&self) -> usize {
        self.num_frames
    }

    fn decode(&mut self) -> Result<Vec<Frame>, Self::Error> {
        // kira runs this on its stream thread, where a
        // decoder panic on a hostile pack file was nobody's to catch (the worker and the game
        // thread contain theirs): it now stops this sound like any decode error.
        let read = crate::engine::contain_prewarm_panic(|| {
            Some(self.reader().map(|reader| reader.read_dec_packet_itl()))
        });
        let packet = match read {
            Some(read) => read?,
            None => {
                return Err(refused(
                    &self.location,
                    "the Vorbis decoder panicked".to_owned(),
                ))
            }
        };
        match packet.map_err(|e| vorbis_err(&self.location, e))? {
            Some(itl) => {
                let frames = interleaved_to_frames(&itl, self.channels);
                self.delivered += frames.len();
                Ok(frames)
            }
            // kira's decode loop spins until it reaches `num_frames` and treats an empty chunk as
            // "try again". If the file decodes shorter than its last page claims, pad with silence
            // so a malformed or truncated resource-pack track ends instead of hanging the thread.
            None => {
                const PADDING_CHUNK: usize = 4096;
                let padding = self
                    .num_frames
                    .saturating_sub(self.delivered)
                    .min(PADDING_CHUNK);
                self.delivered += padding;
                Ok(vec![Frame::ZERO; padding])
            }
        }
    }

    fn seek(&mut self, index: usize) -> Result<usize, Self::Error> {
        // kira seeks every stream to its start position before the first decode. A page seek to 0
        // lands on the first page, which holds the Vorbis headers, so the next decode would fail
        // and kira would stop the sound without reporting it. The audio starts after the headers:
        // an untouched reader is already there, and otherwise the stream is opened again.
        if index == 0 {
            if self.delivered > 0 {
                self.reader = Some(open_reader(&self.bytes, &self.location)?);
                self.delivered = 0;
            }
            return Ok(0);
        }
        // `lewton` only seeks to page (granule-position) granularity, not the exact sample —
        // within the trait's stated contract ("OK if the decoder seeks to an earlier sample").
        // The page bisection can panic on a damaged stream
        // (an unchecked page sequence subtraction in the Ogg reader).
        let seeked = crate::engine::contain_prewarm_panic(|| {
            Some(
                self.reader()
                    .map(|reader| reader.seek_absgp_pg(index as u64)),
            )
        });
        match seeked {
            Some(seeked) => seeked?.map_err(|e| vorbis_err(&self.location, e))?,
            None => {
                return Err(refused(
                    &self.location,
                    "the Vorbis decoder panicked while seeking".to_owned(),
                ))
            }
        }
        self.delivered = index.min(self.num_frames);
        Ok(self.delivered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal real Ogg Vorbis file, generated once with a pure-Rust encoder and embedded as
    /// bytes — this crate has no synthesis tooling of its own, so integration tests reach for a
    /// real asset from `MCP-919-main/jars/` instead (see `tests/vanilla_sounds.rs`). This unit
    /// test only exercises the pure PCM-scaling math, which needs no file at all.
    #[test]
    fn interleaved_to_frames_scales_and_splits_channels() {
        let mono = interleaved_to_frames(&[0, 16384, -16384, i16::MAX, i16::MIN], 1);
        assert_eq!(mono.len(), 5);
        assert_eq!(mono[0], Frame::new(0.0, 0.0));
        assert!((mono[1].left - 0.5).abs() < 1e-4);
        assert!((mono[2].left + 0.5).abs() < 1e-4);
        assert!(
            mono[1].left == mono[1].right,
            "mono duplicates to both channels"
        );

        let stereo = interleaved_to_frames(&[0, 0, 16384, -16384], 2);
        assert_eq!(stereo.len(), 2);
        assert_eq!(stereo[0], Frame::new(0.0, 0.0));
        assert!((stereo[1].left - 0.5).abs() < 1e-4);
        assert!((stereo[1].right + 0.5).abs() < 1e-4);
    }

    #[test]
    fn zero_channels_yields_no_frames() {
        assert!(interleaved_to_frames(&[1, 2, 3], 0).is_empty());
    }

    /// Containment nests: the decode API contains itself
    /// inside the engine's containment, and the outer scope must still be contained after the
    /// inner one returns.
    #[test]
    fn nested_decoder_containment_keeps_the_outer_scope_contained() {
        use crate::engine::prewarm_panic_is_being_contained as contained_now;
        assert!(!contained_now());
        let outer = crate::engine::contain_prewarm_panic(|| {
            let inner = crate::engine::contain_prewarm_panic(|| Some(contained_now()));
            Some((inner, contained_now()))
        });
        assert_eq!(outer, Some((Some(true), true)));
        assert!(!contained_now(), "cleared once the outermost scope ends");
    }

    #[test]
    fn decode_all_rejects_garbage() {
        let err = decode_all(b"not an ogg file", "test:garbage").unwrap_err();
        assert!(matches!(err, AudioError::Decode { .. }));
    }

    /// Real vanilla `.ogg` bytes from the jars directory (`MC_JARS_DIR`, or the checkout's own).
    fn vanilla_ogg(path: &str) -> Option<Arc<[u8]>> {
        let jars = match std::env::var_os("MC_JARS_DIR") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(3)?
                .join("MCP-919-main")
                .join("jars"),
        };
        let Ok(pack) = mc_assets::ObjectStorePack::open(jars.join("assets"), "1.8") else {
            eprintln!("SKIP: no 1.8 asset index; set MC_JARS_DIR to the jars/ directory");
            return None;
        };
        let mut rm = mc_assets::ResourceManager::new();
        rm.push_pack(Box::new(pack));
        rm.load(&mc_assets::ResourceLocation::minecraft(path))
            .ok()
            .map(Arc::from)
    }

    /// The hand-read identification header agrees with `lewton`'s on real files, mono and
    /// stereo, effect and music.
    #[test]
    fn identification_matches_lewton_on_real_files() {
        for path in [
            "sounds/random/pop.ogg",
            "sounds/mob/enderdragon/end.ogg",
            "sounds/records/13.ogg",
            "sounds/music/menu/menu1.ogg",
        ] {
            let Some(bytes) = vanilla_ogg(path) else {
                return;
            };
            let reader = OggStreamReader::new(Cursor::new(Arc::clone(&bytes))).unwrap();
            assert_eq!(
                identification(&bytes),
                Some((
                    reader.ident_hdr.audio_channels,
                    reader.ident_hdr.audio_sample_rate
                )),
                "{path}"
            );
        }
    }

    /// Anything that is not a well-formed identification header is left to `lewton`.
    #[test]
    fn identification_rejects_what_lewton_would_reject() {
        assert_eq!(identification(b""), None);
        assert_eq!(
            identification(b"not an ogg file at all, just text....."),
            None
        );
        let Some(bytes) = vanilla_ogg("sounds/random/pop.ogg") else {
            return;
        };
        let start = 27 + usize::from(bytes[26]);
        let breaks: [(usize, u8); 5] = [
            (start, 3),         // not the identification packet type
            (start + 7, 1),     // a Vorbis version other than 0
            (start + 11, 0),    // no channels
            (start + 28, 0x55), // block sizes 2^5: too small
            (start + 29, 0),    // framing bit clear
        ];
        for (at, value) in breaks {
            let mut broken = bytes.to_vec();
            broken[at] = value;
            assert_eq!(identification(&broken), None, "byte {at} = {value}");
        }
    }

    /// The decoder opens `lewton` only at its first decode, and what it delivers is exactly
    /// `decode_all`'s audio, up to the length the last page gives.
    #[test]
    fn a_lazily_opened_stream_delivers_decode_alls_frames() {
        let Some(bytes) = vanilla_ogg("sounds/random/pop.ogg") else {
            return;
        };
        let mut decoder = VorbisDecoder::new(Arc::clone(&bytes), "pop").unwrap();
        assert!(
            decoder.reader.is_none(),
            "new() must not open the Vorbis reader"
        );
        assert_eq!(decoder.seek(0).unwrap(), 0);
        assert!(
            decoder.reader.is_none(),
            "kira's start seek must not open it either"
        );
        let expected = decode_all(&bytes, "pop").unwrap();
        assert_eq!(decoder.sample_rate(), expected.sample_rate);
        let total = decoder.num_frames();
        let mut delivered = Vec::new();
        while delivered.len() < total {
            delivered.extend(decoder.decode().unwrap());
        }
        let expected = expected.to_frames();
        assert!(total <= expected.len());
        assert_eq!(delivered[..total], expected[..total]);
    }

    #[test]
    fn decoded_audio_duration_and_frame_count() {
        let d = DecodedAudio {
            sample_rate: 44100,
            channels: 2,
            samples: vec![0; 44100 * 2], // 1 second of stereo silence
        };
        assert_eq!(d.num_frames(), 44100);
        assert!((d.duration_secs() - 1.0).abs() < 1e-9);
        assert_eq!(d.to_frames().len(), 44100);
    }
}
