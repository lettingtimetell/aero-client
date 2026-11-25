//! Protocol-5 NBT: a gzip member around a named compound, read with 1.7.10's own tag reader.
//!
//! java17: `nbt/CompressedStreamTools.java` (`compress`, `decompress`), `nbt/NBTSizeTracker.java`,
//! each `nbt/NBTTag*.read`, and `java.util.zip.GZIPInputStream` / `GZIPOutputStream` (JDK 8).
//!
//! # Reading
//!
//! `PacketBuffer.readNBTTagCompoundFromBuffer` hands the blob to
//! `new DataInputStream(new BufferedInputStream(new GZIPInputStream(...)))` and reads one tag.
//! Two consequences shape [`read_gzip_compound`]:
//!
//! * **The stream is pulled, not unpacked.** The gzip header is checked when the stream is
//!   opened (`GZIPInputStream.readHeader`), but the deflate data is inflated only as far as the
//!   tag reader asks, and a member's trailer (CRC-32, size) is read only when the reader asks for
//!   a byte past that member. A well-formed tag therefore never looks at the trailer, and bytes
//!   after it are ignored. [`GzipStream`] reproduces this, including the JDK's rule for when a
//!   further concatenated member is tried.
//! * **1.7.10's tag reader is not 1.8.9's.** Its `NBTSizeTracker` charges payload bits only
//!   (byte 8, short 16, int and float 32, long and double 64, list 8, strings and keys 16 per
//!   char, arrays per element), none of 1.8.9's per-tag overheads. A list's count is never
//!   checked: a negative count is an empty list (which keeps its declared element type), and a
//!   `TAG_End` list with elements is accepted (1.8.9 rejects both). An unknown type id is the
//!   `NullPointerException` of `NBTBase.createNewByType`, here [`NbtError::UnknownTagType`].
//!   Depth is capped at 512, as in 1.8.9. [`mc_nbt`]'s reader carries the 1.8.9 rules, so this
//!   module has its own.
//!
//! # Writing
//!
//! `CompressedStreamTools.compress` writes the named compound through
//! `new GZIPOutputStream(...)`: a fixed ten-byte header ([`JAVA_GZIP_HEADER`]), a raw deflate
//! stream at the default level 6, then the CRC-32 and size trailer. [`write_gzip_compound`]
//! writes exactly that header and trailer around [`write_compound_java17`] (the bytes of
//! [`mc_nbt::write_compound`], plus the `TAG_End` lists only 1.7.10 reads). The deflate body
//! comes from `flate2`, so its bytes can differ from the JDK's zlib while inflating to the same
//! NBT; a receiver only ever sees the inflated tag.
//!
//! # Limits beyond Java's
//!
//! No allocation is sized from a length field. On top of 1.7.10's own 2 MiB accounting, each
//! packet carries an [`NbtBudget`]: at most [`NbtBudget::PACKET_NODES`] decoded values and
//! [`NbtBudget::PACKET_INFLATED_BYTES`] inflated bytes across all of its tags. Gzip lets a few
//! bytes on the wire expand a thousandfold, so without a per-packet cap one frame of tiny item
//! tags could demand gigabytes. Vanilla has no such cap and runs out of heap on the same input;
//! real traffic stays orders of magnitude below both limits.

use std::io::Read;

use flate2::{Compress, Compression, Crc, Decompress, FlushCompress, FlushDecompress, Status};
use mc_nbt::modified_utf8;
use mc_nbt::{Nbt, NbtCompound, NbtError, NbtList, TagType};

use crate::ProtoError;

/// java17: `PacketBuffer.readNBTTagCompoundFromBuffer`, `new NBTSizeTracker(2097152L)`.
pub const SIZE_TRACKER_LIMIT: u64 = 2_097_152;

/// java17: `NBTTagCompound.read` / `NBTTagList.read`, `depth > 512`.
pub const MAX_DEPTH: u32 = 512;

/// The ten bytes `java.util.zip.GZIPOutputStream` (JDK 8) writes before the deflate data:
/// magic, method 8 (deflate), no flags, MTIME 0, XFL 0, OS 0.
pub const JAVA_GZIP_HEADER: [u8; 10] = [0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

/// `java.util.zip.InflaterInputStream`'s input buffer for `new GZIPInputStream(in)`. Used only
/// to decide, exactly as the JDK does, whether a further gzip member is tried.
const JAVA_INFLATER_BUFFER: usize = 512;

/// `java.io.BufferedInputStream`'s default size: how much is inflated per refill.
const INFLATE_CHUNK: usize = 8192;

/// `Deflater.DEFAULT_COMPRESSION`, which zlib maps to level 6.
const JAVA_DEFAULT_LEVEL: u32 = 6;

// Gzip header flag bits (RFC 1952), as `GZIPInputStream.readHeader` tests them.
const FHCRC: u8 = 2;
const FEXTRA: u8 = 4;
const FNAME: u8 = 8;
const FCOMMENT: u8 = 16;

/// Per-packet bound on the NBT one protocol-5 packet may decode. See the module docs.
#[derive(Debug, Clone)]
pub struct NbtBudget {
    nodes_left: u64,
    inflated_left: u64,
}

impl NbtBudget {
    /// Decoded NBT values (every tag, list element and compound) one packet may produce.
    pub const PACKET_NODES: u64 = 1 << 18;
    /// Inflated bytes one packet's gzip members may produce.
    pub const PACKET_INFLATED_BYTES: u64 = 16 << 20;

    /// A fresh budget for one packet.
    #[must_use]
    pub fn per_packet() -> NbtBudget {
        NbtBudget {
            nodes_left: Self::PACKET_NODES,
            inflated_left: Self::PACKET_INFLATED_BYTES,
        }
    }

    fn take_node(&mut self) -> Result<(), NbtError> {
        if self.nodes_left == 0 {
            return Err(NbtError::Message(format!(
                "a protocol-5 packet decoded more than {} NBT values",
                Self::PACKET_NODES
            )));
        }
        self.nodes_left -= 1;
        Ok(())
    }

    fn take_inflated(&mut self, n: usize) -> std::io::Result<()> {
        let n = n as u64;
        if n > self.inflated_left {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "a protocol-5 packet inflated more than {} NBT bytes",
                    Self::PACKET_INFLATED_BYTES
                ),
            ));
        }
        self.inflated_left -= n;
        Ok(())
    }
}

impl Default for NbtBudget {
    fn default() -> Self {
        NbtBudget::per_packet()
    }
}

/// Decode one protocol-5 NBT blob: a gzip stream holding a named compound.
///
/// java17: `CompressedStreamTools.decompress(byte[], NBTSizeTracker)`
///
/// # Errors
///
/// [`ProtoError::Zlib`] for a bad gzip header, [`ProtoError::Nbt`] for a bad deflate stream,
/// a tag 1.7.10 rejects, a root that is not a compound, or an exhausted budget.
pub fn read_gzip_compound(blob: &[u8], budget: &mut NbtBudget) -> Result<NbtCompound, ProtoError> {
    let mut stream = GzipStream::open(blob, budget)?;
    Ok(read_root_compound(&mut stream)?)
}

/// Encode a compound the way `CompressedStreamTools.compress` does: Java's gzip header, a raw
/// deflate stream (level 6) of the named compound with an empty name, then the trailer.
///
/// # Errors
///
/// As [`write_compound_java17`].
pub fn write_gzip_compound(compound: &NbtCompound) -> Result<Vec<u8>, NbtError> {
    gzip_java(&write_compound_java17(compound)?)
}

/// The named compound with an empty name, as java17 `CompressedStreamTools.write` and each
/// `NBTTag*.write` produce it.
///
/// The same bytes as [`mc_nbt::write_compound`] with an empty name, except for one value only the
/// 1.7.10 reader produces: a `TAG_End` list with elements (`NBTTagList.read` accepts one, see the
/// module docs), which `NBTTagList.write` writes as type 0, the count and nothing per element, and
/// which `mc_nbt` refuses. An item carrying such a list therefore goes back to the server as Java
/// sends it instead of failing the whole packet.
///
/// # Errors
///
/// [`NbtError`] for what neither writer can encode: a key or string over 65535 modified-UTF-8
/// bytes (the JDK's `UTFDataFormatException`), a heterogeneous list or `TAG_End` stored in a
/// compound (which no read produces), nesting past 512. `mc_nbt::write_compound` fails on every
/// such compound too.
pub fn write_compound_java17(compound: &NbtCompound) -> Result<Vec<u8>, NbtError> {
    let mut out = vec![TagType::Compound.id(), 0, 0];
    write_compound_entries(&mut out, compound, 0)?;
    Ok(out)
}

fn write_compound_entries(
    out: &mut Vec<u8>,
    compound: &NbtCompound,
    depth: u32,
) -> Result<(), NbtError> {
    if depth > MAX_DEPTH {
        return Err(NbtError::DepthLimit { limit: MAX_DEPTH });
    }
    for (key, value) in compound.iter() {
        // java17 writeEntry would write a bare 0 here and end the compound early; no read
        // produces it, so refuse it as mc_nbt does.
        if value.tag_type() == TagType::End {
            return Err(NbtError::EndInContainer);
        }
        out.push(value.id());
        write_utf(out, key)?;
        write_payload_java17(out, value, depth + 1)?;
    }
    out.push(TagType::End.id());
    Ok(())
}

fn write_utf(out: &mut Vec<u8>, s: &str) -> Result<(), NbtError> {
    let mut encoded = Vec::new();
    let len = modified_utf8::encode_into(s, &mut encoded)?;
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.extend_from_slice(&encoded);
    Ok(())
}

fn write_len(out: &mut Vec<u8>, kind: TagType, len: usize) -> Result<(), NbtError> {
    let len = i32::try_from(len).map_err(|_| NbtError::NegativeLength {
        kind,
        len: i32::MAX,
    })?;
    out.extend_from_slice(&len.to_be_bytes());
    Ok(())
}

fn write_payload_java17(out: &mut Vec<u8>, value: &Nbt, depth: u32) -> Result<(), NbtError> {
    match value {
        Nbt::End => {}
        Nbt::Byte(v) => out.extend_from_slice(&v.to_be_bytes()),
        Nbt::Short(v) => out.extend_from_slice(&v.to_be_bytes()),
        Nbt::Int(v) => out.extend_from_slice(&v.to_be_bytes()),
        Nbt::Long(v) => out.extend_from_slice(&v.to_be_bytes()),
        Nbt::Float(v) => out.extend_from_slice(&v.to_be_bytes()),
        Nbt::Double(v) => out.extend_from_slice(&v.to_be_bytes()),
        Nbt::ByteArray(bytes) => {
            write_len(out, TagType::ByteArray, bytes.len())?;
            out.extend_from_slice(bytes);
        }
        Nbt::String(s) => write_utf(out, s)?,
        Nbt::IntArray(ints) => {
            write_len(out, TagType::IntArray, ints.len())?;
            for i in ints {
                out.extend_from_slice(&i.to_be_bytes());
            }
        }
        Nbt::List(items) => {
            if depth > MAX_DEPTH {
                return Err(NbtError::DepthLimit { limit: MAX_DEPTH });
            }
            // java17 NBTTagList.write: the element type is the first element's, TAG_End when
            // empty; every element is the list's type because only `appendTag` adds them.
            let elem = items.first().map_or(TagType::End, Nbt::tag_type);
            if let Some((index, found)) = items
                .iter()
                .map(Nbt::tag_type)
                .enumerate()
                .find(|(_, found)| *found != elem)
            {
                return Err(NbtError::HeterogeneousList {
                    index,
                    expected: elem,
                    found,
                });
            }
            out.push(elem.id());
            write_len(out, TagType::List, items.len())?;
            for item in items.iter() {
                write_payload_java17(out, item, depth + 1)?;
            }
        }
        Nbt::Compound(compound) => write_compound_entries(out, compound, depth)?,
    }
    Ok(())
}

/// Wrap already-serialised bytes the way `java.util.zip.GZIPOutputStream` (JDK 8) does:
/// [`JAVA_GZIP_HEADER`], a raw deflate stream at level 6, CRC-32 and size (little-endian).
///
/// # Errors
///
/// [`NbtError::Message`] if the deflater fails, which it does not for in-memory input.
pub fn gzip_java(raw: &[u8]) -> Result<Vec<u8>, NbtError> {
    let mut out = Vec::with_capacity(raw.len() / 2 + 32);
    out.extend_from_slice(&JAVA_GZIP_HEADER);
    deflate_raw(raw, &mut out)?;
    let mut crc = Crc::new();
    crc.update(raw);
    out.extend_from_slice(&crc.sum().to_le_bytes());
    out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    Ok(out)
}

/// A raw deflate stream (no zlib or gzip wrapper) at the JDK's default level, appended to `out`.
fn deflate_raw(input: &[u8], out: &mut Vec<u8>) -> Result<(), NbtError> {
    let mut deflate = Compress::new(Compression::new(JAVA_DEFAULT_LEVEL), false);
    loop {
        out.reserve(input.len() / 2 + 64);
        let in_before = deflate.total_in() as usize;
        let out_before = deflate.total_out();
        let status = deflate
            .compress_vec(&input[in_before..], out, FlushCompress::Finish)
            .map_err(|e| NbtError::Message(format!("deflate failed: {e}")))?;
        if matches!(status, Status::StreamEnd) {
            return Ok(());
        }
        if deflate.total_in() as usize == in_before && deflate.total_out() == out_before {
            return Err(NbtError::Message("deflate made no progress".to_owned()));
        }
    }
}

/// Parse a gzip member header as `GZIPInputStream.readHeader` (JDK 8) does and return its
/// length. Reserved flag bits, MTIME, XFL and OS are ignored; FEXTRA, FNAME and FCOMMENT are
/// skipped; FHCRC is checked against the CRC-32 of the header bytes before it.
fn parse_gzip_header(input: &[u8]) -> Result<usize, &'static str> {
    let byte = |i: usize| input.get(i).copied().ok_or("Unexpected end of GZIP header");
    if byte(0)? != 0x1F || byte(1)? != 0x8B {
        return Err("Not in GZIP format");
    }
    if byte(2)? != 8 {
        return Err("Unsupported compression method");
    }
    let flags = byte(3)?;
    // MTIME, XFL and OS end at byte 9.
    byte(9)?;
    let mut n = 10;
    if flags & FEXTRA != 0 {
        let len = usize::from(byte(n)?) | usize::from(byte(n + 1)?) << 8;
        n += 2;
        if input.len() < n + len {
            return Err("Unexpected end of GZIP header");
        }
        n += len;
    }
    if flags & FNAME != 0 {
        while byte(n)? != 0 {
            n += 1;
        }
        n += 1;
    }
    if flags & FCOMMENT != 0 {
        while byte(n)? != 0 {
            n += 1;
        }
        n += 1;
    }
    if flags & FHCRC != 0 {
        let mut crc = Crc::new();
        crc.update(&input[..n]);
        let expected = (crc.sum() & 0xFFFF) as u16;
        let got = u16::from(byte(n)?) | u16::from(byte(n + 1)?) << 8;
        if got != expected {
            return Err("Corrupt GZIP header");
        }
        n += 2;
    }
    Ok(n)
}

fn eof(what: &'static str) -> NbtError {
    NbtError::Io(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        format!("NBT stream ended while reading {what}"),
    ))
}

/// The inflated content of a gzip blob, pulled on demand with `GZIPInputStream`'s semantics.
struct GzipStream<'a, 'b> {
    input: &'a [u8],
    /// Position of the next deflate byte the inflater has not consumed.
    pos: usize,
    /// Where the JDK's 512-byte input refills are aligned for the current member.
    align_base: usize,
    inflate: Decompress,
    crc: Crc,
    member_out: u64,
    done: bool,
    buf: Vec<u8>,
    buf_pos: usize,
    budget: &'b mut NbtBudget,
}

impl<'a, 'b> GzipStream<'a, 'b> {
    /// java: `new GZIPInputStream(new ByteArrayInputStream(blob))`, whose constructor reads the
    /// header.
    fn open(input: &'a [u8], budget: &'b mut NbtBudget) -> Result<GzipStream<'a, 'b>, ProtoError> {
        let header = parse_gzip_header(input).map_err(|m| ProtoError::Zlib(m.to_owned()))?;
        Ok(GzipStream {
            input,
            pos: header,
            align_base: header,
            inflate: Decompress::new(false),
            crc: Crc::new(),
            member_out: 0,
            done: false,
            buf: Vec::new(),
            buf_pos: 0,
            budget,
        })
    }

    /// Inflate the next chunk into `buf`. `Ok(false)` is the end of the data.
    fn refill(&mut self) -> std::io::Result<bool> {
        self.buf.clear();
        self.buf.resize(INFLATE_CHUNK, 0);
        self.buf_pos = 0;
        loop {
            if self.done {
                self.buf.clear();
                return Ok(false);
            }
            let before_in = self.inflate.total_in();
            let before_out = self.inflate.total_out();
            let status = self
                .inflate
                .decompress(
                    &self.input[self.pos..],
                    &mut self.buf,
                    FlushDecompress::None,
                )
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let consumed = (self.inflate.total_in() - before_in) as usize;
            let produced = (self.inflate.total_out() - before_out) as usize;
            self.pos += consumed;
            if produced > 0 {
                self.budget.take_inflated(produced)?;
                self.buf.truncate(produced);
                self.crc.update(&self.buf);
                self.member_out += produced as u64;
                return Ok(true);
            }
            match status {
                Status::StreamEnd => self.end_of_member()?,
                Status::Ok | Status::BufError if consumed > 0 => {}
                Status::Ok | Status::BufError => {
                    // java: InflaterInputStream.fill() on an exhausted ByteArrayInputStream.
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "Unexpected end of ZLIB input stream",
                    ));
                }
            }
        }
    }

    /// java: `GZIPInputStream.readTrailer`, reached only when more data is asked for after a
    /// member's deflate stream has ended.
    fn end_of_member(&mut self) -> std::io::Result<()> {
        let end = self.pos;
        let total = self.input.len();
        // `n` is what the inflater still held of its current 512-byte refill.
        let offset = end - self.align_base;
        let load_end = (self.align_base
            + offset.div_ceil(JAVA_INFLATER_BUFFER) * JAVA_INFLATER_BUFFER)
            .min(total);
        let n = load_end - end;

        let trailer = self
            .input
            .get(end..end + 8)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "EOF"))?;
        let crc = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
        let size = u32::from_le_bytes([trailer[4], trailer[5], trailer[6], trailer[7]]);
        if crc != self.crc.sum() || size != self.member_out as u32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Corrupt GZIP trailer",
            ));
        }

        // The trailer may have been read partly past the refill, out of the byte-array stream,
        // before the JDK asks that stream for `available()`.
        let available = total - load_end.max(end + 8);
        if available == 0 && n <= 26 {
            self.done = true;
            return Ok(());
        }
        match parse_gzip_header(&self.input[end + 8..]) {
            // java: "ignore any malformed, do nothing"
            Err(_) => self.done = true,
            Ok(header) => {
                let m = 8 + header;
                self.pos = end + m;
                if n <= m {
                    // The trailer and header ran past the refill; the next refill starts here.
                    self.align_base = self.pos;
                }
                self.inflate.reset(false);
                self.crc.reset();
                self.member_out = 0;
            }
        }
        Ok(())
    }
}

impl Read for GzipStream<'_, '_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.buf_pos >= self.buf.len() && !self.refill()? {
            return Ok(0);
        }
        let n = out.len().min(self.buf.len() - self.buf_pos);
        out[..n].copy_from_slice(&self.buf[self.buf_pos..self.buf_pos + n]);
        self.buf_pos += n;
        Ok(n)
    }
}

/// java17: `CompressedStreamTools.func_152456_a` and `func_152455_a`.
fn read_root_compound(stream: &mut GzipStream<'_, '_>) -> Result<NbtCompound, NbtError> {
    let mut reader = TagReader {
        inner: stream,
        space_read: 0,
    };
    let id = reader.read_u8("a root tag type")?;
    if id == 0 {
        // java: func_152455_a returns NBTTagEnd -> "Root tag must be a named compound tag".
        return Err(NbtError::UnexpectedType {
            expected: TagType::Compound,
            found: TagType::End,
        });
    }
    let _name = reader.read_utf()?;
    let ty = TagType::from_id(id).ok_or(NbtError::UnknownTagType(id))?;
    match reader.read_payload(ty)? {
        Nbt::Compound(compound) => Ok(compound),
        other => Err(NbtError::UnexpectedType {
            expected: TagType::Compound,
            found: other.tag_type(),
        }),
    }
}

/// An empty list that reports `elem_id` as its element type, as java17 `NBTTagList.read` keeps
/// the type byte it read (`getTagType()`, the `appendTag` check and `equals` see it). `NbtList`
/// takes its type from its first element and keeps it when emptied; an id that is no tag type,
/// which Java also keeps, has no `NbtList` form and becomes `TAG_End`.
fn empty_list_of(elem_id: u8) -> NbtList {
    let placeholder = match TagType::from_id(elem_id) {
        None | Some(TagType::End) => return NbtList::new(),
        Some(TagType::Byte) => Nbt::Byte(0),
        Some(TagType::Short) => Nbt::Short(0),
        Some(TagType::Int) => Nbt::Int(0),
        Some(TagType::Long) => Nbt::Long(0),
        Some(TagType::Float) => Nbt::Float(0.0),
        Some(TagType::Double) => Nbt::Double(0.0),
        Some(TagType::ByteArray) => Nbt::ByteArray(Vec::new()),
        Some(TagType::String) => Nbt::String(String::new()),
        Some(TagType::List) => Nbt::List(NbtList::new()),
        Some(TagType::Compound) => Nbt::Compound(NbtCompound::new()),
        Some(TagType::IntArray) => Nbt::IntArray(Vec::new()),
    };
    let mut list = NbtList::from(vec![placeholder]);
    list.clear();
    list
}

/// One 1.7.10 tag read in progress.
struct TagReader<'s, 'a, 'b> {
    inner: &'s mut GzipStream<'a, 'b>,
    /// java17: `NBTSizeTracker.spaceRead`
    space_read: u64,
}

/// A container whose children are still being read.
enum Frame {
    List {
        elem: TagType,
        remaining: u32,
        items: Vec<Nbt>,
        depth: u32,
    },
    Compound {
        compound: NbtCompound,
        key: Option<String>,
        depth: u32,
    },
}

impl TagReader<'_, '_, '_> {
    /// java17: `NBTSizeTracker.addSpaceRead(bits)`: `spaceRead += bits / 8`, then the limit.
    ///
    /// Java computes the array charges as `(long)(8 * n)` in `int` arithmetic; that product only
    /// wraps for lengths no blob can satisfy (and whose allocation fails first in Java), so the
    /// honest widened charge gives the same verdict.
    fn charge_bits(&mut self, bits: u64) -> Result<(), NbtError> {
        self.space_read = self.space_read.saturating_add(bits / 8);
        if self.space_read > SIZE_TRACKER_LIMIT {
            return Err(NbtError::SizeLimit {
                limit: SIZE_TRACKER_LIMIT,
                requested: self.space_read,
            });
        }
        Ok(())
    }

    fn read_exact_n<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], NbtError> {
        let mut buf = [0u8; N];
        match self.inner.read_exact(&mut buf) {
            Ok(()) => Ok(buf),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Err(eof(what)),
            Err(e) => Err(NbtError::Io(e)),
        }
    }

    fn read_u8(&mut self, what: &'static str) -> Result<u8, NbtError> {
        Ok(self.read_exact_n::<1>(what)?[0])
    }

    fn read_i32(&mut self, what: &'static str) -> Result<i32, NbtError> {
        Ok(i32::from_be_bytes(self.read_exact_n::<4>(what)?))
    }

    /// java: `DataInputStream.readUTF`
    fn read_utf(&mut self) -> Result<String, NbtError> {
        let len = usize::from(u16::from_be_bytes(
            self.read_exact_n::<2>("a string length")?,
        ));
        // Bounded by the u16 prefix.
        let mut buf = vec![0u8; len];
        match self.inner.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(eof("a string payload"))
            }
            Err(e) => return Err(NbtError::Io(e)),
        }
        Ok(modified_utf8::decode(&buf)?)
    }

    /// `DataInput.readFully` into a buffer that grows only as bytes arrive.
    fn read_untrusted(&mut self, n: usize, what: &'static str) -> Result<Vec<u8>, NbtError> {
        let mut buf = Vec::new();
        let got = (&mut *self.inner).take(n as u64).read_to_end(&mut buf)?;
        if got != n {
            return Err(eof(what));
        }
        Ok(buf)
    }

    fn read_len(&mut self, kind: TagType) -> Result<usize, NbtError> {
        let len = self.read_i32("an array length")?;
        // java: `new byte[len]` / `new int[len]` throws NegativeArraySizeException.
        usize::try_from(len).map_err(|_| NbtError::NegativeLength { kind, len })
    }

    fn scalar(&mut self, ty: TagType) -> Result<Nbt, NbtError> {
        Ok(match ty {
            TagType::Byte => {
                self.charge_bits(8)?;
                Nbt::Byte(self.read_exact_n::<1>("a TAG_Byte")?[0] as i8)
            }
            TagType::Short => {
                self.charge_bits(16)?;
                Nbt::Short(i16::from_be_bytes(self.read_exact_n::<2>("a TAG_Short")?))
            }
            TagType::Int => {
                self.charge_bits(32)?;
                Nbt::Int(i32::from_be_bytes(self.read_exact_n::<4>("a TAG_Int")?))
            }
            TagType::Long => {
                self.charge_bits(64)?;
                Nbt::Long(i64::from_be_bytes(self.read_exact_n::<8>("a TAG_Long")?))
            }
            TagType::Float => {
                self.charge_bits(32)?;
                Nbt::Float(f32::from_be_bytes(self.read_exact_n::<4>("a TAG_Float")?))
            }
            TagType::Double => {
                self.charge_bits(64)?;
                Nbt::Double(f64::from_be_bytes(self.read_exact_n::<8>("a TAG_Double")?))
            }
            TagType::ByteArray => {
                let n = self.read_len(TagType::ByteArray)?;
                self.charge_bits(8u64.saturating_mul(n as u64))?;
                Nbt::ByteArray(self.read_untrusted(n, "a TAG_Byte_Array payload")?)
            }
            TagType::String => {
                let s = self.read_utf()?;
                self.charge_bits(16u64.saturating_mul(modified_utf8::utf16_len(&s) as u64))?;
                Nbt::String(s)
            }
            TagType::IntArray => {
                let n = self.read_len(TagType::IntArray)?;
                self.charge_bits(32u64.saturating_mul(n as u64))?;
                let bytes = n
                    .checked_mul(4)
                    .ok_or(NbtError::NegativeLength {
                        kind: TagType::IntArray,
                        len: i32::MAX,
                    })
                    .and_then(|bytes| self.read_untrusted(bytes, "a TAG_Int_Array payload"))?;
                Nbt::IntArray(
                    bytes
                        .chunks_exact(4)
                        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            }
            // java17: NBTTagEnd.read reads nothing; only a TAG_End list element gets here.
            TagType::End | TagType::List | TagType::Compound => Nbt::End,
        })
    }

    /// Read one payload of type `ty` read at depth 0. Open containers are held on a heap stack,
    /// so a 512-deep tag cannot exhaust the thread stack.
    fn read_payload(&mut self, ty: TagType) -> Result<Nbt, NbtError> {
        let mut frames: Vec<Frame> = Vec::new();
        let mut next: Option<(TagType, u32)> = Some((ty, 0));
        let mut value: Option<Nbt> = None;
        loop {
            // Hand a finished value to the innermost open container.
            if let Some(child) = value.take() {
                match frames.last_mut() {
                    None => return Ok(child),
                    Some(Frame::List {
                        elem,
                        remaining,
                        items,
                        depth,
                    }) => {
                        items.push(child);
                        *remaining -= 1;
                        if *remaining == 0 {
                            let Some(Frame::List { items, .. }) = frames.pop() else {
                                unreachable!("the innermost frame is this list");
                            };
                            value = Some(Nbt::List(NbtList::from(items)));
                        } else {
                            next = Some((*elem, *depth + 1));
                        }
                    }
                    Some(Frame::Compound { compound, key, .. }) => {
                        let key = key.take().unwrap_or_default();
                        // java: HashMap.put; a repeated key replaces the earlier value.
                        compound.insert(key, child);
                    }
                }
                continue;
            }

            // Read the next scheduled tag.
            if let Some((ty, depth)) = next.take() {
                self.inner.budget.take_node()?;
                match ty {
                    TagType::List => {
                        if depth > MAX_DEPTH {
                            return Err(NbtError::DepthLimit { limit: MAX_DEPTH });
                        }
                        self.charge_bits(8)?;
                        let elem_id = self.read_u8("a list element type")?;
                        let count = self.read_i32("a list length")?;
                        if count <= 0 {
                            // java: the element loop never runs, whatever the type byte says,
                            // and the list keeps that byte as its type.
                            value = Some(Nbt::List(empty_list_of(elem_id)));
                        } else {
                            let elem = TagType::from_id(elem_id)
                                .ok_or(NbtError::UnknownTagType(elem_id))?;
                            frames.push(Frame::List {
                                elem,
                                remaining: count as u32,
                                items: Vec::new(),
                                depth,
                            });
                            next = Some((elem, depth + 1));
                        }
                    }
                    TagType::Compound => {
                        if depth > MAX_DEPTH {
                            return Err(NbtError::DepthLimit { limit: MAX_DEPTH });
                        }
                        frames.push(Frame::Compound {
                            compound: NbtCompound::new(),
                            key: None,
                            depth,
                        });
                    }
                    scalar => value = Some(self.scalar(scalar)?),
                }
                continue;
            }

            // Nothing to deliver or read: the innermost frame is a compound awaiting its next
            // entry (a list always schedules its next element).
            let Some(Frame::Compound { key, depth, .. }) = frames.last_mut() else {
                unreachable!("only a compound waits for input");
            };
            let depth = *depth;
            let id = self.read_u8("a compound entry type")?;
            if id == 0 {
                let Some(Frame::Compound { compound, .. }) = frames.pop() else {
                    unreachable!("the innermost frame is this compound");
                };
                value = Some(Nbt::Compound(compound));
            } else {
                let name = self.read_utf()?;
                // java17: NBTTagCompound.read charges 16 bits per key char and nothing else.
                self.charge_bits(16u64.saturating_mul(modified_utf8::utf16_len(&name) as u64))?;
                let ty = TagType::from_id(id).ok_or(NbtError::UnknownTagType(id))?;
                *key = Some(name);
                next = Some((ty, depth + 1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gzip(raw: &[u8]) -> Vec<u8> {
        gzip_java(raw).unwrap()
    }

    fn read(blob: &[u8]) -> Result<NbtCompound, ProtoError> {
        read_gzip_compound(blob, &mut NbtBudget::per_packet())
    }

    fn sample() -> NbtCompound {
        let mut display = NbtCompound::new();
        display.insert("Name", Nbt::String("Sword".into()));
        let mut ench = NbtCompound::new();
        ench.insert("id", Nbt::Short(16));
        ench.insert("lvl", Nbt::Short(5));
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        root.insert("ench", Nbt::list(vec![Nbt::Compound(ench)]));
        root.insert("bytes", Nbt::ByteArray(vec![1, 2, 3]));
        root.insert("ints", Nbt::IntArray(vec![-1, 7]));
        root.insert("long", Nbt::Long(-5));
        root.insert("float", Nbt::Float(0.5));
        root.insert("double", Nbt::Double(-0.25));
        root.insert("byte", Nbt::Byte(-1));
        root
    }

    #[test]
    fn writes_javas_header_and_trailer_and_reads_back() {
        let root = sample();
        let blob = write_gzip_compound(&root).unwrap();
        assert_eq!(&blob[..10], &JAVA_GZIP_HEADER);
        let mut raw = Vec::new();
        mc_nbt::write_compound(&mut raw, "", &root).unwrap();
        let mut crc = Crc::new();
        crc.update(&raw);
        let n = blob.len();
        assert_eq!(&blob[n - 8..n - 4], &crc.sum().to_le_bytes());
        assert_eq!(&blob[n - 4..], &(raw.len() as u32).to_le_bytes());
        assert_eq!(read(&blob).unwrap(), root);
    }

    #[test]
    fn header_is_checked_like_gzipinputstream() {
        let blob = write_gzip_compound(&sample()).unwrap();
        let mut bad_magic = blob.clone();
        bad_magic[1] = 0x8C;
        assert!(matches!(read(&bad_magic), Err(ProtoError::Zlib(m)) if m == "Not in GZIP format"));
        let mut bad_method = blob.clone();
        bad_method[2] = 7;
        assert!(read(&bad_method).is_err());
        assert!(
            read(&[]).is_err(),
            "an empty blob is EOF in the constructor"
        );
        assert!(read(&blob[..9]).is_err());

        // FNAME, FCOMMENT, FEXTRA and a correct FHCRC are skipped; reserved bits are ignored.
        let mut fancy = vec![0x1F, 0x8B, 8, FNAME | FCOMMENT | FEXTRA | FHCRC | 0xE0];
        fancy.extend_from_slice(&[1, 2, 3, 4, 0, 3]);
        fancy.extend_from_slice(&[2, 0, 0xAA, 0xBB]);
        fancy.extend_from_slice(b"name\0comment\0");
        let mut crc = Crc::new();
        crc.update(&fancy);
        fancy.extend_from_slice(&((crc.sum() & 0xFFFF) as u16).to_le_bytes());
        let header_crc_at = fancy.len() - 2;
        fancy.extend_from_slice(&blob[10..]);
        assert_eq!(read(&fancy).unwrap(), sample());
        fancy[header_crc_at] ^= 1;
        assert!(matches!(read(&fancy), Err(ProtoError::Zlib(m)) if m == "Corrupt GZIP header"));
    }

    #[test]
    fn the_trailer_is_never_read_for_a_complete_tag() {
        let blob = write_gzip_compound(&sample()).unwrap();
        let n = blob.len();
        let mut bad_crc = blob.clone();
        bad_crc[n - 5] ^= 0xFF;
        assert_eq!(
            read(&bad_crc).unwrap(),
            sample(),
            "java never reaches the trailer"
        );
        assert_eq!(read(&blob[..n - 8]).unwrap(), sample(), "nor needs one");
        let mut trailing = blob.clone();
        trailing.extend_from_slice(&[0xDE; 40]);
        assert_eq!(read(&trailing).unwrap(), sample());
    }

    #[test]
    fn a_tag_split_across_two_members_is_read_when_java_would_try_the_second() {
        let mut raw = Vec::new();
        mc_nbt::write_compound(&mut raw, "", &sample()).unwrap();
        let (a, b) = raw.split_at(raw.len() / 2);
        let first = gzip(a);
        let mut blob = first.clone();
        blob.extend_from_slice(&gzip(b));
        // More than 26 bytes follow the first deflate stream, so the JDK reads the next header.
        assert_eq!(read(&blob).unwrap(), sample());

        // A corrupt first trailer is noticed exactly in this case.
        blob[first.len() - 6] ^= 1;
        assert!(read(&blob).is_err());

        // Without a second member the reader runs dry.
        assert!(read(&first).is_err());
    }

    #[test]
    fn a_truncated_deflate_stream_is_an_error() {
        let blob = write_gzip_compound(&sample()).unwrap();
        assert!(read(&blob[..14]).is_err());
    }

    #[test]
    fn root_must_be_a_compound() {
        assert!(matches!(
            read(&gzip(&[0])),
            Err(ProtoError::Nbt(NbtError::UnexpectedType {
                found: TagType::End,
                ..
            }))
        ));
        // A named TAG_Int root is read, then refused.
        assert!(matches!(
            read(&gzip(&[3, 0, 0, 0, 0, 0, 7])),
            Err(ProtoError::Nbt(NbtError::UnexpectedType {
                found: TagType::Int,
                ..
            }))
        ));
        assert!(matches!(
            read(&gzip(&[12, 0, 0])),
            Err(ProtoError::Nbt(NbtError::UnknownTagType(12)))
        ));
    }

    /// java17: `NBTTagList.read` never checks its count, nor its type unless it has elements.
    #[test]
    fn lists_follow_1_7_10_not_1_8_9() {
        // root { "l": list of strings, count -5 } is an empty list.
        let raw = [10, 0, 0, 9, 0, 1, b'l', 8, 0xFF, 0xFF, 0xFF, 0xFB, 0];
        let root = read(&gzip(&raw)).unwrap();
        assert_eq!(root.get_list("l").map(<[Nbt]>::len), Some(0));
        assert!(
            mc_nbt::from_slice(&raw).is_err(),
            "1.8.9 rejects the negative count"
        );
        // An unknown element type with no elements is fine.
        let raw = [10, 0, 0, 9, 0, 1, b'l', 99, 0, 0, 0, 0, 0];
        assert!(read(&gzip(&raw)).is_ok());
        // An unknown element type with elements is java's NullPointerException.
        let raw = [10, 0, 0, 9, 0, 1, b'l', 99, 0, 0, 0, 1, 0];
        assert!(read(&gzip(&raw)).is_err());
        // TAG_End elements are accepted; 1.8.9 says "Missing type on ListTag".
        let raw = [10, 0, 0, 9, 0, 1, b'l', 0, 0, 0, 0, 3, 0];
        let root = read(&gzip(&raw)).unwrap();
        assert_eq!(root.get_list("l").map(<[Nbt]>::len), Some(3));
        assert!(mc_nbt::from_slice(&raw).is_err());
    }

    /// java17 `NBTTagList.read` keeps the type byte even when it reads no element.
    #[test]
    fn an_empty_list_keeps_its_declared_type() {
        let declared = |elem: u8| {
            let raw = [10, 0, 0, 9, 0, 1, b'l', elem, 0, 0, 0, 0, 0];
            let root = read(&gzip(&raw)).unwrap();
            match root.get("l") {
                Some(Nbt::List(list)) => (list.len(), list.tag_type()),
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(declared(8), (0, TagType::String));
        assert_eq!(declared(10), (0, TagType::Compound));
        assert_eq!(declared(0), (0, TagType::End));
        // An id that is no tag type: Java keeps the byte, NbtList cannot.
        assert_eq!(declared(99), (0, TagType::End));
        // Negative control: equality sees the type, as NBTTagList.equals does.
        let mut strings = NbtCompound::new();
        strings.insert("l", Nbt::List(empty_list_of(8)));
        let mut ends = NbtCompound::new();
        ends.insert("l", Nbt::List(NbtList::new()));
        assert_ne!(strings, ends);
        // Both write an empty list as type 0, as java17 NBTTagList.write does.
        assert_eq!(
            write_compound_java17(&strings).unwrap(),
            write_compound_java17(&ends).unwrap()
        );
    }

    /// A `TAG_End` list read from a server goes back the way java17 writes it, where the 1.8.9
    /// writer refuses it.
    #[test]
    fn end_lists_are_written_the_java_way() {
        let raw = [10, 0, 0, 9, 0, 1, b'l', 0, 0, 0, 0, 3, 0];
        let root = read(&gzip(&raw)).unwrap();
        assert_eq!(write_compound_java17(&root).unwrap(), raw);
        assert!(mc_nbt::write_compound(&mut Vec::new(), "", &root).is_err());
        assert_eq!(read(&write_gzip_compound(&root).unwrap()).unwrap(), root);
    }

    /// The v5 writer is mc_nbt's for everything mc_nbt can write.
    #[test]
    fn the_java17_writer_matches_mc_nbt() {
        let mut nested = NbtCompound::new();
        nested.insert(
            "deep",
            Nbt::list(vec![Nbt::list(vec![Nbt::IntArray(vec![1, -2])])]),
        );
        let mut root = sample();
        root.insert("nested", Nbt::Compound(nested));
        root.insert("empty", Nbt::list(Vec::new()));
        root.insert("unicode", Nbt::String("\u{0}\u{e9}\u{1F600}".into()));
        let mut expected = Vec::new();
        mc_nbt::write_compound(&mut expected, "", &root).unwrap();
        assert_eq!(write_compound_java17(&root).unwrap(), expected);

        // Both refuse what Java cannot write.
        let mut bad = NbtCompound::new();
        bad.insert("x".repeat(65_536), Nbt::Byte(1));
        assert!(write_compound_java17(&bad).is_err());
        let mut mixed = NbtCompound::new();
        mixed.insert("m", Nbt::list(vec![Nbt::Byte(1), Nbt::Short(2)]));
        assert!(write_compound_java17(&mixed).is_err());
    }

    #[test]
    fn size_tracker_uses_1_7_10_charges() {
        // A byte array charges exactly its length: 2097152 fits, 2097153 does not.
        for (len, ok) in [(2_097_152u32, true), (2_097_153, false)] {
            let mut raw = vec![10, 0, 0, 7, 0, 0];
            raw.extend_from_slice(&len.to_be_bytes());
            raw.resize(raw.len() + len as usize, 0);
            raw.push(0);
            assert_eq!(read(&gzip(&raw)).is_ok(), ok, "byte array of {len}");
            // Negative control: 1.8.9's charges (compound 384 bits, array 192 + 8 per byte)
            // reject the same 2097152-byte array.
            assert!(mc_nbt::read_with_limits(&mut &raw[..], mc_nbt::NbtLimits::PROTOCOL).is_err());
        }
    }

    #[test]
    fn depth_is_capped_at_512() {
        // n compounds nested under the key "" inside the root: the innermost is at depth n.
        let nested = |n: usize| {
            let mut raw = vec![10, 0, 0];
            for _ in 0..n {
                raw.extend_from_slice(&[10, 0, 0]);
            }
            raw.extend(std::iter::repeat_n(0, n + 1));
            raw
        };
        assert!(read(&gzip(&nested(512))).is_ok());
        assert!(matches!(
            read(&gzip(&nested(513))),
            Err(ProtoError::Nbt(NbtError::DepthLimit { limit: 512 }))
        ));
    }

    #[test]
    fn hostile_lengths_do_not_allocate() {
        // A byte array claiming 2 GB with nothing behind it fails on the tracker, not malloc.
        let raw = [10, 0, 0, 7, 0, 0, 0x7F, 0xFF, 0xFF, 0xFF, 0];
        assert!(read(&gzip(&raw)).is_err());
        // An int array claiming 2^26 elements: the tracker again.
        let raw = [10, 0, 0, 11, 0, 0, 0x04, 0, 0, 0, 0];
        assert!(read(&gzip(&raw)).is_err());
        // A list of 2^31-1 TAG_End elements costs java nothing per element; the packet budget
        // stops it.
        let raw = [10, 0, 0, 9, 0, 0, 0, 0x7F, 0xFF, 0xFF, 0xFF, 0];
        let mut budget = NbtBudget {
            nodes_left: 10_000,
            inflated_left: u64::MAX,
        };
        assert!(matches!(
            read_gzip_compound(&gzip(&raw), &mut budget),
            Err(ProtoError::Nbt(NbtError::Message(_)))
        ));
    }

    #[test]
    fn the_packet_budget_spans_several_tags() {
        let blob = write_gzip_compound(&sample()).unwrap();
        let mut budget = NbtBudget {
            nodes_left: 20,
            inflated_left: u64::MAX,
        };
        assert!(read_gzip_compound(&blob, &mut budget).is_ok());
        assert!(read_gzip_compound(&blob, &mut budget).is_err());
        let mut budget = NbtBudget {
            nodes_left: u64::MAX,
            inflated_left: 10,
        };
        assert!(read_gzip_compound(&blob, &mut budget).is_err());
    }
}
