//! Decoding NBT from a byte stream, with Java `DataInput` semantics.
//!
//! java: `CompressedStreamTools.func_152455_a`, plus each `NBTTag*.read`

use std::io::Read;

use crate::error::NbtError;
use crate::limits::{NbtLimits, SizeTracker};
use crate::modified_utf8::{self, MAX_ENCODED_LEN};
use crate::tag::{Nbt, NbtCompound, NbtList, TagType};

/// Read one named root tag, applying [`NbtLimits::PROTOCOL`].
///
/// Returns the root's name (usually `""`) and its value. A `TAG_End` root — a lone zero
/// byte — yields `("", Nbt::End)` and consumes exactly one byte, matching vanilla.
///
/// The reader consumes exactly the bytes of this tag and no more, so it is safe to call on
/// a stream that continues with something else (a packet buffer, say). It performs many
/// small reads: wrap a `File` in a `BufReader` first.
///
/// ```
/// # use mc_nbt::{read, Nbt};
/// let bytes = [0x0a, 0x00, 0x00, 0x03, 0x00, 0x01, b'x', 0, 0, 0, 7, 0x00];
/// let (name, value) = read(&mut &bytes[..]).unwrap();
/// assert_eq!(name, "");
/// assert_eq!(value.as_compound().unwrap().get_i32("x"), Some(7));
/// ```
pub fn read(r: &mut impl Read) -> Result<(String, Nbt), NbtError> {
    read_with_limits(r, NbtLimits::default())
}

/// Read one named root tag with explicit limits.
///
/// Use [`NbtLimits::UNLIMITED`] for trusted files on disk — that is what vanilla's
/// `CompressedStreamTools` does.
pub fn read_with_limits(r: &mut impl Read, limits: NbtLimits) -> Result<(String, Nbt), NbtError> {
    let mut reader = Reader {
        inner: r,
        tracker: SizeTracker::new(limits),
        limits,
    };
    let id = reader.read_u8()?;
    let ty = TagType::from_id(id).ok_or(NbtError::UnknownTagType(id))?;
    if ty == TagType::End {
        // java: func_152455_a returns NBTTagEnd without reading a name.
        return Ok((String::new(), Nbt::End));
    }
    let name = reader.read_string()?;
    let value = reader.read_payload(ty, 0)?;
    Ok((name, value))
}

/// Read a root tag and require it to be a compound.
///
/// java: `CompressedStreamTools.read` — "Root tag must be a named compound tag"
pub fn read_compound(r: &mut impl Read) -> Result<(String, NbtCompound), NbtError> {
    read_compound_with_limits(r, NbtLimits::default())
}

/// [`read_compound`] with explicit limits.
pub fn read_compound_with_limits(
    r: &mut impl Read,
    limits: NbtLimits,
) -> Result<(String, NbtCompound), NbtError> {
    let (name, value) = read_with_limits(r, limits)?;
    match value {
        Nbt::Compound(c) => Ok((name, c)),
        other => Err(NbtError::UnexpectedType {
            expected: TagType::Compound,
            found: other.tag_type(),
        }),
    }
}

/// Read a root tag from a byte slice, applying [`NbtLimits::PROTOCOL`].
pub fn from_slice(bytes: &[u8]) -> Result<(String, Nbt), NbtError> {
    read(&mut &bytes[..])
}

/// [`from_slice`] with explicit limits.
pub fn from_slice_with_limits(bytes: &[u8], limits: NbtLimits) -> Result<(String, Nbt), NbtError> {
    read_with_limits(&mut &bytes[..], limits)
}

struct Reader<'a, R: Read> {
    inner: &'a mut R,
    tracker: SizeTracker,
    limits: NbtLimits,
}

impl<R: Read> Reader<'_, R> {
    fn read_exact_n<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], NbtError> {
        let mut buf = [0u8; N];
        match self.inner.read_exact(&mut buf) {
            Ok(()) => Ok(buf),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Err(NbtError::eof(what)),
            Err(e) => Err(NbtError::Io(e)),
        }
    }

    fn read_u8(&mut self) -> Result<u8, NbtError> {
        Ok(self.read_exact_n::<1>("a tag type byte")?[0])
    }

    fn read_i8(&mut self) -> Result<i8, NbtError> {
        Ok(self.read_exact_n::<1>("a TAG_Byte")?[0] as i8)
    }

    fn read_u16(&mut self) -> Result<u16, NbtError> {
        Ok(u16::from_be_bytes(
            self.read_exact_n::<2>("a string length")?,
        ))
    }

    fn read_i16(&mut self) -> Result<i16, NbtError> {
        Ok(i16::from_be_bytes(self.read_exact_n::<2>("a TAG_Short")?))
    }

    fn read_i32(&mut self) -> Result<i32, NbtError> {
        Ok(i32::from_be_bytes(self.read_exact_n::<4>("a TAG_Int")?))
    }

    fn read_i64(&mut self) -> Result<i64, NbtError> {
        Ok(i64::from_be_bytes(self.read_exact_n::<8>("a TAG_Long")?))
    }

    fn read_f32(&mut self) -> Result<f32, NbtError> {
        Ok(f32::from_be_bytes(self.read_exact_n::<4>("a TAG_Float")?))
    }

    fn read_f64(&mut self) -> Result<f64, NbtError> {
        Ok(f64::from_be_bytes(self.read_exact_n::<8>("a TAG_Double")?))
    }

    /// java: `DataInputStream.readUTF` — u16 *byte* length, then modified UTF-8.
    fn read_string(&mut self) -> Result<String, NbtError> {
        let len = usize::from(self.read_u16()?);
        // Bounded by the u16 prefix at 64 KiB, so sizing this allocation is safe.
        debug_assert!(len <= MAX_ENCODED_LEN);
        let mut buf = vec![0u8; len];
        match self.inner.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(NbtError::eof("a string payload"))
            }
            Err(e) => return Err(NbtError::Io(e)),
        }
        Ok(modified_utf8::decode(&buf)?)
    }

    /// Number of elements a container declared, rejecting negatives the way a Rust
    /// allocation would have to anyway.
    fn read_len(&mut self, kind: TagType) -> Result<usize, NbtError> {
        let len = self.read_i32()?;
        usize::try_from(len).map_err(|_| NbtError::NegativeLength { kind, len })
    }

    /// Fill `n` bytes from the stream without trusting `n` enough to pre-allocate it.
    ///
    /// `Read::take` + `read_to_end` grows the buffer as data actually arrives, so a
    /// hostile length header costs a failed read rather than a 2 GB allocation.
    fn read_bytes_untrusted(&mut self, n: usize, what: &'static str) -> Result<Vec<u8>, NbtError> {
        let mut buf = Vec::new();
        let got = self.inner.by_ref().take(n as u64).read_to_end(&mut buf)?;
        if got != n {
            return Err(NbtError::eof(what));
        }
        Ok(buf)
    }

    fn check_depth(&self, depth: u32) -> Result<(), NbtError> {
        // java: `if (depth > 512) throw` — the root is read at depth 0.
        if depth > self.limits.max_depth {
            Err(NbtError::DepthLimit {
                limit: self.limits.max_depth,
            })
        } else {
            Ok(())
        }
    }

    fn read_payload(&mut self, ty: TagType, depth: u32) -> Result<Nbt, NbtError> {
        // The old implementation recursively called this method for every child. That made a
        // malicious but otherwise well-formed chain of 513 compounds overflow the Windows test
        // stack while unwinding, before the depth error could be returned. Keep the Java depth
        // boundary, but hold pending containers on the heap instead.
        enum Frame {
            List {
                elem: TagType,
                remaining: usize,
                items: Vec<Nbt>,
                depth: u32,
            },
            Compound {
                compound: NbtCompound,
                pending_key: Option<String>,
                depth: u32,
            },
        }

        let mut frames = Vec::new();
        let mut next = Some((ty, depth));
        let mut value = None;

        loop {
            if let Some((ty, depth)) = next.take() {
                // The per-tag overheads below are vanilla's, verbatim; see limits.rs.
                let parsed = match ty {
                    TagType::End => {
                        self.tracker.charge_bits(64)?;
                        Some(Nbt::End)
                    }
                    TagType::Byte => {
                        self.tracker.charge_bits(72)?;
                        Some(Nbt::Byte(self.read_i8()?))
                    }
                    TagType::Short => {
                        self.tracker.charge_bits(80)?;
                        Some(Nbt::Short(self.read_i16()?))
                    }
                    TagType::Int => {
                        self.tracker.charge_bits(96)?;
                        Some(Nbt::Int(self.read_i32()?))
                    }
                    TagType::Long => {
                        self.tracker.charge_bits(128)?;
                        Some(Nbt::Long(self.read_i64()?))
                    }
                    TagType::Float => {
                        self.tracker.charge_bits(96)?;
                        Some(Nbt::Float(self.read_f32()?))
                    }
                    TagType::Double => {
                        self.tracker.charge_bits(128)?;
                        Some(Nbt::Double(self.read_f64()?))
                    }
                    TagType::ByteArray => {
                        self.tracker.charge_bits(192)?;
                        let n = self.read_len(TagType::ByteArray)?;
                        self.tracker.charge_bits(8u64.saturating_mul(n as u64))?;
                        Some(Nbt::ByteArray(
                            self.read_bytes_untrusted(n, "a TAG_Byte_Array payload")?,
                        ))
                    }
                    TagType::String => {
                        self.tracker.charge_bits(288)?;
                        let s = self.read_string()?;
                        self.tracker.charge_bits(
                            16u64.saturating_mul(modified_utf8::utf16_len(&s) as u64),
                        )?;
                        Some(Nbt::String(s))
                    }
                    TagType::List => {
                        self.tracker.charge_bits(296)?;
                        self.check_depth(depth)?;
                        let elem_id = self.read_u8()?;
                        let count = self.read_i32()?;
                        // java: `NBTTagList.read` only calls `createNewByType` per element, so
                        // an empty list with an unknown type byte is accepted (a negative count
                        // still fails at `newArrayListWithCapacity`). Java's `write` resets an
                        // empty list's type to TAG_End, which is what this stores.
                        let elem = match TagType::from_id(elem_id) {
                            Some(elem) => elem,
                            None if count == 0 => TagType::End,
                            None => return Err(NbtError::UnknownTagType(elem_id)),
                        };
                        if elem == TagType::End && count > 0 {
                            // java: "Missing type on ListTag"
                            return Err(NbtError::MissingListType { count });
                        }
                        let n = usize::try_from(count).map_err(|_| NbtError::NegativeLength {
                            kind: TagType::List,
                            len: count,
                        })?;
                        self.tracker.charge_bits(32u64.saturating_mul(n as u64))?;
                        if n == 0 {
                            Some(Nbt::List(NbtList::from_read(elem, Vec::new())))
                        } else {
                            // Deliberately not `Vec::with_capacity(n)`: `n` is attacker-controlled.
                            frames.push(Frame::List {
                                elem,
                                remaining: n,
                                items: Vec::new(),
                                depth,
                            });
                            next = Some((elem, depth + 1));
                            None
                        }
                    }
                    TagType::Compound => {
                        self.tracker.charge_bits(384)?;
                        self.check_depth(depth)?;
                        frames.push(Frame::Compound {
                            compound: NbtCompound::new(),
                            pending_key: None,
                            depth,
                        });
                        None
                    }
                    TagType::IntArray => {
                        self.tracker.charge_bits(192)?;
                        let n = self.read_len(TagType::IntArray)?;
                        self.tracker.charge_bits(32u64.saturating_mul(n as u64))?;
                        let bytes = n
                            .checked_mul(4)
                            .ok_or(NbtError::NegativeLength {
                                kind: TagType::IntArray,
                                len: i32::MAX,
                            })
                            .and_then(|bytes| {
                                self.read_bytes_untrusted(bytes, "a TAG_Int_Array payload")
                            })?;
                        let ints = bytes
                            .chunks_exact(4)
                            .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        Some(Nbt::IntArray(ints))
                    }
                };
                if parsed.is_some() {
                    value = parsed;
                }
            }

            if let Some(child) = value.take() {
                match frames.last_mut() {
                    Some(Frame::List {
                        elem,
                        remaining,
                        items,
                        depth,
                    }) => {
                        items.push(child);
                        *remaining -= 1;
                        if *remaining == 0 {
                            let Frame::List { elem, items, .. } = frames.pop().unwrap() else {
                                unreachable!();
                            };
                            value = Some(Nbt::List(NbtList::from_read(elem, items)));
                        } else {
                            next = Some((*elem, *depth + 1));
                        }
                    }
                    Some(Frame::Compound {
                        compound,
                        pending_key,
                        ..
                    }) => {
                        let key = pending_key.take().expect("compound child has a key");
                        if compound.insert(key, child).is_some() {
                            // java: duplicate keys cost an extra map entry.
                            self.tracker.charge_bits(288)?;
                        }
                    }
                    None => return Ok(child),
                }
            }

            if next.is_none() && value.is_none() {
                match frames.last_mut() {
                    Some(Frame::List { .. }) => unreachable!("a list always schedules a child"),
                    Some(Frame::Compound {
                        pending_key, depth, ..
                    }) => {
                        let id = self.read_u8()?;
                        let entry_ty = TagType::from_id(id).ok_or(NbtError::UnknownTagType(id))?;
                        if entry_ty == TagType::End {
                            let Frame::Compound { compound, .. } = frames.pop().unwrap() else {
                                unreachable!();
                            };
                            value = Some(Nbt::Compound(compound));
                        } else {
                            let key = self.read_string()?;
                            self.tracker.charge_bits(
                                224 + 16u64.saturating_mul(modified_utf8::utf16_len(&key) as u64),
                            )?;
                            *pending_key = Some(key);
                            next = Some((entry_ty, *depth + 1));
                        }
                    }
                    None => unreachable!("a payload must produce a value"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::to_vec;

    #[test]
    fn lone_zero_byte_is_an_end_root() {
        let (name, value) = from_slice(&[0]).expect("TAG_End root");
        assert_eq!(name, "");
        assert_eq!(value, Nbt::End);
    }

    #[test]
    fn unknown_tag_type_is_rejected() {
        assert!(matches!(
            from_slice(&[12, 0, 0]),
            Err(NbtError::UnknownTagType(12))
        ));
    }

    /// java: `NBTTagList.read` never instantiates the element
    /// type of an empty list, so any type byte is accepted with count 0; a count above zero
    /// reaches `createNewByType` (null, NPE) and a negative one `newArrayListWithCapacity`.
    #[test]
    fn an_empty_list_accepts_any_element_type_byte_like_java() {
        for elem in [12u8, 99, 0xFF] {
            let (_, value) = from_slice(&[TagType::List.id(), 0, 0, elem, 0, 0, 0, 0])
                .unwrap_or_else(|error| panic!("type {elem}: {error}"));
            assert!(value.as_list().is_some_and(|items| items.is_empty()));
            assert_eq!(value.list_tag_type(), Some(TagType::End), "type {elem}");
            for count in [1i32, -1] {
                let mut bytes = vec![TagType::List.id(), 0, 0, elem];
                bytes.extend_from_slice(&count.to_be_bytes());
                assert!(from_slice(&bytes).is_err(), "type {elem}, count {count}");
            }
        }
    }

    #[test]
    fn truncated_stream_is_an_unexpected_eof_not_a_panic() {
        let err = from_slice(&[0x03, 0x00, 0x00, 0xFF]).unwrap_err();
        match err {
            NbtError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof),
            other => panic!("expected UnexpectedEof, got {other:?}"),
        }
    }

    #[test]
    fn hostile_byte_array_length_does_not_allocate() {
        // TAG_Byte_Array root, empty name, length 0x7FFF_FFFF, then nothing.
        let bytes = [0x07, 0x00, 0x00, 0x7F, 0xFF, 0xFF, 0xFF];
        let err = from_slice_with_limits(&bytes, NbtLimits::UNLIMITED).unwrap_err();
        match err {
            NbtError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof),
            other => panic!("expected UnexpectedEof, got {other:?}"),
        }
        // With the protocol budget it never even reaches the read.
        assert!(matches!(
            from_slice(&bytes),
            Err(NbtError::SizeLimit { .. })
        ));
    }

    #[test]
    fn hostile_int_array_length_does_not_allocate() {
        let bytes = [0x0b, 0x00, 0x00, 0x7F, 0xFF, 0xFF, 0xFF];
        let err = from_slice_with_limits(&bytes, NbtLimits::UNLIMITED).unwrap_err();
        assert!(matches!(err, NbtError::Io(_)), "got {err:?}");
    }

    #[test]
    fn hostile_list_length_does_not_allocate() {
        // TAG_List of TAG_Compound, count 0x7FFF_FFFF, no data.
        let bytes = [0x09, 0x00, 0x00, 0x0a, 0x7F, 0xFF, 0xFF, 0xFF];
        let err = from_slice_with_limits(&bytes, NbtLimits::UNLIMITED).unwrap_err();
        assert!(matches!(err, NbtError::Io(_)), "got {err:?}");
    }

    #[test]
    fn negative_lengths_are_rejected() {
        let bytes = [0x07, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];
        assert!(matches!(
            from_slice_with_limits(&bytes, NbtLimits::UNLIMITED),
            Err(NbtError::NegativeLength {
                kind: TagType::ByteArray,
                len: -1
            })
        ));
    }

    #[test]
    fn list_with_end_type_but_elements_is_rejected() {
        let bytes = [0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01];
        assert!(matches!(
            from_slice(&bytes),
            Err(NbtError::MissingListType { count: 1 })
        ));
    }

    #[test]
    fn nesting_is_depth_limited() {
        // 600 nested single-entry compounds, then close them all.
        let depth = 600usize;
        let mut bytes = vec![0x0a, 0x00, 0x00];
        for _ in 0..depth {
            bytes.extend_from_slice(&[0x0a, 0x00, 0x01, b'c']);
        }
        bytes.extend(std::iter::repeat_n(0x00u8, depth + 1));

        assert!(matches!(
            from_slice_with_limits(&bytes, NbtLimits::UNLIMITED),
            Err(NbtError::DepthLimit { limit: 512 })
        ));
    }

    #[test]
    fn deep_but_legal_nesting_is_accepted() {
        let depth = 200usize;
        let mut bytes = vec![0x0a, 0x00, 0x00];
        for _ in 0..depth {
            bytes.extend_from_slice(&[0x0a, 0x00, 0x01, b'c']);
        }
        bytes.extend(std::iter::repeat_n(0x00u8, depth + 1));

        let (name, value) = from_slice_with_limits(&bytes, NbtLimits::UNLIMITED).expect("legal");
        assert_eq!(name, "");
        assert_eq!(to_vec(&name, &value).expect("re-encode"), bytes);
    }

    #[test]
    fn reader_stops_at_the_end_of_the_tag() {
        // A protocol buffer holds NBT followed by more fields; we must not over-read.
        let mut stream: &[u8] = &[0x01, 0x00, 0x00, 0x2a, 0xde, 0xad, 0xbe, 0xef];
        let (_, value) = read(&mut stream).expect("TAG_Byte root");
        assert_eq!(value, Nbt::Byte(42));
        assert_eq!(stream, &[0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn mixed_nested_containers_resume_siblings_and_leave_trailing_bytes() {
        let mut first = NbtCompound::new();
        first.insert("inner", Nbt::List(vec![Nbt::Int(1), Nbt::Int(2)].into()));
        let mut second = NbtCompound::new();
        second.insert("inner", Nbt::List(vec![Nbt::Int(3)].into()));

        let mut root = NbtCompound::new();
        root.insert(
            "nested",
            Nbt::List(vec![Nbt::Compound(first), Nbt::Compound(second)].into()),
        );
        root.insert("sibling", Nbt::Long(9));

        let bytes = to_vec("root", &Nbt::Compound(root)).expect("encode nested fixture");
        let mut encoded = bytes;
        encoded.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        let mut stream: &[u8] = &encoded;
        let (name, value) = read(&mut stream).expect("decode nested fixture");

        assert_eq!(name, "root");
        let root = value.as_compound().expect("compound root");
        assert_eq!(root.get_i64("sibling"), Some(9));
        let nested = root
            .get("nested")
            .and_then(Nbt::as_list)
            .expect("nested list");
        assert_eq!(nested.len(), 2);
        assert_eq!(
            nested[0]
                .as_compound()
                .and_then(|c| c.get("inner"))
                .and_then(Nbt::as_list)
                .expect("first inner list"),
            &[Nbt::Int(1), Nbt::Int(2)]
        );
        assert_eq!(
            nested[1]
                .as_compound()
                .and_then(|c| c.get("inner"))
                .and_then(Nbt::as_list)
                .expect("second inner list"),
            &[Nbt::Int(3)]
        );
        assert_eq!(stream, &[0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn duplicate_keys_keep_the_last_value_like_javas_hashmap() {
        let mut bytes = vec![0x0a, 0x00, 0x00];
        bytes.extend_from_slice(&[0x01, 0x00, 0x01, b'k', 0x01]);
        bytes.extend_from_slice(&[0x01, 0x00, 0x01, b'k', 0x02]);
        bytes.push(0x00);
        let (_, value) = from_slice(&bytes).expect("duplicate keys parse");
        let c = value.as_compound().expect("compound");
        assert_eq!(c.len(), 1);
        assert_eq!(c.get_i8("k"), Some(2));
    }
}
