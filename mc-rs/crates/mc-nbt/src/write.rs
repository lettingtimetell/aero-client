//! Encoding NBT to a byte stream, with Java `DataOutput` semantics.
//!
//! java: `CompressedStreamTools.writeTag`, plus each `NBTTag*.write`

use std::io::Write;

use crate::error::NbtError;
use crate::limits::NbtLimits;
use crate::modified_utf8;
use crate::tag::{Nbt, NbtCompound, TagType};

/// Write one named root tag: the type byte, the name, then the payload.
///
/// A [`Nbt::End`] root writes a single zero byte and ignores `name`, matching vanilla.
/// Many small writes are performed: wrap a `File` in a `BufWriter` first.
///
/// ```
/// # use mc_nbt::{write, Nbt, NbtCompound};
/// let mut c = NbtCompound::new();
/// c.insert("x", Nbt::Int(7));
/// let mut out = Vec::new();
/// write(&mut out, "", &Nbt::Compound(c)).unwrap();
/// assert_eq!(out, [0x0a, 0x00, 0x00, 0x03, 0x00, 0x01, b'x', 0, 0, 0, 7, 0x00]);
/// ```
pub fn write(w: &mut impl Write, name: &str, v: &Nbt) -> Result<(), NbtError> {
    // java: writeTag writes the id, and only then a name+payload if the id is not 0.
    w.write_all(&[v.id()])?;
    if v.tag_type() == TagType::End {
        return Ok(());
    }
    write_string(w, name)?;
    write_payload(w, v, 0)
}

/// [`write`] into a fresh `Vec`.
pub fn to_vec(name: &str, v: &Nbt) -> Result<Vec<u8>, NbtError> {
    let mut out = Vec::new();
    write(&mut out, name, v)?;
    Ok(out)
}

/// [`write`] a compound root, the shape every `.dat` file and every protocol field uses.
pub fn write_compound(w: &mut impl Write, name: &str, v: &NbtCompound) -> Result<(), NbtError> {
    write(w, name, &Nbt::Compound(v.clone()))
}

fn write_string(w: &mut impl Write, s: &str) -> Result<(), NbtError> {
    // java: DataOutputStream.writeUTF — u16 byte length, then modified UTF-8.
    let mut buf = Vec::new();
    let len = modified_utf8::encode_into(s, &mut buf)?;
    w.write_all(&(len as u16).to_be_bytes())?;
    w.write_all(&buf)?;
    Ok(())
}

fn write_len(w: &mut impl Write, kind: TagType, len: usize) -> Result<(), NbtError> {
    let len = i32::try_from(len).map_err(|_| NbtError::NegativeLength {
        kind,
        len: i32::MAX,
    })?;
    w.write_all(&len.to_be_bytes())?;
    Ok(())
}

fn write_payload(w: &mut impl Write, v: &Nbt, depth: u32) -> Result<(), NbtError> {
    match v {
        Nbt::End => Ok(()),
        Nbt::Byte(x) => Ok(w.write_all(&x.to_be_bytes())?),
        Nbt::Short(x) => Ok(w.write_all(&x.to_be_bytes())?),
        Nbt::Int(x) => Ok(w.write_all(&x.to_be_bytes())?),
        Nbt::Long(x) => Ok(w.write_all(&x.to_be_bytes())?),
        Nbt::Float(x) => Ok(w.write_all(&x.to_be_bytes())?),
        Nbt::Double(x) => Ok(w.write_all(&x.to_be_bytes())?),
        Nbt::ByteArray(bytes) => {
            write_len(w, TagType::ByteArray, bytes.len())?;
            Ok(w.write_all(bytes)?)
        }
        Nbt::String(s) => write_string(w, s),
        Nbt::List(items) => {
            // Match the reader and vanilla's read guard: only containers check depth. A scalar
            // child at depth 513 is legal when its containing list/compound is at depth 512.
            if depth > NbtLimits::VANILLA_MAX_DEPTH {
                return Err(NbtError::DepthLimit {
                    limit: NbtLimits::VANILLA_MAX_DEPTH,
                });
            }
            // java: NBTTagList.write takes the element type from the first element and
            // writes TAG_End for an empty list.
            let elem = items.refresh_type_for_write();
            for (index, item) in items.iter().enumerate() {
                let found = item.tag_type();
                if found != elem {
                    return Err(NbtError::HeterogeneousList {
                        index,
                        expected: elem,
                        found,
                    });
                }
            }
            if elem == TagType::End && !items.is_empty() {
                return Err(NbtError::EndInContainer);
            }
            w.write_all(&[elem.id()])?;
            write_len(w, TagType::List, items.len())?;
            for item in items {
                write_payload(w, item, depth + 1)?;
            }
            Ok(())
        }
        Nbt::Compound(c) => {
            if depth > NbtLimits::VANILLA_MAX_DEPTH {
                return Err(NbtError::DepthLimit {
                    limit: NbtLimits::VANILLA_MAX_DEPTH,
                });
            }
            for (key, value) in c.iter() {
                // java: writeEntry emits a bare 0x00 for a TAG_End value, which would
                // truncate the compound. Refuse rather than silently corrupt.
                if value.tag_type() == TagType::End {
                    return Err(NbtError::EndInContainer);
                }
                w.write_all(&[value.id()])?;
                write_string(w, key)?;
                write_payload(w, value, depth + 1)?;
            }
            Ok(w.write_all(&[TagType::End.id()])?)
        }
        Nbt::IntArray(ints) => {
            write_len(w, TagType::IntArray, ints.len())?;
            let mut buf = Vec::with_capacity(ints.len() * 4);
            for i in ints {
                buf.extend_from_slice(&i.to_be_bytes());
            }
            Ok(w.write_all(&buf)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::from_slice;

    #[test]
    fn end_root_is_one_byte_and_ignores_the_name() {
        assert_eq!(to_vec("ignored", &Nbt::End).expect("end root"), vec![0u8]);
    }

    #[test]
    fn scalars_are_big_endian() {
        assert_eq!(
            to_vec("", &Nbt::Int(0x0102_0304)).expect("int"),
            vec![0x03, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04]
        );
        assert_eq!(
            to_vec("", &Nbt::Long(1)).expect("long"),
            vec![0x04, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 1]
        );
        assert_eq!(
            to_vec("", &Nbt::Float(1.0)).expect("float"),
            vec![0x05, 0x00, 0x00, 0x3F, 0x80, 0x00, 0x00]
        );
    }

    #[test]
    fn empty_list_is_written_with_an_end_element_type() {
        assert_eq!(
            to_vec("", &Nbt::list(vec![])).expect("empty list"),
            vec![0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn heterogeneous_lists_are_rejected() {
        let list = Nbt::list(vec![Nbt::Int(1), Nbt::Byte(2)]);
        assert!(matches!(
            to_vec("", &list),
            Err(NbtError::HeterogeneousList {
                index: 1,
                expected: TagType::Int,
                found: TagType::Byte
            })
        ));
    }

    #[test]
    fn end_inside_a_container_is_rejected() {
        let mut c = NbtCompound::new();
        c.insert("bad", Nbt::End);
        assert!(matches!(
            to_vec("", &Nbt::Compound(c)),
            Err(NbtError::EndInContainer)
        ));
        assert!(matches!(
            to_vec("", &Nbt::list(vec![Nbt::End])),
            Err(NbtError::EndInContainer)
        ));
    }

    #[test]
    fn names_and_strings_use_modified_utf8() {
        let bytes = to_vec("\0", &Nbt::String("\u{1F600}".into())).expect("write");
        assert_eq!(
            bytes,
            vec![
                0x08, // TAG_String
                0x00, 0x02, 0xC0, 0x80, // name "\0" as two bytes
                0x00, 0x06, 0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80, // CESU-8 emoji
            ]
        );
        let (name, value) = from_slice(&bytes).expect("read back");
        assert_eq!(name, "\0");
        assert_eq!(value, Nbt::String("\u{1F600}".into()));
    }

    #[test]
    fn absurdly_deep_hand_built_values_do_not_blow_the_stack() {
        let mut value = Nbt::Byte(0);
        for _ in 0..2000 {
            value = Nbt::list(vec![value]);
        }
        assert!(matches!(
            to_vec("", &value),
            Err(NbtError::DepthLimit { .. })
        ));
    }

    #[test]
    fn scalar_at_depth_513_round_trips_like_the_vanilla_reader_acceptance_set() {
        let mut value = Nbt::Byte(7);
        for _ in 0..=NbtLimits::VANILLA_MAX_DEPTH {
            let mut compound = NbtCompound::new();
            compound.insert("c", value);
            value = Nbt::Compound(compound);
        }

        let bytes = to_vec("", &value).expect("a scalar below the deepest legal compound writes");
        let (name, decoded) =
            from_slice(&bytes).expect("the protocol reader accepts the same tree");
        assert_eq!(name, "");
        assert_eq!(decoded, value);
    }
}
