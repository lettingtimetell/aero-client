//! `Serialize`/`Deserialize` for [`Nbt`] itself, so an NBT tree can travel through any
//! self-describing serde format (JSON dumps in the F3 overlay, test fixtures, and so on).

use std::fmt;

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::tag::{Nbt, NbtCompound};

impl Serialize for Nbt {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Nbt::End => s.serialize_unit(),
            Nbt::Byte(v) => s.serialize_i8(*v),
            Nbt::Short(v) => s.serialize_i16(*v),
            Nbt::Int(v) => s.serialize_i32(*v),
            Nbt::Long(v) => s.serialize_i64(*v),
            Nbt::Float(v) => s.serialize_f32(*v),
            Nbt::Double(v) => s.serialize_f64(*v),
            Nbt::ByteArray(v) => s.serialize_bytes(v),
            Nbt::String(v) => s.serialize_str(v),
            Nbt::List(v) => v.serialize(s),
            Nbt::Compound(v) => v.serialize(s),
            Nbt::IntArray(v) => v.serialize(s),
        }
    }
}

impl Serialize for NbtCompound {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.len()))?;
        for (key, value) in self.iter() {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

struct NbtVisitor;

impl<'de> Visitor<'de> for NbtVisitor {
    type Value = Nbt;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an NBT value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Nbt, E> {
        Ok(Nbt::Byte(i8::from(v)))
    }

    fn visit_i8<E>(self, v: i8) -> Result<Nbt, E> {
        Ok(Nbt::Byte(v))
    }

    fn visit_i16<E>(self, v: i16) -> Result<Nbt, E> {
        Ok(Nbt::Short(v))
    }

    fn visit_i32<E>(self, v: i32) -> Result<Nbt, E> {
        Ok(Nbt::Int(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Nbt, E> {
        Ok(Nbt::Long(v))
    }

    fn visit_u8<E>(self, v: u8) -> Result<Nbt, E> {
        Ok(Nbt::Byte(v as i8))
    }

    fn visit_u16<E>(self, v: u16) -> Result<Nbt, E> {
        Ok(Nbt::Short(v as i16))
    }

    fn visit_u32<E>(self, v: u32) -> Result<Nbt, E> {
        Ok(Nbt::Int(v as i32))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Nbt, E> {
        Ok(Nbt::Long(v as i64))
    }

    fn visit_f32<E>(self, v: f32) -> Result<Nbt, E> {
        Ok(Nbt::Float(v))
    }

    fn visit_f64<E>(self, v: f64) -> Result<Nbt, E> {
        Ok(Nbt::Double(v))
    }

    fn visit_char<E>(self, v: char) -> Result<Nbt, E> {
        Ok(Nbt::String(v.to_string()))
    }

    fn visit_str<E>(self, v: &str) -> Result<Nbt, E> {
        Ok(Nbt::String(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> Result<Nbt, E> {
        Ok(Nbt::String(v))
    }

    fn visit_bytes<E>(self, v: &[u8]) -> Result<Nbt, E> {
        Ok(Nbt::ByteArray(v.to_vec()))
    }

    fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Nbt, E> {
        Ok(Nbt::ByteArray(v))
    }

    fn visit_unit<E>(self) -> Result<Nbt, E> {
        Ok(Nbt::End)
    }

    fn visit_none<E>(self) -> Result<Nbt, E> {
        Ok(Nbt::End)
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Nbt, D::Error> {
        Nbt::deserialize(d)
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, d: D) -> Result<Nbt, D::Error> {
        Nbt::deserialize(d)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Nbt, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<Nbt>()? {
            items.push(item);
        }
        Ok(Nbt::List(items.into()))
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Nbt, A::Error> {
        Ok(Nbt::Compound(visit_compound(map)?))
    }
}

fn visit_compound<'de, A: MapAccess<'de>>(mut map: A) -> Result<NbtCompound, A::Error> {
    let mut out = NbtCompound::with_capacity(map.size_hint().unwrap_or(0).min(64));
    while let Some((key, value)) = map.next_entry::<String, Nbt>()? {
        out.insert(key, value);
    }
    Ok(out)
}

impl<'de> Deserialize<'de> for Nbt {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Nbt, D::Error> {
        d.deserialize_any(NbtVisitor)
    }
}

struct NbtCompoundVisitor;

impl<'de> Visitor<'de> for NbtCompoundVisitor {
    type Value = NbtCompound;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a TAG_Compound")
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<NbtCompound, A::Error> {
        visit_compound(map)
    }
}

impl<'de> Deserialize<'de> for NbtCompound {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<NbtCompound, D::Error> {
        d.deserialize_map(NbtCompoundVisitor)
    }
}
