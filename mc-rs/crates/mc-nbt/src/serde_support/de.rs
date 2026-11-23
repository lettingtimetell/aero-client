//! [`Nbt`] → `Deserialize`.

use serde::de::value::{I32Deserializer, StrDeserializer};
use serde::de::{
    DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess, VariantAccess, Visitor,
};
use serde::{forward_to_deserialize_any, Deserialize};

use crate::error::NbtError;
use crate::tag::{Nbt, TagType};

/// Deserialize a Rust value out of an [`Nbt`] tree.
///
/// See the [module docs](super) for the type mapping. Borrowing works: a field typed
/// `&'de str` reads straight out of the tag.
///
/// ```
/// # use serde::Deserialize;
/// #[derive(Deserialize, PartialEq, Debug)]
/// struct ServerData { name: String, ip: String }
///
/// let bytes = std::fs::read(concat!(
///     env!("CARGO_MANIFEST_DIR"), "/../../../MCP-919-main/jars/servers.dat"
/// )).unwrap();
/// let (_, root) = mc_nbt::read_compound_with_limits(
///     &mut &bytes[..], mc_nbt::NbtLimits::UNLIMITED,
/// ).unwrap();
///
/// let entry: ServerData = mc_nbt::from_nbt(&root.get_list("servers").unwrap()[0]).unwrap();
/// assert_eq!(entry, ServerData { name: "MCP Test Server".into(), ip: "127.0.0.1".into() });
/// ```
pub fn from_nbt<'de, T: Deserialize<'de>>(value: &'de Nbt) -> Result<T, NbtError> {
    T::deserialize(NbtDeserializer::new(value))
}

/// The `serde::Deserializer` behind [`from_nbt`]. Rarely named directly.
#[derive(Debug, Clone, Copy)]
pub struct NbtDeserializer<'de> {
    value: &'de Nbt,
}

impl<'de> NbtDeserializer<'de> {
    /// Wrap an existing tag.
    #[must_use]
    pub fn new(value: &'de Nbt) -> Self {
        Self { value }
    }

    fn wrong_type(&self, expected: TagType) -> NbtError {
        NbtError::UnexpectedType {
            expected,
            found: self.value.tag_type(),
        }
    }
}

impl<'de> serde::Deserializer<'de> for NbtDeserializer<'de> {
    type Error = NbtError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NbtError> {
        match self.value {
            Nbt::End => visitor.visit_unit(),
            Nbt::Byte(v) => visitor.visit_i8(*v),
            Nbt::Short(v) => visitor.visit_i16(*v),
            Nbt::Int(v) => visitor.visit_i32(*v),
            Nbt::Long(v) => visitor.visit_i64(*v),
            Nbt::Float(v) => visitor.visit_f32(*v),
            Nbt::Double(v) => visitor.visit_f64(*v),
            Nbt::ByteArray(v) => visitor.visit_borrowed_bytes(v),
            Nbt::String(v) => visitor.visit_borrowed_str(v),
            Nbt::List(items) => visitor.visit_seq(ListAccess { iter: items.iter() }),
            Nbt::IntArray(ints) => visitor.visit_seq(IntArrayAccess { iter: ints.iter() }),
            Nbt::Compound(c) => visitor.visit_map(CompoundAccess {
                iter: c.as_slice().iter(),
                value: None,
            }),
        }
    }

    /// java: `NBTTagCompound.getBoolean` — a `TAG_Byte`, non-zero being true.
    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NbtError> {
        match self.value {
            Nbt::Byte(v) => visitor.visit_bool(*v != 0),
            _ => self.deserialize_any(visitor),
        }
    }

    // NBT has no unsigned tags; the game reinterprets the bits, so we do too.
    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NbtError> {
        match self.value {
            Nbt::Byte(v) => visitor.visit_u8(*v as u8),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NbtError> {
        match self.value {
            Nbt::Short(v) => visitor.visit_u16(*v as u16),
            // Java's NBTPrimitive widens while still signed, then the Rust target reinterprets
            // that widened value. Byte(-1) is therefore 0xffff, not 0x00ff.
            Nbt::Byte(v) => visitor.visit_u16(i16::from(*v) as u16),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NbtError> {
        match self.value {
            Nbt::Int(v) => visitor.visit_u32(*v as u32),
            Nbt::Short(v) => visitor.visit_u32(i32::from(*v) as u32),
            Nbt::Byte(v) => visitor.visit_u32(i32::from(*v) as u32),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NbtError> {
        match self.value {
            Nbt::Long(v) => visitor.visit_u64(*v as u64),
            Nbt::Int(v) => visitor.visit_u64(i64::from(*v) as u64),
            Nbt::Short(v) => visitor.visit_u64(i64::from(*v) as u64),
            Nbt::Byte(v) => visitor.visit_u64(i64::from(*v) as u64),
            _ => self.deserialize_any(visitor),
        }
    }

    /// A present tag is always `Some`; a missing compound key never reaches us at all.
    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, NbtError> {
        match self.value {
            Nbt::End => visitor.visit_none(),
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, NbtError> {
        visitor.visit_newtype_struct(self)
    }

    /// A fieldless variant is a `TAG_String`; anything else is a single-entry compound.
    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, NbtError> {
        match self.value {
            Nbt::String(s) => visitor.visit_enum(VariantDispatch {
                variant: s,
                value: None,
            }),
            Nbt::Compound(c) if c.len() == 1 => {
                let (variant, value) = &c.as_slice()[0];
                visitor.visit_enum(VariantDispatch {
                    variant,
                    value: Some(value),
                })
            }
            Nbt::Compound(c) => Err(NbtError::Message(format!(
                "an enum wants a single-entry TAG_Compound, found {} entries",
                c.len()
            ))),
            _ => Err(self.wrong_type(TagType::Compound)),
        }
    }

    forward_to_deserialize_any! {
        i8 i16 i32 i64 i128 u128 f32 f64 char str string bytes byte_buf unit unit_struct
        seq tuple tuple_struct map struct identifier ignored_any
    }
}

struct ListAccess<'de> {
    iter: std::slice::Iter<'de, Nbt>,
}

impl<'de> SeqAccess<'de> for ListAccess<'de> {
    type Error = NbtError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, NbtError> {
        match self.iter.next() {
            Some(v) => seed.deserialize(NbtDeserializer::new(v)).map(Some),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

struct IntArrayAccess<'de> {
    iter: std::slice::Iter<'de, i32>,
}

impl<'de> SeqAccess<'de> for IntArrayAccess<'de> {
    type Error = NbtError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, NbtError> {
        match self.iter.next() {
            Some(v) => seed
                .deserialize(I32Deserializer::<NbtError>::new(*v))
                .map(Some),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

struct CompoundAccess<'de> {
    iter: std::slice::Iter<'de, (String, Nbt)>,
    value: Option<&'de Nbt>,
}

impl<'de> MapAccess<'de> for CompoundAccess<'de> {
    type Error = NbtError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, NbtError> {
        match self.iter.next() {
            Some((key, value)) => {
                self.value = Some(value);
                seed.deserialize(StrDeserializer::<NbtError>::new(key))
                    .map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, NbtError> {
        let value = self
            .value
            .take()
            .ok_or_else(|| NbtError::Message("compound value requested before its key".into()))?;
        seed.deserialize(NbtDeserializer::new(value))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

struct VariantDispatch<'de> {
    variant: &'de str,
    value: Option<&'de Nbt>,
}

impl<'de> EnumAccess<'de> for VariantDispatch<'de> {
    type Error = NbtError;
    type Variant = Self;

    fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<(V::Value, Self), NbtError> {
        let variant: StrDeserializer<'de, NbtError> = self.variant.into_deserializer();
        Ok((seed.deserialize(variant)?, self))
    }
}

impl<'de> VariantAccess<'de> for VariantDispatch<'de> {
    type Error = NbtError;

    fn unit_variant(self) -> Result<(), NbtError> {
        match self.value {
            None => Ok(()),
            Some(v) => Err(NbtError::Message(format!(
                "variant `{}` is a unit variant but carries a {}",
                self.variant,
                v.tag_type()
            ))),
        }
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, NbtError> {
        match self.value {
            Some(v) => seed.deserialize(NbtDeserializer::new(v)),
            None => Err(NbtError::Message(format!(
                "variant `{}` carries no value",
                self.variant
            ))),
        }
    }

    fn tuple_variant<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<V::Value, NbtError> {
        match self.value {
            Some(Nbt::List(items)) => visitor.visit_seq(ListAccess { iter: items.iter() }),
            Some(other) => Err(NbtError::UnexpectedType {
                expected: TagType::List,
                found: other.tag_type(),
            }),
            None => Err(NbtError::Message(format!(
                "variant `{}` carries no value",
                self.variant
            ))),
        }
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, NbtError> {
        match self.value {
            Some(Nbt::Compound(c)) => visitor.visit_map(CompoundAccess {
                iter: c.as_slice().iter(),
                value: None,
            }),
            Some(other) => Err(NbtError::UnexpectedType {
                expected: TagType::Compound,
                found: other.tag_type(),
            }),
            None => Err(NbtError::Message(format!(
                "variant `{}` carries no value",
                self.variant
            ))),
        }
    }
}
