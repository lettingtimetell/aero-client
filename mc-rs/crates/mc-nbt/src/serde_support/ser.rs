//! `Serialize` → [`Nbt`].

use serde::ser::{
    self, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
    SerializeTupleStruct, SerializeTupleVariant,
};
use serde::Serialize;

use crate::error::NbtError;
use crate::tag::{Nbt, NbtCompound};

/// Convert any `Serialize` value into an [`Nbt`] tree.
///
/// See the [module docs](super) for the type mapping.
///
/// ```
/// # use serde::Serialize;
/// #[derive(Serialize)]
/// struct ServerData { name: String, ip: String, #[serde(rename = "acceptTextures")] textures: bool }
///
/// let nbt = mc_nbt::to_nbt(&ServerData {
///     name: "MCP Test Server".into(),
///     ip: "127.0.0.1".into(),
///     textures: true,
/// })
/// .unwrap();
///
/// let c = nbt.as_compound().unwrap();
/// assert_eq!(c.get_str("ip"), Some("127.0.0.1"));
/// assert_eq!(c.get_i8("acceptTextures"), Some(1));
/// ```
pub fn to_nbt<T: Serialize + ?Sized>(value: &T) -> Result<Nbt, NbtError> {
    value.serialize(NbtSerializer)
}

/// The `serde::Serializer` behind [`to_nbt`]. Rarely named directly.
#[derive(Debug, Clone, Copy)]
pub struct NbtSerializer;

fn no_null(what: &str) -> NbtError {
    NbtError::Message(format!(
        "NBT has no representation for {what}; use #[serde(skip_serializing_if = \"Option::is_none\")]"
    ))
}

impl ser::Serializer for NbtSerializer {
    type Ok = Nbt;
    type Error = NbtError;

    type SerializeSeq = SeqSerializer;
    type SerializeTuple = SeqSerializer;
    type SerializeTupleStruct = SeqSerializer;
    type SerializeTupleVariant = TupleVariantSerializer;
    type SerializeMap = MapSerializer;
    type SerializeStruct = MapSerializer;
    type SerializeStructVariant = StructVariantSerializer;

    fn serialize_bool(self, v: bool) -> Result<Nbt, NbtError> {
        // java: NBTTagCompound.setBoolean writes a TAG_Byte of 0 or 1.
        Ok(Nbt::Byte(i8::from(v)))
    }

    fn serialize_i8(self, v: i8) -> Result<Nbt, NbtError> {
        Ok(Nbt::Byte(v))
    }

    fn serialize_i16(self, v: i16) -> Result<Nbt, NbtError> {
        Ok(Nbt::Short(v))
    }

    fn serialize_i32(self, v: i32) -> Result<Nbt, NbtError> {
        Ok(Nbt::Int(v))
    }

    fn serialize_i64(self, v: i64) -> Result<Nbt, NbtError> {
        Ok(Nbt::Long(v))
    }

    // NBT is signed-only; the game reinterprets the bits and so do we.
    fn serialize_u8(self, v: u8) -> Result<Nbt, NbtError> {
        Ok(Nbt::Byte(v as i8))
    }

    fn serialize_u16(self, v: u16) -> Result<Nbt, NbtError> {
        Ok(Nbt::Short(v as i16))
    }

    fn serialize_u32(self, v: u32) -> Result<Nbt, NbtError> {
        Ok(Nbt::Int(v as i32))
    }

    fn serialize_u64(self, v: u64) -> Result<Nbt, NbtError> {
        Ok(Nbt::Long(v as i64))
    }

    fn serialize_f32(self, v: f32) -> Result<Nbt, NbtError> {
        Ok(Nbt::Float(v))
    }

    fn serialize_f64(self, v: f64) -> Result<Nbt, NbtError> {
        Ok(Nbt::Double(v))
    }

    fn serialize_char(self, v: char) -> Result<Nbt, NbtError> {
        Ok(Nbt::String(v.to_string()))
    }

    fn serialize_str(self, v: &str) -> Result<Nbt, NbtError> {
        Ok(Nbt::String(v.to_owned()))
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<Nbt, NbtError> {
        Ok(Nbt::ByteArray(v.to_vec()))
    }

    fn serialize_none(self) -> Result<Nbt, NbtError> {
        Err(no_null("None"))
    }

    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<Nbt, NbtError> {
        v.serialize(self)
    }

    fn serialize_unit(self) -> Result<Nbt, NbtError> {
        Err(no_null("()"))
    }

    fn serialize_unit_struct(self, name: &'static str) -> Result<Nbt, NbtError> {
        Err(no_null(name))
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
    ) -> Result<Nbt, NbtError> {
        Ok(Nbt::String(variant.to_owned()))
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        v: &T,
    ) -> Result<Nbt, NbtError> {
        v.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<Nbt, NbtError> {
        let mut c = NbtCompound::with_capacity(1);
        c.insert(variant, v.serialize(self)?);
        Ok(Nbt::Compound(c))
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<SeqSerializer, NbtError> {
        Ok(SeqSerializer {
            items: Vec::with_capacity(len.unwrap_or(0).min(64)),
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<SeqSerializer, NbtError> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<SeqSerializer, NbtError> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<TupleVariantSerializer, NbtError> {
        Ok(TupleVariantSerializer {
            variant,
            items: Vec::with_capacity(len.min(64)),
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<MapSerializer, NbtError> {
        Ok(MapSerializer {
            entries: NbtCompound::with_capacity(len.unwrap_or(0).min(64)),
            key: None,
        })
    }

    fn serialize_struct(self, _name: &'static str, len: usize) -> Result<MapSerializer, NbtError> {
        self.serialize_map(Some(len))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<StructVariantSerializer, NbtError> {
        Ok(StructVariantSerializer {
            variant,
            entries: NbtCompound::with_capacity(len.min(64)),
        })
    }
}

/// Accumulates `TAG_List` elements.
#[derive(Debug)]
pub struct SeqSerializer {
    items: Vec<Nbt>,
}

impl SerializeSeq for SeqSerializer {
    type Ok = Nbt;
    type Error = NbtError;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NbtError> {
        self.items.push(to_nbt(v)?);
        Ok(())
    }

    fn end(self) -> Result<Nbt, NbtError> {
        Ok(Nbt::List(self.items.into()))
    }
}

impl SerializeTuple for SeqSerializer {
    type Ok = Nbt;
    type Error = NbtError;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NbtError> {
        SerializeSeq::serialize_element(self, v)
    }

    fn end(self) -> Result<Nbt, NbtError> {
        SerializeSeq::end(self)
    }
}

impl SerializeTupleStruct for SeqSerializer {
    type Ok = Nbt;
    type Error = NbtError;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NbtError> {
        SerializeSeq::serialize_element(self, v)
    }

    fn end(self) -> Result<Nbt, NbtError> {
        SerializeSeq::end(self)
    }
}

/// Accumulates a tuple enum variant into a single-entry compound.
#[derive(Debug)]
pub struct TupleVariantSerializer {
    variant: &'static str,
    items: Vec<Nbt>,
}

impl SerializeTupleVariant for TupleVariantSerializer {
    type Ok = Nbt;
    type Error = NbtError;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NbtError> {
        self.items.push(to_nbt(v)?);
        Ok(())
    }

    fn end(self) -> Result<Nbt, NbtError> {
        let mut c = NbtCompound::with_capacity(1);
        c.insert(self.variant, Nbt::List(self.items.into()));
        Ok(Nbt::Compound(c))
    }
}

/// Accumulates `TAG_Compound` entries for maps and structs.
#[derive(Debug)]
pub struct MapSerializer {
    entries: NbtCompound,
    key: Option<String>,
}

/// Compound keys are strings; integer keys are stringified, anything else is an error.
fn key_to_string(key: Nbt) -> Result<String, NbtError> {
    match key {
        Nbt::String(s) => Ok(s),
        Nbt::Byte(v) => Ok(v.to_string()),
        Nbt::Short(v) => Ok(v.to_string()),
        Nbt::Int(v) => Ok(v.to_string()),
        Nbt::Long(v) => Ok(v.to_string()),
        other => Err(NbtError::Message(format!(
            "TAG_Compound keys must be strings, got {}",
            other.tag_type()
        ))),
    }
}

impl SerializeMap for MapSerializer {
    type Ok = Nbt;
    type Error = NbtError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, k: &T) -> Result<(), NbtError> {
        self.key = Some(key_to_string(to_nbt(k)?)?);
        Ok(())
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NbtError> {
        let key = self
            .key
            .take()
            .ok_or_else(|| NbtError::Message("map value serialised before its key".into()))?;
        self.entries.insert(key, to_nbt(v)?);
        Ok(())
    }

    fn end(self) -> Result<Nbt, NbtError> {
        Ok(Nbt::Compound(self.entries))
    }
}

impl SerializeStruct for MapSerializer {
    type Ok = Nbt;
    type Error = NbtError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        name: &'static str,
        v: &T,
    ) -> Result<(), NbtError> {
        self.entries.insert(name, to_nbt(v)?);
        Ok(())
    }

    fn end(self) -> Result<Nbt, NbtError> {
        Ok(Nbt::Compound(self.entries))
    }
}

/// Accumulates a struct enum variant into a single-entry compound.
#[derive(Debug)]
pub struct StructVariantSerializer {
    variant: &'static str,
    entries: NbtCompound,
}

impl SerializeStructVariant for StructVariantSerializer {
    type Ok = Nbt;
    type Error = NbtError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        name: &'static str,
        v: &T,
    ) -> Result<(), NbtError> {
        self.entries.insert(name, to_nbt(v)?);
        Ok(())
    }

    fn end(self) -> Result<Nbt, NbtError> {
        let mut c = NbtCompound::with_capacity(1);
        c.insert(self.variant, Nbt::Compound(self.entries));
        Ok(Nbt::Compound(c))
    }
}
