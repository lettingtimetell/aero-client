//! The NBT value type and its ordered compound map.
//!
//! java: `NBTBase.java`, `NBTTagCompound.java`, `NBTTagList.java`

use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::LazyLock;

/// The tag type byte that prefixes every NBT value on the wire.
///
/// java: `NBTBase.NBT_TYPES` / `NBTBase.createNewByType`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TagType {
    /// `TAG_End` (0) — terminates a compound; also the element type of an empty list.
    End = 0,
    /// `TAG_Byte` (1) — one signed byte.
    Byte = 1,
    /// `TAG_Short` (2) — big-endian `i16`.
    Short = 2,
    /// `TAG_Int` (3) — big-endian `i32`.
    Int = 3,
    /// `TAG_Long` (4) — big-endian `i64`.
    Long = 4,
    /// `TAG_Float` (5) — big-endian IEEE-754 binary32.
    Float = 5,
    /// `TAG_Double` (6) — big-endian IEEE-754 binary64.
    Double = 6,
    /// `TAG_Byte_Array` (7) — `i32` length then that many bytes.
    ByteArray = 7,
    /// `TAG_String` (8) — `u16` byte length then modified UTF-8.
    String = 8,
    /// `TAG_List` (9) — element type byte, `i32` count, then bare payloads.
    List = 9,
    /// `TAG_Compound` (10) — named entries until a `TAG_End` byte.
    Compound = 10,
    /// `TAG_Int_Array` (11) — `i32` length then that many big-endian `i32`s.
    IntArray = 11,
}

impl TagType {
    /// All twelve tag types, in wire order.
    pub const ALL: [TagType; 12] = [
        TagType::End,
        TagType::Byte,
        TagType::Short,
        TagType::Int,
        TagType::Long,
        TagType::Float,
        TagType::Double,
        TagType::ByteArray,
        TagType::String,
        TagType::List,
        TagType::Compound,
        TagType::IntArray,
    ];

    /// The type byte written to the wire.
    #[must_use]
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// Look up a tag type by its wire byte, or `None` if the byte is not a valid type.
    #[must_use]
    pub const fn from_id(id: u8) -> Option<TagType> {
        Some(match id {
            0 => TagType::End,
            1 => TagType::Byte,
            2 => TagType::Short,
            3 => TagType::Int,
            4 => TagType::Long,
            5 => TagType::Float,
            6 => TagType::Double,
            7 => TagType::ByteArray,
            8 => TagType::String,
            9 => TagType::List,
            10 => TagType::Compound,
            11 => TagType::IntArray,
            _ => return None,
        })
    }

    /// Vanilla's name for the type, as it appears in crash reports.
    ///
    /// java: `NBTBase.NBT_TYPES`
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            TagType::End => "END",
            TagType::Byte => "BYTE",
            TagType::Short => "SHORT",
            TagType::Int => "INT",
            TagType::Long => "LONG",
            TagType::Float => "FLOAT",
            TagType::Double => "DOUBLE",
            TagType::ByteArray => "BYTE[]",
            TagType::String => "STRING",
            TagType::List => "LIST",
            TagType::Compound => "COMPOUND",
            TagType::IntArray => "INT[]",
        }
    }
}

impl std::fmt::Display for TagType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A single NBT value.
///
/// Every integer is signed and big-endian on the wire, matching Java's `DataInput`.
/// `ByteArray` holds `u8` rather than Java's signed `byte[]`; reinterpret with `as i8`
/// where the semantics call for it (heightmaps and light arrays are unsigned in practice).
///
/// The derived [`PartialEq`] compares `Float`/`Double` with `==`, so `NaN != NaN` and
/// `+0.0 == -0.0` — see [`Nbt::bit_eq`] for a round-trip-safe alternative.
#[derive(Debug, Clone, PartialEq)]
pub enum Nbt {
    /// `TAG_End`. Only ever appears as the element type of an empty list; storing one
    /// inside a compound or list is a programming error and is rejected by [`crate::write`].
    End,
    /// `TAG_Byte`.
    Byte(i8),
    /// `TAG_Short`.
    Short(i16),
    /// `TAG_Int`.
    Int(i32),
    /// `TAG_Long`.
    Long(i64),
    /// `TAG_Float`.
    Float(f32),
    /// `TAG_Double`.
    Double(f64),
    /// `TAG_Byte_Array`.
    ByteArray(Vec<u8>),
    /// `TAG_String`, stored decoded; see [`crate::modified_utf8`] for the wire form.
    String(String),
    /// `TAG_List`. Every element must have the same [`TagType`].
    ///
    /// [`NbtList`] retains `NBTTagList`'s private `tagType` field. In particular,
    /// a zero-length list read as a `TAG_Int` list still reports `TAG_Int` from
    /// `getTagType`, until writing it resets that field to `TAG_End`.
    List(NbtList),
    /// `TAG_Compound`.
    Compound(NbtCompound),
    /// `TAG_Int_Array`.
    IntArray(Vec<i32>),
}

/// The payload of [`Nbt::List`], including vanilla's private element-type field.
///
/// Construct this from a vector with [`From<Vec<Nbt>>`], or use [`Nbt::list`].
/// The vector is dereferenceable for read-only compatibility, but use
/// [`Nbt::append_tag`], [`Nbt::set_tag`], and [`Nbt::remove_tag`] for mutations:
/// they retain `NBTTagList`'s type-invariant behaviour.
#[derive(Debug)]
pub struct NbtList {
    items: Vec<Nbt>,
    // `NBTTagList.write` updates this even though our writer takes `&Nbt`.
    // Atomics permit that Java-observable state change without infecting all writer APIs with
    // mutable borrows.
    tag_type: AtomicU8,
}

impl NbtList {
    /// An empty list whose `getTagType()` is `TAG_End`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            items: Vec::new(),
            tag_type: AtomicU8::new(TagType::End.id()),
        }
    }

    /// The private `NBTTagList.tagType` returned by Java's `getTagType()`.
    #[must_use]
    pub fn tag_type(&self) -> TagType {
        TagType::from_id(self.tag_type.load(Ordering::Relaxed))
            .expect("NBTTagList.tagType is always a valid tag type")
    }

    pub(crate) fn from_read(tag_type: TagType, items: Vec<Nbt>) -> Self {
        Self {
            items,
            tag_type: AtomicU8::new(tag_type.id()),
        }
    }

    pub(crate) fn refresh_type_for_write(&self) -> TagType {
        let tag_type = self.items.first().map_or(TagType::End, Nbt::tag_type);
        self.tag_type.store(tag_type.id(), Ordering::Relaxed);
        tag_type
    }
}

impl Default for NbtList {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for NbtList {
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
            tag_type: AtomicU8::new(self.tag_type.load(Ordering::Relaxed)),
        }
    }
}

impl From<Vec<Nbt>> for NbtList {
    fn from(items: Vec<Nbt>) -> Self {
        let tag_type = items.first().map_or(TagType::End, Nbt::tag_type);
        Self {
            items,
            tag_type: AtomicU8::new(tag_type.id()),
        }
    }
}

impl Deref for NbtList {
    type Target = Vec<Nbt>;

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

impl DerefMut for NbtList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.items
    }
}

impl<'a> IntoIterator for &'a NbtList {
    type Item = &'a Nbt;
    type IntoIter = std::slice::Iter<'a, Nbt>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

// NBTTagList.equals compares both its backing List and its private tagType byte.
impl PartialEq for NbtList {
    fn eq(&self, other: &Self) -> bool {
        self.tag_type() == other.tag_type() && self.items == other.items
    }
}

impl fmt::Display for Nbt {
    /// Vanilla's `NBTBase.toString()` representation.
    ///
    /// This predates modern SNBT: list entries carry numeric indices, int arrays retain a trailing
    /// comma, and strings escape only double quotes (not backslashes or control characters).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Nbt::End => f.write_str("END"),
            Nbt::Byte(value) => write!(f, "{value}b"),
            Nbt::Short(value) => write!(f, "{value}s"),
            Nbt::Int(value) => value.fmt(f),
            Nbt::Long(value) => write!(f, "{value}L"),
            Nbt::Float(value) => {
                write_java_float(f, value)?;
                f.write_char('f')
            }
            Nbt::Double(value) => {
                write_java_float(f, value)?;
                f.write_char('d')
            }
            Nbt::ByteArray(values) => write!(f, "[{} bytes]", values.len()),
            Nbt::String(value) => {
                f.write_char('"')?;
                for part in value.split_inclusive('"') {
                    if let Some(unquoted) = part.strip_suffix('"') {
                        f.write_str(unquoted)?;
                        f.write_str("\\\"")?;
                    } else {
                        f.write_str(part)?;
                    }
                }
                f.write_char('"')
            }
            Nbt::List(values) => {
                f.write_char('[')?;
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        f.write_char(',')?;
                    }
                    write!(f, "{index}:{value}")?;
                }
                f.write_char(']')
            }
            Nbt::Compound(entries) => {
                f.write_char('{')?;
                for (index, (key, value)) in entries.iter().enumerate() {
                    if index != 0 {
                        f.write_char(',')?;
                    }
                    write!(f, "{key}:{value}")?;
                }
                f.write_char('}')
            }
            Nbt::IntArray(values) => {
                f.write_char('[')?;
                for value in values {
                    write!(f, "{value},")?;
                }
                f.write_char(']')
            }
        }
    }
}

// `MathHelper.floor_float` / `floor_double`, including Java's wrapping `i - 1` at the
// negative saturation boundary. Rust's float-to-int casts otherwise match Java's
// truncation, NaN-to-zero, and saturation rules.
fn java_floor_f32_to_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if value < truncated as f32 {
        truncated.wrapping_sub(1)
    } else {
        truncated
    }
}

fn java_floor_f64_to_i32(value: f64) -> i32 {
    let truncated = value as i32;
    if value < f64::from(truncated) {
        truncated.wrapping_sub(1)
    } else {
        truncated
    }
}

/// Re-shape Rust's shortest round-tripping decimal into Java's `Float.toString` /
/// `Double.toString` notation. Both runtimes use a shortest-decimal conversion, but Java switches
/// to scientific notation below 1e-3 and at 1e7, always includes a decimal point, and spells the
/// infinities differently.
fn write_java_float(f: &mut fmt::Formatter<'_>, value: impl fmt::Display) -> fmt::Result {
    let raw = value.to_string();
    match raw.as_str() {
        "NaN" => return f.write_str("NaN"),
        "inf" => return f.write_str("Infinity"),
        "-inf" => return f.write_str("-Infinity"),
        _ => {}
    }

    let (sign, raw) = raw
        .strip_prefix('-')
        .map_or(("", raw.as_str()), |v| ("-", v));
    let (mantissa, explicit_exponent) = raw
        .split_once(['e', 'E'])
        .map_or((raw, 0), |(m, e)| (m, e.parse::<i32>().unwrap()));
    let point = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let mut digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let leading_zeroes = digits.bytes().take_while(|&b| b == b'0').count();
    let decimal_position = point + explicit_exponent - leading_zeroes as i32;
    digits.drain(..leading_zeroes);

    if digits.is_empty() {
        return write!(f, "{sign}0.0");
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }

    let scientific_exponent = decimal_position - 1;
    if (-3..7).contains(&scientific_exponent) {
        write!(f, "{sign}")?;
        if decimal_position <= 0 {
            f.write_str("0.")?;
            for _ in 0..-decimal_position {
                f.write_char('0')?;
            }
            f.write_str(&digits)
        } else {
            let decimal_position = decimal_position as usize;
            if decimal_position >= digits.len() {
                f.write_str(&digits)?;
                for _ in digits.len()..decimal_position {
                    f.write_char('0')?;
                }
                f.write_str(".0")
            } else {
                f.write_str(&digits[..decimal_position])?;
                f.write_char('.')?;
                f.write_str(&digits[decimal_position..])
            }
        }
    } else {
        write!(f, "{sign}{}.", &digits[..1])?;
        if digits.len() == 1 {
            f.write_char('0')?;
        } else {
            f.write_str(&digits[1..])?;
        }
        write!(f, "E{scientific_exponent}")
    }
}

impl Nbt {
    /// Construct a `TAG_List` from homogeneous items.
    ///
    /// This replaces the former `Nbt::List(items)` constructor. The list's initial `getTagType()` is taken
    /// from its first item, exactly as a freshly populated `NBTTagList` would be.
    #[must_use]
    pub fn list(items: Vec<Nbt>) -> Self {
        Self::List(items.into())
    }

    /// The tag type of this value.
    #[must_use]
    pub fn tag_type(&self) -> TagType {
        match self {
            Nbt::End => TagType::End,
            Nbt::Byte(_) => TagType::Byte,
            Nbt::Short(_) => TagType::Short,
            Nbt::Int(_) => TagType::Int,
            Nbt::Long(_) => TagType::Long,
            Nbt::Float(_) => TagType::Float,
            Nbt::Double(_) => TagType::Double,
            Nbt::ByteArray(_) => TagType::ByteArray,
            Nbt::String(_) => TagType::String,
            Nbt::List(_) => TagType::List,
            Nbt::Compound(_) => TagType::Compound,
            Nbt::IntArray(_) => TagType::IntArray,
        }
    }

    /// The wire type byte. java: `NBTBase.getId()`
    #[must_use]
    pub fn id(&self) -> u8 {
        self.tag_type().id()
    }

    /// Directional recursive NBT matching used by Java's `NBTUtil.func_181123_a`.
    ///
    /// Compounds in `self` are requirements: every key must exist and match in `actual`, while
    /// extra keys in `actual` are ignored. With `list_any == false`, lists retain ordinary exact
    /// equality. With `list_any == true`, every required list element may match any actual list
    /// element (without consuming it); an empty required list matches only an empty actual list.
    #[must_use]
    pub fn matches(&self, actual: &Nbt, list_any: bool) -> bool {
        match (self, actual) {
            (Nbt::Compound(expected), Nbt::Compound(actual)) => expected.matches(actual, list_any),
            (Nbt::List(expected), Nbt::List(actual)) if list_any => {
                if expected.is_empty() {
                    actual.is_empty()
                } else {
                    expected
                        .iter()
                        .all(|value| actual.iter().any(|actual| value.matches(actual, true)))
                }
            }
            (Nbt::List(expected), Nbt::List(actual)) => expected == actual,
            _ => self == actual,
        }
    }

    /// The `i8` payload, if this is a `TAG_Byte`.
    #[must_use]
    pub fn as_i8(&self) -> Option<i8> {
        match self {
            Nbt::Byte(v) => Some(*v),
            _ => None,
        }
    }

    /// Vanilla's convention for booleans: any numeric tag converted to a byte, where non-zero is
    /// true.
    ///
    /// java: `NBTTagCompound.getBoolean`
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        self.as_primitive_i8().map(|v| v != 0)
    }

    /// The `i16` payload, if this is a `TAG_Short`.
    #[must_use]
    pub fn as_i16(&self) -> Option<i16> {
        match self {
            Nbt::Short(v) => Some(*v),
            _ => None,
        }
    }

    /// The `i32` payload, if this is a `TAG_Int`.
    #[must_use]
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Nbt::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// The `i64` payload, if this is a `TAG_Long`.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Nbt::Long(v) => Some(*v),
            _ => None,
        }
    }

    /// The `f32` payload, if this is a `TAG_Float`.
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Nbt::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// The `f64` payload, if this is a `TAG_Double`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Nbt::Double(v) => Some(*v),
            _ => None,
        }
    }

    /// This numeric tag converted through Java's `NBTBase.NBTPrimitive.getByte()` contract.
    ///
    /// Unlike [`Self::as_i8`], this accepts every numeric NBT tag (ids 1 through 6). Floating
    /// values use the same floor-before-narrowing rule as Minecraft 1.8.9.
    #[must_use]
    pub fn as_primitive_i8(&self) -> Option<i8> {
        match self {
            Nbt::Byte(v) => Some(*v),
            Nbt::Short(v) => Some(*v as i8),
            Nbt::Int(v) => Some(*v as i8),
            Nbt::Long(v) => Some(*v as i8),
            Nbt::Float(v) => Some(java_floor_f32_to_i32(*v) as i8),
            Nbt::Double(v) => Some(java_floor_f64_to_i32(*v) as i8),
            _ => None,
        }
    }

    /// This numeric tag converted through Java's `NBTBase.NBTPrimitive.getShort()` contract.
    #[must_use]
    pub fn as_primitive_i16(&self) -> Option<i16> {
        match self {
            Nbt::Byte(v) => Some(i16::from(*v)),
            Nbt::Short(v) => Some(*v),
            Nbt::Int(v) => Some(*v as i16),
            Nbt::Long(v) => Some(*v as i16),
            Nbt::Float(v) => Some(java_floor_f32_to_i32(*v) as i16),
            Nbt::Double(v) => Some(java_floor_f64_to_i32(*v) as i16),
            _ => None,
        }
    }

    /// This numeric tag converted through Java's `NBTBase.NBTPrimitive.getInt()` contract.
    #[must_use]
    pub fn as_primitive_i32(&self) -> Option<i32> {
        match self {
            Nbt::Byte(v) => Some(i32::from(*v)),
            Nbt::Short(v) => Some(i32::from(*v)),
            Nbt::Int(v) => Some(*v),
            Nbt::Long(v) => Some(*v as i32),
            Nbt::Float(v) => Some(java_floor_f32_to_i32(*v)),
            Nbt::Double(v) => Some(java_floor_f64_to_i32(*v)),
            _ => None,
        }
    }

    /// This numeric tag converted through Java's `NBTBase.NBTPrimitive.getLong()` contract.
    #[must_use]
    pub fn as_primitive_i64(&self) -> Option<i64> {
        match self {
            Nbt::Byte(v) => Some(i64::from(*v)),
            Nbt::Short(v) => Some(i64::from(*v)),
            Nbt::Int(v) => Some(i64::from(*v)),
            Nbt::Long(v) => Some(*v),
            // Java's NBTTagFloat truncates here, while NBTTagDouble floors first.
            Nbt::Float(v) => Some(*v as i64),
            Nbt::Double(v) => Some(v.floor() as i64),
            _ => None,
        }
    }

    /// This numeric tag converted through Java's `NBTBase.NBTPrimitive.getFloat()` contract.
    #[must_use]
    pub fn as_primitive_f32(&self) -> Option<f32> {
        match self {
            Nbt::Byte(v) => Some(f32::from(*v)),
            Nbt::Short(v) => Some(f32::from(*v)),
            Nbt::Int(v) => Some(*v as f32),
            Nbt::Long(v) => Some(*v as f32),
            Nbt::Float(v) => Some(*v),
            Nbt::Double(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// This numeric tag converted through Java's `NBTBase.NBTPrimitive.getDouble()` contract.
    #[must_use]
    pub fn as_primitive_f64(&self) -> Option<f64> {
        match self {
            Nbt::Byte(v) => Some(f64::from(*v)),
            Nbt::Short(v) => Some(f64::from(*v)),
            Nbt::Int(v) => Some(f64::from(*v)),
            Nbt::Long(v) => Some(*v as f64),
            Nbt::Float(v) => Some(f64::from(*v)),
            Nbt::Double(v) => Some(*v),
            _ => None,
        }
    }

    /// The string payload, if this is a `TAG_String`.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Nbt::String(v) => Some(v),
            _ => None,
        }
    }

    /// The byte-array payload, if this is a `TAG_Byte_Array`.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Nbt::ByteArray(v) => Some(v),
            _ => None,
        }
    }

    /// The int-array payload, if this is a `TAG_Int_Array`.
    #[must_use]
    pub fn as_ints(&self) -> Option<&[i32]> {
        match self {
            Nbt::IntArray(v) => Some(v),
            _ => None,
        }
    }

    /// The elements, if this is a `TAG_List`.
    #[must_use]
    pub fn as_list(&self) -> Option<&[Nbt]> {
        match self {
            Nbt::List(v) => Some(v),
            _ => None,
        }
    }

    /// The elements, mutably, if this is a `TAG_List`.
    ///
    /// Prefer [`Self::append_tag`] and [`Self::set_tag`] when constructing list contents: those
    /// methods preserve vanilla's homogeneous-list invariant instead of deferring a mismatch to
    /// the writer.
    #[must_use]
    pub fn as_list_mut(&mut self) -> Option<&mut Vec<Nbt>> {
        match self {
            Nbt::List(v) => Some(&mut v.items),
            _ => None,
        }
    }

    /// Append to a `TAG_List` with `NBTTagList.appendTag` semantics.
    ///
    /// Returns `false` for a non-list receiver, `TAG_End`, or an element whose type differs from
    /// the list's first element. Vanilla logs and drops those additions; the boolean makes the
    /// same rejection observable without making it a write-time error.
    pub fn append_tag(&mut self, tag: Nbt) -> bool {
        let Nbt::List(items) = self else {
            return false;
        };
        let found = tag.tag_type();
        if found == TagType::End || (items.tag_type() != TagType::End && items.tag_type() != found)
        {
            return false;
        }
        if items.tag_type() == TagType::End {
            items.tag_type.store(found.id(), Ordering::Relaxed);
        }
        items.push(tag);
        true
    }

    /// Replace a `TAG_List` element with `NBTTagList.set` semantics.
    ///
    /// Out-of-range indices, `TAG_End`, non-list receivers, and mismatching element types leave
    /// the list untouched and return `false`, matching vanilla's warn-and-return branches.
    pub fn set_tag(&mut self, index: usize, tag: Nbt) -> bool {
        let Nbt::List(items) = self else {
            return false;
        };
        let found = tag.tag_type();
        if index >= items.len()
            || found == TagType::End
            || (items.tag_type() != TagType::End && items.tag_type() != found)
        {
            return false;
        }
        if items.tag_type() == TagType::End {
            items.tag_type.store(found.id(), Ordering::Relaxed);
        }
        items[index] = tag;
        true
    }

    /// Remove and return a `TAG_List` element, or `None` for a non-list/out-of-range index.
    ///
    /// Java's `removeTag` exposes `List.remove`'s out-of-bounds exception. Rust callers receive
    /// an `Option` instead, while valid removals and retained element ordering are identical.
    pub fn remove_tag(&mut self, index: usize) -> Option<Nbt> {
        let Nbt::List(items) = self else {
            return None;
        };
        (index < items.len()).then(|| items.remove(index))
    }

    /// The entries, if this is a `TAG_Compound`.
    #[must_use]
    pub fn as_compound(&self) -> Option<&NbtCompound> {
        match self {
            Nbt::Compound(v) => Some(v),
            _ => None,
        }
    }

    /// The entries, mutably, if this is a `TAG_Compound`.
    #[must_use]
    pub fn as_compound_mut(&mut self) -> Option<&mut NbtCompound> {
        match self {
            Nbt::Compound(v) => Some(v),
            _ => None,
        }
    }

    /// The list's stored element type, or `None` if this is not a `TAG_List`.
    ///
    /// java: `NBTTagList.getTagType`. An empty list read from the wire or one emptied with
    /// [`Self::remove_tag`] retains this stored type until Java's write path resets it to End.
    #[must_use]
    pub fn list_tag_type(&self) -> Option<TagType> {
        match self {
            Nbt::List(items) => Some(items.tag_type()),
            _ => None,
        }
    }

    /// The list element at `index`, or `TAG_End` for a non-list or out-of-range index.
    ///
    /// java: `NBTTagList.get`
    #[must_use]
    pub fn list_tag_at(&self, index: usize) -> &Nbt {
        static END: Nbt = Nbt::End;
        self.as_list()
            .and_then(|items| items.get(index))
            .unwrap_or(&END)
    }

    /// The compound list element at `index`, or an immutable empty compound when absent or typed
    /// differently. Immutability gives the same no-shared-mutation guarantee as Java's freshly
    /// allocated default without allocating on every miss.
    ///
    /// java: `NBTTagList.getCompoundTagAt`
    #[must_use]
    pub fn list_compound_at(&self, index: usize) -> &NbtCompound {
        static EMPTY: LazyLock<NbtCompound> = LazyLock::new(NbtCompound::new);
        self.as_list()
            .and_then(|items| items.get(index))
            .and_then(Nbt::as_compound)
            .unwrap_or(&EMPTY)
    }

    /// The int-array list element at `index`, or an empty array when absent or typed differently.
    ///
    /// java: `NBTTagList.getIntArrayAt`
    #[must_use]
    pub fn list_int_array_at(&self, index: usize) -> &[i32] {
        self.as_list()
            .and_then(|items| items.get(index))
            .and_then(Nbt::as_ints)
            .unwrap_or(&[])
    }

    /// The exact `TAG_Double` list element at `index`, or zero when absent or typed differently.
    ///
    /// java: `NBTTagList.getDoubleAt`
    #[must_use]
    pub fn list_double_at(&self, index: usize) -> f64 {
        self.as_list()
            .and_then(|items| items.get(index))
            .and_then(Nbt::as_f64)
            .unwrap_or(0.0)
    }

    /// The exact `TAG_Float` list element at `index`, or zero when absent or typed differently.
    ///
    /// java: `NBTTagList.getFloatAt`
    #[must_use]
    pub fn list_float_at(&self, index: usize) -> f32 {
        self.as_list()
            .and_then(|items| items.get(index))
            .and_then(Nbt::as_f32)
            .unwrap_or(0.0)
    }

    /// The string value at `index`, or the element's vanilla `toString()` representation when it
    /// has another type. Only an absent element returns the empty-string default.
    ///
    /// java: `NBTTagList.getStringTagAt`
    #[must_use]
    pub fn list_string_at(&self, index: usize) -> std::borrow::Cow<'_, str> {
        match self.as_list().and_then(|items| items.get(index)) {
            Some(Nbt::String(value)) => std::borrow::Cow::Borrowed(value),
            Some(value) => std::borrow::Cow::Owned(value.to_string()),
            None => std::borrow::Cow::Borrowed(""),
        }
    }

    /// Bit-exact equality: unlike the derived [`PartialEq`], compares `Float`/`Double` payloads
    /// by [`f32::to_bits`]/[`f64::to_bits`] rather than `==`.
    ///
    /// The derived impl inherits IEEE 754's two surprises for a format meant to round-trip
    /// exactly: `NaN != NaN` (so a perfect byte-for-byte round-trip reads as *unequal*), and
    /// `+0.0 == -0.0` (so a **lost sign bit reads as equal**). Both bit `mc-parity`'s NBT fuzzer
    /// on its first run — this is that fix, lifted from `mc_parity::fuzz::nbt_bit_eq` into the
    /// crate the round-trip guarantee actually belongs to.
    #[must_use]
    pub fn bit_eq(&self, other: &Nbt) -> bool {
        match (self, other) {
            (Nbt::Float(a), Nbt::Float(b)) => a.to_bits() == b.to_bits(),
            (Nbt::Double(a), Nbt::Double(b)) => a.to_bits() == b.to_bits(),
            (Nbt::List(a), Nbt::List(b)) => {
                a.tag_type() == b.tag_type()
                    && a.len() == b.len()
                    && a.iter().zip(b).all(|(x, y)| x.bit_eq(y))
            }
            (Nbt::Compound(a), Nbt::Compound(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|((ka, va), (kb, vb))| ka == kb && va.bit_eq(vb))
            }
            _ => self == other,
        }
    }
}

impl From<NbtCompound> for Nbt {
    fn from(c: NbtCompound) -> Self {
        Nbt::Compound(c)
    }
}

/// A `TAG_Compound`: string keys to values, **in insertion order**.
///
/// Vanilla backs this with a `HashMap`, so the order it writes is whatever the JVM's hash
/// order happens to be. We keep insertion order instead, which costs nothing and buys the
/// one property the golden tests need: a file read and immediately written back is
/// byte-identical.
///
/// Equality follows Java's `Map.equals` and ignores insertion order; iteration and writing retain
/// insertion order for stable binary round trips.
///
/// java: `NBTTagCompound.java`
#[derive(Debug, Clone, Default)]
pub struct NbtCompound {
    entries: Vec<(String, Nbt)>,
    index: HashMap<String, usize>,
}

impl PartialEq for NbtCompound {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(key, value)| other.get(key).is_some_and(|other| value == other))
    }
}

impl NbtCompound {
    /// A compound with no entries.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A compound with room for `n` entries.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            entries: Vec::with_capacity(n),
            index: HashMap::with_capacity(n),
        }
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the compound has no entries. java: `NBTBase.hasNoTags()`
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
    }

    /// Insert or replace `key`, returning the previous value if there was one.
    ///
    /// Replacing keeps the original position, so re-serialising is stable.
    pub fn insert(&mut self, key: impl Into<String>, value: Nbt) -> Option<Nbt> {
        let key = key.into();
        match self.index.get(&key) {
            Some(&i) => Some(std::mem::replace(&mut self.entries[i].1, value)),
            None => {
                self.index.insert(key.clone(), self.entries.len());
                self.entries.push((key, value));
                None
            }
        }
    }

    /// Remove `key`, returning its value. Later entries shift down, preserving order.
    pub fn remove(&mut self, key: &str) -> Option<Nbt> {
        let i = self.index.remove(key)?;
        let (_, value) = self.entries.remove(i);
        for slot in self.index.values_mut() {
            if *slot > i {
                *slot -= 1;
            }
        }
        Some(value)
    }

    /// The value stored under `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Nbt> {
        self.index.get(key).map(|&i| &self.entries[i].1)
    }

    /// The value stored under `key`, mutably.
    #[must_use]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Nbt> {
        let i = *self.index.get(key)?;
        Some(&mut self.entries[i].1)
    }

    /// Whether `key` is present. java: `NBTTagCompound.hasKey`
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// Whether `key` is present *and* holds the given type.
    ///
    /// java: `NBTTagCompound.hasKey(String, int)`
    #[must_use]
    pub fn contains_type(&self, key: &str, ty: TagType) -> bool {
        self.get(key).is_some_and(|v| v.tag_type() == ty)
    }

    /// Whether `key` holds one of Java NBT's six primitive numeric tag types.
    ///
    /// java: `NBTTagCompound.hasKey(key, 99)`. The wildcard `99` is an API selector, not a wire
    /// tag id, so it deliberately remains outside [`TagType`].
    #[must_use]
    pub fn contains_numeric_type(&self, key: &str) -> bool {
        self.get(key).is_some_and(|value| {
            matches!(
                value,
                Nbt::Byte(_)
                    | Nbt::Short(_)
                    | Nbt::Int(_)
                    | Nbt::Long(_)
                    | Nbt::Float(_)
                    | Nbt::Double(_)
            )
        })
    }

    /// Directional recursive subset matching for compound roots.
    #[must_use]
    pub fn matches(&self, actual: &NbtCompound, list_any: bool) -> bool {
        self.iter().all(|(key, value)| {
            actual
                .get(key)
                .is_some_and(|actual| value.matches(actual, list_any))
        })
    }

    /// The entries as a slice, in insertion order.
    #[must_use]
    pub fn as_slice(&self) -> &[(String, Nbt)] {
        &self.entries
    }

    /// Entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Nbt)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Entries in insertion order, with mutable values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut Nbt)> {
        self.entries.iter_mut().map(|(k, v)| (k.as_str(), v))
    }

    /// Keys in insertion order. java: `NBTTagCompound.getKeySet`
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }

    /// The numeric tag under `key`, converted like `NBTTagCompound.getByte`.
    #[must_use]
    pub fn get_i8(&self, key: &str) -> Option<i8> {
        self.get(key).and_then(Nbt::as_primitive_i8)
    }

    /// The `TAG_Byte` under `key` read as a boolean. java: `NBTTagCompound.getBoolean`
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(Nbt::as_bool)
    }

    /// The numeric tag under `key`, converted like `NBTTagCompound.getShort`.
    #[must_use]
    pub fn get_i16(&self, key: &str) -> Option<i16> {
        self.get(key).and_then(Nbt::as_primitive_i16)
    }

    /// The numeric tag under `key`, converted like `NBTTagCompound.getInteger`.
    #[must_use]
    pub fn get_i32(&self, key: &str) -> Option<i32> {
        self.get(key).and_then(Nbt::as_primitive_i32)
    }

    /// The numeric tag under `key`, converted like `NBTTagCompound.getLong`.
    #[must_use]
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(Nbt::as_primitive_i64)
    }

    /// The numeric tag under `key`, converted like `NBTTagCompound.getFloat`.
    #[must_use]
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(Nbt::as_primitive_f32)
    }

    /// The numeric tag under `key`, converted like `NBTTagCompound.getDouble`.
    #[must_use]
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(Nbt::as_primitive_f64)
    }

    /// The `TAG_String` under `key`. java: `NBTTagCompound.getString`
    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Nbt::as_str)
    }

    /// The `TAG_Byte_Array` under `key`. java: `NBTTagCompound.getByteArray`
    #[must_use]
    pub fn get_bytes(&self, key: &str) -> Option<&[u8]> {
        self.get(key).and_then(Nbt::as_bytes)
    }

    /// The `TAG_Int_Array` under `key`. java: `NBTTagCompound.getIntArray`
    #[must_use]
    pub fn get_ints(&self, key: &str) -> Option<&[i32]> {
        self.get(key).and_then(Nbt::as_ints)
    }

    /// The `TAG_List` under `key`. java: `NBTTagCompound.getTagList`
    #[must_use]
    pub fn get_list(&self, key: &str) -> Option<&[Nbt]> {
        self.get(key).and_then(Nbt::as_list)
    }

    /// The `TAG_Compound` under `key`. java: `NBTTagCompound.getCompoundTag`
    #[must_use]
    pub fn get_compound(&self, key: &str) -> Option<&NbtCompound> {
        self.get(key).and_then(Nbt::as_compound)
    }
}

impl FromIterator<(String, Nbt)> for NbtCompound {
    fn from_iter<T: IntoIterator<Item = (String, Nbt)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut out = NbtCompound::with_capacity(iter.size_hint().0);
        for (k, v) in iter {
            out.insert(k, v);
        }
        out
    }
}

impl<'a> IntoIterator for &'a NbtCompound {
    type Item = (&'a str, &'a Nbt);
    type IntoIter = Box<dyn Iterator<Item = (&'a str, &'a Nbt)> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

impl IntoIterator for NbtCompound {
    type Item = (String, Nbt);
    type IntoIter = std::vec::IntoIter<(String, Nbt)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_type_round_trips_through_its_byte() {
        for ty in TagType::ALL {
            assert_eq!(TagType::from_id(ty.id()), Some(ty));
        }
        assert_eq!(TagType::from_id(12), None);
        assert_eq!(TagType::from_id(99), None);
        assert_eq!(TagType::from_id(255), None);
    }

    #[test]
    fn bit_eq_catches_what_derived_partial_eq_gets_wrong() {
        // NaN != NaN under `==`, so a perfect round-trip reads as a *failure* with the derived
        // impl; `bit_eq` compares bit patterns, so a NaN round-trips as equal to itself.
        let nan_a = Nbt::Float(f32::NAN);
        let nan_b = Nbt::Float(f32::from_bits(f32::NAN.to_bits())); // same bits, fresh value
        assert_ne!(
            nan_a, nan_a,
            "sanity: derived PartialEq really does say NaN != NaN"
        );
        assert!(nan_a.bit_eq(&nan_b));

        // +0.0 == -0.0 under `==`, so a lost sign bit reads as a *success* with the derived impl.
        let pos_zero = Nbt::Double(0.0);
        let neg_zero = Nbt::Double(-0.0);
        assert_eq!(
            pos_zero, neg_zero,
            "sanity: derived PartialEq really does say +0.0 == -0.0"
        );
        assert!(
            !pos_zero.bit_eq(&neg_zero),
            "a lost sign bit must not read as equal"
        );

        // Nested inside a list and a compound, since those recurse.
        let a = Nbt::list(vec![Nbt::Float(f32::NAN)]);
        let b = Nbt::list(vec![Nbt::Float(f32::NAN)]);
        assert!(a.bit_eq(&b));
        let mut ca = NbtCompound::new();
        ca.insert("x", Nbt::Double(0.0));
        let mut cb = NbtCompound::new();
        cb.insert("x", Nbt::Double(-0.0));
        assert!(!Nbt::Compound(ca).bit_eq(&Nbt::Compound(cb)));

        // Ordinary values still compare equal, and unequal values still compare unequal.
        assert!(Nbt::Int(5).bit_eq(&Nbt::Int(5)));
        assert!(!Nbt::Int(5).bit_eq(&Nbt::Int(6)));
    }

    #[test]
    fn compound_numeric_getters_follow_nbt_primitive_cross_conversions() {
        let mut c = NbtCompound::new();
        c.insert("byte", Nbt::Byte(-2));
        c.insert("short", Nbt::Short(0x1234));
        c.insert("int", Nbt::Int(0x1234_5678));
        c.insert("long", Nbt::Long(0x1_0000_8001));
        c.insert("float", Nbt::Float(-2.5));
        c.insert("double", Nbt::Double(-2.5));
        c.insert("text", Nbt::String("8".to_owned()));
        c.insert("bytes", Nbt::ByteArray(vec![1]));
        c.insert("list", Nbt::list(vec![Nbt::Int(1)]));
        c.insert("compound", Nbt::Compound(NbtCompound::new()));
        c.insert("ints", Nbt::IntArray(vec![1]));
        c.insert("end", Nbt::End);

        for key in ["byte", "short", "int", "long", "float", "double"] {
            assert!(
                c.contains_numeric_type(key),
                "{key} must satisfy wildcard 99"
            );
        }
        for key in [
            "text", "bytes", "list", "compound", "ints", "end", "missing",
        ] {
            assert!(
                !c.contains_numeric_type(key),
                "{key} must not satisfy wildcard 99"
            );
        }

        assert_eq!(c.get_i32("byte"), Some(-2));
        assert_eq!(c.get_i8("short"), Some(0x34));
        assert_eq!(c.get_i16("int"), Some(0x5678));
        assert_eq!(c.get_i8("long"), Some(1));
        assert_eq!(c.get_i16("long"), Some(-32_767));
        assert_eq!(c.get_i32("long"), Some(32_769));
        assert_eq!(c.get_i32("float"), Some(-3));
        assert_eq!(c.get_i64("float"), Some(-2));
        assert_eq!(c.get_i32("double"), Some(-3));
        assert_eq!(c.get_i64("double"), Some(-3));
        assert_eq!(c.get_f64("float"), Some(-2.5));
        assert_eq!(c.get_f32("int"), Some(0x1234_5678_i32 as f32));
        assert_eq!(c.get_i32("text"), None);
        assert_eq!(c.get_i32("missing"), None);

        // The type-exact accessors remain available for callers that are not modelling a Java
        // NBTTagCompound numeric getter.
        assert_eq!(c.get("int").and_then(Nbt::as_i16), None);
    }

    #[test]
    fn recursive_nbt_match_supports_exact_and_any_list_modes() {
        let mut expected_nested = NbtCompound::new();
        expected_nested.insert("required", Nbt::Int(7));
        let mut expected = NbtCompound::new();
        expected.insert("nested", Nbt::Compound(expected_nested));
        expected.insert("list", Nbt::list(vec![Nbt::Int(1), Nbt::Int(2)]));

        let mut actual_nested = NbtCompound::new();
        actual_nested.insert("extra", Nbt::String("kept".into()));
        actual_nested.insert("required", Nbt::Int(7));
        let mut actual = NbtCompound::new();
        actual.insert("extra_root", Nbt::Byte(1));
        actual.insert("nested", Nbt::Compound(actual_nested));
        actual.insert(
            "list",
            Nbt::list(vec![Nbt::Int(2), Nbt::Int(1), Nbt::Int(3)]),
        );

        assert!(
            !expected.matches(&actual, false),
            "false mode keeps lists exact"
        );
        assert!(
            expected.matches(&actual, true),
            "true mode uses any-element lists"
        );

        let mut missing = actual.clone();
        missing.remove("nested");
        assert!(!expected.matches(&missing, true));

        let mut wrong_type = actual.clone();
        wrong_type.insert("nested", Nbt::Int(7));
        assert!(!expected.matches(&wrong_type, true));

        assert!(Nbt::list(vec![]).matches(&Nbt::list(vec![]), true));
        assert!(!Nbt::list(vec![]).matches(&Nbt::list(vec![Nbt::Int(1)]), true));
    }

    #[test]
    fn nbt_primitive_float_edges_match_java_casts_and_math_helper_floor() {
        assert_eq!(Nbt::Float(f32::NAN).as_primitive_i32(), Some(0));
        assert_eq!(Nbt::Double(f64::NAN).as_primitive_i64(), Some(0));
        assert_eq!(Nbt::Float(f32::INFINITY).as_primitive_i32(), Some(i32::MAX));
        assert_eq!(
            Nbt::Double(f64::INFINITY).as_primitive_i64(),
            Some(i64::MAX)
        );
        assert_eq!(
            Nbt::Float(f32::NEG_INFINITY).as_primitive_i32(),
            Some(i32::MAX),
            "MathHelper.floor_float wraps MIN_VALUE - 1 just like Java"
        );
        assert_eq!(
            Nbt::Double(f64::NEG_INFINITY).as_primitive_i32(),
            Some(i32::MAX),
            "MathHelper.floor_double wraps MIN_VALUE - 1 just like Java"
        );
        assert_eq!(
            Nbt::Double(f64::NEG_INFINITY).as_primitive_i64(),
            Some(i64::MIN),
            "NBTTagDouble.getLong floors and then uses Java's saturating double-to-long cast"
        );
    }

    #[test]
    fn nbt_primitive_double_float_narrowing_matches_java_masks_and_long_rules() {
        // NBTTagFloat.getLong truncates toward zero, while its int/short/byte methods
        // floor through MathHelper first. NBTTagDouble.getLong is the deliberate exception:
        // it floors before the long cast as well.
        assert_eq!(Nbt::Float(-0.25).as_primitive_i64(), Some(0));
        assert_eq!(Nbt::Float(-0.25).as_primitive_i32(), Some(-1));
        assert_eq!(Nbt::Double(-0.25).as_primitive_i64(), Some(-1));
        assert_eq!(Nbt::Double(-0.25).as_primitive_i32(), Some(-1));

        // Java masks the floored int before narrowing to byte/short. Rust's wrapping casts
        // produce the same low-bit result, including values beyond the signed target range.
        assert_eq!(Nbt::Float(257.9).as_primitive_i16(), Some(257));
        assert_eq!(Nbt::Float(257.9).as_primitive_i8(), Some(1));
        assert_eq!(Nbt::Double(65_536.2).as_primitive_i16(), Some(0));
        assert_eq!(Nbt::Double(65_536.2).as_primitive_i8(), Some(0));
        assert_eq!(Nbt::Double(-1.1).as_primitive_i16(), Some(-2));
        assert_eq!(Nbt::Double(-1.1).as_primitive_i8(), Some(-2));

        // Java's float/double floor helpers wrap i32::MIN - 1 at negative infinity before
        // the low-bit mask; positive and negative infinity therefore both narrow to -1.
        assert_eq!(Nbt::Float(f32::INFINITY).as_primitive_i8(), Some(-1));
        assert_eq!(Nbt::Float(f32::NEG_INFINITY).as_primitive_i16(), Some(-1));
        assert_eq!(Nbt::Double(f64::INFINITY).as_primitive_i8(), Some(-1));
        assert_eq!(Nbt::Double(f64::NEG_INFINITY).as_primitive_i16(), Some(-1));
    }

    #[test]
    fn compound_keeps_insertion_order() {
        let mut c = NbtCompound::new();
        c.insert("z", Nbt::Byte(1));
        c.insert("a", Nbt::Byte(2));
        c.insert("m", Nbt::Byte(3));
        assert_eq!(c.keys().collect::<Vec<_>>(), ["z", "a", "m"]);
    }

    #[test]
    fn compound_equality_ignores_insertion_order_like_java_map_equals() {
        let mut a = NbtCompound::new();
        a.insert("first", Nbt::Int(1));
        a.insert("second", Nbt::String("two".to_owned()));

        let mut b = NbtCompound::new();
        b.insert("second", Nbt::String("two".to_owned()));
        b.insert("first", Nbt::Int(1));
        assert_eq!(a, b);

        b.insert("first", Nbt::Int(3));
        assert_ne!(a, b);
    }

    #[test]
    fn replacing_a_key_keeps_its_position() {
        let mut c = NbtCompound::new();
        c.insert("a", Nbt::Byte(1));
        c.insert("b", Nbt::Byte(2));
        assert_eq!(c.insert("a", Nbt::Byte(9)), Some(Nbt::Byte(1)));
        assert_eq!(c.keys().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(c.get_i8("a"), Some(9));
    }

    #[test]
    fn remove_reindexes_the_survivors() {
        let mut c = NbtCompound::new();
        c.insert("a", Nbt::Byte(1));
        c.insert("b", Nbt::Byte(2));
        c.insert("c", Nbt::Byte(3));
        assert_eq!(c.remove("a"), Some(Nbt::Byte(1)));
        assert_eq!(c.get_i8("b"), Some(2));
        assert_eq!(c.get_i8("c"), Some(3));
        assert_eq!(c.remove("a"), None);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn list_mutators_drop_end_mismatches_and_out_of_range_sets_like_java() {
        let mut list = Nbt::list(Vec::new());
        assert!(!list.append_tag(Nbt::End));
        assert!(list.append_tag(Nbt::Int(3)));
        assert!(!list.append_tag(Nbt::String("wrong".to_owned())));
        assert_eq!(list.as_list().unwrap(), &[Nbt::Int(3)]);

        assert!(!list.set_tag(1, Nbt::Int(4)));
        assert!(!list.set_tag(0, Nbt::Short(4)));
        assert!(list.set_tag(0, Nbt::Int(4)));
        assert_eq!(list.as_list().unwrap(), &[Nbt::Int(4)]);

        assert_eq!(list.remove_tag(0), Some(Nbt::Int(4)));
        assert_eq!(list.remove_tag(0), None);
        assert!(list.as_list().unwrap().is_empty());

        let mut scalar = Nbt::Int(1);
        assert!(!scalar.append_tag(Nbt::Int(2)));
        assert!(!scalar.set_tag(0, Nbt::Int(2)));
        assert_eq!(scalar.remove_tag(0), None);
    }

    #[test]
    fn empty_and_emptied_lists_retain_java_tag_type_until_write() {
        // A list's wire type lives in NBTTagList.tagType even when its backing list is empty.
        // This is a valid root TAG_List named "" whose element type is TAG_Int and count is zero.
        let typed_empty =
            crate::from_slice(&[TagType::List.id(), 0, 0, TagType::Int.id(), 0, 0, 0, 0])
                .expect("zero-length typed list is valid")
                .1;
        assert!(typed_empty.as_list().is_some_and(|items| items.is_empty()));
        assert_eq!(typed_empty.list_tag_type(), Some(TagType::Int));
        assert_eq!(typed_empty.clone().list_tag_type(), Some(TagType::Int));

        // NBTTagList.equals compares its private tagType byte as well as its backing List.
        assert_ne!(typed_empty, Nbt::list(Vec::new()));

        // NBTTagList.write recomputes tagType from its first element. An empty one is reset to
        // TAG_End both on wire and in the in-memory object afterwards.
        assert_eq!(
            crate::to_vec("", &typed_empty).expect("write typed empty list"),
            vec![TagType::List.id(), 0, 0, TagType::End.id(), 0, 0, 0, 0]
        );
        assert_eq!(typed_empty.list_tag_type(), Some(TagType::End));
        assert_eq!(typed_empty, Nbt::list(Vec::new()));

        let mut emptied = Nbt::list(vec![Nbt::Int(9)]);
        assert_eq!(emptied.remove_tag(0), Some(Nbt::Int(9)));
        assert_eq!(emptied.list_tag_type(), Some(TagType::Int));
        assert!(!emptied.append_tag(Nbt::String("wrong type".to_owned())));
        assert!(emptied.append_tag(Nbt::Int(10)));
    }

    #[test]
    fn indexed_list_accessors_use_vanillas_exact_type_checks_and_defaults() {
        let mut compound = NbtCompound::new();
        compound.insert("answer", Nbt::Int(42));
        let list = Nbt::list(vec![
            Nbt::Compound(compound),
            Nbt::IntArray(vec![1, -2]),
            Nbt::Double(3.5),
            Nbt::Float(-4.25),
            Nbt::String("literal".to_owned()),
            Nbt::Int(7),
        ]);

        assert_eq!(list.list_compound_at(0).get_i32("answer"), Some(42));
        assert_eq!(list.list_int_array_at(1), &[1, -2]);
        assert_eq!(list.list_double_at(2), 3.5);
        assert_eq!(list.list_float_at(3), -4.25);
        assert_eq!(list.list_string_at(4), "literal");
        assert_eq!(list.list_string_at(5), "7");
        assert_eq!(list.list_tag_at(5), &Nbt::Int(7));

        // Negative controls: Java does not numerically widen these list accessors. Replacing the
        // exact-type calls with the NBTPrimitive conversions makes all four assertions fail.
        assert!(list.list_compound_at(1).is_empty());
        assert_eq!(list.list_int_array_at(0), &[]);
        assert_eq!(list.list_double_at(3), 0.0);
        assert_eq!(list.list_float_at(2), 0.0);

        // Missing elements use non-null vanilla defaults; a wrong-typed in-range string is the
        // tag's rendered form instead, so collapsing both cases to `""` is caught here.
        assert_eq!(list.list_string_at(99), "");
        assert_eq!(list.list_tag_at(99), &Nbt::End);
        assert!(list.list_compound_at(99).is_empty());
        assert_eq!(list.list_int_array_at(99), &[]);

        let scalar = Nbt::Int(9);
        assert_eq!(scalar.list_string_at(0), "");
        assert_eq!(scalar.list_tag_at(0), &Nbt::End);
    }

    #[test]
    fn display_matches_vanillas_legacy_nbt_to_string_forms() {
        let mut compound = NbtCompound::new();
        compound.insert("raw key", Nbt::Byte(-1));
        compound.insert(
            "nested",
            Nbt::list(vec![
                Nbt::String("quote \" slash \\ newline\n".to_owned()),
                Nbt::String("plain".to_owned()),
            ]),
        );

        assert_eq!(Nbt::End.to_string(), "END");
        assert_eq!(Nbt::Short(i16::MIN).to_string(), "-32768s");
        assert_eq!(Nbt::Long(i64::MAX).to_string(), "9223372036854775807L");
        assert_eq!(Nbt::ByteArray(vec![0, 255, 1]).to_string(), "[3 bytes]");
        assert_eq!(Nbt::IntArray(vec![1, -2, 3]).to_string(), "[1,-2,3,]");
        assert_eq!(Nbt::IntArray(Vec::new()).to_string(), "[]");
        assert_eq!(
            Nbt::Compound(compound).to_string(),
            "{raw key:-1b,nested:[0:\"quote \\\" slash \\ newline\n\",1:\"plain\"]}"
        );

        // Negative controls for the tempting modern-SNBT implementation: vanilla retains list
        // indices and the int-array trailing comma, and escapes quotes only. A JSON/SNBT escaper
        // would double the backslash and replace the real newline, failing the nested assertion.
        assert_ne!(
            Nbt::list(vec![Nbt::Int(1), Nbt::Int(2)]).to_string(),
            "[1,2]"
        );
        assert_eq!(
            Nbt::list(vec![Nbt::Int(1), Nbt::Int(2)]).to_string(),
            "[0:1,1:2]"
        );
    }

    #[test]
    fn display_uses_java_float_notation_boundaries_and_special_values() {
        assert_eq!(Nbt::Float(0.0).to_string(), "0.0f");
        assert_eq!(Nbt::Float(-0.0).to_string(), "-0.0f");
        assert_eq!(Nbt::Float(0.001).to_string(), "0.001f");
        assert_eq!(Nbt::Float(0.0001).to_string(), "1.0E-4f");
        assert_eq!(Nbt::Float(9_999_999.0).to_string(), "9999999.0f");
        assert_eq!(Nbt::Float(10_000_000.0).to_string(), "1.0E7f");
        assert_eq!(Nbt::Double(1.25e20).to_string(), "1.25E20d");
        assert_eq!(Nbt::Double(f64::NAN).to_string(), "NaNd");
        assert_eq!(Nbt::Double(f64::INFINITY).to_string(), "Infinityd");
        assert_eq!(Nbt::Double(f64::NEG_INFINITY).to_string(), "-Infinityd");

        // Rust spells this `inf` and commonly chooses different exponent notation. This assertion
        // bites if Display merely delegates to Rust and appends the NBT suffix.
        assert_ne!(Nbt::Double(f64::INFINITY).to_string(), "infd");
        assert_ne!(Nbt::Double(10_000_000.0).to_string(), "10000000d");
    }
}
