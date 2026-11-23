//! `serde` integration, behind the `serde` feature.
//!
//! Two layers:
//!
//! * [`Nbt`](crate::Nbt) and [`NbtCompound`](crate::NbtCompound) implement `Serialize` and
//!   `Deserialize` themselves, so an NBT value can be dumped to JSON or read from any
//!   self-describing format.
//! * [`to_nbt`] and [`from_nbt`] map ordinary Rust types to and from `Nbt`, so a packet
//!   payload or a `.dat` file can be a `#[derive(Serialize, Deserialize)]` struct.
//!
//! # Mapping
//!
//! | Rust | NBT |
//! |---|---|
//! | `bool` | `TAG_Byte`, 0 / 1 — vanilla's own convention |
//! | `i8` `i16` `i32` `i64` | `TAG_Byte` `TAG_Short` `TAG_Int` `TAG_Long` |
//! | `u8` `u16` `u32` `u64` | the same tags, **bit-reinterpreted** |
//! | `f32` `f64` | `TAG_Float` `TAG_Double` |
//! | `char`, `String`, `&str` | `TAG_String` |
//! | `&[u8]` via `serde_bytes` | `TAG_Byte_Array` |
//! | `Vec<T>`, tuples | `TAG_List` |
//! | structs, maps | `TAG_Compound` |
//! | fieldless enum variant | `TAG_String` |
//! | data-carrying enum variant | single-entry `TAG_Compound` |
//!
//! NBT has no unsigned types: the game stores unsigned quantities in signed tags and
//! reinterprets the bits, so `u8 → TAG_Byte` is the vanilla-faithful mapping even though
//! `200u8` reads back as `-56i8` in a tag dump. It round-trips exactly.
//!
//! NBT also has no null. `None` and `()` cannot be serialised; use
//! `#[serde(skip_serializing_if = "Option::is_none")]`. On the way back, a missing key
//! deserialises to `None` as usual.

mod de;
mod ser;
mod value;

pub use de::{from_nbt, NbtDeserializer};
pub use ser::{to_nbt, NbtSerializer};
