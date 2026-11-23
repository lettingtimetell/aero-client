//! Blockstate JSON: `assets/minecraft/blockstates/*.json`.
//!
// java: ModelBlockDefinition.java, ModelBlockDefinition.Variant.java
//!
//! 1.8.9 blockstate files use only the `variants` form (`multipart` arrived in 1.9, so it is
//! out of scope here). Each key is either a single variant object or a weighted list of them.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer};

use crate::ResourceLocation;

/// One `x`/`y` rotation of a baked block model, in 90-degree steps.
///
/// java: `ModelRotation` — only the 16 `(x, y) in {0,90,180,270}^2` combinations are legal;
/// [`ModelRotation::from_xy`] is the deserializer's validation point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelRotation {
    /// Quarter turns about X, `0..=3`.
    pub quarters_x: u8,
    /// Quarter turns about Y, `0..=3`.
    pub quarters_y: u8,
}

impl ModelRotation {
    /// No rotation.
    pub const IDENTITY: ModelRotation = ModelRotation {
        quarters_x: 0,
        quarters_y: 0,
    };

    /// Builds a rotation from degrees, validating that both are multiples of 90.
    ///
    /// # Errors
    /// `None` if either angle is not a multiple of 90 degrees.
    #[must_use]
    pub fn from_xy(x: i32, y: i32) -> Option<ModelRotation> {
        if x % 90 != 0 || y % 90 != 0 {
            return None;
        }
        let norm = |v: i32| -> u8 { (v.rem_euclid(360) / 90) as u8 };
        Some(ModelRotation {
            quarters_x: norm(x),
            quarters_y: norm(y),
        })
    }
}

/// One entry of a (possibly weighted) blockstate variant list.
// java: ModelBlockDefinition.Variant
#[derive(Debug, Clone)]
pub struct Variant {
    /// The model to bake, already namespaced+prefixed to `<ns>:block/<path>`.
    pub model: ResourceLocation,
    /// Model rotation.
    pub rotation: ModelRotation,
    /// Whether face UVs should be re-derived from world-space after rotation, so a rotated
    /// texture still tiles with its neighbours instead of spinning with the model.
    pub uvlock: bool,
    /// Relative weight in a multi-variant list. `1` when absent.
    ///
    /// This is Java's signed `int`, not an unsigned count. Zero, negative values, and overflow
    /// narrowing are all observable later in `WeightedBakedModel`.
    pub weight: i32,
}

// java: ModelBlockDefinition.Variant.equals/hashCode intentionally omit `weight`.
// Weight affects selection, but it is not part of a variant's model identity.
impl PartialEq for Variant {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model && self.rotation == other.rotation && self.uvlock == other.uvlock
    }
}

impl Eq for Variant {}

impl Hash for Variant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.model.hash(state);
        self.rotation.hash(state);
        self.uvlock.hash(state);
    }
}

/// A blockstate's full `variants` map: variant key -> weighted candidate list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BlockStateJson {
    /// Keyed by the exact `state_variant_key()` string, e.g. `"facing=east,half=bottom"`.
    pub variants: HashMap<String, Vec<Variant>>,
}

impl<'de> Deserialize<'de> for BlockStateJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            variants: HashMap<String, RawVariantOrList>,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum RawVariantOrList {
            One(RawVariant),
            Many(Vec<RawVariant>),
        }

        #[derive(Deserialize)]
        struct RawVariant {
            #[serde(deserialize_with = "deserialize_gson_string")]
            model: String,
            #[serde(default, deserialize_with = "deserialize_java_int")]
            x: i32,
            #[serde(default, deserialize_with = "deserialize_java_int")]
            y: i32,
            #[serde(default, deserialize_with = "deserialize_gson_bool")]
            uvlock: bool,
            #[serde(default = "default_weight", deserialize_with = "deserialize_java_int")]
            weight: i32,
        }

        fn default_weight() -> i32 {
            1
        }

        let raw = Raw::deserialize(deserializer)?;
        let mut variants = HashMap::with_capacity(raw.variants.len());
        for (key, value) in raw.variants {
            let list = match value {
                RawVariantOrList::One(v) => vec![v],
                RawVariantOrList::Many(v) => v,
            };
            let mut out = Vec::with_capacity(list.len());
            for v in list {
                let rotation = ModelRotation::from_xy(v.x, v.y).ok_or_else(|| {
                    D::Error::custom(format!(
                        "invalid model rotation x={}, y={} (must be a multiple of 90)",
                        v.x, v.y
                    ))
                })?;
                // java: ModelBlockDefinition.Variant.Deserializer.makeModelLocation —
                // "model": "stone" -> "minecraft:block/stone".
                let loc = ResourceLocation::parse(&v.model)
                    .map_err(|e| D::Error::custom(format!("bad model location: {e}")))?;
                let model = loc.with_prefix("block/");
                // `ModelBlockDefinition.Variant.Deserializer` reads this through
                // `JsonUtils.getInt(..., 1)` with no lower-bound clamp. Its Gson-backed signed
                // `int` conversion also narrows overflowed JSON integers modulo 2^32. In
                // particular, zero and negative weights are observable in the later
                // `WeightedBakedModel` selection path; do not repair them here.
                out.push(Variant {
                    model,
                    rotation,
                    uvlock: v.uvlock,
                    weight: v.weight,
                });
            }
            variants.insert(key, out);
        }
        Ok(BlockStateJson { variants })
    }
}

/// Gson's `JsonPrimitive.getAsString`: strings are returned verbatim, while numbers and booleans
/// use their JSON spelling. Objects, arrays, and null are not primitives and remain errors.
fn deserialize_gson_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::String(value) => Ok(value),
        serde_json::Value::Number(value) => Ok(value.to_string()),
        serde_json::Value::Bool(value) => Ok(value.to_string()),
        value => Err(D::Error::custom(format!(
            "expected a JSON primitive accepted by Gson getAsString, got {value}"
        ))),
    }
}

/// Gson's `JsonPrimitive.getAsBoolean`: a real boolean is returned directly; every other
/// primitive is converted to text and compared with `"true"` ignoring ASCII case.
fn deserialize_gson_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Bool(value) => Ok(value),
        serde_json::Value::String(value) => Ok(value.eq_ignore_ascii_case("true")),
        serde_json::Value::Number(_) => Ok(false),
        value => Err(D::Error::custom(format!(
            "expected a JSON primitive accepted by Gson getAsBoolean, got {value}"
        ))),
    }
}

/// java: `JsonUtils.getInt(json, key, default)` -> `getInt(JsonElement)`, whose
/// `isJsonPrimitive() && isNumber()` gate rejects everything but a JSON number, so a variant
/// with `"x": "90"` fails its blockstate file (mc-rs accepted numeric strings before this change). The number is then Gson 2.2.4's `LazilyParsedNumber.intValue`: parse as an `int`
/// when possible, then as a `long` and narrow with Java's `l2i`, then use `BigDecimal.intValue`
/// for decimal or oversized text. (`arbitrary_precision` keeps the JSON text.)
fn deserialize_java_int<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let text = match &value {
        serde_json::Value::Number(value) => value.to_string(),
        _ => {
            return Err(D::Error::custom(format!(
                "expected a JSON number accepted by JsonUtils.getInt, got {value}"
            )))
        }
    };
    parse_java_int_text(&text).ok_or_else(|| {
        D::Error::custom(format!(
            "invalid JSON number accepted by Gson getAsInt: {text:?}"
        ))
    })
}

/// The low 32 bits of a decimal integer after truncation toward zero. This is the observable
/// contract of `BigDecimal.intValue`, including values larger than `long`.
pub(crate) fn parse_java_int_text(text: &str) -> Option<i32> {
    if let Ok(value) = text.parse::<i32>() {
        return Some(value);
    }
    if let Ok(value) = text.parse::<i64>() {
        return Some(value as i32);
    }
    if let Ok(value) = text.parse::<u64>() {
        return Some(value as i32);
    }

    let (negative, unsigned) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    if unsigned.is_empty() {
        return None;
    }
    let mut exponent_parts = unsigned.split(['e', 'E']);
    let mantissa = exponent_parts.next()?;
    let exponent = match exponent_parts.next() {
        Some(value) => value.parse::<i64>().ok()?,
        None => 0,
    };
    if exponent_parts.next().is_some() {
        return None;
    }

    let mut digits = Vec::with_capacity(mantissa.len());
    let mut decimal_at = None;
    for byte in mantissa.bytes() {
        match byte {
            b'0'..=b'9' => digits.push(byte - b'0'),
            b'.' if decimal_at.is_none() => decimal_at = Some(digits.len()),
            _ => return None,
        }
    }
    if digits.is_empty() {
        return None;
    }
    let decimal_at = decimal_at.unwrap_or(digits.len());
    let integer_len = i64::try_from(decimal_at).ok()?.checked_add(exponent)?;
    if integer_len <= 0 {
        return Some(0);
    }

    let retained = usize::try_from(integer_len)
        .ok()
        .map_or(digits.len(), |len| len.min(digits.len()));
    let mut low = 0_u32;
    for &digit in &digits[..retained] {
        low = low.wrapping_mul(10).wrapping_add(u32::from(digit));
    }
    let appended_zeros = integer_len.saturating_sub(i64::try_from(digits.len()).ok()?);
    low = low.wrapping_mul(pow10_wrapping(appended_zeros));
    Some(if negative {
        0_u32.wrapping_sub(low) as i32
    } else {
        low as i32
    })
}

fn pow10_wrapping(mut exponent: i64) -> u32 {
    let mut result = 1_u32;
    let mut base = 10_u32;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exponent >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_variant_normal() {
        let s = r#"{"variants": {"normal": {"model": "stone"}}}"#;
        let bs: BlockStateJson = serde_json::from_str(s).unwrap();
        let v = &bs.variants["normal"];
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].model.to_string(), "minecraft:block/stone");
        assert_eq!(v[0].rotation, ModelRotation::IDENTITY);
        assert!(!v[0].uvlock);
        assert_eq!(v[0].weight, 1);
    }

    #[test]
    fn stairs_rotation_and_uvlock() {
        let s = r#"{"variants": {
            "facing=east,half=bottom,shape=straight": {"model": "block/oak_stairs"},
            "facing=west,half=bottom,shape=straight": {"model": "block/oak_stairs", "y": 180, "uvlock": true}
        }}"#;
        let bs: BlockStateJson = serde_json::from_str(s).unwrap();
        let west = &bs.variants["facing=west,half=bottom,shape=straight"][0];
        assert_eq!(west.rotation.quarters_y, 2);
        assert!(west.uvlock);
    }

    #[test]
    fn weighted_multi_variant_list() {
        let s = r#"{"variants": {"normal": [
            {"model": "block/stone"},
            {"model": "block/stone", "y": 90, "weight": 2},
            {"model": "block/stone", "y": 180}
        ]}}"#;
        let bs: BlockStateJson = serde_json::from_str(s).unwrap();
        let list = &bs.variants["normal"];
        assert_eq!(list.len(), 3);
        assert_eq!(list[1].weight, 2);
        assert_eq!(list[1].rotation.quarters_y, 1);
    }

    #[test]
    fn weighted_candidates_keep_equal_models_with_distinct_weights() {
        // Java keeps both entries for selection even though Variant.equals/hashCode identify
        // them as the same model candidate.  Parsing must not deduplicate the list.
        let s = r#"{"variants": {"normal": [
            {"model": "block/stone", "weight": 1},
            {"model": "block/stone", "weight": 9}
        ]}}"#;
        let bs: BlockStateJson = serde_json::from_str(s).unwrap();
        let list = &bs.variants["normal"];
        assert_eq!(list.len(), 2);
        assert_eq!(list[0], list[1]);
        assert_eq!(list[0].weight, 1);
        assert_eq!(list[1].weight, 9);
    }

    #[test]
    fn zero_weight_is_preserved_so_weighted_selection_can_skip_it() {
        // Positive control: an omitted weight is still Java's default of one.
        // The zero is not normalised: `WeightedRandom`'s cumulative walk can never choose it
        // while the list's total is positive.
        let s = r#"{"variants": {"normal": [
            {"model": "block/stone", "weight": 0},
            {"model": "block/stone"},
            {"model": "block/stone", "weight": 7}
        ]}}"#;
        let bs: BlockStateJson = serde_json::from_str(s).unwrap();
        let list = &bs.variants["normal"];
        assert_eq!(
            list.iter()
                .map(|variant| variant.weight)
                .collect::<Vec<_>>(),
            [0, 1, 7]
        );
    }

    #[test]
    fn signed_weight_defaults_preserves_negative_and_narrows_gson_overflow() {
        // Java `JsonUtils.getInt(..., 1)` does no positive clamp. Gson 2.2.4's intValue first
        // tries int, then long and narrows with `l2i`; the last entry is the overflow control.
        let s = r#"{"variants": {"normal": [
            {"model": "block/stone"},
            {"model": "block/stone", "weight": 0},
            {"model": "block/stone", "weight": -2},
            {"model": "block/stone", "weight": 2147483648},
            {"model": "block/stone", "weight": -2147483649},
            {"model": "block/stone", "weight": 4294967296}
        ]}}"#;
        let bs: BlockStateJson = serde_json::from_str(s).unwrap();
        assert_eq!(
            bs.variants["normal"]
                .iter()
                .map(|variant| variant.weight)
                .collect::<Vec<_>>(),
            [1, 0, -2, i32::MIN, i32::MAX, 0]
        );
    }

    #[test]
    fn all_zero_weights_remain_an_invalid_zero_total() {
        // Do not silently repair the malformed total here.  The baking/selection layer owns
        // its defensive zero-total handling; parsing must faithfully preserve the definition.
        let s = r#"{"variants": {"normal": [
            {"model": "block/stone", "weight": 0},
            {"model": "block/stone", "weight": 0}
        ]}}"#;
        let bs: BlockStateJson = serde_json::from_str(s).unwrap();
        let list = &bs.variants["normal"];
        assert_eq!(
            list.iter()
                .fold(0_i32, |total, variant| total.wrapping_add(variant.weight)),
            0
        );
    }

    #[test]
    fn invalid_rotation_is_rejected() {
        let s = r#"{"variants": {"normal": {"model": "stone", "y": 45}}}"#;
        assert!(serde_json::from_str::<BlockStateJson>(s).is_err());
    }

    #[test]
    fn empty_candidate_list_is_preserved() {
        let s = r#"{"variants": {"normal": []}}"#;
        let bs: BlockStateJson = serde_json::from_str(s).unwrap();
        assert!(bs.variants["normal"].is_empty());
    }

    #[test]
    fn variant_hash_ignores_weight_like_java() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher as _;

        let parse = |weight| {
            let json = format!(
                r#"{{"variants": {{"normal": {{"model": "stone", "weight": {weight}}}}}}}"#
            );
            serde_json::from_str::<BlockStateJson>(&json)
                .unwrap()
                .variants
                .remove("normal")
                .unwrap()
                .pop()
                .unwrap()
        };
        let first = parse(1);
        let second = parse(9);
        let mut first_hash = DefaultHasher::new();
        first.hash(&mut first_hash);
        let mut second_hash = DefaultHasher::new();
        second.hash(&mut second_hash);
        assert_eq!(first, second);
        assert_eq!(first_hash.finish(), second_hash.finish());
    }

    #[test]
    fn variant_primitives_follow_gson_string_int_and_boolean_coercions() {
        // java: `JsonUtils.getInt` takes JSON numbers only (this case used
        // numeric strings, which vanilla rejects), truncating and wrapping them; `getString`
        // and `getBoolean` read any primitive.
        let s = r#"{"variants": {
            "normal": {"model": 7, "x": 90, "y": 90.5, "uvlock": "TrUe", "weight": 4294967298.9},
            "falsey": {"model": true, "uvlock": 1}
        }}"#;
        for rejected in [
            r#""x": "90""#,
            r#""y": "90""#,
            r#""weight": "2""#,
            r#""x": null"#,
        ] {
            let json = format!(r#"{{"variants": {{"normal": {{"model": "stone", {rejected}}}}}}}"#);
            assert!(
                serde_json::from_str::<BlockStateJson>(&json).is_err(),
                "JsonUtils.getInt rejects {rejected}"
            );
        }
        let bs: BlockStateJson = serde_json::from_str(s).unwrap();
        let normal = &bs.variants["normal"][0];
        assert_eq!(normal.model.to_string(), "minecraft:block/7");
        assert_eq!(normal.rotation.quarters_x, 1);
        assert_eq!(normal.rotation.quarters_y, 1);
        assert!(normal.uvlock);
        assert_eq!(normal.weight, 2);

        let falsey = &bs.variants["falsey"][0];
        assert_eq!(falsey.model.to_string(), "minecraft:block/true");
        assert!(!falsey.uvlock, "Boolean.parseBoolean(\"1\") is false");
    }

    #[test]
    fn variants_member_is_required_but_an_empty_object_is_valid() {
        assert!(serde_json::from_str::<BlockStateJson>(r#"{}"#).is_err());
        let empty: BlockStateJson = serde_json::from_str(r#"{"variants": {}}"#).unwrap();
        assert!(empty.variants.is_empty());
    }

    #[test]
    fn java_int_text_uses_big_decimal_low_bits_after_truncation() {
        assert_eq!(parse_java_int_text("18446744073709551617.9"), Some(1));
        assert_eq!(parse_java_int_text("-4294967297.9"), Some(-1));
        assert_eq!(parse_java_int_text("4.2949672989e9"), Some(2));
        assert_eq!(parse_java_int_text("0.999"), Some(0));
    }
}
