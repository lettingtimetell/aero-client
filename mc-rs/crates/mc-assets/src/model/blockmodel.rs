//! Model JSON: `assets/minecraft/models/**/*.json`.
//!
// java: ModelBlock.java, BlockPart.java, BlockPartFace.java, BlockFaceUV.java,
// BlockPartRotation.java
//!
//! Handles `parent` inheritance and `#texture` variable indirection (hazard **H12**) — the
//! parent chain is walked and flattened by [`ResolvedModel::resolve`] before any quad is baked.

use std::collections::HashMap;

use mc_math::facing::Axis;
use mc_math::Facing;
use serde::{de, Deserialize, Deserializer};

use crate::{AssetError, ResourceLocation, ResourceManager};

/// Raw deserialised form of one `models/**/*.json` file, before parent resolution.
// java: ModelBlock
#[derive(Debug, Clone)]
pub struct RawModel {
    /// `"parent": "block/cube_all"`, resolved against the model namespace.
    pub parent: Option<String>,
    /// `#name -> value` texture variable table. A value starting with `#` is itself a
    /// variable reference, resolved through the parent chain.
    pub textures: HashMap<String, String>,
    /// Ambient-occlusion flag consulted only when this model is the root of a parent chain.
    /// Descendants defer to that root exactly as Java's `isAmbientOcclusion` does.
    pub ambient_occlusion: bool,
    /// The element list. Empty when this model only exists to set textures on a parent.
    /// Java selects the parent branch whenever this list is empty, so an explicit
    /// `elements: []` is accepted alongside `parent` but is not a valid standalone model.
    pub elements: Vec<RawElement>,
    /// `"display"`: per-context rotation/translation/scale. java: `ItemCameraTransforms`.
    pub display: Option<super::transforms::RawDisplay>,
}

#[derive(Deserialize)]
struct RawModelWire {
    #[serde(default)]
    parent: JsonField,
    #[serde(default)]
    textures: HashMap<String, serde_json::Value>,
    #[serde(default, rename = "ambientocclusion")]
    ambient_occlusion: JsonField,
    #[serde(default)]
    elements: JsonField,
    #[serde(default)]
    display: JsonField,
}

const MODEL_JSON_PATH_PREFIX: &str = "[[mc-assets-json-path:";
const MODEL_JSON_PATH_SUFFIX: &str = "]];";

fn model_json_path_error(path: impl std::fmt::Display, error: impl std::fmt::Display) -> String {
    format!("{MODEL_JSON_PATH_PREFIX}{path}{MODEL_JSON_PATH_SUFFIX} {error}")
}

/// A JSON member that may be omitted. Gson keeps a present `null` as a member (`has` is true),
/// so `Present(Null)` is what `JsonUtils`' typed getters see and reject.
#[derive(Default)]
pub(super) enum JsonField {
    #[default]
    Missing,
    Present(serde_json::Value),
}

impl<'de> Deserialize<'de> for JsonField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        serde_json::Value::deserialize(deserializer).map(Self::Present)
    }
}

fn gson_string<E: de::Error>(value: serde_json::Value, field: &str) -> Result<String, E> {
    match value {
        serde_json::Value::String(value) => Ok(value),
        serde_json::Value::Bool(value) => Ok(value.to_string()),
        serde_json::Value::Number(value) => Ok(value.to_string()),
        _ => Err(E::custom(format!("{field} must be a JSON primitive"))),
    }
}

/// java: `JsonUtils.getFloat(JsonElement, name)`. Only a JSON **number** passes its
/// `isJsonPrimitive() && isNumber()` gate: a numeric string such as `"8"` throws
/// `JsonSyntaxException` and the whole model fails to load (PA: mc-rs used to
/// parse such strings). The value is then `LazilyParsedNumber.floatValue()`, i.e.
/// `Float.parseFloat` of the JSON text; the workspace's `arbitrary_precision` keeps that text, so
/// it is rounded to `f32` once, as Java does, never through `f64` first.
fn gson_f32<E: de::Error>(value: serde_json::Value, field: &str) -> Result<f32, E> {
    match value {
        serde_json::Value::Number(value) => value
            .to_string()
            .parse::<f32>()
            .map_err(|_| E::custom(format!("{field} must be a number"))),
        _ => Err(E::custom(format!("{field} must be a number"))),
    }
}

/// java: `JsonUtils.getInt(JsonElement, name)`: a JSON number only (`isNumber()`), then Gson
/// 2.2.4's `LazilyParsedNumber.intValue()`, which truncates decimals and keeps the low 32 bits
/// (`90.0` -> 90, `1.9` -> 1, `1e2` -> 100, `4294967296` -> 0). A numeric string throws.
fn gson_i32<E: de::Error>(value: serde_json::Value, field: &str) -> Result<i32, E> {
    match value {
        serde_json::Value::Number(value) => {
            let text = value.to_string();
            super::blockstate::parse_java_int_text(&text)
                .ok_or_else(|| E::custom(format!("{field} must be a number, was {text}")))
        }
        _ => Err(E::custom(format!("{field} must be a number"))),
    }
}

fn gson_bool<E: de::Error>(value: serde_json::Value, field: &str) -> Result<bool, E> {
    match value {
        serde_json::Value::Bool(value) => Ok(value),
        serde_json::Value::String(value) => Ok(value.eq_ignore_ascii_case("true")),
        serde_json::Value::Number(_) => Ok(false),
        _ => Err(E::custom(format!("{field} must be a JSON primitive"))),
    }
}

/// java: `JsonUtils.getJsonArray` then `getFloat` per element: an array of exactly `N` JSON
/// numbers. A present `null` is not an array and is rejected.
pub(super) fn gson_float_array<E: de::Error, const N: usize>(
    value: serde_json::Value,
    field: &str,
) -> Result<[f32; N], E> {
    let serde_json::Value::Array(values) = value else {
        return Err(E::custom(format!("{field} must be an array")));
    };
    if values.len() != N {
        return Err(E::custom(format!(
            "{field} must contain exactly {N} values"
        )));
    }
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| gson_f32::<E>(value, &format!("{field}[{index}]")))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| E::custom(format!("{field} must contain exactly {N} values")))
}

impl<'de> Deserialize<'de> for RawModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RawModelWire::deserialize(deserializer)?;
        let parent = match wire.parent {
            JsonField::Missing => None,
            JsonField::Present(value) => {
                let value = gson_string::<D::Error>(value, "model parent")
                    .map_err(|error| de::Error::custom(model_json_path_error("parent", error)))?;
                (!value.is_empty()).then_some(value)
            }
        };
        let elements = match wire.elements {
            JsonField::Missing => None,
            JsonField::Present(serde_json::Value::Array(values)) => Some(
                values
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| {
                        require_object::<D::Error>(value, "model element")
                            .and_then(|value| {
                                serde_json::from_value::<RawElement>(value)
                                    .map_err(<D::Error as de::Error>::custom)
                            })
                            .map_err(|error| {
                                de::Error::custom(model_json_path_error(
                                    format!("elements[{index}]"),
                                    error,
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            JsonField::Present(_value) => {
                return Err(de::Error::custom(model_json_path_error(
                    "elements",
                    "model elements must be an array",
                )));
            }
        };
        // Java's deserializer branches on the parsed list's emptiness, rather than on whether
        // the JSON member was present.  Consequently `{parent, elements: []}` is the parent
        // form, while `{elements: []}` has neither a usable parent nor any geometry.
        let elements_empty = elements.as_ref().is_none_or(Vec::is_empty);
        match (parent.as_ref(), elements_empty) {
            (Some(_), true) | (None, false) => {}
            (Some(_), false) => {
                return Err(de::Error::custom(
                    "model must not declare both parent and elements",
                ));
            }
            (None, true) => {
                return Err(de::Error::custom(
                    "model must declare either parent or elements",
                ));
            }
        }
        let textures = wire
            .textures
            .into_iter()
            .map(|(name, value)| {
                gson_element_as_string::<D::Error>(value, "model texture")
                    .map(|value| (name.clone(), value))
                    .map_err(|error| {
                        de::Error::custom(model_json_path_error(format!("textures[{name}]"), error))
                    })
            })
            .collect::<Result<_, _>>()?;
        let ambient_occlusion = match wire.ambient_occlusion {
            JsonField::Missing => true,
            JsonField::Present(serde_json::Value::Bool(value)) => value,
            JsonField::Present(value) => gson_string::<D::Error>(value, "ambientocclusion")
                .map_err(|error| {
                    de::Error::custom(model_json_path_error("ambientocclusion", error))
                })?
                .eq_ignore_ascii_case("true"),
        };
        // java: `if (jsonobject.has("display")) JsonUtils.getJsonObject(jsonobject, "display")`:
        // a present `display` must be an object, so `"display": null` fails the model (mc-rs
        // used to read it as absent before this change).
        let display = match wire.display {
            JsonField::Missing => None,
            JsonField::Present(value @ serde_json::Value::Object(_)) => Some(
                serde_json::from_value::<super::transforms::RawDisplay>(value)
                    .map_err(|error| de::Error::custom(model_json_path_error("display", error)))?,
            ),
            JsonField::Present(_) => {
                return Err(de::Error::custom(model_json_path_error(
                    "display",
                    "model display must be a JSON object",
                )));
            }
        };
        Ok(Self {
            parent,
            textures,
            ambient_occlusion,
            elements: elements.unwrap_or_default(),
            display,
        })
    }
}

/// java: `JsonElement.getAsString()`, which `ModelBlock.Deserializer.getTextures` calls on each
/// `textures` value (not `JsonUtils.getString`): a primitive gives its text and a `JsonArray`
/// of exactly one element gives that element's, Gson's single-element-array convenience.
fn gson_element_as_string<E: de::Error>(
    value: serde_json::Value,
    field: &str,
) -> Result<String, E> {
    match value {
        serde_json::Value::Array(mut values) if values.len() == 1 => {
            gson_element_as_string(values.remove(0), field)
        }
        value => gson_string(value, field),
    }
}

/// java: every model part deserializer starts with `json.getAsJsonObject()`, which throws for
/// anything but an object. (serde's derived structs would also take a positional array.)
fn require_object<E: de::Error>(
    value: serde_json::Value,
    field: &str,
) -> Result<serde_json::Value, E> {
    if value.is_object() {
        Ok(value)
    } else {
        Err(E::custom(format!("{field} must be a JSON object")))
    }
}

/// One `elements[]` entry: a from/to box plus per-face UV/texture/cullface.
// java: BlockPart
#[derive(Debug, Clone)]
pub struct RawElement {
    /// Minimum corner, model-space `0..16` (may extend to `-16..32` for oversized parts).
    pub from: [f32; 3],
    /// Maximum corner.
    pub to: [f32; 3],
    /// Optional origin-anchored rotation about one axis.
    pub rotation: Option<RawRotation>,
    /// Per-face data, keyed by face name (`down`,`up`,`north`,`south`,`west`,`east`).
    pub faces: HashMap<String, RawFace>,
    /// Whether this element receives directional shading. `true` unless stated otherwise.
    pub shade: bool,
}

#[derive(Deserialize)]
struct RawElementWire {
    from: serde_json::Value,
    to: serde_json::Value,
    #[serde(default)]
    rotation: JsonField,
    faces: HashMap<String, serde_json::Value>,
    #[serde(default)]
    shade: JsonField,
}

impl<'de> Deserialize<'de> for RawElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RawElementWire::deserialize(deserializer)?;
        let from = gson_float_array::<D::Error, 3>(wire.from, "element from")?;
        let to = gson_float_array::<D::Error, 3>(wire.to, "element to")?;
        if from
            .into_iter()
            .chain(to)
            .any(|coordinate| !(-16.0..=32.0).contains(&coordinate))
        {
            return Err(de::Error::custom(
                "element from/to coordinates must be between -16 and 32",
            ));
        }
        if wire.faces.is_empty() {
            return Err(de::Error::custom("element must declare at least one face"));
        }
        if let Some(face) = wire.faces.keys().find(|face| parse_facing(face).is_none()) {
            return Err(de::Error::custom(format!("unknown element face {face:?}")));
        }
        let faces = wire
            .faces
            .into_iter()
            .map(|(name, value)| {
                require_object::<D::Error>(value, "element face")
                    .and_then(|value| {
                        serde_json::from_value::<RawFace>(value)
                            .map_err(<D::Error as de::Error>::custom)
                    })
                    .map(|face| (name.clone(), face))
                    .map_err(|error| {
                        de::Error::custom(model_json_path_error(format!("faces[{name}]"), error))
                    })
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        // java: `if (has("rotation")) JsonUtils.getJsonObject(jsonobject, "rotation")`: a present
        // rotation must be an object, so `"rotation": null` fails the model (mc-rs read it as
        // absent before this change).
        let rotation = match wire.rotation {
            JsonField::Missing => None,
            JsonField::Present(value) => Some(
                require_object::<D::Error>(value, "element rotation")
                    .and_then(|value| {
                        serde_json::from_value::<RawRotation>(value)
                            .map_err(<D::Error as de::Error>::custom)
                    })
                    .map_err(|error| de::Error::custom(model_json_path_error("rotation", error)))?,
            ),
        };
        // java: `if (has("shade") && !JsonUtils.isBoolean(jsonobject, "shade")) throw`: only a
        // real JSON boolean passes, so `"shade": "false"`, `0` or `null` fail the model, unlike
        // `rescale`, which `getBoolean` reads from any primitive.
        let shade = match wire.shade {
            JsonField::Missing => true,
            JsonField::Present(serde_json::Value::Bool(value)) => value,
            JsonField::Present(_) => {
                return Err(de::Error::custom("element shade must be a JSON boolean"));
            }
        };
        Ok(Self {
            from,
            to,
            rotation,
            faces,
            shade,
        })
    }
}

/// `elements[].rotation`.
// java: BlockPartRotation
#[derive(Debug, Clone)]
pub struct RawRotation {
    /// Pivot, model-space `0..16`.
    pub origin: [f32; 3],
    /// Rotation axis.
    pub axis: String,
    /// Angle in degrees; vanilla only allows -45/-22.5/0/22.5/45.
    pub angle: f32,
    /// Whether the element is rescaled to fill its original bounding box after rotation.
    pub rescale: bool,
}

#[derive(Deserialize)]
struct RawRotationWire {
    origin: serde_json::Value,
    axis: serde_json::Value,
    angle: serde_json::Value,
    #[serde(default)]
    rescale: JsonField,
}

impl<'de> Deserialize<'de> for RawRotation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RawRotationWire::deserialize(deserializer)?;
        let origin = gson_float_array::<D::Error, 3>(wire.origin, "element rotation origin")?;
        let axis = gson_string::<D::Error>(wire.axis, "element rotation axis")?;
        if parse_axis(&axis).is_none() {
            return Err(de::Error::custom(format!(
                "unknown element rotation axis {axis:?}"
            )));
        }
        let angle = gson_f32::<D::Error>(wire.angle, "element rotation angle")?;
        if ![-45.0, -22.5, 0.0, 22.5, 45.0].contains(&angle) {
            return Err(de::Error::custom(format!(
                "invalid element rotation angle {angle}"
            )));
        }
        let rescale = match wire.rescale {
            JsonField::Missing => false,
            JsonField::Present(value) => gson_bool::<D::Error>(value, "element rotation rescale")?,
        };
        Ok(Self {
            origin,
            axis,
            angle,
            rescale,
        })
    }
}

/// `elements[].faces.<name>`.
// java: BlockPartFace, BlockFaceUV
#[derive(Debug, Clone)]
pub struct RawFace {
    /// Explicit UV rect `[u0,v0,u1,v1]` in `0..16` texel space. Defaults to the element's
    /// projected face extents when absent (java: `BlockPart.setDefaultUvs`).
    pub uv: Option<[f32; 4]>,
    /// `#var` texture reference, resolved against the model's texture table.
    pub texture: String,
    /// A face is culled (not baked into `general_quads`, instead keyed under
    /// [`Facing`] so the mesher can skip it against an opaque neighbour) when this is set.
    pub cullface: Option<String>,
    /// UV quarter-turns, `0/90/180/270`.
    pub rotation: i32,
    /// Biome-tint layer index, or `-1` for none.
    pub tintindex: i32,
}

#[derive(Deserialize)]
struct RawFaceWire {
    #[serde(default)]
    uv: JsonField,
    texture: serde_json::Value,
    #[serde(default)]
    cullface: JsonField,
    #[serde(default)]
    rotation: JsonField,
    #[serde(default)]
    tintindex: JsonField,
}

impl<'de> Deserialize<'de> for RawFace {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RawFaceWire::deserialize(deserializer)?;
        let uv = match wire.uv {
            // Java's BlockFaceUV.Deserializer distinguishes an omitted member from a present
            // JSON null: `!jsonobject.has("uv")` returns the default/null UV, while a present
            // null is passed to JsonUtils.getJsonArray and rejected as a malformed array.
            JsonField::Missing => None,
            JsonField::Present(serde_json::Value::Null) => {
                return Err(de::Error::custom("face uv must be an array"));
            }
            JsonField::Present(value) => Some(gson_float_array::<D::Error, 4>(value, "face uv")?),
        };
        let texture = gson_string::<D::Error>(wire.texture, "face texture")?;
        // java: `BlockPartFace.Deserializer.parseCullFace` is
        // `EnumFacing.byName(JsonUtils.getString(json, "cullface", ""))`. A present JSON null
        // reaches `getString(JsonElement)` and throws (the model fails to load), while "" or any
        // unknown name gives `byName`'s null: an ordinary general quad that is never culled.
        // The bake keeps an unparseable name as exactly that (`bake.rs`, `general_quads`).
        let cullface = match wire.cullface {
            JsonField::Missing => None,
            JsonField::Present(value) => {
                let value = gson_string::<D::Error>(value, "face cullface")?;
                (!value.is_empty()).then_some(value)
            }
        };
        let rotation = match wire.rotation {
            JsonField::Missing => 0,
            JsonField::Present(value) => gson_i32::<D::Error>(value, "face rotation")?,
        };
        if ![0, 90, 180, 270].contains(&rotation) {
            return Err(de::Error::custom(format!(
                "invalid face rotation {rotation}"
            )));
        }
        let tintindex = match wire.tintindex {
            JsonField::Missing => -1,
            JsonField::Present(value) => gson_i32::<D::Error>(value, "face tintindex")?,
        };
        Ok(Self {
            uv,
            texture,
            cullface,
            rotation,
            tintindex,
        })
    }
}

/// A model after its `parent` chain has been walked and flattened.
///
/// Mirrors `ModelBlock.getElements()`/`isAmbientOcclusion()`/`resolveTextureName()`, which in
/// the Java all defer to the root through `parent` at call time; here the chain is resolved
/// once up front so baking is a flat, allocation-free walk.
#[derive(Debug, Clone, Default)]
pub struct ResolvedModel {
    /// Elements to bake. Taken from the nearest ancestor (including self) that declares any —
    /// java never merges element lists across a `parent` boundary.
    pub elements: Vec<RawElement>,
    /// Fully resolved textures for this model: every `#var` chased to a concrent
    /// (non-`#`-prefixed) value, or `"missingno"` if the chain never terminates (java:
    /// `resolveTextureName`).
    pub textures: HashMap<String, String>,
    // Kept leaf-to-root solely for `ModelBlock.resolveTextureName`'s unusual Bookkeep
    // recursion. `textures` above is the convenient resolved view used by generated items;
    // it cannot retain the model which supplied an alias, and therefore cannot reproduce the
    // Java same-model upward-reference guard on its own.
    pub(crate) texture_layers: Vec<HashMap<String, String>>,
    /// From the root of the model's parent chain (or this model when it has no real parent).
    pub ambient_occlusion: bool,
    /// Set when the chain terminates in a `builtin/*` sentinel parent instead of a real file
    /// — `Some("generated")`, `Some("compass")`, or `Some("clock")` for the flat-item-icon
    /// expansion ([`super::item::expand_builtin_generated`]), and `Some("entity")` for items
    /// delegated to vanilla's tile-entity item renderer.
    pub builtin: Option<String>,
    /// The `display` block, with each of the six contexts resolved through the parent chain
    /// **independently** (java: `ModelBlock.getAllTransforms`/`getTransform`).
    pub transforms: super::transforms::ItemCameraTransforms,
}

/// Errors resolving a model's parent chain.
#[derive(Debug, thiserror::Error)]
pub enum ModelResolveError {
    /// The chain loops back on itself (java: `ModelBlock.LoopException`).
    #[error("model parent chain loops back on itself at {0}")]
    Loop(ResourceLocation),
    /// Exceeded a sane depth without terminating; almost certainly a bug rather than a real
    /// vanilla asset (real chains are at most 2-3 deep).
    #[error("model parent chain exceeds {0} levels, starting at {1}")]
    TooDeep(usize, ResourceLocation),
    /// Loading or parsing one of the parent files failed.
    #[error(transparent)]
    Asset(#[from] AssetError),
}

impl From<ModelResolveError> for AssetError {
    fn from(e: ModelResolveError) -> Self {
        match e {
            ModelResolveError::Asset(inner) => inner,
            ModelResolveError::Loop(loc) => AssetError::ModelResolve {
                location: loc,
                reason: "parent chain loops back on itself".to_string(),
            },
            ModelResolveError::TooDeep(depth, loc) => AssetError::ModelResolve {
                location: loc,
                reason: format!("parent chain exceeds {depth} levels"),
            },
        }
    }
}

const MAX_PARENT_DEPTH: usize = 32;

/// Normalize the lexical extensions accepted by Gson's lenient `JsonReader` for model files.
/// This is deliberately model-local: language, metadata, and asset-index JSON keep using the
/// strict generic [`ResourceManager::load_json`] path.
fn normalize_lenient_model_json(bytes: &[u8]) -> String {
    let input = String::from_utf8_lossy(bytes);
    let chars = input.chars().collect::<Vec<_>>();
    let mut normalized = String::with_capacity(chars.len());
    let mut index = 0;

    while index < chars.len() {
        match chars[index] {
            '"' => {
                normalized.push('"');
                index += 1;
                while index < chars.len() {
                    let character = chars[index];
                    normalized.push(character);
                    index += 1;
                    if character == '\\' && index < chars.len() {
                        normalized.push(chars[index]);
                        index += 1;
                    } else if character == '"' {
                        break;
                    }
                }
            }
            '\'' => {
                normalized.push('"');
                index += 1;
                while index < chars.len() {
                    let character = chars[index];
                    index += 1;
                    match character {
                        '\'' => {
                            normalized.push('"');
                            break;
                        }
                        '"' => normalized.push_str("\\\""),
                        '\\' if index < chars.len() => {
                            let escaped = chars[index];
                            index += 1;
                            if escaped == '\'' {
                                normalized.push('\'');
                            } else {
                                normalized.push('\\');
                                normalized.push(escaped);
                            }
                        }
                        _ => normalized.push(character),
                    }
                }
            }
            '/' if chars.get(index + 1) == Some(&'/') => {
                index += 2;
                while index < chars.len() && chars[index] != '\n' && chars[index] != '\r' {
                    index += 1;
                }
                normalized.push(' ');
            }
            '/' if chars.get(index + 1) == Some(&'*') => {
                index += 2;
                while index + 1 < chars.len() && !(chars[index] == '*' && chars[index + 1] == '/') {
                    index += 1;
                }
                if index + 1 < chars.len() {
                    index += 2;
                    normalized.push(' ');
                } else {
                    // Do not turn an unterminated comment after an otherwise-complete object
                    // into valid JSON merely by deleting it.
                    normalized.push('\0');
                    index = chars.len();
                }
            }
            '#' => {
                index += 1;
                while index < chars.len() && chars[index] != '\n' && chars[index] != '\r' {
                    index += 1;
                }
                normalized.push(' ');
            }
            character
                if character.is_whitespace()
                    || matches!(character, '{' | '}' | '[' | ']' | ':' | ',') =>
            {
                normalized.push(character);
                index += 1;
            }
            _ => {
                let start = index;
                while index < chars.len()
                    && !chars[index].is_whitespace()
                    && !matches!(chars[index], '{' | '}' | '[' | ']' | ':' | ',' | '\'' | '"')
                    && !(chars[index] == '/' && matches!(chars.get(index + 1), Some('/' | '*')))
                {
                    index += 1;
                }
                let token = chars[start..index].iter().collect::<String>();
                let strict_primitive = serde_json::from_str::<serde_json::Value>(&token)
                    .is_ok_and(|value| value.is_null() || value.is_boolean() || value.is_number());
                if strict_primitive {
                    normalized.push_str(&token);
                } else {
                    normalized.push_str(
                        &serde_json::to_string(&token).expect("a Rust string always encodes"),
                    );
                }
            }
        }
    }

    // Gson accepts a separator immediately before an object/array close. Strip only commas
    // outside strings so quoted punctuation and comment markers remain byte-for-byte content.
    let chars = normalized.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(normalized.len());
    let mut index = 0;
    let mut in_string = false;
    while index < chars.len() {
        let character = chars[index];
        if character == '"' {
            in_string = !in_string;
            out.push(character);
            index += 1;
            continue;
        }
        if in_string && character == '\\' {
            out.push(character);
            index += 1;
            if index < chars.len() {
                out.push(chars[index]);
                index += 1;
            }
            continue;
        }
        if !in_string && character == ',' {
            let mut next = index + 1;
            while next < chars.len() && chars[next].is_whitespace() {
                next += 1;
            }
            if matches!(chars.get(next), Some(']' | '}')) {
                index += 1;
                continue;
            }
        }
        out.push(character);
        index += 1;
    }
    out
}

fn load_raw_model(
    rm: &ResourceManager,
    location: &ResourceLocation,
) -> Result<RawModel, AssetError> {
    let bytes = rm.load(location)?;
    let normalized = normalize_lenient_model_json(&bytes);
    serde_json::from_str(&normalized).map_err(|error| {
        let (path, message) = model_json_error_context(&error);
        if path.is_empty() {
            AssetError::json_context(location.to_string(), &path, error)
        } else {
            AssetError::json_context_prepend_message(
                location.to_string(),
                path.iter().rev().cloned(),
                message,
                error,
            )
        }
    })
}

/// Pulls the path marker added while RawModel recursively deserialises nested members.
///
/// Serde owns the concrete error value, so the marker is carried through its custom error
/// channel. It is removed before display while the original serde error remains the source of
/// the returned AssetError.
fn model_json_error_context(error: &serde_json::Error) -> (Vec<String>, String) {
    let rendered = error.to_string();
    let Some(_) = rendered.find(MODEL_JSON_PATH_PREFIX) else {
        return (Vec::new(), rendered);
    };
    let mut path = Vec::new();
    let mut cursor = 0;
    let mut message_start = rendered.len();
    while let Some(relative_start) = rendered[cursor..].find(MODEL_JSON_PATH_PREFIX) {
        let start = cursor + relative_start;
        let path_start = start + MODEL_JSON_PATH_PREFIX.len();
        let Some(relative_end) = rendered[path_start..].find(MODEL_JSON_PATH_SUFFIX) else {
            return (Vec::new(), rendered);
        };
        let path_end = path_start + relative_end;
        path.extend(
            rendered[path_start..path_end]
                .split("->")
                .map(str::to_owned),
        );
        message_start = path_end + MODEL_JSON_PATH_SUFFIX.len();
        cursor = message_start;
    }
    let message = rendered[message_start..]
        .trim()
        .trim_start_matches(':')
        .trim()
        .to_owned();
    (path, message)
}

impl ResolvedModel {
    /// Loads `loc` (a `models/**/*.json` location, without the `.json` suffix or `models/`
    /// prefix — callers pass e.g. `minecraft:block/oak_stairs`) and walks its `parent` chain
    /// to completion.
    pub fn resolve(
        rm: &ResourceManager,
        loc: &ResourceLocation,
    ) -> Result<Self, ModelResolveError> {
        let mut chain = Vec::with_capacity(4);
        let mut cur = loc.clone();
        let mut seen = std::collections::HashSet::new();
        let mut builtin: Option<String> = None;
        loop {
            if !seen.insert(cur.clone()) {
                return Err(ModelResolveError::Loop(loc.clone()));
            }
            if chain.len() >= MAX_PARENT_DEPTH {
                return Err(ModelResolveError::TooDeep(MAX_PARENT_DEPTH, loc.clone()));
            }
            let model_loc = model_json_location(&cur);
            let raw = load_raw_model(rm, &model_loc)?;
            let parent = raw.parent.clone();
            chain.push(raw);
            match parent {
                // `builtin/*` is a sentinel vanilla's `ModelBakery` special-cases rather than
                // a real file (`models/builtin/generated.json` does not exist in the jar) —
                // `item/generated.json`'s declared parent is `"builtin/generated"`, meaning
                // "run `ItemModelGenerator` instead of loading elements from a parent".
                Some(p) if p.starts_with("builtin/") => {
                    builtin = Some(p.trim_start_matches("builtin/").to_string());
                    break;
                }
                Some(p) => cur = ResourceLocation::parse(&p)?,
                None => break,
            }
        }

        // Walk root-to-leaf so a child's textures override its ancestors', but elements
        // come from the first (leaf-most) entry that has any — matching
        // `ModelBlock.getElements()`'s "mine, else my parent's" recursion.
        let mut textures: HashMap<String, String> = HashMap::new();
        // java: ModelBlock.isAmbientOcclusion(): a child delegates all the way to its
        // root parent. This is deliberately unlike textures and elements: a leaf's own
        // `ambientocclusion` JSON member is ignored whenever it has a real parent.
        // `chain` is leaf-to-root, so its final entry is exactly that root.
        let ambient_occlusion = chain.last().is_none_or(|root| root.ambient_occlusion);
        let mut elements: Vec<RawElement> = Vec::new();
        let mut elements_set = false;
        for raw in chain.iter().rev() {
            for (k, v) in &raw.textures {
                textures.insert(k.clone(), v.clone());
            }
            if !elements_set && !raw.elements.is_empty() {
                elements = raw.elements.clone();
                elements_set = true;
            }
        }

        // java: `ModelBlock.getTransform` recurses to the parent per *context*, not per model,
        // so walk leaf-to-root letting each level fill only what the levels below it left at
        // `ItemTransformVec3f.DEFAULT`.
        let mut transforms = super::transforms::ItemCameraTransforms::DEFAULT;
        for raw in &chain {
            let own = raw
                .display
                .as_ref()
                .map_or(super::transforms::ItemCameraTransforms::DEFAULT, |d| {
                    d.bake()
                });
            transforms = transforms.inherit(own);
        }

        // Keep the layers as Java sees them (leaf first).  The flattened table remains useful
        // to generated-item baking, but `resolve_texture` must retain the layer identity: a
        // same-model `#a -> #b` hop is an upward reference and resolves to missingno.
        let texture_layers: Vec<_> = chain.iter().map(|raw| raw.textures.clone()).collect();
        let mut model = ResolvedModel {
            elements,
            textures: HashMap::new(),
            texture_layers,
            ambient_occlusion,
            builtin,
            transforms,
        };
        let keys: Vec<_> = textures.keys().cloned().collect();
        for key in keys {
            model
                .textures
                .insert(key.clone(), model.resolve_texture(&key).to_owned());
        }

        Ok(model)
    }

    /// Resolves a texture reference using Java `ModelBlock.resolveTextureName` semantics.
    ///
    /// Java prepends `#` even to a bare literal. Thus an unbound `blocks/stone` is *not* a
    /// literal texture location: it is an unresolved texture variable and becomes `missingno`.
    /// The bookkeep guard also rejects a second alias lookup through the same model.
    #[must_use]
    pub fn resolve_texture<'a>(&'a self, reference: &'a str) -> &'a str {
        let name = reference.strip_prefix('#').unwrap_or(reference);
        self.resolve_texture_from_layer(0, name, None)
    }

    fn resolve_texture_from_layer<'a>(
        &'a self,
        layer: usize,
        name: &str,
        model_ext: Option<usize>,
    ) -> &'a str {
        if model_ext == Some(layer) {
            return "missingno";
        }
        let Some(layers) = self.texture_layers.get(layer..) else {
            // Hand-constructed test models have no parent-layer identity. Their one table is
            // still sufficient for a direct Java texture-variable lookup.
            return self
                .textures
                .get(name)
                .filter(|value| !value.starts_with('#'))
                .map_or("missingno", String::as_str);
        };
        if layers.is_empty() {
            return self
                .textures
                .get(name)
                .filter(|value| !value.starts_with('#'))
                .map_or("missingno", String::as_str);
        }
        // java: every model on the way up checks `this ==
        // bookkeep.modelExt` before its own textures, so a lookup that reaches the model holding
        // the alias being followed is an upward reference, wherever the walk meets it (not only
        // at the start). Each followed alias must then be found strictly below the previous
        // one, which bounds the recursion; checking the first layer alone let two parents that
        // alias each other's variables recurse until the stack overflowed.
        let mut found = None;
        for (offset, textures) in layers.iter().enumerate() {
            let at = layer + offset;
            if model_ext == Some(at) {
                return "missingno";
            }
            if let Some(value) = textures.get(name) {
                found = Some((at, value));
                break;
            }
        }
        let Some((owner, value)) = found else {
            return "missingno";
        };
        if let Some(next) = value.strip_prefix('#') {
            // java: `bookkeep.model.resolveTextureName(s, bookkeep)`, where model is the leaf.
            return self.resolve_texture_from_layer(0, next, Some(owner));
        }
        value
    }
}

/// `models/**/<path>.json`, namespaced.
fn model_json_location(loc: &ResourceLocation) -> ResourceLocation {
    ResourceLocation::new(loc.namespace.clone(), format!("models/{}.json", loc.path))
}

/// Parses a face name (`"down"`, `"up"`, ...) into a [`Facing`].
pub fn parse_facing(name: &str) -> Option<Facing> {
    // `EnumFacing.byName` normalizes the serialized name before lookup.  Model JSON uses
    // this path for both `faces` keys and `cullface`, so keep it tied to the exact helper
    // instead of maintaining a second, accidentally case-sensitive table here.
    Facing::by_name(name).ok()
}

/// Parses a rotation axis name (`"x"`, `"y"`, `"z"`).
pub fn parse_axis(name: &str) -> Option<Axis> {
    // Same Java `EnumFacing.Axis.byName` semantics as face/cullface parsing above.
    Axis::by_name(name).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read};

    struct ModelPack {
        files: HashMap<String, Vec<u8>>,
    }

    impl ModelPack {
        fn for_ao_chain(root: &str, middle: &str, leaf: &str) -> Self {
            Self {
                files: HashMap::from([
                    (
                        "minecraft:models/block/root.json".to_owned(),
                        root.as_bytes().to_vec(),
                    ),
                    (
                        "minecraft:models/block/middle.json".to_owned(),
                        middle.as_bytes().to_vec(),
                    ),
                    (
                        "minecraft:models/block/leaf.json".to_owned(),
                        leaf.as_bytes().to_vec(),
                    ),
                ]),
            }
        }
    }

    impl crate::ResourcePack for ModelPack {
        fn open(&self, loc: &ResourceLocation) -> Option<Box<dyn Read>> {
            self.files
                .get(&loc.to_string())
                .map(|bytes| Box::new(Cursor::new(bytes.clone())) as Box<dyn Read>)
        }

        fn contains(&self, loc: &ResourceLocation) -> bool {
            self.files.contains_key(&loc.to_string())
        }
    }

    fn resolve_ao(root: &str, middle: &str, leaf: &str) -> bool {
        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(ModelPack::for_ao_chain(root, middle, leaf)));
        ResolvedModel::resolve(&resources, &ResourceLocation::minecraft("block/leaf"))
            .expect("resolve three-level model chain")
            .ambient_occlusion
    }

    fn resolve_model(root: &str, middle: &str, leaf: &str) -> ResolvedModel {
        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(ModelPack::for_ao_chain(root, middle, leaf)));
        ResolvedModel::resolve(&resources, &ResourceLocation::minecraft("block/leaf"))
            .expect("resolve model")
    }

    #[test]
    fn resolve_texture_chases_hash_chain() {
        let mut textures = HashMap::new();
        textures.insert("all".to_string(), "blocks/stone".to_string());
        textures.insert("particle".to_string(), "#all".to_string());
        let model = ResolvedModel {
            elements: vec![],
            textures,
            texture_layers: vec![],
            ambient_occlusion: true,
            builtin: None,
            transforms: crate::model::transforms::ItemCameraTransforms::DEFAULT,
        };
        assert_eq!(model.resolve_texture("#particle"), "missingno");
        assert_eq!(model.resolve_texture("#all"), "blocks/stone");
    }

    #[test]
    fn unresolved_reference_is_missingno() {
        let textures = HashMap::new();
        let model = ResolvedModel {
            elements: vec![],
            textures,
            texture_layers: vec![],
            ambient_occlusion: true,
            builtin: None,
            transforms: crate::model::transforms::ItemCameraTransforms::DEFAULT,
        };
        assert_eq!(model.resolve_texture("#nope"), "missingno");
    }

    #[test]
    fn bare_literal_is_an_unresolved_texture_variable() {
        let model = ResolvedModel {
            elements: vec![],
            textures: HashMap::new(),
            texture_layers: vec![],
            ambient_occlusion: true,
            builtin: None,
            transforms: crate::model::transforms::ItemCameraTransforms::DEFAULT,
        };
        assert_eq!(model.resolve_texture("blocks/stone"), "missingno");
    }

    #[test]
    fn same_model_texture_alias_hop_is_missingno() {
        let model = resolve_model(
            r#"{"parent":"builtin/generated"}"#,
            r#"{"parent":"block/root"}"#,
            r##"{"textures":{"a":"#b","b":"blocks/stone"},"parent":"block/middle"}"##,
        );
        assert_eq!(model.resolve_texture("#a"), "missingno");
    }

    #[test]
    fn parent_alias_can_resolve_a_child_texture() {
        let model = resolve_model(
            r##"{"textures":{"a":"#b"},"parent":"builtin/generated"}"##,
            r#"{"parent":"block/root"}"#,
            r#"{"parent":"block/root","textures":{"b":"blocks/child"}}"#,
        );
        assert_eq!(model.resolve_texture("#a"), "blocks/child");
    }

    #[test]
    fn parses_all_six_faces() {
        for (name, expect) in [
            ("down", Facing::Down),
            ("up", Facing::Up),
            ("north", Facing::North),
            ("south", Facing::South),
            ("west", Facing::West),
            ("east", Facing::East),
        ] {
            assert_eq!(parse_facing(name), Some(expect));
        }
        assert_eq!(parse_facing("bogus"), None);
    }

    #[test]
    fn ambient_occlusion_is_inherited_from_the_root_not_the_leaf() {
        // Java ModelBlock.isAmbientOcclusion delegates through every parent, rather than
        // reading the leaf's deserialised flag. These controls are intentionally hostile to
        // the old root-to-leaf overwrite: either changing `ambient_occlusion = ...` back
        // inside the loop makes at least the first two cases return the leaf value instead.
        let inherited_from_disabled_root = resolve_ao(
            r#"{"ambientocclusion":false,"parent":"builtin/generated"}"#,
            r#"{"parent":"block/root","ambientocclusion":true}"#,
            r#"{"parent":"block/middle","ambientocclusion":true}"#,
        );
        assert!(
            !inherited_from_disabled_root,
            "a leaf must not re-enable the root's disabled ambient occlusion"
        );

        let inherited_from_enabled_root = resolve_ao(
            r#"{"ambientocclusion":true,"parent":"builtin/generated"}"#,
            r#"{"parent":"block/root","ambientocclusion":false}"#,
            r#"{"parent":"block/middle","ambientocclusion":false}"#,
        );
        assert!(
            inherited_from_enabled_root,
            "a leaf must not disable the root's enabled ambient occlusion"
        );

        let inherited_default = resolve_ao(
            r#"{"parent":"builtin/generated"}"#,
            r#"{"parent":"block/root","ambientocclusion":false}"#,
            r#"{"parent":"block/middle","ambientocclusion":false}"#,
        );
        assert!(
            inherited_default,
            "a root that omits the member retains Java's true default despite leaf overrides"
        );
    }

    #[test]
    fn parsers_keep_java_by_name_case_folding() {
        // Mutation evidence: changing either parser back to a direct lowercase-only match
        // rejects these JSON spellings even though the live baker accepts the Java helpers'
        // case-insensitive names.
        assert_eq!(parse_facing("nOrTh"), Some(Facing::North));
        assert_eq!(parse_facing("SOUTH"), Some(Facing::South));
        assert_eq!(parse_axis("X"), Some(Axis::X));
        assert_eq!(parse_axis("z"), Some(Axis::Z));
    }

    #[test]
    fn raw_model_json_shape() {
        let s = r##"{
            "textures": {"all": "blocks/stone"},
            "elements": [{
                "from": [0,0,0], "to": [16,16,16],
                "faces": {
                    "down": {"texture": "#all", "cullface": "down"}
                }
            }]
        }"##;
        let m: RawModel = serde_json::from_str(s).unwrap();
        assert_eq!(m.parent, None);
        assert_eq!(m.elements.len(), 1);
        assert_eq!(m.elements[0].faces["down"].texture, "#all");
        assert_eq!(
            m.elements[0].faces["down"].cullface.as_deref(),
            Some("down")
        );
        assert_eq!(m.elements[0].faces["down"].tintindex, -1);
    }

    #[test]
    fn block_parts_enforce_java_bounds_angles_face_keys_and_uv_rotations() {
        // java: `JsonUtils.getFloat`/`getInt` take JSON numbers only, `getInt` truncates a
        // decimal, `shade` must be a real boolean, while `rescale` (`getBoolean`) and `texture`
        // (`getString`) read any primitive. (this case used numeric strings,
        // which vanilla rejects; `model_json_java_parity.rs` holds the full Java oracle.)
        let valid: RawModel = serde_json::from_str(
            r##"{
                "elements": [{
                    "from": [-16, 0, 0],
                    "to": [32, 16.0, 16],
                    "rotation": {
                        "origin": [8, 8, 8],
                        "axis": "X",
                        "angle": 22.5,
                        "rescale": "true"
                    },
                    "shade": false,
                    "faces": {
                        "NORTH": {
                            "uv": [0, 0, 16, 16],
                            "texture": 7,
                            "cullface": "SoUtH",
                            "rotation": 270.9,
                            "tintindex": 2.5
                        }
                    }
                }]
            }"##,
        )
        .unwrap();
        let element = &valid.elements[0];
        assert_eq!(element.from, [-16.0, 0.0, 0.0]);
        assert_eq!(element.to, [32.0, 16.0, 16.0]);
        assert!(!element.shade);
        let rotation = element.rotation.as_ref().unwrap();
        assert_eq!(rotation.axis, "X");
        assert_eq!(rotation.angle, 22.5);
        assert!(rotation.rescale);
        let face = &element.faces["NORTH"];
        assert_eq!(face.uv, Some([0.0, 0.0, 16.0, 16.0]));
        assert_eq!(face.texture, "7");
        assert_eq!(face.cullface.as_deref(), Some("SoUtH"));
        assert_eq!(face.rotation, 270);
        assert_eq!(face.tintindex, 2);

        let model = |element: &str| {
            serde_json::from_str::<RawModel>(&format!(r#"{{"elements":[{element}]}}"#))
        };
        let face = r##""faces":{"north":{"texture":"#all"}}"##;
        assert!(model(&format!(r#"{{"from":[-16.1,0,0],"to":[16,16,16],{face}}}"#)).is_err());
        assert!(model(&format!(r#"{{"from":[0,0,0],"to":[32.1,16,16],{face}}}"#)).is_err());
        assert!(model(&format!(r#"{{"from":[0,0,0],"to":[16,16,16],"rotation":{{"origin":[8,8,8],"axis":"y","angle":30}},{face}}}"#)).is_err());
        assert!(model(r##"{"from":[0,0,0],"to":[16,16,16],"faces":{"north":{"texture":"#all","rotation":45}}}"##).is_err());
        assert!(model(r##"{"from":[0,0,0],"to":[16,16,16],"faces":{"north":{"texture":"#all","uv":[0,0,16]}}}"##).is_err());
        assert!(model(
            r##"{"from":[0,0,0],"to":[16,16,16],"faces":{"sideways":{"texture":"#all"}}}"##
        )
        .is_err());
        assert!(model(r##"{"from":[0,0,0],"to":[16,16,16],"faces":{}}"##).is_err());
        // Numeric strings and non-boolean shades fail the model, as `JsonUtils` throws.
        assert!(model(&format!(r#"{{"from":["0",0,0],"to":[16,16,16],{face}}}"#)).is_err());
        assert!(model(&format!(
            r#"{{"from":[0,0,0],"to":[16,16,16],"shade":0,{face}}}"#
        ))
        .is_err());
        assert!(model(&format!(
            r#"{{"from":[0,0,0],"to":[16,16,16],"rotation":null,{face}}}"#
        ))
        .is_err());
        assert!(model(r##"{"from":[0,0,0],"to":[16,16,16],"faces":{"north":{"texture":"#all","rotation":"90"}}}"##).is_err());
        assert!(model(r##"{"from":[0,0,0],"to":[16,16,16],"faces":{"north":{"texture":"#all","tintindex":"2"}}}"##).is_err());
    }

    /// java: `BlockPartFace.Deserializer.parseCullFace` = `EnumFacing.byName(JsonUtils
    /// .getString(json, "cullface", ""))`. An unknown name (a resource pack's `"bottom"`) is a
    /// null cull face, i.e. a general quad, not a failed model; a JSON null makes `getString`
    /// throw, so that model does fail. mc-rs had the two the other way round (H7 2026-11-11).
    #[test]
    fn an_unknown_cullface_is_a_general_quad_and_a_null_one_fails_the_model() {
        let model = |cullface: &str| {
            serde_json::from_str::<RawModel>(&format!(
                r##"{{"elements":[{{"from":[0,0,0],"to":[16,16,16],
                    "faces":{{"down":{{"texture":"#all","cullface":{cullface}}}}}}}]}}"##
            ))
        };
        let bottom = model(r#""bottom""#).expect("java: byName(\"bottom\") is just null");
        let face = &bottom.elements[0].faces["down"];
        assert_eq!(face.cullface.as_deref(), Some("bottom"));
        assert_eq!(
            parse_facing("bottom"),
            None,
            "the bake files an unparseable name with the general quads"
        );
        assert!(model("null").is_err(), "java: getString(JsonNull) throws");
        let empty = model(r#""""#).expect("\"\" is the getString fallback itself");
        assert_eq!(empty.elements[0].faces["down"].cullface, None);
    }

    #[test]
    fn raw_model_requires_exactly_one_parent_or_elements_branch() {
        assert!(serde_json::from_str::<RawModel>(r#"{}"#).is_err());
        let parent_with_empty_elements: RawModel =
            serde_json::from_str(r#"{"parent":"block/base","elements":[]}"#).unwrap();
        assert_eq!(
            parent_with_empty_elements.parent.as_deref(),
            Some("block/base")
        );
        assert!(parent_with_empty_elements.elements.is_empty());
        assert!(serde_json::from_str::<RawModel>(r#"{"parent":""}"#).is_err());
        assert!(serde_json::from_str::<RawModel>(r#"{"parent":"block/base"}"#).is_ok());
        assert!(serde_json::from_str::<RawModel>(r#"{"elements":[]}"#).is_err());

        let non_empty_elements = r##"{"elements":[{"from":[0,0,0],"to":[16,16,16],"faces":{"down":{"texture":"#all"}}}]}"##;
        assert!(serde_json::from_str::<RawModel>(non_empty_elements).is_ok());
        assert!(serde_json::from_str::<RawModel>(
            r##"{"parent":"block/base","elements":[{"from":[0,0,0],"to":[16,16,16],"faces":{"down":{"texture":"#all"}}}]}"##
        )
        .is_err());
        assert!(serde_json::from_str::<RawModel>(r#"{"parent":"","elements":[]}"#).is_err());
        assert!(serde_json::from_str::<RawModel>(r#"{"parent":null}"#).is_err());
    }

    #[test]
    fn empty_elements_parent_branch_inherits_parent_geometry() {
        let model = resolve_model(
            r##"{"elements":[{"from":[0,0,0],"to":[16,16,16],"faces":{"down":{"texture":"#all"}}}]}"##,
            r#"{"parent":"block/root"}"#,
            r#"{"parent":"block/middle","elements":[]}"#,
        );
        assert_eq!(model.elements.len(), 1);
        assert_eq!(model.elements[0].faces["down"].texture, "#all");
    }

    #[test]
    fn malformed_nested_model_reports_java_ordered_file_and_member_context() {
        use std::error::Error;

        let mut resources = ResourceManager::new();
        resources.push_pack(Box::new(ModelPack {
            files: HashMap::from([(
                "minecraft:models/block/bad.json".to_owned(),
                br#"{"elements":[{"from":[0,0,0],"to":[16,16,16],"faces":{"north":{"texture":{}}}}]}"#
                    .to_vec(),
            )]),
        }));
        let error = ResolvedModel::resolve(&resources, &ResourceLocation::minecraft("block/bad"))
            .expect_err("the malformed nested face must be rejected");
        let asset = match error {
            ModelResolveError::Asset(asset) => asset,
            other => panic!("expected the parser error, got {other}"),
        };
        assert!(
            asset
                .to_string()
                .contains("Invalid minecraft:models/block/bad.json elements[0]->faces[north]: "),
            "context must unwind outer-to-inner like Java: {asset}"
        );
        assert!(
            asset.source().is_some(),
            "the serde cause must remain available"
        );
    }

    #[test]
    fn block_face_uv_distinguishes_omitted_null_and_malformed_arrays_with_asset_context() {
        fn load(face_fields: &str) -> Result<ResolvedModel, AssetError> {
            let json = [
                r#"{"elements":[{"from":[0,0,0],"to":[16,16,16],"faces":{"north":{"texture":"all""#,
                face_fields,
                "}}}]}",
            ]
            .concat();
            let mut resources = ResourceManager::new();
            resources.push_pack(Box::new(ModelPack {
                files: HashMap::from([(
                    "minecraft:models/block/uv.json".to_owned(),
                    json.into_bytes(),
                )]),
            }));
            match ResolvedModel::resolve(&resources, &ResourceLocation::minecraft("block/uv")) {
                Ok(model) => Ok(model),
                Err(ModelResolveError::Asset(error)) => Err(error),
                Err(other) => panic!("unexpected model-resolution error: {other}"),
            }
        }

        // Java's absent `uv` member is the only form that gets the element-projected default
        // UV during baking; explicit null must not be silently treated as that omission.
        let omitted = load("").expect("omitted uv is valid");
        assert_eq!(omitted.elements[0].faces["north"].uv, None);
        let explicit = load(r#","uv":[0,0,16,16]"#).expect("a four-value uv array is valid");
        assert_eq!(
            explicit.elements[0].faces["north"].uv,
            Some([0.0, 0.0, 16.0, 16.0])
        );

        for (fields, expected) in [
            (r#","uv":null"#, "face uv must be an array"),
            (r#","uv":[0,0,16]"#, "face uv must contain exactly 4 values"),
            (r#","uv":{}"#, "face uv must be an array"),
        ] {
            let error = load(fields).expect_err("malformed uv must be rejected");
            assert!(
                error
                    .to_string()
                    .contains("Invalid minecraft:models/block/uv.json elements[0]->faces[north]: "),
                "error must retain the Java-ordered model member path: {error}"
            );
            assert!(
                error.to_string().ends_with(expected),
                "error must identify the malformed uv shape: {error}"
            );
            assert!(std::error::Error::source(&error).is_some());
        }
    }

    #[test]
    fn raw_model_coerces_gson_primitive_fields() {
        let numeric_parent: RawModel = serde_json::from_str(r#"{"parent":5}"#).unwrap();
        assert_eq!(numeric_parent.parent.as_deref(), Some("5"));
        let boolean_parent: RawModel = serde_json::from_str(r#"{"parent":true}"#).unwrap();
        assert_eq!(boolean_parent.parent.as_deref(), Some("true"));

        let model: RawModel = serde_json::from_str(
            r#"{"ambientocclusion":"false","textures":{"number":7},"parent":"builtin/generated"}"#,
        )
        .unwrap();
        assert!(!model.ambient_occlusion);
        assert_eq!(model.textures["number"], "7");

        let no: RawModel =
            serde_json::from_str(r#"{"ambientocclusion":"no","parent":"builtin/generated"}"#)
                .unwrap();
        let zero: RawModel =
            serde_json::from_str(r#"{"ambientocclusion":0,"parent":"builtin/generated"}"#).unwrap();
        assert!(!no.ambient_occlusion);
        assert!(!zero.ambient_occlusion);
        assert!(serde_json::from_str::<RawModel>(
            r#"{"textures":{"bad":{}},"parent":"builtin/generated"}"#
        )
        .is_err());
    }

    #[test]
    fn block_model_loader_accepts_gson_lexical_forms_without_touching_string_contents() {
        let root = r#"
            /* pack comment */ {
                parent: 'builtin/generated',
                textures: {
                    all: 'blocks/stone',
                    marker: 'a//b/*c*/,[]{}:\"quoted\" and \'apostrophe\'',
                },
                ambientocclusion: 'false', // line comment
                ignored: [1, 2,],
            }
        "#;
        let model = resolve_model(
            root,
            r#"{"parent":"block/root"}"#,
            r#"{"parent":"block/middle"}"#,
        );

        assert_eq!(model.textures["all"], "blocks/stone");
        assert_eq!(
            model.textures["marker"],
            "a//b/*c*/,[]{}:\"quoted\" and 'apostrophe'"
        );
        assert!(!model.ambient_occlusion);
    }

    #[test]
    fn lenient_model_normalizer_keeps_strict_structure_checks_and_rejects_unclosed_comments() {
        let neither = normalize_lenient_model_json(b"{/*comment*/ bare: value,}");
        assert!(serde_json::from_str::<RawModel>(&neither).is_err());

        let unterminated =
            normalize_lenient_model_json(b"{parent:'builtin/generated'} /* never closed");
        assert!(serde_json::from_str::<RawModel>(&unterminated).is_err());

        let strict = r#"{"textures":{"marker":"a//b/*c*/,[]{}:'"},"parent":"builtin/generated"}"#;
        let normalized = normalize_lenient_model_json(strict.as_bytes());
        let model: RawModel = serde_json::from_str(&normalized).unwrap();
        assert_eq!(model.textures["marker"], "a//b/*c*/,[]{}:'");
    }
}
