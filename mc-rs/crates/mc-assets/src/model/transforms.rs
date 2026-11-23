//! A model's `display` block — the per-context rotation/translation/scale an item is drawn
//! with. Closes the S12 -> S09 cross-section request.
//!
// java: client/renderer/block/model/ItemCameraTransforms.java,
//       client/renderer/block/model/ItemTransformVec3f.java,
//       client/renderer/block/model/ModelBlock.java:166-180 (per-type parent inheritance)
//!
//! Vanilla's `RenderItem` never draws an item model raw: it calls
//! `ItemCameraTransforms.applyTransform(type)` first, which pushes
//! `translate · rotY · rotX · rotZ · scale` for whichever context the item is being drawn in.
//! A dropped `diamond_sword` looks wrong without it not because it is the wrong *size* but
//! because `display.ground` also rotates it.
//!
//! Two vanilla details that are easy to miss and are reproduced here:
//!
//! * **`translation` is divided by 16** (`ItemTransformVec3f.Deserializer` calls
//!   `vector3f1.scale(0.0625F)`), so the JSON's numbers are in model-space texels while the
//!   value carried here is in blocks, then clamped to `±1.5`. `scale` is clamped to `±4`.
//!   `rotation` is in **degrees** and is not scaled or clamped.
//! * **Inheritance is per context, not per block.** `ModelBlock.getTransform` falls back to the
//!   parent only for the contexts this model left at the default, so a child that overrides
//!   only `gui` still inherits `ground`/`thirdperson` from `item/handheld`.

use serde::{de, Deserialize, Deserializer};

use super::blockmodel::gson_float_array;

/// One context's transform. java: `ItemTransformVec3f`.
///
/// [`ItemTransform::DEFAULT`] (identity: no rotation, no translation, unit scale) is what
/// vanilla compares against to decide whether a context was set at all, so it is also the
/// value that triggers parent inheritance — see [`ItemCameraTransforms::inherit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItemTransform {
    /// Euler rotation in **degrees**, applied Y then X then Z (java: `applyTransform`'s three
    /// `GlStateManager.rotate` calls, in that order).
    pub rotation: [f32; 3],
    /// Translation in **blocks** — already the JSON value times `1/16` and clamped to `±1.5`.
    pub translation: [f32; 3],
    /// Scale, clamped to `±4`.
    pub scale: [f32; 3],
}

impl ItemTransform {
    /// java: `ItemTransformVec3f.DEFAULT`.
    pub const DEFAULT: ItemTransform = ItemTransform {
        rotation: [0.0; 3],
        translation: [0.0; 3],
        scale: [1.0; 3],
    };

    /// Whether this context was left at vanilla's default, which is what makes
    /// `ItemCameraTransforms.func_181687_c` return `false` and hands the lookup to the parent.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == ItemTransform::DEFAULT
    }
}

impl Default for ItemTransform {
    fn default() -> Self {
        ItemTransform::DEFAULT
    }
}

/// Which drawing context a transform applies to. java: `ItemCameraTransforms.TransformType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformType {
    /// Held by another entity, seen from outside. JSON key `thirdperson`.
    ThirdPerson,
    /// Held by the local player, seen down their own arm. JSON key `firstperson`.
    FirstPerson,
    /// Worn on the head. JSON key `head`.
    Head,
    /// Inventory / hotbar icon. JSON key `gui`.
    Gui,
    /// A dropped `EntityItem` lying on the ground. JSON key `ground`.
    Ground,
    /// An item frame. JSON key `fixed`.
    Fixed,
}

impl TransformType {
    /// Every context, in `TransformType`'s declaration order (minus `NONE`, which has no JSON
    /// key and always resolves to the default).
    pub const VALUES: [TransformType; 6] = [
        TransformType::ThirdPerson,
        TransformType::FirstPerson,
        TransformType::Head,
        TransformType::Gui,
        TransformType::Ground,
        TransformType::Fixed,
    ];

    /// The `display` sub-object key this context reads.
    #[must_use]
    pub fn json_key(self) -> &'static str {
        match self {
            TransformType::ThirdPerson => "thirdperson",
            TransformType::FirstPerson => "firstperson",
            TransformType::Head => "head",
            TransformType::Gui => "gui",
            TransformType::Ground => "ground",
            TransformType::Fixed => "fixed",
        }
    }
}

/// A model's whole `display` block. java: `ItemCameraTransforms`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ItemCameraTransforms {
    /// `display.thirdperson`.
    pub third_person: ItemTransform,
    /// `display.firstperson`.
    pub first_person: ItemTransform,
    /// `display.head`.
    pub head: ItemTransform,
    /// `display.gui`.
    pub gui: ItemTransform,
    /// `display.ground`.
    pub ground: ItemTransform,
    /// `display.fixed`.
    pub fixed: ItemTransform,
}

impl ItemCameraTransforms {
    /// java: `ItemCameraTransforms.DEFAULT` — every context identity.
    pub const DEFAULT: ItemCameraTransforms = ItemCameraTransforms {
        third_person: ItemTransform::DEFAULT,
        first_person: ItemTransform::DEFAULT,
        head: ItemTransform::DEFAULT,
        gui: ItemTransform::DEFAULT,
        ground: ItemTransform::DEFAULT,
        fixed: ItemTransform::DEFAULT,
    };

    /// java: `ItemCameraTransforms.getTransform`.
    #[must_use]
    pub fn get(&self, ty: TransformType) -> ItemTransform {
        match ty {
            TransformType::ThirdPerson => self.third_person,
            TransformType::FirstPerson => self.first_person,
            TransformType::Head => self.head,
            TransformType::Gui => self.gui,
            TransformType::Ground => self.ground,
            TransformType::Fixed => self.fixed,
        }
    }

    fn get_mut(&mut self, ty: TransformType) -> &mut ItemTransform {
        match ty {
            TransformType::ThirdPerson => &mut self.third_person,
            TransformType::FirstPerson => &mut self.first_person,
            TransformType::Head => &mut self.head,
            TransformType::Gui => &mut self.gui,
            TransformType::Ground => &mut self.ground,
            TransformType::Fixed => &mut self.fixed,
        }
    }

    /// Fill every context this model left at the default from `parent`.
    ///
    /// java: `ModelBlock.getTransform` — `parent != null && !cameraTransforms.func_181687_c(type)
    /// ? parent.getTransform(type) : cameraTransforms.getTransform(type)`. Note this is decided
    /// **per context**: a model that sets only `gui` still inherits the other five.
    #[must_use]
    pub fn inherit(mut self, parent: ItemCameraTransforms) -> Self {
        for ty in TransformType::VALUES {
            if self.get(ty).is_default() {
                *self.get_mut(ty) = parent.get(ty);
            }
        }
        self
    }
}

/// The raw `display` block as it appears in the JSON, before the `1/16` translation scale and
/// the clamps [`ItemTransform`] carries. Parsed as part of [`super::blockmodel::RawModel`];
/// [`RawDisplay::bake`] turns it into the public [`ItemCameraTransforms`].
///
/// A context given as an explicit `null` (`"gui": null`) is read as absent. That is a
/// deliberate leniency: Gson's `context.deserialize` returns `null` for it, and vanilla then
/// throws a `NullPointerException` later, when it bakes the model (every resource pack is
/// dropped) or draws the item in that context (a crash). A context that is an array or a
/// primitive fails the model, as `ItemTransformVec3f.Deserializer`'s `getAsJsonObject` does.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RawDisplay {
    #[serde(default)]
    thirdperson: Option<RawTransform>,
    #[serde(default)]
    firstperson: Option<RawTransform>,
    #[serde(default)]
    head: Option<RawTransform>,
    #[serde(default)]
    gui: Option<RawTransform>,
    #[serde(default)]
    ground: Option<RawTransform>,
    #[serde(default)]
    fixed: Option<RawTransform>,
}

#[derive(Debug, Clone)]
struct RawTransform {
    rotation: Option<[f32; 3]>,
    translation: Option<[f32; 3]>,
    scale: Option<[f32; 3]>,
}

impl<'de> Deserialize<'de> for RawTransform {
    /// java: `ItemTransformVec3f.Deserializer`: `json.getAsJsonObject()`, then per vector
    /// `if (!has(key)) default else JsonUtils.getJsonArray(...)` of exactly three
    /// `JsonUtils.getFloat` numbers. So a present `"rotation": null` (or a numeric string)
    /// fails the model, where mc-rs used to read it as the default before this change.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serde_json::Value::Object(mut object) = serde_json::Value::deserialize(deserializer)?
        else {
            return Err(de::Error::custom("display transform must be a JSON object"));
        };
        let mut vector = |key: &str| -> Result<Option<[f32; 3]>, D::Error> {
            object
                .remove(key)
                .map(|value| gson_float_array::<D::Error, 3>(value, key))
                .transpose()
        };
        Ok(Self {
            rotation: vector("rotation")?,
            translation: vector("translation")?,
            scale: vector("scale")?,
        })
    }
}

impl RawTransform {
    /// java: `ItemTransformVec3f.Deserializer.deserialize` — translation is scaled by `1/16`
    /// and clamped to `±1.5`, scale is clamped to `±4`, rotation is passed through as degrees.
    fn bake(&self) -> ItemTransform {
        let clamp3 = |v: [f32; 3], lo: f32, hi: f32| {
            [v[0].clamp(lo, hi), v[1].clamp(lo, hi), v[2].clamp(lo, hi)]
        };
        let t = self.translation.unwrap_or([0.0; 3]);
        ItemTransform {
            rotation: self.rotation.unwrap_or([0.0; 3]),
            translation: clamp3([t[0] * 0.0625, t[1] * 0.0625, t[2] * 0.0625], -1.5, 1.5),
            scale: clamp3(self.scale.unwrap_or([1.0; 3]), -4.0, 4.0),
        }
    }
}

impl RawDisplay {
    /// This model's own `display` block, with contexts it omits left at the default so
    /// [`ItemCameraTransforms::inherit`] can tell them apart.
    #[must_use]
    pub fn bake(&self) -> ItemCameraTransforms {
        let one = |r: &Option<RawTransform>| {
            r.as_ref()
                .map_or(ItemTransform::DEFAULT, RawTransform::bake)
        };
        ItemCameraTransforms {
            third_person: one(&self.thirdperson),
            first_person: one(&self.firstperson),
            head: one(&self.head),
            gui: one(&self.gui),
            ground: one(&self.ground),
            fixed: one(&self.fixed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_is_divided_by_sixteen_and_clamped() {
        let raw: RawTransform = serde_json::from_str(r#"{"translation": [0, 3, 16]}"#).unwrap();
        let t = raw.bake();
        assert!((t.translation[1] - 3.0 / 16.0).abs() < 1e-6, "{t:?}");
        assert!((t.translation[2] - 1.0).abs() < 1e-6, "{t:?}");
        // java clamps to +-1.5 *after* the 1/16 scale, so 64 texels saturates at 1.5 blocks.
        let raw: RawTransform = serde_json::from_str(r#"{"translation": [64, -64, 0]}"#).unwrap();
        let t = raw.bake();
        assert_eq!(t.translation[0], 1.5);
        assert_eq!(t.translation[1], -1.5);
    }

    #[test]
    fn scale_is_clamped_but_rotation_is_not() {
        let raw: RawTransform =
            serde_json::from_str(r#"{"rotation": [0, 720, -45], "scale": [9, 0.25, -9]}"#).unwrap();
        let t = raw.bake();
        assert_eq!(t.scale, [4.0, 0.25, -4.0]);
        assert_eq!(t.rotation, [0.0, 720.0, -45.0]);
    }

    #[test]
    fn an_absent_context_is_the_identity_and_inherits() {
        let display: RawDisplay = serde_json::from_str(r#"{"gui": {"scale": [2, 2, 2]}}"#).unwrap();
        let child = display.bake();
        assert_eq!(child.gui.scale, [2.0, 2.0, 2.0]);
        assert!(child.ground.is_default());

        let parent = ItemCameraTransforms {
            ground: ItemTransform {
                rotation: [0.0, 45.0, 0.0],
                translation: [0.0, 0.125, 0.0],
                scale: [0.25, 0.25, 0.25],
            },
            gui: ItemTransform {
                rotation: [30.0, 0.0, 0.0],
                ..ItemTransform::DEFAULT
            },
            ..ItemCameraTransforms::DEFAULT
        };
        let merged = child.inherit(parent);
        // Inherited, because the child left `ground` at the default...
        assert_eq!(merged.ground, parent.ground);
        // ...and NOT inherited, because it set `gui` — per context, not per model.
        assert_eq!(merged.gui.scale, [2.0, 2.0, 2.0]);
        assert_eq!(merged.gui.rotation, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn every_transform_type_has_its_json_key() {
        let keys: Vec<&str> = TransformType::VALUES.iter().map(|t| t.json_key()).collect();
        assert_eq!(
            keys,
            [
                "thirdperson",
                "firstperson",
                "head",
                "gui",
                "ground",
                "fixed"
            ]
        );
    }
}
