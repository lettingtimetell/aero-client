//! The model and blockstate JSON readers accept and reject exactly
//! what the real 1.8.9 `ModelBlock.deserialize` and `ModelBlockDefinition.parseFromReader` do,
//! and read the same values.
//!
//! The fixture `tests/fixtures/model_json_ref.tsv` comes from `javaref/ModelJsonRef.java`, which
//! runs the recompiled vanilla classes over a corpus of small documents (regenerate with
//! `javaref/generate-model-json.ps1`). It pins `JsonUtils`' typed getters: `getFloat`/`getInt`
//! take JSON numbers only (a numeric string fails the model), `getInt` truncates decimals and
//! wraps, `shade` must be a real boolean, a present `null` fails where a getter meets it, and a
//! `textures` value may be a one-element array (Gson's `JsonArray.getAsString`). Floats are
//! compared as raw bits, so the reader must round the JSON text to `f32` once, as
//! `Float.parseFloat` does.
//!
//! One deliberate difference, checked as such: a display context given as `null`
//! (`"gui": null`). Gson's `context.deserialize` returns a null transform for it and vanilla then
//! throws a `NullPointerException` while baking (every resource pack is dropped) or drawing (a
//! crash); mc-rs reads it as the default transform.

use std::collections::BTreeMap;

use mc_assets::model::blockmodel::{parse_facing, RawElement, RawModel};
use mc_assets::model::blockstate::BlockStateJson;
use mc_assets::model::{ItemCameraTransforms, ItemTransform};

const FIXTURE: &str = include_str!("fixtures/model_json_ref.tsv");

struct Case<'a> {
    kind: &'a str,
    name: &'a str,
    json: &'a str,
    accept: bool,
    summary: &'a str,
}

fn cases() -> Vec<Case<'static>> {
    FIXTURE
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            assert_eq!(fields.len(), 5, "malformed fixture line: {line}");
            Case {
                kind: fields[0],
                name: fields[1],
                json: fields[2],
                accept: match fields[3] {
                    "ACCEPT" => true,
                    "REJECT" => false,
                    other => panic!("unknown verdict {other}"),
                },
                summary: fields[4],
            }
        })
        .collect()
}

/// java: `Integer.toHexString(Float.floatToIntBits(f))`.
fn bits(value: f32) -> String {
    format!("{:x}", value.to_bits())
}

fn vec3(value: [f32; 3]) -> String {
    format!("{},{},{}", bits(value[0]), bits(value[1]), bits(value[2]))
}

/// java: `BlockPart.getFaceUvs`, which `setDefaultUvs` stores into every face without `uv`.
fn default_uv(facing: &str, from: [f32; 3], to: [f32; 3]) -> [f32; 4] {
    match facing {
        "down" | "up" => [from[0], from[2], to[0], to[2]],
        "north" | "south" => [from[0], 16.0 - to[1], to[0], 16.0 - from[1]],
        _ => [from[2], 16.0 - to[1], to[2], 16.0 - from[1]],
    }
}

fn element_summary(index: usize, element: &RawElement) -> String {
    let mut out = format!(
        ";e{index}=from:{} to:{} shade:{}",
        vec3(element.from),
        vec3(element.to),
        element.shade
    );
    if let Some(rotation) = &element.rotation {
        // java: `parseRotation` stores the origin already scaled by 0.0625F.
        let origin = rotation.origin.map(|value| value * 0.0625);
        out.push_str(&format!(
            " rot:{},{},{},{}",
            rotation.axis.to_ascii_lowercase(),
            bits(rotation.angle),
            rotation.rescale,
            vec3(origin)
        ));
    }
    // java: an `EnumMap`, printed in facing-name order by the generator.
    let faces: BTreeMap<&str, _> = element
        .faces
        .iter()
        .map(|(name, face)| (parse_facing(name).expect("face key").name(), face))
        .collect();
    for (name, face) in faces {
        let cull = face
            .cullface
            .as_deref()
            .and_then(parse_facing)
            .map_or("-", |facing| facing.name());
        let uv = face
            .uv
            .unwrap_or_else(|| default_uv(name, element.from, element.to));
        out.push_str(&format!(
            " {name}:cull={cull},tint={},tex={},rot={},uv={},{},{},{}",
            face.tintindex,
            face.texture,
            face.rotation,
            bits(uv[0]),
            bits(uv[1]),
            bits(uv[2]),
            bits(uv[3])
        ));
    }
    out
}

fn transform(value: ItemTransform) -> String {
    format!(
        "{}/{}/{}",
        vec3(value.rotation),
        vec3(value.translation),
        vec3(value.scale)
    )
}

fn model_summary(model: &RawModel) -> String {
    let parent = model.parent.as_deref().map_or_else(
        || "-".to_owned(),
        |parent| {
            if parent.contains(':') {
                parent.to_owned()
            } else {
                format!("minecraft:{parent}")
            }
        },
    );
    let textures: BTreeMap<&String, &String> = model.textures.iter().collect();
    let textures = textures
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut out = format!(
        "ao={};parent={parent};textures={{{textures}}}",
        model.ambient_occlusion
    );
    for (index, element) in model.elements.iter().enumerate() {
        out.push_str(&element_summary(index, element));
    }
    let display = model
        .display
        .as_ref()
        .map_or(ItemCameraTransforms::DEFAULT, |display| display.bake());
    for (key, value) in [
        ("thirdperson", display.third_person),
        ("firstperson", display.first_person),
        ("head", display.head),
        ("gui", display.gui),
        ("ground", display.ground),
        ("fixed", display.fixed),
    ] {
        out.push_str(&format!(";{key}={}", transform(value)));
    }
    out
}

fn blockstate_summary(state: &BlockStateJson) -> String {
    state.variants["normal"]
        .iter()
        .map(|variant| {
            format!(
                "model={},rotation=X{}_Y{},uvlock={},weight={}",
                variant.model,
                u32::from(variant.rotation.quarters_x) * 90,
                u32::from(variant.rotation.quarters_y) * 90,
                variant.uvlock,
                variant.weight
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// The Java summary with each `null` display context replaced by the default transform: the
/// one documented leniency (see the module docs).
fn with_null_contexts_as_default(summary: &str) -> String {
    let default = transform(ItemTransform::DEFAULT);
    summary
        .split(';')
        .map(|part| match part.split_once('=') {
            Some((key, "null")) => format!("{key}={default}"),
            _ => part.to_owned(),
        })
        .collect::<Vec<_>>()
        .join(";")
}

#[test]
fn model_and_blockstate_json_match_the_vanilla_readers() {
    let cases = cases();
    assert!(cases.len() >= 60, "the fixture lost cases: {}", cases.len());
    let mut failures = Vec::new();
    for case in &cases {
        let rust = match case.kind {
            "model" => serde_json::from_str::<RawModel>(case.json)
                .map(|model| model_summary(&model))
                .map_err(|error| error.to_string()),
            "blockstate" => serde_json::from_str::<BlockStateJson>(case.json)
                .map(|state| blockstate_summary(&state))
                .map_err(|error| error.to_string()),
            // The `.mcmeta` reader is private to `sprite.rs`; its unit test reads these rows.
            "animation" => continue,
            other => panic!("unknown kind {other}"),
        };
        match (case.accept, rust) {
            (true, Ok(summary)) => {
                let expected = with_null_contexts_as_default(case.summary);
                if summary != expected {
                    failures.push(format!(
                        "{}: read differently\n  java: {expected}\n  rust: {summary}",
                        case.name
                    ));
                }
            }
            (true, Err(error)) => {
                failures.push(format!(
                    "{}: java accepts, rust rejects: {error}",
                    case.name
                ));
            }
            (false, Ok(summary)) => failures.push(format!(
                "{}: java rejects ({}), rust accepts: {summary}",
                case.name, case.summary
            )),
            (false, Err(_)) => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The cases the fix is about are really in the fixture, with Java's verdict, so a regenerated
/// fixture cannot silently drop them.
#[test]
fn the_fixture_pins_the_coercion_edges() {
    let cases = cases();
    let verdict = |name: &str| {
        cases
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("fixture lacks {name}"))
            .accept
    };
    for rejected in [
        "from_numeric_string",
        "uv_numeric_strings",
        "rotation_angle_string",
        "rotation_null",
        "shade_zero",
        "shade_string",
        "face_rotation_string",
        "tintindex_string",
        "display_null",
        "display_vector_null",
        "variant_x_string",
        "variant_weight_string",
    ] {
        assert!(!verdict(rejected), "{rejected} must be a Java reject");
    }
    for accepted in [
        "face_rotation_decimal",
        "tintindex_decimal",
        "tintindex_wraps",
        "textures_single_array",
        "cullface_unknown",
        "display_context_null",
        "variant_x_decimal",
        "from_needs_single_rounding",
    ] {
        assert!(verdict(accepted), "{accepted} must be a Java accept");
    }
}
