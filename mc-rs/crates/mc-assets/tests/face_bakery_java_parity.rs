//! Every vanilla block model element face, baked by
//! [`mc_assets::model::facebakery::bake_quad`], is bit-identical to the real 1.8.9
//! `FaceBakery.makeBakedQuad`.
//!
//! The fixture `tests/fixtures/face_bakery_ref.txt` comes from `javaref/FaceBakeryRef.java`,
//! which runs the recompiled vanilla classes (`ModelBlock.deserialize` + `FaceBakery`) over every
//! `models/block/*.json` in the vanilla jar that declares its own elements. Each line holds one
//! quad's four stored vertices (`x y z u v` as raw float bits, in Java's `int[28]` slot order)
//! and `BakedQuad.getFace()`. Regenerate with `javaref/generate-face-bakery.ps1`.
//!
//! This test re-reads the same JSON from the same jar and bakes it on the Rust side. It caught
//! `FaceBakery.SCALE_ROTATION_22_5` typed as `0.0204313` instead of Java's `0.08239222`, which
//! shrank every rescaled 22.5° element (all 26 fire models, the attached tripwire hooks), and
//! `SCALE_ROTATION_GENERAL` one ulp high (every cross plant, flower pot, rail and stem).
//!
//! Like the rest of this directory, it never skips: no jar is a panic naming the path.

mod common;

use std::collections::HashMap;
use std::io::Read;

use mc_assets::model::blockmodel::{parse_facing, RawModel};
use mc_assets::model::facebakery::{bake_quad, BakeRotation};
use mc_assets::sprite::UvRect;

const FIXTURE: &str = include_str!("fixtures/face_bakery_ref.txt");

struct RefQuad<'a> {
    model: &'a str,
    rotation: &'a str,
    uvlock: bool,
    element: usize,
    facing: &'a str,
    /// Per stored slot: x, y, z, u, v raw bits.
    verts: [[u32; 5]; 4],
    face: &'a str,
}

fn parse_bits(s: &str) -> u32 {
    u32::from_str_radix(s, 16).unwrap_or_else(|e| panic!("bad hex {s:?}: {e}"))
}

fn parse_fixture() -> (UvRect, Vec<RefQuad<'static>>) {
    let mut lines = FIXTURE.lines();
    let header: Vec<&str> = lines.next().expect("sprite header").split(' ').collect();
    assert_eq!(header[0], "sprite");
    let sprite = UvRect {
        min_u: f32::from_bits(parse_bits(header[1])),
        max_u: f32::from_bits(parse_bits(header[2])),
        min_v: f32::from_bits(parse_bits(header[3])),
        max_v: f32::from_bits(parse_bits(header[4])),
    };
    let mut quads = Vec::new();
    for line in lines {
        let t: Vec<&'static str> = line.split(' ').collect();
        assert_eq!(t.len(), 5 + 20 + 1, "malformed fixture line {line:?}");
        let mut verts = [[0u32; 5]; 4];
        for (v, slot) in verts.iter_mut().enumerate() {
            for (k, value) in slot.iter_mut().enumerate() {
                *value = parse_bits(t[5 + v * 5 + k]);
            }
        }
        quads.push(RefQuad {
            model: t[0],
            rotation: t[1],
            uvlock: t[2] == "1",
            element: t[3].parse().unwrap(),
            facing: t[4],
            verts,
            face: t[25],
        });
    }
    (sprite, quads)
}

/// `ModelRotation.X<a>_Y<b>` -> quarter turns.
fn bake_rotation(name: &str, uvlock: bool) -> BakeRotation {
    let (x, y) = name
        .strip_prefix('X')
        .and_then(|rest| rest.split_once("_Y"))
        .unwrap_or_else(|| panic!("bad rotation {name:?}"));
    let x: u16 = x.parse().unwrap();
    let y: u16 = y.parse().unwrap();
    BakeRotation {
        quarters_x: (x / 90) as u8,
        quarters_y: (y / 90) as u8,
        uvlock,
    }
}

fn load_models(names: impl Iterator<Item = &'static str>) -> HashMap<&'static str, RawModel> {
    let jar = common::vanilla_jar("face_bakery_java_parity");
    let file = std::fs::File::open(&jar).unwrap_or_else(|e| panic!("open {jar:?}: {e}"));
    let mut zip = zip::ZipArchive::new(file).expect("read 1.8.9.jar");
    let mut models = HashMap::new();
    for name in names {
        if models.contains_key(name) {
            continue;
        }
        let mut json = String::new();
        zip.by_name(&format!("assets/minecraft/models/block/{name}.json"))
            .unwrap_or_else(|e| panic!("{name}.json missing from the jar: {e}"))
            .read_to_string(&mut json)
            .unwrap();
        let model: RawModel =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("{name}.json: {e}"));
        models.insert(name, model);
    }
    models
}

#[test]
fn every_vanilla_block_model_quad_bakes_bit_identically_to_java_face_bakery() {
    let (sprite, quads) = parse_fixture();
    assert_eq!(
        quads.len(),
        5027,
        "fixture size changed; regenerate both sides together"
    );
    let models = load_models(quads.iter().map(|q| q.model));
    assert_eq!(models.len(), 252);

    let mut mismatches = Vec::new();
    for q in &quads {
        let model = &models[q.model];
        let element = &model.elements[q.element];
        let face = &element.faces[q.facing];
        let facing = parse_facing(q.facing).expect("fixture facing");
        let baked = bake_quad(
            element,
            facing,
            face,
            sprite,
            bake_rotation(q.rotation, q.uvlock),
        );
        let got: [[u32; 5]; 4] = std::array::from_fn(|slot| {
            let v = baked.verts[slot];
            [
                v.pos[0].to_bits(),
                v.pos[1].to_bits(),
                v.pos[2].to_bits(),
                v.uv[0].to_bits(),
                v.uv[1].to_bits(),
            ]
        });
        let got_face = baked.face.map(|f| format!("{f:?}").to_lowercase());
        if got != q.verts || got_face.as_deref() != Some(q.face) {
            mismatches.push(format!(
                "{} {} uvlock={} element {} {}: java {:08x?} face {}, rust {:08x?} face {:?}",
                q.model, q.rotation, q.uvlock, q.element, q.facing, q.verts, q.face, got, got_face
            ));
        }
    }
    // Identity-rotation differences first: they are the most direct evidence.
    mismatches.sort_by_key(|m| !m.contains(" X0_Y0 "));
    let mut by_rotation: std::collections::BTreeMap<&str, usize> = Default::default();
    for m in &mismatches {
        let rotation = m.split(' ').nth(1).unwrap_or("?");
        *by_rotation
            .entry(match rotation {
                "X0_Y0" => "X0_Y0",
                "X90_Y90" => "X90_Y90",
                "X0_Y270" => "X0_Y270",
                "X180_Y180" => "X180_Y180",
                _ => "?",
            })
            .or_default() += 1;
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} quads differ from Java's FaceBakery (by rotation {by_rotation:?}); first 10:\n{}",
        mismatches.len(),
        quads.len(),
        mismatches
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Negative control: the pre-fix constant (`0.0204313`) is rejected by the Java fixture.
///
/// `fire_floor` element 0 is the south plane `from (0, 0, 8.8) to (16, 22.4, 8.8)`, rotated
/// -22.5° about X around (8, 8, 8) with `rescale`. Its slot 0 is the top-west corner
/// (0, 1.4, 0.55). Replaying the rotation with the old stretch puts that corner's Y about 0.08
/// blocks lower than Java's, far outside any rounding, so the bit-exact test above cannot pass
/// with the old constant.
#[test]
fn the_old_hand_typed_rescale_constant_is_rejected_by_the_java_fixture() {
    let (_, quads) = parse_fixture();
    let q = quads
        .iter()
        .find(|q| q.model == "fire_floor" && q.rotation == "X0_Y0" && q.element == 0)
        .expect("fire_floor element 0 in the fixture");
    let java_y = f64::from(f32::from_bits(q.verts[0][1]));

    let replay = |stretch: f64| {
        let (y, z, origin) = (22.4 / 16.0, 8.8 / 16.0, 0.5);
        let angle = (-22.5f64).to_radians();
        let rotated = (y - origin) * angle.cos() - (z - origin) * angle.sin();
        rotated * (1.0 + stretch) + origin
    };
    let old = replay(0.020_431_3);
    let fixed = replay(1.0 / (22.5f64.to_radians().cos()) - 1.0);
    assert!(
        (fixed - java_y).abs() < 1e-5,
        "the replay must reproduce Java with Java's constant: {fixed} vs {java_y}"
    );
    assert!(
        (old - java_y).abs() > 0.05,
        "the old constant must land visibly off Java's corner: {old} vs {java_y}"
    );
}

/// The fixture really exercises the two `rescale` constants this test exists for: rescaled
/// 22.5° fire planes and rescaled 45° cross planes are both present, so a regression in either
/// constant cannot pass by absence.
#[test]
fn the_fixture_covers_both_rescale_constants() {
    let (_, quads) = parse_fixture();
    let models = load_models(quads.iter().map(|q| q.model));
    let rescaled = |angle: f32| {
        quads.iter().any(|q| {
            models[q.model].elements[q.element]
                .rotation
                .as_ref()
                .is_some_and(|r| r.rescale && r.angle.abs() == angle)
        })
    };
    assert!(
        rescaled(22.5),
        "no rescaled 22.5° element (fire) in the fixture"
    );
    assert!(
        rescaled(45.0),
        "no rescaled 45° element (cross) in the fixture"
    );
    assert!(quads.iter().any(|q| q.model == "fire_floor"));
    assert!(quads.iter().any(|q| q.model == "cross"));
}
