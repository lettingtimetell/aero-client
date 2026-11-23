//! The HUD layout and F3 formatting, checked against values transcribed **from the Java**.
//!
//! `tests/javaref/gen_hud_golden.py` is a line-by-line Python transcription of
//! `GuiIngame.renderPlayerStats` and `GuiOverlayDebug`, including `MathHelper.ceiling_float_int`
//! and `java.util.Random`'s LCG. It has never been run against this crate; its output is
//! committed as `tests/fixtures/hud_debug_golden.json` and replayed here.
//!
//! That is the point: a self-consistent port passes its own unit tests by construction. These
//! assertions fail unless the Rust agrees with a second, independent reading of the Java —
//! which is how the `% 2d` memory field and the heart row pitch were caught.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::debug::{DebugInfo, DebugOverlay};
use mc_ui::screens::hud::{heart_rows, Hud, PlayerHud};
use mc_ui::widget::{DrawList, ScaledResolution, UiTexture};
use serde_json::Value;

fn fixture() -> Value {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("hud_debug_golden.json");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("missing golden fixture {}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("the fixture is valid JSON")
}

fn ctx(width: i32, height: i32) -> ScreenCtx {
    let res = ScaledResolution::new(width as u32, height as u32, 1, false);
    assert_eq!(res.scale_factor(), 1, "the fixture assumes guiScale 1");
    ScreenCtx::new(res, 0.0)
}

fn f32_of(v: &Value, key: &str) -> f32 {
    v[key].as_f64().unwrap_or_else(|| panic!("{key} missing")) as f32
}

fn i32_of(v: &Value, key: &str) -> i32 {
    v[key].as_i64().unwrap_or_else(|| panic!("{key} missing")) as i32
}

fn bool_of(v: &Value, key: &str) -> bool {
    v[key].as_bool().unwrap_or_else(|| panic!("{key} missing"))
}

fn state_of(v: &Value) -> PlayerHud {
    PlayerHud {
        health: f32_of(v, "health"),
        max_health: f32_of(v, "max_health"),
        absorption: f32_of(v, "absorption"),
        armor: i32_of(v, "armor"),
        food: i32_of(v, "food"),
        saturation: f32_of(v, "saturation"),
        air: i32_of(v, "air"),
        underwater: bool_of(v, "underwater"),
        hardcore: bool_of(v, "hardcore"),
        poisoned: bool_of(v, "poisoned"),
        withered: bool_of(v, "withered"),
        hungry: bool_of(v, "hungry"),
        regenerating: bool_of(v, "regenerating"),
        survival: true,
        ..PlayerHud::default()
    }
}

/// The quads a `DrawList` holds, as `[x, y, u, v]` in the fixture's units.
///
/// Every HUD sprite is 9×9 out of a 256×256 sheet, so the first vertex of each six-vertex run
/// carries the whole tuple.
fn quads_of(list: &DrawList) -> Vec<[i32; 4]> {
    list.batches()
        .iter()
        .filter(|b| matches!(&b.texture, UiTexture::Resource(_)))
        .flat_map(|b| list.vertices()[b.range.clone()].chunks(6))
        .map(|q| {
            [
                q[0].pos[0] as i32,
                q[0].pos[1] as i32,
                (q[0].uv[0] * 256.0).round() as i32,
                (q[0].uv[1] * 256.0).round() as i32,
            ]
        })
        .collect()
}

/// The load-bearing one: every armour, heart, drumstick and bubble sprite lands on exactly the
/// pixel and texel the Java puts it on, for six different player states.
#[test]
fn every_hud_sprite_matches_the_java_transcription() {
    let f = fixture();
    let width = f["screen"][0].as_i64().unwrap() as i32;
    let height = f["screen"][1].as_i64().unwrap() as i32;
    let c = ctx(width, height);

    let cases = f["hud"].as_object().expect("hud cases");
    assert!(cases.len() >= 6, "the fixture should cover several states");
    for (name, case) in cases {
        let state = state_of(&case["state"]);
        let mut hud = Hud::new();
        hud.update_counter = case["state"]["update_counter"].as_i64().unwrap() as i32;

        let mut d = DrawList::new();
        hud.draw_player_stats(&mut d, &c, &state);
        let got = quads_of(&d);

        let want: Vec<[i32; 4]> = case["quads"]
            .as_array()
            .expect("quads")
            .iter()
            .map(|q| {
                let a = q.as_array().expect("quad");
                [
                    a[0].as_i64().unwrap() as i32,
                    a[1].as_i64().unwrap() as i32,
                    a[2].as_i64().unwrap() as i32,
                    a[3].as_i64().unwrap() as i32,
                ]
            })
            .collect();

        assert_eq!(
            got.len(),
            want.len(),
            "{name}: sprite count differs\n got {got:?}\nwant {want:?}"
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "{name}: sprite {i} differs (x, y, u, v)");
        }

        // And the row count the fixture derived independently.
        assert_eq!(
            heart_rows(state.max_health, state.absorption),
            case["heart_rows"].as_i64().unwrap() as i32,
            "{name}: heart row count"
        );
    }
}

/// The F3 left column's four coordinate lines, character for character.
#[test]
fn the_debug_coordinate_lines_match_the_java_format_strings() {
    let f = fixture();
    let overlay = DebugOverlay::new();
    for (name, case) in f["debug_left"].as_object().expect("debug_left") {
        let p = case["position"].as_array().unwrap();
        let r = case["rotation"].as_array().unwrap();
        let info = DebugInfo {
            position: (
                p[0].as_f64().unwrap(),
                p[1].as_f64().unwrap(),
                p[2].as_f64().unwrap(),
            ),
            rotation: (r[0].as_f64().unwrap() as f32, r[1].as_f64().unwrap() as f32),
            ..DebugInfo::default()
        };
        let got = overlay.left_column(&info, None);
        let want: Vec<&str> = case["lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        // The fixture only covers the four coordinate lines, which start after the seven-line
        // header the Java builds from `Minecraft.debug` and friends.
        let coords: Vec<&str> = got.iter().skip(7).map(String::as_str).collect();
        assert_eq!(coords, want, "{name}");
    }
}

/// The two memory lines. `% 2d` is *one* leading space, not a three-column field.
#[test]
fn the_debug_memory_lines_match_the_java_format_strings() {
    let f = fixture();
    let overlay = DebugOverlay::new();
    for (name, case) in f["debug_right"].as_object().expect("debug_right") {
        let m = case["memory"].as_array().unwrap();
        let info = DebugInfo {
            memory: (
                m[0].as_u64().unwrap(),
                m[1].as_u64().unwrap(),
                m[2].as_u64().unwrap(),
            ),
            ..DebugInfo::default()
        };
        let got = overlay.right_column(&info);
        let want: Vec<&str> = case["lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(&got[1..3], &want[..], "{name}");
    }
}

/// A sanity check on the fixture itself: it has to distinguish the states, or the test above
/// would pass against a HUD that ignored its input entirely.
#[test]
fn the_golden_states_really_do_produce_different_layouts() {
    let f = fixture();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (name, case) in f["hud"].as_object().unwrap() {
        let quads = case["quads"].as_array().unwrap();
        seen.insert(name.clone(), quads.len());
    }
    let distinct: std::collections::BTreeSet<usize> = seen.values().copied().collect();
    assert!(
        distinct.len() >= 4,
        "the six states collapse to {} distinct sprite counts: {seen:?}",
        distinct.len()
    );
}
