//! Differential test: every function in `mc-math` against values dumped from a real JVM.
//!
//! **Zero tolerance.** Every comparison is on raw IEEE-754 bit patterns
//! (`f32::to_bits` / `f64::to_bits`), never on a decimal rendering and never with an epsilon.
//! A single-ulp difference in `sin` is exactly the failure this crate exists to prevent
//! (hazard H1), so it must fail the test.
//!
//! The fixtures in `tests/fixtures/` are produced by `javaref/MathRefGen.java`; run
//! `javaref/generate.ps1` (JDK on PATH) to regenerate them. Fixture format, little-endian
//! throughout:
//!
//! ```text
//! "MCRF" u32:version=1 u8:sig_len sig_bytes u32:count  then `count` records
//! ```
//!
//! where `sig` is `<arg types>'>'<result types>` over `f` = f32, `d` = f64, `i` = i32,
//! `l` = i64, `b` = bool-as-u8, and each record is the argument words followed by the result
//! words, all as raw bits.

use std::path::PathBuf;

use mc_math::aabb::Aabb;
use mc_math::facing::Facing;
use mc_math::math_helper as mh;
use mc_math::random::JavaRandom;
use mc_math::BlockPos;

// =============================================================================================
// fixture reader
// =============================================================================================

struct Fixture {
    name: String,
    data: Vec<u8>,
    cursor: usize,
    count: u32,
}

impl Fixture {
    fn open(name: &str, expect_sig: &str) -> Fixture {
        let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures"]
            .iter()
            .collect::<PathBuf>()
            .join(format!("{name}.bin"));
        let data = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "missing fixture {}: {e}\n\
                 regenerate with `javaref/generate.ps1` (needs a JDK on PATH)",
                path.display()
            )
        });
        assert_eq!(&data[0..4], b"MCRF", "{name}: bad magic");
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(version, 1, "{name}: unsupported fixture version");
        let sig_len = data[8] as usize;
        let sig = std::str::from_utf8(&data[9..9 + sig_len]).expect("sig must be ascii");
        assert_eq!(sig, expect_sig, "{name}: signature changed under the test");
        let o = 9 + sig_len;
        let count = u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        Fixture {
            name: name.to_owned(),
            data,
            cursor: o + 4,
            count,
        }
    }

    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }

    fn f64(&mut self) -> f64 {
        f64::from_bits(self.u64())
    }

    fn i32(&mut self) -> i32 {
        self.u32() as i32
    }

    fn i64(&mut self) -> i64 {
        self.u64() as i64
    }

    fn bool(&mut self) -> bool {
        let b = self.data[self.cursor];
        self.cursor += 1;
        b != 0
    }

    fn string(&mut self) -> String {
        let len = self.u32() as usize;
        let end = self.cursor + len;
        let value = String::from_utf8(self.data[self.cursor..end].to_vec())
            .expect("fixture string must be UTF-8");
        self.cursor = end;
        value
    }

    fn u32(&mut self) -> u32 {
        let d = &self.data[self.cursor..self.cursor + 4];
        self.cursor += 4;
        u32::from_le_bytes([d[0], d[1], d[2], d[3]])
    }

    fn u64(&mut self) -> u64 {
        let d = &self.data[self.cursor..self.cursor + 8];
        self.cursor += 8;
        u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
    }

    fn finish(self) {
        assert_eq!(
            self.cursor,
            self.data.len(),
            "{}: {} trailing bytes — reader and generator disagree on the record layout",
            self.name,
            self.data.len() - self.cursor
        );
    }
}

/// Bit-exact assertion helpers. Failures report both the bit pattern and the decimal, because
/// the decimal alone can print identically for two different values.
fn eq_f32(fx: &str, i: u32, args: &str, got: f32, want: f32) {
    assert!(
        got.to_bits() == want.to_bits(),
        "{fx}[{i}]({args}): got {got:?} (0x{:08X}), java gave {want:?} (0x{:08X})",
        got.to_bits(),
        want.to_bits()
    );
}

fn eq_f64(fx: &str, i: u32, args: &str, got: f64, want: f64) {
    assert!(
        got.to_bits() == want.to_bits(),
        "{fx}[{i}]({args}): got {got:?} (0x{:016X}), java gave {want:?} (0x{:016X})",
        got.to_bits(),
        want.to_bits()
    );
}

fn eq<T: PartialEq + std::fmt::Debug>(fx: &str, i: u32, args: &str, got: T, want: T) {
    assert!(
        got == want,
        "{fx}[{i}]({args}): got {got:?}, java gave {want:?}"
    );
}

fn aabb_from_values(v: &[f64]) -> Aabb {
    assert_eq!(v.len(), 6, "AABB fixture must have six coordinates");
    Aabb::from_bounds(v[0], v[1], v[2], v[3], v[4], v[5])
}

fn eq_aabb(fx: &str, i: u32, args: &str, got: Aabb, fixture: &mut Fixture) {
    let want: Vec<f64> = (0..6).map(|_| fixture.f64()).collect();
    eq_aabb_values(fx, i, args, got, &want);
}

fn eq_aabb_values(fx: &str, i: u32, args: &str, got: Aabb, want: &[f64]) {
    for (axis, value) in [
        got.min.x, got.min.y, got.min.z, got.max.x, got.max.y, got.max.z,
    ]
    .into_iter()
    .enumerate()
    {
        eq_f64(fx, i, &format!("{args}; output[{axis}]"), value, want[axis]);
    }
}

// =============================================================================================
// the embedded tables
// =============================================================================================

/// Find an angle whose vanilla index `(int)(a * 10430.378F) & 65535` is exactly `i`.
///
/// Lets the test reach every one of the 65,536 entries through the public [`mh::sin`] API,
/// rather than poking at the private table.
fn probe_angle_for(index: usize) -> f32 {
    let mut f = index as f32 / 10430.378_f32;
    for _ in 0..256 {
        let idx = ((f * 10430.378_f32) as i32 & 65535) as usize;
        if idx == index {
            return f;
        }
        // The index is monotonic in `f` over the positive range, so step one ulp towards it.
        f = if idx < index {
            f32::from_bits(f.to_bits() + 1)
        } else {
            f32::from_bits(f.to_bits() - 1)
        };
    }
    panic!("no f32 angle maps to sine-table index {index}");
}

fn jvm_sin_table() -> Vec<f32> {
    let bytes = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("tables")
            .join("sin_table.bin"),
    )
    .expect("src/tables/sin_table.bin — regenerate with javaref/generate.ps1");
    assert_eq!(bytes.len(), 65536 * 4, "sin table must be 65536 f32 words");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

/// All 65,536 table entries, reached through the public API, must equal the JVM's dump.
///
/// This is hazard H1's keystone: a wrong index, a byte-swapped load, or a recomputation that
/// drifts by one ulp all fail here.
#[test]
fn sin_table_is_the_jvm_table() {
    let want = jvm_sin_table();
    let mut covered = vec![false; 65536];
    for (i, &w) in want.iter().enumerate() {
        let angle = probe_angle_for(i);
        eq_f32(
            "sin_table",
            i as u32,
            &format!("angle={angle:?}"),
            mh::sin(angle),
            w,
        );
        // cos is the same table read a quarter-turn along.
        let cos_angle = probe_angle_for((i + 65536 - 16384) % 65536);
        eq_f32(
            "cos_table",
            i as u32,
            &format!("angle={cos_angle:?}"),
            mh::cos(cos_angle),
            w,
        );
        covered[i] = true;
    }
    assert!(
        covered.iter().all(|c| *c),
        "every table entry must be checked"
    );
}

/// Documents why the table is embedded rather than recomputed.
///
/// The table is `(float)Math.sin(i * PI * 2 / 65536)`. `Math.sin` is only specified to within
/// 1 ulp and `f64::sin` is whatever the host libm provides, so recomputing at startup would
/// make the client's trigonometry depend on the machine it was built for. Reports the delta on
/// *this* host; the crate serves the embedded dump either way.
#[test]
fn recomputing_the_table_locally_is_not_a_substitute() {
    let want = jvm_sin_table();
    let differing = want
        .iter()
        .enumerate()
        .filter(|(i, w)| {
            let recomputed = ((*i as f64) * std::f64::consts::PI * 2.0 / 65536.0).sin() as f32;
            recomputed.to_bits() != w.to_bits()
        })
        .count();
    println!("host f64::sin would produce {differing} differing table entries out of 65536");
}

// =============================================================================================
// MathHelper
// =============================================================================================

macro_rules! check_f_f {
    ($name:literal, $f:expr) => {{
        let mut fx = Fixture::open($name, "f>f");
        for i in 0..fx.count {
            let a = fx.f32();
            let want = fx.f32();
            eq_f32($name, i, &format!("{a:?}"), $f(a), want);
        }
        let n = fx.count;
        fx.finish();
        n
    }};
}

macro_rules! check_f_i {
    ($name:literal, $f:expr) => {{
        let mut fx = Fixture::open($name, "f>i");
        for i in 0..fx.count {
            let a = fx.f32();
            let want = fx.i32();
            eq($name, i, &format!("{a:?}"), $f(a), want);
        }
        let n = fx.count;
        fx.finish();
        n
    }};
}

macro_rules! check_d_i {
    ($name:literal, $f:expr) => {{
        let mut fx = Fixture::open($name, "d>i");
        for i in 0..fx.count {
            let a = fx.f64();
            let want = fx.i32();
            eq($name, i, &format!("{a:?}"), $f(a), want);
        }
        let n = fx.count;
        fx.finish();
        n
    }};
}

macro_rules! check_d_d {
    ($name:literal, $f:expr) => {{
        let mut fx = Fixture::open($name, "d>d");
        for i in 0..fx.count {
            let a = fx.f64();
            let want = fx.f64();
            eq_f64($name, i, &format!("{a:?}"), $f(a), want);
        }
        let n = fx.count;
        fx.finish();
        n
    }};
}

macro_rules! check_i_i {
    ($name:literal, $f:expr) => {{
        let mut fx = Fixture::open($name, "i>i");
        for i in 0..fx.count {
            let a = fx.i32();
            let want = fx.i32();
            eq($name, i, &format!("{a}"), $f(a), want);
        }
        let n = fx.count;
        fx.finish();
        n
    }};
}

macro_rules! check_ii_i {
    ($name:literal, $f:expr) => {{
        let mut fx = Fixture::open($name, "ii>i");
        for i in 0..fx.count {
            let a = fx.i32();
            let b = fx.i32();
            let want = fx.i32();
            eq($name, i, &format!("{a}, {b}"), $f(a, b), want);
        }
        let n = fx.count;
        fx.finish();
        n
    }};
}

macro_rules! check_iii_i {
    ($name:literal, $f:expr) => {{
        let mut fx = Fixture::open($name, "iii>i");
        for i in 0..fx.count {
            let a = fx.i32();
            let b = fx.i32();
            let c = fx.i32();
            let want = fx.i32();
            eq($name, i, &format!("{a}, {b}, {c}"), $f(a, b, c), want);
        }
        let n = fx.count;
        fx.finish();
        n
    }};
}

macro_rules! check_ddd_d {
    ($name:literal, $f:expr) => {{
        let mut fx = Fixture::open($name, "ddd>d");
        for i in 0..fx.count {
            let a = fx.f64();
            let b = fx.f64();
            let c = fx.f64();
            let want = fx.f64();
            eq_f64($name, i, &format!("{a:?}, {b:?}, {c:?}"), $f(a, b, c), want);
        }
        let n = fx.count;
        fx.finish();
        n
    }};
}

#[test]
fn math_helper_trig_and_roots() {
    let mut n = 0;
    n += check_f_f!("sin", mh::sin);
    n += check_f_f!("cos", mh::cos);
    n += check_f_f!("sqrt_float", mh::sqrt_f);
    {
        let mut fx = Fixture::open("sqrt_double", "d>f");
        for i in 0..fx.count {
            let a = fx.f64();
            let want = fx.f32();
            eq_f32("sqrt_double", i, &format!("{a:?}"), mh::sqrt_d(a), want);
        }
        n += fx.count;
        fx.finish();
    }
    {
        let mut fx = Fixture::open("atan2", "dd>d");
        for i in 0..fx.count {
            let a = fx.f64();
            let b = fx.f64();
            let want = fx.f64();
            eq_f64("atan2", i, &format!("{a:?}, {b:?}"), mh::atan2(a, b), want);
        }
        n += fx.count;
        fx.finish();
    }
    n += check_d_d!("fast_inv_sqrt", mh::fast_inv_sqrt);
    println!("trig/roots: {n} records bit-exact");
}

#[test]
fn math_helper_rounding() {
    let mut n = 0;
    n += check_f_i!("floor_float", mh::floor_f32);
    n += check_d_i!("floor_double", mh::floor_f64);
    n += check_f_i!("ceiling_float_int", mh::ceil_f32);
    n += check_d_i!("ceiling_double_int", mh::ceil_f64);
    n += check_d_i!("truncate_double_to_int", mh::truncate_f64_to_i32);
    n += check_d_i!("func_154353_e", mh::fold_sign_to_i32);
    n += check_d_d!("frac_double", mh::frac_f64);
    {
        let mut fx = Fixture::open("floor_double_long", "d>l");
        for i in 0..fx.count {
            let a = fx.f64();
            let want = fx.i64();
            eq(
                "floor_double_long",
                i,
                &format!("{a:?}"),
                mh::floor_f64_i64(a),
                want,
            );
        }
        n += fx.count;
        fx.finish();
    }
    println!("rounding: {n} records bit-exact");
}

#[test]
fn math_helper_abs_clamp_and_angles() {
    let mut n = 0;
    n += check_f_f!("abs_float", mh::abs_f32);
    n += check_i_i!("abs_int", mh::abs_i32);
    n += check_iii_i!("clamp_int", mh::clamp_i32);
    n += check_ddd_d!("clamp_double", mh::clamp_f64);
    n += check_f_f!("wrap_angle_to_180_float", mh::wrap_degrees_f32);
    n += check_d_d!("wrap_angle_to_180_double", mh::wrap_degrees_f64);
    {
        let mut fx = Fixture::open("clamp_float", "fff>f");
        for i in 0..fx.count {
            let a = fx.f32();
            let b = fx.f32();
            let c = fx.f32();
            let want = fx.f32();
            eq_f32(
                "clamp_float",
                i,
                &format!("{a:?}, {b:?}, {c:?}"),
                mh::clamp_f32(a, b, c),
                want,
            );
        }
        n += fx.count;
        fx.finish();
    }
    {
        let mut fx = Fixture::open("abs_max", "dd>d");
        for i in 0..fx.count {
            let a = fx.f64();
            let b = fx.f64();
            let want = fx.f64();
            eq_f64(
                "abs_max",
                i,
                &format!("{a:?}, {b:?}"),
                mh::abs_max(a, b),
                want,
            );
        }
        n += fx.count;
        fx.finish();
    }
    println!("abs/clamp/angles: {n} records bit-exact");
}

#[test]
fn math_helper_integers_and_colours() {
    let mut n = 0;
    n += check_ii_i!("round_up", mh::round_up);
    n += check_ii_i!("multiply_color", mh::multiply_color);
    n += check_i_i!("round_up_to_power_of_two", mh::round_up_to_power_of_two);
    n += check_i_i!("calculate_log_base_two", mh::calculate_log_base_two);
    n += check_iii_i!("pack_color", mh::pack_color);
    {
        let mut fx = Fixture::open("pack_color_float", "fff>i");
        for i in 0..fx.count {
            let a = fx.f32();
            let b = fx.f32();
            let c = fx.f32();
            let want = fx.i32();
            eq(
                "pack_color_float",
                i,
                &format!("{a:?}, {b:?}, {c:?}"),
                mh::pack_color_f32(a, b, c),
                want,
            );
        }
        n += fx.count;
        fx.finish();
    }
    {
        let mut fx = Fixture::open("hsv_to_rgb", "fff>i");
        for i in 0..fx.count {
            let a = fx.f32();
            let b = fx.f32();
            let c = fx.f32();
            let want = fx.i32();
            eq(
                "hsv_to_rgb",
                i,
                &format!("{a:?}, {b:?}, {c:?}"),
                mh::hsv_to_rgb(a, b, c),
                want,
            );
        }
        n += fx.count;
        fx.finish();
    }
    {
        let mut fx = Fixture::open("coordinate_random", "iii>l");
        for i in 0..fx.count {
            let a = fx.i32();
            let b = fx.i32();
            let c = fx.i32();
            let want = fx.i64();
            eq(
                "coordinate_random",
                i,
                &format!("{a}, {b}, {c}"),
                mh::coordinate_random(a, b, c),
                want,
            );
        }
        n += fx.count;
        fx.finish();
    }
    println!("integers/colours: {n} records bit-exact");
}

// =============================================================================================
// AxisAlignedBB
// =============================================================================================

#[test]
fn aabb_sweeps() {
    let sig: String = "d".repeat(13) + ">d";
    let mut n = 0;
    for (name, which) in [
        ("aabb_calculate_x_offset", 0u8),
        ("aabb_calculate_y_offset", 1),
        ("aabb_calculate_z_offset", 2),
    ] {
        let mut fx = Fixture::open(name, &sig);
        for i in 0..fx.count {
            let a: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let b: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let off = fx.f64();
            let want = fx.f64();
            let sb = Aabb::from_bounds(a[0], a[1], a[2], a[3], a[4], a[5]);
            let ob = Aabb::from_bounds(b[0], b[1], b[2], b[3], b[4], b[5]);
            let got = match which {
                0 => sb.calculate_x_offset(&ob, off),
                1 => sb.calculate_y_offset(&ob, off),
                _ => sb.calculate_z_offset(&ob, off),
            };
            eq_f64(name, i, &format!("{sb} vs {ob} @ {off:?}"), got, want);
        }
        n += fx.count;
        fx.finish();
    }

    let isig: String = "d".repeat(12) + ">b";
    let mut fx = Fixture::open("aabb_intersects", &isig);
    for i in 0..fx.count {
        let a: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
        let b: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
        let want = fx.bool();
        let sb = Aabb::from_bounds(a[0], a[1], a[2], a[3], a[4], a[5]);
        let ob = Aabb::from_bounds(b[0], b[1], b[2], b[3], b[4], b[5]);
        eq(
            "aabb_intersects",
            i,
            &format!("{sb} vs {ob}"),
            sb.intersects_with(&ob),
            want,
        );
    }
    n += fx.count;
    fx.finish();
    println!("aabb: {n} records bit-exact");
}

/// Full Java 1.8.9 `AxisAlignedBB` surface, beyond the three sweep methods and overlap check
/// above. These fixtures start with deterministic signed-zero, touching-face, reversed-corner,
/// NaN, infinity, subnormal, and finite-extreme controls, then use a fixed JVM seed.
#[test]
fn aabb_full_api() {
    let mut n = 0;

    {
        let mut fx = Fixture::open("aabb_from_bounds", &("d".repeat(6) + ">" + &"d".repeat(6)));
        for i in 0..fx.count {
            let raw: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let want: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            eq_aabb_values(
                "aabb_from_bounds",
                i,
                &format!("{raw:?}"),
                aabb_from_values(&raw),
                &want,
            );
            eq_aabb_values(
                "aabb_new",
                i,
                &format!("{raw:?}"),
                Aabb::new(
                    glam::DVec3::new(raw[0], raw[1], raw[2]),
                    glam::DVec3::new(raw[3], raw[4], raw[5]),
                ),
                &want,
            );
        }
        n += fx.count;
        fx.finish();
    }

    for (name, operation) in [
        ("aabb_add_coord", 0u8),
        ("aabb_expand", 1),
        ("aabb_contract", 2),
        ("aabb_offset", 3),
    ] {
        let sig = "d".repeat(9) + ">" + &"d".repeat(6);
        let mut fx = Fixture::open(name, &sig);
        for i in 0..fx.count {
            let raw: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let x = fx.f64();
            let y = fx.f64();
            let z = fx.f64();
            let input = aabb_from_values(&raw);
            let got = match operation {
                0 => input.add_coord(x, y, z),
                1 => input.expand(x, y, z),
                2 => input.contract(x, y, z),
                _ => input.offset(x, y, z),
            };
            eq_aabb(
                name,
                i,
                &format!("{input}; ({x:?}, {y:?}, {z:?})"),
                got,
                &mut fx,
            );
        }
        n += fx.count;
        fx.finish();
    }

    {
        let sig = "d".repeat(12) + ">" + &"d".repeat(6);
        let mut fx = Fixture::open("aabb_union", &sig);
        for i in 0..fx.count {
            let a: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let b: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let left = aabb_from_values(&a);
            let right = aabb_from_values(&b);
            eq_aabb(
                "aabb_union",
                i,
                &format!("{left} union {right}"),
                left.union(&right),
                &mut fx,
            );
        }
        n += fx.count;
        fx.finish();
    }

    {
        let mut fx = Fixture::open("aabb_is_vec_inside", &("d".repeat(9) + ">b"));
        for i in 0..fx.count {
            let raw: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let point = glam::DVec3::new(fx.f64(), fx.f64(), fx.f64());
            let input = aabb_from_values(&raw);
            eq(
                "aabb_is_vec_inside",
                i,
                &format!("{input}; {point:?}"),
                input.is_vec_inside(point),
                fx.bool(),
            );
        }
        n += fx.count;
        fx.finish();
    }

    {
        let mut fx = Fixture::open("aabb_average_edge_length", &("d".repeat(6) + ">d"));
        for i in 0..fx.count {
            let raw: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let input = aabb_from_values(&raw);
            eq_f64(
                "aabb_average_edge_length",
                i,
                &format!("{input}"),
                input.average_edge_length(),
                fx.f64(),
            );
        }
        n += fx.count;
        fx.finish();
    }

    {
        let mut fx = Fixture::open("aabb_has_nan", &("d".repeat(6) + ">b"));
        for i in 0..fx.count {
            let raw: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let input = aabb_from_values(&raw);
            eq(
                "aabb_has_nan",
                i,
                &format!("{input}"),
                input.has_nan(),
                fx.bool(),
            );
        }
        n += fx.count;
        fx.finish();
    }

    {
        let mut fx = Fixture::open("aabb_calculate_intercept", &("d".repeat(12) + ">bdddi"));
        for i in 0..fx.count {
            let raw: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            let a = glam::DVec3::new(fx.f64(), fx.f64(), fx.f64());
            let b = glam::DVec3::new(fx.f64(), fx.f64(), fx.f64());
            let present = fx.bool();
            let point = glam::DVec3::new(fx.f64(), fx.f64(), fx.f64());
            let face = fx.i32();
            let got = aabb_from_values(&raw).calculate_intercept(a, b);
            match (got, present) {
                (None, false) => assert_eq!(face, -1, "aabb_calculate_intercept[{i}]: Java no-hit sentinel"),
                (Some((got_point, got_face)), true) => {
                    eq_f64("aabb_calculate_intercept", i, "x", got_point.x, point.x);
                    eq_f64("aabb_calculate_intercept", i, "y", got_point.y, point.y);
                    eq_f64("aabb_calculate_intercept", i, "z", got_point.z, point.z);
                    eq("aabb_calculate_intercept", i, "face", got_face as i32, face);
                }
                (got, want) => panic!("aabb_calculate_intercept[{i}]({raw:?}, {a:?}->{b:?}): Rust {got:?}, Java present={want}"),
            }
        }
        n += fx.count;
        fx.finish();
    }

    {
        let sig = "i".repeat(6) + ">" + &"d".repeat(6);
        let mut fx = Fixture::open("aabb_block_positions", &sig);
        for i in 0..fx.count {
            let first = BlockPos::new(fx.i32(), fx.i32(), fx.i32());
            let second = BlockPos::new(fx.i32(), fx.i32(), fx.i32());
            eq_aabb(
                "aabb_block_positions",
                i,
                &format!("{first:?}, {second:?}"),
                Aabb::from_block_positions(first, second),
                &mut fx,
            );
        }
        n += fx.count;
        fx.finish();
    }

    {
        let mut fx = Fixture::open("aabb_display", &("d".repeat(6) + ">s"));
        for i in 0..fx.count {
            let raw: Vec<f64> = (0..6).map(|_| fx.f64()).collect();
            eq(
                "aabb_display",
                i,
                &format!("{raw:?}"),
                aabb_from_values(&raw).to_string(),
                fx.string(),
            );
        }
        n += fx.count;
        fx.finish();
    }

    println!("aabb full API: {n} records bit-exact");
}

// =============================================================================================
// java.util.Random
// =============================================================================================

#[test]
fn java_random_scalar_draws() {
    let mut n = 0;

    // Each fixture repeats the seed on every record; a new `JavaRandom` is created whenever
    // the seed changes, so the draw index within a seed is implicit.
    macro_rules! seeded {
        ($name:literal, $sig:literal, $read_want:ident, $draw:expr, $cmp:ident) => {{
            let mut fx = Fixture::open($name, $sig);
            let mut rng: Option<(i64, JavaRandom)> = None;
            for i in 0..fx.count {
                let seed = fx.i64();
                let want = fx.$read_want();
                let need_new = !matches!(&rng, Some((s, _)) if *s == seed);
                if need_new {
                    rng = Some((seed, JavaRandom::new(seed)));
                }
                let r = &mut rng.as_mut().expect("initialised above").1;
                #[allow(clippy::redundant_closure_call)]
                let got = ($draw)(r);
                $cmp($name, i, &format!("seed={seed}"), got, want);
            }
            n += fx.count;
            fx.finish();
        }};
    }

    seeded!(
        "rand_next_int",
        "l>i",
        i32,
        |r: &mut JavaRandom| r.next_i32(),
        eq
    );
    seeded!(
        "rand_next_long",
        "l>l",
        i64,
        |r: &mut JavaRandom| r.next_i64(),
        eq
    );
    seeded!(
        "rand_next_float",
        "l>f",
        f32,
        |r: &mut JavaRandom| r.next_f32(),
        eq_f32
    );
    seeded!(
        "rand_next_double",
        "l>d",
        f64,
        |r: &mut JavaRandom| r.next_f64(),
        eq_f64
    );
    seeded!(
        "rand_next_gaussian",
        "l>d",
        f64,
        |r: &mut JavaRandom| r.next_gaussian(),
        eq_f64
    );
    seeded!(
        "rand_next_boolean",
        "l>b",
        bool,
        |r: &mut JavaRandom| r.next_bool(),
        eq
    );

    println!("java.util.Random scalar draws: {n} records bit-exact");
}

#[test]
fn java_random_bounded_and_interleaved() {
    let mut n = 0;

    let mut fx = Fixture::open("rand_next_int_bound", "li>i");
    let mut rng: Option<(i64, JavaRandom)> = None;
    for i in 0..fx.count {
        let seed = fx.i64();
        let bound = fx.i32();
        let want = fx.i32();
        if !matches!(&rng, Some((s, _)) if *s == seed) {
            rng = Some((seed, JavaRandom::new(seed)));
        }
        let r = &mut rng.as_mut().unwrap().1;
        eq(
            "rand_next_int_bound",
            i,
            &format!("seed={seed} bound={bound}"),
            r.next_i32_bound(bound),
            want,
        );
    }
    n += fx.count;
    fx.finish();

    // Interleaved script: proves the nextGaussian cache interacts with the other draws exactly
    // as Java's does (a spurious extra LCG step shows up here and nowhere else).
    let mut fx = Fixture::open("rand_mixed", "lii>l");
    let mut rng: Option<(i64, JavaRandom)> = None;
    for i in 0..fx.count {
        let seed = fx.i64();
        let step = fx.i32();
        let op = fx.i32();
        let want = fx.i64();
        if step == 0 {
            rng = Some((seed, JavaRandom::new(seed)));
        }
        let r = &mut rng.as_mut().expect("script starts at step 0").1;
        let got: i64 = match op {
            0 => i64::from(r.next_i32()),
            1 => r.next_i64(),
            2 => i64::from(r.next_f32().to_bits()),
            3 => r.next_f64().to_bits() as i64,
            4 => r.next_gaussian().to_bits() as i64,
            5 => i64::from(r.next_bool()),
            6 => i64::from(r.next_i32_bound(97)),
            _ => i64::from(r.next_i32_bound(1024)),
        };
        eq(
            "rand_mixed",
            i,
            &format!("seed={seed} step={step} op={op}"),
            got,
            want,
        );
    }
    n += fx.count;
    fx.finish();

    println!("java.util.Random bounded/interleaved: {n} records bit-exact");
}

// =============================================================================================
// BlockPos / EnumFacing
// =============================================================================================

#[test]
fn blockpos_and_facing() {
    let mut n = 0;

    let mut fx = Fixture::open("blockpos_to_long", "iii>l");
    for i in 0..fx.count {
        let x = fx.i32();
        let y = fx.i32();
        let z = fx.i32();
        let want = fx.i64();
        eq(
            "blockpos_to_long",
            i,
            &format!("{x}, {y}, {z}"),
            BlockPos::new(x, y, z).to_long(),
            want,
        );
    }
    n += fx.count;
    fx.finish();

    let mut fx = Fixture::open("blockpos_from_long", "l>iii");
    for i in 0..fx.count {
        let v = fx.i64();
        let want = BlockPos::new(fx.i32(), fx.i32(), fx.i32());
        eq(
            "blockpos_from_long",
            i,
            &format!("{v}"),
            BlockPos::from_long(v),
            want,
        );
    }
    n += fx.count;
    fx.finish();

    let mut fx = Fixture::open("facing_from_angle", "d>i");
    for i in 0..fx.count {
        let a = fx.f64();
        let want = fx.i32();
        eq(
            "facing_from_angle",
            i,
            &format!("{a:?}"),
            i32::from(Facing::from_angle(a).index()),
            want,
        );
    }
    n += fx.count;
    fx.finish();

    println!("blockpos/facing: {n} records bit-exact");
}
