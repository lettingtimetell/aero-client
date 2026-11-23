// The volumetric clouds' tileable noise. Generated once on
// the GPU, in the frame the clouds are first needed: nothing is computed on the CPU and nothing
// happens at startup. Three entry points, one per texture:
//
// * `cs_shape` (128^3, RGBA8): R = Perlin-Worley, G/B/A = Worley fBm at 4, 8 and 16 cells per
//   tile. The clouds' low-frequency body.
// * `cs_detail` (32^3, RGBA8): R/G/B = Worley fBm at 2, 4 and 8 cells per tile, used to erode the
//   edges into wisps near the base and billows higher up.
// * `cs_weather` (256^2, RGBA8): R = coverage (large cloud fields), G = cloud type (0 = stratus,
//   1 = cumulus), B = a finer coverage field the trace blends in so the pattern never visibly
//   repeats.
//
// Every function tiles exactly: lattice and cell coordinates wrap at the texture's period before
// they are hashed, so the textures can be sampled with `Repeat` across the whole world.

@group(0) @binding(0) var shape_out: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var detail_out: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(2) var weather_out: texture_storage_2d<rgba8unorm, write>;

// PCG3D: a three-lane permuted congruential hash (an LCG step, cross-lane products, a 16-bit
// xorshift, cross-lane products again).
fn pcg3d(v_in: vec3<u32>) -> vec3<u32> {
    var v = v_in * 1664525u + vec3<u32>(1013904223u);
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    return v;
}

// A point in [0, 1)^3 for the (already wrapped, so non-negative) lattice cell `c`.
fn hash3(c: vec3<i32>, seed: u32) -> vec3<f32> {
    let s = vec3<u32>(seed * 747796405u, seed * 2891336453u, seed * 277803737u);
    let h = pcg3d(vec3<u32>(c) ^ s);
    return vec3<f32>(h >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

fn wrap3(c: vec3<i32>, period: i32) -> vec3<i32> {
    let p = vec3<i32>(period);
    return ((c % p) + p) % p;
}

fn gradient(c: vec3<i32>, period: i32, seed: u32) -> vec3<f32> {
    return normalize(hash3(wrap3(c, period), seed) * 2.0 - vec3<f32>(1.0) + vec3<f32>(1.0e-4));
}

// Tileable gradient (Perlin) noise, roughly -1..1, with `period` lattice cells per tile.
fn perlin(p: vec3<f32>, period: i32, seed: u32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = p - floor(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let n000 = dot(gradient(i, period, seed), f);
    let n100 = dot(gradient(i + vec3<i32>(1, 0, 0), period, seed), f - vec3<f32>(1.0, 0.0, 0.0));
    let n010 = dot(gradient(i + vec3<i32>(0, 1, 0), period, seed), f - vec3<f32>(0.0, 1.0, 0.0));
    let n110 = dot(gradient(i + vec3<i32>(1, 1, 0), period, seed), f - vec3<f32>(1.0, 1.0, 0.0));
    let n001 = dot(gradient(i + vec3<i32>(0, 0, 1), period, seed), f - vec3<f32>(0.0, 0.0, 1.0));
    let n101 = dot(gradient(i + vec3<i32>(1, 0, 1), period, seed), f - vec3<f32>(1.0, 0.0, 1.0));
    let n011 = dot(gradient(i + vec3<i32>(0, 1, 1), period, seed), f - vec3<f32>(0.0, 1.0, 1.0));
    let n111 = dot(gradient(i + vec3<i32>(1, 1, 1), period, seed), f - vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z) * 1.15;
}

// fBm of `perlin` for `uv` in [0, 1)^3: `octaves` octaves from `period` cells per tile, each
// octave twice the frequency (and twice the period, so it still tiles) at half the amplitude.
fn perlin_fbm(uv: vec3<f32>, period: i32, octaves: i32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var norm = 0.0;
    var cells = period;
    for (var o = 0; o < octaves; o = o + 1) {
        sum = sum + perlin(uv * f32(cells), cells, seed + u32(o) * 13u) * amp;
        norm = norm + amp;
        amp = amp * 0.5;
        cells = cells * 2;
    }
    return sum / norm;
}

// Tileable inverted Worley (cellular) noise for `uv` in [0, 1)^3: 1 at a feature point, falling
// to 0 a cell away. `cells` cells per tile.
fn worley(uv: vec3<f32>, cells: i32, seed: u32) -> f32 {
    let p = uv * f32(cells);
    let id = vec3<i32>(floor(p));
    let f = p - floor(p);
    var best = 1.0e9;
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let o = vec3<i32>(x, y, z);
                let feature = hash3(wrap3(id + o, cells), seed);
                let d = vec3<f32>(o) + feature - f;
                best = min(best, dot(d, d));
            }
        }
    }
    return 1.0 - clamp(sqrt(best), 0.0, 1.0);
}

// Three octaves of `worley`, weighted 0.625, 0.25 and 0.125.
fn worley_fbm(uv: vec3<f32>, cells: i32, seed: u32) -> f32 {
    return worley(uv, cells, seed) * 0.625
        + worley(uv, cells * 2, seed + 101u) * 0.25
        + worley(uv, cells * 4, seed + 202u) * 0.125;
}

// The raw fBms are narrow (Perlin fBm: about 0 +- 0.13; Worley fBm: about 0.48 +- 0.12, measured
// over 20k samples with a CPU port of these functions). Everything stored is stretched to use the
// whole 0..1 range, so the trace's coverage threshold can carve distinct clouds out of it.
fn perlin01(v: f32) -> f32 {
    return clamp(v / 0.27 * 0.5 + 0.5, 0.0, 1.0);
}

fn worley01(v: f32) -> f32 {
    return clamp((v - 0.18) / 0.62, 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn cs_shape(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(shape_out);
    if (any(id >= size)) {
        return;
    }
    let uv = (vec3<f32>(id) + vec3<f32>(0.5)) / vec3<f32>(size);
    let w4 = worley01(worley_fbm(uv, 4, 11u));
    // Low-frequency Perlin fBm "dilated" by the Worley fBm: connected masses with round,
    // cauliflower-like bulges (Perlin-Worley noise). Re-centred on
    // 0.5 with a 0.2 spread (it measured 0.743 +- 0.161).
    let p = perlin01(perlin_fbm(uv, 4, 5, 3u));
    let perlin_worley = w4 + p * (1.0 - w4);
    let shape = clamp((perlin_worley - 0.743) / 0.161 * 0.2 + 0.5, 0.0, 1.0);
    let w8 = worley01(worley_fbm(uv, 8, 23u));
    let w16 = worley01(worley_fbm(uv, 16, 37u));
    textureStore(shape_out, vec3<i32>(id), vec4<f32>(shape, w4, w8, w16));
}

@compute @workgroup_size(4, 4, 4)
fn cs_detail(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(detail_out);
    if (any(id >= size)) {
        return;
    }
    let uv = (vec3<f32>(id) + vec3<f32>(0.5)) / vec3<f32>(size);
    let r = worley01(worley_fbm(uv, 2, 51u));
    let g = worley01(worley_fbm(uv, 4, 61u));
    let b = worley01(worley_fbm(uv, 8, 71u));
    textureStore(detail_out, vec3<i32>(id), vec4<f32>(r, g, b, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_weather(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(weather_out);
    if (any(id.xy >= size)) {
        return;
    }
    // A z slice through tileable 3D noise tiles in x and y.
    let uv = vec3<f32>((vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(size), 0.37);
    // Coverage: broad cloud fields (Perlin fBm) broken up into separate cells (Worley), so the
    // sky has both clusters of cloud and open blue between them. Re-centred on 0.5 with a 0.2
    // spread (it measured 0.463 +- 0.090).
    let fields = clamp(perlin_fbm(uv, 3, 5, 83u) * 0.75 + 0.5, 0.0, 1.0);
    let cells = worley_fbm(uv, 6, 89u);
    let raw = clamp(fields * 0.7 + cells * 0.45 - 0.08, 0.0, 1.0);
    let coverage = clamp((raw - 0.463) / 0.09 * 0.2 + 0.5, 0.0, 1.0);
    // Cloud type drifts slowly across the map: flat stratus in places, towering cumulus in most.
    let kind = clamp(perlin01(perlin_fbm(uv, 2, 3, 97u)) + 0.1, 0.0, 1.0);
    let fine = worley01(worley_fbm(uv, 16, 103u));
    textureStore(weather_out, vec2<i32>(id.xy), vec4<f32>(coverage, kind, fine, 1.0));
}
