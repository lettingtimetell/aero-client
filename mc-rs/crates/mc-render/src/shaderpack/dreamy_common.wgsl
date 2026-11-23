// The Dreamy pack's composite passes. `dreamy.rs` documents the
// frame; `grade.rs`'s `LUT_APPLY_WGSL` is appended to this source at pipeline build time.
//
// * `fs_atmosphere` (before TAA): distance haze with a sun glow, the sky tint, the ground fog's
//   bilateral upsample (`colour * T + L`) and the light rays, in linear light.
// * `fs_glow_prefilter`, `fs_glow_down`, `fs_glow_up`: a dual-filter blur pyramid
//   of the frame, weighted toward highlights; the up passes add each level onto the next one up.
// * `fs_finish` (at POST): chromatic aberration (off by default), the Orton glow and halation,
//   the colour LUT, the vignette, film grain and a one-level dither against banding.

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash2(p: vec2<i32>, seed: u32) -> f32 {
    let h = hash_u32(u32(p.x) * 1973u + u32(p.y) * 9277u + seed * 26699u);
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

