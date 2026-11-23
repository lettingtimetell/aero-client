// ------------------------------------------------------------------------------------------ glow

struct GlowUniform {
    // x = the weight of a dark pixel (0..1), y/z = luma knee where the weight reaches 1.
    knee: vec4<f32>,
};

@group(0) @binding(0) var glow_src: texture_2d<f32>;
@group(0) @binding(1) var glow_sampler: sampler;
@group(0) @binding(2) var<uniform> glow_params: GlowUniform;

// One bilinear tap of the pyramid's source level, `o` source texels (`texel` in UV) from `uv`.
fn glow_tap(uv: vec2<f32>, o: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(glow_src, glow_sampler, uv + o * texel, 0.0).rgb;
}

// The signs of diagonal `k` (0..3): (-, -), (+, -), (-, +), (+, +).
fn glow_diagonal(k: i32) -> vec2<f32> {
    return vec2<f32>(select(-1.0, 1.0, (k & 1) == 1), select(-1.0, 1.0, (k & 2) == 2));
}

// The pyramid's downsample: the centre tap four times and the four taps one source texel out on
// the diagonals once each, over eight.
fn down5(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    var s = glow_tap(uv, vec2<f32>(0.0), texel) * 4.0;
    // The four diagonals written out, not looped: every read is issued before the sums need one.
    s = s + glow_tap(uv, glow_diagonal(0), texel);
    s = s + glow_tap(uv, glow_diagonal(1), texel);
    s = s + glow_tap(uv, glow_diagonal(2), texel);
    s = s + glow_tap(uv, glow_diagonal(3), texel);
    return s / 8.0;
}

// The axis tap that goes with diagonal `k` in the upsample: -x, +x (k = 0, 1), then -y, +y
// (k = 2, 3), one source texel out.
fn glow_axis(k: i32) -> vec2<f32> {
    let d = glow_diagonal(k);
    return select(vec2<f32>(0.0, d.x), vec2<f32>(d.x, 0.0), k < 2);
}

fn weighted(c_display: vec3<f32>) -> vec3<f32> {
    let lin = srgb_to_linear(c_display);
    let w = mix(glow_params.knee.x, 1.0,
        smoothstep(glow_params.knee.y, glow_params.knee.z, dot(lin, LUMA)));
    return lin * w;
}

// Full resolution (display-encoded) to half (linear, weighted toward highlights).
@fragment
fn fs_glow_prefilter(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let src_size = vec2<f32>(textureDimensions(glow_src));
    let texel = 1.0 / src_size;
    let uv = pos.xy * 2.0 * texel;
    // `down5`'s taps and weights, each tap weighted toward highlights first.
    var s = weighted(glow_tap(uv, vec2<f32>(0.0), texel)) * 4.0;
    s = s + weighted(glow_tap(uv, glow_diagonal(0), texel));
    s = s + weighted(glow_tap(uv, glow_diagonal(1), texel));
    s = s + weighted(glow_tap(uv, glow_diagonal(2), texel));
    s = s + weighted(glow_tap(uv, glow_diagonal(3), texel));
    return vec4<f32>(s / 8.0, 1.0);
}

@fragment
fn fs_glow_down(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let src_size = vec2<f32>(textureDimensions(glow_src));
    let texel = 1.0 / src_size;
    return vec4<f32>(down5(pos.xy * 2.0 * texel, texel), 1.0);
}

// The level below, tent-filtered up; blended additively onto this level's own downsample.
@fragment
fn fs_glow_up(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let src_size = vec2<f32>(textureDimensions(glow_src));
    let texel = 1.0 / src_size;
    let uv = pos.xy * 0.5 * texel;
    // A tent of eight taps: half a source texel out on each diagonal (weight 2) and one texel out
    // along each axis (weight 1), written out, not looped: each diagonal, then its axis tap.
    var s = vec3<f32>(0.0);
    s = s + glow_tap(uv, glow_diagonal(0) * 0.5, texel) * 2.0;
    s = s + glow_tap(uv, glow_axis(0), texel);
    s = s + glow_tap(uv, glow_diagonal(1) * 0.5, texel) * 2.0;
    s = s + glow_tap(uv, glow_axis(1), texel);
    s = s + glow_tap(uv, glow_diagonal(2) * 0.5, texel) * 2.0;
    s = s + glow_tap(uv, glow_axis(2), texel);
    s = s + glow_tap(uv, glow_diagonal(3) * 0.5, texel) * 2.0;
    s = s + glow_tap(uv, glow_axis(3), texel);
    return vec4<f32>(s / 12.0, 1.0);
}

