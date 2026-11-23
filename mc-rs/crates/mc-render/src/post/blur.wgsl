// Menu blur: a dual-Kawase blur, written for mc-rs. See
// `blur.rs`.
//
// Every pass draws one fullscreen triangle into a target of its own size:
//   fs_down       level k-1 -> level k (half the size): five bilinear taps
//   fs_up         level k+1 -> level k (twice the size): eight bilinear taps
//   fs_composite  level 1   -> the frame: the last upsample, mixed with the sharp scene
// The chain's levels are Rgba16Float, so ten-odd passes add no visible banding.

struct BlurPass {
    // xy: one texel of this pass's *source* in UV units (1 / source size).
    // zw: 1 / this pass's target size, to turn the fragment position into a UV.
    texel: vec4<f32>,
    // x: tap distance in source texels (the blur's reach scales with it).
    // y: composite only: how much of the blur shows, 0 (sharp scene) to 1 (fade-in).
    control: vec4<f32>,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var<uniform> p: BlurPass;
// The composite's sharp input: the frame so far, read 1:1. Only `fs_composite` uses it.
@group(0) @binding(3) var scene: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src, lin, uv, 0.0).rgb;
}

// A target pixel's centre lands on the corner shared by four source texels, so the centre tap
// is a 2x2 box; the four diagonal taps are the 2x2 boxes `offset` source texels away.
@fragment
fn fs_down(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy * p.texel.zw;
    let d = p.texel.xy * p.control.x;
    var sum = tap(uv) * 4.0;
    sum = sum + tap(uv + vec2<f32>(-d.x, -d.y));
    sum = sum + tap(uv + vec2<f32>(d.x, -d.y));
    sum = sum + tap(uv + vec2<f32>(-d.x, d.y));
    sum = sum + tap(uv + vec2<f32>(d.x, d.y));
    return vec4<f32>(sum * 0.125, 1.0);
}

// A tent of eight taps around the target pixel: the four edge taps `offset` source texels out
// (weight 1) and the four diagonal taps half as far (weight 2).
fn up(uv: vec2<f32>) -> vec3<f32> {
    let d = p.texel.xy * p.control.x;
    let h = d * 0.5;
    var sum = tap(uv + vec2<f32>(-d.x, 0.0));
    sum = sum + tap(uv + vec2<f32>(d.x, 0.0));
    sum = sum + tap(uv + vec2<f32>(0.0, -d.y));
    sum = sum + tap(uv + vec2<f32>(0.0, d.y));
    sum = sum + tap(uv + vec2<f32>(-h.x, -h.y)) * 2.0;
    sum = sum + tap(uv + vec2<f32>(h.x, -h.y)) * 2.0;
    sum = sum + tap(uv + vec2<f32>(-h.x, h.y)) * 2.0;
    sum = sum + tap(uv + vec2<f32>(h.x, h.y)) * 2.0;
    return sum / 12.0;
}

@fragment
fn fs_up(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(up(pos.xy * p.texel.zw), 1.0);
}

// The frame's alpha is kept as it was, so a later pass that blends against destination alpha
// sees exactly what it would have seen without the blur.
@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let sharp = textureLoad(scene, vec2<i32>(pos.xy), 0);
    let blurred = up(pos.xy * p.texel.zw);
    return vec4<f32>(mix(sharp.rgb, blurred, p.control.y), sharp.a);
}
