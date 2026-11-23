// Screen-space light rays. Quarter resolution, three passes:
//
// * `fs_mask`: how much sky this pixel shows (4 depth taps) times a glow around the light's
//   screen position;
// * `fs_blur_coarse` then `fs_blur_fine`: a radial blur toward the light. The coarse pass averages
//   `TAPS` samples spread over `reach` of the way to the light, the fine pass `TAPS` samples over
//   one coarse step, so the pair is `TAPS^2` evenly spaced taps with no banding.
//
// The result is the average sky-glow along each pixel's line to the light: bright where the light
// streams past occluders, dark in their shadow. `godrays.rs` composites nothing itself; a pack adds
// `rays * colour` to its frame (`dreamy.wgsl` does, in linear light).

struct RaysUniform {
    // xy = the light's position in UV (may be off screen), z = glow radius (UV height units),
    // w = aspect (width / height).
    sun: vec4<f32>,
    // x = reach (0..1 of the way to the light), y = decay per coarse tap, z = jitter frame,
    // w = sun-disc boost.
    params: vec4<f32>,
    // xy = this target's size, zw = the full-resolution size.
    size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> rays: RaysUniform;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var src_tex: texture_2d<f32>;
@group(0) @binding(3) var src_sampler: sampler;

const TAPS: i32 = 16;

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fs_mask(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let full = vec2<i32>(rays.size.zw);
    let scale = rays.size.zw / rays.size.xy;
    let base = pos.xy * scale;
    var sky = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let o = vec2<f32>(f32(k & 1), f32(k >> 1)) * 0.5 - vec2<f32>(0.25);
        let c = clamp(vec2<i32>(base + o * scale), vec2<i32>(0), full - vec2<i32>(1));
        sky = sky + select(0.0, 0.25, textureLoad(depth_tex, c, 0) >= 1.0);
    }
    let uv = pos.xy / rays.size.xy;
    let d = (uv - rays.sun.xy) * vec2<f32>(rays.sun.w, 1.0);
    let r = length(d) / max(rays.sun.z, 1.0e-4);
    let glow = exp(-r * r) + rays.params.w * exp(-r * r * 40.0);
    return vec4<f32>(sky * glow, 0.0, 0.0, 1.0);
}

fn radial(uv: vec2<f32>, step: vec2<f32>, decay: f32, jitter: f32) -> f32 {
    var sum = 0.0;
    var weights = 0.0;
    var w = 1.0;
    for (var i = 0; i < TAPS; i = i + 1) {
        let p = uv + step * (f32(i) + jitter);
        sum = sum + textureSampleLevel(src_tex, src_sampler, p, 0.0).r * w;
        weights = weights + w;
        w = w * decay;
    }
    return sum / weights;
}

fn coarse_step(uv: vec2<f32>) -> vec2<f32> {
    return (rays.sun.xy - uv) * rays.params.x / f32(TAPS);
}

@fragment
fn fs_blur_coarse(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / rays.size.xy;
    let jitter = ign(pos.xy + vec2<f32>(5.588238 * rays.params.z));
    return vec4<f32>(radial(uv, coarse_step(uv), rays.params.y, jitter), 0.0, 0.0, 1.0);
}

@fragment
fn fs_blur_fine(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / rays.size.xy;
    let jitter = ign(pos.yx + vec2<f32>(3.1 + 5.588238 * rays.params.z));
    return vec4<f32>(radial(uv, coarse_step(uv) / f32(TAPS), 1.0, jitter), 0.0, 0.0, 1.0);
}
