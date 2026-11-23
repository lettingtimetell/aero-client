// Per-pixel motion blur along the motion target's exposure motion. See `velocity_blur.rs`.

struct BlurParams {
    // x: the longest trail, in pixels. yzw: unused.
    params: vec4<f32>,
    // width, height, 1 / width, 1 / height.
    screen: vec4<f32>,
};

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var velocity: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<uniform> p: BlurParams;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

// Interleaved gradient noise: a per-pixel offset that turns sample banding into fine grain.
fn noise(pos: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pos, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fs_blur(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ip = vec2<i32>(pos.xy);
    let centre = textureLoad(scene, ip, 0);
    // ba: where the surface is now minus where it was at the start of the exposure (UV).
    var trail = textureLoad(velocity, ip, 0).ba * p.screen.xy;
    let len = length(trail);
    if (len < 0.5) {
        return centre; // under half a pixel: nothing to see, and most pixels of a still frame
    }
    if (len > p.params.x) {
        trail *= p.params.x / len;
    }
    let reach = min(len, p.params.x);
    // About one sample per 2 pixels of trail, 3 to 16.
    let count = clamp(i32(reach * 0.5) + 2, 3, 16);
    let step = trail * p.screen.zw;
    let uv = (vec2<f32>(ip) + vec2<f32>(0.5)) * p.screen.zw;
    let offset = noise(pos.xy) - 0.5;
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var i = 0; i < count; i++) {
        // Centred on the pixel: from half the trail behind to half ahead, tent-weighted so the
        // trail fades at its ends (like the frame blend's exponential tail) instead of ending
        // in a hard-edged smear.
        let t = (f32(i) + 0.5 + offset) / f32(count) - 0.5;
        let w = 1.0 - 2.0 * abs(t) + 0.05;
        sum += textureSampleLevel(scene, lin, uv - step * t, 0.0).rgb * w;
        total += w;
    }
    return vec4<f32>(sum / total, centre.a);
}
