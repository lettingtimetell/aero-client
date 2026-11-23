// TAA's optional sharpening. See `taa.rs`.
//
// Reads the resolve this frame just wrote to the history and draws the frame from it with a
// small unsharp mask, clamped to the 4-neighbour range so it never draws halos. The history
// itself stays unsharpened, so the sharpening never compounds.

struct TaaParams {
    // y: sharpness (0..1). The rest is the resolve's.
    params: vec4<f32>,
};

// The resolve: rgb = colour, a = depth (not colour).
@group(0) @binding(0) var resolved: texture_2d<f32>;
@group(0) @binding(1) var<uniform> p: TaaParams;
// The frame before the resolve, for its alpha.
@group(0) @binding(2) var scene: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fs_sharpen(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ip = vec2<i32>(pos.xy);
    let last = vec2<i32>(textureDimensions(resolved)) - vec2<i32>(1);
    let c = textureLoad(resolved, ip, 0).rgb;
    let n = textureLoad(resolved, clamp(ip + vec2<i32>(0, -1), vec2<i32>(0), last), 0).rgb;
    let s = textureLoad(resolved, clamp(ip + vec2<i32>(0, 1), vec2<i32>(0), last), 0).rgb;
    let w = textureLoad(resolved, clamp(ip + vec2<i32>(-1, 0), vec2<i32>(0), last), 0).rgb;
    let e = textureLoad(resolved, clamp(ip + vec2<i32>(1, 0), vec2<i32>(0), last), 0).rgb;
    let blurred = (n + s + w + e) * 0.25;
    let sharpened = c + (c - blurred) * (p.params.y * 1.5);
    let lo = min(c, min(min(n, s), min(w, e)));
    let hi = max(c, max(max(n, s), max(w, e)));
    return vec4<f32>(clamp(sharpened, lo, hi), textureLoad(scene, ip, 0).a);
}
