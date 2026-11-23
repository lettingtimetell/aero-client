// The scene composite: the
// scene's finished frame laid over the game's frame. The title screen draws it at opacity 1 as its
// background (the Glass and Shore menu scenes); `brightness` multiplies the scene.
//
// `params.w` mirrors the scene left to right, and `style.x` darkens the left
// of the frame where the title screen's logo and buttons sit (the greenhouse's own scrim, the same
// shape), so white type keeps its contrast. Both are off (0) by default, and then the output is
// exactly the original curtain's.

struct Curtain {
    // x = opacity (0..1), y = brightness (0..1), z = 1 when the target is an sRGB format (the
    // hardware encodes again, so decode first), w = 1 to mirror the scene horizontally.
    params: vec4<f32>,
    // x = the left scrim's strength (0 = none), yzw = unused.
    style: vec4<f32>,
};

@group(0) @binding(0) var<uniform> curtain: Curtain;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VsOut;
    out.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fn decode_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// The greenhouse's scrim (`menu_scene.wgsl`): the left half, a little more at the top, deepest
// behind the title (left aligned, an eighth of the way down). 0 = untouched, up to 1.8.
fn scrim_shape(uv: vec2<f32>) -> f32 {
    let side = (1.0 - smoothstep(0.0, 0.5, uv.x)) * (0.7 + 0.3 * (1.0 - uv.y));
    let tz = (uv - vec2<f32>(0.2, 0.125)) / vec2<f32>(0.3, 0.12);
    let title = exp(-dot(tz, tz)) * 1.1;
    return min(side + title, 1.8);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(scene_tex));
    var p = clamp(vec2<i32>(in.pos.xy), vec2<i32>(0), size - 1);
    if (curtain.params.w > 0.5) {
        p.x = size.x - 1 - p.x;
    }
    var c = textureLoad(scene_tex, p, 0).rgb * curtain.params.y;
    if (curtain.style.x > 0.0) {
        // In display space (the scene's own encoding), so the same strength reads the same on
        // both kinds of target.
        let uv = in.pos.xy / vec2<f32>(size);
        c = c * (1.0 - curtain.style.x * scrim_shape(uv));
    }
    if (curtain.params.z > 0.5) {
        c = decode_srgb(c);
    }
    let a = curtain.params.x;
    // Premultiplied "over": the pipeline blends One, OneMinusSrcAlpha.
    return vec4<f32>(c * a, a);
}
