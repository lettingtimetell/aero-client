// ---------------------------------------------------------------------------------------- finish

struct FinishUniform {
    // x = glow strength, y = 1 / levels, z = halation strength, w = halation threshold (linear).
    glow: vec4<f32>,
    // rgb = halation colour (linear), w unused.
    halation: vec4<f32>,
    // x = vignette, y = grain, z = chromatic aberration (UV units at the corner), w = grain seed.
    lens: vec4<f32>,
    // x = glow on, y = LUT on, z = dither on, w = aspect (width / height).
    flags: vec4<f32>,
    // xy = frame size, zw = 1 / frame size.
    size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> fin: FinishUniform;
@group(0) @binding(1) var fin_scene: texture_2d<f32>;
@group(0) @binding(2) var fin_glow: texture_2d<f32>;
@group(0) @binding(3) var fin_sampler: sampler;
@group(0) @binding(4) var fin_lut: texture_3d<f32>;
@group(0) @binding(5) var fin_lut_sampler: sampler;

@fragment
fn fs_finish(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(pos.xy);
    let uv = pos.xy * fin.size.zw;
    let src = textureLoad(fin_scene, px, 0);
    var c = src.rgb;
    if (fin.lens.z > 0.0) {
        let off = (uv - vec2<f32>(0.5)) * fin.lens.z;
        c.r = textureSampleLevel(fin_scene, fin_sampler, uv - off, 0.0).r;
        c.b = textureSampleLevel(fin_scene, fin_sampler, uv + off, 0.0).b;
    }
    if (fin.flags.x > 0.5) {
        let glow = textureSampleLevel(fin_glow, fin_sampler, uv, 0.0).rgb * fin.glow.y;
        // Orton: a screen blend of the soft copy. The sharp image stays underneath.
        let soft = linear_to_srgb(glow) * fin.glow.x;
        c = vec3<f32>(1.0) - (vec3<f32>(1.0) - c) * (vec3<f32>(1.0) - soft);
        // Halation: highlights bleed a warm fringe into their surroundings.
        let excess = max(dot(glow, LUMA) - fin.glow.w, 0.0);
        let fringe = linear_to_srgb(fin.halation.rgb * excess) * fin.glow.z;
        c = vec3<f32>(1.0) - (vec3<f32>(1.0) - c) * (vec3<f32>(1.0) - fringe);
    }
    if (fin.flags.y > 0.5) {
        c = grade_lut_apply(fin_lut, fin_lut_sampler, c);
    }
    if (fin.lens.x > 0.0) {
        let q = (uv - vec2<f32>(0.5)) * vec2<f32>(fin.flags.w, 1.0);
        let r = length(q) / length(vec2<f32>(fin.flags.w, 1.0) * 0.5);
        c = c * (1.0 - fin.lens.x * smoothstep(0.35, 1.05, r));
    }
    if (fin.lens.y > 0.0) {
        // Film grain: zero-mean, strongest in the mids, gone in pure black and white.
        let seed = u32(fin.lens.w);
        let n = hash2(px, seed) + hash2(px + vec2<i32>(37, 91), seed) - 1.0;
        let y = dot(c, LUMA);
        c = c + vec3<f32>(n * fin.lens.y * 4.0 * y * (1.0 - y));
    }
    if (fin.flags.z > 0.5) {
        let n = hash2(px, 7u) - hash2(px + vec2<i32>(13, 57), 7u);
        c = c + vec3<f32>(n / 255.0);
    }
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), src.a);
}
