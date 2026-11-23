// ------------------------------------------------------------------------------------ atmosphere

struct AtmosphereUniform {
    // Camera-relative clip space to camera-relative world space (jittered with the depth buffer).
    inv_view_proj: mat4x4<f32>,
    // rgb = haze colour (linear), w = haze density per block (0 = off).
    haze: vec4<f32>,
    // x = haze start distance, y = fog on, z = rays on, w = sky-tint horizon bias.
    switches: vec4<f32>,
    // rgb = sky-tint multiplier (linear, luminance 1), w = strength.
    sky_tint: vec4<f32>,
    // xyz = direction toward the light, w = haze glow sharpness (Henyey-Greenstein g).
    light: vec4<f32>,
    // rgb = the light's glow in the haze (linear), w unused.
    light_glow: vec4<f32>,
    // rgb = light-ray colour (linear) times strength and visibility, w unused.
    rays: vec4<f32>,
    // xy = fog target size, z = fog divisor, w = the sky's distance sentinel.
    fog_size: vec4<f32>,
    // xy = frame size, zw = 1 / frame size.
    size: vec4<f32>,
    // xyz = the true eye in camera-relative space (orientCamera's 0.1 block, F5's orbit).
    eye_rel: vec4<f32>,
};

@group(0) @binding(0) var<uniform> atmo: AtmosphereUniform;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var fog_tex: texture_2d<f32>;
@group(0) @binding(4) var fog_dist_tex: texture_2d<f32>;
@group(0) @binding(5) var rays_tex: texture_2d<f32>;
@group(0) @binding(6) var atmo_sampler: sampler;

// Joint bilateral upsample of the low-resolution fog: bilinear weights times a depth-similarity
// weight, so fog computed for a far surface never lands on a near one (a player in front of the
// fog sea keeps a clean silhouette). Falls back to the best depth match when nothing matches.
fn fog_upsample(p: vec2<f32>, dist: f32) -> vec4<f32> {
    let div = atmo.fog_size.z;
    let low = vec2<i32>(atmo.fog_size.xy);
    let lp = p / div - vec2<f32>(0.5);
    let base = floor(lp);
    let f = lp - base;
    var sum = vec4<f32>(0.0);
    var weights = 0.0;
    var best = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var best_err = 1.0e30;
    for (var k = 0; k < 4; k = k + 1) {
        let o = vec2<f32>(f32(k & 1), f32(k >> 1));
        let c = clamp(vec2<i32>(base + o), vec2<i32>(0), low - vec2<i32>(1));
        let bil = mix(1.0 - f.x, f.x, o.x) * mix(1.0 - f.y, f.y, o.y);
        let d = textureLoad(fog_dist_tex, c, 0).r;
        let err = abs(d - dist) / max(dist, 1.0);
        let w = bil * exp(-(err * err) * 400.0);
        let value = textureLoad(fog_tex, c, 0);
        sum = sum + value * w;
        weights = weights + w;
        if (err < best_err) {
            best_err = err;
            best = value;
        }
    }
    if (weights < 1.0e-3) {
        return best;
    }
    return sum / weights;
}

@fragment
fn fs_atmosphere(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(pos.xy);
    let src = textureLoad(scene_tex, px, 0);
    var lin = srgb_to_linear(src.rgb);
    let d = textureLoad(depth_tex, px, 0);
    let sky = d >= 1.0;
    let uv = pos.xy * atmo.size.zw;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let world = atmo.inv_view_proj * vec4<f32>(ndc, min(d, 1.0), 1.0);
    let rel = world.xyz / world.w - atmo.eye_rel.xyz;
    let rel_len = max(length(rel), 1.0e-4);
    let dir = rel / rel_len;
    let dist = select(rel_len, atmo.fog_size.w, sky);

    if (sky) {
        // The sky tint, strongest toward the horizon and kept off the zenith.
        let horizon = 1.0 - atmo.switches.w * clamp(abs(dir.y), 0.0, 1.0);
        lin = mix(lin, lin * atmo.sky_tint.rgb, atmo.sky_tint.w * horizon);
    } else if (atmo.haze.w > 0.0) {
        let f = 1.0 - exp(-max(dist - atmo.switches.x, 0.0) * atmo.haze.w);
        let g = atmo.light.w;
        let cos_theta = dot(dir, atmo.light.xyz);
        let hg = (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1.0e-4), 1.5);
        let colour = atmo.haze.rgb + atmo.light_glow.rgb * hg;
        lin = mix(lin, colour, f);
    }
    if (atmo.switches.y > 0.5) {
        let fog = fog_upsample(pos.xy, dist);
        lin = lin * fog.a + fog.rgb;
    }
    if (atmo.switches.z > 0.5) {
        // Shafts are light scattered by the air in front of a surface, so they grow with the
        // distance to it: the sky and far terrain get them in full, a player at fighting range
        // (5 blocks: 19 %) keeps nearly its own colours.
        let r = textureSampleLevel(rays_tex, atmo_sampler, uv, 0.0).r;
        let air = 1.0 - exp(-dist / 24.0);
        lin = lin + atmo.rays.rgb * r * air;
    }
    return vec4<f32>(linear_to_srgb(lin), src.a);
}

