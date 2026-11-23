// Ground fog: height fog from the depth buffer, at half or quarter
// resolution, with animated 3D noise and light from the sun or moon. `ground_fog.rs` documents the
// model; the composite (a bilateral upsample) is in `dreamy.wgsl`.
//
// Entry points:
// * `cs_noise`: bakes the 64^3 tileable noise once (R = billows, G = wisps), the first time fog is
//   drawn.
// * `vs_fullscreen` + `fs_march`: one low-resolution pixel's ray through the fog layer. Writes the
//   in-scattered light (linear, premultiplied) and the transmittance, plus the distance of the
//   surface the ray was marched to, which the upsample uses to keep fog off nearer surfaces.

// ------------------------------------------------------------------------------------ noise bake

@group(0) @binding(0) var noise_out: texture_storage_3d<rgba8unorm, write>;

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

fn lattice_gradient(c: vec3<i32>, period: i32, seed: u32) -> vec3<f32> {
    let p = vec3<i32>(period);
    let w = ((c % p) + p) % p;
    let h = pcg3d(vec3<u32>(w) ^ vec3<u32>(seed * 747796405u, seed * 2891336453u, seed * 277803737u));
    let r = vec3<f32>(h >> vec3<u32>(8u)) * (1.0 / 16777216.0);
    return normalize(r * 2.0 - vec3<f32>(1.0) + vec3<f32>(1.0e-4));
}

// Tileable gradient noise with `period` lattice cells per tile, roughly -1..1.
fn perlin(p: vec3<f32>, period: i32, seed: u32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = p - floor(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var n: array<f32, 8>;
    for (var k = 0; k < 8; k = k + 1) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        n[k] = dot(lattice_gradient(i + o, period, seed), f - vec3<f32>(o));
    }
    let x00 = mix(n[0], n[1], u.x);
    let x10 = mix(n[2], n[3], u.x);
    let x01 = mix(n[4], n[5], u.x);
    let x11 = mix(n[6], n[7], u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn fbm(uvw: vec3<f32>, base_period: i32, octaves: i32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    var period = base_period;
    for (var o = 0; o < octaves; o = o + 1) {
        sum = sum + amp * perlin(uvw * f32(period), period, seed + u32(o) * 17u);
        norm = norm + amp;
        amp = amp * 0.5;
        period = period * 2;
    }
    return sum / norm;
}

@compute @workgroup_size(4, 4, 4)
fn cs_noise(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = textureDimensions(noise_out);
    if (id.x >= n.x || id.y >= n.y || id.z >= n.z) {
        return;
    }
    let uvw = (vec3<f32>(id) + vec3<f32>(0.5)) / vec3<f32>(n);
    let billows = fbm(uvw, 4, 4, 1u);
    let wisps = fbm(uvw, 8, 3, 7u);
    let out = clamp(vec2<f32>(billows, wisps) * 1.6 * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0),
        vec2<f32>(1.0));
    textureStore(noise_out, id, vec4<f32>(out, 0.0, 1.0));
}

// ------------------------------------------------------------------------------------- the march

struct FogUniform {
    // Camera-relative clip space to camera-relative world space (jittered with the depth buffer).
    inv_view_proj: mat4x4<f32>,
    // x = eye height (world y), yzw = the eye in noise space (blocks, wrapped to the tile).
    eye: vec4<f32>,
    // x = layer top (world y), y = softness above the top (blocks), z = density (per block),
    // w = how far the noise moves the top (blocks).
    layer: vec4<f32>,
    // x = start distance, y = farthest march (blocks), z = noise scale (tiles per block),
    // w = wisp amount (0..1).
    shape: vec4<f32>,
    // xyz = wind offset in noise space (blocks), w = steps.
    wind: vec4<f32>,
    // xyz = direction toward the light (sun or moon), w = Henyey-Greenstein g.
    light_dir: vec4<f32>,
    // rgb = light colour (linear) times its forward-scatter strength, w unused.
    light_color: vec4<f32>,
    // rgb = ambient fog colour (linear), w = how much darker the fog gets deep inside the layer.
    ambient: vec4<f32>,
    // xy = this target's size, zw = the full-resolution size.
    size: vec4<f32>,
    // x = jitter frame (0 = a fixed pattern), y = full / low scale, z = sky distance sentinel,
    // w = how much of a custom sky to keep in view below the horizon, 0..1 (see `fs_march`).
    misc: vec4<f32>,
    // xyz = the true eye in camera-relative space. Not the origin: the view matrix carries
    // orientCamera's 0.1-block step back and, in F5, the whole orbit offset.
    eye_rel: vec4<f32>,
};

@group(0) @binding(0) var<uniform> fog: FogUniform;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var noise_tex: texture_3d<f32>;
@group(0) @binding(3) var noise_sampler: sampler;

struct MarchOut {
    @location(0) fog: vec4<f32>,
    @location(1) dist: vec4<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

// Interleaved gradient noise: a cheap, well-spread value in [0, 1) per pixel.
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// The density profile's shape at height y: 1 below the top `h`, fading as (1 - u)^3 over the
// softness `s` above it (u = (y - h) / s), and exactly 0 from h + s up. Compact support is the PvP
// guarantee: nothing above h + s is ever fogged, however long the ray.
fn profile(y: f32, h: f32, s: f32) -> f32 {
    let u = clamp((y - h) / s, 0.0, 1.0);
    let k = 1.0 - u;
    return k * k * k;
}

// Optical depth of one segment (from height y0, rising v per block, len blocks) through
// `density * profile(y)`. Exact: integrated over y, the part below h is its height extent and the
// fading part is s/4 * ((1-u0)^4 - (1-u1)^4); divided by |v| for the length along the ray.
fn segment_tau(y0: f32, v: f32, len: f32, h: f32, density: f32, s: f32) -> f32 {
    if (abs(v) * len < 1.0e-4) {
        return density * len * profile(y0 + 0.5 * v * len, h, s);
    }
    let y1 = y0 + v * len;
    let lo = min(y0, y1);
    let hi = max(y0, y1);
    let below = max(min(hi, h) - lo, 0.0);
    let u0 = clamp((lo - h) / s, 0.0, 1.0);
    let u1 = clamp((hi - h) / s, 0.0, 1.0);
    let a = 1.0 - u0;
    let b = 1.0 - u1;
    let fading = 0.25 * s * (a * a * a * a - b * b * b * b);
    return density * (below + fading) / abs(v);
}

@fragment
fn fs_march(@builtin(position) pos: vec4<f32>) -> MarchOut {
    let low = vec2<i32>(pos.xy);
    let scale = i32(fog.misc.y);
    let full = vec2<i32>(fog.size.zw);

    // Checkerboard min/max depth of the block's centre 2x2: neighbouring low-res pixels
    // keep both the nearest and the farthest surface, so the upsample finds a match for each.
    let centre = low * scale + vec2<i32>(max(scale / 2 - 1, 0));
    var pick = centre;
    var pick_d = textureLoad(depth_tex, min(centre, full - vec2<i32>(1)), 0);
    let want_near = ((low.x + low.y) & 1) == 0;
    for (var k = 1; k < 4; k = k + 1) {
        let c = min(centre + vec2<i32>(k & 1, k >> 1), full - vec2<i32>(1));
        let d = textureLoad(depth_tex, c, 0);
        if ((want_near && d < pick_d) || (!want_near && d > pick_d)) {
            pick = c;
            pick_d = d;
        }
    }

    let uv = (vec2<f32>(pick) + vec2<f32>(0.5)) / fog.size.zw;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let sky = pick_d >= 1.0;
    // The sky reconstructs on the (finite) far plane, so its direction is exact; every ray starts
    // at the true eye, not at the camera-relative origin.
    let world = fog.inv_view_proj * vec4<f32>(ndc, min(pick_d, 1.0), 1.0);
    let rel = world.xyz / world.w - fog.eye_rel.xyz;
    let rel_len = max(length(rel), 1.0e-4);
    let dir = rel / rel_len;
    let surface = select(rel_len, fog.misc.z, sky);

    var out: MarchOut;
    out.fog = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.dist = vec4<f32>(surface, 0.0, 0.0, 0.0);

    let eye_y = fog.eye.x;
    let top = fog.layer.x;
    let softness = max(fog.layer.y, 0.25);
    let density = fog.layer.z;
    let amp = fog.layer.w;
    let start = fog.shape.x;
    // Above this height there is no fog at all: nothing to march.
    let y_max = top + 0.5 * amp + softness;

    var t0 = 0.5 * start;
    var t1 = min(surface, fog.shape.y);
    // The share of the fog a texel takes: all of it, except the sky under a custom sky. That sky
    // is a picture of its own below the horizon, so from above the fog it stays as drawn, as far
    // as its layers cover it (misc.w: as they fade out, the fog sea fades back in); the fog sea
    // comes back over the first 4 blocks down to the fog's highest reach, so sinking into the
    // fog (or rising out of it) never pops.
    var share = 1.0;
    if (sky && fog.misc.w > 0.0) {
        share = 1.0 - fog.misc.w * smoothstep(y_max, y_max + 4.0, eye_y);
        if (share <= 0.0) {
            return out;
        }
    }
    if (sky) {
        // The void: the layer is endless below its top, so a downward sky ray ends fully fogged.
        // March it past the far plane, to 48 blocks under the top (surely opaque), so the fog sea
        // reaches the horizon instead of stopping at the render distance and showing vanilla's
        // dark void band there. The segment integral is exact however long the segments are.
        t1 = surface;
        if (dir.y < 0.0) {
            t1 = min(t1, (eye_y - (top - 0.5 * amp - 48.0)) / -dir.y);
        }
    }
    if (eye_y > y_max) {
        if (dir.y >= 0.0) {
            return out;
        }
        t0 = max(t0, (y_max - eye_y) / dir.y);
    } else if (dir.y > 0.0) {
        t1 = min(t1, (y_max - eye_y) / dir.y);
    }
    if (t1 <= t0) {
        return out;
    }

    let steps = i32(fog.wind.w);
    let jitter = ign(pos.xy + vec2<f32>(5.588238 * fog.misc.x));
    let cos_theta = dot(dir, fog.light_dir.xyz);
    let g = fog.light_dir.w;
    let hg = (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1.0e-4), 1.5);
    let lit = fog.ambient.rgb + fog.light_color.rgb * hg;

    var transmittance = 1.0;
    var inscatter = vec3<f32>(0.0);
    var ta = t0;
    for (var i = 0; i < steps; i = i + 1) {
        // Segments grow with distance (quadratic spacing): detail where it is seen.
        let x = f32(i + 1) / f32(steps);
        let tb = t0 + (t1 - t0) * x * x;
        let len = tb - ta;
        let ts = ta + len * jitter;
        let p = fog.eye.yzw + dir * ts + fog.wind.xyz;
        let n = textureSampleLevel(noise_tex, noise_sampler,
            p * vec3<f32>(fog.shape.z, fog.shape.z * 2.0, fog.shape.z), 0.0);
        let h = top + (n.r - 0.5) * amp;
        let wisp = mix(1.0, n.g * 2.0, fog.shape.w);
        let ramp = smoothstep(0.5 * start, start, ts);
        let dtau = segment_tau(eye_y + dir.y * ta, dir.y, len, h, density, softness) * wisp * ramp;
        let alpha = 1.0 - exp(-dtau);
        let y = eye_y + dir.y * ts;
        let deep = clamp((h - y) / (2.0 * softness + 1.0), 0.0, 1.0);
        let shade = 1.0 - fog.ambient.w * deep;
        inscatter = inscatter + transmittance * alpha * lit * shade;
        transmittance = transmittance * (1.0 - alpha);
        ta = tb;
        if (transmittance < 0.003) {
            break;
        }
    }
    if (share < 1.0) {
        inscatter = inscatter * share;
        transmittance = mix(1.0, transmittance, share);
    }
    out.fog = vec4<f32>(inscatter, transmittance);
    return out;
}
