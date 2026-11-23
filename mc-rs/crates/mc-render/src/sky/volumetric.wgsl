// Volumetric clouds. Two programs share this module:
//
// * `fs_trace` raymarches the cloud layer for one pixel of a low-resolution target (a quarter or
//   half of the frame in each direction) and folds the result into the previous frames' through
//   reprojection (temporal accumulation of a jittered trace). It writes premultiplied in-scattered
//   light plus transmittance, and the cloud's distance (for the next frame's reprojection and the
//   composite's depth test).
// * `fs_composite` lays that over the game's frame at full resolution, depth-tested against the
//   world: terrain nearer than the cloud layer keeps its exact pixels.
//
// The density and lighting: Perlin-Worley shape noise eroded by Worley detail noise, a coverage
// and cloud-type weather map, a density profile per cloud type through the layer's height, Beer's
// law with the "powder" term, a dual-lobe Henyey-Greenstein phase summed over multiple-scattering
// octaves, a height-graded sky ambient, a short cone-free march toward the light, a jittered
// low-resolution trace accumulated over frames, and a horizon fade that dissolves far clouds into
// the sky behind them.
//
// `texture_depth_2d` is rewritten to `texture_depth_multisampled_2d` when the game runs with MSAA
// (see `volumetric.rs`); `textureLoad`'s last argument is then the sample index, and 0 is right for
// both spellings.

struct Cloud {
    inv_view_proj: mat4x4<f32>,  // camera-relative clip -> camera-relative world
    prev_view_proj: mat4x4<f32>, // previous frame's camera-relative world -> clip
    cam_delta: vec4<f32>,        // xyz = camera now - camera then (blocks), w = 1 if history is usable
    eye: vec4<f32>,              // x = eye world y, y = 1: stop at scene depth, z = fade start, w = detail range
    origin_weather: vec4<f32>,   // xy = (camera xz + wind) mod weather tile, zw = same for the shape tile
    origin_detail: vec4<f32>,    // xy = (camera xz + detail wind) mod detail tile, z/w = shape/detail y drift
    layer: vec4<f32>,            // x = bottom y, y = top y, z = max distance, w = extinction per block
    scale: vec4<f32>,            // x/y/z = 1 / weather, shape, detail tile (blocks), w = detail erosion
    shape: vec4<f32>,            // x = coverage, y = rain, z = cloud-type bias, w = downwind lean (blocks)
    sun_dir: vec4<f32>,          // xyz = toward the light, w = first light-march step (blocks)
    sun_color: vec4<f32>,        // rgb = light radiance, w = ambient strength
    amb_top: vec4<f32>,          // rgb = sky ambient at the top of the layer
    amb_bottom: vec4<f32>,       // rgb = ambient at the base
    quality: vec4<f32>,          // x = view steps, y = light steps, z = new-frame weight, w = frame
    size: vec4<f32>,             // xy = trace target size, zw = full frame size
    jitter: vec4<f32>,           // xy = sub-pixel offset (0..1), z = temporal noise, w = divisor
    ray_z: vec4<f32>,            // x = NDC z of the near plane, y = a farther NDC z, z = powder
};

@group(0) @binding(0) var<uniform> U: Cloud;
@group(0) @binding(1) var shape_tex: texture_3d<f32>;
@group(0) @binding(2) var detail_tex: texture_3d<f32>;
@group(0) @binding(3) var weather_tex: texture_2d<f32>;
@group(0) @binding(4) var noise_samp: sampler;
@group(0) @binding(5) var history_color: texture_2d<f32>;
@group(0) @binding(6) var history_depth: texture_2d<f32>;
@group(0) @binding(7) var clamp_samp: sampler;
@group(0) @binding(8) var scene_depth: texture_depth_2d;

struct Composite {
    inv_view_proj: mat4x4<f32>,
    eye: vec4<f32>,   // x = camera world y, y = layer bottom, z = layer top, w = exposure
    grid: vec4<f32>,  // xy = trace size * divisor (the full pixels the low-res grid spans), zw = near/far NDC z
};

@group(0) @binding(10) var<uniform> C: Composite;
@group(0) @binding(11) var cloud_color: texture_2d<f32>;
@group(0) @binding(12) var cloud_depth: texture_2d<f32>;
@group(0) @binding(13) var comp_samp: sampler;
@group(0) @binding(14) var comp_depth: texture_depth_2d;

const PI: f32 = 3.14159265;
const MAX_STEPS: i32 = 160;
// How quickly the accumulation hands over to the fresh trace as near cloud moves on screen with
// the camera's translation: the new frame's weight is at least this times the camera's move over
// the cloud's distance. Far clouds (tens of blocks and more) never
// reach the in-game weight of 0.1 with it; cloud a few blocks away while jumping does.
const PARALLAX_TRUST: f32 = 8.0;

// ------------------------------------------------------------------------------------ shared

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(i & 1u) * 4 - 1);
    let y = f32(i32(i >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn unproject(m: mat4x4<f32>, ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let p = m * vec4<f32>(ndc, z, 1.0);
    return p.xyz / p.w;
}

// Where a ray from world height `y0` along `rd` is inside the layer: [x, y], empty when y < x.
fn slab(y0: f32, rd: vec3<f32>, bottom: f32, top: f32) -> vec2<f32> {
    if (abs(rd.y) < 1.0e-6) {
        if (y0 >= bottom && y0 <= top) {
            return vec2<f32>(0.0, 1.0e9);
        }
        return vec2<f32>(1.0, 0.0);
    }
    let ta = (bottom - y0) / rd.y;
    let tb = (top - y0) / rd.y;
    return vec2<f32>(max(min(ta, tb), 0.0), max(ta, tb));
}

fn remap(v: f32, lo: f32, hi: f32, nlo: f32, nhi: f32) -> f32 {
    return nlo + (v - lo) * (nhi - nlo) / (hi - lo);
}

// A per-pixel random number for the ray start, spatially white (a PCG hash of the pixel). The
// frame adds a golden-ratio step to it, so each pixel walks a low-discrepancy sequence over time.
// Interleaved gradient noise was tried first: its structure (it varies mostly along x) survived
// the accumulation as faint vertical stripes one trace pixel apart.
fn pixel_noise(p: vec2<f32>) -> f32 {
    var v = vec2<u32>(p) * vec2<u32>(1664525u, 1013904223u) + vec2<u32>(1013904223u, 1664525u);
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    return f32(v.x >> 8u) * (1.0 / 16777216.0);
}

// ------------------------------------------------------------------------------------ density

// The density profile through the layer (h = 0 at the base, 1 at the top) for a cloud type:
// 0 = stratus (a low, flat sheet), 0.5 = stratocumulus, 1 = cumulus (rounded base, tall, tapering).
fn height_gradient(h: f32, kind: f32) -> f32 {
    let stratus = smoothstep(0.0, 0.07, h) * (1.0 - smoothstep(0.16, 0.30, h));
    let strato = smoothstep(0.0, 0.10, h) * (1.0 - smoothstep(0.32, 0.60, h));
    let cumulus = smoothstep(0.0, 0.10, h) * (1.0 - smoothstep(0.50, 0.97, h));
    let low = mix(stratus, strato, clamp(kind * 2.0, 0.0, 1.0));
    return mix(low, cumulus, clamp(kind * 2.0 - 1.0, 0.0, 1.0));
}

// How much of this column the weather map allows to be cloud, after the coverage setting.
fn coverage_at(w: vec4<f32>) -> f32 {
    // The weather field is 0.5 +- 0.2 on average; `c` slides it: 0 is a clear sky (a few wisps
    // where the field peaks), 0.5 partly cloudy, 1 overcast.
    let field = w.r * 0.8 + w.b * 0.2;
    return clamp(field + (U.shape.x - 0.5) * 1.6, 0.0, 1.0);
}

// The cheap density: weather map, shape noise, height profile and coverage (the base cloud with
// coverage applied, `remap(v, 1 - cover, 1, 0, 1)`), kept in 0..1 so the detail erosion works
// on the scale it was tuned for; thin coverage still makes lighter clouds. `p` is
// camera-relative, `h` the height fraction in the layer.
fn density_base(p: vec3<f32>, h: f32) -> f32 {
    let wth = textureSampleLevel(weather_tex, noise_samp, (U.origin_weather.xy + p.xz) * U.scale.x, 0.0);
    let cover = coverage_at(wth);
    if (cover <= 0.0) {
        return 0.0;
    }
    let profile = height_gradient(h, clamp(wth.g + U.shape.z, 0.0, 1.0));
    if (profile <= 0.0) {
        return 0.0;
    }
    // Clouds lean downwind with height, as real cumulus do.
    let lean = U.shape.w * h;
    let q = vec3<f32>(
        U.origin_weather.z + p.x + lean,
        U.eye.x + p.y + U.origin_detail.z,
        U.origin_weather.w + p.z,
    ) * U.scale.y;
    // The shape channel is already Perlin-Worley (masses with billows), stretched to 0..1.
    let shape = textureSampleLevel(shape_tex, noise_samp, q, 0.0).r;
    let c = clamp((shape * profile + cover - 1.0) / max(cover, 0.05), 0.0, 1.0);
    return c * (0.4 + 0.6 * cover);
}

// The full density: `base` eroded at its edges by the detail noise, wispy near the base and
// billowing higher up. The erosion fades out with distance `t` (a level of detail: far away the
// detail is smaller than a trace pixel, and sampling it there only adds noise).
fn density_full(p: vec3<f32>, h: f32, base: f32, t: f32) -> f32 {
    let erosion = U.scale.w * (1.0 - smoothstep(U.eye.w * 0.4, U.eye.w, t));
    if (erosion <= 0.0) {
        return base;
    }
    let q = vec3<f32>(
        U.origin_detail.x + p.x,
        U.eye.x + p.y + U.origin_detail.w,
        U.origin_detail.y + p.z,
    ) * U.scale.z;
    let d = textureSampleLevel(detail_tex, noise_samp, q, 0.0);
    let fbm = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
    let billow = clamp(h * 5.0, 0.0, 1.0);
    let m = mix(fbm, 1.0 - fbm, billow);
    var dens = max(remap(base, m * erosion, 1.0, 0.0, 1.0), 0.0);
    // Close clouds (Minecraft's are only 60-100 blocks overhead) get a second octave 3.7 times
    // finer, so their edges break into small wisps instead of a soft blur.
    let near = 1.0 - smoothstep(U.eye.w * 0.12, U.eye.w * 0.3, t);
    if (near > 0.0 && dens > 0.0) {
        let d2 = textureSampleLevel(detail_tex, noise_samp, q * 3.7 + vec3<f32>(0.37, 0.71, 0.13), 0.0);
        let f2 = d2.r * 0.5 + d2.g * 0.35 + d2.b * 0.15;
        let m2 = mix(f2, 1.0 - f2, billow);
        dens = max(remap(dens, m2 * erosion * 0.55 * near, 1.0, 0.0, 1.0), 0.0);
    }
    return dens;
}

// ------------------------------------------------------------------------------------ light

// Henyey-Greenstein, normalised so isotropic scattering is 1.
fn hg(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * g * c, 1.0e-4), 1.5);
}

// A strong forward lobe (the bright silver lining looking toward the sun) and a weaker back
// lobe (the lit face of a cloud with the sun behind you). `k` flattens both for the
// multiple-scattering octaves.
fn phase2(c: f32, k: f32) -> f32 {
    return mix(hg(c, 0.72 * k), hg(c, -0.28 * k), 0.32);
}

// Optical depth toward the light from `p`, through the cheap density, in growing steps.
fn light_depth(p: vec3<f32>) -> f32 {
    let n = i32(U.quality.y);
    let thick = U.layer.y - U.layer.x;
    var step = U.sun_dir.w;
    var pos = p;
    var od = 0.0;
    for (var i = 0; i < 8; i = i + 1) {
        if (i >= n) {
            break;
        }
        pos = pos + U.sun_dir.xyz * step;
        let h = (U.eye.x + pos.y - U.layer.x) / thick;
        if (h < 0.0 || h > 1.0) {
            break;
        }
        od = od + density_base(pos, h) * step;
        step = step * 1.75;
    }
    return od * U.layer.w;
}

struct March {
    color: vec4<f32>,
    depth: f32,
    // Whether the ray met any cloud (its depth is then where the cloud is).
    hit: bool,
};

fn march(ro: vec3<f32>, rd: vec3<f32>, t0: f32, t1: f32, px: vec2<f32>) -> March {
    let seg = t1 - t0;
    // Steps stretch with the segment: short looking up, long toward the horizon.
    let dt = max(seg / U.quality.x, 0.4);
    let n = min(i32(ceil(seg / dt)), MAX_STEPS);
    var t = t0 + dt * fract(pixel_noise(px) + U.jitter.z);
    let thick = U.layer.y - U.layer.x;
    let mu = dot(rd, U.sun_dir.xyz);
    // Multiple scattering approximated by octaves: each octave scatters half as much, sees half
    // the extinction and a flatter phase. The phases depend only on the view angle, so they are
    // computed once per ray.
    let ph0 = phase2(mu, 1.0);
    let ph1 = phase2(mu, 0.5) * 0.5;
    let ph2 = phase2(mu, 0.25) * 0.25;
    // The powder term darkens thin cloud seen with the sun behind you; looking toward the sun it
    // would fight the silver lining, so it fades out there.
    let powder_mix = clamp(0.5 - 0.5 * mu, 0.0, 1.0) * U.ray_z.z;
    var trans = 1.0;
    var light = vec3<f32>(0.0);
    var dsum = 0.0;
    var wsum = 0.0;
    for (var i = 0; i < MAX_STEPS; i = i + 1) {
        if (i >= n || trans < 0.01) {
            break;
        }
        let p = ro + rd * t;
        let h = (U.eye.x + p.y - U.layer.x) / thick;
        let base = density_base(p, h);
        if (base > 0.0) {
            let dens = density_full(p, h, base, t);
            if (dens > 0.0) {
                let sigma = dens * U.layer.w;
                let od = light_depth(p);
                // Capped: the forward peak on a thin wisp would otherwise be a lone firefly.
                let beer = min(ph0 * exp(-od) + ph1 * exp(-od * 0.5) + ph2 * exp(-od * 0.25), 6.0);
                let powder = mix(1.0, 1.0 - exp(-dens * 6.0), powder_mix);
                let amb = mix(U.amb_bottom.rgb, U.amb_top.rgb, clamp(h, 0.0, 1.0)) * U.sun_color.w;
                let s = U.sun_color.rgb * beer * powder + amb;
                // Energy-conserving step (analytic integral over the step) for a white
                // (albedo 1) medium.
                let step_t = exp(-sigma * dt);
                let absorbed = trans * (1.0 - step_t);
                light = light + s * absorbed;
                dsum = dsum + t * absorbed;
                wsum = wsum + absorbed;
                trans = trans * step_t;
            }
        }
        t = t + dt;
    }
    var m: March;
    let depth = select(t0, dsum / max(wsum, 1.0e-6), wsum > 1.0e-4);
    // Aerial perspective: far clouds dissolve into the sky behind them.
    let fade = 1.0 - smoothstep(U.eye.z, U.layer.z, depth);
    m.color = vec4<f32>(light * fade, 1.0 - (1.0 - trans) * fade);
    m.depth = depth;
    m.hit = wsum > 1.0e-4;
    return m;
}

// ------------------------------------------------------------------------------------ trace

struct TraceOut {
    @location(0) color: vec4<f32>,
    @location(1) depth: vec4<f32>,
};

@fragment
fn fs_trace(@builtin(position) frag: vec4<f32>) -> TraceOut {
    let cell = floor(frag.xy);
    let k = U.jitter.w;
    // This pixel's ray, through a sub-pixel position that moves every frame, so the accumulated
    // result averages the whole footprint.
    let sample_px = (cell + U.jitter.xy) * k;
    let ndc = vec2<f32>(sample_px.x / U.size.z * 2.0 - 1.0, 1.0 - sample_px.y / U.size.w * 2.0);
    let ro = unproject(U.inv_view_proj, ndc, U.ray_z.x);
    let rd = normalize(unproject(U.inv_view_proj, ndc, U.ray_z.y) - ro);

    // Stop at the world: the farthest depth in the footprint, so a sky pixel next to a
    // silhouette still sees its clouds (the composite cuts terrain out per full-res pixel).
    var t_limit = U.layer.z;
    if (U.eye.y > 0.5) {
        let ki = i32(k);
        let last = vec2<i32>(U.size.zw) - vec2<i32>(1);
        let origin = vec2<i32>(cell) * ki;
        var far = 0.0;
        for (var j = 0; j < 4; j = j + 1) {
            if (j >= ki) {
                break;
            }
            for (var i = 0; i < 4; i = i + 1) {
                if (i >= ki) {
                    break;
                }
                far = max(far, textureLoad(scene_depth, min(origin + vec2<i32>(i, j), last), 0));
            }
        }
        if (far < 1.0) {
            t_limit = min(t_limit, length(unproject(U.inv_view_proj, ndc, far) - ro));
        }
    }

    let seg = slab(U.eye.x + ro.y, rd, U.layer.x, U.layer.y);
    let t1 = min(seg.y, t_limit);
    var fresh = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var fresh_depth = max(seg.x, 0.0);
    var hit = false;
    // The world covers this whole footprint in front of the layer (a player, a nametag, a wall):
    // nothing to trace, and nothing of it is shown (the composite keeps those pixels).
    let occluded = seg.y > seg.x && t1 <= seg.x;
    if (t1 > seg.x) {
        let m = march(ro, rd, seg.x, t1, frag.xy);
        fresh = m.color;
        fresh_depth = m.depth;
        hit = m.hit;
    }

    var out: TraceOut;
    var color = fresh;
    var depth = fresh_depth;
    // A missing layer has no world-space cloud point to reproject. In particular, its
    // fallback distance is zero: a vertical camera delta would project that near-eye point
    // into the old sky and drag cloud history down across the horizon while jumping.
    if (U.cam_delta.w > 0.5 && (t1 > seg.x || occluded)) {
        // Where this pixel's cloud was on screen last frame, through the pixel's centre (not the
        // jittered sample): with a still camera that lands exactly on the same history texel, so
        // the accumulated image never drifts or blurs from resampling.
        let centre = (cell + vec2<f32>(0.5)) * k;
        let cndc = vec2<f32>(centre.x / U.size.z * 2.0 - 1.0, 1.0 - centre.y / U.size.w * 2.0);
        let co = unproject(U.inv_view_proj, cndc, U.ray_z.x);
        let cd = normalize(unproject(U.inv_view_proj, cndc, U.ray_z.y) - co);
        // A ray that met cloud follows its cloud's distance. An empty ray has no point to
        // follow, so it is reprojected as far away: only the camera's turn moves it. Its old stand-in, the distance where the ray enters the layer, is
        // zero inside the layer and nearly zero just under it, and a jump then projected that
        // near-eye point far up or down the screen: the history of the cloud ceiling smeared
        // over the sky below the horizon, "stretching to the top or bottom of the screen".
        var prev: vec4<f32>;
        if (hit) {
            prev = U.prev_view_proj * vec4<f32>(co + cd * fresh_depth + U.cam_delta.xyz, 1.0);
        } else {
            prev = U.prev_view_proj * vec4<f32>(cd, 0.0);
        }
        if (prev.w > 1.0e-5) {
            let q = prev.xy / prev.w;
            // Frame UV to history UV: the trace grid spans `size.xy * divisor` frame pixels, a
            // little more than the frame when the divisor does not divide it.
            let fuv = vec2<f32>(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
            let huv = fuv * U.size.zw / (U.size.xy * k);
            if (all(fuv > vec2<f32>(0.0)) && all(fuv < vec2<f32>(1.0))) {
                let hc = textureSampleLevel(history_color, clamp_samp, huv, 0.0);
                let hd = textureSampleLevel(history_depth, clamp_samp, huv, 0.0).r;
                var a = U.quality.z;
                // Cloud now and cloud then, at very different distances: this pixel saw another
                // cloud last frame (a cloud edge alone, with cloud in one sample and sky in the
                // other, is what the accumulation is for, not a reason to reset it).
                let both = min(1.0 - fresh.a, 1.0 - hc.a);
                if (both > 0.05 && abs(hd - fresh_depth) > fresh_depth * 0.25 + 12.0) {
                    a = max(a, 0.5);
                }
                // Parallax: cloud a few blocks away (in this trace, or in the history sample
                // about to be reused) slides across the screen as the camera moves, and the
                // reprojection is only as good as its distance estimate. Lean on the fresh trace
                // in proportion to the move over that distance (B5). Far clouds are unaffected.
                let near_fresh = select(1.0e9, fresh_depth, hit);
                let near = min(near_fresh, select(1.0e9, hd, 1.0 - hc.a > 0.02));
                let parallax = length(U.cam_delta.xyz) / max(near, 0.25);
                a = max(a, clamp(parallax * PARALLAX_TRUST, 0.0, 1.0));
                // Behind an occluder, carry the accumulated clouds through unchanged instead of
                // writing "no cloud" into them: a moving player or nametag in front of the sky
                // left a cloud-free ghost of itself in the history, which then faded in over
                // the next frames wherever it moved away from.
                a = select(a, 0.0, occluded);
                color = mix(hc, fresh, a);
                depth = mix(hd, fresh_depth, a);
            }
        }
    }
    out.color = color;
    out.depth = vec4<f32>(depth, 0.0, 0.0, 1.0);
    return out;
}

// ------------------------------------------------------------------------------------ composite

// Linear cloud light to the frame's display values (the game draws to a non-sRGB target in
// display space): a soft shoulder so sunlit tops stay white instead of clipping flat, then the
// usual 2.2 encode.
fn to_display(x: vec3<f32>) -> vec3<f32> {
    let shoulder = vec3<f32>(1.0) - exp(-x * 1.15);
    return pow(max(shoulder, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

// The weights of the Mitchell-Netravali cubic (B = C = 1/3) for the four texels around a point a
// fraction `t` of the way from the second texel to the third (per axis). They sum to 1.
struct CubicWeights {
    before: vec2<f32>,
    first: vec2<f32>,
    second: vec2<f32>,
    after: vec2<f32>,
};

fn mitchell_weights(t: vec2<f32>) -> CubicWeights {
    let t2 = t * t;
    let t3 = t2 * t;
    var w: CubicWeights;
    w.before = (1.0 - 9.0 * t + 15.0 * t2 - 7.0 * t3) / 18.0;
    w.first = (16.0 - 36.0 * t2 + 21.0 * t3) / 18.0;
    w.second = (1.0 + 9.0 * t + 27.0 * t2 - 21.0 * t3) / 18.0;
    w.after = (7.0 * t3 - 6.0 * t2) / 18.0;
    return w;
}

// A bicubic (Mitchell-Netravali) upsample of the trace: crisper cloud edges than one bilinear
// tap, with half of Catmull-Rom's negative lobe, which turned the trace's leftover per-pixel noise
// into a visible grid. Per axis the two middle texels share one bilinear tap placed between them
// in proportion to their weights, so a 3x3 grid of bilinear taps covers the 4x4 texels. Clamped,
// since any negative lobe can overshoot.
fn upsample(uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(cloud_color));
    let px = uv * size - 0.5;
    let base = floor(px);
    let w = mitchell_weights(px - base);
    let middle = w.first + w.second;
    let lo = (base - 0.5) / size;
    let mid = (base + 0.5 + w.second / middle) / size;
    let hi = (base + 2.5) / size;
    let xs = vec3<f32>(lo.x, mid.x, hi.x);
    let ys = vec3<f32>(lo.y, mid.y, hi.y);
    let wx = vec3<f32>(w.before.x, middle.x, w.after.x);
    let wy = vec3<f32>(w.before.y, middle.y, w.after.y);
    // The nine taps written out row by row, not looped: every read is issued before the sum
    // needs one.
    var s = vec4<f32>(0.0);
    s = s + textureSampleLevel(cloud_color, comp_samp, vec2<f32>(xs.x, ys.x), 0.0) * (wx.x * wy.x);
    s = s + textureSampleLevel(cloud_color, comp_samp, vec2<f32>(xs.y, ys.x), 0.0) * (wx.y * wy.x);
    s = s + textureSampleLevel(cloud_color, comp_samp, vec2<f32>(xs.z, ys.x), 0.0) * (wx.z * wy.x);
    s = s + textureSampleLevel(cloud_color, comp_samp, vec2<f32>(xs.x, ys.y), 0.0) * (wx.x * wy.y);
    s = s + textureSampleLevel(cloud_color, comp_samp, vec2<f32>(xs.y, ys.y), 0.0) * (wx.y * wy.y);
    s = s + textureSampleLevel(cloud_color, comp_samp, vec2<f32>(xs.z, ys.y), 0.0) * (wx.z * wy.y);
    s = s + textureSampleLevel(cloud_color, comp_samp, vec2<f32>(xs.x, ys.z), 0.0) * (wx.x * wy.z);
    s = s + textureSampleLevel(cloud_color, comp_samp, vec2<f32>(xs.y, ys.z), 0.0) * (wx.y * wy.z);
    s = s + textureSampleLevel(cloud_color, comp_samp, vec2<f32>(xs.z, ys.z), 0.0) * (wx.z * wy.z);
    return vec4<f32>(max(s.rgb, vec3<f32>(0.0)), clamp(s.a, 0.0, 1.0));
}

@fragment
fn fs_composite(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // Reject impossible cloud rays at full resolution too. The low-resolution filter can
    // straddle the horizon; clear scene depth must not let those samples paint below it.
    let full = vec2<f32>(textureDimensions(comp_depth));
    let fuv = frag.xy / full;
    let ndc = vec2<f32>(fuv.x * 2.0 - 1.0, 1.0 - fuv.y * 2.0);
    let ro = unproject(C.inv_view_proj, ndc, C.grid.z);
    let rd = normalize(unproject(C.inv_view_proj, ndc, C.grid.w) - ro);
    let seg = slab(C.eye.x + ro.y, rd, C.eye.y, C.eye.z);
    if (seg.y <= seg.x) {
        discard;
    }
    let uv = frag.xy / C.grid.xy;
    let c = upsample(uv);
    var cover = 1.0 - c.a;
    if (cover < 0.002) {
        discard;
    }
    var rgb = c.rgb;
    let d = textureLoad(comp_depth, vec2<i32>(frag.xy), 0);
    if (d < 1.0) {
        // Something was drawn here: keep it wherever it is nearer than the cloud layer.
        let sp = unproject(C.inv_view_proj, ndc, d);
        let ts = length(sp - ro);
        if (ts <= seg.x) {
            discard;
        }
        // Terrain inside the layer: only the cloud in front of it counts.
        let cd = textureSampleLevel(cloud_depth, comp_samp, uv, 0.0).r;
        let keep = clamp((ts - seg.x) / max(cd - seg.x, 1.0), 0.0, 1.0);
        rgb = rgb * keep;
        cover = cover * keep;
        if (cover < 0.002) {
            discard;
        }
    }
    let display = to_display(rgb / cover * C.eye.w);
    return vec4<f32>(display * cover, 1.0 - cover);
}
