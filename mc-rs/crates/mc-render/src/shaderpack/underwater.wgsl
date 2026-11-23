// The Underwater shader pack's world passes.
//
// Prepended at build time: `post/view_geometry_common.wgsl` (the camera uniform at @group(0),
// the full-screen `vs`, `vg_linear_depth`, `vg_view_position`) and `caustics::wave_wgsl()` (the
// surface waves the caustic texture was baked from).
//
// Two entry points share this module:
//   * `fs_shafts`: the volumetric light shafts at half or quarter resolution. Each texel marches
//     its view ray through the water and sums the caustic light pattern extruded along the
//     refracted sun direction (`rg` = shaft light, the texel's distance for the bilateral
//     upsample).
//   * `fs_composite`: full resolution, reads the frame so far (`scene_tex`) and the depth buffer,
//     and writes the underwater look: caustics on surfaces facing the light, depth darkening,
//     per-channel absorption and in-scatter, the far fade, the water-surface sky, the shafts.
//
// Everything happens in linear light: the frame is display-encoded 8-bit (the surface's byte
// domain), decoded with a 2.2 power and encoded back at the end. Positions are camera-relative
// world offsets (`rel`); the uniform carries every absolute coordinate already wrapped into a
// tile period on the CPU in double precision, so nothing here loses precision far from 0,0.

struct UwParams {
    // xyz = toward the light inside the water (the refracted sun or moon), w = light strength.
    light_dir: vec4<f32>,
    // rgb = light colour (linear), a = shaft strength.
    light_color: vec4<f32>,
    // rgb = the water's in-scatter colour at eye level (linear), a = clear range (blocks).
    water: vec4<f32>,
    // rgb = absorption per block, a = far fade start (blocks).
    sigma: vec4<f32>,
    // x = far fade end (blocks), y = depth darkening per block, z = darkest factor,
    // w = water surface height above the eye (blocks).
    depth: vec4<f32>,
    // x = caustic strength, y = 1 / caustic tile (per block), z = wave loop phase (0..1),
    // w = 1 / shaft tile (per block).
    caustic: vec4<f32>,
    // xy = eye xz in caustic tiles (fractional part), zw = eye on the shaft basis in shaft tiles.
    wrap: vec4<f32>,
    // xyz = shaft basis u (perpendicular to the light), w = shaft march length (blocks).
    shaft_u: vec4<f32>,
    // xyz = shaft basis v, w = shaft steps.
    shaft_v: vec4<f32>,
    // rgb = sky colour overhead (linear), a = 1 to draw the water-surface sky.
    sky: vec4<f32>,
    // rgb = sky colour at the horizon (linear), a = surface ripple strength.
    horizon: vec4<f32>,
    // xyz = sun (or moon) direction in the air, w = disc brightness.
    sun: vec4<f32>,
    // x = the eye is really in water, y = refraction wobble (pixels at 1080p), z = frame index,
    // w = full-resolution pixels per shaft texel.
    flags: vec4<f32>,
    // xy = shaft target size, z = how dark straight down gets (water glow), w = caustic near keep.
    grid: vec4<f32>,
    // xy = the eye relative to the occlusion map's corner (blocks), z = the eye's height,
    // w = 1 when the map is in use.
    occl: vec4<f32>,
};

@group(1) @binding(0) var<uniform> uw: UwParams;
@group(1) @binding(1) var depth_tex: texture_depth_2d;
@group(1) @binding(2) var scene_tex: texture_2d<f32>;
@group(1) @binding(3) var caustic_tex: texture_3d<f32>;
@group(1) @binding(4) var lin_samp: sampler;
@group(1) @binding(5) var shaft_tex: texture_2d<f32>;
// Per block column around the player: one above the highest block that keeps sunlight out.
@group(1) @binding(6) var occlusion_tex: texture_2d<u32>;

const UW_OCCL_SIZE: i32 = 192;

// 1 when sunlight coming down the refracted light direction reaches the point `lift` blocks up
// that ray from `rel`: that point is at or above the top light-blocking block of its column.
fn uw_open_at(rel: vec3<f32>, lift: f32) -> f32 {
    let light = uw.light_dir.xyz;
    let p = rel + light * (lift / max(light.y, 0.25));
    let c = vec2<i32>(floor(uw.occl.xy + p.xz));
    if (any(c < vec2<i32>(0)) || any(c >= vec2<i32>(UW_OCCL_SIZE))) {
        return 1.0;
    }
    let top = f32(textureLoad(occlusion_tex, c, 0).r);
    // A quarter-block of slack: reconstructed positions on a block's top face may sit a hair
    // below it.
    return select(0.0, 1.0, uw.occl.z + p.y >= top - 0.25);
}

// Whether a surface point is in the sunlight: nothing right above it, nor a few blocks up the
// light (a roof, an overhang, the island above). 1 when the map is off.
fn uw_open(rel: vec3<f32>) -> f32 {
    if (uw.occl.w < 0.5) {
        return 1.0;
    }
    return uw_open_at(rel, 0.0) * uw_open_at(rel, 3.0) * uw_open_at(rel, 9.0);
}

const UW_TAU: f32 = 6.2831853;
const UW_ETA: f32 = 1.3333;
// What a sky texel's distance means to the shafts and the upsample.
const UW_SKY_DIST: f32 = 4096.0;

fn uw_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn uw_to_display(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

// The camera's view is rotation-only, so its inverse is the transpose.
fn uw_view_to_world(v: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    return transpose(m) * v;
}

// Interleaved gradient noise: a cheap per-pixel jitter whose error TAA and the
// bilateral upsample both average out well.
fn uw_ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// The water's own glow seen along a direction: brighter toward the surface, darker toward the
// depths (how dark straight down gets is `grid.z`).
fn uw_glow(dir_y: f32) -> vec3<f32> {
    let down = clamp(-dir_y, 0.0, 1.0);
    let up = clamp(dir_y, 0.0, 1.0);
    let f = mix(1.0, uw.grid.z, sqrt(down)) * (1.0 + 0.35 * up);
    return uw.water.rgb * f;
}

// The surface height field's gradient at `p` (caustic tiles) and loop phase `t`, from the same
// waves the caustics were baked from.
fn uw_wave_gradient(p: vec2<f32>, t: f32) -> vec2<f32> {
    var g = vec2<f32>(0.0);
    for (var i = 0u; i < UW_WAVE_COUNT; i = i + 1u) {
        let w = UW_WAVES[i];
        let theta = UW_TAU * (w.x * p.x + w.y * p.y - w.z * t) + UW_WAVE_PHASE[i];
        g = g + w.w * cos(theta) * w.xy;
    }
    return g * UW_TAU;
}

// What the sky looks like through the water surface, along the air-side direction `r`: the real
// frame where that direction is on screen (and still sky there), vanilla's colours otherwise.
fn uw_sky_through(r: vec3<f32>) -> vec3<f32> {
    let h = clamp(r.y, 0.0, 1.0);
    var col = mix(uw.horizon.rgb, uw.sky.rgb, smoothstep(0.0, 0.5, h));
    let sd = max(dot(r, uw.sun.xyz), 0.0);
    col = col + uw.light_color.rgb * (pow(sd, 600.0) * 2.0 + pow(sd, 12.0) * 0.08) * uw.sun.w;
    let clip = camera.view_proj * vec4<f32>(r * 512.0, 1.0);
    if (clip.w > 0.001) {
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let border = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
        if (border > 0.0) {
            let size = vec2<i32>(camera.screen.xy);
            let px = clamp(vec2<i32>(uv * camera.screen.xy), vec2<i32>(0), size - vec2<i32>(1));
            if (textureLoad(depth_tex, px, 0) >= 1.0) {
                let scene = uw_to_linear(textureSampleLevel(scene_tex, lin_samp, uv, 0.0).rgb);
                col = mix(col, scene, smoothstep(0.0, 0.05, border));
            }
        }
    }
    return col;
}

// What a pixel with no geometry shows along `v`: the underside of the water surface (Snell's
// window with the sky in it, total internal reflection outside it) above, open water below.
fn uw_sky(v: vec3<f32>) -> vec3<f32> {
    let glow = uw_glow(v.y);
    if (uw.sky.a < 0.5 || v.y <= 0.004) {
        return glow;
    }
    let height = max(uw.depth.w, 0.5);
    let t = height / v.y;
    let s = v * t;
    // Ripples fade out where a pixel covers more than a fraction of a wave (far away and at
    // grazing angles): there they would alias into stripes, and the real surface averages to a
    // mirror anyway.
    let pixel_angle = 2.0 / (camera.proj[1][1] * camera.screen.y);
    let footprint = t * pixel_angle / max(v.y, 0.05) * uw.caustic.y * 12.0;
    let fade = uw.horizon.a * (1.0 - smoothstep(0.06, 0.25, footprint));
    let g = uw_wave_gradient(uw.wrap.xy + s.xz * uw.caustic.y, uw.caustic.z) * fade;
    let n = normalize(vec3<f32>(-g.x, 1.0, -g.y));
    let cos_i = clamp(dot(v, n), 0.0, 1.0);
    let sin2_t = UW_ETA * UW_ETA * (1.0 - cos_i * cos_i);
    // Transmission falls to nothing at the critical angle (Fresnel, roughly).
    let window = smoothstep(1.0, 0.78, sin2_t);
    let mirror = uw_glow(reflect(v, -n).y) * 0.9;
    var surface = mirror;
    if (window > 0.0) {
        let r = refract(v, -n, UW_ETA);
        surface = mix(mirror, uw_sky_through(r), window);
    }
    let travelled = max(t - uw.water.a, 0.0);
    let trans = exp(-uw.sigma.rgb * travelled);
    return uw_shoulder(surface * trans + glow * (1.0 - trans));
}

// A soft shoulder for the bright window overhead: linear below 0.6, then easing toward 1, so the
// sun and the sky through the surface glow instead of clipping to flat white.
fn uw_shoulder(c: vec3<f32>) -> vec3<f32> {
    let k = 0.6;
    let over = max(c - vec3<f32>(k), vec3<f32>(0.0));
    return min(c, vec3<f32>(k)) + over / (vec3<f32>(1.0) + over / (1.0 - k));
}

// ---------------------------------------------------------------------------- light shafts

// Henyey-Greenstein, normalised to 1 at 90 degrees, blended with isotropic scattering. Mild
// forward scattering: the shafts brighten toward the light without blinding.
fn uw_phase(cos_theta: f32) -> f32 {
    let g = 0.35;
    let g2 = g * g;
    let hg = (1.0 - g2) / pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5);
    let side = (1.0 - g2) / pow(1.0 + g2, 1.5);
    return 0.5 + 0.5 * hg / side;
}

@fragment
fn fs_shafts(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
    let scale = uw.flags.w;
    let full = (floor(pos.xy) + 0.5) * scale;
    let size = vec2<i32>(camera.screen.xy);
    let fp = clamp(vec2<i32>(full), vec2<i32>(0), size - vec2<i32>(1));
    let d = textureLoad(depth_tex, fp, 0);
    let sky = d >= 1.0;
    let dist = select(vg_linear_depth(d), UW_SKY_DIST, sky);
    let rel = uw_view_to_world(vg_view_position(vec2<f32>(fp) + 0.5, dist));
    let rd = select(length(rel), UW_SKY_DIST, sky);
    let v = normalize(rel);
    let light = uw.light_dir.xyz;
    let height = uw.depth.w;
    // Light exists only below the surface: stop where the ray leaves the water going up. And no
    // shaft light inside the clear range, so fighting range stays as crisp as the absorption
    // keeps it.
    var tmax = min(rd, uw.shaft_u.w);
    if (v.y > 0.001) {
        tmax = min(tmax, height / v.y);
    }
    let t0 = min(uw.water.a, tmax);
    let steps = max(uw.shaft_v.w, 1.0);
    let dt = (tmax - t0) / steps;
    let jitter = uw_ign(full + vec2<f32>(5.588238 * uw.flags.z, 3.17 * uw.flags.z));
    let sig = dot(uw.sigma.rgb, vec3<f32>(0.3333));
    let inv_ly = 1.0 / max(light.y, 0.25);
    var acc = 0.0;
    for (var i = 0.0; i < steps; i = i + 1.0) {
        let t = t0 + (i + jitter) * dt;
        let x = v * t;
        let uv = vec2<f32>(dot(x, uw.shaft_u.xyz), dot(x, uw.shaft_v.xyz)) * uw.caustic.w
            + uw.wrap.zw;
        let pattern = textureSampleLevel(caustic_tex, lin_samp, vec3<f32>(uv, uw.caustic.z), 0.0)
            .g * 2.0;
        // Down from the surface to x, then on to the eye; nothing where an island or a roof
        // keeps the light out (the shafts come through the gaps).
        let path = max(height - x.y, 0.0) * inv_ly + max(t - uw.water.a, 0.0);
        var open = 1.0;
        if (uw.occl.w > 0.5) {
            open = uw_open_at(x, 0.0) * uw_open_at(x, 6.0);
        }
        // Only the bright part of the pattern: streaks of light, not a uniform veil.
        acc = acc + max(pattern - 0.85, 0.0) * exp(-sig * path) * open;
    }
    let shafts = acc * dt / 28.0 * uw_phase(dot(v, light));
    return vec2<f32>(shafts, rd);
}

// A 5x5 Gaussian over the raw shafts at their own resolution, weighted by distance agreement:
// the march's per-pixel jitter becomes smooth light instead of a dither pattern, whether or not
// TAA runs, and shafts never bleed across a silhouette. (Here `shaft_tex` is the raw target.)
@fragment
fn fs_shaft_blur(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
    let p = vec2<i32>(pos.xy);
    let last = vec2<i32>(uw.grid.xy) - vec2<i32>(1);
    let centre = textureLoad(shaft_tex, p, 0).rg;
    var sum = 0.0;
    var wsum = 0.0;
    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let q = clamp(p + vec2<i32>(x, y), vec2<i32>(0), last);
            let s = textureLoad(shaft_tex, q, 0).rg;
            let gauss = exp(-f32(x * x + y * y) * 0.3);
            let tolerance = 0.08 * min(centre.y, s.y) + 0.5;
            let agree = 1.0 / (1.0 + abs(s.y - centre.y) / tolerance);
            let w = gauss * agree * agree;
            sum = sum + s.x * w;
            wsum = wsum + w;
        }
    }
    return vec2<f32>(sum / max(wsum, 1e-6), centre.y);
}

// ------------------------------------------------------------------------------- composite

// The shafts at a full-resolution pixel: the four nearest low-resolution texels, weighted by
// distance agreement so shafts never bleed across a silhouette.
fn uw_shafts_at(full_px: vec2<f32>, rd: f32) -> f32 {
    let scale = uw.flags.w;
    let f = full_px / scale - 0.5;
    let base = floor(f);
    let fr = f - base;
    let last = vec2<i32>(uw.grid.xy) - vec2<i32>(1);
    var sum = 0.0;
    var wsum = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let o = vec2<i32>(k & 1, k >> 1u);
        let q = clamp(vec2<i32>(base) + o, vec2<i32>(0), last);
        let s = textureLoad(shaft_tex, q, 0).rg;
        let wx = select(1.0 - fr.x, fr.x, o.x == 1);
        let wy = select(1.0 - fr.y, fr.y, o.y == 1);
        let tolerance = 0.06 * min(rd, s.y) + 0.4;
        let agree = 1.0 / (1.0 + abs(s.y - rd) / tolerance);
        let w = wx * wy * (agree * agree + 1e-3);
        sum = sum + s.x * w;
        wsum = wsum + w;
    }
    return sum / max(wsum, 1e-6);
}

// A world-space surface normal from the depth buffer: per axis, the neighbour on the side whose
// distance continues the centre's surface (so silhouettes do not bend the normal).
fn uw_normal(px: vec2<i32>, center: vec3<f32>, dist: f32) -> vec3<f32> {
    let size = vec2<i32>(camera.screen.xy) - vec2<i32>(1);
    let pl = clamp(px - vec2<i32>(1, 0), vec2<i32>(0), size);
    let pr = clamp(px + vec2<i32>(1, 0), vec2<i32>(0), size);
    let pu = clamp(px - vec2<i32>(0, 1), vec2<i32>(0), size);
    let pd = clamp(px + vec2<i32>(0, 1), vec2<i32>(0), size);
    let dl = vg_linear_depth(textureLoad(depth_tex, pl, 0));
    let dr = vg_linear_depth(textureLoad(depth_tex, pr, 0));
    let du = vg_linear_depth(textureLoad(depth_tex, pu, 0));
    let dd = vg_linear_depth(textureLoad(depth_tex, pd, 0));
    var dx: vec3<f32>;
    if (abs(dr - dist) < abs(dl - dist)) {
        dx = vg_view_position(vec2<f32>(pr) + 0.5, dr) - center;
    } else {
        dx = center - vg_view_position(vec2<f32>(pl) + 0.5, dl);
    }
    var dy: vec3<f32>;
    if (abs(dd - dist) < abs(du - dist)) {
        dy = vg_view_position(vec2<f32>(pd) + 0.5, dd) - center;
    } else {
        dy = center - vg_view_position(vec2<f32>(pu) + 0.5, du);
    }
    var n = normalize(cross(dy, dx));
    if (dot(n, center) > 0.0) {
        n = -n;
    }
    return uw_view_to_world(n);
}

// The caustic light falling on a surface at `rel` with normal `n`, as a multiplier (mean 1).
fn uw_caustics(rel: vec3<f32>, n: vec3<f32>, v: vec3<f32>, rd: f32) -> f32 {
    let light = uw.light_dir.xyz;
    let facing = smoothstep(0.3, 0.92, dot(n, light));
    let near = mix(uw.grid.w, 1.0, smoothstep(2.0, 12.0, rd));
    let far = 1.0 - smoothstep(40.0, 72.0, rd);
    let graze = smoothstep(0.03, 0.2, abs(dot(n, v)));
    var w = uw.caustic.x * uw.light_dir.w * facing * near * far * graze;
    if (w < 0.002) {
        return 1.0;
    }
    // In shadow (under a roof, an overhang or another island) there are no caustics.
    w = w * uw_open(rel + n * 0.1);
    if (w < 0.002) {
        return 1.0;
    }
    let slope = light.xz / max(light.y, 0.25);
    let p = uw.wrap.xy + (rel.xz + (uw.depth.w - rel.y) * slope) * uw.caustic.y;
    let c = textureSampleLevel(caustic_tex, lin_samp, vec3<f32>(p, uw.caustic.z), 0.0).r * 4.0;
    return mix(1.0, c, w);
}

@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(pos.xy);
    let d = textureLoad(depth_tex, px, 0);
    let sky = d >= 1.0;
    let scene = textureLoad(scene_tex, px, 0);

    var colour: vec3<f32>;
    var v: vec3<f32>;
    var rd: f32;
    if (sky) {
        let rel = uw_view_to_world(vg_view_position(pos.xy, 1.0));
        v = normalize(rel);
        rd = UW_SKY_DIST;
        if (uw.sky.a > 0.5) {
            colour = uw_sky(v);
        } else {
            // No water surface: vanilla's own sky, seen through the water's tint.
            colour = mix(uw_to_linear(scene.rgb), uw_glow(v.y), 0.55);
        }
    } else {
        let dist = vg_linear_depth(d);
        let center = vg_view_position(pos.xy, dist);
        let rel = uw_view_to_world(center);
        rd = length(rel);
        v = rel / max(rd, 1e-4);

        // Refraction wobble (off by default): read the frame a few pixels away, but never from
        // something nearer, so foreground edges do not smear into the background.
        var base = scene.rgb;
        if (uw.flags.y > 0.0) {
            let q = pos.xy / camera.screen.y * 7.0;
            let ph = uw.caustic.z * UW_TAU;
            let off = vec2<f32>(
                sin(q.y * 1.7 + ph * 3.0 + sin(q.x * 1.3 + ph * 2.0)),
                cos(q.x * 1.9 - ph * 2.0 + sin(q.y * 1.1 - ph * 3.0))
            ) * uw.flags.y * camera.screen.y / 1080.0;
            let size = vec2<i32>(camera.screen.xy) - vec2<i32>(1);
            let tp = clamp(vec2<i32>(pos.xy + off), vec2<i32>(0), size);
            let td = textureLoad(depth_tex, tp, 0);
            if (td >= d) {
                base = textureSampleLevel(scene_tex, lin_samp, (pos.xy + off) / camera.screen.xy, 0.0)
                    .rgb;
            }
        }
        colour = uw_to_linear(base);

        // The normal (four depth loads) only where caustics can show: in range and lit.
        if (uw.caustic.x > 0.0 && uw.light_dir.w > 0.0 && rd < 72.0) {
            colour = colour * uw_caustics(rel, uw_normal(px, center, dist), v, rd);
        }

        // Depth darkening: what lies well below the eye gets less light from the surface.
        let below = max(-rel.y - 2.0, 0.0);
        colour = colour * max(exp(-uw.depth.y * below), uw.depth.z);

        // Absorption and in-scatter, red first; nothing inside the clear range.
        let travelled = max(rd - uw.water.a, 0.0);
        let trans = exp(-uw.sigma.rgb * travelled);
        colour = colour * trans + uw_glow(v.y) * (1.0 - trans);

        // At the edge of the render distance the world fades into the open water behind it.
        let fade = smoothstep(uw.sigma.a, uw.depth.x, rd);
        if (fade > 0.0) {
            colour = mix(colour, uw_sky(v), fade);
        }
    }

    if (uw.light_color.a > 0.0) {
        // Screen blend: light added to a dark pixel shows, a bright pixel never clips.
        let shaft = clamp(
            uw.light_color.rgb * (uw.light_color.a * uw.light_dir.w) * uw_shafts_at(pos.xy, rd),
            vec3<f32>(0.0),
            vec3<f32>(0.9)
        );
        colour = 1.0 - (1.0 - min(colour, vec3<f32>(1.0))) * (1.0 - shaft);
    }
    return vec4<f32>(uw_to_display(colour), scene.a);
}
