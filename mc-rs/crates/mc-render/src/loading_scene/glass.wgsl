// The glass loading scene. See `glass.rs` for the overview.
//
// Units: "screen heights" (hu). Glass coordinates `g` have their origin at the screen centre,
// x to the right, y DOWN, and span y in [-0.5, 0.5]. The pane is imagined about 48 cm tall, so
// 1 hu is about 48 cm: the wire mesh's 12.7 mm pitch is about 1/38 hu.

struct Params {
    frame: vec4<f32>,   // time (s), aspect (w/h), width (px), height (px)
    layer: vec4<f32>,   // layer width (texels), layer height, mip count, seed (0..1)
    look0: vec4<f32>,   // exposure, frost, refraction, wire
    look1: vec4<f32>,   // sparkle, fan, glow, fan speed (turns/s)
    look2: vec4<f32>,   // pebbles per height, pebble depth, fog, debug view
    logo: vec4<f32>,    // the overlay logo's rect, px (x, y, w, h); zero when none
    spot: vec4<f32>,    // shiny spot x, y (screen fractions), spot size (hu), sway
    extra: vec4<f32>,   // loading progress, curtain opacity, -, -
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var tex: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
// The glass's static per-pixel fields, baked once per window size by `fs_bake` (the glass pass
// only): a = the front face's slope (xy) and its slow form (zw); b = fog, frost / 2, wire
// coverage, wire profile.
@group(0) @binding(3) var baked_a: texture_2d<f32>;
@group(0) @binding(4) var baked_b: texture_2d<f32>;

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

// ---------------------------------------------------------------------------------------------
// Hashes and noise
// ---------------------------------------------------------------------------------------------

fn pcg2d(v_in: vec2<u32>) -> vec2<u32> {
    var v = v_in * 1664525u + vec2<u32>(1013904223u);
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return v;
}

fn hash2(c: vec2<i32>, salt: u32) -> vec2<f32> {
    let h = pcg2d(vec2<u32>(bitcast<u32>(c.x) ^ salt, bitcast<u32>(c.y) ^ (salt * 747796405u)));
    return vec2<f32>(h >> vec2<u32>(8u)) * (1.0 / 16777216.0);
}

fn vnoise(p: vec2<f32>, salt: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i, salt).x;
    let b = hash2(i + vec2<i32>(1, 0), salt).x;
    let c = hash2(i + vec2<i32>(0, 1), salt).x;
    let d = hash2(i + vec2<i32>(1, 1), salt).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>, salt: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0u; i < 4u; i = i + 1u) {
        sum = sum + amp * vnoise(q, salt + i * 101u);
        q = q * 2.03 + vec2<f32>(17.1, 9.7);
        amp = amp * 0.5;
    }
    return sum;
}

// The gradient of `fbm` by central differences (for smooth slope fields).
fn fbm_grad(p: vec2<f32>, salt: u32, e: f32) -> vec2<f32> {
    let x = fbm(p + vec2<f32>(e, 0.0), salt) - fbm(p - vec2<f32>(e, 0.0), salt);
    let y = fbm(p + vec2<f32>(0.0, e), salt) - fbm(p - vec2<f32>(0.0, e), salt);
    return vec2<f32>(x, y) / (2.0 * e);
}

// ---------------------------------------------------------------------------------------------
// Full-screen triangle
// ---------------------------------------------------------------------------------------------

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VsOut;
    out.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// ---------------------------------------------------------------------------------------------
// 1. Plant sprites
// ---------------------------------------------------------------------------------------------

struct SpriteIn {
    @location(0) pos_size: vec4<f32>,
    @location(1) motion: vec4<f32>,
    @location(2) look: vec4<f32>,
    @location(3) extra: vec4<f32>,
};

struct SpriteOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) index: u32,
    @location(2) tint: vec3<f32>,
    @location(3) light: f32,
    // x: 1 for a leaf cluster (rounded), y: its random number.
    @location(4) @interpolate(flat) card: vec2<f32>,
};

// Wind: two incommensurate sines under a slow gust envelope. Gentle, never quite repeating.
fn wind(t: f32, phase: f32) -> f32 {
    let gust = 0.65 + 0.35 * sin(t * 0.53 + phase * 0.37);
    return (sin(t * 1.47 + phase) * 0.7 + sin(t * 3.13 + phase * 1.7) * 0.3) * gust * 1.4;
}

@vertex
fn vs_sprite(@builtin(vertex_index) vi: u32, s: SpriteIn) -> SpriteOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );
    let c = corners[vi];
    let t = P.frame.x;
    let sway = P.spot.w;
    let size = s.pos_size.zw;
    var p: vec2<f32>;
    var card = vec2<f32>(0.0);
    if (s.motion.w < 0.5) {
        // A rooted plant: this quad is one block of a stem whose root is below it.
        let blocks = max(s.extra.w, 1.0);
        let local_up = 1.0 - c.y;
        let up = (s.extra.z * blocks + local_up) * size.y;
        let frac = up / (blocks * size.y);
        let bend = s.motion.y * sway * wind(t, s.motion.z) * pow(frac, 1.6);
        let x = s.pos_size.x + (c.x - 0.5) * size.x + s.motion.x * up + bend;
        let y = s.pos_size.y - local_up * size.y + abs(bend) * 0.12;
        p = vec2<f32>(x, y);
    } else {
        // A leaf cluster: turns about its centre and drifts a little.
        let w1 = wind(t, s.motion.z);
        let w2 = wind(t * 1.31 + 4.0, s.motion.z * 1.7);
        let angle = s.motion.x + w1 * 0.1 * sway;
        let ca = cos(angle);
        let sa = sin(angle);
        let l = (c - 0.5) * size;
        let r = vec2<f32>(l.x * ca - l.y * sa, l.x * sa + l.y * ca);
        p = s.pos_size.xy + r + vec2<f32>(w1, w2 * 0.5) * s.motion.y * sway;
        card = vec2<f32>(1.0, s.extra.z);
    }
    var out: SpriteOut;
    out.pos = vec4<f32>(p.x / (P.frame.y * 0.5), -p.y / 0.5, 0.0, 1.0);
    let u = select(c.x, 1.0 - c.x, s.extra.x > 0.5);
    out.uv = vec2<f32>(u, c.y);
    out.index = u32(s.look.w + 0.5);
    out.tint = s.look.rgb;
    // Daylight from above and behind: brighter toward the top of the screen.
    out.light = s.extra.y * (1.0 - 0.45 * clamp(p.y + 0.2, 0.0, 1.0));
    out.card = card;
    return out;
}

@fragment
fn fs_sprite(in: SpriteOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv, in.index);
    var keep = texel.a >= 0.5;
    var shade = 1.0;
    if (in.card.x > 0.5) {
        // A leaf cluster is a rounded, ragged clump of the leaves texture, not a square: seen
        // through frosted glass a square block would read as a tile.
        let d = in.uv - 0.5;
        let ang = atan2(d.y, d.x);
        let r = length(d) * 2.0;
        let k = in.card.y * 40.0;
        let edge = 0.8 + 0.09 * sin(ang * 3.0 + k) + 0.06 * sin(ang * 7.0 + k * 1.7);
        keep = keep && r < edge;
        // A little self-shading toward the clump's lower edge.
        shade = 1.0 - 0.3 * smoothstep(-0.2, 0.5, d.y) * r;
    }
    if (!keep) {
        discard;
    }
    return vec4<f32>(texel.rgb * in.tint * in.light * shade, 1.0);
}

// ---------------------------------------------------------------------------------------------
// 2. Blur pyramid: one half-size step of all three layers (a dual-filter downsample)
// ---------------------------------------------------------------------------------------------

struct DownOut {
    @location(0) near: vec4<f32>,
    @location(1) mid: vec4<f32>,
    @location(2) far: vec4<f32>,
};

fn down(layer: i32, uv: vec2<f32>, texel: vec2<f32>) -> vec4<f32> {
    var s = textureSampleLevel(tex, samp, uv, layer, 0.0) * 4.0;
    s = s + textureSampleLevel(tex, samp, uv + texel * vec2<f32>(-1.0, -1.0), layer, 0.0);
    s = s + textureSampleLevel(tex, samp, uv + texel * vec2<f32>(1.0, -1.0), layer, 0.0);
    s = s + textureSampleLevel(tex, samp, uv + texel * vec2<f32>(-1.0, 1.0), layer, 0.0);
    s = s + textureSampleLevel(tex, samp, uv + texel * vec2<f32>(1.0, 1.0), layer, 0.0);
    return s * 0.125;
}

@fragment
fn fs_down(in: VsOut) -> DownOut {
    let src = vec2<f32>(textureDimensions(tex, 0));
    let dst = max(floor(src * 0.5), vec2<f32>(1.0));
    let uv = in.pos.xy / dst;
    let texel = 1.0 / src;
    var out: DownOut;
    out.near = down(0, uv, texel);
    out.mid = down(1, uv, texel);
    out.far = down(2, uv, texel);
    return out;
}

// ---------------------------------------------------------------------------------------------
// 3. The glass
// ---------------------------------------------------------------------------------------------

// Smooth Voronoi cells of unit size at `q`: x = height (about the squared distance to the
// nearest cell centre, so each cell is a shallow dimple), yz = its gradient.
fn dimples(q: vec2<f32>, salt: u32, k: f32) -> vec3<f32> {
    let i = vec2<i32>(floor(q));
    let f = fract(q);
    var sum = 0.0;
    var grad = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let c = vec2<i32>(x, y);
            let o = hash2(i + c, salt);
            let d = f - (vec2<f32>(c) + 0.1 + 0.8 * o);
            let d2 = dot(d, d);
            let w = exp(-k * d2);
            sum = sum + w;
            grad = grad + w * 2.0 * d;
        }
    }
    sum = max(sum, 1e-6);
    return vec3<f32>(-log(sum) / k, grad / sum);
}

// The front face's slope field (dh/dg, y down) at `g`: fine pebbling (two octaves of cells), a
// medium ripple that makes edges behind the glass wobble, and the sheet's own slow waviness.
// `.zw` is the slow waviness alone: the smooth form the coherent reflection sees.
fn surface(g: vec2<f32>) -> vec4<f32> {
    let n = P.look2.x;
    let depth = P.look2.y;
    let d1 = dimples(g * n, 11u, 14.0);
    let d2 = dimples(g * n * 2.37 + vec2<f32>(3.7, 1.9), 23u, 9.0);
    let ripple = fbm_grad(g * 9.0, 31u, 0.02) * 0.012;
    let wave = fbm_grad(g * 1.6, 41u, 0.01) * 0.012;
    let bumps = (d1.yz * 0.13 + d2.yz * 0.05) * depth;
    return vec4<f32>(bumps + ripple + wave, wave + ripple * 0.3);
}

// One mip level of a plant layer through a cubic B-spline (four bilinear taps): smooth even on
// the coarsest levels, where plain bilinear filtering shows the pyramid's squares.
fn bspline_level(layer: i32, uv: vec2<f32>, level: f32) -> vec4<f32> {
    let size = max(floor(P.layer.xy / exp2(level)), vec2<f32>(1.0));
    let x = uv * size - 0.5;
    let i = floor(x);
    let f = x - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let h0 = (i - 1.0 + w1 / g0 + 0.5) / size;
    let h1 = (i + 1.0 + w3 / g1 + 0.5) / size;
    let a = textureSampleLevel(tex, samp, vec2<f32>(h0.x, h0.y), layer, level);
    let b = textureSampleLevel(tex, samp, vec2<f32>(h1.x, h0.y), layer, level);
    let c = textureSampleLevel(tex, samp, vec2<f32>(h0.x, h1.y), layer, level);
    let d = textureSampleLevel(tex, samp, vec2<f32>(h1.x, h1.y), layer, level);
    return (a * g0.x + b * g1.x) * g0.y + (c * g0.x + d * g1.x) * g1.y;
}

// A plant layer blurred by `lod` (fractional): two B-spline levels, blended.
fn layer_at(layer: i32, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    let lo = floor(lod);
    let a = bspline_level(layer, uv, lo);
    if (lod - lo < 0.02) {
        return a;
    }
    let b = bspline_level(layer, uv, min(lo + 1.0, P.layer.z - 1.0));
    return mix(a, b, lod - lo);
}

// How brightly each layer is lit (near, mid, far): the leaves are lit from behind and glow
// against the shade between them.
const LAYER_GAIN: array<f32, 3> = array<f32, 3>(2.4, 2.1, 1.35);

// What lies behind all the plants: daylight, bright toward the upper left, deep green toward
// the lower right (always seen through heavy blur, so it is smooth).
fn backdrop(uv: vec2<f32>) -> vec3<f32> {
    let aspect = P.frame.y;
    let p = (uv - vec2<f32>(0.08, 0.02)) * vec2<f32>(aspect, 1.0);
    let r2 = dot(p, p);
    let sky = vec3<f32>(0.78, 0.84, 0.66);
    let deep = vec3<f32>(0.015, 0.04, 0.012);
    let k = exp(-r2 * 2.2);
    let mid = vec3<f32>(0.10, 0.18, 0.06);
    return mix(mix(deep, mid, smoothstep(0.0, 0.45, k)), sky, smoothstep(0.3, 1.0, k));
}

// ---- The room in front of the glass, for reflections -----------------------------------------
//
// The eye is EYE_Z hu in front of the pane's centre (world: x right, y up, z toward the viewer).
// Up and behind the viewer a ceiling fan turns, with a glowing bowl light at its hub, under a
// brightly lit ceiling: the fan's mirror image is the shiny spot. Its plane is tipped part of the
// way toward the glass, so the reflection shows a fan, not a sliver.

const EYE_Z: f32 = 1.35;

struct Room {
    hub: vec3<f32>,
    // The fan plane's normal and two axes in it.
    n: vec3<f32>,
    e1: vec3<f32>,
    e2: vec3<f32>,
    // The mirror direction at the spot (toward the fan).
    dir: vec3<f32>,
};

fn spot_point() -> vec3<f32> {
    let aspect = P.frame.y;
    let g = vec2<f32>((P.spot.x - 0.5) * aspect, P.spot.y - 0.5);
    return vec3<f32>(g.x, -g.y, 0.0);
}

fn room() -> Room {
    let ps = spot_point();
    let v = normalize(ps - vec3<f32>(0.0, 0.0, EYE_Z));
    let r = vec3<f32>(v.x, v.y, -v.z);
    var out: Room;
    out.dir = r;
    out.hub = ps + r * 1.6;
    out.n = normalize(vec3<f32>(0.0, 1.0, 0.0) + r * 0.9);
    out.e1 = normalize(cross(out.n, vec3<f32>(0.0, 0.0, 1.0)));
    out.e2 = cross(out.n, out.e1);
    return out;
}

// The fan's blade coverage at a point of its plane (`local`, hu, relative to the hub),
// motion-blurred over a fixed 1/90 s shutter so it never strobes at any frame rate.
fn fan_cover(local: vec2<f32>, t: f32) -> f32 {
    let r = length(local);
    let size = P.spot.z * 2.2;
    let hub_r = 0.13 * size;
    let tip = size;
    let hub = 1.0 - smoothstep(hub_r * 0.85, hub_r, r);
    let inside = smoothstep(hub_r * 0.9, hub_r * 1.3, r)
        * (1.0 - smoothstep(tip * 0.92, tip, r));
    if (inside <= 0.0) {
        return hub;
    }
    let blades = 5.0;
    let period = TAU / blades;
    let turn = TAU * P.look1.w;
    let angle = atan2(local.y, local.x) - turn * t;
    // Offset from the nearest blade's centre line, radians.
    let x = angle - period * floor(angle / period + 0.5);
    // A paddle, a little wider toward the tip.
    let width = size * (0.085 + 0.05 * clamp((r - hub_r) / (tip - hub_r), 0.0, 1.0));
    let half_angle = atan(width / max(r, 0.02));
    // The blade's angular extent convolved with the shutter's sweep.
    let shutter = max(turn / 90.0, 0.02);
    let covered = clamp(
        (min(x + shutter * 0.5, half_angle) - max(x - shutter * 0.5, -half_angle)) / shutter,
        0.0,
        1.0
    );
    return max(hub, inside * covered);
}

// Radiance seen along the reflected ray `rd` from glass point `po`. `core` scales the small,
// intense source at the heart of the glow (the coherent reflection leaves it out: on the smooth
// form it would be one hard blob; on the pebbles it is glitter).
fn room_radiance(po: vec3<f32>, rd: vec3<f32>, rm: Room, t: f32, core_gain: f32) -> vec3<f32> {
    let warm = vec3<f32>(1.0, 0.95, 0.86);
    let dim = vec3<f32>(0.02, 0.02, 0.018);
    // Below the horizon of the reflection: the dim far side of the room (a soft edge, so the
    // pebbles never draw a line across the pane).
    let above = smoothstep(-0.02, 0.18, rd.y);
    if (above <= 0.0) {
        return dim;
    }
    // The lit ceiling around the fan (a broad glow about the mirror direction) and, at its heart,
    // the bright source (small and intense: on the pebbles it breaks into glitter).
    let ca = dot(rd, rm.dir);
    let a2 = max(2.0 * (1.0 - ca), 0.0);
    let spread = P.spot.z * 1.25;
    let core = P.spot.z * 0.6;
    var c = dim + warm * (4.5 * exp(-a2 / (spread * spread))
        + 18.0 * core_gain * exp(-a2 / (core * core))
        + 0.3 * exp(-a2 / (spread * spread * 5.0)));
    // The fan, in front of the ceiling.
    let denom = dot(rd, rm.n);
    if (abs(denom) > 1e-3) {
        let tf = dot(rm.hub - po, rm.n) / denom;
        if (tf > 0.0) {
            let h = po + rd * tf - rm.hub;
            let local = vec2<f32>(dot(h, rm.e1), dot(h, rm.e2));
            let cover = fan_cover(local, t);
            c = mix(c, vec3<f32>(0.045, 0.04, 0.035), cover * P.look1.y);
            // The bowl light under the hub.
            let bowl = 1.0 - smoothstep(0.075, 0.1, length(local) / (P.spot.z * 2.2));
            c = c + warm * bowl * 9.0;
        }
    }
    return mix(dim, c, above);
}

fn fresnel(cos_theta: f32) -> f32 {
    let m = 1.0 - clamp(cos_theta, 0.0, 1.0);
    return 0.04 + 0.96 * m * m * m * m * m;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn encode_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

struct BakeOut {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
};

// The static half of the glass, once per window size: the pebbled face, the fog, the frost and
// the wire mesh's mask. Everything here is noise and geometry that never moves.
@fragment
fn fs_bake(in: VsOut) -> BakeOut {
    let res = P.frame.zw;
    let px = in.pos.xy;
    let uv = px / res;
    let g = (px - 0.5 * res) / res.y;
    let surf = surface(g);
    // Condensation fog on the front face: thick at the left edge, in patches elsewhere.
    let fog_noise = fbm(g * vec2<f32>(1.1, 1.6) + vec2<f32>(2.0, 7.0), 61u);
    var fog = smoothstep(0.42, 0.95, fog_noise + (0.17 - uv.x) * 2.3 + (uv.y - 0.6) * 0.2);
    fog = clamp(fog, 0.0, 1.0) * P.look2.z;
    // The etched back face: how much it scatters (more under the fog, a little everywhere).
    let frost = clamp(0.35 + fog * 0.9 + (fbm(g * 3.0, 71u) - 0.5) * 0.4, 0.15, 1.3)
        * P.look0.y;
    // The wire mesh, set in the middle of the pane. Seen through the front face it wobbles with
    // the slow ripple, barely with the pebbles.
    let pitch = 1.0 / 27.0;
    let wobble = surf.zw * 0.05 + surf.xy * 0.0011;
    let wq = (g + wobble) / pitch;
    let dist2 = abs(fract(wq + 0.5) - 0.5) * pitch * res.y;
    let dist = min(dist2.x, dist2.y);
    let half_w = max(0.62 * res.y / 1080.0, 0.55);
    let aa = 0.7;
    let cover = 1.0 - smoothstep(half_w - aa, half_w + aa, dist);
    let across = clamp(dist / (half_w + 0.6), 0.0, 1.0);
    let profile = sqrt(max(1.0 - across * across, 0.0));
    // Each length of wire between two crossings catches the light a little differently.
    let vertical = dist2.x < dist2.y;
    let segment = select(
        vec2<i32>(i32(floor(wq.x)), i32(floor(wq.y + 0.5))),
        vec2<i32>(i32(floor(wq.x + 0.5)), i32(floor(wq.y))),
        vertical
    );
    let variation = 0.82 + 0.36 * hash2(segment, select(17u, 29u, vertical)).x;
    let shade = (0.75 + 0.25 * profile) * variation / 1.2;
    var out: BakeOut;
    out.a = surf;
    out.b = vec4<f32>(fog, clamp(frost * 0.5, 0.0, 1.0), cover, clamp(shade, 0.0, 1.0));
    return out;
}

@fragment
fn fs_glass(in: VsOut) -> @location(0) vec4<f32> {
    let res = P.frame.zw;
    let px = in.pos.xy;
    let uv = px / res;
    let aspect = P.frame.y;
    let g = (px - 0.5 * res) / res.y;
    let t = P.frame.x;
    let debug = P.look2.w;

    // --- The surface, fog, frost and wire mask (baked) ------------------------------------
    let ip = vec2<i32>(px);
    let surf = textureLoad(baked_a, ip, 0);
    let fields = textureLoad(baked_b, ip, 0);
    let slope = surf.xy;
    let fog = fields.x;
    let frost = fields.y * 2.0;
    let cover = fields.z;
    let wire_shade = fields.w * 1.2;

    // --- Transmission: three plant layers over the daylight backdrop ----------------------
    let refr = 0.6 * P.look0.z;
    let depths = vec3<f32>(0.045, 0.11, 0.45);
    let scatter = 0.12 + 0.2 * frost;
    let to_uv = vec2<f32>(1.0 / aspect, 1.0);
    let layer_h = P.layer.y;
    let max_lod = P.layer.z - 1.0;

    var col = backdrop(uv + slope * refr * 1.5 * to_uv);
    for (var i = 2; i >= 0; i = i - 1) {
        let d = depths[i];
        let off = slope * refr * d * to_uv;
        let blur_hu = scatter * d + 0.0012;
        let lod = clamp(log2(max(blur_hu * layer_h, 1.0)), 0.0, max_lod);
        let s = layer_at(i, uv + off, lod);
        col = s.rgb * LAYER_GAIN[i] + col * (1.0 - s.a);
    }
    if (debug > 0.5 && debug < 1.5) {
        var flat_col = backdrop(uv);
        for (var i = 2; i >= 0; i = i - 1) {
            let s = textureSampleLevel(tex, samp, uv, i, 0.0);
            flat_col = s.rgb + flat_col * (1.0 - s.a);
        }
        return vec4<f32>(encode_srgb(clamp(flat_col, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
    }
    // Glass absorbs a little and tints faintly green.
    col = col * vec3<f32>(0.9, 0.95, 0.91);

    // The light scattered widely by the etched face: the average glow of everything behind.
    var haze = backdrop(uv * 0.7 + 0.15);
    for (var i = 2; i >= 0; i = i - 1) {
        let s = bspline_level(i, uv, max_lod);
        haze = s.rgb * LAYER_GAIN[i] + haze * (1.0 - s.a);
    }
    col = mix(col, haze, clamp(frost * 0.15, 0.0, 0.5));

    // --- The wire mesh (its mask is baked) ---------------------------------------------
    let ps = spot_point();
    let po = vec3<f32>(g.x, -g.y, 0.0);
    let dspot = length(vec2<f32>(po.x - ps.x, po.y - ps.y));
    let spot_fall = exp(-dspot * dspot / (P.spot.z * P.spot.z * 1.6));
    // Steel catching the room's light: a light grey-green line, brighter near the spot.
    let wire_light = 0.32 + 0.4 * luma(haze) + 0.5 * spot_fall;
    let wire = vec3<f32>(0.55, 0.63, 0.55) * wire_light * wire_shade;
    col = mix(col, wire, cover * P.look0.w * 0.85);

    // --- Reflections off the front face ------------------------------------------------
    let rm = room();
    let v = normalize(po - vec3<f32>(0.0, 0.0, EYE_Z));
    let n_bump = normalize(vec3<f32>(-slope.x, slope.y, 1.0));
    let n_smooth = normalize(vec3<f32>(-surf.z, surf.w, 1.0));
    let rb = reflect(v, n_bump);
    let spec_bump = room_radiance(po, rb, rm, t, 1.0) * fresnel(dot(-v, n_bump));
    // The coherent part: the sheet's smooth form reflects a soft image of the light and the fan.
    let rs = reflect(v, n_smooth);
    let side = normalize(cross(rs, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(side, rs);
    let cone = 0.02;
    let rot = hash2(vec2<i32>(px), 77u).x * TAU;
    var soft = vec3<f32>(0.0);
    for (var k = 0; k < 4; k = k + 1) {
        let a = rot + f32(k) * (TAU / 4.0);
        let dir = normalize(rs + (side * cos(a) + up * sin(a)) * cone);
        soft = soft + room_radiance(po, dir, rm, t, 0.3);
    }
    soft = soft * 0.25 * fresnel(dot(-v, n_smooth));
    // The glitter belongs to the shiny spot: away from it the pebbles only catch the dim room.
    let glitter = 0.05 + 0.95 * exp(-dspot * dspot / (P.spot.z * P.spot.z * 1.9));
    let reflection = mix(spec_bump * P.look1.x * glitter, soft * 1.35, 0.36);

    // --- Condensation fog on the front: veils everything behind it, wires included ----
    let fog_light = 0.55 + 0.35 * spot_fall + 0.25 * luma(haze);
    let fog_col = vec3<f32>(0.7, 0.67, 0.53) * fog_light;
    col = mix(col, fog_col, fog * 0.8);

    col = col + reflection * (1.0 - fog * 0.5);
    // A soft glow around the spot.
    let glow = (exp(-dspot * dspot / 0.01) * 0.12 + exp(-dspot * dspot / 0.12) * 0.08)
        * P.look1.z;
    col = col + vec3<f32>(1.0, 0.95, 0.86) * glow;

    if (debug > 1.5 && debug < 2.5) {
        return vec4<f32>(slope * 2.0 + 0.5, 0.5, 1.0);
    }
    if (debug > 2.5 && debug < 3.5) {
        return vec4<f32>(fog, frost * 0.7, 0.0, 1.0);
    }
    if (debug > 3.5) {
        return vec4<f32>(encode_srgb(clamp(reflection, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
    }

    // --- The overlay's logo: keep the area behind it a little calmer and darker ---------
    if (P.logo.z > 0.0) {
        let c = P.logo.xy + P.logo.zw * 0.5;
        let hs = P.logo.zw * 0.5 + vec2<f32>(P.logo.w * 0.9);
        let q = abs(px - c) / hs;
        let m = 1.0 - smoothstep(0.35, 1.0, max(q.x, q.y));
        col = col * (1.0 - 0.16 * m);
    }

    // --- Lens falloff, tone curve, encode, dither -------------------------------------
    let vq = (uv - 0.5) * vec2<f32>(aspect, 1.0);
    col = col * (1.0 - 0.25 * smoothstep(0.3, 1.15, length(vq)));
    col = col * P.look0.x;
    col = vec3<f32>(1.0) - exp(-col * 1.3);
    var out = encode_srgb(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)));
    // A gentle S-curve in display space: deeper shade, the same highlights.
    out = mix(out, out * out * (3.0 - 2.0 * out), 0.3);
    let dither = (hash2(vec2<i32>(px), 91u).x + hash2(vec2<i32>(px), 92u).x - 1.0) / 255.0;
    out = out + vec3<f32>(dither);
    return vec4<f32>(out, 1.0);
}
