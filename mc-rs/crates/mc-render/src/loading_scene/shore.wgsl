// The neon night shore: waves breaking on a beach of vanilla sand
// blocks at night, their crests drawn in neon light, a sea fog that swallows the horizon, and
// film grain. One full-screen ray cast (`fs_scene`) into an HDR target, a small bloom chain, and a
// composite that tone maps, grades and adds the grain (`fs_composite`).
//
// The shore frame: `u` = world x, positive out to sea; `v` = world z, along the shore. The swash
// terrace is the band of sand blocks whose tops are at y = 0, from its seaward edge (u ~ 0) up the
// beach to the first step (u ~ -12). Still water sits at y = SEA, a little under the terrace.
//
// The waves are analytic, so every pixel can ask any wave where it is: wave `i` leaves the fog at
// `U0` and runs in at `C` blocks a second, slightly oblique (a delay of `obl` seconds per block of
// `v`, so the break runs along the shore). It breaks near `BREAK`, reaches the terrace at `tau0`
// and runs up it as a thin sheet that decelerates (a ballistic swash) to its farthest point, the
// run-up `smax(v)`, at `taum`; then it drains back. Each wave carries a ribbon of five neon bands
// on its front: cyan first, then blue, yellow, orange and pink behind it.
//
// The colour push (the user's rule): as the swash comes to rest at its farthest point, two of the
// trailing colours surge forward through the bands ahead of them, each with its own lobed wave
// shape. They may cover the cyan front band and, once the front has started to drain back, pass
// where it is now, but never the border the cyan front drew: the farthest line it reached so far.
// Where they arrive they take the colour away (they replace it, they do not add to it). The cyan
// line left at that border fades as it is taken over or as the wave drains.

const PI: f32 = 3.14159265;
const TAU: f32 = 6.28318531;

// ---- the shore
const SEA: f32 = -0.15;          // still water (the terrace's top is y = 0)
const C: f32 = 3.8;              // wave speed, blocks per second
const U0: f32 = 62.0;            // where a wave starts, out in the fog
const BREAK: f32 = 9.0;          // where it breaks
const PERIOD: f32 = 5.4;         // seconds between waves
const OBL: f32 = 0.055;          // mean oblique delay, seconds per block along the shore
const NWAVES: i32 = 7;           // waves evaluated per pixel

// ---- the height map (see `shore.rs`)
const HMAP_X0: i32 = -64;
const HMAP_Z0: i32 = -64;
const HMAP_W: i32 = 160;
const HMAP_H: i32 = 256;
const H_MAX: f32 = 4.0;
const H_MIN: f32 = -6.0;
const T_MAX: f32 = 240.0;

// ---- the neon ribbon: band widths (blocks, at full size) and offsets behind the front
const W0: f32 = 0.30;
const W1: f32 = 0.20;
const W2: f32 = 0.34;
const W3: f32 = 0.26;
const W4: f32 = 0.28;

struct Frame {
    cam_pos: vec4<f32>,    // eye; w = scene time, seconds
    cam_fwd: vec4<f32>,    // forward; w = debug mode (0 scene, 1 band diagram, 2 test channels)
    cam_right: vec4<f32>,  // right * tan(half fov x)
    cam_up: vec4<f32>,     // up * tan(half fov y)
    screen: vec4<f32>,     // scene target: w, h, 1/w, 1/h
    out_size: vec4<f32>,   // frame: w, h, 1/w, 1/h
    look: vec4<f32>,       // exposure, grain, bloom, encode to sRGB (1) or not (0)
    extra: vec4<f32>,      // logo shade, grain frame, push clamp (1 on), diagram wave index
    map: vec4<f32>,        // diagram window: u at the top, u span, v at the left, v span
    logo: vec4<f32>,       // the loading screen's logo: x, y, w, h in frame pixels (0 = none)
    table: vec4<f32>,      // x = the first wave in `waves`; y = no block rises above the water
                           // seaward of this x
    waves: array<vec4<f32>, 64>,  // WAVE_SLOTS waves, 4 vec4 each (see `wave`)
}

const WAVE_SLOTS: i32 = 16;

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var blocks: texture_2d_array<f32>;
@group(0) @binding(2) var block_samp: sampler;
@group(0) @binding(3) var heights: texture_2d<u32>;
@group(0) @binding(4) var noise: texture_2d<f32>;
@group(0) @binding(5) var noise_samp: sampler;
@group(0) @binding(6) var lin_samp: sampler;

@group(1) @binding(0) var src_a: texture_2d<f32>;
@group(1) @binding(1) var src_b: texture_2d<f32>;

struct FsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> FsOut {
    let x = f32(i32(i & 1u) * 4 - 1);
    let y = f32(i32(i >> 1u) * 4 - 1);
    var o: FsOut;
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    o.ndc = vec2<f32>(x, y);
    return o;
}

// ---------------------------------------------------------------------------- small helpers

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Soft membership of `x` in `[lo, hi]` with edges `e` wide.
fn cover(x: f32, lo: f32, hi: f32, e: f32) -> f32 {
    if hi <= lo {
        return 0.0;
    }
    return smoothstep(lo - e, lo + e, x) * (1.0 - smoothstep(hi - e, hi + e, x));
}

// `cover` with its own softness at each end.
fn cover2(x: f32, lo: f32, hi: f32, e_lo: f32, e_hi: f32) -> f32 {
    if hi <= lo {
        return 0.0;
    }
    return smoothstep(lo - e_lo, lo + e_lo, x) * (1.0 - smoothstep(hi - e_hi, hi + e_hi, x));
}

// Distance from `x` to the interval `[lo, hi]` (0 inside).
fn dist_to(x: f32, lo: f32, hi: f32) -> f32 {
    return max(max(lo - x, x - hi), 0.0);
}

fn noise_at(p: vec2<f32>, lod: f32) -> vec4<f32> {
    return textureSampleLevel(noise, noise_samp, p, lod);
}

// The noise texture's mip for a sample whose coordinates (in tiles) change by `per_px` a pixel.
fn noise_lod(per_px: f32) -> f32 {
    return max(log2(max(per_px * 256.0, 1e-4)), 0.0);
}

// ---------------------------------------------------------------------------- the camera

fn view_ray(ndc: vec2<f32>) -> vec3<f32> {
    return normalize(F.cam_fwd.xyz + ndc.x * F.cam_right.xyz + ndc.y * F.cam_up.xyz);
}

// One pixel's angular size (radians, vertical) in the scene target.
fn pixel_angle() -> f32 {
    return 2.0 * length(F.cam_up.xyz) * F.screen.w;
}

// ---------------------------------------------------------------------------- the night

const MOON_DIR: vec3<f32> = vec3<f32>(0.36, 0.58, 0.73);

// The night's light on a block face, Minecraft style: one dim, cool level of sky light, shaded
// per face (top 1.0, z sides 0.8, x sides 0.6).
fn night_light() -> vec3<f32> {
    return vec3<f32>(0.080, 0.082, 0.118);
}

fn moon_light() -> vec3<f32> {
    return vec3<f32>(0.060, 0.066, 0.100);
}

fn sky_ambient() -> vec3<f32> {
    return vec3<f32>(0.016, 0.018, 0.030);
}

// The fog's colour seen along `d`: a dusky blue-violet night fog, luminous where the eye looks
// out level (the moonlit fog bank that hides the horizon), darker looking down or up.
fn fog_color(d: vec3<f32>) -> vec3<f32> {
    let base = vec3<f32>(0.021, 0.021, 0.040);
    let level = exp(-abs(d.y - 0.02) * 3.0);
    let moon = pow(max(dot(d, normalize(MOON_DIR)), 0.0), 4.0);
    return base * (0.55 + 0.95 * level) + vec3<f32>(0.016, 0.015, 0.028) * moon;
}

// How much fog lies between the eye and a point `t` along `d`: an exponential sea fog that thins
// with height, plus a little everywhere.
fn fog_amount(o: vec3<f32>, d: vec3<f32>, t: f32) -> f32 {
    let dens = 0.060;
    let height = 8.0;
    let base = 0.0075;
    let k = -d.y / height;
    let at_eye = dens * exp(-(o.y - SEA) / height);
    var integral: f32;
    if abs(k) < 1e-4 {
        integral = at_eye * t;
    } else {
        integral = at_eye * (exp(min(k * t, 30.0)) - 1.0) / k;
    }
    integral = integral + base * t;
    return 1.0 - exp(-integral);
}

// ---------------------------------------------------------------------------- the waves

struct Wave {
    t0: f32,
    obl: f32,
    smax: f32,
    lam: f32,
    ph1: f32,
    ph2: f32,
    ph3: f32,
    gain: f32,
    p1: i32,
    p2: i32,
    lam1: f32,
    lam2: f32,
    ph4: f32,
    ph5: f32,
}

// Wave `i`'s parameters, from the table the CPU fills each frame (`wave_params` in `shore.rs`,
// the one place their formulas live). A wave outside the table never arrives.
fn wave(i: i32) -> Wave {
    var w: Wave;
    let k = i - i32(F.table.x);
    if k < 0 || k >= WAVE_SLOTS {
        w.t0 = 1e9;
        w.obl = OBL;
        w.smax = 8.0;
        w.lam = 12.0;
        w.lam1 = 6.0;
        w.lam2 = 6.0;
        w.p1 = 2;
        w.p2 = 4;
        return w;
    }
    let a = F.waves[k * 4];
    let b = F.waves[k * 4 + 1];
    let c = F.waves[k * 4 + 2];
    let d = F.waves[k * 4 + 3];
    w.t0 = a.x;
    w.obl = a.y;
    w.smax = a.z;
    w.lam = a.w;
    w.ph1 = b.x;
    w.ph2 = b.y;
    w.ph3 = b.z;
    w.gain = b.w;
    w.p1 = i32(c.x);
    w.p2 = i32(c.y);
    w.lam1 = c.z;
    w.lam2 = c.w;
    w.ph4 = d.x;
    w.ph5 = d.y;
    return w;
}

// Where wave `w` is at `v` along the shore at time `t`.
struct Local {
    tau: f32,    // the wave's clock here
    front: f32,  // u of its front: the crest, then the bore, then the swash edge
    tau0: f32,   // the front reaches the terrace (u = 0)
    taum: f32,   // the swash's farthest point
    smax: f32,   // how far up the terrace it runs here
    a: f32,      // the uprush's deceleration
    ab: f32,     // the backwash's acceleration
}

// How far up the terrace wave `w` runs at `v`: its mean run-up shaped into broad, uneven tongues.
fn run_up(w: Wave, v: f32) -> f32 {
    let s1 = sin(TAU * v / w.lam + w.ph3);
    let s2 = sin(TAU * v / (0.41 * w.lam) + 2.0 * w.ph3);
    let lobe = 0.62 * s1 + 0.26 * s2 + 0.12 * sin(TAU * v / (0.17 * w.lam) + 3.0 * w.ph3);
    return w.smax * (1.0 + 0.10 * lobe);
}

fn local(w: Wave, v: f32, t: f32) -> Local {
    var l: Local;
    l.tau = t - w.t0 - v * w.obl;
    let wobble = 1.2 * sin(v * 0.083 + w.ph1) + 0.6 * sin(v * 0.21 + w.ph2);
    let start = U0 + wobble;
    l.tau0 = start / C;
    // The run-up's shape only matters once the wave is on the terrace; out at sea nothing reads
    // it, and the mean stands in (three sines saved for most of the waves at a pixel).
    l.smax = w.smax;
    if l.tau >= l.tau0 {
        l.smax = run_up(w, v);
    }
    l.a = C * C / (2.0 * l.smax);
    l.ab = l.a * 1.3;
    l.taum = l.tau0 + C / l.a;
    if l.tau < l.tau0 {
        l.front = start - C * l.tau;
    } else if l.tau < l.taum {
        let x = l.tau - l.tau0;
        l.front = -(C * x - 0.5 * l.a * x * x);
    } else {
        let x = l.tau - l.taum;
        l.front = -(l.smax - 0.5 * l.ab * x * x);
    }
    return l;
}

// Whether wave `w` can matter at `v` at time `t` at all: spawned, and not long gone (the latest
// any wave rests is about (U0 + 2) / C + 2 * 10.6 / C seconds in; its sand dries for 9 more).
// Cheap, so the trigonometry of `local` is only paid for live waves.
fn alive(w: Wave, v: f32, t: f32) -> bool {
    let tau = t - w.t0 - v * w.obl;
    return tau >= 0.0 && tau < (U0 + 2.0 + 2.0 * 10.6) / C + 9.5;
}

// A wave drops out of the NWAVES evaluated per pixel at least 30.5 s after it spawned (after
// every swash has drained, about 28 s at the latest): what it leaves behind (wet sand, stranded
// foam) has faded smoothly by then, so nothing changes in a jump.
fn late_fade(tau: f32) -> f32 {
    return 1.0 - smoothstep(27.0, 30.0, tau);
}

// The index of the newest wave that can matter at `v` at time `t`.
fn newest_wave(v: f32, t: f32) -> i32 {
    return i32(floor((t - v * OBL) / PERIOD)) + 1;
}

// How big the wave is at its front (for the water's height), by stage.
fn wave_height(l: Local) -> f32 {
    if l.tau < 0.0 {
        return 0.0;
    }
    let grow = smoothstep(U0, BREAK + 2.0, l.front);
    let swell = mix(0.22, 0.80, grow * grow);
    let broken = smoothstep(BREAK, BREAK - 4.0, l.front);
    let bore = mix(0.75, 0.35, smoothstep(BREAK - 1.0, 0.0, l.front));
    return mix(swell, bore, broken);
}

// The water's height at `p` (x, z) and time `t`.
fn water_height(p: vec2<f32>, t: f32) -> f32 {
    let u = p.x;
    let v = p.y;
    let n = newest_wave(v, t);
    var sea = SEA;
    var film = -10.0;
    for (var k = 0; k < NWAVES; k = k + 1) {
        let w = wave(n - k);
        if !alive(w, v, t) {
            continue;
        }
        let l = local(w, v, t);
        if l.tau > l.taum + 8.0 {
            continue;
        }
        let b = u - l.front;
        if l.tau < l.tau0 {
            // A swell, then a bore over the sea: a hump with a steep front.
            let h = wave_height(l);
            let broken = smoothstep(BREAK + 2.0, BREAK - 3.0, l.front);
            // The front steepens towards the break, but only so far: the eye looks down at about
            // 20 degrees, and a face steeper than that would need a full ray march. The breaking
            // look comes from the foam and the neon.
            let wf = mix(3.4, 1.7, smoothstep(U0 * 0.5, BREAK, l.front));
            let wb = mix(5.5, 4.0, broken);
            var prof: f32;
            if b < 0.0 {
                prof = exp(-(b * b) / (wf * wf));
            } else {
                let hump = exp(-(b * b) / (wb * wb));
                let shelf = (0.25 + 0.1 * exp(-b / 3.0)) * exp(-b / 16.0);
                prof = mix(hump, max(hump, shelf), broken);
            }
            sea = sea + h * prof;
        } else {
            // The swash: a thin sheet over the terrace, thickest at its seaward end.
            let age = l.tau - l.tau0;
            let depth = 0.04 + 0.26 * exp(-age / 2.2);
            if b >= -0.02 && u < 1.6 {
                let reach = max(-l.front, 0.6);
                let s = clamp((u - l.front) / reach, 0.0, 1.0);
                let lip = 0.035 * exp(-(b * b) / 0.12);
                let top = depth * sqrt(s) + 0.02 + lip;
                // The sheet lies on the terrace only: just past its seaward edge it pours down
                // into the sea, so it never lifts a plateau over the water out there (whose
                // crease against the next bore tore the ray's intersection, row by row).
                film = max(film, top - smoothstep(0.0, 1.5, u) * (top - SEA + 0.1));
            }
            // The water it leaves over the sea drains back to still.
            if u > 0.0 {
                sea = sea + 0.25 * exp(-age / 2.5) * exp(-u / 6.0);
            }
        }
    }
    return max(sea, film);
}

// ---------------------------------------------------------------------------- the neon bands

fn band_color(k: i32) -> vec3<f32> {
    switch k {
        case 0: {
            return vec3<f32>(0.10, 0.86, 1.00);
        }
        case 1: {
            return vec3<f32>(0.10, 0.30, 1.00);
        }
        case 2: {
            return vec3<f32>(1.00, 0.76, 0.08);
        }
        case 3: {
            return vec3<f32>(1.00, 0.34, 0.05);
        }
        default: {
            return vec3<f32>(1.00, 0.16, 0.50);
        }
    }
}

fn band_width(k: i32) -> f32 {
    switch k {
        case 0: {
            return W0;
        }
        case 1: {
            return W1;
        }
        case 2: {
            return W2;
        }
        case 3: {
            return W3;
        }
        default: {
            return W4;
        }
    }
}

fn band_offset(k: i32) -> f32 {
    switch k {
        case 0: {
            return 0.0;
        }
        case 1: {
            return W0;
        }
        case 2: {
            return W0 + W1;
        }
        case 3: {
            return W0 + W1 + W2;
        }
        default: {
            return W0 + W1 + W2 + W3;
        }
    }
}

// One wave's ribbon at one place along the shore, in `b` (blocks behind the front; negative is
// ahead of it, up the beach).
struct Ribbon {
    scale: f32,       // band widths are scaled by this (thin far out)
    inten: f32,       // brightness (0 = no ribbon)
    border: f32,      // the border the cyan front has drawn, in b (<= 0)
    p1lo: f32,        // the first pushing colour's interval
    p1hi: f32,
    p2lo: f32,        // the second's
    p2hi: f32,
    trail: f32,       // brightness of the cyan line left at the border
    push: f32,        // the push's progress, 0..1
}

fn push_lobe(v: f32, lam: f32, ph: f32) -> f32 {
    let a = 0.5 + 0.5 * sin(TAU * v / lam + ph);
    let b = 0.5 + 0.5 * sin(TAU * v / (0.37 * lam) + 1.7 * ph);
    return smoothstep(0.18, 0.82, 0.72 * a + 0.28 * b);
}

fn ribbon(w: Wave, l: Local, v: f32) -> Ribbon {
    var r: Ribbon;
    r.scale = mix(0.32, 1.0, smoothstep(U0 * 0.75, BREAK, l.front));
    let approach = smoothstep(U0 + 2.0, U0 * 0.55, l.front);
    let drain = 1.0 - smoothstep(l.taum + 0.5, l.taum + 2.9, l.tau);
    let start = smoothstep(0.0, 1.5, l.tau);
    r.inten = w.gain * mix(0.45, 1.0, smoothstep(U0 * 0.6, BREAK, l.front)) * approach * drain * start;
    // The border the front has drawn so far: its farthest point up the beach.
    var drawn = l.front;
    if l.tau >= l.taum {
        drawn = -l.smax;
    }
    r.border = drawn - l.front;
    // The push: it rises as the swash comes to rest and holds a moment into the drain.
    r.push = smoothstep(l.taum - 0.75, l.taum + 0.35, l.tau)
        * (1.0 - smoothstep(l.taum + 1.5, l.taum + 2.9, l.tau));
    let clamp_on = F.extra.z > 0.5;
    for (var j = 0; j < 2; j = j + 1) {
        let k = select(w.p2, w.p1, j == 0);
        let o = band_offset(k) * r.scale;
        let wk = band_width(k) * r.scale;
        let lobe = select(push_lobe(v, w.lam2, w.ph5), push_lobe(v, w.lam1, w.ph4), j == 0);
        // Its own wave: past the front by up to 1.3 blocks where its lobe is full...
        var lead = o - r.push * lobe * (o + 1.3);
        // ...but never past the border the front drew.
        if clamp_on {
            lead = max(lead, r.border);
        }
        let tail = o + wk - 0.35 * max(o - lead, 0.0);
        if j == 0 {
            r.p1lo = lead;
            r.p1hi = tail;
        } else {
            r.p2lo = lead;
            r.p2hi = tail;
        }
    }
    r.trail = smoothstep(l.taum - 0.05, l.taum + 0.25, l.tau)
        * (1.0 - smoothstep(l.taum + 0.6, l.taum + 2.6, l.tau));
    return r;
}

// What one wave's ribbon shows at `b`: the colour (already scaled by brightness), plus the light
// it throws on the surface around it.
struct Neon {
    emit: vec3<f32>,
    glow: vec3<f32>,
    // How much of the surface the ribbon covers here (it is drawn over the surface, not added to
    // it, so the white foam under it does not wash its colours out).
    cover: f32,
}

fn kern(d: f32) -> f32 {
    return 0.55 * exp(-d * 2.6) + 0.10 * exp(-d * 0.8);
}

fn neon(w: Wave, r: Ribbon, b: f32, e: f32) -> Neon {
    var n: Neon;
    n.emit = vec3<f32>(0.0);
    n.glow = vec3<f32>(0.0);
    n.cover = 0.0;
    if r.inten <= 0.0 {
        return n;
    }
    // Inside the ribbon the colours melt into each other a little, like the light trails; its
    // two outer edges stay crisp.
    let soft = max(e, 0.07 * r.scale);
    let total = (W0 + W1 + W2 + W3 + W4) * r.scale;
    // The two pushing colours come first: where they are, nothing else shows.
    let c1 = cover2(b, r.p1lo, r.p1hi, select(soft, e, r.p1lo <= 0.0), soft);
    let c2 = cover2(b, r.p2lo, r.p2hi, select(soft, e, r.p2lo <= 0.0), soft) * (1.0 - c1);
    let pushed = c1 + c2;
    // A surging colour is a light trail of its own: brightest at its leading tip, like the crest
    // of its own little wave, fading back along the length it has stretched to. It still takes
    // away every colour under it (`pushed` is its full coverage); only its light fades.
    let fall1 = exp(-max(b - r.p1lo, 0.0) / max(0.45, 0.4 * (r.p1hi - r.p1lo)));
    let fall2 = exp(-max(b - r.p2lo, 0.0) / max(0.45, 0.4 * (r.p2hi - r.p2lo)));
    let lit1 = mix(1.0, mix(0.3, 1.0, fall1), r.push)
        * (1.0 + 0.4 * exp(-max(b - r.p1lo, 0.0) / 0.3) * r.push);
    let lit2 = mix(1.0, mix(0.3, 1.0, fall2), r.push)
        * (1.0 + 0.4 * exp(-max(b - r.p2lo, 0.0) / 0.3) * r.push);
    var col = band_color(w.p1) * c1 * lit1 + band_color(w.p2) * c2 * lit2;
    var glow = band_color(w.p1) * kern(dist_to(b, r.p1lo, r.p1hi)) * (r.p1hi - r.p1lo)
        + band_color(w.p2) * kern(dist_to(b, r.p2lo, r.p2hi)) * (r.p2hi - r.p2lo);
    // The other three bands, in their places.
    var rib = 0.0;
    for (var k = 0; k < 5; k = k + 1) {
        if k == w.p1 || k == w.p2 {
            continue;
        }
        let lo = band_offset(k) * r.scale;
        let hi = lo + band_width(k) * r.scale;
        let c = cover2(b, lo, hi, select(soft, e, k == 0), select(soft, e, k == 4));
        rib = rib + c;
        col = col + band_color(k) * c * (1.0 - pushed);
        // Light only from the part the pushers left (the front band loses what they cover).
        let lost = max(
            min(hi, r.p1hi) - max(lo, r.p1lo),
            min(hi, r.p2hi) - max(lo, r.p2lo),
        );
        let left = max((hi - lo) - max(lost, 0.0), 0.0);
        glow = glow + band_color(k) * kern(dist_to(b, lo, hi)) * left;
    }
    // A white-hot core on the leading edge, as on a light trail.
    let core = exp(-pow((b - 0.05 * r.scale) / (0.07 * r.scale + e), 2.0)) * (1.0 - pushed);
    col = col + vec3<f32>(0.55, 0.9, 1.0) * core * 0.25;
    // The cyan line left at the border, until a pushing colour takes it.
    let tw = 0.22;
    let tc = cover(b, r.border, r.border + tw, e) * r.trail * (1.0 - pushed) * (1.0 - min(rib, 1.0));
    col = col + band_color(0) * tc * 0.7;
    glow = glow + band_color(0) * kern(dist_to(b, r.border, r.border + tw)) * tw * r.trail * 0.7;
    // Brightest along the front, a little dimmer towards the back of the ribbon.
    let prof = 1.0 - 0.3 * smoothstep(0.0, total, b);
    n.emit = col * r.inten * prof * 0.85;
    n.glow = glow * r.inten;
    // The surface shows through a surge's dim tail (its light covers it as much as it shines).
    let pushed_cover = min(c1 * min(lit1, 1.0) + c2 * min(lit2, 1.0), 1.0);
    n.cover = clamp(pushed_cover + min(rib, 1.0) * (1.0 - pushed) + tc, 0.0, 1.0)
        * min(r.inten * 1.5, 1.0);
    return n;
}

// ---------------------------------------------------------------------------- foam

// How much foam one wave leaves at `b` behind its front (0..1, before the lace texture). The neon
// ribbon is the crest line along the front; the white water starts under its back half and trails
// behind it, solid at first, then lace, then streaks.
fn foam_amount(l: Local, b: f32, u: f32, scale: f32) -> f32 {
    if l.tau < 0.0 {
        return 0.0;
    }
    let wb = b - (W0 + W1 + W2 + W3 + W4) * scale * 0.55;
    var f = 0.0;
    if l.tau < l.tau0 {
        let broken = smoothstep(BREAK + 2.5, BREAK - 1.5, l.front);
        let white = smoothstep(-0.7, 0.15, wb) * exp(-max(wb - 0.8, 0.0) / 2.8);
        let trail = smoothstep(-0.3, 1.2, wb) * exp(-max(wb, 0.0) / 7.5) * 0.6;
        f = broken * max(white, trail);
        // White caps start on the crest just before it breaks.
        let cap = smoothstep(BREAK + 5.0, BREAK + 0.5, l.front) * exp(-abs(wb) / 0.8) * 0.6;
        f = max(f, cap);
    } else {
        let age = l.tau - l.tau0;
        let edge = smoothstep(-0.6, 0.1, wb) * exp(-max(wb - 0.6, 0.0) / 2.2);
        let sheet = smoothstep(-0.3, 0.8, wb) * exp(-max(wb, 0.0) / 6.0) * 0.68;
        f = max(edge * exp(-age / 7.0), sheet * exp(-age / 4.5));
        if l.tau > l.taum {
            // The backwash leaves the foam line stranded at the top of the run.
            let mark_u = -l.smax + 0.25;
            let m = exp(-pow((u - mark_u) / 0.3, 2.0));
            f = max(f, m * 0.75 * exp(-(l.tau - l.taum) / 3.5) * smoothstep(l.taum, l.taum + 0.4, l.tau));
            // Draining sheet: the foam on it thins out.
            f = f * mix(1.0, 0.55, smoothstep(l.taum, l.taum + 2.0, l.tau));
        }
    }
    return f * late_fade(l.tau);
}

// The foam's lace: strands between bubbles, warped and stretched along the wave fronts into
// streaks, with holes torn in it at a larger scale. `amount` 1 is solid white water; lower amounts
// keep only the thicker strands, then only streaks. Returns (coverage, brightness): even solid
// white water is streaky, brighter where the foam is thicker.
fn foam_lace(p: vec2<f32>, amount: f32, t: f32, per_px: f32) -> vec2<f32> {
    if amount <= 0.01 {
        return vec2<f32>(0.0, 1.0);
    }
    // The foam density texture (R: rank-normalised, so a threshold of 1 - amount covers about
    // `amount` of it), stretched about three times along the fronts into streaks, at two scales.
    let q = vec2<f32>(p.x * 0.085, p.y * 0.030) + vec2<f32>(t * 0.003, 0.0);
    let a = noise_at(q, noise_lod(per_px * 0.085)).r;
    let q2 = vec2<f32>(p.x * 0.19, p.y * 0.072) + vec2<f32>(0.37, 0.61 + t * 0.004);
    let c = noise_at(q2, noise_lod(per_px * 0.19)).r;
    let dens = a * 0.68 + c * 0.32;
    let thr = 1.0 - amount;
    let soft = 0.03 + min(per_px * 2.0, 0.25);
    let strands = smoothstep(thr - soft, thr + soft, dens);
    // Far away the lace averages out to its mean coverage.
    let blend = smoothstep(0.04, 0.15, per_px);
    let shade = mix(0.62 + 0.55 * smoothstep(thr, 1.0, dens), 0.85, blend);
    return vec2<f32>(mix(strands, amount * 0.85, blend), shade);
}

// ---------------------------------------------------------------------------- the beach

fn cell_at(c: vec2<i32>) -> vec2<u32> {
    let x = clamp(c.x - HMAP_X0, 0, HMAP_W - 1);
    let z = ((c.y - HMAP_Z0) % HMAP_H + HMAP_H) % HMAP_H;
    return textureLoad(heights, vec2<i32>(x, z), 0).rg;
}

fn height_at(c: vec2<i32>) -> f32 {
    return f32(i32(cell_at(c).r) - 16);
}

struct Hit {
    t: f32,
    n: vec3<f32>,
    kind: u32,  // 0 = nothing, 1 = a block face
    block: u32,
}

// March the ray through the block columns (a 2D DDA over the height map).
fn trace_blocks(o: vec3<f32>, d: vec3<f32>, t_limit: f32) -> Hit {
    var hit: Hit;
    hit.kind = 0u;
    hit.t = t_limit;
    if d.y >= -1e-4 {
        return hit;
    }
    var t = max((H_MAX - o.y) / d.y, 0.0);
    let t_end = min((H_MIN - o.y) / d.y, t_limit);
    var side = vec3<f32>(0.0, 1.0, 0.0);
    // Seaward of `table.y` every block lies under the water (the water hides it, and the water's
    // shading reads the bed from the height map): start where the ray first gets landward of it.
    let x_land = F.table.y;
    if o.x + d.x * t > x_land {
        if d.x >= -1e-5 {
            return hit;
        }
        let t_land = (x_land - o.x) / d.x;
        if t_land > t {
            t = t_land;
            side = vec3<f32>(1.0, 0.0, 0.0);
        }
    }
    var p = o + d * t;
    var cell = vec2<i32>(floor(p.xz));
    let stp = vec2<i32>(select(-1, 1, d.x > 0.0), select(-1, 1, d.z > 0.0));
    let inv = vec2<f32>(1.0 / select(d.x, 1e-6, abs(d.x) < 1e-6), 1.0 / select(d.z, 1e-6, abs(d.z) < 1e-6));
    let delta = abs(inv);
    var tmax = vec2<f32>(
        (f32(cell.x) + select(0.0, 1.0, d.x > 0.0) - o.x) * inv.x,
        (f32(cell.y) + select(0.0, 1.0, d.z > 0.0) - o.z) * inv.y,
    );
    for (var i = 0; i < 200; i = i + 1) {
        if t > t_end {
            break;
        }
        let cv = cell_at(cell);
        let h = f32(i32(cv.r) - 16);
        let y_in = o.y + d.y * t;
        if y_in <= h {
            hit.kind = 1u;
            hit.t = t;
            hit.n = side;
            hit.block = cv.g;
            return hit;
        }
        let t_next = min(tmax.x, tmax.y);
        let y_out = o.y + d.y * t_next;
        if y_out <= h {
            // The top face is met inside this cell: only a hit if it comes before the limit
            // (the water's surface), or a block under the swash's sheet would show through it.
            let t_top = (h - o.y) / d.y;
            if t_top > t_limit {
                return hit;
            }
            hit.kind = 1u;
            hit.t = t_top;
            hit.n = vec3<f32>(0.0, 1.0, 0.0);
            hit.block = cv.g;
            return hit;
        }
        if tmax.x < tmax.y {
            cell.x = cell.x + stp.x;
            t = tmax.x;
            tmax.x = tmax.x + delta.x;
            side = vec3<f32>(f32(-stp.x), 0.0, 0.0);
        } else {
            cell.y = cell.y + stp.y;
            t = tmax.y;
            tmax.y = tmax.y + delta.y;
            side = vec3<f32>(0.0, 0.0, f32(-stp.y));
        }
    }
    return hit;
}

fn block_albedo(p: vec3<f32>, n: vec3<f32>, block: u32, footprint: f32, along: f32) -> vec3<f32> {
    var uv: vec2<f32>;
    if n.y > 0.5 {
        uv = fract(p.xz);
    } else if abs(n.x) > 0.5 {
        uv = vec2<f32>(fract(p.z * sign(n.x)), 1.0 - fract(p.y));
    } else {
        uv = vec2<f32>(fract(-p.x * sign(n.z)), 1.0 - fract(p.y));
    }
    // Texels per pixel picks the mip (Minecraft: crisp texels near, mip-mapped far).
    let lod = log2(max(16.0 * footprint / max(along, 0.2), 1e-4));
    return textureSampleLevel(blocks, block_samp, uv, i32(block), lod).rgb;
}

// How wet the sand of the terrace is at `p` (from the last few swashes).
fn wetness(p: vec2<f32>, t: f32) -> f32 {
    let u = p.x;
    if u > 1.0 {
        return 1.0;
    }
    let s = -u;
    let v = p.y;
    var wet = 0.55 * smoothstep(-5.0, 0.0, u);
    let n = newest_wave(v, t);
    for (var k = 0; k < NWAVES; k = k + 1) {
        let w = wave(n - k);
        if !alive(w, v, t) {
            continue;
        }
        let l = local(w, v, t);
        if l.tau < l.tau0 || s > l.smax {
            continue;
        }
        // The swash reached here at t_in and left at t_out.
        let disc = max(C * C - 2.0 * l.a * max(s, 0.0), 0.0);
        let t_in = l.tau0 + (C - sqrt(disc)) / l.a;
        let t_out = l.taum + sqrt(2.0 * max(l.smax - s, 0.0) / l.ab);
        if l.tau < t_in {
            continue;
        }
        var this_wet = 1.0;
        if l.tau > t_out {
            this_wet = exp(-(l.tau - t_out) / 6.0);
        }
        this_wet = this_wet * late_fade(l.tau);
        wet = max(wet, this_wet);
    }
    return wet;
}

// ---------------------------------------------------------------------------- the scene

struct Surface {
    color: vec3<f32>,
    glow: vec3<f32>,
}

// The neon of every wave near `p`, and the foam: shared by the water and the sand.
struct Waves {
    emit: vec3<f32>,
    glow: vec3<f32>,
    foam: f32,
    cover: f32,
}

fn waves_at(p: vec2<f32>, t: f32, e: f32, per_px: f32) -> Waves {
    var o: Waves;
    o.emit = vec3<f32>(0.0);
    o.glow = vec3<f32>(0.0);
    o.cover = 0.0;
    var foam = 0.0;
    let u = p.x;
    let v = p.y;
    let n = newest_wave(v, t);
    for (var k = 0; k < NWAVES; k = k + 1) {
        let w = wave(n - k);
        if !alive(w, v, t) {
            continue;
        }
        let l = local(w, v, t);
        if l.tau > l.taum + 9.0 {
            continue;
        }
        let b = u - l.front;
        if b < -16.0 || b > 40.0 {
            continue;
        }
        let r = ribbon(w, l, v);
        let nn = neon(w, r, b, e);
        o.emit = o.emit + nn.emit;
        o.glow = o.glow + nn.glow;
        o.cover = max(o.cover, nn.cover);
        foam = max(foam, foam_amount(l, b, u, r.scale));
    }
    o.foam = foam;
    return o;
}

fn shade_sand(p: vec3<f32>, n: vec3<f32>, block: u32, d: vec3<f32>, t: f32, footprint: f32, time: f32) -> Surface {
    var s: Surface;
    let along = abs(dot(n, d));
    var alb = block_albedo(p, n, block, footprint, along);
    // Minecraft's face shading (top 1.0, z sides 0.8, x sides 0.6) under the moon and the sky.
    let face = select(select(0.6, 0.8, abs(n.z) > 0.5), 1.0, n.y > 0.5);
    let wet = select(0.0, wetness(p.xz, time), n.y > 0.5);
    alb = alb * mix(1.0, 0.5, wet);
    let wv = waves_at(p.xz, time, footprint * 0.7, footprint);
    let light = night_light() * face + wv.glow * 0.30;
    var c = alb * light;
    // Wet sand is glossy: it mirrors the fog and the neon.
    let r = reflect(d, n);
    let fres = 0.04 + 0.96 * pow(1.0 - max(dot(-d, n), 0.0), 5.0);
    c = c + wet * (fog_color(r) * fres * 0.9 + wv.glow * (0.06 + 0.4 * fres));
    // Foam stranded on the sand.
    let lace = foam_lace(p.xz, wv.foam, time, footprint * 0.5);
    let foam_col = vec3<f32>(0.80, 0.82, 0.86) * (moon_light() * 1.8 + sky_ambient() + wv.glow * 0.6);
    c = mix(c, foam_col * lace.y, lace.x * 0.92);
    s.color = c * (1.0 - 0.8 * wv.cover) + wv.emit;
    s.glow = wv.glow;
    return s;
}

// `h0`: the water's height where the ray met it (from the intersection's last step).
fn shade_water(p: vec3<f32>, d: vec3<f32>, h0: f32, footprint: f32, time: f32) -> Surface {
    var s: Surface;
    // The surface's slope from the wave heights, plus fine ripples.
    let ex = max(0.12, footprint);
    let hx = water_height(p.xz + vec2<f32>(ex, 0.0), time);
    let hz = water_height(p.xz + vec2<f32>(0.0, ex), time);
    var n = normalize(vec3<f32>(-(hx - h0) / ex, 1.0, -(hz - h0) / ex));
    let rq = p.xz * vec2<f32>(0.20, 0.07) + vec2<f32>(time * 0.010, time * 0.004);
    let rl = noise_lod(footprint * 0.2);
    let g0 = noise_at(rq, rl).g;
    let gx = noise_at(rq + vec2<f32>(1.0 / 256.0, 0.0), rl).g;
    let gz = noise_at(rq + vec2<f32>(0.0, 1.0 / 256.0), rl).g;
    let ripple = vec2<f32>(gx - g0, gz - g0) * 24.0 * (1.0 - smoothstep(0.02, 0.2, footprint));
    n = normalize(n + vec3<f32>(ripple.x, 0.0, ripple.y));

    let wv = waves_at(p.xz, time, footprint * 0.7, footprint);

    // What lies under it: the sand, through water that is clear when thin and murky when deep.
    let cellh = height_at(vec2<i32>(floor(p.xz)));
    let depth = max(h0 - cellh, 0.0);
    let path = depth / max(-d.y, 0.25);
    let bed_p = vec3<f32>(p.x, cellh, p.z);
    // Through a thin sheet the sand reads as wet vanilla sand (a little darker than dry sand).
    let bed = block_albedo(bed_p, vec3<f32>(0.0, 1.0, 0.0), cell_at(vec2<i32>(floor(p.xz))).g, footprint, 1.0) * 0.72;
    let bed_lit = bed * (night_light() * 1.15 + wv.glow * 0.3);
    let murk = vec3<f32>(0.006, 0.0055, 0.0075) + wv.glow * 0.02;
    let trans = exp(-path * vec3<f32>(1.6, 1.25, 1.1));
    var c = mix(murk, bed_lit, trans);

    // The surface: the fog in the reflection, and the neon on the water near it.
    let r = reflect(d, n);
    let fres = 0.02 + 0.98 * pow(1.0 - max(dot(-d, n), 0.0), 5.0);
    c = c * (1.0 - fres) + fog_color(r) * fres * 1.1;
    c = c + wv.glow * (0.05 + 0.35 * fres);

    // Foam on top.
    let lace = foam_lace(p.xz, wv.foam, time, footprint * 0.5);
    let foam_col = vec3<f32>(0.80, 0.83, 0.90) * (moon_light() * 1.8 + sky_ambient() * 1.2 + wv.glow * 0.6);
    c = mix(c, foam_col * lace.y, lace.x * 0.92);
    s.color = c * (1.0 - 0.8 * wv.cover) + wv.emit;
    s.glow = wv.glow;
    return s;
}

// The band diagram (debug modes 1 and 2): a top-down map of one wave's ribbon.
fn diagram(px: vec2<f32>) -> vec4<f32> {
    let uv = px * F.screen.zw;
    let u = F.map.x + uv.y * F.map.y;
    let v = F.map.z + uv.x * F.map.w;
    let w = wave(i32(F.extra.w));
    let l = local(w, v, F.cam_pos.w);
    let r = ribbon(w, l, v);
    let b = u - l.front;
    if F.cam_fwd.w > 1.5 {
        // Test channels: R = the cyan front band (as laid out, before anything covers it),
        // G = a pushing colour, A = visible (always 1).
        let front = select(0.0, 1.0, b >= 0.0 && b <= W0 * r.scale && r.inten > 0.0);
        let p1 = b >= r.p1lo && b <= r.p1hi && r.p1hi > r.p1lo;
        let p2 = b >= r.p2lo && b <= r.p2hi && r.p2hi > r.p2lo;
        let pushed = select(0.0, 1.0, (p1 || p2) && r.inten > 0.0);
        return vec4<f32>(front, pushed, 0.0, 1.0);
    }
    let nn = neon(w, r, b, 0.02);
    var c = nn.emit / max(r.inten, 1e-3) * select(0.0, 1.0, r.inten > 0.0);
    // The terrace edge and the run-up border as thin grey guides.
    let pxu = F.map.y * F.screen.w;
    let guide = max(
        1.0 - smoothstep(0.0, pxu * 1.5, abs(u)),
        1.0 - smoothstep(0.0, pxu * 1.5, abs(u + run_up(w, v))),
    );
    c = max(c, vec3<f32>(0.25) * guide);
    return vec4<f32>(c, 1.0);
}

@fragment
fn fs_scene(in: FsOut) -> @location(0) vec4<f32> {
    if F.cam_fwd.w > 0.5 {
        return diagram(in.pos.xy);
    }
    let time = F.cam_pos.w;
    let o = F.cam_pos.xyz;
    let d = view_ray(in.ndc);
    let pix = pixel_angle();

    // The water's surface: a few fixed-point steps from the still-water plane.
    var t_water = T_MAX;
    var h_water = SEA;
    if d.y < -1e-3 {
        var t = (SEA + 0.25 - o.y) / d.y;
        for (var i = 0; i < 3; i = i + 1) {
            let q = o + d * t;
            h_water = water_height(q.xz, time);
            t = mix(t, (h_water - o.y) / d.y, 0.8);
        }
        t_water = clamp(t, 0.0, T_MAX);
    }
    let hit = trace_blocks(o, d, min(t_water, T_MAX));
    var color: vec3<f32>;
    var glow = vec3<f32>(0.0);
    var t_hit: f32;
    if hit.kind == 1u {
        t_hit = hit.t;
        let p = o + d * t_hit;
        let s = shade_sand(p, hit.n, hit.block, d, t_hit, t_hit * pix, time);
        color = s.color;
        glow = s.glow;
    } else if t_water < T_MAX {
        t_hit = t_water;
        let p = o + d * t_hit;
        let footprint = t_hit * pix / max(-d.y, 0.08);
        let s = shade_water(p, d, h_water, footprint, time);
        color = s.color;
        glow = s.glow;
    } else {
        return vec4<f32>(fog_color(d), 1.0);
    }
    let fog = fog_amount(o, d, t_hit);
    color = mix(color, fog_color(d), fog);
    // The neon lights the fog around it.
    color = color + glow * fog * 0.12;
    return vec4<f32>(color, 1.0);
}

// ---------------------------------------------------------------------------- bloom

// The target's UV of a full-screen pass (exact whatever the target's size).
fn uv_of(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * vec2<f32>(0.5, -0.5) + 0.5;
}

// First step down: a 4-tap box from the full-size scene, keeping only what is bright.
@fragment
fn fs_bloom_first(in: FsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_a));
    let uv = uv_of(in.ndc);
    let px = 1.0 / dims;
    var c = vec3<f32>(0.0);
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(-0.5, -0.5), 0.0).rgb;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(0.5, -0.5), 0.0).rgb;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(-0.5, 0.5), 0.0).rgb;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(0.5, 0.5), 0.0).rgb;
    c = c * 0.25;
    let l = luma(c);
    let keep = max(l - 0.06, 0.0) / max(l, 1e-4);
    return vec4<f32>(c * keep, 1.0);
}

// Down: a 13-tap-ish filter (center plus 4 diagonals) from the level above.
@fragment
fn fs_bloom_down(in: FsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_a));
    let uv = uv_of(in.ndc);
    let px = 1.0 / dims;
    var c = textureSampleLevel(src_a, lin_samp, uv, 0.0).rgb * 4.0;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(-1.0, -1.0), 0.0).rgb;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(1.0, -1.0), 0.0).rgb;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(-1.0, 1.0), 0.0).rgb;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(1.0, 1.0), 0.0).rgb;
    return vec4<f32>(c / 8.0, 1.0);
}

// Up: a tent filter of the level below, added onto this level.
@fragment
fn fs_bloom_up(in: FsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_a));
    let uv = uv_of(in.ndc);
    let px = 1.0 / dims;
    var c = textureSampleLevel(src_a, lin_samp, uv, 0.0).rgb * 4.0;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(-1.0, 0.0), 0.0).rgb * 2.0;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(1.0, 0.0), 0.0).rgb * 2.0;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(0.0, -1.0), 0.0).rgb * 2.0;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(0.0, 1.0), 0.0).rgb * 2.0;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(-1.0, -1.0), 0.0).rgb;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(1.0, -1.0), 0.0).rgb;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(-1.0, 1.0), 0.0).rgb;
    c = c + textureSampleLevel(src_a, lin_samp, uv + px * vec2<f32>(1.0, 1.0), 0.0).rgb;
    return vec4<f32>(c / 16.0, 1.0);
}

// ---------------------------------------------------------------------------- composite

// Film grain: clumpy luminance grain with some colour in it, new every grain frame.
fn grain(px: vec2<f32>) -> vec3<f32> {
    let seed = F.extra.y;
    let o1 = vec2<f32>(fract(seed * 0.6180339) * 256.0, fract(seed * 0.4142135) * 256.0);
    let o2 = vec2<f32>(fract(seed * 0.7320508) * 256.0, fract(seed * 0.2360679) * 256.0);
    let q = px / 1.35;
    let a = textureSampleLevel(noise, noise_samp, (q + o1) / 256.0, 0.0).a - 0.5;
    let b = textureSampleLevel(noise, noise_samp, (q * 1.07 + o2) / 256.0, 0.0).a - 0.5;
    let c = textureSampleLevel(noise, noise_samp, (q.yx * 0.93 + o1.yx) / 256.0, 0.0).a - 0.5;
    let l = a * 1.0;
    return vec3<f32>(l + b * 0.55, l - (b + c) * 0.18, l + c * 0.65);
}

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn hash4(c: vec2<i32>, salt: u32) -> vec4<f32> {
    let a = pcg(bitcast<u32>(c.x) ^ pcg(bitcast<u32>(c.y) ^ pcg(salt)));
    let b = pcg(a);
    let c2 = pcg(b);
    let d = pcg(c2);
    return vec4<f32>(vec4<u32>(a, b, c2, d) >> vec4<u32>(8u)) * (1.0 / 16777216.0);
}

// Sparse, soft specks drifting in the misty air, like the drizzle caught in the foggy photo. Each
// lives about three seconds inside its own 96-pixel cell (so a pixel only checks its cell),
// sinking a little as it fades in and out.
fn specks(px: vec2<f32>, t: f32) -> f32 {
    let s = max(F.out_size.y / 1080.0, 0.25);
    let cell = 96.0 * s;
    let c = floor(px / cell);
    let id = vec2<i32>(c);
    let life = 3.2;
    let phase = t / life + hash4(id, 7u).x;
    let k = fract(phase);
    let h = hash4(id, u32(floor(phase)) * 13u + 101u);
    if h.x > 0.20 {
        return 0.0;
    }
    let pos = c * cell + vec2<f32>(14.0 * s) + h.yz * (cell - 28.0 * s) + vec2<f32>(0.0, k * 9.0 * s);
    let r = (1.1 + 2.4 * h.w) * s;
    let d = length(px - pos);
    return (1.0 - smoothstep(r * 0.35, r, d)) * sin(k * PI) * (0.35 + 0.65 * h.w);
}

// The weights of a Catmull-Rom spline through four texels, for a point a fraction `t` of the way
// from the second texel to the third (per axis). The polynomials are the spline's basis; they sum
// to 1.
struct CubicWeights {
    before: vec2<f32>,
    first: vec2<f32>,
    second: vec2<f32>,
    after: vec2<f32>,
};

fn catmull_rom_weights(t: vec2<f32>) -> CubicWeights {
    let t2 = t * t;
    let t3 = t2 * t;
    var w: CubicWeights;
    w.before = 0.5 * (2.0 * t2 - t3 - t);
    w.first = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0);
    w.second = 0.5 * (4.0 * t2 - 3.0 * t3 + t);
    w.after = 0.5 * (t3 - t2);
    return w;
}

// The scene, upscaled to the frame with a Catmull-Rom filter: sharper than bilinear, so the 0.75x
// ray cast keeps the blocks' texels and the neon's edges crisp. Per axis the four texels' weights
// become three taps: the two middle texels share one bilinear tap placed between them in
// proportion to their weights, so a 3x3 grid of bilinear taps covers the 4x4 texels.
fn scene_upscaled(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(src_a));
    let px = uv * size - 0.5;
    let base = floor(px);
    let w = catmull_rom_weights(px - base);
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
    var c = vec3<f32>(0.0);
    c = c + textureSampleLevel(src_a, lin_samp, vec2<f32>(xs.x, ys.x), 0.0).rgb * (wx.x * wy.x);
    c = c + textureSampleLevel(src_a, lin_samp, vec2<f32>(xs.y, ys.x), 0.0).rgb * (wx.y * wy.x);
    c = c + textureSampleLevel(src_a, lin_samp, vec2<f32>(xs.z, ys.x), 0.0).rgb * (wx.z * wy.x);
    c = c + textureSampleLevel(src_a, lin_samp, vec2<f32>(xs.x, ys.y), 0.0).rgb * (wx.x * wy.y);
    c = c + textureSampleLevel(src_a, lin_samp, vec2<f32>(xs.y, ys.y), 0.0).rgb * (wx.y * wy.y);
    c = c + textureSampleLevel(src_a, lin_samp, vec2<f32>(xs.z, ys.y), 0.0).rgb * (wx.z * wy.y);
    c = c + textureSampleLevel(src_a, lin_samp, vec2<f32>(xs.x, ys.z), 0.0).rgb * (wx.x * wy.z);
    c = c + textureSampleLevel(src_a, lin_samp, vec2<f32>(xs.y, ys.z), 0.0).rgb * (wx.y * wy.z);
    c = c + textureSampleLevel(src_a, lin_samp, vec2<f32>(xs.z, ys.z), 0.0).rgb * (wx.z * wy.z);
    return max(c, vec3<f32>(0.0));
}

@fragment
fn fs_composite(in: FsOut) -> @location(0) vec4<f32> {
    let uv = uv_of(in.ndc);
    if F.cam_fwd.w > 0.5 {
        // The diagram: exactly what the scene pass wrote (test channels included).
        let dims = vec2<f32>(textureDimensions(src_a));
        let p = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0));
        return textureLoad(src_a, p, 0);
    }
    var c = scene_upscaled(uv);
    let bloom = textureSampleLevel(src_b, lin_samp, uv, 0.0).rgb;
    c = c + bloom * F.look.z;
    c = c * F.look.x;
    var m = aces(c);
    // Grade: lifted, blue-violet blacks, like the foggy photo.
    m = m + vec3<f32>(0.005, 0.005, 0.013) * (1.0 - m);
    // Vignette.
    let q = uv - 0.5;
    m = m * (1.0 - dot(q * vec2<f32>(0.85, 1.05), q * vec2<f32>(0.85, 1.05)) * 0.55);
    // A soft, calm shade behind the loading screen's logo, so white type keeps its contrast.
    if F.logo.z > 0.0 {
        let centre = F.logo.xy + F.logo.zw * 0.5;
        let k = (in.pos.xy - centre) / max(F.logo.zw * vec2<f32>(0.75, 1.1), vec2<f32>(1.0));
        m = m * (1.0 - F.extra.x * exp(-dot(k, k)));
    }
    var out = m;
    if F.look.w > 0.5 {
        out = linear_to_srgb(m);
    }
    out = out + vec3<f32>(0.80, 0.83, 0.95) * specks(in.pos.xy, F.cam_pos.w) * 0.32;
    // Grain in display space, strongest in the mid-tones.
    let lum = luma(out);
    let amt = F.look.y * (0.55 + 1.6 * lum * (1.0 - lum));
    out = out + grain(in.pos.xy) * amt;
    return vec4<f32>(clamp(out, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
