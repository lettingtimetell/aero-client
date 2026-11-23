// Temporal anti-aliasing resolve. See `taa.rs`.
//
// Per pixel: the 3x3 neighbourhood of the current (jittered) frame gives a colour box in YCoCg;
// the motion target (`velocity.wgsl`, rg) of the closest surface in that 3x3 says where the
// pixel was last frame; the history there is read with a 5-tap Catmull-Rom filter, rejected
// when the surface was hidden last frame (disocclusion, from the depth kept in history alpha),
// clipped into the box, and blended with the current pixel.

struct Reprojection {
    inv_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    blur_view_proj: mat4x4<f32>,
    prev_delta: vec4<f32>,
    blur_delta: vec4<f32>,
    jitter: vec4<f32>,
    screen: vec4<f32>,
};

struct TaaParams {
    // x: the current frame's weight when nothing moves. y: sharpness (0..1).
    // z: 1 = the history is valid this frame. w: unused.
    params: vec4<f32>,
};

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var velocity: texture_2d<f32>;
@group(0) @binding(3) var<uniform> r: Reprojection;
@group(0) @binding(4) var<uniform> p: TaaParams;
@group(0) @binding(5) var lin: sampler;
// The previous resolve: rgb = colour (display-encoded, like the frame), a = view depth (blocks).
@group(1) @binding(0) var history: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.25, 0.5, 0.25)),
        dot(c, vec3<f32>(0.5, 0.0, -0.5)),
        dot(c, vec3<f32>(-0.25, 0.5, -0.25)),
    );
}

fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Bring the history colour `q` into the box [lo, hi]. Outside the box it moves along the line to
// the box's centre until it meets the box's surface: the colour keeps its direction from the
// centre, where clamping each channel on its own would shift the hue. Inside, it is unchanged.
fn clip_to_box(lo: vec3<f32>, hi: vec3<f32>, q: vec3<f32>) -> vec3<f32> {
    let centre = 0.5 * (lo + hi);
    let half_size = 0.5 * (hi - lo) + vec3<f32>(1e-5);
    let from_centre = q - centre;
    // How far out `q` is, in half-sizes of the box along its worst axis (above 1 = outside).
    let ratio = abs(from_centre) / half_size;
    let reach = max(max(ratio.x, ratio.y), ratio.z);
    return select(q, centre + from_centre / reach, reach > 1.0);
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

// The history at `uv` through a Catmull-Rom filter: sharper than one bilinear tap, which would
// blur a little more on every frame the camera moves. The two middle texels of each axis share
// one bilinear tap placed between them in proportion to their weights; of the resulting 3x3 taps,
// the cross through the centre is read (the four corners carry little weight) and the result is
// normalised by the weights used.
fn history_catmull_rom(uv: vec2<f32>) -> vec3<f32> {
    let px = uv * r.screen.xy - 0.5;
    let base = floor(px);
    let w = catmull_rom_weights(px - base);
    let middle = w.first + w.second;
    let lo = (base - 0.5) * r.screen.zw;
    let mid = (base + 0.5 + w.second / middle) * r.screen.zw;
    let hi = (base + 2.5) * r.screen.zw;
    let centre_w = middle.x * middle.y;
    var sum = textureSampleLevel(history, lin, mid, 0.0).rgb * centre_w;
    var weight = centre_w;
    // Left, right, above, below the centre tap.
    let wl = w.before.x * middle.y;
    let wr = w.after.x * middle.y;
    let wa = middle.x * w.before.y;
    let wb = middle.x * w.after.y;
    sum = sum + textureSampleLevel(history, lin, vec2<f32>(lo.x, mid.y), 0.0).rgb * wl;
    sum = sum + textureSampleLevel(history, lin, vec2<f32>(hi.x, mid.y), 0.0).rgb * wr;
    sum = sum + textureSampleLevel(history, lin, vec2<f32>(mid.x, lo.y), 0.0).rgb * wa;
    sum = sum + textureSampleLevel(history, lin, vec2<f32>(mid.x, hi.y), 0.0).rgb * wb;
    weight = weight + wl + wr + wa + wb;
    return max(sum / weight, vec3<f32>(0.0));
}

// The camera-relative position of the surface at pixel `ip` with depth `depth`.
fn surface_at(ip: vec2<i32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(ip) + vec2<f32>(0.5)) * r.screen.zw;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let v = r.inv_proj * vec4<f32>(ndc, depth, 1.0);
    return (r.inv_view * vec4<f32>(v.xyz / v.w, 1.0)).xyz;
}

struct Resolved {
    colour: vec3<f32>,
    alpha: f32,
    // This pixel's view depth, kept in the history for the next frame's disocclusion test.
    depth: f32,
};

fn resolve(pos: vec4<f32>) -> Resolved {
    let ip = vec2<i32>(pos.xy);
    let last = vec2<i32>(r.screen.xy) - vec2<i32>(1);
    let centre_px = textureLoad(scene, ip, 0);
    let current = centre_px.rgb;

    // The 3x3 neighbourhood: colour moments for the box, and the closest surface's offset.
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var lo = vec3<f32>(1e9);
    var hi = vec3<f32>(-1e9);
    var closest = 2.0;
    var closest_at = ip;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(ip + vec2<i32>(dx, dy), vec2<i32>(0), last);
            let c = to_ycocg(textureLoad(scene, q, 0).rgb);
            m1 += c;
            m2 += c * c;
            lo = min(lo, c);
            hi = max(hi, c);
            let d = textureLoad(depth_tex, q, 0);
            if (d < closest) {
                closest = d;
                closest_at = q;
            }
        }
    }
    let centre_depth = textureLoad(depth_tex, ip, 0);
    let here = surface_at(ip, centre_depth);
    let view_depth = (r.view_proj * vec4<f32>(here, 1.0)).w;

    var out: Resolved;
    out.alpha = centre_px.a;
    out.depth = view_depth;
    out.colour = current;
    if (p.params.z < 0.5) {
        return out;
    }

    let motion = textureLoad(velocity, closest_at, 0).rg;
    let uv = (vec2<f32>(ip) + vec2<f32>(0.5)) * r.screen.zw;
    let prev_uv = uv - motion;
    if (any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0))) {
        return out; // it came from off screen: nothing to reuse
    }

    // Disocclusion: the surface we reproject was behind everything the history held around
    // there (an occluder moved away). The history keeps per-pixel view depth in alpha.
    let surface = surface_at(closest_at, closest);
    let sky = closest >= 1.0;
    let eye = select(1.0, 0.0, sky);
    let expected = (r.prev_view_proj * vec4<f32>(surface + r.prev_delta.xyz * eye, 1.0)).w;
    let texel = prev_uv * r.screen.xy - 0.5;
    let base = clamp(vec2<i32>(floor(texel)), vec2<i32>(0), last);
    let next = min(base + vec2<i32>(1), last);
    let d00 = textureLoad(history, base, 0).a;
    let d10 = textureLoad(history, vec2<i32>(next.x, base.y), 0).a;
    let d01 = textureLoad(history, vec2<i32>(base.x, next.y), 0).a;
    let d11 = textureLoad(history, next, 0).a;
    let farthest = max(max(d00, d10), max(d01, d11));
    if (!sky && expected > farthest * 1.1 + 0.25) {
        return out;
    }

    // Variance box (mean +- 1 sigma) intersected with the min/max box.
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    let box_lo = max(lo, mean - sigma);
    let box_hi = min(hi, mean + sigma);
    let hist = clip_to_box(box_lo, box_hi, to_ycocg(history_catmull_rom(prev_uv)));
    let cur = to_ycocg(current);

    // More of the current frame while things move (history resampling blurs). No inverse-luma
    // weights: they are for HDR highlights, and on this display-encoded frame they made a dark
    // sample count for twice its weight, doubling the ripple on edges.
    let moved_px = length(motion * r.screen.xy);
    let w_cur = clamp(p.params.x + moved_px * 0.02, p.params.x, 0.5);
    out.colour = clamp(from_ycocg(mix(hist, cur, w_cur)), vec3<f32>(0.0), vec3<f32>(1.0));
    return out;
}

struct BothOut {
    @location(0) frame: vec4<f32>,
    @location(1) history: vec4<f32>,
};

// Sharpness 0: the resolve goes straight to the frame and the history.
@fragment
fn fs_resolve(@builtin(position) pos: vec4<f32>) -> BothOut {
    let res = resolve(pos);
    var out: BothOut;
    out.frame = vec4<f32>(res.colour, res.alpha);
    out.history = vec4<f32>(res.colour, res.depth);
    return out;
}

// Sharpness above 0: the history first, then `taa_sharpen.wgsl` draws the frame from it, so
// the sharpening never feeds back into the history.
@fragment
fn fs_resolve_history(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res = resolve(pos);
    return vec4<f32>(res.colour, res.depth);
}
