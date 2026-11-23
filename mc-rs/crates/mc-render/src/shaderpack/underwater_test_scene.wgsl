// A synthetic world for the Underwater pack's GPU test.
//
// Prepended: `post/view_geometry_common.wgsl` (camera, `vs`, `vg_view_position`). One full-screen
// triangle ray-casts, per pixel, an island of grass blocks (y = 63, the eye stands 1.62 above it,
// a second island 70 blocks out), a stone house, a red "player" 5 blocks ahead and a blue one 30
// blocks ahead, and otherwise a vanilla-looking sky; it writes the display-encoded colour and the
// real depth, so the pack sees what it would see in a frame.

struct SceneOut {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

fn ts_view_to_world(v: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    return transpose(m) * v;
}

// Ray-box intersection: (near, far), far < near when missed.
fn ts_box(o: vec3<f32>, d: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / d;
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let t0 = min(a, b);
    let t1 = max(a, b);
    return vec2<f32>(max(max(t0.x, t0.y), t0.z), min(min(t1.x, t1.y), t1.z));
}

fn ts_box_normal(p: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    let c = (lo + hi) * 0.5;
    let h = (hi - lo) * 0.5;
    let q = (p - c) / h;
    let a = abs(q);
    if (a.x > a.y && a.x > a.z) {
        return vec3<f32>(sign(q.x), 0.0, 0.0);
    }
    if (a.y > a.z) {
        return vec3<f32>(0.0, sign(q.y), 0.0);
    }
    return vec3<f32>(0.0, 0.0, sign(q.z));
}

// Vanilla's fixed face shading: top 1.0, north/south 0.8, east/west 0.6, bottom 0.5.
fn ts_shade(n: vec3<f32>) -> f32 {
    if (n.y > 0.5) {
        return 1.0;
    }
    if (n.y < -0.5) {
        return 0.5;
    }
    if (abs(n.z) > 0.5) {
        return 0.8;
    }
    return 0.6;
}

@fragment
fn fs_scene(@builtin(position) pos: vec4<f32>) -> SceneOut {
    let d = normalize(ts_view_to_world(vg_view_position(pos.xy, 1.0)));
    // The eye is at the origin; the grass top is 1.62 below it.
    let o = vec3<f32>(0.0, 0.0, 0.0);
    var t = 1.0e9;
    var colour = vec3<f32>(0.0);
    var n = vec3<f32>(0.0, 1.0, 0.0);
    // Islands: a 40-block square slab of grass ahead and a second one further out.
    let islands = array<vec4<f32>, 2>(
        vec4<f32>(-20.0, -40.0, 20.0, 6.0),
        vec4<f32>(-12.0, -95.0, 12.0, -70.0),
    );
    for (var i = 0; i < 2; i = i + 1) {
        let r = islands[i];
        let lo = vec3<f32>(r.x, -5.62, r.y);
        let hi = vec3<f32>(r.z, -1.62, r.w);
        let hit = ts_box(o, d, lo, hi);
        if (hit.x <= hit.y && hit.y > 0.0 && hit.x > 0.0 && hit.x < t) {
            t = hit.x;
            let p = o + d * t;
            n = ts_box_normal(p, lo, hi);
            let cell = floor(p.xz + vec2<f32>(1000.0)) + floor(vec2<f32>(p.y + 1000.0));
            let checker = fract((cell.x + cell.y) * 0.5) * 0.08;
            if (n.y > 0.5) {
                colour = vec3<f32>(0.37, 0.58, 0.23) * (0.92 + checker);
            } else {
                colour = vec3<f32>(0.47, 0.34, 0.24) * (0.92 + checker);
            }
        }
    }
    // A stone house to the left, and the two players.
    let boxes = array<vec4<f32>, 3>(
        vec4<f32>(-11.0, -16.0, -6.0, 3.5),
        vec4<f32>(-0.3, -5.3, 0.3, 0.18),
        vec4<f32>(4.7, -30.3, 5.3, 0.18),
    );
    let tints = array<vec3<f32>, 3>(
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.75, 0.14, 0.12),
        vec3<f32>(0.14, 0.24, 0.78),
    );
    for (var i = 0; i < 3; i = i + 1) {
        let b = boxes[i];
        var lo = vec3<f32>(b.x, -1.62, b.y);
        var hi = vec3<f32>(b.x + 5.0, b.w, b.y + 5.0);
        if (i > 0) {
            lo = vec3<f32>(b.x, -1.62, b.y);
            hi = vec3<f32>(b.z, b.w, b.y + 0.6);
        }
        let hit = ts_box(o, d, lo, hi);
        if (hit.x <= hit.y && hit.x > 0.0 && hit.x < t) {
            t = hit.x;
            n = ts_box_normal(o + d * t, lo, hi);
            colour = tints[i];
        }
    }
    // A stone roof slab, block-aligned in the world (the eye stands at x .5 / y .62 / z .5):
    // world y 65..66 over x -3889..-3885, z 2050..2054.
    {
        let lo = vec3<f32>(2.5, 0.38, -9.5);
        let hi = vec3<f32>(6.5, 1.38, -5.5);
        let hit = ts_box(o, d, lo, hi);
        if (hit.x <= hit.y && hit.x > 0.0 && hit.x < t) {
            t = hit.x;
            n = ts_box_normal(o + d * t, lo, hi);
            colour = vec3<f32>(0.55, 0.55, 0.55);
        }
    }
    var out: SceneOut;
    if (t > 1.0e8) {
        let up = clamp(d.y, 0.0, 1.0);
        let sky = mix(vec3<f32>(0.75, 0.85, 1.0), vec3<f32>(0.47, 0.65, 1.0), smoothstep(0.0, 0.4, up));
        let below = mix(vec3<f32>(0.75, 0.85, 1.0), vec3<f32>(0.16, 0.18, 0.3), smoothstep(0.0, 0.3, -d.y));
        out.colour = vec4<f32>(select(below, sky, d.y >= 0.0), 1.0);
        out.depth = 1.0;
        return out;
    }
    let p = d * t;
    let clip = camera.view_proj * vec4<f32>(p, 1.0);
    out.colour = vec4<f32>(colour * ts_shade(n), 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 0.99999);
    return out;
}
