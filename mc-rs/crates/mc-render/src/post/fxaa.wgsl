// See `taa.rs`.
//
// Fast approximate anti-aliasing (the FXAA algorithm, its "quality" variant), in this project's
// own code: find a luma edge through the pixel, decide whether it runs horizontally or vertically,
// walk along it both ways until the luma stops matching the edge, and move the pixel's sample
// across the edge by how close the pixel sits to the nearer end (a pixel near an end belongs
// mostly to one side of the edge). A subpixel term also softens features smaller than a pixel.
// The contrast thresholds, the subpixel strength and the walk's step lengths are the algorithm's
// usual settings for its high-quality mode.
//
// Minecraft's textures are pixel art, and classic FXAA also softens every texel boundary inside
// a texture. With `params.z = 0` ("Texture edges" off) a pixel is only touched where the depth
// buffer has a real geometric edge: a silhouette, a block corner, a leaf or glass cut-out. The
// test is the second difference of `depth + proj[2][2]`, which is proportional to 1 / view
// depth and therefore exactly linear across any plane: zero on every flat face, large on an
// edge between two surfaces.

struct FxaaParams {
    // xy: 1 / frame size. z: 1 = also smooth texture edges (classic FXAA).
    // w: the projection's [2][2], for the geometry test.
    params: vec4<f32>,
};

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var<uniform> p: FxaaParams;

// Below this luma contrast (or the share of the local maximum chosen per mode) there is no edge.
const CONTRAST_FLOOR: f32 = 0.0312;
// How strongly the subpixel term may move a sample (0..1).
const SUBPIXEL_STRENGTH: f32 = 0.75;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

// Display-encoded luma: the frame is already in the surface's (gamma) byte domain.
fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn luma_at(ip: vec2<i32>, last: vec2<i32>) -> f32 {
    return luma(textureLoad(scene, clamp(ip, vec2<i32>(0), last), 0).rgb);
}

fn luma_uv(uv: vec2<f32>) -> f32 {
    return luma(textureSampleLevel(scene, lin, uv, 0.0).rgb);
}

// Where the walk along an edge stands: how far each side has got in pixels (x back, y ahead), the
// luma difference at each side's last check, and whether each side has found its end.
struct Walk {
    dist: vec2<f32>,
    diff: vec2<f32>,
    found: vec2<bool>,
};

// One check of the walk, both sides at once: a side that has not found its end reads the luma
// `dist` pixels out; if that is still within `limit` of the edge's average it moves on `stride`.
fn walk_step(w: Walk, start: vec2<f32>, along: vec2<f32>, average: f32, limit: f32,
    stride: f32) -> Walk {
    if (all(w.found)) {
        return w;
    }
    var next = w;
    if (!w.found.x) {
        next.diff.x = luma_uv(start - along * w.dist.x) - average;
        next.found.x = abs(next.diff.x) >= limit;
        if (!next.found.x) {
            next.dist.x = w.dist.x + stride;
        }
    }
    if (!w.found.y) {
        next.diff.y = luma_uv(start + along * w.dist.y) - average;
        next.found.y = abs(next.diff.y) >= limit;
        if (!next.found.y) {
            next.dist.y = w.dist.y + stride;
        }
    }
    return next;
}

// Walk from `start` both ways along the edge (`along`: one pixel, in UV), each side until the
// luma there differs from the edge's local average by at least `limit`, for at most eleven
// checks. Both sides advance in lockstep, so their texture reads overlap. The checks are written
// out rather than looped (a loop here is not unrolled, and its steps then wait on each other):
// one pixel at a time near the start, then longer strides (1.5, 2, 4, 8) so long edges stay
// cheap. Returns how far each side got in pixels (x back, y ahead) and the luma difference at
// each side's last check (z back, w ahead).
fn walk_edge(start: vec2<f32>, along: vec2<f32>, average: f32, limit: f32) -> vec4<f32> {
    var w = Walk(vec2<f32>(1.0), vec2<f32>(0.0), vec2<bool>(false));
    w = walk_step(w, start, along, average, limit, 1.0);
    w = walk_step(w, start, along, average, limit, 1.0);
    w = walk_step(w, start, along, average, limit, 1.0);
    w = walk_step(w, start, along, average, limit, 1.0);
    w = walk_step(w, start, along, average, limit, 1.5);
    w = walk_step(w, start, along, average, limit, 2.0);
    w = walk_step(w, start, along, average, limit, 2.0);
    w = walk_step(w, start, along, average, limit, 2.0);
    w = walk_step(w, start, along, average, limit, 2.0);
    w = walk_step(w, start, along, average, limit, 4.0);
    w = walk_step(w, start, along, average, limit, 8.0);
    return vec4<f32>(w.dist, w.diff);
}

fn depth_term(ip: vec2<i32>, last: vec2<i32>) -> f32 {
    return textureLoad(depth_tex, clamp(ip, vec2<i32>(0), last), 0) + p.params.w;
}

// A real geometric edge at or next to `ip` (module comment).
fn geometry_edge(ip: vec2<i32>, last: vec2<i32>) -> bool {
    let c = depth_term(ip, last);
    let n = depth_term(ip + vec2<i32>(0, -1), last);
    let s = depth_term(ip + vec2<i32>(0, 1), last);
    let w = depth_term(ip + vec2<i32>(-1, 0), last);
    let e = depth_term(ip + vec2<i32>(1, 0), last);
    let bend = max(abs(w + e - 2.0 * c), abs(n + s - 2.0 * c));
    return bend > 0.02 * abs(c);
}

@fragment
fn fs_fxaa(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ip = vec2<i32>(pos.xy);
    let last = vec2<i32>(textureDimensions(scene)) - vec2<i32>(1);
    let centre = textureLoad(scene, ip, 0);
    let texture_edges = p.params.z > 0.5;
    if (!texture_edges && !geometry_edge(ip, last)) {
        return centre;
    }
    let texel = p.params.xy;
    let uv = (vec2<f32>(ip) + vec2<f32>(0.5)) * texel;

    // Contrast of the pixel and its four direct neighbours. Geometry edges are known to be
    // edges, so a lower share of the local maximum is enough there.
    let c = luma(centre.rgb);
    let n = luma_at(ip + vec2<i32>(0, -1), last);
    let s = luma_at(ip + vec2<i32>(0, 1), last);
    let w = luma_at(ip + vec2<i32>(-1, 0), last);
    let e = luma_at(ip + vec2<i32>(1, 0), last);
    let darkest = min(min(c, min(n, s)), min(w, e));
    let brightest = max(max(c, max(n, s)), max(w, e));
    let contrast = brightest - darkest;
    let share = select(0.063, 0.125, texture_edges);
    if (contrast < max(CONTRAST_FLOOR, brightest * share)) {
        return centre;
    }

    // The whole 3x3 neighbourhood's luma as three rows (top to bottom), each left to right.
    let top = vec3<f32>(
        luma_at(ip + vec2<i32>(-1, -1), last),
        n,
        luma_at(ip + vec2<i32>(1, -1), last),
    );
    let mid = vec3<f32>(w, c, e);
    let bottom = vec3<f32>(
        luma_at(ip + vec2<i32>(-1, 1), last),
        s,
        luma_at(ip + vec2<i32>(1, 1), last),
    );

    // Which way the edge runs: compare the luma's second difference across the rows (a
    // horizontal edge) with the one across the columns (a vertical edge), each a 1-2-1 weighted
    // sum over the three lines. A second difference is the two outer values less twice the middle.
    let outer_v = top + bottom;
    let outer_h = vec3<f32>(top.x + top.z, mid.x + mid.z, bottom.x + bottom.z);
    let across_rows = abs(outer_v.x - 2.0 * mid.x) + 2.0 * abs(outer_v.y - 2.0 * c)
        + abs(outer_v.z - 2.0 * mid.z);
    let across_columns = abs(outer_h.x - 2.0 * top.y) + 2.0 * abs(outer_h.y - 2.0 * c)
        + abs(outer_h.z - 2.0 * bottom.y);
    let horizontal = across_rows >= across_columns;

    // The neighbours on either side of the edge: `near` below (horizontal) or left (vertical),
    // `far` above or right. The walk follows the side with the larger luma step; `side` points
    // there, one pixel in UV (y grows downwards).
    let near = select(mid.x, bottom.y, horizontal);
    let far = select(mid.z, top.y, horizontal);
    let near_step = near - c;
    let far_step = far - c;
    let near_side = abs(near_step) >= abs(far_step);
    let limit = 0.25 * max(abs(near_step), abs(far_step));
    let edge_average = 0.5 * (c + select(far, near, near_side));
    var side: vec2<f32>;
    if (horizontal) {
        side = vec2<f32>(0.0, select(-texel.y, texel.y, near_side));
    } else {
        side = vec2<f32>(select(texel.x, -texel.x, near_side), 0.0);
    }

    // Walk both ways along the edge, half a pixel over on the chosen side.
    let along = select(vec2<f32>(0.0, texel.y), vec2<f32>(texel.x, 0.0), horizontal);
    let walk = walk_edge(uv + side * 0.5, along, edge_average, limit);

    // Nearer to one end, the pixel belongs more to the far side: shift up to half a pixel, but
    // only when the luma at that end changes the way this pixel's own luma does.
    let back_nearer = walk.x < walk.y;
    let end_diff = select(walk.w, walk.z, back_nearer);
    let shift = 0.5 - min(walk.x, walk.y) / (walk.x + walk.y);
    var amount = select(0.0, shift, (end_diff < 0.0) != (c < edge_average));

    // A pixel unlike its whole neighbourhood (detail under a pixel): blend by how much it stands
    // out from the 3x3 average without itself (direct neighbours count twice).
    let ring = 2.0 * (top.y + bottom.y + mid.x + mid.z) + top.x + top.z + bottom.x + bottom.z;
    let stand_out = clamp(abs(ring / 12.0 - c) / contrast, 0.0, 1.0);
    let soft = smoothstep(0.0, 1.0, stand_out);
    amount = max(amount, soft * soft * SUBPIXEL_STRENGTH);

    return vec4<f32>(textureSampleLevel(scene, lin, uv + side * amount, 0.0).rgb, centre.a);
}
