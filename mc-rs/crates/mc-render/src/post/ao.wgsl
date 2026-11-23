// Screen-space ambient occlusion. Appended to
// `view_geometry_common.wgsl`.
//
// Three entry points, one per pass, each with its own bind group layout (the binding numbers are
// unique across this file so the module validates as a whole):
//
//   fs_gtao       AO resolution. Horizon-based visibility (GTAO): per slice direction, march
//                 both ways in screen space, keep the highest horizon on each side (with a distance
//                 falloff), clamp to the normal's hemisphere and integrate the cosine-weighted
//                 visible arc analytically. Slice rotation and step offset come from a 4x4 Bayer
//                 pattern that the denoise pass averages away. Writes the visibility AND the same
//                 estimator's unoccluded value (Rg16Float), and samples exactly on the slice line
//                 (`ao_line_point`): together they make every open flat surface exactly 1, where
//                 the textbook version leaves a few percent of false shading (see below).
//   fs_denoise    AO resolution. 4x4 box that covers exactly one noise period, weighted by each
//                 neighbour's distance from the centre's tangent plane (so it never blurs across a
//                 silhouette, and works on surfaces seen at a grazing angle). Outputs the ratio of
//                 the weighted sums: the normalised visibility (R16Float).
//   fs_composite  Full resolution, into the frame. Edge-aware upsample (bilinear where the four
//                 nearest AO texels lie on this pixel's surface, else the nearest-depth texel), the
//                 strength curve and the distance fade, then a multiply blend. Sky pixels and
//                 pixels with nothing to darken `discard`, so they keep their exact bytes.
//
// `texture_depth_2d` is rewritten to `texture_depth_multisampled_2d` under MSAA (sample 0 is read).

struct AoParams {
    // x = AO width, y = AO height, z = full-resolution pixels per AO texel (1 or 2), w = slices.
    grid: vec4<f32>,
    // x = radius in blocks, y = steps per side, z = falloff start (fraction of the radius),
    // w = largest screen radius in AO texels.
    gtao: vec4<f32>,
    // x = strength exponent, y = fade start / z = fade end (fractions of the render distance),
    // w = denoise tangent-plane tolerance in blocks.
    look: vec4<f32>,
};

@group(1) @binding(0) var ao_distance: texture_2d<f32>;
@group(1) @binding(1) var ao_normal: texture_2d<f32>;
@group(1) @binding(2) var<uniform> ao: AoParams;
@group(1) @binding(3) var ao_raw: texture_2d<f32>;
@group(1) @binding(4) var ao_clean: texture_2d<f32>;
@group(1) @binding(5) var ao_frame_depth: texture_depth_2d;

const AO_PI: f32 = 3.14159265358979;
const AO_HALF_PI: f32 = 1.57079632679490;

// 4x4 ordered-dither index in [0, 16), from two 2x2 levels: B2(x, y) = 2 * (x ^ y) + y.
fn ao_bayer4(p: vec2<u32>) -> u32 {
    let lo = ((p.x ^ p.y) & 1u) * 2u + (p.y & 1u);
    let hi = (((p.x >> 1u) ^ (p.y >> 1u)) & 1u) * 2u + ((p.y >> 1u) & 1u);
    return lo * 4u + hi;
}

// The full-resolution pixel centre an AO texel was sampled at (see `view_geometry.wgsl`).
fn ao_texel_px(texel: vec2<i32>) -> vec2<f32> {
    return vec2<f32>(texel) * ao.grid.z + vec2<f32>(0.5);
}

// The view-space point at `offset` AO texels from `texel`, exactly on that line: one whole-texel
// step along the line's major axis, and on the minor axis inverse distance (affine across a
// plane) interpolated between the two texels the line passes between. Rounding to the nearest
// texel instead puts the point up to half a texel beside the slice, and on a plane that sideways
// part reads as a raised horizon: a few percent of false shading on every flat surface, worst
// near a plane's horizon. Across a silhouette the interpolated point floats between the two
// surfaces; the distance falloff makes that harmless (it is more than a radius from one of them,
// or a plausible horizon between both). Next to sky there is no surface to interpolate and the
// geometry texel would sit beside the line (a dark rim along every edge against the sky), so the
// sample counts as empty. `w` = 0 when the point is off screen or empty.
fn ao_line_point(texel: vec2<i32>, offset: vec2<f32>) -> vec4<f32> {
    let size = vec2<i32>(ao.grid.xy);
    let x_major = abs(offset.x) >= abs(offset.y);
    var delta: vec2<f32>;
    if (x_major) {
        let dx = round(offset.x);
        delta = vec2<f32>(dx, offset.y * (dx / offset.x));
    } else {
        let dy = round(offset.y);
        delta = vec2<f32>(offset.x * (dy / offset.y), dy);
    }
    let base = vec2<f32>(texel) + delta;
    let lo = floor(base);
    let t0 = vec2<i32>(lo);
    let step_minor = select(vec2<i32>(1, 0), vec2<i32>(0, 1), x_major);
    let t1 = t0 + step_minor;
    let f = select(base.x - lo.x, base.y - lo.y, x_major);
    if (any(t0 < vec2<i32>(0)) || any(t0 >= size)) {
        return vec4<f32>(0.0);
    }
    let d0 = textureLoad(ao_distance, t0, 0).x;
    var d1 = d0;
    if (all(t1 < size)) {
        d1 = textureLoad(ao_distance, t1, 0).x;
    }
    if (max(d0, d1) >= VG_SKY_LIMIT) {
        return vec4<f32>(0.0);
    }
    let dist = 1.0 / mix(1.0 / d0, 1.0 / d1, f);
    return vec4<f32>(vg_view_position(base * ao.grid.z + vec2<f32>(0.5), dist), 1.0);
}

// The horizon search below (`ao_arc` and `fs_gtao`'s slice loop) follows a published
// implementation of this technique released under the MIT licence. Its notice, word for word, is
// in `NOTICE-ambient-occlusion.txt` beside this file; the portable zip carries it in
// THIRD-PARTY-NOTICES.txt.

// The cosine-weighted visible arc between the view vector and horizon angle `h`, for a slice
// whose projected normal sits at angle `n` (paper, eq. 10, one side).
fn ao_arc(h: f32, n: f32, cos_n: f32, sin_n: f32) -> f32 {
    return 0.25 * (-cos(2.0 * h - n) + cos_n + 2.0 * h * sin_n);
}

@fragment
fn fs_gtao(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(pos.xy);
    let dist = textureLoad(ao_distance, texel, 0).x;
    if (dist >= VG_SKY_LIMIT) {
        return vec4<f32>(1.0);
    }
    let p = vg_view_position(ao_texel_px(texel), dist);
    let n = textureLoad(ao_normal, texel, 0).xyz;
    let v = normalize(-p);

    let radius = ao.gtao.x;
    // Blocks to AO texels at this distance: proj[1][1] maps view y / dist to NDC, half the
    // full-resolution height maps NDC to pixels, the scale maps pixels to AO texels.
    let px_per_block = 0.5 * camera.screen.y * camera.proj[1][1] / (dist * ao.grid.z);
    let radius_px = min(radius * px_per_block, ao.gtao.w);
    if (radius_px < 1.0) {
        return vec4<f32>(1.0);
    }

    let slices = i32(ao.grid.w);
    let steps = i32(ao.gtao.y);
    let dither = ao_bayer4(vec2<u32>(texel));
    let rotation = (f32(dither) + 0.5) / 16.0;
    let jitter = f32((dither * 7u + 5u) & 15u) / 16.0;
    // weight(len) = saturate((radius - len) / (radius * (1 - falloff_start))).
    let falloff_range = max(radius * (1.0 - ao.gtao.z), 1e-3);
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = radius / falloff_range;
    // Never sample the centre texel itself.
    let min_s = 1.3 / radius_px;

    var visibility = 0.0;
    var unoccluded = 0.0;
    for (var slice = 0; slice < slices; slice++) {
        let phi = (f32(slice) + rotation) / f32(slices) * AO_PI;
        let dir = vec2<f32>(cos(phi), sin(phi));
        // Screen +y is view -y.
        let dir_view = vec3<f32>(dir.x, -dir.y, 0.0);
        let ortho = dir_view - dot(dir_view, v) * v;
        let axis = normalize(cross(ortho, v));
        let n_proj = n - axis * dot(n, axis);
        let n_len = length(n_proj);
        let cos_n = clamp(dot(n_proj, v) / max(n_len, 1e-6), -1.0, 1.0);
        let ang_n = sign(dot(ortho, n_proj)) * acos(cos_n);
        let sin_n = sin(ang_n);

        // The unoccluded horizon on each side is the tangent plane: cos(ang_n +- pi/2).
        let low_pos = -sin_n;
        let low_neg = sin_n;
        var best_pos = low_pos;
        var best_neg = low_neg;
        for (var step = 0; step < steps; step++) {
            var s = (f32(step) + jitter) / f32(steps);
            s = s * s + min_s;
            let offset = dir * (s * radius_px);
            for (var side = 0; side < 2; side++) {
                let signed_offset = select(-offset, offset, side == 0);
                let sample = ao_line_point(texel, signed_offset);
                if (sample.w == 0.0) {
                    continue;
                }
                let h = sample.xyz - p;
                let len = length(h);
                let shc = dot(h, v) / max(len, 1e-6);
                let weight = clamp(len * falloff_mul + falloff_add, 0.0, 1.0);
                if (side == 0) {
                    best_pos = max(best_pos, mix(low_pos, shc, weight));
                } else {
                    best_neg = max(best_neg, mix(low_neg, shc, weight));
                }
            }
        }
        var h_pos = acos(clamp(best_pos, -1.0, 1.0));
        var h_neg = -acos(clamp(best_neg, -1.0, 1.0));
        h_pos = min(h_pos, ang_n + AO_HALF_PI);
        h_neg = max(h_neg, ang_n - AO_HALF_PI);
        let arcs = ao_arc(h_pos, ang_n, cos_n, sin_n) + ao_arc(h_neg, ang_n, cos_n, sin_n);
        visibility += n_len * arcs;
        // The same slice with nothing in it (both horizons on the tangent plane).
        unoccluded += n_len * (cos_n + ang_n * sin_n);
    }
    // Slice angles are spread evenly on screen, not evenly around this pixel's view vector, so
    // away from the screen centre the plain average drifts several percent above or below 1 on an
    // open surface. Dividing by the same estimate of the unoccluded hemisphere cancels that: open
    // surfaces get exactly 1. The denoise divides its weighted sums, not per-pixel ratios.
    let count = f32(slices);
    return vec4<f32>(visibility / count, unoccluded / count, 0.0, 0.0);
}

@fragment
fn fs_denoise(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(pos.xy);
    let dist = textureLoad(ao_distance, texel, 0).x;
    if (dist >= VG_SKY_LIMIT) {
        return vec4<f32>(1.0);
    }
    let p = vg_view_position(ao_texel_px(texel), dist);
    let n = textureLoad(ao_normal, texel, 0).xyz;
    let size = vec2<i32>(ao.grid.xy);
    let tolerance = max(ao.look.w, 0.015 * dist);
    // Four consecutive texels on each axis always contain each Bayer column/row once; alternate
    // the window between [-2, 1] and [-1, 2] so the average is centred.
    let start = vec2<i32>(-2) + (texel & vec2<i32>(1));
    // x = weighted visibility, y = weighted unoccluded estimate (see `fs_gtao`).
    var sum = vec2<f32>(0.0);
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            let q = clamp(texel + start + vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1));
            let qd = textureLoad(ao_distance, q, 0).x;
            if (qd >= VG_SKY_LIMIT) {
                continue;
            }
            let qp = vg_view_position(ao_texel_px(q), qd);
            let w = clamp(1.0 - abs(dot(n, qp - p)) / tolerance, 0.0, 1.0);
            sum += w * textureLoad(ao_raw, q, 0).xy;
        }
    }
    if (sum.y <= 1e-4) {
        let own = textureLoad(ao_raw, texel, 0).xy;
        return vec4<f32>(select(1.0, own.x / own.y, own.y > 1e-4));
    }
    return vec4<f32>(sum.x / sum.y);
}

@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(pos.xy);
    let d = textureLoad(ao_frame_depth, px, 0);
    if (d >= 1.0) {
        discard;
    }
    let dist = vg_linear_depth(d);

    var vis: f32;
    if (ao.grid.z < 1.5) {
        vis = textureLoad(ao_clean, px, 0).x;
    } else {
        // AO texel i stands for full-resolution pixel 2i, so pixel `px` sits at `px / 2` on the
        // AO grid: on a texel for even coordinates, halfway between two for odd ones.
        let q = vec2<f32>(px) * 0.5;
        let base = vec2<i32>(floor(q));
        let f = q - vec2<f32>(base);
        let last = vec2<i32>(ao.grid.xy) - vec2<i32>(1);
        let t00 = clamp(base, vec2<i32>(0), last);
        let t10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0), last);
        let t01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0), last);
        let t11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0), last);
        let z00 = textureLoad(ao_distance, t00, 0).x;
        let z10 = textureLoad(ao_distance, t10, 0).x;
        let z01 = textureLoad(ao_distance, t01, 0).x;
        let z11 = textureLoad(ao_distance, t11, 0).x;
        let e = abs(vec4<f32>(z00, z10, z01, z11) - vec4<f32>(dist)) / dist;
        let a00 = textureLoad(ao_clean, t00, 0).x;
        let a10 = textureLoad(ao_clean, t10, 0).x;
        let a01 = textureLoad(ao_clean, t01, 0).x;
        let a11 = textureLoad(ao_clean, t11, 0).x;
        if (max(max(e.x, e.y), max(e.z, e.w)) < 0.02) {
            vis = mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
        } else {
            // A silhouette runs through this pixel's neighbourhood: take the texel on our surface.
            vis = a00;
            var best = e.x;
            if (e.y < best) {
                best = e.y;
                vis = a10;
            }
            if (e.z < best) {
                best = e.z;
                vis = a01;
            }
            if (e.w < best) {
                vis = a11;
            }
        }
    }

    var shade = pow(clamp(vis, 0.0, 1.0), ao.look.x);
    // Fade out towards the render distance, where the radius covers a pixel or two and depth
    // precision runs out; the screen-space fog (recorded later) hides it there anyway.
    let render_distance = camera.screen.w * 0.70710678;
    let fade_start = ao.look.y * render_distance;
    let fade_end = ao.look.z * render_distance;
    let fade = clamp((fade_end - dist) / max(fade_end - fade_start, 1e-3), 0.0, 1.0);
    shade = mix(1.0, shade, fade);
    // Below half an 8-bit step on a white pixel nothing would change: skip the blend.
    if (shade >= 0.998) {
        discard;
    }
    return vec4<f32>(shade, shade, shade, 1.0);
}
