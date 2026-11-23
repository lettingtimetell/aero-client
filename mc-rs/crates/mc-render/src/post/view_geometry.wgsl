// View-geometry prepass. Appended to
// `view_geometry_common.wgsl`.
//
// Reads the shared depth buffer once and writes, per output texel:
//   @location(0) R32Float     planar eye distance in blocks (VG_SKY_DISTANCE where nothing drew)
//   @location(1) Rgba16Float  view-space unit normal (xyz, facing the camera), w = 1 geometry / 0 sky
//
// At scale 2 (half resolution) output texel (x, y) stands for full-resolution pixel (2x, 2y): one
// real depth sample, never an average, so the distance is a surface that exists and the normal
// is that surface's. The normal is reconstructed from full-resolution neighbours either way.
//
// `texture_depth_2d` is rewritten to `texture_depth_multisampled_2d` when the frame is
// multisampled; the third `textureLoad` argument is then the sample index, and 0 is right for
// both spellings.

struct VgParams {
    // x = output width, y = output height, z = full-resolution pixels per output texel (1 or 2).
    out_size: vec4<f32>,
};
@group(1) @binding(0) var vg_depth: texture_depth_2d;
@group(1) @binding(1) var<uniform> vg: VgParams;

struct VgOut {
    @location(0) distance: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

fn vg_load(p: vec2<i32>, size: vec2<i32>) -> f32 {
    return textureLoad(vg_depth, clamp(p, vec2<i32>(0), size - vec2<i32>(1)), 0);
}

@fragment
fn fs_prepass(@builtin(position) pos: vec4<f32>) -> VgOut {
    let size = vec2<i32>(textureDimensions(vg_depth));
    let c = vec2<i32>(pos.xy) * i32(vg.out_size.z);
    let d = vg_load(c, size);
    var out: VgOut;
    if (d >= 1.0) {
        out.distance = vec4<f32>(VG_SKY_DISTANCE, 0.0, 0.0, 0.0);
        out.normal = vec4<f32>(0.0, 0.0, 1.0, 0.0);
        return out;
    }

    let dl = vg_load(c + vec2<i32>(-1, 0), size);
    let dr = vg_load(c + vec2<i32>(1, 0), size);
    let dl2 = vg_load(c + vec2<i32>(-2, 0), size);
    let dr2 = vg_load(c + vec2<i32>(2, 0), size);
    let du = vg_load(c + vec2<i32>(0, -1), size);
    let dd = vg_load(c + vec2<i32>(0, 1), size);
    let du2 = vg_load(c + vec2<i32>(0, -2), size);
    let dd2 = vg_load(c + vec2<i32>(0, 2), size);

    let centre_px = vec2<f32>(c) + vec2<f32>(0.5);
    let dist = vg_linear_depth(d);
    let p = vg_view_position(centre_px, dist);

    // Device depth is affine in screen space across a plane, so on the side that continues the
    // centre's surface the two-tap extrapolation `2 * d1 - d2` lands back on `d`. Pick, per axis,
    // the side that does so best: across a silhouette or a crease the other side is a different
    // surface. Near the screen border only the in-bounds side is usable.
    let left_ok = c.x >= 2;
    let right_ok = c.x + 2 < size.x;
    let up_ok = c.y >= 2;
    let down_ok = c.y + 2 < size.y;
    let err_l = abs(2.0 * dl - dl2 - d);
    let err_r = abs(2.0 * dr - dr2 - d);
    let err_u = abs(2.0 * du - du2 - d);
    let err_d = abs(2.0 * dd - dd2 - d);
    let use_right = (right_ok && (!left_ok || err_r < err_l));
    let use_down = (down_ok && (!up_ok || err_d < err_u));

    var hx: vec3<f32>;
    if (use_right) {
        hx = vg_view_position(centre_px + vec2<f32>(1.0, 0.0), vg_linear_depth(dr)) - p;
    } else {
        hx = p - vg_view_position(centre_px - vec2<f32>(1.0, 0.0), vg_linear_depth(dl));
    }
    var hy: vec3<f32>;
    if (use_down) {
        hy = vg_view_position(centre_px + vec2<f32>(0.0, 1.0), vg_linear_depth(dd)) - p;
    } else {
        hy = p - vg_view_position(centre_px - vec2<f32>(0.0, 1.0), vg_linear_depth(du));
    }

    // +x screen is +x view, +y screen is -y view: cross(down, right) faces the camera.
    let cr = cross(hy, hx);
    let len2 = dot(cr, cr);
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (len2 > 1e-24) {
        n = cr * inverseSqrt(len2);
    }
    if (dot(n, p) > 0.0) {
        n = -n;
    }
    out.distance = vec4<f32>(dist, 0.0, 0.0, 0.0);
    out.normal = vec4<f32>(n, 1.0);
    return out;
}
