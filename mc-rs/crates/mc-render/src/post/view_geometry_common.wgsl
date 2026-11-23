// View-space geometry helpers shared by screen-space passes.
//
// Prepend this file to a pass's own WGSL (`post::view_geometry::COMMON_WGSL`). It declares the
// renderer-wide camera uniform at @group(0) @binding(0), a full-screen triangle `vs`, and the
// two maths every depth-reading effect needs:
//
//   * `vg_linear_depth(d)`: a Depth32Float sample (0 near .. 1 far, standard Z, `Camera::update`'s
//     right-handed `directx` perspective) to the planar eye distance `-z_view` in blocks;
//   * `vg_view_position(full_px, dist)`: a full-resolution pixel position (pixel centre = +0.5)
//     and its planar distance to the right-handed view-space point (camera at the origin, looking
//     down -Z, +Y up, rotation-only view: the same space `CameraUniform::view` maps into).
//
// Both read only `camera.proj` and `camera.screen`, so they stay correct for any projection the
// camera builds, including a sub-pixel jitter in proj[2][0] / proj[2][1] (TAA) and the FOV
// modifiers.

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

// What `ViewGeometry` stores for a pixel with no geometry (the depth buffer still at its cleared
// 1.0: sky, clouds, the void). Far beyond any AO radius, finite so arithmetic stays NaN-free.
const VG_SKY_DISTANCE: f32 = 1000000.0;
// Anything at or above this is sky.
const VG_SKY_LIMIT: f32 = 100000.0;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

// clip.z = P[2][2] * z + P[3][2], clip.w = -z  =>  -z = P[3][2] / (d + P[2][2]).
fn vg_linear_depth(d: f32) -> f32 {
    return camera.proj[3][2] / (d + camera.proj[2][2]);
}

// clip.x = P[0][0] * x + P[2][0] * z  =>  x = (ndc.x + P[2][0]) * dist / P[0][0]; same for y.
fn vg_view_position(full_px: vec2<f32>, dist: f32) -> vec3<f32> {
    let ndc = vec2<f32>(
        full_px.x / camera.screen.x * 2.0 - 1.0,
        1.0 - full_px.y / camera.screen.y * 2.0,
    );
    let x = (ndc.x + camera.proj[2][0]) * dist / camera.proj[0][0];
    let y = (ndc.y + camera.proj[2][1]) * dist / camera.proj[1][1];
    return vec3<f32>(x, y, -dist);
}
