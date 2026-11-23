// Block Overlay mod: the selection box's translucent fill and its
// wide outline. wgpu has no wide lines, so each of the box's twelve edges is one instance of a
// screen-space quad, expanded in this vertex shader to the outline width in framebuffer pixels.

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct Style {
    // Straight-alpha RGBA.
    outline: vec4<f32>,
    fill:    vec4<f32>,
    // x = half the outline width in framebuffer pixels.
    params:  vec4<f32>,
};
@group(1) @binding(0) var<uniform> style: Style;

// ---- fill: the six faces of the expanded box, back faces culled ---------------------------

@vertex
fn vs_fill(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_fill() -> @location(0) vec4<f32> {
    return style.fill;
}

// ---- outline: one quad per edge -------------------------------------------------------------

@vertex
fn vs_edge(
    @builtin(vertex_index) index: u32,
    @location(0) a: vec3<f32>,
    @location(1) b: vec3<f32>,
) -> @builtin(position) vec4<f32> {
    var ca = camera.view_proj * vec4<f32>(a, 1.0);
    var cb = camera.view_proj * vec4<f32>(b, 1.0);
    // Clip the segment to the near plane (z >= 0 in wgpu clip space) so both ends have w > 0
    // before the perspective divide. A segment wholly behind the camera collapses to a point
    // outside the clip volume.
    if (ca.z < 0.0 && cb.z < 0.0) {
        return vec4<f32>(0.0, 0.0, -1.0, 1.0);
    }
    if (ca.z < 0.0) {
        ca = mix(ca, cb, ca.z / (ca.z - cb.z));
    }
    if (cb.z < 0.0) {
        cb = mix(cb, ca, cb.z / (cb.z - ca.z));
    }
    let half_screen = max(camera.screen.xy, vec2<f32>(1.0)) * 0.5;
    let pa = ca.xy / ca.w * half_screen;
    let pb = cb.xy / cb.w * half_screen;
    var dir = pb - pa;
    let len = length(dir);
    if (len > 1e-4) {
        dir = dir / len;
    } else {
        dir = vec2<f32>(1.0, 0.0);
    }
    let normal = vec2<f32>(-dir.y, dir.x);
    // Two triangles: (start,-) (start,+) (end,-) and (end,-) (start,+) (end,+). No caps, like
    // OpenGL's wide lines.
    var end_of = array<f32, 6>(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    var side_of = array<f32, 6>(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    let at_end = end_of[index] > 0.5;
    let c = select(ca, cb, at_end);
    let offset_px = normal * side_of[index] * style.params.x;
    let offset_ndc = offset_px / half_screen;
    return vec4<f32>(c.xy + offset_ndc * c.w, c.z, c.w);
}

@fragment
fn fs_edge() -> @location(0) vec4<f32> {
    return style.outline;
}
