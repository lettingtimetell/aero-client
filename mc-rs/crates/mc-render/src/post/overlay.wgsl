// Full-screen tint overlay — S20. One flat colour, blended "over" the frame so far.
//
// Used for both the underwater/lava/void immersion tint and the gamma "brighten" pass — see
// overlay.rs's module docs for why both are expressed the same way (a solid alpha-blended quad)
// rather than needing to sample the already-rendered scene.

struct OverlayUniform {
    color: vec4<f32>, // rgb = tint colour, a = blend alpha
};
@group(0) @binding(0) var<uniform> overlay: OverlayUniform;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(overlay.color.rgb, overlay.color.a);
}
