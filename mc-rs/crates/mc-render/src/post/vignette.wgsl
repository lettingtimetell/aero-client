// World-border branch of GuiIngame.renderVignette.
//
// Java's four POSITION_TEX vertices cover the scaled framebuffer. `@builtin(position)` is in
// framebuffer pixels with the same top-left origin as the UI phase, so the UV is a plain divide.
struct VignetteUniform {
    color: vec4<f32>,
    viewport: vec4<f32>, // x = width, y = height
};
@group(0) @binding(0) var<uniform> vignette: VignetteUniform;
@group(0) @binding(1) var vignette_tex: texture_2d<f32>;
@group(0) @binding(2) var vignette_samp: sampler;

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
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(pos.x / vignette.viewport.x, pos.y / vignette.viewport.y);
    return textureSample(vignette_tex, vignette_samp, uv) * vignette.color;
}
