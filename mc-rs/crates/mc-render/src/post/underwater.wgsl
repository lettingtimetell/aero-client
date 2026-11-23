// java: ItemRenderer.renderWaterOverlayTexture (1.8.9 :505-532).
// The quad spans a four-by-four repeat of textures/misc/underwater.png. Java submits its left
// edge at U = 4 - yaw/64 and right edge at U = -yaw/64, hence the intentionally reversed U.

struct WaterUniform {
    // x = player brightness, y = yaw degrees, z = pitch degrees, w unused.
    camera: vec4<f32>,
    // x/y = framebuffer width/height.
    viewport: vec4<f32>,
};
@group(0) @binding(0) var<uniform> water: WaterUniform;
@group(0) @binding(1) var water_tex: texture_2d<f32>;
@group(0) @binding(2) var water_samp: sampler;

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
    let screen = vec2<f32>(pos.x / water.viewport.x, pos.y / water.viewport.y);
    let uv = vec2<f32>(
        4.0 - screen.x * 4.0 - water.camera.y / 64.0,
        screen.y * 4.0 + water.camera.z / 64.0,
    );
    let texel = textureSample(water_tex, water_samp, uv);
    // java: GlStateManager.color(f, f, f, 0.5F) — RGB is multiplied, alpha is fixed.
    return vec4<f32>(texel.rgb * water.camera.x, texel.a * 0.5);
}
