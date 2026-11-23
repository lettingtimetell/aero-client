// Nether-portal screen overlay — S20. java: GuiIngame.renderPortal (line 1003)
//
// Vanilla stretches *one animation frame* of `textures/blocks/portal.png` — whatever frame the
// block atlas currently holds, since `TextureMap.updateAnimations` re-uploads it in place every
// tick — across the whole screen as a single quad, with `color(1, 1, 1, timeInPortal)` and
// `tryBlendFuncSeparate(770, 771, 1, 0)` (straight alpha "over"). The swirl is therefore entirely
// the texture's own 32-frame animation; there is no procedural distortion.
//
// UV mapping matches the Java's four `.tex(...)` calls: screen top-left is the frame's top-left
// (minU, minV), screen bottom-right is (maxU, maxV). `@builtin(position)` is already in
// framebuffer pixels with (0,0) at the top-left, so the mapping is a plain divide.

struct PortalUniform {
    // x = alpha (the ramped `timeInPortal`), y = viewport width, z = viewport height, w unused.
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> portal: PortalUniform;
@group(0) @binding(1) var portal_tex: texture_2d<f32>;
@group(0) @binding(2) var portal_samp: sampler;

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
    let uv = vec2<f32>(pos.x / portal.params.y, pos.y / portal.params.z);
    let texel = textureSample(portal_tex, portal_samp, uv);
    // java: GlStateManager.color(1,1,1,timeInPortal) modulates the sampled texel, so rgb passes
    // through and only alpha is scaled.
    return vec4<f32>(texel.rgb, texel.a * portal.params.x);
}
