// ItemRenderer.renderBlockInHand: the selected baked-model particle sprite on a 2x2 quad at eye
// z = -0.5, through renderHand's projection (the vertices arrive projected, with Java's mirrored
// U: left = maxU, right = minU; top = minV, bottom = maxV). Only the middle of the sprite is on
// screen. The block-atlas texel is modulated by GlStateManager.color(0.1, 0.1, 0.1, 0.5) with
// GL_BLEND and the alpha test off, so the alpha never matters: the quad is opaque, and the
// pipeline writes colour only.

struct OpaqueUniform {
    // GlStateManager.color(0.1, 0.1, 0.1, 0.5).
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> opaque: OpaqueUniform;
@group(0) @binding(1) var block_atlas: texture_2d<f32>;
@group(0) @binding(2) var block_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@location(0) clip: vec4<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.clip = clip;
    out.uv = uv;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(block_atlas, block_samp, in.uv);
    return vec4<f32>(texel.rgb * opaque.color.rgb, 1.0);
}
