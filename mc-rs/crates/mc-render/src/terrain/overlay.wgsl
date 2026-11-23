// Block overlay shader — S11: the selection-box outline and the block-breaking crack texture.

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

// ---- selection outline ---------------------------------------------------------------------
// java: RenderGlobal.drawSelectionBoundingBox — black at 40% alpha, no texture, no depth write.

struct LineOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_line(@location(0) pos: vec3<f32>) -> LineOut {
    var out: LineOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    return out;
}

@fragment
fn fs_line() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.4);
}

// ---- breaking overlay ----------------------------------------------------------------------
// java: RenderGlobal.preRenderDamagedBlocks — GL_DST_COLOR/GL_SRC_COLOR blend at 50% colour,
// which darkens what is already there rather than painting over it.

@group(1) @binding(0) var crack_tex: texture_2d<f32>;
@group(1) @binding(1) var crack_samp: sampler;

struct CrackOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_crack(@location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>) -> CrackOut {
    var out: CrackOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_crack(in: CrackOut) -> @location(0) vec4<f32> {
    let texel = textureSample(crack_tex, crack_samp, in.uv);
    // java: `preRenderDamagedBlocks` sets `color(1, 1, 1, 0.5)` and `alphaFunc(GL_GREATER, 0.1)`;
    // fixed-function GL tests the *combined* alpha, so a texel survives only above 0.2. Vanilla's
    // own crack textures are alpha 1/255 or 255 (no difference); a pack's soft-edged cracks are.
    let alpha = texel.a * 0.5;
    if (alpha <= 0.1) {
        discard;
    }
    return vec4<f32>(texel.rgb, alpha);
}
