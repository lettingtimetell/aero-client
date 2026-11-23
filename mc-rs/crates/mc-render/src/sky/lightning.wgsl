// Lightning bolts — S20. java: RenderLightningBolt.doRender
//
// Untextured `POSITION_COLOR` geometry, exactly what the Java emits after
// `GlStateManager.disableTexture2D()`. The pale blue-white colour and its 0.3 alpha come from the
// vertices themselves (`sky::weather::BOLT_COLOR`); the *brightness* comes from the pipeline's
// additive blend (`GL_SRC_ALPHA, GL_ONE`), which is why four concentric tubes of a 30%-alpha
// colour read as a white-hot core rather than as a translucent tube.

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = camera.proj * camera.view * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
