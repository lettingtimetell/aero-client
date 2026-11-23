// Rain, snow, and world-border texture/color draws. These retain the ordinary world
// projection and two-binding material layout; clouds own a separate fog/projection uniform.
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var weather_tex: texture_2d<f32>;
@group(1) @binding(1) var weather_samp: sampler;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @invariant @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = camera.proj * camera.view * vec4<f32>(in.pos, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(weather_tex, weather_samp, in.uv) * in.color;
    // Java's shared GL_GREATER alpha test, unchanged from the former shared shader.
    if (color.a <= 0.1) {
        discard;
    }
    return color;
}
