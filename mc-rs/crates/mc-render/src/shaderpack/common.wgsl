// Shader packs: WGSL shared by every pack pass.
//
// Prepend `shaderpack::COMMON_WGSL` to a pass's own source. It declares structs and pure helper
// functions only, no bindings and no entry points, so it can sit in front of any module (the
// terrain variant, a post pass, another lane's pack). Bind the buffers yourself:
//
//   @group(N) @binding(M) var<uniform> pack: PackFrame;        // ShaderCore::frame_buffer()
//   @group(N) @binding(K) var<uniform> shadow: ShadowUniform;  // ShaderCore::shadow().uniform
//
// Conventions: world space is +X east, +Y up, +Z south; directions point toward the light;
// `*_display` colours are vanilla's display-encoded values (what the 8-bit frame holds), every
// other colour is linear light. Convert with `pk_to_linear` / `pk_to_display`.

struct PackFrame {
    // xyz: unit vector toward the sun. w: vanilla celestial angle (0 noon, .25 sunset,
    // .5 midnight, .75 sunrise).
    sun_dir: vec4<f32>,
    // xyz: toward the light that casts shadows (sun by day, moon by night). w: its direct
    // strength 0..1 (0 at the horizon switch, lower in rain).
    light_dir: vec4<f32>,
    // rgb: the direct light's colour (linear, unit luminance times strength). w: day 0..1.
    sun_color: vec4<f32>,
    // rgb: sky light tint (linear, unit luminance). w: golden hour 0..1.
    ambient_color: vec4<f32>,
    // rgb: vanilla fog colour (display). w: 0 linear fog, 1 exponential.
    fog_color: vec4<f32>,
    // x: linear fog start, y: end (blocks), z: exp density, w: 1 = radial distance.
    fog_params: vec4<f32>,
    // rgb: vanilla sky colour (display). w: rain 0..1.
    sky_color: vec4<f32>,
    // xyz: eye position wrapped into [0, 1024) (world-anchored noise must tile every 1024
    // blocks or a divisor). w: absolute eye y.
    eye: vec4<f32>,
    // x: render seconds, y: time of day in ticks (6000 noon), z: thunder 0..1, w: lightning.
    time: vec4<f32>,
    // x: eye medium (0 air, 1 water, 2 lava), y: dimension (0 overworld, -1 nether, 1 end),
    // z: vanilla sun brightness, w: star brightness.
    misc: vec4<f32>,
    // x: width, y: height, z: 1/width, w: 1/height (pixels).
    screen: vec4<f32>,
    // xy: sun position in screen UV (y down), z: 1 when in front of the camera,
    // w: 1 when the shadow light is the sun.
    sun_screen: vec4<f32>,
    // xy: moon UV, z: 1 when in front, w: render distance in blocks.
    moon_screen: vec4<f32>,
    // rgb: vanilla sunrise/sunset glow (display), a: its strength.
    sunrise: vec4<f32>,
    // xyz: toward the sun disc as the sky draws it (vanilla's path). `sun_dir` is the lighting
    // direction, which a pack's sun-path tilt may turn; anchor glints and rays on this one.
    sun_disc: vec4<f32>,
};

struct ShadowUniform {
    // Camera-relative world position -> shadow clip (xy -1..1, depth 0..1 = near the light).
    receiver: mat4x4<f32>,
    // Caster-origin-relative position -> shadow clip (the shadow pass's own matrix).
    caster: mat4x4<f32>,
    // x: texel size in blocks, y: softness in texels, z: strength 0..1, w: 1 / resolution.
    params: vec4<f32>,
    // x: fade start, y: fade end (blocks from the eye), z: distance, w: light strength.
    fade: vec4<f32>,
    // xyz: the (quantised) direction toward the light the map was rendered from.
    light: vec4<f32>,
};

const PK_PI: f32 = 3.14159265;

// Vanilla's display encoding is plain gamma; 2.2 matches what the frame's bytes mean.
fn pk_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn pk_to_display(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

fn pk_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Interleaved gradient noise: 0..1, blue-ish, cheap. Add `frame * 5.588238` to the
// pixel's x for a per-frame rotation that TAA averages away.
fn pk_ign(px: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(px, vec2<f32>(0.06711056, 0.00583715))));
}

// The direction of a pixel in world space (camera-relative, unit), from the camera uniform's
// rotation-only view and projection: `uv` 0..1 with y down.
fn pk_view_ray(uv: vec2<f32>, proj: mat4x4<f32>, view: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let v = vec3<f32>(
        (ndc.x + proj[2][0]) / proj[0][0],
        (ndc.y + proj[2][1]) / proj[1][1],
        -1.0,
    );
    // `view` is a pure rotation: its inverse is its transpose.
    let r = transpose(mat3x3<f32>(view[0].xyz, view[1].xyz, view[2].xyz));
    return normalize(r * v);
}
