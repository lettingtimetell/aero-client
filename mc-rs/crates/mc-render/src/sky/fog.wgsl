// Screen-space distance fog — S20. java: the per-fragment half of `EntityRenderer.setupFog`,
// which in 1.8.9 is fixed-function `GL_FOG` applied during every terrain/entity/particle draw.
//
// Invariant I2 says the fixed-function pipeline is *rewritten*, not transliterated, and this is
// the rewrite: instead of every shader in the crate growing a fog term (which would mean editing
// `terrain/**` (S11), `entity/**` (S12) and `particle/**` (S18) — three modules S20 does not own),
// one full-screen pass reconstructs each fragment's eye-space depth from the shared depth buffer
// and blends the fog colour over the frame. The visible result is the same blend GL_FOG performs:
//   dst = fog.rgb * (1 - factor) + dst * factor
// which is exactly straight alpha-blending with alpha = 1 - factor, so the fragment shader only
// has to emit the fog colour and the factor.
//
// The `texture_depth_2d` below is rewritten to `texture_depth_multisampled_2d` at pipeline build
// time when `Gpu::sample_count() > 1`; `textureLoad`'s third argument is the mip level in the
// first case and the sample index in the second, and 0 is right for both.

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct FogUniform {
    // rgb = fog colour, a = 0 for FogMode::Linear, 1 for FogMode::Exp.
    color: vec4<f32>,
    // x = linear start, y = linear end, z = exp density, w = 0 planar / 1 radial distance.
    params: vec4<f32>,
};
@group(1) @binding(0) var<uniform> fog: FogUniform;
@group(1) @binding(1) var depth_tex: texture_depth_2d;

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
    let coord = vec2<i32>(pos.xy);
    let d = textureLoad(depth_tex, coord, 0);

    // Depth still at the cleared 1.0 means nothing opaque was drawn here: this pixel is sky.
    // Vanilla disables fog for the sky pass too, and the sky dome already paints the fog colour
    // at the horizon, which is what makes distant terrain appear to dissolve *into* the sky
    // rather than into a separate grey.
    //
    // Clouds do write depth (vanilla keeps it; see `clouds.rs`), through their own projection,
    // whose far plane is farther than the one inverted below. Read through this one, a cloud
    // lands nearer than it is (1/d = 1/z + 1/far - 1/far_clouds), inside the fog start until it
    // is past about 1.14x the render distance, where its own fog has already made it fully
    // fog-coloured: past there this pass only pulls the sky showing through it toward the fog.
    if (d >= 1.0) {
        discard;
    }

    // Invert the right-handed 0..1-depth perspective projection glam builds
    // (`Camera::update` -> `rh::proj::directx::perspective`):
    //   clip.z = proj[2][2] * z_eye + proj[3][2],  clip.w = -z_eye
    //   depth  = clip.z / clip.w   =>   -z_eye = proj[3][2] / (depth + proj[2][2])
    // `-z_eye` is the planar eye-space distance.  Java's NVIDIA reference reports
    // `GL_NV_fog_distance` + `GL_EYE_RADIAL_NV`, so the opt-in radial branch below reconstructs
    // the complete eye-to-fragment ray from this depth, the perspective scales, and the
    // fragment position. Depth already includes the prepared view's local camera effects;
    // reconstructing eye space does not require adding the world-space camera position.
    let planar_dist = camera.proj[3][2] / (d + camera.proj[2][2]);
    let ndc = pos.xy / camera.screen.xy * 2.0 - vec2<f32>(1.0, 1.0);
    let eye_x = ndc.x * planar_dist / camera.proj[0][0];
    let eye_y = ndc.y * planar_dist / camera.proj[1][1];
    let radial_dist = length(vec3<f32>(eye_x, eye_y, planar_dist));
    let dist = select(planar_dist, radial_dist, fog.params.w > 0.5);

    var factor: f32;
    if (fog.color.a < 0.5) {
        let start = fog.params.x;
        let end = fog.params.y;
        if (end <= start) {
            factor = 1.0;
        } else {
            factor = clamp((end - dist) / (end - start), 0.0, 1.0);
        }
    } else {
        factor = clamp(exp(-fog.params.z * dist), 0.0, 1.0);
    }

    // Inside the fog start nothing changes: alpha would be `1 - 1 = 0`, and straight alpha
    // blending with a zero source alpha returns the destination in every channel. Skipping the
    // blend there is the same image, minus the colour read-modify-write for every near pixel.
    if (factor >= 1.0) {
        discard;
    }

    return vec4<f32>(fog.color.rgb, 1.0 - factor);
}
