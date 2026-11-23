// Clouds — S20, fast and fancy. java: RenderGlobal.renderClouds / renderCloudsFancy
//
// Same eye-relative-position + texture*vertex-colour shape as dome.wgsl; kept as a separate
// module/shader (rather than sharing one) so clouds stay independently toggleable
// (GameSettings.renderClouds) without pulling in the sky dome's pipeline.

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var cloud_tex: texture_2d<f32>;
@group(1) @binding(1) var cloud_samp: sampler;

struct CloudFogUniform {
    // rgb = fog colour, a = 0 for linear or 1 for exponential fog.
    color: vec4<f32>,
    // x = linear start, y = linear end, z = exponential density, w = distance metric.
    params: vec4<f32>,
    // x = Java's temporary cloud far plane; remaining lanes are reserved.
    projection: vec4<f32>,
};

@group(1) @binding(2) var<uniform> cloud_fog: CloudFogUniform;

// Fancy clouds: everything renderCloudsFancy computes per frame, one entry per distinct value
// (`FancyCloudFrame::table` in clouds.rs). The resident lattice stores indices into it, so the
// vertex stage only selects CPU-computed f32 values and sees the bits the per-frame CPU rebuild
// used to upload. xs/zs: 8 cells x 17 terms; ys: bottom, top, top face; uv_offset.xy: f17/f18;
// colors: top, bottom, X side, Z side.
struct FancyCloudTable {
    xs: array<vec4<f32>, 34>,
    zs: array<vec4<f32>, 34>,
    ys: vec4<f32>,
    uv_offset: vec4<f32>,
    colors: array<vec4<f32>, 4>,
};
@group(1) @binding(3) var<uniform> fancy: FancyCloudTable;

// java: Minecraft.startGame -> GlStateManager.alphaFunc(516, 0.1F) = glAlphaFunc(GL_GREATER, 0.1).
// GL_ALPHA_TEST is on for the whole world pass, so clouds.png's "empty" texels — which are
// (255,255,255,1), i.e. alpha 1/255, not 0 — are killed outright rather than blended at 0.4%.
// This is also what makes the depth pre-pass below meaningful: without it the pre-pass would
// stamp depth over the transparent parts of every quad.
const ALPHA_CUTOFF: f32 = 0.1;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    // `@invariant` so the depth pre-pass and the colour pass agree bit for bit: the colour pass
    // depth-tests GL_LEQUAL against depths the pre-pass wrote from the same vertices, and a
    // compiler free to reassociate the transform differently per pipeline would z-fight.
    @invariant @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    // Camera-space position used by both planar and radial fog distance modes.
    @location(2) eye_position: vec3<f32>,
};

struct FancyIn {
    // The frame-independent part of the UV (`cloud_uv / 256`); `f17`/`f18` is added here.
    @location(0) uv: vec2<f32>,
    // x | z << 8 | y << 16 | colour << 24, indices into `fancy`.
    @location(1) index: u32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    return cloud_vertex(in.pos, in.uv, in.color);
}

@vertex
fn vs_fancy(in: FancyIn) -> VsOut {
    let xi = in.index & 0xffu;
    let zi = (in.index >> 8u) & 0xffu;
    let yi = (in.index >> 16u) & 0xffu;
    let ci = (in.index >> 24u) & 0xffu;
    let pos = vec3<f32>(fancy.xs[xi >> 2u][xi & 3u], fancy.ys[yi], fancy.zs[zi >> 2u][zi & 3u]);
    // One correctly rounded addition, exactly `cloud_u * FANCY_TEXEL + f17` on the CPU (the
    // product is exact, so a fused multiply-add could not round differently either).
    return cloud_vertex(pos, in.uv + fancy.uv_offset.xy, fancy.colors[ci]);
}

fn cloud_vertex(pos: vec3<f32>, uv: vec2<f32>, color: vec4<f32>) -> VsOut {
    var out: VsOut;
    let eye = camera.view * vec4<f32>(pos, 1.0);
    var clip = camera.proj * eye;
    // Java rebuilds the projection around renderClouds with far = farPlaneDistance * 4,
    // then restores the terrain projection immediately afterward. Keep the real frame's x/y/w
    // terms (FOV, aspect, view effects, and anaglyph shift) and replace only the depth mapping.
    let near = camera.screen.z;
    let cloud_far = max(cloud_fog.projection.x, near + 0.000001);
    let a = cloud_far / (near - cloud_far);
    let b = near * cloud_far / (near - cloud_far);
    clip.z = a * eye.z + b;
    out.clip = clip;
    out.uv = uv;
    out.color = color;
    out.eye_position = eye.xyz;
    return out;
}

fn cloud_fog_factor(distance: f32) -> f32 {
    if (cloud_fog.color.a < 0.5) {
        let start = cloud_fog.params.x;
        let end = cloud_fog.params.y;
        if (end <= start) {
            return 1.0;
        }
        return clamp((end - distance) / (end - start), 0.0, 1.0);
    }
    return clamp(exp(-cloud_fog.params.z * distance), 0.0, 1.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(cloud_tex, cloud_samp, in.uv) * in.color;
    if (c.a <= ALPHA_CUTOFF) {
        discard;
    }
    let distance = select(-in.eye_position.z, length(in.eye_position), cloud_fog.params.w > 0.5);
    let fog_factor = cloud_fog_factor(distance);
    let rgb = cloud_fog.color.rgb * (1.0 - fog_factor) + c.rgb * fog_factor;
    return vec4<f32>(rgb, c.a);
}

// The fancy depth pre-pass. Its colour target has an empty write mask, so the only observable
// result is the depth write, i.e. `fs`'s discard decision — computed here by the same expression,
// without the fog maths `fs` would spend on a colour nobody stores.
@fragment
fn fs_prepass(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(cloud_tex, cloud_samp, in.uv) * in.color;
    if (c.a <= ALPHA_CUTOFF) {
        discard;
    }
    return c;
}
