// Sky dome — S20. Sky/void plane, sunset glow, sun, moon, stars, End skybox.
//
// All geometry is **eye-relative** (see dome.rs), mirroring vanilla's own behaviour of
// recentring the sky's display lists on the player before drawing — so, unlike terrain, no
// chunk-offset term is needed here: position 0 in these buffers already means "at the eye".
// The planes and stars are baked once; a per-draw `DrawIn` record replaces their colour and Y
// (vanilla's `color(..)` + `translate(0, dy, 0)` around a display list).
//
// java: RenderGlobal.renderSky / renderStars / renderSkyEnd

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var dome_tex: texture_2d<f32>;
@group(1) @binding(1) var dome_samp: sampler;

// GL_FOG for the sky pass. java: EntityRenderer.setupFog(-1, partialTicks), enabled around the
// sky plane (RenderGlobal.java:1231-1248) and again around the horizon plane (:1349), and left
// OFF for the sunrise fan, sun, moon and stars. `mode` selects which: 0 = off, 1 = GL_LINEAR,
// 2 = GL_EXP. Distance is the **radial** eye distance, which is what vanilla asks for via
// GL_EYE_RADIAL_NV (EntityRenderer.java:2018-2021).
struct SkyFog {
    // rgb = fog colour, a = mode.
    color:  vec4<f32>,
    // x = linear start, y = linear end, z = exp density, w unused.
    params: vec4<f32>,
};
@group(2) @binding(0) var<uniform> sky_fog: SkyFog;
// Java's EntityRenderer.renderWorldPass temporarily replaces the projection's far plane with
// `farPlaneDistance * 2.0` while RenderGlobal.renderSky draws. The terrain projection remains in
// the shared camera uniform; this phase-local matrix is the equivalent temporary GL_PROJECTION.
@group(2) @binding(1) var<uniform> sky_projection: mat4x4<f32>;

// The frame's clear colour, as geometry. java: EntityRenderer.updateFogColor ends with
// `GlStateManager.clearColor(fogColorRed, fogColorGreen, fogColorBlue, 0)`, and renderSky then
// paints over it — the sky plane (at +16, rim 2.4 degrees above the horizon) and the horizon
// plane (at World.getHorizon(), rim just below it) do NOT meet, and it is the clear colour that
// fills the seam. Painting it here rather than relying on PhaseList::clear_color keeps the sky
// self-contained: the phase no longer depends on its caller having set a matching clear colour.
@vertex
fn vs_backdrop(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 1.0, 1.0);
}

@fragment
fn fs_backdrop() -> @location(0) vec4<f32> {
    return vec4<f32>(sky_fog.color.rgb, 1.0);
}

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) eye_pos: vec3<f32>,
};

// `DomeDraw` in dome.rs, stepped per instance; `first_instance` picks the draw's record.
// Both overrides *replace* a value, they never do arithmetic: a baked plane's vertices all share
// one Y, and the CPU has already added that Y and the draw's translation exactly as the old
// per-frame rebuild did, so the vertex stage sees bit-identical positions and colours.
struct DrawIn {
    @location(3) tint: vec4<f32>,
    // x = 1: colour := tint. y = 1: position.y := z.
    @location(4) mode: vec4<f32>,
    // z = 1: rotate by Ry(-90) . Rx(angle), x = cos(angle), y = sin(angle) (the stars, opt-in).
    @location(5) rot: vec4<f32>,
};

@vertex
fn vs(in: VsIn, per_draw: DrawIn) -> VsOut {
    var out: VsOut;
    var pos = in.pos;
    if (per_draw.mode.y > 0.5) {
        pos.y = per_draw.mode.z;
    }
    if (per_draw.rot.z > 0.5) {
        // java: `rotate(-90, 0,1,0); rotate(angle * 360, 1,0,0)` — Rx first, then Ry(-90),
        // which maps (x, y, z) to (-z, y, x).
        let c = per_draw.rot.x;
        let s = per_draw.rot.y;
        let r = vec3<f32>(pos.x, pos.y * c - pos.z * s, pos.y * s + pos.z * c);
        pos = vec3<f32>(-r.z, r.y, r.x);
    }
    var color = in.color;
    if (per_draw.mode.x > 0.5) {
        color = per_draw.tint;
    }
    // `camera.view` is rotation-only (see the `Camera` docs), which is exactly what an
    // eye-relative sky dome wants: it turns with the player's look but never translates away.
    out.clip = sky_projection * camera.view * vec4<f32>(pos, 1.0);
    out.uv = in.uv;
    out.color = color;
    // Already eye-relative, so the position *is* the offset from the eye.
    out.eye_pos = pos;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(dome_tex, dome_samp, in.uv);
    var c = texel * in.color;
    if (c.a <= 0.003) {
        discard;
    }
    return sky_fogged(c, in.eye_pos);
}

// The opaque planes and the void box (`plane_pipeline`, no blending). Those draws always bind the
// 1x1 white texel and carry vertex alpha 1, so `fs`'s `texel * in.color` is exactly `in.color`
// and it never discards; skipping the fetch changes nothing.
@fragment
fn fs_plane(in: VsOut) -> @location(0) vec4<f32> {
    return sky_fogged(in.color, in.eye_pos);
}

fn sky_fogged(color: vec4<f32>, eye_pos: vec3<f32>) -> vec4<f32> {
    var c = color;
    let mode = sky_fog.color.a;
    if (mode > 0.5) {
        let d = length(eye_pos);
        var f = 1.0;
        if (mode < 1.5) {
            let span = max(sky_fog.params.y - sky_fog.params.x, 1e-4);
            f = clamp((sky_fog.params.y - d) / span, 0.0, 1.0);
        } else {
            f = clamp(exp(-sky_fog.params.z * d), 0.0, 1.0);
        }
        // GL_FOG blends colour only; alpha is untouched.
        c = vec4<f32>(mix(sky_fog.color.rgb, c.rgb, f), c.a);
    }
    return c;
}
