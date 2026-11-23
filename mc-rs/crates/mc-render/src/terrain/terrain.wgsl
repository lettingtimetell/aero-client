// Terrain shader — S11.
//
// Geometry arrives in SECTION-LOCAL space (0..16) and is moved into camera-relative space by a
// per-draw instance attribute, because `Camera::view` is rotation-only and `view_proj` expects
// camera-relative positions (see the `Camera` docs). Doing the subtraction in f64 on the CPU
// and handing the shader a small f32 offset is what keeps terrain from jittering thousands of
// blocks from the origin.
//
// Three texture units' worth of vanilla state live here, in vanilla's own order:
//   final = atlas_texel * vertex_colour * lightmap_texel,  then fogged.
// `vertex_colour` is biome tint x per-face brightness x ambient occlusion, baked on the CPU by
// the mesher. `lightmap_texel` is S20's `EntityRenderer.updateLightmap` table, uploaded as a
// 16x16 RGBA8 texture and sampled with the interpolated (block, sky) coordinate exactly the way
// vanilla's second texture unit does. The fog term is `EntityRenderer.setupFog`'s GL_FOG,
// applied per draw rather than screen-space so a translucent surface is fogged at *its own*
// depth (see `sky::fog`'s module docs for why the screen-space pass cannot be).

struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_samp: sampler;

struct TerrainUniform {
    // x = alpha cutoff, y = 1 when the fog term below is live, z = 1 for radial fog distance
    // (else planar), w = reserved.
    params: vec4<f32>,
    // rgb = fog colour, a = 0 for FogMode::Linear and 1 for FogMode::Exp. Same packing as
    // `sky/fog.wgsl`'s uniform, deliberately, so the two agree by construction.
    fog_color: vec4<f32>,
    // x = linear start, y = linear end, z = exp density, w = unused.
    fog_params: vec4<f32>,
};
@group(1) @binding(2) var<uniform> terrain: TerrainUniform;

// java: EntityRenderer.lightmapTexture — 16 block-light columns x 16 sky-light rows, GL_LINEAR,
// clamped. `TerrainRenderer` uploads `sky::lightmap::Lightmap::rgba` straight into it.
@group(1) @binding(3) var lightmap_tex: texture_2d<f32>;
@group(1) @binding(4) var lightmap_samp: sampler;

// Vanilla `DefaultVertexFormats.BLOCK`: exact f32 position and UV values written by
// `WorldRenderer`, with no pre-upload quantisation.
struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) light: vec4<u32>,
    // Instance-stepped: xyz = section origin minus eye position, in f32.
    @location(4) chunk_offset: vec4<f32>,
};

// The old 20-byte `Unorm16` format, reachable only through `packed_vertices` as a regression
// control. It deliberately reconstructs the former grid so precision tests can reject it.
struct VsInPacked {
    @location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) light: vec4<u32>,
    @location(4) chunk_offset: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) lm_uv: vec2<f32>,
    // Planar eye-space distance, which is what fixed-function GL_FOG uses as its fog
    // coordinate on essentially every driver — and what `sky/fog.wgsl` reconstructs from depth.
    @location(3) eye_dist: f32,
};

// Everything after the attribute decode is shared by the faithful and regression-control paths.
fn shade_vertex(
    local: vec3<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    light: vec4<u32>,
    chunk_offset: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    let world = (local - vec3<f32>(8.0)) * 1.000001 + vec3<f32>(8.0) + chunk_offset.xyz;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    out.uv = uv;
    out.color = color;

    // java: EntityRenderer.enableLightmap's texture matrix — scale(1/256), translate(8,8,8), so
    // a lightmap coordinate of `level * 16` lands on the texel *centre* `(level + 0.5) / 16`.
    // The two vertex bytes are already vanilla's `level * 16` (see `terrain::light`), and the
    // lightmap image is indexed `sky << 4 | block`, i.e. block across and sky down.
    out.lm_uv = vec2<f32>(
        (f32(light.y) + 8.0) / 256.0,
        (f32(light.x) + 8.0) / 256.0,
    );

    // `Camera::view` is rotation-only and right-handed, so the eye looks down -Z.
    // `params.z == 1`: the radial eye distance instead, per vertex,
    // as Java's `GL_NV_fog_distance` / `GL_EYE_RADIAL_NV` computes its fog coordinate; the
    // client's screen-space world fog uses the same metric (`FogDistance::Radial`).
    let eye = (camera.view * vec4<f32>(world, 1.0)).xyz;
    out.eye_dist = select(-eye.z, length(eye), terrain.params.z > 0.5);
    return out;
}

@vertex
fn vs(in: VsIn) -> VsOut {
    return shade_vertex(in.pos, in.uv, in.color, in.light, in.chunk_offset);
}

// The old packed control path; see `VsInPacked`.
@vertex
fn vs_packed(in: VsInPacked) -> VsOut {
    // `round` first: the hardware hands back `code / 65535`, and recovering the integer exactly
    // before dividing by a power of two is what makes the decode bit-identical to the CPU-side
    // `decode_pos`/`decode_uv`. Without it the UNORM conversion's last ulp would leak into every
    // position, which is the difference between "byte-identical" and "close enough".
    let pos_code = round(in.pos.xyz * 65535.0);
    let local = pos_code * (1.0 / 2048.0) + vec3<f32>(-8.0);
    let uv = round(in.uv * 65535.0) * (1.0 / 32768.0);
    return shade_vertex(local, uv, in.color, in.light, in.chunk_offset);
}


@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_samp, in.uv);
    // java: GL_GREATER. `phase.rs` supplies 0.5 for CUTOUT_MIPPED/CUTOUT (the reference seeded
    // by EntityRenderer.renderWorld) and 0.1 for TRANSLUCENT (reset after the cutout passes).
    if (texel.a <= terrain.params.x) {
        discard;
    }
    return shade(in, texel);
}

// java: `RenderGlobal.renderBlockLayer` draws SOLID with `disableAlpha`, so nothing is rejected.
// Having no `discard` also lets the GPU settle depth before shading, which is cheaper when
// geometry overlaps.
@fragment
fn fs_solid(in: VsOut) -> @location(0) vec4<f32> {
    return shade(in, textureSample(atlas_tex, atlas_samp, in.uv));
}

// Tint, lightmap and fog, shared by every terrain layer.
fn shade(in: VsOut, texel: vec4<f32>) -> vec4<f32> {
    let lm = textureSample(lightmap_tex, lightmap_samp, in.lm_uv);
    var out = texel * in.color * vec4<f32>(lm.rgb, 1.0);

    if (terrain.params.y > 0.5) {
        var factor: f32;
        if (terrain.fog_color.a < 0.5) {
            let start = terrain.fog_params.x;
            let end = terrain.fog_params.y;
            if (end <= start) {
                factor = 1.0;
            } else {
                factor = clamp((end - in.eye_dist) / (end - start), 0.0, 1.0);
            }
        } else {
            factor = clamp(exp(-terrain.fog_params.z * in.eye_dist), 0.0, 1.0);
        }
        // java: GL_FOG blends `fogColor * (1 - factor) + fragment * factor` and leaves alpha
        // alone, so a translucent quad keeps its own transparency and only its colour fogs.
        out = vec4<f32>(mix(terrain.fog_color.rgb, out.rgb, factor), out.a);
    }
    return out;
}
