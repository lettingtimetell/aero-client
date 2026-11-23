// Shader packs' terrain variant.
//
// Built as `shaderpack/common.wgsl` + `terrain.wgsl` + this file, into a SECOND shader module
// used only while a shader pack runs with a terrain feature on. The pack-off path compiles
// `terrain.wgsl` alone, exactly as before.
//
// Lighting model (linear light). Vanilla's light is `lightmap(block, sky)`; split it into the
// block-light part `lightmap(block, 0)` and the sky-light part (the rest). Shadows and the
// sun-facing term scale only the sky-light part, so torches, lava and glowstone light a shaded
// spot exactly as in vanilla. A fully sunlit surface facing the light gets vanilla's own value,
// so the pack changes contrast and colour, not the overall exposure.
//
//   sky   = lightmap(b, s) - lightmap(b, 0)
//   d     = sun share x relative light strength       (how much of daylight is direct)
//   vis   = N.L x shadow                               (0..1)
//   m     = (1 - d) x hemi(N) x ambient tint + d x vis x sun tint
//   light = lightmap(b, 0) + sky x max(m, floor x m_lit)

@group(2) @binding(0) var<uniform> pack: PackFrame;
@group(2) @binding(1) var<uniform> shadow: ShadowUniform;
@group(2) @binding(2) var shadow_map: texture_depth_2d;
@group(2) @binding(3) var shadow_samp: sampler_comparison;
@group(2) @binding(4) var<uniform> tp: TerrainPackUniform;

struct TerrainPackUniform {
    // x: 1 = shadows, y: sun share, z: shade floor, w: 1 = sun shading.
    light: vec4<f32>,
    // x: 1 = plants wave, y: 1 = leaves wave, z: amplitude, w: speed.
    waving: vec4<f32>,
    // x: 1 = shader water, y: wave strength, z: clarity, w: 1 = reflections.
    water: vec4<f32>,
    // rgb: water tint (linear), w: shadow strength.
    water_tint: vec4<f32>,
    // xyz: the shadow casters' origin wrapped to [0, 1024), w: unused.
    caster_origin: vec4<f32>,
};

// Material codes the mesher writes into `light.z` while flags are on (see `terrain/pack.rs`).
const MAT_PLANT: u32 = 1u;
const MAT_LEAVES: u32 = 2u;
const MAT_WATER: u32 = 3u;

struct VsOutPack {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) lm_uv: vec2<f32>,
    @location(3) eye_dist: f32,
    // Camera-relative world position as drawn (after waving).
    @location(4) rel: vec3<f32>,
    // The same position at rest (shadow lookups use it: the cached map holds plants at rest).
    @location(5) rest: vec3<f32>,
    @location(6) @interpolate(flat) material: u32,
};

// A wind sway: periodic in 1024 blocks (frequencies are whole multiples of 2pi/1024), so it is
// continuous across the eye wrap. `weight` is 0 at a plant's root, 1 at its top.
fn wave_offset(world: vec3<f32>, material: u32, weight: f32) -> vec3<f32> {
    var amp = 0.0;
    if (material == MAT_PLANT && tp.waving.x > 0.5) {
        amp = 0.075 * weight;
    } else if (material == MAT_LEAVES && tp.waving.y > 0.5) {
        amp = 0.035;
    }
    if (amp <= 0.0) {
        return vec3<f32>(0.0);
    }
    amp = amp * tp.waving.z;
    let t = pack.time.x * tp.waving.w;
    let k = 6.28318530 / 1024.0;
    // Spatial periods of 16, 11.1 and 7.3 blocks (64, 92 and 140 cycles per 1024).
    let p1 = (world.x * 64.0 + world.z * 32.0) * k + t * 1.30;
    let p2 = (world.x * -40.0 + world.z * 92.0) * k + t * 1.95;
    let p3 = (world.x * 140.0 + world.z * -76.0) * k + t * 3.10;
    let gust = 0.65 + 0.35 * sin((world.x * 8.0 + world.z * 12.0) * k + t * 0.37);
    let sx = sin(p1) * 0.55 + sin(p3) * 0.20;
    let sz = sin(p2) * 0.45 + cos(p3) * 0.15;
    return vec3<f32>(sx, 0.15 * sin(p1 + p2), sz) * amp * gust;
}

@vertex
fn vs_pack(in: VsIn) -> VsOutPack {
    var out: VsOutPack;
    let rest = (in.pos - vec3<f32>(8.0)) * 1.000001 + vec3<f32>(8.0) + in.chunk_offset.xyz;
    let material = in.light.z;
    var rel = rest;
    if (material == MAT_PLANT || material == MAT_LEAVES) {
        rel += wave_offset(rest + pack.eye.xyz, material, f32(in.light.w) / 255.0);
    }
    out.clip = camera.view_proj * vec4<f32>(rel, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.lm_uv = vec2<f32>(
        (f32(in.light.y) + 8.0) / 256.0,
        (f32(in.light.x) + 8.0) / 256.0,
    );
    let eye = (camera.view * vec4<f32>(rel, 1.0)).xyz;
    out.eye_dist = select(-eye.z, length(eye), terrain.params.z > 0.5);
    out.rel = rel;
    out.rest = rest;
    out.material = material;
    return out;
}

// --- Shadows -------------------------------------------------------------------------------

// Fraction of the light reaching `rest` (1 = lit). `n` faces the camera; `ndl` = N.L.
fn shadow_visibility(rest: vec3<f32>, n: vec3<f32>, ndl: f32, px: vec2<f32>) -> f32 {
    let dist = length(rest);
    if (dist >= shadow.fade.y) {
        return 1.0;
    }
    // Normal offset: a texel and a half along the normal, more at grazing angles.
    let grazing = 1.0 - clamp(ndl, 0.0, 1.0);
    let offset = n * shadow.params.x * (1.2 + 2.5 * grazing);
    let sp = shadow.receiver * vec4<f32>(rest + offset, 1.0);
    let uv = vec2<f32>(sp.x * 0.5 + 0.5, 0.5 - sp.y * 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || sp.z >= 1.0) {
        return 1.0;
    }
    let reference = sp.z - 0.00012;
    let radius = max(shadow.params.y, 0.35) * shadow.params.w;
    let spin = pk_ign(px) * 6.28318530;
    var sum = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let r = sqrt((f32(i) + 0.5) / 8.0) * radius;
        let a = f32(i) * 2.39996323 + spin;
        let o = vec2<f32>(cos(a), sin(a)) * r;
        sum += textureSampleCompareLevel(shadow_map, shadow_samp, uv + o, reference);
    }
    let vis = sum / 8.0;
    let fade = smoothstep(shadow.fade.x, shadow.fade.y, dist);
    return mix(vis, 1.0, fade);
}

// --- Lighting ------------------------------------------------------------------------------

// The face normal from screen-space derivatives, toward the camera, snapped to an axis when it
// is one (terrain mostly is), so flat faces light evenly.
fn face_normal(rel: vec3<f32>) -> vec3<f32> {
    var n = normalize(cross(dpdy(rel), dpdx(rel)));
    if (dot(n, rel) > 0.0) {
        n = -n;
    }
    let a = abs(n);
    if (a.x > 0.985) {
        n = vec3<f32>(sign(n.x), 0.0, 0.0);
    } else if (a.y > 0.985) {
        n = vec3<f32>(0.0, sign(n.y), 0.0);
    } else if (a.z > 0.985) {
        n = vec3<f32>(0.0, 0.0, sign(n.z));
    }
    return n;
}

// Vanilla's fixed per-face shade (top 1, north/south 0.8, east/west 0.6, bottom 0.5), as a
// smooth function of the normal: exact on axis-aligned faces. Plants and leaves (vanilla draws
// cross models unshaded) get 1.
fn vanilla_face_shade(n: vec3<f32>, material: u32) -> f32 {
    if (material == MAT_PLANT || material == MAT_LEAVES) {
        return 1.0;
    }
    let n2 = n * n;
    return n2.y * select(0.5, 1.0, n.y > 0.0) + n2.x * 0.6 + n2.z * 0.8;
}

// The light reaching a surface (linear): vanilla's lightmap split into block and sky parts,
// the sky part shaded by direction and shadow. While sun shading is on the mesh carries no
// face shade, so vanilla's is applied here to the block-light part (torch-lit caves look exactly
// like vanilla) and to the sky's ambient part.
fn pack_light(lm_uv: vec2<f32>, n: vec3<f32>, material: u32, rest: vec3<f32>, px: vec2<f32>)
    -> vec3<f32> {
    let full = pk_to_linear(textureSampleLevel(lightmap_tex, lightmap_samp, lm_uv, 0.0).rgb);
    let block_uv = vec2<f32>(lm_uv.x, 8.0 / 256.0);
    let block = pk_to_linear(textureSampleLevel(lightmap_tex, lightmap_samp, block_uv, 0.0).rgb);
    let sky = max(full - block, vec3<f32>(0.0));
    let shading = tp.light.w > 0.5;
    let shadows = tp.light.x > 0.5;
    if (!shading && !shadows) {
        return full;
    }
    let l = pack.light_dir.xyz;
    let is_sun = pack.sun_screen.w > 0.5;
    let peak = select(0.35, 1.0, is_sun);
    let rel_strength = clamp(pack.light_dir.w / peak, 0.0, 1.0);
    // Plants and leaves scatter light through themselves: no hard terminator.
    var ndl: f32;
    if (material == MAT_PLANT || material == MAT_LEAVES) {
        ndl = 0.55 + 0.45 * clamp(l.y, 0.0, 1.0);
    } else {
        ndl = clamp(dot(n, l), 0.0, 1.0);
    }
    // The cast shadow alone: 1 = lit, down to 1 - strength in full shadow. Only faces turned
    // toward the light can be in a cast shadow; strength 0 is exactly no shadow.
    var shadow_k = 1.0;
    if (shadows && ndl > 0.0 && rel_strength > 0.0) {
        let s = shadow_visibility(rest, n, ndl, px);
        shadow_k = mix(1.0, s, tp.water_tint.w);
    }
    if (!shading) {
        // Shadows alone. Vanilla's face shading stays in the vertex colour and already darkens
        // faces turned away from the sun, so a cast shadow takes away the sun's share of the sky
        // light on the faces it covers and nothing else: a wall facing away from the sun keeps
        // all of its sky light. The shadow fades in over the first degrees of a face turning
        // toward the light, so grazing walls never pick up the shadow test's edge noise.
        let d = tp.light.y * rel_strength;
        let toward = smoothstep(0.0, 0.15, ndl);
        let k = 1.0 - d * (1.0 - shadow_k) * toward;
        return block + sky * k;
    }
    let vis = ndl * shadow_k;
    let d = tp.light.y * rel_strength;
    let face = vanilla_face_shade(n, material);
    // Sky light comes from the whole dome: a little softer than vanilla's per-face steps.
    let hemi = mix(face, 0.78 + 0.22 * n.y, 0.35);
    let ambient_tint = pack.ambient_color.rgb;
    let sun_tint = pack.sun_color.rgb / max(pk_luma(pack.sun_color.rgb), 1e-3);
    // Direct sun a little brighter than vanilla's daylight: sunlit faces glow, the tone map's
    // shoulder rolls the excess off.
    let m = (1.0 - d) * hemi * ambient_tint + d * vis * sun_tint * 1.3;
    let m_lit = (1.0 - d) * hemi + d;
    let floor_m = tp.light.z * m_lit;
    let m_floor = max(m, vec3<f32>(floor_m) * ambient_tint);
    return block * face + sky * m_floor;
}

// Vanilla's fog, at the fragment's own distance, in display space (as GL_FOG blends).
fn pack_fog(display: vec3<f32>, eye_dist: f32) -> vec3<f32> {
    if (terrain.params.y <= 0.5) {
        return display;
    }
    var factor: f32;
    if (terrain.fog_color.a < 0.5) {
        let start = terrain.fog_params.x;
        let end = terrain.fog_params.y;
        if (end <= start) {
            factor = 1.0;
        } else {
            factor = clamp((end - eye_dist) / (end - start), 0.0, 1.0);
        }
    } else {
        factor = clamp(exp(-terrain.fog_params.z * eye_dist), 0.0, 1.0);
    }
    return mix(terrain.fog_color.rgb, display, factor);
}

fn pack_shade(in: VsOutPack, texel: vec4<f32>, n: vec3<f32>) -> vec4<f32> {
    let albedo = texel * in.color;
    let light = pack_light(in.lm_uv, n, in.material, in.rest, in.clip.xy);
    let lit = pk_to_display(pk_to_linear(albedo.rgb) * light);
    return vec4<f32>(pack_fog(lit, in.eye_dist), albedo.a);
}

@fragment
fn fs_pack_solid(in: VsOutPack) -> @location(0) vec4<f32> {
    let n = face_normal(in.rel);
    return pack_shade(in, textureSample(atlas_tex, atlas_samp, in.uv), n);
}

@fragment
fn fs_pack_cutout(in: VsOutPack) -> @location(0) vec4<f32> {
    let n = face_normal(in.rel);
    let texel = textureSample(atlas_tex, atlas_samp, in.uv);
    if (texel.a <= terrain.params.x) {
        discard;
    }
    return pack_shade(in, texel, n);
}

// --- Water ---------------------------------------------------------------------------------

@group(3) @binding(0) var scene_tex: texture_2d<f32>;
@group(3) @binding(1) var scene_depth: texture_depth_2d;

// The water surface's slope (d height / dx, d height / dz), analytic, from four travelling waves
// periodic in 1024 blocks (whole cycles per 1024, so the eye wrap is seamless).
fn water_slope(p: vec2<f32>, t: f32) -> vec2<f32> {
    let k = 6.28318530 / 1024.0;
    // (cycles in x, cycles in z, speed, amplitude in blocks)
    var waves = array<vec4<f32>, 4>(
        vec4<f32>(96.0, 40.0, 1.1, 0.030),
        vec4<f32>(-52.0, 130.0, 1.5, 0.024),
        vec4<f32>(210.0, 150.0, 2.3, 0.012),
        vec4<f32>(-300.0, -260.0, 3.1, 0.008),
    );
    var slope = vec2<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let w = waves[i];
        let phase = (p.x * w.x + p.y * w.y) * k + t * w.z;
        slope += w.w * cos(phase) * vec2<f32>(w.x, w.y) * k;
    }
    return slope;
}

fn water_normal(world: vec3<f32>) -> vec3<f32> {
    let s = water_slope(world.xz, pack.time.x) * (tp.water.y * 4.0);
    return normalize(vec3<f32>(-s.x, 1.0, -s.y));
}

fn linear_depth(d: f32) -> f32 {
    return camera.proj[3][2] / (d + camera.proj[2][2]);
}

// The sky seen along a direction: vanilla's fog colour at the horizon to the sky colour above,
// plus the sun's glint.
fn sky_along(dir: vec3<f32>) -> vec3<f32> {
    let horizon = pk_to_linear(pack.fog_color.rgb);
    let zenith = pk_to_linear(pack.sky_color.rgb);
    var c = mix(horizon, zenith, smoothstep(0.0, 0.45, dir.y));
    let sun = pack.sun_disc.xyz;
    let glint = pow(max(dot(dir, sun), 0.0), 600.0) * pack.light_dir.w
        * select(0.0, 1.0, pack.sun_screen.w > 0.5);
    c += pack.sun_color.rgb * glint * 3.0;
    return c;
}

// Where a camera-relative point lands: screen UV (y down) and its planar distance. `w <= 0` is
// behind the camera.
fn ssr_project(p: vec3<f32>) -> vec4<f32> {
    let clip = camera.view_proj * vec4<f32>(p, 1.0);
    let ndc = clip.xy / max(clip.w, 1e-4);
    return vec4<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, clip.w, 0.0);
}

// Screen-space reflection: march the reflected ray against the opaque depth in growing steps,
// refine the crossing by bisection, and return the scene colour with a confidence in `w`
// (0 = use the sky): it fades toward the screen edges, with the march distance and on rays that
// pass behind thin geometry, so a miss blends into the sky instead of cutting off.
fn reflect_ssr(origin_rel: vec3<f32>, dir: vec3<f32>, px: vec2<f32>) -> vec4<f32> {
    let start_dist = length(origin_rel);
    var step_len = clamp(start_dist * 0.04, 0.25, 2.0);
    var prev = origin_rel;
    var p = origin_rel + dir * step_len * (0.3 + 0.7 * pk_ign(px));
    var travelled = 0.0;
    for (var i = 0u; i < 24u; i++) {
        let s = ssr_project(p);
        if (s.z <= 0.0 || any(s.xy < vec2<f32>(0.0)) || any(s.xy > vec2<f32>(1.0))) {
            return vec4<f32>(0.0);
        }
        let scene_d = textureLoad(scene_depth, vec2<i32>(s.xy * pack.screen.xy), 0);
        let scene_dist = linear_depth(scene_d);
        if (scene_d < 1.0 && s.z > scene_dist) {
            // Bisect between the last point in front and this one.
            var a = prev;
            var b = p;
            for (var j = 0u; j < 5u; j++) {
                let m = (a + b) * 0.5;
                let sm = ssr_project(m);
                let dm = linear_depth(textureLoad(scene_depth, vec2<i32>(sm.xy * pack.screen.xy), 0));
                if (sm.z > dm) {
                    b = m;
                } else {
                    a = m;
                }
            }
            let hit = ssr_project(b);
            let hit_px = vec2<i32>(hit.xy * pack.screen.xy);
            let hit_dist = linear_depth(textureLoad(scene_depth, hit_px, 0));
            // A ray far behind the surface it crossed went under thin geometry: no hit.
            let behind = hit.z - hit_dist;
            let thick = 1.0 - smoothstep(step_len * 1.5, step_len * 4.0, behind);
            let edge = min(min(hit.x, 1.0 - hit.x), min(hit.y, 1.0 - hit.y));
            let fade = smoothstep(0.0, 0.1, edge) * (1.0 - smoothstep(30.0, 60.0, travelled));
            let c = pk_to_linear(textureLoad(scene_tex, hit_px, 0).rgb);
            return vec4<f32>(c, fade * thick);
        }
        prev = p;
        travelled += step_len;
        step_len *= 1.12;
        p += dir * step_len;
    }
    return vec4<f32>(0.0);
}

fn water_shade(in: VsOutPack, texel: vec4<f32>, n_geo: vec3<f32>) -> vec4<f32> {
    let world = in.rel + pack.eye.xyz;
    var n = n_geo;
    if (n_geo.y > 0.5) {
        n = water_normal(world);
    }
    let view_dir = normalize(in.rel);
    // Thickness of water along the view ray, from the opaque depth behind it.
    let ipx = vec2<i32>(in.clip.xy);
    let behind = textureLoad(scene_depth, ipx, 0);
    let surface_dist = in.clip.w;
    var thickness = 64.0;
    if (behind < 1.0) {
        thickness = max(linear_depth(behind) - surface_dist, 0.0);
    }
    let albedo = texel * in.color;
    let light = pack_light(in.lm_uv, n_geo, MAT_WATER, in.rest, in.clip.xy);
    // Vanilla's water is a deep saturated blue; pull it toward a cleaner teal so the reflections
    // and what lies below read through it.
    let lin = pk_to_linear(albedo.rgb);
    let teal = vec3<f32>(pk_luma(lin)) * vec3<f32>(0.55, 0.95, 1.25);
    let water_color = mix(lin, teal, 0.6) * 0.85 * tp.water_tint.rgb * light;
    // Clear near the surface, vanilla's opacity in deep water.
    let base_alpha = albedo.a;
    let clear_alpha = base_alpha * mix(1.0, 0.3, tp.water.z);
    let depth_mix = 1.0 - exp(-thickness * 0.22);
    let alpha = mix(clear_alpha, max(base_alpha, 0.8), depth_mix);
    // Fresnel (Schlick, F0 = 0.03).
    let cos_v = clamp(dot(-view_dir, n), 0.0, 1.0);
    let fresnel = 0.03 + 0.97 * pow(1.0 - cos_v, 5.0);
    var r = reflect(view_dir, n);
    // Never reflect below the horizon (a strong wave normal at a grazing angle).
    r.y = max(r.y, 0.02);
    r = normalize(r);
    var refl = sky_along(r);
    // The sky light reaching the surface dims the sky reflection in caves and at night.
    let sky_level = clamp(in.lm_uv.y * 256.0 / 240.0, 0.0, 1.0);
    refl = refl * mix(0.15, 1.0, sky_level * sky_level);
    if (tp.water.w > 0.5 && n_geo.y > 0.5) {
        let hit = reflect_ssr(in.rel, r, in.clip.xy);
        refl = mix(refl, hit.rgb, hit.w);
    }
    // What is behind keeps (1 - alpha)(1 - fresnel); the straight-alpha blend gets the rest.
    let out_alpha = alpha + (1.0 - alpha) * fresnel;
    let colour = (water_color * alpha * (1.0 - fresnel) + refl * fresnel) / max(out_alpha, 1e-3);
    return vec4<f32>(pack_fog(pk_to_display(colour), in.eye_dist), out_alpha);
}

@fragment
fn fs_pack_translucent(in: VsOutPack) -> @location(0) vec4<f32> {
    let n = face_normal(in.rel);
    let texel = textureSample(atlas_tex, atlas_samp, in.uv);
    if (texel.a <= terrain.params.x) {
        discard;
    }
    if (in.material == MAT_WATER && tp.water.x > 0.5) {
        return water_shade(in, texel, n);
    }
    return pack_shade(in, texel, n);
}

// The translucent layer without scene inputs (shader water off): shading and shadows only.
@fragment
fn fs_pack_translucent_plain(in: VsOutPack) -> @location(0) vec4<f32> {
    let n = face_normal(in.rel);
    let texel = textureSample(atlas_tex, atlas_samp, in.uv);
    if (texel.a <= terrain.params.x) {
        discard;
    }
    return pack_shade(in, texel, n);
}

// --- The shadow pass -------------------------------------------------------------------------

struct VsShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Casters at rest (no waving: the map is cached between changes), relative to the casters'
// origin.
@vertex
fn vs_shadow(in: VsIn) -> VsShadowOut {
    var out: VsShadowOut;
    let rel = (in.pos - vec3<f32>(8.0)) * 1.000001 + vec3<f32>(8.0) + in.chunk_offset.xyz;
    out.clip = shadow.caster * vec4<f32>(rel, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_shadow_cutout(in: VsShadowOut) {
    let a = textureSampleLevel(atlas_tex, atlas_samp, in.uv, 0.0).a;
    if (a <= 0.5) {
        discard;
    }
}
