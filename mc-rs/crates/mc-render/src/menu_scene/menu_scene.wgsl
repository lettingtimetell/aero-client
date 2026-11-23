// The Aero title screen's background. One module, many entry
// points; see `render.rs` for the pass order. Everything before the composite is linear HDR.

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    shadow_mat: mat4x4<f32>,
    cam_pos: vec4<f32>,    // xyz eye, w = seconds since the title screen appeared
    cam_fwd: vec4<f32>,    // xyz forward, w = near plane
    cam_right: vec4<f32>,  // xyz screen right, w = tan(fov_x / 2)
    cam_up: vec4<f32>,     // xyz screen up, w = tan(fov_y / 2)
    sun_dir: vec4<f32>,    // xyz toward the sun, w = intro 0..1
    sun_color: vec4<f32>,  // rgb sunlight, w = exposure
    screen: vec4<f32>,     // width, height, 1 / width, 1 / height
    box_min: vec4<f32>,    // interior volume; w = haze density per block
    box_max: vec4<f32>,    // w = light-shaft strength
    params: vec4<f32>,     // x = shadow texel, y = bloom strength, z = VHS strength, w = frame
    look: vec4<f32>,       // x = haze mix, y = glow, z = debug (1 = no post), w = left scrim
    clouds: vec4<f32>,     // x = volumetric clouds' share (0 = painted only), y/z = 1 / traced px
};

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;
@group(0) @binding(3) var shadow_tex: texture_depth_2d;
@group(0) @binding(4) var shadow_samp: sampler_comparison;
@group(0) @binding(5) var noise_tex: texture_2d<f32>;
@group(0) @binding(6) var noise_samp: sampler;
@group(0) @binding(7) var lin_samp: sampler;

@group(1) @binding(0) var tex_a: texture_2d<f32>;
@group(1) @binding(1) var tex_b: texture_2d<f32>;  // planar depth: textureLoad only
@group(1) @binding(2) var tex_c: texture_2d<f32>;
@group(1) @binding(3) var tex_d: texture_2d<f32>;

@group(2) @binding(0) var ms_depth: texture_depth_multisampled_2d;

const MAT_SOLID: u32 = 0u;
const MAT_QUARTZ: u32 = 1u;
const MAT_LEAVES: u32 = 2u;
const MAT_PLANT: u32 = 3u;
const MAT_VINE: u32 = 4u;
const MAT_GLASS: u32 = 5u;
const MAT_LIGHT: u32 = 6u;
const MAT_GROUND: u32 = 7u;
const MAT_GLASS_EDGE: u32 = 8u;

// Whether an atlas UV is on the outer texel ring of its 16x16 sprite: the glass frame proper.
fn on_sprite_border(uv: vec2<f32>) -> bool {
    let local = fract(uv * vec2<f32>(textureDimensions(atlas_tex)) / 16.0) * 16.0;
    return min(min(local.x, 16.0 - local.x), min(local.y, 16.0 - local.y)) < 1.0;
}

const PI: f32 = 3.14159265;

// ---------------------------------------------------------------------------- helpers

fn nz(p: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(noise_tex, noise_samp, p, 0.0);
}

fn fbm(p_in: vec2<f32>, octaves: i32) -> f32 {
    var p = p_in;
    var a = 0.5;
    var s = 0.0;
    var norm = 0.0;
    for (var i = 0; i < octaves; i = i + 1) {
        s = s + a * nz(p).r;
        norm = norm + a;
        p = mat2x2<f32>(1.62, 1.18, -1.18, 1.62) * p + vec2<f32>(0.371, 0.713);
        a = a * 0.5;
    }
    return s / norm;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Interleaved gradient noise: cheap per-pixel dither.
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

// Henyey-Greenstein, scaled so isotropic scattering is 1.
fn phase_hg(mu: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5);
}

fn view_ray(ndc: vec2<f32>) -> vec3<f32> {
    return normalize(
        F.cam_fwd.xyz + F.cam_right.xyz * ndc.x * F.cam_right.w + F.cam_up.xyz * ndc.y * F.cam_up.w,
    );
}

fn ray_box_exit(o: vec3<f32>, d: vec3<f32>) -> f32 {
    let dd = select(d, vec3<f32>(1e-5), abs(d) < vec3<f32>(1e-5));
    let t0 = (F.box_min.xyz - o) / dd;
    let t1 = (F.box_max.xyz - o) / dd;
    let tmax = max(t0, t1);
    return max(min(min(tmax.x, tmax.y), tmax.z), 0.0);
}

// ---------------------------------------------------------------------------- sky

fn sky_base(d: vec3<f32>) -> vec3<f32> {
    let sun = F.sun_dir.xyz;
    let h = d.y;
    let zenith = vec3<f32>(0.028, 0.13, 0.62);
    let horizon = vec3<f32>(0.40, 0.64, 1.0);
    var c = mix(zenith, horizon, pow(1.0 - clamp(h, 0.0, 1.0), 3.5));
    if (h < 0.0) {
        // Far below the horizon: a pale, hazy country.
        let g = smoothstep(0.0, -0.3, h);
        c = mix(horizon, vec3<f32>(0.40, 0.55, 0.52), g * 0.7);
    }
    let mu = max(dot(d, sun), 0.0);
    c = c + F.sun_color.rgb * (0.03 * pow(mu, 3.0) + 0.12 * pow(mu, 24.0) + 0.9 * pow(mu, 400.0));
    c = c + F.sun_color.rgb * 18.0 * smoothstep(0.99955, 0.99975, mu);
    return c;
}

fn cloud_shape(q: vec2<f32>) -> f32 {
    let cells = nz(q * 0.8).b * 0.6 + nz(q * 1.7 + vec2<f32>(0.31, 0.77)).b * 0.4;
    return cells * 0.62 + fbm(q * 2.1, 4) * 0.55;
}

// A layer of fair-weather cumulus ~600 blocks up, drifting on the wind. Returns premultiplied
// colour and coverage.
fn clouds(o: vec3<f32>, d: vec3<f32>) -> vec4<f32> {
    if (d.y < 0.01) {
        return vec4<f32>(0.0);
    }
    let t = F.cam_pos.w;
    let dist = (620.0 - o.y) / d.y;
    let p = o.xz + d.xz * dist;
    let wind = vec2<f32>(t * 5.5, t * 1.8);
    var q = (p + wind) * 0.00022;
    let warp = vec2<f32>(fbm(q * 0.8 + vec2<f32>(5.2, 1.3), 2), fbm(q * 0.8 + vec2<f32>(1.7, 9.2), 2)) - 0.5;
    q = q + warp * 0.2 + vec2<f32>(t * 0.0004, 0.0);
    // Separate rounded clumps (inverted cellular noise), billowed by gradient fbm: fair-weather
    // cumulus rather than a uniform sheet.
    let shape = cloud_shape(q);
    let cover = 0.70;
    var dens = smoothstep(cover, cover + 0.11, shape);
    let detail = fbm(q * 6.5 + vec2<f32>(t * 0.002, t * 0.001), 3);
    dens = clamp(dens - (1.0 - dens) * (detail - 0.3) * 1.6, 0.0, 1.0);
    if (dens <= 0.001) {
        return vec4<f32>(0.0);
    }
    // Self-shadowing: how much cloud lies between this point and the sun.
    let to_sun = normalize(F.sun_dir.xz + vec2<f32>(1e-4));
    let toward = cloud_shape(q + to_sun * 0.022);
    let occl = smoothstep(cover - 0.04, cover + 0.24, toward);
    let thick = clamp((shape - cover) * 3.6, 0.0, 1.0);
    let lit = vec3<f32>(1.95, 1.88, 1.78);
    let shade = vec3<f32>(0.52, 0.60, 0.76);
    var col = mix(lit, shade, clamp(occl * 0.85 + thick * 0.35, 0.0, 1.0));
    // Silver lining: thin edges glow toward the sun.
    let mu = max(dot(d, F.sun_dir.xyz), 0.0);
    col = col + F.sun_color.rgb * pow(mu, 6.0) * (1.0 - thick) * 0.45;
    // Aerial perspective toward the horizon.
    let fade = smoothstep(0.01, 0.22, d.y);
    col = mix(sky_base(d), col, 0.55 + 0.45 * fade);
    let a = dens * fade;
    return vec4<f32>(col * a, a);
}

// High, thin cirrus streaks above the cumulus.
fn cirrus(o: vec3<f32>, d: vec3<f32>) -> vec4<f32> {
    if (d.y < 0.02) {
        return vec4<f32>(0.0);
    }
    let t = F.cam_pos.w;
    let p = (o.xz + d.xz * ((1400.0 - o.y) / d.y)) * 0.00016 + vec2<f32>(t * 0.0012, t * 0.0003);
    let s = fbm(vec2<f32>(p.x * 0.35, p.y * 1.6), 4);
    let a = smoothstep(0.55, 0.85, s) * 0.35 * smoothstep(0.02, 0.3, d.y);
    return vec4<f32>(vec3<f32>(1.35, 1.35, 1.38) * a, a);
}

fn sky(d: vec3<f32>) -> vec3<f32> {
    let o = F.cam_pos.xyz;
    var c = sky_base(d);
    let ci = cirrus(o, d);
    c = c * (1.0 - ci.a) + ci.rgb;
    let cu = clouds(o, d);
    return c * (1.0 - cu.a) + cu.rgb;
}

// ---------------------------------------------------------------------------- blocks

struct SceneIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) normal: vec4<f32>,
    @location(4) extra: vec4<u32>,
};

struct SceneOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) world: vec3<f32>,
    @location(4) @interpolate(flat) material: u32,
    @location(5) skyvis: f32,
};

fn sway(p: vec3<f32>, weight: f32, material: u32) -> vec3<f32> {
    let t = F.cam_pos.w;
    let gust = 0.6 + 0.4 * sin(t * 0.37 + p.x * 0.05 - p.z * 0.04);
    let ph = p.x * 0.71 + p.z * 0.93 + p.y * 0.31;
    var amp = 0.025;
    if (material == MAT_PLANT) {
        amp = 0.085;
    } else if (material == MAT_VINE) {
        amp = 0.04;
    }
    let s = vec3<f32>(sin(t * 1.35 + ph), 0.25 * sin(t * 1.9 + ph * 1.3), cos(t * 1.05 + ph * 1.21));
    return s * amp * weight * gust;
}

@vertex
fn vs_scene(in: SceneIn) -> SceneOut {
    var o: SceneOut;
    let material = in.extra.x;
    let w = f32(in.extra.y) / 255.0;
    var p = in.pos;
    if (w > 0.0) {
        p = p + sway(in.pos, w, material);
    }
    o.clip = F.view_proj * vec4<f32>(p, 1.0);
    o.uv = in.uv;
    var tint = srgb_to_linear(in.color.rgb);
    if (material == MAT_LEAVES || material == MAT_VINE || material == MAT_PLANT) {
        // The jungle colormap is very saturated; pull it a little toward its own luma.
        tint = mix(tint, vec3<f32>(luma(tint)), 0.18);
    }
    o.color = tint;
    o.normal = in.normal.xyz;
    o.world = p;
    o.material = material;
    o.skyvis = f32(in.extra.z) / 255.0;
    return o;
}

fn shadow_at(world: vec3<f32>, n: vec3<f32>) -> f32 {
    let p = world + n * 0.05 + F.sun_dir.xyz * 0.02;
    let s = F.shadow_mat * vec4<f32>(p, 1.0);
    let uv = vec2<f32>(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0 || s.z >= 1.0) {
        return 1.0;
    }
    let k = F.params.x * 1.25;
    let z = s.z - 0.0002;
    var sum = textureSampleCompareLevel(shadow_tex, shadow_samp, uv + vec2<f32>(-0.7, -0.3) * k, z);
    sum = sum + textureSampleCompareLevel(shadow_tex, shadow_samp, uv + vec2<f32>(0.3, -0.7) * k, z);
    sum = sum + textureSampleCompareLevel(shadow_tex, shadow_samp, uv + vec2<f32>(0.7, 0.3) * k, z);
    sum = sum + textureSampleCompareLevel(shadow_tex, shadow_samp, uv + vec2<f32>(-0.3, 0.7) * k, z);
    return sum * 0.25;
}

fn light_surface(
    albedo: vec3<f32>,
    n_in: vec3<f32>,
    world: vec3<f32>,
    material: u32,
    skyvis: f32,
    front: bool,
) -> vec3<f32> {
    var n = normalize(n_in);
    if (!front) {
        n = -n;
    }
    let l = F.sun_dir.xyz;
    let v = normalize(F.cam_pos.xyz - world);
    let sh = shadow_at(world, n);
    let ndl = dot(n, l);
    var direct = max(ndl, 0.0);
    var through = vec3<f32>(0.0);
    var scatter = vec3<f32>(0.0);
    var vis = mix(0.3, 1.0, skyvis * skyvis);
    let foliage = material == MAT_LEAVES || material == MAT_PLANT || material == MAT_VINE;
    if (foliage) {
        // Thin leaves: wrapped diffuse, plus sunlight passing through toward the viewer, plus
        // light scattered through the canopy that reaches even the shaded leaves.
        direct = clamp((ndl + 0.5) / 1.5, 0.0, 1.0);
        let back = pow(clamp(dot(-v, l), 0.0, 1.0), 4.0) * 1.1 + 0.3 * max(-ndl, 0.0);
        through = vec3<f32>(0.85, 1.0, 0.45) * back;
        scatter = vec3<f32>(0.42, 0.55, 0.22) * mix(0.35, 1.0, skyvis);
        vis = mix(0.55, 1.0, skyvis);
    }
    // Sky light from above and green bounce from below, dimmed under canopies: a glass house
    // is full of bright diffuse light.
    let up = n.y * 0.5 + 0.5;
    let amb = mix(vec3<f32>(0.30, 0.38, 0.24), vec3<f32>(0.78, 0.88, 1.02), up);
    var c = albedo * (amb * vis + scatter + F.sun_color.rgb * sh * (vec3<f32>(direct) + through));
    if (material == MAT_GLASS || material == MAT_GLASS_EDGE) {
        // The panes' frames read as slim silver-blue mullions rather than white paint.
        c = c * vec3<f32>(0.66, 0.74, 0.80);
    }
    if (material == MAT_QUARTZ || material == MAT_GLASS || material == MAT_GLASS_EDGE) {
        let h = normalize(l + v);
        c = c + F.sun_color.rgb * pow(max(dot(n, h), 0.0), 60.0) * 0.18 * sh;
    }
    if (material == MAT_QUARTZ && n.y > 0.9) {
        // Polished walkway: a soft fresnel reflection of the bright sky and the glass above.
        let f = 0.04 + 0.96 * pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 5.0);
        c = c + sky_base(reflect(-v, n)) * f * 0.55 * vis;
    }
    if (material == MAT_LIGHT) {
        c = albedo * (amb * 0.5 + vec3<f32>(0.4)) + albedo * vec3<f32>(5.5, 6.0, 6.5);
    }
    return c;
}

@fragment
fn fs_opaque(in: SceneOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas_tex, atlas_samp, in.uv);
    let c = light_surface(tex.rgb * in.color, in.normal, in.world, in.material, in.skyvis, front);
    return vec4<f32>(c, 1.0);
}

@fragment
fn fs_cutout(in: SceneOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas_tex, atlas_samp, in.uv);
    // Sharpened alpha for alpha-to-coverage: crisp leaf and frame edges, antialiased by MSAA.
    var a = (tex.a - 0.5) / max(fwidth(tex.a), 0.0001) + 0.5;
    // Glass: only the outer ring is frame; the glare pixels inside belong to the glass pass.
    if (in.material == MAT_GLASS && !on_sprite_border(in.uv)) {
        a = 0.0;
    }
    let c = light_surface(tex.rgb * in.color, in.normal, in.world, in.material, in.skyvis, front);
    return vec4<f32>(c, clamp(a, 0.0, 1.0));
}

// The see-through part of glass blocks and panes: a faint reflection, stronger at grazing
// angles, with sun glints. Premultiplied.
@fragment
fn fs_glass(in: SceneOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas_tex, atlas_samp, in.uv);
    if (tex.a > 0.5 && on_sprite_border(in.uv)) {
        discard;
    }
    var n = normalize(in.normal);
    if (!front) {
        n = -n;
    }
    let v = normalize(F.cam_pos.xyz - in.world);
    let cos_t = clamp(abs(dot(n, v)), 0.0, 1.0);
    let fres = 0.035 + 0.965 * pow(1.0 - cos_t, 5.0);
    let env = sky_base(reflect(-v, n));
    let h = normalize(F.sun_dir.xyz + v);
    let glint = pow(max(dot(n, h), 0.0), 700.0) * 30.0;
    // Vanilla's glare streaks, kept as a faint sheen rather than opaque specks.
    let glare = tex.a * 0.07;
    let rgb = env * fres * 0.4 + F.sun_color.rgb * glint * fres + vec3<f32>(glare * 1.3);
    let a = clamp(fres * 0.35 + 0.01 + glare, 0.0, 1.0);
    return vec4<f32>(rgb, a);
}

struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) material: u32,
};

@vertex
fn vs_shadow(in: SceneIn) -> ShadowOut {
    var o: ShadowOut;
    o.clip = F.shadow_mat * vec4<f32>(in.pos, 1.0);
    o.uv = in.uv;
    o.material = in.extra.x;
    return o;
}

@fragment
fn fs_shadow_cutout(in: ShadowOut) {
    let a = textureSampleLevel(atlas_tex, atlas_samp, in.uv, 0.0).a;
    if (a < 0.5 || (in.material == MAT_GLASS && !on_sprite_border(in.uv))) {
        discard;
    }
}

// ---------------------------------------------------------------------------- full screen

struct FsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

// One triangle covering the screen at the far plane (reverse Z: depth 0).
@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> FsOut {
    let x = f32(i32(i & 1u) * 4 - 1);
    let y = f32(i32(i >> 1u) * 4 - 1);
    var o: FsOut;
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    o.ndc = vec2<f32>(x, y);
    return o;
}

@fragment
fn fs_sky(in: FsOut) -> @location(0) vec4<f32> {
    let d = view_ray(in.ndc);
    let share = F.clouds.x;
    // The painted clouds, until the volumetric ones are ready (and during the cross-fade).
    var painted = vec3<f32>(0.0);
    if (share < 1.0) {
        painted = sky(d);
    }
    if (share <= 0.0) {
        return vec4<f32>(painted, 1.0);
    }
    // The volumetric clouds (`sky/volumetric.rs`), traced
    // into their own small target before this pass, in front of the sky and the high cirrus.
    let o = F.cam_pos.xyz;
    var c = sky_base(d);
    let ci = cirrus(o, d);
    c = c * (1.0 - ci.a) + ci.rgb;
    let v = textureSampleLevel(tex_a, lin_samp, in.pos.xy * F.clouds.yz, 0.0);
    c = c * v.a + v.rgb;
    return vec4<f32>(mix(painted, c, share), 1.0);
}

// Multisampled reverse-Z depth to planar view depth (nearest of the samples).
@fragment
fn fs_depth(in: FsOut) -> @location(0) vec4<f32> {
    let c = vec2<i32>(in.pos.xy);
    var d = 0.0;
    let n = i32(textureNumSamples(ms_depth));
    for (var s = 0; s < n; s = s + 1) {
        d = max(d, textureLoad(ms_depth, c, s));
    }
    let z = select(1.0e6, F.cam_fwd.w / max(d, 1e-9), d > 0.0);
    return vec4<f32>(z, 0.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------- water

struct WaterIn {
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) params: vec4<f32>,
};

struct WaterOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) kind: u32,
    @location(2) normal: vec3<f32>,
    @location(3) seed: f32,
    @location(4) params: vec4<f32>,
};

@vertex
fn vs_water(in: WaterIn) -> WaterOut {
    var o: WaterOut;
    let kind = u32(in.pos.w + 0.5);
    var p = in.pos.xyz;
    if (kind == 2u) {
        let size = in.params.z;
        p = p + (F.cam_right.xyz * in.params.x + F.cam_up.xyz * in.params.y * 0.6) * size;
    }
    o.clip = F.view_proj * vec4<f32>(p, 1.0);
    o.world = p;
    o.kind = kind;
    o.normal = in.normal.xyz;
    o.seed = in.normal.w;
    o.params = in.params;
    return o;
}

fn planar_depth(world: vec3<f32>) -> f32 {
    return dot(world - F.cam_pos.xyz, F.cam_fwd.xyz);
}

fn scene_depth(frag: vec2<f32>) -> f32 {
    return textureLoad(tex_b, vec2<i32>(frag), 0).r;
}

fn fall_color(in: WaterOut, v: vec3<f32>, suv: vec2<f32>) -> vec4<f32> {
    let t = F.cam_pos.w;
    let u = in.params.x;
    let top = in.params.y;
    let pool_y = in.params.z;
    let width = in.params.w;
    let y = in.world.y;
    let fallen = top - y;
    let across = u * width * 0.5 + in.seed;
    let speed = 7.5;
    let s1 = nz(vec2<f32>(across * 0.55, (y + t * speed) * 0.020)).r;
    let s2 = nz(vec2<f32>(across * 1.3 + 0.37, (y + t * speed * 1.25) * 0.045)).g;
    let s3 = nz(vec2<f32>(across * 3.1 + 0.71, (y + t * speed * 1.6) * 0.11)).r;
    let streak = clamp(s1 * 0.45 + s2 * 0.4 + s3 * 0.3 - 0.33, 0.0, 1.0);

    let n0 = normalize(in.normal);
    let side = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), n0));
    let bump = vec2<f32>(s2 - 0.5, s3 - 0.5);
    // Bend the normal toward the face edges so the square column reads as rounded water.
    let edge = u * 2.0 - 1.0;
    var n = normalize(n0 + side * (edge * 0.75 + bump.x * 0.8) + vec3<f32>(0.0, bump.y * 0.25, 0.0));

    // Refraction: look through the water at a displaced point of the scene behind it.
    let off = vec2<f32>(dot(n - n0, F.cam_right.xyz), -dot(n - n0, F.cam_up.xyz)) * 0.045;
    let behind = textureSampleLevel(tex_a, lin_samp, suv + off, 0.0).rgb;
    let behind2 = textureSampleLevel(tex_a, lin_samp, suv + off * 1.8 + vec2<f32>(0.0, 0.004), 0.0).rgb;
    let cos_t = clamp(dot(n, v), 0.0, 1.0);
    let fres = 0.02 + 0.98 * pow(1.0 - cos_t, 5.0);
    let refl = sky_base(reflect(-v, n));
    let h = normalize(F.sun_dir.xyz + v);
    let nh = max(dot(n, h), 0.0);
    let spec = pow(nh, 160.0) * 5.0 + pow(nh, 20.0) * 0.18;
    let tint = vec3<f32>(0.78, 0.93, 0.95);
    var col = mix(behind, behind2, 0.4) * tint * (1.0 - fres) + refl * fres;
    // Aerated sheets catch the light, more as the water gathers speed.
    let aer = streak * (0.3 + 0.7 * smoothstep(1.0, 12.0, fallen));
    let white = vec3<f32>(0.62, 0.72, 0.80) + F.sun_color.rgb * 0.30;
    col = mix(col, white, aer * 0.6);
    col = col + F.sun_color.rgb * spec * (0.4 + streak);
    // Foam where it meets the pool.
    let near_pool = 1.0 - smoothstep(0.0, 2.6, y - pool_y);
    let foam = nz(vec2<f32>(across * 1.6, (y + t * 2.5) * 0.25)).b;
    col = mix(col, vec3<f32>(1.05, 1.12, 1.18) + F.sun_color.rgb * 0.15, clamp(near_pool * (0.35 + foam * 0.8), 0.0, 1.0));
    // A soft top where the water leaves the spout.
    let a = smoothstep(0.0, 0.6, fallen);
    return vec4<f32>(col * a, a);
}

fn pool_color(in: WaterOut, v: vec3<f32>, suv: vec2<f32>) -> vec4<f32> {
    let t = F.cam_pos.w;
    let centre = in.params.xy;
    let r = in.params.z;
    let dxz = in.world.xz - centre;
    let dist = length(dxz);
    let radial = dxz / max(dist, 0.01);
    let ripple = cos(dist * 6.5 - t * 6.0) * exp(-dist * 0.55);
    let q = in.world.xz * 0.30;
    let a = nz(q + vec2<f32>(t * 0.045, t * 0.027)).r;
    let b = nz(q * 2.3 - vec2<f32>(t * 0.033, -t * 0.051)).g;
    let n = normalize(vec3<f32>((a - 0.5) * 0.45 + radial.x * ripple * 0.35, 1.0, (b - 0.5) * 0.45 + radial.y * ripple * 0.35));
    let cos_t = clamp(dot(n, v), 0.0, 1.0);
    let fres = 0.02 + 0.98 * pow(1.0 - cos_t, 5.0);
    let off = vec2<f32>(dot(n.xz, F.cam_right.xz), -dot(n.xz, F.cam_fwd.xz)) * 0.04;
    let refr = textureSampleLevel(tex_a, lin_samp, suv + off, 0.0).rgb;
    var col = refr * vec3<f32>(0.50, 0.82, 0.80) * (1.0 - fres) + sky_base(reflect(-v, n)) * fres;
    let h = normalize(F.sun_dir.xyz + v);
    col = col + F.sun_color.rgb * pow(max(dot(n, h), 0.0), 260.0) * 10.0;
    let foam = smoothstep(2.2, 0.5, dist) * nz(q * 3.0 + vec2<f32>(t * 0.05, t * 0.21)).b;
    col = mix(col, vec3<f32>(1.0, 1.08, 1.12) + F.sun_color.rgb * 0.12, clamp(foam * 0.85, 0.0, 1.0));
    let edge = smoothstep(r + 0.6, r + 0.4, dist);
    return vec4<f32>(col * edge, edge);
}

@fragment
fn fs_water(in: WaterOut) -> @location(0) vec4<f32> {
    let frag = in.clip.xy;
    if (planar_depth(in.world) > scene_depth(frag) + 0.03) {
        discard;
    }
    let v = normalize(F.cam_pos.xyz - in.world);
    let suv = frag * F.screen.zw;
    if (in.kind == 0u) {
        return fall_color(in, v, suv);
    }
    return pool_color(in, v, suv);
}

@fragment
fn fs_mist(in: WaterOut) -> @location(0) vec4<f32> {
    let frag = in.clip.xy;
    let soft = clamp((scene_depth(frag) - planar_depth(in.world)) * 0.7, 0.0, 1.0);
    let uv = in.params.xy;
    let fall_off = clamp(1.0 - dot(uv, uv), 0.0, 1.0);
    let t = F.cam_pos.w;
    let n1 = nz(uv * 0.30 + vec2<f32>(in.seed * 0.13, -t * 0.035)).r;
    let n2 = nz(uv * 0.55 + vec2<f32>(t * 0.02, in.seed * 0.07)).g;
    let a = fall_off * fall_off * n1 * n2 * soft * 0.55;
    let col = (vec3<f32>(0.80, 0.88, 0.95) + F.sun_color.rgb * 0.22) * a;
    return vec4<f32>(col, 0.0);
}

// ---------------------------------------------------------------------------- bloom

fn uv_of(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * vec2<f32>(0.5, -0.5) + 0.5;
}

// One bilinear tap of the bloom source, `o` source texels (`tx` in UV) away from `uv`.
fn bloom_tap(uv: vec2<f32>, o: vec2<f32>, tx: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(tex_a, lin_samp, uv + o * tx, 0.0).rgb;
}

// The bloom chain's downsample: thirteen bilinear taps on a grid two source texels out in every
// direction, weighted into a wide, smooth low-pass, so a single bright texel cannot flicker as the
// view moves. Weights by ring (they sum to 1): the centre 1/8; the four taps one texel out on the
// diagonals 1/8 each; the four taps two texels out along the axes 1/16 each; the four corners two
// texels out on both axes 1/32 each.
fn down13(uv: vec2<f32>) -> vec3<f32> {
    let tx = 1.0 / vec2<f32>(textureDimensions(tex_a));
    var diagonal = vec3<f32>(0.0);
    var axis = vec3<f32>(0.0);
    var corner = vec3<f32>(0.0);
    // The four quadrants (+x+y, -x+y, -x-y, +x-y) written out, not looped, so every read is
    // issued before the sums need one. One axis tap per quadrant, two texels out: +x, +y, -x, -y.
    diagonal = diagonal + bloom_tap(uv, vec2<f32>(1.0, 1.0), tx);
    corner = corner + bloom_tap(uv, vec2<f32>(1.0, 1.0) * 2.0, tx);
    axis = axis + bloom_tap(uv, vec2<f32>(1.0, 0.0) * 2.0, tx);
    diagonal = diagonal + bloom_tap(uv, vec2<f32>(-1.0, 1.0), tx);
    corner = corner + bloom_tap(uv, vec2<f32>(-1.0, 1.0) * 2.0, tx);
    axis = axis + bloom_tap(uv, vec2<f32>(0.0, 1.0) * 2.0, tx);
    diagonal = diagonal + bloom_tap(uv, vec2<f32>(-1.0, -1.0), tx);
    corner = corner + bloom_tap(uv, vec2<f32>(-1.0, -1.0) * 2.0, tx);
    axis = axis + bloom_tap(uv, vec2<f32>(-1.0, 0.0) * 2.0, tx);
    diagonal = diagonal + bloom_tap(uv, vec2<f32>(1.0, -1.0), tx);
    corner = corner + bloom_tap(uv, vec2<f32>(1.0, -1.0) * 2.0, tx);
    axis = axis + bloom_tap(uv, vec2<f32>(0.0, -1.0) * 2.0, tx);
    let centre = bloom_tap(uv, vec2<f32>(0.0), tx);
    return (centre + diagonal) * 0.125 + axis * 0.0625 + corner * 0.03125;
}

@fragment
fn fs_bloom_first(in: FsOut) -> @location(0) vec4<f32> {
    var c = down13(uv_of(in.ndc));
    // Tame single-pixel sparkles (sun glints) before they become bloom blobs.
    c = c / (1.0 + luma(c) * 0.12);
    // Soft threshold: everything glows a little (the dreamy look), highlights much more.
    let l = luma(c);
    let knee = 0.6;
    let w = clamp((l - 0.45) / knee, 0.0, 1.0);
    let k = w * w * 0.7 + smoothstep(1.0, 3.0, l) * 0.6 + 0.015;
    return vec4<f32>(c * k, 1.0);
}

@fragment
fn fs_bloom_down(in: FsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(down13(uv_of(in.ndc)), 1.0);
}

@fragment
fn fs_bloom_up(in: FsOut) -> @location(0) vec4<f32> {
    let uv = uv_of(in.ndc);
    let tx = 1.0 / vec2<f32>(textureDimensions(tex_a));
    var s = textureSampleLevel(tex_a, lin_samp, uv, 0.0).rgb * 4.0;
    s = s + textureSampleLevel(tex_a, lin_samp, uv + tx * vec2<f32>(-1.0, 0.0), 0.0).rgb * 2.0;
    s = s + textureSampleLevel(tex_a, lin_samp, uv + tx * vec2<f32>(1.0, 0.0), 0.0).rgb * 2.0;
    s = s + textureSampleLevel(tex_a, lin_samp, uv + tx * vec2<f32>(0.0, -1.0), 0.0).rgb * 2.0;
    s = s + textureSampleLevel(tex_a, lin_samp, uv + tx * vec2<f32>(0.0, 1.0), 0.0).rgb * 2.0;
    s = s + textureSampleLevel(tex_a, lin_samp, uv + tx * vec2<f32>(-1.0, -1.0), 0.0).rgb;
    s = s + textureSampleLevel(tex_a, lin_samp, uv + tx * vec2<f32>(1.0, -1.0), 0.0).rgb;
    s = s + textureSampleLevel(tex_a, lin_samp, uv + tx * vec2<f32>(-1.0, 1.0), 0.0).rgb;
    s = s + textureSampleLevel(tex_a, lin_samp, uv + tx * vec2<f32>(1.0, 1.0), 0.0).rgb;
    return vec4<f32>(s / 16.0, 1.0);
}

// ---------------------------------------------------------------------------- light shafts

fn sun_visible(p: vec3<f32>) -> f32 {
    let s = F.shadow_mat * vec4<f32>(p, 1.0);
    let uv = vec2<f32>(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) {
        return 1.0;
    }
    return textureSampleCompareLevel(shadow_tex, shadow_samp, uv, s.z - 0.0005);
}

@fragment
fn fs_shafts(in: FsOut) -> @location(0) vec4<f32> {
    let d = view_ray(in.ndc);
    let uv = uv_of(in.ndc);
    let dims = vec2<f32>(textureDimensions(tex_b));
    let zp = textureLoad(tex_b, vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0)), 0).r;
    let dist = zp / max(dot(d, F.cam_fwd.xyz), 1e-3);
    let seg = min(min(dist, ray_box_exit(F.cam_pos.xyz, d)), 110.0);
    let steps = 32;
    let dt = seg / f32(steps);
    let jitter = ign(in.pos.xy + vec2<f32>(F.params.w * 7.0, F.params.w * 3.0));
    let density = F.box_min.w;
    var acc = 0.0;
    for (var i = 0; i < steps; i = i + 1) {
        let t = (f32(i) + jitter) * dt;
        acc = acc + sun_visible(F.cam_pos.xyz + d * t) * exp(-t * density);
    }
    return vec4<f32>(acc * dt, 0.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------- composite

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_composite(in: FsOut) -> @location(0) vec4<f32> {
    // The scene may be rendered smaller than the frame (see `internal_size`): sample by UV. At
    // equal sizes this lands on texel centres, so it is an exact copy.
    let uv = uv_of(in.ndc);
    var c = textureSampleLevel(tex_a, lin_samp, uv, 0.0).rgb;
    let dims = vec2<f32>(textureDimensions(tex_b));
    let zp = textureLoad(tex_b, vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0)), 0).r;
    let d = view_ray(in.ndc);
    let dist = zp / max(dot(d, F.cam_fwd.xyz), 1e-3);
    let seg = min(dist, ray_box_exit(F.cam_pos.xyz, d));
    let mu = dot(d, F.sun_dir.xyz);
    let phase = phase_hg(mu, 0.45);

    let post = 1.0 - F.look.z;
    // Humid greenhouse air: a soft, bright haze, warmer toward the sun.
    let density = F.box_min.w;
    let fog = 1.0 - exp(-seg * density);
    let haze = vec3<f32>(0.48, 0.60, 0.60) + F.sun_color.rgb * 0.08 * phase;
    c = mix(c, haze, fog * F.look.x * post);

    // Sunbeams. Faded where the view looks straight up at the roof, so the sky stays crisp.
    let shafts = textureSampleLevel(tex_d, lin_samp, uv, 0.0).r;
    let up_fade = 1.0 - 0.65 * smoothstep(0.25, 0.75, d.y);
    c = c + F.sun_color.rgb * vec3<f32>(1.0, 0.97, 0.88) * shafts * F.box_max.w * (0.3 + phase) * post * up_fade;

    // Bloom (a touch warm, like film halation), then a soft "Orton" glow on top for the dreamy
    // look.
    let bloom = textureSampleLevel(tex_c, lin_samp, uv, 0.0).rgb * vec3<f32>(1.0, 0.97, 0.9);
    c = c + bloom * F.params.y * post;

    // The opening: the scene settles out of an over-exposed, glowing bloom.
    let k = 1.0 - F.sun_dir.w;
    c = c * F.sun_color.w * (1.0 + 2.5 * k);
    var m = aces(c);
    let glow = aces(bloom * F.sun_color.w * 0.9 * (1.0 + 3.0 * k));
    m = 1.0 - (1.0 - m) * (1.0 - glow * min(F.look.y * post + k * 0.8, 1.0));

    // Grade: gentle extra saturation, cool shadows, warm highlights.
    let l = luma(m);
    m = mix(vec3<f32>(l), m, 1.08);
    m = m * mix(vec3<f32>(0.96, 1.0, 1.04), vec3<f32>(1.03, 1.0, 0.95), smoothstep(0.2, 0.9, l));
    // A soft scrim on the left, where the buttons sit, deepest behind the title (left aligned,
    // an eighth of the way down) so white type keeps its contrast over the bright structure.
    let side = (1.0 - smoothstep(0.0, 0.5, uv.x)) * (0.7 + 0.3 * (1.0 - uv.y));
    let tz = (uv - vec2<f32>(0.2, 0.125)) / vec2<f32>(0.3, 0.12);
    let title = exp(-dot(tz, tz)) * 1.1;
    m = m * (1.0 - F.look.w * min(side + title, 1.8));
    // Vignette.
    let q = uv - 0.5;
    m = m * (1.0 - dot(q * vec2<f32>(0.9, 1.1), q * vec2<f32>(0.9, 1.1)) * 0.42);
    m = mix(m, vec3<f32>(0.97, 0.98, 0.96), k * k * 0.6);

    var out = linear_to_srgb(clamp(m, vec3<f32>(0.0), vec3<f32>(1.0)));
    out = out + (ign(in.pos.xy) - 0.5) / 255.0;
    return vec4<f32>(out, 1.0);
}

// ---------------------------------------------------------------------------- VHS finish

fn to_yiq(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.299, 0.587, 0.114)),
        dot(c, vec3<f32>(0.596, -0.274, -0.322)),
        dot(c, vec3<f32>(0.211, -0.523, 0.312)),
    );
}

fn from_yiq(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        c.x + 0.956 * c.y + 0.621 * c.z,
        c.x - 0.272 * c.y - 0.647 * c.z,
        c.x - 1.106 * c.y + 1.703 * c.z,
    );
}

// An integer mix: one LCG step, then a permutation of its bits (a shift chosen by its own top
// bits, a multiply, a final xor-shift).
fn mix32(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// The film grain's noise: a value in [0, 1) for each whole-pixel position.
fn grain_hash(p: vec2<f32>) -> f32 {
    let c = bitcast<vec2<u32>>(vec2<i32>(floor(p)));
    return f32(mix32(c.x ^ mix32(c.y)) >> 8u) * (1.0 / 16777216.0);
}

// Over the composed frame, UI included (display-encoded values): a light horizontal luma blur,
// wider chroma smear with a small delay, a faint tape wobble and tracking band, scanlines and
// grain. Mixed back toward the sharp frame so text stays readable.
@fragment
fn fs_finish(in: FsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(tex_a));
    let px = in.pos.xy;
    let t = F.cam_pos.w;
    let s = F.params.z;
    let k = max(dims.y / 1080.0, 0.5);
    let wob = (sin(px.y * 0.011 / k + t * 1.6) * 0.6 + sin(px.y * 0.067 / k - t * 2.3) * 0.3) * 0.45 * k * s;
    let band_y = fract(t * 0.031) * (dims.y + 600.0) - 300.0;
    let band = exp(-pow((px.y - band_y) / (22.0 * k), 2.0)) * s;
    let x = px.x + wob + band * 2.5 * k;
    var luma_acc = 0.0;
    var chroma = vec2<f32>(0.0);
    var wl = array<f32, 7>(0.03, 0.12, 0.22, 0.26, 0.22, 0.12, 0.03);
    for (var i = 0; i < 7; i = i + 1) {
        let o = f32(i - 3);
        let yl = to_yiq(textureSampleLevel(tex_a, lin_samp, vec2<f32>(x + o * 0.75 * k, px.y) / dims, 0.0).rgb).x;
        luma_acc = luma_acc + yl * wl[i];
        let cc = to_yiq(textureSampleLevel(tex_a, lin_samp, vec2<f32>(x + (o * 1.9 - 1.6) * k, px.y) / dims, 0.0).rgb).yz;
        chroma = chroma + cc / 7.0;
    }
    var vhs = from_yiq(vec3<f32>(luma_acc, chroma * 1.04));
    let sharp = textureLoad(tex_a, vec2<i32>(clamp(px, vec2<f32>(0.0), dims - 1.0)), 0).rgb;
    var c = mix(sharp, vhs, 0.82 * s);
    c = c + vec3<f32>(band * 0.05);
    let scan = 1.0 - 0.03 * s * (0.5 + 0.5 * cos(px.y * PI / k));
    c = c * scan;
    c = c + (grain_hash(px + vec2<f32>(fract(t * 13.0) * 517.0, fract(t * 7.0) * 211.0)) - 0.5) * 0.028 * s;
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
