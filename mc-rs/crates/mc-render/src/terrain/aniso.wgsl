// Anisotropic filtering for the block atlas.
//
// Appended to `terrain.wgsl` to build a SECOND shader module, used only while the Anisotropic
// Filtering mod is on. The off path compiles `terrain.wgsl` alone, exactly as before; the vertex
// stage here is that file's `vs`, so positions and depth are bit-identical either way.
//
// Why not a hardware anisotropic sampler: wgpu requires linear min/mag/mip filters for
// `anisotropy_clamp > 1`, the hardware's taps cannot be kept inside one atlas sprite (every
// block edge at a grazing angle would bleed the neighbouring sprite in), and Minecraft's look
// needs nearest-neighbour magnification. So the taps are taken here, explicitly:
//
//   * magnified or 1:1 (the larger screen-space derivative of the texel coordinate <= 1 texel per
//     pixel): today's sampler at level 0 -> exactly today's texel, pixel-sharp;
//   * minified: EXT_texture_filter_anisotropic's footprint maths. N = min(ceil(Pmax / Pmin),
//     max anisotropy) taps spread along the major axis, LOD = log2(Pmax / N), each tap clamped
//     to the 2^L-aligned atlas cell the fragment's texel lies in. The stitcher reserves every
//     sprite on multiples of 2^L (L = atlas mip levels), so a cell never holds two sprites and no
//     tap can read a neighbour; the inset is half a texel of the coarser mip a trilinear tap
//     reads. Taps are averaged with premultiplied alpha.
//
// Solid uses linear taps (the classic AF look); cutout-mipped (leaves) uses today's
// nearest-within-level sampler per tap, so alpha-tested edges keep vanilla's colours instead of
// being bilinearly mixed with transparent texels. Translucent and the non-mipped cutout layer
// keep today's pipelines (see `aniso.rs`).

struct AnisoUniform {
    // x = atlas width, y = atlas height (texels), z = highest mip level, w = max anisotropy.
    atlas: vec4<f32>,
    // x = cell size in level-0 texels (2^L), yzw = reserved.
    cell: vec4<f32>,
};
@group(2) @binding(0) var aniso_linear: sampler;
@group(2) @binding(1) var<uniform> aniso: AnisoUniform;

// `nearest_taps`: sample each tap with the layer's own sampler (`atlas_samp`) instead of the
// linear one. `premultiply`: weight colours by alpha (layers that test or blend alpha); the solid
// layer ignores alpha like vanilla, so its taps average plainly. Derivatives first, in uniform
// control flow; everything after uses explicit LODs.
fn sample_aniso(uv: vec2<f32>, nearest_taps: bool, premultiply: bool) -> vec4<f32> {
    let size = aniso.atlas.xy;
    let t = uv * size;
    let dx = dpdx(t);
    let dy = dpdy(t);
    let lx2 = dot(dx, dx);
    let ly2 = dot(dy, dy);
    let major2 = max(lx2, ly2);
    if (major2 <= 1.0) {
        return textureSampleLevel(atlas_tex, atlas_samp, uv, 0.0);
    }
    let major_len = sqrt(major2);
    let minor_len = sqrt(min(lx2, ly2));
    let axis = select(dy, dx, lx2 >= ly2);
    let n = clamp(ceil(major_len / max(minor_len, 1e-6)), 1.0, aniso.atlas.w);
    let top = aniso.atlas.z;
    let lod = clamp(log2(major_len / n), 0.0, top);
    let cell = aniso.cell.x;
    let cell_min = floor(t / cell) * cell;
    let inset = 0.5 * exp2(min(floor(lod) + 1.0, top));
    let lo = cell_min + vec2<f32>(inset);
    let hi = cell_min + vec2<f32>(cell - inset);
    var plain = vec3<f32>(0.0);
    var weighted = vec3<f32>(0.0);
    var alpha = 0.0;
    let count = u32(n);
    for (var i = 0u; i < count; i++) {
        let f = (f32(i) + 0.5) / n - 0.5;
        let p = clamp(t + axis * f, lo, hi) / size;
        var s: vec4<f32>;
        if (nearest_taps) {
            s = textureSampleLevel(atlas_tex, atlas_samp, p, lod);
        } else {
            s = textureSampleLevel(atlas_tex, aniso_linear, p, lod);
        }
        plain += s.rgb;
        weighted += s.rgb * s.a;
        alpha += s.a;
    }
    if (!premultiply || alpha <= 1e-4) {
        return vec4<f32>(plain / n, alpha / n);
    }
    return vec4<f32>(weighted / alpha, alpha / n);
}

// SOLID: no alpha test (java: `disableAlpha` for the solid layer), linear taps.
@fragment
fn fs_solid_aniso(in: VsOut) -> @location(0) vec4<f32> {
    return shade(in, sample_aniso(in.uv, false, false));
}

// CUTOUT_MIPPED: java's 0.5 alpha reference, nearest-within-level taps.
@fragment
fn fs_cutout_aniso(in: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_aniso(in.uv, true, true);
    if (texel.a <= terrain.params.x) {
        discard;
    }
    return shade(in, texel);
}
