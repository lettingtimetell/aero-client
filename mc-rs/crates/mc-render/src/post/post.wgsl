// Native wgpu translations of vanilla's `assets/minecraft/shaders/program/*.fsh` — S20.
//
// One shared fullscreen-triangle vertex stage and one fragment entry point per translated
// program. Each entry point is a line-for-line translation of the GLSL of the same name out of
// `1.8.9.jar`; the GLSL's `texCoord`/`oneTexel` varyings become `in.uv` and
// `1.0 / post.sizes.xy`, and its `uniform` declarations become fields of `PostUniform` (whose
// values `post/pipeline.rs` fills from the program JSON's declared defaults plus the pass's
// `uniforms` overrides — see `PROGRAM_DEFAULTS` there).
//
// # Deliberate scope limits, so nothing here reads as more finished than it is
//
// * Only the **fragment** stages are translated. Vanilla ships four distinct vertex programs
//   (`blit.vsh`, `sobel.vsh`, `blobs.vsh`, `downscale.vsh`, `rotscale.vsh`, `flip.vsh`) that
//   differ in how they derive `texCoord`; every entry point below assumes `blit.vsh`'s
//   convention (`(0,0)` = top-left of the input). `flip` is the one exception, because flipping
//   is its whole purpose and it lives entirely in `flip.vsh`. `blobs.vsh`'s missing y-flip and
//   `rotscale.vsh`'s rotation/scale are **not** reproduced.
// * `fxaa`, `ntsc_encode`, `ntsc_decode`, `outline_watercolor` and `spider` are translated below;
//   `fxaa` and `spider` receive dedicated vertex entry points from `pipeline.rs` for their extra
//   varyings. Custom resource-pack GLSL remains outside this baked WGSL module.

@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_samp: sampler;
@group(0) @binding(3) var aux_tex: texture_2d<f32>;
@group(0) @binding(4) var aux_samp: sampler;

struct PostUniform {
    // xy = InSize (the input target's pixel size, so oneTexel = 1/InSize), zw = OutSize.
    sizes: vec4<f32>,
    // blit's ColorModulate.
    color_modulate: vec4<f32>,
    // x = InverseAmount, y = Radius, z = Resolution, w = MosaicSize.
    scalars: vec4<f32>,
    // x = Saturation, y = LumaRamp, z = LumaLevel, w = Time.
    scalars2: vec4<f32>,
    // xy = BlurDir, zw = Frequency.
    blur_dir: vec4<f32>,
    // xy = WobbleAmount, or spider InScale; zw = flip screen size, or spider InOffset.
    wobble: vec4<f32>,
    // rgb = Gray; w = fxaa SpanMax.
    gray: vec4<f32>,
    // rgb = RedMatrix; w = fxaa SubPixelShift.
    red_matrix: vec4<f32>,
    // rgb = GreenMatrix; w = fxaa ReduceMul.
    green_matrix: vec4<f32>,
    blue_matrix: vec4<f32>,
    offset: vec4<f32>,
    color_scale: vec4<f32>,
    // deconverge ConvergeX, or spider Scissor.
    converge_x: vec4<f32>,
    // deconverge ConvergeY, or spider Vignette.
    converge_y: vec4<f32>,
    // deconverge RadialConvergeX; w = spider InRotation.
    radial_x: vec4<f32>,
    radial_y: vec4<f32>,
    phosphor: vec4<f32>,
};
@group(0) @binding(2) var<uniform> post: PostUniform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FxaaIn {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) pos_pos: vec4<f32>,
};

struct SpiderIn {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) scaled_coord: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    var out: VsOut;
    let p = corners[index];
    out.clip = vec4<f32>(p, 0.0, 1.0);
    // java: blit.vsh — `texCoord = Position.xy / OutSize; texCoord.y = 1.0 - texCoord.y;`, i.e.
    // (0,0) is the top-left texel.
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return out;
}

fn one_texel() -> vec2<f32> {
    return vec2<f32>(1.0) / max(post.sizes.xy, vec2<f32>(1.0));
}

fn tex(uv: vec2<f32>) -> vec4<f32> {
    return textureSample(in_tex, in_samp, uv);
}

// ---------------------------------------------------------------- blit / invert / flip

// java: program/blit.fsh
@fragment
fn blit(in: VsOut) -> @location(0) vec4<f32> {
    return tex(in.uv) * post.color_modulate;
}

// java: program/invert.fsh — note the GLSL inverts *all four* channels then forces alpha to 1.
@fragment
fn invert(in: VsOut) -> @location(0) vec4<f32> {
    let diffuse = tex(in.uv);
    let inverted = vec4<f32>(1.0) - diffuse;
    let out_color = mix(diffuse, inverted, post.scalars.x);
    return vec4<f32>(out_color.rgb, 1.0);
}

// java: program/flip.vsh + blit.fsh — the flip lives in the vertex stage, which at
// InSize == OutSize == ScreenSize reduces to dropping blit.vsh's `1.0 - y`.
@fragment
fn flip(in: VsOut) -> @location(0) vec4<f32> {
    return tex(vec2<f32>(in.uv.x, 1.0 - in.uv.y)) * post.color_modulate;
}

// ---------------------------------------------------------------- colour transforms

// java: program/color_convolve.fsh — a 3x3 colour matrix, then scale+offset, then a saturation
// push around the Gray-weighted luma. This is the program `desaturate.json`, `green.json`,
// `creeper.json`, `ntsc.json` and `color_convolve.json` all build their look out of, purely
// through pass-level uniform overrides.
@fragment
fn color_convolve(in: VsOut) -> @location(0) vec4<f32> {
    let in_texel = tex(in.uv);
    let red_value = dot(in_texel.rgb, post.red_matrix.rgb);
    let green_value = dot(in_texel.rgb, post.green_matrix.rgb);
    let blue_value = dot(in_texel.rgb, post.blue_matrix.rgb);
    var out_color = vec3<f32>(red_value, green_value, blue_value);
    out_color = (out_color * post.color_scale.rgb) + post.offset.rgb;
    let luma = dot(out_color, post.gray.rgb);
    let chroma = out_color - luma;
    out_color = (chroma * post.scalars2.x) + luma;
    return vec4<f32>(out_color, 1.0);
}

// java: program/bits.fsh — mosaic, quantise to `Resolution` levels, then push saturation.
@fragment
fn bits(in: VsOut) -> @location(0) vec4<f32> {
    let mosaic_in_size = post.sizes.xy / max(post.scalars.w, 1.0);
    let fract_pix = fract(in.uv * mosaic_in_size) / mosaic_in_size;

    var base_texel = tex(in.uv - fract_pix);
    let resolution = max(post.scalars.z, 1.0);
    let fract_texel = base_texel.rgb - fract(base_texel.rgb * resolution) / resolution;
    let luma = dot(fract_texel, vec3<f32>(0.3, 0.59, 0.11));
    let chroma = (fract_texel - luma) * post.scalars2.x;
    return vec4<f32>(luma + chroma, 1.0);
}

// java: program/deconverge.fsh — samples r/g/b at three separately offset coordinates, the CRT
// "colour fringing" look.
@fragment
fn deconverge(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    var coord_x = in.uv.x * post.radial_x.rgb;
    var coord_y = in.uv.y * post.radial_y.rgb;
    coord_x = coord_x + post.converge_x.rgb * one.x - (post.radial_x.rgb - 1.0) * 0.5;
    coord_y = coord_y + post.converge_y.rgb * one.y - (post.radial_y.rgb - 1.0) * 0.5;

    let r = tex(vec2<f32>(coord_x.r, coord_y.r)).r;
    let g = tex(vec2<f32>(coord_x.g, coord_y.g)).g;
    let b = tex(vec2<f32>(coord_x.b, coord_y.b)).b;
    return vec4<f32>(r, g, b, 1.0);
}

// java: program/wobble.fsh — a sinusoidal UV wobble plus a full hue rotation driven by `Time`.
fn wobble_hue(h: f32) -> vec3<f32> {
    let r = abs(h * 6.0 - 3.0) - 1.0;
    let g = 2.0 - abs(h * 6.0 - 2.0);
    let b = 2.0 - abs(h * 6.0 - 4.0);
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    return ((wobble_hue(hsv.x) - 1.0) * hsv.y + 1.0) * hsv.z;
}

fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    var hsv = vec3<f32>(0.0);
    hsv.z = max(rgb.r, max(rgb.g, rgb.b));
    let lo = min(rgb.r, min(rgb.g, rgb.b));
    let c = hsv.z - lo;
    if (c != 0.0) {
        hsv.y = c / hsv.z;
        var delta = (hsv.z - rgb) / c;
        delta = delta - delta.brg;
        delta = vec3<f32>(delta.r + 2.0, delta.g + 4.0, delta.b);
        if (rgb.r >= hsv.z) {
            hsv.x = delta.b;
        } else if (rgb.g >= hsv.z) {
            hsv.x = delta.r;
        } else {
            hsv.x = delta.g;
        }
        hsv.x = fract(hsv.x / 6.0);
    }
    return hsv;
}

@fragment
fn wobble(in: VsOut) -> @location(0) vec4<f32> {
    let time = post.scalars2.w;
    let x_offset = sin(in.uv.y * post.blur_dir.z + time * 3.1415926535 * 2.0) * post.wobble.x;
    let y_offset = cos(in.uv.x * post.blur_dir.w + time * 3.1415926535 * 2.0) * post.wobble.y;
    let rgb = tex(in.uv + vec2<f32>(x_offset, y_offset));
    var hsv = rgb_to_hsv(rgb.rgb);
    hsv.x = fract(hsv.x + time);
    return vec4<f32>(hsv_to_rgb(hsv), 1.0);
}

// ---------------------------------------------------------------- edge detection

// java: program/sobel.fsh
@fragment
fn sobel(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let center = tex(in.uv);
    let left = tex(in.uv - vec2<f32>(one.x, 0.0));
    let right = tex(in.uv + vec2<f32>(one.x, 0.0));
    let up = tex(in.uv - vec2<f32>(0.0, one.y));
    let down = tex(in.uv + vec2<f32>(0.0, one.y));
    let total = clamp(
        (center - left) + (center - right) + (center - up) + (center - down),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
    return vec4<f32>(total.rgb, 1.0);
}

// java: program/entity_sobel.fsh (the `entity_outline` program) — the alpha-keyed variant used
// for the spectator/glowing entity outline, which keeps `total` in the alpha channel.
@fragment
fn entity_outline(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let center = tex(in.uv);
    let left = tex(in.uv - vec2<f32>(one.x, 0.0));
    let right = tex(in.uv + vec2<f32>(one.x, 0.0));
    let up = tex(in.uv - vec2<f32>(0.0, one.y));
    let down = tex(in.uv + vec2<f32>(0.0, one.y));
    let total = clamp(
        abs(center.a - left.a) + abs(center.a - right.a)
            + abs(center.a - up.a) + abs(center.a - down.a),
        0.0,
        1.0,
    );
    let out_color = center.rgb * center.a + left.rgb * left.a + right.rgb * right.a
        + up.rgb * up.a + down.rgb * down.a;
    return vec4<f32>(out_color * 0.2, total);
}

// java: program/outline.fsh — note the GLSL's `up`/`down`/`left`/`right` names do not match their
// offsets (down samples +x, right samples +y); transcribed as written, not as named.
@fragment
fn outline(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let center = tex(in.uv);
    let up = tex(in.uv + vec2<f32>(0.0, -one.y));
    let down = tex(in.uv + vec2<f32>(one.x, 0.0));
    let left = tex(in.uv + vec2<f32>(-one.x, 0.0));
    let right = tex(in.uv + vec2<f32>(0.0, one.y));
    let sum = (center - up) + (center - down) + (center - left) + (center - right);
    let clamped = clamp(center.rgb - sum.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(clamped, 1.0);
}

// java: program/outline_soft.fsh — the `pencil.json` look: an eight-tap edge sum modulating a
// quantised luma.
@fragment
fn outline_soft(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let center = tex(in.uv);
    let up = tex(in.uv + vec2<f32>(0.0, -one.y));
    let up2 = tex(in.uv + vec2<f32>(0.0, -one.y) * 2.0);
    let down = tex(in.uv + vec2<f32>(one.x, 0.0));
    let down2 = tex(in.uv + vec2<f32>(one.x, 0.0) * 2.0);
    let left = tex(in.uv + vec2<f32>(-one.x, 0.0));
    let left2 = tex(in.uv + vec2<f32>(-one.x, 0.0) * 2.0);
    let right = tex(in.uv + vec2<f32>(0.0, one.y));
    let right2 = tex(in.uv + vec2<f32>(0.0, one.y) * 2.0);
    let sum = abs(center - up) + abs(center - down) + abs(center - left) + abs(center - right)
        + abs(center - up2) + abs(center - down2) + abs(center - left2) + abs(center - right2);
    let gray = vec4<f32>(0.3, 0.59, 0.11, 0.0);
    let sum_luma = 1.0 - dot(clamp(sum, vec4<f32>(0.0), vec4<f32>(1.0)), gray);

    let ramp = post.scalars2.y;
    let level = max(post.scalars2.z, 2.0);
    var center_luma = dot(center + (center - pow(max(center, vec4<f32>(0.0)), vec4<f32>(ramp))), gray);
    center_luma = center_luma - fract(center_luma * level) / level;
    center_luma = center_luma * (level / (level - 1.0));
    center_luma = center_luma * sum_luma;
    return vec4<f32>(center_luma, center_luma, center_luma, 1.0);
}

// java: program/outline_combine.fsh — `OutlineSampler` is the pass's first auxtarget.
@fragment
fn outline_combine(in: VsOut) -> @location(0) vec4<f32> {
    let diffuse = tex(in.uv);
    let outline_texel = textureSample(aux_tex, aux_samp, in.uv);
    return vec4<f32>(diffuse.rgb + diffuse.rgb * outline_texel.rgb * vec3<f32>(0.75), 1.0);
}

// ---------------------------------------------------------------- filters

// java: program/blur.fsh — a separable box/tent blur along `BlurDir`. The GLSL's loop steps by
// 1.0 from -Radius to +Radius; an integer loop is exact whenever Radius is a whole number, which
// it is in every shipped `post/*.json`.
@fragment
fn blur(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let radius = max(post.scalars.y, 0.0);
    let n = i32(radius);
    var blurred = vec4<f32>(0.0);
    var total_alpha = 0.0;
    for (var i = -n; i <= n; i = i + 1) {
        let s = tex(in.uv + one * f32(i) * post.blur_dir.xy);
        total_alpha = total_alpha + s.a;
        blurred = blurred + s;
    }
    return vec4<f32>(blurred.rgb / (radius * 2.0 + 1.0), total_alpha);
}

// java: program/antialias.fsh — a fixed five-tap-per-axis smoothing kernel.
@fragment
fn antialias(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let c = tex(in.uv);
    let u1 = tex(in.uv + vec2<f32>(0.0, -one.y));
    let u2 = tex(in.uv + vec2<f32>(0.0, -one.y * 2.0));
    let d1 = tex(in.uv + vec2<f32>(0.0, one.y));
    let d2 = tex(in.uv + vec2<f32>(0.0, one.y * 2.0));
    let l1 = tex(in.uv + vec2<f32>(-one.x, 0.0));
    let l2 = tex(in.uv + vec2<f32>(-one.x * 2.0, 0.0));
    let r1 = tex(in.uv + vec2<f32>(one.x, 0.0));
    let r2 = tex(in.uv + vec2<f32>(one.x * 2.0, 0.0));

    let v1 = mix(c, mix(l1, l2, 0.667), 0.75);
    let v2 = mix(c, mix(r1, r2, 0.667), 0.75);
    let v3 = mix(c, mix(u1, u2, 0.667), 0.75);
    let v4 = mix(c, mix(d1, d2, 0.667), 0.75);
    let color = mix(mix(v1, v2, 0.5), mix(v3, v4, 0.5), 0.5);
    return vec4<f32>(color.rgb, 1.0);
}

// java: program/blobs.fsh — a fixed-radius erosion (min filter) over 13 taps.
@fragment
fn blobs(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let u = tex(in.uv + vec2<f32>(0.0, -one.y));
    let d = tex(in.uv + vec2<f32>(0.0, one.y));
    let l = tex(in.uv + vec2<f32>(-one.x, 0.0));
    let r = tex(in.uv + vec2<f32>(one.x, 0.0));
    let v3 = min(min(l, r), min(u, d));

    let ul = tex(in.uv + vec2<f32>(-one.x, -one.y));
    let dr = tex(in.uv + vec2<f32>(one.x, one.y));
    let dl = tex(in.uv + vec2<f32>(-one.x, one.y));
    let ur = tex(in.uv + vec2<f32>(one.x, -one.y));
    let v6 = min(min(ul, dr), min(ur, dl));

    let uu = tex(in.uv + vec2<f32>(0.0, -one.y * 2.0));
    let dd = tex(in.uv + vec2<f32>(0.0, one.y * 2.0));
    let ll = tex(in.uv + vec2<f32>(-one.x * 2.0, 0.0));
    let rr = tex(in.uv + vec2<f32>(one.x * 2.0, 0.0));
    let v10 = min(min(uu, dd), min(ll, rr));

    let color = min(tex(in.uv), min(min(v3, v6), v10));
    return vec4<f32>(color.rgb, 1.0);
}

// java: program/blobs2.fsh — a circular dilation (max filter) of radius `Radius`.
@fragment
fn blobs2(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let radius = max(post.scalars.y, 0.0);
    let n = i32(radius);
    var max_val = tex(in.uv);
    for (var u = 0; u <= n; u = u + 1) {
        for (var v = 0; v <= n; v = v + 1) {
            let fu = f32(u);
            let fv = f32(v);
            var weight = 1.0;
            if (radius <= 0.0 || sqrt(fu * fu + fv * fv) / radius > 1.0) {
                weight = 0.0;
            }
            let s0 = tex(in.uv + vec2<f32>(-fu * one.x, -fv * one.y));
            let s1 = tex(in.uv + vec2<f32>(fu * one.x, fv * one.y));
            let s2 = tex(in.uv + vec2<f32>(-fu * one.x, fv * one.y));
            let s3 = tex(in.uv + vec2<f32>(fu * one.x, -fv * one.y));
            let temp_max = max(max(s0, s1), max(s2, s3));
            max_val = mix(max_val, max(max_val, temp_max), weight);
        }
    }
    return vec4<f32>(max_val.rgb, 1.0);
}

// java: program/bumpy.fsh — a fake emboss from the dot products of neighbouring normalised
// texels.
@fragment
fn bumpy(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let c = tex(in.uv);
    let u = tex(in.uv + vec2<f32>(0.0, -one.y));
    let d = tex(in.uv + vec2<f32>(0.0, one.y));
    let l = tex(in.uv + vec2<f32>(-one.x, 0.0));
    let r = tex(in.uv + vec2<f32>(one.x, 0.0));

    // `normalize` of a zero vector is undefined in both GLSL and WGSL; guard it so a black
    // frame produces black rather than NaN.
    let nc = safe_normalize(c);
    let du = dot(nc, safe_normalize(u));
    let dd = dot(nc, safe_normalize(d));
    let dl = dot(nc, safe_normalize(l));
    let dr = dot(nc, safe_normalize(r));

    let i = 64.0;
    var f = 1.0;
    f = f + (du * i) - (dd * i);
    f = f + (dr * i) - (dl * i);
    let color = c * clamp(f, 0.5, 2.0);
    return vec4<f32>(color.rgb, 1.0);
}

fn safe_normalize(v: vec4<f32>) -> vec4<f32> {
    let len = length(v);
    if (len <= 0.0) {
        return vec4<f32>(0.0);
    }
    return v / len;
}

// java: program/notch.fsh — `DitherSampler` is the pass's first auxtarget (a real `dither.png`
// in the jar, which `PostChain` binds as an aux texture when the pass declares it).
@fragment
fn notch(in: VsOut) -> @location(0) vec4<f32> {
    let half_size = post.sizes.xy * 0.5;
    var stepped = in.uv;
    stepped.x = f32(i32(stepped.x * half_size.x)) / max(half_size.x, 1.0);
    stepped.y = f32(i32(stepped.y * half_size.y)) / max(half_size.y, 1.0);

    let noise = textureSample(aux_tex, aux_samp, stepped * half_size / 4.0);
    let col = tex(stepped) + noise * vec4<f32>(1.0 / 12.0, 1.0 / 12.0, 1.0 / 6.0, 1.0);
    let r = f32(i32(col.r * 8.0)) / 8.0;
    let g = f32(i32(col.g * 8.0)) / 8.0;
    let b = f32(i32(col.b * 4.0)) / 4.0;
    return vec4<f32>(r, g, b, 1.0);
}

// java: program/phosphor.fsh — `PrevSampler` (the pass's first auxtarget) decayed by `Phosphor`
// and maxed against the current frame, the CRT persistence trail.
@fragment
fn phosphor(in: VsOut) -> @location(0) vec4<f32> {
    let curr = tex(in.uv);
    let prev = textureSample(aux_tex, aux_samp, in.uv);
    return vec4<f32>(max(prev.rgb * post.phosphor.rgb, curr.rgb), 1.0);
}

// java: program/scan_pincushion.fsh — CRT pincushion warp plus scanlines. Every tunable in the
// GLSL is a `const`, not a uniform, so they are inlined here exactly as written.
@fragment
fn scan_pincushion(in: VsOut) -> @location(0) vec4<f32> {
    let pincushion_amount = 0.02;
    let scanline_amount = 0.8;
    let scanline_scale = 1.0;
    let scanline_height = 1.0;
    let scanline_bright_scale = 1.0;
    let scanline_offset = 0.0;
    let floor_color = vec3<f32>(0.05, 0.05, 0.05);
    let power = vec3<f32>(0.8, 0.8, 0.8);
    let pi = 3.1415926535;

    let in_texel = tex(in.uv);
    let pin_unit = in.uv * 2.0 - 1.0;
    let pin_r2 = pow(length(pin_unit), 2.0);
    let pin_curve = pin_unit * pincushion_amount * pin_r2;

    var scan_coord = in.uv;
    scan_coord = scan_coord * (1.0 - pincushion_amount * 0.2);
    scan_coord = scan_coord + pincushion_amount * 0.1;
    scan_coord = scan_coord + pin_curve;

    if (scan_coord.x < 0.0 || scan_coord.y < 0.0 || scan_coord.x > 1.0 || scan_coord.y > 1.0) {
        discard;
    }

    let inner_sine = scan_coord.y * post.sizes.y * scanline_scale * 0.25;
    let scan_bright_mod = sin(inner_sine * pi + scanline_offset * post.sizes.y * 0.25);
    let scan_brightness = mix(
        1.0,
        (pow(scan_bright_mod * scan_bright_mod, scanline_height) * scanline_bright_scale + 1.0) * 0.5,
        scanline_amount,
    );
    var texel = in_texel.rgb * scan_brightness;
    texel = floor_color + (vec3<f32>(1.0) - floor_color) * texel;
    texel = pow(max(texel, vec3<f32>(0.0)), power);
    return vec4<f32>(texel, 1.0);
}

// ---------------------------------------------------------------- remaining 1.8.9 programs

// java: program/fxaa.vsh + fxaa.fsh. `pos_pos` is carried by the dedicated vertex entry point;
// SpanMax, SubPixelShift and ReduceMul occupy otherwise-unused fourth components of the packed
// post uniform block. The integer GLSL texture offsets are expressed as normalized texel steps;
// the post sampler is nearest/clamp-to-edge just like vanilla's framebuffer sampler.
@fragment
fn fxaa(in: FxaaIn) -> @location(0) vec4<f32> {
    let rcp_frame = vec2<f32>(1.0) / max(post.sizes.zw, vec2<f32>(1.0));
    let rgb_nw = textureSample(in_tex, in_samp, in.pos_pos.zw).rgb;
    // GLSL textureOffset offsets in input-texture texels; only the direction term uses the
    // shader's explicit `1.0 / OutSize` frame reciprocal.
    let rgb_ne = textureSample(in_tex, in_samp, in.pos_pos.zw, vec2<i32>(1, 0)).rgb;
    let rgb_sw = textureSample(in_tex, in_samp, in.pos_pos.zw, vec2<i32>(0, 1)).rgb;
    let rgb_se = textureSample(in_tex, in_samp, in.pos_pos.zw, vec2<i32>(1, 1)).rgb;
    let rgb_m = textureSample(in_tex, in_samp, in.pos_pos.xy).rgb;

    let luma_weights = vec3<f32>(0.299, 0.587, 0.114);
    let luma_nw = dot(rgb_nw, luma_weights);
    let luma_ne = dot(rgb_ne, luma_weights);
    let luma_sw = dot(rgb_sw, luma_weights);
    let luma_se = dot(rgb_se, luma_weights);
    let luma_m = dot(rgb_m, luma_weights);
    let luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    let luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    var direction = vec2<f32>(
        -((luma_nw + luma_ne) - (luma_sw + luma_se)),
        (luma_nw + luma_sw) - (luma_ne + luma_se),
    );
    let direction_reduce = max(
        (luma_nw + luma_ne + luma_sw + luma_se) * (0.25 * post.green_matrix.w),
        1.0 / 128.0,
    );
    let reciprocal_direction_min = 1.0
        / (min(abs(direction.x), abs(direction.y)) + direction_reduce);
    direction = min(
        vec2<f32>(post.gray.w),
        max(vec2<f32>(-post.gray.w), direction * reciprocal_direction_min),
    ) * rcp_frame;

    let rgb_a = 0.5 * (
        textureSample(in_tex, in_samp, in.pos_pos.xy + direction * vec2<f32>(1.0 / 3.0 - 0.5)).rgb
            + textureSample(in_tex, in_samp, in.pos_pos.xy + direction * vec2<f32>(2.0 / 3.0 - 0.5)).rgb
    );
    let rgb_b = rgb_a * 0.5 + 0.25 * (
        textureSample(in_tex, in_samp, in.pos_pos.xy + direction * vec2<f32>(0.0 / 3.0 - 0.5)).rgb
            + textureSample(in_tex, in_samp, in.pos_pos.xy + direction * vec2<f32>(3.0 / 3.0 - 0.5)).rgb
    );
    let luma_b = dot(rgb_b, luma_weights);
    let output_rgb = select(rgb_b, rgb_a, luma_b < luma_min || luma_b > luma_max);
    return vec4<f32>(output_rgb, 1.0);
}

// java: program/ntsc_encode.fsh. The source's `sobel.vsh` supplies oneTexel and texCoord;
// deriving those from the shared top-left UV is algebraically identical for the fullscreen pass.
@fragment
fn ntsc_encode(in: VsOut) -> @location(0) vec4<f32> {
    let one = vec2<f32>(1.0) / max(post.sizes.xy, vec2<f32>(1.0));
    let inverse_p = one;
    let c0 = in.uv;
    let c1 = in.uv + inverse_p * 0.25;
    let c2 = in.uv + inverse_p * 0.50;
    let c3 = in.uv + inverse_p * 0.75;
    let texel0 = tex(c0);
    let texel1 = tex(c1);
    let texel2 = tex(c2);
    let texel3 = tex(c3);

    let pi2 = 6.283185307;
    let scan_time = 52.6;
    let cc_frequency = 3.59754545;
    let t = vec4<f32>(in.uv.y) * post.sizes.y + vec4<f32>(0.5) + vec4<f32>(
        c0.x,
        c1.x,
        c2.x,
        c3.x,
    );
    let tw = t * vec4<f32>(pi2 * scan_time * cc_frequency);
    let y_transform = vec4<f32>(0.299, 0.587, 0.114, 0.0);
    let i_transform = vec4<f32>(0.595716, -0.274453, -0.321263, 0.0);
    let q_transform = vec4<f32>(0.211456, -0.522591, 0.31135, 0.0);
    let y = vec4<f32>(
        dot(texel0, y_transform),
        dot(texel1, y_transform),
        dot(texel2, y_transform),
        dot(texel3, y_transform),
    );
    let i = vec4<f32>(
        dot(texel0, i_transform),
        dot(texel1, i_transform),
        dot(texel2, i_transform),
        dot(texel3, i_transform),
    );
    let q = vec4<f32>(
        dot(texel0, q_transform),
        dot(texel1, q_transform),
        dot(texel2, q_transform),
        dot(texel3, q_transform),
    );
    let encoded = y + i * cos(tw) + q * sin(tw);
    return (encoded - vec4<f32>(-1.1183)) * vec4<f32>(1.0 / 3.2366);
}

fn sinc4(value: vec4<f32>) -> vec4<f32> {
    // GLSL's source explicitly replaces each zero numerator/denominator with one. `select`
    // preserves that Java/GLSL branch result while keeping the vectorized WGSL translation.
    return select(sin(value) / value, vec4<f32>(1.0), value == vec4<f32>(0.0));
}

// java: program/ntsc_decode.fsh. The 83-sample loop, four-lane accumulation and zero-safe sinc
// terms are kept in the same order as the downlevel GLSL source.
@fragment
fn ntsc_decode(in: VsOut) -> @location(0) vec4<f32> {
    let zero = vec4<f32>(0.0);
    let one = vec4<f32>(1.0);
    let pi = 3.1415926535;
    let pi2 = 6.283185307;
    let cc_frequency = 3.59754545;
    let notch_width = 2.0;
    let notch_upper_frequency = cc_frequency + notch_width;
    let notch_lower_frequency = cc_frequency - notch_width;
    let y_frequency = 6.0;
    let i_frequency = 1.2;
    let q_frequency = 0.6;
    let scan_time = 52.6;
    let yiq2r = vec3<f32>(1.0, 0.956, 0.621);
    let yiq2g = vec3<f32>(1.0, -0.272, -0.647);
    let yiq2b = vec3<f32>(1.0, -1.106, 1.703);
    let min_c = -1.1183;
    let c_range = 3.2366;
    let pi2_length = pi2 / 83.0;
    let notch_offset = vec4<f32>(0.0, 1.0, 2.0, 3.0);
    let w = vec4<f32>(pi2 * cc_frequency * scan_time);

    var y_accum = zero;
    var i_accum = zero;
    var q_accum = zero;
    let quad_x_size = post.sizes.x * 4.0;
    let time_per_sample = scan_time / quad_x_size;
    let fc_y1 = notch_lower_frequency * time_per_sample;
    let fc_y2 = notch_upper_frequency * time_per_sample;
    let fc_y3 = y_frequency * time_per_sample;
    let fc_i = i_frequency * time_per_sample;
    let fc_q = q_frequency * time_per_sample;
    let pi2_fc_y1 = fc_y1 * pi2;
    let pi2_fc_y2 = fc_y2 * pi2;
    let pi2_fc_y3 = fc_y3 * pi2;
    let pi2_fc_i = fc_i * pi2;
    let pi2_fc_q = fc_q * pi2;
    let fc_y1_2 = fc_y1 * 2.0;
    let fc_y2_2 = fc_y2 * 2.0;
    let fc_y3_2 = fc_y3 * 2.0;
    let fc_i_2 = fc_i * 2.0;
    let fc_q_2 = fc_q * 2.0;
    let coord_y = vec4<f32>(in.uv.y);
    let one_texel = vec2<f32>(1.0) / max(post.sizes.xy, vec2<f32>(1.0));

    for (var n: i32 = -41; n < 42; n = n + 4) {
        let n4 = vec4<f32>(f32(n)) + notch_offset;
        let coord_x = vec4<f32>(in.uv.x) + one_texel.x * n4 * 0.25;
        let c = tex(vec2<f32>(coord_x.x, coord_y.y)) * c_range + vec4<f32>(min_c);
        let wt = w * (vec4<f32>(coord_x) + coord_y * post.sizes.y + vec4<f32>(0.5));
        let cosine = vec4<f32>(0.54) + 0.46 * cos(pi2_length * n4);

        let sinc_y1 = sinc4(pi2_fc_y1 * n4);
        let sinc_y2 = sinc4(pi2_fc_y2 * n4);
        let sinc_y3 = sinc4(pi2_fc_y3 * n4);
        let filter_y = cosine * ((fc_y1_2 * sinc_y1 - fc_y2_2 * sinc_y2) + fc_y3_2 * sinc_y3);

        let sinc_i = sinc4(pi2_fc_i * n4);
        let filter_i = cosine * (fc_i_2 * sinc_i);

        let sinc_q = sinc4(pi2_fc_q * n4);
        let filter_q = cosine * (fc_q_2 * sinc_q);

        y_accum = y_accum + c * filter_y;
        i_accum = i_accum + c * cos(wt) * filter_i;
        q_accum = q_accum + c * sin(wt) * filter_q;
    }

    let y = dot(y_accum, one);
    let i = dot(i_accum, one) * 2.0;
    let q = dot(q_accum, one) * 2.0;
    let yiq = vec3<f32>(y, i, q);
    let out_rgb = vec3<f32>(dot(yiq, yiq2r), dot(yiq, yiq2g), dot(yiq, yiq2b));
    return vec4<f32>(out_rgb, 1.0);
}

// java: program/outline_watercolor.fsh. LumaRamp is declared by the source but never read; it
// remains in the uniform table for exact JSON/default semantics, while the twelve differences
// below follow the source's intentionally rotated neighbour names and offsets.
@fragment
fn outline_watercolor(in: VsOut) -> @location(0) vec4<f32> {
    let one = one_texel();
    let center = tex(in.uv);
    let up = tex(in.uv + vec2<f32>(0.0, -one.y));
    let up2 = tex(in.uv + vec2<f32>(0.0, -one.y) * 2.0);
    let down = tex(in.uv + vec2<f32>(one.x, 0.0));
    let down2 = tex(in.uv + vec2<f32>(one.x, 0.0) * 2.0);
    let left = tex(in.uv + vec2<f32>(-one.x, 0.0));
    let left2 = tex(in.uv + vec2<f32>(-one.x, 0.0) * 2.0);
    let right = tex(in.uv + vec2<f32>(0.0, one.y));
    let right2 = tex(in.uv + vec2<f32>(0.0, one.y) * 2.0);
    let ul = tex(in.uv + vec2<f32>(-one.x, -one.y));
    let ur = tex(in.uv + vec2<f32>(one.x, -one.y));
    let bl = tex(in.uv + vec2<f32>(-one.x, one.y));
    let br = tex(in.uv + vec2<f32>(one.x, one.y));
    let gray = vec4<f32>(0.3, 0.59, 0.11, 0.0);
    let u_diff = dot(abs(center - up), gray);
    let d_diff = dot(abs(center - down), gray);
    let l_diff = dot(abs(center - left), gray);
    let r_diff = dot(abs(center - right), gray);
    let u2_diff = dot(abs(center - up2), gray);
    let d2_diff = dot(abs(center - down2), gray);
    let l2_diff = dot(abs(center - left2), gray);
    let r2_diff = dot(abs(center - right2), gray);
    let ul_diff = dot(abs(center - ul), gray);
    let ur_diff = dot(abs(center - ur), gray);
    let bl_diff = dot(abs(center - bl), gray);
    let br_diff = dot(abs(center - br), gray);
    let sum = u_diff + d_diff + l_diff + r_diff + u2_diff + d2_diff + l2_diff + r2_diff
        + ul_diff + ur_diff + bl_diff + br_diff;
    let sum_luma = clamp(sum, 0.0, 1.0);
    return vec4<f32>(sum_luma, sum_luma, sum_luma, 1.0);
}

// java: program/spiderclip.fsh. The source's bare `clamp(scaledCoord, ...)` has its return
// value discarded, so this translation deliberately keeps the original un-clamped coordinate.
@fragment
fn spider(in: SpiderIn) -> @location(0) vec4<f32> {
    let scaled_texel = tex(in.scaled_coord);
    let blur_texel = textureSample(aux_tex, aux_samp, in.uv);
    var out_texel = scaled_texel;
    let scissor = post.converge_x;
    let vignette = post.converge_y;

    if (in.scaled_coord.x < scissor.x) {
        out_texel = blur_texel;
    }
    if (in.scaled_coord.y < scissor.y) {
        out_texel = blur_texel;
    }
    if (in.scaled_coord.x > scissor.z) {
        out_texel = blur_texel;
    }
    if (in.scaled_coord.y > scissor.w) {
        out_texel = blur_texel;
    }

    if (in.scaled_coord.x < vignette.x) {
        out_texel = mix(
            blur_texel,
            out_texel,
            (scissor.x - in.scaled_coord.x) / (scissor.x - vignette.x),
        );
    }
    if (in.scaled_coord.y < vignette.y) {
        out_texel = mix(
            blur_texel,
            out_texel,
            (scissor.y - in.scaled_coord.y) / (scissor.y - vignette.y),
        );
    }
    if (in.scaled_coord.x > vignette.z) {
        out_texel = mix(
            blur_texel,
            out_texel,
            (scissor.z - in.scaled_coord.x) / (scissor.z - vignette.z),
        );
    }
    if (in.scaled_coord.y > vignette.w) {
        out_texel = mix(
            blur_texel,
            out_texel,
            (scissor.w - in.scaled_coord.y) / (scissor.w - vignette.w),
        );
    }
    return vec4<f32>(out_texel.rgb, 1.0);
}
