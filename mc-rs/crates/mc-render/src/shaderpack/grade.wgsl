// The colour-grading LUT bake. Every colour operation of a shader
// pack's grade runs here once per LUT entry, never per pixel: `cs_bake` writes a 33^3 table that a
// pack's final pass reads with one filtered lookup (`grade::LUT_APPLY_WGSL`). The table is re-baked
// only when a setting changes.
//
// `grade.rs` holds the CPU reference (`GradeParams::apply`), which runs the same steps in the same
// order on the same derived uniform, and the tests that compare the two.
//
// Input and output are display-encoded RGB in [0, 1] (the frame's byte domain: the surface is
// `Bgra8Unorm` and vanilla blends in gamma space). The steps:
//
//   1. decode to linear light (the sRGB transfer function);
//   2. one 3x3 colour matrix: the white balance (worked out on the CPU, see `white_balance_matrix`
//      in grade.rs) times the exposure multiplier (2^EV);
//   3. contrast as a power around linear mid grey 0.18, so black stays black;
//   4. hue shift, saturation and vibrance in OKLab (lightness is untouched);
//   5. a highlight shoulder that is exactly the identity unless 1-3 can push white above 1;
//   6. encode back to display values, then, in display space: lift / gamma / gain colour wheels,
//      split toning (soft light toward a shadow and a highlight colour), a gradient map (3 stops)
//      and fade (lifted blacks).
//
// Every step is the identity at its neutral setting, and the uniform's flags skip the OKLab round
// trip when it would change nothing, so the neutral grade bakes an exact identity table.

struct GradeUniform {
    // Columns (xyz) of the linear-light colour matrix, white balance times exposure; colour0.w is
    // 1 when the matrix is not the identity.
    colour0: vec4<f32>,
    colour1: vec4<f32>,
    colour2: vec4<f32>,
    // x = contrast (power around 0.18), y = saturation, z = vibrance, w = hue shift (radians).
    tone: vec4<f32>,
    // x = 1 when the OKLab step runs, y = shoulder maximum (>= 1), z = fade level, w = gradient mix.
    flags: vec4<f32>,
    // rgb = lift offset at black (display units), w unused.
    lift: vec4<f32>,
    // rgb = per-channel gamma (1 = unchanged), w = 1 when the display-space steps run.
    gamma: vec4<f32>,
    // rgb = per-channel gain (1 = unchanged), w unused.
    gain: vec4<f32>,
    // rgb = split-toning shadow colour, a = split-toning strength.
    split_shadow: vec4<f32>,
    // rgb = split-toning highlight colour, a = balance (-1 .. 1, + favours highlights).
    split_highlight: vec4<f32>,
    // Gradient-map stops (rgb; a unused): dark, mid, light.
    grad_dark: vec4<f32>,
    grad_mid: vec4<f32>,
    grad_light: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: GradeUniform;
@group(0) @binding(1) var lut_out: texture_storage_3d<rgba16float, write>;

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

fn cbrt(x: f32) -> f32 {
    return sign(x) * pow(abs(x), 1.0 / 3.0);
}

// OKLab, linear sRGB in and out.
fn linear_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let l_ = cbrt(l);
    let m_ = cbrt(m);
    let s_ = cbrt(s);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}

fn oklab_to_linear(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

// Steps 2 and 3 on linear light: the colour matrix, then contrast.
fn grade_linear(lin_in: vec3<f32>) -> vec3<f32> {
    var lin = lin_in;
    if (u.colour0.w > 0.5) {
        lin = mat3x3<f32>(u.colour0.xyz, u.colour1.xyz, u.colour2.xyz) * lin;
    }
    lin = max(lin, vec3<f32>(0.0));
    if (u.tone.x != 1.0) {
        lin = 0.18 * pow(lin / 0.18, vec3<f32>(u.tone.x));
    }
    return lin;
}

// The shoulder of step 5: the identity below 0.75 and when `m` <= 1, otherwise a curve with slope 1
// at the knee that lands `m` exactly on 1.
fn shoulder(x: f32, m: f32) -> f32 {
    let knee = 0.75;
    if (m <= 1.0 || x <= knee) {
        return x;
    }
    let n = (m - knee) / (1.0 - knee);
    let t = clamp((x - knee) / (m - knee), 0.0, 1.0);
    return knee + (1.0 - knee) * (1.0 - pow(1.0 - t, n));
}

// W3C soft light, `a` the base and `b` the blend colour; b = 0.5 leaves a unchanged.
fn soft_light(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let d = select(sqrt(a), ((16.0 * a - vec3<f32>(12.0)) * a + vec3<f32>(4.0)) * a,
        a <= vec3<f32>(0.25));
    let dark = a - (vec3<f32>(1.0) - 2.0 * b) * a * (vec3<f32>(1.0) - a);
    let light = a + (2.0 * b - vec3<f32>(1.0)) * (d - a);
    return select(light, dark, b <= vec3<f32>(0.5));
}

// The whole grade for one display-encoded colour.
fn grade(c_in: vec3<f32>) -> vec3<f32> {
    let c = clamp(c_in, vec3<f32>(0.0), vec3<f32>(1.0));
    var lin = grade_linear(srgb_to_linear(c));

    if (u.flags.x > 0.5) {
        var lab = linear_to_oklab(lin);
        let cs = cos(u.tone.w);
        let sn = sin(u.tone.w);
        var ab = vec2<f32>(lab.y * cs - lab.z * sn, lab.y * sn + lab.z * cs);
        let chroma = length(ab);
        let vib = 1.0 + u.tone.z * (1.0 - clamp(chroma / 0.3, 0.0, 1.0));
        ab = ab * max(u.tone.y * vib, 0.0);
        lab = vec3<f32>(lab.x, ab);
        lin = max(oklab_to_linear(lab), vec3<f32>(0.0));
    }

    let m = u.flags.y;
    lin = vec3<f32>(shoulder(lin.r, m), shoulder(lin.g, m), shoulder(lin.b, m));
    var g = clamp(linear_to_srgb(lin), vec3<f32>(0.0), vec3<f32>(1.0));

    if (u.gamma.w > 0.5) {
        // Lift / gamma / gain: gain scales toward white, lift adds at black and fades out toward
        // white, gamma bends the mids per channel.
        g = g * u.gain.rgb + u.lift.rgb * (vec3<f32>(1.0) - g);
        g = pow(clamp(g, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0) / u.gamma.rgb);

        // Split toning.
        let strength = u.split_shadow.a;
        if (strength > 0.0) {
            let t = clamp(dot(g, LUMA) + u.split_highlight.a * 0.5, 0.0, 1.0);
            let shadow_blend = mix(vec3<f32>(0.5), u.split_shadow.rgb, strength * (1.0 - t));
            let highlight_blend = mix(vec3<f32>(0.5), u.split_highlight.rgb, strength * t);
            g = soft_light(g, shadow_blend);
            g = soft_light(g, highlight_blend);
        }

        // Gradient map over luma, three stops.
        if (u.flags.w > 0.0) {
            let y = clamp(dot(g, LUMA), 0.0, 1.0);
            let lower = mix(u.grad_dark.rgb, u.grad_mid.rgb, clamp(y * 2.0, 0.0, 1.0));
            let upper = mix(u.grad_mid.rgb, u.grad_light.rgb, clamp(y * 2.0 - 1.0, 0.0, 1.0));
            let mapped = select(upper, lower, y < 0.5);
            g = mix(g, mapped, u.flags.w);
        }

        // Fade: black lifts to the fade level, white stays white.
        g = vec3<f32>(u.flags.z) + g * (1.0 - u.flags.z);
    }
    return clamp(g, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(4, 4, 4)
fn cs_bake(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = textureDimensions(lut_out);
    if (id.x >= n.x || id.y >= n.y || id.z >= n.z) {
        return;
    }
    let c = vec3<f32>(id) / vec3<f32>(n - vec3<u32>(1u));
    textureStore(lut_out, id, vec4<f32>(grade(c), 1.0));
}
