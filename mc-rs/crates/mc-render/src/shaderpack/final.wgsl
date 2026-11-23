// Shader packs' final pass: bloom and tone mapping.
//
// Appended to `common.wgsl`. Bloom is a dual filter: a thresholded,
// Karis-averaged prefilter to half resolution, 5-tap downsamples to 1/4..1/64, then 8-tap tent
// upsamples added back level by level. Everything bloom stores is linear light. The composite
// reads the frame (display-encoded, 8-bit), linearises it, adds the bloom, applies exposure, the
// tone curve, contrast, saturation and vibrance, and writes it back display-encoded with a
// sub-LSB dither so gradients do not band.

struct FinalParams {
    // x: bloom threshold (linear luma), y: soft knee, z: bloom strength, w: 1 = bloom on.
    bloom: vec4<f32>,
    // x: exposure multiplier, y: contrast, z: saturation, w: vibrance.
    tone: vec4<f32>,
    // x: curve (0 filmic, 1 aces, 2 linear), y: 1 = tone map on, z: bloom level count,
    // w: frame index (dither rotation).
    mode: vec4<f32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> params: FinalParams;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let c = corners[index];
    var out: VsOut;
    out.pos = vec4<f32>(c, 0.0, 1.0);
    out.uv = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    return out;
}

fn karis_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + pk_luma(c));
}

// Soft-knee threshold on linear luminance (keeps hue: scales the colour). With `over` the
// luminance above the threshold, the kept luminance is 0 below -knee, `over` itself above +knee,
// and in between the parabola that meets both lines with matching slopes,
// (over + knee)^2 / (4 knee).
fn threshold(c: vec3<f32>) -> vec3<f32> {
    let knee = max(params.bloom.y, 1e-4);
    let l = pk_luma(c);
    let over = l - params.bloom.x;
    let q = over + knee;
    let curve = select(q * q / (4.0 * knee), 0.0, over <= -knee);
    let kept = select(curve, over, over >= knee);
    return c * (kept / max(l, 1e-4));
}

// Full resolution (display-encoded) -> half resolution (linear, thresholded). Four bilinear
// taps cover the 4x4 source texels under the output texel; each is thresholded and weighted by
// 1/(1+luma) so one bright pixel (a torch flame, a hit spark) cannot flicker a whole glow.
@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let o = texel;
    let a = threshold(pk_to_linear(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-o.x, -o.y), 0.0).rgb));
    let b = threshold(pk_to_linear(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(o.x, -o.y), 0.0).rgb));
    let c = threshold(pk_to_linear(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-o.x, o.y), 0.0).rgb));
    let d = threshold(pk_to_linear(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(o.x, o.y), 0.0).rgb));
    let wa = karis_weight(a);
    let wb = karis_weight(b);
    let wc = karis_weight(c);
    let wd = karis_weight(d);
    let sum = a * wa + b * wb + c * wc + d * wd;
    return vec4<f32>(sum / max(wa + wb + wc + wd, 1e-4), 1.0);
}

// One bilinear tap of the bloom chain's source level, `o` source texels (`texel` in UV) from `uv`.
fn chain_tap(uv: vec2<f32>, o: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src_tex, src_samp, uv + o * texel, 0.0).rgb;
}

// The signs of diagonal `k` (0..3): (-, -), (+, -), (-, +), (+, +).
fn diagonal_signs(k: i32) -> vec2<f32> {
    return vec2<f32>(select(-1.0, 1.0, (k & 1) == 1), select(-1.0, 1.0, (k & 2) == 2));
}

// The axis tap that goes with diagonal `k` in the upsample: -x, +x (k = 0, 1), then -y, +y
// (k = 2, 3), one source texel out.
fn up_axis(k: i32) -> vec2<f32> {
    let d = diagonal_signs(k);
    return select(vec2<f32>(0.0, d.x), vec2<f32>(d.x, 0.0), k < 2);
}

// Dual-filter downsample: the tap at the target texel's centre counts four times, the four taps
// half a source texel out on the diagonals once each.
@fragment
fn fs_down(in: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));
    var sum = chain_tap(in.uv, vec2<f32>(0.0), texel) * 4.0;
    // The four diagonals written out, not looped: every read is issued before the sums need one.
    sum += chain_tap(in.uv, diagonal_signs(0) * 0.5, texel);
    sum += chain_tap(in.uv, diagonal_signs(1) * 0.5, texel);
    sum += chain_tap(in.uv, diagonal_signs(2) * 0.5, texel);
    sum += chain_tap(in.uv, diagonal_signs(3) * 0.5, texel);
    return vec4<f32>(sum / 8.0, 1.0);
}

// Dual-filter upsample, a tent of eight taps: one source texel out along each axis (weight 1) and
// half a texel out on each diagonal (weight 2). Added onto the level below by the blend state.
@fragment
fn fs_up(in: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));
    var sum = vec3<f32>(0.0);
    // Written out, not looped: each diagonal tap (weight 2) and then its axis tap, -x, +x, -y, +y.
    sum += chain_tap(in.uv, diagonal_signs(0) * 0.5, texel) * 2.0;
    sum += chain_tap(in.uv, up_axis(0), texel);
    sum += chain_tap(in.uv, diagonal_signs(1) * 0.5, texel) * 2.0;
    sum += chain_tap(in.uv, up_axis(1), texel);
    sum += chain_tap(in.uv, diagonal_signs(2) * 0.5, texel) * 2.0;
    sum += chain_tap(in.uv, up_axis(2), texel);
    sum += chain_tap(in.uv, diagonal_signs(3) * 0.5, texel) * 2.0;
    sum += chain_tap(in.uv, up_axis(3), texel);
    return vec4<f32>(sum / 12.0, 1.0);
}

// A fitted ACES curve, normalised so 1.0 in stays (almost) 1.0 out.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let s = x * 0.8;
    let y = (s * (2.51 * s + 0.03)) / (s * (2.43 * s + 0.59) + 0.14);
    return y / 0.6;
}

// A soft shoulder: identity below `knee`, then an exponential roll-off toward 1. Bloom and
// exposure push values above 1 without ever clipping to a flat white.
fn shoulder(x: vec3<f32>) -> vec3<f32> {
    let knee = 0.78;
    let w = 1.0 - knee;
    let over = max(x - vec3<f32>(knee), vec3<f32>(0.0));
    let rolled = vec3<f32>(knee) + w * (vec3<f32>(1.0) - exp(-over / w));
    return select(x, rolled, x > vec3<f32>(knee));
}

fn tone(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in * params.tone.x;
    let curve = params.mode.x;
    // Contrast around mid grey, in log space (keeps hue; 1.0 = unchanged).
    let pivot = 0.18;
    c = pivot * pow(max(c, vec3<f32>(1e-6)) / pivot, vec3<f32>(params.tone.y));
    if (curve < 0.5) {
        c = shoulder(c);
    } else if (curve < 1.5) {
        c = min(aces(c), vec3<f32>(1.0));
    } else {
        c = min(c, vec3<f32>(1.0));
    }
    // Saturation and vibrance around luminance.
    let l = pk_luma(c);
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let sat_now = (mx - mn) / max(mx, 1e-4);
    let vib = 1.0 + params.tone.w * (1.0 - sat_now);
    c = vec3<f32>(l) + (c - vec3<f32>(l)) * params.tone.z * vib;
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let src = textureLoad(src_tex, vec2<i32>(in.pos.xy), 0);
    var c = pk_to_linear(src.rgb);
    if (params.bloom.w > 0.5) {
        let glow = textureSampleLevel(bloom_tex, src_samp, in.uv, 0.0).rgb;
        c += glow * params.bloom.z;
    }
    if (params.mode.y > 0.5) {
        c = tone(c);
    }
    var out = pk_to_display(c);
    // A fixed triangular dither below one 8-bit step (fixed, so it can never flicker).
    let n = pk_ign(in.pos.xy) + pk_ign(in.pos.yx + vec2<f32>(3.1, 7.0)) - 1.0;
    out += vec3<f32>(n / 255.0);
    return vec4<f32>(out, src.a);
}
