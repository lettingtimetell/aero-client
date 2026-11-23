// Video export on the GPU.
//
// `accumulate` adds one rendered sub-frame (display-encoded, converted to linear light) into a
// float accumulator, for frame-blending motion blur. `convert` turns either the frame itself or
// the accumulator's mean into NV12 (BT.709, limited range), packed four bytes per u32, exactly as
// `mc_video::nv12::from_rgba` does on the CPU (the reference the tests compare against).

struct Params {
    width: u32,
    height: u32,
    // 1: the source texture is an sRGB format, so `textureLoad` returns linear light.
    source_srgb: u32,
    // Sub-frames in the accumulator; 0: `convert` reads the source texture directly.
    samples: u32,
    // 1: `accumulate` starts a new output frame (writes instead of adding).
    first: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> nv12: array<u32>;

const KR: f32 = 0.2126;
const KB: f32 = 0.0722;
const KG: f32 = 0.7152;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// The display-encoded colour of one pixel of the output frame.
fn display_rgb(x: u32, y: u32) -> vec3<f32> {
    if (params.samples > 0u) {
        let sum = accum[y * params.width + x].rgb;
        return linear_to_srgb(clamp(sum / f32(params.samples), vec3<f32>(0.0), vec3<f32>(1.0)));
    }
    let c = textureLoad(source, vec2<i32>(i32(x), i32(y)), 0).rgb;
    if (params.source_srgb == 1u) {
        return linear_to_srgb(c);
    }
    return c;
}

fn quantize(v: f32) -> u32 {
    return u32(clamp(floor(v + 0.5), 0.0, 255.0));
}

fn luma_prime(c: vec3<f32>) -> f32 {
    return KR * c.r + KG * c.g + KB * c.b;
}

// Byte `b` of the NV12 frame.
fn nv12_byte(b: u32) -> u32 {
    let w = params.width;
    let luma_bytes = w * params.height;
    if (b < luma_bytes) {
        let y = b / w;
        let x = b - y * w;
        return quantize(16.0 + 219.0 * luma_prime(display_rgb(x, y)));
    }
    // Chroma: rows of w bytes, each a (U, V) pair per 2x2 block.
    let c = b - luma_bytes;
    let row = c / w;
    let col = c - row * w;
    let bx = (col / 2u) * 2u;
    let by = row * 2u;
    let is_v = (col & 1u) == 1u;
    var sum = 0.0;
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
        for (var dx = 0u; dx < 2u; dx = dx + 1u) {
            let rgb = display_rgb(bx + dx, by + dy);
            let yp = luma_prime(rgb);
            var d: f32;
            if (is_v) {
                d = (rgb.r - yp) / (2.0 * (1.0 - KR));
            } else {
                d = (rgb.b - yp) / (2.0 * (1.0 - KB));
            }
            sum = sum + 128.0 + 224.0 * d;
        }
    }
    return quantize(sum * 0.25);
}

@compute @workgroup_size(64)
fn convert(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let word = id.x + id.y * groups.x * 64u;
    let total = params.width * params.height * 3u / 2u;
    let first_byte = word * 4u;
    if (first_byte >= total) {
        return;
    }
    var packed = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let b = first_byte + k;
        if (b < total) {
            packed = packed | (nv12_byte(b) << (8u * k));
        }
    }
    nv12[word] = packed;
}

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    var c = textureLoad(source, vec2<i32>(i32(id.x), i32(id.y)), 0).rgb;
    if (params.source_srgb == 0u) {
        c = srgb_to_linear(c);
    }
    let i = id.y * params.width + id.x;
    if (params.first == 1u) {
        accum[i] = vec4<f32>(c, 1.0);
    } else {
        accum[i] = accum[i] + vec4<f32>(c, 1.0);
    }
}
