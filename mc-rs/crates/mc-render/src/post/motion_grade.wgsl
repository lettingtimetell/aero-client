// Motion blur + ASC CDL v1.2 colour grade in one fullscreen pass.
//
// See `motion_grade.rs` for the maths and why each step is there. Three entry points, one per
// pipeline, so no pixel pays for a branch: `fs_blur_grade` and `fs_blur` write the frame plus
// the Rgba16Float history (two colour targets), `fs_grade` writes the frame only.
//
// Every read is a `textureLoad` at the fragment's own integer pixel, like the scene blit in
// `pipeline.rs`: no sampler, no filtering, no half-texel question.

struct Params {
    // xyz: effective CDL slope. w: saturation.
    slope: vec4<f32>,
    // xyz: effective CDL offset. w: this frame's history weight `a` (0 = show the scene sharp).
    offset: vec4<f32>,
    // xyz: effective CDL power (>= 0.01). w: the convergence snap epsilon.
    power: vec4<f32>,
    // x: 1 = the fp16 progress floor is on (0 only in the test that shows why it exists).
    misc: vec4<f32>,
};

// The frame so far (the PhaseList snapshot), display-encoded: the surface's byte domain.
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var<uniform> p: Params;
// The previous blend, un-graded. Only the two blur pipelines bind this group.
@group(1) @binding(0) var hist_in: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

// ASC CDL v1.2 forward (the clamped form) on display-encoded values.
fn cdl(c: vec3<f32>) -> vec3<f32> {
    // 1-3: slope, offset, then clamp to [0, 1] *before* the power (v1.2).
    var v = clamp(c * p.slope.xyz + p.offset.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    // 4: power. v >= 0 and power >= 0.01, so pow is defined everywhere (pow(0, y) = 0).
    v = pow(v, p.power.xyz);
    // 5: saturation around the Rec.709 luma of the post-power value.
    let luma = vec3<f32>(dot(v, vec3<f32>(0.2126, 0.7152, 0.0722)));
    v = luma + p.slope.w * (v - luma);
    // 6: final clamp.
    return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
}

// The spacing of fp16 values at `h` (h >= 0): 2^(exponent - 10), or 2^-24 below fp16's normal
// range. Read straight from the f32 exponent bits, so it is exact even at powers of two.
fn ulp16(h: vec3<f32>) -> vec3<f32> {
    let exponent = (bitcast<vec3<u32>>(abs(h)) >> vec3<u32>(23u)) & vec3<u32>(0xffu);
    let clamped = max(exponent, vec3<u32>(113u)); // 113 = 127 - 14: fp16's smallest normal
    return bitcast<vec3<f32>>((clamped - vec3<u32>(10u)) << vec3<u32>(23u));
}

// `c` rounded to the nearest fp16 value (ties to even), so the history target stores it exactly
// and never applies its own rounding, which Vulkan leaves to the implementation. The RTX 5070 Ti
// this was written on truncates (`tests/m7_postfx.rs` probes it), which lost about half a spacing
// per frame on every rising trail: 2/255 behind at 3000 FPS, measured. Exact values make the
// trail the same on every GPU. Below fp16's normal range this uses its subnormal spacing.
fn to_fp16(c: vec3<f32>) -> vec3<f32> {
    let u = ulp16(c);
    return round(c / u) * u;
}

// One step of the time-based feedback blend: mix the scene into the history by `a`.
fn blend(curr: vec3<f32>, ip: vec2<i32>) -> vec3<f32> {
    let h = textureLoad(hist_in, ip, 0).rgb;
    var c = mix(curr, h, vec3<f32>(p.offset.w));
    let d = curr - h;
    // Progress floor: a step smaller than one fp16 spacing can be rounded straight back to `h`
    // when stored, and the pixel would stay a ghost forever. Above roughly 77 % strength at
    // 3000 FPS that band is wider than the snap below, so move at least one spacing instead.
    let u = ulp16(h) * p.misc.x;
    c = select(c, h + sign(d) * u, abs(c - h) < u);
    // Convergence snap: once within epsilon (or within one spacing), land exactly on the scene.
    let snap = (abs(c - curr) < vec3<f32>(p.power.w)) | (abs(d) <= u);
    return select(c, curr, snap);
}

struct BlurOut {
    @location(0) frame: vec4<f32>,
    @location(1) history: vec4<f32>,
};

@fragment
fn fs_blur_grade(@builtin(position) pos: vec4<f32>) -> BlurOut {
    let ip = vec2<i32>(pos.xy);
    let scene_px = textureLoad(scene, ip, 0);
    let c = blend(scene_px.rgb, ip);
    var out: BlurOut;
    // The history keeps the *un-graded* blend, so the grade never compounds through feedback.
    out.history = vec4<f32>(to_fp16(c), 1.0);
    out.frame = vec4<f32>(cdl(c), scene_px.a);
    return out;
}

@fragment
fn fs_blur(@builtin(position) pos: vec4<f32>) -> BlurOut {
    let ip = vec2<i32>(pos.xy);
    let scene_px = textureLoad(scene, ip, 0);
    let c = blend(scene_px.rgb, ip);
    var out: BlurOut;
    out.history = vec4<f32>(to_fp16(c), 1.0);
    out.frame = vec4<f32>(c, scene_px.a);
    return out;
}

@fragment
fn fs_grade(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let scene_px = textureLoad(scene, vec2<i32>(pos.xy), 0);
    return vec4<f32>(cdl(scene_px.rgb), scene_px.a);
}
