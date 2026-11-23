// Custom Sky: one resource-pack sky layer, a cube
// textured from a 3x2 grid of faces. See custom.rs.
//
// The CPU has already folded the sky projection, the camera's rotation-only view and the layer's
// own time-of-day rotation into `view_proj`, and the per-blend-mode fade into `color` (the
// format's `GlStateManager.color` for that mode).

struct Layer {
    view_proj: mat4x4<f32>,
    // The texture is multiplied by this (the fade, on the channels the blend mode fades).
    color: vec4<f32>,
    // x: discard when the faded alpha is at most this (`replace`: vanilla's 0.1 alpha test;
    // every other mode: -1, never). zw: half a texel in UV units, to keep samples inside the face.
    params: vec4<f32>,
};

@group(0) @binding(0) var layer_tex: texture_2d<f32>;
@group(0) @binding(1) var layer_samp: sampler;
@group(1) @binding(0) var<uniform> layer: Layer;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    // The face's cell in the texture: u0, v0, u1, v1.
    @location(2) cell: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) cell: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = layer.view_proj * vec4<f32>(in.pos, 1.0);
    out.uv = in.uv;
    out.cell = in.cell;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Never sample the neighbouring face across a cell border.
    let lo = in.cell.xy + layer.params.zw;
    let hi = max(in.cell.zw - layer.params.zw, lo);
    let uv = clamp(in.uv, lo, hi);
    let c = textureSample(layer_tex, layer_samp, uv) * layer.color;
    if (c.a <= layer.params.x) {
        discard;
    }
    return c;
}
