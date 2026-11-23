// The window's live preview while a video export renders
// offscreen. Draws one textured rectangle per call: the export frame (letterboxed) or the
// progress panel.

struct Quad {
    // Rectangle in clip space: x0, y0 (top-left), x1, y1 (bottom-right).
    rect: vec4<f32>,
    // Multiplies the sampled colour (dimming, fading).
    tint: vec4<f32>,
};

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@group(0) @binding(2) var<uniform> quad: Quad;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
    // Two triangles: (0,1,2) (2,1,3) over the corners TL, BL, TR, BR.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let c = corners[index];
    var out: VertexOut;
    out.position = vec4<f32>(
        mix(quad.rect.x, quad.rect.z, c.x),
        mix(quad.rect.y, quad.rect.w, c.y),
        0.0,
        1.0,
    );
    out.uv = c;
    return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(image, image_sampler, in.uv) * quad.tint;
}
