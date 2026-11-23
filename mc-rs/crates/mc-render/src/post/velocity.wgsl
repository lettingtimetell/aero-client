// Motion vectors from depth. See `velocity.rs`.
//
// Output, one Rgba16Float texel per pixel, in UV units (0..1 across the screen, y down):
//   rg = where this pixel's surface is now minus where it was on the previous frame
//        (the motion TAA reprojects with: history UV = pixel UV - rg)
//   ba = the same over the motion-blur exposure (the trail the blur draws)
// Both exclude the TAA jitter: a still camera over a still scene writes exactly zero.

struct Reprojection {
    // Inverse of this frame's *jittered* projection: what the depth buffer was rasterised with.
    inv_proj: mat4x4<f32>,
    // Inverse of this frame's view (rotation plus the client's small view-space offsets).
    inv_view: mat4x4<f32>,
    // This frame's unjittered projection * view.
    view_proj: mat4x4<f32>,
    // The previous frame's unjittered projection * view.
    prev_view_proj: mat4x4<f32>,
    // This frame's projection * the view at the start of the blur exposure.
    blur_view_proj: mat4x4<f32>,
    // xyz: this eye minus the previous frame's eye (blocks). w: 1 = the previous frame is valid.
    prev_delta: vec4<f32>,
    // xyz: this eye minus the eye at the start of the exposure (blocks). w: 1 = blur wanted.
    blur_delta: vec4<f32>,
    // xy: this frame's jitter (NDC). zw: the previous frame's jitter (NDC).
    jitter: vec4<f32>,
    // width, height, 1 / width, 1 / height.
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> r: Reprojection;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

// The camera-relative position (`Camera::relative` space) of the surface at pixel `ip`.
fn surface_at(ip: vec2<i32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(ip) + vec2<f32>(0.5)) * r.screen.zw;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let v = r.inv_proj * vec4<f32>(ndc, depth, 1.0);
    let view = v.xyz / v.w;
    return (r.inv_view * vec4<f32>(view, 1.0)).xyz;
}

// Clip space to UV. A point behind the eye lands far off screen, which TAA rejects and the blur
// clamps.
fn clip_to_uv(clip: vec4<f32>) -> vec2<f32> {
    if (clip.w <= 1e-6) {
        return vec2<f32>(16.0, 16.0);
    }
    let ndc = clip.xy / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// Both motions of a surface point `p` whose own displacement since the previous frame is
// `moved` and over the exposure is `moved_blur` (zero for the world, the object's for an entity).
fn motion(p: vec3<f32>, sky: bool, moved: vec3<f32>, moved_blur: vec3<f32>) -> vec4<f32> {
    // The sky is at the far plane, standing in for infinity: only rotation moves it.
    let eye = select(1.0, 0.0, sky);
    let now = clip_to_uv(r.view_proj * vec4<f32>(p, 1.0));
    var out = vec4<f32>(0.0);
    if (r.prev_delta.w > 0.5) {
        let before = r.prev_view_proj * vec4<f32>(p + r.prev_delta.xyz * eye - moved, 1.0);
        out = vec4<f32>(now - clip_to_uv(before), out.zw);
    }
    if (r.blur_delta.w > 0.5) {
        let start = r.blur_view_proj * vec4<f32>(p + r.blur_delta.xyz * eye - moved_blur, 1.0);
        out = vec4<f32>(out.xy, now - clip_to_uv(start));
    }
    return out;
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

// Camera motion for every pixel.
@fragment
fn fs_camera(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ip = vec2<i32>(pos.xy);
    let depth = textureLoad(depth_tex, ip, 0);
    let p = surface_at(ip, depth);
    return motion(p, depth >= 1.0, vec3<f32>(0.0), vec3<f32>(0.0));
}

// ---- object motion: one box per moving entity ----

struct ObjectIn {
    @location(0) min_corner: vec4<f32>,
    @location(1) max_corner: vec4<f32>,
    // xyz: the object's displacement since the previous frame (blocks).
    @location(2) moved: vec4<f32>,
    // xyz: its displacement over the blur exposure (blocks). w: 1 = use it for the blur.
    @location(3) moved_blur: vec4<f32>,
};

struct ObjectOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) min_corner: vec3<f32>,
    @location(1) @interpolate(flat) max_corner: vec3<f32>,
    @location(2) @interpolate(flat) moved: vec3<f32>,
    @location(3) @interpolate(flat) moved_blur: vec4<f32>,
};

@vertex
fn vs_object(@builtin(vertex_index) index: u32, o: ObjectIn) -> ObjectOut {
    // 36 vertices: 6 faces x 2 triangles. Corner bits: x = 1, y = 2, z = 4. No culling, so
    // the winding does not matter and a camera inside the box still covers it.
    var corners = array<u32, 36>(
        0u, 2u, 1u, 1u, 2u, 3u,   // -z
        4u, 5u, 6u, 5u, 7u, 6u,   // +z
        0u, 1u, 4u, 1u, 5u, 4u,   // -y
        2u, 6u, 3u, 3u, 6u, 7u,   // +y
        0u, 4u, 2u, 2u, 4u, 6u,   // -x
        1u, 3u, 5u, 3u, 7u, 5u,   // +x
    );
    let c = corners[index];
    let pick = vec3<f32>(f32(c & 1u), f32((c >> 1u) & 1u), f32((c >> 2u) & 1u));
    let p = mix(o.min_corner.xyz, o.max_corner.xyz, pick);
    var out: ObjectOut;
    out.clip = r.view_proj * vec4<f32>(p, 1.0);
    out.min_corner = o.min_corner.xyz;
    out.max_corner = o.max_corner.xyz;
    out.moved = o.moved.xyz;
    out.moved_blur = o.moved_blur;
    return out;
}

// Keep only the pixels whose surface lies inside the box: the entity itself, not the world
// behind or in front of it. Entity rendering is untouched.
@fragment
fn fs_object(in: ObjectOut) -> @location(0) vec4<f32> {
    let ip = vec2<i32>(in.clip.xy);
    let depth = textureLoad(depth_tex, ip, 0);
    let p = surface_at(ip, depth);
    if (depth >= 1.0 || any(p < in.min_corner) || any(p > in.max_corner)) {
        discard;
    }
    let blur_moved = in.moved_blur.xyz * in.moved_blur.w;
    return motion(p, false, in.moved, blur_moved);
}
