// The Underwater pack's marine snow.
//
// Prepended at build time: `post/view_geometry_common.wgsl` (the camera uniform at @group(0)).
//
// No vertex buffer: speck `i` is instance `i`, six vertices each. Its home is a hash of `i` inside
// a box of `eye.w` blocks; the box repeats through the world and is wrapped around the eye, so
// specks never run out, never pop in front of the camera (they fade in at the box edge, which is
// far away) and cost nothing on the CPU. Specks are depth-tested against the world, fade out
// close to the eye (nothing ever covers the crosshair) and are absorbed by their own distance.

struct SnowParams {
    // xyz = eye position modulo the box (blocks), w = box size (blocks).
    eye: vec4<f32>,
    // xyz = how far the water has drifted (modulo the box), w = wave loop phase (0..1).
    drift: vec4<f32>,
    // rgb = speck colour (display), a = opacity.
    color: vec4<f32>,
    // rgb = absorption per block, a = clear range (blocks).
    sigma: vec4<f32>,
    // x = speck radius (blocks), y = smallest radius (pixels), z = fully visible from (blocks),
    // w = fade-out start as a fraction of half the box.
    size: vec4<f32>,
};
@group(1) @binding(0) var<uniform> snow: SnowParams;

struct SnowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) alpha: f32,
};

fn snow_hash(seed: u32) -> u32 {
    // PCG-style integer hash.
    var h = seed * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn snow_hash3(seed: u32) -> vec3<f32> {
    let a = snow_hash(seed);
    let b = snow_hash(a ^ 0x68E31DA4u);
    let c = snow_hash(b ^ 0xB5297A4Du);
    return vec3<f32>(vec3<u32>(a, b, c) >> vec3<u32>(8u)) / 16777216.0;
}

@vertex
fn vs_snow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SnowOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vi];
    let home = snow_hash3(ii);
    let shape = snow_hash3(ii ^ 0x9E3779B9u);
    let box = snow.eye.w;

    // Each speck also circles slowly on its own (integer cycles per loop, so the loop is seamless).
    let ph = snow.drift.w * 6.2831853;
    let fx = f32(1u + (ii % 3u));
    let fz = f32(1u + ((ii / 3u) % 2u));
    let wobble = vec3<f32>(
        sin(ph * fx + shape.x * 6.2831853),
        0.5 * sin(ph * fz + shape.z * 6.2831853),
        cos(ph * fz + shape.y * 6.2831853)
    ) * 0.4;
    let p = home * box + snow.drift.xyz + wobble;
    // Camera-relative position, wrapped into [-box/2, box/2) around the eye.
    let rel = (fract((p - snow.eye.xyz) / box + 0.5) - 0.5) * box;
    let center = camera.view_proj * vec4<f32>(rel, 1.0);
    let dist = length(rel);

    var out: SnowOut;
    out.corner = corner;
    if (center.w <= 0.05) {
        out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.alpha = 0.0;
        return out;
    }
    let px_per_block = camera.proj[1][1] * camera.screen.y * 0.5 / center.w;
    let r_true = snow.size.x * (0.6 + 0.8 * shape.z) * px_per_block;
    let r_px = max(r_true, snow.size.y);
    // A speck smaller than the smallest radius is drawn at that radius but fainter, so its total
    // light matches its real size (no shimmering sub-pixel dots).
    let coverage = min(r_true / snow.size.y, 1.0);
    out.clip = center + vec4<f32>(corner * r_px * 2.0 / camera.screen.xy * center.w, 0.0, 0.0);
    let half_box = box * 0.5;
    let near = smoothstep(0.8, snow.size.z, dist);
    let far = 1.0 - smoothstep(snow.size.w * half_box, half_box, dist);
    let sig = dot(snow.sigma.rgb, vec3<f32>(0.3333));
    let absorbed = exp(-sig * max(dist - snow.sigma.a, 0.0));
    out.alpha = snow.color.a * near * far * absorbed * coverage * coverage * (0.45 + 0.55 * shape.x);
    return out;
}

@fragment
fn fs_snow(in: SnowOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.corner, in.corner);
    if (r2 >= 1.0 || in.alpha <= 0.0) {
        discard;
    }
    let falloff = 1.0 - r2;
    let a = in.alpha * falloff;
    // Premultiplied: blended with (One, OneMinusSrcAlpha) on colour only.
    return vec4<f32>(snow.color.rgb * a, a);
}
