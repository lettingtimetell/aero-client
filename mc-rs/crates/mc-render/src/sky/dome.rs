//! The sky dome: sky/void plane, sunset glow, sun, moon, stars, and the End's skybox.
//!
//! java: `RenderGlobal.renderSky(float, int)` (line 1203), `renderSky(WorldRenderer, float,
//! boolean)` (the plane builder, line 340), `renderStars` (line 403), `renderSkyEnd` (line 1148).
//!
//! # Eye-relative geometry
//!
//! Every vertex this module emits is expressed as an offset from the camera's eye, not as a
//! world position — mirroring vanilla's own behaviour of recentring the sky plane / void plane /
//! star field on the player before drawing each frame (`RenderGlobal` re-issues the sky draw
//! from a matrix that has already translated to the eye). Since [`crate::Camera::view`] is
//! rotation-only, feeding it eye-relative positions directly (no [`crate::Camera::relative`]
//! subtraction needed — there is no world position here to subtract *from*) reproduces that for
//! free: the dome turns with the camera's look but never appears to translate as the player
//! walks.

use glam::{Mat3, Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::sky::environment::{Dimension, Environment};
use crate::sky::fog::{sky_pass_fog_mode, FogInputs, FogMode, FogState};
use crate::sky::stars::generate_stars;
use crate::sky::textures::{self, SkyTexture};
use crate::{Camera, DepthAttachment, Gpu, RenderPhase};

/// One sky-dome vertex: eye-relative position, texture UV (unused — `(0,0)` — for untextured
/// elements), and a per-vertex colour that already has this frame's alpha/tint baked in.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DomeVertex {
    pos: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

impl DomeVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<DomeVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4],
    };
}

/// One draw's overrides, read by `dome.wgsl` as an instance-rate attribute (`first_instance`
/// selects the record), so the baked planes and star field stay resident on the GPU.
///
/// java: `glSkyList`/`glSkyList2`/`starGLCallList` are display lists compiled once; per draw
/// vanilla only changes `GlStateManager.color` and a Y `translate`. Both overrides here *replace*
/// a value rather than doing arithmetic on the GPU: every vertex of a plane shares one baked Y,
/// so `y` is that Y plus the translation, added on the CPU exactly as [`flat_verts`] did, and the
/// colour is the very `[f32; 4]` [`flat_verts`] used to write into each vertex. The vertex shader
/// therefore sees bit-identical positions and colours to the per-frame rebuild it replaces.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct DomeDraw {
    /// Replacement vertex colour when `mode[0] == 1`.
    tint: [f32; 4],
    /// `x` = 1 replaces the vertex colour with `tint`; `y` = 1 replaces the vertex Y with `z`.
    mode: [f32; 4],
    /// `z` = 1 rotates the vertex by [`celestial_rotation`] given `x = cos`, `y = sin` of the
    /// celestial angle in radians. Only the star field uses it, and only with
    /// [`SkyPhase::rotate_stars`].
    rot: [f32; 4],
}

impl DomeDraw {
    /// Use the vertex stream as-is: the per-frame sun, moon, fan and void box, and the End cube.
    const PASSTHROUGH: Self = Self {
        tint: [0.0; 4],
        mode: [0.0; 4],
        rot: [0.0; 4],
    };

    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<DomeDraw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4],
    };

    /// `GlStateManager.color(..)` around a display list whose positions are left alone.
    fn tinted(color: [f32; 4]) -> Self {
        Self {
            tint: color,
            mode: [1.0, 0.0, 0.0, 0.0],
            rot: [0.0; 4],
        }
    }

    /// Also apply `rotate(-90, 0,1,0); rotate(celestialAngle * 360, 1,0,0)` to the positions.
    fn celestially_rotated(mut self, celestial_angle: f32) -> Self {
        let (sin, cos) = (celestial_angle * std::f32::consts::TAU).sin_cos();
        self.rot = [cos, sin, 1.0, 0.0];
        self
    }

    /// `color(..)` + `translate(0, dy, 0)` around a baked plane: `y` is the plane's shared baked
    /// Y with `dy` already added, see [`plane_draw_y`].
    fn plane(color: [f32; 4], y: f32) -> Self {
        Self {
            tint: color,
            mode: [1.0, 1.0, y, 0.0],
            rot: [0.0; 4],
        }
    }
}

/// The most draw records one frame writes: the passthrough record, the sky plane, the stars, the
/// black void plane and the horizon plane.
const MAX_DOME_DRAWS: usize = 5;

/// Vertices rebuilt per frame (everything that is not a baked plane or the star field): the
/// sunrise fan, the sun and moon quads, and the void box.
const DYNAMIC_DOME_VERTICES: usize =
    SUNRISE_FAN_VERTICES + 2 * VERTICES_PER_QUAD + 5 * VERTICES_PER_QUAD;

/// The star field as a white triangle list. java: `RenderGlobal.renderStars`, compiled once into
/// `starGLCallList`; the per-frame brightness is a [`DomeDraw::tinted`] record.
fn star_vertices() -> Vec<DomeVertex> {
    let mut out = Vec::new();
    for star in generate_stars() {
        let color = [1.0, 1.0, 1.0, 1.0];
        let v = |i: usize| DomeVertex {
            pos: star.corners[i].to_array(),
            uv: [0.0, 0.0],
            color,
        };
        out.push(v(0));
        out.push(v(1));
        out.push(v(2));
        out.push(v(0));
        out.push(v(2));
        out.push(v(3));
    }
    out
}

/// The Y every vertex of a baked plane ends up at once [`flat_verts`] has shifted it by `dy`.
/// Every vertex of a [`plane_positions`] grid shares one Y, so this is that value, computed with
/// the same single `f32` addition `flat_verts` performs per vertex.
fn plane_draw_y(quads: &[[Vec3; 4]], dy: f32) -> f32 {
    quads.first().map_or(dy, |q| q[0].y + dy)
}

/// `dome.wgsl`'s `SkyFog` uniform: the sky pass's own `GL_FOG` state.
/// java: `EntityRenderer.setupFog(-1, partialTicks)` — see [`sky_pass_fog_mode`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyFogUniform {
    /// `rgb` = fog colour, `a` = mode: `0` off, `1` linear, `2` exp.
    color: [f32; 4],
    /// `x` = linear start, `y` = linear end, `z` = exp density, `w` unused.
    params: [f32; 4],
}

impl SkyFogUniform {
    /// `GlStateManager.disableFog()` — what the sun/moon/stars/fan and the End cube draw under.
    const OFF: Self = Self {
        color: [0.0; 4],
        params: [0.0; 4],
    };

    fn new(color: Vec3, mode: FogMode) -> Self {
        match mode {
            FogMode::Linear { start, end } => Self {
                color: [color.x, color.y, color.z, 1.0],
                params: [start, end, 0.0, 0.0],
            },
            FogMode::Exp { density } => Self {
                color: [color.x, color.y, color.z, 2.0],
                params: [0.0, 0.0, density, 0.0],
            },
        }
    }
}

/// The projection used only while vanilla draws the sky. `EntityRenderer.renderWorldPass`
/// temporarily widens the far plane to `farPlaneDistance * 2.0` around `renderSky`, then restores
/// the terrain projection. Keeping this in a sky-local uniform preserves the shared camera
/// uniform for every other phase.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyProjectionUniform {
    matrix: [[f32; 4]; 4],
}

impl SkyProjectionUniform {
    fn from_camera(cam: &Camera) -> Self {
        Self {
            matrix: sky_projection(cam).to_cols_array_2d(),
        }
    }
}

/// Vanilla's undiagonalised render distance, recovered from the camera's diagonal far plane.
/// `EntityRenderer.renderWorldPass` uses the resulting `farPlaneDistance * 2.0` only for the sky
/// projection; terrain continues to use [`Camera::proj`].
#[inline]
fn render_distance_chunks(cam: &Camera) -> u32 {
    ((cam.far / std::f32::consts::SQRT_2 / 16.0).round() as u32).clamp(2, 32)
}

/// Java's temporary sky projection. The main camera is deliberately left untouched so subsequent
/// terrain, entity, and overlay phases retain the ordinary `farPlaneDistance * sqrt(2)` matrix.
#[inline]
pub(crate) fn sky_projection(cam: &Camera) -> Mat4 {
    let far_plane = render_distance_chunks(cam) as f32 * 16.0 * 2.0;
    glam::camera::rh::proj::directx::perspective(
        cam.effective_fov().to_radians(),
        cam.aspect.max(f32::EPSILON),
        cam.near,
        far_plane,
    )
}

fn push_quad(out: &mut Vec<DomeVertex>, p: [Vec3; 4], uv: [Vec2; 4], color: [f32; 4]) {
    let v = |i: usize| DomeVertex {
        pos: p[i].to_array(),
        uv: uv[i].to_array(),
        color,
    };
    out.push(v(0));
    out.push(v(1));
    out.push(v(2));
    out.push(v(0));
    out.push(v(2));
    out.push(v(3));
}

const UV_QUAD: [Vec2; 4] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
];

/// java: `renderSkyEnd`'s `tex(0,0) / tex(0,16) / tex(16,16) / tex(16,0)` — one 200-block face
/// carries 16×16 copies of `end_sky.png`.
const UV_TILE_16: [Vec2; 4] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 16.0),
    Vec2::new(16.0, 16.0),
    Vec2::new(16.0, 0.0),
];

/// `-384..=384` in 64-block cells: the sky plane / void plane grid.
/// java: `RenderGlobal.renderSky(WorldRenderer, float, boolean)`
fn plane_positions(y: f32, reverse_x: bool) -> Vec<[Vec3; 4]> {
    let mut quads = Vec::new();
    let mut x = -384i32;
    while x <= 384 {
        let mut z = -384i32;
        while z <= 384 {
            let (a, b) = if reverse_x {
                ((x + 64) as f32, x as f32)
            } else {
                (x as f32, (x + 64) as f32)
            };
            let z0 = z as f32;
            let z1 = (z + 64) as f32;
            quads.push([
                Vec3::new(a, y, z0),
                Vec3::new(b, y, z0),
                Vec3::new(b, y, z1),
                Vec3::new(a, y, z1),
            ]);
            z += 64;
        }
        x += 64;
    }
    quads
}

/// Vertices in one quad, as [`push_quad`] writes it.
const VERTICES_PER_QUAD: usize = 6;

/// Vertices in the sunrise fan: three for each of its sixteen segments.
const SUNRISE_FAN_VERTICES: usize = 16 * 3;

/// Vertices in the fullest frame [`SkyPhase`] can draw: dusk or dawn seen from below the horizon,
/// which draws every mesh below once and the void plane twice. The arguments are the lengths of
/// the fields: plane grids in quads, stars in vertices. Only the per-frame part of that is
/// uploaded any more ([`DYNAMIC_DOME_VERTICES`]); the planes and stars are baked once.
#[cfg(test)]
fn max_frame_vertices(
    sky_plane_quads: usize,
    void_plane_quads: usize,
    star_vertices: usize,
) -> usize {
    let sky_plane = sky_plane_quads * VERTICES_PER_QUAD;
    let sun_and_moon = 2 * VERTICES_PER_QUAD;
    let black_and_horizon_planes = 2 * void_plane_quads * VERTICES_PER_QUAD;
    let void_box = 5 * VERTICES_PER_QUAD;
    sky_plane
        + SUNRISE_FAN_VERTICES
        + sun_and_moon
        + star_vertices
        + black_and_horizon_planes
        + void_box
}

/// Emit a baked plane grid, shifted `dy` blocks along Y.
///
/// The shift is vanilla's `GlStateManager.translate(0, ..., 0)` around each `callList(glSkyList2)`
/// — the same ±16 plane is re-issued at three different heights in one `renderSky` call
/// (`RenderGlobal.java:1358`, `:1412`), so the grid is baked once and translated per draw rather
/// than baked per height.
fn flat_verts(quads: &[[Vec3; 4]], color: [f32; 4], dy: f32) -> Vec<DomeVertex> {
    let mut out = Vec::with_capacity(quads.len() * 6);
    for q in quads {
        push_quad(
            &mut out,
            q.map(|p| Vec3::new(p.x, p.y + dy, p.z)),
            UV_QUAD,
            color,
        );
    }
    out
}

/// The black box vanilla wraps the player in once its **eyes** drop below the horizon, so the sky
/// cannot be seen from underground. 2x2 in X/Z, open at the top, reaching down to `top`.
/// java: `RenderGlobal.renderSky` (`RenderGlobal.java:1378`-`:1399`) — `f19 = -(d0 + 65)`.
fn void_box(top: f32) -> Vec<DomeVertex> {
    let black = [0.0, 0.0, 0.0, 1.0];
    let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
    let faces: [[Vec3; 4]; 5] = [
        // +Z, -Z, +X, -X, bottom — transcribed in vanilla's own vertex order.
        [
            v(-1.0, top, 1.0),
            v(1.0, top, 1.0),
            v(1.0, -1.0, 1.0),
            v(-1.0, -1.0, 1.0),
        ],
        [
            v(-1.0, -1.0, -1.0),
            v(1.0, -1.0, -1.0),
            v(1.0, top, -1.0),
            v(-1.0, top, -1.0),
        ],
        [
            v(1.0, -1.0, -1.0),
            v(1.0, -1.0, 1.0),
            v(1.0, top, 1.0),
            v(1.0, top, -1.0),
        ],
        [
            v(-1.0, top, -1.0),
            v(-1.0, top, 1.0),
            v(-1.0, -1.0, 1.0),
            v(-1.0, -1.0, -1.0),
        ],
        [
            v(-1.0, -1.0, -1.0),
            v(-1.0, -1.0, 1.0),
            v(1.0, -1.0, 1.0),
            v(1.0, -1.0, -1.0),
        ],
    ];
    let mut out = Vec::with_capacity(30);
    for face in faces {
        push_quad(&mut out, face, UV_QUAD, black);
    }
    out
}

/// Vanilla's `RenderGlobal.renderSky` horizon coordinate, narrowed at the same point as its
/// `GlStateManager.translate` argument.
///
/// `thePlayer.getPositionEyes(partialTicks).yCoord` is the interpolated local-player eye, *not*
/// the feet and not the third-person orbit camera. Keep the subtraction in `f64`: world
/// coordinates are doubles in Java, and narrowing each operand first changes the horizon by
/// visible fractions at large Y coordinates.
#[must_use]
pub fn horizon_frame_offset(frame_origin_y: f64, horizon: f64) -> f32 {
    horizon_frame_delta(frame_origin_y, horizon) as f32
}

/// Keep the horizon comparison and the per-draw Java constant arithmetic in `f64` until the
/// exact call site narrows it.  Vanilla keeps `d0` as a `double` for both the underground test
/// and `f19 = -((float)(d0 + 65.0D))` / `translate(..., -((float)(d0 - 16.0D)), ...)`.
/// Narrowing the subtraction first can erase a tiny negative value (changing whether the void
/// box is emitted) and can move the large-coordinate casts by an ulp.
#[inline]
fn horizon_frame_delta(frame_origin_y: f64, horizon: f64) -> f64 {
    frame_origin_y - horizon
}

/// Return the three values the lower-sky branch derives from Java's `double d0`:
/// the unrounded below-horizon predicate, the void-box top, and the horizon-plane translation.
/// Each output is narrowed only where the corresponding Java call takes a `float`.
#[inline]
fn horizon_draw_offsets(frame_origin_y: f64, horizon: f64) -> (bool, f32, f32) {
    let d0 = horizon_frame_delta(frame_origin_y, horizon);
    (d0 < 0.0, -(d0 + 65.0) as f32, -((d0 - 16.0) as f32))
}

/// Sun/moon rotation. java: `RenderGlobal.renderSky:1299-1300` —
/// `rotate(-90, 0,1,0); rotate(angle*360, 1,0,0);`.
///
/// `glRotate` post-multiplies (`M <- M * R`), so the *second* call is applied to the vertex
/// **first**: the composed map is `Ry(-90) . Rx(a)`, not `Rx(a) . Ry(-90)`. Written the wrong
/// way round the sun rises in the north and sets in the south. `sunrise_fan` below already
/// transcribes its three rotations in GL order; this one did not.
fn celestial_rotation(celestial_angle: f32) -> Mat3 {
    Mat3::from_rotation_y(-std::f32::consts::FRAC_PI_2)
        * Mat3::from_rotation_x(celestial_angle * std::f32::consts::TAU)
}

fn celestial_quad(
    rot: Mat3,
    local_y: f32,
    half: f32,
    uv: [Vec2; 4],
    color: [f32; 4],
) -> Vec<DomeVertex> {
    let local = [
        Vec3::new(-half, local_y, -half),
        Vec3::new(half, local_y, -half),
        Vec3::new(half, local_y, half),
        Vec3::new(-half, local_y, half),
    ];
    let world = local.map(|p| rot * p);
    let mut out = Vec::with_capacity(6);
    push_quad(&mut out, world, uv, color);
    out
}

/// Moon texture coordinates in the exact vertex order used by [`celestial_quad`].
///
/// `RenderGlobal.renderSky` selects one cell of `moon_phases.png`'s 4×2 sheet, but writes the
/// U coordinate in reverse across its quad.  Keeping that ordering in a small pure helper makes
/// it testable without a GPU and prevents a harmless-looking "normalise the UVs" edit from
/// mirroring every crescent.
fn moon_phase_uvs(env: &Environment) -> [Vec2; 4] {
    let (u0, v0, u1, v1) = env.moon_phase_uv();
    [
        Vec2::new(u1, v0),
        Vec2::new(u0, v0),
        Vec2::new(u0, v1),
        Vec2::new(u1, v1),
    ]
}

/// The sunrise/sunset glow: a 17-vertex `GL_TRIANGLE_FAN`, apex 100 blocks out on the horizon on
/// the sun's side, rim a radius-120 ring around the eye that fades to fully transparent.
///
/// java: `RenderGlobal.renderSky` (`RenderGlobal.java:1259`-`:1290`). Vanilla's transform is
/// `rotate(90, X) · rotate(sin(celestialAngleRadians) < 0 ? 180 : 0, Z) · rotate(90, Z)`, applied
/// to an apex at `(0, 100, 0)` and rim vertices `(sin θ · 120, cos θ · 120, -cos θ · 40 · alpha)`.
/// Composing those by hand: the apex lands at `(∓100, 0, 0)` — *on the horizon*, not up with the
/// sun — and the rim becomes a ring in the XZ plane tilted up by `40 · alpha` on the sun's side.
///
/// An earlier pass of this port reconstructed the element in the sun's own rotating frame
/// (because `renderSky`'s glow block had not been read), which put the glow wherever the sun was
/// rather than pinned to the horizon. This is the line-for-line transcription.
fn sunrise_fan(flip: bool, rgba: [f32; 4]) -> Vec<DomeVertex> {
    const SEGMENTS: usize = 16;
    // rotate(90, 0,0,1) then optionally rotate(180, 0,0,1) then rotate(90, 1,0,0).
    let rot = Mat3::from_rotation_x(std::f32::consts::FRAC_PI_2)
        * Mat3::from_rotation_z(if flip { std::f32::consts::PI } else { 0.0 })
        * Mat3::from_rotation_z(std::f32::consts::FRAC_PI_2);

    let apex = rot * Vec3::new(0.0, 100.0, 0.0);
    let apex_color = rgba;
    let rim_color = [rgba[0], rgba[1], rgba[2], 0.0];
    let alpha = rgba[3];

    let rim = |l: usize| {
        // java: `MathHelper.sin`/`MathHelper.cos` over
        // `(float)l * (float)Math.PI * 2.0F / 16.0F`, not platform libm.  The table lookup is
        // observable in the rim's f32 vertex coordinates even though the fan is only scenery.
        let theta = l as f32 * std::f32::consts::PI * 2.0 / SEGMENTS as f32;
        let s = mc_math::sin(theta);
        let c = mc_math::cos(theta);
        rot * Vec3::new(s * 120.0, c * 120.0, -c * 40.0 * alpha)
    };

    let mut out = Vec::with_capacity(SEGMENTS * 3);
    for l in 0..SEGMENTS {
        out.push(DomeVertex {
            pos: apex.to_array(),
            uv: [0.0, 0.0],
            color: apex_color,
        });
        out.push(DomeVertex {
            pos: rim(l).to_array(),
            uv: [0.0, 0.0],
            color: rim_color,
        });
        out.push(DomeVertex {
            pos: rim(l + 1).to_array(),
            uv: [0.0, 0.0],
            color: rim_color,
        });
    }
    out
}

/// java: `RenderGlobal.renderSkyEnd` — a static ±100 cube, fixed vertex colour `(40,40,40,255)`,
/// textured with `end_sky.png` at UVs `0..16` (`RenderGlobal.java:1190-1193`), so the 16×16
/// tiling is what makes the End sky a dense starfield rather than one magnified blur. The
/// texture is uploaded with `clamp = false`, i.e. `GL_REPEAT`, which the dome sampler now is.
fn end_sky_cube() -> Vec<DomeVertex> {
    const R: f32 = 100.0;
    let color = [40.0 / 255.0, 40.0 / 255.0, 40.0 / 255.0, 1.0];
    // Keep the six faces in the exact order and vertex order of Java's
    // `renderSkyEnd`: it starts from the -Y face and applies the six GL rotations
    // (none, +90/-90/180 around X, then +90/-90 around Z). Spelling them out
    // prevents a later canonical-cube rewrite from silently changing texture
    // orientation.
    let faces: [[Vec3; 4]; 6] = [
        // i == 0: no rotation (the base -Y face).
        [
            Vec3::new(-R, -R, -R),
            Vec3::new(-R, -R, R),
            Vec3::new(R, -R, R),
            Vec3::new(R, -R, -R),
        ],
        // i == 1: rotate(+90, 1, 0, 0), the -Z face.
        [
            Vec3::new(-R, R, -R),
            Vec3::new(-R, -R, -R),
            Vec3::new(R, -R, -R),
            Vec3::new(R, R, -R),
        ],
        // i == 2: rotate(-90, 1, 0, 0), the +Z face.
        [
            Vec3::new(-R, -R, R),
            Vec3::new(-R, R, R),
            Vec3::new(R, R, R),
            Vec3::new(R, -R, R),
        ],
        // i == 3: rotate(180, 1, 0, 0), the +Y face.
        [
            Vec3::new(-R, R, R),
            Vec3::new(-R, R, -R),
            Vec3::new(R, R, -R),
            Vec3::new(R, R, R),
        ],
        // i == 4: rotate(+90, 0, 0, 1), the +X face.
        [
            Vec3::new(R, -R, -R),
            Vec3::new(R, -R, R),
            Vec3::new(R, R, R),
            Vec3::new(R, R, -R),
        ],
        // i == 5: rotate(-90, 0, 0, 1), the -X face.
        [
            Vec3::new(-R, R, -R),
            Vec3::new(-R, R, R),
            Vec3::new(-R, -R, R),
            Vec3::new(-R, -R, -R),
        ],
    ];
    let mut out = Vec::with_capacity(36);
    for face in faces {
        push_quad(&mut out, face, UV_TILE_16, color);
    }
    out
}

fn bind_group(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    tex: &SkyTexture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sky dome group1"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&tex.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Sky dome, sun, moon, stars, and the End's skybox. Registered at [`crate::PhaseOrder::SKY`].
///
/// Externally driven, exactly like [`crate::terrain::phase::TerrainRenderer::sky_brightness`]:
/// set [`SkyPhase::environment`] and [`SkyPhase::partial_ticks`] before each
/// [`crate::PhaseList::render`] call. `RenderPhase::record` does not receive `partial_ticks` —
/// only [`crate::pipeline::PhaseList::render`]'s caller does — so anything CPU-side that needs it
/// (the sun's rotation, the moon's phase, the sky colour) is cached on the phase itself.
pub struct SkyPhase {
    /// This frame's time/weather/dimension state.
    pub environment: Environment,
    /// This frame's render interpolation fraction, `0.0..1.0`.
    pub partial_ticks: f32,
    /// World Y of the horizon the lower sky plane is pinned to, and below which the local player
    /// gets the void box. java: `World.getHorizon()` (`World.java:3702`) — `63.0` in a normal
    /// world, `0.0` for `WorldType.FLAT`.
    ///
    /// This is **not** optional scenery: `RenderGlobal.renderSky` (`RenderGlobal.java:1411`-`:1414`)
    /// draws the lower plane *unconditionally*, translated to `-(eyeY - horizon)` so it always
    /// sits at the horizon. Leaving it out (as an earlier pass of this port did, gating it on the
    /// eye being below `0.0`) leaves everything below the horizon painted in the frame's clear
    /// colour — a hard black band across the bottom half of the sky.
    pub horizon: f64,
    /// Interpolated Y coordinate of `Minecraft.thePlayer.getPositionEyes(partialTicks)`: the
    /// local player's **eye**. `RenderGlobal.renderSky` uses this value for both the lower sky
    /// plane translation and the underground void box; it must not use [`Camera::pos`], which is
    /// an eye in first person and an orbit point in third person.
    ///
    /// The temporary `None` fallback retains the historical camera-eye behaviour for standalone
    /// render users. A live client must set this once per frame after interpolating the local
    /// player's eye (including sleeping and sneaking) and before recording the sky phase. It
    /// deliberately needs no X/Z value: every dome vertex is already camera-relative and this
    /// vanilla branch only reads Y.
    pub frame_origin_y: Option<f64>,
    /// The eye column's biome temperature (height-adjusted), for the sky plane's
    /// `BiomeGenBase.getSkyColorByTemp` tint — see [`crate::sky::fog::FogInputs::biome_temperature`]
    /// for the fog-side counterpart. Vanilla's temperate default is `0.8`.
    pub biome_temperature: f32,
    /// Camera-local fog inputs, used for the sky pass's own `GL_FOG`
    /// ([`crate::sky::fog::sky_pass_fog_mode`]) — the term that fades the sky and horizon planes
    /// into the fog colour instead of meeting in a hard seam. Drive it from the same source as
    /// [`crate::sky::fog::FogPhase::inputs`] for the medium/blindness/biome fields.
    ///
    /// `render_distance_chunks` is the exception: [`SkyPhase::record`] overwrites it every frame
    /// from [`crate::Camera::far`], because that is exactly where vanilla gets it
    /// (`EntityRenderer.setupCameraTransform` sets `farPlaneDistance = renderDistanceChunks * 16`,
    /// and `Camera::set_view_distance_chunks` sets `far` to the same thing times `sqrt(2)`). That
    /// keeps the sky's horizon fade correct with no per-frame driving at all.
    pub fog_inputs: FogInputs,
    /// Paint [`FogState::color`] over the whole target before the planes (the `backdrop`).
    ///
    /// The backdrop stands in for vanilla's `clearColor(fogColor…)`, so it is redundant whenever
    /// the host clears the frame to that same colour — which the client does every frame
    /// (`PhaseList::clear_color` from the world fog state) and, once [`Self::fog_inputs`] is driven
    /// from that state, to the bit-identical `f32` value. The client turns it off to save one
    /// full-screen fill; standalone users keep the self-contained default of `true`.
    pub paint_backdrop: bool,
    /// Rotate the star field with the celestial angle, as the sun and moon already are.
    ///
    /// java: `RenderGlobal.renderSky` draws `starGLCallList` inside the same
    /// `rotate(-90, 0,1,0); rotate(getCelestialAngle * 360, 1,0,0)` block as the sun and moon
    /// (the `popMatrix` comes after the stars), so vanilla's stars wheel across the night with
    /// the moon, and a frozen-time world shows them turned by its fixed angle. This port drew
    /// them fixed until H7 (2026-11-11) confirmed the Java source; `true`, the default, is
    /// vanilla.
    pub rotate_stars: bool,
    /// Back-face cull the planes and the void box, as vanilla does.
    ///
    /// java: `EntityRenderer.renderWorldPass` starts with `GlStateManager.enableCull()` and
    /// `RenderGlobal.renderSky` never disables it (neither does anything between the two);
    /// `glSkyList2` is built with reversed X winding (`renderSky(.., -16, true)`), so the lower
    /// plane faces up. Translated to `World.getHorizon()` it is therefore culled whenever the eye
    /// is *below* the horizon, leaving the sky plane, sun, moon and stars visible overhead.
    /// Unculled (this port's `cull_mode: None` until H7, 2026-11-11, confirmed the Java source)
    /// it was drawn last, over the whole upper sky, in the dark horizon colour whenever the eye
    /// was below y = 63. `true`, the default, is vanilla.
    pub cull_planes: bool,
    /// Resource-pack custom sky layers (the `custom_sky` mod, see [`crate::sky::custom`]),
    /// drawn between the sunrise glow and the sun; while the world has layers the vanilla stars
    /// are not drawn. `None`, the mod off, costs one check per frame.
    pub custom_sky: Option<crate::sky::custom::CustomSkyRenderer>,
    /// Skip the whole sky pass this frame. The client sets it while its Aero title screen is up,
    /// whose own background covers every pixel. `false`, the default, is today's behaviour.
    pub hidden: bool,
    /// The Sky mod; `true`, the default, is vanilla. `false`
    /// skips every vanilla sky draw: the sky plane, the sunrise/sunset glow, the sun, moon and
    /// stars, the void and horizon planes and the End's sky box. What stays is the fog colour
    /// behind everything (the frame's clear colour, or the backdrop when
    /// [`Self::paint_backdrop`]) and a Custom Sky pack's layers ([`Self::custom_sky`]), which no
    /// longer fight a vanilla sun, glow or sky plane.
    pub vanilla_sky: bool,
    /// The Sky mod's "Sun and moon" option: `false` hides only the vanilla sun and moon (a sky
    /// pack that brings its own). `true`, the default, is vanilla.
    pub vanilla_sun_and_moon: bool,
    /// The Sky mod's "Stars" option: `false` hides only the vanilla star field. `true`, the
    /// default, is vanilla.
    pub vanilla_stars: bool,

    pipeline: wgpu::RenderPipeline,
    /// The sky, void and horizon planes and the void box: unblended, untextured (`fs_plane`).
    /// Every one of those draws has vertex alpha 1 and samples the 1×1 white texel, so `fs`
    /// returns `in.color` (fogged) and `ALPHA_BLENDING` with source alpha 1 returns the source:
    /// the same pixels, without a texture fetch or a destination read over ~one full screen.
    plane_pipeline: wgpu::RenderPipeline,
    /// [`Self::plane_pipeline`] with vanilla's back-face culling, see [`Self::cull_planes`].
    plane_pipeline_culled: wgpu::RenderPipeline,
    /// java: `RenderGlobal.java:1295` — `tryBlendFuncSeparate(770, 1, 1, 0)`, i.e.
    /// `GL_SRC_ALPHA, GL_ONE`. Vanilla switches to it before the sun and leaves it on through the
    /// moon and stars. It matters: `sun.png` and `moon_phases.png` are **fully opaque** with black
    /// backgrounds, so drawing them with ordinary alpha blending paints a black square around each.
    additive_pipeline: wgpu::RenderPipeline,
    /// Full-screen fog-colour fill, standing in for vanilla's `clearColor(fogColor…)`.
    backdrop_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    fog_buffer: wgpu::Buffer,
    projection_buffer: wgpu::Buffer,
    fog_write: crate::context::ChangedWrite,
    projection_write: crate::context::ChangedWrite,
    /// The sky plane, the reversed `-16` plane and the star field, baked once at construction
    /// with white vertex colours; [`DomeDraw`] supplies each draw's colour and Y. java:
    /// `glSkyList`, `glSkyList2` and `starGLCallList`, compiled once by `RenderGlobal`.
    static_buffer: wgpu::Buffer,
    sky_plane_range: std::ops::Range<u32>,
    void_plane_range: std::ops::Range<u32>,
    star_range: std::ops::Range<u32>,
    /// The meshes that genuinely change per frame (sunrise fan, sun, moon, void box), in draw
    /// order, uploaded once per frame. At most [`DYNAMIC_DOME_VERTICES`].
    dynamic_vertices: Vec<DomeVertex>,
    dynamic_buffer: wgpu::Buffer,
    /// This frame's [`DomeDraw`] records. Slot 0 is always [`DomeDraw::PASSTHROUGH`], which is
    /// also what the buffer is created with, so the End path can use it without a write.
    draw_records: Vec<DomeDraw>,
    draw_buffer: wgpu::Buffer,
    /// The End's sky cube never changes, so it is built once.
    end_cube_buffer: Option<wgpu::Buffer>,
    fog_group: wgpu::BindGroup,
    fog_off_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    /// Java's global `GameSettings.anaglyph` upload switch for resident sun/moon/End textures.
    anaglyph: bool,

    // Kept alongside `white_group` even though it is never read directly: `wgpu` resources are
    // internally reference-counted, so this is defensive rather than load-bearing, but it keeps
    // the ownership story obvious at the call site that built `white_group` from it.
    _white: SkyTexture,
    white_group: wgpu::BindGroup,
    sun: Option<(SkyTexture, wgpu::BindGroup)>,
    moon: Option<(SkyTexture, wgpu::BindGroup)>,
    end_sky: Option<(SkyTexture, wgpu::BindGroup)>,

    sky_plane: Vec<[Vec3; 4]>,
    void_plane: Vec<[Vec3; 4]>,
    stars: Vec<DomeVertex>,
    end_cube: Vec<DomeVertex>,
}

impl SkyPhase {
    /// Build the phase. Sun/moon/End textures start unset — call [`SkyPhase::load_textures`]
    /// once a [`mc_assets::ResourceManager`] is available; until then those elements silently
    /// draw nothing rather than erroring, matching how [`crate::entity::skin`]'s cache treats an
    /// unresolved texture.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sky dome"),
                source: wgpu::ShaderSource::Wgsl(include_str!("dome.wgsl").into()),
            });

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("sky dome group1"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let fog_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sky dome group2 (fog + projection)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sky dome layout"),
                bind_group_layouts: &[
                    Some(gpu.camera_layout()),
                    Some(&bind_group_layout),
                    Some(&fog_layout),
                ],
                immediate_size: 0,
            });

        let make_pipeline = |label: &str,
                             blend: Option<wgpu::BlendState>,
                             fs: &str,
                             cull_mode: Option<wgpu::Face>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[Some(DomeVertex::LAYOUT), Some(DomeDraw::LAYOUT)],
                    },
                    primitive: wgpu::PrimitiveState {
                        // Viewed from inside the dome; only the planes carry vanilla's winding
                        // (sky plane facing down, void plane reversed), so only
                        // `plane_pipeline_culled` culls.
                        cull_mode,
                        ..wgpu::PrimitiveState::default()
                    },
                    // No depth test/write: the sky paints the background before anything else
                    // records (`PhaseOrder::SKY` is the lowest order in the list), exactly the
                    // "sky renders first/behind everything" contract described for this seam.
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };

        let pipeline = make_pipeline(
            "sky dome",
            Some(wgpu::BlendState::ALPHA_BLENDING),
            "fs",
            None,
        );
        let plane_pipeline = make_pipeline("sky dome planes", None, "fs_plane", None);
        let plane_pipeline_culled = make_pipeline(
            "sky dome planes (culled)",
            None,
            "fs_plane",
            Some(wgpu::Face::Back),
        );
        // java: `EntityRenderer.updateFogColor`'s `clearColor`, drawn as a full-screen triangle.
        let backdrop_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("sky backdrop"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_backdrop"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_backdrop"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
        // java: `RenderGlobal.java:1295` — `tryBlendFuncSeparate(770, 1, 1, 0)`.
        let additive_pipeline = make_pipeline(
            "sky dome additive",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::Zero,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            "fs",
            None,
        );

        let fog_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sky dome fog"),
                contents: bytemuck::bytes_of(&SkyFogUniform::OFF),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let fog_off_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sky dome fog off"),
                contents: bytemuck::bytes_of(&SkyFogUniform::OFF),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let projection_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sky dome projection"),
                contents: bytemuck::bytes_of(&SkyProjectionUniform {
                    matrix: Mat4::IDENTITY.to_cols_array_2d(),
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let fog_bind = |buffer: &wgpu::Buffer, label: &str| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &fog_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: projection_buffer.as_entire_binding(),
                    },
                ],
            })
        };
        let fog_group = fog_bind(&fog_buffer, "sky dome fog group");
        let fog_off_group = fog_bind(&fog_off_buffer, "sky dome fog-off group");

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky dome sampler"),
            // java: `SimpleTexture.loadTexture` passes `blur = false` unless a `.mcmeta` asks
            // for blur, and vanilla ships none for `textures/environment/**`, so
            // `TextureUtil.setTextureBlurMipmap(false, false)` selects `GL_NEAREST` for both
            // filters. At 1080p and a 70 degree FOV each texel of the 32x32 sun covers about
            // 14 screen pixels, so linear filtering here turned its hard edges into a blur.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            // java: every sky texture goes through `SimpleTexture.loadTexture` ->
            // `TextureUtil.uploadTextureImageAllocate(..., blur, clamp)` with `clamp = false`
            // unless a `.mcmeta` says otherwise, i.e. `GL_REPEAT`. `end_sky.png` *needs* it
            // (its UVs run 0..16); sun and moon stay inside 0..1 either way.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let white = textures::white_pixel(gpu, "sky dome white");
        let white_group = bind_group(gpu, &bind_group_layout, &white, &sampler);

        let sky_plane = plane_positions(16.0, false);
        let void_plane = plane_positions(-16.0, true);
        let stars = star_vertices();
        // Baked once, untranslated and white: every per-draw colour and Y comes from `DomeDraw`.
        let baked_rgba = [1.0, 1.0, 1.0, 1.0];
        let mut baked = flat_verts(&sky_plane, baked_rgba, 0.0);
        let sky_plane_range = 0..baked.len() as u32;
        baked.extend(flat_verts(&void_plane, baked_rgba, 0.0));
        let void_plane_range = sky_plane_range.end..baked.len() as u32;
        baked.extend_from_slice(&stars);
        let star_range = void_plane_range.end..baked.len() as u32;
        let static_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sky dome static vertices"),
                contents: bytemuck::cast_slice(&baked),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let dynamic_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky dome frame vertices"),
            size: (DYNAMIC_DOME_VERTICES * std::mem::size_of::<DomeVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sky dome draw records"),
                contents: bytemuck::cast_slice(&[DomeDraw::PASSTHROUGH; MAX_DOME_DRAWS]),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            environment: Environment::default(),
            partial_ticks: 0.0,
            // java: `World.getHorizon()` (`World.java:3702`) — sea level, not `0.0`.
            horizon: 63.0,
            frame_origin_y: None,
            biome_temperature: 0.8,
            fog_inputs: FogInputs::default(),
            paint_backdrop: true,
            // java: both vanilla (see the field docs).
            rotate_stars: true,
            cull_planes: true,
            custom_sky: None,
            hidden: false,
            // Vanilla
            vanilla_sky: true,
            vanilla_sun_and_moon: true,
            vanilla_stars: true,
            pipeline,
            additive_pipeline,
            plane_pipeline,
            plane_pipeline_culled,
            backdrop_pipeline,
            bind_group_layout,
            fog_buffer,
            projection_buffer,
            fog_write: crate::context::ChangedWrite::default(),
            projection_write: crate::context::ChangedWrite::default(),
            static_buffer,
            sky_plane_range,
            void_plane_range,
            star_range,
            dynamic_vertices: Vec::with_capacity(DYNAMIC_DOME_VERTICES),
            dynamic_buffer,
            draw_records: Vec::with_capacity(MAX_DOME_DRAWS),
            draw_buffer,
            end_cube_buffer: None,
            fog_group,
            fog_off_group,
            sampler,
            anaglyph: false,
            _white: white,
            white_group,
            sun: None,
            moon: None,
            end_sky: None,
            sky_plane,
            void_plane,
            stars,
            end_cube: end_sky_cube(),
        }
    }

    /// Build the phase with Java's global upload-time anaglyph switch selected.
    ///
    /// The flag is retained before any resource-pack textures are loaded, so the first upload
    /// already has the requested colour transform. Use [`Self::set_anaglyph_with_gpu`] when
    /// changing an installed phase live.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let mut phase = Self::new(gpu);
        phase.anaglyph = anaglyph;
        phase
    }

    /// Load `sun.png`, `moon_phases.png`, `end_sky.png` from a resource manager. Any texture
    /// that fails to load is left unset (that element simply does not draw), rather than
    /// failing the whole phase — the same "degrade, don't abort" policy S09's atlas bake uses
    /// for a single missing sprite.
    pub fn load_textures(&mut self, gpu: &Gpu, rm: &mc_assets::ResourceManager) {
        if let Ok(tex) = textures::load_with_anaglyph(
            gpu,
            rm,
            "textures/environment/sun.png",
            "sun",
            self.anaglyph,
        ) {
            let group = bind_group(gpu, &self.bind_group_layout, &tex, &self.sampler);
            self.sun = Some((tex, group));
        }
        if let Ok(tex) = textures::load_with_anaglyph(
            gpu,
            rm,
            "textures/environment/moon_phases.png",
            "moon",
            self.anaglyph,
        ) {
            let group = bind_group(gpu, &self.bind_group_layout, &tex, &self.sampler);
            self.moon = Some((tex, group));
        }
        if let Ok(tex) = textures::load_with_anaglyph(
            gpu,
            rm,
            "textures/environment/end_sky.png",
            "end_sky",
            self.anaglyph,
        ) {
            let group = bind_group(gpu, &self.bind_group_layout, &tex, &self.sampler);
            self.end_sky = Some((tex, group));
        }
    }

    /// Whether Java's upload-time anaglyph conversion is selected for sky textures.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global transform and replay every resident sun/moon/End texture from its
    /// retained decoded source bytes.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;
        for (texture, _) in [&self.sun, &self.moon, &self.end_sky].into_iter().flatten() {
            texture.set_anaglyph(gpu, enabled);
        }
    }

    /// Set the transform for subsequent resource uploads without touching resident GPU
    /// textures. Use [`Self::set_anaglyph_with_gpu`] for a live setting change.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Supply the interpolated local player's eye for this frame's horizon/void transform.
    ///
    /// This is intentionally separate from [`Camera::pos`]: calling code can retain the true
    /// eye/orbit camera for culling and projection while preserving vanilla's local-player-eye
    /// sky origin.
    pub fn set_frame_origin_y(&mut self, eye_y: f64) {
        self.frame_origin_y = Some(eye_y);
    }
}

impl RenderPhase for SkyPhase {
    fn name(&self) -> &'static str {
        "sky/dome"
    }

    // One `bool`; always true unless the client hides the sky.
    // With the Sky mod off and nothing else to draw (no custom sky layers, the
    // client's clear colour standing in for the backdrop) the pass is skipped outright. With the
    // vanilla sky on, the first test short-circuits: the same one `bool` as before.
    fn enabled(&self) -> bool {
        !self.hidden && (self.vanilla_sky || self.custom_sky.is_some() || self.paint_backdrop)
    }

    // `record` makes its queue writes, then opens its one pass on the frame's
    // colour target with `Gpu::begin_pass` before drawing anything, and never touches the shared
    // depth buffer: that pass can clear the colour, and the depth clear can wait for a later pass.
    fn frame_clear_handoff(&self) -> crate::pipeline::FrameClearHandoff {
        crate::pipeline::FrameClearHandoff::ColorAndDepth
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        mc_core::profile_scope!("sky.dome.record");
        let env = self.environment;
        let dim = env.dimension;
        let pt = self.partial_ticks;

        // java: `EntityRenderer.updateFogColor` for the colour, `setupFog(-1, ..)` for the
        // falloff. Written before the pass opens so the whole submission sees one value.
        //
        // `farPlaneDistance` comes straight off the camera rather than from a driver-supplied
        // render distance — see `SkyPhase::fog_inputs`.
        let mut fog_inputs = self.fog_inputs;
        fog_inputs.render_distance_chunks = render_distance_chunks(cam);
        let fog_color = FogState::compute(&env, fog_inputs, pt).color;
        let fog = SkyFogUniform::new(fog_color, sky_pass_fog_mode(fog_inputs));
        // Both buffers are read only by this phase's pass.
        self.fog_write
            .write_for_frame(gpu, &self.fog_buffer, bytemuck::bytes_of(&fog));
        self.projection_write.write_for_frame(
            gpu,
            &self.projection_buffer,
            bytemuck::bytes_of(&SkyProjectionUniform::from_camera(cam)),
        );

        let mut pass = gpu.begin_pass(enc, view, "sky/dome", DepthAttachment::None);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &self.white_group, &[]);
        pass.set_pipeline(&self.pipeline);
        // java: `RenderGlobal.renderSky` opens with fog disabled and only turns it on around the
        // planes, so "off" is the right default for everything else in this pass.
        pass.set_bind_group(2, &self.fog_off_group, &[]);
        // Per-draw colour/Y records; instance 0 is the passthrough record every draw of a
        // per-frame mesh (and the End cube) uses.
        pass.set_vertex_buffer(1, self.draw_buffer.slice(..));

        // The Sky mod: which vanilla parts draw this frame (all of them by default).
        let vanilla_sky = self.vanilla_sky;
        let vanilla_sun_and_moon = vanilla_sky && self.vanilla_sun_and_moon;
        let vanilla_stars = vanilla_sky && self.vanilla_stars;
        // java: `EntityRenderer.renderWorldPass` (`:1346`) calls `RenderGlobal.renderSky` only
        // when `renderDistanceChunks >= 4`. At 2 and 3 chunks the frame keeps the fog-coloured
        // clear `updateFogColor` set (the backdrop's colour, below) and nothing else: no planes,
        // glow, sun, moon, stars, End box, or Custom Sky layers (the format draws those inside
        // `renderSky` too).
        let sky_drawn = render_distance_chunks(cam) >= 4;
        if !sky_drawn && dim != Dimension::Overworld {
            return;
        }

        if dim == Dimension::End {
            if let Some((_, group)) = self.end_sky.as_ref().filter(|_| vanilla_sky) {
                let end_cube = &self.end_cube;
                let vbuf = self.end_cube_buffer.get_or_insert_with(|| {
                    gpu.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("sky end cube"),
                            contents: bytemuck::cast_slice(end_cube),
                            usage: wgpu::BufferUsages::VERTEX,
                        })
                });
                pass.set_bind_group(1, group, &[]);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.draw(0..self.end_cube.len() as u32, 0..1);
            }
            // `world1` layers, over the End's sky box.
            if let Some(custom) = self.custom_sky.as_mut() {
                custom.draw(gpu, &mut pass, &env, pt, sky_projection(cam) * cam.view);
            }
            return;
        }

        if !dim.has_sky_dome() {
            // The Nether: `RenderGlobal.renderSky` skips its whole `isSurfaceWorld()` branch —
            // no geometry at all. The clear colour / `post::overlay` fog tint carries the look.
            return;
        }

        // The planes and the star field are baked into `static_buffer` once; each draw of them
        // takes its colour and Y from a `DomeDraw` record instead of a rebuilt copy of the mesh.
        // Only the fan, sun, moon and void box are rebuilt, into `dynamic_buffer`. Both small
        // writes happen after the pass is recorded: queue writes run before the submitted
        // encoder, so the draws see this frame's data.
        let mut dynamic = std::mem::take(&mut self.dynamic_vertices);
        dynamic.clear();
        let mut draws = std::mem::take(&mut self.draw_records);
        draws.clear();
        draws.push(DomeDraw::PASSTHROUGH);
        let static_buffer = &self.static_buffer;
        let dynamic_buffer = &self.dynamic_buffer;
        let mut draw = |pass: &mut wgpu::RenderPass<'_>, _label: &'static str, v: &[DomeVertex]| {
            if v.is_empty() {
                return;
            }
            let start = dynamic.len() as u32;
            dynamic.extend_from_slice(v);
            pass.set_vertex_buffer(0, dynamic_buffer.slice(..));
            pass.draw(start..dynamic.len() as u32, 0..1);
        };
        let mut draw_baked =
            |pass: &mut wgpu::RenderPass<'_>, range: std::ops::Range<u32>, record: DomeDraw| {
                if range.is_empty() {
                    return;
                }
                let instance = draws.len() as u32;
                draws.push(record);
                pass.set_vertex_buffer(0, static_buffer.slice(..));
                pass.draw(range, instance..instance + 1);
            };

        // java: `EntityRenderer.updateFogColor`'s `clearColor(fogColor…)`, as geometry. It fills
        // the seam the two sky planes leave at the horizon: the upper plane's rim sits ~2.4°
        // *above* the horizon and the lower plane's rim just below it, and in vanilla it is the
        // clear colour showing through that gap. Painting it here rather than depending on the
        // caller's `PhaseList::clear_color` keeps the sky self-contained; a host that clears to
        // the same fog colour turns it off with `paint_backdrop`.
        //
        // Overworld only: `Environment::base_fog_color` is verified against
        // `WorldProviderHell`/`WorldProviderEnd` for the *colour*, but `renderSkyEnd` disables fog
        // outright and the Nether draws no sky at all, so neither is repainted here — those two
        // keep relying on the caller's clear colour, exactly as before this change.
        pass.set_bind_group(2, &self.fog_group, &[]);
        if self.paint_backdrop {
            pass.set_pipeline(&self.backdrop_pipeline);
            pass.draw(0..3, 0..1);
        }
        if !sky_drawn {
            return;
        }
        let plane_pipeline = if self.cull_planes {
            &self.plane_pipeline_culled
        } else {
            &self.plane_pipeline
        };
        pass.set_pipeline(plane_pipeline);

        // Sky plane, flat sky colour, **fogged** (`RenderGlobal.java:1231` `enableFog()`).
        let sky_color = env.sky_color(self.biome_temperature, pt);
        if vanilla_sky {
            draw_baked(
                &mut pass,
                self.sky_plane_range.clone(),
                DomeDraw::plane(
                    [sky_color.x, sky_color.y, sky_color.z, 1.0],
                    plane_draw_y(&self.sky_plane, 0.0),
                ),
            );
        }

        // java: `RenderGlobal.java:1248` `disableFog()` — the fan, sun, moon and stars are all
        // drawn un-fogged, and (from `:1295`) additively.
        pass.set_bind_group(2, &self.fog_off_group, &[]);
        pass.set_pipeline(&self.pipeline);

        let angle = env.celestial_angle(pt);
        let rot = celestial_rotation(angle);

        // Sunrise/sunset glow. Still ordinary alpha blending
        // (`RenderGlobal.java:1251` — `tryBlendFuncSeparate(770, 771, 1, 0)`).
        if let Some(rgba) = env.sunrise_sunset_color(pt).filter(|_| vanilla_sky) {
            // java: `RenderGlobal.java:1261` — `sin(getCelestialAngleRadians()) < 0` flips the
            // glow to the opposite horizon, which is what makes it sunrise in the east and
            // sunset in the west rather than always the same side.
            let flip = mc_math::sin(angle * std::f32::consts::TAU) < 0.0;
            draw(&mut pass, "sunrise fan", &sunrise_fan(flip, rgba));
        }

        // Custom sky layers are drawn here: after the sunrise glow,
        // before the sun, moon and stars (`RenderGlobal.renderSky`, right after
        // `rotate(-90, 0,1,0)`). A drawing renderer leaves its own pipeline and groups bound.
        if let Some(custom) = self.custom_sky.as_mut() {
            if custom.draw(gpu, &mut pass, &env, pt, sky_projection(cam) * cam.view) {
                pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
                pass.set_bind_group(2, &self.fog_off_group, &[]);
            }
        }

        pass.set_pipeline(&self.additive_pipeline);
        // java: `RenderGlobal.java:1297` — `f16` dims sun *and* moon in rain, not just the sun.
        let celestial_alpha = 1.0 - env.weather.rain_strength;

        // Sun.
        if let Some((_, group)) = self.sun.as_ref().filter(|_| vanilla_sun_and_moon) {
            pass.set_bind_group(1, group, &[]);
            draw(
                &mut pass,
                "sun quad",
                &celestial_quad(rot, 100.0, 30.0, UV_QUAD, [1.0, 1.0, 1.0, celestial_alpha]),
            );
        }

        // Moon, with the current phase's UV rect.
        if let Some((_, group)) = self.moon.as_ref().filter(|_| vanilla_sun_and_moon) {
            // java: `RenderGlobal.java:1318-1321`, read back onto `celestial_quad`'s corner
            // order `(-h,-h), (+h,-h), (+h,+h), (-h,+h)`:
            //   (-h, -h) -> (u1, v0)   (+h, -h) -> (u0, v0)
            //   (+h, +h) -> (u0, v1)   (-h, +h) -> (u1, v1)
            // i.e. u runs *backwards* across the quad. Mirroring it (which is what this used to
            // do) flips every crescent phase left-to-right, and v as well turned the moon
            // upside down.
            let uv = moon_phase_uvs(&env);
            pass.set_bind_group(1, group, &[]);
            draw(
                &mut pass,
                "moon quad",
                &celestial_quad(rot, -100.0, 20.0, uv, [1.0, 1.0, 1.0, celestial_alpha]),
            );
        }

        // Stars. java: `RenderGlobal.java:1325` — `getStarBrightness * (1 - rainStrength)`, as
        // `GlStateManager.color(f15, f15, f15, f15)` around the baked star list.
        let star_brightness = env.star_brightness(pt) * celestial_alpha;
        // As the format specifies, a world with custom layers has no
        // vanilla stars (sky packs draw their own).
        if vanilla_stars
            && star_brightness > 0.0
            && !self.stars.is_empty()
            && !self
                .custom_sky
                .as_ref()
                .is_some_and(|custom| custom.has_layers(dim))
        {
            pass.set_bind_group(1, &self.white_group, &[]);
            let mut record = DomeDraw::tinted([star_brightness; 4]);
            if self.rotate_stars {
                record = record.celestially_rotated(angle);
            }
            draw_baked(&mut pass, self.star_range.clone(), record);
        }

        // java: `RenderGlobal.java:1347`-`:1414`. Blending off, fog back **on**, and two more
        // draws of the same reversed `-16` plane. This is the block whose absence left the whole
        // lower half of the sky painted in the frame's clear colour. All three draws are opaque
        // (alpha 1, white texel), hence `plane_pipeline`.
        pass.set_pipeline(plane_pipeline);
        pass.set_bind_group(1, &self.white_group, &[]);
        pass.set_bind_group(2, &self.fog_group, &[]);

        // java: `RenderGlobal.java:1353` — the local player's interpolated *eye*
        // (`thePlayer.getPositionEyes(partialTicks).yCoord`) minus `getHorizon()`. Do not use the
        // orbit camera here: F5 moves the void box, if we do.
        //
        // `None` is a compatibility fallback for direct `mc-render` consumers. The client
        // driver must call `set_frame_origin_y` every live frame; see that method's contract.
        let (below_horizon, void_top, horizon_translation) =
            horizon_draw_offsets(self.frame_origin_y.unwrap_or(cam.pos.y), self.horizon);

        // The void and horizon planes are vanilla sky too: with the Sky mod off a
        // custom sky's lower faces show, as the pack drew them, instead of a dark-blue band.
        if vanilla_sky && below_horizon {
            // Underground/below sea level: a black plane 12 above the baked `-16` one, plus the
            // black box that seals off everything below. java: `RenderGlobal.java:1355`-`:1400`.
            draw_baked(
                &mut pass,
                self.void_plane_range.clone(),
                DomeDraw::plane([0.0, 0.0, 0.0, 1.0], plane_draw_y(&self.void_plane, 12.0)),
            );
            // java: `f19 = -((float)(d0 + 65.0D))`; keep the addition in double precision.
            draw(&mut pass, "void box", &void_box(void_top));
        }

        // The horizon plane. java: `RenderGlobal.java:1402`-`:1414` — `isSkyColored()` is true in
        // the Overworld, so the colour is the sky colour pushed toward a dark blue, and the
        // translate puts the plane exactly at `World.getHorizon()`.
        let horizon_color = [
            sky_color.x * 0.2 + 0.04,
            sky_color.y * 0.2 + 0.04,
            sky_color.z * 0.6 + 0.1,
            1.0,
        ];
        if vanilla_sky {
            draw_baked(
                &mut pass,
                self.void_plane_range.clone(),
                // java: `GlStateManager.translate(0.0F, -((float)(d0 - 16.0D)), 0.0F)`.
                DomeDraw::plane(
                    horizon_color,
                    plane_draw_y(&self.void_plane, horizon_translation),
                ),
            );
        }
        drop(pass);
        debug_assert!(dynamic.len() <= DYNAMIC_DOME_VERTICES);
        debug_assert!(draws.len() <= MAX_DOME_DRAWS);
        if !dynamic.is_empty() {
            gpu.write_buffer_staged(&self.dynamic_buffer, 0, bytemuck::cast_slice(&dynamic));
        }
        gpu.write_buffer_staged(&self.draw_buffer, 0, bytemuck::cast_slice(&draws));
        self.dynamic_vertices = dynamic;
        self.draw_records = draws;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// `cargo check` never parses WGSL; this does, with the same naga front end and validator
    /// wgpu runs at pipeline creation, so a shader mistake fails a unit test instead of a
    /// player's first frame.
    #[test]
    fn dome_wgsl_parses_and_validates() {
        let source = include_str!("dome.wgsl");
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("dome.wgsl validates");
    }

    #[test]
    fn dome_resource_owner_routes_all_decoded_textures_through_anaglyph_uploads() {
        let source = include_str!("dome.rs");
        let loader = concat!("textures::load_with_anaglyph", "(");
        assert_eq!(
            source.matches(loader).count(),
            3,
            "sun, moon, and End sky must use the anaglyph-aware loader"
        );
        for path in [
            "textures/environment/sun.png",
            "textures/environment/moon_phases.png",
            "textures/environment/end_sky.png",
        ] {
            let owner_call = source
                .split_once(path)
                .map(|(_, rest)| rest)
                .expect("dome resource path must have an owner call");
            assert!(
                owner_call.contains("self.anaglyph"),
                "dome resource owner lost anaglyph state for {path}"
            );
        }
        let setter = concat!("texture", ".set_anaglyph(gpu, enabled)");
        assert!(
            source.contains(setter),
            "live dome toggles must replay every resident source texture"
        );
    }

    #[test]
    fn frame_buffer_bound_is_exactly_the_fullest_frame() {
        let color = [1.0, 1.0, 1.0, 1.0];
        let sky_plane = plane_positions(16.0, false);
        let void_plane = plane_positions(-16.0, true);
        let star_vertices = 9_000;
        let rot = celestial_rotation(0.0);
        let fullest = flat_verts(&sky_plane, color, 0.0).len()
            + sunrise_fan(false, color).len()
            + celestial_quad(rot, 100.0, 30.0, UV_QUAD, color).len()
            + celestial_quad(rot, -100.0, 20.0, UV_QUAD, color).len()
            + star_vertices
            + flat_verts(&void_plane, color, 12.0).len()
            + void_box(-1.0).len()
            + flat_verts(&void_plane, color, 0.0).len();
        assert_eq!(
            max_frame_vertices(sky_plane.len(), void_plane.len(), star_vertices),
            fullest
        );
    }

    #[test]
    fn plane_grid_covers_the_expected_span() {
        let quads = plane_positions(16.0, false);
        // -384..=384 in steps of 64 is 13 positions per axis -> 169 quads.
        assert_eq!(quads.len(), 169);
        for q in &quads {
            for v in q {
                assert_eq!(v.y, 16.0);
                assert!(v.x >= -384.0 && v.x <= 448.0);
            }
        }
    }

    #[test]
    fn reverse_x_swaps_winding_edges() {
        let normal = plane_positions(16.0, false)[0];
        let reversed = plane_positions(16.0, true)[0];
        assert_eq!(normal[0].x, -384.0);
        assert_eq!(reversed[0].x, -384.0 + 64.0);
    }

    #[test]
    fn celestial_rotation_places_sun_and_moon_on_opposite_sides() {
        let rot = celestial_rotation(0.25); // noon
        let sun = rot * Vec3::new(0.0, 100.0, 0.0);
        let moon = rot * Vec3::new(0.0, -100.0, 0.0);
        assert!((sun + moon).length() < 1e-3, "{sun:?} {moon:?}");
        assert!((sun.length() - 100.0).abs() < 1e-3);
    }

    #[test]
    fn moon_quad_uses_the_live_eight_phase_cell_without_mirroring_it() {
        // Day 6 is the bottom-row third cell.  This catches both feeding a time-of-day-only
        // value (which would select phase zero) and putting U in ordinary left-to-right order.
        let env = Environment {
            world_time: 6 * 24_000 + 12_345,
            ..Environment::default()
        };
        assert_eq!(env.moon_phase(), 6);
        assert_eq!(
            moon_phase_uvs(&env),
            [
                Vec2::new(0.75, 0.5),
                Vec2::new(0.5, 0.5),
                Vec2::new(0.5, 1.0),
                Vec2::new(0.75, 1.0),
            ]
        );
    }

    #[test]
    fn end_cube_spans_plus_minus_100() {
        let cube = end_sky_cube();
        assert_eq!(cube.len(), 36);
        for v in &cube {
            assert!(v.pos.iter().all(|c| c.abs() <= 100.0 + 1e-3));
            assert_eq!(v.color, [40.0 / 255.0, 40.0 / 255.0, 40.0 / 255.0, 1.0]);
        }
        // java: `tex(0,0) / tex(0,16) / tex(16,16) / tex(16,0)` — the texture tiles 16x16 per
        // face. Mapped 0..1 (what this did before) the End sky is one stretched copy.
        assert!(
            cube.iter().any(|v| v.uv == [16.0, 16.0]),
            "end sky must be tiled 16x, not stretched 1x"
        );
        for v in &cube {
            assert!(v.uv.iter().all(|c| *c == 0.0 || *c == 16.0), "{:?}", v.uv);
        }
    }

    #[test]
    fn end_cube_preserves_java_face_rotation_and_uv_order() {
        // Java starts each face from the same -Y quad, applies the six GL rotations in loop
        // order, and emits tex(0,0), tex(0,16), tex(16,16), tex(16,0). Comparing the expanded
        // triangle-list vertices catches both a canonical-cube face reorder and the tempting
        // left-to-right UV rewrite. Either mutant leaves a six-face cube that still spans 100.
        let cube = end_sky_cube();
        let r = 100.0_f32;
        let faces = [
            [
                Vec3::new(-r, -r, -r),
                Vec3::new(-r, -r, r),
                Vec3::new(r, -r, r),
                Vec3::new(r, -r, -r),
            ],
            [
                Vec3::new(-r, r, -r),
                Vec3::new(-r, -r, -r),
                Vec3::new(r, -r, -r),
                Vec3::new(r, r, -r),
            ],
            [
                Vec3::new(-r, -r, r),
                Vec3::new(-r, r, r),
                Vec3::new(r, r, r),
                Vec3::new(r, -r, r),
            ],
            [
                Vec3::new(-r, r, r),
                Vec3::new(-r, r, -r),
                Vec3::new(r, r, -r),
                Vec3::new(r, r, r),
            ],
            [
                Vec3::new(r, -r, -r),
                Vec3::new(r, -r, r),
                Vec3::new(r, r, r),
                Vec3::new(r, r, -r),
            ],
            [
                Vec3::new(-r, r, -r),
                Vec3::new(-r, r, r),
                Vec3::new(-r, -r, r),
                Vec3::new(-r, -r, -r),
            ],
        ];
        let uv = [[0.0_f32, 0.0], [0.0, 16.0], [16.0, 16.0], [16.0, 0.0]];
        for (face_index, expected) in faces.into_iter().enumerate() {
            let actual = &cube[face_index * 6..(face_index + 1) * 6];
            let expected_indices = [0, 1, 2, 0, 2, 3];
            let expected_uv_indices = [0, 1, 2, 0, 2, 3];
            for ((vertex, &position_index), &uv_index) in actual
                .iter()
                .zip(expected_indices.iter())
                .zip(expected_uv_indices.iter())
            {
                assert_eq!(vertex.pos, expected[position_index].to_array());
                assert_eq!(vertex.uv, uv[uv_index]);
            }
        }
    }

    #[test]
    fn sunrise_fan_fades_to_transparent_at_the_rim() {
        let fan = sunrise_fan(false, [1.0, 0.5, 0.2, 0.8]);
        // Every triangle is apex, rim, rim -> indices 0,3,6.. are apex vertices.
        assert_eq!(fan[0].color[3], 0.8);
        assert_eq!(fan[1].color[3], 0.0);
        assert_eq!(fan[2].color[3], 0.0);
    }

    #[test]
    fn sunrise_fan_apex_sits_on_the_horizon_not_up_with_the_sun() {
        // java: `RenderGlobal.java:1260`-`:1262` composed by hand — `Rx(90)·Rz(0|180)·Rz(90)`
        // maps the apex `(0, 100, 0)` to `(∓100, 0, 0)`. The pre-fix port put it wherever the
        // sun was, which is what a "the glow follows the sun into the sky" bug looks like.
        for (flip, expected_x) in [(false, -100.0_f32), (true, 100.0)] {
            let fan = sunrise_fan(flip, [1.0, 0.5, 0.2, 0.8]);
            let apex = Vec3::from_array(fan[0].pos);
            assert!(
                (apex.x - expected_x).abs() < 1e-3 && apex.y.abs() < 1e-3 && apex.z.abs() < 1e-3,
                "flip={flip} apex={apex:?}"
            );
        }
    }

    #[test]
    fn sunrise_fan_rim_is_a_radius_120_ring_around_the_eye() {
        let fan = sunrise_fan(false, [1.0, 0.5, 0.2, 1.0]);
        for tri in fan.chunks(3) {
            for rim in &tri[1..] {
                let p = Vec3::from_array(rim.pos);
                // Radius 120 in XZ; Y carries only the `-cos θ · 40 · alpha` tilt.
                let r = (p.x * p.x + p.z * p.z).sqrt();
                assert!((r - 120.0).abs() < 1e-2, "{p:?} r={r}");
                assert!(p.y.abs() <= 40.0 + 1e-3, "{p:?}");
            }
        }
    }

    #[test]
    fn flat_verts_translates_the_baked_plane() {
        let quads = plane_positions(-16.0, true);
        let at_zero = flat_verts(&quads, [1.0; 4], 0.0);
        let shifted = flat_verts(&quads, [1.0; 4], 12.0);
        assert_eq!(at_zero[0].pos[1], -16.0);
        assert_eq!(shifted[0].pos[1], -4.0);
    }

    #[test]
    fn horizon_frame_offset_uses_java_player_eye_not_feet_or_f5_orbit() {
        // Java passes the local player's `getPositionEyes` Y, so a player whose feet are on the
        // normal-world horizon contributes its 1.62-block eye height to d0.
        let feet_y = 63.0;
        let first_person_eye_y = feet_y + 1.62;
        assert_eq!(horizon_frame_offset(feet_y, 63.0), 0.0);
        assert_eq!(horizon_frame_offset(first_person_eye_y, 63.0), 1.62);

        // F5 moves the projection camera through the orbit/raytrace path, while the Java sky
        // branch continues to consume the local player's eye rather than that orbit point.
        let third_person_orbit_y = first_person_eye_y + 2.75;
        assert_eq!(horizon_frame_offset(first_person_eye_y, 63.0), 1.62);
        assert_eq!(horizon_frame_offset(third_person_orbit_y, 63.0), 4.37);

        // Sleeping changes the camera arrangement, but `getPositionEyes` still adds the current
        // 0.2F eye height to the interpolated local-player position.
        let sleeping_feet_y = 65.6875;
        let sleeping_eye_y = sleeping_feet_y + 0.2;
        assert_eq!(horizon_frame_offset(sleeping_feet_y, 63.0), 2.6875);
        assert_eq!(horizon_frame_offset(sleeping_eye_y, 63.0), 2.8875);
        assert_eq!(horizon_frame_offset(63.0, 63.0), 0.0);
    }

    #[test]
    fn sky_projection_doubles_java_far_plane_without_mutating_terrain_projection() {
        let mut cam = Camera::default();
        cam.set_view_distance_chunks(8);
        cam.update();
        let terrain_projection = cam.proj;
        let sky = SkyProjectionUniform::from_camera(&cam);
        let expected = glam::camera::rh::proj::directx::perspective(
            cam.effective_fov().to_radians(),
            cam.aspect,
            cam.near,
            256.0,
        );

        assert_eq!(sky.matrix, expected.to_cols_array_2d());
        assert_ne!(sky.matrix, terrain_projection.to_cols_array_2d());
        assert_eq!(cam.proj, terrain_projection);
        assert_eq!(render_distance_chunks(&cam), 8);
    }

    #[test]
    fn horizon_frame_offset_subtracts_in_double_before_narrowing() {
        // At this magnitude each operand loses its low bits as f32. Java's `double d0` retains
        // the 1.25-block difference until the subsequent GL float argument, as must we.
        assert_eq!(horizon_frame_offset(16_777_217.25, 16_777_216.0), 1.25);
    }

    #[test]
    fn horizon_draw_offsets_preserve_java_double_predicate_and_cast_sites() {
        // A negative f64 subnormal is still below the horizon in Java. Casting d0 first would
        // turn it into -0.0f, making the old `d0 < 0` gate incorrectly skip the void geometry.
        let tiny_negative = f64::from_bits(1);
        let (below, _, _) = horizon_draw_offsets(-tiny_negative, 0.0);
        assert!(below);

        // Java adds its constants while d0 is a double, then narrows each result independently.
        // At this magnitude, narrowing d0 before adding 65 or subtracting 16 changes the f32
        // result by a full representable step.
        let frame_origin = 16_777_217.25;
        let (below, void_top, horizon_translation) = horizon_draw_offsets(frame_origin, 0.0);
        assert!(!below);
        assert_eq!(void_top, (-(frame_origin + 65.0)) as f32);
        assert_eq!(horizon_translation, -((frame_origin - 16.0) as f32));
        assert_ne!(
            void_top,
            -((frame_origin as f32) + 65.0),
            "the d0-first narrowing mutant must be distinguishable"
        );
    }

    #[test]
    fn sunrise_fan_uses_the_vanilla_math_helper_table_for_rim_trig() {
        // l=1 is deliberately not a cardinal angle: the platform libm result and Java's
        // MathHelper SIN_TABLE result differ in their low f32 bits.
        let theta = std::f32::consts::PI * 2.0 / 16.0;
        let fan = sunrise_fan(false, [1.0, 1.0, 1.0, 1.0]);
        let rim = Vec3::from_array(fan[2].pos);
        let expected_cos = mc_math::cos(theta);
        let expected_sin = mc_math::sin(theta);
        let rot = Mat3::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat3::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let expected = rot
            * Vec3::new(
                expected_sin * 120.0,
                expected_cos * 120.0,
                -expected_cos * 40.0,
            );
        assert_eq!(
            rim.to_array().map(f32::to_bits),
            expected.to_array().map(f32::to_bits)
        );
    }

    #[test]
    fn void_box_is_five_quads_open_at_the_top() {
        // java: `RenderGlobal.java:1378`-`:1399` — 20 vertices as `GL_QUADS` = 5 faces = 30
        // triangle-list vertices here. Open at the top: no face at `y == top` spanning X/Z.
        let verts = void_box(-64.0);
        assert_eq!(verts.len(), 30);
        for v in &verts {
            assert!(v.pos[0].abs() <= 1.0 && v.pos[2].abs() <= 1.0, "{v:?}");
            assert_eq!(v.color, [0.0, 0.0, 0.0, 1.0]);
        }
        // The only full X/Z-spanning face is the bottom, at y = -1.
        let bottom = &verts[24..30];
        assert!(bottom.iter().all(|v| (v.pos[1] - -1.0).abs() < 1e-6));
    }

    #[test]
    fn sky_fog_uniform_encodes_the_mode_in_the_colour_alpha() {
        assert_eq!(SkyFogUniform::OFF.color[3], 0.0);
        let lin = SkyFogUniform::new(
            Vec3::new(0.1, 0.2, 0.3),
            FogMode::Linear {
                start: 0.0,
                end: 192.0,
            },
        );
        assert_eq!(lin.color, [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(lin.params[1], 192.0);
        let exp = SkyFogUniform::new(Vec3::ZERO, FogMode::Exp { density: 0.1 });
        assert_eq!(exp.color[3], 2.0);
        assert_eq!(exp.params[2], 0.1);
    }

    /// CPU mirror of `dome.wgsl`'s `vs` overrides: what the vertex stage receives for one draw
    /// of a baked mesh. Both overrides are plain replacements, so this is exact.
    fn shade_baked(baked: &[DomeVertex], record: DomeDraw) -> Vec<DomeVertex> {
        baked
            .iter()
            .map(|v| {
                let mut out = *v;
                if record.mode[1] > 0.5 {
                    out.pos[1] = record.mode[2];
                }
                if record.mode[0] > 0.5 {
                    out.color = record.tint;
                }
                if record.rot[2] > 0.5 {
                    let [x, y, z] = out.pos;
                    let (c, s) = (record.rot[0], record.rot[1]);
                    let (y1, z1) = (y * c - z * s, y * s + z * c);
                    out.pos = [-z1, y1, x];
                }
                out
            })
            .collect()
    }

    fn vertex_bits(vertices: &[DomeVertex]) -> Vec<[u32; 9]> {
        vertices
            .iter()
            .map(|v| {
                let [x, y, z] = v.pos.map(f32::to_bits);
                let [u, w] = v.uv.map(f32::to_bits);
                let [r, g, b, a] = v.color.map(f32::to_bits);
                [x, y, z, u, w, r, g, b, a]
            })
            .collect()
    }

    /// #17 / SKY-2: the baked planes plus a per-draw record must hand the vertex stage exactly
    /// the vertices the per-frame `flat_verts` rebuild produced, for every translation the sky
    /// can use (including eye heights where the f32 addition rounds).
    #[test]
    fn baked_planes_with_draw_records_match_the_per_frame_rebuild_bit_for_bit() {
        let white = [1.0, 1.0, 1.0, 1.0];
        let sky_plane = plane_positions(16.0, false);
        let void_plane = plane_positions(-16.0, true);
        let baked_sky = flat_verts(&sky_plane, white, 0.0);
        let baked_void = flat_verts(&void_plane, white, 0.0);
        let colors = [
            [0.47, 0.65, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.134, 0.170_000_02, 0.700_000_1, 1.0],
        ];
        for color in colors {
            assert_eq!(
                vertex_bits(&shade_baked(
                    &baked_sky,
                    DomeDraw::plane(color, plane_draw_y(&sky_plane, 0.0))
                )),
                vertex_bits(&flat_verts(&sky_plane, color, 0.0))
            );
            for frame_origin in [
                -200.0,
                -64.0,
                0.0,
                62.9,
                63.0,
                64.62,
                255.0,
                16_777_217.25,
                1.0e9,
            ] {
                let (_, _, horizon_translation) = horizon_draw_offsets(frame_origin, 63.0);
                for dy in [12.0, horizon_translation, 0.1, -0.3] {
                    assert_eq!(
                        vertex_bits(&shade_baked(
                            &baked_void,
                            DomeDraw::plane(color, plane_draw_y(&void_plane, dy))
                        )),
                        vertex_bits(&flat_verts(&void_plane, color, dy)),
                        "frame_origin={frame_origin} dy={dy}"
                    );
                }
            }
        }
    }

    /// The star field is baked white; the brightness record must reproduce the old per-frame
    /// `.map(..).collect()` recolour exactly, positions untouched.
    #[test]
    fn baked_stars_with_a_tint_record_match_the_per_frame_recolour() {
        let stars = star_vertices();
        assert!(!stars.is_empty());
        for brightness in [1.0e-6_f32, 0.25, 0.5, 0.375 * 0.6] {
            let old: Vec<DomeVertex> = stars
                .iter()
                .map(|v| DomeVertex {
                    color: [brightness, brightness, brightness, brightness],
                    ..*v
                })
                .collect();
            assert_eq!(
                vertex_bits(&shade_baked(&stars, DomeDraw::tinted([brightness; 4]))),
                vertex_bits(&old)
            );
        }
        // Passthrough leaves every per-frame mesh exactly as built.
        let fan = sunrise_fan(true, [1.0, 0.5, 0.2, 0.8]);
        assert_eq!(
            vertex_bits(&shade_baked(&fan, DomeDraw::PASSTHROUGH)),
            vertex_bits(&fan)
        );
    }

    /// N1: with `rotate_stars`, the star field turns by the same `Ry(-90)·Rx(angle)` as the
    /// sun and moon (`celestial_rotation`); the shader uses the exact `Ry(-90)` permutation.
    #[test]
    fn rotated_stars_follow_the_sun_and_moon_rotation() {
        let stars = star_vertices();
        for angle in [0.0_f32, 0.13, 0.25, 0.5, 0.77] {
            let rotated = shade_baked(
                &stars,
                DomeDraw::tinted([1.0; 4]).celestially_rotated(angle),
            );
            let rot = celestial_rotation(angle);
            for (before, after) in stars.iter().zip(&rotated) {
                let expected = rot * Vec3::from_array(before.pos);
                let got = Vec3::from_array(after.pos);
                assert!(
                    (expected - got).length() < 1e-3,
                    "{angle}: {expected:?} {got:?}"
                );
            }
        }
        // Without the option the stars stay exactly where they were baked.
        let still = shade_baked(&stars, DomeDraw::tinted([1.0; 4]));
        assert_eq!(
            vertex_bits(&still[..6]),
            vertex_bits(
                &stars
                    .iter()
                    .map(|v| DomeVertex {
                        color: [1.0; 4],
                        ..*v
                    })
                    .collect::<Vec<_>>()[..6]
            )
        );
    }

    /// `cull_planes`: with counter-clockwise front faces (wgpu's default, GL's too), every sky
    /// plane triangle faces down at the eye and every reversed `-16` plane triangle faces up, so
    /// back-face culling drops the horizon plane exactly when it is translated above the eye.
    #[test]
    fn baked_plane_winding_matches_vanilla_so_culling_hides_the_horizon_plane_from_below() {
        let normal_y = |tri: &[DomeVertex]| {
            let [a, b, c] = [0, 1, 2].map(|i| Vec3::from_array(tri[i].pos));
            (b - a).cross(c - a).y
        };
        let white = [1.0; 4];
        let sky = flat_verts(&plane_positions(16.0, false), white, 0.0);
        let void = flat_verts(&plane_positions(-16.0, true), white, 0.0);
        assert!(sky.chunks(3).all(|tri| normal_y(tri) < 0.0));
        assert!(void.chunks(3).all(|tri| normal_y(tri) > 0.0));
    }

    #[test]
    fn per_frame_upload_is_bounded_by_the_dynamic_meshes_alone() {
        let color = [1.0, 1.0, 1.0, 1.0];
        let rot = celestial_rotation(0.3);
        assert_eq!(
            DYNAMIC_DOME_VERTICES,
            sunrise_fan(false, color).len()
                + celestial_quad(rot, 100.0, 30.0, UV_QUAD, color).len()
                + celestial_quad(rot, -100.0, 20.0, UV_QUAD, color).len()
                + void_box(-1.0).len()
        );
        // Passthrough + sky plane + stars + black void plane + horizon plane.
        assert_eq!(MAX_DOME_DRAWS, 5);
        assert!(DYNAMIC_DOME_VERTICES * std::mem::size_of::<DomeVertex>() < 4 * 1024);
    }
}
