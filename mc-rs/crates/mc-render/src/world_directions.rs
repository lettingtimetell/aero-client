//! The F3 axes gizmo: `EntityRenderer.renderWorldDirections`.
//!
//! While the debug screen is open, `GuiIngame.showCrosshair` hides the 2D crosshair, and
//! `renderWorldPass` draws this in its place, straight after `renderHand`
//! (`EntityRenderer.java:1482-1487`): three tiny world-aligned boxes at the render-view entity's
//! eye, red along +X (east), green along +Y (up) and blue along +Z (south).
//!
//! The method (`EntityRenderer.java:1265-1288`) loads an identity model-view and replays only
//! `orientCamera`, so the gizmo has no view bobbing, hurt shake or portal warp. It keeps the
//! projection `renderHand` installed, which uses the fixed hand FOV
//! ([`Camera::hand_fov`](crate::Camera::hand_fov)), not the world FOV. In first person the boxes
//! sit `0.1` in front of the eye (`orientCamera`'s `translate(0, 0, -0.1)`), so the X and Z arms
//! are about 7% of the half-height of the screen. In F5 they sit at the player's eye, a few
//! blocks away, and are a dot or two wide.
//!
//! State: blend `SRC_ALPHA, ONE_MINUS_SRC_ALPHA`, line width `1`, texturing off, depth writes
//! off. Depth testing is left on in Java, but the depth buffer was cleared for the hand and
//! holds nothing nearer than the gizmo's `0.1`, so this phase uses no depth attachment.

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::pipeline::first_person_projection;
use crate::{Camera, DepthAttachment, Gpu, RenderPhase};

/// One `RenderGlobal.drawOutlinedBoundingBox` call: the box's maximum corner (its minimum is
/// the origin) and its RGBA colour, in Java's draw order (`EntityRenderer.java:1280-1282`).
pub const WORLD_DIRECTION_BOXES: [([f64; 3], [u8; 4]); 3] = [
    // X: `AxisAlignedBB(0, 0, 0, 0.005, 1.0E-4, 1.0E-4)`, red.
    ([0.005, 1.0e-4, 1.0e-4], [255, 0, 0, 255]),
    // Z: `AxisAlignedBB(0, 0, 0, 1.0E-4, 1.0E-4, 0.005)`, blue.
    ([1.0e-4, 1.0e-4, 0.005], [0, 0, 255, 255]),
    // Y: `AxisAlignedBB(0, 0, 0, 1.0E-4, 0.0033, 1.0E-4)`, green.
    ([1.0e-4, 0.0033, 1.0e-4], [0, 255, 0, 255]),
];

/// Vertices per box: twelve edges as line-list pairs.
pub const WORLD_DIRECTION_BOX_VERTICES: usize = 24;

/// `orientCamera`'s first-person `translate(0, 0, -0.1)`, the view-space offset that puts the
/// gizmo just in front of the eye.
pub const FIRST_PERSON_VIEW_OFFSET: Vec3 = Vec3::new(0.0, 0.0, -0.1);

/// One gizmo vertex: a position relative to the eye in world axes, and a colour.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldDirectionVertex {
    /// Position relative to the gizmo origin, in world axes, as Java's `WorldRenderer.pos`
    /// stores it (`(float)` of the double box coordinate).
    pub pos: [f32; 3],
    /// `POSITION_COLOR`'s four colour bytes.
    pub color: [u8; 4],
}

const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<WorldDirectionVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Unorm8x4,
            offset: 12,
            shader_location: 1,
        },
    ],
};

/// Append one `RenderGlobal.drawOutlinedBoundingBox` as line-list pairs.
///
/// java: the bottom rectangle as a five-vertex `GL_LINE_STRIP`, the top rectangle the same
/// way, then the four vertical edges as `GL_LINES`. Each strip segment becomes one pair here,
/// in the same order, so overlapping later edges still land on top.
fn push_outlined_box(max: [f64; 3], color: [u8; 4], out: &mut Vec<WorldDirectionVertex>) {
    let (x0, y0, z0) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (x1, y1, z1) = (max[0] as f32, max[1] as f32, max[2] as f32);
    let v = |x: f32, y: f32, z: f32| WorldDirectionVertex {
        pos: [x, y, z],
        color,
    };
    for y in [y0, y1] {
        let ring = [
            v(x0, y, z0),
            v(x1, y, z0),
            v(x1, y, z1),
            v(x0, y, z1),
            v(x0, y, z0),
        ];
        for pair in ring.windows(2) {
            out.extend_from_slice(pair);
        }
    }
    for (x, z) in [(x0, z0), (x1, z0), (x1, z1), (x0, z1)] {
        out.push(v(x, y0, z));
        out.push(v(x, y1, z));
    }
}

/// Every gizmo vertex, red X then blue Z then green Y, as Java draws them.
#[must_use]
pub fn world_direction_vertices() -> Vec<WorldDirectionVertex> {
    let mut out = Vec::with_capacity(WORLD_DIRECTION_BOXES.len() * WORLD_DIRECTION_BOX_VERTICES);
    for (max, color) in WORLD_DIRECTION_BOXES {
        push_outlined_box(max, color, &mut out);
    }
    out
}

/// Clip-space matrix for the gizmo's eye-relative vertices.
///
/// java: the projection is `renderHand`'s ([`first_person_projection`]); the model-view is
/// `loadIdentity`, `orientCamera`, then `translate(0, eyeHeight, 0)`, which cancels
/// `orientCamera`'s closing `translate(0, -eyeHeight, 0)`. What remains is the view-space
/// offset (`view_offset`: [`FIRST_PERSON_VIEW_OFFSET`] in first person, zero in F5), the
/// camera's pitch and `yaw + 180` rotation, and the eye's position relative to the render
/// camera (`origin`: zero in first person, the F5 orbit's offset otherwise). Using
/// `Camera::yaw/pitch` keeps the F5 front view's flipped orientation, which Java builds from
/// the same rotations.
#[must_use]
pub fn world_directions_matrix(cam: &Camera, origin: Vec3, view_offset: Vec3) -> Mat4 {
    first_person_projection(cam)
        * Mat4::from_translation(view_offset)
        * Mat4::from_rotation_x(cam.pitch.to_radians())
        * Mat4::from_rotation_y((cam.yaw + 180.0).to_radians())
        * Mat4::from_translation(origin)
}

const WORLD_DIRECTIONS_WGSL: &str = r#"
struct DirectionsUniform {
    transform: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> directions: DirectionsUniform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    out.clip = directions.transform * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

// java: `disableTexture2D()` with lighting and fog already off; the vertex colour is the pixel.
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// `EntityRenderer.renderWorldDirections`. The client sets the three public fields each frame.
pub struct WorldDirectionsPhase {
    /// Java's gate: `showDebugInfo && !hideGUI && !thePlayer.hasReducedDebug() &&
    /// !reducedDebugInfo`, in a world. The client also clears it while the player sleeps:
    /// Java still draws then, but the bed camera puts the boxes behind the near plane.
    pub visible: bool,
    /// The render-view entity's eye relative to [`Camera::pos`] (`eye - camera.pos`).
    pub origin: Vec3,
    /// `orientCamera`'s view-space translation; see [`world_directions_matrix`].
    pub view_offset: Vec3,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_group: wgpu::BindGroup,
    uniform_write: crate::context::ChangedWrite,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

impl WorldDirectionsPhase {
    /// Build the line pipeline and the static gizmo geometry. Hidden until the client shows it.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("world directions"),
                source: wgpu::ShaderSource::Wgsl(WORLD_DIRECTIONS_WGSL.into()),
            });
        let uniform_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("world directions transform"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("world directions layout"),
                bind_group_layouts: &[Some(gpu.camera_layout()), Some(&uniform_layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("world directions"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(VERTEX_LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        // java: `tryBlendFuncSeparate(770, 771, 1, 0)`.
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::Zero,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let identity = Mat4::IDENTITY.to_cols_array_2d();
        let uniform_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("world directions transform"),
                contents: bytemuck::bytes_of(&identity),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let uniform_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world directions transform"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let vertices = world_direction_vertices();
        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("world directions vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        Self {
            visible: false,
            origin: Vec3::ZERO,
            view_offset: FIRST_PERSON_VIEW_OFFSET,
            pipeline,
            uniform_buffer,
            uniform_group,
            uniform_write: crate::context::ChangedWrite::default(),
            vertex_buffer,
            vertex_count: vertices.len() as u32,
        }
    }
}

impl RenderPhase for WorldDirectionsPhase {
    fn name(&self) -> &'static str {
        "world-directions"
    }

    fn enabled(&self) -> bool {
        self.visible
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        if !self.visible {
            return;
        }
        let transform = world_directions_matrix(cam, self.origin, self.view_offset);
        self.uniform_write.write(
            gpu,
            &self.uniform_buffer,
            bytemuck::bytes_of(&transform.to_cols_array_2d()),
        );
        let mut pass = gpu.begin_pass(enc, view, "world-directions", DepthAttachment::None);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &self.uniform_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_replay_draw_outlined_bounding_box_edge_for_edge() {
        let vertices = world_direction_vertices();
        assert_eq!(vertices.len(), 3 * WORLD_DIRECTION_BOX_VERTICES);
        // Java's order: red X, then blue Z, then green Y.
        let colors: Vec<[u8; 4]> = vertices
            .chunks(WORLD_DIRECTION_BOX_VERTICES)
            .map(|chunk| chunk[0].color)
            .collect();
        assert_eq!(
            colors,
            [[255, 0, 0, 255], [0, 0, 255, 255], [0, 255, 0, 255]]
        );
        // The first box's first strip: (min,min,min) -> (max,min,min) -> (max,min,max) ...
        let x = &vertices[..WORLD_DIRECTION_BOX_VERTICES];
        assert_eq!(x[0].pos, [0.0, 0.0, 0.0]);
        assert_eq!(x[1].pos, [0.005, 0.0, 0.0]);
        assert_eq!(x[2].pos, [0.005, 0.0, 0.0]);
        assert_eq!(x[3].pos, [0.005, 0.0, 1.0e-4]);
        // ... then the top ring at maxY, then four verticals.
        assert_eq!(x[8].pos, [0.0, 1.0e-4, 0.0]);
        assert_eq!(x[16].pos, [0.0, 0.0, 0.0]);
        assert_eq!(x[17].pos, [0.0, 1.0e-4, 0.0]);
        // Every box spans exactly its Java extents and no more.
        for (chunk, (max, _)) in vertices
            .chunks(WORLD_DIRECTION_BOX_VERTICES)
            .zip(WORLD_DIRECTION_BOXES)
        {
            for (axis, &extent) in max.iter().enumerate() {
                let hi = chunk.iter().map(|v| v.pos[axis]).fold(f32::MIN, f32::max);
                let lo = chunk.iter().map(|v| v.pos[axis]).fold(f32::MAX, f32::min);
                assert_eq!((lo, hi), (0.0, extent as f32));
            }
        }
    }

    /// A camera facing south (yaw 0) at the default 70-degree hand FOV.
    fn camera(yaw: f32, pitch: f32, fov: f32) -> Camera {
        let mut cam = Camera {
            yaw,
            pitch,
            fov,
            ..Camera::default()
        };
        cam.set_viewport(1920, 1080);
        cam.update();
        cam
    }

    #[test]
    fn first_person_gizmo_sits_at_the_screen_centre_at_the_hand_fov() {
        let cam = camera(0.0, 0.0, 110.0);
        let m = world_directions_matrix(&cam, Vec3::ZERO, FIRST_PERSON_VIEW_OFFSET);
        let centre = m.project_point3(Vec3::ZERO);
        assert!(centre.x.abs() < 1e-6 && centre.y.abs() < 1e-6, "{centre:?}");

        // Facing south, east (+X) is screen-left and up (+Y) is screen-up.
        let x_tip = m.project_point3(Vec3::new(0.005, 0.0, 0.0));
        let y_tip = m.project_point3(Vec3::new(0.0, 0.0033, 0.0));
        assert!(x_tip.x < 0.0 && x_tip.y.abs() < 1e-6, "{x_tip:?}");
        assert!(y_tip.y > 0.0 && y_tip.x.abs() < 1e-6, "{y_tip:?}");

        // gluPerspective(70, ...): NDC y = 0.0033 / (0.1 * tan(35 deg)), independent of the
        // world FOV of 110 given to this camera.
        let expected = 0.0033 / (0.1 * (35.0_f32).to_radians().tan());
        assert!(
            (y_tip.y - expected).abs() < 1e-4,
            "{} vs {expected}",
            y_tip.y
        );
        let pixels = y_tip.y * 540.0;
        assert!(
            (pixels - 25.45).abs() < 0.05,
            "Y arm is {pixels} px at 1080p"
        );
    }

    #[test]
    fn the_axes_turn_with_the_camera_but_the_gizmo_has_no_bobbing() {
        // Facing west (yaw 90): +Z (south) is screen-left.
        let cam = camera(90.0, 0.0, 70.0);
        let m = world_directions_matrix(&cam, Vec3::ZERO, FIRST_PERSON_VIEW_OFFSET);
        let z_tip = m.project_point3(Vec3::new(0.0, 0.0, 0.005));
        assert!(z_tip.x < 0.0 && z_tip.y.abs() < 1e-5, "{z_tip:?}");

        // Only yaw/pitch reach the gizmo; `Camera::view` effects (bobbing, hurt) do not.
        let mut shaken = cam;
        shaken.view = Mat4::from_rotation_z(0.5) * cam.view;
        assert_eq!(
            world_directions_matrix(&shaken, Vec3::ZERO, FIRST_PERSON_VIEW_OFFSET),
            m
        );
    }

    #[test]
    fn third_person_gizmo_sits_at_the_eye_ahead_of_the_orbit() {
        // F5 rear view, looking south from 4 blocks behind the eye: the eye is 4 ahead.
        let cam = camera(0.0, 0.0, 70.0);
        let origin = Vec3::new(0.0, 0.0, 4.0);
        let m = world_directions_matrix(&cam, origin, Vec3::ZERO);
        let centre = m.project_point3(Vec3::ZERO);
        assert!(centre.x.abs() < 1e-6 && centre.y.abs() < 1e-6, "{centre:?}");
        let x_tip = m.project_point3(Vec3::new(0.005, 0.0, 0.0));
        let first_person = world_directions_matrix(&cam, Vec3::ZERO, FIRST_PERSON_VIEW_OFFSET)
            .project_point3(Vec3::new(0.005, 0.0, 0.0));
        assert!(
            (x_tip.x / first_person.x - 0.1 / 4.0).abs() < 1e-4,
            "the arm shrinks with distance: {} vs {}",
            x_tip.x,
            first_person.x
        );
    }
}
