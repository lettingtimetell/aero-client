//! The Block Overlay mod's selection box: a translucent fill and a
//! wide, coloured outline around the block under the crosshair.
//!
//! [`BlockOverlayPhase`](super::BlockOverlayPhase) draws vanilla's 1-px black `LineList` while
//! its `selection_style` is `None`, and this module's two pipelines instead while it is `Some`.
//! The pipelines are built on first use, so a client that never enables the mod never compiles
//! them.
//!
//! * **Fill:** the six faces of the same `0.002`-expanded box, back faces culled so no face
//!   doubles up in alpha, depth-tested (`LessEqual`) without depth writes. The expansion keeps
//!   the fill just in front of the block's own faces.
//! * **Outline:** wgpu has no wide lines (`LineList` is always one pixel), so each of the twelve
//!   edges is an instance of a quad that the vertex shader expands perpendicular to the edge's
//!   on-screen direction, by the width in framebuffer pixels (vanilla's `glLineWidth` unit).
//!   Same depth state as the fill, drawn after it. Like OpenGL's wide lines the quads have no
//!   caps, so corners show a small overlap rather than a mitre.

use crate::{Camera, Gpu};

use super::SelectionBox;

/// How the Block Overlay mod draws the selection box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectionStyle {
    /// Outline colour, straight-alpha RGBA in `0..=1`. Alpha 0 draws no outline.
    pub outline: [f32; 4],
    /// Outline width in framebuffer pixels.
    pub width_px: f32,
    /// Fill colour, straight-alpha RGBA in `0..=1`. Alpha 0 draws no fill.
    pub fill: [f32; 4],
}

impl SelectionStyle {
    /// A style from packed `0xAARRGGBB` colours, the mod settings' encoding.
    #[must_use]
    pub fn from_argb(outline_argb: u32, width_px: f32, fill_argb: u32) -> Self {
        Self {
            outline: unpack_argb(outline_argb),
            width_px: if width_px.is_finite() {
                width_px.clamp(0.0, 64.0)
            } else {
                1.0
            },
            fill: unpack_argb(fill_argb),
        }
    }

    /// Whether the outline pass draws anything.
    #[must_use]
    pub fn draws_outline(&self) -> bool {
        self.outline[3] > 0.0 && self.width_px > 0.0
    }

    /// Whether the fill pass draws anything.
    #[must_use]
    pub fn draws_fill(&self) -> bool {
        self.fill[3] > 0.0
    }
}

fn unpack_argb(argb: u32) -> [f32; 4] {
    [
        ((argb >> 16) & 0xFF) as f32 / 255.0,
        ((argb >> 8) & 0xFF) as f32 / 255.0,
        (argb & 0xFF) as f32 / 255.0,
        (argb >> 24) as f32 / 255.0,
    ]
}

/// The box's eight corners, camera-relative: bottom `0..4`, top `4..8`, each loop starting at
/// the minimum corner (the same order as [`SelectionBox::line_vertices`]).
fn corners(selection: &SelectionBox, cam: &Camera) -> [[f32; 3]; 8] {
    let lo = cam.relative(selection.min);
    let hi = cam.relative(selection.max);
    [
        [lo.x, lo.y, lo.z],
        [hi.x, lo.y, lo.z],
        [hi.x, lo.y, hi.z],
        [lo.x, lo.y, hi.z],
        [lo.x, hi.y, lo.z],
        [hi.x, hi.y, lo.z],
        [hi.x, hi.y, hi.z],
        [lo.x, hi.y, hi.z],
    ]
}

/// The twelve edges as corner pairs: bottom loop, top loop, four uprights.
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// The six faces as two triangles each, counter-clockwise seen from outside the box (the
/// winding back-face culling keeps): bottom, top, north (-Z), south (+Z), west (-X), east (+X).
const FILL_TRIANGLES: [usize; 36] = [
    0, 1, 2, 0, 2, 3, // bottom
    4, 7, 6, 4, 6, 5, // top
    0, 4, 5, 0, 5, 1, // north
    3, 2, 6, 3, 6, 7, // south
    0, 3, 7, 0, 7, 4, // west
    1, 5, 6, 1, 6, 2, // east
];

/// The fill pass's 36 camera-relative vertices.
#[must_use]
pub fn fill_vertices(selection: &SelectionBox, cam: &Camera) -> [[f32; 3]; 36] {
    let c = corners(selection, cam);
    FILL_TRIANGLES.map(|corner| c[corner])
}

/// The outline pass's twelve edge instances, `(start, end)` camera-relative.
#[must_use]
pub fn edge_instances(selection: &SelectionBox, cam: &Camera) -> [[[f32; 3]; 2]; 12] {
    let c = corners(selection, cam);
    EDGES.map(|(a, b)| [c[a], c[b]])
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StyleUniform {
    outline: [f32; 4],
    fill: [f32; 4],
    params: [f32; 4],
}

const FILL_BYTES: u64 = 36 * 12;
const EDGE_BYTES: u64 = 12 * 24;

/// The GPU half: two pipelines, the style uniform and two fixed-size vertex buffers.
pub(crate) struct StyledOutline {
    fill_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    fill_buffer: wgpu::Buffer,
    edge_buffer: wgpu::Buffer,
    /// What `prepare` uploaded: which passes `draw` records.
    fill: bool,
    outline: bool,
}

impl StyledOutline {
    pub(crate) fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("block overlay styled outline"),
            source: wgpu::ShaderSource::Wgsl(include_str!("styled_outline.wgsl").into()),
        });
        let style_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("block overlay style"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block overlay style"),
            size: std::mem::size_of::<StyleUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block overlay style"),
            layout: &style_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("block overlay styled outline"),
            bind_group_layouts: &[Some(gpu.camera_layout()), Some(&style_layout)],
            immediate_size: 0,
        });
        // java: drawSelectionBox's `depthMask(false)` with the depth test left on (LEQUAL).
        let depth = || wgpu::DepthStencilState {
            format: crate::DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let target = [Some(wgpu::ColorTargetState {
            format: gpu.config.format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let fill_attributes = wgpu::vertex_attr_array![0 => Float32x3];
        let fill_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("block overlay styled fill"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fill"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &fill_attributes,
                })],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: Some(depth()),
            multisample: wgpu::MultisampleState {
                count: gpu.sample_count(),
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_fill"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &target,
            }),
            multiview_mask: None,
            cache: None,
        });
        let edge_attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("block overlay styled edges"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_edge"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &edge_attributes,
                })],
            },
            // The quad's winding depends on the edge's screen direction.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(depth()),
            multisample: wgpu::MultisampleState {
                count: gpu.sample_count(),
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_edge"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &target,
            }),
            multiview_mask: None,
            cache: None,
        });
        let fill_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block overlay styled fill"),
            size: FILL_BYTES,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let edge_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block overlay styled edges"),
            size: EDGE_BYTES,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            fill_pipeline,
            edge_pipeline,
            uniform,
            bind_group,
            fill_buffer,
            edge_buffer,
            fill: false,
            outline: false,
        }
    }

    /// Upload this frame's geometry and style. Call before the pass begins.
    pub(crate) fn prepare(
        &mut self,
        gpu: &Gpu,
        selection: &SelectionBox,
        style: &SelectionStyle,
        cam: &Camera,
    ) {
        self.fill = style.draws_fill();
        self.outline = style.draws_outline();
        if !self.fill && !self.outline {
            return;
        }
        gpu.write_buffer_staged(
            &self.uniform,
            0,
            bytemuck::bytes_of(&StyleUniform {
                outline: style.outline,
                fill: style.fill,
                params: [style.width_px * 0.5, 0.0, 0.0, 0.0],
            }),
        );
        if self.fill {
            let vertices = fill_vertices(selection, cam);
            gpu.write_buffer_staged(&self.fill_buffer, 0, bytemuck::cast_slice(&vertices));
        }
        if self.outline {
            let edges = edge_instances(selection, cam);
            gpu.write_buffer_staged(&self.edge_buffer, 0, bytemuck::cast_slice(&edges));
        }
    }

    /// Record the fill, then the outline. Group 0 (the camera) must already be bound. Leaves
    /// vertex slot 0 and group 1 changed.
    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if !self.fill && !self.outline {
            return;
        }
        pass.set_bind_group(1, &self.bind_group, &[]);
        if self.fill {
            pass.set_pipeline(&self.fill_pipeline);
            pass.set_vertex_buffer(0, self.fill_buffer.slice(..));
            pass.draw(0..36, 0..1);
        }
        if self.outline {
            pass.set_pipeline(&self.edge_pipeline);
            pass.set_vertex_buffer(0, self.edge_buffer.slice(..));
            pass.draw(0..6, 0..12);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use mc_math::DVec3;

    fn unit_box() -> SelectionBox {
        SelectionBox {
            min: DVec3::new(0.0, 0.0, 0.0),
            max: DVec3::new(1.0, 1.0, 1.0),
        }
    }

    #[test]
    fn colours_unpack_from_the_settings_argb() {
        let style = SelectionStyle::from_argb(0x9900_0000, 3.0, 0x2600_0000);
        assert_eq!(style.outline, [0.0, 0.0, 0.0, 153.0 / 255.0]);
        assert_eq!(style.fill, [0.0, 0.0, 0.0, 38.0 / 255.0]);
        assert_eq!(style.width_px, 3.0);
        assert!(style.draws_outline() && style.draws_fill());
        let none = SelectionStyle::from_argb(0x00FF_FFFF, 3.0, 0x0012_3456);
        assert!(!none.draws_outline() && !none.draws_fill());
        assert_eq!(SelectionStyle::from_argb(0, f32::NAN, 0).width_px, 1.0);
    }

    #[test]
    fn every_fill_triangle_faces_out_of_the_box() {
        let cam = Camera::default();
        let vertices = fill_vertices(&unit_box(), &cam);
        let centre = {
            let c = corners(&unit_box(), &cam);
            c.iter().fold(Vec3::ZERO, |sum, p| sum + Vec3::from(*p)) / 8.0
        };
        for triangle in vertices.chunks_exact(3) {
            let [a, b, c] = [triangle[0], triangle[1], triangle[2]].map(Vec3::from);
            let normal = (b - a).cross(c - a);
            let outward = (a + b + c) / 3.0 - centre;
            assert!(
                normal.dot(outward) > 0.0,
                "triangle {triangle:?} winds inward: back-face culling would keep the far side"
            );
        }
    }

    #[test]
    fn the_twelve_edges_are_the_vanilla_line_list() {
        let cam = Camera::default();
        let selection = unit_box();
        let lines = selection.line_vertices(&cam);
        let edges = edge_instances(&selection, &cam);
        assert_eq!(lines.len(), 24);
        for (edge, pair) in edges.iter().zip(lines.chunks_exact(2)) {
            assert_eq!(edge[0], pair[0].pos);
            assert_eq!(edge[1], pair[1].pos);
        }
    }
}
