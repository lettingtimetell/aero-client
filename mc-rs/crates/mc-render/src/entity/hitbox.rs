//! Wide-line hitboxes for the Hitboxes mod.
//!
//! java: `RenderManager.renderDebugBoundingBox` draws F3+B with `GL_LINES`/`GL_LINE_STRIP` at the
//! driver's line width. wgpu has no wide lines (`PrimitiveTopology::LineList` is always one pixel),
//! so every segment here is an **instanced screen-facing quad**: the vertex shader projects both
//! endpoints, offsets them perpendicular to the segment by half the width in *pixels*, and extends
//! each end by the same half width so box corners close (square caps). The width is therefore the
//! same on screen at any distance, like a GL line.
//!
//! # Depth
//!
//! Depth-tested against the shared depth buffer, never written (`depthMask(false)` in vanilla), so
//! terrain and bodies in front hide the lines exactly as they hide F3+B. Each endpoint is pulled
//! 0.2 % of its distance toward the camera before projection: a box's bottom edges lie exactly on
//! the ground, and without the pull the half of a wide quad that falls "into" the ground would fail
//! the depth test and the bottom edges would look half as thick as the rest.
//!
//! # Cost
//!
//! The phase is registered once (no GPU objects); [`HitboxPhase::finish`] creates the pipeline and
//! the instance buffer the first time it has segments. With nothing queued [`RenderPhase::enabled`]
//! is `false`, so a frame with the mod off records nothing and allocates nothing.
//!
//! # Future shaders
//!
//! The phase reads only the shared camera uniform and depth buffer and writes the scene colour at
//! [`HITBOX_PHASE_ORDER`] (342), after the opaque world and its ambient occlusion: a shader pack
//! that replaces the entity pass keeps it by drawing after its own entity output and lighting,
//! before translucent terrain.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::context::{Gpu, DEPTH_FORMAT};
use crate::frame::Camera;
use crate::pipeline::{PassTarget, PhaseOrder, RenderPhase};

/// Where the phase records: after the entity pass and the screen-space ambient occlusion (340),
/// before the client's nameplates (345), the block overlay (350) and translucent terrain (500),
/// which is where vanilla's per-entity F3+B boxes land.
///
/// After AO (X15; it was `ENTITIES + 5`): the lines write no depth, so
/// AO's multiply would grey them with the occlusion of the surface behind. Nothing else records
/// between 305 and 342, so with AO off the frame is the one the old order drew.
pub const HITBOX_PHASE_ORDER: PhaseOrder = PhaseOrder(PhaseOrder::AMBIENT_OCCLUSION.0 + 2);

/// One line segment, camera-relative, one instance of the quad.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct HitboxSegment {
    /// Start point (`xyz`) and the line width in physical pixels (`w`).
    pub a: [f32; 4],
    /// End point (`xyz`); `w` is unused padding.
    pub b: [f32; 4],
    /// Straight RGBA.
    pub color: [f32; 4],
}

const SEGMENT_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: core::mem::size_of::<HitboxSegment>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 16,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 32,
            shader_location: 2,
        },
    ],
};

/// Screen-space wide lines. `screen.xy` is the viewport in physical pixels.
const HITBOX_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Clip `p` toward `q` so it lies on or in front of the near plane (wgpu clip z >= 0).
fn clip_near(p: vec4<f32>, q: vec4<f32>) -> vec4<f32> {
    if (p.z >= 0.0) {
        return p;
    }
    let t = p.z / (p.z - q.z);
    return mix(p, q, t);
}

@vertex
fn vs(
    @builtin(vertex_index) index: u32,
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.color = color;
    // Pull both ends 0.2% toward the camera (positions are camera-relative): see the module docs.
    let ca0 = camera.view_proj * vec4<f32>(a.xyz * 0.998, 1.0);
    let cb0 = camera.view_proj * vec4<f32>(b.xyz * 0.998, 1.0);
    if (ca0.z < 0.0 && cb0.z < 0.0) {
        // Entirely behind the near plane: a degenerate quad outside the clip volume.
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let ca = clip_near(ca0, cb0);
    let cb = clip_near(cb0, ca0);
    let viewport = max(camera.screen.xy, vec2<f32>(1.0, 1.0));
    let pa = ca.xy / ca.w * 0.5 * viewport;
    let pb = cb.xy / cb.w * 0.5 * viewport;
    var dir = pb - pa;
    let len = length(dir);
    if (len < 1e-4) {
        dir = vec2<f32>(1.0, 0.0);
    } else {
        dir = dir / len;
    }
    let normal = vec2<f32>(-dir.y, dir.x);
    let half_width = a.w * 0.5;
    // Two triangles: (0, 1, 2) and (2, 1, 3); corner bit 0 = which end, bit 1 = which side.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[index];
    let end_clip = select(ca, cb, corner.x > 0.5);
    let cap = select(-dir, dir, corner.x > 0.5);
    let offset_px = (normal * corner.y + cap) * half_width;
    let offset_ndc = offset_px / (0.5 * viewport);
    out.clip = vec4<f32>(end_clip.xy + offset_ndc * end_clip.w, end_clip.z, end_clip.w);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// The GPU objects, created on first use.
struct HitboxGpu {
    pipeline: wgpu::RenderPipeline,
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
}

/// Queued hitbox segments and the pipeline that draws them as wide quads.
pub struct HitboxPhase {
    segments: Vec<HitboxSegment>,
    gpu: Option<HitboxGpu>,
    uploaded: u32,
}

impl Default for HitboxPhase {
    fn default() -> Self {
        Self::new()
    }
}

impl HitboxPhase {
    /// An empty phase. Holds no GPU objects until [`Self::finish`] first has segments.
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            gpu: None,
            uploaded: 0,
        }
    }

    /// Drop this frame's segments (keeps the allocation).
    pub fn begin(&mut self) {
        self.segments.clear();
    }

    /// Segments queued since [`Self::begin`].
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// The queued segments, for tests.
    #[must_use]
    pub fn segments(&self) -> &[HitboxSegment] {
        &self.segments
    }

    /// Whether the pipeline exists (it is created the first time a frame has segments).
    #[must_use]
    pub fn has_gpu_resources(&self) -> bool {
        self.gpu.is_some()
    }

    /// Queue one segment `a`-`b` (camera-relative), `width` physical pixels wide.
    pub fn push_line(&mut self, a: Vec3, b: Vec3, width: f32, color: [f32; 4]) {
        self.segments.push(HitboxSegment {
            a: [a.x, a.y, a.z, width],
            b: [b.x, b.y, b.z, 0.0],
            color,
        });
    }

    /// Queue the twelve edges of the box `min`-`max`, as `RenderGlobal.drawOutlinedBoundingBox`
    /// draws them: bottom, top, then the four uprights.
    pub fn push_box(&mut self, min: Vec3, max: Vec3, width: f32, color: [f32; 4]) {
        let c = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];
        for (i, j) in [
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
        ] {
            self.push_line(c[i], c[j], width, color);
        }
    }

    /// Queue a horizontal rectangle at height `y` over the box's `x`/`z` extent (the eye-height
    /// line; vanilla draws a 0.02-tall box there, whose two outlines merge at any useful width).
    pub fn push_rect(&mut self, min: Vec3, max: Vec3, y: f32, width: f32, color: [f32; 4]) {
        let c = [
            Vec3::new(min.x, y, min.z),
            Vec3::new(max.x, y, min.z),
            Vec3::new(max.x, y, max.z),
            Vec3::new(min.x, y, max.z),
        ];
        for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            self.push_line(c[i], c[j], width, color);
        }
    }

    /// Upload this frame's segments, creating the pipeline and buffer on first use.
    pub fn finish(&mut self, gpu: &Gpu) {
        if self.segments.is_empty() {
            self.uploaded = 0;
            return;
        }
        let state = self.gpu.get_or_insert_with(|| HitboxGpu {
            pipeline: build_pipeline(gpu),
            buffer: None,
            capacity: 0,
        });
        let bytes: &[u8] = bytemuck::cast_slice(&self.segments);
        let needed = bytes.len() as u64;
        if state.buffer.is_none() || state.capacity < needed {
            let capacity = needed.next_power_of_two().max(4096);
            state.buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc-render hitbox segments"),
                size: capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            state.capacity = capacity;
        }
        if let Some(buffer) = &state.buffer {
            gpu.queue.write_buffer(buffer, 0, bytes);
        }
        self.uploaded = u32::try_from(self.segments.len()).unwrap_or(u32::MAX);
    }

    /// Release the GPU objects (the mod was switched off). The next segments rebuild them.
    pub fn release(&mut self) {
        self.segments.clear();
        self.uploaded = 0;
        self.gpu = None;
    }
}

fn build_pipeline(gpu: &Gpu) -> wgpu::RenderPipeline {
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mc-render hitboxes"),
            source: wgpu::ShaderSource::Wgsl(HITBOX_WGSL.into()),
        });
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc-render hitboxes"),
            bind_group_layouts: &[Some(gpu.camera_layout())],
            immediate_size: 0,
        });
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mc-render hitboxes"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(SEGMENT_LAYOUT)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                // java: `depthMask(false)`, depth test left on.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: gpu.sample_count(),
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    // Straight alpha, so a translucent colour option blends; opaque is a copy.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
}

impl RenderPhase for HitboxPhase {
    fn name(&self) -> &'static str {
        "hitboxes"
    }

    fn enabled(&self) -> bool {
        self.uploaded > 0
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.record_to(gpu, PassTarget::Own { enc, view });
    }

    // One pass, nothing else on the encoder: it joins the world pass, so
    // turning hitboxes on does not split it.
    fn shares_world_pass(&self) -> bool {
        true
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, _cam: &Camera) {
        self.record_to(gpu, PassTarget::Shared(pass));
    }
}

impl HitboxPhase {
    /// The phase's draw, into a pass of its own or the shared world pass.
    fn record_to(&self, gpu: &Gpu, target: PassTarget<'_>) {
        let Some(state) = &self.gpu else {
            return;
        };
        let Some(buffer) = &state.buffer else {
            return;
        };
        if self.uploaded == 0 {
            return;
        }
        let mut own = None;
        let pass = target.begin(gpu, "hitboxes", &mut own);
        pass.set_pipeline(&state.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        let bytes = u64::from(self.uploaded) * core::mem::size_of::<HitboxSegment>() as u64;
        pass.set_vertex_buffer(0, buffer.slice(..bytes));
        pass.draw(0..6, 0..self.uploaded);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_is_twelve_segments_and_an_eye_rect_four() {
        let mut phase = HitboxPhase::new();
        phase.push_box(Vec3::ZERO, Vec3::ONE, 2.0, [1.0; 4]);
        assert_eq!(phase.segment_count(), 12);
        phase.push_rect(Vec3::ZERO, Vec3::ONE, 0.5, 2.0, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(phase.segment_count(), 16);
        assert!(phase.segments()[12..]
            .iter()
            .all(|s| s.a[1] == 0.5 && s.b[1] == 0.5));
        assert!(phase.segments().iter().all(|s| s.a[3] == 2.0));
        phase.begin();
        assert_eq!(phase.segment_count(), 0);
        assert!(
            !phase.enabled(),
            "nothing uploaded: the phase records nothing"
        );
        assert!(
            !phase.has_gpu_resources(),
            "no GPU objects until segments are finished"
        );
    }

    #[test]
    fn the_segment_layout_matches_the_shader_inputs() {
        assert_eq!(core::mem::size_of::<HitboxSegment>(), 48);
        assert!(HITBOX_WGSL.contains("@location(0) a: vec4<f32>"));
        assert!(HITBOX_WGSL.contains("@location(2) color: vec4<f32>"));
        // X15: after AO, which would otherwise shade the lines, and before the block overlay.
        assert_eq!(HITBOX_PHASE_ORDER.0, 342);
        assert!(HITBOX_PHASE_ORDER > PhaseOrder::AMBIENT_OCCLUSION);
        assert!(HITBOX_PHASE_ORDER < PhaseOrder::BLOCK_OVERLAY);
    }
}
