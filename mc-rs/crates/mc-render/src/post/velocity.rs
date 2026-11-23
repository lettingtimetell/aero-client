//! Motion vectors for TAA and per-pixel motion blur.
//!
//! [`VelocityPhase`] writes one screen-sized [`VELOCITY_FORMAT`] target per frame at
//! [`crate::PhaseOrder::TEMPORAL_AA`], before the anti-aliasing resolve that reads it. Every
//! value is in UV units (0..1 across the screen, y down) and excludes the TAA jitter, so a still
//! camera over a still scene writes exactly zero:
//!
//! * `rg`: where the pixel's surface is now minus where it was on the **previous frame**. TAA
//!   samples its history at `uv - rg`; a future shader pack's temporal passes do the same.
//! * `ba`: the same over the **motion-blur exposure** (see below); the per-pixel motion blur
//!   (`post/velocity_blur.rs`) draws its trail along it.
//!
//! # Camera motion: from depth, for every pixel
//!
//! A fullscreen pass reads the depth buffer (the world's, jittered while TAA runs), rebuilds the
//! camera-relative position of each pixel's surface with this frame's inverse jittered
//! projection and inverse view, and projects it with the previous frame's unjittered
//! view-projection, offset by how far the eye moved (positions are camera-relative, see
//! [`Camera`]). The sky (depth 1) stands in for infinity and only rotates. The numbers live in
//! one uniform, [`ReprojectionUniform`], which [`VelocityOutput`] shares with the readers.
//!
//! # Object motion: entity boxes, entity rendering untouched
//!
//! The client passes each moving entity's box, its displacement since the previous frame (the
//! difference of the two frames' render positions) and its velocity ([`ObjectMotion`]). A second
//! pass draws the boxes; its fragment shader keeps only the pixels whose depth-rebuilt surface
//! lies inside the box, i.e. the entity itself and not the wall behind it, and writes the motion
//! of that surface with the entity's own displacement added. That is per-object motion for
//! translation (walking, strafing, knockback, falling), which is what moves on screen in PvP;
//! limb swing and body turns use the camera's motion. No entity shader, vertex format or draw
//! changes, so the off path cannot be affected.
//!
//! # The blur exposure
//!
//! A frame at 1000 FPS moves a pixel by a fraction of a pixel, and mouse input arrives in uneven
//! steps, so the blur does not scale one frame's motion. It reprojects against a **lagging
//! camera** that follows the real one as a first-order lag with time constant `exposure`
//! (seconds), solved exactly per frame (`LagCamera`). Turning at a steady speed `w` keeps the
//! lag `w * exposure` behind, so the trail is `w * exposure` long at any frame rate, and uneven
//! input is smoothed. Objects use `velocity * exposure`. The projection is always this frame's,
//! so a FOV change (sprint, zoom) does not blur radially.
//!
//! # Cost
//!
//! Off (no consumer, outside a world, or under MSAA): `enabled() == false`, no texture exists,
//! nothing records. On: one fullscreen pass writing 8 bytes per pixel plus one small instanced
//! draw; the uniform (384 bytes) and the instance data go through the per-frame upload ring.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use glam::{DVec3, Mat4, Quat, Vec2, Vec3};

use crate::{Camera, Gpu, RenderPhase};

/// The motion target's format: two motions of two channels each, fp16.
pub const VELOCITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The most moving entities one frame passes to the object pass. Further ones (rare: a crowded
/// lobby) keep the camera's motion.
pub const MAX_OBJECTS: usize = 256;

/// A frame longer than this, in seconds, restarts the motion: the previous frame is too old to
/// reproject from (a hitch, a minimised window, a loading screen).
pub const MAX_FRAME_DT: f32 = 0.25;

/// The eye moving further than this in one frame, in blocks, is a teleport, not motion.
pub const TELEPORT_DISTANCE: f64 = 8.0;

/// The most the blur's lagging camera may trail, in radians, so a 180-degree flick at a long
/// exposure smears a bounded amount (the blur also clamps its length on screen).
pub const MAX_LAG_ANGLE: f32 = 0.35;

/// The most the blur's lagging eye may trail, in blocks.
pub const MAX_LAG_DISTANCE: f64 = 2.0;

/// The uniform behind the motion target, shared with its readers (TAA). WGSL declaration in
/// `velocity.wgsl` (`struct Reprojection`); field meanings there.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReprojectionUniform {
    /// Inverse of this frame's jittered projection.
    pub inv_proj: [[f32; 4]; 4],
    /// Inverse of this frame's view.
    pub inv_view: [[f32; 4]; 4],
    /// This frame's unjittered projection * view.
    pub view_proj: [[f32; 4]; 4],
    /// The previous frame's unjittered projection * view.
    pub prev_view_proj: [[f32; 4]; 4],
    /// This frame's projection * the view at the start of the blur exposure.
    pub blur_view_proj: [[f32; 4]; 4],
    /// `xyz`: this eye minus the previous frame's eye. `w`: 1 when the previous frame is valid.
    pub prev_delta: [f32; 4],
    /// `xyz`: this eye minus the eye at the exposure start. `w`: 1 when the blur wants `ba`.
    pub blur_delta: [f32; 4],
    /// `xy`: this frame's jitter (NDC). `zw`: the previous frame's.
    pub jitter: [f32; 4],
    /// Width, height, 1 / width, 1 / height.
    pub screen: [f32; 4],
}

/// One moving entity for the object pass.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ObjectMotion {
    /// The box's low corner relative to this frame's eye ([`Camera::relative`]), in blocks,
    /// already grown to cover the model (arms and held items stick out of the hitbox).
    pub min: Vec3,
    /// The box's high corner, same space.
    pub max: Vec3,
    /// How far the entity's render position moved since the previous frame, in blocks: the
    /// difference of the two frames' positions, not velocity times frame time, so it stays exact
    /// across tick boundaries, knockback and a lockstep or slowed game clock (`rg`).
    pub frame_delta: Vec3,
    /// The entity's render velocity in blocks per second (`ba`: times the blur exposure).
    pub velocity: Vec3,
}

/// Per-instance data of the object pass (`ObjectIn` in `velocity.wgsl`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectInstance {
    min: [f32; 4],
    max: [f32; 4],
    moved: [f32; 4],
    moved_blur: [f32; 4],
}

/// What the readers of the motion target bind: refreshed every frame the velocity pass records.
pub struct VelocityFrame {
    /// The motion target's texture (`COPY_SRC` too, so a test or a debug view can read it).
    pub texture: wgpu::Texture,
    /// The motion target ([`VELOCITY_FORMAT`], the frame's size).
    pub view: wgpu::TextureView,
    /// The frame's [`ReprojectionUniform`].
    pub uniform: wgpu::Buffer,
    /// Changes whenever `view` or `uniform` is a new object, so a reader rebuilds its bind
    /// group then and never per frame.
    pub generation: u64,
    /// Counts the frames the pass recorded. A reader that sees the same count twice knows the
    /// target is stale this frame.
    pub frame: u64,
    /// Whether the previous frame was valid for reprojection this frame (`prev_delta.w`).
    pub has_previous: bool,
}

/// The read side of [`VelocityPhase`]: a cheap handle a reader (TAA, the motion blur, a future
/// shader-pack pass) keeps and borrows while it records. Phases record one after another on one
/// thread, so the borrow never overlaps the writer's.
#[derive(Clone, Default)]
pub struct VelocityOutput(Rc<RefCell<Option<VelocityFrame>>>);

impl VelocityOutput {
    /// Borrow this frame's motion target, or `None` before the pass first ran.
    #[must_use]
    pub fn get(&self) -> std::cell::Ref<'_, Option<VelocityFrame>> {
        self.0.borrow()
    }
}

/// Which readers want the target. The pass runs only while one does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VelocityConsumers {
    /// TAA reads `rg` (and the object motion, against ghosting on moving players).
    pub taa: bool,
    /// The per-pixel motion blur reads `ba`.
    pub blur: bool,
    /// The blur uses the objects' own motion (Camera + objects). Off: entity pixels blur with
    /// the camera only.
    pub blur_objects: bool,
}

impl VelocityConsumers {
    /// Whether any reader is on.
    #[must_use]
    pub fn any(self) -> bool {
        self.taa || self.blur
    }
}

/// The previous frame's camera, for `rg`.
#[derive(Debug, Clone, Copy)]
struct PrevCamera {
    view_proj: Mat4,
    pos: DVec3,
    jitter: Vec2,
}

/// A camera's pose: the view's rotation and view-space offset, and the eye.
#[derive(Debug, Clone, Copy)]
struct Pose {
    rotation: Quat,
    translation: Vec3,
    pos: DVec3,
}

impl Pose {
    fn of(cam: &Camera) -> Pose {
        Pose {
            rotation: Quat::from_mat4(&cam.view).normalize(),
            translation: cam.view.w_axis.truncate(),
            pos: cam.pos,
        }
    }
}

/// The blur's lagging camera (module docs), kept as how far it trails the real one.
///
/// It is the continuous first-order lag `dy/dt = (camera - y) / exposure` solved exactly for a
/// camera that moves linearly within each frame: `trail' = trail * k + step * (1 - k) / x` with
/// `x = dt / exposure`, `k = e^-x`. A steady turn of `w` per second settles at exactly
/// `w * exposure` behind whatever the frame time; a plain `lerp(lag, camera, 1 - k)` would trail
/// only `w * dt * k / (1 - k)`, under half of it at 60 FPS.
#[derive(Debug, Clone, Copy)]
struct LagCamera {
    /// Rotation from the lagging view to the camera's, as a scaled axis (radians).
    rot: Vec3,
    /// The camera's view-space offset minus the lagging one's.
    trans: Vec3,
    /// The camera's eye minus the lagging one's, in blocks.
    pos: DVec3,
    /// The camera's pose on the previous frame.
    last: Pose,
}

impl LagCamera {
    fn at(cam: &Camera) -> LagCamera {
        LagCamera {
            rot: Vec3::ZERO,
            trans: Vec3::ZERO,
            pos: DVec3::ZERO,
            last: Pose::of(cam),
        }
    }

    /// Follow `cam` for `dt` seconds with time constant `exposure`, then clamp the trail.
    fn follow(&mut self, cam: &Camera, dt: f32, exposure: f32) {
        let now = Pose::of(cam);
        let x = dt / exposure.max(1e-4);
        let k = (-x).exp();
        // (1 - e^-x) / x, exact near 0 where the frame is much shorter than the exposure.
        let f = if x > 1e-6 { -(-x).exp_m1() / x } else { 1.0 };
        let mut step = now.rotation * self.last.rotation.inverse();
        if step.w < 0.0 {
            step = -step; // the short way round
        }
        self.rot = self.rot * k + step.to_scaled_axis() * f;
        self.trans = self.trans * k + (now.translation - self.last.translation) * f;
        self.pos = self.pos * f64::from(k) + (now.pos - self.last.pos) * f64::from(f);
        let angle = self.rot.length();
        if angle > MAX_LAG_ANGLE {
            self.rot *= MAX_LAG_ANGLE / angle;
        }
        let distance = self.pos.length();
        if distance > MAX_LAG_DISTANCE {
            self.pos *= MAX_LAG_DISTANCE / distance;
        }
        self.last = now;
    }

    fn view(&self) -> Mat4 {
        let rotation = Quat::from_scaled_axis(self.rot).inverse() * self.last.rotation;
        Mat4::from_rotation_translation(rotation, self.last.translation - self.trans)
    }

    fn pos(&self) -> DVec3 {
        self.last.pos - self.pos
    }
}

/// The CPU half of the pass: the camera bookkeeping behind [`ReprojectionUniform`]. Free of GPU
/// state so the maths is unit-tested on its own.
#[derive(Debug, Default)]
pub struct Reprojector {
    prev: Option<PrevCamera>,
    lag: Option<LagCamera>,
}

impl Reprojector {
    /// Forget the previous frame and the blur's trail (teleport, world change, resize).
    pub fn reset(&mut self) {
        self.prev = None;
        self.lag = None;
    }

    /// The uniform for this frame, then remember `cam` as the previous frame. `dt` is the time
    /// since the previous frame in seconds; `exposure` the blur's in seconds (0 = no blur).
    pub fn advance(
        &mut self,
        cam: &Camera,
        width: u32,
        height: u32,
        dt: f32,
        exposure: f32,
    ) -> ReprojectionUniform {
        let teleported = self
            .prev
            .is_some_and(|prev| (cam.pos - prev.pos).length() > TELEPORT_DISTANCE);
        if !dt.is_finite() || dt > MAX_FRAME_DT || teleported {
            self.reset();
        }
        let view_proj = cam.proj * cam.view;
        // f64 inverses: the projection's near/far ratio makes an f32 inverse lose bits.
        let inv_proj = cam.jittered_proj().as_dmat4().inverse().as_mat4();
        let inv_view = cam.view.as_dmat4().inverse().as_mat4();
        let (prev_view_proj, prev_delta, prev_jitter) = match self.prev {
            Some(prev) => (
                prev.view_proj,
                (cam.pos - prev.pos).as_vec3().extend(1.0),
                prev.jitter,
            ),
            None => (view_proj, Vec3::ZERO.extend(0.0), cam.jitter),
        };
        let (blur_view_proj, blur_delta) = if exposure > 0.0 {
            let lag = match self.lag.as_mut() {
                Some(lag) => {
                    lag.follow(cam, dt.max(0.0), exposure);
                    *lag
                }
                None => *self.lag.insert(LagCamera::at(cam)),
            };
            (
                cam.proj * lag.view(),
                (cam.pos - lag.pos()).as_vec3().extend(1.0),
            )
        } else {
            self.lag = None;
            (view_proj, Vec3::ZERO.extend(0.0))
        };
        self.prev = Some(PrevCamera {
            view_proj,
            pos: cam.pos,
            jitter: cam.jitter,
        });
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        ReprojectionUniform {
            inv_proj: inv_proj.to_cols_array_2d(),
            inv_view: inv_view.to_cols_array_2d(),
            view_proj: view_proj.to_cols_array_2d(),
            prev_view_proj: prev_view_proj.to_cols_array_2d(),
            blur_view_proj: blur_view_proj.to_cols_array_2d(),
            prev_delta: prev_delta.to_array(),
            blur_delta: blur_delta.to_array(),
            jitter: [cam.jitter.x, cam.jitter.y, prev_jitter.x, prev_jitter.y],
            screen: [w, h, 1.0 / w, 1.0 / h],
        }
    }
}

/// Pipelines and buffers, built on first use.
struct Pipelines {
    layout: wgpu::BindGroupLayout,
    camera: wgpu::RenderPipeline,
    objects: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    instances: wgpu::Buffer,
}

impl Pipelines {
    fn new(gpu: &Gpu) -> Pipelines {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post/velocity"),
            source: wgpu::ShaderSource::Wgsl(include_str!("velocity.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/velocity"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post/velocity"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let target = [Some(wgpu::ColorTargetState {
            format: VELOCITY_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let instance_attributes = wgpu::vertex_attr_array![
            0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4
        ];
        let instance_layout = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ObjectInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &instance_attributes,
        })];
        let make =
            |label: &str, vs: &str, fs: &str, buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(vs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers,
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &target,
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            };
        let camera = make("post/velocity camera", "vs_fullscreen", "fs_camera", &[]);
        let objects = make(
            "post/velocity objects",
            "vs_object",
            "fs_object",
            &instance_layout,
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post/velocity reprojection"),
            size: std::mem::size_of::<ReprojectionUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post/velocity objects"),
            size: (std::mem::size_of::<ObjectInstance>() * MAX_OBJECTS) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Pipelines {
            layout,
            camera,
            objects,
            uniform,
            instances,
        }
    }
}

/// The motion target and the bind group that reads this depth buffer.
struct Target {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// The depth texture `group` reads; a resize replaces it.
    depth: wgpu::Texture,
    group: wgpu::BindGroup,
}

/// Writes the motion target. See the [module docs](self).
pub struct VelocityPhase {
    consumers: VelocityConsumers,
    in_world: bool,
    msaa: bool,
    /// The blur exposure in seconds (0 = no blur trail).
    exposure: f32,
    objects: Vec<ObjectInstance>,
    /// The objects as passed, kept to scale by this frame's `dt` at record time.
    motions: Vec<ObjectMotion>,
    reprojector: Reprojector,
    fixed_dt: Option<f32>,
    last_record: Option<Instant>,
    pipelines: Option<Pipelines>,
    target: Option<Target>,
    output: VelocityOutput,
    generation: u64,
    frames: u64,
}

impl std::fmt::Debug for VelocityPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VelocityPhase")
            .field("consumers", &self.consumers)
            .field("in_world", &self.in_world)
            .field("msaa", &self.msaa)
            .field("exposure", &self.exposure)
            .field("objects", &self.motions.len())
            .field("target", &self.target.is_some())
            .field("frames", &self.frames)
            .finish_non_exhaustive()
    }
}

impl VelocityPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "post/velocity";

    /// An idle phase: no consumer, not in a world. Allocates nothing on the GPU.
    #[must_use]
    pub fn new(gpu: &Gpu) -> VelocityPhase {
        VelocityPhase {
            consumers: VelocityConsumers::default(),
            in_world: false,
            msaa: gpu.sample_count() > 1,
            exposure: 0.0,
            objects: Vec::new(),
            motions: Vec::new(),
            reprojector: Reprojector::default(),
            fixed_dt: None,
            last_record: None,
            pipelines: None,
            target: None,
            output: VelocityOutput::default(),
            generation: 0,
            frames: 0,
        }
    }

    /// The read handle for TAA, the motion blur or a shader-pack pass.
    #[must_use]
    pub fn output(&self) -> VelocityOutput {
        self.output.clone()
    }

    /// Who wants the target. With none the phase is off and frees its texture.
    pub fn set_consumers(&mut self, consumers: VelocityConsumers) {
        if consumers != self.consumers {
            self.consumers = consumers;
            self.release_unused();
        }
    }

    /// The current consumers.
    #[must_use]
    pub fn consumers(&self) -> VelocityConsumers {
        self.consumers
    }

    /// Whether a world is on screen. Coming back restarts the motion.
    pub fn set_in_world(&mut self, in_world: bool) {
        if in_world != self.in_world {
            self.in_world = in_world;
            self.release_unused();
        }
    }

    /// The motion-blur exposure in seconds (the lagging camera's time constant). 0 = no trail.
    pub fn set_exposure(&mut self, seconds: f32) {
        self.exposure = if seconds.is_finite() {
            seconds.max(0.0)
        } else {
            0.0
        };
    }

    /// This frame's moving entities (at most [`MAX_OBJECTS`] are used). Reuses its storage.
    pub fn set_objects(&mut self, objects: &[ObjectMotion]) {
        self.motions.clear();
        self.motions
            .extend_from_slice(&objects[..objects.len().min(MAX_OBJECTS)]);
    }

    /// Forget the previous frame and the blur trail: the next frame has no motion (teleport,
    /// world change).
    pub fn reset_history(&mut self) {
        self.reprojector.reset();
    }

    /// Use this frame time, in seconds, instead of the wall clock. For tests and benchmarks.
    pub fn set_fixed_dt(&mut self, dt: Option<f32>) {
        self.fixed_dt = dt;
    }

    /// Whether the pass runs this frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.in_world && !self.msaa && self.consumers.any()
    }

    /// Whether the motion target texture exists (only while the pass is active).
    #[must_use]
    pub fn target_allocated(&self) -> bool {
        self.target.is_some()
    }

    /// How many frames the pass recorded.
    #[must_use]
    pub fn frames_recorded(&self) -> u64 {
        self.frames
    }

    fn release_unused(&mut self) {
        if !self.is_active() {
            self.target = None;
            *self.output.0.borrow_mut() = None;
            self.reprojector.reset();
            self.last_record = None;
        }
    }

    fn prepare(&mut self, gpu: &Gpu) {
        let pipelines = self.pipelines.get_or_insert_with(|| Pipelines::new(gpu));
        let size = (gpu.config.width.max(1), gpu.config.height.max(1));
        let stale = self.target.as_ref().is_none_or(|target| {
            (target.width, target.height) != size || target.depth != *gpu.depth_texture()
        });
        if !stale {
            return;
        }
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post/velocity target"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VELOCITY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = gpu.depth_texture().clone();
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("post/velocity depth"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..wgpu::TextureViewDescriptor::default()
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post/velocity"),
            layout: &pipelines.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pipelines.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
            ],
        });
        self.target = Some(Target {
            width: size.0,
            height: size.1,
            texture,
            view,
            depth,
            group,
        });
        self.generation += 1;
        // A new size: the previous frame's motion would be in the wrong pixels' terms anyway.
        self.reprojector.reset();
    }

    fn frame_dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = self.fixed_dt.unwrap_or_else(|| {
            self.last_record.map_or(f32::INFINITY, |last| {
                now.saturating_duration_since(last).as_secs_f32()
            })
        });
        self.last_record = Some(now);
        dt
    }
}

impl RenderPhase for VelocityPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.is_active()
    }

    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {
        self.target = None;
        self.reprojector.reset();
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        if !self.is_active() {
            return;
        }
        self.prepare(gpu);
        let dt = self.frame_dt();
        let exposure = if self.consumers.blur {
            self.exposure
        } else {
            0.0
        };
        let target = self.target.as_ref().expect("built by prepare");
        let uniform = self
            .reprojector
            .advance(cam, target.width, target.height, dt, exposure);
        let has_previous = uniform.prev_delta[3] > 0.5;
        let pipelines = self.pipelines.as_ref().expect("built by prepare");
        gpu.write_buffer_staged(&pipelines.uniform, 0, bytemuck::bytes_of(&uniform));

        // Object motion: the frame's displacement for `rg` (none without a previous frame), the
        // exposure's for `ba`.
        let continuous = f32::from(u8::from(has_previous));
        let blur_objects = f32::from(u8::from(self.consumers.blur_objects));
        self.objects.clear();
        self.objects
            .extend(self.motions.iter().map(|m| ObjectInstance {
                min: m.min.extend(0.0).to_array(),
                max: m.max.extend(0.0).to_array(),
                moved: (m.frame_delta * continuous).extend(0.0).to_array(),
                moved_blur: (m.velocity * exposure).extend(blur_objects).to_array(),
            }));
        if !self.objects.is_empty() {
            gpu.write_buffer_staged(&pipelines.instances, 0, bytemuck::cast_slice(&self.objects));
        }

        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(Self::NAME),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // The camera pass writes every pixel.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &target.group, &[]);
        pass.set_pipeline(&pipelines.camera);
        pass.draw(0..3, 0..1);
        if !self.objects.is_empty() {
            pass.set_pipeline(&pipelines.objects);
            pass.set_vertex_buffer(0, pipelines.instances.slice(..));
            pass.draw(0..36, 0..self.objects.len() as u32);
        }
        drop(pass);

        self.frames += 1;
        *self.output.0.borrow_mut() = Some(VelocityFrame {
            texture: target.texture.clone(),
            view: target.view.clone(),
            uniform: pipelines.uniform.clone(),
            generation: self.generation,
            frame: self.frames,
            has_previous,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(yaw: f32, pitch: f32, pos: DVec3) -> Camera {
        let mut cam = Camera {
            yaw,
            pitch,
            pos,
            aspect: 16.0 / 9.0,
            ..Camera::default()
        };
        cam.update();
        cam
    }

    fn mat(m: [[f32; 4]; 4]) -> Mat4 {
        Mat4::from_cols_array_2d(&m)
    }

    /// `rg` for one surface point, on the CPU, the way `velocity.wgsl` computes it.
    fn motion_of(u: &ReprojectionUniform, p: Vec3) -> Vec2 {
        let uv = |clip: glam::Vec4| {
            let ndc = clip.truncate().truncate() / clip.w;
            Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5)
        };
        let now = uv(mat(u.view_proj) * p.extend(1.0));
        let before =
            uv(mat(u.prev_view_proj) * (p + Vec3::from_slice(&u.prev_delta[..3])).extend(1.0));
        now - before
    }

    /// `Camera::jittered_proj` (what the world phases draw with while TAA runs): the projection
    /// itself, bit for bit, at zero jitter; otherwise every point moves by exactly the jitter in
    /// NDC and keeps its depth.
    #[test]
    fn the_jittered_projection_moves_every_point_by_the_jitter() {
        let mut cam = camera(20.0, -10.0, DVec3::ZERO);
        assert_eq!(
            cam.jittered_proj(),
            cam.proj,
            "zero jitter is the projection itself"
        );
        cam.jitter = Vec2::new(0.25 / 960.0, -0.4 / 540.0);
        for p in [Vec3::new(1.0, 2.0, 8.0), Vec3::new(-3.0, -1.0, 40.0)] {
            let a = (cam.proj * cam.view).project_point3(p);
            let b = (cam.jittered_proj() * cam.view).project_point3(p);
            let moved = (b - a).truncate();
            assert!(
                (moved - cam.jitter).abs().max_element() < 1e-6,
                "{moved} vs {}",
                cam.jitter
            );
            assert!((b.z - a.z).abs() < 1e-7, "the depth changed");
        }
    }

    #[test]
    fn a_still_camera_has_no_motion_even_while_jittered() {
        let mut r = Reprojector::default();
        let mut cam = camera(30.0, 10.0, DVec3::new(100.0, 64.0, -50.0));
        cam.jitter = Vec2::new(0.3 / 960.0, -0.2 / 540.0);
        let first = r.advance(&cam, 1920, 1080, 1.0 / 240.0, 0.0);
        assert_eq!(first.prev_delta[3], 0.0, "no previous frame yet");
        cam.jitter = Vec2::new(-0.4 / 960.0, 0.1 / 540.0);
        let second = r.advance(&cam, 1920, 1080, 1.0 / 240.0, 0.0);
        assert_eq!(second.prev_delta[3], 1.0);
        for p in [Vec3::new(1.0, 2.0, 10.0), Vec3::new(-5.0, 0.5, -3.0)] {
            assert_eq!(motion_of(&second, p), Vec2::ZERO, "{p}");
        }
    }

    #[test]
    fn turning_moves_a_point_by_the_projected_angle() {
        let mut r = Reprojector::default();
        let pos = DVec3::new(0.5, 70.0, 0.5);
        r.advance(&camera(0.0, 0.0, pos), 1920, 1080, 0.01, 0.0);
        let u = r.advance(&camera(1.0, 0.0, pos), 1920, 1080, 0.01, 0.0);
        // A point straight ahead (+z, yaw 0 faces south) at the new view's centre.
        let ahead = mat(u.inv_view) * Vec3::new(0.0, 0.0, -20.0).extend(1.0);
        let m = motion_of(&u, ahead.truncate());
        // Turning right (yaw grows clockwise from above) by 1 degree: the point that is now in
        // the centre was right of it, so it moved left on screen, negative motion in x.
        let tan_half_x = (70.0f32.to_radians() / 2.0).tan() * 16.0 / 9.0;
        let expected = -(1.0f32.to_radians().tan() / tan_half_x) * 0.5;
        assert!((m.x - expected).abs() < 1e-4, "{m} vs {expected}");
        assert!(m.y.abs() < 1e-5, "{m}");
    }

    #[test]
    fn walking_moves_near_points_more_than_far_ones() {
        let mut r = Reprojector::default();
        r.advance(&camera(0.0, 0.0, DVec3::ZERO), 1920, 1080, 0.01, 0.0);
        // Strafe 0.1 blocks east (+x).
        let u = r.advance(
            &camera(0.0, 0.0, DVec3::new(0.1, 0.0, 0.0)),
            1920,
            1080,
            0.01,
            0.0,
        );
        let near = motion_of(&u, Vec3::new(0.0, 0.0, 2.0)).x.abs();
        let far = motion_of(&u, Vec3::new(0.0, 0.0, 50.0)).x.abs();
        assert!(near > 10.0 * far && far > 0.0, "{near} {far}");
    }

    #[test]
    fn a_teleport_or_a_long_frame_restarts_the_motion() {
        let mut r = Reprojector::default();
        r.advance(&camera(0.0, 0.0, DVec3::ZERO), 64, 64, 0.01, 0.0);
        let u = r.advance(
            &camera(0.0, 0.0, DVec3::new(20.0, 0.0, 0.0)),
            64,
            64,
            0.01,
            0.0,
        );
        assert_eq!(u.prev_delta[3], 0.0, "a 20-block jump is a teleport");
        let u = r.advance(
            &camera(0.0, 0.0, DVec3::new(20.0, 0.0, 0.0)),
            64,
            64,
            0.5,
            0.0,
        );
        assert_eq!(u.prev_delta[3], 0.0, "a half-second frame is too old");
        let u = r.advance(
            &camera(0.0, 0.0, DVec3::new(20.0, 0.0, 0.0)),
            64,
            64,
            0.01,
            0.0,
        );
        assert_eq!(u.prev_delta[3], 1.0);
    }

    /// Turning at a steady rate, the lagging camera settles `rate * exposure` behind at any
    /// frame rate: the trail is defined in time.
    #[test]
    fn the_blur_trail_is_the_same_length_at_60_and_1000_fps() {
        let exposure = 0.012;
        let rate = 200.0; // degrees per second
        let trail = |fps: f32| {
            let mut r = Reprojector::default();
            let dt = 1.0 / fps;
            let mut yaw = 0.0;
            let mut last = None;
            for _ in 0..(fps as usize) {
                yaw += rate * dt;
                last = Some(r.advance(&camera(yaw, 0.0, DVec3::ZERO), 1920, 1080, dt, exposure));
            }
            let u = last.unwrap();
            let lag_view = mat(u.blur_view_proj);
            let now = mat(u.view_proj);
            // Angle between the two views' forward axes.
            let forward = |m: Mat4| {
                let inv = m.inverse();
                let a = inv.project_point3(Vec3::new(0.0, 0.0, 0.5));
                let b = inv.project_point3(Vec3::new(0.0, 0.0, 0.9));
                (b - a).normalize()
            };
            forward(lag_view).angle_between(forward(now)).to_degrees()
        };
        let (slow, fast) = (trail(60.0), trail(1000.0));
        let expected = rate * exposure;
        assert!(
            (fast - expected).abs() < 0.15 * expected,
            "{fast} vs {expected}"
        );
        assert!(
            (slow - fast).abs() < 0.2 * expected,
            "60 fps {slow}, 1000 fps {fast}"
        );
    }

    #[test]
    fn the_wgsl_parses_and_validates() {
        let source = include_str!("velocity.wgsl");
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("velocity.wgsl validates");
    }

    #[test]
    fn the_uniform_matches_the_wgsl_layout() {
        // 5 matrices + 4 vectors, no padding.
        assert_eq!(std::mem::size_of::<ReprojectionUniform>(), 5 * 64 + 4 * 16);
        assert_eq!(std::mem::size_of::<ObjectInstance>(), 64);
    }
}
