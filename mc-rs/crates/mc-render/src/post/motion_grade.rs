//! Motion blur and ASC CDL colour grading in one fullscreen pass.
//!
//! [`MotionGradePhase`] is the render half of two mods, `motion_blur` and `color_grading`. The
//! client pushes their settings in between frames ([`MotionGradePhase::set_blur_strength`],
//! [`MotionGradePhase::set_grade`], [`MotionGradePhase::set_in_world`]) and the phase does the
//! rest. Register it at [`crate::PhaseOrder::POST`] after the vanilla post chain: it then sees
//! the world, the hand, the screen overlays and any F4 shader group, and not the HUD, chat,
//! screens or the mod menu. Vanilla runs its shader group at the same point (after
//! `renderWorld`, before `renderGameOverlay`).
//!
//! This pass is the `motion_blur` mod's **Frame blend** mode. Its per-pixel modes (Camera,
//! Camera + objects) are [`super::velocity_blur`] over the
//! motion target [`super::velocity`], before the hand; the client then passes this phase a blur
//! strength of 0.
//!
//! # Off costs nothing
//!
//! With the blur off and the grade at the identity (or outside a world, or under MSAA) the phase
//! reports `enabled() == false` and `needs_scene_color() == false`. [`crate::PhaseList`] then
//! keeps the direct frame path byte for byte and never records this phase. No texture of the
//! phase exists then either: the two history textures live only while the blur is on in a world.
//! The three pipelines and the 64-byte uniform buffer are built on first use and kept.
//!
//! # Motion blur: frame feedback, defined in time
//!
//! ```text
//! a    = s^(120 · dt)        s = strength (0..=0.95), dt = seconds since the previous blend
//! out  = mix(curr, hist, a)
//! hist = out                 (un-graded, Rgba16Float, ping-pong)
//! ```
//!
//! A fixed per-frame feedback decays the history by `f^n` after `n` frames, so it looks
//! completely different at 60 and at 3000 FPS (a 50 % mix at 3000 FPS has a 0.33 ms half-life,
//! which is invisible). Deriving the weight from the frame time makes the trail a function of
//! time alone: strength `s` means "the per-frame feedback a 120 Hz player would set", with a
//! half-life of `ln 2 / (120 · ln(1/s))` seconds, 8.3 ms at the 50 % default.
//!
//! `dt` is wall-clock time between successive [`RenderPhase::record_scene`] calls. A frame
//! longer than [`MAX_BLUR_DT`] (a hitch, a minimised window) or a history that is not valid
//! (first frame after enabling, a resize, leaving the world, the blur having been off) shows the
//! scene sharp: `a = 0`.
//!
//! **Precision.** At 3000 FPS `1 - a` is 0.027 at the default and 0.002 at 95 %, so one frame's
//! step can be smaller than the history's storage spacing and be rounded away. An 8-bit history
//! leaves ghosts of up to 18 levels on screen forever; `Rgba16Float` narrows that band, and three
//! rules close it:
//! * the shader rounds the stored blend to fp16 itself, to nearest with ties to even, so the
//!   history target's write is exact. How a GPU rounds a float target write is up to the
//!   implementation; this machine's RTX 5070 Ti truncates, which held every rising trail about
//!   2/255 back at 3000 FPS. With exact values the trail is the same on every GPU;
//! * the convergence snap from the spec ([`snap_epsilon`]): a channel within epsilon of the
//!   scene lands on it exactly;
//! * a progress floor, which is not in the spec: a step smaller than one fp16 spacing moves one
//!   spacing instead. Without it a ghost of about 30 levels stays forever at high strength and
//!   frame rate, where the spec's capped snap (6/255) does not reach. `tests/m7_postfx.rs` shows
//!   all three on the GPU.
//!
//! # Colour grading: ASC CDL v1.2
//!
//! [`CdlParams::apply`] is the CPU reference and the shader runs the same six steps: slope,
//! offset, clamp to [0, 1], power, saturation around Rec.709 luma, clamp (the v1.2 forward
//! form). It grades the display-encoded frame, the surface's byte domain, which is the domain a
//! Rec.709 grade works in. With the blur on, the history keeps the un-graded
//! blend and only the displayed frame is graded, so the grade never compounds.
//!
//! # Pipelined submit
//!
//! The one per-frame upload, the uniform, goes through [`Gpu::write_buffer_staged`]: it travels
//! with the frame's own submission and never waits on the frame pipe's gate. Only this phase's
//! pass reads that buffer, as the upload ring's ordering contract requires.
//!
//! # MSAA
//!
//! The UI phase draws through the multisampled attachment and resolves over the frame, which
//! would erase this pass's output. So the phase stays off when [`Gpu::sample_count`] is above 1
//! (the startup-only `msaaSamples` option, 1 by default and in the user's options).

use std::time::Instant;

use crate::{Camera, Gpu, RenderPhase, SceneTextures};

/// The strongest blur the menu offers (95 %). Its half-life is 113 ms; past it the trail is a
/// smear.
pub const MAX_BLUR_STRENGTH: f32 = 0.95;

/// The rate, in Hz, that the blur strength is defined at: a strength is the per-frame feedback a
/// 120 Hz player would set. 120 is the geometric middle of 60 to 240 Hz.
pub const BLUR_REFERENCE_HZ: f32 = 120.0;

/// A frame longer than this, in seconds, shows the scene sharp and restarts the trail.
pub const MAX_BLUR_DT: f32 = 0.1;

/// The history's format. An 8-bit history would leave visible ghosts at high frame rates; see
/// the module docs.
pub const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The previous blend's weight for a frame that took `dt` seconds: `s^(120 · dt)`.
///
/// `strength` is a fraction (0.5 = 50 %), capped at [`MAX_BLUR_STRENGTH`]. A non-positive or
/// non-finite input gives 0, which shows the scene sharp.
#[must_use]
pub fn blur_weight(strength: f32, dt: f32) -> f32 {
    let usable = strength.is_finite() && strength > 0.0 && dt.is_finite() && dt >= 0.0;
    if !usable {
        return 0.0;
    }
    let s = f64::from(strength.min(MAX_BLUR_STRENGTH));
    s.powf(f64::from(BLUR_REFERENCE_HZ) * f64::from(dt)) as f32
}

/// The convergence snap for history weight `a`: a channel this close to the scene lands on it
/// exactly. `clamp(0.0003 / max(1 - a, 1e-4), 1/255, 6/255)`, from the research spec: `0.0003`
/// sits just above fp16's half-spacing below 1.0.
#[must_use]
pub fn snap_epsilon(a: f32) -> f32 {
    (0.0003 / (1.0 - a).max(1e-4)).clamp(1.0 / 255.0, 6.0 / 255.0)
}

/// One ASC CDL grade: slope, offset and power per channel (R, G, B) plus saturation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CdlParams {
    /// Per-channel multiplier, 1 = unchanged. Never negative.
    pub slope: [f32; 3],
    /// Per-channel addition after the slope, in full-scale units (0.2 = 51/255).
    pub offset: [f32; 3],
    /// Per-channel exponent after the clamp, 1 = unchanged. At least [`CdlParams::MIN_POWER`].
    pub power: [f32; 3],
    /// 0 = grey (Rec.709 luma), 1 = unchanged, 2 = double.
    pub saturation: f32,
}

impl Default for CdlParams {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl CdlParams {
    /// The grade that changes nothing. At the identity the pass is skipped.
    pub const IDENTITY: CdlParams = CdlParams {
        slope: [1.0; 3],
        offset: [0.0; 3],
        power: [1.0; 3],
        saturation: 1.0,
    };

    /// The smallest power uploaded: keeps `pow` defined and the grade finite.
    pub const MIN_POWER: f32 = 0.01;

    /// Rec.709 luma weights, the CDL's saturation axis.
    pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

    /// The effective grade from the menu's controls: the master slope and power scale every
    /// channel's slope and power, and the master offset is added to every channel's offset.
    #[must_use]
    pub fn from_controls(
        master_slope: f32,
        master_offset: f32,
        master_power: f32,
        saturation: f32,
        slope: [f32; 3],
        offset: [f32; 3],
        power: [f32; 3],
    ) -> CdlParams {
        CdlParams {
            slope: slope.map(|s| s * master_slope),
            offset: offset.map(|o| o + master_offset),
            power: power.map(|p| p * master_power),
            saturation,
        }
        .sanitized()
    }

    /// Whether this grade changes nothing. Exact comparison: the menu's sliders snap to their
    /// steps, so a grade at its defaults is exactly the identity.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// The grade made safe to upload: a non-finite number becomes the identity's, slope and
    /// saturation are at least 0, power is at least [`CdlParams::MIN_POWER`].
    #[must_use]
    pub fn sanitized(self) -> CdlParams {
        let finite_or = |value: f32, fallback: f32| {
            if value.is_finite() {
                value
            } else {
                fallback
            }
        };
        CdlParams {
            slope: self.slope.map(|s| finite_or(s, 1.0).max(0.0)),
            offset: self.offset.map(|o| finite_or(o, 0.0)),
            power: self.power.map(|p| finite_or(p, 1.0).max(Self::MIN_POWER)),
            saturation: finite_or(self.saturation, 1.0).max(0.0),
        }
    }

    /// The CPU reference: ASC CDL v1.2 on one display-encoded colour in [0, 1], in `f64`.
    ///
    /// 1. `v = in * slope`;
    /// 2. `v += offset`;
    /// 3. clamp to [0, 1] (NaN becomes 0);
    /// 4. `v = v^power`;
    /// 5. `v = luma + sat * (v - luma)`, with the Rec.709 luma of the post-power value;
    /// 6. clamp to [0, 1].
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let v: [f64; 3] = std::array::from_fn(|c| {
            let x = f64::from(rgb[c]) * f64::from(self.slope[c]) + f64::from(self.offset[c]);
            let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
            x.powf(f64::from(self.power[c]))
        });
        let luma: f64 = (0..3).map(|c| f64::from(Self::LUMA[c]) * v[c]).sum();
        let saturation = f64::from(self.saturation);
        std::array::from_fn(|c| (luma + saturation * (v[c] - luma)).clamp(0.0, 1.0) as f32)
    }
}

/// The pass's uniform. Field meanings are in `motion_grade.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    slope: [f32; 4],
    offset: [f32; 4],
    power: [f32; 4],
    misc: [f32; 4],
}

/// The phase's pipelines and uniform, built once for the frame's colour format.
struct Pipelines {
    format: wgpu::TextureFormat,
    scene_layout: wgpu::BindGroupLayout,
    history_layout: wgpu::BindGroupLayout,
    blur_grade: wgpu::RenderPipeline,
    blur_only: wgpu::RenderPipeline,
    grade_only: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
}

impl Pipelines {
    fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> Pipelines {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post/motion-grade"),
            source: wgpu::ShaderSource::Wgsl(include_str!("motion_grade.wgsl").into()),
        });
        // `textureLoad` only, so the textures need not be filterable.
        let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/motion-grade scene"),
            entries: &[
                texture(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let history_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/motion-grade history"),
            entries: &[texture(0)],
        });
        let grade_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post/motion-grade grade"),
            bind_group_layouts: &[Some(&scene_layout)],
            immediate_size: 0,
        });
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post/motion-grade blur"),
            bind_group_layouts: &[Some(&scene_layout), Some(&history_layout)],
            immediate_size: 0,
        });
        let frame_target = target(format);
        let blur_targets = [frame_target.clone(), target(HISTORY_FORMAT)];
        let blur_grade = pipeline(gpu, &shader, &blur_layout, "fs_blur_grade", &blur_targets);
        let blur_only = pipeline(gpu, &shader, &blur_layout, "fs_blur", &blur_targets);
        let grade_only = pipeline(gpu, &shader, &grade_layout, "fs_grade", &[frame_target]);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post/motion-grade params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Pipelines {
            format,
            scene_layout,
            history_layout,
            blur_grade,
            blur_only,
            grade_only,
            uniform,
        }
    }
}

fn target(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })
}

/// A colour attachment the pass overwrites completely. `Load` rather than `Clear`: on an
/// immediate-mode GPU it costs nothing, and it matches `PhaseList`'s own scene blit.
fn attachment(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
    })
}

fn pipeline(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry: &'static str,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            // Single-sampled: the phase is off under MSAA (module docs).
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets,
            }),
            multiview_mask: None,
            cache: None,
        })
}

/// The scene bind group, cached against the snapshot view it samples. `PhaseList` rebuilds its
/// snapshot only on a resize or format change, so this is rebuilt then and not per frame.
struct SceneBinding {
    /// A clone of the sampled view, compared by identity every frame. Holding it also keeps that
    /// view alive, so a new view can never reuse its address while it is compared against.
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

impl SceneBinding {
    fn new(gpu: &Gpu, pipelines: &Pipelines, view: &wgpu::TextureView) -> SceneBinding {
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post/motion-grade scene"),
            layout: &pipelines.scene_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pipelines.uniform.as_entire_binding(),
                },
            ],
        });
        SceneBinding {
            view: view.clone(),
            group,
        }
    }
}

/// The blur's ping-pong history: the frame reads `views[latest]` and writes the other one.
struct History {
    width: u32,
    height: u32,
    /// Each view keeps its texture alive.
    views: [wgpu::TextureView; 2],
    /// `groups[i]` samples `views[i]`.
    groups: [wgpu::BindGroup; 2],
    latest: usize,
}

impl History {
    fn new(gpu: &Gpu, pipelines: &Pipelines, width: u32, height: u32) -> History {
        let views: [wgpu::TextureView; 2] = std::array::from_fn(|_| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("post/motion-grade history"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HISTORY_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });
        let groups = std::array::from_fn(|i| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post/motion-grade history"),
                layout: &pipelines.history_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[i]),
                }],
            })
        });
        History {
            width,
            height,
            views,
            groups,
            latest: 0,
        }
    }
}

/// Motion blur plus ASC CDL grading over the world frame. See the [module docs](self).
pub struct MotionGradePhase {
    /// Blur strength as a fraction, `0..=MAX_BLUR_STRENGTH`. 0 = blur off.
    blur_strength: f32,
    grade: CdlParams,
    /// `!grade.is_identity()`, cached.
    grade_on: bool,
    in_world: bool,
    /// The context is multisampled: the phase never runs (module docs).
    msaa: bool,
    progress_floor: bool,
    fixed_dt: Option<f32>,
    last_blend: Option<Instant>,
    /// The history holds the previous frame's blend of this same scene.
    history_valid: bool,
    pipelines: Option<Pipelines>,
    scene_binding: Option<SceneBinding>,
    history: Option<History>,
    /// What the uniform buffer holds, so an unchanged frame uploads nothing.
    uploaded: Option<Params>,
    passes: u64,
}

impl std::fmt::Debug for MotionGradePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MotionGradePhase")
            .field("blur_strength", &self.blur_strength)
            .field("grade", &self.grade)
            .field("in_world", &self.in_world)
            .field("msaa", &self.msaa)
            .field("history", &self.history.is_some())
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl MotionGradePhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "post/motion-grade";

    /// An idle phase: blur off, grade at the identity, not in a world. Allocates nothing on the
    /// GPU; that happens on the first frame it is active.
    #[must_use]
    pub fn new(gpu: &Gpu) -> MotionGradePhase {
        MotionGradePhase {
            blur_strength: 0.0,
            grade: CdlParams::IDENTITY,
            grade_on: false,
            in_world: false,
            msaa: gpu.sample_count() > 1,
            progress_floor: true,
            fixed_dt: None,
            last_blend: None,
            history_valid: false,
            pipelines: None,
            scene_binding: None,
            history: None,
            uploaded: None,
            passes: 0,
        }
    }

    /// Set the blur strength as a fraction (0.5 = 50 %), clamped to `0..=`[`MAX_BLUR_STRENGTH`].
    /// 0 turns the blur off and frees its history. Turning it on restarts the trail.
    pub fn set_blur_strength(&mut self, strength: f32) {
        let strength = if strength.is_finite() {
            strength.clamp(0.0, MAX_BLUR_STRENGTH)
        } else {
            0.0
        };
        if strength != self.blur_strength {
            self.blur_strength = strength;
            self.release_unused();
        }
    }

    /// The blur strength as a fraction.
    #[must_use]
    pub fn blur_strength(&self) -> f32 {
        self.blur_strength
    }

    /// Set the colour grade. The identity turns the grade off.
    pub fn set_grade(&mut self, grade: CdlParams) {
        let grade = grade.sanitized();
        if grade != self.grade {
            self.grade = grade;
            self.grade_on = !grade.is_identity();
            self.release_unused();
        }
    }

    /// The colour grade.
    #[must_use]
    pub fn grade(&self) -> CdlParams {
        self.grade
    }

    /// Whether a world is on screen. Outside one (menus, loading and terrain screens) the phase
    /// is off, and coming back restarts the trail.
    pub fn set_in_world(&mut self, in_world: bool) {
        if in_world != self.in_world {
            self.in_world = in_world;
            self.release_unused();
        }
    }

    /// Whether the phase runs this frame: in a world, single-sampled, and blurring or grading.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.in_world && !self.msaa && (self.blur_on() || self.grade_on)
    }

    /// Use this frame time, in seconds, for every blend instead of the wall clock. For tests and
    /// benchmarks; `None` (the default) measures real time.
    pub fn set_fixed_dt(&mut self, dt: Option<f32>) {
        self.fixed_dt = dt;
    }

    /// How many passes this phase has recorded. Stays put while the phase is off.
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }

    /// Whether the blur's history textures exist (only while the blur is on in a world).
    #[must_use]
    pub fn history_allocated(&self) -> bool {
        self.history.is_some()
    }

    /// Switch the fp16 progress floor (module docs) off, to show what it prevents. Tests only.
    #[doc(hidden)]
    pub fn set_progress_floor(&mut self, on: bool) {
        self.progress_floor = on;
    }

    fn blur_on(&self) -> bool {
        self.blur_strength > 0.0
    }

    /// Drop what the current settings no longer need: the history once the blur cannot run
    /// (which also restarts the trail), the scene binding once nothing runs.
    fn release_unused(&mut self) {
        if !(self.blur_on() && self.is_active()) {
            self.history = None;
            self.history_valid = false;
        }
        if !self.is_active() {
            self.scene_binding = None;
        }
    }

    /// Build or rebuild what this frame needs: the pipelines for the frame's format, the scene
    /// bind group for this snapshot view, and (blur on) a history of the snapshot's size. A new
    /// history starts invalid, so its first frame shows the scene sharp.
    fn prepare(&mut self, gpu: &Gpu, scene: &SceneTextures<'_>) {
        if self
            .pipelines
            .as_ref()
            .is_none_or(|pipelines| pipelines.format != scene.format)
        {
            self.pipelines = Some(Pipelines::new(gpu, scene.format));
            // Everything below was built against the old pipelines' layouts and buffer.
            self.uploaded = None;
            self.scene_binding = None;
            self.history = None;
            self.history_valid = false;
        }
        let pipelines = self.pipelines.as_ref().expect("built above");
        if self
            .scene_binding
            .as_ref()
            .is_none_or(|binding| binding.view != *scene.color)
        {
            self.scene_binding = Some(SceneBinding::new(gpu, pipelines, scene.color));
        }
        let size = (scene.width, scene.height);
        if self.blur_on()
            && self
                .history
                .as_ref()
                .is_none_or(|history| (history.width, history.height) != size)
        {
            self.history = Some(History::new(gpu, pipelines, size.0, size.1));
            self.history_valid = false;
        }
    }

    /// The history weight for this frame, and the clock bookkeeping behind it.
    fn frame_weight(&mut self) -> f32 {
        let now = Instant::now();
        let dt = self.fixed_dt.unwrap_or_else(|| {
            self.last_blend.map_or(f32::INFINITY, |last| {
                now.saturating_duration_since(last).as_secs_f32()
            })
        });
        self.last_blend = Some(now);
        if self.history_valid && dt <= MAX_BLUR_DT {
            blur_weight(self.blur_strength, dt)
        } else {
            0.0
        }
    }
}

impl RenderPhase for MotionGradePhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.is_active()
    }

    fn needs_scene_color(&self) -> bool {
        self.is_active()
    }

    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {
        // A new size means a new snapshot and a history of the wrong size: restart the trail.
        self.history = None;
        self.history_valid = false;
        self.scene_binding = None;
    }

    fn record(
        &mut self,
        _gpu: &Gpu,
        _enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        // Only reachable when driven outside `PhaseList`: without a snapshot to read there is
        // nothing correct to draw, so draw nothing.
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        if !self.is_active() {
            return;
        }
        self.prepare(gpu, scene);
        let blur = self.blur_on();
        let a = if blur { self.frame_weight() } else { 0.0 };

        let pipelines = self.pipelines.as_ref().expect("built by prepare");
        let grade = self.grade;
        let params = Params {
            slope: [
                grade.slope[0],
                grade.slope[1],
                grade.slope[2],
                grade.saturation,
            ],
            offset: [grade.offset[0], grade.offset[1], grade.offset[2], a],
            power: [
                grade.power[0],
                grade.power[1],
                grade.power[2],
                snap_epsilon(a),
            ],
            misc: [f32::from(u8::from(self.progress_floor)), 0.0, 0.0, 0.0],
        };
        if self.uploaded != Some(params) {
            gpu.write_buffer_staged(&pipelines.uniform, 0, bytemuck::bytes_of(&params));
            self.uploaded = Some(params);
        }

        let scene_group = &self.scene_binding.as_ref().expect("built by prepare").group;
        if let (true, Some(history)) = (blur, self.history.as_mut()) {
            let read = history.latest;
            let write = 1 - read;
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(Self::NAME),
                color_attachments: &[attachment(view), attachment(&history.views[write])],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(if self.grade_on {
                &pipelines.blur_grade
            } else {
                &pipelines.blur_only
            });
            pass.set_bind_group(0, scene_group, &[]);
            pass.set_bind_group(1, &history.groups[read], &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            history.latest = write;
            self.history_valid = true;
        } else {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(Self::NAME),
                color_attachments: &[attachment(view)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.grade_only);
            pass.set_bind_group(0, scene_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.passes += 1;
    }
}
