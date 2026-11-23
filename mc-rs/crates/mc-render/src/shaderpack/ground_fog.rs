//! Ground fog: a layer of fog that pools below a height, marched
//! from the depth buffer at half or quarter resolution and lit by the sun or moon.
//!
//! # The model
//!
//! The density is `density * profile(y)`: 1 below the layer's top, fading as `(1 - u)³` over the
//! softness above it (`u = (y - top) / softness`), and **exactly zero** from `top + softness` up.
//! (An exponential tail was the first version; it never reaches zero, so long near-horizontal rays
//! at a player's own height picked up a few percent of haze. Compact support makes "no fog above
//! this height" a guarantee, which the PvP tests rely on.) Each low-resolution pixel marches its
//! view ray from half the
//! start distance to the surface it hits, clipped to where the layer exists, in a few segments
//! that grow with distance. A sky ray (no surface: the void under the islands) is marched past the
//! far plane, down to where the layer is surely opaque, so the fog sea meets the horizon instead of
//! ending at the render distance. Under a resource pack's custom sky
//! ([`GroundFog::set_keep_sky`]) the sea stays off the sky as far as the pack's layers cover it
//! while the eye is above the fog, so the pack's sky shows at and below the horizon as it was
//! drawn. The optical
//! depth of each segment is integrated **exactly** for the base profile (`segment_tau` in
//! `ground_fog.wgsl`), so the fog's top never stair-steps however few segments there are; a baked
//! 64³ noise moves the top up and down (billows) and modulates the density (wisps), and drifts with
//! the wind. In-scattering is the ambient fog colour plus the light's Henyey-Greenstein forward
//! scatter, darker deep inside the layer. The output is the in-scattered light (linear,
//! premultiplied) and the transmittance, which the Dreamy composite upsamples bilaterally (with the
//! marched distance, so fog never bleeds onto a nearer surface) and applies as
//! `colour * T + L`.
//!
//! Nothing within the start distance is fogged (a smooth ramp from half of it to all of it), which
//! keeps fighting range clear wherever the layer is.
//!
//! # Cost
//!
//! At 1080p, half resolution is 518,400 rays of at most [`MAX_STEPS`] segments with one noise
//! fetch each; rays that never enter the layer (anything level with or above it) return before the
//! loop. The noise bake runs once, the first frame fog is drawn. Nothing exists before
//! [`GroundFog::new`], which the Dreamy pack calls the first time fog is on.
//!
//! # MSAA
//!
//! The depth binding is a plain `texture_depth_2d`, so this must never run with `msaaSamples`
//! above 1; the packs switch themselves off then.

use glam::{DVec3, Mat4, Vec3};

use crate::{Camera, Gpu};

/// Noise texels per axis.
pub const NOISE_SIZE: u32 = 64;
/// The noise's format: R = billows, G = wisps.
pub const NOISE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Blocks one noise tile spans horizontally (half that vertically).
pub const NOISE_TILE_BLOCKS: f64 = 160.0;
/// The fog target: rgb = in-scattered light (linear, premultiplied), a = transmittance.
pub const FOG_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The marched distance per low-resolution pixel, for the bilateral upsample.
pub const DIST_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// The distance the sky (no surface) is marched to and stored as, capped at the far plane.
pub const SKY_DISTANCE: f32 = 60_000.0;
/// Most segments per ray.
pub const MAX_STEPS: u32 = 24;

/// How many full-resolution pixels one fog pixel covers per axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FogResolution {
    /// Half resolution: a quarter of the pixels (the default; the fog's top stays crisp).
    #[default]
    Half,
    /// Quarter resolution: a sixteenth of the pixels.
    Quarter,
}

impl FogResolution {
    /// The divisor per axis.
    #[must_use]
    pub fn divisor(self) -> u32 {
        match self {
            FogResolution::Half => 2,
            FogResolution::Quarter => 4,
        }
    }
}

/// The fog layer's shape, in blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogLayer {
    /// World height where the density starts to fade.
    pub top: f32,
    /// Blocks over which the density fades to nothing above the top.
    pub softness: f32,
    /// Density inside the layer, per block (0.1 = 10 % of the light per block).
    pub density: f32,
    /// How far the noise moves the top up and down, in blocks (billows).
    pub billows: f32,
    /// How much the noise thins and thickens the fog, 0..1 (wisps).
    pub wisps: f32,
    /// No fog nearer than this along the view ray (a smooth ramp from half of it).
    pub start: f32,
    /// Segments per ray, `1..=`[`MAX_STEPS`].
    pub steps: u32,
    /// The march's resolution.
    pub resolution: FogResolution,
}

impl Default for FogLayer {
    fn default() -> Self {
        FogLayer {
            top: 62.0,
            softness: 6.0,
            density: 0.08,
            billows: 3.0,
            wisps: 0.5,
            start: 16.0,
            steps: 10,
            resolution: FogResolution::Half,
        }
    }
}

impl FogLayer {
    /// The transmittance of the base profile (the noise at its mean) between the eye at height
    /// `eye_y` and a point `rel` blocks away from it, start ramp included: the CPU reference the
    /// PvP tests use ("does fog hide a player at this distance?"). Exact up to the numeric
    /// integration of the ramp.
    #[must_use]
    pub fn transmittance(&self, eye_y: f32, rel: Vec3) -> f32 {
        let dist = rel.length();
        if dist <= 0.0 {
            return 1.0;
        }
        let dir = rel / dist;
        let start = self.start.max(0.0);
        let t0 = 0.5 * start;
        if dist <= t0 {
            return 1.0;
        }
        let steps = 4096;
        let mut tau = 0.0_f64;
        for i in 0..steps {
            let ta = t0 + (dist - t0) * i as f32 / steps as f32;
            let tb = t0 + (dist - t0) * (i + 1) as f32 / steps as f32;
            let tm = 0.5 * (ta + tb);
            let ramp = smoothstep(0.5 * start, start, tm);
            let seg = segment_tau(
                eye_y + dir.y * ta,
                dir.y,
                tb - ta,
                self.top,
                self.density,
                self.softness.max(0.25),
            );
            tau += f64::from(seg * ramp);
        }
        (-tau).exp() as f32
    }

    /// The highest point any fog can reach: the top raised by the billows, plus the softness.
    #[must_use]
    pub fn ceiling(&self) -> f32 {
        self.top + 0.5 * self.billows.max(0.0) + self.softness.max(0.25)
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `profile` of `ground_fog.wgsl`: 1 below `h`, `(1 - u)³` over the softness `s` above it, 0
/// from `h + s` up.
#[must_use]
pub fn profile(y: f32, h: f32, s: f32) -> f32 {
    let k = 1.0 - ((y - h) / s).clamp(0.0, 1.0);
    k * k * k
}

/// `segment_tau` of `ground_fog.wgsl`: the exact optical depth of one segment from height `y0`,
/// rising `v` per block for `len` blocks, through `density * profile(y)`.
#[must_use]
pub fn segment_tau(y0: f32, v: f32, len: f32, h: f32, density: f32, s: f32) -> f32 {
    if v.abs() * len < 1.0e-4 {
        return density * len * profile(y0 + 0.5 * v * len, h, s);
    }
    let y1 = y0 + v * len;
    let (lo, hi) = (y0.min(y1), y0.max(y1));
    let below = (hi.min(h) - lo).max(0.0);
    let a = 1.0 - ((lo - h) / s).clamp(0.0, 1.0);
    let b = 1.0 - ((hi - h) / s).clamp(0.0, 1.0);
    let fading = 0.25 * s * (a.powi(4) - b.powi(4));
    density * (below + fading) / v.abs()
}

/// The light on the fog this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogLight {
    /// Unit direction toward the light (the sun by day, the moon by night).
    pub dir: Vec3,
    /// The light's colour (linear) times its forward-scatter strength.
    pub color: Vec3,
    /// Henyey-Greenstein asymmetry, 0 = even, toward 1 = a tight glow around the light.
    pub g: f32,
    /// The fog's own colour (linear): the sky's light inside it.
    pub ambient: Vec3,
    /// How much darker the fog gets deep inside the layer, 0..1.
    pub depth_darkening: f32,
}

impl Default for FogLight {
    fn default() -> Self {
        FogLight {
            dir: Vec3::Y,
            color: Vec3::ZERO,
            g: 0.5,
            ambient: Vec3::splat(0.6),
            depth_darkening: 0.35,
        }
    }
}

/// `FogUniform` in `ground_fog.wgsl` (224 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct FogUniform {
    inv_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    layer: [f32; 4],
    shape: [f32; 4],
    wind: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient: [f32; 4],
    size: [f32; 4],
    misc: [f32; 4],
    eye_rel: [f32; 4],
}

/// Where the camera's eye really is in camera-relative space. Not the origin: the client prepends
/// local camera effects to `Camera::view` (vanilla `orientCamera`'s 0.1-block step back, and in
/// F5 the whole orbit offset), so rays and distances must start here. Reconstructing a sky
/// direction from the origin instead fogged the sky above a player looking up (night pictures).
#[must_use]
pub fn eye_relative(cam: &Camera) -> Vec3 {
    cam.view.inverse().transform_point3(Vec3::ZERO)
}

/// The inverse of the camera-relative view-projection the depth buffer was drawn with: jittered
/// while TAA jitters the world (`Camera::jitter`), so reconstructed positions match the depth.
#[must_use]
pub fn world_inverse_view_proj(cam: &Camera) -> Mat4 {
    let proj = if cam.jitter == glam::Vec2::ZERO {
        cam.proj
    } else {
        cam.jittered_proj()
    };
    (proj * cam.view).inverse()
}

/// `value` wrapped into `[0, period)`, in `f64` so a far-out world coordinate keeps its precision.
fn wrap(value: f64, period: f64) -> f32 {
    value.rem_euclid(period) as f32
}

/// The pipelines and samplers, built once.
struct Statics {
    noise_pipeline: wgpu::ComputePipeline,
    noise_layout: wgpu::BindGroupLayout,
    march_pipeline: wgpu::RenderPipeline,
    march_layout: wgpu::BindGroupLayout,
    noise_sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
}

/// The low-resolution targets and the bind group that reads the scene depth.
struct Targets {
    size: (u32, u32),
    full: (u32, u32),
    divisor: u32,
    depth: wgpu::Texture,
    fog_view: wgpu::TextureView,
    dist_view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

/// The ground fog's GPU half: noise, march pipeline and low-resolution targets. See the
/// [module docs](self).
pub struct GroundFog {
    statics: Statics,
    noise_view: wgpu::TextureView,
    noise_pending: Option<wgpu::BindGroup>,
    targets: Option<Targets>,
    layer: FogLayer,
    light: FogLight,
    wind: Vec3,
    jitter_frame: u32,
    /// How much of the sky below the horizon is a picture to keep, 0..1
    /// ([`GroundFog::set_keep_sky`]).
    keep_sky: f32,
    uploaded: Option<FogUniform>,
    marches: u64,
}

impl std::fmt::Debug for GroundFog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroundFog")
            .field("layer", &self.layer)
            .field("light", &self.light)
            .field("targets", &self.targets.as_ref().map(|t| t.size))
            .field("marches", &self.marches)
            .finish_non_exhaustive()
    }
}

impl GroundFog {
    /// The shader source.
    pub const WGSL: &'static str = include_str!("ground_fog.wgsl");

    /// Build the pipelines and the (not yet generated) noise texture.
    #[must_use]
    pub fn new(gpu: &Gpu) -> GroundFog {
        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/ground-fog"),
            source: wgpu::ShaderSource::Wgsl(Self::WGSL.into()),
        });
        let noise_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/ground-fog noise"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: NOISE_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            }],
        });
        let noise_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shaderpack/ground-fog noise"),
                bind_group_layouts: &[Some(&noise_layout)],
                immediate_size: 0,
            });
        let noise_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shaderpack/ground-fog noise"),
            layout: Some(&noise_pipeline_layout),
            module: &module,
            entry_point: Some("cs_noise"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let march_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/ground-fog march"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: fragment,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: fragment,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: fragment,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: fragment,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let march_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shaderpack/ground-fog march"),
                bind_group_layouts: &[Some(&march_layout)],
                immediate_size: 0,
            });
        let target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let march_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shaderpack/ground-fog march"),
            layout: Some(&march_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_march"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[target(FOG_FORMAT), target(DIST_FORMAT)],
            }),
            multiview_mask: None,
            cache: None,
        });
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shaderpack/ground-fog noise"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shaderpack/ground-fog"),
            size: std::mem::size_of::<FogUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let noise = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shaderpack/ground-fog noise"),
            size: wgpu::Extent3d {
                width: NOISE_SIZE,
                height: NOISE_SIZE,
                depth_or_array_layers: NOISE_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: NOISE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let noise_view = noise.create_view(&wgpu::TextureViewDescriptor::default());
        let noise_pending = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shaderpack/ground-fog noise"),
            layout: &noise_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&noise_view),
            }],
        }));
        GroundFog {
            statics: Statics {
                noise_pipeline,
                noise_layout,
                march_pipeline,
                march_layout,
                noise_sampler,
                uniform,
            },
            noise_view,
            noise_pending,
            targets: None,
            layer: FogLayer::default(),
            light: FogLight::default(),
            wind: Vec3::ZERO,
            jitter_frame: 0,
            keep_sky: 0.0,
            uploaded: None,
            marches: 0,
        }
    }

    /// Set the layer's shape.
    pub fn set_layer(&mut self, layer: FogLayer) {
        self.layer = layer;
    }

    /// The layer's shape.
    #[must_use]
    pub fn layer(&self) -> FogLayer {
        self.layer
    }

    /// Set this frame's light.
    pub fn set_light(&mut self, light: FogLight) {
        self.light = light;
    }

    /// Set the wind offset: how far the noise has drifted, in blocks (any size; wrapped here).
    pub fn set_wind(&mut self, offset: DVec3) {
        let tile = NOISE_TILE_BLOCKS;
        self.wind = Vec3::new(
            wrap(offset.x, tile),
            wrap(offset.y, tile * 0.5),
            wrap(offset.z, tile),
        );
    }

    /// Set the jitter pattern's frame: 0 keeps one fixed pattern (no TAA to average it), a
    /// changing value moves it every frame (TAA averages it into a smooth result).
    pub fn set_jitter_frame(&mut self, frame: u32) {
        self.jitter_frame = frame;
    }

    /// Keep the sky below the horizon as the sky pass drew it: how much of the sky a resource
    /// pack's custom sky layers cover, 0..1 (1 while one is fully drawn). The endless layer then
    /// covers the sky with only the rest of the fog sea while the eye is above the fog (fading in
    /// over the first blocks above its highest reach, so leaving the fog never pops), so the sea
    /// comes back smoothly as the layers fade out; terrain is fogged exactly as before. 0, the
    /// default, draws the whole fog sea to the horizon. Values outside 0..1 (or NaN) are clamped
    /// (NaN to 0).
    pub fn set_keep_sky(&mut self, keep: f32) {
        self.keep_sky = if keep.is_nan() {
            0.0
        } else {
            keep.clamp(0.0, 1.0)
        };
    }

    /// How many marches have been recorded.
    #[must_use]
    pub fn marches(&self) -> u64 {
        self.marches
    }

    /// Record this frame's fog: the noise bake (first frame only) and the march into the
    /// low-resolution targets. `full` is the frame's size.
    pub fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        cam: &Camera,
        full: (u32, u32),
    ) {
        if let Some(group) = self.noise_pending.take() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shaderpack/ground-fog noise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.statics.noise_pipeline);
            pass.set_bind_group(0, &group, &[]);
            let groups = NOISE_SIZE / 4;
            pass.dispatch_workgroups(groups, groups, groups);
        }
        let divisor = self.layer.resolution.divisor();
        let depth = gpu.depth_texture();
        let fits = self
            .targets
            .as_ref()
            .is_some_and(|t| t.full == full && t.divisor == divisor && t.depth == *depth);
        if !fits {
            self.targets = Some(self.make_targets(gpu, full, divisor));
        }
        let targets = self.targets.as_ref().expect("built above");

        let layer = self.layer;
        let light = self.light;
        let tile = NOISE_TILE_BLOCKS;
        let eye_rel = eye_relative(cam);
        let eye = cam.pos + eye_rel.as_dvec3();
        let uniform = FogUniform {
            inv_view_proj: world_inverse_view_proj(cam).to_cols_array_2d(),
            eye: [
                eye.y as f32,
                wrap(eye.x, tile),
                wrap(eye.y, tile * 0.5),
                wrap(eye.z, tile),
            ],
            layer: [
                layer.top,
                layer.softness.max(0.25),
                layer.density.max(0.0),
                layer.billows.max(0.0),
            ],
            shape: [
                layer.start.max(0.0),
                cam.far.max(64.0),
                (1.0 / tile) as f32,
                layer.wisps.clamp(0.0, 1.0),
            ],
            wind: [
                self.wind.x,
                self.wind.y,
                self.wind.z,
                layer.steps.clamp(1, MAX_STEPS) as f32,
            ],
            light_dir: [
                light.dir.x,
                light.dir.y,
                light.dir.z,
                light.g.clamp(0.0, 0.95),
            ],
            light_color: [light.color.x, light.color.y, light.color.z, 0.0],
            ambient: [
                light.ambient.x,
                light.ambient.y,
                light.ambient.z,
                light.depth_darkening.clamp(0.0, 1.0),
            ],
            size: [
                targets.size.0 as f32,
                targets.size.1 as f32,
                full.0 as f32,
                full.1 as f32,
            ],
            misc: [
                self.jitter_frame as f32,
                divisor as f32,
                SKY_DISTANCE,
                self.keep_sky,
            ],
            eye_rel: [eye_rel.x, eye_rel.y, eye_rel.z, 0.0],
        };
        if self.uploaded != Some(uniform) {
            gpu.write_buffer_staged(&self.statics.uniform, 0, bytemuck::bytes_of(&uniform));
            self.uploaded = Some(uniform);
        }

        let attachment = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shaderpack/ground-fog march"),
            color_attachments: &[
                attachment(&targets.fog_view),
                attachment(&targets.dist_view),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.statics.march_pipeline);
        pass.set_bind_group(0, &targets.group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        self.marches += 1;
    }

    fn make_targets(&self, gpu: &Gpu, full: (u32, u32), divisor: u32) -> Targets {
        let size = (
            full.0.div_ceil(divisor).max(1),
            full.1.div_ceil(divisor).max(1),
        );
        let make = |label: &str, format| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let fog_view = make("shaderpack/ground-fog fog", FOG_FORMAT);
        let dist_view = make("shaderpack/ground-fog distance", DIST_FORMAT);
        let depth = gpu.depth_texture().clone();
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shaderpack/ground-fog march"),
            layout: &self.statics.march_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.statics.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.statics.noise_sampler),
                },
            ],
        });
        Targets {
            size,
            full,
            divisor,
            depth,
            fog_view,
            dist_view,
            group,
        }
    }

    /// The fog target (rgb = in-scattered light, a = transmittance) after [`GroundFog::record`].
    #[must_use]
    pub fn fog_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.fog_view)
    }

    /// The marched-distance target after [`GroundFog::record`].
    #[must_use]
    pub fn dist_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.dist_view)
    }

    /// The low-resolution size and divisor after [`GroundFog::record`].
    #[must_use]
    pub fn low_size(&self) -> Option<((u32, u32), u32)> {
        self.targets.as_ref().map(|t| (t.size, t.divisor))
    }

    /// The noise bind group layout (kept so the struct owns everything it built).
    #[doc(hidden)]
    #[must_use]
    pub fn noise_layout(&self) -> &wgpu::BindGroupLayout {
        &self.statics.noise_layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_parses_and_validates() {
        let source = GroundFog::WGSL;
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("ground_fog.wgsl validates");
    }

    #[test]
    fn the_uniform_matches_the_wgsl_struct_size() {
        assert_eq!(std::mem::size_of::<FogUniform>(), 64 + 10 * 16);
    }

    /// The segment integral matches brute-force integration, crossing the top both ways, flat,
    /// steep, inside and above the layer.
    #[test]
    fn segment_tau_is_exact() {
        let (h, density, s) = (60.0, 0.1, 6.0);
        let cases = [
            (70.0, -0.5, 30.0),
            (50.0, 0.4, 40.0),
            (63.0, 0.0, 25.0),
            (55.0, -0.9, 10.0),
            (75.0, 0.2, 50.0),
            (61.0, -0.01, 80.0),
            (59.0, 0.01, 80.0),
            (52.0, 0.5, 30.0),
            (68.0, -0.05, 200.0),
        ];
        for (y0, v, len) in cases {
            let n = 200_000;
            let brute: f64 = (0..n)
                .map(|i| {
                    let t = (i as f32 + 0.5) / n as f32 * len;
                    let rho = density * profile(y0 + v * t, h, s);
                    f64::from(rho) * f64::from(len) / f64::from(n)
                })
                .sum();
            let exact = f64::from(segment_tau(y0, v, len, h, density, s));
            assert!(
                (exact - brute).abs() < 1e-3 * brute.max(1e-3),
                "y0 {y0} v {v} len {len}: {exact} vs {brute}"
            );
        }
    }

    /// The PvP guarantee: nothing above the ceiling is ever fogged, however far the ray runs.
    #[test]
    fn no_fog_above_the_ceiling() {
        let layer = FogLayer {
            top: 60.0,
            density: 0.5,
            ..FogLayer::default()
        };
        let ceiling = layer.ceiling();
        assert_eq!(
            segment_tau(ceiling - 0.5 * layer.billows, 0.0, 500.0, 60.0, 0.5, 6.0),
            0.0
        );
        // A horizontal ray just above the (billow-free) top's fade: exactly clear at any length.
        let eye = 60.0 + layer.softness + 0.01;
        for distance in [10.0_f32, 60.0, 200.0] {
            let t = layer.transmittance(eye, Vec3::new(distance, 0.0, 0.0));
            assert_eq!(t, 1.0, "{distance}");
        }
    }

    /// Fog inside the start distance is zero, and the ramp reaches full strength at it.
    #[test]
    fn nothing_is_fogged_inside_half_the_start_distance() {
        let layer = FogLayer {
            top: 100.0,
            density: 0.2,
            start: 20.0,
            ..FogLayer::default()
        };
        // The eye is deep inside the layer: everything is fog, except the near range.
        assert_eq!(layer.transmittance(80.0, Vec3::new(9.0, 0.0, 0.0)), 1.0);
        // 10..15 blocks: tau = 0.2 * 10 * (integral of smoothstep over 0..0.5 = 0.09375).
        let t15 = layer.transmittance(80.0, Vec3::new(15.0, 0.0, 0.0));
        assert!((t15 - (-0.1875_f32).exp()).abs() < 2e-3, "{t15}");
        // 40 blocks: tau = 0.2 * (5 for the ramp + 20 at full density) = 5.
        let t40 = layer.transmittance(80.0, Vec3::new(40.0, 0.0, 0.0));
        assert!((t40 - (-5.0_f32).exp()).abs() < 1e-3, "{t40}");
    }
}
