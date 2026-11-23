//! Screen-space light rays: shafts from the sun (or moon) over a
//! sky mask, at quarter resolution, fading as the light leaves the screen. Usable by every pack.
//!
//! # Use
//!
//! 1. Each frame, find the light on screen with [`light_on_screen`]. `None` (behind the camera,
//!    or far enough off screen that nothing would show) means: record nothing, add nothing.
//! 2. Otherwise [`LightRays::record`] with [`RaysParams`]; it builds its three quarter-resolution
//!    `R16Float` targets on first use (and on a resize) and records three small passes.
//! 3. Read [`LightRays::output`] in the pack's composite with a linear sampler at the pixel's UV
//!    and add `rays * colour * strength * visibility` in linear light.
//!
//! The mask is "how much sky this pixel shows" (four depth taps, so the passes must run before the
//! hand clears the depth buffer, at an order below `PhaseOrder::FIRST_PERSON`) times a glow around
//! the light. Two radial blurs average it toward the light (`godrays.wgsl`). Nothing is
//! accumulated over frames, so there is no ghosting when the camera turns; with TAA on, the jitter
//! pattern moves every frame and TAA smooths it.
//!
//! # Cost
//!
//! At 1080p the targets are 480x270: three passes over 129,600 pixels, 4 depth loads for the
//! mask and 16 filtered taps for each blur. Nothing exists before [`LightRays::new`].

use glam::{Vec2, Vec3, Vec4};

use crate::{Camera, Gpu};

/// The targets' format.
pub const RAYS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// Full-resolution pixels per ray pixel, per axis.
pub const RAYS_DIVISOR: u32 = 4;

/// Where the light is on screen and how visible its rays are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightOnScreen {
    /// The light's position in UV (0..1 on screen, may be outside).
    pub uv: Vec2,
    /// 1 while the light is well on screen, falling to 0 as it leaves it.
    pub visibility: f32,
}

/// The light's screen position for the unjittered camera, or `None` when it is behind the camera
/// or so far off screen that its rays would not show.
#[must_use]
pub fn light_on_screen(cam: &Camera, dir_to_light: Vec3) -> Option<LightOnScreen> {
    let clip = cam.view_proj() * Vec4::new(dir_to_light.x, dir_to_light.y, dir_to_light.z, 0.0);
    if clip.w <= 1.0e-4 {
        return None;
    }
    let ndc = Vec2::new(clip.x, clip.y) / clip.w;
    let uv = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let edge = (uv - Vec2::splat(0.5)).abs().max_element();
    let t = ((edge - 0.55) / (1.0 - 0.55)).clamp(0.0, 1.0);
    let visibility = 1.0 - t * t * (3.0 - 2.0 * t);
    (visibility > 0.0).then_some(LightOnScreen { uv, visibility })
}

/// The rays' shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaysParams {
    /// The light's position in UV.
    pub light_uv: Vec2,
    /// Glow radius around the light, in units of the screen height.
    pub radius: f32,
    /// How far toward the light each pixel looks, 0..1 of the distance.
    pub reach: f32,
    /// Weight falloff per coarse tap (1 = even; lower favours the pixel's own end, shorter rays).
    pub decay: f32,
    /// Extra brightness of the light's disc in the mask.
    pub disc: f32,
    /// Jitter frame: 0 keeps one pattern, a changing value moves it (with TAA).
    pub jitter_frame: u32,
}

impl Default for RaysParams {
    fn default() -> Self {
        RaysParams {
            light_uv: Vec2::splat(0.5),
            radius: 0.35,
            reach: 0.85,
            decay: 0.97,
            disc: 1.0,
            jitter_frame: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct RaysUniform {
    sun: [f32; 4],
    params: [f32; 4],
    size: [f32; 4],
}

struct Targets {
    full: (u32, u32),
    size: (u32, u32),
    depth: wgpu::Texture,
    mask_view: wgpu::TextureView,
    a_view: wgpu::TextureView,
    b_view: wgpu::TextureView,
    mask_group: wgpu::BindGroup,
    coarse_group: wgpu::BindGroup,
    fine_group: wgpu::BindGroup,
}

/// The light rays' GPU half. See the [module docs](self).
pub struct LightRays {
    layout: wgpu::BindGroupLayout,
    mask: wgpu::RenderPipeline,
    coarse: wgpu::RenderPipeline,
    fine: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    uploaded: Option<RaysUniform>,
    targets: Option<Targets>,
    records: u64,
}

impl std::fmt::Debug for LightRays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LightRays")
            .field("targets", &self.targets.as_ref().map(|t| t.size))
            .field("records", &self.records)
            .finish_non_exhaustive()
    }
}

impl LightRays {
    /// The shader source.
    pub const WGSL: &'static str = include_str!("godrays.wgsl");

    /// Build the pipelines. Targets come on the first [`LightRays::record`].
    #[must_use]
    pub fn new(gpu: &Gpu) -> LightRays {
        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/light-rays"),
            source: wgpu::ShaderSource::Wgsl(Self::WGSL.into()),
        });
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/light-rays"),
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
                        view_dimension: wgpu::TextureViewDimension::D2,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaderpack/light-rays"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
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
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: RAYS_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let mask = pipeline("fs_mask");
        let coarse = pipeline("fs_blur_coarse");
        let fine = pipeline("fs_blur_fine");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shaderpack/light-rays"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shaderpack/light-rays"),
            size: std::mem::size_of::<RaysUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        LightRays {
            layout,
            mask,
            coarse,
            fine,
            sampler,
            uniform,
            uploaded: None,
            targets: None,
            records: 0,
        }
    }

    /// How many times the rays have been recorded.
    #[must_use]
    pub fn records(&self) -> u64 {
        self.records
    }

    /// Record the mask and both blurs for a frame of size `full`. Reads the shared depth buffer.
    pub fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        full: (u32, u32),
        params: &RaysParams,
    ) {
        let depth = gpu.depth_texture();
        let fits = self
            .targets
            .as_ref()
            .is_some_and(|t| t.full == full && t.depth == *depth);
        if !fits {
            self.targets = Some(self.make_targets(gpu, full));
        }
        let targets = self.targets.as_ref().expect("built above");
        let uniform = RaysUniform {
            sun: [
                params.light_uv.x,
                params.light_uv.y,
                params.radius.max(1.0e-3),
                full.0 as f32 / full.1.max(1) as f32,
            ],
            params: [
                params.reach.clamp(0.0, 1.0),
                params.decay.clamp(0.5, 1.0),
                params.jitter_frame as f32,
                params.disc.max(0.0),
            ],
            size: [
                targets.size.0 as f32,
                targets.size.1 as f32,
                full.0 as f32,
                full.1 as f32,
            ],
        };
        if self.uploaded != Some(uniform) {
            gpu.write_buffer_staged(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.uploaded = Some(uniform);
        }
        let passes = [
            (&self.mask, &targets.mask_group, &targets.mask_view),
            (&self.coarse, &targets.coarse_group, &targets.a_view),
            (&self.fine, &targets.fine_group, &targets.b_view),
        ];
        for (pipeline, group, view) in passes {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shaderpack/light-rays"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.records += 1;
    }

    fn make_targets(&self, gpu: &Gpu, full: (u32, u32)) -> Targets {
        let size = (
            full.0.div_ceil(RAYS_DIVISOR).max(1),
            full.1.div_ceil(RAYS_DIVISOR).max(1),
        );
        let make = |label: &str| {
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
                    format: RAYS_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let mask_view = make("shaderpack/light-rays mask");
        let a_view = make("shaderpack/light-rays a");
        let b_view = make("shaderpack/light-rays b");
        let depth = gpu.depth_texture().clone();
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        // Each pass reads a target it does not write: the mask pass binds `a` (unread).
        let group = |src: &wgpu::TextureView| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shaderpack/light-rays"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let mask_group = group(&a_view);
        let coarse_group = group(&mask_view);
        let fine_group = group(&a_view);
        Targets {
            full,
            size,
            depth,
            mask_view,
            a_view,
            b_view,
            mask_group,
            coarse_group,
            fine_group,
        }
    }

    /// The finished rays (one channel, average sky-glow toward the light) after
    /// [`LightRays::record`].
    #[must_use]
    pub fn output(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.b_view)
    }

    /// A linear, clamping sampler suitable for reading [`LightRays::output`].
    #[must_use]
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_parses_and_validates() {
        let source = LightRays::WGSL;
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("godrays.wgsl validates");
    }

    fn camera(yaw: f32, pitch: f32) -> Camera {
        let mut cam = Camera {
            yaw,
            pitch,
            ..Camera::default()
        };
        cam.update();
        cam
    }

    /// A light straight ahead is at the centre and fully visible; behind the camera there is no
    /// light; well off to the side the rays have faded out.
    #[test]
    fn the_light_is_found_on_screen_and_fades_off_it() {
        // Yaw 0 faces +Z.
        let ahead = light_on_screen(&camera(0.0, 0.0), Vec3::Z).expect("ahead");
        assert!((ahead.uv - Vec2::splat(0.5)).length() < 1e-3, "{ahead:?}");
        assert!((ahead.visibility - 1.0).abs() < 1e-6);
        assert!(light_on_screen(&camera(0.0, 0.0), -Vec3::Z).is_none());
        let side = light_on_screen(&camera(0.0, 0.0), Vec3::new(1.0, 0.0, 0.3).normalize());
        assert!(side.is_none_or(|light| light.visibility < 0.05), "{side:?}");
        // Looking up toward the noon sun.
        let up = light_on_screen(&camera(0.0, -80.0), Vec3::Y).expect("up");
        assert!(up.visibility > 0.9, "{up:?}");
    }
}
