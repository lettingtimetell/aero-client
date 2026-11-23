//! The window's picture while a video export renders offscreen: the latest exported frame scaled
//! to fit the window (letterboxed, slightly dimmed) with a progress panel on top.
//!
//! The export owns the frame graph at its own resolution (`Gpu::set_render_override`), so the
//! window cannot be drawn by the usual phases: this is a tiny compositor of its own, one pipeline,
//! no depth, no MSAA, drawing straight into the window's image.

use crate::Gpu;

const SHADER: &str = include_str!("preview.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Quad {
    rect: [f32; 4],
    tint: [f32; 4],
}

struct Bound {
    texture: wgpu::Texture,
    group: wgpu::BindGroup,
}

/// See the module docs.
pub struct PreviewCompositor {
    format: wgpu::TextureFormat,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    image_quad: wgpu::Buffer,
    panel_quad: wgpu::Buffer,
    image: Option<Bound>,
    panel: Option<(Bound, (u32, u32))>,
}

/// Where the panel goes and how the frame is shown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewLayout {
    /// Multiplies the frame's colour (1 = as rendered).
    pub dim: f32,
    /// The panel's top-left corner in window pixels, if a panel is shown.
    pub panel_at: Option<(f32, f32)>,
}

impl PreviewCompositor {
    /// A compositor drawing into images of `format` (the window's swap-chain format).
    #[must_use]
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> Self {
        let device = &gpu.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("export preview layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("export preview pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("export preview"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("export preview"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("export preview sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let quad = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Quad>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            format,
            layout,
            pipeline,
            sampler,
            image_quad: quad("export preview image quad"),
            panel_quad: quad("export preview panel quad"),
            image: None,
            panel: None,
        }
    }

    /// The window format this compositor draws into.
    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    fn group(&self, gpu: &Gpu, texture: &wgpu::Texture, quad: &wgpu::Buffer) -> wgpu::BindGroup {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("export preview group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: quad.as_entire_binding(),
                },
            ],
        })
    }

    /// Replace the progress panel's pixels (RGBA, display-encoded, straight alpha).
    pub fn set_panel(&mut self, gpu: &Gpu, width: u32, height: u32, rgba: &[u8]) {
        let (width, height) = (width.max(1), height.max(1));
        if rgba.len() < (width * height * 4) as usize {
            return;
        }
        let reuse = self
            .panel
            .as_ref()
            .is_some_and(|(_, size)| *size == (width, height));
        if !reuse {
            // Same sRGB-ness as the window, so sampling and writing cancel out.
            let format = if self.format.is_srgb() {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            };
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("export preview panel"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let group = self.group(gpu, &texture, &self.panel_quad);
            self.panel = Some((Bound { texture, group }, (width, height)));
        }
        if let Some((bound, _)) = &self.panel {
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &bound.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba[..(width * height * 4) as usize],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Draw the window's picture: `image` (`image_size`, the export's frame) letterboxed into
    /// `target` (`target_size`, the window's image), then the panel, if any, at
    /// `layout.panel_at`. Submits its own command buffer.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        image: &wgpu::Texture,
        image_size: (u32, u32),
        target: &wgpu::TextureView,
        target_size: (u32, u32),
        layout: PreviewLayout,
    ) {
        if self
            .image
            .as_ref()
            .is_none_or(|bound| &bound.texture != image)
        {
            let group = self.group(gpu, image, &self.image_quad);
            self.image = Some(Bound {
                texture: image.clone(),
                group,
            });
        }
        let (tw, th) = (target_size.0.max(1) as f32, target_size.1.max(1) as f32);
        let (iw, ih) = (image_size.0.max(1) as f32, image_size.1.max(1) as f32);
        let scale = (tw / iw).min(th / ih);
        let (w, h) = (iw * scale, ih * scale);
        let (x0, y0) = ((tw - w) * 0.5, (th - h) * 0.5);
        let to_clip = |x: f32, y: f32| [x / tw * 2.0 - 1.0, 1.0 - y / th * 2.0];
        let [ax, ay] = to_clip(x0, y0);
        let [bx, by] = to_clip(x0 + w, y0 + h);
        let dim = layout.dim;
        let image_quad = Quad {
            rect: [ax, ay, bx, by],
            tint: [dim, dim, dim, 1.0],
        };
        gpu.queue
            .write_buffer(&self.image_quad, 0, bytemuck::bytes_of(&image_quad));
        let panel = layout.panel_at.zip(self.panel.as_ref());
        if let Some(((px, py), (_, (pw, ph)))) = panel {
            let [ax, ay] = to_clip(px, py);
            let [bx, by] = to_clip(px + *pw as f32, py + *ph as f32);
            let panel_quad = Quad {
                rect: [ax, ay, bx, by],
                tint: [1.0; 4],
            };
            gpu.queue
                .write_buffer(&self.panel_quad, 0, bytemuck::bytes_of(&panel_quad));
        }
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("export preview"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("export preview"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            pass.set_pipeline(&self.pipeline);
            if let Some(bound) = &self.image {
                pass.set_bind_group(0, &bound.group, &[]);
                pass.draw(0..6, 0..1);
            }
            if let Some((_, (bound, _))) = panel {
                pass.set_bind_group(0, &bound.group, &[]);
                pass.draw(0..6, 0..1);
            }
        }
        gpu.queue.submit(Some(encoder.finish()));
    }
}

/// Read a single-sampled RGBA8/BGRA8 texture of `size` back as tightly packed RGBA (blocking; for
/// tests and one-off pictures, never per frame). Unlike `Gpu::read_pixels` it takes the size, so
/// it also reads the window's image while an export renders offscreen at another size.
#[must_use]
pub fn read_rgba(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> Option<Vec<u8>> {
    let (width, height) = size;
    let unpadded = width * 4;
    let padded =
        unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("export picture readback"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("export picture readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = tx.send(result);
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    rx.recv().ok()?.ok()?;
    let view = staging.get_mapped_range(..).ok()?;
    let mut out = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height as usize {
        let start = row * padded as usize;
        out.extend_from_slice(&view[start..start + unpadded as usize]);
    }
    drop(view);
    staging.unmap();
    if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in out.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    Some(out)
}
