//! Frame blending, NV12 conversion and pipelined readback for a video export.
//!
//! Each rendered frame is handed to [`ExportCapture::capture`] (the client calls it from
//! `after_render`, with the offscreen target the frame was drawn into). With motion blur, the
//! frame is added to a float accumulator and only every `samples`-th call produces an output
//! frame (the mean). An output frame is converted to NV12 by a compute shader, copied into one of a
//! ring of mappable buffers and mapped asynchronously; [`ExportCapture::poll`] hands finished
//! frames out in order. The CPU waits only when every buffer of the ring is still in flight, so
//! the GPU never waits for the CPU.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::Gpu;

const SHADER: &str = include_str!("nv12.wgsl");

const FREE: u8 = 0;
const PENDING: u8 = 1;
const MAPPED: u8 = 2;
const FAILED: u8 = 3;

/// The uniform block `nv12.wgsl` reads.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    source_srgb: u32,
    samples: u32,
    first: u32,
    pad: [u32; 3],
}

struct Slot {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
}

/// What [`ExportCapture::capture`] did with a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Captured {
    /// Added to the motion-blur accumulator; the output frame is not complete yet.
    Accumulated,
    /// Output frame `index` was converted and is on its way back.
    Queued(u64),
}

/// Counters for the export's report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaptureStats {
    /// Frames handed to [`ExportCapture::capture`].
    pub rendered: u64,
    /// Output frames converted.
    pub queued: u64,
    /// Output frames read back and delivered.
    pub delivered: u64,
    /// Times `capture` had to wait for a readback buffer (the ring was full).
    pub ring_waits: u64,
    /// Readbacks that failed to map (their frames are lost).
    pub failed: u64,
}

/// See the module docs.
pub struct ExportCapture {
    width: u32,
    height: u32,
    samples: u32,
    sub: u32,
    source_srgb: bool,
    frame_bytes: usize,
    padded_bytes: u64,
    layout: wgpu::BindGroupLayout,
    convert: wgpu::ComputePipeline,
    accumulate: wgpu::ComputePipeline,
    /// The uniform both passes bind; each pass's values are copied into it in the command stream
    /// just before the pass (one submission runs both passes with different values, which
    /// `write_buffer` alone cannot do).
    params: wgpu::Buffer,
    accumulate_params: wgpu::Buffer,
    convert_params: wgpu::Buffer,
    accum: wgpu::Buffer,
    nv12: wgpu::Buffer,
    bound: Option<(wgpu::Texture, wgpu::BindGroup)>,
    ring: Vec<Slot>,
    next_frame: u64,
    next_deliver: u64,
    stats: CaptureStats,
}

impl ExportCapture {
    /// A capture for `width` x `height` frames (both even) of `format` (the offscreen target's),
    /// blending `samples` rendered frames into each output frame (1 = no blending), with
    /// `ring` readback buffers (at least 2).
    #[must_use]
    pub fn new(
        gpu: &Gpu,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        samples: u32,
        ring: usize,
    ) -> Self {
        let device = &gpu.device;
        let samples = samples.max(1);
        let frame_bytes = (width as usize) * (height as usize) * 3 / 2;
        let padded_bytes = (frame_bytes as u64).div_ceil(4) * 4;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("export nv12 layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(2),
                storage_entry(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("export nv12 pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("export nv12"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let convert = pipeline("convert");
        let accumulate = pipeline("accumulate");
        let params_buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Params>() as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let accum_size = if samples > 1 {
            u64::from(width) * u64::from(height) * 16
        } else {
            16
        };
        let accum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export blend accumulator"),
            size: accum_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let nv12 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export nv12"),
            size: padded_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ring = (0..ring.max(2))
            .map(|_| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("export readback"),
                    size: padded_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
            })
            .collect();
        Self {
            width,
            height,
            samples,
            sub: 0,
            source_srgb: format.is_srgb(),
            frame_bytes,
            padded_bytes,
            layout,
            convert,
            accumulate,
            params: params_buffer("export params", wgpu::BufferUsages::UNIFORM),
            accumulate_params: params_buffer(
                "export accumulate params",
                wgpu::BufferUsages::COPY_SRC,
            ),
            convert_params: params_buffer("export convert params", wgpu::BufferUsages::COPY_SRC),
            accum,
            nv12,
            bound: None,
            ring,
            next_frame: 0,
            next_deliver: 0,
            stats: CaptureStats::default(),
        }
    }

    /// [`Self::new`], refused instead of panicking when the GPU cannot hold the
    /// capture: a buffer over the device's limits (the blend accumulator is 16 bytes a pixel,
    /// 531 MB at 7680x4320, and is bound whole as a storage buffer), no memory for it, or a lost
    /// device (found as an export crash on an AMD GPU).
    ///
    /// # Errors
    ///
    /// What the GPU refused ([`crate::gpu_health::AllocError`]).
    pub fn try_new(
        gpu: &Gpu,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        samples: u32,
        ring: usize,
    ) -> Result<Self, crate::gpu_health::AllocError> {
        use crate::gpu_health::AllocError;
        if gpu.is_device_lost() {
            return Err(AllocError::DeviceLost);
        }
        let limits = gpu.device.limits();
        let nv12 = ((width as u64) * (height as u64) * 3 / 2).div_ceil(4) * 4;
        let accum = if samples > 1 {
            u64::from(width) * u64::from(height) * 16
        } else {
            16
        };
        let binding_limit = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        for requested in [nv12, accum] {
            if requested > binding_limit {
                return Err(AllocError::TooLarge {
                    requested,
                    limit: binding_limit,
                });
            }
        }
        let device = gpu.device.clone();
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let invalid = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let capture = Self::new(gpu, width, height, format, samples, ring);
        let invalid = pollster::block_on(invalid.pop());
        if let Some(error) = pollster::block_on(oom.pop()) {
            return Err(AllocError::OutOfMemory(error.to_string()));
        }
        if let Some(error) = invalid {
            return Err(AllocError::Invalid(error.to_string()));
        }
        Ok(capture)
    }

    /// Bytes of one NV12 output frame.
    #[must_use]
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// The output size.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Rendered frames blended into each output frame.
    #[must_use]
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Counters so far.
    #[must_use]
    pub fn stats(&self) -> CaptureStats {
        self.stats
    }

    /// Output frames converted but not yet delivered.
    #[must_use]
    pub fn in_flight(&self) -> u64 {
        self.next_frame - self.next_deliver
    }

    fn params(&self, samples: u32, first: bool) -> Params {
        Params {
            width: self.width,
            height: self.height,
            source_srgb: u32::from(self.source_srgb),
            samples,
            first: u32::from(first),
            pad: [0; 3],
        }
    }

    fn bind(&mut self, gpu: &Gpu, source: &wgpu::Texture) {
        if self
            .bound
            .as_ref()
            .is_some_and(|(bound, _)| bound == source)
        {
            return;
        }
        let view = source.create_view(&wgpu::TextureViewDescriptor::default());
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("export nv12 group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.accum.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.nv12.as_entire_binding(),
                },
            ],
        });
        self.bound = Some((source.clone(), group));
    }

    /// One rendered frame, drawn into `source` (`width` x `height`, the format given to
    /// [`Self::new`]). Returns once its GPU work is submitted; frames that become ready while
    /// waiting for a free readback buffer go to `sink(index, nv12)`.
    pub fn capture(
        &mut self,
        gpu: &Gpu,
        source: &wgpu::Texture,
        sink: &mut impl FnMut(u64, &[u8]),
    ) -> Captured {
        self.stats.rendered += 1;
        // On a lost device the readback buffers are destroyed: mapping one is a
        // validation panic. Nothing more is captured (the caller sees the loss and stops).
        if gpu.is_device_lost() {
            self.stats.failed += 1;
            return Captured::Accumulated;
        }
        self.bind(gpu, source);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("export capture"),
            });
        if self.samples > 1 {
            let first = self.sub == 0;
            let params = self.params(0, first);
            gpu.queue
                .write_buffer(&self.accumulate_params, 0, bytemuck::bytes_of(&params));
            encoder.copy_buffer_to_buffer(
                &self.accumulate_params,
                0,
                &self.params,
                0,
                std::mem::size_of::<Params>() as u64,
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("export accumulate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.accumulate);
                pass.set_bind_group(0, &self.bound.as_ref().expect("bound above").1, &[]);
                pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
            }
            self.sub += 1;
            if self.sub < self.samples {
                gpu.queue.submit(Some(encoder.finish()));
                return Captured::Accumulated;
            }
            self.sub = 0;
        }

        // A readback buffer for this output frame: the ring's oldest, once it is delivered.
        let slot_index = (self.next_frame % self.ring.len() as u64) as usize;
        if self.ring[slot_index].state.load(Ordering::Acquire) != FREE {
            self.stats.ring_waits += 1;
            while self.ring[slot_index].state.load(Ordering::Acquire) != FREE {
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                self.deliver_ready(sink, gpu.is_device_lost());
            }
        }

        let samples = if self.samples > 1 { self.samples } else { 0 };
        let params = self.params(samples, false);
        // `write_buffer` lands before the whole submission; the in-stream copy puts these values
        // in the bound uniform only after the accumulate pass has run with its own.
        gpu.queue
            .write_buffer(&self.convert_params, 0, bytemuck::bytes_of(&params));
        encoder.copy_buffer_to_buffer(
            &self.convert_params,
            0,
            &self.params,
            0,
            std::mem::size_of::<Params>() as u64,
        );
        let words = (self.frame_bytes as u32).div_ceil(4);
        let groups = words.div_ceil(64);
        let x = groups.min(65_535);
        let y = groups.div_ceil(x);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("export nv12"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.convert);
            pass.set_bind_group(0, &self.bound.as_ref().expect("bound above").1, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        let slot = &self.ring[slot_index];
        encoder.copy_buffer_to_buffer(&self.nv12, 0, &slot.buffer, 0, self.padded_bytes);
        gpu.queue.submit(Some(encoder.finish()));
        slot.state.store(PENDING, Ordering::Release);
        let state = Arc::clone(&slot.state);
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                state.store(
                    if result.is_ok() { MAPPED } else { FAILED },
                    Ordering::Release,
                );
            });
        let index = self.next_frame;
        self.next_frame += 1;
        self.stats.queued += 1;
        Captured::Queued(index)
    }

    /// Hand every finished frame to `sink`, in order, without waiting for the GPU.
    pub fn poll(&mut self, gpu: &Gpu, sink: &mut impl FnMut(u64, &[u8])) {
        if self.in_flight() == 0 {
            return;
        }
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        self.deliver_ready(sink, gpu.is_device_lost());
    }

    /// Wait for every frame still in flight and hand it to `sink`. A half-blended output frame
    /// at the end (fewer than `samples` rendered frames) is dropped.
    pub fn finish(&mut self, gpu: &Gpu, sink: &mut impl FnMut(u64, &[u8])) {
        self.sub = 0;
        while self.in_flight() > 0 {
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            self.deliver_ready(sink, gpu.is_device_lost());
        }
    }

    /// `lost`: the device is lost, so its buffers are destroyed. Every frame
    /// still in flight counts as failed without touching its buffer (an unmap would be a
    /// validation panic, and a map that never completes would wait forever).
    fn deliver_ready(&mut self, sink: &mut impl FnMut(u64, &[u8]), lost: bool) {
        while self.next_deliver < self.next_frame {
            let slot = &self.ring[(self.next_deliver % self.ring.len() as u64) as usize];
            if lost {
                self.stats.failed += 1;
                slot.state.store(FREE, Ordering::Release);
                self.next_deliver += 1;
                continue;
            }
            match slot.state.load(Ordering::Acquire) {
                MAPPED => {
                    {
                        let slice = slot.buffer.slice(..);
                        if let Ok(view) = slice.get_mapped_range() {
                            sink(self.next_deliver, &view[..self.frame_bytes]);
                        }
                    }
                    slot.buffer.unmap();
                    self.stats.delivered += 1;
                }
                FAILED => {
                    self.stats.failed += 1;
                }
                _ => return,
            }
            slot.state.store(FREE, Ordering::Release);
            self.next_deliver += 1;
        }
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
