//! X14 measurement, not a regression test: what one render pass costs
//! in wgpu-core's `CommandEncoder::finish` + `Queue::submit` on this machine, against the same
//! draws recorded into one pass. Every `RenderPhase` opens its own pass through `Gpu::begin_pass`,
//! and `gpu.encoder_finish` is the largest main-thread scope in the SkyWars profile.
//!
//! Run by hand: `cargo test -p mc-render --test x14_pass_cost -- --ignored --nocapture`.
//! Timings are provisional on a busy machine; use the ratio, not the absolute numbers.

use std::time::Instant;

use mc_render::{DepthAttachment, Gpu, GpuOptions};

const WGSL: &str = r"
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i & 1u) * 0.01;
    let y = f32(i >> 1u) * 0.01;
    return vec4<f32>(x, y, 0.5, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 1.0, 1.0); }
";

fn pipeline(gpu: &Gpu) -> wgpu::RenderPipeline {
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("x14 pass cost"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("x14 pass cost"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: mc_render::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: gpu.sample_count(),
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
}

/// Mean CPU microseconds of `finish` and of `submit` for one frame of `passes` passes with
/// `draws_per_pass` draws each.
fn measure(
    gpu: &mut Gpu,
    pipe: &wgpu::RenderPipeline,
    passes: usize,
    draws_per_pass: usize,
) -> (f64, f64) {
    let frame = gpu.acquire().expect("offscreen frame");
    let mut finish_us = 0.0;
    let mut submit_us = 0.0;
    let rounds = 200;
    for round in 0..rounds + 20 {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for _ in 0..passes {
            let mut pass =
                gpu.begin_pass(&mut encoder, frame.view(), "x14", DepthAttachment::Shared);
            pass.set_pipeline(pipe);
            for _ in 0..draws_per_pass {
                pass.draw(0..3, 0..1);
            }
        }
        let start = Instant::now();
        let commands = encoder.finish();
        let finished = Instant::now();
        gpu.queue.submit([commands]);
        let submitted = Instant::now();
        if round >= 20 {
            finish_us += (finished - start).as_secs_f64() * 1e6;
            submit_us += (submitted - finished).as_secs_f64() * 1e6;
        }
        if round % 20 == 0 {
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
        }
    }
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    (finish_us / rounds as f64, submit_us / rounds as f64)
}

#[test]
#[ignore = "measurement; run by hand with --ignored --nocapture"]
fn render_pass_overhead_in_encoder_finish() {
    let mut gpu = Gpu::headless(1280, 720, GpuOptions::default()).expect("needs a GPU");
    let pipe = pipeline(&gpu);
    for (passes, draws) in [(1, 0), (1, 32), (32, 1), (1, 96), (24, 4), (96, 1)] {
        let (finish, submit) = measure(&mut gpu, &pipe, passes, draws);
        println!(
            "passes={passes:3} draws/pass={draws:3} total draws={:3}: finish {finish:8.1} us, submit {submit:8.1} us",
            passes * draws
        );
    }
}
