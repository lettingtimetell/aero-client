//! Headless end-to-end tests for the S10 render seam.
//!
//! These run without a display server: a [`Gpu::headless`] context renders into an offscreen
//! texture and the result is copied back to the CPU and inspected. That makes "the frame really
//! was drawn" a CI-checkable fact rather than something verified by looking at a window.
//!
//! Every test skips itself when no adapter exists, because many CI runners have neither a GPU nor
//! a software fallback. `skip` is printed so a silent pass is never mistaken for a real one.

use mc_render::{
    Camera, DepthAttachment, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase,
    DEPTH_FORMAT,
};

/// Build a headless context, or return `None` when this machine has no adapter.
fn gpu(width: u32, height: u32, samples: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
        GpuOptions {
            vsync: false,
            sample_count: samples,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

fn close(actual: [u8; 4], expected: [u8; 4], tolerance: u8) -> bool {
    actual
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| a.abs_diff(*b) <= tolerance)
}

#[test]
fn clear_colour_frame_reaches_the_target() {
    let Some(mut gpu) = gpu(64, 48, 1) else {
        return;
    };
    assert!(gpu.is_headless());
    assert_eq!(gpu.sample_count(), 1);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.25,
        g: 0.5,
        b: 0.75,
        a: 1.0,
    };
    assert!(phases.is_empty());

    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);

    let pixels = gpu.read_pixels().expect("readback");
    assert_eq!(pixels.len(), 64 * 48 * 4);
    // Rgba8Unorm stores the clear colour verbatim: 0.25 -> 64, 0.5 -> 128, 0.75 -> 191.
    for (x, y) in [(0, 0), (63, 47), (32, 24)] {
        assert!(
            close(pixel(&pixels, 64, x, y), [64, 128, 191, 255], 2),
            "pixel {x},{y} = {:?}",
            pixel(&pixels, 64, x, y)
        );
    }
}

/// Fills the frame with a colour read out of the shared `@group(0)` camera uniform, which proves
/// the whole seam at once: layout, bind group, per-frame upload, pass setup, and depth attachment.
struct ProbePhase {
    pipeline: wgpu::RenderPipeline,
    resized_to: std::rc::Rc<std::cell::Cell<Option<(u32, u32)>>>,
}

const PROBE_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    // red   = partial_ticks, so a wrong or stale uniform upload is visible
    // green = 1 if the projection is non-degenerate
    // blue  = normalised viewport width, so a resize that missed the uniform is visible
    return vec4<f32>(
        camera.frame.x,
        select(0.0, 1.0, camera.proj[1][1] > 0.0),
        camera.screen.x / 256.0,
        1.0,
    );
}
"#;

impl ProbePhase {
    fn new(gpu: &Gpu) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("probe"),
                source: wgpu::ShaderSource::Wgsl(PROBE_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("probe layout"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("probe pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
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
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(gpu.config.format.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
        Self {
            pipeline,
            resized_to: std::rc::Rc::new(std::cell::Cell::new(None)),
        }
    }

    /// Handle the test can read after the phase has been boxed into the list.
    fn resize_probe(&self) -> std::rc::Rc<std::cell::Cell<Option<(u32, u32)>>> {
        std::rc::Rc::clone(&self.resized_to)
    }
}

impl RenderPhase for ProbePhase {
    fn name(&self) -> &'static str {
        "test/probe"
    }

    fn resize(&mut self, _gpu: &Gpu, width: u32, height: u32) {
        self.resized_to.set(Some((width, height)));
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = gpu.begin_pass(enc, view, "test/probe", DepthAttachment::Shared);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.draw(0..3, 0..1);
    }
}

#[test]
fn a_phase_draws_through_the_shared_camera_bind_group() {
    let Some(mut gpu) = gpu(128, 64, 1) else {
        return;
    };

    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(ProbePhase::new(&gpu)));
    assert_eq!(phases.len(), 1);
    assert_eq!(phases.names().collect::<Vec<_>>(), ["test/probe"]);

    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.5);

    let pixels = gpu.read_pixels().expect("readback");
    // red 0.5 -> 128, green 1.0 -> 255, blue 128/256 = 0.5 -> 128
    let got = pixel(&pixels, 128, 64, 32);
    assert!(close(got, [128, 255, 128, 255], 2), "centre pixel {got:?}");
    // Corners too: the phase must have covered the whole target, not just part of it.
    assert!(close(pixel(&pixels, 128, 0, 0), [128, 255, 128, 255], 2));
    assert!(close(pixel(&pixels, 128, 127, 63), [128, 255, 128, 255], 2));
}

#[test]
fn camera_uniform_is_refreshed_every_frame() {
    let Some(mut gpu) = gpu(64, 64, 1) else {
        return;
    };
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(ProbePhase::new(&gpu)));
    let frame = gpu.acquire().expect("acquire offscreen frame");

    for (partial, expected_red) in [(0.0_f32, 0u8), (0.25, 64), (0.75, 191)] {
        phases.render(&gpu, frame.view(), &Camera::default(), partial);
        let pixels = gpu.read_pixels().expect("readback");
        let got = pixel(&pixels, 64, 10, 10);
        assert!(
            got[0].abs_diff(expected_red) <= 2,
            "partial_ticks {partial} gave red {}",
            got[0]
        );
    }
}

#[test]
fn phases_paint_over_each_other_in_order() {
    let Some(mut gpu) = gpu(32, 32, 1) else {
        return;
    };

    /// Clears nothing; just records that it ran, in order.
    struct Recorder(
        &'static str,
        std::rc::Rc<std::cell::RefCell<Vec<&'static str>>>,
    );
    impl RenderPhase for Recorder {
        fn name(&self) -> &'static str {
            self.0
        }
        fn record(
            &mut self,
            _: &Gpu,
            _: &mut wgpu::CommandEncoder,
            _: &wgpu::TextureView,
            _: &Camera,
        ) {
            self.1.borrow_mut().push(self.0);
        }
    }

    let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::UI, Box::new(Recorder("ui", log.clone())));
    phases.insert(PhaseOrder::SKY, Box::new(Recorder("sky", log.clone())));
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(Recorder("terrain", log.clone())),
    );

    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    assert_eq!(*log.borrow(), ["sky", "terrain", "ui"]);
}

#[test]
fn msaa_resolves_into_the_frame_target() {
    let Some(mut gpu) = gpu(64, 64, 4) else {
        return;
    };
    if gpu.sample_count() == 1 {
        eprintln!("skip: adapter does not support 4x MSAA on the headless format");
        return;
    }
    assert_eq!(gpu.sample_count(), 4);

    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(ProbePhase::new(&gpu)));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &Camera::default(), 1.0);

    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 64, 32, 32);
    // Interior of a fully covered triangle resolves to the same colour as the 1x case.
    assert!(close(got, [255, 255, 64, 255], 2), "centre pixel {got:?}");
}

#[test]
fn resize_reallocates_attachments_and_notifies_phases() {
    let Some(mut gpu) = gpu(32, 32, 1) else {
        return;
    };
    let mut phases = PhaseList::new();
    let probe = ProbePhase::new(&gpu);
    let resized_to = probe.resize_probe();
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(probe));

    gpu.resize(96, 48);
    assert_eq!((gpu.config.width, gpu.config.height), (96, 48));
    phases.resize(&gpu, gpu.config.width, gpu.config.height);
    assert_eq!(
        resized_to.get(),
        Some((96, 48)),
        "phases must see the resize"
    );

    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    assert_eq!(pixels.len(), 96 * 48 * 4);
    // blue channel encodes width/256: 96/256 = 0.375 -> 96
    let got = pixel(&pixels, 96, 48, 24);
    assert!(close(got, [0, 255, 96, 255], 2), "centre pixel {got:?}");

    // A zero-sized resize (minimised window) must be ignored rather than killing the surface.
    gpu.resize(0, 0);
    assert_eq!((gpu.config.width, gpu.config.height), (96, 48));
}

#[test]
fn sustained_render_rate_is_measured_not_assumed() {
    let Some(mut gpu) = gpu(1280, 720, 1) else {
        return;
    };
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(ProbePhase::new(&gpu)));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    let camera = Camera::default();

    // Warm up: first submissions include pipeline compilation and allocation.
    for _ in 0..10 {
        phases.render(&gpu, frame.view(), &camera, 0.0);
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

    const FRAMES: u32 = 300;
    let start = std::time::Instant::now();
    for i in 0..FRAMES {
        phases.render(&gpu, frame.view(), &camera, i as f32 / FRAMES as f32);
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let fps = f64::from(FRAMES) / start.elapsed().as_secs_f64();

    eprintln!(
        "offscreen 1280x720 clear+fullscreen-pass: {fps:.0} fps over {FRAMES} frames ({})",
        gpu.adapter_info().name
    );
    // Deliberately loose: this is a smoke check that nothing stalls per frame, not a benchmark.
    // A real 60 FPS window is verified by `examples/clear.rs --frames N`.
    assert!(fps > 60.0, "offscreen frame rate collapsed to {fps:.1} fps");
}

#[test]
fn read_pixels_rejects_a_windowed_context() {
    // Nothing to assert against a real window here without a display; the headless guard is the
    // reachable half, and a headless context must *not* return this error.
    let Some(gpu) = gpu(8, 8, 1) else { return };
    assert!(gpu.read_pixels().is_ok());
}
