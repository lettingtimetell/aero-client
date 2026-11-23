//! The shared world pass. Consecutive phases that opt in record into one render pass instead of one
//! each; the frame must be the same bytes, with and without MSAA, and a phase that does not opt
//! in must split the run without reordering anything.
//!
//! ```text
//! cargo test -p mc-render --test nw_f2_shared_pass -- --nocapture
//! ```

use std::cell::Cell;
use std::rc::Rc;

use mc_render::pipeline::PassTarget;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase};

const SIZE: u32 = 96;

fn headless(sample_count: u32) -> Option<Gpu> {
    match Gpu::headless(
        SIZE,
        SIZE,
        GpuOptions {
            vsync: false,
            sample_count,
            low_power: false,
        },
    ) {
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

/// One translucent, depth-tested triangle: which pixels it covers, its colour, its depth and
/// whether it writes depth all differ per phase, so the image depends on the draw order and on
/// the depth state each phase leaves for the next.
struct Tri {
    pipeline: wgpu::RenderPipeline,
    shares: bool,
    shared_calls: Rc<Cell<u32>>,
    own_calls: Rc<Cell<u32>>,
}

impl Tri {
    fn new(gpu: &Gpu, corners: [[f32; 2]; 3], rgba: [f32; 4], z: f32, write: bool) -> Self {
        let p = |c: [f32; 2]| format!("vec2<f32>({:.3}, {:.3})", c[0], c[1]);
        let src = format!(
            "@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
    var c = array<vec2<f32>, 3>({}, {}, {});
    return vec4<f32>(c[i], {z:.4}, 1.0);
}}
@fragment
fn fs() -> @location(0) vec4<f32> {{
    return vec4<f32>({:.3}, {:.3}, {:.3}, {:.3});
}}",
            p(corners[0]),
            p(corners[1]),
            p(corners[2]),
            rgba[0],
            rgba[1],
            rgba[2],
            rgba[3],
        );
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nw-f2 tri"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("nw-f2 tri"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("nw-f2 tri"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: mc_render::DEPTH_FORMAT,
                    depth_write_enabled: Some(write),
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
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        Self {
            pipeline,
            shares: true,
            shared_calls: Rc::new(Cell::new(0)),
            own_calls: Rc::new(Cell::new(0)),
        }
    }

    fn draw(&self, gpu: &Gpu, target: PassTarget<'_>) {
        let mut own = None;
        let pass = target.begin(gpu, "nw-f2 tri", &mut own);
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }
}

impl RenderPhase for Tri {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.own_calls.set(self.own_calls.get() + 1);
        self.draw(gpu, PassTarget::Own { enc, view });
    }

    fn name(&self) -> &'static str {
        "nw-f2 tri"
    }

    fn shares_world_pass(&self) -> bool {
        self.shares
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, _cam: &Camera) {
        self.shared_calls.set(self.shared_calls.get() + 1);
        self.draw(gpu, PassTarget::Shared(pass));
    }
}

/// One test triangle: corners, colour, depth, writes depth, shares the world pass.
type TriSpec = ([[f32; 2]; 3], [f32; 4], f32, bool, bool);

/// Renders the list: four sharing triangles, one that does not share in the middle, two more
/// sharing ones after it. Returns the pixels and (own, shared) record counts.
fn render(gpu: &mut Gpu, share: bool) -> (Vec<u8>, u32, u32) {
    let specs: [TriSpec; 7] = [
        (
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
            [0.9, 0.1, 0.1, 0.6],
            0.50,
            true,
            true,
        ),
        (
            [[-0.6, -1.0], [1.0, 0.2], [-1.0, 0.9]],
            [0.1, 0.8, 0.2, 0.5],
            0.40,
            false,
            true,
        ),
        (
            [[-1.0, 0.0], [1.0, -1.0], [0.6, 1.0]],
            [0.2, 0.2, 0.9, 0.7],
            0.45,
            true,
            true,
        ),
        (
            [[0.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            [0.9, 0.9, 0.1, 0.4],
            0.30,
            true,
            true,
        ),
        // Does not share: ends the run and starts its own pass.
        (
            [[-1.0, -0.5], [0.8, -1.0], [0.3, 1.0]],
            [0.6, 0.1, 0.8, 0.5],
            0.35,
            true,
            false,
        ),
        // Behind depth that earlier phases wrote in places, so the depth each run inherits shows.
        (
            [[-0.9, -0.9], [0.9, -0.2], [-0.2, 0.9]],
            [0.1, 0.9, 0.9, 0.6],
            0.38,
            false,
            true,
        ),
        (
            [[-1.0, 1.0], [1.0, 1.0], [0.0, -1.0]],
            [0.8, 0.5, 0.2, 0.5],
            0.47,
            true,
            true,
        ),
    ];
    let mut phases = PhaseList::new();
    phases.set_shared_world_pass(share);
    phases.clear_color = wgpu::Color {
        r: 0.05,
        g: 0.1,
        b: 0.2,
        a: 1.0,
    };
    let mut counters = Vec::new();
    for (i, (corners, rgba, z, write, shares)) in specs.into_iter().enumerate() {
        let mut tri = Tri::new(gpu, corners, rgba, z, write);
        tri.shares = shares;
        counters.push((Rc::clone(&tri.own_calls), Rc::clone(&tri.shared_calls)));
        phases.insert(PhaseOrder(200 + 10 * i as u32), Box::new(tri));
    }
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let own = counters.iter().map(|(own, _)| own.get()).sum();
    let shared = counters.iter().map(|(_, shared)| shared.get()).sum();
    (pixels, own, shared)
}

#[test]
fn a_shared_world_pass_draws_the_same_frame_as_a_pass_per_phase() {
    for samples in [1, 4] {
        let Some(mut gpu) = headless(samples) else {
            return;
        };
        let (separate, own, shared) = render(&mut gpu, false);
        assert_eq!(
            (own, shared),
            (7, 0),
            "sharing off: every phase records alone"
        );
        let (together, own, shared) = render(&mut gpu, true);
        assert_eq!(
            (own, shared),
            (1, 6),
            "sharing on: only the opted-out phase records alone"
        );
        let differing = separate
            .chunks_exact(4)
            .zip(together.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(differing, 0, "{samples}x MSAA: {differing} pixels differ");
        // Not vacuous: the triangles overlap and blend into many distinct colours.
        let mut colours: Vec<&[u8]> = separate.chunks_exact(4).collect();
        colours.sort_unstable();
        colours.dedup();
        assert!(
            colours.len() >= 12,
            "only {} distinct colours",
            colours.len()
        );
        eprintln!(
            "{samples}x MSAA: identical, {} distinct colours",
            colours.len()
        );
    }
}

/// With GPU timestamps on (`MC_GPU_TIMING=1`, which this test needs), a shared pass is timed as
/// one `world (shared)` scope: its timestamps go on the encoder before the pass opens and after
/// it ends, never inside it, and the frame is still the same bytes.
#[test]
fn gpu_timestamps_time_a_shared_pass_as_one_scope() {
    if !mc_render::context::gpu_timing_requested() {
        eprintln!("SKIP: set MC_GPU_TIMING=1");
        return;
    }
    let Some(mut gpu) = headless(1) else {
        return;
    };
    let (separate, _, _) = render(&mut gpu, false);
    let mut phases = PhaseList::new();
    phases.insert(
        PhaseOrder(200),
        Box::new(Tri::new(
            &gpu,
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
            [1.0; 4],
            0.5,
            true,
        )),
    );
    phases.insert(
        PhaseOrder(210),
        Box::new(Tri::new(
            &gpu,
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]],
            [0.5; 4],
            0.4,
            true,
        )),
    );
    for _ in 0..8 {
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
        let _ = gpu.read_pixels().expect("readback");
    }
    let timings = phases.gpu_timings();
    if timings.is_empty() {
        eprintln!("SKIP: the adapter granted no timestamp queries");
        return;
    }
    let names: Vec<&str> = timings.scopes.iter().map(|(name, _)| *name).collect();
    assert!(names.contains(&"world (shared)"), "scopes: {names:?}");
    assert!(
        !names.contains(&"nw-f2 tri"),
        "a phase inside the shared pass timed alone: {names:?}"
    );
    let (together, _, _) = render(&mut gpu, true);
    assert_eq!(separate, together, "timestamps changed the shared frame");
    eprintln!("scopes with sharing: {names:?}");
}
