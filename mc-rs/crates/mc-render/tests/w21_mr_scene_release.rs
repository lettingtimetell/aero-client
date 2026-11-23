//! The frame's scene snapshot is dropped after
//! `w21_mr::SCENE_RELEASE_IDLE_FRAMES` frames without a scene-reading phase and made again when
//! one returns, and the frames drawn after that are exactly what a fresh list draws.
//!
//! The reader is the real `MotionGradePhase` (blur off, a visible grade), which caches a bind
//! group over the snapshot view and rebuilds it when the view changes. It stays registered and
//! keeps that cache while gated off, and the scene changes while it is off, so a cache that kept
//! sampling the old snapshot would show the old picture. Skips (printing `SKIP:`) without an
//! adapter.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use mc_render::post::motion_grade::{CdlParams, MotionGradePhase};
use mc_render::w21_mr::{scene_release_enabled, SCENE_RELEASE_IDLE_FRAMES};
use mc_render::{
    Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase, SceneTextures,
    HEADLESS_FORMAT,
};

const SNAPSHOT: &str = "mc-render scene snapshot";

fn headless(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

/// Two full-screen patterns; `second` picks which one the next frame draws.
const PATTERN_WGSL: &str = r#"
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
fn fs_ramp(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let r = u32(pos.x) * 4u & 255u;
    let b = u32(pos.y) * 16u & 255u;
    return vec4<f32>(f32(r) / 255.0, f32(255u - r) / 255.0, f32(b) / 255.0, 1.0);
}

@fragment
fn fs_noise(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    var h = (u32(pos.x) * 73856093u) ^ (u32(pos.y) * 19349663u);
    h = (h ^ (h >> 16u)) * 0x45d9f3bu;
    h = h ^ (h >> 16u);
    let c = vec3<f32>(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u));
    return vec4<f32>(c / 255.0, 1.0);
}
"#;

struct Pattern {
    ramp: wgpu::RenderPipeline,
    noise: wgpu::RenderPipeline,
    second: Rc<Cell<bool>>,
}

impl Pattern {
    fn new(gpu: &Gpu, second: Rc<Cell<bool>>) -> Pattern {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("w21-mr test pattern"),
                source: wgpu::ShaderSource::Wgsl(PATTERN_WGSL.into()),
            });
        let pipeline = |entry: &str| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("w21-mr test pattern"),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(entry),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: HEADLESS_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        Pattern {
            ramp: pipeline("fs_ramp"),
            noise: pipeline("fs_noise"),
            second,
        }
    }
}

impl RenderPhase for Pattern {
    fn name(&self) -> &'static str {
        "test/pattern"
    }

    fn record(
        &mut self,
        _gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("w21-mr test pattern"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let pipeline = if self.second.get() {
            &self.noise
        } else {
            &self.ramp
        };
        pass.set_pipeline(pipeline);
        pass.draw(0..3, 0..1);
    }
}

/// The real grade phase behind a gate: gated off, it is disabled and reads no scene, but keeps
/// its state (and its cached bind group over the old snapshot view).
struct Gated {
    on: Rc<Cell<bool>>,
    inner: Rc<RefCell<MotionGradePhase>>,
}

impl RenderPhase for Gated {
    fn name(&self) -> &'static str {
        MotionGradePhase::NAME
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.inner.borrow_mut().record(gpu, enc, view, cam);
    }

    fn enabled(&self) -> bool {
        self.on.get() && self.inner.borrow().enabled()
    }

    fn needs_scene_color(&self) -> bool {
        self.on.get() && self.inner.borrow().needs_scene_color()
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        self.inner
            .borrow_mut()
            .record_scene(gpu, enc, view, cam, scene);
    }
}

/// A grade that visibly changes every pattern colour; no blur (so a frame depends only on the
/// scene drawn in it).
fn grade_phase(gpu: &Gpu) -> Rc<RefCell<MotionGradePhase>> {
    let mut phase = MotionGradePhase::new(gpu);
    phase.set_in_world(true);
    phase.set_blur_strength(0.0);
    phase.set_grade(CdlParams {
        slope: [0.7, 1.2, 0.9],
        offset: [0.05, -0.02, 0.1],
        power: [1.1, 0.9, 1.0],
        saturation: 0.8,
    });
    phase.set_fixed_dt(Some(1.0 / 60.0));
    Rc::new(RefCell::new(phase))
}

fn list(gpu: &Gpu, second: &Rc<Cell<bool>>, gate: &Rc<Cell<bool>>) -> PhaseList {
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(
        PhaseOrder::SKY,
        Box::new(Pattern::new(gpu, Rc::clone(second))),
    );
    phases.insert(
        PhaseOrder::POST,
        Box::new(Gated {
            on: Rc::clone(gate),
            inner: grade_phase(gpu),
        }),
    );
    phases
}

fn frame(gpu: &mut Gpu, phases: &mut PhaseList) -> Vec<u8> {
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    drop(frame);
    gpu.read_pixels().expect("readback")
}

fn snapshot_bytes(gpu: &Gpu) -> Option<u64> {
    let report = gpu.device.generate_allocator_report()?;
    Some(
        report
            .allocations
            .iter()
            .filter(|allocation| allocation.name == SNAPSHOT)
            .map(|allocation| allocation.size)
            .sum(),
    )
}

/// A scene reader that keeps nothing of the snapshot between frames (like the mod menu's
/// glass, which makes its bind group per frame).
struct KeepsNothing {
    on: Rc<Cell<bool>>,
}

impl RenderPhase for KeepsNothing {
    fn name(&self) -> &'static str {
        "test/scene-reader"
    }

    fn record(
        &mut self,
        _gpu: &Gpu,
        _enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
    }

    fn enabled(&self) -> bool {
        self.on.get()
    }

    fn needs_scene_color(&self) -> bool {
        self.on.get()
    }
}

fn settle(gpu: &Gpu) {
    let _ = gpu.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}

/// The memory really goes: with a reader that keeps nothing, the snapshot's allocation is gone
/// after the idle limit (and stays while the reader is on), and comes back with the reader.
#[test]
fn w21_mr_an_idle_snapshot_frees_its_memory_and_returns_on_demand() {
    let Some(mut gpu) = headless(64, 16) else {
        return;
    };
    let second = Rc::new(Cell::new(false));
    let on = Rc::new(Cell::new(true));
    let mut phases = PhaseList::new();
    phases.insert(
        PhaseOrder::SKY,
        Box::new(Pattern::new(&gpu, Rc::clone(&second))),
    );
    phases.insert(
        PhaseOrder::POST,
        Box::new(KeepsNothing { on: Rc::clone(&on) }),
    );

    for _ in 0..SCENE_RELEASE_IDLE_FRAMES + 8 {
        frame(&mut gpu, &mut phases);
    }
    settle(&gpu);
    let Some(held) = snapshot_bytes(&gpu) else {
        eprintln!("SKIP: no allocator report on this backend");
        return;
    };
    assert!(held > 0, "a reader that stays on keeps the snapshot");

    on.set(false);
    for _ in 0..SCENE_RELEASE_IDLE_FRAMES - 1 {
        frame(&mut gpu, &mut phases);
    }
    settle(&gpu);
    assert!(
        snapshot_bytes(&gpu).unwrap() > 0,
        "kept until the idle limit"
    );
    for _ in 0..4 {
        frame(&mut gpu, &mut phases);
    }
    settle(&gpu);
    let after = snapshot_bytes(&gpu).unwrap();
    if scene_release_enabled() {
        assert_eq!(after, 0, "the idle snapshot was released");
    } else {
        assert!(after > 0, "MC_W21_MR_SCENE_RELEASE=0 keeps the snapshot");
    }

    on.set(true);
    frame(&mut gpu, &mut phases);
    assert!(snapshot_bytes(&gpu).unwrap() > 0, "made again on demand");
}

/// A reader that caches a bind group over the snapshot view (the real grade phase) stays
/// registered through the idle period while the scene changes: the first frame after it returns
/// is exactly what a fresh list draws, so it never samples the old snapshot.
#[test]
fn w21_mr_a_remade_snapshot_draws_what_a_fresh_list_draws() {
    let Some(mut gpu) = headless(64, 16) else {
        return;
    };
    let second = Rc::new(Cell::new(false));
    let gate = Rc::new(Cell::new(true));
    let mut phases = list(&gpu, &second, &gate);
    let graded_ramp = frame(&mut gpu, &mut phases);

    gate.set(false);
    for _ in 0..SCENE_RELEASE_IDLE_FRAMES + 4 {
        frame(&mut gpu, &mut phases);
    }

    // The scene changes, the reader returns.
    second.set(true);
    gate.set(true);
    let after = frame(&mut gpu, &mut phases);

    let control_second = Rc::new(Cell::new(true));
    let control_gate = Rc::new(Cell::new(true));
    let mut control = list(&gpu, &control_second, &control_gate);
    let expected = frame(&mut gpu, &mut control);
    assert_eq!(
        after, expected,
        "the remade snapshot draws what a fresh list draws"
    );

    // Negative controls: the scene really changed, and the grade really reads it.
    assert_ne!(
        after, graded_ramp,
        "the two patterns differ after the grade"
    );
    control_gate.set(false);
    let ungraded = frame(&mut gpu, &mut control);
    assert_ne!(ungraded, expected, "the grade changes the picture");
}
