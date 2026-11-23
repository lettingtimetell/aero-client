//! `MenuBlurPhase`, the menu blur, on a real GPU.
//!
//! Every test renders whole frames through a real [`PhaseList`] on a headless context and reads
//! them back, so what is asserted is what reaches the screen. They skip (printing `SKIP:`) when
//! the machine has no adapter.
//!
//! * Inactive: the frame is byte-identical to a frame without the phase, the list stays on its
//!   direct path, and the phase owns no GPU resource at all.
//! * Fade 0 is the sharp scene byte for byte; deactivating drops the textures, `release` the
//!   pipelines too.
//! * Active: stripes are smoothed and their mean brightness is kept.
//! * The measured spread of a blurred edge follows [`KawasePlan`]'s model at 1080p.
//! * Under 4x MSAA the blur survives a later phase's resolve.

use std::cell::RefCell;
use std::rc::Rc;

use mc_render::post::blur::{KawasePlan, MenuBlurPhase};
use mc_render::post::{OverlayKind, OverlayPhase};
use mc_render::{
    Camera, DepthAttachment, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase,
    SceneTextures, HEADLESS_FORMAT,
};

// ------------------------------------------------------------------------------- harness

fn headless(width: u32, height: u32, sample_count: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
        GpuOptions {
            vsync: false,
            sample_count,
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

/// The client's `Shared`: the list owns one handle, the test keeps the other.
#[derive(Clone)]
struct Handle(Rc<RefCell<MenuBlurPhase>>);

impl Handle {
    fn new(phase: MenuBlurPhase) -> Handle {
        Handle(Rc::new(RefCell::new(phase)))
    }

    fn with<R>(&self, f: impl FnOnce(&mut MenuBlurPhase) -> R) -> R {
        f(&mut self.0.borrow_mut())
    }
}

impl RenderPhase for Handle {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.0.borrow_mut().record(gpu, enc, view, cam);
    }

    fn name(&self) -> &'static str {
        MenuBlurPhase::NAME
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        self.0.borrow_mut().resize(gpu, width, height);
    }

    fn enabled(&self) -> bool {
        self.0.borrow().enabled()
    }

    fn needs_scene_color(&self) -> bool {
        self.0.borrow().needs_scene_color()
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        self.0.borrow_mut().record_scene(gpu, enc, view, cam, scene);
    }
}

/// Test scenes drawn over the clear. `EDGE` (the x where the white half ends) is prepended per
/// frame size.
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

// Exact bytes everywhere, thousands of colours: the byte-identity scene.
@fragment
fn fs_ramp(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);
    let r = x & 255u;
    let g = 255u - r;
    let b = (x * 37u + y * 101u) & 255u;
    return vec4<f32>(f32(r) / 255.0, f32(g) / 255.0, f32(b) / 255.0, 1.0);
}

// Vertical stripes four pixels wide, white on black.
@fragment
fn fs_stripes(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let on = f32((u32(pos.x) / 4u) % 2u);
    return vec4<f32>(on, on, on, 1.0);
}

// White left of EDGE, black right of it: the step whose blurred profile is the kernel's CDF.
@fragment
fn fs_edge(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let on = select(0.0, 1.0, pos.x < EDGE);
    return vec4<f32>(on, on, on, 1.0);
}
"#;

struct Pattern {
    pipeline: wgpu::RenderPipeline,
}

impl Pattern {
    fn new(gpu: &Gpu, entry: &'static str) -> Pattern {
        let source = format!(
            "const EDGE: f32 = {}.0;\n{PATTERN_WGSL}",
            gpu.config.width / 2
        );
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("x10 test pattern"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("x10 test pattern"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("x10 test pattern"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..wgpu::MultisampleState::default()
                },
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
            });
        Pattern { pipeline }
    }
}

impl RenderPhase for Pattern {
    fn name(&self) -> &'static str {
        "test/pattern"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        // Through `begin_pass`, like every real phase: under MSAA this draws the multisampled
        // attachment and resolves onto the frame.
        let mut pass = gpu.begin_pass(enc, view, "x10 test pattern", DepthAttachment::None);
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }
}

/// Render one frame of `phases` and read it back (RGBA8, top row first).
fn frame(gpu: &mut Gpu, phases: &mut PhaseList) -> Vec<u8> {
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

/// A list with `entry`'s pattern as the world and, when given, the blur at its real slot.
fn list(gpu: &Gpu, entry: &'static str, blur: Option<&Handle>) -> PhaseList {
    let mut phases = PhaseList::new();
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(Pattern::new(gpu, entry)),
    );
    if let Some(blur) = blur {
        phases.insert(PhaseOrder::MENU_BLUR, Box::new(blur.clone()));
    }
    phases
}

fn red(pixels: &[u8], width: u32, x: u32, y: u32) -> u8 {
    pixels[((y * width + x) * 4) as usize]
}

/// A phase whose radius is `radius_px` physical pixels on a frame `height` pixels high.
fn blur_at(radius_px: f32, height: u32, fade: f32) -> Handle {
    let mut phase = MenuBlurPhase::new();
    phase.set_radius(radius_px * 1080.0 / height as f32);
    phase.set_fade(fade);
    phase.set_active(true);
    Handle::new(phase)
}

// ------------------------------------------------------------------------------- tests

#[test]
fn an_inactive_blur_changes_no_byte_and_owns_nothing() {
    let Some(mut gpu) = headless(96, 64, 1) else {
        return;
    };
    let mut plain = list(&gpu, "fs_ramp", None);
    let reference = frame(&mut gpu, &mut plain);

    let blur = Handle::new(MenuBlurPhase::new());
    let mut with_blur = list(&gpu, "fs_ramp", Some(&blur));
    assert!(!with_blur.scene_mode(), "an idle blur must not opt in");
    let got = frame(&mut gpu, &mut with_blur);
    assert_eq!(got, reference, "an idle blur changed the frame");
    assert!(!blur.with(|b| b.targets_allocated()));
    assert!(!blur.with(|b| b.pipelines_built()));
    assert_eq!(blur.with(|b| b.passes_recorded()), 0);
}

#[test]
fn fade_zero_is_the_sharp_scene_and_turning_off_frees_the_textures() {
    let Some(mut gpu) = headless(96, 64, 1) else {
        return;
    };
    let mut plain = list(&gpu, "fs_ramp", None);
    let reference = frame(&mut gpu, &mut plain);

    let blur = blur_at(6.0, 64, 0.0);
    let mut phases = list(&gpu, "fs_ramp", Some(&blur));
    assert!(phases.scene_mode());
    let got = frame(&mut gpu, &mut phases);
    assert_eq!(got, reference, "fade 0 must be the scene itself");
    assert_eq!(blur.with(|b| b.passes_recorded()), 1);
    assert!(blur.with(|b| b.targets_allocated() && b.pipelines_built()));

    // The screen closes: the textures go at once, the pipelines stay for the next menu.
    blur.with(|b| b.set_active(false));
    assert!(!phases.scene_mode());
    assert!(!blur.with(|b| b.targets_allocated()));
    assert!(blur.with(|b| b.pipelines_built()));
    let after = frame(&mut gpu, &mut phases);
    assert_eq!(
        after, reference,
        "closing the menu must restore the frame exactly"
    );
    assert_eq!(
        blur.with(|b| b.passes_recorded()),
        1,
        "an idle phase records nothing"
    );

    // The mod is turned off: everything goes.
    blur.with(MenuBlurPhase::release);
    assert!(!blur.with(|b| b.pipelines_built()));
}

#[test]
fn an_active_blur_smooths_stripes_and_keeps_their_brightness() {
    let (width, height) = (256u32, 144u32);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    let mut plain = list(&gpu, "fs_stripes", None);
    let sharp = frame(&mut gpu, &mut plain);

    let blur = blur_at(12.0, height, 1.0);
    let mut phases = list(&gpu, "fs_stripes", Some(&blur));
    let blurred = frame(&mut gpu, &mut phases);

    let y = height / 2;
    let row = |pixels: &[u8]| -> Vec<u8> { (64..192).map(|x| red(pixels, width, x, y)).collect() };
    let (sharp_row, blurred_row) = (row(&sharp), row(&blurred));
    let spread =
        |row: &[u8]| i32::from(*row.iter().max().unwrap()) - i32::from(*row.iter().min().unwrap());
    assert_eq!(
        spread(&sharp_row),
        255,
        "the scene is full-contrast stripes"
    );
    assert!(
        spread(&blurred_row) <= 8,
        "4-pixel stripes under a 12-pixel blur must be nearly flat: {blurred_row:?}"
    );
    let mean = |row: &[u8]| row.iter().map(|&v| f32::from(v)).sum::<f32>() / row.len() as f32;
    assert!(
        (mean(&blurred_row) - mean(&sharp_row)).abs() < 4.0,
        "the blur must keep the mean: {} vs {}",
        mean(&blurred_row),
        mean(&sharp_row)
    );
    // Alpha is kept as the scene had it.
    assert!(blurred.chunks_exact(4).all(|p| p[3] == 255));
}

/// The standard deviation of the kernel that blurred a white-to-black step, from the step's
/// profile: the kernel is the profile's (negated) derivative.
fn measured_sigma(pixels: &[u8], width: u32, y: u32) -> f32 {
    let profile: Vec<f32> = (0..width)
        .map(|x| f32::from(red(pixels, width, x, y)) / 255.0)
        .collect();
    let kernel: Vec<f32> = profile.windows(2).map(|w| (w[0] - w[1]).max(0.0)).collect();
    let total: f32 = kernel.iter().sum();
    let mean: f32 = kernel
        .iter()
        .enumerate()
        .map(|(x, k)| x as f32 * k)
        .sum::<f32>()
        / total;
    let variance: f32 = kernel
        .iter()
        .enumerate()
        .map(|(x, k)| (x as f32 - mean).powi(2) * k)
        .sum::<f32>()
        / total;
    variance.sqrt()
}

#[test]
fn the_measured_spread_follows_the_plan_at_1080p() {
    let (width, height) = (1920u32, 1080u32);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    let mut worst: f32 = 0.0;
    for radius in [4.0f32, 8.0, 14.0, 24.0, 32.0, 48.0] {
        let blur = blur_at(radius, height, 1.0);
        let plan: KawasePlan = blur.with(|b| b.plan(width, height));
        let mut phases = list(&gpu, "fs_edge", Some(&blur));
        let pixels = frame(&mut gpu, &mut phases);
        let sigma = measured_sigma(&pixels, width, height / 2);
        let ratio = sigma / plan.sigma();
        eprintln!(
            "radius {radius:>4} px: {plan:?}, model sigma {:.2}, measured {:.2} ({:+.0} %)",
            plan.sigma(),
            sigma,
            (ratio - 1.0) * 100.0
        );
        worst = worst.max((ratio - 1.0).abs());
    }
    // Measured 2026-11-09 on an RTX 5070 Ti: within 3 % at every radius.
    assert!(
        worst < 0.10,
        "the model is off by {:.0} % somewhere",
        worst * 100.0
    );
}

/// Not a check: the blur's GPU time at 1920x1080 (needs `MC_GPU_TIMING=1`). Timings taken while
/// other work shares the GPU are provisional.
/// `MC_GPU_TIMING=1 cargo test -p mc-render --test x10_menu_blur -- --ignored gpu_cost`
#[test]
#[ignore = "benchmark; needs MC_GPU_TIMING=1"]
fn gpu_cost_at_1080p() {
    const FRAMES: u32 = 400;
    let (width, height) = (1920u32, 1080u32);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("SKIP: set MC_GPU_TIMING=1 (and the adapter needs timestamp queries)");
        return;
    }
    eprintln!("{width}x{height}, {FRAMES} frames per row, mean ms per frame:");
    eprintln!("radius | levels | post/menu-blur | gpu.scene_blit | gpu.frame");
    for radius in [0.0f32, 8.0, 20.0, 48.0] {
        let blur = blur_at(radius, height, 1.0);
        if radius == 0.0 {
            blur.with(|b| b.set_active(false));
        }
        let levels = blur.with(|b| b.plan(width, height).levels);
        let mut phases = list(&gpu, "fs_ramp", Some(&blur));
        for _ in 0..FRAMES {
            let frame = gpu.acquire().expect("acquire");
            phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
        }
        let _ = gpu.read_pixels();
        let timings = phases.gpu_timings();
        let mean_ms = |name: &str| {
            timings
                .scopes
                .iter()
                .find(|(scope, _)| *scope == name)
                .map_or(0.0, |(_, stats)| stats.mean_ns() / 1e6)
        };
        eprintln!(
            "{radius:>6} | {levels:>6} | {:>14.4} | {:>14.4} | {:>9.4}",
            mean_ms(MenuBlurPhase::NAME),
            mean_ms("gpu.scene_blit"),
            timings.frame_ns.mean_ns() / 1e6,
        );
    }
}

#[test]
fn the_blur_survives_msaa_and_a_later_resolve() {
    let (width, height) = (128u32, 64u32);
    let Some(mut gpu) = headless(width, height, 4) else {
        return;
    };
    if gpu.sample_count() == 1 {
        eprintln!("SKIP: adapter does not support 4x MSAA on this format");
        return;
    }
    let blur = blur_at(10.0, height, 1.0);
    let mut phases = list(&gpu, "fs_stripes", Some(&blur));
    // A later phase (where the open screen draws) whose own pass resolves the multisampled
    // attachment over the frame.
    let mut trailing = OverlayPhase::new(&gpu);
    trailing.gamma = 0.0;
    trailing.kind = OverlayKind::None;
    phases.insert(PhaseOrder::UI_SCREEN, Box::new(trailing));
    let pixels = frame(&mut gpu, &mut phases);
    let row: Vec<u8> = (32..96)
        .map(|x| red(&pixels, width, x, height / 2))
        .collect();
    let spread = i32::from(*row.iter().max().unwrap()) - i32::from(*row.iter().min().unwrap());
    assert!(
        spread <= 12,
        "the blur was resolved away under MSAA: {row:?}"
    );
}
