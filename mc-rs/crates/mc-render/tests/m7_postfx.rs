//! `MotionGradePhase` (motion blur + ASC CDL grade) on a real GPU.
//!
//! Every test renders whole frames through a real [`PhaseList`] on a headless context and reads
//! them back, so what is asserted is what reaches the screen. They skip (printing `SKIP:`) when
//! the machine has no adapter.
//!
//! * Off, at the identity, outside a world: the phase records nothing and the frame is
//!   byte-identical to a frame without it.
//! * The grade matches the ASC CDL v1.2 CPU reference ([`CdlParams::apply`]) on a ramp.
//! * The blur's step response depends on time, not on the frame rate.
//! * The history keeps the un-graded blend.
//! * At 95 % and 3000 FPS a slow tail converges exactly; without the progress floor a ghost
//!   stays on screen for good.
//! * `gpu_cost_at_1080p` (ignored): the GPU time at 1920x1080, with `MC_GPU_TIMING=1`.

use std::cell::RefCell;
use std::rc::Rc;

use mc_render::post::motion_grade::{blur_weight, snap_epsilon, CdlParams, MotionGradePhase};
use mc_render::{
    Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase, SceneTextures,
    HEADLESS_FORMAT,
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

/// The client's `Shared`: the list owns one handle, the test keeps the other and changes the
/// settings between frames.
#[derive(Clone)]
struct Handle(Rc<RefCell<MotionGradePhase>>);

impl Handle {
    /// A phase in a world, with the given blur strength and grade, on a fixed frame time.
    fn new(gpu: &Gpu, blur: f32, grade: CdlParams, dt: f32) -> Handle {
        let mut phase = MotionGradePhase::new(gpu);
        phase.set_in_world(true);
        phase.set_blur_strength(blur);
        phase.set_grade(grade);
        phase.set_fixed_dt(Some(dt));
        Handle(Rc::new(RefCell::new(phase)))
    }

    fn with<R>(&self, f: impl FnOnce(&mut MotionGradePhase) -> R) -> R {
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
        MotionGradePhase::NAME
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

/// A test scene drawn over the clear: `fs_ramp` is [`ramp`] (exact bytes, 4096 colours over
/// 256x16), `fs_noise` an integer hash that no framebuffer compression can shrink (benchmark).
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
    let x = u32(pos.x);
    let y = u32(pos.y);
    let r = x & 255u;
    let g = 255u - r;
    let b = (x * 37u + y * 101u) & 255u;
    return vec4<f32>(f32(r) / 255.0, f32(g) / 255.0, f32(b) / 255.0, 1.0);
}

@fragment
fn fs_noise(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    var h = (u32(pos.x) * 73856093u) ^ (u32(pos.y) * 19349663u);
    h = (h ^ (h >> 16u)) * 0x45d9f3bu;
    h = (h ^ (h >> 16u)) * 0x45d9f3bu;
    h = h ^ (h >> 16u);
    let c = vec3<f32>(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u));
    return vec4<f32>(c / 255.0, 1.0);
}
"#;

/// The ramp `fs_ramp` draws, on the CPU.
fn ramp(x: u32, y: u32) -> [u8; 3] {
    let r = x & 255;
    [r as u8, (255 - r) as u8, ((x * 37 + y * 101) & 255) as u8]
}

struct Pattern {
    pipeline: wgpu::RenderPipeline,
}

impl Pattern {
    fn new(gpu: &Gpu, entry: &'static str) -> Pattern {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("m7 test pattern"),
                source: wgpu::ShaderSource::Wgsl(PATTERN_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("m7 test pattern"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("m7 test pattern"),
                layout: Some(&layout),
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
        _gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("m7 test pattern"),
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
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }
}

/// A list holding an optional test pattern (at `SKY`) and an optional M7 phase (at `POST`, as the
/// client registers it).
fn list(gpu: &Gpu, pattern: Option<&'static str>, phase: Option<&Handle>) -> PhaseList {
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    if let Some(entry) = pattern {
        phases.insert(PhaseOrder::SKY, Box::new(Pattern::new(gpu, entry)));
    }
    if let Some(phase) = phase {
        phases.insert(PhaseOrder::POST, Box::new(phase.clone()));
    }
    phases
}

/// A clear colour that lands on exactly this byte in every channel.
fn grey(byte: u8) -> wgpu::Color {
    let v = f64::from(byte) / 255.0;
    wgpu::Color {
        r: v,
        g: v,
        b: v,
        a: 1.0,
    }
}

/// Record and submit one frame.
fn draw(gpu: &mut Gpu, phases: &mut PhaseList) {
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
}

/// One frame, read back as RGBA8 rows.
fn frame(gpu: &mut Gpu, phases: &mut PhaseList) -> Vec<u8> {
    draw(gpu, phases);
    gpu.read_pixels().expect("readback")
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// The colour of a frame whose pixels are all the same, or `None`.
fn uniform(pixels: &[u8]) -> Option<[u8; 4]> {
    let first = [pixels[0], pixels[1], pixels[2], pixels[3]];
    pixels
        .chunks_exact(4)
        .all(|px| px == first)
        .then_some(first)
}

// ------------------------------------------------------------- CPU model of the blur shader

/// fp16's spacing at `h`, as `ulp16` in `motion_grade.wgsl` computes it.
fn ulp16(h: f32) -> f32 {
    let exponent = ((h.abs().to_bits() >> 23) & 0xff).max(113);
    f32::from_bits((exponent - 10) << 23)
}

/// `x` stored in an `Rgba16Float` texel (round to nearest even; fp16 subnormals included).
fn to_fp16(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let spacing = ulp16(x);
    (x / spacing).round_ties_even() * spacing
}

/// WGSL's `sign`: 0 at 0.
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// One channel of the shader's `blend`: the frame's value (unrounded) for scene `curr`, history
/// `h`, weight `a`.
fn model_blend(curr: f32, h: f32, a: f32, floor: bool) -> f32 {
    let mut c = curr * (1.0 - a) + h * a;
    let d = curr - h;
    let u = if floor { ulp16(h) } else { 0.0 };
    if (c - h).abs() < u {
        c = h + sign(d) * u;
    }
    if (c - curr).abs() < snap_epsilon(a) || d.abs() <= u {
        c = curr;
    }
    c
}

/// A grey frame's journey through the blur on the CPU: the history (as stored, fp16) and the
/// byte each frame shows.
struct Model {
    history: Option<f32>,
    floor: bool,
}

impl Model {
    fn new(floor: bool) -> Model {
        Model {
            history: None,
            floor,
        }
    }

    /// One frame of scene byte `scene` at weight `a` (ignored on the first frame, which is
    /// sharp). Returns the byte on screen.
    fn frame(&mut self, scene: u8, a: f32) -> u8 {
        let curr = f32::from(scene) / 255.0;
        let c = match self.history {
            None => curr,
            Some(h) => model_blend(curr, h, a, self.floor),
        };
        self.history = Some(to_fp16(c));
        (c * 255.0).round() as u8
    }
}

// --------------------------------------------------------------------------------- tests

/// How this GPU rounds an `f32` written to an `Rgba16Float` colour target, which Vulkan leaves to
/// the implementation. Writes `0.5 + 0.75 * 2^-11`, three quarters of the way from fp16 `0x3800`
/// (0.5) to `0x3801`: round-to-nearest stores `0x3801`, truncation `0x3800`. Returns the stored
/// bits.
fn fp16_target_store(gpu: &Gpu, value: f32) -> u16 {
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("m7 fp16 probe"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
                         var c = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0),
                                                     vec2<f32>(3.0, 1.0));
                         return vec4<f32>(c[i], 0.0, 1.0);
                     }}
                     @fragment fn fs() -> @location(0) vec4<f32> {{
                         return vec4<f32>(bitcast<f32>({}u));
                     }}",
                    value.to_bits()
                )
                .into(),
            ),
        });
    let pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("m7 fp16 probe"),
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
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("m7 fp16 probe"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("m7 fp16 probe"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("m7 fp16 probe"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        size,
    );
    gpu.queue.submit(Some(enc.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("map the probe");
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let bytes = buffer.slice(..).get_mapped_range().expect("mapped");
    u16::from_le_bytes([bytes[0], bytes[1]])
}

/// Not a pass/fail property of the phase: it records which rounding this GPU uses, which is why
/// the shader rounds the history to fp16 itself (`to_fp16` in `motion_grade.wgsl`) instead of
/// leaving it to the target write.
#[test]
fn fp16_render_target_rounding_is_reported() {
    let Some(gpu) = headless(4, 4, 1) else {
        return;
    };
    let value = 0.5 + 0.75 * 2f32.powi(-11);
    let bits = fp16_target_store(&gpu, value);
    let mode = match bits {
        0x3801 => "round to nearest",
        0x3800 => "truncation (round toward zero)",
        _ => "unexpected",
    };
    eprintln!("fp16 target write of {value}: 0x{bits:04x} = {mode}");
    assert!(bits == 0x3800 || bits == 0x3801, "0x{bits:04x}");
}

/// The CPU reference the GPU test trusts, against hand-computed ASC CDL v1.2 values.
#[test]
fn the_cdl_reference_follows_asc_cdl_v1_2() {
    let near = |got: [f32; 3], want: [f32; 3]| {
        assert!(
            got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-4),
            "{got:?} != {want:?}"
        );
    };
    near(CdlParams::IDENTITY.apply([0.2, 0.5, 0.9]), [0.2, 0.5, 0.9]);
    // Clamp before power: 0.2 * 0.5 - 0.2 = -0.1 clamps to 0 (a squared -0.1 would be 0.01).
    let clamp_first = CdlParams {
        slope: [0.5; 3],
        offset: [-0.2; 3],
        power: [2.0; 3],
        saturation: 1.0,
    };
    near(clamp_first.apply([0.2, 0.6, 1.0]), [0.0, 0.01, 0.09]);
    // Saturation 0 is Rec.709 luma; saturation 2 needs the final clamp at both ends.
    let no_saturation = CdlParams {
        saturation: 0.0,
        ..CdlParams::IDENTITY
    };
    near(no_saturation.apply([1.0, 0.0, 0.0]), [0.2126; 3]);
    let vivid = CdlParams {
        saturation: 2.0,
        ..CdlParams::IDENTITY
    };
    // luma = 0.2126 * 0.8 + 0.7152 * 0.4 + 0.0722 * 0.2 = 0.4706
    near(vivid.apply([0.8, 0.4, 0.2]), [1.0, 0.3294, 0.0]);
    // The masters scale (slope, power) or shift (offset) every channel.
    let mixed = CdlParams::from_controls(
        2.0,
        0.1,
        0.5,
        1.0,
        [0.5, 1.0, 1.0],
        [0.0, 0.0, -0.1],
        [1.0, 2.0, 1.0],
    );
    near(mixed.slope, [1.0, 2.0, 2.0]);
    near(mixed.offset, [0.1, 0.1, 0.0]);
    near(mixed.power, [0.5, 1.0, 0.5]);
    // Unsafe inputs become safe ones; the identity survives a negative-zero offset.
    let wild = CdlParams {
        slope: [f32::NAN, -1.0, 2.0],
        offset: [f32::INFINITY, 0.1, 0.0],
        power: [0.0, f32::NAN, 3.0],
        saturation: -1.0,
    }
    .sanitized();
    near(wild.slope, [1.0, 0.0, 2.0]);
    near(wild.offset, [0.0, 0.1, 0.0]);
    near(wild.power, [CdlParams::MIN_POWER, 1.0, 3.0]);
    assert_eq!(wild.saturation, 0.0);
    let defaults = CdlParams::from_controls(1.0, -0.0, 1.0, 1.0, [1.0; 3], [-0.0; 3], [1.0; 3]);
    assert!(defaults.is_identity());
    assert!(!vivid.is_identity());
}

#[test]
fn the_blur_weight_and_snap_follow_the_spec() {
    // research/motion_blur.md, "per-frame weight a at the default (s = 0.5)".
    for (fps, want) in [
        (60.0, 0.25),
        (144.0, 0.561),
        (240.0, 0.707),
        (1000.0, 0.920),
        (3000.0, 0.973),
    ] {
        let a = blur_weight(0.5, 1.0 / fps);
        assert!((a - want).abs() < 0.001, "{fps} FPS: a = {a}, want {want}");
    }
    // The cap, and inputs that must show the scene sharp.
    assert_eq!(
        blur_weight(1.0, 1.0 / 120.0),
        blur_weight(0.95, 1.0 / 120.0)
    );
    assert_eq!(blur_weight(0.0, 0.01), 0.0);
    assert_eq!(blur_weight(f32::NAN, 0.01), 0.0);
    assert_eq!(blur_weight(0.5, f32::NAN), 0.0);
    // The snap stays within [1/255, 6/255].
    assert_eq!(snap_epsilon(0.0), 1.0 / 255.0);
    assert_eq!(snap_epsilon(0.999), 6.0 / 255.0);
}

#[test]
fn off_identity_or_outside_a_world_records_nothing_and_changes_no_byte() {
    let Some(mut gpu) = headless(256, 16, 1) else {
        return;
    };
    let mut bare = list(&gpu, Some("fs_ramp"), None);
    let baseline = frame(&mut gpu, &mut bare);
    for y in 0..16 {
        for x in 0..256 {
            let [r, g, b, a] = pixel(&baseline, 256, x, y);
            assert_eq!([r, g, b], ramp(x, y), "harness ramp at ({x}, {y})");
            assert_eq!(a, 255);
        }
    }

    let vivid = CdlParams::from_controls(1.05, -0.01, 1.0, 1.25, [1.0; 3], [0.0; 3], [1.0; 3]);
    let phase = Handle::new(&gpu, 0.0, CdlParams::IDENTITY, 1.0 / 240.0);
    let mut phases = list(&gpu, Some("fs_ramp"), Some(&phase));
    let expect_untouched = |gpu: &mut Gpu, phases: &mut PhaseList, case: &str| {
        assert!(!phases.scene_mode(), "{case}: scene mode is on");
        assert_eq!(frame(gpu, phases), baseline, "{case}: the frame changed");
        assert_eq!(phase.with(|p| p.passes_recorded()), 0, "{case}: a pass ran");
        assert!(
            !phase.with(|p| p.history_allocated()),
            "{case}: history exists"
        );
    };
    expect_untouched(&mut gpu, &mut phases, "both off, in a world");

    phase.with(|p| {
        p.set_in_world(false);
        p.set_blur_strength(0.5);
        p.set_grade(vivid);
    });
    expect_untouched(&mut gpu, &mut phases, "both on, outside a world");

    phase.with(|p| {
        p.set_blur_strength(0.0);
        p.set_grade(CdlParams::IDENTITY);
        p.set_in_world(true);
    });
    expect_untouched(&mut gpu, &mut phases, "grade back at the identity");

    // Switched on, it does run; and its first frame is sharp, so even then no byte changes.
    phase.with(|p| p.set_blur_strength(0.5));
    assert!(phases.scene_mode());
    assert_eq!(frame(&mut gpu, &mut phases), baseline, "first blur frame");
    assert_eq!(phase.with(|p| p.passes_recorded()), 1);
    assert!(phase.with(|p| p.history_allocated()));

    // Off again: the history is freed at once and the direct path is back.
    phase.with(|p| p.set_blur_strength(0.0));
    assert!(!phase.with(|p| p.history_allocated()));
    assert!(!phases.scene_mode());
    assert_eq!(frame(&mut gpu, &mut phases), baseline);
    assert_eq!(phase.with(|p| p.passes_recorded()), 1);

    // Under MSAA the UI pass would resolve over this pass's output: the phase never runs.
    if let Some(msaa) = headless(16, 16, 4).filter(|g| g.sample_count() > 1) {
        let mut phase = MotionGradePhase::new(&msaa);
        phase.set_in_world(true);
        phase.set_blur_strength(0.5);
        phase.set_grade(vivid);
        assert!(!phase.is_active());
        assert!(!RenderPhase::needs_scene_color(&phase));
    }
}

#[test]
fn the_grade_matches_the_asc_cdl_cpu_reference_on_a_ramp() {
    let Some(mut gpu) = headless(256, 16, 1) else {
        return;
    };
    let cases: [(&str, CdlParams); 5] = [
        (
            "slope, offset and power per channel",
            CdlParams {
                slope: [1.25, 0.8, 1.1],
                offset: [0.03, -0.05, 0.0],
                power: [0.7, 1.4, 1.0],
                saturation: 1.0,
            },
        ),
        (
            "clamp before power (in*0.5-0.2 < 0 must give 0, not a squared negative)",
            CdlParams {
                slope: [0.5; 3],
                offset: [-0.2; 3],
                power: [2.0; 3],
                saturation: 1.0,
            },
        ),
        (
            "saturation 2 needs the final clamp",
            CdlParams {
                slope: [1.1, 1.0, 0.9],
                offset: [0.0; 3],
                power: [1.0; 3],
                saturation: 2.0,
            },
        ),
        (
            "saturation 0 is Rec.709 grey",
            CdlParams {
                saturation: 0.0,
                ..CdlParams::IDENTITY
            },
        ),
        (
            "the master controls (research: punchy contrast)",
            CdlParams::from_controls(1.10, -0.04, 1.10, 1.15, [1.0; 3], [0.0; 3], [1.0; 3]),
        ),
    ];
    for (case, grade) in cases {
        let phase = Handle::new(&gpu, 0.0, grade, 1.0 / 240.0);
        let mut phases = list(&gpu, Some("fs_ramp"), Some(&phase));
        let pixels = frame(&mut gpu, &mut phases);
        assert_eq!(phase.with(|p| p.passes_recorded()), 1, "{case}");
        assert!(
            !phase.with(|p| p.history_allocated()),
            "{case}: grade-only must not allocate the blur history"
        );
        let (mut worst, mut off_by_one) = (0u8, 0usize);
        for y in 0..16 {
            for x in 0..256 {
                let input = ramp(x, y);
                let want = grade.apply(input.map(|v| f32::from(v) / 255.0));
                let want = want.map(|v| (v * 255.0).round() as u8);
                let got = pixel(&pixels, 256, x, y);
                assert_eq!(got[3], 255, "{case}: alpha at ({x}, {y})");
                for (&channel, &reference) in got.iter().zip(&want) {
                    let diff = channel.abs_diff(reference);
                    assert!(
                        diff <= 1,
                        "{case}: ({x}, {y}) in {input:?} -> {got:?}, CPU reference {want:?}"
                    );
                    worst = worst.max(diff);
                    off_by_one += usize::from(diff == 1);
                }
                if grade.saturation == 0.0 {
                    assert!(got[0] == got[1] && got[1] == got[2], "{case}: {got:?}");
                }
            }
        }
        eprintln!(
            "{case}: worst {worst}/255, {off_by_one} of {} channels off by one",
            256 * 16 * 3
        );
    }
}

/// Black for one frame, then grey 200 for `frames` frames of `dt` each, at strength 50 %: the
/// byte the last frame shows, checked against the CPU model of the shader on the way.
fn step_response(gpu: &mut Gpu, dt: f32, frames: u32) -> u8 {
    let phase = Handle::new(gpu, 0.5, CdlParams::IDENTITY, dt);
    let mut phases = list(gpu, None, Some(&phase));
    let a = blur_weight(0.5, dt);
    let mut model = Model::new(true);
    phases.clear_color = grey(0);
    let first = frame(gpu, &mut phases);
    assert_eq!(
        uniform(&first),
        Some([0, 0, 0, 255]),
        "the first frame is sharp"
    );
    model.frame(0, a);
    phases.clear_color = grey(200);
    let mut shown = 0;
    for _ in 0..frames {
        let pixels = frame(gpu, &mut phases);
        let px = uniform(&pixels).expect("a flat scene stays flat");
        let want = model.frame(200, a);
        assert!(
            px[0].abs_diff(want) <= 1 && px[0] == px[1] && px[1] == px[2],
            "dt {dt}: GPU {px:?}, CPU model {want}"
        );
        shown = px[0];
    }
    shown
}

#[test]
fn the_blur_is_frame_rate_independent() {
    let Some(mut gpu) = headless(16, 16, 1) else {
        return;
    };
    // 8.33 ms at 50 % is one half-life: 50 % of the way from 0 to 200, whatever the rate.
    for (fps, frames) in [(3000.0, 25), (240.0, 2), (120.0, 1)] {
        let shown = step_response(&mut gpu, 1.0 / fps, frames);
        eprintln!("{fps} FPS x {frames} frames (8.33 ms): {shown}, ideal 100");
        assert!(shown.abs_diff(100) <= 2, "{fps} FPS: {shown}");
    }
    // 16.7 ms is two half-lives: 75 %.
    for (fps, frames) in [(3000.0, 50), (240.0, 4), (120.0, 2), (60.0, 1)] {
        let shown = step_response(&mut gpu, 1.0 / fps, frames);
        eprintln!("{fps} FPS x {frames} frames (16.7 ms): {shown}, ideal 150");
        assert!(shown.abs_diff(150) <= 2, "{fps} FPS: {shown}");
    }
}

#[test]
fn the_history_stays_ungraded() {
    let Some(mut gpu) = headless(16, 16, 1) else {
        return;
    };
    // The grade halves every channel; dt = 1/120 gives a = 0.5 at strength 50 %.
    let halve = CdlParams {
        slope: [0.5; 3],
        ..CdlParams::IDENTITY
    };
    let phase = Handle::new(&gpu, 0.5, halve, 1.0 / 120.0);
    let mut phases = list(&gpu, None, Some(&phase));
    let shown = |pixels: Vec<u8>| uniform(&pixels).expect("flat")[0];

    // Frame 0 is sharp: the history takes 200, the screen shows the graded 100.
    phases.clear_color = grey(200);
    let first = shown(frame(&mut gpu, &mut phases));
    assert!(first.abs_diff(100) <= 1, "{first}");

    // Frame 1, scene 40: blend (40 + 200) / 2 = 120, shown graded as 60. A graded history
    // would blend (40 + 100) / 2 = 70 and show 35.
    phases.clear_color = grey(40);
    let second = shown(frame(&mut gpu, &mut phases));
    assert!(second.abs_diff(60) <= 1, "{second}");

    // Frame 2, grade off (the blur keeps its history): (40 + 120) / 2 = 80 on screen. Had the
    // history held the graded 60, it would show (40 + 60) / 2 = 50.
    phase.with(|p| p.set_grade(CdlParams::IDENTITY));
    assert!(phase.with(|p| p.history_allocated()));
    let third = shown(frame(&mut gpu, &mut phases));
    assert!(third.abs_diff(80) <= 1, "{third}");
}

/// The client's pipelined submit (`MC_PIPELINED_SUBMIT`, which the user's launcher turns on)
/// records with `render_deferred` and leaves the frame for the present thread; the phase's
/// uniform travels through the upload ring with it. Every frame must equal the serial path's,
/// with the scene, the grade, the strength and the frame time all changing every frame.
#[test]
fn the_pipelined_submit_path_shows_the_same_frames() {
    let Some(mut gpu) = headless(16, 16, 1) else {
        return;
    };
    let halve = CdlParams {
        slope: [0.5; 3],
        ..CdlParams::IDENTITY
    };
    let warm = CdlParams {
        slope: [1.06, 1.0, 0.94],
        saturation: 1.05,
        ..CdlParams::IDENTITY
    };
    let steps: [(u8, CdlParams, f32, f32); 7] = [
        (0, CdlParams::IDENTITY, 0.5, 1.0 / 240.0),
        (200, halve, 0.5, 1.0 / 240.0),
        (40, warm, 0.8, 1.0 / 60.0),
        (120, CdlParams::IDENTITY, 0.3, 1.0 / 1000.0),
        (255, halve, 0.95, 1.0 / 144.0),
        (10, warm, 0.5, 1.0 / 30.0),
        (90, CdlParams::IDENTITY, 0.7, 1.0 / 120.0),
    ];
    let mut run = |deferred: bool| -> Vec<Vec<u8>> {
        let phase = Handle::new(&gpu, 0.5, CdlParams::IDENTITY, 1.0 / 240.0);
        let mut phases = list(&gpu, None, Some(&phase));
        let mut frames = Vec::new();
        for (scene, grade, strength, dt) in steps {
            phase.with(|p| {
                p.set_grade(grade);
                p.set_blur_strength(strength);
                p.set_fixed_dt(Some(dt));
            });
            phases.clear_color = grey(scene);
            let frame = gpu.acquire().expect("acquire");
            if deferred {
                phases.render_deferred(&gpu, frame.view(), &Camera::default(), 0.0);
            } else {
                phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
            }
            // The readback submits a deferred frame first, as the F2 screenshot does.
            frames.push(gpu.read_pixels().expect("readback"));
        }
        assert_eq!(phase.with(|p| p.passes_recorded()), steps.len() as u64);
        frames
    };
    let serial = run(false);
    let pipelined = run(true);
    let distinct = serial.windows(2).filter(|w| w[0] != w[1]).count();
    assert_eq!(
        distinct,
        steps.len() - 1,
        "every step must show a new frame"
    );
    for (i, (s, p)) in serial.iter().zip(&pipelined).enumerate() {
        assert_eq!(
            s, p,
            "frame {i} differs between serial and pipelined submit"
        );
    }
}

/// Strength 95 % at 3000 FPS, black then grey 200: returns the byte shown after `frames`
/// frames, and after `frames` more.
fn slow_tail(gpu: &mut Gpu, floor: bool, frames: u32) -> (u8, u8) {
    let dt = 1.0 / 3000.0;
    let phase = Handle::new(gpu, 0.95, CdlParams::IDENTITY, dt);
    phase.with(|p| p.set_progress_floor(floor));
    let mut phases = list(gpu, None, Some(&phase));
    phases.clear_color = grey(0);
    frame(gpu, &mut phases);
    phases.clear_color = grey(200);
    let mut model = Model::new(floor);
    let a = blur_weight(0.95, dt);
    model.frame(0, a);
    let mut at = [0u8; 2];
    for slot in &mut at {
        for i in 1..=frames {
            let want = model.frame(200, a);
            if i % 250 == 0 {
                // Reading back also keeps the queue from running far ahead.
                let px = uniform(&frame(gpu, &mut phases)).expect("flat");
                assert!(
                    px[0].abs_diff(want) <= 1,
                    "floor {floor}, frame {i}: GPU {px:?}, CPU {want}"
                );
                *slot = px[0];
            } else {
                draw(gpu, &mut phases);
            }
        }
    }
    (at[0], at[1])
}

#[test]
fn a_slow_tail_converges_exactly_and_without_the_floor_a_ghost_stays() {
    let Some(mut gpu) = headless(16, 16, 1) else {
        return;
    };
    // a = 0.95^(120/3000) = 0.99795: one frame moves 0.2 % of the way. Once that step is below
    // half an fp16 spacing it rounds back to the stored value, and the spec's snap (at most
    // 6/255) is far too small to catch the ~30 levels still missing.
    let (soon, later) = slow_tail(&mut gpu, false, 1500);
    eprintln!("without the floor: {soon} after 1500 frames, {later} after 3000 (scene 200)");
    assert!(soon.abs_diff(200) > 10, "expected a ghost, got {soon}");
    assert_eq!(soon, later, "the ghost must be stuck, not merely slow");

    // With the floor every frame moves at least one fp16 spacing, so the tail reaches the
    // scene exactly and stays there.
    let (soon, later) = slow_tail(&mut gpu, true, 1500);
    eprintln!("with the floor: {soon} after 1500 frames, {later} after 3000 (scene 200)");
    assert_eq!(soon, 200);
    assert_eq!(later, 200);
}

/// The phase's GPU cost at 1080p, from the per-phase timestamps. Run with
/// `MC_GPU_TIMING=1 cargo test -p mc-render --test m7_postfx -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark; needs MC_GPU_TIMING=1"]
fn gpu_cost_at_1080p() {
    const FRAMES: u32 = 600;
    let (width, height) = (1920, 1080);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("SKIP: set MC_GPU_TIMING=1 (and the adapter needs timestamp queries)");
        return;
    }
    let vivid = CdlParams::from_controls(1.05, -0.01, 1.0, 1.25, [1.0; 3], [0.0; 3], [1.0; 3]);
    let configs: [(&str, f32, CdlParams); 4] = [
        ("off", 0.0, CdlParams::IDENTITY),
        ("grade only", 0.0, vivid),
        ("blur only", 0.5, CdlParams::IDENTITY),
        ("blur + grade", 0.5, vivid),
    ];
    eprintln!("{width}x{height}, {FRAMES} frames per row, noise scene, mean ms per frame:");
    eprintln!("config         | post/motion-grade | gpu.scene_blit | gpu.frame | measured frames");
    for (label, blur, grade) in configs {
        let phase = Handle::new(&gpu, blur, grade, 1.0 / 240.0);
        let mut phases = list(&gpu, Some("fs_noise"), Some(&phase));
        for i in 0..FRAMES {
            if i % 50 == 49 {
                let _ = frame(&mut gpu, &mut phases);
            } else {
                draw(&mut gpu, &mut phases);
            }
        }
        let timings = phases.gpu_timings();
        let mean_ms = |name: &str| {
            timings
                .scopes
                .iter()
                .find(|(scope, _)| *scope == name)
                .map_or(0.0, |(_, stats)| stats.mean_ns() / 1e6)
        };
        eprintln!(
            "{label:<14} | {:>17.4} | {:>14.4} | {:>9.4} | {}",
            mean_ms(MotionGradePhase::NAME),
            mean_ms("gpu.scene_blit"),
            timings.frame_ns.mean_ns() / 1e6,
            timings.frames,
        );
        assert!(timings.frames > 0, "{label}: no frame was timed");
    }
}
