//! Fewer render passes per frame, and frames recorded before their
//! swap-chain image exists, each drawing the same bytes as before.
//!
//! * `MC_W18_OB_CLEAR_FOLD`: the frame's clear is done by the first passes
//!   that touch colour and depth instead of a pass of its own.
//! * `MC_W18_OB_DEPTH_READ_PASS`: the screen-space fog and translucent
//!   terrain share one pass whose depth buffer is attached read-only.
//! * `MC_W18_OB_DECOUPLE`: `PhaseList::record_unsubmitted` records into
//!   another target, and one copy of it gives the frame `PhaseList::render` draws in place.
//!
//! Every comparison runs at 1x and 4x MSAA. The negative controls show the comparison would see
//! a clear that was wrongly deferred.
//!
//! ```text
//! cargo test -p mc-render --test w18_ob_passes -- --nocapture
//! ```

use std::cell::Cell;
use std::rc::Rc;

use mc_render::pipeline::{FrameClearHandoff, PassTarget};
use mc_render::sky::fog::CameraMedium;
use mc_render::sky::FogPhase;
use mc_render::{
    Camera, DepthAttachment, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase,
};

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
        Ok(g) => {
            let info = g.adapter_info();
            eprintln!("adapter: {} ({:?})", info.name, info.backend);
            Some(g)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

/// How a test phase records its one triangle.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    /// A pass of its own through `Gpu::begin_pass`, with this depth attachment.
    Own(DepthAttachment),
    /// The shared world pass (a pass of its own with the shared depth when sharing is off).
    World,
    /// The shared read-only-depth pass (a pass of its own with the shared depth when that is off).
    DepthRead,
    /// A pass opened directly on the colour target (loaded), without `Gpu::begin_pass` and
    /// without depth. Only a phase that wrongly takes over the colour clear does this: negative
    /// controls, single-sampled.
    DirectColor,
    /// A pass opened directly on the colour target and the shared depth buffer (both loaded).
    /// Negative controls, single-sampled.
    DirectDepth,
    /// Opens no pass at all (a phase that returns early this frame).
    Nothing,
}

impl Mode {
    fn attaches_depth(self) -> bool {
        !matches!(
            self,
            Mode::Own(DepthAttachment::None) | Mode::DirectColor | Mode::Nothing
        )
    }
}

/// One translucent triangle with its own coverage, colour and depth.
struct Tri {
    pipeline: wgpu::RenderPipeline,
    mode: Mode,
    handoff: FrameClearHandoff,
    /// Records in its own pass, into the world pass, into the read-only-depth pass.
    calls: Rc<[Cell<u32>; 3]>,
}

#[derive(Clone, Copy)]
struct Spec {
    corners: [[f32; 2]; 3],
    rgba: [f32; 4],
    z: f32,
    write: bool,
    mode: Mode,
    handoff: FrameClearHandoff,
}

fn spec(corners: [[f32; 2]; 3], rgba: [f32; 4], z: f32, write: bool, mode: Mode) -> Spec {
    Spec {
        corners,
        rgba,
        z,
        write,
        mode,
        handoff: FrameClearHandoff::None,
    }
}

impl Tri {
    fn new(gpu: &Gpu, s: Spec) -> Self {
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
            p(s.corners[0]),
            p(s.corners[1]),
            p(s.corners[2]),
            s.rgba[0],
            s.rgba[1],
            s.rgba[2],
            s.rgba[3],
            z = s.z,
        );
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("w18-ob tri"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("w18-ob tri"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let samples = if matches!(s.mode, Mode::DirectColor | Mode::DirectDepth) {
            1
        } else {
            gpu.sample_count()
        };
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("w18-ob tri"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: s.mode.attaches_depth().then(|| wgpu::DepthStencilState {
                    format: mc_render::DEPTH_FORMAT,
                    depth_write_enabled: Some(s.write && s.mode != Mode::DepthRead),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
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
            mode: s.mode,
            handoff: s.handoff,
            calls: Rc::new([Cell::new(0), Cell::new(0), Cell::new(0)]),
        }
    }

    fn count(&self, slot: usize) {
        self.calls[slot].set(self.calls[slot].get() + 1);
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
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
        self.count(0);
        match self.mode {
            Mode::Own(depth) => {
                let mut pass = gpu.begin_pass(enc, view, "w18-ob tri", depth);
                self.draw(&mut pass);
            }
            Mode::World => {
                let mut own = None;
                let pass = PassTarget::Own { enc, view }.begin(gpu, "w18-ob tri", &mut own);
                self.draw(pass);
            }
            Mode::DepthRead => {
                let mut pass = gpu.begin_pass(enc, view, "w18-ob tri", DepthAttachment::Shared);
                self.draw(&mut pass);
            }
            Mode::DirectColor | Mode::DirectDepth => {
                let depth = (self.mode == Mode::DirectDepth).then(|| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: gpu.depth_view(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                });
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("w18-ob direct"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: depth,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                self.draw(&mut pass);
            }
            Mode::Nothing => {}
        }
    }

    fn name(&self) -> &'static str {
        "w18-ob tri"
    }

    fn shares_world_pass(&self) -> bool {
        self.mode == Mode::World
    }

    fn record_into_pass(
        &mut self,
        _gpu: &Gpu,
        pass: &mut wgpu::RenderPass<'static>,
        _cam: &Camera,
    ) {
        self.count(1);
        self.draw(pass);
    }

    fn frame_clear_handoff(&self) -> FrameClearHandoff {
        self.handoff
    }

    fn shares_depth_read_pass(&self) -> bool {
        self.mode == Mode::DepthRead
    }

    fn record_into_depth_read_pass(
        &mut self,
        _gpu: &Gpu,
        pass: &mut wgpu::RenderPass<'static>,
        _cam: &Camera,
    ) {
        self.count(2);
        self.draw(pass);
    }
}

const FULL: [[f32; 2]; 3] = [[-1.0, -3.0], [-1.0, 1.0], [3.0, 1.0]];

/// Leave stale colour and depth behind (red, depth 0.1 everywhere), so a clear that does not
/// happen where it should shows in the next frame.
fn stale_frame(gpu: &mut Gpu) {
    let mut phases = PhaseList::new();
    phases.set_clear_fold(false);
    phases.clear_color = wgpu::Color::RED;
    phases.insert(
        PhaseOrder(200),
        Box::new(Tri::new(
            gpu,
            spec(
                FULL,
                [0.9, 0.2, 0.1, 1.0],
                0.1,
                true,
                Mode::Own(DepthAttachment::Shared),
            ),
        )),
    );
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    let _ = gpu.read_pixels().expect("readback");
}

struct Switches {
    fold: bool,
    depth_read: bool,
}

/// Render `specs` (plus `extra`, registered as given) after a stale frame; the pixels and the
/// records (own pass, world pass, read-only-depth pass) summed over the triangles.
fn render(
    gpu: &mut Gpu,
    specs: &[(u32, Spec)],
    extra: Extra,
    switches: &Switches,
    cam: &Camera,
) -> (Vec<u8>, [u32; 3]) {
    stale_frame(gpu);
    let mut phases = PhaseList::new();
    phases.set_clear_fold(switches.fold);
    phases.set_depth_read_pass(switches.depth_read);
    phases.clear_color = wgpu::Color {
        r: 0.05,
        g: 0.1,
        b: 0.2,
        a: 1.0,
    };
    let mut counters = Vec::new();
    for (order, s) in specs {
        let tri = Tri::new(gpu, *s);
        counters.push(Rc::clone(&tri.calls));
        phases.insert(PhaseOrder(*order), Box::new(tri));
    }
    for (order, phase) in extra {
        phases.insert(order, phase);
    }
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let mut calls = [0; 3];
    for c in &counters {
        for (slot, cell) in c.iter().enumerate() {
            calls[slot] += cell.get();
        }
    }
    (pixels, calls)
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count()
}

fn distinct_colours(pixels: &[u8]) -> usize {
    let mut colours: Vec<&[u8]> = pixels.chunks_exact(4).collect();
    colours.sort_unstable();
    colours.dedup();
    colours.len()
}

/// A sky-like phase: no depth, takes over the whole clear.
fn sky(rgba: [f32; 4]) -> Spec {
    Spec {
        handoff: FrameClearHandoff::ColorAndDepth,
        ..spec(
            [[-1.0, 1.0], [1.0, 1.0], [0.0, -0.6]],
            rgba,
            0.9,
            false,
            Mode::Own(DepthAttachment::None),
        )
    }
}

/// A cloud-like phase: writes the shared depth in its own pass, lets the depth clear wait.
fn clouds(mode: Mode) -> Spec {
    Spec {
        handoff: FrameClearHandoff::Depth,
        ..spec(
            [[-1.0, 0.2], [1.0, 0.6], [-0.2, 1.0]],
            [0.95, 0.95, 0.95, 0.7],
            0.6,
            true,
            mode,
        )
    }
}

/// The opaque world: three depth-writing triangles in the shared world pass.
fn world() -> [Spec; 3] {
    [
        spec(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
            [0.9, 0.1, 0.1, 0.6],
            0.5,
            true,
            Mode::World,
        ),
        spec(
            [[-0.6, -1.0], [1.0, 0.2], [-1.0, 0.9]],
            [0.1, 0.8, 0.2, 0.5],
            0.55,
            true,
            Mode::World,
        ),
        spec(
            [[0.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            [0.2, 0.2, 0.9, 0.7],
            0.45,
            true,
            Mode::World,
        ),
    ]
}

/// A hand-like phase that clears depth, then a depth-tested triangle after the world.
fn late() -> [Spec; 2] {
    [
        spec(
            [[-1.0, 0.0], [1.0, -1.0], [0.6, 1.0]],
            [0.6, 0.1, 0.8, 0.5],
            0.52,
            false,
            Mode::Own(DepthAttachment::Shared),
        ),
        spec(
            [[-0.9, -0.9], [0.9, -0.2], [-0.2, 0.9]],
            [0.1, 0.9, 0.9, 0.6],
            0.7,
            true,
            Mode::Own(DepthAttachment::Clear),
        ),
    ]
}

fn scenario(head: &[Spec]) -> Vec<(u32, Spec)> {
    let mut out: Vec<(u32, Spec)> = head
        .iter()
        .enumerate()
        .map(|(i, s)| (100 + i as u32, *s))
        .collect();
    out.extend(
        world()
            .into_iter()
            .enumerate()
            .map(|(i, s)| (200 + i as u32, s)),
    );
    out.extend(
        late()
            .into_iter()
            .enumerate()
            .map(|(i, s)| (600 + i as u32, s)),
    );
    out
}

#[test]
fn a_frame_clear_folded_into_the_first_passes_draws_the_same_frame() {
    let sky_rgba = [0.3, 0.5, 0.9, 0.8];
    let cases: Vec<(&str, Vec<(u32, Spec)>)> = vec![
        (
            "sky, depth-writing clouds, world",
            scenario(&[sky(sky_rgba), clouds(Mode::Own(DepthAttachment::Shared))]),
        ),
        (
            "sky, clouds drawing nothing, world",
            scenario(&[sky(sky_rgba), clouds(Mode::Nothing)]),
        ),
        (
            "sky drawing nothing, clouds, world",
            scenario(&[
                Spec {
                    mode: Mode::Nothing,
                    ..sky(sky_rgba)
                },
                clouds(Mode::Own(DepthAttachment::Shared)),
            ]),
        ),
        (
            "clouds first (colour still pending)",
            scenario(&[clouds(Mode::Own(DepthAttachment::Shared))]),
        ),
        (
            "a first phase that does not opt in",
            scenario(&[Spec {
                handoff: FrameClearHandoff::None,
                ..sky(sky_rgba)
            }]),
        ),
        ("world first", scenario(&[])),
        (
            "nothing draws",
            vec![
                (
                    100,
                    Spec {
                        mode: Mode::Nothing,
                        ..sky(sky_rgba)
                    },
                ),
                (101, clouds(Mode::Nothing)),
            ],
        ),
        (
            "sky, then a read-only-depth phase",
            vec![
                (100, sky(sky_rgba)),
                (
                    500,
                    spec(
                        [[-1.0, -1.0], [1.0, -0.5], [0.0, 1.0]],
                        [0.2, 0.7, 0.7, 0.5],
                        0.5,
                        false,
                        Mode::DepthRead,
                    ),
                ),
            ],
        ),
    ];
    for samples in [1, 4] {
        let Some(mut gpu) = headless(samples) else {
            return;
        };
        let cam = Camera::default();
        for (name, specs) in &cases {
            let off = Switches {
                fold: false,
                depth_read: true,
            };
            let on = Switches {
                fold: true,
                depth_read: true,
            };
            let (old, _) = render(&mut gpu, specs, Vec::new(), &off, &cam);
            let (new, _) = render(&mut gpu, specs, Vec::new(), &on, &cam);
            let n = differing(&old, &new);
            assert_eq!(n, 0, "{samples}x MSAA, {name}: {n} pixels differ");
            // Not vacuous: no stale red survives, and the frame has its structure.
            assert!(
                new.chunks_exact(4)
                    .all(|p| p != [230_u8, 51, 26, 255].as_slice()),
                "{name}: the stale frame shows through"
            );
            eprintln!(
                "{samples}x MSAA, {name}: identical ({} colours)",
                distinct_colours(&old)
            );
        }
    }
}

/// Negative controls: a phase that claims to take over a clear it does not do. The fold then
/// leaves the stale frame visible, and the comparison above would have caught it.
#[test]
fn a_phase_that_wrongly_takes_over_the_clear_is_caught() {
    let Some(mut gpu) = headless(1) else {
        return;
    };
    let cam = Camera::default();
    let liar_colour = Spec {
        handoff: FrameClearHandoff::ColorAndDepth,
        ..spec(
            [[-1.0, 1.0], [1.0, 1.0], [0.0, -0.6]],
            [0.3, 0.5, 0.9, 0.5],
            0.9,
            false,
            Mode::DirectColor,
        )
    };
    let liar_depth = Spec {
        handoff: FrameClearHandoff::Depth,
        ..spec(
            [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]],
            [0.9, 0.9, 0.2, 0.8],
            0.6,
            true,
            Mode::DirectDepth,
        )
    };
    for (name, specs, at_least) in [
        ("colour", scenario(&[liar_colour]), 500),
        (
            "depth",
            scenario(&[sky([0.3, 0.5, 0.9, 0.8]), liar_depth]),
            500,
        ),
    ] {
        let off = Switches {
            fold: false,
            depth_read: true,
        };
        let on = Switches {
            fold: true,
            depth_read: true,
        };
        let (old, _) = render(&mut gpu, &specs, Vec::new(), &off, &cam);
        let (new, _) = render(&mut gpu, &specs, Vec::new(), &on, &cam);
        let n = differing(&old, &new);
        assert!(n >= at_least, "{name}: only {n} pixels differ");
        eprintln!("negative control ({name}): {n} pixels differ, as they must");
    }
}

/// Phases registered as they are, beside the test triangles.
type Extra = Vec<(PhaseOrder, Box<dyn RenderPhase>)>;

/// The real screen-space fog (water, so it fogs within a few blocks) between the opaque world
/// and a translucent, depth-tested, non-writing triangle; a depth-tested triangle afterwards
/// shows that the depth buffer comes out of the read-only pass unchanged.
fn fog_scene(gpu: &Gpu, fog_visible: bool) -> (Vec<(u32, Spec)>, Extra) {
    let specs = vec![
        (
            200,
            spec(
                [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
                [0.9, 0.6, 0.3, 1.0],
                0.990,
                true,
                Mode::World,
            ),
        ),
        (
            201,
            spec(
                [[-0.6, -1.0], [1.0, 0.2], [-1.0, 0.9]],
                [0.3, 0.8, 0.4, 1.0],
                0.997,
                true,
                Mode::World,
            ),
        ),
        (
            500,
            spec(
                [[0.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                [0.2, 0.3, 0.9, 0.6],
                0.994,
                false,
                Mode::DepthRead,
            ),
        ),
        (
            600,
            spec(
                [[-1.0, 0.0], [1.0, -1.0], [0.6, 1.0]],
                [0.9, 0.2, 0.7, 0.7],
                0.995,
                true,
                Mode::Own(DepthAttachment::Shared),
            ),
        ),
    ];
    let mut fog = FogPhase::new(gpu);
    fog.inputs.render_distance_chunks = 8;
    fog.inputs.medium = CameraMedium::Water;
    fog.visible = fog_visible;
    let extra: Extra = vec![(FogPhase::BEFORE_TRANSLUCENT_ORDER, Box::new(fog))];
    (specs, extra)
}

fn fog_camera() -> Camera {
    let mut cam = Camera::default();
    cam.set_viewport(SIZE, SIZE);
    cam.update();
    cam
}

#[test]
fn fog_and_translucent_share_a_read_only_depth_pass_and_draw_the_same_frame() {
    for samples in [1, 4] {
        let Some(mut gpu) = headless(samples) else {
            return;
        };
        let cam = fog_camera();
        let off = Switches {
            fold: true,
            depth_read: false,
        };
        let on = Switches {
            fold: true,
            depth_read: true,
        };
        let (specs, extra) = fog_scene(&gpu, true);
        let (old, old_calls) = render(&mut gpu, &specs, extra, &off, &cam);
        let (specs, extra) = fog_scene(&gpu, true);
        let (new, new_calls) = render(&mut gpu, &specs, extra, &on, &cam);
        assert_eq!(old_calls[2], 0, "sharing off: no read-only-depth pass");
        assert_eq!(
            new_calls[2], 1,
            "sharing on: the translucent triangle records into the read-only-depth pass"
        );
        let n = differing(&old, &new);
        assert_eq!(n, 0, "{samples}x MSAA: {n} pixels differ");
        // Not vacuous: the fog changes the frame.
        let (specs, extra) = fog_scene(&gpu, false);
        let (unfogged, _) = render(&mut gpu, &specs, extra, &on, &cam);
        let fogged = differing(&new, &unfogged);
        assert!(
            fogged > 1000,
            "{samples}x MSAA: the fog changed only {fogged} pixels"
        );
        eprintln!(
            "{samples}x MSAA: identical; the fog changes {fogged} pixels; {} colours",
            distinct_colours(&new)
        );
    }
}

/// Read a single-sampled RGBA8 texture back as tightly packed rows.
fn read_texture(gpu: &Gpu, texture: &wgpu::Texture) -> Vec<u8> {
    let row = SIZE * 4;
    let padded = row.div_ceil(256) * 256;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("w18-ob readback"),
        size: u64::from(padded * SIZE),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(SIZE),
            },
        },
        texture.size(),
    );
    gpu.queue.submit(Some(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).expect("map result receiver");
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("readback poll");
    rx.recv().expect("map callback").expect("readback map");
    let bytes = buffer.get_mapped_range(..).expect("readback view").to_vec();
    buffer.unmap();
    bytes
        .chunks_exact(padded as usize)
        .flat_map(|r| r[..row as usize].iter().copied())
        .collect()
}

#[test]
fn a_frame_recorded_before_its_image_and_copied_matches_the_frame_drawn_in_place() {
    for samples in [1, 4] {
        let Some(mut gpu) = headless(samples) else {
            return;
        };
        let cam = fog_camera();
        let on = Switches {
            fold: true,
            depth_read: true,
        };
        // In place, through `PhaseList::render`.
        let (specs, extra) = fog_scene(&gpu, true);
        let (in_place, _) = render(&mut gpu, &specs, extra, &on, &cam);

        // Recorded into another target first, then copied, as the shell does.
        stale_frame(&mut gpu);
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: 0.05,
            g: 0.1,
            b: 0.2,
            a: 1.0,
        };
        let (specs, extra) = fog_scene(&gpu, true);
        for (order, s) in &specs {
            phases.insert(PhaseOrder(*order), Box::new(Tri::new(&gpu, *s)));
        }
        for (order, phase) in extra {
            phases.insert(order, phase);
        }
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let make = |label: &str| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: gpu.config.format,
                usage: usage | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let recorded = make("w18-ob recorded");
        let image = make("w18-ob image");
        let view = recorded.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = phases
            .record_unsubmitted(&gpu, &view, &cam, 0.0)
            .expect("no GPU timestamps in this test");
        enc.copy_texture_to_texture(
            recorded.as_image_copy(),
            image.as_image_copy(),
            recorded.size(),
        );
        let _ = gpu.submit_frame(enc.finish());
        let copied = read_texture(&gpu, &image);
        let n = differing(&in_place, &copied);
        assert_eq!(n, 0, "{samples}x MSAA: {n} pixels differ");
        assert!(distinct_colours(&copied) >= 6, "a frame with structure");
        eprintln!("{samples}x MSAA: the copied frame is identical");
    }
}
