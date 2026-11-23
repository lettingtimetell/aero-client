//! The Dreamy pack's pieces on a real GPU.
//!
//! * The colour LUT's GPU bake equals its CPU reference ([`GradeParams::apply`]) at every one of
//!   the 35,937 entries, and the identity grade bakes an exact identity.
//! * With the Dreamy pack not running, its phases record nothing, allocate nothing, and the frame
//!   is byte-identical to a frame without them; running, they change the frame.
//! * Ground fog on a synthetic scene: far ground inside the layer is fogged, a "player" at your
//!   height at fighting range keeps its exact pixels (the depth-aware upsample keeps fog off its
//!   silhouette), and the sky above the layer is untouched.
//! * The finish pass grades the world but not what the UI draws after it.
//!
//! Every test skips (printing `SKIP:`) when the machine has no adapter.

use glam::{DVec3, Vec3};
use mc_render::shaderpack::dreamy::{
    DreamyAtmospherePhase, DreamyFinishPhase, DreamySettings, FogSettings,
};
use mc_render::shaderpack::grade::{GradeLut, GradeParams, Wheel, LUT_SIZE};
use mc_render::shaderpack::ground_fog::{FogLayer, FogResolution};
use mc_render::shaderpack::{order, PackFrame, PackId, ShaderCore};
use mc_render::{
    Camera, DepthAttachment, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase,
    SceneTextures, DEPTH_FORMAT, HEADLESS_FORMAT,
};
use std::cell::RefCell;
use std::rc::Rc;

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

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Bake `params` and read the whole table back as RGB triples, indexed `[b][g][r]`.
fn bake_and_read(gpu: &Gpu, params: GradeParams) -> Vec<[f32; 3]> {
    let mut lut = GradeLut::new(gpu);
    lut.set_params(params);
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    assert!(lut.record_bake(gpu, &mut enc), "a new table bakes");
    let n = LUT_SIZE;
    let row = 512_u32; // 33 texels * 8 bytes = 264, padded to 256-byte rows.
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lut readback"),
        size: u64::from(row * n * n),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: lut.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(n),
            },
        },
        wgpu::Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: n,
        },
    );
    // The bake's uniform travels in the upload ring: submit it with the frame.
    gpu.submit_frame(enc.finish());
    buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("map the table");
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let bytes = buffer.slice(..).get_mapped_range().expect("mapped");
    let mut out = Vec::with_capacity((n * n * n) as usize);
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let at = ((b * n + g) * row + r * 8) as usize;
                let channel = |c: usize| {
                    f16_to_f32(u16::from_le_bytes([
                        bytes[at + 2 * c],
                        bytes[at + 2 * c + 1],
                    ]))
                };
                out.push([channel(0), channel(1), channel(2)]);
            }
        }
    }
    out
}

fn grid_colour(r: u32, g: u32, b: u32) -> [f32; 3] {
    let n = (LUT_SIZE - 1) as f32;
    [r as f32 / n, g as f32 / n, b as f32 / n]
}

/// A grade with every control away from neutral (the Dreamy preset plus a gradient map).
fn busy_grade() -> GradeParams {
    GradeParams {
        exposure: 0.25,
        temperature: 0.12,
        tint: 0.06,
        contrast: 1.1,
        saturation: 1.15,
        vibrance: 0.2,
        hue_shift: 6.0,
        lift: Wheel {
            hue: 255.0,
            strength: 0.2,
        },
        gamma: Wheel {
            hue: 30.0,
            strength: 0.1,
        },
        gain: Wheel {
            hue: 35.0,
            strength: 0.15,
        },
        split_shadow: [0.42, 0.35, 0.8],
        split_highlight: [1.0, 0.78, 0.63],
        split_balance: 0.1,
        split_strength: 0.3,
        gradient: [[0.1, 0.06, 0.2], [0.6, 0.42, 0.62], [1.0, 0.9, 0.78]],
        gradient_mix: 0.15,
        fade: 0.1,
    }
}

#[test]
fn the_lut_bake_matches_the_cpu_reference() {
    let Some(gpu) = headless(8, 8) else {
        return;
    };
    let n = LUT_SIZE;
    // Identity: every entry is its own grid colour (to f16 precision).
    let identity = bake_and_read(&gpu, GradeParams::IDENTITY);
    let mut worst = 0.0_f32;
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let got = identity[((b * n + g) * n + r) as usize];
                let want = grid_colour(r, g, b);
                for c in 0..3 {
                    worst = worst.max((got[c] - want[c]).abs());
                }
            }
        }
    }
    eprintln!("identity table: worst error {worst}");
    assert!(worst < 1.0e-3, "identity table off by {worst}");

    // A busy grade: the GPU table and the CPU reference agree everywhere.
    let params = busy_grade();
    let table = bake_and_read(&gpu, params);
    let mut worst = 0.0_f32;
    let mut moved = 0.0_f32;
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let got = table[((b * n + g) * n + r) as usize];
                let input = grid_colour(r, g, b);
                let want = params.apply(input);
                for c in 0..3 {
                    worst = worst.max((got[c] - want[c]).abs());
                    moved = moved.max((want[c] - input[c]).abs());
                }
            }
        }
    }
    eprintln!("busy table: worst GPU/CPU difference {worst}, largest change {moved}");
    assert!(moved > 0.1, "the busy grade must actually grade");
    assert!(
        worst < 2.0e-3,
        "GPU bake differs from the CPU reference by {worst}"
    );
}

// ------------------------------------------------------------------------- scene harness

/// The client's `Shared`: the list owns one handle, the test keeps the other.
struct Handle<T>(Rc<RefCell<T>>);

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle(Rc::clone(&self.0))
    }
}

impl<T> Handle<T> {
    fn new(value: T) -> Self {
        Handle(Rc::new(RefCell::new(value)))
    }

    fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(&mut self.0.borrow_mut())
    }
}

impl<T: RenderPhase> RenderPhase for Handle<T> {
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
        self.0.borrow().name()
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

/// A world for the fog to sit in, drawn with depth from the camera uniform: a ground plane at
/// y = 60 (inside the fog layer), a "player" (a 0.6 x 1.8 box, red) standing 8 blocks in front of
/// the camera at the camera's own feet height (74), and the sky (the clear colour) above.
const SCENE_WGSL: &str = r"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) colour: vec3<f32>,
};

fn corner(i: u32) -> vec2<f32> {
    var c = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    return c[i];
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    let quad = index / 6u;
    let c = corner(index % 6u);
    var world: vec3<f32>;
    var colour: vec3<f32>;
    if (quad == 0u) {
        // Ground: 600 x 600 blocks at y = 60.
        world = vec3<f32>(c.x * 600.0 - 300.0, 60.0, c.y * 600.0 - 300.0);
        colour = vec3<f32>(0.25, 0.55, 0.25);
    } else {
        // Player: the face toward the camera, at z = 8, feet at 74.
        world = vec3<f32>(c.x * 0.6 - 0.3, 74.0 + c.y * 1.8, 8.0);
        colour = vec3<f32>(0.85, 0.15, 0.15);
    }
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(world - camera.camera_pos.xyz, 1.0);
    out.colour = colour;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.colour, 1.0);
}
";

struct SceneGeometry {
    pipeline: wgpu::RenderPipeline,
}

impl SceneGeometry {
    fn new(gpu: &Gpu) -> SceneGeometry {
        SceneGeometry::with_source(gpu, SCENE_WGSL)
    }

    fn with_source(gpu: &Gpu, source: &str) -> SceneGeometry {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sb test scene"),
                source: wgpu::ShaderSource::Wgsl(source.to_owned().into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sb test scene"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sb test scene"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
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
        SceneGeometry { pipeline }
    }
}

impl RenderPhase for SceneGeometry {
    fn name(&self) -> &'static str {
        "test/scene"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = gpu.begin_pass(enc, view, "test/scene", DepthAttachment::Clear);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.draw(0..12, 0..1);
    }
}

/// A solid white rectangle in the top-left corner, drawn at `UI` (after the pack's passes).
const HUD_WGSL: &str = r"
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var c = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, 1.0), vec2<f32>(-0.6, 1.0), vec2<f32>(-0.6, 0.6),
        vec2<f32>(-1.0, 1.0), vec2<f32>(-0.6, 0.6), vec2<f32>(-1.0, 0.6),
    );
    return vec4<f32>(c[index], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.9, 0.9, 0.2, 1.0);
}
";

struct Hud {
    pipeline: wgpu::RenderPipeline,
}

impl Hud {
    fn new(gpu: &Gpu) -> Hud {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sb test hud"),
                source: wgpu::ShaderSource::Wgsl(HUD_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sb test hud"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sb test hud"),
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
                    entry_point: Some("fs"),
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
        Hud { pipeline }
    }
}

impl RenderPhase for Hud {
    fn name(&self) -> &'static str {
        "test/hud"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = gpu.begin_pass(enc, view, "test/hud", DepthAttachment::None);
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..6, 0..1);
    }
}

const W: u32 = 320;
const H: u32 = 180;
const SKY: wgpu::Color = wgpu::Color {
    r: 0.55,
    g: 0.7,
    b: 0.95,
    a: 1.0,
};

/// The camera: eye 1.62 above feet at y = 74, looking south (+Z) and 20 degrees down.
fn camera() -> Camera {
    let mut cam = Camera {
        pos: DVec3::new(0.0, 75.62, 0.0),
        yaw: 0.0,
        pitch: 20.0,
        ..Camera::default()
    };
    cam.set_viewport(W, H);
    cam.update();
    cam
}

/// A core running the Dreamy pack in a world, for this camera.
fn running_core(gpu: &Gpu, cam: &Camera) -> ShaderCore {
    let core = ShaderCore::new();
    core.set_active(Some(PackId::Dreamy));
    core.set_frame(
        gpu,
        PackFrame {
            in_world: true,
            eye: cam.pos,
            view: cam.view,
            proj: cam.proj,
            far: cam.far,
            width: W,
            height: H,
            fog_color_display: Vec3::new(0.75, 0.85, 1.0),
            ..PackFrame::default()
        },
    );
    core
}

fn fog_only() -> DreamySettings {
    DreamySettings {
        fog: Some(FogSettings {
            layer: FogLayer {
                top: 0.0,
                softness: 6.0,
                density: 0.12,
                billows: 3.0,
                wisps: 0.55,
                start: 16.0,
                steps: 10,
                resolution: FogResolution::Half,
            },
            color: None,
            light: 0.5,
        }),
        ..DreamySettings::default()
    }
}

struct Rig {
    phases: PhaseList,
    atmosphere: Handle<DreamyAtmospherePhase>,
    finish: Handle<DreamyFinishPhase>,
}

fn rig(gpu: &Gpu, core: &ShaderCore, with_pack: bool) -> Rig {
    let mut phases = PhaseList::new();
    phases.clear_color = SKY;
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(SceneGeometry::new(gpu)));
    phases.insert(PhaseOrder::UI, Box::new(Hud::new(gpu)));
    let atmosphere = Handle::new(DreamyAtmospherePhase::new(core.clone()));
    let finish = Handle::new(DreamyFinishPhase::new(core.clone()));
    if with_pack {
        phases.insert(order::WORLD, Box::new(atmosphere.clone()));
        phases.insert(order::FINAL, Box::new(finish.clone()));
    }
    Rig {
        phases,
        atmosphere,
        finish,
    }
}

fn render(gpu: &mut Gpu, phases: &mut PhaseList, cam: &Camera) -> Vec<u8> {
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    gpu.read_pixels().expect("readback")
}

fn px(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

fn diff(a: [u8; 3], b: [u8; 3]) -> i32 {
    (0..3)
        .map(|c| (i32::from(a[c]) - i32::from(b[c])).abs())
        .max()
        .unwrap_or(0)
}

/// Where the player's face and the far ground land on screen, found from the baseline frame's
/// colours (red player, green ground).
fn find(pixels: &[u8], test: impl Fn([u8; 3]) -> bool) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            if test(px(pixels, x, y)) {
                out.push((x, y));
            }
        }
    }
    out
}

#[test]
fn off_the_pack_costs_nothing_and_changes_nothing() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let cam = camera();
    // Not running: the core has no pack (or another pack) switched on.
    let core = ShaderCore::new();
    let mut plain = rig(&gpu, &core, false);
    let mut with = rig(&gpu, &core, true);
    let settings = DreamySettings {
        haze: 0.5,
        glow: 0.4,
        vignette: 0.2,
        grade: busy_grade(),
        ..fog_only()
    };
    with.atmosphere.with(|p| p.set_settings(settings));
    with.finish.with(|p| p.set_settings(settings));
    for active in [None, Some(PackId::Classic)] {
        core.set_active(active);
        let a = render(&mut gpu, &mut plain.phases, &cam);
        let b = render(&mut gpu, &mut with.phases, &cam);
        assert!(!with.phases.scene_mode(), "{active:?}: no scene copy");
        assert_eq!(
            a, b,
            "{active:?}: the frame is byte-identical without the pack"
        );
        assert!(!with.atmosphere.with(|p| p.allocated()));
        assert!(!with.finish.with(|p| p.allocated()));
        assert_eq!(with.atmosphere.with(|p| p.passes_recorded()), 0);
        assert_eq!(with.finish.with(|p| p.passes_recorded()), 0);
    }
    // Running: both phases draw and the frame changes.
    let core = running_core(&gpu, &cam);
    let mut on = rig(&gpu, &core, true);
    on.atmosphere.with(|p| {
        p.set_settings(settings);
        p.set_fog_top(62.0);
    });
    on.finish.with(|p| p.set_settings(settings));
    let base = render(&mut gpu, &mut plain.phases, &cam);
    let graded = render(&mut gpu, &mut on.phases, &cam);
    assert!(on.phases.scene_mode());
    assert_eq!(on.atmosphere.with(|p| p.passes_recorded()), 1);
    assert_eq!(on.finish.with(|p| p.passes_recorded()), 1);
    assert_eq!(on.finish.with(|p| p.lut_bakes()), 1);
    assert_ne!(base, graded);
    // A second frame with the same settings does not re-bake the LUT.
    let _ = render(&mut gpu, &mut on.phases, &cam);
    assert_eq!(on.finish.with(|p| p.lut_bakes()), 1);
    // Releasing frees everything.
    on.atmosphere.with(DreamyAtmospherePhase::release);
    on.finish.with(DreamyFinishPhase::release);
    assert!(!on.atmosphere.with(|p| p.allocated()) && !on.finish.with(|p| p.allocated()));
}

#[test]
fn ground_fog_fills_the_low_ground_and_never_touches_a_player_at_fighting_range() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let cam = camera();
    let core = running_core(&gpu, &cam);
    let mut plain = rig(&gpu, &core, false);
    let base = render(&mut gpu, &mut plain.phases, &cam);
    let player = find(&base, |c| c[0] > 180 && c[1] < 80);
    let ground = find(&base, |c| c[1] > 100 && c[0] < 100 && c[2] < 100);
    // The sky well above the horizon (the top fifth looks 1 to 15 degrees up; just below the
    // horizon the far plane clips the ground and the fog rightly fills that "sky"), right of
    // the HUD rectangle.
    let sky: Vec<_> = find(&base, |c| c[2] > 200)
        .into_iter()
        .filter(|&(x, y)| y < H / 5 && x > W / 5 + 2)
        .collect();
    assert!(
        player.len() > 40,
        "the player is on screen: {}",
        player.len()
    );
    assert!(
        ground.len() > 1000,
        "the ground is on screen: {}",
        ground.len()
    );
    assert!(sky.len() > 1000, "the sky is on screen: {}", sky.len());

    let mut on = rig(&gpu, &core, true);
    on.atmosphere.with(|p| {
        p.set_settings(fog_only());
        // Below you: 12 blocks under the feet (74), so the ground at 60 is 2 blocks inside it.
        p.set_fog_top(62.0);
    });
    let fogged = render(&mut gpu, &mut on.phases, &cam);
    assert_eq!(on.atmosphere.with(|p| p.sub_passes()).0, 1, "one fog march");

    // The player keeps its pixels (the sRGB round trip may move a byte by one).
    let worst_player = player
        .iter()
        .map(|&(x, y)| diff(px(&base, x, y), px(&fogged, x, y)))
        .max()
        .unwrap_or(0);
    assert!(
        worst_player <= 1,
        "fog touched the player by {worst_player}"
    );
    // The sky above the layer is untouched.
    let worst_sky = sky
        .iter()
        .map(|&(x, y)| diff(px(&base, x, y), px(&fogged, x, y)))
        .max()
        .unwrap_or(0);
    assert!(worst_sky <= 1, "fog touched the sky by {worst_sky}");
    // The ground, inside the layer, turns toward the fog colour (pale blue): the average
    // green-to-blue change is large.
    let moved: f64 = ground
        .iter()
        .map(|&(x, y)| f64::from(diff(px(&base, x, y), px(&fogged, x, y))))
        .sum::<f64>()
        / ground.len() as f64;
    eprintln!("player worst {worst_player}, sky worst {worst_sky}, ground mean change {moved:.1}");
    assert!(
        moved > 40.0,
        "the low ground is fogged (mean change {moved})"
    );
}

/// Under a resource pack's custom sky the fog sea stays off the sky: looking down past a small
/// island into the void, the sky below the horizon keeps the colour the sky pass drew (orange
/// here, standing in for a sky pack), while the island inside the layer is fogged exactly as
/// without the custom sky. Without the switch (vanilla's sky) the fog sea covers that sky, and
/// from inside the fog both are the same.
#[test]
fn a_custom_sky_keeps_its_lower_half_while_the_eye_is_above_the_fog() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let cam = camera();
    let core = running_core(&gpu, &cam);
    // A 40 x 50 island at y = 60 reaching 40 blocks ahead, instead of the 600-block ground: it
    // fills the lower half, and between its edge and the horizon the view looks down into the
    // void.
    let island = SCENE_WGSL.replace(
        "c.x * 600.0 - 300.0, 60.0, c.y * 600.0 - 300.0",
        "c.x * 40.0 - 20.0, 60.0, c.y * 50.0 - 10.0",
    );
    let sky_pack = wgpu::Color {
        r: 0.9,
        g: 0.45,
        b: 0.1,
        a: 1.0,
    };
    let scene = |gpu: &Gpu, with_pack: bool| {
        let mut phases = PhaseList::new();
        phases.clear_color = sky_pack;
        phases.insert(
            PhaseOrder::TERRAIN_SOLID,
            Box::new(SceneGeometry::with_source(gpu, &island)),
        );
        let atmosphere = Handle::new(DreamyAtmospherePhase::new(core.clone()));
        if with_pack {
            phases.insert(order::WORLD, Box::new(atmosphere.clone()));
        }
        (phases, atmosphere)
    };
    let (mut plain, _) = scene(&gpu, false);
    let base = render(&mut gpu, &mut plain, &cam);
    let sky = find(&base, |c| {
        c[0] > 200 && c[1] > 90 && c[1] < 150 && c[2] < 60
    });
    let ground = find(&base, |c| c[1] > 100 && c[0] < 100 && c[2] < 100);
    assert!(sky.len() > 20_000, "the sky fills the view: {}", sky.len());
    assert!(
        ground.len() > 100,
        "the island is on screen: {}",
        ground.len()
    );

    // `keep`: how much of the sky the custom sky covers (1 = fully drawn, 0 = faded out).
    let mut frame = |keep: f32, top: f32| {
        let (mut phases, atmosphere) = scene(&gpu, true);
        atmosphere.with(|p| {
            p.set_settings(fog_only());
            p.set_fog_top(top);
            p.set_keep_sky(keep);
        });
        render(&mut gpu, &mut phases, &cam)
    };
    let worst = |pixels: &[(u32, u32)], a: &[u8], b: &[u8]| {
        pixels
            .iter()
            .map(|&(x, y)| diff(px(a, x, y), px(b, x, y)))
            .max()
            .unwrap_or(0)
    };
    // The fog 12 blocks under the feet: the eye is far above it.
    let sea = frame(0.0, 62.0);
    let kept = frame(1.0, 62.0);
    let covered = sky
        .iter()
        .filter(|&&(x, y)| diff(px(&base, x, y), px(&sea, x, y)) > 40)
        .count();
    eprintln!("fog sea covers {covered} of {} sky pixels", sky.len());
    assert!(
        covered > sky.len() / 4,
        "without a custom sky the fog sea covers the sky below the horizon"
    );
    // The sky keeps its pixels. Only on the island's outline may one differ: the half-resolution
    // fog's upsample gives a sky pixel the island's fog when all its low-resolution neighbours
    // are island (as at every outline, sky or not).
    let near_island = |&(x, y): &(u32, u32)| {
        ground
            .iter()
            .any(|&(gx, gy)| gx.abs_diff(x) <= 3 && gy.abs_diff(y) <= 3)
    };
    let moved: Vec<_> = sky
        .iter()
        .filter(|&&(x, y)| diff(px(&base, x, y), px(&kept, x, y)) > 1)
        .collect();
    eprintln!(
        "custom sky kept: {} sky pixels moved, all on the outline",
        moved.len()
    );
    assert!(
        moved.iter().all(|p| near_island(p)) && moved.len() * 1000 < sky.len(),
        "the custom sky moved away from the island's outline: {moved:?}"
    );
    let ground_fog = worst(&ground, &base, &kept);
    assert!(ground_fog > 40, "the island is still fogged ({ground_fog})");
    assert_eq!(
        worst(&ground, &sea, &kept),
        0,
        "the island's fog does not depend on the sky"
    );
    // From inside the fog the sky is fogged either way.
    let inside_sea = frame(0.0, 80.0);
    let inside_kept = frame(1.0, 80.0);
    assert_eq!(
        inside_sea, inside_kept,
        "inside the fog the switch changes nothing"
    );
    // A custom sky partly faded out keeps part of the sky: the fog sea fades back in with its
    // layers, steadily (no pop when the last one fades out), and the island's fog never depends
    // on it. (The fog blends in linear light, so half the cover is not half the display levels.)
    let half = frame(0.5, 62.0);
    let most = frame(0.75, 62.0);
    let total = |frame: &[u8]| -> i64 {
        sky.iter()
            .map(|&(x, y)| i64::from(diff(px(&base, x, y), px(frame, x, y))))
            .sum()
    };
    let (sea_total, half_total, most_total) = (total(&sea), total(&half), total(&most));
    eprintln!(
        "sky covered 0 / 0.5 / 0.75: the fog sea moves it {sea_total} / {half_total} / \
         {most_total} levels"
    );
    assert!(
        half_total * 10 > sea_total * 3 && half_total * 10 < sea_total * 9,
        "half the cover keeps part of the sky: {half_total} of {sea_total}"
    );
    assert!(
        most_total > 0 && most_total < half_total,
        "more cover keeps more of the sky: {most_total} vs {half_total}"
    );
    assert_eq!(
        worst(&ground, &sea, &half),
        0,
        "the island's fog does not depend on the cover"
    );
}

/// The client prepends local camera effects to the view matrix: vanilla `orientCamera`'s 0.1-block
/// step back, and in F5 the whole orbit (4 blocks). Rays must start at that true eye. Looking up
/// 50 degrees over a fog layer below, nothing may be fogged, with either offset. (The first
/// version reconstructed sky directions from the camera-relative origin and a point 0.1 block in
/// front of the eye: in the real client that fogged the sky above a player looking up, which the
/// night pictures showed as a grey plane; the synthetic tests above have no offset and missed it.)
#[test]
fn fog_rays_start_at_the_true_eye_with_the_clients_camera_offsets() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    for back in [0.1_f32, 4.0] {
        let mut cam = Camera {
            pos: DVec3::new(0.0, 75.62, 0.0),
            yaw: -90.0,
            pitch: -50.0,
            ..Camera::default()
        };
        cam.set_viewport(W, H);
        cam.update();
        cam.view = glam::Mat4::from_translation(Vec3::new(0.0, 0.0, -back)) * cam.view;
        let core = running_core(&gpu, &cam);
        let mut plain = rig(&gpu, &core, false);
        let base = render(&mut gpu, &mut plain.phases, &cam);
        let mut on = rig(&gpu, &core, true);
        on.atmosphere.with(|p| {
            p.set_settings(fog_only());
            p.set_fog_top(62.0);
        });
        let fogged = render(&mut gpu, &mut on.phases, &cam);
        assert_eq!(
            on.atmosphere.with(|p| p.sub_passes()).0,
            1,
            "the fog marched"
        );
        // Everything on screen is sky above the horizon (the HUD rectangle excepted, which is
        // drawn after the pack anyway).
        let worst = (0..H)
            .flat_map(|y| (0..W).map(move |x| (x, y)))
            .map(|(x, y)| diff(px(&base, x, y), px(&fogged, x, y)))
            .max()
            .unwrap_or(0);
        eprintln!("camera {back} blocks back: worst change {worst}");
        assert!(
            worst <= 1,
            "fog on an upward sky with a {back}-block camera offset: {worst}"
        );
    }
}

/// A pillar standing in the sky to the right of the sun: from the eye, its left face at x = 5,
/// 30 blocks ahead, 4 wide, from y = 40 to 160 (well above and below the view).
const PILLAR_WGSL: &str = r"
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
    var c = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let k = c[index];
    // Facing the camera, in camera-relative space.
    let rel = vec3<f32>(-5.0 - k.x * 4.0, -40.0 + k.y * 120.0, 30.0);
    return camera.view_proj * vec4<f32>(rel, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.3, 0.35, 1.0);
}
";

struct Pillar {
    pipeline: wgpu::RenderPipeline,
}

impl Pillar {
    fn new(gpu: &Gpu) -> Pillar {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sb test pillar"),
                source: wgpu::ShaderSource::Wgsl(PILLAR_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sb test pillar"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sb test pillar"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
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
        Pillar { pipeline }
    }
}

impl RenderPhase for Pillar {
    fn name(&self) -> &'static str {
        "test/pillar"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = gpu.begin_pass(enc, view, "test/pillar", DepthAttachment::Clear);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.draw(0..6, 0..1);
    }
}

/// The light rays: the sun straight ahead, a pillar to its right. The sky beyond the pillar (on
/// the far side from the sun, in its shadow) gets clearly less ray light than the same spot
/// mirrored to the sun's left, where nothing blocks it; and the light is centred on the sun.
#[test]
fn light_rays_stream_past_occluders_and_leave_shadows_behind_them() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let mut cam = Camera {
        pos: DVec3::new(0.0, 100.0, 0.0),
        ..Camera::default()
    };
    cam.set_viewport(W, H);
    cam.update();
    let core = ShaderCore::new();
    core.set_active(Some(PackId::Dreamy));
    let sun = Vec3::new(0.0, 0.02, 1.0).normalize();
    core.set_frame(
        &gpu,
        PackFrame {
            in_world: true,
            eye: cam.pos,
            view: cam.view,
            proj: cam.proj,
            far: cam.far,
            width: W,
            height: H,
            sun_dir: sun,
            moon_dir: -sun,
            light_dir: sun,
            light_is_sun: true,
            light_strength: 1.0,
            fog_color_display: Vec3::new(0.75, 0.85, 1.0),
            ..PackFrame::default()
        },
    );
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.2,
        g: 0.3,
        b: 0.5,
        a: 1.0,
    };
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(Pillar::new(&gpu)));
    let atmosphere = Handle::new(DreamyAtmospherePhase::new(core.clone()));
    phases.insert(order::WORLD, Box::new(atmosphere.clone()));
    let base = {
        atmosphere.with(|p| p.set_settings(DreamySettings::default()));
        render(&mut gpu, &mut phases, &cam)
    };
    atmosphere.with(|p| {
        p.set_settings(DreamySettings {
            // Moderate, so nothing clips to white and the comparison means something.
            rays: Some(mc_render::shaderpack::dreamy::RaySettings {
                strength: 0.3,
                length: 1.0,
            }),
            ..DreamySettings::default()
        });
    });
    let lit = render(&mut gpu, &mut phases, &cam);
    assert_eq!(atmosphere.with(|p| p.sub_passes()).1, 1, "one ray pass");
    let added = |x: u32, y: u32| -> i32 {
        let (a, b) = (px(&base, x, y), px(&lit, x, y));
        (0..3).map(|c| i32::from(b[c]) - i32::from(a[c])).sum()
    };
    // The sun is at the centre. The pillar spans 5..9 blocks right at 30 ahead: tan 0.167..0.3
    // against a half-width of tan 1.245 (70 degree vertical FOV, 16:9), i.e. 21..38 pixels right
    // of the centre at 320 wide. A sky pixel just past it (45 px) looks at the sun across the
    // pillar for over a third of the way; the mirrored pixel looks across open sky.
    let (cx, cy) = (W / 2, H / 2);
    let shadow_x = cx + 45;
    let mirror_x = cx - 45;
    let centre = added(cx, cy);
    let shadowed = added(shadow_x, cy);
    let open = added(mirror_x, cy);
    let far_open = added(cx - W * 36 / 100, cy);
    eprintln!(
        "ray light added: centre {centre}, open side {open}, behind the pillar {shadowed}, \
         far open {far_open}"
    );
    assert!(
        centre > open && open > far_open,
        "the light falls off away from the sun"
    );
    assert!(
        far_open > 4,
        "open sky away from the sun still gets rays ({far_open})"
    );
    assert!(
        (shadowed as f32) < 0.8 * open as f32,
        "the pillar's shadow is darker than the open side ({shadowed} vs {open})"
    );
}

/// The pack at (approximately) its menu defaults.
fn default_like() -> DreamySettings {
    DreamySettings {
        grade: busy_grade(),
        haze: 0.3,
        haze_start: 32.0,
        sky_tint: [1.0, 0.7, 0.85],
        sky_tint_strength: 0.3,
        rays: Some(mc_render::shaderpack::dreamy::RaySettings {
            strength: 0.25,
            length: 0.6,
        }),
        glow: 0.28,
        glow_radius: 0.6,
        halation: 0.15,
        vignette: 0.15,
        grain: 0.1,
        ..fog_only()
    }
}

/// GPU cost at 1080p on a worst case: the fog layer under the whole lower half of the screen
/// (every fog ray there marches), the sun on screen (the rays run), a noise-textured scene (no
/// framebuffer compression shortcut). Run with `MC_GPU_TIMING=1`; timings are per phase, from
/// timestamp queries, and are provisional while other work shares the GPU.
#[test]
#[ignore = "GPU timing run; needs MC_GPU_TIMING=1"]
fn dreamy_gpu_cost_at_1080p() {
    const FRAMES: u32 = 300;
    let (w, h) = (1920_u32, 1080_u32);
    let Some(mut gpu) = Gpu::headless(
        w,
        h,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    )
    .ok() else {
        eprintln!("SKIP: no adapter");
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("SKIP: set MC_GPU_TIMING=1 (and the adapter needs timestamp queries)");
        return;
    }
    let mut cam = Camera {
        pos: DVec3::new(0.0, 75.62, 0.0),
        pitch: 12.0,
        ..Camera::default()
    };
    cam.set_viewport(w, h);
    cam.update();
    let sun = Vec3::new(0.0, 0.15, 1.0).normalize();
    let frame_for = |core: &ShaderCore, gpu: &Gpu| {
        core.set_frame(
            gpu,
            PackFrame {
                in_world: true,
                eye: cam.pos,
                view: cam.view,
                proj: cam.proj,
                far: cam.far,
                width: w,
                height: h,
                sun_dir: sun,
                moon_dir: -sun,
                light_dir: sun,
                light_is_sun: true,
                light_strength: 1.0,
                fog_color_display: Vec3::new(0.75, 0.85, 1.0),
                ..PackFrame::default()
            },
        );
    };
    let noisy = SCENE_WGSL.replace(
        "return vec4<f32>(in.colour, 1.0);",
        "var x = u32(in.clip.x) * 73856093u ^ u32(in.clip.y) * 19349663u;\n    \
         x = (x ^ (x >> 16u)) * 0x45d9f3bu;\n    \
         let n = f32(x & 255u) / 255.0;\n    \
         return vec4<f32>(in.colour * (0.75 + 0.5 * n), 1.0);",
    );
    let configs: [(&str, Option<DreamySettings>); 5] = [
        ("off", None),
        ("defaults", Some(default_like())),
        (
            "fog quarter res",
            Some(DreamySettings {
                fog: fog_only().fog.map(|f| FogSettings {
                    layer: FogLayer {
                        resolution: FogResolution::Quarter,
                        ..f.layer
                    },
                    ..f
                }),
                ..default_like()
            }),
        ),
        (
            "no fog, no rays",
            Some(DreamySettings {
                fog: None,
                rays: None,
                ..default_like()
            }),
        ),
        (
            "grade only",
            Some(DreamySettings {
                grade: busy_grade(),
                ..DreamySettings::default()
            }),
        ),
    ];
    eprintln!("{w}x{h}, {FRAMES} frames per row, mean GPU ms per frame:");
    for (label, settings) in configs {
        let core = ShaderCore::new();
        if settings.is_some() {
            core.set_active(Some(PackId::Dreamy));
        }
        frame_for(&core, &gpu);
        let mut phases = PhaseList::new();
        phases.clear_color = SKY;
        let geometry = SceneGeometry::with_source(&gpu, &noisy);
        phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(geometry));
        let atmosphere = Handle::new(DreamyAtmospherePhase::new(core.clone()));
        let finish = Handle::new(DreamyFinishPhase::new(core.clone()));
        if let Some(settings) = settings {
            atmosphere.with(|p| {
                p.set_settings(settings);
                p.set_fog_top(62.0);
            });
            finish.with(|p| p.set_settings(settings));
        }
        phases.insert(order::WORLD, Box::new(atmosphere.clone()));
        phases.insert(order::FINAL, Box::new(finish.clone()));
        for _ in 0..FRAMES {
            let target = gpu.acquire().expect("acquire");
            phases.render(&gpu, target.view(), &cam, 0.0);
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
            "{label:<16} | atmosphere {:>7.4} | finish {:>7.4} | scene blit {:>7.4} | frame {:>7.4} \
             | frames {}",
            mean_ms(DreamyAtmospherePhase::NAME),
            mean_ms(DreamyFinishPhase::NAME),
            mean_ms("gpu.scene_blit"),
            timings.frame_ns.mean_ns() / 1e6,
            timings.frames,
        );
        assert!(timings.frames > 0, "{label}: no frame was timed");
    }
}

#[test]
fn the_finish_grades_the_world_but_never_the_hud() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let cam = camera();
    let core = running_core(&gpu, &cam);
    let mut plain = rig(&gpu, &core, false);
    let base = render(&mut gpu, &mut plain.phases, &cam);
    let mut on = rig(&gpu, &core, true);
    let settings = DreamySettings {
        grade: busy_grade(),
        glow: 0.5,
        halation: 0.3,
        vignette: 0.3,
        grain: 0.2,
        ..DreamySettings::default()
    };
    on.finish.with(|p| p.set_settings(settings));
    let graded = render(&mut gpu, &mut on.phases, &cam);
    // The HUD rectangle (top-left 20% x 20%) is drawn after the pack: identical bytes.
    for y in 0..(H / 5 - 1) {
        for x in 0..(W / 5 - 1) {
            assert_eq!(px(&base, x, y), px(&graded, x, y), "HUD pixel {x},{y}");
        }
    }
    // The world is graded.
    let changed = (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter(|&(x, y)| diff(px(&base, x, y), px(&graded, x, y)) > 4)
        .count();
    assert!(
        changed > (W * H / 2) as usize,
        "only {changed} pixels graded"
    );
}
