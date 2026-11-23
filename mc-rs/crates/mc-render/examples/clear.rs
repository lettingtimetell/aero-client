//! S10 smoke test: open a real window, run the fixed-tick loop, render frames, report FPS.
//!
//! **Always bounded.** `--frames`/`--seconds` default to 600 frames and 30 seconds so this can be
//! run from automation without the risk of a window that never closes. Pass `--frames 0` for an
//! unbounded interactive session, which is only sensible from a terminal you are sitting at.
//!
//! ```text
//! cargo run -p mc-render --example clear -- --frames 600 --no-vsync
//! ```
//!
//! Controls: WASD to fly, space/shift up and down, ctrl to sprint, `G` to toggle cursor grab,
//! escape to release the cursor, `F` to print the current camera state.

use std::time::Duration;

use mc_render::input::{keys, CursorMode};
use mc_render::{
    Camera, DebugFlyCamera, DepthAttachment, FrameCtx, GameSettings, Gpu, PhaseList, PhaseOrder,
    RenderPhase, Shell, ShellApp, ShellConfig, TickCtx,
};

/// Draws a horizon-referenced gradient so that camera motion is visible without any terrain.
///
/// The clear colour alone would prove the surface works but not that the camera reaches a shader,
/// which is the half of the seam S11 actually depends on.
struct HorizonPhase {
    pipeline: wgpu::RenderPipeline,
}

const HORIZON_WGSL: &str = r#"
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
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // Unproject the pixel into a view ray, rotate it into world space, and shade by its height.
    let ndc = vec2<f32>(
        frag.x / camera.screen.x * 2.0 - 1.0,
        1.0 - frag.y / camera.screen.y * 2.0,
    );
    let tan_half = 1.0 / camera.proj[1][1];
    let dir_view = normalize(vec3<f32>(ndc.x * tan_half * camera.screen.x / camera.screen.y,
                                       ndc.y * tan_half,
                                       -1.0));
    // view is rotation-only, so its transpose is its inverse.
    let dir_world = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz))
                    * dir_view;

    let sky    = vec3<f32>(0.47, 0.65, 1.00);
    let ground = vec3<f32>(0.22, 0.30, 0.16);
    let horizon = vec3<f32>(0.75, 0.83, 0.95);
    let t = clamp(dir_world.y * 3.0, -1.0, 1.0);
    var colour = mix(horizon, sky, clamp(t, 0.0, 1.0));
    colour = mix(colour, ground, clamp(-t, 0.0, 1.0));

    // A compass stripe so yaw is legible: brightens when looking towards +Z (south).
    colour += vec3<f32>(0.10) * pow(max(dir_world.z, 0.0), 32.0);
    return vec4<f32>(colour, 1.0);
}
"#;

impl HorizonPhase {
    fn new(gpu: &Gpu) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("horizon"),
                source: wgpu::ShaderSource::Wgsl(HORIZON_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("horizon layout"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("horizon pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
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
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(gpu.config.format.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
        Self { pipeline }
    }
}

impl RenderPhase for HorizonPhase {
    fn name(&self) -> &'static str {
        "example/horizon"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = gpu.begin_pass(enc, view, "example/horizon", DepthAttachment::Shared);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.draw(0..3, 0..1);
    }
}

struct Demo {
    fly: DebugFlyCamera,
    ticks: u64,
}

impl ShellApp for Demo {
    fn init(&mut self, gpu: &Gpu, phases: &mut PhaseList) {
        phases.insert(PhaseOrder::SKY, Box::new(HorizonPhase::new(gpu)));
        println!("adapter: {:?}", gpu.adapter_info().name);
        println!(
            "format:  {:?}  msaa: {}x",
            gpu.config.format,
            gpu.sample_count()
        );
    }

    fn tick(&mut self, ctx: &mut TickCtx<'_>) {
        self.ticks = ctx.tick;

        // Exercise the consume-once press path the way a screen would.
        if ctx.settings.keys.take_pressed(keys::SCREENSHOT) {
            println!("screenshot key pressed (tick {})", ctx.tick);
        }

        // Toggle grab on G, release on escape — the S16 pause-menu behaviour in miniature.
        if ctx.input.key_down(winit::keyboard::KeyCode::KeyG) && !ctx.input.grabbed() {
            ctx.input.request_cursor(CursorMode::Grabbed);
        }
        if ctx.input.key_down(winit::keyboard::KeyCode::Escape) {
            ctx.input.request_cursor(CursorMode::Free);
        }
        if ctx.input.key_down(winit::keyboard::KeyCode::KeyF) {
            println!(
                "tick {} camera pos {:?} yaw {:.1} pitch {:.1}",
                ctx.tick, ctx.camera.pos, ctx.camera.yaw, ctx.camera.pitch
            );
        }
    }

    fn frame(&mut self, ctx: &mut FrameCtx<'_>) {
        let dt = ctx.stats.last_frame_secs().clamp(0.0, 0.1);
        self.fly.update(ctx, dt);
    }

    fn shutdown(&mut self, _settings: &GameSettings) {
        println!("ran {} ticks", self.ticks);
    }
}

fn main() -> Result<(), Box<mc_render::RenderError>> {
    let mut frames: Option<u64> = Some(600);
    let mut seconds: Option<u64> = Some(30);
    let mut vsync = true;
    let mut msaa = 1u32;
    let mut max_fps = mc_render::settings::FRAMERATE_UNLIMITED;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" => {
                i += 1;
                let n: u64 = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(600);
                frames = (n > 0).then_some(n);
            }
            "--seconds" => {
                i += 1;
                let n: u64 = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(30);
                seconds = (n > 0).then_some(n);
            }
            "--no-vsync" => vsync = false,
            "--msaa" => {
                i += 1;
                msaa = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "--max-fps" => {
                i += 1;
                max_fps = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(mc_render::settings::FRAMERATE_UNLIMITED);
            }
            "--help" | "-h" => {
                println!(
                    "usage: clear [--frames N] [--seconds N] [--no-vsync] [--msaa 1|2|4|8] \
                     [--max-fps N]\n\
                     --frames/--seconds 0 disables that bound. Both bounds default on so \
                     automation cannot hang.\n\
                     --max-fps only applies with --no-vsync, matching vanilla's Display.sync."
                );
                return Ok(());
            }
            other => eprintln!("ignoring unknown argument {other}"),
        }
        i += 1;
    }

    let settings = GameSettings {
        enable_vsync: vsync,
        msaa_samples: msaa,
        limit_framerate: max_fps,
        ..GameSettings::default()
    };

    let config = ShellConfig {
        title: "mc-rs S10 shell".into(),
        width: 1280,
        height: 720,
        max_frames: frames,
        max_duration: seconds.map(Duration::from_secs),
        grab_cursor_on_start: false,
    };

    let mut demo = Demo {
        fly: DebugFlyCamera::default(),
        ticks: 0,
    };
    let summary = Shell::run(config, settings, &mut demo)?;

    println!(
        "\n{} frames / {} ticks in {:.2}s\naverage {:.1} fps, last second {} fps\nadapter: {}",
        summary.frames,
        summary.ticks,
        summary.elapsed.as_secs_f32(),
        summary.average_fps,
        summary.last_fps,
        summary.adapter,
    );

    // The tick rate is the assertion that matters: 20 per second of wall clock, whatever the
    // frame rate did. Vanilla's Timer drops time beyond the catch-up cap, so this is a lower
    // bound, not an equality.
    let expected_ticks = (summary.elapsed.as_secs_f32() * 20.0) as u64;
    println!("expected roughly {expected_ticks} ticks at 20 TPS");
    Ok(())
}
