//! `AsciiPhase`, the ASCII Super Secret Setting, on a real GPU.
//!
//! Every test renders whole frames through a real [`PhaseList`] on a headless context and reads
//! them back, so what is asserted is what reaches the screen. They skip (printing `SKIP:`) when
//! the machine has no adapter.
//!
//! * Inactive: the frame is byte-identical to a frame without the phase, the list stays on its
//!   direct path, and the phase owns no GPU resource at all.
//! * Active: white cells hold exactly the ramp's densest glyph, black cells are empty paper, for
//!   the built-in ramp and for a font sheet's ramp.
//! * Switching off restores the frame byte for byte; `release` drops the pipelines and the ramp.
//! * Under 4x MSAA the characters survive a later phase's resolve.

use std::cell::RefCell;
use std::rc::Rc;

use mc_render::post::ascii::{
    cell_size, fallback_ramp, glyph_ramp, AsciiColors, AsciiPhase, AsciiSize, GlyphRamp,
};
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
struct Handle(Rc<RefCell<AsciiPhase>>);

impl Handle {
    fn new(phase: AsciiPhase) -> Handle {
        Handle(Rc::new(RefCell::new(phase)))
    }

    fn with<R>(&self, f: impl FnOnce(&mut AsciiPhase) -> R) -> R {
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
        AsciiPhase::NAME
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

/// Test scenes drawn over the clear. `EDGE` (the x where the white half ends) is prepended.
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

// White left of EDGE, black right of it.
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
    fn new(gpu: &Gpu, entry: &'static str, edge: u32) -> Pattern {
        let source = format!("const EDGE: f32 = {edge}.0;\n{PATTERN_WGSL}");
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rfx test pattern"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rfx test pattern"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rfx test pattern"),
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
        let mut pass = gpu.begin_pass(enc, view, "rfx test pattern", DepthAttachment::None);
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

/// A list with `entry`'s pattern as the world and, when given, the effect at its real slot.
fn list(gpu: &Gpu, entry: &'static str, edge: u32, ascii: Option<&Handle>) -> PhaseList {
    let mut phases = PhaseList::new();
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(Pattern::new(gpu, entry, edge)),
    );
    if let Some(ascii) = ascii {
        phases.insert(AsciiPhase::ORDER, Box::new(ascii.clone()));
    }
    phases
}

fn rgb(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 3] {
    let at = ((y * width + x) * 4) as usize;
    [pixels[at], pixels[at + 1], pixels[at + 2]]
}

/// An active effect in White on black with `font` (`None` = the built-in ramp).
fn white_ascii(size: AsciiSize, font: Option<(&[u8], u32, u32)>) -> Handle {
    let mut phase = AsciiPhase::new();
    phase.set_style(size, AsciiColors::White);
    phase.set_font(font);
    phase.set_active(true);
    Handle::new(phase)
}

/// The White ink byte: `0.93` in `ascii.wgsl`.
const INK: u8 = 237;

/// Check every cell of a white-left, black-right frame: white cells hold the ramp's densest glyph
/// (scaled to the cell), except the column against the black half, which holds the `|` edge
/// glyph; black cells are paper.
fn assert_glyph_cells(pixels: &[u8], width: u32, height: u32, edge: u32, ramp: &GlyphRamp) {
    let cell = cell_size(AsciiSize::Small, height);
    let densest = ramp.glyphs.len() as u32 - 1;
    assert!(ramp.edges, "the test ramps have edge glyphs");
    let bar = ramp.glyphs.len() as u32;
    let mut edge_ink = 0;
    let mut ink_pixels = 0;
    for y in 0..height {
        for x in 0..width {
            let got = rgb(pixels, width, x, y);
            let (cx, cy) = (x / cell, y / cell);
            let white_cell = (cx + 1) * cell <= edge;
            let black_cell = cx * cell >= edge;
            if black_cell {
                assert_eq!(got, [0, 0, 0], "black cell pixel ({x},{y})");
                continue;
            }
            if !white_cell {
                continue;
            }
            let (lx, ly) = (x - cx * cell, y - cy * cell);
            let tx = lx * ramp.glyph_width / cell;
            let ty = ly * ramp.glyph_height / cell;
            let against_black = (cx + 1) * cell == edge;
            let slot = if against_black { bar } else { densest };
            let at = (ty * ramp.width() + slot * ramp.glyph_width + tx) as usize;
            let ink = ramp.coverage[at] > 0;
            let want = if ink { [INK, INK, INK] } else { [0, 0, 0] };
            assert_eq!(got, want, "white cell ({cx},{cy}) pixel ({x},{y})");
            if against_black {
                edge_ink += u32::from(ink);
            } else {
                ink_pixels += u32::from(ink);
            }
        }
    }
    assert!(ink_pixels > 0, "no glyph ink at all");
    assert!(edge_ink > 0, "no `|` along the white half's edge");
}

/// A 128x128 font sheet: candidate glyph `i` of the ramp's list has `i` ink texels (row-major
/// from the cell's top-left).
fn synthetic_font() -> Vec<u8> {
    let mut rgba = vec![0u8; 128 * 128 * 4];
    for (i, c) in mc_render::post::ascii::CANDIDATES.chars().enumerate() {
        let code = c as u32;
        let (ox, oy) = ((code % 16) * 8, (code / 16) * 8);
        for n in 0..(i as u32).min(64) {
            let (x, y) = (n % 8, n / 8);
            let at = (((oy + y) * 128 + ox + x) * 4) as usize;
            rgba[at..at + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    rgba
}

// ------------------------------------------------------------------------------- tests

#[test]
fn an_inactive_effect_changes_no_byte_and_owns_nothing() {
    let Some(mut gpu) = headless(96, 64, 1) else {
        return;
    };
    let mut plain = list(&gpu, "fs_ramp", 48, None);
    let reference = frame(&mut gpu, &mut plain);

    let ascii = Handle::new(AsciiPhase::new());
    let mut with_ascii = list(&gpu, "fs_ramp", 48, Some(&ascii));
    assert!(!with_ascii.scene_mode(), "an idle effect must not opt in");
    let got = frame(&mut gpu, &mut with_ascii);
    assert_eq!(got, reference, "an idle effect changed the frame");
    assert!(!ascii.with(|a| a.targets_allocated()));
    assert!(!ascii.with(|a| a.pipelines_built()));
    assert!(!ascii.with(|a| a.ramp_built()));
    assert_eq!(ascii.with(|a| a.passes_recorded()), 0);
}

#[test]
fn white_cells_hold_the_densest_built_in_glyph_and_black_cells_are_paper() {
    let (width, height, edge) = (96u32, 64u32, 48u32);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    let ascii = white_ascii(AsciiSize::Small, None);
    let mut phases = list(&gpu, "fs_edge", edge, Some(&ascii));
    assert!(phases.scene_mode());
    let pixels = frame(&mut gpu, &mut phases);
    assert_glyph_cells(&pixels, width, height, edge, &fallback_ramp());
    assert_eq!(ascii.with(|a| a.passes_recorded()), 1);
    assert_eq!(ascii.with(|a| a.grid()), Some((12, 8, 8)));
}

#[test]
fn a_font_sheet_supplies_the_glyphs() {
    let (width, height, edge) = (96u32, 64u32, 48u32);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    let font = synthetic_font();
    let ramp = glyph_ramp(&font, 128, 128).expect("the synthetic sheet is a font");
    let ascii = white_ascii(AsciiSize::Small, Some((&font, 128, 128)));
    let mut phases = list(&gpu, "fs_edge", edge, Some(&ascii));
    let pixels = frame(&mut gpu, &mut phases);
    assert_glyph_cells(&pixels, width, height, edge, &ramp);
    // The built-in ramp's densest glyph differs, so this really is the font's glyph.
    assert_ne!(
        ramp.coverage,
        fallback_ramp().coverage,
        "the test font must not look like the built-in ramp"
    );
}

#[test]
fn switching_off_restores_the_frame_and_release_drops_everything() {
    let Some(mut gpu) = headless(96, 64, 1) else {
        return;
    };
    let mut plain = list(&gpu, "fs_ramp", 48, None);
    let reference = frame(&mut gpu, &mut plain);

    let ascii = white_ascii(AsciiSize::Medium, None);
    let mut phases = list(&gpu, "fs_ramp", 48, Some(&ascii));
    let on = frame(&mut gpu, &mut phases);
    assert_ne!(on, reference, "the effect drew nothing");
    assert!(ascii.with(|a| a.targets_allocated() && a.pipelines_built() && a.ramp_built()));

    // Off (another choice): the target goes at once, the frame is the scene again.
    ascii.with(|a| a.set_active(false));
    assert!(!phases.scene_mode());
    assert!(!ascii.with(|a| a.targets_allocated()));
    let off = frame(&mut gpu, &mut phases);
    assert_eq!(
        off, reference,
        "switching off must restore the frame exactly"
    );
    assert_eq!(
        ascii.with(|a| a.passes_recorded()),
        1,
        "an idle phase draws nothing"
    );

    // The mod is turned off: everything goes, and the font is asked for again.
    ascii.with(|a| a.release());
    assert!(!ascii.with(|a| a.pipelines_built() || a.ramp_built() || a.targets_allocated()));
    assert!(ascii.with(|a| a.needs_font()));
    assert_eq!(frame(&mut gpu, &mut phases), reference);
}

#[test]
fn the_characters_survive_msaa_and_a_later_resolve() {
    let (width, height, edge) = (96u32, 64u32, 48u32);
    let Some(mut gpu) = headless(width, height, 4) else {
        return;
    };
    if gpu.sample_count() == 1 {
        eprintln!("SKIP: adapter does not support 4x MSAA on this format");
        return;
    }
    let ascii = white_ascii(AsciiSize::Small, None);
    let mut phases = list(&gpu, "fs_edge", edge, Some(&ascii));
    // A later phase (the UI's place) whose own pass resolves the multisampled attachment.
    let mut trailing = OverlayPhase::new(&gpu);
    trailing.gamma = 0.0;
    trailing.kind = OverlayKind::None;
    phases.insert(PhaseOrder::UI, Box::new(trailing));
    let pixels = frame(&mut gpu, &mut phases);
    assert_glyph_cells(&pixels, width, height, edge, &fallback_ramp());
}

/// GPU cost of the effect at 1080p (provisional while other work shares the GPU).
#[test]
#[ignore = "benchmark; needs MC_GPU_TIMING=1"]
fn gpu_cost_at_1080p() {
    const FRAMES: u32 = 300;
    let (width, height) = (1920u32, 1080u32);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("SKIP: set MC_GPU_TIMING=1 (and the adapter needs timestamp queries)");
        return;
    }
    eprintln!("{width}x{height}, {FRAMES} frames per row, mean ms per frame:");
    eprintln!("size   | post/ascii | gpu.scene_blit | gpu.frame");
    for size in [
        None,
        Some(AsciiSize::Small),
        Some(AsciiSize::Medium),
        Some(AsciiSize::Large),
    ] {
        let ascii = Handle::new(AsciiPhase::new());
        if let Some(size) = size {
            ascii.with(|a| {
                a.set_style(size, AsciiColors::Color);
                a.set_font(None);
                a.set_active(true);
            });
        }
        let mut phases = list(&gpu, "fs_ramp", width / 2, Some(&ascii));
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
            "{:<6} | {:>10.4} | {:>14.4} | {:>9.4}",
            size.map_or("off".to_owned(), |s| format!("{s:?}")),
            mean_ms(AsciiPhase::NAME),
            mean_ms("gpu.scene_blit"),
            timings.frame_ns.mean_ns() / 1e6,
        );
    }
}
