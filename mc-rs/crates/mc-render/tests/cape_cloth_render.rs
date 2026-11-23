//! A small real-GPU preview of the cape styles, independent of game assets and live servers.
//!
//! Set `MC_CAPE_RENDER_OUT` to save labeled idle/moving contact sheets and individual frames.
//! Columns are Vanilla, Wavy, Flowing, Windy, Draped. Each sheet compares standing/sneaking
//! at t=0.35/1.10 seconds. Idle has zero chase; moving has an explicit fixed chase offset.
//! This verifies visible geometry/animation, not frame-rate performance.

use glam::Vec3;
use mc_render::entity::anim::{AnimParams, EntityModel};
use mc_render::entity::layers::{emit_cape, CapeCloth, CapeClothStyle, CapeState};
use mc_render::entity::models::biped::PlayerModel;
use mc_render::entity::render::{entity_matrix, EntityPhase, EntityTransform};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const SIZE: u32 = 256;
const TIMES: [f64; 2] = [0.35, 1.10];
const STYLES: [(&str, Option<CapeClothStyle>); 5] = [
    ("vanilla", None),
    ("wavy", Some(CapeClothStyle::Wavy)),
    ("flowing", Some(CapeClothStyle::Flowing)),
    ("windy", Some(CapeClothStyle::Windy)),
    ("draped", Some(CapeClothStyle::Draped)),
];

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        SIZE,
        SIZE,
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
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

/// The vanilla 10x16x1 cape UV net: red fabric with gold borders and a white center stripe.
/// The broad pattern makes folds and stretched/broken UVs readable without checker noise.
/// Edge faces are gold, top/bottom magenta. Unused texture is transparent so an
/// incorrect atlas offset cannot silently render an equally plausible solid-colored cape.
fn cape_texture() -> Vec<u8> {
    let mut pixels = vec![0; 64 * 32 * 4];
    for y in 0..17 {
        for x in 0..22 {
            let color = if y == 0 && (1..21).contains(&x) {
                [245, 40, 230, 255]
            } else if y > 0 && (x == 0 || x == 11) {
                [245, 225, 25, 255]
            } else if y > 0 && ((1..11).contains(&x) || (12..22).contains(&x)) {
                let face_x = if x < 11 { x - 1 } else { x - 12 };
                if face_x == 0 || face_x == 9 || y == 16 {
                    [245, 205, 30, 255]
                } else if face_x == 4 || face_x == 5 {
                    [240, 235, 220, 255]
                } else {
                    [220, 35, 50, 255]
                }
            } else {
                [0, 0, 0, 0]
            };
            pixels[(y * 64 + x) * 4..][..4].copy_from_slice(&color);
        }
    }
    pixels
}

fn render(
    gpu: &mut Gpu,
    texture: &[u8],
    style: Option<CapeClothStyle>,
    sneaking: bool,
    moving: bool,
    time: f64,
) -> Vec<u8> {
    let mut model = PlayerModel::new(0.0, false);
    let mut params = AnimParams::still();
    params.entity.child = false;
    params.entity.sneaking = sneaking;
    let base = entity_matrix(&EntityTransform {
        rel: Vec3::new(0.0, -1.45, 2.35),
        body_yaw: 35.0,
        ..Default::default()
    });
    let mut body = Vec::new();
    // The real renderer shares the posed body model with LayerCape. In particular this sets
    // the cape's sneak pivot; drawing a newly-created model here would miss that integration.
    model.pose_and_render(&params, base, &mut body);
    let state = CapeState {
        chase_dx: if moving { 0.08 } else { 0.0 },
        chase_dy: if moving { 0.04 } else { 0.0 },
        chase_dz: if moving { -0.3 } else { 0.0 },
        body_yaw: 35.0,
        sneaking,
        cloth: style.map(|style| CapeCloth {
            style,
            time_seconds: time,
            phase: 0.37,
            strength: 0.6,
            speed: 1.0,
            stiffness: 0.5,
        }),
        ..Default::default()
    };
    let mut cape = Vec::new();
    emit_cape(&mut model, &state, 0.0625, base, &mut cape);
    assert!(!cape.is_empty(), "the cape produced geometry");
    for vertex in &cape {
        assert!(vertex.pos.iter().all(|value| value.is_finite()));
        assert!(vertex.normal.iter().all(|value| value.is_finite()));
        assert!(vertex
            .uv
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
    }

    let mut phase = EntityPhase::new(gpu);
    phase.upload_texture(gpu, "body", &[130, 130, 130, 255].repeat(64 * 64), 64, 64);
    phase.upload_texture(gpu, "cape", texture, 64, 32);
    phase.begin();
    phase.push("body", &body);
    phase.push("cape", &cape);
    phase.finish(gpu);
    assert_eq!(
        phase.batch_count(),
        2,
        "body/cape use the normal two texture batches"
    );

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let mut camera = Camera::default();
    camera.set_viewport(SIZE, SIZE);
    camera.update();
    let frame = gpu.acquire().expect("acquire headless frame");
    phases.render(gpu, frame.view(), &camera, 0.0);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("read rendered cape pixels")
}

fn cape_pixels(pixels: &[u8]) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|pixel| {
            let max = *pixel[..3].iter().max().unwrap();
            let min = *pixel[..3].iter().min().unwrap();
            max > 70 && max - min > 50
        })
        .count()
}

fn changed_pixels(left: &[u8], right: &[u8]) -> usize {
    left.chunks_exact(4)
        .zip(right.chunks_exact(4))
        .filter(|(left, right)| left != right)
        .count()
}

/// Small fixed glyphs keep labels in the artifact without depending on a game/font asset.
fn glyph(ch: char) -> [u8; 7] {
    match ch {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 14],
        'I' => [31, 4, 4, 4, 4, 4, 31],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 25, 21, 19, 19, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '=' => [0, 0, 31, 0, 31, 0, 0],
        '.' => [0, 0, 0, 0, 0, 12, 12],
        _ => [0; 7],
    }
}

fn label(sheet: &mut [u8], width: usize, x: usize, y: usize, text: &str) {
    for (column, ch) in text.chars().enumerate() {
        for (row, bits) in glyph(ch.to_ascii_uppercase()).into_iter().enumerate() {
            for bit in 0..5 {
                if bits & (1 << (4 - bit)) == 0 {
                    continue;
                }
                for dy in 0..2 {
                    for dx in 0..2 {
                        let at = ((y + row * 2 + dy) * width + x + column * 12 + bit * 2 + dx) * 4;
                        sheet[at..at + 4].copy_from_slice(&[230, 230, 235, 255]);
                    }
                }
            }
        }
    }
}

fn save_frames(frames: &[Vec<u8>]) {
    let Some(directory) = std::env::var_os("MC_CAPE_RENDER_OUT") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory).expect("create cape preview directory");
    const LEFT: usize = 100;
    const TOP: usize = 44;
    let width = LEFT + SIZE as usize * STYLES.len();
    let height = TOP + SIZE as usize * 4;
    for (condition, frames) in frames.chunks_exact(STYLES.len() * 4).enumerate() {
        let condition_name = if condition == 0 { "idle" } else { "moving" };
        let mut sheet = [0, 0, 0, 255].repeat(width * height);
        label(&mut sheet, width, 8, 5, condition_name);
        for (column, (style, _)) in STYLES.iter().enumerate() {
            label(
                &mut sheet,
                width,
                LEFT + column * SIZE as usize + 75,
                23,
                style,
            );
        }
        for (row, group) in frames.chunks_exact(STYLES.len()).enumerate() {
            let pose = if row < 2 { "stand" } else { "sneak" };
            label(&mut sheet, width, 8, TOP + row * SIZE as usize + 100, pose);
            label(
                &mut sheet,
                width,
                8,
                TOP + row * SIZE as usize + 122,
                if row % 2 == 0 { "T=0.35" } else { "T=1.10" },
            );
            for (column, pixels) in group.iter().enumerate() {
                let name = format!(
                    "{}-{condition_name}-{pose}-t{}.png",
                    STYLES[column].0,
                    row % 2
                );
                image::save_buffer(
                    directory.join(name),
                    pixels,
                    SIZE,
                    SIZE,
                    image::ColorType::Rgba8,
                )
                .expect("write cape frame");
                for y in 0..SIZE as usize {
                    let source = y * SIZE as usize * 4;
                    let dest =
                        ((TOP + row * SIZE as usize + y) * width + LEFT + column * SIZE as usize)
                            * 4;
                    sheet[dest..dest + SIZE as usize * 4]
                        .copy_from_slice(&pixels[source..source + SIZE as usize * 4]);
                }
            }
        }
        image::save_buffer(
            directory.join(format!("{condition_name}-contact-sheet.png")),
            &sheet,
            width as u32,
            height as u32,
            image::ColorType::Rgba8,
        )
        .expect("write cape contact sheet");
    }
    std::fs::write(directory.join("layout.txt"),
        "Columns: Vanilla | Wavy | Flowing | Windy | Draped\nEach sheet: standing t=0.35/1.10, then sneaking t=0.35/1.10 seconds.\nIdle: exactly zero chase in all axes. Moving: chase (0.08, 0.04, -0.3), held constant so only cloth time changes.\nSynthetic red cape with gold borders and a white center stripe, back-quarter view. GPU correctness preview, not a performance benchmark.\n",
    ).expect("write cape preview layout");
}

#[test]
fn cape_styles_are_still_at_idle_and_animate_with_movement_in_both_poses() {
    let Some(mut gpu) = headless() else { return };
    let texture = cape_texture();
    let mut frames = Vec::with_capacity(STYLES.len() * 8);
    for moving in [false, true] {
        for sneaking in [false, true] {
            for time in TIMES {
                for (_, style) in STYLES {
                    frames.push(render(&mut gpu, &texture, style, sneaking, moving, time));
                }
            }
        }
    }
    // Save before image comparisons so a failing idle/animation assertion remains inspectable.
    save_frames(&frames);
    for (index, pixels) in frames.iter().enumerate() {
        let coverage = cape_pixels(pixels);
        assert!(
            coverage > 100,
            "frame {index}: cape is visible ({coverage} colored pixels)"
        );
        assert!(
            coverage < (SIZE * SIZE / 2) as usize,
            "frame {index}: cape must not fill the frame"
        );
    }
    for (condition, condition_frames) in frames.chunks_exact(STYLES.len() * 4).enumerate() {
        for pose_start in [0, STYLES.len() * 2] {
            for (style, (name, _)) in STYLES.iter().enumerate() {
                let first = &condition_frames[pose_start + style];
                let second = &condition_frames[pose_start + STYLES.len() + style];
                if condition == 0 || style == 0 {
                    assert_eq!(
                        changed_pixels(first, second),
                        0,
                        "{name}: idle cloth and vanilla must be invariant to clock time"
                    );
                } else {
                    assert!(
                        changed_pixels(first, second) > 10,
                        "{name} must visibly animate while moving"
                    );
                    assert!(
                        changed_pixels(first, &condition_frames[pose_start]) > 10,
                        "{name} differs from rigid vanilla while moving"
                    );
                }
            }
        }
        for (style, (name, _)) in STYLES.iter().enumerate() {
            assert!(
                changed_pixels(
                    &condition_frames[style],
                    &condition_frames[STYLES.len() * 2 + style]
                ) > 100,
                "{name}: sneaking must move body/cape geometry"
            );
        }
    }
}
