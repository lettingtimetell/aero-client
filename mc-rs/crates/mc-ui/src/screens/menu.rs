//! The main menu: the rotating panorama, the wordmark, the splash text, the buttons.
//!
//! java: `client/gui/GuiMainMenu.java`
//!
//! # What is and is not reproduced
//!
//! The panorama uses vanilla's six-face cube, rotation schedule, square 120-degree projection,
//! 64-pass jittered accumulation, and transposed centre-crop framing. [`Panorama::draw_to_target`]
//! and [`PanoramaRenderPlan`] expose the CPU-side seam for vanilla's private 256x256 target and
//! final crop. The renderer's [`mc_render::pipeline::panorama_blur::PanoramaBlur`] owns the
//! snapshot-then-sample feedback loop for the seven `rotateAndBlurSkybox` passes: each pass
//! copies the target, then re-blends three offset copies of that snapshot. The root UI phase
//! submits the seed and foreground separately, so the blurred composite is painted before menu
//! gradients, title, splash, and buttons.
//!
//! The compatibility [`Panorama::draw`] path still maps the square target's crop directly into a
//! shared draw list for headless callers that do not have a private render target. A live menu
//! frame uses [`MainMenuScreen::draw_panorama_seed`] plus [`MainMenuScreen::draw_foreground`],
//! leaving the actual texture feedback loop in the renderer where wgpu can snapshot safely.

use chrono::{Datelike, Local};
use mc_assets::{Language, ResourceLocation, ResourceManager};

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{draw_dirt_background, TITLE_TEXTURE};
use crate::screens::Action;
use crate::widget::draw::{DrawList, UiCull, UiDepth, UiTexture};
use crate::widget::{draw_centered_string, draw_string, Button, Transform, WIDGETS_TEXTURE};

/// The six panorama faces, in vanilla's order.
// java: GuiMainMenu.java:75
pub const PANORAMA_FACES: [&str; 6] = [
    "textures/gui/title/background/panorama_0.png",
    "textures/gui/title/background/panorama_1.png",
    "textures/gui/title/background/panorama_2.png",
    "textures/gui/title/background/panorama_3.png",
    "textures/gui/title/background/panorama_4.png",
    "textures/gui/title/background/panorama_5.png",
];

/// Width and height of vanilla's private panorama framebuffer.
// java: GuiMainMenu.java:496 (`viewport(0, 0, 256, 256)`)
pub const PANORAMA_TARGET_SIZE: u32 = 256;

/// Number of jittered cube renders accumulated by `drawPanorama`.
// java: GuiMainMenu.java:397-399 (`i = 8`, `j < i * i`)
pub const PANORAMA_ACCUMULATION_SAMPLES: u32 = 64;

/// Number of self-sampling `rotateAndBlurSkybox` passes.
// java: GuiMainMenu.java:498-504
pub const PANORAMA_BLUR_PASSES: u32 = 7;

/// Number of overlapping full-target quads in one `rotateAndBlurSkybox` call.
// java: GuiMainMenu.java:479-489 (`i = 3`, `j < i`)
pub const PANORAMA_BLUR_TAPS: usize = 3;

/// One of the three source-over draws in `rotateAndBlurSkybox`.
///
/// `offset_u` is in normalised target-texture coordinates. The offset is deliberately on the
/// source **u** axis: vanilla's vertex data swaps the source axes (with the bottom-up texture
/// `glCopyTexSubImage2D` fills, that mirrors the image across its diagonal) before the three
/// taps are blended, so changing this to `v` (or dropping it) changes the blur direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanoramaBlurTap {
    /// Source-u offset, measured in 256x256 target texels.
    pub offset_u: f32,
    /// Alpha passed to vanilla's source-over blend for this tap.
    pub blend_alpha: f32,
}

impl PanoramaBlurTap {
    /// The literal tap descriptor from `GuiMainMenu.rotateAndBlurSkybox`.
    #[must_use]
    pub const fn vanilla(index: usize) -> Self {
        match index {
            0 => Self {
                offset_u: -1.0 / 256.0,
                blend_alpha: 1.0,
            },
            1 => Self {
                offset_u: 0.0,
                blend_alpha: 1.0 / 2.0,
            },
            2 => Self {
                offset_u: 1.0 / 256.0,
                blend_alpha: 1.0 / 3.0,
            },
            _ => panic!("vanilla panorama blur has exactly three taps"),
        }
    }
}

/// The fixed full-target quad and source mapping used by every vanilla blur pass.
///
/// Both arrays use [`DrawList::quad`] order: top-left, bottom-left, top-right, bottom-right.
/// `source_uv` is Java's literal GL `(s, t)`, whose `t = 0` is the bottom row of the texture
/// `glCopyTexSubImage2D` filled; a renderer whose textures store the top row first flips `t`.
/// Keeping this geometry in the UI contract lets a CPU test reject an identity blit that would
/// look plausible on a symmetric test texture while the GPU helper remains a renderer detail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanoramaBlurQuad {
    /// Full-target positions in clip space.
    pub screen_ndc: [[f32; 2]; 4],
    /// Source UVs before adding a [`PanoramaBlurTap::offset_u`].
    pub source_uv: [[f32; 2]; 4],
}

impl PanoramaBlurQuad {
    /// The literal source coordinates read from Java's four vertex calls.
    #[must_use]
    pub const fn vanilla() -> Self {
        Self {
            screen_ndc: [[-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, -1.0]],
            source_uv: [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
    }
}

/// CPU-visible contract for the second panorama blur stage.
///
/// A renderer records [`passes`] copies of this same three-tap descriptor, snapshotting the
/// previous target before each pass. This is intentionally a descriptor rather than a CPU image
/// filter: the six-face seed remains ordinary [`DrawList`] geometry, and the actual texture
/// feedback loop belongs to the renderer seam.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanoramaBlurPlan {
    /// The fixed private target dimensions.
    pub target_size: [u32; 2],
    /// Number of repeated self-sampling passes.
    pub passes: u32,
    /// The three source-over taps in one pass.
    pub taps: [PanoramaBlurTap; PANORAMA_BLUR_TAPS],
    /// Full-target positions and the swapped-axis source mapping.
    pub quad: PanoramaBlurQuad,
}

impl PanoramaBlurPlan {
    /// Construct the exact `rotateAndBlurSkybox` contract from 1.8.9.
    #[must_use]
    pub const fn vanilla() -> Self {
        Self {
            target_size: [PANORAMA_TARGET_SIZE; 2],
            passes: PANORAMA_BLUR_PASSES,
            taps: [
                PanoramaBlurTap::vanilla(0),
                PanoramaBlurTap::vanilla(1),
                PanoramaBlurTap::vanilla(2),
            ],
            quad: PanoramaBlurQuad::vanilla(),
        }
    }
}

/// The renderer-facing contract for vanilla's private panorama target and final composite.
///
/// [`Panorama::draw_to_target`] emits the first stage into a 256×256 GUI-space draw list. A
/// renderer can draw that list into its private target, apply [`blur_passes`](Self::blur_passes)
/// self-sampling passes, then composite it over the window with these UVs. The UV order is the
/// order [`DrawList::quad`] accepts: top-left, bottom-left, top-right, bottom-right. They are
/// Java's literal GL texture coordinates (`t = 0` is the bottom row of the copied framebuffer),
/// and vanilla's final quad samples the copy the 7th pass took, i.e. the result of six passes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanoramaRenderPlan {
    /// Fixed private render-target dimensions.
    pub target_size: [u32; 2],
    /// Number of accumulated jitter samples in the first stage.
    pub accumulation_samples: u32,
    /// Number of three-tap rotating blur passes in the second stage.
    pub blur_passes: u32,
    /// UVs for the final full-window composite quad.
    pub composite_uv: [[f32; 2]; 4],
}

impl PanoramaRenderPlan {
    /// Return the fixed second-stage descriptor shared by every window size.
    #[must_use]
    pub const fn blur_plan() -> PanoramaBlurPlan {
        PanoramaBlurPlan::vanilla()
    }

    /// Construct the exact fixed-size target and aspect-dependent final crop from
    /// `GuiMainMenu.renderSkybox`.
    #[must_use]
    pub fn for_window(width: i32, height: i32) -> Self {
        let width = width.max(1) as f32;
        let height = height.max(1) as f32;
        let scale = 120.0 / width.max(height);
        let crop_u = height * scale / PANORAMA_TARGET_SIZE as f32;
        let crop_v = width * scale / PANORAMA_TARGET_SIZE as f32;

        Self {
            target_size: [PANORAMA_TARGET_SIZE; 2],
            accumulation_samples: PANORAMA_ACCUMULATION_SAMPLES,
            blur_passes: PANORAMA_BLUR_PASSES,
            // java: GuiMainMenu.java:515-522, reordered from its GL winding into DrawList order.
            composite_uv: [
                [0.5 + crop_u, 0.5 + crop_v],
                [0.5 - crop_u, 0.5 + crop_v],
                [0.5 + crop_u, 0.5 - crop_v],
                [0.5 - crop_u, 0.5 - crop_v],
            ],
        }
    }
}

/// One of vanilla's 64 jittered first-stage samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanoramaSample {
    /// Model-view translation applied before the animated pitch/yaw rotations.
    pub translation: [f32; 2],
    /// Integer vertex alpha (`255 / (sample + 1)`).
    pub alpha: u8,
}

impl PanoramaSample {
    /// Produce one literal iteration of `GuiMainMenu.drawPanorama`'s `j` loop.
    #[must_use]
    pub fn vanilla(sample: u32) -> Self {
        debug_assert!(sample < PANORAMA_ACCUMULATION_SAMPLES);
        Self {
            translation: [
                ((sample % 8) as f32 / 8.0 - 0.5) / 64.0,
                ((sample / 8) as f32 / 8.0 - 0.5) / 64.0,
            ],
            alpha: (255 / (sample + 1)) as u8,
        }
    }
}

/// `texts/splashes.txt`, the splash-message list.
pub const SPLASHES: &str = "texts/splashes.txt";

/// The one splash vanilla refuses to show, identified by its Java `String.hashCode`.
///
/// The string is `"Now with 15% more MOJANG!"`… except that it is not: the constant is a
/// leftover and the hash matches nothing in the shipped 1.8.9 `splashes.txt`, so in practice
/// the `while` loop always exits on its first iteration. Reproduced anyway, because a resource
/// pack can add a line that hashes to it and vanilla would then skip it.
// java: GuiMainMenu.java:119
pub const BANNED_SPLASH_HASH: i32 = 125_780_783;

/// Java's `String.hashCode`: `s[0]*31^(n-1) + s[1]*31^(n-2) + … + s[n-1]`, over **UTF-16 code
/// units**, with wrapping `i32` arithmetic.
#[must_use]
pub fn java_hash_code(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// The splash-message pool.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Splashes {
    lines: Vec<String>,
}

impl Splashes {
    /// Parse `splashes.txt`: one message per line, trimmed, blanks dropped.
    // java: GuiMainMenu.java:99
    #[must_use]
    pub fn parse(text: &str) -> Self {
        Self {
            lines: text
                .lines()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
        }
    }

    /// Load from the resource packs. A missing file yields an empty pool — vanilla catches the
    /// `IOException` and keeps `"missingno"`.
    #[must_use]
    pub fn load(assets: &ResourceManager) -> Self {
        match assets.load(&ResourceLocation::minecraft(SPLASHES)) {
            Ok(bytes) => Self::parse(&String::from_utf8_lossy(&bytes)),
            Err(_) => Self::default(),
        }
    }

    /// How many messages there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Whether the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Pick the splash for a given random draw.
    ///
    /// `next_index` is `Random.nextInt(size)`; the loop re-draws while the pick is banned, so
    /// this takes an iterator of draws and mirrors that. An empty pool gives `"missingno"`,
    /// vanilla's initial value.
    // java: GuiMainMenu.java:113
    #[must_use]
    pub fn pick(&self, draws: &mut impl Iterator<Item = usize>) -> String {
        if self.lines.is_empty() {
            return "missingno".to_owned();
        }
        for i in draws.by_ref() {
            let s = &self.lines[i % self.lines.len()];
            if java_hash_code(s) != BANNED_SPLASH_HASH {
                return s.clone();
            }
        }
        "missingno".to_owned()
    }

    /// The three calendar overrides, applied after the random pick.
    ///
    /// `month` is 1-based, matching the Java's `calendar.get(2) + 1`.
    // java: GuiMainMenu.java:198
    #[must_use]
    pub fn calendar_override(month: u32, day: u32) -> Option<&'static str> {
        match (month, day) {
            (12, 24) => Some("Merry X-mas!"),
            (1, 1) => Some("Happy new year!"),
            (10, 31) => Some("OOoooOOOoooo! Spooky!"),
            _ => None,
        }
    }
}

/// Return the current host-local date.
///
/// Vanilla asks the JVM's local `java.util.Calendar`
/// for these fields. `chrono::Local` uses the platform timezone provider in-process, including
/// daylight-saving transitions on Windows, while the crate itself remains `unsafe_code`-free.
/// Tests use [`MainMenuScreen::init_on_date`] to inject deterministic boundary dates.
#[must_use]
fn current_local_month_day() -> (u32, u32) {
    let now = Local::now();
    (now.month(), now.day())
}

/// Convert an injected instant to a civil date after applying a caller-supplied offset.
///
/// Production uses [`current_local_month_day`], which delegates timezone and DST handling to the
/// host provider through `chrono::Local`. Keeping this arithmetic helper separate makes the
/// Christmas/New Year's midnight seam deterministic without invoking a process or consulting
/// global mutable state.
#[must_use]
#[cfg(test)]
fn month_day_from_system_time(now: std::time::SystemTime, offset_seconds: i32) -> (u32, u32) {
    let seconds = now
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs() as i64);
    month_day_from_unix_seconds_with_offset(seconds, offset_seconds)
}

/// Convert a Unix timestamp to a civil `(month, day)` after applying a fixed local offset.
///
/// A fixed offset is not used by production (the host provider handles DST), but makes timezone
/// boundary behavior deterministic and injectable in tests. In particular, a timestamp just after
/// UTC midnight can still belong to the previous local date.
#[must_use]
#[cfg(test)]
fn month_day_from_unix_seconds_with_offset(seconds: i64, offset_seconds: i32) -> (u32, u32) {
    let local_seconds = seconds.saturating_add(i64::from(offset_seconds));
    month_day_from_unix_days(local_seconds.div_euclid(86_400))
}

/// Convert a Unix day number to a Gregorian `(month, day)`.
///
/// This is the standard civil-from-days transform with its epoch shifted to 1970-01-01.
#[must_use]
#[cfg(test)]
fn month_day_from_unix_days(days: i64) -> (u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    (month as u32, day as u32)
}

/// The splash's pulsing scale factor.
///
/// `1.8 - |sin(ms % 1000 / 1000 * 2π)| * 0.1`, then scaled to `100 / (width + 32)`. Note
/// vanilla's operator precedence: the `0.1` multiplies the *sine*, then the absolute value is
/// taken of the product — so the term is always subtracted, and the factor swings between 1.7
/// and 1.8 rather than 1.7 and 1.9.
// java: GuiMainMenu.java:563
#[must_use]
pub fn splash_scale(system_time_ms: u64, splash_width: i32) -> f32 {
    let phase = (system_time_ms % 1000) as f32 / 1000.0 * std::f32::consts::PI * 2.0;
    let f = 1.8 - (mc_math::sin(phase) * 0.1).abs();
    f * 100.0 / (splash_width + 32) as f32
}

/// Button ids, matching vanilla's.
mod ids {
    pub const OPTIONS: i32 = 0;
    pub const SINGLEPLAYER: i32 = 1;
    pub const MULTIPLAYER: i32 = 2;
    pub const QUIT: i32 = 4;
    pub const LANGUAGE: i32 = 5;
    pub const REALMS: i32 = 14;
}

/// `gluPerspective`: the OpenGL-convention right-handed projection, with the depth range
/// `-1..=1` rather than wgpu's `0..=1`.
///
/// Written out rather than taken from `glam` because `Mat4::perspective_rh_gl` is deprecated in
/// glam 0.33 and its replacement moved to a module that is not re-exported here. Only x/y and
/// w matter for the panorama — the result is divided through by w and the z is discarded — so
/// this is exactly `Project.gluPerspective(fovy, aspect, znear, zfar)`.
// java: org.lwjgl.util.glu.Project.gluPerspective, called from GuiMainMenu.drawPanorama
#[must_use]
fn perspective_gl(fov_y: f32, aspect: f32, near: f32, far: f32) -> glam::Mat4 {
    let f = 1.0 / (fov_y / 2.0).tan();
    glam::Mat4::from_cols_array_2d(&[
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), -1.0],
        [0.0, 0.0, 2.0 * far * near / (near - far), 0.0],
    ])
}

/// Map square-target NDC through `renderSkybox`'s transposed centre crop.
#[must_use]
fn panorama_crop_point(width: i32, height: i32, ndc_x: f32, ndc_y: f32) -> [f32; 2] {
    // From Java's f = 120/max(width,height), f1 = height*f/256 and
    // f2 = width*f/256. Solving its four UV corners for screen x/y leaves a common
    // max(width,height)/(4*120/256) = max/1.875 scale.
    let crop_scale = width.max(height).max(1) as f32 / 1.875;
    [
        width as f32 * 0.5 - ndc_y * crop_scale,
        height as f32 * 0.5 - ndc_x * crop_scale,
    ]
}

/// The rotating panorama behind the main menu.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Panorama {
    /// Ticks since the screen opened. java: `panoramaTimer`
    pub timer: i32,
}

impl Panorama {
    /// Advance one client tick.
    pub fn tick(&mut self) {
        self.timer = self.timer.wrapping_add(1);
    }

    /// Pitch and yaw in degrees at `partial_ticks` into the current tick.
    ///
    /// Uses `mc_math::sin` — the 65536-entry LUT — because vanilla does (`MathHelper.sin`), and
    /// hazard H1 says not to substitute libm even where it "would not matter".
    // java: GuiMainMenu.java:402
    #[must_use]
    pub fn angles(&self, partial_ticks: f32) -> (f32, f32) {
        let t = self.timer as f32 + partial_ticks;
        (mc_math::sin(t / 400.0) * 25.0 + 20.0, -t * 0.1)
    }

    /// The camera matrix: vanilla's fixed `rotate(180, X)`, `rotate(90, Z)` orientation
    /// followed by the animated pitch and yaw.
    #[must_use]
    pub fn view(&self, partial_ticks: f32) -> glam::Mat4 {
        let (pitch, yaw) = self.angles(partial_ticks);
        glam::Mat4::from_rotation_x(180f32.to_radians())
            * glam::Mat4::from_rotation_z(90f32.to_radians())
            * glam::Mat4::from_rotation_x(pitch.to_radians())
            * glam::Mat4::from_rotation_y(yaw.to_radians())
    }

    /// The per-face model rotation. java: the `if (k == n)` ladder in `drawPanorama`.
    #[must_use]
    pub fn face_rotation(face: usize) -> glam::Mat4 {
        match face {
            1 => glam::Mat4::from_rotation_y(90f32.to_radians()),
            2 => glam::Mat4::from_rotation_y(180f32.to_radians()),
            3 => glam::Mat4::from_rotation_y((-90f32).to_radians()),
            4 => glam::Mat4::from_rotation_x(90f32.to_radians()),
            5 => glam::Mat4::from_rotation_x((-90f32).to_radians()),
            _ => glam::Mat4::IDENTITY,
        }
    }

    /// Emit vanilla's 64-sample panorama accumulation into a draw list.
    ///
    /// `steps` is the per-face subdivision count needed because this CPU projection cannot leave
    /// perspective-correct interpolation and clipping to the GPU. It does **not** replace
    /// vanilla's outer 8×8 jitter loop: every call emits all 64 samples, in sample/face order,
    /// with Java's integer `255 / (sample + 1)` alpha schedule.
    ///
    /// The camera sits **inside** the cube, so every face has an edge that passes through it and
    /// sub-quads straddling that edge would project to infinity. Each sub-quad is therefore
    /// clipped against the near plane in homogeneous space before the perspective divide —
    /// which is the step a GPU does for free and a CPU-side projection has to do by hand. The
    /// result is fanned into triangles and submitted as degenerate quads, because
    /// [`DrawList::quad`] is the only primitive S15 exposes.
    pub fn draw(&self, d: &mut DrawList, width: i32, height: i32, partial_ticks: f32, steps: u32) {
        self.draw_mapped(d, partial_ticks, steps, |c| {
            panorama_crop_point(width, height, c.x / c.w, c.y / c.w)
        });
    }

    /// Emit the first panorama stage in the coordinate space of vanilla's private 256×256
    /// target.
    ///
    /// Unlike [`Self::draw`], this does not algebraically skip the intermediate target. It is the
    /// exact seed a renderer needs before running the seven `rotateAndBlurSkybox` passes described
    /// by [`PanoramaRenderPlan`]. Coordinates use the GUI convention (origin at top-left), so NDC
    /// `y = +1` maps to target `y = 0`.
    pub fn draw_to_target(&self, d: &mut DrawList, partial_ticks: f32, steps: u32) {
        let half = PANORAMA_TARGET_SIZE as f32 * 0.5;
        self.draw_mapped(d, partial_ticks, steps, |c| {
            let ndc_x = c.x / c.w;
            let ndc_y = c.y / c.w;
            [(ndc_x + 1.0) * half, (1.0 - ndc_y) * half]
        });
    }

    fn draw_mapped(
        &self,
        d: &mut DrawList,
        partial_ticks: f32,
        steps: u32,
        to_screen: impl Fn(glam::Vec4) -> [f32; 2],
    ) {
        // java: GuiMainMenu.drawPanorama disables culling while drawing the inside of the cube.
        // The direct crop transposes the projection and reverses its winding relative to the
        // private target; both paths must retain Java's two-sided state.
        d.with_render_state(UiDepth::Overlay, UiCull::None, |d| {
            self.draw_mapped_geometry(d, partial_ticks, steps, to_screen);
        });
    }

    fn draw_mapped_geometry(
        &self,
        d: &mut DrawList,
        partial_ticks: f32,
        steps: u32,
        to_screen: impl Fn(glam::Vec4) -> [f32; 2],
    ) {
        const NEAR: f32 = 0.05;
        let steps = steps.max(1);
        // java: drawPanorama always renders a square 256×256 target with aspect 1.0.
        let proj = perspective_gl(120f32.to_radians(), 1.0, NEAR, 10.0);
        let (pitch, yaw) = self.angles(partial_ticks);
        let fixed_view = glam::Mat4::from_rotation_x(180f32.to_radians())
            * glam::Mat4::from_rotation_z(90f32.to_radians());

        for sample in 0..PANORAMA_ACCUMULATION_SAMPLES {
            let sample = PanoramaSample::vanilla(sample);
            let view = fixed_view
                * glam::Mat4::from_translation(glam::Vec3::new(
                    sample.translation[0],
                    sample.translation[1],
                    0.0,
                ))
                * glam::Mat4::from_rotation_x(pitch.to_radians())
                * glam::Mat4::from_rotation_y(yaw.to_radians());
            let alpha = sample.alpha as f32 / 255.0;
            let color = [[1.0, 1.0, 1.0, alpha]; 4];

            for (face, path) in PANORAMA_FACES.iter().enumerate() {
                let m = proj * view * Self::face_rotation(face);
                let tex = UiTexture::gui(path);
                for row in 0..steps {
                    for col in 0..steps {
                        let u0 = col as f32 / steps as f32;
                        let u1 = (col + 1) as f32 / steps as f32;
                        let v0 = row as f32 / steps as f32;
                        let v1 = (row + 1) as f32 / steps as f32;
                        // The face quad spans (-1, -1, 1) .. (1, 1, 1) with (0, 0) at the
                        // bottom-left, matching `worldrenderer.pos(-1, -1, 1).tex(0, 0)`.
                        let corner = |u: f32, v: f32| {
                            (
                                m * glam::Vec4::new(u * 2.0 - 1.0, v * 2.0 - 1.0, 1.0, 1.0),
                                [u, v],
                            )
                        };
                        // Winding order: DrawList::quad wants top-left, bottom-left, top-right,
                        // bottom-right, which for this face is (u0,v1), (u0,v0), (u1,v1),
                        // (u1,v0).
                        let poly = [
                            corner(u0, v1),
                            corner(u0, v0),
                            corner(u1, v1),
                            corner(u1, v0),
                        ];

                        if poly.iter().all(|(c, _)| c.w > NEAR) {
                            let s = poly.map(|(c, _)| to_screen(c));
                            d.quad(
                                &tex,
                                [s[0], s[1], s[2], s[3]],
                                [poly[0].1, poly[1].1, poly[2].1, poly[3].1],
                                color,
                            );
                            continue;
                        }
                        // Sutherland–Hodgman against `w >= NEAR`, in the fan order 0,1,3,2 so
                        // the polygon is convex.
                        let ring = [poly[0], poly[1], poly[3], poly[2]];
                        let mut clipped: Vec<(glam::Vec4, [f32; 2])> = Vec::with_capacity(5);
                        for i in 0..4 {
                            let (a, au) = ring[i];
                            let (b, bu) = ring[(i + 1) % 4];
                            let (ain, bin) = (a.w > NEAR, b.w > NEAR);
                            if ain {
                                clipped.push((a, au));
                            }
                            if ain != bin {
                                let t = (NEAR - a.w) / (b.w - a.w);
                                clipped.push((
                                    a + (b - a) * t,
                                    [au[0] + (bu[0] - au[0]) * t, au[1] + (bu[1] - au[1]) * t],
                                ));
                            }
                        }
                        if clipped.len() < 3 {
                            continue;
                        }
                        let first = to_screen(clipped[0].0);
                        for i in 1..clipped.len() - 1 {
                            let b = to_screen(clipped[i].0);
                            let c = to_screen(clipped[i + 1].0);
                            // A degenerate quad: `quad` emits indices 0,1,2 then 1,3,2, and with
                            // vertex 3 == vertex 2 the second triangle has zero area.
                            d.quad(
                                &tex,
                                [first, b, c, c],
                                [
                                    clipped[0].1,
                                    clipped[i].1,
                                    clipped[i + 1].1,
                                    clipped[i + 1].1,
                                ],
                                color,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The main menu.
#[derive(Debug)]
pub struct MainMenuScreen {
    /// The rotating background.
    pub panorama: Panorama,
    /// The splash message, already picked.
    pub splash: String,
    /// The version line in the bottom-left.
    pub version_line: String,
    /// The splash's pulsing scale factor as of the last frame. See the note in
    /// [`MainMenuScreen::draw`].
    pub splash_scale: f32,
    /// The constructor-time random draw that selects vanilla's 1-in-10,000 `minceraft`
    /// wordmark. Despite its Java name, this value never advances.
    // java: GuiMainMenu.java:145, 546
    pub update_counter: f32,
    buttons: Vec<Button>,
}

impl Default for MainMenuScreen {
    fn default() -> Self {
        Self::new("missingno")
    }
}

impl MainMenuScreen {
    /// A menu showing `splash`.
    #[must_use]
    pub fn new(splash: impl Into<String>) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as i64);
        let update_counter = mc_math::JavaRandom::new(seed).next_f32();
        Self::with_update_counter(splash, update_counter)
    }

    /// A menu with an explicit constructor-time title draw.
    ///
    /// This keeps the rare title branch deterministic in tests. Normal callers should use
    /// [`Self::new`].
    #[must_use]
    pub fn with_update_counter(splash: impl Into<String>, update_counter: f32) -> Self {
        Self {
            panorama: Panorama::default(),
            splash: splash.into(),
            version_line: "Minecraft 1.8.9".to_owned(),
            splash_scale: 1.0,
            update_counter,
            buttons: Vec::new(),
        }
    }

    /// Lay out the buttons.
    ///
    /// **The multiplayer-only rule deviation, deliberate and visible:** the Singleplayer and Realms buttons
    /// keep vanilla's exact positions but are permanently disabled, because this client has no
    /// integrated server and Realms is out of scope (§2). Removing them would shift Multiplayer
    /// up 24 pixels and every screenshot comparison with it.
    // java: GuiMainMenu.java:194 (initGui)
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (month, day) = current_local_month_day();
        self.init_on_date(ctx, lang, month, day);
    }

    /// Lay out the menu and apply vanilla's date-specific splash override.
    ///
    /// Keeping the date explicit makes the otherwise dead seasonal branch testable and lets a
    /// platform shell supply local civil time. Like `GuiMainMenu.initGui`, this is intentionally
    /// called again on every resize, so a matching date re-asserts the override.
    // java: GuiMainMenu.java:194-225 (initGui)
    pub fn init_on_date(&mut self, ctx: &ScreenCtx, lang: &Language, month: u32, day: u32) {
        if let Some(splash) = Splashes::calendar_override(month, day) {
            self.splash = splash.to_owned();
        }
        let (w, h) = (ctx.width, ctx.height);
        let j = h / 4 + 48;
        let tr = |k: &str| lang.tr(k).to_owned();
        let mut single = Button::new(ids::SINGLEPLAYER, w / 2 - 100, j, tr("menu.singleplayer"));
        single.enabled = false;
        let mut realms = Button::new(ids::REALMS, w / 2 - 100, j + 48, tr("menu.online"));
        realms.enabled = false;
        self.buttons = vec![
            single,
            Button::new(
                ids::MULTIPLAYER,
                w / 2 - 100,
                j + 24,
                tr("menu.multiplayer"),
            ),
            realms,
            Button::sized(
                ids::OPTIONS,
                w / 2 - 100,
                j + 84,
                98,
                20,
                tr("menu.options"),
            ),
            Button::sized(ids::QUIT, w / 2 + 2, j + 84, 98, 20, tr("menu.quit")),
            // java: GuiButtonLanguage is a bare 20×20 icon button.
            Button::sized(ids::LANGUAGE, w / 2 - 124, j + 84, 20, 20, ""),
        ];
    }

    /// Advance the panorama.
    pub fn tick(&mut self) {
        self.panorama.tick();
    }

    /// Handle a click.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> Action {
        if button != 0 {
            return Action::None;
        }
        for b in &self.buttons {
            if b.mouse_pressed(x, y) {
                return match b.id {
                    ids::MULTIPLAYER => Action::OpenMultiplayer,
                    ids::OPTIONS => Action::OpenOptions,
                    ids::LANGUAGE => {
                        Action::OpenOptionsPage(crate::screens::options::OptionsPage::Language)
                    }
                    ids::QUIT => Action::Quit,
                    _ => Action::None,
                };
            }
        }
        Action::None
    }

    /// Emit the panorama's sharp first stage into vanilla's private 256×256 coordinate space
    /// and return the renderer contract for this window.
    ///
    /// The returned list contains only panorama-face batches. Submit it through
    /// `UiPhase::submit_with_panorama`, then draw [`Self::draw_foreground`] as that submission's
    /// ordinary foreground list.
    pub fn draw_panorama_seed(
        &self,
        d: &mut DrawList,
        ctx: &ScreenCtx,
        panorama_steps: u32,
    ) -> PanoramaRenderPlan {
        self.panorama
            .draw_to_target(d, ctx.partial_ticks, panorama_steps);
        PanoramaRenderPlan::for_window(ctx.width, ctx.height)
    }

    /// Draw everything vanilla places after `renderSkybox`: both full-screen gradients, the
    /// normal or rare `minceraft` title, seasonal/current splash, version, copyright, and buttons.
    ///
    /// No dirt or panorama-face batch is emitted. This is the foreground half expected by
    /// `UiPhase::submit_with_panorama` after it composites the blurred private target.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_foreground(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        system_time_ms: u64,
    ) {
        self.draw_contents(d, font, ctx, mouse, system_time_ms, true);
    }

    /// Draw the menu.
    ///
    /// `panorama_steps` of 0 skips the panorama entirely and falls back to the dirt background,
    /// which is what a headless test wants when it does not have the six 256×256 textures.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        system_time_ms: u64,
        panorama_steps: u32,
    ) {
        let (w, h) = (ctx.width, ctx.height);
        if panorama_steps == 0 {
            draw_dirt_background(d, w, h, 0.0);
        } else {
            self.panorama
                .draw(d, w, h, ctx.partial_ticks, panorama_steps);
        }
        self.draw_contents(d, font, ctx, mouse, system_time_ms, panorama_steps != 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_contents(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        system_time_ms: u64,
        skybox_gradients: bool,
    ) {
        let (w, h) = (ctx.width, ctx.height);
        if skybox_gradients {
            // java: two full-screen gradients over the skybox, -2130706433 → 16777215 and
            // 0 → Integer.MIN_VALUE. They are foreground, not part of the blurred target.
            d.gradient_rect(0, 0, w, h, 0x80FF_FFFF, 0x00FF_FFFF);
            d.gradient_rect(0, 0, w, h, 0x0000_0000, 0x8000_0000);
        }

        // java: i = 274; j = width / 2 - i / 2; k = 30. One constructor-time draw in 10,000
        // rearranges the first half into the `minceraft` easter egg.
        let j = w / 2 - 137;
        let title = UiTexture::gui(TITLE_TEXTURE);
        if f64::from(self.update_counter) < 1.0e-4 {
            d.blit(&title, j, 30, 0, 0, 99, 44);
            d.blit(&title, j + 99, 30, 129, 0, 27, 44);
            d.blit(&title, j + 125, 30, 126, 0, 3, 44);
            d.blit(&title, j + 128, 30, 99, 0, 26, 44);
            d.blit(&title, j + 155, 30, 0, 45, 155, 44);
        } else {
            d.blit(&title, j, 30, 0, 0, 155, 44);
            d.blit(&title, j + 155, 30, 0, 45, 155, 44);
        }

        // Vanilla draws the splash under `translate(width/2 + 90, 70)`, `rotate(-20°)` and
        // `scale(f)`, then draws it at local `(0, -8)` in opaque yellow.
        self.splash_scale = splash_scale(system_time_ms, font.string_width(&self.splash));
        d.push_transform(
            Transform::translate((w / 2 + 90) as f32, 70.0)
                .then_rotate_degrees(-20.0)
                .then_scale(self.splash_scale),
        );
        draw_centered_string(d, font, &self.splash, 0, -8, -256);
        d.pop_transform();

        draw_string(d, font, &self.version_line, 2, h - 10, -1);
        const COPYRIGHT: &str = "Copyright Mojang AB. Do not distribute!";
        draw_string(
            d,
            font,
            COPYRIGHT,
            w - font.string_width(COPYRIGHT) - 2,
            h - 10,
            -1,
        );

        for b in &mut self.buttons {
            if b.id == ids::LANGUAGE {
                draw_language_button(d, b, mouse);
            } else {
                b.draw(d, font, mouse.0, mouse.1);
            }
        }
    }
}

/// `GuiButtonLanguage.drawButton`: one 20-pixel-high globe plate, with the texture row shifted by
/// the button's own height while hovered. It deliberately bypasses generic button state art and
/// draws no label.
// java: GuiButtonLanguage.java:14-27
fn draw_language_button(d: &mut DrawList, button: &Button, mouse: (i32, i32)) {
    if !button.visible {
        return;
    }
    let hovered = button.rect.contains(mouse.0, mouse.1);
    let texture_y = 106 + if hovered { button.rect.height } else { 0 };
    d.blit(
        &UiTexture::gui(WIDGETS_TEXTURE),
        button.rect.x,
        button.rect.y,
        0,
        texture_y,
        button.rect.width,
        button.rect.height,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::{ScaledResolution, OPTIONS_BACKGROUND};

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    /// Golden values taken from the Java specification of `String.hashCode`, not from this
    /// implementation: `"" = 0`, `"a" = 97`, `"ab" = 3105`, `"Hello" = 69609650`. The last two
    /// are the standard published examples.
    #[test]
    fn the_java_string_hash_matches_the_published_values() {
        assert_eq!(java_hash_code(""), 0);
        assert_eq!(java_hash_code("a"), 97);
        assert_eq!(java_hash_code("ab"), 3105);
        assert_eq!(java_hash_code("Hello"), 69_609_650);
        assert_eq!(java_hash_code("hello world"), 1_794_106_052);
        // Overflow must wrap, not panic or saturate.
        let long = "x".repeat(64);
        let _ = java_hash_code(&long);
    }

    #[test]
    fn splash_parsing_trims_and_drops_blank_lines() {
        let s = Splashes::parse("  one  \n\n\ttwo\t\n   \nthree\n");
        assert_eq!(s.len(), 3);
        assert_eq!(s.lines, ["one", "two", "three"]);
        assert!(Splashes::parse("\n \n").is_empty());
    }

    #[test]
    fn an_empty_pool_gives_vanillas_missingno() {
        let s = Splashes::default();
        assert_eq!(s.pick(&mut std::iter::repeat(0)), "missingno");
    }

    #[test]
    fn the_banned_splash_is_redrawn_past() {
        // Construct a pool whose first entry hashes to the banned value.
        let banned = banned_string();
        assert_eq!(java_hash_code(&banned), BANNED_SPLASH_HASH);
        let s = Splashes::parse(&format!("{banned}\nfine"));
        let mut draws = [0usize, 0, 1].into_iter();
        assert_eq!(s.pick(&mut draws), "fine");
    }

    #[test]
    fn banned_splash_redraws_do_not_have_a_non_vanilla_attempt_cap() {
        let banned = banned_string();
        let s = Splashes::parse(&format!("{banned}\nfine"));
        let mut draws = std::iter::repeat_n(0usize, 65).chain(std::iter::once(1));
        assert_eq!(s.pick(&mut draws), "fine");
    }

    /// A string whose Java hash is exactly [`BANNED_SPLASH_HASH`], built by solving the
    /// polynomial backwards rather than by asking this crate's own hash function for candidates.
    ///
    /// `hash("PPPP" + abc) = hash("PPPP") * 31^3 + a*961 + b*31 + c`, so pick a four-character
    /// prefix that puts the residue inside the range three printable ASCII characters can
    /// express, then read the three characters straight out of it.
    fn banned_string() -> String {
        // For a four-character string, `hash = a*31^3 + b*31^2 + c*31 + d`. Pin b, c and d to
        // offset base-31 digits in `32..=62` — ordinary printable ASCII — which lets the tail
        // take every value in `[32*993, 62*993]`, then read `a` straight off. Exactly one `a`
        // puts the tail in that window.
        const W: i64 = 29_791; // 31^3
        const SUM: i64 = 961 + 31 + 1;
        let target = i64::from(BANNED_SPLASH_HASH);
        for a in 1..=0xFFFFi64 {
            let tail = target - a * W;
            if tail < 32 * SUM {
                break;
            }
            if tail > 62 * SUM {
                continue;
            }
            let mut rem = tail - 32 * SUM;
            let mut s = String::new();
            s.push(char::from_u32(a as u32).expect("a is below the surrogate range"));
            for w in [961i64, 31, 1] {
                let digit = rem / w;
                rem -= digit * w;
                s.push(char::from_u32(32 + digit as u32).expect("printable ASCII"));
            }
            return s;
        }
        unreachable!("exactly one four-character preimage exists by construction")
    }

    #[test]
    fn the_three_calendar_overrides_are_vanillas() {
        assert_eq!(Splashes::calendar_override(12, 24), Some("Merry X-mas!"));
        assert_eq!(Splashes::calendar_override(1, 1), Some("Happy new year!"));
        assert_eq!(
            Splashes::calendar_override(10, 31),
            Some("OOoooOOOoooo! Spooky!")
        );
        assert_eq!(Splashes::calendar_override(12, 25), None);
        assert_eq!(Splashes::calendar_override(7, 4), None);
    }

    #[test]
    fn unix_day_conversion_covers_each_override_and_leap_day() {
        // Independently calculated Unix day numbers. These pin both sides of leap day so a
        // month-only or non-leap civil conversion cannot accidentally make the integration test
        // pass.
        assert_eq!(month_day_from_unix_days(0), (1, 1)); // 1970-01-01
        assert_eq!(month_day_from_unix_days(19_722), (12, 31)); // 2023-12-31
        assert_eq!(month_day_from_unix_days(19_782), (2, 29)); // 2024-02-29
        assert_eq!(month_day_from_unix_days(20_085), (12, 28)); // 2024-12-28
    }

    #[test]
    fn local_offset_seam_crosses_utc_midnight_and_keeps_ordinary_dates_ordinary() {
        // 2023-12-24 00:30 UTC is still 2023-12-23 in UTC-05:00. This is the boundary that a
        // UTC-only implementation gets wrong for the Christmas override.
        let christmas_boundary = 19_715 * 86_400 + 30 * 60;
        assert_eq!(
            month_day_from_unix_seconds_with_offset(christmas_boundary, 0),
            (12, 24)
        );
        let local_date = month_day_from_unix_seconds_with_offset(christmas_boundary, -5 * 3_600);
        assert_eq!(local_date, (12, 23));
        assert_eq!(
            Splashes::calendar_override(local_date.0, local_date.1),
            None
        );

        // The opposite direction pins the year boundary: 2023-12-31 23:30 UTC is already
        // New Year's Day in UTC+01:00.
        let new_year_boundary = 19_722 * 86_400 + 23 * 3_600 + 30 * 60;
        assert_eq!(
            month_day_from_unix_seconds_with_offset(new_year_boundary, 0),
            (12, 31)
        );
        let local_date = month_day_from_unix_seconds_with_offset(new_year_boundary, 3_600);
        assert_eq!(local_date, (1, 1));
        assert_eq!(
            Splashes::calendar_override(local_date.0, local_date.1),
            Some("Happy new year!")
        );
    }

    #[test]
    fn injected_clock_controls_the_utc_fallback_without_process_spawning() {
        let christmas_boundary = 19_715 * 86_400 + 30 * 60;
        let christmas =
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(christmas_boundary as u64);
        assert_eq!(
            month_day_from_system_time(christmas, 0),
            (12, 24),
            "the in-process UTC fallback must preserve the injected instant"
        );

        // A caller that has a safe host-local offset can inject it without relying on a shell
        // command. Just after UTC midnight, UTC-05:00 is still the previous calendar day.
        assert_eq!(month_day_from_system_time(christmas, -5 * 3_600), (12, 23));

        let new_year_boundary = 19_722 * 86_400 + 23 * 3_600 + 30 * 60;
        let new_year =
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(new_year_boundary as u64);
        assert_eq!(month_day_from_system_time(new_year, 0), (12, 31));
        assert_eq!(month_day_from_system_time(new_year, 3_600), (1, 1));
    }

    #[test]
    fn production_date_reader_uses_the_host_local_clock() {
        let source = include_str!("menu.rs");
        let (_, reader) = source
            .split_once("fn current_local_month_day()")
            .expect("production date reader remains present");
        let (reader, _) = reader
            .split_once("/// Convert an injected instant")
            .expect("date arithmetic helper remains below the production reader");
        assert!(reader.contains("Local::now()"));
        assert!(
            !reader.contains("month_day_from_system_time") && !reader.contains("UNIX_EPOCH"),
            "production must use the host timezone provider rather than the old UTC fallback"
        );

        // Also exercise the live wiring against the host clock. Permit the date to advance
        // between the surrounding samples so a midnight boundary cannot make this flaky.
        let before = Local::now();
        let actual = current_local_month_day();
        let after = Local::now();
        let before_date = (before.month(), before.day());
        let after_date = (after.month(), after.day());
        assert!(
            actual == before_date || actual == after_date,
            "production date {actual:?} must come from the host-local clock ({before_date:?}..={after_date:?})"
        );
    }

    #[test]
    fn init_calls_the_calendar_override_and_resize_reasserts_it() {
        let mut menu = MainMenuScreen::with_update_counter("ordinary random splash", 1.0);
        let language = Language::new("en_US");
        menu.init_on_date(&ctx(), &language, 12, 24);
        assert_eq!(menu.splash, "Merry X-mas!");

        // A second init models the driver's resize path. This deliberately changes the field in
        // between: a test that only calls calendar_override directly cannot detect the dead-wire
        // defect that prompted F-P2-client.gui-09-2.
        menu.splash = "mutated after first init".to_owned();
        menu.init_on_date(&ctx(), &language, 10, 31);
        assert_eq!(menu.splash, "OOoooOOOoooo! Spooky!");

        menu.splash = "ordinary random splash".to_owned();
        menu.init_on_date(&ctx(), &language, 7, 4);
        assert_eq!(menu.splash, "ordinary random splash");
    }

    /// The scale is `1.8 - |sin(phase) * 0.1|`, so it never exceeds 1.8 and never drops below
    /// 1.7 — a port that read the precedence as `|1.8 - sin(phase)| * 0.1` or that took the
    /// absolute value before the multiply would fail this.
    #[test]
    fn the_splash_pulse_stays_between_one_point_seven_and_one_point_eight() {
        for ms in (0..1000).step_by(7) {
            let raw = splash_scale(ms, 68); // width + 32 == 100, so the factor passes through
            assert!((1.699..=1.801).contains(&raw), "ms={ms} gave {raw}");
        }
        // At phase 0 the sine is 0, so the factor is exactly 1.8.
        assert!((splash_scale(0, 68) - 1.8).abs() < 1e-5);
        // A quarter period in, |sin| is 1 and the factor bottoms out at 1.7.
        assert!((splash_scale(250, 68) - 1.7).abs() < 1e-3);
        // A wider splash shrinks proportionally: 100 / (width + 32).
        assert!((splash_scale(0, 168) - 1.8 * 100.0 / 200.0).abs() < 1e-5);
    }

    /// Hand-derived from `MathHelper.sin(t / 400) * 25 + 20` and `-t * 0.1`.
    #[test]
    fn the_panorama_rotation_follows_vanillas_schedule() {
        let p = Panorama::default();
        let (pitch, yaw) = p.angles(0.0);
        assert!((pitch - 20.0).abs() < 1e-3, "sin(0) = 0, so pitch = 20");
        assert!((yaw - 0.0).abs() < 1e-6);

        let p = Panorama { timer: 400 };
        let (pitch, yaw) = p.angles(0.0);
        // sin(1 rad) = 0.841471, so 20 + 21.0368 = 41.0368.
        assert!((pitch - 41.036_8).abs() < 0.02, "got {pitch}");
        assert!((yaw + 40.0).abs() < 1e-4);

        // Partial ticks advance it smoothly.
        let p = Panorama { timer: 10 };
        let (_, y0) = p.angles(0.0);
        let (_, y1) = p.angles(0.5);
        assert!((y1 - y0 + 0.05).abs() < 1e-5);
    }

    #[test]
    fn panorama_crop_is_vanillas_transposed_widescreen_mapping() {
        let centre = panorama_crop_point(854, 480, 0.0, 0.0);
        assert_eq!(centre, [427.0, 240.0]);

        // At 854×480 Java composites v=0.03125..0.96875 across screen x. Those texture
        // coordinates are square-target NDC y=-0.9375..0.9375. The transposition is important:
        // changing NDC y moves screen x, not screen y.
        let left = panorama_crop_point(854, 480, 0.0, 0.9375);
        let right = panorama_crop_point(854, 480, 0.0, -0.9375);
        assert!(left[0].abs() < 1.0e-4, "left={left:?}");
        assert!((right[0] - 854.0).abs() < 1.0e-4, "right={right:?}");
        assert_eq!(left[1], 240.0);
        assert_eq!(right[1], 240.0);

        // The corresponding u crop is height/max(width) as wide, reaching both y edges.
        let ndc_x_edge = 480.0 * 0.9375 / 854.0;
        assert!(panorama_crop_point(854, 480, ndc_x_edge, 0.0)[1].abs() < 1.0e-4);
        assert!((panorama_crop_point(854, 480, -ndc_x_edge, 0.0)[1] - 480.0).abs() < 1.0e-4);
    }

    #[test]
    fn panorama_render_plan_exposes_the_fixed_target_seven_blurs_and_literal_crop_uvs() {
        let plan = PanoramaRenderPlan::for_window(854, 480);
        assert_eq!(plan.target_size, [256, 256]);
        assert_eq!(plan.accumulation_samples, 64);
        assert_eq!(plan.blur_passes, 7);

        let expected = [
            [0.763_466, 0.968_75],
            [0.236_534, 0.968_75],
            [0.763_466, 0.031_25],
            [0.236_534, 0.031_25],
        ];
        for (actual, expected) in plan.composite_uv.into_iter().zip(expected) {
            assert!((actual[0] - expected[0]).abs() < 1.0e-5, "u: {actual:?}");
            assert!((actual[1] - expected[1]).abs() < 1.0e-5, "v: {actual:?}");
        }

        // Mutation controls: using the window aspect directly instead of Java's square target,
        // or forgetting that x samples texture-v, changes these independently.
        assert_ne!(plan.composite_uv[0][0], plan.composite_uv[0][1]);
        assert!(plan.composite_uv[0][0] < plan.composite_uv[0][1]);
    }

    #[test]
    fn panorama_blur_plan_exposes_the_rotated_three_tap_pass_not_a_sharp_identity() {
        let plan = PanoramaRenderPlan::blur_plan();
        assert_eq!(plan.target_size, [PANORAMA_TARGET_SIZE; 2]);
        assert_eq!(plan.passes, PANORAMA_BLUR_PASSES);
        assert_eq!(plan.taps.len(), PANORAMA_BLUR_TAPS);

        let offsets: Vec<_> = plan.taps.iter().map(|tap| tap.offset_u).collect();
        let alphas: Vec<_> = plan.taps.iter().map(|tap| tap.blend_alpha).collect();
        assert_eq!(offsets, [-1.0 / 256.0, 0.0, 1.0 / 256.0]);
        assert_eq!(alphas, [1.0, 1.0 / 2.0, 1.0 / 3.0]);

        // Java's blur quad is a 90-degree source rotation in DrawList corner order. An identity
        // source mapping or a sharp one-tap pass must fail these sentinels independently.
        assert_eq!(
            plan.quad.screen_ndc,
            [[-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, -1.0]]
        );
        assert_eq!(
            plan.quad.source_uv,
            [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        );
        assert_ne!(
            plan.quad.source_uv,
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        );
        assert!(plan.taps.iter().any(|tap| tap.offset_u != 0.0));

        // The literal alpha cascade has equal effective weights. Keep this CPU-only check tied to
        // the descriptor so a tempting replacement with one opaque centre sample is observable.
        let source = [0.9_f32, 0.4, 0.65];
        let mut cascade = 0.123_f32;
        for (tap, value) in plan.taps.iter().zip(source) {
            cascade = value * tap.blend_alpha + cascade * (1.0 - tap.blend_alpha);
        }
        let average = source.iter().sum::<f32>() / source.len() as f32;
        assert!((cascade - average).abs() < 1.0e-5);
        assert_ne!(
            cascade, source[1],
            "a sharp centre-only pass is not the blur"
        );
    }

    #[test]
    fn panorama_crop_plan_tracks_landscape_portrait_and_square_framing() {
        let landscape = PanoramaRenderPlan::for_window(854, 480);
        let portrait = PanoramaRenderPlan::for_window(480, 854);
        let square = PanoramaRenderPlan::for_window(512, 512);

        // The crop is aspect-dependent but always derived from the 256-square target. Portrait
        // swaps which axis receives the wide crop; square keeps the two ranges identical.
        let expected_landscape = [
            [0.763_466, 0.968_75],
            [0.236_534, 0.968_75],
            [0.763_466, 0.031_25],
            [0.236_534, 0.031_25],
        ];
        for (actual, expected) in landscape.composite_uv.into_iter().zip(expected_landscape) {
            assert!((actual[0] - expected[0]).abs() < 1.0e-5, "u: {actual:?}");
            assert!((actual[1] - expected[1]).abs() < 1.0e-5, "v: {actual:?}");
        }
        let expected_portrait = [
            [0.968_75, 0.763_466],
            [0.031_25, 0.763_466],
            [0.968_75, 0.236_534],
            [0.031_25, 0.236_534],
        ];
        for (actual, expected) in portrait.composite_uv.into_iter().zip(expected_portrait) {
            assert!((actual[0] - expected[0]).abs() < 1.0e-5, "u: {actual:?}");
            assert!((actual[1] - expected[1]).abs() < 1.0e-5, "v: {actual:?}");
        }
        assert_eq!(
            square.composite_uv,
            [
                [0.968_75, 0.968_75],
                [0.031_25, 0.968_75],
                [0.968_75, 0.031_25],
                [0.031_25, 0.031_25],
            ]
        );

        // Negative controls: using a direct window projection or forgetting the transposition
        // cannot produce all three aspect-specific UV contracts.
        assert_ne!(landscape.composite_uv, portrait.composite_uv);
        assert_ne!(landscape.composite_uv, square.composite_uv);
        assert_ne!(portrait.composite_uv, square.composite_uv);
    }

    #[test]
    fn panorama_sample_schedule_is_the_literal_java_integer_loop() {
        let first = PanoramaSample::vanilla(0);
        assert_eq!(first.translation, [-0.007_812_5, -0.007_812_5]);
        assert_eq!(first.alpha, 255);

        let second = PanoramaSample::vanilla(1);
        assert_eq!(second.translation, [-0.005_859_375, -0.007_812_5]);
        assert_eq!(second.alpha, 127);

        let ninth = PanoramaSample::vanilla(8);
        assert_eq!(ninth.translation, [-0.007_812_5, -0.005_859_375]);
        assert_eq!(ninth.alpha, 28);

        let last = PanoramaSample::vanilla(63);
        assert_eq!(last.translation, [0.005_859_375, 0.005_859_375]);
        assert_eq!(last.alpha, 3);

        // A centred 8×8 kernel or floating-point 255/n both look plausible but are not what
        // Java executes. These asymmetry/truncation assertions bite both mutations.
        assert_ne!(first.translation[0], -last.translation[0]);
        assert_ne!(second.alpha, 128);
    }

    #[test]
    fn panorama_emits_vanillas_sixty_four_integer_alpha_samples() {
        let p = Panorama { timer: 17 };
        let mut d = DrawList::new();
        p.draw(&mut d, 320, 240, 0.25, 1);

        assert!(d.batches().iter().all(|batch| batch.cull == UiCull::None));
        assert!(d
            .batches()
            .iter()
            .all(|batch| batch.depth == UiDepth::Overlay));

        let mut actual: Vec<u8> = d
            .vertices()
            .iter()
            .map(|vertex| (vertex.color[3] * 255.0).round() as u8)
            .collect();
        actual.dedup();
        let mut expected: Vec<u8> = (1..=64).map(|n| (255 / n) as u8).collect();
        expected.dedup();
        assert_eq!(actual, expected);

        // Negative controls: a single opaque draw has one alpha, and floating 255/n would round
        // sample two to 128 instead of Java's integer-truncated 127.
        assert!(actual.len() > 1);
        assert_eq!(actual[1], 127);
        d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        assert_eq!(d.batches().last().unwrap().cull, UiCull::Back);
    }

    #[test]
    fn every_panorama_face_is_drawn_and_the_geometry_covers_the_screen() {
        // At any one instant at least one face is entirely behind the camera and is culled
        // (vanilla relies on GL to clip it instead), so the six only all appear over time.
        let mut faces = std::collections::BTreeSet::new();
        for timer in [0, 300, 600, 900, 1200, 1800] {
            let p = Panorama { timer };
            let mut d = DrawList::new();
            p.draw(&mut d, 320, 240, 0.0, 4);
            for b in d.batches() {
                if let UiTexture::Resource(r) = &b.texture {
                    faces.insert(r.path.clone());
                }
            }
        }
        assert_eq!(faces.len(), 6, "all six cube faces: {faces:?}");

        let p = Panorama::default();
        let mut d = DrawList::new();
        p.draw(&mut d, 320, 240, 0.0, 4);
        // The cube surrounds the camera, so some geometry must reach past every screen edge.
        let xs: Vec<f32> = d.vertices().iter().map(|v| v.pos[0]).collect();
        let ys: Vec<f32> = d.vertices().iter().map(|v| v.pos[1]).collect();
        assert!(xs.iter().cloned().fold(f32::MAX, f32::min) <= 0.0);
        assert!(xs.iter().cloned().fold(f32::MIN, f32::max) >= 320.0);
        assert!(ys.iter().cloned().fold(f32::MAX, f32::min) <= 0.0);
        assert!(ys.iter().cloned().fold(f32::MIN, f32::max) >= 240.0);
    }

    #[test]
    fn sub_quads_behind_the_camera_are_dropped_not_smeared() {
        let p = Panorama { timer: 137 };
        let mut d = DrawList::new();
        p.draw(&mut d, 320, 240, 0.0, 8);
        // A vertex that projected through a near-zero w would land absurdly far off screen.
        for v in d.vertices() {
            assert!(
                v.pos[0].abs() < 1.0e5 && v.pos[1].abs() < 1.0e5,
                "runaway vertex {:?}",
                v.pos
            );
        }
        assert!(!d.is_empty());
    }

    #[test]
    fn target_draw_preserves_the_full_accumulation_for_the_renderer_seam() {
        let p = Panorama { timer: 137 };
        let mut target = DrawList::new();
        let mut direct = DrawList::new();
        p.draw_to_target(&mut target, 0.25, 4);
        p.draw(&mut direct, 854, 480, 0.25, 4);

        assert!(!target.is_empty());
        assert_eq!(target.vertices().len(), direct.vertices().len());
        let xs = target.vertices().iter().map(|v| v.pos[0]);
        let ys = target.vertices().iter().map(|v| v.pos[1]);
        assert!(xs.clone().fold(f32::MAX, f32::min) <= 0.0);
        assert!(xs.fold(f32::MIN, f32::max) >= PANORAMA_TARGET_SIZE as f32);
        assert!(ys.clone().fold(f32::MAX, f32::min) <= 0.0);
        assert!(ys.fold(f32::MIN, f32::max) >= PANORAMA_TARGET_SIZE as f32);

        // The two paths differ only in where the projected vertices land. Any accidental second
        // implementation of the 64-pass schedule would show up in colors, UVs, or batching.
        for (a, b) in target.vertices().iter().zip(direct.vertices()) {
            assert_eq!(a.color, b.color);
            assert_eq!(a.uv, b.uv);
        }
        assert_eq!(target.batches().len(), direct.batches().len());
    }

    #[test]
    fn main_menu_seed_api_emits_only_the_private_target_and_returns_the_literal_plan() {
        let mut menu = MainMenuScreen::with_update_counter("holiday", 1.0);
        menu.panorama.timer = 137;
        let c = ctx();
        let mut seed = DrawList::new();
        let plan = menu.draw_panorama_seed(&mut seed, &c, 4);

        let mut raw = DrawList::new();
        menu.panorama.draw_to_target(&mut raw, c.partial_ticks, 4);
        assert_eq!(
            seed, raw,
            "the screen wrapper must not add foreground batches"
        );
        assert_eq!(plan, PanoramaRenderPlan::for_window(c.width, c.height));
        assert_eq!(plan.target_size, [256, 256]);
        assert_eq!(plan.blur_passes, 7);
        assert!(!seed.is_empty());

        for batch in seed.batches() {
            let UiTexture::Resource(location) = &batch.texture else {
                panic!("seed contained non-resource batch: {:?}", batch.texture);
            };
            assert!(
                PANORAMA_FACES.contains(&location.path.as_str()),
                "seed leaked non-panorama texture {}",
                location.path
            );
        }
    }

    #[test]
    fn foreground_keeps_gradients_holiday_splash_title_buttons_and_footer_but_no_background() {
        let mut menu = MainMenuScreen::with_update_counter("ordinary", 1.0);
        let c = ctx();
        let language = Language::new("en_US");
        menu.init_on_date(&c, &language, 12, 24);
        assert_eq!(menu.splash, "Merry X-mas!");

        let mut font = uniform_font();
        let mut foreground = DrawList::new();
        menu.draw_foreground(&mut foreground, &mut font, &c, (0, 0), 250);

        let resources: Vec<&str> = foreground
            .batches()
            .iter()
            .filter_map(|batch| match &batch.texture {
                UiTexture::Resource(location) => Some(location.path.as_str()),
                _ => None,
            })
            .collect();
        assert!(!resources.contains(&OPTIONS_BACKGROUND));
        assert!(
            !resources.iter().any(|path| PANORAMA_FACES.contains(path)),
            "foreground leaked panorama faces: {resources:?}"
        );
        assert!(resources.contains(&TITLE_TEXTURE), "wordmark must remain");
        assert!(resources.contains(&WIDGETS_TEXTURE), "buttons must remain");

        let solid = foreground
            .batches()
            .iter()
            .find(|batch| batch.texture == UiTexture::Solid)
            .expect("two post-skybox gradients");
        assert_eq!(
            solid.range.len(),
            12,
            "exactly two six-vertex fullscreen gradients"
        );

        // The seasonal splash is the only opaque-yellow text on an unhovered menu. Its
        // foreground glyphs prove the holiday override survived the foreground split.
        let yellow = foreground
            .vertices()
            .iter()
            .filter(|vertex| vertex.color == [1.0, 1.0, 0.0, 1.0])
            .count();
        assert!(yellow >= 6, "seasonal splash foreground glyphs are present");

        // Version begins at x=2 and both footer strings sit at h-10. This pins the footer rather
        // than merely observing that some font texture happened to be used by button labels.
        assert!(foreground
            .vertices()
            .iter()
            .any(|vertex| vertex.pos == [2.0, (c.height - 10) as f32]));
        const COPYRIGHT: &str = "Copyright Mojang AB. Do not distribute!";
        let copyright_x = c.width - font.string_width(COPYRIGHT) - 2;
        assert!(foreground
            .vertices()
            .iter()
            .any(|vertex| vertex.pos == [copyright_x as f32, (c.height - 10) as f32]));
    }

    #[test]
    fn the_wordmark_is_two_one_five_five_halves_centred_on_two_seven_four() {
        let mut menu = MainMenuScreen::with_update_counter("hi", 1.0);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        let c = ctx();
        menu.init(&c, &Language::new("en_US"));
        menu.draw(&mut d, &mut font, &c, (0, 0), 0, 0);
        let left = 854 / 2 - 137;
        assert!(d.vertices().iter().any(|v| v.pos == [left as f32, 30.0]));
        assert!(
            d.vertices()
                .iter()
                .any(|v| v.pos == [(left + 310) as f32, 74.0]),
            "the right half ends 310 px along and 44 px down"
        );
    }

    #[test]
    fn the_minceraft_draw_uses_vanillas_five_exact_blits() {
        let mut menu = MainMenuScreen::with_update_counter("hi", 0.0);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        let c = ctx();
        menu.init(&c, &Language::new("en_US"));
        menu.draw(&mut d, &mut font, &c, (0, 0), 0, 0);

        let title = UiTexture::gui(TITLE_TEXTURE);
        let batch = d
            .batches()
            .iter()
            .find(|batch| batch.texture == title)
            .expect("title batch");
        let vertices = &d.vertices()[batch.range.clone()];
        assert_eq!(vertices.len(), 5 * 6, "five easter-egg blits");

        let left = 854 / 2 - 137;
        let expected = [
            ([left as f32, 30.0], [0.0, 0.0]),
            ([(left + 99) as f32, 30.0], [129.0 / 256.0, 0.0]),
            ([(left + 125) as f32, 30.0], [126.0 / 256.0, 0.0]),
            ([(left + 128) as f32, 30.0], [99.0 / 256.0, 0.0]),
            ([(left + 155) as f32, 30.0], [0.0, 45.0 / 256.0]),
        ];
        for (quad, (pos, uv)) in vertices.chunks_exact(6).zip(expected) {
            assert_eq!(quad[0].pos, pos);
            assert_eq!(quad[0].uv, uv);
        }
    }

    #[test]
    fn splash_vertices_receive_vanillas_translate_rotate_and_pulse_transform() {
        let splash = "AB";
        let c = ctx();
        let draw_frame = |system_time_ms| {
            let mut menu = MainMenuScreen::with_update_counter(splash, 1.0);
            let mut font = uniform_font();
            let mut d = DrawList::new();
            menu.init(&c, &Language::new("en_US"));
            menu.draw(&mut d, &mut font, &c, (0, 0), system_time_ms, 0);

            let positions: Vec<_> = d
                .vertices()
                .iter()
                .filter(|vertex| vertex.color == [1.0, 1.0, 0.0, 1.0])
                .map(|vertex| vertex.pos)
                .collect();
            assert_eq!(d.transform_depth(), 0, "the matrix stack is balanced");
            (menu.splash_scale, positions)
        };

        let (scale_at_zero, actual_at_zero) = draw_frame(0);
        let (scale_at_quarter_period, actual_at_quarter_period) = draw_frame(250);
        assert!(scale_at_zero > scale_at_quarter_period);
        assert_ne!(
            actual_at_zero, actual_at_quarter_period,
            "the pulse must change emitted positions, not only the stored field"
        );

        // Build the untransformed glyph geometry once. The expected positions below are an
        // independent projection of that geometry through each Java call in source order.
        let mut plain_font = uniform_font();
        let mut plain = DrawList::new();
        draw_centered_string(&mut plain, &mut plain_font, splash, 0, -8, -256);
        let plain_positions: Vec<_> = plain
            .vertices()
            .iter()
            .filter(|vertex| vertex.color == [1.0, 1.0, 0.0, 1.0])
            .map(|vertex| vertex.pos)
            .collect();
        let width = plain_font.string_width(splash);
        let anchor = [(c.width / 2 + 90) as f32, 70.0];
        let project = |system_time_ms, scale| {
            let transform = Transform::translate(anchor[0], anchor[1])
                .then_rotate_degrees(-20.0)
                .then_scale(scale);
            let expected: Vec<_> = plain_positions
                .iter()
                .copied()
                .map(|position| transform.apply(position))
                .collect();
            assert_eq!(
                scale,
                splash_scale(system_time_ms, width),
                "the emitted pulse must use the current frame's scale"
            );
            expected
        };

        for (system_time_ms, scale, actual) in [
            (0, scale_at_zero, &actual_at_zero),
            (250, scale_at_quarter_period, &actual_at_quarter_period),
        ] {
            let expected = project(system_time_ms, scale);
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual[0] - expected[0]).abs() < 1.0e-4);
                assert!((actual[1] - expected[1]).abs() < 1.0e-4);
            }

            // Negative controls for the three visible failure modes: an axis-aligned draw,
            // the wrong rotation sign, or a missing pulse scale cannot match the live geometry.
            let axis_aligned: Vec<_> = plain_positions
                .iter()
                .map(|[x, y]| [x + anchor[0], y + anchor[1]])
                .collect();
            assert_ne!(actual, &axis_aligned, "splash must not be axis-aligned");
            let wrong_sign = Transform::translate(anchor[0], anchor[1])
                .then_rotate_degrees(20.0)
                .then_scale(scale);
            let wrong_sign_positions: Vec<_> = plain_positions
                .iter()
                .copied()
                .map(|position| wrong_sign.apply(position))
                .collect();
            assert_ne!(
                actual, &wrong_sign_positions,
                "splash rotation must be -20 degrees"
            );
            let wrong_anchor = Transform::translate((c.width / 2) as f32, anchor[1])
                .then_rotate_degrees(-20.0)
                .then_scale(scale);
            let wrong_anchor_positions: Vec<_> = plain_positions
                .iter()
                .copied()
                .map(|position| wrong_anchor.apply(position))
                .collect();
            assert_ne!(
                actual, &wrong_anchor_positions,
                "splash must stay centred at width / 2 + 90"
            );
            let unscaled = Transform::translate(anchor[0], anchor[1]).then_rotate_degrees(-20.0);
            let unscaled_positions: Vec<_> = plain_positions
                .iter()
                .copied()
                .map(|position| unscaled.apply(position))
                .collect();
            assert_ne!(
                actual, &unscaled_positions,
                "splash pulse scale must reach vertices"
            );
        }
    }

    #[test]
    fn the_buttons_land_on_vanillas_coordinates_with_singleplayer_disabled() {
        let mut menu = MainMenuScreen::default();
        let c = ctx();
        menu.init(&c, &Language::new("en_US"));
        let j = 480 / 4 + 48;
        let by = |id: i32| menu.buttons.iter().find(|b| b.id == id).expect("button");
        assert_eq!(
            (by(ids::SINGLEPLAYER).rect.x, by(ids::SINGLEPLAYER).rect.y),
            (327, j)
        );
        assert_eq!(by(ids::MULTIPLAYER).rect.y, j + 24);
        assert_eq!(by(ids::REALMS).rect.y, j + 48);
        assert_eq!(
            (by(ids::OPTIONS).rect.x, by(ids::OPTIONS).rect.y),
            (327, j + 84)
        );
        assert_eq!(by(ids::OPTIONS).rect.width, 98);
        assert_eq!(by(ids::QUIT).rect.x, 854 / 2 + 2);
        assert_eq!(
            (by(ids::LANGUAGE).rect.x, by(ids::LANGUAGE).rect.width),
            (854 / 2 - 124, 20)
        );
        // The multiplayer-only rule: no integrated server, and Realms is out of scope.
        assert!(!by(ids::SINGLEPLAYER).enabled);
        assert!(!by(ids::REALMS).enabled);
        assert!(by(ids::MULTIPLAYER).enabled);
    }

    #[test]
    fn live_language_button_draws_the_globe_rows_not_generic_button_art() {
        let draw = |mouse: (i32, i32)| {
            let mut menu = MainMenuScreen::with_update_counter("hi", 1.0);
            let c = ctx();
            menu.init(&c, &Language::new("en_US"));
            let mut font = uniform_font();
            let mut d = DrawList::new();
            menu.draw_foreground(&mut d, &mut font, &c, mouse, 0);
            let widgets = UiTexture::gui(WIDGETS_TEXTURE);
            let batch = d
                .batches()
                .iter()
                .rev()
                .find(|batch| batch.texture == widgets)
                .expect("language widgets batch");
            d.vertices()[batch.range.clone()].to_vec()
        };

        let x = 854 / 2 - 124;
        let y = 480 / 4 + 48 + 84;
        let idle = draw((0, 0));
        assert_eq!(idle.len(), 6, "one icon quad, not generic two-half art");
        assert_eq!(idle[0].pos, [x as f32, y as f32]);
        assert_eq!(idle[0].uv, [0.0, 106.0 / 256.0]);

        let hovered = draw((x, y));
        assert_eq!(hovered.len(), 6);
        assert_eq!(hovered[0].uv, [0.0, 126.0 / 256.0]);

        // Java's bottom-right edges are exclusive. The row offset is `+ height`, not a second
        // hard-coded icon coordinate, and this boundary sample catches an inclusive hit test.
        let right_edge = draw((x + 20, y));
        assert_eq!(right_edge[0].uv, [0.0, 106.0 / 256.0]);
    }

    #[test]
    fn the_buttons_produce_the_right_actions_and_disabled_ones_produce_nothing() {
        let mut menu = MainMenuScreen::default();
        let c = ctx();
        menu.init(&c, &Language::new("en_US"));
        let j = 480 / 4 + 48;
        assert_eq!(menu.mouse_clicked(430, j + 30, 0), Action::OpenMultiplayer);
        assert_eq!(
            menu.mouse_clicked(430, j + 5, 0),
            Action::None,
            "singleplayer is off"
        );
        // Options is w/2-100..w/2-2 and Quit is w/2+2..w/2+100, so the x matters here.
        assert_eq!(menu.mouse_clicked(350, j + 90, 0), Action::OpenOptions);
        assert_eq!(menu.mouse_clicked(500, j + 90, 0), Action::Quit);
        assert_eq!(
            menu.mouse_clicked(430, j + 30, 1),
            Action::None,
            "right click"
        );
    }
}
