//! Screen-space underwater overlay, the animated nether-portal overlay, and a
//! gamma "brighten" pass.
//!
//! java: `ItemRenderer.renderOverlays` (underwater/lava/pumpkin screen tint — confirmed by
//! research to live there, **not** in `EntityRenderer.java`, and not part of the
//! `client/shader/` `ShaderGroup` system either) for the tint; `GuiIngame.renderPortal` (line
//! 1003) for the portal overlay; and `EntityRenderer.updateLightmap`'s gamma-slider lerp for
//! gamma.
//!
//! # Screen effects, textured water and first-person fire
//!
//! The immersion tint is one solid colour alpha-blended "over" the frame so far, which is
//! literally what vanilla draws — `dst = src*(1-a) + tint*a` is exactly what the fixed
//! alpha-blend pipeline stage computes, with no need to read the frame back.
//!
//! The **portal** overlay is not a tint: vanilla stretches one frame of the 32-frame
//! `textures/blocks/portal.png` animation across the whole screen and modulates its alpha by
//! `timeInPortal`. That is what [`OverlayPhase::portal_time`] +
//! [`OverlayPhase::load_portal_texture`] reproduce, frame animation included (see `portal.wgsl`).
//! [`OverlayKind::Portal`]'s flat purple is only the fallback for "no texture loaded yet".
//!
//! Vanilla's gamma slider only ever affects the lightmap texture
//! (`EntityRenderer.updateLightmap`'s `gammaSetting` lerp, ported at
//! [`crate::sky::lightmap`]); `ItemRenderer.renderOverlays` (`ItemRenderer.java:421-464`) draws
//! water, fire and suffocation and nothing else. This phase used to blend an extra full-screen
//! white quad at `gamma * 0.15`, which washed the world out and *doubled* the brightness
//! response on top of the lightmap ramp mc-rs already applies correctly. It no longer draws
//! anything for gamma; [`OverlayPhase::gamma`] is retained only so its existing writer keeps
//! compiling and is deliberately ignored.
//! None of these needs [`crate::SceneTextures`] — they only write the frame, never read it. The
//! phase that does read it is [`crate::post::PostPhase`].

use std::cell::RefCell;

use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::pipeline::first_person_projection;
use crate::{Camera, DepthAttachment, Gpu, RenderPhase};

/// What the camera's eye is immersed in, for the screen tint. java: `ItemRenderer.renderOverlays`'s
/// material dispatch (mirrors [`crate::sky::fog::CameraMedium`], kept separate since a screen
/// overlay also needs the "portal" case fog has no notion of).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverlayKind {
    /// No tint.
    #[default]
    None,
    /// Underwater texture, with a blue fallback only if its resource fails to load.
    Water,
    /// Flat purple stand-in for the nether-portal overlay, used **only** when
    /// [`OverlayPhase::portal_time`] is non-zero and no portal texture has been loaded yet (see
    /// [`OverlayPhase::load_portal_texture`]). The real effect is the animated
    /// `textures/blocks/portal.png` swirl; this is the degraded path, not the intended one.
    Portal,
}

impl OverlayKind {
    /// The flat tint colour and alpha for fallback media. Water's value is used only if its
    /// standalone texture could not be loaded.
    #[must_use]
    pub fn tint(self) -> Option<(Vec3, f32)> {
        match self {
            OverlayKind::None => None,
            OverlayKind::Water => Some((Vec3::new(0.0, 0.2, 0.6), 0.4)),
            OverlayKind::Portal => Some((Vec3::new(0.4, 0.0, 0.6), 0.5)),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OverlayUniform {
    color: [f32; 4],
}

/// Screen-space tint + the animated nether-portal overlay + gamma brighten. Registered at
/// [`crate::PhaseOrder::POST`].
pub struct OverlayPhase {
    /// What the camera is immersed in this frame.
    pub kind: OverlayKind,
    /// `0.0..=1.0`, `GameSettings.gamma`. **Ignored.** 1.8.9's brightness slider only reshapes
    /// the lightmap texture, which [`crate::sky::lightmap`] already does; there is no
    /// full-screen gamma quad in vanilla. Kept as a field so the existing writer compiles.
    pub gamma: f32,
    /// java: `prevTimeInPortal + (timeInPortal - prevTimeInPortal) * partialTicks` from
    /// `GuiIngame.renderFrame`. `0.0` draws no portal overlay; it climbs by `0.0125`/tick while
    /// the player stands in a portal and falls by `0.05`/tick after leaving
    /// (`EntityPlayerSP.updateEntityActionState`). Driven from outside, like every other S20
    /// per-frame field.
    pub portal_time: f32,
    /// Client tick counter driving the portal texture's own 32-frame animation. java:
    /// `TextureMap.updateAnimations`'s per-tick frame advance.
    pub portal_tick: u64,
    /// Whether the local player is burning. The frame driver supplies the local fire flag;
    /// `true` draws Java's two `fire_layer_1` quads in front of the first-person view.
    pub fire: bool,
    /// The Fire Height mod's offset and alpha for those quads. `None` is
    /// vanilla: [`first_person_fire_vertices`] at [`FIRST_PERSON_FIRE_ALPHA`].
    pub fire_style: Option<FireOverlayStyle>,
    /// Atlas rectangle selected from the last visually-opaque eye corner. `None` suppresses the
    /// first-person suffocation overlay for open, no-clip, third-person and sleeping views.
    pub opaque_block_uv: Option<mc_assets::sprite::UvRect>,
    /// Whether the shared depth buffer holds only the first-person hand
    /// when this phase records, so the suffocation quad can be depth-tested against it.
    ///
    /// java: `renderHand` clears depth, draws the hand, then `renderOverlays` draws the quad at
    /// eye depth 0.5 with `GL_DEPTH_TEST` on and `GL_LEQUAL` (measured live by
    /// `javaref/SuffocationRef.java`). The parts of the held item nearer than 0.5 therefore stay
    /// visible in front of it: most of a drawn bow, a sliver of a sword at the right edge. The
    /// frame driver sets this together with [`crate::pipeline::FirstPersonPhase`]'s
    /// `clear_depth_for_overlay`, which guarantees that depth; `false` (the default, for callers
    /// without that guarantee) draws the quad over everything.
    pub opaque_block_hand_depth: bool,
    /// Java's world-border override for `GuiIngame.renderVignette`. The colour is `(0, f, f, 1)`
    /// while the player is inside the warning radius and `None` otherwise. The driver only feeds
    /// this when Fancy graphics is enabled, matching `renderGameOverlay`'s gate.
    pub border_warning: Option<[f32; 4]>,
    /// Java's persistent `GuiIngame.prevVignetteBrightness`, initialized to fully dark and
    /// eased by one percent toward `1 - playerBrightness` on each rendered Fancy frame.
    pub prev_vignette_brightness: f32,
    /// Whether this frame is rendering the Fancy in-world HUD vignette. Fast graphics and
    /// worldless screens suppress the draw without advancing the persistent value.
    pub vignette_active: bool,
    /// The Vignette mod's switch; `true`, the default, is
    /// vanilla. `false` skips only the ordinary brightness darkening: the world-border warning
    /// ([`Self::border_warning`]) still draws through the same texture, and
    /// [`Self::prev_vignette_brightness`] keeps easing, so switching back on resumes exactly
    /// where vanilla would be.
    pub ordinary_vignette: bool,

    /// Java's global `GameSettings.anaglyph`, applied at every decoded overlay upload boundary.
    /// Raw resource bytes stay with each resident texture so a live toggle never double-converts.
    anaglyph: bool,

    pipeline: wgpu::RenderPipeline,
    tint_buffer: wgpu::Buffer,
    tint_group: wgpu::BindGroup,

    opaque_atlas: Option<OpaqueAtlas>,
    opaque_pipeline: wgpu::RenderPipeline,
    /// [`Self::opaque_pipeline`] with Java's `GL_LEQUAL` depth test.
    opaque_depth_pipeline: wgpu::RenderPipeline,
    opaque_buffer: wgpu::Buffer,
    /// The six projected vertices of `renderBlockInHand`'s quad (two triangles).
    opaque_vertex_buffer: wgpu::Buffer,

    portal: Option<PortalTexture>,
    portal_pipeline: wgpu::RenderPipeline,
    portal_layout: wgpu::BindGroupLayout,
    portal_buffer: wgpu::Buffer,
    portal_sampler: wgpu::Sampler,

    vignette: Option<VignetteTexture>,
    vignette_pipeline: wgpu::RenderPipeline,
    vignette_buffer: wgpu::Buffer,
    vignette_write: crate::context::ChangedWrite,
    vignette_sampler: wgpu::Sampler,

    /// java: `thePlayer.getBrightness(partialTicks)`. The frame driver owns this world query;
    /// `1.0` is a bright fallback until that source is supplied.
    pub water_brightness: f32,
    water: Option<WaterTexture>,
    water_pipeline: wgpu::RenderPipeline,
    water_layout: wgpu::BindGroupLayout,
    water_buffer: wgpu::Buffer,
    water_sampler: wgpu::Sampler,

    fire_texture: Option<FireTexture>,
    fire_pipeline: wgpu::RenderPipeline,
    fire_layout: wgpu::BindGroupLayout,
    fire_buffer: wgpu::Buffer,
    fire_sampler: wgpu::Sampler,
    fire_vertex_buffer: wgpu::Buffer,
    fire_vertex_capacity: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PortalUniform {
    params: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OpaqueUniform {
    /// `GlStateManager.color(0.1F, 0.1F, 0.1F, 0.5F)`.
    color: [f32; 4],
}

/// Java's fixed colour multiplier for the suffocation block quad.
pub const OPAQUE_BLOCK_RGB: f32 = 0.1;
/// Java's fixed alpha multiplier for the suffocation block quad. It never reaches the screen:
/// `GL_BLEND` is off when `renderBlockInHand` draws (see [`opaque_block_vertices`]).
pub const OPAQUE_BLOCK_ALPHA: f32 = 0.5;

/// The eye-space depth of `renderBlockInHand`'s quad, `pos(.., .., -0.5)`.
pub const OPAQUE_BLOCK_EYE_DEPTH: f32 = 0.5;

/// `ItemRenderer.renderBlockInHand`'s quad through `renderHand`'s
/// projection, as two triangles of clip-space vertices with their atlas UVs.
///
/// java: after `renderItemInFirstPerson` the model-view is the identity and the projection is
/// still the hand's (`gluPerspective(getFOVModifier(pt, false), aspect, 0.05, far * 2)`), so the
/// quad `(-1..1, -1..1)` at eye `z = -0.5` reaches far past the screen edges. Only its middle is
/// visible: 62% of the sprite's width and 35% of its height at 16:9 and 70 degrees, with square
/// texels (about 193 px each at 1080p). U is mirrored (left = `maxU`), V runs top = `minV`.
///
/// It is drawn with blending **off** and the alpha test off, so the dark texture covers the
/// screen completely; measured on the real renderer by `javaref/SuffocationRef.java`.
#[must_use]
pub fn opaque_block_vertices(cam: &Camera, uv: mc_assets::sprite::UvRect) -> [FireVertex; 6] {
    let projection = first_person_projection(cam);
    let corner = |x: f32, y: f32, u: f32, v: f32| FireVertex {
        clip: (projection * Vec4::new(x, y, -OPAQUE_BLOCK_EYE_DEPTH, 1.0)).to_array(),
        uv: [u, v],
    };
    // The four `POSITION_TEX` vertices, in Java's order, split into two triangles.
    let a = corner(-1.0, -1.0, uv.max_u, uv.max_v);
    let b = corner(1.0, -1.0, uv.min_u, uv.max_v);
    let c = corner(1.0, 1.0, uv.min_u, uv.min_v);
    let d = corner(-1.0, 1.0, uv.max_u, uv.min_v);
    [a, b, c, a, c, d]
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VignetteUniform {
    color: [f32; 4],
    viewport: [f32; 4],
}

/// java: `ItemRenderer.renderWaterOverlayTexture` (`1.8.9`, :505-532). Two vec4s make the
/// Rust/WGSL layout explicit: brightness/yaw/pitch, then framebuffer size.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WaterUniform {
    camera: [f32; 4],
    viewport: [f32; 4],
}

/// Fire's two-quad alpha uniform. java: `GlStateManager.color(1, 1, 1, 0.9F)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FireUniform {
    color: [f32; 4],
}

/// A first-person fire vertex with its clip-space position already projected on the CPU.
///
/// Java switches to the hand projection before `renderOverlays`; carrying clip coordinates lets
/// this post phase preserve that projection without touching the world camera's depth state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FireVertex {
    /// Position after the separate first-person projection.
    pub clip: [f32; 4],
    /// Horizontally mirrored fire-sprite UV.
    pub uv: [f32; 2],
}

/// Vertex layout used by [`FireVertex`].
pub const FIRE_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: core::mem::size_of::<FireVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 16,
            shader_location: 1,
        },
    ],
};

/// Two copies of `ItemRenderer.renderFireInFirstPerson`'s sprite.
const FIRE_WGSL: &str = r#"
struct FireUniform {
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> fire: FireUniform;
@group(0) @binding(1) var fire_tex: texture_2d<f32>;
@group(0) @binding(2) var fire_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@location(0) clip: vec4<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.clip = clip;
    out.uv = uv;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(fire_tex, fire_samp, in.uv);
    // `renderOverlays` starts with `GlStateManager.disableAlpha`; unlike the held-item model,
    // first-person fire therefore has no alpha-test discard, only the fixed 0.9 blend colour.
    return vec4<f32>(texel.rgb * fire.color.rgb, texel.a * fire.color.a);
}
"#;

/// The standalone fire sprite and the one animation frame currently uploaded for it.
struct FireTexture {
    sprite: mc_assets::Sprite,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    uploaded_frame: Option<usize>,
}

/// Java's fixed alpha for the first-person fire copies.
pub const FIRST_PERSON_FIRE_ALPHA: f32 = 0.9;

/// Matrix for one copy of `ItemRenderer.renderFireInFirstPerson` before projection.
#[must_use]
pub fn first_person_fire_quad_matrix(index: usize) -> Mat4 {
    let side = index as f32 * 2.0 - 1.0;
    Mat4::from_translation(Vec3::new(-side * 0.24, -0.3, 0.0))
        * Mat4::from_rotation_y((side * 10.0).to_radians())
}

/// Emit the two mirrored fire quads under Java's separate first-person projection.
///
/// The vertex order, `z = -0.5`, ±0.24 translations, ±10° Y rotations, and horizontal UV
/// reversal are direct transcriptions of `ItemRenderer.renderFireInFirstPerson` (`:539-579`).
#[must_use]
pub fn first_person_fire_vertices(cam: &Camera) -> Vec<FireVertex> {
    let projection = first_person_projection(cam);
    let positions = [
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
    ];
    // f2/f1 and f3/f4 are the sprite's max/min U/V in Java. Standalone loading makes the full
    // texture rect exactly [0,1], but preserve the mirrored order explicitly.
    let uv = [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]];
    let mut out = Vec::with_capacity(12);
    for index in 0..2 {
        let matrix = projection * first_person_fire_quad_matrix(index);
        let quad: [FireVertex; 4] = core::array::from_fn(|corner| FireVertex {
            clip: (matrix
                * Vec4::new(
                    positions[corner].x,
                    positions[corner].y,
                    positions[corner].z,
                    1.0,
                ))
            .into(),
            uv: uv[corner],
        });
        for &corner in &[0usize, 1, 2, 0, 2, 3] {
            out.push(quad[corner]);
        }
    }
    out
}

/// The Fire Height mod's look for the first-person fire.
///
/// Lowering is an offset added to both quads' Y translation
/// before the rotation, so each `0.1` moves the flames by about 14% of the screen (the hand
/// projection spans `y` in `-0.35..0.35` at the quads' depth).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FireOverlayStyle {
    /// Added to vanilla's `-0.3` Y translation. Negative lowers the flames.
    pub height_offset: f32,
    /// The blend alpha that replaces [`FIRST_PERSON_FIRE_ALPHA`].
    pub alpha: f32,
}

/// [`first_person_fire_quad_matrix`] with the Fire Height mod's vertical offset added to the
/// translation.
#[must_use]
pub fn first_person_fire_quad_matrix_offset(index: usize, height_offset: f32) -> Mat4 {
    let side = index as f32 * 2.0 - 1.0;
    Mat4::from_translation(Vec3::new(-side * 0.24, -0.3 + height_offset, 0.0))
        * Mat4::from_rotation_y((side * 10.0).to_radians())
}

/// [`first_person_fire_vertices`] with both quads moved by `height_offset`. The vanilla function
/// is left untouched so the mod's off path stays bit-identical.
#[must_use]
pub fn first_person_fire_vertices_offset(cam: &Camera, height_offset: f32) -> Vec<FireVertex> {
    let projection = first_person_projection(cam);
    let positions = [
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
    ];
    let uv = [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]];
    let mut out = Vec::with_capacity(12);
    for index in 0..2 {
        let matrix = projection * first_person_fire_quad_matrix_offset(index, height_offset);
        let quad: [FireVertex; 4] = core::array::from_fn(|corner| FireVertex {
            clip: (matrix * positions[corner].extend(1.0)).into(),
            uv: uv[corner],
        });
        for &corner in &[0usize, 1, 2, 0, 2, 3] {
            out.push(quad[corner]);
        }
    }
    out
}

/// The decoded `textures/blocks/portal.png` animation plus the single-frame GPU texture vanilla's
/// `updateAnimations` would be rewriting in the block atlas.
struct PortalTexture {
    sprite: mc_assets::Sprite,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    /// Which entry of `sprite.frames` currently lives in `texture`.
    uploaded_frame: Option<usize>,
}

/// The standalone `textures/misc/vignette.png` image used by `GuiIngame.renderVignette`.
struct VignetteTexture {
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    /// Raw decoded bytes retained so a live anaglyph toggle can rebuild the upload without
    /// applying Java's integer transform twice.
    source_rgba: Vec<u8>,
}

/// Java's `GuiIngame.renderVignette` blend: `(GL_ZERO, GL_ONE_MINUS_SRC_COLOR, GL_ONE, GL_ZERO)`.
/// The warning colour `(0, f, f)` therefore suppresses only the green/blue channels at the
/// forcefield edge, leaving the red channel bright exactly as vanilla's cyan warning does.
fn vignette_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::OneMinusSrc,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// The largest ordinary-vignette brightness `b` whose draw provably leaves every colour channel of
/// the frame unchanged, `2^-25`.
///
/// The colour blend is `src * 0 + dst * (1 - src)` with `src = texel * b` and `texel <= 1`. For
/// `b <= 2^-25`, `1 - src` rounds to exactly `1.0` in `f32` (the next value below `1.0` is
/// `1 - 2^-24`; `1 - 2^-25` is the midpoint and ties to the even `1.0`), and any coarser blend
/// unit quantises `src` to `0` first, so `dst * 1` is returned bit for bit. `b` gets here outdoors:
/// in full light the target is `0`, and `0.99^n` drops below `2^-25` after ~1,730 rendered frames,
/// then sticks at a subnormal (it never reaches `0.0` exactly, so an `== 0.0` gate would never
/// fire). The value itself keeps easing every frame; only the no-op draw is skipped.
///
/// The draw's alpha factors are `(One, Zero)`, i.e. it would also overwrite destination alpha with
/// `vignette.png`'s alpha. Alpha is never presented (the surface is opaque) and F2 screenshots
/// are written opaque like vanilla's `TYPE_INT_RGB` (`save_screenshot_png`), so that write is not
/// observable either.
const VIGNETTE_IDENTITY_MAX: f32 = 1.0 / 33_554_432.0;

/// One rendered-frame update of `GuiIngame.prevVignetteBrightness`.
fn advance_vignette_brightness(previous: f32, player_brightness: f32) -> f32 {
    let target = (1.0 - player_brightness).clamp(0.0, 1.0);
    (f64::from(previous) + f64::from(target - previous) * 0.01) as f32
}

/// The standalone `textures/misc/underwater.png` texture. Vanilla binds this file directly;
/// it is not a block-atlas sprite.
struct WaterTexture {
    // Own the texture for as long as its bind group is installed on the phase.
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    /// Raw decoded bytes retained for exact false -> true -> false setting toggles.
    source_rgba: Vec<u8>,
}

/// The stitched block atlas sampled by `ItemRenderer.renderBlockInHand`'s suffocation quad.
struct OpaqueAtlas {
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    /// Raw level-zero and resident mip bytes.  Java converts each temporary upload buffer, not
    /// the decoded atlas, so this cache is also what makes animation and live toggles exact.
    /// Shared by the owners of the shared atlas.
    source_levels: std::rc::Rc<RefCell<Vec<Vec<u8>>>>,
}

/// Which animation frame is showing after `tick` client ticks. java: `TextureAtlasSprite.updateAnimation`
/// — `++frameCounter; if (frameCounter >= frameTime) { frameIndex = (frameIndex + 1) % frames; }`,
/// generalised here to honour per-frame `time` overrides in the `.mcmeta` (vanilla's
/// `getFrameTimeSingle`). `portal.png.mcmeta` is `{"animation": {}}`, i.e. 32 frames at 1 tick
/// each, so this reduces to `tick % 32` for the real asset.
#[must_use]
fn frame_at_tick(sprite: &mc_assets::Sprite, tick: u64) -> usize {
    let durations: Vec<u64> = sprite
        .frames
        .iter()
        .map(|f| {
            u64::from(
                f.duration_ticks
                    .unwrap_or(sprite.default_duration_ticks)
                    .max(1),
            )
        })
        .collect();
    let total: u64 = durations.iter().sum();
    if total == 0 {
        return 0;
    }
    let mut t = tick % total;
    for (i, d) in durations.iter().enumerate() {
        if t < *d {
            return i;
        }
        t -= d;
    }
    0
}

fn mip_dimensions(width: u32, height: u32, mip_level: u32) -> (u32, u32) {
    (
        width.checked_shr(mip_level).unwrap_or(0).max(1),
        height.checked_shr(mip_level).unwrap_or(0).max(1),
    )
}

fn rgba_len(width: u32, height: u32) -> Option<usize> {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
}

fn valid_source_levels(levels: &[Vec<u8>], width: u32, height: u32) -> bool {
    !levels.is_empty()
        && levels.iter().enumerate().all(|(level, rgba)| {
            let (level_width, level_height) = mip_dimensions(width, height, level as u32);
            rgba_len(level_width, level_height).is_some_and(|expected| rgba.len() == expected)
        })
}

fn atlas_animation_rect_valid(
    upload: &mc_assets::AtlasAnimationUpload,
    width: u32,
    height: u32,
    mip_level_count: u32,
) -> bool {
    if upload.mip_level >= mip_level_count
        || upload.width == 0
        || upload.height == 0
        || rgba_len(upload.width, upload.height)
            .is_none_or(|expected| upload.rgba.len() != expected)
    {
        return false;
    }
    let (level_width, level_height) = mip_dimensions(width, height, upload.mip_level);
    upload
        .x
        .checked_add(upload.width)
        .is_some_and(|end| end <= level_width)
        && upload
            .y
            .checked_add(upload.height)
            .is_some_and(|end| end <= level_height)
}

/// Build the temporary RGBA8 upload array used by Java's `TextureUtil.copyToBufferPos`.
///
/// Every decoded overlay resource reaches the GPU through this one boundary.  The source is
/// never mutated: Java applies `TextureUtil.anaglyphColor` to the temporary upload array, and
/// retaining raw bytes is what makes live setting toggles and animated frames reversible.
fn overlay_upload_rgba(source_rgba: &[u8], anaglyph: bool) -> Vec<u8> {
    mc_assets::sprite::anaglyph_rgba(source_rgba, anaglyph)
}

fn ensure_fire_vertex_buffer(
    gpu: &Gpu,
    buffer: &mut wgpu::Buffer,
    capacity: &mut usize,
    vertices: &[FireVertex],
) {
    if vertices.len() > *capacity {
        *buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post first-person fire vertices"),
            size: core::mem::size_of_val(vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *capacity = vertices.len();
    }
    gpu.queue
        .write_buffer(buffer, 0, bytemuck::cast_slice(vertices));
}

/// Java's `tryBlendFuncSeparate(770, 771, 1, 0)` used by `renderFireInFirstPerson`.
fn fire_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// java: `GuiIngame.renderPortal`'s alpha ramp — `t^4 * 0.8 + 0.2`, applied only below 1.0, so
/// the overlay snaps to a visible 20% the instant the player enters a portal instead of fading
/// in from nothing.
#[must_use]
pub fn portal_alpha(time_in_portal: f32) -> f32 {
    if time_in_portal < 1.0 {
        let t2 = time_in_portal * time_in_portal;
        let t4 = t2 * t2;
        t4 * 0.8 + 0.2
    } else {
        time_in_portal
    }
}

/// The four-tile UV mapping of `ItemRenderer.renderWaterOverlayTexture`, expressed for a
/// top-left-origin normalized screen coordinate. The WGSL water phase mirrors this exactly.
#[must_use]
pub fn underwater_uv(screen: [f32; 2], yaw: f32, pitch: f32) -> [f32; 2] {
    [
        4.0 - screen[0] * 4.0 - yaw / 64.0,
        screen[1] * 4.0 + pitch / 64.0,
    ]
}

/// Sanitize the local eye-block brightness before it reaches the underwater shader. Vanilla's
/// world light query is bounded, while this explicit fallback keeps a missing producer visible
/// instead of turning a malformed value into a black overlay.
#[must_use]
pub fn water_brightness_value(brightness: f32) -> f32 {
    if brightness.is_finite() {
        brightness.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

impl OverlayPhase {
    /// Build the phase.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        Self::new_with_anaglyph(gpu, false)
    }

    /// Build the phase with Java's global upload-time anaglyph switch selected.
    ///
    /// The compatibility [`Self::new`] constructor keeps anaglyph disabled.  A client that owns
    /// `GameSettings` can seed the phase here, or call [`Self::set_anaglyph`] before its first
    /// resource-backed upload.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post overlay"),
                source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
            });

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("post overlay group0"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("post overlay layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post overlay"),
                layout: Some(&pipeline_layout),
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

        let make_buffer_and_group = |label: &str| {
            let buffer = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::bytes_of(&OverlayUniform {
                        color: [0.0, 0.0, 0.0, 0.0],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            (buffer, group)
        };
        let (tint_buffer, tint_group) = make_buffer_and_group("post overlay tint");

        // ---- the animated portal overlay's own pipeline (uniform + texture + sampler)
        let portal_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post portal"),
                source: wgpu::ShaderSource::Wgsl(include_str!("portal.wgsl").into()),
            });
        let portal_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post portal group0"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let portal_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("post portal layout"),
                    bind_group_layouts: &[Some(&portal_layout)],
                    immediate_size: 0,
                });
        let portal_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post portal"),
                layout: Some(&portal_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &portal_shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &portal_shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        // java: tryBlendFuncSeparate(770, 771, 1, 0) == straight alpha "over".
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let portal_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("post portal uniform"),
                contents: bytemuck::bytes_of(&PortalUniform {
                    params: [0.0, 1.0, 1.0, 0.0],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        // Nearest, like every block texture in 1.8.9 — the portal overlay is a 16x16 frame
        // stretched over the whole screen, and vanilla shows it as 16x16 blocks of flat colour.
        let portal_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post portal sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // ---- first-person opaque-block overlay: one block-atlas particle sprite ------------
        let opaque_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post opaque block"),
                source: wgpu::ShaderSource::Wgsl(include_str!("opaque.wgsl").into()),
            });
        let opaque_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("post opaque block layout"),
                    bind_group_layouts: &[Some(&portal_layout)],
                    immediate_size: 0,
                });
        // java: the quad is drawn with GL_BLEND off (and the alpha
        // test off), so its 0.5 alpha never reaches the screen and the dark texture covers it
        // completely. Only colour is written: the frame's alpha stays what the world left, as it
        // did under the old 50% blend over an opaque frame. The second variant adds Java's
        // GL_LEQUAL test against the first-person hand's depth (`opaque_block_hand_depth`).
        let make_opaque_pipeline = |label: &str, depth: Option<wgpu::DepthStencilState>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&opaque_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &opaque_shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[Some(FIRE_VERTEX_LAYOUT)],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: depth,
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &opaque_shader,
                        entry_point: Some("fs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::COLOR,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let opaque_pipeline = make_opaque_pipeline("post opaque block", None);
        let opaque_depth_pipeline = make_opaque_pipeline(
            "post opaque block (hand depth)",
            Some(wgpu::DepthStencilState {
                format: crate::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        );
        let opaque_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("post opaque block uniform"),
                contents: bytemuck::bytes_of(&OpaqueUniform {
                    color: [
                        OPAQUE_BLOCK_RGB,
                        OPAQUE_BLOCK_RGB,
                        OPAQUE_BLOCK_RGB,
                        OPAQUE_BLOCK_ALPHA,
                    ],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let opaque_vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post opaque block vertices"),
            size: (6 * core::mem::size_of::<FireVertex>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // `vignette.png.mcmeta` requests blur=true; unlike the block portal sprite, its edge
        // gradient therefore uses linear filtering in vanilla.
        let vignette_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post vignette sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // ---- Java's full-screen world-border vignette override -----------------------------
        // `GuiIngame.renderVignette` uses the same texture file for ordinary darkness and for
        // the border warning. This phase currently receives only the warning branch; retaining
        // the standalone pipeline here keeps its texture/blend semantics exact without making
        // the mc-ui crate know about world state.
        let vignette_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post vignette"),
                source: wgpu::ShaderSource::Wgsl(include_str!("vignette.wgsl").into()),
            });
        let vignette_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("post vignette layout"),
                    bind_group_layouts: &[Some(&portal_layout)],
                    immediate_size: 0,
                });
        let vignette_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("post vignette"),
                    layout: Some(&vignette_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &vignette_shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &vignette_shader,
                        entry_point: Some("fs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(vignette_blend_state()),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
        let vignette_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("post vignette uniform"),
                contents: bytemuck::bytes_of(&VignetteUniform {
                    color: [0.0, 0.0, 0.0, 1.0],
                    viewport: [1.0, 1.0, 0.0, 0.0],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let water_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post underwater"),
                source: wgpu::ShaderSource::Wgsl(include_str!("underwater.wgsl").into()),
            });
        let water_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post underwater group0"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let water_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("post underwater layout"),
                    bind_group_layouts: &[Some(&water_layout)],
                    immediate_size: 0,
                });
        let water_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post underwater"),
                layout: Some(&water_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &water_shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &water_shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        // java: `tryBlendFuncSeparate(770, 771, 1, 0)`.
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let water_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("post underwater uniform"),
                contents: bytemuck::bytes_of(&WaterUniform {
                    camera: [1.0, 0.0, 0.0, 0.0],
                    viewport: [1.0, 1.0, 0.0, 0.0],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let water_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post underwater sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        // ---- first-person fire: two clip-space quads, always in front of the world
        let fire_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post first-person fire"),
                source: wgpu::ShaderSource::Wgsl(FIRE_WGSL.into()),
            });
        let fire_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post first-person fire group0"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let fire_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("post first-person fire layout"),
                    bind_group_layouts: &[Some(&fire_layout)],
                    immediate_size: 0,
                });
        let fire_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post first-person fire"),
                layout: Some(&fire_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &fire_shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(FIRE_VERTEX_LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // Both copies are screen overlays and must not be removed by winding/cull
                    // state inherited from an earlier world draw.
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fire_shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: Some(fire_blend_state()),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let fire_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("post first-person fire uniform"),
                contents: bytemuck::bytes_of(&FireUniform {
                    color: [1.0, 1.0, 1.0, FIRST_PERSON_FIRE_ALPHA],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let fire_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post first-person fire sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let fire_vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("post first-person fire vertices"),
                contents: bytemuck::bytes_of(&FireVertex {
                    clip: [0.0; 4],
                    uv: [0.0; 2],
                }),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            kind: OverlayKind::None,
            gamma: 0.0,
            portal_time: 0.0,
            portal_tick: 0,
            fire: false,
            fire_style: None,
            opaque_block_uv: None,
            opaque_block_hand_depth: false,
            border_warning: None,
            prev_vignette_brightness: 1.0,
            vignette_active: false,
            ordinary_vignette: true,
            anaglyph,
            pipeline,
            tint_buffer,
            tint_group,
            opaque_atlas: None,
            opaque_pipeline,
            opaque_depth_pipeline,
            opaque_buffer,
            opaque_vertex_buffer,
            portal: None,
            portal_pipeline,
            portal_layout,
            portal_buffer,
            portal_sampler,
            vignette: None,
            vignette_pipeline,
            vignette_buffer,
            vignette_write: crate::context::ChangedWrite::default(),
            vignette_sampler,
            water_brightness: 1.0,
            water: None,
            water_pipeline,
            water_layout,
            water_buffer,
            water_sampler,
            fire_texture: None,
            fire_pipeline,
            fire_layout,
            fire_buffer,
            fire_sampler,
            fire_vertex_buffer,
            fire_vertex_capacity: 1,
        }
    }

    fn install_block_atlas_levels(
        &mut self,
        gpu: &Gpu,
        source_levels: Vec<Vec<u8>>,
        width: u32,
        height: u32,
    ) {
        debug_assert!(valid_source_levels(&source_levels, width, height));
        // Adopts the one shared GPU block atlas inside the client's
        // install sweep; a plain create anywhere else (including the anaglyph replay below).
        let texture = crate::atlas_transfer::share_atlas_texture(
            gpu,
            &wgpu::TextureDescriptor {
                label: Some("post opaque block atlas"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: source_levels.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            &source_levels,
            self.anaglyph,
        );
        for (level, source_rgba) in source_levels.iter().enumerate() {
            let level = level as u32;
            let (level_width, level_height) = mip_dimensions(width, height, level);
            let upload_rgba = overlay_upload_rgba(source_rgba, self.anaglyph);
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &upload_rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(level_width * 4),
                    rows_per_image: Some(level_height),
                },
                wgpu::Extent3d {
                    width: level_width,
                    height: level_height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post opaque block group0"),
            layout: &self.portal_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.opaque_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.portal_sampler),
                },
            ],
        });
        // One raw shadow for every owner of the shared atlas.
        let source_levels = crate::atlas_transfer::share_atlas_shadow(&texture, source_levels);
        self.opaque_atlas = Some(OpaqueAtlas {
            _texture: texture,
            bind_group,
            width,
            height,
            source_levels,
        });
    }

    /// Install the stitched block atlas sampled by the first-person suffocation overlay.
    ///
    /// The complete source mip chain is retained and uploaded level by level.  This mirrors
    /// `TextureMap`'s sprite-local mip uploads while keeping the atlas's nearest/clamp sampler and
    /// all existing overlay blend/depth semantics unchanged.
    pub fn set_block_atlas(&mut self, gpu: &Gpu, atlas: &mc_assets::Atlas) {
        if atlas.w == 0 || atlas.h == 0 {
            return;
        }
        let mut source_levels = Vec::with_capacity(1 + atlas.mipmaps.len());
        source_levels.push(atlas.texture.clone());
        source_levels.extend(atlas.mipmaps.iter().cloned());
        if !valid_source_levels(&source_levels, atlas.w, atlas.h) {
            return;
        }
        self.install_block_atlas_levels(gpu, source_levels, atlas.w, atlas.h);
    }

    /// Apply an animated block-atlas rectangle to the suffocation overlay's retained copy.
    pub fn upload_block_atlas_animation(
        &self,
        gpu: &Gpu,
        upload: &mc_assets::AtlasAnimationUpload,
    ) -> bool {
        self.upload_block_atlas_animation_with_sink(upload, |texture, anaglyph| {
            let upload_rgba = overlay_upload_rgba(&upload.rgba, anaglyph);
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: upload.mip_level,
                    origin: wgpu::Origin3d {
                        x: upload.x,
                        y: upload.y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload_rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.width * 4),
                    rows_per_image: Some(upload.height),
                },
                wgpu::Extent3d {
                    width: upload.width,
                    height: upload.height,
                    depth_or_array_layers: 1,
                },
            );
        })
    }

    /// Experimental transfer seam: validate and retain raw pixels before invoking the sink.
    /// The sink receives this owner's exact texture and current transform setting.
    #[doc(hidden)]
    pub fn upload_block_atlas_animation_with_sink(
        &self,
        upload: &mc_assets::AtlasAnimationUpload,
        sink: impl FnOnce(&wgpu::Texture, bool),
    ) -> bool {
        let Some(atlas) = &self.opaque_atlas else {
            return false;
        };
        let mip_level_count = atlas.source_levels.borrow().len() as u32;
        if !atlas_animation_rect_valid(upload, atlas.width, atlas.height, mip_level_count) {
            return false;
        }
        let (level_width, _) = mip_dimensions(atlas.width, atlas.height, upload.mip_level);
        {
            let mut source_levels = atlas.source_levels.borrow_mut();
            let source = &mut source_levels[upload.mip_level as usize];
            for row in 0..upload.height {
                let src_start = (row * upload.width * 4) as usize;
                let dst_start = (((upload.y + row) * level_width + upload.x) * 4) as usize;
                let count = (upload.width * 4) as usize;
                source[dst_start..dst_start + count]
                    .copy_from_slice(&upload.rgba[src_start..src_start + count]);
            }
        }
        sink(&atlas._texture, self.anaglyph);
        true
    }

    /// Select the block-model particle sprite for this frame's suffocation overlay.
    pub fn set_opaque_block_uv(&mut self, uv: Option<mc_assets::sprite::UvRect>) {
        self.opaque_block_uv = uv;
    }

    /// Load vanilla's standalone `textures/misc/vignette.png` image. A malformed resource pack
    /// leaves the warning state live but harmlessly skips its draw, just like the other optional
    /// overlay textures in this phase.
    pub fn load_vignette_texture(&mut self, gpu: &Gpu, rm: &mc_assets::ResourceManager) {
        let Ok(loc) = mc_assets::ResourceLocation::parse("textures/misc/vignette") else {
            return;
        };
        let Ok(sprite) = mc_assets::sprite::load_sprite(rm, &loc) else {
            return;
        };
        let Some(frame) = sprite.frames.first() else {
            return;
        };
        if sprite.width == 0
            || sprite.height == 0
            || rgba_len(sprite.width, sprite.height)
                .is_none_or(|expected| frame.rgba.len() != expected)
        {
            return;
        }

        self.install_vignette_source(gpu, frame.rgba.clone(), sprite.width, sprite.height);
    }

    fn install_vignette_source(
        &mut self,
        gpu: &Gpu,
        source_rgba: Vec<u8>,
        width: u32,
        height: u32,
    ) {
        debug_assert_eq!(rgba_len(width, height), Some(source_rgba.len()));
        let upload_rgba = overlay_upload_rgba(&source_rgba, self.anaglyph);

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post vignette texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post vignette group0"),
            layout: &self.portal_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.vignette_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.vignette_sampler),
                },
            ],
        });
        self.vignette = Some(VignetteTexture {
            _texture: texture,
            bind_group,
            width,
            height,
            source_rgba,
        });
    }

    /// Feed the Java world-border branch of `GuiIngame.renderVignette` for this frame.
    pub fn set_border_warning(&mut self, warning: Option<[f32; 4]>) {
        self.border_warning = warning;
    }

    /// Advance and gate Java's ordinary brightness vignette for this rendered frame.
    ///
    /// `Some` means the Fancy in-world HUD is being drawn and advances the persistent value
    /// exactly once. `None` suppresses this frame without mutating it, matching the Fast branch
    /// that skips `GuiIngame.renderVignette` entirely.
    pub fn set_vignette_brightness(&mut self, player_brightness: Option<f32>) {
        self.vignette_active = player_brightness.is_some();
        if let Some(player_brightness) = player_brightness {
            self.prev_vignette_brightness =
                advance_vignette_brightness(self.prev_vignette_brightness, player_brightness);
        }
    }

    /// Whether this frame's ordinary brightness vignette would change any colour channel. False
    /// once `prev_vignette_brightness` has eased to [`VIGNETTE_IDENTITY_MAX`] or below (outdoors),
    /// where the full-screen draw is an exact identity; see that constant for the proof.
    fn ordinary_vignette_visible(&self) -> bool {
        // W4: the Vignette mod off hides only this branch, never the border warning.
        self.ordinary_vignette
            && self.vignette_active
            && self.prev_vignette_brightness > VIGNETTE_IDENTITY_MAX
    }

    /// Whether the standalone vignette image is available to the warning draw.
    #[must_use]
    pub fn has_vignette_texture(&self) -> bool {
        self.vignette.is_some()
    }

    /// Load vanilla's standalone underwater texture. A failed resource lookup leaves the old flat
    /// fallback intact, which keeps a malformed resource pack non-fatal.
    pub fn load_water_texture(&mut self, gpu: &Gpu, rm: &mc_assets::ResourceManager) {
        let Ok(loc) = mc_assets::ResourceLocation::parse("textures/misc/underwater") else {
            return;
        };
        let Ok(sprite) = mc_assets::sprite::load_sprite(rm, &loc) else {
            return;
        };
        let Some(frame) = sprite.frames.first() else {
            return;
        };
        if sprite.width == 0
            || sprite.height == 0
            || rgba_len(sprite.width, sprite.height)
                .is_none_or(|expected| frame.rgba.len() != expected)
        {
            return;
        }

        self.install_water_source(gpu, frame.rgba.clone(), sprite.width, sprite.height);
    }

    fn install_water_source(&mut self, gpu: &Gpu, source_rgba: Vec<u8>, width: u32, height: u32) {
        debug_assert_eq!(rgba_len(width, height), Some(source_rgba.len()));
        let upload_rgba = overlay_upload_rgba(&source_rgba, self.anaglyph);

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post underwater texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post underwater group0"),
            layout: &self.water_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.water_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.water_sampler),
                },
            ],
        });
        self.water = Some(WaterTexture {
            _texture: texture,
            bind_group,
            width,
            height,
            source_rgba,
        });
    }

    /// Set the local player's eye-block brightness used by the underwater texture. Java's
    /// `getBrightness(partialTicks)` is in `0..=1`; non-finite input falls back to the same bright
    /// value used before the world-light producer is connected.
    pub fn set_water_brightness(&mut self, brightness: f32) {
        self.water_brightness = water_brightness_value(brightness);
    }

    /// Load vanilla's `textures/blocks/fire_layer_1` sprite for the local first-person fire
    /// overlay. The decoded frame is kept separately from the atlas because this phase uploads
    /// only the selected frame and does not own S09's atlas animation clock.
    pub fn load_fire_texture(&mut self, gpu: &Gpu, rm: &mc_assets::ResourceManager) {
        let Ok(loc) = mc_assets::ResourceLocation::parse("textures/blocks/fire_layer_1") else {
            return;
        };
        let Ok(sprite) = mc_assets::sprite::load_sprite(rm, &loc) else {
            return;
        };
        if sprite.frames.is_empty() || sprite.width == 0 || sprite.height == 0 {
            return;
        }
        if !sprite.frames.iter().all(|frame| {
            rgba_len(sprite.width, sprite.height)
                .is_some_and(|expected| frame.rgba.len() == expected)
        }) {
            return;
        }

        self.install_fire_sprite(gpu, sprite, Some(0));
    }

    fn install_fire_sprite(
        &mut self,
        gpu: &Gpu,
        sprite: mc_assets::Sprite,
        uploaded_frame: Option<usize>,
    ) {
        debug_assert!(!sprite.frames.is_empty());
        debug_assert_eq!(
            rgba_len(sprite.width, sprite.height),
            Some(sprite.frames[0].rgba.len())
        );

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post first-person fire texture"),
            size: wgpu::Extent3d {
                width: sprite.width,
                height: sprite.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let uploaded_frame = uploaded_frame.filter(|&index| index < sprite.frames.len());
        if let Some(index) = uploaded_frame {
            let upload_rgba = overlay_upload_rgba(&sprite.frames[index].rgba, self.anaglyph);
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &upload_rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(sprite.width * 4),
                    rows_per_image: Some(sprite.height),
                },
                wgpu::Extent3d {
                    width: sprite.width,
                    height: sprite.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post first-person fire group0"),
            layout: &self.fire_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.fire_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.fire_sampler),
                },
            ],
        });
        self.fire_texture = Some(FireTexture {
            sprite,
            texture,
            bind_group,
            uploaded_frame,
        });
    }

    /// Whether the local fire sprite has loaded successfully.
    #[must_use]
    pub fn has_fire_texture(&self) -> bool {
        self.fire_texture.is_some()
    }

    /// Which fire animation frame is selected by [`OverlayPhase::portal_tick`]. The shared tick
    /// is intentional: Java's `TextureMap.updateAnimations` advances both sprites together.
    #[must_use]
    pub fn fire_frame(&self) -> Option<usize> {
        self.fire_texture
            .as_ref()
            .map(|f| frame_at_tick(&f.sprite, self.portal_tick))
    }

    fn upload_fire_frame(&mut self, gpu: &Gpu) {
        let tick = self.portal_tick;
        let anaglyph = self.anaglyph;
        let Some(fire) = &mut self.fire_texture else {
            return;
        };
        let index = frame_at_tick(&fire.sprite, tick);
        if fire.uploaded_frame == Some(index) {
            return;
        }
        let frame = &fire.sprite.frames[index];
        let upload_rgba = overlay_upload_rgba(&frame.rgba, anaglyph);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fire.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(fire.sprite.width * 4),
                rows_per_image: Some(fire.sprite.height),
            },
            wgpu::Extent3d {
                width: fire.sprite.width,
                height: fire.sprite.height,
                depth_or_array_layers: 1,
            },
        );
        fire.uploaded_frame = Some(index);
    }

    /// Load `textures/blocks/portal.png` and its `.mcmeta` animation, so [`OverlayPhase::portal_time`]
    /// draws vanilla's real swirling overlay rather than the [`OverlayKind::Portal`] flat fallback.
    ///
    /// Uses S09's [`mc_assets::sprite::load_sprite`], which is what parses the `.mcmeta` frame list and
    /// `frametime`; the 16x512 strip in `1.8.9.jar` becomes 32 individual 16x16 frames. Only the
    /// *current* frame lives on the GPU at a time and is re-uploaded when it changes, which is
    /// exactly what `TextureMap.updateAnimations` does to the block atlas.
    ///
    /// A missing or malformed texture leaves the phase on the flat-tint fallback rather than
    /// failing — this runs during asset load, where a broken resource pack must not be fatal.
    pub fn load_portal_texture(&mut self, gpu: &Gpu, rm: &mc_assets::ResourceManager) {
        // Kept as the existing public asset-load entry point, so the already-live constructor
        // loads the standalone overlay textures without a second driver edit.
        self.load_vignette_texture(gpu, rm);
        self.load_water_texture(gpu, rm);
        self.load_fire_texture(gpu, rm);
        let Ok(loc) = mc_assets::ResourceLocation::parse("textures/blocks/portal") else {
            return;
        };
        let Ok(sprite) = mc_assets::sprite::load_sprite(rm, &loc) else {
            return;
        };
        if sprite.frames.is_empty() || sprite.width == 0 || sprite.height == 0 {
            return;
        }
        if !sprite.frames.iter().all(|frame| {
            rgba_len(sprite.width, sprite.height)
                .is_some_and(|expected| frame.rgba.len() == expected)
        }) {
            return;
        }

        self.install_portal_sprite(gpu, sprite, None);
    }

    fn install_portal_sprite(
        &mut self,
        gpu: &Gpu,
        sprite: mc_assets::Sprite,
        uploaded_frame: Option<usize>,
    ) {
        debug_assert!(!sprite.frames.is_empty());

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post portal frame"),
            size: wgpu::Extent3d {
                width: sprite.width,
                height: sprite.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Linear, not `Rgba8UnormSrgb`: 1.8.9 uploads the block atlas as plain `GL_RGBA8`
            // with no sRGB conversion anywhere, so a texel's stored bytes are the values the
            // blend operates on. Keeping that here is both faithful and what lets
            // `tests/s20_post.rs` assert exact expected pixels.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let uploaded_frame = uploaded_frame.filter(|&index| index < sprite.frames.len());
        if let Some(index) = uploaded_frame {
            let upload_rgba = overlay_upload_rgba(&sprite.frames[index].rgba, self.anaglyph);
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &upload_rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(sprite.width * 4),
                    rows_per_image: Some(sprite.height),
                },
                wgpu::Extent3d {
                    width: sprite.width,
                    height: sprite.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post portal group0"),
            layout: &self.portal_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.portal_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.portal_sampler),
                },
            ],
        });

        self.portal = Some(PortalTexture {
            sprite,
            texture,
            bind_group,
            uploaded_frame,
        });
    }

    /// Whether the real animated portal texture is loaded. `false` means
    /// [`OverlayPhase::portal_time`] falls back to [`OverlayKind::Portal`]'s flat tint.
    #[must_use]
    pub fn has_portal_texture(&self) -> bool {
        self.portal.is_some()
    }

    /// Which animation frame [`OverlayPhase::portal_tick`] currently selects, or `None` when no
    /// portal texture is loaded. For tests and the F3 overlay.
    #[must_use]
    pub fn portal_frame(&self) -> Option<usize> {
        self.portal
            .as_ref()
            .map(|p| frame_at_tick(&p.sprite, self.portal_tick))
    }

    /// Push the frame `portal_tick` selects to the GPU, if it is not already there.
    fn upload_portal_frame(&mut self, gpu: &Gpu) {
        let tick = self.portal_tick;
        let anaglyph = self.anaglyph;
        let Some(portal) = &mut self.portal else {
            return;
        };
        let index = frame_at_tick(&portal.sprite, tick);
        if portal.uploaded_frame == Some(index) {
            return;
        }
        let frame = &portal.sprite.frames[index];
        let upload_rgba = overlay_upload_rgba(&frame.rgba, anaglyph);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &portal.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(portal.sprite.width * 4),
                rows_per_image: Some(portal.sprite.height),
            },
            wgpu::Extent3d {
                width: portal.sprite.width,
                height: portal.sprite.height,
                depth_or_array_layers: 1,
            },
        );
        portal.uploaded_frame = Some(index);
    }

    /// Whether this phase applies Java's global upload-time anaglyph transform.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global transform and rebuild every resident overlay texture from its raw
    /// source.  Replaying the retained sources is important: transforming current GPU bytes
    /// would double-convert an already-enabled texture, and would lose the latest dirty atlas
    /// mip or animated fire/portal frame.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }

        // Snapshot all raw sources before replacing any GPU allocation.  The texture install
        // helpers also update the corresponding phase slot, so no old bind group or frame state
        // survives accidentally.
        let opaque = self.opaque_atlas.as_ref().map(|atlas| {
            (
                atlas.width,
                atlas.height,
                atlas.source_levels.borrow().clone(),
            )
        });
        let vignette = self
            .vignette
            .as_ref()
            .map(|texture| (texture.width, texture.height, texture.source_rgba.clone()));
        let water = self
            .water
            .as_ref()
            .map(|texture| (texture.width, texture.height, texture.source_rgba.clone()));
        let fire = self
            .fire_texture
            .as_ref()
            .map(|texture| (texture.sprite.clone(), texture.uploaded_frame));
        let portal = self
            .portal
            .as_ref()
            .map(|texture| (texture.sprite.clone(), texture.uploaded_frame));

        self.anaglyph = enabled;

        if let Some((width, height, source_levels)) = opaque {
            self.install_block_atlas_levels(gpu, source_levels, width, height);
        }
        if let Some((width, height, source_rgba)) = vignette {
            self.install_vignette_source(gpu, source_rgba, width, height);
        }
        if let Some((width, height, source_rgba)) = water {
            self.install_water_source(gpu, source_rgba, width, height);
        }
        if let Some((sprite, uploaded_frame)) = fire {
            self.install_fire_sprite(gpu, sprite, uploaded_frame);
        }
        if let Some((sprite, uploaded_frame)) = portal {
            self.install_portal_sprite(gpu, sprite, uploaded_frame);
        }
    }

    /// Set the transform for subsequent uploads without touching resident GPU textures.
    ///
    /// Use [`Self::set_anaglyph_with_gpu`] when an already-installed overlay must change in place.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }
}

impl RenderPhase for OverlayPhase {
    fn name(&self) -> &'static str {
        "post/overlay"
    }

    fn enabled(&self) -> bool {
        self.kind != OverlayKind::None
            || (self.opaque_block_uv.is_some() && self.opaque_atlas.is_some())
            || self.portal_time > 0.0
            || self.fire
            || self.border_warning.is_some()
            || self.ordinary_vignette_visible()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        let mut tint = self.kind.tint();
        // java: Water does not use the generic blue fallback when its direct texture loaded.
        // Other media retain their deliberately separate fallback behaviour.
        let textured_water = self.kind == OverlayKind::Water && self.water.is_some();
        let textured_opaque = self.opaque_block_uv.filter(|_| self.opaque_atlas.is_some());
        if textured_water {
            tint = None;
        }

        // java: GuiIngame.renderFrame gates the whole portal overlay on `f > 0.0F`.
        let portal_alpha = (self.portal_time > 0.0).then(|| portal_alpha(self.portal_time));
        let textured_portal = portal_alpha.filter(|_| self.portal.is_some());
        let textured_fire = self.fire && self.fire_texture.is_some();
        let vignette_color = self
            .border_warning
            .or_else(|| {
                self.ordinary_vignette_visible().then_some([
                    self.prev_vignette_brightness,
                    self.prev_vignette_brightness,
                    self.prev_vignette_brightness,
                    1.0,
                ])
            })
            // java: the colour goes through `GlStateManager.color`, and
            // fixed-function GL clamps the vertex colour to [0, 1] before the texture modulate.
            // Outside a world border the warning strength `1 - distance / warningDistance`
            // exceeds 1; unclamped, the red edge was far stronger than vanilla's (pinned by `tests/x7_vignette_java_parity.rs`).
            .map(|color| color.map(|channel| channel.clamp(0.0, 1.0)));
        let textured_vignette = vignette_color.filter(|_| self.vignette.is_some());
        if let Some(alpha) = portal_alpha {
            if textured_portal.is_none() {
                // Degraded path: no `textures/blocks/portal.png` loaded, so the flat purple
                // stand-in stands in — with vanilla's own alpha ramp, at least.
                if let Some((color, _)) = OverlayKind::Portal.tint() {
                    tint = Some((color, alpha));
                }
            }
        }

        if tint.is_none()
            && textured_opaque.is_none()
            && !textured_water
            && textured_portal.is_none()
            && !textured_fire
            && textured_vignette.is_none()
        {
            return;
        }

        if textured_portal.is_some() {
            self.upload_portal_frame(gpu);
        }
        let fire_vertices = if textured_fire {
            self.upload_fire_frame(gpu);
            // The Fire Height mod replaces the alpha and moves the quads. `None`
            // is vanilla's exact path.
            let alpha = self
                .fire_style
                .map_or(FIRST_PERSON_FIRE_ALPHA, |style| style.alpha);
            gpu.queue.write_buffer(
                &self.fire_buffer,
                0,
                bytemuck::bytes_of(&FireUniform {
                    color: [1.0, 1.0, 1.0, alpha],
                }),
            );
            match self.fire_style {
                None => first_person_fire_vertices(cam),
                Some(style) => first_person_fire_vertices_offset(cam, style.height_offset),
            }
        } else {
            Vec::new()
        };
        if !fire_vertices.is_empty() {
            ensure_fire_vertex_buffer(
                gpu,
                &mut self.fire_vertex_buffer,
                &mut self.fire_vertex_capacity,
                &fire_vertices,
            );
        }

        if let Some((color, alpha)) = tint {
            gpu.queue.write_buffer(
                &self.tint_buffer,
                0,
                bytemuck::bytes_of(&OverlayUniform {
                    color: [color.x, color.y, color.z, alpha],
                }),
            );
        }

        if let Some(alpha) = textured_portal {
            gpu.queue.write_buffer(
                &self.portal_buffer,
                0,
                bytemuck::bytes_of(&PortalUniform {
                    params: [
                        alpha,
                        gpu.config.width.max(1) as f32,
                        gpu.config.height.max(1) as f32,
                        0.0,
                    ],
                }),
            );
        }
        if let Some(color) = textured_vignette {
            self.vignette_write.write(
                gpu,
                &self.vignette_buffer,
                bytemuck::bytes_of(&VignetteUniform {
                    color,
                    viewport: [
                        gpu.config.width.max(1) as f32,
                        gpu.config.height.max(1) as f32,
                        0.0,
                        0.0,
                    ],
                }),
            );
        }
        if textured_water {
            gpu.queue.write_buffer(
                &self.water_buffer,
                0,
                bytemuck::bytes_of(&WaterUniform {
                    camera: [
                        water_brightness_value(self.water_brightness),
                        cam.yaw,
                        cam.pitch,
                        0.0,
                    ],
                    viewport: [
                        gpu.config.width.max(1) as f32,
                        gpu.config.height.max(1) as f32,
                        0.0,
                        0.0,
                    ],
                }),
            );
        }
        // java: `ItemRenderer.renderOverlays` draws the suffocation quad
        // inside `renderHand`, before the water and fire quads and long before
        // `GuiIngame.renderVignette`. Being opaque, it must precede every draw of the pass below
        // (the vignette especially, which vanilla darkens it with), so it gets its own pass:
        // depth-tested against the hand when the driver guarantees a hand-only depth buffer.
        if let (Some(uv), Some(atlas)) = (textured_opaque, &self.opaque_atlas) {
            let vertices = opaque_block_vertices(cam, uv);
            gpu.queue.write_buffer(
                &self.opaque_vertex_buffer,
                0,
                bytemuck::cast_slice(&vertices),
            );
            let (depth, pipeline) = if self.opaque_block_hand_depth {
                (DepthAttachment::Shared, &self.opaque_depth_pipeline)
            } else {
                (DepthAttachment::None, &self.opaque_pipeline)
            };
            let mut pass = gpu.begin_pass(enc, view, "post/overlay opaque block", depth);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &atlas.bind_group, &[]);
            pass.set_vertex_buffer(0, self.opaque_vertex_buffer.slice(..));
            pass.draw(0..vertices.len() as u32, 0..1);
        }
        if tint.is_none()
            && !textured_water
            && textured_portal.is_none()
            && fire_vertices.is_empty()
            && textured_vignette.is_none()
        {
            return;
        }

        let mut pass = gpu.begin_pass(enc, view, "post/overlay", DepthAttachment::None);
        // java draw order: `EntityRenderer.renderHand` →
        // `ItemRenderer.renderOverlays` (the suffocation quad, recorded above in its own pass,
        // then water, then fire) runs inside the world render; `GuiIngame.renderGameOverlay` runs
        // after it and begins with `renderVignette`, then `renderPortal`. So the vignette darkens
        // the water and fire quads too, and the portal swirl goes over the vignette. The flat tint
        // stands in for one of two things: the water texture (renderOverlays stage, before the
        // vignette) or the portal texture (GUI stage, after it).
        let tint_is_portal = self.kind == OverlayKind::Portal
            || (portal_alpha.is_some() && textured_portal.is_none());
        if tint.is_some() && !tint_is_portal {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.tint_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if textured_water {
            if let Some(water) = &self.water {
                pass.set_pipeline(&self.water_pipeline);
                pass.set_bind_group(0, &water.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        // java: ItemRenderer.renderOverlays draws water first and fire immediately after it;
        // the portal HUD effect is a later pass in GuiIngame.renderPortal.
        if !fire_vertices.is_empty() {
            pass.set_pipeline(&self.fire_pipeline);
            if let Some(fire) = &self.fire_texture {
                pass.set_bind_group(0, &fire.bind_group, &[]);
                pass.set_vertex_buffer(0, self.fire_vertex_buffer.slice(..));
                pass.draw(0..fire_vertices.len() as u32, 0..1);
            }
        }
        // java: `renderVignette` runs at the beginning of `GuiIngame.renderGameOverlay`, after
        // the world (and so after renderOverlays' water and fire) and before the later portal HUD
        // effect. Its special blend leaves the red scene channel intact while suppressing
        // green/blue according to the cyan warning colour.
        if textured_vignette.is_some() {
            if let Some(vignette) = &self.vignette {
                pass.set_pipeline(&self.vignette_pipeline);
                pass.set_bind_group(0, &vignette.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        if tint.is_some() && tint_is_portal {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.tint_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // java: renderPortal runs inside GuiIngame, i.e. after the immersion tint that
        // ItemRenderer.renderOverlays draws and before the rest of the HUD.
        if textured_portal.is_some() {
            if let Some(portal) = &self.portal {
                pass.set_pipeline(&self.portal_pipeline);
                pass.set_bind_group(0, &portal.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headless(width: u32, height: u32) -> Option<Gpu> {
        match Gpu::headless(width, height, crate::context::GpuOptions::default()) {
            Ok(gpu) => Some(gpu),
            Err(crate::RenderError::NoAdapter) => {
                eprintln!("skip: no wgpu adapter in this environment");
                None
            }
            Err(error) => panic!("headless context creation failed: {error}"),
        }
    }

    #[test]
    fn overlay_upload_mixes_colours_once_and_preserves_alpha() {
        let source = [
            255, 0, 0, 7, // red, translucent
            0, 255, 0, 19, // green
            0, 0, 255, 31, // blue
            128, 128, 128, 64,
        ];
        assert_eq!(overlay_upload_rgba(&source, false), source);
        assert_eq!(
            overlay_upload_rgba(&source, true),
            [76, 76, 76, 7, 178, 150, 0, 19, 0, 28, 178, 31, 128, 128, 128, 64,],
            "Java's integer anaglyph transform mixes RGB only"
        );
    }

    #[test]
    fn overlay_animation_rects_validate_the_target_mip() {
        let valid = mc_assets::AtlasAnimationUpload {
            mip_level: 1,
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            rgba: vec![0xAB; 8],
        };
        assert!(atlas_animation_rect_valid(&valid, 4, 4, 2));

        let mut out_of_bounds = valid.clone();
        out_of_bounds.x = 2;
        assert!(!atlas_animation_rect_valid(&out_of_bounds, 4, 4, 2));
        let mut wrong_level = valid.clone();
        wrong_level.mip_level = 2;
        assert!(!atlas_animation_rect_valid(&wrong_level, 4, 4, 2));
        let mut wrong_payload = valid;
        wrong_payload.rgba.pop();
        assert!(!atlas_animation_rect_valid(&wrong_payload, 4, 4, 2));
    }

    #[test]
    fn overlay_animated_frames_are_converted_at_upload_time() {
        let sprite = mc_assets::Sprite {
            location: mc_assets::ResourceLocation::minecraft("textures/blocks/portal"),
            width: 1,
            height: 1,
            uv: mc_assets::sprite::UvRect::FULL,
            frames: vec![
                mc_assets::sprite::SpriteFrame {
                    rgba: vec![255, 0, 0, 7],
                    duration_ticks: Some(2),
                },
                mc_assets::sprite::SpriteFrame {
                    rgba: vec![0, 255, 0, 19],
                    duration_ticks: Some(1),
                },
            ],
            default_duration_ticks: 1,
            interpolate: false,
            texture_metadata: Default::default(),
        };
        let frame = frame_at_tick(&sprite, 2);
        assert_eq!(frame, 1);
        assert_eq!(
            overlay_upload_rgba(&sprite.frames[frame].rgba, true),
            [178, 150, 0, 19]
        );
        assert_eq!(
            sprite.frames[frame].rgba,
            [0, 255, 0, 19],
            "animation source remains raw after conversion"
        );
    }

    #[test]
    fn block_atlas_mips_and_dirty_frames_survive_live_toggle_without_double_conversion() {
        let Some(gpu) = headless(16, 16) else {
            return;
        };
        let base = [255u8, 0, 0, 7].repeat(4 * 4);
        let mip = [0u8, 0, 255, 19].repeat(2 * 2);
        let atlas = mc_assets::Atlas {
            texture: base.clone(),
            mipmaps: vec![mip.clone()],
            w: 4,
            h: 4,
        };
        let mut phase = OverlayPhase::new(&gpu);
        phase.set_block_atlas(&gpu, &atlas);
        let animated = mc_assets::AtlasAnimationUpload {
            mip_level: 1,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            rgba: vec![0, 255, 0, 23],
        };
        assert!(phase.upload_block_atlas_animation(&gpu, &animated));
        let expected_mip = {
            let mut expected = mip.clone();
            expected[..4].copy_from_slice(&animated.rgba);
            expected
        };
        assert_eq!(
            phase
                .opaque_atlas
                .as_ref()
                .expect("opaque atlas")
                .source_levels
                .borrow()
                .as_slice(),
            &[base.clone(), expected_mip.clone()]
        );

        phase.set_anaglyph_with_gpu(&gpu, true);
        assert!(phase.anaglyph_enabled());
        assert_eq!(
            phase
                .opaque_atlas
                .as_ref()
                .expect("converted opaque atlas")
                .source_levels
                .borrow()
                .as_slice(),
            &[base.clone(), expected_mip.clone()],
            "enabling anaglyph must retain raw atlas and dirty-mip bytes"
        );
        phase.set_anaglyph_with_gpu(&gpu, false);
        assert!(!phase.anaglyph_enabled());
        assert_eq!(
            phase
                .opaque_atlas
                .as_ref()
                .expect("restored opaque atlas")
                .source_levels
                .borrow()
                .as_slice(),
            &[base, expected_mip],
            "disabling anaglyph must restore raw coloured bytes"
        );
    }

    #[test]
    fn tint_colours_match_their_medium() {
        assert!(OverlayKind::None.tint().is_none());
        let (water, _) = OverlayKind::Water.tint().unwrap();
        assert!(water.z > water.x, "{water:?}");
        let (portal, alpha) = OverlayKind::Portal.tint().unwrap();
        assert!(portal.z > portal.x, "{portal:?}");
        assert_eq!(alpha, 0.5);
    }

    /// The quad's four vertices land exactly where the real
    /// `EntityRenderer.renderHand` -> `renderBlockInHand` put them (javaref/SuffocationRef.java,
    /// 1920x1080, hand FOV 70, far plane 2 x 128): window (-582.3996, 2082.3996) to
    /// (2502.3996, -1002.3996), depth 0.9001758, U mirrored.
    #[test]
    fn opaque_block_quad_is_javas_hand_projected_quad() {
        let uv = mc_assets::sprite::UvRect {
            min_u: 0.25,
            max_u: 0.5,
            min_v: 0.625,
            max_v: 0.875,
        };
        let mut cam = Camera::default();
        cam.set_viewport(1920, 1080);
        cam.far = 128.0;
        let vertices = opaque_block_vertices(&cam, uv);
        let window = |v: &FireVertex| {
            let [x, y, z, w] = v.clip;
            [
                (f64::from(x / w) * 0.5 + 0.5) * 1920.0,
                (1.0 - (f64::from(y / w) * 0.5 + 0.5)) * 1080.0,
                f64::from(z / w),
            ]
        };
        // Java's vertex order: (-1,-1) (1,-1) (1,1) (-1,1); ours are triangles a b c / a c d.
        let java = [
            ([-582.3996, 2082.3996], [0.5, 0.875]),
            ([2502.3996, 2082.3996], [0.25, 0.875]),
            ([2502.3996, -1002.3996], [0.25, 0.625]),
            ([-582.3996, -1002.3996], [0.5, 0.625]),
        ];
        for (index, (xy, expected_uv)) in [0, 1, 2, 5].into_iter().zip(java) {
            let [x, y, depth] = window(&vertices[index]);
            assert!(
                (x - xy[0]).abs() < 0.01 && (y - xy[1]).abs() < 0.01,
                "vertex {index}: ({x}, {y}) vs Java {xy:?}"
            );
            assert!(
                (depth - 0.900_175_8).abs() < 1e-6,
                "vertex {index}: depth {depth}"
            );
            assert_eq!(vertices[index].uv, expected_uv, "vertex {index}");
        }
        assert_eq!(vertices[3], vertices[0]);
        assert_eq!(vertices[4], vertices[2]);
        assert_eq!(OPAQUE_BLOCK_RGB, 0.1);
        let shader = include_str!("opaque.wgsl");
        assert!(shader.contains("texel.rgb * opaque.color.rgb, 1.0"));
    }

    #[test]
    fn underwater_uvs_are_the_java_four_tile_quad_with_eye_rotation_offsets() {
        // java's bottom-left `(-1,-1)` gets `(4-yaw/64, 4+pitch/64)`; converting that
        // OpenGL corner to our top-left screen coordinates yields this four-corner oracle.
        assert_eq!(underwater_uv([0.0, 0.0], 64.0, -32.0), [3.0, -0.5]);
        assert_eq!(underwater_uv([1.0, 0.0], 64.0, -32.0), [-1.0, -0.5]);
        assert_eq!(underwater_uv([0.0, 1.0], 64.0, -32.0), [3.0, 3.5]);
        assert_eq!(underwater_uv([1.0, 1.0], 64.0, -32.0), [-1.0, 3.5]);
    }

    #[test]
    fn fire_copies_keep_java_offsets_and_mirrored_uvs() {
        let left = first_person_fire_quad_matrix(0).transform_point3(Vec3::ZERO);
        let right = first_person_fire_quad_matrix(1).transform_point3(Vec3::ZERO);
        assert!((left.x - 0.24).abs() < 1e-6);
        assert!((right.x + 0.24).abs() < 1e-6);
        assert!((left.y + 0.3).abs() < 1e-6);
        assert!((right.y + 0.3).abs() < 1e-6);

        let verts = first_person_fire_vertices(&Camera::default());
        assert_eq!(
            verts.len(),
            12,
            "two GL_QUADS converted to two triangle fans"
        );
        assert_eq!(verts[0].uv, [1.0, 1.0]);
        assert_eq!(verts[1].uv, [0.0, 1.0]);
        assert_eq!(verts[2].uv, [0.0, 0.0]);
        assert_eq!(verts[6].uv, [1.0, 1.0]);
        assert_ne!(
            verts[0].clip, verts[6].clip,
            "the ±10° copies must not collapse"
        );
    }

    #[test]
    fn water_brightness_control_is_bounded_and_has_a_nonfinite_fallback() {
        // This does not construct a GPU phase; it locks the CPU-side producer contract used by
        // the live setter once the driver supplies the local eye-block brightness.
        assert_eq!(water_brightness_value(-1.0), 0.0);
        assert_eq!(water_brightness_value(0.375), 0.375);
        assert_eq!(water_brightness_value(2.0), 1.0);
        assert_eq!(water_brightness_value(f32::NAN), 1.0);
    }

    #[test]
    fn fire_blend_replaces_destination_alpha_like_java() {
        let blend = fire_blend_state();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn border_vignette_uses_java_colour_multiply_and_live_warning_slot() {
        let blend = vignette_blend_state();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::Zero);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrc);
        assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::Zero);

        // The warning branch is intentionally a full RGBA colour, not a scalar alpha: under
        // GL_ZERO/GL_ONE_MINUS_SRC_COLOR this leaves the red scene channel untouched and dims
        // green/blue. A scalar-only implementation would lose this Java-visible distinction.
        let warning = Some([0.0, 0.625, 0.625, 1.0]);
        assert_eq!(warning, Some([0.0, 0.625, 0.625, 1.0]));
        assert!(include_str!("vignette.wgsl").contains("textureSample"));
    }

    #[test]
    fn ordinary_vignette_eases_one_percent_per_rendered_fancy_frame() {
        let first = advance_vignette_brightness(1.0, 1.0);
        let second = advance_vignette_brightness(first, 1.0);
        assert!((first - 0.99).abs() < 1e-6);
        assert!((second - 0.9801).abs() < 1e-6);
        assert_eq!(advance_vignette_brightness(0.25, -1.0), 0.2575);
        assert_eq!(advance_vignette_brightness(0.25, 2.0), 0.2475);

        let snap_mutant = 0.0_f32;
        let frozen_mutant = 1.0_f32;
        assert_ne!(first, snap_mutant);
        assert_ne!(first, frozen_mutant);
    }

    /// #28 / SHADER-3 / POST-2: the ordinary vignette is skipped only where its blend is an
    /// exact identity, and outdoors it gets there within a few thousand frames.
    #[test]
    fn outdoor_vignette_reaches_the_exact_identity_bound_and_is_skipped_there() {
        // `1 - src` is exactly 1.0 for every texel at the bound, and not just below the bound
        // for the round-to-nearest `b * 255 < 0.5` threshold (which would change an f32 blend).
        assert_eq!(1.0_f32 - VIGNETTE_IDENTITY_MAX, 1.0);
        assert_eq!(1.0_f32 - VIGNETTE_IDENTITY_MAX * 0.5, 1.0);
        // The round-to-nearest cut for an 8-bit blend input (`b * 255 < 0.5`) is not an identity.
        const EIGHT_BIT_ROUNDING_CUT: f32 = 1.0 / 512.0;
        const { assert!(1.0_f32 - EIGHT_BIT_ROUNDING_CUT < 1.0) };
        const { assert!((1.0_f32 - 2.0 * VIGNETTE_IDENTITY_MAX) < 1.0) };

        let mut b = 1.0_f32;
        let mut frames = 0;
        while b > VIGNETTE_IDENTITY_MAX {
            b = advance_vignette_brightness(b, 1.0);
            frames += 1;
            assert!(
                frames < 2_000,
                "outdoor vignette must fade within ~1,730 frames"
            );
        }
        assert!(frames > 1_500, "{frames}");
        // It never reaches exactly zero (subnormal fixed point), so an `== 0.0` gate is dead.
        for _ in 0..20_000 {
            b = advance_vignette_brightness(b, 1.0);
        }
        assert!(b > 0.0 && b <= VIGNETTE_IDENTITY_MAX, "{b:e}");
        // Back indoors the value climbs above the bound on the very first frame.
        assert!(advance_vignette_brightness(b, 0.0) > VIGNETTE_IDENTITY_MAX);
    }

    #[test]
    fn the_ordinary_vignette_gate_follows_the_identity_bound_but_the_border_warning_does_not() {
        let source = include_str!("overlay.rs");
        let record = source
            .split_once("fn record(")
            .map(|(_, rest)| rest)
            .expect("overlay records");
        // The border warning takes precedence; only then the ordinary vignette (X7: the chain
        // is split over lines and ends in GL's colour clamp).
        let colour = record
            .split_once("let vignette_color = self")
            .map(|(_, rest)| rest.trim_start())
            .expect("the vignette colour");
        assert!(colour.starts_with(".border_warning"));
        assert!(colour.contains(".or_else(|| {"));
        assert!(colour.contains("channel.clamp(0.0, 1.0)"));
        assert!(record.contains("self.ordinary_vignette_visible().then_some(["));
        assert!(source.contains("|| self.ordinary_vignette_visible()\n    }"));
    }
}

#[cfg(test)]
mod atlas_batch_owner_contract_tests {
    use super::*;

    // Fixed three-level atlas and explicit byte-offset expectations below form the oracle;
    // no production rectangle validator or source-copy helper computes expected output.
    fn atlas() -> mc_assets::Atlas {
        mc_assets::Atlas {
            texture: [17, 29, 43, 7].repeat(16),
            mipmaps: vec![[53, 67, 79, 19].repeat(4), vec![83, 97, 109, 31]],
            w: 4,
            h: 4,
        }
    }

    fn records() -> Vec<mc_assets::AtlasAnimationUpload> {
        let base = mc_assets::AtlasAnimationUpload {
            mip_level: 0,
            x: 1,
            y: 1,
            width: 2,
            height: 2,
            rgba: vec![
                151, 13, 79, 23, 2, 173, 61, 37, 83, 7, 199, 47, 113, 157, 11, 59,
            ],
        };
        let mip = mc_assets::AtlasAnimationUpload {
            mip_level: 1,
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            rgba: vec![3, 211, 41, 67, 227, 17, 89, 71],
        };
        let tail = mc_assets::AtlasAnimationUpload {
            mip_level: 2,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            rgba: vec![251, 101, 5, 83],
        };
        let mut short = base.clone();
        short.rgba.pop();
        let mut right = mip.clone();
        right.x = 2;
        let mut zero_width = base.clone();
        zero_width.width = 0;
        zero_width.rgba.clear();
        let mut level = tail.clone();
        level.mip_level = 3;
        let mut overflow = tail.clone();
        overflow.x = u32::MAX;
        let mut overlap = base.clone();
        overlap.rgba.reverse();
        let mut bottom = mip.clone();
        bottom.y = 1;
        let mut zero_height = base.clone();
        zero_height.height = 0;
        zero_height.rgba.clear();
        let mut long = tail.clone();
        long.rgba.push(1);
        vec![
            base,
            short,
            mip,
            right,
            zero_width,
            tail,
            level,
            overflow,
            overlap,
            bottom,
            zero_height,
            long,
        ]
    }

    #[test]
    fn actual_owner_sink_stream_shadow_settings_and_standalone_contract() {
        let gpu = Gpu::headless(16, 16, crate::context::GpuOptions::default())
            .expect("atlas owner qualification requires a working GPU adapter");
        let atlas = atlas();
        let mut direct = OverlayPhase::new(&gpu);
        let mut captured = OverlayPhase::new(&gpu);
        direct.set_block_atlas(&gpu, &atlas);
        captured.set_block_atlas(&gpu, &atlas);
        let mut expected = vec![
            atlas.texture.clone(),
            atlas.mipmaps[0].clone(),
            atlas.mipmaps[1].clone(),
        ];
        let stream = records();
        let mut receipts = Vec::new();
        for enabled in [false, true, false] {
            direct.set_anaglyph_with_gpu(&gpu, enabled);
            captured.set_anaglyph_with_gpu(&gpu, enabled);
            assert_eq!(
                *direct
                    .opaque_atlas
                    .as_ref()
                    .expect("opaque atlas")
                    .source_levels
                    .borrow(),
                expected,
                "toggle must retain raw bytes"
            );
            assert_eq!(
                *captured
                    .opaque_atlas
                    .as_ref()
                    .expect("opaque atlas")
                    .source_levels
                    .borrow(),
                expected,
                "toggle must retain raw bytes"
            );
            for (index, upload) in stream.iter().enumerate() {
                let accepted = matches!(index, 0 | 2 | 5 | 8);
                match index {
                    0 | 8 => {
                        expected[0][20..28].copy_from_slice(&upload.rgba[..8]);
                        expected[0][36..44].copy_from_slice(&upload.rgba[8..]);
                    }
                    2 => {
                        expected[1][4..8].copy_from_slice(&upload.rgba[..4]);
                        expected[1][12..16].copy_from_slice(&upload.rgba[4..]);
                    }
                    5 => expected[2].copy_from_slice(&upload.rgba),
                    _ => {}
                }
                let before_count = receipts.len();
                let actual =
                    captured.upload_block_atlas_animation_with_sink(upload, |texture, setting| {
                        assert!(
                            std::ptr::eq(
                                texture,
                                &captured
                                    .opaque_atlas
                                    .as_ref()
                                    .expect("opaque atlas")
                                    ._texture
                            ),
                            "sink must receive this owner's texture"
                        );
                        assert_eq!(
                            setting, enabled,
                            "sink must receive the owner's current setting"
                        );
                        // This borrow also proves the mutation borrow ended before the callback.
                        assert_eq!(
                            *captured
                                .opaque_atlas
                                .as_ref()
                                .expect("opaque atlas")
                                .source_levels
                                .borrow(),
                            expected,
                            "raw shadow must be ready before transfer"
                        );
                        receipts.push((enabled, index, upload.mip_level));
                    });
                assert_eq!(actual, accepted, "capture acceptance at record {index}");
                assert_eq!(
                    direct.upload_block_atlas_animation(&gpu, upload),
                    accepted,
                    "standalone acceptance at record {index}"
                );
                assert_eq!(
                    receipts.len() - before_count,
                    usize::from(accepted),
                    "one sink call per accepted record"
                );
                assert_eq!(
                    *captured
                        .opaque_atlas
                        .as_ref()
                        .expect("opaque atlas")
                        .source_levels
                        .borrow(),
                    expected,
                    "capture raw bytes at record {index}"
                );
                assert_eq!(
                    *direct
                        .opaque_atlas
                        .as_ref()
                        .expect("opaque atlas")
                        .source_levels
                        .borrow(),
                    expected,
                    "standalone raw bytes at record {index}"
                );
            }
        }
        assert_eq!(
            receipts,
            vec![
                (false, 0, 0),
                (false, 2, 1),
                (false, 5, 2),
                (false, 8, 0),
                (true, 0, 0),
                (true, 2, 1),
                (true, 5, 2),
                (true, 8, 0),
                (false, 0, 0),
                (false, 2, 1),
                (false, 5, 2),
                (false, 8, 0),
            ]
        );
        // Exercise the actual optional-owner absence gate, not a synthetic validator.
        let absent = OverlayPhase::new(&gpu);
        assert!(
            !absent.upload_block_atlas_animation_with_sink(&stream[0], |_, _| panic!(
                "absent owner called sink"
            ))
        );
        assert!(!absent.upload_block_atlas_animation(&gpu, &stream[0]));
    }
}
