//! The Glass and Shore scenes: a plant swaying behind wired,
//! frosted glass, and neon-crested waves on a night shore.
//!
//! **They are title-screen scenes now, not loading screens.**
//! Requested: "loading screens were supposed to be main menu screens, it just increases loading
//! time like this. please make it into 3 randomly selected main menus". The title screen picks
//! the greenhouse (`crate::menu_scene`), Glass or Shore (the Aero Title Screen mod's "Menu
//! scene", Random by default) and draws Glass and Shore through [`LoadingCurtain`] at opacity 1 in
//! its background slot, with a [`BackdropStyle`]; the loading screen no longer builds a scene. The
//! module keeps its name (and the curtain its) so the existing harnesses still read. The client
//! half is `mc-client/src/app/mods/nw_menu.rs` and `aero_title.rs`.
//!
//! # How it fits together
//!
//! * A **scene** ([`Scene`]) draws one finished, opaque frame of its animation into a texture
//!   the frame gives it ([`SCENE_FORMAT`], the window's size). It owns its own GPU resources and
//!   gets the vanilla textures it asked for ([`SceneKind::jar_textures`]) read straight from the
//!   1.8.9 jar ([`JarTextures`]), never the player's resource packs (like the greenhouse).
//! * The **curtain** ([`LoadingCurtain`]) owns the scene and lays its frame over the game's
//!   frame with an opacity. The title screen inserts its phase into its own background slot
//!   (`menu_scene::BACKGROUND_ORDER`, under the UI) at opacity 1. Dropping it frees every GPU
//!   resource of the scene; [`LoadingCurtain::release_gpu_memory`] frees the screen-sized part.
//! * The client (`mc-client/src/app/mods/nw_menu.rs`) picks the scene, drives its clock and
//!   parks it while the title screen is away. (`tw_loading.rs` drove it as a loading
//!   screen over everything at [`CURTAIN_ORDER`]; that loading screen is scene-less now.)

use std::cell::RefCell;
use std::rc::Rc;

use crate::pipeline::{PhaseOrder, RenderPhase};
use crate::Gpu;

mod curtain;
mod glass;
mod jar;

pub use glass::GlassLook;
pub use jar::{JarImage, JarPrefetch, JarTextures};

/// The glass scene with a given look and, for the render harness, an edited copy of
/// `glass.wgsl` (`None` uses the compiled-in shader). The client uses [`SceneKind::create`].
#[must_use]
pub fn glass_scene(
    gpu: &Gpu,
    init: &SceneInit<'_>,
    look: GlassLook,
    wgsl: Option<&str>,
) -> Box<dyn Scene> {
    Box::new(glass::GlassScene::with_shader(
        gpu,
        init,
        wgsl.unwrap_or(include_str!("glass.wgsl")),
        look,
    ))
}

pub mod shore;

/// Where the curtain goes: above everything else in the frame (the UI, the title screen's
/// layers at `UI + 40..=UI + 90`, the mod menu at `UI + 50`).
pub const CURTAIN_ORDER: PhaseOrder = PhaseOrder(PhaseOrder::UI.0 + 180);
/// Where the client's logo and progress overlay goes: just above the curtain.
pub const OVERLAY_ORDER: PhaseOrder = PhaseOrder(CURTAIN_ORDER.0 + 1);
/// [`RenderPhase::name`] of the curtain.
pub const CURTAIN_PHASE_NAME: &str = "loading/curtain";
/// The format of the texture a [`Scene`] draws into. Write display-encoded colour (the same byte
/// domain as the swap chain: sRGB-curve values, not linear), alpha ignored.
pub const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One frame's inputs for a [`Scene`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneFrame {
    /// The scene's clock, seconds since its first frame. It advances by the real frame time but
    /// at most [`MAX_STEP_S`] per frame, so a stall of the loading thread (the asset stages run
    /// on it between frames) pauses the animation instead of making it jump.
    pub time_s: f32,
    /// This frame's step of [`SceneFrame::time_s`], seconds (`0..=MAX_STEP_S`).
    pub dt_s: f32,
    /// The target's size in physical pixels: the window.
    pub width: u32,
    /// See [`SceneFrame::width`].
    pub height: u32,
    /// Loading progress, `0..=1`, smoothed (it never jumps or goes back).
    pub progress: f32,
    /// Whether the game has finished loading (the curtain may still be up for the minimum time).
    pub loaded: bool,
    /// The curtain's opacity this frame, `0..=1`: 1 while loading, falling to 0 during the fade
    /// into the title screen. The frame applies it; a scene may read it (for example to skip
    /// work that would not show).
    pub opacity: f32,
    /// Where the overlay's logo sits this frame, `[x, y, width, height]` in physical pixels
    /// (all zero when there is none), so a scene can keep that area calm or darken it a little.
    pub logo: [f32; 4],
    /// A random number fixed for the whole run (the same one [`SceneInit::seed`] had).
    pub seed: u32,
}

impl Default for SceneFrame {
    fn default() -> Self {
        SceneFrame {
            time_s: 0.0,
            dt_s: 0.0,
            width: 1,
            height: 1,
            progress: 0.0,
            loaded: false,
            opacity: 1.0,
            logo: [0.0; 4],
            seed: 0,
        }
    }
}

/// The largest step of [`SceneFrame::time_s`] per frame, seconds.
pub const MAX_STEP_S: f32 = 1.0 / 20.0;

/// What a [`Scene`] is built from.
#[derive(Debug, Clone, Copy)]
pub struct SceneInit<'a> {
    /// The vanilla textures [`SceneKind::jar_textures`] listed, read from the 1.8.9 jar. Any of
    /// them may be missing (no jar): draw something sensible anyway.
    pub textures: &'a JarTextures,
    /// A random number for this run, so a scene can vary its layout from launch to launch.
    pub seed: u32,
}

/// A scene: an animated, full-screen picture (a title-screen background; a loading screen).
///
/// Built when the title screen first shows it ([`SceneKind::create`]) and kept while the title
/// screen comes and goes; dropping it must free all of its GPU resources (own them; nothing
/// global). Main thread only.
pub trait Scene {
    /// A short stable name for logs and tests (`"glass"`, `"shore"`).
    fn name(&self) -> &'static str;

    /// Record one frame into `target`: a single-sampled [`SCENE_FORMAT`] view of
    /// `frame.width` x `frame.height` pixels. Write every pixel (clear it first): the result is
    /// the whole picture under the title screen's UI. Size-dependent resources are the scene's
    /// own to (re)build when the size changes. Keep it under about half a millisecond of GPU time
    /// at 1080p: the title screen runs at the monitor's refresh rate for as long as it is open.
    fn render(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        frame: &SceneFrame,
    );

    /// Free the window-size-dependent GPU memory (render targets and what is baked at a size),
    /// keeping the pipelines and the textures: the next [`Scene::render`] builds it again.
    /// Called while the title screen is away. The default keeps
    /// everything.
    fn release_gpu_memory(&mut self) {}
}

/// How [`LoadingCurtain`] lays its scene over the frame when it is the title screen's background.
/// The default changes nothing.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BackdropStyle {
    /// Mirror the scene left to right (the glass's bright condensation is on its left, where the
    /// title screen's logo sits).
    pub mirror: bool,
    /// How much the left of the frame is darkened, where the title screen's logo and buttons sit
    /// (the greenhouse's own scrim, the same shape), `0..=0.5`; 0 = none.
    pub scrim: f32,
}

/// The scenes of this module (the "Menu scene" option's Glass and Shore).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneKind {
    /// A Minecraft plant swaying behind wired, frosted, hammered glass.
    Glass,
    /// Neon-crested waves breaking on a Minecraft beach at night.
    Shore,
}

impl SceneKind {
    /// Every scene, in option order.
    pub const ALL: [SceneKind; 2] = [SceneKind::Glass, SceneKind::Shore];

    /// The scene's name (the same as [`Scene::name`]).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            SceneKind::Glass => "glass",
            SceneKind::Shore => "shore",
        }
    }

    /// The vanilla textures the scene reads from the jar (paths relative to
    /// `assets/minecraft/`). The client reads them on a background thread before the window
    /// opens ([`JarPrefetch`]) and hands them to [`SceneKind::create`].
    #[must_use]
    pub fn jar_textures(self) -> &'static [&'static str] {
        match self {
            SceneKind::Glass => glass::JAR_TEXTURES,
            SceneKind::Shore => shore::JAR_TEXTURES,
        }
    }

    /// Build the scene's GPU side.
    #[must_use]
    pub fn create(self, gpu: &Gpu, init: &SceneInit<'_>) -> Box<dyn Scene> {
        match self {
            SceneKind::Glass => Box::new(glass::GlassScene::new(gpu, init)),
            SceneKind::Shore => Box::new(shore::ShoreScene::new(gpu, init)),
        }
    }

    /// Pick one at random from `seed` (the "Random" choice).
    #[must_use]
    pub fn random(seed: u32) -> SceneKind {
        // A multiplicative hash, so neighbouring seeds (clock readings) do not alternate.
        let mixed = seed.wrapping_mul(0x9E37_79B9).rotate_right(13);
        SceneKind::ALL[(mixed % SceneKind::ALL.len() as u32) as usize]
    }
}

/// The loading screen's curtain: the chosen scene laid over the whole frame with an opacity.
///
/// Insert [`LoadingCurtain::phase`] at [`CURTAIN_ORDER`] (again after the phase list is rebuilt:
/// the handle keeps the scene), call [`LoadingCurtain::update`] every frame before rendering,
/// and drop the handle and the phase once the opacity has reached 0: that frees everything.
pub struct LoadingCurtain {
    shared: Rc<RefCell<curtain::Curtain>>,
}

impl std::fmt::Debug for LoadingCurtain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let curtain = self.shared.borrow();
        f.debug_struct("LoadingCurtain")
            .field("scene", &curtain.scene_name())
            .field("frame", &curtain.frame)
            .field("brightness", &curtain.brightness)
            .finish_non_exhaustive()
    }
}

impl LoadingCurtain {
    /// Build `kind` and the curtain around it. Fully opaque until [`LoadingCurtain::update`]
    /// says otherwise.
    #[must_use]
    pub fn new(gpu: &Gpu, kind: SceneKind, textures: &JarTextures, seed: u32) -> LoadingCurtain {
        let scene = kind.create(gpu, &SceneInit { textures, seed });
        LoadingCurtain::with_scene(gpu, scene, seed)
    }

    /// The curtain around a scene built elsewhere (render harnesses).
    #[must_use]
    pub fn with_scene(gpu: &Gpu, scene: Box<dyn Scene>, seed: u32) -> LoadingCurtain {
        let mut curtain = curtain::Curtain::new(gpu, scene);
        curtain.frame.seed = seed;
        LoadingCurtain {
            shared: Rc::new(RefCell::new(curtain)),
        }
    }

    /// A phase handle to insert at [`CURTAIN_ORDER`]. It is disabled while the opacity is 0.
    #[must_use]
    pub fn phase(&self) -> Box<dyn RenderPhase> {
        Box::new(curtain::CurtainPhase(Rc::clone(&self.shared)))
    }

    /// Set this frame's inputs. `frame.width`/`height` are filled in by the phase from the
    /// frame it draws into; `brightness` (`0..=1`) multiplies the scene (the client opens at 0.75
    /// and settles to 1).
    pub fn update(&mut self, frame: SceneFrame, brightness: f32) {
        let mut curtain = self.shared.borrow_mut();
        let size = (curtain.frame.width, curtain.frame.height);
        curtain.frame = SceneFrame {
            width: size.0,
            height: size.1,
            ..frame
        };
        curtain.brightness = brightness;
    }

    /// The scene's name.
    #[must_use]
    pub fn scene_name(&self) -> &'static str {
        self.shared.borrow().scene_name()
    }

    /// How many frames the scene has drawn.
    #[must_use]
    pub fn frames_drawn(&self) -> u64 {
        self.shared.borrow().frames
    }

    /// The size of the last frame drawn (physical pixels).
    #[must_use]
    pub fn last_size(&self) -> (u32, u32) {
        let curtain = self.shared.borrow();
        (curtain.frame.width, curtain.frame.height)
    }

    /// Mirror and scrim the scene (the title screen's background).
    pub fn set_style(&mut self, style: BackdropStyle) {
        self.shared.borrow_mut().style = style;
    }

    /// The style in use.
    #[must_use]
    pub fn style(&self) -> BackdropStyle {
        self.shared.borrow().style
    }

    /// Free the screen-sized GPU memory (the scene's target and its size-dependent resources)
    /// while keeping the scene, its pipelines and its textures: the title screen calls this once
    /// it has been away a while. The next frame drawn builds it again.
    pub fn release_gpu_memory(&mut self) {
        self.shared.borrow_mut().release();
    }

    /// Whether the screen-sized memory is currently freed (and nothing drawn since).
    #[must_use]
    pub fn gpu_memory_released(&self) -> bool {
        self.shared.borrow().is_released()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_curtain_and_overlay_sit_above_every_other_phase() {
        // The title screen's highest layer is UI + 90 (its crisp marks); the mod menu is UI + 50.
        assert!(CURTAIN_ORDER > PhaseOrder(PhaseOrder::UI.0 + 90));
        assert!(OVERLAY_ORDER > CURTAIN_ORDER);
    }

    #[test]
    fn random_picks_every_scene_and_is_stable_per_seed() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0..64 {
            let kind = SceneKind::random(seed);
            assert_eq!(kind, SceneKind::random(seed));
            seen.insert(kind);
        }
        assert_eq!(seen.len(), SceneKind::ALL.len());
    }

    #[test]
    fn scene_names_are_stable() {
        assert_eq!(SceneKind::Glass.name(), "glass");
        assert_eq!(SceneKind::Shore.name(), "shore");
    }
}
