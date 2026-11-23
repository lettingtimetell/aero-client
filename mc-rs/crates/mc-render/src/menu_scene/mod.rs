//! The Aero title screen's animated background.
//!
//! A glass greenhouse built from real 1.8.9 blocks ([`greenhouse`]), meshed by the real terrain
//! mesher and drawn with its own lighting: a sun with a static shadow map, sky light, swaying
//! foliage, a procedural sky with drifting clouds, two shader waterfalls (not Minecraft water)
//! with pools and mist, volumetric light shafts, haze and bloom, all seen through a slow
//! cinematic camera drift. A second phase lays a light VHS finish over the whole composed frame,
//! UI included.
//!
//! The sky's clouds are raymarched volumetric clouds (`crate::sky::volumetric`), traced at a
//! quarter of the scene's size before the scene pass and
//! sampled by its sky shader. They are built off-thread on the first frame; the painted 2D clouds
//! show until then and cross-fade into them ([`MenuScene::set_volumetric_clouds`]).
//!
//! **Always vanilla textures.** By design, the menu scene never uses
//! their resource packs, so [`MenuScene::new`] bakes vanilla 1.8.9 privately from the jar, once
//! per process ([`prepare`]), and meshes inside `mc_assets::model::with_thread_registry` /
//! [`crate::terrain::tint::with_thread_colormaps`]. The game's registries are never touched.
//!
//! # Use
//!
//! ```no_run
//! # fn demo(gpu: &mc_render::Gpu, phases: &mut mc_render::PhaseList, jar: &std::path::Path)
//! # -> Result<(), mc_render::menu_scene::MenuSceneError> {
//! use mc_render::menu_scene::{self, MenuScene};
//! menu_scene::prewarm(jar); // at startup: bake and mesh on a background thread
//! let mut scene = MenuScene::new(gpu, jar)?; // when the title screen appears
//! phases.insert(menu_scene::BACKGROUND_ORDER, scene.background_phase());
//! phases.insert(menu_scene::FINISH_ORDER, scene.finish_phase());
//! scene.update(0.0, 1920, 1080); // every frame, before rendering
//! # Ok(()) }
//! ```
//!
//! # Cost
//!
//! The menu runs at the monitor's refresh rate, so everything here is budgeted for well under
//! 1 ms of GPU time per frame at 1080p; the static shadow map is drawn once. See the report.

mod camera;
mod greenhouse;
mod prepare;
mod render;

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

pub use prepare::PrepareStats;
pub use render::Look;

use crate::pipeline::{PhaseOrder, RenderPhase, SceneTextures};
use crate::{Camera, Gpu};

/// [`RenderPhase::name`] of [`MenuScene::background_phase`].
pub const BACKGROUND_PHASE_NAME: &str = "menu/background";
/// [`RenderPhase::name`] of [`MenuScene::finish_phase`].
pub const FINISH_PHASE_NAME: &str = "menu/finish";
/// Where the background belongs: just below the UI. Not below the world phases: the client's
/// sky and cloud phases still record on the title screen, and this phase must land on top of
/// them. It overwrites every pixel and uses its own depth buffer, so whatever ran before is
/// irrelevant (the choice, 2026-11-07).
pub const BACKGROUND_ORDER: PhaseOrder = PhaseOrder(PhaseOrder::UI.0 - 10);
/// Where the finish belongs: above the UI and the mod menu's glass.
pub const FINISH_ORDER: PhaseOrder = PhaseOrder(PhaseOrder::UI.0 + 80);

/// Why the menu scene could not be built. Keep the stand-in background when this happens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuSceneError {
    /// The vanilla jar could not be opened.
    Jar(String),
    /// The private vanilla bake failed.
    Bake(String),
}

impl std::fmt::Display for MenuSceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MenuSceneError::Jar(e) => write!(f, "menu scene: vanilla jar unavailable: {e}"),
            MenuSceneError::Bake(e) => write!(f, "menu scene: vanilla bake failed: {e}"),
        }
    }
}

impl std::error::Error for MenuSceneError {}

/// Start baking vanilla and meshing the greenhouse on a background thread, so
/// [`MenuScene::new`] is instant when the title screen appears. Later calls are no-ops.
pub fn prewarm(vanilla_jar: &Path) {
    prepare::prewarm(vanilla_jar);
}

/// Whether the background bake has finished, so [`MenuScene::new`] would not wait for it. Never
/// blocks. (switching the title screen's scene to the
/// greenhouse keeps the old scene up until this is true, instead of freezing on the bake.)
#[must_use]
pub fn is_prepared() -> bool {
    prepare::is_prepared()
}

/// The CPU-side preparation's measurements: blocks until it has finished (see [`prewarm`]).
///
/// # Errors
/// As [`MenuScene::new`].
pub fn prepare_stats(vanilla_jar: &Path) -> Result<PrepareStats, MenuSceneError> {
    prepare::prepared(vanilla_jar).map(|p| p.stats)
}

/// The title screen's background and finish. Main-thread only (the phases share it through an
/// `Rc`).
///
/// **Keep one for the life of the client** and reuse it on every title-screen visit: take its
/// phases out of the phase list when the title screen goes, put new handles back
/// ([`MenuScene::background_phase`], [`MenuScene::finish_phase`]) when it returns, and call
/// [`MenuScene::release_gpu_memory`] once the title screen has been gone a while. Measured
/// (aero-title leak follow-up, 2026-11-07): building and dropping a scene per visit freed
/// everything on wgpu's side and in Rust's heap, yet grew the GPU driver's own memory by about
/// 23 MB over the first ~14 visits before it levelled off; reused, it does not. Dropping it,
/// and the two phases, frees all of its GPU memory; the CPU-side bake and meshes stay cached.
pub struct MenuScene {
    shared: Rc<RefCell<render::Renderer>>,
}

impl MenuScene {
    /// Build the GPU side. Blocks until the vanilla bake and the greenhouse mesh are ready:
    /// instant after [`prewarm`] has finished, otherwise the bake runs here.
    ///
    /// # Errors
    /// The jar could not be read or baked. Nothing global changed.
    pub fn new(gpu: &Gpu, vanilla_jar: &Path) -> Result<Self, MenuSceneError> {
        let prepared = prepare::prepared(vanilla_jar)?;
        Ok(Self {
            shared: Rc::new(RefCell::new(render::Renderer::new(gpu, prepared))),
        })
    }

    /// The phase that draws the whole background into the frame (every pixel). Insert it at
    /// [`BACKGROUND_ORDER`]. It ignores the `Camera` the phase list hands it.
    #[must_use]
    pub fn background_phase(&self) -> Box<dyn RenderPhase> {
        Box::new(BackgroundPhase(Rc::clone(&self.shared)))
    }

    /// The phase that lays the VHS finish over the composed frame, UI included. Insert it at
    /// [`FINISH_ORDER`]. It reads the frame through [`RenderPhase::needs_scene_color`].
    #[must_use]
    pub fn finish_phase(&self) -> Box<dyn RenderPhase> {
        Box::new(FinishPhase(Rc::clone(&self.shared)))
    }

    /// Advance to `time_s` seconds since the title screen appeared. The first
    /// [`INTRO_SECONDS`] are a soft fade-in from a bright glow with a short camera glide; pass a
    /// larger time to skip it. `width`/`height` are advisory: the passes size themselves to the
    /// frame they draw into.
    pub fn update(&mut self, time_s: f64, width: u32, height: u32) {
        self.shared.borrow_mut().update(time_s, width, height);
    }

    /// The strength of the VHS finish, `0..=1` (default 1, which is already light). Lower it if
    /// the UI needs to be crisper.
    pub fn set_vhs_strength(&mut self, strength: f32) {
        self.shared.borrow_mut().vhs_strength = strength.clamp(0.0, 1.0);
    }

    /// The grading knobs in use ([`Look::default`] unless [`MenuScene::set_look`] changed them).
    #[must_use]
    pub fn look(&self) -> Look {
        self.shared.borrow().look
    }

    /// Replace the grading knobs. The defaults are the tuned look; this is for the render
    /// harness.
    pub fn set_look(&mut self, look: Look) {
        self.shared.borrow_mut().look = look;
    }

    /// Free the big GPU memory (the screen-sized targets, the shadow map, the geometry and the
    /// atlas: about 200 MB at 1080p) while keeping the scene. It is all rebuilt, and the shadow
    /// map redrawn, the next time the background draws (a few milliseconds, no shader or
    /// pipeline work). Call it when the title screen has been gone for a while; see the module
    /// docs for why the scene itself should be kept rather than dropped.
    pub fn release_gpu_memory(&mut self) {
        self.shared.borrow_mut().release();
    }

    /// Whether the big GPU memory is currently freed ([`MenuScene::release_gpu_memory`], and
    /// nothing drawn since).
    #[must_use]
    pub fn gpu_memory_released(&self) -> bool {
        self.shared.borrow().is_released()
    }

    /// Whether the sky draws volumetric clouds (`crate::sky::volumetric`; on by default). They
    /// are built off-thread the first time the background draws, and the painted clouds show
    /// until then (a 0.6 s cross-fade). Off gives back everything of theirs and keeps the painted
    /// clouds.
    pub fn set_volumetric_clouds(&mut self, enabled: bool) {
        self.shared.borrow_mut().set_volumetric_clouds(enabled);
    }

    /// Block until the volumetric clouds are built (at most `timeout`) and show them at full
    /// strength at once: for offline renders and tests. The title screen itself never waits.
    /// Returns whether they are ready.
    pub fn wait_for_clouds(&mut self, gpu: &Gpu, timeout: std::time::Duration) -> bool {
        self.shared.borrow_mut().wait_for_clouds(gpu, timeout)
    }
}

/// The length of the opening fade-in, seconds.
pub const INTRO_SECONDS: f64 = camera::INTRO_SECONDS;

struct BackgroundPhase(Rc<RefCell<render::Renderer>>);

impl RenderPhase for BackgroundPhase {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.0.borrow_mut().record_background(gpu, enc, view);
    }

    fn name(&self) -> &'static str {
        BACKGROUND_PHASE_NAME
    }
}

struct FinishPhase(Rc<RefCell<render::Renderer>>);

impl RenderPhase for FinishPhase {
    fn record(
        &mut self,
        _gpu: &Gpu,
        _enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        // Only reached if the phase list did not hand over the frame; nothing to finish then.
    }

    fn name(&self) -> &'static str {
        FINISH_PHASE_NAME
    }

    fn enabled(&self) -> bool {
        self.0.borrow().vhs_strength > 0.0
    }

    fn needs_scene_color(&self) -> bool {
        true
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        self.0.borrow_mut().record_finish(gpu, enc, view, scene);
    }
}
