//! The title screen's render phase: its own instance of the mod menu's glass phase
//! ([`ModMenuPhase`]: the Aero liquid glass, then content in physical pixels), registered under
//! its own name so the phase list, the profiler and the GPU timer can tell the two apart. It
//! also holds the title bitmap ([`super::font`]) in its renderer.
//!
//! Register it at `PhaseOrder(PhaseOrder::UI.0 + 40)`: after the UI phase (the backdrop under
//! the glass is the greenhouse plus whatever the UI phase drew), before the Shift+Tab menu's
//! phase (`UI + 50`) and the greenhouse's VHS finish (`UI + 80`). **Cost when the title screen is
//! not up: none**, exactly as the mod menu's: [`RenderPhase::enabled`] is false, so the list skips
//! it and asks for no scene copy.

use mc_assets::TextureMetadata;
use mc_render::{Camera, Gpu, RenderPhase, SceneTextures};

use super::font;
use crate::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use crate::screens::modmenu::text;
use crate::widget::{DrawList, UiRenderer};

/// The Aero title screen's phase.
#[derive(Debug)]
pub struct AeroTitlePhase {
    inner: ModMenuPhase,
}

impl AeroTitlePhase {
    /// Build the pipelines (nothing runs per frame until the title screen submits).
    #[must_use]
    pub fn new(gpu: &Gpu) -> AeroTitlePhase {
        AeroTitlePhase {
            inner: ModMenuPhase::new(gpu),
        }
    }

    /// Whether the title screen is on screen.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    /// An empty list to build the next frame into (it reuses an old list's allocations).
    pub fn take_spare_list(&mut self) -> DrawList {
        self.inner.take_spare_list()
    }

    /// Hand over this frame's content (physical pixels) and glass.
    pub fn submit(&mut self, list: DrawList, glass: GlassFrame) {
        self.inner.submit(list, glass);
    }

    /// The title screen is gone: stop drawing and stop asking for the scene copy.
    pub fn deactivate(&mut self) {
        self.inner.deactivate();
    }

    /// The submitted glass (tests).
    #[must_use]
    pub fn glass(&self) -> &GlassFrame {
        self.inner.glass()
    }

    /// The renderer that draws the content (texture uploads, the Minecraft-font fallback).
    pub fn renderer(&mut self) -> &mut UiRenderer {
        &mut self.inner.renderer
    }

    /// Upload the Segoe UI glyph atlas of the button labels ([`text::MenuText`]).
    pub fn upload_text_atlas(&mut self, gpu: &Gpu, rgba: &[u8]) {
        self.inner.upload_atlas(gpu, rgba, text::ATLAS_SIZE);
    }

    /// Upload the title bitmap ([`font::TitleFont::take_upload`]).
    pub fn upload_title(&mut self, gpu: &Gpu, rgba: &[u8], width: u32, height: u32) {
        let options = TextureMetadata {
            blur: true,
            clamp: true,
            ..TextureMetadata::default()
        };
        self.inner.renderer.upload_with_options(
            gpu,
            font::atlas_key(),
            rgba,
            width,
            height,
            &options,
        );
    }
}

impl RenderPhase for AeroTitlePhase {
    fn name(&self) -> &'static str {
        "aero_title"
    }

    fn enabled(&self) -> bool {
        self.inner.enabled()
    }

    fn needs_scene_color(&self) -> bool {
        self.inner.needs_scene_color()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.inner.record(gpu, enc, view, cam);
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        self.inner.record_scene(gpu, enc, view, cam, scene);
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        self.inner.resize(gpu, width, height);
    }
}
