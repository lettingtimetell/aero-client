//! [`ShaderCore`]: the state every pack phase shares.
//!
//! One per client, created by the glue (`app/mods/tw_shaders.rs`) and cloned into every pack
//! phase (clones share one state; it is an `Rc`). It holds which pack is on, this frame's
//! [`PackFrame`], the running pack's [`CoreConfig`], and the GPU resources the core features
//! share: the frame uniform buffer and the shadow map. All of them are built on first use while a
//! pack runs and dropped when no pack runs ([`ShaderCore::release_unused`]), so with every pack
//! off the core holds no GPU memory and its phases report `enabled() == false`.

use std::cell::RefCell;
use std::rc::Rc;

use super::config::CoreConfig;
use super::frame::{LightColors, PackFrame, PackFrameUniform, PackId};
use super::shadow::{ShadowMap, ShadowView};
use crate::Gpu;

/// The shared shader-pack state. Cheap to clone (an `Rc`); every clone sees the same state.
#[derive(Clone, Default)]
pub struct ShaderCore {
    inner: Rc<RefCell<CoreState>>,
}

impl std::fmt::Debug for ShaderCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.inner.borrow();
        f.debug_struct("ShaderCore")
            .field("active", &state.active)
            .field("running", &state.running())
            .field("config", &state.config)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct CoreState {
    /// The pack switched on in the settings (exclusivity already applied by the glue).
    active: Option<PackId>,
    /// The frame is single-sampled. Packs switch themselves off under MSAA, like every post
    /// phase.
    single_sampled: bool,
    frame: PackFrame,
    light: Option<LightColors>,
    config: CoreConfig,
    /// Bumped by every [`ShaderCore::set_config`] that changes the config.
    config_generation: u64,
    gpu: Option<CoreGpu>,
    shadow: ShadowMap,
}

impl CoreState {
    fn running(&self) -> Option<PackId> {
        self.active
            .filter(|_| self.single_sampled && self.frame.in_world)
    }
}

/// The frame uniform buffer.
struct CoreGpu {
    frame_buffer: wgpu::Buffer,
    uploaded: Option<PackFrameUniform>,
}

impl ShaderCore {
    /// An idle core: no pack, nothing allocated.
    #[must_use]
    pub fn new() -> ShaderCore {
        ShaderCore::default()
    }

    /// Whether two handles share one state.
    #[must_use]
    pub fn same(&self, other: &ShaderCore) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// Glue: which pack is switched on (after exclusivity). `None` turns the core off.
    ///
    /// Any change of pack resets the core features to [`CoreConfig::OFF`]: a pack gets only the
    /// features its own glue asks for afterwards ([`Self::set_config`]), never the previous
    /// pack's shadows, bloom or tone map.
    pub fn set_active(&self, pack: Option<PackId>) {
        let mut state = self.inner.borrow_mut();
        if state.active != pack {
            state.active = pack;
            if state.config != CoreConfig::OFF {
                state.config = CoreConfig::OFF;
                state.config_generation += 1;
            }
        }
    }

    /// The pack switched on in the settings, running or not.
    #[must_use]
    pub fn active(&self) -> Option<PackId> {
        self.inner.borrow().active
    }

    /// The pack that renders this frame: switched on, a world on screen, no MSAA. Pack phases
    /// gate `enabled()` on this.
    #[must_use]
    pub fn running(&self) -> Option<PackId> {
        self.inner.borrow().running()
    }

    /// Whether `pack` renders this frame.
    #[must_use]
    pub fn is_running(&self, pack: PackId) -> bool {
        self.running() == Some(pack)
    }

    /// Whether any pack renders this frame.
    #[must_use]
    pub fn any_running(&self) -> bool {
        self.running().is_some()
    }

    /// Glue, once per frame: this frame's inputs. Uploads the frame uniform while a pack runs
    /// (built on first use) and computes the light colours from the running config.
    pub fn set_frame(&self, gpu: &Gpu, frame: PackFrame) {
        let mut state = self.inner.borrow_mut();
        state.single_sampled = gpu.sample_count() <= 1;
        state.frame = frame;
        if state.running().is_none() {
            state.light = None;
            drop(state);
            self.release_unused();
            return;
        }
        let light = match state.config.lighting {
            Some(lighting) => LightColors::compute(&frame, lighting.warmth, lighting.night_blue),
            None => LightColors::compute(&frame, 1.0, 1.0),
        };
        state.light = Some(light);
        let uniform = frame.uniform(&light);
        let core_gpu = state.gpu.get_or_insert_with(|| CoreGpu {
            frame_buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shaderpack frame"),
                size: std::mem::size_of::<PackFrameUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            uploaded: None,
        });
        if core_gpu.uploaded != Some(uniform) {
            gpu.write_buffer_staged(&core_gpu.frame_buffer, 0, bytemuck::bytes_of(&uniform));
            core_gpu.uploaded = Some(uniform);
        }
        let shadows = state.config.shadows;
        let tilt = state.config.sun_tilt;
        let frame = state.frame;
        state.shadow.update(gpu, shadows, tilt, &frame);
    }

    /// This frame's inputs (a copy).
    #[must_use]
    pub fn frame(&self) -> PackFrame {
        self.inner.borrow().frame
    }

    /// This frame's direct and ambient light colours, while a pack runs.
    #[must_use]
    pub fn light_colors(&self) -> Option<LightColors> {
        self.inner.borrow().light
    }

    /// The frame uniform buffer (WGSL `PackFrame`, [`super::COMMON_WGSL`]), once a pack has run.
    /// Bind it as `var<uniform>`; it is rewritten each frame, never reallocated while a pack
    /// stays on.
    #[must_use]
    pub fn frame_buffer(&self) -> Option<wgpu::Buffer> {
        self.inner
            .borrow()
            .gpu
            .as_ref()
            .map(|g| g.frame_buffer.clone())
    }

    /// The running pack's glue: its core features. Sanitized on the way in; a changed config
    /// bumps [`Self::config_generation`].
    pub fn set_config(&self, config: CoreConfig) {
        let config = config.sanitized();
        let mut state = self.inner.borrow_mut();
        if state.config != config {
            state.config = config;
            state.config_generation += 1;
        }
    }

    /// The core features in effect (all off while no pack is switched on).
    #[must_use]
    pub fn config(&self) -> CoreConfig {
        self.inner.borrow().config
    }

    /// The core features in effect this frame: [`Self::config`] while a pack runs, else
    /// [`CoreConfig::OFF`].
    #[must_use]
    pub fn running_config(&self) -> CoreConfig {
        let state = self.inner.borrow();
        if state.running().is_some() {
            state.config
        } else {
            CoreConfig::OFF
        }
    }

    /// Bumped whenever the config changes.
    #[must_use]
    pub fn config_generation(&self) -> u64 {
        self.inner.borrow().config_generation
    }

    /// The shadow map and its matrices while shadows run (for a pack that wants "in shadow",
    /// e.g. caustics). `None` while shadows are off.
    #[must_use]
    pub fn shadow(&self) -> Option<ShadowView> {
        let state = self.inner.borrow();
        if state.running().is_none() || state.config.shadows.is_none() {
            return None;
        }
        state.shadow.view()
    }

    /// Run `f` with the shadow map state (the terrain variant renders and samples it).
    pub(crate) fn with_shadow<R>(&self, f: impl FnOnce(&mut ShadowMap) -> R) -> R {
        f(&mut self.inner.borrow_mut().shadow)
    }

    /// Drop every GPU resource no running feature needs. Called by [`Self::set_frame`] when no
    /// pack runs; harmless to call any time.
    pub fn release_unused(&self) {
        let mut state = self.inner.borrow_mut();
        if state.running().is_none() {
            state.gpu = None;
            state.shadow.release();
        } else if state.config.shadows.is_none() {
            state.shadow.release();
        }
    }

    /// Whether the core holds any GPU resource (tests: off must hold none).
    #[must_use]
    pub fn allocated(&self) -> bool {
        let state = self.inner.borrow();
        state.gpu.is_some() || state.shadow.allocated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaderpack::config::{ShadowSettings, TonemapSettings};

    #[test]
    fn a_pack_runs_only_switched_on_in_a_world_without_msaa() {
        let core = ShaderCore::new();
        assert_eq!(core.running(), None);
        core.set_active(Some(PackId::Classic));
        // No frame yet: not in a world.
        assert_eq!(core.running(), None);
        {
            let mut state = core.inner.borrow_mut();
            state.single_sampled = true;
            state.frame.in_world = true;
        }
        assert_eq!(core.running(), Some(PackId::Classic));
        assert!(core.is_running(PackId::Classic));
        assert!(!core.is_running(PackId::Dreamy));
        core.inner.borrow_mut().single_sampled = false;
        assert_eq!(core.running(), None, "MSAA switches packs off");
    }

    #[test]
    fn switching_packs_never_carries_the_previous_packs_features() {
        let core = ShaderCore::new();
        core.set_active(Some(PackId::Classic));
        core.set_config(CoreConfig {
            shadows: Some(ShadowSettings::default()),
            tonemap: Some(TonemapSettings::default()),
            ..CoreConfig::OFF
        });
        let generation = core.config_generation();
        // Classic -> Underwater: nothing of Classic's survives until Underwater asks.
        core.set_active(Some(PackId::Underwater));
        assert_eq!(core.config(), CoreConfig::OFF);
        assert!(core.config_generation() > generation);
        // Re-selecting the same pack keeps what it set.
        core.set_config(CoreConfig {
            tonemap: Some(TonemapSettings::default()),
            ..CoreConfig::OFF
        });
        core.set_active(Some(PackId::Underwater));
        assert!(core.config().tonemap.is_some());
    }

    #[test]
    fn switching_every_pack_off_clears_the_config() {
        let core = ShaderCore::new();
        core.set_active(Some(PackId::Dreamy));
        core.set_config(CoreConfig {
            tonemap: Some(TonemapSettings::default()),
            shadows: Some(ShadowSettings::default()),
            ..CoreConfig::OFF
        });
        let generation = core.config_generation();
        assert!(core.config().tonemap.is_some());
        core.set_active(None);
        assert_eq!(core.config(), CoreConfig::OFF);
        assert!(core.config_generation() > generation);
        assert!(!core.allocated());
    }
}
