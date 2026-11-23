// NOTE: `#![forbid(unsafe_code)]` is deliberately absent here, and only here. The no-`unsafe` rule
// exempts `mc-render` because wgpu buffer mapping and vertex-pod casts need `unsafe`. Every
// other crate in the workspace forbids it; keep it that way.
//
// The crate writes no `unsafe` by hand. The only `unsafe impl`s in the crate come from
// `#[derive(bytemuck::Pod, bytemuck::Zeroable)]`, whose safety conditions the derive checks at
// compile time. If hand-written `unsafe` is ever needed, comment why at the site.
#![warn(missing_docs)]

//! The wgpu renderer.
//!
//! Its modules:
//!
//! | Module | What |
//! |---|---|
//! | [`context`], [`frame`], [`input`], [`pipeline`], [`settings`], [`shell`] | shell, frame loop, input |
//! | [`terrain`]`/**` | mesher and chunk renderer |
//! | [`entity`]`/**` | models, animation, rendering |
//! | [`particle`]`/**` | particles |
//! | [`sky`]`/**`, [`post`]`/**` | sky, weather, post effects |
//!
//! By design, the fixed-function GL in `client/renderer/` is **rewritten** into a retained
//! pipeline, not transliterated: there is no `GlStateManager` equivalent, and render state lives
//! in the `wgpu::RenderPipeline` each phase owns.
//!
//! # The seam
//!
//! Everything a downstream section needs is three types:
//!
//! * [`Gpu`] — device, queue, surface, swap-chain config, shared depth buffer, MSAA sample
//!   count, and the `@group(0)` camera bind group.
//! * [`RenderPhase`] — one ordered step of the frame. Implement it, box it, and register it on
//!   the [`PhaseList`] at a [`PhaseOrder`]; it is recorded once per frame in order.
//! * [`Camera`] — view and projection, with vanilla FOV semantics and **camera-relative**
//!   geometry (see the [`Camera`] docs before writing a vertex shader).
//!
//! Read [`pipeline`]'s module docs for the full phase contract.
//!
//! # A minimal client
//!
//! ```no_run
//! use mc_render::{GameSettings, Shell, ShellApp, ShellConfig};
//! use std::time::Duration;
//!
//! struct Demo;
//! impl ShellApp for Demo {}
//!
//! # fn main() -> Result<(), mc_render::RenderError> {
//! let config = ShellConfig {
//!     // Always bound the loop in automation; see the `shell` module docs.
//!     max_duration: Some(Duration::from_secs(5)),
//!     ..ShellConfig::default()
//! };
//! let summary = Shell::run(config, GameSettings::default(), &mut Demo)?;
//! println!("{} frames at {:.1} fps", summary.frames, summary.average_fps);
//! # Ok(())
//! # }
//! ```

pub mod atlas_transfer;
// The frame pacing of a window in the background.
mod background_wait;
// R1-8: the per-frame upload ring (`Gpu::write_buffer_staged`).
pub mod context;
// The safe DLL search order, set first thing in the client's `main`.
pub mod dll_search;
pub mod upload;
// #14 step 1b: the frame's finish/submit on the present thread (`Gpu::queue` is a `GatedQueue`).
pub mod entity;
// Video export (offscreen capture, NV12, preview).
pub mod export;
pub mod frame;
pub mod frame_pipe;
// Device-lost record, panic-free buffers.
pub mod gpu_health;
pub mod input;
// The animated loading screens.
pub mod loading_scene;
// The Aero title screen's 3D background.
pub mod menu_scene;
pub mod particle;
pub mod pipeline;
pub mod post;
pub mod settings;
// Shader packs.
pub mod shaderpack;
pub mod shell;
pub mod sky;
pub mod terrain;
// Within-run A/B interleave of its switches (measurement).
pub mod w18_ob_ab;
// Render-side memory (switches, measurement aid).
pub mod w21_mr;
// The allocator's freed pages at transitions.
pub mod w21_mb;
pub mod window_icon;
pub mod world_directions;

pub use context::{AcquiredFrame, DepthAttachment, Gpu, GpuOptions, DEPTH_FORMAT, HEADLESS_FORMAT};
pub use entity::tile::BeaconPhase;
pub use entity::tilepass::{TileModels, TilePhase, TileVertex};
pub use frame::{
    Camera, FrameStats, MouseFilter, SmoothCamera, Timer, MAX_CATCH_UP_TICKS, TICKS_PER_SECOND,
};
pub use input::{
    CursorMode, Input, InputState, KeyBinding, KeyBindings, MouseEventKind, TimedMouseEvent,
    MOUSE_EVENT_MAX_AGE_MS,
};
pub use pipeline::{CameraUniform, PhaseList, PhaseOrder, RenderPhase, SceneTextures};
pub use settings::GameSettings;
pub use shell::{DebugFlyCamera, FrameCtx, RunSummary, Shell, ShellApp, ShellConfig, TickCtx};

use thiserror::Error;

/// Stable, presentation-ready facts about the adapter selected for a render context.
///
/// Keeping these as strings lets crash reporting retain them after the GPU and window have been
/// dropped, and lets CPU-only tests inject the same data without constructing a `wgpu::Adapter`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterFacts {
    /// Adapter/renderer name reported by the backend.
    pub renderer: String,
    /// PCI vendor identifier when the backend exposes one.
    pub vendor: String,
    /// Graphics API/backend in use.
    pub backend: String,
    /// Driver name and additional driver information.
    pub driver: String,
    /// PCI device identifier and device class.
    pub device: String,
}

impl AdapterFacts {
    /// Snapshot a wgpu adapter description before the context can fail or be dropped.
    #[must_use]
    pub fn from_info(info: &wgpu::AdapterInfo) -> AdapterFacts {
        let driver = match (info.driver.is_empty(), info.driver_info.is_empty()) {
            (true, true) => "(unavailable)".to_owned(),
            (false, true) => info.driver.clone(),
            (true, false) => info.driver_info.clone(),
            (false, false) => format!("{} ({})", info.driver, info.driver_info),
        };
        AdapterFacts {
            renderer: info.name.clone(),
            vendor: format!("0x{:04X}", info.vendor),
            backend: format!("{:?}", info.backend),
            driver,
            device: format!("0x{:04X} ({:?})", info.device, info.device_type),
        }
    }
}

/// Errors returned by `mc-render`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RenderError {
    /// A code path that S11/S12/S18/S20 have not implemented yet.
    #[error("mc-render: {0} is not implemented yet")]
    Unimplemented(&'static str),

    /// No wgpu adapter could drive this machine (or this surface).
    ///
    /// Tests should treat this as "skip", not "fail": CI runners frequently have no GPU and no
    /// software fallback installed.
    #[error("mc-render: no suitable GPU adapter was found")]
    NoAdapter,

    /// The adapter refused to hand out a device with the requested features or limits.
    #[error("mc-render: could not create a GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    /// The window handle could not be turned into a drawable surface.
    #[error("mc-render: could not create a surface: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),

    /// This frame's colour target could not be acquired — occluded window, or a compositor
    /// timeout. Skip the frame and try the next one; it is not fatal.
    #[error("mc-render: no frame is available right now")]
    FrameUnavailable,

    /// An operation that needs a window was attempted on a headless context.
    #[error("mc-render: {0} requires a window; this context is headless")]
    Headless(&'static str),

    /// Reading a texture back to the CPU failed.
    #[error("mc-render: GPU readback failed: {0}")]
    Readback(String),

    /// The window could not be created.
    #[error("mc-render: could not create a window: {0}")]
    Window(String),

    /// The winit event loop could not be created or run.
    #[error("mc-render: event loop error: {0}")]
    EventLoop(String),

    /// An explicitly requested diagnostic frame trace could not be configured or written.
    #[error("mc-render: frame trace error: {0}")]
    FrameTrace(String),

    /// `options.txt` could not be read or written. Parse problems are never errors — see
    /// [`settings`].
    #[error("mc-render: settings i/o failed: {0}")]
    Settings(String),

    /// The GPU device was lost (the driver stopped responding or was reset):
    /// nothing can be drawn any more. The text is the driver's reason.
    #[error(
        "mc-render: the graphics device was lost (the graphics driver stopped responding or was \
         reset): {0}"
    )]
    DeviceLost(String),

    /// A failure after adapter selection, retaining the facts a crash report can still present.
    #[error("{source}")]
    WithAdapter {
        /// Original render failure.
        #[source]
        source: Box<RenderError>,
        /// Adapter facts captured before the failure.
        facts: AdapterFacts,
    },
}

impl RenderError {
    /// Attach adapter facts unless an inner render boundary already did so.
    #[must_use]
    pub fn with_adapter(self, facts: AdapterFacts) -> RenderError {
        if self.adapter_facts().is_some() {
            self
        } else {
            RenderError::WithAdapter {
                source: Box::new(self),
                facts,
            }
        }
    }

    /// Adapter facts retained by a post-selection failure, if selection completed.
    #[must_use]
    pub fn adapter_facts(&self) -> Option<&AdapterFacts> {
        match self {
            RenderError::WithAdapter { facts, .. } => Some(facts),
            _ => None,
        }
    }
}

#[cfg(test)]
mod adapter_fact_tests {
    use super::*;

    fn facts() -> AdapterFacts {
        AdapterFacts {
            renderer: "Synthetic Renderer".to_owned(),
            vendor: "0x1234".to_owned(),
            backend: "Vulkan".to_owned(),
            driver: "Synthetic Driver (1.2.3)".to_owned(),
            device: "0x5678 (DiscreteGpu)".to_owned(),
        }
    }

    #[test]
    fn a_post_selection_error_retains_adapter_facts_without_a_gpu() {
        let expected = facts();
        let error = RenderError::Readback("boom".to_owned()).with_adapter(expected.clone());

        assert_eq!(error.adapter_facts(), Some(&expected));
        assert_eq!(error.to_string(), "mc-render: GPU readback failed: boom");
    }
}
