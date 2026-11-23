//! Shader packs: Classic, Dreamy and Underwater.
//!
//! This module file holds the shared core (pack selection, the shadow map, sun lighting, water,
//! tone mapping, bloom and the shared WGSL helpers). The Dreamy pack brings its own shared pieces
//! (the colour-grading LUT, ground fog, screen-space light rays), and the Underwater pack its
//! caustics, absorption, light shafts and marine snow. Each pack declares its submodules below at
//! its own marker line.
//!
//! The rule every pack keeps: with no pack on, nothing here allocates, records or costs anything
//! beyond a disabled phase's `enabled()` test.
//!
//! # How a pack plugs in (the core)
//!
//! * **State:** one [`ShaderCore`] per client (an `Rc`; clone it into every pack phase). The glue
//!   in `app/mods/tw_shaders.rs` sets the switched-on pack ([`ShaderCore::set_active`], with
//!   exclusivity), and every frame this frame's [`PackFrame`] ([`ShaderCore::set_frame`]).
//! * **Phases:** a pack is its own [`crate::RenderPhase`]s plus one [`CoreConfig`]. Register the
//!   phases once at install (they live for the client's life) at an order from [`order`], and gate `enabled()` on [`ShaderCore::is_running`] plus your own
//!   switches. Build GPU resources on the first enabled frame and drop them when not running.
//! * **Core features:** the running pack's glue hands its [`CoreConfig`] to
//!   [`ShaderCore::set_config`]: shadows, sun shading, water, waving plants (terrain variant),
//!   bloom and tone mapping (the final pass). `None` = off.
//! * **GPU inputs:** [`COMMON_WGSL`] declares `PackFrame` and `ShadowUniform` and helpers; bind
//!   [`ShaderCore::frame_buffer`] and, for shadows, [`ShaderCore::shadow`]. The camera is
//!   `@group(0)` as everywhere (`Gpu::camera_bind_group`).
//! * **MSAA:** with `msaaSamples` above 1 no pack runs ([`ShaderCore::running`] is `None`).

pub mod config;
pub mod frame;
pub mod post;
pub mod rays;
pub mod shadow;
pub mod shared;
pub mod sky;

pub use config::{
    BloomSettings, CoreConfig, LightingSettings, PackMesh, ShadowSettings, ToneCurve,
    TonemapSettings, WaterSettings, WavingSettings,
};
pub use frame::{EyeMedium, FrameInputs, LightColors, PackFrame, PackFrameUniform, PackId};
pub use post::FinalPhase;
pub use rays::RaysPhase;
pub use shadow::{ShadowFrame, ShadowPhase, ShadowUniform, ShadowView};
pub use shared::ShaderCore;
pub use sky::SkyGlowPhase;

/// WGSL shared by every pack pass: the `PackFrame` and `ShadowUniform` structs and pure helpers
/// (no bindings, no entry points). Prepend it to your module's source.
pub const COMMON_WGSL: &str = include_str!("common.wgsl");

/// Phase orders for pack passes.
pub mod order {
    use crate::PhaseOrder;

    /// The shadow map: before the sky and terrain (it needs only the terrain's buffers).
    pub const SHADOW: PhaseOrder = PhaseOrder(150);
    /// World-space composites (ground fog, light shafts, haze): after the screen-space fog (640),
    /// before the late volumetric clouds (642) and TAA (643), so TAA cleans their noise.
    pub const WORLD: PhaseOrder = PhaseOrder(641);
    /// Final passes (bloom, tone map, grading): after the hand, before the HUD.
    pub const FINAL: PhaseOrder = PhaseOrder::POST;
}
pub mod dreamy;
pub mod godrays;
pub mod grade;
pub mod ground_fog;
pub mod caustics;
pub mod underwater;
