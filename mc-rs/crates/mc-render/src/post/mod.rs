//! Post-processing pipeline seam and vanilla shader-group loading.
//!
//! # Layout
//!
//! | module | covers | java |
//! |---|---|---|
//! | [`shader_group`] | parses `post/*.json`/`program/*.json` (hand-rolled JSON reader — see its module docs for why) | `client/shader/ShaderGroup.java`/`ShaderManager.java` |
//! | [`pipeline`] | [`pipeline::PostChain`], a `wgpu`-native multi-pass compositor driven by a parsed group, and [`pipeline::PostPhase`], which runs it over the live frame | `client/shader/Shader.java`/`Framebuffer.java` (rewritten, not ported — invariant I2) |
//! | [`overlay`] | [`overlay::OverlayPhase`]: underwater screen texture, the animated nether-portal overlay, and a retained gamma input | `ItemRenderer.renderOverlays`, `GuiIngame.renderPortal`, `EntityRenderer.updateLightmap`'s gamma lerp |
//!
//! This *is* a working screen-space post-processing pipeline: [`shader_group`] parses actual
//! vanilla JSON from the `1.8.9.jar`, [`pipeline::PostChain`] executes the multi-pass framebuffer
//! graph with real `wgpu` draws, and [`pipeline::PostPhase`] reaches the live frame through
//! [`crate::SceneTextures`]. All 25 of vanilla's post-processing programs have a native WGSL
//! translation; [`pipeline::PostEffect::PASSTHROUGH_PROGRAMS`] is empty (`tests/s20_post.rs`
//! counts it from the jar). Read [`pipeline`]'s module docs for the vertex-stage details.

// Motion blur and ASC CDL grading: one fullscreen pass.
pub mod motion_grade;
// The ASCII Super Secret Setting.
pub mod ascii;
// Lane-owned modules.
pub mod ao;
// Reusable planar distance + view normals from depth.
pub mod blur;
pub mod taa;
pub mod velocity;
pub mod view_geometry;
// Per-pixel motion blur over the motion target.
pub mod overlay;
pub mod pipeline;
pub mod shader_group;
pub mod super_secret;
pub mod velocity_blur;

pub use ao::{AoPhase, AoQuality, AoSettings};
pub use blur::MenuBlurPhase;
pub use motion_grade::{CdlParams, MotionGradePhase};
pub use overlay::{portal_alpha, OverlayKind, OverlayPhase};
pub use pipeline::{
    entity_outline_enabled, PostChain, PostEffect, PostPhase, PostUniforms, UniformDefaults,
    PROGRAM_DEFAULTS,
};
pub use shader_group::{
    AuxTargetSpec, BlendEquation, BlendFactor, BlendSpec, Json, JsonError, PassSpec, ProgramSpec,
    SamplerSpec, ShaderGroupSpec, TargetSpec, UniformDecl, UniformOverride, MAIN_TARGET,
};
pub use super_secret::{
    SuperSecretShaderAction, SuperSecretShaderCycle, SUPER_SECRET_OFF_INDEX,
    SUPER_SECRET_SHADER_GROUPS,
};
pub use taa::{AaMode, AntiAliasPhase};
pub use velocity::{ObjectMotion, VelocityConsumers, VelocityOutput, VelocityPhase};
pub use velocity_blur::VelocityBlurPhase;
