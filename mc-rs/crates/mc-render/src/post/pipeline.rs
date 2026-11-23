//! `PostChain`: a `wgpu`-native multi-pass fullscreen compositor, driven by a parsed
//! [`super::shader_group::ShaderGroupSpec`], plus [`PostPhase`], which plugs it into the live
//! frame.
//!
//! java: `client/shader/ShaderGroup.java`/`Shader.java`/`Framebuffer.java` — see
//! `super::shader_group`'s module docs for exactly what those classes do; this module is the
//! **rewrite** (no fixed-function GL state machine, no `glUseProgram`/uniform
//! location lookups) rather than a transliteration.
//!
//! # What this can and cannot do
//!
//! [`PostChain`] executes a real vanilla shader-group JSON's *pass graph* — target allocation,
//! `intarget`/`outtarget`/`auxtargets` chaining, ping-pong through a `"swap"` target — and runs a
//! hand-written WGSL translation of each pass's `program/<name>.fsh` where one exists. It also
//! threads the program's declared uniform defaults and the pass's `uniforms` overrides into that
//! WGSL, which is what makes `desaturate.json`, `green.json` and `creeper.json` genuinely
//! different from each other: all three run the *same* `color_convolve` program and differ only
//! in the colour matrix they override it with.
//!
//! **All 25 of vanilla's post-processing programs have a native WGSL translation.**
//! `fxaa`, `ntsc_encode`/`ntsc_decode`, `outline_watercolor`, and `spider` are baked below;
//! `spider` also carries the shipped `rotscale.vsh` coordinate transform. Resource-pack GLSL
//! overrides remain explicitly rejected because this backend only executes baked translations.
//!
//! The infallible [`PostChain::new`] API still records malformed graph/source errors explicitly;
//! all known vanilla program names use their baked WGSL entry point.
//!
//! The shared WGSL module covers the common `blit`/`sobel` coordinate convention. The native
//! `blobs`, `blobs2`, `antialias`, and `bumpy` programs use their shipped bottom-left vertex
//! convention through a small alternate vertex module; `flip`, `fxaa`, and `spider` keep their
//! Java vertex coordinate paths as well. Custom resource-pack GLSL is reported rather than
//! silently treated as a baked translation or a blit.
//!
//! `auxtargets` whose `id` names a *file* rather than an allocated framebuffer **is** wired:
//! [`PostChain::new`] takes a [`mc_assets::ResourceManager`] and mirrors vanilla's
//! `ShaderGroup.parseGroup` — look the id up as a framebuffer first, and when nothing matches
//! (nor is it `"minecraft:main"`), load `textures/effect/<id>.png` through it, exactly what
//! `notch.json`'s `DitherSampler` needs. A missing or corrupt static texture is reported as
//! [`PostChainError::AuxiliaryTexture`] during graph validation; the compatibility constructor
//! still retains its historical white placeholder only after recording that error. See
//! `tests/s20_post.rs`'s `the_real_notch_group_uses_the_actual_dither_texture_not_a_flat_placeholder`
//! for the GPU-verified proof against the real jar.
//!
//! # Reaching the live frame
//!
//! [`PostPhase`] is the seam. It opts into [`crate::SceneTextures`] via
//! [`crate::RenderPhase::needs_scene_color`], so [`crate::PhaseList`] hands it a readable
//! snapshot of the frame so far and it composites the chain's output straight back onto the
//! frame's colour target. Registering one is the whole integration:
//!
//! ```no_run
//! # use mc_render::{Gpu, PhaseList, PhaseOrder};
//! # use mc_render::post::{PostChain, PostPhase, ShaderGroupSpec};
//! # fn wire(gpu: &Gpu, rm: &mc_assets::ResourceManager, phases: &mut PhaseList, spec: &ShaderGroupSpec) {
//! let mut post = PostPhase::new(gpu);
//! post.set_chain(PostChain::new(gpu, rm, spec, gpu.config.format, (gpu.config.width, gpu.config.height)));
//! phases.insert(PhaseOrder::POST, Box::new(post));
//! # }
//! ```

use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    fmt,
};

use mc_assets::{sprite::anaglyph_rgba, ResourceLocation, ResourceManager};
use wgpu::util::DeviceExt;

use super::shader_group::{
    BlendEquation, BlendFactor, BlendSpec, PassSpec, ProgramSpec, ShaderGroupSpec,
    ShaderSourceError, ShaderSourceInfo, ShaderStage, MAIN_TARGET,
};
use crate::{Camera, Gpu, RenderPhase, SceneTextures};

/// Which native WGSL fragment entry point a pass runs. java: the *effect* a `program/<name>.json`
/// would have produced, for the twenty this port reimplements directly.
///
/// The variants are named after vanilla's **program** names (`program/*.json` stems), not after
/// the **group** names (`post/*.json` stems) — a distinction worth stating because they overlap
/// confusingly: `desaturate` and `green` are groups with no program of the same name, and both
/// are built out of [`PostEffect::ColorConvolve`] plus uniform overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PostEffect {
    /// `blit` — copy modulated by `ColorModulate`. Also the fallback for an unknown program.
    Blit,
    /// `invert` — `mix(c, 1 - c, InverseAmount)`.
    Invert,
    /// `flip` — vertical mirror (vanilla puts this in `flip.vsh`).
    Flip,
    /// `color_convolve` — 3x3 colour matrix, offset/scale, saturation. Five groups use it.
    ColorConvolve,
    /// `bits` — mosaic + colour quantisation + saturation.
    Bits,
    /// `deconverge` — per-channel UV offset (CRT colour fringing).
    Deconverge,
    /// `wobble` — sinusoidal UV wobble plus a hue rotation.
    Wobble,
    /// `sobel` — signed four-tap edge difference.
    Sobel,
    /// `entity_outline` (`entity_sobel.fsh`) — the alpha-keyed edge pass.
    EntityOutline,
    /// `outline` — edge difference subtracted from the centre texel.
    Outline,
    /// `outline_soft` — eight-tap edge sum modulating a quantised luma.
    OutlineSoft,
    /// `outline_combine` — screen-adds an auxiliary outline target back over the frame.
    OutlineCombine,
    /// `blur` — separable tent blur along `BlurDir` with radius `Radius`.
    Blur,
    /// `antialias` — fixed five-tap-per-axis smoothing.
    Antialias,
    /// `blobs` — 13-tap erosion (min filter).
    Blobs,
    /// `blobs2` — circular dilation (max filter) of radius `Radius`.
    Blobs2,
    /// `bumpy` — fake emboss from neighbouring normalised texels.
    Bumpy,
    /// `notch` — stepped coordinates + dither texture + heavy quantisation.
    Notch,
    /// `phosphor` — CRT persistence against an auxiliary previous-frame target.
    Phosphor,
    /// `scan_pincushion` — pincushion warp plus scanlines.
    ScanPincushion,
    /// `fxaa` - luma-adaptive anti-aliasing.
    Fxaa,
    /// `ntsc_encode` - composite NTSC encoding.
    NtscEncode,
    /// `ntsc_decode` - composite NTSC reconstruction.
    NtscDecode,
    /// `outline_watercolor` - twelve-neighbour luma outline.
    OutlineWatercolor,
    /// `spider` - rotate/scale clipping with a blur auxiliary.
    Spider,
}

/// The coordinate convention of a vanilla vertex shader.
///
/// Most post programs use `sobel.vsh`/`blit.vsh`, which expose a top-left UV to the fragment
/// shader after the OpenGL viewport transform. `blobs.vsh` and `bumpy.vsh` instead pass the
/// clip-space mapping through without the y inversion. Keeping this in the pipeline key prevents
/// two programs that share a fragment implementation from accidentally sharing the wrong vertex
/// stage as well.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum VertexMode {
    TopLeft,
    BottomLeft,
    Flip,
    Fxaa,
    RotateScale,
}

/// Vanilla's outline pass gate, kept pure so each conjunct (live shader/target, spectator
/// viewer, held key) can be covered without starting a window or a GPU queue.
#[must_use]
pub fn entity_outline_enabled(
    has_shader: bool,
    has_target: bool,
    local_player_is_spectator: bool,
    key_held: bool,
) -> bool {
    has_shader && has_target && local_player_is_spectator && key_held
}

impl PostEffect {
    /// Compatibility list of vanilla program names with no baked translation. It is empty for
    /// 1.8.9; custom resource-pack GLSL is still rejected before pipeline creation.
    ///
    /// `spider` additionally uses `rotscale.vsh`'s vertex-stage rotate/scale through its dedicated
    /// `RotateScale` pipeline mode.
    pub const PASSTHROUGH_PROGRAMS: &'static [&'static str] = &[];

    /// Map a `pass.name` (a `program/<name>.json` stem) to a native effect, or [`PostEffect::Blit`]
    /// when this port has no native implementation for it.
    #[must_use]
    pub fn from_program_name(name: &str) -> Self {
        match name {
            "invert" => PostEffect::Invert,
            "flip" => PostEffect::Flip,
            "color_convolve" => PostEffect::ColorConvolve,
            "bits" => PostEffect::Bits,
            "deconverge" => PostEffect::Deconverge,
            "wobble" => PostEffect::Wobble,
            "sobel" => PostEffect::Sobel,
            "entity_outline" => PostEffect::EntityOutline,
            "outline" => PostEffect::Outline,
            "outline_soft" => PostEffect::OutlineSoft,
            "outline_combine" => PostEffect::OutlineCombine,
            "blur" => PostEffect::Blur,
            "antialias" => PostEffect::Antialias,
            "blobs" => PostEffect::Blobs,
            "blobs2" => PostEffect::Blobs2,
            "bumpy" => PostEffect::Bumpy,
            "notch" => PostEffect::Notch,
            "phosphor" => PostEffect::Phosphor,
            "scan_pincushion" => PostEffect::ScanPincushion,
            "fxaa" => PostEffect::Fxaa,
            "ntsc_encode" => PostEffect::NtscEncode,
            "ntsc_decode" => PostEffect::NtscDecode,
            "outline_watercolor" => PostEffect::OutlineWatercolor,
            "spider" => PostEffect::Spider,
            _ => PostEffect::Blit,
        }
    }

    /// Whether `name` really is one of the programs this port translates, as opposed to falling
    /// back to [`PostEffect::Blit`]. `"blit"` itself and every shipped 1.8.9 program are native.
    #[must_use]
    pub fn is_native_program(name: &str) -> bool {
        name == "blit" || PostEffect::from_program_name(name) != PostEffect::Blit
    }

    /// Whether a pass name identifies a program this module knows by name. Every shipped 1.8.9
    /// program is backed by a baked WGSL entry point; resource-pack source overrides are checked
    /// separately by [`inspect_shader_sources`].
    #[must_use]
    fn is_known_program(name: &str) -> bool {
        name == "blit"
            || Self::is_native_program(name)
            || Self::PASSTHROUGH_PROGRAMS.contains(&name)
    }

    fn vertex_mode(self) -> VertexMode {
        match self {
            Self::Flip => VertexMode::Flip,
            Self::Antialias | Self::Blobs | Self::Blobs2 | Self::Bumpy => VertexMode::BottomLeft,
            Self::Fxaa => VertexMode::Fxaa,
            Self::Spider => VertexMode::RotateScale,
            _ => VertexMode::TopLeft,
        }
    }

    fn expected_aux_sampler(self) -> Option<&'static str> {
        match self {
            Self::OutlineCombine => Some("OutlineSampler"),
            Self::Notch => Some("DitherSampler"),
            Self::Phosphor => Some("PrevSampler"),
            Self::NtscDecode => Some("BaseSampler"),
            Self::Spider => Some("BlurSampler"),
            _ => None,
        }
    }

    fn entry_point(self) -> &'static str {
        match self {
            PostEffect::Blit => "blit",
            PostEffect::Invert => "invert",
            PostEffect::Flip => "flip",
            PostEffect::ColorConvolve => "color_convolve",
            PostEffect::Bits => "bits",
            PostEffect::Deconverge => "deconverge",
            PostEffect::Wobble => "wobble",
            PostEffect::Sobel => "sobel",
            PostEffect::EntityOutline => "entity_outline",
            PostEffect::Outline => "outline",
            PostEffect::OutlineSoft => "outline_soft",
            PostEffect::OutlineCombine => "outline_combine",
            PostEffect::Blur => "blur",
            PostEffect::Antialias => "antialias",
            PostEffect::Blobs => "blobs",
            PostEffect::Blobs2 => "blobs2",
            PostEffect::Bumpy => "bumpy",
            PostEffect::Notch => "notch",
            PostEffect::Phosphor => "phosphor",
            PostEffect::ScanPincushion => "scan_pincushion",
            PostEffect::Fxaa => "fxaa",
            PostEffect::NtscEncode => "ntsc_encode",
            PostEffect::NtscDecode => "ntsc_decode",
            PostEffect::OutlineWatercolor => "outline_watercolor",
            PostEffect::Spider => "spider",
        }
    }

    #[cfg(test)]
    const ALL: [PostEffect; 25] = [
        PostEffect::Blit,
        PostEffect::Invert,
        PostEffect::Flip,
        PostEffect::ColorConvolve,
        PostEffect::Bits,
        PostEffect::Deconverge,
        PostEffect::Wobble,
        PostEffect::Sobel,
        PostEffect::EntityOutline,
        PostEffect::Outline,
        PostEffect::OutlineSoft,
        PostEffect::OutlineCombine,
        PostEffect::Blur,
        PostEffect::Antialias,
        PostEffect::Blobs,
        PostEffect::Blobs2,
        PostEffect::Bumpy,
        PostEffect::Notch,
        PostEffect::Phosphor,
        PostEffect::ScanPincushion,
        PostEffect::Fxaa,
        PostEffect::NtscEncode,
        PostEffect::NtscDecode,
        PostEffect::OutlineWatercolor,
        PostEffect::Spider,
    ];
}

/// A program can share its WGSL entry point with another program yet require a different blend
/// mode, so [`PostEffect`] alone is not enough to identify a `wgpu` pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PostPipelineKey {
    effect: PostEffect,
    blend: BlendSpec,
    vertex: VertexMode,
    cull: bool,
}

/// Translate the resolved `JsonBlendingMode` state to the two independent `wgpu` blend
/// components. `None` is vanilla's disabled/default state.
///
/// `BlendFactor::Invalid` is deliberately not representable in `wgpu`: vanilla sends `-1` to GL,
/// receives `GL_INVALID_ENUM`, and leaves the prior GL factor state in place. A retained wgpu
/// render pipeline cannot reproduce that invalid mutable-state operation, so malformed pack
/// factors take the disabled fallback below rather than being silently rewritten to a valid mode.
fn pipeline_blend_state(blend: BlendSpec) -> Option<wgpu::BlendState> {
    if !blend.enabled {
        return None;
    }

    let operation = match blend.func {
        BlendEquation::Add => wgpu::BlendOperation::Add,
        BlendEquation::Subtract => wgpu::BlendOperation::Subtract,
        BlendEquation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        BlendEquation::Min => wgpu::BlendOperation::Min,
        BlendEquation::Max => wgpu::BlendOperation::Max,
    };
    let factor = |factor| match factor {
        BlendFactor::Zero => Some(wgpu::BlendFactor::Zero),
        BlendFactor::One => Some(wgpu::BlendFactor::One),
        BlendFactor::SrcColor => Some(wgpu::BlendFactor::Src),
        BlendFactor::OneMinusSrcColor => Some(wgpu::BlendFactor::OneMinusSrc),
        BlendFactor::DstColor => Some(wgpu::BlendFactor::Dst),
        BlendFactor::OneMinusDstColor => Some(wgpu::BlendFactor::OneMinusDst),
        BlendFactor::SrcAlpha => Some(wgpu::BlendFactor::SrcAlpha),
        BlendFactor::OneMinusSrcAlpha => Some(wgpu::BlendFactor::OneMinusSrcAlpha),
        BlendFactor::DstAlpha => Some(wgpu::BlendFactor::DstAlpha),
        BlendFactor::OneMinusDstAlpha => Some(wgpu::BlendFactor::OneMinusDstAlpha),
        BlendFactor::Invalid => None,
    };
    let color = wgpu::BlendComponent {
        src_factor: factor(blend.srcrgb)?,
        dst_factor: factor(blend.dstrgb)?,
        operation,
    };
    let (srcalpha, dstalpha) = blend
        .separate_alpha_factors()
        .unwrap_or((blend.srcrgb, blend.dstrgb));
    Some(wgpu::BlendState {
        color,
        alpha: wgpu::BlendComponent {
            src_factor: factor(srcalpha)?,
            dst_factor: factor(dstalpha)?,
            operation,
        },
    })
}

/// Read the exact program JSON which vanilla's `ShaderManager` would load for this pass. Missing
/// or malformed files already make this rewrite fall back to its safe native/passthrough path, so
/// their blend is likewise disabled rather than guessed from the program name.
fn program_blend_spec(rm: &ResourceManager, program_name: &str) -> BlendSpec {
    let Ok(location) = ResourceLocation::parse(&format!("shaders/program/{program_name}.json"))
    else {
        return BlendSpec::default();
    };
    rm.load_string(&location)
        .ok()
        .and_then(|text| ProgramSpec::parse(&text).ok())
        .and_then(|program| program.blend)
        .unwrap_or_default()
}

fn program_cull(rm: &ResourceManager, program_name: &str) -> bool {
    let Ok(location) = ResourceLocation::parse(&format!("shaders/program/{program_name}.json"))
    else {
        return true;
    };
    rm.load_string(&location)
        .ok()
        .and_then(|text| ProgramSpec::parse(&text).ok())
        .is_none_or(|program| program.cull)
}

/// Errors that can be surfaced before a post chain is allowed to use the baked WGSL path.
///
/// Java reports source failures while constructing `ShaderManager`; this renderer keeps its
/// existing infallible [`PostChain::new`] compatibility wrapper, but [`PostChain::try_new`] and
/// [`PostChain::validate_shader_sources`] expose the same failure boundary explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostChainError {
    /// A pass names no program known to the vanilla post-program table.
    UnknownProgram {
        /// The `passes[].name` value which could not be resolved.
        program: String,
    },
    /// A known vanilla program has no honest baked WGSL implementation. The compatibility
    /// constructor can still record a no-op fallback, but [`PostChain::try_new`] reports this
    /// instead of claiming that a blit is the program's effect.
    UnsupportedProgram {
        /// The known program whose GLSL/vertex behavior is not implemented here.
        program: String,
    },
    /// A pass refers to a framebuffer which Java's `ShaderGroup` would reject during loading.
    MissingTarget {
        /// Program name owning the invalid pass.
        program: String,
        /// Target spelling from `intarget` or `outtarget`.
        target: String,
        /// Whether the missing target was used as input or output.
        direction: &'static str,
    },
    /// Two object-shaped target declarations use the same name. Java's object branch fails in
    /// `initTarget`; its bare-string branch intentionally replaces the first map entry.
    DuplicateTarget {
        /// Duplicate target spelling.
        target: String,
    },
    /// Java's fixed backend can bind all declared aux samplers. This rewrite has one auxiliary
    /// texture slot in its shared WGSL interface, so a second slot is rejected explicitly rather
    /// than being silently dropped.
    UnsupportedAuxiliarySamplers {
        /// Program owning the pass.
        program: String,
        /// Number of aux entries declared by the pass.
        count: usize,
    },
    /// A native program's fragment shader declares an auxiliary sampler, but the pass did not
    /// provide the matching `auxtargets[]` entry. Binding the white placeholder would hide a
    /// Java sampler-state mismatch, so the graph reports it explicitly.
    MissingAuxiliarySampler {
        /// Program owning the pass.
        program: String,
        /// Sampler name required by the native translation.
        sampler: String,
    },
    /// A static aux texture was named but its Java-required dimensions were not supplied.
    MissingAuxiliaryDimensions {
        /// Program owning the pass.
        program: String,
        /// Sampler name from `auxtargets[].name`.
        sampler: String,
        /// Texture id from `auxtargets[].id`.
        id: String,
    },
    /// A static aux texture omitted Java's required nearest/bilinear filter selector.
    MissingAuxiliaryFilter {
        /// Program owning the pass.
        program: String,
        /// Sampler name from `auxtargets[].name`.
        sampler: String,
        /// Texture id from `auxtargets[].id`.
        id: String,
    },
    /// A static aux texture could not be found or decoded.
    AuxiliaryTexture {
        /// Program owning the pass.
        program: String,
        /// Sampler name from `auxtargets[].name`.
        sampler: String,
        /// Texture id from `auxtargets[].id`.
        id: String,
        /// Exact resource location Java would request.
        location: Box<ResourceLocation>,
        /// Resource-manager/PNG diagnostic.
        message: Box<str>,
    },
    /// A pass-level override names no uniform declared by the baked vanilla program.
    UnknownUniform {
        /// Program owning the pass.
        program: String,
        /// Unknown uniform name.
        uniform: String,
    },
    /// Java only has ShaderUniform setters for zero through four float components in a group
    /// override. Values beyond that are a load-time error, never a truncating upload.
    UniformArity {
        /// Program owning the pass.
        program: String,
        /// Uniform name.
        uniform: String,
        /// Number of values supplied.
        count: usize,
    },
    /// A program manifest resource could not be read.
    ProgramResource {
        /// Program name being loaded.
        program: String,
        /// Exact manifest location.
        location: ResourceLocation,
        /// Resource-manager diagnostic.
        message: String,
    },
    /// A program manifest was present but malformed.
    ProgramJson {
        /// Program name being loaded.
        program: String,
        /// Exact manifest location.
        location: ResourceLocation,
        /// JSON parser diagnostic, including its nested path where available.
        message: String,
    },
    /// A referenced GLSL source failed deterministic load/validation.
    ShaderSource(ShaderSourceError),
    /// A valid GLSL source override exists, but no honest GLSL-to-WGSL translation is available.
    ///
    /// Keeping this error explicit prevents a resource-pack shader from being silently ignored
    /// while also refusing to feed GLSL bytes to `wgpu::ShaderSource::Wgsl`.
    UnsupportedGlsl {
        /// The first source override in deterministic pass/stage order.
        source: ShaderSourceInfo,
    },
}

impl fmt::Display for PostChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownProgram { program } => {
                write!(f, "shader program {program} is not known to the post renderer")
            }
            Self::UnsupportedProgram { program } => write!(
                f,
                "shader program {program} has no executable WGSL translation (GLSL/vertex behavior is unsupported)"
            ),
            Self::MissingTarget {
                program,
                target,
                direction,
            } => write!(
                f,
                "shader program {program} {direction} target {target:?} does not exist"
            ),
            Self::DuplicateTarget { target } => {
                write!(f, "shader target {target:?} is already defined")
            }
            Self::UnsupportedAuxiliarySamplers { program, count } => write!(
                f,
                "shader program {program} declares {count} auxiliary samplers, but the WGSL post backend supports one"
            ),
            Self::MissingAuxiliarySampler { program, sampler } => write!(
                f,
                "shader program {program} requires auxiliary sampler {sampler}, but the pass declares none"
            ),
            Self::MissingAuxiliaryDimensions {
                program,
                sampler,
                id,
            } => write!(
                f,
                "shader program {program} auxiliary sampler {sampler} texture {id:?} has no width/height"
            ),
            Self::MissingAuxiliaryFilter {
                program,
                sampler,
                id,
            } => write!(
                f,
                "shader program {program} auxiliary sampler {sampler} texture {id:?} has no bilinear flag"
            ),
            Self::AuxiliaryTexture {
                program,
                sampler,
                id,
                location,
                message,
            } => write!(
                f,
                "shader program {program} auxiliary sampler {sampler} texture {id:?} at {location} could not be loaded: {message}"
            ),
            Self::UnknownUniform { program, uniform } => write!(
                f,
                "shader program {program} uniform {uniform:?} does not exist"
            ),
            Self::UniformArity {
                program,
                uniform,
                count,
            } => write!(
                f,
                "shader program {program} uniform {uniform:?} has {count} values; expected at most four"
            ),
            Self::ProgramResource {
                program,
                location,
                message,
            } => write!(
                f,
                "shader program {program} manifest {location} could not be read: {message}"
            ),
            Self::ProgramJson {
                program,
                location,
                message,
            } => write!(
                f,
                "invalid shader program {program} manifest {location}: {message}"
            ),
            Self::ShaderSource(error) => error.fmt(f),
            Self::UnsupportedGlsl { source } => write!(
                f,
                "shader source override {} from pack {:?} is GLSL and cannot be executed by the WGSL renderer",
                source.location,
                source.pack_name
            ),
        }
    }
}

impl std::error::Error for PostChainError {}

fn source_candidate_location(pass_name: &str, stage: ShaderStage) -> ResourceLocation {
    ResourceLocation::parse(&format!("shaders/program/{pass_name}{}", stage.extension()))
        .expect("constructed shader source candidate location")
}

/// Inspect every program manifest referenced by a group and resolve its Java shader resources.
///
/// The returned list is stable in pass order and vertex-before-fragment order, with duplicate
/// program stages removed. It is deliberately independent of a GPU so malformed resource-pack
/// input can be tested and reported before pipeline creation.
fn inspect_shader_sources(
    rm: &ResourceManager,
    spec: &ShaderGroupSpec,
) -> Result<Vec<ShaderSourceInfo>, PostChainError> {
    let mut seen = HashSet::new();
    let mut resolved = Vec::new();
    for pass in &spec.passes {
        // java: ShaderManager constructs `new ResourceLocation("shaders/program/" +
        // programName + ".json")`; preserve its first-colon namespace split for custom names.
        let location = ResourceLocation::parse(&format!("shaders/program/{}.json", pass.name))
            .expect("constructed shader program location");
        let text = if !rm.contains(&location) {
            // A source file without its manifest cannot be mapped to Java's vertex/fragment names;
            // surface that malformed pack shape instead of guessing from the pass stem.
            let has_bare_source = [ShaderStage::Vertex, ShaderStage::Fragment]
                .into_iter()
                .any(|stage| rm.contains(&source_candidate_location(&pass.name, stage)));
            if has_bare_source {
                return Err(PostChainError::ProgramResource {
                    program: pass.name.clone(),
                    location,
                    message: "manifest is missing but a shader source is present".to_owned(),
                });
            }
            continue;
        } else {
            rm.load_string(&location)
                .map_err(|error| PostChainError::ProgramResource {
                    program: pass.name.clone(),
                    location: location.clone(),
                    message: error.to_string(),
                })?
        };
        let program = ProgramSpec::parse(&text).map_err(|error| PostChainError::ProgramJson {
            program: pass.name.clone(),
            location: location.clone(),
            message: error.to_string(),
        })?;
        let sources = program
            .resolve_shader_sources(rm, &pass.name)
            .map_err(PostChainError::ShaderSource)?;
        for source in sources.iter() {
            if seen.insert((source.stage, source.location.clone())) {
                resolved.push(source.clone());
            }
        }
    }
    Ok(resolved)
}

fn static_aux_location(id: &str) -> Option<ResourceLocation> {
    ResourceLocation::parse(&format!("textures/effect/{id}.png")).ok()
}

fn baked_uniform_is_known(program: &str, name: &str) -> bool {
    let engine_uniform = matches!(name, "ProjMat" | "InSize" | "OutSize")
        || (program == "flip" && name == "ScreenSize");
    engine_uniform
        || PROGRAM_DEFAULTS
            .iter()
            .find(|(program_name, _)| *program_name == program)
            .is_some_and(|(_, uniforms)| uniforms.iter().any(|(uniform, _)| *uniform == name))
}

/// Validate the parts of `ShaderGroup.parseGroup` which are otherwise easy for the infallible
/// compatibility constructor to turn into a plausible-looking but wrong draw. The graph is
/// checked without a GPU so [`PostChain::try_new`] has the same fail-fast boundary as Java;
/// [`PostChain::new`] records the first error and deliberately keeps its historical safe fallback.
fn validate_post_graph(rm: &ResourceManager, spec: &ShaderGroupSpec) -> Result<(), PostChainError> {
    let mut targets = HashSet::new();
    for target in &spec.targets {
        // java: `initTarget` explicitly rejects duplicate object declarations, while the bare
        // string branch calls `addFramebuffer` directly and therefore overwrites an earlier map
        // entry. Preserve that slightly odd distinction instead of inventing a blanket error.
        if matches!(target, super::shader_group::TargetSpec::Sized { .. })
            && !targets.insert(target.name().to_owned())
        {
            return Err(PostChainError::DuplicateTarget {
                target: target.name().to_owned(),
            });
        }
        targets.insert(target.name().to_owned());
    }

    let target_exists = |name: &str| name == MAIN_TARGET || targets.contains(name);
    for pass in &spec.passes {
        if !PostEffect::is_known_program(&pass.name) {
            return Err(PostChainError::UnknownProgram {
                program: pass.name.clone(),
            });
        }
        if !PostEffect::is_native_program(&pass.name) {
            return Err(PostChainError::UnsupportedProgram {
                program: pass.name.clone(),
            });
        }
        if !target_exists(&pass.intarget) {
            return Err(PostChainError::MissingTarget {
                program: pass.name.clone(),
                target: pass.intarget.clone(),
                direction: "input",
            });
        }
        if !target_exists(&pass.outtarget) {
            return Err(PostChainError::MissingTarget {
                program: pass.name.clone(),
                target: pass.outtarget.clone(),
                direction: "output",
            });
        }
        if pass.auxtargets.len() > 1 {
            return Err(PostChainError::UnsupportedAuxiliarySamplers {
                program: pass.name.clone(),
                count: pass.auxtargets.len(),
            });
        }
        if let Some(expected) = PostEffect::from_program_name(&pass.name).expected_aux_sampler() {
            if !pass.auxtargets.iter().any(|aux| aux.name == expected) {
                return Err(PostChainError::MissingAuxiliarySampler {
                    program: pass.name.clone(),
                    sampler: expected.to_owned(),
                });
            }
        }

        for aux in &pass.auxtargets {
            if target_exists(&aux.id) {
                continue;
            }
            let Some(location) = static_aux_location(&aux.id) else {
                return Err(PostChainError::AuxiliaryTexture {
                    program: pass.name.clone(),
                    sampler: aux.name.clone(),
                    id: aux.id.clone(),
                    location: Box::new(ResourceLocation::minecraft("textures/effect/invalid.png")),
                    message: "invalid resource location".into(),
                });
            };
            if aux.width.is_none() || aux.height.is_none() {
                return Err(PostChainError::MissingAuxiliaryDimensions {
                    program: pass.name.clone(),
                    sampler: aux.name.clone(),
                    id: aux.id.clone(),
                });
            }
            if !aux.bilinear_specified {
                return Err(PostChainError::MissingAuxiliaryFilter {
                    program: pass.name.clone(),
                    sampler: aux.name.clone(),
                    id: aux.id.clone(),
                });
            }
            if !rm.contains(&location) {
                return Err(PostChainError::AuxiliaryTexture {
                    program: pass.name.clone(),
                    sampler: aux.name.clone(),
                    id: aux.id.clone(),
                    location: Box::new(location),
                    message: "resource is missing".into(),
                });
            }
            if let Err(error) = rm.load_png_rgba(&location) {
                return Err(PostChainError::AuxiliaryTexture {
                    program: pass.name.clone(),
                    sampler: aux.name.clone(),
                    id: aux.id.clone(),
                    location: Box::new(location),
                    message: error.to_string().into_boxed_str(),
                });
            }
        }

        for uniform in &pass.uniforms {
            if uniform.values.len() > 4 {
                return Err(PostChainError::UniformArity {
                    program: pass.name.clone(),
                    uniform: uniform.name.clone(),
                    count: uniform.values.len(),
                });
            }
            if PostEffect::is_native_program(&pass.name)
                && !baked_uniform_is_known(&pass.name, &uniform.name)
            {
                return Err(PostChainError::UnknownUniform {
                    program: pass.name.clone(),
                    uniform: uniform.name.clone(),
                });
            }
        }
    }
    Ok(())
}

/// The uniform block every translated program shares, matching `post.wgsl`'s `PostUniform`.
///
/// Vanilla gives each program its own uniform set; packing them into one block is the `wgpu`
/// rewrite of `ShaderManager`'s per-name `ShaderUniform` list. Fields are grouped four floats at
/// a time because `std140` pads a `vec3`/`float` to 16 bytes anyway.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostUniforms {
    /// `xy` = `InSize` (the input target's pixel size), `zw` = `OutSize`.
    pub sizes: [f32; 4],
    /// `blit`'s `ColorModulate`.
    pub color_modulate: [f32; 4],
    /// `InverseAmount`, `Radius`, `Resolution`, `MosaicSize`.
    pub scalars: [f32; 4],
    /// `Saturation`, `LumaRamp`, `LumaLevel`, `Time`.
    pub scalars2: [f32; 4],
    /// `BlurDir.xy`, `Frequency.xy`.
    pub blur_dir: [f32; 4],
    /// `WobbleAmount.xy`; for `spider`, `InScale.xy` and `InOffset.zw`.
    pub wobble: [f32; 4],
    /// `color_convolve`'s `Gray`; `fxaa`'s `SpanMax` in `w`.
    pub gray: [f32; 4],
    /// `color_convolve`'s `RedMatrix`; `fxaa`'s `SubPixelShift` in `w`.
    pub red_matrix: [f32; 4],
    /// `color_convolve`'s `GreenMatrix`; `fxaa`'s `ReduceMul` in `w`.
    pub green_matrix: [f32; 4],
    /// `color_convolve`'s `BlueMatrix`.
    pub blue_matrix: [f32; 4],
    /// `color_convolve`'s `Offset`.
    pub offset: [f32; 4],
    /// `color_convolve`'s `ColorScale`.
    pub color_scale: [f32; 4],
    /// `deconverge`'s `ConvergeX`; `spider`'s `Scissor`.
    pub converge_x: [f32; 4],
    /// `deconverge`'s `ConvergeY`; `spider`'s `Vignette`.
    pub converge_y: [f32; 4],
    /// `deconverge`'s `RadialConvergeX`; `spider`'s `InRotation` in `w`.
    pub radial_x: [f32; 4],
    /// `deconverge`'s `RadialConvergeY`.
    pub radial_y: [f32; 4],
    /// `phosphor`'s `Phosphor`.
    pub phosphor: [f32; 4],
}

impl Default for PostUniforms {
    fn default() -> Self {
        Self {
            sizes: [1.0, 1.0, 1.0, 1.0],
            color_modulate: [1.0, 1.0, 1.0, 1.0],
            scalars: [0.0, 0.0, 1.0, 1.0],
            scalars2: [1.0, 1.0, 2.0, 0.0],
            blur_dir: [1.0, 0.0, 1.0, 1.0],
            wobble: [0.0, 0.0, 0.0, 0.0],
            gray: [0.3, 0.59, 0.11, 0.0],
            red_matrix: [1.0, 0.0, 0.0, 0.0],
            green_matrix: [0.0, 1.0, 0.0, 0.0],
            blue_matrix: [0.0, 0.0, 1.0, 0.0],
            offset: [0.0; 4],
            color_scale: [1.0, 1.0, 1.0, 0.0],
            converge_x: [0.0; 4],
            converge_y: [0.0; 4],
            radial_x: [1.0, 1.0, 1.0, 0.0],
            radial_y: [1.0, 1.0, 1.0, 0.0],
            phosphor: [0.0; 4],
        }
    }
}

impl PostUniforms {
    /// Set a vanilla uniform by its GLSL name. Unknown names are ignored, exactly as vanilla's
    /// `ShaderManager.getShaderUniform` returns `null` for a uniform the program never declared.
    pub fn set_named(&mut self, name: &str, values: &[f32]) {
        // `ShaderGroup.initUniform` chooses ShaderUniform.set's arity from the number of JSON
        // values.  A short vector therefore updates only its leading components; it does not
        // zero the components omitted by the pack.  An empty values array is a no-op.  Keeping
        // that distinction matters for e.g. a one-component ColorModulate override over the
        // program's `[1, 1, 1, 1]` default.
        let write = |dst: &mut [f32]| {
            for (slot, value) in dst.iter_mut().zip(values.iter().copied()) {
                *slot = value;
            }
        };
        let write_scalar = |dst: &mut f32| {
            if let Some(value) = values.first().copied() {
                *dst = value;
            }
        };
        match name {
            "ColorModulate" => write(&mut self.color_modulate),
            "InverseAmount" => write_scalar(&mut self.scalars[0]),
            "Radius" => write_scalar(&mut self.scalars[1]),
            "Resolution" => write_scalar(&mut self.scalars[2]),
            "MosaicSize" => write_scalar(&mut self.scalars[3]),
            "Saturation" => write_scalar(&mut self.scalars2[0]),
            "LumaRamp" => write_scalar(&mut self.scalars2[1]),
            "LumaLevel" => write_scalar(&mut self.scalars2[2]),
            "Time" => write_scalar(&mut self.scalars2[3]),
            "BlurDir" => write(&mut self.blur_dir[..2]),
            "Frequency" => write(&mut self.blur_dir[2..]),
            "WobbleAmount" => write(&mut self.wobble[..2]),
            "Gray" => write(&mut self.gray[..3]),
            "RedMatrix" => write(&mut self.red_matrix[..3]),
            "GreenMatrix" => write(&mut self.green_matrix[..3]),
            "BlueMatrix" => write(&mut self.blue_matrix[..3]),
            "Offset" => write(&mut self.offset[..3]),
            "ColorScale" => write(&mut self.color_scale[..3]),
            "ConvergeX" => write(&mut self.converge_x[..3]),
            "ConvergeY" => write(&mut self.converge_y[..3]),
            "RadialConvergeX" => write(&mut self.radial_x[..3]),
            "RadialConvergeY" => write(&mut self.radial_y[..3]),
            "Phosphor" => write(&mut self.phosphor[..3]),
            "SpanMax" => write_scalar(&mut self.gray[3]),
            "SubPixelShift" => write_scalar(&mut self.red_matrix[3]),
            "ReduceMul" => write_scalar(&mut self.green_matrix[3]),
            "InScale" => write(&mut self.wobble[..2]),
            "InOffset" => write(&mut self.wobble[2..4]),
            "InRotation" => write_scalar(&mut self.radial_x[3]),
            "Scissor" => write(&mut self.converge_x),
            "Vignette" => write(&mut self.converge_y),
            // ProjMat / InSize / OutSize are supplied by the engine. VxOffset is declared by
            // vanilla's FXAA fragment but never read, so it is intentionally not packed here.
            _ => {}
        }
    }

    /// Start from `program`'s declared defaults, then apply `pass`'s `uniforms` overrides —
    /// java: `ShaderManager`'s constructor reads the program JSON's `uniforms` block, then
    /// `ShaderGroup.parseGroup` applies the pass's own overrides on top.
    #[must_use]
    pub fn for_pass(pass: &PassSpec) -> Self {
        let mut out = Self::default();
        for (name, values) in program_defaults(&pass.name) {
            out.set_named(name, values);
        }
        for over in &pass.uniforms {
            out.set_named(&over.name, &over.values);
        }
        out
    }

    /// Apply uniforms that vanilla supplies immediately before drawing every frame.
    ///
    /// `Time` is intentionally applied after the program default and the group's per-pass
    /// overrides. java: `Shader.loadShader` calls `getShaderUniformOrDefault("Time").set(...)`
    /// after `ShaderGroup.parseGroup` has installed those static values.
    #[must_use]
    fn for_frame(mut self, time: f32) -> Self {
        self.set_named("Time", &[time]);
        self
    }
}

/// Vanilla's `ShaderGroup` clock, expressed in ticks and normalised to `0..=1` for the shaders.
/// java: `ShaderGroup.loadShaderGroup(float)` (`field_148036_j` / `field_148037_k`).
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct ShaderTime {
    elapsed_ticks: f32,
    previous_partial_ticks: f32,
}

impl ShaderTime {
    /// Advance from one render partial-tick sample to the next.
    ///
    /// The branch and strict `> 20.0` wrap are kept in Java's order: a falling partial-tick
    /// value crosses a whole-tick boundary, while an equal value advances nothing.
    fn advance(&mut self, partial_ticks: f32) -> f32 {
        if partial_ticks < self.previous_partial_ticks {
            self.elapsed_ticks += 1.0 - self.previous_partial_ticks;
            self.elapsed_ticks += partial_ticks;
        } else {
            self.elapsed_ticks += partial_ticks - self.previous_partial_ticks;
        }

        self.previous_partial_ticks = partial_ticks;
        while self.elapsed_ticks > 20.0 {
            self.elapsed_ticks -= 20.0;
        }
        self.elapsed_ticks / 20.0
    }
}

/// One program's `(uniform name, default values)` pairs, as declared in its `program/*.json`.
pub type UniformDefaults = &'static [(&'static str, &'static [f32])];

/// Every non-engine uniform each vanilla program declares, with its default, transcribed from
/// `assets/minecraft/shaders/program/<name>.json` in `1.8.9.jar`.
///
/// `ProjMat`/`InSize`/`OutSize` are omitted: the engine supplies them. `tests/s20_post.rs`'s
/// `program_uniform_defaults_match_the_real_jar` re-reads the jar and asserts this table is
/// still right, so it cannot drift from the asset it was copied out of.
pub const PROGRAM_DEFAULTS: &[(&str, UniformDefaults)] = &[
    ("blit", &[("ColorModulate", &[1.0, 1.0, 1.0, 1.0])]),
    ("flip", &[("ColorModulate", &[1.0, 1.0, 1.0, 1.0])]),
    ("invert", &[("InverseAmount", &[0.0])]),
    (
        "color_convolve",
        &[
            ("Gray", &[0.3, 0.59, 0.11]),
            ("RedMatrix", &[1.0, 0.0, 0.0]),
            ("GreenMatrix", &[0.0, 1.0, 0.0]),
            ("BlueMatrix", &[0.0, 0.0, 1.0]),
            ("Offset", &[0.0, 0.0, 0.0]),
            ("ColorScale", &[1.0, 1.0, 1.0]),
            ("Saturation", &[1.8]),
        ],
    ),
    (
        "bits",
        &[
            ("Resolution", &[4.0]),
            ("Saturation", &[1.5]),
            ("MosaicSize", &[8.0]),
        ],
    ),
    (
        "deconverge",
        &[
            ("ConvergeX", &[-4.0, 0.0, 2.0]),
            ("ConvergeY", &[0.0, -4.0, 2.0]),
            ("RadialConvergeX", &[1.0, 1.0, 1.0]),
            ("RadialConvergeY", &[1.0, 1.0, 1.0]),
        ],
    ),
    (
        "wobble",
        &[
            ("Time", &[0.0]),
            ("Frequency", &[512.0, 288.0]),
            ("WobbleAmount", &[0.002, 0.002]),
        ],
    ),
    ("blur", &[("BlurDir", &[1.0, 1.0]), ("Radius", &[5.0])]),
    ("blobs2", &[("Radius", &[7.0])]),
    ("phosphor", &[("Phosphor", &[0.3, 0.3, 0.3])]),
    (
        "fxaa",
        &[
            ("SpanMax", &[8.0]),
            ("SubPixelShift", &[0.25]),
            ("ReduceMul", &[0.125]),
        ],
    ),
    ("ntsc_encode", &[]),
    ("ntsc_decode", &[]),
    ("outline_watercolor", &[("LumaRamp", &[16.0])]),
    (
        "spider",
        &[
            ("InScale", &[1.0, 1.0]),
            ("InOffset", &[0.0, 0.0]),
            ("InRotation", &[0.0]),
            ("Time", &[0.0]),
            ("Scissor", &[0.0, 0.0, 1.0, 1.0]),
            ("Vignette", &[0.0, 0.0, 1.0, 1.0]),
        ],
    ),
    (
        "outline_soft",
        &[("LumaRamp", &[16.0]), ("LumaLevel", &[4.0])],
    ),
    ("sobel", &[]),
    ("entity_outline", &[]),
    ("outline", &[]),
    ("outline_combine", &[]),
    ("antialias", &[]),
    ("blobs", &[]),
    ("bumpy", &[]),
    ("notch", &[]),
    ("scan_pincushion", &[]),
];

fn program_defaults(name: &str) -> UniformDefaults {
    PROGRAM_DEFAULTS
        .iter()
        .find(|(n, _)| *n == name)
        .map_or(&[][..], |(_, u)| *u)
}

struct PostTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    /// A `None` component follows the live frame size; a `Some` component is the fixed size from
    /// a vanilla `Sized` target declaration.  Keeping this beside the allocation makes resize
    /// lifecycle explicit instead of silently stretching a stale framebuffer.
    fixed_width: Option<u32>,
    fixed_height: Option<u32>,
}

/// The largest side a target with a fixed size from a shader group's JSON gets (see
/// [`PostTarget::with_size_spec`]).
const MAX_FIXED_TARGET_SIDE: u32 = 4096;

impl PostTarget {
    fn new(gpu: &Gpu, width: u32, height: u32, format: wgpu::TextureFormat, label: &str) -> Self {
        Self::with_size_spec(gpu, width, height, format, label, None, None)
    }

    /// A target side as created. A size can come from a resource
    /// pack's JSON (`"width": 100000`): past the device's limit, creating the texture is a
    /// validation error, which ends the process. Sides are held to the device limit, and a fixed
    /// size to [`MAX_FIXED_TARGET_SIDE`] as well (no shipped group fixes a target near it), so
    /// one declaration cannot ask for gigabytes of video memory.
    fn bounded_side(gpu: &Gpu, size: u32, fixed: Option<u32>) -> u32 {
        let mut limit = gpu.device.limits().max_texture_dimension_2d.max(1);
        if fixed.is_some() {
            limit = limit.min(MAX_FIXED_TARGET_SIDE);
        }
        size.clamp(1, limit)
    }

    fn with_size_spec(
        gpu: &Gpu,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
        fixed_width: Option<u32>,
        fixed_height: Option<u32>,
    ) -> Self {
        let width = Self::bounded_side(gpu, width, fixed_width);
        let height = Self::bounded_side(gpu, height, fixed_height);
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width,
            height,
            fixed_width,
            fixed_height,
        }
    }

    fn resize(
        &mut self,
        gpu: &Gpu,
        main_width: u32,
        main_height: u32,
        format: wgpu::TextureFormat,
    ) {
        let width = Self::bounded_side(
            gpu,
            self.fixed_width.unwrap_or(main_width),
            self.fixed_width,
        );
        let height = Self::bounded_side(
            gpu,
            self.fixed_height.unwrap_or(main_height),
            self.fixed_height,
        );
        if width == self.width && height == self.height {
            return;
        }
        let replacement = Self::with_size_spec(
            gpu,
            width,
            height,
            format,
            "post chain resized target",
            self.fixed_width,
            self.fixed_height,
        );
        *self = replacement;
    }
}

struct ResolvedPass {
    effect: PostEffect,
    /// Parsed from this pass's `program/<name>.json`, not inferred from the shared WGSL effect.
    blend: BlendSpec,
    vertex: VertexMode,
    cull: bool,
    native: bool,
    valid_targets: bool,
    intarget: String,
    outtarget: String,
    /// The first `auxtargets` entry this chain can actually satisfy, if any — either an
    /// allocated framebuffer target's name (or `"minecraft:main"`) or an id resolved to a static
    /// `textures/effect/<id>.png` in [`PostChain::aux_textures`]. The `OutlineSampler` /
    /// `PrevSampler` / `DitherSampler` slot. java: `ShaderGroup.parseGroup`'s
    /// `addAuxFramebuffer`/file-fallback.
    auxtarget: Option<ResolvedAux>,
    uniforms: PostUniforms,
}

#[derive(Debug, Clone)]
struct ResolvedAux {
    id: String,
    width: u32,
    height: u32,
    bilinear: bool,
}

/// One static `textures/effect/<id>.png` upload and the raw bytes Java keeps on the CPU side
/// until `TextureUtil.copyToBufferPos` applies the optional anaglyph transform.  The texture view
/// is intentionally retained separately from the source bytes: changing the setting rewrites the
/// existing allocation, so target dimensions and all sampler state remain exactly as resolved by
/// the pass graph.
struct AuxTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    source_rgba: Vec<u8>,
    width: u32,
    height: u32,
}

impl AuxTexture {
    fn new(gpu: &Gpu, source_rgba: Vec<u8>, width: u32, height: u32, anaglyph: bool) -> Self {
        let upload_rgba = anaglyph_rgba(&source_rgba, anaglyph);
        let texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("post chain aux texture"),
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
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &upload_rgba,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            source_rgba,
            width,
            height,
        }
    }

    /// Re-upload from the retained raw source, never from the currently transformed bytes.
    /// `COPY_DST` is part of the static aux texture allocation specifically so a live option
    /// change can preserve its existing view, dimensions, and sampler bindings.
    fn set_anaglyph(&self, gpu: &Gpu, enabled: bool) {
        let upload_rgba = anaglyph_rgba(&self.source_rgba, enabled);
        gpu.queue.write_texture(
            self.texture.as_image_copy(),
            &upload_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Loads `textures/effect/<id>.png` for an `auxtargets[]` entry whose `id` names no allocated
/// framebuffer target — the file-fallback half of java's `ShaderGroup.parseGroup` aux
/// resolution. It uses the resource manager's direct PNG decoder so Java's static texture path
/// does not accidentally inherit animated-sprite frame semantics.
///
/// Returns `None` — never panics — when `rm` has no such resource (missing pack, corrupt PNG,
/// zero-sized image). Graph validation reports that condition explicitly; the infallible builder
/// keeps using its white compatibility placeholder after retaining the error.
fn load_aux_texture(
    gpu: &Gpu,
    rm: &ResourceManager,
    id: &str,
    anaglyph: bool,
) -> Option<AuxTexture> {
    let loc = ResourceLocation::parse(&format!("textures/effect/{id}.png")).ok()?;
    let (rgba, width, height, _) = rm.load_png_rgba(&loc).ok()?;
    Some(AuxTexture::new(gpu, rgba, width, height, anaglyph))
}

/// `blobs.vsh`/`bumpy.vsh` expose the clip-space y direction directly. The fragment entry points
/// already carry the shipped GLSL offsets, so only the vertex UV convention needs a second module.
const BOTTOM_LEFT_VERTEX_WGSL: &str = r#"
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_bottom_left(@builtin(vertex_index) index: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    var out: VsOut;
    let p = corners[index];
    out.clip = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + vec2<f32>(0.5);
    return out;
}
"#;

/// The two remaining vanilla vertex stages which cannot share the ordinary fullscreen varying:
/// `fxaa.vsh` carries a half-texel-shifted neighbourhood origin, while `rotscale.vsh` carries a
/// rotated/scaled coordinate for `spiderclip.fsh`. The packed fields are aliases documented on
/// [`PostUniforms`]; they occupy the same 17-vec4 ABI as the common module.
const SPECIAL_VERTEX_WGSL: &str = r#"
struct PostUniform {
    sizes: vec4<f32>,
    color_modulate: vec4<f32>,
    scalars: vec4<f32>,
    scalars2: vec4<f32>,
    blur_dir: vec4<f32>,
    wobble: vec4<f32>,
    gray: vec4<f32>,
    red_matrix: vec4<f32>,
    green_matrix: vec4<f32>,
    blue_matrix: vec4<f32>,
    offset: vec4<f32>,
    color_scale: vec4<f32>,
    converge_x: vec4<f32>,
    converge_y: vec4<f32>,
    radial_x: vec4<f32>,
    radial_y: vec4<f32>,
    phosphor: vec4<f32>,
};

@group(0) @binding(2) var<uniform> post: PostUniform;

struct FxaaOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) pos_pos: vec4<f32>,
};

struct SpiderOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) scaled_coord: vec2<f32>,
};

fn fullscreen_corner(index: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return corners[index];
}

fn top_left_uv(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
}

@vertex
fn vs_fxaa(@builtin(vertex_index) index: u32) -> FxaaOut {
    let p = fullscreen_corner(index);
    let uv = top_left_uv(p);
    var out: FxaaOut;
    out.clip = vec4<f32>(p, 0.2, 1.0);
    out.uv = uv;
    out.pos_pos = vec4<f32>(
        uv,
        uv - (vec2<f32>(0.5 + post.red_matrix.w) / max(post.sizes.zw, vec2<f32>(1.0))),
    );
    return out;
}

@vertex
fn vs_spider(@builtin(vertex_index) index: u32) -> SpiderOut {
    let p = fullscreen_corner(index);
    let tex_coord = top_left_uv(p);
    let radians = post.radial_x.w * 0.0174532925;
    let cosine = cos(radians);
    let sine = sin(radians);
    let rot_u = tex_coord.x * cosine - tex_coord.y * sine;
    let rot_v = tex_coord.y * cosine + tex_coord.x * sine;

    var out: SpiderOut;
    out.clip = vec4<f32>(p, 0.2, 1.0);
    out.uv = tex_coord;
    out.scaled_coord = vec2<f32>(rot_u, rot_v) * post.wobble.xy + post.wobble.zw;
    return out;
}
"#;

/// The Java `flip` program uses `blit.fsh` with a distinct vertex shader. Keeping both stages in
/// this small module avoids the old fragment-only y inversion, which was only accidentally right
/// for equal-sized targets.
const FLIP_SHADER_WGSL: &str = r#"
@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_samp: sampler;

struct PostUniform {
    sizes: vec4<f32>,
    color_modulate: vec4<f32>,
    scalars: vec4<f32>,
    scalars2: vec4<f32>,
    blur_dir: vec4<f32>,
    // zw is the display/screen size; xy remains WobbleAmount for the shared layout.
    wobble: vec4<f32>,
};
@group(0) @binding(2) var<uniform> post: PostUniform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    var out: VsOut;
    let p = corners[index];
    out.clip = vec4<f32>(p, 0.0, 1.0);
    let base = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    let in_out = post.sizes.zw / max(post.sizes.xy, vec2<f32>(1.0));
    let in_screen = post.wobble.zw / max(post.sizes.xy, vec2<f32>(1.0));
    out.uv = vec2<f32>(base.x * in_out.x,
                       base.y * in_out.y - (1.0 - in_screen.y));
    return out;
}

@fragment
fn flip(in: VsOut) -> @location(0) vec4<f32> {
    // `post.wgsl`'s historical flip entry point performs this y inversion in the fragment
    // stage. Keep it here while the vertex stage supplies Java's size-ratio coordinates; this
    // preserves the shipped group effect for equal-sized targets and fixes resized targets.
    return textureSample(in_tex, in_samp, vec2<f32>(in.uv.x, 1.0 - in.uv.y))
        * post.color_modulate;
}
"#;

/// The internal copy of `"minecraft:main"` that a group reading the
/// main target after writing it renders into (see [`route_main_reads_after_writes`]).
const MAIN_COPY_TARGET: &str = "minecraft:main (copy)";

/// A group that reads `"minecraft:main"` again after a pass wrote it
/// (vanilla's `art`, `green` and `ntsc`) would sample the frame's own target there: a swap-chain
/// image, or the target a frame is recorded into, neither of which can be sampled. wgpu refused
/// the bind group and the client crashed when one of those groups was picked. Such a group
/// renders every write of the main target, and every read after the first write, into an
/// internal copy of it instead, and one more `blit` pass copies that to the frame at the end.
/// java: `"minecraft:main"` is an ordinary framebuffer that can be read and written. `None`
/// (nothing changes) for every other group, and when a group already names a target like the copy.
fn route_main_reads_after_writes(spec: &ShaderGroupSpec) -> Option<ShaderGroupSpec> {
    let mut written = false;
    let mut reads_after_write = false;
    for pass in &spec.passes {
        let writes = pass.outtarget == MAIN_TARGET;
        let reads =
            pass.intarget == MAIN_TARGET || pass.auxtargets.iter().any(|aux| aux.id == MAIN_TARGET);
        reads_after_write |= written && reads && !writes;
        written |= writes;
    }
    if !reads_after_write
        || spec
            .targets
            .iter()
            .any(|target| target.name() == MAIN_COPY_TARGET)
    {
        return None;
    }
    let mut routed = spec.clone();
    routed.targets.push(super::shader_group::TargetSpec::Named(
        MAIN_COPY_TARGET.to_owned(),
    ));
    let mut written = false;
    for pass in &mut routed.passes {
        let writes = pass.outtarget == MAIN_TARGET;
        // A pass that reads and writes main keeps reading the scene snapshot, as before.
        if written && !writes {
            if pass.intarget == MAIN_TARGET {
                MAIN_COPY_TARGET.clone_into(&mut pass.intarget);
            }
            for aux in &mut pass.auxtargets {
                if aux.id == MAIN_TARGET {
                    MAIN_COPY_TARGET.clone_into(&mut aux.id);
                }
            }
        }
        if writes {
            MAIN_COPY_TARGET.clone_into(&mut pass.outtarget);
            written = true;
        }
    }
    routed.passes.push(PassSpec {
        name: "blit".to_owned(),
        intarget: MAIN_COPY_TARGET.to_owned(),
        outtarget: MAIN_TARGET.to_owned(),
        auxtargets: Vec::new(),
        uniforms: Vec::new(),
    });
    Some(routed)
}

/// A runnable, `wgpu`-backed shader group: allocated framebuffers plus an ordered pass list.
/// See the module docs for exactly what "runnable" means here.
pub struct PostChain {
    pipelines: HashMap<PostPipelineKey, wgpu::RenderPipeline>,
    /// Same effects, built at [`Gpu::sample_count`] for the pass that writes
    /// `"minecraft:main"` — that one goes through [`Gpu::begin_pass`], which under MSAA routes to
    /// the multisampled attachment and resolves. Writing the resolve target directly instead
    /// would be silently overwritten by the next phase's resolve.
    final_pipelines: HashMap<PostPipelineKey, wgpu::RenderPipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    dummy_aux: wgpu::TextureView,
    /// Static `textures/effect/<id>.png` aux textures, keyed by the `auxtargets[].id` that
    /// resolved to them (e.g. `"dither"`). Populated once, in [`PostChain::new`], for every aux
    /// id that names neither an allocated framebuffer target nor `"minecraft:main"` — java: the
    /// file-fallback half of `ShaderGroup.parseGroup`'s aux resolution. An id that resolves to
    /// neither a target nor a loadable file keeps using [`Self::dummy_aux`], same as every aux
    /// slot did before this field existed.
    /// Raw decoded pixels stay beside each allocation so a live anaglyph toggle can replay the
    /// exact upload without double-converting the already-transformed GPU copy.
    aux_textures: HashMap<String, AuxTexture>,
    targets: HashMap<String, PostTarget>,
    passes: Vec<ResolvedPass>,
    uniform_buffers: Vec<wgpu::Buffer>,
    main_size: (u32, u32),
    format: wgpu::TextureFormat,
    /// Resource-backed GLSL stages found while inspecting this chain's manifests.
    shader_sources: Vec<ShaderSourceInfo>,
    /// Validation/translation status retained by the infallible compatibility constructor.
    shader_source_error: Option<PostChainError>,
    /// Java's global upload-time texture transform.
    anaglyph: bool,
    /// Whether the bytes currently resident in `aux_textures` reflect `anaglyph`. A chain can be
    /// installed into a phase before the phase has a `Gpu` available to rewrite its allocations.
    aux_anaglyph_uploaded: Cell<bool>,
}

impl PostChain {
    /// Build a chain while retaining the historical baked-WGSL fallback behavior.
    ///
    /// Resource-backed GLSL is inspected first. If it is malformed, or if a valid source would
    /// require an unavailable GLSL-to-WGSL translation, the chain still uses its existing baked
    /// effect so current callers remain infallible; [`Self::shader_source_error`] exposes that
    /// decision. Call [`Self::try_new`] when the Java-like fail-fast boundary is required.
    #[must_use]
    pub fn new(
        gpu: &Gpu,
        rm: &ResourceManager,
        spec: &ShaderGroupSpec,
        format: wgpu::TextureFormat,
        main_size: (u32, u32),
    ) -> Self {
        let (shader_sources, source_error) = match inspect_shader_sources(rm, spec) {
            Ok(sources) => {
                let error = sources
                    .first()
                    .cloned()
                    .map(|source| PostChainError::UnsupportedGlsl { source });
                (sources, error)
            }
            Err(error) => (Vec::new(), Some(error)),
        };
        let graph_error = validate_post_graph(rm, spec).err();
        let mut chain = Self::build_baked(gpu, rm, spec, format, main_size);
        chain.shader_sources = shader_sources;
        // Graph errors take precedence over a source override because an unknown program or
        // missing target is already unreplayable before GLSL translation is considered.
        chain.shader_source_error = graph_error.or(source_error);
        chain
    }

    /// Build a chain with deterministic Java-style shader-source validation.
    ///
    /// A group with no resource-backed `.vsh`/`.fsh` files uses the existing baked WGSL table and
    /// succeeds. Any present source is validated and then rejected as unsupported because this
    /// crate has no GLSL compiler or translator; this is intentional, since passing GLSL bytes to
    /// `wgpu::ShaderSource::Wgsl` would make a resource-pack override appear to work while being
    /// semantically false.
    ///
    /// # Errors
    /// [`PostChainError`] for malformed program manifests, malformed/missing source stages, or a
    /// valid GLSL source that cannot be executed by this WGSL backend.
    pub fn try_new(
        gpu: &Gpu,
        rm: &ResourceManager,
        spec: &ShaderGroupSpec,
        format: wgpu::TextureFormat,
        main_size: (u32, u32),
    ) -> Result<Self, PostChainError> {
        validate_post_graph(rm, spec)?;
        let shader_sources = inspect_shader_sources(rm, spec)?;
        if let Some(source) = shader_sources.first().cloned() {
            return Err(PostChainError::UnsupportedGlsl { source });
        }
        let mut chain = Self::build_baked(gpu, rm, spec, format, main_size);
        chain.shader_sources = shader_sources;
        Ok(chain)
    }

    /// Validate a group's resource-backed shader sources without constructing a GPU pipeline.
    ///
    /// This is the source/override oracle used by reload code and tests. The returned entries are
    /// ordered by first pass use, with each program stage listed once, and include the exact pack
    /// provenance and byte fingerprint. A non-empty result is still GLSL, not executable WGSL.
    pub fn validate_shader_sources(
        rm: &ResourceManager,
        spec: &ShaderGroupSpec,
    ) -> Result<Vec<ShaderSourceInfo>, PostChainError> {
        inspect_shader_sources(rm, spec)
    }

    /// Resource-backed shader stages discovered for this chain.
    #[must_use]
    pub fn shader_sources(&self) -> &[ShaderSourceInfo] {
        &self.shader_sources
    }

    /// Alias emphasizing that these entries are the visible resource-pack override oracle.
    #[must_use]
    pub fn shader_source_overrides(&self) -> &[ShaderSourceInfo] {
        self.shader_sources()
    }

    /// The source or graph validation status retained by [`Self::new`], if any. A non-`None`
    /// value means the infallible constructor had to retain a compatibility fallback and the
    /// caller should prefer [`Self::try_new`] when it needs Java's fail-fast behavior.
    #[must_use]
    pub fn shader_source_error(&self) -> Option<&PostChainError> {
        self.shader_source_error.as_ref()
    }

    /// Whether this chain saw any resource-backed GLSL source.
    #[must_use]
    pub fn has_shader_source_overrides(&self) -> bool {
        !self.shader_sources.is_empty()
    }

    /// Whether Java's upload-time anaglyph transform is selected for this chain's static aux
    /// textures.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's upload-time anaglyph transform for resident static aux textures.
    ///
    /// The decoded source bytes are retained by each [`AuxTexture`], so a live false → true →
    /// false sequence always rebuilds from the same raw pixels and never applies the integer
    /// transform twice. Rewriting the existing `COPY_DST` allocation also leaves the resolved
    /// Java dimensions and the nearest/linear sampler choice untouched.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        self.anaglyph = enabled;
        self.ensure_anaglyph_uploaded(gpu);
    }

    /// Upload a phase-seeded setting before the first draw. This keeps `PostPhase::set_chain`
    /// compatible with its historical no-`Gpu` signature while still making a chain loaded after
    /// an already-enabled setting correct on its very first frame.
    fn ensure_anaglyph_uploaded(&self, gpu: &Gpu) {
        if self.aux_anaglyph_uploaded.get() == self.anaglyph {
            return;
        }
        for texture in self.aux_textures.values() {
            texture.set_anaglyph(gpu, self.anaglyph);
        }
        self.aux_anaglyph_uploaded.set(self.anaglyph);
    }

    /// Seed the logical setting when a phase installs this chain before its next GPU fanout.
    /// The following [`Self::set_anaglyph_with_gpu`] call notices that the allocation bytes are
    /// still in their previous mode and performs the upload.
    fn set_anaglyph_state(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Build every declared target and resolve every pass's effect. `main_size` sizes any target
    /// the spec did not give explicit dimensions to (java: `ShaderGroup.addFramebuffer`'s
    /// `mainFramebufferWidth`/`Height` default).
    ///
    /// `rm` is where an `auxtargets[]` entry whose `id` names no allocated target falls back to
    /// — java: `ShaderGroup.parseGroup` is itself handed an `IResourceManager` for exactly this,
    /// to load `textures/effect/<id>.png`. A `rm` that does not carry the file (an empty
    /// [`ResourceManager`], or any pack stack missing that texture) remains a white compatibility
    /// placeholder for this infallible builder, while `new` records the exact
    /// [`PostChainError::AuxiliaryTexture`] for callers that need to reject the chain.
    #[must_use]
    fn build_baked(
        gpu: &Gpu,
        rm: &ResourceManager,
        spec: &ShaderGroupSpec,
        format: wgpu::TextureFormat,
        main_size: (u32, u32),
    ) -> Self {
        // A group that reads the main target after writing it runs
        // through a copy of it; every other group exactly as written.
        let routed = route_main_reads_after_writes(spec);
        let spec = routed.as_ref().unwrap_or(spec);
        // `ShaderGroup` resolves a pass name to `program/<name>.json`, and `ShaderManager`
        // owns that program's `JsonBlendingMode`. Do this before pipeline creation because two
        // passes sharing an effect entry point may legitimately need distinct blend states.
        let pass_states: Vec<(BlendSpec, bool)> = spec
            .passes
            .iter()
            .map(|pass| {
                (
                    program_blend_spec(rm, &pass.name),
                    program_cull(rm, &pass.name),
                )
            })
            .collect();
        let pipeline_keys: HashSet<PostPipelineKey> = spec
            .passes
            .iter()
            .zip(&pass_states)
            .map(|(pass, &(blend, cull))| PostPipelineKey {
                effect: PostEffect::from_program_name(&pass.name),
                blend,
                vertex: PostEffect::from_program_name(&pass.name).vertex_mode(),
                cull,
            })
            .collect();

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post chain"),
                source: wgpu::ShaderSource::Wgsl(include_str!("post.wgsl").into()),
            });
        let bottom_left_shader = pipeline_keys
            .iter()
            .any(|key| key.vertex == VertexMode::BottomLeft)
            .then(|| {
                gpu.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("post chain bottom-left vertex"),
                        source: wgpu::ShaderSource::Wgsl(BOTTOM_LEFT_VERTEX_WGSL.into()),
                    })
            });
        let flip_shader = pipeline_keys
            .iter()
            .any(|key| key.vertex == VertexMode::Flip)
            .then(|| {
                gpu.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("post chain flip vertex"),
                        source: wgpu::ShaderSource::Wgsl(FLIP_SHADER_WGSL.into()),
                    })
            });
        let special_vertex_shader = pipeline_keys
            .iter()
            .any(|key| matches!(key.vertex, VertexMode::Fxaa | VertexMode::RotateScale))
            .then(|| {
                gpu.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("post chain special vertex"),
                        source: wgpu::ShaderSource::Wgsl(SPECIAL_VERTEX_WGSL.into()),
                    })
            });

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("post chain group0"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            // The alternate `flip` vertex module reads the packed size block to
                            // reproduce Java's vertex-stage coordinate ratios.
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("post chain layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let build = |samples: u32| -> HashMap<PostPipelineKey, wgpu::RenderPipeline> {
            pipeline_keys
                .iter()
                .copied()
                .map(|key| {
                    let (vertex_module, fragment_module, fragment_entry) = match key.vertex {
                        VertexMode::TopLeft => (&shader, &shader, key.effect.entry_point()),
                        VertexMode::BottomLeft => (
                            bottom_left_shader.as_ref().expect("bottom-left module"),
                            &shader,
                            key.effect.entry_point(),
                        ),
                        VertexMode::Flip => (
                            flip_shader.as_ref().expect("flip module"),
                            flip_shader.as_ref().expect("flip module"),
                            "flip",
                        ),
                        VertexMode::Fxaa => (
                            special_vertex_shader
                                .as_ref()
                                .expect("special vertex module"),
                            &shader,
                            key.effect.entry_point(),
                        ),
                        VertexMode::RotateScale => (
                            special_vertex_shader
                                .as_ref()
                                .expect("special vertex module"),
                            &shader,
                            key.effect.entry_point(),
                        ),
                    };
                    let pipeline =
                        gpu.device
                            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("post chain pass"),
                                layout: Some(&pipeline_layout),
                                vertex: wgpu::VertexState {
                                    module: vertex_module,
                                    entry_point: Some(match key.vertex {
                                        VertexMode::BottomLeft => "vs_bottom_left",
                                        VertexMode::Fxaa => "vs_fxaa",
                                        VertexMode::RotateScale => "vs_spider",
                                        _ => "vs",
                                    }),
                                    compilation_options: wgpu::PipelineCompilationOptions::default(
                                    ),
                                    buffers: &[],
                                },
                                primitive: wgpu::PrimitiveState {
                                    cull_mode: key.cull.then_some(wgpu::Face::Back),
                                    // The fullscreen triangle is authored in the same y-down
                                    // framebuffer convention as vanilla's quad, so its winding is
                                    // clockwise in clip space.
                                    front_face: wgpu::FrontFace::Cw,
                                    ..Default::default()
                                },
                                depth_stencil: None,
                                multisample: wgpu::MultisampleState {
                                    count: samples,
                                    ..Default::default()
                                },
                                fragment: Some(wgpu::FragmentState {
                                    module: fragment_module,
                                    entry_point: Some(fragment_entry),
                                    compilation_options: wgpu::PipelineCompilationOptions::default(
                                    ),
                                    targets: &[Some(wgpu::ColorTargetState {
                                        format,
                                        blend: pipeline_blend_state(key.blend),
                                        write_mask: wgpu::ColorWrites::ALL,
                                    })],
                                }),
                                multiview_mask: None,
                                cache: None,
                            });
                    (key, pipeline)
                })
                .collect()
        };
        let pipelines = build(1);
        let final_pipelines = if gpu.sample_count() == 1 {
            build(1)
        } else {
            build(gpu.sample_count())
        };

        // java: Framebuffer.setFramebufferFilter defaults to GL_NEAREST; only a target declared
        // `"bilinear": true` uses GL_LINEAR. Nearest across the board is the closer default and
        // keeps the `oneTexel`-stepping shaders (sobel, blur, notch) sampling texel centres.
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post chain sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let linear_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post chain bilinear aux sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Bound to binding 3 whenever a pass declares no usable auxtarget, so the layout is
        // always satisfiable. White, so `phosphor`'s `prev * Phosphor` degrades to the decay
        // constant rather than to black.
        let dummy_aux = gpu
            .device
            .create_texture_with_data(
                &gpu.queue,
                &wgpu::TextureDescriptor {
                    label: Some("post chain aux placeholder"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255, 255, 255, 255],
            )
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut targets = HashMap::new();
        for t in &spec.targets {
            let (w, h) = match t {
                super::shader_group::TargetSpec::Named(_) => main_size,
                super::shader_group::TargetSpec::Sized { width, height, .. } => {
                    (width.unwrap_or(main_size.0), height.unwrap_or(main_size.1))
                }
            };
            targets.insert(
                t.name().to_owned(),
                match t {
                    super::shader_group::TargetSpec::Named(_) => {
                        PostTarget::with_size_spec(gpu, w, h, format, t.name(), None, None)
                    }
                    super::shader_group::TargetSpec::Sized {
                        width: fixed_width,
                        height: fixed_height,
                        ..
                    } => PostTarget::with_size_spec(
                        gpu,
                        w,
                        h,
                        format,
                        t.name(),
                        *fixed_width,
                        *fixed_height,
                    ),
                },
            );
        }

        // java: the file-fallback half of `ShaderGroup.parseGroup`'s aux resolution — an
        // `auxtargets[]` id that names neither a declared target nor `"minecraft:main"` is a
        // `textures/effect/<id>.png`. Resolved once here, not lazily in `run`, so a broken or
        // missing resource is not re-attempted every frame.
        let mut aux_textures: HashMap<String, AuxTexture> = HashMap::new();
        for p in &spec.passes {
            for a in &p.auxtargets {
                if targets.contains_key(&a.id) || a.id == MAIN_TARGET {
                    continue;
                }
                if aux_textures.contains_key(&a.id) {
                    continue;
                }
                if let Some(texture) = load_aux_texture(gpu, rm, &a.id, false) {
                    aux_textures.insert(a.id.clone(), texture);
                }
            }
        }

        let passes: Vec<ResolvedPass> = spec
            .passes
            .iter()
            .zip(pass_states)
            .map(|(p, (blend, cull))| {
                let effect = PostEffect::from_program_name(&p.name);
                let expected_aux = effect.expected_aux_sampler();
                let auxtarget = p
                    .auxtargets
                    .iter()
                    .find(|a| {
                        expected_aux.is_some_and(|expected| a.name == expected)
                            && (targets.contains_key(&a.id)
                                || a.id == MAIN_TARGET
                                || aux_textures.contains_key(&a.id))
                    })
                    .map(|a| {
                        let (width, height) = if a.id == MAIN_TARGET {
                            main_size
                        } else if let Some(target) = targets.get(&a.id) {
                            (target.width, target.height)
                        } else {
                            (a.width.unwrap_or(1), a.height.unwrap_or(1))
                        };
                        ResolvedAux {
                            id: a.id.clone(),
                            width,
                            height,
                            // Java only applies `bilinear` while configuring a static texture;
                            // framebuffer targets retain their own nearest filter regardless of
                            // the JSON member.
                            bilinear: !targets.contains_key(&a.id)
                                && a.id != MAIN_TARGET
                                && a.bilinear,
                        }
                    });
                ResolvedPass {
                    effect,
                    blend,
                    vertex: effect.vertex_mode(),
                    cull,
                    native: PostEffect::is_native_program(&p.name),
                    valid_targets: (targets.contains_key(&p.intarget) || p.intarget == MAIN_TARGET)
                        && (targets.contains_key(&p.outtarget) || p.outtarget == MAIN_TARGET),
                    intarget: p.intarget.clone(),
                    outtarget: p.outtarget.clone(),
                    auxtarget,
                    uniforms: PostUniforms::for_pass(p),
                }
            })
            .collect();

        let uniform_buffers = passes
            .iter()
            .map(|_| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("post chain pass uniforms"),
                    size: std::mem::size_of::<PostUniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Self {
            pipelines,
            final_pipelines,
            bind_group_layout,
            sampler,
            linear_sampler,
            dummy_aux,
            aux_textures,
            targets,
            passes,
            uniform_buffers,
            main_size,
            format,
            shader_sources: Vec::new(),
            shader_source_error: None,
            anaglyph: false,
            aux_anaglyph_uploaded: Cell::new(false),
        }
    }

    /// Number of passes this chain will record. For the F3 overlay / tests.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// How many of this chain's passes run a real native translation rather than a passthrough
    /// copy. A `blit` pass and every shipped 1.8.9 program count as native.
    #[must_use]
    pub fn native_pass_count(&self) -> usize {
        self.passes.iter().filter(|p| p.native).count()
    }

    /// How many of this chain's passes fall back to a passthrough copy because their program has
    /// no WGSL translation. `pass_count() - native_pass_count()`, spelled out because it is the
    /// number that matters when judging whether a group really works.
    #[must_use]
    pub fn passthrough_pass_count(&self) -> usize {
        self.passes.len() - self.native_pass_count()
    }

    /// The uniform block pass `index` will upload, after program defaults and the pass's own
    /// overrides. For tests — this is how `desaturate.json` is shown to differ from `green.json`
    /// without rendering anything.
    #[must_use]
    pub fn pass_uniforms(&self, index: usize) -> Option<&PostUniforms> {
        self.passes.get(index).map(|p| &p.uniforms)
    }

    /// The Java `AuxSize0` dimensions retained for a pass's bound auxiliary texture. `None`
    /// means the pass has no sampler this baked backend can bind. Static textures use the JSON
    /// dimensions (the values Java uploads), while framebuffer aux targets use their allocation
    /// size.
    #[must_use]
    pub fn pass_auxiliary_size(&self, index: usize) -> Option<(u32, u32)> {
        self.passes
            .get(index)
            .and_then(|pass| pass.auxtarget.as_ref())
            .map(|aux| (aux.width, aux.height))
    }

    /// The view for a named framebuffer target, if this group declared one.
    ///
    /// This is intentionally read-only: callers can composite a target after running a group,
    /// while target allocation and resize remain owned by [`PostChain`].
    #[must_use]
    pub fn target_view(&self, name: &str) -> Option<&wgpu::TextureView> {
        self.targets.get(name).map(|target| &target.view)
    }

    /// Copy a same-sized mask into a named target before running this chain.
    ///
    /// Entity outlines use a dedicated render target as the group's `final` input. The target
    /// carries `COPY_DST` so this copy remains ordered in the same command encoder as the post
    /// passes, with no CPU/GPU synchronization or feedback loop.
    pub fn copy_to_target(
        &self,
        enc: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        name: &str,
    ) -> bool {
        let Some(target) = self.targets.get(name) else {
            return false;
        };
        enc.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        true
    }

    /// Resize every live target. Named targets follow the frame; explicit dimensions remain
    /// fixed exactly like `ShaderGroup.addFramebuffer(name, width, height)`.
    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        self.main_size = (width.max(1), height.max(1));
        for target in self.targets.values_mut() {
            target.resize(gpu, self.main_size.0, self.main_size.1, self.format);
        }
    }

    /// Resolve a target name to a view.
    ///
    /// `"minecraft:main"` is the one name that is two textures here rather than one: the frame
    /// *before* the chain (`main_in`, the snapshot [`crate::SceneTextures`] provides) and the
    /// frame the chain is building (`main_out`). Vanilla has a single `mainFramebuffer` and
    /// happily lets a pass read and write it at once — `green.json` does exactly that — which is
    /// a GL feedback loop with undefined contents. Splitting it in two makes that case defined:
    ///
    /// * a pass **writing** main gets `main_out`;
    /// * a pass **reading** main gets `main_out` if an earlier pass already wrote main and this
    ///   pass is writing somewhere else — which is what a chain like `green.json`'s third pass
    ///   actually means;
    /// * otherwise, reading main gets `main_in`. That covers both "nothing has written main yet"
    ///   and the read-and-write-main case, where `main_out` would be simultaneously a bind-group
    ///   entry and a render attachment and `wgpu` would (correctly) reject it.
    fn view_for<'a>(
        &'a self,
        name: &str,
        main_in: &'a wgpu::TextureView,
        main_out: &'a wgpu::TextureView,
        as_output: bool,
        main_already_written: bool,
        writes_main: bool,
    ) -> &'a wgpu::TextureView {
        if name == MAIN_TARGET {
            if as_output || (main_already_written && !writes_main) {
                main_out
            } else {
                main_in
            }
        } else {
            &self.targets[name].view
        }
    }

    fn size_of(&self, name: &str) -> (u32, u32) {
        if name == MAIN_TARGET {
            self.main_size
        } else {
            self.targets
                .get(name)
                .map_or(self.main_size, |t| (t.width, t.height))
        }
    }

    /// Record every pass in order: `main_in` feeds any pass whose `intarget` is
    /// `"minecraft:main"`, `main_out` receives whatever pass's `outtarget` is
    /// `"minecraft:main"` (the last pass, in every real shipped group). Named targets ping-pong
    /// through the framebuffers allocated in [`PostChain::new`].
    ///
    /// A pass writing `"minecraft:main"` goes through [`Gpu::begin_pass`] so it is MSAA-correct;
    /// every other pass writes its own single-sampled target directly.
    pub fn run(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        main_in: &wgpu::TextureView,
        main_out: &wgpu::TextureView,
        time: f32,
    ) {
        self.ensure_anaglyph_uploaded(gpu);
        let mut main_written = false;
        for (index, pass) in self.passes.iter().enumerate() {
            if !pass.valid_targets {
                // `PostChain::new` is intentionally infallible for legacy callers, but a
                // malformed graph must never reach the indexing paths below. Its exact Java-like
                // failure is retained by `shader_source_error()`.
                continue;
            }
            let writes_main = pass.outtarget == MAIN_TARGET;
            let input = self.view_for(
                &pass.intarget,
                main_in,
                main_out,
                false,
                main_written,
                writes_main,
            );
            let aux = match &pass.auxtarget {
                Some(binding) => self.aux_textures.get(&binding.id).map_or_else(
                    || {
                        self.view_for(
                            &binding.id,
                            main_in,
                            main_out,
                            false,
                            main_written,
                            writes_main,
                        )
                    },
                    |texture| &texture.view,
                ),
                None => &self.dummy_aux,
            };
            let aux_sampler = pass
                .auxtarget
                .as_ref()
                .filter(|binding| binding.bilinear)
                .map_or(&self.sampler, |_| &self.linear_sampler);

            // java: Shader.loadShader replaces the parsed `Time` value immediately before every
            // draw, after all static program defaults and group overrides have been applied.
            let mut uniforms = pass.uniforms.for_frame(time);
            let (in_w, in_h) = self.size_of(&pass.intarget);
            let (out_w, out_h) = self.size_of(&pass.outtarget);
            uniforms.sizes = [in_w as f32, in_h as f32, out_w as f32, out_h as f32];
            // `flip.vsh` receives the actual display size independently of the pass output
            // target. The packed `wobble.zw` slots are otherwise `spider`'s InOffset, so only
            // the flip vertex path may overwrite them here.
            if pass.vertex == VertexMode::Flip {
                uniforms.wobble[2..4]
                    .copy_from_slice(&[self.main_size.0 as f32, self.main_size.1 as f32]);
            }
            gpu.queue.write_buffer(
                &self.uniform_buffers[index],
                0,
                bytemuck::bytes_of(&uniforms),
            );

            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post chain pass group0"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform_buffers[index].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(aux),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(aux_sampler),
                    },
                ],
            });

            if writes_main {
                // java: Shader.loadShader calls framebufferClear before binding framebufferOut.
                // This is observable for the two shipped blended programs (`blit` and
                // `entity_outline`): loading the live main target would blend over stale scene
                // pixels instead of the freshly cleared framebuffer.  begin_clear_pass keeps the
                // same MSAA resolve path as begin_pass while reproducing that transparent clear.
                // Post pipelines are colour-only, so their pass must not bind the shared depth
                // attachment used by ordinary scene clears.
                let mut rp = gpu.begin_clear_color_pass(enc, main_out, wgpu::Color::TRANSPARENT);
                rp.set_pipeline(
                    &self.final_pipelines[&PostPipelineKey {
                        effect: pass.effect,
                        blend: pass.blend,
                        vertex: pass.vertex,
                        cull: pass.cull,
                    }],
                );
                rp.set_bind_group(0, &bind_group, &[]);
                rp.draw(0..3, 0..1);
                main_written = true;
            } else {
                let output = self.view_for(
                    &pass.outtarget,
                    main_in,
                    main_out,
                    true,
                    main_written,
                    writes_main,
                );
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("post chain pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(
                    &self.pipelines[&PostPipelineKey {
                        effect: pass.effect,
                        blend: pass.blend,
                        vertex: pass.vertex,
                        cull: pass.cull,
                    }],
                );
                rp.set_bind_group(0, &bind_group, &[]);
                rp.draw(0..3, 0..1);
            }
        }
    }

    /// A self-contained render: builds its own input texture from `input_rgba` (tightly packed,
    /// `width*height*4` bytes), runs the chain, and reads the `"minecraft:main"` output back to
    /// the CPU. This is how `tests/s20_post.rs` proves the engine runs a *real* vanilla JSON
    /// end-to-end independently of the live-frame integration [`PostPhase`] provides.
    ///
    /// Requires a `gpu` with [`Gpu::sample_count`] of 1 — it supplies its own single-sampled
    /// output texture, which cannot serve as an MSAA colour attachment. With no live frame clock,
    /// the first pass's parsed `Time` value is used as the deterministic standalone sample; the
    /// normal [`PostChain::run`] path always replaces it with the engine-supplied frame time.
    ///
    /// # Panics
    /// If `input_rgba`'s length does not match `width * height * 4`, or if `gpu` is multisampled.
    pub fn run_standalone(
        &self,
        gpu: &Gpu,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        input_rgba: &[u8],
    ) -> Vec<u8> {
        assert_eq!(input_rgba.len(), (width * height * 4) as usize);
        assert_eq!(
            gpu.sample_count(),
            1,
            "run_standalone needs a single-sampled context"
        );

        let in_texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("post chain standalone input"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            input_rgba,
        );
        let in_view = in_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let out_target =
            PostTarget::new(gpu, width, height, format, "post chain standalone output");

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("post chain standalone"),
            });
        let standalone_time = self
            .passes
            .first()
            .map_or(0.0, |pass| pass.uniforms.scalars2[3]);
        self.run(gpu, &mut enc, &in_view, &out_target.view, standalone_time);

        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post chain standalone readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &out_target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll for readback");
        rx.recv().expect("map callback").expect("map succeeded");

        let view = staging.get_mapped_range(..).expect("mapped range");
        let mut out = Vec::with_capacity((unpadded * height) as usize);
        for row in 0..height as usize {
            let start = row * padded as usize;
            out.extend_from_slice(&view[start..start + unpadded as usize]);
        }
        drop(view);
        staging.unmap();
        out
    }
}

// ---------------------------------------------------------------- the live-frame seam

/// Runs a [`PostChain`] over the real frame. Register at [`crate::PhaseOrder::POST`].
///
/// This is the phase that closes S20's "post pipeline seam" gap: it opts into
/// [`crate::SceneTextures`], so [`crate::PhaseList`] renders the frame into an intermediate
/// target, hands this phase a readable snapshot of it, and blits the result back out. Without a
/// chain set it does nothing at all and — importantly — does **not** opt in, so a `PostPhase`
/// parked in the phase list with no active shader costs exactly one `enabled()` call per frame
/// and leaves the frame path byte-identical to having no `PostPhase` at all.
///
/// java: `EntityRenderer.loadShader`/`theShaderGroup.render(partialTicks)` — vanilla likewise
/// runs the whole group over the frame after the world is drawn and before the HUD.
pub struct PostPhase {
    chain: Option<PostChain>,
    /// Java's global upload-time texture transform, retained even while no shader group is
    /// installed so the next chain loaded by `EntityRenderer` inherits the live setting.
    anaglyph: bool,
    /// Set `false` to keep the chain loaded but skip it this frame (java: `EntityRenderer.
    /// shaderGroup != null && useShader`).
    pub visible: bool,
    /// This frame's render partial ticks. The live caller updates it before phase recording,
    /// exactly as the sky/fog phases' partial-tick inputs are driven.
    pub partial_ticks: f32,
    shader_time: ShaderTime,
}

impl PostPhase {
    /// An empty phase — no chain, nothing drawn.
    #[must_use]
    pub fn new(_gpu: &Gpu) -> Self {
        Self {
            chain: None,
            anaglyph: false,
            visible: true,
            partial_ticks: 0.0,
            shader_time: ShaderTime::default(),
        }
    }

    /// Install a chain. java: `EntityRenderer.loadShader(ResourceLocation)`.
    pub fn set_chain(&mut self, chain: PostChain) {
        let mut chain = chain;
        chain.set_anaglyph_state(self.anaglyph);
        self.chain = Some(chain);
        // A newly constructed vanilla ShaderGroup starts both clock fields at zero.
        self.shader_time = ShaderTime::default();
    }

    /// Drop the chain. java: `EntityRenderer.stopUseShader`.
    pub fn clear_chain(&mut self) {
        self.chain = None;
        self.shader_time = ShaderTime::default();
    }

    /// The installed chain, if any.
    #[must_use]
    pub fn chain(&self) -> Option<&PostChain> {
        self.chain.as_ref()
    }

    /// Whether Java's upload-time anaglyph transform is selected for this post phase.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's upload-time anaglyph transform for the installed chain's static auxiliary
    /// textures. If the phase is currently empty, the setting is retained for the next chain.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        self.anaglyph = enabled;
        if let Some(chain) = &mut self.chain {
            chain.set_anaglyph_with_gpu(gpu, enabled);
        }
    }
}

impl RenderPhase for PostPhase {
    fn name(&self) -> &'static str {
        "post/chain"
    }

    fn enabled(&self) -> bool {
        self.visible && self.chain.is_some()
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        if let Some(chain) = &mut self.chain {
            chain.resize(gpu, width, height);
        }
    }

    fn needs_scene_color(&self) -> bool {
        self.visible && self.chain.is_some()
    }

    fn record(
        &mut self,
        _gpu: &Gpu,
        _enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        // Only reachable if a caller drove this phase outside `PhaseList` (or toggled `visible`
        // between `needs_scene_color` and `record`). Without a snapshot to read there is nothing
        // correct to draw, so draw nothing rather than something wrong.
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        let time = self.shader_time.advance(self.partial_ticks);
        if let Some(chain) = &self.chain {
            chain.run(gpu, enc, scene.color, view, time);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuOptions;

    fn headless(w: u32, h: u32) -> Option<Gpu> {
        match Gpu::headless(
            w,
            h,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => Some(gpu),
            Err(crate::RenderError::NoAdapter) => {
                eprintln!("skip: no wgpu adapter available on this machine");
                None
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        }
    }

    #[test]
    fn entity_outline_gate_requires_every_vanilla_conjunct() {
        assert!(entity_outline_enabled(true, true, true, true));
        for missing in 0..4 {
            let mut args = [true; 4];
            args[missing] = false;
            assert!(!entity_outline_enabled(args[0], args[1], args[2], args[3]));
        }
    }

    /// A resource manager backing no packs at all — every one of these unit tests exercises the
    /// pass graph with hand-written specs, not real assets, so `PostChain::new`'s aux-texture
    /// fallback keeps the flat compatibility placeholder, while `shader_source_error` retains
    /// the exact Java-like failure. `tests/s20_post.rs` proves the real-jar path.
    fn empty_resources() -> ResourceManager {
        ResourceManager::new()
    }

    #[test]
    fn auxiliary_upload_matches_java_colored_byte_and_alpha_oracles() {
        // java: TextureUtil.anaglyphColor performs integer division by 100 and leaves alpha in
        // the high byte.  This is deliberately a coloured, translucent fixture rather than a
        // grayscale pixel, so both the channel weights and the alpha-preserving boundary are
        // exercised at once.
        let source = [255, 0, 0, 7, 0, 255, 0, 19, 0, 0, 255, 31];
        assert_eq!(anaglyph_rgba(&source, false), source);
        assert_eq!(
            anaglyph_rgba(&source, true),
            [76, 76, 76, 7, 178, 150, 0, 19, 0, 28, 178, 31]
        );
    }

    #[test]
    fn auxiliary_upload_toggle_replays_retained_raw_source() {
        // Replaying the raw source is the important live-toggle invariant.  Applying the helper
        // to the already converted upload would produce a different second transform, while
        // false must recover the original bytes exactly.
        let source = [17, 91, 201, 0, 123, 45, 67, 128];
        let enabled_upload = anaglyph_rgba(&source, true);
        assert_eq!(anaglyph_rgba(&source, false), source);
        assert_ne!(anaglyph_rgba(&enabled_upload, true), enabled_upload);
        assert_eq!(anaglyph_rgba(&source, true), enabled_upload);
    }

    #[test]
    fn native_auxiliary_post_pass_observes_toggle_and_restores_raw_pixels() {
        let Some(gpu) = headless(1, 1) else { return };
        let spec = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"outline_combine","intarget":"minecraft:main","outtarget":"minecraft:main",
                "auxtargets":[{"name":"OutlineSampler","id":"fixture","width":1,"height":1,"bilinear":false}]}]}"#,
        )
        .unwrap();
        let mut chain = PostChain::new(
            &gpu,
            &empty_resources(),
            &spec,
            crate::HEADLESS_FORMAT,
            (1, 1),
        );

        // Install a tiny decoded static aux fixture directly so this unit oracle does not need a
        // real jar or a second resource-pack test harness. It is still exercised by the native
        // `outline_combine` WGSL pass, rather than by the byte helper alone.
        chain.aux_textures.insert(
            "fixture".to_owned(),
            AuxTexture::new(&gpu, vec![255, 0, 0, 37], 1, 1, false),
        );
        chain.passes[0].auxtarget = Some(ResolvedAux {
            id: "fixture".to_owned(),
            width: 1,
            height: 1,
            bilinear: false,
        });

        let input = [128, 128, 128, 255];
        let raw = chain.run_standalone(&gpu, crate::HEADLESS_FORMAT, 1, 1, &input);
        chain.set_anaglyph_with_gpu(&gpu, true);
        let converted = chain.run_standalone(&gpu, crate::HEADLESS_FORMAT, 1, 1, &input);
        chain.set_anaglyph_with_gpu(&gpu, false);
        let restored = chain.run_standalone(&gpu, crate::HEADLESS_FORMAT, 1, 1, &input);

        assert_ne!(
            converted, raw,
            "native aux sampler ignored anaglyph upload bytes"
        );
        for (actual, expected) in restored.iter().zip(raw.iter()) {
            assert!(
                actual.abs_diff(*expected) <= 1,
                "raw aux upload was not restored: {restored:?} vs {raw:?}"
            );
        }
    }

    #[test]
    fn program_names_map_to_the_right_native_effect() {
        assert_eq!(PostEffect::from_program_name("invert"), PostEffect::Invert);
        assert_eq!(PostEffect::from_program_name("blur"), PostEffect::Blur);
        assert_eq!(PostEffect::from_program_name("flip"), PostEffect::Flip);
        assert_eq!(
            PostEffect::from_program_name("color_convolve"),
            PostEffect::ColorConvolve
        );
        assert_eq!(PostEffect::from_program_name("blit"), PostEffect::Blit);
        assert_eq!(
            PostEffect::from_program_name("some_unimplemented_glsl_effect"),
            PostEffect::Blit
        );
    }

    #[test]
    fn all_shipped_programs_are_native_and_no_passthroughs_remain() {
        for name in PostEffect::PASSTHROUGH_PROGRAMS {
            assert!(
                !PostEffect::is_native_program(name),
                "{name} is declared passthrough but maps to a native effect"
            );
        }
        assert!(PostEffect::PASSTHROUGH_PROGRAMS.is_empty());
        for name in [
            "fxaa",
            "ntsc_encode",
            "ntsc_decode",
            "outline_watercolor",
            "spider",
        ] {
            assert!(PostEffect::is_native_program(name), "{name} is not native");
        }
        // `green` and `desaturate` are *group* names with no program of the same name; they must
        // not be mistaken for either category.
        assert!(!PostEffect::is_native_program("green"));
        assert!(!PostEffect::is_native_program("desaturate"));
        assert!(PostEffect::is_native_program("blit"));
    }

    #[test]
    fn vertex_modes_and_translation_errors_follow_the_java_program_contract() {
        assert_eq!(PostEffect::Antialias.vertex_mode(), VertexMode::BottomLeft);
        assert_eq!(PostEffect::Blobs.vertex_mode(), VertexMode::BottomLeft);
        assert_eq!(PostEffect::Bumpy.vertex_mode(), VertexMode::BottomLeft);
        assert_eq!(PostEffect::Flip.vertex_mode(), VertexMode::Flip);
        assert_eq!(PostEffect::Fxaa.vertex_mode(), VertexMode::Fxaa);
        assert_eq!(PostEffect::Spider.vertex_mode(), VertexMode::RotateScale);
        assert_eq!(PostEffect::Blur.vertex_mode(), VertexMode::TopLeft);

        let translated = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"fxaa","intarget":"minecraft:main","outtarget":"minecraft:main"}]}"#,
        )
        .unwrap();
        assert!(validate_post_graph(&empty_resources(), &translated).is_ok());

        let unknown = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"custom","intarget":"minecraft:main","outtarget":"minecraft:main"}]}"#,
        )
        .unwrap();
        assert!(matches!(
            validate_post_graph(&empty_resources(), &unknown),
            Err(PostChainError::UnknownProgram { program }) if program == "custom"
        ));
    }

    #[test]
    fn graph_validation_preserves_java_auxiliary_and_target_failures() {
        let duplicate = ShaderGroupSpec::parse(
            r#"{"targets": [{"name":"swap"}, {"name":"swap"}], "passes": []}"#,
        )
        .unwrap();
        assert!(matches!(
            validate_post_graph(&empty_resources(), &duplicate),
            Err(PostChainError::DuplicateTarget { target }) if target == "swap"
        ));

        let missing_target = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"blit","intarget":"missing","outtarget":"minecraft:main"}]}"#,
        )
        .unwrap();
        assert!(matches!(
            validate_post_graph(&empty_resources(), &missing_target),
            Err(PostChainError::MissingTarget { program, target, direction })
                if program == "blit" && target == "missing" && direction == "input"
        ));

        let missing_aux_dimensions = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"notch","intarget":"minecraft:main","outtarget":"minecraft:main",
                "auxtargets":[{"name":"DitherSampler","id":"dither"}]}]}"#,
        )
        .unwrap();
        assert!(matches!(
            validate_post_graph(&empty_resources(), &missing_aux_dimensions),
            Err(PostChainError::MissingAuxiliaryDimensions { program, sampler, id })
                if program == "notch" && sampler == "DitherSampler" && id == "dither"
        ));

        let missing_aux_sampler = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"notch","intarget":"minecraft:main","outtarget":"minecraft:main"}]}"#,
        )
        .unwrap();
        assert!(matches!(
            validate_post_graph(&empty_resources(), &missing_aux_sampler),
            Err(PostChainError::MissingAuxiliarySampler { program, sampler })
                if program == "notch" && sampler == "DitherSampler"
        ));

        let missing_aux_file = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"notch","intarget":"minecraft:main","outtarget":"minecraft:main",
                "auxtargets":[{"name":"DitherSampler","id":"dither","width":4,"height":4,"bilinear":false}]}]}"#,
        )
        .unwrap();
        assert!(matches!(
            validate_post_graph(&empty_resources(), &missing_aux_file),
            Err(PostChainError::AuxiliaryTexture { program, sampler, id, location, .. })
                if program == "notch"
                    && sampler == "DitherSampler"
                    && id == "dither"
                    && location.path == "textures/effect/dither.png"
        ));

        let missing_aux_filter = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"notch","intarget":"minecraft:main","outtarget":"minecraft:main",
                "auxtargets":[{"name":"DitherSampler","id":"dither","width":4,"height":4}]}]}"#,
        )
        .unwrap();
        assert!(matches!(
            validate_post_graph(&empty_resources(), &missing_aux_filter),
            Err(PostChainError::MissingAuxiliaryFilter { program, sampler, id })
                if program == "notch" && sampler == "DitherSampler" && id == "dither"
        ));

        let unknown_uniform = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"blit","intarget":"minecraft:main","outtarget":"minecraft:main",
                "uniforms":[{"name":"NotDeclared","values":[1.0]}]}]}"#,
        )
        .unwrap();
        assert!(matches!(
            validate_post_graph(&empty_resources(), &unknown_uniform),
            Err(PostChainError::UnknownUniform { program, uniform })
                if program == "blit" && uniform == "NotDeclared"
        ));
    }

    #[test]
    fn every_effect_has_a_distinct_entry_point() {
        let mut seen = std::collections::HashSet::new();
        for e in PostEffect::ALL {
            assert!(seen.insert(e.entry_point()), "duplicate {e:?}");
        }
        assert_eq!(seen.len(), 25);
    }

    #[test]
    fn parsed_combined_blend_reaches_both_wgpu_components() {
        let state = pipeline_blend_state(BlendSpec {
            enabled: true,
            separate_alpha: false,
            func: BlendEquation::Add,
            srcrgb: BlendFactor::SrcAlpha,
            dstrgb: BlendFactor::OneMinusSrcAlpha,
            srcalpha: BlendFactor::One,
            dstalpha: BlendFactor::Zero,
        })
        .expect("enabled valid blend");

        // Java's non-separate constructor calls `glBlendFunc(srcRgb, dstRgb)`: OpenGL applies
        // those same two factors to alpha, not the parser's default alpha pair.
        assert_eq!(state.color.operation, wgpu::BlendOperation::Add);
        assert_eq!(state.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(state.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(state.alpha, state.color);
    }

    #[test]
    fn parsed_separate_blend_keeps_the_independent_alpha_component() {
        let state = pipeline_blend_state(BlendSpec {
            enabled: true,
            separate_alpha: true,
            func: BlendEquation::ReverseSubtract,
            srcrgb: BlendFactor::One,
            dstrgb: BlendFactor::Zero,
            srcalpha: BlendFactor::SrcAlpha,
            dstalpha: BlendFactor::OneMinusDstAlpha,
        })
        .expect("enabled valid separate blend");

        assert_eq!(state.color.operation, wgpu::BlendOperation::ReverseSubtract);
        assert_eq!(state.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(state.color.dst_factor, wgpu::BlendFactor::Zero);
        assert_eq!(state.alpha.operation, wgpu::BlendOperation::ReverseSubtract);
        assert_eq!(state.alpha.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(state.alpha.dst_factor, wgpu::BlendFactor::OneMinusDstAlpha);
    }

    #[test]
    fn disabled_or_invalid_parsed_modes_do_not_make_a_valid_wgpu_blend_state() {
        assert_eq!(pipeline_blend_state(BlendSpec::default()), None);
        assert_eq!(
            pipeline_blend_state(BlendSpec {
                enabled: true,
                srcrgb: BlendFactor::Invalid,
                ..BlendSpec::default()
            }),
            None,
            "the deliberate vanilla -1 sentinel cannot be invented as a wgpu factor"
        );
    }

    #[test]
    fn a_two_pass_group_reports_two_passes() {
        let spec = ShaderGroupSpec::parse(
            r#"{
                "targets": ["swap"],
                "passes": [
                    {"name":"blur","intarget":"minecraft:main","outtarget":"swap"},
                    {"name":"blur","intarget":"swap","outtarget":"minecraft:main"}
                ]
            }"#,
        )
        .unwrap();
        let Some(gpu) = headless(16, 16) else { return };
        let chain = PostChain::new(
            &gpu,
            &empty_resources(),
            &spec,
            crate::HEADLESS_FORMAT,
            (16, 16),
        );
        assert_eq!(chain.pass_count(), 2);
        // `blur` is now a real translation, so both passes are native.
        assert_eq!(chain.native_pass_count(), 2);
        assert_eq!(chain.passthrough_pass_count(), 0);
    }

    #[test]
    fn translated_program_is_recorded_as_native() {
        let spec = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"fxaa","intarget":"minecraft:main","outtarget":"minecraft:main"}]}"#,
        )
        .unwrap();
        let Some(gpu) = headless(4, 4) else { return };
        let chain = PostChain::new(
            &gpu,
            &empty_resources(),
            &spec,
            crate::HEADLESS_FORMAT,
            (4, 4),
        );
        assert_eq!(chain.passthrough_pass_count(), 0);
        assert_eq!(chain.native_pass_count(), 1);
        assert!(chain.shader_source_error().is_none());
    }

    #[test]
    fn remaining_vanilla_programs_build_and_render_as_native_wgsl() {
        let Some(gpu) = headless(4, 4) else { return };
        let specs = [
            r#"{"name":"fxaa","intarget":"minecraft:main","outtarget":"minecraft:main"}"#,
            r#"{"name":"ntsc_encode","intarget":"minecraft:main","outtarget":"minecraft:main"}"#,
            r#"{"name":"ntsc_decode","intarget":"minecraft:main","outtarget":"minecraft:main","auxtargets":[{"name":"BaseSampler","id":"minecraft:main"}]}"#,
            r#"{"name":"outline_watercolor","intarget":"minecraft:main","outtarget":"minecraft:main"}"#,
            r#"{"name":"spider","intarget":"minecraft:main","outtarget":"minecraft:main","auxtargets":[{"name":"BlurSampler","id":"minecraft:main"}],"uniforms":[{"name":"InScale","values":[0.0,0.0]},{"name":"InOffset","values":[0.0,0.0]},{"name":"Scissor","values":[0.0,0.0,1.0,1.0]},{"name":"Vignette","values":[0.0,0.0,1.0,1.0]}]}"#,
        ];
        let mut input = vec![0u8; 4 * 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                let offset = ((y * 4 + x) * 4) as usize;
                let value = if x >= y { 255 } else { 0 };
                input[offset..offset + 4].copy_from_slice(&[value, value, value, 255]);
            }
        }
        for json in specs {
            let spec = ShaderGroupSpec::parse(&format!("{{\"targets\":[],\"passes\":[{json}]}}"))
                .expect("remaining program spec");
            let chain = PostChain::new(
                &gpu,
                &empty_resources(),
                &spec,
                crate::HEADLESS_FORMAT,
                (4, 4),
            );
            assert_eq!(chain.native_pass_count(), 1, "{json}");
            assert_eq!(chain.passthrough_pass_count(), 0, "{json}");
            let solid = [20u8, 80, 160, 255].repeat(16);
            let test_input = if json.contains("ntsc_encode")
                || json.contains("ntsc_decode")
                || json.contains("outline_watercolor")
            {
                &solid
            } else {
                &input
            };
            let output = chain.run_standalone(&gpu, crate::HEADLESS_FORMAT, 4, 4, test_input);
            if json.contains("fxaa") {
                // A diagonal luma edge exercises the shifted NW/NE/SW/SE taps; the GLSL oracle
                // selects the B blend at this edge, which quantizes to approximately 127.
                assert!(
                    output[16].abs_diff(127) <= 2,
                    "fxaa edge: {:?}",
                    &output[..32]
                );
            }
            if json.contains("ntsc_encode") {
                assert_eq!(&output[..4], &[93, 111, 128, 126]);
            }
            if json.contains("ntsc_decode") {
                assert_eq!(&output[..4], &[0, 0, 255, 255]);
            }
            if json.contains("outline_watercolor") {
                assert_eq!(&output[..4], &[0, 0, 0, 255]);
            }
            if json.contains("spider") {
                // InScale=(0,0), InOffset=(0,0) maps every fragment to the top-left source texel.
                assert_eq!(&output[16..20], &[255, 255, 255, 255]);
            }
            assert_eq!(output.len(), input.len(), "{json}");
        }
    }

    #[test]
    fn invert_chain_actually_inverts_pixels_on_the_gpu() {
        // `invert.fsh` is `mix(c, 1 - c, InverseAmount)`, and the program's declared default for
        // `InverseAmount` is 0.0 — so a bare invert pass is a no-op until the group overrides it,
        // which is exactly what the real `post/invert.json` does.
        let spec = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"invert","intarget":"minecraft:main","outtarget":"minecraft:main",
                "uniforms":[{"name":"InverseAmount","values":[1.0]}]}]}"#,
        )
        .unwrap();
        let Some(gpu) = headless(4, 4) else { return };
        let chain = PostChain::new(
            &gpu,
            &empty_resources(),
            &spec,
            crate::HEADLESS_FORMAT,
            (4, 4),
        );
        assert_eq!(chain.native_pass_count(), 1);

        let input = [64u8, 128, 200, 255].repeat(16); // one solid colour, 4x4
        let output = chain.run_standalone(&gpu, crate::HEADLESS_FORMAT, 4, 4, &input);

        let px = &output[0..4];
        assert!(px[0].abs_diff(255 - 64) <= 2, "{px:?}");
        assert!(px[1].abs_diff(255 - 128) <= 2, "{px:?}");
        assert!(px[2].abs_diff(255 - 200) <= 2, "{px:?}");
    }

    #[test]
    fn blit_chain_passes_pixels_through_unchanged() {
        let spec = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"blit","intarget":"minecraft:main","outtarget":"minecraft:main"}]}"#,
        )
        .unwrap();
        let Some(gpu) = headless(4, 4) else { return };
        let chain = PostChain::new(
            &gpu,
            &empty_resources(),
            &spec,
            crate::HEADLESS_FORMAT,
            (4, 4),
        );

        let input = [10u8, 20, 30, 255].repeat(16);
        let output = chain.run_standalone(&gpu, crate::HEADLESS_FORMAT, 4, 4, &input);
        let px = &output[0..4];
        assert!(px[0].abs_diff(10) <= 2 && px[1].abs_diff(20) <= 2 && px[2].abs_diff(30) <= 2);
    }

    #[test]
    fn two_pass_ping_pong_through_a_named_target_runs_end_to_end() {
        // A real structural shape (java: blur.json): proves target allocation + ping-pong.
        let spec = ShaderGroupSpec::parse(
            r#"{
                "targets": ["swap"],
                "passes": [
                    {"name":"invert","intarget":"minecraft:main","outtarget":"swap",
                     "uniforms":[{"name":"InverseAmount","values":[1.0]}]},
                    {"name":"invert","intarget":"swap","outtarget":"minecraft:main",
                     "uniforms":[{"name":"InverseAmount","values":[1.0]}]}
                ]
            }"#,
        )
        .unwrap();
        let Some(gpu) = headless(4, 4) else { return };
        let chain = PostChain::new(
            &gpu,
            &empty_resources(),
            &spec,
            crate::HEADLESS_FORMAT,
            (4, 4),
        );

        // Inverting twice must return (approximately) the original colour.
        let input = [50u8, 90, 150, 255].repeat(16);
        let output = chain.run_standalone(&gpu, crate::HEADLESS_FORMAT, 4, 4, &input);
        let px = &output[0..4];
        assert!(
            px[0].abs_diff(50) <= 3 && px[1].abs_diff(90) <= 3 && px[2].abs_diff(150) <= 3,
            "{px:?}"
        );
    }

    #[test]
    fn color_convolve_with_a_zero_saturation_override_is_a_greyscale_pass() {
        // java: post/desaturate.json overrides Saturation to 0, which collapses the chroma term
        // and leaves the Gray-weighted luma — the exact mechanism vanilla's greyscale uses.
        let spec = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"color_convolve","intarget":"minecraft:main","outtarget":"minecraft:main",
                "uniforms":[{"name":"Saturation","values":[0.0]}]}]}"#,
        )
        .unwrap();
        let Some(gpu) = headless(2, 2) else { return };
        let chain = PostChain::new(
            &gpu,
            &empty_resources(),
            &spec,
            crate::HEADLESS_FORMAT,
            (2, 2),
        );
        let input = [200u8, 100, 50, 255].repeat(4);
        let output = chain.run_standalone(&gpu, crate::HEADLESS_FORMAT, 2, 2, &input);
        // luma = 200*0.3 + 100*0.59 + 50*0.11 = 60 + 59 + 5.5 = 124.5
        let px = &output[0..4];
        assert!(px[0].abs_diff(124) <= 2, "{px:?}");
        assert_eq!(px[0], px[1], "{px:?}");
        assert_eq!(px[1], px[2], "{px:?}");
    }

    #[test]
    fn pass_uniforms_layer_program_defaults_under_group_overrides() {
        let spec = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"color_convolve","intarget":"minecraft:main","outtarget":"minecraft:main",
                "uniforms":[{"name":"GreenMatrix","values":[0.0, 2.0, 0.0]}]}]}"#,
        )
        .unwrap();
        let u = PostUniforms::for_pass(&spec.passes[0]);
        // Overridden.
        assert_eq!(u.green_matrix, [0.0, 2.0, 0.0, 0.0]);
        // Untouched program defaults survive.
        assert_eq!(u.red_matrix, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(u.gray, [0.3, 0.59, 0.11, 0.0]);
        assert!((u.scalars2[0] - 1.8).abs() < 1e-6, "{:?}", u.scalars2);
    }

    #[test]
    fn short_uniform_overrides_preserve_java_unwritten_components() {
        // java: ShaderGroup.initUniform selects ShaderUniform.set's overload from the JSON
        // value count.  A one-component vector update leaves the remaining components at their
        // program defaults, and an empty array selects no overload at all.
        let spec = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"blit","intarget":"minecraft:main","outtarget":"minecraft:main",
                "uniforms":[{"name":"ColorModulate","values":[0.25]}, {"name":"ColorModulate","values":[]}] }]}"#,
        )
        .unwrap();
        let u = PostUniforms::for_pass(&spec.passes[0]);
        assert_eq!(u.color_modulate, [0.25, 1.0, 1.0, 1.0]);

        let bits = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"bits","intarget":"minecraft:main","outtarget":"minecraft:main",
                "uniforms":[{"name":"MosaicSize","values":[]}] }]}"#,
        )
        .unwrap();
        let bits_uniforms = PostUniforms::for_pass(&bits.passes[0]);
        assert_eq!(bits_uniforms.scalars[3], 8.0);
    }

    #[test]
    fn shader_time_matches_vanilla_across_tick_and_twenty_tick_rollovers() {
        // Java oracle, ShaderGroup.loadShaderGroup:
        // 0.00 -> 0.25 advances 0.25 ticks; 0.25 -> 0.75 advances 0.50; and a falling
        // 0.75 -> 0.25 sample crosses the tick boundary and advances another 0.50.
        let mut time = ShaderTime::default();
        assert!((time.advance(0.25) - 0.0125).abs() < f32::EPSILON);
        assert!((time.advance(0.75) - 0.0375).abs() < f32::EPSILON);
        assert!((time.advance(0.25) - 0.0625).abs() < f32::EPSILON);

        // Vanilla wraps only while the accumulator is strictly greater than 20. Exactly 20
        // reaches Time=1.0 for one sample; the next quarter tick wraps it back to 0.0125.
        let mut boundary = ShaderTime {
            elapsed_ticks: 19.75,
            previous_partial_ticks: 0.75,
        };
        assert!((boundary.advance(0.0) - 1.0).abs() < f32::EPSILON);
        assert!((boundary.advance(0.25) - 0.0125).abs() < f32::EPSILON);
    }

    #[test]
    fn runtime_time_replaces_program_and_group_values_after_parsing() {
        // The real wobble program declares Time=0.0. Use a deliberately conflicting group
        // override to prove the engine-supplied per-frame value wins over both static layers,
        // in the same order as Shader.loadShader's final `Time.set(frameTime)` call.
        let spec = ShaderGroupSpec::parse(
            r#"{"targets": [], "passes": [{"name":"wobble","intarget":"minecraft:main","outtarget":"minecraft:main",
                "uniforms":[{"name":"Time","values":[0.75]}]}]}"#,
        )
        .unwrap();
        let parsed = PostUniforms::for_pass(&spec.passes[0]);
        assert!((parsed.scalars2[3] - 0.75).abs() < f32::EPSILON);

        let frame = parsed.for_frame(0.25);
        assert!((frame.scalars2[3] - 0.25).abs() < f32::EPSILON);
        assert_eq!(frame.blur_dir[2..4], [512.0, 288.0]);
    }

    #[test]
    fn post_uniforms_is_std140_sized() {
        assert_eq!(std::mem::size_of::<PostUniforms>(), 17 * 16);
        assert_eq!(std::mem::size_of::<PostUniforms>() % 16, 0);
    }

    #[test]
    fn a_post_phase_without_a_chain_never_asks_for_scene_input() {
        let Some(gpu) = headless(8, 8) else { return };
        let phase = PostPhase::new(&gpu);
        assert!(!RenderPhase::enabled(&phase));
        assert!(!RenderPhase::needs_scene_color(&phase));
    }
}
