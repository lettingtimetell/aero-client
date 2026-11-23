//! GPU context: instance, adapter, device, queue, surface, depth buffer, MSAA.
//!
//! This is the S10 half of the `mc-render` seam. [`Gpu`] is created once at startup and handed
//! to every [`RenderPhase`](crate::pipeline::RenderPhase) on every frame; phases never create
//! their own device, queue, or swap chain.
//!
//! By design nothing here is a transliteration of `client/renderer/`'s fixed-function GL —
//! `GlStateManager`, `OpenGlHelper` and friends have no counterpart. Render state lives in
//! pipeline objects, which each phase owns.

// `RenderError` intentionally keeps adapter facts inline for diagnostics. Boxing this public error
// or changing every constructor's result type would alter the renderer's error API.
#![allow(clippy::result_large_err)]

use std::sync::Arc;

use crate::RenderError;

/// Depth-stencil format shared by every phase.
///
/// Vanilla asks LWJGL for a 24-bit depth buffer; we use `Depth32Float` because it is the format
/// wgpu guarantees on every backend and it removes the z-fighting the 1.8.9 renderer has at long
/// render distances. Phases that build a `RenderPipeline` with a depth attachment **must** use
/// this format or pass creation will fail validation.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Colour format used by [`Gpu::headless`], where there is no surface to negotiate one with.
pub const HEADLESS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Environment variable that turns on GPU timestamp queries.
///
/// Deliberately **separate** from `mc_core`'s `MC_PROFILE`, for two reasons. The mundane one is
/// that `mc_core::profile` sits behind that crate's `subscriber` feature, which `mc-render` does
/// not enable and whose `Cargo.toml` this change does not own. The substantive one is that this
/// is a heavier instrument than the CPU profiler: it enables two device features, allocates a
/// query set and a ring of mappable buffers, and adds a non-blocking `device.poll` to every
/// frame. Being able to run the CPU profile *without* perturbing the GPU path — and thereby to
/// tell what the GPU instrument itself costs — is worth its own switch.
///
/// Same grammar as `MC_PROFILE`: on unless unset, empty, `0`, or `false`.
pub const GPU_TIMING_ENV: &str = "MC_GPU_TIMING";

/// Whether [`GPU_TIMING_ENV`] asks for timestamp queries. Read once and cached, like
/// `mc_core::profile::enabled`.
#[must_use]
pub fn gpu_timing_requested() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(GPU_TIMING_ENV) {
        Ok(value) => {
            let value = value.trim();
            !(value.is_empty() || value == "0" || value.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    })
}

/// Whether a render pass binds the shared depth buffer.
///
/// Note that *writing* depth is a pipeline setting (`DepthStencilState::depth_write_enabled`),
/// not a pass setting: a translucent phase attaches [`DepthAttachment::Shared`] and disables
/// depth writes in its own pipeline, exactly as vanilla does with `glDepthMask(false)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthAttachment {
    /// Bind the shared depth buffer: load what previous phases wrote, store what this one writes.
    Shared,
    /// Bind the shared depth buffer after clearing it to the far plane. This is the hand pass's
    /// `glClear(GL_DEPTH_BUFFER_BIT)` seam: colour remains loaded for first-person rendering.
    Clear,
    /// No depth-stencil attachment at all — for screen-space work (UI, post effects).
    None,
}

/// Tunables read from [`GameSettings`](crate::settings::GameSettings) when the context is built.
#[derive(Debug, Clone, Copy)]
pub struct GpuOptions {
    /// Wait for vertical blank. Maps to `PresentMode::AutoVsync` / `AutoNoVsync`.
    pub vsync: bool,
    /// MSAA sample count: 1, 2, 4, or 8. Unsupported counts silently fall back to 1.
    pub sample_count: u32,
    /// Prefer a low-power adapter (laptops on battery). Vanilla has no such option.
    pub low_power: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            vsync: true,
            sample_count: 1,
            low_power: false,
        }
    }
}

/// Which swap-chain present mode to ask the surface for.
///
/// # Why this is separate from [`GpuOptions::vsync`]
///
/// `vsync` is vanilla's `enableVsync`, a boolean, and it can only ever say "block on vblank" or
/// "do not". That maps onto `AutoVsync`/`AutoNoVsync`, and on this rig `AutoVsync` resolves to
/// `FifoRelaxed` — a mode whose `acquire` **blocks** until the presentation engine has a free
/// image. That block was measured at 2.239 ms of a 2.783 ms frame, 80.5 %.
///
/// `Mailbox` is the third option a boolean cannot express: the swap chain keeps a queue of
/// rendered images and shows the newest at vblank, so presentation is still tear-free but
/// `acquire` does not have to wait for the display. It is not universally supported, which is
/// exactly why this is a *preference* and not an instruction — see
/// [`resolve_present_mode`] for the fallback chains, which are chosen so that asking for a
/// tear-free mode can never silently land you on a tearing one.
///
/// [`PresentPreference::Auto`] is the default and reproduces the behaviour that existed before
/// this type did, bit for bit. Any other value **overrides** `enableVsync` for presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PresentPreference {
    /// Derive the mode from [`GpuOptions::vsync`], as the context always has:
    /// `AutoVsync` when vsync is on, `AutoNoVsync` when it is off.
    #[default]
    Auto,
    /// `Fifo` — classic vsync. Blocks on vblank, never tears, guaranteed supported everywhere.
    Fifo,
    /// `FifoRelaxed` — `Fifo`, except a frame that misses vblank is shown immediately (and may
    /// tear) rather than waiting a whole refresh. What `AutoVsync` resolves to on this rig.
    FifoRelaxed,
    /// `Mailbox` — tear-free, non-blocking. The T3.2 candidate.
    Mailbox,
    /// `Immediate` — present as soon as the frame is ready. Tears. Lowest latency.
    Immediate,
}

/// The swap chain's `desired_maximum_frame_latency` the windowed path has always run with.
///
/// `Surface::get_default_config` returns 2, and the windowed context never overrode it, so 2 is
/// the "preserve current behaviour" value for the non-vanilla `frameLatency` option. wgpu clamps
/// the request to what the surface supports: on Vulkan it becomes `minImageCount = latency + 1`
/// (so 1 means double buffering and some drivers round it back up to 2); on DX12 it is
/// `IDXGISwapChain2::SetMaximumFrameLatency` plus a wait on the frame-latency waitable object in
/// `get_current_texture`, i.e. a hard bound on how far the CPU may run ahead of the display.
///
/// **1 is the low-latency setting.** It removes up to one frame of queued input-to-photon latency
/// when the GPU (or vsync) is the bottleneck, at the price of less CPU/GPU overlap — expect lower
/// average FPS in GPU-bound scenes. Pair it with `Immediate` or `Fifo`; with `Mailbox` keep 2 or
/// more, because a two-image mailbox has no spare image and blocks like `Fifo`.
pub const DEFAULT_FRAME_LATENCY: u32 = 2;

/// The largest `frameLatency` this client asks for. More frames in flight only add latency.
pub const MAX_FRAME_LATENCY: u32 = 3;

/// Clamp a configured frame latency into `1..=MAX_FRAME_LATENCY`. `0` (and anything unparsable
/// that reached here) means "not set" and keeps [`DEFAULT_FRAME_LATENCY`].
#[must_use]
pub fn clamp_frame_latency(requested: u32) -> u32 {
    if requested == 0 {
        DEFAULT_FRAME_LATENCY
    } else {
        requested.min(MAX_FRAME_LATENCY)
    }
}

/// Environment variable that forces one wgpu backend for the windowed context: `vulkan`, `dx12`,
/// or `auto` (the default, unset). A measurement switch for the Windows Vulkan-vs-DX12 question
/// (PRESENT-5); see [`BackendOverride`].
pub const GPU_BACKEND_ENV: &str = "MC_GPU_BACKEND";

/// Environment variable that lowers the device's `max_buffer_size` to this many
/// MiB: a test switch to run this PC with a smaller GPU's
/// limits. Unset, empty, zero or not a number: the adapter's own limits.
pub const LIMIT_BUFFER_MB_ENV: &str = "MC_GPU_LIMIT_BUFFER_MB";

/// [`LIMIT_BUFFER_MB_ENV`] in bytes, read when a device is requested (once per context).
fn limit_buffer_bytes_from_env() -> Option<u64> {
    parse_limit_mb(&std::env::var(LIMIT_BUFFER_MB_ENV).ok()?)
}

/// The parse half of [`limit_buffer_bytes_from_env`].
fn parse_limit_mb(raw: &str) -> Option<u64> {
    let mb: u64 = raw.trim().parse().ok()?;
    (mb > 0).then(|| mb.saturating_mul(1024 * 1024))
}

/// Which backend [`Gpu::for_window_with`] opens first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendOverride {
    /// The historical policy: Vulkan alone when it reaches a discrete GPU, otherwise every
    /// backend and wgpu's own choice. Byte-for-byte the pre-existing behaviour.
    #[default]
    Auto,
    /// Vulkan only (any adapter type). Falls back to every backend if Vulkan finds nothing.
    Vulkan,
    /// DirectX 12 only (any adapter type). Falls back to every backend if DX12 finds nothing,
    /// which is always the case off Windows.
    Dx12,
}

impl BackendOverride {
    /// Parse a [`GPU_BACKEND_ENV`] value. Case-insensitive; `None` for anything unrecognised so
    /// the caller keeps [`BackendOverride::Auto`].
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Self::Auto),
            "vulkan" | "vk" => Some(Self::Vulkan),
            "dx12" | "d3d12" => Some(Self::Dx12),
            _ => None,
        }
    }

    /// Read [`GPU_BACKEND_ENV`] once. Unset or unrecognised means [`BackendOverride::Auto`].
    #[must_use]
    pub fn from_env() -> Self {
        std::env::var(GPU_BACKEND_ENV)
            .ok()
            .and_then(|value| {
                let parsed = Self::parse(&value);
                if parsed.is_none() {
                    tracing::warn!(value = %value, "unrecognised {GPU_BACKEND_ENV}; using auto");
                }
                parsed
            })
            .unwrap_or_default()
    }

    /// The backends this override opens first, or `None` for [`BackendOverride::Auto`].
    #[must_use]
    pub fn backends(self) -> Option<wgpu::Backends> {
        match self {
            Self::Auto => None,
            Self::Vulkan => Some(wgpu::Backends::VULKAN),
            Self::Dx12 => Some(wgpu::Backends::DX12),
        }
    }
}

/// After an acquire whose reconfigure-and-retry also failed, how long [`Gpu::acquire`] waits
/// before reconfiguring again. `Surface::configure` drains the GPU (`wait_indefinitely`) and
/// rebuilds the swap chain, so retrying it on every loop of an occluded window is a stall per
/// iteration. Size changes do not wait for this: [`Gpu::resize`] reconfigures directly.
const FAILED_RECONFIGURE_BACKOFF: std::time::Duration = std::time::Duration::from_millis(100);

impl PresentPreference {
    /// Every value, in `options.txt` order — for the video options screen and for tests.
    pub const ALL: [Self; 5] = [
        Self::Auto,
        Self::Fifo,
        Self::FifoRelaxed,
        Self::Mailbox,
        Self::Immediate,
    ];

    /// The spelling used in `options.txt`. Lower-case and stable; do not change it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fifo => "fifo",
            Self::FifoRelaxed => "fifo_relaxed",
            Self::Mailbox => "mailbox",
            Self::Immediate => "immediate",
        }
    }

    /// Parse an `options.txt` value. Case- and separator-insensitive (`fifoRelaxed`,
    /// `fifo-relaxed` and `fifo_relaxed` all work); `None` for anything unrecognised, so the
    /// caller can keep its default rather than fail the load.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        let normalised: String = value
            .trim()
            .chars()
            .filter(|c| *c != '_' && *c != '-' && *c != ' ')
            .flat_map(char::to_lowercase)
            .collect();
        match normalised.as_str() {
            "auto" => Some(Self::Auto),
            "fifo" => Some(Self::Fifo),
            "fiforelaxed" => Some(Self::FifoRelaxed),
            "mailbox" => Some(Self::Mailbox),
            "immediate" => Some(Self::Immediate),
            _ => None,
        }
    }

    /// Whether this preference guarantees a tear-free image on every frame it presents.
    ///
    /// `FifoRelaxed` is deliberately **not** tear-free: its whole point is that a late frame is
    /// shown immediately instead of waiting for the next vblank, and that is a tear.
    #[must_use]
    pub fn is_tear_free(self) -> bool {
        matches!(self, Self::Fifo | Self::Mailbox)
    }
}

/// A colour target acquired for one frame.
///
/// Dropping this without calling [`AcquiredFrame::present`] is legal (the frame is discarded);
/// the windowed path always presents.
pub struct AcquiredFrame {
    view: wgpu::TextureView,
    surface_texture: Option<wgpu::SurfaceTexture>,
    /// The frame as recorded, when its swap-chain image is not on this thread
    /// ([`Self::from_recorded_target`]).
    recorded: Option<wgpu::Texture>,
}

impl AcquiredFrame {
    /// The view phases render into. With MSAA enabled this is the *resolve* target, not the
    /// multisampled attachment — [`Gpu::begin_pass`] handles that distinction for you.
    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Copy the resolved colour target into tightly packed RGBA8 rows before presentation.
    ///
    /// A window frame owns its [`wgpu::SurfaceTexture`] until [`Self::present`] consumes it, so
    /// this is the one safe readback seam for screenshots: the phase list has already rendered
    /// (including post/UI composition), while the compositor has not seen the image yet.  The
    /// headless variant delegates to [`Gpu::read_pixels`].
    pub fn read_pixels(&self, gpu: &Gpu) -> Result<Vec<u8>, RenderError> {
        // The recorded frame holds the image's bytes.
        if let Some(recorded) = &self.recorded {
            return gpu.read_texture_pixels(recorded);
        }
        let Some(surface_texture) = &self.surface_texture else {
            return gpu.read_pixels();
        };
        gpu.read_texture_pixels(&surface_texture.texture)
    }

    /// The colour texture behind this frame, for a caller that records its own copy of it
    /// without blocking (the asynchronous F2 screenshot): the
    /// acquired surface texture, or a headless context's offscreen target. Like
    /// [`Self::read_pixels`] it holds the resolved, composited image until [`Self::present`].
    /// Its size and format are `gpu.config`'s; copying needs `COPY_SRC` in `gpu.config.usage`.
    #[must_use]
    pub fn color_texture<'a>(&'a self, gpu: &'a Gpu) -> Option<&'a wgpu::Texture> {
        // The recorded frame holds the image's bytes.
        if let Some(recorded) = &self.recorded {
            return Some(recorded);
        }
        match &self.surface_texture {
            Some(surface_texture) => Some(&surface_texture.texture),
            None => gpu.offscreen.as_ref(),
        }
    }

    /// Hand the frame to the compositor. A no-op for a headless context, whose target is just an
    /// ordinary texture that [`Gpu::read_pixels`] can copy back.
    pub fn present(self, queue: &wgpu::Queue) {
        if let Some(tex) = self.surface_texture {
            queue.present(tex);
        }
    }

    /// Wrap a surface texture acquired off the main thread (the pipelined present thread).
    pub(crate) fn from_surface_texture(tex: wgpu::SurfaceTexture) -> Self {
        let view = tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            view,
            surface_texture: Some(tex),
            recorded: None,
        }
    }

    /// A frame whose swap-chain image is not on this thread: the pipelined
    /// present thread copies `texture`, the frame as recorded (the image's size and format), into
    /// the image. The copy is exact, so a reader of this frame
    /// ([`Self::read_pixels`], [`Self::color_texture`]) gets the bytes the image will hold; a
    /// readback submitted through the queue is ordered after the frame's own commands and before
    /// anything recorded later. Presenting it is a no-op: the present thread presents the image.
    pub(crate) fn from_recorded_target(texture: wgpu::Texture, view: wgpu::TextureView) -> Self {
        Self {
            view,
            surface_texture: None,
            recorded: Some(texture),
        }
    }

    /// Give the surface texture up unpresented, for the pipelined present thread to present.
    /// `None` for a headless frame.
    pub(crate) fn into_surface_texture(self) -> Option<wgpu::SurfaceTexture> {
        self.surface_texture
    }
}

/// A wgpu instance that opens only `backends`.
fn new_instance(backends: wgpu::Backends) -> wgpu::Instance {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    // wgpu checks indirect draw arguments in a compute pass by default, even in a release build.
    // The only indirect draws here are terrain's, whose arguments it builds itself from the ranges
    // it uploaded, so release builds skip that work. Debug builds and tests keep the check.
    if !cfg!(debug_assertions) {
        descriptor
            .flags
            .remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    }
    // wgpu's own debugging switches (`WGPU_VALIDATION`, `WGPU_DEBUG`,
    // `WGPU_GPU_BASED_VALIDATION`, ...) apply when set; unset (always, for players) the flags stay
    // exactly as above. D3D12 GPU-based validation is the strongest local check for GPU faults
    // that one vendor's hardware tolerates and another's does not (the suggestion).
    descriptor.flags = descriptor.flags.with_env();
    wgpu::Instance::new(descriptor)
}

/// The adapter `instance` offers, if it is a discrete GPU.
///
/// Opening every backend starts OpenGL and D3D12 as well as Vulkan. Their threads then idle for
/// the life of the process, because wgpu prefers Vulkan whenever it reaches a discrete GPU. So
/// that case opens Vulkan alone, and anything else (an integrated GPU, a software adapter, no
/// Vulkan at all) falls back to the full choice among every backend.
fn discrete_adapter(
    instance: &wgpu::Instance,
    power_preference: wgpu::PowerPreference,
    compatible_surface: Option<&wgpu::Surface<'_>>,
) -> Option<wgpu::Adapter> {
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference,
        compatible_surface,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    (adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu).then_some(adapter)
}

/// Environment variable that picks the GPU by name: any case-insensitive part of the adapter's name, such as `radeon` or `nvidia`. A
/// test switch for a PC with two GPUs, so that a second vendor's driver can be run there (the
/// development PC has an AMD iGPU beside its NVIDIA card). Vulkan unless [`GPU_BACKEND_ENV`] names
/// another backend. Unset, empty, or matching no adapter: the usual choice, unchanged.
pub const GPU_ADAPTER_ENV: &str = "MC_GPU_ADAPTER";

/// How many offscreen frames (a video export) may be queued on the GPU at once:
/// the swap chain's default frame latency ([`DEFAULT_FRAME_LATENCY`]). See [`Gpu::pace_offscreen`].
const OFFSCREEN_FRAMES_IN_FLIGHT: usize = 2;

/// The longest [`Gpu::pace_offscreen`] waits for one frame before carrying on
/// (a hung GPU must not freeze the window for good).
const OFFSCREEN_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// `MC_NW_F8_OFFSCREEN_PACING=0` turns [`Gpu::pace_offscreen`] off (A/B and
/// diagnosis). Read once.
fn offscreen_pacing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("MC_NW_F8_OFFSCREEN_PACING").is_none_or(|value| value != "0")
    })
}

/// The adapter [`GPU_ADAPTER_ENV`] names, from a new instance of `backend`'s
/// backends (Vulkan for [`BackendOverride::Auto`]). With `window`, also a surface for it, and
/// only an adapter that can present to that surface. Read when a context is built, never per
/// frame.
fn adapter_named_in_env(
    backend: BackendOverride,
    window: Option<&Arc<winit::window::Window>>,
) -> Option<(
    wgpu::Instance,
    Option<wgpu::Surface<'static>>,
    wgpu::Adapter,
)> {
    let wanted = std::env::var(GPU_ADAPTER_ENV)
        .ok()?
        .trim()
        .to_ascii_lowercase();
    if wanted.is_empty() {
        return None;
    }
    let backends = backend.backends().unwrap_or(wgpu::Backends::VULKAN);
    let instance = new_instance(backends);
    let surface = match window {
        Some(window) => Some(instance.create_surface(Arc::clone(window)).ok()?),
        None => None,
    };
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    let found = adapters.into_iter().find(|adapter| {
        adapter
            .get_info()
            .name
            .to_ascii_lowercase()
            .contains(&wanted)
            && surface
                .as_ref()
                .is_none_or(|surface| adapter.is_surface_supported(surface))
    });
    match found {
        Some(adapter) => {
            tracing::warn!(
                adapter = %adapter.get_info().name,
                backend = ?adapter.get_info().backend,
                "{GPU_ADAPTER_ENV}: using the named adapter (testing)"
            );
            Some((instance, surface, adapter))
        }
        None => {
            tracing::warn!(%wanted, ?backends, "{GPU_ADAPTER_ENV} matched no adapter; ignored");
            None
        }
    }
}

/// Writes a small buffer only when its contents have changed.
///
/// `Queue::write_buffer` allocates a fresh staging buffer per call, and a frame makes dozens of
/// these writes, most of which repeat the previous frame's bytes exactly.
#[derive(Debug, Default)]
pub struct ChangedWrite {
    last: Vec<u8>,
}

impl ChangedWrite {
    /// Upload `bytes` to the start of `buffer` unless they match what this cache last wrote.
    ///
    /// One cache must serve one buffer for its whole lifetime; a rebuilt buffer needs a new cache.
    pub fn write(&mut self, gpu: &Gpu, buffer: &wgpu::Buffer, bytes: &[u8]) {
        if self.last != bytes {
            gpu.queue.write_buffer(buffer, 0, bytes);
            self.last.clear();
            self.last.extend_from_slice(bytes);
        }
    }

    /// [`Self::write`] for a buffer that only the frame encoder reads
    /// and that nothing else writes: staged through the upload ring
    /// ([`Gpu::write_buffer_staged`]) while [`W19_O1_UNIFORM_RING_ENV`] is on, so recording does
    /// not wait at the queue's gate for the previous frame's submit.
    /// The bytes reach the buffer at the same place relative to the frame's draws (see the
    /// ordering contract in [`crate::upload`]).
    pub fn write_for_frame(&mut self, gpu: &Gpu, buffer: &wgpu::Buffer, bytes: &[u8]) {
        if !uniform_ring_enabled() {
            self.write(gpu, buffer, bytes);
            return;
        }
        if self.last != bytes {
            gpu.write_buffer_staged(buffer, 0, bytes);
            self.last.clear();
            self.last.extend_from_slice(bytes);
        }
    }
}

/// Set to `0` to send the small per-frame uniform writes of the sky dome,
/// the clouds and the first-person hand through the queue again. Read
/// once.
///
/// They were the first gated queue calls of a frame's recording (the attribution, wave
/// 2026-11-18, and this lane's): with the pipelined submit, a gated call waits until the present
/// thread has submitted the previous frame, and once frames are handed off without their image
/// (`MC_W19_O1_HANDOFF`) recording starts earlier, so that wait, which comes after input was
/// read, grew. Each of these buffers has one writer and is read only by its phase's passes in the
/// frame encoder, which is the upload ring's contract.
pub const W19_O1_UNIFORM_RING_ENV: &str = "MC_W19_O1_UNIFORM_RING";

/// Whether [`W19_O1_UNIFORM_RING_ENV`] is on (it is not `0`). Read once.
fn uniform_ring_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(W19_O1_UNIFORM_RING_ENV).is_none_or(|value| value != "0"))
}

/// Device, queue, surface, and the screen-sized attachments every phase shares.
///
/// Construct with [`Gpu::for_window`] or [`Gpu::headless`] — never with a struct literal, since
/// the extra fields beyond the four public ones are private and version-dependent.
///
/// # Bind group conventions
///
/// `@group(0)` is **reserved** for the shared camera uniform ([`Gpu::camera_layout`] /
/// [`Gpu::camera_bind_group`]), which the frame loop refreshes before any phase records.
/// Phases declare their own layouts starting at `@group(1)`.
pub struct Gpu {
    /// The logical device. Phases create buffers, textures, and pipelines from this.
    pub device: wgpu::Device,
    /// The command queue. Phases submit uploads here; the frame loop submits the frame encoder.
    ///
    /// A [`GatedQueue`](crate::frame_pipe::GatedQueue): the same calls as `wgpu::Queue` (and a
    /// `Deref` to it), each ordered after any frame the opt-in pipelined submit still owes the
    /// queue. With that path off, the gate is two atomic loads.
    pub queue: crate::frame_pipe::GatedQueue,
    /// The window surface, or `None` for a headless context (tests, CI, offscreen capture).
    ///
    /// Phases must not touch this. The frame loop owns acquisition and presentation.
    ///
    /// Shared (`Arc`) only so the opt-in pipelined present thread (#14 step 1a, `shell.rs`) can
    /// acquire and present on it; everything else still goes through [`Gpu`].
    pub surface: Option<Arc<wgpu::Surface<'static>>>,
    /// Current swap-chain configuration. `format`, `width`, and `height` are the fields phases
    /// care about; for a headless context they describe the offscreen target instead.
    pub config: wgpu::SurfaceConfiguration,

    adapter: wgpu::Adapter,
    present: PresentPreference,
    gpu_timing: bool,
    multi_draw_indirect: bool,
    timestamp_period_ns: f32,
    /// R1-8: staged per-frame uploads, submitted with the frame. See [`crate::upload`].
    uploads: std::sync::Mutex<crate::upload::UploadRing>,
    /// Whether [`Gpu::write_buffer_staged`] uses the ring (`MC_UPLOAD_RING`, default on).
    upload_ring: bool,
    sample_count: u32,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    msaa_view: Option<wgpu::TextureView>,
    offscreen: Option<wgpu::Texture>,
    camera_buffer: wgpu::Buffer,
    camera_layout: wgpu::BindGroupLayout,
    camera_bind_group: wgpu::BindGroup,
    /// The last [`Gpu::resize`] asked for a zero-area surface (Windows reports 0×0 when the
    /// window is minimised). [`Gpu::acquire`] then never reconfigures: a zero-sized surface
    /// cannot be configured, and a stale-size configure is a GPU drain per loop iteration.
    zero_area: bool,
    /// See [`FAILED_RECONFIGURE_BACKOFF`]. `None` once an acquire succeeds.
    reconfigure_backoff_until: Option<std::time::Instant>,
    /// Video export: while `Some`, every frame
    /// renders into `offscreen` at this size instead of the window, and `config.width`/`height`
    /// say so; the swap chain stays at `window_size`. `None` (always, outside an export) is the
    /// path that existed before.
    render_override: Option<(u32, u32)>,
    /// The window's size: what the swap chain is configured with. Equal to `config`'s size
    /// whenever `render_override` is `None`.
    window_size: (u32, u32),
    /// The device-lost callback's record
    /// ([`crate::gpu_health::DeviceHealth`]). Before it, a lost device was never even logged.
    health: Arc<crate::gpu_health::DeviceHealth>,
    /// The last offscreen frames' submissions, oldest first (see
    /// [`Gpu::pace_offscreen`]). At most [`OFFSCREEN_FRAMES_IN_FLIGHT`] long; empty until a video
    /// export first renders offscreen. Locked only while an export renders.
    offscreen_marks: std::sync::Mutex<std::collections::VecDeque<wgpu::SubmissionIndex>>,
    /// The part of the frame's clear that `PhaseList` handed to the first pass
    /// touching each attachment (see [`Gpu::arm_frame_clear`]); empty outside a frame.
    frame_clear: std::sync::Mutex<FrameClear>,
    /// Whether [`Gpu::frame_clear`] holds anything: [`Gpu::begin_pass`]'s one
    /// atomic load when nothing is pending, which is always outside the first phases of a frame.
    frame_clear_armed: std::sync::atomic::AtomicBool,
}

/// Whether [`Gpu::write_camera`] stages the camera uniform through the
/// upload ring. Read once; `MC_W18_OB_CAMERA_RING=0` writes it through the queue as before.
fn camera_ring_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_W18_OB_CAMERA_RING").is_none_or(|value| value != "0"))
}

/// The frame's clear, still to be done.
#[derive(Default)]
struct FrameClear {
    /// The frame's colour target and the colour it is cleared to, until a pass opens on it.
    color: Option<(wgpu::TextureView, wgpu::Color)>,
    /// The shared depth buffer still waits for its clear to `1.0`.
    depth: bool,
}

impl Gpu {
    /// Build a context that presents to `window`.
    ///
    /// Blocks on adapter and device acquisition (`pollster`), which is what every wgpu desktop
    /// application does at startup; nothing else in the frame loop is async.
    ///
    /// # Errors
    /// [`RenderError::CreateSurface`] if the window handle is unusable, [`RenderError::NoAdapter`]
    /// if no backend can drive it, [`RenderError::DeviceRequest`] if the device is refused.
    pub fn for_window(
        window: Arc<winit::window::Window>,
        opts: GpuOptions,
    ) -> Result<Self, RenderError> {
        Self::for_window_with(window, opts, PresentPreference::Auto)
    }

    /// [`Gpu::for_window`], plus an explicit swap-chain present mode.
    ///
    /// `for_window` is this with [`PresentPreference::Auto`], which is what it always did.
    ///
    /// The preference is *stored*, so a later [`Gpu::apply_options`] keeps honouring it rather
    /// than silently reverting to `AutoVsync`; change it with
    /// [`Gpu::set_present_preference`].
    ///
    /// **This is deliberately not a field on [`GpuOptions`].** `GpuOptions` is built with a
    /// struct literal in twenty-odd places across five crates' tests and benches; adding a field
    /// to it would break every one of them for a setting only the windowed path can use.
    ///
    /// # Errors
    /// The same as [`Gpu::for_window`].
    pub fn for_window_with(
        window: Arc<winit::window::Window>,
        opts: GpuOptions,
        present: PresentPreference,
    ) -> Result<Self, RenderError> {
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        let power_preference = if opts.low_power {
            wgpu::PowerPreference::LowPower
        } else {
            wgpu::PowerPreference::HighPerformance
        };
        // `MC_GPU_BACKEND` (PRESENT-5): an explicit backend accepts any adapter type it reaches;
        // `auto` (unset) is exactly the historical Vulkan-for-discrete policy below.
        let backend = BackendOverride::from_env();
        // `MC_GPU_ADAPTER` (testing): the named GPU wins over both policies.
        let named = match adapter_named_in_env(backend, Some(&window)) {
            Some((instance, Some(surface), adapter)) => Some((instance, surface, adapter)),
            _ => None,
        };
        let vulkan = match backend.backends() {
            _ if named.is_some() => named,
            Some(backends) => {
                let instance = new_instance(backends);
                let found = instance
                    .create_surface(window.clone())
                    .ok()
                    .and_then(|surface| {
                        let adapter = pollster::block_on(instance.request_adapter(
                            &wgpu::RequestAdapterOptions {
                                power_preference,
                                compatible_surface: Some(&surface),
                                force_fallback_adapter: false,
                                apply_limit_buckets: false,
                            },
                        ))
                        .ok()?;
                        Some((surface, adapter))
                    });
                if found.is_none() {
                    tracing::warn!(
                        ?backend,
                        "{GPU_BACKEND_ENV} backend found no adapter; falling back to every backend"
                    );
                }
                found.map(|(surface, adapter)| (instance, surface, adapter))
            }
            None => (!opts.low_power)
                .then(|| {
                    let instance = new_instance(wgpu::Backends::VULKAN);
                    let surface = instance.create_surface(window.clone()).ok()?;
                    let adapter = discrete_adapter(&instance, power_preference, Some(&surface))?;
                    Some((instance, surface, adapter))
                })
                .flatten(),
        };
        let (_instance, surface, adapter) = match vulkan {
            Some(found) => found,
            None => {
                let instance = new_instance(wgpu::Backends::all());
                let surface = instance
                    .create_surface(window)
                    .map_err(RenderError::CreateSurface)?;
                let adapter =
                    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference,
                        compatible_surface: Some(&surface),
                        force_fallback_adapter: false,
                        apply_limit_buckets: false,
                    }))
                    .map_err(|_| RenderError::NoAdapter)?;
                (instance, surface, adapter)
            }
        };
        let adapter_facts = crate::AdapterFacts::from_info(&adapter.get_info());

        let (device, queue) = Self::request_device(&adapter)
            .map_err(|error| error.with_adapter(adapter_facts.clone()))?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| RenderError::NoAdapter.with_adapter(adapter_facts))?;
        config.usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;

        // Diagnostic for the frame-pacing question: `PresentMode::AutoVsync` is
        // documented (see the doc comment on `wgpu::PresentMode::AutoVsync`) to fall back through
        // `FifoRelaxed` then `Fifo` — both of which block on vblank — so it should never silently
        // degrade to a non-blocking mode the way `AutoNoVsync` can. Logging the surface's actual
        // supported list plus the resolved mode turns "is AutoVsync doing what it says" into
        // something a log line answers instead of a hypothesis. Not gated behind a feature flag or
        // env var: this is one `get_capabilities` call and a log line, at startup only, at
        // `info` level like `adapter_info` already is a few lines up the call chain.
        //
        // Since T3.2 the same capability list also *drives* the choice rather than only
        // describing it, so this log line is now the record of a decision, not a diagnostic.
        let caps = surface.get_capabilities(&adapter);
        // Java 1.8.9 renders stored texture/color bytes into an ordinary RGBA framebuffer
        // (Display PixelFormat without sRGB). An sRGB swap chain would encode those bytes
        // again, brightening even the clear sky. Match the headless byte-domain target.
        config.format = java_surface_format(config.format, &caps.formats);
        if config.format.is_srgb() {
            tracing::warn!(format = ?config.format, "surface has no plain RGBA format; Java color parity unavailable");
        }
        // Screenshots read the resolved swap-chain image immediately before present.  Not every
        // surface exposes COPY_SRC (notably a few browser/compositor paths), so request it only
        // when the negotiated capabilities allow it; unsupported surfaces retain the normal
        // render-only configuration and report a readback error at the explicit seam.
        if caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            config.usage |= wgpu::TextureUsages::COPY_SRC;
        }
        // Pipelined frames recorded before their image reach it by one copy
        // (`shell::W18_OB_DECOUPLE_ENV`). Asked for only on that path and where the surface
        // allows it; otherwise the configuration is the one above, and frames record into the
        // image as before.
        if crate::shell::decouple_requested() && caps.usages.contains(wgpu::TextureUsages::COPY_DST)
        {
            config.usage |= wgpu::TextureUsages::COPY_DST;
        }
        let requested = resolve_present_mode(present, opts.vsync, &caps.present_modes);
        config.present_mode = requested;
        tracing::info!(
            preference = present.as_str(),
            vsync = opts.vsync,
            requested = ?requested,
            supported = ?caps.present_modes,
            resolved = ?resolve_auto_present_mode(requested, &caps.present_modes),
            format = ?config.format,
            supported_formats = ?caps.formats,
            "surface present-mode capabilities",
        );

        surface.configure(&device, &config);

        Ok(Self::assemble(
            adapter,
            device,
            queue,
            Some(surface),
            config,
            None,
            opts.sample_count,
            present,
        ))
    }

    /// Build a context with no window, rendering into an offscreen texture.
    ///
    /// This is the path the automated tests use: it needs no display server, so the phase seam
    /// and the clear-colour frame can be verified in CI. [`Gpu::acquire`] hands back a view of
    /// the offscreen texture, and [`Gpu::read_pixels`] copies it back to the CPU.
    ///
    /// # Errors
    /// [`RenderError::NoAdapter`] when the machine exposes no usable adapter at all — callers in
    /// tests should treat that as "skip", not "fail".
    pub fn headless(width: u32, height: u32, opts: GpuOptions) -> Result<Self, RenderError> {
        let (width, height) = (width.max(1), height.max(1));
        // `MC_GPU_ADAPTER` (testing) picks the GPU, with `MC_GPU_BACKEND`'s
        // backend; unset, the usual discrete-Vulkan-first choice below.
        let named = adapter_named_in_env(BackendOverride::from_env(), None);
        let adapter = match named {
            Some((_, _, adapter)) => adapter,
            None => match discrete_adapter(
                &new_instance(wgpu::Backends::VULKAN),
                wgpu::PowerPreference::None,
                None,
            ) {
                Some(adapter) => adapter,
                None => pollster::block_on(new_instance(wgpu::Backends::all()).request_adapter(
                    &wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::None,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                        apply_limit_buckets: false,
                    },
                ))
                .map_err(|_| RenderError::NoAdapter)?,
            },
        };

        let (device, queue) = Self::request_device(&adapter)?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: HEADLESS_FORMAT,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: present_mode(opts.vsync),
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
        };

        let offscreen = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mc-render offscreen colour"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: config.usage,
            view_formats: &[],
        });

        Ok(Self::assemble(
            adapter,
            device,
            queue,
            None,
            config,
            Some(offscreen),
            opts.sample_count,
            PresentPreference::Auto,
        ))
    }

    /// The two features [`crate::pipeline::PhaseList`]'s GPU timer needs.
    ///
    /// `TIMESTAMP_QUERY` is what allows a query set to be created and resolved at all;
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS` is what allows `CommandEncoder::write_timestamp`, which
    /// is how per-phase timings are taken **without** reaching into any phase's own render pass —
    /// `TIMESTAMP_QUERY_INSIDE_PASSES` would be the finer instrument but it would require every
    /// phase to cooperate.
    #[must_use]
    fn gpu_timing_features() -> wgpu::Features {
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
    }

    /// In wgpu 30, `multi_draw_indexed_indirect` is native on Vulkan whenever the device has
    /// `multiDrawIndirect` and on D3D12 always (one `ExecuteIndirect`); GL and Metal emulate it
    /// with a loop. The count feature is only needed for `*_indirect_count` (GPU-written draw
    /// counts), which nothing calls yet. The terrain shader relies on `first_instance` to select
    /// the camera-relative section offset, which `INDIRECT_FIRST_INSTANCE` provides.
    #[must_use]
    fn multi_draw_features() -> wgpu::Features {
        wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | wgpu::Features::INDIRECT_FIRST_INSTANCE
    }

    fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue), RenderError> {
        // Requested only when [`GPU_TIMING_ENV`] asks, and only when the adapter has them.
        // Timestamp queries remain opt-in, while native multi-draw is requested whenever the
        // adapter can support the production terrain path. Unsupported adapters keep the direct
        // fallback without changing device creation requirements.
        let timing = Self::gpu_timing_features();
        let multi_draw = Self::multi_draw_features();
        let mut required_features = wgpu::Features::empty();
        if gpu_timing_requested() && adapter.features().contains(timing) {
            required_features |= timing;
        }
        if adapter.features().contains(multi_draw)
            && adapter
                .get_downlevel_capabilities()
                .flags
                .contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
        {
            required_features |= multi_draw;
        }
        // Take whatever the adapter offers rather than the conservative downlevel defaults: the
        // terrain mesher (S11) uploads far more than the default buffer-size limit.
        let mut required_limits = adapter.limits();
        // `MC_GPU_LIMIT_BUFFER_MB`: a smaller GPU's buffer limit, for testing
        // on this PC what an AMD or older card would allow. Unset (always, for players): the
        // adapter's own limits, exactly as before.
        if let Some(bytes) = limit_buffer_bytes_from_env() {
            required_limits.max_buffer_size = required_limits.max_buffer_size.min(bytes);
            required_limits.max_storage_buffer_binding_size = required_limits
                .max_storage_buffer_binding_size
                .min(required_limits.max_buffer_size);
            tracing::warn!(
                max_buffer_size = required_limits.max_buffer_size,
                "{LIMIT_BUFFER_MB_ENV}: lowered GPU buffer limit (testing)"
            );
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mc-render device"),
            required_features,
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            // `MemoryHints::Performance` as before, unless
            // `MC_W21_MR_GPU_BLOCKS` asks for other block sizes (see `w21_mr::GPU_BLOCKS_ENV`).
            memory_hints: crate::w21_mr::memory_hints(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(RenderError::DeviceRequest)?;
        // Pack PNGs that become textures of their own are refused at
        // decode when a side is over this device's limit (a validation error ends the process).
        mc_assets::limit_texture_side(device.limits().max_texture_dimension_2d);
        // Measurement aid; unset (players): one env read, nothing else.
        crate::w21_mr::start_memory_report(&device);
        Ok((device, queue))
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: Option<wgpu::Surface<'static>>,
        config: wgpu::SurfaceConfiguration,
        offscreen: Option<wgpu::Texture>,
        requested_samples: u32,
        present: PresentPreference,
    ) -> Self {
        let sample_count = clamp_sample_count(&adapter, config.format, requested_samples);
        // Read from the *device*, not from the env var or the adapter: what matters downstream is
        // what was actually granted. A machine that asked for timestamps and did not get them
        // must take the untimed path, not record garbage.
        let gpu_timing = device.features().contains(Self::gpu_timing_features());
        let multi_draw_indirect = device.features().contains(Self::multi_draw_features())
            && adapter
                .get_downlevel_capabilities()
                .flags
                .contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION);
        let timestamp_period_ns = if gpu_timing {
            queue.get_timestamp_period()
        } else {
            0.0
        };
        let (depth_texture, depth_view) = create_depth(&device, &config, sample_count);
        let msaa_view = create_msaa(&device, &config, sample_count);

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc-render camera uniform"),
            size: std::mem::size_of::<crate::pipeline::CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mc-render camera layout (group 0)"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc-render camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let pipe = Arc::new(crate::frame_pipe::FramePipe::new());
        let window_size = (config.width, config.height);
        let health = crate::gpu_health::DeviceHealth::watch(&device);
        Self {
            render_override: None,
            window_size,
            health,
            device,
            queue: crate::frame_pipe::GatedQueue::new(queue, pipe),
            surface: surface.map(Arc::new),
            config,
            adapter,
            present,
            gpu_timing,
            multi_draw_indirect,
            timestamp_period_ns,
            uploads: std::sync::Mutex::new(crate::upload::UploadRing::new()),
            upload_ring: crate::upload::upload_ring_enabled_from_env(),
            sample_count,
            depth_texture,
            depth_view,
            msaa_view,
            offscreen,
            camera_buffer,
            camera_layout,
            camera_bind_group,
            zero_area: false,
            reconfigure_backoff_until: None,
            offscreen_marks: std::sync::Mutex::new(std::collections::VecDeque::new()),
            frame_clear: std::sync::Mutex::new(FrameClear::default()),
            frame_clear_armed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Information about the adapter actually in use — for the F3 overlay (S16) and crash reports.
    #[must_use]
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// The adapter in use (tests that need a second device with other limits).
    #[must_use]
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// Whether the device has been lost (a driver reset or hang, or an error
    /// wgpu treats as fatal). One atomic load. Nothing created after the loss works, and wgpu
    /// raises no error for it, so callers that would otherwise panic later check this.
    #[inline]
    #[must_use]
    pub fn is_device_lost(&self) -> bool {
        self.health.is_lost()
    }

    /// Why the device was lost (the driver's message), once it has been.
    #[must_use]
    pub fn device_lost(&self) -> Option<String> {
        self.health.reason()
    }

    /// The device-lost record itself, for code that learns of a loss another way.
    #[must_use]
    pub fn device_health(&self) -> &Arc<crate::gpu_health::DeviceHealth> {
        &self.health
    }

    /// Maximum width or height, in texels, of a 2D texture this *granted* device accepts.
    ///
    /// This is the `wgpu` equivalent of vanilla's
    /// `Minecraft.getGLMaximumTextureSize()`.  Asset loading must use this value as the atlas
    /// stitch cap rather than a policy constant: the atlas is uploaded as a single 2D texture,
    /// so using an adapter's larger advertised limit (or a fixed 2048) would respectively risk a
    /// validation failure or reject resource packs that the active device can render.
    ///
    /// In particular, callers that bake after [`Gpu`] creation should pass this directly to
    /// [`mc_assets::bake_all`].  It is deliberately read from [`wgpu::Device::limits`], not the
    /// adapter, because downstream resources are constrained by what `request_device` actually
    /// granted.
    #[must_use]
    pub fn atlas_max_dimension(&self) -> u32 {
        atlas_max_dimension(&self.device.limits())
    }

    /// `true` when there is no surface, i.e. this context was built by [`Gpu::headless`].
    #[must_use]
    pub fn is_headless(&self) -> bool {
        self.surface.is_none()
    }

    /// Whether this device was granted the timestamp-query features, i.e. whether
    /// [`PhaseList`](crate::pipeline::PhaseList) can time phases on the GPU.
    ///
    /// `false` for every run that did not set [`GPU_TIMING_ENV`], and for any adapter that lacks
    /// the features even when it did.
    #[must_use]
    pub fn supports_gpu_timing(&self) -> bool {
        self.gpu_timing
    }

    /// Whether native multi-draw indexed indirect is available for terrain submission.
    ///
    /// This is false on adapters that would emulate the multi-draw call or cannot honor the
    /// non-zero `first_instance` used by the chunk-offset vertex stream.
    #[must_use]
    pub fn supports_multi_draw_indirect(&self) -> bool {
        self.multi_draw_indirect
    }

    /// Nanoseconds per timestamp-query tick, from `Queue::get_timestamp_period`. Zero when
    /// [`Gpu::supports_gpu_timing`] is `false`.
    #[must_use]
    pub fn timestamp_period_ns(&self) -> f32 {
        self.timestamp_period_ns
    }

    /// MSAA sample count actually in effect (1 when multisampling is off or unsupported).
    ///
    /// Phases **must** set this on `MultisampleState::count` in every pipeline they create.
    #[must_use]
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// View of the shared depth buffer. Prefer [`Gpu::begin_pass`], which attaches it for you.
    #[must_use]
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// The shared depth buffer's *texture*, for a phase that needs to **read** what earlier
    /// phases wrote rather than write to it — screen-space fog, SSAO, soft particles.
    ///
    /// Added for the screen-space fog pass, which samples the depth buffer.
    /// The texture is created with `TEXTURE_BINDING` (and `COPY_SRC` when
    /// [`Gpu::sample_count`] is 1) on top of the `RENDER_ATTACHMENT` it always had, so this is
    /// purely additive: nothing that used [`Gpu::depth_view`] before behaves differently.
    ///
    /// # Reading it correctly
    ///
    /// * A pass that samples this **must not also attach it** — open the pass with
    ///   [`DepthAttachment::None`], or wgpu will reject the simultaneous read/write.
    /// * When [`Gpu::sample_count`] is greater than 1 the texture is multisampled: bind it as
    ///   `texture_depth_multisampled_2d` and read it with `textureLoad(t, coords, sample)`.
    ///   There is no `texture_depth_2d` view of it and no sampler will work.
    /// * `Depth32Float` is not filterable anywhere, so `textureLoad` (integer texel coords) is
    ///   the only correct read even in the single-sampled case.
    #[must_use]
    pub fn depth_texture(&self) -> &wgpu::Texture {
        &self.depth_texture
    }

    /// Bind group layout for `@group(0)`: the camera uniform. Every pipeline that reads the
    /// camera puts this first in its `PipelineLayoutDescriptor::bind_group_layouts`.
    #[must_use]
    pub fn camera_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_layout
    }

    /// The camera bind group to `set_bind_group(0, ...)`. Its contents are refreshed by the frame
    /// loop before any phase records, so it is always current for the frame being drawn.
    #[must_use]
    pub fn camera_bind_group(&self) -> &wgpu::BindGroup {
        &self.camera_bind_group
    }

    /// The buffer behind [`Gpu::camera_bind_group`], for
    /// [`PhaseList`](crate::PhaseList)'s in-encoder restore of the unjittered camera at
    /// [`PhaseOrder::TEMPORAL_AA`](crate::PhaseOrder::TEMPORAL_AA) (see [`crate::Camera::jitter`]).
    pub(crate) fn camera_buffer(&self) -> &wgpu::Buffer {
        &self.camera_buffer
    }

    fn lock_upload_ring(&self) -> std::sync::MutexGuard<'_, crate::upload::UploadRing> {
        self.uploads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// `Queue::write_buffer` through the per-frame upload ring (R1-8 / GPURES-2).
    ///
    /// Same argument contract and the same position relative to this frame's draws as
    /// `write_buffer`, without a staging-buffer creation per call. Read the ordering contract in
    /// [`crate::upload`] before moving a stream here: every writer of the buffer moves together,
    /// and only the frame encoder may read it. Falls back to `write_buffer` when the ring is off.
    pub fn write_buffer_staged(&self, buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]) {
        if self.upload_ring {
            // After a device loss the ring's staging chunks cannot be mapped, and
            // wgpu's staging belt panics on that ("Failed to get mapped range for staging belt
            // buffer"). Nothing drawn after a loss is shown and the shell stops at the next
            // frame, so the write is dropped. One atomic load beside the ring's lock.
            if self.is_device_lost() {
                return;
            }
            self.lock_upload_ring()
                .write(&self.device, buffer, offset, bytes);
        } else {
            self.queue.write_buffer(buffer, offset, bytes);
        }
    }

    /// [`Gpu::write_buffer_staged`] for a writer that produces its `size`
    /// bytes straight into the staging memory, instead of into a buffer of its own that is then
    /// copied. `fill` must write all `size` bytes it is given. With the
    /// ring off, the same bytes go through `Queue::write_buffer_with`.
    pub fn write_buffer_staged_with(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
        fill: impl FnOnce(wgpu::WriteOnly<'_, [u8]>),
    ) {
        let Some(size) = wgpu::BufferSize::new(size) else {
            return;
        };
        if self.upload_ring {
            // As in `write_buffer_staged`.
            if self.is_device_lost() {
                return;
            }
            self.lock_upload_ring()
                .write_with(&self.device, buffer, offset, size, fill);
        } else if let Some(mut view) = self.queue.write_buffer_with(buffer, offset, size) {
            fill(view.slice(..));
        }
    }

    /// Submit the frame encoder's commands, preceded in the same submission by every staged
    /// upload of the frame. [`PhaseList::render`](crate::pipeline::PhaseList::render) ends here.
    pub fn submit_frame(&self, frame: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        // The lock is not held across `submit`: map callbacks may run inside it.
        let staged = {
            mc_core::profile_scope!("gpu.upload_ring_finish");
            self.lock_upload_ring().finish()
        };
        let index = {
            mc_core::profile_scope!("gpu.submit_call");
            self.queue.submit(staged.into_iter().chain(Some(frame)))
        };
        // No swap chain holds an offscreen frame back: bound it like one would.
        if self.render_override.is_some() {
            self.pace_offscreen(&index);
        }
        index
    }

    /// The pipelined-submit counterpart of [`Self::submit_frame`]: close this frame's staged
    /// uploads and leave them, with `frame`, pending for [`Self::take_pending_frame`] instead of
    /// submitting. Any later main-thread queue call submits it first (see [`crate::frame_pipe`]).
    pub(crate) fn defer_frame(&self, frame: wgpu::CommandEncoder) {
        let staged = {
            mc_core::profile_scope!("gpu.upload_ring_finish");
            self.lock_upload_ring().finish()
        };
        self.queue.pipe().defer(self.queue.ungated(), staged, frame);
    }

    /// [`Self::defer_frame`] for a frame the shell hands to the present thread
    /// at once, behind the frames already handed off: no wait for their submits
    /// ([`FramePipe::defer_behind_handed`](crate::frame_pipe::FramePipe::defer_behind_handed)).
    pub(crate) fn defer_frame_for_handoff(&self, frame: wgpu::CommandEncoder) {
        let staged = {
            mc_core::profile_scope!("gpu.upload_ring_finish");
            self.lock_upload_ring().finish()
        };
        self.queue
            .pipe()
            .defer_behind_handed(self.queue.ungated(), staged, frame);
    }

    /// The frame left by [`Self::defer_frame`], counted as handed off: the caller must give it to
    /// [`FramePipe::submit_handed`](crate::frame_pipe::FramePipe::submit_handed) (the present
    /// thread), or call [`FramePipe::abandon`](crate::frame_pipe::FramePipe::abandon) if it
    /// cannot.
    pub(crate) fn take_pending_frame(&self) -> Option<crate::frame_pipe::PendingFrame> {
        self.queue.pipe().take_for_handoff()
    }

    /// The frame pipe, for the present thread's end of it.
    pub(crate) fn frame_pipe(&self) -> Arc<crate::frame_pipe::FramePipe> {
        Arc::clone(self.queue.pipe())
    }

    /// Submit staged uploads now, for a path that submits its own command buffer and reads a
    /// staged buffer. A no-op when nothing is staged.
    pub fn flush_staged_uploads(&self) {
        let staged = self.lock_upload_ring().finish();
        if let Some(staged) = staged {
            let _ = self.queue.submit(Some(staged));
        }
    }

    /// Upload-ring counters since the previous call (for frame telemetry / F3).
    pub fn take_upload_ring_stats(&self) -> crate::upload::UploadRingStats {
        self.lock_upload_ring().take_stats()
    }

    /// Enable or disable the upload ring (tests and A/B runs). Flushes anything staged first.
    pub fn set_upload_ring_enabled(&mut self, enabled: bool) {
        self.flush_staged_uploads();
        self.upload_ring = enabled;
    }

    /// Upload a fresh camera uniform. Called once per frame by
    /// [`PhaseList::render`](crate::pipeline::PhaseList::render); phases do not call this.
    ///
    /// Staged through the upload ring: the
    /// frame encoder is the camera buffer's only reader and this its only writer (the TAA restore
    /// is a copy inside the frame encoder, after the staged copy), so the bytes reach the frame's
    /// passes exactly as a queue write would. A queue write had to wait first for the present
    /// thread to submit the previous frame (`GatedQueue`), and since the frame is recorded before
    /// its image (`MC_W18_OB_DECOUPLE`) that wait fell inside recording on most frames.
    /// `MC_W18_OB_CAMERA_RING=0` writes through the queue as before.
    pub fn write_camera(&self, uniform: &crate::pipeline::CameraUniform) {
        self.write_camera_uniform(&self.camera_buffer, uniform);
    }

    /// Write a camera uniform into `buffer` (the camera buffer, or the
    /// unjittered copy `PhaseList` restores under TAA): staged, or through the queue with
    /// `MC_W18_OB_CAMERA_RING=0`. Both buffers are read only by the frame encoder.
    pub(crate) fn write_camera_uniform(
        &self,
        buffer: &wgpu::Buffer,
        uniform: &crate::pipeline::CameraUniform,
    ) {
        if camera_ring_enabled() && !crate::w18_ob_ab::off(crate::w18_ob_ab::CAMERA_RING) {
            self.write_buffer_staged(buffer, 0, bytemuck::bytes_of(uniform));
        } else {
            self.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(uniform));
        }
    }

    /// Resize the swap chain and every screen-sized attachment. A no-op for a zero-area window
    /// (minimising on Windows reports 0×0, and configuring a zero-sized surface is a hard error).
    ///
    /// Returns whether anything was rebuilt, so the shell can skip phase and app resizes for a
    /// repeated or zero-area size (PRESENT-9).
    pub fn resize(&mut self, width: u32, height: u32) -> bool {
        // The one test of whether the swap chain changes, shared with
        // `resize_configures` (the shell decides from it whether to take back an image first).
        let changes = self.resize_changes_window_size(width, height);
        // Remembered before the early return: while minimised, `acquire` must not try to
        // reconfigure the stale-sized swap chain on every loop (see `zero_area`).
        self.zero_area = width == 0 || height == 0;
        if self.render_override.is_some() {
            // Rendering offscreen at the export's size: only the swap chain
            // follows the window, and nothing that phases see changes.
            if changes {
                self.window_size = (width, height);
                self.reconfigure_backoff_until = None;
                if let Some(surface) = &self.surface {
                    surface.configure(&self.device, &self.surface_config());
                }
            }
            return false;
        }
        if !changes {
            return false;
        }
        self.reconfigure_backoff_until = None;
        self.config.width = width;
        self.config.height = height;
        self.window_size = (width, height);
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.surface_config());
        }
        if self.offscreen.is_some() {
            self.offscreen = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("mc-render offscreen colour"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: self.config.usage,
                view_formats: &[],
            }));
        }
        let (depth_texture, depth_view) =
            create_depth(&self.device, &self.config, self.sample_count);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.msaa_view = create_msaa(&self.device, &self.config, self.sample_count);
        true
    }

    /// Whether [`Self::resize`] to this size changes the size the
    /// swap chain is configured at: a size with an area (a minimised window reports 0×0) that is
    /// not the window's current one ([`Self::window_size`]).
    fn resize_changes_window_size(&self, width: u32, height: u32) -> bool {
        width != 0 && height != 0 && (width, height) != self.window_size()
    }

    /// Whether [`Self::resize`] to this size configures the swap
    /// chain. The shell takes back the pipelined present thread's image only then: an image
    /// dropped without being presented is not given back to a Vulkan swap chain until its next
    /// configure.
    pub(crate) fn resize_configures(&self, width: u32, height: u32) -> bool {
        self.surface.is_some() && self.resize_changes_window_size(width, height)
    }

    /// Render every frame offscreen at
    /// `Some((width, height))` (a video export at its own resolution, whatever the window's size),
    /// or back to the window with `None`.
    ///
    /// While set, `config.width`/`height` and the depth (and MSAA) attachments are the export's
    /// size, [`Self::acquire`] hands out the offscreen target (usable as a render attachment, a
    /// copy source and a sampled texture: [`AcquiredFrame::color_texture`] returns it), and the
    /// swap chain keeps following the window ([`Self::resize`], [`Self::acquire_window`]). The
    /// caller resizes the phases and the camera when this returns `true`, exactly as after a
    /// window resize. It must be called while no surface texture is held.
    ///
    /// Returns whether anything changed.
    pub fn set_render_override(&mut self, size: Option<(u32, u32)>) -> bool {
        let size = size.map(|(w, h)| (w.max(1), h.max(1)));
        if size == self.render_override {
            return false;
        }
        if self.render_override.is_none() {
            self.window_size = (self.config.width, self.config.height);
        }
        self.render_override = size;
        let (width, height) = size.unwrap_or(self.window_size);
        self.config.width = width;
        self.config.height = height;
        let keep_offscreen = self.surface.is_none() || size.is_some();
        self.offscreen = keep_offscreen.then(|| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("mc-render offscreen colour"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: self.config.usage
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });
        let (depth_texture, depth_view) =
            create_depth(&self.device, &self.config, self.sample_count);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.msaa_view = create_msaa(&self.device, &self.config, self.sample_count);
        true
    }

    /// The offscreen size frames render at, while [`Self::set_render_override`]
    /// is in force.
    #[must_use]
    pub fn render_override(&self) -> Option<(u32, u32)> {
        self.render_override
    }

    /// The window's size (the swap chain's), which differs from `config`'s only
    /// during a [`Self::set_render_override`].
    #[must_use]
    pub fn window_size(&self) -> (u32, u32) {
        if self.render_override.is_some() {
            self.window_size
        } else {
            (self.config.width, self.config.height)
        }
    }

    /// The configuration the swap chain is configured with: `config`, at the window's size.
    fn surface_config(&self) -> std::borrow::Cow<'_, wgpu::SurfaceConfiguration> {
        if self.render_override.is_none() {
            return std::borrow::Cow::Borrowed(&self.config);
        }
        let mut config = self.config.clone();
        config.width = self.window_size.0;
        config.height = self.window_size.1;
        std::borrow::Cow::Owned(config)
    }

    /// Change vsync and MSAA at runtime, e.g. from the video options screen (S16).
    ///
    /// The stored [`PresentPreference`] is honoured: with anything but
    /// [`PresentPreference::Auto`] set, `opts.vsync` no longer decides the present mode. Change
    /// the preference with [`Gpu::set_present_preference`].
    pub fn apply_options(&mut self, opts: GpuOptions) {
        let samples = clamp_sample_count(&self.adapter, self.config.format, opts.sample_count);
        let mode = self.wanted_present_mode(self.present, opts.vsync);
        if samples == self.sample_count && mode == self.config.present_mode {
            return;
        }
        self.sample_count = samples;
        self.config.present_mode = mode;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.surface_config());
        }
        let (depth_texture, depth_view) =
            create_depth(&self.device, &self.config, self.sample_count);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.msaa_view = create_msaa(&self.device, &self.config, self.sample_count);
    }

    /// Switch present mode at runtime, reconfiguring the surface only when
    /// the resolved mode actually changes.
    ///
    /// `vsync` is the current [`GpuOptions::vsync`], needed because
    /// [`PresentPreference::Auto`] still derives from it.
    ///
    /// Returns the mode now configured, which may not be the one asked for: an unsupported
    /// preference falls back down the chain documented on [`resolve_present_mode`]. A caller
    /// that wants to tell the user what it actually got should use the return value rather than
    /// echo the request.
    pub fn set_present_preference(
        &mut self,
        present: PresentPreference,
        vsync: bool,
    ) -> wgpu::PresentMode {
        self.present = present;
        let mode = self.wanted_present_mode(present, vsync);
        if mode == self.config.present_mode {
            return mode;
        }
        self.config.present_mode = mode;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.surface_config());
        }
        mode
    }

    /// Apply a present preference, `enableVsync` and a frame latency together, at runtime
    /// (C1-10 / KNOBS-2), with **at most one** `Surface::configure`.
    ///
    /// This is what makes the in-game VSync button take effect without a restart. MSAA is
    /// deliberately not part of it: a sample-count change invalidates every pipeline built for
    /// the old count, which needs a phase re-install (see [`Gpu::apply_options`]).
    ///
    /// Each configure drains the GPU (`wgpu-core` waits for the queue to go idle) and rebuilds
    /// the swap chain, so this is a one-frame hitch when something actually changed and free
    /// otherwise. It must be called while no [`AcquiredFrame`] is alive — `wgpu` panics if a
    /// surface texture outlives a configure — which is why the shell calls it before `acquire`.
    ///
    /// Returns the mode now configured (unresolved `AutoVsync`/`AutoNoVsync` for
    /// [`PresentPreference::Auto`], exactly as [`Gpu::present_mode`] reports it).
    pub fn set_present_options(
        &mut self,
        present: PresentPreference,
        vsync: bool,
        frame_latency: u32,
    ) -> wgpu::PresentMode {
        self.present = present;
        let mode = self.wanted_present_mode(present, vsync);
        let latency = clamp_frame_latency(frame_latency);
        // The same test as `present_options_configure`.
        if !self.present_options_change(mode, latency) {
            return mode;
        }
        self.config.present_mode = mode;
        self.config.desired_maximum_frame_latency = latency;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.surface_config());
        }
        mode
    }

    /// Whether a resolved mode and a clamped latency differ from the
    /// ones the swap chain is configured with.
    fn present_options_change(&self, mode: wgpu::PresentMode, latency: u32) -> bool {
        mode != self.config.present_mode || latency != self.config.desired_maximum_frame_latency
    }

    /// Whether [`Self::set_present_options`] with these options
    /// configures the swap chain: they resolve to another mode or latency. The in-game VSync
    /// button with an explicit `presentMode`, for one, does not. See [`Self::resize_configures`].
    pub(crate) fn present_options_configure(
        &self,
        present: PresentPreference,
        vsync: bool,
        frame_latency: u32,
    ) -> bool {
        self.surface.is_some()
            && self.present_options_change(
                self.wanted_present_mode(present, vsync),
                clamp_frame_latency(frame_latency),
            )
    }

    /// The `desired_maximum_frame_latency` the swap chain was last configured with. `wgpu`
    /// clamps it to the surface's supported range when configuring.
    #[must_use]
    pub fn frame_latency(&self) -> u32 {
        self.config.desired_maximum_frame_latency
    }

    /// The preference this context was built with, or last had set.
    #[must_use]
    pub fn present_preference(&self) -> PresentPreference {
        self.present
    }

    /// The present mode the swap chain is currently configured with — for the F3 overlay and for
    /// the T3.2 before/after measurement, which needs to know what it actually ran under.
    #[must_use]
    pub fn present_mode(&self) -> wgpu::PresentMode {
        self.config.present_mode
    }

    /// Resolve a preference against *this* surface's capabilities. A headless context has no
    /// surface to ask, so it keeps the unresolved auto/explicit mode, which is what it always did.
    fn wanted_present_mode(&self, present: PresentPreference, vsync: bool) -> wgpu::PresentMode {
        match &self.surface {
            Some(surface) => resolve_present_mode(
                present,
                vsync,
                &surface.get_capabilities(&self.adapter).present_modes,
            ),
            None => match present {
                PresentPreference::Auto => present_mode(vsync),
                other => explicit_present_mode(other),
            },
        }
    }

    /// Acquire this frame's colour target.
    ///
    /// # Errors
    /// [`RenderError::FrameUnavailable`] means "skip this frame and try again" (the window is
    /// occluded, or the compositor timed out). Outdated and lost surfaces are reconfigured and
    /// retried once before that is returned.
    pub fn acquire(&mut self) -> Result<AcquiredFrame, RenderError> {
        // An export renders into the offscreen target, like a headless context.
        if self.surface.is_none() || self.render_override.is_some() {
            let texture = self
                .offscreen
                .as_ref()
                .ok_or(RenderError::FrameUnavailable)?;
            return Ok(AcquiredFrame {
                view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
                surface_texture: None,
                recorded: None,
            });
        }
        self.acquire_window()
    }

    /// Keep the CPU at most [`OFFSCREEN_FRAMES_IN_FLIGHT`] offscreen frames ahead
    /// of the GPU, as a swap chain's frame latency does in the window. Called by
    /// [`Self::submit_frame`] with the frame it just submitted, only while rendering offscreen.
    ///
    /// A video export renders offscreen and uncapped, and only its 20 Hz preview ever waits for
    /// a swap-chain image. On a GPU slower than the CPU's frame loop (an AMD iGPU at 4K with the
    /// Classic pack and volumetric clouds: about 80 ms of GPU work per 2 ms CPU frame) whole
    /// seconds of frames piled up in the queue, and AMD's Vulkan driver then lost the device.
    /// Waiting here, after the frame's submit, rather than at
    /// acquire: a device loss this wait uncovers then meets the shell's lost-device check at the
    /// start of the next frame, before anything new is staged or uploaded.
    fn pace_offscreen(&self, frame: &wgpu::SubmissionIndex) {
        if !offscreen_pacing_enabled() {
            return;
        }
        let oldest = {
            let mut marks = self
                .offscreen_marks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            marks.push_back(frame.clone());
            if marks.len() > OFFSCREEN_FRAMES_IN_FLIGHT {
                marks.pop_front()
            } else {
                None
            }
        };
        if let Some(oldest) = oldest {
            mc_core::profile_scope!("gpu.offscreen_pacing");
            // A timeout or a lost device: carry on; the device-lost record reports a loss.
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(oldest),
                timeout: Some(OFFSCREEN_WAIT_TIMEOUT),
            });
        }
    }

    /// The window's own image, even while [`Self::set_render_override`] sends
    /// frames offscreen: what a video export draws its live preview into. Its size is
    /// [`Self::window_size`], not `config`'s. Same errors and reconfigure policy as
    /// [`Self::acquire`]; [`RenderError::FrameUnavailable`] for a headless context.
    ///
    /// # Errors
    /// As [`Self::acquire`].
    pub fn acquire_window(&mut self) -> Result<AcquiredFrame, RenderError> {
        let Some(surface) = &self.surface else {
            return Err(RenderError::FrameUnavailable);
        };

        let mut acquired = surface.get_current_texture();
        if matches!(
            acquired,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost
        ) && reconfigure_allowed(
            self.zero_area,
            self.reconfigure_backoff_until,
            std::time::Instant::now(),
        ) {
            surface.configure(&self.device, &self.surface_config());
            acquired = surface.get_current_texture();
            if !matches!(
                acquired,
                wgpu::CurrentSurfaceTexture::Success(_)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(_)
            ) {
                // PRESENT-7: the retry failed too (occluded, mid-transition). Do not drain the
                // GPU and rebuild the swap chain again on every loop until something changes.
                self.reconfigure_backoff_until =
                    Some(std::time::Instant::now() + FAILED_RECONFIGURE_BACKOFF);
            }
        }

        match acquired {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => {
                self.reconfigure_backoff_until = None;
                let view = tex
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                Ok(AcquiredFrame {
                    view,
                    surface_texture: Some(tex),
                    recorded: None,
                })
            }
            _ => Err(RenderError::FrameUnavailable),
        }
    }

    /// The window surface, for the pipelined present thread. `None` for a headless context.
    pub(crate) fn shared_surface(&self) -> Option<Arc<wgpu::Surface<'static>>> {
        self.surface.clone()
    }

    /// Configure the window's swap chain again now, at the window's
    /// size (nothing while minimised or headless): every image it hands out is fresh. The caller
    /// guarantees no surface texture is alive. Returns whether it configured.
    pub(crate) fn rebuild_swap_chain(&mut self) -> bool {
        if self.zero_area {
            return false;
        }
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.surface_config());
            self.reconfigure_backoff_until = None;
            return true;
        }
        false
    }

    /// A pipelined present thread acquired successfully: clear the reconfigure back-off, as the
    /// serial [`Gpu::acquire`] does on success.
    pub(crate) fn note_acquired(&mut self) {
        self.reconfigure_backoff_until = None;
    }

    /// The reconfigure half of [`Gpu::acquire`] for a pipelined present thread that reported an
    /// outdated or lost surface. Same policy: never while zero-area, and not again within the
    /// back-off. The caller guarantees no surface texture is alive (the thread is idle).
    pub(crate) fn reconfigure_after_failed_acquire(&mut self) {
        let Some(surface) = &self.surface else {
            return;
        };
        let now = std::time::Instant::now();
        if reconfigure_allowed(self.zero_area, self.reconfigure_backoff_until, now) {
            surface.configure(&self.device, &self.surface_config());
            // Whether the retry succeeds is only known at the thread's next acquire; until then a
            // persistently failing surface must not be reconfigured on every loop.
            self.reconfigure_backoff_until = Some(now + FAILED_RECONFIGURE_BACKOFF);
        }
    }

    /// Hand every `queue.write_*` staged since the last submit to the GPU without a frame.
    ///
    /// `Queue::write_buffer`/`write_texture` stage into fresh buffers that `wgpu` keeps until the
    /// next `submit`. A loop iteration whose acquire failed has run `ShellApp::frame` (which
    /// stages entity, particle, UI and uniform data) but records nothing, so without this every
    /// skipped iteration's staging stays alive until the window is visible again — unbounded
    /// memory while minimised, then one large burst on restore (PRESENT-7). An empty submit
    /// executes exactly those writes, in order, earlier than the next frame would have; the next
    /// frame's commands still observe all of them, so nothing visible changes.
    ///
    /// The upload ring's copies go in the same submission, after those writes, which is where
    /// `submit_frame` would have put them; without that, every skipped iteration would add its
    /// staged streams (entities, particles, UI) to one ever-growing upload encoder.
    pub fn flush_staged_writes(&self) {
        let staged = self.lock_upload_ring().finish();
        self.queue.submit(staged);
    }

    /// Open a render pass configured the way every phase wants it.
    ///
    /// The colour target is loaded, never cleared: [`PhaseList::render`](crate::pipeline::PhaseList::render)
    /// has already run a clear pass over colour and depth before the first phase records. With
    /// MSAA on, this points the pass at the multisampled attachment and resolves into `target`,
    /// so phases need no MSAA-specific code beyond setting `sample_count` on their pipelines.
    pub fn begin_pass<'e>(
        &self,
        enc: &'e mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        label: &str,
        depth: DepthAttachment,
    ) -> wgpu::RenderPass<'e> {
        let (view, resolve_target) = match &self.msaa_view {
            Some(msaa) => (msaa, Some(target)),
            None => (target, None),
        };
        // The frame's clear, when `PhaseList` handed it to this pass: the
        // colour part if this is the frame's target, the depth part if it attaches the depth.
        let (clear_color, clear_depth) = if self
            .frame_clear_armed
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.take_frame_clear(target, !matches!(depth, DepthAttachment::None))
        } else {
            (None, false)
        };
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: clear_color.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: match depth {
                DepthAttachment::Shared | DepthAttachment::Clear => {
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: if matches!(depth, DepthAttachment::Clear) || clear_depth {
                                wgpu::LoadOp::Clear(1.0)
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    })
                }
                DepthAttachment::None => None,
            },
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Open a pass whose shared depth buffer is attached **read-only**: the
    /// colour target as [`Gpu::begin_pass`] attaches it (loaded, stored, resolved with MSAA), the
    /// depth buffer loaded and never written, so a phase in it may also sample that buffer
    /// ([`Gpu::depth_texture`]) while another depth-tests against it.
    /// Every pipeline drawn in it needs a depth state with depth writes off. `PhaseList` opens it
    /// for a run of phases that opt in with [`crate::RenderPhase::shares_depth_read_pass`].
    pub fn begin_depth_read_pass<'e>(
        &self,
        enc: &'e mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        label: &str,
    ) -> wgpu::RenderPass<'e> {
        let (view, resolve_target) = match &self.msaa_view {
            Some(msaa) => (msaa, Some(target)),
            None => (target, None),
        };
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Hand the frame's clear (colour `color` on `target`, depth to `1.0`) to
    /// the first passes that touch each attachment, instead of a clear pass of its own. Internal
    /// to `PhaseList`, which flushes whatever is left ([`Gpu::flush_frame_clear`]) before any
    /// phase that did not opt in and at the end of the frame.
    pub(crate) fn arm_frame_clear(&self, target: &wgpu::TextureView, color: wgpu::Color) {
        let mut clear = self.lock_frame_clear();
        clear.color = Some((target.clone(), color));
        clear.depth = true;
        self.frame_clear_armed
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the frame's colour clear and depth clear are still pending.
    pub(crate) fn frame_clear_pending(&self) -> (bool, bool) {
        if !self
            .frame_clear_armed
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return (false, false);
        }
        let clear = self.lock_frame_clear();
        (clear.color.is_some(), clear.depth)
    }

    /// Record what is left of the frame's clear as one pass, exactly as the
    /// frame's clear pass records it (colour and depth, or the one still pending), and disarm.
    pub(crate) fn flush_frame_clear(&self, enc: &mut wgpu::CommandEncoder) {
        if !self
            .frame_clear_armed
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let (color, depth) = {
            let mut clear = self.lock_frame_clear();
            self.frame_clear_armed
                .store(false, std::sync::atomic::Ordering::Relaxed);
            (clear.color.take(), std::mem::take(&mut clear.depth))
        };
        match (color, depth) {
            (Some((target, color)), true) => drop(self.begin_clear_pass(enc, &target, color)),
            (Some((target, color)), false) => {
                drop(self.begin_clear_color_pass(enc, &target, color));
            }
            (None, true) => drop(enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mc-render clear depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })),
            (None, false) => {}
        }
    }

    /// Take the parts of the pending frame clear a pass on `target` (attaching
    /// the shared depth buffer when `depth_attached`) performs.
    fn take_frame_clear(
        &self,
        target: &wgpu::TextureView,
        depth_attached: bool,
    ) -> (Option<wgpu::Color>, bool) {
        let mut clear = self.lock_frame_clear();
        let color = match &clear.color {
            Some((view, color)) if view == target => {
                let color = *color;
                clear.color = None;
                Some(color)
            }
            _ => None,
        };
        let depth = depth_attached && std::mem::take(&mut clear.depth);
        if clear.color.is_none() && !clear.depth {
            self.frame_clear_armed
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        (color, depth)
    }

    fn lock_frame_clear(&self) -> std::sync::MutexGuard<'_, FrameClear> {
        self.frame_clear
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Open the frame's clearing pass. Internal to the frame loop — phases use
    /// [`Gpu::begin_pass`], which loads instead of clearing.
    pub(crate) fn begin_clear_pass<'e>(
        &self,
        enc: &'e mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clear: wgpu::Color,
    ) -> wgpu::RenderPass<'e> {
        let (view, resolve_target) = match &self.msaa_view {
            Some(msaa) => (msaa, Some(target)),
            None => (target, None),
        };
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mc-render clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Open a colour-only clearing pass, preserving the frame's MSAA resolve path.
    ///
    /// Post-processing pipelines do not have a depth-stencil attachment.  They still need a
    /// fresh colour target (and, when MSAA is enabled, the same multisampled attachment and
    /// resolve target used by the ordinary clear pass), but binding the shared depth view would
    /// make those pipelines incompatible with their render pass.
    pub(crate) fn begin_clear_color_pass<'e>(
        &self,
        enc: &'e mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clear: wgpu::Color,
    ) -> wgpu::RenderPass<'e> {
        let (view, resolve_target) = match &self.msaa_view {
            Some(msaa) => (msaa, Some(target)),
            None => (target, None),
        };
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mc-render clear colour"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Copy the offscreen colour target back to the CPU as tightly packed RGBA8 rows.
    ///
    /// Headless contexts use this for automated checks. Windowed screenshots use
    /// [`AcquiredFrame::read_pixels`] so the acquired surface texture remains alive through the
    /// copy and is only presented afterwards.
    ///
    /// # Errors
    /// [`RenderError::Headless`] if this context has a surface, [`RenderError::Readback`] if the
    /// staging buffer cannot be mapped.
    pub fn read_pixels(&self) -> Result<Vec<u8>, RenderError> {
        let texture = self
            .offscreen
            .as_ref()
            .ok_or(RenderError::Headless("read_pixels"))?;

        self.read_texture_pixels(texture)
    }

    /// Read a single-sampled RGBA/BGRA colour texture into tightly packed RGBA8 rows.
    ///
    /// `Texture::format` is not exposed by wgpu, but every texture handed to this helper is
    /// created from `config.format`; the format conversion therefore lives at this shared seam.
    /// Surface textures are the post-MSAA resolve target, so copying them here observes the same
    /// resolved pixels that presentation will consume.
    fn read_texture_pixels(&self, texture: &wgpu::Texture) -> Result<Vec<u8>, RenderError> {
        let format = self.config.format;
        if !matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Bgra8Unorm
                | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            return Err(RenderError::Readback(format!(
                "screenshot readback requires an RGBA8/BGRA8 surface, got {format:?}"
            )));
        }

        let (width, height) = (self.config.width, self.config.height);
        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc-render readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc-render readback"),
            });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
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
        self.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RenderError::Readback(e.to_string()))?;
        rx.recv()
            .map_err(|_| RenderError::Readback("map callback dropped".into()))?
            .map_err(|e| RenderError::Readback(e.to_string()))?;

        let view = staging
            .get_mapped_range(..)
            .map_err(|e| RenderError::Readback(e.to_string()))?;
        let mut out = Vec::with_capacity((unpadded * height) as usize);
        for row in 0..height as usize {
            let start = row * padded as usize;
            out.extend_from_slice(&view[start..start + unpadded as usize]);
        }
        drop(view);
        staging.unmap();

        if matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for pixel in out.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        Ok(out)
    }
}

/// Convert tightly packed RGBA8 rows to the orientation expected by an image encoder.
///
/// wgpu readback rows are top-to-bottom, while a caller that receives a legacy bottom-left
/// source can request a vertical flip explicitly. Keeping this pure makes the screenshot seam
/// testable without creating a device and prevents an accidental second flip on modern wgpu.
pub fn rgba8_for_png(
    width: u32,
    height: u32,
    pixels: &[u8],
    source_bottom_left: bool,
) -> Result<Vec<u8>, RenderError> {
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| RenderError::Readback("screenshot row is too wide".into()))?;
    let expected = row_bytes
        .checked_mul(height)
        .ok_or_else(|| RenderError::Readback("screenshot is too large".into()))?
        as usize;
    if pixels.len() != expected {
        return Err(RenderError::Readback(format!(
            "screenshot pixel length {} does not match {}x{} RGBA8",
            pixels.len(),
            width,
            height
        )));
    }
    if !source_bottom_left {
        return Ok(pixels.to_vec());
    }
    let mut out = vec![0; expected];
    for row in 0..height as usize {
        let src = row * row_bytes as usize;
        let dst = (height as usize - 1 - row) * row_bytes as usize;
        out[dst..dst + row_bytes as usize].copy_from_slice(&pixels[src..src + row_bytes as usize]);
    }
    Ok(out)
}

/// The atlas stitch cap corresponding to a device's granted limits.
///
/// Kept as a pure helper so the startup handoff can be tested without requiring a window or a
/// GPU adapter.  [`Gpu::atlas_max_dimension`] is the normal production entry point.
#[must_use]
pub fn atlas_max_dimension(limits: &wgpu::Limits) -> u32 {
    limits.max_texture_dimension_2d
}

/// Whether [`Gpu::acquire`] may reconfigure an outdated or lost surface right now.
///
/// Never while the window is zero-area (minimised): there is no valid size to configure, and the
/// restore's `Resized` event reconfigures through [`Gpu::resize`]. Otherwise only outside the
/// back-off window that a failed reconfigure-and-retry opens. Pure, so it is unit-testable
/// without a surface.
fn reconfigure_allowed(
    zero_area: bool,
    backoff_until: Option<std::time::Instant>,
    now: std::time::Instant,
) -> bool {
    !zero_area && backoff_until.is_none_or(|until| now >= until)
}

fn java_surface_format(
    preferred: wgpu::TextureFormat,
    supported: &[wgpu::TextureFormat],
) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as F;
    [preferred.remove_srgb_suffix(), F::Bgra8Unorm, F::Rgba8Unorm]
        .into_iter()
        .find(|format| {
            matches!(format, F::Bgra8Unorm | F::Rgba8Unorm) && supported.contains(format)
        })
        .unwrap_or(preferred)
}

fn present_mode(vsync: bool) -> wgpu::PresentMode {
    if vsync {
        wgpu::PresentMode::AutoVsync
    } else {
        wgpu::PresentMode::AutoNoVsync
    }
}

/// The concrete `wgpu` mode an explicit preference names. [`PresentPreference::Auto`] has no
/// concrete mode of its own and maps to `AutoVsync`, which is what it means with vsync on.
fn explicit_present_mode(present: PresentPreference) -> wgpu::PresentMode {
    match present {
        PresentPreference::Auto => wgpu::PresentMode::AutoVsync,
        PresentPreference::Fifo => wgpu::PresentMode::Fifo,
        PresentPreference::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
        PresentPreference::Mailbox => wgpu::PresentMode::Mailbox,
        PresentPreference::Immediate => wgpu::PresentMode::Immediate,
    }
}

/// Turn a [`PresentPreference`] into the mode to configure the surface with.
///
/// # The fallback chains, and why they are what they are
///
/// This rig reports `[Fifo, FifoRelaxed, Mailbox, Immediate]`, so every preference is available
/// here and every chain below is a no-op *on this machine*. They exist because the next machine
/// is not this one: `Fifo` is the only mode the WebGPU/Vulkan surface contract guarantees, and
/// `Mailbox` in particular is commonly missing (it is optional in Vulkan and unavailable on some
/// compositors entirely).
///
/// | preference | chain | rationale |
/// |---|---|---|
/// | `Auto` | `AutoVsync` / `AutoNoVsync` | exactly the behaviour that existed before T3.2 |
/// | `Fifo` | `Fifo` | always supported; no chain needed |
/// | `FifoRelaxed` | `FifoRelaxed` → `Fifo` | both block on vblank |
/// | `Mailbox` | `Mailbox` → `Fifo` | **never `Immediate`** — see below |
/// | `Immediate` | `Immediate` → `Mailbox` → `Fifo` | the setting asks not to wait; `Mailbox` is the closest thing that also does not wait |
///
/// The asymmetry is deliberate and is the one judgement call in this function. `Mailbox` and
/// `Immediate` are both non-blocking, so it is tempting to treat them as interchangeable
/// fallbacks for each other. They are not: `Mailbox` is tear-free and `Immediate` is not. A user
/// who selects `Mailbox` has asked for a clean image, and quietly handing them a tearing one on a
/// machine that lacks `Mailbox` would be a defect, not a fallback — so that chain drops to `Fifo`,
/// which is slower but still clean. Going the other way is safe, because a user who selected
/// `Immediate` has already accepted tearing and `Mailbox` is strictly better than what they asked
/// for.
///
/// Note that `Auto` is returned **unresolved** (`AutoVsync`/`AutoNoVsync`) rather than being
/// collapsed here, so that path stays byte-identical to the pre-T3.2 code and `wgpu-core` keeps
/// doing its own resolution. [`resolve_auto_present_mode`] predicts what that will be, for the log.
#[must_use]
pub fn resolve_present_mode(
    present: PresentPreference,
    vsync: bool,
    supported: &[wgpu::PresentMode],
) -> wgpu::PresentMode {
    use wgpu::PresentMode as M;
    let chain: &[M] = match present {
        PresentPreference::Auto => return present_mode(vsync),
        PresentPreference::Fifo => &[M::Fifo],
        PresentPreference::FifoRelaxed => &[M::FifoRelaxed, M::Fifo],
        PresentPreference::Mailbox => &[M::Mailbox, M::Fifo],
        PresentPreference::Immediate => &[M::Immediate, M::Mailbox, M::Fifo],
    };
    chain
        .iter()
        .copied()
        .find(|mode| supported.contains(mode))
        // `Fifo` is guaranteed by the surface contract, so this is unreachable in practice; it is
        // still the right last resort rather than a panic in a code path that opens a window.
        .unwrap_or(M::Fifo)
}

/// Predict what `wgpu-core` will resolve `requested` to against a surface's supported modes.
///
/// Diagnostic-only (see the log line in [`Gpu::for_window`]) — this mirrors the fallback order
/// documented on `wgpu::PresentMode::AutoVsync`/`AutoNoVsync` themselves, not a private
/// implementation detail: `AutoVsync` tries `FifoRelaxed` then `Fifo`; `AutoNoVsync` tries
/// `Immediate`, `Mailbox`, then `Fifo`. `Fifo` is guaranteed supported everywhere, so both lists
/// always resolve to *something* — the point of logging it is to see *which*, since `FifoRelaxed`
/// and `Fifo` both block on vblank but `Immediate`/`Mailbox` do not.
fn resolve_auto_present_mode(
    requested: wgpu::PresentMode,
    supported: &[wgpu::PresentMode],
) -> wgpu::PresentMode {
    if supported.contains(&requested) {
        return requested;
    }
    let fallbacks: &[wgpu::PresentMode] = match requested {
        wgpu::PresentMode::AutoVsync => &[wgpu::PresentMode::FifoRelaxed, wgpu::PresentMode::Fifo],
        wgpu::PresentMode::AutoNoVsync => &[
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Fifo,
        ],
        other => return other,
    };
    fallbacks
        .iter()
        .copied()
        .find(|mode| supported.contains(mode))
        .unwrap_or(wgpu::PresentMode::Fifo)
}

fn clamp_sample_count(adapter: &wgpu::Adapter, format: wgpu::TextureFormat, requested: u32) -> u32 {
    if requested <= 1 {
        return 1;
    }
    // `request_device` never asks for `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`, so
    // the device accepts only the sample counts WebGPU guarantees (1 and 4) whatever more the
    // adapter reports, and the shared depth buffer must accept the count too. Taking the
    // adapter's word alone made `msaaSamples:2` and `:8` a validation panic at start-up on every
    // GPU.
    let allowed = |format: wgpu::TextureFormat| {
        adapter.get_texture_format_features(format).flags
            & format
                .guaranteed_format_features(wgpu::Features::empty())
                .flags
    };
    let flags = allowed(format) & allowed(DEPTH_FORMAT);
    let supported = |n: u32| match n {
        2 => flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X2),
        4 => flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4),
        8 => flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X8),
        _ => false,
    };
    // Fall back down the ladder rather than failing: an unsupported MSAA level in options.txt
    // must not stop the game from starting.
    [requested, 8, 4, 2]
        .into_iter()
        .find(|&n| n <= requested && supported(n))
        .unwrap_or(1)
}

fn create_depth(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    // `RENDER_ATTACHMENT` is what the depth buffer has always needed. `TEXTURE_BINDING` is
    // additive, requested for the screen-space fog pass —
    // see `Gpu::depth_texture`. `COPY_SRC` is only legal on a single-sampled texture, so it is
    // added conditionally rather than unconditionally; nothing in-tree needs it yet, but a
    // depth readback (S22's parity harness) would.
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    if sample_count == 1 {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mc-render depth"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_msaa(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
) -> Option<wgpu::TextureView> {
    if sample_count <= 1 {
        return None;
    }
    Some(
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("mc-render msaa colour"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::PresentMode as M;

    /// What one development PC's GPU reported.
    const THIS_RIG: [M; 4] = [M::Fifo, M::FifoRelaxed, M::Mailbox, M::Immediate];

    /// The windowed path ran with `get_default_config`'s latency of 2 before `frameLatency`
    /// existed; an unset option must keep exactly that.
    #[test]
    fn frame_latency_default_preserves_the_historical_two() {
        assert_eq!(DEFAULT_FRAME_LATENCY, 2);
        assert_eq!(clamp_frame_latency(0), 2, "unset keeps the old value");
        assert_eq!(clamp_frame_latency(1), 1, "the low-latency option");
        assert_eq!(clamp_frame_latency(2), 2);
        assert_eq!(clamp_frame_latency(3), 3);
        assert_eq!(clamp_frame_latency(99), MAX_FRAME_LATENCY);
    }

    #[test]
    fn backend_override_parses_and_defaults_to_auto() {
        assert_eq!(BackendOverride::default(), BackendOverride::Auto);
        assert_eq!(BackendOverride::parse(""), Some(BackendOverride::Auto));
        assert_eq!(
            BackendOverride::parse(" AUTO "),
            Some(BackendOverride::Auto)
        );
        assert_eq!(
            BackendOverride::parse("Vulkan"),
            Some(BackendOverride::Vulkan)
        );
        assert_eq!(BackendOverride::parse("dx12"), Some(BackendOverride::Dx12));
        assert_eq!(BackendOverride::parse("d3d12"), Some(BackendOverride::Dx12));
        assert_eq!(BackendOverride::parse("metal"), None);
        assert_eq!(
            BackendOverride::Auto.backends(),
            None,
            "auto keeps the old policy"
        );
        assert_eq!(BackendOverride::Dx12.backends(), Some(wgpu::Backends::DX12));
    }

    /// PRESENT-7: an outdated surface is reconfigured at once (unchanged behaviour), never while
    /// minimised, and not again within the back-off after a failed retry.
    #[test]
    fn acquire_reconfigure_policy() {
        let now = std::time::Instant::now();
        assert!(
            reconfigure_allowed(false, None, now),
            "first failure: retry as before"
        );
        assert!(
            !reconfigure_allowed(true, None, now),
            "zero-area: never configure"
        );
        let until = now + FAILED_RECONFIGURE_BACKOFF;
        assert!(
            !reconfigure_allowed(false, Some(until), now),
            "inside back-off"
        );
        assert!(
            reconfigure_allowed(false, Some(until), until),
            "back-off over"
        );
    }

    #[test]
    fn java_surface_uses_supported_byte_domain_format() {
        use wgpu::TextureFormat as F;
        assert_eq!(
            java_surface_format(F::Bgra8UnormSrgb, &[F::Bgra8UnormSrgb, F::Bgra8Unorm]),
            F::Bgra8Unorm,
        );
        assert_eq!(
            java_surface_format(F::Bgra8UnormSrgb, &[F::Bgra8UnormSrgb, F::Rgba8Unorm]),
            F::Rgba8Unorm,
        );
        assert_eq!(
            java_surface_format(F::Rgba8Unorm, &[F::Bgra8Unorm, F::Rgba8Unorm]),
            F::Rgba8Unorm,
        );
        // Never configure an unsupported surface format on sRGB-only platforms.
        assert_eq!(
            java_surface_format(F::Bgra8UnormSrgb, &[F::Bgra8UnormSrgb]),
            F::Bgra8UnormSrgb,
        );
    }

    /// A resource-pack atlas must be capped by the active device, not the historical 2048 test
    /// fixture.  The two controls make a hardcoded fallback and an accidental adapter-default
    /// lookup both fail this test without needing a real GPU in CI.
    #[test]
    fn atlas_cap_is_the_granted_2d_texture_limit_exactly() {
        let small_device = wgpu::Limits {
            max_texture_dimension_2d: 1_024,
            ..wgpu::Limits::default()
        };
        let modern_device = wgpu::Limits {
            max_texture_dimension_2d: 16_384,
            ..wgpu::Limits::default()
        };

        assert_eq!(atlas_max_dimension(&small_device), 1_024);
        assert_eq!(atlas_max_dimension(&modern_device), 16_384);
        assert_ne!(atlas_max_dimension(&modern_device), 2_048);
    }

    /// The control for every other test in this module: with the default preference, T3.2 must
    /// not have changed the mode the surface is configured with at all.
    #[test]
    fn auto_reproduces_the_pre_t3_2_behaviour() {
        assert_eq!(
            resolve_present_mode(PresentPreference::Auto, true, &THIS_RIG),
            M::AutoVsync
        );
        assert_eq!(
            resolve_present_mode(PresentPreference::Auto, false, &THIS_RIG),
            M::AutoNoVsync
        );
        // Even against a surface that supports nothing but Fifo, `Auto` stays unresolved and
        // leaves the decision to wgpu-core, exactly as the old `present_mode(vsync)` did.
        assert_eq!(
            resolve_present_mode(PresentPreference::Auto, true, &[M::Fifo]),
            M::AutoVsync
        );
        assert_eq!(PresentPreference::default(), PresentPreference::Auto);
    }

    #[test]
    fn every_preference_is_honoured_on_a_surface_that_supports_them_all() {
        for (pref, expected) in [
            (PresentPreference::Fifo, M::Fifo),
            (PresentPreference::FifoRelaxed, M::FifoRelaxed),
            (PresentPreference::Mailbox, M::Mailbox),
            (PresentPreference::Immediate, M::Immediate),
        ] {
            assert_eq!(
                resolve_present_mode(pref, true, &THIS_RIG),
                expected,
                "{pref:?} on this rig"
            );
            // An explicit preference overrides `enableVsync`; that is the documented contract.
            assert_eq!(resolve_present_mode(pref, false, &THIS_RIG), expected);
        }
    }

    /// The judgement call in `resolve_present_mode`, pinned: asking for a tear-free mode must
    /// never quietly deliver a tearing one.
    #[test]
    fn mailbox_falls_back_to_fifo_and_never_to_immediate() {
        let no_mailbox = [M::Fifo, M::FifoRelaxed, M::Immediate];
        let got = resolve_present_mode(PresentPreference::Mailbox, true, &no_mailbox);
        assert_eq!(got, M::Fifo);
        assert_ne!(
            got,
            M::Immediate,
            "Immediate tears; a Mailbox request must not land on it"
        );
        assert!(PresentPreference::Mailbox.is_tear_free());
        assert!(!PresentPreference::Immediate.is_tear_free());
        // FifoRelaxed is not tear-free either — it shows a late frame immediately.
        assert!(!PresentPreference::FifoRelaxed.is_tear_free());
    }

    /// The other direction is safe, and takes it: a user who accepted tearing gets the better
    /// non-blocking mode when the one they named is missing.
    #[test]
    fn immediate_falls_back_to_mailbox_before_fifo() {
        let no_immediate = [M::Fifo, M::Mailbox];
        assert_eq!(
            resolve_present_mode(PresentPreference::Immediate, true, &no_immediate),
            M::Mailbox
        );
        assert_eq!(
            resolve_present_mode(PresentPreference::Immediate, true, &[M::Fifo]),
            M::Fifo
        );
    }

    /// A surface that (against the spec) reports nothing at all must still yield a legal mode
    /// rather than panic in the code path that opens the game window.
    #[test]
    fn an_empty_capability_list_still_resolves() {
        for pref in PresentPreference::ALL {
            let got = resolve_present_mode(pref, true, &[]);
            assert!(
                matches!(got, M::Fifo | M::AutoVsync),
                "{pref:?} resolved to {got:?}"
            );
        }
    }

    #[test]
    fn preference_names_round_trip() {
        for pref in PresentPreference::ALL {
            assert_eq!(PresentPreference::parse(pref.as_str()), Some(pref));
        }
        // Spelling tolerance, so a hand-edited options.txt is forgiving like the rest of the file.
        assert_eq!(
            PresentPreference::parse("  FifoRelaxed "),
            Some(PresentPreference::FifoRelaxed)
        );
        assert_eq!(
            PresentPreference::parse("fifo-relaxed"),
            Some(PresentPreference::FifoRelaxed)
        );
        assert_eq!(
            PresentPreference::parse("MAILBOX"),
            Some(PresentPreference::Mailbox)
        );
        assert_eq!(PresentPreference::parse("triple-buffered"), None);
        assert_eq!(PresentPreference::parse(""), None);
    }

    #[test]
    fn png_orientation_flips_only_a_bottom_left_source_and_rejects_bad_rows() {
        // Two rows make the vertical contract visible without a GPU.  The screenshot readback
        // path passes `false` because wgpu already returns top-left rows; legacy GL callers can
        // opt into the explicit flip.
        let top = [0x10, 0x11, 0x12, 0xFF];
        let bottom = [0x20, 0x21, 0x22, 0xFF];
        let mut rows = Vec::new();
        rows.extend_from_slice(&top);
        rows.extend_from_slice(&bottom);
        assert_eq!(rgba8_for_png(1, 2, &rows, false).unwrap(), rows);
        let flipped = rgba8_for_png(1, 2, &rows, true).unwrap();
        assert_eq!(&flipped[..4], &bottom);
        assert_eq!(&flipped[4..], &top);
        assert!(rgba8_for_png(1, 2, &rows[..4], false).is_err());
    }

    /// Outside an export `submit_frame` never paces (normal play and tests pay
    /// nothing); while rendering offscreen it keeps at most `OFFSCREEN_FRAMES_IN_FLIGHT` frames
    /// queued and still delivers every frame. `MC_GPU_ADAPTER` picks the GPU (an AMD iGPU too).
    #[test]
    fn offscreen_frames_are_paced_only_during_an_export() {
        let Ok(mut gpu) = Gpu::headless(64, 64, GpuOptions::default()) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let draw = |gpu: &Gpu, frame: &AcquiredFrame| {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            drop(gpu.begin_clear_color_pass(&mut encoder, frame.view(), wgpu::Color::RED));
            let _ = gpu.submit_frame(encoder.finish());
        };
        let marks = |gpu: &Gpu| gpu.offscreen_marks.lock().map_or(usize::MAX, |m| m.len());
        for _ in 0..4 {
            let frame = gpu.acquire().expect("headless frame");
            draw(&gpu, &frame);
        }
        assert_eq!(marks(&gpu), 0, "paced outside an export");

        assert!(gpu.set_render_override(Some((48, 32))));
        for _ in 0..6 {
            let frame = gpu.acquire().expect("offscreen frame");
            draw(&gpu, &frame);
            assert!(marks(&gpu) <= OFFSCREEN_FRAMES_IN_FLIGHT);
        }
        if offscreen_pacing_enabled() {
            assert_eq!(marks(&gpu), OFFSCREEN_FRAMES_IN_FLIGHT);
        }
        assert!(gpu.set_render_override(None));
        assert_eq!((gpu.config.width, gpu.config.height), (64, 64));
    }

    /// `msaaSamples` 2 and 8 are not counts WebGPU guarantees, and the device has
    /// no adapter-specific format features, so they fall back (8 -> 4, 2 -> 1) instead of
    /// failing validation when the depth buffer is made (a panic at start-up on every GPU).
    #[test]
    fn unguaranteed_msaa_counts_fall_back_instead_of_failing_validation() {
        for (requested, expected) in [(8, 4), (4, 4), (2, 1), (1, 1)] {
            let opts = GpuOptions {
                vsync: false,
                sample_count: requested,
                low_power: false,
            };
            let Ok(gpu) = Gpu::headless(32, 32, opts) else {
                eprintln!("no GPU adapter; skipped");
                return;
            };
            assert_eq!(gpu.sample_count(), expected, "msaaSamples:{requested}");
        }
    }

    /// The shell takes back the pipelined present thread's image
    /// only when `resize_configures` or `present_options_configure` say a configure follows, so
    /// their tests (`resize_changes_window_size`, `present_options_change`) must predict exactly
    /// what `resize` and `set_present_options` do. Headless (no swap chain to configure),
    /// `resize` rebuilds, or during an export moves the swap chain's size, exactly when the test
    /// says so, and `set_present_options` changes the configured mode or latency exactly then. A
    /// repeated size, 0×0, VSync with an explicit mode and an unset latency configure nothing.
    #[test]
    fn the_shell_knows_exactly_when_the_swap_chain_is_configured() {
        use PresentPreference as P;
        let Ok(mut gpu) = Gpu::headless(64, 48, GpuOptions::default()) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        assert!(!gpu.resize_configures(80, 60), "headless: no swap chain");
        let sizes = [
            (64, 48),
            (0, 0),
            (64, 48),
            (0, 48),
            (80, 60),
            (80, 60),
            (64, 48),
            (1, 1),
            (64, 48),
        ];
        let mut changes = 0;
        for (w, h) in sizes {
            let predicted = gpu.resize_changes_window_size(w, h);
            assert_eq!(predicted, gpu.resize(w, h), "window resize to {w}x{h}");
            changes += usize::from(predicted);
        }
        assert_eq!(changes, 4, "80x60, 64x48, 1x1 and 64x48 change the size");
        assert!(gpu.set_render_override(Some((32, 32))));
        for (w, h) in sizes {
            let before = gpu.window_size();
            let predicted = gpu.resize_changes_window_size(w, h);
            assert!(
                !gpu.resize(w, h),
                "an export's attachments keep the export's size"
            );
            let moved = gpu.window_size() != before;
            assert_eq!(predicted, moved, "export resize to {w}x{h}");
        }
        assert!(gpu.set_render_override(None));

        for preference in [P::Auto, P::Fifo, P::FifoRelaxed, P::Mailbox, P::Immediate] {
            for vsync in [true, false] {
                for latency in [0, 1, 2, 3, 4] {
                    let before = (gpu.present_mode(), gpu.frame_latency());
                    let mode = gpu.wanted_present_mode(preference, vsync);
                    let predicted = gpu.present_options_change(mode, clamp_frame_latency(latency));
                    gpu.set_present_options(preference, vsync, latency);
                    let after = (gpu.present_mode(), gpu.frame_latency());
                    assert_eq!(
                        predicted,
                        after != before,
                        "{preference:?}, vsync {vsync}, latency {latency}"
                    );
                }
            }
        }
        let unchanged = |gpu: &Gpu, preference: P, vsync: bool, latency: u32| {
            let mode = gpu.wanted_present_mode(preference, vsync);
            !gpu.present_options_change(mode, clamp_frame_latency(latency))
        };
        gpu.set_present_options(P::Immediate, true, 2);
        assert!(
            unchanged(&gpu, P::Immediate, false, 2),
            "VSync with an explicit mode"
        );
        assert!(
            unchanged(&gpu, P::Immediate, true, 0),
            "an unset latency is the default 2"
        );
        assert!(!unchanged(&gpu, P::Immediate, true, 1));
        gpu.set_present_options(P::Auto, true, 2);
        assert!(
            !unchanged(&gpu, P::Auto, false, 2),
            "VSync with the auto mode switches modes"
        );
    }
}
