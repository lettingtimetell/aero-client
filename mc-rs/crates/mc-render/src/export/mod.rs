//! Video export support: the GPU half of turning a replay into an
//! MP4. The encoder itself is `mc-video`; the export driver is `mc-client`'s
//! `app/mods/tw_replay_export.rs`.
//!
//! * The frame graph renders offscreen at the export's resolution while
//!   [`Gpu::set_render_override`](crate::Gpu::set_render_override) is in force (`context.rs`);
//!   the shell then skips the window's present and calls the app's preview hook instead.
//! * [`capture::ExportCapture`]: frame-blending motion blur, NV12 conversion on the GPU, and a
//!   ring of readback buffers so the GPU never waits for the CPU.
//! * [`preview::PreviewCompositor`]: the window's live preview and progress panel.
//!
//! Nothing here exists or runs outside an export.

pub mod capture;
pub mod preview;

pub use capture::{CaptureStats, Captured, ExportCapture};
pub use preview::{read_rgba, PreviewCompositor, PreviewLayout};
