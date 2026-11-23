//! F2 without the freeze, plus `[COPY]`.
//!
//! Before this lane, F2 read the frame back with a blocking `device.poll(wait)` and encoded the
//! PNG on the game thread inside `after_render`: a 1080p screenshot stalled the frame for the
//! whole GPU wait plus the PNG encode (tens to hundreds of milliseconds). Now:
//!
//! 1. **The F2 frame** (`after_render`, [`Client::x2_screenshot_after_render`]): one copy of the
//!    composited surface texture into a fresh `MAP_READ` buffer is recorded and submitted
//!    (through the gated queue, so a pipelined frame is submitted first), and `map_async` is
//!    armed with a callback that only stores a flag. No wait.
//! 2. **Later frames** ([`Client::x2_frame`]): while a readback is pending, a
//!    non-blocking `device.poll` and a flag check; once mapped, the buffer itself is handed to the
//!    encoder thread. The game thread never touches the pixels.
//! 3. **The encoder thread** (started on the first F2, then reused): de-pads the rows, swaps BGRA,
//!    unmaps, reserves the file name and writes the PNG, then sends the result back. The next
//!    frame posts vanilla's chat line, "Saved screenshot as <name>", the name underlined and
//!    clickable (opens the file), followed by a clickable **`[COPY]`**.
//! 4. **`[COPY]`** (a click in the chat screen, [`Client::x2_copy_screenshot`]): the encoder thread
//!    reads the PNG back, decodes it and puts it on the Windows clipboard as `CF_DIB` (24-bit) plus
//!    the registered `PNG` format, then the chat says so. The click sends nothing to the server.
//!
//! File names are vanilla's (`ScreenShotHelper.getTimestampedPNGFileForDirectory`):
//! `yyyy-MM-dd_HH.mm.ss.png`, then `_2`, `_3`, ... within the same second. The timestamp is taken
//! when F2 is pressed.
//!
//! **Cost when idle:** [`Client::x2_frame`] is one `bool` test (`ShotPipeline::idle`); the F2 path
//! in `after_render` is the `screenshot_requested` test it always was.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

use mc_render::{AcquiredFrame, Gpu};
use mc_ui::font::component::{
    ChatComponent, ChatStyle, ClickAction, ClickEvent, HoverAction, HoverEvent,
};
use mc_ui::font::TextFormat;
use mc_ui::screens::chat::ChatLayout;

use super::super::{Client, Ui};

/// `map_async` has not called back yet.
const WAITING: u8 = 0;
/// The buffer is mapped and readable.
const MAPPED: u8 = 1;
/// The map failed (device lost, buffer destroyed).
const FAILED: u8 = 2;

/// The text of the `[COPY]` link.
pub(crate) const COPY_LABEL: &str = "[COPY]";
const COPY_HOVER: &str = "Copy the screenshot to the clipboard";
const COPIED: &str = "Copied the screenshot to the clipboard.";

/// How long shutdown waits for a screenshot still being written.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(3);

/// Where the readback rows sit in the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadbackLayout {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Bytes per row in the buffer (`width * 4` rounded up to wgpu's 256-byte copy alignment).
    pub(crate) padded: u32,
    /// The surface is BGRA: swap to RGBA.
    pub(crate) bgra: bool,
}

/// A copy submitted on the F2 frame whose `map_async` is still outstanding.
#[derive(Debug)]
pub(crate) struct Readback {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) state: Arc<AtomicU8>,
    pub(crate) layout: ReadbackLayout,
}

#[derive(Debug)]
struct Pending {
    readback: Readback,
    timestamp: String,
    directory: PathBuf,
}

/// Work for the encoder thread.
#[derive(Debug)]
enum Job {
    /// Turn a mapped readback into a PNG in `directory`.
    Encode {
        readback: Readback,
        timestamp: String,
        directory: PathBuf,
    },
    /// Put the PNG at `path` on the clipboard.
    Copy { path: PathBuf },
}

/// What the encoder thread sends back.
#[derive(Debug)]
enum Done {
    Saved(Result<PathBuf, String>),
    Copied(Result<(), String>),
}

#[derive(Debug)]
struct Worker {
    jobs: mpsc::Sender<Job>,
    done: mpsc::Receiver<Done>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn spawn() -> io::Result<Self> {
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (done_tx, done) = mpsc::channel::<Done>();
        let thread = std::thread::Builder::new()
            .name("mc-rs screenshots".to_owned())
            .spawn(move || {
                for job in job_rx {
                    if done_tx.send(run_job(job)).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            jobs,
            done,
            thread: Some(thread),
        })
    }
}

/// F2's state: readbacks waiting for their map, the encoder thread, jobs it has not answered.
#[derive(Debug, Default)]
pub(crate) struct ShotPipeline {
    pending: Vec<Pending>,
    worker: Option<Worker>,
    in_flight: usize,
}

impl ShotPipeline {
    /// Nothing to poll: no readback waiting, no job out.
    pub(crate) fn idle(&self) -> bool {
        self.pending.is_empty() && self.in_flight == 0
    }

    /// Jobs handed to the encoder thread whose answer has not been posted yet.
    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// Hand `job` to the encoder thread (started on first use). A thread that cannot start runs
    /// the job inline instead, so a screenshot is never lost; a dead one is replaced.
    fn submit(&mut self, job: Job) -> Option<Done> {
        if self.worker.is_none() {
            match Worker::spawn() {
                Ok(worker) => self.worker = Some(worker),
                Err(error) => {
                    tracing::warn!(%error, "could not start the screenshot thread; working inline");
                    return Some(run_job(job));
                }
            }
        }
        let worker = self.worker.as_ref().expect("started above");
        match worker.jobs.send(job) {
            Ok(()) => {
                self.in_flight += 1;
                None
            }
            Err(mpsc::SendError(job)) => {
                // The thread is gone (it panicked). Its unanswered jobs are lost; start afresh.
                tracing::warn!("the screenshot thread stopped; working inline");
                self.worker = None;
                self.in_flight = 0;
                Some(run_job(job))
            }
        }
    }

    /// The next answer from the encoder thread, if one is ready.
    fn try_done(&mut self) -> Option<Done> {
        let worker = self.worker.as_ref()?;
        match worker.done.try_recv() {
            Ok(done) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                Some(done)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.worker = None;
                self.in_flight = 0;
                None
            }
        }
    }
}

/// Record and submit the copy of `texture` (the composited frame) into a fresh readback buffer
/// and arm `map_async`. Returns at once: the GPU work and the map complete on later frames.
pub(crate) fn start_readback(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<Readback, String> {
    let bgra = match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
        other => {
            return Err(format!(
                "screenshot readback requires an RGBA8/BGRA8 surface, got {other:?}"
            ))
        }
    };
    if width == 0 || height == 0 {
        return Err("the window has no area".to_owned());
    }
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = (width * 4).div_ceil(align) * align;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("x2 screenshot readback"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("x2 screenshot readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
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
    // The gated queue submits a frame the pipelined path still owes first, so this copy reads
    // the finished image.
    gpu.queue.submit(Some(encoder.finish()));
    let state = Arc::new(AtomicU8::new(WAITING));
    let flag = Arc::clone(&state);
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        flag.store(
            if result.is_ok() { MAPPED } else { FAILED },
            Ordering::Release,
        );
    });
    Ok(Readback {
        buffer,
        state,
        layout: ReadbackLayout {
            width,
            height,
            padded,
            bgra,
        },
    })
}

/// Tightly packed RGBA8 rows from a mapped readback buffer (then unmapped).
pub(crate) fn read_mapped(readback: &Readback) -> Result<Vec<u8>, String> {
    let layout = readback.layout;
    let row = layout.width as usize * 4;
    let mut out = Vec::with_capacity(row * layout.height as usize);
    {
        let view = readback
            .buffer
            .get_mapped_range(..)
            .map_err(|error| format!("could not read the screenshot buffer: {error}"))?;
        for y in 0..layout.height as usize {
            let start = y * layout.padded as usize;
            out.extend_from_slice(&view[start..start + row]);
        }
    }
    readback.buffer.unmap();
    if layout.bgra {
        for pixel in out.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    Ok(out)
}

fn run_job(job: Job) -> Done {
    match job {
        Job::Encode {
            readback,
            timestamp,
            directory,
        } => Done::Saved(encode(&readback, &timestamp, &directory)),
        Job::Copy { path } => Done::Copied(copy_png_to_clipboard(&path)),
    }
}

/// The encoder thread's half of a screenshot: pixels out of the buffer, PNG on disk.
fn encode(readback: &Readback, timestamp: &str, directory: &Path) -> Result<PathBuf, String> {
    let pixels = read_mapped(readback)?;
    let layout = readback.layout;
    let path = super::super::save_screenshot_png(
        directory,
        timestamp,
        layout.width,
        layout.height,
        &pixels,
    )?;
    // java: the click event carries `file2.getAbsolutePath()`.
    Ok(std::path::absolute(&path).unwrap_or(path))
}

/// A `CF_DIB`: a `BITMAPINFOHEADER` then bottom-up 24-bit BGR rows padded to 4 bytes, the
/// format every Windows program that pastes images reads (image editors, chat apps, browsers).
#[must_use]
pub(crate) fn dib_from_rgba(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let row = (width as usize * 3).div_ceil(4) * 4;
    let image_size = row * height as usize;
    let mut out = Vec::with_capacity(40 + image_size);
    out.extend_from_slice(&40u32.to_le_bytes()); // biSize
    out.extend_from_slice(&(width as i32).to_le_bytes()); // biWidth
    out.extend_from_slice(&(height as i32).to_le_bytes()); // biHeight > 0: bottom-up
    out.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
    out.extend_from_slice(&24u16.to_le_bytes()); // biBitCount
    out.extend_from_slice(&0u32.to_le_bytes()); // biCompression = BI_RGB
    out.extend_from_slice(&(image_size as u32).to_le_bytes()); // biSizeImage
    out.extend_from_slice(&2835i32.to_le_bytes()); // biXPelsPerMeter (72 dpi)
    out.extend_from_slice(&2835i32.to_le_bytes()); // biYPelsPerMeter
    out.extend_from_slice(&0u32.to_le_bytes()); // biClrUsed
    out.extend_from_slice(&0u32.to_le_bytes()); // biClrImportant
    let src_row = width as usize * 4;
    for y in (0..height as usize).rev() {
        let start = out.len();
        for pixel in rgba[y * src_row..(y + 1) * src_row].chunks_exact(4) {
            out.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
        }
        out.resize(start + row, 0);
    }
    out
}

/// `[COPY]`: read the PNG back, decode it and put it on the clipboard.
fn copy_png_to_clipboard(path: &Path) -> Result<(), String> {
    let (dib, png) = clipboard_data(path)?;
    set_clipboard_image(&dib, &png)
}

/// The two clipboard formats for the PNG at `path`: its `CF_DIB` and the file's own bytes.
fn clipboard_data(path: &Path) -> Result<(Vec<u8>, Vec<u8>), String> {
    let png = fs::read(path).map_err(|error| error.to_string())?;
    let image = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
        .map_err(|error| error.to_string())?
        .into_rgba8();
    let (width, height) = image.dimensions();
    Ok((dib_from_rgba(width, height, image.as_raw()), png))
}

/// Replace the clipboard with `dib` (`CF_DIB`) and `png` (the registered `PNG` format, which
/// browsers and chat apps prefer). Serialised with the text clipboard (`app::clipboard`).
pub(crate) fn set_clipboard_image(dib: &[u8], png: &[u8]) -> Result<(), String> {
    let _guard = crate::app::clipboard::lock();
    set_clipboard_image_locked(dib, png)
}

/// [`set_clipboard_image`] for a caller already holding `app::clipboard::lock()`.
#[cfg(windows)]
fn set_clipboard_image_locked(dib: &[u8], png: &[u8]) -> Result<(), String> {
    let _clipboard = clipboard_win::Clipboard::new_attempts(10)
        .map_err(|error| format!("the clipboard is busy ({error})"))?;
    // `raw::set` empties the clipboard first, so nothing stale rides along.
    clipboard_win::raw::set(clipboard_win::formats::CF_DIB, dib)
        .map_err(|error| error.to_string())?;
    if let Some(format) = clipboard_win::register_format("PNG") {
        if let Err(error) = clipboard_win::raw::set_without_clear(format.get(), png) {
            tracing::debug!(%error, "the PNG clipboard format was refused; CF_DIB is there");
        }
    }
    Ok(())
}

/// Copying an image needs the Windows clipboard.
#[cfg(not(windows))]
fn set_clipboard_image_locked(_dib: &[u8], _png: &[u8]) -> Result<(), String> {
    Err("copying images is only supported on Windows".to_owned())
}

/// Vanilla's F2 chat line (`ScreenShotHelper.saveScreenshot`'s return value) plus `[COPY]`.
#[must_use]
pub(crate) fn screenshot_message(result: &Result<PathBuf, String>) -> ChatComponent {
    match result {
        Ok(path) => {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("screenshot.png")
                .to_owned();
            let value = path.to_string_lossy().into_owned();
            let linked = ChatComponent::text(name).with_style(ChatStyle {
                underlined: Some(true),
                click_event: Some(ClickEvent {
                    action: ClickAction::OpenFile,
                    value: value.clone(),
                }),
                ..ChatStyle::default()
            });
            let copy = ChatComponent::text(COPY_LABEL).with_style(ChatStyle {
                color: Some(TextFormat::Aqua),
                bold: Some(true),
                click_event: Some(ClickEvent {
                    action: ClickAction::CopyImage,
                    value,
                }),
                hover_event: Some(HoverEvent {
                    action: HoverAction::ShowText,
                    value: Box::new(ChatComponent::text(COPY_HOVER)),
                }),
                ..ChatStyle::default()
            });
            ChatComponent::translate("screenshot.success", vec![linked])
                .with_extra(ChatComponent::text(" "))
                .with_extra(copy)
        }
        Err(error) => ChatComponent::translate(
            "screenshot.failure",
            vec![ChatComponent::text(error.clone())],
        ),
    }
}

fn copied_message(result: &Result<(), String>) -> ChatComponent {
    match result {
        Ok(()) => ChatComponent::text(COPIED).with_style(ChatStyle {
            color: Some(TextFormat::Gray),
            ..ChatStyle::default()
        }),
        Err(error) => ChatComponent::text(format!("Couldn't copy the screenshot: {error}")),
    }
}

impl Client {
    /// The F2 half of `after_render`: start the readback of this composited frame. Everything
    /// slow happens on later frames and the encoder thread.
    pub(crate) fn x2_screenshot_after_render(&mut self, gpu: &Gpu, frame: &AcquiredFrame) {
        if !std::mem::take(&mut self.screenshot_requested) {
            return;
        }
        let timestamp = super::super::screenshot_timestamp(SystemTime::now());
        let directory = self.config.game_dir.join("screenshots");
        let started = if !gpu.config.usage.contains(wgpu::TextureUsages::COPY_SRC) {
            Err("this display surface cannot be read back".to_owned())
        } else if let Some(texture) = frame.color_texture(gpu) {
            start_readback(
                gpu,
                texture,
                gpu.config.width,
                gpu.config.height,
                gpu.config.format,
            )
        } else {
            Err("the frame has no colour target".to_owned())
        };
        match started {
            Ok(readback) => self.x2.shots.pending.push(Pending {
                readback,
                timestamp,
                directory,
            }),
            Err(error) => self.x2_chat_line(screenshot_message(&Err(error))),
        }
    }

    /// Move finished readbacks to the encoder thread and post its answers.
    /// One `bool` test when no screenshot is in progress.
    pub(crate) fn x2_frame(&mut self, gpu: &Gpu) {
        if self.x2.shots.idle() {
            return;
        }
        if !self.x2.shots.pending.is_empty() {
            // Non-blocking: lets wgpu run the map callback if the copy has finished.
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            while let Some(first) = self.x2.shots.pending.first() {
                match first.readback.state.load(Ordering::Acquire) {
                    WAITING => break,
                    MAPPED => {
                        let pending = self.x2.shots.pending.remove(0);
                        let job = Job::Encode {
                            readback: pending.readback,
                            timestamp: pending.timestamp,
                            directory: pending.directory,
                        };
                        if let Some(done) = self.x2.shots.submit(job) {
                            self.x2_done(done);
                        }
                    }
                    _ => {
                        self.x2.shots.pending.remove(0);
                        let error = "could not read the frame back".to_owned();
                        self.x2_chat_line(screenshot_message(&Err(error)));
                    }
                }
            }
        }
        self.x2_poll_results();
    }

    /// Post every answer the encoder thread has sent.
    pub(crate) fn x2_poll_results(&mut self) {
        while let Some(done) = self.x2.shots.try_done() {
            self.x2_done(done);
        }
    }

    fn x2_done(&mut self, done: Done) {
        match done {
            Done::Saved(result) => {
                if let Err(error) = &result {
                    tracing::warn!(%error, "couldn't save screenshot");
                }
                self.x2_chat_line(screenshot_message(&result));
            }
            Done::Copied(result) => self.x2_chat_line(copied_message(&result)),
        }
    }

    /// Add a client-side line to the chat log, as vanilla's `printChatMessage` does.
    pub(crate) fn x2_chat_line(&mut self, component: ChatComponent) {
        let layout = ChatLayout::from_settings(&self.settings);
        self.chat.add_message(
            component,
            0,
            self.tick as i32,
            &layout,
            &self.font,
            &self.assets.language,
            matches!(self.ui, Ui::Chat(_)),
        );
    }

    /// The canonical screenshot path behind a chat link, if it really is a PNG directly inside
    /// this game directory's `screenshots` folder. Defence in depth: only the client builds
    /// these links, but nothing else is ever opened or read.
    pub(crate) fn x2_screenshot_file(&self, path: &Path) -> Option<PathBuf> {
        let directory = self
            .config
            .game_dir
            .join("screenshots")
            .canonicalize()
            .ok()?;
        let file = path.canonicalize().ok()?;
        let png = file
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png"));
        (png && file.is_file() && file.parent() == Some(directory.as_path())).then_some(file)
    }

    /// A click on the F2 line's file name: open the screenshot with the system's viewer.
    /// java: `GuiScreen.handleComponentClick`, `OPEN_FILE` -> `openWebLink(file.toURI())`.
    pub(crate) fn x2_open_screenshot(&mut self, path: &Path) {
        match self.x2_screenshot_file(path) {
            Some(file) => {
                if let Err(error) = super::super::open_external_target(&file) {
                    tracing::warn!(%error, path = %file.display(), "could not open screenshot");
                }
            }
            None => tracing::warn!(path = %path.display(), "not opening a missing screenshot"),
        }
    }

    /// A click on `[COPY]`: the encoder thread copies the image; the chat reports the result.
    pub(crate) fn x2_copy_screenshot(&mut self, path: PathBuf) {
        let Some(file) = self.x2_screenshot_file(&path) else {
            let error = "the file is gone".to_owned();
            self.x2_chat_line(copied_message(&Err(error)));
            return;
        };
        if let Some(done) = self.x2.shots.submit(Job::Copy { path: file }) {
            self.x2_done(done);
        }
    }

    /// Shutdown: give a screenshot still being written a moment to land, so quitting right
    /// after F2 never leaves a half-written PNG.
    pub(crate) fn x2_shutdown(&mut self) {
        let Some(worker) = self.x2.shots.worker.take() else {
            return;
        };
        let deadline = Instant::now() + SHUTDOWN_WAIT;
        let mut in_flight = self.x2.shots.in_flight;
        while in_flight > 0 {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            match worker.done.recv_timeout(left) {
                Ok(_) => in_flight -= 1,
                Err(_) => break,
            }
        }
        self.x2.shots.in_flight = 0;
        drop(worker.jobs);
        if in_flight == 0 {
            if let Some(thread) = worker.thread {
                let _ = thread.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(width: u32, height: u32) -> Option<Gpu> {
        match Gpu::headless(
            width,
            height,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => Some(gpu),
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter available on this machine");
                None
            }
            Err(error) => panic!("headless context creation failed: {error}"),
        }
    }

    /// A noisy, poorly compressible test image (a real scene's textures are noisy too).
    fn noise(width: u32, height: u32) -> Vec<u8> {
        let mut state = 0x2545_F491_u32;
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let grain = (state & 0x1F) as u8;
                out.extend_from_slice(&[
                    (x * 255 / width) as u8 ^ grain,
                    (y * 255 / height) as u8 ^ grain,
                    ((x + y) & 0xFF) as u8,
                    0x80,
                ]);
            }
        }
        out
    }

    fn noise_texture(gpu: &Gpu, width: u32, height: u32, pixels: &[u8]) -> wgpu::Texture {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("x2 test frame"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        texture
    }

    /// The pre-wave F2 path, verbatim in shape: copy, submit, block on `poll(wait)`, de-pad,
    /// then encode on the calling thread. Only for the before/after timing.
    fn blocking_readback(gpu: &Gpu, texture: &wgpu::Texture, width: u32, height: u32) -> Vec<u8> {
        let readback = start_readback(gpu, texture, width, height, wgpu::TextureFormat::Rgba8Unorm)
            .expect("readback");
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        assert_eq!(readback.state.load(Ordering::Acquire), MAPPED);
        read_mapped(&readback).expect("pixels")
    }

    fn wait_mapped(gpu: &Gpu, readback: &Readback) -> u32 {
        let started = Instant::now();
        let mut polls = 0;
        while readback.state.load(Ordering::Acquire) == WAITING {
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            polls += 1;
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "map never completed"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        polls
    }

    /// The asynchronous readback produces exactly the frame's pixels and a PNG that decodes to
    /// them (RGB, like vanilla's `TYPE_INT_RGB`), and the F2 frame's own cost is only the copy
    /// submission. Prints the before/after game-thread cost (provisional).
    #[test]
    fn the_async_screenshot_matches_the_frame_and_does_not_block() {
        for (width, height) in [(1920u32, 1080u32), (2560, 1440)] {
            let Some(gpu) = gpu(64, 64) else { return };
            let pixels = noise(width, height);
            let texture = noise_texture(&gpu, width, height, &pixels);
            let dir = tempfile::tempdir().expect("tempdir");

            // Before: everything on the calling thread.
            let before = Instant::now();
            let blocking = blocking_readback(&gpu, &texture, width, height);
            let path = super::super::super::save_screenshot_png(
                dir.path(),
                "2026-11-09_02.00.00",
                width,
                height,
                &blocking,
            )
            .expect("save");
            let before_ms = before.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(blocking, pixels, "the blocking readback is the frame");
            assert_eq!(
                path.file_name().and_then(|n| n.to_str()),
                Some("2026-11-09_02.00.00.png")
            );

            // After: the F2 frame only records and submits.
            let f2 = Instant::now();
            let readback = start_readback(
                &gpu,
                &texture,
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .expect("readback");
            let f2_ms = f2.elapsed().as_secs_f64() * 1000.0;
            let polls = wait_mapped(&gpu, &readback);
            let poll = Instant::now();
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            let poll_us = poll.elapsed().as_secs_f64() * 1e6;
            let worker = Instant::now();
            let saved = encode(&readback, "2026-11-09_02.00.00", dir.path()).expect("encode");
            let worker_ms = worker.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(
                saved.file_name().and_then(|n| n.to_str()),
                Some("2026-11-09_02.00.00_2.png"),
                "vanilla's collision suffix starts at _2"
            );
            let decoded = image::open(&saved).expect("decode");
            assert_eq!(
                decoded.color(),
                image::ColorType::Rgb8,
                "vanilla writes RGB"
            );
            let rgb = decoded.into_rgb8();
            assert_eq!(rgb.dimensions(), (width, height));
            for (got, want) in rgb.as_raw().chunks_exact(3).zip(pixels.chunks_exact(4)) {
                assert_eq!(got, &want[..3]);
            }
            eprintln!(
                "{width}x{height} (provisional): before = {before_ms:.1} ms on the game thread; \
                 after = {f2_ms:.2} ms on the F2 frame + {poll_us:.0} us per polling frame \
                 ({polls} polls until mapped), {worker_ms:.1} ms on the encoder thread"
            );
        }
    }

    #[test]
    fn a_dib_is_bottom_up_bgr_with_padded_rows() {
        // 3x2: row 0 red, green, blue; row 1 white, black, grey.
        let rgba = [
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, //
            255, 255, 255, 255, 0, 0, 0, 255, 128, 128, 128, 255,
        ];
        let dib = dib_from_rgba(3, 2, &rgba);
        let u32_at = |i: usize| u32::from_le_bytes(dib[i..i + 4].try_into().unwrap());
        assert_eq!(u32_at(0), 40, "BITMAPINFOHEADER");
        assert_eq!(u32_at(4), 3);
        assert_eq!(u32_at(8), 2, "positive height = bottom-up");
        assert_eq!(u16::from_le_bytes([dib[14], dib[15]]), 24);
        assert_eq!(u32_at(16), 0, "BI_RGB");
        assert_eq!(u32_at(20), 2 * 12, "rows of 9 bytes padded to 12");
        assert_eq!(dib.len(), 40 + 24);
        // The first stored row is the image's last.
        assert_eq!(
            &dib[40..52],
            &[255, 255, 255, 0, 0, 0, 128, 128, 128, 0, 0, 0]
        );
        assert_eq!(&dib[52..64], &[0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn the_chat_line_is_vanillas_with_a_copy_link() {
        let path = PathBuf::from("C:/game/screenshots/2026-11-09_02.00.00.png");
        let message = screenshot_message(&Ok(path.clone()));
        let mut lang = mc_assets::Language::new("en_US");
        lang.insert("screenshot.success", "Saved screenshot as %s");
        assert_eq!(
            message.unformatted_text(&lang),
            "Saved screenshot as 2026-11-09_02.00.00.png [COPY]"
        );
        let spans = message.spans(&lang);
        let name = spans
            .iter()
            .find(|span| span.text == "2026-11-09_02.00.00.png")
            .expect("the file name");
        assert!(name.style.underlined);
        assert_eq!(
            name.style.click_event,
            Some(ClickEvent {
                action: ClickAction::OpenFile,
                value: path.to_string_lossy().into_owned(),
            })
        );
        let copy = spans
            .iter()
            .find(|span| span.text == COPY_LABEL)
            .expect("[COPY]");
        assert_eq!(
            copy.style.click_event.as_ref().map(|click| click.action),
            Some(ClickAction::CopyImage)
        );
        assert!(copy.style.hover_event.is_some());
        assert!(copy.style.bold);
    }

    /// The clipboard really holds the image afterwards (Windows): the PNG is decoded into a
    /// `CF_DIB` and the file's bytes ride along as `PNG`. The whole sequence holds the process's
    /// clipboard lock, so the other clipboard tests in this binary cannot interleave, and the
    /// previous text is put back: the test leaves the clipboard as it found it.
    #[cfg(windows)]
    #[test]
    fn copy_puts_a_dib_and_a_png_on_the_windows_clipboard() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pixels = noise(7, 5);
        let path = super::super::super::save_screenshot_png(
            dir.path(),
            "2026-11-09_02.00.00",
            7,
            5,
            &pixels,
        )
        .expect("save");
        let (dib_in, png_in) = clipboard_data(&path).expect("decode");
        let (result, dib, png) = {
            let _guard = crate::app::clipboard::lock();
            let previous = clipboard_win::get_clipboard_string().ok();
            let result = set_clipboard_image_locked(&dib_in, &png_in);
            let read = || -> Option<(Vec<u8>, Vec<u8>)> {
                let _clipboard = clipboard_win::Clipboard::new_attempts(10).ok()?;
                let get = |format: u32| {
                    let size = clipboard_win::size(format)?.get();
                    let mut out = vec![0; size];
                    let n = clipboard_win::raw::get(format, &mut out).ok()?;
                    out.truncate(n);
                    Some(out)
                };
                let png_format = clipboard_win::register_format("PNG")?.get();
                Some((get(clipboard_win::formats::CF_DIB)?, get(png_format)?))
            };
            let (dib, png) = read().unzip();
            if let Some(text) = &previous {
                let _ = clipboard_win::set_clipboard_string(text);
            }
            (result, dib, png)
        };
        if let Err(error) = result {
            eprintln!("SKIP: the clipboard refused the image ({error})");
            return;
        }
        // `GlobalSize` may round the block up, so compare what was written, not the tail.
        let dib = dib.expect("a CF_DIB on the clipboard");
        let rgb: Vec<u8> = pixels
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect();
        let want = dib_from_rgba(7, 5, &rgb);
        assert!(dib.len() >= want.len(), "{} < {}", dib.len(), want.len());
        assert_eq!(&dib[..40], &want[..40], "the header");
        assert_eq!(&dib[..want.len()], &want[..], "the pixels");
        let png = png.expect("the PNG format");
        let file = fs::read(&path).expect("file");
        assert!(png.len() >= file.len());
        assert_eq!(&png[..file.len()], &file[..]);
    }
}
