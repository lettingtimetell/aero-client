//! #14 step 1b: the frame's `finish()` and `submit` on the present thread (opt-in,
//! `MC_PIPELINED_SUBMIT=1`, see `shell.rs`).
//!
//! wgpu 30 records a frame's passes as commands and encodes them into the backend's command
//! buffer in `CommandEncoder::finish`; `Queue::submit` then hands that to the driver. On the
//! SkyWars scene the two are ~0.17 ms of a ~0.6 ms frame, and the present after them another
//! ~0.11 ms, all on the main thread. The pipelined path records the frame on the main thread
//! exactly as before and leaves it *pending* here; the shell hands it, with the swap-chain image,
//! to the present thread, which finishes, submits and presents it while the main thread runs the
//! next iteration's ticks and frame build.
//!
//! # Ordering: the same as the serial loop, by construction
//!
//! `Queue::write_*` is applied at the next submission, before that submission's command buffers.
//! In the serial loop a write made while building frame N+1 therefore lands after frame N's
//! commands. With frame N's submit on another thread, a main-thread write could reach the queue
//! before that submit and change what frame N reads. So every main-thread use of the queue goes
//! through [`GatedQueue`], which first *settles* the pipeline:
//!
//! 1. a frame recorded but not yet handed off (possible only between `render_deferred` and the
//!    hand-off, e.g. an F2 screenshot readback in `after_render`) is finished and submitted right
//!    there, on the calling thread;
//! 2. it then waits until every handed-off frame has been submitted.
//!
//! Staged uploads (the [`crate::upload`] ring) need no gate: frame N's staged copies are taken
//! when it is deferred and travel with it, and frame N+1's go into its own upload command
//! buffer. The ring's chunks are re-mapped by the submission that used them
//! (`StagingBelt::finish_and_recall_on_submit`), whichever thread makes it, so no thread can map
//! a chunk another frame has closed but not submitted yet. Nothing ever waits for a present: the
//! present thread signals as soon as its `submit` returns, before it presents.
//!
//! The wait has no time limit: a submit that never returns is a wedged driver, which the serial
//! loop would hang in too, and giving up would break the order for every later frame. The wait
//! ends early only if the present thread exits ([`FramePipe::abandon`], from a guard on that
//! thread, so a panic there cannot leave the main thread waiting).
//!
//! When the pipelined path is off, nothing is ever deferred or handed off and settling is three
//! atomic loads per queue call.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// A frame recorded on the main thread whose `finish()` and `submit` have not run yet.
pub struct PendingFrame {
    /// The frame's staged uploads, already closed: they belong in the same submission, before
    /// the frame's own commands.
    staged: Option<wgpu::CommandBuffer>,
    encoder: wgpu::CommandEncoder,
}

impl std::fmt::Debug for PendingFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingFrame")
            .field("staged", &self.staged.is_some())
            .finish_non_exhaustive()
    }
}

/// How often a main-thread queue call still waiting for the present thread's `submit` logs it. A
/// submit takes well under a millisecond, so one log line means a stalled driver.
const SUBMIT_WAIT_REPORT: Duration = Duration::from_secs(2);

/// Shared state between the main thread's [`GatedQueue`] and the present thread.
pub struct FramePipe {
    pending: Mutex<Option<PendingFrame>>,
    /// Fast path for [`Self::settle`]: whether `pending` may hold a frame.
    has_pending: AtomicBool,
    /// Frames handed to the present thread so far.
    handed: AtomicU64,
    /// Frames the present thread has submitted so far. Never ahead of `handed`.
    submitted: AtomicU64,
    wake_lock: Mutex<()>,
    wake: Condvar,
    /// The present thread is gone: stop waiting for it.
    dead: AtomicBool,
}

impl std::fmt::Debug for FramePipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FramePipe")
            .field("handed", &self.handed.load(Ordering::Relaxed))
            .field("submitted", &self.submitted.load(Ordering::Relaxed))
            .field("dead", &self.dead.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl FramePipe {
    pub(crate) fn new() -> Self {
        Self {
            pending: Mutex::new(None),
            has_pending: AtomicBool::new(false),
            handed: AtomicU64::new(0),
            submitted: AtomicU64::new(0),
            wake_lock: Mutex::new(()),
            wake: Condvar::new(),
            dead: AtomicBool::new(false),
        }
    }

    fn lock_pending(&self) -> MutexGuard<'_, Option<PendingFrame>> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Leave `encoder` (and the staged uploads closed with it) pending instead of submitting.
    /// Anything older — handed-off frames, and a frame still pending, which the shell never
    /// leaves but another caller of `render_deferred` might — is settled first, so frames never
    /// reorder.
    pub(crate) fn defer(
        &self,
        queue: &wgpu::Queue,
        staged: Option<wgpu::CommandBuffer>,
        encoder: wgpu::CommandEncoder,
    ) {
        self.settle(queue);
        *self.lock_pending() = Some(PendingFrame { staged, encoder });
        self.has_pending.store(true, Ordering::Release);
    }

    /// [`Self::defer`] for a frame the shell hands to the present thread at
    /// once, behind the frames handed before it. The present thread
    /// submits in hand-off order, so this frame is ordered after them without waiting here for
    /// their submits; only a frame still pending on this thread (which the shell never leaves) is
    /// settled first, exactly as `defer` would.
    pub(crate) fn defer_behind_handed(
        &self,
        queue: &wgpu::Queue,
        staged: Option<wgpu::CommandBuffer>,
        encoder: wgpu::CommandEncoder,
    ) {
        if self.has_pending.load(Ordering::Acquire) {
            self.settle(queue);
        }
        *self.lock_pending() = Some(PendingFrame { staged, encoder });
        self.has_pending.store(true, Ordering::Release);
    }

    /// Count a hand-off that carries no pending frame (a readback on this
    /// thread already submitted it) but still makes the present thread submit work of its own,
    /// the copy into the image: main-thread queue calls then wait for that submit as for any
    /// handed-off frame, so nothing recorded later reaches the GPU before the copy.
    pub(crate) fn count_bare_handoff(&self) {
        self.handed.fetch_add(1, Ordering::AcqRel);
    }

    /// The present thread's side of a hand-off: record `append` (the copy into
    /// the swap-chain image) at the end of the frame's encoder, then finish and submit it with
    /// its staged uploads, as [`Self::submit_handed`] does.
    pub(crate) fn submit_handed_with(
        &self,
        queue: &wgpu::Queue,
        frame: PendingFrame,
        append: impl FnOnce(&mut wgpu::CommandEncoder),
    ) {
        let PendingFrame {
            staged,
            mut encoder,
        } = frame;
        append(&mut encoder);
        Self::submit_frame(queue, PendingFrame { staged, encoder });
        self.mark_submitted();
    }

    /// The present thread's side of a bare hand-off
    /// ([`Self::count_bare_handoff`]): submit `commands` (the copy alone, or nothing when there is
    /// no image to copy into), then wake any waiting queue call.
    pub(crate) fn submit_bare_handoff(
        &self,
        queue: &wgpu::Queue,
        commands: Option<wgpu::CommandBuffer>,
    ) {
        if let Some(commands) = commands {
            mc_core::profile_scope!("gpu.submit_call");
            queue.submit(Some(commands));
        }
        self.mark_submitted();
    }

    /// Take the pending frame to hand to the present thread, counting it as handed off.
    pub(crate) fn take_for_handoff(&self) -> Option<PendingFrame> {
        if !self.has_pending.load(Ordering::Acquire) {
            return None;
        }
        let frame = self.lock_pending().take();
        self.has_pending.store(false, Ordering::Release);
        if frame.is_some() {
            self.handed.fetch_add(1, Ordering::AcqRel);
        }
        frame
    }

    /// `finish()` and submit `frame`: its staged uploads, then its commands, in one submission —
    /// exactly what [`crate::Gpu::submit_frame`] does on the serial path.
    fn submit_frame(queue: &wgpu::Queue, frame: PendingFrame) {
        let PendingFrame { staged, encoder } = frame;
        let commands = {
            mc_core::profile_scope!("gpu.encoder_finish");
            encoder.finish()
        };
        mc_core::profile_scope!("gpu.submit_call");
        queue.submit(staged.into_iter().chain(Some(commands)));
    }

    /// The present thread's side: submit a handed-off frame, then wake any waiting queue call
    /// before anything else (the present comes after).
    pub(crate) fn submit_handed(&self, queue: &wgpu::Queue, frame: PendingFrame) {
        Self::submit_frame(queue, frame);
        self.mark_submitted();
    }

    fn mark_submitted(&self) {
        self.submitted.fetch_add(1, Ordering::AcqRel);
        // Taking the lock orders this wake after a waiter's check-then-wait.
        let _guard = self
            .wake_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.wake.notify_all();
    }

    /// The present thread is gone, and a frame handed to it may never be submitted: stop
    /// waiting for the pipeline from now on. Called by a guard when that thread exits, normally
    /// or by panic, and by the shell when it finds the thread dead.
    pub(crate) fn abandon(&self) {
        self.dead.store(true, Ordering::Release);
        let _guard = self
            .wake_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.wake.notify_all();
    }

    /// Make every queue operation so far visible in serial order before the caller's own: wait
    /// for every handed-off frame's submit, then submit a frame left pending on this thread
    /// (always newer than any handed-off one).
    pub(crate) fn settle(&self, queue: &wgpu::Queue) {
        self.wait_for_handed();
        if self.has_pending.load(Ordering::Acquire) {
            let frame = self.lock_pending().take();
            self.has_pending.store(false, Ordering::Release);
            if let Some(frame) = frame {
                Self::submit_frame(queue, frame);
            }
        }
    }

    /// Block until the present thread has submitted every frame handed to it, or has exited.
    fn wait_for_handed(&self) {
        let handed = self.handed.load(Ordering::Acquire);
        if self.submitted.load(Ordering::Acquire) >= handed || self.dead.load(Ordering::Acquire) {
            return;
        }
        mc_core::profile_scope!("gpu.pipeline_settle_wait");
        let mut guard = self
            .wake_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        while self.submitted.load(Ordering::Acquire) < handed && !self.dead.load(Ordering::Acquire)
        {
            let (next, timeout) = self
                .wake
                .wait_timeout(guard, SUBMIT_WAIT_REPORT)
                .unwrap_or_else(PoisonError::into_inner);
            guard = next;
            if timeout.timed_out() {
                tracing::warn!(
                    handed,
                    submitted = self.submitted.load(Ordering::Acquire),
                    "pipelined submit: still waiting for the present thread's submit"
                );
            }
        }
    }

    /// Frames handed off and frames submitted, for tests and diagnostics.
    #[must_use]
    pub fn counts(&self) -> (u64, u64) {
        (
            self.handed.load(Ordering::Acquire),
            self.submitted.load(Ordering::Acquire),
        )
    }
}

/// The device queue as the main thread sees it: every call first settles the frame pipeline
/// (see the module docs), then forwards to `wgpu::Queue`.
///
/// `Deref` settles too, so passing `&gpu.queue` to an API that takes `&wgpu::Queue` (texture
/// creation with initial data, an inline present) is ordered like everything else.
pub struct GatedQueue {
    raw: wgpu::Queue,
    pipe: Arc<FramePipe>,
}

impl std::fmt::Debug for GatedQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatedQueue")
            .field("pipe", &self.pipe)
            .finish_non_exhaustive()
    }
}

/// A clone shares the gate. (Without this impl, `gpu.queue.clone()` would auto-deref to
/// `wgpu::Queue::clone` and hand out an ungated queue.)
impl Clone for GatedQueue {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            pipe: Arc::clone(&self.pipe),
        }
    }
}

impl GatedQueue {
    pub(crate) fn new(raw: wgpu::Queue, pipe: Arc<FramePipe>) -> Self {
        Self { raw, pipe }
    }

    fn settle(&self) {
        self.pipe.settle(&self.raw);
    }

    /// `Queue::write_buffer`, after settling.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: wgpu::BufferAddress, data: &[u8]) {
        self.settle();
        self.raw.write_buffer(buffer, offset, data);
    }

    /// `Queue::write_texture`, after settling.
    pub fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        self.settle();
        self.raw.write_texture(texture, data, data_layout, size);
    }

    /// `Queue::submit`, after settling.
    pub fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(
        &self,
        command_buffers: I,
    ) -> wgpu::SubmissionIndex {
        self.settle();
        self.raw.submit(command_buffers)
    }

    /// The queue without the gate, for the present thread (which is the other side of it).
    pub(crate) fn ungated(&self) -> &wgpu::Queue {
        &self.raw
    }

    pub(crate) fn pipe(&self) -> &Arc<FramePipe> {
        &self.pipe
    }
}

impl std::ops::Deref for GatedQueue {
    type Target = wgpu::Queue;

    fn deref(&self) -> &wgpu::Queue {
        self.settle();
        &self.raw
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crate::Gpu;

    fn gpu() -> Gpu {
        Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter")
    }

    fn buffer(gpu: &Gpu, label: &str) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// A "frame" whose only command copies `source` into `seen`: `seen` records what it read.
    fn copying_frame(
        gpu: &Gpu,
        source: &wgpu::Buffer,
        seen: &wgpu::Buffer,
    ) -> wgpu::CommandEncoder {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, seen, 0, 16);
        encoder
    }

    /// Read `buffer` back through the serial frame path (staged uploads first), as a frame would.
    fn read(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<u8> {
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame pipe readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, 16);
        let _ = gpu.submit_frame(encoder.finish());
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).expect("map result receiver");
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("readback poll");
        rx.recv().expect("map callback").expect("readback map");
        let bytes = readback
            .get_mapped_range(..)
            .expect("readback view")
            .to_vec();
        readback.unmap();
        bytes
    }

    /// A main-thread queue write made after a frame was deferred lands after that frame's
    /// commands, exactly as in the serial loop: the gate submits the pending frame first.
    #[test]
    fn a_write_after_a_deferred_frame_lands_after_its_commands() {
        let gpu = gpu();
        let source = buffer(&gpu, "source");
        let seen = buffer(&gpu, "seen by the deferred frame");
        gpu.queue.write_buffer(&source, 0, &[1; 16]);
        gpu.defer_frame(copying_frame(&gpu, &source, &seen));
        gpu.queue.write_buffer(&source, 0, &[2; 16]);
        assert_eq!(
            read(&gpu, &seen),
            vec![1; 16],
            "the frame read the earlier write"
        );
        assert_eq!(read(&gpu, &source), vec![2; 16]);
        assert_eq!(gpu.frame_pipe().counts(), (0, 0), "never handed off");
    }

    /// The present thread's side. A frame handed off and submitted late on another thread still
    /// precedes a main-thread queue call made meanwhile, which waits for that submit; the frame
    /// carries its own staged uploads, and the next frame's staging neither waits nor leaks in.
    #[test]
    fn main_thread_queue_calls_wait_for_a_handed_off_frame() {
        let gpu = gpu();
        let source = buffer(&gpu, "source");
        let seen = buffer(&gpu, "seen by the handed-off frame");
        let other = buffer(&gpu, "written while the frame is in flight");
        gpu.queue.write_buffer(&source, 0, &[1; 16]);
        gpu.write_buffer_staged(&source, 0, &[3; 16]);
        gpu.defer_frame(copying_frame(&gpu, &source, &seen));
        let frame = gpu.take_pending_frame().expect("the deferred frame");
        // The next frame's staging: recorded into its own upload buffer, so no wait.
        gpu.write_buffer_staged(&source, 0, &[4; 16]);

        let pipe = gpu.frame_pipe();
        let queue = gpu.queue.ungated().clone();
        // The clock starts before the submitter does, so a slow spawn cannot shorten the wait
        // this measures below the sleeper's 60 ms.
        let started = Instant::now();
        let submitter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(60));
            pipe.submit_handed(&queue, frame);
        });
        gpu.queue.write_buffer(&other, 0, &[9; 16]);
        let waited = started.elapsed();
        submitter.join().expect("submitter thread");
        assert!(
            waited >= Duration::from_millis(40),
            "the write waited for the handed-off submit ({waited:?})"
        );
        assert_eq!(gpu.frame_pipe().counts(), (1, 1));
        assert_eq!(
            read(&gpu, &seen),
            vec![3; 16],
            "its own staged write, not the next one"
        );
        assert_eq!(read(&gpu, &source), vec![4; 16]);
        assert_eq!(read(&gpu, &other), vec![9; 16]);
    }

    /// A handed-off frame that will never be submitted (the present thread died) must not make
    /// later queue calls wait.
    #[test]
    fn an_abandoned_pipe_never_blocks() {
        let gpu = gpu();
        let source = buffer(&gpu, "source");
        let seen = buffer(&gpu, "seen");
        gpu.defer_frame(copying_frame(&gpu, &source, &seen));
        drop(gpu.take_pending_frame().expect("the deferred frame"));
        gpu.frame_pipe().abandon();
        let started = Instant::now();
        gpu.queue.write_buffer(&source, 0, &[5; 16]);
        assert!(started.elapsed() < Duration::from_millis(500));
        assert_eq!(read(&gpu, &source), vec![5; 16]);
    }

    /// A buffer filled with `value`, through a plain (gated) queue write.
    fn filled(gpu: &Gpu, label: &str, value: u8) -> wgpu::Buffer {
        let filled = buffer(gpu, label);
        gpu.queue.write_buffer(&filled, 0, &[value; 16]);
        filled
    }

    /// A hand-off's frame with the copy into the image appended on the present
    /// thread lands exactly like the serial path's frame with the copy recorded on this thread:
    /// its staged upload first, then its commands, then the copy.
    #[test]
    fn w19_o1_a_handed_frame_with_the_copy_appended_equals_the_serial_frame() {
        let run = |handed: bool| -> Vec<u8> {
            let gpu = gpu();
            let source = filled(&gpu, "source", 1);
            let recorded = buffer(&gpu, "frame as recorded");
            let image = buffer(&gpu, "image");
            // This frame's staged upload, which its own commands read.
            gpu.write_buffer_staged(&source, 0, &[7; 16]);
            let frame = copying_frame(&gpu, &source, &recorded);
            if handed {
                gpu.defer_frame_for_handoff(frame);
                let pending = gpu.take_pending_frame().expect("the deferred frame");
                let pipe = gpu.frame_pipe();
                let queue = gpu.queue.ungated().clone();
                let (from, to) = (recorded.clone(), image.clone());
                std::thread::spawn(move || {
                    pipe.submit_handed_with(&queue, pending, |enc| {
                        enc.copy_buffer_to_buffer(&from, 0, &to, 0, 16);
                    });
                })
                .join()
                .expect("present thread stand-in");
                assert_eq!(gpu.frame_pipe().counts(), (1, 1));
            } else {
                let mut frame = frame;
                frame.copy_buffer_to_buffer(&recorded, 0, &image, 0, 16);
                let _ = gpu.submit_frame(frame.finish());
            }
            read(&gpu, &image)
        };
        let serial = run(false);
        assert_eq!(serial, vec![7; 16], "the staged upload reached the image");
        assert_eq!(run(true), serial);
    }

    /// `defer_frame_for_handoff` does not wait for a handed-off frame's submit
    /// (`defer_frame` does), and the present thread's in-order submits still put the later frame
    /// after the earlier one.
    #[test]
    fn w19_o1_a_frame_deferred_for_handoff_does_not_wait_and_stays_in_order() {
        let gpu = gpu();
        let first = filled(&gpu, "written by the first frame's source", 1);
        let recorded = buffer(&gpu, "frame as recorded");
        let image = buffer(&gpu, "image");
        // Frame 1 writes the recorded buffer; frame 2 copies it on (reads frame 1's result).
        gpu.defer_frame_for_handoff(copying_frame(&gpu, &first, &recorded));
        let one = gpu.take_pending_frame().expect("frame 1");
        let (tx, rx) = std::sync::mpsc::channel::<crate::frame_pipe::PendingFrame>();
        let pipe = gpu.frame_pipe();
        let queue = gpu.queue.ungated().clone();
        // The present thread stand-in: frames in hand-off order, the first one held back.
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(250));
            while let Ok(frame) = rx.recv() {
                pipe.submit_handed(&queue, frame);
            }
        });
        tx.send(one).expect("hand frame 1 off");
        let started = Instant::now();
        gpu.defer_frame_for_handoff(copying_frame(&gpu, &recorded, &image));
        let deferred_in = started.elapsed();
        let two = gpu.take_pending_frame().expect("frame 2");
        tx.send(two).expect("hand frame 2 off");
        drop(tx);
        thread.join().expect("present thread stand-in");
        assert!(
            deferred_in < Duration::from_millis(150),
            "deferring frame 2 waited for frame 1's submit ({deferred_in:?})"
        );
        assert_eq!(gpu.frame_pipe().counts(), (2, 2));
        assert_eq!(read(&gpu, &image), vec![1; 16], "frame 2 ran after frame 1");

        // Contrast: the serial-loop defer waits for the handed-off frame's submit.
        let serial = super::tests::gpu();
        let source = filled(&serial, "source", 3);
        let seen = buffer(&serial, "seen");
        serial.defer_frame(copying_frame(&serial, &source, &seen));
        let one = serial.take_pending_frame().expect("frame 1");
        let pipe = serial.frame_pipe();
        let queue = serial.queue.ungated().clone();
        let started = Instant::now();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(250));
            pipe.submit_handed(&queue, one);
        });
        serial.defer_frame(copying_frame(&serial, &seen, &source));
        let waited = started.elapsed();
        thread.join().expect("submitter");
        serial.flush_staged_writes();
        assert!(
            waited >= Duration::from_millis(200),
            "`defer_frame` waits for the handed-off submit ({waited:?})"
        );
    }

    /// When a readback on the main thread already submitted a frame, its
    /// hand-off is bare: the present thread submits only the copy into the image. Counting that
    /// hand-off keeps every later main-thread queue call, including the next frame's submit, after
    /// the copy, so the image gets this frame and not the next one. Without the count (the
    /// contrast run) the next frame overtakes the late copy.
    #[test]
    fn w19_o1_a_bare_handoff_keeps_the_next_frame_after_its_copy() {
        let run = |counted: bool| -> Vec<u8> {
            let gpu = gpu();
            let this = filled(&gpu, "this frame's content", 1);
            let next = filled(&gpu, "the next frame's content", 2);
            let recorded = buffer(&gpu, "frame as recorded");
            let image = buffer(&gpu, "image");
            gpu.defer_frame_for_handoff(copying_frame(&gpu, &this, &recorded));
            // A readback: any queue call submits the pending frame first.
            gpu.flush_staged_writes();
            assert!(
                gpu.take_pending_frame().is_none(),
                "the readback submitted it"
            );
            let pipe = gpu.frame_pipe();
            if counted {
                pipe.count_bare_handoff();
            }
            let queue = gpu.queue.ungated().clone();
            let device = gpu.device.clone();
            let (from, to) = (recorded.clone(), image.clone());
            // Counted, the main thread's submit below must wait for the copy, so the copy comes
            // late on its own (250 ms). Uncounted, the copy waits for the main thread's signal,
            // which makes the overtaking deterministic.
            let (go, wait) = std::sync::mpsc::channel::<()>();
            let thread = std::thread::spawn(move || {
                if counted {
                    std::thread::sleep(Duration::from_millis(250));
                } else {
                    wait.recv().expect("the main thread's submit");
                }
                let mut enc =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                enc.copy_buffer_to_buffer(&from, 0, &to, 0, 16);
                if counted {
                    pipe.submit_bare_handoff(&queue, Some(enc.finish()));
                } else {
                    queue.submit(Some(enc.finish()));
                }
            });
            // The next frame writes the recorded buffer again; a readback submits it at once.
            gpu.defer_frame_for_handoff(copying_frame(&gpu, &next, &recorded));
            gpu.flush_staged_writes();
            let _ = go.send(());
            thread.join().expect("present thread stand-in");
            if counted {
                assert_eq!(gpu.frame_pipe().counts(), (1, 1));
            }
            read(&gpu, &image)
        };
        assert_eq!(run(true), vec![1; 16], "the image holds this frame");
        assert_eq!(
            run(false),
            vec![2; 16],
            "contrast: uncounted, the next frame overtakes the copy"
        );
    }

    /// A frame over the recorded target reads back the target's bytes.
    #[test]
    fn w19_o1_a_recorded_target_frame_reads_back_the_target() {
        let gpu = Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame as recorded"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        drop(enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fill"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        gpu.defer_frame_for_handoff(enc);
        let frame = crate::AcquiredFrame::from_recorded_target(texture.clone(), view);
        assert_eq!(
            frame.color_texture(&gpu).map(wgpu::Texture::usage),
            Some(texture.usage()),
            "the frame's colour texture is the recorded target"
        );
        // The headless context's own target is never drawn here: red can only come from the
        // recorded target.
        let pixels = frame.read_pixels(&gpu).expect("read back");
        assert_eq!(pixels.len(), 4 * 4 * 4);
        assert!(
            pixels.chunks(4).all(|p| p == [255, 0, 0, 255]),
            "the readback submitted the pending frame first: {:?}",
            &pixels[..4]
        );
        assert!(
            gpu.take_pending_frame().is_none(),
            "submitted by the readback"
        );
    }

    /// Phase review finding 1: the ring's chunks are re-mapped by the submission that used them.
    /// Here the next frame closes its staging (a skipped frame's flush) while the previous one is
    /// still in flight on another thread. With an explicit `StagingBelt::recall` after that
    /// frame's submit, the recall mapped the next frame's closed chunk before its own submit,
    /// and that submit was a validation error.
    #[test]
    fn a_frame_in_flight_does_not_remap_the_next_frames_staging() {
        let gpu = gpu();
        let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let source = buffer(&gpu, "source");
        let seen = buffer(&gpu, "seen by the frame in flight");
        gpu.write_buffer_staged(&source, 0, &[3; 16]);
        gpu.defer_frame(copying_frame(&gpu, &source, &seen));
        let frame = gpu.take_pending_frame().expect("the deferred frame");
        // The next frame's staging, closed by the flush below while the first is in flight.
        gpu.write_buffer_staged(&source, 0, &[4; 16]);

        let pipe = gpu.frame_pipe();
        let queue = gpu.queue.ungated().clone();
        let submitter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            pipe.submit_handed(&queue, frame);
        });
        gpu.flush_staged_writes();
        submitter.join().expect("submitter thread");
        assert_eq!(read(&gpu, &seen), vec![3; 16]);
        assert_eq!(read(&gpu, &source), vec![4; 16]);
        assert!(
            pollster::block_on(errors.pop()).is_none(),
            "unexpected wgpu validation error"
        );
    }
}
