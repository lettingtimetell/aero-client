//! R1-8 / GPURES-2: the per-frame upload ring.
//!
//! `Queue::write_buffer` creates one staging buffer per call and destroys it after the next
//! submission completes (wgpu 30 documents this on the method), and a frame makes dozens of
//! those calls. [`UploadRing`] sub-allocates every staged write of a frame from a recycled
//! `wgpu::util::StagingBelt` and records the copies into one upload encoder, which
//! [`Gpu::submit_frame`](crate::Gpu::submit_frame) submits in the same `Queue::submit` as the
//! frame encoder, immediately before it. In steady state a frame creates no staging buffers.
//!
//! # Ordering contract (read before migrating a stream)
//!
//! * Staged copies execute after every `Queue::write_*` of the frame and before every command
//!   of the frame encoder: the same position relative to the frame's draws that
//!   `write_buffer` has. A phase may therefore stage from `record`, before or after it opens its
//!   pass, exactly where it called `write_buffer` (the dome's write-after-pass is fine).
//! * Staged copies run in call order, so several staged writes to one buffer resolve to the last.
//! * A buffer must not be written by both paths in one frame: a `write_buffer` always lands
//!   before every staged copy, whatever the call order.
//! * A staged buffer must only be read by the frame encoder. A path that submits its own
//!   command buffer (`PanoramaBlur::read_rgba`, the screenshot readback, the atlas flush,
//!   standalone harnesses) must call
//!   [`Gpu::flush_staged_uploads`](crate::Gpu::flush_staged_uploads) first if it reads one.
//!
//! `MC_UPLOAD_RING=0` restores plain `write_buffer` for every staged call (A/B and rollback).

/// Environment switch: `0` routes [`Gpu::write_buffer_staged`](crate::Gpu::write_buffer_staged)
/// straight to `Queue::write_buffer`.
pub const UPLOAD_RING_ENV: &str = "MC_UPLOAD_RING";

/// Whether the ring is enabled for a new [`Gpu`](crate::Gpu): on unless the env var is `0`.
#[must_use]
pub fn upload_ring_enabled_from_env() -> bool {
    std::env::var(UPLOAD_RING_ENV).map_or(true, |value| value.trim() != "0")
}

/// Counters since the last [`UploadRing::take_stats`], for frame telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadRingStats {
    /// Staged writes (one `copy_buffer_to_buffer` each).
    pub writes: u64,
    /// Bytes staged.
    pub bytes: u64,
    /// Upload command buffers handed to a submission.
    pub submissions: u64,
}

/// One frame's staged copies plus the recycled staging memory behind them.
pub struct UploadRing {
    belt: Option<wgpu::util::StagingBelt>,
    encoder: Option<wgpu::CommandEncoder>,
    stats: UploadRingStats,
}

impl std::fmt::Debug for UploadRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadRing")
            .field("belt", &self.belt)
            .field("pending", &self.encoder.is_some())
            .field("stats", &self.stats)
            .finish()
    }
}

impl UploadRing {
    /// Chunk size: larger than any single per-frame stream measured by GPURES-2. A larger write
    /// bypasses the belt (see [`Self::write`]).
    pub const CHUNK_BYTES: u64 = 1 << 20;

    /// An empty ring. Nothing is allocated until the first staged write.
    #[must_use]
    pub fn new() -> Self {
        Self {
            belt: None,
            encoder: None,
            stats: UploadRingStats::default(),
        }
    }

    /// Stage `bytes` for `target` at `offset`, recording the copy into this frame's encoder.
    ///
    /// Same contract as `Queue::write_buffer`: `offset` and `bytes.len()` are multiples of
    /// `wgpu::COPY_BUFFER_ALIGNMENT` and `target` has `COPY_DST`. Empty writes are dropped.
    pub fn write(
        &mut self,
        device: &wgpu::Device,
        target: &wgpu::Buffer,
        offset: u64,
        bytes: &[u8],
    ) {
        let Some(size) = wgpu::BufferSize::new(bytes.len() as u64) else {
            return;
        };
        let encoder = self.encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc-render staged uploads"),
            })
        });
        self.stats.writes += 1;
        self.stats.bytes += size.get();
        // Phase 4d review S2: the belt serves a write larger than a chunk from a dedicated chunk
        // and then keeps that chunk in its free list for good, so one server-driven burst (an
        // uncapped lightning storm) would pin its peak until exit. Such a write gets a one-shot
        // staging buffer instead, recorded into the same encoder: call order is unchanged and
        // the memory goes once the copy has run.
        if size.get() > Self::CHUNK_BYTES {
            if let Some(staging) = one_shot_staging(device, bytes) {
                encoder.copy_buffer_to_buffer(&staging, 0, target, offset, size.get());
                return;
            }
        }
        let belt = self
            .belt
            .get_or_insert_with(|| wgpu::util::StagingBelt::new(device.clone(), Self::CHUNK_BYTES));
        belt.write_buffer(encoder, target, offset, size)
            .copy_from_slice(bytes);
    }

    /// [`Self::write`] for a writer that produces its `size` bytes straight
    /// into the staging memory: `fill` gets exactly `size` bytes to
    /// write, all of which it must write. The copy, its position and the bytes are those of
    /// [`Self::write`] with the same bytes; only the caller's own serialisation buffer and the
    /// copy out of it are gone.
    pub fn write_with(
        &mut self,
        device: &wgpu::Device,
        target: &wgpu::Buffer,
        offset: u64,
        size: wgpu::BufferSize,
        fill: impl FnOnce(wgpu::WriteOnly<'_, [u8]>),
    ) {
        let encoder = self.encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc-render staged uploads"),
            })
        });
        self.stats.writes += 1;
        self.stats.bytes += size.get();
        // As in `write`: an oversized write gets a one-shot staging buffer.
        if size.get() > Self::CHUNK_BYTES {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc-render oversized staged upload"),
                size: size.get(),
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            // An invalid buffer (a lost device) cannot be mapped or unmapped.
            let Ok(mut view) = staging.get_mapped_range_mut(..) else {
                return;
            };
            fill(view.slice(..));
            drop(view);
            staging.unmap();
            encoder.copy_buffer_to_buffer(&staging, 0, target, offset, size.get());
            return;
        }
        let belt = self
            .belt
            .get_or_insert_with(|| wgpu::util::StagingBelt::new(device.clone(), Self::CHUNK_BYTES));
        let mut view = belt.write_buffer(encoder, target, offset, size);
        fill(view.slice(..));
    }

    /// Close this frame's staging and return its copies, or `None` if nothing was staged. The
    /// caller must submit the command buffer.
    ///
    /// Its chunks are re-mapped by that submission (`StagingBelt::finish_and_recall_on_submit`),
    /// not by a later `recall()` call: `recall` maps every closed chunk, including one closed by
    /// a frame that is still waiting to be submitted (the pipelined submit hands frames to
    /// another thread), and a mapped chunk in a submission is a validation error.
    pub fn finish(&mut self) -> Option<wgpu::CommandBuffer> {
        let encoder = self.encoder.take()?;
        if let Some(belt) = self.belt.as_mut() {
            belt.finish_and_recall_on_submit(&encoder);
        }
        self.stats.submissions += 1;
        Some(encoder.finish())
    }

    /// Counters since the previous call.
    pub fn take_stats(&mut self) -> UploadRingStats {
        std::mem::take(&mut self.stats)
    }
}

impl Default for UploadRing {
    fn default() -> Self {
        Self::new()
    }
}

/// A staging buffer created mapped and filled with `bytes`, or `None` if it cannot be mapped
/// (the caller then falls back to the belt).
fn one_shot_staging(device: &wgpu::Device, bytes: &[u8]) -> Option<wgpu::Buffer> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mc-render oversized staged upload"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    let filled = match staging.get_mapped_range_mut(..) {
        Ok(mut view) => {
            view.slice(..).copy_from_slice(bytes);
            true
        }
        Err(_) => false,
    };
    // No mapping means an invalid buffer (a lost device, or no memory for it):
    // unmapping that is a wgpu validation error, which panics by default. Give up quietly.
    if !filled {
        return None;
    }
    staging.unmap();
    Some(staging)
}

#[cfg(test)]
mod tests {
    use crate::Gpu;

    fn read_back(gpu: &Gpu, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upload ring readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, size);
        // Exactly the frame path: staged copies first, then this "frame" command buffer.
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

    /// R1-8 (U7): staged writes land after the frame's plain queue writes, before the frame's
    /// commands and in call order among themselves, across frames that recycle belt chunks;
    /// with the ring off the same calls are plain writes with the same result.
    #[test]
    fn retained_upload_ring_orders_like_write_buffer_across_frames() {
        for enabled in [true, false] {
            let mut gpu = Gpu::headless(4, 4, crate::GpuOptions::default())
                .expect("this qualification requires a working GPU adapter");
            gpu.set_upload_ring_enabled(enabled);
            let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let target = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upload ring target"),
                size: 64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            for frame in 0..4_u8 {
                gpu.queue.write_buffer(&target, 0, &[frame; 64]);
                gpu.write_buffer_staged(&target, 8, &[frame + 100; 16]);
                gpu.write_buffer_staged(&target, 16, &[frame + 200; 16]);
                let mut expected = vec![frame; 64];
                expected[8..24].fill(frame + 100);
                expected[16..32].fill(frame + 200);
                assert_eq!(
                    read_back(&gpu, &target, 64),
                    expected,
                    "frame={frame} ring={enabled}"
                );
            }
            let stats = gpu.take_upload_ring_stats();
            assert_eq!(stats.writes, if enabled { 8 } else { 0 });
            assert_eq!(stats.bytes, if enabled { 8 * 16 } else { 0 });
            assert!(
                pollster::block_on(errors.pop()).is_none(),
                "unexpected wgpu validation error"
            );
        }
    }

    /// Phase 4d review S2: a write larger than a belt chunk goes through a one-shot staging
    /// buffer recorded into the same encoder, so it still lands after the frame's plain writes
    /// and before a smaller staged write made after it.
    #[test]
    fn oversized_staged_writes_keep_call_order() {
        let mut gpu = Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        gpu.set_upload_ring_enabled(true);
        let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let size = super::UploadRing::CHUNK_BYTES + 256;
        let target = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oversized upload target"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&target, 0, &vec![1; size as usize]);
        gpu.write_buffer_staged(&target, 0, &vec![2; size as usize]);
        gpu.write_buffer_staged(&target, 16, &[3; 16]);
        let mut expected = vec![2; size as usize];
        expected[16..32].fill(3);
        assert!(read_back(&gpu, &target, size) == expected);
        let stats = gpu.take_upload_ring_stats();
        assert_eq!((stats.writes, stats.bytes), (2, size + 16));
        assert!(
            pollster::block_on(errors.pop()).is_none(),
            "unexpected wgpu validation error"
        );
    }

    /// On a lost device the one-shot staging buffer is invalid: it gives up
    /// quietly instead of unmapping a buffer that was never mapped (a wgpu error, which panics
    /// by default). The loss is played with `Device::destroy`.
    #[test]
    fn oversized_staging_on_a_lost_device_gives_up_without_a_panic() {
        let Ok(gpu) = Gpu::headless(4, 4, crate::GpuOptions::default()) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        assert!(super::one_shot_staging(&gpu.device, &[7; 256]).is_some());
        gpu.device.destroy();
        assert!(super::one_shot_staging(&gpu.device, &[7; 256]).is_none());
    }

    /// After a device loss, staged writes are dropped: wgpu's staging belt would
    /// panic mapping a new chunk ("Failed to get mapped range for staging belt buffer"), the same
    /// panic family as the AMD export crash. Small and oversized writes, then the frame's flush.
    #[test]
    fn staged_writes_after_a_device_loss_are_dropped_without_a_panic() {
        let Ok(mut gpu) = Gpu::headless(4, 4, crate::GpuOptions::default()) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        gpu.set_upload_ring_enabled(true);
        let size = super::UploadRing::CHUNK_BYTES + 256;
        let target = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lost-device upload target"),
            size,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // `destroy` reports the loss at the next poll; a driver's loss is reported at once.
        gpu.device.destroy();
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        assert!(gpu.is_device_lost());
        gpu.write_buffer_staged(&target, 0, &[1; 64]);
        gpu.write_buffer_staged(&target, 0, &vec![2; size as usize]);
        gpu.flush_staged_uploads();
    }
}
