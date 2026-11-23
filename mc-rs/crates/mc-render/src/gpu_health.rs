//! What the renderer does when the GPU says no.
//!
//! A friend's AMD PC crashed exporting a replay with
//! `Failed to get mapped range for buffer created with mapped_at_creation: ... Buffer with
//! 'terrain batch' label is invalid`. That message means the device had **already been lost**:
//! wgpu-core's `Device::create_buffer` starts with `check_is_valid()`, whose `DeviceError::Lost`
//! wgpu drops silently ("will be surfaced via callback"), so the buffer comes back invalid and
//! only the mapping that `create_buffer_init` does next panics. A size over the device's
//! `max_buffer_size` or a failed allocation would have panicked earlier, at the `create_buffer`
//! itself, with "Validation Error" or "Out of Memory" (wgpu's default error handler).
//!
//! This module holds the three pieces the fix needs:
//!
//! * [`DeviceHealth`]: the device-lost callback's record. The client had none, so a lost device
//!   was never even logged. The frame loop and the export read it with one atomic load and stop
//!   cleanly, naming the driver's reason, instead of panicking at whichever buffer comes next.
//! * [`try_create_buffer_init`] and [`try_create_buffer`]: buffer creation that returns an
//!   [`AllocError`] instead of panicking: a size over the device's limit is refused before
//!   wgpu sees it, an allocation the device cannot satisfy is caught by an error scope, and a lost
//!   device is reported. The GPU work is exactly what `create_buffer_init` and `create_buffer` do
//!   (the same mapped-at-creation upload), so a working device draws the same bytes at the same
//!   cost; the scope adds two short lock-and-push operations per buffer, and the callers use it
//!   only where a buffer is created rarely (terrain arena pages, the index buffer) or off the
//!   normal play path (the export's per-section batches).
//! * [`GpuMemory`]: the allocator's own report (bytes in use and reserved), for the export's logs
//!   and the memory-growth measurements.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Whether the device is still usable, as the device-lost callback last said.
///
/// Built by [`DeviceHealth::watch`] when the device is created ([`crate::Gpu`] does it). wgpu
/// calls the callback when the device is lost (a driver reset, a hang the driver detected, or an
/// error wgpu treats as fatal, such as running out of memory inside a submission) and when
/// `Device::destroy` finishes. Nothing is reported when the device is simply dropped.
#[derive(Debug, Default)]
pub struct DeviceHealth {
    lost: AtomicBool,
    reason: Mutex<Option<String>>,
}

impl DeviceHealth {
    /// Install the device-lost callback on `device` and return the record it fills in.
    pub fn watch(device: &wgpu::Device) -> Arc<DeviceHealth> {
        let health = Arc::new(DeviceHealth::default());
        let weak = Arc::downgrade(&health);
        device.set_device_lost_callback(move |reason, message| {
            if let Some(health) = weak.upgrade() {
                health.mark_lost(reason, &message);
            }
        });
        health
    }

    /// Record a loss. Public so tests and callers that learn of a loss another way (a failed
    /// mapping on a buffer they just created) can record it too; the first reason is kept.
    pub fn mark_lost(&self, reason: wgpu::DeviceLostReason, message: &str) {
        let text = match (reason, message.trim()) {
            (wgpu::DeviceLostReason::Destroyed, "") => "the device was destroyed".to_owned(),
            (wgpu::DeviceLostReason::Destroyed, text) => {
                format!("the device was destroyed: {text}")
            }
            (_, "") => "the graphics driver reported the device as lost".to_owned(),
            (_, text) => text.to_owned(),
        };
        if !self.lost.swap(true, Ordering::AcqRel) {
            tracing::error!(?reason, message = %text, "the GPU device was lost");
        }
        if let Ok(mut slot) = self.reason.lock() {
            slot.get_or_insert(text);
        }
    }

    /// Whether the device has been lost. One atomic load: cheap enough for every frame.
    #[inline]
    #[must_use]
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Why the device was lost, once it has been.
    #[must_use]
    pub fn reason(&self) -> Option<String> {
        if !self.is_lost() {
            return None;
        }
        self.reason.lock().ok().and_then(|slot| slot.clone())
    }
}

/// Why a buffer could not be created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    /// Bigger than the device's `max_buffer_size`. Refused before wgpu sees it.
    TooLarge {
        /// Bytes asked for (after wgpu's 4-byte padding).
        requested: u64,
        /// The device's limit.
        limit: u64,
    },
    /// The device could not find the memory.
    OutOfMemory(String),
    /// Another validation error (the message wgpu gave).
    Invalid(String),
    /// The device is lost: nothing can be created any more.
    DeviceLost,
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::TooLarge { requested, limit } => write!(
                f,
                "a {} buffer is over this GPU's {} limit",
                mib(*requested),
                mib(*limit)
            ),
            AllocError::OutOfMemory(_) => f.write_str("the GPU is out of memory"),
            AllocError::Invalid(message) => write!(f, "the GPU refused the buffer: {message}"),
            AllocError::DeviceLost => f.write_str("the graphics device was lost"),
        }
    }
}

impl std::error::Error for AllocError {}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// The size wgpu's `create_buffer_init` gives `len` bytes: rounded up to
/// `COPY_BUFFER_ALIGNMENT`, and at least that.
fn padded_size(len: u64) -> u64 {
    let mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
    ((len + mask) & !mask).max(wgpu::COPY_BUFFER_ALIGNMENT)
}

/// Run `create` inside out-of-memory and validation error scopes and report what they caught.
fn scoped<T>(device: &wgpu::Device, create: impl FnOnce() -> T) -> Result<T, AllocError> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let made = create();
    // Popped innermost first. The futures are ready at once on native wgpu.
    let invalid = pollster::block_on(validation.pop());
    let out_of_memory = pollster::block_on(oom.pop());
    if let Some(error) = out_of_memory {
        return Err(AllocError::OutOfMemory(error.to_string()));
    }
    if let Some(error) = invalid {
        return Err(AllocError::Invalid(error.to_string()));
    }
    Ok(made)
}

/// `wgpu::util::DeviceExt::create_buffer_init` without its panics.
///
/// The same GPU work: a buffer created mapped, filled, unmapped. What differs is what happens when
/// that fails: `create_buffer_init` panics (its `expect` on the mapping is the friend's crash);
/// this returns an [`AllocError`] and creates nothing usable, so the caller can defer, degrade or
/// report.
///
/// # Errors
///
/// [`AllocError::TooLarge`] over the device's `max_buffer_size`, [`AllocError::OutOfMemory`] or
/// [`AllocError::Invalid`] when wgpu refuses the buffer, [`AllocError::DeviceLost`] when the
/// device is lost (the buffer comes back invalid without any error being raised).
pub fn try_create_buffer_init(
    device: &wgpu::Device,
    desc: &wgpu::util::BufferInitDescriptor<'_>,
) -> Result<wgpu::Buffer, AllocError> {
    if desc.contents.is_empty() {
        // `create_buffer_init` skips the mapping for an empty buffer too.
        return scoped(device, || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: desc.label,
                size: 0,
                usage: desc.usage,
                mapped_at_creation: false,
            })
        });
    }
    let len = desc.contents.len() as u64;
    let size = padded_size(len);
    let limit = device.limits().max_buffer_size;
    if size > limit {
        return Err(AllocError::TooLarge {
            requested: size,
            limit,
        });
    }
    let buffer = scoped(device, || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: desc.label,
            size,
            usage: desc.usage,
            mapped_at_creation: true,
        })
    })?;
    // A buffer wgpu raised no error for is invalid only when the device is lost (its creation
    // error is dropped silently): the mapping is where `create_buffer_init` panicked.
    let Ok(mut view) = buffer.get_mapped_range_mut(..) else {
        return Err(AllocError::DeviceLost);
    };
    view.slice(..desc.contents.len())
        .copy_from_slice(desc.contents);
    drop(view);
    buffer.unmap();
    Ok(buffer)
}

/// `Device::create_buffer` (not mapped) without its panic on failure.
///
/// # Errors
///
/// As [`try_create_buffer_init`], except that a lost device cannot be told from a working one
/// here (an unmapped buffer is never touched): check [`DeviceHealth`] for that.
pub fn try_create_buffer(
    device: &wgpu::Device,
    desc: &wgpu::BufferDescriptor<'_>,
) -> Result<wgpu::Buffer, AllocError> {
    let limit = device.limits().max_buffer_size;
    if desc.size > limit {
        return Err(AllocError::TooLarge {
            requested: desc.size,
            limit,
        });
    }
    scoped(device, || device.create_buffer(desc))
}

/// What the device's memory allocator holds, from `Device::generate_allocator_report`
/// (Vulkan and DX12; `None` elsewhere). Taken a handful of times per export for the logs;
/// never per frame in normal play.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuMemory {
    /// Bytes in live allocations (buffers and textures).
    pub allocated: u64,
    /// Bytes the allocator has reserved from the driver (its blocks).
    pub reserved: u64,
    /// Live allocations.
    pub allocations: usize,
    /// Memory blocks.
    pub blocks: usize,
}

impl GpuMemory {
    /// The device's current report.
    #[must_use]
    pub fn of(device: &wgpu::Device) -> Option<GpuMemory> {
        let report = device.generate_allocator_report()?;
        Some(GpuMemory {
            allocated: report.total_allocated_bytes,
            reserved: report.total_reserved_bytes,
            allocations: report.allocations.len(),
            blocks: report.blocks.len(),
        })
    }
}

impl fmt::Display for GpuMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} in {} allocations, {} reserved in {} blocks",
            mib(self.allocated),
            self.allocations,
            mib(self.reserved),
            self.blocks
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt as _;

    fn headless() -> Option<crate::Gpu> {
        crate::Gpu::headless(8, 8, crate::GpuOptions::default()).ok()
    }

    fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default()
    }

    /// The friend's crash, reproduced: once the device is lost, wgpu's own `create_buffer_init`
    /// panics with exactly the reported message (the negative control), and
    /// `try_create_buffer_init` reports `DeviceLost` instead.
    #[test]
    fn a_lost_device_is_reported_not_a_panic() {
        let Some(gpu) = headless() else {
            eprintln!("skip: no GPU adapter");
            return;
        };
        let desc = wgpu::util::BufferInitDescriptor {
            label: Some("terrain batch"),
            contents: &[7; 64],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        };
        // Working device: both paths succeed.
        assert!(try_create_buffer_init(&gpu.device, &desc).is_ok());
        let _ = gpu.device.create_buffer_init(&desc);

        gpu.device.destroy();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gpu.device.create_buffer_init(&desc)
        }));
        let message = panic_text(&*panicked.expect_err("wgpu's create_buffer_init panics"));
        assert!(
            message.contains("Failed to get mapped range")
                && message.contains("Buffer with 'terrain batch' label is invalid"),
            "the friend's message, got: {message}"
        );
        assert_eq!(
            try_create_buffer_init(&gpu.device, &desc).err(),
            Some(AllocError::DeviceLost)
        );
        // Destroying fires the callback (reason Destroyed) once the queue drains.
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        assert!(
            gpu.device_lost().is_some(),
            "the health record saw the loss"
        );
    }

    /// Over the device's `max_buffer_size`: refused with `TooLarge`, where wgpu's own call would
    /// panic with a validation error (the negative control).
    #[test]
    fn a_buffer_over_the_limit_is_refused_not_a_panic() {
        let Some(gpu) = headless() else {
            eprintln!("skip: no GPU adapter");
            return;
        };
        let limit = 1 << 20;
        let (device, _queue) =
            pollster::block_on(gpu.adapter().request_device(&wgpu::DeviceDescriptor {
                label: Some("lowered limits"),
                required_limits: wgpu::Limits {
                    max_buffer_size: limit,
                    ..gpu.adapter().limits()
                },
                ..wgpu::DeviceDescriptor::default()
            }))
            .expect("a device with a lowered buffer limit");
        let contents = vec![1u8; (limit + 4) as usize];
        let desc = wgpu::util::BufferInitDescriptor {
            label: Some("terrain batch"),
            contents: &contents,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        };
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.create_buffer_init(&desc)
        }));
        let message = panic_text(&*panicked.expect_err("wgpu's create_buffer_init panics"));
        assert!(
            message.contains("Validation Error"),
            "a validation panic, not the friend's: {message}"
        );
        assert_eq!(
            try_create_buffer_init(&device, &desc).err(),
            Some(AllocError::TooLarge {
                requested: limit + 4,
                limit
            })
        );
        let unmapped = wgpu::BufferDescriptor {
            label: Some("terrain vertex arena"),
            size: limit * 2,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        };
        assert!(matches!(
            try_create_buffer(&device, &unmapped),
            Err(AllocError::TooLarge { .. })
        ));
        // Under the limit, the bytes arrive.
        let small = wgpu::util::BufferInitDescriptor {
            label: Some("terrain batch"),
            contents: &[9; 12],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
        };
        assert!(try_create_buffer_init(&device, &small).is_ok());
    }

    #[test]
    fn the_allocator_report_counts_a_new_buffer() {
        let Some(gpu) = headless() else {
            eprintln!("skip: no GPU adapter");
            return;
        };
        let Some(before) = GpuMemory::of(&gpu.device) else {
            eprintln!("skip: this backend has no allocator report");
            return;
        };
        let buffer = try_create_buffer(
            &gpu.device,
            &wgpu::BufferDescriptor {
                label: Some("report probe"),
                size: 4 << 20,
                usage: wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            },
        )
        .expect("a 4 MiB buffer");
        let after = GpuMemory::of(&gpu.device).expect("a report");
        assert!(
            after.allocated >= before.allocated + (4 << 20),
            "{before} -> {after}"
        );
        drop(buffer);
    }
}
