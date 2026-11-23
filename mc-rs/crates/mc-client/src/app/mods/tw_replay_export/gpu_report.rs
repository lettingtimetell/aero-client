//! What the export says when the GPU refuses, and its memory logs.
//!
//! A friend's AMD PC crashed exporting a replay: the GPU device had been lost (a driver reset or
//! hang), and the next terrain buffer panicked. Now a lost device, a capture or target the GPU has
//! no memory for, refused terrain and failed readbacks each end the export with a sentence the
//! player can act on, and the client stops cleanly on a lost device (see
//! `mc_render::gpu_health`).

use std::sync::OnceLock;

use mc_render::gpu_health::{AllocError, GpuMemory};
use mc_render::Gpu;

/// `MC_RD_GPU_MEM=<n>`: log the GPU allocator's numbers every `n` rendered export frames (for
/// memory-growth measurements). Read once.
pub(super) fn memory_log_every() -> Option<u64> {
    static EVERY: OnceLock<Option<u64>> = OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("MC_RD_GPU_MEM")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
    })
}

/// One INFO line with the allocator's numbers (nothing on a backend without a report).
pub(super) fn log_memory(gpu: &Gpu, what: &str) {
    if let Some(memory) = GpuMemory::of(&gpu.device) {
        tracing::info!(
            allocated_mib = memory.allocated / (1024 * 1024),
            reserved_mib = memory.reserved / (1024 * 1024),
            allocations = memory.allocations,
            blocks = memory.blocks,
            "{what}"
        );
    }
}

/// The fix when the graphics card's video encoder or driver fails during an export,
/// worded once so every message names the same place. (A friend's AMD crash stopped through a
/// path that never said it.)
pub(crate) const PROCESSOR_ONLY_HINT: &str =
    "If this keeps happening, set Mods \u{203a} Replay \u{203a} Video encoder to Processor only.";

/// An encoder failure during an export (`what`: what failed), with the fix when
/// the graphics card's encoder was in use or could have been (`hardware`).
pub(super) fn encoder_failure(what: &str, error: &dyn std::fmt::Display, hardware: bool) -> String {
    if hardware {
        format!("{what}: {error}. Nothing was saved. {PROCESSOR_ONLY_HINT}")
    } else {
        format!("{what}: {error}. Nothing was saved.")
    }
}

/// The export's capture buffers were refused.
pub(super) fn export_failure(error: &AllocError, (w, h): (u32, u32), blur: u32) -> String {
    let what = match error {
        AllocError::DeviceLost => {
            return format!(
                "The graphics driver stopped responding during the export. Nothing was saved. \
                 {PROCESSOR_ONLY_HINT}"
            );
        }
        AllocError::TooLarge { .. } => "is over what this graphics card allows in one buffer",
        AllocError::OutOfMemory(_) => "needs more graphics memory than is free",
        AllocError::Invalid(_) => "was refused by the graphics card",
    };
    let blur_hint = if blur > 1 {
        " or turn motion blur off"
    } else {
        ""
    };
    tracing::warn!(%error, w, h, blur, "export: capture refused");
    format!("A {w}\u{00D7}{h} export {what}. Try a smaller size{blur_hint}. Nothing was saved.")
}

/// Rendering at the export's size was refused (`ShellApp::offscreen_failed`).
pub(super) fn offscreen_failure(error: &str, (w, h): (u32, u32)) -> String {
    tracing::warn!(%error, w, h, "export: rendering at the export's size was refused");
    format!(
        "This graphics card has no memory for a {w}\u{00D7}{h} export with the current effects. \
         Try a smaller size, or turn shader packs and other effects off. Nothing was saved."
    )
}

/// The device was lost (the driver's `reason`) while an export ran: for the
/// export window and the crash report alike.
pub(super) fn export_device_lost_text(reason: &str) -> String {
    format!(
        "The graphics driver stopped responding during the export ({reason}). Nothing was saved; \
         Aero Client has to restart. {PROCESSOR_ONLY_HINT}"
    )
}

/// The device is lost. For the export window (`export`) or the client's crash report.
pub(super) fn device_lost_message(gpu: &Gpu, export: bool) -> String {
    let reason = gpu
        .device_lost()
        .unwrap_or_else(|| "no reason given".to_owned());
    if export {
        export_device_lost_text(&reason)
    } else {
        format!(
            "The graphics device was lost: the graphics driver stopped responding or was reset \
             ({reason}). Restart Aero Client. If it keeps happening, update the graphics driver \
             and try a lower render distance or export size."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusals_say_what_to_do() {
        let too_large = AllocError::TooLarge {
            requested: 600 << 20,
            limit: 512 << 20,
        };
        let text = export_failure(&too_large, (7680, 4320), 8);
        assert!(text.contains("7680\u{00D7}4320"), "{text}");
        assert!(text.contains("motion blur"), "{text}");
        assert!(text.ends_with("Nothing was saved."), "{text}");
        let text = export_failure(&AllocError::OutOfMemory("x".into()), (3840, 2160), 1);
        assert!(!text.contains("motion blur"), "{text}");
        assert!(export_failure(&AllocError::DeviceLost, (1, 1), 1).contains("driver"));
        assert!(offscreen_failure("oom", (3840, 2160)).contains("3840\u{00D7}2160"));
    }

    /// Every failure the graphics card's encoder or driver can cause names the
    /// fix; a failure of the processor's encoder does not send the player to the setting they
    /// already use.
    #[test]
    fn encoder_and_driver_failures_name_processor_only() {
        let path = "Mods \u{203a} Replay \u{203a} Video encoder to Processor only";
        assert!(PROCESSOR_ONLY_HINT.contains(path));
        assert!(export_failure(&AllocError::DeviceLost, (1, 1), 1).contains(path));
        let text = encoder_failure("The encoder stopped", &"0x887A0005", true);
        assert!(
            text.starts_with("The encoder stopped: 0x887A0005."),
            "{text}"
        );
        assert!(text.contains(path), "{text}");
        let text = encoder_failure("The encoder stopped", &"0x887A0005", false);
        assert!(!text.contains("Processor only"), "{text}");
        assert!(text.ends_with("Nothing was saved."), "{text}");
        let text = export_device_lost_text("Unexpected error variant");
        assert!(text.contains("(Unexpected error variant)"), "{text}");
        assert!(text.contains(path), "{text}");
    }
}
