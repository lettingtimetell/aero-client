//! Leak probe for the Aero title screen's UI phases (aero-title 2026-11-07 follow-up).
//!
//! Ignored: builds and drops the title screen's two `AeroTitlePhase`s again and again on a
//! headless GPU, the way the client does on every title-screen visit, and prints the process's
//! resident set after each cycle.
//!
//! ```text
//! cargo test -p mc-ui --test aero_title_leak -- --ignored --nocapture
//! ```
//!
//! `LEAK_MODE=empty` renders the same frames without the phases (the noise floor).

use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::screens::aero_title::phase::AeroTitlePhase;
use mc_ui::screens::modmenu::glass::GlassFrame;
use mc_ui::widget::DrawList;

/// The working set in bytes, the way `mc_client::app::crash::process_memory_bytes` reads it.
fn resident() -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        let pid = std::process::id();
        let output = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&output.stdout);
        let field = text.rsplit('"').nth(1)?;
        let digits: String = field.chars().filter(char::is_ascii_digit).collect();
        digits.parse::<u64>().ok().map(|kib| kib * 1024)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }
}

fn frame(gpu: &mut Gpu, phases: &mut PhaseList) {
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
}

#[test]
#[ignore = "resident-set probe; run by hand"]
fn aero_title_phase_cycles() {
    let mut gpu = match Gpu::headless(
        256,
        192,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => g,
        Err(RenderError::NoAdapter) => return,
        Err(e) => panic!("headless context: {e}"),
    };
    let empty = std::env::var("LEAK_MODE").is_ok_and(|m| m == "empty");
    let cycles: usize = std::env::var("LEAK_CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    let mut phases = PhaseList::new();
    let mut samples = Vec::new();
    for cycle in 0..3 + cycles {
        if !empty {
            let mut ui = AeroTitlePhase::new(&gpu);
            let mut crisp = AeroTitlePhase::new(&gpu);
            ui.submit(DrawList::new(), GlassFrame::default());
            crisp.submit(DrawList::new(), GlassFrame::default());
            phases.insert(PhaseOrder(PhaseOrder::UI.0 + 40), Box::new(ui));
            phases.insert(PhaseOrder(PhaseOrder::UI.0 + 90), Box::new(crisp));
            for _ in 0..4 {
                frame(&mut gpu, &mut phases);
            }
            phases.clear();
        } else {
            for _ in 0..4 {
                frame(&mut gpu, &mut phases);
            }
        }
        for _ in 0..6 {
            frame(&mut gpu, &mut phases);
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        if cycle >= 2 {
            samples.push(resident().unwrap_or(0));
        }
    }
    let first = samples[0];
    let last = *samples.last().expect("samples");
    let deltas: Vec<i64> = samples
        .windows(2)
        .map(|w| (w[1] as i64 - w[0] as i64) / 1024)
        .collect();
    println!(
        "ui phases {}: {:.1} MB -> {:.1} MB over {cycles} cycles = {:.0} KB/cycle; per-cycle KB: {deltas:?}",
        if empty { "empty" } else { "built" },
        first as f64 / 1_048_576.0,
        last as f64 / 1_048_576.0,
        (last as f64 - first as f64) / cycles as f64 / 1024.0,
    );
}
