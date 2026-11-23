//! Compare two real captures without changing the parity thresholds.
//!
//! cargo run -p mc-parity --example compare_snapshots -- reference.png actual.png report.json
//! A failed comparison still writes the report and an absolute RGB difference PNG.

use std::path::PathBuf;

use mc_parity::snapshot::{perceptual_diff, DiffThresholds, Snapshot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: compare_snapshots reference.png actual.png report.json".into());
    }
    let reference_path = PathBuf::from(&args[0]);
    let actual_path = PathBuf::from(&args[1]);
    let report_path = PathBuf::from(&args[2]);
    let difference_path = report_path.with_extension("diff.png");
    let reference = Snapshot::load_png(&reference_path)?;
    let actual = Snapshot::load_png(&actual_path)?;
    let diff = perceptual_diff(&reference, &actual)?;
    let thresholds = DiffThresholds::default();
    let passed = diff.is_within(&thresholds);

    let mut difference = reference.clone();
    for ((out, left), right) in difference
        .rgba
        .chunks_exact_mut(4)
        .zip(reference.rgba.chunks_exact(4))
        .zip(actual.rgba.chunks_exact(4))
    {
        for channel in 0..3 {
            out[channel] = left[channel].abs_diff(right[channel]);
        }
        out[3] = 255;
    }
    difference.save_png(&difference_path)?;
    let report = serde_json::json!({
        "reference": reference_path,
        "actual": actual_path,
        "difference": difference_path,
        "passed": passed,
        "width": diff.width,
        "height": diff.height,
        "max_delta_e": diff.max_delta_e,
        "mean_delta_e": diff.mean_delta_e,
        "pixels_over_jnd": diff.pixels_over_jnd,
        "fraction_over_jnd": diff.fraction_over_jnd,
        "ssim": diff.ssim,
        "max_channel_delta": diff.max_channel_delta,
        "max_alpha_delta": diff.max_alpha_delta,
        "differing_pixels": diff.differing_pixels,
        "worst_pixel": diff.worst_pixel,
        "thresholds": {
            "max_delta_e": thresholds.max_delta_e,
            "max_fraction_over_jnd": thresholds.max_fraction_over_jnd,
            "min_ssim": thresholds.min_ssim,
            "max_alpha_delta": thresholds.max_alpha_delta,
        },
    });
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}; passed={passed}; report={}",
        diff.summary(),
        report_path.display()
    );
    if !passed {
        std::process::exit(1);
    }
    Ok(())
}
