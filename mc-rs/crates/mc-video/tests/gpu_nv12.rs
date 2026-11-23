//! The export's GPU path (mc-render `export::ExportCapture`: frame blending, NV12 compute,
//! pipelined readback) against this crate's CPU reference (`nv12::from_rgba`).
//!
//! Runs on a headless wgpu device; passes without checking when the machine has no adapter.

use mc_render::export::{Captured, ExportCapture};
use mc_render::{Gpu, GpuOptions, HEADLESS_FORMAT};
use mc_video::nv12::{self, PixelOrder};

fn gpu(width: u32, height: u32) -> Option<Gpu> {
    let mut gpu = match Gpu::headless(16, 16, GpuOptions::default()) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("no GPU adapter ({error}): skipping");
            return None;
        }
    };
    // The export's offscreen target, exactly as the client gets it.
    assert!(gpu.set_render_override(Some((width, height))));
    assert_eq!((gpu.config.width, gpu.config.height), (width, height));
    Some(gpu)
}

fn upload(gpu: &mut Gpu, rgba: &[u8]) {
    let frame = gpu.acquire().expect("offscreen frame");
    let texture = frame.color_texture(gpu).expect("offscreen texture").clone();
    let (w, h) = (gpu.config.width, gpu.config.height);
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

fn texture(gpu: &mut Gpu) -> wgpu::Texture {
    let frame = gpu.acquire().expect("offscreen frame");
    frame.color_texture(gpu).expect("offscreen texture").clone()
}

/// A busy test image: gradients, hard edges and a colour checkerboard.
fn image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = vec![0_u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            let checker = ((x / 3 + y / 5 + seed) % 2) * 120;
            out[o] = ((x * 255 / w.max(1)) as u8).wrapping_add((seed * 40) as u8);
            out[o + 1] = ((y * 255 / h.max(1)) as u8) ^ checker as u8;
            out[o + 2] = ((x * 7 + y * 13 + seed * 31) % 256) as u8;
            out[o + 3] = 255;
        }
    }
    out
}

fn max_difference(a: &[u8], b: &[u8]) -> (u8, usize) {
    assert_eq!(a.len(), b.len());
    let mut worst = (0_u8, 0_usize);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = x.abs_diff(*y);
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

fn check_sizes(w: u32, h: u32) {
    let Some(mut gpu) = gpu(w, h) else {
        return;
    };
    let rgba = image(w, h, 1);
    upload(&mut gpu, &rgba);
    let source = texture(&mut gpu);
    let mut capture = ExportCapture::new(&gpu, w, h, HEADLESS_FORMAT, 1, 3);
    let mut got = Vec::new();
    assert_eq!(
        capture.capture(&gpu, &source, &mut |_, bytes: &[u8]| got
            .push(bytes.to_vec())),
        Captured::Queued(0)
    );
    capture.finish(&gpu, &mut |_, bytes: &[u8]| got.push(bytes.to_vec()));
    assert_eq!(got.len(), 1);
    let mut expected = vec![0_u8; nv12::frame_bytes(w, h)];
    nv12::from_rgba(
        &rgba,
        w,
        h,
        (w * 4) as usize,
        PixelOrder::Rgba,
        &mut expected,
    );
    let (worst, at) = max_difference(&got[0], &expected);
    assert!(
        worst <= 1,
        "{w}x{h}: byte {at} differs by {worst} (gpu {}, cpu {})",
        got[0][at],
        expected[at]
    );
    // Negative control: the same comparison against red and blue swapped fails loudly, so the
    // check above can tell a wrong conversion from a right one.
    let mut swapped = vec![0_u8; nv12::frame_bytes(w, h)];
    nv12::from_rgba(
        &rgba,
        w,
        h,
        (w * 4) as usize,
        PixelOrder::Bgra,
        &mut swapped,
    );
    assert!(
        max_difference(&got[0], &swapped).0 > 20,
        "{w}x{h}: control did not differ"
    );
}

#[test]
fn gpu_nv12_matches_the_cpu_reference() {
    check_sizes(64, 32);
    // A width that is not a multiple of 4: words straddle rows.
    check_sizes(34, 18);
    check_sizes(1280, 720);
}

#[test]
fn readback_ring_delivers_every_frame_in_order() {
    let (w, h) = (96, 54);
    let Some(mut gpu) = gpu(w, h) else {
        return;
    };
    let mut capture = ExportCapture::new(&gpu, w, h, HEADLESS_FORMAT, 1, 3);
    let mut delivered: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut expected = Vec::new();
    for seed in 0..10 {
        let rgba = image(w, h, seed);
        let mut nv = vec![0_u8; nv12::frame_bytes(w, h)];
        nv12::from_rgba(&rgba, w, h, (w * 4) as usize, PixelOrder::Rgba, &mut nv);
        expected.push(nv);
        upload(&mut gpu, &rgba);
        let source = texture(&mut gpu);
        let mut sink = |index: u64, bytes: &[u8]| delivered.push((index, bytes.to_vec()));
        capture.capture(&gpu, &source, &mut sink);
        capture.poll(&gpu, &mut sink);
    }
    capture.finish(&gpu, &mut |index: u64, bytes: &[u8]| {
        delivered.push((index, bytes.to_vec()))
    });
    let indices: Vec<u64> = delivered.iter().map(|(i, _)| *i).collect();
    assert_eq!(indices, (0..10).collect::<Vec<_>>());
    for ((index, bytes), want) in delivered.iter().zip(&expected) {
        let (worst, _) = max_difference(bytes, want);
        assert!(worst <= 1, "frame {index} differs by {worst}");
    }
    let stats = capture.stats();
    eprintln!("{stats:?}");
    assert_eq!(stats.delivered, 10);
    assert_eq!(stats.failed, 0);
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[test]
fn frame_blending_averages_in_linear_light() {
    let (w, h) = (40, 24);
    let Some(mut gpu) = gpu(w, h) else {
        return;
    };
    let samples = 4;
    let mut capture = ExportCapture::new(&gpu, w, h, HEADLESS_FORMAT, samples, 2);
    let images: Vec<Vec<u8>> = (0..samples).map(|seed| image(w, h, seed * 3)).collect();
    let mut got = Vec::new();
    for (i, rgba) in images.iter().enumerate() {
        upload(&mut gpu, rgba);
        let source = texture(&mut gpu);
        let result = capture.capture(&gpu, &source, &mut |_, b: &[u8]| got.push(b.to_vec()));
        if i + 1 < samples as usize {
            assert_eq!(result, Captured::Accumulated);
        } else {
            assert_eq!(result, Captured::Queued(0));
        }
    }
    capture.finish(&gpu, &mut |_, b: &[u8]| got.push(b.to_vec()));
    assert_eq!(got.len(), 1, "four rendered frames make one output frame");
    // CPU reference: the mean of the four frames in linear light, back to display bytes.
    let mut mean = vec![0_u8; (w * h * 4) as usize];
    for (p, out) in mean.chunks_exact_mut(4).enumerate() {
        for c in 0..3 {
            let sum: f32 = images
                .iter()
                .map(|img| srgb_to_linear(f32::from(img[p * 4 + c]) / 255.0))
                .sum();
            out[c] = (linear_to_srgb(sum / samples as f32) * 255.0).round() as u8;
        }
        out[3] = 255;
    }
    let mut expected = vec![0_u8; nv12::frame_bytes(w, h)];
    nv12::from_rgba(
        &mean,
        w,
        h,
        (w * 4) as usize,
        PixelOrder::Rgba,
        &mut expected,
    );
    let (worst, at) = max_difference(&got[0], &expected);
    // The GPU keeps the mean in float; the reference rounds it to 8 bits first.
    assert!(worst <= 2, "byte {at} differs by {worst}");
    // And it really is a blend: not equal to any single frame's conversion.
    let mut first = vec![0_u8; nv12::frame_bytes(w, h)];
    nv12::from_rgba(
        &images[0],
        w,
        h,
        (w * 4) as usize,
        PixelOrder::Rgba,
        &mut first,
    );
    assert!(max_difference(&got[0], &first).0 > 10);
}

#[test]
fn override_round_trip_restores_the_original_size() {
    let Some(mut gpu) = gpu(200, 100) else {
        return;
    };
    assert_eq!(gpu.render_override(), Some((200, 100)));
    assert!(gpu.set_render_override(None));
    assert_eq!((gpu.config.width, gpu.config.height), (16, 16));
    assert_eq!(gpu.render_override(), None);
    assert!(!gpu.set_render_override(None), "no change");
}
