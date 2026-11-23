//! With the render adapter in the config, the hardware encoder runs on
//! that adapter.
//!
//! For every graphics adapter wgpu lists (Vulkan), a short H.264 encode names that adapter in
//! `EncoderConfig::adapter`. Passes without checking when Media Foundation or a hardware encoder
//! is missing. `MC_W19_R2_TEST_ADAPTER=<part of a name>` limits it to one adapter;
//! `MC_W19_R2_TEST_SIZE=1920x1080` and `MC_W19_R2_TEST_FPS=480` change the encode.

#![cfg(windows)]

use std::path::PathBuf;

use mc_video::{probe, EncoderConfig, FrameRate, GpuAdapter, Mp4Writer, VideoCodec};

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("MC_VIDEO_SAMPLES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("mc-video-tests"));
    std::fs::create_dir_all(&dir).expect("output dir");
    dir
}

/// wgpu's native futures are ready at once; a no-op waker is enough.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::yield_now();
    }
}

fn hardware_adapters() -> Vec<wgpu::AdapterInfo> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let only = std::env::var("MC_W19_R2_TEST_ADAPTER")
        .ok()
        .map(|name| name.to_ascii_lowercase());
    block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
        .into_iter()
        .map(|adapter| adapter.get_info())
        .filter(|info| {
            matches!(
                info.device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            )
        })
        .filter(|info| {
            only.as_ref()
                .is_none_or(|name| info.name.to_ascii_lowercase().contains(name.as_str()))
        })
        .collect()
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

#[test]
fn the_hardware_encoder_runs_on_the_render_adapter() {
    if probe::codec_support(VideoCodec::H264) != Some(true) {
        eprintln!("no hardware H.264 encoder: skipping");
        return;
    }
    // A file per adapter and setting; a fresh name keeps runs from clobbering each other.
    let (width, height) = std::env::var("MC_W19_R2_TEST_SIZE")
        .ok()
        .and_then(|size| {
            let (w, h) = size.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((1280_u32, 720_u32));
    let fps: u32 = env_or("MC_W19_R2_TEST_FPS", 60);
    let frames: u64 = env_or("MC_W19_R2_TEST_FRAMES", 60);
    let codec = match std::env::var("MC_W19_R2_TEST_CODEC").as_deref() {
        Ok("h265") => VideoCodec::H265,
        _ => VideoCodec::H264,
    };
    let adapters = hardware_adapters();
    if adapters.is_empty() {
        eprintln!("no hardware adapter: skipping");
        return;
    }
    for info in adapters {
        let adapter = GpuAdapter {
            vendor: info.vendor,
            device: info.device,
            name: info.name.clone(),
        };
        let path = out_dir().join(format!(
            "w19-r2-{:04x}-{width}x{height}-{fps}.mp4",
            info.device
        ));
        let mut config = EncoderConfig::new(&path, width, height, FrameRate::whole(fps), codec);
        config.adapter = Some(adapter);
        let started = std::time::Instant::now();
        let mut writer = Mp4Writer::create(config).expect("encoder");
        let opened = started.elapsed();
        let encoder = writer.info().clone();
        let mut frame = vec![0_u8; writer.config().nv12_frame_bytes()];
        for index in 0..frames {
            // A moving gradient: something to encode.
            for (i, byte) in frame.iter_mut().enumerate() {
                *byte = (i as u64 / 7 + index * 3) as u8;
            }
            writer.write_frame(&frame).expect("frame");
        }
        let summary = writer.finish().expect("finish");
        eprintln!(
            "adapter {:?} {:04X}:{:04X} -> encoder {:?} hardware={} nominal={} frames={} bytes={} \
             open={opened:?} total={:?}",
            info.name,
            info.vendor,
            info.device,
            encoder.name,
            encoder.hardware,
            encoder.nominal_rate.as_f64(),
            summary.frames,
            summary.bytes,
            started.elapsed(),
        );
        assert_eq!(summary.frames, frames);
        assert!(encoder.hardware, "{encoder:?}");
        // The encoder belongs to the adapter named in the config. (With the device switched off,
        // `MC_W19_R2_ENCODER_DEVICE=0`, this PC's AMD adapter got NVIDIA's encoder: the negative
        // control.)
        if std::env::var_os("MC_W19_R2_ENCODER_DEVICE").is_none()
            && std::env::var_os("MC_W19_R2_ENCODER_ADAPTER").is_none()
        {
            let vendor = match info.vendor {
                0x1002 => Some("AMD"),
                0x10DE => Some("NVIDIA"),
                0x8086 => Some("Intel"),
                _ => None,
            };
            if let Some(vendor) = vendor {
                assert!(
                    encoder.name.contains(vendor),
                    "{} rendered, but the encoder is {:?}",
                    info.name,
                    encoder.name
                );
            }
        }
        if std::env::var_os("MC_VIDEO_SAMPLES_DIR").is_none() {
            let _ = std::fs::remove_file(&path);
        }
    }
}
