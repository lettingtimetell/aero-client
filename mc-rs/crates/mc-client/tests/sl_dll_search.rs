//! The safe DLL search order the client sets first thing in `main`
//! (`mc_render::dll_search::harden_dll_search_order`).
//!
//! The search order is process-wide and cannot be undone, so every case runs in a child process
//! (this test binary again, one `child_*` test, selected with `SL_DLL_CHILD`).
//!
//! * `a_dll_planted_in_the_current_directory_no_longer_loads`: the launcher starts the client
//!   in the shareable `game` folder; a `dxcompiler.dll` there ran when a DX12 instance was made.
//!   Needs `SL_PLANTED_DLL`: a DLL whose `DllMain` writes the file named by
//!   `SL_PLANTED_MARKER` (a harmless stand-in for a planted one).
//! * `gpus_audio_and_video_work_with_the_safe_search_order`: with the safe order, both GPUs
//!   render on Vulkan and on DX12, the default audio output opens (nothing is played) and Media
//!   Foundation encodes H.264 on each GPU's encoder and on the processor. Needs `SL_DLL_VERIFY=1`
//!   (it uses this PC's real devices).

use std::ffi::OsStr;
use std::path::Path;
use std::process::{Command, Output};

const CHILD: &str = "SL_DLL_CHILD";

/// Run this binary's test `name` alone, in `dir`, with `env`.
fn run_child(name: &str, dir: &Path, env: &[(&str, &OsStr)]) -> Output {
    let exe = std::env::current_exe().expect("the test binary");
    let mut command = Command::new(exe);
    command
        .args([name, "--exact", "--nocapture", "--test-threads=1"])
        .current_dir(dir)
        .env(CHILD, "1")
        .env_remove(mc_render::dll_search::DLL_SEARCH_ENV);
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().expect("run the child")
}

fn report(what: &str, output: &Output) -> String {
    format!(
        "{what}: {}\n--- stdout\n{}\n--- stderr\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn in_child() -> bool {
    std::env::var_os(CHILD).is_some()
}

fn harden_unless_plain() {
    if std::env::var_os("SL_DLL_PLAIN").is_none() {
        assert!(
            mc_render::dll_search::harden_dll_search_order(),
            "the safe search order was refused"
        );
    }
}

#[test]
fn a_dll_planted_in_the_current_directory_no_longer_loads() {
    let Some(planted) = std::env::var_os("SL_PLANTED_DLL") else {
        eprintln!("SKIP: set SL_PLANTED_DLL to a marker DLL (see the module docs)");
        return;
    };
    for hardened in [false, true] {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::copy(&planted, dir.path().join("dxcompiler.dll")).expect("plant the DLL");
        let marker = dir.path().join("planted-dll-ran.txt");
        let mut env: Vec<(&str, &OsStr)> = vec![("SL_PLANTED_MARKER", marker.as_os_str())];
        if !hardened {
            env.push(("SL_DLL_PLAIN", OsStr::new("1")));
        }
        let output = run_child("child_creates_a_dx12_instance", dir.path(), &env);
        assert!(output.status.success(), "{}", report("child", &output));
        assert_eq!(
            marker.exists(),
            !hardened,
            "hardened={hardened}: the DLL in the current directory {}",
            if marker.exists() {
                "ran"
            } else {
                "did not run"
            }
        );
        eprintln!(
            "hardened={hardened}: planted dxcompiler.dll {}",
            if marker.exists() {
                "RAN"
            } else {
                "was not loaded"
            }
        );
    }
}

/// Child: make a DX12 instance (its shader compiler is looked up by name when it is made).
#[test]
fn child_creates_a_dx12_instance() {
    if !in_child() {
        return;
    }
    harden_unless_plain();
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::DX12;
    drop(wgpu::Instance::new(descriptor));
}

#[test]
fn gpus_audio_and_video_work_with_the_safe_search_order() {
    if std::env::var_os("SL_DLL_VERIFY").is_none() {
        eprintln!("SKIP: set SL_DLL_VERIFY=1 (uses this PC's GPUs, audio output and encoders)");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    for (adapter, backend) in [
        ("nvidia", "vulkan"),
        ("nvidia", "dx12"),
        ("radeon", "vulkan"),
        ("radeon", "dx12"),
    ] {
        let output = run_child(
            "child_renders_a_frame",
            dir.path(),
            &[
                ("MC_GPU_ADAPTER", OsStr::new(adapter)),
                ("MC_GPU_BACKEND", OsStr::new(backend)),
            ],
        );
        let what = format!("{adapter} {backend}");
        assert!(output.status.success(), "{}", report(&what, &output));
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = marked_lines(&stdout, "SL-GPU ")
            .next()
            .expect("the child names its adapter");
        eprintln!("{what}: {line}");
    }
    let output = run_child("child_opens_audio_and_encodes_video", dir.path(), &[]);
    assert!(
        output.status.success(),
        "{}",
        report("audio and video", &output)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(marked_lines(&stdout, "SL-AUDIO ").next().is_some());
    for line in marked_lines(&stdout, "SL-") {
        eprintln!("{line}");
    }
}

/// The child's lines carrying `marker`, from the marker on (libtest prints a test's first line
/// of output after its `test name ...` header).
fn marked_lines<'a>(stdout: &'a str, marker: &'a str) -> impl Iterator<Item = &'a str> + 'a {
    stdout
        .lines()
        .filter_map(move |line| line.find(marker).map(|at| &line[at..]))
}

/// Child: the GPU `MC_GPU_ADAPTER`/`MC_GPU_BACKEND` name, through the client's own selection,
/// clears a frame that reads back exactly.
#[test]
fn child_renders_a_frame() {
    if !in_child() {
        return;
    }
    harden_unless_plain();
    let mut gpu = mc_render::Gpu::headless(
        64,
        48,
        mc_render::GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    )
    .expect("a GPU");
    let info = gpu.adapter().get_info();
    println!(
        "SL-GPU {} ({:?}, driver {})",
        info.name, info.backend, info.driver_info
    );
    let wanted = std::env::var("MC_GPU_ADAPTER").unwrap_or_default();
    assert!(
        info.name.to_lowercase().contains(&wanted.to_lowercase()),
        "asked for {wanted}, got {}",
        info.name
    );
    let backend = std::env::var("MC_GPU_BACKEND").unwrap_or_default();
    let expected = match backend.as_str() {
        "dx12" => wgpu::Backend::Dx12,
        _ => wgpu::Backend::Vulkan,
    };
    assert_eq!(info.backend, expected);

    let mut phases = mc_render::PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.25,
        g: 0.5,
        b: 0.75,
        a: 1.0,
    };
    let frame = gpu.acquire().expect("acquire the offscreen frame");
    phases.render(&gpu, frame.view(), &mc_render::Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("read back");
    let centre = (24 * 64 + 32) * 4;
    let pixel = &pixels[centre..centre + 4];
    assert!(
        pixel[0].abs_diff(64) <= 2 && pixel[1].abs_diff(128) <= 2 && pixel[2].abs_diff(191) <= 2,
        "{pixel:?}"
    );
}

/// Child: the default audio output through the client's audio engine (opened, nothing played),
/// then H.264 through Media Foundation on each GPU's encoder and on the processor.
#[test]
fn child_opens_audio_and_encodes_video() {
    if !in_child() {
        return;
    }
    harden_unless_plain();
    let mut engine =
        mc_audio::AudioEngine::<mc_audio::DefaultBackend>::new(mc_audio::SoundRegistry::empty(), 0);
    assert!(
        engine.backend_mut().is_some(),
        "the default audio output did not open"
    );
    drop(engine);
    println!("SL-AUDIO the default output device opened");

    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(descriptor);
    let mut targets: Vec<(
        String,
        mc_video::EncoderPreference,
        Option<mc_video::GpuAdapter>,
    )> = vec![(
        "processor".to_owned(),
        mc_video::EncoderPreference::SoftwareOnly,
        None,
    )];
    for adapter in pollster_block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN)) {
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            continue;
        }
        targets.push((
            info.name.clone(),
            mc_video::EncoderPreference::Auto,
            Some(mc_video::GpuAdapter {
                vendor: info.vendor,
                device: info.device,
                name: info.name.clone(),
            }),
        ));
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (width, height) = (320u32, 180u32);
    for (index, (name, preference, adapter)) in targets.into_iter().enumerate() {
        let path = dir.path().join(format!("encode-{index}.mp4"));
        let mut config = mc_video::EncoderConfig::new(
            &path,
            width,
            height,
            mc_video::FrameRate::whole(30),
            mc_video::VideoCodec::H264,
        );
        config.encoder = preference;
        config.adapter = adapter;
        let mut encoder = mc_video::EncoderThread::spawn(config, None, 4).expect("start");
        for _ in 0..30 {
            encoder
                .submit(vec![128u8; (width * height * 3 / 2) as usize])
                .expect("submit a frame");
        }
        let summary = encoder.finish().expect("finish the file");
        assert_eq!(summary.frames, 30, "{name}");
        assert!(summary.bytes > 1000, "{name}: {} bytes", summary.bytes);
        println!(
            "SL-VIDEO {name}: {} bytes, {:?}",
            summary.bytes, summary.info
        );
    }
}

/// Wait for a future (`wgpu`'s native adapter list is ready at once; no executor is needed).
fn pollster_block_on<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::yield_now();
    }
}
