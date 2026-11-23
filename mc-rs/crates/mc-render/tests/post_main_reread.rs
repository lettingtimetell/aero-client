//! Post shader groups that read `"minecraft:main"` again after a pass wrote it (vanilla's `art`,
//! `green` and `ntsc`), run the way the game runs them: the scene's snapshot in, the frame's own
//! target out. That target (a swap-chain image, or the target a frame is recorded into before
//! its image) cannot be sampled, so a group that sampled it after writing it failed wgpu's
//! validation, and picking one of those groups in Super Secret Settings crashed the client. The
//! picture must equal the same group run into a target that can be sampled
//! (`PostChain::run_standalone`).
//!
//! Skips when the vanilla jar or a GPU adapter is unavailable (`MC_JARS_DIR` points at `jars/`).

use std::path::{Path, PathBuf};

use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::post::{PostChain, ShaderGroupSpec, TargetSpec};
use mc_render::{Gpu, GpuOptions, RenderError, HEADLESS_FORMAT};
use wgpu::util::DeviceExt;

const SIZE: u32 = 8;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let path = PathBuf::from(dir);
        return path.is_dir().then_some(path);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let path = root.join("MCP-919-main").join("jars");
    path.is_dir().then_some(path)
}

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        SIZE,
        SIZE,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

/// A checkerboard: every group's programs are neighbourhood filters or colour transforms, which
/// would leave a flat image unchanged.
fn checkerboard() -> Vec<u8> {
    let mut pixels = vec![0u8; (SIZE * SIZE * 4) as usize];
    for (index, pixel) in pixels.chunks_exact_mut(4).enumerate() {
        let (x, y) = (index as u32 % SIZE, index as u32 / SIZE);
        let on = (x + y) % 2 == 0;
        pixel.copy_from_slice(&if on {
            [220, 120, 40, 255]
        } else {
            [30, 200, 90, 255]
        });
    }
    pixels
}

/// The group run as the post phase runs it: `main_in` a sampleable snapshot of the scene,
/// `main_out` the frame's target, which (like the window's) is only a render attachment and a
/// copy source.
fn run_into_the_frame(gpu: &mut Gpu, chain: &PostChain, input: &[u8]) -> Vec<u8> {
    let snapshot = gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: Some("scene snapshot (test)"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HEADLESS_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        input,
    );
    let snapshot_view = snapshot.create_view(&wgpu::TextureViewDescriptor::default());
    let frame = gpu.acquire().expect("acquire the frame");
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("post chain into the frame (test)"),
        });
    chain.run(gpu, &mut enc, &snapshot_view, frame.view(), 0.0);
    gpu.queue.submit([enc.finish()]);
    let pixels = frame.read_pixels(gpu).expect("read the frame back");
    frame.present(&gpu.queue);
    pixels
}

#[test]
fn a_group_reading_main_after_writing_it_runs_into_the_frames_own_target() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: no vanilla jar");
        return;
    };
    let Some(mut gpu) = headless() else {
        return;
    };
    let input = checkerboard();
    // `invert` writes main only in its last pass: the control, on the path it always took.
    for name in ["art", "green", "ntsc", "invert"] {
        let location = ResourceLocation::parse(&format!("shaders/post/{name}.json")).unwrap();
        let text = rm.load_string(&location).expect("the group's JSON");
        let spec = ShaderGroupSpec::parse(&text).expect("the group parses");
        // A fresh chain for each run: `ntsc` keeps the previous frame between runs.
        let reference = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (SIZE, SIZE))
            .run_standalone(&gpu, HEADLESS_FORMAT, SIZE, SIZE, &input);
        let chain = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (SIZE, SIZE));
        let framed = run_into_the_frame(&mut gpu, &chain, &input);
        assert_ne!(framed, input, "{name}: the group must change the picture");
        assert_eq!(
            framed, reference,
            "{name}: the frame must show what a sampleable target gets"
        );
    }
}

/// The reference above is built by `PostChain::new` too, so it goes through the same routing.
/// This one does not: a group that already names a target like the internal copy is built
/// exactly as written, and run into a target that can be sampled it reads the main target back
/// directly (vanilla's data flow: `"minecraft:main"` is one framebuffer, read and written in
/// pass order). The routed group, run into the frame, must draw the same picture.
#[test]
fn the_routed_group_draws_what_the_group_as_written_draws() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: no vanilla jar");
        return;
    };
    let Some(mut gpu) = headless() else {
        return;
    };
    let input = checkerboard();
    for name in ["art", "green", "ntsc"] {
        let location = ResourceLocation::parse(&format!("shaders/post/{name}.json")).unwrap();
        let text = rm.load_string(&location).expect("the group's JSON");
        let spec = ShaderGroupSpec::parse(&text).expect("the group parses");
        let mut as_written = spec.clone();
        as_written
            .targets
            .push(TargetSpec::Named("minecraft:main (copy)".to_owned()));
        let reference = PostChain::new(&gpu, &rm, &as_written, HEADLESS_FORMAT, (SIZE, SIZE))
            .run_standalone(&gpu, HEADLESS_FORMAT, SIZE, SIZE, &input);
        let chain = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (SIZE, SIZE));
        let framed = run_into_the_frame(&mut gpu, &chain, &input);
        assert_ne!(
            reference, input,
            "{name}: the group must change the picture"
        );
        assert_eq!(
            framed, reference,
            "{name}: the routed group must draw what the group as written draws"
        );
    }
}
