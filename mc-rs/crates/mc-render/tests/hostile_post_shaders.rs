//! A resource pack can replace the post-processing shader groups (`shaders/post/*.json`, read
//! at every resource load for the entity outline, and by the shader menu) and their programs
//! (`shaders/program/*.json`). Their JSON reader must not recurse without bound, and a target
//! size the JSON names must not reach the graphics device unchecked.

use std::sync::mpsc;
use std::time::Duration;

use mc_assets::ResourceManager;
use mc_render::post::{PostChain, ProgramSpec, ShaderGroupSpec};
use mc_render::{Gpu, GpuOptions, RenderError};

/// Runs `work` on a thread with the main thread's 1 MiB of stack.
fn on_small_stack<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            let _ = tx.send(work());
        })
        .expect("spawn");
    rx.recv_timeout(Duration::from_secs(60))
        .expect("the parse returned")
}

#[test]
fn deeply_nested_shader_json_is_refused_without_overflowing_the_stack() {
    let refused = on_small_stack(|| {
        let arrays = "[".repeat(200_000);
        let objects = r#"{"a":"#.repeat(200_000);
        [
            ShaderGroupSpec::parse(&arrays).is_err(),
            ShaderGroupSpec::parse(&objects).is_err(),
            ProgramSpec::parse(&arrays).is_err(),
        ]
    });
    assert_eq!(refused, [true; 3]);
    // Ordinary nesting still parses.
    let group = ShaderGroupSpec::parse(
        r#"{"targets":["swap"],"passes":[{"name":"blit","intarget":"minecraft:main",
        "outtarget":"swap","uniforms":[{"name":"ColorScale","values":[1.0,1.0,1.0]}]}]}"#,
    )
    .expect("parses");
    assert_eq!(group.passes.len(), 1);
}

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        64,
        64,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless GPU: {error}"),
    }
}

/// A fixed-size target wider than any device allows used to reach `create_texture` as given: a
/// validation error, which wgpu's default handler turns into a panic, at every resource load
/// while the pack is selected (the entity outline's group is read then).
#[test]
fn a_shader_target_larger_than_the_device_allows_does_not_reach_the_device() {
    let Some(gpu) = gpu() else {
        return;
    };
    let limit = gpu.device.limits().max_texture_dimension_2d;
    let spec = ShaderGroupSpec::parse(&format!(
        r#"{{"targets":[{{"name":"huge","width":{},"height":4}},{{"name":"far","width":1e12,"height":1e12}}],
        "passes":[{{"name":"blit","intarget":"minecraft:main","outtarget":"huge"}},
                  {{"name":"blit","intarget":"huge","outtarget":"far"}}]}}"#,
        u64::from(limit) + 1
    ))
    .expect("parses");
    let rm = ResourceManager::new();
    let chain = PostChain::new(&gpu, &rm, &spec, wgpu::TextureFormat::Rgba8Unorm, (64, 64));
    // Reaching here without wgpu's validation panic is the test; the chain itself is usable.
    let _ = chain.shader_source_error();
}
