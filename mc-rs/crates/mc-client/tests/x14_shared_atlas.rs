//! The live client's render-stack install gives the five
//! full-mip block-atlas owners (terrain, particles, block overlay, first person, screen overlay)
//! one GPU texture, and `MC_SHARED_ATLAS=0` restores five private copies.
//!
//! Pixel identity of the shared texture against private copies is proven in
//! `mc-render/tests/x14_shared_atlas.rs`; `texture_animation.rs` shows animated sprites still reach
//! the screen through this install. This test pins the wiring: that `RenderStack::install` really
//! runs inside an active `AtlasShareScope` with the real 1.8.9 atlas.
//!
//! One test function on purpose: it sets a process environment variable between two installs, so
//! nothing else may run in this binary at the same time.

mod common;

use common::{config, gpu};
use mc_assets::AtlasAnimationUpload;
use mc_client::app::Client;
use mc_render::PhaseList;

/// Each full-mip owner's texture, captured through its animation sink with a record that rewrites
/// texel (0, 0) of level 0 with its current value (the owner's raw shadow is unchanged).
fn owner_textures(client: &Client) -> Vec<wgpu::Texture> {
    let render = client.render().expect("installed render stack");
    let probe = AtlasAnimationUpload {
        mip_level: 0,
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        rgba: client.assets.atlas.texture[..4].to_vec(),
    };
    let mut textures = Vec::new();
    let mut capture = |texture: &wgpu::Texture, _: bool| textures.push(texture.clone());
    assert!(render
        .terrain
        .borrow()
        .upload_atlas_animation_with_sink(&probe, &mut capture));
    assert!(render
        .particles
        .with(|phase| phase.upload_atlas_animation_with_sink(&probe, &mut capture)));
    assert!(render
        .block_overlay
        .with(|phase| phase.upload_atlas_animation_with_sink(&probe, &mut capture)));
    assert!(render
        .first_person
        .with(|phase| phase.upload_atlas_animation_with_sink(&probe, &mut capture)));
    assert!(render
        .overlay
        .with(|phase| phase.upload_block_atlas_animation_with_sink(&probe, &mut capture)));
    textures
}

fn installed_client(gpu: &mc_render::Gpu, dir: &std::path::Path) -> Client {
    let config = config(dir, "Dev").expect("this test needs the real 1.8.9 assets");
    let mut phases = PhaseList::new();
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    client.install(gpu, &mut phases, 64, 64);
    assert!(
        !client.assets.atlas.mipmaps.is_empty(),
        "the real atlas has a mip chain, so this checks full-mip sharing"
    );
    client
}

#[test]
fn the_client_install_shares_one_block_atlas_unless_switched_off() {
    let gpu = gpu(64, 64).expect("this test needs the real GPU");
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::remove_var(mc_render::atlas_transfer::SHARED_ATLAS_ENV);

    let shared = installed_client(&gpu, dir.path());
    let textures = owner_textures(&shared);
    assert_eq!(textures.len(), 5);
    for (owner, texture) in textures.iter().enumerate() {
        assert_eq!(
            *texture, textures[0],
            "owner {owner} must sample the shared atlas"
        );
    }
    drop(shared);

    std::env::set_var(mc_render::atlas_transfer::SHARED_ATLAS_ENV, "0");
    let private = installed_client(&gpu, dir.path());
    std::env::remove_var(mc_render::atlas_transfer::SHARED_ATLAS_ENV);
    let textures = owner_textures(&private);
    for (a, first) in textures.iter().enumerate() {
        for second in &textures[a + 1..] {
            assert_ne!(
                first, second,
                "MC_SHARED_ATLAS=0 keeps every owner's own copy"
            );
        }
    }
}
