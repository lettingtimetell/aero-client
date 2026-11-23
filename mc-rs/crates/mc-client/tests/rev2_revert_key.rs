//! A reload that reverts to the default packs while a server's resource
//! pack is active must publish the default packs' bake, not the bake earlier published for the
//! stack with the server's pack.
//!
//! java: `Minecraft.refreshResources` reloads with `defaultResourcePacks` alone when the stitch
//! fails; the server's pack (`resourcePackInstance`) stays set for the next refresh but is not in
//! that list. The reload keys its bake for the on-disk cache and for reusing a published model
//! set; that key stamped the active server pack, so the reverted bake was saved under, and
//! answered with, the set of a stack it is not: the published sprites (the server pack's) and
//! the atlas uploaded from the reverted bake (the default packs') no longer matched.

use std::path::Path;

use mc_client::app::assets::{activate_server_resource_pack, server_resource_pack_dir, GameAssets};
use mc_client::app::config::ClientConfig;
use mc_render::settings::GameSettings;

/// A server pack's cache key: 40 lowercase hex digits (its file name in the cache folder).
const HASH: &str = "0123456789abcdef0123456789abcdef01234567";
const ATLAS: u32 = 1024;

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A zip with stored (uncompressed) entries.
fn stored_zip(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = u32::try_from(out.len()).expect("small");
        let crc = crc32(data);
        let size = u32::try_from(data.len()).expect("small");
        let name_len = u16::try_from(name.len()).expect("short");
        let mut fields = Vec::new();
        fields.extend_from_slice(&20u16.to_le_bytes()); // version needed
        fields.extend_from_slice(&0u16.to_le_bytes()); // flags
        fields.extend_from_slice(&0u16.to_le_bytes()); // stored
        fields.extend_from_slice(&0u16.to_le_bytes()); // time
        fields.extend_from_slice(&0x21u16.to_le_bytes()); // date: 1980-01-01
        fields.extend_from_slice(&crc.to_le_bytes());
        fields.extend_from_slice(&size.to_le_bytes());
        fields.extend_from_slice(&size.to_le_bytes());
        fields.extend_from_slice(&name_len.to_le_bytes());
        fields.extend_from_slice(&0u16.to_le_bytes()); // extra
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&fields);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes()); // version made by
        central.extend_from_slice(&fields);
        central.extend_from_slice(&0u16.to_le_bytes()); // comment
        central.extend_from_slice(&0u16.to_le_bytes()); // disk
        central.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        central.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let central_offset = u32::try_from(out.len()).expect("small");
    let central_size = u32::try_from(central.len()).expect("small");
    let count = u16::try_from(entries.len()).expect("few");
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&central_size.to_le_bytes());
    out.extend_from_slice(&central_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

fn png(side: u32) -> Vec<u8> {
    let image = image::RgbaImage::from_pixel(side, side, image::Rgba([90, 90, 90, 255]));
    let mut bytes = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, image::ImageFormat::Png)
        .expect("encode");
    bytes.into_inner()
}

const MCMETA: &[u8] = br#"{"pack":{"pack_format":1,"description":"x"}}"#;

/// A folder pack whose dirt is twice the side the atlas may have: its stitch fails. (Not stone:
/// the server's pack sits above every user pack, so its stone would hide this one.)
fn huge_pack(packs: &Path) {
    let blocks = packs.join("Huge/assets/minecraft/textures/blocks");
    std::fs::create_dir_all(&blocks).expect("mkdir");
    std::fs::write(packs.join("Huge/pack.mcmeta"), MCMETA).expect("mcmeta");
    std::fs::write(blocks.join("dirt.png"), png(ATLAS * 2)).expect("dirt");
}

fn stone_width() -> u32 {
    mc_assets::sprite(&mc_assets::ResourceLocation::minecraft(
        "textures/blocks/stone",
    ))
    .width
}

#[test]
fn a_revert_with_a_server_pack_active_publishes_the_default_packs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = ClientConfig {
        game_dir: dir.path().join("game"),
        resource_pack_dir: dir.path().join("packs"),
        ..ClientConfig::default()
    }
    .with_discovered_assets();
    if !config.version_jar.as_ref().is_some_and(|jar| jar.is_file()) {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    }
    // The server's pack: stone at 32x32 (vanilla's is 16x16).
    let server_dir = server_resource_pack_dir(&config);
    std::fs::create_dir_all(&server_dir).expect("server pack folder");
    let zip = stored_zip(&[
        ("pack.mcmeta", MCMETA.to_vec()),
        ("assets/minecraft/textures/blocks/stone.png", png(32)),
    ]);
    std::fs::write(server_dir.join(HASH), zip).expect("server pack");
    activate_server_resource_pack(&config, HASH).expect("activate");
    huge_pack(&config.resource_pack_dir);

    let mut settings = GameSettings::default();
    let mut assets = GameAssets::load(&config, &settings, &mut |_, _| {}).expect("load");
    // The control: the stack with the server's pack, at this atlas size, is published.
    let control = assets
        .reload_packs_with_atlas_dimension(&config, &mut settings, ATLAS, &mut |_, _| {})
        .expect("the server pack fits");
    assert!(!control.reverted);
    assert_eq!(stone_width(), 32, "the server pack's stone is in");

    settings.resource_packs.push("Huge".to_owned());
    let outcome = assets
        .reload_packs_with_atlas_dimension(&config, &mut settings, ATLAS, &mut |_, _| {})
        .expect("a pack that cannot be stitched is recovered from");
    assert!(outcome.reverted);
    assert!(settings.resource_packs.is_empty());
    assert_eq!(
        stone_width(),
        16,
        "after the revert the default packs are loaded and uploaded: the published sprites \
         must be theirs, not the set published for the stack with the server's pack"
    );
}
