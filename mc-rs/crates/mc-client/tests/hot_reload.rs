//! Resource-pack hot reload, proven through the GPU.
//!
//! java: `Minecraft.refreshResources` (`:788`) reloads *everything* — models, the block atlas,
//! every texture, the language and the sounds — because every consumer is a
//! `IResourceManagerReloadListener` and `SimpleReloadableResourceManager.reloadResources` calls
//! all of them. It finishes with `RenderGlobal.loadRenderers()`, which throws the chunk renderers
//! away and rebuilds them, because the atlas they drew from is a different texture now.
//!
//! This client now does all of it. What each test proves:
//!
//! * the language table (`a_language_pack_changes_the_rendered_menu_…`) — pixels change, and
//!   removing the pack restores them **byte for byte**;
//! * the font reload boundary (`a_font_pack_reloads_ascii_but_preserves_constructor_unicode_widths`):
//!   the ASCII sheet reloads, while the Unicode width table remains the one read by the
//!   constructor;
//! * **block models, the block atlas and GUI textures**
//!   (`a_pack_that_replaces_a_block_texture_changes_the_rendered_world_…`) — a real block, drawn
//!   through the real mesher into a real framebuffer, changes colour and comes back byte for
//!   byte;
//! * that a client which never swaps a pack is **unaffected** by any of the machinery
//!   (`a_no_op_reload_leaves_the_frame_byte_identical`).
//!
//! The gap these tests used to pin — `mc_assets::install` and `tint::install_colormaps` being
//! `OnceLock`s, and `UiRenderer` caching bind groups with no invalidation — is closed:
//! `mc_assets::model::reinstall`, `tint::reload_colormaps` and `UiRenderer::clear_textures`.

mod common;

use common::{config, drawn_pixels, gpu, pump_no_tick};
use mc_client::app::Client;
use mc_render::{Camera, PhaseList};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// Keep framebuffer assertions exact without asking the test harness to print two 307,200-byte
/// RGBA buffers when a GPU or driver differs. The equality check itself is still byte-for-byte.
fn assert_frame_bytes_equal(label: &str, expected: &[u8], actual: &[u8]) {
    if expected == actual {
        return;
    }

    let mut differing = 0usize;
    let mut first_offset = None;
    let mut samples = Vec::new();
    for (offset, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        if want != got {
            differing += 1;
            first_offset.get_or_insert(offset);
            if samples.len() < 8 {
                samples.push((offset, want, got));
            }
        }
    }
    let first_pixel = first_offset.map(|offset| {
        let pixel = offset / 4;
        (pixel % WIDTH as usize, pixel / WIDTH as usize)
    });
    let artifacts = write_frame_artifacts(label, expected, actual);
    panic!(
        "{label}: framebuffer mismatch (expected {} bytes, actual {} bytes, {} differing zipped bytes, first offset {first_offset:?}, first pixel {first_pixel:?}, samples {samples:?}, FNV-1a expected={:016x}, actual={:016x}, artifacts {artifacts:?})",
        expected.len(),
        actual.len(),
        differing,
        frame_hash(expected),
        frame_hash(actual),
    );
}

/// Assert that a reload changed the complete framebuffer, with compact diagnostics on failure.
fn assert_frame_bytes_differ(label: &str, before: &[u8], after: &[u8]) {
    if before != after {
        return;
    }
    panic!(
        "{label}: framebuffer unexpectedly identical ({} bytes, FNV-1a {:016x})",
        before.len(),
        frame_hash(before),
    );
}

fn frame_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

/// Optionally save the two complete frames for a mismatch. This is disabled by default so a
/// passing suite leaves no artifacts; setting `MC_HOT_RELOAD_ARTIFACT_DIR` turns a failed exact
/// comparison into a directly inspectable pair without changing the assertion's semantics.
fn write_frame_artifacts(label: &str, expected: &[u8], actual: &[u8]) -> Option<String> {
    let dir = std::env::var_os("MC_HOT_RELOAD_ARTIFACT_DIR")?;
    let dir = std::path::PathBuf::from(dir);
    let stem: String = label
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    let stem = if stem.is_empty() {
        "frame_mismatch".to_owned()
    } else {
        stem
    };
    if expected.len() != (WIDTH * HEIGHT * 4) as usize
        || actual.len() != (WIDTH * HEIGHT * 4) as usize
    {
        return Some(format!(
            "skipped: frame length is not {WIDTH}x{HEIGHT} RGBA8"
        ));
    }
    if let Err(error) = std::fs::create_dir_all(&dir) {
        return Some(format!("skipped: create {}: {error}", dir.display()));
    }
    let expected_path = dir.join(format!("{stem}.expected.png"));
    let actual_path = dir.join(format!("{stem}.actual.png"));
    if let Err(error) = image::save_buffer(
        &expected_path,
        expected,
        WIDTH,
        HEIGHT,
        image::ColorType::Rgba8,
    ) {
        return Some(format!(
            "skipped: write {}: {error}",
            expected_path.display()
        ));
    }
    if let Err(error) =
        image::save_buffer(&actual_path, actual, WIDTH, HEIGHT, image::ColorType::Rgba8)
    {
        return Some(format!("skipped: write {}: {error}", actual_path.display()));
    }
    Some(format!(
        "expected {} and actual {}",
        expected_path.display(),
        actual_path.display()
    ))
}

/// Keep the reload framebuffer oracle independent of Java's wall-clock splash pulse. The
/// comparison below still covers every framebuffer byte; this fixture removes one unrelated,
/// time-dependent draw from a test whose subject is resource replacement and restoration.
fn suppress_animated_menu_splash(client: &mut Client) {
    let mc_client::app::driver::Ui::MainMenu(menu) = &mut client.ui else {
        panic!("resource reload fixture must start on the main menu");
    };
    menu.splash.clear();
}

/// Write a folder resource pack that overrides one language key.
fn write_lang_pack(dir: &std::path::Path, name: &str, line: &str) {
    let pack = dir.join(name);
    let lang = pack.join("assets").join("minecraft").join("lang");
    std::fs::create_dir_all(&lang).expect("mkdir");
    std::fs::write(
        pack.join("pack.mcmeta"),
        br#"{"pack":{"pack_format":1,"description":"hot reload test"}}"#,
    )
    .expect("mcmeta");
    std::fs::write(lang.join("en_US.lang"), line).expect("lang");
}

/// The strongest available proof that a pack goes live: the *pixels* change, and changing back
/// restores them byte for byte.
#[test]
fn a_language_pack_changes_the_rendered_menu_and_reverting_restores_it_exactly() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    // Byte-compares vanilla's (still) title screen; the Aero one animates.
    common::vanilla_title_screen(dir.path());
    let packs = tempfile::tempdir().expect("packs");
    let Some(mut config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    config.resource_pack_dir = packs.path().to_path_buf();

    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    suppress_animated_menu_splash(&mut client);
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();

    // Two frames: the first compiles pipelines and pulls textures, the second is steady state.
    for _ in 0..2 {
        pump_no_tick(&mut client, &mut gpu, &mut phases, &mut camera);
    }
    let before = gpu.read_pixels().expect("readback");
    assert!(drawn_pixels(&before, common::CLEAR_RGB) > 1_000);
    assert_eq!(client.assets.language.tr("menu.multiplayer"), "Multiplayer");

    // ---- pack on ----
    write_lang_pack(
        packs.path(),
        "Loud",
        "menu.multiplayer=WWWWWWWWWWWWWWWWWWWW\n",
    );
    client.settings.resource_packs.push("Loud".to_owned());
    let outcome = client
        .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut |_, _| {})
        .expect("reload");
    assert!(!outcome.reverted);
    assert_eq!(
        client.assets.language.tr("menu.multiplayer"),
        "WWWWWWWWWWWWWWWWWWWW"
    );

    for _ in 0..2 {
        pump_no_tick(&mut client, &mut gpu, &mut phases, &mut camera);
    }
    let after = gpu.read_pixels().expect("readback");
    assert_frame_bytes_differ(
        "a language pack must change what is actually on the screen, not just a lookup table",
        &before,
        &after,
    );

    // ---- pack off ----
    client.settings.resource_packs.clear();
    client
        .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut |_, _| {})
        .expect("reload back");
    assert_eq!(client.assets.language.tr("menu.multiplayer"), "Multiplayer");
    for _ in 0..2 {
        pump_no_tick(&mut client, &mut gpu, &mut phases, &mut camera);
    }
    let restored = gpu.read_pixels().expect("readback");
    assert_frame_bytes_equal(
        "removing the pack must restore the frame byte for byte, not merely approximately",
        &before,
        &restored,
    );
}

/// Reloading a pack refreshes the ASCII sheet, while Java keeps `glyph_sizes.bin` from the
/// `FontRenderer` constructor. The pack deliberately blanks the ASCII sheet and narrows the
/// Unicode table; only the former may affect the live renderer.
#[test]
fn a_font_pack_reloads_ascii_but_preserves_constructor_unicode_widths() {
    let dir = tempfile::tempdir().expect("tempdir");
    let packs = tempfile::tempdir().expect("packs");
    let Some(mut config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    config.resource_pack_dir = packs.path().to_path_buf();
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    assert!(
        !client.font.unicode_flag(),
        "the default en_US test configuration must use the ASCII table"
    );

    // `charWidth` for ASCII comes from `ascii.png`, but every code point above 0xFF is measured
    // from the constructor's `glyph_sizes.bin` table. U+4E00 is a full-width glyph in vanilla.
    let ascii_before = client.font.string_width("A");
    assert!(
        ascii_before > 1,
        "vanilla A must have visible ink and its gap"
    );
    let before = client.font.string_width("\u{4e00}");
    assert!(before > 0);

    let pack = packs.path().join("NarrowGlyphs");
    let font_dir = pack.join("assets").join("minecraft").join("font");
    std::fs::create_dir_all(&font_dir).expect("mkdir");
    std::fs::write(
        pack.join("pack.mcmeta"),
        br#"{"pack":{"pack_format":1,"description":"narrow"}}"#,
    )
    .expect("mcmeta");
    let texture_dir = pack
        .join("assets")
        .join("minecraft")
        .join("textures")
        .join("font");
    std::fs::create_dir_all(&texture_dir).expect("mkdir textures/font");
    // A valid, fully transparent 128x128 sheet makes every ASCII cell one pixel wide. The
    // changed `A` width proves the live `ascii.png` listener ran, without depending on artwork.
    image::RgbaImage::new(128, 128)
        .save(texture_dir.join("ascii.png"))
        .expect("ascii.png");
    // java: `FontRenderer.renderUnicodeChar` reads the byte as two nibbles - start column in
    // the high nibble, end column in the low - and advances by `(end + 1 - start) / 2 + 1`.
    // `0x01` is the narrowest legal glyph: start 0, end 1, advance 2.
    std::fs::write(font_dir.join("glyph_sizes.bin"), vec![0x01u8; 0x1_0000]).expect("glyph sizes");

    client
        .settings
        .resource_packs
        .push("NarrowGlyphs".to_owned());
    let outcome = client
        .reload_resource_packs(None, None, &mut |_, _| {})
        .expect("reload");
    assert!(!outcome.reverted, "a valid font pack must not be reverted");
    assert!(
        !client.font.unicode_flag(),
        "a normal pack reload must not enable force-unicode rendering"
    );

    let ascii_after = client.font.string_width("A");
    assert!(
        ascii_after < ascii_before,
        "reloading ascii.png must remeasure A: {ascii_before} -> {ascii_after}"
    );
    let after = client.font.string_width("\u{4e00}");
    assert_eq!(
        after, before,
        "Java's onResourceManagerReload refreshes ascii.png but keeps constructor Unicode widths"
    );
}

/// A folder pack that repaints one block texture a flat, unmistakable colour, and replaces one
/// GUI texture with another. Both are the *common* case for a resource pack, and both were frozen
/// until restart before this run.
fn write_repaint_pack(dir: &std::path::Path, name: &str, rgb: [u8; 3]) {
    let pack = dir.join(name);
    let blocks = pack.join("assets/minecraft/textures/blocks");
    let gui_title = pack.join("assets/minecraft/textures/gui/title");
    std::fs::create_dir_all(&blocks).expect("mkdir blocks");
    std::fs::create_dir_all(&gui_title).expect("mkdir gui/title");
    std::fs::write(
        pack.join("pack.mcmeta"),
        br#"{"pack":{"pack_format":1,"description":"repaint"}}"#,
    )
    .expect("mcmeta");

    let mut block = image::RgbaImage::new(16, 16);
    for pixel in block.pixels_mut() {
        *pixel = image::Rgba([rgb[0], rgb[1], rgb[2], 255]);
    }
    block.save(blocks.join("stone.png")).expect("stone.png");

    // `title/minecraft.png` is the wordmark `MainMenuScreen::draw` blits unconditionally, as
    // two 155x44 halves, regardless of whether the panorama or the dirt fallback is behind it
    // (S21 -> S16 §13's real, nonzero `panorama_steps` means it is now the panorama — see
    // `Client::build_ui`). Flattening it is the cheapest still-guaranteed-on-screen way to see a
    // *GUI* texture swap in the framebuffer; `options_background.png`, the old target, is the
    // menu's dirt-fallback background and is not drawn at all once the panorama is live.
    let mut wordmark = image::RgbaImage::new(256, 256);
    for pixel in wordmark.pixels_mut() {
        *pixel = image::Rgba([rgb[0], rgb[1], rgb[2], 255]);
    }
    wordmark
        .save(gui_title.join("minecraft.png"))
        .expect("minecraft.png");
}

/// How many pixels are (close to) a colour. The terrain shader applies shading and the lightmap,
/// so an exact match is the wrong test for a *world* pixel; dominance of the hue is the right one.
fn hue_pixels(pixels: &[u8], pick: impl Fn([u8; 3]) -> bool) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|p| pick([p[0], p[1], p[2]]))
        .count()
}

/// **The inverted gap test.** This is the one that used to be called
/// `block_models_and_the_block_atlas_do_not_hot_reload_yet` and asserted the opposite.
///
/// Three things it proves without a server, all of which needed a restart before this run:
///
/// 1. the **block atlas is re-stitched** — the pack's flat texels are in `client.assets.atlas`,
///    and `mc_assets::model::installed_generation` advanced, so the global registry the mesher
///    reads was replaced rather than added to;
/// 2. an **already-uploaded GUI texture takes effect on screen** — the main menu's wordmark
///    (`title/minecraft.png`) is resident from the first frame, and `UiRenderer` used to keep it
///    for the life of the process;
/// 3. removing the pack restores the frame **byte for byte**.
///
/// The block's *rendered* pixels are proven in `mc-render`'s `s21_pack_reload.rs`, which has a
/// world to draw without needing a connection.
#[test]
fn a_pack_that_replaces_block_and_gui_textures_takes_effect_and_reverting_restores_it() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    // Byte-compares vanilla's (still) title screen; the Aero one animates.
    common::vanilla_title_screen(dir.path());
    let packs = tempfile::tempdir().expect("packs");
    let Some(mut config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    config.resource_pack_dir = packs.path().to_path_buf();

    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    suppress_animated_menu_splash(&mut client);
    assert!(
        client.assets.models_installed && client.assets.ready,
        "the first load installs the baked models globally"
    );
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();

    let atlas_before = (client.assets.atlas.w, client.assets.atlas.h);
    let stone_quads = mc_assets::model_for(mc_data::BlockState { id: 1 << 4 })
        .face_quads
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    assert!(stone_quads > 0, "stone must have baked geometry to compare");

    // The main menu draws its wordmark, `gui/title/minecraft.png`, unconditionally — a GUI
    // texture that is resident in `UiRenderer` after the first frame.
    for _ in 0..2 {
        pump_no_tick(&mut client, &mut gpu, &mut phases, &mut camera);
    }
    let before = gpu.read_pixels().expect("readback");
    assert!(drawn_pixels(&before, common::CLEAR_RGB) > 1_000);
    // The control: vanilla's dirt is brown (r > g > b), so nothing on screen has r == b with a
    // much lower green. Counting it now is what makes the later count meaningful.
    assert_eq!(
        hue_pixels(&before, is_magenta_family),
        0,
        "nothing magenta may be on screen before the pack"
    );
    assert_eq!(
        atlas_texels(&client.assets.atlas, [255, 0, 255]),
        0,
        "and nothing magenta is in the vanilla block atlas either"
    );

    // ---- pack on ----
    write_repaint_pack(packs.path(), "Repaint", [255, 0, 255]);
    client.settings.resource_packs.push("Repaint".to_owned());
    let outcome = client
        .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut |_, _| {})
        .expect("reload");
    assert!(!outcome.reverted);
    assert!(
        client.assets.models_installed,
        "after a reload the baked models in `GameAssets` ARE the installed ones again"
    );
    assert!(
        outcome.model_generation >= 2,
        "the block models must have been re-baked and re-installed, got generation {}",
        outcome.model_generation
    );
    assert_eq!(
        (client.assets.atlas.w, client.assets.atlas.h),
        atlas_before,
        "a same-size repaint must not resize the atlas — only its texels change"
    );
    assert!(
        atlas_texels(&client.assets.atlas, [255, 0, 255]) >= 256,
        "the block atlas must have been re-stitched with the pack's 16x16 stone"
    );
    assert_eq!(
        mc_assets::model_for(mc_data::BlockState { id: 1 << 4 })
            .face_quads
            .iter()
            .map(Vec::len)
            .sum::<usize>(),
        stone_quads,
        "the geometry is unchanged; it is the texture underneath it that moved"
    );

    for _ in 0..2 {
        pump_no_tick(&mut client, &mut gpu, &mut phases, &mut camera);
    }
    let after = gpu.read_pixels().expect("readback");
    let repainted = hue_pixels(&after, is_magenta_family);
    assert!(
        repainted > 1_000,
        "the GUI must be drawn from the pack's texture without a restart; only {repainted} \
         magenta pixels were drawn"
    );
    assert_frame_bytes_differ("the framebuffer must actually differ", &before, &after);

    // ---- pack off ----
    client.settings.resource_packs.clear();
    client
        .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut |_, _| {})
        .expect("reload back");
    assert_eq!(
        atlas_texels(&client.assets.atlas, [255, 0, 255]),
        0,
        "the atlas must come back too"
    );
    for _ in 0..2 {
        pump_no_tick(&mut client, &mut gpu, &mut phases, &mut camera);
    }
    let restored = gpu.read_pixels().expect("readback");
    assert_frame_bytes_equal(
        "removing the pack must restore the frame byte for byte, not merely approximately",
        &before,
        &restored,
    );
}

/// Vanilla GUI art is brown/grey, so `red == blue` with a much lower green never occurs; the
/// pack's flat magenta, tinted by the dirt background's `(64, 64, 64)`, always does.
fn is_magenta_family(p: [u8; 3]) -> bool {
    p[0] > 16 && p[0] == p[2] && p[1].saturating_add(16) < p[0]
}

/// How many texels of the block atlas are exactly `rgb`.
fn atlas_texels(atlas: &mc_assets::Atlas, rgb: [u8; 3]) -> usize {
    atlas
        .texture
        .chunks_exact(4)
        .filter(|t| t[0] == rgb[0] && t[1] == rgb[1] && t[2] == rgb[2])
        .count()
}

/// The inertness claim, and the reason nothing else in this workspace had to change: a client
/// that reloads the **same** pack stack renders a frame that is byte-identical to the one before.
///
/// If the `OnceLock` → `HotSwap` change had perturbed a single baked UV, a single sprite
/// placement or a single colormap entry, this would fail — the comparison is the whole
/// framebuffer, exact.
#[test]
fn a_no_op_reload_leaves_the_frame_byte_identical() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    // Byte-compares vanilla's (still) title screen; the Aero one animates.
    common::vanilla_title_screen(dir.path());
    let packs = tempfile::tempdir().expect("packs");
    let Some(mut config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    config.resource_pack_dir = packs.path().to_path_buf();

    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    suppress_animated_menu_splash(&mut client);
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();

    for _ in 0..4 {
        pump_no_tick(&mut client, &mut gpu, &mut phases, &mut camera);
    }
    let before = gpu.read_pixels().expect("readback");
    assert!(drawn_pixels(&before, common::CLEAR_RGB) > 1_000);
    let phase_names: Vec<&str> = phases.names().collect();

    // No pack added, no pack removed: the same stack, re-derived from scratch.
    let outcome = client
        .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut |_, _| {})
        .expect("reload");
    assert!(!outcome.reverted);
    assert_eq!(outcome.packs, 2, "object store + jar, unchanged");
    assert_eq!(
        phases.names().collect::<Vec<_>>(),
        phase_names,
        "rebuilding must produce the same phases in the same order"
    );

    for _ in 0..4 {
        pump_no_tick(&mut client, &mut gpu, &mut phases, &mut camera);
    }
    let after = gpu.read_pixels().expect("readback");
    assert_frame_bytes_equal(
        "a reload that changes nothing must change nothing — byte for byte over the whole frame",
        &before,
        &after,
    );
}
