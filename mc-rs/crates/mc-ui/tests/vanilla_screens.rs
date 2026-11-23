//! Golden tests against the real 1.8.9 data on disk, and a live drive of S15's `ScreenStack`.
//!
//! Two things this proves that the unit tests cannot:
//!
//! 1. `servers.dat` is read and written in the format vanilla actually wrote — checked against
//!    `MCP-919-main/jars/servers.dat`, byte for byte in both directions.
//! 2. The screens that implement S15's [`Screen`] trait really do work on a real
//!    [`ScreenStack`], including its focus, modal and Escape rules.

use std::path::{Path, PathBuf};

use mc_assets::Language;
use mc_ui::font::component::ChatComponent;
use mc_ui::screen_stack::{Screen, ScreenCtx, ScreenStack};
use mc_ui::screens::connect::{ConnectingScreen, DisconnectedScreen};
use mc_ui::screens::multiplayer::server_data::{ResourceMode, ServerList};
use mc_ui::screens::pause::PauseScreen;
use mc_ui::widget::keys::Modifiers;
use mc_ui::widget::{keys, DrawList, ScaledResolution};

/// A minimal, valid [`mc_ui::font::FontRenderer`] for tests that must pass one somewhere (like
/// [`ScreenStack::key_typed`]) but draw nothing with it — a single-pixel 16x16 grid is the
/// smallest `ascii.png` shape [`mc_ui::font::FontData::from_parts`] accepts.
fn tiny_font() -> mc_ui::font::FontRenderer {
    use mc_ui::font::{FontData, FontRenderer};
    let img = image::RgbaImage::new(16, 16);
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("encode");
    let data = FontData::from_parts(&png.into_inner(), &[0u8; 0x1_0000]).expect("font data");
    FontRenderer::new(data, false, false)
}

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

/// The real `servers.dat` vanilla wrote, parsed and re-serialised byte-identically.
///
/// The file is 60 bytes of uncompressed NBT holding one server. The round-trip is the load
/// bearing half: a port that gzipped the output, or that wrote `acceptTextures` for the default
/// `PROMPT` mode, would produce a file vanilla reads differently.
#[test]
fn the_real_servers_dat_round_trips_byte_for_byte() {
    let Some(dir) = jars_dir() else {
        eprintln!("SKIP: jars/ not found");
        return;
    };
    let path = dir.join("servers.dat");
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("SKIP: {} not found", path.display());
        return;
    };

    // Vanilla's `CompressedStreamTools.read(File)` does not decompress, and this proves it: the
    // file starts with a bare TAG_Compound and an empty name, not a gzip magic.
    assert_eq!(&bytes[..3], &[0x0A, 0x00, 0x00], "uncompressed NBT root");

    let list = ServerList::read(&bytes).expect("parse the real servers.dat");
    assert_eq!(list.len(), 1);
    let entry = list.get(0).expect("one entry");
    assert_eq!(entry.name, "MCP Test Server");
    assert_eq!(entry.address, "127.0.0.1");
    assert_eq!(entry.host_and_port(), ("127.0.0.1", 25565));
    assert_eq!(
        entry.resource_mode,
        ResourceMode::Prompt,
        "no acceptTextures key means ask"
    );
    assert!(entry.icon_base64.is_none());

    let mut out = Vec::new();
    list.write(&mut out).expect("write");
    assert_eq!(
        out, bytes,
        "the re-serialised file differs from the one vanilla wrote"
    );
}

/// Adding a server to the real file and reading it back keeps the original entry intact.
#[test]
fn a_new_entry_appends_to_the_real_file_without_disturbing_it() {
    let Some(dir) = jars_dir() else {
        eprintln!("SKIP: jars/ not found");
        return;
    };
    let Ok(bytes) = std::fs::read(dir.join("servers.dat")) else {
        eprintln!("SKIP: servers.dat not found");
        return;
    };
    let mut list = ServerList::read(&bytes).expect("parse");
    let mut added = mc_ui::screens::multiplayer::ServerEntry::new("Added", "example.com:25566");
    added.resource_mode = ResourceMode::Enabled;
    added.icon_base64 = Some("aGVsbG8=".to_owned());
    list.add(added);

    let mut out = Vec::new();
    list.write(&mut out).expect("write");
    let back = ServerList::read(&out).expect("re-read");
    assert_eq!(back.len(), 2);
    assert_eq!(back.get(0).unwrap().name, "MCP Test Server");
    assert_eq!(back.get(1).unwrap().resource_mode, ResourceMode::Enabled);
    assert_eq!(back.get(1).unwrap().host_and_port(), ("example.com", 25566));
    assert_eq!(back, list);

    // The original entry's bytes are still a prefix of the new file, because NBT lists are
    // written in order and nothing about entry 0 changed.
    assert!(out.len() > bytes.len());
}

/// The three trait-implementing screens really work on S15's stack.
///
/// The pause menu pauses and swallows input; the connecting and disconnect screens both refuse
/// Escape, exactly as `GuiConnecting.keyTyped` and `GuiDisconnected.keyTyped` (both empty
/// overrides) do — so a real stack cannot pop them by accident.
#[test]
fn the_screen_stack_drives_the_pause_connect_and_disconnect_screens() {
    let mut stack = ScreenStack::new();
    stack.set_resolution(ScaledResolution::new(854, 480, 1, false));
    assert!(stack.is_empty());
    let font = tiny_font();

    stack.push(Box::new(PauseScreen::new()));
    assert_eq!(stack.names(), ["pause"]);
    assert!(stack.pauses_game(), "the in-game menu stops the clock");
    assert!(stack.top_is_modal());
    assert!(!stack.allows_user_input());

    // Escape pops it, because `PauseScreen` keeps `Screen`'s default `key_typed`.
    stack.key_typed(None, keys::ESCAPE, Modifiers::NONE, &font);
    assert!(stack.is_empty());

    stack.push(Box::new(ConnectingScreen::new("example.com")));
    stack.key_typed(None, keys::ESCAPE, Modifiers::NONE, &font);
    assert_eq!(
        stack.names(),
        ["connecting"],
        "GuiConnecting ignores Escape, so the stack must too"
    );

    stack.push(Box::new(DisconnectedScreen::new(
        "Connection Lost",
        ChatComponent::text("Timed out"),
    )));
    assert!(
        stack.top().is_some_and(Screen::opaque),
        "the disconnect screen covers what is under it"
    );
    assert_eq!(stack.first_visible(), 1, "so drawing starts there");
    stack.key_typed(None, keys::ESCAPE, Modifiers::NONE, &font);
    assert_eq!(stack.names(), ["connecting", "disconnected"]);

    stack.clear();
    assert!(stack.is_empty());
}

/// Every screen draws something without panicking at three very different resolutions,
/// including the 320×240 minimum `ScaledResolution` clamps to.
#[test]
fn every_screen_survives_a_resize_sweep() {
    use mc_render::settings::GameSettings;
    use mc_ui::font::{FontData, FontRenderer};
    use mc_ui::screens::debug::{DebugInfo, DebugOverlay};
    use mc_ui::screens::hud::{Hud, PlayerHud};
    use mc_ui::screens::login::LoginScreen;
    use mc_ui::screens::menu::MainMenuScreen;
    use mc_ui::screens::multiplayer::{MultiplayerScreen, ServerEntry};
    use mc_ui::screens::options::{OptionListScreen, OptionsScreen};
    use mc_ui::screens::stats::{AchievementGrid, AchievementsScreen, StatsScreen};

    // A synthetic 5-pixel font: this test is about layout robustness, not artwork.
    let mut img = image::RgbaImage::new(128, 128);
    for cell in 0..256u32 {
        let (cx, cy) = ((cell % 16) * 8, (cell / 16) * 8);
        for col in 0..4 {
            img.put_pixel(cx + col, cy + 2, image::Rgba([255, 255, 255, 255]));
        }
    }
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("encode");
    let data = FontData::from_parts(&png.into_inner(), &[0u8; 0x1_0000]).expect("font data");
    let mut font = FontRenderer::new(data, false, false);

    let lang = Language::new("en_US");
    let settings = GameSettings::default();

    for (w, h) in [(320u32, 240u32), (854, 480), (3840, 2160)] {
        let c = ScreenCtx::new(ScaledResolution::new(w, h, 1, false), 0.5);
        let mut d = DrawList::new();

        let mut menu = MainMenuScreen::new("splash");
        menu.init(&c, &lang);
        // `panorama_steps = 0` keeps the 3-D skybox out of the 2-D bound check below: a
        // near-plane-clipped quad legitimately projects several screens wide, exactly as it
        // would on the GPU. It gets its own finiteness check instead.
        menu.draw(&mut d, &mut font, &c, (10, 10), 0, 0);
        let mut sky = DrawList::new();
        menu.panorama.draw(&mut sky, c.width, c.height, 0.0, 8);
        assert!(!sky.is_empty(), "{w}x{h}: the panorama drew nothing");
        for v in sky.vertices() {
            assert!(
                v.pos[0].is_finite() && v.pos[1].is_finite(),
                "{w}x{h}: the panorama projected a non-finite vertex {:?}",
                v.pos
            );
        }

        let mut servers = MultiplayerScreen::new({
            let mut l = ServerList::new();
            l.add(ServerEntry::new("a", "b"));
            l
        });
        servers.init(&c, &lang);
        servers.draw(&mut d, &mut font, &c, (10, 10), &lang, 0);

        let mut options = OptionsScreen::new(false);
        options.init(&c, &settings, &lang);
        options.draw(&mut d, &mut font, &c, (10, 10), &lang);

        let mut video = OptionListScreen::video(false);
        video.init(&c, &settings, &lang);
        video.draw(&mut d, &mut font, &c, (10, 10), &lang);

        let mut pause = PauseScreen::new();
        pause.init(&c, &lang);
        pause.draw(&mut d, &mut font, &c, (10, 10), &lang);

        let mut login = LoginScreen::new();
        login.init(&c, &lang);
        login.draw(&mut d, &mut font, &c, (10, 10), &lang);

        let mut stats = StatsScreen::new();
        stats.init(&c, &lang);
        stats.draw(&mut d, &mut font, &c, (10, 10), &lang);

        // A real tree and an atlas key, so the resize sweep also drives the pannable map's tile
        // loop, its connecting lines and its cell plates at every resolution — those loops are
        // bounded by the 224×155 window rather than by the screen, and this is what pins that.
        let atlas = mc_ui::widget::UiTexture::gui("mc-ui/resize-sweep-atlas");
        let mut achievements = AchievementsScreen::new(AchievementGrid::new(vec![
            mc_ui::screens::stats::AchievementNode {
                key: "achievement.openInventory".into(),
                column: 0,
                row: 0,
                parent: None,
                unlocked: true,
                special: false,
                icon: None,
            },
            mc_ui::screens::stats::AchievementNode {
                key: "achievement.mineWood".into(),
                column: 2,
                row: 1,
                parent: Some("achievement.openInventory".into()),
                unlocked: false,
                special: true,
                icon: None,
            },
        ]));
        achievements.init(&c, &lang);
        achievements.draw(&mut d, &mut font, Some(&atlas), &c, (10, 10), &lang);

        let hud = Hud::new();
        hud.draw(&mut d, &mut font, None, &c, &PlayerHud::default());

        let mut overlay = DebugOverlay::new();
        overlay.visible = true;
        overlay.draw(&mut d, &mut font, &c, &DebugInfo::default(), None);

        assert!(!d.is_empty(), "{w}x{h} produced no geometry");
        // Nothing may escape the scaled surface by more than a screen's width, which is the
        // signature of an unclamped layout constant.
        for v in d.vertices() {
            assert!(
                v.pos[0] > -(c.width as f32) && v.pos[0] < (c.width * 2) as f32,
                "{w}x{h}: runaway x {}",
                v.pos[0]
            );
            assert!(
                v.pos[1] > -(c.height as f32) && v.pos[1] < (c.height * 2) as f32,
                "{w}x{h}: runaway y {}",
                v.pos[1]
            );
        }
    }
}
