//! Headless GPU tests for S17's `Screen`-trait container GUIs.
//!
//! Same technique as `tests/container_render.rs` and S15's `tests/headless_ui.rs`: render
//! through the real `UI 800` phase into an offscreen target with `Gpu::headless`, read the
//! framebuffer back, and compare it against the **decoded source PNG from the vanilla
//! `1.8.9.jar`**. Every assertion below is therefore a statement about actual pixels, not about
//! the `DrawList` that produced them.
//!
//! The scaled resolution is pinned to `guiScale = 1` throughout, so one GUI pixel is one
//! framebuffer pixel and vanilla's coordinates can be quoted directly.
//!
//! Skips (loudly) with no wgpu adapter or no vanilla jar, exactly like its two siblings.

use std::path::{Path, PathBuf};

use image::RgbaImage;
use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::{Screen, ScreenCtx};
use mc_ui::screens::container::creative::{CreativeTab, TABS};
use mc_ui::screens::container::creative_screen::{tab_plate, CreativeScreen, Y_SIZE};
use mc_ui::screens::container::screen::{
    AnvilArt, BeaconArt, BrewingStandArt, ContainerArt, ContainerScreen, EnchantingArt, FurnaceArt,
};
use mc_ui::screens::container::{tooltip_box, ItemStack};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, UiTexture};

/// Big enough for the widest panel (the beacon's 230) and the tallest (the double chest's 222,
/// and the creative screen's tab strip, which sticks out above and below the panel).
const W: u32 = 256;
const H: u32 = 256;

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
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

fn assets() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

/// The GUI texture behind a [`UiTexture`], decoded straight out of the jar — the ground truth
/// every pixel assertion below compares against.
fn source_png(rm: &ResourceManager, texture: &UiTexture) -> RgbaImage {
    let location = match texture {
        UiTexture::Resource(l) => l.clone(),
        other => panic!("{other:?} has no source PNG"),
    };
    let bytes = rm.load(&location).expect("texture is in the vanilla jar");
    image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .expect("decodes as PNG")
        .to_rgba8()
}

fn container_png(rm: &ResourceManager, name: &str) -> RgbaImage {
    source_png(
        rm,
        &UiTexture::Resource(ResourceLocation::minecraft(format!(
            "textures/gui/container/{name}"
        ))),
    )
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

fn resolution() -> ScaledResolution {
    let r = ScaledResolution::new(W, H, 1, false);
    assert_eq!(r.scale_factor(), 1);
    r
}

/// Render one `DrawList`, loading whatever textures it names out of the jar, and hand back the
/// framebuffer.
fn render(gpu: &mut Gpu, rm: &ResourceManager, list: DrawList) -> Vec<u8> {
    let mut phase = UiPhase::new(gpu, W, H, 1, false);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(
        missing.is_empty(),
        "textures missing from the jar: {missing:?}"
    );
    phase.set_resolution(resolution());
    phase.submit(list);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(resolution(), 0.0)
}

fn stack(name: &str, count: i32) -> ItemStack {
    let id = mc_data::item_by_name(name).expect("known item");
    ItemStack {
        item_id: id.0 as i16,
        count,
        damage: 0,
        nbt: None,
    }
}

/// A rectangle of screen pixels that some later draw call covers, so the panel underneath it
/// cannot be compared against the sheet. Window-relative, `(x, y, width, height)`.
type Excluded = (i32, i32, i32, i32);

/// Compare a rendered rectangle against the same rectangle of a source sheet, texel for texel.
///
/// Transparent source texels are skipped (the framebuffer shows whatever was under them);
/// every fully opaque one has to match **exactly**. Returns how many texels were checked, so a
/// caller can assert the sample was not vacuously empty.
#[allow(clippy::too_many_arguments)]
fn assert_texels_match_except(
    px: &[u8],
    source: &RgbaImage,
    screen_x: i32,
    screen_y: i32,
    u: i32,
    v: i32,
    width: i32,
    height: i32,
    exclude: &[Excluded],
    what: &str,
) -> usize {
    let mut checked = 0;
    for dy in 0..height {
        for dx in 0..width {
            let (sx, sy) = (u + dx, v + dy);
            if sx < 0 || sy < 0 || sx as u32 >= source.width() || sy as u32 >= source.height() {
                continue;
            }
            let src = source.get_pixel(sx as u32, sy as u32).0;
            if src[3] != 255 {
                continue;
            }
            let (fx, fy) = (screen_x + dx, screen_y + dy);
            if fx < 0 || fy < 0 || fx as u32 >= W || fy as u32 >= H {
                continue;
            }
            if exclude
                .iter()
                .any(|(ex, ey, ew, eh)| fx >= *ex && fx < ex + ew && fy >= *ey && fy < ey + eh)
            {
                continue;
            }
            let got = pixel(px, fx as u32, fy as u32);
            assert_eq!(
                [got[0], got[1], got[2]],
                [src[0], src[1], src[2]],
                "{what}: framebuffer ({fx}, {fy}) should be texel ({sx}, {sy})"
            );
            checked += 1;
        }
    }
    checked
}

/// [`assert_texels_match_except`] with nothing excluded.
#[allow(clippy::too_many_arguments)]
fn assert_texels_match(
    px: &[u8],
    source: &RgbaImage,
    screen_x: i32,
    screen_y: i32,
    u: i32,
    v: i32,
    width: i32,
    height: i32,
    what: &str,
) -> usize {
    assert_texels_match_except(
        px,
        source,
        screen_x,
        screen_y,
        u,
        v,
        width,
        height,
        &[],
        what,
    )
}

// ---------------------------------------------------------------- panel art

/// The load-bearing one: every container kind's background layer has to reach the framebuffer as
/// the exact texels vanilla's own `textures/gui/container/*.png` holds, at the exact offsets
/// `drawGuiContainerBackgroundLayer` blits them from.
#[test]
fn every_container_panel_lands_texel_for_texel_on_the_vanilla_art() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };

    // The excluded rectangles are the *dynamic* art each kind paints over its own panel even in
    // its resting state — vanilla draws them too, so they are not gaps, just regions where the
    // panel is legitimately no longer visible. Each has its own dedicated test below.
    let cases: Vec<(&str, ContainerScreen, Vec<Excluded>)> = vec![
        ("inventory.png", ContainerScreen::player(0), vec![]),
        (
            "hopper.png",
            ContainerScreen::hopper(1, 5, "Hopper"),
            vec![],
        ),
        (
            "dispenser.png",
            ContainerScreen::dispenser(1, "Dispenser"),
            vec![],
        ),
        (
            "furnace.png",
            ContainerScreen::furnace(1, "Furnace"),
            // The progress arrow always blits `l + 1` columns, so an idle furnace still paints a
            // single 1x16 column. java: `drawTexturedModalRect(i + 79, j + 34, 176, 14, l+1, 16)`.
            vec![(79, 34, 1, 16)],
        ),
        (
            "brewing_stand.png",
            ContainerScreen::brewing_stand(1, "Brewing Stand"),
            vec![],
        ),
        (
            "enchanting_table.png",
            ContainerScreen::enchanting_table(1, "Enchant"),
            // The three galley plates, drawn over the panel whether or not an offer exists.
            vec![(60, 14, 108, 57)],
        ),
        (
            "anvil.png",
            ContainerScreen::anvil(1, "Repair & Name"),
            // The rename box, always drawn from the strip below the panel.
            vec![(59, 20, 110, 16)],
        ),
    ];

    for (name, screen, exclude) in cases {
        let mut list = DrawList::new();
        screen
            .art
            .draw_background(&mut list, 0, 0, &screen.container);
        let px = render(&mut gpu, &rm, list);
        let source = container_png(&rm, name);
        let checked = assert_texels_match_except(
            &px,
            &source,
            0,
            0,
            0,
            0,
            screen.art.x_size(),
            screen.art.y_size(),
            &exclude,
            name,
        );
        assert!(
            checked > 10_000,
            "{name}: only {checked} opaque texels checked, expected a full panel"
        );
    }
}

/// `GuiChest` is the one kind whose panel is **two** blits, and the seam between them moves with
/// the row count: the container half is `rows * 18 + 17` tall from `v = 0`, and the player half
/// is 96 tall from `v = 126`. Getting the seam wrong shifts the player's inventory art.
#[test]
fn the_chest_panel_joins_its_two_halves_at_the_row_dependent_seam() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let source = container_png(&rm, "generic_54.png");

    for rows in [1u8, 3, 6] {
        let screen = ContainerScreen::chest(1, rows, "Chest");
        let mut list = DrawList::new();
        screen
            .art
            .draw_background(&mut list, 0, 0, &screen.container);
        let px = render(&mut gpu, &rm, list);

        let seam = i32::from(rows) * 18 + 17;
        let top = assert_texels_match(&px, &source, 0, 0, 0, 0, 176, seam, "chest top half");
        let bottom =
            assert_texels_match(&px, &source, 0, seam, 0, 126, 176, 96, "chest bottom half");
        assert!(top > 1_000 && bottom > 5_000, "{rows} rows: {top}/{bottom}");

        // The very first row of the player half must be sheet row 126, not row `seam`.
        let want = source.get_pixel(88, 126).0;
        if want[3] == 255 {
            let got = pixel(&px, 88, seam as u32);
            assert_eq!(
                [got[0], got[1], got[2]],
                [want[0], want[1], want[2]],
                "{rows} rows: the seam pixel must come from v = 126"
            );
        }
    }
}

// ---------------------------------------------------------------- dynamic art

/// The furnace's flame and progress arrow are blitted from the sheet's right-hand strip at
/// `u = 176`, and both are *clipped by their own progress value* — the flame grows upward from
/// its bottom edge, the arrow rightward from its left edge.
#[test]
fn the_furnace_flame_and_arrow_draw_from_the_sheets_right_hand_strip() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let source = container_png(&rm, "furnace.png");

    let mut screen = ContainerScreen::furnace(1, "Furnace");
    screen.art = ContainerArt::Furnace(FurnaceArt {
        burn_time: 800,
        current_item_burn_time: 1600,
        cook_time: 100,
        total_cook_time: 200,
    });
    let ContainerArt::Furnace(f) = screen.art.clone() else {
        unreachable!()
    };
    let flame = f.burn_left_scaled(13);
    let arrow = f.cook_progress_scaled(24);
    assert_eq!((flame, arrow), (6, 12), "half-burnt, half-cooked");

    let mut list = DrawList::new();
    screen
        .art
        .draw_background(&mut list, 0, 0, &screen.container);
    let px = render(&mut gpu, &rm, list);

    // java: `drawTexturedModalRect(i + 56, j + 36 + 12 - k, 176, 12 - k, 14, k + 1)`.
    let checked = assert_texels_match(
        &px,
        &source,
        56,
        36 + 12 - flame,
        176,
        12 - flame,
        14,
        flame + 1,
        "furnace flame",
    );
    assert!(checked > 20, "flame: {checked} texels");

    // java: `drawTexturedModalRect(i + 79, j + 34, 176, 14, l + 1, 16)`.
    let checked = assert_texels_match(
        &px,
        &source,
        79,
        34,
        176,
        14,
        arrow + 1,
        16,
        "furnace arrow",
    );
    assert!(checked > 50, "arrow: {checked} texels");

    // Past the arrow's own width the panel art has to show through untouched.
    let beyond = 79 + arrow + 1;
    let want = source.get_pixel(beyond as u32, 40).0;
    let got = pixel(&px, beyond as u32, 40);
    assert_eq!(
        [got[0], got[1], got[2]],
        [want[0], want[1], want[2]],
        "the arrow must stop at its progress, not fill the whole track"
    );
}

/// The brewing stand's bubble column steps through a seven-entry table and its arrow fills as
/// `brewTime` counts down; both come from the sheet's `u = 176`/`u = 185` strips.
#[test]
fn the_brewing_stand_bubbles_and_arrow_draw_from_their_own_strips() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let source = container_png(&rm, "brewing_stand.png");

    let mut screen = ContainerScreen::brewing_stand(1, "Brewing Stand");
    let art = BrewingStandArt { brew_time: 200 };
    screen.art = ContainerArt::BrewingStand(art);
    let (arrow, bubbles) = (art.arrow_height(), art.bubble_height());
    assert_eq!((arrow, bubbles), (14, 20));

    let mut list = DrawList::new();
    screen
        .art
        .draw_background(&mut list, 0, 0, &screen.container);
    let px = render(&mut gpu, &rm, list);

    let checked = assert_texels_match(&px, &source, 97, 16, 176, 0, 9, arrow, "brew arrow");
    assert!(checked > 30, "arrow: {checked} texels");
    let checked = assert_texels_match(
        &px,
        &source,
        65,
        14 + 29 - bubbles,
        185,
        29 - bubbles,
        12,
        bubbles,
        "brew bubbles",
    );
    assert!(checked > 30, "bubbles: {checked} texels");

    // An idle stand draws neither.
    let mut idle = ContainerScreen::brewing_stand(1, "Brewing Stand");
    idle.art = ContainerArt::BrewingStand(BrewingStandArt { brew_time: 0 });
    let mut list = DrawList::new();
    idle.art.draw_background(&mut list, 0, 0, &idle.container);
    let idle_px = render(&mut gpu, &rm, list);
    let want = source.get_pixel(97, 16).0;
    let got = pixel(&idle_px, 97, 16);
    assert_eq!(
        [got[0], got[1], got[2]],
        [want[0], want[1], want[2]],
        "an idle brewing stand shows bare panel art where the arrow would be"
    );
}

/// The anvil's "these do not combine" cross sits at `u = xSize` on its own sheet and only
/// appears with an input present and the output slot empty.
#[test]
fn the_anvil_cross_appears_only_with_an_input_and_no_output() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let source = container_png(&rm, "anvil.png");

    let mut screen = ContainerScreen::anvil(1, "Repair & Name");
    screen.art = ContainerArt::Anvil(AnvilArt::default());

    let mut bare = DrawList::new();
    screen
        .art
        .draw_background(&mut bare, 0, 0, &screen.container);
    let bare_px = render(&mut gpu, &rm, bare);

    screen
        .container
        .set_slot_contents(0, Some(stack("stone", 1)));
    let mut crossed = DrawList::new();
    screen
        .art
        .draw_background(&mut crossed, 0, 0, &screen.container);
    let crossed_px = render(&mut gpu, &rm, crossed);

    // java: `drawTexturedModalRect(i + 99, j + 45, this.xSize, 0, 28, 21)`.
    let checked = assert_texels_match(&crossed_px, &source, 99, 45, 176, 0, 28, 21, "anvil cross");
    assert!(checked > 50, "cross: {checked} texels");

    // Somewhere inside the cross the two renders have to differ.
    let differs = (0..21)
        .flat_map(|dy| (0..28).map(move |dx| (99 + dx, 45 + dy)))
        .any(|(x, y)| pixel(&bare_px, x, y) != pixel(&crossed_px, x, y));
    assert!(differs, "the cross has to change the framebuffer");

    screen
        .container
        .set_slot_contents(2, Some(stack("stone", 1)));
    let mut with_output = DrawList::new();
    screen
        .art
        .draw_background(&mut with_output, 0, 0, &screen.container);
    let output_px = render(&mut gpu, &rm, with_output);
    let mut same = true;
    for dy in 0..21u32 {
        for dx in 0..28u32 {
            same &= pixel(&bare_px, 99 + dx, 45 + dy) == pixel(&output_px, 99 + dx, 45 + dy);
        }
    }
    assert!(same, "a filled output slot removes the cross entirely");
}

/// The enchanting table's three galley plates switch between three source rows: `v = 166`
/// (affordable), `v = 185` (dimmed / no offer) and the level pips at `v = 223` or `v = 239`.
#[test]
fn the_enchanting_plates_pick_their_row_from_affordability() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let source = container_png(&rm, "enchanting_table.png");

    let mut screen = ContainerScreen::enchanting_table(1, "Enchant");
    screen.art = ContainerArt::Enchanting(EnchantingArt {
        enchant_levels: [1, 5, 30],
        player_level: 10,
        enchantment_ids: [-1; 3],
        ..EnchantingArt::default()
    });
    screen.container.set_slot_contents(1, Some(stack("dye", 3)));

    let mut list = DrawList::new();
    screen
        .art
        .draw_background(&mut list, 0, 0, &screen.container);
    let px = render(&mut gpu, &rm, list);

    // The level pip covers the plate's left 17 columns, so the plate is compared past it.
    // Offer 0: level 1, 1 lapis needed, player has 10 levels -> affordable, v = 166.
    let checked = assert_texels_match(&px, &source, 77, 14, 17, 166, 91, 19, "offer 0 plate");
    assert!(checked > 200, "offer 0: {checked} texels");
    let checked = assert_texels_match(&px, &source, 61, 15, 0, 223, 16, 16, "offer 0 pip");
    assert!(checked > 20, "offer 0 pip: {checked} texels");

    // Offer 2: level 30 > the player's 10 -> dimmed, v = 185, pip row v = 239.
    let checked = assert_texels_match(
        &px,
        &source,
        77,
        14 + 38,
        17,
        185,
        91,
        19,
        "offer 2 dimmed plate",
    );
    assert!(checked > 200, "offer 2: {checked} texels");
    let checked = assert_texels_match(&px, &source, 61, 15 + 38, 32, 239, 16, 16, "offer 2 pip");
    assert!(checked > 20, "offer 2 pip: {checked} texels");

    // The two rows really are different art: the affordable plate and the dimmed one cannot
    // agree pixel-for-pixel, or the affordability branch would be untestable.
    let affordable = pixel(&px, 90, 20);
    let dimmed = pixel(&px, 90, 20 + 38);
    assert_ne!(
        affordable, dimmed,
        "an affordable and an unaffordable offer must not render identically"
    );
}

/// The beacon's confirm/cancel plates come from `beacon.png`'s `v = 219` strip, and a disabled
/// button samples 44 texels further right than an enabled one.
#[test]
fn the_beacon_buttons_pick_their_plate_from_the_enabled_state() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let source = container_png(&rm, "beacon.png");

    let screen = ContainerScreen::beacon(1, "Beacon");
    let mut list = DrawList::new();
    screen
        .art
        .draw_background(&mut list, 0, 0, &screen.container);
    let px = render(&mut gpu, &rm, list);

    // Each button's own glyph covers its inner 18x18, so the plate is compared on its 2-pixel
    // border. Confirm at (164, 107), disabled with no payment -> u = 44. Cancel at (190, 107),
    // always enabled -> u = 0.
    let border = |x: i32| -> Vec<Excluded> { vec![(x + 2, 109, 18, 18)] };
    let checked = assert_texels_match_except(
        &px,
        &source,
        164,
        107,
        44,
        219,
        22,
        22,
        &border(164),
        "confirm plate (disabled)",
    );
    assert!(checked > 100, "confirm: {checked} texels");
    let checked = assert_texels_match_except(
        &px,
        &source,
        190,
        107,
        0,
        219,
        22,
        22,
        &border(190),
        "cancel plate",
    );
    assert!(checked > 100, "cancel: {checked} texels");
    // The cancel glyph is blitted 2 pixels inside, from (112, 220).
    let checked = assert_texels_match(&px, &source, 192, 109, 112, 220, 18, 18, "cancel glyph");
    assert!(checked > 100, "cancel glyph: {checked} texels");

    // With a payment and a chosen primary, confirm becomes enabled and switches to u = 0.
    let mut enabled = ContainerScreen::beacon(1, "Beacon");
    enabled.art = ContainerArt::Beacon(BeaconArt {
        levels: 4,
        primary: 1,
        secondary: 0,
    });
    enabled
        .container
        .set_slot_contents(0, Some(stack("emerald", 1)));
    let mut list = DrawList::new();
    enabled
        .art
        .draw_background(&mut list, 0, 0, &enabled.container);
    let enabled_px = render(&mut gpu, &rm, list);
    let checked = assert_texels_match_except(
        &enabled_px,
        &source,
        164,
        107,
        0,
        219,
        22,
        22,
        &border(164),
        "confirm plate (enabled)",
    );
    assert!(checked > 100, "confirm enabled: {checked} texels");
    assert_ne!(
        pixel(&px, 164, 107),
        pixel(&enabled_px, 164, 107),
        "the enabled and disabled plates are different art"
    );

    // A full pyramid also draws seven effect buttons, each with a potion icon from inventory.png.
    let inventory = container_png(&rm, "inventory.png");
    // Speed is `setIconIndex(0, 0)` -> icon 0 -> texels (0, 198). The tier-0 row's first button
    // sits at (53, 22), so its icon lands at (55, 24).
    let checked = assert_texels_match(
        &enabled_px,
        &inventory,
        55,
        24,
        0,
        198,
        18,
        18,
        "speed effect icon",
    );
    assert!(checked > 100, "speed icon: {checked} texels");
}

// ---------------------------------------------------------------- hover + tooltip

/// Vanilla's slot highlight is a `0x80ffffff` gradient over exactly the slot's 16x16 box — not
/// its 18x18 *hit* box. The two being different sizes is the whole point of this test.
#[test]
fn the_slot_hover_highlight_covers_exactly_sixteen_by_sixteen_pixels() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");

    let mut screen = ContainerScreen::chest(1, 3, "Chest");
    screen.init(&ctx());
    let slot = screen.container.slots()[10];
    let (sx, sy) = (screen.gui_left() + slot.x, screen.gui_top() + slot.y);

    let mut away = DrawList::new();
    screen.draw(&mut away, &mut font, &ctx(), (0, 0));
    let away_px = render(&mut gpu, &rm, away);

    let mut over = DrawList::new();
    screen.draw(&mut over, &mut font, &ctx(), (sx + 8, sy + 8));
    assert_eq!(screen.hovered_slot(), Some(10));
    let over_px = render(&mut gpu, &rm, over);

    let mut changed = 0;
    for dy in 0..16 {
        for dx in 0..16 {
            let (x, y) = ((sx + dx) as u32, (sy + dy) as u32);
            let before = pixel(&away_px, x, y);
            let after = pixel(&over_px, x, y);
            assert!(
                after[0] >= before[0] && after[1] >= before[1] && after[2] >= before[2],
                "the highlight only ever lightens: ({x}, {y}) {before:?} -> {after:?}"
            );
            if after != before {
                changed += 1;
            }
        }
    }
    assert_eq!(changed, 256, "the whole 16x16 slot has to lighten");

    // One pixel outside the slot on every side is untouched, even though the *hit* box reaches
    // a pixel further out.
    for (x, y) in [(sx - 1, sy), (sx + 16, sy), (sx, sy - 1), (sx, sy + 16)] {
        assert_eq!(
            pixel(&away_px, x as u32, y as u32),
            pixel(&over_px, x as u32, y as u32),
            "the highlight leaked to ({x}, {y})"
        );
    }
}

/// The tooltip's frame is four gradient strips over a near-opaque panel; the border is visibly
/// blue-violet and the fill is not, which is the cheapest true statement about "the box
/// rendered" that does not depend on exact blend rounding.
#[test]
fn the_tooltip_box_paints_a_dark_panel_inside_a_violet_border() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");

    let mut screen = ContainerScreen::chest(1, 3, "Chest");
    screen.init(&ctx());
    screen
        .container
        .set_slot_contents(0, Some(stack("apple", 1)));
    let slot = screen.container.slots()[0];
    let mouse = (
        screen.gui_left() + slot.x + 8,
        screen.gui_top() + slot.y + 8,
    );

    let mut list = DrawList::new();
    screen.draw(&mut list, &mut font, &ctx(), mouse);
    let lines = screen.hovered_tooltip().expect("apple has a tooltip");
    let layout = tooltip_box::layout(&font, &lines, mouse.0, mouse.1, W as i32, H as i32)
        .expect("the tooltip is laid out");
    let px = render(&mut gpu, &rm, list);

    // The left border strip sits at `x - 3`, one pixel wide, from `y - 2`.
    let border = pixel(&px, (layout.x - 3) as u32, (layout.y + 2) as u32);
    // The fill just inside it.
    let fill = pixel(&px, (layout.x - 1) as u32, (layout.y + 2) as u32);
    assert!(
        border[2] > border[0] && border[2] > border[1],
        "the border gradient is blue-violet: {border:?}"
    );
    assert!(
        u32::from(border[2]) > u32::from(fill[2]) + 20,
        "the border has to stand out from the fill: {border:?} vs {fill:?}"
    );
    assert!(
        fill[0] < 60 && fill[1] < 60 && fill[2] < 70,
        "the tooltip fill is near-black: {fill:?}"
    );

    // Above the tooltip's own top border, the chest panel still shows through unchanged.
    let (_, top, _, _) = layout.panel_bounds();
    let mut without = DrawList::new();
    screen.draw(&mut without, &mut font, &ctx(), (0, 0));
    let without_px = render(&mut gpu, &rm, without);
    assert_eq!(
        pixel(&px, layout.x as u32, (top - 1) as u32),
        pixel(&without_px, layout.x as u32, (top - 1) as u32),
        "the tooltip painted outside its own panel bounds"
    );
}

// ---------------------------------------------------------------- creative screen

/// The creative tab strip is the piece the "Creative-mode inventory with tabs and search" task
/// was left unticked for. Every one of the twelve plates has to land on `tabs.png`'s own texels,
/// and the **selected** plate has to sample 32 rows further down the sheet than an idle one.
#[test]
fn the_creative_tab_strip_lands_on_the_real_tabs_png_texels() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");
    let tabs = source_png(
        &rm,
        &UiTexture::gui("textures/gui/container/creative_inventory/tabs.png"),
    );

    let mut screen = CreativeScreen::new();
    screen.init(&ctx());
    screen.set_tab(0);
    assert_eq!(screen.selected_tab(), 0);
    let mut list = DrawList::new();
    // Mouse parked off the strip so no hover tooltip is drawn over the plates.
    screen.draw(&mut list, &mut font, &ctx(), (W as i32 - 1, H as i32 - 1));
    let px = render(&mut gpu, &rm, list);

    let (left, top) = (screen.gui_left(), screen.gui_top());

    // `GuiContainerCreative.drawTab` paints the item icon after the plate.  The icon is
    // intentionally absent from `tabs.png`, so exclude its small overlay while comparing the
    // plate's own texels.  Keep this in window coordinates because that is what the framebuffer
    // helper consumes.
    let icon_exclusion = |plate: mc_ui::screens::container::creative_screen::TabPlate| {
        vec![(left + plate.icon.0, top + plate.icon.1, 18, 18)]
    };

    // The selected tab (index 0, top row) is drawn last, so all 32 of its rows are visible.
    let selected = tab_plate(0, true);
    assert_eq!(selected.v, 32, "a selected top-row plate is sheet row 32");
    let checked = assert_texels_match_except(
        &px,
        &tabs,
        left + selected.x,
        top + selected.y,
        selected.u,
        selected.v,
        28,
        32,
        &icon_exclusion(selected),
        "selected tab 0",
    );
    assert!(checked > 400, "selected plate: {checked} texels");

    // Idle top-row plates: the panel covers their bottom 4 rows, so check the 28 above it.
    for index in 1..6u8 {
        let plate = tab_plate(index, false);
        assert_eq!(plate.v, 0, "an idle top-row plate is sheet row 0");
        let checked = assert_texels_match_except(
            &px,
            &tabs,
            left + plate.x,
            top + plate.y,
            plate.u,
            plate.v,
            28,
            28,
            &icon_exclusion(plate),
            &format!("idle tab {index}"),
        );
        assert!(checked > 300, "idle tab {index}: {checked} texels");
    }

    // Idle bottom-row plates sample rows 64..96 and start 4 pixels below the panel's edge.
    for index in 6..12u8 {
        let plate = tab_plate(index, false);
        assert_eq!(plate.v, 64, "an idle bottom-row plate is sheet row 64");
        let checked = assert_texels_match_except(
            &px,
            &tabs,
            left + plate.x,
            top + plate.y + 4,
            plate.u,
            plate.v + 4,
            28,
            28,
            &icon_exclusion(plate),
            &format!("idle tab {index}"),
        );
        assert!(checked > 300, "idle tab {index}: {checked} texels");
    }

    // Switching tabs moves the selected art: tab 0's plate goes idle and tab 7's lights up.
    screen.set_tab(7);
    let mut list = DrawList::new();
    screen.draw(&mut list, &mut font, &ctx(), (W as i32 - 1, H as i32 - 1));
    let px = render(&mut gpu, &rm, list);
    let selected = tab_plate(7, true);
    assert_eq!(
        selected.v, 96,
        "a selected bottom-row plate is sheet row 96"
    );
    let checked = assert_texels_match_except(
        &px,
        &tabs,
        left + selected.x,
        top + selected.y,
        selected.u,
        selected.v,
        28,
        32,
        &icon_exclusion(selected),
        "selected tab 7",
    );
    assert!(checked > 400, "selected tab 7: {checked} texels");
}

/// The panel behind the grid is a different file per tab family, and the search tab's box plus
/// the scrollbar handle are drawn on top of it.
#[test]
fn the_creative_panel_search_box_and_scrollbar_land_on_their_own_texels() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");
    let tabs = source_png(
        &rm,
        &UiTexture::gui("textures/gui/container/creative_inventory/tabs.png"),
    );

    let mut screen = CreativeScreen::new();
    screen.init(&ctx());
    screen.set_tab(0);
    let (left, top) = (screen.gui_left(), screen.gui_top());
    assert!(screen.needs_scroll_bars(), "building blocks overflows");

    let mut list = DrawList::new();
    screen.draw(&mut list, &mut font, &ctx(), (W as i32 - 1, H as i32 - 1));
    let px = render(&mut gpu, &rm, list);

    // The panel itself. Only the rows below the tab strip's overlap are pure panel: the strip's
    // bottom 4 rows sit on top of rows 0..4, and the item grid + scrollbar over the rest, so
    // sample the left gutter column, which nothing else covers.
    let items_panel = source_png(
        &rm,
        &UiTexture::gui("textures/gui/container/creative_inventory/tab_items.png"),
    );
    let checked = assert_texels_match(
        &px,
        &items_panel,
        left,
        top + 20,
        0,
        20,
        9,
        Y_SIZE - 25,
        "creative panel gutter",
    );
    assert!(checked > 500, "panel gutter: {checked} texels");

    // The scrollbar handle at scroll 0: 12x15 from (232, 0), because the bar is live.
    let checked = assert_texels_match(
        &px,
        &tabs,
        left + 175,
        top + 18,
        232,
        0,
        12,
        15,
        "scrollbar handle (live)",
    );
    assert!(checked > 100, "handle: {checked} texels");

    // The Inventory tab has no grid, so its handle is the greyed art 12 texels to the right —
    // and vanilla does not draw a handle there at all, so switch to search with no results.
    screen.set_tab(CreativeTab::Search.index().expect("search is registered"));
    screen.apply_pending_input(&font);
    // An *empty* query lists the whole pool in vanilla (`"".contains("")`), so it overflows.
    // Type something no item name contains to reach the no-results state this asserts on.
    for c in "zzqxzz".chars() {
        screen.key_typed_with_font(Some(c), 0, mc_ui::widget::Modifiers::NONE, &font);
    }
    assert!(
        !screen.needs_scroll_bars(),
        "a query with no matches lists nothing"
    );
    let mut list = DrawList::new();
    screen.draw(&mut list, &mut font, &ctx(), (W as i32 - 1, H as i32 - 1));
    let px = render(&mut gpu, &rm, list);
    let checked = assert_texels_match(
        &px,
        &tabs,
        left + 175,
        top + 18,
        244,
        0,
        12,
        15,
        "scrollbar handle (inert)",
    );
    assert!(checked > 100, "inert handle: {checked} texels");

    // ...and the search tab's own panel file is a *different* image.
    let search_panel = source_png(
        &rm,
        &UiTexture::gui("textures/gui/container/creative_inventory/tab_item_search.png"),
    );
    let checked = assert_texels_match(
        &px,
        &search_panel,
        left,
        top + 20,
        0,
        20,
        9,
        Y_SIZE - 25,
        "search panel gutter",
    );
    assert!(checked > 500, "search panel gutter: {checked} texels");
    assert_eq!(
        TABS[screen.selected_tab() as usize],
        CreativeTab::Search,
        "still on the search tab"
    );
}

/// Scrolling has to move real pixels: the handle slides down the gutter as the grid advances.
/// (The *icons* changing is proved in `tests/creative_search_render.rs`, which needs the baked
/// item models installed and therefore has to be its own process — see that file's module doc.)
#[test]
fn scrolling_the_creative_grid_slides_the_scrollbar_handle_down_the_gutter() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");

    let mut screen = CreativeScreen::new();
    screen.init(&ctx());
    screen.set_tab(0);
    let (left, top) = (screen.gui_left(), screen.gui_top());

    let mut top_list = DrawList::new();
    screen.draw(
        &mut top_list,
        &mut font,
        &ctx(),
        (W as i32 - 1, H as i32 - 1),
    );
    let top_px = render(&mut gpu, &rm, top_list);

    screen.scroll_by_wheel(-1.0);
    assert!(screen.scroll() > 0.0);
    let mut scrolled_list = DrawList::new();
    screen.draw(
        &mut scrolled_list,
        &mut font,
        &ctx(),
        (W as i32 - 1, H as i32 - 1),
    );
    let scrolled_px = render(&mut gpu, &rm, scrolled_list);

    // The handle moved down the gutter.
    let handle_top_row = |px: &[u8]| -> Option<u32> {
        (0..112u32).find(|dy| {
            let a = pixel(px, (left + 176) as u32, (top + 18) as u32 + dy);
            let b = pixel(px, (left + 176) as u32, (top + 18) as u32 + dy + 1);
            a != b
        })
    };
    let before = handle_top_row(&top_px);
    let after = handle_top_row(&scrolled_px);
    assert!(
        before.is_some() && after.is_some() && after > before,
        "the scrollbar handle must move down: {before:?} -> {after:?}"
    );
}
