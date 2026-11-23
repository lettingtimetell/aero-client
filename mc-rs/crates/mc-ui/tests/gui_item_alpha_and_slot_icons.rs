//! Three GUI item-rendering reports, checked on real pixels.
//!
//! * "Enchantment glint covers whole inventory slot rather than just the item".
//!   `RenderItem.renderItemIntoGUI` draws the model *and* its `renderEffect` foil under
//!   `enableAlpha(); alphaFunc(GL_GREATER, 0.1F)`. A generated item card's transparent texels are
//!   discarded and write no depth, so the `GL_EQUAL` foil replay can only land on the item's own
//!   pixels. Without the alpha test the card's whole 16x16 quad wrote depth and the foil filled
//!   the slot.
//! * "beacon just looks like glass in inventory". `block/beacon.json` lists the glass shell
//!   first. With the alpha test the shell's clear texels write no depth, so the obsidian base and
//!   the core inside still pass `GL_LEQUAL` and show through.
//! * "Armor slots in inventory look like they are displaying an entire texture map".
//!   `GuiContainer.drawSlot` binds the block atlas and blits
//!   `getAtlasSprite("minecraft:items/empty_armor_slot_*")` into the 16x16 slot.
//!
//! Each test renders through the real `UI 800` phase against the vanilla 1.8.9 jar and reads the
//! framebuffer back. The jar's models are baked and installed (a process-wide registry), so this
//! is its own test binary. Skips (loudly) without a wgpu adapter or the jar; set `MC_JARS_DIR`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use mc_assets::{
    bake_all, install, item_model_for, Atlas, BakedQuad, ResourceLocation, ResourceManager,
    ZipResourcePack,
};
use mc_data::item_by_name;
use mc_nbt::{Nbt, NbtCompound};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::{Screen, ScreenCtx};
use mc_ui::screens::container::creative_screen::CreativeScreen;
use mc_ui::screens::container::render::{draw_iso_block_icon, draw_item_icon_tinted};
use mc_ui::screens::container::screen::ContainerScreen;
use mc_ui::screens::container::{default_atlas_texture, ItemStack};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, UiTexture};

const W: u32 = 256;
const H: u32 = 256;
/// A distinctive clear colour, so an opaque black item texel cannot pass for "background".
const CLEAR: [u8; 3] = [51, 102, 153];

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
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let dir = PathBuf::from(dir);
        return dir.is_dir().then_some(dir);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let dir = root.join("MCP-919-main").join("jars");
    dir.is_dir().then_some(dir)
}

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

/// The jar's stitched block atlas, baked and installed once for this process.
fn atlas() -> Option<&'static Atlas> {
    static ATLAS: OnceLock<Option<Atlas>> = OnceLock::new();
    ATLAS
        .get_or_init(|| {
            let rm = resources()?;
            let (baked, _report) = bake_all(&rm, 4096).expect("bake_all against the real jar");
            let atlas = baked.atlas.clone();
            install(baked);
            Some(atlas)
        })
        .as_ref()
}

fn resolution() -> ScaledResolution {
    let resolution = ScaledResolution::new(W, H, 1, false);
    assert_eq!(resolution.scale_factor(), 1);
    resolution
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(resolution(), 0.0)
}

/// Render one list with the block atlas uploaded under `atlas_key` and every other texture it
/// names loaded from the jar.
fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    atlas: &Atlas,
    atlas_key: &UiTexture,
    list: DrawList,
) -> Vec<u8> {
    let mut phase = UiPhase::new(gpu, W, H, 1, false);
    phase
        .renderer
        .upload(gpu, atlas_key.clone(), &atlas.texture, atlas.w, atlas.h);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(
        missing.is_empty(),
        "textures missing from the jar: {missing:?}"
    );
    phase.set_resolution(resolution());
    phase.submit(list);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: f64::from(CLEAR[0]) / 255.0,
        g: f64::from(CLEAR[1]) / 255.0,
        b: f64::from(CLEAR[2]) / 255.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

fn rgb(px: &[u8], x: i32, y: i32) -> [u8; 3] {
    let i = ((y as u32 * W + x as u32) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

fn stack(name: &str) -> ItemStack {
    let id = item_by_name(name).expect("a real 1.8.9 item");
    ItemStack {
        item_id: id.0 as i16,
        count: 1,
        damage: 0,
        nbt: None,
    }
}

/// An ordinary stack carrying an (empty) `ench` list: `ItemStack.isItemEnchanted()` is true.
fn enchanted(mut stack: ItemStack) -> ItemStack {
    let mut root = NbtCompound::new();
    root.insert("ench", Nbt::list(vec![]));
    stack.nbt = Some(root);
    stack
}

#[test]
fn an_enchanted_items_foil_stays_on_the_items_own_pixels() {
    let Some(mut gpu) = gpu() else { return };
    let (Some(rm), Some(atlas)) = (resources(), atlas()) else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };
    let key = default_atlas_texture();
    let (x, y) = (40, 40);

    // (A golden apple would not do: `ItemAppleGold.hasEffect` ignores `ench` and foils only
    // metadata 1, so its "enchanted" stack correctly draws no foil at all.)
    for name in ["diamond_sword", "bow", "diamond_chestplate"] {
        let draw = |stack: &ItemStack| {
            let mut list = DrawList::new();
            draw_item_icon_tinted(&mut list, &key, x, y, stack, 0xFFFF_FFFF);
            list
        };
        let plain = render(&mut gpu, &rm, atlas, &key, draw(&stack(name)));
        let foiled = render(&mut gpu, &rm, atlas, &key, draw(&enchanted(stack(name))));

        let (mut item_pixels, mut foil_on_item, mut foil_off_item) = (0, 0, 0);
        for dy in -2..18 {
            for dx in -2..18 {
                let (before, after) = (rgb(&plain, x + dx, y + dy), rgb(&foiled, x + dx, y + dy));
                if before == CLEAR {
                    foil_off_item += usize::from(after != CLEAR);
                } else {
                    item_pixels += 1;
                    foil_on_item += usize::from(after != before);
                }
            }
        }
        eprintln!("{name}: {item_pixels} item px, foil on {foil_on_item}, off {foil_off_item}");
        assert!(item_pixels > 20, "{name}: the plain icon was not drawn");
        assert!(
            foil_on_item > 0,
            "{name}: the foil must still reach the item's pixels"
        );
        assert_eq!(
            foil_off_item, 0,
            "{name}: the foil coloured {foil_off_item} slot pixels the item does not cover"
        );
    }
}

#[test]
fn the_beacons_core_shows_through_its_glass_shell() {
    let Some(mut gpu) = gpu() else { return };
    let (Some(rm), Some(atlas)) = (resources(), atlas()) else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };
    let key = default_atlas_texture();
    let (x, y) = (40, 40);

    let beacon = item_model_for(item_by_name("beacon").expect("beacon"));
    assert!(beacon.gui_3d, "the beacon item is a GUI-3D block model");
    // `block/beacon.json`: glass shell, then obsidian base, then the core. Keep only the quads
    // that sample the glass sprite to get the shell on its own.
    let glass = mc_assets::sprite(&ResourceLocation::minecraft(
        "textures/blocks/glass".to_owned(),
    ))
    .uv;
    let is_glass = |quad: &BakedQuad| {
        quad.verts.iter().all(|vertex| {
            (glass.min_u..=glass.max_u).contains(&vertex.uv[0])
                && (glass.min_v..=glass.max_v).contains(&vertex.uv[1])
        })
    };
    let mut shell = beacon.clone();
    shell.general_quads.retain(|quad| is_glass(quad));
    for face in &mut shell.face_quads {
        face.retain(|quad| is_glass(quad));
    }
    let count = |model: &mc_assets::BakedModel| {
        model.general_quads.len() + model.face_quads.iter().map(Vec::len).sum::<usize>()
    };
    assert_eq!(count(&shell), 6, "the glass shell is one cube");
    assert_eq!(count(beacon), 18, "glass shell, obsidian base and core");

    let draw = |model: &mc_assets::BakedModel| {
        let mut list = DrawList::new();
        draw_iso_block_icon(&mut list, &key, x, y, model);
        list
    };
    let full = render(&mut gpu, &rm, atlas, &key, draw(beacon));
    let glass_only = render(&mut gpu, &rm, atlas, &key, draw(&shell));

    let (mut differing, mut see_through) = (0, 0);
    for dy in 0..16 {
        for dx in 0..16 {
            let (a, b) = (rgb(&full, x + dx, y + dy), rgb(&glass_only, x + dx, y + dy));
            differing += usize::from(a != b);
            see_through += usize::from(b == CLEAR);
        }
    }
    eprintln!("beacon: {differing} px differ from the bare shell, {see_through} clear in it");
    assert!(
        see_through > 20,
        "the bare glass shell must be see-through in its clear texels"
    );
    assert!(
        differing > 40,
        "the obsidian base and the core must show through the glass ({differing} px)"
    );
}

/// Compare the 16x16 slot at `(sx, sy)` with the sprite PNG: every opaque texel must reach the
/// framebuffer unchanged. Returns how many texels were compared.
fn assert_slot_shows_sprite(px: &[u8], rm: &ResourceManager, sx: i32, sy: i32, name: &str) {
    let location = ResourceLocation::minecraft(format!("textures/items/{name}.png"));
    let bytes = rm.load(&location).expect("sprite is in the vanilla jar");
    let sprite = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .expect("decodes as PNG")
        .to_rgba8();
    assert_eq!(sprite.dimensions(), (16, 16), "{name}");
    let mut checked = 0;
    for ty in 0..16 {
        for tx in 0..16 {
            let texel = sprite.get_pixel(tx, ty).0;
            if texel[3] != 255 {
                continue;
            }
            assert_eq!(
                rgb(px, sx + tx as i32, sy + ty as i32),
                [texel[0], texel[1], texel[2]],
                "{name}: slot pixel ({tx}, {ty}) must be the sprite's texel"
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 20,
        "{name}: only {checked} opaque texels compared"
    );
}

const ARMOUR_SPRITES: [&str; 4] = [
    "empty_armor_slot_helmet",
    "empty_armor_slot_chestplate",
    "empty_armor_slot_leggings",
    "empty_armor_slot_boots",
];

#[test]
fn the_empty_armour_slots_show_their_sprites_not_the_whole_atlas() {
    let Some(mut gpu) = gpu() else { return };
    let (Some(rm), Some(atlas)) = (resources(), atlas()) else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");

    let mut screen = ContainerScreen::player(0);
    screen.init(&ctx());
    let mut list = DrawList::new();
    screen.draw(&mut list, &mut font, &ctx(), (W as i32 - 1, H as i32 - 1));
    let px = render(&mut gpu, &rm, atlas, &screen.atlas, list);

    // java: ContainerPlayer's slots 5..=8 are helmet, chestplate, leggings, boots.
    for (index, name) in (5..9).zip(ARMOUR_SPRITES) {
        let slot = screen.container.slots()[index];
        let (sx, sy) = (screen.gui_left() + slot.x, screen.gui_top() + slot.y);
        assert_slot_shows_sprite(&px, &rm, sx, sy, name);
    }
}

#[test]
fn the_creative_inventory_tabs_armour_cells_show_their_sprites_too() {
    let Some(mut gpu) = gpu() else { return };
    let (Some(rm), Some(atlas)) = (resources(), atlas()) else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");

    let mut screen = CreativeScreen::new();
    screen.init(&ctx());
    screen.set_tab(11); // CreativeTab::Inventory
    let mut list = DrawList::new();
    screen.draw(&mut list, &mut font, &ctx(), (W as i32 - 1, H as i32 - 1));
    let px = render(&mut gpu, &rm, atlas, &screen.atlas, list);

    // `slot_rects` lists the inventory tab's armour cells first, helmet to boots.
    let rects = screen.slot_rects();
    for ((_, x, y), name) in rects.iter().take(4).zip(ARMOUR_SPRITES) {
        let (sx, sy) = (screen.gui_left() + x, screen.gui_top() + y);
        assert_slot_shows_sprite(&px, &rm, sx, sy, name);
    }
}
