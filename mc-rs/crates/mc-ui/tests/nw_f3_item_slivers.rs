//! [`DrawList::enable_item_sliver_culling`] is pixel-identical.
//!
//! Every item in the 1.8.9 registry (plus metadata variants with extra layers or tints) is drawn
//! as a GUI icon in a grid, plain and with an enchantment glint, at GUI scales 1, 2 and 3 and
//! under HUD-element transforms (scales 0.5 to 1.5 at a whole-pixel offset, like a placed mod HUD
//! element). Each page is rendered twice through the real `UI 800` phase: once with sliver
//! culling off (the old path) and once on. The frames must be byte-identical; at element scale 1
//! the culled lists must be much smaller (the culling really ran), and under any other scale
//! exactly as large (off the whole-pixel grid nothing is culled). A negative control drops one
//! real icon and must change the frame, so the comparison can see a missing quad.
//!
//! The glint clock is pinned, so both renders use the same foil UVs.
//!
//! `MC_NW_F3_PNG_DIR=<dir>` also writes the first page of every configuration for a human look.
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent.

use std::path::{Path, PathBuf};

use mc_assets::{bake_all, install, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_nbt::{Nbt, NbtCompound};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::screens::container::render::draw_item_icon_tinted;
use mc_ui::screens::container::stack_ext::ItemStack;
use mc_ui::widget::{DrawList, ScaledResolution, Transform, UiPhase, UiTexture};

const W: u32 = 1280;
const H: u32 = 960;
/// One slot: an icon is 16 GUI pixels, plus a 2-pixel gap.
const CELL: f32 = 18.0;

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

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

fn atlas_key() -> UiTexture {
    UiTexture::Resource(ResourceLocation::minecraft("mc-ui/nw-f3-item-atlas"))
}

fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    atlas: &mc_assets::Atlas,
    gui_scale: i32,
    list: DrawList,
) -> Vec<u8> {
    let mut phase = UiPhase::new(gpu, W, H, gui_scale, false);
    phase
        .renderer
        .upload(gpu, atlas_key(), &atlas.texture, atlas.w, atlas.h);
    // A texture that cannot load (a builtin item's entity sheet) falls back to white in both
    // renders alike, so it does not weaken the comparison.
    let _ = phase.renderer.ensure_list_textures(gpu, rm, &list);
    phase.set_resolution(ScaledResolution::new(W, H, gui_scale, false));
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

fn enchanted(mut stack: ItemStack) -> ItemStack {
    let mut entry = NbtCompound::new();
    entry.insert("id", Nbt::Short(16));
    entry.insert("lvl", Nbt::Short(3));
    let mut root = NbtCompound::new();
    root.insert("ench", Nbt::list(vec![Nbt::Compound(entry)]));
    stack.nbt = Some(root);
    stack
}

/// Every registered item at damage 0, plus variants whose icons have extra layers or tints:
/// potions (splash and drinkable), spawn eggs, dyes, golden apple 1, a damaged bow and sword.
fn all_stacks() -> Vec<ItemStack> {
    let mut stacks: Vec<ItemStack> = mc_data::item::items()
        .iter()
        .map(|props| ItemStack {
            item_id: props.id.0 as i16,
            count: 1,
            damage: 0,
            nbt: None,
        })
        .collect();
    let named = |name: &str, damage: i16| ItemStack {
        item_id: mc_data::item_by_name(name)
            .unwrap_or_else(|| panic!("{name} is a 1.8.9 item"))
            .0 as i16,
        count: 1,
        damage,
        nbt: None,
    };
    for (name, damage) in [
        ("potion", 8193),
        ("potion", 16417),
        ("potion", 8226),
        ("spawn_egg", 50),
        ("spawn_egg", 90),
        ("spawn_egg", 120),
        ("dye", 1),
        ("dye", 4),
        ("dye", 15),
        ("golden_apple", 1),
        ("bow", 200),
        ("diamond_sword", 1200),
        ("leather_chestplate", 0),
        ("fishing_rod", 0),
        ("filled_map", 0),
    ] {
        stacks.push(named(name, damage));
    }
    stacks
}

/// The configurations: (GUI scale, HUD element scale). Every element is translated to a whole
/// GUI pixel first, as `ModHud` places it.
const CONFIGS: [(i32, f32); 12] = [
    (1, 1.0),
    (2, 1.0),
    (3, 1.0),
    (3, 0.5),
    (3, 0.7),
    (3, 0.75),
    (3, 0.8),
    (3, 1.25),
    (3, 1.5),
    (2, 0.7),
    (2, 1.5),
    (1, 0.6),
];

/// Draw `stacks` as one page of the grid into a list (culling per `cull`).
fn page(
    stacks: &[ItemStack],
    columns: usize,
    element_scale: f32,
    cull: bool,
    skip: Option<usize>,
) -> DrawList {
    let mut list = DrawList::new();
    if cull {
        list.enable_item_sliver_culling();
    }
    list.push_transform(Transform::translate(3.0, 5.0).then_scale(element_scale));
    for (index, stack) in stacks.iter().enumerate() {
        if skip == Some(index) {
            continue;
        }
        let x = (index % columns) as f32 * CELL;
        let y = (index / columns) as f32 * CELL;
        draw_item_icon_tinted(
            &mut list,
            &atlas_key(),
            x as i32,
            y as i32,
            stack,
            0xFFFF_FFFF,
        );
    }
    list.pop_transform();
    list
}

#[test]
fn sliver_culling_leaves_every_item_icon_pixel_identical() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let (baked, _report) = bake_all(&rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    install(baked);
    // Both renders of a page must see the same foil UVs.
    mc_render::entity::item::pin_item_glint_clock(Some(1_234));
    let png_dir = std::env::var_os("MC_NW_F3_PNG_DIR").map(PathBuf::from);

    let plain = all_stacks();
    let shiny: Vec<ItemStack> = plain.iter().cloned().map(enchanted).collect();
    let mut checked_pages = 0;
    let (mut vertices_off, mut vertices_on) = (0_usize, 0_usize);
    for (gui_scale, element_scale) in CONFIGS {
        let gui_w = W as f32 / gui_scale as f32;
        let gui_h = H as f32 / gui_scale as f32;
        let columns = (((gui_w - 8.0) / element_scale) / CELL).floor().max(1.0) as usize;
        let rows = (((gui_h - 8.0) / element_scale) / CELL).floor().max(1.0) as usize;
        for (set_name, set) in [("plain", &plain), ("glint", &shiny)] {
            for (page_index, chunk) in set.chunks(columns * rows).enumerate() {
                let off = page(chunk, columns, element_scale, false, None);
                let on = page(chunk, columns, element_scale, true, None);
                if element_scale == 1.0 {
                    vertices_off += off.vertices().len();
                    vertices_on += on.vertices().len();
                } else {
                    // A scaled element is off the whole-pixel grid: every quad is kept.
                    assert_eq!(
                        on.vertices().len(),
                        off.vertices().len(),
                        "element scale {element_scale}: nothing may be culled off the pixel grid"
                    );
                }
                let frame_off = render(&mut gpu, &rm, &atlas, gui_scale, off);
                let frame_on = render(&mut gpu, &rm, &atlas, gui_scale, on);
                let differing = frame_off
                    .chunks_exact(4)
                    .zip(frame_on.chunks_exact(4))
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(
                    differing, 0,
                    "GUI scale {gui_scale}, element scale {element_scale}, {set_name} page \
                     {page_index}: {differing} pixels differ with sliver culling on"
                );
                if page_index == 0 {
                    if let Some(dir) = &png_dir {
                        std::fs::create_dir_all(dir).expect("png dir");
                        let name = format!("items-gui{gui_scale}-x{element_scale}-{set_name}.png");
                        image::save_buffer(
                            dir.join(name),
                            &frame_on,
                            W,
                            H,
                            image::ExtendedColorType::Rgba8,
                        )
                        .expect("save png");
                    }
                }
                checked_pages += 1;
            }
        }
    }
    eprintln!(
        "{checked_pages} pages identical; UI vertices {vertices_off} -> {vertices_on} ({:.1}%)",
        100.0 * vertices_on as f64 / vertices_off as f64
    );
    assert!(checked_pages >= 24, "only {checked_pages} pages compared");
    assert!(
        vertices_on * 3 < vertices_off,
        "culling barely ran: {vertices_off} -> {vertices_on} vertices"
    );

    // Negative control: one real icon fewer must show, so a dropped front card cannot hide.
    let sword = plain
        .iter()
        .position(|stack| {
            Some(stack.item_id) == mc_data::item_by_name("diamond_sword").map(|id| id.0 as i16)
        })
        .expect("a diamond sword in the registry");
    let chunk = &plain[..(sword + 1).min(plain.len())];
    let full = render(&mut gpu, &rm, &atlas, 3, page(chunk, 23, 1.0, true, None));
    let missing = render(
        &mut gpu,
        &rm,
        &atlas,
        3,
        page(chunk, 23, 1.0, true, Some(sword)),
    );
    let differing = full
        .chunks_exact(4)
        .zip(missing.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > 50,
        "the comparison cannot see a missing sword icon ({differing} pixels differ)"
    );
    mc_render::entity::item::pin_item_glint_clock(None);
}
