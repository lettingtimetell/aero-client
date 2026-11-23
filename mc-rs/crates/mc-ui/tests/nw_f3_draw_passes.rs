//! The pass-ordered hotbar items
//! ([`Hud::draw_hotbar_items_in_order`]) and Tab list rows (`TabListOverlay::set_pass_order`)
//! draw exactly the pixels of the element-by-element order, with fewer UI batches.
//!
//! Each scene is rendered through the real `UI 800` phase twice, element by element and in
//! passes, at GUI scales 1, 2 and 3, and the frames are compared byte for byte:
//! * hotbars: enchanted (glint), GUI-3D and flat items, counts of 2, 16, 64, 1000 and -1, a
//!   damaged tool (durability bar), and the pick-up pop animation at several partial ticks;
//! * Tab lists of 40 rows: latency bars and numbers, heads with and without the hat layer,
//!   integer and hearts objectives, spectators, a header and a footer.
//!
//! The glint clock is pinned. `MC_NW_F3_PNG_DIR=<dir>` also writes the pass-ordered frames.
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent.

use std::path::{Path, PathBuf};

use mc_assets::{bake_all, install, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_nbt::{Nbt, NbtCompound};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::container::stack_ext::ItemStack;
use mc_ui::screens::hud::{Hud, PlayerHud};
use mc_ui::screens::tab_list::{
    sort_rows, TabListOverlay, TabListStyle, TabObjective, TabPing, TabRenderType, TabRow,
};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, UiTexture};

const W: u32 = 1280;
const H: u32 = 720;

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
    UiTexture::Resource(ResourceLocation::minecraft("mc-ui/nw-f3-passes-atlas"))
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

fn stack(name: &str, count: i32, damage: i16, enchanted: bool) -> ItemStack {
    let id = mc_data::item_by_name(name).unwrap_or_else(|| panic!("{name} is a 1.8.9 item"));
    let nbt = enchanted.then(|| {
        let mut entry = NbtCompound::new();
        entry.insert("id", Nbt::Short(16));
        entry.insert("lvl", Nbt::Short(3));
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(vec![Nbt::Compound(entry)]));
        root
    });
    ItemStack {
        item_id: id.0 as i16,
        count,
        damage,
        nbt,
    }
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(p, q)| p != q)
        .count()
}

fn save(name: &str, pixels: &[u8]) {
    if let Some(dir) = std::env::var_os("MC_NW_F3_PNG_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("png dir");
        image::save_buffer(
            dir.join(name),
            pixels,
            W,
            H,
            image::ExtendedColorType::Rgba8,
        )
        .expect("save png");
    }
}

#[test]
fn pass_ordered_hotbars_and_tab_lists_are_pixel_identical_with_fewer_batches() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let (baked, _report) = bake_all(&rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    install(baked);
    mc_render::entity::item::pin_item_glint_clock(Some(4_321));
    let mut font = FontRenderer::load(&rm, false, false).expect("font");

    // ---- hotbars --------------------------------------------------------------------------
    let hotbars: [[Option<ItemStack>; 9]; 2] = [
        [
            Some(stack("diamond_sword", 1, 0, true)),
            Some(stack("bow", 1, 0, true)),
            Some(stack("fishing_rod", 1, 0, false)),
            Some(stack("ender_pearl", 16, 0, false)),
            Some(stack("stone", 64, 0, false)),
            Some(stack("golden_apple", 2, 0, false)),
            Some(stack("water_bucket", 1, 0, false)),
            Some(stack("snowball", 16, 0, false)),
            None,
        ],
        [
            Some(stack("diamond_pickaxe", 1, 900, true)),
            Some(stack("arrow", 1000, 0, false)),
            Some(stack("cobblestone", -1, 0, false)),
            Some(stack("golden_apple", 3, 1, false)),
            Some(stack("iron_chestplate", 1, 120, true)),
            Some(stack("enchanted_book", 1, 0, false)),
            Some(stack("potion", 1, 16417, false)),
            Some(stack("wool", 12, 14, false)),
            Some(stack("diamond_helmet", 1, 300, true)),
        ],
    ];
    let mut hotbar_frames = 0;
    for gui_scale in [1, 2, 3] {
        let res = ScaledResolution::new(W, H, gui_scale, false);
        for (set, items) in hotbars.iter().enumerate() {
            for (animations, partial) in [
                ([0; 9], 0.0_f32),
                ([5, 3, 1, 4, 2, 5, 0, 3, 1], 0.0),
                ([5, 3, 1, 4, 2, 5, 0, 3, 1], 0.37),
                ([2; 9], 0.9),
            ] {
                let ctx = ScreenCtx::new(res, partial);
                let state = PlayerHud {
                    hotbar: items.clone(),
                    animations_to_go: animations,
                    ..PlayerHud::default()
                };
                let hud = Hud::new();
                let mut lists = [DrawList::new(), DrawList::new()];
                for (list, passes) in lists.iter_mut().zip([false, true]) {
                    list.enable_solid_merging();
                    list.enable_item_sliver_culling();
                    hud.draw_hotbar_items_in_order(
                        list,
                        &mut font,
                        &atlas_key(),
                        &ctx,
                        &state,
                        passes,
                    );
                }
                let [ordered, passed] = lists;
                assert!(
                    passed.batches().len() < ordered.batches().len(),
                    "hotbar {set} at GUI {gui_scale}: {} pass batches vs {} in order",
                    passed.batches().len(),
                    ordered.batches().len()
                );
                let a = render(&mut gpu, &rm, &atlas, gui_scale, ordered);
                let b = render(&mut gpu, &rm, &atlas, gui_scale, passed);
                let diff = differing(&a, &b);
                assert_eq!(
                    diff, 0,
                    "hotbar {set} at GUI {gui_scale}, animations {animations:?} at {partial}: \
                     {diff} pixels differ"
                );
                if animations == [0; 9] {
                    save(&format!("hotbar{set}-gui{gui_scale}.png"), &b);
                }
                hotbar_frames += 1;
            }
        }
    }
    eprintln!("{hotbar_frames} hotbar frames identical");

    // ---- Tab lists ------------------------------------------------------------------------
    let mut rows: Vec<TabRow> = (0..40)
        .map(|i| {
            let mut row = TabRow::new(format!("Player_{i:02}"), i * 37 - 5);
            row.display = format!("\u{a7}{}Player_{i:02}", (b'a' + (i % 6) as u8) as char);
            row.team = format!("t{}", i % 7);
            row.score = Some(i * 3 - 4);
            row.spectator = i % 9 == 0;
            row.hat = i % 2 == 0;
            row
        })
        .collect();
    sort_rows(&mut rows);
    let objectives = [
        None,
        Some(TabObjective {
            display_name: "Kills".to_owned(),
            render_type: TabRenderType::Integer,
        }),
        Some(TabObjective {
            display_name: "Health".to_owned(),
            render_type: TabRenderType::Hearts,
        }),
    ];
    let mut tab_frames = 0;
    for gui_scale in [2, 3] {
        let width = (W as i32 + gui_scale - 1) / gui_scale;
        for ping in [TabPing::Bars, TabPing::Numbers] {
            for objective in &objectives {
                for show_heads in [false, true] {
                    let mut frames = Vec::new();
                    let mut batches = Vec::new();
                    for passes in [false, true] {
                        let mut overlay = TabListOverlay::new();
                        overlay.style = TabListStyle {
                            ping,
                            ..TabListStyle::VANILLA
                        };
                        overlay.set_pass_order(Some(passes));
                        overlay.update_with_counter(true, 10_000, 200);
                        let mut list = DrawList::new();
                        list.enable_solid_merging();
                        overlay.draw(
                            &mut list,
                            &mut font,
                            width,
                            &rows,
                            Some("\u{a7}eYou are playing on \u{a7}6MC.EXAMPLE.NET"),
                            Some("\u{a7}bRanks, Boosters & MORE! \u{a7}cSTORE.EXAMPLE.NET"),
                            objective.as_ref(),
                            show_heads,
                        );
                        batches.push(list.batches().len());
                        frames.push(render(&mut gpu, &rm, &atlas, gui_scale, list));
                    }
                    let what = format!(
                        "Tab at GUI {gui_scale}, {ping:?}, objective {:?}, heads {show_heads}",
                        objective.as_ref().map(|o| o.render_type)
                    );
                    let diff = differing(&frames[0], &frames[1]);
                    assert_eq!(diff, 0, "{what}: {diff} pixels differ");
                    // Never more batches; strictly fewer with vanilla's latency bars (an icon
                    // between every two names). With latency numbers and no heads or score
                    // icons, plates and text are already one batch in either order.
                    assert!(
                        batches[1] <= batches[0],
                        "{what}: {} pass batches vs {} in order",
                        batches[1],
                        batches[0]
                    );
                    if ping == TabPing::Bars {
                        assert!(
                            batches[1] < batches[0],
                            "{what}: {} pass batches vs {} in order",
                            batches[1],
                            batches[0]
                        );
                    }
                    if objective.is_none() && !show_heads {
                        save(&format!("tab-gui{gui_scale}-{ping:?}.png"), &frames[1]);
                    }
                    eprintln!("{what}: batches {} -> {}", batches[0], batches[1]);
                    tab_frames += 1;
                }
            }
        }
    }
    eprintln!("{tab_frames} Tab frames identical");

    // A list too wide for the screen keeps the row order (names may overflow their column):
    // the pass order must not change its batches at all.
    let narrow = 160;
    let mut counts = Vec::new();
    for passes in [false, true] {
        let mut overlay = TabListOverlay::new();
        overlay.set_pass_order(Some(passes));
        let mut list = DrawList::new();
        overlay.draw(&mut list, &mut font, narrow, &rows, None, None, None, false);
        counts.push((list.batches().len(), list.vertices().len()));
    }
    assert_eq!(
        counts[0], counts[1],
        "a capped list must keep the row order"
    );
    mc_render::entity::item::pin_item_glint_clock(None);
}
