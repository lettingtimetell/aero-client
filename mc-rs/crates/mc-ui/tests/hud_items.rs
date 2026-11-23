//! Headless GPU tests for the HUD's hotbar **items**.
//!
//! Same technique as `headless_screens.rs` and `container_render.rs`: render through the real
//! `UI 800` phase with `mc_render::Gpu::headless` and read the framebuffer back, then compare
//! against the *source texels* of the real `1.8.9.jar`. "The hotbar draws items now" is only
//! worth saying if the icon's pixels are the ones vanilla samples.
//!
//! The comparison target is deliberately `assets/minecraft/textures/items/*.png` **decoded
//! straight from the jar**, not the stitched atlas: that checks the whole chain (model bake →
//! atlas UV → GUI projection → sampler) against the original artwork rather than against
//! another part of the same pipeline. It also means a re-stitch that moves the sprite cannot
//! silently invalidate this test.
//!
//! `guiScale` is 1 throughout, so one GUI pixel is one framebuffer pixel, and the UI sampler is
//! `Nearest`, so a pixel-aligned 16×16 blit of a 16×16 sprite is texel-exact.
//!
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent.

use std::path::{Path, PathBuf};

use mc_assets::{bake_all, install, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_data::item_by_name;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::container::render::{
    draw_item_icon_tinted, draw_item_icon_tinted_at_z_level, draw_item_in_slot,
    GUI_SLOT_ITEM_RENDER_Z_LEVEL,
};
use mc_ui::screens::container::stack_ext::ItemStack;
use mc_ui::screens::hud::{Hud, PlayerHud};
use mc_ui::widget::draw::UiBlend;
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, UiTexture};

const W: u32 = 480;
const H: u32 = 320;

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

/// Bake and install the real jar's models + atlas, once per test process.
fn baked(rm: &ResourceManager) -> mc_assets::Atlas {
    let (baked, _report) = bake_all(rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    install(baked);
    atlas
}

fn texture(rm: &ResourceManager, path: &str) -> (Vec<u8>, u32, u32) {
    let bytes = rm
        .load(&ResourceLocation::minecraft(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .to_rgba8();
    let (w, h) = (img.width(), img.height());
    (img.into_raw(), w, h)
}

fn texel(src: &(Vec<u8>, u32, u32), x: u32, y: u32) -> [u8; 4] {
    let i = ((y * src.1 + x) * 4) as usize;
    [src.0[i], src.0[i + 1], src.0[i + 2], src.0[i + 3]]
}

struct Frame {
    pixels: Vec<u8>,
}

impl Frame {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * W + x) * 4) as usize;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }
}

fn atlas_key() -> UiTexture {
    UiTexture::Resource(ResourceLocation::minecraft("mc-ui/hud-hotbar-atlas"))
}

fn render(gpu: &mut Gpu, rm: &ResourceManager, atlas: &mc_assets::Atlas, list: DrawList) -> Frame {
    let mut phase = UiPhase::new(gpu, W, H, 1, false);
    phase
        .renderer
        .upload(gpu, atlas_key(), &atlas.texture, atlas.w, atlas.h);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(ScaledResolution::new(W, H, 1, false));
    phase.submit(list);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    Frame {
        pixels: gpu.read_pixels().expect("readback"),
    }
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(ScaledResolution::new(W, H, 1, false), 0.0)
}

fn stack(name: &str, count: i32) -> ItemStack {
    let id = item_by_name(name).unwrap_or_else(|| panic!("{name} is a real 1.8.9 item"));
    ItemStack {
        item_id: id.0 as i16,
        count,
        damage: 0,
        nbt: None,
    }
}

/// The whole point of the task line: a stack in the hotbar draws its real icon, on the exact
/// texels of the exact artwork, at `hotbar_item_pos`.
#[test]
fn a_hotbar_stack_lands_texel_for_texel_on_its_own_item_texture() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let atlas = baked(&rm);
    // A stick is a plain `Item`, so vanilla bakes it through `builtin/generated` — the flat
    // 2-D path, whose projection is an exact rectangle (see `render.rs`'s module doc).
    let source = texture(&rm, "textures/items/stick.png");
    assert_eq!(
        (source.1, source.2),
        (16, 16),
        "stick.png is a plain 16x16 sprite"
    );

    let hud = Hud::new();
    let c = ctx();
    let mut state = PlayerHud::default();
    state.hotbar[0] = Some(stack("stick", 1));

    let mut list = DrawList::new();
    let mut font = mc_ui::font::FontRenderer::load(&rm, false, false).expect("font");
    hud.draw_hotbar_items(&mut list, &mut font, &atlas_key(), &c, &state);
    assert!(!list.is_empty(), "the hotbar emitted no item geometry");

    let f = render(&mut gpu, &rm, &atlas, list);
    let (x0, y0) = Hud::hotbar_item_pos(&c, 0);
    let (x0, y0) = (x0 as u32, y0 as u32);

    let mut compared = 0;
    for dy in 0..16u32 {
        for dx in 0..16u32 {
            let src = texel(&source, dx, dy);
            if src[3] != 255 {
                continue; // anti-aliased/empty texels blend with the clear colour
            }
            compared += 1;
            assert_eq!(
                f.at(x0 + dx, y0 + dy),
                src,
                "hotbar slot 0, icon texel ({dx}, {dy})"
            );
        }
    }
    assert!(
        compared > 20,
        "only {compared} opaque stick texels compared; the icon is not landing"
    );
    // Nothing leaked into the neighbouring slot.
    let (x1, y1) = Hud::hotbar_item_pos(&c, 1);
    for dy in 0..16u32 {
        for dx in 0..16u32 {
            assert_eq!(
                f.at(x1 as u32 + dx, y1 as u32 + dy),
                [0, 0, 0, 255],
                "slot 1 is empty and must stay black"
            );
        }
    }
}

/// A block stack takes the tilted three-face path and still paints inside its own 16×16 box,
/// with the top face brighter than the right face — vanilla's classic GUI cube shading.
#[test]
fn a_block_stack_draws_a_shaded_cube_inside_its_slot() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let atlas = baked(&rm);

    let hud = Hud::new();
    let c = ctx();
    let mut state = PlayerHud::default();
    // A stack of one, so `renderItemOverlays`' white count text cannot contaminate the shading
    // measurement below; slot 8 carries the counted stack instead.
    state.hotbar[4] = Some(stack("stone", 1));
    state.hotbar[8] = Some(stack("stone", 64));

    let mut list = DrawList::new();
    let mut font = mc_ui::font::FontRenderer::load(&rm, false, false).expect("font");
    hud.draw_hotbar_items(&mut list, &mut font, &atlas_key(), &c, &state);
    let f = render(&mut gpu, &rm, &atlas, list);

    let (x, y) = Hud::hotbar_item_pos(&c, 4);
    let (x, y) = (x as u32, y as u32);
    let lum = |p: [u8; 4]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
    let mut painted = 0;
    let mut upper = (0u32, 0u32);
    let mut lower = (0u32, 0u32);
    for dy in 0..16u32 {
        for dx in 0..16u32 {
            let p = f.at(x + dx, y + dy);
            if p == [0, 0, 0, 255] {
                continue;
            }
            painted += 1;
            if dy < 8 {
                upper = (upper.0 + lum(p), upper.1 + 1);
            } else {
                lower = (lower.0 + lum(p), lower.1 + 1);
            }
        }
    }
    // The tilted cube is a hexagon inscribed in the 16×16 box, so a bit under half the box is
    // background even when everything is drawn.
    assert!(painted > 100, "only {painted} of 256 pixels painted");
    assert!(upper.1 > 20 && lower.1 > 20, "{upper:?} {lower:?}");
    // The cube's lit top face fills the upper half; the two dimmer side faces fill the lower.
    let upper_mean = upper.0 / upper.1;
    let lower_mean = lower.0 / lower.1;
    assert!(
        upper_mean > lower_mean,
        "the top face should outshine the sides: {upper_mean} vs {lower_mean}"
    );

    // The counted stack draws its size as white text in the slot's lower right.
    let (cx, cy) = Hud::hotbar_item_pos(&c, 8);
    let (cx, cy) = (cx as u32, cy as u32);
    let white = (0..16u32)
        .flat_map(|dy| (0..18u32).map(move |dx| (dx, dy)))
        .filter(|(dx, dy)| f.at(cx + dx, cy + dy) == [255, 255, 255, 255])
        .count();
    assert!(
        white > 5,
        "the stack count text is missing ({white} white pixels)"
    );
}

/// A white caller tint must preserve the ordinary stack renderer, including intrinsic item
/// colours and metadata/builtin model selection. Raw model helpers intentionally omit those.
#[test]
fn tinting_with_opaque_white_is_byte_identical_to_the_untinted_paths() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let _atlas = baked(&rm);
    let key = atlas_key();
    let mut font = mc_ui::font::FontRenderer::load(&rm, false, false).expect("font");

    // Every registered item, so the equality is proved over the whole real registry rather than
    // over a hand-picked handful — the two branches must agree everywhere or not at all.
    let mut checked = (0usize, 0usize);
    for props in mc_data::items() {
        if !props.registered {
            continue;
        }
        let s = ItemStack {
            item_id: props.id.0 as i16,
            count: 1,
            damage: 0,
            nbt: None,
        };
        let mut untinted = DrawList::new();
        draw_item_in_slot(&mut untinted, &key, &mut font, 7, 11, &s);
        let mut tinted = DrawList::new();
        draw_item_icon_tinted_at_z_level(
            &mut tinted,
            &key,
            7,
            11,
            &s,
            0xFFFF_FFFF,
            GUI_SLOT_ITEM_RENDER_Z_LEVEL,
        );
        assert_eq!(untinted.batches(), tinted.batches(), "{}", props.name);
        for batch in untinted.batches() {
            for index in batch.range.clone() {
                let a = &untinted.vertices()[index];
                let b = &tinted.vertices()[index];
                if batch.blend == UiBlend::Glint {
                    // The two public calls sample wall-clock foil UVs independently. Tint must
                    // still preserve its geometry/colour; ordinary texture UVs remain exact.
                    assert_eq!(a.pos, b.pos, "{}", props.name);
                    assert_eq!(a.color, b.color, "{}", props.name);
                } else {
                    assert_eq!(a, b, "{}", props.name);
                }
            }
        }
        if !untinted.is_empty() {
            if props.block.is_some() {
                checked.1 += 1;
            } else {
                checked.0 += 1;
            }
        }
    }
    assert!(
        checked.0 > 100 && checked.1 > 100,
        "both branches must be exercised on real geometry, got {checked:?}"
    );
}

/// And a non-white tint really does darken every channel, on both branches.
#[test]
fn a_dim_tint_darkens_the_icon_on_both_branches() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let _atlas = baked(&rm);
    let key = atlas_key();
    for name in ["stick", "stone"] {
        let s = stack(name, 1);
        let mut bright = DrawList::new();
        draw_item_icon_tinted(&mut bright, &key, 0, 0, &s, 0xFFFF_FFFF);
        let mut dim = DrawList::new();
        draw_item_icon_tinted(&mut dim, &key, 0, 0, &s, 0xFF19_1919);
        assert_eq!(bright.vertices().len(), dim.vertices().len());
        for (b, d) in bright.vertices().iter().zip(dim.vertices()) {
            assert_eq!(b.pos, d.pos, "{name}: the tint must not move geometry");
            assert_eq!(b.uv, d.uv);
            for ch in 0..3 {
                assert!(d.color[ch] < b.color[ch] + 1e-6);
            }
            assert!(d.color[0] < 0.2, "{name}: 0x19 tint is barely lit");
        }
    }
}
