//! Headless GPU renders of the HUD elements: `fps`, `cps`, `ping`,
//! `coordinates`, `armor_status`, `potion_status`, `reach_display`, `item_info` and
//! `keystrokes`, all drawn through the real `ModHud` pipeline at their default placements with
//! the real font and textures of the 1.8.9 jar, rendered by the `UI 800` phase and read back.
//!
//! What the frame must show (GUI scale 2, a 1080p window, so 960×540 GUI pixels):
//! * every element is drawn, inside the screen, and the default layout has no overlaps
//!   (research `HUD-EDITOR.md`'s table);
//! * the Speed icon lands texel for texel on `inventory.png`'s cell, and the helmet icon on
//!   `diamond_helmet.png`;
//! * keystrokes paints its held keys in the pressed colour and the others in the released one.
//!
//! Set `MC_M1_RENDER_PNG=<file.png>` to also save the frame for a human look.
//!
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent.

use std::path::{Path, PathBuf};

use mc_assets::{bake_all, install, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_mods::ModSettings;
use mc_nbt::{Nbt, NbtCompound};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::modhud::{HudCtx, HudData, HudEffect, HudKeys, HudReach, ModHud};
use mc_ui::screens::container::default_atlas_texture;
use mc_ui::screens::container::stack_ext::ItemStack;
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase};

const SCALE: u32 = 2;
const W: u32 = 960 * SCALE;
const H: u32 = 540 * SCALE;

const M1_MODS: [&str; 9] = [
    "fps",
    "cps",
    "ping",
    "coordinates",
    "armor_status",
    "potion_status",
    "reach_display",
    "item_info",
    "keystrokes",
];

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

    /// The pixel at the centre of GUI pixel `(x, y)`.
    fn gui(&self, x: f32, y: f32) -> [u8; 4] {
        self.at(x as u32 * SCALE + SCALE / 2, y as u32 * SCALE + SCALE / 2)
    }
}

/// The colour channels only: a translucent quad also writes its own alpha (`GL_ONE, GL_ZERO`
/// for alpha), so a pixel under the 0x6F box reads alpha 111 even where it stays black.
fn rgb(pixel: [u8; 4]) -> [u8; 3] {
    [pixel[0], pixel[1], pixel[2]]
}

fn stack(item_id: i16, damage: i16) -> ItemStack {
    ItemStack {
        item_id,
        count: 1,
        damage,
        nbt: None,
    }
}

/// A diamond sword with Sharpness V and Fire Aspect II.
fn enchanted_sword() -> ItemStack {
    let list = [(16_i16, 5_i16), (20, 2)]
        .iter()
        .map(|&(id, lvl)| {
            let mut entry = NbtCompound::new();
            entry.insert("id", Nbt::Short(id));
            entry.insert("lvl", Nbt::Short(lvl));
            Nbt::Compound(entry)
        })
        .collect();
    let mut root = NbtCompound::new();
    root.insert("ench", Nbt::list(list));
    ItemStack {
        nbt: Some(root),
        ..stack(276, 61)
    }
}

/// A mid-fight SkyWars frame: full diamond, an enchanted sword, two effects, a recent hit, W and
/// the left button held.
fn fight_data() -> HudData {
    HudData {
        now_ms: 10_000,
        tick: 200,
        fps: 240,
        cps_left: 12,
        cps_right: 4,
        ping_ms: Some(45),
        position: [128.7, 64.0, -301.2],
        yaw: 225.0,
        pitch: 10.0,
        armor: [
            Some(stack(310, 0)),
            Some(stack(311, 100)),
            Some(stack(312, 300)),
            Some(stack(313, 400)),
        ],
        held: Some(enchanted_sword()),
        effects: vec![
            HudEffect {
                id: 1,
                amplifier: 1,
                duration_ticks: 90 * 20,
                duration_max: false,
            },
            HudEffect {
                id: 5,
                amplifier: 0,
                duration_ticks: 30 * 20,
                duration_max: false,
            },
        ],
        reach: Some(HudReach {
            distance: 2.87,
            at_ms: 9_500,
        }),
        keys: HudKeys {
            forward: true,
            attack: true,
            ..HudKeys::default()
        },
        arrow_count: 23,
        held_total: 1,
        ..HudData::default()
    }
}

fn all_m1_on() -> ModSettings {
    let mut settings = ModSettings::new();
    for id in M1_MODS {
        settings.set_enabled(id, true);
    }
    settings
}

/// The on-screen rect of every M1 slot after `hud.draw`, by mod id.
fn m1_rects(hud: &ModHud) -> Vec<(&'static str, mc_mods::HudRect)> {
    let catalog = mc_mods::catalog::all();
    hud.slots()
        .iter()
        .filter_map(|slot| {
            let id = catalog[slot.mod_index].id;
            M1_MODS.contains(&id).then(|| {
                let rect = slot.rect.unwrap_or_else(|| panic!("{id} was not drawn"));
                assert!(!slot.placeholder, "{id} drew a placeholder, not itself");
                (id, rect)
            })
        })
        .collect()
}

#[test]
fn every_m1_element_renders_at_its_default_place() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let (baked, _report) = bake_all(&rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    install(baked);

    let settings = all_m1_on();
    let mut hud = ModHud::new();
    hud.sync(&settings);
    let data = fight_data();
    let mut font = mc_ui::font::FontRenderer::load(&rm, false, false).expect("font");
    let ctx = HudCtx {
        data: &data,
        settings: &settings,
        screen: ((W / SCALE) as f32, (H / SCALE) as f32),
        pixel_scale: SCALE as f32,
        editing: false,
    };
    let mut list = DrawList::new();
    hud.draw(&ctx, &mut font, &mut list);

    // Every element is drawn, on screen, and none overlaps another.
    let rects = m1_rects(&hud);
    assert_eq!(rects.len(), M1_MODS.len());
    let screen = ctx.screen;
    for (id, r) in &rects {
        assert!(
            r.x >= 0.0 && r.y >= 0.0 && r.x + r.w <= screen.0 && r.y + r.h <= screen.1,
            "{id} leaves the screen: {r:?}"
        );
    }
    for (i, (a, ra)) in rects.iter().enumerate() {
        for (b, rb) in &rects[i + 1..] {
            let apart = ra.x + ra.w <= rb.x
                || rb.x + rb.w <= ra.x
                || ra.y + ra.h <= rb.y
                || rb.y + rb.h <= ra.y;
            assert!(apart, "{a} {ra:?} overlaps {b} {rb:?}");
        }
    }

    let mut phase = UiPhase::new(&gpu, W, H, SCALE as i32, false);
    phase.renderer.upload(
        &gpu,
        default_atlas_texture(),
        &atlas.texture,
        atlas.w,
        atlas.h,
    );
    let missing = phase.renderer.ensure_list_textures(&gpu, &rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(ScaledResolution::new(W, H, SCALE as i32, false));
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, target.view(), &Camera::default(), 0.0);
    let frame = Frame {
        pixels: gpu.read_pixels().expect("readback"),
    };
    if let Ok(path) = std::env::var("MC_M1_RENDER_PNG") {
        image::RgbaImage::from_raw(W, H, frame.pixels.clone())
            .expect("frame size")
            .save(&path)
            .expect("save the frame");
        eprintln!("saved {path}");
    }
    let rect = |id: &str| rects.iter().find(|(m, _)| *m == id).unwrap().1;

    // Every element put something on the black frame inside its rect.
    for (id, r) in &rects {
        let (x0, x1) = (r.x as u32 * SCALE, (r.x + r.w).ceil() as u32 * SCALE);
        let (y0, y1) = (r.y as u32 * SCALE, (r.y + r.h).ceil() as u32 * SCALE);
        let lit = (y0..y1)
            .flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .filter(|&(x, y)| rgb(frame.at(x, y)) != [0, 0, 0])
            .count();
        assert!(lit > 40, "{id} lit only {lit} pixels");
    }

    // The Speed icon (cell 0 of the v = 198 grid), texel for texel: 3 px of box padding, 1 px
    // into the row. Opaque texels only, as the translucent ones blend with the box.
    let inventory = texture(&rm, "textures/gui/container/inventory.png");
    let potions = rect("potion_status");
    let (ix, iy) = (potions.x as u32 + 3, potions.y as u32 + 4);
    let mut compared = 0;
    for dy in 0..18 {
        for dx in 0..18 {
            let src = texel(&inventory, dx, 198 + dy);
            if src[3] == 255 {
                compared += 1;
                let px = frame.gui((ix + dx) as f32, (iy + dy) as f32);
                assert_eq!(px, src, "Speed icon texel ({dx}, {dy})");
            }
        }
    }
    assert!(
        compared > 40,
        "only {compared} opaque Speed texels compared"
    );

    // The undamaged diamond helmet, top row of armor status (no count, no bar), likewise.
    let helmet = texture(&rm, "textures/items/diamond_helmet.png");
    let armor = rect("armor_status");
    let mut compared = 0;
    for dy in 0..16 {
        for dx in 0..16 {
            let src = texel(&helmet, dx, dy);
            if src[3] == 255 {
                compared += 1;
                let px = frame.gui(armor.x + dx as f32, armor.y + dy as f32);
                assert_eq!(px, src, "helmet texel ({dx}, {dy})");
            }
        }
    }
    assert!(
        compared > 40,
        "only {compared} opaque helmet texels compared"
    );

    // Keystrokes: W and LMB held (0x6FFFFFFF over black is 111 grey), S and RMB not
    // (0x6F000000 over black stays black). Sampled in a key corner, clear of its label.
    let keys = rect("keystrokes");
    let grey = |p: [u8; 4]| p[0].abs_diff(111) <= 1 && p[1] == p[0] && p[2] == p[0];
    let held_w = frame.gui(keys.x + 21.0, keys.y + 1.0);
    assert!(grey(held_w), "W is held: {held_w:?}");
    let held_lmb = frame.gui(keys.x + 1.0, keys.y + 41.0);
    assert!(grey(held_lmb), "LMB is held: {held_lmb:?}");
    assert_eq!(
        rgb(frame.gui(keys.x + 21.0, keys.y + 21.0)),
        [0, 0, 0],
        "S is not"
    );
    assert_eq!(
        rgb(frame.gui(keys.x + 31.0, keys.y + 41.0)),
        [0, 0, 0],
        "RMB is not"
    );
}

/// The HUD editor needs something to grab: with no data at all, every M1 element still lays
/// itself out (a sample or live zeros), never the framework's placeholder box.
#[test]
fn the_editor_shows_every_m1_element_without_game_data() {
    let settings = all_m1_on();
    let mut hud = ModHud::new();
    hud.sync(&settings);
    let data = HudData::default();
    let ctx = HudCtx {
        data: &data,
        settings: &settings,
        screen: (960.0, 540.0),
        pixel_scale: 2.0,
        editing: true,
    };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let mut font = mc_ui::font::FontRenderer::load(&rm, false, false).expect("font");
    let mut list = DrawList::new();
    hud.draw(&ctx, &mut font, &mut list);
    assert_eq!(m1_rects(&hud).len(), M1_MODS.len());
}
