//! GPU readbacks for the two mods that draw through mc-ui: the Crosshair mod's custom crosshair
//! and the Shiny Pots mod's square potion glint.
//!
//! Same technique as `hud_items.rs`: the real `UiPhase` into a headless target, read back.
//! `guiScale` is 1, so one GUI pixel is one framebuffer pixel. For both mods: the mod's look
//! reaches the framebuffer, and with the mod off (also after it was on) the draw is vanilla's.
//!
//! The crosshair tests need no assets. The shiny-pots tests skip (printing `SKIP:`) without the
//! vanilla jar; point `MC_JARS_DIR` at a `jars/` directory to override its location.

use std::path::{Path, PathBuf};

use mc_assets::{bake_all, install, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_data::item_by_name;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::modvisual::{set_shiny_pots, CustomCrosshair, ShinyPotsArea};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::container::render::draw_item_in_slot;
use mc_ui::screens::container::stack_ext::ItemStack;
use mc_ui::screens::hud::{CrosshairState, Hud};
use mc_ui::widget::draw::{UiBlend, UiDepth};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, UiTexture};

const W: u32 = 160;
const H: u32 = 120;
/// The crosshair tests' backdrop: `(51, 102, 153)`, so inversion is visible in every channel.
const BACKDROP: u32 = 0xFF33_6699;

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

fn ctx() -> ScreenCtx {
    ScreenCtx::new(ScaledResolution::new(W, H, 1, false), 0.0)
}

fn atlas_key() -> UiTexture {
    UiTexture::Resource(ResourceLocation::minecraft("mc-ui/m4-readback-atlas"))
}

/// Render `list` through the real UI phase over a black clear and read the frame back.
fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    atlas: Option<&mc_assets::Atlas>,
    list: DrawList,
) -> Vec<u8> {
    let mut phase = UiPhase::new(gpu, W, H, 1, false);
    if let Some(atlas) = atlas {
        phase
            .renderer
            .upload(gpu, atlas_key(), &atlas.texture, atlas.w, atlas.h);
    }
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(ScaledResolution::new(W, H, 1, false));
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    drop(frame);
    gpu.read_pixels().expect("readback")
}

fn at(pixels: &[u8], x: i32, y: i32) -> [u8; 3] {
    let i = ((y as u32 * W + x as u32) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

// ---- Crosshair ----------------------------------------------------------------------------

fn custom(color: Option<u32>) -> CustomCrosshair {
    CustomCrosshair {
        length: 4,
        gap: 2,
        thickness: 1,
        dot: false,
        color,
        show_in_f5: true,
    }
}

/// The backdrop, then the HUD's crosshair for `state`.
fn crosshair_list(hud: &Hud, state: CrosshairState) -> DrawList {
    let mut list = DrawList::new();
    list.rect(0, 0, W as i32, H as i32, BACKDROP);
    hud.draw_crosshair_for(&mut list, &ctx(), state);
    list
}

#[test]
fn with_the_mod_off_the_crosshair_draw_is_vanillas_also_after_it_was_on() {
    let vanilla = crosshair_list(&Hud::default(), CrosshairState::default());
    let mut hud = Hud::default();
    hud.custom_crosshair = Some(custom(None));
    let on = crosshair_list(&hud, CrosshairState::default());
    hud.custom_crosshair = None;
    let off = crosshair_list(&hud, CrosshairState::default());
    assert_eq!(off.vertices(), vanilla.vertices());
    assert_eq!(off.batches(), vanilla.batches());
    assert_ne!(on.vertices(), vanilla.vertices());
    // The custom crosshair is solid quads, never the icons sprite.
    assert!(!on
        .batches()
        .iter()
        .any(|b| b.texture == *UiTexture::icons()));
}

#[test]
fn the_custom_crosshair_inverts_or_colours_exactly_its_arms() {
    let Some(mut gpu) = gpu() else { return };
    let rm = ResourceManager::new();
    let (cx, cy) = (W as i32 / 2, H as i32 / 2);
    let backdrop = [0x33, 0x66, 0x99];
    let inverted = [0xFF - 0x33, 0xFF - 0x66, 0xFF - 0x99];

    let mut hud = Hud::default();
    hud.custom_crosshair = Some(custom(None));
    let list = crosshair_list(&hud, CrosshairState::default());
    let frame = render(&mut gpu, &rm, None, list);
    // Arm length 4, gap 2, thickness 1 around the centre pixel (cx, cy).
    for d in 3..=6 {
        assert_eq!(at(&frame, cx - d, cy), inverted, "left arm at -{d}");
        assert_eq!(at(&frame, cx + d, cy), inverted, "right arm at +{d}");
        assert_eq!(at(&frame, cx, cy - d), inverted, "top arm at -{d}");
        assert_eq!(at(&frame, cx, cy + d), inverted, "bottom arm at +{d}");
    }
    for (x, y) in [
        (cx, cy),
        (cx - 1, cy),
        (cx - 2, cy),
        (cx - 7, cy),
        (cx + 7, cy),
        (cx - 4, cy - 1),
        (cx - 4, cy + 1),
    ] {
        assert_eq!(at(&frame, x, y), backdrop, "({x}, {y}) is outside the arms");
    }
    // Exactly 16 pixels changed: four arms of four.
    let changed = (0..H as i32)
        .flat_map(|y| (0..W as i32).map(move |x| (x, y)))
        .filter(|&(x, y)| at(&frame, x, y) != backdrop)
        .count();
    assert_eq!(changed, 16);

    hud.custom_crosshair = Some(custom(Some(0xFFFF_0000)));
    let list = crosshair_list(&hud, CrosshairState::default());
    let frame = render(&mut gpu, &rm, None, list);
    assert_eq!(at(&frame, cx - 4, cy), [255, 0, 0]);
    assert_eq!(at(&frame, cx, cy + 5), [255, 0, 0]);
    assert_eq!(at(&frame, cx, cy), backdrop);
}

#[test]
fn the_custom_crosshair_keeps_vanillas_gates_and_its_third_person_switch() {
    let mut hud = Hud::default();
    hud.custom_crosshair = Some(CustomCrosshair {
        show_in_f5: false,
        ..custom(None)
    });
    let f3_state = CrosshairState {
        show_debug_info: true,
        ..CrosshairState::default()
    };
    let f5_state = CrosshairState {
        third_person: true,
        ..CrosshairState::default()
    };
    let backdrop_only = crosshair_list(&Hud::default(), f3_state);
    // F3 hides it exactly as it hides vanilla's.
    let f3 = crosshair_list(&hud, f3_state);
    assert_eq!(f3.vertices(), backdrop_only.vertices());
    // "Show in third person" off hides it in F5 and nowhere else.
    let f5 = crosshair_list(&hud, f5_state);
    assert_eq!(f5.vertices(), backdrop_only.vertices());
    let first_person = crosshair_list(&hud, CrosshairState::default());
    assert_ne!(first_person.vertices(), backdrop_only.vertices());
    // On (the default), it shows in F5 like vanilla's.
    hud.custom_crosshair = Some(custom(None));
    let f5 = crosshair_list(&hud, f5_state);
    assert_eq!(f5.vertices(), first_person.vertices());
}

// ---- Shiny Pots ---------------------------------------------------------------------------

/// A drinkable potion of Regeneration (metadata 8193): it has an effect, so it glints.
fn regeneration_potion() -> ItemStack {
    let id = item_by_name("potion").expect("potion is a 1.8.9 item");
    ItemStack {
        item_id: id.0 as i16,
        count: 1,
        damage: 8193,
        nbt: None,
    }
}

const SLOT: (i32, i32) = (60, 40);

fn potion_list(rm: &ResourceManager, area: Option<ShinyPotsArea>) -> DrawList {
    let mut font = mc_ui::font::FontRenderer::load(rm, false, false).expect("font");
    let mut list = DrawList::new();
    set_shiny_pots(area);
    draw_item_in_slot(
        &mut list,
        &atlas_key(),
        &mut font,
        SLOT.0,
        SLOT.1,
        &regeneration_potion(),
    );
    set_shiny_pots(None);
    list
}

#[test]
fn shiny_pots_swaps_the_bottle_foil_for_a_square_and_off_is_the_bottle_foil() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let (baked, _) = bake_all(&rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    install(baked);

    // CPU: off is vanilla's model foil (depth-equal), on is the square (no depth test).
    let glints = |list: &DrawList| -> Vec<UiDepth> {
        list.batches()
            .iter()
            .filter(|b| b.blend == UiBlend::Glint)
            .map(|b| b.depth)
            .collect()
    };
    let off = potion_list(&rm, None);
    assert!(
        !glints(&off).is_empty(),
        "a regeneration potion glints in vanilla"
    );
    assert!(glints(&off).iter().all(|&d| d == UiDepth::Equal));
    let square = potion_list(&rm, Some(ShinyPotsArea::ItemBox));
    assert!(
        !glints(&square).is_empty(),
        "the square glint was not drawn"
    );
    assert!(glints(&square).iter().all(|&d| d == UiDepth::Overlay));
    // Off after on: identical geometry to a never-touched draw (the clock only moves UVs).
    let positions = |list: &DrawList| list.vertices().iter().map(|v| v.pos).collect::<Vec<_>>();
    assert_eq!(positions(&potion_list(&rm, None)), positions(&off));

    // GPU: which pixels the glint reaches.
    let Some(mut gpu) = gpu() else { return };
    let off_px = render(&mut gpu, &rm, Some(&atlas), off);
    let box_px = render(&mut gpu, &rm, Some(&atlas), square);
    let slot_px = render(
        &mut gpu,
        &rm,
        Some(&atlas),
        potion_list(&rm, Some(ShinyPotsArea::WholeSlot)),
    );
    let black = [0, 0, 0];
    let (x0, y0) = SLOT;
    let in_box = |x: i32, y: i32| (x0..x0 + 16).contains(&x) && (y0..y0 + 16).contains(&y);
    let in_slot = |x: i32, y: i32| (x0 - 2..x0 + 18).contains(&x) && (y0 - 2..y0 + 18).contains(&y);
    // The empty texels of the icon: black with the mod off (vanilla's foil stops at the bottle).
    let mut empty_texels = 0;
    let (mut lit_box, mut lit_slot) = (0, 0);
    let (mut ring, mut lit_ring) = (0, 0);
    for y in y0 - 4..y0 + 20 {
        for x in x0 - 4..x0 + 20 {
            let off = at(&off_px, x, y);
            if !in_slot(x, y) {
                assert_eq!(off, black, "({x}, {y}) is outside the slot");
                assert_eq!(
                    at(&box_px, x, y),
                    black,
                    "item box: ({x}, {y}) outside the slot"
                );
                assert_eq!(
                    at(&slot_px, x, y),
                    black,
                    "whole slot: ({x}, {y}) outside it"
                );
            } else if !in_box(x, y) {
                assert_eq!(
                    off, black,
                    "vanilla draws nothing in the slot's border ({x}, {y})"
                );
                assert_eq!(
                    at(&box_px, x, y),
                    black,
                    "item box stops at the icon ({x}, {y})"
                );
                ring += 1;
                lit_ring += usize::from(at(&slot_px, x, y) != black);
            } else if off == black {
                empty_texels += 1;
                lit_box += usize::from(at(&box_px, x, y) != black);
                lit_slot += usize::from(at(&slot_px, x, y) != black);
            }
        }
    }
    println!(
        "shiny pots: item box lit {lit_box}/{empty_texels} empty icon texels, whole slot \
         {lit_slot}/{empty_texels}, border {lit_ring}/{ring}"
    );
    assert!(empty_texels > 60, "only {empty_texels} empty icon texels");
    // The glint texture has dark streaks, so not every pixel lights; most must.
    assert!(
        lit_box * 2 > empty_texels,
        "item box lit {lit_box} of {empty_texels} empty icon texels"
    );
    assert!(
        lit_slot * 2 > empty_texels,
        "whole slot lit {lit_slot} of {empty_texels}"
    );
    assert!(
        lit_ring * 2 > ring,
        "whole slot lit {lit_ring} of its {ring}-pixel border"
    );
}
