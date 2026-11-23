//! The drag-split preview in the creative screen and in the survival
//! inventory, rendered through the real `UI 800` phase with the baked item atlas and read back.
//!
//! java: `GuiContainer.drawSlot:243-273` (each swept slot shows the stack it would receive over a
//! `0x80FFFFFF` rect, with a yellow count when the item or the slot caps it; a lone swept slot is
//! not drawn at all), `updateDragSplitting:309-336` and `drawScreen:145-170` (the held stack shows
//! `dragSplittingRemnant`, a yellow `0` when nothing would be left). `GuiContainerCreative`
//! inherits all of it, so the creative screen (Inventory tab and item tabs) must match the
//! survival inventory.
//!
//! Set `CI_PNG_DIR` to also write every shot (3x, nearest neighbour) and the creative/survival
//! pairs side by side. Skips (loudly) with no wgpu adapter or no vanilla jar.

use std::path::{Path, PathBuf};

use image::RgbaImage;
use mc_assets::{Atlas, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::{Screen, ScreenCtx};
use mc_ui::screens::container::creative::CreativeTab;
use mc_ui::screens::container::creative_screen::{CreativeScreen, CreativeSlot};
use mc_ui::screens::container::screen::ContainerScreen;
use mc_ui::screens::container::ItemStack;
use mc_ui::widget::{DrawList, Modifiers, ScaledResolution, UiPhase, UiTexture};

const W: u32 = 256;
const H: u32 = 256;
const SCALE: u32 = 3;

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

fn jar_pack() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

/// Bake the real jar's models and install them, handing back the stitched atlas.
fn baked_atlas(rm: &ResourceManager) -> Atlas {
    let (baked, _report) = mc_assets::bake_all(rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    mc_assets::install(baked);
    atlas
}

fn resolution() -> ScaledResolution {
    let r = ScaledResolution::new(W, H, 1, false);
    assert_eq!(r.scale_factor(), 1);
    r
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(resolution(), 0.0)
}

fn atlas_key() -> UiTexture {
    UiTexture::Resource(ResourceLocation::minecraft("mc-ui/ci-drag-atlas"))
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

struct Rig {
    gpu: Gpu,
    rm: ResourceManager,
    atlas: Atlas,
    font: FontRenderer,
}

impl Rig {
    fn new() -> Option<Self> {
        let gpu = gpu()?;
        let Some(rm) = jar_pack() else {
            eprintln!("SKIP: vanilla jar not found");
            return None;
        };
        let atlas = baked_atlas(&rm);
        let font = FontRenderer::load(&rm, false, false).expect("font loads");
        Some(Self {
            gpu,
            rm,
            atlas,
            font,
        })
    }

    /// Draw `screen` with the mouse at `mouse` and read the framebuffer back.
    fn shoot(&mut self, screen: &mut dyn Screen, mouse: (i32, i32)) -> Vec<u8> {
        let mut list = DrawList::new();
        screen.draw(&mut list, &mut self.font, &ctx(), mouse);

        let mut phase = UiPhase::new(&self.gpu, W, H, 1, false);
        phase.renderer.upload(
            &self.gpu,
            atlas_key(),
            &self.atlas.texture,
            self.atlas.w,
            self.atlas.h,
        );
        let missing = phase
            .renderer
            .ensure_list_textures(&self.gpu, &self.rm, &list);
        assert!(missing.is_empty(), "missing textures: {missing:?}");
        phase.set_resolution(resolution());
        phase.submit(list);

        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::UI, Box::new(phase));
        let frame = self.gpu.acquire().expect("acquire offscreen frame");
        phases.render(&self.gpu, frame.view(), &Camera::default(), 0.0);
        self.gpu.read_pixels().expect("readback")
    }
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

/// How many pixels differ between two shots inside the 16x16 cell whose top-left is `(x, y)`.
fn cell_diff(a: &[u8], b: &[u8], (x, y): (i32, i32)) -> usize {
    (0..16u32)
        .flat_map(|dy| (0..16u32).map(move |dx| (dx, dy)))
        .filter(|(dx, dy)| {
            let (px, py) = (x as u32 + dx, y as u32 + dy);
            pixel(a, px, py) != pixel(b, px, py)
        })
        .count()
}

/// How many pixels of the cell are brighter (by channel sum) in `b` than in `a`: the
/// `0x80FFFFFF` rect lightens the slot background around the ghost stack.
fn cell_lightened(a: &[u8], b: &[u8], (x, y): (i32, i32)) -> usize {
    (0..16u32)
        .flat_map(|dy| (0..16u32).map(move |dx| (dx, dy)))
        .filter(|(dx, dy)| {
            let (px, py) = (x as u32 + dx, y as u32 + dy);
            let sum = |p: [u8; 4]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
            sum(pixel(b, px, py)) > sum(pixel(a, px, py)) + 60
        })
        .count()
}

fn upscaled(px: &[u8]) -> RgbaImage {
    let src = RgbaImage::from_raw(W, H, px.to_vec()).expect("framebuffer size");
    let mut out = RgbaImage::new(W * SCALE, H * SCALE);
    for (x, y, p) in out.enumerate_pixels_mut() {
        *p = *src.get_pixel(x / SCALE, y / SCALE);
        p.0[3] = 255;
    }
    out
}

fn png_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("CI_PNG_DIR")?);
    std::fs::create_dir_all(&dir).expect("create CI_PNG_DIR");
    Some(dir)
}

fn save(name: &str, px: &[u8]) {
    if let Some(dir) = png_dir() {
        upscaled(px)
            .save(dir.join(format!("{name}.png")))
            .expect("write png");
    }
}

fn save_pair(name: &str, left: &[u8], right: &[u8]) {
    if let Some(dir) = png_dir() {
        let (l, r) = (upscaled(left), upscaled(right));
        let gap = 12;
        let mut out = RgbaImage::from_pixel(
            l.width() + gap + r.width(),
            l.height(),
            image::Rgba([40, 40, 40, 255]),
        );
        image::imageops::replace(&mut out, &l, 0, 0);
        image::imageops::replace(&mut out, &r, i64::from(l.width() + gap), 0);
        out.save(dir.join(format!("{name}.png")))
            .expect("write png");
    }
}

/// One drag: the held stack, what the swept slots hold beforehand, the button and the slots
/// swept (by player-container slot id, in order).
struct Drag {
    cursor: ItemStack,
    existing: Vec<(u16, ItemStack)>,
    button: i32,
    slots: Vec<u16>,
}

fn creative_on(tab: CreativeTab, drag: &Drag) -> CreativeScreen {
    let mut screen = CreativeScreen::new();
    screen.atlas = atlas_key();
    screen.init(&ctx());
    screen.set_tab(tab.index().expect("registered tab"));
    for (slot, item) in &drag.existing {
        screen
            .player
            .set_slot_contents(usize::from(*slot), Some(item.clone()));
    }
    screen.player.set_cursor(Some(drag.cursor.clone()));
    screen
}

fn creative_centre(screen: &CreativeScreen, id: u16) -> (i32, i32) {
    let (_, x, y) = screen
        .slot_rects()
        .into_iter()
        .find(|(slot, _, _)| *slot == CreativeSlot::Player(id))
        .expect("slot is on this tab");
    (screen.gui_left() + x + 8, screen.gui_top() + y + 8)
}

/// Press on the first slot, sweep every slot, and stay on the last one (the button held).
fn creative_sweep(screen: &mut CreativeScreen, drag: &Drag) -> (i32, i32) {
    let first = creative_centre(screen, drag.slots[0]);
    let _ = screen.mouse_clicked_with_mods(first.0, first.1, drag.button, Modifiers::NONE);
    let mut at = first;
    for id in &drag.slots {
        at = creative_centre(screen, *id);
        screen.mouse_dragged(at.0, at.1, drag.button, 0);
    }
    at
}

fn survival_with(drag: &Drag) -> ContainerScreen {
    let mut screen = ContainerScreen::player(0);
    screen.atlas = atlas_key();
    screen.init(&ctx());
    for (slot, item) in &drag.existing {
        screen
            .container
            .set_slot_contents(usize::from(*slot), Some(item.clone()));
    }
    screen.container.set_cursor(Some(drag.cursor.clone()));
    screen
}

fn survival_centre(screen: &ContainerScreen, id: u16) -> (i32, i32) {
    let slot = &screen.container.slots()[usize::from(id)];
    (
        screen.gui_left() + slot.x + 8,
        screen.gui_top() + slot.y + 8,
    )
}

fn survival_sweep(screen: &mut ContainerScreen, drag: &Drag) -> (i32, i32) {
    let first = survival_centre(screen, drag.slots[0]);
    let _ = screen.mouse_clicked_with_mods(first.0, first.1, drag.button, Modifiers::NONE);
    let mut at = first;
    for id in &drag.slots {
        at = survival_centre(screen, *id);
        screen.mouse_dragged(at.0, at.1, drag.button, 0);
    }
    at
}

fn top_left(centre: (i32, i32)) -> (i32, i32) {
    (centre.0 - 8, centre.1 - 8)
}

/// The left-button split from a bug report: 64 diamonds over three slots, the middle one
/// already holding 50 (21 each; 21 + 50 caps at a yellow 64; 8 stay on the cursor).
fn left_split(slots: [u16; 3]) -> Drag {
    Drag {
        cursor: stack("minecraft:diamond", 64),
        existing: vec![(slots[1], stack("minecraft:diamond", 50))],
        button: 0,
        slots: slots.to_vec(),
    }
}

/// A right-button sweep that uses the whole held stack: 3 over three slots, a yellow `0`.
fn right_split(slots: [u16; 3]) -> Drag {
    Drag {
        cursor: stack("minecraft:gold_ingot", 3),
        existing: Vec::new(),
        button: 1,
        slots: slots.to_vec(),
    }
}

/// A middle-button (pick block) sweep, which only a creative player can finish: every swept
/// slot previews a full stack and the held stack goes negative, drawn in red.
fn middle_split(slots: [u16; 3]) -> Drag {
    Drag {
        cursor: stack("minecraft:diamond", 5),
        existing: Vec::new(),
        button: 2,
        slots: slots.to_vec(),
    }
}

/// Every failed check of a test, reported together at the end so that a run on code without
/// the preview still writes every picture.
#[derive(Default)]
struct Failures(Vec<String>);

impl Failures {
    fn check(&mut self, ok: bool, message: String) {
        if !ok {
            self.0.push(message);
        }
    }

    fn assert_none(self) {
        assert!(self.0.is_empty(), "{}", self.0.join("\n"));
    }
}

/// Check one creative scene: every swept slot but the hovered one is lightened around a ghost
/// stack, and the held stack's count changes (parked off every slot to see it alone).
fn check_creative(
    rig: &mut Rig,
    failures: &mut Failures,
    name: &str,
    tab: CreativeTab,
    drag: &Drag,
) -> Vec<u8> {
    let mut plain = creative_on(tab, drag);
    let last = creative_centre(&plain, *drag.slots.last().expect("a slot"));
    let before = rig.shoot(&mut plain, last);

    let mut dragging = creative_on(tab, drag);
    let at = creative_sweep(&mut dragging, drag);
    assert_eq!(at, last);
    let during = rig.shoot(&mut dragging, at);
    save(&format!("{name}_plain"), &before);
    save(name, &during);

    for id in &drag.slots[..drag.slots.len() - 1] {
        let cell = top_left(creative_centre(&dragging, *id));
        let lit = cell_lightened(&before, &during, cell);
        failures.check(
            lit >= 64,
            format!(
                "{name}: swept slot {id} has to show the 0x80FFFFFF drag rect around its ghost \
                 stack (java GuiContainer.drawSlot:297); only {lit} pixels lightened"
            ),
        );
    }

    // Parked over the panel art, off every slot: only the held stack's count can differ.
    let park = (dragging.gui_left() + 120, dragging.gui_top() + 20);
    let parked_plain = rig.shoot(&mut plain, park);
    let parked_drag = rig.shoot(&mut dragging, park);
    save(&format!("{name}_parked"), &parked_drag);
    let changed = cell_diff(
        &parked_plain,
        &parked_drag,
        (park.0 - 8 + 1, park.1 - 8 + 1),
    );
    failures.check(
        changed > 0,
        format!(
            "{name}: the held stack has to show the remnant (java GuiContainer.drawScreen:158-167)"
        ),
    );
    during
}

fn check_survival(rig: &mut Rig, failures: &mut Failures, name: &str, drag: &Drag) -> Vec<u8> {
    let mut plain = survival_with(drag);
    let last = survival_centre(&plain, *drag.slots.last().expect("a slot"));
    let before = rig.shoot(&mut plain, last);

    let mut dragging = survival_with(drag);
    let at = survival_sweep(&mut dragging, drag);
    assert_eq!(at, last);
    let during = rig.shoot(&mut dragging, at);
    save(&format!("{name}_plain"), &before);
    save(name, &during);

    for id in &drag.slots[..drag.slots.len() - 1] {
        let cell = top_left(survival_centre(&dragging, *id));
        let lit = cell_lightened(&before, &during, cell);
        failures.check(
            lit >= 64,
            format!("{name}: swept slot {id} has to show the drag rect; only {lit} lightened"),
        );
    }
    during
}

/// The creative Inventory tab and an item tab show the same preview as the survival inventory,
/// for a left split with a capped slot and for a right sweep that empties the held stack.
#[test]
fn the_creative_screen_previews_a_drag_split_like_the_survival_inventory() {
    let Some(mut rig) = Rig::new() else { return };
    let mut failures = Failures::default();
    let f = &mut failures;

    // Main inventory row 1 (slots 9..11) on the Inventory tab and in the survival inventory;
    // the hotbar (slots 36..38) on an item tab, the only cells a drag reaches there.
    let inventory = CreativeTab::Inventory;
    let items = CreativeTab::Items("materials");
    let drag = left_split([9, 10, 11]);
    let c = check_creative(&mut rig, f, "creative_inventory_tab_left", inventory, &drag);
    let s = check_survival(&mut rig, f, "survival_inventory_left", &drag);
    save_pair("pair_left_split", &c, &s);

    let drag = left_split([36, 37, 38]);
    let c = check_creative(&mut rig, f, "creative_items_tab_left", items, &drag);
    let s = check_survival(&mut rig, f, "survival_hotbar_left", &drag);
    save_pair("pair_hotbar_left_split", &c, &s);

    let drag = right_split([27, 28, 29]);
    let c = check_creative(
        &mut rig,
        f,
        "creative_inventory_tab_right",
        inventory,
        &drag,
    );
    let s = check_survival(&mut rig, f, "survival_inventory_right", &drag);
    save_pair("pair_right_split", &c, &s);

    // The creative-only full-stack sweep (java `Container.isValidDragMode`: mode 2 needs a
    // creative player), on an item tab's hotbar.
    let drag = middle_split([39, 40, 41]);
    let _ = check_creative(&mut rig, f, "creative_items_tab_middle", items, &drag);
    failures.assert_none();
}

/// java: `drawSlot:245-248`, `if (dragSplittingSlots.size() == 1) return;`: with one swept slot
/// the slot draws nothing (its stack is hidden; only the hover highlight is left) and the held
/// stack keeps its full count.
#[test]
fn a_lone_swept_creative_slot_hides_its_stack_and_keeps_the_full_cursor() {
    let Some(mut rig) = Rig::new() else { return };
    let drag = Drag {
        cursor: stack("minecraft:diamond", 64),
        existing: vec![(37, stack("minecraft:diamond", 10))],
        button: 0,
        slots: vec![37],
    };
    let tab = CreativeTab::Items("materials");
    let mut plain = creative_on(tab, &drag);
    let park = (plain.gui_left() + 120, plain.gui_top() + 20);
    let before = rig.shoot(&mut plain, park);

    let mut dragging = creative_on(tab, &drag);
    let _ = creative_sweep(&mut dragging, &drag);
    let during = rig.shoot(&mut dragging, park);
    save("creative_items_tab_lone_slot_plain", &before);
    save("creative_items_tab_lone_slot", &during);

    // The slot's own 10 diamonds disappear: the cell turns into the empty slot background.
    let cell = top_left(creative_centre(&dragging, 37));
    let empty_cell = top_left(creative_centre(&dragging, 40));
    let blank: Vec<[u8; 4]> = (0..16u32)
        .flat_map(|dy| (0..16u32).map(move |dx| (dx, dy)))
        .map(|(dx, dy)| pixel(&during, empty_cell.0 as u32 + dx, empty_cell.1 as u32 + dy))
        .collect();
    let shown: Vec<[u8; 4]> = (0..16u32)
        .flat_map(|dy| (0..16u32).map(move |dx| (dx, dy)))
        .map(|(dx, dy)| pixel(&during, cell.0 as u32 + dx, cell.1 as u32 + dy))
        .collect();
    let mut failures = Failures::default();
    failures.check(
        shown == blank,
        "a lone swept slot draws nothing (java GuiContainer.drawSlot:245-248)".to_owned(),
    );
    // The held stack is unchanged: 64 is still shown.
    let cursor_cell = (park.0 - 8, park.1 - 8);
    let changed = cell_diff(&before, &during, cursor_cell);
    failures.check(
        changed == 0,
        format!("the held stack keeps its full count with one swept slot; {changed} px changed"),
    );
    failures.assert_none();
}
