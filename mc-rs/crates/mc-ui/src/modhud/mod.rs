//! The mod HUD pipeline: draggable HUD elements drawn after the vanilla HUD.
//!
//! ```text
//!  ModSettings ──sync on generation change──► ModHud (one slot per element)
//!  Client::fill_hud_data ──► HudData ──► each enabled element: layout → resolve placement →
//!                                        draw under translate+scale ──► the HUD's DrawList
//! ```
//!
//! * **Zero cost when off.** [`ModHud::sync`] is one `u64` compare unless settings changed, and
//!   the client skips the whole pipeline (data fill included) when [`ModHud::any_enabled`] is
//!   false.
//! * **Elements draw in local coordinates.** `(0, 0)` is the element's top-left; the pipeline
//!   applies the placement's offset and scale with a [`Transform`]. An element never needs to
//!   know where it is on screen.
//! * **Cache your strings.** `layout` runs every frame. Rebuild the text only when the value
//!   behind it changed (see [`elements::fps`]); `format!` per frame is what this rule forbids.
//! * **The HUD editor** (`screens::modmenu`) reads [`ModHud::slots`] for each drawn rectangle
//!   and writes placements back through `ModSettings::set_placement_at`.
//!
//! # Adding an element
//!
//! 1. Give the mod's descriptor a `HudElementDescriptor` (`mc-mods/src/mods/<id>.rs`).
//! 2. Implement [`HudElement`] in `modhud/elements/<id>.rs` (stubs already exist for every HUD
//!    element in the catalog).
//! 3. It is already listed in [`elements::registry`]; add a line there only for a brand-new
//!    element.
//!
//! # Turning a vanilla HUD piece into an element (boss bar, scoreboard)
//!
//! Skip the vanilla draw while the mod is enabled (the client has a cached flag), and in the
//! element's `draw` call the existing vanilla drawing code with its origin moved to `(0, 0)`:
//! the pipeline's transform puts it wherever the user dragged it. Pass the data the vanilla
//! code needs through a [`HudData`] field.

pub mod data;
pub mod elements;
pub mod style;

use mc_mods::{resolve_rect, HudPlacement, HudRect, ModSettings};

use crate::font::FontRenderer;
use crate::widget::{DrawList, Transform, UiTexture};

pub use data::{HudData, HudEffect, HudKeys, HudReach, SprintHudState};
pub use style::HudTextStyle;

/// What an element sees while it lays out and draws.
#[derive(Debug, Clone, Copy)]
pub struct HudCtx<'a> {
    /// This frame's game state.
    pub data: &'a HudData,
    /// The settings store (prefer the snapshot you took in [`HudElement::sync`]).
    pub settings: &'a ModSettings,
    /// The scaled GUI size.
    pub screen: (f32, f32),
    /// Physical pixels per GUI pixel (the GUI scale factor).
    pub pixel_scale: f32,
    /// The HUD editor is open: draw a representative sample even when there is nothing to show
    /// (no effects, no boss), so the user can see and drag the element.
    pub editing: bool,
}

/// One draggable thing on the HUD.
pub trait HudElement {
    /// `(mod id, element id)`, as in the mod's `HudElementDescriptor`.
    fn id(&self) -> (&'static str, &'static str);

    /// Settings changed (and once before the first frame): refresh any cached options.
    fn sync(&mut self, settings: &ModSettings) {
        let _ = settings;
    }

    /// This frame's **unscaled** size in GUI pixels, or `None` when there is nothing to draw
    /// (and, while editing, no sample to show: the pipeline then draws a labelled placeholder).
    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)>;

    /// Draw at local `(0, 0)` with the size `layout` returned.
    fn draw(&mut self, ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList);

    /// Where `placement` puts an element of this unscaled `size` on a `screen`-sized GUI, before
    /// the pipeline rounds it to whole pixels. The default is [`mc_mods::resolve_rect`]; override
    /// it (together with [`HudElement::placement_for`]) only to reproduce a vanilla position the
    /// 9-point model cannot express, e.g. the sidebar, which vanilla pins 2/3 down its height.
    fn resolve(&self, placement: &HudPlacement, size: (f32, f32), screen: (f32, f32)) -> HudRect {
        resolve_rect(placement, size, screen)
    }

    /// The exact inverse of [`HudElement::resolve`]: the placement that puts this element at the
    /// on-screen `rect` (the HUD editor's drag preview, drop and scroll-scale). The default is
    /// [`mc_mods::placement_from_rect`].
    fn placement_for(&self, rect: &HudRect, scale: f32, screen: (f32, f32)) -> HudPlacement {
        mc_mods::placement_from_rect(rect, scale, screen)
    }

    /// Whether this element still draws while the F3 debug screen is open. The mod HUD hides
    /// under F3, but an element that stands in for a vanilla HUD piece vanilla
    /// draws with F3 open (the boss bar, the sidebar) returns `true` so that turning its mod on
    /// does not make the piece vanish under F3. See [`ModHud::draw_under_debug`].
    fn draws_under_debug(&self) -> bool {
        false
    }
}

/// One registered element and where it was last drawn.
pub struct HudSlot {
    element: Box<dyn HudElement>,
    /// Index of the owning mod in `ModSettings::catalog()`.
    pub mod_index: usize,
    /// Index into that mod's `hud_elements`.
    pub element_index: usize,
    /// The element's label in the HUD editor.
    pub name: &'static str,
    /// Whether the owning mod is enabled (as of the last sync).
    pub enabled: bool,
    /// The placement as of the last sync.
    pub placement: HudPlacement,
    /// The on-screen rectangle of the last draw, `None` if it was not drawn.
    pub rect: Option<HudRect>,
    /// The last draw was a placeholder (no `layout`), not the element itself.
    pub placeholder: bool,
}

impl std::fmt::Debug for HudSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HudSlot")
            .field("id", &self.element.id())
            .field("enabled", &self.enabled)
            .field("rect", &self.rect)
            .finish_non_exhaustive()
    }
}

/// The pipeline. One lives on the client (`Client::mods.hud`).
#[derive(Debug)]
pub struct ModHud {
    slots: Vec<HudSlot>,
    seen_generation: u64,
    any_enabled: bool,
}

impl Default for ModHud {
    fn default() -> Self {
        Self::new()
    }
}

impl ModHud {
    /// Every element in [`elements::registry`], mapped onto the real catalog.
    #[must_use]
    pub fn new() -> ModHud {
        ModHud::with_elements(mc_mods::catalog::all(), elements::registry())
    }

    /// Map `elements` onto `catalog`. An element whose ids are not in the catalog is dropped.
    #[must_use]
    pub fn with_elements(
        catalog: &'static [&'static mc_mods::ModDescriptor],
        elements: Vec<Box<dyn HudElement>>,
    ) -> ModHud {
        let slots = elements
            .into_iter()
            .filter_map(|element| {
                let (mod_id, element_id) = element.id();
                let mod_index = catalog.iter().position(|d| d.id == mod_id)?;
                let descriptor = catalog[mod_index];
                let element_index = descriptor.hud_element_index(element_id)?;
                let element_descriptor = &descriptor.hud_elements[element_index];
                Some(HudSlot {
                    element,
                    mod_index,
                    element_index,
                    name: element_descriptor.name,
                    enabled: false,
                    placement: element_descriptor.default_placement,
                    rect: None,
                    placeholder: false,
                })
            })
            .collect();
        ModHud {
            slots,
            seen_generation: 0,
            any_enabled: false,
        }
    }

    /// Pick up settings changes. One `u64` compare when nothing changed. Returns whether it
    /// re-synced.
    pub fn sync(&mut self, settings: &ModSettings) -> bool {
        if self.seen_generation == settings.generation() {
            return false;
        }
        self.seen_generation = settings.generation();
        self.any_enabled = false;
        for slot in &mut self.slots {
            slot.enabled = settings.enabled_at(slot.mod_index);
            if let Some(placement) = settings.placement_at(slot.mod_index, slot.element_index) {
                slot.placement = placement;
            }
            if slot.enabled {
                slot.element.sync(settings);
            } else {
                slot.rect = None;
            }
            self.any_enabled |= slot.enabled;
        }
        true
    }

    /// Whether any element's mod is enabled. When `false` the client skips the HUD data fill
    /// and [`ModHud::draw`] entirely.
    #[must_use]
    pub fn any_enabled(&self) -> bool {
        self.any_enabled
    }

    /// Whether the element `(mod_id, element_id)` exists and its mod is enabled.
    #[must_use]
    pub fn is_enabled(&self, mod_id: &str) -> bool {
        self.slots
            .iter()
            .any(|slot| slot.enabled && slot.element.id().0 == mod_id)
    }

    /// Every slot, enabled or not, in draw order.
    #[must_use]
    pub fn slots(&self) -> &[HudSlot] {
        &self.slots
    }

    /// The topmost drawn slot under the GUI-pixel point `(x, y)`.
    #[must_use]
    pub fn hit(&self, x: f32, y: f32) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .rev()
            .find(|(_, slot)| slot.enabled && slot.rect.is_some_and(|r| r.contains(x, y)))
            .map(|(index, _)| index)
    }

    /// Override one slot's placement until the next sync (the HUD editor's live drag preview).
    pub fn preview_placement(&mut self, slot: usize, placement: HudPlacement) {
        if let Some(slot) = self.slots.get_mut(slot) {
            slot.placement = placement;
        }
    }

    /// The placement that puts slot `slot` at the on-screen `rect`, through the element's own
    /// [`HudElement::placement_for`] (the inverse of what [`ModHud::draw`] resolves). The HUD
    /// editor uses this, not `mc_mods::placement_from_rect`, for drag preview, drop and scale.
    #[must_use]
    pub fn placement_for(
        &self,
        slot: usize,
        rect: &HudRect,
        scale: f32,
        screen: (f32, f32),
    ) -> Option<HudPlacement> {
        self.slots
            .get(slot)
            .map(|slot| slot.element.placement_for(rect, scale, screen))
    }

    /// Lay out and draw every enabled element into `d` (GUI pixel space).
    pub fn draw(&mut self, ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        self.draw_slots(ctx, font, d, false);
    }

    /// [`ModHud::draw`] while the F3 debug screen is open (and the HUD editor is not): only the
    /// enabled elements whose [`HudElement::draws_under_debug`] is `true` are drawn; every other
    /// slot is left undrawn (`rect = None`).
    pub fn draw_under_debug(
        &mut self,
        ctx: &HudCtx<'_>,
        font: &mut FontRenderer,
        d: &mut DrawList,
    ) {
        self.draw_slots(ctx, font, d, true);
    }

    fn draw_slots(
        &mut self,
        ctx: &HudCtx<'_>,
        font: &mut FontRenderer,
        d: &mut DrawList,
        debug_only: bool,
    ) {
        // Attribution only (a no-op unless `MC_PROFILE` is set).
        mc_core::profile_scope!("hud.mods.draw_slots");
        for slot in &mut self.slots {
            if !slot.enabled {
                continue;
            }
            if debug_only && !slot.element.draws_under_debug() {
                slot.rect = None;
                continue;
            }
            let (size, placeholder) = match slot.element.layout(ctx, font) {
                Some(size) => (size, false),
                None if ctx.editing => (placeholder_size(font, slot.name), true),
                None => {
                    slot.rect = None;
                    continue;
                }
            };
            let mut rect = slot.element.resolve(&slot.placement, size, ctx.screen);
            // Whole GUI pixels, so the Minecraft font stays crisp at scale 1.
            rect.x = rect.x.round();
            rect.y = rect.y.round();
            slot.rect = Some(rect);
            slot.placeholder = placeholder;
            let scale = slot.placement.scale;
            d.with_transform(
                Transform::translate(rect.x, rect.y).then_scale(scale),
                |d| {
                    if placeholder {
                        draw_placeholder(d, font, slot.name, size);
                    } else {
                        slot.element.draw(ctx, font, d);
                    }
                },
            );
        }
    }
}

fn placeholder_size(font: &FontRenderer, name: &str) -> (f32, f32) {
    (font.string_width(name).max(0) as f32 + 7.0, 16.0)
}

fn draw_placeholder(d: &mut DrawList, font: &mut FontRenderer, name: &str, size: (f32, f32)) {
    d.quad_rect(
        &UiTexture::Solid,
        (0.0, 0.0, size.0, size.1),
        (0.0, 0.0, 1.0, 1.0),
        0x5000_0000,
    );
    font.draw_string(d, name, 4.0, 4.0, 0xFFAA_AAAA_u32 as i32, true);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_catalog_hud_element_has_an_implementation_and_vice_versa() {
        let hud = ModHud::new();
        let registered = elements::registry().len();
        assert_eq!(
            hud.slots().len(),
            registered,
            "a registry element names a mod/element the catalog does not have"
        );
        for descriptor in mc_mods::catalog::all() {
            for element in descriptor.hud_elements {
                assert!(
                    hud.slots().iter().any(|slot| {
                        let d = mc_mods::catalog::all()[slot.mod_index];
                        d.id == descriptor.id && d.hud_elements[slot.element_index].id == element.id
                    }),
                    "{}.{} has no HudElement in modhud::elements::registry",
                    descriptor.id,
                    element.id
                );
            }
        }
    }

    #[test]
    fn nothing_is_enabled_by_default_and_sync_is_generation_gated() {
        let mut settings = ModSettings::new();
        let mut hud = ModHud::new();
        assert!(hud.sync(&settings));
        assert!(!hud.any_enabled());
        assert!(!hud.sync(&settings), "same generation: no work");
        settings.set_enabled("fps", true);
        assert!(hud.sync(&settings));
        assert!(hud.any_enabled());
        assert!(hud.is_enabled("fps"));
        settings.set_enabled("fps", false);
        hud.sync(&settings);
        assert!(!hud.any_enabled());
    }
}
