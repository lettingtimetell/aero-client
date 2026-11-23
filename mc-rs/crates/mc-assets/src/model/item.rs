//! Item model baking, including the `builtin/generated` flat-icon expansion.
//!
// java: ItemModelMesher.java, ItemModelGenerator.java
//!
//! Vanilla's default item model is a normal JSON model whose ultimate parent is the
//! `builtin/generated` sentinel (`assets/minecraft/models/item/generated.json` declares
//! `"parent": "builtin/generated"`). [`super::blockmodel::ResolvedModel::resolve`] stops the
//! parent walk there and records it; this module turns that into real geometry — a thin
//! double-sided core plane plus a perimeter of edge quads traced from the icon's own alpha
//! channel, so a flat 16x16 icon reads as a thin 3D card instead of a paper cutout with
//! visible see-through edges.

use std::collections::HashMap;
use std::sync::OnceLock;

use mc_data::ItemId;
use mc_math::Facing;

use super::blockmodel::{RawElement, RawFace, RawRotation};
use super::facebakery::{bake_quad, BakeRotation};
use super::{BakedModel, BakedQuad};
use crate::sprite::{Sprite, UvRect};
use crate::ResourceLocation;

/// The live item property that changes an item's atlas frame without changing its baked
/// geometry.  Vanilla registers compass and clock as ordinary `builtin/generated` item models,
/// then replaces their atlas sprites with `TextureCompass`/`TextureClock` subclasses.  Keeping
/// this identity beside model lookup gives every renderer (held, GUI, dropped, and fixed/frame)
/// one property oracle instead of each call site growing a name/id special case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemModelProperty {
    /// No dynamic item property; the baked model and ordinary atlas animation are sufficient.
    None,
    /// `Items.compass` / `TextureCompass`.
    Compass,
    /// `Items.clock` / `TextureClock`.
    Clock,
}

impl ItemModelProperty {
    /// Resolve the property attached to an item registry entry.
    #[must_use]
    pub fn for_item(id: ItemId) -> Self {
        match mc_data::item_props(id).name {
            "minecraft:compass" => Self::Compass,
            "minecraft:clock" => Self::Clock,
            _ => Self::None,
        }
    }

    /// Whether this property selects one of vanilla's live atlas sprites.
    #[must_use]
    pub const fn is_dynamic(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// The model plus the live item-property identity needed by a render consumer.
///
/// This is deliberately a small, copyable value: models remain process-global baked references,
/// while the property is a stateless discriminator for [`crate::AtlasAnimator`].  Callers should
/// retain the property when they cache a model or split a draw into another pass; asking the
/// registry again in each pass is how compass/clock fallbacks drift between held/UI/frame paths.
#[derive(Debug, Clone, Copy)]
pub struct ItemModelOverride {
    /// The baked model selected by item id and stack metadata.
    pub model: &'static BakedModel,
    /// The live property carried by this item model.
    pub property: ItemModelProperty,
}

// `BakedModel` deliberately contains floating-point transforms, so it cannot derive `Eq`.
// Models in this registry are process-global baked references; identity is the equality Java's
// model caches actually care about, while the property remains a value comparison.
impl PartialEq for ItemModelOverride {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.model, other.model) && self.property == other.property
    }
}

impl Eq for ItemModelOverride {}

impl ItemModelOverride {
    /// Whether the selected model is rendered through the GUI-3D branch.
    #[must_use]
    pub const fn gui_3d(self) -> bool {
        self.model.gui_3d
    }
}

/// One entry in vanilla's `RenderItem.registerItems` table.
///
/// Most items use `(metadata 0, registry name)`. The exceptions below are the complete
/// `ModelBakery.registerVariantNames` set, plus the duplicate golden-apple registration from
/// `RenderItem`. Keeping the table here makes the bake and the runtime lookup use one oracle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ItemModelRegistration {
    pub metadata: u16,
    pub model: String,
}

const BLOCK_COLOURS: [&str; 16] = [
    "white",
    "orange",
    "magenta",
    "light_blue",
    "yellow",
    "lime",
    "pink",
    "gray",
    "silver",
    "cyan",
    "purple",
    "blue",
    "brown",
    "green",
    "red",
    "black",
];

const DYE_COLOURS: [&str; 16] = [
    "black",
    "red",
    "green",
    "brown",
    "blue",
    "purple",
    "cyan",
    "silver",
    "gray",
    "pink",
    "lime",
    "yellow",
    "light_blue",
    "magenta",
    "orange",
    "white",
];

fn enumerated(names: &[&str]) -> Vec<ItemModelRegistration> {
    names
        .iter()
        .enumerate()
        .map(|(metadata, model)| ItemModelRegistration {
            metadata: metadata as u16,
            model: (*model).to_owned(),
        })
        .collect()
}

fn coloured(suffix: &str) -> Vec<ItemModelRegistration> {
    BLOCK_COLOURS
        .iter()
        .enumerate()
        .map(|(metadata, colour)| ItemModelRegistration {
            metadata: metadata as u16,
            model: format!("{colour}_{suffix}"),
        })
        .collect()
}

/// Vanilla's item-model registrations for one item registry path.
///
/// java: `ModelBakery.registerVariantNames` (the names that are loaded) and
/// `RenderItem.registerItems` (the metadata keys that select them). The colour-family arrays are
/// deliberately in `EnumDyeColor.getMetadata()` order (white = 0, black = 15); dye items use
/// the inverse `getDyeDamage()` order.
pub(super) fn registrations_for(registry_path: &str) -> Vec<ItemModelRegistration> {
    let names: Option<&[&str]> = match registry_path {
        "stone" => Some(&[
            "stone",
            "granite",
            "granite_smooth",
            "diorite",
            "diorite_smooth",
            "andesite",
            "andesite_smooth",
        ]),
        "dirt" => Some(&["dirt", "coarse_dirt", "podzol"]),
        "planks" => Some(&[
            "oak_planks",
            "spruce_planks",
            "birch_planks",
            "jungle_planks",
            "acacia_planks",
            "dark_oak_planks",
        ]),
        "sapling" => Some(&[
            "oak_sapling",
            "spruce_sapling",
            "birch_sapling",
            "jungle_sapling",
            "acacia_sapling",
            "dark_oak_sapling",
        ]),
        "sand" => Some(&["sand", "red_sand"]),
        "log" => Some(&["oak_log", "spruce_log", "birch_log", "jungle_log"]),
        "leaves" => Some(&[
            "oak_leaves",
            "spruce_leaves",
            "birch_leaves",
            "jungle_leaves",
        ]),
        "sponge" => Some(&["sponge", "sponge_wet"]),
        "sandstone" => Some(&["sandstone", "chiseled_sandstone", "smooth_sandstone"]),
        "red_sandstone" => Some(&[
            "red_sandstone",
            "chiseled_red_sandstone",
            "smooth_red_sandstone",
        ]),
        "tallgrass" => Some(&["dead_bush", "tall_grass", "fern"]),
        "deadbush" => Some(&["dead_bush"]),
        "yellow_flower" => Some(&["dandelion"]),
        "red_flower" => Some(&[
            "poppy",
            "blue_orchid",
            "allium",
            "houstonia",
            "red_tulip",
            "orange_tulip",
            "white_tulip",
            "pink_tulip",
            "oxeye_daisy",
        ]),
        // `old_wood_slab` (metadata 2) is intentionally absent. RenderItem registers it, but
        // ModelBakery does not put that name in `variantNames`, so vanilla resolves the missing
        // model for this legacy/unobtainable subtype too.
        "stone_slab" => Some(&[
            "stone_slab",
            "sandstone_slab",
            "",
            "cobblestone_slab",
            "brick_slab",
            "stone_brick_slab",
            "nether_brick_slab",
            "quartz_slab",
        ]),
        "stone_slab2" => Some(&["red_sandstone_slab"]),
        "monster_egg" => Some(&[
            "stone_monster_egg",
            "cobblestone_monster_egg",
            "stone_brick_monster_egg",
            "mossy_brick_monster_egg",
            "cracked_brick_monster_egg",
            "chiseled_brick_monster_egg",
        ]),
        "stonebrick" => Some(&[
            "stonebrick",
            "mossy_stonebrick",
            "cracked_stonebrick",
            "chiseled_stonebrick",
        ]),
        "wooden_slab" => Some(&[
            "oak_slab",
            "spruce_slab",
            "birch_slab",
            "jungle_slab",
            "acacia_slab",
            "dark_oak_slab",
        ]),
        "cobblestone_wall" => Some(&["cobblestone_wall", "mossy_cobblestone_wall"]),
        "anvil" => Some(&[
            "anvil_intact",
            "anvil_slightly_damaged",
            "anvil_very_damaged",
        ]),
        "quartz_block" => Some(&["quartz_block", "chiseled_quartz_block", "quartz_column"]),
        "leaves2" => Some(&["acacia_leaves", "dark_oak_leaves"]),
        "log2" => Some(&["acacia_log", "dark_oak_log"]),
        "prismarine" => Some(&["prismarine", "prismarine_bricks", "dark_prismarine"]),
        "double_plant" => Some(&[
            "sunflower",
            "syringa",
            "double_grass",
            "double_fern",
            "double_rose",
            "paeonia",
        ]),
        "bow" => Some(&["bow", "bow_pulling_0", "bow_pulling_1", "bow_pulling_2"]),
        "coal" => Some(&["coal", "charcoal"]),
        "fishing_rod" => Some(&["fishing_rod", "fishing_rod_cast"]),
        "fish" => Some(&["cod", "salmon", "clownfish", "pufferfish"]),
        "cooked_fish" => Some(&["cooked_cod", "cooked_salmon"]),
        // Runtime selection canonicalises every non-splash value to 0 and every splash value
        // to 0x4000, matching ItemPotion.isSplash in RenderItem's ItemMeshDefinition.
        "potion" => {
            return vec![
                ItemModelRegistration {
                    metadata: 0,
                    model: "bottle_drinkable".to_owned(),
                },
                ItemModelRegistration {
                    metadata: 0x4000,
                    model: "bottle_splash".to_owned(),
                },
            ];
        }
        "skull" => Some(&[
            "skull_skeleton",
            "skull_wither",
            "skull_zombie",
            "skull_char",
            "skull_creeper",
        ]),
        "fence_gate" => Some(&["oak_fence_gate"]),
        "fence" => Some(&["oak_fence"]),
        "wooden_door" => Some(&["oak_door"]),
        "golden_apple" => Some(&["golden_apple", "golden_apple"]),
        _ => None,
    };

    match registry_path {
        "wool" => coloured("wool"),
        "carpet" => coloured("carpet"),
        "stained_glass" => coloured("stained_glass"),
        "stained_glass_pane" => coloured("stained_glass_pane"),
        "stained_hardened_clay" => coloured("stained_hardened_clay"),
        "dye" => DYE_COLOURS
            .iter()
            .enumerate()
            .map(|(metadata, colour)| ItemModelRegistration {
                metadata: metadata as u16,
                model: format!("dye_{colour}"),
            })
            .collect(),
        _ => names.map_or_else(
            || {
                vec![ItemModelRegistration {
                    metadata: 0,
                    model: registry_path.to_owned(),
                }]
            },
            |names| {
                enumerated(names)
                    .into_iter()
                    .filter(|registration| !registration.model.is_empty())
                    .collect()
            },
        ),
    }
}

#[inline]
pub(super) fn item_model_key(id: ItemId, metadata: u16) -> u32 {
    (u32::from(id.0) << 16) | u32::from(metadata)
}

fn canonical_metadata(id: ItemId, metadata: u16) -> u16 {
    let props = mc_data::item_props(id);
    if props.damageable {
        return 0;
    }
    let path = props.name.strip_prefix("minecraft:").unwrap_or(props.name);
    match path {
        "potion" => metadata & 0x4000,
        // These are the four constant ItemMeshDefinitions in RenderItem (spawn eggs and
        // banners are the other two; enchanted books and filled maps complete the set).
        "spawn_egg" | "banner" | "enchanted_book" | "filled_map" => 0,
        _ => metadata,
    }
}

/// `ItemModelGenerator.LAYERS`.
pub const LAYERS: [&str; 5] = ["layer0", "layer1", "layer2", "layer3", "layer4"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanFacing {
    Up,
    Down,
    Left,
    Right,
}

impl SpanFacing {
    const ALL: [SpanFacing; 4] = [
        SpanFacing::Up,
        SpanFacing::Down,
        SpanFacing::Left,
        SpanFacing::Right,
    ];

    fn offset(self) -> (i32, i32) {
        match self {
            SpanFacing::Up => (0, -1),
            SpanFacing::Down => (0, 1),
            SpanFacing::Left => (-1, 0),
            SpanFacing::Right => (1, 0),
        }
    }

    fn is_vertical(self) -> bool {
        matches!(self, SpanFacing::Up | SpanFacing::Down)
    }

    /// java: `SpanFacing.getFacing` — note Left/Right map to East/West, not to themselves.
    fn output_facing(self) -> Facing {
        match self {
            SpanFacing::Up => Facing::Up,
            SpanFacing::Down => Facing::Down,
            SpanFacing::Left => Facing::East,
            SpanFacing::Right => Facing::West,
        }
    }
}

struct SpanAcc {
    facing: SpanFacing,
    min: i32,
    max: i32,
    fixed: i32,
}

/// Whether the texel is empty for the edge faces: outside the icon, or alpha at most
/// `empty_max_alpha` (vanilla: 0, so only alpha exactly 0 is empty).
fn is_transparent(rgba: &[u8], x: i32, y: i32, w: i32, h: i32, empty_max_alpha: u8) -> bool {
    if x < 0 || y < 0 || x >= w || y >= h {
        return true;
    }
    let idx = ((y * w + x) * 4 + 3) as usize;
    rgba.get(idx).copied().unwrap_or(0) <= empty_max_alpha
}

/// java: `ItemModelGenerator.getSpans` + `addOrExtendSpan`. Vanilla passes `empty_max_alpha` 0.
fn detect_spans_into(rgba: &[u8], w: i32, h: i32, empty_max_alpha: u8, spans: &mut Vec<SpanAcc>) {
    for y in 0..h {
        for x in 0..w {
            let opaque_here = !is_transparent(rgba, x, y, w, h, empty_max_alpha);
            if !opaque_here {
                continue;
            }
            for dir in SpanFacing::ALL {
                let (dx, dy) = dir.offset();
                if !is_transparent(rgba, x + dx, y + dy, w, h, empty_max_alpha) {
                    continue;
                }
                let (fixed, var) = if dir.is_vertical() { (y, x) } else { (x, y) };
                if let Some(s) = spans
                    .iter_mut()
                    .find(|s| s.facing == dir && s.fixed == fixed)
                {
                    if var < s.min {
                        s.min = var;
                    } else if var > s.max {
                        s.max = var;
                    }
                } else {
                    spans.push(SpanAcc {
                        facing: dir,
                        min: var,
                        max: var,
                        fixed,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
fn detect_spans(rgba: &[u8], w: i32, h: i32) -> Vec<SpanAcc> {
    let mut spans = Vec::new();
    detect_spans_into(rgba, w, h, 0, &mut spans);
    spans
}

/// java: `ItemModelGenerator.func_178397_a`'s per-span geometry math.
///
/// Each span becomes a thin (zero-width-or-height) box with one face: the *position*
/// outputs (`px0/py0/px1/py1`, a `0..16` box like any other element) and the *UV* outputs
/// (`fu0/fv0/fu1/fv1`, a `0..16` texel-space rect like any other face) start from the same
/// `(lo, hi, fixed)` triple but are scaled differently — position by the flat `16/size`
/// factor, UV by `16/(size-1)` on the axis that runs *along* the icon's edge (`f11` for
/// up/down, `f10` for left/right), which is what keeps a multi-pixel-tall perimeter strip's
/// texture from being squashed by one texel at the seam. Java computes both from one shared
/// set of mutably-aliased locals; this keeps them as two clearly-named passes instead.
fn span_element(span: &SpanAcc, w: f32, h: f32, texture: &str, tint: i32) -> (RawElement, Facing) {
    let lo = span.min as f32;
    let hi = span.max as f32;
    let fixed = span.fixed as f32;
    let facing = span.facing.output_facing();

    let f10;
    let f11;
    match span.facing {
        SpanFacing::Up | SpanFacing::Down => {
            f10 = 16.0 / w;
            f11 = 16.0 / (h - 1.0);
        }
        SpanFacing::Left | SpanFacing::Right => {
            f10 = 16.0 / (w - 1.0);
            f11 = 16.0 / h;
        }
    }
    let f15 = 16.0 / w;
    let f16 = 16.0 / h;

    let (mut px0, mut py0, mut px1, mut py1): (f32, f32, f32, f32);
    let (mut fu0, mut fv0, mut fu1, mut fv1): (f32, f32, f32, f32);
    match span.facing {
        SpanFacing::Up => {
            fu0 = lo;
            px0 = lo;
            fu1 = hi + 1.0;
            px1 = hi + 1.0;
            fv0 = fixed;
            py0 = fixed;
            fv1 = fixed;
            py1 = fixed;
        }
        SpanFacing::Down => {
            fv1 = fixed;
            fv0 = fixed;
            fu0 = lo;
            px0 = lo;
            fu1 = hi + 1.0;
            px1 = hi + 1.0;
            py0 = fixed + 1.0;
            py1 = fixed + 1.0;
        }
        SpanFacing::Left => {
            fu0 = fixed;
            px0 = fixed;
            fu1 = fixed;
            px1 = fixed;
            fv1 = lo;
            py0 = lo;
            py1 = hi + 1.0;
            fv0 = hi + 1.0;
        }
        SpanFacing::Right => {
            fu1 = fixed;
            fu0 = fixed;
            px0 = fixed + 1.0;
            px1 = fixed + 1.0;
            fv1 = lo;
            py0 = lo;
            py1 = hi + 1.0;
            fv0 = hi + 1.0;
        }
    }
    px0 *= f15;
    px1 *= f15;
    py0 *= f16;
    py1 *= f16;
    py0 = 16.0 - py0;
    py1 = 16.0 - py1;
    fu0 *= f10;
    fu1 *= f10;
    fv0 *= f11;
    fv1 *= f11;

    let (from, to) = match span.facing {
        SpanFacing::Up => ([px0, py0, 7.5], [px1, py0, 8.5]),
        SpanFacing::Down => ([px0, py1, 7.5], [px1, py1, 8.5]),
        SpanFacing::Left => ([px0, py0, 7.5], [px0, py1, 8.5]),
        SpanFacing::Right => ([px1, py0, 7.5], [px1, py1, 8.5]),
    };

    let mut faces = HashMap::new();
    faces.insert(
        facing_name(facing).to_string(),
        RawFace {
            uv: Some([fu0, fv0, fu1, fv1]),
            texture: texture.to_string(),
            cullface: None,
            rotation: 0,
            tintindex: tint,
        },
    );
    (
        RawElement {
            from,
            to,
            rotation: None as Option<RawRotation>,
            faces,
            shade: true,
        },
        facing,
    )
}

fn facing_name(f: Facing) -> &'static str {
    match f {
        Facing::Down => "down",
        Facing::Up => "up",
        Facing::North => "north",
        Facing::South => "south",
        Facing::West => "west",
        Facing::East => "east",
    }
}

/// Bakes one texture layer of a `builtin/generated` item model: a double-sided core plane
/// plus a traced perimeter, all tinted by `layer_index` (vanilla's per-layer tint convention
/// — layer 0 usually untinted in practice since most single-layer icons don't request a
/// tint, but the mechanism matches `ItemModelGenerator` exactly).
fn bake_layer(layer_index: i32, texture: &str, sprite: &Sprite, uv: UvRect) -> Vec<BakedQuad> {
    let frames: Vec<&[u8]> = sprite.frames.iter().map(|f| f.rgba.as_slice()).collect();
    bake_layer_frames(
        layer_index,
        texture,
        &frames,
        (sprite.width, sprite.height),
        uv,
        0,
    )
}

/// The alpha at or below which 1.7 counted a texel as empty: its `GL_GREATER 0.1` alpha test
/// discards alpha `0..=25` (25 / 255 < 0.1 < 26 / 255), the same test 1.8.9 draws items with.
pub const OLD_ITEM_EMPTY_ALPHA: u8 = 25;

/// One layer of a `builtin/generated` item with 1.7's edges (1.7 Animations' "1.7 item edges").
///
/// Exactly [`bake_layer`]'s quads, except that a texel counts as empty for the edge faces when
/// its alpha is at most [`OLD_ITEM_EMPTY_ALPHA`], not only when it is exactly 0. Vanilla 1.8 treats
/// an alpha `1..=25` texel as solid when it builds the edges, so the solid texel beside it gets no
/// side face, yet draws it with an alpha test that discards it: a see-through gap along the
/// item's side (MC-70336, "Edges of Items transparent at pixels with alpha value > 0 and < 26",
/// which older resource packs' anti-aliased edges hit). 1.7's `ItemRenderer.renderItemIn2D` drew a
/// side strip for every texel column and row, so every drawn texel had its sides.
///
/// For a texture whose alpha is only 0 or 255 the result is identical to vanilla's, quad for quad.
/// `rgba` is one frame of the sprite (`width` x `height` RGBA8 texels) and `uv` its atlas
/// rectangle.
#[must_use]
pub fn bake_generated_layer_17(
    layer_index: i32,
    rgba: &[u8],
    width: u32,
    height: u32,
    uv: UvRect,
) -> Vec<BakedQuad> {
    bake_layer_frames(
        layer_index,
        "layer",
        &[rgba],
        (width, height),
        uv,
        OLD_ITEM_EMPTY_ALPHA,
    )
}

/// [`bake_layer`] over explicit frames, with the edge faces' emptiness threshold.
fn bake_layer_frames(
    layer_index: i32,
    texture: &str,
    frames: &[&[u8]],
    (width, height): (u32, u32),
    uv: UvRect,
    empty_max_alpha: u8,
) -> Vec<BakedQuad> {
    let mut out = Vec::new();

    // The core double-sided plane (java: `func_178394_a`'s first element).
    let mut faces = HashMap::new();
    faces.insert(
        "south".to_string(),
        RawFace {
            uv: Some([0.0, 0.0, 16.0, 16.0]),
            texture: texture.to_string(),
            cullface: None,
            rotation: 0,
            tintindex: layer_index,
        },
    );
    faces.insert(
        "north".to_string(),
        RawFace {
            uv: Some([16.0, 0.0, 0.0, 16.0]),
            texture: texture.to_string(),
            cullface: None,
            rotation: 0,
            tintindex: layer_index,
        },
    );
    let core = RawElement {
        from: [0.0, 0.0, 7.5],
        to: [16.0, 16.0, 8.5],
        rotation: None,
        faces,
        shade: true,
    };
    // `RawElement::faces` is a HashMap. Use Facing's canonical ordinal order just like
    // `bake_resolved_model`, otherwise the two core quads change order across processes.
    for name in ["north", "south"] {
        let face = &core.faces[name];
        let facing = super::blockmodel::parse_facing(name).unwrap();
        out.push(bake_quad(&core, facing, face, uv, BakeRotation::default()));
    }

    let w = width as i32;
    let h = height as i32;
    if w >= 1 && h >= 1 {
        // java: ItemModelGenerator scans every animation frame into one shared span list. The
        // compass and clock move opaque pixels between frames, so frame zero alone cuts holes in
        // the rim used by later frames.
        let mut spans = Vec::new();
        for frame in frames {
            detect_spans_into(frame, w, h, empty_max_alpha, &mut spans);
        }
        for span in spans {
            let (element, facing) =
                span_element(&span, width as f32, height as f32, texture, layer_index);
            let face = &element.faces[facing_name(facing)];
            out.push(bake_quad(
                &element,
                facing,
                face,
                uv,
                BakeRotation::default(),
            ));
        }
    }

    out
}

/// Bakes a `builtin/generated` item model from its already-resolved (`layerN -> texture`)
/// texture table.
#[must_use]
pub fn expand_builtin_generated(
    textures: &HashMap<String, String>,
    sprites: &HashMap<ResourceLocation, Sprite>,
    missing_uv: UvRect,
) -> BakedModel {
    let mut model = BakedModel {
        ambient_occlusion: false,
        ..BakedModel::default()
    };
    for (i, layer) in LAYERS.iter().enumerate() {
        let Some(resolved) = textures.get(*layer) else {
            break; // java stops at the first absent layer, not just skips it
        };
        if resolved == "missingno" {
            if i == 0 {
                model.particle_uv = Some(missing_uv);
            }
            break;
        }
        let (sprite, uv) = match texture_location(resolved) {
            Some(loc) => match sprites.get(&loc) {
                Some(s) => (Some(s), s.uv),
                None => (None, missing_uv),
            },
            None => (None, missing_uv),
        };
        if i == 0 {
            // java: `ItemModelGenerator.makeItemModel` aliases `particle` to `layer0`.
            model.particle_uv = Some(uv);
        }
        let quads = match sprite {
            Some(s) => bake_layer(i as i32, resolved, s, uv),
            None => bake_layer(i as i32, resolved, &placeholder_sprite(), missing_uv),
        };
        model.general_quads.extend(quads);
    }
    model
}

fn placeholder_sprite() -> Sprite {
    Sprite {
        location: ResourceLocation::minecraft("missingno"),
        width: 16,
        height: 16,
        uv: UvRect::FULL,
        frames: vec![crate::sprite::SpriteFrame {
            rgba: vec![0xFFu8; 16 * 16 * 4],
            duration_ticks: None,
        }],
        default_duration_ticks: 1,
        interpolate: false,
        texture_metadata: Default::default(),
    }
}

/// Bake Java's shared `ModelBakery.MODEL_MISSING` cube for an item-model miss.
///
/// `ModelManager.getModel` does not return an empty model for an unregistered
/// `(Item, metadata)` key: it returns the `builtin/missing` model, whose six
/// cull-face quads all sample the shared `missingno` sprite.  Keep this helper
/// beside the item registry so the fallback can use the current bake's
/// stitched `missingno` rectangle rather than accidentally addressing the
/// whole atlas with `UvRect::FULL`.
pub(super) fn missing_item_model(missing_uv: UvRect) -> BakedModel {
    let mut faces = HashMap::new();
    for name in ["down", "up", "north", "south", "west", "east"] {
        faces.insert(
            name.to_owned(),
            RawFace {
                uv: Some([0.0, 0.0, 16.0, 16.0]),
                texture: "missingno".to_owned(),
                cullface: Some(name.to_owned()),
                rotation: 0,
                tintindex: -1,
            },
        );
    }
    let element = RawElement {
        from: [0.0, 0.0, 0.0],
        to: [16.0, 16.0, 16.0],
        rotation: None,
        faces,
        shade: true,
    };
    let mut model = BakedModel {
        gui_3d: true,
        particle_uv: Some(missing_uv),
        ..BakedModel::default()
    };
    // Use the same canonical facing order as the ordinary model baker.  The
    // bucket is selected from the JSON cullface, exactly as `bakeModel` does.
    for facing in [
        Facing::Down,
        Facing::Up,
        Facing::North,
        Facing::South,
        Facing::West,
        Facing::East,
    ] {
        let face = &element.faces[facing_name(facing)];
        model.face_quads[facing.index() as usize].push(bake_quad(
            &element,
            facing,
            face,
            missing_uv,
            BakeRotation::default(),
        ));
    }
    model
}

/// `"blocks/stone"` (a resolved model texture value) -> `minecraft:textures/blocks/stone`.
pub(super) fn texture_location(resolved: &str) -> Option<ResourceLocation> {
    if resolved == "missingno" || resolved.is_empty() {
        return None;
    }
    ResourceLocation::parse(resolved)
        .ok()
        .map(|l| l.with_prefix("textures/"))
}

struct InstalledItemModels {
    models: HashMap<u32, BakedModel>,
    missing: BakedModel,
}

static ITEM_MODELS: super::HotSwap<InstalledItemModels> = super::HotSwap::new();

pub(super) fn install_item_models(models: HashMap<u32, BakedModel>, missing: BakedModel) {
    ITEM_MODELS.install(InstalledItemModels { models, missing });
}

/// [`install_item_models`] under the block registry's bake key (see
/// [`super::bake::install_keyed`]).
pub(super) fn install_item_models_keyed(
    models: HashMap<u32, BakedModel>,
    missing: BakedModel,
    key: Option<String>,
) {
    ITEM_MODELS.install_keyed(key, InstalledItemModels { models, missing });
}

/// Replace the item models on a resource-pack reload. Called by [`super::reinstall`] so the item
/// registry and the block registry always describe the same pack stack.
/// Under the same key as the block registry, so the two reuse the same earlier generation.
pub(super) fn reinstall_item_models_keyed(
    models: HashMap<u32, BakedModel>,
    missing: BakedModel,
    key: Option<String>,
) {
    ITEM_MODELS.reinstall_keyed(key, InstalledItemModels { models, missing });
}

/// The baked model for an item's default (undamaged, subtype-0) appearance.
///
/// Additive S09 API — items aren't in the frozen `model_for`/`sprite` contract (which is
/// block-state shaped), but item model baking is an explicit S09 task, so this is the
/// equivalent entry point S12/S17 (item and inventory rendering) can build on. Returns an
/// empty model before [`super::bake_all`]/`install` has run; after an install, an item or
/// metadata key with no registered model gets Java's shared `builtin/missing` cube.
#[must_use]
pub fn item_model_for(id: ItemId) -> &'static BakedModel {
    item_model_for_damage(id, 0)
}

/// The baked model selected for an item stack's damage/metadata value.
///
/// java: `ItemModelMesher.getMetadata` + `getIndex`. Damageable tools collapse to metadata zero;
/// subtype items retain the full unsigned-short metadata. The four constant mesh definitions and
/// potion's splash-bit definition are canonicalised before the `(item id << 16 | metadata)`
/// lookup. A missing registration selects the installed bake's shared `builtin/missing` cube,
/// rather than silently disappearing or aliasing another subtype's geometry.
#[must_use]
pub fn item_model_for_damage(id: ItemId, damage: i16) -> &'static BakedModel {
    static EMPTY: OnceLock<BakedModel> = OnceLock::new();
    let metadata = canonical_metadata(id, damage.max(0) as u16);
    let key = item_model_key(id, metadata);
    match ITEM_MODELS.load() {
        Some(registry) => registry.models.get(&key).unwrap_or(&registry.missing),
        None => EMPTY.get_or_init(BakedModel::default),
    }
}

/// Resolve a stack's baked model and its live item-property override as one coherent lookup.
///
/// java: `ItemModelMesher.getItemModel(stack)` followed by the `TextureMap` registrations for
/// `Items.compass` and `Items.clock`.  The model is still the generated icon (including the
/// missing-model fallback before/after an asset install); the property tells downstream atlas
/// consumers which dynamic selector must be fed the current world context.
#[must_use]
pub fn item_model_override_for(id: ItemId, damage: i16) -> ItemModelOverride {
    ItemModelOverride {
        model: item_model_for_damage(id, damage),
        property: ItemModelProperty::for_item(id),
    }
}

/// The baked model for one explicitly registered item-model variant.
///
/// Unlike [`item_model_for_damage`], this does not canonicalise damageable items to metadata
/// zero. Vanilla's runtime item renderer needs that distinction for models selected by item-use
/// state rather than stack damage, notably `bow_pulling_0..2` (registered beside `bow` at raw
/// indices `1..=3`). Ordinary stack rendering should keep using [`item_model_for_damage`].
#[must_use]
pub fn item_model_for_variant(id: ItemId, variant: u16) -> &'static BakedModel {
    static EMPTY: OnceLock<BakedModel> = OnceLock::new();
    let key = item_model_key(id, variant);
    match ITEM_MODELS.load() {
        Some(registry) => registry.models.get(&key).unwrap_or(&registry.missing),
        None => EMPTY.get_or_init(BakedModel::default),
    }
}

/// Resolve an explicitly registered model variant and its live item-property override.
///
/// This is the stateful counterpart to [`item_model_override_for`] for use-state model keys such
/// as `bow_pulling_0..2`; the property discriminator still comes from the item registry entry,
/// not from the variant name.
#[must_use]
pub fn item_model_override_for_variant(id: ItemId, variant: u16) -> ItemModelOverride {
    ItemModelOverride {
        model: item_model_for_variant(id, variant),
        property: ItemModelProperty::for_item(id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opaque_square(w: i32, h: i32) -> Vec<u8> {
        vec![0xFFu8; (w * h * 4) as usize]
    }

    #[test]
    fn fully_opaque_square_has_no_interior_spans_only_border() {
        // A solid NxN square's only transparent neighbours are outside the image, so every
        // border pixel emits exactly one span per side — the perimeter, not the interior.
        let rgba = opaque_square(4, 4);
        let spans = detect_spans(&rgba, 4, 4);
        // 4 directions, each contributes at least one span (the whole 0..3 border run per
        // side, since every border pixel on that side is transparent-adjacent that direction
        // and they all share the same "fixed" coordinate).
        let by_dir = |d: SpanFacing| spans.iter().filter(|s| s.facing == d).count();
        assert_eq!(by_dir(SpanFacing::Up), 1);
        assert_eq!(by_dir(SpanFacing::Down), 1);
        assert_eq!(by_dir(SpanFacing::Left), 1);
        assert_eq!(by_dir(SpanFacing::Right), 1);
    }

    #[test]
    fn fully_transparent_image_has_no_spans() {
        let rgba = vec![0u8; 4 * 4 * 4];
        let spans = detect_spans(&rgba, 4, 4);
        assert!(spans.is_empty());
    }

    #[test]
    fn animated_icons_union_the_silhouette_of_every_frame() {
        let mut first = vec![0u8; 2 * 4];
        first[3] = 0xFF;
        let mut second = vec![0u8; 2 * 4];
        second[7] = 0xFF;

        let frame_zero = detect_spans(&first, 2, 1);
        let mut union = Vec::new();
        detect_spans_into(&first, 2, 1, 0, &mut union);
        detect_spans_into(&second, 2, 1, 0, &mut union);

        let up_zero = frame_zero
            .iter()
            .find(|span| span.facing == SpanFacing::Up)
            .unwrap();
        let up_union = union
            .iter()
            .find(|span| span.facing == SpanFacing::Up)
            .unwrap();
        assert_eq!((up_zero.min, up_zero.max), (0, 0));
        assert_eq!((up_union.min, up_union.max), (0, 1));
        assert_ne!(
            up_zero.max, up_union.max,
            "frame-zero-only mutant must bite"
        );

        let sprite = Sprite {
            location: ResourceLocation::minecraft("textures/items/animated_test"),
            width: 2,
            height: 1,
            uv: UvRect::FULL,
            frames: vec![
                crate::sprite::SpriteFrame {
                    rgba: first,
                    duration_ticks: None,
                },
                crate::sprite::SpriteFrame {
                    rgba: second,
                    duration_ticks: None,
                },
            ],
            default_duration_ticks: 1,
            interpolate: false,
            texture_metadata: Default::default(),
        };
        assert_eq!(
            bake_layer(0, "items/animated_test", &sprite, UvRect::FULL).len(),
            8,
            "two core quads plus the six-span union; frame zero alone emits only six"
        );
    }

    /// 1.7's edges: identical to vanilla for 0/255 alpha, and a solid texel
    /// beside an alpha 1..=25 texel gets the side face vanilla leaves out (MC-70336).
    #[test]
    fn old_item_edges_close_the_gap_beside_faint_texels_and_leave_binary_icons_alone() {
        // A 4x4 icon: a solid 2x2 block at texels (1..=2, 1..=2).
        let mut binary = vec![0u8; 4 * 4 * 4];
        for (x, y) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            binary[(y * 4 + x) * 4..][..4].copy_from_slice(&[200, 50, 50, 255]);
        }
        let sprite = |rgba: Vec<u8>| Sprite {
            location: ResourceLocation::minecraft("textures/items/edge_test"),
            width: 4,
            height: 4,
            uv: UvRect::FULL,
            frames: vec![crate::sprite::SpriteFrame {
                rgba,
                duration_ticks: None,
            }],
            default_duration_ticks: 1,
            interpolate: false,
            texture_metadata: Default::default(),
        };
        // Binary alpha: quad for quad the same as vanilla.
        assert_eq!(
            bake_generated_layer_17(0, &binary, 4, 4, UvRect::FULL),
            bake_layer(0, "items/edge_test", &sprite(binary.clone()), UvRect::FULL)
        );
        // Add a faint (alpha 10) texel at (3, 1), right of the block's top row.
        let mut faint = binary;
        faint[(4 + 3) * 4..][..4].copy_from_slice(&[200, 50, 50, 10]);
        let vanilla = bake_layer(0, "items/edge_test", &sprite(faint.clone()), UvRect::FULL);
        let old = bake_generated_layer_17(0, &faint, 4, 4, UvRect::FULL);
        // The side faces in the plane x = `x` (block units), and the height they cover.
        let sides_at = |quads: &[BakedQuad], x: f32| -> (usize, f32) {
            let sides: Vec<&BakedQuad> = quads
                .iter()
                .filter(|q| matches!(q.face, Some(Facing::East | Facing::West)))
                .filter(|q| q.verts.iter().all(|v| (v.pos[0] - x).abs() < 1e-6))
                .collect();
            let height = sides
                .iter()
                .map(|q| {
                    let lo = q.verts.iter().map(|v| v.pos[1]).fold(f32::MAX, f32::min);
                    let hi = q.verts.iter().map(|v| v.pos[1]).fold(f32::MIN, f32::max);
                    hi - lo
                })
                .sum();
            (sides.len(), height)
        };
        // Vanilla: the faint texel counts as solid, so the block's right side at x = 3/4 covers
        // only row 2 and the gap is row 1; the faint texel gets its own (discarded) side at x = 1.
        let (count, height) = sides_at(&vanilla, 0.75);
        assert_eq!(count, 1);
        assert!((height - 0.25).abs() < 1e-6, "vanilla: one row, {height}");
        assert_eq!(sides_at(&vanilla, 1.0).0, 1);
        // 1.7: the block's right side covers both rows, and the faint texel has no side.
        let (count, height) = sides_at(&old, 0.75);
        assert_eq!(count, 1);
        assert!((height - 0.5).abs() < 1e-6, "1.7: both rows, {height}");
        assert_eq!(sides_at(&old, 1.0).0, 0, "a faint texel is empty in 1.7");
    }

    #[test]
    fn texture_location_resolves_namespace_and_prefix() {
        let loc = texture_location("blocks/stone").unwrap();
        assert_eq!(loc.to_string(), "minecraft:textures/blocks/stone");
        assert!(texture_location("missingno").is_none());
    }

    #[test]
    fn generated_model_produces_geometry_for_a_solid_icon() {
        let sprite = Sprite {
            location: ResourceLocation::minecraft("textures/items/stick"),
            width: 4,
            height: 4,
            uv: UvRect::FULL,
            frames: vec![crate::sprite::SpriteFrame {
                rgba: opaque_square(4, 4),
                duration_ticks: None,
            }],
            default_duration_ticks: 1,
            interpolate: false,
            texture_metadata: Default::default(),
        };
        let mut sprites = HashMap::new();
        sprites.insert(ResourceLocation::minecraft("textures/items/stick"), sprite);
        let mut textures = HashMap::new();
        textures.insert("layer0".to_string(), "items/stick".to_string());

        let model = expand_builtin_generated(&textures, &sprites, UvRect::FULL);
        // core (2 quads) + at least the 4 perimeter spans.
        assert!(
            model.general_quads.len() >= 6,
            "{}",
            model.general_quads.len()
        );
        assert!(!model.ambient_occlusion);
        assert!(
            !model.gui_3d,
            "ItemModelGenerator output is flat in GUI/world-item policy"
        );
        assert_eq!(model.particle_uv, Some(UvRect::FULL));
        assert_eq!(model.general_quads[0].face, Some(Facing::North));
        assert_eq!(model.general_quads[1].face, Some(Facing::South));
    }

    #[test]
    fn stops_at_the_first_missing_layer() {
        let mut textures = HashMap::new();
        textures.insert("layer0".to_string(), "missingno".to_string());
        textures.insert("layer1".to_string(), "items/should_be_ignored".to_string());
        let model = expand_builtin_generated(&textures, &HashMap::new(), UvRect::FULL);
        assert!(model.general_quads.is_empty());
        assert_eq!(model.particle_uv, Some(UvRect::FULL));
    }

    #[test]
    fn missing_item_cube_keeps_the_three_dimensional_baked_policy() {
        let model = missing_item_model(UvRect::FULL);
        assert!(model.gui_3d);
        assert_eq!(model.face_quads.iter().map(Vec::len).sum::<usize>(), 6);
    }

    #[test]
    fn live_item_property_override_keeps_compass_clock_and_fallback_distinct() {
        let compass = mc_data::item_by_name("minecraft:compass").expect("compass");
        let clock = mc_data::item_by_name("minecraft:clock").expect("clock");
        let stick = mc_data::item_by_name("minecraft:stick").expect("ordinary item");

        assert_eq!(
            ItemModelProperty::for_item(compass),
            ItemModelProperty::Compass
        );
        assert_eq!(ItemModelProperty::for_item(clock), ItemModelProperty::Clock);
        assert_eq!(ItemModelProperty::for_item(stick), ItemModelProperty::None);
        assert!(ItemModelProperty::Compass.is_dynamic());
        assert!(ItemModelProperty::Clock.is_dynamic());
        assert!(!ItemModelProperty::None.is_dynamic());

        // The property wrapper must not fork model lookup: before an asset install all three
        // still receive the same empty fallback as the underlying API.
        for id in [compass, clock, stick] {
            let selection = item_model_override_for(id, 0);
            assert!(
                core::ptr::eq(selection.model, item_model_for_damage(id, 0)),
                "property lookup must preserve the baked-model reference"
            );
            assert_eq!(selection.gui_3d(), selection.model.gui_3d);
        }
    }

    #[test]
    fn registrations_cover_vanillas_arbitrary_names_and_metadata() {
        let planks = registrations_for("planks");
        assert_eq!(planks.len(), 6);
        assert_eq!(planks[0].model, "oak_planks");
        assert_eq!(planks[5].model, "dark_oak_planks");

        let door = registrations_for("wooden_door");
        assert_eq!(door[0].model, "oak_door");
        let dye = registrations_for("dye");
        assert_eq!(dye[0].model, "dye_black");
        assert_eq!(dye[15].model, "dye_white");

        // Biting controls for the two tempting but wrong conventions: registry-name-only
        // lookup and block-colour order for dye damage.
        assert_ne!(door[0].model, "wooden_door");
        assert_ne!(dye[0].model, "dye_white");
    }

    #[test]
    fn metadata_canonicalisation_matches_item_model_mesher() {
        let sword = mc_data::item_by_name("diamond_sword").unwrap();
        let potion = mc_data::item_by_name("potion").unwrap();
        let spawn_egg = mc_data::item_by_name("spawn_egg").unwrap();
        let wool = mc_data::item_by_name("wool").unwrap();

        assert_eq!(canonical_metadata(sword, 123), 0);
        assert_eq!(canonical_metadata(potion, 7), 0);
        assert_eq!(canonical_metadata(potion, 0x4007), 0x4000);
        assert_eq!(canonical_metadata(spawn_egg, 54), 0);
        assert_eq!(canonical_metadata(wool, 14), 14);

        // Mutant controls: raw durability and the whole potion damage value are both wrong.
        assert_ne!(canonical_metadata(sword, 123), 123);
        assert_ne!(canonical_metadata(potion, 0x4007), 0x4007);
    }
}
