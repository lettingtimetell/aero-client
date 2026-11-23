//! Item rendering in slots: 2D flat icons, a simple isometric path for block items, the stack
//! count, and the durability bar. java: `RenderItem.java` (`renderItemAndEffectIntoGUI`,
//! `renderItemOverlayIntoGUI`).
//!
//! # The 2D/3D split, and how each is projected
//!
//! Vanilla asks the baked model `isGui3d()`: a flat generated icon (most items) says no and is
//! drawn as a plain sprite; a real block model says yes and is drawn through its baked GUI
//! display transform plus `RenderItem.setupGuiTransform`'s 3-D camera, which is what makes a
//! block look like a little cube in the inventory while a sword looks like a flat icon. The
//! production matrix below composes the model transform with the setup's 210-degree X and
//! -135-degree Y rotations. `mc_assets::BakedModel`
//! carries no such flag (see its module doc), so this port uses a bounded `ItemBlock` proxy.
//! `ItemLilyPad` is the necessary 1.8.9 exception: it is an `ItemBlock` but its item model is
//! `builtin/generated`, so vanilla's generated model returns `isGui3d = false` and it draws as
//! a flat sprite. The remaining block items retain the isometric path here.
//!
//! [`draw_flat_icon`] projects a baked model's quads with an **orthographic front view that
//! discards Z** — for the flat `builtin/generated` icons every quad's `(x, y)` corners are
//! already axis-aligned (they are screen-facing planes with only a Z offset between front and
//! back layers), so a plain bounding box of the projected corners *is* the exact rectangle,
//! with no approximation. This is verified pixel-exact by a headless GPU test, the same
//! technique S15 used for glyphs and widget art.
//!
//! [`draw_iso_block_icon`] replays every baked face and general quad through the model's GUI
//! display transform and vanilla's tilted GUI camera. This keeps generated and block-shaped
//! models on the same quad path, including per-quad tint indices; the quads are triangulated
//! using their baked boundary winding before projection. The small analytical cube helper is
//! retained only in tests as a geometry control, not as the production renderer.

use glam::{Mat4, Vec3};

use mc_assets::{BakedModel, BakedQuad};
use mc_math::Facing;
use mc_render::entity::tilepass::{
    builtin_tile_item_kind, builtin_tile_item_kind_for_model, is_builtin_entity_model,
    BuiltinSkullOwner, BuiltinTileItem, BuiltinTileItemKind,
};
use mc_world::MapData;

use crate::font::FontRenderer;
use crate::map::{draw_map, dynamic_texture_key, MAP_ICONS_TEXTURE};
use crate::widget::draw::{UiBlend, UiCull, UiDepth, GUI_ITEM_SLIVER_EXTENT};
use crate::widget::{DrawList, UiTexture};

use super::stack_ext::{self, ItemStack};

/// `RenderItem.RES_ITEM_GLINT` — the resource sampled by both 1.8.9 foil passes.
pub const ENCHANTED_ITEM_GLINT_TEXTURE: &str = "textures/misc/enchanted_item_glint.png";

/// The periods used by the two `RenderItem.renderEffect` texture-matrix passes.
pub const GLINT_FIRST_PERIOD_MS: u64 = 3_000;
/// The second foil pass's independent texture-scroll period, in milliseconds.
pub const GLINT_SECOND_PERIOD_MS: u64 = 4_873;

/// Java scales the glint texture matrix by eight before applying each pass's translation and
/// rotation. Keeping this public makes the CPU UV contract auditable without a GPU capture.
pub const GLINT_TEXTURE_SCALE: f32 = 8.0;

/// The packed colour passed to `WorldRenderer.putColor4` by vanilla's `renderEffect`.
/// `-8372020 == 0xFF8040CC`; it is *not* an alpha-tinted approximation.
pub const GLINT_COLOR_ARGB: u32 = 0xFF80_40CC;

/// Exact colour bytes emitted by Java's `renderEffect` overlay.
const GLINT_COLOR: [f32; 4] = [128.0 / 255.0, 64.0 / 255.0, 204.0 / 255.0, 1.0];

/// One of the two animated texture-matrix states used by vanilla's item foil.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItemGlintPass {
    /// The clock period used for this pass, in milliseconds.
    pub period_ms: u64,
    /// The post-scale translation argument (`fraction / 8`) from Java.
    pub translation: f32,
    /// The texture-matrix rotation in degrees.
    pub rotation_degrees: f32,
}

/// Build the two `RenderItem.renderEffect` pass states at a Java-compatible system-clock value.
///
/// Java computes `(Minecraft.getSystemTime() % period) / period / 8` in `float` arithmetic,
/// then applies `scale(8)`, `translate(±f, 0)`, and `rotate(angle)` to the texture matrix. The
/// returned values intentionally preserve that order and precision so callers/tests can inspect
/// the animation contract without opening a display or GPU.
#[must_use]
pub fn item_glint_passes(system_time_ms: u64) -> [ItemGlintPass; 2] {
    let first = (system_time_ms % GLINT_FIRST_PERIOD_MS) as f32
        / GLINT_FIRST_PERIOD_MS as f32
        / GLINT_TEXTURE_SCALE;
    let second = (system_time_ms % GLINT_SECOND_PERIOD_MS) as f32
        / GLINT_SECOND_PERIOD_MS as f32
        / GLINT_TEXTURE_SCALE;
    [
        ItemGlintPass {
            period_ms: GLINT_FIRST_PERIOD_MS,
            translation: first,
            rotation_degrees: -50.0,
        },
        ItemGlintPass {
            period_ms: GLINT_SECOND_PERIOD_MS,
            translation: -second,
            rotation_degrees: 10.0,
        },
    ]
}

/// Apply one Java glint texture-matrix pass to an atlas/model UV.
///
/// OpenGL post-multiplies the calls in `RenderItem.renderEffect`, so the matrix is
/// `scale(8) * translate(±f) * rotate(angle)`. This CPU form is equivalent and deliberately
/// leaves the resulting coordinates outside `0..1`; the UI sampler repeats the glint texture,
/// just as the vanilla texture state does. `pass` must be `0` or `1`.
#[must_use]
pub fn item_glint_uv(uv: [f32; 2], system_time_ms: u64, pass: usize) -> [f32; 2] {
    let state = item_glint_passes(system_time_ms)
        .get(pass)
        .copied()
        .unwrap_or_else(|| panic!("item glint pass must be 0 or 1, got {pass}"));
    let angle = state.rotation_degrees.to_radians();
    let (sin, cos) = angle.sin_cos();
    let rotated_u = uv[0] * cos - uv[1] * sin;
    let rotated_v = uv[0] * sin + uv[1] * cos;
    [
        GLINT_TEXTURE_SCALE * (rotated_u + state.translation),
        GLINT_TEXTURE_SCALE * rotated_v,
    ]
}

/// Read the wall-clock millisecond source used by the UI's live wrapper. Tests use
/// [`item_glint_passes`] / [`item_glint_uv`] with an explicit value instead of racing this clock.
#[must_use]
fn glint_system_time_ms() -> u64 {
    // The one glint clock, so a replay can pin it.
    mc_render::entity::item::item_glint_system_time_ms()
}

/// The baked model, rather than the item class, owns vanilla's GUI presentation policy.
#[must_use]
fn model_uses_gui_3d(model: &BakedModel) -> bool {
    model.gui_3d
}

/// Decode the five item families whose JSON model terminates at `builtin/entity`.
///
/// java: `RenderItem.renderItem` delegates these stacks to
/// `TileEntityItemStackRenderer.renderByItem` instead of traversing the empty sentinel model.
/// Keeping the small stack adapter in the UI crate makes every inventory, hotbar, creative and
/// stats caller reach the same render-side tile-item descriptor without coupling `mc-render` to
/// the wire crate.
#[must_use]
fn builtin_tile_item_from_stack(stack: &ItemStack) -> Option<BuiltinTileItem> {
    let item_id = u16::try_from(stack.item_id).ok()?;
    match builtin_tile_item_kind(item_id)? {
        BuiltinTileItemKind::Chest => Some(BuiltinTileItem::Chest { trapped: false }),
        BuiltinTileItemKind::TrappedChest => Some(BuiltinTileItem::Chest { trapped: true }),
        BuiltinTileItemKind::EnderChest => Some(BuiltinTileItem::EnderChest),
        BuiltinTileItemKind::Banner => Some(BuiltinTileItem::Banner {
            state: banner_item_state(stack),
        }),
        BuiltinTileItemKind::Skull => {
            let skull_type = u8::try_from(stack.damage.max(0)).unwrap_or(u8::MAX);
            if skull_type == 3 {
                // The GUI has no client tab-list borrow, so a
                // legacy name remains the same incomplete profile/default-skin fallback as vanilla.
                if let Some(owner) = player_skull_owner(stack) {
                    return Some(BuiltinTileItem::PlayerSkull { skull_type, owner });
                }
            }
            Some(BuiltinTileItem::Skull { skull_type })
        }
    }
}

/// Resolve an item-stack `SkullOwner` profile for the GUI's builtin/entity path.
///
/// `mc-ui` intentionally cannot depend on `mc-client`, so this mirrors the small profile-property
/// decode in `app::appearance` while handing the actual request to the render crate's shared skin
/// queue.  No HTTP or file I/O happens here; `AppearanceState::skins.poll` owns that boundary.
fn player_skull_owner(stack: &ItemStack) -> Option<BuiltinSkullOwner> {
    // Match TileEntityItemStackRenderer's type-10/type-8 split.
    // The UI cannot query MinecraftServer's profile cache, therefore a name-only owner is kept
    // as a non-fetchable descriptor and renders with its offline-UUID-derived default texture.
    // The GUI's item stack is borrowed immutably, so mirror Java's string-to-compound mutation on
    // a render-local clone.  The mutable client adapter owns the persistent form for callers that
    // can write back to the inventory stack; this keeps the two paths on one NBT/profile decoder.
    //
    // A compound owner is only *read* by the normaliser, so the clone (a whole NBT tree, every
    // frame the icon is drawn) is only needed for the legacy string form.
    let profile = match stack.nbt.as_ref()?.get("SkullOwner") {
        Some(mc_nbt::Nbt::Compound(owner)) => mc_auth::read_game_profile_from_nbt(owner)?,
        _ => {
            let mut normalized = stack.clone();
            let root = normalized.nbt.as_mut()?;
            mc_auth::normalize_legacy_skull_owner(root, None)?
        }
    };
    let Some(id) = profile.id else {
        // TileEntitySkullRenderer calls EntityPlayer.getUUID on this incomplete profile; Java
        // derives the offline UUID from the name and uses it for the default Steve/Alex choice.
        let offline_id = mc_auth::offline::offline_uuid(&profile.name);
        let fallback = mc_render::entity::SkinModel::default_for_uuid(offline_id.as_u128());
        return Some(BuiltinSkullOwner::legacy(&profile.name, fallback));
    };
    let fallback = mc_render::entity::SkinModel::default_for_uuid(id.as_u128());
    // java: inventory icons reach `TileEntitySkullRenderer` too, so
    // they share the world heads' descriptor cache (`SkinManager.skinCacheLoader`: id + name,
    // 15 s expire-after-access) and their hash-named skins; the request still runs per draw.
    Some(mc_render::entity::skin::player_head_textures(
        id.as_u128(),
        &profile.name,
        || gui_profile_textures(&profile, fallback),
        |textures| match &textures.skin {
            Some((key, url)) => {
                mc_render::entity::skin::request_shared_skin(key, url, textures.model);
                BuiltinSkullOwner::new(key.as_str(), textures.model, fallback)
            }
            // java: no SKIN in the textures map: `DefaultPlayerSkin.getDefaultSkin(uuid)`.
            None => BuiltinSkullOwner::new(
                format!("mc-render/entity/skin/{id}"),
                textures.model,
                fallback,
            ),
        },
    ))
}

/// java: `MinecraftSessionService.getTextures(profile, false)` for a GUI head icon: the first
/// `textures` property, decoded once per distinct value (SKIN-4), as the shared descriptor
/// cache stores it.
fn gui_profile_textures(
    profile: &mc_auth::GameProfile,
    fallback: mc_render::entity::SkinModel,
) -> mc_render::entity::skin::PlayerHeadTextures {
    let Some(property) = profile
        .properties
        .iter()
        .find(|property| property.name == "textures")
    else {
        return mc_render::entity::skin::PlayerHeadTextures {
            skin: None,
            model: fallback,
        };
    };
    GUI_SKULL_TEXTURES.with(|memo| {
        memo.borrow_mut()
            .with(&property.value, decode_profile_textures, |textures| {
                let model = textures.as_ref().map_or(fallback, |(_, model)| *model);
                let skin = textures
                    .as_ref()
                    .and_then(|(url, _)| url.as_deref())
                    .and_then(|url| {
                        mc_render::entity::skin::skin_key_for_texture_url(url)
                            .map(|key| (key, url.to_owned()))
                    });
                mc_render::entity::skin::PlayerHeadTextures { skin, model }
            })
    })
}

/// A decoded `textures` property: the skin URL, if any, and the declared arm model.
type GuiSkullTextures = Option<(Option<String>, mc_render::entity::SkinModel)>;

thread_local! {
    /// Player-head icons are drawn every frame; their profile decode is pure, so it runs once
    /// per distinct `textures` value (SKIN-4).
    static GUI_SKULL_TEXTURES: std::cell::RefCell<
        mc_render::entity::skin::TexturesPropertyMemo<GuiSkullTextures>,
    > = std::cell::RefCell::new(mc_render::entity::skin::TexturesPropertyMemo::new());
}

/// Resolve a stack against a model that has already selected Java's custom item renderer.
///
/// The strict stack adapter covers the five shipped 1.8.9 ids.  If a resource pack (or a future
/// item family) points another item at `builtin/entity`, Java's final `else` renders a normal
/// chest; an ordinary/missing model must remain on the normal model path.  The model-aware helper
/// keeps that distinction local to the GUI caller.
#[must_use]
fn builtin_tile_item_from_stack_for_model(
    stack: &ItemStack,
    model: &BakedModel,
) -> Option<BuiltinTileItem> {
    // Before the first resource bake the registry returns the zero/default model.  Keep the
    // established direct adapter usable for that startup window, but once a real non-empty
    // ordinary model is selected, Java's `isBuiltInRenderer()` gate must win even for a known
    // vanilla id.
    let startup_empty_model = model.is_empty() && !model.gui_3d && model.particle_uv.is_none();
    if !is_builtin_entity_model(model) && !startup_empty_model {
        return None;
    }
    if let Some(item) = builtin_tile_item_from_stack(stack) {
        return Some(item);
    }
    let item_id = u16::try_from(stack.item_id).ok()?;
    let kind = builtin_tile_item_kind_for_model(item_id, model)?;
    match kind {
        BuiltinTileItemKind::Chest => Some(BuiltinTileItem::Chest { trapped: false }),
        BuiltinTileItemKind::TrappedChest => Some(BuiltinTileItem::Chest { trapped: true }),
        BuiltinTileItemKind::EnderChest => Some(BuiltinTileItem::EnderChest),
        BuiltinTileItemKind::Banner => Some(BuiltinTileItem::Banner {
            state: banner_item_state(stack),
        }),
        BuiltinTileItemKind::Skull => {
            let skull_type = u8::try_from(stack.damage.max(0)).unwrap_or(u8::MAX);
            if skull_type == 3 {
                if let Some(owner) = player_skull_owner(stack) {
                    return Some(BuiltinTileItem::PlayerSkull { skull_type, owner });
                }
            }
            Some(BuiltinTileItem::Skull { skull_type })
        }
    }
}

/// The part of `SkinManager.loadProfileTextures` needed by a player skull.
///
/// Keep the authlib host gate in lockstep with `app::appearance::decode_textures_property`: a
/// malformed/off-domain property is a fallback skin, never a reason to fetch an arbitrary URL.
fn decode_profile_textures(value: &str) -> Option<(Option<String>, mc_render::entity::SkinModel)> {
    use base64::Engine as _;

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(value.trim())
        .ok()?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let textures = json.get("textures")?;
    let skin_url = textures
        .get("SKIN")
        .and_then(|skin| skin.get("url"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let cape_url = textures
        .get("CAPE")
        .and_then(|cape| cape.get("url"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    // Plus the downloader's own host rule, as the client's decoder does.
    if !skin_url.iter().chain(cape_url.iter()).all(|url| {
        authlib_texture_domain(url) && mc_render::entity::skin::is_mojang_texture_url(url)
    }) {
        return None;
    }
    let model = textures
        .get("SKIN")
        .and_then(|skin| skin.get("metadata"))
        .and_then(|metadata| metadata.get("model"))
        .and_then(serde_json::Value::as_str)
        .map_or(
            mc_render::entity::SkinModel::Classic,
            mc_render::entity::SkinModel::parse,
        );
    Some((skin_url, model))
}

/// authlib 1.5.21's case-sensitive `.minecraft.net`/`.mojang.com` host suffix check.
fn authlib_texture_domain(url: &str) -> bool {
    if url.is_empty()
        || !url.is_ascii()
        || url
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ')
    {
        return false;
    }
    let authority = if let Some(rest) = url.strip_prefix("//") {
        rest
    } else {
        let Some(colon) = url.find(':') else {
            return false;
        };
        let scheme = &url[..colon];
        if scheme.is_empty()
            || !scheme.bytes().enumerate().all(|(index, byte)| {
                byte.is_ascii_alphabetic() || (index != 0 && matches!(byte, b'+' | b'-' | b'.'))
            })
        {
            return false;
        }
        let Some(rest) = url[colon + 1..].strip_prefix("//") else {
            return false;
        };
        rest
    };
    let Some(authority) = authority
        .split(['/', '?', '#'])
        .next()
        .filter(|authority| !authority.is_empty())
    else {
        return false;
    };
    let host_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    if host_port.starts_with('[') {
        return false;
    }
    let host = host_port
        .split_once(':')
        .map_or(host_port, |(host, _)| host);
    !host.is_empty() && (host.ends_with(".minecraft.net") || host.ends_with(".mojang.com"))
}

/// `TileEntityBanner.setItemValues`'s item-state extraction.
///
/// `Base` is accepted through Java's broad numeric `hasKey(..., 99)` gate and then
/// `getInteger`; malformed-present values therefore become zero rather than falling back to the
/// stack damage. Pattern entries require a compound, a string id known to the banner registry,
/// and are capped later by the render-side layered-texture helper.
#[must_use]
fn banner_item_state(stack: &ItemStack) -> mc_world::BannerState {
    let mut state = mc_world::BannerState {
        base_color: stack.damage.max(0) as i32 & 15,
        patterns: Vec::new(),
        initialized: true,
    };
    let Some(tag) = stack
        .nbt
        .as_ref()
        .and_then(|root| root.get_compound("BlockEntityTag"))
    else {
        return state;
    };

    if tag.contains_numeric_type("Base") {
        state.base_color = tag.get_i32("Base").unwrap_or(0);
    }

    if let Some(patterns) = tag.get_list("Patterns") {
        for entry in patterns {
            let Some(pattern) = entry.as_compound() else {
                continue;
            };
            let Some(id) = pattern.get_str("Pattern") else {
                continue;
            };
            if mc_world::tileent::banner_pattern_name(id).is_none() {
                continue;
            }
            state
                .patterns
                .push((id.to_owned(), pattern.get_i32("Color").unwrap_or(0)));
        }
    }
    state
}

/// java: `ItemStack.getItem().showDurabilityBar(stack)`, which for the vanilla default
/// (`Item.showDurabilityBar`) is just "damageable and currently damaged".
#[must_use]
pub fn durability_fraction(stack: &ItemStack) -> Option<f32> {
    if !stack_ext::is_item_damaged(stack) {
        return None;
    }
    let max = f32::from(stack_ext::props(stack).max_damage);
    if max <= 0.0 {
        return None;
    }
    let damage = f32::from(stack.damage).clamp(0.0, max);
    Some(1.0 - damage / max)
}

/// java: `RenderItem.renderItemOverlayIntoGUI` — a linear red/green cross-fade, where `i` is
/// `round(255 * health)`, yielding `(255-i, i, 0)`. This intentionally is not the HSV ramp used
/// by other Minecraft versions.
#[must_use]
pub fn durability_bar_color(health_fraction: f32) -> u32 {
    let green = (f64::from(health_fraction.clamp(0.0, 1.0)) * 255.0).round() as i32;
    durability_bar_color_from_green(green)
}

fn durability_bar_color_from_green(green: i32) -> u32 {
    let green = green.clamp(0, 255) as u32;
    0xFF00_0000 | ((255 - green) << 16) | (green << 8)
}

fn durability_track_color(green: i32) -> u32 {
    let green = green.clamp(0, 255) as u32;
    0xFF00_0000 | (((255 - green) / 4) << 16) | (64 << 8)
}

fn durability_bar_metrics(stack: &ItemStack) -> Option<(i32, i32)> {
    if !stack_ext::is_item_damaged(stack) {
        return None;
    }
    let max = f64::from(stack_ext::props(stack).max_damage);
    if max <= 0.0 {
        return None;
    }
    let damage = f64::from(stack.damage);
    let width = (13.0 - damage * 13.0 / max).round() as i32;
    let green = (255.0 - damage * 255.0 / max).round() as i32;
    Some((width, green))
}

/// Draws the durability bar at its fixed slot-relative position (`x+2, y+13`, `13x2` black
/// backing, a `12x1` dark track, then a `1`px-tall coloured foreground scaled to the remaining
/// durability) if `stack` is damaged. No-op otherwise. java:
/// `RenderItem.renderItemOverlayIntoGUI`'s durability-bar block.
pub fn draw_durability_bar(draw: &mut DrawList, x: i32, y: i32, stack: &ItemStack) {
    let Some((width, green)) = durability_bar_metrics(stack) else {
        return;
    };
    draw.rect(x + 2, y + 13, x + 2 + 13, y + 15, 0xFF00_0000);
    draw.rect(
        x + 2,
        y + 13,
        x + 2 + 12,
        y + 14,
        durability_track_color(green),
    );
    draw.rect(
        x + 2,
        y + 13,
        x + 2 + width,
        y + 14,
        durability_bar_color_from_green(green),
    );
}

/// Draws the stack-size count, right/bottom-aligned in the slot with a drop shadow.
/// No-op only for a stack of `1`; Java still draws zero and negative authoritative counts in red.
/// java: the count-text half of `RenderItem.renderItemOverlayIntoGUI`.
pub fn draw_count_text(
    draw: &mut DrawList,
    font: &mut FontRenderer,
    x: i32,
    y: i32,
    stack: &ItemStack,
) {
    draw_count_text_with_override(draw, font, x, y, stack, None);
}

/// `RenderItem.renderItemOverlayIntoGUI`'s count string: `"§c" + count` when the count is below
/// one, otherwise the decimal count, written into `buf`.
fn count_text(buf: &mut [u8; 16], count: i32) -> &str {
    use std::io::Write as _;
    let len = {
        let mut cursor = std::io::Cursor::new(&mut buf[..]);
        let written = if count < 1 {
            write!(cursor, "\u{a7}c{count}")
        } else {
            write!(cursor, "{count}")
        };
        written.expect("`§c` and any i32 fit in 16 bytes");
        cursor.position() as usize
    };
    std::str::from_utf8(&buf[..len]).expect("formatted from a UTF-8 literal and ASCII digits")
}

/// Draws the stack count or the caller-provided drag-preview override text.
pub fn draw_count_text_with_override(
    draw: &mut DrawList,
    font: &mut FontRenderer,
    x: i32,
    y: i32,
    stack: &ItemStack,
    override_text: Option<&str>,
) {
    if stack.count == 1 && override_text.is_none() {
        return;
    }
    // C2-15/CONT-4: format into a stack buffer instead of a `String` per counted slot per frame
    // (and borrow an override instead of copying it). Same bytes as the former `format!`.
    let mut buf = [0u8; 16];
    let text = match override_text {
        Some(text) => text,
        None => count_text(&mut buf, stack.count),
    };
    let width = font.string_width(text);
    font.draw_string_with_shadow(
        draw,
        text,
        (x + 17 - width) as f32,
        (y + 9) as f32,
        0x00FF_FFFF,
    );
}

/// Project a baked model's quads as a flat, head-on 2D icon into a 16×16 box at `(x, y)`. Every
/// quad's projected `(x, y)` bounding box is drawn as one textured rect (see the module doc for
/// why that is exact, not an approximation, for this kind of model). `texture` must already be
/// the atlas `model`'s UVs were baked against (see [`crate::widget::UiRenderer::upload`] —
/// callers upload `mc_assets::BakedAssets::atlas`'s pixels under whatever [`UiTexture`] key they
/// pass here).
pub fn draw_flat_icon(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    model: &BakedModel,
) {
    draw_flat_icon_colored(draw, texture, x, y, model, |_| [1.0; 4]);
}

fn draw_flat_icon_colored(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    model: &BakedModel,
    mut color: impl FnMut(Option<u8>) -> [f32; 4],
) {
    draw_baked_gui_model(draw, texture, x, y, model, false, |quad| {
        color(quad.tint_index)
    });
}

/// Resolve `Item.getColorFromItemStack` for the item families this renderer can express.
///
/// The result is opaque RGB (not ARGB): callers that do not have a tinted baked quad pass
/// `None` and receive vanilla's white default.  Keeping this stack-aware policy public lets
/// the dropped-item renderer use the same exact NBT and metadata rules as GUI icons.
#[must_use]
pub fn item_quad_color_rgb(stack: &ItemStack, tint_index: Option<u8>) -> u32 {
    if tint_index == Some(0) {
        if let Some(rgb) = super::tooltip::potion_liquid_color(stack) {
            return rgb;
        }
    }
    if tint_index == Some(1) {
        if let Some(rgb) = super::tooltip::firework_star_color(stack) {
            return rgb;
        }
    }
    if let Some(rgb) = spawn_egg_color(stack, tint_index) {
        return rgb;
    }
    if let Some(rgb) = leather_armor_color(stack, tint_index) {
        return rgb;
    }
    if let Some(rgb) = colored_block_color(stack, tint_index) {
        return rgb;
    }
    if let Some(rgb) = banner_color(stack, tint_index) {
        return rgb;
    }
    0xFF_FF_FF
}

fn item_quad_color(stack: &ItemStack, tint_index: Option<u8>) -> [f32; 4] {
    crate::widget::draw::unpack_argb(0xFF00_0000 | item_quad_color_rgb(stack, tint_index))
}

/// `ItemMonsterPlacer.getColorFromItemStack`.  `EntityInfo` is generated from
/// `EntityList.entityEggs`, so it carries the exact 1.8.9 primary/secondary palette rather
/// than a renderer-owned duplicate.
fn spawn_egg_color(stack: &ItemStack, tint_index: Option<u8>) -> Option<u32> {
    if stack_ext::props(stack).name != "minecraft:spawn_egg" {
        return None;
    }
    let entity = mc_data::entity_type(u8::try_from(stack.damage).ok()?)?;
    let info = entity.info();
    match tint_index? {
        0 => info.egg_primary,
        _ => info.egg_secondary,
    }
}

/// `ItemArmor.getColorFromItemStack` for the four leather pieces.  The default is
/// `ArmorMaterial.LEATHER`'s `0xA06540`; Java accepts a custom value only when `display.color`
/// is a `TAG_Int`, not merely any numeric NBT primitive.
fn leather_armor_color(stack: &ItemStack, tint_index: Option<u8>) -> Option<u32> {
    if tint_index != Some(0)
        || !matches!(
            stack_ext::props(stack).name,
            "minecraft:leather_helmet"
                | "minecraft:leather_chestplate"
                | "minecraft:leather_leggings"
                | "minecraft:leather_boots"
        )
    {
        return None;
    }

    let color = stack
        .nbt
        .as_ref()
        .and_then(|root| root.get_compound("display"))
        .and_then(|display| match display.get("color") {
            Some(mc_nbt::Nbt::Int(color)) => Some(*color as u32),
            _ => None,
        })
        .unwrap_or(0xA0_65_40);
    Some(color & 0x00FF_FFFF)
}

/// The block-derived item tint overrides from 1.8.9's `ItemColored`, `ItemLeaves`,
/// `ItemLilyPad`, and `ItemDoublePlant` classes.
///
/// `Block.getRenderColor` is deliberately used instead of the terrain colour-multiplier path:
/// item icons have no world position, so Java asks the block for its position-independent render
/// colour.  The existing terrain seam owns those exact `Block*` overrides and their loaded
/// colormaps; keeping this adapter here avoids a second table that could drift from block
/// particles and the model renderer.
fn colored_block_color(stack: &ItemStack, tint_index: Option<u8>) -> Option<u32> {
    // RenderItem calls this only for a quad carrying a tint index.  In particular, a caller that
    // asks the public resolver for `None` must retain Item's ordinary opaque-white default.
    tint_index?;
    let props = stack_ext::props(stack);
    let block = props.block?;
    let class = props.java_class;

    let metadata = if class == "net.minecraft.item.ItemColored" && block.0 == 31 {
        // BlockTallGrass.EnumType.byMetadata falls back to DEAD_BUSH for every value outside
        // 0..=2.  Do that normalization before packing a BlockState; simply truncating a large
        // or negative item damage to four bits would turn e.g. 3 into a green grass variant.
        match i32::from(stack.damage) {
            0..=2 => stack.damage as u8,
            _ => 0,
        }
    } else {
        // The remaining item overrides either ignore metadata (grass, lily pad, vine) or mask
        // it in their block's own getStateFromMeta implementation.  BlockState::new keeps the
        // same low-four-bit packing used by that metadata boundary, including negative values.
        stack.damage as u8
    };

    let state = mc_data::BlockState::new(block, metadata);
    match class {
        "net.minecraft.item.ItemColored"
        | "net.minecraft.item.ItemLeaves"
        | "net.minecraft.item.ItemLilyPad" => Some(mc_render::terrain::tint::render_color(state)),
        "net.minecraft.item.ItemDoublePlant" => {
            // ItemDoublePlant does not delegate its grass/fern branch through the block: it
            // directly asks ColorizerGrass for the fixed (.5, 1.0) colour. BlockTallGrass's
            // render-colour override is the existing public adapter for that same call.
            match (tint_index, i32::from(stack.damage)) {
                (Some(0), 2 | 3) => Some(mc_render::terrain::tint::render_color(
                    mc_data::BlockState::new(mc_data::BlockId(31), 1),
                )),
                _ => None,
            }
        }
        _ => None,
    }
}

/// `ItemBanner.getColorFromItemStack`'s two render passes.
///
/// The banner's second pass uses `EnumDyeColor.getMapColor().colorValue`, not the standing-banner
/// block's wood map colour.  The generated wool states carry the exact same sixteen
/// `MapColor.colorValue` entries, so they provide one authoritative data-backed lookup without
/// adding a duplicate dye palette to the UI crate.  `getBaseColor` deliberately treats a present
/// non-numeric `Base` as Java's `NBTTagCompound.getInteger` value `0`, rather than falling back
/// to stack metadata.
fn banner_color(stack: &ItemStack, tint_index: Option<u8>) -> Option<u32> {
    if stack_ext::props(stack).name != "minecraft:banner" {
        return None;
    }
    match tint_index? {
        0 => Some(0xFF_FF_FF),
        _ => {
            let damage = stack
                .nbt
                .as_ref()
                .and_then(|root| root.get_compound("BlockEntityTag"))
                .and_then(|tag| {
                    tag.contains_key("Base")
                        .then(|| tag.get_i32("Base").unwrap_or(0))
                })
                .unwrap_or_else(|| i32::from(stack.damage));
            let dye = mc_data::item::dye_color_by_dye_damage(damage);
            Some(
                mc_data::block_props(mc_data::BlockState::new(mc_data::BlockId(35), dye.metadata))
                    .map_color,
            )
        }
    }
}

fn modulate_color(base: [f32; 4], item: [f32; 4]) -> [f32; 4] {
    [
        base[0] * item[0],
        base[1] * item[1],
        base[2] * item[2],
        base[3] * item[3],
    ]
}

// This pre-z-level wrapper is retained for the headless geometry tests and for the legacy
// icon-only seam; live container callers use the explicit z-level variant below.
#[allow(dead_code)]
fn draw_flat_item_icon(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    stack: &ItemStack,
    model: &BakedModel,
) {
    draw_flat_icon_colored(draw, texture, x, y, model, |tint_index| {
        item_quad_color(stack, tint_index)
    });
}

/// The GUI counterpart of `ItemStack.hasEffect()`. Most item classes flow through the shared
/// item predicate, but the UI owns the final render dispatch and must also preserve the two
/// always-foiled classes not represented there: `ItemEditableBook` (`written_book`) and
/// `ItemSimpleFoiled` (`nether_star`). java: `ItemEditableBook.hasEffect`,
/// `ItemSimpleFoiled.hasEffect`, and `RenderItem.renderItem`.
#[must_use]
fn has_gui_effect(stack: &ItemStack) -> bool {
    super::tooltip::has_effect(stack)
        || matches!(
            stack_ext::props(stack).name,
            "minecraft:written_book" | "minecraft:nether_star"
        )
}

/// Apply the GUI foil after the item geometry has been selected. Keeping this tiny dispatch seam
/// separate makes the predicate observable with concrete baked geometry in headless tests.
// This pre-z-level wrapper is retained for the legacy icon-only seam; live callers use the
// explicit z-level variant so ordinary and carried stacks receive Java's distinct depth.
#[allow(dead_code)]
fn draw_item_glint_if_needed(
    draw: &mut DrawList,
    x: i32,
    y: i32,
    stack: &ItemStack,
    model: &BakedModel,
    gui_3d: bool,
) {
    draw_item_glint_if_needed_at_z_level(
        draw,
        x,
        y,
        stack,
        model,
        gui_3d,
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_item_glint_if_needed_at_z_level(
    draw: &mut DrawList,
    x: i32,
    y: i32,
    stack: &ItemStack,
    model: &BakedModel,
    gui_3d: bool,
    item_render_z_level: f32,
    slivers: Option<&[bool]>,
) {
    if has_gui_effect(stack) {
        // The `shiny_pots` mod: a potion's glint fills the whole square, as in 1.7,
        // instead of following the bottle. Off (the default) is the model foil below.
        if let Some(area) = crate::modvisual::shiny_pots() {
            if stack_ext::try_props(stack).is_some_and(|props| props.name == "minecraft:potion") {
                crate::modvisual::draw_square_glint(draw, x, y, area, glint_system_time_ms());
                return;
            }
        }
        draw_glint_overlay_at_z_level(
            draw,
            x,
            y,
            model,
            gui_3d,
            glint_system_time_ms(),
            item_render_z_level,
            slivers,
        );
    }
}

/// Deterministic/testable form of [`draw_item_glint_if_needed`].
#[cfg(test)]
fn draw_item_glint_if_needed_at(
    draw: &mut DrawList,
    x: i32,
    y: i32,
    stack: &ItemStack,
    model: &BakedModel,
    gui_3d: bool,
    system_time_ms: u64,
) {
    if has_gui_effect(stack) {
        draw_glint_overlay_at(draw, x, y, model, gui_3d, system_time_ms);
    }
}

/// Emit the two geometry-matched foil passes that `RenderItem.renderEffect` makes after an item
/// model. The texture is a regular UI resource, so `UiPhase::ensure_list_textures` loads it on
/// the same live path as slot, creative and hotbar sprites.
// This pre-z-level wrapper is retained for the deterministic legacy test seam; live callers use
// the explicit z-level variant so the overlay lands on its item's depth layer.
#[allow(dead_code)]
fn draw_glint_overlay(draw: &mut DrawList, x: i32, y: i32, model: &BakedModel, gui_3d: bool) {
    draw_glint_overlay_at_z_level(
        draw,
        x,
        y,
        model,
        gui_3d,
        glint_system_time_ms(),
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
        None,
    );
}

/// Deterministic/testable form of [`draw_glint_overlay`] that carries the exact Java clock value
/// into both animated texture-matrix passes.
#[allow(dead_code)]
fn draw_glint_overlay_at(
    draw: &mut DrawList,
    x: i32,
    y: i32,
    model: &BakedModel,
    gui_3d: bool,
    system_time_ms: u64,
) {
    draw_glint_overlay_at_z_level(
        draw,
        x,
        y,
        model,
        gui_3d,
        system_time_ms,
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_glint_overlay_at_z_level(
    draw: &mut DrawList,
    x: i32,
    y: i32,
    model: &BakedModel,
    gui_3d: bool,
    system_time_ms: u64,
    item_render_z_level: f32,
    slivers: Option<&[bool]>,
) {
    // C2-12: one shared key instead of two `String` allocations per enchanted item per frame.
    static GLINT_TEXTURE_KEY: std::sync::LazyLock<UiTexture> =
        std::sync::LazyLock::new(|| UiTexture::gui(ENCHANTED_ITEM_GLINT_TEXTURE));
    let texture: &UiTexture = &GLINT_TEXTURE_KEY;
    // Java replays `renderModel` twice under two texture matrices.  The CPU list carries the
    // resulting UVs and a dedicated `UiBlend::Glint` batch, while both passes replay the model's
    // actual projected geometry.
    for pass in 0..2 {
        draw_baked_gui_model_with_blend_at_z_level(
            draw,
            texture,
            x,
            y,
            model,
            gui_3d,
            UiBlend::Glint,
            |uv| item_glint_uv(uv, system_time_ms, pass),
            |_| GLINT_COLOR,
            item_render_z_level,
            slivers,
        );
    }
}

/// One unit cube corner, before projection: `(±0.5, ±0.5, ±0.5)`.
#[cfg(test)]
fn cube_corner(dx: f32, dy: f32, dz: f32) -> Vec3 {
    Vec3::new(dx, dy, dz)
}

/// A compact 30/225-degree cube projection used only by the legacy `iso_faces` test helper.
/// Production rendering uses [`baked_gui_matrix`], which composes the baked model's GUI display
/// transform with `RenderItem.setupGuiTransform`. Returns the 8 rotated, unprojected corners.
#[cfg(test)]
fn rotated_cube_corners() -> [Vec3; 8] {
    let m = Mat4::from_rotation_x(30f32.to_radians()) * Mat4::from_rotation_y(225f32.to_radians());
    let mut out = [Vec3::ZERO; 8];
    let mut i = 0;
    for dx in [-0.5f32, 0.5] {
        for dy in [-0.5f32, 0.5] {
            for dz in [-0.5f32, 0.5] {
                out[i] = m.transform_point3(cube_corner(dx, dy, dz));
                i += 1;
            }
        }
    }
    out
}

/// Boundary-ordered corners `[a, b, c, d]` (`a -> b -> c -> d -> a` traces the quad's edge) to
/// the `[top-left, bottom-left, top-right, bottom-right]`-style strip order
/// [`DrawList::quad`](crate::widget::DrawList::quad) needs (see its own doc: two triangles,
/// `[0,1,2,1,3,2]`). For any convex planar quad this is `[a, d, b, c]` — the same reindexing
/// [`DrawList::quad_rect`] uses internally for an axis-aligned rectangle, generalised to an
/// arbitrary boundary order.
#[cfg(test)]
fn boundary_to_strip<T: Copy>(corners: [T; 4]) -> [T; 4] {
    [corners[0], corners[3], corners[1], corners[2]]
}

#[cfg(test)]
fn project_to_gui(p: Vec3, scale: f32, cx: f32, cy: f32) -> [f32; 2] {
    // World space is right-handed (Y up); GUI pixels are Y-down, hence the negation.
    [cx + p.x * scale, cy - p.y * scale]
}

#[cfg(test)]
fn uv_bounds(model: &BakedModel, face: Facing) -> Option<(f32, f32, f32, f32)> {
    let quad = model.face_quads[face.index() as usize]
        .first()
        .or_else(|| model.general_quads.first())?;
    let (mut min_u, mut max_u, mut min_v, mut max_v) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for v in &quad.verts {
        min_u = min_u.min(v.uv[0]);
        max_u = max_u.max(v.uv[0]);
        min_v = min_v.min(v.uv[1]);
        max_v = max_v.max(v.uv[1]);
    }
    Some((min_u, min_v, max_u, max_v))
}

/// One projected, shaded, textured face of the isometric cube, ready to hand to
/// [`DrawList::quad`](crate::widget::DrawList::quad).
#[cfg(test)]
#[allow(dead_code)]
struct IsoFace {
    pos: [[f32; 2]; 4],
    uv: [[f32; 2]; 4],
    shade: f32,
}

/// Builds the three visible faces of the isometric cube icon (top, "west", "south" — the three
/// faces vanilla's own GUI rotation brings into view), already positioned to fit inside a
/// 16×16 box at `(x, y)`. Exposed separately from [`draw_iso_block_icon`] so geometry can be
/// asserted on without a `DrawList`.
#[cfg(test)]
fn iso_faces(model: &BakedModel, x: i32, y: i32) -> Vec<IsoFace> {
    let corners = rotated_cube_corners();
    // Corner index = (dx<<2)|(dy<<1)|dz per `rotated_cube_corners`'s nested-loop order.
    let c = |dx: usize, dy: usize, dz: usize| corners[dx * 4 + dy * 2 + dz];

    // Normalise scale so the whole rotated cube fits the 16px box with a little margin.
    let extent = corners
        .iter()
        .map(|p| p.x.abs().max(p.y.abs()))
        .fold(0.0f32, f32::max);
    let scale = if extent > 0.0 { 7.5 / extent } else { 8.0 };
    let cx = x as f32 + 8.0;
    let cy = y as f32 + 8.0;
    let proj = |p: Vec3| project_to_gui(p, scale, cx, cy);

    let mut faces = Vec::with_capacity(3);

    // Top face (+Y): visible because the cube tips forward. Boundary order chosen so the
    // texture's own U axis runs along world X and V along world Z (matches a block's north-up
    // texture orientation closely enough for a small GUI icon).
    if let Some((u0, v0, u1, v1)) = uv_bounds(model, Facing::Up) {
        let boundary = [
            proj(c(0, 1, 0)),
            proj(c(1, 1, 0)),
            proj(c(1, 1, 1)),
            proj(c(0, 1, 1)),
        ];
        faces.push(IsoFace {
            pos: boundary_to_strip(boundary),
            uv: boundary_to_strip([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]),
            shade: 1.0,
        });
    }

    // East face (+X): the screen-left side under this test helper's 30/225 projection.  The
    // production GUI cull test below uses the same front-facing set after the full Java matrix.
    if let Some((u0, v0, u1, v1)) = uv_bounds(model, Facing::East) {
        let boundary = [
            proj(c(1, 1, 0)),
            proj(c(1, 1, 1)),
            proj(c(1, 0, 1)),
            proj(c(1, 0, 0)),
        ];
        faces.push(IsoFace {
            pos: boundary_to_strip(boundary),
            uv: boundary_to_strip([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]),
            shade: 0.637,
        });
    }

    // North face (-Z): the screen-right side under this helper, with Java's darker shade.
    if let Some((u0, v0, u1, v1)) = uv_bounds(model, Facing::North) {
        let boundary = [
            proj(c(0, 1, 0)),
            proj(c(1, 1, 0)),
            proj(c(1, 0, 0)),
            proj(c(0, 0, 0)),
        ];
        faces.push(IsoFace {
            pos: boundary_to_strip(boundary),
            uv: boundary_to_strip([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]),
            shade: 0.435,
        });
    }

    faces
}

/// Composes the model display transform and vanilla's GUI camera setup. The resulting matrix
/// maps each block-local baked vertex into GUI pixels relative to the slot centre; production
/// rendering then applies it to every face and general quad.
fn baked_gui_matrix(model: &BakedModel, gui_3d: bool) -> Mat4 {
    let gui = model.transforms.gui;
    let display = Mat4::from_translation(Vec3::from(gui.translation))
        * Mat4::from_rotation_y(gui.rotation[1].to_radians())
        * Mat4::from_rotation_x(gui.rotation[0].to_radians())
        * Mat4::from_rotation_z(gui.rotation[2].to_radians())
        * Mat4::from_scale(Vec3::from(gui.scale));
    let setup = if gui_3d {
        Mat4::from_scale(Vec3::splat(20.0))
            * Mat4::from_rotation_x(210f32.to_radians())
            * Mat4::from_rotation_y((-135f32).to_radians())
    } else {
        Mat4::from_scale(Vec3::splat(32.0)) * Mat4::from_rotation_x(180f32.to_radians())
    };
    setup * display * Mat4::from_scale(Vec3::splat(0.5)) * Mat4::from_translation(Vec3::splat(-0.5))
}

/// The GUI orthographic depth mapping used by `EntityRenderer.setupOverlayRendering`.
///
/// Vanilla translates `RenderItem` to `z = 100 + zLevel` under a modelview translation of
/// `-2000`, then uses `glOrtho(..., 1000, 3000)`. WGPU's depth range is `0..1`, so OpenGL's
/// `[-1, 1]` value becomes `0.5 - z / 2000`. `renderItemAndEffectIntoGUI` adds another `50`
/// to `RenderItem.zLevel` for the duration of every item call. The two container call sites
/// therefore set the pre-call levels below: ordinary slots use `100`, while the carried cursor
/// and touchscreen returning stack use `200`.
/// Java's `RenderItem.zLevel` immediately before an ordinary `GuiContainer.drawSlot` item call.
pub const GUI_SLOT_ITEM_RENDER_Z_LEVEL: f32 = 100.0;
/// Java's `RenderItem.zLevel` immediately before a carried cursor or returning-stack item call.
pub const GUI_CURSOR_ITEM_RENDER_Z_LEVEL: f32 = 200.0;
const GUI_RENDER_ITEM_EFFECT_Z_LEVEL: f32 = 50.0;
// Keep the legacy icon-only helpers byte/depth-compatible with their pre-layer API. Container
// callers use the explicit slot/cursor constants above.
const GUI_DEFAULT_ITEM_RENDER_Z_LEVEL: f32 = 50.0;
const GUI_RENDER_ITEM_TRANSLATE_Z: f32 = 100.0;
const GUI_ORTHO_DEPTH_CENTER: f32 = 0.5;
const GUI_ORTHO_DEPTH_SCALE: f32 = 1.0 / 2000.0;

#[must_use]
fn gui_depth_from_view_z(view_z: f32) -> f32 {
    (GUI_ORTHO_DEPTH_CENTER - view_z * GUI_ORTHO_DEPTH_SCALE).clamp(0.0, 1.0)
}

#[must_use]
// Keep the original default-depth helper as a compatibility/test seam; live callers pass their
// Java RenderItem.zLevel explicitly through `gui_item_depth_at_z_level`.
#[allow(dead_code)]
fn gui_item_depth(model_z: f32) -> f32 {
    gui_item_depth_at_z_level(model_z, GUI_DEFAULT_ITEM_RENDER_Z_LEVEL)
}

#[must_use]
fn gui_item_depth_at_z_level(model_z: f32, item_render_z_level: f32) -> f32 {
    gui_depth_from_view_z(
        GUI_RENDER_ITEM_TRANSLATE_Z
            + item_render_z_level
            + GUI_RENDER_ITEM_EFFECT_Z_LEVEL
            + model_z,
    )
}

/// Java's fixed-function `RenderHelper.enableGUIStandardItemLighting` result for one baked quad.
///
/// `RenderItem.putQuadNormal` supplies `quad.getFace()` rather than the old baked directional
/// shade multiplier. OpenGL then transforms that normal by the inverse transpose of the complete
/// GUI modelview. The projected geometry matrix intentionally omits `setupGuiTransform`'s Z
/// reflection because Z is discarded by the UI, but the reflection remains observable to the two
/// directional lights and therefore belongs in this lighting-only matrix.
fn gui_standard_item_light(model: &BakedModel, quad: &BakedQuad) -> f32 {
    let local_normal = quad
        .face
        .or(quad.shade_face)
        .map(facing_normal)
        .unwrap_or_else(|| geometric_quad_normal(quad));
    let modelview = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0)) * baked_gui_matrix(model, true);
    let transformed = modelview
        .inverse()
        .transpose()
        .transform_vector3(local_normal);
    gui_standard_item_light_for_eye_normal(normalized_or(transformed, local_normal))
}

/// Whether a baked quad survives the GUI model's fixed-function back-face test.
///
/// `GuiContainer` enters the inventory draw with depth testing disabled for the background and
/// foreground widgets, but each non-placeholder `drawSlot` call re-enables depth immediately
/// before `RenderItem`. Vanilla still has the renderer's ordinary `GL_CULL_FACE/GL_BACK` state,
/// so culling belongs at this model replay seam too. The full GPU path carries the transformed Z
/// beside each projected vertex; the CPU normal test here remains a cheap early filter for the
/// GUI-3D face buckets. The complete Java modelview includes `setupGuiTransform`'s
/// `scale(1,1,-1)` reflection, which is why the normal transform here is the inverse transpose of
/// that reflected matrix rather than merely the projected 2-D matrix.
fn gui_3d_quad_is_front_facing(matrix: Mat4, quad: &BakedQuad) -> bool {
    let local_normal = quad
        .face
        .or(quad.shade_face)
        .map(facing_normal)
        .unwrap_or_else(|| geometric_quad_normal(quad));
    let modelview = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0)) * matrix;
    let transformed = modelview
        .inverse()
        .transpose()
        .transform_vector3(local_normal);
    let length_squared = transformed.length_squared();
    length_squared.is_finite() && length_squared > f32::EPSILON && transformed.z > 0.0
}

fn gui_standard_item_light_for_eye_normal(normal: Vec3) -> f32 {
    let normal = normalized_or(normal, Vec3::Z);
    // RenderHelper's light positions are captured after rotateY(-30), rotateX(165). Directional
    // positions have w=0, so the surrounding GUI translation cannot affect them.
    let light_frame = Mat4::from_rotation_y((-30.0f32).to_radians())
        * Mat4::from_rotation_x(165.0f32.to_radians());
    // These are Java's float literals; the extra digits preserve the captured light directions.
    #[allow(clippy::excessive_precision)]
    let light0 =
        light_frame.transform_vector3(Vec3::new(0.200_000_003, 1.0, -0.699_999_988).normalize());
    #[allow(clippy::excessive_precision)]
    let light1 =
        light_frame.transform_vector3(Vec3::new(-0.200_000_003, 1.0, 0.699_999_988).normalize());
    (0.4 + 0.6 * normal.dot(light0).max(0.0) + 0.6 * normal.dot(light1).max(0.0)).clamp(0.0, 1.0)
}

fn facing_normal(face: Facing) -> Vec3 {
    let (x, y, z) = face.offset();
    Vec3::new(x as f32, y as f32, z as f32)
}

fn geometric_quad_normal(quad: &BakedQuad) -> Vec3 {
    let a = Vec3::from(quad.verts[0].pos);
    let b = Vec3::from(quad.verts[1].pos);
    let c = Vec3::from(quad.verts[2].pos);
    normalized_or((b - a).cross(c - a), Vec3::Z)
}

fn normalized_or(value: Vec3, fallback: Vec3) -> Vec3 {
    let length_squared = value.length_squared();
    if length_squared.is_finite() && length_squared > f32::EPSILON {
        value / length_squared.sqrt()
    } else {
        fallback.normalize_or_zero()
    }
}

fn apply_gui_standard_item_light(color: [f32; 4], light: f32) -> [f32; 4] {
    [
        color[0] * light,
        color[1] * light,
        color[2] * light,
        color[3],
    ]
}

/// The matrix on the GL stack immediately before `RenderItem.renderItem`'s builtin branch.
///
/// java: `RenderItem.setupGuiTransform` followed by
/// `ItemCameraTransforms.applyTransform(GUI)`.  Unlike the ordinary baked-quad helper above,
/// this keeps the setup's Z reflection because the tile-item emitter applies additional 3-D
/// rotations (`rotateY(180)` and the skull's floor transform) before the UI projection discards
/// Z.  Omitting that reflection is visually harmless for a generated front card but mirrors a
/// chest/banner as soon as the builtin branch mixes all three axes.
#[must_use]
fn builtin_gui_parent(model: &BakedModel) -> Mat4 {
    let gui = model.transforms.gui;
    let display = Mat4::from_translation(Vec3::from(gui.translation))
        * Mat4::from_rotation_y(gui.rotation[1].to_radians())
        * Mat4::from_rotation_x(gui.rotation[0].to_radians())
        * Mat4::from_rotation_z(gui.rotation[2].to_radians())
        * Mat4::from_scale(Vec3::from(gui.scale));
    // setupGuiTransform: scale(1,1,-1), scale(.5), scale(40), rotate(210,X), rotate(-135,Y).
    Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0))
        * Mat4::from_scale(Vec3::splat(0.5))
        * Mat4::from_scale(Vec3::splat(40.0))
        * Mat4::from_rotation_x(210f32.to_radians())
        * Mat4::from_rotation_y((-135f32).to_radians())
        * display
}

/// Emit a `builtin/entity` item into the 2-D UI list while retaining the tile renderer's real
/// model geometry and texture key.
///
/// `TileModels::emit_builtin_item` already applies `RenderItem.renderItem`'s common half-scale,
/// 180-degree turn and recenter (plus the skull-only floor transform).  This helper supplies the
/// GUI setup/display parent, projects the resulting camera-independent vertices to screen pixels,
/// and records the descriptor for [`UiRenderer::ensure_list_textures`](crate::widget::UiRenderer)
/// to upload before the frame is submitted. The UI phase attaches the freshly-cleared shared
/// depth buffer; this helper carries each tile vertex's GUI depth into the item state window while
/// preserving the emitter's triangle order. Skull renderers retain Java's local cull-disabled
/// exception; chests/banners use the inherited `GL_BACK` pipeline.
// This pre-z-level wrapper is retained for the legacy builtin-item test seam; live callers use
// the explicit z-level variant so tile items share the slot/cursor depth contract.
#[allow(dead_code)]
fn draw_builtin_tile_item(
    draw: &mut DrawList,
    x: i32,
    y: i32,
    stack: &ItemStack,
    item: &BuiltinTileItem,
    base: [f32; 4],
) -> bool {
    draw_builtin_tile_item_at_z_level(
        draw,
        x,
        y,
        stack,
        item,
        base,
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
    )
}

fn draw_builtin_tile_item_at_z_level(
    draw: &mut DrawList,
    x: i32,
    y: i32,
    stack: &ItemStack,
    item: &BuiltinTileItem,
    base: [f32; 4],
    item_render_z_level: f32,
) -> bool {
    let props = stack_ext::props(stack);
    let model = mc_assets::item_model_override_for(props.id, stack.damage).model;
    let parent = builtin_gui_parent(model);
    let mut entity_vertices = Vec::new();
    let Some(texture_path) = draw.with_builtin_tile_models(|models| {
        models.emit_builtin_item(item, parent, &mut entity_vertices)
    }) else {
        return false;
    };

    draw.register_builtin_tile_item(item.clone());
    let texture = UiTexture::gui(&texture_path);
    let cull = if matches!(
        item,
        BuiltinTileItem::Skull { .. } | BuiltinTileItem::PlayerSkull { .. }
    ) {
        UiCull::None
    } else {
        UiCull::Back
    };
    // java: the builtin branch still runs inside `renderItemIntoGUI`'s alpha-test window.
    draw.with_alpha_test(|draw| {
        draw.with_item_depth(
            gui_item_depth_at_z_level(0.0, item_render_z_level),
            |draw| {
                draw.with_render_state(UiDepth::Item, cull, |draw| {
                    for triangle in entity_vertices.chunks_exact(3) {
                        let points = core::array::from_fn(|index| {
                            let vertex = &triangle[index];
                            let point = Vec3::from(vertex.pos);
                            // setupGuiTransform's camera rotations already convert model-up into
                            // the GUI screen axis (the same convention as
                            // draw_projected_baked_quad_with_blend), so retain the transformed Y
                            // sign here instead of mirroring the tile model vertically.
                            [x as f32 + 8.0 + point.x, y as f32 + 8.0 + point.y]
                        });
                        let depth = core::array::from_fn(|index| {
                            gui_item_depth_at_z_level(
                                Vec3::from(triangle[index].pos).z,
                                item_render_z_level,
                            )
                        });
                        let uv = core::array::from_fn(|index| triangle[index].uv);
                        let color = core::array::from_fn(|index| {
                            let vertex = &triangle[index];
                            let light =
                                gui_standard_item_light_for_eye_normal(Vec3::from(vertex.normal));
                            [
                                base[0] * vertex.color[0] * light,
                                base[1] * vertex.color[1] * light,
                                base[2] * vertex.color[2] * light,
                                base[3] * vertex.color[3],
                            ]
                        });
                        draw.triangle_with_blend_depth(
                            &texture,
                            UiBlend::Alpha,
                            depth,
                            points,
                            uv,
                            color,
                        );
                    }
                });
            },
        );
    });
    true
}

#[allow(dead_code, clippy::too_many_arguments)]
fn draw_projected_baked_quad_with_blend(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    matrix: Mat4,
    quad: &BakedQuad,
    blend: UiBlend,
    map_uv: impl FnMut([f32; 2]) -> [f32; 2],
    color: [f32; 4],
) {
    draw_projected_baked_quad_with_blend_at_z_level(
        draw,
        texture,
        x,
        y,
        matrix,
        quad,
        blend,
        map_uv,
        color,
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
    );
}

/// Which quads of a flat item model, in `RenderItem.renderModel` order (face
/// buckets, then general quads), the GUI camera sees edge-on (see
/// [`DrawList::enable_item_sliver_culling`]). Computed once per model with the exact projection
/// [`draw_projected_baked_quad_with_blend_at_z_level`] uses, at slot origin `(0, 0)`: slot
/// positions are whole GUI pixels and change an extent by at most an `f32` rounding step, while a
/// sliver is ~1e-7 wide and the thinnest real face 1/16.
///
/// Keyed by the model's address. Only `&'static` models are accepted: installed models are never
/// freed (`mc_assets::reinstall` leaks the previous set on purpose), so an address can never come
/// back with other geometry.
fn flat_item_sliver_mask(model: &'static BakedModel) -> std::rc::Rc<[bool]> {
    thread_local! {
        static MASKS: std::cell::RefCell<mc_core::FastMap<usize, std::rc::Rc<[bool]>>> =
            std::cell::RefCell::new(mc_core::FastMap::default());
    }
    let key = std::ptr::from_ref(model) as usize;
    MASKS.with(|masks| {
        if let Some(mask) = masks.borrow().get(&key) {
            return mask.clone();
        }
        let matrix = baked_gui_matrix(model, false);
        let mask: std::rc::Rc<[bool]> = model
            .face_quads
            .iter()
            .flatten()
            .chain(&model.general_quads)
            .map(|quad| {
                let projected = quad
                    .verts
                    .map(|vertex| matrix.transform_point3(Vec3::from(vertex.pos)));
                is_gui_item_sliver(&projected.map(|point| [8.0 + point.x, 8.0 + point.y]))
            })
            .collect();
        masks.borrow_mut().insert(key, mask.clone());
        mask
    })
}

/// Whether a projected item quad (slot origin `(0, 0)`) is seen edge-on **on a
/// whole GUI pixel line**: at most [`GUI_ITEM_SLIVER_EXTENT`] GUI pixels across on one screen
/// axis, and on that axis within the same distance of a whole GUI pixel (see
/// [`DrawList::enable_item_sliver_culling`]). An edge-on quad off the pixel grid (a pack's
/// fractional GUI display transform) is kept. NaN corners compare false: never a sliver.
fn is_gui_item_sliver(points: &[[f32; 2]; 4]) -> bool {
    let [mut min_x, mut min_y] = points[0];
    let (mut max_x, mut max_y) = (min_x, min_y);
    for &[px, py] in &points[1..] {
        min_x = min_x.min(px);
        max_x = max_x.max(px);
        min_y = min_y.min(py);
        max_y = max_y.max(py);
    }
    let on_pixel_line = |lo: f32, hi: f32| {
        hi - lo <= GUI_ITEM_SLIVER_EXTENT
            && (lo - lo.round()).abs() <= GUI_ITEM_SLIVER_EXTENT
            && (hi - hi.round()).abs() <= GUI_ITEM_SLIVER_EXTENT
    };
    on_pixel_line(min_x, max_x) || on_pixel_line(min_y, max_y)
}

#[allow(clippy::too_many_arguments)]
fn draw_projected_baked_quad_with_blend_at_z_level(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    matrix: Mat4,
    quad: &BakedQuad,
    blend: UiBlend,
    mut map_uv: impl FnMut([f32; 2]) -> [f32; 2],
    color: [f32; 4],
    item_render_z_level: f32,
) {
    // Each corner is transformed once and feeds both its screen position and its depth (the
    // same `transform_point3` on the same input used to run twice per corner, CONT-3).
    let projected = quad
        .verts
        .map(|vertex| matrix.transform_point3(Vec3::from(vertex.pos)));
    let points = projected.map(|point| [x as f32 + 8.0 + point.x, y as f32 + 8.0 + point.y]);
    let depth = projected.map(|point| {
        // `baked_gui_matrix` intentionally omits setupGuiTransform's leftmost Z reflection for
        // the already-correct screen X/Y and lighting calculations. Reapply that reflection for
        // the actual eye-space depth used by Java's orthographic projection.
        gui_item_depth_at_z_level(-point.z, item_render_z_level)
    });
    let uv = quad.verts.map(|vertex| map_uv(vertex.uv));
    // BakedQuad's four vertices are boundary-wound.  Triangles preserve that winding without
    // pretending every projected shape can use DrawList::quad's screen-strip ordering.
    for indices in [[0, 1, 2], [0, 2, 3]] {
        draw.triangle_with_blend_depth(
            texture,
            blend,
            indices.map(|index| depth[index]),
            indices.map(|index| points[index]),
            indices.map(|index| uv[index]),
            [color; 3],
        );
    }
}

/// Java `RenderItem.renderModel`: every face bucket in enum order, then general quads.
#[allow(clippy::too_many_arguments)]
fn draw_baked_gui_model(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    model: &BakedModel,
    gui_3d: bool,
    color: impl FnMut(&BakedQuad) -> [f32; 4],
) {
    draw_baked_gui_model_at_z_level(
        draw,
        texture,
        x,
        y,
        model,
        gui_3d,
        color,
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_baked_gui_model_at_z_level(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    model: &BakedModel,
    gui_3d: bool,
    color: impl FnMut(&BakedQuad) -> [f32; 4],
    item_render_z_level: f32,
    slivers: Option<&[bool]>,
) {
    draw_baked_gui_model_with_blend_at_z_level(
        draw,
        texture,
        x,
        y,
        model,
        gui_3d,
        UiBlend::Alpha,
        |uv| uv,
        color,
        item_render_z_level,
        slivers,
    );
}

#[allow(dead_code, clippy::too_many_arguments)]
fn draw_baked_gui_model_with_blend(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    model: &BakedModel,
    gui_3d: bool,
    blend: UiBlend,
    map_uv: impl FnMut([f32; 2]) -> [f32; 2],
    color: impl FnMut(&BakedQuad) -> [f32; 4],
) {
    draw_baked_gui_model_with_blend_at_z_level(
        draw,
        texture,
        x,
        y,
        model,
        gui_3d,
        blend,
        map_uv,
        color,
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_baked_gui_model_with_blend_at_z_level(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    model: &BakedModel,
    gui_3d: bool,
    blend: UiBlend,
    mut map_uv: impl FnMut([f32; 2]) -> [f32; 2],
    mut color: impl FnMut(&BakedQuad) -> [f32; 4],
    item_render_z_level: f32,
    slivers: Option<&[bool]>,
) {
    // Attribution only (a no-op unless `MC_PROFILE` is set).
    mc_core::profile_scope!("ui.item_model");
    let matrix = baked_gui_matrix(model, gui_3d);
    let depth = if blend == UiBlend::Glint {
        UiDepth::Equal
    } else {
        UiDepth::Item
    };
    // java: `RenderItem.renderItemIntoGUI` runs `enableAlpha(); alphaFunc(GL_GREATER, 0.1F)`
    // around `renderItem`, i.e. around the model *and* its `renderEffect` foil. A transparent
    // texel therefore writes no depth, so the GL_EQUAL foil lands only on the item's own pixels
    // and a beacon's core stays visible through the clear texels of its glass shell.
    draw.with_alpha_test(|draw| {
        draw.with_render_state(depth, UiCull::Back, |draw| {
            for (index, quad) in model
                .face_quads
                .iter()
                .flatten()
                .chain(&model.general_quads)
                .enumerate()
            {
                // An edge-on side quad of a flat item covers no pixel (see
                // `DrawList::enable_item_sliver_culling`, mask from `flat_item_sliver_mask`).
                // `map_uv` and `color` are pure, so skipping it changes nothing for later quads.
                if slivers.is_some_and(|mask| mask.get(index).copied().unwrap_or(false)) {
                    continue;
                }
                // GUI-3D models can cheaply reject their six back-facing buckets before
                // flattening; generated/flat cards are retained here so the GPU's inherited
                // GL_BACK pipeline makes the same per-triangle decision as Java, including
                // opposite front/back cards.
                if gui_3d && !gui_3d_quad_is_front_facing(matrix, quad) {
                    continue;
                }
                draw_projected_baked_quad_with_blend_at_z_level(
                    draw,
                    texture,
                    x,
                    y,
                    matrix,
                    quad,
                    blend,
                    &mut map_uv,
                    color(quad),
                    item_render_z_level,
                );
            }
        });
    });
}

/// Draw a 3D GUI item by replaying its baked quads through vanilla's GUI camera transform.
pub fn draw_iso_block_icon(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    model: &BakedModel,
) {
    draw_baked_gui_model(draw, texture, x, y, model, true, |quad| {
        apply_gui_standard_item_light([1.0; 4], gui_standard_item_light(model, quad))
    });
}

/// Draw a GUI-3D item model while applying the same per-tint-index item colour policy as the
/// flat icon path.  Vanilla's `RenderItem.renderQuads` does not special-case block-shaped models:
/// it walks their face/general quads and calls `getColorFromItemStack` for every tagged quad.
#[allow(dead_code, clippy::too_many_arguments)]
fn draw_gui_item_model_tinted(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    stack: &ItemStack,
    model: &BakedModel,
    base: [f32; 4],
) {
    draw_baked_gui_model(draw, texture, x, y, model, true, |quad| {
        let tinted = modulate_color(base, item_quad_color(stack, quad.tint_index));
        apply_gui_standard_item_light(tinted, gui_standard_item_light(model, quad))
    });
}

/// [`draw_item_in_slot`]'s **icon half** with a colour multiplier, and no overlays.
///
/// java: `RenderItem.renderItemAndEffectIntoGUI` drawn under a
/// `GlStateManager.color(f, f, f, 1.0F)` tint. `GuiAchievements.drawAchievementScreen` is the
/// call site that needs it: it dims every entry by how far out of reach it is — `0.75` unlocked,
/// `1.0` reachable, then `0.3` / `0.2` / `0.1` by how many locked prerequisites deep — and it
/// calls `renderItemAndEffectIntoGUI` *without* a following `renderItemOverlays`, so no stack
/// count and no durability bar are drawn there.
///
/// This is an **additive** helper; the ordinary icon paths are unchanged.
/// `tinting_with_opaque_white_is_byte_identical_to_the_untinted_paths` pins that this function
/// with `0xFFFF_FFFF` emits vertex-for-vertex identical geometry to both ordinary icon paths.
/// Item-specific per-quad tint multiplies this caller-supplied colour, as OpenGL's current colour
/// multiplies `RenderItem.renderQuads`' vertex colour in Java.
pub fn draw_item_icon_tinted(
    draw: &mut DrawList,
    atlas_texture: &UiTexture,
    x: i32,
    y: i32,
    stack: &ItemStack,
    argb: u32,
) {
    draw_item_icon_tinted_at_z_level(
        draw,
        atlas_texture,
        x,
        y,
        stack,
        argb,
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
    );
}

/// [`draw_item_icon_tinted`]'s `parts` (see [`draw_item_icon_parts_at_z_level`]).
pub fn draw_item_icon_parts(
    draw: &mut DrawList,
    atlas_texture: &UiTexture,
    x: i32,
    y: i32,
    stack: &ItemStack,
    argb: u32,
    parts: ItemIconParts,
) {
    draw_item_icon_parts_at_z_level(
        draw,
        atlas_texture,
        x,
        y,
        stack,
        argb,
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
        parts,
    );
}

/// Draw only an item icon while retaining Java's per-call `RenderItem.zLevel`.
///
/// `item_render_z_level` is the value held by `RenderItem` immediately before
/// `renderItemAndEffectIntoGUI` adds its temporary `50`. Container slots pass
/// [`GUI_SLOT_ITEM_RENDER_Z_LEVEL`]; the carried cursor and touchscreen returning stack pass
/// [`GUI_CURSOR_ITEM_RENDER_Z_LEVEL`].
pub fn draw_item_icon_tinted_at_z_level(
    draw: &mut DrawList,
    atlas_texture: &UiTexture,
    x: i32,
    y: i32,
    stack: &ItemStack,
    argb: u32,
    item_render_z_level: f32,
) {
    draw_item_icon_parts_at_z_level(
        draw,
        atlas_texture,
        x,
        y,
        stack,
        argb,
        item_render_z_level,
        ItemIconParts::Both,
    );
}

/// Which part of an item icon [`draw_item_icon_parts_at_z_level`] draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemIconParts {
    /// The model, then its enchantment glint: [`draw_item_icon_tinted_at_z_level`].
    Both,
    /// Only the model (or a builtin item's complete icon).
    Model,
    /// Only the enchantment glint; nothing for builtin items, which have none.
    Glint,
}

/// [`draw_item_icon_tinted_at_z_level`], or one of its two halves. Drawing `Model` for a row of
/// icons and then `Glint` for the same row puts every pixel through the same operations, in the
/// same order, as `Both` per icon, as long as the icons do not overlap (the glint's `GL_EQUAL`
/// depth test finds its own icon's depth either way): the hotbar's pass order
/// uses it to draw all icons as one batch and all glints as another.
#[allow(clippy::too_many_arguments)]
pub fn draw_item_icon_parts_at_z_level(
    draw: &mut DrawList,
    atlas_texture: &UiTexture,
    x: i32,
    y: i32,
    stack: &ItemStack,
    argb: u32,
    item_render_z_level: f32,
    parts: ItemIconParts,
) {
    let Some(props) = stack_ext::try_props(stack) else {
        return;
    };
    let base = crate::widget::draw::unpack_argb(argb);
    draw.with_item_depth(
        gui_item_depth_at_z_level(0.0, item_render_z_level),
        |draw| {
            let model = mc_assets::item_model_override_for(props.id, stack.damage).model;
            if let Some(item) = builtin_tile_item_from_stack_for_model(stack, model) {
                // `RenderItem.renderItem`'s builtin branch does not run `renderEffect`; the tile entity
                // is the complete icon, so do not send an empty-model foil pass afterwards.
                if parts != ItemIconParts::Glint {
                    let _ = draw_builtin_tile_item_at_z_level(
                        draw,
                        x,
                        y,
                        stack,
                        &item,
                        base,
                        item_render_z_level,
                    );
                }
            } else {
                let gui_3d = model_uses_gui_3d(model);
                // The edge-on side quads of a flat model, when the list culls them
                // and the icon sits on whole pixels (no scaling or fractional transform).
                let slivers =
                    (!gui_3d && draw.item_sliver_culling() && draw.whole_pixel_placement())
                        .then(|| flat_item_sliver_mask(model));
                if parts != ItemIconParts::Glint {
                    if gui_3d {
                        draw_gui_item_model_tinted_at_z_level(
                            draw,
                            atlas_texture,
                            x,
                            y,
                            stack,
                            model,
                            base,
                            item_render_z_level,
                        );
                    } else {
                        draw_flat_icon_colored_at_z_level(
                            draw,
                            atlas_texture,
                            x,
                            y,
                            model,
                            |tint_index| {
                                let item = item_quad_color(stack, tint_index);
                                modulate_color(base, item)
                            },
                            item_render_z_level,
                            slivers.as_deref(),
                        );
                    }
                }
                if parts != ItemIconParts::Model {
                    draw_item_glint_if_needed_at_z_level(
                        draw,
                        x,
                        y,
                        stack,
                        model,
                        gui_3d,
                        item_render_z_level,
                        slivers.as_deref(),
                    );
                }
            }
        },
    );
}

// The at-z-level icon route is kept beside the legacy helper so generic HUD/stat callers can
// retain their old default while container callers explicitly select slot/cursor depth.
#[allow(clippy::too_many_arguments)]
fn draw_flat_icon_colored_at_z_level(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    model: &BakedModel,
    mut color: impl FnMut(Option<u8>) -> [f32; 4],
    item_render_z_level: f32,
    slivers: Option<&[bool]>,
) {
    draw_baked_gui_model_at_z_level(
        draw,
        texture,
        x,
        y,
        model,
        false,
        |quad| color(quad.tint_index),
        item_render_z_level,
        slivers,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_gui_item_model_tinted_at_z_level(
    draw: &mut DrawList,
    texture: &UiTexture,
    x: i32,
    y: i32,
    stack: &ItemStack,
    model: &BakedModel,
    base: [f32; 4],
    item_render_z_level: f32,
) {
    draw_baked_gui_model_at_z_level(
        draw,
        texture,
        x,
        y,
        model,
        true,
        |quad| {
            let tinted = modulate_color(base, item_quad_color(stack, quad.tint_index));
            apply_gui_standard_item_light(tinted, gui_standard_item_light(model, quad))
        },
        item_render_z_level,
        None,
    );
}

/// Draw only the icon half of a slot, resolving a filled map through the live `map_<id>` record
/// when the host supplied one. Returns `true` when the live map path was used; a `false` result
/// leaves icon emission to the ordinary baked-model caller.
///
/// Vanilla's ordinary GUI item renderer falls back to the baked `map_filled` sprite until
/// `ItemMap.getMapData` finds a world record. Keeping that fallback in the caller is important
/// for the first frame after an inventory snapshot and for maps whose S34 record has not arrived
/// yet. A matching record instead reuses [`crate::map::draw_map`] at `1:8`, which gives the 16 by
/// 16 slot the same nearest-filtered pixels and ordered decoration pass as the other map callers.
pub fn draw_item_icon_with_map_data(
    draw: &mut DrawList,
    _atlas_texture: &UiTexture,
    x: i32,
    y: i32,
    stack: &ItemStack,
    map_data: Option<&MapData>,
) -> bool {
    draw_item_icon_with_map_data_at_z_level(
        draw,
        _atlas_texture,
        x,
        y,
        stack,
        map_data,
        GUI_DEFAULT_ITEM_RENDER_Z_LEVEL,
    )
}

/// Draw a filled-map icon while retaining Java's per-call `RenderItem.zLevel`.
pub fn draw_item_icon_with_map_data_at_z_level(
    draw: &mut DrawList,
    _atlas_texture: &UiTexture,
    x: i32,
    y: i32,
    stack: &ItemStack,
    map_data: Option<&MapData>,
    item_render_z_level: f32,
) -> bool {
    let Some(props) = stack_ext::try_props(stack) else {
        return false;
    };
    let is_filled_map = props.name == "minecraft:filled_map";
    let map_id = i32::from(stack.damage);
    let Some(map_data) = map_data.filter(|data| is_filled_map && data.id == map_id) else {
        return false;
    };

    draw.with_item_depth(
        gui_item_depth_at_z_level(0.0, item_render_z_level),
        |draw| {
            draw.with_render_state(UiDepth::Item, UiCull::Back, |draw| {
                draw_map(
                    draw,
                    &dynamic_texture_key(map_id),
                    &UiTexture::gui(MAP_ICONS_TEXTURE),
                    [x as f32, y as f32],
                    1.0 / 8.0,
                    map_data,
                    false,
                );
            });
        },
    );
    true
}

/// The full slot draw: icon (flat or isometric, per [`BakedModel::gui_3d`]),
/// stack count, and durability bar. `atlas_texture` is whatever [`UiTexture`] key the caller
/// uploaded `mc_assets::BakedAssets::atlas`'s pixels under.
pub fn draw_item_in_slot(
    draw: &mut DrawList,
    atlas_texture: &UiTexture,
    font: &mut FontRenderer,
    x: i32,
    y: i32,
    stack: &ItemStack,
) {
    draw_item_in_slot_at_z_level(
        draw,
        atlas_texture,
        font,
        x,
        y,
        stack,
        GUI_SLOT_ITEM_RENDER_Z_LEVEL,
    );
}

/// Draw an ordinary Java `RenderItem` inventory icon at an explicit z level.
///
/// This is the `GuiContainer` path: even for a filled map, `RenderItem` resolves the static
/// `filled_map` `ItemMeshDefinition`/`map_filled` model. Live `MapData` belongs to held-map and
/// item-frame renderers, not to an inventory slot. The map-aware helper below remains available
/// for callers that intentionally implement one of those dynamic GUI surfaces.
#[allow(clippy::too_many_arguments)]
pub fn draw_item_in_slot_at_z_level(
    draw: &mut DrawList,
    atlas_texture: &UiTexture,
    font: &mut FontRenderer,
    x: i32,
    y: i32,
    stack: &ItemStack,
    item_render_z_level: f32,
) {
    draw_item_in_slot_with_map_data_at_z_level(
        draw,
        atlas_texture,
        font,
        x,
        y,
        stack,
        None,
        item_render_z_level,
    );
}

/// Draw a complete inventory slot, using live filled-map pixels and decorations when the keyed
/// world record is available. All non-map items and unknown maps retain the ordinary item-model,
/// glint, durability and count paths byte-for-byte.
pub fn draw_item_in_slot_with_map_data(
    draw: &mut DrawList,
    atlas_texture: &UiTexture,
    font: &mut FontRenderer,
    x: i32,
    y: i32,
    stack: &ItemStack,
    map_data: Option<&MapData>,
) {
    draw_item_in_slot_with_map_data_at_z_level(
        draw,
        atlas_texture,
        font,
        x,
        y,
        stack,
        map_data,
        GUI_SLOT_ITEM_RENDER_Z_LEVEL,
    );
}

/// Draw a complete container item at an explicit Java `RenderItem.zLevel`.
///
/// The caller supplies the value held immediately before `renderItemAndEffectIntoGUI`; its
/// temporary `+50` is applied by the depth projection. This keeps ordinary slots (`100`) and
/// carried/returning stacks (`200`) distinct even when their 2-D positions overlap.
#[allow(clippy::too_many_arguments)]
pub fn draw_item_in_slot_with_map_data_at_z_level(
    draw: &mut DrawList,
    atlas_texture: &UiTexture,
    font: &mut FontRenderer,
    x: i32,
    y: i32,
    stack: &ItemStack,
    map_data: Option<&MapData>,
    item_render_z_level: f32,
) {
    if stack_ext::try_props(stack).is_none() {
        return;
    }
    if draw_item_icon_with_map_data_at_z_level(
        draw,
        atlas_texture,
        x,
        y,
        stack,
        map_data,
        item_render_z_level,
    ) {
        // Filled maps are not damageable, but retaining this call keeps this helper's overlay
        // contract aligned with the ordinary path should a custom item definition ever mark one
        // as damageable.
        draw_durability_bar(draw, x, y, stack);
        draw_count_text(draw, font, x, y, stack);
        return;
    }
    let props = stack_ext::props(stack);
    draw.with_item_depth(
        gui_item_depth_at_z_level(0.0, item_render_z_level),
        |draw| {
            let model = mc_assets::item_model_override_for(props.id, stack.damage).model;
            if let Some(item) = builtin_tile_item_from_stack_for_model(stack, model) {
                // The builtin renderer owns its complete icon (including banners' layered texture), and
                // vanilla does not run the baked-model glint branch for it.
                let _ = draw_builtin_tile_item_at_z_level(
                    draw,
                    x,
                    y,
                    stack,
                    &item,
                    [1.0; 4],
                    item_render_z_level,
                );
            } else {
                let gui_3d = model_uses_gui_3d(model);
                // The edge-on side quads of a flat model, when the list culls them
                // and the icon sits on whole pixels (no scaling or fractional transform).
                let slivers =
                    (!gui_3d && draw.item_sliver_culling() && draw.whole_pixel_placement())
                        .then(|| flat_item_sliver_mask(model));
                if gui_3d {
                    draw_gui_item_model_tinted_at_z_level(
                        draw,
                        atlas_texture,
                        x,
                        y,
                        stack,
                        model,
                        [1.0; 4],
                        item_render_z_level,
                    );
                } else {
                    draw_flat_icon_colored_at_z_level(
                        draw,
                        atlas_texture,
                        x,
                        y,
                        model,
                        |tint_index| item_quad_color(stack, tint_index),
                        item_render_z_level,
                        slivers.as_deref(),
                    );
                }
                draw_item_glint_if_needed_at_z_level(
                    draw,
                    x,
                    y,
                    stack,
                    model,
                    gui_3d,
                    item_render_z_level,
                    slivers.as_deref(),
                );
            }
        },
    );
    draw_durability_bar(draw, x, y, stack);
    draw_count_text(draw, font, x, y, stack);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_text_matches_the_former_format_for_every_edge_value() {
        let mut buf = [0u8; 16];
        for count in [2, 9, 10, 64, 127, 0, -1, -128, i32::MAX, i32::MIN] {
            let expected = if count < 1 {
                format!("\u{a7}c{count}")
            } else {
                count.to_string()
            };
            assert_eq!(count_text(&mut buf, count), expected);
        }
    }
    use crate::widget::UiVertex;
    use mc_nbt::{Nbt, NbtCompound};

    fn stack(name: &str, count: i32) -> ItemStack {
        let id = mc_data::item_by_name(name).unwrap();
        ItemStack {
            item_id: id.0 as i16,
            count,
            damage: 0,
            nbt: None,
        }
    }

    #[test]
    fn gui_presentation_follows_the_baked_model_bit_not_item_registry_shape() {
        let flat = BakedModel::default();
        let three_dimensional = BakedModel {
            gui_3d: true,
            ..BakedModel::default()
        };
        assert!(!model_uses_gui_3d(&flat));
        assert!(model_uses_gui_3d(&three_dimensional));

        let production = include_str!("render.rs")
            .split_once("mod tests {")
            .expect("container renderer keeps inline tests")
            .0;
        assert_eq!(production.matches("model_uses_gui_3d(model)").count(), 2);
        assert!(!production.contains("props.block.is_some()"));
    }

    #[test]
    fn compass_and_clock_ui_models_keep_the_live_property_and_flat_fallback() {
        for (name, property) in [
            ("minecraft:compass", mc_assets::ItemModelProperty::Compass),
            ("minecraft:clock", mc_assets::ItemModelProperty::Clock),
        ] {
            let id = mc_data::item_by_name(name).expect("dynamic item");
            let selection = mc_assets::item_model_override_for(id, 0);
            assert_eq!(selection.property, property);
            // `builtin/generated` compass/clock icons follow the same flat GUI projection as
            // every ordinary generated icon; dynamic atlas selection must not invent a 3-D UI
            // route when a resource pack has not installed a model yet.
            assert!(!selection.gui_3d());
        }
        let ordinary = mc_data::item_by_name("minecraft:stick").expect("ordinary item");
        assert_eq!(
            mc_assets::item_model_override_for(ordinary, 0).property,
            mc_assets::ItemModelProperty::None
        );
    }

    #[test]
    fn all_builtin_entity_items_reach_real_gui_geometry_and_sheet_descriptors() {
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let cases = [
            ("chest", 0, "textures/entity/chest/normal.png", 3 * 6 * 6),
            (
                "trapped_chest",
                0,
                "textures/entity/chest/trapped.png",
                3 * 6 * 6,
            ),
            (
                "ender_chest",
                0,
                "textures/entity/chest/ender.png",
                3 * 6 * 6,
            ),
            ("skull", 0, "textures/entity/skeleton/skeleton.png", 6 * 6),
            ("banner", 14, "mc-render/banner/b14", 3 * 6 * 6),
        ];

        for (name, damage, texture_path, expected_vertices) in cases {
            let id = mc_data::item_by_name(name).expect("builtin item");
            let stack = ItemStack {
                item_id: id.0 as i16,
                count: 1,
                damage,
                nbt: None,
            };
            let item = builtin_tile_item_from_stack(&stack).expect("builtin/entity descriptor");
            let mut draw = DrawList::new();
            draw_item_icon_tinted(&mut draw, &atlas, 20, 30, &stack, 0xFFFF_FFFF);
            assert_eq!(draw.vertices().len(), expected_vertices, "{name} geometry");
            assert_eq!(draw.builtin_tile_items(), &[item]);
            assert_eq!(draw.batches().len(), 1);
            assert_eq!(draw.batches()[0].texture, UiTexture::gui(texture_path));
            assert_eq!(draw.batches()[0].range, 0..expected_vertices);
            // Negative control: builtin geometry is not accidentally submitted against the
            // stitched atlas key that the ordinary baked-item path receives.
            assert_ne!(draw.batches()[0].texture, atlas);
        }
    }

    #[test]
    fn builtin_gui_parent_matches_vanilla_setup_and_display_call_order() {
        let mut model = BakedModel::default();
        model.transforms.gui = mc_assets::model::ItemTransform {
            rotation: [17.0, -31.0, 43.0],
            translation: [0.25, -0.125, 0.0625],
            scale: [0.75, 1.25, 0.9],
        };
        let gui = model.transforms.gui;
        let point = Vec3::new(0.2, -0.3, 0.4);

        // ItemCameraTransforms.applyTransform(GUI) calls scale, Z/X/Y rotations, then
        // translation on the incoming point; setupGuiTransform then calls Y/X rotations,
        // scale(40), scale(.5), and the Z reflection.  Applying those operations to a point in
        // reverse matrix order is independent of the compact product in builtin_gui_parent.
        let mut expected = Mat4::from_scale(Vec3::from(gui.scale)).transform_point3(point);
        expected = Mat4::from_rotation_z(gui.rotation[2].to_radians()).transform_point3(expected);
        expected = Mat4::from_rotation_x(gui.rotation[0].to_radians()).transform_point3(expected);
        expected = Mat4::from_rotation_y(gui.rotation[1].to_radians()).transform_point3(expected);
        expected += Vec3::from(gui.translation);
        expected = Mat4::from_rotation_y((-135.0f32).to_radians()).transform_point3(expected);
        expected = Mat4::from_rotation_x(210.0f32.to_radians()).transform_point3(expected);
        expected *= 20.0;
        expected.z = -expected.z;

        let actual = builtin_gui_parent(&model).transform_point3(point);
        for (got, want) in actual.to_array().into_iter().zip(expected.to_array()) {
            assert!((got - want).abs() < 1e-5, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn builtin_gui_projection_keeps_the_camera_transformed_y_sign() {
        let stack = stack("chest", 1);
        let item = builtin_tile_item_from_stack(&stack).expect("builtin chest");
        let model =
            mc_assets::item_model_override_for(stack_ext::props(&stack).id, stack.damage).model;
        let parent = builtin_gui_parent(model);
        let mut expected_entity = Vec::new();
        let mut models = mc_render::entity::tilepass::TileModels::new(false);
        models
            .emit_builtin_item(&item, parent, &mut expected_entity)
            .expect("chest geometry");

        let mut draw = DrawList::new();
        assert!(draw_builtin_tile_item(
            &mut draw, 13, 17, &stack, &item, [1.0; 4]
        ));
        let mut lights = Vec::new();
        for (ui, entity) in draw.vertices().iter().zip(expected_entity.iter()) {
            assert!((ui.pos[0] - (13.0 + 8.0 + entity.pos[0])).abs() < 1e-5);
            assert!((ui.pos[1] - (17.0 + 8.0 + entity.pos[1])).abs() < 1e-5);
            let light = gui_standard_item_light_for_eye_normal(Vec3::from(entity.normal));
            assert_eq!(ui.color, [light, light, light, 1.0]);
            lights.push(light);
        }
        assert!(
            lights.iter().any(|light| *light < 0.99),
            "builtin/entity normals must reach standard GUI lighting"
        );
        assert!(
            lights
                .windows(2)
                .any(|pair| (pair[0] - pair[1]).abs() > 0.01),
            "a chest's differently oriented faces must not stay uniformly lit"
        );
    }

    #[test]
    fn banner_builtin_gui_descriptor_keeps_base_and_pattern_state_for_gpu_upload() {
        let id = mc_data::item_by_name("banner").expect("1.8.9 banner item");
        let mut block_entity = NbtCompound::new();
        block_entity.insert("Base", Nbt::Int(14));
        let mut pattern = NbtCompound::new();
        pattern.insert("Pattern", Nbt::String("cr".to_owned()));
        pattern.insert("Color", Nbt::Int(1));
        block_entity.insert("Patterns", Nbt::list(vec![Nbt::Compound(pattern)]));
        let mut root = NbtCompound::new();
        root.insert("BlockEntityTag", Nbt::Compound(block_entity));
        let banner_stack = ItemStack {
            item_id: id.0 as i16,
            count: 1,
            damage: 0,
            nbt: Some(root),
        };

        let descriptor = builtin_tile_item_from_stack(&banner_stack).expect("banner descriptor");
        assert_eq!(
            descriptor,
            BuiltinTileItem::Banner {
                state: mc_world::BannerState {
                    base_color: 14,
                    patterns: vec![("cr".to_owned(), 1)],
                    initialized: true,
                },
            }
        );
        assert_eq!(
            descriptor.texture_key(false).as_deref(),
            Some("mc-render/banner/b14cr1")
        );

        for base in [
            Nbt::Byte(6),
            Nbt::Short(6),
            Nbt::Int(6),
            Nbt::Long(6),
            Nbt::Float(6.0),
            Nbt::Double(6.0),
        ] {
            let mut block_entity = NbtCompound::new();
            block_entity.insert("Base", base);
            let mut root = NbtCompound::new();
            root.insert("BlockEntityTag", Nbt::Compound(block_entity));
            let mut numeric = stack("banner", 1);
            numeric.damage = 3;
            numeric.nbt = Some(root);
            assert_eq!(banner_item_state(&numeric).base_color, 6);
        }

        let mut string_base = NbtCompound::new();
        string_base.insert("Base", Nbt::String("6".into()));
        let mut string_root = NbtCompound::new();
        string_root.insert("BlockEntityTag", Nbt::Compound(string_base));
        let mut nonnumeric = stack("banner", 1);
        nonnumeric.damage = 3;
        nonnumeric.nbt = Some(string_root);
        assert_eq!(
            banner_item_state(&nonnumeric).base_color,
            3,
            "setItemValues rejects present nonnumeric Base"
        );

        let mut draw = DrawList::new();
        draw_item_icon_tinted(
            &mut draw,
            &UiTexture::gui("textures/atlas/blocks.png"),
            0,
            0,
            &banner_stack,
            0xFFFF_FFFF,
        );
        assert_eq!(draw.builtin_tile_items(), &[descriptor]);
        assert_eq!(
            draw.batches()[0].texture,
            UiTexture::gui("mc-render/banner/b14cr1")
        );
    }

    #[test]
    fn ordinary_empty_model_does_not_take_builtin_entity_upload_route() {
        let stone = stack("stone", 1);
        assert!(builtin_tile_item_from_stack(&stone).is_none());
        let chest = stack("chest", 1);
        assert!(builtin_tile_item_from_stack(&chest).is_some());
    }

    #[test]
    fn unexpected_builtin_model_takes_java_chest_fallback_but_ordinary_model_does_not() {
        let stone = stack("stone", 1);
        let sentinel = BakedModel {
            ambient_occlusion: false,
            gui_3d: true,
            particle_uv: None,
            ..BakedModel::default()
        };
        assert_eq!(
            builtin_tile_item_from_stack_for_model(&stone, &sentinel),
            Some(BuiltinTileItem::Chest { trapped: false })
        );
        assert_eq!(
            builtin_tile_item_from_stack_for_model(&stone, &BakedModel::default()),
            None,
            "the normal empty/default model is not an isBuiltInRenderer branch"
        );

        let mut ordinary_known = sentinel.clone();
        ordinary_known.particle_uv = Some(mc_assets::sprite::UvRect::FULL);
        let chest = stack("chest", 1);
        assert_eq!(
            builtin_tile_item_from_stack_for_model(&chest, &ordinary_known),
            None,
            "even a known family must obey RenderItem's model gate"
        );

        let mut malformed_sentinel = sentinel.clone();
        malformed_sentinel.particle_uv = Some(mc_assets::sprite::UvRect::FULL);
        assert_eq!(
            builtin_tile_item_from_stack_for_model(&stone, &malformed_sentinel),
            None,
            "a hostile/ordinary model carrying a particle sprite must not trigger TEISR"
        );
    }

    /// A head's textures map is written by the server, unsigned. The
    /// request's parser ends an http authority at a backslash, so this URL connects to
    /// `evil.example`; authlib's suffix rule done by hand read the host after the `@`.
    #[test]
    fn a_skull_texture_naming_another_host_is_not_fetched() {
        use base64::Engine as _;

        let encode = |url: &str| {
            let url = url.replace('\\', "\\\\");
            base64::engine::general_purpose::STANDARD
                .encode(format!(r#"{{"textures":{{"SKIN":{{"url":"{url}"}}}}}}"#))
        };
        let smuggled = r"http://evil.example\@textures.minecraft.net/texture/abc";
        assert!(
            authlib_texture_domain(smuggled),
            "negative control: the old gate"
        );
        for url in [
            smuggled,
            "https://api.mojang.com/texture/abc",
            "http://textures.minecraft.net:8080/texture/abc",
        ] {
            assert_eq!(decode_profile_textures(&encode(url)), None, "{url}");
        }
        let (skin, _) = decode_profile_textures(&encode("http://textures.minecraft.net/texture/a"))
            .expect("Mojang's texture host");
        assert_eq!(
            skin.as_deref(),
            Some("http://textures.minecraft.net/texture/a")
        );
    }

    #[test]
    fn player_skull_gui_switches_from_uuid_fallback_to_cached_profile_texture() {
        use base64::Engine as _;

        let uuid = "facecafe-1234-5678-9abc-def012345678";
        let payload = r#"{"textures":{"SKIN":{"url":"https://textures.minecraft.net/texture/gui-player-skull-test","metadata":{"model":"slim"}}}}"#;
        let mut property = NbtCompound::new();
        property.insert(
            "Value",
            Nbt::String(base64::engine::general_purpose::STANDARD.encode(payload)),
        );
        let mut properties = NbtCompound::new();
        properties.insert("textures", Nbt::list(vec![Nbt::Compound(property)]));
        let mut owner = NbtCompound::new();
        owner.insert("Id", Nbt::String(uuid.to_owned()));
        owner.insert("Name", Nbt::String("GuiProfileHead".to_owned()));
        owner.insert("Properties", Nbt::Compound(properties));
        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::Compound(owner));
        let mut skull = stack("skull", 1);
        skull.damage = 3;
        skull.nbt = Some(root);

        let descriptor = builtin_tile_item_from_stack(&skull).expect("player skull descriptor");
        let BuiltinTileItem::PlayerSkull { owner, .. } = &descriptor else {
            panic!("valid SkullOwner must use the player-skull descriptor");
        };
        assert_eq!(owner.model, mc_render::entity::SkinModel::Slim);
        assert_eq!(owner.texture_key(), "textures/entity/steve.png");

        let mut fallback_draw = DrawList::new();
        draw_item_icon_tinted(
            &mut fallback_draw,
            &UiTexture::gui("textures/atlas/blocks.png"),
            0,
            0,
            &skull,
            0xFFFF_FFFF,
        );
        assert_eq!(
            fallback_draw.batches()[0].texture,
            UiTexture::gui("textures/entity/steve.png")
        );

        // Publish directly through SkinCache's no-network insertion seam.  The next GUI list
        // must bind the normalized profile key instead of continuing to load Steve.
        mc_render::entity::skin::publish_shared_skin(
            &owner.cache_key,
            &mc_render::entity::SkinTexture {
                rgba: vec![255; 64 * 64 * 4].into(),
                layout: mc_render::entity::SkinLayout::Modern64x64,
                model: mc_render::entity::SkinModel::Slim,
            },
        );
        let mut ready_draw = DrawList::new();
        draw_item_icon_tinted(
            &mut ready_draw,
            &UiTexture::gui("textures/atlas/blocks.png"),
            0,
            0,
            &skull,
            0xFFFF_FFFF,
        );
        assert_eq!(
            ready_draw.batches()[0].texture,
            UiTexture::gui(&owner.cache_key)
        );
        assert_eq!(ready_draw.builtin_tile_items(), &[descriptor]);
    }

    #[test]
    fn player_skull_gui_uses_legacy_name_offline_uuid_default_without_lookup() {
        let mut root = NbtCompound::new();
        root.insert("SkullOwner", Nbt::String("jeb_".to_owned()));
        let mut skull = stack("skull", 1);
        skull.damage = 3;
        skull.nbt = Some(root);

        let descriptor = builtin_tile_item_from_stack(&skull).expect("legacy skull descriptor");
        let BuiltinTileItem::PlayerSkull { owner, .. } = &descriptor else {
            panic!("a non-empty legacy owner must stay in the player-skull builtin family");
        };
        assert_eq!(owner.legacy_name.as_deref(), Some("jeb_"));
        assert_eq!(owner.fallback_model, mc_render::entity::SkinModel::Slim);
        assert_eq!(owner.texture_key(), "textures/entity/alex.png");

        let mut draw = DrawList::new();
        draw_item_icon_tinted(
            &mut draw,
            &UiTexture::gui("textures/atlas/blocks.png"),
            0,
            0,
            &skull,
            0xFFFF_FFFF,
        );
        assert_eq!(
            draw.batches()[0].texture,
            UiTexture::gui("textures/entity/alex.png")
        );
        assert_eq!(draw.builtin_tile_items(), &[descriptor]);

        // Java only promotes a non-empty TAG_String.  Missing, empty, wrong-typed, and empty
        // compound owners all leave the skull renderer's ordinary type-3/Steve fallback in
        // place; none may fabricate a UUID or a cache request.
        for owner in [
            None,
            Some(Nbt::String(String::new())),
            Some(Nbt::Int(9)),
            Some(Nbt::Compound(NbtCompound::new())),
        ] {
            let mut malformed = stack("skull", 1);
            malformed.damage = 3;
            let mut malformed_root = NbtCompound::new();
            if let Some(owner) = owner {
                malformed_root.insert("SkullOwner", owner);
            }
            malformed.nbt = Some(malformed_root);
            assert_eq!(
                builtin_tile_item_from_stack(&malformed),
                Some(BuiltinTileItem::Skull { skull_type: 3 }),
                "malformed/missing owner must retain the ordinary player-skull family"
            );
        }
    }

    #[test]
    fn builtin_entity_items_are_reachable_from_the_full_inventory_slot_renderer() {
        let chest = stack("chest", 1);
        let mut draw = DrawList::new();
        let mut font = crate::widget::test_support::uniform_font();
        draw_item_in_slot(
            &mut draw,
            &UiTexture::gui("textures/atlas/blocks.png"),
            &mut font,
            4,
            5,
            &chest,
        );

        assert_eq!(draw.vertices().len(), 3 * 6 * 6);
        assert_eq!(
            draw.builtin_tile_items(),
            &[BuiltinTileItem::Chest { trapped: false }]
        );
        assert_eq!(
            draw.batches().len(),
            1,
            "builtin icons do not take the empty-model glint path"
        );
    }

    #[test]
    fn unregistered_item_ids_emit_no_slot_icon_or_overlays() {
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        for raw_id in [8_u16, 60_000] {
            let stack = ItemStack {
                item_id: raw_id as i16,
                count: 64,
                damage: 7,
                nbt: None,
            };
            let mut draw = DrawList::new();
            let mut font = crate::widget::test_support::uniform_font();
            draw_item_in_slot(&mut draw, &atlas, &mut font, 4, 5, &stack);
            assert!(draw.vertices().is_empty(), "unregistered item id {raw_id}");
            assert!(draw.batches().is_empty(), "unregistered item id {raw_id}");
            assert!(
                draw.builtin_tile_items().is_empty(),
                "unregistered item id {raw_id}"
            );
        }
    }

    #[test]
    fn a_matching_filled_map_slot_uses_the_keyed_image_and_ordered_icons() {
        let stack = ItemStack {
            item_id: stack("filled_map", 1).item_id,
            count: 1,
            damage: -7,
            nbt: None,
        };
        let mut data = MapData::new(-7);
        data.colors[3 + 4 * mc_world::MAP_SIZE] = 8 * 4 + 2;
        data.decorations.push(mc_world::MapDecoration {
            key: "icon-0".to_owned(),
            icon_type: 1,
            x: 0,
            z: 0,
            direction: 4,
        });

        let mut draw = DrawList::new();
        let mut font = crate::widget::test_support::uniform_font();
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        assert!(draw_item_icon_with_map_data(
            &mut draw,
            &atlas,
            20,
            30,
            &stack,
            Some(&data),
        ));
        assert_eq!(draw.vertices().len(), 12, "map image plus one icon");
        assert_eq!(
            draw.batches()[0].texture,
            crate::map::dynamic_texture_key(-7)
        );
        assert_eq!(draw.batches()[1].texture, UiTexture::gui(MAP_ICONS_TEXTURE));
        assert_eq!(
            draw.vertices()[0..6].iter().map(|vertex| vertex.pos).fold(
                (
                    [f32::INFINITY, f32::INFINITY],
                    [f32::NEG_INFINITY, f32::NEG_INFINITY]
                ),
                |(min, max), pos| {
                    (
                        [min[0].min(pos[0]), min[1].min(pos[1])],
                        [max[0].max(pos[0]), max[1].max(pos[1])],
                    )
                },
            ),
            ([20.0, 30.0], [36.0, 46.0]),
            "1:8 map scale fills exactly one GUI slot"
        );

        // The full helper must retain the normal overlay contract after the live icon pass.
        let mut full = DrawList::new();
        draw_item_in_slot_with_map_data(&mut full, &atlas, &mut font, 20, 30, &stack, Some(&data));
        assert_eq!(full.vertices().len(), 12);
    }

    #[test]
    fn an_unknown_or_miskeyed_filled_map_keeps_the_baked_fallback() {
        let mut stack = stack("filled_map", 1);
        stack.damage = 12;
        let wrong = MapData::new(13);
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let mut draw = DrawList::new();
        assert!(!draw_item_icon_with_map_data(
            &mut draw,
            &atlas,
            0,
            0,
            &stack,
            Some(&wrong),
        ));
        assert!(draw
            .batches()
            .iter()
            .all(|batch| batch.texture != crate::map::dynamic_texture_key(12)));

        let mut no_record = DrawList::new();
        assert!(!draw_item_icon_with_map_data(
            &mut no_record,
            &atlas,
            0,
            0,
            &stack,
            None,
        ));
        assert!(no_record
            .batches()
            .iter()
            .all(|batch| batch.texture != crate::map::dynamic_texture_key(12)));
    }

    fn assert_rect(vertices: &[UiVertex], bounds: (f32, f32, f32, f32), color: [f32; 4]) {
        assert_eq!(vertices.len(), 6);
        let min_x = vertices
            .iter()
            .map(|vertex| vertex.pos[0])
            .fold(f32::INFINITY, f32::min);
        let min_y = vertices
            .iter()
            .map(|vertex| vertex.pos[1])
            .fold(f32::INFINITY, f32::min);
        let max_x = vertices
            .iter()
            .map(|vertex| vertex.pos[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let max_y = vertices
            .iter()
            .map(|vertex| vertex.pos[1])
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!((min_x, min_y, max_x, max_y), bounds);
        assert!(vertices.iter().all(|vertex| vertex.color == color));
    }

    #[test]
    fn undamaged_items_show_no_durability_bar() {
        assert_eq!(durability_fraction(&stack("diamond_sword", 1)), None);
    }

    #[test]
    fn a_freshly_broken_almost_item_is_nearly_empty() {
        let mut s = stack("diamond_sword", 1);
        s.damage = 1560; // max_damage is 1561
        let f = durability_fraction(&s).unwrap();
        assert!(f > 0.0 && f < 0.01, "{f}");
    }

    #[test]
    fn half_damaged_is_half_full() {
        let mut s = stack("diamond_sword", 1);
        s.damage = 1561 / 2;
        let f = durability_fraction(&s).unwrap();
        assert!((f - 0.5).abs() < 0.01, "{f}");
    }

    #[test]
    fn durability_colour_is_vanillas_linear_red_green_cross_fade() {
        let full = durability_bar_color(1.0);
        let half = durability_bar_color(0.5);
        let empty = durability_bar_color(0.0);
        assert_eq!(full, 0xFF00_FF00);
        assert_eq!(half, 0xFF7F_8000);
        assert_eq!(empty, 0xFFFF_0000);
    }

    #[test]
    fn durability_bar_draws_nothing_for_an_undamaged_stack() {
        let mut d = DrawList::new();
        draw_durability_bar(&mut d, 0, 0, &stack("diamond_sword", 1));
        assert!(d.is_empty());
    }

    #[test]
    fn durability_bar_draws_backing_track_and_foreground_with_java_1_8_metrics() {
        let mut d = DrawList::new();
        let mut s = stack("diamond_sword", 1);
        s.damage = 1561 / 2; // ~half
        draw_durability_bar(&mut d, 10, 20, &s);

        // Java computes width=7 and i=128 for damage 780 / maxDamage 1561. The draw order is
        // significant: the 12x1 dark track covers the black backing, then the 7x1 foreground
        // covers the filled portion of that track.
        assert_eq!(durability_bar_metrics(&s), Some((7, 128)));
        assert_eq!(d.vertices().len(), 18);
        assert_rect(
            &d.vertices()[0..6],
            (12.0, 33.0, 25.0, 35.0),
            [0.0, 0.0, 0.0, 1.0],
        );
        assert_rect(
            &d.vertices()[6..12],
            (12.0, 33.0, 24.0, 34.0),
            [31.0 / 255.0, 64.0 / 255.0, 0.0, 1.0],
        );
        assert_rect(
            &d.vertices()[12..18],
            (12.0, 33.0, 19.0, 34.0),
            [127.0 / 255.0, 128.0 / 255.0, 0.0, 1.0],
        );
    }

    #[test]
    fn fully_spent_item_keeps_javas_zero_width_red_foreground_call() {
        let mut d = DrawList::new();
        let mut s = stack("diamond_sword", 1);
        s.damage = 1561; // exactly max_damage: Java still enters isItemDamaged().
        draw_durability_bar(&mut d, 10, 20, &s);

        // RenderItem does not special-case an empty bar: its three draw() calls include a
        // zero-width red foreground after the 13x2 black backing and 12x1 dark track. This
        // catches both a tempting `width > 0` omission and the later-version HSV ramp.
        assert_eq!(durability_bar_metrics(&s), Some((0, 0)));
        assert_eq!(d.vertices().len(), 18);
        assert_rect(
            &d.vertices()[0..6],
            (12.0, 33.0, 25.0, 35.0),
            [0.0, 0.0, 0.0, 1.0],
        );
        assert_rect(
            &d.vertices()[6..12],
            (12.0, 33.0, 24.0, 34.0),
            [63.0 / 255.0, 64.0 / 255.0, 0.0, 1.0],
        );
        assert_rect(
            &d.vertices()[12..18],
            (12.0, 33.0, 12.0, 34.0),
            [1.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn count_of_one_draws_nothing() {
        let mut d = DrawList::new();
        let mut font = crate::widget::test_support::uniform_font();
        draw_count_text(&mut d, &mut font, 0, 0, &stack("stone", 1));
        assert!(d.is_empty());
    }

    #[test]
    fn count_above_one_draws_text() {
        let mut d = DrawList::new();
        let mut font = crate::widget::test_support::uniform_font();
        draw_count_text(&mut d, &mut font, 0, 0, &stack("stone", 32));
        assert!(!d.is_empty());
    }

    #[test]
    fn zero_negative_and_override_counts_follow_javas_overlay_gate() {
        for count in [0, -1] {
            let mut draw = DrawList::new();
            let mut font = crate::widget::test_support::uniform_font();
            draw_count_text(&mut draw, &mut font, 0, 0, &stack("stone", count));
            assert!(!draw.is_empty(), "count {count} remains visible");
            assert!(
                draw.vertices()
                    .iter()
                    .any(|vertex| vertex.color[0] > vertex.color[1]),
                "count {count} uses the red formatting branch"
            );
        }

        let mut draw = DrawList::new();
        let mut font = crate::widget::test_support::uniform_font();
        draw_count_text_with_override(
            &mut draw,
            &mut font,
            0,
            0,
            &stack("stone", 1),
            Some("override"),
        );
        assert!(
            !draw.is_empty(),
            "override text bypasses the count-one gate"
        );
    }

    #[test]
    fn cube_projection_produces_three_faces_when_the_model_has_all_three() {
        let model = full_cube_model();
        let faces = iso_faces(&model, 0, 0);
        assert_eq!(faces.len(), 3);
    }

    #[test]
    fn gui_3d_path_replays_every_baked_quad_not_a_synthetic_three_face_cube() {
        let mut model = single_flat_quad_model();
        let quad = model.general_quads[0].clone();
        model
            .general_quads
            .extend([quad.clone(), quad.clone(), quad]);
        let mut draw = DrawList::new();
        draw_iso_block_icon(
            &mut draw,
            &UiTexture::gui("textures/atlas/blocks.png"),
            0,
            0,
            &model,
        );

        assert_eq!(
            draw.vertices().len(),
            24,
            "four baked quads become eight triangles"
        );
    }

    #[test]
    fn gui_3d_production_path_applies_java_directional_face_lighting() {
        let model = full_cube_model();
        let mut draw = DrawList::new();
        draw_iso_block_icon(
            &mut draw,
            &UiTexture::gui("textures/atlas/blocks.png"),
            0,
            0,
            &model,
        );

        assert_eq!(
            draw.vertices().len(),
            18,
            "GUI back-face culling keeps the top/east/north model faces"
        );
        let face_lights: Vec<f32> = draw
            .vertices()
            .chunks_exact(6)
            .map(|face| {
                let color = face[0].color;
                assert!(face.iter().all(|vertex| vertex.color == color));
                assert_eq!(color[3], 1.0, "fixed-function lighting preserves alpha");
                assert!((color[0] - color[1]).abs() < f32::EPSILON);
                assert!((color[1] - color[2]).abs() < f32::EPSILON);
                color[0]
            })
            .collect();
        assert!(
            face_lights.iter().any(|light| *light < 0.99),
            "directional lighting must not collapse to uniform white: {face_lights:?}"
        );
        assert!(
            face_lights
                .windows(2)
                .any(|pair| (pair[0] - pair[1]).abs() > 0.01),
            "opposing baked faces need observably different illumination: {face_lights:?}"
        );
    }

    #[test]
    fn gui_display_transform_is_applied_to_flat_baked_vertices() {
        let mut translated = single_flat_quad_model();
        translated.transforms.gui.translation = [0.25, 0.0, 0.0];
        let texture = UiTexture::gui("textures/atlas/blocks.png");
        let mut baseline = DrawList::new();
        let mut moved = DrawList::new();
        draw_flat_icon(&mut baseline, &texture, 0, 0, &single_flat_quad_model());
        draw_flat_icon(&mut moved, &texture, 0, 0, &translated);

        let left = |draw: &DrawList| {
            draw.vertices()
                .iter()
                .map(|vertex| vertex.pos[0])
                .fold(f32::INFINITY, f32::min)
        };
        assert_eq!(left(&moved) - left(&baseline), 8.0);
    }

    #[test]
    fn the_top_face_is_brightest_and_the_north_face_is_darkest() {
        let model = full_cube_model();
        let faces = iso_faces(&model, 0, 0);
        let top = faces[0].shade;
        let east = faces[1].shade;
        let north = faces[2].shade;
        assert!(top > east && east > north, "{top} {east} {north}");
    }

    #[test]
    fn gui_3d_cull_matches_java_front_faces_and_rejects_the_back_faces() {
        let model = full_cube_model();
        let matrix = baked_gui_matrix(&model, true);
        let front = [Facing::Up, Facing::North, Facing::East];
        let back = [Facing::Down, Facing::South, Facing::West];

        for face in front {
            let quad = model.face_quads[face.index() as usize]
                .first()
                .expect("full-cube control has every face");
            assert!(
                gui_3d_quad_is_front_facing(matrix, quad),
                "Java GUI camera should expose {face:?}"
            );
        }
        for face in back {
            let quad = model.face_quads[face.index() as usize]
                .first()
                .expect("full-cube control has every face");
            assert!(
                !gui_3d_quad_is_front_facing(matrix, quad),
                "Java back-face cull should reject {face:?}"
            );
        }
    }

    #[test]
    fn gui_3d_submission_preserves_baked_order_and_retains_java_depth_values() {
        let mut first = single_flat_quad_model().general_quads[0].clone();
        first.verts[0].uv = [0.10, 0.20];
        let mut second = first.clone();
        second.verts[0].uv = [0.70, 0.80];
        // Put the second card at a different model depth. Java's drawSlot re-enables depth before
        // RenderItem, but the model traversal itself remains in list order; the depth stream must
        // retain both facts instead of sorting the CPU submission.
        second
            .verts
            .iter_mut()
            .for_each(|vertex| vertex.pos[2] = 0.125);
        let model = BakedModel {
            general_quads: vec![first, second],
            ..single_flat_quad_model()
        };
        let mut draw = DrawList::new();
        draw_iso_block_icon(
            &mut draw,
            &UiTexture::gui("textures/atlas/blocks.png"),
            0,
            0,
            &model,
        );

        assert_eq!(draw.vertices().len(), 12);
        assert_eq!(draw.vertices()[0].uv, [0.10, 0.20]);
        assert_eq!(draw.vertices()[6].uv, [0.70, 0.80]);
        assert_eq!(draw.batches()[0].depth, UiDepth::Item);
        assert_eq!(draw.batches()[0].cull, UiCull::Back);
        assert_ne!(draw.depths()[0], draw.depths()[6]);
    }

    #[test]
    fn every_projected_iso_corner_lands_inside_the_sixteen_pixel_box_with_margin() {
        let model = full_cube_model();
        let faces = iso_faces(&model, 100, 200);
        for face in &faces {
            for p in face.pos {
                assert!(p[0] >= 99.0 && p[0] <= 117.0, "x {p:?}");
                assert!(p[1] >= 199.0 && p[1] <= 217.0, "y {p:?}");
            }
        }
    }

    #[test]
    fn flat_icon_bounding_box_matches_the_quads_own_extent() {
        let model = single_flat_quad_model();
        let mut d = DrawList::new();
        let tex = UiTexture::gui("textures/atlas/blocks.png");
        draw_flat_icon(&mut d, &tex, 0, 0, &model);
        assert_eq!(d.vertices().len(), 6);
        let xs: Vec<f32> = d.vertices().iter().map(|v| v.pos[0]).collect();
        let ys: Vec<f32> = d.vertices().iter().map(|v| v.pos[1]).collect();
        assert!((xs.iter().cloned().fold(f32::MAX, f32::min) - 0.0).abs() < 0.001);
        assert!((xs.iter().cloned().fold(f32::MIN, f32::max) - 16.0).abs() < 0.001);
        assert!((ys.iter().cloned().fold(f32::MAX, f32::min) - 0.0).abs() < 0.001);
        assert!((ys.iter().cloned().fold(f32::MIN, f32::max) - 16.0).abs() < 0.001);
        assert!(d.vertices().iter().all(|vertex| vertex.color == [1.0; 4]));
    }

    #[test]
    fn gui_3d_matrix_centers_the_cube_and_scales_each_unit_edge_to_ten_pixels() {
        let matrix = baked_gui_matrix(&full_cube_model(), true);
        let origin = matrix.transform_point3(Vec3::ZERO);
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let edge = matrix.transform_point3(axis) - origin;
            assert!((edge.length() - 10.0).abs() < 0.000_1, "edge {edge:?}");
        }

        let center = matrix.transform_point3(Vec3::splat(0.5));
        let screen_center = [center.x + 8.0, center.y + 8.0];
        assert!((screen_center[0] - 8.0).abs() < 0.000_1);
        assert!((screen_center[1] - 8.0).abs() < 0.000_1);
    }

    #[test]
    fn potion_flat_icon_tints_only_liquid_layer_zero() {
        let model = two_layer_flat_model();
        let mut potion = stack("potion", 1);
        potion.damage = 8194;
        let mut draw = DrawList::new();
        let texture = UiTexture::gui("textures/atlas/blocks.png");
        draw_flat_item_icon(&mut draw, &texture, 0, 0, &potion, &model);

        assert_eq!(draw.vertices().len(), 12);
        let liquid = [124.0 / 255.0, 175.0 / 255.0, 198.0 / 255.0, 1.0];
        assert!(draw.vertices()[..6]
            .iter()
            .all(|vertex| vertex.color == liquid));
        assert!(draw.vertices()[6..]
            .iter()
            .all(|vertex| vertex.color == [1.0; 4]));

        assert_eq!(item_quad_color(&potion, Some(0)), liquid);
        assert_eq!(item_quad_color(&potion, Some(1)), [1.0; 4]);
        assert_eq!(item_quad_color(&potion, None), [1.0; 4]);
        assert_eq!(item_quad_color(&stack("apple", 1), Some(0)), [1.0; 4]);
        assert_eq!(
            modulate_color([0.5, 0.25, 1.0, 0.75], liquid),
            [
                0.5 * 124.0 / 255.0,
                0.25 * 175.0 / 255.0,
                198.0 / 255.0,
                0.75,
            ],
            "caller dimming multiplies rather than replaces item tint"
        );
    }

    #[test]
    fn firework_star_flat_icon_tints_only_overlay_layer_one() {
        let model = two_layer_flat_model();
        let mut explosion = NbtCompound::new();
        explosion.insert("Colors", Nbt::IntArray(vec![11_743_532]));
        let mut root = NbtCompound::new();
        root.insert("Explosion", Nbt::Compound(explosion));
        let mut star = stack("firework_charge", 1);
        star.nbt = Some(root);

        let mut draw = DrawList::new();
        let texture = UiTexture::gui("textures/atlas/blocks.png");
        draw_flat_item_icon(&mut draw, &texture, 0, 0, &star, &model);

        assert_eq!(draw.vertices().len(), 12);
        let red = [179.0 / 255.0, 49.0 / 255.0, 44.0 / 255.0, 1.0];
        assert!(draw.vertices()[..6]
            .iter()
            .all(|vertex| vertex.color == [1.0; 4]));
        assert!(draw.vertices()[6..]
            .iter()
            .all(|vertex| vertex.color == red));

        assert_eq!(item_quad_color(&star, Some(0)), [1.0; 4]);
        assert_eq!(item_quad_color(&star, Some(1)), red);
        assert_eq!(item_quad_color(&star, None), [1.0; 4]);
        assert_eq!(
            item_quad_color(&stack("firework_charge", 1), Some(1)),
            [138.0 / 255.0, 138.0 / 255.0, 138.0 / 255.0, 1.0],
            "missing Colors uses ItemFireworkCharge's 0x8A8A8A fallback"
        );

        let mut potion = stack("potion", 1);
        potion.damage = 8194;
        assert_eq!(
            item_quad_color(&potion, Some(1)),
            [1.0; 4],
            "the shared tint path remains item- and index-specific"
        );
    }

    /// Oracle: `ItemMonsterPlacer.getColorFromItemStack`; `EntityList.java` registers Zombie
    /// (metadata 54) as `(44975, 7969893)`.  The tint-index controls cover the Java `== 0`
    /// branch and its deliberately broad nonzero branch, while the metadata controls cover an
    /// entity that has no egg and a metadata value outside the `u8` entity registry.
    #[test]
    fn spawn_egg_flat_icon_uses_exact_entity_palette_per_tint_index() {
        let model = two_layer_flat_model();
        let mut egg = stack("spawn_egg", 1);
        egg.damage = 54;
        let mut draw = DrawList::new();
        let texture = UiTexture::gui("textures/atlas/blocks.png");
        draw_flat_item_icon(&mut draw, &texture, 0, 0, &egg, &model);

        let primary = [0.0, 175.0 / 255.0, 175.0 / 255.0, 1.0];
        let secondary = [121.0 / 255.0, 156.0 / 255.0, 101.0 / 255.0, 1.0];
        assert!(draw.vertices()[..6]
            .iter()
            .all(|vertex| vertex.color == primary));
        assert!(draw.vertices()[6..]
            .iter()
            .all(|vertex| vertex.color == secondary));
        assert_eq!(item_quad_color(&egg, Some(0)), primary);
        assert_eq!(item_quad_color(&egg, Some(1)), secondary);
        assert_eq!(
            item_quad_color(&egg, Some(7)),
            secondary,
            "Java chooses secondary for every tint index except zero"
        );
        assert_eq!(item_quad_color(&egg, None), [1.0; 4]);

        egg.damage = 53; // Giant: registered entity, but absent from EntityList.entityEggs.
        assert_eq!(item_quad_color(&egg, Some(0)), [1.0; 4]);
        egg.damage = 300;
        assert_eq!(item_quad_color(&egg, Some(1)), [1.0; 4]);
    }

    /// Oracle: `ItemArmor.getColorFromItemStack`: layer zero is leather's custom `TAG_Int` or
    /// `0xA06540`; every other render pass is opaque white.
    #[test]
    fn leather_armor_flat_icon_tints_only_layer_zero_with_a_real_int_tag() {
        let mut leather = stack("leather_helmet", 1);
        let mut display = NbtCompound::new();
        display.insert("color", Nbt::Int(0x12_3456));
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        leather.nbt = Some(root);

        assert_eq!(
            item_quad_color(&leather, Some(0)),
            [18.0 / 255.0, 52.0 / 255.0, 86.0 / 255.0, 1.0]
        );
        assert_eq!(item_quad_color(&leather, Some(1)), [1.0; 4]);

        let plain = stack("leather_boots", 1);
        assert_eq!(
            item_quad_color(&plain, Some(0)),
            [160.0 / 255.0, 101.0 / 255.0, 64.0 / 255.0, 1.0]
        );

        let mut wrong_tag_type = stack("leather_chestplate", 1);
        let mut display = NbtCompound::new();
        display.insert("color", Nbt::Byte(0x12));
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        wrong_tag_type.nbt = Some(root);
        assert_eq!(
            item_quad_color(&wrong_tag_type, Some(0)),
            [160.0 / 255.0, 101.0 / 255.0, 64.0 / 255.0, 1.0],
            "NBTTagCompound.hasKey(\"color\", 3) rejects a TAG_Byte"
        );
    }

    /// Oracle: `ItemColored`/`ItemLeaves`/`ItemLilyPad` delegate to their block's
    /// `getRenderColor`, while `ItemDoublePlant` asks `ColorizerGrass` only for GRASS and FERN.
    /// The leaves/vine/lily values are fixed Java constants; the grass values intentionally come
    /// from the shared colormap seam, whose zero-filled pre-reload state is also vanilla's state.
    #[test]
    fn colored_and_foliage_item_tints_follow_the_block_render_color_overrides() {
        let vine = stack("vine", 1);
        assert_eq!(item_quad_color_rgb(&vine, Some(0)), 0x48_B5_18);
        assert_eq!(
            item_quad_color_rgb(&vine, None),
            0xFF_FF_FF,
            "untagged quads keep Item's white default"
        );

        let mut leaves = stack("leaves", 1);
        for (metadata, expected) in [
            (0, 0x48_B5_18), // oak -> foliage basic
            (1, 0x61_99_61), // spruce -> foliage pine
            (2, 0x80_A7_55), // birch -> foliage birch
            (3, 0x48_B5_18), // jungle -> foliage basic
        ] {
            leaves.damage = metadata;
            assert_eq!(
                item_quad_color_rgb(&leaves, Some(0)),
                expected,
                "leaves metadata {metadata}"
            );
        }

        let mut leaves2 = stack("leaves2", 1);
        leaves2.damage = 1;
        assert_eq!(
            item_quad_color_rgb(&leaves2, Some(7)),
            0x48_B5_18,
            "acacia/dark-oak ItemLeaves uses BlockNewLeaf's basic foliage colour for every pass"
        );

        let mut lily_pad = stack("waterlily", 1);
        lily_pad.damage = 15;
        assert_eq!(
            item_quad_color_rgb(&lily_pad, Some(9)),
            0x71_C3_5C,
            "ItemLilyPad delegates to BlockLilyPad.getRenderColor, independent of metadata"
        );

        let grass_basic = mc_render::terrain::tint::render_color(mc_data::BlockState::new(
            mc_data::BlockId(31),
            1,
        ));
        let grass = stack("grass", 1);
        assert_eq!(
            item_quad_color_rgb(&grass, Some(0)),
            mc_render::terrain::tint::render_color(mc_data::BlockState::new(
                mc_data::BlockId(2),
                0,
            )),
            "ItemColored grass uses BlockGrass.getRenderColor's fixed (.5, 1.0) lookup"
        );

        let mut tallgrass = stack("tallgrass", 1);
        tallgrass.damage = 1;
        assert_eq!(item_quad_color_rgb(&tallgrass, Some(0)), grass_basic);
        tallgrass.damage = 2;
        assert_eq!(item_quad_color_rgb(&tallgrass, Some(0)), grass_basic);
        tallgrass.damage = 3;
        assert_eq!(
            item_quad_color_rgb(&tallgrass, Some(0)),
            0xFF_FF_FF,
            "BlockTallGrass.byMetadata(3) falls back to DEAD_BUSH, not grass"
        );

        let mut double_plant = stack("double_plant", 1);
        for metadata in [2, 3] {
            double_plant.damage = metadata;
            assert_eq!(
                item_quad_color_rgb(&double_plant, Some(0)),
                grass_basic,
                "ItemDoublePlant grass/fern metadata {metadata}"
            );
            assert_eq!(
                item_quad_color_rgb(&double_plant, Some(1)),
                0xFF_FF_FF,
                "ItemDoublePlant only tints render pass zero (metadata {metadata})"
            );
        }
        double_plant.damage = 4;
        assert_eq!(
            item_quad_color_rgb(&double_plant, Some(0)),
            0xFF_FF_FF,
            "rose delegates to Item's white default"
        );
    }

    /// Oracle: `ItemBanner.getBaseColor` prefers `BlockEntityTag.Base` whenever the key exists,
    /// then `EnumDyeColor.byDyeDamage` supplies `MapColor.colorValue`; render pass zero is white.
    /// The malformed-string control is important because Java's untyped `getInteger` turns it
    /// into zero (black), rather than falling back to the stack's metadata.
    #[test]
    fn banner_item_tint_uses_base_dye_map_color_and_java_nbt_fallbacks() {
        let mut banner = stack("banner", 1);
        banner.damage = 1; // EnumDyeColor.RED -> MapColor.redColor.
        assert_eq!(item_quad_color_rgb(&banner, Some(0)), 0xFF_FF_FF);
        assert_eq!(item_quad_color_rgb(&banner, Some(1)), 0x99_33_33);
        assert_eq!(
            item_quad_color_rgb(&banner, Some(7)),
            0x99_33_33,
            "every nonzero render pass uses the base dye colour"
        );
        assert_eq!(item_quad_color_rgb(&banner, None), 0xFF_FF_FF);

        let mut block_entity = NbtCompound::new();
        block_entity.insert("Base", Nbt::Int(11)); // yellow dye damage.
        let mut root = NbtCompound::new();
        root.insert("BlockEntityTag", Nbt::Compound(block_entity));
        banner.nbt = Some(root);
        assert_eq!(
            item_quad_color_rgb(&banner, Some(1)),
            0xE5_E5_33,
            "BlockEntityTag.Base overrides the stack metadata"
        );

        let mut wrong_type = NbtCompound::new();
        wrong_type.insert("Base", Nbt::String("yellow".into()));
        let mut wrong_root = NbtCompound::new();
        wrong_root.insert("BlockEntityTag", Nbt::Compound(wrong_type));
        banner.nbt = Some(wrong_root);
        assert_eq!(
            item_quad_color_rgb(&banner, Some(1)),
            0x19_19_19,
            "present nonnumeric Base is Java getInteger(\"Base\") == 0 (black dye)"
        );

        banner.nbt = None;
        banner.damage = 99;
        assert_eq!(
            item_quad_color_rgb(&banner, Some(1)),
            0x19_19_19,
            "EnumDyeColor.byDyeDamage out of range normalizes to black"
        );
    }

    #[test]
    fn gui_3d_item_model_applies_stack_tint_only_to_tagged_baked_quads() {
        let mut model = single_flat_quad_model();
        model.general_quads[0].tint_index = Some(0);
        let mut untagged = model.general_quads[0].clone();
        untagged.tint_index = None;
        model.general_quads.push(untagged);

        let leaves = stack("leaves", 1);
        let mut draw = DrawList::new();
        draw_gui_item_model_tinted(
            &mut draw,
            &UiTexture::gui("textures/atlas/blocks.png"),
            0,
            0,
            &leaves,
            &model,
            [1.0; 4],
        );

        assert_eq!(
            draw.vertices().len(),
            12,
            "two baked quads become four triangles"
        );
        let foliage = [72.0 / 255.0, 181.0 / 255.0, 24.0 / 255.0, 1.0];
        let light = gui_standard_item_light(&model, &model.general_quads[0]);
        let lit_foliage = apply_gui_standard_item_light(foliage, light);
        assert!(draw.vertices()[..6]
            .iter()
            .all(|vertex| vertex.color == lit_foliage));
        assert!(draw.vertices()[6..]
            .iter()
            .all(|vertex| vertex.color == [light, light, light, 1.0]));

        let mut dimmed = DrawList::new();
        draw_gui_item_model_tinted(
            &mut dimmed,
            &UiTexture::gui("textures/atlas/blocks.png"),
            0,
            0,
            &leaves,
            &model,
            [0.5, 0.25, 1.0, 0.75],
        );
        assert_eq!(
            dimmed.vertices()[0].color,
            [
                0.5 * 72.0 / 255.0 * light,
                0.25 * 181.0 / 255.0 * light,
                24.0 / 255.0 * light,
                0.75,
            ],
            "caller colour and stack tint multiply the GUI light while alpha stays unchanged"
        );
    }

    /// The slot renderer's foil is an actual resource-texture batch over the baked geometry,
    /// not merely a boolean exposed for callers to forget. A flat generated icon has one quad,
    /// so RenderItem's two glint passes make exactly twelve overlay vertices. The explicit batch
    /// kind and UV assertions below pin the owned UI representation's blend and animation state.
    #[test]
    fn glint_overlay_is_visible_as_two_violet_texture_passes() {
        let model = single_flat_quad_model();
        let mut draw = DrawList::new();
        draw_glint_overlay_at(&mut draw, 10, 20, &model, false, 1_500);

        assert_eq!(draw.vertices().len(), 12);
        assert_eq!(draw.batches().len(), 1);
        assert_eq!(
            draw.batches()[0].texture,
            UiTexture::gui(ENCHANTED_ITEM_GLINT_TEXTURE)
        );
        assert_eq!(draw.batches()[0].blend, UiBlend::Glint);
        assert!(draw.vertices().iter().all(|v| v.color == GLINT_COLOR));
        assert_eq!(draw.vertices()[0].pos, [10.0, 36.0]);
        assert_eq!(
            draw.vertices()[0].pos,
            draw.vertices()[6].pos,
            "both Java foil passes replay the same baked geometry"
        );
        assert_eq!(
            draw.vertices()[0].uv,
            item_glint_uv([0.0, 1.0], 1_500, 0),
            "the first foil pass carries its rotated/scrolled texture coordinates"
        );
        assert_eq!(
            draw.vertices()[6].uv,
            item_glint_uv([0.0, 1.0], 1_500, 1),
            "the second foil pass carries its independent rotated/scrolled coordinates"
        );
        assert_ne!(
            draw.vertices()[0].uv,
            draw.vertices()[6].uv,
            "the two periods/angles must not collapse into a static duplicate"
        );
    }

    #[test]
    fn glint_uv_matches_java_texture_matrix_periods_and_rotations() {
        let passes = item_glint_passes(1_500);
        assert_eq!(passes[0].period_ms, 3_000);
        assert_eq!(passes[1].period_ms, 4_873);
        assert_eq!(passes[0].rotation_degrees, -50.0);
        assert_eq!(passes[1].rotation_degrees, 10.0);
        assert_eq!(passes[0].translation, 0.0625);

        // At t=0 the translation is zero, so these are independent controls for the two
        // rotations and the scale(8) in RenderItem.renderEffect.
        let first = item_glint_uv([1.0, 0.0], 0, 0);
        assert!((first[0] - 5.142_301).abs() < 0.000_01, "{first:?}");
        assert!((first[1] + 6.128_355_5).abs() < 0.000_01, "{first:?}");
        let second = item_glint_uv([1.0, 0.0], 0, 1);
        assert!((second[0] - 7.878_462).abs() < 0.000_01, "{second:?}");
        assert!((second[1] - 1.389_185_4).abs() < 0.000_01, "{second:?}");

        // Java's first pass is periodic in 3000 ms and its post-scale translation is the
        // remainder/period (the /8 is cancelled by the preceding scale(8)).
        assert_eq!(
            item_glint_uv([0.0, 0.0], 3_000, 0),
            item_glint_uv([0.0, 0.0], 0, 0)
        );
        assert!((item_glint_uv([0.0, 0.0], 1_500, 0)[0] - 0.5).abs() < 0.000_001);
    }

    /// Java has six relevant `Item.hasEffect` implementations: the ordinary enchanted-stack
    /// default; potion and golden-apple metadata/effect overrides; plus four unconditional
    /// classes. The shared tooltip predicate covers the first four, while this owned GUI seam
    /// adds the writable book's signed form and the sole `ItemSimpleFoiled` registration.
    #[test]
    fn gui_glint_dispatch_covers_every_vanilla_item_class_override() {
        let mut enchanted_sword = stack("diamond_sword", 1);
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(vec![]));
        enchanted_sword.nbt = Some(root);

        for name in [
            "enchanted_book",
            "experience_bottle",
            "written_book",
            "nether_star",
        ] {
            assert!(has_gui_effect(&stack(name, 1)), "{name} is always foiled");
        }
        assert!(has_gui_effect(&enchanted_sword));
        let mut speed_potion = stack("potion", 1);
        speed_potion.damage = 8194;
        assert!(has_gui_effect(&speed_potion));

        let mut apple = stack("golden_apple", 1);
        assert!(!has_gui_effect(&apple));
        apple.damage = 1;
        assert!(has_gui_effect(&apple));
        assert!(!has_gui_effect(&stack("writable_book", 1)));
        assert!(!has_gui_effect(&stack("diamond_sword", 1)));
    }

    #[test]
    fn always_foiled_gui_items_reach_the_two_pass_overlay() {
        for name in ["written_book", "nether_star"] {
            let stack = stack(name, 1);
            let mut draw = DrawList::new();
            draw_item_glint_if_needed_at(
                &mut draw,
                0,
                0,
                &stack,
                &single_flat_quad_model(),
                false,
                0,
            );

            let glint_batches: Vec<_> = draw
                .batches()
                .iter()
                .filter(|batch| batch.texture == UiTexture::gui(ENCHANTED_ITEM_GLINT_TEXTURE))
                .collect();
            assert_eq!(glint_batches.len(), 1, "{name}: foil texture batch");
            assert_eq!(glint_batches[0].blend, UiBlend::Glint);
            let glint_vertices = &draw.vertices()[glint_batches[0].range.clone()];
            assert!(
                !glint_vertices.is_empty() && glint_vertices.len().is_multiple_of(12),
                "{name}: each baked quad needs both six-vertex foil passes"
            );
            assert!(glint_vertices
                .iter()
                .all(|vertex| vertex.color == GLINT_COLOR));
        }
    }

    #[test]
    fn container_slot_cursor_and_returning_z_levels_match_java_and_reverse_order_bites() {
        let slot_depth = gui_item_depth_at_z_level(0.0, GUI_SLOT_ITEM_RENDER_Z_LEVEL);
        let cursor_depth = gui_item_depth_at_z_level(0.0, GUI_CURSOR_ITEM_RENDER_Z_LEVEL);
        assert_eq!(slot_depth, 0.375);
        assert_eq!(cursor_depth, 0.325);
        assert!(
            cursor_depth < slot_depth,
            "the carried item is in front of slots"
        );
        assert_eq!(
            cursor_depth,
            gui_item_depth_at_z_level(0.0, GUI_CURSOR_ITEM_RENDER_Z_LEVEL),
            "returningStack shares GuiContainer.drawItemStack's zLevel=200"
        );

        // A flat/map quad inherits the scoped call depth even when the nearer cursor is emitted
        // first. If the override is ignored, the two streams collapse to the old `.4` center and
        // this biting control loses the Java separation.
        let mut reversed = DrawList::new();
        reversed.with_item_depth(cursor_depth, |draw| {
            draw.with_render_state(UiDepth::Item, UiCull::Back, |draw| {
                draw.quad_rect(
                    &UiTexture::Solid,
                    (4.0, 4.0, 12.0, 12.0),
                    (0.0, 0.0, 1.0, 1.0),
                    0xFFFF_FFFF,
                );
            });
        });
        reversed.with_item_depth(slot_depth, |draw| {
            draw.with_render_state(UiDepth::Item, UiCull::Back, |draw| {
                draw.quad_rect(
                    &UiTexture::Solid,
                    (4.0, 4.0, 12.0, 12.0),
                    (0.0, 0.0, 1.0, 1.0),
                    0xFFFF_FFFF,
                );
            });
        });
        assert_eq!(&reversed.depths()[..6], &[cursor_depth; 6]);
        assert_eq!(&reversed.depths()[6..], &[slot_depth; 6]);
        assert_eq!(reversed.batches()[0].depth, UiDepth::Item);
        assert_eq!(reversed.batches()[0].cull, UiCull::Back);
    }

    /// A model with one axis-aligned quad spanning the full unit square at `y = 0.5`,
    /// UV `0..1` — exactly what a `builtin/generated` flat icon's front layer looks like.
    fn single_flat_quad_model() -> BakedModel {
        use mc_assets::{BakedQuad, Vertex};
        let mut m = BakedModel::default();
        m.general_quads.push(BakedQuad {
            verts: [
                Vertex {
                    pos: [0.0, 0.0, 0.5],
                    uv: [0.0, 1.0],
                },
                Vertex {
                    pos: [0.0, 1.0, 0.5],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 1.0, 0.5],
                    uv: [1.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 0.0, 0.5],
                    uv: [1.0, 1.0],
                },
            ],
            face: None,
            shade_face: None,
            tint_index: None,
            shade: false,
        });
        m
    }

    fn two_layer_flat_model() -> BakedModel {
        let mut model = single_flat_quad_model();
        let mut liquid = model.general_quads.pop().expect("flat quad");
        liquid.tint_index = Some(0);
        let mut bottle = liquid.clone();
        bottle.tint_index = Some(1);
        model.general_quads.extend([liquid, bottle]);
        model
    }

    /// A full unit cube with one quad per face, each with a distinct UV rect so the three iso
    /// faces are told apart in tests.
    fn full_cube_model() -> BakedModel {
        use mc_assets::{BakedQuad, Vertex};
        let mut m = BakedModel::default();
        let all_faces = [
            Facing::Down,
            Facing::Up,
            Facing::North,
            Facing::South,
            Facing::West,
            Facing::East,
        ];
        for face in all_faces {
            let uv = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
            m.face_quads[face.index() as usize].push(BakedQuad {
                verts: [
                    Vertex {
                        pos: [0.0, 0.0, 0.0],
                        uv: uv[0],
                    },
                    Vertex {
                        pos: [0.0, 1.0, 0.0],
                        uv: uv[1],
                    },
                    Vertex {
                        pos: [1.0, 1.0, 1.0],
                        uv: uv[2],
                    },
                    Vertex {
                        pos: [1.0, 0.0, 1.0],
                        uv: uv[3],
                    },
                ],
                face: Some(face),
                shade_face: Some(face),
                tint_index: None,
                shade: true,
            });
        }
        m
    }
}
