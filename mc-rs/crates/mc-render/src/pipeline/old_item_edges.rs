//! 1.7 Animations' "1.7 item edges" (option id `translucent_held`): the held item drawn the way 1.7 drew it, for older resource packs.
//!
//! # What 1.7 did, and what 1.8.9 does instead
//!
//! * 1.7 drew a flat held item with `ItemRenderer.renderItemIn2D`: the front and back faces plus
//!   one side strip for every texel column and row, under the `GL_GREATER 0.1` alpha test and
//!   with **blending off** (`EntityRenderer.renderWorld` disables `GL_BLEND` right before the
//!   `"hand"` section; `ItemRenderer.renderItem` turns it on only for render-pass-1 blocks, and
//!   `renderItemInFirstPerson` for `ItemCloth`, the stained glass). So every texel that passed the
//!   alpha test was a solid little box with all its sides.
//! * 1.8.9 draws `ItemModelGenerator`'s model: side faces only where a texel borders one with
//!   alpha exactly 0, drawn with the same 0.1 alpha test **and** blending on
//!   (`RenderItem.renderItemModelTransform`). A texel with alpha 1..=25 counts as solid for the
//!   sides but is discarded when drawn, so the solid texel beside it has no side face: a
//!   see-through gap along the item's edge. That is MC-70336, "Edges of Items transparent at
//!   pixels with alpha value > 0 and < 26" (resolved Invalid: "The resource packs need to be
//!   fixed"), which older packs' anti-aliased edges hit; its reporter: "it works in 1.7".
//!
//! # What this does
//!
//! With the option on, a flat held item whose sprite has any partly transparent texel (alpha
//! 1..=254) is re-baked with 1.7's emptiness test (alpha at most 25, the texels the alpha test
//! discards: [`mc_assets::model::OLD_ITEM_EMPTY_ALPHA`]) and drawn solid, without blending, as
//! 1.7 drew it. Every visible texel then has its sides, and no edge is see-through.
//!
//! An item whose texels are all fully opaque or fully clear (every vanilla item) is left exactly
//! as it is: same model, same pipeline, same pixels. So are GUI-3D models (blocks), `builtin/entity`
//! items, maps, and translucent blocks with a flat model (stained glass panes, which 1.7 blended
//! too).
//!
//! The re-bake runs once when the held model changes (it reads the sprite from the atlas bytes
//! the phase already keeps), never per frame.

use mc_assets::model::{bake_generated_layer_17, BakedModel, BakedQuad};
use mc_assets::sprite::UvRect;
use mc_math::Facing;

/// A texel rectangle in the atlas: `(x, y, width, height)`.
type TexelRect = (u32, u32, u32, u32);

/// The 1.7-edged copy of a flat held item, or `None` to keep the vanilla model and pipeline
/// (nothing partly transparent, or not a `builtin/generated` model, or the sprite is not where
/// its UVs say in `atlas`).
///
/// `atlas` is the item atlas's level-0 RGBA8 bytes, `width` x `height` texels.
#[must_use]
pub fn old_edged_model(
    model: &BakedModel,
    atlas: &[u8],
    width: u32,
    height: u32,
) -> Option<BakedModel> {
    if model.gui_3d || model.face_quads.iter().any(|quads| !quads.is_empty()) {
        return None;
    }
    if atlas.len() < (width as usize) * (height as usize) * 4 || width == 0 || height == 0 {
        return None;
    }
    let quads = &model.general_quads;
    // `ItemModelGenerator` emits, per layer: the north core quad, the south core quad, then the
    // edge quads (up/down/east/west only). So every north quad starts a layer.
    let starts: Vec<usize> = quads
        .iter()
        .enumerate()
        .filter(|(_, quad)| quad.face == Some(Facing::North))
        .map(|(index, _)| index)
        .collect();
    if starts.first() != Some(&0) {
        return None;
    }
    let mut changed = false;
    let mut out: Vec<BakedQuad> = Vec::with_capacity(quads.len());
    for (layer, &start) in starts.iter().enumerate() {
        let end = starts.get(layer + 1).copied().unwrap_or(quads.len());
        let layer_quads = &quads[start..end];
        let south = layer_quads
            .get(1)
            .filter(|q| q.face == Some(Facing::South))?;
        let (rect, uv) = sprite_rect(south, width, height)?;
        let rgba = copy_rect(atlas, width, rect);
        let partial = rgba
            .chunks_exact(4)
            .any(|texel| texel[3] != 0 && texel[3] != 255);
        if partial {
            let layer_index = south.tint_index.map_or(-1, i32::from);
            out.extend(bake_generated_layer_17(
                layer_index,
                &rgba,
                rect.2,
                rect.3,
                uv,
            ));
            changed = true;
        } else {
            out.extend_from_slice(layer_quads);
        }
    }
    if !changed {
        return None;
    }
    let mut edged = model.clone();
    edged.general_quads = out;
    Some(edged)
}

/// The sprite's texel rectangle `(x, y, w, h)` in the atlas and its UV rectangle, from a core
/// quad's corners.
fn sprite_rect(core: &BakedQuad, width: u32, height: u32) -> Option<(TexelRect, UvRect)> {
    let us = core.verts.iter().map(|v| v.uv[0]);
    let vs = core.verts.iter().map(|v| v.uv[1]);
    let uv = UvRect {
        min_u: us.clone().fold(f32::MAX, f32::min),
        max_u: us.fold(f32::MIN, f32::max),
        min_v: vs.clone().fold(f32::MAX, f32::min),
        max_v: vs.fold(f32::MIN, f32::max),
    };
    let texel = |value: f32, size: u32| (value * size as f32).round();
    let (x0, x1) = (texel(uv.min_u, width), texel(uv.max_u, width));
    let (y0, y1) = (texel(uv.min_v, height), texel(uv.max_v, height));
    let inside = x0 >= 0.0 && y0 >= 0.0 && x1 <= width as f32 && y1 <= height as f32;
    if !inside {
        return None;
    }
    let (w, h) = (x1 - x0, y1 - y0);
    if w < 1.0 || h < 1.0 {
        return None;
    }
    Some(((x0 as u32, y0 as u32, w as u32, h as u32), uv))
}

/// The RGBA8 texels of `rect` out of a `width`-texel-wide atlas.
fn copy_rect(atlas: &[u8], width: u32, (x, y, w, h): TexelRect) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in y..y + h {
        let at = ((row * width + x) * 4) as usize;
        out.extend_from_slice(&atlas[at..at + (w * 4) as usize]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-layer generated model for the sprite at texel rect `(x, y, 4, 4)` of a 16x16 atlas,
    /// baked by vanilla's rules (`bake_generated_layer_17` on 0/255 texels is exactly vanilla).
    fn model_for(atlas: &[u8], x: u32, y: u32) -> BakedModel {
        let uv = UvRect {
            min_u: x as f32 / 16.0,
            max_u: (x + 4) as f32 / 16.0,
            min_v: y as f32 / 16.0,
            max_v: (y + 4) as f32 / 16.0,
        };
        // Vanilla's edges: treat every texel with alpha > 0 as solid.
        let mut binary = copy_rect(atlas, 16, (x, y, 4, 4));
        for texel in binary.chunks_exact_mut(4) {
            texel[3] = if texel[3] > 0 { 255 } else { 0 };
        }
        BakedModel {
            general_quads: bake_generated_layer_17(0, &binary, 4, 4, uv),
            ambient_occlusion: false,
            gui_3d: false,
            particle_uv: Some(uv),
            ..BakedModel::default()
        }
    }

    fn atlas_with(texels: &[(u32, u32, u8)]) -> Vec<u8> {
        let mut atlas = vec![0u8; 16 * 16 * 4];
        for &(x, y, a) in texels {
            atlas[((y * 16 + x) * 4) as usize..][..4].copy_from_slice(&[90, 160, 220, a]);
        }
        atlas
    }

    #[test]
    fn an_all_opaque_or_clear_sprite_keeps_the_vanilla_model() {
        let atlas = atlas_with(&[(5, 5, 255), (6, 5, 255), (5, 6, 255)]);
        let model = model_for(&atlas, 4, 4);
        assert!(old_edged_model(&model, &atlas, 16, 16).is_none());
    }

    #[test]
    fn a_faint_texel_is_re_baked_with_its_neighbours_sides() {
        // A solid texel at (5, 5) with a faint one (alpha 10) right of it, in the sprite at (4, 4).
        let atlas = atlas_with(&[(5, 5, 255), (6, 5, 10)]);
        let model = model_for(&atlas, 4, 4);
        let edged = old_edged_model(&model, &atlas, 16, 16).expect("partial alpha re-bakes");
        // The faint texel no longer counts: the solid texel now has its right-hand side, and the
        // model is exactly what the 1.7 bake of that sprite gives.
        let sprite = copy_rect(&atlas, 16, (4, 4, 4, 4));
        let uv = model.particle_uv.unwrap();
        assert_eq!(
            edged.general_quads,
            bake_generated_layer_17(0, &sprite, 4, 4, uv)
        );
        assert_ne!(edged.general_quads, model.general_quads);
        // A semi-transparent (alpha 128) texel is solid in both; it still switches the item to
        // the solid pipeline, so the model is returned.
        let atlas = atlas_with(&[(5, 5, 255), (6, 5, 128)]);
        let model = model_for(&atlas, 4, 4);
        let edged = old_edged_model(&model, &atlas, 16, 16).expect("partial alpha");
        assert_eq!(edged.general_quads, model.general_quads);
    }

    #[test]
    fn blocks_and_odd_models_are_left_alone() {
        let atlas = atlas_with(&[(5, 5, 10)]);
        let mut model = model_for(&atlas, 4, 4);
        model.gui_3d = true;
        assert!(old_edged_model(&model, &atlas, 16, 16).is_none());
        let empty = BakedModel::default();
        assert!(old_edged_model(&empty, &atlas, 16, 16).is_none());
        // A short atlas is refused rather than read out of bounds.
        let model = model_for(&atlas, 4, 4);
        assert!(old_edged_model(&model, &atlas[..64], 16, 16).is_none());
    }
}
