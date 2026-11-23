//! S23: `install` publishes item models as well as block models.
//!
//! `S23 → S09`: "one additive constructor … **plus item-model install from outside
//! `bake`** — so `mc_assets::cache::BakedSnapshot` can be `install`ed."
//!
//! # Why this needs its own test binary
//!
//! [`mc_assets::bake_all`] publishes the item models it just baked on its way out, first-wins.
//! Any test that calls `bake_all` therefore has a populated item registry *before* it installs
//! anything, and an assertion about `item_model_for` after `install` would pass whether or not
//! `install` published anything at all. The only way to make the claim testable is a process that
//! never calls `bake_all` — hence a separate binary with exactly one `install` in it.
//!
//! Nothing here touches the real jar, so it runs everywhere.

use mc_assets::model::{BakedAssets, BakedModel, Vertex};
use mc_assets::sprite::{Sprite, SpriteFrame, UvRect};
use mc_assets::{Atlas, BakedQuad, ResourceLocation};
use mc_data::ItemId;

/// A model with recognisable geometry, so "the right model came back" is checkable rather than
/// merely "a model came back".
fn model(seed: f32) -> BakedModel {
    let mut verts = [Vertex::default(); 4];
    for (i, v) in verts.iter_mut().enumerate() {
        v.pos = [seed + i as f32, seed, seed * 2.0];
        v.uv = [seed / 16.0, i as f32 / 4.0];
    }
    BakedModel {
        general_quads: vec![BakedQuad {
            verts,
            face: Some(mc_math::Facing::Up),
            shade_face: Some(mc_math::Facing::Up),
            tint_index: None,
            shade: true,
        }],
        ambient_occlusion: true,
        ..BakedModel::default()
    }
}

const STONE_ITEM: u16 = 1;
const DIAMOND_SWORD: u16 = 276;
const NOT_INSTALLED: u16 = 4095;

#[test]
fn install_publishes_block_models_sprites_and_item_models_together() {
    // Before: nothing at all. This is what makes every assertion below non-vacuous.
    assert_eq!(mc_assets::model::installed_generation(), 0);
    assert!(
        mc_assets::item_model_for(ItemId(STONE_ITEM)).is_empty(),
        "control: the item registry must start empty"
    );
    assert!(mc_assets::model_for(mc_data::BlockState { id: 1 << 4 }).is_empty());

    let sprite = Sprite {
        location: ResourceLocation::minecraft("textures/blocks/stone"),
        width: 16,
        height: 16,
        uv: UvRect {
            min_u: 0.25,
            max_u: 0.5,
            min_v: 0.0,
            max_v: 0.125,
        },
        frames: vec![SpriteFrame {
            rgba: vec![7u8; 16 * 16 * 4],
            duration_ticks: None,
        }],
        default_duration_ticks: 1,
        interpolate: false,
        texture_metadata: Default::default(),
    };

    let mut models = vec![BakedModel::default(); mc_data::STATE_COUNT];
    models[1 << 4] = model(3.0);

    let assets = BakedAssets::from_parts(
        Atlas {
            texture: vec![0u8; 4],
            mipmaps: Vec::new(),
            w: 1,
            h: 1,
        },
        vec![sprite.clone()],
        models,
        vec![(2 << 4, vec![(5, model(9.0)), (1, model(11.0))])],
        vec![(STONE_ITEM, model(4.0)), (DIAMOND_SWORD, model(5.0))],
    );
    mc_assets::install(assets);
    assert_eq!(mc_assets::model::installed_generation(), 1);

    // Block models.
    assert!(mc_assets::cache::model_eq(
        mc_assets::model_for(mc_data::BlockState { id: 1 << 4 }),
        &model(3.0)
    ));

    // Sprites.
    assert_eq!(mc_assets::sprite(&sprite.location), &sprite);

    // Weighted variants.
    let weighted = mc_assets::model_for_at(mc_data::BlockState { id: 2 << 4 }, [1, 64, 2]);
    assert!(
        mc_assets::cache::model_eq(weighted, &model(9.0))
            || mc_assets::cache::model_eq(weighted, &model(11.0))
    );

    // **Item models** — the thing this binary exists for. Before S23 `install` never touched
    // them, so these two would have come back empty.
    assert!(mc_assets::cache::model_eq(
        mc_assets::item_model_for(ItemId(STONE_ITEM)),
        &model(4.0)
    ));
    assert!(mc_assets::cache::model_eq(
        mc_assets::item_model_for(ItemId(DIAMOND_SWORD)),
        &model(5.0)
    ));

    // Java's ModelManager returns its shared builtin/missing cube for an id the bake had no
    // model for; it must not silently disappear or alias somebody else's geometry.
    let missing = mc_assets::item_model_for(ItemId(NOT_INSTALLED));
    assert_eq!(
        missing
            .face_quads
            .iter()
            .map(std::vec::Vec::len)
            .sum::<usize>(),
        6
    );
    assert!(missing.general_quads.is_empty());
    assert!(!mc_assets::cache::model_eq(missing, &model(4.0)));
}

#[test]
fn from_parts_is_total_for_a_model_vector_of_the_wrong_length() {
    // A cache written against a different block registry can never be *read* (`STATE_COUNT` is in
    // the key), but `from_parts` is public and must not be able to panic or index out of bounds.
    let short = BakedAssets::from_parts(
        Atlas {
            texture: vec![0u8; 4],
            mipmaps: Vec::new(),
            w: 1,
            h: 1,
        },
        Vec::new(),
        vec![model(1.0)],
        Vec::new(),
        Vec::new(),
    );
    let long = BakedAssets::from_parts(
        Atlas {
            texture: vec![0u8; 4],
            mipmaps: Vec::new(),
            w: 1,
            h: 1,
        },
        Vec::new(),
        vec![BakedModel::default(); mc_data::STATE_COUNT + 64],
        Vec::new(),
        Vec::new(),
    );
    for assets in [&short, &long] {
        let (_, _, models, _, _) = assets.to_parts();
        assert_eq!(models.len(), mc_data::STATE_COUNT);
    }
}
