//! Models drawn by the tile-entity renderers, plus the two head models.
//!
//! java: `ModelChest.java`, `ModelLargeChest.java`, `ModelSign.java`, `ModelBanner.java`,
//! `ModelBook.java`, `ModelSkeletonHead.java`, `ModelHumanoidHead.java`.
//!
//! None of these is an `EntityLivingBase` model, and only two of them override
//! `setRotationAngles` at all. Vanilla drives them by reaching into the public `ModelRenderer`
//! fields from the renderer — `TileEntityChestRenderer` writes `chestLid.rotateAngleX`,
//! `TileEntityBannerRenderer` writes `bannerSlate.rotateAngleX`, `TileEntitySignRenderer` writes
//! `signBoard.rotateAngleX` — so every part is exposed as a `pub` [`PartId`] for
//! [`crate::entity::tile`] to pose.
//!
//! # Texture sizes
//!
//! Four of these draw against something other than the `ModelBase` 64x32 default, and each does
//! it a different way. `ModelChest` leaves `textureWidth/Height` alone and calls
//! `setTextureSize(64, 64)` on every *part*; `ModelLargeChest` does the same with `(128, 64)`;
//! `ModelBanner` assigns `textureWidth = textureHeight = 64` on the model. `ModelSign` and
//! `ModelBook` really are 64x32. All four are reproduced in the form the Java uses, because
//! [`PartBuilder::tex_size`](crate::entity::model::PartBuilder::tex_size) is per-part just as
//! `ModelRenderer.setTextureSize` is.

use glam::Mat4;
use mc_math::sin;

use crate::entity::anim::{model_radians, AnimParams, EntityModel};
use crate::entity::model::{Model, PartId};
use crate::entity::EntityVertex;

use core::f32::consts::PI;

// ---------------------------------------------------------------------------------------------
// ModelChest / ModelLargeChest
// ---------------------------------------------------------------------------------------------

/// The single chest.
///
/// java: `ModelChest`
#[derive(Clone, Debug)]
pub struct ChestModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelChest.chestLid` — `TileEntityChestRenderer` writes `rotateAngleX` here.
    pub chest_lid: PartId,
    /// java: `ModelChest.chestKnob` — follows the lid's X angle.
    pub chest_knob: PartId,
    /// java: `ModelChest.chestBelow`
    pub chest_below: PartId,
}

impl ChestModel {
    /// java: `ModelChest()` — three parts, each `setTextureSize(64, 64)`.
    ///
    /// `chestLid` is a field initialiser, so it is created before the two parts the constructor
    /// body makes; that order is also the `renderAll` draw order.
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        let chest_lid = m
            .part("chest_lid", 0, 0)
            .tex_size(64, 64)
            .cube(0.0, -5.0, -14.0, 14, 5, 14, 0.0)
            .at(1.0, 7.0, 15.0)
            .id();
        let chest_knob = m
            .part("chest_knob", 0, 0)
            .tex_size(64, 64)
            .cube(-1.0, -2.0, -15.0, 2, 4, 1, 0.0)
            .at(8.0, 7.0, 15.0)
            .id();
        let chest_below = m
            .part("chest_below", 0, 19)
            .tex_size(64, 64)
            .cube(0.0, 0.0, 0.0, 14, 10, 14, 0.0)
            .at(1.0, 6.0, 1.0)
            .id();
        Self {
            model: m,
            chest_lid,
            chest_knob,
            chest_below,
        }
    }

    /// java: `ModelChest.renderAll` line 1 — `chestKnob.rotateAngleX = chestLid.rotateAngleX`.
    pub fn sync_knob_to_lid(&mut self) {
        let a = self.model.get(self.chest_lid).rotate_angle[0];
        self.model.get_mut(self.chest_knob).rotate_angle[0] = a;
    }
}

impl Default for ChestModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for ChestModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    /// `ModelChest` has no `setRotationAngles`; the one statement it runs before drawing is the
    /// knob-follows-lid coupling at the top of `renderAll`, which lands here because
    /// [`EntityModel::pose_and_render`] calls this immediately before `render`.
    fn set_rotation_angles(&mut self, _p: &AnimParams) {
        self.sync_knob_to_lid();
    }
}

/// The double chest: the same three parts, 30 wide instead of 14, on a 128-wide texture.
///
/// java: `ModelLargeChest`
#[derive(Clone, Debug)]
pub struct LargeChestModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelChest.chestLid`, replaced by `ModelLargeChest`'s constructor.
    pub chest_lid: PartId,
    /// java: `ModelChest.chestKnob`, replaced — and moved to `x = 16`, the double chest's centre.
    pub chest_knob: PartId,
    /// java: `ModelChest.chestBelow`, replaced.
    pub chest_below: PartId,
}

impl LargeChestModel {
    /// java: `ModelLargeChest()`.
    ///
    /// The Java runs `ModelChest`'s constructor first and then overwrites all three fields, so
    /// the single-chest parts are built and immediately orphaned; only the three replacements are
    /// ever drawn, and only those are built here.
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        let chest_lid = m
            .part("chest_lid", 0, 0)
            .tex_size(128, 64)
            .cube(0.0, -5.0, -14.0, 30, 5, 14, 0.0)
            .at(1.0, 7.0, 15.0)
            .id();
        let chest_knob = m
            .part("chest_knob", 0, 0)
            .tex_size(128, 64)
            .cube(-1.0, -2.0, -15.0, 2, 4, 1, 0.0)
            .at(16.0, 7.0, 15.0)
            .id();
        let chest_below = m
            .part("chest_below", 0, 19)
            .tex_size(128, 64)
            .cube(0.0, 0.0, 0.0, 30, 10, 14, 0.0)
            .at(1.0, 6.0, 1.0)
            .id();
        Self {
            model: m,
            chest_lid,
            chest_knob,
            chest_below,
        }
    }

    /// java: `ModelChest.renderAll` line 1, inherited unchanged.
    pub fn sync_knob_to_lid(&mut self) {
        let a = self.model.get(self.chest_lid).rotate_angle[0];
        self.model.get_mut(self.chest_knob).rotate_angle[0] = a;
    }
}

impl Default for LargeChestModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for LargeChestModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    fn set_rotation_angles(&mut self, _p: &AnimParams) {
        self.sync_knob_to_lid();
    }
}

// ---------------------------------------------------------------------------------------------
// ModelSign
// ---------------------------------------------------------------------------------------------

/// The standing/wall sign.
///
/// java: `ModelSign`
#[derive(Clone, Debug)]
pub struct SignModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelSign.signBoard` — the written face; `TileEntitySignRenderer` rotates it.
    pub sign_board: PartId,
    /// java: `ModelSign.signStick` — hidden by the renderer for wall signs.
    pub sign_stick: PartId,
}

impl SignModel {
    /// java: `ModelSign()` — two parts on the 64x32 `ModelBase` default.
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        let sign_board = m
            .part("sign_board", 0, 0)
            .cube(-12.0, -14.0, -1.0, 24, 12, 2, 0.0)
            .id();
        let sign_stick = m
            .part("sign_stick", 0, 14)
            .cube(-1.0, -2.0, -1.0, 2, 14, 2, 0.0)
            .id();
        Self {
            model: m,
            sign_board,
            sign_stick,
        }
    }
}

impl Default for SignModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for SignModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    /// java: `ModelSign` has no `setRotationAngles` — `renderSign` just draws the two parts.
    fn set_rotation_angles(&mut self, _p: &AnimParams) {}
}

// ---------------------------------------------------------------------------------------------
// ModelBanner
// ---------------------------------------------------------------------------------------------

/// The banner.
///
/// java: `ModelBanner`
#[derive(Clone, Debug)]
pub struct BannerModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelBanner.bannerSlate` — the cloth; `TileEntityBannerRenderer` waves it by
    /// writing `rotateAngleX`.
    pub banner_slate: PartId,
    /// java: `ModelBanner.bannerStand` — hidden for a wall banner.
    pub banner_stand: PartId,
    /// java: `ModelBanner.bannerTop` — the crossbar.
    pub banner_top: PartId,
}

impl BannerModel {
    /// java: `ModelBanner()` — `textureWidth = textureHeight = 64` on the model itself, then
    /// three parts.
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 64);
        let banner_slate = m
            .part("banner_slate", 0, 0)
            .cube(-10.0, 0.0, -2.0, 20, 40, 1, 0.0)
            .id();
        let banner_stand = m
            .part("banner_stand", 44, 0)
            .cube(-1.0, -30.0, -1.0, 2, 42, 2, 0.0)
            .id();
        let banner_top = m
            .part("banner_top", 0, 42)
            .cube(-10.0, -32.0, -1.0, 20, 2, 2, 0.0)
            .id();
        Self {
            model: m,
            banner_slate,
            banner_stand,
            banner_top,
        }
    }
}

impl Default for BannerModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for BannerModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    /// java: `ModelBanner.renderBanner` line 1 — `bannerSlate.rotationPointY = -32.0F`.
    ///
    /// The Java re-asserts the pivot on every draw rather than in the constructor (the renderer
    /// only ever writes `rotateAngleX`), so it lives in the pre-draw hook here for the same
    /// reason `ModelChest`'s knob does.
    fn set_rotation_angles(&mut self, _p: &AnimParams) {
        let slate = self.banner_slate;
        self.model.get_mut(slate).rotation_point[1] = -32.0;
    }
}

// ---------------------------------------------------------------------------------------------
// ModelBook
// ---------------------------------------------------------------------------------------------

/// The enchanting-table book (also the lectern-style book on a bookshelf render).
///
/// java: `ModelBook`
#[derive(Clone, Debug)]
pub struct BookModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelBook.coverRight`
    pub cover_right: PartId,
    /// java: `ModelBook.coverLeft`
    pub cover_left: PartId,
    /// java: `ModelBook.pagesRight`
    pub pages_right: PartId,
    /// java: `ModelBook.pagesLeft`
    pub pages_left: PartId,
    /// java: `ModelBook.flippingPageRight`
    pub flipping_page_right: PartId,
    /// java: `ModelBook.flippingPageLeft`
    pub flipping_page_left: PartId,
    /// java: `ModelBook.bookSpine`
    pub book_spine: PartId,
}

impl BookModel {
    /// java: `ModelBook()` — seven parts, all built as field initialisers (which fixes their
    /// creation order) and then three tweaks in the constructor body.
    ///
    /// Every box goes through the six-arg `addBox` overload, whose `scaleFactor` is a hard `0`.
    /// The covers, the flipping pages and the spine are zero-*depth* boxes: single quads drawn
    /// from both sides.
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        let cover_right = m
            .part("cover_right", 0, 0)
            .cube(-6.0, -5.0, 0.0, 6, 10, 0, 0.0)
            .at(0.0, 0.0, -1.0)
            .id();
        let cover_left = m
            .part("cover_left", 16, 0)
            .cube(0.0, -5.0, 0.0, 6, 10, 0, 0.0)
            .at(0.0, 0.0, 1.0)
            .id();
        let pages_right = m
            .part("pages_right", 0, 10)
            .cube(0.0, -4.0, -0.99, 5, 8, 1, 0.0)
            .id();
        let pages_left = m
            .part("pages_left", 12, 10)
            .cube(0.0, -4.0, -0.01, 5, 8, 1, 0.0)
            .id();
        let flipping_page_right = m
            .part("flipping_page_right", 24, 10)
            .cube(0.0, -4.0, 0.0, 5, 8, 0, 0.0)
            .id();
        let flipping_page_left = m
            .part("flipping_page_left", 24, 10)
            .cube(0.0, -4.0, 0.0, 5, 8, 0, 0.0)
            .id();
        let book_spine = m
            .part("book_spine", 12, 0)
            .cube(-1.0, -5.0, 0.0, 2, 10, 0, 0.0)
            .angles(0.0, PI / 2.0, 0.0)
            .id();
        Self {
            model: m,
            cover_right,
            cover_left,
            pages_right,
            pages_left,
            flipping_page_right,
            flipping_page_left,
            book_spine,
        }
    }

    /// The `ModelBook.render` draw order, which is **not** the creation order — the spine is
    /// drawn third, between the covers and the pages.
    #[must_use]
    pub fn draw_order(&self) -> [PartId; 7] {
        [
            self.cover_right,
            self.cover_left,
            self.book_spine,
            self.pages_right,
            self.pages_left,
            self.flipping_page_right,
            self.flipping_page_left,
        ]
    }
}

impl Default for BookModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for BookModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// `ModelBook.setRotationAngles`, statement for statement.
    ///
    /// The argument names lie here: `TileEntityEnchantmentTableRenderer` passes the book's *open*
    /// factor in the `netHeadYaw` slot and the two page-flip phases in `limbSwingAmount` and
    /// `ageInTicks`, so none of the three is an angle in the usual sense.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let f = (sin(p.limb_swing * 0.02) * 0.1 + 1.25) * p.net_head_yaw;
        let m = &mut self.model;
        m.get_mut(self.cover_right).rotate_angle[1] = PI + f;
        m.get_mut(self.cover_left).rotate_angle[1] = -f;
        m.get_mut(self.pages_right).rotate_angle[1] = f;
        m.get_mut(self.pages_left).rotate_angle[1] = -f;
        m.get_mut(self.flipping_page_right).rotate_angle[1] = f - f * 2.0 * p.limb_swing_amount;
        m.get_mut(self.flipping_page_left).rotate_angle[1] = f - f * 2.0 * p.age_in_ticks;
        m.get_mut(self.pages_right).rotation_point[0] = sin(f);
        m.get_mut(self.pages_left).rotation_point[0] = sin(f);
        m.get_mut(self.flipping_page_right).rotation_point[0] = sin(f);
        m.get_mut(self.flipping_page_left).rotation_point[0] = sin(f);
    }

    /// java: `ModelBook.render` — seven parts in an order the creation order does not match.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.model.emit_some(&self.draw_order(), p.scale, base, out);
    }
}

// ---------------------------------------------------------------------------------------------
// ModelSkeletonHead / ModelHumanoidHead
// ---------------------------------------------------------------------------------------------

/// A mob head block: one 8x8x8 box.
///
/// java: `ModelSkeletonHead`
#[derive(Clone, Debug)]
pub struct SkeletonHeadModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelSkeletonHead.skeletonHead`
    pub skeleton_head: PartId,
}

impl SkeletonHeadModel {
    /// java: `ModelSkeletonHead(int texOffX, int texOffY, int textureWidth, int textureHeight)`.
    ///
    /// The Java no-arg constructor is `this(0, 35, 64, 64)`; use [`Self::new_wither_skull`] for
    /// that renderer-owned form. Other call sites pass their own four numbers
    /// (`ModelHumanoidHead` uses `(0, 0, 64, 64)`, `TileEntitySkullRenderer` builds several
    /// variants), so the general constructor remains available too.
    #[must_use]
    pub fn new(u: i32, v: i32, tw: i32, th: i32) -> Self {
        let mut m = Model::new(tw as u32, th as u32);
        let skeleton_head = m
            .part("skeleton_head", u, v)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, 0.0)
            .at(0.0, 0.0, 0.0)
            .id();
        Self {
            model: m,
            skeleton_head,
        }
    }

    /// The head model owned by `RenderWitherSkull`.
    ///
    /// java: `new ModelSkeletonHead()` → `this(0, 35, 64, 64)`. This named constructor keeps the
    /// projectile's 64×64 `wither.png` sheet separate from the tile renderer's 64×32 skeleton
    /// sheet instead of making callers repeat a load-bearing tuple of magic numbers.
    #[must_use]
    pub fn new_wither_skull() -> Self {
        Self::new(0, 35, 64, 64)
    }
}

impl EntityModel for SkeletonHeadModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    /// java: `skeletonHead.rotateAngleY = netHeadYaw / (180F / PI)`, pitch likewise on X.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let id = self.skeleton_head;
        self.model.get_mut(id).rotate_angle[1] = model_radians(p.net_head_yaw);
        self.model.get_mut(id).rotate_angle[0] = model_radians(p.head_pitch);
    }
}

/// A player head: a skull with the skin's hat layer over it.
///
/// java: `ModelHumanoidHead`
#[derive(Clone, Debug)]
pub struct HumanoidHeadModel {
    /// The inherited skull. java: `super(0, 0, 64, 64)`
    pub skull: SkeletonHeadModel,
    /// java: `ModelHumanoidHead.head` — the hat layer, inflated by `0.25`, at (32, 0).
    pub head: PartId,
}

impl HumanoidHeadModel {
    /// java: `ModelHumanoidHead()`.
    ///
    /// `head` is a field initialiser, so in the Java it is constructed *before* `super(...)`
    /// finishes adding `skeletonHead` to `boxList` — but the draw order comes from
    /// `ModelHumanoidHead.render`, which is `super.render` (the skull) and then the hat.
    #[must_use]
    pub fn new() -> Self {
        let mut skull = SkeletonHeadModel::new(0, 0, 64, 64);
        let head = skull
            .model
            .part("head", 32, 0)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, 0.25)
            .at(0.0, 0.0, 0.0)
            .id();
        Self { skull, head }
    }
}

impl Default for HumanoidHeadModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for HumanoidHeadModel {
    fn model(&self) -> &Model {
        &self.skull.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.skull.model
    }
    /// java: `super.setRotationAngles`, then `head.rotateAngleY/X = skeletonHead.rotateAngleY/X`.
    ///
    /// Note this is not `copyModelAngles`: the Z angle and the pivot are left alone.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.skull.set_rotation_angles(p);
        let a = self.skull.model.get(self.skull.skeleton_head).rotate_angle;
        let head = self.skull.model.get_mut(self.head);
        head.rotate_angle[1] = a[1];
        head.rotate_angle[0] = a[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every UV a model emits must land inside its own texture — the assertion that catches a
    /// missed `setTextureSize`.
    fn assert_uvs_in_range(m: &mut dyn EntityModel, what: &str) {
        let mut out = Vec::new();
        m.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
        assert!(!out.is_empty(), "{what} emitted nothing");
        for v in &out {
            assert!(
                (-0.001..=1.001).contains(&v.uv[0]) && (-0.001..=1.001).contains(&v.uv[1]),
                "{what} uv out of range: {:?}",
                v.uv
            );
        }
    }

    /// java: `chestLid.addBox(0, -5, -14, 14, 5, 14, 0)` with the pivot at `(1, 7, 15)`.
    #[test]
    fn chest_parts_match_the_java_boxes_and_pivots() {
        let c = ChestModel::new();
        let lid = c.model.get(c.chest_lid);
        assert_eq!(lid.texture_offset, [0, 0]);
        assert_eq!(lid.cubes[0].pos1, [0.0, -5.0, -14.0]);
        assert_eq!(lid.cubes[0].pos2, [14.0, 0.0, 0.0]);
        assert_eq!(lid.rotation_point, [1.0, 7.0, 15.0]);

        let knob = c.model.get(c.chest_knob);
        assert_eq!(knob.cubes[0].pos1, [-1.0, -2.0, -15.0]);
        assert_eq!(knob.cubes[0].pos2, [1.0, 2.0, -14.0]);
        assert_eq!(knob.rotation_point, [8.0, 7.0, 15.0]);

        let below = c.model.get(c.chest_below);
        assert_eq!(below.texture_offset, [0, 19]);
        assert_eq!(below.cubes[0].pos1, [0.0, 0.0, 0.0]);
        assert_eq!(below.cubes[0].pos2, [14.0, 10.0, 14.0]);
        assert_eq!(below.rotation_point, [1.0, 6.0, 1.0]);
    }

    /// java: `ModelLargeChest` widens every box from 14 to 30 and moves the knob from x = 8 to 16.
    #[test]
    fn large_chest_is_thirty_wide_where_the_single_chest_is_fourteen() {
        let small = ChestModel::new();
        let large = LargeChestModel::new();

        let s_lid = &small.model.get(small.chest_lid).cubes[0];
        let l_lid = &large.model.get(large.chest_lid).cubes[0];
        assert_eq!(s_lid.pos2[0] - s_lid.pos1[0], 14.0);
        assert_eq!(l_lid.pos2[0] - l_lid.pos1[0], 30.0);

        let s_below = &small.model.get(small.chest_below).cubes[0];
        let l_below = &large.model.get(large.chest_below).cubes[0];
        assert_eq!(s_below.pos2, [14.0, 10.0, 14.0]);
        assert_eq!(l_below.pos2, [30.0, 10.0, 14.0]);

        // Same knob box, different pivot: the double chest's centre is at x = 16.
        assert_eq!(
            small.model.get(small.chest_knob).cubes[0].pos1,
            large.model.get(large.chest_knob).cubes[0].pos1
        );
        assert_eq!(small.model.get(small.chest_knob).rotation_point[0], 8.0);
        assert_eq!(large.model.get(large.chest_knob).rotation_point[0], 16.0);
    }

    /// java: `(new ModelRenderer(this, 0, 0)).setTextureSize(64, 64)` — and `(128, 64)` for the
    /// double chest. The `ModelBase` texture size stays at its 64x32 default in both.
    #[test]
    fn chest_parts_carry_their_own_texture_size() {
        let small = ChestModel::new();
        for id in [small.chest_lid, small.chest_knob, small.chest_below] {
            let p = small.model.get(id);
            assert_eq!((p.texture_width, p.texture_height), (64.0, 64.0));
        }
        assert_eq!(
            small.model.texture_height, 32,
            "ModelBase's default is untouched"
        );

        let large = LargeChestModel::new();
        for id in [large.chest_lid, large.chest_knob, large.chest_below] {
            let p = large.model.get(id);
            assert_eq!((p.texture_width, p.texture_height), (128.0, 64.0));
        }
        // The lid's -X face: u = 0..14 of 128, v = 14..19 of 64.
        let west = large.model.get(large.chest_lid).cubes[0].quads[1];
        assert_eq!(west.verts[1].uv, [0.0, 14.0 / 64.0]);
        assert_eq!(west.verts[3].uv, [14.0 / 128.0, 19.0 / 64.0]);
    }

    /// java: `renderAll` — `chestKnob.rotateAngleX = chestLid.rotateAngleX`.
    #[test]
    fn chest_knob_follows_the_lid() {
        let mut c = ChestModel::new();
        c.model.get_mut(c.chest_lid).rotate_angle[0] = -1.25;
        c.set_rotation_angles(&AnimParams::still());
        assert_eq!(c.model.get(c.chest_knob).rotate_angle[0], -1.25);
        // Only X is copied.
        assert_eq!(c.model.get(c.chest_knob).rotate_angle[1], 0.0);

        let mut l = LargeChestModel::new();
        l.model.get_mut(l.chest_lid).rotate_angle[0] = 0.5;
        l.set_rotation_angles(&AnimParams::still());
        assert_eq!(l.model.get(l.chest_knob).rotate_angle[0], 0.5);
    }

    /// java: `signBoard.addBox(-12, -14, -1, 24, 12, 2, 0)` and
    /// `signStick.addBox(-1, -2, -1, 2, 14, 2, 0)` at (0, 14), both on 64x32.
    #[test]
    fn sign_board_and_stick_match_the_java() {
        let s = SignModel::new();
        let board = s.model.get(s.sign_board);
        assert_eq!(board.texture_offset, [0, 0]);
        assert_eq!(board.cubes[0].pos1, [-12.0, -14.0, -1.0]);
        assert_eq!(board.cubes[0].pos2, [12.0, -2.0, 1.0]);
        assert_eq!(board.rotation_point, [0.0, 0.0, 0.0]);

        let stick = s.model.get(s.sign_stick);
        assert_eq!(stick.texture_offset, [0, 14]);
        assert_eq!(stick.cubes[0].pos1, [-1.0, -2.0, -1.0]);
        assert_eq!(stick.cubes[0].pos2, [1.0, 12.0, 1.0]);
        assert_eq!((stick.texture_width, stick.texture_height), (64.0, 32.0));
    }

    /// java: `textureWidth = textureHeight = 64` and three parts; `renderBanner` then pins the
    /// slate's pivot to `y = -32`.
    #[test]
    fn banner_parts_match_the_java_and_the_slate_pivot_is_set_at_draw_time() {
        let mut b = BannerModel::new();
        assert_eq!((b.model.texture_width, b.model.texture_height), (64, 64));

        let slate = b.model.get(b.banner_slate);
        assert_eq!(slate.cubes[0].pos1, [-10.0, 0.0, -2.0]);
        assert_eq!(slate.cubes[0].pos2, [10.0, 40.0, -1.0]);
        assert_eq!(
            slate.rotation_point,
            [0.0, 0.0, 0.0],
            "zero until renderBanner runs"
        );

        let stand = b.model.get(b.banner_stand);
        assert_eq!(stand.texture_offset, [44, 0]);
        assert_eq!(stand.cubes[0].pos1, [-1.0, -30.0, -1.0]);
        assert_eq!(stand.cubes[0].pos2, [1.0, 12.0, 1.0]);

        let top = b.model.get(b.banner_top);
        assert_eq!(top.texture_offset, [0, 42]);
        assert_eq!(top.cubes[0].pos1, [-10.0, -32.0, -1.0]);
        assert_eq!(top.cubes[0].pos2, [10.0, -30.0, 1.0]);

        b.set_rotation_angles(&AnimParams::still());
        assert_eq!(
            b.model.get(b.banner_slate).rotation_point,
            [0.0, -32.0, 0.0]
        );
    }

    /// java: the seven field initialisers, plus the three statements in the constructor body.
    #[test]
    fn book_parts_match_the_java() {
        let b = BookModel::new();
        let m = &b.model;

        let cr = m.get(b.cover_right);
        assert_eq!(cr.texture_offset, [0, 0]);
        assert_eq!(cr.cubes[0].pos1, [-6.0, -5.0, 0.0]);
        assert_eq!(cr.cubes[0].pos2, [0.0, 5.0, 0.0], "a zero-depth quad");
        assert_eq!(cr.rotation_point, [0.0, 0.0, -1.0]);

        let cl = m.get(b.cover_left);
        assert_eq!(cl.texture_offset, [16, 0]);
        assert_eq!(cl.rotation_point, [0.0, 0.0, 1.0]);

        assert_eq!(m.get(b.pages_right).texture_offset, [0, 10]);
        assert_eq!(m.get(b.pages_right).cubes[0].pos1, [0.0, -4.0, -0.99]);
        assert_eq!(m.get(b.pages_left).texture_offset, [12, 10]);
        assert_eq!(m.get(b.pages_left).cubes[0].pos1, [0.0, -4.0, -0.01]);

        // Both flipping pages share one texture offset and one box.
        assert_eq!(m.get(b.flipping_page_right).texture_offset, [24, 10]);
        assert_eq!(m.get(b.flipping_page_left).texture_offset, [24, 10]);
        assert_eq!(
            m.get(b.flipping_page_right).cubes[0].pos2,
            m.get(b.flipping_page_left).cubes[0].pos2
        );

        let spine = m.get(b.book_spine);
        assert_eq!(spine.texture_offset, [12, 0]);
        assert_eq!(spine.cubes[0].pos1, [-1.0, -5.0, 0.0]);
        assert_eq!(spine.rotate_angle, [0.0, PI / 2.0, 0.0]);
    }

    /// java: `f = (sin(limbSwing * 0.02) * 0.1 + 1.25) * netHeadYaw`, then ten assignments.
    #[test]
    fn book_pose_is_driven_by_the_open_factor() {
        let mut b = BookModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 0.0; // sin(0) = 0, so f = 1.25 * netHeadYaw
        p.net_head_yaw = 1.0;
        p.limb_swing_amount = 0.25;
        p.age_in_ticks = 0.75;
        b.set_rotation_angles(&p);

        let f = 1.25_f32;
        let m = &b.model;
        assert!((m.get(b.cover_right).rotate_angle[1] - (PI + f)).abs() < 1e-6);
        assert!((m.get(b.cover_left).rotate_angle[1] + f).abs() < 1e-6);
        assert!((m.get(b.pages_right).rotate_angle[1] - f).abs() < 1e-6);
        assert!((m.get(b.pages_left).rotate_angle[1] + f).abs() < 1e-6);
        assert!((m.get(b.flipping_page_right).rotate_angle[1] - (f - f * 2.0 * 0.25)).abs() < 1e-6);
        assert!((m.get(b.flipping_page_left).rotate_angle[1] - (f - f * 2.0 * 0.75)).abs() < 1e-6);
        // All four page pivots slide along X by sin(f) — vanilla's LUT sine, not libm's.
        for id in [
            b.pages_right,
            b.pages_left,
            b.flipping_page_right,
            b.flipping_page_left,
        ] {
            assert_eq!(m.get(id).rotation_point[0], sin(f));
        }
        // The covers and the spine keep their constructor pivots.
        assert_eq!(m.get(b.cover_right).rotation_point, [0.0, 0.0, -1.0]);
    }

    /// java: `ModelBook.render` draws the spine third, though it is created last.
    #[test]
    fn book_draws_the_spine_between_the_covers_and_the_pages() {
        let b = BookModel::new();
        assert_eq!(b.draw_order()[2], b.book_spine);
        assert_eq!(b.model.roots()[6], b.book_spine, "but it is created last");

        let p = AnimParams::still();
        let mut got = Vec::new();
        b.render(&p, Mat4::IDENTITY, &mut got);

        let mut want = Vec::new();
        for id in b.draw_order() {
            b.model.emit_part(id, p.scale, Mat4::IDENTITY, &mut want);
        }
        assert_eq!(got, want);

        // ...and that really is a different order from the default root walk.
        let mut roots = Vec::new();
        b.model.emit_all(p.scale, Mat4::IDENTITY, &mut roots);
        assert_eq!(roots.len(), got.len());
        assert_ne!(roots, got);
    }

    /// java: `ModelSkeletonHead(int, int, int, int)` honours all four arguments; our factory
    /// calls it as `(0, 0, 64, 32)`.
    #[test]
    fn skeleton_head_honours_all_four_constructor_arguments() {
        let h = SkeletonHeadModel::new(0, 0, 64, 32);
        assert_eq!((h.model.texture_width, h.model.texture_height), (64, 32));
        let p = h.model.get(h.skeleton_head);
        assert_eq!(p.texture_offset, [0, 0]);
        assert_eq!(p.cubes[0].pos1, [-4.0, -8.0, -4.0]);
        assert_eq!(p.cubes[0].pos2, [4.0, 0.0, 4.0]);
        assert_eq!(p.rotation_point, [0.0, 0.0, 0.0]);
        // -Z face: u = 8..16 of 64, v = 8..16 of 32.
        assert_eq!(p.cubes[0].quads[4].verts[1].uv, [8.0 / 64.0, 8.0 / 32.0]);

        // The Java no-arg form is `(0, 35, 64, 64)`; the offsets and sizes must both move.
        let vanilla = SkeletonHeadModel::new(0, 35, 64, 64);
        assert_eq!(
            vanilla.model.get(vanilla.skeleton_head).texture_offset,
            [0, 35]
        );
        assert_eq!(
            vanilla.model.get(vanilla.skeleton_head).cubes[0].quads[4].verts[1].uv,
            [8.0 / 64.0, 43.0 / 64.0]
        );
    }

    /// java: `skeletonHead.rotateAngleY = netHeadYaw / (180F / PI)`.
    #[test]
    fn heads_follow_the_yaw_and_pitch() {
        let mut h = SkeletonHeadModel::new(0, 0, 64, 32);
        let mut p = AnimParams::still();
        p.net_head_yaw = 45.0;
        p.head_pitch = 20.0;
        h.set_rotation_angles(&p);
        let a = h.model.get(h.skeleton_head).rotate_angle;
        assert_eq!(a, [model_radians(20.0), model_radians(45.0), 0.0]);

        let mut hh = HumanoidHeadModel::new();
        hh.set_rotation_angles(&p);
        let skull = hh.skull.model.get(hh.skull.skeleton_head).rotate_angle;
        let hat = hh.skull.model.get(hh.head).rotate_angle;
        assert_eq!(hat[0], skull[0]);
        assert_eq!(hat[1], skull[1]);
    }

    /// java: `ModelHumanoidHead` is `super(0, 0, 64, 64)` plus an 8x8x8 box inflated by 0.25 at
    /// texture (32, 0).
    #[test]
    fn humanoid_head_adds_an_inflated_hat_layer() {
        let h = HumanoidHeadModel::new();
        assert_eq!(
            (h.skull.model.texture_width, h.skull.model.texture_height),
            (64, 64)
        );
        assert_eq!(h.skull.model.roots().len(), 2);
        let hat = h.skull.model.get(h.head);
        assert_eq!(hat.texture_offset, [32, 0]);
        // pos1/pos2 record the box before inflation...
        assert_eq!(hat.cubes[0].pos1, [-4.0, -8.0, -4.0]);
        assert_eq!(hat.cubes[0].pos2, [4.0, 0.0, 4.0]);
        // ...while the baked vertices are 0.25 outside it on every axis.
        assert_eq!(hat.cubes[0].quads[4].verts[1].pos, [-4.25, -8.25, -4.25]);
        assert_eq!(hat.cubes[0].quads[4].verts[3].pos, [4.25, 0.25, -4.25]);
        // The hat is drawn after the skull.
        assert_eq!(h.skull.model.roots()[1], h.head);
    }

    #[test]
    fn every_tile_entity_models_uvs_land_inside_its_texture() {
        assert_uvs_in_range(&mut ChestModel::new(), "chest");
        assert_uvs_in_range(&mut LargeChestModel::new(), "large chest");
        assert_uvs_in_range(&mut SignModel::new(), "sign");
        assert_uvs_in_range(&mut BannerModel::new(), "banner");
        assert_uvs_in_range(&mut BookModel::new(), "book");
        assert_uvs_in_range(&mut SkeletonHeadModel::new(0, 0, 64, 32), "skeleton head");
        assert_uvs_in_range(
            &mut SkeletonHeadModel::new(0, 35, 64, 64),
            "skull (vanilla ctor)",
        );
        assert_uvs_in_range(&mut HumanoidHeadModel::new(), "humanoid head");
    }
}
