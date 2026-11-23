//! Box geometry and pivot transforms — the foundation of every entity model.
//!
//! java: `client/model/ModelRenderer.java`, `ModelBox.java`, `TexturedQuad.java`,
//! `PositionTextureVertex.java`, `ModelBase.java`.
//!
//! # What vanilla actually does
//!
//! A vanilla model is a forest of [`ModelPart`]s (`ModelRenderer`). Each part owns zero or more
//! axis-aligned [`ModelBox`]es, expressed in *model units* — sixteenths of a block — and a pivot
//! (`rotationPoint`) about which its three Euler angles are applied. Boxes are baked once into
//! six [`TexturedQuad`]s with UVs derived from a single `(textureOffsetX, textureOffsetY)`
//! corner using the classic "unwrapped cross" layout; the part tree is then walked at draw time
//! with a matrix stack.
//!
//! `ModelRenderer.render` builds its local matrix as
//!
//! ```text
//! T(offset) · T(rotationPoint · scale) · Rz(θz) · Ry(θy) · Rx(θx)
//! ```
//!
//! and children are drawn *inside* that matrix — including inside the unscaled `offset`
//! translate, since vanilla only undoes it after the child loop. The three branches in the Java
//! (`angles == 0 && pivot == 0`, `angles == 0`, general) are pure fast paths: they hand-undo the
//! translates instead of push/popping, so all three are equivalent to the single composition
//! above. [`ModelPart::local_matrix`] is that composition. [`ModelPart::post_render_matrix`]
//! is the otherwise-identical `postRender` composition, which deliberately omits `offset`.
//!
//! Note `offset` is **not** multiplied by `scale` (java: `ModelRenderer.java:137`), unlike
//! `rotationPoint`. That asymmetry is real; do not "fix" it.
//!
//! # Coordinate conventions
//!
//! Model space is Y-down and Z-forward-ish: an entity is authored with its feet near `y = 24`
//! and its head near `y = 0`. The renderer flips it with a `scale(-1, -1, 1)` and lifts it by
//! `-1.5078125` blocks (java: `RendererLivingEntity.doRender`). [`crate::entity::render`] owns
//! that outer transform; everything in this module stays in raw model space.

use glam::{Mat4, Vec3};

/// One vertex of a baked quad, in model units, with its texture coordinate already normalised
/// into `0..1` against the model's texture size.
///
/// java: `PositionTextureVertex`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadVertex {
    /// Position in model units (sixteenths of a block), before the draw-time `scale`.
    pub pos: [f32; 3],
    /// Texture coordinate in `0..1`.
    pub uv: [f32; 2],
}

/// One of a box's six faces: four vertices wound for a `GL_QUADS`-style draw.
///
/// java: `TexturedQuad`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexturedQuad {
    /// The four corners, in vanilla's winding order.
    pub verts: [QuadVertex; 4],
}

impl TexturedQuad {
    /// Reverses the winding, as vanilla does for every face of a mirrored box.
    ///
    /// UVs travel with their vertex — vanilla flips *after* assigning texture coordinates, so a
    /// mirrored box shows the same texels seen from the other side.
    ///
    /// java: `TexturedQuad.flipFace`
    fn flip_face(&mut self) {
        self.verts.reverse();
    }

    /// The face normal vanilla computes at draw time: `normalize((v2 - v1) × (v0 - v1))`.
    ///
    /// java: `TexturedQuad.draw` — `subtractReverse(a)` returns `a - this`, and `Vec3.normalize`
    /// returns the zero vector below a `1e-4` length, which is reproduced here.
    #[must_use]
    pub fn normal(&self) -> [f32; 3] {
        let p = |i: usize| {
            let v = self.verts[i].pos;
            [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
        };
        let (v0, v1, v2) = (p(0), p(1), p(2));
        let a = [v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]];
        let b = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
        // vec31.crossProduct(vec3) == b × a
        let c = [
            b[1] * a[2] - b[2] * a[1],
            b[2] * a[0] - b[0] * a[2],
            b[0] * a[1] - b[1] * a[0],
        ];
        let len = f64::from(mc_math::sqrt_d(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]));
        if len < 1.0e-4 {
            [0.0, 0.0, 0.0]
        } else {
            [
                (c[0] / len) as f32,
                (c[1] / len) as f32,
                (c[2] / len) as f32,
            ]
        }
    }
}

/// An axis-aligned textured box, baked into its six faces at construction time.
///
/// java: `ModelBox`
#[derive(Clone, Debug, PartialEq)]
pub struct ModelBox {
    /// The six faces, in vanilla's order: `+X`, `-X`, `-Y`, `+Y`, `-Z`, `+Z`.
    ///
    /// Model space is Y-down, so `-Y` is the face the *top* texture region lands on.
    pub quads: [TexturedQuad; 6],
    /// `quads[i].normal()`, baked once here instead of per quad per frame in
    /// [`Model::emit_part`]. The normal depends only on the bind-pose corners, which nothing
    /// mutates after construction, so the cached value is bit-identical to recomputing it.
    pub normals: [[f32; 3]; 6],
    /// The eight inflated corners every face vertex is a copy of, and which corner each
    /// `quads[f].verts[k]` is (`corner_index[f][k]`, after the mirror flip). [`Model::emit_part`]
    /// transforms the 8 corners once instead of 24 face vertices; identical inputs under the
    /// same matrix give identical bits, so the stream is unchanged (ENTM-5 item 2).
    pub corners: [[f32; 3]; 8],
    /// See [`Self::corners`].
    pub corner_index: [[u8; 4]; 6],
    /// Lower corner, excluding `delta`. java: `ModelBox.posX1/posY1/posZ1`
    pub pos1: [f32; 3],
    /// Upper corner, excluding `delta`. java: `ModelBox.posX2/posY2/posZ2`
    pub pos2: [f32; 3],
}

impl ModelBox {
    /// Bakes a box exactly as `ModelBox`'s constructor does.
    ///
    /// `delta` is vanilla's `scaleFactor`/inflation: it grows the box outward on every axis
    /// *without* changing the UV layout, which is how armour layers and the player's hat layer
    /// sit just outside the body. `mirror` swaps the two X extents and then reverses every
    /// face's winding.
    ///
    /// java: `ModelBox.<init>`
    #[allow(clippy::too_many_arguments)] // one-for-one with the Java constructor
    #[must_use]
    pub fn new(
        texture_width: f32,
        texture_height: f32,
        texture_x: i32,
        texture_y: i32,
        x: f32,
        y: f32,
        z: f32,
        width: i32,
        height: i32,
        depth: i32,
        delta: f32,
        mirror: bool,
    ) -> Self {
        let (w, h, d) = (width as f32, height as f32, depth as f32);
        let pos1 = [x, y, z];
        let pos2 = [x + w, y + h, z + d];

        let mut x1 = x - delta;
        let mut y1 = y - delta;
        let z1 = z - delta;
        let mut x2 = x + w + delta;
        let y2 = y + h + delta;
        let z2 = z + d + delta;

        if mirror {
            core::mem::swap(&mut x1, &mut x2);
        }
        // `y1` is only ever reassigned through the swap above in vanilla's mirrored branch for
        // X; keep the binding mutable-free noise down.
        let _ = &mut y1;

        // java: the eight `positiontexturevertex*` locals, in declaration order.
        let c = [
            [x1, y1, z1], // 0
            [x2, y1, z1], // 1
            [x2, y2, z1], // 2
            [x1, y2, z1], // 3
            [x1, y1, z2], // 4
            [x2, y1, z2], // 5
            [x2, y2, z2], // 6
            [x1, y2, z2], // 7
        ];

        let (tx, ty) = (texture_x, texture_y);
        let (wi, hi, di) = (width, height, depth);

        // (corner indices, u1, v1, u2, v2) — verbatim from `ModelBox.<init>`'s six
        // `new TexturedQuad(...)` calls.
        let faces: [([usize; 4], i32, i32, i32, i32); 6] = [
            (
                [5, 1, 2, 6],
                tx + di + wi,
                ty + di,
                tx + di + wi + di,
                ty + di + hi,
            ),
            ([0, 4, 7, 3], tx, ty + di, tx + di, ty + di + hi),
            ([5, 4, 0, 1], tx + di, ty, tx + di + wi, ty + di),
            ([2, 3, 7, 6], tx + di + wi, ty + di, tx + di + wi + wi, ty),
            ([1, 0, 3, 2], tx + di, ty + di, tx + di + wi, ty + di + hi),
            (
                [4, 5, 6, 7],
                tx + di + wi + di,
                ty + di,
                tx + di + wi + di + wi,
                ty + di + hi,
            ),
        ];

        let quads = core::array::from_fn(|i| {
            let (idx, u1, v1, u2, v2) = faces[i];
            // java: `TexturedQuad`'s 6-arg constructor. The `f`/`f1` epsilons are `0.0F / size`,
            // i.e. exactly zero in 1.8.9 — kept out rather than multiplied by nothing.
            let (u1, v1, u2, v2) = (u1 as f32, v1 as f32, u2 as f32, v2 as f32);
            let uv = [
                [u2 / texture_width, v1 / texture_height],
                [u1 / texture_width, v1 / texture_height],
                [u1 / texture_width, v2 / texture_height],
                [u2 / texture_width, v2 / texture_height],
            ];
            let mut quad = TexturedQuad {
                verts: core::array::from_fn(|k| QuadVertex {
                    pos: c[idx[k]],
                    uv: uv[k],
                }),
            };
            if mirror {
                quad.flip_face();
            }
            quad
        });
        // After the mirror flip: the winding decides the normal's sign.
        let normals = core::array::from_fn(|i| quads[i].normal());
        // `flip_face` reverses the vertex order, so the corner order reverses with it.
        let corner_index = core::array::from_fn(|i| {
            let idx = faces[i].0;
            core::array::from_fn(|k| idx[if mirror { 3 - k } else { k }] as u8)
        });

        Self {
            quads,
            normals,
            corners: c,
            corner_index,
            pos1,
            pos2,
        }
    }
}

/// Index of a [`ModelPart`] inside its owning [`Model`].
///
/// Parts form a tree, and animation code needs to reach any of them by name after construction.
/// Vanilla uses object references; the arena-index rule rules those out for the world graph, and an
/// arena index is both cheaper and easier to test, so parts live in a `Vec` and this is the
/// handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PartId(pub u16);

impl PartId {
    /// The arena slot this handle points at.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One node of a model's part tree: a pivot, three Euler angles, and some boxes.
///
/// java: `ModelRenderer`
#[derive(Clone, Debug)]
pub struct ModelPart {
    /// Vanilla's `boxName`, used by layers and tests to find a part.
    pub name: &'static str,
    /// Texture width this part's UVs were baked against. java: `ModelRenderer.textureWidth`
    pub texture_width: f32,
    /// Texture height this part's UVs were baked against. java: `ModelRenderer.textureHeight`
    pub texture_height: f32,
    /// Current UV origin for the next `add_box`. java: `ModelRenderer.textureOffsetX/Y`
    pub texture_offset: [i32; 2],
    /// Pivot, in model units. Scaled by the draw-time `scale`. java: `rotationPointX/Y/Z`
    pub rotation_point: [f32; 3],
    /// Euler angles in radians, applied Z then Y then X. java: `rotateAngleX/Y/Z`
    pub rotate_angle: [f32; 3],
    /// Extra translate applied *before* the pivot and **not** scaled. java: `offsetX/Y/Z`
    pub offset: [f32; 3],
    /// Whether boxes added from now on are mirrored. java: `ModelRenderer.mirror`
    pub mirror: bool,
    /// Drawn at all? java: `ModelRenderer.showModel`
    pub show_model: bool,
    /// Force-hidden, overriding `show_model`. java: `ModelRenderer.isHidden`
    pub is_hidden: bool,
    /// This part's boxes. java: `ModelRenderer.cubeList`
    pub cubes: Vec<ModelBox>,
    /// Children, drawn inside this part's matrix. java: `ModelRenderer.childModels`
    pub children: Vec<PartId>,
}

impl ModelPart {
    /// This part's local transform, exactly as `ModelRenderer.render` composes it.
    ///
    /// `scale` is the draw-time model scale (`0.0625` for every living entity — one model unit
    /// is a sixteenth of a block).
    #[must_use]
    pub fn local_matrix(&self, scale: f32) -> Mat4 {
        let [ox, oy, oz] = self.offset;
        let [px, py, pz] = self.rotation_point;
        let [ax, ay, az] = self.rotate_angle;
        Mat4::from_translation(Vec3::new(ox, oy, oz))
            * Mat4::from_translation(Vec3::new(px * scale, py * scale, pz * scale))
            * Mat4::from_rotation_z(az)
            * Mat4::from_rotation_y(ay)
            * Mat4::from_rotation_x(ax)
    }

    /// This part's transform as `ModelRenderer.postRender` leaves it on the GL matrix stack.
    ///
    /// Unlike [`Self::local_matrix`], this intentionally omits `offset`: Java's `render`
    /// applies and later undoes `offsetX/Y/Z` around the part and its children, whereas
    /// `postRender` only applies `rotationPoint * scale` and Z/Y/X rotations.  Layers that
    /// anchor an item or attachment after `postRender` therefore must use this transform.
    /// Hidden or disabled parts leave the matrix stack unchanged, matching Java's early return.
    #[must_use]
    pub fn post_render_matrix(&self, scale: f32) -> Mat4 {
        if self.is_hidden || !self.show_model {
            return Mat4::IDENTITY;
        }
        let [px, py, pz] = self.rotation_point;
        let [ax, ay, az] = self.rotate_angle;
        Mat4::from_translation(Vec3::new(px * scale, py * scale, pz * scale))
            * Mat4::from_rotation_z(az)
            * Mat4::from_rotation_y(ay)
            * Mat4::from_rotation_x(ax)
    }

    /// Sets the pivot. java: `ModelRenderer.setRotationPoint`
    pub fn set_rotation_point(&mut self, x: f32, y: f32, z: f32) {
        self.rotation_point = [x, y, z];
    }

    /// Sets all three Euler angles at once.
    pub fn set_angles(&mut self, x: f32, y: f32, z: f32) {
        self.rotate_angle = [x, y, z];
    }

    /// Adds a box at the current texture offset. java: `ModelRenderer.addBox`
    #[allow(clippy::too_many_arguments)] // one-for-one with the Java method
    pub fn add_box(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        width: i32,
        height: i32,
        depth: i32,
        delta: f32,
    ) {
        self.cubes.push(ModelBox::new(
            self.texture_width,
            self.texture_height,
            self.texture_offset[0],
            self.texture_offset[1],
            x,
            y,
            z,
            width,
            height,
            depth,
            delta,
            self.mirror,
        ));
    }
}

/// A whole entity model: a part forest plus the per-frame flags vanilla keeps on `ModelBase`.
///
/// java: `ModelBase`
#[derive(Clone, Debug)]
pub struct Model {
    parts: Vec<ModelPart>,
    roots: Vec<PartId>,
    /// java: `ModelBase.textureWidth`
    pub texture_width: u32,
    /// java: `ModelBase.textureHeight`
    pub texture_height: u32,
    /// Attack-swing progress in `0..1`. java: `ModelBase.swingProgress`
    pub swing_progress: f32,
    /// java: `ModelBase.isRiding`
    pub is_riding: bool,
    /// java: `ModelBase.isChild` — note vanilla initialises this to `true`.
    pub is_child: bool,
}

impl Model {
    /// An empty model with the given texture size.
    #[must_use]
    pub fn new(texture_width: u32, texture_height: u32) -> Self {
        Self {
            parts: Vec::new(),
            roots: Vec::new(),
            texture_width,
            texture_height,
            swing_progress: 0.0,
            is_riding: false,
            is_child: true, // java: `ModelBase.isChild = true`
        }
    }

    /// Adds a root part and returns a builder for it.
    ///
    /// `u`/`v` are the part's texture-offset corner, as `new ModelRenderer(this, u, v)`.
    pub fn part(&mut self, name: &'static str, u: i32, v: i32) -> PartBuilder<'_> {
        let id = PartId(u16::try_from(self.parts.len()).expect("model part count fits in u16"));
        self.parts.push(ModelPart {
            name,
            texture_width: self.texture_width as f32,
            texture_height: self.texture_height as f32,
            texture_offset: [u, v],
            rotation_point: [0.0; 3],
            rotate_angle: [0.0; 3],
            offset: [0.0; 3],
            mirror: false,
            show_model: true,
            is_hidden: false,
            cubes: Vec::new(),
            children: Vec::new(),
        });
        self.roots.push(id);
        PartBuilder { model: self, id }
    }

    /// Adds a part parented to `parent`. Equivalent to `parent.addChild(child)` in the Java.
    pub fn child(&mut self, parent: PartId, name: &'static str, u: i32, v: i32) -> PartBuilder<'_> {
        let id = self.part(name, u, v).id();
        self.reparent(parent, id);
        PartBuilder { model: self, id }
    }

    /// Moves an existing root under `parent`. java: `ModelRenderer.addChild`
    pub fn reparent(&mut self, parent: PartId, child: PartId) {
        self.roots.retain(|r| *r != child);
        self.parts[parent.index()].children.push(child);
    }

    /// Borrows a part.
    #[inline]
    #[must_use]
    pub fn get(&self, id: PartId) -> &ModelPart {
        &self.parts[id.index()]
    }

    /// Mutably borrows a part.
    #[inline]
    pub fn get_mut(&mut self, id: PartId) -> &mut ModelPart {
        &mut self.parts[id.index()]
    }

    /// Every part, in creation order — vanilla's `boxList`.
    #[inline]
    #[must_use]
    pub fn parts(&self) -> &[ModelPart] {
        &self.parts
    }

    /// The parts with no parent, in creation order.
    #[inline]
    #[must_use]
    pub fn roots(&self) -> &[PartId] {
        &self.roots
    }

    /// Finds a part by its `boxName`. Layers and tests use this; hot paths should keep the
    /// [`PartId`] a model handed out at construction.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<PartId> {
        self.parts
            .iter()
            .position(|p| p.name == name)
            .map(|i| PartId(i as u16))
    }

    /// Copies pivot and angles from one part to another.
    ///
    /// java: `ModelBase.copyModelAngles` — used to pin the hat layer to the head.
    pub fn copy_angles(&mut self, from: PartId, to: PartId) {
        let (angle, point) = {
            let p = self.get(from);
            (p.rotate_angle, p.rotation_point)
        };
        let d = self.get_mut(to);
        d.rotate_angle = angle;
        d.rotation_point = point;
    }

    /// Emits one part and its subtree.
    ///
    /// `base` is the matrix already on the stack — for a living entity that is the entity
    /// transform from [`crate::entity::render::entity_matrix`].
    pub fn emit_part(
        &self,
        id: PartId,
        scale: f32,
        base: Mat4,
        out: &mut Vec<crate::entity::EntityVertex>,
    ) {
        let part = self.get(id);
        if part.is_hidden || !part.show_model {
            return;
        }
        let m = base * part.local_matrix(scale);
        for cube in &part.cubes {
            let corners: [Vec3; 8] = core::array::from_fn(|i| {
                let v = cube.corners[i];
                m.transform_point3(Vec3::new(v[0] * scale, v[1] * scale, v[2] * scale))
            });
            for ((quad, &n), index) in cube.quads.iter().zip(&cube.normals).zip(&cube.corner_index)
            {
                let n = m.transform_vector3(Vec3::from(n)).normalize_or_zero();
                let p: [Vec3; 4] = core::array::from_fn(|i| corners[usize::from(index[i])]);
                // GL_QUADS -> two triangles, matching the 0-1-2 / 0-2-3 fan wgpu wants.
                let face: [crate::entity::EntityVertex; 4] = core::array::from_fn(|i| {
                    crate::entity::EntityVertex {
                        pos: p[i].into(),
                        uv: quad.verts[i].uv,
                        normal: n.into(),
                        // Same untinted display-list vertices and triangle order.
                        color: crate::entity::EntityVertex::WHITE,
                    }
                });
                out.extend_from_slice(&[face[0], face[1], face[2], face[0], face[2], face[3]]);
            }
        }
        for &c in &part.children {
            self.emit_part(c, scale, m, out);
        }
    }

    /// Emits every root part and its subtree — the default `ModelBase.render` body.
    pub fn emit_all(&self, scale: f32, base: Mat4, out: &mut Vec<crate::entity::EntityVertex>) {
        for &r in &self.roots {
            self.emit_part(r, scale, base, out);
        }
    }

    /// Emits a named subset of the roots, in the order given.
    ///
    /// Several vanilla models draw only some parts under a given matrix (the child-scaling
    /// branch of `ModelBiped.render`, `ModelSheep2`'s wool pass); this is the seam for those.
    pub fn emit_some(
        &self,
        ids: &[PartId],
        scale: f32,
        base: Mat4,
        out: &mut Vec<crate::entity::EntityVertex>,
    ) {
        for &id in ids {
            self.emit_part(id, scale, base, out);
        }
    }
}

/// Chaining builder for a [`ModelPart`], mirroring the Java call sequence.
///
/// ```text
/// this.head = new ModelRenderer(this, 0, 0);       ->  let head = m.part("head", 0, 0)
/// this.head.addBox(-4, -8, -4, 8, 8, 8, size);     ->      .cube(-4.0, -8.0, -4.0, 8, 8, 8, size)
/// this.head.setRotationPoint(0, 0, 0);             ->      .at(0.0, 0.0, 0.0)
///                                                  ->      .id();
/// ```
///
/// Order matters exactly as it does in the Java: [`PartBuilder::mirrored`] and
/// [`PartBuilder::tex`] affect only the cubes added *after* them.
pub struct PartBuilder<'a> {
    model: &'a mut Model,
    id: PartId,
}

impl PartBuilder<'_> {
    /// Adds a box. java: `ModelRenderer.addBox(x, y, z, w, h, d, scaleFactor)`
    #[allow(clippy::too_many_arguments)] // one-for-one with the Java method
    #[must_use]
    pub fn cube(self, x: f32, y: f32, z: f32, w: i32, h: i32, d: i32, delta: f32) -> Self {
        self.model.get_mut(self.id).add_box(x, y, z, w, h, d, delta);
        self
    }

    /// Sets the pivot. java: `ModelRenderer.setRotationPoint`
    #[must_use]
    pub fn at(self, x: f32, y: f32, z: f32) -> Self {
        self.model.get_mut(self.id).set_rotation_point(x, y, z);
        self
    }

    /// Sets the Euler angles. Most models set these per-frame instead.
    #[must_use]
    pub fn angles(self, x: f32, y: f32, z: f32) -> Self {
        self.model.get_mut(self.id).set_angles(x, y, z);
        self
    }

    /// Mirrors every box added after this call. java: `ModelRenderer.mirror = true`
    #[must_use]
    pub fn mirrored(self) -> Self {
        self.model.get_mut(self.id).mirror = true;
        self
    }

    /// Clears the mirror flag again.
    #[must_use]
    pub fn unmirrored(self) -> Self {
        self.model.get_mut(self.id).mirror = false;
        self
    }

    /// Moves the UV origin for subsequent boxes. java: `ModelRenderer.setTextureOffset`
    #[must_use]
    pub fn tex(self, u: i32, v: i32) -> Self {
        self.model.get_mut(self.id).texture_offset = [u, v];
        self
    }

    /// Overrides the texture size this part's UVs divide by.
    ///
    /// java: `ModelRenderer.setTextureSize` — a handful of models (the dragon, the horse, the
    /// wither) use a texture larger than their `ModelBase` default.
    #[must_use]
    pub fn tex_size(self, w: i32, h: i32) -> Self {
        let p = self.model.get_mut(self.id);
        p.texture_width = w as f32;
        p.texture_height = h as f32;
        self
    }

    /// Sets the unscaled offset. java: `ModelRenderer.offsetX/Y/Z`
    #[must_use]
    pub fn offset(self, x: f32, y: f32, z: f32) -> Self {
        self.model.get_mut(self.id).offset = [x, y, z];
        self
    }

    /// Hides the part by default. java: `ModelRenderer.showModel = false`
    #[must_use]
    pub fn hidden(self) -> Self {
        self.model.get_mut(self.id).show_model = false;
        self
    }

    /// Finishes the chain.
    #[must_use]
    pub fn id(self) -> PartId {
        self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The player head: `new ModelRenderer(this, 0, 0).addBox(-4, -8, -4, 8, 8, 8, 0)` on a
    /// 64x32 texture. Every number below is hand-derived from `ModelBox.<init>`.
    #[test]
    fn head_box_geometry_is_exact() {
        let b = ModelBox::new(64.0, 32.0, 0, 0, -4.0, -8.0, -4.0, 8, 8, 8, 0.0, false);
        assert_eq!(b.pos1, [-4.0, -8.0, -4.0]);
        assert_eq!(b.pos2, [4.0, 0.0, 4.0]);

        // Face 4 is the z = min face; corners 1, 0, 3, 2.
        let f = b.quads[4];
        assert_eq!(f.verts[0].pos, [4.0, -8.0, -4.0]);
        assert_eq!(f.verts[1].pos, [-4.0, -8.0, -4.0]);
        assert_eq!(f.verts[2].pos, [-4.0, 0.0, -4.0]);
        assert_eq!(f.verts[3].pos, [4.0, 0.0, -4.0]);
        // u1 = 0 + 8 = 8, v1 = 0 + 8 = 8, u2 = 8 + 8 = 16, v2 = 8 + 8 = 16.
        assert_eq!(f.verts[0].uv, [16.0 / 64.0, 8.0 / 32.0]);
        assert_eq!(f.verts[1].uv, [8.0 / 64.0, 8.0 / 32.0]);
        assert_eq!(f.verts[2].uv, [8.0 / 64.0, 16.0 / 32.0]);
        assert_eq!(f.verts[3].uv, [16.0 / 64.0, 16.0 / 32.0]);

        // Face 2 is the y = min face — the *top* of the head, since model space is Y-down.
        // u1 = 8, v1 = 0, u2 = 16, v2 = 8: the 8x8 block at (8, 0) in the skin.
        let top = b.quads[2];
        assert!(top.verts.iter().all(|v| v.pos[1] == -8.0));
        assert_eq!(top.verts[1].uv, [8.0 / 64.0, 0.0]);
        assert_eq!(top.verts[3].uv, [16.0 / 64.0, 8.0 / 32.0]);

        // Face 3 is the y = max face — the bottom, whose V range vanilla writes inverted.
        let bottom = b.quads[3];
        assert!(bottom.verts.iter().all(|v| v.pos[1] == 0.0));
        // u1 = 16, v1 = 8, u2 = 24, v2 = 0 -> V decreasing across the quad.
        assert_eq!(bottom.verts[1].uv, [16.0 / 64.0, 8.0 / 32.0]);
        assert_eq!(bottom.verts[2].uv, [16.0 / 64.0, 0.0]);
    }

    /// `bipedRightArm`: offset (40, 16), box (-3, -2, -2, 4, 12, 4).
    #[test]
    fn arm_box_uv_layout_matches_the_skin() {
        let b = ModelBox::new(64.0, 32.0, 40, 16, -3.0, -2.0, -2.0, 4, 12, 4, 0.0, false);
        // -X face (index 1): u = 40..44, v = 20..32.
        let west = b.quads[1];
        assert_eq!(west.verts[1].uv, [40.0 / 64.0, 20.0 / 32.0]);
        assert_eq!(west.verts[3].uv, [44.0 / 64.0, 32.0 / 32.0]);
        // -Z face (index 4): u = 44..48, v = 20..32.
        let north = b.quads[4];
        assert_eq!(north.verts[1].uv, [44.0 / 64.0, 20.0 / 32.0]);
        assert_eq!(north.verts[0].uv, [48.0 / 64.0, 20.0 / 32.0]);
        // +X face (index 0): u = 48..52.
        assert_eq!(b.quads[0].verts[1].uv, [48.0 / 64.0, 20.0 / 32.0]);
        // +Z face (index 5): u = 52..56.
        assert_eq!(b.quads[5].verts[1].uv, [52.0 / 64.0, 20.0 / 32.0]);
    }

    #[test]
    fn delta_inflates_without_moving_the_uvs() {
        let plain = ModelBox::new(64.0, 64.0, 0, 0, -4.0, -8.0, -4.0, 8, 8, 8, 0.0, false);
        let fat = ModelBox::new(64.0, 64.0, 0, 0, -4.0, -8.0, -4.0, 8, 8, 8, 0.5, false);
        assert_eq!(plain.pos1, fat.pos1, "delta must not change posX1..posZ2");
        assert_eq!(plain.pos2, fat.pos2);
        assert_eq!(fat.quads[4].verts[1].pos, [-4.5, -8.5, -4.5]);
        assert_eq!(fat.quads[4].verts[3].pos, [4.5, 0.5, -4.5]);
        for i in 0..6 {
            for k in 0..4 {
                assert_eq!(plain.quads[i].verts[k].uv, fat.quads[i].verts[k].uv);
            }
        }
    }

    #[test]
    fn mirroring_swaps_x_and_reverses_winding() {
        let plain = ModelBox::new(64.0, 64.0, 40, 16, -1.0, -2.0, -2.0, 4, 12, 4, 0.0, false);
        let mirr = ModelBox::new(64.0, 64.0, 40, 16, -1.0, -2.0, -2.0, 4, 12, 4, 0.0, true);
        // pos1/pos2 are recorded before the swap, so they are untouched.
        assert_eq!(plain.pos1, mirr.pos1);
        assert_eq!(plain.pos2, mirr.pos2);
        // Every mirrored quad is the plain one read backwards.
        for i in 0..6 {
            for k in 0..4 {
                let m = mirr.quads[i].verts[k];
                let p = plain.quads[i].verts[3 - k];
                assert_eq!(m.uv, p.uv, "face {i} vertex {k} uv");
                // ...with the two X extents exchanged.
                assert_eq!(m.pos[1], p.pos[1]);
                assert_eq!(m.pos[2], p.pos[2]);
                assert_eq!(m.pos[0], if p.pos[0] == -1.0 { 3.0 } else { -1.0 });
            }
        }
        // Mirroring does *two* things that each flip the winding — the X swap and `flipFace` —
        // so they cancel and every face still faces outward. That cancellation is the entire
        // reason `ModelBox` calls `flipFace`; without it a mirrored limb would be inside out.
        assert_eq!(plain.quads[4].normal(), [0.0, 0.0, -1.0]);
        assert_eq!(mirr.quads[4].normal(), [0.0, 0.0, -1.0]);
        // The +X face of a mirrored box is geometrically its -X face, and its normal follows.
        assert_eq!(plain.quads[0].normal(), [1.0, 0.0, 0.0]);
        assert_eq!(mirr.quads[0].normal(), [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn normals_point_out_of_each_face() {
        let b = ModelBox::new(64.0, 64.0, 0, 0, -4.0, -8.0, -4.0, 8, 8, 8, 0.0, false);
        // Faces are +X, -X, -Y, +Y, -Z, +Z in vanilla's construction order.
        assert_eq!(b.quads[0].normal(), [1.0, 0.0, 0.0]);
        assert_eq!(b.quads[1].normal(), [-1.0, 0.0, 0.0]);
        assert_eq!(b.quads[2].normal(), [0.0, -1.0, 0.0]);
        assert_eq!(b.quads[3].normal(), [0.0, 1.0, 0.0]);
        assert_eq!(b.quads[4].normal(), [0.0, 0.0, -1.0]);
        assert_eq!(b.quads[5].normal(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn local_matrix_is_translate_pivot_then_zyx() {
        let mut m = Model::new(64, 32);
        let id = m.part("arm", 40, 16).at(-5.0, 2.0, 0.0).id();
        m.get_mut(id).rotate_angle = [0.5, 0.25, 0.125];

        let got = m.get(id).local_matrix(0.0625);
        let want = Mat4::from_translation(Vec3::new(-5.0 * 0.0625, 2.0 * 0.0625, 0.0))
            * Mat4::from_rotation_z(0.125)
            * Mat4::from_rotation_y(0.25)
            * Mat4::from_rotation_x(0.5);
        for (a, b) in got.to_cols_array().iter().zip(want.to_cols_array()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn pivot_is_scaled_but_offset_is_not() {
        // java: ModelRenderer.render translates `offset` raw and `rotationPoint * scale`.
        let mut m = Model::new(64, 32);
        let id = m
            .part("p", 0, 0)
            .at(16.0, 0.0, 0.0)
            .offset(1.0, 0.0, 0.0)
            .id();
        let p = m.get(id).local_matrix(0.0625).transform_point3(Vec3::ZERO);
        assert!(
            (p.x - 2.0).abs() < 1e-6,
            "1.0 offset + 16*0.0625 pivot = 2.0, got {p}"
        );
    }

    #[test]
    fn post_render_omits_offset_but_keeps_pivot_and_rotation_order() {
        // java: ModelRenderer.postRender has no offsetX/Y/Z translate, but otherwise uses the
        // same pivot * scale then Z/Y/X composition as render.
        let mut m = Model::new(64, 32);
        let id = m
            .part("nose", 0, 0)
            .at(16.0, -8.0, 4.0)
            .offset(0.0, 0.1875, -0.09375)
            .id();
        m.get_mut(id).rotate_angle = [0.5, 0.25, 0.125];
        let part = m.get(id);
        let scale = 0.0625;

        let want_post = Mat4::from_translation(Vec3::new(1.0, -0.5, 0.25))
            * Mat4::from_rotation_z(0.125)
            * Mat4::from_rotation_y(0.25)
            * Mat4::from_rotation_x(0.5);
        let got_post = part.post_render_matrix(scale);
        for (got, want) in got_post
            .to_cols_array()
            .iter()
            .zip(want_post.to_cols_array())
        {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }

        // A cube point is drawn under `render`'s leading raw offset, while an attachment put
        // on the GL stack by `postRender` is not. This is the witch-potion divergence.
        let point = Vec3::new(0.25, -0.5, 0.75);
        let rendered = part.local_matrix(scale).transform_point3(point);
        let attached = got_post.transform_point3(point);
        assert!(
            (rendered - attached - Vec3::new(0.0, 0.1875, -0.09375)).length() < 1e-6,
            "rendered {rendered}, postRender attachment {attached}"
        );
    }

    #[test]
    fn post_render_and_local_matrix_coincide_at_zero_offset() {
        let mut m = Model::new(64, 32);
        let id = m.part("head", 0, 0).at(3.0, -2.0, 1.0).id();
        m.get_mut(id).rotate_angle = [0.5, 0.25, 0.125];
        let part = m.get(id);
        for (local, post) in part
            .local_matrix(0.0625)
            .to_cols_array()
            .iter()
            .zip(part.post_render_matrix(0.0625).to_cols_array())
        {
            assert!((local - post).abs() < 1e-6, "{local} != {post}");
        }
    }

    #[test]
    fn hidden_or_disabled_parts_leave_post_render_attachments_untransformed() {
        let mut m = Model::new(64, 32);
        let hidden = m.part("hidden", 0, 0).at(16.0, -8.0, 4.0).id();
        let visible = m.part("visible", 0, 0).at(-4.0, 6.0, 2.0).id();
        m.get_mut(hidden).rotate_angle = [0.5, 0.25, 0.125];
        m.get_mut(visible).rotate_angle = [-0.25, 0.125, -0.5];
        let visible_matrix = m.get(visible).post_render_matrix(0.0625);
        assert_ne!(
            visible_matrix,
            Mat4::IDENTITY,
            "visible control needs a biting pivot"
        );

        m.get_mut(hidden).show_model = false;
        assert_eq!(m.get(hidden).post_render_matrix(0.0625), Mat4::IDENTITY);

        m.get_mut(hidden).show_model = true;
        m.get_mut(hidden).is_hidden = true;
        assert_eq!(m.get(hidden).post_render_matrix(0.0625), Mat4::IDENTITY);
        assert_eq!(
            m.get(visible).post_render_matrix(0.0625),
            visible_matrix,
            "hiding one part must not alter a visible sibling attachment"
        );
    }

    #[test]
    fn children_inherit_the_parent_matrix() {
        let mut m = Model::new(64, 32);
        let body = m.part("body", 0, 0).at(0.0, 4.0, 0.0).id();
        let head = m
            .child(body, "head", 0, 0)
            .at(0.0, -8.0, 0.0)
            .cube(0.0, 0.0, 0.0, 1, 1, 1, 0.0)
            .id();
        assert_eq!(m.roots(), &[body]);
        assert_eq!(m.get(body).children, vec![head]);

        let mut out = Vec::new();
        m.emit_all(1.0, Mat4::IDENTITY, &mut out);
        // 6 faces x 6 verts.
        assert_eq!(out.len(), 36);
        // The cube's origin corner lands at body pivot + head pivot = (0, -4, 0).
        assert!(out.iter().any(|v| (v.pos[1] + 4.0).abs() < 1e-6));
    }

    #[test]
    fn hidden_parts_emit_nothing_and_skip_their_children() {
        let mut m = Model::new(64, 32);
        let body = m.part("body", 0, 0).cube(0.0, 0.0, 0.0, 1, 1, 1, 0.0).id();
        let _head = m
            .child(body, "head", 0, 0)
            .cube(0.0, 0.0, 0.0, 1, 1, 1, 0.0)
            .id();

        let mut out = Vec::new();
        m.emit_all(1.0, Mat4::IDENTITY, &mut out);
        assert_eq!(out.len(), 72);

        m.get_mut(body).show_model = false;
        out.clear();
        m.emit_all(1.0, Mat4::IDENTITY, &mut out);
        assert!(out.is_empty(), "a hidden parent must hide its subtree too");

        m.get_mut(body).show_model = true;
        m.get_mut(body).is_hidden = true;
        out.clear();
        m.emit_all(1.0, Mat4::IDENTITY, &mut out);
        assert!(out.is_empty());
    }

    /// The pre-bake `emit_part`, kept verbatim as the oracle: the normal is recomputed from the
    /// bind-pose corners for every quad on every call.
    fn emit_part_recomputing_normals(
        model: &Model,
        id: PartId,
        scale: f32,
        base: Mat4,
        out: &mut Vec<crate::entity::EntityVertex>,
    ) {
        let part = model.get(id);
        if part.is_hidden || !part.show_model {
            return;
        }
        let m = base * part.local_matrix(scale);
        for cube in &part.cubes {
            for quad in &cube.quads {
                let n = quad.normal();
                let n = m.transform_vector3(Vec3::from(n)).normalize_or_zero();
                let p: [Vec3; 4] = core::array::from_fn(|i| {
                    let v = quad.verts[i].pos;
                    m.transform_point3(Vec3::new(v[0] * scale, v[1] * scale, v[2] * scale))
                });
                for &i in &[0usize, 1, 2, 0, 2, 3] {
                    out.push(crate::entity::EntityVertex {
                        pos: p[i].into(),
                        uv: quad.verts[i].uv,
                        normal: n.into(),
                        color: crate::entity::EntityVertex::WHITE,
                    });
                }
            }
        }
        for &c in &part.children {
            emit_part_recomputing_normals(model, c, scale, m, out);
        }
    }

    /// #1 step 1 (a): `ModelBox::normals` is baked from the same untransformed corners the draw
    /// used to read, so every box of every model kind, at every inflation a layer or armour
    /// model uses, emits the same vertex stream bit for bit.
    #[test]
    fn baked_normals_reproduce_the_per_frame_normals_bit_for_bit() {
        use crate::entity::anim::AnimParams;
        let mut p = AnimParams::still();
        p.limb_swing = 12.5;
        p.limb_swing_amount = 0.7;
        p.age_in_ticks = 40.0;
        p.net_head_yaw = 25.0;
        p.head_pitch = -15.0;
        let base = Mat4::from_translation(Vec3::new(3.25, -1.5, 7.0))
            * Mat4::from_rotation_y(0.7)
            * Mat4::from_scale(Vec3::new(-0.9375, -0.9375, 0.9375));
        let mut quads = 0usize;
        for &kind in crate::entity::models::ModelKind::all() {
            for inflation in [0.0_f32, 0.5, 1.0, 2.0] {
                let mut m = kind.build(inflation);
                let mut posed = Vec::new();
                m.pose_and_render(&p, base, &mut posed);
                let model = m.model();
                for part in model.parts() {
                    for cube in &part.cubes {
                        for (face, (quad, normal)) in
                            cube.quads.iter().zip(cube.normals).enumerate()
                        {
                            assert_eq!(normal.map(f32::to_bits), quad.normal().map(f32::to_bits));
                            for (k, vertex) in quad.verts.iter().enumerate() {
                                let corner = cube.corners[usize::from(cube.corner_index[face][k])];
                                assert_eq!(corner.map(f32::to_bits), vertex.pos.map(f32::to_bits));
                            }
                            quads += 1;
                        }
                    }
                }
                let (mut baked, mut oracle) = (Vec::new(), Vec::new());
                for index in 0..model.parts().len() {
                    let id = PartId(index as u16);
                    model.emit_part(id, 0.0625, base, &mut baked);
                    emit_part_recomputing_normals(model, id, 0.0625, base, &mut oracle);
                }
                assert_eq!(
                    bytemuck::cast_slice::<_, u32>(&baked[..]),
                    bytemuck::cast_slice::<_, u32>(&oracle[..]),
                    "{kind:?} at inflation {inflation}"
                );
            }
        }
        assert!(
            quads > 1000,
            "the sweep must cover the model zoo, saw {quads} quads"
        );
    }

    #[test]
    fn copy_angles_pins_the_hat_to_the_head() {
        let mut m = Model::new(64, 32);
        let head = m.part("head", 0, 0).id();
        let hat = m.part("hat", 32, 0).id();
        m.get_mut(head).rotate_angle = [0.1, 0.2, 0.3];
        m.get_mut(head).rotation_point = [0.0, 1.0, 0.0];
        m.copy_angles(head, hat);
        assert_eq!(m.get(hat).rotate_angle, [0.1, 0.2, 0.3]);
        assert_eq!(m.get(hat).rotation_point, [0.0, 1.0, 0.0]);
    }
}
