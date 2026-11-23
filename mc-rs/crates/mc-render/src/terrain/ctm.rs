//! Connected textures in the mesher.
//!
//! [`mc_assets::ctm::CtmTable`] resolved every rule at bake time. This module does the per-quad
//! part of the format's tile choice for `method=ctm`:
//!
//! 1. the stacked-pane **edge merge**: a `glass_pane_top*` Up/Down quad between two panes of the
//!    same stored state is not emitted;
//! 2. the **neighbour mask** over the face's four in-plane sides and, in Fancy mode, its corners;
//! 3. the **47-tile lookup** ([`mc_assets::ctm::CTM_LUT`]) and the **UV remap** onto that tile.
//!
//! The face's left/right/down/up are derived from the quad's own UVs (the "texture"
//! orientation). That equals the format's fixed per-face table on every cube face, and keeps the
//! tile frame on the right side of a glass pane's mirrored back face.
//!
//! # Cost
//! * Off, or no table installed: [`CtmSection::new`] returns `None` once per section and the
//!   mesher never calls into this module.
//! * On, non-CTM block: one `u16` array load per block ([`CtmSection::block`]).
//! * On, CTM face: 4–8 `block_at` reads and compares, a 256-entry table load, 8 multiply-adds.
//!   A `connect=tile` neighbour that is not the same state is answered from a precomputed
//!   per-state face table; only an actual-state neighbour (a pane, a fence) that can show the
//!   same sprite takes the exact model lookup. No locks; no allocation except that rare lookup.

use mc_assets::ctm::{self, mask, CtmConnect, CtmEntry, CtmRuleRt, CtmSlot, CtmTable};
use mc_assets::model::{model_for_actual, BakedQuad};
use mc_assets::sprite::UvRect;
use mc_data::BlockState;
use mc_math::{BlockPos, Facing};
use mc_world::World;

/// The Connected Textures mod's mesher switch ([`super::MeshSettings::connected_textures`]).
///
/// The mode defaults to Fancy; the mod is off by default, like every mod in this client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CtmMode {
    /// Vanilla meshing: the table is never read.
    #[default]
    Off,
    /// "Fast": a corner counts as connected whenever both of its sides do, without
    /// reading it (16 of the 47 tiles are reachable).
    Fast,
    /// "Fancy": corners are read too, so inner corners get their notch tiles.
    Fancy,
}

/// Per-section CTM context, built once per section from [`super::MeshSettings`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct CtmSection {
    table: &'static CtmTable,
    fancy: bool,
}

impl CtmSection {
    /// `None` when the mod is off or the installed bake has no table: the vanilla path.
    pub(crate) fn new(mode: CtmMode) -> Option<CtmSection> {
        let fancy = match mode {
            CtmMode::Off => return None,
            CtmMode::Fast => false,
            CtmMode::Fancy => true,
        };
        mc_assets::model::ctm_table().map(|table| CtmSection { table, fancy })
    }

    /// The block's CTM context, or `None` for the overwhelmingly common non-CTM state.
    #[inline]
    pub(crate) fn block(self, state: BlockState) -> Option<CtmBlock> {
        self.table.slot(state).map(|slot| CtmBlock {
            table: self.table,
            slot,
            fancy: self.fancy,
        })
    }
}

/// CTM context for one block that has entries in the table.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CtmBlock {
    table: &'static CtmTable,
    slot: &'static CtmSlot,
    fancy: bool,
}

impl CtmBlock {
    /// The quad to emit: `quad` itself, a copy with remapped UVs written into `scratch`, or
    /// `None` when the quad is a merged pane edge.
    pub(crate) fn apply<'q>(
        &self,
        world: &World,
        state: BlockState,
        pos: BlockPos,
        quad: &'q BakedQuad,
        scratch: &'q mut Option<BakedQuad>,
    ) -> Option<&'q BakedQuad> {
        let Some(face) = quad.face else {
            return Some(quad);
        };
        let Some(entry) = self.slot.entry_at(ctm::uv_centroid(quad)) else {
            return Some(quad);
        };
        // The stacked-pane edge merge, before any rule. Stored states are
        // compared, so arms do not matter.
        if entry.pane_edge
            && matches!(face, Facing::Up | Facing::Down)
            && world.block_at(pos.relative(face)) == state
        {
            return None;
        }
        let Some(rule) = self.table.rule(entry.rule_by_face[face.index() as usize]) else {
            return Some(quad);
        };
        let src = entry.src;
        if src.max_u <= src.min_u || src.max_v <= src.min_v {
            return Some(quad);
        }
        let sides = orientation(quad, face);
        let neighbours = Neighbours {
            world,
            state,
            face,
            entry,
            rule,
            table: self.table,
        };
        let tile = usize::from(ctm::CTM_LUT[usize::from(neighbours.mask(pos, sides, self.fancy))]);
        if rule.keep & (1 << tile) != 0 {
            return Some(quad);
        }
        let dst = rule.tiles[tile];
        let remapped = scratch.insert(quad.clone());
        for vertex in &mut remapped.verts {
            vertex.uv = remap(vertex.uv, &src, &dst);
        }
        Some(&*remapped)
    }
}

/// Everything `connects` needs, so the mask loop reads like the spec.
struct Neighbours<'a> {
    world: &'a World,
    state: BlockState,
    face: Facing,
    entry: &'a CtmEntry,
    rule: &'a CtmRuleRt,
    table: &'a CtmTable,
}

impl Neighbours<'_> {
    /// The neighbour mask. `sides` is `[left, right, down, up]`.
    fn mask(&self, pos: BlockPos, sides: [Facing; 4], fancy: bool) -> u8 {
        let [left, right, down, up] = sides;
        let mut bits = 0u8;
        for (bit, side) in [
            (mask::L, left),
            (mask::D, down),
            (mask::R, right),
            (mask::U, up),
        ] {
            if self.connects(pos.relative(side)) {
                bits |= bit;
            }
        }
        // A corner only matters when both of its sides connect; Fast assumes it does.
        for (corner, a, b, side_a, side_b) in [
            (mask::DL, mask::D, mask::L, down, left),
            (mask::DR, mask::D, mask::R, down, right),
            (mask::UR, mask::U, mask::R, up, right),
            (mask::UL, mask::U, mask::L, up, left),
        ] {
            if bits & a != 0
                && bits & b != 0
                && (!fancy || self.connects(pos.relative(side_a).relative(side_b)))
            {
                bits |= corner;
            }
        }
        bits
    }

    /// `connects(q)` with `innerSeams`.
    fn connects(&self, q: BlockPos) -> bool {
        if self.rule.inner_seams {
            self.connects_raw(q) && !self.connects_raw(q.relative(self.face))
        } else {
            self.connects_raw(q)
        }
    }

    fn connects_raw(&self, q: BlockPos) -> bool {
        let neighbour = self.world.block_at(q);
        // The same stored state connects in every mode.
        if neighbour == self.state {
            return true;
        }
        match self.rule.connect {
            // Same block id and metadata is the same packed state, answered above.
            CtmConnect::Block => false,
            CtmConnect::Material => {
                !neighbour.props().is_air()
                    && neighbour.props().material == self.state.props().material
            }
            CtmConnect::Tile => self.shows_same_sprite(neighbour, q),
        }
    }

    /// `connect=tile`: the neighbour's un-CTM'd model shows this quad's source sprite on the
    /// same face (its first culled quad, else its first general quad).
    fn shows_same_sprite(&self, neighbour: BlockState, q: BlockPos) -> bool {
        if neighbour.props().is_air() {
            return false;
        }
        if !self.world.is_debug_world() && !super::cull_face::class_info(neighbour).actual_state {
            return self.table.face_src(neighbour, self.face) == self.entry.src_id;
        }
        // Panes, fences, ...: the face depends on the neighbour's own neighbours. Skip the exact
        // (allocating) lookup when no variant of that state can show a CTM sprite there.
        if !self.table.face_may_show_src(neighbour, self.face) {
            return false;
        }
        let actual = super::mesher::actual_state(self.world, neighbour, q);
        ctm::first_face_quad(model_for_actual(&actual), self.face)
            .is_some_and(|quad| ctm::rect_contains(&self.entry.src, ctm::uv_centroid(quad)))
    }
}

/// The world directions that are the face's texture-left, -right, -down and -up, derived from
/// the quad's UVs: `u` grows toward right, `v` toward down. Falls back to the fixed table for a
/// degenerate UV mapping.
#[must_use]
pub fn orientation(quad: &BakedQuad, face: Facing) -> [Facing; 4] {
    derived_orientation(quad, face).unwrap_or_else(|| fixed_orientation(face))
}

fn derived_orientation(quad: &BakedQuad, face: Facing) -> Option<[Facing; 4]> {
    let v = &quad.verts;
    let edge = |i: usize| {
        [
            v[i].pos[0] - v[0].pos[0],
            v[i].pos[1] - v[0].pos[1],
            v[i].pos[2] - v[0].pos[2],
        ]
    };
    let (e1, e3) = (edge(1), edge(3));
    let (du1, dv1) = (v[1].uv[0] - v[0].uv[0], v[1].uv[1] - v[0].uv[1]);
    let (du3, dv3) = (v[3].uv[0] - v[0].uv[0], v[3].uv[1] - v[0].uv[1]);
    let (du, along_u) = if du1.abs() >= du3.abs() {
        (du1, e1)
    } else {
        (du3, e3)
    };
    let (dv, along_v) = if dv1.abs() >= dv3.abs() {
        (dv1, e1)
    } else {
        (dv3, e3)
    };
    if du == 0.0 || dv == 0.0 {
        return None;
    }
    let right = snap(along_u, du.signum())?;
    let down = snap(along_v, dv.signum())?;
    if right.axis() == down.axis() || right.axis() == face.axis() || down.axis() == face.axis() {
        return None;
    }
    Some([right.opposite(), right, down, down.opposite()])
}

/// The facing of the largest component of `v * sign`.
fn snap(v: [f32; 3], sign: f32) -> Option<Facing> {
    let [x, y, z] = [v[0] * sign, v[1] * sign, v[2] * sign];
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    if ax == 0.0 && ay == 0.0 && az == 0.0 {
        return None;
    }
    Some(if ax >= ay && ax >= az {
        if x > 0.0 {
            Facing::East
        } else {
            Facing::West
        }
    } else if ay >= az {
        if y > 0.0 {
            Facing::Up
        } else {
            Facing::Down
        }
    } else if z > 0.0 {
        Facing::South
    } else {
        Facing::North
    })
}

/// The format's fixed `[left, right, down, up]` per face. This is exactly vanilla's default cube
/// UV orientation.
#[must_use]
pub fn fixed_orientation(face: Facing) -> [Facing; 4] {
    use Facing::{Down, East, North, South, Up, West};
    match face {
        Down => [West, East, North, South],
        Up => [West, East, South, North],
        North => [East, West, Down, Up],
        South => [West, East, Down, Up],
        West => [North, South, Down, Up],
        East => [South, North, Down, Up],
    }
}

/// The same relative position, moved from the source sprite to the tile.
#[inline]
fn remap(uv: [f32; 2], src: &UvRect, dst: &UvRect) -> [f32; 2] {
    let tu = (uv[0] - src.min_u) / (src.max_u - src.min_u);
    let tv = (uv[1] - src.min_v) / (src.max_v - src.min_v);
    [
        dst.min_u + tu * (dst.max_u - dst.min_u),
        dst.min_v + tv * (dst.max_v - dst.min_v),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_orientation_is_a_right_handed_frame_in_each_face_plane() {
        for face in Facing::VALUES {
            let [left, right, down, up] = fixed_orientation(face);
            assert_eq!(left, right.opposite(), "{face:?}");
            assert_eq!(down, up.opposite(), "{face:?}");
            assert_ne!(right.axis(), face.axis(), "{face:?}");
            assert_ne!(down.axis(), face.axis(), "{face:?}");
            assert_ne!(right.axis(), down.axis(), "{face:?}");
        }
    }

    #[test]
    fn remap_keeps_the_relative_position_inside_the_tile() {
        let src = UvRect {
            min_u: 0.25,
            max_u: 0.5,
            min_v: 0.0,
            max_v: 0.25,
        };
        let dst = UvRect {
            min_u: 0.5,
            max_u: 0.75,
            min_v: 0.5,
            max_v: 1.0,
        };
        // A pane arm's slice: texels 7..16 of the source become texels 7..16 of the tile.
        let a = remap([src.interpolate_u(7.0), src.interpolate_v(0.0)], &src, &dst);
        let b = remap(
            [src.interpolate_u(16.0), src.interpolate_v(16.0)],
            &src,
            &dst,
        );
        assert!((a[0] - dst.interpolate_u(7.0)).abs() < 1e-6);
        assert!((a[1] - dst.interpolate_v(0.0)).abs() < 1e-6);
        assert!((b[0] - dst.interpolate_u(16.0)).abs() < 1e-6);
        assert!((b[1] - dst.interpolate_v(16.0)).abs() < 1e-6);
    }

    #[test]
    fn snapping_picks_the_dominant_axis_and_sign() {
        assert_eq!(snap([0.0, 0.0, 1.0], 1.0), Some(Facing::South));
        assert_eq!(snap([0.0, 0.0, 1.0], -1.0), Some(Facing::North));
        assert_eq!(snap([-1.0, 0.1, 0.0], 1.0), Some(Facing::West));
        assert_eq!(snap([0.0, -0.5, 0.0], 1.0), Some(Facing::Down));
        assert_eq!(snap([0.0, 0.0, 0.0], 1.0), None);
    }

    #[test]
    fn the_mode_defaults_to_off() {
        assert_eq!(CtmMode::default(), CtmMode::Off);
        assert!(CtmSection::new(CtmMode::Off).is_none());
    }
}
