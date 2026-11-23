//! Ender Dragon renderer-only geometry.
//!
//! java: `RenderDragon.drawRechargeRay` and `LayerEnderDragonDeath`.  Both paths are deliberately
//! CPU emitters: the entity phase already owns transformed `EntityVertex` batches, so keeping the
//! fixed-function matrix/vertex recipes here makes the beam and death shards independently
//! testable without introducing renderer state into the model cache.

use glam::{Mat4, Vec3};
use mc_math::{cos, sin, JavaRandom};

use super::EntityVertex;

/// java: `RenderDragon.enderDragonCrystalBeamTextures`.
pub const DRAGON_BEAM_TEXTURE: &str = "textures/entity/endercrystal/endercrystal_beam.png";

/// java: `RenderDragon.enderDragonExplodingTextures`.
pub const DRAGON_EXPLODING_TEXTURE: &str = "textures/entity/enderdragon/dragon_exploding.png";

/// Emit `RenderDragon.drawRechargeRay`'s beam (`:106-145`) as a triangle list.
///
/// Java draws one 18-vertex `GL_TRIANGLE_STRIP` (`begin(5, ...)`): nine columns around the beam,
/// each with a vertex at the dragon end (radius 0.15, black) and one at the crystal end (radius
/// 0.75, white), smooth-shaded between them. The entity phase draws a plain triangle list, so the
/// strip's 16 triangles are written out in strip order, odd ones with their first two vertices
/// swapped as a strip keeps its winding. Submitted as the strip itself, the list drew only every
/// third triangle.
///
/// Both positions are camera-relative and use the same origin that the body renderer receives.
/// The dragon position is translated two blocks upward before the Java yaw/pitch rotations; the
/// crystal position is used only for the direction and length, exactly as the fixed-function
/// implementation does. The alpha test, the lightmap at the dragon's own light and the depth state
/// belong to [`LayerBlend::DragonBeam`](super::LayerBlend::DragonBeam). Checked vertex for vertex
/// against the real renderer by `tests/dragon_beam_java_parity.rs`.
pub fn emit_dragon_beam(
    dragon_position: Vec3,
    crystal_position: Vec3,
    inner_rotation: f32,
    ticks_existed: u32,
    partial_ticks: f32,
    out: &mut Vec<EntityVertex>,
) {
    let phase = inner_rotation + partial_ticks;
    let pulse = sin(phase * 0.2) / 2.0 + 0.5;
    let pulse = (pulse * pulse + pulse) * 0.2;
    let delta = crystal_position - dragon_position;
    // Java narrows these values to float before the square roots/atan2 calls.
    let dx = delta.x;
    let dy = pulse + delta.y - 1.0;
    let dz = delta.z;
    let horizontal = (dx * dx + dz * dz).sqrt();
    let length = (dx * dx + dy * dy + dz * dz).sqrt();
    if !length.is_finite() || length <= f32::EPSILON {
        return;
    }

    let yaw = -dz.atan2(dx).to_degrees() - 90.0;
    let pitch = -horizontal.atan2(dy).to_degrees() - 90.0;
    let base = Mat4::from_translation(dragon_position + Vec3::new(0.0, 2.0, 0.0))
        * Mat4::from_rotation_y(yaw.to_radians())
        * Mat4::from_rotation_x(pitch.to_radians());
    let scroll = -(ticks_existed as f32 + partial_ticks) * 0.01;
    let outer_v = length / 32.0 - (ticks_existed as f32 + partial_ticks) * 0.01;
    let normal = base.transform_vector3(Vec3::Z).normalize_or_zero().into();
    // Strip vertex `2 * column` sits at the dragon end, `2 * column + 1` at the crystal end.
    let strip: [EntityVertex; 18] = core::array::from_fn(|index| {
        let edge = ((index / 2) % 8) as f32;
        let angle = edge * core::f32::consts::TAU / 8.0;
        let radius_x = sin(angle) * 0.75;
        let radius_y = cos(angle) * 0.75;
        let u = edge / 8.0;
        if index.is_multiple_of(2) {
            EntityVertex {
                pos: base
                    .transform_point3(Vec3::new(radius_x * 0.2, radius_y * 0.2, 0.0))
                    .into(),
                uv: [u, outer_v],
                normal,
                color: [0.0, 0.0, 0.0, 1.0],
            }
        } else {
            EntityVertex {
                pos: base
                    .transform_point3(Vec3::new(radius_x, radius_y, length))
                    .into(),
                uv: [u, scroll],
                normal,
                color: EntityVertex::WHITE,
            }
        }
    });
    for triangle in 0..16_usize {
        let [a, b, c] = if triangle.is_multiple_of(2) {
            [triangle, triangle + 1, triangle + 2]
        } else {
            [triangle + 1, triangle, triangle + 2]
        };
        out.extend_from_slice(&[strip[a], strip[b], strip[c]]);
    }
}

/// Emit one frame of the fixed-seed death shard layer.
///
/// Vanilla seeds a fresh `java.util.Random(432)` every frame, then deliberately keeps the matrix
/// rotations cumulative across the fan loop. Each shard is its own 5-vertex `GL_TRIANGLE_FAN`
/// (`begin(6, ...)`, a draw per shard): the entity phase draws a plain triangle list, so every fan
/// is written out as its 3 triangles `(0, 1, 2)`, `(0, 2, 3)`, `(0, 3, 4)`, keeping the winding the
/// layer's back-face culling depends on. Concatenated as fans, the list stitched triangles across
/// shards. The output is untextured; the caller submits it under
/// [`LayerBlend::DragonDeathShards`](super::LayerBlend::DragonDeathShards). Checked against the
/// real `LayerEnderDragonDeath` by `tests/dragon_death_java_parity.rs`.
pub fn emit_dragon_death_shards(
    base: Mat4,
    death_ticks: u32,
    partial_ticks: f32,
    out: &mut Vec<EntityVertex>,
) {
    if death_ticks == 0 {
        return;
    }
    let f = (death_ticks as f32 + partial_ticks) / 200.0;
    let f1 = if f > 0.8 { (f - 0.8) / 0.2 } else { 0.0 };
    let mut random = JavaRandom::new(432);
    let count = (f + f * f) / 2.0 * 60.0;
    let mut matrix = base * Mat4::from_translation(Vec3::new(0.0, -1.0, -2.0));
    // java: `color(255, 255, 255, (int)(255.0F * (1.0F - f1)))`: a byte, so the last frames of the
    // fade reach exactly 0 and draw nothing.
    let center_alpha = f32::from((255.0 * (1.0 - f1)) as i32 as u8) / 255.0;
    let center_color = [1.0, 1.0, 1.0, center_alpha];
    let outer_color = [1.0, 0.0, 1.0, 0.0];
    let mut index = 0.0;
    while index < count {
        matrix *= Mat4::from_rotation_x(random.next_f32() * 360.0_f32.to_radians());
        matrix *= Mat4::from_rotation_y(random.next_f32() * 360.0_f32.to_radians());
        matrix *= Mat4::from_rotation_z(random.next_f32() * 360.0_f32.to_radians());
        matrix *= Mat4::from_rotation_x(random.next_f32() * 360.0_f32.to_radians());
        matrix *= Mat4::from_rotation_y(random.next_f32() * 360.0_f32.to_radians());
        matrix *= Mat4::from_rotation_z(
            random.next_f32() * 360.0_f32.to_radians() + f * 90.0_f32.to_radians(),
        );
        let radius = random.next_f32() * 20.0 + 5.0 + f1 * 10.0;
        let width = random.next_f32() * 2.0 + 1.0 + f1 * 2.0;
        let normal = matrix.transform_vector3(Vec3::Y).normalize_or_zero().into();
        let vertices = [
            Vec3::ZERO,
            Vec3::new(-0.866 * width, radius, -0.5 * width),
            Vec3::new(0.866 * width, radius, -0.5 * width),
            Vec3::new(0.0, radius, width),
            Vec3::new(-0.866 * width, radius, -0.5 * width),
        ];
        let fan: [EntityVertex; 5] = core::array::from_fn(|corner| EntityVertex {
            pos: matrix.transform_point3(vertices[corner]).into(),
            uv: [0.0, 0.0],
            normal,
            color: if corner == 0 {
                center_color
            } else {
                outer_color
            },
        });
        for rim in 1..4 {
            out.extend_from_slice(&[fan[0], fan[rim], fan[rim + 1]]);
        }
        index += 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// java: `drawRechargeRay` is one 18-vertex `GL_TRIANGLE_STRIP` of nine columns. The entity
    /// phase draws a triangle list, so the strip arrives as its 16 triangles: `(i, i+1, i+2)`,
    /// odd ones as `(i+1, i, i+2)`.
    #[test]
    fn beam_is_the_sixteen_triangles_of_vanillas_nine_column_strip() {
        let mut vertices = Vec::new();
        emit_dragon_beam(
            Vec3::ZERO,
            Vec3::new(0.0, 4.0, 8.0),
            17.0,
            20,
            0.5,
            &mut vertices,
        );
        assert_eq!(vertices.len(), 48);
        assert!(vertices
            .iter()
            .all(|vertex| vertex.pos.iter().all(|v| v.is_finite())));
        // Rebuild the strip from the list: triangle 0, then each later triangle's last vertex.
        let mut strip: Vec<EntityVertex> = vertices[..3].to_vec();
        strip.extend(vertices.chunks_exact(3).skip(1).map(|t| t[2]));
        assert_eq!(strip.len(), 18);
        for (i, triangle) in vertices.chunks_exact(3).enumerate() {
            let expected = if i.is_multiple_of(2) {
                [i, i + 1, i + 2]
            } else {
                [i + 1, i, i + 2]
            };
            for (vertex, index) in triangle.iter().zip(expected) {
                assert_eq!(vertex.pos, strip[index].pos, "triangle {i}");
            }
        }
        // Black at the dragon end, white at the crystal end; the ninth column closes the tube.
        assert_eq!(strip[0].color, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(strip[1].color, EntityVertex::WHITE);
        assert_eq!(strip[16].pos, strip[0].pos);
        assert_eq!(strip[17].pos, strip[1].pos);
    }

    /// java: `LayerEnderDragonDeath` draws every shard as its own 5-vertex `GL_TRIANGLE_FAN`;
    /// as a triangle list each is `(0, 1, 2)`, `(0, 2, 3)`, `(0, 3, 4)`.
    #[test]
    fn death_shards_are_empty_before_death_and_grow_afterward() {
        let mut vertices = Vec::new();
        emit_dragon_death_shards(Mat4::IDENTITY, 0, 0.5, &mut vertices);
        assert!(vertices.is_empty());
        emit_dragon_death_shards(Mat4::IDENTITY, 20, 0.5, &mut vertices);
        // f = 0.1025: ceil((f + f * f) / 2 * 60) = 4 shards.
        assert_eq!(vertices.len(), 4 * 9);
        for fan in vertices.chunks_exact(9) {
            let centre = fan[0].pos;
            assert!(fan.chunks_exact(3).all(|t| t[0].pos == centre));
            assert_eq!(fan[2].pos, fan[4].pos, "(0, 1, 2) then (0, 2, 3)");
            assert_eq!(fan[5].pos, fan[7].pos, "(0, 2, 3) then (0, 3, 4)");
            assert_eq!(
                fan[8].pos, fan[1].pos,
                "the fan closes on its first rim vertex"
            );
        }
    }

    /// java: the centre alpha is `(int)(255.0F * (1.0F - f1))`, a byte, so the fade's last frames
    /// draw nothing at all.
    #[test]
    fn death_shard_centre_alpha_is_a_truncated_byte() {
        let alpha = |death_ticks: u32, partial: f32| {
            let mut vertices = Vec::new();
            emit_dragon_death_shards(Mat4::IDENTITY, death_ticks, partial, &mut vertices);
            (vertices[0].color[3] * 255.0).round() as u8
        };
        assert_eq!(alpha(100, 0.0), 255);
        // f = 0.85125, f1 = 0.25625: 255 * 0.74375 = 189.66 -> 189.
        assert_eq!(alpha(170, 0.25), 189);
        // f = 0.9995, f1 = 0.9975: 255 * 0.0025 = 0.64 -> 0, not a faint 0.0025.
        assert_eq!(alpha(199, 0.9), 0);
    }
}
