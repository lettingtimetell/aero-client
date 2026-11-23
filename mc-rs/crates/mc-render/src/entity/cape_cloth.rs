//! Bounded, connected cape deformation. No simulation state or per-node allocation is kept.
//!
//! The native cape box is subdivided into a closed 8-by-2 surface. Its shoulder row stays
//! attached to the posed cape pivot; travelling bends grow toward the hem. Each length segment
//! keeps its rest length, and bends remain behind the undeformed cape instead of entering the
//! player's body. The original box's six UV rectangles are subdivided without repacking them.

use glam::{Mat4, Vec3};

use super::layers::{cape_angles, CapeState};
use super::model::{ModelBox, ModelPart};
use super::EntityVertex;

/// The motion profile of a connected cape surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CapeClothStyle {
    /// Regular travelling ripples with a little side-to-side motion.
    #[default]
    Wavy,
    /// Slower, longer bends with a soft trailing hem.
    Flowing,
    /// Faster, irregular ripples and a stronger sideways flutter.
    Windy,
    /// Broad, slow folds with a gently bowed centre and little sideways motion.
    Draped,
}

/// Already-sampled animation inputs; tick interpolation belongs to the client.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapeCloth {
    /// Shape and motion profile.
    pub style: CapeClothStyle,
    /// Interpolated game time. Double precision keeps long sessions smooth.
    pub time_seconds: f64,
    /// Stable per-player phase, in radians.
    pub phase: f32,
    /// Deformation amount, clamped to `0..=1`.
    pub strength: f32,
    /// Animation speed, clamped to `0.25..=2`.
    pub speed: f32,
    /// Resistance to bending, clamped to `0..=1`; one is rigid.
    pub stiffness: f32,
}

impl Default for CapeCloth {
    fn default() -> Self {
        Self {
            style: CapeClothStyle::Wavy,
            time_seconds: 0.0,
            phase: 0.0,
            strength: 0.6,
            speed: 1.0,
            stiffness: 0.5,
        }
    }
}

const WIDTH_SEGMENTS: usize = 2;
const LENGTH_SEGMENTS: usize = 8;
const ROW_WIDTH: usize = WIDTH_SEGMENTS + 1;
const SHEET_NODES: usize = ROW_WIDTH * (LENGTH_SEGMENTS + 1);
const NODE_COUNT: usize = SHEET_NODES * 2;
/// Fixed geometry budget: front/back, both sides, and both closed ends.
pub const CAPE_CLOTH_VERTICES: usize =
    (2 * WIDTH_SEGMENTS * LENGTH_SEGMENTS + 2 * LENGTH_SEGMENTS + 2 * WIDTH_SEGMENTS) * 6;

fn finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn finite64(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// Reduce time before conversion to f32, rather than passing a growing angle to the sine table.
fn wave(time: f64, rate: f64, phase: f32) -> f32 {
    let tau = core::f64::consts::TAU;
    let phase = f64::from(finite(phase, 0.0)).rem_euclid(tau);
    let angle = (finite64(time).rem_euclid(tau / rate) * rate + phase).rem_euclid(tau);
    mc_math::sin(angle as f32)
}

fn root_matrix(state: &CapeState) -> Mat4 {
    // Keep LayerCape's transform and order. Only the optional cloth branch limits extreme
    // chase offsets (e.g. teleports), which otherwise turn the whole cape over its shoulders.
    let (x, z, y) = cape_angles(state);
    Mat4::from_translation(Vec3::new(0.0, 0.0, 0.125))
        * Mat4::from_rotation_x(finite(x, 6.0).clamp(0.0, 75.0).to_radians())
        * Mat4::from_rotation_z(finite(z, 0.0).clamp(-15.0, 15.0).to_radians())
        * Mat4::from_rotation_y(finite(y, 0.0).clamp(-15.0, 15.0).to_radians())
        * Mat4::from_rotation_y(core::f32::consts::PI)
}

fn node(x: usize, y: usize, z: usize) -> usize {
    z * SHEET_NODES + y * ROW_WIDTH + x
}

/// Chase is already interpolated and decays by the vanilla tick update after movement stops.
/// A smooth deadzone eliminates its tiny residual without adding another delayed simulation.
fn movement_weight(dx: f32, dy: f32, dz: f32) -> f32 {
    let motion = dx.hypot(dz) + dy.abs() * 0.5;
    let t = ((motion - 0.0025) / (0.25 - 0.0025)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn nodes(cube: &ModelBox, state: &CapeState, cloth: CapeCloth) -> [Vec3; NODE_COUNT] {
    let origin = Vec3::from(cube.corners[0]);
    let width = cube.corners[1][0] - origin.x;
    let length = cube.corners[3][1] - origin.y;
    let thickness = cube.corners[4][2] - origin.z;
    let segment = length / LENGTH_SEGMENTS as f32;
    let flexibility = finite(cloth.strength, 0.0).clamp(0.0, 1.0)
        * (1.0 - finite(cloth.stiffness, 1.0).clamp(0.0, 1.0));
    let speed = f64::from(finite(cloth.speed, 1.0).clamp(0.25, 2.0));
    let dx = finite64(state.chase_dx).clamp(-2.0, 2.0) as f32;
    let dy = finite64(state.chase_dy).clamp(-2.0, 2.0) as f32;
    let dz = finite64(state.chase_dz).clamp(-2.0, 2.0) as f32;
    let movement = movement_weight(dx, dy, dz);
    let yaw = finite(state.body_yaw, 0.0).to_radians();
    let lateral = (-dx * mc_math::cos(yaw) - dz * mc_math::sin(yaw)).clamp(-0.5, 0.5);
    // There is no ambient wind floor: even Windy hangs still once chase motion settles.
    // Full movement keeps the former maximum amplitude, but longer waves make the fixed
    // eight-segment mesh read as continuous fabric instead of a succession of sharp kinks.
    let amplitude = 3.6 * flexibility * movement;
    let mut result = [Vec3::ZERO; NODE_COUNT];
    for column in 0..=WIDTH_SEGMENTS {
        let u = column as f32 / WIDTH_SEGMENTS as f32 * 2.0 - 1.0;
        let x0 = origin.x + width * column as f32 / WIDTH_SEGMENTS as f32;
        let mut previous = Vec3::new(x0, origin.y, origin.z);
        for row in 0..=LENGTH_SEGMENTS {
            let v = row as f32 / LENGTH_SEGMENTS as f32;
            if row != 0 {
                let phase = finite(cloth.phase, 0.0);
                let (back, side) = match cloth.style {
                    CapeClothStyle::Wavy => (
                        1.4 + 0.65
                            * wave(cloth.time_seconds, 4.0 * speed, phase - v * 5.4 + u * 0.2),
                        0.16 * wave(cloth.time_seconds, 2.0 * speed, phase - v * 2.8),
                    ),
                    CapeClothStyle::Flowing => (
                        1.9 + 0.75 * wave(cloth.time_seconds, 2.2 * speed, phase - v * 3.2),
                        0.3 * wave(cloth.time_seconds, 1.2 * speed, phase - v * 2.0),
                    ),
                    CapeClothStyle::Windy => (
                        1.6 + 0.55
                            * wave(cloth.time_seconds, 5.6 * speed, phase - v * 6.8 + u * 0.3)
                            + 0.18
                                * wave(cloth.time_seconds, 8.0 * speed, phase - v * 8.8 - u * 0.2),
                        0.45 * wave(cloth.time_seconds, 3.8 * speed, phase - v * 4.0 + u * 0.2),
                    ),
                    CapeClothStyle::Draped => (
                        1.1 + 0.55 * (1.0 - u * u)
                            + 0.3 * wave(cloth.time_seconds, 1.65 * speed, phase - v * 2.6),
                        0.1 * wave(cloth.time_seconds, 1.0 * speed, phase - v * 1.8),
                    ),
                };
                // Ease out of the shoulders and toward the hem, distributing curvature
                // along the whole surface instead of concentrating it in the final rows.
                let envelope = amplitude * v * v * (3.0 - 2.0 * v);
                let desired_x = x0 + envelope * (side + lateral);
                // Local -Z becomes the player's back after LayerCape's trailing Y rotation.
                let desired_z = origin.z - envelope * back;
                let mut bend = Vec3::new(desired_x - previous.x, 0.0, desired_z - previous.z);
                let max_bend = segment.abs() * 0.85;
                let magnitude = bend.length();
                if magnitude > max_bend {
                    bend *= max_bend / magnitude;
                }
                // An inextensible column: increasing bend shortens its vertical projection.
                let down = (segment * segment - bend.length_squared()).max(0.0).sqrt();
                previous += Vec3::new(bend.x, down, bend.z);
            }
            result[node(column, row, 0)] = previous;
            result[node(column, row, 1)] = previous + Vec3::new(0.0, 0.0, thickness);
        }
    }
    result
}

// ModelBox's corner order; mirrored boxes reverse face indices, not this logical grid.
const CORNERS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [2, 0, 0],
    [2, 8, 0],
    [0, 8, 0],
    [0, 0, 1],
    [2, 0, 1],
    [2, 8, 1],
    [0, 8, 1],
];
const FACE_SEGMENTS: [(usize, usize); 6] = [(1, 8), (1, 8), (2, 1), (2, 1), (2, 8), (2, 8)];

fn face_node(cube: &ModelBox, face: usize, u: usize, v: usize) -> usize {
    let corners = cube.corner_index[face];
    let a = CORNERS[usize::from(corners[0])];
    let b = CORNERS[usize::from(corners[1])];
    let d = CORNERS[usize::from(corners[3])];
    let (nu, nv) = FACE_SEGMENTS[face];
    let c: [usize; 3] = core::array::from_fn(|axis| {
        (a[axis]
            + (b[axis] - a[axis]) * u as i32 / nu as i32
            + (d[axis] - a[axis]) * v as i32 / nv as i32) as usize
    });
    node(c[0], c[1], c[2])
}

pub(super) fn emit(
    part: &ModelPart,
    state: &CapeState,
    cloth: CapeCloth,
    scale: f32,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    if part.is_hidden || !part.show_model {
        return;
    }
    let matrix = base * root_matrix(state) * part.local_matrix(scale);
    for cube in &part.cubes {
        let local = nodes(cube, state, cloth);
        let world = local.map(|p| matrix.transform_point3(p * scale));
        for (face, &(nu, nv)) in FACE_SEGMENTS.iter().enumerate() {
            let quad = &cube.quads[face];
            let vertex = |u: usize, v: usize| {
                let pos = world[face_node(cube, face, u, v)];
                let du = world[face_node(cube, face, (u + 1).min(nu), v)]
                    - world[face_node(cube, face, u.saturating_sub(1), v)];
                let dv = world[face_node(cube, face, u, (v + 1).min(nv))]
                    - world[face_node(cube, face, u, v.saturating_sub(1))];
                let (a, b) = (u as f32 / nu as f32, v as f32 / nv as f32);
                let uv = core::array::from_fn(|axis| {
                    let top = quad.verts[0].uv[axis] * (1.0 - a) + quad.verts[1].uv[axis] * a;
                    let bottom = quad.verts[3].uv[axis] * (1.0 - a) + quad.verts[2].uv[axis] * a;
                    top * (1.0 - b) + bottom * b
                });
                EntityVertex {
                    pos: pos.into(),
                    uv,
                    normal: du.cross(dv).normalize_or_zero().into(),
                    color: EntityVertex::WHITE,
                }
            };
            for v in 0..nv {
                for u in 0..nu {
                    let q = [
                        vertex(u, v),
                        vertex(u + 1, v),
                        vertex(u + 1, v + 1),
                        vertex(u, v + 1),
                    ];
                    out.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::layers::{cape_matrix, emit_cape};
    use super::super::models::biped::PlayerModel;
    use super::*;

    const STYLES: [CapeClothStyle; 4] = [
        CapeClothStyle::Wavy,
        CapeClothStyle::Flowing,
        CapeClothStyle::Windy,
        CapeClothStyle::Draped,
    ];

    fn mesh(cloth: CapeCloth) -> Vec<EntityVertex> {
        mesh_for_state(
            cloth,
            CapeState {
                chase_dx: 0.08,
                chase_dy: 0.04,
                chase_dz: -0.3,
                ..Default::default()
            },
        )
    }

    fn mesh_for_state(cloth: CapeCloth, state: CapeState) -> Vec<EntityVertex> {
        let mut model = PlayerModel::new(0.0, false);
        let mut out = Vec::new();
        emit_cape(
            &mut model,
            &CapeState {
                cloth: Some(cloth),
                ..state
            },
            1.0 / 16.0,
            Mat4::IDENTITY,
            &mut out,
        );
        out
    }

    #[test]
    fn disabled_is_the_exact_vanilla_cape_stream() {
        let mut model = PlayerModel::new(0.0, false);
        let state = CapeState {
            chase_dx: 0.07,
            chase_dz: -0.12,
            sneaking: true,
            ..Default::default()
        };
        let cape = model.wear.cape;
        model.biped.model.get_mut(cape).rotation_point[1] = 2.0;
        let mut actual = Vec::new();
        emit_cape(&mut model, &state, 0.0625, Mat4::IDENTITY, &mut actual);
        let mut expected = Vec::new();
        model
            .biped
            .model
            .emit_part(cape, 0.0625, cape_matrix(&state), &mut expected);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&actual),
            bytemuck::cast_slice::<_, u8>(&expected)
        );
        assert_eq!(actual.len(), 36);
    }

    #[test]
    fn connected_mesh_is_closed_with_valid_normals_uvs_and_winding() {
        use std::collections::HashMap;
        for (style, time_seconds) in STYLES.into_iter().flat_map(|style| {
            [0.0, 0.35, 0.7, 1.1, 1.25, 2.4]
                .into_iter()
                .map(move |time| (style, time))
        }) {
            let vertices = mesh(CapeCloth {
                style,
                time_seconds,
                strength: 1.0,
                stiffness: 0.0,
                ..Default::default()
            });
            assert_eq!(vertices.len(), CAPE_CLOTH_VERTICES);
            let mut edges = HashMap::new();
            for triangle in vertices.chunks_exact(3) {
                let face = (Vec3::from(triangle[1].pos) - Vec3::from(triangle[0].pos))
                    .cross(Vec3::from(triangle[2].pos) - Vec3::from(triangle[0].pos));
                assert!(face.length_squared() > 1e-12);
                for i in 0..3 {
                    let vertex = &triangle[i];
                    assert!(vertex
                        .pos
                        .iter()
                        .chain(vertex.normal.iter())
                        .chain(vertex.uv.iter())
                        .all(|x| x.is_finite()));
                    assert!(vertex.uv[0] >= 0.0 && vertex.uv[0] <= 22.0 / 64.0);
                    assert!(vertex.uv[1] >= 0.0 && vertex.uv[1] <= 17.0 / 32.0);
                    assert!((Vec3::from(vertex.normal).length() - 1.0).abs() < 1e-5);
                    assert!(face.dot(Vec3::from(vertex.normal)) > 0.0);
                    let mut edge = [
                        vertex.pos.map(f32::to_bits),
                        triangle[(i + 1) % 3].pos.map(f32::to_bits),
                    ];
                    edge.sort();
                    *edges.entry(edge).or_insert(0) += 1;
                }
            }
            assert!(
                edges.values().all(|&count| count == 2),
                "every seam must have exactly two incident faces"
            );
        }
    }

    #[test]
    fn rigid_surface_preserves_native_face_uvs_and_sneaking_pivot() {
        let mut model = PlayerModel::new(0.0, false);
        let cape = model.wear.cape;
        model.biped.model.get_mut(cape).rotation_point[1] = 2.0;
        let state = CapeState {
            sneaking: true,
            cloth: Some(CapeCloth {
                stiffness: 1.0,
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut actual = Vec::new();
        emit_cape(&mut model, &state, 0.0625, Mat4::IDENTITY, &mut actual);
        let part = model.biped.model.get(cape);
        let matrix = cape_matrix(&state) * part.local_matrix(0.0625);
        let cube = &part.cubes[0];
        let mut offset = 0;
        for (face, &(nu, nv)) in FACE_SEGMENTS.iter().enumerate() {
            // The outer corners of each subdivided face must retain the original atlas
            // coordinates and the posed native cape's positions, including its sneak pivot.
            let corners = [
                offset,
                offset + (nu - 1) * 6 + 1,
                offset + (nu * nv - 1) * 6 + 2,
                offset + (nv - 1) * nu * 6 + 5,
            ];
            for (corner, index) in corners.into_iter().enumerate() {
                let reference = &cube.quads[face].verts[corner];
                assert_eq!(actual[index].uv, reference.uv);
                let expected = matrix.transform_point3(Vec3::from(reference.pos) * 0.0625);
                assert!((Vec3::from(actual[index].pos) - expected).length() < 1e-6);
            }
            offset += nu * nv * 6;
        }
    }

    #[test]
    fn shoulders_are_fixed_segments_keep_length_and_fabric_stays_behind() {
        let model = PlayerModel::new(0.0, false);
        let cube = &model.biped.model.get(model.wear.cape).cubes[0];
        let state = CapeState {
            chase_dx: 0.15,
            chase_dz: -0.25,
            ..Default::default()
        };
        for style in STYLES {
            for time in [0.0, 1.0, 123456.125] {
                let p = nodes(
                    cube,
                    &state,
                    CapeCloth {
                        style,
                        time_seconds: time,
                        strength: 1.0,
                        stiffness: 0.0,
                        ..Default::default()
                    },
                );
                for x in 0..=WIDTH_SEGMENTS {
                    assert_eq!(
                        p[node(x, 0, 0)],
                        Vec3::new(-5.0 + x as f32 * 5.0, 0.0, -1.0)
                    );
                    for y in 1..=LENGTH_SEGMENTS {
                        assert!(
                            ((p[node(x, y, 0)] - p[node(x, y - 1, 0)]).length() - 2.0).abs() < 1e-5
                        );
                        assert!(p[node(x, y, 0)].z <= -1.0);
                        assert!(p[node(x, y, 0)].y > p[node(x, y - 1, 0)].y);
                    }
                }
            }
        }
    }

    #[test]
    fn styles_time_stiffness_and_chase_change_the_connected_surface() {
        let model = PlayerModel::new(0.0, false);
        let cube = &model.biped.model.get(model.wear.cape).cubes[0];
        let state = CapeState {
            chase_dx: 0.08,
            chase_dz: -0.12,
            ..Default::default()
        };
        let cloth = CapeCloth {
            time_seconds: 0.7,
            ..Default::default()
        };
        let plain = nodes(cube, &state, cloth);
        assert_ne!(
            plain,
            nodes(
                cube,
                &state,
                CapeCloth {
                    style: CapeClothStyle::Flowing,
                    ..cloth
                }
            )
        );
        assert_ne!(
            plain,
            nodes(
                cube,
                &state,
                CapeCloth {
                    style: CapeClothStyle::Windy,
                    ..cloth
                }
            )
        );
        assert_ne!(
            plain,
            nodes(
                cube,
                &state,
                CapeCloth {
                    style: CapeClothStyle::Draped,
                    ..cloth
                }
            )
        );
        assert_ne!(
            plain,
            nodes(
                cube,
                &state,
                CapeCloth {
                    time_seconds: 0.9,
                    ..cloth
                }
            )
        );
        assert_ne!(
            plain,
            nodes(
                cube,
                &CapeState {
                    chase_dz: -0.3,
                    ..state
                },
                cloth
            )
        );
        let rigid = nodes(
            cube,
            &state,
            CapeCloth {
                stiffness: 1.0,
                ..cloth
            },
        );
        assert_eq!(rigid[node(1, 8, 0)], Vec3::new(0.0, 16.0, -1.0));
        assert!(plain[node(1, 8, 0)].z < rigid[node(1, 8, 0)].z);
    }

    #[test]
    fn all_styles_stop_waving_at_rest_and_with_tiny_chase_residual() {
        for style in STYLES {
            for state in [
                CapeState::default(),
                CapeState {
                    chase_dx: 0.001,
                    chase_dy: 0.001,
                    chase_dz: -0.001,
                    ..Default::default()
                },
            ] {
                let cloth = CapeCloth {
                    style,
                    strength: 1.0,
                    stiffness: 0.0,
                    ..Default::default()
                };
                let settled = mesh_for_state(cloth, state);
                for time_seconds in [0.35, 1.1, 100_000.125] {
                    let later = mesh_for_state(
                        CapeCloth {
                            time_seconds,
                            phase: 1.3,
                            ..cloth
                        },
                        state,
                    );
                    assert_eq!(
                        bytemuck::cast_slice::<_, u8>(&settled),
                        bytemuck::cast_slice::<_, u8>(&later),
                        "{style:?} must not flutter at rest"
                    );
                }
            }
        }
    }

    #[test]
    fn interpolated_chase_has_a_smooth_onset_and_settles_without_extra_history() {
        assert_eq!(movement_weight(0.0, 0.0, 0.0), 0.0);
        assert_eq!(movement_weight(0.0025, 0.0, 0.0), 0.0);
        assert!(movement_weight(0.0026, 0.0, 0.0) < 0.000_001);
        assert_eq!(movement_weight(0.25, 0.0, 0.0), 1.0);
        assert_eq!(
            movement_weight(0.0, 0.5, 0.0),
            1.0,
            "vertical movement also excites the cloth"
        );
        let mut residual = 0.4;
        let mut previous = 1.0;
        for _ in 0..24 {
            let weight = movement_weight(0.0, 0.0, residual);
            assert!(weight <= previous && weight >= 0.0);
            previous = weight;
            residual *= 0.75; // EntityPlayer's existing chase decay.
        }
        assert_eq!(
            previous, 0.0,
            "the cloth fully settles after movement stops"
        );

        let model = PlayerModel::new(0.0, false);
        let cube = &model.biped.model.get(model.wear.cape).cubes[0];
        for style in STYLES {
            let cloth = CapeCloth {
                style,
                time_seconds: 0.7,
                ..Default::default()
            };
            let before = nodes(
                cube,
                &CapeState {
                    chase_dz: -0.12,
                    ..Default::default()
                },
                cloth,
            );
            let after = nodes(
                cube,
                &CapeState {
                    chase_dz: -0.1201,
                    ..Default::default()
                },
                cloth,
            );
            assert!(
                before
                    .iter()
                    .zip(after)
                    .all(|(a, b)| (*a - b).length() < 0.01),
                "{style:?}: interpolated movement must not snap"
            );
        }
    }

    #[test]
    fn malformed_controls_and_long_sessions_stay_finite_and_smooth() {
        let vertices = mesh(CapeCloth {
            time_seconds: f64::INFINITY,
            phase: f32::NAN,
            strength: f32::NAN,
            speed: f32::NAN,
            stiffness: f32::NAN,
            ..Default::default()
        });
        assert!(vertices.iter().all(|v| v
            .pos
            .iter()
            .chain(v.normal.iter())
            .all(|x| x.is_finite())));
        let a = wave(86_400.0 * 100.0, 4.2, 0.3);
        let b = wave(86_400.0 * 100.0 + 0.001, 4.2, 0.3);
        assert!((a - b).abs() < 0.005 && a != b);
        let state = CapeState {
            chase_dx: f64::MAX,
            chase_dy: f64::MAX,
            chase_dz: f64::MAX,
            ..Default::default()
        };
        assert!(root_matrix(&state).is_finite());
    }
}
