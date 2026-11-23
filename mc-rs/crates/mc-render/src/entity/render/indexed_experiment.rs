//! Isolated P11b prototype. Compact complete quad-pattern triangle-list batches only.
//! All shader attributes must match bit-for-bit; strips/irregular geometry fall back.
//! This experiment does not move model transforms or animation to the GPU.
use super::{blend_is_strip, DrawBatch, EntityVertex, Gpu};

pub(super) fn logical_count(batch: &DrawBatch) -> usize {
    let stored = (batch.end - batch.start) as usize;
    if batch.indexed {
        stored / 4 * 6
    } else {
        stored
    }
}

pub(super) fn vertex_index(batch: &DrawBatch, logical: usize) -> usize {
    batch.start as usize
        + if batch.indexed {
            logical / 6 * 4 + [0, 1, 2, 0, 2, 3][logical % 6]
        } else {
            logical
        }
}

fn same(a: &EntityVertex, b: &EntityVertex) -> bool {
    // Float equality would collapse signed zero and reject identical NaN payloads.
    bytemuck::bytes_of(a) == bytemuck::bytes_of(b)
}

pub(super) fn compact(
    verts: &mut Vec<EntityVertex>,
    lights: &mut Vec<u32>,
    batches: &mut [DrawBatch],
) {
    assert_eq!(verts.len(), lights.len());
    let mut write = 0;
    for batch in batches {
        let start = batch.start as usize;
        let end = batch.end as usize;
        let indexed = !batch.indexed
            && !blend_is_strip(batch.blend)
            && (end - start).is_multiple_of(6)
            && end <= i32::MAX as usize
            && (start..end).step_by(6).all(|i| {
                same(&verts[i], &verts[i + 3])
                    && same(&verts[i + 2], &verts[i + 4])
                    && lights[i] == lights[i + 3]
                    && lights[i + 2] == lights[i + 4]
            });
        batch.start = write as u32;
        if indexed {
            for read in (start..end).step_by(6) {
                // Copy locals first: input and output can overlap during in-place compaction.
                let quad = [
                    verts[read],
                    verts[read + 1],
                    verts[read + 2],
                    verts[read + 5],
                ];
                let light = [
                    lights[read],
                    lights[read + 1],
                    lights[read + 2],
                    lights[read + 5],
                ];
                verts[write..write + 4].copy_from_slice(&quad);
                lights[write..write + 4].copy_from_slice(&light);
                write += 4;
            }
            batch.indexed = true;
        } else {
            verts.copy_within(start..end, write);
            lights.copy_within(start..end, write);
            write += end - start;
        }
        batch.end = write as u32;
    }
    verts.truncate(write);
    lights.truncate(write);
}

pub(super) fn ensure_indices(
    gpu: &Gpu,
    quads: u32,
    buffer: &mut Option<wgpu::Buffer>,
    capacity: &mut u32,
) -> u64 {
    if quads <= *capacity {
        return 0;
    }
    let count = quads.next_power_of_two().max(64);
    let indices: Vec<u32> = (0..count)
        .flat_map(|q| [0, 1, 2, 0, 2, 3].map(|i| q * 4 + i))
        .collect();
    let bytes = bytemuck::cast_slice(&indices);
    let new = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("P11b static quad indices"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.write_buffer_staged(&new, 0, bytes);
    *buffer = Some(new);
    *capacity = count;
    bytes.len() as u64
}

#[cfg(test)]
mod tests {
    use super::super::{
        append_draw, EntityPhase, LayerBlend, TextureInterner, FULL_BRIGHT_COMBINED_LIGHT,
    };
    use super::*;
    use crate::entity::model::Model;
    use crate::{Camera, GpuOptions, PhaseList, PhaseOrder};
    use glam::{Mat4, Vec3};

    fn expanded(
        v: &[EntityVertex],
        l: &[u32],
        batches: &[DrawBatch],
    ) -> (Vec<EntityVertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut lights = Vec::new();
        for b in batches {
            for i in 0..logical_count(b) {
                let j = vertex_index(b, i);
                verts.push(v[j]);
                lights.push(l[j]);
            }
        }
        (verts, lights)
    }
    fn assert_stream(
        v: &[EntityVertex],
        l: &[u32],
        b: &[DrawBatch],
        expected: &(Vec<EntityVertex>, Vec<u32>),
    ) {
        let actual = expanded(v, l, b);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&actual.0),
            bytemuck::cast_slice::<_, u8>(&expected.0)
        );
        assert_eq!(actual.1, expected.1);
    }
    fn quad() -> [EntityVertex; 6] {
        let four = core::array::from_fn::<_, 4, _>(|i| EntityVertex {
            pos: [
                [-0.8, -0.8, 3.0],
                [0.8, -0.8, 3.0],
                [0.8, 0.8, 3.0],
                [-0.8, 0.8, 3.0],
            ][i],
            uv: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]][i],
            normal: [0.0, 0.0, -1.0],
            color: [0.7, 0.6, 0.9, 0.7],
        });
        [four[0], four[1], four[2], four[0], four[2], four[3]]
    }
    #[test]
    fn indexed_model_stream_is_bit_exact_over_transforms_hierarchy_and_mirrors() {
        let mut model = Model::new(64, 64);
        let root = model
            .part("body", 16, 16)
            .cube(-4., 0., -2., 8, 12, 4, 0.)
            .angles(0.12, -0.33, 0.04)
            .id();
        model
            .child(root, "head", 0, 0)
            .cube(-4., -8., -4., 8, 8, 8, 0.)
            .angles(-0.2, 0.4, 0.)
            .id();
        model
            .part("mirror", 40, 16)
            .mirrored()
            .cube(-1., -2., -2., 4, 12, 4, 0.)
            .at(5., 2., 0.)
            .id();
        let texture = TextureInterner::default().intern("test");
        for frame in 0..256 {
            let mut input = Vec::new();
            model.emit_all(
                1. / 16.,
                Mat4::from_translation(Vec3::new(frame as f32 * 0.1, 0., 4.))
                    * Mat4::from_rotation_y(frame as f32 * 0.071)
                    * Mat4::from_scale(Vec3::new(-1., -1., 1.)),
                &mut input,
            );
            let lights: Vec<_> = (0..input.len())
                .map(|i| [1, 2, 3, 1, 3, 4][i % 6])
                .collect();
            let (mut v, mut l, mut b) = (Vec::new(), Vec::new(), Vec::new());
            append_draw(
                &mut v,
                &mut l,
                &mut b,
                texture,
                LayerBlend::Opaque,
                0,
                &input,
                Some(&lights),
                false,
            );
            let expected = (v.clone(), l.clone());
            compact(&mut v, &mut l, &mut b);
            assert_eq!(v.len() * 3, input.len() * 2);
            assert_stream(&v, &l, &b, &expected);
            compact(&mut v, &mut l, &mut b); // repeated finish cannot compact twice
            assert_stream(&v, &l, &b, &expected);
            let before = b.len();
            append_draw(
                &mut v,
                &mut l,
                &mut b,
                texture,
                LayerBlend::Opaque,
                0,
                &input,
                Some(&lights),
                false,
            );
            assert_eq!(
                b.len(),
                before + 1,
                "new triangle input cannot merge into compact storage"
            );
            let mut twice = expected.clone();
            twice.0.extend_from_slice(&input);
            twice.1.extend_from_slice(&lights);
            compact(&mut v, &mut l, &mut b);
            assert_stream(&v, &l, &b, &twice);
        }
    }
    #[test]
    fn indexed_preserves_all_attributes_strips_irregular_batches_and_float_bits() {
        let texture = TextureInterner::default().intern("test");
        for case in 0..10 {
            let mut input = quad();
            let mut lights = vec![1, 2, 3, 1, 3, 4];
            let mut blend = LayerBlend::Opaque;
            match case {
                0 => {}
                1 => input[3].uv[0] = 0.2,
                2 => input[4].normal[0] = 0.3,
                3 => input[3].color[1] = 0.2,
                4 => lights[3] = 8,
                5 => blend = LayerBlend::FishingLine,
                6 => blend = LayerBlend::Leash,
                7 => {
                    input[0].pos[0] = 0.0;
                    input[3].pos[0] = -0.0;
                }
                8 => {
                    input[0].pos[0] = f32::from_bits(0x7fc01234);
                    input[3].pos[0] = input[0].pos[0];
                }
                9 => {
                    input[0].pos[0] = f32::from_bits(0x7fc01234);
                    input[3].pos[0] = f32::from_bits(0x7fc01235);
                }
                _ => unreachable!(),
            }
            let (mut v, mut l, mut b) = (Vec::new(), Vec::new(), Vec::new());
            // Compact + fallback + compact tests overlapping copies and batch boundaries.
            for (geometry, mode) in [
                (&quad()[..], LayerBlend::Opaque),
                (&input[..], blend),
                (&quad()[..3], LayerBlend::Opaque),
                (&quad()[..], LayerBlend::Opaque),
            ] {
                append_draw(
                    &mut v,
                    &mut l,
                    &mut b,
                    texture,
                    mode,
                    0,
                    geometry,
                    Some(&lights[..geometry.len()]),
                    true,
                );
            }
            let expected = (v.clone(), l.clone());
            compact(&mut v, &mut l, &mut b);
            assert_eq!(b[1].indexed, matches!(case, 0 | 8));
            assert!(!b[2].indexed);
            assert_stream(&v, &l, &b, &expected);
        }
    }

    #[test]
    fn deferred_maps_preserve_indexed_geometry_across_repeated_finish() {
        let gpu = Gpu::headless(32, 32, GpuOptions::default()).expect("GPU required");
        let mut phase = EntityPhase::new(&gpu);
        phase.begin();
        let q = quad();
        phase.push_lit(
            EntityPhase::FALLBACK_TEXTURE,
            FULL_BRIGHT_COMBINED_LIGHT,
            &q,
        );
        let map = phase
            .map_atlas_slot(&gpu, 7, 1, || Some(vec![255u8; 128 * 128 * 4]))
            .expect("map atlas slot");
        assert!(map.opaque);
        for copies in [2, 3] {
            phase.push_map_deferred(
                map.slot,
                LayerBlend::MapTranslucentDepthWrite,
                FULL_BRIGHT_COMBINED_LIGHT,
                &q,
            );
            phase.finish(&gpu);
            assert!(phase.batches.iter().all(|batch| batch.indexed));
            assert_eq!(phase.vertex_count(), copies * q.len());
            let expected = (
                q.repeat(copies),
                vec![FULL_BRIGHT_COMBINED_LIGHT; copies * q.len()],
            );
            assert_stream(
                &phase.verts,
                &phase.vertex_lights,
                &phase.batches,
                &expected,
            );
            phase.finish(&gpu);
            assert_stream(
                &phase.verts,
                &phase.vertex_lights,
                &phase.batches,
                &expected,
            );
        }
        phase.begin();
        assert_eq!(phase.vertex_count(), 0);
    }

    fn pixels(gpu: &mut Gpu, indexed: bool, blend: LayerBlend, deferred: bool) -> Vec<u8> {
        let mut phase = EntityPhase::new(gpu);
        phase.indexed_experiment = indexed;
        phase.set_deferred_overlays(deferred);
        phase.set_diagnostics_enabled(true);
        phase.upload_texture(
            gpu,
            "test",
            &[
                220, 130, 70, 255, 30, 240, 90, 100, 130, 50, 220, 255, 0, 0, 0, 0,
            ],
            2,
            2,
        );
        phase.begin();
        let mut q = quad();
        // Both windings ensure a visible primitive under culled pipelines as well.
        let reverse = [q[5], q[2], q[1], q[5], q[1], q[0]];
        let body: Vec<_> = q.into_iter().chain(reverse).collect();
        phase.push_lit("test", FULL_BRIGHT_COMBINED_LIGHT, &body);
        for v in &mut q {
            v.pos[0] += 0.15;
            v.pos[1] += 0.1;
        }
        let lights = [
            0x00300070, 0x00700030, 0x00f000f0, 0x00300070, 0x00f000f0, 0x00100090,
        ];
        // Both selected windings also exercise culled pipelines visibly.
        let (selected, selected_lights) = if !super::blend_is_strip(blend) {
            let reverse = [q[5], q[2], q[1], q[5], q[1], q[0]];
            let reverse_lights = [
                lights[5], lights[2], lights[1], lights[5], lights[1], lights[0],
            ];
            (
                q.into_iter().chain(reverse).collect::<Vec<_>>(),
                lights.into_iter().chain(reverse_lights).collect::<Vec<_>>(),
            )
        } else {
            (q.to_vec(), lights.to_vec())
        };
        phase.push_blended_lit_with_vertex_lights("test", blend, 0, &selected, &selected_lights);
        // An irregular triangle between quad batches exercises fallback/base-vertex offsets.
        phase.push_blended_lit_separate(
            "test",
            LayerBlend::Translucent,
            FULL_BRIGHT_COMBINED_LIGHT,
            &[],
            &q[..3],
        );
        for v in &mut q {
            v.pos[0] -= 0.3;
            v.pos[1] -= 0.2;
        }
        phase.push_blended_lit(
            "test",
            LayerBlend::OpaqueUnlit,
            FULL_BRIGHT_COMBINED_LIGHT,
            &q,
        );
        phase.finish(gpu);
        let first = phase.diagnostic_stats().unwrap();
        phase.finish(gpu);
        let second = phase.diagnostic_stats().unwrap();
        assert_eq!(
            first.index_upload_bytes, second.index_upload_bytes,
            "indices must not be rewritten every frame"
        );
        assert_eq!(
            phase.vertex_count(),
            if super::blend_is_strip(blend) { 27 } else { 33 }
        );
        if indexed {
            assert!(phase.verts.len() < phase.vertex_count());
            assert!(
                phase
                    .batches
                    .iter()
                    .any(|b| b.indexed && b.blend == LayerBlend::Opaque),
                "opaque must actually use indices"
            );
            if !super::blend_is_strip(blend) {
                assert!(
                    phase.batches.iter().any(|b| b.indexed && b.blend == blend),
                    "selected nonstrip blend must actually use indices: {blend:?}"
                );
            } else {
                assert!(phase
                    .batches
                    .iter()
                    .filter(|b| b.blend == blend)
                    .all(|b| !b.indexed));
            }
        }
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
        let mut camera = Camera::default();
        camera.set_viewport(160, 120);
        camera.update();
        let frame = gpu.acquire().expect("frame");
        phases.render(gpu, frame.view(), &camera, 0.0);
        frame.present(&gpu.queue);
        gpu.read_pixels().expect("pixel readback")
    }
    #[test]
    fn indexed_gpu_matches_unindexed_pixels_for_every_blend_and_deferred_mode() {
        let blends = [
            LayerBlend::Opaque,
            LayerBlend::OpaqueUnlit,
            LayerBlend::OpaqueUnlitCulled,
            LayerBlend::Additive,
            LayerBlend::AdditiveNoDepthWrite,
            LayerBlend::Translucent,
            LayerBlend::TranslucentDepthWrite,
            LayerBlend::MapTranslucentDepthWrite,
            LayerBlend::Glint,
            LayerBlend::DroppedItem,
            LayerBlend::DroppedItemGlint,
            LayerBlend::LivingHurt,
            LayerBlend::LivingHurtTranslucent,
            LayerBlend::LivingHurtGhost,
            LayerBlend::LivingHurtGlint,
            LayerBlend::WhiteFlash,
            LayerBlend::TntFlash,
            LayerBlend::DragonDeathBody,
            LayerBlend::DragonDeathShards,
            LayerBlend::DragonBeam,
            LayerBlend::FishingLine,
            LayerBlend::Leash,
            LayerBlend::GuardianBeam,
        ];
        for samples in [1, 4] {
            let mut gpu = Gpu::headless(
                160,
                120,
                GpuOptions {
                    vsync: false,
                    sample_count: samples,
                    low_power: false,
                },
            )
            .expect("native GPU required; no skip");
            let info = gpu.adapter_info();
            eprintln!("P11_COVERAGE_GPU samples={samples}: {info:?}");
            for blend in blends {
                for deferred in [false, true] {
                    let control = pixels(&mut gpu, false, blend, deferred);
                    let candidate = pixels(&mut gpu, true, blend, deferred);
                    assert!(
                        control
                            .chunks_exact(4)
                            .any(|p| p[0] > 8 || p[1] > 8 || p[2] > 8),
                        "empty fixture"
                    );
                    assert_eq!(
                        control, candidate,
                        "pixel mismatch {blend:?}, deferred={deferred}, samples={samples}"
                    );
                    if let Some(directory) = std::env::var_os("MC_INDEXED_TEST_OUTPUT") {
                        let directory = std::path::PathBuf::from(directory);
                        std::fs::create_dir_all(&directory).unwrap();
                        for (name, pixels) in [("control", &control), ("indexed", &candidate)] {
                            let path = directory.join(format!(
                                "{blend:?}-defer{deferred}-msaa{samples}-{name}.png"
                            ));
                            image::save_buffer(path, pixels, 160, 120, image::ColorType::Rgba8)
                                .unwrap();
                        }
                    }
                }
            }
        }
    }
}
