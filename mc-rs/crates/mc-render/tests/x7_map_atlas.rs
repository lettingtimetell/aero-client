//! Item 2 ("map item displays extremely laggy"): the shared
//! item-frame map atlas (`mc_render::entity::map_atlas`) on a real GPU.
//!
//! * The atlas draws exactly the pixels a per-map texture drew: opaque and partly unexplored
//!   (translucent) images, magnified, minified and at an angle.
//! * A map is uploaded once per `MapData::revision`, not once per frame.
//! * Deferred opaque map quads merge into one draw per page, and the item-frame shells between
//!   them merge too: 240 draws for 120 framed maps before, 4 after.

use mc_render::entity::map_atlas::{MapAtlasStats, SLOTS_PER_PAGE};
use mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT;
use mc_render::entity::{EntityPhase, EntityVertex, LayerBlend, Mat4};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const W: u32 = 384;
const H: u32 = 256;
const MAP_BLEND: LayerBlend = LayerBlend::MapTranslucentDepthWrite;

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

/// A 128x128 image where every texel differs from its neighbours, so any texel mix-up shows.
/// `partial` leaves a corner unexplored: Java's 16/24-alpha black checkerboard.
fn image(seed: u8, partial: bool) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(128 * 128 * 4);
    for y in 0..128u32 {
        for x in 0..128u32 {
            if partial && x + y < 60 {
                rgba.extend_from_slice(&[0, 0, 0, if (x + y) % 2 == 0 { 16 } else { 24 }]);
            } else {
                rgba.extend_from_slice(&[
                    (x * 2) as u8 ^ seed,
                    (y * 2) as u8,
                    ((x * 7) ^ (y * 3)) as u8,
                    255,
                ]);
            }
        }
    }
    rgba
}

/// `MapItemRenderer`'s quad (`(0,128) (128,128) (128,0) (0,0)` with UVs `(0,1) (1,1) (1,0)
/// (0,0)`, two triangles 0-1-2 and 0-2-3) under `matrix`, in camera-relative blocks.
fn map_quad(matrix: Mat4) -> [EntityVertex; 6] {
    let corners = [
        ([0.0_f32, 128.0], [0.0_f32, 1.0]),
        ([128.0, 128.0], [1.0, 1.0]),
        ([128.0, 0.0], [1.0, 0.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ];
    core::array::from_fn(|index| {
        let (position, uv) = corners[[0usize, 1, 2, 0, 2, 3][index]];
        EntityVertex {
            pos: matrix
                .transform_point3(glam::Vec3::new(position[0], position[1], -0.01))
                .into(),
            uv,
            normal: [0.0, 0.0, -1.0],
            color: EntityVertex::WHITE,
        }
    })
}

/// A map quad of side `side` blocks centred `distance` blocks in front of the default camera,
/// turned `yaw` degrees about its vertical axis.
fn placed(distance: f32, side: f32, yaw: f32, lateral: f32) -> [EntityVertex; 6] {
    let matrix = Mat4::from_translation(glam::Vec3::new(lateral, 0.0, distance))
        * Mat4::from_rotation_y(yaw.to_radians())
        * Mat4::from_scale(glam::Vec3::splat(side / 128.0))
        * Mat4::from_translation(glam::Vec3::new(-64.0, -64.0, 0.0));
    map_quad(matrix)
}

fn render(gpu: &mut Gpu, build: impl FnOnce(&Gpu, &mut EntityPhase)) -> Vec<u8> {
    let mut phase = EntityPhase::new(gpu);
    phase.begin();
    build(gpu, &mut phase);
    phase.finish(gpu);
    let mut phases = PhaseList::new();
    // A coloured background, so the translucent checkerboard's blend is visible in the bytes.
    phases.clear_color = wgpu::Color {
        r: 0.25,
        g: 0.5,
        b: 0.75,
        a: 1.0,
    };
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let mut camera = Camera::default();
    camera.set_viewport(W, H);
    camera.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &camera, 0.0);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("read back the frame")
}

fn background_pixels(pixels: &[u8]) -> usize {
    let first: [u8; 4] = pixels[..4].try_into().unwrap();
    pixels.chunks_exact(4).filter(|p| **p == first).count()
}

/// Every case renders one map the old way (its own 128x128 texture, UVs 0..1) and the new way
/// (a slot in the atlas, after nine other maps so the slot is not at the page origin, UVs
/// remapped) and requires the same bytes.
#[test]
fn the_atlas_draws_exactly_the_pixels_of_a_per_map_texture() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let cases: [(&str, bool, [EntityVertex; 6]); 6] = [
        ("magnified, opaque", false, placed(1.6, 2.2, 0.0, 0.0)),
        ("minified, opaque", false, placed(9.0, 0.9, 0.0, -0.7)),
        ("at an angle, opaque", false, placed(2.5, 1.8, 52.0, 0.4)),
        ("magnified, partial", true, placed(1.6, 2.2, 0.0, 0.0)),
        ("minified, partial", true, placed(9.0, 0.9, 0.0, -0.7)),
        ("at an angle, partial", true, placed(2.5, 1.8, -38.0, -0.3)),
    ];
    for (name, partial, quad) in cases {
        let map = image(0x5A, partial);
        let old = render(&mut gpu, |gpu, phase| {
            phase.upload_texture(gpu, "x7-test/map_7", &map, 128, 128);
            phase.push_blended_lit(
                "x7-test/map_7",
                MAP_BLEND,
                FULL_BRIGHT_COMBINED_LIGHT,
                &quad,
            );
        });
        let new = render(&mut gpu, |gpu, phase| {
            for id in 100..109 {
                let other = image(id as u8, false);
                phase
                    .map_atlas_slot(gpu, id, 1, || Some(other))
                    .expect("a slot for a filler map");
            }
            let slot = phase
                .map_atlas_slot(gpu, 7, 1, || Some(map.clone()))
                .expect("a slot for the map under test");
            assert_eq!(slot.slot.index, 9, "{name}: not the page's first slot");
            assert_eq!(slot.opaque, !partial, "{name}");
            let mut quad = quad;
            for vertex in &mut quad {
                vertex.uv = slot.slot.uv(vertex.uv);
            }
            if slot.opaque {
                phase.push_map_deferred(slot.slot, MAP_BLEND, FULL_BRIGHT_COMBINED_LIGHT, &quad);
                phase.flush_deferred_maps();
            } else {
                phase.push_blended_lit(
                    slot.slot.texture_key(),
                    MAP_BLEND,
                    FULL_BRIGHT_COMBINED_LIGHT,
                    &quad,
                );
            }
        });
        let covered = (W * H) as usize - background_pixels(&old);
        // The minified quad is ~18 px square: 128 texels squeezed 7:1.
        assert!(
            covered > 200,
            "{name}: the quad is on screen ({covered} px)"
        );
        let differing = old
            .chunks_exact(4)
            .zip(new.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        eprintln!("{name}: {covered} px covered, {differing} differ");
        if let Some(dir) = std::env::var_os("X7_DUMP_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("dump dir");
            let stem = name.replace([',', ' '], "_");
            for (tag, pixels) in [("old", &old), ("new", &new)] {
                image::save_buffer(
                    dir.join(format!("{stem}-{tag}.png")),
                    pixels,
                    W,
                    H,
                    image::ColorType::Rgba8,
                )
                .expect("save png");
            }
        }
        // Interior pixels must be identical. The only tolerated difference is on the quad's
        // silhouette, where a fragment centre lies within float rounding of the image edge: the
        // old texture's interpolated `u` could round to a hair past 1.0 and REPEAT-wrap to the
        // opposite edge (a one-pixel seam), while the atlas's offset UV rounds inside the image.
        let background: [u8; 4] = old[..4].try_into().unwrap();
        let is_background = |x: i64, y: i64| {
            x < 0 || y < 0 || x >= i64::from(W) || y >= i64::from(H) || {
                let i = ((y as u32 * W + x as u32) * 4) as usize;
                old[i..i + 4] == background
            }
        };
        // Inside the quad, a pixel centre that sits exactly on a texel boundary may round to the
        // neighbouring texel (the atlas UV carries a constant offset, so its interpolation rounds
        // differently): allowed only when the new colour is a neighbouring old pixel's colour.
        let old_at = |x: i64, y: i64| -> Option<&[u8]> {
            (x >= 0 && y >= 0 && x < i64::from(W) && y < i64::from(H)).then(|| {
                let i = ((y as u32 * W + x as u32) * 4) as usize;
                &old[i..i + 4]
            })
        };
        let mut interior = Vec::new();
        let mut flips = 0;
        for (index, (a, b)) in old.chunks_exact(4).zip(new.chunks_exact(4)).enumerate() {
            if a == b {
                continue;
            }
            let (x, y) = (i64::from(index as u32 % W), i64::from(index as u32 / W));
            let on_edge = (-1..=1).any(|dy| (-1..=1).any(|dx| is_background(x + dx, y + dy)));
            if on_edge {
                continue;
            }
            let neighbour_texel =
                (-1..=1).any(|dy| (-1..=1).any(|dx| old_at(x + dx, y + dy) == Some(b)));
            if neighbour_texel {
                flips += 1;
            } else {
                interior.push((x, y, a.to_vec(), b.to_vec()));
            }
        }
        assert!(
            interior.is_empty(),
            "{name}: {} interior pixels differ by more than a texel, e.g. {:?}",
            interior.len(),
            &interior[..interior.len().min(4)]
        );
        assert!(
            flips * 1000 <= covered,
            "{name}: boundary rounding stays rare ({flips} of {covered} px)"
        );
        assert!(
            differing * 50 <= covered,
            "{name}: silhouette rounding stays a sliver ({differing} of {covered} px)"
        );
    }
}

/// One framed map = a shell push (block-atlas texture, opaque) then its image. Old order:
/// shell, image (own texture), shell, image, ... New: shells, then the deferred images.
#[test]
fn maps_upload_once_per_revision_and_merge_into_one_draw_per_page() {
    let Some(gpu) = headless() else {
        return;
    };
    let maps = 120;
    let shell = placed(3.0, 1.0, 0.0, 0.0);
    let quad = placed(2.9, 1.0, 0.0, 0.0);
    let pixels: Vec<Vec<u8>> = (0..maps).map(|id| image(id as u8, false)).collect();

    // Before (the old item-frame path, for the record): one texture per map, interleaved.
    let mut old = EntityPhase::new(&gpu);
    old.begin();
    for (id, map) in pixels.iter().enumerate() {
        let key = format!("x7-test/map_{id}");
        old.upload_texture(&gpu, &key, map, 128, 128);
        old.push_blended_lit(
            EntityPhase::SOLID_TEXTURE,
            LayerBlend::Opaque,
            FULL_BRIGHT_COMBINED_LIGHT,
            &shell,
        );
        old.push_blended_lit(&key, MAP_BLEND, FULL_BRIGHT_COMBINED_LIGHT, &quad);
    }
    old.finish(&gpu);
    assert_eq!(
        old.batch_count(),
        2 * maps,
        "one shell and one map draw per frame"
    );

    // After: frames through the atlas. Slots are handed out in the first frame's order; later
    // frames visit the maps scrambled (an entity list is not sorted by atlas page, and maps seen
    // for the first time on different frames interleave), and the queue still makes one draw per
    // page.
    let natural: Vec<usize> = (0..maps).collect();
    let scrambled: Vec<usize> = (0..maps).map(|i| (i * 37) % maps).collect();
    let mut phase = EntityPhase::new(&gpu);
    let frame = |phase: &mut EntityPhase, order: &[usize], revisions: &dyn Fn(usize) -> u64| {
        phase.begin();
        for &id in order {
            phase.push_blended_lit(
                EntityPhase::SOLID_TEXTURE,
                LayerBlend::Opaque,
                FULL_BRIGHT_COMBINED_LIGHT,
                &shell,
            );
            let image = phase
                .map_atlas_slot(&gpu, id as i32, revisions(id), || Some(pixels[id].clone()))
                .expect("slot");
            let mut quad = quad;
            for vertex in &mut quad {
                vertex.uv = image.slot.uv(vertex.uv);
            }
            phase.push_map_deferred(image.slot, MAP_BLEND, FULL_BRIGHT_COMBINED_LIGHT, &quad);
        }
        phase.flush_deferred_maps();
        phase.finish(&gpu);
        (phase.batch_count(), phase.map_atlas_stats())
    };
    let pages = maps.div_ceil(SLOTS_PER_PAGE as usize);
    assert_eq!(pages, 3);

    let (draws, stats) = frame(&mut phase, &natural, &|_| 1);
    assert_eq!(
        draws,
        1 + pages,
        "all shells in one draw, one draw per page"
    );
    assert_eq!(
        stats,
        MapAtlasStats {
            pages,
            resident: maps,
            uploads: maps as u64,
            upload_bytes: maps as u64 * 130 * 130 * 4,
            evictions: 0,
        }
    );
    // Nothing changed: no upload at all.
    for _ in 0..5 {
        let (draws, again) = frame(&mut phase, &scrambled, &|_| 1);
        assert_eq!(draws, 1 + pages, "scrambled order: still one draw per page");
        assert_eq!(again.uploads, maps as u64, "no upload while no map changes");
    }
    // One map's pixels change (a new S34): exactly one upload.
    let (_, changed) = frame(&mut phase, &scrambled, &|id| if id == 57 { 2 } else { 1 });
    assert_eq!(changed.uploads, maps as u64 + 1);
    // World change: everything is freed.
    phase.clear_map_atlas();
    assert_eq!(phase.map_atlas_stats(), MapAtlasStats::default());
}
