//! The client half of `tall_grass`, and the F5 camera ray.
//!
//! * **Tall Grass** (`mc_mods::mods::tall_grass`) is named after the vanilla feature and
//!   defaults **on**. Off hides grass and ferns, one and two blocks tall
//!   (`mc_render::terrain::tall_grass`). The switch is folded into the mesher's settings
//!   ([`Client::tall_grass_mesh_settings`]), so a toggle
//!   re-meshes every resident section once, through the same path as Fancy/Fast graphics.
//!   The world is never changed: targeting, the selection outline, breaking, collisions and
//!   every packet stay vanilla, so hidden grass still takes the crosshair and still blocks a
//!   melee hit on a player behind it, exactly as visible grass does in 1.8.9.
//! * **The F5 camera** ([`third_person_camera_distance`]) is `EntityRenderer.orientCamera`'s
//!   eight corner rays, ported bit for bit and checked against the real 1.8.9 classes
//!   (`javaref/CameraRayRef.java`, `javaref/camera_ray_ref.txt`). Every block is tested against
//!   its own vanilla bounds: tall grass is 0.1-0.9 x 0-0.8 x 0.1-0.9, a double plant a full
//!   cube, a flower 0.3-0.7 x 0-0.6. With the mod off the rays pass through hidden grass, so
//!   the camera never bumps into plants you cannot see.
//!
//! Cost with the mod on (vanilla): one `u64` compare per frame here, one `bool` test per block
//! at mesh time, nothing extra in the F5 rays. Off: two packed-id compares per block at mesh
//! time and in each F5 ray step.

use mc_data::BlockState;
use mc_math::{BlockPos, DVec3};
use mc_render::terrain::MeshSettings;
use mc_world::World;

use super::super::Client;
use crate::app::interact;

/// The mod's id in the catalog and the settings file.
const MOD_ID: &str = "tall_grass";

/// What the client last applied.
#[derive(Debug, Default)]
pub(crate) struct W5State {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    /// The mod is off: grass is hidden from the mesher and from the F5 camera rays.
    pub(crate) hidden: bool,
}

impl Client {
    /// Follow the switch. One `u64` compare per frame while no mod setting
    /// changes; turning Tall Grass on or off re-meshes every resident section once.
    pub(crate) fn drive_tall_grass(&mut self) {
        let generation = self.mods.settings.generation();
        if self.w5.seen == generation {
            return;
        }
        self.w5.seen = generation;
        let hidden = !self.mods.settings.enabled(MOD_ID);
        if hidden == self.w5.hidden {
            return;
        }
        self.w5.hidden = hidden;
        self.sync_mesh_settings();
    }

    /// Fold the switch into the mesher's settings, so a change
    /// re-meshes every resident section through the existing path.
    pub(crate) fn tall_grass_mesh_settings(&self, wanted: MeshSettings) -> MeshSettings {
        MeshSettings {
            hide_tall_grass: self.w5.hidden,
            ..wanted
        }
    }
}

/// The third-person camera's distance from the eye.
///
/// java: `EntityRenderer.orientCamera:665-704`, the F5 branch outside the debug camera,
/// transcribed operation for operation so the result is bit-identical:
///
/// * `d3` starts at the interpolated `thirdPersonDistance`, which is 4.0 for the whole game in
///   1.8.9 (both of its fields are `4.0F`).
/// * The direction uses the render-view entity's `rotationYaw`/`rotationPitch` (front view:
///   pitch + 180) through `MathHelper`'s sine table, in `float`, widened after the multiply.
/// * Corner `i` offsets both ends by `(float)(+-1) * 0.1F` per axis, and the end point's X also
///   takes the Z offset (`d0 - d4 + f3 + f5`), in that association.
/// * Each ray is `World.rayTraceBlocks(start, end)`, which tests every block against its own
///   bounds ([`interact::raytrace_blocks_except`]); the hit distance is `Vec3.distanceTo`,
///   whose square root goes through `float`.
///
/// `hide_tall_grass` (the Tall Grass mod off) lets the rays pass through the grass the mesher
/// does not draw; `false` is vanilla and runs no extra test at all.
pub(crate) fn third_person_camera_distance(
    world: &World,
    eye: DVec3,
    yaw: f32,
    pitch: f32,
    third_person_view: u8,
    hide_tall_grass: bool,
) -> f64 {
    const PI: f32 = std::f32::consts::PI;
    // `thirdPersonDistanceTemp + (thirdPersonDistance - thirdPersonDistanceTemp) * partialTicks`
    // with both fields at 4.0F.
    let mut d3 = 4.0_f64;
    let f1 = yaw;
    let f2 = if third_person_view == 2 {
        pitch + 180.0
    } else {
        pitch
    };
    let d4 = f64::from(-mc_math::sin(f1 / 180.0 * PI) * mc_math::cos(f2 / 180.0 * PI)) * d3;
    let d5 = f64::from(mc_math::cos(f1 / 180.0 * PI) * mc_math::cos(f2 / 180.0 * PI)) * d3;
    let d6 = f64::from(-mc_math::sin(f2 / 180.0 * PI)) * d3;
    for i in 0..8_i32 {
        let f3 = f64::from(((i & 1) * 2 - 1) as f32 * 0.1_f32);
        let f4 = f64::from((((i >> 1) & 1) * 2 - 1) as f32 * 0.1_f32);
        let f5 = f64::from((((i >> 2) & 1) * 2 - 1) as f32 * 0.1_f32);
        let start = DVec3::new(eye.x + f3, eye.y + f4, eye.z + f5);
        let end = DVec3::new(eye.x - d4 + f3 + f5, eye.y - d6 + f4, eye.z - d5 + f5);
        let hit = if hide_tall_grass {
            interact::raytrace_blocks_except(world, start, end, hidden_tall_grass)
        } else {
            interact::raytrace_blocks_except(world, start, end, |_, _, _| false)
        };
        if let Some(hit) = hit {
            let d7 = vec3_distance_to(hit.hit_vec, eye);
            if d7 < d3 {
                d3 = d7;
            }
        }
    }
    d3
}

/// The camera rays' pass-through test with the mod off.
fn hidden_tall_grass(world: &World, state: BlockState, pos: BlockPos) -> bool {
    mc_render::terrain::is_hidden_tall_grass(world, state, pos)
}

/// java: `Vec3.distanceTo` - `(double)MathHelper.sqrt_double(...)`, and `sqrt_double` returns
/// `(float)Math.sqrt(value)`.
fn vec3_distance_to(this: DVec3, vec: DVec3) -> f64 {
    let d0 = vec.x - this.x;
    let d1 = vec.y - this.y;
    let d2 = vec.z - this.z;
    f64::from((d0 * d0 + d1 * d1 + d2 * d2).sqrt() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use mc_data::BlockId;
    use mc_render::terrain::fixture;

    /// `javaref/CameraRayRef.java`'s output: real 1.8.9 `orientCamera` distances.
    const REFERENCE: &str = include_str!("../../../javaref/camera_ray_ref.txt");

    #[derive(Debug, Clone, Copy)]
    struct Config {
        view: u8,
        yaw: f32,
        pitch: f32,
        eye: DVec3,
    }

    #[derive(Debug, Default)]
    struct Reference {
        configs: Vec<Config>,
        scenes: Vec<(String, Vec<(BlockPos, BlockState)>)>,
        /// `(scene, config, d3)`.
        rays: Vec<(usize, usize, f64)>,
        /// `(scene, fnv over every config)`.
        hashes: Vec<(usize, u64)>,
        /// `(block id, storage metas, fnv)`: the block alone at (0, 64, 0).
        sweep: Vec<(u16, Vec<u8>, u64)>,
    }

    fn hex64(s: &str) -> u64 {
        u64::from_str_radix(s, 16).expect("hex u64")
    }

    fn hex32(s: &str) -> u32 {
        u32::from_str_radix(s, 16).expect("hex u32")
    }

    fn reference() -> Reference {
        let mut out = Reference::default();
        for line in REFERENCE
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let f: Vec<&str> = line.split(' ').collect();
            match f[0] {
                "c" => {
                    assert_eq!(f[1].parse::<usize>().unwrap(), out.configs.len());
                    out.configs.push(Config {
                        view: f[2].parse().unwrap(),
                        yaw: f32::from_bits(hex32(f[3])),
                        pitch: f32::from_bits(hex32(f[4])),
                        eye: DVec3::new(
                            f64::from_bits(hex64(f[5])),
                            f64::from_bits(hex64(f[6])),
                            f64::from_bits(hex64(f[7])),
                        ),
                    });
                }
                "s" => {
                    assert_eq!(f[1].parse::<usize>().unwrap(), out.scenes.len());
                    let blocks = f[3]
                        .split(';')
                        .map(|b| {
                            let n: Vec<i32> = b.split(',').map(|v| v.parse().unwrap()).collect();
                            let packed = u16::try_from(n[3]).unwrap();
                            (
                                BlockPos::new(n[0], n[1], n[2]),
                                BlockState::from_raw(packed),
                            )
                        })
                        .collect();
                    out.scenes.push((f[2].to_owned(), blocks));
                }
                "r" => out.rays.push((
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                    f64::from_bits(hex64(f[3])),
                )),
                "h" => out.hashes.push((f[1].parse().unwrap(), hex64(f[2]))),
                "b" => out.sweep.push((
                    f[1].parse().unwrap(),
                    f[2].bytes()
                        .map(|c| (c as char).to_digit(16).unwrap() as u8)
                        .collect(),
                    hex64(f[3]),
                )),
                other => panic!("unknown reference line {other:?}"),
            }
        }
        assert_eq!(out.configs.len(), 64);
        out
    }

    fn fnv(hash: &mut u64, d3: f64) {
        for b in d3.to_bits().to_le_bytes() {
            *hash ^= u64::from(b);
            *hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }

    const FNV_START: u64 = 0xcbf2_9ce4_8422_2325;

    /// One world reused for every scene (a blank world is nine full chunk columns).
    struct Bench {
        world: World,
        placed: Vec<BlockPos>,
    }

    impl Bench {
        fn new() -> Bench {
            Bench {
                world: fixture::blank_world(1),
                placed: Vec::new(),
            }
        }

        fn set(&mut self, blocks: &[(BlockPos, BlockState)]) -> &World {
            for pos in self.placed.drain(..) {
                self.world.set_block(pos, BlockState::AIR);
            }
            for &(pos, state) in blocks {
                self.world.set_block(pos, state);
                self.placed.push(pos);
            }
            &self.world
        }
    }

    fn distance(world: &World, c: &Config, hide: bool) -> f64 {
        third_person_camera_distance(world, c.eye, c.yaw, c.pitch, c.view, hide)
    }

    /// Parity: every scene and every block's sweep reproduces the real 1.8.9 `orientCamera`
    /// distance bit for bit.
    #[test]
    fn the_f5_camera_matches_the_real_1_8_9_classes_bit_for_bit() {
        let r = reference();
        let mut bench = Bench::new();
        let mut wrong = Vec::new();

        let mut placed = None;
        for &(scene, cfg, java) in &r.rays {
            if placed != Some(scene) {
                bench.set(&r.scenes[scene].1);
                placed = Some(scene);
            }
            let ours = distance(&bench.world, &r.configs[cfg], false);
            if ours.to_bits() != java.to_bits() {
                wrong.push(format!(
                    "{} config {cfg} {:?}: java {java}, mc-rs {ours}",
                    r.scenes[scene].0, r.configs[cfg]
                ));
            }
        }
        for &(scene, java) in &r.hashes {
            let world = bench.set(&r.scenes[scene].1);
            let mut hash = FNV_START;
            for c in &r.configs {
                fnv(&mut hash, distance(world, c, false));
            }
            if hash != java {
                wrong.push(format!("scene {}: hash differs", r.scenes[scene].0));
            }
        }
        let origin = BlockPos::new(0, 64, 0);
        for (id, metas, java) in &r.sweep {
            let mut hash = FNV_START;
            for &meta in metas {
                let world = bench.set(&[(origin, BlockState::new(BlockId(*id), meta))]);
                for c in &r.configs {
                    fnv(&mut hash, distance(world, c, false));
                }
            }
            if hash != *java {
                let name = mc_data::block_info(BlockId(*id)).name;
                wrong.push(format!("block {id} {name} (metas {metas:?}): hash differs"));
            }
        }
        assert!(
            r.rays.len() >= 11 * 64 && r.sweep.len() >= 190,
            "the reference covers the plant scenes and every block"
        );
        assert!(
            wrong.is_empty(),
            "{} F5 camera distances differ from vanilla:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }

    /// The reference's chunk-storage legality agrees with mc-data's, so the sweep places the same
    /// states on both sides.
    #[test]
    fn the_sweep_covers_exactly_the_legal_storage_states() {
        let r = reference();
        let mut java: Vec<u16> = r
            .sweep
            .iter()
            .flat_map(|(id, metas, _)| {
                metas
                    .iter()
                    .map(move |&m| BlockState::new(BlockId(*id), m).id)
            })
            .collect();
        let mut ours: Vec<u16> = mc_data::legal_states()
            .map(|s| s.id)
            .filter(|&id| id >> 4 != 0)
            .collect();
        java.sort_unstable();
        ours.sort_unstable();
        assert_eq!(java, ours);
    }

    /// With the mod off the rays ignore hidden grass entirely: every scene gives exactly the
    /// vanilla distance of the same scene with that grass taken out, and the scenes of grass
    /// alone never pull the camera in.
    #[test]
    fn with_tall_grass_off_the_camera_passes_through_hidden_grass_only() {
        let r = reference();
        let mut bench = Bench::new();
        let mut pulled_in = 0usize;
        for (name, blocks) in &r.scenes {
            let world = bench.set(blocks);
            let kept: Vec<(BlockPos, BlockState)> = blocks
                .iter()
                .copied()
                .filter(|&(pos, state)| {
                    !mc_render::terrain::is_hidden_tall_grass(world, state, pos)
                })
                .collect();
            let hidden_any = kept.len() != blocks.len();
            let off: Vec<f64> = r.configs.iter().map(|c| distance(world, c, true)).collect();
            let vanilla: Vec<f64> = r
                .configs
                .iter()
                .map(|c| distance(world, c, false))
                .collect();
            let world = bench.set(&kept);
            for (i, c) in r.configs.iter().enumerate() {
                let without = distance(world, c, false);
                assert_eq!(
                    off[i].to_bits(),
                    without.to_bits(),
                    "{name} config {i}: off must equal the scene without its grass"
                );
                if !hidden_any {
                    assert_eq!(off[i].to_bits(), vanilla[i].to_bits(), "{name} config {i}");
                }
            }
            if matches!(
                name.as_str(),
                "tall_grass" | "fern" | "double_grass_lower" | "double_fern_upper"
            ) {
                assert!(hidden_any, "{name} is hidden grass");
                assert!(
                    off.iter().all(|&d| d == 4.0),
                    "{name}: never pulled in when off"
                );
                pulled_in += vanilla.iter().filter(|&&d| d < 4.0).count();
            }
            if matches!(
                name.as_str(),
                "shrub" | "dead_bush" | "sunflower" | "dandelion" | "poppy" | "cactus"
            ) {
                assert!(
                    !hidden_any,
                    "{name} stays visible, so it still stops the camera"
                );
            }
        }
        assert!(
            pulled_in > 50,
            "control: visible grass pulls the camera in on many configurations ({pulled_in})"
        );
    }

    /// The bounds really are the plant's own, not a full cube: a ray just above tall grass's
    /// 0.8 top (inside the cell) passes, while the full-cube double plant stops it at its face.
    #[test]
    fn tall_grass_stops_the_camera_at_its_own_box_and_double_plants_at_the_cube() {
        let mut bench = Bench::new();
        let eye = DVec3::new(0.5, 64.0 + 0.92, 3.0);
        let tall_grass = BlockState::new(BlockId(31), 1);
        let world = bench.set(&[(BlockPos::new(0, 64, 0), tall_grass)]);
        // yaw 0 looks south (+Z), so the camera goes north, through the plant's cell at z 0..1.
        // The four lower rays (y 64.82) clear the 0.8 top; the upper four (y 65.02) leave the
        // cell. Nothing stops the camera.
        assert_eq!(
            third_person_camera_distance(world, eye, 0.0, 0.0, 1, false),
            4.0
        );
        let lower = DVec3::new(0.5, 64.0 + 0.5, 3.0);
        let stopped = third_person_camera_distance(world, lower, 0.0, 0.0, 1, false);
        // The box's south face is at z 0.9: about 2.1 blocks behind the eye at z 3.0.
        assert!(
            (stopped - 2.1).abs() < 0.01,
            "stopped at the 0.9 face: {stopped}"
        );
        assert_eq!(
            third_person_camera_distance(world, lower, 0.0, 0.0, 1, true),
            4.0
        );

        let world = bench.set(&[
            (BlockPos::new(0, 64, 0), BlockState::new(BlockId(175), 2)),
            (BlockPos::new(0, 65, 0), BlockState::new(BlockId(175), 10)),
        ]);
        let cube = third_person_camera_distance(world, eye, 0.0, 0.0, 1, false);
        assert!(
            (cube - 2.0).abs() < 0.01,
            "a double plant is a full cube: {cube}"
        );
        assert_eq!(
            third_person_camera_distance(world, eye, 0.0, 0.0, 1, true),
            4.0
        );
    }

    /// Cactus: rays (the crosshair's and the camera's) meet the full cube, because vanilla's
    /// `collisionRayTrace` reads the bound fields, while the outline keeps
    /// `getSelectedBoundingBox`'s 1/16 inset. mc-rs used to trace the inset outline.
    #[test]
    fn cactus_rays_meet_the_full_cube_and_the_outline_stays_inset() {
        assert_eq!(mc_data::block_by_name("cactus"), Some(BlockId(81)));
        let mut bench = Bench::new();
        let pos = BlockPos::new(0, 64, 0);
        let cactus = BlockState::new(BlockId(81), 0);
        let world = bench.set(&[(pos, cactus)]);
        // Northwards at x = 0.03, inside the cube but outside the inset outline.
        let start = DVec3::new(0.03, 64.5, 3.0);
        let end = DVec3::new(0.03, 64.5, -3.0);
        let hit = interact::raytrace_blocks(world, start, end)
            .and_then(interact::RayTrace::as_block)
            .expect("vanilla's ray meets the full cube");
        assert_eq!(hit.pos, pos);
        assert!(
            (hit.hit_vec.z - 1.0).abs() < 1e-9,
            "the cube's south face, not the outline's 0.9375: {:?}",
            hit.hit_vec
        );
        let outline = mc_render::terrain::selected_bounding_box(world, pos, cactus);
        assert_eq!(outline.min.x, 0.0625);
        assert_eq!(outline.max.z, 0.9375);
        // The camera stops at the cube too: 2.0 behind an eye at z 3.0, plus the corner offsets.
        let eye = DVec3::new(0.03, 64.5, 3.0);
        let d3 = third_person_camera_distance(world, eye, 0.0, 0.0, 1, false);
        assert!(d3 < 2.02, "camera stopped at the full cube's face: {d3}");
    }

    /// A headless client over the real assets, its game dir a temp dir (so the mod settings
    /// file can never land in the user's `mc-rs`). `None` without the vanilla jar.
    fn headless_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    #[test]
    fn the_switch_reaches_the_mesher_and_remeshes_only_on_a_change() {
        let Some((mut client, _dir)) = headless_client() else {
            return;
        };
        // On (the default): vanilla meshing, and the hook leaves everything alone.
        assert!(
            client.mods.settings.enabled(MOD_ID),
            "Tall Grass defaults on"
        );
        client.drive_tall_grass();
        assert!(!client.w5.hidden);
        assert!(!client.mesh_queue.settings().hide_tall_grass);
        let vanilla = client.mesh_queue.settings();

        // Resident sections, so a change has something to re-mesh.
        let queued = |client: &Client| client.mesh_queue.len() + client.mesh_queue.deferred_len();
        assert_eq!(queued(&client), 0);
        for y in 3..6 {
            client
                .uploaded_sections
                .insert(mc_math::ChunkSectionPos::new(0, y, 0));
        }

        client.mods.settings.set_enabled(MOD_ID, false);
        client.drive_tall_grass();
        assert!(client.w5.hidden);
        assert_eq!(
            client.mesh_queue.settings(),
            MeshSettings {
                hide_tall_grass: true,
                ..vanilla
            },
            "off only flips the grass switch"
        );
        assert_eq!(
            queued(&client),
            3,
            "turning it off re-meshes each resident section once"
        );

        // A frame with no settings change, or a change to another mod, re-meshes nothing.
        let before = client.mesh_queue.settings();
        client.drive_tall_grass();
        client.mods.settings.set_enabled("fps", true);
        client.drive_tall_grass();
        assert_eq!(client.mesh_queue.settings(), before);
        assert_eq!(queued(&client), 3, "no second re-mesh");

        // Back on: vanilla again.
        client.mods.settings.set_enabled(MOD_ID, true);
        client.drive_tall_grass();
        assert!(!client.w5.hidden);
        assert_eq!(client.mesh_queue.settings(), vanilla);

        // Off survives leaving the world, which rebuilds the mesh queue's settings.
        client.mods.settings.set_enabled(MOD_ID, false);
        client.drive_tall_grass();
        client.unload_world();
        assert!(
            client.mesh_queue.settings().hide_tall_grass,
            "kept across a world change"
        );
    }
}
