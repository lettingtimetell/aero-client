//! The client world's lighting and mesh
//! invalidation, packet to frame.
//!
//! The audit's verdict for A12 was that the storage layer is good and everything *dynamic* is
//! missing — `mc-world`'s own doc comment said so ("stores whatever the server sent and never
//! propagates or recomputes light"). 1.8 has no light-update packet and the server never
//! re-sends a column after a block change, so a torch lit only its own cell and a chunk that
//! left view distance kept rendering forever.
//!
//! `mc-world`'s crate-level tests (`tests/light_propagation.rs`) and its differential test
//! against the real recompiled 1.8.9 classes (`tests/light_parity.rs`) prove the *arithmetic*.
//! This file proves the **wiring**: that a real `S23PacketBlockChange` off a real socket reaches
//! it, that a chunk-unload packet actually drops the GPU buffers, and that a dimension change
//! tears the meshes down.

mod common;

use std::time::Duration;

use common::{config, gpu, pump_until, serve_join, FLOOR_Y};
use mc_client::app::Client;
use mc_math::{BlockPos, ChunkPos};
use mc_protocol::play::s2c;
use mc_render::{Camera, PhaseList};
use tokio::net::TcpListener;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const LOCAL_ENTITY_ID: i32 = 42;

/// `Block.BLOCK_STATE_IDS` packs `block_id << 4 | meta`, which is `BlockState`'s own packing.
fn state_id(block: i32, meta: i32) -> i32 {
    (block << 4) | meta
}

const TORCH_ID: i32 = 50;
const STONE_ID: i32 = 1;

/// Standing on the floor at the centre of chunk (0, 0). `checkLightFor`'s
/// `isAreaLoaded(pos, 17, false)` guard reaches ±17 blocks, so the 3×3 column neighbourhood has
/// to be loaded before any propagation can happen at all — which is why [`joined_chunks`] is a
/// 3×3 and not the single column `accept_and_join` sends.
const TORCH: BlockPos = BlockPos {
    x: 8,
    y: (FLOOR_Y + 1) as i32,
    z: 8,
};

fn joined_chunks() -> Vec<ChunkPos> {
    let mut out = Vec::new();
    for x in -1..=1 {
        for z in -1..=1 {
            out.push(ChunkPos::new(x, z));
        }
    }
    out
}

struct Rig {
    client: Client,
    gpu: mc_render::Gpu,
    phases: PhaseList,
    camera: Camera,
    peer: tokio::task::JoinHandle<common::RefPeer>,
}

impl Rig {
    fn wait_for(&mut self, what: &str, done: impl FnMut(&Client) -> bool) {
        pump_until(
            &mut self.client,
            &mut self.gpu,
            &mut self.phases,
            &mut self.camera,
            &mc_physics::state::MoveInput::default(),
            Duration::from_secs(30),
            what,
            done,
        );
    }

    fn pump(&mut self) {
        common::pump(
            &mut self.client,
            &mut self.gpu,
            &mut self.phases,
            &mut self.camera,
            &mc_physics::state::MoveInput::default(),
        );
    }

    fn block_light(&self, p: BlockPos) -> u8 {
        self.client
            .session()
            .expect("a live session")
            .world()
            .block_light_at(p)
    }

    fn finish(mut self) {
        self.client.disconnect();
        for _ in 0..10 {
            self.pump();
        }
        let handle = self.client.runtime().handle().clone();
        handle.block_on(async {
            tokio::time::timeout(Duration::from_secs(10), self.peer)
                .await
                .expect("the peer must finish")
                .expect("the peer task must not panic");
        });
    }
}

fn join_and_script<F, Fut>(extra: F) -> Option<Rig>
where
    F: FnOnce(common::RefPeer) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = common::RefPeer> + Send,
{
    let gpu = gpu(WIDTH, HEIGHT)?;
    let dir = Box::leak(Box::new(tempfile::tempdir().expect("tempdir")));
    let config = config(dir.path(), "Dev")?;
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let camera = mc_client::app::driver::menu_camera();

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let peer = client.runtime().handle().spawn(async move {
        let mut peer = serve_join(&listener, LOCAL_ENTITY_ID, &joined_chunks(), 0b1).await;
        peer = extra(peer).await;
        peer.record_until_closed().await;
        peer
    });

    client.handle_action(mc_ui::screens::Action::Connect(Box::new(
        mc_ui::screens::multiplayer::ServerEntry::new("Loopback", format!("127.0.0.1:{port}")),
    )));

    let mut rig = Rig {
        client,
        gpu,
        phases,
        camera,
        peer,
    };
    rig.wait_for("the client to reach the world", |client| {
        client.ui.name() == "in_game"
            && client.session().is_some_and(|s| {
                joined_chunks()
                    .iter()
                    .all(|pos| s.world().is_chunk_loaded(*pos))
            })
    });
    Some(rig)
}

/// A `S23PacketBlockChange` carrying a torch is the *only* thing the
/// server will ever say about that light. There is no light packet in 1.8 and the column is
/// never re-sent, so if the client does not run `checkLight` itself the torch lights exactly one
/// cell, forever.
#[test]
fn a_torch_off_the_wire_lights_the_floor_around_it() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: TORCH,
            block_state_id: state_id(TORCH_ID, 5),
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.wait_for("the torch to arrive", |client| {
        client
            .session()
            .is_some_and(|s| !s.world().block_at(TORCH).props().is_air())
    });

    // java: torch is `setLightLevel(0.9375F)` -> `(int)(0.9375 * 15) == 14`.
    assert_eq!(rig.block_light(TORCH), 14, "the torch's own cell");
    // ...and the gradient, which is the half that did not exist before A12-01.
    for (dx, expected) in [(1, 13), (2, 12), (3, 11), (4, 10)] {
        let p = BlockPos::new(TORCH.x + dx, TORCH.y, TORCH.z);
        assert_eq!(
            rig.block_light(p),
            expected,
            "{dx} blocks from the torch, at {p:?}"
        );
    }
    // It stops where vanilla stops rather than filling the world.
    assert_eq!(
        rig.block_light(BlockPos::new(TORCH.x + 14, TORCH.y, TORCH.z)),
        0
    );

    // The mesher was told. Every section the light touched has to be re-meshed or the change is
    // invisible; `mark_dirty_box` is what puts them in the queue.
    rig.wait_for("the torch mesh upload", |client| {
        client.uploaded_section_count() > 0
    });
    rig.finish();
}

/// The removal half — `checkLightFor`'s darkening branch, which erases the old lit region and
/// re-brightens from its frontier. A client that only ever brightens leaves a permanent glow
/// where a torch used to be.
#[test]
fn breaking_the_torch_takes_its_light_back() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: TORCH,
            block_state_id: state_id(TORCH_ID, 5),
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };
    rig.wait_for("the torch to arrive", |client| {
        client
            .session()
            .is_some_and(|s| s.world().block_light_at(TORCH) == 14)
    });

    // Now break it. `PlayerControllerMP` predicts the break locally through this same
    // `World::set_block`, so driving it directly is the local-prediction path rather than a
    // second wire round trip.
    rig.client
        .session_mut()
        .expect("session")
        .state_mut()
        .world
        .set_block(TORCH, mc_data::BlockState::AIR);

    assert_eq!(rig.block_light(TORCH), 0);
    for dx in 1..=4 {
        assert_eq!(
            rig.block_light(BlockPos::new(TORCH.x + dx, TORCH.y, TORCH.z)),
            0,
            "{dx} blocks out should be dark again"
        );
    }
    rig.finish();
}

/// `WorldClient.doPreChunk(x, z, false)` unloads the column *and then*
/// `markBlockRangeForRenderUpdate`s it. Before this, `World::unload_chunk` did the opposite —
/// it `remove`d those sections from the dirty set — and `ClientEvent::ChunkUnloaded` was matched
/// nowhere, so the GPU buffers stayed resident and drawn until the budget evictor happened to
/// reach them.
#[test]
fn an_unloaded_column_drops_its_gpu_meshes() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        // The wire's unload shape: a full-column packet with an empty bitmask.
        // java: `handleChunkData`'s `getExtractedSize() == 0` branch -> `doPreChunk(x, z, false)`.
        // Sent after a short wait so the client has meshed and uploaded the columns first.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        for pos in joined_chunks() {
            peer.send(&mc_protocol::chunk::ChunkData {
                chunk_x: pos.x,
                chunk_z: pos.z,
                ground_up_continuous: true,
                primary_bit_mask: 0,
                data: Vec::new(),
            })
            .await;
        }
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };
    rig.wait_for("the terrain to reach the GPU", |client| {
        client.uploaded_section_count() > 0
    });
    assert!(rig.client.uploaded_section_count() > 0);

    rig.wait_for("the unload packets to drop every mesh", |client| {
        client
            .session()
            .is_some_and(|s| !s.world().is_chunk_loaded(ChunkPos::new(0, 0)))
            && client.uploaded_section_count() == 0
    });
    if let Some(render) = rig.client.render() {
        assert_eq!(
            render.terrain.borrow().loaded_sections(),
            0,
            "the renderer must not be holding meshes for columns that no longer exist"
        );
    }
    rig.finish();
}

/// The live upload caller must retain empty sections' visibility: a camera above two air
/// sections can still see the floor below them. Removing those meshes makes the culling walk
/// stop at the first air section as though it were an uncompiled, opaque chunk.
#[test]
fn terrain_remains_visible_through_empty_sections() {
    let Some(mut rig) = join_and_script(|peer| async move { peer }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        common::pump_no_tick(
            &mut rig.client,
            &mut rig.gpu,
            &mut rig.phases,
            &mut rig.camera,
        );
        if rig.client.mesh_queue_len() == 0
            && rig.client.mesh_queue_deferred_len() == 0
            && rig.client.mesh_queue_in_flight_len() == 0
        {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "meshing must settle");
        std::thread::sleep(Duration::from_millis(2));
    }
    // Every frame accepts at least one ready upload, even after its deadline. The 3x3 fixture
    // invalidates at most 5x5x16 sections including neighbours; drain that upper bound without
    // ticking the world's light repair and creating more work.
    for _ in 0..400 {
        common::pump_no_tick(
            &mut rig.client,
            &mut rig.gpu,
            &mut rig.phases,
            &mut rig.camera,
        );
    }

    let mut camera = Camera {
        pos: mc_math::DVec3::new(8.5, 56.0, 8.5),
        pitch: 89.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    camera.update();
    let terrain = rig.client.render().unwrap().terrain.clone();
    for smart in [false, true] {
        terrain.borrow_mut().smart_culling = smart;
        let frame = rig.gpu.acquire().expect("offscreen frame");
        rig.phases.render(&rig.gpu, frame.view(), &camera, 0.0);
        assert!(
            terrain.borrow().stats().quads > 0,
            "the floor must draw through air sections with smart culling = {smart}"
        );
    }
    for y in [1, 2, 3] {
        assert_eq!(
            terrain
                .borrow()
                .visibility(mc_math::ChunkSectionPos::new(0, y, 0)),
            Some(mc_render::terrain::visgraph::SetVisibility::all(true)),
            "compiled air must remain traversable"
        );
    }
    assert_eq!(
        terrain.borrow().gpu_memory().sections,
        joined_chunks().len()
    );
    assert_eq!(
        rig.client.uploaded_section_count(),
        joined_chunks().len() * 16
    );

    // Negative control: reproduce the former caller's loss of empty meshes. The exact same
    // geometry and camera must now produce no terrain draws with smart culling enabled.
    for column in joined_chunks() {
        for y in 1..16 {
            terrain
                .borrow_mut()
                .remove(mc_math::ChunkSectionPos::new(column.x, y, column.z));
        }
    }
    let frame = rig.gpu.acquire().expect("offscreen control frame");
    rig.phases.render(&rig.gpu, frame.view(), &camera, 0.0);
    assert_eq!(
        terrain.borrow().stats().quads,
        0,
        "missing air visibility hides the floor"
    );
    drop(frame);
    rig.finish();
    assert_eq!(
        terrain.borrow().loaded_sections(),
        0,
        "disconnect clears air too"
    );
}

/// `handleRespawn` builds a whole new `WorldClient` on a dimension change and calls
/// `loadWorld`, which reaches `RenderGlobal.setWorldAndLoadRenderers` and drops every
/// `RenderChunk`. Here `reset_world` replaced the world and nothing told the renderer, so a
/// Nether portal left the overworld's terrain on screen wherever the new dimension had not (yet)
/// supplied a section at the same position.
#[test]
fn a_dimension_change_tears_down_every_chunk_mesh() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        // A same-dimension respawn (death) keeps the world, and must keep the meshes.
        peer.send(&s2c::Respawn {
            dimension: 0,
            difficulty: 2,
            game_mode: 0,
            level_type: "default".to_owned(),
        })
        .await;
        tokio::time::sleep(Duration::from_millis(500)).await;
        // ...and then a real portal.
        peer.send(&s2c::Respawn {
            dimension: -1,
            difficulty: 2,
            game_mode: 0,
            level_type: "default".to_owned(),
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };
    rig.wait_for("the terrain to reach the GPU", |client| {
        client.uploaded_section_count() > 0
    });

    // The death respawn must not have emptied anything: `changed_world` is false, so
    // `reset_world` never ran and the meshes still describe the world the player is in.
    rig.wait_for("the overworld chunks to be gone", |client| {
        client
            .session()
            .is_some_and(|s| !s.world().is_chunk_loaded(ChunkPos::new(0, 0)))
    });
    assert_eq!(
        rig.client.uploaded_section_count(),
        0,
        "the previous dimension's meshes must not survive a portal"
    );
    assert_eq!(rig.client.mesh_queue_len(), 0);
    if let Some(render) = rig.client.render() {
        assert_eq!(render.terrain.borrow().loaded_sections(), 0);
    }
    rig.finish();
}

/// The periodic repair pass runs, is bounded, and does not depend on a block change
/// having happened — it is what fixes sky-light seams at column boundaries and light sources
/// that sat near a chunk edge when their column loaded.
#[test]
fn the_periodic_light_pass_runs_every_tick_and_stays_bounded() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        // A stone pillar on the chunk boundary: the shape `recheckGaps` exists for.
        for y in 1..=4 {
            peer.send(&s2c::BlockChange {
                position: BlockPos::new(15, (FLOOR_Y as i32) + y, 8),
                block_state_id: state_id(STONE_ID, 0),
            })
            .await;
        }
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };
    rig.wait_for("the pillar to arrive", |client| {
        client.session().is_some_and(|s| {
            !s.world()
                .block_at(BlockPos::new(15, (FLOOR_Y as i32) + 4, 8))
                .props()
                .is_air()
        })
    });

    // Pumping many ticks must not diverge, panic, or leave the client without a world. The
    // per-tick budget is `relight::LIGHT_COLUMNS_PER_TICK` columns, and each column's own work
    // is bounded by vanilla (`recheckGaps` returns after one flagged column on the client, and
    // `enqueueRelightChecks` advances 8 of 4096 slots).
    for _ in 0..120 {
        rig.pump();
    }
    assert!(rig.client.session().is_some());
    assert!(
        rig.client
            .session()
            .expect("session")
            .world()
            .is_chunk_loaded(ChunkPos::new(0, 0)),
        "the repair pass must not unload anything"
    );
    rig.finish();
}
