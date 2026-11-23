//! Tile entities, packet to frame.
//!
//! The audit's verdict for this area was that there is **no tile-entity subsystem in the Rust
//! client at all**: `mc-render/src/entity/{tile.rs, models/tileent.rs}` were 2,232 lines of
//! faithful, GPU-tested, entirely unreachable code, `driver.rs`'s event match dropped
//! `TileEntityUpdated`/`SignUpdated`/`OpenSignEditor`/`BlockAction` in its "no reader yet" arm,
//! and `terrain/mesher.rs` correctly emits nothing for `RenderType::TileEntity` — so a chest was
//! a hole in the world you could still open.
//!
//! This file drives the real client against the loopback double (`tests/common/mod.rs`) and
//! asserts on the three halves that were missing at once: the world's tile-entity store, the
//! per-tick animation, and the geometry the registered `TilePhase` uploads.
//!
//! **Proved here:** a chest block off a socket becomes a drawable tile entity with no
//! description packet; `S24PacketBlockAction` opens its lid and the lid integrates at 20 Hz;
//! `S33PacketUpdateSign` and `S35PacketUpdateTileEntity` land in the store; the tile pass
//! uploads real vertices in a real frame.
//!
//! **Not proved here:** that those pixels look right. That is `mc-render`'s
//! `tests/a05_tile_entities.rs`, which reads the framebuffer back.

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump_until, CHUNK, FLOOR_Y, SPAWN};
use mc_client::app::Client;
use mc_math::BlockPos;
use mc_protocol::play::s2c;
use mc_render::{Camera, PhaseList};
use mc_world::tileent::{TileEntity, TileKind};
use tokio::net::TcpListener;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const LOCAL_ENTITY_ID: i32 = 42;

/// `minecraft:chest`, facing south (metadata 3), one block up from the floor and two blocks
/// north of spawn so it is in front of the camera rather than inside it.
const CHEST: BlockPos = BlockPos {
    x: 8,
    y: (FLOOR_Y + 1) as i32,
    z: 6,
};
/// `minecraft:standing_sign`, beside the chest.
const SIGN: BlockPos = BlockPos {
    x: 10,
    y: (FLOOR_Y + 1) as i32,
    z: 6,
};
/// `minecraft:skull`, on the other side.
const SKULL: BlockPos = BlockPos {
    x: 6,
    y: (FLOOR_Y + 1) as i32,
    z: 6,
};

const CHEST_ID: i32 = 54;
const STANDING_SIGN_ID: i32 = 63;
const SKULL_ID: i32 = 144;

/// `Block.BLOCK_STATE_IDS` packs `block_id << 4 | meta`, which is `BlockState`'s own packing.
fn state_id(block: i32, meta: i32) -> i32 {
    (block << 4) | meta
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
        let mut peer = accept_and_join(listener, LOCAL_ENTITY_ID).await;
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
            && client
                .session()
                .is_some_and(|s| s.world().is_chunk_loaded(CHUNK))
    });
    Some(rig)
}

/// A chest block arrives with no description packet behind it —
/// vanilla never sends one for a chest — and must still become a drawable tile entity whose
/// geometry reaches the GPU phase.
#[test]
fn a_chest_block_becomes_drawable_geometry_with_no_description_packet() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: CHEST,
            block_state_id: state_id(CHEST_ID, 3),
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    // java: `Chunk.createNewTileEntity` is a pure function of the block, which is what makes the
    // chest exist without a packet about it.
    rig.wait_for("the chest block to arrive and be indexed", |client| {
        client
            .session()
            .is_some_and(|s| s.world().tile_kind_at(CHEST) == Some(TileKind::Chest))
            && client
                .tile_index()
                .iter()
                .any(|(pos, kind)| pos == CHEST && kind == TileKind::Chest)
    });

    // The number the audit was about: before this change `RenderStack::install` registered no
    // tile-entity phase, so this was structurally incapable of being anything but zero.
    rig.wait_for("the tile pass to upload the chest", |client| {
        client.tile_geometry().0 > 0
    });
    let (vertices, batches) = rig.client.tile_geometry();
    // A single chest is three parts x six faces x six vertices.
    assert_eq!(vertices, 108, "one chest, {batches} batch(es)");
    assert_eq!(batches, 1);

    rig.finish();
}

/// `S24PacketBlockAction` is the *only* input the chest lid animation has,
/// and `driver.rs` dropped it. After it lands, `TileEntityChest.update`'s ±0.1-per-tick
/// integrator has to run on the client.
#[test]
fn a_block_action_opens_the_chest_and_the_lid_integrates_at_twenty_hertz() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: CHEST,
            block_state_id: state_id(CHEST_ID, 3),
        })
        .await;
        // java: `BlockContainer.onBlockEventReceived` -> `TileEntityChest.receiveClientEvent(1, 1)`
        peer.send(&s2c::BlockAction {
            position: CHEST,
            data1: 1,
            data2: 1,
            block_id: CHEST_ID,
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.wait_for("the chest to record a user", |client| {
        client.session().is_some_and(|s| {
            s.world()
                .tile_entity(CHEST)
                .and_then(TileEntity::chest)
                .is_some_and(|c| c.num_players_using == 1)
        })
    });

    // java: `lidAngle += 0.1F` per tick while `numPlayersUsing > 0`, clamped to 1.
    let read = |client: &Client| {
        client
            .session()
            .and_then(|s| s.world().tile_entity(CHEST))
            .and_then(TileEntity::chest)
            .map_or((0.0, 0.0), |c| (c.prev_lid_angle, c.lid_angle))
    };
    let before = read(&rig.client);
    rig.pump();
    let after = read(&rig.client);
    assert!(
        after.1 > before.1,
        "the lid must open: {before:?} then {after:?}"
    );
    assert!(
        (after.1 - after.0 - 0.1).abs() < 1e-5,
        "exactly 0.1 per tick: {after:?}"
    );

    for _ in 0..15 {
        rig.pump();
    }
    assert_eq!(read(&rig.client).1, 1.0, "clamped fully open");

    // And closing brings it back down.
    rig.client
        .session_mut()
        .expect("session")
        .state_mut()
        .world
        .apply_block_action(CHEST, CHEST_ID, 1, 0);
    for _ in 0..15 {
        rig.pump();
    }
    assert_eq!(read(&rig.client).1, 0.0, "clamped shut");

    rig.finish();
}

/// `S33PacketUpdateSign` was decoded, wrapped in a `ClientEvent` and thrown away.
#[test]
fn sign_text_arrives_and_is_stored_as_display_text() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: SIGN,
            block_state_id: state_id(STANDING_SIGN_ID, 8),
        })
        .await;
        peer.send(&s2c::UpdateSign {
            position: SIGN,
            lines: [
                r#"{"text":"Shop"}"#.to_owned(),
                r#"{"text":"diamonds","color":"aqua"}"#.to_owned(),
                r#"{"text":""}"#.to_owned(),
                r#"{"text":"open"}"#.to_owned(),
            ],
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.wait_for("the sign text to land", |client| {
        client.session().is_some_and(|s| {
            s.world()
                .tile_entity(SIGN)
                .and_then(TileEntity::sign)
                .is_some_and(|sign| sign.lines[0].contains("Shop"))
        })
    });
    let sign = rig
        .client
        .session()
        .and_then(|s| s.world().tile_entity(SIGN))
        .and_then(TileEntity::sign)
        .expect("a sign")
        .clone();
    // java: the renderer draws `getFormattedText()`, so the `§b` from `"color":"aqua"` survives.
    assert!(sign.lines[1].contains("diamonds"), "{:?}", sign.lines);
    assert!(
        sign.lines[1].contains('\u{a7}'),
        "the colour code must survive into the display text: {:?}",
        sign.lines[1]
    );
    // java: `ChatComponentStyle.getFormattedText` (`util/ChatComponentStyle.java:87`-`:99`)
    // appends `EnumChatFormatting.RESET` after **every** component, unconditionally — there is no
    // "only if it had a style" branch. The sign renderer formats the first line `splitText`
    // returns: an empty `ChatComponentText` root (one `§r`) with the row's piece appended (its
    // own `§r`). The 1.8.9 bytecode turns `{"text":"Shop"}` into `§rShop§r`
    // (`mc-ui/tests/vanilla_sign_text.rs`), so a plain unstyled row is wrapped in two resets.
    assert_eq!(sign.lines[3], "\u{a7}ropen\u{a7}r");
    // java: `TileEntitySign.lineBeingEdited = -1` until the edit GUI opens.
    assert_eq!(sign.line_being_edited, -1);

    // The board itself is geometry too.
    rig.wait_for("the sign board to be uploaded", |client| {
        client.tile_geometry().0 > 0
    });

    rig.finish();
}

/// `S35PacketUpdateTileEntity` action 4 is a skull, and the action byte is a
/// *type check* against the block, not a constructor — a mismatched action must be dropped
/// exactly as `NetHandlerPlayClient:1276`'s `instanceof` conjunction drops it.
#[test]
fn a_skull_description_packet_lands_and_a_mismatched_action_does_not() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: SKULL,
            // metadata 1 is `EnumFacing.UP` — a floor skull.
            block_state_id: state_id(SKULL_ID, 1),
        })
        .await;
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("id", mc_nbt::Nbt::String("Skull".to_owned()));
        nbt.insert("SkullType", mc_nbt::Nbt::Byte(2));
        nbt.insert("Rot", mc_nbt::Nbt::Byte(7));
        peer.send(&s2c::UpdateTileEntity {
            position: SKULL,
            action: 4,
            nbt: Some(nbt),
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.wait_for("the skull payload to land", |client| {
        client.session().is_some_and(|s| {
            s.world()
                .tile_entity(SKULL)
                .and_then(TileEntity::skull)
                .is_some_and(|skull| skull.skull_type == 2)
        })
    });
    let skull = rig
        .client
        .session()
        .and_then(|s| s.world().tile_entity(SKULL))
        .and_then(TileEntity::skull)
        .expect("a skull")
        .clone();
    assert_eq!(skull.rotation, 7);

    // A beacon payload (action 3) aimed at a skull block must be refused.
    let mut wrong = mc_nbt::NbtCompound::new();
    wrong.insert("Levels", mc_nbt::Nbt::Int(4));
    let applied = rig
        .client
        .session_mut()
        .expect("session")
        .state_mut()
        .world
        .apply_tile_entity_nbt(
            SKULL,
            3,
            mc_world::TileData::Beacon(mc_world::BeaconState::default()),
        );
    assert!(!applied, "an action/block mismatch must be dropped");
    assert_eq!(
        rig.client
            .session()
            .and_then(|s| s.world().tile_entity(SKULL))
            .and_then(TileEntity::skull)
            .map(|s| s.skull_type),
        Some(2),
        "and must not have clobbered the skull"
    );

    rig.finish();
}

/// A double chest draws once, from its `-X`/`-Z` half, with the large model —
/// `chest_matrix`'s `double_offset` and `chest_lid_angle`'s `neighbour_lid` were parameters no
/// caller in the tree ever supplied.
#[test]
fn a_double_chest_draws_once_with_the_large_model() {
    let east = BlockPos {
        x: CHEST.x + 1,
        ..CHEST
    };
    let Some(mut rig) = join_and_script(move |mut peer| async move {
        for pos in [CHEST, east] {
            peer.send(&s2c::BlockChange {
                position: pos,
                // metadata 2 is NORTH, which is the facing whose `+X` neighbour takes the
                // `translate(1, 0, 0)` in `TileEntityChestRenderer:143`.
                block_state_id: state_id(CHEST_ID, 2),
            })
            .await;
        }
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.wait_for("both chest blocks to be indexed", |client| {
        client
            .tile_index()
            .iter()
            .filter(|(_, kind)| *kind == TileKind::Chest)
            .count()
            == 2
    });
    rig.wait_for("the pair to be uploaded", |client| {
        client.tile_geometry().0 > 0
    });

    let adjacency = rig
        .client
        .session()
        .expect("session")
        .world()
        .chest_neighbours(CHEST);
    assert!(adjacency.is_double() && adjacency.draws());
    assert!(
        !rig.client
            .session()
            .expect("session")
            .world()
            .chest_neighbours(east)
            .draws(),
        "the +X half must not draw"
    );

    // One large chest, drawn once: still three parts x six faces x six vertices, not two chests'
    // worth. A double chest that drew as two singles would be 216.
    assert_eq!(
        rig.client.tile_geometry().0,
        108,
        "a double chest is one draw of the large model"
    );

    rig.finish();
}

/// **A05-01, the other direction.** Unloading the column drops the store and the index with it,
/// so a long session cannot leak tile entities for chunks it no longer has.
#[test]
fn unloading_a_column_drops_its_tile_entities() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        peer.send(&s2c::BlockChange {
            position: CHEST,
            block_state_id: state_id(CHEST_ID, 3),
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.wait_for("the chest to be stored", |client| {
        client.tile_geometry().0 > 0
    });
    // Force the entry to exist in the store, not just in the index.
    rig.client
        .session_mut()
        .expect("session")
        .state_mut()
        .world
        .tile_entity_or_create(CHEST);
    assert_eq!(
        rig.client
            .session()
            .expect("session")
            .world()
            .tile_entity_count(),
        1
    );

    rig.client
        .session_mut()
        .expect("session")
        .state_mut()
        .world
        .unload_chunk(CHUNK);
    assert_eq!(
        rig.client
            .session()
            .expect("session")
            .world()
            .tile_entity_count(),
        0,
        "java: the map lives on the Chunk and goes with it"
    );
    rig.pump();
    assert_eq!(
        rig.client.tile_index().iter().count(),
        0,
        "and the index follows the loaded set"
    );
    let _ = SPAWN;

    rig.finish();
}

/// **"Cant see text on signs".**
///
/// A real 1.8.8 server sends every sign row as an empty root with the text in `extra`:
/// `{"extra":["-CXXVI"],"text":""}` is verbatim from
/// `mc-parity/tests/fixtures/spigot-1.8.8-offline-join-walk-chat.mccap`. Vanilla draws it,
/// because `GuiUtilRenderComponents.splitText` builds its work list with
/// `Lists.newArrayList(Iterable)` over the component (bytecode `avu.a`, offset 20), and an
/// `IChatComponent` iterates itself *and its siblings*.
///
/// This drives that shape through the socket and the real frame, then reads the pixels back:
/// the sign that received the rows must show black glyph pixels that the identical text-less
/// sign beside it does not.
#[test]
fn an_extra_only_sign_row_draws_black_glyphs_on_the_board() {
    // Two standing signs south of spawn, rotation 8 (facing north, back at the camera, which
    // faces south). Facing south, +X is screen-left, so `WITH_TEXT` lands right of centre and
    // `NO_TEXT` mirrors it on the left. Their text blocks are located by projection below.
    const WITH_TEXT: BlockPos = BlockPos {
        x: 7,
        y: (FLOOR_Y + 1) as i32,
        z: 10,
    };
    const NO_TEXT: BlockPos = BlockPos {
        x: 9,
        y: (FLOOR_Y + 1) as i32,
        z: 10,
    };
    let row = r#"{"extra":["MMMMMMMMMMMMMM"],"text":""}"#.to_owned();
    let lines = [row.clone(), row.clone(), row.clone(), row];
    let Some(mut rig) = join_and_script(move |mut peer| async move {
        for position in [WITH_TEXT, NO_TEXT] {
            peer.send(&s2c::BlockChange {
                position,
                block_state_id: state_id(STANDING_SIGN_ID, 8),
            })
            .await;
        }
        peer.send(&s2c::UpdateSign {
            position: WITH_TEXT,
            lines,
        })
        .await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.wait_for("the extra-only sign packet to land", |client| {
        client.session().is_some_and(|s| {
            s.world()
                .tile_entity(WITH_TEXT)
                .and_then(TileEntity::sign)
                .is_some_and(|sign| sign.components[0].contains("MMMM"))
        })
    });
    rig.wait_for("both boards to be uploaded", |client| {
        client.tile_geometry().0 > 0
    });
    for _ in 0..3 {
        rig.pump();
    }
    let pixels = rig.gpu.read_pixels().expect("readback");
    let stored = rig
        .client
        .session()
        .and_then(|s| s.world().tile_entity(WITH_TEXT))
        .and_then(TileEntity::sign)
        .map(|sign| sign.lines[0].clone());

    // Opt-in: `MC_L7_SIGN_FRAME=<file.png>` saves the frame these counts are taken from.
    if let Some(path) = std::env::var_os("MC_L7_SIGN_FRAME") {
        image::RgbaImage::from_raw(WIDTH, HEIGHT, pixels.clone())
            .expect("a full frame")
            .save(path)
            .expect("write the debug frame");
    }

    // Each board's text block on screen: the corners of the four 84-pixel rows
    // (`TileEntitySignRenderer`'s `x = -width / 2`, `y = j * 10 - 20`, 8-pixel glyphs) taken
    // through `sign_text_matrix` and the frame's own camera.
    let text_box = |pos: BlockPos| {
        let rel = rig.camera.relative(mc_math::DVec3::new(
            f64::from(pos.x),
            f64::from(pos.y),
            f64::from(pos.z),
        ));
        let text = mc_render::entity::tile::sign_text_matrix(mc_render::entity::tile::sign_matrix(
            rel, true, 8,
        ));
        let view_proj = rig.camera.view_proj();
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (x, y) in [(-42.0, -20.0), (42.0, -20.0), (42.0, 18.0), (-42.0, 18.0)] {
            let world = text.transform_point3(mc_math::Vec3::new(x, y, 0.0));
            let clip = view_proj * world.extend(1.0);
            let (sx, sy) = (
                (clip.x / clip.w + 1.0) * 0.5 * WIDTH as f32,
                (1.0 - clip.y / clip.w) * 0.5 * HEIGHT as f32,
            );
            (x0, y0, x1, y1) = (x0.min(sx), y0.min(sy), x1.max(sx), y1.max(sy));
        }
        let clamp = |v: f32, max: u32| (v.max(0.0) as u32).min(max);
        (
            clamp(x0, WIDTH),
            clamp(y0, HEIGHT),
            clamp(x1.ceil(), WIDTH),
            clamp(y1.ceil(), HEIGHT),
        )
    };
    // java: `drawString(s, x, y, 0)` — colour 0 is opaque black, while the wood of `sign.png` is
    // never darker than (56, 43, 24) even before shading.
    let dark = |(x0, y0, x1, y1): (u32, u32, u32, u32)| {
        let mut count = 0;
        for y in y0..y1 {
            for x in x0..x1 {
                let i = ((y * WIDTH + x) * 4) as usize;
                if pixels[i] <= 12 && pixels[i + 1] <= 12 && pixels[i + 2] <= 12 {
                    count += 1;
                }
            }
        }
        count
    };
    let with_box = text_box(WITH_TEXT);
    let without_box = text_box(NO_TEXT);
    let with_text = dark(with_box);
    let without_text = dark(without_box);
    eprintln!(
        "sign text: {with_text} black pixels in {with_box:?}, {without_text} in {without_box:?}"
    );
    assert!(
        with_box.2 > with_box.0 + 20 && with_box.3 > with_box.1 + 10,
        "the text block must be on screen: {with_box:?}"
    );
    assert!(
        with_text >= without_text + 150,
        "the sign with four extra-only rows must show black glyphs: {with_text} dark pixels \
         in its text block {with_box:?} vs {without_text} in the text-less sign's {without_box:?} \
         (stored row 0 = {stored:?})"
    );

    rig.finish();
}
