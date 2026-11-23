//! Server-sent particles, end to end.
//!
//! Before this, `mc_render::particle` was a complete, separately-tested simulator and render
//! phase with nothing feeding it: `sync::dispatch` produced `ClientEvent::Particles`, `react`
//! matched no such variant, `mc_render::particle::packet::apply` had no caller anywhere in the
//! workspace, and `IconLookup` — which `apply` needs before it will build a block-break or
//! item-break particle at all — had no implementor. The debug overlay's `P:` read `0` forever.
//!
//! This file drives the real client against the loopback double `atmosphere.rs`/`smoke.rs` use
//! (see `tests/common/mod.rs` for exactly what that is and is not) and asserts on
//! `Client::particles`, which is the same `ParticleSystem` the overlay counts, the tick loop
//! advances and `upload_particles` hands to the GPU phase.
//!
//! **Proved here:** a `0x2A Particles` frame off a socket reaches `ParticleSystem`; the count and
//! the range gate match `RenderGlobal.spawnParticle`; a `BLOCK_CRACK` packet resolves a real
//! sprite out of the real stitched atlas through `AtlasIcons` and lands on `FxLayer::Terrain`.
//!
//! **Not proved here:** that those pixels are visually correct. The GPU side of that lives in
//! `mc-render`'s own readback tests (`particle::phase::tests`, `tests/s18_particle.rs`).

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump_until, CHUNK, SPAWN};
use mc_client::app::Client;
use mc_data::{block_by_name, BlockState};
use mc_protocol::play::s2c;
use mc_render::particle::FxLayer;
use mc_render::{Camera, PhaseList};
use tokio::net::TcpListener;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const LOCAL_ENTITY_ID: i32 = 42;

/// `EnumParticleTypes.FLAME.getParticleID()`.
const FLAME: i32 = 26;
/// `EnumParticleTypes.BLOCK_CRACK.getParticleID()`.
const BLOCK_CRACK: i32 = 37;
/// How many flames the scripted server asks for.
const FLAME_COUNT: i32 = 8;

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

    fn finish(mut self) {
        self.client.disconnect();
        for _ in 0..10 {
            common::pump(
                &mut self.client,
                &mut self.gpu,
                &mut self.phases,
                &mut self.camera,
                &mc_physics::state::MoveInput::default(),
            );
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

/// Join the loopback double and script `extra` packets immediately after the join sequence, so
/// they are already queued inbound by the time the client reaches the world.
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

/// A `0x2A Particles` frame at the player's own feet.
fn at_spawn(particle_id: i32, count: i32, extra_data: Vec<i32>) -> s2c::Particles {
    s2c::Particles {
        particle_id,
        long_distance: false,
        x: SPAWN.0 as f32,
        y: SPAWN.1 as f32,
        z: SPAWN.2 as f32,
        offset_x: 0.2,
        offset_y: 0.2,
        offset_z: 0.2,
        particle_speed: 0.02,
        particle_count: count,
        extra_data,
    }
}

/// Two server packets — a plain misc-layer burst and a
/// parameterised `BLOCK_CRACK` that only spawns if `IconLookup` resolves a texture — have to end
/// up in the set `Client::particles` holds, which is what the `P:` counter formats, what
/// `Client::tick` advances and what `upload_particles` uploads.
#[test]
fn server_sent_particles_reach_the_clients_particle_system() {
    let stone = BlockState::new(block_by_name("stone").expect("stone"), 0).to_java_state_id();
    let Some(mut rig) = join_and_script(move |mut peer| async move {
        peer.send(&at_spawn(FLAME, FLAME_COUNT, vec![])).await;
        // java: `handleParticles` -> `EntityDiggingFX`, whose ctor looks the block's particle
        // texture up through `BlockModelShapes.getTexture(state)`. `count = 0` means "one
        // particle, offsets are the exact velocity".
        peer.send(&at_spawn(BLOCK_CRACK, 0, vec![stone])).await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    // `P: 0` for a whole session was the defect; wait for the moment it stops being true.
    rig.wait_for("the particle packets to be applied", |client| {
        client
            .particles
            .particles()
            .iter()
            .any(|p| p.layer == FxLayer::Terrain)
            && client
                .particles
                .particles()
                .iter()
                .any(|p| p.layer == FxLayer::Misc)
    });

    let live = rig.client.particles.particles();
    let misc = live.iter().filter(|p| p.layer == FxLayer::Misc).count();
    assert_eq!(
        misc, FLAME_COUNT as usize,
        "java: `particleCount` spawn attempts, all within the 16-block gate"
    );

    let dug: Vec<_> = live
        .iter()
        .filter(|p| p.layer == FxLayer::Terrain)
        .collect();
    assert_eq!(dug.len(), 1, "`count = 0` is exactly one particle");

    // The point of A14-04's `IconLookup`: without an implementor this particle is dropped by
    // `packet::build`'s `?`, so its mere existence is the proof. Assert the rect is a real
    // stitched sub-rect of the atlas rather than a `0..1` placeholder, so a lookup that silently
    // fell back to "the whole atlas" would fail here.
    let icon = dug[0]
        .icon
        .expect("a Terrain particle always carries an icon");
    assert!(
        icon.min_u < icon.max_u && icon.min_v < icon.max_v,
        "non-degenerate rect: {icon:?}"
    );
    assert!(
        icon.max_u - icon.min_u < 0.5 && icon.max_v - icon.min_v < 0.5,
        "a single 16px sprite in the real (>=512px) atlas, not the whole sheet: {icon:?}"
    );
    assert!(
        icon.min_u >= 0.0 && icon.max_u <= 1.0 && icon.min_v >= 0.0 && icon.max_v <= 1.0,
        "inside the atlas: {icon:?}"
    );

    rig.finish();
}

/// java: `RenderGlobal.spawnParticle` (`:2077`) — `d0*d0 + d1*d1 + d2*d2 > 256.0D` measured from
/// the render-view entity drops the particle entirely. The control above already shows a
/// near-field packet arriving, so a zero here is the gate working rather than the wiring missing.
#[test]
fn a_particle_packet_beyond_sixteen_blocks_is_culled() {
    let Some(mut rig) = join_and_script(|mut peer| async move {
        let mut far = at_spawn(FLAME, 16, vec![]);
        far.z += 200.0;
        peer.send(&far).await;
        // A near packet behind it, so the wait below has something to key on that is not the
        // absence of the far one.
        peer.send(&at_spawn(BLOCK_CRACK, 0, vec![1 << 4])).await;
        peer
    }) else {
        eprintln!("skip: no GPU adapter or assets");
        return;
    };

    rig.wait_for("the trailing near-field packet to be applied", |client| {
        client
            .particles
            .particles()
            .iter()
            .any(|p| p.layer == FxLayer::Terrain)
    });

    assert_eq!(
        rig.client
            .particles
            .particles()
            .iter()
            .filter(|p| p.layer == FxLayer::Misc)
            .count(),
        0,
        "200 blocks away, longDistance unset: every one of the 16 attempts is culled"
    );

    rig.finish();
}
