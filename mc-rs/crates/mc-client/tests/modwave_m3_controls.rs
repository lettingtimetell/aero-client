//! Toggle Sprint must send exactly the packets of a player who
//! physically holds the sprint key. The user's words: "if toggled should function as if im
//! always holding my sprint key". Servers run anticheat, so "about the same" is not enough.
//!
//! Each run joins the loopback double (`tests/common`), then drives the real shell sequence
//! (`ShellApp::before_ticks`, one `tick`, one `frame`) for a fixed key script, and records every
//! serverbound packet. Runs are compared byte for byte from the first `C0B START_SPRINTING`
//! onwards; before it the client is idle, and the number of idle ticks while joining is not
//! deterministic.
//!
//! * **Held vs tapped.** Vanilla (mod off) with Ctrl held from tick 1 against the mod on with
//!   Ctrl tapped on ticks 1-2. W is held from tick 3, released for two ticks mid-run and pressed
//!   again, so the re-sprint after a stop is covered too.
//! * **Toggling off.** Vanilla releasing Ctrl after tick 10 against the mod with a second tap
//!   on ticks 11-12; W is released and pressed again afterwards, where neither may re-sprint.
//!
//! The mod is switched on through `<gameDir>/mc-rs-mods.json` in a temp dir, exactly as the user's
//! settings file would. Skips without the vanilla jar (`MC_JARS_DIR`) or a GPU adapter.

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_protocol::packet::Packet;
use mc_protocol::play::c2s;
use mc_render::input::keys;
use mc_render::{FrameCtx, FrameStats, GameSettings, InputState, PhaseList, ShellApp, TickCtx};
use tokio::net::TcpListener;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const ENTITY_ID: i32 = 77;
/// Scripted shell iterations, one tick each.
const TICKS: u32 = 30;

/// What the sprint key does in a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ctrl {
    /// Held from tick 1 to the end.
    HeldThroughout,
    /// Held on ticks 1-10, then released.
    HeldThenReleased,
    /// Pressed on ticks 1-2 only (one tap).
    TapOnce,
    /// Pressed on ticks 1-2 and again on 11-12 (two taps: on, then off).
    TapTwice,
}

impl Ctrl {
    fn down(self, tick: u32) -> bool {
        match self {
            Ctrl::HeldThroughout => true,
            Ctrl::HeldThenReleased => tick <= 10,
            Ctrl::TapOnce => tick <= 2,
            Ctrl::TapTwice => tick <= 2 || (11..=12).contains(&tick),
        }
    }
}

/// W: held on 3-10, released on 11-12... (`gap`) and held again until 20, then released.
fn forward_down(tick: u32, gap: std::ops::RangeInclusive<u32>) -> bool {
    (3..=20).contains(&tick) && !gap.contains(&tick)
}

/// Join, play the script, disconnect; every serverbound `(id, body)` the double received.
fn run(
    mod_on: bool,
    ctrl: Ctrl,
    gap: std::ops::RangeInclusive<u32>,
) -> Option<Vec<(i32, Vec<u8>)>> {
    let mut gpu = gpu(WIDTH, HEIGHT)?;
    let dir = tempfile::tempdir().expect("tempdir");
    if mod_on {
        std::fs::write(
            dir.path().join(mc_mods::SETTINGS_FILE_NAME),
            r#"{"format":1,"mods":{"toggle_sprint":{"enabled":true}}}"#,
        )
        .expect("write the temp mod settings");
    }
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
        return None;
    };
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = mc_client::app::driver::menu_camera();
    camera.set_viewport(WIDTH, HEIGHT);

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let server = client.runtime().handle().spawn(async move {
        let mut peer = accept_and_join(listener, ENTITY_ID).await;
        peer.record_until_closed().await;
        peer.seen
    });

    client.connect("127.0.0.1", port);
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(30),
        "the client to reach the world",
        |client| {
            client.machine.state().is_in_world()
                && client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|session| session.world().is_chunk_loaded(CHUNK))
        },
    );

    let mut input = InputState::default();
    let mut settings = GameSettings::default();
    let stats = FrameStats::new();
    let sprint = settings.keys.get(keys::SPRINT).expect("sprint").binding();
    let forward = settings.keys.get(keys::FORWARD).expect("forward").binding();
    for tick in 1..=TICKS {
        // The shell applies key edges to the bindings between iterations.
        settings.keys.set_state(sprint, ctrl.down(tick));
        settings
            .keys
            .set_state(forward, forward_down(tick, gap.clone()));
        ShellApp::before_ticks(&mut client, &mut camera);
        ShellApp::tick(
            &mut client,
            &mut TickCtx {
                input: &mut input,
                settings: &mut settings,
                camera: &mut camera,
                tick: u64::from(tick),
            },
        );
        ShellApp::frame(
            &mut client,
            &mut FrameCtx {
                gpu: &gpu,
                phases: &mut phases,
                camera: &mut camera,
                input: &mut input,
                settings: &settings,
                partial_ticks: 0.0,
                stats: &stats,
            },
        );
    }

    client.disconnect();
    for _ in 0..10 {
        pump(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &MoveInput::default(),
        );
    }
    let seen = client.runtime().handle().block_on(async {
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .expect("the peer must finish")
            .expect("the peer task must not panic")
    });
    Some(seen)
}

fn sprint_actions(packets: &[(i32, Vec<u8>)]) -> Vec<c2s::EntityActionType> {
    packets
        .iter()
        .filter(|(id, _)| *id == c2s::EntityAction::ID)
        .map(|(_, body)| common::decode::<c2s::EntityAction>(body).action)
        .collect()
}

/// Everything from the first `C0B START_SPRINTING` on. Panics if there is none: then the run
/// never sprinted, which for these scripts means the mod or the vanilla path is broken.
fn from_first_sprint(packets: &[(i32, Vec<u8>)], label: &str) -> Vec<(i32, Vec<u8>)> {
    let start = packets
        .iter()
        .position(|(id, body)| {
            *id == c2s::EntityAction::ID
                && common::decode::<c2s::EntityAction>(body).action
                    == c2s::EntityActionType::StartSprinting
        })
        .unwrap_or_else(|| panic!("{label}: never started sprinting; ids {:?}", ids(packets)));
    packets[start..].to_vec()
}

fn ids(packets: &[(i32, Vec<u8>)]) -> Vec<String> {
    packets.iter().map(|(id, _)| format!("{id:#04x}")).collect()
}

fn assert_same_bytes(vanilla: &[(i32, Vec<u8>)], modded: &[(i32, Vec<u8>)], what: &str) {
    let vanilla = from_first_sprint(vanilla, "vanilla");
    let modded = from_first_sprint(modded, "toggle sprint");
    assert!(
        vanilla.len() > TICKS as usize / 2,
        "{what}: the comparison must cover the scripted ticks, got {} packets",
        vanilla.len()
    );
    if vanilla != modded {
        let first = vanilla
            .iter()
            .zip(&modded)
            .position(|(a, b)| a != b)
            .unwrap_or(vanilla.len().min(modded.len()));
        panic!(
            "{what}: toggle sprint's packets differ from a held key at packet {first}\n\
             vanilla ids {:?}\nmodded  ids {:?}",
            ids(&vanilla),
            ids(&modded)
        );
    }
}

#[test]
fn a_toggled_sprint_sends_exactly_the_packets_of_a_held_key() {
    // A: vanilla, Ctrl held. B: the mod, Ctrl tapped once.
    let Some(held) = run(false, Ctrl::HeldThroughout, 11..=12) else {
        return;
    };
    let Some(tapped) = run(true, Ctrl::TapOnce, 11..=12) else {
        return;
    };
    use c2s::EntityActionType::{StartSprinting, StopSprinting};
    // The W gap stops the sprint, the re-press starts it again (the held key re-sprints at
    // once), and the final release stops it: three edges after the first start.
    assert_eq!(
        sprint_actions(&held),
        [StartSprinting, StopSprinting, StartSprinting, StopSprinting],
        "the vanilla control run itself"
    );
    assert_same_bytes(&held, &tapped, "held vs tapped");
}

#[test]
fn toggling_off_behaves_exactly_like_releasing_the_key() {
    // Vanilla lets go of Ctrl after tick 10; the mod taps it again on 11-12. W is released on
    // 13-14 and pressed again: with the key "up" neither re-sprints (vanilla only arms the
    // double-tap timer).
    let Some(released) = run(false, Ctrl::HeldThenReleased, 13..=14) else {
        return;
    };
    let Some(toggled_off) = run(true, Ctrl::TapTwice, 13..=14) else {
        return;
    };
    use c2s::EntityActionType::{StartSprinting, StopSprinting};
    assert_eq!(
        sprint_actions(&released),
        [StartSprinting, StopSprinting],
        "the vanilla control run itself"
    );
    assert_same_bytes(&released, &toggled_off, "released vs toggled off");
}
