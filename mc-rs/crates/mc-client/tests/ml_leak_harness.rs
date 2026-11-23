//! The leak harness: run one long-session cycle many times and
//! report, after every cycle, what the process holds.
//!
//! Per cycle it prints the live Rust heap (bytes and blocks, every thread, from this binary's
//! counting allocator), private bytes and working set, open handles, threads, and the GPU
//! allocator's live bytes and allocations (`Device::generate_allocator_report`). A leak is
//! growth that keeps coming back cycle after cycle; one-time warm-up (pipelines, arenas, caches
//! that fill once) shows up in the first cycles and then stops.
//!
//! With `ML_TRACE=1` it also records the call stack of every allocation made after the warm-up
//! and, at the end, lists the stacks whose blocks are still alive (`tests/ml_leak_support`). A
//! structure that grows once per cycle shows up as one site with N, 2N, ... blocks after N
//! traced cycles.
//!
//! Every scenario is an ignored test (GPU + vanilla jar). Run one with:
//!
//! ```text
//! MC_JARS_DIR=<jars> cargo test -p mc-client --test ml_leak_harness -- --ignored --nocapture reconnect
//! ```
//!
//! Allocation sites are only trustworthy with line tables (test binaries carry public symbols
//! only, and a private function would be reported as the nearest public one). Cargo keeps
//! artifacts for a different profile apart, so this rebuilds nothing else:
//!
//! ```text
//! cargo --config 'profile.test.debug="line-tables-only"' test -p mc-client \
//!     --test ml_leak_harness -- --ignored --nocapture reconnect     (with ML_TRACE=1)
//! ```
//!
//! Environment: `ML_CYCLES` (measured cycles, default 20), `ML_WARMUP` (default 3),
//! `ML_TRACE=1` (allocation sites), `ML_TRACE_SITES` (sites printed, default 25),
//! `ML_TRACE_FRAMES` (frames per site, default 14), `ML_GPU_NAMES=1` (GPU allocations the
//! measured cycles added or freed, by debug name), `ML_REPLAY` (an `.aeroreplay` for the
//! `replay` scenario).
//!
//! Scenarios: `reconnect` (join a loopback server, play, leave to the title screen),
//! `reconnect_recording` (the same with the Replay mod recording every session),
//! `reconnect_list` (leave to the server list instead, as the pause menu's Disconnect does),
//! `dimension` (respawn into the Nether and back with a fresh world each time, the way networks
//! move players between servers), `chat` (300 messages a cycle), `players` (50 new players a
//! cycle: tab list, entities, nameplates, then gone), `scoreboard` (objectives, scores, teams,
//! tab header/footer created and removed), `titles` (titles, subtitles, action bar, a named
//! wither's boss bar), `resize` (three window sizes a cycle), `packs` (a resource pack reload a
//! cycle), `menu` (the mod menu and its glass, open and closed), `shaders` (each shader pack on
//! in turn, then all off), `replay` (open, seek, close), `replay_play` (a recording played on
//! and on at 4x, 100 frames a cycle).
//!
//! The server is a loopback double written here (framing only; packet bodies use
//! `mc-protocol`'s own encoders), like `tests/common`'s: it proves what the client keeps, not
//! what a real server would send.

mod common;
mod ml_leak_support;

use std::sync::Mutex;
use std::time::Duration;

use common::{decode, gpu, FLOOR_Y, SPAWN, STONE};
use mc_client::app::state::GameState;
use mc_client::app::Client;
use mc_math::ChunkPos;
use mc_physics::state::MoveInput;
use mc_protocol::buffer::Angle;
use mc_protocol::login::{LoginStart, LoginSuccess};
use mc_protocol::packet::Packet;
use mc_protocol::play::s2c;
use mc_protocol::PacketBuffer;
use mc_render::{Camera, Gpu, PhaseList};
use ml_leak_support::{ProcessCounters, TracingAlloc};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpListener;
use uuid::Uuid;

#[global_allocator]
static GLOBAL: TracingAlloc = TracingAlloc;

/// One scenario at a time: they share the allocator's counters.
static SERIAL: Mutex<()> = Mutex::new(());

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------------------------
// The loopback server
// ---------------------------------------------------------------------------------------------

fn write_varint(out: &mut Vec<u8>, mut value: i32) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value = ((value as u32) >> 7) as i32;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            return;
        }
    }
}

/// One uncompressed frame: length, packet id, body.
fn frame<P: Packet>(packet: &P) -> Vec<u8> {
    let mut body = PacketBuffer::new();
    packet.write(&mut body);
    let mut payload = Vec::new();
    write_varint(&mut payload, P::ID);
    payload.extend_from_slice(body.as_slice());
    let mut out = Vec::with_capacity(payload.len() + 5);
    write_varint(&mut out, payload.len() as i32);
    out.extend_from_slice(&payload);
    out
}

async fn read_varint(read: &mut OwnedReadHalf) -> Option<i32> {
    let mut value = 0i32;
    for shift in (0..35).step_by(7) {
        let byte = read.read_u8().await.ok()?;
        value |= i32::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

async fn read_frame(read: &mut OwnedReadHalf) -> Option<(i32, Vec<u8>)> {
    let length = usize::try_from(read_varint(read).await?).ok()?;
    let mut frame = vec![0u8; length];
    read.read_exact(&mut frame).await.ok()?;
    let mut id = 0i32;
    let mut index = 0;
    for shift in (0..35).step_by(7) {
        let byte = *frame.get(index)?;
        index += 1;
        id |= i32::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            break;
        }
    }
    Some((id, frame[index..].to_vec()))
}

/// A chunk column with a stone floor in section 0 of `mask`; sky light only where the
/// dimension has it (java: `S21PacketChunkData` leaves it out in the Nether and the End).
fn chunk_bytes(mask: u16, sky_light: bool) -> Vec<u8> {
    let length = mc_world::expected_chunk_data_len(mask, sky_light, true);
    let mut data = vec![0u8; length];
    for z in 0..16usize {
        for x in 0..16usize {
            let index = (FLOOR_Y << 8) | (z << 4) | x;
            data[index * 2] = (STONE & 0xFF) as u8;
            data[index * 2 + 1] = (STONE >> 8) as u8;
        }
    }
    if sky_light {
        let sections = mask.count_ones() as usize;
        let start = sections * (8192 + 2048);
        for byte in &mut data[start..start + sections * 2048] {
            *byte = 0xFF;
        }
    }
    data
}

fn chunks() -> Vec<ChunkPos> {
    let mut out = Vec::new();
    for x in -1..=1 {
        for z in -1..=1 {
            out.push(ChunkPos { x, z });
        }
    }
    out
}

/// The world a join or a respawn sends: nine columns, all sixteen sections, then the spawn.
fn world_frames(sky_light: bool) -> Vec<Vec<u8>> {
    let mut frames: Vec<Vec<u8>> = chunks()
        .into_iter()
        .map(|chunk| {
            frame(&mc_protocol::chunk::ChunkData {
                chunk_x: chunk.x,
                chunk_z: chunk.z,
                ground_up_continuous: true,
                primary_bit_mask: 0xFFFF,
                data: chunk_bytes(0xFFFF, sky_light),
            })
        })
        .collect();
    frames.push(frame(&s2c::PlayerPositionLook {
        x: SPAWN.0,
        y: SPAWN.1,
        z: SPAWN.2,
        yaw: 0.0,
        pitch: 30.0,
        flags: 0,
    }));
    frames
}

enum Cmd {
    /// Accept the next connection and serve a join on it.
    Accept { entity_id: i32 },
    /// Write frames on the current connection; answer when flushed.
    Send(Vec<Vec<u8>>, std::sync::mpsc::Sender<()>),
}

async fn serve(listener: TcpListener, mut commands: tokio::sync::mpsc::UnboundedReceiver<Cmd>) {
    let mut writer: Option<OwnedWriteHalf> = None;
    while let Some(command) = commands.recv().await {
        match command {
            Cmd::Accept { entity_id } => {
                // The previous connection's half, if the client has not hung up on it yet.
                drop(writer.take());
                let (socket, _) = listener.accept().await.expect("accept");
                socket.set_nodelay(true).expect("nodelay");
                let (mut read, mut write) = socket.into_split();
                let _handshake = read_frame(&mut read).await.expect("handshake");
                let (id, body) = read_frame(&mut read).await.expect("login start");
                assert_eq!(id, LoginStart::ID);
                let login: LoginStart = decode(&body);
                // Everything the client says from here on is read and dropped, so its socket
                // never backs up into its own queues.
                tokio::spawn(async move {
                    let mut sink = vec![0u8; 16 * 1024];
                    while matches!(read.read(&mut sink).await, Ok(count) if count > 0) {}
                });
                let mut frames = vec![
                    frame(&LoginSuccess {
                        uuid: Uuid::from_u128(0x0000_0000_0000_4000_8000_0000_0000_0001),
                        username: login.username,
                    }),
                    frame(&s2c::JoinGame {
                        entity_id,
                        hardcore_mode: false,
                        game_mode: 0,
                        dimension: 0,
                        difficulty: 2,
                        max_players: 20,
                        level_type: "default".into(),
                        reduced_debug_info: false,
                    }),
                ];
                frames.extend(world_frames(true));
                for frame in frames {
                    write.write_all(&frame).await.expect("join write");
                }
                write.flush().await.expect("flush");
                writer = Some(write);
            }
            Cmd::Send(frames, done) => {
                if let Some(write) = writer.as_mut() {
                    for frame in frames {
                        if write.write_all(&frame).await.is_err() {
                            break;
                        }
                    }
                    let _ = write.flush().await;
                }
                let _ = done.send(());
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The client side
// ---------------------------------------------------------------------------------------------

struct Rig {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    port: u16,
    commands: tokio::sync::mpsc::UnboundedSender<Cmd>,
    next_entity_id: i32,
    /// Also run the shell's `before_ticks` hook every frame, as the real window loop does: the
    /// replay recorder and the replay viewer run there (`common::pump` does not call it).
    shell_hooks: bool,
    _dir: tempfile::TempDir,
}

impl Rig {
    /// `mod_settings`: the scratch game directory's mod settings file, when a scenario needs a
    /// mod switched on (the way `common::vanilla_title_screen` writes one).
    fn new(mod_settings: Option<&str>) -> Option<Rig> {
        let gpu = gpu(WIDTH, HEIGHT)?;
        let dir = tempfile::tempdir().expect("tempdir");
        if let Some(settings) = mod_settings {
            std::fs::write(dir.path().join(mc_mods::SETTINGS_FILE_NAME), settings)
                .expect("write the scratch mod settings");
        }
        let Some(config) = common::config(dir.path(), "Dev") else {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        };
        let mut client = Client::new(config, &mut |_, _| {}).expect("start");
        // Never reach the network: a real replay carries signed skin profiles, and the skin and
        // cape worker would otherwise fetch them from the texture host (what benchmarks do with
        // `MC_NO_SKIN_DOWNLOADS=1`).
        client.set_skin_downloads(false);
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, WIDTH, HEIGHT);
        let listener = client
            .runtime()
            .handle()
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let (commands, receiver) = tokio::sync::mpsc::unbounded_channel();
        client.runtime().handle().spawn(serve(listener, receiver));
        Some(Rig {
            client,
            gpu,
            phases,
            camera: Camera::default(),
            port,
            commands,
            next_entity_id: 40,
            shell_hooks: false,
            _dir: dir,
        })
    }

    fn pump(&mut self, frames: usize) {
        for _ in 0..frames {
            if self.shell_hooks {
                mc_render::ShellApp::before_ticks(&mut self.client, &mut self.camera);
            }
            if self.client.replay_is_active() {
                // The shell's tick does nothing while a replay is open (the viewer runs the
                // recorded ticks itself), so no live tick here either.
                common::pump_no_tick(
                    &mut self.client,
                    &mut self.gpu,
                    &mut self.phases,
                    &mut self.camera,
                );
            } else {
                common::pump(
                    &mut self.client,
                    &mut self.gpu,
                    &mut self.phases,
                    &mut self.camera,
                    &MoveInput::default(),
                );
            }
        }
    }

    fn pump_until(&mut self, what: &str, mut done: impl FnMut(&Client) -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            self.pump(1);
            if done(&self.client) {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {what}; machine={:?}, ui={}",
                self.client.machine.state(),
                self.client.ui.name()
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Write frames to the client and wait until they are on the socket.
    fn send(&mut self, frames: Vec<Vec<u8>>) {
        let (done, flushed) = std::sync::mpsc::channel();
        self.commands
            .send(Cmd::Send(frames, done))
            .expect("server task alive");
        flushed
            .recv_timeout(Duration::from_secs(30))
            .expect("frames written");
    }

    fn world_loaded(client: &Client, dimension: i32) -> bool {
        client.machine.state().is_in_world()
            && client.session().is_some_and(|session| {
                session.state().local.dimension == dimension
                    && session.world().loaded_chunk_count() == 9
            })
    }

    /// Connect, and play until the nine columns are in and meshed.
    fn join(&mut self) {
        assert!(matches!(self.client.machine.state(), GameState::Menu));
        self.next_entity_id += 1;
        self.commands
            .send(Cmd::Accept {
                entity_id: self.next_entity_id,
            })
            .expect("server task alive");
        let port = self.port;
        self.client.connect("127.0.0.1", port);
        self.pump_until("the world to load", |client| Rig::world_loaded(client, 0));
        self.pump(12);
    }

    /// Let other threads finish (mesh workers, the connection task's drop) and the GPU retire
    /// its work, so a sample sees what is really still held.
    fn settle(&mut self) {
        for _ in 0..3 {
            self.pump(1);
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

// ---------------------------------------------------------------------------------------------
// Samples
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Sample {
    live: i64,
    blocks: i64,
    process: ProcessCounters,
    gpu_bytes: u64,
    gpu_allocations: usize,
}

fn sample(rig: &mut Rig) -> Sample {
    rig.settle();
    let (gpu_bytes, gpu_allocations) = rig
        .gpu
        .device
        .generate_allocator_report()
        .map_or((0, 0), |report| {
            (report.total_allocated_bytes, report.allocations.len())
        });
    Sample {
        live: ml_leak_support::live_bytes(),
        blocks: ml_leak_support::live_blocks(),
        process: ProcessCounters::now(),
        gpu_bytes,
        gpu_allocations,
    }
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_048_576.0
}

fn print_header(scenario: &str) {
    println!("scenario {scenario}");
    println!(
        "cycle    live_kb   d_kb   blocks  d_blk  private_mb  ws_mb  handles  threads  \
         gpu_mb  gpu_allocs"
    );
}

fn print_row(cycle: usize, now: &Sample, previous: &Sample) {
    println!(
        "{cycle:5} {:10.1} {:6.1} {:8} {:6} {:11.1} {:6.1} {:8} {:8} {:7.1} {:11}",
        now.live as f64 / 1024.0,
        (now.live - previous.live) as f64 / 1024.0,
        now.blocks,
        now.blocks - previous.blocks,
        mb(now.process.private_bytes),
        mb(now.process.working_set),
        now.process.handles,
        now.process.threads,
        mb(now.gpu_bytes),
        now.gpu_allocations,
    );
}

/// Growth per cycle over the measured cycles, and over their second half alone (a plateau
/// reads ~0 there; a leak does not).
fn print_summary(samples: &[Sample]) {
    type Field = fn(&Sample) -> f64;
    let per_cycle = |field: Field, from: usize| -> f64 {
        let last = samples.len() - 1;
        if last <= from {
            return 0.0;
        }
        (field(&samples[last]) - field(&samples[from])) / (last - from) as f64
    };
    let half = (samples.len() - 1) / 2;
    let fields: [(&str, Field); 6] = [
        ("live heap KB", |s| s.live as f64 / 1024.0),
        ("live blocks", |s| s.blocks as f64),
        ("private MB", |s| mb(s.process.private_bytes)),
        ("handles", |s| f64::from(s.process.handles)),
        ("threads", |s| f64::from(s.process.threads)),
        ("GPU allocations", |s| s.gpu_allocations as f64),
    ];
    for (name, field) in fields {
        println!(
            "per cycle: {name:16} all measured {:+9.2}   second half {:+9.2}",
            per_cycle(field, 0),
            per_cycle(field, half)
        );
    }
}

/// GPU allocations by name: (count, bytes).
fn gpu_allocations_by_name(gpu: &Gpu) -> std::collections::BTreeMap<String, (i64, i64)> {
    let mut names = std::collections::BTreeMap::new();
    if let Some(report) = gpu.device.generate_allocator_report() {
        for allocation in &report.allocations {
            let entry: &mut (i64, i64) = names.entry(allocation.name.clone()).or_default();
            entry.0 += 1;
            entry.1 += allocation.size as i64;
        }
    }
    names
}

/// `ML_GPU_NAMES=1`: which GPU allocations (by debug name) the measured cycles added or freed.
fn print_gpu_name_changes(
    before: &std::collections::BTreeMap<String, (i64, i64)>,
    after: &std::collections::BTreeMap<String, (i64, i64)>,
) {
    println!("---- GPU allocations by name, measured cycles (count, KB) ----");
    let names: std::collections::BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    for name in names {
        let (count_before, bytes_before) = before.get(name).copied().unwrap_or_default();
        let (count_after, bytes_after) = after.get(name).copied().unwrap_or_default();
        if (count_before, bytes_before) != (count_after, bytes_after) {
            println!(
                "{:+4} allocations {:+10.1} KB  {name:?}",
                count_after - count_before,
                (bytes_after - bytes_before) as f64 / 1024.0
            );
        }
    }
}

/// Run `cycle` `ML_WARMUP` + `ML_CYCLES` times, sampling after each.
fn run(scenario: &str, setup: impl FnOnce(&mut Rig), cycle: impl FnMut(&mut Rig, usize)) {
    run_with(scenario, None, setup, cycle);
}

/// [`run`] with a mod settings file in the scratch game directory.
fn run_with(
    scenario: &str,
    mod_settings: Option<&str>,
    setup: impl FnOnce(&mut Rig),
    mut cycle: impl FnMut(&mut Rig, usize),
) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut rig) = Rig::new(mod_settings) else {
        return;
    };
    let warmup = env_usize("ML_WARMUP", 3);
    let cycles = env_usize("ML_CYCLES", 20);
    let trace = std::env::var("ML_TRACE").is_ok_and(|value| value == "1");
    let gpu_names = std::env::var("ML_GPU_NAMES").is_ok_and(|value| value == "1");
    setup(&mut rig);
    print_header(scenario);
    let mut previous = sample(&mut rig);
    let mut measured = Vec::new();
    let mut names_before = std::collections::BTreeMap::new();
    for index in 0..warmup + cycles {
        cycle(&mut rig, index);
        let now = sample(&mut rig);
        print_row(index, &now, &previous);
        previous = now;
        if index + 1 >= warmup {
            measured.push(now);
        }
        if index + 1 == warmup {
            if gpu_names {
                names_before = gpu_allocations_by_name(&rig.gpu);
            }
            if trace {
                ml_leak_support::clear_tracing();
                assert!(ml_leak_support::start_tracing(), "tracing unavailable");
            }
        }
    }
    ml_leak_support::stop_tracing();
    print_summary(&measured);
    if gpu_names {
        print_gpu_name_changes(&names_before, &gpu_allocations_by_name(&rig.gpu));
    }
    if trace {
        let sites = ml_leak_support::live_sites(env_usize("ML_TRACE_SITES", 25));
        ml_leak_support::print_sites(
            &format!("{scenario}: blocks allocated in the last {cycles} cycles, still alive"),
            &sites,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn reconnect() {
    run(
        "reconnect",
        |_| {},
        |rig, _| {
            rig.join();
            rig.client.disconnect();
            rig.pump(3);
        },
    );
}

/// `reconnect` with the Replay mod on, so every session is recorded to the scratch game
/// directory: its writer thread, file handles and buffers must all go with the session.
#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn reconnect_recording() {
    run_with(
        "reconnect_recording",
        Some(r#"{"format":1,"mods":{"replay":{"enabled":true}}}"#),
        |rig| rig.shell_hooks = true,
        |rig, _| {
            rig.join();
            rig.pump(20);
            rig.client.disconnect();
            rig.pump(3);
            // Evidence the scenario records at all: one file per session so far.
            println!(
                "        recordings in the game directory: {}",
                count_files(&rig._dir.path().join("replays"), "aeroreplay")
            );
        },
    );
}

/// Files with `extension` under `dir`, recursively.
fn count_files(dir: &std::path::Path, extension: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                count_files(&path, extension)
            } else {
                usize::from(path.extension().is_some_and(|ext| ext == extension))
            }
        })
        .sum()
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn reconnect_list() {
    run(
        "reconnect_list",
        |_| {},
        |rig, _| {
            rig.join();
            rig.client.disconnect_to_server_list();
            rig.pump(3);
        },
    );
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn dimension() {
    run("dimension", Rig::join, |rig, _| {
        for (dimension, sky_light) in [(-1, false), (0, true)] {
            let mut frames = vec![frame(&s2c::Respawn {
                dimension,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            })];
            frames.extend(world_frames(sky_light));
            rig.send(frames);
            rig.pump_until("the respawned world", |client| {
                Rig::world_loaded(client, dimension)
            });
            rig.pump(8);
        }
    });
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn chat() {
    run("chat", Rig::join, |rig, cycle| {
        let frames = (0..300)
            .map(|line| {
                let text = format!(
                    "cycle {cycle} line {line}: the quick brown fox jumps over the lazy dog, \
                     twice, so that this message wraps onto a second chat line"
                );
                frame(&s2c::Chat {
                    component: serde_json::json!({ "text": text }).to_string(),
                    position: if line % 10 == 0 { 2 } else { 0 },
                })
            })
            .collect();
        rig.send(frames);
        rig.pump(10);
        let chat = &rig.client.chat;
        assert!(
            chat.lines().len() <= chat.max_lines() && chat.received_count() <= chat.max_lines(),
            "chat must stay within its history length"
        );
    });
}

fn player_uuid(cycle: usize, index: usize) -> Uuid {
    Uuid::from_u128(
        0x1000_0000_0000_4000_8000_0000_0000_0000 | ((cycle as u128) << 16) | index as u128,
    )
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn players() {
    const PER_CYCLE: usize = 50;
    run("players", Rig::join, |rig, cycle| {
        let first_id = 10_000 + (cycle * PER_CYCLE) as i32;
        let entries = (0..PER_CYCLE)
            .map(|index| s2c::AddPlayerEntry {
                uuid: player_uuid(cycle, index),
                username: format!("p{cycle}x{index}"),
                properties: Vec::new(),
                game_mode: s2c::GameMode::Survival,
                ping: 20 + index as i32,
                display_name: (index % 2 == 0).then(|| {
                    serde_json::json!({ "text": format!("Player {cycle}/{index}") }).to_string()
                }),
            })
            .collect();
        let mut frames = vec![frame(&s2c::PlayerListItem::AddPlayer(entries))];
        for index in 0..PER_CYCLE {
            let angle = index as f64 / PER_CYCLE as f64 * std::f64::consts::TAU;
            frames.push(frame(&s2c::SpawnPlayer {
                entity_id: first_id + index as i32,
                player_uuid: player_uuid(cycle, index),
                x: ((SPAWN.0 + 4.0 * angle.cos()) * 32.0) as i32,
                y: (SPAWN.1 * 32.0) as i32,
                z: ((SPAWN.2 + 4.0 * angle.sin()) * 32.0) as i32,
                yaw: Angle(0),
                pitch: Angle(0),
                current_item: 0,
                metadata: Vec::new(),
            }));
        }
        rig.send(frames);
        rig.pump(12);
        rig.send(vec![
            frame(&s2c::DestroyEntities {
                entity_ids: (0..PER_CYCLE)
                    .map(|index| first_id + index as i32)
                    .collect(),
            }),
            frame(&s2c::PlayerListItem::RemovePlayer(
                (0..PER_CYCLE)
                    .map(|index| player_uuid(cycle, index))
                    .collect(),
            )),
        ]);
        // Past one prune period (100 ticks), so the per-player state of everyone who just left
        // is gone: the count is the local player's, not one per player ever seen.
        rig.pump(110);
        println!(
            "        per-player appearance entries: {}",
            rig.client.leak_harness_per_player_entries()
        );
    });
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn scoreboard() {
    run("scoreboard", Rig::join, |rig, cycle| {
        let objective = format!("obj{cycle}");
        let mut frames = vec![
            frame(&s2c::ScoreboardObjective {
                objective_name: objective.clone(),
                mode: s2c::ScoreboardObjectiveMode::Create {
                    display_name: format!("Game {cycle}"),
                    render_type: s2c::ScoreboardRenderType::Integer,
                },
            }),
            frame(&s2c::DisplayScoreboard {
                position: 1,
                objective_name: objective.clone(),
            }),
            frame(&s2c::PlayerListHeaderFooter {
                header: serde_json::json!({ "text": format!("header {cycle}") }).to_string(),
                footer: serde_json::json!({ "text": format!("footer {cycle}") }).to_string(),
            }),
        ];
        for line in 0..15 {
            let name = format!("\u{a7}{:x}line{cycle}x{line}", line % 16);
            frames.push(frame(&s2c::Teams {
                team_name: format!("t{cycle}x{line}"),
                action: s2c::TeamsAction::Create {
                    info: s2c::TeamInfo {
                        display_name: format!("team {line}"),
                        prefix: format!("\u{a7}a{cycle}"),
                        suffix: format!("\u{a7}7{line}"),
                        friendly_flags: 0,
                        name_tag_visibility: "always".into(),
                        color: -1,
                    },
                    players: vec![name.clone()],
                },
            }));
            frames.push(frame(&s2c::UpdateScore {
                player_name: name,
                objective_name: objective.clone(),
                action: s2c::UpdateScoreAction::Change { value: line },
            }));
        }
        rig.send(frames);
        rig.pump(8);
        let mut frames = Vec::new();
        for line in 0..15 {
            let name = format!("\u{a7}{:x}line{cycle}x{line}", line % 16);
            frames.push(frame(&s2c::UpdateScore {
                player_name: name,
                objective_name: objective.clone(),
                action: s2c::UpdateScoreAction::Remove,
            }));
            frames.push(frame(&s2c::Teams {
                team_name: format!("t{cycle}x{line}"),
                action: s2c::TeamsAction::Remove,
            }));
        }
        frames.push(frame(&s2c::ScoreboardObjective {
            objective_name: objective,
            mode: s2c::ScoreboardObjectiveMode::Remove,
        }));
        rig.send(frames);
        rig.pump(4);
    });
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn titles() {
    run("titles", Rig::join, |rig, cycle| {
        let wither = 50_000 + cycle as i32;
        rig.send(vec![
            frame(&s2c::Title::SetTimes {
                fade_in_ticks: 2,
                stay_ticks: 10,
                fade_out_ticks: 2,
            }),
            frame(&s2c::Title::SetTitle {
                text: serde_json::json!({ "text": format!("Round {cycle}") }).to_string(),
            }),
            frame(&s2c::Title::SetSubtitle {
                text: serde_json::json!({ "text": format!("subtitle {cycle}") }).to_string(),
            }),
            frame(&s2c::Chat {
                component: serde_json::json!({ "text": format!("action bar {cycle}") }).to_string(),
                position: 2,
            }),
            frame(&s2c::SpawnMob {
                entity_id: wither,
                entity_type: 64,
                x: (SPAWN.0 * 32.0) as i32,
                y: ((SPAWN.1 + 3.0) * 32.0) as i32,
                z: ((SPAWN.2 + 6.0) * 32.0) as i32,
                yaw: Angle(0),
                pitch: Angle(0),
                head_pitch: Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: vec![mc_protocol::metadata::MetadataEntry {
                    index: 2,
                    value: mc_protocol::metadata::MetadataValue::String(format!("Boss {cycle}")),
                }],
            }),
        ]);
        rig.pump(12);
        rig.send(vec![
            frame(&s2c::DestroyEntities {
                entity_ids: vec![wither],
            }),
            frame(&s2c::Title::Clear),
            frame(&s2c::Title::Reset),
        ]);
        rig.pump(4);
    });
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn resize() {
    run("resize", Rig::join, |rig, _| {
        for (width, height) in [(640, 360), (320, 240), (WIDTH, HEIGHT)] {
            // As the shell applies a resize: the GPU's targets, the
            // camera, every phase's screen-sized targets, then the app.
            let rebuilt = rig.gpu.resize(width, height);
            let (width, height) = (rig.gpu.config.width, rig.gpu.config.height);
            rig.camera.set_viewport(width, height);
            rig.camera.update();
            if rebuilt {
                rig.phases.resize(&rig.gpu, width, height);
                let Rig { client, gpu, .. } = rig;
                mc_render::ShellApp::resized(client, gpu, width, height);
            }
            rig.pump(4);
        }
    });
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn packs() {
    run(
        "packs",
        |_| {},
        |rig, _| {
            let Rig {
                client,
                gpu,
                phases,
                ..
            } = rig;
            client
                .reload_resource_packs(Some(&*gpu), Some(phases), &mut |_, _| {})
                .expect("reload");
            rig.pump(4);
        },
    );
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn menu() {
    run("menu", Rig::join, |rig, _| {
        rig.client.leak_harness_open_mod_menu();
        rig.pump(12);
        rig.client.leak_harness_close_mod_menu();
        // The closing animation, then the phase switches itself off.
        rig.pump(20);
    });
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement"]
fn shaders() {
    run("shaders", Rig::join, |rig, _| {
        for pack in ["shaders_classic", "shaders_dreamy", "shaders_underwater"] {
            rig.client.mod_settings_mut().set_enabled(pack, true);
            rig.pump(8);
        }
        for pack in ["shaders_classic", "shaders_dreamy", "shaders_underwater"] {
            rig.client.mod_settings_mut().set_enabled(pack, false);
        }
        rig.pump(8);
    });
}

#[test]
#[ignore = "GPU + vanilla jar + ML_REPLAY; a measurement"]
fn replay() {
    let Some(path) = std::env::var_os("ML_REPLAY").map(std::path::PathBuf::from) else {
        eprintln!("skip: set ML_REPLAY to an .aeroreplay file");
        return;
    };
    run(
        "replay",
        |_| {},
        |rig, _| {
            rig.client.replay_open(&path).expect("open the replay");
            // The viewer is driven the way the exporter and the seek tests drive it: land on a
            // time, then draw frames there (start, the middle, a quarter, the end).
            let (start, end) = rig.client.replay_view_range();
            let span = end - start;
            for target in [start, start + span / 2, start + span / 4, end] {
                rig.client.replay_render_at(target);
                rig.pump(8);
            }
            assert!(rig.client.replay_is_active());
            rig.client.replay_close();
            rig.pump(3);
        },
    );
}

/// The replay viewer playing a real recording for a long stretch (4x speed, starting over at
/// the end), as the window loop drives it: players and NPCs coming and going, chat and
/// scoreboards changing, the way a long session does. One cycle is 100 frames of playback.
#[test]
#[ignore = "GPU + vanilla jar + ML_REPLAY; a measurement"]
fn replay_play() {
    let Some(path) = std::env::var_os("ML_REPLAY").map(std::path::PathBuf::from) else {
        eprintln!("skip: set ML_REPLAY to an .aeroreplay file");
        return;
    };
    run(
        "replay_play",
        |rig| {
            rig.shell_hooks = true;
            rig.client.replay_open(&path).expect("open the replay");
            rig.client.replay_set_speed(4.0);
            rig.client.replay_play();
        },
        |rig, _| {
            if !rig.client.replay_is_playing() {
                rig.client.replay_play();
            }
            rig.pump(100);
            println!(
                "        replay at {:.1} s, per-player appearance entries: {}",
                rig.client.replay_time() as f64 / 1_000_000.0,
                rig.client.leak_harness_per_player_entries()
            );
        },
    );
}

// ---------------------------------------------------------------------------------------------
// The tracer's own control (no GPU, no jar: runs in every `cargo test`)
// ---------------------------------------------------------------------------------------------

/// Leaks three 4000-byte blocks per call, from a frame with a findable name. Unmangled and
/// exported, so it has a public symbol of its own even in the default test build (which keeps
/// public symbols only); a private function would be named after its nearest public neighbour.
#[inline(never)]
#[no_mangle]
fn ml_deliberate_leak_site(kept: &mut Vec<Vec<u8>>) {
    for _ in 0..3 {
        kept.push(std::hint::black_box(vec![7u8; 4000]));
    }
}

/// Allocates and frees: must not appear among the live sites.
#[inline(never)]
fn ml_churn_site() {
    for size in [100usize, 5000, 70_000] {
        drop(std::hint::black_box(vec![1u8; size]));
    }
}

/// A "no leak" verdict is only worth something if the instrument can name a leak: six traced
/// cycles each keep three blocks from one function and churn others; the tracer must report
/// that function's stack with exactly eighteen live blocks.
#[test]
fn the_tracer_names_a_deliberate_per_cycle_leak() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !cfg!(windows) {
        return;
    }
    let mut kept = Vec::with_capacity(64);
    let before = ml_leak_support::live_blocks();
    ml_leak_support::clear_tracing();
    assert!(ml_leak_support::start_tracing());
    for _ in 0..6 {
        ml_deliberate_leak_site(&mut kept);
        ml_churn_site();
    }
    ml_leak_support::stop_tracing();
    let sites = ml_leak_support::live_sites(50);
    ml_leak_support::print_sites("tracer control", &sites);
    let leak = sites
        .iter()
        .find(|site| {
            site.frames
                .iter()
                .any(|frame| frame.contains("ml_deliberate_leak_site"))
        })
        .expect("the deliberate leak's stack must be named");
    assert_eq!(leak.blocks, 18, "three blocks per cycle, six cycles");
    assert_eq!(leak.bytes, 18 * 4000);
    assert!(
        !sites.iter().any(|site| site
            .frames
            .iter()
            .any(|frame| frame.contains("ml_churn_site"))),
        "freed blocks must not be reported"
    );
    assert!(ml_leak_support::live_blocks() - before >= 18);
    assert_eq!(kept.len(), 18);
    ml_leak_support::clear_tracing();
}

#[test]
fn demangles_symbols_without_their_hash() {
    // A v0 symbol as this crate's test binaries carry them.
    assert_eq!(
        ml_leak_support::demangle(
            "_RNvXsJ_NtCs55qC6OcLGgs_4core6optionINtB5_6OptionRNtNtNtNtCseVtxHSQACov_\
             11mc_protocol4play3s2c6batch315ProfilePropertyENtNtNtB7_3ops9try_trait3Try6branch\
             Cs4C63HuTBCcJ_9mc_client"
        ),
        "<core::option::Option<&mc_protocol::play::s2c::batch3::ProfileProperty> as \
         core::ops::try_trait::Try>::branch"
    );
    assert_eq!(
        ml_leak_support::demangle("_ZN9mc_client3app7connect17h0123456789abcdefE"),
        "mc_client::app::connect"
    );
    assert_eq!(ml_leak_support::demangle("plain_c_name"), "plain_c_name");
}
