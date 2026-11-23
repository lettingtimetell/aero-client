//! Measures the two costs this lane removes.
//!
//! 1. **First-play stall.** A long `.ogg` registered as an ordinary (non-streaming) effect, the
//!    way a resource pack that swaps a hit or level-up sound for a song does it, is played once
//!    inside a simulated game loop. Each "frame" runs only the engine calls the client makes
//!    (`play` on the frame the sound starts, `update_all_sounds` every third frame); the mock
//!    backend's mixing step stands in for the audio thread and is not timed. Reported: the frame
//!    that starts the sound, the slowest other frame, and the parts of the old inline path
//!    (read, decode) on their own.
//! 2. **Start-up scan.** `SoundRegistry::load` over the vanilla asset index, with every pack
//!    wrapped in a counter of `contains`/`open` calls (each is a file-system call for the object
//!    store; `open` on a zip entry reads and inflates the whole file).
//!
//! Silent by construction: kira's `MockBackend` never opens an output device.
//!
//! ```text
//! SOUND_STALL_OGG=<long .ogg>      required for part 1
//! MC_JARS_DIR=<jars dir>           optional; part 2 and the vanilla comparison need it
//! cargo run -p mc-audio --example sound_stall
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use mc_assets::{
    FolderResourcePack, LanguageInfo, ObjectStorePack, PackMetadata, ResourceLocation,
    ResourceManager, ResourcePack,
};
use mc_audio::{AudioEngine, Listener, MockBackend, PlayRequest, SoundRegistry};

const FRAMES: usize = 180;
const PLAY_AT: usize = 90;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

/// Counts every `contains` and `open` the registry load makes on one pack.
struct Counting {
    inner: Box<dyn ResourcePack>,
    contains: Arc<AtomicUsize>,
    opens: Arc<AtomicUsize>,
}

impl ResourcePack for Counting {
    fn open(&self, l: &ResourceLocation) -> Option<Box<dyn std::io::Read>> {
        self.opens.fetch_add(1, Ordering::Relaxed);
        self.inner.open(l)
    }

    fn contains(&self, l: &ResourceLocation) -> bool {
        self.contains.fetch_add(1, Ordering::Relaxed);
        self.inner.contains(l)
    }

    // Forwarded, or the wrapper would hide the index from the registry
    // load. In memory only, so not counted as a storage call.
    fn lists(&self, l: &ResourceLocation) -> Option<bool> {
        self.inner.lists(l)
    }

    // Likewise: the path lets the prewarm's worker read the file. No storage call.
    fn file_path(&self, l: &ResourceLocation) -> Option<PathBuf> {
        self.inner.file_path(l)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn namespaces(&self) -> Vec<String> {
        self.inner.namespaces()
    }

    fn metadata(&self) -> Option<&PackMetadata> {
        self.inner.metadata()
    }

    fn language_descriptors(&self) -> &[LanguageInfo] {
        self.inner.language_descriptors()
    }

    fn list_files(&self, namespace: &str, dir_prefix: &str) -> Vec<String> {
        self.inner.list_files(namespace, dir_prefix)
    }
}

/// A folder pack holding `bytes` as the only sound of a non-streaming `test.long` event.
fn long_sound_pack(dir: &Path, bytes: &[u8]) -> FolderResourcePack {
    let assets = dir.join("assets").join("minecraft");
    std::fs::create_dir_all(assets.join("sounds").join("test")).expect("pack dirs");
    std::fs::write(
        assets.join("sounds.json"),
        br#"{"test.long": {"category": "master", "sounds": ["test/long"]}}"#,
    )
    .expect("sounds.json");
    std::fs::write(assets.join("sounds").join("test").join("long.ogg"), bytes).expect("ogg");
    FolderResourcePack::open(dir).expect("folder pack")
}

struct FrameReport {
    play_frame: Duration,
    worst_other: Duration,
    median_other: Duration,
    /// How the sound was started: "static" (decoded cache) or "stream".
    kind: &'static str,
}

/// Plays `event` once at frame [`PLAY_AT`] of a [`FRAMES`]-frame loop and times every frame.
fn frame_loop(
    engine: &mut AudioEngine<MockBackend>,
    rm: &ResourceManager,
    event: &str,
) -> FrameReport {
    let listener = Listener::origin();
    let request = PlayRequest::ambient(ResourceLocation::minecraft(event));
    let mut frames = Vec::with_capacity(FRAMES);
    let mut kind = "";
    for frame in 0..FRAMES {
        let started = Instant::now();
        if frame == PLAY_AT {
            match engine.play(&request, &listener, rm) {
                Ok(Some(mc_audio::PlayingSound::Static(_))) => kind = "static",
                Ok(Some(mc_audio::PlayingSound::Streaming(_))) => kind = "stream",
                other => panic!("{event} did not start: {other:?}"),
            }
        }
        if frame % 3 == 0 {
            let errors = engine.update_all_sounds(&listener, rm);
            assert!(errors.is_empty(), "update errors: {errors:?}");
        }
        frames.push(started.elapsed());
        // The audio thread's turn, untimed: the client never runs this on the game thread.
        if let Some(backend) = engine.backend_mut() {
            backend.on_start_processing();
            backend.process();
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let play_frame = frames[PLAY_AT];
    let mut others: Vec<Duration> = frames
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != PLAY_AT)
        .map(|(_, d)| *d)
        .collect();
    others.sort();
    FrameReport {
        play_frame,
        worst_other: *others.last().expect("frames"),
        median_other: others[others.len() / 2],
        kind,
    }
}

fn first_play(ogg: &Path) {
    let bytes = std::fs::read(ogg).expect("read SOUND_STALL_OGG");
    println!("== first play of a long non-streaming sound ==");
    println!("file: {} ({} bytes)", ogg.display(), bytes.len());

    // The old inline path's parts, measured on their own.
    let started = Instant::now();
    let decoded = mc_audio::ogg::decode_all(&bytes, "test").expect("decodes");
    let decode = started.elapsed();
    let started = Instant::now();
    let frames = decoded.to_frames();
    let convert = started.elapsed();
    println!(
        "clip: {:.1} s, {} frames; decode_all {:.1} ms, to_frames {:.1} ms",
        decoded.duration_secs(),
        frames.len(),
        ms(decode),
        ms(convert)
    );
    drop(frames);

    let temp = tempfile::tempdir().expect("tempdir");
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(long_sound_pack(temp.path(), &bytes)));
    let started = Instant::now();
    let loaded = rm
        .load(&ResourceLocation::minecraft("sounds/test/long.ogg"))
        .expect("loads");
    println!("read through the pack: {:.2} ms", ms(started.elapsed()));
    drop(loaded);

    let registry = SoundRegistry::load(&rm).expect("registry");
    for run in 1..=3 {
        // A fresh engine each run: an empty decode cache, as at the first play of a session.
        let mut engine = AudioEngine::<MockBackend>::new(registry.clone(), 1);
        let first = frame_loop(&mut engine, &rm, "test.long");
        // Same engine again: the second play of the same sound.
        let second = frame_loop(&mut engine, &rm, "test.long");
        println!(
            "run {run}: first-play frame {:.2} ms, {} (other frames: worst {:.3} ms, median \
             {:.3} ms); second-play frame {:.2} ms, {}",
            ms(first.play_frame),
            first.kind,
            ms(first.worst_other),
            ms(first.median_other),
            ms(second.play_frame),
            second.kind,
        );
    }
    println!("streamed first play, part by part:");
    stream_breakdown(&rm, "sounds/test/long.ogg");
}

/// The parts of a streamed first play, each timed alone: reading through the pack, the last-page
/// scan, opening the Vorbis stream (header parse) and handing the stream to kira (which spawns
/// its decode thread).
fn stream_breakdown(rm: &ResourceManager, file: &str) {
    use kira::sound::streaming::StreamingSoundData;
    let location = ResourceLocation::minecraft(file);
    let mut manager = kira::AudioManager::<MockBackend>::new(Default::default()).expect("mock");
    for run in 1..=3 {
        let started = Instant::now();
        let bytes = rm.load(&location).expect("loads");
        let read = started.elapsed();
        let started = Instant::now();
        let bytes: Arc<[u8]> = Arc::from(bytes);
        let to_arc = started.elapsed();
        let started = Instant::now();
        let frames = mc_audio::ogg::frames_from_last_page(&bytes);
        let scan = started.elapsed();
        let started = Instant::now();
        let decoder = mc_audio::ogg::VorbisDecoder::new(Arc::clone(&bytes), file).expect("opens");
        let open = started.elapsed();
        let started = Instant::now();
        let mut handle = manager
            .play(StreamingSoundData::from_decoder(decoder))
            .expect("plays");
        let play = started.elapsed();
        handle.stop(Default::default());
        // What kira's stream thread does before the first frame exists: open the Vorbis reader
        // (setup header) and decode the first packet. Timed here on a twin decoder.
        let mut twin = mc_audio::ogg::VorbisDecoder::new(Arc::clone(&bytes), file).expect("opens");
        let started = Instant::now();
        let mut first = Vec::new();
        while first.len() < 2 {
            first.extend(kira::sound::streaming::Decoder::decode(&mut twin).expect("decodes"));
        }
        let first_packet = started.elapsed();
        println!(
            "  {file} run {run}: read {:.3} ms, Arc copy {:.3} ms, last page {:.3} ms ({frames:?}), \
             Vorbis open {:.3} ms, kira play {:.3} ms | on kira's thread before the first frame: \
             {:.3} ms",
            ms(read),
            ms(to_arc),
            ms(scan),
            ms(open),
            ms(play),
            ms(first_packet)
        );
    }
}

fn vanilla_manager(jars: &Path) -> Option<(ResourceManager, Arc<AtomicUsize>, Arc<AtomicUsize>)> {
    let assets = jars.join("assets");
    let pack = ObjectStorePack::open(&assets, "1.8").ok()?;
    let contains = Arc::new(AtomicUsize::new(0));
    let opens = Arc::new(AtomicUsize::new(0));
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(Counting {
        inner: Box::new(pack),
        contains: Arc::clone(&contains),
        opens: Arc::clone(&opens),
    }));
    Some((rm, contains, opens))
}

fn startup_scan(jars: &Path) {
    println!("== start-up: SoundRegistry::load over the vanilla 1.8 asset index ==");
    let Some((rm, contains, opens)) = vanilla_manager(jars) else {
        println!("SKIP: no assets/indexes/1.8.json under {}", jars.display());
        return;
    };
    let mut times = Vec::new();
    for run in 0..5 {
        contains.store(0, Ordering::Relaxed);
        opens.store(0, Ordering::Relaxed);
        let started = Instant::now();
        let registry = SoundRegistry::load(&rm).expect("registry");
        let took = started.elapsed();
        times.push(took);
        if run == 0 {
            println!(
                "events {}, pack calls during load: contains {}, open {}",
                registry.len(),
                contains.load(Ordering::Relaxed),
                opens.load(Ordering::Relaxed)
            );
        }
    }
    times.sort();
    println!(
        "load time (warm OS cache, 5 runs): min {:.1} ms, median {:.1} ms, max {:.1} ms",
        ms(times[0]),
        ms(times[2]),
        ms(times[4])
    );

    // The biggest vanilla non-streaming effect, for scale: its first play decodes inline today.
    let registry = SoundRegistry::load(&rm).expect("registry");
    let mut engine = AudioEngine::<MockBackend>::new(registry, 1);
    let report = frame_loop(&mut engine, &rm, "mob.enderdragon.end");
    println!(
        "vanilla mob.enderdragon.end (257 KB) first-play frame {:.2} ms, {} (worst other {:.3} ms)",
        ms(report.play_frame),
        report.kind,
        ms(report.worst_other)
    );
    println!("streamed first play, part by part:");
    stream_breakdown(&rm, "sounds/mob/enderdragon/end.ogg");
    stream_breakdown(&rm, "sounds/random/pop.ogg");
}

/// The benchmark situation (`MC_NO_AUDIO`): an engine without a backend plays every vanilla
/// event once, so every file wins its first pick. Counts storage calls and times the plays.
fn silent_first_picks(jars: &Path) {
    println!("== first picks with no audio backend (MC_NO_AUDIO), every vanilla event ==");
    let Some((rm, contains, opens)) = vanilla_manager(jars) else {
        return;
    };
    let registry = SoundRegistry::load(&rm).expect("registry");
    let events: Vec<ResourceLocation> = registry.events().map(|(event, _)| event.clone()).collect();
    let mut engine = AudioEngine::<MockBackend>::silent(registry, 1);
    let listener = Listener::origin();
    for round in 1..=2 {
        contains.store(0, Ordering::Relaxed);
        opens.store(0, Ordering::Relaxed);
        let mut worst = Duration::ZERO;
        let started = Instant::now();
        for event in &events {
            let request = PlayRequest::ambient(event.clone());
            let one = Instant::now();
            let _ = engine.play(&request, &listener, &rm);
            worst = worst.max(one.elapsed());
        }
        println!(
            "round {round}: {} plays in {:.2} ms, slowest {:.3} ms; pack calls contains {}, open {}",
            events.len(),
            ms(started.elapsed()),
            ms(worst),
            contains.load(Ordering::Relaxed),
            opens.load(Ordering::Relaxed)
        );
    }
}

/// A copy of the client's `PREWARMED_SOUND_EVENTS` (`mc-client/src/app/audio.rs`).
const PREWARMED_SOUND_EVENTS: &[&str] = &[
    "random.",
    "dig.",
    "step.",
    "game.player.",
    "game.tnt.",
    "game.potion.",
    "note.",
    "mob.villager.",
    "fire.",
    "liquid.",
];

/// The client's start-up prewarm (`build_engine`, audio on): what it reads on the game thread.
fn startup_prewarm(jars: &Path) {
    println!("== start-up: the client's prewarm (audio on) ==");
    let Some((rm, contains, opens)) = vanilla_manager(jars) else {
        return;
    };
    let registry = SoundRegistry::load(&rm).expect("registry");
    for run in 1..=3 {
        let mut engine = AudioEngine::<MockBackend>::new(registry.clone(), 1);
        contains.store(0, Ordering::Relaxed);
        opens.store(0, Ordering::Relaxed);
        let started = Instant::now();
        let queued = engine.prewarm_static_sounds(&rm, |event| {
            event.namespace == "minecraft"
                && PREWARMED_SOUND_EVENTS
                    .iter()
                    .any(|prefix| event.path.starts_with(prefix))
        });
        let call = started.elapsed();
        let calls = (
            contains.load(Ordering::Relaxed),
            opens.load(Ordering::Relaxed),
        );
        let listener = Listener::origin();
        println!(
            "run {run}: prewarm call {:.2} ms on the calling thread; {queued} files queued; pack \
             calls on the calling thread: contains {}, open {}",
            ms(call),
            calls.0,
            calls.1,
        );
        // Let the worker finish (decodes, then the read-ahead), taking its results in every 50 ms
        // the way the client's ticks do, and time each of those drains.
        let mut worst_drain = Duration::ZERO;
        let mut drains = 0;
        let worker_started = Instant::now();
        while engine.decode_cache_stats().pending_files > 0 {
            std::thread::sleep(Duration::from_millis(50));
            let started = Instant::now();
            engine.drain_finished_decodes();
            worst_drain = worst_drain.max(started.elapsed());
            drains += 1;
        }
        let stats = engine.decode_cache_stats();
        println!(
            "  worker done {:.0} ms after the prewarm call; {drains} tick drains, slowest {:.3} ms; \
             {stats:?}",
            ms(worker_started.elapsed() + call),
            ms(worst_drain)
        );
        // First plays after it: a PvP effect (decoded) and one outside the PvP set (read ahead).
        // Neither should touch storage.
        for event in ["random.pop", "mob.enderdragon.end"] {
            contains.store(0, Ordering::Relaxed);
            opens.store(0, Ordering::Relaxed);
            let request = PlayRequest::ambient(ResourceLocation::minecraft(event));
            let started = Instant::now();
            let sound = engine.play(&request, &listener, &rm).expect("plays");
            let took = started.elapsed();
            println!(
                "  then {event} first play: {:.3} ms, {}; pack calls contains {}, open {}",
                ms(took),
                match sound {
                    Some(mc_audio::PlayingSound::Static(_)) => "static",
                    Some(mc_audio::PlayingSound::Streaming(_)) => "stream",
                    None => "none",
                },
                contains.load(Ordering::Relaxed),
                opens.load(Ordering::Relaxed)
            );
        }
        engine.stop_all_sounds();
    }
}

/// The same first play, from a zip resource pack (how packs usually ship), whose `open` reads and
/// inflates the whole entry. The pack must register `test.long` itself.
fn first_play_from_zip(zip: &Path) {
    println!("== first play from a zip pack: {} ==", zip.display());
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(zip).expect("zip pack"),
    ));
    let registry = SoundRegistry::load(&rm).expect("registry");
    for run in 1..=3 {
        let mut engine = AudioEngine::<MockBackend>::new(registry.clone(), 1);
        let first = frame_loop(&mut engine, &rm, "test.long");
        let second = frame_loop(&mut engine, &rm, "test.long");
        println!(
            "run {run}: first-play frame {:.2} ms, {}; second-play frame {:.2} ms, {}",
            ms(first.play_frame),
            first.kind,
            ms(second.play_frame),
            second.kind,
        );
    }
    stream_breakdown(&rm, "sounds/test/long.ogg");

    // As the client does after loading a pack: the prewarm takes the pack's own sounds (whatever
    // their event; the PvP event filter selects nothing here), then the first play mid-game.
    let mut engine = AudioEngine::<MockBackend>::new(registry, 1);
    let started = Instant::now();
    let queued = engine.prewarm_static_sounds(&rm, |_| false);
    let call = started.elapsed();
    while engine.decode_cache_stats().pending_files > 0 {
        std::thread::sleep(Duration::from_millis(50));
        engine.drain_finished_decodes();
    }
    let ready = started.elapsed();
    let first = frame_loop(&mut engine, &rm, "test.long");
    println!(
        "after the pack prewarm ({queued} queued, call {:.2} ms on the calling thread, ready \
         {:.0} ms later; {:?}): first-play frame {:.2} ms, {}",
        ms(call),
        ms(ready),
        engine.decode_cache_stats(),
        ms(first.play_frame),
        first.kind
    );
}

fn main() {
    match std::env::var_os("SOUND_STALL_OGG") {
        Some(path) => first_play(Path::new(&path)),
        None => println!("SKIP first play: set SOUND_STALL_OGG to a long .ogg"),
    }
    if let Some(zip) = std::env::var_os("SOUND_STALL_ZIP") {
        first_play_from_zip(Path::new(&zip));
    }
    match jars_dir() {
        Some(jars) => {
            startup_scan(&jars);
            silent_first_picks(&jars);
            startup_prewarm(&jars);
        }
        None => println!("SKIP start-up scan: set MC_JARS_DIR"),
    }
}
