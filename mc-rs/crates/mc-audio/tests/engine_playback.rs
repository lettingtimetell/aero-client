//! End-to-end playback through the real `kira` mixing graph, using
//! [`kira::backend::mock::MockBackend`] — no audio output device required (this sandbox has
//! none). This is the piece [`mc_audio::engine::compute_playback`]'s unit tests don't cover:
//! proof that a decoded clip is actually accepted by `kira::AudioManager::play`, rendered by
//! stepping the backend, and reaches [`kira::sound::PlaybackState::Stopped`] on its own once
//! exhausted — i.e. the pipeline wiring in [`mc_audio::AudioEngine::play`] works, not just the
//! gain/pitch/attenuation arithmetic feeding it.
//!
//! **Limitation, stated plainly:** `MockBackend` renders into a private buffer with no public
//! accessor in kira 0.12 — this crate cannot read back the actual output samples, so these tests
//! confirm the pipeline *runs* (accepts the sound, advances, completes) but not that the
//! resulting waveform is perceptually correct. That last step needs a human with speakers, which
//! this sandbox does not have either. `tests/vanilla_sounds.rs::a_real_resolved_ogg_decodes_to_sane_pcm`
//! is the closest this crate gets to a "does it sound right" check, and it inspects the decoded
//! PCM directly rather than kira's rendered output.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use kira::backend::mock::{MockBackend, MockBackendSettings};
use kira::sound::PlaybackState;
use kira::AudioManagerSettings;
use mc_assets::{ObjectStorePack, ResourceLocation, ResourceManager};
use mc_audio::{AttenuationType, AudioEngine, Listener, PlayRequest, PlayingSound, SoundRegistry};

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn assets_manager() -> Option<ResourceManager> {
    let assets = jars_dir()?.join("assets");
    if !assets.join("indexes").join("1.8.json").is_file() {
        return None;
    }
    let pack = ObjectStorePack::open(&assets, "1.8").ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

macro_rules! require {
    ($opt:expr, $what:literal) => {
        match $opt {
            Some(v) => v,
            None => {
                eprintln!(
                    "SKIP: {} not found; set MC_JARS_DIR to the jars/ directory",
                    $what
                );
                return;
            }
        }
    };
}

/// A fresh `AudioEngine<MockBackend>`, seeded deterministically.
///
/// `AudioEngine::new` builds its `AudioManager` from `AudioManagerSettings::default()`, which
/// for `MockBackend` means a nominal `sample_rate = 1` — fine for "does `play()` accept the
/// sound and report `Playing`" checks (this file's other two tests), but too coarse to usefully
/// step through a real clip to `Stopped`. `plays_a_real_click_to_completion_through_kira` below
/// instead builds its `AudioManager` directly with a realistic sample rate, bypassing this
/// helper, to exercise that path.
fn mock_engine(registry: SoundRegistry) -> AudioEngine<MockBackend> {
    AudioEngine::<MockBackend>::new(registry, 1)
}

#[test]
fn engine_runs_silent_without_panicking_when_backend_is_absent_in_spirit() {
    // Exercises the "no output device" path AudioEngine::new falls back to for the *real*
    // DefaultBackend when this sandbox has no audio hardware — MockBackend::setup is infallible,
    // so this instead confirms `play()` on an *empty* registry behaves the documented way
    // (an unregistered event is a hard error, not a silent no-op) without needing real assets.
    let mut engine = mock_engine(SoundRegistry::empty());
    assert!(!engine.is_silent(), "MockBackend::setup never fails");
    let rm = ResourceManager::new();
    let listener = Listener::origin();
    let request = PlayRequest::ambient(ResourceLocation::minecraft("nope.not.registered"));
    let err = engine.play(&request, &listener, &rm).unwrap_err();
    assert!(matches!(err, mc_audio::AudioError::UnknownEvent(_)));
}

#[test]
fn plays_a_real_click_to_completion_through_kira() {
    let rm = require!(assets_manager(), "jars/assets");
    let registry = SoundRegistry::load(&rm).expect("real sounds.json should parse");
    if !registry.contains(&ResourceLocation::minecraft("random.click")) {
        eprintln!("SKIP: random.click not registered in this asset set");
        return;
    }

    let mut manager = kira::AudioManager::<MockBackend>::new(AudioManagerSettings {
        backend_settings: MockBackendSettings {
            sample_rate: 44_100,
        },
        ..Default::default()
    })
    .expect("MockBackend::setup is infallible");

    let event = ResourceLocation::minecraft("random.click");
    let mut rng = mc_math::JavaRandom::new(1);
    let variant = registry.pick_variant(&event, &mut rng).expect("resolves");
    let bytes = rm.load(&variant.location).expect("loads");
    let decoded =
        mc_audio::ogg::decode_all(&bytes, &variant.location.to_string()).expect("decodes");
    assert!(decoded.num_frames() > 0);

    let data = kira::sound::static_sound::StaticSoundData {
        sample_rate: decoded.sample_rate,
        frames: std::sync::Arc::from(decoded.to_frames()),
        settings: kira::sound::static_sound::StaticSoundSettings::new(),
        slice: None,
    };

    let handle = manager.play(data).expect("kira accepts the decoded clip");
    manager.backend_mut().on_start_processing();

    let mut saw_playing = false;
    let mut stopped = false;
    for _ in 0..2000 {
        manager.backend_mut().process();
        match handle.state() {
            PlaybackState::Playing => saw_playing = true,
            PlaybackState::Stopped => {
                stopped = true;
                break;
            }
            _ => {}
        }
    }
    assert!(
        saw_playing,
        "clip should have been observed in the Playing state"
    );
    assert!(
        stopped,
        "a short clip should finish within 2000 mock-processed buffers"
    );
}

/// The Ogg bytes of one real streamed vanilla track, and its name.
fn streamed_track(rm: &ResourceManager) -> Option<(std::sync::Arc<[u8]>, String)> {
    let registry = SoundRegistry::load(rm).expect("real sounds.json should parse");
    let event = ResourceLocation::minecraft("records.13");
    if !registry.contains(&event) {
        eprintln!("SKIP: records.13 not registered in this asset set");
        return None;
    }
    let mut rng = mc_math::JavaRandom::new(1);
    let variant = registry.pick_variant(&event, &mut rng).expect("resolves");
    let bytes = rm.load(&variant.location).expect("loads");
    Some((std::sync::Arc::from(bytes), variant.location.to_string()))
}

/// kira seeks every stream to its start position before decoding it, whatever that position is.
/// If that seek lands on the Vorbis headers, the first decode fails, kira stops the sound and
/// nothing is reported, so music and records play as silence.
#[test]
fn a_streamed_track_keeps_playing_through_kira() {
    let rm = require!(assets_manager(), "jars/assets");
    let Some((bytes, name)) = streamed_track(&rm) else {
        return;
    };
    let decoder = mc_audio::ogg::VorbisDecoder::new(bytes, name).expect("opens");
    let mut manager = kira::AudioManager::<MockBackend>::new(AudioManagerSettings {
        backend_settings: MockBackendSettings {
            sample_rate: 44_100,
        },
        ..Default::default()
    })
    .expect("MockBackend::setup is infallible");
    let handle = manager
        .play(kira::sound::streaming::StreamingSoundData::from_decoder(
            decoder,
        ))
        .expect("kira accepts the stream");
    // The decoder runs on kira's own thread, so give it real time between mixer steps. A sound
    // publishes its position when a step starts, so each pass needs both calls.
    for _ in 0..100 {
        manager.backend_mut().on_start_processing();
        manager.backend_mut().process();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    manager.backend_mut().on_start_processing();
    assert_eq!(handle.state(), PlaybackState::Playing);
    assert!(handle.position() > 0.0, "the stream never advanced");
}

/// Going back to the start gives the same audio again, and a whole pass delivers exactly the
/// length the decoder reported to kira.
#[test]
fn seeking_a_stream_to_its_start_leaves_it_on_audio() {
    use kira::sound::streaming::Decoder;
    let rm = require!(assets_manager(), "jars/assets");
    let Some((bytes, name)) = streamed_track(&rm) else {
        return;
    };
    let mut decoder = mc_audio::ogg::VorbisDecoder::new(bytes, name).expect("opens");
    let total = decoder.num_frames();
    let first_chunks = |decoder: &mut mc_audio::ogg::VorbisDecoder| {
        let mut frames = Vec::new();
        while frames.len() < 8192 {
            frames.extend(decoder.decode().expect("decodes"));
        }
        frames.truncate(8192);
        frames
    };

    assert_eq!(decoder.seek(0).expect("seeks"), 0);
    let first = first_chunks(&mut decoder);
    assert!(first
        .iter()
        .any(|frame| frame.left != 0.0 || frame.right != 0.0));

    assert_eq!(decoder.seek(0).expect("seeks"), 0);
    assert!(first == first_chunks(&mut decoder), "the restart differs");

    assert_eq!(decoder.seek(0).expect("seeks"), 0);
    let mut delivered = 0;
    while delivered < total {
        let chunk = decoder.decode().expect("decodes").len();
        delivered += chunk;
    }
    assert_eq!(delivered, total);
    assert!(decoder.decode().expect("decodes").is_empty());
}

/// The vanilla object store, counting every `open` of a sound file, with one extra pack on top: a
/// `sounds.json` that registers two real vanilla clips as ordinary, non-streaming effects, the way
/// a resource pack swaps a hit or level-up sound for something long: the 18 s
/// `mob/enderdragon/end` as `test.long_effect` and the three-minute `records/13` as
/// `test.song_effect`.
fn long_effect_manager() -> Option<(ResourceManager, std::sync::Arc<AtomicUsize>)> {
    use std::io::{Cursor, Read};

    struct SoundsJson;
    impl mc_assets::ResourcePack for SoundsJson {
        fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
            self.contains(l).then(|| {
                let json = br#"{"test.long_effect": {"category": "master", "sounds": ["mob/enderdragon/end"]},
                    "test.song_effect": {"category": "master", "sounds": ["records/13"]}}"#;
                Box::new(Cursor::new(json.to_vec())) as Box<dyn Read>
            })
        }
        fn contains(&self, l: &ResourceLocation) -> bool {
            l.namespace == "minecraft" && l.path == "sounds.json"
        }
        fn namespaces(&self) -> Vec<String> {
            vec!["minecraft".to_owned()]
        }
        fn name(&self) -> &str {
            "long-effect"
        }
    }

    /// Forwards everything to the object store and counts the opens of `.ogg` files.
    struct CountingStore {
        inner: ObjectStorePack,
        ogg_opens: std::sync::Arc<AtomicUsize>,
    }
    impl mc_assets::ResourcePack for CountingStore {
        fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
            if l.path.ends_with(".ogg") {
                self.ogg_opens.fetch_add(1, Ordering::Relaxed);
            }
            self.inner.open(l)
        }
        fn contains(&self, l: &ResourceLocation) -> bool {
            self.inner.contains(l)
        }
        fn lists(&self, l: &ResourceLocation) -> Option<bool> {
            self.inner.lists(l)
        }
        fn file_path(&self, l: &ResourceLocation) -> Option<PathBuf> {
            self.inner.file_path(l)
        }
        fn namespaces(&self) -> Vec<String> {
            self.inner.namespaces()
        }
    }

    let assets = jars_dir()?.join("assets");
    let store = ObjectStorePack::open(&assets, "1.8").ok()?;
    let ogg_opens = std::sync::Arc::new(AtomicUsize::new(0));
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(CountingStore {
        inner: store,
        ogg_opens: std::sync::Arc::clone(&ogg_opens),
    }));
    rm.push_pack(Box::new(SoundsJson));
    Some((rm, ogg_opens))
}

/// A clip longer than 30 s is never decoded into the cache: every
/// play streams it from the compressed bytes kept in memory, so it is read once, and its first
/// play does not freeze the calling (game) thread either.
#[test]
fn a_clip_longer_than_thirty_seconds_always_streams_and_is_read_once() {
    let (rm, ogg_opens) = require!(long_effect_manager(), "jars/assets");
    let registry = SoundRegistry::load(&rm).expect("real sounds.json should parse");
    let mut engine = mock_engine(registry);
    let listener = Listener::origin();
    let request = PlayRequest::ambient(ResourceLocation::minecraft("test.song_effect"));
    for _ in 0..5 {
        let sound = engine
            .play(&request, &listener, &rm)
            .expect("plays")
            .expect("audible");
        assert!(matches!(sound, PlayingSound::Streaming(_)));
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(
        ogg_opens.load(Ordering::Relaxed),
        1,
        "read once, then kept compressed"
    );
    engine.stop_all_sounds();
    stop_streams_in_the_mock_mixer(&mut engine);
}

/// Lets the mock mixer apply the stops: kira ends a stream's decode thread only once the mixer
/// marks the stream stopped, and the mock backend has no audio thread to do it.
fn stop_streams_in_the_mock_mixer(engine: &mut AudioEngine<MockBackend>) {
    if let Some(backend) = engine.backend_mut() {
        backend.on_start_processing();
        backend.process();
    }
}

/// The first play of a long effect must not decode the whole clip
/// on the calling thread, which is the game thread in the client (a 3-minute clip froze it for
/// about a quarter of a second). It streams from the compressed bytes at once; the full decode
/// runs on a worker, and once it lands later plays come from the decoded cache.
#[test]
fn a_long_effect_streams_at_first_play_and_is_cached_for_later_plays() {
    let (rm, _) = require!(long_effect_manager(), "jars/assets");
    let registry = SoundRegistry::load(&rm).expect("real sounds.json should parse");
    let mut engine = mock_engine(registry);
    let listener = Listener::origin();
    let request = PlayRequest::ambient(ResourceLocation::minecraft("test.long_effect"));

    let started = std::time::Instant::now();
    let first = engine
        .play(&request, &listener, &rm)
        .expect("plays")
        .expect("audible at full volume");
    eprintln!(
        "first play returned after {:.1} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    assert!(
        matches!(first, PlayingSound::Streaming(_)),
        "the first play of an uncached effect decoded the whole clip on the calling thread"
    );
    assert_eq!(first.state(), PlaybackState::Playing);

    // The worker decodes the clip; a later play is served from the cache as a static sound.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        std::thread::sleep(std::time::Duration::from_millis(250));
        let again = engine
            .play(&request, &listener, &rm)
            .expect("plays")
            .expect("audible");
        if matches!(again, PlayingSound::Static(_)) {
            eprintln!(
                "served from the decoded cache {:.1} s after the first play",
                started.elapsed().as_secs_f64()
            );
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the background decode never reached the cache"
        );
    }
    engine.stop_all_sounds();
    stop_streams_in_the_mock_mixer(&mut engine);
}

#[test]
fn audio_engine_play_returns_a_handle_that_reaches_kiras_mixer() {
    let rm = require!(assets_manager(), "jars/assets");
    let registry = SoundRegistry::load(&rm).expect("real sounds.json should parse");
    if !registry.contains(&ResourceLocation::minecraft("random.click")) {
        eprintln!("SKIP: random.click not registered in this asset set");
        return;
    }

    let mut engine = mock_engine(registry);
    let listener = Listener::origin();
    let request = PlayRequest::positional(
        ResourceLocation::minecraft("random.click"),
        listener.position,
    );

    let played = engine
        .play(&request, &listener, &rm)
        .expect("compute_playback resolves and kira accepts it")
        .expect("gain is nonzero at zero distance, so this must not be silently dropped");

    assert_eq!(played.state(), PlaybackState::Playing);
    assert_eq!(
        AttenuationType::Linear.type_int(),
        2,
        "sanity: positional sounds are LINEAR"
    );
}
