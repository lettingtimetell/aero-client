//! Resource packs are other people's files: players share them and any server can offer one.
//! Every decoder a pack reaches is driven here with seeded random damage to valid files, and the
//! worst cases found by review are pinned with their minimized inputs. Nothing in a pack may
//! panic, overflow the stack, allocate without bound or take super-linear time.
//!
//! The fuzz rounds are bounded (a few seconds for the whole file in a debug build). For a longer
//! run by hand, multiply them and pick another seed:
//!
//! ```text
//! PACK_FUZZ_ROUNDS=20 PACK_FUZZ_SEED=12345 cargo test -p mc-assets --test hostile_pack_decoders -- --nocapture
//! ```
//!
//! This binary counts the bytes live on each thread (a counting global allocator; the library is
//! unchanged) and refuses any single allocation over 1 GiB, so a regression fails here instead
//! of committing gigabytes on the machine running the suite.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::io::{Cursor, Read, Write as _};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use mc_assets::ctm::CtmRules;
use mc_assets::custom_sky;
use mc_assets::model::blockmodel::ResolvedModel;
use mc_assets::model::blockstate::BlockStateJson;
use mc_assets::{
    decode_texture_png, format_template, parse_language_sections, parse_texture_metadata, Language,
    PackMetadata, ResourceLocation, ResourceManager, ResourcePack, ZipResourcePack,
};

// ---------------------------------------------------------------------------------------------
// Measuring: a per-thread counting allocator with a refusal above 1 GiB.

struct Counting;

const REFUSED_ALLOCATION: usize = 1 << 30;

thread_local! {
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

/// Every thread together, for work that fans out to worker threads (the model bake).
static PROCESS_LIVE: AtomicI64 = AtomicI64::new(0);
static PROCESS_PEAK: AtomicI64 = AtomicI64::new(0);

fn grew(bytes: usize) {
    let _ = LIVE.try_with(|live| {
        live.set(live.get() + bytes as i64);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(live.get())));
    });
    let now = PROCESS_LIVE.fetch_add(bytes as i64, Ordering::Relaxed) + bytes as i64;
    PROCESS_PEAK.fetch_max(now, Ordering::Relaxed);
}

fn shrank(bytes: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - bytes as i64));
    PROCESS_LIVE.fetch_sub(bytes as i64, Ordering::Relaxed);
}

// SAFETY: forwards every call to the system allocator unchanged, except that a request over
// `REFUSED_ALLOCATION` gets a null pointer, which every caller must already handle (the standard
// library turns it into an allocation-failure abort).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(layout.size());
        // SAFETY: the caller's contract is passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(layout.size());
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(new_size);
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const MIB: i64 = 1 << 20;

/// The most bytes live on this thread at once while `run` ran, above what was live before it.
fn peak_during<T>(run: impl FnOnce() -> T) -> (T, i64) {
    let before = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(before));
    let out = run();
    (out, PEAK.with(Cell::get) - before)
}

/// Runs `work` on its own thread with `stack` bytes of stack (the Windows main thread has 1 MiB),
/// and fails the test if it has not returned after `limit`.
fn on_thread<T: Send + 'static>(
    name: &str,
    stack: usize,
    limit: Duration,
    work: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(stack)
        .spawn(move || {
            let _ = tx.send(work());
        })
        .expect("spawn the worker");
    match rx.recv_timeout(limit) {
        Ok(value) => value,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{name}: still running after {limit:?}")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{name}: the worker panicked"),
    }
}

fn show(input: &[u8]) -> String {
    let shown: String = input
        .escape_ascii()
        .to_string()
        .chars()
        .take(1500)
        .collect();
    format!("{} bytes: \"{shown}\"", input.len())
}

// ---------------------------------------------------------------------------------------------
// The mutator: a fixed-seed splitmix64 stream, so every run of the suite sees the same inputs.

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

const INTERESTING_BYTES: &[u8] = &[
    0, 0xFF, 0x7F, 0x80, b'"', b'\\', b'{', b'}', b'[', b']', b'%', b'-', b'0', b'9', b'#', b'\n',
    b'\r', b'=', b':', b',', b' ', b'.', b'$', b'/',
];

const BOUNDARY_NUMBERS: &[&[u8]] = &[
    b"0",
    b"-1",
    b"1",
    b"15",
    b"16",
    b"47",
    b"255",
    b"256",
    b"4096",
    b"32767",
    b"32768",
    b"65535",
    b"65536",
    b"2147483647",
    b"2147483648",
    b"-2147483648",
    b"-2147483649",
    b"4294967296",
    b"9223372036854775807",
    b"18446744073709551616",
    b"1e309",
    b"-1e309",
    b"1e-400",
    b"0.5",
    b"NaN",
    b"Infinity",
    b"-0",
    b"00000000000000000000001",
];

fn mutate(rng: &mut Rng, data: &mut Vec<u8>, tokens: &[&[u8]], max_len: usize) {
    for _ in 0..=rng.below(4) {
        let len = data.len();
        let token = tokens[rng.below(tokens.len())];
        match rng.below(11) {
            0 if len > 0 => {
                let at = rng.below(len);
                data[at] ^= 1 << rng.below(8);
            }
            1 if len > 0 => {
                let at = rng.below(len);
                data[at] = INTERESTING_BYTES[rng.below(INTERESTING_BYTES.len())];
            }
            2 => {
                let at = rng.below(len + 1);
                data.splice(at..at, token.iter().copied());
            }
            3 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(16)).min(len);
                data.splice(at..end, token.iter().copied());
            }
            4 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(32)).min(len);
                data.drain(at..end);
            }
            5 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(64)).min(len);
                let piece = data[at..end].to_vec();
                let to = rng.below(data.len() + 1);
                data.splice(to..to, piece);
            }
            6 => {
                // One token many times over: depth, width and long runs.
                let times = 1 + rng.below(300);
                let at = rng.below(len + 1);
                let run: Vec<u8> = token
                    .iter()
                    .copied()
                    .cycle()
                    .take(token.len() * times)
                    .collect();
                data.splice(at..at, run);
            }
            7 if len > 1 => data.truncate(rng.below(len)),
            8 => {
                let at = rng.below(len + 1);
                let random: Vec<u8> = (0..1 + rng.below(8)).map(|_| rng.next() as u8).collect();
                data.splice(at..at, random);
            }
            9 if len > 0 => {
                // A number swapped for a boundary value.
                let from = rng.below(len);
                if let Some(start) = (from..len).find(|&i| data[i].is_ascii_digit()) {
                    let end = (start..len)
                        .find(|&i| !data[i].is_ascii_digit())
                        .unwrap_or(len);
                    let number = BOUNDARY_NUMBERS[rng.below(BOUNDARY_NUMBERS.len())];
                    data.splice(start..end, number.iter().copied());
                }
            }
            _ => {
                let at = rng.below(len + 1);
                let number = BOUNDARY_NUMBERS[rng.below(BOUNDARY_NUMBERS.len())];
                data.splice(at..at, number.iter().copied());
            }
        }
    }
    data.truncate(max_len);
}

/// `PACK_FUZZ_ROUNDS` multiplies every target's rounds (default 1).
fn round_multiplier() -> usize {
    std::env::var("PACK_FUZZ_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
        .max(1)
}

/// `PACK_FUZZ_SEED` replaces the fixed base seed.
fn base_seed() -> u64 {
    std::env::var("PACK_FUZZ_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0x005E_ED0F_FAC3_D00D)
}

/// No single damaged input may take longer than this, even in a debug build on a busy machine.
const PER_INPUT_TIME: Duration = Duration::from_secs(5);
/// Nor hold more than this at once.
const PER_INPUT_PEAK: i64 = 256 * MIB;

/// Runs `rounds` damaged copies of `seeds` through `run`, on a 1 MiB stack, and fails on a
/// panic, a slow input or a large allocation, printing the input.
fn fuzz(
    target: &'static str,
    seeds: Vec<Vec<u8>>,
    tokens: &'static [&'static [u8]],
    rounds: usize,
    max_len: usize,
    run: impl Fn(&[u8]) + Send + 'static,
) {
    let seed = base_seed()
        ^ target
            .bytes()
            .fold(0u64, |h, b| h.rotate_left(5) ^ u64::from(b));
    let rounds = rounds * round_multiplier();
    let limit = PER_INPUT_TIME * 4 + Duration::from_millis(rounds as u64 * 50);
    let current = Arc::new(Mutex::new(Vec::new()));
    let watched = Arc::clone(&current);
    let (done, finished) = mpsc::channel();
    let worker = move || {
        let mut rng = Rng(seed);
        let started = Instant::now();
        for round in 0..rounds {
            let mut input = seeds[rng.below(seeds.len())].clone();
            if rng.below(8) == 0 {
                // Crossover: the front of one seed, the back of another.
                let other = &seeds[rng.below(seeds.len())];
                let cut = rng.below(input.len() + 1);
                let from = rng.below(other.len() + 1);
                input.truncate(cut);
                input.extend_from_slice(&other[from..]);
            }
            mutate(&mut rng, &mut input, tokens, max_len);
            if let Ok(mut current) = current.lock() {
                current.clone_from(&input);
            }
            let began = Instant::now();
            let (outcome, peak) =
                peak_during(|| std::panic::catch_unwind(AssertUnwindSafe(|| run(&input))));
            let took = began.elapsed();
            assert!(
                outcome.is_ok(),
                "{target}: round {round} (seed {seed:#x}) panicked on {}",
                show(&input)
            );
            assert!(
                took < PER_INPUT_TIME,
                "{target}: round {round} (seed {seed:#x}) took {took:?} on {}",
                show(&input)
            );
            assert!(
                peak < PER_INPUT_PEAK,
                "{target}: round {round} (seed {seed:#x}) held {peak} bytes on {}",
                show(&input)
            );
        }
        eprintln!(
            "{target}: {rounds} rounds, seed {seed:#x}, {:?}",
            started.elapsed()
        );
        let _ = done.send(());
    };
    std::thread::Builder::new()
        .name(target.to_owned())
        .stack_size(1 << 20)
        .spawn(worker)
        .expect("spawn the fuzz worker");
    match finished.recv_timeout(limit) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let input = watched.lock().map(|i| show(&i)).unwrap_or_default();
            panic!("{target}: an input never returned after {limit:?}: {input}")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{target}: the fuzz worker failed"),
    }
}

// ---------------------------------------------------------------------------------------------
// An in-memory pack, so a round costs no disk access.

#[derive(Clone, Default)]
struct MemPack {
    files: BTreeMap<(String, String), Vec<u8>>,
}

impl MemPack {
    fn with(mut self, path: &str, bytes: impl Into<Vec<u8>>) -> Self {
        self.files
            .insert(("minecraft".to_owned(), path.to_owned()), bytes.into());
        self
    }

    fn into_manager(self) -> ResourceManager {
        let mut manager = ResourceManager::new();
        manager.push_pack(Box::new(self));
        manager
    }

    fn key(l: &ResourceLocation) -> (String, String) {
        (l.namespace.clone(), l.path.clone())
    }
}

impl ResourcePack for MemPack {
    fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
        self.files
            .get(&Self::key(l))
            .map(|bytes| Box::new(Cursor::new(bytes.clone())) as Box<dyn Read>)
    }

    fn contains(&self, l: &ResourceLocation) -> bool {
        self.files.contains_key(&Self::key(l))
    }

    fn lists(&self, l: &ResourceLocation) -> Option<bool> {
        Some(self.contains(l))
    }

    fn name(&self) -> &str {
        "in-memory test pack"
    }

    fn namespaces(&self) -> Vec<String> {
        let mut namespaces: Vec<String> = self.files.keys().map(|(ns, _)| ns.clone()).collect();
        namespaces.dedup();
        namespaces
    }

    fn list_files(&self, namespace: &str, dir_prefix: &str) -> Vec<String> {
        self.files
            .keys()
            .filter(|(ns, path)| ns == namespace && path.starts_with(dir_prefix))
            .map(|(_, path)| path.clone())
            .collect()
    }
}

fn png(width: u32, height: u32) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image::RgbaImage::from_fn(width, height, |x, y| {
        image::Rgba([(x * 16) as u8, (y * 4) as u8, ((x ^ y) * 8) as u8, 255])
    })
    .write_to(&mut out, image::ImageFormat::Png)
    .expect("encode a test PNG");
    out.into_inner()
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

// ---------------------------------------------------------------------------------------------
// Seeds and token dictionaries.

const SKY_SEEDS: &[&str] = &[
    "source=./sky1.png\nstartFadeIn=17:30\nendFadeIn=18:30\nstartFadeOut=5:00\nendFadeOut=6:00\n\
     blend=add\nrotate=true\nspeed=1.0\naxis=0.0 0.0 1.0\nweather=clear rain\n\
     biomes=plains desert minecraft:forest 2\nheights=0-64 (-5)-10 100\ntransition=1.5\n",
    "# a comment\r\n! another\r\nsource = ~/sky/world0/stars\r\nblend: screen\r\nrotate false\r\n\
     speed=0.25\r\naxis=1 \\\n  0 0\r\nweather=thunder\r\nkey\\u0041=value\\tx\r\n",
    "source=minecraft:textures/environment/sun.png\nblend=replace\nstartFadeIn=0:00\n\
     endFadeIn=0:01\nendFadeOut=23:59\nheights=(-64)-(-1)\n",
];

const PROPERTIES_TOKENS: &[&[u8]] = &[
    b"=",
    b":",
    b"\\\n",
    b"\\",
    b"\\\\",
    b"\r",
    b"\n",
    b"\r\n",
    b"#",
    b"!",
    b" ",
    b"\t",
    b"\\u00",
    b"\\uFFFF",
    b"source",
    b"startFadeIn",
    b"endFadeIn",
    b"startFadeOut",
    b"endFadeOut",
    b"0-4095",
    b"(-5)-10",
    b"(",
    b")",
    b",",
    b"-",
    b"999999999999",
    b"23:59",
    b"24:00",
    b"matchTiles",
    b"matchBlocks",
    b"tiles",
    b"method",
    b"ctm",
    b"glass",
    b"metadata",
    b"faces",
    b"biomes",
    b"heights",
    b"weather",
    b"axis",
    b"speed",
    b"blend",
    b"rotate",
    b"transition",
    b"<skip>",
    b"<default>",
    b"../",
    b"minecraft:",
    b"~/",
    b"./",
    b"optifine/",
    b"20:1,2,4-6",
    b"connect",
    b"innerSeams",
    b"all",
    b"sides",
    b"\xC3\xA9",
    b"\xFF",
    b"\x00",
];

const CTM_SEEDS: &[&str] = &[
    "method=ctm\ntiles=0-46\nmatchTiles=glass\nmetadata=0 1,2 4-6\nfaces=sides top\n\
     connect=block\ninnerSeams=false\n",
    "method=glass\nmatchBlocks=20 glass:1,2 minecraft:glass\ntiles=0-15 16 17 18-46 <skip> \
     <default>\nconnect=tile\n",
    "method = ctm\r\nmatchTiles = blocks/glass glass.png\r\ntiles = ./0 ./1 2-46\r\n\
     faces = all\r\n",
];

const LANG_SEEDS: &[&str] = &[
    "gui.done=Done\nbook.byAuthor=by %1$s\nbook.pageIndicator=Page %1$s of %2$s\n\
     key.width=%-5s|%5s|%.2s\npercent=100%% done %n\nweird=%d %5.2f %2$d\n# comment\n\
     reuse=%s %<s %S\n",
    "a=%s%s%s%s\rb=%3$s%2$s%1$s\r\nc=x=y=z\n=empty key\nnovalue\n",
];

const LANG_TOKENS: &[&[u8]] = &[
    b"%",
    b"%s",
    b"%1$s",
    b"%2$s",
    b"%99$s",
    b"%2147483648$s",
    b"%<s",
    b"%-",
    b"%0",
    b"%.",
    b"%5",
    b"%999999s",
    b"%-99999s",
    b"%.99999s",
    b"$",
    b"=",
    b"\n",
    b"\r",
    b"#",
    b"%%",
    b"%n",
    b"%d",
    b"%5.2f",
    b"%S",
];

const JSON_TOKENS: &[&[u8]] = &[
    b"{",
    b"}",
    b"[",
    b"]",
    b"\"",
    b":",
    b",",
    b"null",
    b"true",
    b"false",
    b"1e309",
    b"-1",
    b"2147483648",
    b"-2147483649",
    b"0.5",
    b"\"#a\"",
    b"\"#b\"",
    b"\"parent\"",
    b"\"textures\"",
    b"\"elements\"",
    b"\"from\"",
    b"\"to\"",
    b"\"faces\"",
    b"\"uv\"",
    b"\"rotation\"",
    b"\"angle\"",
    b"\"axis\"",
    b"\"origin\"",
    b"\"rescale\"",
    b"\"cullface\"",
    b"\"tintindex\"",
    b"\"texture\"",
    b"\"display\"",
    b"\"variants\"",
    b"\"model\"",
    b"\"x\"",
    b"\"y\"",
    b"\"uvlock\"",
    b"\"weight\"",
    b"\"animation\"",
    b"\"frames\"",
    b"\"frametime\"",
    b"\"index\"",
    b"\"time\"",
    b"\"width\"",
    b"\"height\"",
    b"\"interpolate\"",
    b"\"pack\"",
    b"\"pack_format\"",
    b"\"description\"",
    b"\"language\"",
    b"\"region\"",
    b"\"name\"",
    b"\"bidirectional\"",
    b"\"\\ud800\"",
    b"\"\\u0000\"",
    b"'",
    b"//",
    b"/*",
    b"*/",
    b"#",
    b"\"block/fuzz_root\"",
    b"\"block/fuzz_mid\"",
    b"\"block/fuzz_leaf\"",
    b"\"builtin/generated\"",
    b"\"extra\"",
    b"\"text\"",
    b"\"translate\"",
    b"\"with\"",
    b"\"mipmaps\"",
    b"\"blur\"",
    b"\"clamp\"",
];

const MODEL_LEAF: &str =
    r##"{"parent":"block/fuzz_mid","textures":{"all":"blocks/stone","top":"#all"}}"##;
const MODEL_MID: &str = r##"{"parent":"block/fuzz_root","ambientocclusion":false,
"textures":{"particle":"#all","side":"#all"},
"elements":[{"from":[0,0,0],"to":[16,16,16],
"rotation":{"origin":[8,8,8],"axis":"y","angle":22.5,"rescale":true},"shade":true,
"faces":{"down":{"uv":[0,0,16,16],"texture":"#all","cullface":"down","rotation":90,"tintindex":0},
"up":{"texture":"#top"},"north":{"texture":"#side","uv":[16,16,0,0]}}}]}"##;
const MODEL_ROOT: &str = r##"{"display":{"thirdperson":{"rotation":[10,-45,170],
"translation":[0,1.5,-2.75],"scale":[0.375,0.375,0.375]}},"textures":{"x":"#side"},
"elements":[{"from":[0,0,0],"to":[16,8,16],"faces":{"up":{"texture":"#x"},"down":{"texture":"#side"}}}]}"##;

const BLOCKSTATE_SEEDS: &[&str] = &[
    r#"{"variants":{"normal":{"model":"stone"},"facing=north":[{"model":"stone","y":90,"weight":2},{"model":"stone","x":180,"uvlock":true}]}}"#,
    r#"{"variants":{"axis=y":{"model":"log","x":0,"y":0,"weight":0},"axis=x":[{"model":"log","x":90,"y":90,"weight":-2}]}}"#,
];

const PACK_MCMETA_SEEDS: &[&str] = &[
    r#"{"pack":{"pack_format":1,"description":"A pack \u00a7aGreen"},"language":{"xx_XX":{"region":"Somewhere","name":"Testish","bidirectional":false},"yy_YY":{"region":1,"name":true,"bidirectional":"true"}}}"#,
    r#"{"pack":{"pack_format":1,"description":{"text":"x","color":"red","extra":[{"translate":"a.b","with":["c",{"text":"d","bold":true}]}]}}}"#,
];

const TEXTURE_MCMETA_SEEDS: &[&str] = &[
    r#"{"animation":{"frametime":2,"interpolate":true,"frames":[0,{"index":1,"time":3},2,3,null,[1]],"width":16,"height":16},"texture":{"blur":true,"clamp":false,"mipmaps":[1,2]}}"#,
    r#"{"animation":{}}"#,
    r#"{"animation":{"frames":[3,2,1,0,1,2]},"texture":{"blur":"true","mipmaps":[0,1.5,"2"]}}"#,
];

fn strings(seeds: &[&str]) -> Vec<Vec<u8>> {
    seeds.iter().map(|seed| seed.as_bytes().to_vec()).collect()
}

// ---------------------------------------------------------------------------------------------
// Fuzz targets.

#[test]
fn custom_sky_layers_survive_damage() {
    fuzz(
        "custom sky properties",
        strings(SKY_SEEDS),
        PROPERTIES_TOKENS,
        4000,
        64 << 10,
        |bytes| {
            let text = latin1(bytes);
            let file = ResourceLocation::minecraft("mcpatcher/sky/world0/sky1.properties");
            let _ = custom_sky::parse_properties(&text);
            if let Ok(layer) = custom_sky::parse_layer(0, 1, &file, &text) {
                for time in [0, 6000, 12000, 18000, 23999, -1, i32::MAX] {
                    if let Some(fade) = &layer.fade {
                        let _ = fade.brightness(time);
                    }
                }
            }
        },
    );
}

fn ctm_pack(properties: &[u8]) -> MemPack {
    let tile = png(16, 16);
    let mut pack = MemPack::default().with("mcpatcher/ctm/glass/glass.properties", properties);
    for index in 0..47 {
        pack = pack.with(&format!("mcpatcher/ctm/glass/{index}.png"), tile.clone());
    }
    pack
}

fn glass_textures() -> HashSet<ResourceLocation> {
    HashSet::from([ResourceLocation::minecraft("textures/blocks/glass")])
}

#[test]
fn connected_texture_rules_survive_damage() {
    let base = ctm_pack(b"");
    fuzz(
        "connected-texture properties",
        strings(CTM_SEEDS),
        PROPERTIES_TOKENS,
        1500,
        64 << 10,
        move |bytes| {
            let mut pack = base.clone();
            pack.files.insert(
                (
                    "minecraft".to_owned(),
                    "mcpatcher/ctm/glass/glass.properties".to_owned(),
                ),
                bytes.to_vec(),
            );
            let rules = CtmRules::load(&pack.into_manager(), &glass_textures());
            let _ = rules.atlas_textures();
            for rule in rules.rules() {
                let _ = (rule.source(), rule.connect(), rule.match_tiles());
                let _ = rule.match_block_ids().count();
                for index in 0..50 {
                    let _ = rule.tile(index);
                }
            }
        },
    );
}

#[test]
fn language_files_and_templates_survive_damage() {
    fuzz(
        "language files",
        strings(LANG_SEEDS),
        LANG_TOKENS,
        4000,
        64 << 10,
        |bytes| {
            let mut lang = Language::new("en_US");
            lang.merge_lang_bytes(bytes);
            lang.recompute_unicode();
            let keys: Vec<String> = lang.iter().take(200).map(|(k, _)| k.to_owned()).collect();
            for key in keys {
                let _ = lang.format(&key, &["A", "B\u{1F600}", "C"]);
                let _ = lang.tr(&key);
            }
            let template = String::from_utf8_lossy(bytes);
            let _ = format_template(&template, &["x", "yy", ""]);
        },
    );
}

#[test]
fn pack_and_language_metadata_survive_damage() {
    fuzz(
        "pack.mcmeta",
        strings(PACK_MCMETA_SEEDS),
        JSON_TOKENS,
        4000,
        64 << 10,
        |bytes| {
            if let Ok(meta) = PackMetadata::parse(bytes) {
                let _ = meta.is_supported();
            }
            let _ = parse_language_sections(bytes);
        },
    );
}

#[test]
fn texture_metadata_and_animations_survive_damage() {
    let sheet = png(16, 64);
    fuzz(
        "texture .mcmeta",
        strings(TEXTURE_MCMETA_SEEDS),
        JSON_TOKENS,
        1500,
        64 << 10,
        move |bytes| {
            let texture = ResourceLocation::minecraft("textures/blocks/fuzz");
            let _ = parse_texture_metadata(&texture, Some(bytes));
            let rm = MemPack::default()
                .with("textures/blocks/fuzz.png", sheet.clone())
                .with("textures/blocks/fuzz.png.mcmeta", bytes)
                .into_manager();
            if let Ok(sprite) = mc_assets::sprite::load_sprite(&rm, &texture) {
                let _ = sprite.generate_mipmaps(4);
            }
        },
    );
}

#[test]
fn model_parent_chains_survive_damage() {
    let seeds = vec![
        MODEL_LEAF.as_bytes().to_vec(),
        MODEL_MID.as_bytes().to_vec(),
        MODEL_ROOT.as_bytes().to_vec(),
    ];
    fuzz("model JSON", seeds, JSON_TOKENS, 2000, 64 << 10, |bytes| {
        // The damaged file takes each place in the chain in turn.
        for slot in ["fuzz_leaf", "fuzz_mid", "fuzz_root"] {
            let mut pack = MemPack::default()
                .with("models/block/fuzz_leaf.json", MODEL_LEAF)
                .with("models/block/fuzz_mid.json", MODEL_MID)
                .with("models/block/fuzz_root.json", MODEL_ROOT);
            pack = pack.with(&format!("models/block/{slot}.json"), bytes);
            let rm = pack.into_manager();
            let leaf = ResourceLocation::minecraft("block/fuzz_leaf");
            if let Ok(model) = ResolvedModel::resolve(&rm, &leaf) {
                for key in model.textures.keys() {
                    let _ = model.resolve_texture(key);
                    let _ = model.resolve_texture(&format!("#{key}"));
                }
            }
        }
    });
}

#[test]
fn blockstate_json_survives_damage() {
    fuzz(
        "blockstate JSON",
        strings(BLOCKSTATE_SEEDS),
        JSON_TOKENS,
        4000,
        64 << 10,
        |bytes| {
            if let Ok(state) = serde_json::from_slice::<BlockStateJson>(bytes) {
                let _ = state.variants.values().map(Vec::len).sum::<usize>();
            }
        },
    );
}

fn small_zip() -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let deflated = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file("pack.mcmeta", deflated).expect("entry");
    zip.write_all(PACK_MCMETA_SEEDS[0].as_bytes())
        .expect("write");
    zip.start_file("assets/minecraft/lang/en_US.lang", stored)
        .expect("entry");
    zip.write_all(LANG_SEEDS[0].as_bytes()).expect("write");
    zip.start_file("assets/minecraft/textures/blocks/stone.png", deflated)
        .expect("entry");
    zip.write_all(&png(16, 16)).expect("write");
    zip.finish().expect("finish").into_inner()
}

#[test]
fn zip_packs_survive_damage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("damaged.zip");
    fuzz(
        "zip pack",
        vec![small_zip()],
        &[
            b"PK\x03\x04",
            b"PK\x01\x02",
            b"PK\x05\x06",
            b"\xFF\xFF\xFF\xFF",
            b"\x00\x00",
        ],
        600,
        64 << 10,
        move |bytes| {
            std::fs::write(&path, bytes).expect("write the damaged zip");
            let Ok(pack) = ZipResourcePack::open(&path) else {
                return;
            };
            let _ = pack.metadata();
            let _ = pack.namespaces();
            for namespace in pack.namespaces() {
                for file in pack.list_files(&namespace, "") {
                    let location = ResourceLocation::new(&namespace, file);
                    if let Some(mut reader) = pack.open(&location) {
                        let mut sink = Vec::new();
                        let _ = reader.read_to_end(&mut sink);
                    }
                }
            }
        },
    );
}

#[test]
fn png_textures_survive_damage() {
    fuzz(
        "PNG texture",
        vec![png(16, 16), png(16, 64), png(1, 1)],
        &[
            b"IHDR",
            b"IDAT",
            b"IEND",
            b"PLTE",
            b"tRNS",
            b"zTXt",
            b"\xFF\xFF\xFF\xFF",
        ],
        3000,
        64 << 10,
        |bytes| {
            let _ = decode_texture_png(bytes);
            let _ = image::load_from_memory_with_format(bytes, image::ImageFormat::Png);
        },
    );
}

// ---------------------------------------------------------------------------------------------
// Regressions: the worst inputs found by review, minimized.

/// Measures `run`, which must finish within `limit` on a 1 MiB stack.
fn bounded<T: Send + 'static>(
    name: &str,
    limit: Duration,
    run: impl FnOnce() -> T + Send + 'static,
) -> (T, Duration, i64) {
    on_thread(name, 1 << 20, limit * 4, move || {
        let began = Instant::now();
        let (out, peak) = peak_during(run);
        (out, began.elapsed(), peak)
    })
}

/// A continued line used to re-scan every backslash of the
/// joined line at each join: one line of backslashes followed by many one-backslash lines was
/// quadratic in the file (Java's `Properties.load` reads it in one pass).
#[test]
fn continued_sky_property_lines_parse_in_linear_time() {
    let n = 60_000;
    let mut text = String::from("source=");
    text.extend(std::iter::repeat_n('\\', 2 * n + 1));
    text.push('\n');
    for _ in 0..n {
        text.push_str("\\\n");
    }
    text.push_str("rotate=false\n");
    let limit = Duration::from_secs(3);
    let (props, took, _) = bounded("sky continuation", limit, move || {
        custom_sky::parse_properties(&text)
    });
    eprintln!("{n} continuation lines: {took:?}");
    assert!(took < limit, "took {took:?}");
    // Each join drops the continuing backslash; the pairs left unescape to one backslash each,
    // and the last line is swallowed by the continuation, as `Properties.load` does.
    let expected = format!("{}rotate=false", "\\".repeat(n));
    assert_eq!(props.get("source"), Some(&expected));
    assert_eq!(props.get("rotate"), None);
}

/// Each `biomes=` token used to squash every registered biome's name
/// again to compare (and log one warning per unknown name): a million tokens took seconds of
/// allocations. The names are now squashed once per layer, and unknown names are counted.
#[test]
fn a_long_custom_sky_biome_list_parses_in_linear_time() {
    let mut text = String::from("biomes=plains ");
    for _ in 0..1_000_000 {
        text.push_str("x ");
    }
    text.push_str("desert 2\n");
    let file = ResourceLocation::minecraft("mcpatcher/sky/world0/sky1.properties");
    let limit = Duration::from_secs(5);
    let (biomes, took, _) = bounded("sky biomes", limit, move || {
        custom_sky::parse_layer(0, 1, &file, &text).map(|layer| layer.biomes)
    });
    eprintln!("a million unknown biome names: {took:?}");
    assert!(took < limit, "took {took:?}");
    // plains (1), desert (2) and the id 2 again: two biomes, in order.
    assert_eq!(biomes, Ok(Some(vec![1, 2])));
}

/// Each key used to be looked for among all the keys before it
/// (`retain` over the list): a rule file of many distinct keys was quadratic in its length.
#[test]
fn connected_texture_rules_with_many_keys_parse_in_linear_time() {
    let mut text = String::from("method=ctm\nmatchTiles=glass\ntiles=0-46\n");
    for index in 0..100_000 {
        text.push_str(&format!("k{index}=v\n"));
    }
    let pack = ctm_pack(text.as_bytes());
    let limit = Duration::from_secs(5);
    let (count, took, _) = bounded("ctm many keys", limit, move || {
        CtmRules::load(&pack.into_manager(), &glass_textures())
            .rules()
            .len()
    });
    eprintln!("100000 keys: {took:?}");
    assert!(took < limit, "took {took:?}");
    assert_eq!(count, 1, "the rule itself is still read");
}

/// `tiles=0-4095 0-4095 ...` used to expand every range in full before
/// the list was cut to the 47 tiles a rule uses: 300 such tokens (2 KB) held over a million tile
/// names.
#[test]
fn connected_texture_tile_ranges_stop_at_the_tiles_a_rule_uses() {
    let mut text = String::from("method=ctm\nmatchTiles=glass\ntiles=");
    for _ in 0..300 {
        text.push_str("0-4095 ");
    }
    text.push('\n');
    let pack = ctm_pack(text.as_bytes());
    let (tiles, took, peak) = bounded("ctm tile ranges", Duration::from_secs(10), move || {
        let rules = CtmRules::load(&pack.into_manager(), &glass_textures());
        rules.rules().first().map(|rule| {
            (0..50)
                .filter_map(|i| rule.tile(i).cloned())
                .collect::<Vec<_>>()
        })
    });
    eprintln!("tile ranges: {took:?}, peak {} KiB", peak >> 10);
    let tiles = tiles.expect("the rule loads");
    assert_eq!(tiles.len(), 47);
    assert_eq!(
        tiles[46],
        ResourceLocation::minecraft("mcpatcher/ctm/glass/46")
    );
    assert!(peak < 8 * MIB, "held {peak} bytes");
}

/// `matchTiles` lists were unbounded and searched linearly inside
/// loops over rules (the shadowing check) and over every quad of every block state (the table
/// build): six rules listing the same 20,000 tiles took minutes to load. A rule listing more
/// than 256 tiles is refused (real rules list a handful), at most 512 rules are kept, and the
/// lists are looked up through sets.
#[test]
fn connected_texture_rules_with_long_match_lists_load_in_bounded_time() {
    let names: Vec<String> = (0..20_000).map(|i| format!("t{i}")).collect();
    let mut pack = ctm_pack(b"method=ctm\nmatchTiles=glass\ntiles=0-46\n");
    for rule in 0..6 {
        pack = pack.with(
            &format!("mcpatcher/ctm/glass/r{rule}.properties"),
            format!("method=ctm\ntiles=0-46\nmatchTiles={}\n", names.join(" ")),
        );
    }
    let limit = Duration::from_secs(5);
    let (count, took, _) = bounded("ctm match lists", limit, move || {
        CtmRules::load(&pack.into_manager(), &glass_textures())
            .rules()
            .len()
    });
    eprintln!("six 20,000-tile rules: {took:?}, {count} rules kept");
    assert!(took < limit, "took {took:?}");
    assert_eq!(
        count, 1,
        "the ordinary rule is kept, the oversized ones refused"
    );
}

/// Past 512 valid rules, the rest of a pack's rules are left out, so the shadowing check and the
/// table build stay bounded however many rule files a pack carries.
#[test]
fn a_pack_with_thousands_of_connected_texture_rules_keeps_a_bounded_number() {
    let mut pack = ctm_pack(b"method=ctm\nmatchTiles=glass\ntiles=0-46\n");
    for rule in 0..2000 {
        // Distinct metadata masks, so no rule shadows another.
        pack = pack.with(
            &format!("mcpatcher/ctm/glass/r{rule:04}.properties"),
            format!(
                "method=ctm\ntiles=0-46\nmatchTiles=glass t{rule}\nmetadata={}\nfaces={}\n",
                rule % 16,
                ["top", "bottom", "north", "south", "east", "west"][rule % 6]
            ),
        );
    }
    let limit = Duration::from_secs(10);
    let (count, took, _) = bounded("ctm many rules", limit, move || {
        CtmRules::load(&pack.into_manager(), &glass_textures())
            .rules()
            .len()
    });
    eprintln!("2001 rule files: {took:?}, {count} rules kept");
    assert!(took < limit, "took {took:?}");
    assert!(count <= 512, "{count} rules kept");
}

fn chain(mid_textures: &str, root_textures: &str, leaf_textures: &str) -> MemPack {
    MemPack::default()
        .with(
            "models/block/fuzz_leaf.json",
            format!(r#"{{"parent":"block/fuzz_mid","textures":{leaf_textures}}}"#),
        )
        .with(
            "models/block/fuzz_mid.json",
            format!(r#"{{"parent":"block/fuzz_root","textures":{mid_textures}}}"#),
        )
        .with(
            "models/block/fuzz_root.json",
            format!(
                r#"{{"textures":{root_textures},"elements":[{{"from":[0,0,0],"to":[16,16,16],"faces":{{"up":{{"texture":"x"}}}}}}]}}"#
            ),
        )
}

fn resolved(pack: MemPack) -> BTreeMap<String, String> {
    // A small stack: a resolution that recursed without end overflows it at once.
    on_thread(
        "model resolve",
        256 << 10,
        Duration::from_secs(20),
        move || {
            let leaf = ResourceLocation::minecraft("block/fuzz_leaf");
            let model = ResolvedModel::resolve(&pack.into_manager(), &leaf)
                .unwrap_or_else(|error| panic!("the chain does not resolve: {error:?}"));
            model.textures.into_iter().collect()
        },
    )
}

/// Two parent models naming each other's texture variables
/// (`{"a":"#b"}` above `{"b":"#a"}`) used to recurse until the stack overflowed, which ends the
/// process with no crash report. java: `ModelBlock.resolveTextureName` stops ("upward reference")
/// at the model that holds the alias being followed, wherever the walk meets it.
#[test]
fn an_alias_cycle_across_parent_models_resolves_to_missingno() {
    let textures = resolved(chain(r##"{"a":"#b"}"##, r##"{"b":"#a"}"##, "{}"));
    assert_eq!(textures.get("a").map(String::as_str), Some("missingno"));
    assert_eq!(textures.get("b").map(String::as_str), Some("missingno"));
}

/// The same rule, without a cycle: an alias may only name a variable defined below the model that
/// holds it (nearer the leaf), as in vanilla's `cube_all`; one defined above it is an upward
/// reference.
#[test]
fn texture_aliases_follow_vanillas_upward_reference_rule() {
    let textures = resolved(chain(
        r##"{"a":"#b","c":"#d"}"##,
        r##"{"b":"blocks/stone"}"##,
        r##"{"d":"blocks/dirt"}"##,
    ));
    // `b` lives above the model that says `#b`: java returns missingno.
    assert_eq!(textures.get("a").map(String::as_str), Some("missingno"));
    assert_eq!(textures.get("b").map(String::as_str), Some("blocks/stone"));
    // `d` lives in the leaf, below: resolved.
    assert_eq!(textures.get("c").map(String::as_str), Some("blocks/dirt"));
    assert_eq!(textures.get("d").map(String::as_str), Some("blocks/dirt"));
}

/// Every entry of an animation's `frames` list used to get its
/// own copy of the frame's pixels: `[0,0,0,...]` in a 200 KB `.mcmeta` held 100 MB for one 16x16
/// texture (java keeps each sheet frame once and plays the list by index).
#[test]
fn repeated_animation_frames_do_not_multiply_texture_memory() {
    let mut mcmeta = String::from(r#"{"animation":{"frames":[0"#);
    for _ in 1..100_000 {
        mcmeta.push_str(",0");
    }
    mcmeta.push_str("]}}");
    let pack = MemPack::default()
        .with("textures/blocks/fuzz.png", png(16, 16))
        .with("textures/blocks/fuzz.png.mcmeta", mcmeta);
    let (loaded, took, peak) = bounded("animation copies", Duration::from_secs(10), move || {
        let rm = pack.into_manager();
        mc_assets::sprite::load_sprite(&rm, &ResourceLocation::minecraft("textures/blocks/fuzz"))
            .map(|sprite| sprite.frames.len())
    });
    eprintln!(
        "100000 playback entries: {loaded:?} in {took:?}, peak {} KiB",
        peak >> 10
    );
    assert!(peak < 32 * MIB, "held {peak} bytes");
}

/// A long but ordinary playback list still loads in full.
#[test]
fn an_ordinary_long_animation_still_loads() {
    let frames: Vec<String> = (0..1000).map(|i| (i % 4).to_string()).collect();
    let mcmeta = format!(
        r#"{{"animation":{{"frametime":2,"frames":[{}]}}}}"#,
        frames.join(",")
    );
    let rm = MemPack::default()
        .with("textures/blocks/fuzz.png", png(16, 64))
        .with("textures/blocks/fuzz.png.mcmeta", mcmeta)
        .into_manager();
    let sprite =
        mc_assets::sprite::load_sprite(&rm, &ResourceLocation::minecraft("textures/blocks/fuzz"))
            .expect("loads");
    assert_eq!(sprite.frames.len(), 1000);
    assert_eq!(sprite.frames[5].rgba, sprite.frames[1].rgba);
}

/// A language file's template decides how much text a format
/// produces: `%500000000s` padded to half a gigabyte, and `%1$s` repeated over a long argument (a
/// book's author, an item's name: server text) multiplied it.
#[test]
fn language_templates_cannot_demand_gigabytes() {
    let (padded, _, peak) = bounded("format width", Duration::from_secs(10), || {
        format_template("%500000000s", &["x"]).len()
    });
    eprintln!("%500000000s: {padded} bytes, peak {} KiB", peak >> 10);
    assert!(peak < 16 * MIB, "held {peak} bytes");

    let (repeated, _, peak) = bounded("format repetition", Duration::from_secs(10), || {
        let template = "%1$s".repeat(10_000);
        let argument = "a".repeat(32 << 10);
        format_template(&template, &[&argument]).len()
    });
    eprintln!("10000 x 32 KiB: {repeated} bytes, peak {} KiB", peak >> 10);
    assert!(peak < 16 * MIB, "held {peak} bytes");

    // Ordinary widths and precisions are Java's, unchanged.
    assert_eq!(
        format_template("%5s|%-5s|%.2s", &["a", "b", "cdef"]),
        "    a|b    |cd"
    );
}

/// The bake with connected textures on, over the vanilla jar with 30 damaged rule files per
/// round (and their 47 tiles): rule parsing, shadowing, stitching and the mesher's table build.
#[test]
#[ignore = "needs the vanilla jar: MC_JARS_DIR=<jars> ... -- --ignored damaged_connected_texture"]
fn damaged_connected_texture_rules_bake_without_panics() {
    let jars = std::env::var_os("MC_JARS_DIR")
        .map(std::path::PathBuf::from)
        .expect("MC_JARS_DIR");
    let jar_path = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    let seeds: Vec<Vec<u8>> = strings(CTM_SEEDS)
        .into_iter()
        .chain([
            b"method=ctm\ntiles=0-46\nmatchBlocks=20 95 160:1,2 stone\nconnect=material\n".to_vec(),
            b"method=ctm\ntiles=0-46\nmatchTiles=planks_oak stone glass_red\nfaces=top\n".to_vec(),
        ])
        .collect();
    let mut rng = Rng(base_seed() ^ 0xC7A1);
    for round in 0..10 * round_multiplier() {
        let mut overlay = ctm_pack(b"method=ctm\nmatchTiles=glass\ntiles=0-46\n");
        for index in 0..30 {
            let mut damaged = seeds[rng.below(seeds.len())].clone();
            mutate(&mut rng, &mut damaged, PROPERTIES_TOKENS, 64 << 10);
            overlay = overlay.with(&format!("mcpatcher/ctm/glass/r{index}.properties"), damaged);
        }
        let jar = jar_path.clone();
        let (outcome, took, peak) =
            on_thread("ctm bake", 2 << 20, Duration::from_secs(600), move || {
                let mut rm = ResourceManager::new();
                rm.push_pack(Box::new(ZipResourcePack::open(&jar).expect("the jar")));
                rm.push_pack(Box::new(overlay));
                let base = PROCESS_LIVE.load(Ordering::Relaxed);
                PROCESS_PEAK.store(base, Ordering::Relaxed);
                let began = Instant::now();
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    mc_assets::model::bake_all_with_ctm(&rm, 4096, 4, true)
                        .ok()
                        .and_then(|(assets, _)| assets.ctm().map(|table| table.rule_count()))
                }));
                let peak = PROCESS_PEAK.load(Ordering::Relaxed) - base;
                (outcome.ok(), began.elapsed(), peak)
            });
        eprintln!(
            "ctm bake round {round}: rules={outcome:?} in {took:?}, peak {} MiB",
            peak / MIB
        );
        assert!(outcome.is_some(), "round {round}: the bake panicked");
        assert!(
            took < Duration::from_secs(120),
            "round {round}: took {took:?}"
        );
    }
}

/// The whole model bake over the vanilla jar with an overlay pack of 200 damaged block models
/// and blockstates per round (parsing, parent chains, element rotation, UV and cull-face maths,
/// variant selection). The first round bakes the jar alone as the memory baseline.
#[test]
#[ignore = "needs the vanilla jar: MC_JARS_DIR=<jars> ... -- --ignored damaged_models_bake"]
fn damaged_models_bake_without_panics() {
    let jars = std::env::var_os("MC_JARS_DIR")
        .map(std::path::PathBuf::from)
        .expect("MC_JARS_DIR");
    let jar_path = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    let vanilla = ZipResourcePack::open(&jar_path).expect("the 1.8.9 jar");
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    for prefix in ["models/block/", "blockstates/"] {
        for path in vanilla.list_files("minecraft", prefix) {
            let location = ResourceLocation::minecraft(path.clone());
            if let Some(mut reader) = vanilla.open(&location) {
                let mut bytes = Vec::new();
                if reader.read_to_end(&mut bytes).is_ok() {
                    files.push((path, bytes));
                }
            }
        }
    }
    files.sort();
    assert!(
        files.len() > 500,
        "{} model and blockstate files",
        files.len()
    );
    let mut rng = Rng(base_seed() ^ 0xBA4E);
    let mut baseline = None;
    for round in 0..=4 * round_multiplier() {
        let mut overlay = MemPack::default();
        if round > 0 {
            for _ in 0..200 {
                let (path, bytes) = &files[rng.below(files.len())];
                let mut damaged = bytes.clone();
                mutate(&mut rng, &mut damaged, JSON_TOKENS, 64 << 10);
                overlay = overlay.with(path, damaged);
            }
        }
        let jar = jar_path.clone();
        let (outcome, took, peak) =
            on_thread("model bake", 2 << 20, Duration::from_secs(600), move || {
                let mut rm = ResourceManager::new();
                rm.push_pack(Box::new(ZipResourcePack::open(&jar).expect("the jar")));
                rm.push_pack(Box::new(overlay));
                let base = PROCESS_LIVE.load(Ordering::Relaxed);
                PROCESS_PEAK.store(base, Ordering::Relaxed);
                let began = Instant::now();
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    mc_assets::bake_all(&rm, 4096).is_ok()
                }));
                let peak = PROCESS_PEAK.load(Ordering::Relaxed) - base;
                (outcome.ok(), began.elapsed(), peak)
            });
        eprintln!(
            "bake round {round}: baked={outcome:?} in {took:?}, peak {} MiB",
            peak / MIB
        );
        assert!(outcome.is_some(), "round {round}: the bake panicked");
        let baseline = *baseline.get_or_insert(peak);
        assert!(
            peak < baseline * 2 + 256 * MIB,
            "round {round}: held {peak} bytes (the jar alone: {baseline})"
        );
    }
}
