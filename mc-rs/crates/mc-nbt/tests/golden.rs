//! Golden tests against real data.
//!
//! Two sources of truth:
//!
//! * every `.dat` shipped in `MCP-919-main/jars/` — `servers.dat` (raw NBT) and the
//!   development world's `level.dat` (gzipped);
//! * `tests/data/java_writeutf.nbt`, produced by `tests/data/GenNbt.java` running on a real
//!   JDK's `DataOutputStream`, which pins the hazard-H2 modified-UTF-8 behaviour to Java's
//!   own output rather than to our reading of the spec.
//!
//! **What "byte-identical" means here.** Gzip re-compression is not reproducible across
//! encoders — flate2 and `java.util.zip` will not agree on a byte stream — so the property
//! under test is that the *decompressed NBT payload* survives a parse/serialise cycle
//! unchanged. For `servers.dat`, which is stored raw, that is the whole file.

use std::path::{Path, PathBuf};

use mc_nbt::{compress, Framing, Nbt, NbtLimits, TagType};

/// `<repo root>` — the manifest lives at `<repo>/mc-rs/crates/mc-nbt`.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("mc-nbt sits three levels below the repo root")
        .to_path_buf()
}

fn jars_dir() -> PathBuf {
    repo_root().join("MCP-919-main").join("jars")
}

/// Every `.dat`/`.dat_old` under `jars/`, recursively.
fn dat_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dat_files(&path, out);
        } else {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "dat" || ext == "dat_old" {
                out.push(path);
            }
        }
    }
}

/// The raw NBT bytes of a `.dat`, whatever framing it uses.
fn decompressed(path: &Path) -> Vec<u8> {
    use std::io::Read;
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    match compress::sniff(&raw) {
        Framing::Raw => raw,
        Framing::Gzip => {
            let mut out = Vec::new();
            flate2_gunzip(&raw, &mut out);
            out
        }
        Framing::Zlib => {
            let mut out = Vec::new();
            flate2::read::ZlibDecoder::new(&raw[..])
                .read_to_end(&mut out)
                .unwrap_or_else(|e| panic!("inflate {}: {e}", path.display()));
            out
        }
    }
}

fn flate2_gunzip(raw: &[u8], out: &mut Vec<u8>) {
    use std::io::Read;
    flate2::read::GzDecoder::new(raw)
        .read_to_end(out)
        .expect("gunzip");
}

#[test]
fn every_dat_in_jars_round_trips_byte_identically() {
    let jars = jars_dir();
    assert!(
        jars.is_dir(),
        "expected the reference data at {}",
        jars.display()
    );

    let mut files = Vec::new();
    dat_files(&jars, &mut files);
    files.sort();
    assert!(
        files.len() >= 2,
        "expected at least servers.dat and level.dat under {}, found {files:?}",
        jars.display()
    );

    for path in &files {
        let payload = decompressed(path);
        let (name, value) = mc_nbt::from_slice_with_limits(&payload, NbtLimits::UNLIMITED)
            .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        let reencoded = mc_nbt::to_vec(&name, &value)
            .unwrap_or_else(|e| panic!("re-encode {}: {e}", path.display()));
        assert_eq!(
            reencoded,
            payload,
            "{} did not round-trip byte-identically",
            path.display()
        );

        // And the same value survives a second parse of our own output.
        let (name2, value2) = mc_nbt::from_slice_with_limits(&reencoded, NbtLimits::UNLIMITED)
            .expect("re-parse our own output");
        assert_eq!(name2, name);
        assert_eq!(value2, value);
    }

    eprintln!("round-tripped {} .dat files byte-identically", files.len());
}

#[test]
fn servers_dat_parses_to_the_expected_server_list() {
    // The exact bytes are short enough to state: raw NBT, no gzip.
    let path = jars_dir().join("servers.dat");
    let bytes = std::fs::read(&path).expect("servers.dat");
    assert_eq!(compress::sniff(&bytes), Framing::Raw);

    let (name, root) = mc_nbt::read_compound_with_limits(&mut &bytes[..], NbtLimits::UNLIMITED)
        .expect("servers.dat is a named root compound");
    assert_eq!(name, "");

    let servers = root.get_list("servers").expect("`servers` list");
    assert_eq!(servers.len(), 1);
    let entry = servers[0].as_compound().expect("entry is a compound");
    assert_eq!(entry.get_str("name"), Some("MCP Test Server"));
    assert_eq!(entry.get_str("ip"), Some("127.0.0.1"));

    // This is the shape S16's multiplayer screen codes against.
    assert_eq!(root.get("servers").map(Nbt::tag_type), Some(TagType::List));
    assert_eq!(mc_nbt::to_vec(&name, &Nbt::Compound(root)).unwrap(), bytes);
}

#[test]
fn level_dat_parses_and_is_gzipped() {
    let path = jars_dir()
        .join("saves")
        .join("MCP Development World")
        .join("level.dat");
    let raw = std::fs::read(&path).expect("level.dat");
    assert_eq!(compress::sniff(&raw), Framing::Gzip);

    let (name, value) = mc_nbt::read_gzip_with_limits(&mut &raw[..], NbtLimits::UNLIMITED)
        .expect("gzipped level.dat");
    assert_eq!(name, "");

    let data = value
        .as_compound()
        .and_then(|c| c.get_compound("Data"))
        .expect("Data compound");

    assert_eq!(data.get_str("LevelName"), Some("mcpworld"));
    assert_eq!(data.get_str("generatorName"), Some("largeBiomes"));
    assert_eq!(data.get_i32("version"), Some(19133));
    assert_eq!(data.get_i64("RandomSeed"), Some(-1_587_754_402));
    assert_eq!(data.get_i64("Time"), Some(1620));
    assert_eq!(data.get_i32("SpawnX"), Some(584));
    assert_eq!(data.get_i32("SpawnY"), Some(64));
    assert_eq!(data.get_i32("SpawnZ"), Some(166));
    assert_eq!(data.get_bool("initialized"), Some(true));
    assert_eq!(data.get_bool("raining"), Some(false));

    let rules = data.get_compound("GameRules").expect("GameRules");
    assert_eq!(rules.len(), 9);
    assert_eq!(rules.get_str("keepInventory"), Some("false"));

    // The decompressed payload round-trips; the gzip container itself does not, because
    // flate2 and java.util.zip pick different (equally valid) deflate encodings.
    let payload = decompressed(&path);
    assert_eq!(mc_nbt::to_vec(&name, &value).unwrap(), payload);

    // Our own gzip container does round-trip through our own reader.
    let mut regz = Vec::new();
    compress::write_gzip(&mut regz, &name, &value).expect("gzip write");
    let (name2, value2) =
        mc_nbt::read_gzip_with_limits(&mut &regz[..], NbtLimits::UNLIMITED).expect("gzip read");
    assert_eq!((name2, value2), (name, value));
}

/// Hazard H2, pinned against Java itself.
///
/// `tests/data/java_writeutf.nbt` was written by `DataOutputStream.writeUTF` on JDK 21.
/// If this test fails, our modified-UTF-8 codec has drifted from Java's.
#[test]
fn java_written_modified_utf8_matches_byte_for_byte() {
    let bytes = include_bytes!("data/java_writeutf.nbt");
    let (name, value) =
        mc_nbt::from_slice_with_limits(bytes, NbtLimits::UNLIMITED).expect("Java-written NBT");

    assert_eq!(name, "r\u{f6}\u{f6}t");
    let c = value.as_compound().expect("root compound");

    // NUL is two bytes on the wire, in values...
    assert_eq!(c.get_str("nul"), Some("a\0b"));
    // ...and in keys.
    assert_eq!(c.get_str("k\0ey"), Some("nul in the key too"));
    // Astral characters are CESU-8 surrogate pairs, six bytes each.
    assert_eq!(c.get_str("emoji"), Some("\u{1F600}\u{1F680}"));
    assert_eq!(
        c.get_str("mixed"),
        Some("caf\u{e9} \u{4E2D}\u{6587} \u{10FFFF}")
    );

    assert_eq!(c.get_i8("byte"), Some(i8::MIN));
    assert_eq!(c.get_i16("short"), Some(i16::MIN));
    assert_eq!(c.get_i32("int"), Some(-559_038_737)); // 0xDEADBEEF
    assert_eq!(c.get_i64("long"), Some(i64::MIN));
    assert_eq!(c.get_f32("float"), Some(f32::MAX));
    assert_eq!(c.get_f64("double"), Some(-0.0));
    assert!(c.get_f64("double").unwrap().is_sign_negative());
    assert_eq!(c.get_bytes("bytes"), Some(&[0x00, 0xFF, 0x7F, 0x80][..]));
    assert_eq!(c.get_ints("ints"), Some(&[i32::MIN, 0, i32::MAX][..]));
    assert_eq!(c.get_list("emptyList").map(<[Nbt]>::len), Some(0));

    let list = c.get_list("list").expect("list of compounds");
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].as_compound().unwrap().get_str("s"), Some("one"));
    assert_eq!(list[1].as_compound().unwrap().get_str("s"), Some("two\0"));

    // The whole point: what we write back is what the JVM wrote.
    assert_eq!(mc_nbt::to_vec(&name, &value).unwrap(), bytes);
}

/// The fixture must actually exercise the trap, not merely happen to pass.
#[test]
fn the_java_fixture_is_not_valid_standard_utf8() {
    let bytes: Vec<u8> = include_bytes!("data/java_writeutf.nbt").to_vec();
    assert!(
        bytes.windows(2).any(|w| w == [0xC0, 0x80]),
        "Java's overlong NUL is present"
    );
    assert!(
        bytes
            .windows(6)
            .any(|w| w == [0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80]),
        "the CESU-8 encoding of U+1F600 is present"
    );
    assert!(
        !bytes.windows(4).any(|w| w == [0xF0, 0x9F, 0x98, 0x80]),
        "and the standard-UTF-8 encoding of U+1F600 is not"
    );
    // Both sequences are illegal standard UTF-8, so the naive decode fails outright.
    assert!(String::from_utf8(bytes).is_err(), "hazard H2, demonstrated");
}
