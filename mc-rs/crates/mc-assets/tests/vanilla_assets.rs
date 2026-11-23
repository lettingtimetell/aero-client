//! Integration tests against the real 1.8.9 assets checked in under `MCP-919-main/jars/`.
//!
//! These are the "Done when" tests for S04: *any vanilla texture, lang key, and model JSON
//! loads by `ResourceLocation`*.
//!
//! **These tests do not skip.** They used to: `require!` printed `SKIP:` and `return`ed, which
//! *passes*, and `cargo test` swallows a passing test's output — so on any checkout without the
//! jar this file reported eight green tests of the vanilla asset pipeline having opened
//! nothing. If the data is absent, every test now panics naming the path it looked at. Point
//! `MC_JARS_DIR` at a `jars/` directory to override. See `tests/common/mod.rs`.

mod common;

use common::{assets_dir, vanilla_jar};

use mc_assets::{
    parse_language_sections, AssetIndex, Language, ObjectStorePack, ResourceLocation,
    ResourceManager, ResourcePack, ZipResourcePack, PACK_FORMAT_1_8_9,
};

/// Was `let Some(x) = ... else { SKIP; return }`. Now the locator itself panics; this stays
/// only so the call sites keep naming what they need in the failure message.
macro_rules! require {
    ($f:ident, $what:literal) => {
        $f(concat!("vanilla_assets (", $what, ")"))
    };
}

fn loc(s: &str) -> ResourceLocation {
    ResourceLocation::parse(s).expect("test location should parse")
}

// ---------------------------------------------------------------- asset index

#[test]
fn asset_index_1_8_parses_and_points_at_real_blobs() {
    let assets = require!(assets_dir, "jars/assets");
    let index = AssetIndex::load(&assets.join("indexes").join("1.8.json")).expect("parse 1.8.json");

    assert!(
        index.len() > 700,
        "1.8.json should hold ~734 objects, got {}",
        index.len()
    );

    // Vanilla rewrites `minecraft/lang/en_GB.lang` to the location `minecraft:lang/en_GB.lang`.
    let l = loc("minecraft:lang/en_GB.lang");
    let obj = index.get(&l).expect("en_GB.lang should be indexed");
    assert_eq!(obj.hash.len(), 40);

    let blob = obj.path_in(&assets.join("objects"));
    assert!(blob.is_file(), "blob missing on disk: {}", blob.display());
    assert_eq!(
        std::fs::metadata(&blob).expect("stat blob").len(),
        obj.size,
        "blob size should match the index"
    );

    // The one entry without a namespace is only reachable as a raw key.
    assert!(index.get_raw("pack.mcmeta").is_some());
    assert!(index.namespaces().any(|n| n == "minecraft"));
    assert!(index.namespaces().any(|n| n == "realms"));
}

#[test]
fn object_store_pack_reads_a_real_lang_file() {
    let assets = require!(assets_dir, "jars/assets");
    let pack = ObjectStorePack::open(&assets, "1.8").expect("open object store");

    let l = loc("minecraft:lang/en_GB.lang");
    assert!(pack.contains(&l));

    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    let text = rm.load_string(&l).expect("read en_GB.lang");
    assert!(
        text.contains("gui.done="),
        "en_GB.lang should carry gui.done"
    );

    let mut lang = Language::new("en_GB");
    lang.merge_lang_str(&text);
    assert_eq!(lang.tr("gui.done"), "Done");
}

#[test]
fn the_real_pack_mcmeta_is_pack_format_1() {
    let assets = require!(assets_dir, "jars/assets");
    let pack = ObjectStorePack::open(&assets, "1.8").expect("open object store");

    let meta = pack
        .metadata()
        .expect("the object store carries pack.mcmeta");
    assert_eq!(meta.pack_format, i64::from(PACK_FORMAT_1_8_9));
    // `ResourcePackRepository.Entry` renders `IChatComponent.getFormattedText()`, whose
    // otherwise-styleless component still carries its trailing reset.
    assert_eq!(meta.description, "The default look of Minecraft\u{a7}r");
    meta.require_supported().expect("1.8.9 packs are format 1");

    let raw = pack.read_raw("pack.mcmeta").expect("read pack.mcmeta blob");
    let langs = parse_language_sections(&raw).expect("parse language sections");
    assert!(
        langs.len() > 70,
        "1.8 declares ~75 languages, got {}",
        langs.len()
    );
    let gb = langs
        .iter()
        .find(|l| l.code == "en_GB")
        .expect("en_GB should be declared");
    assert_eq!(gb.name, "English");
    assert_eq!(gb.to_string(), "English (UK)");
    assert!(
        langs.iter().any(|l| l.bidirectional),
        "1.8 declares at least one RTL language"
    );
}

// ---------------------------------------------------------------- vanilla jar

#[test]
fn vanilla_jar_opens_as_a_zip_pack() {
    let jar = require!(vanilla_jar, "jars/versions/1.8.9/1.8.9.jar");
    let pack = ZipResourcePack::open(&jar).expect("open 1.8.9.jar");

    assert_eq!(pack.name(), "1.8.9.jar");
    assert!(pack.entry_count() > 5_000, "jar should hold ~5,597 entries");
    assert_eq!(pack.namespaces(), vec!["minecraft".to_owned()]);
    // The vanilla jar genuinely has no pack.mcmeta — that must not be an open failure.
    assert!(pack.metadata().is_none());
}

#[test]
fn a_vanilla_texture_model_and_lang_file_load_by_location() {
    let jar = require!(vanilla_jar, "jars/versions/1.8.9/1.8.9.jar");
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));

    // Texture.
    let png = rm
        .load(&loc("textures/blocks/stone.png"))
        .expect("load stone.png");
    assert_eq!(
        &png[..8],
        b"\x89PNG\r\n\x1a\n",
        "stone.png should have a PNG signature"
    );

    // Model JSON.
    #[derive(serde::Deserialize)]
    struct Model {
        parent: String,
        textures: std::collections::BTreeMap<String, String>,
    }
    let model: Model = rm
        .load_json(&loc("models/block/stone.json"))
        .expect("load stone.json");
    assert_eq!(model.parent, "block/cube_all");
    assert_eq!(
        model.textures.get("all").map(String::as_str),
        Some("blocks/stone")
    );

    // Blockstate JSON — S09's other input.
    assert!(rm.contains(&loc("blockstates/stone.json")));

    // Lang.
    let lang = Language::load(&rm, &["en_US"]);
    assert!(
        lang.len() > 2_500,
        "en_US.lang has ~2,553 keys, got {}",
        lang.len()
    );
    assert_eq!(lang.tr("gui.done"), "Done");
    assert_eq!(lang.tr("menu.singleplayer"), "Singleplayer");
    assert_eq!(lang.tr("tile.stone.stone.name"), "Stone");
    assert_eq!(lang.code(), "en_US");
    assert!(!lang.is_unicode(), "en_US is pure ASCII");
}

#[test]
fn the_real_translation_test_key_formats_like_java() {
    let jar = require!(vanilla_jar, "jars/versions/1.8.9/1.8.9.jar");
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let lang = Language::load(&rm, &["en_US"]);

    // `translation.test.complex` exists in en_US.lang precisely to exercise the mixed
    // implicit/explicit argument rules. Expected output taken from JDK 21's String.format
    // applied to the vanilla-normalised template.
    assert_eq!(
        lang.tr("translation.test.complex"),
        "Prefix, %s%2$s again %s and %1$s lastly %s and also %1$s again!"
    );
    assert_eq!(
        lang.format("translation.test.complex", &["A", "B", "C", "D", "E"]),
        "Prefix, AB again B and A lastly C and also A again!"
    );

    // Explicit positional form, straight out of the same file.
    assert_eq!(
        lang.format("demo.help.movement", &["W", "A", "S", "D"]),
        "Use W, A, S, D and the mouse to move around"
    );

    // Mojang ships these two malformed entries as Formatter test vectors. Locale catches
    // each IllegalFormatException and replaces the whole output, including the untouched
    // template, rather than partially substituting it.
    assert_eq!(lang.tr("translation.test.invalid"), "hi %");
    assert_eq!(
        lang.format("translation.test.invalid", &["A"]),
        "Format error: hi %"
    );
    assert_eq!(lang.tr("translation.test.invalid2"), "hi %  s");
    assert_eq!(
        lang.format("translation.test.invalid2", &["A"]),
        "Format error: hi %  s"
    );
}

#[test]
fn unresolved_w_lang_keys_are_caller_bugs_not_locale_aliases() {
    let jar = require!(vanilla_jar, "jars/versions/1.8.9/1.8.9.jar");
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let lang = Language::load(&rm, &["en_US"]);

    // These base keys are absent because the Java Item/Block override appends a metadata
    // variant before asking Locale. Guessing aliases here would give every metadata value the
    // same English name and would also bypass translations supplied by resource packs.
    for key in [
        "tile.sapling.name",
        "tile.cobbleWall.name",
        "tile.stoneSlab2.name",
        "item.fish.name",
        "item.skull.name",
    ] {
        assert!(!lang.has_key(key), "{key} unexpectedly exists in 1.8.9");
        assert_eq!(lang.tr(key), key, "Locale must preserve a genuine miss");
    }
    for key in [
        "tile.sapling.oak.name",
        "tile.cobbleWall.normal.name",
        "tile.stoneSlab2.red_sandstone.name",
        "item.fish.cod.raw.name",
        "item.skull.skeleton.name",
    ] {
        assert!(lang.has_key(key), "metadata-specific key {key} is shipped");
        assert_ne!(lang.tr(key), key);
    }

    // These W-LANG strings need their existing 1.8.9 keys routed to Language by their UI
    // callers; no compatibility alias is needed or correct.
    for key in ["disconnect.lost", "addServer.title", "container.inventory"] {
        assert!(lang.has_key(key), "UI key {key} is shipped by 1.8.9");
    }
    assert!(
        !lang.has_key("multiplayer.status.and_more"),
        "OldServerPinger hardcodes this English sentence in Java"
    );
}

// ------------------------------------------------------------ stacked manager

#[test]
fn the_full_vanilla_stack_resolves_every_asset_kind() {
    let assets = require!(assets_dir, "jars/assets");
    let jar = require!(vanilla_jar, "jars/versions/1.8.9/1.8.9.jar");

    // Vanilla order: the launcher's object store underneath, the version jar on top.
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        ObjectStorePack::open(&assets, "1.8").expect("open object store"),
    ));
    rm.push_pack(Box::new(
        ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));

    assert_eq!(rm.pack_count(), 2);
    assert!(rm.namespaces().contains("minecraft"));
    assert!(
        rm.namespaces().contains("realms"),
        "the object store contributes the realms namespace"
    );

    // Texture — only the jar has it.
    assert!(rm.contains(&loc("textures/blocks/dirt.png")));
    // Sound — only the object store has it.
    assert!(rm.contains(&loc("sounds/random/click.ogg")));
    // Model JSON — only the jar.
    assert!(rm.load(&loc("models/item/diamond_sword.json")).is_ok());
    // Font metadata used by S15.
    assert!(rm.contains(&loc("textures/font/ascii.png")));

    // en_US comes from the jar, en_GB from the object store: both resolve through one stack.
    let lang = Language::load(&rm, &["en_US", "en_GB"]);
    assert_eq!(lang.code(), "en_GB");
    assert_eq!(lang.tr("gui.done"), "Done");
    assert!(lang.has_key("tile.stone.stone.name"));
    // en_GB has fewer keys than en_US; loading en_US first as a base is exactly why the
    // union survives instead of the smaller file replacing the larger one.
    assert!(
        lang.len() > 2_500,
        "en_US ∪ en_GB should exceed either alone, got {}",
        lang.len()
    );
}

#[test]
fn a_folder_pack_on_top_overrides_per_resource() {
    let jar = require!(vanilla_jar, "jars/versions/1.8.9/1.8.9.jar");

    // Build a one-file folder pack that overrides exactly one model.
    let dir = std::env::temp_dir().join(format!("mc-assets-s04-{}", std::process::id()));
    let models = dir.join("assets/minecraft/models/block");
    std::fs::create_dir_all(&models).expect("create temp pack");
    std::fs::write(
        models.join("stone.json"),
        br#"{"parent":"block/overridden"}"#,
    )
    .expect("write override");
    std::fs::write(
        dir.join("pack.mcmeta"),
        br#"{"pack":{"pack_format":1,"description":"S04 test pack"}}"#,
    )
    .expect("write pack.mcmeta");

    let result = std::panic::catch_unwind(|| {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(
            ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
        ));
        let folder = mc_assets::FolderResourcePack::open(&dir).expect("open folder pack");
        assert_eq!(folder.metadata().map(|m| m.pack_format), Some(1));
        assert_eq!(folder.namespaces(), vec!["minecraft".to_owned()]);
        rm.push_pack(Box::new(folder));

        // Overridden by the top pack...
        let overridden = rm
            .load_string(&loc("models/block/stone.json"))
            .expect("load overridden stone.json");
        assert!(overridden.contains("block/overridden"));
        assert_eq!(
            rm.resource(&loc("models/block/stone.json"))
                .unwrap()
                .pack_name,
            dir.file_name().unwrap().to_string_lossy()
        );

        // ...while everything else still falls through to the jar. This is the vanilla
        // FallbackResourceManager behaviour: fallthrough is per resource, not per pack.
        let dirt = rm
            .load_string(&loc("models/block/dirt.json"))
            .expect("load dirt.json from the jar below");
        assert!(dirt.contains("cube_all"));
        assert_eq!(
            rm.resource(&loc("models/block/dirt.json"))
                .unwrap()
                .pack_name,
            "1.8.9.jar"
        );
    });

    std::fs::remove_dir_all(&dir).ok();
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn reloading_swaps_the_stack_at_runtime() {
    let jar = require!(vanilla_jar, "jars/versions/1.8.9/1.8.9.jar");

    let mut rm = ResourceManager::new();
    assert_eq!(rm.generation(), 0);
    assert!(rm.load(&loc("models/block/stone.json")).is_err());

    rm.reload(vec![Box::new(
        ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    )]);

    assert_eq!(rm.generation(), 1);
    assert_eq!(rm.pack_names(), ["1.8.9.jar"]);
    assert!(rm.load(&loc("models/block/stone.json")).is_ok());

    rm.reload(Vec::new());
    assert_eq!(rm.generation(), 2);
    assert_eq!(rm.pack_count(), 0);
    assert!(rm.load(&loc("models/block/stone.json")).is_err());
}

#[test]
fn every_indexed_object_resolves_or_is_absent_from_disk_without_panicking() {
    let assets = require!(assets_dir, "jars/assets");
    let index = AssetIndex::load(&assets.join("indexes").join("1.8.json")).expect("parse 1.8.json");
    let pack = ObjectStorePack::open(&assets, "1.8").expect("open object store");

    let mut resolved = 0usize;
    for (key, _) in index.iter() {
        // `pack.mcmeta` carries no namespace, so it parses as `minecraft:pack.mcmeta` and
        // simply misses — which is the point: nothing here may panic.
        let Ok(l) = ResourceLocation::parse(key) else {
            continue;
        };
        if pack.contains(&l) {
            resolved += 1;
        }
    }
    assert!(
        resolved > 600,
        "most of the 734 indexed objects should be present, got {resolved}"
    );
}

// -------------------------------------------------------- the negative control

/// The control on "these tests do not skip".
///
/// It drives [`common::locate_jar`]/[`common::locate_assets`]' absent branch directly rather
/// than mutating `MC_JARS_DIR`, which would race the other tests in this binary, and it also
/// asserts the *real* data is found — so it cannot pass because everything is missing.
#[test]
fn the_vanilla_suite_fails_loudly_when_the_data_is_absent() {
    let missing = common::definitely_missing();
    let err = common::locate_jar(&missing).expect_err("a missing directory must not yield a jar");
    assert!(err.contains("is not a directory"), "{err}");
    let err =
        common::locate_assets(&missing).expect_err("a missing directory must not yield assets");
    assert!(err.contains("is not a directory"), "{err}");

    let empty = tempfile::tempdir().expect("tempdir");
    let err =
        common::locate_jar(empty.path()).expect_err("an empty directory must not yield a jar");
    assert!(err.ends_with("1.8.9.jar is not a file"), "{err}");
    let err =
        common::locate_assets(empty.path()).expect_err("an empty directory must not yield assets");
    assert!(err.ends_with("1.8.json is not a file"), "{err}");

    // ...and the real thing is genuinely present, so the four assertions above are not passing
    // because this checkout has no data at all.
    let jars = common::jars_dir();
    assert!(
        common::locate_jar(&jars).is_ok(),
        "the real jar must be found under {}",
        jars.display()
    );
    assert!(
        common::locate_assets(&jars).is_ok(),
        "the real asset index must be found under {}",
        jars.display()
    );
}
