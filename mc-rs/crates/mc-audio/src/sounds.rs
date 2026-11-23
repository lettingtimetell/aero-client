//! `sounds.json` parsing and the weighted event → variant registry.
//!
// java: SoundHandler.java (onResourceManagerReload, loadSoundResource),
// SoundList.java, SoundListSerializer.java, SoundEventAccessorComposite.java

use std::collections::HashMap;

use mc_assets::{Presence, ResourceLocation, ResourceManager};
use mc_math::JavaRandom;

use crate::category::SoundCategory;
use crate::AudioError;

/// How deep [`SoundRegistry`] will chase `"type": "event"` indirection before giving up.
///
/// Vanilla has no such guard — a cyclic `sounds.json` (`a` points at `b`, `b` points at `a`)
/// would recurse forever in `SoundEventAccessorComposite.cloneEntry`. A malicious resource pack
/// is untrusted input (per §11), so this crate bounds it instead. No vanilla asset comes close;
/// the deepest real chain (`music.game.creative` → `music.game`) is 2.
const MAX_INDIRECTION_DEPTH: u8 = 16;

/// How many events one pick, or one prewarm walk from one
/// event, may expand. The depth cap alone bounded the length of a chain, not the work: every
/// level sums the weight of each of its entries again, so one event naming itself ten times
/// asked for 10^17 steps, at prewarm (after every pack load) and at each play. A real chain
/// expands a handful of events.
const MAX_INDIRECTION_STEPS: u32 = 4096;

/// The largest `sounds.json` parsed: vanilla's is 37 KB.
pub const MAX_SOUNDS_JSON_BYTES: usize = 4 << 20;

/// One file the event can play, or a reference to another event.
#[derive(Debug, Clone, PartialEq)]
enum VariantKind {
    /// `sounds/<path>.ogg` plus its per-entry pitch/volume/streaming multipliers.
    File {
        location: ResourceLocation,
        volume: f32,
        pitch: f32,
        streaming: bool,
        /// The pack only listed the file (`mc_assets::Presence::Listed`); nothing opened it yet.
        /// [`crate::AudioEngine`]'s worker checks it after load; picks count it as present.
        presumed: bool,
    },
    /// `{"type": "event", "name": "..."}` — re-roll against another registered event.
    Event { location: ResourceLocation },
}

/// One entry in an event's weighted pool.
#[derive(Debug, Clone, PartialEq)]
struct Variant {
    /// Always `>= 1`; `SoundListSerializer` rejects non-positive weights on read.
    weight: u32,
    kind: VariantKind,
}

/// A registered sound event: its mixer category and its weighted variant pool.
// java: SoundEventAccessorComposite
#[derive(Debug, Clone, Default, PartialEq)]
struct SoundEventDef {
    category: SoundCategory,
    variants: Vec<Variant>,
}

/// A specific file, chosen and fully resolved: the `.ogg` [`ResourceLocation`], its pitch/volume
/// multipliers, and whether it should stream rather than be fully buffered.
///
/// java: the [`SoundPoolEntry`] `SoundEventAccessorComposite.cloneEntry()` returns, after this
/// crate resolves any `event`-type indirection down to an actual file.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedVariant {
    /// The `.ogg` resource, e.g. `minecraft:sounds/damage/fallbig.ogg`.
    pub location: ResourceLocation,
    /// The per-entry volume multiplier (`sounds.json`'s `"volume"`, default `1.0`).
    pub volume: f32,
    /// The per-entry pitch multiplier (`sounds.json`'s `"pitch"`, default `1.0`).
    pub pitch: f32,
    /// Whether this file should be decoded incrementally (music, records) rather than fully
    /// buffered up front (short effects).
    pub streaming: bool,
}

/// The event → weighted-variant-list table, loaded from every `<namespace>:sounds.json` a
/// [`ResourceManager`] can see.
///
/// java: `SoundHandler` (the registry half of it — playback is [`crate::AudioEngine`]).
#[derive(Debug, Clone, Default)]
pub struct SoundRegistry {
    events: HashMap<ResourceLocation, SoundEventDef>,
}

impl SoundRegistry {
    /// An empty registry with no events. [`SoundRegistry::category_of`] and
    /// [`SoundRegistry::pick_variant`] return `None` for everything until [`load`] populates it.
    ///
    /// [`load`]: SoundRegistry::load
    pub fn empty() -> Self {
        Self::default()
    }

    /// Loads every `<namespace>:sounds.json` the manager's stacked packs provide, merging them
    /// exactly as `SoundHandler.onResourceManagerReload` does.
    ///
    /// Packs are walked bottom-to-top (lowest priority first, matching
    /// [`ResourceManager::all_resources`]). For a given event key, by default each pack's
    /// entries are *appended* to the ones before it — a resource pack that only wants to add a
    /// couple of extra variants doesn't need to repeat vanilla's. A pack can instead set
    /// `"replace": true` on an event to discard everything registered for it so far and start
    /// that event's pool over from just its own entries.
    // java: SoundHandler.onResourceManagerReload / loadSoundResource
    /// # Errors
    ///
    /// Never, in practice — the signature keeps `Result` because it is public API, but every
    /// per-pack failure is now contained. See below.
    pub fn load(resources: &ResourceManager) -> Result<Self, AudioError> {
        let mut registry = Self::default();
        for namespace in resources.namespaces().clone() {
            let sounds_json = ResourceLocation::new(&namespace, "sounds.json");
            for resource in resources.all_resources(&sounds_json) {
                // java: `SoundHandler.onResourceManagerReload:69-83` puts
                // `catch (RuntimeException) { logger.warn("Invalid sounds.json", …) }` **inside**
                // the per-`IResource` loop, so a bad category, a non-positive weight or a JSON
                // syntax error costs exactly that one pack's one file. This used to propagate
                // with `?`, and `GameAssets` turns any `Err` into `SoundRegistry::empty()` — so
                // one typo in one installed resource pack silenced the whole game
                // (`F-P2-client.audio-02-5`).
                // Parsed, every one-letter sound name costs dozens of bytes:
                // a pack's 60 MB file (a few dozen KB zipped) peaked at 1.5 GB. A file over the
                // cap, over a hundred times vanilla's, is skipped like an invalid one.
                if resource.data.len() > MAX_SOUNDS_JSON_BYTES {
                    tracing::warn!(
                        namespace = %namespace,
                        bytes = resource.data.len(),
                        "sounds.json larger than any real one; skipped"
                    );
                    continue;
                }
                let raw = match parse_sounds_json(&resource.data, &sounds_json.to_string()) {
                    Ok(raw) => raw,
                    Err(error) => {
                        tracing::warn!(%error, namespace = %namespace, "invalid sounds.json");
                        continue;
                    }
                };
                // java: the category is validated by `SoundListSerializer.deserialize`
                // (`:158`), i.e. inside `getSoundMap` — *before* any event of this file is
                // registered. So one bad category costs the whole file, not one event, and
                // never anything outside it. Checked up front here for the same reason.
                if let Some(bad) = raw
                    .iter()
                    .find(|(_, list)| SoundCategory::parse(&list.category).is_none())
                {
                    tracing::warn!(
                        namespace = %namespace,
                        event = %bad.0,
                        category = %bad.1.category,
                        "invalid sounds.json: unknown category, skipping this file"
                    );
                    continue;
                }
                for (event_name, list) in raw {
                    if let Err(error) =
                        registry.merge_event(resources, &namespace, &event_name, list)
                    {
                        tracing::warn!(%error, namespace = %namespace, "invalid sounds.json");
                    }
                }
            }
        }
        Ok(registry)
    }

    fn merge_event(
        &mut self,
        resources: &ResourceManager,
        namespace: &str,
        event_name: &str,
        list: RawSoundList,
    ) -> Result<(), AudioError> {
        let location = ResourceLocation::new(namespace, event_name);
        let category =
            SoundCategory::parse(&list.category).ok_or_else(|| AudioError::InvalidSoundList {
                location: location.to_string(),
                reason: format!("unknown category {:?}", list.category),
            })?;

        // A later pack's category wins even without `replace`, matching the Java: the composite
        // object is only ever constructed once (first sighting or a `replace`), so a non-replace
        // resighting keeps the *original* category. Mirror that: an explicit `replace`
        // unconditionally overwrites; otherwise fall back to whatever is already there (or
        // construct fresh on first sighting).
        if list.replace {
            self.events.insert(
                location.clone(),
                SoundEventDef {
                    category,
                    variants: Vec::new(),
                },
            );
        }
        let def = self
            .events
            .entry(location.clone())
            .or_insert_with(|| SoundEventDef {
                category,
                variants: Vec::new(),
            });

        for raw_entry in list.sounds {
            // Keep the two Java locations distinct.  `new ResourceLocation(s)` applies the
            // historical one-character-domain rule (`a:b` is minecraft:`b`), and the FILE
            // branch then uses that parsed path.  The SOUND_EVENT branch intentionally passes
            // the *raw* name as the path to the two-argument constructor, including a colon.
            // This looks odd, but it is what `SoundHandler.loadSoundResource` does and matters
            // for malformed/cross-domain pack entries.
            let parsed_name = ResourceLocation::parse(&raw_entry.name).expect("parse is total");
            let entry_namespace = if raw_entry.name.contains(':') {
                parsed_name.namespace.clone()
            } else {
                namespace.to_owned()
            };
            let file_target = ResourceLocation::new(&entry_namespace, parsed_name.path);
            let event_target = ResourceLocation::new(&entry_namespace, raw_entry.name.clone());

            // `SOUND_EVENT` intentionally keeps the raw entry name as its path.
            let variant = match raw_entry.kind {
                RawKind::File => {
                    let ogg = file_target.with_prefix("sounds/").with_suffix(".ogg");
                    // java: `loadSoundResource`'s `FILE` branch *opens* the `.ogg` at load time
                    // and `continue`s on `FileNotFoundException` with
                    // `"File {} does not exist, cannot add it to event {}"`, so the entry never
                    // reaches the pool and never contributes to `getWeight()`. Keeping it (the
                    // old behaviour) both skewed the weighted draw and produced silence whenever
                    // the missing variant won. `F-P2-client.audio-02-6`.
                    // Java opens the resource while building the accessor and drops both a
                    // missing file and an I/O failure.  `contains` only answers the first half:
                    // an indexed object can still be absent on disk, and a zip/folder resource
                    // can be present but unreadable.  Probe the same reader seam here so such a
                    // variant cannot remain in the weighted pool and later resolve to silence.
                    //
                    // Opening every file was most of the
                    // start-up time (1.4 s cold, 4-9 s from a USB disk: 617 opens for vanilla,
                    // and a zip pack's `open` inflates the whole file). A pack that lists the
                    // file in memory (the asset index, a zip's entries) is now believed without
                    // an open. The engine's worker checks such files after load and a missing
                    // one then leaves every pool, as here; a pick never asks the disk
                    // (`AudioEngine::pick_existing`).
                    let presumed = match resources.presence(&ogg) {
                        Presence::Listed => true,
                        Presence::Readable => false,
                        Presence::Missing => {
                            tracing::warn!(
                                file = %ogg,
                                event = %location,
                                "file does not exist, cannot add it to event"
                            );
                            continue;
                        }
                    };
                    Variant {
                        weight: raw_entry.weight,
                        kind: VariantKind::File {
                            location: ogg,
                            volume: raw_entry.volume,
                            pitch: raw_entry.pitch,
                            streaming: raw_entry.stream,
                            presumed,
                        },
                    }
                }
                RawKind::Event => Variant {
                    weight: raw_entry.weight,
                    kind: VariantKind::Event {
                        location: event_target,
                    },
                },
            };
            def.variants.push(variant);
        }
        Ok(())
    }

    /// The mixer category registered for `event`, or `None` if nothing registered it.
    pub fn category_of(&self, event: &ResourceLocation) -> Option<SoundCategory> {
        self.events.get(event).map(|d| d.category)
    }

    /// Whether any `sounds.json` registered this event at all.
    pub fn contains(&self, event: &ResourceLocation) -> bool {
        self.events.contains_key(event)
    }

    /// Number of distinct events registered.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the registry has no events at all.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Every registered event and its category, in no particular order (the Sounds mod's list).
    pub fn events(&self) -> impl Iterator<Item = (&ResourceLocation, SoundCategory)> {
        self.events.iter().map(|(event, def)| (event, def.category))
    }

    /// Chooses one registered event whose category occurs in `categories`.
    ///
    /// This is the registry half of `SoundHandler.getRandomSoundFromCategories`.  Vanilla walks
    /// its `HashMap` key set, keeps every composite for which `ArrayUtils.contains(categories,
    /// category)` is true, then calls a fresh `Random().nextInt(matches.size())`.  Duplicate
    /// categories therefore do not weight an event twice, and an empty match set returns `null`
    /// without drawing randomness.  The Java map's iteration order and fresh wall-clock-seeded
    /// `Random` are deliberately not observable contracts, so this API takes the caller's
    /// `JavaRandom` explicitly while retaining the same uniform `nextInt(count)` selection.
    ///
    /// The only vanilla caller supplies ANIMALS, BLOCKS, MOBS, PLAYERS and WEATHER for the
    /// "Super Secret Settings..." button.  It must create
    /// `PlayRequest::ambient_with_pitch(event, 0.5)` after this returns `Some`, exactly as
    /// `PositionedSoundRecord.create(location, 0.5F)` does.
    // java: SoundHandler.getRandomSoundFromCategories:260-282
    pub fn random_event_from_categories(
        &self,
        categories: &[SoundCategory],
        rng: &mut JavaRandom,
    ) -> Option<ResourceLocation> {
        let matching: Vec<&ResourceLocation> = self
            .events
            .iter()
            .filter_map(|(event, definition)| {
                categories.contains(&definition.category).then_some(event)
            })
            .collect();
        let count = i32::try_from(matching.len()).ok()?;
        if count == 0 {
            return None;
        }
        let choice = rng.next_i32_bound(count) as usize;
        Some(matching[choice].clone())
    }

    /// Rolls a weighted pick against `event`'s pool, resolving `event`-type indirection
    /// recursively (a fresh independent weighted draw against the referenced event's own pool,
    /// exactly as `SoundEventAccessorComposite.cloneEntry` re-invokes the referenced composite).
    ///
    /// Returns `None` when the event isn't registered, its pool is empty, or every branch of an
    /// indirection chain bottoms out empty — the three cases vanilla folds into the single
    /// `SoundHandler.missing_sound` sentinel.
    // java: SoundEventAccessorComposite.getWeight / cloneEntry
    pub fn pick_variant(
        &self,
        event: &ResourceLocation,
        rng: &mut mc_math::JavaRandom,
    ) -> Option<ResolvedVariant> {
        let mut steps = MAX_INDIRECTION_STEPS;
        self.pick_inner(event, rng, 0, &mut steps)
            .map(|(variant, _)| variant)
    }

    /// Settles presumed files: every `File` variant whose file `fate` maps to `Some(true)` is
    /// known to exist from now on, every presumed one mapped to `Some(false)` leaves every pool
    /// it is in, exactly as `SoundHandler.loadSoundResource` would have skipped it at load. A
    /// file that was opened at load stays even if it is gone later, as in vanilla. Event weights
    /// are summed on demand, so the odds of the rest follow. One pass over the registry.
    pub(crate) fn settle_files(&mut self, mut fate: impl FnMut(&ResourceLocation) -> Option<bool>) {
        for def in self.events.values_mut() {
            def.variants.retain_mut(|variant| match &mut variant.kind {
                VariantKind::File {
                    location, presumed, ..
                } => match fate(location) {
                    Some(true) => {
                        *presumed = false;
                        true
                    }
                    Some(false) => !*presumed,
                    None => true,
                },
                VariantKind::Event { .. } => true,
            });
        }
    }

    /// Every non-streaming `.ogg` that an event selected by `keep` can play, following `event`
    /// indirection, each file once. Draws no random numbers and changes nothing; it exists so
    /// the decode cache can be warmed off the game thread (HITCH-3 / AUDIO-1).
    pub fn static_files_where(
        &self,
        keep: impl Fn(&ResourceLocation) -> bool,
    ) -> Vec<ResourceLocation> {
        let mut seen = std::collections::HashSet::new();
        let mut files = Vec::new();
        for event in self.events.keys().filter(|event| keep(event)) {
            let mut steps = MAX_INDIRECTION_STEPS;
            self.collect_static_files(event, 0, &mut steps, &mut seen, &mut files);
        }
        files
    }

    /// Every streamed (`"stream": true`) `.ogg` any event can play, each once: music and
    /// records. Lets the engine check their existence off the game thread.
    pub(crate) fn streamed_files(&self) -> Vec<ResourceLocation> {
        let mut seen = std::collections::HashSet::new();
        let mut files = Vec::new();
        for def in self.events.values() {
            for variant in &def.variants {
                if let VariantKind::File {
                    location,
                    streaming: true,
                    ..
                } = &variant.kind
                {
                    if seen.insert(location.clone()) {
                        files.push(location.clone());
                    }
                }
            }
        }
        files
    }

    fn collect_static_files(
        &self,
        event: &ResourceLocation,
        depth: u8,
        steps: &mut u32,
        seen: &mut std::collections::HashSet<ResourceLocation>,
        files: &mut Vec<ResourceLocation>,
    ) {
        if depth > MAX_INDIRECTION_DEPTH || *steps == 0 {
            return;
        }
        *steps -= 1;
        let Some(def) = self.events.get(event) else {
            return;
        };
        for variant in &def.variants {
            match &variant.kind {
                VariantKind::File {
                    location,
                    streaming: false,
                    ..
                } => {
                    if seen.insert(location.clone()) {
                        files.push(location.clone());
                    }
                }
                VariantKind::File { .. } => {}
                VariantKind::Event { location } => {
                    self.collect_static_files(location, depth + 1, steps, seen, files);
                }
            }
        }
    }

    fn total_weight(&self, event: &ResourceLocation, depth: u8, steps: &mut u32) -> i64 {
        if depth > MAX_INDIRECTION_DEPTH || *steps == 0 {
            return 0;
        }
        *steps -= 1;
        let Some(def) = self.events.get(event) else {
            return 0;
        };
        def.variants
            .iter()
            .map(|v| match &v.kind {
                VariantKind::File { .. } => v.weight as i64,
                VariantKind::Event { location } => self.total_weight(location, depth + 1, steps),
            })
            .sum()
    }

    fn pick_inner(
        &self,
        event: &ResourceLocation,
        rng: &mut mc_math::JavaRandom,
        depth: u8,
        steps: &mut u32,
    ) -> Option<(ResolvedVariant, bool)> {
        if depth > MAX_INDIRECTION_DEPTH {
            return None;
        }
        let def = self.events.get(event)?;
        let total = self.total_weight(event, depth, steps);
        if def.variants.is_empty() || total <= 0 {
            return None;
        }
        // java: `this.rnd.nextInt(i)` where `i` is the total weight (always >0 here).
        let mut roll = rng.next_i32_bound(total.min(i32::MAX as i64) as i32) as i64;
        for variant in &def.variants {
            let weight = match &variant.kind {
                VariantKind::File { .. } => variant.weight as i64,
                VariantKind::Event { location } => self.total_weight(location, depth + 1, steps),
            };
            roll -= weight;
            if roll < 0 {
                return match &variant.kind {
                    VariantKind::File {
                        location,
                        volume,
                        pitch,
                        streaming,
                        presumed,
                    } => Some((
                        ResolvedVariant {
                            location: location.clone(),
                            volume: *volume,
                            pitch: *pitch,
                            streaming: *streaming,
                        },
                        *presumed,
                    )),
                    VariantKind::Event { location } => {
                        self.pick_inner(location, rng, depth + 1, steps)
                    }
                };
            }
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawKind {
    File,
    Event,
}

#[derive(Debug, Clone)]
struct RawSoundEntry {
    name: String,
    volume: f32,
    pitch: f32,
    weight: u32,
    kind: RawKind,
    stream: bool,
}

#[derive(Debug, Clone)]
struct RawSoundList {
    category: String,
    replace: bool,
    sounds: Vec<RawSoundEntry>,
}

/// Parses one `sounds.json` body into its event → raw-list map.
///
/// This walks `serde_json::Value` by hand rather than deriving `Deserialize`, mirroring
/// `SoundListSerializer`'s manual `JsonElement` inspection: the `"sounds"` array is
/// heterogeneous (a bare string, or an object with `name`/`volume`/`pitch`/`weight`/`type`/
/// `stream`), which an ordinary `#[derive(Deserialize)]` enum handles awkwardly and less
/// leniently than an ordinary derived enum. Invalid input is still returned as an error rather
/// than panicking: java's `SoundListSerializer` lets every `JsonUtils`/`Validate` failure escape
/// `getSoundMap`, and `SoundHandler` catches it around exactly one pack's `sounds.json`. There is
/// no per-entry catch in that loop, so one invalid entry rejects that file atomically.
fn parse_sounds_json(
    bytes: &[u8],
    location: &str,
) -> Result<Vec<(String, RawSoundList)>, AudioError> {
    let root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| AudioError::InvalidSoundList {
            location: location.to_owned(),
            reason: format!("not valid JSON: {e}"),
        })?;
    let Some(obj) = root.as_object() else {
        return Err(AudioError::InvalidSoundList {
            location: location.to_owned(),
            reason: "root is not a JSON object".to_owned(),
        });
    };

    let mut out = Vec::with_capacity(obj.len());
    for (event_name, value) in obj {
        let entry_obj = value.as_object().ok_or_else(|| {
            invalid_sound_list(location, format!("event {event_name:?} is not an object"))
        })?;
        let replace = match entry_obj.get("replace") {
            None => false,
            Some(value) => gson_primitive_boolean(value).ok_or_else(|| {
                invalid_sound_list(
                    location,
                    format!("event {event_name:?} has a non-primitive replace"),
                )
            })?,
        };
        let category = match entry_obj.get("category") {
            None => SoundCategory::Master.category_name().to_owned(),
            Some(value) => gson_primitive_string(value).ok_or_else(|| {
                invalid_sound_list(
                    location,
                    format!("event {event_name:?} has a non-primitive category"),
                )
            })?,
        };

        let mut sounds = Vec::new();
        if let Some(value) = entry_obj.get("sounds") {
            let arr = value.as_array().ok_or_else(|| {
                invalid_sound_list(
                    location,
                    format!("event {event_name:?} has a non-array sounds field"),
                )
            })?;
            for sound in arr {
                sounds.push(parse_sound_entry(sound, location, event_name)?);
            }
        }

        out.push((
            event_name.clone(),
            RawSoundList {
                category,
                replace,
                sounds,
            },
        ));
    }
    Ok(out)
}

fn invalid_sound_list(location: &str, reason: String) -> AudioError {
    AudioError::InvalidSoundList {
        location: location.to_owned(),
        reason,
    }
}

fn parse_sound_entry(
    value: &serde_json::Value,
    location: &str,
    event_name: &str,
) -> Result<RawSoundEntry, AudioError> {
    if let Some(name) = value.as_str() {
        return Ok(RawSoundEntry {
            name: name.to_owned(),
            volume: 1.0,
            pitch: 1.0,
            weight: 1,
            kind: RawKind::File,
            stream: false,
        });
    }

    let invalid = |reason: String| {
        invalid_sound_list(
            location,
            format!("event {event_name:?} has an invalid sound entry: {reason}"),
        )
    };
    let obj = value
        .as_object()
        .ok_or_else(|| invalid("entry is neither a string nor an object".to_owned()))?;
    let name = obj
        .get("name")
        .and_then(gson_primitive_string)
        .ok_or_else(|| invalid("missing or non-primitive name".to_owned()))?;
    let kind = match obj.get("type") {
        None => RawKind::File,
        Some(value) => match gson_primitive_string(value).as_deref() {
            Some("file") => RawKind::File,
            Some("event") => RawKind::Event,
            Some(other) => return Err(invalid(format!("unknown type {other:?}"))),
            None => return Err(invalid("non-primitive type".to_owned())),
        },
    };
    let volume = positive_f32(obj.get("volume"), "volume", &invalid)?;
    let pitch = positive_f32(obj.get("pitch"), "pitch", &invalid)?;
    let weight = match obj.get("weight") {
        None => 1,
        Some(value) => {
            let weight = value
                .as_number()
                .and_then(java_int_value)
                .ok_or_else(|| invalid("weight is not a Java int".to_owned()))?;
            if weight <= 0 {
                return Err(invalid(format!("weight must be positive, got {weight}")));
            }
            weight as u32
        }
    };
    let stream = match obj.get("stream") {
        None => false,
        Some(value) => gson_primitive_boolean(value)
            .ok_or_else(|| invalid("stream is not a primitive".to_owned()))?,
    };

    Ok(RawSoundEntry {
        name,
        volume,
        pitch,
        weight,
        kind,
        stream,
    })
}

/// `JsonPrimitive.getAsString()`: Gson accepts strings, numbers and booleans for fields that
/// `SoundListSerializer` reads through `JsonUtils.getString`.  Serde's typed accessors are
/// intentionally stricter, so keep this coercion local to the Java-compatible sound parser.
fn gson_primitive_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(value) => Some(value.clone()),
        serde_json::Value::Number(value) => Some(value.to_string()),
        serde_json::Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

/// `JsonPrimitive.getAsBoolean()`: a real boolean is returned directly; every other primitive
/// is passed through Java's case-insensitive `Boolean.parseBoolean`, which is true only for the
/// string `"true"`.
fn gson_primitive_boolean(value: &serde_json::Value) -> Option<bool> {
    Some(match value {
        serde_json::Value::Bool(value) => *value,
        _ => gson_primitive_string(value)?.eq_ignore_ascii_case("true"),
    })
}

/// Converts a JSON number exactly as Gson 2.2.4's `JsonPrimitive.getAsInt()` does for the
/// `SoundListSerializer` weight field.  `LazilyParsedNumber.intValue()` tries `int`, then
/// `long` (both narrowing on overflow), and finally `BigDecimal.intValue()` (truncate toward
/// zero, then keep the low 32 bits).  Parsing the lexical number here avoids losing that Java
/// behavior to an intermediate `f64`, including decimal weights such as `3.75`.
fn java_int_value(number: &serde_json::Number) -> Option<i32> {
    let text = number.to_string();
    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(value) => (true, value),
        None => (false, text.as_str()),
    };

    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(index) => {
            let exponent = unsigned[index + 1..].parse::<i64>().ok()?;
            (&unsigned[..index], exponent)
        }
        None => (unsigned, 0),
    };

    let mut digits = String::with_capacity(mantissa.len());
    let mut fractional_digits = 0_i64;
    let mut after_decimal = false;
    for byte in mantissa.bytes() {
        match byte {
            b'.' => after_decimal = true,
            b'0'..=b'9' => {
                digits.push(byte as char);
                if after_decimal {
                    fractional_digits = fractional_digits.checked_add(1)?;
                }
            }
            _ => return None,
        }
    }
    if digits.is_empty() {
        return None;
    }

    // `scale` is the number of decimal zeroes to append (positive) or decimal digits to drop
    // (negative) after truncating toward zero.
    let scale = exponent.checked_sub(fractional_digits)?;
    let kept_len = if scale >= 0 {
        digits.len()
    } else {
        let drop = scale
            .checked_neg()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(usize::MAX);
        digits.len().saturating_sub(drop)
    };

    let mut magnitude = 0_u32;
    for byte in digits[..kept_len].bytes() {
        magnitude = magnitude
            .wrapping_mul(10)
            .wrapping_add(u32::from(byte - b'0'));
    }

    // 10^32 is divisible by 2^32, so larger positive scales are already zero modulo the
    // low-32-bit narrowing performed by Java's BigDecimal.intValue().
    if scale > 0 {
        let zeroes = u64::try_from(scale).ok()?.min(32);
        for _ in 0..zeroes {
            magnitude = magnitude.wrapping_mul(10);
        }
    }
    if negative {
        magnitude = magnitude.wrapping_neg();
    }
    Some(magnitude as i32)
}

/// `SoundListSerializer` reads a Java `float` and then `Validate`s it as strictly positive.
/// Absent fields default to `1.0`; present invalid fields reject the containing file.
fn positive_f32(
    value: Option<&serde_json::Value>,
    field: &str,
    invalid: &impl Fn(String) -> AudioError,
) -> Result<f32, AudioError> {
    let Some(value) = value else {
        return Ok(1.0);
    };
    let parsed = value
        .as_f64()
        .map(|number| number as f32)
        .ok_or_else(|| invalid(format!("{field} is not a number")))?;
    if parsed <= 0.0 || parsed.is_nan() {
        return Err(invalid(format!("{field} must be positive, got {parsed}")));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap as StdHashMap;
    use std::io::{Cursor, Read};

    use mc_assets::ResourcePack;

    use super::*;

    /// A tiny in-memory pack, so tests don't touch the disk.
    struct MemPack {
        namespace: String,
        files: StdHashMap<String, Vec<u8>>,
    }

    impl MemPack {
        fn boxed(namespace: &str, files: &[(&str, &str)]) -> Box<dyn ResourcePack> {
            Box::new(Self {
                namespace: namespace.to_owned(),
                files: files
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), v.as_bytes().to_vec()))
                    .collect(),
            })
        }
    }

    impl ResourcePack for MemPack {
        fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
            self.files
                .get(&l.pack_entry_name())
                .cloned()
                .map(|v| Box::new(Cursor::new(v)) as Box<dyn Read>)
        }
        fn contains(&self, l: &ResourceLocation) -> bool {
            self.files.contains_key(&l.pack_entry_name())
        }
        fn namespaces(&self) -> Vec<String> {
            vec![self.namespace.clone()]
        }
        fn name(&self) -> &str {
            "mem"
        }
    }

    /// A pack that claims to contain every `sounds/**.ogg` and nothing else.
    ///
    /// `SoundRegistry::load` now checks each `FILE` variant's `.ogg` against the manager before
    /// adding it, because `SoundHandler.loadSoundResource` opens it at load time and `continue`s
    /// on `FileNotFoundException` (`F-P2-client.audio-02-6`). Every test below is about the
    /// *registry*, not about missing assets, so they run in the state the real 1.8.9 asset set is
    /// in: all 615 referenced files present. `missing_ogg_variants_are_dropped_from_the_pool`
    /// is the one test that deliberately omits this pack.
    struct AllOggsPack;

    impl ResourcePack for AllOggsPack {
        fn open(&self, _l: &ResourceLocation) -> Option<Box<dyn Read>> {
            Some(Box::new(Cursor::new(Vec::new())) as Box<dyn Read>)
        }
        fn contains(&self, l: &ResourceLocation) -> bool {
            l.path.starts_with("sounds/") && l.path.ends_with(".ogg")
        }
        fn namespaces(&self) -> Vec<String> {
            Vec::new()
        }
        fn name(&self) -> &str {
            "all-oggs"
        }
    }

    /// A pack with an index entry whose underlying reader is unavailable.  This is distinct from
    /// a missing resource: Java's loader catches an `IOException` from `getInputStream()` just as
    /// it catches `FileNotFoundException`, while a `contains`-only probe cannot tell the two
    /// states apart.
    struct UnreadableOggPack;

    impl ResourcePack for UnreadableOggPack {
        fn open(&self, _l: &ResourceLocation) -> Option<Box<dyn Read>> {
            None
        }

        fn contains(&self, l: &ResourceLocation) -> bool {
            l.namespace == "minecraft" && l.path == "sounds/probe/unreadable.ogg"
        }

        fn namespaces(&self) -> Vec<String> {
            vec!["minecraft".to_owned()]
        }

        fn name(&self) -> &str {
            "unreadable-ogg"
        }
    }

    fn manager_with(files: &[(&str, &str)]) -> ResourceManager {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(AllOggsPack));
        rm.push_pack(MemPack::boxed("minecraft", files));
        rm
    }

    #[test]
    fn simple_string_variants_default_to_uniform_weight() {
        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
        )]);
        let reg = SoundRegistry::load(&rm).unwrap();
        let event = ResourceLocation::minecraft("random.click");
        assert_eq!(reg.category_of(&event), Some(SoundCategory::Players));

        let mut rng = mc_math::JavaRandom::new(1);
        let picked = reg.pick_variant(&event, &mut rng).unwrap();
        assert_eq!(
            picked.location.to_string(),
            "minecraft:sounds/random/click.ogg"
        );
        assert_eq!(picked.volume, 1.0);
        assert_eq!(picked.pitch, 1.0);
        assert!(!picked.streaming);
    }

    #[test]
    fn object_variants_carry_stream_and_overrides() {
        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{"records.cat": {"category": "record",
                "sounds": [{"name": "records/cat", "stream": true}]}}"#,
        )]);
        let reg = SoundRegistry::load(&rm).unwrap();
        let event = ResourceLocation::minecraft("records.cat");
        let mut rng = mc_math::JavaRandom::new(1);
        let picked = reg.pick_variant(&event, &mut rng).unwrap();
        assert_eq!(
            picked.location.to_string(),
            "minecraft:sounds/records/cat.ogg"
        );
        assert!(picked.streaming);
        assert_eq!(reg.category_of(&event), Some(SoundCategory::Records));
    }

    #[test]
    fn unknown_event_has_no_category_and_no_variant() {
        let rm = manager_with(&[]);
        let reg = SoundRegistry::load(&rm).unwrap();
        let event = ResourceLocation::minecraft("nope.not.a.sound");
        assert_eq!(reg.category_of(&event), None);
        let mut rng = mc_math::JavaRandom::new(1);
        assert!(reg.pick_variant(&event, &mut rng).is_none());
    }

    #[test]
    fn random_event_from_categories_matches_only_requested_categories() {
        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{
                "animal": {"category": "neutral", "sounds": ["animal"]},
                "block": {"category": "block", "sounds": ["block"]},
                "mob": {"category": "hostile", "sounds": ["mob"]},
                "player": {"category": "player", "sounds": ["player"]},
                "weather": {"category": "weather", "sounds": ["weather"]},
                "music": {"category": "music", "sounds": ["music"]},
                "record": {"category": "record", "sounds": ["record"]}
            }"#,
        )]);
        let reg = SoundRegistry::load(&rm).unwrap();
        let requested = [
            SoundCategory::Animals,
            SoundCategory::Blocks,
            SoundCategory::Mobs,
            SoundCategory::Players,
            SoundCategory::Weather,
        ];

        let mut chosen = std::collections::HashSet::new();
        for seed in 0..128 {
            let mut rng = JavaRandom::new(seed);
            let event = reg
                .random_event_from_categories(&requested, &mut rng)
                .expect("the requested categories have five events");
            assert!(requested.contains(&reg.category_of(&event).unwrap()));
            chosen.insert(event);
        }
        assert_eq!(
            chosen.len(),
            5,
            "nextInt(count) must reach every eligible event"
        );
    }

    #[test]
    fn random_category_picker_neither_weights_duplicate_categories_nor_draws_on_empty() {
        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{
                "block": {"category": "block", "sounds": ["block"]},
                "music": {"category": "music", "sounds": ["music"]}
            }"#,
        )]);
        let reg = SoundRegistry::load(&rm).unwrap();

        let mut once_rng = JavaRandom::new(0x1234);
        let once = reg
            .random_event_from_categories(&[SoundCategory::Blocks], &mut once_rng)
            .unwrap();
        let mut duplicate_rng = JavaRandom::new(0x1234);
        let duplicate = reg
            .random_event_from_categories(
                &[SoundCategory::Blocks, SoundCategory::Blocks],
                &mut duplicate_rng,
            )
            .unwrap();
        assert_eq!(once, duplicate);
        assert_eq!(once_rng.raw_seed(), duplicate_rng.raw_seed());

        let mut empty_rng = JavaRandom::new(0x5678);
        let before = empty_rng.raw_seed();
        assert_eq!(
            reg.random_event_from_categories(&[SoundCategory::Weather], &mut empty_rng),
            None
        );
        assert_eq!(
            empty_rng.raw_seed(),
            before,
            "an empty Java list does not call nextInt"
        );
    }

    #[test]
    fn event_indirection_resolves_to_a_real_file() {
        // Mirrors music.game.creative: an "event" entry pointing at music.game, plus its own
        // files, all in one weighted pool.
        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{
                "music.game": {"category": "music", "sounds": ["music/game/calm1", "music/game/calm2"]},
                "music.game.creative": {"category": "music", "sounds": [
                    {"type": "event", "name": "music.game"},
                    {"name": "music/game/creative/creative1", "stream": true}
                ]}
            }"#,
        )]);
        let reg = SoundRegistry::load(&rm).unwrap();
        let event = ResourceLocation::minecraft("music.game.creative");

        // Every roll must resolve to a real file, never None, across many seeds.
        let mut saw_calm = false;
        let mut saw_creative = false;
        for seed in 0..200 {
            let mut rng = mc_math::JavaRandom::new(seed);
            let picked = reg.pick_variant(&event, &mut rng).expect("must resolve");
            let path = picked.location.path;
            assert!(path.starts_with("sounds/music/game/"), "{path}");
            if path.contains("creative") {
                saw_creative = true;
            } else {
                saw_calm = true;
            }
        }
        assert!(saw_calm, "event indirection into music.game never chosen");
        assert!(saw_creative, "own file variant never chosen");
    }

    #[test]
    fn indirection_cycle_is_bounded_not_infinite() {
        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{
                "a": {"category": "master", "sounds": [{"type": "event", "name": "b"}]},
                "b": {"category": "master", "sounds": [{"type": "event", "name": "a"}]}
            }"#,
        )]);
        let reg = SoundRegistry::load(&rm).unwrap();
        let event = ResourceLocation::minecraft("a");
        let mut rng = mc_math::JavaRandom::new(1);
        // Must terminate (the test itself times out otherwise) and settle on "no variant".
        assert!(reg.pick_variant(&event, &mut rng).is_none());
    }

    #[test]
    fn replace_discards_earlier_packs_entries() {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(AllOggsPack));
        rm.push_pack(MemPack::boxed(
            "minecraft",
            &[(
                "assets/minecraft/sounds.json",
                r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
            )],
        ));
        rm.push_pack(MemPack::boxed(
            "minecraft",
            &[(
                "assets/minecraft/sounds.json",
                r#"{"random.click": {"category": "player", "replace": true, "sounds": ["random/pack_click"]}}"#,
            )],
        ));
        let reg = SoundRegistry::load(&rm).unwrap();
        let event = ResourceLocation::minecraft("random.click");
        let mut rng = mc_math::JavaRandom::new(1);
        let picked = reg.pick_variant(&event, &mut rng).unwrap();
        assert_eq!(picked.location.path, "sounds/random/pack_click.ogg");
    }

    #[test]
    fn non_replacing_pack_appends_to_the_existing_pool() {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(AllOggsPack));
        rm.push_pack(MemPack::boxed(
            "minecraft",
            &[(
                "assets/minecraft/sounds.json",
                r#"{"random.click": {"category": "player", "sounds": ["random/click"]}}"#,
            )],
        ));
        rm.push_pack(MemPack::boxed(
            "minecraft",
            &[(
                "assets/minecraft/sounds.json",
                r#"{"random.click": {"category": "player", "sounds": ["random/extra_click"]}}"#,
            )],
        ));
        let reg = SoundRegistry::load(&rm).unwrap();
        let event = ResourceLocation::minecraft("random.click");

        // One evolving RNG across every draw, not a fresh `JavaRandom` per seed: vanilla's
        // `SoundEventAccessorComposite` reuses a single `Random rnd` field across every
        // `cloneEntry()` call, and this matters here beyond just fidelity — a *fresh* LCG state
        // reseeded from small consecutive integers is a documented degenerate case for a
        // power-of-two bound specifically (`nextInt(2)` reads the LCG's top bit, and seeds that
        // differ only in their low bits barely perturb that top bit after a single step), so
        // reseeding per draw would make this test flaky by construction rather than by bug.
        let mut rng = mc_math::JavaRandom::new(1);
        let mut saw = std::collections::HashSet::new();
        for _ in 0..50 {
            saw.insert(reg.pick_variant(&event, &mut rng).unwrap().location.path);
        }
        assert!(saw.contains("sounds/random/click.ogg"));
        assert!(saw.contains("sounds/random/extra_click.ogg"));
    }

    /// `F-P2-client.audio-02-5`: an invalid category used to propagate out of
    /// `SoundRegistry::load`, and `GameAssets` turns any `Err` into `SoundRegistry::empty()` — so
    /// one typo in one installed pack silenced the entire game.
    ///
    /// java: `SoundListSerializer:158` `Validate`s the category while GSON is still building the
    /// map, inside `SoundHandler`'s per-`IResource` `try`, so the throw costs exactly that file:
    /// the bad event *and* its file-mates are lost, and every other pack is untouched. All three
    /// halves are asserted, which is what stops this passing for a `load` that ignores categories
    /// outright (`bad.one` would then be registered) or for one that still aborts everything
    /// (`good.one`, in another pack, would be gone).
    #[test]
    fn an_invalid_category_costs_that_file_and_nothing_else() {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(AllOggsPack));
        rm.push_pack(MemPack::boxed(
            "minecraft",
            &[(
                "assets/minecraft/sounds.json",
                r#"{"good.one": {"category": "player", "sounds": ["random/click"]}}"#,
            )],
        ));
        rm.push_pack(MemPack::boxed(
            "minecraft",
            &[(
                "assets/minecraft/sounds.json",
                r#"{"bad.one": {"category": "not_a_real_category", "sounds": ["x"]},
                    "collateral.one": {"category": "player", "sounds": ["random/click"]}}"#,
            )],
        ));
        let reg = SoundRegistry::load(&rm).expect("one bad pack must not fail the whole load");
        assert!(
            reg.contains(&ResourceLocation::minecraft("good.one")),
            "a clean pack must survive a later pack's typo"
        );
        assert!(!reg.contains(&ResourceLocation::minecraft("bad.one")));
        assert!(
            !reg.contains(&ResourceLocation::minecraft("collateral.one")),
            "vanilla throws inside getSoundMap, so the bad file's other events go too"
        );
    }

    /// `F-P2-client.audio-02-6`: java opens each `FILE` variant's `.ogg` at load time and
    /// `continue`s on `FileNotFoundException`, so a missing file never enters the weighted pool.
    /// Keeping it both skewed the draw and produced silence whenever it won.
    ///
    /// The negative control is the second half: with the file present, the same pool does hand
    /// it out, so this cannot pass by `merge_event` having stopped adding variants.
    #[test]
    fn missing_ogg_variants_are_dropped_from_the_pool() {
        let json = r#"{"e": {"category": "master", "sounds": ["probe/exists", "probe/absent"]}}"#;

        let mut rm = ResourceManager::new();
        rm.push_pack(MemPack::boxed(
            "minecraft",
            &[
                ("assets/minecraft/sounds.json", json),
                ("assets/minecraft/sounds/probe/exists.ogg", ""),
            ],
        ));
        let reg = SoundRegistry::load(&rm).unwrap();
        let event = ResourceLocation::minecraft("e");
        let mut rng = mc_math::JavaRandom::new(1);
        for _ in 0..200 {
            let picked = reg.pick_variant(&event, &mut rng).unwrap();
            assert_eq!(
                picked.location.path, "sounds/probe/exists.ogg",
                "the absent variant must never be drawn"
            );
        }

        let mut both = ResourceManager::new();
        both.push_pack(Box::new(AllOggsPack));
        both.push_pack(MemPack::boxed(
            "minecraft",
            &[("assets/minecraft/sounds.json", json)],
        ));
        let reg = SoundRegistry::load(&both).unwrap();
        let mut rng = mc_math::JavaRandom::new(1);
        let mut saw = std::collections::HashSet::new();
        for _ in 0..200 {
            saw.insert(reg.pick_variant(&event, &mut rng).unwrap().location.path);
        }
        assert_eq!(saw.len(), 2, "with both files present, both are drawn");
    }

    #[test]
    fn unreadable_ogg_variants_are_dropped_even_when_the_pack_claims_them() {
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(UnreadableOggPack));
        rm.push_pack(MemPack::boxed(
            "minecraft",
            &[
                (
                    "assets/minecraft/sounds.json",
                    r#"{"e":{"category":"master","sounds":["probe/available","probe/unreadable"]}}"#,
                ),
                ("assets/minecraft/sounds/probe/available.ogg", "bytes"),
            ],
        ));

        let reg = SoundRegistry::load(&rm).unwrap();
        let event = ResourceLocation::minecraft("e");
        let mut rng = JavaRandom::new(1);
        for _ in 0..200 {
            let picked = reg.pick_variant(&event, &mut rng).unwrap();
            assert_eq!(
                picked.location.path, "sounds/probe/available.ogg",
                "an I/O failure must be filtered before weighted selection"
            );
        }
    }

    #[test]
    fn sound_entry_locations_keep_java_domain_and_event_path_quirks() {
        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{
                "a:b": {"category":"master","sounds":["probe/file"]},
                "file_probe": {"category":"master","sounds":[{"name":"a:b"}]},
                "event_probe": {"category":"master","sounds":[{"name":"a:b","type":"event"}]}
            }"#,
        )]);

        let reg = SoundRegistry::load(&rm).unwrap();
        let mut rng = JavaRandom::new(1);

        // `new ResourceLocation("a:b")` treats a one-character prefix as part of the path,
        // so the FILE branch resolves minecraft:sounds/b.  A naive split_once implementation
        // incorrectly manufactures the namespace `a`.
        let file = reg
            .pick_variant(&ResourceLocation::minecraft("file_probe"), &mut rng)
            .unwrap();
        assert_eq!(file.location, ResourceLocation::minecraft("sounds/b.ogg"));

        // The SOUND_EVENT branch passes the raw name as the path, making it target the literal
        // registration key minecraft:a:b.  Stripping the colon here would resolve b instead and
        // silently turn this valid Java lookup into an empty event.
        let event = reg
            .pick_variant(&ResourceLocation::minecraft("event_probe"), &mut rng)
            .unwrap();
        assert_eq!(
            event.location,
            ResourceLocation::minecraft("sounds/probe/file.ogg")
        );
    }

    #[test]
    fn serializer_uses_gson_primitive_coercions_and_java_int_narrowing() {
        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{
                "coerced": {"category": true, "replace": "true", "sounds": [
                    {"name": 42, "volume": 1.25, "pitch": 0.75, "weight": 3.75, "stream": 1}
                ]}
            }"#,
        )]);

        // The category still invalidates the file: `JsonUtils.getString` converts the boolean
        // to "true", then `Validate.notNull` rejects that category.  This control would pass
        // under a parser that silently defaults/partially registers malformed files.
        assert!(!SoundRegistry::load(&rm)
            .unwrap()
            .contains(&ResourceLocation::minecraft("coerced")));

        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{
                "coerced": {"category": "master", "replace": "true", "sounds": [
                    {"name": 42, "volume": 1.25, "pitch": 0.75, "weight": 3.75, "stream": 1}
                ]}
            }"#,
        )]);
        let reg = SoundRegistry::load(&rm).unwrap();
        let mut rng = JavaRandom::new(1);
        let variant = reg
            .pick_variant(&ResourceLocation::minecraft("coerced"), &mut rng)
            .expect("Gson primitive fields should deserialize");
        assert_eq!(
            variant.location,
            ResourceLocation::minecraft("sounds/42.ogg")
        );
        assert_eq!(variant.volume, 1.25);
        assert_eq!(variant.pitch, 0.75);
        assert!(
            !variant.streaming,
            "numeric stream=1 parses as Boolean.parseBoolean(\"1\")"
        );

        // Gson's LazyParsedNumber.intValue truncates a decimal through BigDecimal before
        // SoundListSerializer validates it.  A strict integer-only parser rejects 3.75, while
        // Java stores the resulting positive weight as 3.
        assert_eq!(
            java_int_value(&serde_json::from_str("3.75").unwrap()),
            Some(3)
        );
        assert_eq!(
            java_int_value(&serde_json::from_str("-3.75").unwrap()),
            Some(-3)
        );
        assert_eq!(
            java_int_value(&serde_json::from_str("4294967297").unwrap()),
            Some(1)
        );
    }

    /// `F-P2-client.audio-02-9`: all four explicit `Validate` calls in
    /// `SoundListSerializer` escape `getSoundMap`. `SoundHandler` catches them outside the map
    /// iteration, so the containing file is atomic: neither the invalid event nor a valid
    /// file-mate is registered, while events from an earlier pack remain intact.
    ///
    /// Each case would fail if the bad value were silently replaced by its default (the old
    /// behavior), or if only the bad entry/event were skipped. The clean control below proves
    /// that the same fields are accepted when valid and that their non-default values survive.
    #[test]
    fn invalid_serializer_fields_reject_the_containing_file_atomically() {
        let invalid_entries = [
            r#"{"name":"probe/bad","type":"nonsense"}"#,
            r#"{"name":"probe/bad","volume":0.0}"#,
            r#"{"name":"probe/bad","pitch":-1.0}"#,
            r#"{"name":"probe/bad","weight":0}"#,
        ];

        for bad_entry in invalid_entries {
            let invalid_file = format!(
                r#"{{
                    "bad.one": {{"category":"master","sounds":[{bad_entry}]}},
                    "collateral.one": {{"category":"player","sounds":["probe/good"]}}
                }}"#
            );
            let mut rm = ResourceManager::new();
            rm.push_pack(Box::new(AllOggsPack));
            rm.push_pack(MemPack::boxed(
                "minecraft",
                &[(
                    "assets/minecraft/sounds.json",
                    r#"{"base.one":{"category":"player","sounds":["probe/base"]}}"#,
                )],
            ));
            rm.push_pack(MemPack::boxed(
                "minecraft",
                &[("assets/minecraft/sounds.json", &invalid_file)],
            ));

            let reg = SoundRegistry::load(&rm).expect("a bad file must not abort the reload");
            assert!(
                reg.contains(&ResourceLocation::minecraft("base.one")),
                "an earlier clean pack was lost for {bad_entry}"
            );
            assert!(
                !reg.contains(&ResourceLocation::minecraft("bad.one")),
                "invalid event was registered for {bad_entry}"
            );
            assert!(
                !reg.contains(&ResourceLocation::minecraft("collateral.one")),
                "the invalid file was partially registered for {bad_entry}"
            );
        }

        let rm = manager_with(&[(
            "assets/minecraft/sounds.json",
            r#"{"valid.one":{"category":"master","sounds":[{
                "name":"probe/valid","type":"file","volume":0.25,"pitch":2.0,"weight":3
            }]}}"#,
        )]);
        let reg = SoundRegistry::load(&rm).unwrap();
        let mut rng = mc_math::JavaRandom::new(1);
        let picked = reg
            .pick_variant(&ResourceLocation::minecraft("valid.one"), &mut rng)
            .expect("valid non-default fields must register");
        assert_eq!(picked.volume, 0.25);
        assert_eq!(picked.pitch, 2.0);
        assert_eq!(picked.location.path, "sounds/probe/valid.ogg");
    }

    /// The serializer has no per-entry recovery: `JsonUtils.getJsonObject`/`getString` failures
    /// take the same per-file catch as the explicit validation failures above. This was an
    /// adjacent audit miss in the same validation seam.
    #[test]
    fn malformed_entries_reject_the_file_instead_of_being_skipped() {
        for malformed in ["42", r#"{"no_name_field":true}"#] {
            let json =
                format!(r#"{{"e":{{"category":"master","sounds":[{malformed},"ok/one"]}}}}"#);
            let rm = manager_with(&[("assets/minecraft/sounds.json", &json)]);
            let reg = SoundRegistry::load(&rm).unwrap();
            assert!(
                !reg.contains(&ResourceLocation::minecraft("e")),
                "malformed entry {malformed} must reject its file"
            );
        }
    }
}
