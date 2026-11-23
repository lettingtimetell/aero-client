//! Custom Sky: the custom sky layers of the active resource packs. This module finds and parses the layer files; `mc_render::sky::custom`
//! decodes their textures and draws them.
//!
//! # Files
//!
//! A pack has one layer per file, `assets/minecraft/<root>/sky/world<W>/sky<N>.properties`:
//! `<root>` is `optifine` (newer packs) or `mcpatcher` (older ones), `W` is the dimension (`0`
//! the overworld, `1` the End; vanilla draws no sky in the Nether, so there is no `world-1`),
//! and `N = 1, 2, ...` is the draw order. Every file is read through the whole pack stack, so
//! layer `N` comes from the top pack that has `sky<N>.properties`. A number present under
//! `optifine/` wins over the same number under `mcpatcher/`.
//!
//! # Properties
//!
//! The format's documented semantics, written from its documentation and published behaviour,
//! not from any implementation's code:
//!
//! | key | meaning | default |
//! |---|---|---|
//! | `source` | the texture: `./x` is next to the file, `~/x` under the root, `ns:x`, or a path in `assets/minecraft` | `sky<N>.png` next to the file |
//! | `startFadeIn`, `endFadeIn`, `startFadeOut`, `endFadeOut` | `hh:mm`; brightness rises, holds, falls, stays off | always on |
//! | `blend` | `add`, `subtract`, `multiply`, `dodge`, `burn`, `screen`, `replace`, `overlay`, `alpha` | `add` |
//! | `rotate`, `speed`, `axis` | turn with the time of day, `speed` turns per day, about `axis` (world `x y z`) | `true`, `1`, `0 0 1` |
//! | `weather` | `clear`, `rain`, `thunder` | `clear` (so a layer fades out in rain) |
//! | `biomes`, `heights` | where the camera must be | anywhere |
//! | `transition` | seconds to fade in/out when the camera crosses a biome/height border | `1` |
//!
//! `startFadeOut` may be left out: it is then `endFadeOut - (endFadeIn - startFadeIn)`. The four
//! phases must cover the day exactly once, or the layer is invalid. Without any fade time the
//! layer is always at full brightness. Broken layers (a missing texture, bad times) are skipped
//! and reported once in [`CustomSkyFiles::skipped`], never every frame.

use std::collections::{BTreeSet, HashMap};

use crate::{ResourceLocation, ResourceManager};

/// Ticks in a Minecraft day.
pub const TICKS_PER_DAY: i32 = 24_000;

/// The dimensions a pack can give a custom sky: the overworld (`world0`) and the End (`world1`).
pub const SKY_WORLDS: [i32; 2] = [0, 1];

/// The two folders packs use, in precedence order: the newer one, then the older one.
const SKY_ROOTS: [&str; 2] = ["optifine/", "mcpatcher/"];

/// The layer numbers the format allows: `sky1` .. `sky999`.
const MAX_LAYER_NUMBER: u32 = 999;

/// How a layer is combined with what is already drawn: the vanilla sky, then earlier layers.
///
/// The GL blend factors each mode stands for are the format's published blend table;
/// `mc_render::sky::custom` maps them onto wgpu blend states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkyBlend {
    /// `alpha`: weighted by the texture's alpha (`SRC_ALPHA, ONE_MINUS_SRC_ALPHA`).
    Alpha,
    /// `add`, the default: adds the texture, scaled by its alpha (`SRC_ALPHA, ONE`).
    Add,
    /// `subtract` (`ONE_MINUS_DST_COLOR, ZERO`).
    Subtract,
    /// `multiply` (`DST_COLOR, ONE_MINUS_SRC_ALPHA`).
    Multiply,
    /// `dodge` (`ONE, ONE`).
    Dodge,
    /// `burn` (`ZERO, ONE_MINUS_SRC_COLOR`).
    Burn,
    /// `screen` (`ONE, ONE_MINUS_SRC_COLOR`).
    Screen,
    /// `overlay` (`DST_COLOR, SRC_COLOR`).
    Overlay,
    /// `replace`: no blending; every texel that passes vanilla's alpha test overwrites the sky.
    Replace,
}

impl SkyBlend {
    /// Every mode, in [`SkyBlend::index`] order.
    pub const ALL: [SkyBlend; 9] = [
        SkyBlend::Alpha,
        SkyBlend::Add,
        SkyBlend::Subtract,
        SkyBlend::Multiply,
        SkyBlend::Dodge,
        SkyBlend::Burn,
        SkyBlend::Screen,
        SkyBlend::Overlay,
        SkyBlend::Replace,
    ];

    /// Position in [`SkyBlend::ALL`].
    #[must_use]
    pub fn index(self) -> usize {
        self as usize
    }

    /// A `blend=` value: trimmed, any case. The older aliases `color` (overlay) and `none`
    /// (replace) are accepted too. `None` for anything else.
    #[must_use]
    pub fn parse(value: &str) -> Option<SkyBlend> {
        Some(match value.trim().to_ascii_lowercase().as_str() {
            "alpha" => SkyBlend::Alpha,
            "add" => SkyBlend::Add,
            "subtract" => SkyBlend::Subtract,
            "multiply" => SkyBlend::Multiply,
            "dodge" => SkyBlend::Dodge,
            "burn" => SkyBlend::Burn,
            "screen" => SkyBlend::Screen,
            "overlay" | "color" => SkyBlend::Overlay,
            "replace" | "none" => SkyBlend::Replace,
            _ => return None,
        })
    }
}

/// A layer's fade times, in ticks after 06:00, i.e. `/time set` units: 06:00 is `0`, noon
/// `6000`, 18:00 `12000`, midnight `18000`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkyFade {
    /// Brightness starts rising from 0.
    pub start_fade_in: i32,
    /// Brightness reaches 1.
    pub end_fade_in: i32,
    /// Brightness starts falling from 1.
    pub start_fade_out: i32,
    /// Brightness reaches 0, and stays there until [`SkyFade::start_fade_in`].
    pub end_fade_out: i32,
}

/// `t` folded into `0..TICKS_PER_DAY`.
fn day_ticks(t: i32) -> i32 {
    t.rem_euclid(TICKS_PER_DAY)
}

/// Whether `t` lies in the inclusive window `start..=end`, which may wrap past midnight.
fn in_window(t: i32, start: i32, end: i32) -> bool {
    if start <= end {
        start <= t && t <= end
    } else {
        t >= start || t <= end
    }
}

impl SkyFade {
    /// Build and check fade times. A missing `startFadeOut` is derived so the fade-out lasts as
    /// long as the fade-in (pulled back to `endFadeIn` if that would overlap the fade-in). The
    /// four phases must add up to exactly one day, i.e. the times must be in cyclic order.
    ///
    /// # Errors
    /// A message when the phases do not cover exactly one day.
    pub fn new(
        start_fade_in: i32,
        end_fade_in: i32,
        start_fade_out: Option<i32>,
        end_fade_out: i32,
    ) -> Result<SkyFade, String> {
        let (sfi, efi, efo) = (
            day_ticks(start_fade_in),
            day_ticks(end_fade_in),
            day_ticks(end_fade_out),
        );
        let fade_in = day_ticks(efi - sfi);
        let sfo = match start_fade_out {
            Some(t) => day_ticks(t),
            None => {
                let derived = day_ticks(efo - fade_in);
                if in_window(derived, sfi, efi) {
                    efi
                } else {
                    derived
                }
            }
        };
        let total = fade_in + day_ticks(sfo - efi) + day_ticks(efo - sfo) + day_ticks(sfi - efo);
        if total != TICKS_PER_DAY {
            return Err(format!(
                "fade times are not in day order (the phases add up to {total} ticks, not 24000)"
            ));
        }
        Ok(SkyFade {
            start_fade_in: sfi,
            end_fade_in: efi,
            start_fade_out: sfo,
            end_fade_out: efo,
        })
    }

    /// Brightness `0..=1` at a time of day in ticks (`worldTime % 24000`): rising linearly
    /// through the fade-in, 1 until the fade-out, falling linearly through it, then 0.
    #[must_use]
    pub fn brightness(&self, time_of_day: i32) -> f32 {
        let t = day_ticks(time_of_day);
        let (sfi, efi, sfo, efo) = (
            self.start_fade_in,
            self.end_fade_in,
            self.start_fade_out,
            self.end_fade_out,
        );
        if in_window(t, sfi, efi) {
            let span = day_ticks(efi - sfi);
            if span == 0 {
                1.0
            } else {
                day_ticks(t - sfi) as f32 / span as f32
            }
        } else if in_window(t, efi, sfo) {
            1.0
        } else if in_window(t, sfo, efo) {
            let span = day_ticks(efo - sfo);
            if span == 0 {
                0.0
            } else {
                1.0 - day_ticks(t - sfo) as f32 / span as f32
            }
        } else {
            0.0
        }
    }
}

/// Parse `hh:mm` (24 h) into ticks after 06:00. `None` for anything else.
#[must_use]
pub fn parse_time(value: &str) -> Option<i32> {
    let (hours, minutes) = value.trim().split_once(':')?;
    let hours: i32 = hours.trim().parse().ok()?;
    let minutes: i32 = minutes.trim().parse().ok()?;
    if !(0..=23).contains(&hours) || !(0..=59).contains(&minutes) {
        return None;
    }
    // One hour is 1000 ticks; minutes are scaled the same way and truncated.
    let minute_ticks = (f64::from(minutes) / 60.0 * 1000.0) as i32;
    Some(day_ticks((hours - 6) * 1000) + minute_ticks)
}

/// Which weather a layer shows in (`weather=`). The default is clear only, which is what
/// makes every ordinary layer fade out as rain sets in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkyWeather {
    /// Shown in clear weather.
    pub clear: bool,
    /// Shown in rain (without thunder).
    pub rain: bool,
    /// Shown in a thunderstorm.
    pub thunder: bool,
}

impl Default for SkyWeather {
    fn default() -> Self {
        SkyWeather {
            clear: true,
            rain: false,
            thunder: false,
        }
    }
}

/// One parsed, valid sky layer.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyLayer {
    /// The dimension: `0` the overworld, `1` the End.
    pub world: i32,
    /// `N` of `sky<N>.properties`: the draw order within the world.
    pub number: u32,
    /// The properties file, for log lines.
    pub file: ResourceLocation,
    /// The texture, a 3x2 grid of cube faces.
    pub source: ResourceLocation,
    /// `None`: no fade times, always at full brightness.
    pub fade: Option<SkyFade>,
    /// How the layer combines with the sky drawn so far.
    pub blend: SkyBlend,
    /// Whether the layer turns with the time of day.
    pub rotate: bool,
    /// Turns per day.
    pub speed: f32,
    /// Unit rotation axis in world space (+X east, +Y up, +Z south). Seen looking along it, the
    /// layer turns clockwise; the default, south, makes it rise in the east like the sun.
    pub axis: [f32; 3],
    /// The weather it shows in.
    pub weather: SkyWeather,
    /// Biome ids the camera must be in; `None` for every biome.
    pub biomes: Option<Vec<u8>>,
    /// Inclusive Y ranges the camera must be in; `None` for every height.
    pub heights: Option<Vec<(i32, i32)>>,
    /// Seconds a biome/height change takes to fade the layer in or out.
    pub transition: f32,
}

/// Everything the active packs say about custom skies.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CustomSkyFiles {
    /// Valid layers, by world, then by number.
    pub layers: Vec<SkyLayer>,
    /// Layer files that were skipped, with the reason.
    pub skipped: Vec<(ResourceLocation, String)>,
}

impl CustomSkyFiles {
    /// Find and parse every layer of the active pack stack. Never fails: a broken layer is
    /// listed in [`CustomSkyFiles::skipped`] and left out.
    #[must_use]
    pub fn load(rm: &ResourceManager) -> CustomSkyFiles {
        let mut out = CustomSkyFiles::default();
        for world in SKY_WORLDS {
            for (number, file) in layer_files(rm, world) {
                let text = match rm.load(&file) {
                    Ok(bytes) => latin1(&bytes),
                    Err(error) => {
                        out.skipped.push((file, format!("unreadable: {error}")));
                        continue;
                    }
                };
                match parse_layer(world, number, &file, &text) {
                    Ok(layer) if !rm.contains(&layer.source) => {
                        let why = format!("texture {} not found", layer.source);
                        out.skipped.push((file, why));
                    }
                    Ok(layer) => out.layers.push(layer),
                    Err(why) => out.skipped.push((file, why)),
                }
            }
        }
        out
    }

    /// The layers of one world, in draw order.
    pub fn world(&self, world: i32) -> impl Iterator<Item = &SkyLayer> + '_ {
        self.layers.iter().filter(move |layer| layer.world == world)
    }
}

/// Java's `Properties.load(InputStream)` reads ISO-8859-1: every byte is one `char`.
fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

/// `(N, file)` for every `sky<N>.properties` of `world` in any pack, ascending by `N`. Each `N`
/// resolves through the whole stack; `optifine/` wins over `mcpatcher/`.
fn layer_files(rm: &ResourceManager, world: i32) -> Vec<(u32, ResourceLocation)> {
    let mut numbers = BTreeSet::new();
    for root in SKY_ROOTS {
        let dir = format!("{root}sky/world{world}/");
        for listing in rm.list_files_by_pack("minecraft", &dir) {
            for path in listing {
                if let Some(number) = path.strip_prefix(&dir).and_then(layer_number) {
                    numbers.insert(number);
                }
            }
        }
    }
    numbers
        .into_iter()
        .filter_map(|number| {
            SKY_ROOTS.iter().find_map(|root| {
                let file = ResourceLocation::minecraft(format!(
                    "{root}sky/world{world}/sky{number}.properties"
                ));
                rm.contains(&file).then_some((number, file))
            })
        })
        .collect()
}

/// `N` from a file name `sky<N>.properties`, written the way the format numbers layers (no
/// sign, no leading zero), `1..=999`.
fn layer_number(name: &str) -> Option<u32> {
    let digits = name.strip_prefix("sky")?.strip_suffix(".properties")?;
    if digits.is_empty() || digits.starts_with('0') || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits
        .parse()
        .ok()
        .filter(|n| (1..=MAX_LAYER_NUMBER).contains(n))
}

/// Parse one layer file. `file` is its location, which relative `source` paths resolve against.
///
/// # Errors
/// Why the layer is unusable.
pub fn parse_layer(
    world: i32,
    number: u32,
    file: &ResourceLocation,
    text: &str,
) -> Result<SkyLayer, String> {
    let props = parse_properties(text);
    let get = |key: &str| props.get(key).map(String::as_str).filter(|v| !v.is_empty());

    let dir = file.path.rsplit_once('/').map_or("", |(dir, _)| dir);
    let source = match get("source") {
        Some(value) => resolve_source(value, &file.namespace, dir)
            .ok_or_else(|| format!("bad source path {value:?}"))?,
        None => ResourceLocation::new(&file.namespace, format!("{dir}/sky{number}.png")),
    };
    if !source.is_safe() {
        return Err(format!("unsafe source path {source}"));
    }

    let time = |key: &str| -> Result<Option<i32>, String> {
        match get(key) {
            None => Ok(None),
            Some(value) => parse_time(value)
                .map(Some)
                .ok_or_else(|| format!("bad {key} time {value:?}")),
        }
    };
    let start_fade_in = time("startFadeIn")?;
    let end_fade_in = time("endFadeIn")?;
    let start_fade_out = time("startFadeOut")?;
    let end_fade_out = time("endFadeOut")?;
    let fade = match (start_fade_in, end_fade_in, end_fade_out) {
        (None, None, None) if start_fade_out.is_none() => None,
        (Some(sfi), Some(efi), Some(efo)) => Some(SkyFade::new(sfi, efi, start_fade_out, efo)?),
        _ => return Err("startFadeIn, endFadeIn and endFadeOut must be given together".into()),
    };

    // A value that cannot be read falls back to the default with a warning; only bad
    // times, a negative speed and bad biomes/heights make the whole layer unusable.
    let fallback = |key: &str, value: &str| {
        tracing::warn!(file = %file, key, value, "custom sky: unreadable value, using the default");
    };
    let blend = match get("blend") {
        None => SkyBlend::Add,
        Some(value) => SkyBlend::parse(value).unwrap_or_else(|| {
            fallback("blend", value);
            SkyBlend::Add
        }),
    };
    // java: `Boolean.parseBoolean`: only `true` (any case) is true.
    let rotate = get("rotate").is_none_or(|value| value.eq_ignore_ascii_case("true"));
    let speed = match get("speed").map(|value| (value, value.parse::<f32>())) {
        None => 1.0,
        Some((_, Ok(speed))) if speed.is_finite() && speed >= 0.0 => speed,
        Some((value, Ok(_))) => return Err(format!("bad speed {value:?}")),
        Some((value, Err(_))) => {
            fallback("speed", value);
            1.0
        }
    };
    let default_axis = [0.0, 0.0, 1.0];
    let axis = match get("axis") {
        None => default_axis,
        Some(value) => parse_axis(value).unwrap_or_else(|| {
            fallback("axis", value);
            default_axis
        }),
    };
    let weather = match get("weather") {
        None => SkyWeather::default(),
        Some(value) => parse_weather(value).unwrap_or_else(|| {
            fallback("weather", value);
            SkyWeather::default()
        }),
    };
    let biomes = match get("biomes") {
        None => None,
        Some(value) => {
            Some(parse_biomes(value).ok_or_else(|| format!("no known biome in {value:?}"))?)
        }
    };
    let heights = match get("heights") {
        None => None,
        Some(value) => Some(parse_ranges(value).ok_or_else(|| format!("bad heights {value:?}"))?),
    };
    let transition = match get("transition") {
        None => 1.0,
        Some(value) => value
            .parse::<f32>()
            .ok()
            .filter(|s| s.is_finite() && *s >= 0.0)
            .unwrap_or_else(|| {
                fallback("transition", value);
                1.0
            }),
    };

    Ok(SkyLayer {
        world,
        number,
        file: file.clone(),
        source,
        fade,
        blend,
        rotate,
        speed,
        axis,
        weather,
        biomes,
        heights,
        transition,
    })
}

/// Resolve a `source=` path. `dir` is the properties file's folder within `namespace`.
fn resolve_source(value: &str, namespace: &str, dir: &str) -> Option<ResourceLocation> {
    let value = value.trim().replace('\\', "/");
    let with_png = |path: String| {
        let last = path.rsplit('/').next().unwrap_or(&path);
        if last.contains('.') {
            path
        } else {
            format!("{path}.png")
        }
    };
    let location = if let Some(rest) = value.strip_prefix("./") {
        ResourceLocation::new(namespace, with_png(format!("{dir}/{rest}")))
    } else if let Some(rest) = value
        .strip_prefix("~/")
        .or_else(|| value.strip_prefix("/~/"))
    {
        // `~` is the folder the file sits under: `optifine/` or `mcpatcher/`.
        let root = dir.split('/').next().unwrap_or("mcpatcher");
        ResourceLocation::new(namespace, with_png(format!("{root}/{rest}")))
    } else if let Some(rest) = value.strip_prefix("assets/minecraft/") {
        ResourceLocation::minecraft(with_png(rest.to_owned()))
    } else if let Some((ns, path)) = value.split_once(':') {
        ResourceLocation::new(ns, with_png(path.trim_start_matches('/').to_owned()))
    } else {
        ResourceLocation::new(
            namespace,
            with_png(value.trim_start_matches('/').to_owned()),
        )
    };
    (!location.path.is_empty()).then_some(location)
}

/// `axis=x y z`, any whitespace between, not all zero; returned normalised.
fn parse_axis(value: &str) -> Option<[f32; 3]> {
    let mut parts = value.split_whitespace().map(|s| s.parse::<f32>().ok());
    let (x, y, z) = (parts.next()??, parts.next()??, parts.next()??);
    if parts.next().is_some() {
        return None;
    }
    let len = (x * x + y * y + z * z).sqrt();
    (len.is_finite() && len > 1e-4).then(|| [x / len, y / len, z / len])
}

/// `weather=` tokens; unknown ones are ignored. `None` when none is known.
fn parse_weather(value: &str) -> Option<SkyWeather> {
    let mut weather = SkyWeather {
        clear: false,
        rain: false,
        thunder: false,
    };
    for token in value.split_whitespace() {
        match token.to_ascii_lowercase().as_str() {
            "clear" => weather.clear = true,
            "rain" => weather.rain = true,
            "thunder" => weather.thunder = true,
            _ => {}
        }
    }
    (weather.clear || weather.rain || weather.thunder).then_some(weather)
}

/// Biome names (any case, spaces and underscores ignored, `minecraft:` allowed) or numeric ids.
/// Unknown names are ignored with a warning; `None` when no biome is known.
fn parse_biomes(value: &str) -> Option<Vec<u8>> {
    let squash = |s: &str| -> String {
        s.chars()
            .filter(|c| *c != ' ' && *c != '_')
            .flat_map(char::to_lowercase)
            .collect()
    };
    // The registered names squashed once per layer (each token used
    // to squash every one of them again), the first name winning as `find` did, and one warning
    // for the unknown names (one per token could be millions, each formatted while loading).
    let mut by_name: HashMap<String, u8> = HashMap::new();
    for biome in mc_data::biomes().filter(|biome| biome.registered) {
        by_name.entry(squash(biome.name)).or_insert(biome.id);
    }
    let mut ids = Vec::new();
    let (mut unknown, mut first_unknown) = (0usize, None);
    for token in value.split_whitespace() {
        let token = token.strip_prefix("minecraft:").unwrap_or(token);
        let id = match token.parse::<u8>() {
            Ok(id) => Some(id),
            Err(_) => by_name.get(&squash(token)).copied(),
        };
        match id {
            Some(id) if !ids.contains(&id) => ids.push(id),
            Some(_) => {}
            None => {
                unknown += 1;
                first_unknown.get_or_insert(token);
            }
        }
    }
    if let Some(first) = first_unknown {
        tracing::warn!(
            biome = first,
            unknown,
            "custom sky: unknown biome names ignored"
        );
    }
    (!ids.is_empty()).then_some(ids)
}

/// Integer ranges like `heights=0-64 100 (-3)-10`: one value or `a-b`; negative numbers are
/// written in parentheses.
fn parse_ranges(value: &str) -> Option<Vec<(i32, i32)>> {
    fn number(s: &str) -> Option<i32> {
        let s = s.trim();
        let s = s
            .strip_prefix('(')
            .and_then(|inner| inner.strip_suffix(')'))
            .unwrap_or(s);
        s.parse().ok()
    }
    let mut ranges = Vec::new();
    for token in value.split_whitespace() {
        // The separating `-` is the first one after the first character, or after a leading
        // `(...)`. Both offsets are char boundaries (`)` is ASCII).
        let split_from = if token.starts_with('(') {
            token.find(')').map_or(token.len(), |i| i + 1)
        } else {
            token.chars().next().map_or(0, char::len_utf8)
        };
        let (low, high) = match token[split_from..].find('-') {
            Some(i) => {
                let at = split_from + i;
                (number(&token[..at])?, number(&token[at + 1..])?)
            }
            None => {
                let n = number(token)?;
                (n, n)
            }
        };
        ranges.push((low.min(high), low.max(high)));
    }
    (!ranges.is_empty()).then_some(ranges)
}

/// A Java `Properties.load` parse: `#`/`!` comment lines, keys ending at an unescaped `=`, `:` or
/// whitespace, CR, LF or CRLF line ends (in any mix, as real packs have), backslash line
/// continuations, and escapes in keys and values (`\:`, `\=`, `\t`, `\n`, `\uXXXX`, ...). The last
/// duplicate key wins. Every value is trimmed.
#[must_use]
pub fn parse_properties(text: &str) -> HashMap<String, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut out = HashMap::new();
    let mut lines = physical_lines(text).into_iter();
    while let Some(first) = lines.next() {
        let first = first.trim_start_matches([' ', '\t', '\u{c}']);
        if first.is_empty() || first.starts_with('#') || first.starts_with('!') {
            continue;
        }
        // Join continuation lines: a line ending in an odd number of backslashes.
        let mut logical = String::from(first);
        let mut continues = ends_in_odd_backslashes(first);
        while continues {
            logical.pop();
            match lines.next() {
                Some(next) => {
                    let next = next.trim_start_matches([' ', '\t', '\u{c}']);
                    logical.push_str(next);
                    // After the pop the joined line ends in
                    // an even run of backslashes, so only the appended line decides (java:
                    // `LineReader` resets its backslash flag at each join). Re-scanning the
                    // whole joined line made a file of continued backslash lines quadratic.
                    continues = ends_in_odd_backslashes(next);
                }
                None => break,
            }
        }
        let (key, value) = split_entry(&logical);
        if !key.is_empty() {
            out.insert(key, value.trim().to_owned());
        }
    }
    out
}

/// Lines split on `\r\n`, `\n` or a lone `\r`.
fn physical_lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                out.push(&text[start..i]);
                start = i + 1;
            }
            b'\r' => {
                out.push(&text[start..i]);
                if bytes.get(i + 1) == Some(&b'\n') {
                    i += 1;
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

fn ends_in_odd_backslashes(s: &str) -> bool {
    let trailing = s.len() - s.trim_end_matches('\\').len();
    trailing % 2 == 1
}

/// Split a logical line into an unescaped key and value.
fn split_entry(line: &str) -> (String, String) {
    let mut chars = line.chars().peekable();
    let mut key = String::new();
    let mut separated = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => key.push(unescape(chars.next(), &mut chars)),
            '=' | ':' => {
                separated = true;
                break;
            }
            ' ' | '\t' | '\u{c}' => break,
            _ => key.push(c),
        }
    }
    // After whitespace, one `=` or `:` may still follow before the value.
    while let Some(&c) = chars.peek() {
        if matches!(c, ' ' | '\t' | '\u{c}') {
            chars.next();
        } else {
            if !separated && matches!(c, '=' | ':') {
                chars.next();
                while chars
                    .peek()
                    .is_some_and(|c| matches!(c, ' ' | '\t' | '\u{c}'))
                {
                    chars.next();
                }
            }
            break;
        }
    }
    let mut value = String::new();
    while let Some(c) = chars.next() {
        if c == '\\' {
            value.push(unescape(chars.next(), &mut chars));
        } else {
            value.push(c);
        }
    }
    (key, value)
}

/// The character a backslash escape stands for. `\uXXXX` reads its four hex digits.
fn unescape(c: Option<char>, rest: &mut std::iter::Peekable<std::str::Chars<'_>>) -> char {
    match c {
        Some('t') => '\t',
        Some('n') => '\n',
        Some('r') => '\r',
        Some('f') => '\u{c}',
        Some('u') => {
            let mut code = 0u32;
            for _ in 0..4 {
                match rest.peek().and_then(|c| c.to_digit(16)) {
                    Some(d) => {
                        code = code * 16 + d;
                        rest.next();
                    }
                    None => break,
                }
            }
            char::from_u32(code).unwrap_or('\u{fffd}')
        }
        Some(other) => other,
        None => '\\',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn properties_parse_like_java_with_escapes_and_mixed_line_ends() {
        // The shapes in the reference pack: `\:` escapes, CRLF, a lone CR, blank lines between
        // entries, a timestamp comment, and no final newline.
        let text = "#Fri Jun 28 14:26:07 CEST 2013\r\n\nstartFadeIn=19\\:00\r\n\nblend=add\n\r\n\
                    rotate=true\r\nendFadeIn=20\\:00\n\r\nsource=./stars_galaxy.png";
        let p = parse_properties(text);
        assert_eq!(p["startFadeIn"], "19:00");
        assert_eq!(p["endFadeIn"], "20:00");
        assert_eq!(p["blend"], "add");
        assert_eq!(p["rotate"], "true");
        assert_eq!(p["source"], "./stars_galaxy.png");
        assert_eq!(p.len(), 5);

        let p = parse_properties(
            "a = 1\rb:2\r\n  c 3\n! bang comment\nd=\\u0041\\=\\\\x\nkey\\ with\\ space=v",
        );
        assert_eq!(p["a"], "1");
        assert_eq!(p["b"], "2");
        assert_eq!(p["c"], "3");
        assert_eq!(p["d"], "A=\\x");
        assert_eq!(p["key with space"], "v");

        // A continuation joins the next line without its leading blanks; the last duplicate wins.
        let p = parse_properties("axis=1 \\\n    0 0\naxis=0 1 0\nsource=a\\\n  b.png");
        assert_eq!(p["axis"], "0 1 0");
        assert_eq!(p["source"], "ab.png");
    }

    #[test]
    fn times_are_ticks_after_six_am() {
        assert_eq!(parse_time("6:00"), Some(0));
        assert_eq!(parse_time("12:00"), Some(6000));
        assert_eq!(parse_time("18:00"), Some(12_000));
        assert_eq!(parse_time("0:00"), Some(18_000));
        assert_eq!(parse_time("00:00"), Some(18_000));
        assert_eq!(parse_time("5:59"), Some(23_983));
        assert_eq!(parse_time("19:20"), Some(13_333));
        assert_eq!(parse_time(" 05:10 "), Some(23_166));
        for bad in ["24:00", "12:60", "12", "12:00:00", "-1:00", "ab:cd", ""] {
            assert_eq!(parse_time(bad), None, "{bad:?}");
        }
        // The double-precision minute scaling truncates to the same tick as integer maths.
        for minute in 0..60 {
            assert_eq!(
                (f64::from(minute) / 60.0 * 1000.0) as i32,
                minute * 1000 / 60
            );
        }
    }

    #[test]
    fn fades_rise_hold_fall_and_wrap_past_midnight() {
        // The user's sky41: 19:00 -> 20:00 in, out by 05:00, startFadeOut derived (04:00).
        let fade = SkyFade::new(
            parse_time("19:00").unwrap(),
            parse_time("20:00").unwrap(),
            None,
            parse_time("5:00").unwrap(),
        )
        .unwrap();
        assert_eq!(fade.start_fade_out, parse_time("4:00").unwrap());
        let at = |hhmm: &str| fade.brightness(parse_time(hhmm).unwrap());
        assert_eq!(at("12:00"), 0.0);
        assert_eq!(at("19:00"), 0.0);
        assert!((at("19:30") - 0.5).abs() < 1e-3);
        assert_eq!(at("20:00"), 1.0);
        assert_eq!(at("0:00"), 1.0);
        assert_eq!(at("4:00"), 1.0);
        assert!((at("4:30") - 0.5).abs() < 1e-3);
        assert_eq!(at("5:00"), 0.0);
        assert_eq!(at("5:01"), 0.0);
    }

    #[test]
    fn a_derived_fade_out_that_would_overlap_the_fade_in_starts_at_its_end() {
        // The user's sky44: 18:15 -> 18:30 in, out by 18:45. Derived 18:30 = endFadeIn.
        let fade = SkyFade::new(
            parse_time("18:15").unwrap(),
            parse_time("18:30").unwrap(),
            None,
            parse_time("18:45").unwrap(),
        )
        .unwrap();
        assert_eq!(fade.start_fade_out, fade.end_fade_in);
        // A fade-out that would start inside the fade-in is pulled back to its end.
        let fade = SkyFade::new(0, 1000, None, 1200).unwrap();
        assert_eq!(fade.start_fade_out, 1000);
        assert!((fade.brightness(1100) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fade_times_out_of_day_order_are_rejected() {
        // endFadeOut inside the fade-in.
        assert!(SkyFade::new(1000, 3000, Some(4000), 2000).is_err());
        // All four the same.
        assert!(SkyFade::new(5, 5, Some(5), 5).is_err());
        // An explicit startFadeOut before endFadeIn.
        assert!(SkyFade::new(1000, 3000, Some(2000), 4000).is_err());
        assert!(SkyFade::new(1000, 3000, Some(3500), 4000).is_ok());
    }

    #[test]
    fn blend_names_parse_with_aliases_and_any_case() {
        assert_eq!(SkyBlend::parse(" Add "), Some(SkyBlend::Add));
        assert_eq!(SkyBlend::parse("SCREEN"), Some(SkyBlend::Screen));
        assert_eq!(SkyBlend::parse("color"), Some(SkyBlend::Overlay));
        assert_eq!(SkyBlend::parse("none"), Some(SkyBlend::Replace));
        assert_eq!(SkyBlend::parse("lighten"), None);
        for (i, blend) in SkyBlend::ALL.iter().enumerate() {
            assert_eq!(blend.index(), i);
        }
    }

    fn file(path: &str) -> ResourceLocation {
        ResourceLocation::minecraft(path.to_owned())
    }

    #[test]
    fn sources_resolve_like_pack_paths() {
        let f = file("mcpatcher/sky/world0/sky7.properties");
        let source = |text: &str| parse_layer(0, 7, &f, text).unwrap().source;
        assert_eq!(source(""), file("mcpatcher/sky/world0/sky7.png"));
        assert_eq!(
            source("source=./starfield.png"),
            file("mcpatcher/sky/world0/starfield.png")
        );
        assert_eq!(
            source("source=~/sky/clouds.png"),
            file("mcpatcher/sky/clouds.png")
        );
        assert_eq!(
            source("source=textures/environment/sun.png"),
            file("textures/environment/sun.png")
        );
        assert_eq!(
            source("source=assets/minecraft/textures/a.png"),
            file("textures/a.png")
        );
        assert_eq!(
            source("source=mymod:sky/b"),
            ResourceLocation::new("mymod", "sky/b.png")
        );
        let f = file("optifine/sky/world1/sky2.properties");
        let layer = parse_layer(1, 2, &f, "source=~/sky/end.png").unwrap();
        assert_eq!(layer.source, file("optifine/sky/end.png"));
        assert!(parse_layer(
            0,
            7,
            &file("mcpatcher/sky/world0/sky7.properties"),
            "source=../../../x.png"
        )
        .is_err());
    }

    #[test]
    fn a_layer_with_every_property() {
        let text = "source=./a.png\nstartFadeIn=17:00\nendFadeIn=17:40\nendFadeOut=20:00\n\
                    blend=burn\nrotate=true\nspeed=1.3\naxis=1  0  0\nweather=rain thunder\n\
                    biomes=Plains extreme_hills 2\nheights=(-3)-64 100\ntransition=3";
        let layer = parse_layer(0, 4, &file("mcpatcher/sky/world0/sky4.properties"), text).unwrap();
        assert_eq!(layer.blend, SkyBlend::Burn);
        assert!(layer.rotate);
        assert!((layer.speed - 1.3).abs() < 1e-6);
        assert_eq!(layer.axis, [1.0, 0.0, 0.0]);
        assert_eq!(
            layer.weather,
            SkyWeather {
                clear: false,
                rain: true,
                thunder: true
            }
        );
        assert_eq!(layer.biomes, Some(vec![1, 3, 2]));
        assert_eq!(layer.heights, Some(vec![(-3, 64), (100, 100)]));
        assert_eq!(layer.transition, 3.0);
        let fade = layer.fade.unwrap();
        assert_eq!(fade.start_fade_in, 11_000);
        assert_eq!(fade.end_fade_in, 11_666);
        assert_eq!(fade.end_fade_out, 14_000);
    }

    #[test]
    fn defaults_and_rejections() {
        let f = file("mcpatcher/sky/world0/sky1.properties");
        let layer = parse_layer(0, 1, &f, "").unwrap();
        assert_eq!(layer.fade, None, "no fade times: always on");
        assert_eq!(layer.blend, SkyBlend::Add);
        assert!(layer.rotate);
        assert_eq!(layer.speed, 1.0);
        assert_eq!(layer.axis, [0.0, 0.0, 1.0]);
        assert_eq!(layer.weather, SkyWeather::default());
        assert_eq!(layer.transition, 1.0);
        // Normalised axis.
        let layer = parse_layer(0, 1, &f, "axis=0.0 -0.2 0.0").unwrap();
        assert_eq!(layer.axis, [0.0, -1.0, 0.0]);
        // An unreadable value falls back to its default...
        let lenient = |text: &str| parse_layer(0, 1, &f, text).unwrap();
        assert_eq!(lenient("blend=lighten").blend, SkyBlend::Add);
        assert_eq!(lenient("axis=0 0 0").axis, [0.0, 0.0, 1.0]);
        assert_eq!(lenient("axis=1 2").axis, [0.0, 0.0, 1.0]);
        assert_eq!(lenient("speed=fast").speed, 1.0);
        assert!(!lenient("rotate=maybe").rotate, "Java parseBoolean");
        assert!(lenient("rotate=TRUE").rotate);
        assert_eq!(lenient("weather=snow").weather, SkyWeather::default());
        assert!(lenient("weather=snow rain").weather.rain);
        assert_eq!(lenient("transition=-2").transition, 1.0);
        assert_eq!(lenient("biomes=Desert NotABiome").biomes, Some(vec![2]));
        // ...while these make the layer unusable.
        for bad in [
            "startFadeIn=1:00",
            "startFadeIn=1:00\nendFadeIn=2:00",
            "startFadeOut=1:00",
            "speed=-1",
            "biomes=NotABiome",
            "heights=a-b",
            "heights=(5",
            "startFadeIn=25:00\nendFadeIn=2:00\nendFadeOut=3:00",
        ] {
            assert!(
                parse_layer(0, 1, &f, bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
        // Multi-byte input never panics the range parser.
        assert!(parse_layer(0, 1, &f, "heights=\u{e9}-1").is_err());
        assert_eq!(
            parse_layer(0, 1, &f, "heights=10-(-3) -5").unwrap().heights,
            Some(vec![(-3, 10), (-5, -5)])
        );
    }

    #[test]
    fn layer_numbers_are_canonical_and_bounded() {
        assert_eq!(layer_number("sky1.properties"), Some(1));
        assert_eq!(layer_number("sky47.properties"), Some(47));
        assert_eq!(layer_number("sky999.properties"), Some(999));
        for bad in [
            "sky0.properties",
            "sky01.properties",
            "sky1000.properties",
            "sky.properties",
            "sky1.png",
            "sun.properties",
            "skyx.properties",
            "sky-1.properties",
        ] {
            assert_eq!(layer_number(bad), None, "{bad}");
        }
    }

    fn folder_pack(files: &[(&str, &[u8])]) -> (tempfile::TempDir, crate::FolderResourcePack) {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("pack.mcmeta"),
            r#"{"pack":{"pack_format":1,"description":"sky"}}"#,
        )
        .unwrap();
        for (path, bytes) in files {
            let path = dir.path().join("assets/minecraft").join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
        }
        let pack = crate::FolderResourcePack::open(dir.path()).expect("folder pack");
        (dir, pack)
    }

    #[test]
    fn discovery_orders_layers_prefers_the_newer_folder_and_skips_missing_textures() {
        let times = b"startFadeIn=18:00\nendFadeIn=19:00\nendFadeOut=6:00\n";
        let (_low_dir, low) = folder_pack(&[
            // Layer 3 only in the lower pack; layer 2 is overridden by the upper pack.
            ("mcpatcher/sky/world0/sky2.properties", b"source=./low.png"),
            ("mcpatcher/sky/world0/low.png", b"png"),
            ("mcpatcher/sky/world0/sky3.properties", times),
            ("mcpatcher/sky/world0/sky3.png", b"png"),
        ]);
        let (_high_dir, high) = folder_pack(&[
            (
                "mcpatcher/sky/world0/sky1.properties",
                b"source=./missing.png",
            ),
            (
                "mcpatcher/sky/world0/sky2.properties",
                b"source=./high.png\nblend=screen",
            ),
            ("mcpatcher/sky/world0/high.png", b"png"),
            (
                "mcpatcher/sky/world0/sky10.properties",
                b"source=./high.png",
            ),
            // The same number under optifine/ wins.
            (
                "optifine/sky/world0/sky10.properties",
                b"source=./o.png\nblend=alpha",
            ),
            ("optifine/sky/world0/o.png", b"png"),
            ("optifine/sky/world1/sky1.properties", b"source=./end.png"),
            ("optifine/sky/world1/end.png", b"png"),
            ("mcpatcher/sky/world0/sky4.properties", b"blend=sideways"),
            ("mcpatcher/sky/world0/sun.properties", b"blend=add"),
        ]);
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(low));
        rm.push_pack(Box::new(high));
        let sky = CustomSkyFiles::load(&rm);
        let order: Vec<(i32, u32)> = sky.layers.iter().map(|l| (l.world, l.number)).collect();
        assert_eq!(order, [(0, 2), (0, 3), (0, 10), (1, 1)]);
        let layer = |w: i32, n: u32| {
            sky.layers
                .iter()
                .find(|l| l.world == w && l.number == n)
                .unwrap()
        };
        assert_eq!(
            layer(0, 2).blend,
            SkyBlend::Screen,
            "the top pack's sky2 wins"
        );
        assert_eq!(layer(0, 2).source, file("mcpatcher/sky/world0/high.png"));
        assert_eq!(layer(0, 3).source, file("mcpatcher/sky/world0/sky3.png"));
        assert_eq!(
            layer(0, 10).blend,
            SkyBlend::Alpha,
            "optifine/ wins over mcpatcher/"
        );
        assert_eq!(layer(1, 1).source, file("optifine/sky/world1/end.png"));
        let skipped: Vec<&str> = sky.skipped.iter().map(|(f, _)| f.path.as_str()).collect();
        assert_eq!(
            skipped,
            [
                "mcpatcher/sky/world0/sky1.properties",
                "mcpatcher/sky/world0/sky4.properties"
            ]
        );
        assert!(
            sky.skipped[0].1.contains("not found"),
            "{}",
            sky.skipped[0].1
        );
        assert_eq!(sky.world(0).count(), 3);
        assert_eq!(sky.world(1).count(), 1);
    }

    /// The resource pack the counts below were taken from (never checked in, read-only):
    /// `MC_REFERENCE_PACK`, or else the only folder in the workspace's `resourcepacks/`.
    fn users_pack() -> Option<std::path::PathBuf> {
        if let Some(dir) = std::env::var_os("MC_REFERENCE_PACK") {
            return Some(dir.into());
        }
        let packs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resourcepacks");
        let mut dirs = std::fs::read_dir(packs)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir());
        let only = dirs.next()?;
        dirs.next().is_none().then_some(only)
    }

    /// The reference pack (read-only): 47 layer files, 21 with a texture that exists, every one of
    /// those with valid fade times; the other 26 name PNGs the pack does not ship.
    #[test]
    fn the_users_pack_has_21_drawable_layers_out_of_47() {
        let Some(pack) = users_pack() else {
            eprintln!(
                "SKIP: no reference pack (MC_REFERENCE_PACK, or one folder in resourcepacks)"
            );
            return;
        };
        let mut rm = ResourceManager::new();
        rm.push_pack(Box::new(
            crate::FolderResourcePack::open(&pack).expect("pack"),
        ));
        let sky = CustomSkyFiles::load(&rm);
        let numbers: Vec<u32> = sky.world(0).map(|l| l.number).collect();
        assert_eq!(
            numbers,
            [2, 3, 4, 5, 8, 11, 14, 15, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47]
        );
        assert_eq!(sky.skipped.len(), 26);
        assert!(
            sky.skipped.iter().all(|(_, why)| why.contains("not found")),
            "only missing textures are skipped: {:?}",
            sky.skipped
        );
        let layer = |n: u32| sky.layers.iter().find(|l| l.number == n).unwrap();
        // sky2: the daytime box, replace, fixed.
        assert_eq!(layer(2).blend, SkyBlend::Replace);
        assert!(!layer(2).rotate);
        // sky8: depth.png, add by default, speed 1.3 about east (`axis=1  0  0`).
        assert_eq!(layer(8).blend, SkyBlend::Add);
        assert_eq!(layer(8).axis, [1.0, 0.0, 0.0]);
        assert!((layer(8).speed - 1.3).abs() < 1e-6);
        // sky14: a shooting star, screen, 200 turns a day for six game minutes after midnight.
        assert_eq!(layer(14).blend, SkyBlend::Screen);
        assert_eq!(layer(14).speed, 200.0);
        let star = layer(14).fade.unwrap();
        assert_eq!(star.brightness(parse_time("0:03").unwrap()), 1.0);
        assert_eq!(star.brightness(parse_time("0:07").unwrap()), 0.0);
        // sky37: starfield02.png with burn.
        assert_eq!(layer(37).blend, SkyBlend::Burn);
        for layer in sky.world(0) {
            assert!(
                layer.weather.clear && !layer.weather.rain,
                "default weather"
            );
        }
    }
}
