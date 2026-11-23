//! `options.txt` — vanilla's settings file, read and written in its own format.
//!
//! java: `client/settings/GameSettings.java` (`loadOptions` / `saveOptions`)
//!
//! The format is one `key:value` per line, no sections, no escaping, no header. Everything about
//! it is forgiving by design, and this implementation is more forgiving still:
//!
//! * **Unknown keys round-trip.** A key we do not model is kept verbatim and rewritten at the end
//!   of the file. That means a `options.txt` written by vanilla 1.8.9 — or by a future section of
//!   this port — survives a load/save cycle here without losing settings.
//! * **Malformed values are ignored, never fatal.** A corrupt line falls back to the default for
//!   that field, matching vanilla's per-line `try`/`catch`.
//! * **Colon handling follows `GameSettings`.** Most options use Java's `split(":")` second
//!   field (with trailing empty fields discarded). `lastServer` is one of the deliberate
//!   exceptions: it retains everything after its first colon, including a port.

// Settings I/O deliberately returns the renderer's rich, inline diagnostic error type; changing
// that public Result shape just to satisfy the size heuristic would be an API change.
#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;
use std::path::Path;

use crate::context::PresentPreference;
use crate::input::KeyBindings;
use crate::RenderError;

/// Vanilla's sound categories, in ordinal order. java: `client/audio/SoundCategory.java`
pub const SOUND_CATEGORIES: [&str; 9] = [
    "master", "music", "record", "weather", "block", "hostile", "neutral", "player", "ambient",
];

/// Vanilla's toggleable skin overlays, in ordinal order.
/// java: `entity/player/EnumPlayerModelParts.java`
pub const MODEL_PARTS: [&str; 7] = [
    "cape",
    "jacket",
    "left_sleeve",
    "right_sleeve",
    "left_pants_leg",
    "right_pants_leg",
    "hat",
];

/// The value `maxFps` takes to mean "unlimited". java: `GameSettings.Options.FRAMERATE_LIMIT` max
pub const FRAMERATE_UNLIMITED: i32 = 260;
/// Highest finite native-client frame cap. Keep 260 as the legacy Unlimited save value.
pub const MAX_FRAMERATE_LIMIT: i32 = 2000;

/// Vanilla's ordinary render-distance slider ceiling, in chunks.
pub const RENDER_DISTANCE_MAX_NORMAL: i32 = 16;
/// Vanilla's enlarged render-distance slider ceiling, in chunks.
pub const RENDER_DISTANCE_MAX_EXTREME: i32 = 32;
/// Exact `Runtime.maxMemory()` threshold for the enlarged slider. This is decimal one billion,
/// not one gibibyte. java: `GameSettings.java:195`
pub const RENDER_DISTANCE_EXTREME_MEMORY_BYTES: u64 = 1_000_000_000;

/// Reproduce the render-distance ceiling chosen by the 1.8.9 `GameSettings` constructor.
///
/// `max_memory_bytes` is Java's maximum heap, not its currently free heap or the process's
/// resident set. Keeping it explicit makes the policy usable by launchers with an allocation
/// budget and independently testable without platform-specific memory APIs.
// java: GameSettings.java:195-202
#[must_use]
pub const fn render_distance_max_for(is_64_bit: bool, max_memory_bytes: u64) -> i32 {
    if is_64_bit && max_memory_bytes >= RENDER_DISTANCE_EXTREME_MEMORY_BYTES {
        RENDER_DISTANCE_MAX_EXTREME
    } else {
        RENDER_DISTANCE_MAX_NORMAL
    }
}

/// Render-distance ceiling for this native process when no launcher heap cap exists.
///
/// Unlike a JVM, Rust's allocator has no configured `-Xmx`; its representable allocation ceiling
/// follows the process address width. Thus a native 64-bit process satisfies Java's one-billion
/// byte maximum-heap condition, while a 32-bit process deliberately keeps the conservative cap.
/// Java's later ATI-without-VBO downgrade has no reachable wgpu counterpart: vertex buffers are a
/// required renderer primitive, so an adapter unable to provide them cannot run this client.
#[must_use]
pub const fn native_render_distance_max() -> i32 {
    render_distance_max_for(cfg!(target_pointer_width = "64"), u64::MAX)
}

/// Fresh-install render distance selected by vanilla solely from process bitness.
/// java: `GameSettings.java:204`
#[must_use]
pub const fn default_render_distance_for(is_64_bit: bool) -> i32 {
    if is_64_bit {
        12
    } else {
        8
    }
}

/// Fresh-install render distance for this native process.
#[must_use]
pub const fn native_default_render_distance() -> i32 {
    default_render_distance_for(cfg!(target_pointer_width = "64"))
}

/// Everything in `options.txt`, plus the key bindings.
///
/// Field names are the Rust spelling of the vanilla ones; the doc comment on each gives the
/// `options.txt` key where it is not obvious. Only the settings this port can actually honour
/// have behaviour attached — the rest exist so the file round-trips.
#[derive(Debug, Clone)]
pub struct GameSettings {
    /// `mouseSensitivity`, 0..1. Feeds [`InputState::look_delta`](crate::InputState::look_delta).
    pub mouse_sensitivity: f32,
    /// `invertYMouse`.
    pub invert_mouse: bool,
    /// `smoothCamera`. This is deliberately session-only: 1.8.9 exposes the field and toggles
    /// it through `key.smoothCamera`, but neither `loadOptions` nor `saveOptions` serializes it.
    pub smooth_camera: bool,
    /// `fov` — stored on disk as `(degrees - 70) / 40`, held here in **vertical degrees**.
    pub fov: f32,
    /// `gamma`.
    pub gamma: f32,
    /// `saturation`.
    pub saturation: f32,
    /// `renderDistance`, in chunks. Drives the camera's far plane.
    pub render_distance_chunks: i32,
    /// `guiScale`: 0 = auto, 1..3 = fixed.
    pub gui_scale: i32,
    /// `particles`: 0 = all, 1 = decreased, 2 = minimal.
    pub particles: i32,
    /// `bobView`.
    pub view_bobbing: bool,
    /// `anaglyph3d`.
    pub anaglyph: bool,
    /// `maxFps`. [`FRAMERATE_UNLIMITED`] means no cap.
    pub limit_framerate: i32,
    /// `fboEnable`.
    pub fbo_enable: bool,
    /// `difficulty`, 0..3.
    pub difficulty: i32,
    /// `fancyGraphics`.
    pub fancy_graphics: bool,
    /// `ao` — smooth lighting: 0 = off, 1 = minimum, 2 = maximum.
    pub ambient_occlusion: i32,
    /// `renderClouds`: 0 = off (`false`), 1 = fast, 2 = fancy (`true`).
    pub clouds: i32,
    /// `resourcePacks`, a JSON array of pack names.
    pub resource_packs: Vec<String>,
    /// `incompatibleResourcePacks`.
    pub incompatible_resource_packs: Vec<String>,
    /// `lastServer`.
    pub last_server: String,
    /// `lang`.
    pub language: String,
    /// `chatVisibility`: 0 = full, 1 = system, 2 = hidden.
    pub chat_visibility: i32,
    /// `chatColors`.
    pub chat_colors: bool,
    /// `chatLinks`.
    pub chat_links: bool,
    /// `chatLinksPrompt`.
    pub chat_links_prompt: bool,
    /// `chatOpacity`.
    pub chat_opacity: f32,
    /// `snooperEnabled`. Nothing in this port phones home; kept for file compatibility.
    pub snooper_enabled: bool,
    /// `fullscreen`.
    pub fullscreen: bool,
    /// `enableVsync`. Selects the wgpu present mode.
    pub enable_vsync: bool,
    /// `useVbo`. Meaningless under wgpu (everything is a buffer); kept for compatibility.
    pub use_vbo: bool,
    /// `hideServerAddress`.
    pub hide_server_address: bool,
    /// `advancedItemTooltips`.
    pub advanced_item_tooltips: bool,
    /// `pauseOnLostFocus`.
    pub pause_on_lost_focus: bool,
    /// `touchscreen`.
    pub touchscreen: bool,
    /// `overrideWidth`.
    pub override_width: i32,
    /// `overrideHeight`.
    pub override_height: i32,
    /// `heldItemTooltips`.
    pub held_item_tooltips: bool,
    /// `chatHeightFocused`.
    pub chat_height_focused: f32,
    /// `chatHeightUnfocused`.
    pub chat_height_unfocused: f32,
    /// `chatScale`.
    pub chat_scale: f32,
    /// `chatWidth`.
    pub chat_width: f32,
    /// `showInventoryAchievementHint`.
    pub show_inventory_achievement_hint: bool,
    /// `mipmapLevels`.
    pub mipmap_levels: i32,
    /// `forceUnicodeFont`.
    pub force_unicode_font: bool,
    /// `allowBlockAlternatives`.
    pub allow_block_alternatives: bool,
    /// `reducedDebugInfo`.
    pub reduced_debug_info: bool,
    /// `useNativeTransport`.
    pub use_native_transport: bool,
    /// `entityShadows`.
    pub entity_shadows: bool,
    /// `frameLatency` — **not a vanilla key** (PRESENT-5). The swap chain's
    /// `desired_maximum_frame_latency`, 1..=3. The default, 2, is what the client always ran
    /// with; 1 is the low-latency choice (see [`crate::context::DEFAULT_FRAME_LATENCY`]).
    /// Applied live, together with `enableVsync` and `presentMode`.
    pub frame_latency: u32,
    /// `realmsNotifications`. Realms is permanently out of scope (§2); kept for compatibility.
    pub realms_notifications: bool,
    /// `soundCategory_*`, 0..1 per category. See [`SOUND_CATEGORIES`].
    pub sound_levels: BTreeMap<String, f32>,
    /// `modelPart_*`. See [`MODEL_PARTS`].
    pub model_parts: BTreeMap<String, bool>,
    /// `msaaSamples` — **not a vanilla key**. 1, 2, 4, or 8; 1 disables multisampling.
    /// Vanilla 1.8.9 has no MSAA option at all, so this is an addition rather than a port.
    pub msaa_samples: u32,
    /// `presentMode` — **not a vanilla key**. One of `auto`, `fifo`,
    /// `fifo_relaxed`, `mailbox`, `immediate`.
    ///
    /// Vanilla has only the `enableVsync` boolean, which cannot express `Mailbox` — tear-free
    /// presentation *without* the blocking wait that was measured at 2.239 ms of a
    /// 2.783 ms frame. [`PresentPreference::Auto`] is the default and behaves exactly as the
    /// client did before this key existed; anything else overrides `enableVsync`.
    pub present_mode: PresentPreference,
    /// `fullbright` — **not a vanilla key**, default `false`. Forces every lightmap texel to full
    /// white ([`crate::sky::lightmap::LightmapParams::fullbright`]) and lights tile entities at
    /// full brightness. Render-only: client light is still propagated and read wherever vanilla
    /// reads it outside a draw (the mushroom placement gate before C08), so nothing sent changes.
    pub fullbright: bool,
    /// `playerRenderDistance` — **not a vanilla key**. Blocks; `0` (the default) leaves remote
    /// players on vanilla's own range (`EntityOtherPlayerMP`: ~640 blocks, so in practice the
    /// server's tracking range). A positive value additionally hides players at or beyond it.
    pub player_render_distance: i32,
    /// `entityRenderDistance` — **not a vanilla key**. Blocks; `0` (the default) is vanilla. A
    /// positive value hides every other entity at or beyond it, except projectiles, the fishing
    /// hook and primed TNT, which keep vanilla's range because they can hit the player.
    pub entity_render_distance: i32,
    /// Rebindable actions, persisted as `key_key.forward:17` lines.
    pub keys: KeyBindings,

    /// Keys we do not model, preserved verbatim so nothing is silently dropped on save.
    ///
    /// Public only so that `..GameSettings::default()` struct-update syntax works from other
    /// crates (S16's options screen builds these). Read it through
    /// [`GameSettings::unknown_keys`]; clearing it discards settings a future section or a
    /// vanilla install may still want.
    pub unknown: Vec<(String, String)>,
}

impl Default for GameSettings {
    /// Vanilla's out-of-the-box settings. java: the `GameSettings` field initialisers
    fn default() -> Self {
        Self {
            mouse_sensitivity: 0.5,
            invert_mouse: false,
            smooth_camera: false,
            fov: crate::frame::DEFAULT_FOV,
            gamma: 0.0,
            saturation: 0.0,
            // Java chooses 12 on a 64-bit process and 8 on a 32-bit process before loading the
            // options file. Memory affects the slider ceiling, not this default.
            render_distance_chunks: native_default_render_distance(),
            gui_scale: 0,
            particles: 0,
            view_bobbing: true,
            anaglyph: false,
            limit_framerate: 120,
            fbo_enable: true,
            difficulty: 2,
            fancy_graphics: true,
            ambient_occlusion: 2,
            clouds: 2,
            resource_packs: Vec::new(),
            incompatible_resource_packs: Vec::new(),
            last_server: String::new(),
            language: "en_US".into(),
            chat_visibility: 0,
            chat_colors: true,
            chat_links: true,
            chat_links_prompt: true,
            chat_opacity: 1.0,
            snooper_enabled: true,
            fullscreen: false,
            enable_vsync: true,
            use_vbo: false,
            hide_server_address: false,
            advanced_item_tooltips: false,
            pause_on_lost_focus: true,
            touchscreen: false,
            override_width: 0,
            override_height: 0,
            held_item_tooltips: true,
            chat_height_focused: 1.0,
            chat_height_unfocused: 0.443_661_96,
            chat_scale: 1.0,
            chat_width: 1.0,
            show_inventory_achievement_hint: true,
            mipmap_levels: 4,
            force_unicode_font: false,
            allow_block_alternatives: true,
            reduced_debug_info: false,
            use_native_transport: true,
            entity_shadows: true,
            frame_latency: crate::context::DEFAULT_FRAME_LATENCY,
            realms_notifications: true,
            sound_levels: SOUND_CATEGORIES
                .iter()
                .map(|c| ((*c).to_owned(), 1.0))
                .collect(),
            model_parts: MODEL_PARTS
                .iter()
                .map(|p| ((*p).to_owned(), true))
                .collect(),
            msaa_samples: 1,
            present_mode: PresentPreference::Auto,
            fullbright: false,
            player_render_distance: 0,
            entity_render_distance: 0,
            keys: KeyBindings::vanilla(),
            unknown: Vec::new(),
        }
    }
}

impl GameSettings {
    /// Load from `path`. A missing file yields the defaults, exactly as vanilla does — a first
    /// run is not an error.
    ///
    /// The bytes are decoded line by line ([`decode_options_bytes`]): a byte that is not UTF-8
    /// costs at most its own line's exact characters, never the file. Vanilla reads through a
    /// `FileReader`, which cannot fail on encoding either; failing the whole load here used to
    /// put every option back to its default and the next save then wrote those defaults over the
    /// player's file.
    ///
    /// # Errors
    /// [`RenderError::Settings`] only if the file exists but cannot be read.
    pub fn load(path: &Path) -> Result<Self, RenderError> {
        match std::fs::read(path) {
            Ok(bytes) => Ok(Self::parse(&decode_options_bytes(&bytes))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(RenderError::Settings(e.to_string())),
        }
    }

    /// [`Self::load`] for the game itself: never fails, and keeps a copy of a file it could not
    /// use before anything can overwrite it.
    ///
    /// A file is *damaged* when it cannot be read, holds NUL bytes (what a power cut during a
    /// write leaves behind), or is not blank but has no `key:value` line at all. Its readable
    /// lines are still used (NUL bytes dropped), the rest falls back to the defaults, and the
    /// file is first copied to `options.txt.bad`; [`OptionsFileState::Damaged`] says so.
    #[must_use]
    pub fn load_keeping_damaged(path: &Path) -> (Self, OptionsFileState) {
        let bytes = match read_patiently(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return (Self::default(), OptionsFileState::Fine);
            }
            Err(e) => {
                let backup = mc_core::durable::keep_aside(path, DAMAGED_SUFFIX);
                return (
                    Self::default(),
                    OptionsFileState::Damaged {
                        reason: format!("could not be read: {e}"),
                        backup,
                    },
                );
            }
        };
        let text = decode_options_bytes(&bytes);
        let (settings, lines_read) = Self::parse_counting(&text);
        let reason = if bytes.contains(&0) {
            Some("holds NUL bytes (an interrupted write)")
        } else if lines_read == 0 && !text.trim().is_empty() {
            Some("has no `key:value` line")
        } else {
            None
        };
        match reason {
            None => (settings, OptionsFileState::Fine),
            Some(reason) => (
                settings,
                OptionsFileState::Damaged {
                    reason: reason.to_owned(),
                    backup: mc_core::durable::keep_aside(path, DAMAGED_SUFFIX),
                },
            ),
        }
    }

    /// Write to `path`, creating parent directories if needed.
    ///
    /// The file is replaced atomically and flushed to the disk first
    /// ([`mc_core::durable::write_atomic`]): a crash or a power cut during the save leaves the
    /// old `options.txt` or the new one, never a truncated file. Vanilla truncates and writes in
    /// place.
    ///
    /// # Errors
    /// [`RenderError::Settings`] if the file cannot be written (the old one is then untouched).
    pub fn save(&self, path: &Path) -> Result<(), RenderError> {
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(|e| RenderError::Settings(e.to_string()))?;
            }
        }
        mc_core::durable::write_atomic(path, self.to_options_txt().as_bytes())
            .map_err(|e| RenderError::Settings(e.to_string()))
    }

    /// Parse `options.txt` content. Never fails: unparseable values keep their defaults.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        Self::parse_counting(text).0
    }

    /// [`Self::parse`], plus how many `key:value` lines it read (known keys or not).
    fn parse_counting(text: &str) -> (Self, usize) {
        let mut lines_read = 0;
        let mut s = Self::default();
        // Vanilla clears the sound map before loading so that absent categories fall back to 1.0
        // through `getSoundLevel`; we keep the full map and overwrite what is present instead,
        // which produces identical volumes with no `Option` handling downstream.
        for line in text.lines() {
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                continue;
            }
            // Java uses `String.split(":")`, whose default limit drops trailing empty fields.
            // The branches below deliberately recover the whole remainder with
            // `substring(indexOf(':') + 1)`: Gson owns both resource-pack arrays, and the
            // lastServer/streamPreferredServer values can contain colons too.
            let mut fields: Vec<&str> = line.split(':').collect();
            while fields.last() == Some(&"") {
                fields.pop();
            }
            if fields.len() < 2 || fields[0].is_empty() {
                continue;
            }
            lines_read += 1;
            let key = fields[0];
            let remainder = line
                .split_once(':')
                .expect("a parsed options line has a colon")
                .1;
            let value = match key {
                "resourcePacks"
                | "incompatibleResourcePacks"
                | "lastServer"
                | "streamPreferredServer" => remainder,
                _ => fields[1],
            };

            // Unknown keys are a port extension, rather than Java behaviour. Preserve their
            // complete value so the extension remains lossless even though known Java options
            // use only split field 1.
            let unknown_before = s.unknown.len();
            s.apply(key, value);
            if s.unknown.len() != unknown_before {
                s.unknown.last_mut().expect("entry just pushed").1 = remainder.to_owned();
            }
        }
        (s, lines_read)
    }

    fn apply(&mut self, key: &str, value: &str) {
        // Prefixed families first, so `key_key.forward` is not mistaken for an unknown key.
        if let Some(name) = key.strip_prefix("key_") {
            if let Ok(code) = value.trim().parse::<i32>() {
                if self.keys.get(name).is_some() {
                    self.keys
                        .set_binding(name, crate::input::Input::from_lwjgl(code));
                    return;
                }
            }
            self.unknown.push((key.to_owned(), value.to_owned()));
            return;
        }
        if let Some(name) = key.strip_prefix("soundCategory_") {
            // GameSettings only loops over SoundCategory.values(). Keep an unrecognised
            // extension in the lossless unknown list rather than inserting it into the map and
            // silently dropping it again in the fixed-category writer below.
            if !SOUND_CATEGORIES.contains(&name) {
                self.unknown.push((key.to_owned(), value.to_owned()));
                return;
            }
            if let Some(v) = parse_f32(value) {
                self.sound_levels.insert(name.to_owned(), v.clamp(0.0, 1.0));
                return;
            }
            self.unknown.push((key.to_owned(), value.to_owned()));
            return;
        }
        if let Some(name) = key.strip_prefix("modelPart_") {
            // As with sounds, vanilla recognises only the seven EnumPlayerModelParts values.
            // Unknown model-part keys must not become entries that the writer can never emit.
            if !MODEL_PARTS.contains(&name) {
                self.unknown.push((key.to_owned(), value.to_owned()));
                return;
            }
            self.model_parts.insert(name.to_owned(), value == "true");
            return;
        }

        match key {
            "mouseSensitivity" => set_f32(&mut self.mouse_sensitivity, value),
            "invertYMouse" => self.invert_mouse = value == "true",
            // java: fovSetting = parseFloat(v) * 40 + 70
            "fov" => {
                if let Some(v) = parse_f32(value) {
                    self.fov = v * 40.0 + 70.0;
                }
            }
            "gamma" => set_f32(&mut self.gamma, value),
            "saturation" => set_f32(&mut self.saturation, value),
            "renderDistance" => set_i32(&mut self.render_distance_chunks, value),
            "guiScale" => set_i32(&mut self.gui_scale, value),
            "particles" => set_i32(&mut self.particles, value),
            "bobView" => self.view_bobbing = value == "true",
            "anaglyph3d" => self.anaglyph = value == "true",
            "maxFps" => set_i32(&mut self.limit_framerate, value),
            "fboEnable" => self.fbo_enable = value == "true",
            // java: EnumDifficulty.getDifficultyEnum(id) indexes ID_LOOKUP[id % 4]. Negative
            // values throw and leave the previous value because loadOptions catches the line.
            "difficulty" => {
                if let Some(v) = parse_nonnegative_ordinal(value, 4) {
                    self.difficulty = v;
                }
            }
            "fancyGraphics" => self.fancy_graphics = value == "true",
            "ao" => {
                // java: 1.7 wrote "true"/"false" here, 1.8 writes 0/1/2. Both must load.
                self.ambient_occlusion = match value {
                    "true" => 2,
                    "false" => 0,
                    v => v.trim().parse().unwrap_or(self.ambient_occlusion),
                };
            }
            "renderClouds" => {
                self.clouds = match value {
                    "true" => 2,
                    "fast" => 1,
                    "false" => 0,
                    _ => self.clouds,
                };
            }
            // java: these two branches pass the complete substring after the first colon to
            // Gson, not `astring[1]`; malformed JSON leaves the previous list untouched because
            // loadOptions catches the exception for this line.
            "resourcePacks" => {
                if let Some(packs) = parse_json_string_array(value) {
                    self.resource_packs = packs;
                }
            }
            "incompatibleResourcePacks" => {
                if let Some(packs) = parse_json_string_array(value) {
                    self.incompatible_resource_packs = packs;
                }
            }
            "lastServer" => self.last_server = value.to_owned(),
            "lang" => self.language = value.to_owned(),
            // java: EnumChatVisibility.getEnumChatVisibility(id) wraps non-negative ordinals
            // through ID_LOOKUP[id % 3]; a negative id throws and is ignored for that line.
            "chatVisibility" => {
                if let Some(v) = parse_nonnegative_ordinal(value, 3) {
                    self.chat_visibility = v;
                }
            }
            "chatColors" => self.chat_colors = value == "true",
            "chatLinks" => self.chat_links = value == "true",
            "chatLinksPrompt" => self.chat_links_prompt = value == "true",
            "chatOpacity" => set_f32(&mut self.chat_opacity, value),
            "snooperEnabled" => self.snooper_enabled = value == "true",
            "fullscreen" => self.fullscreen = value == "true",
            "enableVsync" => self.enable_vsync = value == "true",
            "useVbo" => self.use_vbo = value == "true",
            "hideServerAddress" => self.hide_server_address = value == "true",
            "advancedItemTooltips" => self.advanced_item_tooltips = value == "true",
            "pauseOnLostFocus" => self.pause_on_lost_focus = value == "true",
            "touchscreen" => self.touchscreen = value == "true",
            "overrideWidth" => set_i32(&mut self.override_width, value),
            "overrideHeight" => set_i32(&mut self.override_height, value),
            "heldItemTooltips" => self.held_item_tooltips = value == "true",
            "chatHeightFocused" => set_f32(&mut self.chat_height_focused, value),
            "chatHeightUnfocused" => set_f32(&mut self.chat_height_unfocused, value),
            "chatScale" => set_f32(&mut self.chat_scale, value),
            "chatWidth" => set_f32(&mut self.chat_width, value),
            "showInventoryAchievementHint" => {
                self.show_inventory_achievement_hint = value == "true";
            }
            "mipmapLevels" => set_i32(&mut self.mipmap_levels, value),
            "forceUnicodeFont" => self.force_unicode_font = value == "true",
            "allowBlockAlternatives" => self.allow_block_alternatives = value == "true",
            "reducedDebugInfo" => self.reduced_debug_info = value == "true",
            "useNativeTransport" => self.use_native_transport = value == "true",
            "entityShadows" => self.entity_shadows = value == "true",
            // Same "malformed values are ignored" rule: only 1..=3 replace the default.
            "frameLatency" => {
                if let Ok(v @ 1..=crate::context::MAX_FRAME_LATENCY) = value.trim().parse::<u32>() {
                    self.frame_latency = v;
                }
            }
            "realmsNotifications" => self.realms_notifications = value == "true",
            "msaaSamples" => {
                if let Ok(v) = value.trim().parse::<u32>() {
                    self.msaa_samples = match v {
                        2 | 4 | 8 => v,
                        _ => 1,
                    };
                }
            }
            // An unrecognised value keeps the default, matching this file's per-line
            // "malformed values are ignored, never fatal" rule rather than refusing to start.
            "presentMode" => {
                if let Some(mode) = PresentPreference::parse(value) {
                    self.present_mode = mode;
                }
            }
            // Non-vanilla, default-off render options. A malformed value keeps the default.
            "fullbright" => self.fullbright = value == "true",
            "playerRenderDistance" => set_i32(&mut self.player_render_distance, value),
            "entityRenderDistance" => set_i32(&mut self.entity_render_distance, value),
            _ => self.unknown.push((key.to_owned(), value.to_owned())),
        }
    }

    /// Render to `options.txt` form, in vanilla's field order, with unknown keys appended.
    #[must_use]
    pub fn to_options_txt(&self) -> String {
        let mut out = String::with_capacity(2048);
        let mut put = |k: &str, v: &str| {
            out.push_str(k);
            out.push(':');
            out.push_str(v);
            out.push_str(OPTIONS_LINE_ENDING);
        };

        put("invertYMouse", bool_str(self.invert_mouse));
        put("mouseSensitivity", &java_f32(self.mouse_sensitivity));
        put("fov", &java_f32((self.fov - 70.0) / 40.0));
        put("gamma", &java_f32(self.gamma));
        put("saturation", &java_f32(self.saturation));
        put("renderDistance", &self.render_distance_chunks.to_string());
        put("guiScale", &self.gui_scale.to_string());
        put("particles", &self.particles.to_string());
        put("bobView", bool_str(self.view_bobbing));
        put("anaglyph3d", bool_str(self.anaglyph));
        put("maxFps", &self.limit_framerate.to_string());
        put("fboEnable", bool_str(self.fbo_enable));
        put("difficulty", &self.difficulty.rem_euclid(4).to_string());
        put("fancyGraphics", bool_str(self.fancy_graphics));
        put("ao", &self.ambient_occlusion.to_string());
        // java's switch has no default. An out-of-range hand-edited value therefore emits no
        // renderClouds line instead of being silently rewritten as fancy clouds.
        match self.clouds {
            0 => put("renderClouds", "false"),
            1 => put("renderClouds", "fast"),
            2 => put("renderClouds", "true"),
            _ => {}
        }
        put("resourcePacks", &json_string_array(&self.resource_packs));
        put(
            "incompatibleResourcePacks",
            &json_string_array(&self.incompatible_resource_packs),
        );
        put("lastServer", &self.last_server);
        put("lang", &self.language);
        put(
            "chatVisibility",
            &self.chat_visibility.rem_euclid(3).to_string(),
        );
        put("chatColors", bool_str(self.chat_colors));
        put("chatLinks", bool_str(self.chat_links));
        put("chatLinksPrompt", bool_str(self.chat_links_prompt));
        put("chatOpacity", &java_f32(self.chat_opacity));
        put("snooperEnabled", bool_str(self.snooper_enabled));
        put("fullscreen", bool_str(self.fullscreen));
        put("enableVsync", bool_str(self.enable_vsync));
        put("useVbo", bool_str(self.use_vbo));
        put("hideServerAddress", bool_str(self.hide_server_address));
        put(
            "advancedItemTooltips",
            bool_str(self.advanced_item_tooltips),
        );
        put("pauseOnLostFocus", bool_str(self.pause_on_lost_focus));
        put("touchscreen", bool_str(self.touchscreen));
        put("overrideWidth", &self.override_width.to_string());
        put("overrideHeight", &self.override_height.to_string());
        put("heldItemTooltips", bool_str(self.held_item_tooltips));
        put("chatHeightFocused", &java_f32(self.chat_height_focused));
        put("chatHeightUnfocused", &java_f32(self.chat_height_unfocused));
        put("chatScale", &java_f32(self.chat_scale));
        put("chatWidth", &java_f32(self.chat_width));
        put(
            "showInventoryAchievementHint",
            bool_str(self.show_inventory_achievement_hint),
        );
        put("mipmapLevels", &self.mipmap_levels.to_string());
        put("forceUnicodeFont", bool_str(self.force_unicode_font));
        put(
            "allowBlockAlternatives",
            bool_str(self.allow_block_alternatives),
        );
        put("reducedDebugInfo", bool_str(self.reduced_debug_info));
        put("useNativeTransport", bool_str(self.use_native_transport));
        put("entityShadows", bool_str(self.entity_shadows));
        put("realmsNotifications", bool_str(self.realms_notifications));
        put("msaaSamples", &self.msaa_samples.to_string());
        put("presentMode", self.present_mode.as_str());
        put("fullbright", bool_str(self.fullbright));
        put(
            "playerRenderDistance",
            &self.player_render_distance.to_string(),
        );
        put(
            "entityRenderDistance",
            &self.entity_render_distance.to_string(),
        );

        for binding in self.keys.iter() {
            put(
                &format!("key_{}", binding.description()),
                &binding.binding().to_lwjgl().to_string(),
            );
        }
        for name in SOUND_CATEGORIES {
            let level = self.sound_levels.get(name).copied().unwrap_or(1.0);
            put(&format!("soundCategory_{name}"), &java_f32(level));
        }
        for name in MODEL_PARTS {
            let on = self.model_parts.get(name).copied().unwrap_or(true);
            put(&format!("modelPart_{name}"), bool_str(on));
        }
        // Non-vanilla keys this port added after the original set; kept out of the vanilla run.
        put("frameLatency", &self.frame_latency.to_string());
        for (k, v) in &self.unknown {
            put(k, v);
        }
        out
    }

    /// Volume for a sound category, defaulting to full. java: `GameSettings.getSoundLevel`
    #[must_use]
    pub fn sound_level(&self, category: &str) -> f32 {
        self.sound_levels.get(category).copied().unwrap_or(1.0)
    }

    /// Frame cap in FPS, or `None` when `maxFps` is [`FRAMERATE_UNLIMITED`].
    /// java: `Minecraft.isFramerateLimitBelowMax`
    #[must_use]
    pub fn framerate_limit(&self) -> Option<u32> {
        if self.limit_framerate == FRAMERATE_UNLIMITED
            || self.limit_framerate > MAX_FRAMERATE_LIMIT
            || self.limit_framerate <= 0
        {
            None
        } else {
            Some(self.limit_framerate as u32)
        }
    }

    /// The [`GpuOptions`](crate::GpuOptions) implied by these settings.
    #[must_use]
    pub fn gpu_options(&self) -> crate::GpuOptions {
        crate::GpuOptions {
            vsync: self.enable_vsync,
            sample_count: self.msaa_samples,
            low_power: false,
        }
    }

    /// Keys present in the loaded file that this port does not model, preserved for save.
    #[must_use]
    pub fn unknown_keys(&self) -> &[(String, String)] {
        &self.unknown
    }
}

/// Appended to the name of a settings file the client could not use, for the copy kept of it
/// (`options.txt.bad`; the mod settings use the same suffix).
pub const DAMAGED_SUFFIX: &str = ".bad";

/// How many times [`read_patiently`] tries, and the pause between tries.
const READ_ATTEMPTS: u32 = 3;
const READ_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(50);

/// `fs::read`, tried again after a short pause when it fails for a reason other than "missing":
/// an antivirus scan or a backup tool can hold the file for a moment at start-up, and taking that
/// moment for damage would start the session on the defaults, which the first save then writes
/// over the player's file. At most 100 ms, and only when the first read failed.
fn read_patiently(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut attempt = 1;
    loop {
        match std::fs::read(path) {
            Err(error)
                if error.kind() != std::io::ErrorKind::NotFound && attempt < READ_ATTEMPTS =>
            {
                attempt += 1;
                std::thread::sleep(READ_RETRY_DELAY);
            }
            result => return result,
        }
    }
}

/// What [`GameSettings::load_keeping_damaged`] found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionsFileState {
    /// Missing (a first run) or read normally.
    Fine,
    /// The file was damaged: see [`GameSettings::load_keeping_damaged`].
    Damaged {
        /// What was wrong, for the log.
        reason: String,
        /// Where the copy of the damaged file went; `None` if it could not be copied.
        backup: Option<std::path::PathBuf>,
    },
}

/// Decode `options.txt` bytes one line at a time: UTF-8 (what this client writes) where the line
/// is valid UTF-8, otherwise Windows-1252, which is what vanilla's `FileWriter` writes on a
/// Western Windows install. NUL bytes, which only an interrupted write leaves, are dropped, and so
/// is a UTF-8 byte-order mark a text editor may have added (it would hide the first option).
/// Never fails, so one odd byte can no longer cost the whole file.
#[must_use]
pub fn decode_options_bytes(bytes: &[u8]) -> std::borrow::Cow<'_, str> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    if !bytes.contains(&0) {
        if let Ok(text) = std::str::from_utf8(bytes) {
            return std::borrow::Cow::Borrowed(text);
        }
    }
    let mut text = String::with_capacity(bytes.len());
    for line in bytes.split_inclusive(|&byte| byte == b'\n') {
        let kept: Vec<u8> = line.iter().copied().filter(|&byte| byte != 0).collect();
        match std::str::from_utf8(&kept) {
            Ok(valid) => text.push_str(valid),
            Err(_) => text.extend(kept.iter().map(|&byte| windows_1252(byte))),
        }
    }
    std::borrow::Cow::Owned(text)
}

/// One Windows-1252 byte as a `char`, as Java's `Cp1252` decoder maps it (the five unassigned
/// bytes become U+FFFD).
fn windows_1252(byte: u8) -> char {
    const HIGH: [char; 32] = [
        '\u{20AC}', '\u{FFFD}', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}',
        '\u{2021}', '\u{02C6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{FFFD}',
        '\u{017D}', '\u{FFFD}', '\u{FFFD}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}',
        '\u{2022}', '\u{2013}', '\u{2014}', '\u{02DC}', '\u{2122}', '\u{0161}', '\u{203A}',
        '\u{0153}', '\u{FFFD}', '\u{017E}', '\u{0178}',
    ];
    match byte {
        0x80..=0x9F => HIGH[usize::from(byte - 0x80)],
        _ => char::from(byte),
    }
}

fn bool_str(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

#[cfg(windows)]
const OPTIONS_LINE_ENDING: &str = "\r\n";
#[cfg(not(windows))]
const OPTIONS_LINE_ENDING: &str = "\n";

/// Format like Java's `Float.toString`: shortest round-trip decimal, decimal point retained,
/// and scientific notation below `1e-3` or at `1e7`.
fn java_f32(v: f32) -> String {
    let raw = v.to_string();
    match raw.as_str() {
        "NaN" => return raw,
        "inf" => return "Infinity".into(),
        "-inf" => return "-Infinity".into(),
        _ => {}
    }

    let (sign, raw) = raw
        .strip_prefix('-')
        .map_or(("", raw.as_str()), |unsigned| ("-", unsigned));
    let (mantissa, explicit_exponent) = raw
        .split_once(['e', 'E'])
        .map_or((raw, 0), |(digits, exponent)| {
            (digits, exponent.parse::<i32>().expect("f32 exponent"))
        });
    let point = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let mut digits: String = mantissa.chars().filter(|&ch| ch != '.').collect();
    let leading_zeroes = digits.bytes().take_while(|&byte| byte == b'0').count();
    let decimal_position = point + explicit_exponent - leading_zeroes as i32;
    digits.drain(..leading_zeroes);

    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }

    let scientific_exponent = decimal_position - 1;
    if (-3..7).contains(&scientific_exponent) {
        if decimal_position <= 0 {
            format!(
                "{sign}0.{}{digits}",
                "0".repeat((-decimal_position) as usize)
            )
        } else {
            let decimal_position = decimal_position as usize;
            if decimal_position >= digits.len() {
                format!(
                    "{sign}{digits}{}.0",
                    "0".repeat(decimal_position - digits.len())
                )
            } else {
                format!(
                    "{sign}{}.{}",
                    &digits[..decimal_position],
                    &digits[decimal_position..]
                )
            }
        }
    } else {
        let fraction = if digits.len() == 1 { "0" } else { &digits[1..] };
        format!("{sign}{}.{fraction}E{scientific_exponent}", &digits[..1])
    }
}

fn parse_f32(v: &str) -> Option<f32> {
    // java: `GameSettings.parseFloat` treats the two legacy boolean spellings specially before
    // delegating every other value to `Float.parseFloat`.  Keep the existing finite-number
    // policy for every non-boolean input; only the exact Java literals gain this tolerance.
    match v {
        "true" => Some(1.0),
        "false" => Some(0.0),
        _ => v.trim().parse::<f32>().ok().filter(|f| f.is_finite()),
    }
}

fn set_f32(slot: &mut f32, v: &str) {
    if let Some(parsed) = parse_f32(v) {
        *slot = parsed;
    }
}

fn set_i32(slot: &mut i32, v: &str) {
    if let Ok(parsed) = v.trim().parse::<i32>() {
        *slot = parsed;
    }
}

/// Java's enum lookup for the non-negative ordinals used by options.txt.
///
/// `EnumDifficulty.getDifficultyEnum` and `EnumChatVisibility.getEnumChatVisibility` both use
/// `ID_LOOKUP[id % len]`. Negative values throw in Java and are ignored by the per-line load
/// catch, so they deliberately return `None` here rather than using `rem_euclid`.
fn parse_nonnegative_ordinal(v: &str, len: i32) -> Option<i32> {
    let parsed = v.parse::<i32>().ok()?;
    (parsed >= 0).then(|| parsed % len)
}

/// Parse the flat `["a","b"]` array Gson writes for `resourcePacks`.
///
/// This is deliberately not a general JSON parser: the values vanilla writes here are an array
/// of strings, and pulling `serde_json` into the renderer to read two lines is not a good trade.
/// `None` means Gson would reject the value (so Java leaves the previous list intact); `Some([])`
/// covers both `[]` and Gson's `null` result, which Java turns into a new empty list.
fn parse_json_string_array(v: &str) -> Option<Vec<String>> {
    let trimmed = v.trim();
    if trimmed == "null" {
        return Some(Vec::new());
    }
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))?;

    let mut out = Vec::new();
    let mut chars = inner.chars().peekable();
    loop {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.peek().is_none() {
            return Some(out);
        }
        if chars.next() != Some('"') {
            return None;
        }

        let mut current = String::new();
        let mut closed = false;
        while let Some(c) = chars.next() {
            match c {
                '"' => {
                    closed = true;
                    break;
                }
                '\\' => {
                    let escaped = chars.next()?;
                    match escaped {
                        '"' | '\\' | '/' => current.push(escaped),
                        'b' => current.push('\u{0008}'),
                        'f' => current.push('\u{000c}'),
                        'n' => current.push('\n'),
                        'r' => current.push('\r'),
                        't' => current.push('\t'),
                        'u' => {
                            let mut value = 0u16;
                            for _ in 0..4 {
                                let digit = chars.next()?.to_digit(16)? as u16;
                                value = value * 16 + digit;
                            }
                            let codepoint = u32::from(value);
                            // Gson decodes ordinary BMP escapes. Reject lone surrogates rather
                            // than manufacturing invalid UTF-16 that Rust strings cannot hold.
                            current.push(char::from_u32(codepoint)?);
                        }
                        _ => return None,
                    }
                }
                c if c.is_control() => return None,
                _ => current.push(c),
            }
        }
        if !closed {
            return None;
        }
        out.push(current);

        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        match chars.next() {
            None => return Some(out),
            Some(',') => {}
            Some(_) => return None,
        }
    }
}

fn json_string_array(items: &[String]) -> String {
    let mut out = String::from("[");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        for c in item.chars() {
            match c {
                '"' | '\\' => {
                    out.push('\\');
                    out.push(c);
                }
                '\u{0008}' => out.push_str("\\b"),
                '\u{000c}' => out.push_str("\\f"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c.is_control() => {
                    use std::fmt::Write;
                    write!(out, "\\u{:04x}", c as u32).expect("writing to String cannot fail");
                }
                _ => out.push(c),
            }
        }
        out.push('"');
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::keys;

    /// A real vanilla 1.8.9 `options.txt`, including the Twitch keys this port does not model.
    const VANILLA_SAMPLE: &str = "\
invertYMouse:false
mouseSensitivity:0.7176471
fov:0.25
gamma:1.0
saturation:0.0
renderDistance:12
guiScale:2
particles:0
bobView:true
anaglyph3d:false
maxFps:120
fboEnable:true
difficulty:2
fancyGraphics:true
ao:2
renderClouds:fast
resourcePacks:[\"Clean 16x.zip\",\"my pack.zip\"]
incompatibleResourcePacks:[]
lastServer:play.example.com:25565
lang:en_GB
chatVisibility:0
chatColors:true
chatLinks:true
chatLinksPrompt:true
chatOpacity:1.0
snooperEnabled:false
fullscreen:false
enableVsync:false
useVbo:true
hideServerAddress:false
advancedItemTooltips:false
pauseOnLostFocus:true
touchscreen:false
overrideWidth:0
overrideHeight:0
heldItemTooltips:true
chatHeightFocused:1.0
chatHeightUnfocused:0.44366196
chatScale:1.0
chatWidth:1.0
showInventoryAchievementHint:true
mipmapLevels:4
streamBytesPerPixel:0.5
streamMicVolume:1.0
streamSystemVolume:1.0
streamKbps:0.5412844
streamFps:0.31690142
streamCompression:1
streamSendMetadata:true
streamPreferredServer:
streamChatEnabled:0
streamChatUserFilter:0
streamMicToggleBehavior:0
forceUnicodeFont:false
allowBlockAlternatives:true
reducedDebugInfo:false
useNativeTransport:true
entityShadows:true
realmsNotifications:true
key_key.attack:-100
key_key.use:-99
key_key.forward:17
key_key.left:30
key_key.back:31
key_key.right:32
key_key.jump:57
key_key.sneak:42
key_key.sprint:29
key_key.drop:16
key_key.inventory:18
key_key.chat:20
key_key.playerlist:15
key_key.pickItem:-98
key_key.command:53
key_key.screenshot:60
key_key.togglePerspective:63
key_key.smoothCamera:0
key_key.streamStartStop:64
key_key.streamPauseUnpause:65
key_key.streamCommercial:0
key_key.streamToggleMic:0
key_key.fullscreen:87
key_key.spectatorOutlines:0
key_key.hotbar.1:2
key_key.hotbar.2:3
key_key.hotbar.3:4
key_key.hotbar.4:5
key_key.hotbar.5:6
key_key.hotbar.6:7
key_key.hotbar.7:8
key_key.hotbar.8:9
key_key.hotbar.9:10
soundCategory_master:0.5
soundCategory_music:0.0
soundCategory_record:1.0
soundCategory_weather:1.0
soundCategory_block:1.0
soundCategory_hostile:1.0
soundCategory_neutral:1.0
soundCategory_player:1.0
soundCategory_ambient:1.0
modelPart_cape:true
modelPart_jacket:true
modelPart_left_sleeve:true
modelPart_right_sleeve:true
modelPart_left_pants_leg:true
modelPart_right_pants_leg:true
modelPart_hat:false
";

    #[test]
    fn parses_a_real_vanilla_file() {
        let s = GameSettings::parse(VANILLA_SAMPLE);
        assert!((s.mouse_sensitivity - 0.717_647_1).abs() < 1e-6);
        assert_eq!(s.render_distance_chunks, 12);
        assert_eq!(s.gui_scale, 2);
        assert_eq!(s.clouds, 1); // "fast"
        assert!(!s.enable_vsync);
        assert_eq!(s.language, "en_GB");
        assert_eq!(s.resource_packs, ["Clean 16x.zip", "my pack.zip"]);
        assert!(s.incompatible_resource_packs.is_empty());
        assert!((s.sound_level("master") - 0.5).abs() < 1e-6);
        assert_eq!(s.sound_level("music"), 0.0);
        assert!(!s.model_parts["hat"]);
        assert!(s.model_parts["cape"]);
    }

    #[test]
    fn fov_uses_vanillas_stored_encoding() {
        // java: fovSetting = v * 40 + 70, saved back as (fov - 70) / 40
        let s = GameSettings::parse("fov:0.25\n");
        assert!((s.fov - 80.0).abs() < 1e-4, "{}", s.fov);
        assert!(s
            .to_options_txt()
            .contains(&format!("fov:0.25{OPTIONS_LINE_ENDING}")));

        let default = GameSettings::default();
        assert_eq!(default.fov, 70.0);
        assert!(default
            .to_options_txt()
            .contains(&format!("fov:0.0{OPTIONS_LINE_ENDING}")));
    }

    #[test]
    fn values_containing_colons_survive() {
        // GameSettings's lastServer branch deliberately uses the complete post-colon remainder.
        let s = GameSettings::parse(VANILLA_SAMPLE);
        assert_eq!(s.last_server, "play.example.com:25565");
        assert!(s.to_options_txt().contains(&format!(
            "lastServer:play.example.com:25565{OPTIONS_LINE_ENDING}"
        )));
    }

    #[test]
    fn resource_pack_arrays_use_the_complete_post_colon_remainder() {
        // Java passes substring(indexOf(':') + 1) to Gson here. Splitting on every colon would
        // turn the second entry into a malformed/truncated pack name.
        let s = GameSettings::parse(
            r#"resourcePacks:["pack:one.zip","https://example.invalid/pack.zip"]
incompatibleResourcePacks:["legacy:pack.zip"]
"#,
        );
        assert_eq!(
            s.resource_packs,
            ["pack:one.zip", "https://example.invalid/pack.zip"]
        );
        assert_eq!(s.incompatible_resource_packs, ["legacy:pack.zip"]);

        // Gson throws on malformed JSON; GameSettings catches that line and retains the value
        // loaded immediately before it instead of silently clearing the list.
        let retained = GameSettings::parse(
            r#"resourcePacks:["good:pack.zip"]
resourcePacks:[broken
"#,
        );
        assert_eq!(retained.resource_packs, ["good:pack.zip"]);
    }

    #[test]
    fn unknown_sound_and_model_part_prefixes_round_trip() {
        let s = GameSettings::parse("soundCategory_custom:0.25\nmodelPart_tail:true\n");
        assert_eq!(s.sound_level("custom"), 1.0);
        assert!(s
            .unknown_keys()
            .contains(&("soundCategory_custom".into(), "0.25".into())));
        assert!(s
            .unknown_keys()
            .contains(&("modelPart_tail".into(), "true".into())));
        let written = s.to_options_txt();
        assert!(written.contains(&format!("soundCategory_custom:0.25{OPTIONS_LINE_ENDING}")));
        assert!(written.contains(&format!("modelPart_tail:true{OPTIONS_LINE_ENDING}")));
    }

    #[test]
    fn enum_option_ordinals_wrap_like_java_and_reject_negative_ids() {
        let s = GameSettings::parse(
            "difficulty:5\nchatVisibility:4\ndifficulty:-1\nchatVisibility:-1\n",
        );
        assert_eq!(s.difficulty, 1, "5 % EnumDifficulty.values().length");
        assert_eq!(
            s.chat_visibility, 1,
            "4 % EnumChatVisibility.values().length"
        );
        assert!(s
            .to_options_txt()
            .contains(&format!("difficulty:1{OPTIONS_LINE_ENDING}")));
        assert!(s
            .to_options_txt()
            .contains(&format!("chatVisibility:1{OPTIONS_LINE_ENDING}")));
    }

    #[test]
    fn invalid_cloud_enum_values_are_not_saved_as_fancy() {
        let s = GameSettings {
            clouds: 99,
            ..GameSettings::default()
        };
        assert!(!s
            .to_options_txt()
            .lines()
            .any(|line| line.starts_with("renderClouds:")));
    }

    #[test]
    fn options_parser_uses_java_split_rules_but_preserves_last_server_remainder() {
        let s = GameSettings::parse(
            "lang:\nlastServer:\nmouseSensitivity:0.5:junk\nlang:fr_FR:ignored\nlastServer:play.example.com:25565\n:::\nfutureOption:still:here\n",
        );

        // Trailing empty fields disappear under Java String.split, so constructor values remain.
        assert_eq!(s.language, "fr_FR");
        assert_eq!(s.last_server, "play.example.com:25565");
        // Normal branches take field 1, unlike lastServer's explicit substring branch.
        assert_eq!(s.mouse_sensitivity, 0.5);
        // A colon-only line throws inside Java's per-line catch and does not become an option.
        assert!(s.unknown_keys().iter().all(|(key, _)| !key.is_empty()));
        // Keeping unmodelled extension keys is intentional, and should remain lossless.
        assert!(s
            .unknown_keys()
            .contains(&("futureOption".into(), "still:here".into())));
    }

    #[test]
    fn key_bindings_load_from_disk() {
        let s = GameSettings::parse("key_key.forward:200\nkey_key.jump:-100\n");
        assert_eq!(
            s.keys.get(keys::FORWARD).unwrap().binding(),
            crate::input::Input::Key(winit::keyboard::KeyCode::ArrowUp)
        );
        assert_eq!(
            s.keys.get(keys::JUMP).unwrap().binding(),
            crate::input::Input::MOUSE_LEFT
        );
    }

    #[test]
    fn unknown_keys_are_preserved_not_dropped() {
        let sample = format!("{VANILLA_SAMPLE}futureOption:still-here\n");
        let s = GameSettings::parse(&sample);
        let unknown: Vec<&str> = s.unknown_keys().iter().map(|(k, _)| k.as_str()).collect();
        assert!(unknown.contains(&"streamKbps"));
        assert!(unknown.contains(&"futureOption"));
        assert!(!unknown.contains(&"key_key.streamStartStop"));

        let written = s.to_options_txt();
        assert!(written.contains(&format!("streamKbps:0.5412844{OPTIONS_LINE_ENDING}")));
        assert!(written.contains(&format!("key_key.streamStartStop:64{OPTIONS_LINE_ENDING}")));
        assert!(written.contains(&format!("futureOption:still-here{OPTIONS_LINE_ENDING}")));
        assert!(
            !written.contains("streamPreferredServer:"),
            "Java split drops the empty trailing value, so it is not an unknown-key round trip"
        );
    }

    #[test]
    fn unknown_lwjgl_keycodes_round_trip_through_options_txt() {
        let settings = GameSettings::parse("key_key.forward:144\n");
        assert_eq!(
            settings.keys.get(keys::FORWARD).unwrap().binding(),
            crate::input::Input::RawLwjgl(144)
        );
        assert!(settings
            .to_options_txt()
            .contains(&format!("key_key.forward:144{OPTIONS_LINE_ENDING}")));
    }

    #[test]
    fn save_load_round_trips_byte_for_byte() {
        let first = GameSettings::parse(VANILLA_SAMPLE).to_options_txt();
        let second = GameSettings::parse(&first).to_options_txt();
        assert_eq!(first, second);
    }

    #[test]
    fn defaults_round_trip() {
        let a = GameSettings::default().to_options_txt();
        let b = GameSettings::parse(&a).to_options_txt();
        assert_eq!(a, b);
    }

    #[test]
    fn smooth_camera_is_a_session_flag_not_an_options_txt_key() {
        let settings = GameSettings {
            smooth_camera: true,
            ..Default::default()
        };
        assert!(settings.smooth_camera);
        assert!(
            settings
                .to_options_txt()
                .lines()
                .all(|line| !line.starts_with("smoothCamera:")),
            "1.8.9's GameSettings does not persist this field"
        );

        // A stray line is not treated as a supported persisted option either; preservation is
        // this port's generic forward-compatibility behavior, not an accidental load path.
        let loaded = GameSettings::parse("smoothCamera:true\n");
        assert!(!loaded.smooth_camera);
        assert_eq!(
            loaded.unknown_keys(),
            [("smoothCamera".into(), "true".into())]
        );
    }

    #[test]
    fn garbage_lines_do_not_break_the_load() {
        let s = GameSettings::parse(
            "mouseSensitivity:not-a-number\nrenderDistance:\n\n:::\nno-colon-at-all\nguiScale:3\n",
        );
        assert_eq!(s.mouse_sensitivity, 0.5); // default kept
        assert_eq!(
            s.render_distance_chunks,
            native_default_render_distance(),
            "a malformed value keeps the bitness-derived constructor default"
        );
        assert_eq!(s.gui_scale, 3); // later good line still applies
    }

    #[test]
    fn java_float_parser_accepts_only_exact_legacy_boolean_literals() {
        // java: `GameSettings.parseFloat` maps these pre-1.8 spellings before calling
        // `Float.parseFloat`; every setting which takes a float shares that helper.
        let boolean = GameSettings::parse(
            "mouseSensitivity:true\ngamma:false\nfov:true\nchatOpacity:false\nsoundCategory_music:true\n",
        );
        assert_eq!(boolean.mouse_sensitivity, 1.0);
        assert_eq!(boolean.gamma, 0.0);
        assert_eq!(boolean.fov, 110.0);
        assert_eq!(boolean.chat_opacity, 0.0);
        assert_eq!(boolean.sound_level("music"), 1.0);

        // Ordinary finite numbers retain the existing parser behavior.
        let numeric = GameSettings::parse("mouseSensitivity:0.625\ngamma:0.25\n");
        assert_eq!(numeric.mouse_sensitivity, 0.625);
        assert_eq!(numeric.gamma, 0.25);

        // The Java tolerance is exact: casing and surrounding whitespace are not booleans, and
        // malformed/non-finite values must continue to leave the constructor defaults alone.
        let invalid = GameSettings::parse(
            "mouseSensitivity:TRUE\ngamma: true \nfov:not-a-number\nchatOpacity:NaN\n",
        );
        assert_eq!(
            invalid.mouse_sensitivity,
            GameSettings::default().mouse_sensitivity
        );
        assert_eq!(invalid.gamma, GameSettings::default().gamma);
        assert_eq!(invalid.fov, GameSettings::default().fov);
        assert_eq!(invalid.chat_opacity, GameSettings::default().chat_opacity);
    }

    #[test]
    fn legacy_boolean_ao_still_loads() {
        // java: 1.7 wrote ao:true / ao:false before smooth lighting gained a third level.
        assert_eq!(GameSettings::parse("ao:true\n").ambient_occlusion, 2);
        assert_eq!(GameSettings::parse("ao:false\n").ambient_occlusion, 0);
        assert_eq!(GameSettings::parse("ao:1\n").ambient_occlusion, 1);
    }

    #[test]
    fn missing_file_yields_defaults() {
        let path = std::env::temp_dir().join("mc-render-no-such-options-file.txt");
        let _ = std::fs::remove_file(&path);
        let s = GameSettings::load(&path).expect("missing file must not be an error");
        assert_eq!(s.mouse_sensitivity, 0.5);
    }

    #[test]
    fn save_then_load_from_disk() {
        // A directory of its own: a fixed name under the system temp dir is shared with every
        // other process running this test at the same time, and one run's cleanup could delete
        // the file between another's save and load.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("options.txt");
        let mut s = GameSettings {
            render_distance_chunks: 16,
            msaa_samples: 4,
            ..GameSettings::default()
        };
        s.keys
            .set_binding(keys::JUMP, crate::input::Input::MOUSE_RIGHT);
        s.save(&path).expect("save");

        let loaded = GameSettings::load(&path).expect("load");
        assert_eq!(loaded.render_distance_chunks, 16);
        assert_eq!(loaded.msaa_samples, 4);
        assert_eq!(
            loaded.keys.get(keys::JUMP).unwrap().binding(),
            crate::input::Input::MOUSE_RIGHT
        );
    }

    #[test]
    fn framerate_limit_treats_260_as_unlimited() {
        let mut s = GameSettings {
            limit_framerate: FRAMERATE_UNLIMITED,
            ..GameSettings::default()
        };
        assert_eq!(s.framerate_limit(), None);
        s.limit_framerate = 60;
        assert_eq!(s.framerate_limit(), Some(60));
    }

    #[test]
    fn extended_framerate_caps_survive_options_round_trip() {
        for cap in [250, 300, 500, 1000, 1500, 2000] {
            let settings = GameSettings::parse(&format!("maxFps:{cap}\n"));
            assert_eq!(settings.framerate_limit(), Some(cap as u32));
            assert_eq!(
                GameSettings::parse(&settings.to_options_txt()).framerate_limit(),
                Some(cap as u32)
            );
        }
        for cap in [0, -1, FRAMERATE_UNLIMITED, MAX_FRAMERATE_LIMIT + 1] {
            assert_eq!(
                GameSettings::parse(&format!("maxFps:{cap}\n")).framerate_limit(),
                None
            );
        }
    }

    #[test]
    fn msaa_falls_back_for_illegal_values() {
        assert_eq!(GameSettings::parse("msaaSamples:3\n").msaa_samples, 1);
        assert_eq!(GameSettings::parse("msaaSamples:4\n").msaa_samples, 4);
    }

    /// T3.2. The key is ours, not vanilla's, so this pins both halves of the round trip and the
    /// default — which must stay `auto`, because `auto` is the pre-T3.2 behaviour.
    #[test]
    fn present_mode_round_trips_and_defaults_to_auto() {
        assert_eq!(
            GameSettings::default().present_mode,
            PresentPreference::Auto
        );
        assert!(GameSettings::default()
            .to_options_txt()
            .contains(&format!("presentMode:auto{OPTIONS_LINE_ENDING}")));

        for pref in PresentPreference::ALL {
            let text = format!("presentMode:{}\n", pref.as_str());
            let parsed = GameSettings::parse(&text);
            assert_eq!(parsed.present_mode, pref, "{text:?}");
            assert!(
                parsed.to_options_txt().contains(&format!(
                    "presentMode:{}{OPTIONS_LINE_ENDING}",
                    pref.as_str()
                )),
                "{text:?}"
            );
        }
    }

    /// A vanilla 1.8.9 file has no `presentMode` line at all, and must therefore keep the
    /// behaviour it had before the key existed rather than picking up a new one.
    #[test]
    fn a_file_without_present_mode_keeps_auto() {
        let s = GameSettings::parse(VANILLA_SAMPLE);
        assert_eq!(s.present_mode, PresentPreference::Auto);
        assert!(
            !s.unknown_keys().iter().any(|(k, _)| k == "presentMode"),
            "presentMode must be a modelled key, not fall through to the unknown list"
        );
    }

    /// The three non-vanilla render options default off, round-trip, ignore malformed values,
    /// and stay off for a vanilla 1.8.9 file that has never heard of them.
    #[test]
    fn non_vanilla_render_options_default_off_and_round_trip() {
        let defaults = GameSettings::default();
        assert!(!defaults.fullbright);
        assert_eq!(defaults.player_render_distance, 0);
        assert_eq!(defaults.entity_render_distance, 0);
        let written = defaults.to_options_txt();
        for line in [
            "fullbright:false",
            "playerRenderDistance:0",
            "entityRenderDistance:0",
        ] {
            assert!(
                written.contains(&format!("{line}{OPTIONS_LINE_ENDING}")),
                "{line}"
            );
        }

        let parsed = GameSettings::parse(
            "fullbright:true\nplayerRenderDistance:96\nentityRenderDistance:48\n",
        );
        assert!(parsed.fullbright);
        assert_eq!(parsed.player_render_distance, 96);
        assert_eq!(parsed.entity_render_distance, 48);
        let again = GameSettings::parse(&parsed.to_options_txt());
        assert!(again.fullbright);
        assert_eq!(again.player_render_distance, 96);
        assert_eq!(again.entity_render_distance, 48);

        let malformed = GameSettings::parse(
            "fullbright:yes\nplayerRenderDistance:far\nentityRenderDistance:1.5\n",
        );
        assert!(!malformed.fullbright);
        assert_eq!(malformed.player_render_distance, 0);
        assert_eq!(malformed.entity_render_distance, 0);

        let vanilla = GameSettings::parse(VANILLA_SAMPLE);
        assert!(!vanilla.fullbright);
        assert_eq!(vanilla.player_render_distance, 0);
        assert_eq!(vanilla.entity_render_distance, 0);
        assert!(!parsed.unknown_keys().iter().any(|(key, _)| {
            matches!(
                key.as_str(),
                "fullbright" | "playerRenderDistance" | "entityRenderDistance"
            )
        }));
    }

    #[test]
    fn an_unrecognised_present_mode_keeps_the_default() {
        // Same rule as every other value in this file: ignored, never fatal.
        assert_eq!(
            GameSettings::parse("presentMode:triple-buffered\n").present_mode,
            PresentPreference::Auto
        );
        // ...and a good line after a bad one still applies.
        assert_eq!(
            GameSettings::parse("presentMode:nonsense\npresentMode:mailbox\n").present_mode,
            PresentPreference::Mailbox
        );
    }

    /// `frameLatency` (PRESENT-5): absent or malformed keeps 2, the latency the windowed path
    /// always ran with; 1..=3 round-trip; it is modelled, not an unknown key.
    #[test]
    fn frame_latency_defaults_to_two_and_round_trips() {
        assert_eq!(GameSettings::default().frame_latency, 2);
        assert_eq!(GameSettings::parse(VANILLA_SAMPLE).frame_latency, 2);
        for bad in ["0", "4", "-1", "fast", ""] {
            assert_eq!(
                GameSettings::parse(&format!("frameLatency:{bad}\n")).frame_latency,
                2,
                "{bad:?}"
            );
        }
        for good in 1..=3u32 {
            let parsed = GameSettings::parse(&format!("frameLatency:{good}\n"));
            assert_eq!(parsed.frame_latency, good);
            assert!(!parsed
                .unknown_keys()
                .iter()
                .any(|(k, _)| k == "frameLatency"));
            assert!(parsed
                .to_options_txt()
                .contains(&format!("frameLatency:{good}{OPTIONS_LINE_ENDING}")));
        }
    }

    #[test]
    fn json_array_helpers_round_trip() {
        let items = vec!["a".to_string(), "b \"quoted\"".to_string()];
        let text = json_string_array(&items);
        assert_eq!(parse_json_string_array(&text), Some(items));
        assert_eq!(parse_json_string_array("[]"), Some(Vec::<String>::new()));
        assert_eq!(parse_json_string_array("null"), Some(Vec::<String>::new()));
        assert_eq!(parse_json_string_array("garbage"), None);
        assert_eq!(parse_json_string_array("[broken]"), None);
    }

    #[test]
    fn java_float_formatting_keeps_java_notation_boundaries() {
        assert_eq!(java_f32(1.0), "1.0");
        assert_eq!(java_f32(0.0), "0.0");
        assert_eq!(java_f32(0.5), "0.5");
        assert_eq!(java_f32(0.443_661_96), "0.44366196");
        assert_eq!(java_f32(0.001), "0.001");
        assert_eq!(java_f32(0.0005), "5.0E-4");
        assert_eq!(java_f32(10_000_000.0), "1.0E7");
    }

    #[test]
    fn options_writer_uses_the_platform_separator() {
        let text = GameSettings::default().to_options_txt();
        assert!(text.ends_with(OPTIONS_LINE_ENDING));
        assert!(!text.contains("\n") || OPTIONS_LINE_ENDING == "\n" || text.contains("\r\n"));
    }

    /// Java uses a conjunction and a decimal-billion threshold. The neighboring cases ensure a
    /// mistaken `||`, `>`, or `1 << 30` implementation cannot pass.
    #[test]
    fn render_distance_max_matches_bitness_and_heap_boundary() {
        assert_eq!(
            render_distance_max_for(false, u64::MAX),
            RENDER_DISTANCE_MAX_NORMAL,
            "memory alone cannot enable Extreme on a 32-bit process"
        );
        assert_eq!(
            render_distance_max_for(true, RENDER_DISTANCE_EXTREME_MEMORY_BYTES - 1),
            RENDER_DISTANCE_MAX_NORMAL,
            "one byte below Java's threshold stays at 16"
        );
        assert_eq!(
            render_distance_max_for(true, RENDER_DISTANCE_EXTREME_MEMORY_BYTES),
            RENDER_DISTANCE_MAX_EXTREME,
            "the threshold itself passes because Java uses >="
        );
        assert_eq!(
            render_distance_max_for(true, 1 << 30),
            RENDER_DISTANCE_MAX_EXTREME,
            "a conventional one-gibibyte launcher heap also passes"
        );
    }

    #[test]
    fn native_render_distance_policy_follows_target_pointer_width() {
        assert_eq!(default_render_distance_for(false), 8);
        assert_eq!(default_render_distance_for(true), 12);
        let expected_max = if cfg!(target_pointer_width = "64") {
            32
        } else {
            16
        };
        let expected_default = if cfg!(target_pointer_width = "64") {
            12
        } else {
            8
        };
        assert_eq!(native_render_distance_max(), expected_max);
        assert_eq!(native_default_render_distance(), expected_default);
        assert_eq!(
            GameSettings::default().render_distance_chunks,
            expected_default
        );
    }

    /// One byte that is not UTF-8 (here vanilla's Windows-1252 `é` in a server
    /// address) used to fail the whole load, the game fell back to the defaults, and the next
    /// save wrote them over the player's keybinds and sensitivity.
    #[test]
    fn a_non_utf8_byte_costs_one_line_not_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("options.txt");
        let mut bytes = b"mouseSensitivity:0.8\nguiScale:3\nlastServer:caf".to_vec();
        bytes.extend_from_slice(b"\xE9.example.net\nkey_key.jump:45\n");
        std::fs::write(&path, &bytes).expect("write");
        // Negative control: the old reader refused the file outright.
        assert!(std::fs::read_to_string(&path).is_err());

        let loaded = GameSettings::load(&path).expect("encoding is never a load error");
        assert_eq!(loaded.mouse_sensitivity, 0.8);
        assert_eq!(loaded.gui_scale, 3);
        assert_eq!(
            loaded.last_server, "café.example.net",
            "decoded as Windows-1252"
        );
        assert_eq!(
            loaded.keys.get(keys::JUMP).unwrap().binding().to_lwjgl(),
            45
        );
        let (kept, state) = GameSettings::load_keeping_damaged(&path);
        assert_eq!(
            state,
            OptionsFileState::Fine,
            "a foreign encoding is not damage"
        );
        assert_eq!(kept.mouse_sensitivity, 0.8);
    }

    #[test]
    fn the_decoder_keeps_utf8_lines_and_maps_windows_1252_ones() {
        // A UTF-8 line survives next to a Windows-1252 one: decoding is per line.
        let bytes = "lastServer:ü.example\n"
            .bytes()
            .chain(*b"lang:\x80\x81\x9F\n");
        let text = decode_options_bytes(&bytes.collect::<Vec<u8>>()).into_owned();
        assert_eq!(
            text,
            "lastServer:ü.example\nlang:\u{20AC}\u{FFFD}\u{0178}\n"
        );
        // A byte-order mark no longer hides the first option.
        let bom = GameSettings::parse(&decode_options_bytes(b"\xEF\xBB\xBFguiScale:2\n"));
        assert_eq!(bom.gui_scale, 2);
        assert!(bom.unknown_keys().is_empty(), "{:?}", bom.unknown_keys());
        // Plain UTF-8 is borrowed, not copied.
        assert!(matches!(
            decode_options_bytes(b"guiScale:2\n"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    /// What a power cut during a non-atomic save leaves: zeros. The readable lines are kept and
    /// the damaged file is copied aside before the next save can overwrite it.
    #[test]
    fn a_file_with_nul_bytes_is_recovered_and_kept_aside() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("options.txt");
        let mut bytes = b"mouseSensitivity:0.8\nguiScale:3\n".to_vec();
        bytes.extend_from_slice(&[0; 64]);
        bytes.extend_from_slice(b"\0\0gamma:0.25\n");
        std::fs::write(&path, &bytes).expect("write");

        let (loaded, state) = GameSettings::load_keeping_damaged(&path);
        assert_eq!(loaded.mouse_sensitivity, 0.8);
        assert_eq!(loaded.gui_scale, 3);
        assert_eq!(loaded.gamma, 0.25, "a line after the zeros is still read");
        assert!(loaded.unknown_keys().is_empty(), "no key made of NULs");
        let backup = dir.path().join("options.txt.bad");
        assert_eq!(
            state,
            OptionsFileState::Damaged {
                reason: "holds NUL bytes (an interrupted write)".into(),
                backup: Some(backup.clone()),
            }
        );
        assert_eq!(std::fs::read(&backup).expect("backup"), bytes);
        assert!(!loaded.to_options_txt().contains('\0'));
    }

    #[test]
    fn a_file_of_zeros_or_without_options_is_damaged_and_a_normal_one_is_not() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("options.txt");
        std::fs::write(&path, [0_u8; 4096]).expect("zeros");
        let (loaded, state) = GameSettings::load_keeping_damaged(&path);
        assert_eq!(
            loaded.mouse_sensitivity,
            GameSettings::default().mouse_sensitivity
        );
        assert!(matches!(
            state,
            OptionsFileState::Damaged {
                backup: Some(_),
                ..
            }
        ));

        std::fs::write(&path, b"this is not an options file\n").expect("prose");
        let (_, state) = GameSettings::load_keeping_damaged(&path);
        assert!(
            matches!(state, OptionsFileState::Damaged { .. }),
            "{state:?}"
        );

        // Negative controls: a normal file, a blank one and a missing one are fine, and make no
        // backup.
        std::fs::remove_file(dir.path().join("options.txt.bad")).expect("clear backup");
        GameSettings::default().save(&path).expect("save");
        assert_eq!(
            GameSettings::load_keeping_damaged(&path).1,
            OptionsFileState::Fine
        );
        std::fs::write(&path, b"\r\n\n").expect("blank");
        assert_eq!(
            GameSettings::load_keeping_damaged(&path).1,
            OptionsFileState::Fine
        );
        std::fs::remove_file(&path).expect("remove");
        assert_eq!(
            GameSettings::load_keeping_damaged(&path).1,
            OptionsFileState::Fine
        );
        assert!(!dir.path().join("options.txt.bad").exists());
    }

    /// A file another program holds for a moment at start-up (an antivirus scan) is read once it
    /// is free instead of being taken for damage. Negative control: held for longer than the
    /// retries, it is damaged.
    #[cfg(windows)]
    #[test]
    fn a_file_held_for_a_moment_is_read_once_it_is_free() {
        use std::os::windows::fs::OpenOptionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("options.txt");
        std::fs::write(&path, b"guiScale:3\n").expect("write");
        let hold = |millis: u64| {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .share_mode(0)
                .open(&path)
                .expect("an exclusive open");
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(millis));
                drop(file);
            })
        };
        let released = hold(10);
        let (loaded, state) = GameSettings::load_keeping_damaged(&path);
        released.join().expect("holder");
        assert_eq!(state, OptionsFileState::Fine);
        assert_eq!(loaded.gui_scale, 3);

        let released = hold(600);
        let (_, state) = GameSettings::load_keeping_damaged(&path);
        released.join().expect("holder");
        assert!(
            matches!(state, OptionsFileState::Damaged { .. }),
            "{state:?}"
        );
    }

    #[test]
    fn an_unreadable_file_is_damaged_and_never_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A folder where the file should be cannot be read as a file.
        let path = dir.path().join("options.txt");
        std::fs::create_dir_all(&path).expect("folder");
        let (loaded, state) = GameSettings::load_keeping_damaged(&path);
        assert_eq!(loaded.gui_scale, GameSettings::default().gui_scale);
        assert!(
            matches!(state, OptionsFileState::Damaged { .. }),
            "{state:?}"
        );
    }

    /// A save over an existing file replaces it in one step and leaves no temporary behind; a
    /// save that cannot happen leaves the old file as it was.
    #[test]
    fn save_replaces_atomically_and_a_failed_save_keeps_the_old_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("options.txt");
        let first = GameSettings {
            gui_scale: 2,
            ..GameSettings::default()
        };
        first.save(&path).expect("save");
        let before = std::fs::read(&path).expect("read");
        let second = GameSettings {
            gui_scale: 3,
            ..GameSettings::default()
        };
        // The temporary's name is taken by a folder, so this save fails before the rename.
        let blocker = dir.path().join("options.txt.tmp");
        std::fs::create_dir_all(&blocker).expect("blocker");
        assert!(second.save(&path).is_err());
        assert_eq!(
            std::fs::read(&path).expect("read"),
            before,
            "old file untouched"
        );
        std::fs::remove_dir_all(&blocker).expect("unblock");
        second.save(&path).expect("save");
        assert_eq!(GameSettings::load(&path).expect("load").gui_scale, 3);
        assert!(!blocker.exists(), "no temporary left behind");
    }
}
