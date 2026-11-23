//! Sound categories and per-category volume, matching `options.txt`.
//!
// java: SoundCategory.java, GameSettings.java (getSoundLevel/setSoundLevel, options.txt lines
// 1041-1043 and 1169-1171: `soundCategory_<name>:<float>`)

use std::fmt;

/// The nine sound categories vanilla 1.8.9 mixes independently.
///
/// Ordinal order and IDs match `SoundCategory.java` exactly (not that the ordinal is
/// wire-relevant here — no packet carries it — but it keeps [`SoundCategory::ALL`] and any
/// future table stable against the Java source).
// java: SoundCategory.java:6
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoundCategory {
    /// The global bus. Not a category any individual sound belongs to; scales everything.
    Master,
    /// Music discs aside — background/game music (`music.*` events).
    Music,
    /// Jukebox records (`records.*` events).
    Records,
    /// Rain and thunder ambience.
    Weather,
    /// Block break/place/step sounds.
    Blocks,
    /// Hostile mobs.
    Mobs,
    /// Passive/neutral mobs.
    Animals,
    /// Player sounds (hurt, eat, splash, ...).
    Players,
    /// Cave ambience and other environment noise.
    Ambient,
}

impl SoundCategory {
    /// Every category, in `SoundCategory.java` declaration order.
    pub const ALL: [SoundCategory; 9] = [
        SoundCategory::Master,
        SoundCategory::Music,
        SoundCategory::Records,
        SoundCategory::Weather,
        SoundCategory::Blocks,
        SoundCategory::Mobs,
        SoundCategory::Animals,
        SoundCategory::Players,
        SoundCategory::Ambient,
    ];

    /// The vanilla category ID (`SoundCategory.getCategoryId`). Unused on the wire in 1.8.9,
    /// kept for parity with the Java source and as a stable array index.
    pub const fn id(self) -> u8 {
        match self {
            SoundCategory::Master => 0,
            SoundCategory::Music => 1,
            SoundCategory::Records => 2,
            SoundCategory::Weather => 3,
            SoundCategory::Blocks => 4,
            SoundCategory::Mobs => 5,
            SoundCategory::Animals => 6,
            SoundCategory::Players => 7,
            SoundCategory::Ambient => 8,
        }
    }

    /// The name used in `sounds.json`'s `"category"` field and in `options.txt`'s
    /// `soundCategory_<name>` keys. **Not** the enum's Rust identifier — vanilla's own names
    /// are irregular (`record`, not `records`; `hostile`, not `mobs`; `neutral`, not `animals`).
    // java: SoundCategory.java's constructor arguments
    pub const fn category_name(self) -> &'static str {
        match self {
            SoundCategory::Master => "master",
            SoundCategory::Music => "music",
            SoundCategory::Records => "record",
            SoundCategory::Weather => "weather",
            SoundCategory::Blocks => "block",
            SoundCategory::Mobs => "hostile",
            SoundCategory::Animals => "neutral",
            SoundCategory::Players => "player",
            SoundCategory::Ambient => "ambient",
        }
    }

    /// Looks up a category by its `sounds.json`/`options.txt` name.
    // java: SoundCategory.getCategory
    pub fn parse(name: &str) -> Option<SoundCategory> {
        SoundCategory::ALL
            .into_iter()
            .find(|c| c.category_name() == name)
    }
}

impl Default for SoundCategory {
    /// `Master`, matching `SoundListSerializer`'s `"category"` default
    /// (`SoundCategory.MASTER.getCategoryName()`).
    fn default() -> Self {
        SoundCategory::Master
    }
}

impl fmt::Display for SoundCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.category_name())
    }
}

/// Per-category volume sliders, `[0.0, 1.0]` each, defaulting to `1.0`.
///
/// # Example
/// ```
/// use mc_audio::{CategoryVolumes, SoundCategory};
///
/// let mut volumes = CategoryVolumes::new();
/// volumes.set(SoundCategory::Music, 0.4);
/// volumes.set(SoundCategory::Master, 0.0); // the fader itself still records...
///
/// assert_eq!(volumes.effective(SoundCategory::Music), 0.4);
/// // ...but `effective` never folds Master in per-sound; the engine applies it once, globally.
/// assert_eq!(volumes.effective(SoundCategory::Master), 1.0);
/// ```
///
/// This is the *runtime* analogue of `GameSettings.mapSoundLevels` and nothing more. It used to
/// carry its own `soundCategory_*` parser and writer as well, which made it a second, rival
/// owner of nine `options.txt` fields that `mc_render::settings::GameSettings` already reads and
/// writes — and the rival was the dead one, since `GameSettings` is what the client
/// actually loads and saves. That half is gone: `GameSettings` owns the file format, and
/// `Client::sync_sound_volumes` (`mc-client`'s `app::audio`) copies the nine floats in here on
/// load and whenever the options screen changes one. Vanilla has exactly one owner too —
/// `SoundManager.getSoundCategoryVolume` reads `GameSettings.getSoundLevel` directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CategoryVolumes {
    levels: [f32; 9],
}

impl Default for CategoryVolumes {
    fn default() -> Self {
        Self { levels: [1.0; 9] }
    }
}

impl CategoryVolumes {
    /// All categories at full volume, vanilla's default.
    pub fn new() -> Self {
        Self::default()
    }

    /// The raw slider for `category`. `1.0` (vanilla's default) if never set.
    // java: GameSettings.getSoundLevel
    pub fn get(&self, category: SoundCategory) -> f32 {
        self.levels[category.id() as usize]
    }

    /// Sets the slider for `category`, clamped to `[0.0, 1.0]`.
    ///
    /// Vanilla's slider UI already clamps before calling `setSoundLevel`; this defends against
    /// a hand-edited `options.txt` doing something out of range.
    // java: GameSettings.setSoundLevel
    pub fn set(&mut self, category: SoundCategory, level: f32) {
        self.levels[category.id() as usize] = level.clamp(0.0, 1.0);
    }

    /// The multiplier `SoundManager.getNormalizedVolume` applies for `category`: always `1.0`
    /// for [`SoundCategory::Master`] (the master fader is applied once, globally, not
    /// per-category — see [`crate::AudioEngine`]'s use of the main mixer track), the slider
    /// value for everything else.
    // java: SoundManager.getSoundCategoryVolume
    pub fn effective(&self, category: SoundCategory) -> f32 {
        if category == SoundCategory::Master {
            1.0
        } else {
            self.get(category)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_names_match_java_exactly() {
        assert_eq!(SoundCategory::Master.category_name(), "master");
        assert_eq!(SoundCategory::Music.category_name(), "music");
        assert_eq!(SoundCategory::Records.category_name(), "record");
        assert_eq!(SoundCategory::Weather.category_name(), "weather");
        assert_eq!(SoundCategory::Blocks.category_name(), "block");
        assert_eq!(SoundCategory::Mobs.category_name(), "hostile");
        assert_eq!(SoundCategory::Animals.category_name(), "neutral");
        assert_eq!(SoundCategory::Players.category_name(), "player");
        assert_eq!(SoundCategory::Ambient.category_name(), "ambient");
    }

    #[test]
    fn parse_round_trips_every_category() {
        for c in SoundCategory::ALL {
            assert_eq!(SoundCategory::parse(c.category_name()), Some(c));
        }
        assert_eq!(SoundCategory::parse("nonsense"), None);
    }

    #[test]
    fn default_is_full_volume_everywhere() {
        let v = CategoryVolumes::new();
        for c in SoundCategory::ALL {
            assert_eq!(v.get(c), 1.0);
            assert_eq!(v.effective(c), 1.0);
        }
    }

    #[test]
    fn master_effective_is_always_one() {
        let mut v = CategoryVolumes::new();
        v.set(SoundCategory::Master, 0.0);
        // The slider itself changed...
        assert_eq!(v.get(SoundCategory::Master), 0.0);
        // ...but `effective` never applies it per-sound; the engine applies it once, globally.
        assert_eq!(v.effective(SoundCategory::Master), 1.0);
    }

    #[test]
    fn set_clamps_out_of_range() {
        let mut v = CategoryVolumes::new();
        v.set(SoundCategory::Blocks, 5.0);
        assert_eq!(v.get(SoundCategory::Blocks), 1.0);
        v.set(SoundCategory::Blocks, -5.0);
        assert_eq!(v.get(SoundCategory::Blocks), 0.0);
    }

    /// The `options.txt` names are the *only* part of this type that has to agree
    /// with `mc_render::settings::GameSettings`, now that the parser lives solely over there.
    /// `Client::sync_sound_volumes` looks each category up by [`SoundCategory::category_name`],
    /// so a rename on either side silently zeroes a slider — pin the nine strings here and in
    /// `mc-client`'s `app::audio` tests, from both directions.
    #[test]
    fn every_category_name_is_a_key_game_settings_also_uses() {
        // Transcribed from `GameSettings.java`'s `soundCategory_` writer, not from either
        // Rust side's output.
        let vanilla = [
            "master", "music", "record", "weather", "block", "hostile", "neutral", "player",
            "ambient",
        ];
        let ours: Vec<&str> = SoundCategory::ALL
            .iter()
            .map(|c| c.category_name())
            .collect();
        assert_eq!(ours, vanilla);
    }
}
