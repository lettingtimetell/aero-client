//! Per-event volume multipliers: the Sounds mod.
//!
//! An [`EventGains`] table maps a sound event (`minecraft:random.bow`) to a linear gain. The
//! client builds it from the mod's settings **when they change**, never per play, and hands it to
//! [`crate::AudioEngine::set_event_gains`]. The engine looks an event up once when it starts a
//! channel and stores the result on the channel, so the per-tick refresh and the category sliders
//! keep applying it without another lookup.
//!
//! The gain multiplies the final channel gain, after vanilla's
//! `clamp(sound.volume * entry.volume, 0, 1) * category` and after distance attenuation. It is
//! deliberately **not** folded into the request's volume: vanilla widens a sound's audible range
//! when its volume is above 1.0 (`fade_distance`), and a louder setting must not make a sound
//! carry further. 0 silences the event (it is resolved but never decoded or started, like a
//! zeroed category); 2.0 is +6 dB.
//!
//! An empty table (the mod off, or nothing changed) costs one `is_empty` test per play and
//! leaves every number exactly as vanilla computes it.

use std::collections::HashMap;

use mc_assets::ResourceLocation;

/// The loudest an event can be set: 200 %.
pub const MAX_EVENT_GAIN: f32 = 2.0;

/// Sound event → linear gain. Events not in the table play at `1.0` (vanilla).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EventGains {
    gains: HashMap<ResourceLocation, f32>,
}

impl EventGains {
    /// An empty table: every event at vanilla volume.
    #[must_use]
    pub fn new() -> EventGains {
        EventGains::default()
    }

    /// Set one event's gain, clamped to `0.0..=MAX_EVENT_GAIN` (a NaN counts as `1.0`). `1.0`
    /// removes the entry, so the table only ever holds real changes.
    pub fn set(&mut self, event: ResourceLocation, gain: f32) {
        let gain = if gain.is_nan() {
            1.0
        } else {
            gain.clamp(0.0, MAX_EVENT_GAIN)
        };
        if gain == 1.0 {
            self.gains.remove(&event);
        } else {
            self.gains.insert(event, gain);
        }
    }

    /// The gain for `event`: `1.0` unless the table changes it. An empty table answers without
    /// hashing.
    #[must_use]
    pub fn get(&self, event: &ResourceLocation) -> f32 {
        if self.gains.is_empty() {
            return 1.0;
        }
        self.gains.get(event).copied().unwrap_or(1.0)
    }

    /// Whether no event is changed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gains.is_empty()
    }

    /// How many events are changed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.gains.len()
    }

    /// Every changed event and its gain, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (&ResourceLocation, f32)> {
        self.gains.iter().map(|(event, &gain)| (event, gain))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(path: &str) -> ResourceLocation {
        ResourceLocation::minecraft(path)
    }

    #[test]
    fn unknown_events_play_at_vanilla_volume() {
        let mut gains = EventGains::new();
        assert_eq!(gains.get(&loc("random.bow")), 1.0);
        gains.set(loc("random.bow"), 1.5);
        assert_eq!(gains.get(&loc("random.bow")), 1.5);
        assert_eq!(gains.get(&loc("random.orb")), 1.0);
        assert_eq!(
            gains.get(&ResourceLocation::new("mypack", "random.bow")),
            1.0
        );
    }

    #[test]
    fn gains_are_clamped_and_vanilla_entries_are_dropped() {
        let mut gains = EventGains::new();
        gains.set(loc("a"), 5.0);
        gains.set(loc("b"), -1.0);
        gains.set(loc("c"), f32::NAN);
        gains.set(loc("d"), 0.0);
        assert_eq!(gains.get(&loc("a")), MAX_EVENT_GAIN);
        assert_eq!(gains.get(&loc("b")), 0.0);
        assert_eq!(gains.get(&loc("d")), 0.0);
        assert_eq!(gains.len(), 3, "NaN means vanilla, so `c` is not stored");
        gains.set(loc("a"), 1.0);
        assert_eq!(gains.len(), 2, "back to 100 % removes the entry");
        assert_eq!(gains.get(&loc("a")), 1.0);
    }
}
