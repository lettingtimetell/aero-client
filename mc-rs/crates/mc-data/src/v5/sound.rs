//! Sound events as a 1.7.10 client resolves them.
//!
//! 1.7.10 mode plays the 1.8.9 assets: every 1.7.10 event exists in 1.8.9's `sounds.json` with
//! the same category and the same files, byte for byte (the `minecraft/sounds.json` of the
//! 1.7.10 and 1.8 asset indexes, compared event by event). Two things differ. 1.8.9 adds 17
//! events, which a 1.7.10 client does not know and drops ("Unable to play unknown soundEvent",
//! java17: `SoundManager.playSound`). And the two door events list one file each in 1.7.10 where
//! 1.8.9 lists both door files under each.

use crate::misc::{sound_event, SoundEvent};

/// The events 1.8.9's `sounds.json` adds (1.7.10 asset index against 1.8's).
const ONLY_IN_1_8_9: [&str; 17] = [
    "item.fireCharge.use",
    "mob.guardian.attack",
    "mob.guardian.curse",
    "mob.guardian.death",
    "mob.guardian.elder.death",
    "mob.guardian.elder.hit",
    "mob.guardian.elder.idle",
    "mob.guardian.flop",
    "mob.guardian.hit",
    "mob.guardian.idle",
    "mob.guardian.land.death",
    "mob.guardian.land.hit",
    "mob.guardian.land.idle",
    "mob.rabbit.death",
    "mob.rabbit.hop",
    "mob.rabbit.hurt",
    "mob.rabbit.idle",
];

/// A sound event 1.7.10 knows, `None` for the 17 events only 1.8.9 has and for names neither
/// version has.
///
/// # Example
/// ```
/// use mc_data::v5::sound_event_v5;
/// assert!(sound_event_v5("random.click").is_some());
/// assert!(sound_event_v5("mob.rabbit.hop").is_none());
/// ```
#[must_use]
pub fn sound_event_v5(name: &str) -> Option<&'static SoundEvent> {
    if ONLY_IN_1_8_9.contains(&name) {
        return None;
    }
    sound_event(name)
}

/// The sound files 1.7.10's own `sounds.json` lists for `name` where 1.8.9's lists others, as
/// `sounds.json` names them; `None` when the 1.8.9 list is the 1.7.10 one.
///
/// Only the door events differ: 1.7.10 `random.door_open` is `random/door_open` alone and
/// `random.door_close` is `random/door_close` alone; 1.8.9 lists both files under each.
#[must_use]
pub fn sound_files_v5(name: &str) -> Option<&'static [&'static str]> {
    match name {
        "random.door_open" => Some(&["random/door_open"]),
        "random.door_close" => Some(&["random/door_close"]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::sound_events;

    #[test]
    fn the_1_8_9_only_events_are_unknown() {
        for name in ONLY_IN_1_8_9 {
            assert!(sound_event(name).is_some(), "{name} is a 1.8.9 event");
            assert!(sound_event_v5(name).is_none(), "{name}");
        }
        // 1.7.10 has 229 events, all of them in 1.8.9's 246.
        let known = sound_events()
            .iter()
            .filter(|e| sound_event_v5(e.name).is_some())
            .count();
        assert_eq!(sound_events().len(), 246);
        assert_eq!(known, 229);
        // Names neither version has, including the witch sounds 1.7.10 code asks for.
        assert!(sound_event_v5("mob.witch.idle").is_none());
    }

    #[test]
    fn door_events_have_one_file_each() {
        assert_eq!(
            sound_files_v5("random.door_open"),
            Some(&["random/door_open"][..])
        );
        assert_eq!(
            sound_files_v5("random.door_close"),
            Some(&["random/door_close"][..])
        );
        assert_eq!(sound_event("random.door_open").map(|e| e.variants), Some(2));
        assert_eq!(sound_files_v5("random.click"), None);
    }
}
