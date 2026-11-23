//! The live markers and the rules that remove them.
//!
//! Pure data, so every rule is unit-testable without a client. The client
//! (`app/mods/x13_markers.rs`) owns one [`MarkerSet`], places markers where the player looks,
//! draws them, and applies these rules:
//!
//! * **A few at most.** [`MarkerSet::place`] keeps at most `max` markers: one more drops the
//!   oldest. [`MarkerSet::limit`] applies a lowered maximum at once.
//! * **Pressing the key on a marker removes it.** [`MarkerSet::nearest_within`] finds a marker
//!   next to the spot the player just aimed at. (The client tests the drawn label first; that
//!   geometry is `mc_ui::marker_label::aimed_at`.)
//! * **Clear key.** [`MarkerSet::clear`].
//! * **Timeout.** [`MarkerSet::expire`] drops the markers older than the lifetime.
//! * **World change.** [`MarkerSet::world_changed`]: a new world (JoinGame, a respawn into
//!   another dimension) or a disconnect.
//! * **Far teleport.** [`MarkerSet::track_position`] is fed the player's position once a tick
//!   and clears every marker when one tick moved the player further than the teleport distance.
//!   Walking, sprinting, falling (at most 3.92 blocks a tick), knockback and riding stay far
//!   below any distance the option allows.
//!
//! Positions are world coordinates. Times are milliseconds on whatever monotonic clock the
//! caller uses (the client: time since start).

/// One marker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Marker {
    /// Unique within its set: counts up from 1 and is never reused.
    pub id: u64,
    /// The marked point, in world coordinates.
    pub pos: [f64; 3],
    /// When it was placed, on the caller's clock (milliseconds).
    pub placed_ms: u64,
}

/// The markers that are up, oldest first, plus what the far-teleport rule remembers.
#[derive(Debug, Clone, Default)]
pub struct MarkerSet {
    markers: Vec<Marker>,
    next_id: u64,
    revision: u64,
    /// The player's position at the last [`MarkerSet::track_position`] while markers were up.
    /// `None` until the first sample after the set stopped being empty.
    last_pos: Option<[f64; 3]>,
}

impl MarkerSet {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no marker is up.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }

    /// How many markers are up.
    #[must_use]
    pub fn len(&self) -> usize {
        self.markers.len()
    }

    /// The markers, oldest first.
    #[must_use]
    pub fn markers(&self) -> &[Marker] {
        &self.markers
    }

    /// Bumped by every change to the list, so a caller can cache per-marker data by index.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Put a marker at `pos`, then drop the oldest ones so at most `max` (at least 1) are up.
    /// Returns the new marker's id.
    pub fn place(&mut self, pos: [f64; 3], now_ms: u64, max: usize) -> u64 {
        if self.markers.is_empty() {
            // The first marker of a new run: an older run's position sample is stale.
            self.last_pos = None;
        }
        self.next_id += 1;
        let id = self.next_id;
        self.markers.push(Marker {
            id,
            pos,
            placed_ms: now_ms,
        });
        self.drop_oldest(max);
        self.revision += 1;
        id
    }

    /// Drop the oldest markers until at most `max` (at least 1) are up. Whether any went.
    pub fn limit(&mut self, max: usize) -> bool {
        let before = self.markers.len();
        self.drop_oldest(max);
        self.changed(before)
    }

    /// Remove the marker with this id. Whether it was up.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.markers.len();
        self.markers.retain(|marker| marker.id != id);
        self.changed(before)
    }

    /// Remove every marker (the clear key). Whether any was up.
    pub fn clear(&mut self) -> bool {
        let before = self.markers.len();
        self.markers.clear();
        self.changed(before)
    }

    /// Remove every marker placed `lifetime_ms` or longer before `now_ms`. Whether any went.
    pub fn expire(&mut self, now_ms: u64, lifetime_ms: u64) -> bool {
        let before = self.markers.len();
        self.markers
            .retain(|marker| now_ms.saturating_sub(marker.placed_ms) < lifetime_ms);
        self.changed(before)
    }

    /// The world was replaced (JoinGame, a respawn into another dimension) or the connection
    /// ended: every marker belongs to the old world. Whether any was up.
    pub fn world_changed(&mut self) -> bool {
        self.clear()
    }

    /// The far-teleport rule: feed the player's position once a tick. When it is more than
    /// `teleport_distance` blocks from the previous sample, every marker is removed and this
    /// returns `true`. A non-finite position is ignored; with no marker up nothing is kept.
    pub fn track_position(&mut self, pos: [f64; 3], teleport_distance: f64) -> bool {
        if self.markers.is_empty() {
            self.last_pos = None;
            return false;
        }
        if !pos.iter().all(|c| c.is_finite()) {
            return false;
        }
        let jumped = self
            .last_pos
            .is_some_and(|last| distance(last, pos) > teleport_distance);
        self.last_pos = Some(pos);
        jumped && self.clear()
    }

    /// The marker closest to `pos`, if one is within `radius` blocks of it.
    #[must_use]
    pub fn nearest_within(&self, pos: [f64; 3], radius: f64) -> Option<u64> {
        self.markers
            .iter()
            .map(|marker| (distance(marker.pos, pos), marker.id))
            .filter(|&(d, _)| d <= radius)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, id)| id)
    }

    fn drop_oldest(&mut self, max: usize) {
        let max = max.max(1);
        if self.markers.len() > max {
            let excess = self.markers.len() - max;
            self.markers.drain(..excess);
        }
    }

    /// Bookkeeping after a removal: bump the revision if the list changed, and forget the
    /// position sample once nothing is up.
    fn changed(&mut self, before: usize) -> bool {
        let removed = self.markers.len() != before;
        if removed {
            self.revision += 1;
        }
        if self.markers.is_empty() {
            self.last_pos = None;
        }
        removed
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(set: &MarkerSet) -> Vec<u64> {
        set.markers().iter().map(|marker| marker.id).collect()
    }

    #[test]
    fn placing_past_the_maximum_drops_the_oldest() {
        let mut set = MarkerSet::new();
        let a = set.place([0.0, 64.0, 0.0], 0, 3);
        let b = set.place([1.0, 64.0, 0.0], 10, 3);
        let c = set.place([2.0, 64.0, 0.0], 20, 3);
        assert_eq!(ids(&set), [a, b, c]);
        let d = set.place([3.0, 64.0, 0.0], 30, 3);
        assert_eq!(ids(&set), [b, c, d], "the oldest went");
        assert_eq!(set.markers()[2].pos, [3.0, 64.0, 0.0]);
        assert_eq!(set.markers()[2].placed_ms, 30);
        // A maximum of 0 still keeps the marker just placed.
        let e = set.place([4.0, 64.0, 0.0], 40, 0);
        assert_eq!(ids(&set), [e]);
        assert!(
            a < b && b < c && c < d && d < e,
            "ids count up and are never reused"
        );
    }

    #[test]
    fn a_lowered_maximum_applies_at_once() {
        let mut set = MarkerSet::new();
        let ids_placed: Vec<u64> = (0..5)
            .map(|i| set.place([f64::from(i), 0.0, 0.0], 0, 10))
            .collect();
        let revision = set.revision();
        assert!(!set.limit(5), "nothing over the limit");
        assert_eq!(set.revision(), revision);
        assert!(set.limit(2));
        assert_eq!(ids(&set), ids_placed[3..]);
        assert!(set.revision() > revision);
    }

    #[test]
    fn pressing_on_a_marker_finds_the_nearest_one_within_the_radius() {
        let mut set = MarkerSet::new();
        let near = set.place([10.0, 64.0, 10.0], 0, 5);
        let other = set.place([10.0, 64.0, 11.5], 0, 5);
        assert_eq!(set.nearest_within([10.2, 64.0, 10.1], 1.0), Some(near));
        assert_eq!(set.nearest_within([10.0, 64.0, 11.2], 1.0), Some(other));
        assert_eq!(
            set.nearest_within([10.0, 64.0, 10.75], 1.0),
            Some(near),
            "a tie goes to the older"
        );
        assert_eq!(set.nearest_within([13.0, 64.0, 10.0], 1.0), None);
        assert!(set.remove(near));
        assert!(!set.remove(near), "already gone");
        assert_eq!(ids(&set), [other]);
    }

    #[test]
    fn the_clear_key_removes_every_marker() {
        let mut set = MarkerSet::new();
        assert!(!set.clear(), "nothing to clear");
        set.place([0.0, 0.0, 0.0], 0, 5);
        set.place([5.0, 0.0, 0.0], 0, 5);
        let revision = set.revision();
        assert!(set.clear());
        assert!(set.is_empty());
        assert!(set.revision() > revision);
    }

    #[test]
    fn markers_expire_after_their_lifetime() {
        const MINUTE: u64 = 60_000;
        let mut set = MarkerSet::new();
        set.place([0.0, 0.0, 0.0], 1_000, 5);
        let second = set.place([1.0, 0.0, 0.0], 1_000 + 30_000, 5);
        let revision = set.revision();
        assert!(!set.expire(1_000 + MINUTE - 1, MINUTE), "not yet");
        assert_eq!(set.revision(), revision, "no change, no new revision");
        assert!(set.expire(1_000 + MINUTE, MINUTE), "exactly one minute old");
        assert_eq!(ids(&set), [second]);
        assert!(!set.expire(1_000 + MINUTE + 29_999, MINUTE));
        assert!(set.expire(1_000 + MINUTE + 30_000, MINUTE));
        assert!(set.is_empty());
        // A clock that went backwards (or a marker from "the future") never underflows.
        set.place([0.0, 0.0, 0.0], 5_000, 5);
        assert!(!set.expire(0, MINUTE));
    }

    #[test]
    fn a_world_change_clears_every_marker() {
        let mut set = MarkerSet::new();
        assert!(!set.world_changed());
        set.place([0.0, 64.0, 0.0], 0, 5);
        set.place([100.0, 64.0, 0.0], 0, 5);
        assert!(set.world_changed());
        assert!(set.is_empty());
    }

    #[test]
    fn a_far_teleport_clears_but_ordinary_movement_does_not() {
        const LIMIT: f64 = 200.0;
        let mut set = MarkerSet::new();
        set.place([0.0, 64.0, 0.0], 0, 5);
        let mut pos = [0.0, 64.0, 0.0];
        assert!(
            !set.track_position(pos, LIMIT),
            "the first sample only seeds"
        );
        // Sprint-jumping, terminal-velocity falling and a long ender-pearl throw.
        for step in [[0.6, 0.4, 0.0], [0.0, -3.92, 0.0], [45.0, 3.0, 20.0]] {
            pos = [pos[0] + step[0], pos[1] + step[1], pos[2] + step[2]];
            assert!(!set.track_position(pos, LIMIT));
            assert_eq!(set.len(), 1);
        }
        // Exactly the distance is not "over" it.
        pos[0] += LIMIT;
        assert!(!set.track_position(pos, LIMIT));
        assert_eq!(set.len(), 1);
        // A NaN position (a broken packet) is ignored, not a teleport.
        assert!(!set.track_position([f64::NAN, 0.0, 0.0], LIMIT));
        // One tick, 300 blocks: a teleport.
        assert!(set.track_position([pos[0] + 300.0, pos[1], pos[2]], LIMIT));
        assert!(set.is_empty());
    }

    #[test]
    fn the_teleport_rule_never_compares_against_an_older_run_of_markers() {
        let mut set = MarkerSet::new();
        set.place([0.0, 64.0, 0.0], 0, 5);
        assert!(!set.track_position([0.0, 64.0, 0.0], 50.0));
        assert!(set.clear());
        // The player walks 1000 blocks with no marker up (the client skips the hook then, and
        // even a sample now keeps nothing).
        assert!(!set.track_position([500.0, 64.0, 0.0], 50.0));
        set.place([1000.0, 64.0, 0.0], 0, 5);
        assert!(
            !set.track_position([1000.0, 64.0, 0.0], 50.0),
            "the first sample after placing seeds"
        );
        assert_eq!(set.len(), 1);
    }
}
