//! The 1.7.10 tab list: a list of names in arrival order, each with its ping.
//!
//! A 1.7.10 server's `S38PacketPlayerListItem` carries only a name, an online flag and a ping
//! (java17 `network/play/server/S38PacketPlayerListItem.java`), and the client keys its list by
//! that name: `NetHandlerPlayClient.playerInfoMap` (name to `GuiPlayerInfo`) plus
//! `playerInfoList` in arrival order, which is the order the tab overlay draws (java17
//! `client/network/NetHandlerPlayClient.java:1345-1366`, `client/gui/GuiPlayerInfo.java`). There
//! is no profile, game mode or display name, and the list never feeds a player spawn (that comes
//! from `S0CPacketSpawnPlayer` itself).
//!
//! Like `playerInfoMap` the list lives as long as the connection's handler: a respawn or a new
//! world keeps it.

use std::collections::HashMap;

/// One row: java17 `GuiPlayerInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerInfoV5 {
    /// `GuiPlayerInfo.name`, the key.
    pub name: String,
    /// `GuiPlayerInfo.responseTime`, the last ping the server sent, in milliseconds.
    pub ping_ms: i32,
}

/// java17 `NetHandlerPlayClient.playerInfoMap` and `playerInfoList`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlayerListV5 {
    /// Arrival order; names are unique.
    entries: Vec<PlayerInfoV5>,
    /// Each name's position in `entries`, so a packet or a frame finds a row without a scan
    /// (`playerInfoMap` is a `HashMap`): a server may send any number of names.
    index: HashMap<String, usize>,
    /// Bumped whenever a row is added or removed or its ping changes.
    revision: u64,
}

impl PlayerListV5 {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one `S38PacketPlayerListItem`.
    ///
    /// java17 `NetHandlerPlayClient.handlePlayerListItem` (`:1345-1366`): an unknown name that
    /// is online is appended; a known name that is offline is removed; a known name that is
    /// online takes the packet's ping. An unknown name that is offline changes nothing.
    pub fn handle_player_list_item(&mut self, name: &str, online: bool, ping: i16) {
        let index = self.index.get(name).copied();
        match (index, online) {
            (None, true) => {
                self.index.insert(name.to_owned(), self.entries.len());
                self.entries.push(PlayerInfoV5 {
                    name: name.to_owned(),
                    ping_ms: i32::from(ping),
                });
                self.revision += 1;
            }
            (Some(index), false) => {
                self.index.remove(name);
                self.entries.remove(index);
                // The rows after it moved up by one (`playerInfoList.remove` is linear too).
                for (position, entry) in self.entries.iter().enumerate().skip(index) {
                    if let Some(slot) = self.index.get_mut(&entry.name) {
                        *slot = position;
                    }
                }
                self.revision += 1;
            }
            (Some(index), true) => {
                let ping_ms = i32::from(ping);
                let entry = &mut self.entries[index];
                if entry.ping_ms != ping_ms {
                    entry.ping_ms = ping_ms;
                    self.revision += 1;
                }
            }
            (None, false) => {}
        }
    }

    /// The rows in arrival order (`playerInfoList`), the order the tab overlay draws.
    pub fn iter(&self) -> impl Iterator<Item = &PlayerInfoV5> {
        self.entries.iter()
    }

    /// The row for `name` (`playerInfoMap.get`).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PlayerInfoV5> {
        self.index.get(name).map(|&index| &self.entries[index])
    }

    /// How many rows there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A counter that changes whenever a row is added or removed or a ping changes, so a reader
    /// can rebuild what it draws only then.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &PlayerListV5) -> Vec<&str> {
        list.iter().map(|entry| entry.name.as_str()).collect()
    }

    /// The arrival-order case of java17 `handlePlayerListItem`: add B, A, C; remove A; add A ->
    /// B, C, A (the list is not sorted and a returning name goes to the end).
    #[test]
    fn rows_keep_arrival_order_and_a_returning_name_goes_last() {
        let mut list = PlayerListV5::new();
        for name in ["B", "A", "C"] {
            list.handle_player_list_item(name, true, 10);
        }
        list.handle_player_list_item("A", false, 0);
        assert_eq!(names(&list), ["B", "C"]);
        list.handle_player_list_item("A", true, 25);
        assert_eq!(names(&list), ["B", "C", "A"]);
        assert_eq!(list.get("A").map(|entry| entry.ping_ms), Some(25));
        assert_eq!(list.len(), 3);
        assert!(!list.is_empty());
    }

    /// Online updates the ping of a known name; offline for an unknown name does nothing; the
    /// name is the key, case included (`playerInfoMap` is a plain `HashMap<String, ...>`).
    #[test]
    fn ping_updates_and_unknown_removals() {
        let mut list = PlayerListV5::new();
        list.handle_player_list_item("Steve", true, 40);
        list.handle_player_list_item("Steve", true, -1);
        assert_eq!(list.get("Steve").map(|entry| entry.ping_ms), Some(-1));
        list.handle_player_list_item("Alex", false, 99);
        list.handle_player_list_item("steve", false, 0);
        assert_eq!(names(&list), ["Steve"]);
        assert!(list.get("steve").is_none());
    }

    /// Lookups by name follow the rows when earlier rows leave, and a removed name is gone.
    #[test]
    fn lookups_follow_removals() {
        let mut list = PlayerListV5::new();
        for (ping, name) in ["A", "B", "C", "D"].into_iter().enumerate() {
            list.handle_player_list_item(name, true, ping as i16);
        }
        list.handle_player_list_item("B", false, 0);
        list.handle_player_list_item("A", false, 0);
        for (name, ping) in [("C", 2), ("D", 3)] {
            let row = list
                .get(name)
                .map(|entry| (entry.name.as_str(), entry.ping_ms));
            assert_eq!(row, Some((name, ping)));
        }
        assert!(list.get("A").is_none() && list.get("B").is_none());
        list.handle_player_list_item("D", true, 9);
        list.handle_player_list_item("A", true, 1);
        assert_eq!(names(&list), ["C", "D", "A"]);
        assert_eq!(list.get("D").map(|entry| entry.ping_ms), Some(9));
        assert_eq!(list.get("A").map(|entry| entry.ping_ms), Some(1));
    }

    /// The revision moves on every visible change and only then.
    #[test]
    fn revision_counts_visible_changes() {
        let mut list = PlayerListV5::new();
        assert_eq!(list.revision(), 0);
        list.handle_player_list_item("A", true, 5);
        assert_eq!(list.revision(), 1);
        list.handle_player_list_item("A", true, 5);
        assert_eq!(list.revision(), 1, "the same ping again changes nothing");
        list.handle_player_list_item("A", true, 6);
        assert_eq!(list.revision(), 2);
        list.handle_player_list_item("B", false, 6);
        assert_eq!(
            list.revision(),
            2,
            "removing an unknown name changes nothing"
        );
        list.handle_player_list_item("A", false, 6);
        assert_eq!(list.revision(), 3);
        assert!(list.is_empty());
    }
}
