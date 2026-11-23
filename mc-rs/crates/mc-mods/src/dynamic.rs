//! Menu content a mod supplies at runtime, and what the menu sends back.
//!
//! Static options cover most mods. Waypoints (a list that changes) and voice chat (audio
//! devices found at runtime, a live mic meter, a status line, a "Test mic" button) need more:
//!
//! * [`crate::OptionKind::Button`]: a button row in the options list; a click becomes
//!   [`ModEvent::Button`].
//! * [`DynamicRows`]: extra rows under a mod's expanded section, rebuilt by the client each
//!   frame **while the menu shows that section**: a label, an optional badge (value or status
//!   text), an optional level bar `0..=1`, and up to two small buttons. Clicks become
//!   [`ModEvent::RowAction`].
//! * Dynamic choices: a row with `cycle = true` draws its badge as a clickable choice; a click
//!   becomes [`ModEvent::RowCycle`] (`forward = false` for right-click). Keep the chosen value in
//!   a persisted option (an [`crate::OptionKind::HiddenText`] holds a device name without drawing
//!   a text field) and show it in the badge.
//!
//! Rows reuse their `String`s: [`DynamicRows::begin`] then [`DynamicRows::push`] per row is
//! allocation-free once the list has been as long as it gets.

/// One thing the user did in the menu that the client must act on. The client keeps this
/// frame's events in `Client::mods.events`; they are cleared before the next frame's input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModEvent {
    /// A [`crate::OptionKind::Button`] option was clicked.
    Button {
        /// The mod that owns the option.
        mod_id: &'static str,
        /// The option's id.
        option_id: &'static str,
    },
    /// One of a dynamic row's buttons was clicked.
    RowAction {
        /// The mod whose section holds the row.
        mod_id: &'static str,
        /// The row's [`DynamicRow::key`].
        row: u64,
        /// Which button: the index into [`DynamicRow::buttons`] (0 or 1).
        action: u8,
    },
    /// A dynamic row's cycle badge was clicked.
    RowCycle {
        /// The mod whose section holds the row.
        mod_id: &'static str,
        /// The row's [`DynamicRow::key`].
        row: u64,
        /// `true` for a left click (next), `false` for a right click (previous).
        forward: bool,
    },
}

impl ModEvent {
    /// The mod this event belongs to.
    #[must_use]
    pub fn mod_id(&self) -> &'static str {
        match self {
            ModEvent::Button { mod_id, .. }
            | ModEvent::RowAction { mod_id, .. }
            | ModEvent::RowCycle { mod_id, .. } => mod_id,
        }
    }
}

/// A small glyph the menu can draw on a row button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowIcon {
    /// Visible (an open eye).
    Eye,
    /// Hidden (a struck-through eye).
    EyeOff,
    /// Delete (a cross).
    Delete,
    /// Start / play.
    Play,
    /// Stop.
    Stop,
    /// Add.
    Plus,
    /// Refresh / rescan.
    Refresh,
}

/// The face of a row button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowButton {
    /// A short text button, e.g. `"Test"`.
    Text(&'static str),
    /// A glyph.
    Icon(RowIcon),
}

/// One extra row under an expanded mod.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DynamicRow {
    /// Your id for the row, echoed in its events (a waypoint id, a device index...).
    pub key: u64,
    /// Left-aligned label.
    pub label: String,
    /// Right-aligned badge text (a distance, a status, a device name). Empty for none.
    pub badge: String,
    /// The badge is a clickable choice: clicks send [`ModEvent::RowCycle`].
    pub cycle: bool,
    /// A level bar under the label, `0.0..=1.0` (clamped when drawn).
    pub level: Option<f32>,
    /// Up to two buttons at the right edge, sending [`ModEvent::RowAction`] with their index.
    pub buttons: [Option<RowButton>; 2],
    /// Draw the row greyed out (e.g. a hidden waypoint).
    pub muted: bool,
}

/// A mod's dynamic rows, reused frame to frame.
#[derive(Debug, Clone, Default)]
pub struct DynamicRows {
    rows: Vec<DynamicRow>,
    len: usize,
}

impl DynamicRows {
    /// Start this frame's rows (keeps every row's allocations for reuse).
    pub fn begin(&mut self) {
        self.len = 0;
    }

    /// Append a row and return it, reset to defaults with `key` set. Reuses an old row's
    /// strings when there is one.
    pub fn push(&mut self, key: u64) -> &mut DynamicRow {
        if self.len == self.rows.len() {
            self.rows.push(DynamicRow::default());
        }
        let row = &mut self.rows[self.len];
        self.len += 1;
        row.key = key;
        row.label.clear();
        row.badge.clear();
        row.cycle = false;
        row.level = None;
        row.buttons = [None; 2];
        row.muted = false;
        row
    }

    /// This frame's rows.
    #[must_use]
    pub fn rows(&self) -> &[DynamicRow] {
        &self.rows[..self.len]
    }

    /// Number of rows this frame.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no rows this frame.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_reused_without_reallocating() {
        let mut rows = DynamicRows::default();
        rows.begin();
        let row = rows.push(7);
        row.label.push_str("Spawn point");
        row.badge.push_str("120 m");
        row.buttons = [Some(RowButton::Icon(RowIcon::Eye)), None];
        rows.push(8).label.push_str("Mid");
        assert_eq!(rows.len(), 2);
        let label_ptr = rows.rows()[0].label.as_ptr();

        rows.begin();
        assert!(rows.is_empty());
        let row = rows.push(9);
        assert!(row.label.is_empty() && row.badge.is_empty() && row.buttons == [None, None]);
        row.label.push_str("Base");
        assert_eq!(rows.rows()[0].label.as_ptr(), label_ptr, "same allocation");
        assert_eq!(rows.rows().len(), 1);
        assert_eq!(rows.rows()[0].key, 9);
    }

    #[test]
    fn events_name_their_mod() {
        let events = [
            ModEvent::Button {
                mod_id: "voice_chat",
                option_id: "test_mic",
            },
            ModEvent::RowAction {
                mod_id: "waypoints",
                row: 3,
                action: 1,
            },
            ModEvent::RowCycle {
                mod_id: "voice_chat",
                row: 0,
                forward: false,
            },
        ];
        let ids: Vec<_> = events.iter().map(ModEvent::mod_id).collect();
        assert_eq!(ids, ["voice_chat", "waypoints", "voice_chat"]);
    }
}
