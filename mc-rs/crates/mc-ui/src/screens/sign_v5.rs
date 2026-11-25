//! Minecraft 1.7.10's sign and command-block editors. A child of `sign`, for the editors'
//! private state.
//!
//! - **Sign lines**: java17 `GuiEditSign.keyTyped` (`client/gui/inventory/GuiEditSign.java:67-91`)
//!   appends a typed character while the line holds fewer than 15 characters, whatever their
//!   width (1.8.9: while the line stays within 90 pixels). A 1.7.10 server replaces a longer
//!   line with `!?` (`NetHandlerPlayServer.processUpdateSign`), so narrow characters must not
//!   get past 15 here.
//! - **Command blocks**: java17 `GuiCommandBlock` (`client/gui/GuiCommandBlock.java:32-53`) has no
//!   track-output toggle; `MC|AdvCdm` carries no track-output flag in 1.7.10 either.

use mc_version::GameVersion;

use super::{CommandBlockScreen, EditSignScreen};

/// The longest sign line 1.7.10's editor lets the player type, in UTF-16 units.
pub const MAX_LINE_LENGTH_V5: usize = 15;

impl EditSignScreen {
    /// Edit with `version`'s line rule. Editors start with 1.8.9's.
    pub fn set_game_version(&mut self, version: GameVersion) {
        self.version = version;
    }
}

/// java17 `GuiEditSign.keyTyped`: whether `candidate`, the line with the typed character
/// appended, is one the editor keeps: at most 15 UTF-16 units, the length the 1.7.10 server
/// checks (`String.length()`). java17 appends one UTF-16 unit per key event, so after 14 units
/// a character outside the Basic Multilingual Plane would leave half of itself; this editor
/// takes whole characters and refuses it.
pub(super) fn line_accepts_v5(candidate: &str) -> bool {
    candidate.encode_utf16().count() <= MAX_LINE_LENGTH_V5
}

impl CommandBlockScreen {
    /// Edit with `version`'s controls: a 1.7.10 editor has no track-output toggle.
    pub fn set_game_version(&mut self, version: GameVersion) {
        self.version = version;
        self.hide_track_output_v5();
    }

    /// The toggle is neither drawn nor clickable in a 1.7.10 editor (it is rebuilt by each
    /// layout, so the layout calls this too).
    pub(super) fn hide_track_output_v5(&mut self) {
        if self.version.is_1_7_10() {
            self.track_output_button.visible = false;
        }
    }
}

#[cfg(test)]
#[path = "sign_v5_tests.rs"]
mod tests;
