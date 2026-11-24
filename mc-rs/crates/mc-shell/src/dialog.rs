//! Plain pop-ups and the folder picker.
//!
//! Test mode: with [`NO_GUI_ENV`] set to `1`, nothing is shown. A message is printed to standard
//! output instead, `[Warning] <text>`, as the PowerShell launcher printed its own, and answered
//! with [`Answer::Cancel`]; the folder picker answers `None`.

use std::path::PathBuf;

/// The test hook: `AERO_LAUNCHER_NOGUI=1` prints messages instead of showing them.
pub const NO_GUI_ENV: &str = "AERO_LAUNCHER_NOGUI";

/// A message's icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    /// For information.
    Information,
    /// Something needs the player.
    Warning,
    /// Something went wrong.
    Error,
    /// A question.
    Question,
}

/// A message's buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buttons {
    /// OK.
    Ok,
    /// OK and Cancel.
    OkCancel,
    /// Yes and No.
    YesNo,
}

/// The button the player chose. Closing a message counts as Cancel (or No).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    /// OK.
    Ok,
    /// Cancel, or a message that was not shown (test mode).
    Cancel,
    /// Yes.
    Yes,
    /// No.
    No,
}

/// Whether the test hook [`NO_GUI_ENV`] is on.
#[must_use]
pub fn no_gui() -> bool {
    std::env::var_os(NO_GUI_ENV).is_some_and(|value| value == "1")
}

/// What test mode prints for a message.
#[must_use]
pub fn printed(icon: Icon, text: &str) -> String {
    format!("[{icon:?}] {text}")
}

/// Show `text` in a message titled `title` and wait for the player's answer.
pub fn message(title: &str, text: &str, icon: Icon, buttons: Buttons) -> Answer {
    if no_gui() {
        println!("{}", printed(icon, text));
        return Answer::Cancel;
    }
    #[cfg(windows)]
    {
        crate::sys::message_box(title, text, icon, buttons)
    }
    #[cfg(not(windows))]
    {
        let _ = buttons;
        eprintln!("{title}: {}", printed(icon, text));
        Answer::Cancel
    }
}

/// Ask the player for a folder with the shell's folder picker, titled `title`. `None` when they
/// cancel (or in test mode, or where there is no picker).
#[must_use]
pub fn pick_folder(title: &str) -> Option<PathBuf> {
    if no_gui() {
        return None;
    }
    #[cfg(windows)]
    {
        crate::sys::pick_folder(title)
    }
    #[cfg(not(windows))]
    {
        let _ = title;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_prints_what_the_launcher_printed() {
        assert_eq!(
            printed(Icon::Warning, "That folder doesn't have the files."),
            "[Warning] That folder doesn't have the files."
        );
        assert_eq!(printed(Icon::Question, "OK?"), "[Question] OK?");
    }
}
