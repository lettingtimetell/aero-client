//! Which media session is Spotify's, and which process owns a session's audio.
//!
//! Windows names a media session by the player's app id (`SourceAppUserModelId`). Measured on the
//! user's PC on 2026-11-07: Spotify desktop (installed under `%APPDATA%\Spotify`) reports
//! `Spotify.exe`. The Microsoft Store build reports `SpotifyAB.SpotifyMusic_<hash>!Spotify`.
//! Browsers report a short name (`Chrome`, `MSEdge`) or, for Firefox, an install hash.

/// The app id belongs to Spotify's desktop app (classic installer or Microsoft Store build).
///
/// The web player runs inside a browser and reports the browser's id, so it never matches.
#[must_use]
pub(crate) fn is_spotify(app_id: &str) -> bool {
    app_id.to_ascii_lowercase().contains("spotify")
}

/// The executable file name (lowercase, for example `spotify.exe`) whose Windows audio session
/// carries this player's sound, or `None` when the id says nothing reliable about a process.
#[must_use]
pub(crate) fn audio_exe_for(app_id: &str) -> Option<String> {
    let id = app_id.trim();
    if is_spotify(id) {
        // Both builds play from Spotify.exe.
        return Some("spotify.exe".to_owned());
    }
    // A Win32 player without a registered id is named by its executable, sometimes a full path.
    let name = id
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(id)
        .to_ascii_lowercase();
    if name.is_empty() || name.starts_with('.') {
        return None;
    }
    if name.ends_with(".exe") {
        return Some(name);
    }
    // A packaged app (`Family!App`) does not name its process.
    if name.contains('!') || name.contains(char::is_whitespace) {
        return None;
    }
    // Firefox reports an install hash (16 hex digits), not a name.
    if name.len() >= 12 && name.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    // Chromium browsers: `Chrome` plays from chrome.exe, `MSEdge` from msedge.exe.
    if name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Some(format!("{name}.exe"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spotify_is_recognised_in_both_builds_and_nothing_else_is() {
        assert!(is_spotify("Spotify.exe"), "the id measured on a test PC");
        assert!(is_spotify("SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"));
        assert!(!is_spotify("Chrome"));
        assert!(!is_spotify("MSEdge"));
        assert!(!is_spotify("308046B0AF4A39CB"));
        assert!(!is_spotify(""));
    }

    #[test]
    fn audio_exe_follows_the_app_id_where_it_names_a_process() {
        assert_eq!(audio_exe_for("Spotify.exe").as_deref(), Some("spotify.exe"));
        assert_eq!(
            audio_exe_for("SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify").as_deref(),
            Some("spotify.exe")
        );
        assert_eq!(audio_exe_for("Chrome").as_deref(), Some("chrome.exe"));
        assert_eq!(audio_exe_for("MSEdge").as_deref(), Some("msedge.exe"));
        assert_eq!(
            audio_exe_for("musicplayer.exe").as_deref(),
            Some("musicplayer.exe")
        );
        assert_eq!(
            audio_exe_for(r"C:\Program Files\Some Player\Player.exe").as_deref(),
            Some("player.exe")
        );
        assert_eq!(
            audio_exe_for("308046B0AF4A39CB"),
            None,
            "Firefox's install hash"
        );
        assert_eq!(
            audio_exe_for("Microsoft.ZuneMusic_8wekyb3d8bbwe!Microsoft.ZuneMusic"),
            None
        );
        assert_eq!(audio_exe_for(""), None);
        assert_eq!(audio_exe_for("   "), None);
        assert_eq!(audio_exe_for(".exe"), None);
    }
}
