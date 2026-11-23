//! Clips to video: the clip settings (length is the clip's own, frame rate, resolution, maximum
//! file size), a one-click export for the replay browser, and the automatic export of new clips.
//!
//! # Automatic export, and why it waits for an idle title screen
//!
//! An export takes over the frame loop (the frame graph renders the clip offscreen at the clip's
//! size; the window shows the preview and a progress panel). So it cannot run behind a title
//! screen the player is using. Instead, with "Export clips to video" on, a new clip is exported
//! once the player has been on the title screen for a few seconds without touching anything;
//! any key or mouse button stops it at once (no file is left) and it is tried again at the next
//! idle moment. A clip that fails is not retried until the game restarts. Clips are always H.264
//! (chat apps and every phone play it) into `Videos\Aero Client\Clips\<clip name>.mp4`; a clip
//! whose video already exists is never exported again.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_mods::mods::replay::{self as replay_mod, ids};
use mc_video::{FrameRate, VideoCodec};

use super::{
    default_export_folder, resolution_for_choice, safe_file_name, ExportPhase, ExportRequest,
    ExportTarget,
};
use crate::app::driver::{Client, Ui};

/// How long the title screen must sit untouched before a clip is exported by itself.
const IDLE_BEFORE_EXPORT: Duration = Duration::from_secs(4);
/// How often the clips folder is looked at while idle.
const SCAN_INTERVAL: Duration = Duration::from_secs(5);

/// The automatic clip export's state.
#[derive(Debug)]
pub(crate) struct ClipExports {
    /// Clips that failed this run (not retried until restart).
    failed: HashSet<PathBuf>,
    /// The last key, click or mouse move on the title screen.
    last_input: Instant,
    last_cursor: (f64, f64),
    next_scan: Option<Instant>,
    /// The clip opened for an export; closed again when that export ends.
    open: Option<OpenClip>,
}

#[derive(Debug)]
struct OpenClip {
    path: PathBuf,
    /// Started by the idle title screen (any input cancels it), or by the browser.
    automatic: bool,
}

impl Default for ClipExports {
    fn default() -> Self {
        Self {
            failed: HashSet::new(),
            last_input: Instant::now(),
            last_cursor: (0.0, 0.0),
            next_scan: None,
            open: None,
        }
    }
}

impl ClipExports {
    /// Whether the running export is an automatic clip export (any input stops it).
    pub(crate) fn automatic(&self) -> bool {
        self.open.as_ref().is_some_and(|open| open.automatic)
    }

    /// A key or a click happened.
    pub(crate) fn note_input(&mut self) {
        self.last_input = Instant::now();
    }

    /// Whether a clip is open for an export (closed again once the export has ended).
    pub(crate) fn is_open(&self) -> bool {
        self.open.is_some()
    }
}

/// Where a clip's video goes.
#[must_use]
pub(crate) fn clip_video_path(game_dir: &Path, clip: &Path) -> PathBuf {
    let stem = clip
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Clip".to_owned());
    default_export_folder(game_dir)
        .join("Clips")
        .join(format!("{}.mp4", safe_file_name(&stem)))
}

impl Client {
    /// The export request for the open clip with the `replay` mod's clip settings.
    fn clip_request(&self, clip: &Path) -> ExportRequest {
        let settings = &self.mods.settings;
        let (start, end) = self.replay_view_range();
        let fps = replay_mod::CLIP_FPS_CHOICES
            .get(settings.choice("replay", ids::CLIP_FPS))
            .and_then(|label| label.parse().ok())
            .unwrap_or(60);
        let (width, height) = resolution_for_choice(
            settings.choice("replay", ids::CLIP_RESOLUTION),
            self.tw_export.window,
        );
        let target = replay_mod::CLIP_MAX_SIZE_BYTES
            .get(settings.choice("replay", ids::CLIP_MAX_SIZE))
            .copied()
            .flatten()
            .map_or(ExportTarget::Quality(2), ExportTarget::MaxBytes);
        ExportRequest {
            start,
            end,
            width,
            height,
            fps: FrameRate::whole(fps),
            codec: VideoCodec::H264,
            target,
            motion_blur: 1,
            game_audio: settings.bool("replay", ids::EXPORT_AUDIO),
            mic: settings.bool("replay", ids::EXPORT_MIC),
            mic_volume: settings.f32("replay", ids::MIC_VOLUME) / 100.0,
            // Desktop audio as in exports; never separate tracks in a clip (clips go
            // to a chat app, under a size limit).
            game_volume: settings.f32("replay", ids::GAME_VOLUME) / 100.0,
            desktop: settings.bool("replay", ids::EXPORT_DESKTOP),
            desktop_volume: settings.f32("replay", ids::DESKTOP_VOLUME) / 100.0,
            separate_tracks: false,
            path: clip_video_path(&self.config.game_dir, clip),
        }
    }

    /// Export a clip with the clip settings (the browser's one-click Export). Opens the clip,
    /// exports it, and closes it again when done (back to the title screen; the outcome is in
    /// [`Client::export_take_finished`]).
    ///
    /// # Errors
    /// A message for the browser.
    pub(crate) fn export_clip(&mut self, clip: &Path) -> Result<(), String> {
        self.export_clip_inner(clip, false)
    }

    fn export_clip_inner(&mut self, clip: &Path, automatic: bool) -> Result<(), String> {
        if self.export_running() {
            return Err("An export is already running.".to_owned());
        }
        if !self.replay_is_active() {
            self.replay_open(clip)?;
        }
        let request = self.clip_request(clip);
        self.tw_export.clips.open = Some(OpenClip {
            path: clip.to_path_buf(),
            automatic,
        });
        if let Err(error) = self.export_start(request) {
            self.tw_export_clip_ended(false);
            return Err(error);
        }
        Ok(())
    }

    /// An export of an opened clip ended: close the clip; remember a failure.
    pub(super) fn tw_export_clip_ended(&mut self, failed: bool) {
        let Some(open) = self.tw_export.clips.open.take() else {
            return;
        };
        if failed {
            self.tw_export.clips.failed.insert(open.path.clone());
        }
        // Any input during an automatic export cancels it; wait for the next idle moment.
        self.tw_export.clips.last_input = Instant::now();
        if self.replay_is_active() {
            self.replay_close();
        }
    }

    /// Once per frame without an export running: watch for input on the title screen, and
    /// export the next new clip once it has been idle long enough.
    pub(super) fn tw_export_clips_idle(&mut self, cursor: (f64, f64)) {
        let clips = &mut self.tw_export.clips;
        if cursor != clips.last_cursor {
            clips.last_cursor = cursor;
            clips.last_input = Instant::now();
        }
        if !matches!(self.ui, Ui::MainMenu(_)) || self.session.is_some() || self.replay_is_active()
        {
            return;
        }
        // The loading screen still covers the title screen: the idle time starts after it.
        if self.tw_loading_frame_cap().is_some() {
            self.tw_export.clips.last_input = Instant::now();
            return;
        }
        let clips = &mut self.tw_export.clips;
        let now = Instant::now();
        if now.duration_since(clips.last_input) < IDLE_BEFORE_EXPORT
            || clips.next_scan.is_some_and(|next| now < next)
        {
            return;
        }
        clips.next_scan = Some(now + SCAN_INTERVAL);
        // The option belongs to the Replay mod: with the mod off it does nothing.
        if !self.mods.settings.enabled("replay")
            || !self.mods.settings.bool("replay", ids::CLIP_AUTO_EXPORT)
        {
            return;
        }
        let game_dir = self.config.game_dir.clone();
        let library = mc_replay::Library::new(&game_dir);
        let mut candidates: Vec<_> = library
            .list_dir(true)
            .into_iter()
            .filter(|entry| !self.tw_export.clips.failed.contains(&entry.path))
            .filter(|entry| !clip_video_path(&game_dir, &entry.path).exists())
            .collect();
        // Oldest first: clips come out in the order they were made.
        candidates.sort_by_key(|entry| entry.meta.started_unix_ms);
        let Some(next) = candidates.into_iter().next() else {
            return;
        };
        // The clip's file name only (players share logs).
        let name = super::nw_rq_file_name(&next.path);
        tracing::info!(clip = %name, "exporting a new clip (idle title screen)");
        if let Err(error) = self.export_clip_inner(&next.path, true) {
            tracing::warn!(%error, clip = %name, "automatic clip export could not start");
            self.tw_export.clips.failed.insert(next.path);
        }
    }

    /// After an export ended: if it was a clip's, close the clip.
    pub(super) fn tw_export_after_clip(&mut self, phase: &ExportPhase) {
        if self.tw_export.clips.open.is_some() {
            self.tw_export_clip_ended(matches!(phase, ExportPhase::Failed(_)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_videos_are_named_after_the_clip() {
        let game = Path::new("C:/game");
        let path = clip_video_path(
            game,
            Path::new("C:/game/replays/clips/2026-11-12 21.43 mc.example.net.aeroreplay"),
        );
        assert_eq!(
            path.file_name().unwrap().to_string_lossy(),
            "2026-11-12 21.43 mc.example.net.mp4"
        );
        assert!(path.parent().unwrap().ends_with("Clips"));
    }
}
