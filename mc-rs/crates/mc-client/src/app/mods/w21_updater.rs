//! The updater on the Aero title screen.
//!
//! - Only a packaged portable install updates itself (`mc_update::install::portable_install`:
//!   `app\aero-install.json` next to the exe). Every development build skips all of this after
//!   one cached file check.
//! - The "Check for updates" option (Aero Title Screen > Updates, on by default) starts the
//!   updater thread the first time the Aero title screen draws; it asks GitHub at most once per
//!   launch and once every 6 hours (`mc_update::service`). Off: no thread, no request, no card.
//!   With the Aero title screen off (vanilla's title screen) nothing is checked or shown.
//! - The card (`mc_ui::screens::aero_title::update_notice`, bottom left) shows what the updater
//!   says, and its buttons call back here: Update downloads (only a build with an update
//!   signing key installs; one without links the release page), "Restart to update" hands over
//!   to the update helper and quits (only from the title screen, so never with a world loaded).
//! - Per title-screen frame: one atomic load (the updater's generation), two more while a
//!   download runs; the view is rebuilt only when the status changes.

use mc_ui::screens::aero_title::update_notice::{NoticeButton, NoticeView, UpdateNotice};
use mc_update::service::{ServiceConfig, Status, Updater};

use crate::app::Client;

/// The `aero_title` mod's option.
pub(crate) const CHECK_OPTION: &str = "check_for_updates";

/// The client's side of the updater (`AeroTitleState::update`).
#[derive(Debug, Default)]
pub(crate) struct UpdateUi {
    /// The updater thread, while "Check for updates" is on in a portable install.
    updater: Option<Updater>,
    /// The settings generation the option was last read at (`0`: never).
    seen_settings: u64,
    /// No updater this launch: not a portable install, or this build's update setup is unusable.
    unavailable: bool,
    /// The updater generation `view` was built from.
    view_generation: u64,
    /// What the updater said last.
    status: Option<Status>,
    /// The card for `status`.
    view: Option<NoticeView>,
    /// "Later" on a ready update: hidden until the next launch (it stays staged).
    hidden_ready: Option<String>,
    /// Tests: a card shown without an updater, and its download progress (pictures).
    #[cfg(test)]
    pinned: Option<(u64, u64)>,
}

impl UpdateUi {
    /// Start or stop the updater when the option changed (a `u64` compare per frame otherwise).
    fn follow_setting(&mut self, settings: &mc_mods::ModSettings) {
        let generation = settings.generation();
        if self.seen_settings == generation {
            return;
        }
        self.seen_settings = generation;
        let wanted = settings.bool("aero_title", CHECK_OPTION);
        if !wanted {
            if self.updater.take().is_some() {
                self.status = None;
                self.view = None;
                self.view_generation = 0;
            }
            return;
        }
        if self.updater.is_some() || self.unavailable {
            return;
        }
        let Some(paths) = mc_update::install::portable_install() else {
            self.unavailable = true;
            return;
        };
        let started = ServiceConfig::shipped().and_then(|config| {
            if !config.can_install() {
                tracing::info!("no update signing key in this build: updates are announced only");
            }
            Updater::start(paths.clone(), config)
        });
        match started {
            Ok(updater) => {
                updater.check();
                self.updater = Some(updater);
            }
            Err(error) => {
                tracing::warn!(%error, "the updater is off this launch");
                self.unavailable = true;
            }
        }
    }

    /// The card for the updater's status.
    fn rebuild_view(&mut self, status: &Status, can_install: bool) {
        let hidden = |version: &str| self.hidden_ready.as_deref() == Some(version);
        self.view = match status {
            Status::Idle => None,
            Status::Available { version, notes_url } => {
                let mut buttons = Vec::with_capacity(3);
                if can_install {
                    buttons.push(NoticeButton::Update);
                }
                if notes_url.is_some() {
                    buttons.push(NoticeButton::ReleaseNotes);
                }
                buttons.push(NoticeButton::Later);
                Some(NoticeView {
                    title: format!("Aero Client {version} is available"),
                    detail: if can_install {
                        String::new()
                    } else {
                        "Get it from the release page.".to_owned()
                    },
                    progress: false,
                    buttons,
                })
            }
            Status::Downloading { version } => Some(NoticeView {
                title: format!("Downloading Aero Client {version}"),
                detail: String::new(),
                progress: true,
                buttons: vec![NoticeButton::Cancel],
            }),
            Status::Installing { version } => Some(NoticeView {
                title: format!("Installing Aero Client {version}"),
                detail: "Checking the download...".to_owned(),
                progress: false,
                buttons: Vec::new(),
            }),
            Status::Ready { version, .. } if hidden(version) => None,
            Status::Ready { version, .. } => Some(NoticeView {
                title: format!("Aero Client {version} is ready"),
                detail: "Restart Aero Client to finish updating.".to_owned(),
                progress: false,
                buttons: vec![NoticeButton::Restart, NoticeButton::Later],
            }),
            Status::Failed { version, message } => Some(NoticeView {
                title: format!("Couldn't update to {version}"),
                detail: (*message).to_owned(),
                progress: false,
                buttons: vec![NoticeButton::TryAgain, NoticeButton::Later],
            }),
            // "won't be offered again" only when it is not.
            Status::RolledBack {
                version,
                kept,
                offered_again: false,
            } => Some(NoticeView {
                title: format!("Aero Client {version} didn't start"),
                detail: format!("You're back on {kept}. It won't be offered again."),
                progress: false,
                buttons: vec![NoticeButton::Ok],
            }),
            Status::RolledBack {
                version,
                kept,
                offered_again: true,
            } => Some(NoticeView {
                title: format!("Aero Client {version} wasn't installed"),
                detail: format!("You're still on {kept}. It will be offered again."),
                progress: false,
                buttons: vec![NoticeButton::Ok],
            }),
        };
    }

    /// The release page of the version on offer.
    fn notes_url(&self) -> Option<&str> {
        match self.status.as_ref()? {
            Status::Available { notes_url, .. } | Status::Ready { notes_url, .. } => {
                notes_url.as_deref()
            }
            _ => None,
        }
    }
}

/// Bring the title screen's update card up to date (title-screen
/// frames only; see the module docs for the cost).
pub(crate) fn sync_notice(
    ui: &mut UpdateUi,
    notice: &mut UpdateNotice,
    settings: &mc_mods::ModSettings,
) {
    ui.follow_setting(settings);
    #[cfg(test)]
    if let Some((done, total)) = ui.pinned {
        if notice.generation() != ui.view_generation {
            notice.set_view(ui.view_generation, ui.view.clone());
        }
        notice.set_progress(done, total);
        return;
    }
    let Some(updater) = ui.updater.as_ref() else {
        if notice.generation() != 0 {
            notice.set_view(0, None);
        }
        return;
    };
    let generation = updater.generation();
    let progress =
        matches!(ui.status, Some(Status::Downloading { .. })).then(|| updater.progress());
    if generation != ui.view_generation {
        let status = updater.status();
        let can_install = updater.can_install();
        ui.view_generation = generation;
        ui.rebuild_view(&status, can_install);
        ui.status = Some(status);
    }
    if notice.generation() != ui.view_generation {
        notice.set_view(ui.view_generation, ui.view.clone());
    }
    if let Some((done, total)) = progress {
        notice.set_progress(done, total);
    }
}

/// Open the release page in the player's browser (only `https://github.com/...`).
fn open_release_page(url: &str) {
    if !mc_update::github::release_page_ok(url) {
        return;
    }
    #[cfg(target_os = "windows")]
    {
        let spawned = std::process::Command::new("rundll32.exe")
            .arg("url.dll,FileProtocolHandler")
            .arg(url)
            .spawn();
        if let Err(error) = spawned {
            tracing::warn!(%error, "couldn't open the release page");
        }
    }
}

impl Client {
    /// A mouse edge on the Aero title screen: `Some(true)` when the
    /// update card took it (a press on the card, or the release of one), `None` to let the rest
    /// of the screen have it.
    pub(crate) fn upd_title_mouse(&mut self, button: i32, pressed: bool) -> Option<bool> {
        let (screen, (x, y)) = self.aero.live_screen()?;
        let button = u8::try_from(button).ok()?;
        if pressed {
            return screen.notice.mouse_pressed(x, y, button).then_some(true);
        }
        if !screen.notice.has_press() {
            return None;
        }
        if let Some(clicked) = screen.notice.mouse_released(x, y, button) {
            self.play_button_press();
            self.upd_clicked(clicked);
        }
        Some(true)
    }

    /// One of the card's buttons was clicked.
    fn upd_clicked(&mut self, clicked: NoticeButton) {
        let ui = &mut self.aero.update;
        let Some(updater) = ui.updater.as_ref() else {
            return;
        };
        match clicked {
            NoticeButton::Update | NoticeButton::TryAgain => updater.download(),
            NoticeButton::Cancel => updater.cancel(),
            NoticeButton::ReleaseNotes => {
                if let Some(url) = ui.notes_url() {
                    open_release_page(url);
                }
            }
            NoticeButton::Later => match ui.status.as_ref() {
                // Already downloaded: just not now (offered again next launch).
                Some(Status::Ready { version, .. }) => {
                    ui.hidden_ready = Some(version.clone());
                    ui.view = None;
                    ui.view_generation = 0;
                }
                _ => updater.later(),
            },
            NoticeButton::Ok => updater.dismiss_failure(),
            NoticeButton::Restart => {
                // Never with a world loaded (the card only shows on the title screen anyway).
                if self.session.is_some() {
                    return;
                }
                match updater.restart_to_update() {
                    Ok(()) => {
                        tracing::info!("restarting to update");
                        self.quit();
                    }
                    Err(error) => tracing::warn!(%error, "couldn't start the update"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(status: Status, can_install: bool) -> Option<NoticeView> {
        let mut ui = UpdateUi::default();
        ui.rebuild_view(&status, can_install);
        ui.view
    }

    #[test]
    fn each_status_has_its_card() {
        let notes = Some("https://github.com/o/r/releases/tag/v0.2.0".to_owned());
        let available = Status::Available {
            version: "0.2.0".to_owned(),
            notes_url: notes.clone(),
        };
        let card = view(available.clone(), true).unwrap();
        assert_eq!(card.title, "Aero Client 0.2.0 is available");
        assert_eq!(
            card.buttons,
            [
                NoticeButton::Update,
                NoticeButton::ReleaseNotes,
                NoticeButton::Later
            ]
        );
        // A build without a signing key only announces.
        let card = view(available, false).unwrap();
        assert_eq!(
            card.buttons,
            [NoticeButton::ReleaseNotes, NoticeButton::Later]
        );
        assert_eq!(card.detail, "Get it from the release page.");

        let card = view(
            Status::Downloading {
                version: "0.2.0".to_owned(),
            },
            true,
        )
        .unwrap();
        assert!(card.progress);
        assert_eq!(card.buttons, [NoticeButton::Cancel]);
        let card = view(
            Status::Ready {
                version: "0.2.0".to_owned(),
                notes_url: notes,
            },
            true,
        )
        .unwrap();
        assert_eq!(card.title, "Aero Client 0.2.0 is ready");
        assert_eq!(card.buttons, [NoticeButton::Restart, NoticeButton::Later]);
        let card = view(
            Status::RolledBack {
                version: "0.2.0".to_owned(),
                kept: "0.1.0".to_owned(),
                offered_again: false,
            },
            true,
        )
        .unwrap();
        assert_eq!(card.buttons, [NoticeButton::Ok]);
        assert!(
            card.detail.contains("won't be offered again"),
            "{}",
            card.detail
        );
        // Not applied for a passing reason: no such promise.
        let card = view(
            Status::RolledBack {
                version: "0.2.0".to_owned(),
                kept: "0.1.0".to_owned(),
                offered_again: true,
            },
            true,
        )
        .unwrap();
        assert_eq!(card.buttons, [NoticeButton::Ok]);
        assert!(!card.detail.contains("won't"), "{}", card.detail);
        assert_eq!(view(Status::Idle, true), None);
    }

    /// Pictures of the card on the real title screen (real assets, headless GPU), each status's
    /// card exactly as `rebuild_view` makes it, at 1080p and 1440p, plus one with the pointer on
    /// Update. Set `MC_UPD_REVIEW_DIR`; skips without it, the vanilla jar or a GPU adapter.
    #[test]
    fn update_card_review_renders() {
        use mc_render::{Gpu, GpuOptions, PhaseList};
        use std::time::Duration;

        use crate::app::config::{ClientConfig, LoginMode};

        let Some(out) = std::env::var_os("MC_UPD_REVIEW_DIR").map(std::path::PathBuf::from) else {
            eprintln!("skip: set MC_UPD_REVIEW_DIR to write the pictures");
            return;
        };
        std::fs::create_dir_all(&out).unwrap();
        let notes = Some("https://github.com/o/r/releases/tag/v0.2.0".to_owned());
        let v = |text: &str| text.to_owned();
        let cases: Vec<(&str, Status, bool, (u64, u64))> = vec![
            (
                "available",
                Status::Available {
                    version: v("0.2.0"),
                    notes_url: notes.clone(),
                },
                true,
                (0, 0),
            ),
            (
                "available-no-key",
                Status::Available {
                    version: v("0.2.0"),
                    notes_url: notes.clone(),
                },
                false,
                (0, 0),
            ),
            (
                "downloading",
                Status::Downloading {
                    version: v("0.2.0"),
                },
                true,
                (8_820_000, 21_000_000),
            ),
            (
                "ready",
                Status::Ready {
                    version: v("0.2.0"),
                    notes_url: notes.clone(),
                },
                true,
                (0, 0),
            ),
            (
                "failed",
                Status::Failed {
                    version: v("0.2.0"),
                    message: "The download was damaged. Try again.",
                },
                true,
                (0, 0),
            ),
            (
                "rolled-back",
                Status::RolledBack {
                    version: v("0.2.0"),
                    kept: v("0.1.0"),
                    offered_again: false,
                },
                true,
                (0, 0),
            ),
        ];
        // Also small and narrow windows, where the card (bottom left)
        // and the account corner (top right) are closest to the button column.
        for (w, h, tag) in [
            (1920_u32, 1080_u32, "1080p"),
            (2560, 1440, "1440p"),
            (1280, 720, "720p"),
            (854, 480, "480p"),
            (1024, 768, "1024x768"),
        ] {
            let options = GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            };
            let mut gpu = match Gpu::headless(w, h, options) {
                Ok(gpu) => gpu,
                Err(mc_render::RenderError::NoAdapter) => {
                    eprintln!("skip: no wgpu adapter");
                    return;
                }
                Err(error) => panic!("headless GPU: {error}"),
            };
            let dir = tempfile::tempdir().unwrap();
            let config = ClientConfig {
                game_dir: dir.path().to_path_buf(),
                login: LoginMode::Offline("Dev".to_owned()),
                ..ClientConfig::default()
            }
            .with_discovered_assets();
            if !config.version_jar.as_ref().is_some_and(|jar| jar.is_file()) {
                eprintln!("skip: no vanilla jar (set MC_JARS_DIR)");
                return;
            }
            let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets load");
            let mut phases = PhaseList::new();
            client.install(&gpu, &mut phases, w, h);
            let mut camera = crate::app::driver::menu_camera();
            camera.set_viewport(w, h);
            client.frame(&gpu, &mut camera, 0.0);
            let settled = (Duration::from_secs(12), Duration::from_secs(11));
            for (index, (name, status, can_install, progress)) in cases.iter().enumerate() {
                for hover in [false, true] {
                    if hover && *name != "available" {
                        continue;
                    }
                    let ui = &mut client.aero.update;
                    ui.rebuild_view(status, *can_install);
                    ui.status = Some(status.clone());
                    ui.view_generation = 1000 + index as u64 * 2 + u64::from(hover);
                    ui.pinned = Some(*progress);
                    client.aero.set_live_clocks(settled.0, settled.1);
                    client.frame(&gpu, &mut camera, 0.0);
                    if hover {
                        let (screen, _) = client.aero.live_screen().expect("title screen");
                        let (x, y) = screen.notice.button_rects()[0].center();
                        client.aero_title_pointer((f64::from(x), f64::from(y)));
                    }
                    // A few frames: uploads done, the card's own pop-in (200 ms, real time) and
                    // a hover eased in.
                    for _ in 0..3 {
                        client.aero.set_live_clocks(settled.0, settled.1);
                        client.frame(&gpu, &mut camera, 0.0);
                        std::thread::sleep(Duration::from_millis(120));
                    }
                    client.aero.set_live_clocks(settled.0, settled.1);
                    client.frame(&gpu, &mut camera, 0.0);
                    let card = client
                        .aero
                        .live_screen()
                        .and_then(|(screen, _)| screen.notice.card())
                        .expect("the card is on screen");
                    assert!(card.x > 0.0 && card.bottom() < h as f32, "{name}: {card:?}");
                    assert!(card.right() < w as f32 * 0.5, "{name}: left half");
                    assert!(card.y > h as f32 * 0.6, "{name}: bottom part: {card:?}");
                    let target = gpu.acquire().expect("offscreen frame");
                    phases.render(&gpu, target.view(), &camera, 0.0);
                    let pixels = gpu.read_pixels().expect("readback");
                    let suffix = if hover { "-hover" } else { "" };
                    let path = out.join(format!("update-card-{name}{suffix}-{tag}.png"));
                    image::save_buffer(&path, &pixels, w, h, image::ExtendedColorType::Rgba8)
                        .expect("write the PNG");
                    eprintln!("wrote {}", path.display());
                    client.aero_title_pointer((0.0, 0.0));
                }
            }
        }
    }

    #[test]
    fn a_development_build_never_starts_the_updater() {
        // The test binary is not in a portable install: the option is on by default, and still
        // nothing starts.
        let settings = mc_mods::ModSettings::new();
        assert!(settings.bool("aero_title", CHECK_OPTION));
        let mut ui = UpdateUi::default();
        let mut notice = UpdateNotice::default();
        sync_notice(&mut ui, &mut notice, &settings);
        assert!(ui.updater.is_none());
        assert!(ui.unavailable);
        assert_eq!(notice.generation(), 0);
    }
}
