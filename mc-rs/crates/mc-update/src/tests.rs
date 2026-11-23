//! Shared test fixtures, and the updater end to end against a local mock HTTP server (never
//! GitHub): check, offer, download, verify (digest, `.sha256`, signature), stage, and refusals.

use std::cell::{Cell, RefCell};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::helper::Hooks;
use crate::install::{InstallMarker, InstallPaths, APP_DIR, EXE_NAME, LAUNCHER_CMD, MARKER_FILE};
use crate::net::HttpPolicy;
use crate::service::{ServiceConfig, Status, Updater};
use crate::state::{self, StagedUpdate};
use crate::version::Version;

// ---- fixtures -----------------------------------------------------------------------------

/// A fake portable install of `version` in `dir`: `Aero Client\app\{mc-client.exe, launcher.ps1,
/// aero-install.json}`, `Play Aero Client.cmd` and `README.txt`, each naming the version.
pub(crate) fn fake_install(dir: &Path, version: &str) -> InstallPaths {
    let root = dir.join("Aero Client");
    write_package_files(&root, version);
    InstallPaths::at_root(&root).unwrap()
}

fn write_package_files(root: &Path, version: &str) {
    let app = root.join(APP_DIR);
    std::fs::create_dir_all(&app).unwrap();
    std::fs::write(app.join(EXE_NAME), format!("exe {version}")).unwrap();
    std::fs::write(app.join("launcher.ps1"), "# launcher").unwrap();
    std::fs::write(app.join(MARKER_FILE), InstallMarker::new(version).to_json()).unwrap();
    std::fs::write(root.join(LAUNCHER_CMD), format!("@rem cmd {version}")).unwrap();
    std::fs::write(root.join("README.txt"), format!("readme {version}")).unwrap();
}

/// A staged release of `version` in `paths`' staging folder; returns its `Aero Client` folder.
pub(crate) fn fake_staged(paths: &InstallPaths, version: &str) -> PathBuf {
    let root = paths.staged_root(version);
    write_package_files(&root, version);
    root
}

/// [`fake_staged`], recorded in the state as staged (with its exe's hash).
pub(crate) fn fake_staged_recorded(paths: &InstallPaths, version: &str) -> PathBuf {
    let root = fake_staged(paths, version);
    let exe_sha256 = crate::verify::sha256_file(&root.join(APP_DIR).join(EXE_NAME)).unwrap();
    state::update(paths, |state| {
        state.staged = Some(StagedUpdate {
            version: version.to_owned(),
            exe_sha256,
            zip_sha256: "0".repeat(64),
        });
    })
    .unwrap();
    root
}

/// One zip entry.
#[derive(Clone)]
pub(crate) enum ZipItem {
    File(String, Vec<u8>),
    Symlink(String, String),
}

impl ZipItem {
    pub(crate) fn file(name: &str, bytes: &[u8]) -> ZipItem {
        ZipItem::File(name.to_owned(), bytes.to_vec())
    }

    pub(crate) fn symlink(name: &str, target: &str) -> ZipItem {
        ZipItem::Symlink(name.to_owned(), target.to_owned())
    }
}

/// A zip of `items`, deflated, names written exactly as given.
pub(crate) fn write_zip(items: &[ZipItem]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for item in items {
        match item {
            ZipItem::File(name, bytes) => {
                writer.start_file(name.as_str(), options).unwrap();
                writer.write_all(bytes).unwrap();
            }
            ZipItem::Symlink(name, target) => {
                writer
                    .add_symlink(name.as_str(), target.as_str(), options)
                    .unwrap();
            }
        }
    }
    writer.finish().unwrap().into_inner()
}

/// The release zip of `version` as the packaging script lays it out, plus `extra`.
pub(crate) fn package_zip(version: &str, extra: &[ZipItem]) -> Vec<u8> {
    let marker = InstallMarker::new(version).to_json();
    let mut items = vec![
        ZipItem::file(
            "Aero Client/Play Aero Client.cmd",
            format!("@rem cmd {version}").as_bytes(),
        ),
        ZipItem::file(
            "Aero Client/README.txt",
            format!("readme {version}").as_bytes(),
        ),
        ZipItem::file(
            "Aero Client/app/mc-client.exe",
            format!("exe {version}").as_bytes(),
        ),
        ZipItem::file("Aero Client/app/launcher.ps1", b"# launcher"),
        ZipItem::file("Aero Client/app/aero-install.json", marker.as_bytes()),
        ZipItem::file("Aero Client/app/licenses/NOTICE.txt", b"notice"),
    ];
    items.extend(extra.iter().cloned());
    write_zip(&items)
}

/// Processes and time for [`crate::helper::apply`], scripted: the old client has exited (unless
/// told otherwise), a launched client does whatever `on_launch` does and is gone by the next
/// poll, and `sleep` advances a virtual clock.
pub(crate) struct FakeHooks {
    base: Instant,
    elapsed: Cell<Duration>,
    old_running: Cell<bool>,
    self_test_ok: Cell<bool>,
    self_tests: Cell<u32>,
    launches: Cell<u32>,
    #[allow(clippy::type_complexity)]
    on_launch: RefCell<Box<dyn Fn(&InstallPaths)>>,
}

impl FakeHooks {
    pub(crate) fn new(_paths: &InstallPaths) -> FakeHooks {
        FakeHooks {
            base: Instant::now(),
            elapsed: Cell::new(Duration::ZERO),
            old_running: Cell::new(false),
            self_test_ok: Cell::new(true),
            self_tests: Cell::new(0),
            launches: Cell::new(0),
            on_launch: RefCell::new(Box::new(|_| {})),
        }
    }

    pub(crate) fn on_launch(&self, action: impl Fn(&InstallPaths) + 'static) {
        *self.on_launch.borrow_mut() = Box::new(action);
    }

    pub(crate) fn self_test_passes(&self, ok: bool) {
        self.self_test_ok.set(ok);
    }

    pub(crate) fn old_client_never_exits(&self) {
        self.old_running.set(true);
    }

    pub(crate) fn launches(&self) -> u32 {
        self.launches.get()
    }

    pub(crate) fn self_tests(&self) -> u32 {
        self.self_tests.get()
    }
}

impl Hooks for FakeHooks {
    fn client_running(&self, _exe: &Path) -> bool {
        self.old_running.get()
    }

    fn self_test(&self, exe: &Path, version: &str) -> bool {
        self.self_tests.set(self.self_tests.get() + 1);
        assert!(exe.is_file(), "the self-test runs the staged exe");
        assert!(!version.is_empty());
        self.self_test_ok.get()
    }

    fn launch(&self, paths: &InstallPaths) -> std::io::Result<()> {
        self.launches.set(self.launches.get() + 1);
        (self.on_launch.borrow())(paths);
        Ok(())
    }

    fn sleep(&self, duration: Duration) {
        self.elapsed.set(self.elapsed.get() + duration);
    }

    fn now(&self) -> Instant {
        self.base + self.elapsed.get()
    }
}

// ---- the service against a mock server -----------------------------------------------------

const ZIP_NAME: &str = "Aero-Client-0.2.0-windows-x64.zip";

fn sha256(bytes: &[u8]) -> String {
    crate::verify::hex(ring::digest::digest(&ring::digest::SHA256, bytes).as_ref())
}

fn config(server: &MockServer, key: Option<[u8; 32]>) -> ServiceConfig {
    ServiceConfig {
        repo: "o/r".to_owned(),
        policy: HttpPolicy::loopback(&server.uri()),
        public_key: key,
        current: Version::new(0, 1, 0),
        check_interval: crate::CHECK_INTERVAL,
        // The SHA-256 paths are tested without a key too; the shipped config never allows it.
        allow_unsigned_install: true,
    }
}

/// The API's answer for release 0.2.0 whose zip is `zip`, served at `/dl/<name>`.
fn release_json(server: &MockServer, zip: &[u8], digest: bool, extra: &[Value]) -> Value {
    let mut assets = vec![json!({
        "name": ZIP_NAME,
        "size": zip.len(),
        "browser_download_url": format!("{}/dl/{ZIP_NAME}", server.uri()),
        "digest": digest.then(|| format!("sha256:{}", sha256(zip))),
        "state": "uploaded"
    })];
    assets.extend(extra.iter().cloned());
    json!({
        "tag_name": "v0.2.0",
        "draft": false,
        "prerelease": false,
        "html_url": "https://github.com/o/r/releases/tag/v0.2.0",
        "assets": assets
    })
}

fn sidecar(server: &MockServer, name: &str, size: usize) -> Value {
    json!({
        "name": name,
        "size": size,
        "browser_download_url": format!("{}/dl/{name}", server.uri()),
        "state": "uploaded"
    })
}

async fn serve(server: &MockServer, url_path: &str, response: ResponseTemplate) {
    Mock::given(method("GET"))
        .and(path(url_path))
        .respond_with(response)
        .mount(server)
        .await;
}

async fn serve_release(server: &MockServer, release: &Value) {
    serve(
        server,
        "/repos/o/r/releases/latest",
        ResponseTemplate::new(200).set_body_json(release),
    )
    .await;
}

/// Wait until the updater thread has finished `count` commands (the start counts as one).
async fn settle(updater: &Updater, count: u64) -> Status {
    for _ in 0..800 {
        if updater.processed() >= count {
            return updater.status();
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the updater did not finish; status {:?}", updater.status());
}

fn exe_in_staging(paths: &InstallPaths) -> String {
    std::fs::read_to_string(paths.staged_root("0.2.0").join(APP_DIR).join(EXE_NAME)).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_update_is_found_downloaded_verified_and_staged() {
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    let zip = package_zip("0.2.0", &[]);
    serve_release(&server, &release_json(&server, &zip, true, &[])).await;
    // The download redirects once (as GitHub's does), on an allowed host.
    serve(
        &server,
        &format!("/dl/{ZIP_NAME}"),
        ResponseTemplate::new(302)
            .insert_header("Location", format!("{}/files/x.zip", server.uri())),
    )
    .await;
    serve(
        &server,
        "/files/x.zip",
        ResponseTemplate::new(200).set_body_bytes(zip.clone()),
    )
    .await;

    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    assert_eq!(
        settle(&updater, 1).await,
        Status::Idle,
        "nothing before the check"
    );
    updater.check();
    let status = settle(&updater, 2).await;
    assert_eq!(
        status,
        Status::Available {
            version: "0.2.0".to_owned(),
            notes_url: Some("https://github.com/o/r/releases/tag/v0.2.0".to_owned()),
        }
    );
    assert!(
        !paths.download_dir().exists(),
        "nothing downloaded before Update"
    );
    updater.download();
    let status = settle(&updater, 3).await;
    assert!(
        matches!(status, Status::Ready { ref version, .. } if version == "0.2.0"),
        "{status:?}"
    );
    assert_eq!(updater.progress(), (zip.len() as u64, zip.len() as u64));
    assert_eq!(exe_in_staging(&paths), "exe 0.2.0");
    let state = state::read(&paths);
    let staged = state.staged.expect("recorded");
    assert_eq!(staged.version, "0.2.0");
    assert_eq!(staged.zip_sha256, sha256(&zip));
    assert!(state.last_check_unix.is_some());
    assert!(
        !paths.download_dir().exists(),
        "the zip is gone once staged"
    );
    assert_eq!(
        std::fs::read_to_string(paths.exe()).unwrap(),
        "exe 0.1.0",
        "nothing is swapped before a restart"
    );

    // A second check in the same launch asks nothing; a new launch shows "ready" from the
    // saved state without the network.
    updater.check();
    settle(&updater, 4).await;
    drop(updater);
    let again = Updater::start(paths.clone(), config(&server, None)).unwrap();
    assert!(matches!(settle(&again, 1).await, Status::Ready { .. }));
    let api_calls = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| request.url.path().contains("/releases/latest"))
        .count();
    assert_eq!(api_calls, 1, "one check per launch, and per interval");
}

/// A build without a signing key (the shipped policy) tells the player about the release and
/// never downloads it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_a_key_the_shipped_policy_only_announces() {
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    let zip = package_zip("0.2.0", &[]);
    serve_release(&server, &release_json(&server, &zip, true, &[])).await;
    Mock::given(method("GET"))
        .and(path(format!("/dl/{ZIP_NAME}")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(zip.clone()))
        .expect(0)
        .mount(&server)
        .await;
    let config = ServiceConfig {
        allow_unsigned_install: false,
        ..config(&server, None)
    };
    assert!(!config.can_install());
    let updater = Updater::start(paths.clone(), config).unwrap();
    assert!(!updater.can_install());
    updater.check();
    assert!(matches!(
        settle(&updater, 2).await,
        Status::Available { .. }
    ));
    updater.download();
    assert!(matches!(
        settle(&updater, 3).await,
        Status::Available { .. }
    ));
    assert!(!paths.download_dir().exists() && !paths.staging_root().exists());
    drop(updater);
    server.verify().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_signed_build_installs_only_a_correctly_signed_release() {
    use ring::signature::{Ed25519KeyPair, KeyPair};
    let key = Ed25519KeyPair::from_seed_unchecked(&[7_u8; 32]).unwrap();
    let public: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
    let zip = package_zip("0.2.0", &[]);
    let good = key.sign(crate::verify::signed_message(ZIP_NAME, "0.2.0", &sha256(&zip)).as_bytes());
    let other_version =
        key.sign(crate::verify::signed_message(ZIP_NAME, "0.3.0", &sha256(&zip)).as_bytes());
    let encode = |signature: &[u8]| {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(signature)
    };

    for (case, signature, expect_ready) in [
        ("good", Some(encode(good.as_ref())), true),
        (
            "signed for another version",
            Some(encode(other_version.as_ref())),
            false,
        ),
        ("garbage", Some("bm90IGEgc2lnbmF0dXJl".to_owned()), false),
        ("missing", None, false),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), "0.1.0");
        let server = MockServer::start().await;
        let sig_name = format!("{ZIP_NAME}.sig");
        let extra: Vec<Value> = signature
            .iter()
            .map(|text| sidecar(&server, &sig_name, text.len()))
            .collect();
        serve_release(&server, &release_json(&server, &zip, true, &extra)).await;
        serve(
            &server,
            &format!("/dl/{ZIP_NAME}"),
            ResponseTemplate::new(200).set_body_bytes(zip.clone()),
        )
        .await;
        if let Some(text) = &signature {
            serve(
                &server,
                &format!("/dl/{sig_name}"),
                ResponseTemplate::new(200).set_body_string(text.clone()),
            )
            .await;
        }
        let updater = Updater::start(paths.clone(), config(&server, Some(public))).unwrap();
        updater.check();
        let offered = settle(&updater, 2).await;
        if signature.is_none() {
            assert_eq!(
                offered,
                Status::Idle,
                "{case}: an unsigned release is not offered"
            );
            continue;
        }
        assert!(
            matches!(offered, Status::Available { .. }),
            "{case}: {offered:?}"
        );
        updater.download();
        let status = settle(&updater, 3).await;
        if expect_ready {
            assert!(matches!(status, Status::Ready { .. }), "{case}: {status:?}");
            assert_eq!(exe_in_staging(&paths), "exe 0.2.0");
        } else {
            assert_eq!(
                status,
                Status::Failed {
                    version: "0.2.0".to_owned(),
                    message: "The download isn't signed by Aero Client, so it wasn't used.",
                },
                "{case}"
            );
            assert!(!paths.staging_root().exists(), "{case}: nothing staged");
            assert_eq!(state::read(&paths).staged, None, "{case}");
        }
    }
}

/// Serve release 0.2.0 whose zip download answers `response`, download it, and return the
/// status and the install.
async fn download_with(
    zip: &[u8],
    response: ResponseTemplate,
) -> (Status, InstallPaths, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    serve_release(&server, &release_json(&server, zip, true, &[])).await;
    serve(&server, &format!("/dl/{ZIP_NAME}"), response).await;
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    updater.check();
    assert!(matches!(
        settle(&updater, 2).await,
        Status::Available { .. }
    ));
    updater.download();
    (settle(&updater, 3).await, paths, dir)
}

fn failed(message: &'static str) -> Status {
    Status::Failed {
        version: "0.2.0".to_owned(),
        message,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn damaged_or_oversized_downloads_are_refused() {
    let zip = package_zip("0.2.0", &[]);
    let damaged = "The download was damaged. Try again.";

    // Same length, one byte different: the SHA-256 differs.
    let mut flipped = zip.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xff;
    let (status, paths, _dir) =
        download_with(&zip, ResponseTemplate::new(200).set_body_bytes(flipped)).await;
    assert_eq!(status, failed(damaged));
    assert!(!paths.staging_root().exists() && !paths.download_dir().exists());

    // Longer and shorter than the release data says.
    let mut longer = zip.clone();
    longer.extend_from_slice(&[0; 1000]);
    let (status, _paths, _dir) =
        download_with(&zip, ResponseTemplate::new(200).set_body_bytes(longer)).await;
    assert_eq!(status, failed(damaged));
    let (status, _paths, _dir) = download_with(
        &zip,
        ResponseTemplate::new(200).set_body_bytes(zip[..zip.len() - 10].to_vec()),
    )
    .await;
    assert_eq!(status, failed(damaged));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_redirect_off_the_allowed_hosts_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    let zip = package_zip("0.2.0", &[]);
    serve_release(&server, &release_json(&server, &zip, true, &[])).await;
    // `localhost` is this same server, but not an allowed host name, and the redirect target
    // serves the real zip: only the host policy can stop it.
    let elsewhere = server.uri().replace("127.0.0.1", "localhost");
    serve(
        &server,
        &format!("/dl/{ZIP_NAME}"),
        ResponseTemplate::new(302).insert_header("Location", format!("{elsewhere}/files/x.zip")),
    )
    .await;
    serve(
        &server,
        "/files/x.zip",
        ResponseTemplate::new(200).set_body_bytes(zip.clone()),
    )
    .await;
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    updater.check();
    assert!(matches!(
        settle(&updater, 2).await,
        Status::Available { .. }
    ));
    updater.download();
    assert_eq!(
        settle(&updater, 3).await,
        failed("Couldn't reach GitHub. Try again later.")
    );
    assert!(!paths.staging_root().exists());
    let followed = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| request.url.path() == "/files/x.zip")
        .count();
    assert_eq!(followed, 0, "the refused hop was never requested");
    // An HTTP error status is a failure too.
    let (status, _paths, _dir) = download_with(&zip, ResponseTemplate::new(500)).await;
    assert_eq!(status, failed("Couldn't reach GitHub. Try again later."));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hostile_zip_with_a_good_hash_is_still_refused_at_staging() {
    let zip = package_zip(
        "0.2.0",
        &[ZipItem::file("Aero Client/app/../../../escaped.txt", b"x")],
    );
    let (status, paths, dir) =
        download_with(&zip, ResponseTemplate::new(200).set_body_bytes(zip.clone())).await;
    assert_eq!(
        status,
        failed("This release can't be installed here. Get it from the release page.")
    );
    assert!(!paths.staging_root().exists());
    assert!(!dir.path().join("escaped.txt").exists());
    assert!(!paths.update.join("escaped.txt").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sha256_asset_stands_in_for_a_missing_digest() {
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    let zip = package_zip("0.2.0", &[]);
    let sha_name = format!("{ZIP_NAME}.sha256");
    let line = format!("{}  {ZIP_NAME}\n", sha256(&zip));
    let extra = [sidecar(&server, &sha_name, line.len())];
    serve_release(&server, &release_json(&server, &zip, false, &extra)).await;
    serve(
        &server,
        &format!("/dl/{ZIP_NAME}"),
        ResponseTemplate::new(200).set_body_bytes(zip.clone()),
    )
    .await;
    serve(
        &server,
        &format!("/dl/{sha_name}"),
        ResponseTemplate::new(200).set_body_string(line),
    )
    .await;
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    updater.check();
    assert!(matches!(
        settle(&updater, 2).await,
        Status::Available { .. }
    ));
    updater.download();
    assert!(matches!(settle(&updater, 3).await, Status::Ready { .. }));
    assert_eq!(exe_in_staging(&paths), "exe 0.2.0");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_check_honours_the_interval_and_quiet_answers() {
    // Checked an hour ago: no request at all, the release found then is offered.
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    let zip = package_zip("0.2.0", &[]);
    let release = crate::github::parse_release(
        release_json(&server, &zip, true, &[])
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    let info = crate::github::choose_release(
        &release,
        &Version::new(0, 1, 0),
        &HttpPolicy::loopback(&server.uri()),
        false,
    )
    .unwrap();
    state::update(&paths, |state| {
        state.last_check_unix = Some(crate::now_unix() - 3600);
        state.known_release = info;
    })
    .unwrap();
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    assert!(
        matches!(settle(&updater, 1).await, Status::Available { .. }),
        "from the saved state"
    );
    updater.check();
    settle(&updater, 2).await;
    drop(updater);
    server.verify().await;

    // No release (404) and a rate limit (403): nothing offered, the check is recorded.
    for status_code in [404_u16, 403] {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), "0.1.0");
        let server = MockServer::start().await;
        serve(
            &server,
            "/repos/o/r/releases/latest",
            ResponseTemplate::new(status_code),
        )
        .await;
        let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
        updater.check();
        assert_eq!(settle(&updater, 2).await, Status::Idle, "{status_code}");
        assert!(
            state::read(&paths).last_check_unix.is_some(),
            "{status_code}"
        );
    }

    // A server error is not recorded as a check (the next launch tries again).
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    serve(
        &server,
        "/repos/o/r/releases/latest",
        ResponseTemplate::new(502),
    )
    .await;
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    updater.check();
    assert_eq!(settle(&updater, 2).await, Status::Idle);
    assert_eq!(state::read(&paths).last_check_unix, None);
}

/// A hand-edited `state.json` can't point the download at another file name or a huge size.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tampered_saved_release_is_not_downloaded() {
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    let zip = package_zip("0.2.0", &[]);
    serve_release(&server, &release_json(&server, &zip, true, &[])).await;
    Mock::given(method("GET"))
        .and(path(format!("/dl/{ZIP_NAME}")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(zip.clone()))
        .expect(0)
        .mount(&server)
        .await;
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    updater.check();
    assert!(matches!(
        settle(&updater, 2).await,
        Status::Available { .. }
    ));
    let tampers: [fn(&mut crate::github::ReleaseInfo); 2] = [
        |release| release.zip.name = "..\\..\\evil.exe".to_owned(),
        |release| release.zip.size = u64::MAX,
    ];
    for tamper in tampers {
        state::update(&paths, |state| {
            tamper(state.known_release.as_mut().unwrap())
        })
        .unwrap();
        let before = updater.processed();
        updater.download();
        assert!(matches!(
            settle(&updater, before + 1).await,
            Status::Available { .. }
        ));
        assert!(!paths.download_dir().exists());
        assert!(!dir.path().join("evil.exe.part").exists());
    }
    drop(updater);
    server.verify().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn later_hides_that_version_for_good() {
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    let zip = package_zip("0.2.0", &[]);
    serve_release(&server, &release_json(&server, &zip, true, &[])).await;
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    updater.check();
    assert!(matches!(
        settle(&updater, 2).await,
        Status::Available { .. }
    ));
    updater.later();
    assert_eq!(settle(&updater, 3).await, Status::Idle);
    assert_eq!(
        state::read(&paths).dismissed_version.as_deref(),
        Some("0.2.0")
    );
    drop(updater);
    // The next launch remembers.
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    assert_eq!(settle(&updater, 1).await, Status::Idle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rolled_back_update_is_reported_once() {
    let dir = tempfile::tempdir().unwrap();
    let paths = fake_install(dir.path(), "0.1.0");
    let server = MockServer::start().await;
    state::update(&paths, |state| {
        state.last_failure = Some(state::Failure {
            version: "0.2.0".to_owned(),
            kept_version: "0.1.0".to_owned(),
            reason: "it closed before its title screen".to_owned(),
            shown: false,
        });
        state.bad_versions.push("0.2.0".to_owned());
    })
    .unwrap();
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    assert_eq!(
        settle(&updater, 1).await,
        Status::RolledBack {
            version: "0.2.0".to_owned(),
            kept: "0.1.0".to_owned(),
            offered_again: false,
        }
    );
    updater.dismiss_failure();
    assert_eq!(settle(&updater, 2).await, Status::Idle);
    drop(updater);
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    assert_eq!(settle(&updater, 1).await, Status::Idle, "said once");
    drop(updater);

    // An update that was not applied for a passing reason (the swap
    // found a file held open) is not marked as failing here: the card must not say it won't
    // be offered again.
    state::update(&paths, |state| {
        state.last_failure = Some(state::Failure {
            version: "0.3.0".to_owned(),
            kept_version: "0.1.0".to_owned(),
            reason: "the swap failed".to_owned(),
            shown: false,
        });
    })
    .unwrap();
    let updater = Updater::start(paths.clone(), config(&server, None)).unwrap();
    assert_eq!(
        settle(&updater, 1).await,
        Status::RolledBack {
            version: "0.3.0".to_owned(),
            kept: "0.1.0".to_owned(),
            offered_again: true,
        }
    );
}
