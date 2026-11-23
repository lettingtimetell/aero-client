//! A sign-in token given to the client never reaches what players share: the
//! log (stdout, which the zip's launcher saves as `logs/latest-launch.log`), stderr (saved as
//! `logs/latest-launch.errors.log`) or a crash report.
//!
//! The test runs a child process (this test binary again, picked by an environment variable)
//! that sets up logging and the panic hook the way `mc-client`'s `main` does and receives an
//! obviously fake token through `--accessToken`. The child then does the careless things a future
//! change might do: it logs the token directly, logs an error that quotes it, panics on a worker
//! thread with the token in the message, and writes the crash report for that panic as the game
//! loop would. The parent reads every byte the child wrote to stdout and stderr, and the report
//! on disk.
//!
//! Controls: every line that carries the token also carries a plain marker, which must come
//! through verbatim (so the lines were really written and read), and the token's place must hold
//! `<redacted>` (so it was replaced, not dropped with its line).

use std::path::Path;
use std::process::Command;

use mc_client::app::config::ClientConfig;
use mc_client::app::crash::{self, CrashExit};

/// Set (to the child's game folder) only in the child process.
const CHILD_ENV: &str = "MC_W21_SA_SECRETS_CHILD";

/// Obviously fake, but shaped like a Minecraft access token (a JWT: header, claims, signature).
const FAKE_TOKEN: &str = "eyJraWQiOiJmYWtlIiwiYWxnIjoiUlMyNTYifQ.\
                          eyJ4dWlkIjoiZmFrZSIsInRlc3QiOiJ3MjEtc2EtZmFrZSJ9.\
                          ZmFrZS1zaWduYXR1cmUtZm9yLWEtdGVzdC1vbmx5LW5vdC1hLXRva2Vu";

/// Never registered as a secret: it must come through verbatim.
const PLAIN_MARKER: &str = "w21-sa-plain-marker-7f3e";

/// The negative control: token-shaped, logged the same careless way, but never registered. It
/// must come through verbatim, which shows that a token the client failed to register would be
/// caught by this test.
const UNREGISTERED_CONTROL: &str = "eyJjb250cm9sIjoidW5yZWdpc3RlcmVkIn0.bmVnYXRpdmUtY29udHJvbA";

/// Fake tokens in the wrong place on the command line: `--accessToken=<token>` (a form the parser
/// does not read) and a bare argument. Neither is ever registered, so only the parser's own
/// logging keeps them out of the log.
const STRAY_EQUALS_TOKEN: &str = "eyJzdHJheSI6ImVxdWFscy1mb3JtIn0.ZmFrZS1zdHJheS1lcXVhbHMtdG9rZW4";
const STRAY_BARE_TOKEN: &str =
    "eyJzdHJheSI6ImJhcmUtYXJndW1lbnQifQ.ZmFrZS1zdHJheS1iYXJlLWFyZ3VtZW50LXRva2Vu.c2ln";

#[test]
fn a_fake_token_never_reaches_the_log_stderr_or_a_crash_report() {
    if std::env::var_os(CHILD_ENV).is_some() {
        // Inside the child this test has nothing to do; `child_process_body` runs instead.
        return;
    }
    let game_dir = tempfile::tempdir().expect("tempdir");
    let output = Command::new(std::env::current_exe().expect("this test binary"))
        .args([
            "child_process_body",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, game_dir.path())
        .env("RUST_LOG", "info")
        .env("MC_NO_AUDIO", "1")
        .output()
        .expect("run the child process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "the child failed\n--- stdout\n{stdout}\n--- stderr\n{stderr}"
    );

    // The controls: the lines were written, read, and the token's place says `<redacted>`.
    assert!(
        stdout.contains(&format!("{PLAIN_MARKER} joining with <redacted>")),
        "the careless log line is missing or unredacted\n{stdout}"
    );
    assert!(
        stdout.contains(&format!("{PLAIN_MARKER} refresh failed for <redacted>")),
        "the crash report printed to stdout is missing or unredacted\n{stdout}"
    );
    assert!(
        stderr.contains(&format!("{PLAIN_MARKER} refresh failed for <redacted>")),
        "the panic message on stderr is missing or unredacted\n{stderr}"
    );
    assert!(
        stdout.contains("unrecognised option, ignored")
            && stdout.contains("completely ignored long argument"),
        "the parser's warnings about the stray arguments are missing\n{stdout}"
    );
    assert!(
        stdout.contains(&format!("{PLAIN_MARKER} control {UNREGISTERED_CONTROL}")),
        "the negative control (an unregistered token) must come through verbatim\n{stdout}"
    );

    let reports: Vec<_> = std::fs::read_dir(game_dir.path().join("crash-reports"))
        .expect("the child wrote a crash-reports folder")
        .map(|entry| entry.expect("entry").path())
        .collect();
    assert_eq!(reports.len(), 1, "one crash report: {reports:?}");
    let report = std::fs::read_to_string(&reports[0]).expect("read the crash report");
    assert!(
        report.contains(&format!("{PLAIN_MARKER} refresh failed for <redacted>")),
        "{report}"
    );

    // And the token itself, or any of its three parts, appears nowhere.
    let token = FAKE_TOKEN.replace(' ', "");
    for (what, text) in [
        ("stdout", &*stdout),
        ("stderr", &*stderr),
        ("the crash report", report.as_str()),
    ] {
        assert!(!text.contains(&token), "{what} holds the token:\n{text}");
        for part in token
            .split('.')
            .chain(STRAY_EQUALS_TOKEN.split('.'))
            .chain(STRAY_BARE_TOKEN.split('.'))
        {
            assert!(
                !text.contains(part),
                "{what} holds part of a token:\n{text}"
            );
        }
    }
}

/// The child process: wired like `main`, then careless with the token on purpose.
#[test]
fn child_process_body() {
    let Some(game_dir) = std::env::var_os(CHILD_ENV) else {
        return;
    };
    let game_dir = Path::new(&game_dir);
    let token = FAKE_TOKEN.replace(' ', "");

    // As `main` does, in this order.
    mc_core::init_tracing();
    let _flush_logs = mc_core::FlushLogsOnDrop;
    crash::install_panic_hook();
    let game_dir_text = game_dir.to_string_lossy().into_owned();
    let equals_form = format!("--accessToken={STRAY_EQUALS_TOKEN}");
    let config = ClientConfig::parse_args(&[
        "--gameDir",
        game_dir_text.as_str(),
        "--username",
        "SecretsTest",
        "--uuid",
        "986dec87b7ec47ff89ff033fdb95c4b5",
        "--accessToken",
        token.as_str(),
        equals_form.as_str(),
        STRAY_BARE_TOKEN,
    ])
    .expect("launcher arguments");

    // What the client logs at start (the driver's own line), and the session's `Debug`.
    tracing::info!(login = ?config.login, "resolved login configuration");
    let session = config.login.launcher_session().expect("a launcher session");
    tracing::info!(?session, "session in hand");

    // Careless: the token formatted directly, and an error that quotes a response body.
    tracing::warn!("{PLAIN_MARKER} joining with {}", session.expose());
    tracing::warn!("{PLAIN_MARKER} control {UNREGISTERED_CONTROL}");
    let body = format!(
        r#"{{"accessToken":"{}","error":"echoed"}}"#,
        session.expose()
    );
    tracing::error!(error = %body, "{PLAIN_MARKER} the server echoed the request");

    // A worker thread panics with the token in its message...
    let leaked = session.expose().to_owned();
    let joined = std::thread::Builder::new()
        .name("w21-sa-worker".to_owned())
        .spawn(move || panic!("{PLAIN_MARKER} refresh failed for {leaked}"))
        .expect("spawn")
        .join();
    assert!(joined.is_err(), "the worker panicked");

    // ...and the game loop reports it, as `main` does.
    let mut report = crash::take_recorded_crash().expect("the hook recorded the panic");
    mc_core::flush_logs();
    assert_eq!(
        crash::display_crash_report(&mut report, game_dir),
        CrashExit::Saved
    );
    mc_core::flush_logs();
}
