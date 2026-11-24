//! `aero-start` (the portable folder's `Aero Client.exe`) end to end, on Windows. Its own `main`
//! (no test harness): in a temporary portable folder, `app\mc-client.exe` is a copy of this same
//! binary, which, started by the starter with `AERO_START_TEST_ECHO` set, writes down the
//! arguments and the working folder it was given.
//!
//! - Every argument arrives whole and unchanged (spaces, quotes, backslashes, an empty one, one
//!   that tries to smuggle in arguments of its own), and none arrives when there were none: the
//!   starter's start without arguments is the client's packaged start.
//! - The client runs in the portable folder, and the starter exits without waiting for it.
//! - Without the client, the starter says why (its test mode prints the message) and starts
//!   nothing; started from a zip viewer's temporary copy, it says to extract the zip first.
//! - [`mc_shell::stdio::redirect`]: `println!`, `eprintln!` and a panic message reach the files.
//!   The starter is a windowless program here, and its printed message reaches a pipe, as a
//!   windowless client's output does.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

/// Set by the tests for the stand-in client: the file to write what it got into.
const ECHO_ENV: &str = "AERO_START_TEST_ECHO";
/// Set by the tests for the stand-in client: it keeps running until this file exists.
const HOLD_ENV: &str = "AERO_START_TEST_HOLD";
/// Set by the redirect test for this binary: the folder for the two output files.
const REDIRECT_ENV: &str = "AERO_START_TEST_REDIRECT";

fn main() -> ExitCode {
    if let Some(out) = std::env::var_os(ECHO_ENV) {
        return echo(Path::new(&out));
    }
    if let Some(dir) = std::env::var_os(REDIRECT_ENV) {
        redirected(Path::new(&dir));
    }
    if !cfg!(windows) {
        println!("aero_start: Windows only; nothing to test here");
        return ExitCode::SUCCESS;
    }
    let tests: [(&str, fn()); 5] = [
        (
            "every_argument_arrives_whole_and_the_starter_does_not_wait",
            every_argument_arrives_whole_and_the_starter_does_not_wait,
        ),
        (
            "no_arguments_stay_no_arguments",
            no_arguments_stay_no_arguments,
        ),
        (
            "a_missing_client_is_explained_and_nothing_starts",
            a_missing_client_is_explained_and_nothing_starts,
        ),
        (
            "a_starter_run_from_inside_the_zip_says_to_extract_it",
            a_starter_run_from_inside_the_zip_says_to_extract_it,
        ),
        (
            "redirected_output_and_panics_reach_the_files",
            redirected_output_and_panics_reach_the_files,
        ),
    ];
    let mut failed = 0;
    for (name, test) in tests {
        match std::panic::catch_unwind(test) {
            Ok(()) => println!("test {name} ... ok"),
            Err(_) => {
                println!("test {name} ... FAILED");
                failed += 1;
            }
        }
    }
    println!(
        "\ntest result: {}. {} passed; {failed} failed",
        if failed == 0 { "ok" } else { "FAILED" },
        tests.len() - failed
    );
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

// ---- the stand-ins ----------------------------------------------------------------------------

/// `text`'s UTF-8 bytes in hex: one line per value, whatever the value holds.
fn hex(text: &str) -> String {
    text.bytes().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> String {
    let bytes: Vec<u8> = (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).expect("hex"))
        .collect();
    String::from_utf8(bytes).expect("UTF-8")
}

/// The stand-in client: write `ARG <hex>` per argument and `CWD <hex>`, all at once (a rename),
/// then keep running until [`HOLD_ENV`]'s file exists.
fn echo(out: &Path) -> ExitCode {
    let mut text = String::new();
    for arg in std::env::args_os().skip(1) {
        text.push_str(&format!("ARG {}\n", hex(&arg.to_string_lossy())));
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    text.push_str(&format!("CWD {}\n", hex(&cwd.to_string_lossy())));
    let partial = out.with_extension("partial");
    if std::fs::write(&partial, text).is_err() || std::fs::rename(&partial, out).is_err() {
        return ExitCode::FAILURE;
    }
    if let Some(hold) = std::env::var_os(HOLD_ENV) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !Path::new(&hold).exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    ExitCode::SUCCESS
}

/// Redirect to `<dir>\out.txt` and `<dir>\err.txt`, print to both, then panic.
fn redirected(dir: &Path) -> ! {
    let out = std::fs::File::create(dir.join("out.txt")).expect("out.txt");
    let err = std::fs::File::create(dir.join("err.txt")).expect("err.txt");
    mc_shell::stdio::redirect(out, err).expect("redirected");
    println!("a line for standard output");
    eprintln!("a line for standard error");
    panic!("a deliberate panic after the redirect");
}

// ---- helpers ----------------------------------------------------------------------------------

/// A portable folder in `dir`: `Aero Client\Aero Client.exe` (the starter) and, with
/// `with_client`, `app\mc-client.exe` (a copy of this binary).
fn package(dir: &Path, with_client: bool) -> PathBuf {
    let root = dir.join("pkg").join("Aero Client");
    std::fs::create_dir_all(root.join("app")).unwrap();
    std::fs::copy(
        env!("CARGO_BIN_EXE_aero-start"),
        root.join("Aero Client.exe"),
    )
    .unwrap();
    if with_client {
        let me = std::env::current_exe().unwrap();
        std::fs::copy(me, root.join("app").join("mc-client.exe")).unwrap();
    }
    root
}

/// The starter at `exe`, with its temporary folder moved to `<dir>\elsewhere` (so the test's
/// own temporary folder doesn't read as a zip viewer's) and test mode on.
fn starter(exe: &Path, dir: &Path) -> Command {
    let elsewhere = dir.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    let mut command = Command::new(exe);
    command
        .env("TMP", &elsewhere)
        .env("TEMP", &elsewhere)
        .env(mc_shell::dialog::NO_GUI_ENV, "1")
        .stdin(Stdio::null());
    command
}

/// Wait up to 60 s for `done`.
fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting: {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Whether a program runs from `exe`: Windows refuses to open a running program for writing.
fn running(exe: &Path) -> bool {
    match std::fs::OpenOptions::new().write(true).open(exe) {
        Ok(_) => false,
        Err(error) => error.raw_os_error() == Some(32),
    }
}

/// What the stand-in client wrote: its arguments and its working folder.
fn read_echo(file: &Path) -> (Vec<String>, PathBuf) {
    let text = std::fs::read_to_string(file).unwrap();
    let mut args = Vec::new();
    let mut cwd = PathBuf::new();
    for line in text.lines() {
        match line.split_once(' ') {
            Some(("ARG", value)) => args.push(unhex(value)),
            Some(("CWD", value)) => cwd = PathBuf::from(unhex(value)),
            _ => panic!("unexpected line {line:?}"),
        }
    }
    (args, cwd)
}

fn same_folder(a: &Path, b: &Path) -> bool {
    let full = |path: &Path| {
        std::fs::canonicalize(path)
            .unwrap()
            .to_string_lossy()
            .to_lowercase()
    };
    full(a) == full(b)
}

/// Run the starter in a fresh portable folder with `args`; returns what the client got, the
/// portable folder, and the temporary folder that holds it (deleted when dropped).
fn start_and_echo(args: &[&str]) -> (Vec<String>, PathBuf, PathBuf, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let root = package(dir.path(), true);
    let echo_file = dir.path().join("echo.txt");
    let hold = dir.path().join("hold");
    let status = starter(&root.join("Aero Client.exe"), dir.path())
        .args(args)
        .env(ECHO_ENV, &echo_file)
        .env(HOLD_ENV, &hold)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(status.success(), "{status:?}");
    let client = root.join("app").join("mc-client.exe");
    wait_for("the client's record", || echo_file.is_file());
    assert!(
        running(&client),
        "the starter has exited and the client still runs"
    );
    std::fs::write(&hold, "").unwrap();
    wait_for("the client to exit", || !running(&client));
    let (got, cwd) = read_echo(&echo_file);
    (got, cwd, root, dir)
}

// ---- the tests --------------------------------------------------------------------------------

fn every_argument_arrives_whole_and_the_starter_does_not_wait() {
    let smuggler = r#"x\" --gameDir \\example-host\share --userType \""#;
    let values = [
        "plain",
        "two words",
        r"C:\Folder With Space\",
        r"ends\\",
        r#"a "quoted" word"#,
        r#"back\"slash"#,
        "",
        "--example",
        smuggler,
    ];
    let (got, cwd, root, _dir) = start_and_echo(&values);
    assert_eq!(got, values, "one for one, unchanged");
    assert!(same_folder(&cwd, &root), "{cwd:?} is the portable folder");
}

fn no_arguments_stay_no_arguments() {
    let (got, cwd, root, _dir) = start_and_echo(&[]);
    assert!(got.is_empty(), "{got:?}");
    assert!(same_folder(&cwd, &root));
}

fn a_missing_client_is_explained_and_nothing_starts() {
    let dir = tempfile::tempdir().unwrap();
    let root = package(dir.path(), false);
    let output = starter(&root.join("Aero Client.exe"), dir.path())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let printed = String::from_utf8_lossy(&output.stdout);
    assert!(
        printed.contains(
            r"[Warning] Aero Client can't start: the game file app\mc-client.exe is missing."
        ),
        "{printed}"
    );
    assert!(printed.contains("Extract the zip again"), "{printed}");
}

fn a_starter_run_from_inside_the_zip_says_to_extract_it() {
    // What Windows' zip viewer does on "Run": only the clicked file, in a folder named after
    // the zip.
    let dir = tempfile::tempdir().unwrap();
    let copy = dir
        .path()
        .join("Temp1_Aero-Client-1.0.0-windows-x64.zip")
        .join("Aero Client");
    std::fs::create_dir_all(&copy).unwrap();
    let exe = copy.join("Aero Client.exe");
    std::fs::copy(env!("CARGO_BIN_EXE_aero-start"), &exe).unwrap();
    let output = starter(&exe, dir.path()).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let printed = String::from_utf8_lossy(&output.stdout);
    assert!(
        printed.contains("[Warning] Aero Client can't start from inside the zip file."),
        "{printed}"
    );
    assert!(printed.contains("Extract All..."), "{printed}");
}

fn redirected_output_and_panics_reach_the_files() {
    let dir = tempfile::tempdir().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .env(REDIRECT_ENV, dir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(101), "the panic's exit code");
    let out = std::fs::read_to_string(dir.path().join("out.txt")).unwrap();
    let err = std::fs::read_to_string(dir.path().join("err.txt")).unwrap();
    assert!(out.contains("a line for standard output"), "{out}");
    assert!(!out.contains("standard error"), "{out}");
    assert!(err.contains("a line for standard error"), "{err}");
    assert!(
        err.contains("a deliberate panic after the redirect"),
        "the panic message: {err}"
    );
}
