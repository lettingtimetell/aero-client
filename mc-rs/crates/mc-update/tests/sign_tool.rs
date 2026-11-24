//! `aero-update-sign` end to end, with a throwaway key made in a temporary folder (never a real
//! signing key): keygen, sign, verify, and what it refuses.

use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Output};

use mc_update::install::InstallMarker;
use mc_update::verify;

fn tool(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aero-update-sign"))
        .args(args)
        .output()
        .expect("run aero-update-sign")
}

fn text(path: &Path) -> &str {
    path.to_str().expect("UTF-8 temp path")
}

/// A packaged zip of `version`, laid out as `package-aero-client.ps1` makes it.
fn package(path: &Path, version: &str) {
    let mut writer = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let marker = InstallMarker::new(version).to_json();
    for (name, bytes) in [
        (
            "Aero Client\\Aero Client.exe",
            b"MZ fake starter".as_slice(),
        ),
        ("Aero Client\\Play Aero Client.cmd", b"@echo off".as_slice()),
        ("Aero Client\\README.txt", b"readme".as_slice()),
        (
            "Aero Client\\app\\mc-client.exe",
            b"MZ fake client".as_slice(),
        ),
        ("Aero Client\\app\\aero-install.json", marker.as_bytes()),
    ] {
        writer.start_file(name, options).unwrap();
        writer.write_all(bytes).unwrap();
    }
    writer.finish().unwrap();
}

#[test]
fn keygen_sign_and_verify_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("throwaway.pk8");
    let made = tool(&["keygen", "--out", text(&key)]);
    assert!(made.status.success(), "{made:?}");
    let stdout = String::from_utf8_lossy(&made.stdout);
    let public = stdout
        .lines()
        .last()
        .expect("the public key line")
        .trim()
        .to_owned();
    let public_key = verify::parse_public_key(&public).expect("a 32-byte base64 key");
    // Never overwrites a key.
    assert!(!tool(&["keygen", "--out", text(&key)]).status.success());
    // `pubkey` prints the same key.
    let again = tool(&["pubkey", "--key", text(&key)]);
    assert_eq!(String::from_utf8_lossy(&again.stdout).trim(), public);

    let zip = dir.path().join("Aero Client.zip");
    package(&zip, "0.2.0");
    let out = dir.path().join("upload");
    let signed = tool(&[
        "sign",
        "--key",
        text(&key),
        "--zip",
        text(&zip),
        "--version",
        "0.2.0",
        "--out",
        text(&out),
    ]);
    assert!(signed.status.success(), "{signed:?}");
    let name = "Aero-Client-0.2.0-windows-x64.zip";
    let asset = out.join(name);
    let sha256 = verify::sha256_file(&asset).unwrap();
    assert_eq!(
        sha256,
        verify::sha256_file(&zip).unwrap(),
        "the zip, renamed"
    );
    let sidecar = std::fs::read(out.join(format!("{name}.sha256"))).unwrap();
    assert_eq!(
        verify::parse_sha256_sidecar(&sidecar, name).unwrap(),
        sha256
    );
    // The signature is what the client checks: over the asset name, version and hash.
    let signature =
        verify::parse_signature(&std::fs::read(out.join(format!("{name}.sig"))).unwrap()).unwrap();
    let message = verify::signed_message(name, "0.2.0", &sha256);
    verify::verify_signature(&public_key, message.as_bytes(), &signature).expect("verifies");
    let other = verify::signed_message(name, "0.3.0", &sha256);
    assert!(verify::verify_signature(&public_key, other.as_bytes(), &signature).is_err());

    let checked = tool(&[
        "verify",
        "--pubkey",
        &public,
        "--dir",
        text(&out),
        "--version",
        "0.2.0",
    ]);
    assert!(checked.status.success(), "{checked:?}");
    // A changed zip no longer verifies.
    let mut bytes = std::fs::read(&asset).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&asset, bytes).unwrap();
    let tampered = tool(&[
        "verify",
        "--pubkey",
        &public,
        "--dir",
        text(&out),
        "--version",
        "0.2.0",
    ]);
    assert!(!tampered.status.success());
}

#[test]
fn sign_refuses_a_zip_that_is_not_the_named_release() {
    let dir = tempfile::tempdir().unwrap();
    let zip = dir.path().join("Aero Client.zip");
    package(&zip, "0.2.0");
    let out = dir.path().join("upload");
    // The marker inside says 0.2.0.
    let wrong = tool(&[
        "sign",
        "--unsigned",
        "--zip",
        text(&zip),
        "--version",
        "0.3.0",
        "--out",
        text(&out),
    ]);
    assert!(!wrong.status.success());
    assert!(!out.exists(), "nothing written");
    // Prerelease versions are not releases.
    let pre = tool(&[
        "sign",
        "--unsigned",
        "--zip",
        text(&zip),
        "--version",
        "0.2.0-beta.1",
        "--out",
        text(&out),
    ]);
    assert!(!pre.status.success());
    // Unsigned: the zip and its .sha256, no .sig.
    let unsigned = tool(&[
        "sign",
        "--unsigned",
        "--zip",
        text(&zip),
        "--version",
        "0.2.0",
        "--out",
        text(&out),
    ]);
    assert!(unsigned.status.success(), "{unsigned:?}");
    assert!(out.join("Aero-Client-0.2.0-windows-x64.zip").is_file());
    assert!(out
        .join("Aero-Client-0.2.0-windows-x64.zip.sha256")
        .is_file());
    assert!(!out.join("Aero-Client-0.2.0-windows-x64.zip.sig").exists());
    // Both or neither of --key and --unsigned is a usage error.
    assert!(!tool(&[
        "sign",
        "--zip",
        text(&zip),
        "--version",
        "0.2.0",
        "--out",
        text(&out)
    ])
    .status
    .success());
}
