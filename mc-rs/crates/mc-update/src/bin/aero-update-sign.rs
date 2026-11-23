#![forbid(unsafe_code)]

//! `aero-update-sign`: makes the updater's signing key, and turns a packaged zip into the three
//! files a GitHub release needs.
//!
//! # Publishing an update, step by step (run from `mc-rs`)
//!
//! 1. **Once, make the key pair** (on a PC you trust, never in the repository):
//!
//!    ```text
//!    cargo run -p mc-update --bin aero-update-sign -- keygen --out <folder outside the repo>\aero-update-signing.pk8
//!    ```
//!
//!    It writes the private key (PKCS#8) and prints the public key in base64. Keep the private
//!    key offline and backed up. If it leaks, anyone can sign an update every client accepts;
//!    if it is lost, clients built with its public key can't update again until players
//!    download a new zip by hand.
//! 2. **Build with the public key**: set `AERO_UPDATE_PUBKEY` to the printed base64 before
//!    packaging (`$env:AERO_UPDATE_PUBKEY = '...'` in the same PowerShell, then
//!    `package-aero-client.ps1`). From then on that client installs only releases signed with
//!    the private key. A build without the variable never installs an update: it only announces
//!    a new version and links its release page.
//! 3. **Package** with `package-aero-client.ps1` as usual (it writes `release\Aero Client.zip`
//!    with the version marker `app\aero-install.json` in it).
//! 4. **Sign**:
//!
//!    ```text
//!    cargo run -p mc-update --bin aero-update-sign -- sign --key <...>.pk8 --zip "..\release\Aero Client.zip" --version 0.2.0 --out <folder>
//!    ```
//!
//!    It first unpacks the zip into a temporary folder with the updater's own checks (the layout,
//!    and a marker naming the same version), then writes `Aero-Client-0.2.0-windows-x64.zip`,
//!    `Aero-Client-0.2.0-windows-x64.zip.sha256` and `Aero-Client-0.2.0-windows-x64.zip.sig` into
//!    `<folder>`. `--unsigned` instead of `--key` writes the zip and `.sha256` only: every client
//!    then announces the release, and none installs it.
//! 5. **Publish** a GitHub release tagged `v0.2.0` in the repository the client reads
//!    (`mc_update::REPO`), not a draft and not a prerelease, with those files attached. The
//!    version must be higher than the players' (semver), and the workspace version in
//!    `Cargo.toml` must be the release's version when the zip is built.
//!
//! `verify --pubkey <base64> --dir <folder> --version 0.2.0` checks the three files the way the
//! client will.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use base64::Engine as _;
use mc_update::github::{asset_name, sha256_asset_name, signature_asset_name};
use mc_update::stage::{stage_release, Limits};
use mc_update::verify::{self, signed_message};
use mc_update::version::parse_release_tag;
use ring::signature::{Ed25519KeyPair, KeyPair};

const USAGE: &str = "usage:
  aero-update-sign keygen --out <private key file>
  aero-update-sign pubkey --key <private key file>
  aero-update-sign sign (--key <private key file> | --unsigned) --zip <packaged zip> --version <x.y.z> --out <folder>
  aero-update-sign verify --pubkey <base64> --dir <folder> --version <x.y.z>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("aero-update-sign: {message}");
            ExitCode::FAILURE
        }
    }
}

/// `--name value` pairs and lone `--flags` after the command.
struct Options(Vec<(String, Option<String>)>);

impl Options {
    fn parse(args: &[String]) -> Result<Options, String> {
        let mut options = Vec::new();
        let mut index = 0;
        while index < args.len() {
            let name = &args[index];
            if !name.starts_with("--") {
                return Err(format!("unexpected argument {name}\n{USAGE}"));
            }
            if name == "--unsigned" {
                options.push((name.clone(), None));
                index += 1;
            } else {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))?;
                options.push((name.clone(), Some(value.clone())));
                index += 2;
            }
        }
        Ok(Options(options))
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, value)| value.as_deref())
    }

    fn need(&self, name: &str) -> Result<&str, String> {
        self.get(name)
            .ok_or_else(|| format!("missing {name}\n{USAGE}"))
    }

    fn flag(&self, name: &str) -> bool {
        self.0.iter().any(|(n, _)| n == name)
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let (command, rest) = args.split_first().ok_or(USAGE)?;
    let options = Options::parse(rest)?;
    match command.as_str() {
        "keygen" => keygen(Path::new(options.need("--out")?)),
        "pubkey" => {
            let key = load_key(Path::new(options.need("--key")?))?;
            println!("{}", public_key_text(&key));
            Ok(())
        }
        "sign" => sign(&options),
        "verify" => verify_dir(&options),
        _ => Err(USAGE.to_owned()),
    }
}

fn public_key_text(key: &Ed25519KeyPair) -> String {
    base64::engine::general_purpose::STANDARD.encode(key.public_key().as_ref())
}

fn keygen(out: &Path) -> Result<(), String> {
    if out.exists() {
        return Err("the key file already exists; refusing to overwrite a key".to_owned());
    }
    let rng = ring::rand::SystemRandom::new();
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng)
        .map_err(|_| "the system's random generator failed".to_owned())?;
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
        .map_err(|_| "the new key does not load".to_owned())?;
    {
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out)
            .map_err(|error| format!("writing the key: {error}"))?;
        file.write_all(pkcs8.as_ref())
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("writing the key: {error}"))?;
    }
    println!("private key written; keep it offline and backed up");
    println!("public key (AERO_UPDATE_PUBKEY):");
    println!("{}", public_key_text(&key));
    Ok(())
}

fn load_key(path: &Path) -> Result<Ed25519KeyPair, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("reading the key: {error}"))?;
    Ed25519KeyPair::from_pkcs8(&bytes).map_err(|_| "not an Ed25519 PKCS#8 key".to_owned())
}

fn release_version(options: &Options) -> Result<String, String> {
    let text = options.need("--version")?;
    let version = parse_release_tag(text)
        .ok_or_else(|| format!("{text} is not a release version (x.y.z, no prerelease)"))?;
    Ok(version.to_string())
}

fn sign(options: &Options) -> Result<(), String> {
    let version = release_version(options)?;
    let zip = PathBuf::from(options.need("--zip")?);
    let out = PathBuf::from(options.need("--out")?);
    let key = match (options.get("--key"), options.flag("--unsigned")) {
        (Some(path), false) => Some(load_key(Path::new(path))?),
        (None, true) => None,
        _ => return Err(format!("give exactly one of --key and --unsigned\n{USAGE}")),
    };

    // The zip must unpack the way the client will unpack it.
    let scratch = std::env::temp_dir().join(format!(
        "aero-update-sign-{}-{}",
        std::process::id(),
        mc_update::now_unix()
    ));
    let checked = stage_release(&zip, &scratch.join("x"), &version, &Limits::default());
    let _ = std::fs::remove_dir_all(&scratch);
    let staged = checked.map_err(|error| format!("the zip is not release {version}: {error}"))?;
    println!(
        "checked: {} files, {} bytes, unpacked as release {version}",
        staged.files, staged.bytes
    );

    std::fs::create_dir_all(&out)
        .map_err(|error| format!("creating {}: {error}", out.display()))?;
    let semver = parse_release_tag(&version).ok_or("bad version")?;
    let name = asset_name(&semver);
    let target = out.join(&name);
    std::fs::copy(&zip, &target).map_err(|error| format!("copying the zip: {error}"))?;
    let sha256 =
        verify::sha256_file(&target).map_err(|error| format!("hashing the zip: {error}"))?;
    std::fs::write(
        out.join(sha256_asset_name(&name)),
        format!("{sha256}  {name}\n"),
    )
    .map_err(|error| format!("writing the .sha256: {error}"))?;
    println!("{name}\n  sha256 {sha256}");
    if let Some(key) = key {
        let message = signed_message(&name, &version, &sha256);
        let signature = key.sign(message.as_bytes());
        let text = base64::engine::general_purpose::STANDARD.encode(signature.as_ref());
        std::fs::write(out.join(signature_asset_name(&name)), format!("{text}\n"))
            .map_err(|error| format!("writing the .sig: {error}"))?;
        println!("  signed with public key {}", public_key_text(&key));
    } else {
        println!("  not signed (--unsigned): clients announce this release but install none");
    }
    println!(
        "attach the files to a GitHub release tagged v{version} (not a draft, not a prerelease)"
    );
    Ok(())
}

fn verify_dir(options: &Options) -> Result<(), String> {
    let version = release_version(options)?;
    let key = verify::parse_public_key(options.need("--pubkey")?).map_err(|e| e.to_string())?;
    let dir = PathBuf::from(options.need("--dir")?);
    let semver = parse_release_tag(&version).ok_or("bad version")?;
    let name = asset_name(&semver);
    let sha256 = verify::sha256_file(&dir.join(&name))
        .map_err(|error| format!("reading {name}: {error}"))?;
    let sidecar = std::fs::read(dir.join(sha256_asset_name(&name)))
        .map_err(|error| format!("reading the .sha256: {error}"))?;
    let published = verify::parse_sha256_sidecar(&sidecar, &name).map_err(|e| e.to_string())?;
    if published != sha256 {
        return Err("the .sha256 does not match the zip".to_owned());
    }
    let signature = std::fs::read(dir.join(signature_asset_name(&name)))
        .map_err(|error| format!("reading the .sig: {error}"))?;
    let signature = verify::parse_signature(&signature).map_err(|e| e.to_string())?;
    let message = signed_message(&name, &version, &sha256);
    verify::verify_signature(&key, message.as_bytes(), &signature).map_err(|e| e.to_string())?;
    println!("ok: {name} matches its .sha256 and its signature verifies");
    Ok(())
}
