//! `Aero Client.exe`'s Windows resources (the `aero-start` binary): the logo as its icon, its
//! version information and its manifest. See `build/resources.rs`, which `mc-client`'s build
//! script shares.

#[path = "build/resources.rs"]
mod resources;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build/resources.rs");
    let manifest_dir = std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    );
    let icons = manifest_dir
        .join("..")
        .join("mc-client")
        .join("assets")
        .join("icon");
    resources::embed(&resources::Exe {
        bin: Some("aero-start"),
        icon_pngs: resources::icon_pngs(&icons),
        description: "Aero Client",
        internal_name: "aero-start",
        original_filename: "Aero Client.exe",
        stem: "aero-start",
    });
}
