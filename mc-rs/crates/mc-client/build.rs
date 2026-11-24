//! `mc-client.exe`'s Windows resources: the Aero Client logo as the file's icon
//! (Explorer, shortcuts and a pinned taskbar button show it), its version information and its
//! manifest. See `../mc-shell/build/resources.rs`, which the starter's build script shares; the
//! icon's pictures are this crate's `assets/icon/aero-<side>.png`.

#[path = "../mc-shell/build/resources.rs"]
mod resources;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../mc-shell/build/resources.rs");
    let manifest_dir = std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    );
    resources::embed(&resources::Exe {
        bin: Some("mc-client"),
        icon_pngs: resources::icon_pngs(&manifest_dir.join("assets").join("icon")),
        description: "Aero Client",
        internal_name: "mc-client",
        original_filename: "mc-client.exe",
        stem: "mc-client",
    });
}
