#![windows_subsystem = "windows"]
#![forbid(unsafe_code)]

//! `Aero Client.exe`, what the player double-clicks in the portable folder (the packaging copies
//! this binary there under that name): it starts `app\mc-client.exe` with the same arguments,
//! working folder its own, and exits at once. A windowless program, so no console or terminal
//! window ever opens; with no arguments the client makes its packaged start (it finds the
//! Minecraft files, keeps its logs in `game\logs` and reports problems itself).
//!
//! When the client can't be started it says why in a message ([`mc_shell::start`]).

use std::process::ExitCode;

use mc_shell::dialog::{self, Buttons, Icon};
use mc_shell::start::{self, TITLE};

fn main() -> ExitCode {
    let folder = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf));
    let Some(folder) = folder else {
        dialog::message(TITLE, start::MISSING, Icon::Warning, Buttons::Ok);
        return ExitCode::from(1);
    };
    match start::start_client(&folder, std::env::args_os().skip(1)) {
        Ok(client) => {
            start::allow_set_foreground(client.id());
            ExitCode::SUCCESS
        }
        Err(message) => {
            dialog::message(TITLE, &message, Icon::Warning, Buttons::Ok);
            ExitCode::from(1)
        }
    }
}
