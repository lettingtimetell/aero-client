//! The client half of `connected_textures`.
//!
//! Connected textures are the resource pack's `mcpatcher/ctm` rules (the CTM format). Their
//! tiles live in the block atlas and the mesher applies them, so the switch has two halves:
//!
//! * **Off at startup** (the default): `GameAssets::connected_textures` is false when the first
//!   bake runs, so the bake, the atlas, the bake cache and every mesh are exactly vanilla, and
//!   the mesher never reads a CTM table. Cost: one `u64` compare per frame, here.
//! * **Turned on:** if the installed bake has no CTM table, a full resource reload is queued,
//!   which re-stitches the atlas with the tiles; every resident section is re-meshed. A one-off
//!   hitch of about a second.
//! * **Turned off, or Fancy/Fast changed:** only the meshes are rebuilt. The tiles stay in the
//!   atlas, never sampled, until the next reload or restart.
//!
//! Spec: The mesher half is
//! `mc_render::terrain::ctm`, the rules and table `mc_assets::ctm`.

use mc_render::terrain::{CtmMode, MeshSettings};

use super::super::Client;
use crate::app::assets::GameAssets;

/// The mod's id in the catalog and the settings file.
const MOD_ID: &str = "connected_textures";

/// What the client last applied.
#[derive(Debug, Default)]
pub(crate) struct M8State {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    /// The mode folded into the mesher's settings by [`Client::ctm_mesh_settings`].
    mode: CtmMode,
}

/// The first bake must already know whether to stitch the tiles.
pub(crate) fn seed_assets(mut assets: GameAssets, settings: &mc_mods::ModSettings) -> GameAssets {
    assets.connected_textures = settings.enabled(MOD_ID);
    assets
}

/// The mode the settings ask for. The `mode` choice is `["Fancy", "Fast"]`.
fn wanted_mode(settings: &mc_mods::ModSettings) -> CtmMode {
    if !settings.enabled(MOD_ID) {
        CtmMode::Off
    } else if settings.choice(MOD_ID, "mode") == 1 {
        CtmMode::Fast
    } else {
        CtmMode::Fancy
    }
}

impl Client {
    /// Apply a changed switch or mode. One `u64` compare per frame while no
    /// mod setting changes.
    pub(crate) fn drive_connected_textures(&mut self) {
        let generation = self.mods.settings.generation();
        if self.m8.seen == generation {
            return;
        }
        self.m8.seen = generation;
        let mode = wanted_mode(&self.mods.settings);
        let enabled = mode != CtmMode::Off;
        if mode == self.m8.mode && self.assets.connected_textures == enabled {
            return;
        }
        self.m8.mode = mode;
        // Every later bake (a pack change, the mipmap slider, a restart) follows the switch.
        self.assets.connected_textures = enabled;
        if enabled && mc_assets::model::ctm_table().is_none() {
            // Off -> Fast/Fancy needs a resource reload to stitch the tiles. The deferred reload
            // runs at the start of the next frame, where the GPU and the phases are in scope.
            tracing::info!("connected textures on: reloading resources to stitch the tiles");
            self.pending_pack_reload = true;
        }
        self.sync_mesh_settings();
    }

    /// Fold the mode into the mesher's settings, so a change
    /// re-meshes every resident section through the existing path.
    pub(crate) fn ctm_mesh_settings(&self, wanted: MeshSettings) -> MeshSettings {
        MeshSettings {
            connected_textures: self.m8.mode,
            ..wanted
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};

    /// A headless client over the real assets, its game dir a temp dir (so the mod settings
    /// file can never land in the user's `mc-rs`). `None` without the vanilla jar.
    fn headless_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    #[test]
    fn on_queues_a_reload_and_off_or_a_mode_change_only_remeshes() {
        let Some((mut client, _dir)) = headless_client() else {
            return;
        };
        // Off (the default): nothing moves, and the bake stays vanilla.
        client.drive_connected_textures();
        assert!(!client.pending_pack_reload);
        assert!(!client.assets.connected_textures);
        assert_eq!(
            client.mesh_queue.settings().connected_textures,
            CtmMode::Off
        );

        // On: the installed bake has no table yet, so the tiles must be stitched (a reload),
        // and the mesher switches to Fancy.
        client.mods.settings.set_enabled(MOD_ID, true);
        client.drive_connected_textures();
        assert!(
            client.assets.connected_textures,
            "later bakes stitch the tiles"
        );
        assert_eq!(
            client.pending_pack_reload,
            mc_assets::model::ctm_table().is_none()
        );
        assert_eq!(
            client.mesh_queue.settings().connected_textures,
            CtmMode::Fancy
        );

        // Fast: the mesher follows.
        client.mods.settings.set_choice(MOD_ID, "mode", 1);
        client.drive_connected_textures();
        assert_eq!(
            client.mesh_queue.settings().connected_textures,
            CtmMode::Fast
        );

        // Off: re-mesh only, never a reload; later bakes are vanilla again.
        client.pending_pack_reload = false;
        client.mods.settings.set_enabled(MOD_ID, false);
        client.drive_connected_textures();
        assert!(!client.pending_pack_reload, "turning it off never reloads");
        assert!(!client.assets.connected_textures);
        assert_eq!(
            client.mesh_queue.settings().connected_textures,
            CtmMode::Off
        );

        // A frame with no settings change is one compare.
        let before = client.mesh_queue.settings();
        client.drive_connected_textures();
        assert_eq!(client.mesh_queue.settings(), before);
    }

    #[test]
    fn the_mode_follows_the_switch_and_the_choice() {
        let mut settings = mc_mods::ModSettings::default();
        assert_eq!(wanted_mode(&settings), CtmMode::Off, "off by default");
        settings.set_enabled(MOD_ID, true);
        assert_eq!(wanted_mode(&settings), CtmMode::Fancy, "Fancy is choice 0");
        settings.set_choice(MOD_ID, "mode", 1);
        assert_eq!(wanted_mode(&settings), CtmMode::Fast);
        settings.set_enabled(MOD_ID, false);
        assert_eq!(wanted_mode(&settings), CtmMode::Off);
    }
}
