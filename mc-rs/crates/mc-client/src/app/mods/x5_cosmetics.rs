//! Capes and 3D skins.
//!
//! * **Item 30, cape physics accuracy.** `EntityPlayer.onUpdate` advances every player's
//!   `chasingPos` once per tick after its movement, the local player in first person included.
//!   mc-rs advanced the chase only when a render path asked for the cape, so the history froze in
//!   first person (and for remote players outside the frustum) and the first F5 frame stepped
//!   from a stale point: the cape flicked up. [`Client::x5_tick`] now runs the chase for the local
//!   player and every tracked player at the end of the client tick; the render paths only sample
//!   (`AppearanceState::cape_state`).
//! * **Item 27, Cape Physics mod** (`cape_physics`): the "Motion" choice becomes
//!   [`CapeMotion`], applied to every sampled cape state. Off = vanilla (`CapeMotion::Vanilla`
//!   hands the state back untouched).
//! * **Item 29, 3D Skins mod** (`skins_3d`): players within the distance option draw their
//!   outer skin layer as voxels ([`emit_player_body_3d`], `mc_render::entity::skin3d`), in the
//!   same vertex batch as the body; the local first-person sleeve too
//!   ([`Client::x5_first_person_sleeve`]). Meshes are built once per skin texture and arm width.
//!
//! Off cost: one `u64` generation compare per tick, one cached `bool` per player body and one per
//! frame for the first-person sleeve. The skin meshes are dropped while 3D Skins is off.

use std::collections::HashMap;
use std::sync::Arc;

use mc_render::entity::skin3d::{self, Skin3dCache, SkinLayerMesh, WearMask, WearPart};

use super::super::Client;
use crate::app::appearance::CapeMotion;

/// The `cape_physics` mod id (the settings-file key).
pub(crate) const CAPE_PHYSICS: &str = "cape_physics";
/// The `skins_3d` mod id.
pub(crate) const SKINS_3D: &str = "skins_3d";

/// The lane's state: one field on `Client`.
#[derive(Debug, Default)]
pub(crate) struct X5State {
    /// The settings generation last synced; 0 = never (the store starts at 1).
    seen: u64,
    /// The cached Cape Physics style and controls.
    pub(crate) cape_motion: CapeMotion,
    /// 3D Skins is on.
    pub(crate) skins_on: bool,
    /// The wear parts that get voxels (the part toggles).
    pub(crate) skin_parts: WearMask,
    /// The distance option, squared (blocks²): farther players keep the flat layer.
    pub(crate) skin_distance_sq: f32,
    /// The first-person sleeve option.
    pub(crate) skin_first_person: bool,
    /// Built meshes by skin texture key and arm width.
    pub(crate) skin_meshes: Skin3dCache,
    /// The resource pack's default (Steve/Alex) skin pixels by texture path; `None` = not a
    /// loadable 64x64 image. Cleared on a resource reload.
    default_skins: HashMap<String, Option<Vec<u8>>>,
}

impl X5State {
    /// Refresh the typed snapshot when the settings generation moved. Returns whether it did.
    pub(crate) fn sync(&mut self, settings: &mc_mods::ModSettings) -> bool {
        if self.seen == settings.generation() {
            return false;
        }
        self.seen = settings.generation();
        self.cape_motion = cape_motion_from(settings);
        self.skins_on = settings.enabled(SKINS_3D);
        if self.skins_on {
            let mut parts = 0;
            for (option, bits) in [
                ("hat", WearPart::Hat.bit()),
                ("jacket", WearPart::Jacket.bit()),
                (
                    "sleeves",
                    WearPart::LeftSleeve.bit() | WearPart::RightSleeve.bit(),
                ),
                (
                    "pants",
                    WearPart::LeftPants.bit() | WearPart::RightPants.bit(),
                ),
            ] {
                if settings.bool(SKINS_3D, option) {
                    parts |= bits;
                }
            }
            self.skin_parts = parts;
            let distance = settings.f32(SKINS_3D, "distance");
            self.skin_distance_sq = distance * distance;
            self.skin_first_person = settings.bool(SKINS_3D, "first_person");
        } else {
            // Give the memory back; re-enabling rebuilds lazily (~0.1-1 ms per skin).
            self.skin_meshes.clear();
            self.default_skins.clear();
        }
        true
    }

    /// A resource reload can replace the default skins' pixels under the same texture path.
    pub(crate) fn resources_reloaded(&mut self) {
        self.skin_meshes.clear();
        self.default_skins.clear();
    }
}

/// Whether the head slot keeps the flat hat: any head item that is not armour (`LayerCustomHead`:
/// a worn skull is 9.5 texels wide and would meet the 0.75-deep voxel hat exactly; pumpkins and
/// other blocks too, for simplicity). A helmet (1.0 out) sits outside the voxels.
pub(crate) fn hat_blocked(head: Option<&mc_protocol::ItemStack>) -> bool {
    head.is_some_and(|stack| {
        u16::try_from(stack.item_id).map_or(true, |id| {
            mc_data::item_props(mc_data::ItemId(id))
                .armor_material
                .is_none()
        })
    })
}

/// A resource-pack texture's RGBA when it is a 64x64 image (a default player skin).
fn load_skin_rgba(resources: &mc_assets::ResourceManager, path: &str) -> Option<Vec<u8>> {
    let bytes = resources
        .load(&mc_assets::ResourceLocation::minecraft(path))
        .ok()?;
    let image = image::load_from_memory(&bytes).ok()?.to_rgba8();
    (image.width() == 64 && image.height() == 64).then(|| image.into_raw())
}

/// The pixels of the texture a player body draws with: a profile skin from the appearance
/// cache, or the descriptor's default skin from the resource pack (loaded once).
fn skin_pixels<'a>(
    skins: &'a mc_render::entity::SkinCache,
    defaults: &'a mut HashMap<String, Option<Vec<u8>>>,
    resources: &mc_assets::ResourceManager,
    texture: &str,
) -> Option<(&'a [u8], u32, u32)> {
    if let Some(skin) = skins.ready(texture) {
        return Some((&skin.rgba[..], 64, 64));
    }
    if !texture.starts_with("textures/") {
        // A profile skin that is not resident (yet): retry on a later frame.
        return None;
    }
    defaults
        .entry(texture.to_owned())
        .or_insert_with(|| load_skin_rgba(resources, texture))
        .as_deref()
        .map(|rgba| (rgba, 64, 64))
}

/// Draws one player body with 3D skins, when the mod applies: on, the
/// player within the distance option, a player model, a mesh for its texture. Returns whether it
/// emitted the body; `false` leaves the caller's ordinary `emit_with` to draw the flat body.
///
/// The pose, vertex order of the base parts, texture and draw are exactly the flat body's: the
/// voxels are appended to the same batch (`skin3d::emit_player_body`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_player_body_3d(
    x5: &mut X5State,
    models: &mut mc_render::entity::RenderManager,
    skins: &mc_render::entity::SkinCache,
    resources: &mc_assets::ResourceManager,
    kind: mc_render::entity::ModelKind,
    texture: &str,
    head: Option<&mc_protocol::ItemStack>,
    base: mc_render::entity::Mat4,
    params: &mc_render::entity::AnimParams,
    distance_sq: f32,
    out: &mut Vec<mc_render::entity::EntityVertex>,
) -> bool {
    if !x5.skins_on || distance_sq > x5.skin_distance_sq {
        return false;
    }
    let slim = match kind {
        mc_render::entity::ModelKind::Player => false,
        mc_render::entity::ModelKind::PlayerSlim => true,
        _ => return false,
    };
    let mut parts = x5.skin_parts;
    if hat_blocked(head) {
        parts &= !WearPart::Hat.bit();
    }
    if parts == 0 {
        return false;
    }
    let X5State {
        skin_meshes,
        default_skins,
        ..
    } = x5;
    let Some(mesh) = skin_meshes.get_or_build(texture, slim, || {
        skin_pixels(skins, default_skins, resources, texture)
    }) else {
        return false;
    };
    let Some(model) = models.player_model_mut(kind) else {
        return false;
    };
    skin3d::emit_player_body(model, params, base, &mesh, parts, out);
    true
}

/// The Cape Physics mod's cloth settings: [`CapeMotion::Vanilla`] while the mod
/// is off, otherwise the Style choice and normalized wave sliders.
pub(crate) fn cape_motion_from(settings: &mc_mods::ModSettings) -> CapeMotion {
    if !settings.enabled(CAPE_PHYSICS) {
        return CapeMotion::Vanilla;
    }
    use mc_render::entity::layers::{CapeCloth, CapeClothStyle};
    CapeMotion::Cloth(CapeCloth {
        style: match settings.choice(CAPE_PHYSICS, "motion") {
            1 => CapeClothStyle::Flowing,
            2 => CapeClothStyle::Windy,
            3 => CapeClothStyle::Draped,
            _ => CapeClothStyle::Wavy,
        },
        time_seconds: 0.0,
        phase: 0.0,
        strength: settings.f32(CAPE_PHYSICS, "strength") / 100.0,
        speed: settings.f32(CAPE_PHYSICS, "speed") / 100.0,
        stiffness: settings.f32(CAPE_PHYSICS, "stiffness") / 100.0,
    })
}

impl Client {
    /// The lane's per-tick hook (`<<FW-X5:cape-tick>>` in `Client::tick`, after the session tick
    /// moved the local player and every tracked entity).
    ///
    /// java: the chase block at the end of `EntityPlayer.onUpdate`, which `World.updateEntities`
    /// runs for every player in the client world each tick: the local `EntityPlayerSP` (while it
    /// is simulated) and every `EntityOtherPlayerMP`, drawn or not.
    pub(crate) fn x5_tick(&mut self) {
        if self.x5.sync(&self.mods.settings) {
            self.appearance.cape_motion = self.x5.cape_motion;
        }
        let tick = self.tick;
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let state = session.state();
        if session.player_is_simulated() {
            if let Some(predictor) = session.predictor() {
                self.appearance
                    .tick_cape_chase(state.local.uuid, tick, predictor.state.pos);
            }
        }
        let world = &state.world;
        for (network_id, tracked) in state.entities.iter() {
            if tracked.kind != crate::sync::entities::TrackedKind::Player
                || network_id == state.local.entity_id
            {
                continue;
            }
            let Some(uuid) = tracked.uuid else {
                continue;
            };
            let Some(position) = world
                .ecs
                .get::<&mc_world::entity::Position>(tracked.entity)
                .ok()
                .map(|position| position.0)
            else {
                continue;
            };
            self.appearance.tick_cape_chase(uuid, tick, position);
        }
    }

    /// The voxel mesh for the local first-person sleeve, or `None` for
    /// the vanilla flat sleeve (3D Skins off, its first-person or sleeve option off, or no mesh).
    /// `custom` is the profile skin the hand draws with (key and pixels); without one the hand
    /// draws `default_texture` from the resource pack.
    pub(crate) fn x5_first_person_sleeve(
        &mut self,
        slim: bool,
        custom: Option<&(String, crate::app::appearance::SkinPixels)>,
        default_texture: &str,
    ) -> Option<Arc<SkinLayerMesh>> {
        let x5 = &mut self.x5;
        if !(x5.skins_on
            && x5.skin_first_person
            && x5.skin_parts & WearPart::RightSleeve.bit() != 0)
        {
            return None;
        }
        let X5State {
            skin_meshes,
            default_skins,
            ..
        } = x5;
        match custom {
            Some((key, rgba)) => skin_meshes.get_or_build(key, slim, || Some((&rgba[..], 64, 64))),
            None => {
                let resources = &self.assets.resources;
                skin_meshes.get_or_build(default_texture, slim, || {
                    default_skins
                        .entry(default_texture.to_owned())
                        .or_insert_with(|| load_skin_rgba(resources, default_texture))
                        .as_deref()
                        .map(|rgba| (rgba, 64, 64))
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> mc_mods::ModSettings {
        mc_mods::ModSettings::default()
    }

    #[test]
    fn cape_physics_off_is_vanilla_and_controls_reach_the_cloth_snapshot() {
        use mc_render::entity::layers::CapeClothStyle;
        let mut s = settings();
        assert_eq!(cape_motion_from(&s), CapeMotion::Vanilla, "off by default");
        s.set_enabled(CAPE_PHYSICS, true);
        for (choice, expected) in [
            CapeClothStyle::Wavy,
            CapeClothStyle::Flowing,
            CapeClothStyle::Windy,
            CapeClothStyle::Draped,
        ]
        .into_iter()
        .enumerate()
        {
            s.set_choice(CAPE_PHYSICS, "motion", choice);
            let CapeMotion::Cloth(cloth) = cape_motion_from(&s) else {
                panic!("cloth enabled")
            };
            assert_eq!(cloth.style, expected);
            assert_eq!(
                (cloth.strength, cloth.speed, cloth.stiffness),
                (0.6, 1.0, 0.5)
            );
        }
        s.set_value(CAPE_PHYSICS, "strength", mc_mods::OptionValue::Float(90.0));
        s.set_value(CAPE_PHYSICS, "speed", mc_mods::OptionValue::Float(175.0));
        s.set_value(CAPE_PHYSICS, "stiffness", mc_mods::OptionValue::Float(20.0));
        let CapeMotion::Cloth(cloth) = cape_motion_from(&s) else {
            panic!("cloth enabled")
        };
        assert_eq!(
            (cloth.strength, cloth.speed, cloth.stiffness),
            (0.9, 1.75, 0.2)
        );
        s.set_enabled(CAPE_PHYSICS, false);
        assert_eq!(cape_motion_from(&s), CapeMotion::Vanilla);
    }

    #[test]
    fn the_snapshot_only_resyncs_when_the_generation_moves() {
        let mut s = settings();
        let mut x5 = X5State::default();
        assert!(x5.sync(&s), "first sync");
        assert!(!x5.sync(&s), "unchanged settings are cached");
        s.set_enabled(CAPE_PHYSICS, true);
        assert!(x5.sync(&s));
        assert_eq!(x5.cape_motion, cape_motion_from(&s));
        s.set_value(CAPE_PHYSICS, "strength", mc_mods::OptionValue::Float(25.0));
        assert!(x5.sync(&s));
        let CapeMotion::Cloth(cloth) = x5.cape_motion else {
            panic!("cloth enabled")
        };
        assert_eq!(cloth.strength, 0.25);
    }

    #[test]
    fn three_d_skins_options_reach_the_snapshot_and_off_drops_the_meshes() {
        let mut s = settings();
        let mut x5 = X5State::default();
        x5.sync(&s);
        assert!(!x5.skins_on, "off by default");
        s.set_enabled(SKINS_3D, true);
        x5.sync(&s);
        assert!(x5.skins_on && x5.skin_first_person);
        assert_eq!(x5.skin_parts, skin3d::ALL_WEAR);
        assert_eq!(x5.skin_distance_sq, 16.0 * 16.0);
        s.set_bool(SKINS_3D, "sleeves", false);
        s.set_f32(SKINS_3D, "distance", 8.0);
        x5.sync(&s);
        assert_eq!(
            x5.skin_parts,
            skin3d::ALL_WEAR & !(WearPart::LeftSleeve.bit() | WearPart::RightSleeve.bit())
        );
        assert_eq!(x5.skin_distance_sq, 64.0);
        let rgba = vec![255u8; 64 * 64 * 4];
        assert!(x5
            .skin_meshes
            .get_or_build("k", false, || Some((&rgba[..], 64, 64)))
            .is_some());
        s.set_enabled(SKINS_3D, false);
        x5.sync(&s);
        assert!(x5.skin_meshes.is_empty(), "off gives the mesh memory back");
    }

    #[test]
    fn a_helmet_keeps_the_voxel_hat_but_a_skull_or_pumpkin_does_not() {
        let stack = |name: &str| mc_protocol::ItemStack {
            item_id: i16::try_from(mc_data::item_by_name(name).expect(name).0).unwrap(),
            count: 1,
            damage: 0,
            nbt: None,
        };
        assert!(!hat_blocked(None));
        assert!(!hat_blocked(Some(&stack("diamond_helmet"))));
        assert!(!hat_blocked(Some(&stack("leather_helmet"))));
        assert!(hat_blocked(Some(&stack("skull"))));
        assert!(hat_blocked(Some(&stack("pumpkin"))));
    }

    // ---- end to end through `upload_entities` -------------------------------------------

    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        use crate::app::config::{ClientConfig, LoginMode};
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// The local F5 body and a remote player, both on the descriptor's default skin (no
    /// profile textures offline; the test swaps in a full outer layer for the default skins).
    /// 3D Skins on adds voxels to both bodies; the distance option gates the farther one; off
    /// is exactly the flat vertex count again; the first-person sleeve follows its option.
    #[test]
    fn three_d_skins_reach_the_live_f5_and_remote_bodies_and_off_restores_them() {
        use mc_protocol::play::{s2c, S2CPlay};
        use mc_render::{Gpu, GpuOptions, PhaseList, ShellApp as _};

        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let Ok(gpu) = Gpu::headless(64, 64, GpuOptions::default()) else {
            eprintln!("SKIP: no wgpu adapter");
            return;
        };
        let (handle, to_game, mut to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session =
            crate::session::Session::new(handle, crate::session::JoinSettings::default());
        session.state_mut().world = mc_world::World::new();
        client.session = Some(session);
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, 64, 64);
        let send = |packet: S2CPlay| {
            to_game
                .try_send(crate::net::ToGame::Packet(packet))
                .expect("packet channel");
        };
        send(S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 27,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        }));
        client.poll();
        // An empty loaded column (after JoinGame's fresh world), so the spawned remote player is
        // resident: the tracker parks entities in unloaded columns, as `ChunkProviderClient` does.
        client
            .session
            .as_mut()
            .expect("session")
            .state_mut()
            .world
            .install_full_column(mc_math::ChunkPos { x: 0, z: 0 }, mc_world::Chunk::empty());
        send(S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
            x: 7.5,
            y: 64.0,
            z: 2.5,
            yaw: 30.0,
            pitch: 10.0,
            flags: 0,
        }));
        // A remote player three blocks ahead of the local one (yaw 30 faces -x/+z).
        let remote = uuid::Uuid::from_u128(0x5eed_0000_0000_0000_0000_0000_0000_0005);
        send(S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(
            vec![s2c::AddPlayerEntry {
                uuid: remote,
                username: "Voxel".into(),
                properties: vec![],
                game_mode: s2c::GameMode::Survival,
                ping: 0,
                display_name: None,
            }],
        )));
        let fixed = |v: f64| (v * 32.0).floor() as i32;
        send(S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
            entity_id: 500,
            player_uuid: remote,
            x: fixed(6.0),
            y: fixed(64.0),
            z: fixed(5.1),
            yaw: mc_protocol::Angle::from_degrees(210.0),
            pitch: mc_protocol::Angle(0),
            current_item: 0,
            // Watched 10: every skin part shown (what a real client's C15 makes servers send).
            metadata: vec![mc_protocol::MetadataEntry {
                index: 10,
                value: mc_protocol::MetadataValue::Byte(0x7F),
            }],
        }));
        let mut camera = super::super::super::menu_camera();
        client.before_ticks(&mut camera);
        while to_net.try_recv().is_ok() {}
        client.third_person_view = 1;
        client.place_camera(&mut camera, 0.0);
        camera.update();

        let vertices = |client: &mut Client| {
            client.upload_entities(&gpu, &camera, 0.0);
            client
                .render
                .as_ref()
                .expect("installed render stack")
                .entities
                .with(|phase| phase.vertex_count())
        };
        let flat = vertices(&mut client);
        assert_eq!(flat, 2 * 12 * 36, "both flat bodies drawn (12 boxes each)");

        // The jar's Steve/Alex have almost no outer layer: give both a full one.
        let mut full = vec![255u8; 64 * 64 * 4];
        for px in full.chunks_exact_mut(4) {
            px[0] = 120;
        }
        for path in ["textures/entity/steve.png", "textures/entity/alex.png"] {
            client
                .x5
                .default_skins
                .insert(path.to_owned(), Some(full.clone()));
        }
        client.mods.settings.set_enabled(SKINS_3D, true);
        client.mods.settings.set_f32(SKINS_3D, "distance", 5.0);
        client.x5_tick();
        let near_only = vertices(&mut client);
        client.mods.settings.set_f32(SKINS_3D, "distance", 16.0);
        client.x5_tick();
        let both = vertices(&mut client);
        assert!(
            near_only > flat && both > near_only,
            "flat {flat}, local only (5 blocks) {near_only}, both (16 blocks) {both}"
        );
        assert!(client.x5.skin_meshes.builds() >= 1);

        // The first-person sleeve follows the mod and its option.
        let sleeve = |client: &mut Client| {
            client.prepare_first_person_builtin_texture(&gpu);
            client
                .render
                .as_ref()
                .unwrap()
                .first_person
                .with(|phase| phase.skin3d_sleeve.is_some())
        };
        assert!(sleeve(&mut client), "on: the voxel sleeve");
        client
            .mods
            .settings
            .set_bool(SKINS_3D, "first_person", false);
        client.x5_tick();
        assert!(
            !sleeve(&mut client),
            "first-person option off: the flat sleeve"
        );

        client.mods.settings.set_enabled(SKINS_3D, false);
        client.x5_tick();
        assert_eq!(vertices(&mut client), flat, "off: exactly the flat bodies");
        assert!(!sleeve(&mut client));
    }
}
