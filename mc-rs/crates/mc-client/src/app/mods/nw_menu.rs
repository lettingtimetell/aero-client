//! The title screen's three scenes.
//!
//! Requested: "loading screens were supposed to be main menu screens, it just increases loading
//! time like this. please make it into 3 randomly selected main menus". The earlier two
//! animated loading screens, Glass and Shore (`mc_render::loading_scene`), joined the greenhouse
//! (`mc_render::menu_scene`) as title-screen scenes, and the loading screen became fast again
//! (`tw_loading.rs`: the logo and a progress bar, no scene, no minimum time).
//!
//! Like every file under `app/mods/`, this module is declared inside `app::driver`, so it can
//! read `Client`'s private fields.
//!
//! # What happens
//!
//! * The Aero Title Screen mod's **"Menu scene"** option ([`OPT_MENU_SCENE`]): Random (the
//!   default), Greenhouse, Glass or Shore.
//! * **Random is decided once per launch**, at boot ([`Client::nw_menu_boot`], from
//!   `Client::new_deferred`, the game binary's start, before the window exists), with the
//!   launch's seed. The chosen scene starts preparing right there on a background thread: the
//!   greenhouse's bake (`menu_scene::prewarm`, which used to start on the first frame), or the
//!   scene's few vanilla textures from the jar (`JarPrefetch`). Nothing else is prepared, so a
//!   Glass or Shore launch never bakes the greenhouse.
//! * A client built by `Client::new` (tests, tools, benches) has no launch to pick for: Random is
//!   the Greenhouse there, so every harness sees the same title screen as before.
//! * The title screen (`aero_title.rs`) draws the scene in its background slot (under the UI):
//!   the greenhouse exactly as before (its background, its VHS finish and the crisp layer), or
//!   Glass or Shore through `mc_render::loading_scene::LoadingCurtain` at opacity 1 with a
//!   [`BackdropStyle`] (a mirror and a left scrim that keep the logo and buttons readable) and no
//!   finish.
//! * Glass and Shore run on their own clock while the title screen (or the replay browser or the
//!   mod menu over it) is up: real time, in steps of at most `MAX_STEP_S`, so a stall slows the
//!   animation for a frame instead of making it jump; paused while the title screen is away.
//! * Changing the option while the title screen is up switches the scene at once (switching to
//!   the greenhouse keeps the old scene up until its bake, started on a background thread, is
//!   done). Like the greenhouse, a scene is parked while the title screen is away and gives its
//!   screen-sized GPU memory back after a minute away.
//!
//! # Cost
//! Mod off, or any screen but the title screen: nothing of this runs (the title screen's own
//! `bool`/enum/`Option` tests). On the title screen: the scene's GPU time (measured in the
//! lane's report) plus one full-screen composite; on the CPU, a clock step and one small uniform
//! write per frame. The option is re-read only when the settings change.

use std::path::PathBuf;
use std::time::Instant;

use mc_render::loading_scene::{
    self, BackdropStyle, GlassLook, JarPrefetch, JarTextures, LoadingCurtain, SceneFrame,
    SceneInit, SceneKind, MAX_STEP_S,
};
use mc_render::RenderPhase;

use super::super::Client;

/// The Aero Title Screen mod (it owns the option).
pub(crate) const MOD_ID: &str = "aero_title";
/// "Menu scene": Random, Greenhouse, Glass, Shore.
pub(crate) const OPT_MENU_SCENE: &str = "menu_scene";

/// The "Menu scene" option's choices, in its order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MenuChoice {
    /// One of the three, picked each launch.
    Random,
    /// Always the greenhouse.
    Greenhouse,
    /// Always the glass.
    Glass,
    /// Always the shore.
    Shore,
}

impl MenuChoice {
    /// The choice stored as index `index` (out of range: the default, Random).
    pub(crate) fn from_index(index: usize) -> MenuChoice {
        match index {
            1 => MenuChoice::Greenhouse,
            2 => MenuChoice::Glass,
            3 => MenuChoice::Shore,
            _ => MenuChoice::Random,
        }
    }

    /// The scene this choice shows, Random being `pick` (this launch's).
    pub(crate) fn resolve(self, pick: MenuKind) -> MenuKind {
        match self {
            MenuChoice::Random => pick,
            MenuChoice::Greenhouse => MenuKind::Greenhouse,
            MenuChoice::Glass => MenuKind::Glass,
            MenuChoice::Shore => MenuKind::Shore,
        }
    }
}

/// One of the title screen's three scenes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MenuKind {
    /// The block greenhouse.
    Greenhouse,
    /// A Minecraft plant swaying behind wired, frosted glass.
    Glass,
    /// Neon-crested waves on a Minecraft shore at night.
    Shore,
}

impl MenuKind {
    /// Every scene, in the option's order.
    pub(crate) const ALL: [MenuKind; 3] = [MenuKind::Greenhouse, MenuKind::Glass, MenuKind::Shore];

    /// Pick one at random from `seed` (Random's choice for a launch).
    pub(crate) fn random(seed: u32) -> MenuKind {
        // A multiplicative hash, so neighbouring seeds (clock readings) do not alternate.
        let mixed = seed.wrapping_mul(0x9E37_79B9).rotate_right(15);
        MenuKind::ALL[(mixed % MenuKind::ALL.len() as u32) as usize]
    }

    /// The `mc_render::loading_scene` scene behind it (`None` for the greenhouse).
    pub(crate) fn scene(self) -> Option<SceneKind> {
        match self {
            MenuKind::Greenhouse => None,
            MenuKind::Glass => Some(SceneKind::Glass),
            MenuKind::Shore => Some(SceneKind::Shore),
        }
    }

    /// A short stable name for logs and tests.
    pub(crate) fn name(self) -> &'static str {
        match self {
            MenuKind::Greenhouse => "greenhouse",
            MenuKind::Glass => "glass",
            MenuKind::Shore => "shore",
        }
    }

    /// How the scene sits under the title screen's logo and buttons (the greenhouse has its own
    /// scrim in its shader).
    ///
    /// * Glass: mirrored, so the pane's thick condensation (bright, near white, on the scene's
    ///   left) is on the right and the dark leaves are behind the logo and the buttons; and a
    ///   light scrim on top.
    /// * Shore: a night scene, dark already; a lighter scrim keeps the neon crests that run
    ///   under the buttons from competing with their labels.
    pub(crate) fn style(self) -> BackdropStyle {
        match self {
            MenuKind::Greenhouse => BackdropStyle::default(),
            MenuKind::Glass => BackdropStyle {
                mirror: true,
                scrim: GLASS_SCRIM,
            },
            MenuKind::Shore => BackdropStyle {
                mirror: false,
                scrim: SHORE_SCRIM,
            },
        }
    }
}

/// The glass scene's look on the title screen: the earlier tuned look, with the shiny spot
/// (the ceiling light's reflection, and the fan turning in it) moved from the top centre to the
/// right of the logo. In the scene's own coordinates: the title screen mirrors it
/// ([`MenuKind::style`]), so 0.36 lands at 0.64 of the window's width.
pub(crate) fn menu_glass_look() -> GlassLook {
    GlassLook {
        spot: [0.36, 0.15],
        ..GlassLook::default()
    }
}

/// The glass scene's scrim strength (the greenhouse's is 0.12, in linear light; this one is in
/// display space).
const GLASS_SCRIM: f32 = 0.16;
/// The shore's.
const SHORE_SCRIM: f32 = 0.10;
/// The UI's pop-in waits this long on the session's first title screen with Glass or Shore,
/// so it rises while the loading screen's fade uncovers the scene rather than hidden under it.
/// (The greenhouse keeps its own, longer intro delay.)
pub(crate) const SCENE_UI_DELAY_MS: u64 = 250;

/// The seed for this launch: the clock's nanoseconds and the process id, mixed.
pub(crate) fn launch_seed() -> u32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mixed = (nanos as u64) ^ ((nanos >> 64) as u64) ^ u64::from(std::process::id());
    (mixed ^ (mixed >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9) as u32
}

/// The launch's menu decision and the work started for it (`AeroTitleState::menu`).
#[derive(Debug, Default)]
pub(crate) struct MenuState {
    /// What Random shows this launch: decided at boot, or (a client without a launch) the
    /// greenhouse when first asked.
    pick: Option<MenuKind>,
    /// The launch's seed (the scenes vary their layout and waves with it).
    seed: u32,
    /// A Glass or Shore scene's textures being read from the jar, for its first build.
    prefetch: Option<(SceneKind, JarPrefetch)>,
    /// The settings generation [`MenuState::choice`] was read at; `0` = never.
    seen: u64,
    choice: Option<MenuChoice>,
}

impl MenuState {
    /// Decide this launch's Random pick (boot). Later calls keep the first decision.
    pub(crate) fn decide(&mut self, seed: u32) -> MenuKind {
        if self.pick.is_none() {
            self.pick = Some(MenuKind::random(seed));
            self.seed = seed;
        }
        self.pick.unwrap_or(MenuKind::Greenhouse)
    }

    /// This launch's Random pick: the greenhouse when no launch decided one.
    pub(crate) fn pick(&mut self) -> MenuKind {
        *self.pick.get_or_insert(MenuKind::Greenhouse)
    }

    /// The launch's seed.
    pub(crate) fn seed(&self) -> u32 {
        self.seed
    }

    /// The scene the option asks for now (re-read only when the settings changed).
    pub(crate) fn kind(&mut self, settings: &mc_mods::ModSettings) -> MenuKind {
        let generation = settings.generation();
        let choice = match self.choice {
            Some(choice) if self.seen == generation => choice,
            _ => {
                let choice = MenuChoice::from_index(settings.choice(MOD_ID, OPT_MENU_SCENE));
                self.seen = generation;
                self.choice = Some(choice);
                choice
            }
        };
        let pick = self.pick();
        choice.resolve(pick)
    }

    /// The scene whose textures are being read for its first build (tests).
    #[cfg(test)]
    pub(crate) fn prefetching(&self) -> Option<SceneKind> {
        self.prefetch.as_ref().map(|(kind, _)| *kind)
    }

    /// Start reading `scene`'s textures from the jar on a background thread (once).
    pub(crate) fn prefetch(&mut self, scene: SceneKind, jar: PathBuf) {
        if self
            .prefetch
            .as_ref()
            .is_some_and(|(kind, _)| *kind == scene)
        {
            return;
        }
        self.prefetch = Some((scene, JarPrefetch::start(Some(jar), scene.jar_textures())));
    }

    /// `scene`'s textures: the prefetched ones when they are for it (waiting for the thread if
    /// it is still reading), otherwise read now (a few milliseconds), or none without a jar.
    pub(crate) fn textures(
        &mut self,
        scene: SceneKind,
        jar: Option<&std::path::Path>,
    ) -> JarTextures {
        if self
            .prefetch
            .as_ref()
            .is_some_and(|(kind, _)| *kind == scene)
        {
            if let Some((_, prefetch)) = self.prefetch.take() {
                return prefetch.take();
            }
        }
        match jar {
            Some(jar) => JarTextures::load(jar, scene.jar_textures()),
            None => JarTextures::empty(),
        }
    }
}

/// Glass or Shore as the title screen's background: the scene drawn into its own target and laid
/// over the frame by [`LoadingCurtain`]'s composite, at opacity 1, in the title screen's
/// background slot. Dropping it frees all of its GPU memory.
pub(crate) struct SceneBackdrop {
    kind: MenuKind,
    curtain: LoadingCurtain,
    seed: u32,
    /// The scene's clock, seconds: real time while shown, in steps of at most `MAX_STEP_S`.
    clock: f32,
    /// The last frame drawn; `None` after a pause (the next step is 0).
    last: Option<Instant>,
}

impl std::fmt::Debug for SceneBackdrop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneBackdrop")
            .field("kind", &self.kind)
            .field("clock", &self.clock)
            .field("frames", &self.curtain.frames_drawn())
            .finish_non_exhaustive()
    }
}

impl SceneBackdrop {
    /// Build `kind`'s scene (its pipelines and textures; the screen-sized targets come with the
    /// first frame). `kind` must be Glass or Shore.
    pub(crate) fn new(
        gpu: &mc_render::Gpu,
        kind: MenuKind,
        textures: &JarTextures,
        seed: u32,
    ) -> Option<SceneBackdrop> {
        SceneBackdrop::with_look(gpu, kind, textures, seed, menu_glass_look(), kind.style())
    }

    /// [`SceneBackdrop::new`] with another glass look and style (the review renders try
    /// variants with it).
    pub(crate) fn with_look(
        gpu: &mc_render::Gpu,
        kind: MenuKind,
        textures: &JarTextures,
        seed: u32,
        glass: GlassLook,
        style: BackdropStyle,
    ) -> Option<SceneBackdrop> {
        let scene = kind.scene()?;
        let started = Instant::now();
        let init = SceneInit { textures, seed };
        let scene = match scene {
            SceneKind::Glass => loading_scene::glass_scene(gpu, &init, glass, None),
            SceneKind::Shore => scene.create(gpu, &init),
        };
        let mut curtain = LoadingCurtain::with_scene(gpu, scene, seed);
        curtain.set_style(style);
        tracing::info!(
            scene = kind.name(),
            textures = textures.len(),
            ms = started.elapsed().as_secs_f64() * 1000.0,
            "title screen scene built"
        );
        Some(SceneBackdrop {
            kind,
            curtain,
            seed,
            clock: 0.0,
            last: None,
        })
    }

    pub(crate) fn kind(&self) -> MenuKind {
        self.kind
    }

    /// A phase handle for the title screen's background slot.
    pub(crate) fn phase(&self) -> Box<dyn RenderPhase> {
        self.curtain.phase()
    }

    /// Step the clock to `now` and hand the scene this frame's inputs (a `width` x `height`
    /// window, physical pixels).
    pub(crate) fn advance(&mut self, now: Instant, width: u32, height: u32) {
        let step = self
            .last
            .map_or(0.0, |last| {
                now.saturating_duration_since(last).as_secs_f32()
            })
            .min(MAX_STEP_S);
        self.last = Some(now);
        self.clock += step;
        self.curtain.update(
            SceneFrame {
                time_s: self.clock,
                dt_s: step,
                width: width.max(1),
                height: height.max(1),
                progress: 1.0,
                loaded: true,
                opacity: 1.0,
                logo: [0.0; 4],
                seed: self.seed,
            },
            1.0,
        );
    }

    /// The title screen went away: the clock stops (the next frame steps 0, wherever it is).
    pub(crate) fn pause(&mut self) {
        self.last = None;
    }

    /// The scene's clock, seconds (tests).
    #[cfg(test)]
    pub(crate) fn clock(&self) -> f32 {
        self.clock
    }

    /// Set the scene's clock (review renders and tests).
    #[cfg(test)]
    pub(crate) fn set_clock(&mut self, seconds: f32) {
        self.clock = seconds;
        self.last = None;
    }

    /// Free the screen-sized GPU memory, keeping the scene (rebuilt on its next frame).
    pub(crate) fn release_gpu_memory(&mut self) {
        self.curtain.release_gpu_memory();
    }

    /// Whether the screen-sized GPU memory is freed (and nothing drawn since; tests).
    #[cfg(test)]
    pub(crate) fn gpu_memory_released(&self) -> bool {
        self.curtain.gpu_memory_released()
    }
}

impl Client {
    /// Decide this launch's menu scene (Random's pick) and start preparing
    /// it on a background thread, before the window exists. From `Client::new_deferred` only (the
    /// game binary): tests and tools build clients with `Client::new` and get the greenhouse.
    pub(crate) fn nw_menu_boot(&mut self) {
        if self.config.replay.is_some() || self.diagnostic_progress.is_some() {
            return;
        }
        let kind = self.aero.menu.decide(launch_seed());
        tracing::info!(
            random = kind.name(),
            shown = self.aero.menu.kind(&self.mods.settings).name(),
            "title screen scene picked for this launch"
        );
        self.aero_title_prewarm();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_follow_the_option_order() {
        let labels = match mc_mods::mods::aero_title::MENU_SCENE.kind {
            mc_mods::OptionKind::Choice { choices } => choices,
            _ => panic!("Menu scene is a choice"),
        };
        assert_eq!(labels, ["Random", "Greenhouse", "Glass", "Shore"]);
        assert_eq!(mc_mods::mods::aero_title::MENU_SCENE.id, OPT_MENU_SCENE);
        let pick = MenuKind::Shore;
        assert_eq!(MenuChoice::from_index(0).resolve(pick), MenuKind::Shore);
        assert_eq!(
            MenuChoice::from_index(1).resolve(pick),
            MenuKind::Greenhouse
        );
        assert_eq!(MenuChoice::from_index(2).resolve(pick), MenuKind::Glass);
        assert_eq!(MenuChoice::from_index(3).resolve(pick), MenuKind::Shore);
        assert_eq!(MenuChoice::from_index(99), MenuChoice::Random);
        assert_eq!(MenuKind::Greenhouse.scene(), None);
        assert_eq!(MenuKind::Glass.scene(), Some(SceneKind::Glass));
        assert_eq!(MenuKind::Shore.scene(), Some(SceneKind::Shore));
    }

    /// Random shows each of the three about a third of the launches, and a launch's pick is
    /// stable for its seed.
    #[test]
    fn random_picks_each_scene_about_a_third_of_the_time() {
        let mut counts = [0_u32; 3];
        for i in 0..3000_u32 {
            // Seeds the way launches make them: spread out, not consecutive.
            let seed = i.wrapping_mul(0x2545_F491).rotate_left(7) ^ 0x5bd1_e995;
            let kind = MenuKind::random(seed);
            assert_eq!(kind, MenuKind::random(seed));
            counts[MenuKind::ALL.iter().position(|k| *k == kind).unwrap_or(0)] += 1;
        }
        for (kind, count) in MenuKind::ALL.iter().zip(counts) {
            assert!((850..=1150).contains(&count), "{kind:?}: {count} of 3000");
        }
        // Consecutive seeds too.
        let mut seen = std::collections::HashSet::new();
        for seed in 0..16 {
            seen.insert(MenuKind::random(seed));
        }
        assert_eq!(seen.len(), 3);
    }

    /// The launch decides Random once; a client without a launch shows the greenhouse; the
    /// option is followed (re-read when the settings change) and Random keeps the launch's pick.
    #[test]
    fn the_launch_decides_random_once_and_the_option_overrides_it() {
        let mut settings = mc_mods::ModSettings::new();
        let mut none = MenuState::default();
        assert_eq!(none.kind(&settings), MenuKind::Greenhouse, "no launch");

        let seed = (0..64)
            .find(|seed| MenuKind::random(*seed) == MenuKind::Shore)
            .expect("a seed for the shore");
        let mut launch = MenuState::default();
        assert_eq!(launch.decide(seed), MenuKind::Shore);
        let other = (0..64)
            .find(|s| MenuKind::random(*s) != MenuKind::Shore)
            .expect("another seed");
        assert_eq!(launch.decide(other), MenuKind::Shore, "decided once");
        assert_eq!(launch.seed(), seed);
        assert_eq!(launch.kind(&settings), MenuKind::Shore);
        settings.set_choice(MOD_ID, OPT_MENU_SCENE, 2);
        assert_eq!(launch.kind(&settings), MenuKind::Glass);
        settings.set_choice(MOD_ID, OPT_MENU_SCENE, 1);
        assert_eq!(launch.kind(&settings), MenuKind::Greenhouse);
        settings.set_choice(MOD_ID, OPT_MENU_SCENE, 0);
        assert_eq!(
            launch.kind(&settings),
            MenuKind::Shore,
            "back to the launch's pick"
        );
    }

    /// The game binary's start (`Client::new_deferred`): the boot decides the launch's scene and
    /// prepares only it, before the window exists: Glass and Shore have their textures read on a
    /// thread; the greenhouse gets no texture read (its bake is started instead). Random
    /// prepares whatever it picked. Needs the vanilla jar (skips without it; `MC_JARS_DIR`).
    #[test]
    fn the_boot_prepares_only_the_scene_this_launch_shows() {
        use crate::app::config::{ClientConfig, LoginMode};
        let deferred = |options: &str, enabled: bool| {
            let dir = tempfile::tempdir().expect("tempdir");
            let settings = format!(
                r#"{{"format":1,"mods":{{"aero_title":{{"enabled":{enabled},"options":{options}}}}}}}"#
            );
            std::fs::write(dir.path().join("mc-rs-mods.json"), settings).expect("settings");
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
            let client = Client::new_deferred(config).expect("a deferred client");
            Some((client, dir))
        };
        for (options, kind) in [
            (r#"{"menu_scene":"Glass"}"#, MenuKind::Glass),
            (r#"{"menu_scene":"Shore"}"#, MenuKind::Shore),
            (r#"{"menu_scene":"Greenhouse"}"#, MenuKind::Greenhouse),
        ] {
            let Some((mut client, _dir)) = deferred(options, true) else {
                return;
            };
            assert!(
                client.aero.menu.pick.is_some(),
                "the launch decided Random's pick"
            );
            assert_eq!(client.aero.menu.kind(&client.mods.settings), kind);
            assert_eq!(client.aero.menu.prefetching(), kind.scene(), "{options}");
            assert!(
                client.aero.prewarmed,
                "prepared at boot, not on the first frame"
            );
        }
        let Some((mut client, _dir)) = deferred(r#"{"menu_scene":"Random"}"#, true) else {
            return;
        };
        let kind = client.aero.menu.kind(&client.mods.settings);
        assert_eq!(
            client.aero.menu.pick,
            Some(kind),
            "Random shows the launch's pick"
        );
        assert_eq!(client.aero.menu.prefetching(), kind.scene());
        // The mod off (vanilla's title screen): nothing is read for a scene.
        let Some((client, _dir)) = deferred(r#"{"menu_scene":"Glass"}"#, false) else {
            return;
        };
        assert_eq!(
            client.aero.menu.prefetching(),
            None,
            "the mod off prepares nothing"
        );
    }

    #[test]
    fn the_glass_is_mirrored_and_both_scenes_have_a_light_scrim() {
        assert_eq!(MenuKind::Greenhouse.style(), BackdropStyle::default());
        assert!(MenuKind::Glass.style().mirror);
        assert!(!MenuKind::Shore.style().mirror);
        for kind in [MenuKind::Glass, MenuKind::Shore] {
            let scrim = kind.style().scrim;
            assert!(scrim > 0.0 && scrim <= 0.25, "{kind:?}: {scrim}");
        }
    }
}
