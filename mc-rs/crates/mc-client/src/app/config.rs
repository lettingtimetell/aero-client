//! Launch configuration: where the game's files are and what it should do on start.
//!
//! java: `client/main/Main.java` builds a `GameConfiguration` out of `joptsimple` options and
//! hands it to `new Minecraft(...)`. The option *names* here are vanilla's, so a launcher that
//! knows how to start 1.8.9 can start this client.
//!
//! Two options vanilla does not have, both forced by this being a from-scratch client rather
//! than a jar the launcher already put on the classpath:
//!
//! * `--versionJar` — vanilla's `DefaultResourcePack` reads models, blockstates and block
//!   textures out of the jar it is *running from*. There is no such jar here, so the path has to
//!   be given (or discovered).
//! * `--offline` — the multiplayer-only rule is multiplayer-only and S03's real login needs an Azure app
//!   registration a developer may not have. `mc_auth`'s `offline` feature is dev-only and this
//!   flag is how it is reached.
//!
//! # Launcher-supplied credentials
//!
//! Added by S21c. The project is staying private and the Azure app registration is deferred, so
//! the client needs a way to be *given* a Minecraft token rather than fetching one: an external
//! launcher (the official launcher, or the portable zip's launcher script) authenticates with
//! **its own** registration and passes the resulting token to the program it launches.
//!
//! The three option names are `Main.java`'s own — `--username`, `--uuid`, `--accessToken` — and
//! that is the whole point: those are precisely what launchers already pass to the Minecraft
//! jar, so this client drops into an existing launcher instance with no custom configuration.
//! See [`LoginMode::Launcher`].

use std::path::{Path, PathBuf};

use mc_version::GameVersion;
use uuid::Uuid;

use crate::app::campath::{CameraScript, DEFAULT_PERIOD_FRAMES, DEFAULT_RADIUS};

/// Vanilla's default server port. java: `Main.java`'s `optionspec1` default.
pub const DEFAULT_PORT: u16 = 25_565;

/// The asset index this client reads. java: the launcher passes `--assetIndex 1.8` for 1.8.9.
pub const DEFAULT_ASSET_INDEX: &str = "1.8";

/// The replay step required by the frozen steady-state diagnostic: one 20 Hz tick per frame.
pub const DIAGNOSTIC_REPLAY_STEP_MICROS: u64 = 50_000;

/// Upper bound for an opt-in diagnostic warmup, preventing accidental unbounded runs.
pub const MAX_DIAGNOSTIC_WARMUP_FRAMES: u64 = 2_000_000;

/// Ignored arguments longer than this are logged by length only: anything
/// that long is more likely a token passed in the wrong place than a word worth reading.
const MAX_PRINTED_ARGUMENT: usize = 64;

/// A Minecraft services access token supplied by a launcher.
///
/// A newtype rather than a bare `String` so that the secret carries its own redaction with it:
/// `Debug` prints `AccessToken(<redacted>)` and there is deliberately **no** `Display`, no
/// `Deref` and no `AsRef<str>`, so the only way to get the bytes out is the loudly-named
/// [`AccessToken::expose`]. That makes every place a token can escape greppable.
///
/// This is the device `mc_protocol::crypt::SharedSecret` already uses for the AES key, and the
/// one `mc_auth::Session` uses for this same token.
#[derive(Clone, PartialEq, Eq)]
pub struct AccessToken(String);

impl AccessToken {
    /// Wrap a token.
    ///
    /// The token is registered with `mc_core::secrets`, so the log writer,
    /// crash reports and panic messages replace it even where it is formatted by mistake.
    #[must_use]
    pub fn new(token: impl Into<String>) -> AccessToken {
        let token = token.into();
        mc_core::secrets::register(&token);
        AccessToken(token)
    }

    /// The token itself.
    ///
    /// Named to be conspicuous at the call site: every use is a point where a secret leaves its
    /// wrapper, and there should never be many of them.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for AccessToken {
    /// Never the token. See the type's documentation.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AccessToken(<redacted>)")
    }
}

/// How the player is authenticated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoginMode {
    /// The Microsoft device-code flow (S03), with the token cache consulted first.
    Microsoft,
    /// A local, unauthenticated name. Only works against an offline-mode server; the access
    /// token is empty and no session-server join is attempted.
    ///
    /// java: there is no vanilla equivalent — `Main.java` always requires `--accessToken`.
    Offline(String),
    /// Credentials handed to this process by an external launcher, as
    /// `--username <name> --uuid <uuid> --accessToken <token>`.
    ///
    /// java: `Main.main` builds exactly this triple into
    /// `new Session(username, playerID, accessToken, userType)` (`Main.java:104`). Two vanilla
    /// details are deliberately **not** reproduced:
    ///
    /// * `Main.java:99` defaults the player id to the *username* when `--uuid` is absent
    ///   (`optionset.has(optionspec10) ? uuid : username`). That produces a `Session` whose
    ///   `getPlayerID()` is a name, which the session server cannot resolve — so here all three
    ///   are required together and a token without a uuid is a parse error rather than a login
    ///   that fails later, in a harder place to diagnose.
    /// * `--userType` is accepted and ignored, as it already was: this client speaks only the
    ///   modern (`msa`) session-server protocol, and `mojang`/`legacy` accounts no longer exist.
    ///
    /// This drives the ordinary online login path — `mc_protocol::connection::login_online`,
    /// which calls `mc_auth::join_server` (hazard H3). The client does no OAuth of its own.
    Launcher {
        /// The display name. `C00PacketLoginStart` carries this, and an online-mode server
        /// checks it against the profile the session server returns for the token.
        username: String,
        /// The account's Minecraft UUID, accepted dashed or undashed and normalised here.
        uuid: Uuid,
        /// The Minecraft services access token. **Secret** — see [`AccessToken`].
        access_token: AccessToken,
    },
}

/// Deterministic simulation controls for the explicitly requested replay diagnostic.
///
/// `None` on [`ClientConfig`] preserves ordinary replay behavior. Explicit selection requests
/// fixed ticks, isolated seeded entity/particle streams, and a declared readiness barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiagnosticSimulation {
    /// Seed for the deterministic simulation streams.
    pub seed: i64,
    /// Number of simulation frames after the final replay packet before the diagnostic window.
    /// This is simulation warmup, not an implicit collector-side discard, and remains diagnostic
    /// evidence that is never scoring-eligible.
    pub warmup_frames: u64,
}

impl LoginMode {
    /// The session this mode can log in with, when it can be built without contacting Microsoft.
    ///
    /// `Some` only for [`LoginMode::Launcher`]: [`LoginMode::Microsoft`] has to run the
    /// device-code flow first, and [`LoginMode::Offline`] is `mc-auth`'s dev-only synthesised
    /// session, which is built elsewhere because it is behind a cargo feature.
    #[must_use]
    pub fn launcher_session(&self) -> Option<mc_auth::Session> {
        match self {
            LoginMode::Launcher {
                username,
                uuid,
                access_token,
            } => Some(mc_auth::Session::new(
                *uuid,
                username.clone(),
                access_token.expose(),
            )),
            LoginMode::Microsoft | LoginMode::Offline(_) => None,
        }
    }
}

/// Everything the client needs to know before it opens a window.
///
/// java: `GameConfiguration` and its five nested records.
#[derive(Debug, Clone, PartialEq)]
pub struct ClientConfig {
    /// java: `--gameDir`, default `.`. `options.txt`, `servers.dat`, `crash-reports/` and
    /// `resourcepacks/` all hang off this.
    pub game_dir: PathBuf,
    /// java: `--assetsDir`, default `<game_dir>/assets`.
    pub assets_dir: PathBuf,
    /// java: `--resourcePackDir`, default `<game_dir>/resourcepacks`.
    pub resource_pack_dir: PathBuf,
    /// Standard Minecraft installation's resourcepacks folder, searched after the client folder.
    /// A same-named pack in `resource_pack_dir` takes precedence. `None` disables this fallback.
    pub resource_pack_fallback_dir: Option<PathBuf>,
    /// java: `--assetIndex`.
    pub asset_index: String,
    /// The version label supplied by the launcher. java: `--version`, required by
    /// `client/main/Main.java`; the default keeps the development-only no-argument launch path
    /// usable while still allowing launcher-specific labels to reach a later UI caller.
    pub version: String,
    /// The vanilla jar that supplies models, blockstates and block textures. No vanilla
    /// equivalent — see this module's header.
    pub version_jar: Option<PathBuf>,
    /// `--gameVersion`: the Minecraft version to play for this run instead of the one picked on
    /// the title screen; nothing is saved, and picking one on the title screen ends it. No
    /// vanilla equivalent: it is for tests and `--server` launches, which skip the title screen.
    /// Unrelated to [`ClientConfig::version`], the launcher's label.
    pub game_version: Option<GameVersion>,
    /// java: `--server` / `--port`. When set, vanilla goes straight to `GuiConnecting` instead
    /// of `GuiMainMenu` (`Minecraft.startGame:571`).
    pub server: Option<(String, u16)>,
    /// Java's `Proxy(Type.SOCKS, ...)`, applied to session-service and server-resource-pack
    /// HTTP(S) requests.  The game socket itself remains direct, matching 1.8.9's
    /// `NetworkManager` path.
    pub proxy: Option<mc_auth::SocksProxy>,
    /// java: `--width`, default 854.
    pub width: u32,
    /// java: `--height`, default 480.
    pub height: u32,
    /// java: `--fullscreen`.
    pub fullscreen: bool,
    /// How to log in.
    pub login: LoginMode,
    /// Stop after this many rendered frames. No vanilla equivalent — this exists so the binary
    /// itself can be launched non-interactively (`--frames 5`) on a machine where a human is not
    /// present to close the window.
    pub max_frames: Option<u64>,

    /// Drive the session from this `.mccap` capture instead of a socket. `--replay`.
    ///
    /// No vanilla equivalent. Every in-world number in its §1 came from a
    /// live server whose world is not reproducible, so an A/B of two rendering changes has no
    /// control. With this set the same recorded bytes drive the same world on every run. See
    /// [`crate::net::replay`].
    pub replay: Option<PathBuf>,

    /// Microseconds of *captured* time the replay advances per game-loop poll, i.e. per frame.
    /// `--replayStep`. Ignored unless [`ClientConfig::replay`] is set.
    pub replay_step_micros: u64,

    /// The scripted camera path a replay drives. `--replayCamera`, `--replayCameraRadius`,
    /// `--replayCameraPeriod`. Ignored unless [`ClientConfig::replay`] is set.
    ///
    /// Defaults to an orbit rather than to `Fixed` on purpose: a baseline measurement found
    /// terrain CPU at 0.072 ms with a stationary camera against 0.142 ms with a moving one, so a
    /// bench that stands still measures the cull cache instead of the renderer.
    pub replay_camera: CameraScript,

    /// Optional fixed-step, frozen steady-state diagnostic controls. Any run using these options
    /// remains diagnostic and unqualified; this field never grants scoring eligibility.
    pub diagnostic_simulation: Option<DiagnosticSimulation>,
}

/// Minecraft's standard resource-pack location, independent of this client's game directory.
fn minecraft_resource_pack_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("APPDATA").map(|p| PathBuf::from(p).join(".minecraft"));
    #[cfg(target_os = "macos")]
    let base = std::env::var_os("HOME")
        .map(|p| PathBuf::from(p).join("Library/Application Support/minecraft"));
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let base = std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".minecraft"));
    base.map(|p| p.join("resourcepacks"))
}

impl Default for ClientConfig {
    fn default() -> Self {
        let game_dir = PathBuf::from(".");
        ClientConfig {
            assets_dir: game_dir.join("assets"),
            resource_pack_dir: game_dir.join("resourcepacks"),
            resource_pack_fallback_dir: minecraft_resource_pack_dir(),
            asset_index: DEFAULT_ASSET_INDEX.to_owned(),
            version: "1.8.9".to_owned(),
            version_jar: None,
            game_version: None,
            server: None,
            proxy: None,
            width: 854,
            height: 480,
            fullscreen: false,
            login: LoginMode::Microsoft,
            max_frames: None,
            replay: None,
            replay_step_micros: crate::net::replay::DEFAULT_STEP_MICROS,
            replay_camera: CameraScript::default(),
            diagnostic_simulation: None,
            game_dir,
        }
    }
}

impl ClientConfig {
    /// Validate an explicitly constructed diagnostic configuration.
    ///
    /// Command-line parsing also requires the diagnostic flags themselves to be present. This
    /// method validates the resulting public configuration instead, so callers constructing a
    /// [`ClientConfig`] directly receive the same replay, camera, cadence, and warmup gates.
    ///
    /// # Errors
    ///
    /// A diagnostic configuration without a replay, fixed camera, required 20 Hz step, or
    /// bounded positive warmup.
    pub fn validate_diagnostic_simulation(&self) -> Result<(), String> {
        let Some(diagnostic) = self.diagnostic_simulation else {
            return Ok(());
        };
        validate_diagnostic_values(
            self.replay.as_deref(),
            self.replay_camera,
            self.replay_step_micros,
            diagnostic,
        )
    }

    /// `<game_dir>/options.txt`. java: `new GameSettings(this, this.mcDataDir)`.
    #[must_use]
    pub fn options_path(&self) -> PathBuf {
        self.game_dir.join("options.txt")
    }

    /// `<game_dir>/servers.dat`. java: `ServerList.loadServerList`.
    #[must_use]
    pub fn servers_path(&self) -> PathBuf {
        self.game_dir.join("servers.dat")
    }

    /// `<game_dir>/crash-reports`. java: `Minecraft.displayCrashReport:762`.
    #[must_use]
    pub fn crash_reports_dir(&self) -> PathBuf {
        self.game_dir.join("crash-reports")
    }

    /// `<game_dir>/cache/baked-assets.bin` — where [`mc_assets::cache`] keeps the last bake.
    ///
    /// No vanilla equivalent; vanilla re-bakes every start. Added by S23. The file is derived
    /// data and safe to delete at any time: it is keyed on the resource-pack stack, carries a
    /// format version and a checksum, and anything it cannot prove is an answer to the exact
    /// question being asked degrades to a full bake.
    #[must_use]
    pub fn bake_cache_path(&self) -> PathBuf {
        self.game_dir.join("cache").join("baked-assets.bin")
    }

    /// Parse vanilla-shaped command-line options.
    ///
    /// java: `Main.main`'s `OptionParser`, which `allowsUnrecognizedOptions()` — so an unknown
    /// `--flag` is ignored rather than fatal, and that is reproduced. A flag whose value is
    /// missing *is* an error here, where `joptsimple` would throw.
    ///
    /// # Errors
    ///
    /// A recognised option with a missing or unparseable value.
    pub fn parse_args<S: AsRef<str>>(args: &[S]) -> Result<ClientConfig, String> {
        let mut config = ClientConfig::default();
        let mut game_dir: Option<PathBuf> = None;
        let mut assets_dir: Option<PathBuf> = None;
        let mut pack_dir: Option<PathBuf> = None;
        let mut server: Option<String> = None;
        let mut port = DEFAULT_PORT;
        let mut index = 0usize;
        // The four credential options are *collected* rather than applied as they are seen, so
        // that the result does not depend on the order a launcher happens to emit them in. The
        // old code set `Offline` the moment it saw `--username`, which would have made
        // `--username X --accessToken Y --uuid Z` mean something different from
        // `--accessToken Y --uuid Z --username X`. Resolution happens once, below.
        let mut offline: Option<String> = None;
        let mut username: Option<String> = None;
        let mut uuid: Option<String> = None;
        let mut access_token: Option<String> = None;
        // Vanilla's proxy is a SOCKS proxy used by authlib's HTTP calls and resource-pack
        // downloads. Keep parsing all four names so a launcher argument vector is consumed
        // deterministically; resolve it only once all options have been seen.
        let mut proxy_host: Option<String> = None;
        let mut proxy_port: Option<u16> = None;
        let mut proxy_user: Option<String> = None;
        let mut proxy_pass: Option<String> = None;
        // The camera options are collected for the same reason the credential ones are: the
        // script is chosen once, at the end, so `--replayCamera orbit --replayCameraRadius 24`
        // and the reverse order mean the same thing.
        let mut camera_mode: Option<String> = None;
        let mut camera_radius: Option<f64> = None;
        let mut camera_period: Option<u64> = None;
        let mut replay_step_explicit = false;
        let mut diagnostic_mode: Option<String> = None;
        let mut diagnostic_seed: Option<i64> = None;
        let mut diagnostic_warmup_frames: Option<u64> = None;

        // A leading `--name value` pair; vanilla accepts one hyphen or two.
        let flag = |arg: &str| -> Option<String> {
            let stripped = arg.strip_prefix("--").or_else(|| arg.strip_prefix('-'))?;
            Some(stripped.to_owned())
        };

        while index < args.len() {
            let arg = args[index].as_ref();
            index += 1;
            let Some(name) = flag(arg) else {
                // java: `optionparser.nonOptions()` — printed and ignored. A
                // long one is described, not printed: a stray token is not yet registered as a
                // secret here (tokens are registered once the options are resolved, below).
                if arg.len() <= MAX_PRINTED_ARGUMENT {
                    tracing::warn!(argument = arg, "completely ignored argument");
                } else {
                    tracing::warn!(characters = arg.len(), "completely ignored long argument");
                }
                continue;
            };
            let mut value = |what: &str| -> Result<String, String> {
                let next = args
                    .get(index)
                    .ok_or_else(|| format!("--{what} needs a value"))?;
                index += 1;
                Ok(next.as_ref().to_owned())
            };

            match name.as_str() {
                "gameDir" => game_dir = Some(PathBuf::from(value("gameDir")?)),
                "assetsDir" => assets_dir = Some(PathBuf::from(value("assetsDir")?)),
                "resourcePackDir" => pack_dir = Some(PathBuf::from(value("resourcePackDir")?)),
                "assetIndex" => config.asset_index = value("assetIndex")?,
                "version" => config.version = value("version")?,
                "versionJar" => config.version_jar = Some(PathBuf::from(value("versionJar")?)),
                "gameVersion" => {
                    let raw = value("gameVersion")?;
                    let version = GameVersion::from_name(&raw).ok_or_else(|| {
                        let names = GameVersion::ALL.map(GameVersion::name).join(" or ");
                        format!("--gameVersion must be {names}, got {raw:?}")
                    })?;
                    config.game_version = Some(version);
                }
                "server" => server = Some(value("server")?),
                "port" => {
                    let raw = value("port")?;
                    port = raw
                        .parse()
                        .map_err(|_| format!("--port must be a number, got {raw:?}"))?;
                }
                "width" => {
                    let raw = value("width")?;
                    config.width = raw
                        .parse()
                        .map_err(|_| format!("--width must be a number, got {raw:?}"))?;
                }
                "height" => {
                    let raw = value("height")?;
                    config.height = raw
                        .parse()
                        .map_err(|_| format!("--height must be a number, got {raw:?}"))?;
                }
                "fullscreen" => config.fullscreen = true,
                "frames" => {
                    let raw = value("frames")?;
                    config.max_frames = Some(
                        raw.parse()
                            .map_err(|_| format!("--frames must be a number, got {raw:?}"))?,
                    );
                }
                // Named in the same `--camelCase` style as vanilla's own
                // options even though vanilla has none of these, so that a launcher's argument
                // vector and a bench's read the same way.
                "replay" => config.replay = Some(PathBuf::from(value("replay")?)),
                "replayStep" => {
                    let raw = value("replayStep")?;
                    config.replay_step_micros = raw
                        .parse()
                        .map_err(|_| format!("--replayStep must be a number, got {raw:?}"))?;
                    replay_step_explicit = true;
                }
                "replayCamera" => camera_mode = Some(value("replayCamera")?),
                "replayCameraRadius" => {
                    let raw = value("replayCameraRadius")?;
                    let radius: f64 = raw.parse().map_err(|_| {
                        format!("--replayCameraRadius must be a number, got {raw:?}")
                    })?;
                    if !(radius.is_finite() && radius > 0.0) {
                        return Err(format!(
                            "--replayCameraRadius must be a positive distance in blocks, got {raw:?}"
                        ));
                    }
                    camera_radius = Some(radius);
                }
                "replayCameraPeriod" => {
                    let raw = value("replayCameraPeriod")?;
                    let period: u64 = raw.parse().map_err(|_| {
                        format!("--replayCameraPeriod must be a number of frames, got {raw:?}")
                    })?;
                    if period == 0 {
                        return Err("--replayCameraPeriod must be at least one frame".to_owned());
                    }
                    camera_period = Some(period);
                }
                "diagnosticSimulation" => diagnostic_mode = Some(value("diagnosticSimulation")?),
                "diagnosticSeed" => {
                    let raw = value("diagnosticSeed")?;
                    diagnostic_seed =
                        Some(raw.parse().map_err(|_| {
                            format!("--diagnosticSeed must be an i64, got {raw:?}")
                        })?);
                }
                "diagnosticWarmupFrames" => {
                    let raw = value("diagnosticWarmupFrames")?;
                    diagnostic_warmup_frames = Some(raw.parse().map_err(|_| {
                        format!("--diagnosticWarmupFrames must be a positive integer, got {raw:?}")
                    })?);
                }
                "offline" => offline = Some(value("offline")?),
                // java: `Main.java`'s `optionspec9`/`optionspec10`/`optionspec11`. These three
                // names are vanilla's exactly, which is what lets a launcher start this client
                // with no custom configuration at all.
                "username" => username = Some(value("username")?),
                "uuid" => uuid = Some(value("uuid")?),
                "accessToken" => access_token = Some(value("accessToken")?),
                // java: these values are part of `GameConfiguration.UserInformation`, but this
                // client has no Twitch stream or pre-join skin-property consumer yet. Consume
                // them explicitly so their values do not become misleading bare arguments.
                "userProperties" => {
                    let _ = value("userProperties")?;
                }
                "profileProperties" => {
                    let _ = value("profileProperties")?;
                }
                "userType" => {
                    let _ = value("userType")?;
                }
                // java: accepted flags in `Main.main`; singleplayer and GL diagnostics are
                // outside this multiplayer client's current scope, but must not be warnings.
                "demo" | "checkGlErrors" => {}
                // java: `proxyHost`/`proxyPort`/`proxyUser`/`proxyPass` are fed to a SOCKS
                // `Proxy` for the session-server HTTP client. Retain the parsed values only
                // until the final capability check below; silently dropping them would make a
                // configured launcher proxy look like a successful direct connection.
                "proxyHost" => proxy_host = Some(value("proxyHost")?),
                "proxyPort" => {
                    let raw = value("proxyPort")?;
                    proxy_port = Some(
                        raw.parse()
                            .map_err(|_| format!("--proxyPort must be a number, got {raw:?}"))?,
                    );
                }
                "proxyUser" => proxy_user = Some(value("proxyUser")?),
                "proxyPass" => proxy_pass = Some(value("proxyPass")?),
                // Only the name: `--accessToken=<token>` (a form this parser
                // does not read) would otherwise put the token in the log.
                other => tracing::warn!(
                    option = other.split_once('=').map_or(other, |(name, _)| name),
                    "unrecognised option, ignored"
                ),
            }
        }

        if proxy_host.is_none()
            && (proxy_port.is_some() || proxy_user.is_some() || proxy_pass.is_some())
        {
            return Err("--proxyPort/--proxyUser/--proxyPass require --proxyHost".to_owned());
        }
        config.proxy = proxy_host
            .map(|host| {
                mc_auth::SocksProxy::new(host, proxy_port.unwrap_or(8080), proxy_user, proxy_pass)
            })
            .transpose()
            .map_err(|reason| format!("invalid SOCKS proxy: {reason}"))?;

        config.login = resolve_login(offline, username, uuid, access_token)?;

        if let Some(dir) = game_dir {
            // java: `--assetsDir`/`--resourcePackDir` default *relative to the game dir*, so they
            // must be recomputed if only `--gameDir` moved.
            config.assets_dir = assets_dir.unwrap_or_else(|| dir.join("assets"));
            config.resource_pack_dir = pack_dir.unwrap_or_else(|| dir.join("resourcepacks"));
            config.game_dir = dir;
        } else {
            if let Some(dir) = assets_dir {
                config.assets_dir = dir;
            }
            if let Some(dir) = pack_dir {
                config.resource_pack_dir = dir;
            }
        }
        config.diagnostic_simulation = resolve_diagnostic_simulation(
            config.replay.as_deref(),
            camera_mode.as_deref(),
            config.replay_step_micros,
            replay_step_explicit,
            diagnostic_mode.as_deref(),
            diagnostic_seed,
            diagnostic_warmup_frames,
        )?;
        config.replay_camera = resolve_camera(camera_mode, camera_radius, camera_period)?;
        config.server = server.map(|host| (host, port));
        // `--replay` on its own is a complete instruction, so it implies the `--server` that
        // makes `finish_loading` open a session (java: `startGame:571`'s "skip the main menu").
        // The host and port are never dialled — `Client::open_connection` builds a capture-backed
        // handle instead — but they are what the connecting screen shows, so they name the file.
        if config.server.is_none() {
            if let Some(path) = &config.replay {
                config.server = Some((format!("replay:{}", path.display()), 0));
            }
        }
        Ok(config)
    }

    /// Parse this process's own arguments, skipping `argv[0]`.
    ///
    /// # Errors
    ///
    /// As [`ClientConfig::parse_args`].
    pub fn from_env() -> Result<ClientConfig, String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        ClientConfig::parse_args(&args)
    }

    /// Fill in `assets_dir` and `version_jar` from a development checkout if they are not
    /// already pointing at something real.
    ///
    /// The repository ships the real 1.8.9 assets ("Reference inputs available on
    /// disk"), and every crate's tests already find them by walking up from `CARGO_MANIFEST_DIR`
    /// or reading `MC_JARS_DIR`. Tests, examples and a development build of the binary use the
    /// same rule so that a developer with the repository checked out can launch with no
    /// arguments at all.
    ///
    /// An optimised `mc-client` (everything that ships) calls
    /// [`ClientConfig::with_assets_from_env`] instead: the checkout path is compiled in here, and
    /// it names a folder of the PC that built the binary.
    #[must_use]
    pub fn with_discovered_assets(self) -> ClientConfig {
        self.with_jars_dir(discover_jars_dir())
    }

    /// Fill in `assets_dir` and `version_jar` from `MC_JARS_DIR` only, if they are not already
    /// pointing at something real.
    ///
    /// What an optimised (shipped) client does when the command line does not name the files:
    /// the launchers pass `--assetsDir` and `--versionJar`, and the benchmark and PGO tools set
    /// `MC_JARS_DIR`. No path is compiled in.
    #[must_use]
    pub fn with_assets_from_env(self) -> ClientConfig {
        self.with_jars_dir(jars_dir_from_env())
    }

    fn with_jars_dir(mut self, jars: Option<PathBuf>) -> ClientConfig {
        let Some(jars) = jars else {
            return self;
        };
        if !self.assets_dir.join("indexes").is_dir() {
            let assets = jars.join("assets");
            if assets.join("indexes").is_dir() {
                self.assets_dir = assets;
            }
        }
        if self.version_jar.as_ref().is_none_or(|jar| !jar.is_file()) {
            for candidate in [
                jars.join("versions").join("1.8.9").join("1.8.9.jar"),
                jars.join("versions").join("1.8.9.jar"),
            ] {
                if candidate.is_file() {
                    self.version_jar = Some(candidate);
                    break;
                }
            }
        }
        self
    }
}

/// Decide the login mode from the four credential options, once all of them have been seen.
///
/// The precedence, in order:
///
/// 1. `--offline <name>` wins over everything. It is a deliberate developer override and it
///    already behaved this way ("… but it must not override an explicit `--offline`").
/// 2. `--accessToken` selects [`LoginMode::Launcher`] and then **requires** `--username` and
///    `--uuid`. Silently degrading an incomplete credential set to offline mode would be the
///    worst possible failure: the user believes they are authenticated and the client sends an
///    unauthenticated name instead.
/// 3. An explicit `--username` or `--uuid` without a token is an error. Vanilla declares
///    `--accessToken` required, and `--offline` is the only route to [`LoginMode::Offline`].
/// 4. Nothing means [`LoginMode::Microsoft`], unchanged.
fn resolve_login(
    offline: Option<String>,
    username: Option<String>,
    uuid: Option<String>,
    access_token: Option<String>,
) -> Result<LoginMode, String> {
    if let Some(name) = offline {
        return Ok(LoginMode::Offline(name));
    }
    let Some(token) = access_token else {
        if username.is_some() {
            return Err(
                "--username needs --accessToken; use --offline for an unauthenticated login"
                    .to_owned(),
            );
        }
        if uuid.is_some() {
            return Err("--uuid needs --accessToken".to_owned());
        }
        return Ok(LoginMode::Microsoft);
    };

    let username = username.ok_or_else(|| {
        "--accessToken needs --username as well; a launcher passes all three of \
         --username, --uuid and --accessToken"
            .to_owned()
    })?;
    let uuid = uuid.ok_or_else(|| {
        // java: `Main.java:99` falls back to the username here. That builds a `Session` whose
        // player id is a name, which the session server cannot resolve, so refusing early is
        // strictly more useful than reproducing it.
        "--accessToken needs --uuid as well; vanilla falls back to using --username as the \
         player id, which the session server cannot resolve"
            .to_owned()
    })?;
    let uuid = parse_uuid(&uuid)?;
    Ok(LoginMode::Launcher {
        username,
        uuid,
        access_token: AccessToken::new(token),
    })
}

/// Resolve the optional deterministic replay diagnostic after all options are collected.
///
/// The mode is intentionally strict: it cannot silently select a replay, camera, or step
/// default because that would make the diagnostic claim a different simulation than requested.
fn resolve_diagnostic_simulation(
    replay: Option<&Path>,
    replay_camera: Option<&str>,
    replay_step_micros: u64,
    replay_step_explicit: bool,
    mode: Option<&str>,
    seed: Option<i64>,
    warmup_frames: Option<u64>,
) -> Result<Option<DiagnosticSimulation>, String> {
    let has_orphan_values = seed.is_some() || warmup_frames.is_some();
    let Some(mode) = mode else {
        if has_orphan_values {
            return Err(
                "--diagnosticSeed/--diagnosticWarmupFrames require --diagnosticSimulation"
                    .to_owned(),
            );
        }
        return Ok(None);
    };
    if mode != "frozen_steady_state" {
        return Err(format!(
            "--diagnosticSimulation must be `frozen_steady_state`, got {mode:?}"
        ));
    }
    if replay.is_none() {
        return Err("--diagnosticSimulation requires --replay".to_owned());
    }
    if replay_camera != Some("fixed") {
        return Err("--diagnosticSimulation requires an explicit --replayCamera fixed".to_owned());
    }
    if !replay_step_explicit || replay_step_micros != DIAGNOSTIC_REPLAY_STEP_MICROS {
        return Err("--diagnosticSimulation requires an explicit --replayStep 50000".to_owned());
    }
    let seed = seed.ok_or_else(|| "--diagnosticSimulation requires --diagnosticSeed".to_owned())?;
    let warmup_frames = warmup_frames
        .ok_or_else(|| "--diagnosticSimulation requires --diagnosticWarmupFrames".to_owned())?;
    if warmup_frames == 0 || warmup_frames > MAX_DIAGNOSTIC_WARMUP_FRAMES {
        return Err(format!(
            "--diagnosticWarmupFrames must be in 1..={MAX_DIAGNOSTIC_WARMUP_FRAMES}"
        ));
    }
    let diagnostic = DiagnosticSimulation {
        seed,
        warmup_frames,
    };
    validate_diagnostic_values(replay, CameraScript::Fixed, replay_step_micros, diagnostic)?;
    Ok(Some(diagnostic))
}

fn validate_diagnostic_values(
    replay: Option<&Path>,
    replay_camera: CameraScript,
    replay_step_micros: u64,
    diagnostic: DiagnosticSimulation,
) -> Result<(), String> {
    if replay.is_none() {
        return Err("diagnostic simulation requires --replay".to_owned());
    }
    if replay_camera != CameraScript::Fixed {
        return Err("diagnostic simulation requires a fixed replay camera".to_owned());
    }
    if replay_step_micros != DIAGNOSTIC_REPLAY_STEP_MICROS {
        return Err(format!(
            "diagnostic simulation requires replay step {DIAGNOSTIC_REPLAY_STEP_MICROS}"
        ));
    }
    if diagnostic.warmup_frames == 0 || diagnostic.warmup_frames > MAX_DIAGNOSTIC_WARMUP_FRAMES {
        return Err(format!(
            "diagnostic warmup must be in 1..={MAX_DIAGNOSTIC_WARMUP_FRAMES}"
        ));
    }
    Ok(())
}

/// Decide the scripted camera path from the three `--replayCamera*` options.
///
/// `--replayCamera fixed` leaves the camera wherever the session put it, which is the control
/// The baseline measurements make it necessary rather than a convenience: the cull cache makes a
/// stationary camera roughly twice as cheap, so "fixed" is a different measurement, not a
/// quieter one. Anything else is an orbit, and the radius and period refine it.
///
/// A radius or period given alongside `fixed` is a contradiction and is refused, rather than
/// silently ignored — an ignored bench option is a benchmark that did not run what was asked.
///
/// # Errors
///
/// An unrecognised mode, or a tuning option that cannot apply to the chosen mode.
fn resolve_camera(
    mode: Option<String>,
    radius: Option<f64>,
    period: Option<u64>,
) -> Result<CameraScript, String> {
    let fixed = match mode.as_deref().map(str::trim) {
        None | Some("orbit") => false,
        Some("fixed" | "none" | "off") => true,
        Some(other) => {
            return Err(format!(
                "--replayCamera must be `orbit` or `fixed`, got {other:?}"
            ))
        }
    };
    if fixed {
        if radius.is_some() || period.is_some() {
            return Err(
                "--replayCameraRadius/--replayCameraPeriod have no meaning with \
                 --replayCamera fixed"
                    .to_owned(),
            );
        }
        return Ok(CameraScript::Fixed);
    }
    Ok(CameraScript::Orbit {
        radius: radius.unwrap_or(DEFAULT_RADIUS),
        period_frames: period.unwrap_or(DEFAULT_PERIOD_FRAMES),
    })
}

/// Parse a UUID in either the dashed or the undashed form.
///
/// Launchers disagree: the official launcher passes the undashed 32-hex form that
/// `Session.getPlayerID()` uses, while launchers in general have passed both over the years.
/// `Uuid::parse_str` accepts both (and the braced and URN forms), so normalising is a parse
/// rather than a string transformation — which also means an almost-right UUID is rejected
/// instead of being silently truncated.
///
/// # Errors
///
/// A message naming the offending value, for [`ClientConfig::parse_args`] to return.
fn parse_uuid(raw: &str) -> Result<Uuid, String> {
    Uuid::parse_str(raw.trim())
        .map_err(|error| format!("--uuid is not a UUID: {raw:?} ({error}); dashed or undashed"))
}

/// Locate the repository's `jars/` directory, honouring `MC_JARS_DIR`.
///
/// The same rule every test in this workspace uses, so a development build and the tests agree
/// about where the vanilla assets are. The fallback compiles in this crate's absolute source
/// path, a folder of the PC that built it: an optimised `mc-client` never calls this (it uses
/// [`ClientConfig::with_assets_from_env`]), so the linker drops the function and the path.
#[must_use]
pub fn discover_jars_dir() -> Option<PathBuf> {
    if std::env::var_os("MC_JARS_DIR").is_some() {
        return jars_dir_from_env();
    }
    // `crates/mc-client` → `crates` → `mc-rs` → the repository root.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    [root.join("MCP-919-main").join("jars"), root.join("jars")]
        .into_iter()
        .find(|candidate| candidate.is_dir())
}

/// `MC_JARS_DIR`, when it names a directory. Nothing compiled in.
fn jars_dir_from_env() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("MC_JARS_DIR")?);
    path.is_dir().then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_vanillas_defaults() {
        let config = ClientConfig::default();
        assert_eq!(config.game_dir, PathBuf::from("."));
        assert_eq!(config.assets_dir, PathBuf::from(".").join("assets"));
        assert_eq!(
            config.resource_pack_dir,
            PathBuf::from(".").join("resourcepacks")
        );
        assert_eq!((config.width, config.height), (854, 480));
        assert!(!config.fullscreen);
        assert_eq!(config.server, None);
        assert_eq!(config.asset_index, "1.8");
        assert_eq!(config.version, "1.8.9");
    }

    /// java: `--assetsDir` and `--resourcePackDir` are *derived from* `--gameDir` when absent,
    /// so moving the game dir must move both.
    #[test]
    fn game_dir_moves_the_derived_directories() {
        let config = ClientConfig::parse_args(&["--gameDir", "/tmp/mc"]).unwrap();
        assert_eq!(config.game_dir, PathBuf::from("/tmp/mc"));
        assert_eq!(config.assets_dir, PathBuf::from("/tmp/mc").join("assets"));
        assert_eq!(
            config.resource_pack_dir,
            PathBuf::from("/tmp/mc").join("resourcepacks")
        );
        assert_eq!(config.options_path(), PathBuf::from("/tmp/mc/options.txt"));
        assert_eq!(config.servers_path(), PathBuf::from("/tmp/mc/servers.dat"));
        assert_eq!(
            config.crash_reports_dir(),
            PathBuf::from("/tmp/mc/crash-reports")
        );
    }

    #[test]
    fn an_explicit_assets_dir_wins_over_the_derived_one() {
        let config =
            ClientConfig::parse_args(&["--gameDir", "/tmp/mc", "--assetsDir", "/srv/assets"])
                .unwrap();
        assert_eq!(config.assets_dir, PathBuf::from("/srv/assets"));
        assert_eq!(
            config.resource_pack_dir,
            PathBuf::from("/tmp/mc").join("resourcepacks")
        );
    }

    /// java: `--port` defaults to 25565 and is only meaningful alongside `--server`.
    #[test]
    fn server_and_port_pair_up() {
        let config = ClientConfig::parse_args(&["--server", "mc.example.com"]).unwrap();
        assert_eq!(config.server, Some(("mc.example.com".to_owned(), 25_565)));

        let config =
            ClientConfig::parse_args(&["--server", "mc.example.com", "--port", "25566"]).unwrap();
        assert_eq!(config.server, Some(("mc.example.com".to_owned(), 25_566)));

        // A port with no server is not a connection request.
        let config = ClientConfig::parse_args(&["--port", "25566"]).unwrap();
        assert_eq!(config.server, None);
    }

    /// java: `optionparser.allowsUnrecognizedOptions()` — a launcher passes plenty of flags this
    /// client has no use for (`--versionType`, …) and they must not be fatal. Options that vanilla
    /// explicitly accepts are consumed by dedicated parser arms below.
    #[test]
    fn unrecognised_options_are_ignored_like_vanilla() {
        let config = ClientConfig::parse_args(&[
            "--userType",
            "msa",
            "--demo",
            "stray-non-option",
            "--width",
            "1280",
        ])
        .unwrap();
        assert_eq!(config.width, 1280);
    }

    #[test]
    fn game_version_picks_the_version_for_one_run_and_is_not_the_launcher_label() {
        let parse = |args: &[&str]| ClientConfig::parse_args(args).map(|c| c.game_version);
        assert_eq!(parse(&[]), Ok(None), "the title screen's pick");
        assert_eq!(
            parse(&["--gameVersion", "1.7.10"]),
            Ok(Some(GameVersion::V1_7_10))
        );
        assert_eq!(
            parse(&["--gameVersion", "1.8.9"]),
            Ok(Some(GameVersion::V1_8_9))
        );
        let error = parse(&["--gameVersion", "1.8"]).expect_err("not a version");
        assert!(error.contains("1.7.10 or 1.8.9"), "{error}");
        assert!(parse(&["--gameVersion"]).is_err(), "a value is required");
        // `--version` is the launcher's label and never selects what is played.
        let config = ClientConfig::parse_args(&["--version", "1.7.10"]).expect("parse");
        assert_eq!(config.version, "1.7.10");
        assert_eq!(config.game_version, None);
    }

    /// The launcher-facing options that have no current Rust consumer still need to consume their
    /// values. Otherwise a valid `--profileProperties {}` pair is misreported as two ignored
    /// arguments and hides the real option boundary.
    #[test]
    fn vanilla_launcher_options_are_consumed_and_version_is_retained() {
        let config = ClientConfig::parse_args(&[
            "--version",
            "1.8.9-custom",
            "--userProperties",
            r#"{"twitch_access_token":"redacted"}"#,
            "--profileProperties",
            r#"{"textures":[]}"#,
            "--userType",
            "msa",
            "--demo",
            "--checkGlErrors",
        ])
        .expect("vanilla launcher options should parse");
        assert_eq!(config.version, "1.8.9-custom");
        assert_eq!(config.login, LoginMode::Microsoft);

        // The two required-value options remain required after becoming explicit arms.
        assert!(ClientConfig::parse_args(&["--version"]).is_err());
        assert!(ClientConfig::parse_args(&["--profileProperties"]).is_err());
    }

    /// Java's proxy options produce one typed proxy shared by auth and resource-pack transport.
    #[test]
    fn proxy_options_parse_into_java_socks_transport() {
        let config = ClientConfig::parse_args(&[
            "--proxyHost",
            "127.0.0.1",
            "--proxyPort",
            "1080",
            "--proxyUser",
            "alice",
            "--proxyPass",
            "secret",
        ])
        .expect("proxy should be wired");
        let proxy = config.proxy.expect("proxy");
        assert_eq!(proxy.host(), "127.0.0.1");
        assert_eq!(proxy.port(), 1080);
        assert_eq!(proxy.username(), Some("alice"));
        assert!(!format!("{proxy:?}").contains("secret"));
        assert_eq!(
            proxy.to_reqwest_url(),
            "socks5h://alice:secret@127.0.0.1:1080"
        );
        assert_eq!(
            ClientConfig::parse_args(&["--proxyHost", "proxy.example"])
                .expect("default Java proxy port")
                .proxy
                .expect("proxy")
                .port(),
            8080
        );
        for args in [
            vec!["--proxyPort", "1080"],
            vec!["--proxyUser", "alice"],
            vec!["--proxyPass", "secret"],
        ] {
            assert!(ClientConfig::parse_args(&args).is_err());
        }
        let err = ClientConfig::parse_args(&["--proxyPort", "not-a-number"])
            .expect_err("proxy port must be validated before capability reporting");
        assert!(err.contains("--proxyPort"), "{err}");
    }

    #[test]
    fn a_flag_with_a_missing_or_bad_value_is_an_error() {
        assert!(ClientConfig::parse_args(&["--gameDir"]).is_err());
        assert!(ClientConfig::parse_args(&["--port", "not-a-number"]).is_err());
        assert!(ClientConfig::parse_args(&["--width", "-"]).is_err());
    }

    #[test]
    fn offline_mode_is_opt_in_and_names_the_player() {
        assert_eq!(ClientConfig::default().login, LoginMode::Microsoft);
        let config = ClientConfig::parse_args(&["--offline", "Dev"]).unwrap();
        assert_eq!(config.login, LoginMode::Offline("Dev".to_owned()));
        // A launcher username without the required token is a parse error, not an accidental
        // offline login. `--offline` remains the explicit developer-only route.
        let err = ClientConfig::parse_args(&["--username", "Steve"])
            .expect_err("vanilla requires --accessToken");
        assert!(err.contains("--accessToken"), "{err}");
        // ... but it must not override an explicit `--offline`.
        let config =
            ClientConfig::parse_args(&["--offline", "Dev", "--username", "Steve"]).unwrap();
        assert_eq!(config.login, LoginMode::Offline("Dev".to_owned()));
    }

    /// The whole point of the option names: this is the exact argument vector a launcher builds
    /// for the Minecraft jar.
    #[test]
    fn a_launcher_supplied_triple_produces_launcher_credentials() {
        let config = ClientConfig::parse_args(&[
            "--username",
            "Notch",
            "--uuid",
            "069a79f4-44e9-4726-a5be-fca90e38aaf5",
            "--accessToken",
            "eyJhbGciOiJIUzI1NiJ9.secret",
            "--userType",
            "msa",
            "--versionType",
            "release",
        ])
        .unwrap();
        assert_eq!(
            config.login,
            LoginMode::Launcher {
                username: "Notch".to_owned(),
                uuid: Uuid::parse_str("069a79f4-44e9-4726-a5be-fca90e38aaf5").unwrap(),
                access_token: AccessToken::new("eyJhbGciOiJIUzI1NiJ9.secret"),
            }
        );
    }

    /// Launchers disagree about dashes, and vanilla's own `Session.getPlayerID()` is undashed.
    /// Both must land on the same UUID.
    #[test]
    fn a_uuid_is_accepted_dashed_or_undashed_and_normalised() {
        let dashed = ClientConfig::parse_args(&[
            "--username",
            "Notch",
            "--uuid",
            "069a79f4-44e9-4726-a5be-fca90e38aaf5",
            "--accessToken",
            "t",
        ])
        .unwrap();
        let undashed = ClientConfig::parse_args(&[
            "--username",
            "Notch",
            "--uuid",
            "069a79f444e94726a5befca90e38aaf5",
            "--accessToken",
            "t",
        ])
        .unwrap();
        assert_eq!(dashed.login, undashed.login);

        // ... and something that is merely UUID-shaped is rejected rather than truncated.
        assert!(ClientConfig::parse_args(&[
            "--username",
            "Notch",
            "--uuid",
            "069a79f4-44e9-4726-a5be",
            "--accessToken",
            "t"
        ])
        .is_err());
    }

    /// Order-independence: a launcher may emit these in any order, and two orders that carry the
    /// same information must not mean two different things. The pre-S21c parser set `Offline`
    /// the instant it saw `--username`, so this is a real regression guard.
    #[test]
    fn the_credential_options_do_not_depend_on_their_order() {
        let a = ClientConfig::parse_args(&[
            "--username",
            "Notch",
            "--uuid",
            "069a79f444e94726a5befca90e38aaf5",
            "--accessToken",
            "t",
        ])
        .unwrap();
        let b = ClientConfig::parse_args(&[
            "--accessToken",
            "t",
            "--uuid",
            "069a79f444e94726a5befca90e38aaf5",
            "--username",
            "Notch",
        ])
        .unwrap();
        assert_eq!(a.login, b.login);
        assert!(matches!(a.login, LoginMode::Launcher { .. }));
    }

    /// An incomplete credential set must be a loud error, never a silent downgrade to an
    /// unauthenticated login the user did not ask for.
    #[test]
    fn an_incomplete_credential_set_is_an_error_not_a_silent_downgrade() {
        let err = ClientConfig::parse_args(&["--username", "Notch", "--accessToken", "t"])
            .expect_err("a token with no uuid must not become offline mode");
        assert!(err.contains("--uuid"), "{err}");

        let err = ClientConfig::parse_args(&[
            "--uuid",
            "069a79f444e94726a5befca90e38aaf5",
            "--accessToken",
            "t",
        ])
        .expect_err("a token with no username must not become offline mode");
        assert!(err.contains("--username"), "{err}");
    }

    /// `--offline` is a developer override and keeps beating everything, including a full
    /// launcher triple.
    #[test]
    fn offline_still_overrides_launcher_credentials() {
        let config = ClientConfig::parse_args(&[
            "--offline",
            "Dev",
            "--username",
            "Notch",
            "--uuid",
            "069a79f444e94726a5befca90e38aaf5",
            "--accessToken",
            "t",
        ])
        .unwrap();
        assert_eq!(config.login, LoginMode::Offline("Dev".to_owned()));
    }

    /// A `--uuid` with no token is also an incomplete launcher credential set; it must not open a
    /// second implicit offline route.
    #[test]
    fn a_uuid_without_a_token_is_an_error() {
        let err = ClientConfig::parse_args(&[
            "--username",
            "Steve",
            "--uuid",
            "069a79f444e94726a5befca90e38aaf5",
        ])
        .expect_err("an explicit username/uuid pair still needs the required token");
        assert!(err.contains("--accessToken"), "{err}");
    }

    /// The token must not reach a log line, a crash report, or a `Debug` of the whole config.
    /// `ClientConfig` derives `Debug`, so this is the assertion that keeps that safe.
    #[test]
    fn the_access_token_is_redacted_everywhere_debug_can_reach() {
        const SECRET: &str = "eyJhbGciOiJIUzI1NiJ9.THIS-MUST-NOT-APPEAR";
        let config = ClientConfig::parse_args(&[
            "--username",
            "Notch",
            "--uuid",
            "069a79f444e94726a5befca90e38aaf5",
            "--accessToken",
            SECRET,
        ])
        .unwrap();

        for rendered in [
            format!("{:?}", config.login),
            format!("{config:?}"),
            format!("{:#?}", config.login),
            format!("{:?}", config.login.launcher_session().expect("a session")),
        ] {
            assert!(
                !rendered.contains(SECRET),
                "a secret escaped through Debug: {rendered}"
            );
            assert!(rendered.contains("redacted"), "{rendered}");
        }
        // And the one sanctioned way out still works.
        assert_eq!(
            config.login.launcher_session().expect("a session").expose(),
            SECRET
        );
    }

    /// The session handed to `login_online` must carry all three parts through unchanged.
    #[test]
    fn launcher_credentials_become_an_mc_auth_session() {
        let config = ClientConfig::parse_args(&[
            "--username",
            "Notch",
            "--uuid",
            "069a79f4-44e9-4726-a5be-fca90e38aaf5",
            "--accessToken",
            "t0ken",
        ])
        .unwrap();
        let session = config.login.launcher_session().expect("a session");
        assert_eq!(session.username, "Notch");
        assert_eq!(session.uuid_undashed(), "069a79f444e94726a5befca90e38aaf5");
        assert_eq!(session.expose(), "t0ken");

        // The other two modes cannot build one without more work happening first.
        assert!(LoginMode::Microsoft.launcher_session().is_none());
        assert!(LoginMode::Offline("Dev".to_owned())
            .launcher_session()
            .is_none());
    }

    /// `--frames` is this client's own option, and it is the only way to launch the binary
    /// without a human to close the window.
    #[test]
    fn frames_bounds_the_run() {
        assert_eq!(ClientConfig::default().max_frames, None);
        let config = ClientConfig::parse_args(&["--frames", "5"]).unwrap();
        assert_eq!(config.max_frames, Some(5));
        assert!(ClientConfig::parse_args(&["--frames", "many"]).is_err());
    }

    /// The flag. `--replay` alone must be a complete instruction: it implies
    /// the `--server` that makes the client leave the main menu, so a bench command line is
    /// `--replay <file> --frames <n>` and nothing else.
    #[test]
    fn replay_is_a_complete_instruction_on_its_own() {
        assert_eq!(ClientConfig::default().replay, None);
        assert_eq!(
            ClientConfig::default().replay_step_micros,
            crate::net::replay::DEFAULT_STEP_MICROS
        );

        let config = ClientConfig::parse_args(&["--replay", "captures/lobby.mccap"]).unwrap();
        assert_eq!(config.replay, Some(PathBuf::from("captures/lobby.mccap")));
        let (host, port) = config.server.expect("--replay implies a session");
        assert!(host.contains("lobby.mccap"), "{host}");
        assert_eq!(port, 0, "nothing is ever dialled");

        // ... but an explicit `--server` still wins, so a capture can be loaded next to a real
        // connection request without the two fighting.
        let config =
            ClientConfig::parse_args(&["--replay", "a.mccap", "--server", "example.com"]).unwrap();
        assert_eq!(config.server, Some(("example.com".to_owned(), 25_565)));
    }

    #[test]
    fn the_replay_step_is_a_number_of_microseconds() {
        let config =
            ClientConfig::parse_args(&["--replay", "a.mccap", "--replayStep", "250"]).unwrap();
        assert_eq!(config.replay_step_micros, 250);
        assert!(ClientConfig::parse_args(&["--replayStep", "fast"]).is_err());
        assert!(ClientConfig::parse_args(&["--replayStep"]).is_err());
    }

    #[test]
    fn ordinary_replay_keeps_diagnostic_settings_absent() {
        let config = ClientConfig::parse_args(&["--replay", "a.mccap"]).unwrap();
        assert_eq!(config.diagnostic_simulation, None);
        assert!(config.validate_diagnostic_simulation().is_ok());
    }

    #[test]
    fn programmatic_diagnostic_configuration_accepts_all_required_values() {
        let config = ClientConfig {
            replay: Some(PathBuf::from("a.mccap")),
            replay_step_micros: DIAGNOSTIC_REPLAY_STEP_MICROS,
            replay_camera: CameraScript::Fixed,
            diagnostic_simulation: Some(DiagnosticSimulation {
                seed: -17,
                warmup_frames: 120,
            }),
            ..ClientConfig::default()
        };
        assert!(config.validate_diagnostic_simulation().is_ok());
    }

    #[test]
    fn programmatic_diagnostic_configuration_rejects_invalid_values() {
        let base = ClientConfig {
            replay: Some(PathBuf::from("a.mccap")),
            replay_step_micros: DIAGNOSTIC_REPLAY_STEP_MICROS,
            replay_camera: CameraScript::Fixed,
            diagnostic_simulation: Some(DiagnosticSimulation {
                seed: 1,
                warmup_frames: 1,
            }),
            ..ClientConfig::default()
        };

        for config in [
            ClientConfig {
                replay: None,
                ..base.clone()
            },
            ClientConfig {
                replay_camera: CameraScript::default(),
                ..base.clone()
            },
            ClientConfig {
                replay_step_micros: DIAGNOSTIC_REPLAY_STEP_MICROS - 1,
                ..base.clone()
            },
            ClientConfig {
                diagnostic_simulation: Some(DiagnosticSimulation {
                    seed: 1,
                    warmup_frames: 0,
                }),
                ..base.clone()
            },
            ClientConfig {
                diagnostic_simulation: Some(DiagnosticSimulation {
                    seed: 1,
                    warmup_frames: MAX_DIAGNOSTIC_WARMUP_FRAMES + 1,
                }),
                ..base.clone()
            },
        ] {
            assert!(config.validate_diagnostic_simulation().is_err());
        }
    }

    #[test]
    fn frozen_steady_state_diagnostic_accepts_permuted_required_options() {
        let config = ClientConfig::parse_args(&[
            "--diagnosticWarmupFrames",
            "120",
            "--replayCamera",
            "fixed",
            "--diagnosticSimulation",
            "frozen_steady_state",
            "--replayStep",
            "50000",
            "--diagnosticSeed",
            "-17",
            "--replay",
            "a.mccap",
        ])
        .unwrap();
        assert_eq!(
            config.diagnostic_simulation,
            Some(DiagnosticSimulation {
                seed: -17,
                warmup_frames: 120
            })
        );
    }

    #[test]
    fn frozen_steady_state_requires_replay_fixed_camera_and_explicit_step() {
        assert!(ClientConfig::parse_args(&[
            "--diagnosticSimulation",
            "frozen_steady_state",
            "--diagnosticSeed",
            "1",
            "--diagnosticWarmupFrames",
            "120",
        ])
        .is_err());
        assert!(ClientConfig::parse_args(&[
            "--diagnosticSimulation",
            "frozen_steady_state",
            "--replay",
            "a.mccap",
            "--replayCamera",
            "orbit",
            "--replayStep",
            "50000",
            "--diagnosticSeed",
            "1",
            "--diagnosticWarmupFrames",
            "120",
        ])
        .is_err());
        assert!(ClientConfig::parse_args(&[
            "--diagnosticSimulation",
            "frozen_steady_state",
            "--replay",
            "a.mccap",
            "--replayCamera",
            "fixed",
            "--diagnosticSeed",
            "1",
            "--diagnosticWarmupFrames",
            "120",
        ])
        .is_err());
    }

    #[test]
    fn frozen_steady_state_rejects_orphans_unknown_mode_and_bad_warmup() {
        assert!(ClientConfig::parse_args(&["--diagnosticSeed", "1"]).is_err());
        assert!(ClientConfig::parse_args(&["--diagnosticWarmupFrames", "1"]).is_err());
        assert!(ClientConfig::parse_args(&[
            "--diagnosticSimulation",
            "ordinary",
            "--diagnosticSeed",
            "1",
            "--diagnosticWarmupFrames",
            "1",
        ])
        .is_err());
        for warmup in ["0", "2000001"] {
            assert!(ClientConfig::parse_args(&[
                "--diagnosticSimulation",
                "frozen_steady_state",
                "--replay",
                "a.mccap",
                "--replayCamera",
                "fixed",
                "--replayStep",
                "50000",
                "--diagnosticSeed",
                "1",
                "--diagnosticWarmupFrames",
                warmup,
            ])
            .is_err());
        }
    }

    /// The camera default is an orbit, not `Fixed`, and that is a measurement decision: §1.6
    /// records 0.072 ms of terrain CPU with a stationary camera against 0.142 ms with a moving
    /// one, so defaulting to "still" would default to measuring the cull cache.
    #[test]
    fn the_scripted_camera_defaults_to_a_moving_orbit() {
        assert_eq!(
            ClientConfig::default().replay_camera,
            CameraScript::Orbit {
                radius: DEFAULT_RADIUS,
                period_frames: DEFAULT_PERIOD_FRAMES,
            }
        );
        let config = ClientConfig::parse_args(&[
            "--replay",
            "a.mccap",
            "--replayCameraRadius",
            "24",
            "--replayCameraPeriod",
            "600",
        ])
        .unwrap();
        assert_eq!(
            config.replay_camera,
            CameraScript::Orbit {
                radius: 24.0,
                period_frames: 600
            }
        );
        // Order must not matter, exactly as it must not for the credential options.
        let reversed = ClientConfig::parse_args(&[
            "--replayCameraPeriod",
            "600",
            "--replayCameraRadius",
            "24",
            "--replay",
            "a.mccap",
        ])
        .unwrap();
        assert_eq!(reversed.replay_camera, config.replay_camera);
    }

    #[test]
    fn the_camera_can_be_pinned_and_a_pinned_camera_refuses_orbit_options() {
        let config =
            ClientConfig::parse_args(&["--replay", "a.mccap", "--replayCamera", "fixed"]).unwrap();
        assert_eq!(config.replay_camera, CameraScript::Fixed);

        // A tuning option that cannot apply is a bench that did not run what was asked for.
        let err =
            ClientConfig::parse_args(&["--replayCamera", "fixed", "--replayCameraRadius", "24"])
                .expect_err("radius has no meaning for a fixed camera");
        assert!(err.contains("--replayCameraRadius"), "{err}");

        assert!(ClientConfig::parse_args(&["--replayCamera", "spiral"]).is_err());
        assert!(ClientConfig::parse_args(&["--replayCameraRadius", "0"]).is_err());
        assert!(ClientConfig::parse_args(&["--replayCameraRadius", "-4"]).is_err());
        assert!(ClientConfig::parse_args(&["--replayCameraPeriod", "0"]).is_err());
    }

    #[test]
    fn single_hyphen_options_are_accepted_too() {
        let config = ClientConfig::parse_args(&["-fullscreen", "-width", "640"]).unwrap();
        assert!(config.fullscreen);
        assert_eq!(config.width, 640);
    }

    /// The development-checkout discovery must find the real vanilla assets this repository
    /// ships, since every other section's tests depend on them being where this looks.
    #[test]
    fn asset_discovery_finds_the_repositorys_real_jars() {
        let Some(jars) = discover_jars_dir() else {
            eprintln!("skip: no jars/ directory in this checkout");
            return;
        };
        assert!(jars
            .join("assets")
            .join("indexes")
            .join("1.8.json")
            .is_file());

        let config = ClientConfig::default().with_discovered_assets();
        assert!(
            config.assets_dir.join("indexes").is_dir(),
            "{:?}",
            config.assets_dir
        );
        let jar = config.version_jar.expect("the 1.8.9 jar must be found");
        assert!(jar.is_file(), "{jar:?}");
        assert_eq!(jar.file_name().unwrap(), "1.8.9.jar");
    }
}
