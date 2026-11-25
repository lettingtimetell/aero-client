//! S21 — integration: the parts that turn fourteen separately-proven crates into one client.
//!
//! S14 owns the library half of this crate ([`crate::net`], [`crate::sync`],
//! [`crate::outbound`], [`crate::session`], [`crate::error`]); everything here is the *host* —
//! the game-state machine, the asset load, the crash handler, the shutdown ordering, and the
//! per-tick drivers the renderer needs.
//!
//! java: this module is `client/Minecraft.java`'s job — `startGame`, `runGameLoop`, `runTick`
//! and `shutdownMinecraftApplet` — split into pieces that can each be tested without a display.
//!
//! | Module | Java equivalent |
//! |---|---|
//! | [`campath`] | no equivalent — the scripted benchmark camera |
//! | [`config`] | `client/main/Main.java` + `GameConfiguration` |
//! | [`state`] | `Minecraft`'s `theWorld`/`currentScreen`/`myNetworkManager` lifecycle |
//! | [`loading`] | `LoadingScreenRenderer.java` |
//! | [`assets`] | `Minecraft.startGame`'s resource/model/atlas block + `refreshResources` |
//! | [`crash`] | `net/minecraft/crash/**` + `Minecraft.displayCrashReport` |
//! | [`shutdown`] | `shutdownMinecraftApplet` + `loadWorld(null)` |
//! | [`clipboard`] | `GuiScreen.setClipboardString`/`getClipboardString` |
//! | [`input`] | `MovementInputFromOptions` + `Minecraft.runTick`'s key handling |
//! | [`interact`] | `World.rayTraceBlocks` + `EntityRenderer.getMouseOver` + `PlayerControllerMP` |
//! | [`driver`] | `runGameLoop`/`runTick`, minus the window |
//! | [`audio`] | `Minecraft.mcSoundHandler` + the `SoundManager.playSound` funnel |
//! | [`particles`] | `RenderGlobal.spawnParticle` + `EffectRenderer`'s input side |

pub mod abilities;
pub mod appearance;
pub mod assets;
pub mod audio;
pub mod campath;
pub mod clipboard;
pub mod config;
pub mod crash;
mod diagnostic;
pub mod driver;
// 1.7.10 sessions: the world events whose 1.7.10 table differs.
mod effects_v5;
pub mod entrender;
// Which chat/book link values may leave the client.
mod external_links;
// ItemRenderer.updateEquippedItem's object-identity stack comparison.
pub mod first_person_equip;
pub(crate) mod frame_scratch;
pub mod input;
pub mod interact;
pub mod loading;
pub mod particles;
pub mod relight;
pub mod shutdown;
pub mod stability;
pub mod state;
pub mod tiles;

pub use entrender::{upload_objects, Hanging, ObjectData, XpValue};
pub use tiles::TileIndex;
// `Client::local_limb_swing` and `Client::view_bob_distance` are the
// walk-animation and view-bobbing inputs `mc_render::entity::anim::AnimParams` and
// `EntityRenderer.setupViewBobbing` want; they are inherent methods on `Client`, so nothing has
// to be re-exported for a caller that already has one. This slot is left as a marker.

pub use config::ClientConfig;
pub use crash::{CrashReport, CrashReportCategory};
pub use driver::Client;
pub use state::{GameState, StateMachine, Transition};
